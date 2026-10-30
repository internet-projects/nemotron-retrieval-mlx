"""NVIDIA Nemotron 3.5 retrieval models (Embed-1B, Embed-8B, Rerank-8B) on Apple Silicon with MLX: text and images.

The language tower is a bidirectional Ministral3 (mlx-lm's `ministral3` blocks, run with a padding-only mask: every token
sees every other non-pad token, as the original's `is_causal: false`). Inputs are rendered with the checkpoint's own chat
template and tokenized without BOS/EOS, as in NVIDIA's reference code. Documents may carry an image (page, chart, photo):
it goes through the checkpoint's Pixtral encoder and projector (vision.py) and fills the [IMG] tokens of the input.

  Embed:  mean of the last hidden states over the input tokens -> L2-normalised vector (2048-d for 1B, 4096-d for 8B);
          queries get the "query: " prefix, documents "passage: ". Matryoshka: keep the first `dim` values and re-normalise.
  Rerank: mean of the last hidden states -> float32 linear head (4096 -> 1) = relevance logit (higher = more relevant);
          the pair is rendered "[IMG] query: {q}\\n\\npassage: {d}" ("[IMG] " only when the document has an image).

A document is a string, or a dict {"text": ..., "image": PIL image | path | bytes | base64 | data URL | http(s) URL}.
Loads the original Hugging Face folders (bf16) or folders written by convert.py. NEMOTRON_TEXT_ONLY=1 skips the vision part.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

SCORE_FILE = "score.safetensors"


def _args(config):
    from mlx_lm.models.ministral3 import ModelArgs
    tc = dict(config.get("text_config", config))
    tc["model_type"] = "ministral3"
    return ModelArgs.from_dict(tc)


def _split_weights(weights, converted):
    """-> (language-tower weights in mlx-lm keys, score head or None, vision/projector weights in original-ish keys)."""
    text, score, vision = {}, None, {}
    for k, v in weights.items():
        if k.endswith("score.weight"):
            score = v
        elif "vision_tower." in k or "multi_modal_projector." in k:
            vision[k] = v
        elif converted:
            text[k] = v
        else:
            for p in ("model.language_model.", "language_model.model.", "language_model."):
                if k.startswith(p):
                    k2 = k[len(p):]
                    if "rotary_emb.inv_freq" not in k2 and not k2.startswith("lm_head"):
                        text[k2] = v
                    break
    return text, score, vision


def load_backbone(path, lazy=False):
    """-> (mlx-lm ministral3 LanguageModel, config dict, score head [1, hidden] float32 or None, vision weights dict)."""
    import glob
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.ministral3 import LanguageModel
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    model = LanguageModel(_args(config))
    files = sorted(glob.glob(os.path.join(path, "model*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no model*.safetensors in {path}")
    weights = {}
    for f in files:
        weights.update(mx.load(f))
    text, score, vision = _split_weights(weights, bool(config.get("nemotron_mlx")))
    if (q := config.get("quantization")) is not None:
        def pred(p, m):
            if p in q:
                return q[p]
            return f"{p}.scales" in text
        nn.quantize(model, group_size=q["group_size"], bits=q["bits"], mode=q.get("mode", "affine"), class_predicate=pred)
    model.load_weights(list(text.items()), strict=True)
    if os.path.exists(os.path.join(path, SCORE_FILE)):
        score = mx.load(os.path.join(path, SCORE_FILE))["weight"]
    if score is not None:
        score = score.astype(mx.float32)
    if not lazy:
        mx.eval(model.parameters())
        if score is not None:
            mx.eval(score)
    return model, config, score, vision


def set_memory_limits():
    """Prompt lengths vary, so MLX's buffer cache would otherwise keep growing; cap it (NEMOTRON_CACHE_LIMIT_GB, default 2)."""
    import mlx.core as mx
    mx.set_cache_limit(int(float(os.environ.get("NEMOTRON_CACHE_LIMIT_GB", "2")) * 1e9))
    if os.environ.get("NEMOTRON_MEMORY_LIMIT_GB"):
        mx.set_memory_limit(int(float(os.environ["NEMOTRON_MEMORY_LIMIT_GB"]) * 1e9))


def _doc(d):
    """str | {"text", "image"} -> (text or None, image or None)."""
    if isinstance(d, str):
        return d, None
    if isinstance(d, dict):
        if d.get("text") is None and d.get("image") is None:
            raise ValueError("a document dict needs 'text' and/or 'image'")
        return d.get("text"), d.get("image")
    raise ValueError("a document must be a string or a dict with 'text' and/or 'image'")


class _Base:
    max_len = 32768

    def __init__(self, path, lazy=False, token_budget=None, vision=None):
        import mlx.core as mx
        from transformers import AutoTokenizer
        t0 = time.time()
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            path = snapshot_download(path)
        self.path = path
        self.mx = mx
        set_memory_limits()
        self.model, self.config, self.score_w, vw = load_backbone(path, lazy=lazy)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        tpl = os.path.join(path, "chat_template.jinja")
        if os.path.exists(tpl):
            with open(tpl) as f:
                self.tokenizer.chat_template = f.read()
        self.img_id = self.tokenizer.convert_tokens_to_ids("[IMG]")
        use_vision = vision if vision is not None else os.environ.get("NEMOTRON_TEXT_ONLY", "0") != "1"
        self.vision = None
        if use_vision and vw:
            from .vision import Vision, vision_weights
            self.vision = Vision(self.config, vision_weights(vw, self.config.get("vision_config", {})), path)
        self.token_budget = int(token_budget or os.environ.get("NEMOTRON_TOKEN_BUDGET", "16384"))
        self.args = self.model.args
        self.load_s = time.time() - t0

    # ------------------------------------------------------------------ inputs
    def _encode(self, messages_list, images_list=None):
        """Render with the chat template, expand each [IMG] placeholder to the image's tokens, tokenize (no BOS/EOS).
        -> (token id lists, per-input list of (pixel_values, size) or None)."""
        texts, prepared = [], []
        for i, m in enumerate(messages_list):
            t = self.tokenizer.apply_chat_template(m, tokenize=False)
            imgs = (images_list or [None] * len(messages_list))[i] or []
            if imgs:
                if self.vision is None:
                    raise ValueError("this model was loaded without its vision part; images are not available")
                parts = t.split("[IMG]")
                if len(parts) - 1 != len(imgs):
                    raise ValueError("image placeholder count does not match the images")
                out, pre = parts[0], []
                for img, rest in zip(imgs, parts[1:]):
                    pv, size, s = self.vision.prepare(img)
                    pre.append((pv, size))
                    out += s + rest
                t = out
                prepared.append(pre)
            else:
                prepared.append(None)
            texts.append(t)
        ids = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        for x in ids:
            if len(x) > self.max_len:
                raise ValueError(f"input of {len(x)} tokens exceeds {self.max_len}; shorten the text or lower the image size")
        return ids, prepared

    # ------------------------------------------------------------------ forward
    def _hidden_mean(self, batch_ids, batch_imgs):
        """Right-padded batch -> mean-pooled last hidden states [B, H] float32 (pads excluded as keys and from the mean)."""
        mx = self.mx
        from mlx_lm.models.ministral3 import _get_llama_4_attn_scale
        B, L = len(batch_ids), max(len(x) for x in batch_ids)
        ids = np.zeros((B, L), dtype=np.int32)
        valid = np.zeros((B, L), dtype=bool)
        for i, x in enumerate(batch_ids):
            ids[i, :len(x)] = x
            valid[i, :len(x)] = True
        lm = self.model
        h = lm.embed_tokens(mx.array(ids))
        if any(batch_imgs):
            rows = []
            for i, imgs in enumerate(batch_imgs):
                row = h[i]
                if imgs:
                    f = mx.concatenate([self.vision.features(pv, size) for pv, size in imgs]).astype(h.dtype)
                    is_img = ids[i] == self.img_id
                    if int(is_img.sum()) != f.shape[0]:
                        raise ValueError(f"{int(is_img.sum())} image tokens but {f.shape[0]} image features")
                    gather = mx.array(np.maximum(np.cumsum(is_img) - 1, 0))
                    row = mx.where(mx.array(is_img)[:, None], f[gather], row)
                rows.append(row)
            h = mx.stack(rows)
        mask = None if valid.all() else mx.array(valid)[:, None, None, :]      # keys only: bidirectional, pads hidden
        scale = _get_llama_4_attn_scale(L, 0, self.args.rope_parameters["llama_4_scaling_beta"],
                                        self.args.rope_parameters["original_max_position_embeddings"]).astype(h.dtype)
        for layer in lm.layers:
            h = layer(h, scale, mask, cache=None)
        h = lm.norm(h).astype(mx.float32)
        m = mx.array(valid.astype(np.float32))[..., None]
        return (h * m).sum(axis=1) / m.sum(axis=1)

    def _run(self, id_lists, imgs, fn):
        """Length-sorted micro-batches under the padded-token budget; returns fn(pooled) rows in input order."""
        mx = self.mx
        order = sorted(range(len(id_lists)), key=lambda i: len(id_lists[i]))
        out = [None] * len(id_lists)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and (j - i + 1) * len(id_lists[order[j]]) <= self.token_budget:
                j += 1
            idx = order[i:j]
            r = fn(self._hidden_mean([id_lists[k] for k in idx], [imgs[k] for k in idx]))
            mx.eval(r)
            r = np.array(r)
            for k, row in zip(idx, r):
                out[k] = row
            mx.clear_cache()
            i = j
        return out


class Embedder(_Base):
    """e = Embedder("netprojects/Nemotron-3.5-Embed-1B-mlx-8bit" or a folder); e.embed_queries([...]); e.embed_documents([...]).
    Documents may be strings or {"text": ..., "image": ...}; queries are text."""

    def __init__(self, path, **kw):
        super().__init__(path, **kw)
        if self.score_w is not None:
            raise ValueError("this is a reranker checkpoint; use Reranker")
        self.dim = int(self.args.hidden_size)

    def inputs(self, items, kind):
        prefix = {"query": "query:", "document": "passage:"}[kind]
        msgs, imgs = [], []
        for it in items:
            text, image = _doc(it)
            if kind == "query" and image is not None:
                raise ValueError("queries are text only")
            content = ([{"type": "image"}] if image is not None else []) + ([{"type": "text", "text": text}] if text is not None else [])
            msgs.append([{"role": "system", "content": prefix}, {"role": "user", "content": content}])
            imgs.append([image] if image is not None else None)
        return self._encode(msgs, imgs)

    def ids(self, items, kind):
        return self.inputs(items, kind)[0]

    def embed(self, items, kind="document", dim=None):
        """-> np.ndarray [N, dim] float32, L2-normalised. kind: "query" or "document". dim: Matryoshka truncation."""
        if dim is not None and not (isinstance(dim, int) and 1 <= dim <= self.dim):
            raise ValueError(f"dimensions must be an integer from 1 to {self.dim}")
        if not items:
            self.last_tokens = 0
            return np.zeros((0, dim or self.dim), np.float32)
        ids, imgs = self.inputs(items, kind)
        self.last_tokens = sum(len(x) for x in ids)
        e = np.stack(self._run(ids, imgs, lambda p: p)).astype(np.float32)
        if dim:
            e = e[:, :dim]
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def embed_queries(self, texts, dim=None):
        return self.embed(texts, "query", dim)

    def embed_documents(self, docs, dim=None):
        return self.embed(docs, "document", dim)


class Reranker(_Base):
    """r = Reranker(path); r.score(query, [doc, ...]) -> relevance logits (higher = more relevant); r.rank(query, docs, top_k).
    Documents may be strings or {"text": ..., "image": ...} (one image per document)."""

    def __init__(self, path, **kw):
        super().__init__(path, **kw)
        if self.score_w is None:
            raise ValueError("no score head in this checkpoint; use Embedder")

    def inputs(self, pairs):
        msgs, imgs = [], []
        for q, d in pairs:
            text, image = _doc(d)
            content = ([{"type": "text", "text": text}] if text is not None else []) + ([{"type": "image"}] if image is not None else [])
            msgs.append([{"role": "query", "content": [{"type": "text", "text": q}]}, {"role": "document", "content": content}])
            imgs.append([image] if image is not None else None)
        return self._encode(msgs, imgs)

    def ids(self, query, documents):
        return self.inputs([(query, d) for d in documents])[0]

    def score_pairs(self, pairs):
        """pairs: [(query, document), ...] -> float32 logits."""
        ids, imgs = self.inputs(pairs)
        self.last_tokens = sum(len(x) for x in ids)
        w = self.score_w
        rows = self._run(ids, imgs, lambda p: (p @ w.T)[:, 0])
        return np.array([float(r) for r in rows], dtype=np.float32)

    def score(self, query, documents):
        return self.score_pairs([(query, d) for d in documents])

    def rank(self, query, documents, top_k=None):
        if top_k is not None and not (isinstance(top_k, int) and top_k >= 1):
            raise ValueError("top_n must be a positive integer")
        if not documents:
            return []
        s = self.score(query, documents)
        order = np.argsort(-s)[: top_k or len(documents)]
        return [{"index": int(i), "score": float(s[i]), "document": documents[i]} for i in order]
