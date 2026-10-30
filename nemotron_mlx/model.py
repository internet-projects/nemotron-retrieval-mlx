"""NVIDIA Nemotron 3.5 retrieval models (Embed-1B, Embed-8B, Rerank-8B) on Apple Silicon with MLX: text inputs.

The language tower is a bidirectional Ministral3 (mlx-lm's `ministral3` blocks, run with a padding-only mask: every token
sees every other non-pad token, as the original's `is_causal: false`). Inputs are rendered with the checkpoint's own chat
template and tokenized without BOS/EOS, as in NVIDIA's reference code.

  Embed:  mean of the last hidden states over the input tokens -> L2-normalised vector (2048-d for 1B, 4096-d for 8B);
          queries get the "query: " prefix, documents "passage: ". Matryoshka: keep the first `dim` values and re-normalise.
  Rerank: mean of the last hidden states -> float32 linear head (4096 -> 1) = relevance logit (higher = more relevant);
          the pair is rendered "query: {q}\\n\\npassage: {d}".

Loads the original Hugging Face folders (bf16; the vision tower is ignored) or folders written by convert.py.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

SCORE_FILE = "score.safetensors"


def _args(config):
    from mlx_lm.models.ministral3 import ModelArgs
    tc = dict(config.get("text_config", config))
    tc["model_type"] = "ministral3"
    return ModelArgs.from_dict(tc)


def _text_weights(weights):
    """Original checkpoint keys -> mlx-lm LanguageModel keys; the score head is returned separately."""
    out, score = {}, None
    for k, v in weights.items():
        if k.endswith("score.weight"):
            score = v
            continue
        for p in ("model.language_model.", "language_model.model.", "language_model."):
            if k.startswith(p):
                k2 = k[len(p):]
                if "rotary_emb.inv_freq" not in k2 and not k2.startswith("lm_head"):
                    out[k2] = v
                break
    return out, score


def load_backbone(path, lazy=False):
    """-> (mlx-lm ministral3 LanguageModel, config dict, score head [1, hidden] float32 or None)."""
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
    weights, score = _text_weights(weights) if not config.get("nemotron_mlx") else (weights, None)
    if (q := config.get("quantization")) is not None:
        def pred(p, m):
            if p in q:
                return q[p]
            return f"{p}.scales" in weights
        nn.quantize(model, group_size=q["group_size"], bits=q["bits"], mode=q.get("mode", "affine"), class_predicate=pred)
    model.load_weights(list(weights.items()), strict=True)
    if os.path.exists(os.path.join(path, SCORE_FILE)):
        score = mx.load(os.path.join(path, SCORE_FILE))["weight"]
    if score is not None:
        score = score.astype(mx.float32)
    if not lazy:
        mx.eval(model.parameters())
        if score is not None:
            mx.eval(score)
    return model, config, score


def set_memory_limits():
    """Prompt lengths vary, so MLX's buffer cache would otherwise keep growing; cap it (NEMOTRON_CACHE_LIMIT_GB, default 2)."""
    import mlx.core as mx
    mx.set_cache_limit(int(float(os.environ.get("NEMOTRON_CACHE_LIMIT_GB", "2")) * 1e9))
    if os.environ.get("NEMOTRON_MEMORY_LIMIT_GB"):
        mx.set_memory_limit(int(float(os.environ["NEMOTRON_MEMORY_LIMIT_GB"]) * 1e9))


class _Base:
    max_len = 32768

    def __init__(self, path, lazy=False, token_budget=None):
        import mlx.core as mx
        from transformers import AutoTokenizer
        t0 = time.time()
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            path = snapshot_download(path)
        self.path = path
        self.mx = mx
        set_memory_limits()
        self.model, self.config, self.score_w = load_backbone(path, lazy=lazy)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        tpl = os.path.join(path, "chat_template.jinja")
        if os.path.exists(tpl):
            self.tokenizer.chat_template = open(tpl).read()
        self.token_budget = int(token_budget or os.environ.get("NEMOTRON_TOKEN_BUDGET", "16384"))
        self.args = self.model.args
        self.load_s = time.time() - t0

    # ------------------------------------------------------------------ forward
    def _hidden_mean(self, batch_ids):
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
        mask = None if valid.all() else mx.array(valid)[:, None, None, :]      # keys only: bidirectional, pads hidden
        scale = _get_llama_4_attn_scale(L, 0, self.args.rope_parameters["llama_4_scaling_beta"],
                                        self.args.rope_parameters["original_max_position_embeddings"]).astype(h.dtype)
        for layer in lm.layers:
            h = layer(h, scale, mask, cache=None)
        h = lm.norm(h).astype(mx.float32)
        m = mx.array(valid.astype(np.float32))[..., None]
        pooled = (h * m).sum(axis=1) / m.sum(axis=1)
        return pooled

    def _run(self, id_lists, fn):
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
            r = fn(self._hidden_mean([id_lists[k] for k in idx]))
            mx.eval(r)
            r = np.array(r)
            for k, row in zip(idx, r):
                out[k] = row
            mx.clear_cache()
            i = j
        return out

    def _encode(self, messages_list):
        texts = [self.tokenizer.apply_chat_template(m, tokenize=False) for m in messages_list]
        ids = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        for x in ids:
            if len(x) > self.max_len:
                raise ValueError(f"input of {len(x)} tokens exceeds {self.max_len}; shorten the text")
        return ids


class Embedder(_Base):
    """e = Embedder("netprojects/Nemotron-3.5-Embed-1B-mlx-8bit" or a folder); e.embed_queries([...]); e.embed_documents([...])."""

    def __init__(self, path, **kw):
        super().__init__(path, **kw)
        if self.score_w is not None:
            raise ValueError("this is a reranker checkpoint; use Reranker")
        self.dim = int(self.args.hidden_size)

    def ids(self, texts, kind):
        prefix = {"query": "query:", "document": "passage:"}[kind]
        return self._encode([[{"role": "system", "content": prefix}, {"role": "user", "content": t}] for t in texts])

    def embed(self, texts, kind="document", dim=None):
        """-> np.ndarray [N, dim] float32, L2-normalised. kind: "query" or "document". dim: Matryoshka truncation."""
        mx = self.mx
        rows = self._run(self.ids(texts, kind), lambda p: p)
        e = np.stack(rows).astype(np.float32)
        if dim:
            e = e[:, :int(dim)]
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def embed_queries(self, texts, dim=None):
        return self.embed(texts, "query", dim)

    def embed_documents(self, texts, dim=None):
        return self.embed(texts, "document", dim)


class Reranker(_Base):
    """r = Reranker(path); r.score(query, [doc, ...]) -> relevance logits (higher = more relevant); r.rank(query, docs, top_k)."""

    def __init__(self, path, **kw):
        super().__init__(path, **kw)
        if self.score_w is None:
            raise ValueError("no score head in this checkpoint; use Embedder")

    def ids(self, query, documents):
        return self._encode([[{"role": "query", "content": [{"type": "text", "text": query}]},
                              {"role": "document", "content": [{"type": "text", "text": d}]}] for d in documents])

    def score(self, query, documents):
        mx = self.mx
        w = self.score_w
        rows = self._run(self.ids(query, documents), lambda p: (p @ w.T)[:, 0])
        return np.array([float(r) for r in rows], dtype=np.float32)

    def score_pairs(self, pairs):
        """pairs: [(query, document), ...] -> logits."""
        mx = self.mx
        msgs = [[{"role": "query", "content": [{"type": "text", "text": q}]}, {"role": "document", "content": [{"type": "text", "text": d}]}]
                for q, d in pairs]
        w = self.score_w
        rows = self._run(self._encode(msgs), lambda p: (p @ w.T)[:, 0])
        return np.array([float(r) for r in rows], dtype=np.float32)

    def rank(self, query, documents, top_k=None):
        s = self.score(query, documents)
        order = np.argsort(-s)[: top_k or len(documents)]
        return [{"index": int(i), "score": float(s[i]), "document": documents[i]} for i in order]
