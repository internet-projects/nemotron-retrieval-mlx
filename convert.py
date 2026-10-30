#!/usr/bin/env python3
"""Convert an original Nemotron 3.5 retrieval checkpoint (Embed-1B / Embed-8B / Rerank-8B, bf16) to an MLX build.

    python convert.py --src <original folder> --out <folder> --bits 16|8|6|4

--bits 16 writes the language tower in bf16 (no quantization); 8 / 4 quantize every Linear and the token embedding of the
language tower with MLX affine quantization, group size 64. The vision encoder and projector are kept in bf16 (--text-only
leaves them out).
The rerank score head is kept in float32 in its own file (score.safetensors). The output folder carries the tokenizer, the
chat template, the OpenMDW-1.1 LICENSE and the NOTICE files.
"""
import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
COPY = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"]
SHARD_BYTES = 5 * 1024 ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, required=True, choices=[16, 8, 6, 4])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--revision", default=None, help="upstream revision recorded in config.json")
    ap.add_argument("--text-only", action="store_true", help="leave out the vision encoder and projector")
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "config.json")):
        sys.exit(f"{a.out} already has a build")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from nemotron_mlx.model import load_backbone

    t0 = time.time()
    model, config, score, vision = load_backbone(a.src, lazy=True)
    q = None
    if a.bits < 16:
        nn.quantize(model, group_size=a.group_size, bits=a.bits, mode="affine",
                    class_predicate=lambda p, m: hasattr(m, "to_quantized") and m.weight.shape[-1] % a.group_size == 0)
        q = {"group_size": a.group_size, "bits": a.bits, "mode": "affine"}
    else:
        from mlx.utils import tree_map
        model.update(tree_map(lambda x: x.astype(mx.bfloat16), model.parameters()))
    weights = dict(tree_flatten(model.parameters()))
    if vision and not a.text_only:                   # vision encoder + projector: original precision (bf16), MLX layout
        from nemotron_mlx.vision import vision_weights
        weights.update({k: v.astype(mx.bfloat16) for k, v in vision_weights(vision, config.get("vision_config", {})).items()})
    os.makedirs(a.out, exist_ok=True)
    shards, cur, size = [], {}, 0
    for k in sorted(weights):
        nb = weights[k].nbytes
        if cur and size + nb > SHARD_BYTES:
            shards.append(cur)
            cur, size = {}, 0
        cur[k] = weights[k]
        size += nb
    shards.append(cur)
    wmap = {}
    for i, sh in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors" if len(shards) > 1 else "model.safetensors"
        mx.save_safetensors(os.path.join(a.out, name), sh, metadata={"format": "mlx"})
        wmap.update({k: name for k in sh})
    json.dump({"metadata": {"total_size": sum(v.nbytes for v in weights.values())}, "weight_map": wmap},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
    if score is not None:
        mx.save_safetensors(os.path.join(a.out, "score.safetensors"), {"weight": score.astype(mx.float32)}, metadata={"format": "mlx"})
    cfg = {k: v for k, v in config.items() if not (a.text_only and k == "vision_config")}
    cfg["nemotron_mlx"] = {"kind": "rerank" if score is not None else "embed", "source": os.path.basename(os.path.normpath(a.src)),
                           "revision": a.revision, "bits": a.bits, "text_only": bool(a.text_only or not vision),
                           "kept_full_precision": ["norms"] + (["vision encoder and projector (bf16)"] if vision and not a.text_only else [])
                           + (["score head (float32)"] if score is not None else [])}
    if q:
        cfg["quantization"] = q
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    for f in COPY:
        if os.path.exists(os.path.join(a.src, f)):
            shutil.copy2(os.path.join(a.src, f), os.path.join(a.out, f))
    # the Embed repos ship no LICENSE / NOTICE file: add NVIDIA's OpenMDW-1.1 LICENSE (as shipped with Rerank-8B) and a
    # NOTICE naming the Apache-2.0 Ministral base, as the original cards state (OpenMDW-1.1: keep the agreement and notices)
    if not os.path.exists(os.path.join(a.out, "LICENSE")):
        shutil.copy2(os.path.join(HERE, "licenses", "LICENSE"), os.path.join(a.out, "LICENSE"))
    if not os.path.exists(os.path.join(a.out, "NOTICE")):
        base = "Ministral-3-3B-Instruct-2512-BF16" if model.args.hidden_size <= 2048 else "Ministral-3-8B-Instruct-2512-BF16"
        open(os.path.join(a.out, "NOTICE"), "w").write(
            f"mistralai/{base} is licensed under the Apache-2.0 License.\n\n"
            "This folder holds an MLX conversion of NVIDIA's model, made by NetProjects; "
            "the weights were converted to MLX format" + (" and quantized" if a.bits < 16 else "") + ".\n")
    gb = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out) if f.endswith(".safetensors")) / 1e9
    print(f"wrote {a.out}: {gb:.2f} GB in {time.time() - t0:.0f} s ({'bf16' if a.bits == 16 else f'{a.bits}-bit'})")


if __name__ == "__main__":
    main()
