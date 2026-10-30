#!/usr/bin/env python3
"""Throughput on Apple Silicon: Embed documents/s, or Rerank pairs/s, at fixed token lengths and batch sizes; MLX peak memory.

    python speed.py --model <folder> --lengths 512,4096 --batches 1,16,64 --json speed.json
"""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FILLER = "Quarterly results show revenue growth across regions, with energy, water and transport projects leading. "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="512,4096")
    ap.add_argument("--batches", default="1,16,64")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-batch-tokens", type=int, default=131072, help="skip a batch size whose tokens exceed this")
    ap.add_argument("--json")
    a = ap.parse_args()
    import mlx.core as mx
    from nemotron_mlx.model import Embedder, Reranker, load_backbone
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    rerank = os.path.exists(os.path.join(a.model, "score.safetensors")) or "ForSequenceClassification" in str(cfg.get("architectures"))
    m = (Reranker if rerank else Embedder)(a.model)
    tok = m.tokenizer
    filler = tok(FILLER * 3000, add_special_tokens=False)["input_ids"]
    out = {"model": a.model, "kind": "rerank" if rerank else "embed", "load_s": round(m.load_s, 1), "runs": []}
    for L in [int(x) for x in a.lengths.split(",")]:
        for B in [int(x) for x in a.batches.split(",")]:
            if B * L > a.max_batch_tokens:
                out["runs"].append({"tokens": L, "batch": B, "skipped": f"{B * L} tokens > --max-batch-tokens"})
                continue
            text = tok.decode(filler[:L - 8])
            m.token_budget = B * (L + 16)
            if rerank:
                docs = [text] * B
                fn = lambda: m.score("What drove revenue growth?", docs)
            else:
                docs = [text] * B
                fn = lambda: m.embed_documents(docs)
            n_tok = len(m.ids("What drove revenue growth?", docs[:1])[0]) if rerank else len(m.ids(docs[:1], "document")[0])
            fn()
            mx.clear_cache()
            mx.reset_peak_memory()
            ts = []
            for _ in range(a.repeats):
                t0 = time.perf_counter()
                fn()
                ts.append(time.perf_counter() - t0)
            p50 = statistics.median(ts)
            r = {"tokens": n_tok, "batch": B, "p50_s": round(p50, 3), ("pairs_per_s" if rerank else "docs_per_s"): round(B / p50, 2),
                 "tokens_per_s": round(B * n_tok / p50), "peak_gb": round(mx.get_peak_memory() / 1e9, 2)}
            out["runs"].append(r)
            print(json.dumps(r), flush=True)
            mx.clear_cache()
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
