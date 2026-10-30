#!/usr/bin/env python3
"""Image retrieval on ViDoRe v3 (vidore/vidore_v3_<name>_mteb_format, English split): text queries against page images.
nDCG@10 with linear gains, as MTEB reports it, so the numbers line up with NVIDIA's published MTEB results.

    python vidore_eval.py --dataset computer_science --embed <embed folder> --out results/vidore-cs-<label>.json
        [--rerank <rerank folder> --rerank-top 20 --max-queries 100]

Page embeddings are cached per (dataset, embed folder).
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from beir_eval import ndcg_at, recall_at  # noqa: E402


def load_vidore(name, lang="english"):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    repo = f"vidore/vidore_v3_{name}_mteb_format"
    files = [s.rfilename for s in HfApi().dataset_info(repo).siblings]
    rev = HfApi().dataset_info(repo).sha
    q = pq.read_table(hf_hub_download(repo, f"{lang}-queries/test-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
    rel = pq.read_table(hf_hub_download(repo, f"{lang}-qrels/test-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
    qrels = {}
    for r in rel:
        if r["score"] > 0:
            qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    corpus = []
    for f in sorted(x for x in files if x.startswith(f"{lang}-corpus/")):
        for row in pq.read_table(hf_hub_download(repo, f, repo_type="dataset")).to_pylist():
            corpus.append((str(row["id"]), row["image"]["bytes"]))
    queries = [(str(x["id"]), x["text"]) for x in q if str(x["id"]) in qrels]
    return rev, queries, corpus, qrels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="computer_science, hr, energy, ... (vidore_v3_<name>_mteb_format)")
    ap.add_argument("--embed", required=True)
    ap.add_argument("--rerank")
    ap.add_argument("--rerank-top", type=int, default=20)
    ap.add_argument("--max-queries", type=int)
    ap.add_argument("--dims", default="", help="also score Matryoshka-truncated embeddings, e.g. 512,1024 (no extra model work)")
    ap.add_argument("--max-pages", type=int, help="smoke tests only: keep the first N pages (scores are then not comparable)")
    ap.add_argument("--cache", default=os.path.join(HERE, "results", "emb-cache"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from nemotron_mlx.model import Embedder, Reranker
    rev, queries, corpus, qrels = load_vidore(a.dataset)
    if a.max_queries:
        queries = queries[:a.max_queries]
    if a.max_pages:
        corpus = corpus[:a.max_pages]
        queries = [(q, t) for q, t in queries if any(c in qrels[q] for c, _ in corpus)] or queries[:3]
    res = {"dataset": f"vidore_v3_{a.dataset} (english)", "dataset_revision": rev, "embed": a.embed, "queries": len(queries), "pages": len(corpus)}
    os.makedirs(a.cache, exist_ok=True)
    key = hashlib.sha1(f"vidore|{a.dataset}|{os.path.abspath(a.embed)}|{os.environ.get('NEMOTRON_IMAGE_LONGEST_EDGE', '')}|{len(corpus)}".encode()).hexdigest()[:12]
    cpath = os.path.join(a.cache, f"vidore-{a.dataset}-{key}.npz")
    e = Embedder(a.embed)
    if os.path.exists(cpath):
        D, res["page_embed_s"] = np.load(cpath)["d"], float(np.load(cpath)["s"])
    else:
        t0 = time.time()
        D = []
        for i in range(0, len(corpus), 32):                      # 32 pages at a time keeps memory flat
            D.append(e.embed_documents([{"image": b} for _, b in corpus[i:i + 32]]))
        D = np.concatenate(D)
        res["page_embed_s"] = round(time.time() - t0, 1)
        np.savez(cpath, d=D, s=res["page_embed_s"])
    res["pages_per_s"] = round(len(corpus) / res["page_embed_s"], 2)
    Q = e.embed_queries([t for _, t in queries])
    S = Q @ D.T
    top = np.argsort(-S, axis=1)[:, :max(100, a.rerank_top)]
    ids = [c for c, _ in corpus]
    first = [[ids[j] for j in row] for row in top]
    res["ndcg@10"] = round(float(np.mean([ndcg_at(r, qrels[q]) for r, (q, _) in zip(first, queries)])), 5)
    res["recall@100"] = round(float(np.mean([recall_at(r, qrels[q]) for r, (q, _) in zip(first, queries)])), 4)
    for k in [int(x) for x in a.dims.split(",") if x]:
        Dk = D[:, :k] / np.linalg.norm(D[:, :k], axis=1, keepdims=True)
        Qk = Q[:, :k] / np.linalg.norm(Q[:, :k], axis=1, keepdims=True)
        tk = np.argsort(-(Qk @ Dk.T), axis=1)[:, :10]
        res[f"ndcg@10_dim{k}"] = round(float(np.mean([ndcg_at([ids[j] for j in r], qrels[q]) for r, (q, _) in zip(tk, queries)])), 5)
    print(json.dumps(res), flush=True)
    if a.rerank:
        del e
        import gc
        gc.collect()
        r = Reranker(a.rerank)
        t1 = time.time()
        rr = []
        for (_, qt), row in zip(queries, top):
            cand = row[:a.rerank_top]
            s = r.score(qt, [{"image": corpus[j][1]} for j in cand])
            rr.append([ids[cand[i]] for i in np.argsort(-s)] + [ids[j] for j in row[a.rerank_top:]])
        res.update(rerank=a.rerank, rerank_top=a.rerank_top, rerank_s=round(time.time() - t1, 1),
                   **{"rerank_ndcg@10": round(float(np.mean([ndcg_at(x, qrels[q]) for x, (q, _) in zip(rr, queries)])), 5)})
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
