#!/usr/bin/env python3
"""Retrieval quality on BEIR test sets (mteb/<name> on Hugging Face): nDCG@10 and Recall@100 for an embedder, optionally
followed by a reranker on the embedder's top-k.

    python beir_eval.py --dataset scifact --embed <embed folder> [--rerank <rerank folder> --rerank-top 100 --max-queries 300]
        --cache results/emb-cache --out results/beir-scifact-<label>.json

Document embeddings are cached per (dataset, embed folder), so several rerankers can be scored on the same first stage.
Inputs follow the cards: queries "query: ", documents "passage: " (title + text), rerank "query: … passage: …".
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load_beir(name):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    repo = f"mteb/{name}"
    q = pd.read_json(hf_hub_download(repo, "queries.jsonl", repo_type="dataset"), lines=True)
    c = pd.read_json(hf_hub_download(repo, "corpus.jsonl", repo_type="dataset"), lines=True)
    rel = pd.read_csv(hf_hub_download(repo, "qrels/test.tsv", repo_type="dataset"), sep="\t")
    qrels = {}
    for qid, did, s in zip(rel["query-id"].astype(str), rel["corpus-id"].astype(str), rel["score"]):
        if s > 0:
            qrels.setdefault(qid, {})[did] = int(s)
    qtext = dict(zip(q["_id"].astype(str), q["text"]))
    qids = sorted(qid for qid in qrels if qid in qtext)
    dids = c["_id"].astype(str).tolist()
    txt = [x if isinstance(x, str) else "" for x in c["text"]]
    dtext = [(t + "\n" + x).strip() if isinstance(t, str) and t else x for t, x in zip(c["title"], txt)]
    return qids, [qtext[i] for i in qids], dids, dtext, qrels


def ndcg_at(ranked, rel, k=10):
    """nDCG@k with linear gains, as trec_eval / pytrec_eval (and so MTEB) compute it."""
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at(ranked, rel, k=100):
    return len(set(ranked[:k]) & set(rel)) / len(rel) if rel else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--embed", required=True)
    ap.add_argument("--rerank")
    ap.add_argument("--rerank-top", type=int, default=100)
    ap.add_argument("--max-queries", type=int)
    ap.add_argument("--max-docs", type=int, help="score on a slice: the selected queries' relevant documents plus random others up to N (a quick relative check; not comparable to full-corpus scores)")
    ap.add_argument("--dim", type=int)
    ap.add_argument("--max-doc-tokens", type=int, default=8192, help="documents are cut to this many tokens (the cards allow 32K)")
    ap.add_argument("--cache", default=os.path.join(HERE, "results", "emb-cache"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from nemotron_mlx.model import Embedder, Reranker
    qids, qtexts, dids, dtexts, qrels = load_beir(a.dataset)
    if a.max_queries:
        qids, qtexts = qids[:a.max_queries], qtexts[:a.max_queries]
    if a.max_docs and a.max_docs < len(dids):
        import random
        keep = {d for q in qids for d in qrels[q]}
        rest = [i for i, d in enumerate(dids) if d not in keep]
        random.Random(7).shuffle(rest)
        idx = sorted([i for i, d in enumerate(dids) if d in keep] + rest[:max(0, a.max_docs - len(keep))])
        dids, dtexts = [dids[i] for i in idx], [dtexts[i] for i in idx]
    res = {"dataset": a.dataset, "embed": a.embed, "queries": len(qids), "docs": len(dids)}
    os.makedirs(a.cache, exist_ok=True)
    key = hashlib.sha1(f"{a.dataset}|{os.path.abspath(a.embed)}|{a.max_doc_tokens}|{len(dids)}|{a.max_queries}".encode()).hexdigest()[:12]
    cpath = os.path.join(a.cache, f"{a.dataset}-{key}.npz")
    e = Embedder(a.embed)
    cut = []
    for t in dtexts:                                          # trim very long documents to max_doc_tokens
        ids = e.tokenizer(t, add_special_tokens=False)["input_ids"]
        cut.append(t if len(ids) <= a.max_doc_tokens else e.tokenizer.decode(ids[:a.max_doc_tokens]))
    res["docs_trimmed"] = sum(1 for x, y in zip(cut, dtexts) if x is not y)
    t0 = time.time()
    if os.path.exists(cpath):
        D = np.load(cpath)["d"]
        res["doc_embed_s"] = float(np.load(cpath)["s"])
    else:
        D = e.embed_documents(cut)
        res["doc_embed_s"] = time.time() - t0
        np.savez(cpath, d=D, s=res["doc_embed_s"])
    Q = e.embed_queries(qtexts)
    if a.dim:
        D = D[:, :a.dim] / np.linalg.norm(D[:, :a.dim], axis=1, keepdims=True)
        Q = Q[:, :a.dim] / np.linalg.norm(Q[:, :a.dim], axis=1, keepdims=True)
        res["dim"] = a.dim
    S = Q @ D.T
    top = np.argsort(-S, axis=1)[:, :max(100, a.rerank_top)]
    first = [[dids[j] for j in row] for row in top]
    res["ndcg@10"] = round(float(np.mean([ndcg_at(r, qrels[q]) for r, q in zip(first, qids)])), 4)
    res["recall@100"] = round(float(np.mean([recall_at(r, qrels[q]) for r, q in zip(first, qids)])), 4)
    print(json.dumps(res), flush=True)
    if a.rerank:
        del e
        import gc
        gc.collect()
        r = Reranker(a.rerank)
        t1 = time.time()
        reranked = []
        for qt, row in zip(qtexts, top):
            cand = row[:a.rerank_top]
            s = r.score(qt, [cut[j] for j in cand])
            reranked.append([dids[cand[i]] for i in np.argsort(-s)] + [dids[j] for j in row[a.rerank_top:]])
        res.update(rerank=a.rerank, rerank_top=a.rerank_top, rerank_s=round(time.time() - t1, 1),
                   **{"rerank_ndcg@10": round(float(np.mean([ndcg_at(rr, qrels[q]) for rr, q in zip(reranked, qids)])), 4)})
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
