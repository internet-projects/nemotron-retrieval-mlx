#!/usr/bin/env python3
"""Parity of the MLX port against NVIDIA's transformers reference code (the model cards' "Transformers" examples).

    python parity.py cases --out cases.json [--beir scifact --n-queries 40 --n-docs 160]
    python parity.py ref  --model <original folder> --kind embed|rerank --cases cases.json --out ref.npz [--device mps --dtype float32]
    python parity.py mlx  --model <original or converted folder> --kind embed|rerank --cases cases.json --out mlx.npz
    python parity.py compare ref.npz mlx.npz [--json out.json]

Run the reference and MLX sides in separate processes. The reference embeds / scores one input at a time, exactly as the
cards' code (processor.apply_chat_template, no padding); the MLX side runs in padded batches, which also checks the padding.
Compared: token ids, embedding cosine (min / mean), top-10 retrieval overlap, rerank Spearman / max abs diff / top-1 agreement.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORDS = ("protein cell gene expression cancer therapy patient trial dose response immune virus infection model data "
         "network market price risk energy climate water soil policy law court evidence football league match goal").split()


# ------------------------------------------------------------------------------------------------------------------ cases
def synthetic_cases(nq=40, nd=160, seed=3):
    rnd = random.Random(seed)
    docs = []
    for i in range(nd):
        n = rnd.choice([20, 80, 300, 900] + ([3200] if i % 20 == 0 else []))   # every 20th doc is long (4K+ tokens)
        docs.append(" ".join(rnd.choice(WORDS) for _ in range(n)) + (" [note] <tag>" if i % 7 == 0 else ""))
    queries = ["What is known about " + " ".join(rnd.choice(WORDS) for _ in range(rnd.choice([3, 8, 15]))) + "?" for _ in range(nq)]
    return {"queries": queries, "docs": docs}


def beir_cases(name, nq, nd, seed=3):
    """A slice of a BEIR dataset (mteb/<name>): nq test queries with their relevant docs, filled up to nd docs."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    rnd = random.Random(seed)
    repo = f"mteb/{name}"
    q = pd.read_json(hf_hub_download(repo, "queries.jsonl", repo_type="dataset"), lines=True)
    c = pd.read_json(hf_hub_download(repo, "corpus.jsonl", repo_type="dataset"), lines=True)
    rel = pd.read_csv(hf_hub_download(repo, "qrels/test.tsv", repo_type="dataset"), sep="\t")
    qids = sorted(set(rel["query-id"].astype(str)))
    rnd.shuffle(qids)
    qids = qids[:nq]
    qtext = dict(zip(q["_id"].astype(str), q["text"]))
    want = set(rel[rel["query-id"].astype(str).isin(qids)]["corpus-id"].astype(str))
    c["_id"] = c["_id"].astype(str)
    pool = c[c["_id"].isin(want)]
    rest = c[~c["_id"].isin(want)].sample(n=max(0, nd - len(pool)), random_state=seed)
    docs = pd.concat([pool, rest])
    return {"queries": [qtext[i] for i in qids], "docs": [(t + "\n" + x).strip() for t, x in zip(docs["title"].fillna(""), docs["text"])]}


# --------------------------------------------------------------------------------------------------------------- reference
def ref_run(a, cases):
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoProcessor
    dtype = getattr(torch, a.dtype)
    proc = AutoProcessor.from_pretrained(a.model)
    if a.kind == "embed":
        model = AutoModel.from_pretrained(a.model, dtype=dtype).to(a.device).eval()

        def one(prefix, text):
            inputs = proc.apply_chat_template([{"role": "system", "content": prefix}, {"role": "user", "content": [{"type": "text", "text": text}]}],
                                              tokenize=True, return_dict=True, return_tensors="pt").to(a.device)
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                e = (hidden * mask).sum(1) / mask.sum(1)
            return torch.nn.functional.normalize(e.float(), dim=-1)[0].cpu().numpy(), inputs["input_ids"][0].tolist()

        Q = [one("query:", t) for t in cases["queries"]]
        D = [one("passage:", t) for t in cases["docs"]]
        return {"q": np.stack([x[0] for x in Q]), "d": np.stack([x[0] for x in D]),
                "ids": np.array([json.dumps(x[1]) for x in Q + D])}
    model = AutoModelForSequenceClassification.from_pretrained(a.model, trust_remote_code=True, dtype=dtype).to(a.device).eval()
    pairs, scores, ids = rerank_pairs(cases), [], []
    for qi, di in pairs:
        inputs = proc.apply_chat_template([{"role": "query", "content": [{"type": "text", "text": cases["queries"][qi]}]},
                                           {"role": "document", "content": [{"type": "text", "text": cases["docs"][di]}]}],
                                          tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt").to(a.device)
        with torch.inference_mode():
            scores.append(float(model(**inputs).logits.squeeze().float()))
        ids.append(json.dumps(inputs["input_ids"][0].tolist()))
    return {"s": np.array(scores, dtype=np.float32), "pairs": np.array(pairs), "ids": np.array(ids)}


def rerank_pairs(cases, per_query=10, seed=5):
    """Each query against per_query documents (its first ones by a fixed shuffle): 40 x 10 = 400 pairs by default."""
    rnd = random.Random(seed)
    out = []
    for qi in range(len(cases["queries"])):
        ds = list(range(len(cases["docs"])))
        rnd.shuffle(ds)
        out += [(qi, d) for d in ds[:per_query]]
    return out


def mlx_run(a, cases):
    sys.path.insert(0, HERE)
    from nemotron_mlx.model import Embedder, Reranker
    t0 = time.time()
    if a.kind == "embed":
        e = Embedder(a.model)
        ids = e.ids(cases["queries"], "query") + e.ids(cases["docs"], "document")
        q = e.embed_queries(cases["queries"])
        t1 = time.time()
        d = e.embed_documents(cases["docs"])
        return {"q": q, "d": d, "ids": np.array([json.dumps(x) for x in ids]), "seconds": np.array([e.load_s, time.time() - t0])}
    r = Reranker(a.model)
    pairs = rerank_pairs(cases)
    s = r.score_pairs([(cases["queries"][qi], cases["docs"][di]) for qi, di in pairs])
    ids = r._encode([[{"role": "query", "content": [{"type": "text", "text": cases["queries"][qi]}]},
                      {"role": "document", "content": [{"type": "text", "text": cases["docs"][di]}]}] for qi, di in pairs])
    return {"s": s, "pairs": np.array(pairs), "ids": np.array([json.dumps(x) for x in ids]), "seconds": np.array([r.load_s, time.time() - t0])}


# ---------------------------------------------------------------------------------------------------------------- compare
def compare(ref, test):
    from scipy.stats import spearmanr
    out = {"token_ids_identical": int((ref["ids"] == test["ids"]).sum()), "inputs": int(len(ref["ids"]))}
    if "q" in ref:
        cq = (ref["q"] * test["q"]).sum(1)
        cd = (ref["d"] * test["d"]).sum(1)
        cos = np.concatenate([cq, cd])
        out.update(cosine_min=round(float(cos.min()), 6), cosine_mean=round(float(cos.mean()), 6), cosine_p01=round(float(np.quantile(cos, 0.01)), 6))
        sr, st = ref["q"] @ ref["d"].T, test["q"] @ test["d"].T
        k = min(10, sr.shape[1])
        tr, tt = np.argsort(-sr, 1)[:, :k], np.argsort(-st, 1)[:, :k]
        out["top10_overlap_mean"] = round(float(np.mean([len(set(a) & set(b)) / k for a, b in zip(tr, tt)])), 4)
        out["top1_same"] = f"{int((tr[:, 0] == tt[:, 0]).sum())}/{len(tr)}"
        out["score_abs_diff_max"] = round(float(np.abs(sr - st).max()), 5)
    else:
        s1, s2 = ref["s"], test["s"]
        out.update(spearman=round(float(spearmanr(s1, s2).correlation), 6), score_abs_diff_max=round(float(np.abs(s1 - s2).max()), 5),
                   score_abs_diff_mean=round(float(np.abs(s1 - s2).mean()), 5))
        pairs = ref["pairs"]
        same = n = 0
        for qi in sorted(set(pairs[:, 0])):
            m = pairs[:, 0] == qi
            n += 1
            same += int(np.argmax(s1[m]) == np.argmax(s2[m]))
        out["top1_same"] = f"{same}/{n}"
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cases")
    c.add_argument("--out", required=True)
    c.add_argument("--beir")
    c.add_argument("--n-queries", type=int, default=40)
    c.add_argument("--n-docs", type=int, default=160)
    for n in ("ref", "mlx"):
        p = sub.add_parser(n)
        p.add_argument("--model", required=True)
        p.add_argument("--kind", choices=["embed", "rerank"], required=True)
        p.add_argument("--cases", required=True)
        p.add_argument("--out", required=True)
        if n == "ref":
            p.add_argument("--device", default="cpu")
            p.add_argument("--dtype", default="float32")
    p = sub.add_parser("compare")
    p.add_argument("ref")
    p.add_argument("test")
    p.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "cases":
        cases = synthetic_cases(a.n_queries, a.n_docs) if not a.beir else beir_cases(a.beir, a.n_queries, a.n_docs)
        if a.beir:                                     # add a few long (4K+ token) documents to every set
            syn = synthetic_cases(0, 160)
            cases["docs"] += [d for d in syn["docs"] if len(d.split()) > 3000]
        json.dump(cases, open(a.out, "w"))
        print(f"{len(cases['queries'])} queries, {len(cases['docs'])} documents -> {a.out}")
    elif a.cmd in ("ref", "mlx"):
        cases = json.load(open(a.cases))
        res = (ref_run if a.cmd == "ref" else mlx_run)(a, cases)
        np.savez(a.out, **res)
        print(f"saved {a.out}")
    else:
        r = compare(dict(np.load(a.ref)), dict(np.load(a.test)))
        print(json.dumps(r, indent=1))
        if a.json:
            json.dump(r, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
