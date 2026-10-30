#!/usr/bin/env python3
"""Publish the Nemotron 3.5 MLX builds (Paris approved 30 Oct 2026 11:47) and add them to every NetProjects summary view.
Dry run by default; --go to act.     python scripts/publish_netprojects.py --models <models dir> [--go]

1. Hugging Face: create netprojects/<build> and upload each model folder (card, LICENSE, NOTICE, weights).
2. Add the six builds to the "NetProjects MLX builds" collection.
3. Org card (Space netprojects/README): Nemotron 3.5 row in the model table and the code repo line.
4. GitHub profile READMEs (internet-projects/.github and pmichalas/pmichalas): Nemotron row / line (via gh, live text edited).
Each edit is idempotent: it is skipped when the text already mentions nemotron-retrieval-mlx.
"""
import argparse
import base64
import json
import os
import subprocess
import sys

BUILDS = ["Nemotron-3.5-Embed-1B-mlx-bf16", "Nemotron-3.5-Embed-1B-mlx-8bit", "Nemotron-3.5-Embed-8B-mlx-8bit",
          "Nemotron-3.5-Embed-8B-mlx-4bit", "Nemotron-3.5-Rerank-8B-mlx-8bit", "Nemotron-3.5-Rerank-8B-mlx-4bit"]
NOTES = {"Nemotron-3.5-Embed-1B-mlx-bf16": "NVIDIA Nemotron 3.5 Embed 1B: text + page-image embeddings, bf16",
         "Nemotron-3.5-Embed-1B-mlx-8bit": "NVIDIA Nemotron 3.5 Embed 1B: text + page-image embeddings, 8-bit",
         "Nemotron-3.5-Embed-8B-mlx-8bit": "NVIDIA Nemotron 3.5 Embed 8B: text + page-image embeddings, 8-bit",
         "Nemotron-3.5-Embed-8B-mlx-4bit": "NVIDIA Nemotron 3.5 Embed 8B: text + page-image embeddings, 4-bit",
         "Nemotron-3.5-Rerank-8B-mlx-8bit": "NVIDIA Nemotron 3.5 Rerank 8B: text + page-image reranker, 8-bit",
         "Nemotron-3.5-Rerank-8B-mlx-4bit": "NVIDIA Nemotron 3.5 Rerank 8B: text + page-image reranker, 4-bit"}
COLLECTION_TITLE = "NetProjects MLX builds"
HF = "https://huggingface.co/netprojects/"
GH = "https://github.com/internet-projects/nemotron-retrieval-mlx"
MARK = "nemotron-retrieval-mlx"
LINKS_1B = f"[bf16]({HF}Nemotron-3.5-Embed-1B-mlx-bf16) · [8-bit]({HF}Nemotron-3.5-Embed-1B-mlx-8bit)"
LINKS_8B = f"[8-bit]({HF}Nemotron-3.5-Embed-8B-mlx-8bit) · [4-bit]({HF}Nemotron-3.5-Embed-8B-mlx-4bit)"
LINKS_RR = f"[8-bit]({HF}Nemotron-3.5-Rerank-8B-mlx-8bit) · [4-bit]({HF}Nemotron-3.5-Rerank-8B-mlx-4bit)"

ORG_ROW = (f"| **NVIDIA Nemotron 3.5 Embed / Rerank** | Embed-1B {LINKS_1B} — Embed-8B {LINKS_8B} — Rerank-8B {LINKS_RR} "
           "| Retrieval: embeddings and reranking for text and page images (scores match NVIDIA's published ViDoRe v3 results) |")
ORG_CODE = f"- [{MARK}]({GH}): Nemotron 3.5 Embed / Rerank with page images, `/v1/embeddings` and `/v1/rerank` server"
GHORG_ROW = (f"| [{MARK}]({GH}) | NVIDIA Nemotron 3.5 Embed 1B / 8B and Rerank 8B: retrieval for text and page images "
             f"| Embed-1B {LINKS_1B} · Embed-8B {LINKS_8B} · Rerank-8B {LINKS_RR} |")
PM_LINE = f"- [{MARK}]({GH}): NVIDIA Nemotron 3.5 Embed / Rerank retrieval models, text and page images"


def insert_after(text, anchor_sub, new_line):
    """Insert new_line after the last line containing anchor_sub."""
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if anchor_sub in l]
    if not idx:
        raise SystemExit(f"anchor not found: {anchor_sub!r}")
    lines.insert(idx[-1] + 1, new_line)
    return "\n".join(lines)


def gh_get(repo, path):
    out = subprocess.run(["gh", "api", f"repos/{repo}/contents/{path}"], capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    return base64.b64decode(d["content"]).decode(), d["sha"]


def gh_put(repo, path, text, sha, msg):
    body = json.dumps({"message": msg, "content": base64.b64encode(text.encode()).decode(), "sha": sha})
    subprocess.run(["gh", "api", "-X", "PUT", f"repos/{repo}/contents/{path}", "--input", "-"], input=body, text=True,
                   capture_output=True, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()
    if "netprojects" not in [o.get("name") for o in who.get("orgs", [])]:
        sys.exit("HF login is not in the netprojects org")

    def act(desc, fn):
        print(("DO   " if a.go else "PLAN ") + desc, flush=True)
        if a.go:
            return fn()

    # 1. model repos
    for b in BUILDS:
        d = os.path.join(a.models, b)
        need = ["README.md", "LICENSE", "NOTICE", "config.json", "model.safetensors.index.json"]
        miss = [f for f in need if not os.path.exists(os.path.join(d, f))]
        if miss:
            sys.exit(f"{b}: missing {miss}")
        if a.skip_upload:
            continue
        rid = f"netprojects/{b}"
        act(f"create {rid}", lambda r=rid: api.create_repo(r, repo_type="model", exist_ok=True))
        src = "nvidia/" + b.split("-mlx-")[0] + "-BF16"
        info = act(f"upload {d} -> {rid}", lambda r=rid, p=d, s=src: api.upload_folder(
            repo_id=r, folder_path=p, repo_type="model", commit_message=f"MLX conversion of {s} (text + page images)"))
        if info is not None:
            print("     ", getattr(info, "commit_url", info))

    # 2. collection
    coll = next((c for c in api.list_collections(owner="netprojects") if c.title == COLLECTION_TITLE), None)
    if not coll:
        sys.exit("collection not found")
    have = {i.item_id for i in api.get_collection(coll.slug).items}
    for b in BUILDS:
        rid = f"netprojects/{b}"
        if rid in have:
            print("skip collection item", rid)
        else:
            act(f"add {rid} to the collection", lambda r=rid, n=NOTES[b]: api.add_collection_item(
                coll.slug, item_id=r, item_type="model", note=n, exists_ok=True))

    # 3. org card
    from huggingface_hub import hf_hub_download
    live = open(hf_hub_download("netprojects/README", "README.md", repo_type="space", force_download=True)).read()
    if MARK in live:
        print("skip org card (already lists", MARK + ")")
    else:
        org = insert_after(live, "nemotron-dory1.0-4b-mlx-bf16", ORG_ROW)
        org = insert_after(org, "](https://github.com/internet-projects/dory-mlx)", ORG_CODE)
        act("org card (Space netprojects/README)", lambda: api.upload_file(
            path_or_fileobj=org.encode(), path_in_repo="README.md", repo_id="netprojects/README", repo_type="space",
            commit_message="Org card: NVIDIA Nemotron 3.5 Embed / Rerank builds and code repo"))
        if not a.go:
            print(org)

    # 4. GitHub profile READMEs
    for repo, path, anchor, line, msg in [
            ("internet-projects/.github", "profile/README.md", "](https://github.com/internet-projects/dory-mlx)", GHORG_ROW,
             "Profile: NVIDIA Nemotron 3.5 Embed / Rerank MLX port"),
            ("pmichalas/pmichalas", "README.md", "](https://github.com/internet-projects/dory-mlx)", PM_LINE,
             "Profile: Nemotron 3.5 retrieval MLX port")]:
        text, sha = gh_get(repo, path)
        if MARK in text:
            print("skip", repo, "(already lists", MARK + ")")
            continue
        new = insert_after(text, anchor, line)
        act(f"{repo}/{path}", lambda r=repo, p=path, t=new, s=sha, m=msg: gh_put(r, p, t, s, m))
        if not a.go:
            print(new)
    print("collection:", coll.url)


if __name__ == "__main__":
    main()
