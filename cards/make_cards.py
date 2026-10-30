#!/usr/bin/env python3
"""Write the Hugging Face model cards (README.md per build) from the measured results.
    python cards/make_cards.py --results <results dir> --out <dir>      (results/ = stage 1, results/v2 = images)"""
import argparse, glob, json, os

REV = {"Embed-1B": "e24ca1e346bcc53572bd7bdc02b3e8ebf66c197c", "Embed-8B": "781e779ee4ca9c2d2a4f38321993556baeab49a1",
       "Rerank-8B": "f88f8d9104726747791283ab156d2fcdb29e90a7"}
NV = {"Embed-1B": 0.76422, "Embed-8B": 0.77356}          # NVIDIA, MTEB ViDoRe v3 Computer Science (English), nDCG@10
BASE = {"Embed-1B": "Ministral-3-3B-Instruct-2512-BF16", "Embed-8B": "Ministral-3-8B-Instruct-2512-BF16", "Rerank-8B": "Ministral-3-8B-Instruct-2512-BF16"}


def j(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def label(bits):
    return {"bf16": "bf16", "8bit": "8-bit", "4bit": "4-bit"}[bits]


def card(model, bits, R, size_gb):
    rerank = model.startswith("Rerank")
    name = f"Nemotron-3.5-{model}-mlx-{bits}"
    up = f"nvidia/Nemotron-3.5-{model}-BF16"
    fm = ["---", "license: other", "license_name: openmdw-1.1", "license_link: LICENSE", f"base_model: {up}"]
    if bits != "bf16":
        fm.append("base_model_relation: quantized")
    fm += ["library_name: mlx", f"pipeline_tag: {'text-classification' if rerank else 'feature-extraction'}", "tags:", "- mlx", "- apple-silicon",
           *(["- reranker", "- text-ranking"] if rerank else ["- embeddings", "- sentence-similarity", "- matryoshka"]),
           "- multimodal", "- retrieval", "- nemotron", "language:", "- en", "---", ""]
    b = []
    w = b.append
    w(f"# {name}\n")
    w(f"[NVIDIA {up.split('/')[1]}](https://huggingface.co/{up}) converted to [MLX](https://github.com/ml-explore/mlx) for Apple Silicon"
      + ("" if bits == "bf16" else f", with the language tower quantized to {label(bits)}") + ". "
      + ("A reranker: it scores a query against a document (text, a page image, or both) and returns a relevance logit."
         if rerank else "An embedding model for retrieval: text queries and text or page-image documents go into the same vector space.") + "\n")
    w("Built by [NetProjects](https://netprojects.com). Not affiliated with NVIDIA. All credit for the model goes to NVIDIA; this repository only changes the format"
      + ("" if bits == "bf16" else " and the precision") + ".\n")
    w("## What is in this build\n")
    kept = "vision encoder and projector (bf16), norms" + (", score head (float32)" if rerank else "") + (", mean pooling and L2 normalisation (computed in float32)" if not rerank else ", mean pooling (float32)")
    w(f"- Language tower: {'bf16' if bits == 'bf16' else label(bits) + ' affine quantization, group size 64'}")
    w(f"- Kept at original precision: {kept}")
    w(f"- Size: {size_gb}")
    w("- Same inputs as the original: queries `query: …`, documents `passage: …` through NVIDIA's chat template; one image per document; up to 32,768 tokens"
      if not rerank else "- Same inputs as the original: NVIDIA's chat template with `query` and `document` roles; one image per document; up to 32,768 tokens")
    w(f"- Converted from upstream revision `{REV[model]}`\n")
    w("## Quick start\n")
    w("```bash\npip install git+https://github.com/internet-projects/nemotron-retrieval-mlx   # or clone it and pip install -e .\n```\n")
    if rerank:
        w(f"""```python
from nemotron_mlx import Reranker

r = Reranker("netprojects/{name}")
docs = ["The Acropolis Museum in Athens displays the Parthenon sculptures.",
        "A heat pump moves heat.",
        {{"image": "page.png"}},                       # a page image (path, bytes, PIL image or data URL)
        {{"text": "Figure 3", "image": "chart.png"}}]
for hit in r.rank("Where can I see the Parthenon sculptures?", docs, top_k=3):
    print(round(hit["score"], 3), hit["index"])
```
""")
    else:
        w(f"""```python
from nemotron_mlx import Embedder

e = Embedder("netprojects/{name}")
docs = e.embed_documents(["The Acropolis Museum in Athens displays the Parthenon sculptures.",
                          {{"image": "page.png"}},                      # a page image (path, bytes, PIL image or data URL)
                          {{"text": "Figure 3", "image": "chart.png"}}])
q = e.embed_queries(["Where can I see the Parthenon sculptures?"])
print(q @ docs.T)                                          # cosine similarity (vectors are L2-normalised)
small = e.embed_queries(["…"], dim=512)                    # Matryoshka: shorter vectors
```
""")
    w("Local server (OpenAI-style `/v1/embeddings`, Cohere/Jina-style `/v1/rerank`; images as data URLs or base64):\n")
    w(f"```bash\npython -m nemotron_mlx.server --{'rerank' if rerank else 'embed'} netprojects/{name} --port 8095\n```\n")
    w("## Measured results\n")
    w("All numbers below were measured on this build. The reference is NVIDIA's own transformers code from the original card, run in float32.\n")
    # parity
    kind = "Rerank-8B" if rerank else model
    p1 = j(f"{R}/parity-Nemotron-3.5-{kind}-{bits}-vs-ref.json")
    p2 = j(f"{R}/v2/parity-Nemotron-3.5-{kind}-{bits}-vs-ref.json")
    w("### Same output as NVIDIA's code\n")
    if rerank:
        w("| Test set | Pairs | Spearman | Same top document | Max score difference |\n|---|---|---|---|---|")
        if p1:
            w(f"| SciFact queries x abstracts | {p1['inputs']} | {p1['spearman']:.4f} | {p1['top1_same']} | {p1['score_abs_diff_max']:.3f} |")
        if p2:
            w(f"| long documents (4K+ tokens) and page images | {p2['inputs']} ({p2.get('n_long_inputs', 0)} long, {p2.get('n_image_inputs', 0)} image) | {p2['spearman']:.4f} | {p2['top1_same']} | {p2['score_abs_diff_max']:.3f} |")
    else:
        w("| Test set | Inputs | Cosine to NVIDIA (min / mean) | Same top-1 | Top-10 overlap |\n|---|---|---|---|---|")
        if p1:
            w(f"| SciFact queries and abstracts | {p1['inputs']} | {p1['cosine_min']:.4f} / {p1['cosine_mean']:.4f} | {p1['top1_same']} | {p1['top10_overlap_mean']:.2f} |")
        if p2:
            w(f"| long documents (4K+ tokens) and page images | {p2['inputs']} ({p2.get('n_long_inputs', 0)} long, {p2.get('n_image_inputs', 0)} image) | {p2['cosine_min']:.4f} / {p2['cosine_mean']:.4f} | {p2['top1_same']} | {p2['top10_overlap_mean']:.2f} |")
        if p2 and "cosine_min_image" in p2:
            w(f"\nLowest cosine on page images: {p2['cosine_min_image']:.4f}; on 4K+ token documents: {p2['cosine_min_long']:.4f}.")
    w("\nTokenisation is identical to the original on every input.\n" if (p1 or p2) else "")
    # retrieval
    if not rerank:
        v = j(f"{R}/v2/vidore3-computer_science-Nemotron-3.5-{model}-mlx-{bits}.json") or j(f"{R}/v2/vidore-computer_science-Nemotron-3.5-{model}-mlx-{bits}.json")
        if v:
            w("### Page-image retrieval: ViDoRe v3 Computer Science (English)\n")
            w("Text questions against 1,360 page images, nDCG@10 as MTEB computes it, dataset revision `fb7fb69`.\n")
            w("| | nDCG@10 | Recall@100 |\n|---|---|---|")
            w(f"| **This build** | **{v['ndcg@10']:.4f}** | {v['recall@100']:.4f} |")
            w(f"| NVIDIA's published score (MTEB, original model) | {NV[model]:.4f} | — |")
            for k in (1024, 512):
                if f"ndcg@10_dim{k}" in v:
                    w(f"| This build, {k}-d Matryoshka vectors | {v[f'ndcg@10_dim{k}']:.4f} | — |")
            w("")
        s = j(f"{R}/v2/beir-scifact-Nemotron-3.5-{model}-mlx-{bits}.json")
        if s:
            w("### Text retrieval: SciFact (BEIR)\n")
            w(f"Full test set: {s['queries']} queries, {s['docs']:,} abstracts. nDCG@10 **{s['ndcg@10']:.4f}**, Recall@100 {s['recall@100']:.2f}.\n")
    if model == "Embed-8B":
        sl = j(f"{R}/v2/beir-scifact-slice-Embed-8B-mlx-{bits}.json")
        ref = j(f"{R}/v2/beir-scifact-slice-Embed-8B-BF16.json")
        if sl:
            w("### Text retrieval: SciFact (BEIR) slice\n")
            w(f"First {sl['queries']} test queries against their relevant abstracts plus random others ({sl['docs']:,} in all): nDCG@10 **{sl['ndcg@10']:.4f}**"
              + (f" (the original weights through the same code: {ref['ndcg@10']:.4f})" if ref else "") + ".\n")
    if rerank and bits == "8bit":
        vr = j(f"{R}/v2/vidore3-computer_science-rerank-mlx-8bit.json")
        br = j(f"{R}/v2/beir-scifact-slice-rerank-mlx-8bit.json")
        if vr or br:
            w("### Reranking\n")
            w("First stage: Nemotron-3.5-Embed-8B-mlx-8bit. nDCG@10 before and after reranking its top results with this build.\n")
            w("| Test | Queries | Reranked | Embedding only | After reranking |\n|---|---|---|---|---|")
            if vr:
                w(f"| ViDoRe v3 Computer Science (page images) | {vr['queries']} | top {vr['rerank_top']} | {vr['ndcg@10']:.4f} | **{vr['rerank_ndcg@10']:.4f}** |")
            if br:
                w(f"| SciFact slice ({br['docs']:,} abstracts) | {br['queries']} | top {br['rerank_top']} | {br['ndcg@10']:.4f} | {br['rerank_ndcg@10']:.4f} |")
            w("")
    # speed
    sp = j(f"{R}/speed-Nemotron-3.5-{model}-mlx-{bits}.json")
    if sp:
        unit = "pairs_per_s" if rerank else "docs_per_s"
        w("### Speed (text)\n")
        w(f"{'Pairs' if rerank else 'Documents'} per second; measured with other model servers stopped.\n")
        w("| Tokens per input | Batch | " + ("Pairs/s" if rerank else "Docs/s") + " | Tokens/s | Peak memory |\n|---|---|---|---|---|")
        for r in sp["runs"]:
            w(f"| {r['tokens']} | {r['batch']} | {r[unit]} | {r['tokens_per_s']:,} | {r['peak_gb']} GB |")
        w("")
    w("## Licence\n")
    w("The model is NVIDIA's, released under the [OpenMDW License Agreement, version 1.1](LICENSE); the `LICENSE` file in this repository is that agreement, kept as it requires. "
      f"It is built on mistralai/{BASE[model]} (Apache-2.0); see `NOTICE`" + (" and `THIRD_PARTY_NOTICES.md`" if rerank else "") + ".\n")
    w("## Credits\n")
    w(f"- Model: NVIDIA, [{up}](https://huggingface.co/{up})")
    w("- MLX conversion and tests: Paris Michalas, Partner at [NetProjects](https://netprojects.com)")
    w("- Built on [MLX](https://github.com/ml-explore/mlx), [mlx-lm](https://github.com/ml-explore/mlx-lm) and [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)\n")
    w(f"Built and tested on Apple M5 Max (128 GB) · upstream revision `{REV[model]}`")
    return name, "\n".join(fm + b) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--sizes", required=True, help="json {build name: size string}")
    ap.add_argument("--builds", default="Embed-1B:bf16,Embed-1B:8bit,Embed-8B:8bit,Embed-8B:4bit,Rerank-8B:8bit,Rerank-8B:4bit")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sizes = json.load(open(a.sizes))
    for spec in a.builds.split(","):
        m, b = spec.split(":")
        name, text = card(m, b, a.results, sizes.get(f"Nemotron-3.5-{m}-mlx-{b}", "?"))
        os.makedirs(os.path.join(a.out, name), exist_ok=True)
        open(os.path.join(a.out, name, "README.md"), "w").write(text)
        print("wrote", name)


if __name__ == "__main__":
    main()
