# Nemotron 3.5 retrieval on Apple Silicon (MLX)

Run NVIDIA's Nemotron 3.5 retrieval models natively on a Mac with [MLX](https://github.com/ml-explore/mlx), for text **and page images**:

| Model | What it does | MLX builds |
|---|---|---|
| [nvidia/Nemotron-3.5-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16) | embeddings, 2,048-d (Matryoshka) | [bf16](https://huggingface.co/netprojects/Nemotron-3.5-Embed-1B-mlx-bf16) · [8-bit](https://huggingface.co/netprojects/Nemotron-3.5-Embed-1B-mlx-8bit) |
| [nvidia/Nemotron-3.5-Embed-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16) | embeddings, 4,096-d (Matryoshka) | [8-bit](https://huggingface.co/netprojects/Nemotron-3.5-Embed-8B-mlx-8bit) · [4-bit](https://huggingface.co/netprojects/Nemotron-3.5-Embed-8B-mlx-4bit) |
| [nvidia/Nemotron-3.5-Rerank-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16) | reranker (relevance logit) | [8-bit](https://huggingface.co/netprojects/Nemotron-3.5-Rerank-8B-mlx-8bit) · [4-bit](https://huggingface.co/netprojects/Nemotron-3.5-Rerank-8B-mlx-4bit) |

Built by [NetProjects](https://netprojects.com). Not affiliated with NVIDIA; the models are NVIDIA's, under the OpenMDW-1.1 licence.

## Install

```bash
git clone https://github.com/internet-projects/nemotron-retrieval-mlx && cd nemotron-retrieval-mlx
pip install -e ".[server]"
```

## Use

```python
from nemotron_mlx import Embedder, Reranker

e = Embedder("netprojects/Nemotron-3.5-Embed-1B-mlx-8bit")
docs = ["The Acropolis Museum in Athens displays the Parthenon sculptures.",
        {"image": "slide-12.png"},                          # a page image: path, bytes, PIL image or data URL
        {"text": "Figure 3: opening hours", "image": "poster.jpg"}]
D = e.embed_documents(docs)
q = e.embed_queries(["Where can I see the Parthenon sculptures?"])
print(q @ D.T)                                              # vectors are L2-normalised: this is cosine similarity

r = Reranker("netprojects/Nemotron-3.5-Rerank-8B-mlx-8bit")
print(r.rank("Where can I see the Parthenon sculptures?", docs, top_k=2))
```

Inputs follow NVIDIA's cards exactly (chat template, `query:` / `passage:` prefixes, one image per document, up to 32,768 tokens);
queries are text only. `dim=512` (or any size) gives shorter Matryoshka vectors.

### Local server

```bash
python -m nemotron_mlx.server --embed netprojects/Nemotron-3.5-Embed-1B-mlx-8bit \
                              --rerank netprojects/Nemotron-3.5-Rerank-8B-mlx-8bit --port 8095
```

- `POST /v1/embeddings` (OpenAI format): `{"input": [...], "input_type": "query" | "document", "dimensions": 512}`
- `POST /v1/rerank` (Cohere / Jina format): `{"query": "...", "documents": [...], "top_n": 5}`
- An item is a string or `{"text": ..., "image": "data:image/png;base64,..."}`. Image URLs are fetched only with
  `NEMOTRON_ALLOW_IMAGE_URLS=1`; file paths in requests are never read. `NEMOTRON_API_KEY` turns on bearer auth.

## How the port works

- **Language tower**: mlx-lm's Ministral 3 layers run **bidirectionally** (NVIDIA's `is_causal: false`), with YaRN rope as in
  the checkpoint; embeddings are the masked mean of the last hidden states, L2-normalised; the reranker applies NVIDIA's
  linear score head (kept in float32) to the same mean.
- **Images**: the checkpoint's Pixtral vision encoder and Mistral 3 projector via mlx-vlm, NVIDIA's preprocessing (longest edge
  1,120 px, 28 px blocks) and image-token layout; both kept in bf16 in every build.
- **Quantized builds**: MLX affine quantization, group size 64, of the language tower's linear layers and token embedding.
- **Memory**: inputs are length-sorted into micro-batches under a token budget (`NEMOTRON_TOKEN_BUDGET`, 16,384); MLX's buffer
  cache is capped (`NEMOTRON_CACHE_LIMIT_GB`, 2) and cleared after each batch.

## Checks

| Script | What it checks |
|---|---|
| `parity.py` | same output as NVIDIA's transformers code (float32): cosine / Spearman, token ids, long documents, images |
| `vidore_eval.py` | ViDoRe v3 page-image retrieval (nDCG@10 as MTEB computes it), compared with NVIDIA's published scores |
| `beir_eval.py` | BEIR text retrieval (nDCG@10, Recall@100), optional rerank of the top k |
| `speed.py` | documents / pairs per second and peak memory |
| `tests/test_tiny.sh` | parity on tiny random models with the real layout, no big download |

`convert.py --src <original> --out <folder> --bits 16|8|4` makes a build. Measured results are on each model card.

## Licence

Code in this repository: Apache-2.0 (`LICENSE`). The models: NVIDIA, OpenMDW-1.1 (each model repository carries the agreement
and the notices it requires).

Built and tested on Apple M5 Max (128 GB) by Paris Michalas, Partner at [NetProjects](https://netprojects.com).
