---
license: other
license_name: openmdw-1.1
license_link: LICENSE
base_model: nvidia/Nemotron-3.5-Embed-8B-BF16
base_model_relation: quantized
library_name: mlx
pipeline_tag: feature-extraction
tags:
- mlx
- apple-silicon
- embeddings
- sentence-similarity
- matryoshka
- multimodal
- retrieval
- nemotron
language:
- en
---

# Nemotron-3.5-Embed-8B-mlx-4bit

[NVIDIA Nemotron-3.5-Embed-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16) converted to [MLX](https://github.com/ml-explore/mlx) for Apple Silicon, with the language tower quantized to 4-bit. An embedding model for retrieval: text queries and text or page-image documents go into the same vector space.

Built by [NetProjects](https://netprojects.com). Not affiliated with NVIDIA. All credit for the model goes to NVIDIA; this repository only changes the format and the precision.

## What is in this build

- Language tower: 4-bit affine quantization, group size 64
- Kept at original precision: vision encoder and projector (bf16), norms, mean pooling and L2 normalisation (computed in float32)
- Size: 5.0 GB
- Same inputs as the original: queries `query: …`, documents `passage: …` through NVIDIA's chat template; one image per document; up to 32,768 tokens
- Converted from upstream revision `781e779ee4ca9c2d2a4f38321993556baeab49a1`

## Quick start

```bash
pip install git+https://github.com/internet-projects/nemotron-retrieval-mlx   # or clone it and pip install -e .
```

```python
from nemotron_mlx import Embedder

e = Embedder("netprojects/Nemotron-3.5-Embed-8B-mlx-4bit")
docs = e.embed_documents(["The Acropolis Museum in Athens displays the Parthenon sculptures.",
                          {"image": "page.png"},                      # a page image (path, bytes, PIL image or data URL)
                          {"text": "Figure 3", "image": "chart.png"}])
q = e.embed_queries(["Where can I see the Parthenon sculptures?"])
print(q @ docs.T)                                          # cosine similarity (vectors are L2-normalised)
small = e.embed_queries(["…"], dim=512)                    # Matryoshka: shorter vectors
```

Local server (OpenAI-style `/v1/embeddings`, Cohere/Jina-style `/v1/rerank`; images as data URLs or base64):

```bash
python -m nemotron_mlx.server --embed netprojects/Nemotron-3.5-Embed-8B-mlx-4bit --port 8095
```

## Measured results

All numbers below were measured on this build. The reference is NVIDIA's own transformers code from the original card, run in float32.

### Same output as NVIDIA's code

| Test set | Inputs | Cosine to NVIDIA (min / mean) | Same top-1 | Top-10 overlap |
|---|---|---|---|---|
| SciFact queries and abstracts | 200 | 0.9786 / 0.9836 | 39/40 | 0.95 |
| long documents (4K+ tokens) and page images | 52 (3 long, 8 image) | 0.9702 / 0.9831 | 10/10 | 0.95 |

Lowest cosine on page images: 0.9702; on 4K+ token documents: 0.9940.

Tokenisation is identical to the original on every input.

### Page-image retrieval: ViDoRe v3 Computer Science (English)

Text questions against 1,360 page images, nDCG@10 as MTEB computes it, dataset revision `fb7fb69`.

| | nDCG@10 | Recall@100 |
|---|---|---|
| **This build** | **0.7732** | 0.9783 |
| NVIDIA's published score (MTEB, original model) | 0.7736 | — |
| This build, 1024-d Matryoshka vectors | 0.7479 | — |
| This build, 512-d Matryoshka vectors | 0.6678 | — |

### Text retrieval: SciFact (BEIR) slice

First 100 test queries against their relevant abstracts plus random others (1,000 in all): nDCG@10 **0.9080** (the original weights through the same code: 0.9134).

### Speed (text)

Documents per second; measured with other model servers stopped.

| Tokens per input | Batch | Docs/s | Tokens/s | Peak memory |
|---|---|---|---|---|
| 506 | 1 | 6.36 | 3,221 | 5.04 GB |
| 506 | 16 | 5.34 | 2,704 | 5.75 GB |
| 506 | 64 | 5.35 | 2,709 | 8.59 GB |
| 4090 | 1 | 0.58 | 2,384 | 5.46 GB |
| 4090 | 16 | 0.58 | 2,379 | 12.78 GB |
| 4090 | 64 | 0.58 | 2,363 | 37.71 GB |

## Licence

The model is NVIDIA's, released under the [OpenMDW License Agreement, version 1.1](LICENSE); the `LICENSE` file in this repository is that agreement, kept as it requires. It is built on mistralai/Ministral-3-8B-Instruct-2512-BF16 (Apache-2.0); see `NOTICE`.

## Credits

- Model: NVIDIA, [nvidia/Nemotron-3.5-Embed-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16)
- MLX conversion and tests: Paris Michalas, Partner at [NetProjects](https://netprojects.com)
- Built on [MLX](https://github.com/ml-explore/mlx), [mlx-lm](https://github.com/ml-explore/mlx-lm) and [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)

Built and tested on Apple M5 Max (128 GB) · upstream revision `781e779ee4ca9c2d2a4f38321993556baeab49a1`

**More NetProjects MLX builds:** [NetProjects MLX builds collection](https://huggingface.co/collections/netprojects/netprojects-mlx-builds-6ae3b2683a59a1b4fc4c0c3a) · [all models at huggingface.co/netprojects](https://huggingface.co/netprojects) · code: [internet-projects/nemotron-retrieval-mlx](https://github.com/internet-projects/nemotron-retrieval-mlx)
