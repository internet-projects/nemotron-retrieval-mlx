---
license: other
license_name: openmdw-1.1
license_link: LICENSE
base_model: nvidia/Nemotron-3.5-Rerank-8B-BF16
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-classification
tags:
- mlx
- apple-silicon
- reranker
- text-ranking
- multimodal
- retrieval
- nemotron
language:
- en
---

# Nemotron-3.5-Rerank-8B-mlx-8bit

[NVIDIA Nemotron-3.5-Rerank-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16) converted to [MLX](https://github.com/ml-explore/mlx) for Apple Silicon, with the language tower quantized to 8-bit. A reranker: it scores a query against a document (text, a page image, or both) and returns a relevance logit.

Built by [NetProjects](https://netprojects.com). Not affiliated with NVIDIA. All credit for the model goes to NVIDIA; this repository only changes the format and the precision.

## What is in this build

- Language tower: 8-bit affine quantization, group size 64
- Kept at original precision: vision encoder and projector (bf16), norms, score head (float32), mean pooling (float32)
- Size: 8.7 GB
- Same inputs as the original: NVIDIA's chat template with `query` and `document` roles; one image per document; up to 32,768 tokens
- Converted from upstream revision `f88f8d9104726747791283ab156d2fcdb29e90a7`

## Quick start

```bash
pip install git+https://github.com/internet-projects/nemotron-retrieval-mlx   # or clone it and pip install -e .
```

```python
from nemotron_mlx import Reranker

r = Reranker("netprojects/Nemotron-3.5-Rerank-8B-mlx-8bit")
docs = ["The Acropolis Museum in Athens displays the Parthenon sculptures.",
        "A heat pump moves heat.",
        {"image": "page.png"},                       # a page image (path, bytes, PIL image or data URL)
        {"text": "Figure 3", "image": "chart.png"}]
for hit in r.rank("Where can I see the Parthenon sculptures?", docs, top_k=3):
    print(round(hit["score"], 3), hit["index"])
```

Local server (OpenAI-style `/v1/embeddings`, Cohere/Jina-style `/v1/rerank`; images as data URLs or base64):

```bash
python -m nemotron_mlx.server --rerank netprojects/Nemotron-3.5-Rerank-8B-mlx-8bit --port 8095
```

## Measured results

All numbers below were measured on this build. The reference is NVIDIA's own transformers code from the original card, run in float32.

### Same output as NVIDIA's code

| Test set | Pairs | Spearman | Same top document | Max score difference |
|---|---|---|---|---|
| SciFact queries x abstracts | 400 | 0.9998 | 37/40 | 0.015 |
| long documents (4K+ tokens) and page images | 120 (14 long, 31 image) | 0.9999 | 10/10 | 0.020 |

Tokenisation is identical to the original on every input.

### Reranking

First stage: Nemotron-3.5-Embed-8B-mlx-8bit. nDCG@10 before and after reranking its top results with this build.

| Test | Queries | Reranked | Embedding only | After reranking |
|---|---|---|---|---|
| ViDoRe v3 Computer Science (page images) | 50 | top 10 | 0.7952 | **0.8315** |
| SciFact slice (1,000 abstracts) | 50 | top 20 | 0.9019 | 0.8847 |

### Speed (text)

Pairs per second; measured with other model servers stopped.

| Tokens per input | Batch | Pairs/s | Tokens/s | Peak memory |
|---|---|---|---|---|
| 513 | 1 | 5.56 | 2,850 | 8.82 GB |
| 513 | 16 | 4.93 | 2,531 | 9.74 GB |
| 513 | 64 | 5.0 | 2,563 | 12.62 GB |
| 4097 | 1 | 0.55 | 2,255 | 9.44 GB |
| 4097 | 16 | 0.55 | 2,261 | 16.77 GB |
| 4097 | 64 | 0.55 | 2,240 | 41.75 GB |

## Licence

The model is NVIDIA's, released under the [OpenMDW License Agreement, version 1.1](LICENSE); the `LICENSE` file in this repository is that agreement, kept as it requires. It is built on mistralai/Ministral-3-8B-Instruct-2512-BF16 (Apache-2.0); see `NOTICE` and `THIRD_PARTY_NOTICES.md`.

## Credits

- Model: NVIDIA, [nvidia/Nemotron-3.5-Rerank-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16)
- MLX conversion and tests: Paris Michalas, Partner at [NetProjects](https://netprojects.com)
- Built on [MLX](https://github.com/ml-explore/mlx), [mlx-lm](https://github.com/ml-explore/mlx-lm) and [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)

Built and tested on Apple M5 Max (128 GB) · upstream revision `f88f8d9104726747791283ab156d2fcdb29e90a7`
