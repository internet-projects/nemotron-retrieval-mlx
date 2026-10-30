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

# Nemotron-3.5-Rerank-8B-mlx-4bit

[NVIDIA Nemotron-3.5-Rerank-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16) converted to [MLX](https://github.com/ml-explore/mlx) for Apple Silicon, with the language tower quantized to 4-bit. A reranker: it scores a query against a document (text, a page image, or both) and returns a relevance logit.

Built by [NetProjects](https://netprojects.com). Not affiliated with NVIDIA. All credit for the model goes to NVIDIA; this repository only changes the format and the precision.

## What is in this build

- Language tower: 4-bit affine quantization, group size 64
- Kept at original precision: vision encoder and projector (bf16), norms, score head (float32), mean pooling (float32)
- Size: 5.0 GB
- Same inputs as the original: NVIDIA's chat template with `query` and `document` roles; one image per document; up to 32,768 tokens
- Converted from upstream revision `f88f8d9104726747791283ab156d2fcdb29e90a7`

## Quick start

```bash
pip install git+https://github.com/internet-projects/nemotron-retrieval-mlx   # or clone it and pip install -e .
```

```python
from nemotron_mlx import Reranker

r = Reranker("netprojects/Nemotron-3.5-Rerank-8B-mlx-4bit")
docs = ["The Acropolis Museum in Athens displays the Parthenon sculptures.",
        "A heat pump moves heat.",
        {"image": "page.png"},                       # a page image (path, bytes, PIL image or data URL)
        {"text": "Figure 3", "image": "chart.png"}]
for hit in r.rank("Where can I see the Parthenon sculptures?", docs, top_k=3):
    print(round(hit["score"], 3), hit["index"])
```

Local server (OpenAI-style `/v1/embeddings`, Cohere/Jina-style `/v1/rerank`; images as data URLs or base64):

```bash
python -m nemotron_mlx.server --rerank netprojects/Nemotron-3.5-Rerank-8B-mlx-4bit --port 8095
```

## Measured results

All numbers below were measured on this build. The reference is NVIDIA's own transformers code from the original card, run in float32.

### Same output as NVIDIA's code

| Test set | Pairs | Spearman | Same top document | Max score difference |
|---|---|---|---|---|
| SciFact queries x abstracts | 400 | 0.9920 | 37/40 | 0.096 |
| long documents (4K+ tokens) and page images | 120 (14 long, 31 image) | 0.9975 | 9/10 | 0.487 |

Tokenisation is identical to the original on every input.

### Speed (text)

Pairs per second; measured with other model servers stopped.

| Tokens per input | Batch | Pairs/s | Tokens/s | Peak memory |
|---|---|---|---|---|
| 513 | 1 | 5.86 | 3,007 | 5.05 GB |
| 513 | 16 | 5.17 | 2,652 | 5.77 GB |
| 513 | 64 | 5.23 | 2,681 | 8.64 GB |
| 4097 | 1 | 0.58 | 2,378 | 5.46 GB |
| 4097 | 16 | 0.58 | 2,365 | 12.8 GB |
| 4097 | 64 | 0.57 | 2,346 | 37.77 GB |

## Licence

The model is NVIDIA's, released under the [OpenMDW License Agreement, version 1.1](LICENSE); the `LICENSE` file in this repository is that agreement, kept as it requires. It is built on mistralai/Ministral-3-8B-Instruct-2512-BF16 (Apache-2.0); see `NOTICE` and `THIRD_PARTY_NOTICES.md`.

## Credits

- Model: NVIDIA, [nvidia/Nemotron-3.5-Rerank-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16)
- MLX conversion and tests: Paris Michalas, Partner at [NetProjects](https://netprojects.com)
- Built on [MLX](https://github.com/ml-explore/mlx), [mlx-lm](https://github.com/ml-explore/mlx-lm) and [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)

Built and tested on Apple M5 Max (128 GB) · upstream revision `f88f8d9104726747791283ab156d2fcdb29e90a7`

**More NetProjects MLX builds:** [NetProjects MLX builds collection](https://huggingface.co/collections/netprojects/netprojects-mlx-builds-6ae3b2683a59a1b4fc4c0c3a) · [all models at huggingface.co/netprojects](https://huggingface.co/netprojects) · code: [internet-projects/nemotron-retrieval-mlx](https://github.com/internet-projects/nemotron-retrieval-mlx)
