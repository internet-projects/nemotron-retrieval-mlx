First release: NVIDIA Nemotron 3.5 retrieval models on Apple Silicon with MLX, for text and page images.

- `Embedder` and `Reranker` for nvidia/Nemotron-3.5-Embed-1B, Embed-8B and Rerank-8B: bidirectional Ministral 3 tower on mlx-lm, mean pooling, NVIDIA's chat template and prompts; page images through the checkpoint's Pixtral encoder and projector via mlx-vlm.
- Local server: OpenAI-style `/v1/embeddings` and Cohere/Jina-style `/v1/rerank`, images as data URLs or base64.
- Six builds on Hugging Face: Embed-1B bf16 / 8-bit, Embed-8B 8-bit / 4-bit, Rerank-8B 8-bit / 4-bit.

Measured on an Apple M5 Max (128 GB):
- Same output as NVIDIA's transformers code (float32) on text, page images and 4K+ token documents; identical tokenisation on every input.
- ViDoRe v3 Computer Science (page images, nDCG@10): Embed-1B 0.7645 (NVIDIA published 0.7642), Embed-8B 8-bit 0.7730 and 4-bit 0.7732 (NVIDIA published 0.7736); Rerank-8B 8-bit on the Embed-8B top 10: 0.795 → 0.832.
- SciFact (nDCG@10): Embed-1B 0.750 (bf16) / 0.752 (8-bit).

Models are NVIDIA's under OpenMDW-1.1; code Apache-2.0. Not affiliated with NVIDIA. By Paris Michalas, Partner at [NetProjects](https://netprojects.com).
