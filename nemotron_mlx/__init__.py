"""NVIDIA Nemotron 3.5 retrieval models (Embed-1B / Embed-8B / Rerank-8B) on Apple Silicon (MLX). Not affiliated with NVIDIA."""
__version__ = "0.1.0"


def __getattr__(name):
    if name in ("Embedder", "Reranker"):
        from . import model
        return getattr(model, name)
    raise AttributeError(name)
