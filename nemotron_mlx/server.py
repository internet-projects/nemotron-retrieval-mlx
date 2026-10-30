"""Local HTTP server: OpenAI-compatible POST /v1/embeddings and a Cohere/Jina-style POST /v1/rerank.

    python -m nemotron_mlx.server --embed <embed folder or HF id> [--rerank <rerank folder or HF id>] --port 8095

/v1/embeddings  {"model": ..., "input": "text" | ["text", ...], "input_type": "query" | "document" (default "document"),
                 "dimensions": optional Matryoshka size}
                -> {"object": "list", "data": [{"object": "embedding", "index": i, "embedding": [...]}], "model": ..., "usage": {...}}
/v1/rerank      {"model": ..., "query": "...", "documents": ["...", ...], "top_n": optional}
                -> {"results": [{"index": i, "relevance_score": logit, "document": {"text": ...}}], "model": ...}
GLiDE-style memory hygiene: one model per role, all MLX work on one worker thread, MLX buffer cache capped and cleared.
Set NEMOTRON_API_KEY to require "Authorization: Bearer <key>".
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hmac
import os
import time


def create_app(embed_path=None, rerank_path=None):
    from fastapi import FastAPI, Header
    from fastapi.responses import JSONResponse
    from contextlib import asynccontextmanager
    key = os.environ.get("NEMOTRON_API_KEY", "")
    worker = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="nemotron-mlx")
    ctx = {}

    def load():
        from .model import Embedder, Reranker
        if embed_path:
            ctx["embed"] = Embedder(embed_path)
        if rerank_path:
            ctx["rerank"] = Reranker(rerank_path)

    @asynccontextmanager
    async def lifespan(_app):
        worker.submit(load).result()
        yield
        worker.shutdown(wait=False)

    app = FastAPI(title="Nemotron 3.5 retrieval (MLX)", lifespan=lifespan)

    def err(code, msg):
        return JSONResponse(status_code=code, content={"error": {"message": msg}})

    def auth(h):
        return not key or hmac.compare_digest(h.encode(), f"Bearer {key}".encode())

    @app.get("/health")
    def health():
        return {"status": "ok", "embed": embed_path, "rerank": rerank_path}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": p, "object": "model"} for p in (embed_path, rerank_path) if p]}

    @app.post("/v1/embeddings")
    def embeddings(body: dict, authorization: str = Header(default="")):
        if not auth(authorization):
            return err(401, "unauthorized")
        if "embed" not in ctx:
            return err(404, "no embedding model loaded")
        inp = body.get("input")
        texts = [inp] if isinstance(inp, str) else inp
        if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
            return err(400, "input must be a string or a list of strings")
        kind = body.get("input_type", "document")
        if kind not in ("query", "document"):
            return err(400, "input_type must be 'query' or 'document'")
        e = ctx["embed"]
        try:
            vecs, n = worker.submit(lambda: (e.embed(texts, kind, body.get("dimensions")), sum(len(x) for x in e.ids(texts, kind)))).result()
        except ValueError as ex:
            return err(400, str(ex))
        return {"object": "list", "model": body.get("model") or embed_path,
                "data": [{"object": "embedding", "index": i, "embedding": v.tolist()} for i, v in enumerate(vecs)],
                "usage": {"prompt_tokens": n, "total_tokens": n}}

    @app.post("/v1/rerank")
    def rerank(body: dict, authorization: str = Header(default="")):
        if not auth(authorization):
            return err(401, "unauthorized")
        if "rerank" not in ctx:
            return err(404, "no rerank model loaded")
        q, docs = body.get("query"), body.get("documents")
        docs = [d if isinstance(d, str) else (d or {}).get("text") for d in (docs or [])]
        if not isinstance(q, str) or not docs or not all(isinstance(d, str) for d in docs):
            return err(400, "query must be a string and documents a non-empty list of strings")
        try:
            ranked = worker.submit(lambda: ctx["rerank"].rank(q, docs, body.get("top_n"))).result()
        except ValueError as ex:
            return err(400, str(ex))
        return {"model": body.get("model") or rerank_path,
                "results": [{"index": r["index"], "relevance_score": r["score"], "document": {"text": r["document"]}} for r in ranked]}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed")
    ap.add_argument("--rerank")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8095)
    a = ap.parse_args()
    if not (a.embed or a.rerank):
        ap.error("give --embed and/or --rerank")
    import uvicorn
    uvicorn.run(create_app(a.embed, a.rerank), host=a.host, port=a.port, workers=1)


if __name__ == "__main__":
    main()
