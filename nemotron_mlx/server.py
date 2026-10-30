"""Local HTTP server: OpenAI-compatible POST /v1/embeddings and a Cohere/Jina-style POST /v1/rerank.

    python -m nemotron_mlx.server --embed <embed folder or HF id> [--rerank <rerank folder or HF id>] --port 8095

/v1/embeddings  {"model": ..., "input": item | [item, ...], "input_type": "query" | "document" (default "document"),
                 "dimensions": optional Matryoshka size}
                -> {"object": "list", "data": [{"object": "embedding", "index": i, "embedding": [...]}], "model": ..., "usage": {...}}
/v1/rerank      {"model": ..., "query": "...", "documents": [item, ...], "top_n": optional}
                -> {"results": [{"index": i, "relevance_score": logit, "document": {"text": ...}}], "model": ..., "usage": {...}}
An item is a string, or for documents {"text": ..., "image": ...} with the image as a data URL or base64 string (one image
per document; queries are text only). http(s) image URLs are fetched only when NEMOTRON_ALLOW_IMAGE_URLS=1; local file
paths are never read from requests.
GLiDE-style memory hygiene: one model per role, all MLX work on one worker thread, MLX buffer cache capped and cleared.
Set NEMOTRON_API_KEY to require "Authorization: Bearer <key>".
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hmac
import logging
import os
import time

MAX_IMAGE_CHARS = int(os.environ.get("NEMOTRON_MAX_IMAGE_MB", "20")) * 1024 * 1024   # per image, as sent (base64 / data URL)
log = logging.getLogger("nemotron_mlx.server")


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

    allow_urls = os.environ.get("NEMOTRON_ALLOW_IMAGE_URLS") == "1"

    def item(x):
        """Request item -> str or {"text", "image"}; raises ValueError on anything else."""
        if isinstance(x, str):
            return x
        if not isinstance(x, dict) or not (isinstance(x.get("text"), str) or isinstance(x.get("image"), str)):
            raise ValueError("each item must be a string or an object with a string 'text' and/or 'image'")
        if x.get("text") is not None and not isinstance(x["text"], str):
            raise ValueError("'text' must be a string")
        img = x.get("image")
        if img is not None:
            if not isinstance(img, str):
                raise ValueError("'image' must be a data URL or base64 string")
            if len(img) > MAX_IMAGE_CHARS:
                raise ValueError(f"image is larger than {MAX_IMAGE_CHARS // 2 ** 20} MB (NEMOTRON_MAX_IMAGE_MB)")
            if img.startswith(("http://", "https://")) and not allow_urls:
                raise ValueError("image URLs are disabled on this server (set NEMOTRON_ALLOW_IMAGE_URLS=1); send a data URL or base64")
            if os.path.exists(img):
                raise ValueError("'image' must be a data URL or base64 string, not a file path")
        return {k: x[k] for k in ("text", "image") if x.get(k) is not None}

    def auth(h):
        return not key or hmac.compare_digest(h.encode(), f"Bearer {key}".encode())

    def name(p):
        return os.path.basename(os.path.normpath(p)) if p else None

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models(authorization: str = Header(default="")):
        if not auth(authorization):
            return err(401, "unauthorized")
        return {"object": "list", "data": [{"id": name(p), "object": "model"} for p in (embed_path, rerank_path) if p]}

    def failed(ex):
        if isinstance(ex, ValueError):                 # bad input, an unreadable image, too many tokens
            return err(400, str(ex))
        log.exception("request failed")
        return err(500, "internal error")

    @app.post("/v1/embeddings")
    def embeddings(body: dict, authorization: str = Header(default="")):
        if not auth(authorization):
            return err(401, "unauthorized")
        if "embed" not in ctx:
            return err(404, "no embedding model loaded")
        inp = body.get("input")
        items = inp if isinstance(inp, list) else [inp]
        kind = body.get("input_type", "document")
        if kind not in ("query", "document"):
            return err(400, "input_type must be 'query' or 'document'")
        e = ctx["embed"]
        try:
            if not items or inp is None:
                raise ValueError("input must be an item or a non-empty list of items")
            items = [item(x) for x in items]
            vecs, n = worker.submit(lambda: (e.embed(items, kind, body.get("dimensions")), e.last_tokens)).result()
        except Exception as ex:
            return failed(ex)
        return {"object": "list", "model": body.get("model") or name(embed_path),
                "data": [{"object": "embedding", "index": i, "embedding": v.tolist()} for i, v in enumerate(vecs)],
                "usage": {"prompt_tokens": n, "total_tokens": n}}

    @app.post("/v1/rerank")
    def rerank(body: dict, authorization: str = Header(default="")):
        if not auth(authorization):
            return err(401, "unauthorized")
        if "rerank" not in ctx:
            return err(404, "no rerank model loaded")
        q, docs = body.get("query"), body.get("documents")
        r = ctx["rerank"]
        try:
            if not isinstance(q, str) or not isinstance(docs, list) or not docs:
                raise ValueError("query must be a string and documents a non-empty list")
            docs = [item(d) for d in docs]
            ranked, n = worker.submit(lambda: (r.rank(q, docs, body.get("top_n")), r.last_tokens)).result()
        except Exception as ex:
            return failed(ex)
        def out(d):
            return {"text": d} if isinstance(d, str) else ({"text": d["text"]} if "text" in d else {})
        return {"model": body.get("model") or name(rerank_path), "usage": {"prompt_tokens": n, "total_tokens": n},
                "results": [{"index": x["index"], "relevance_score": x["score"], "document": out(x["document"])} for x in ranked]}

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
