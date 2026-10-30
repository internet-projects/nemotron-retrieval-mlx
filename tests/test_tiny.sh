#!/bin/bash
# Port check without the big downloads: tiny random embed and rerank models with the originals' layout, tokenizer and chat
# template, run through NVIDIA's transformers reference code and through this port (bf16 MLX path, padded batches,
# 4K-token documents). Expect cosine ~1.0 and Spearman ~1.0.   tests/test_tiny.sh <Embed-1B folder> <Rerank-8B folder>
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"; T="$(mktemp -d)"
"$PY" scripts/make_tiny.py --src "$1" --out "$T/embed"
"$PY" scripts/make_tiny.py --src "$2" --out "$T/rerank" --rerank
"$PY" parity.py cases --out "$T/cases.json" --n-queries 12 --n-docs 40
for k in embed rerank; do
  "$PY" parity.py ref --model "$T/$k" --kind $k --cases "$T/cases.json" --out "$T/$k-ref.npz" > /dev/null 2>&1
  "$PY" parity.py mlx --model "$T/$k" --kind $k --cases "$T/cases.json" --out "$T/$k-mlx.npz" > /dev/null 2>&1
  "$PY" parity.py compare "$T/$k-ref.npz" "$T/$k-mlx.npz" --json "$T/$k.json"
done
"$PY" -c "import json,sys; e=json.load(open(sys.argv[1])); r=json.load(open(sys.argv[2])); ok = e['cosine_min'] > 0.999 and r['spearman'] > 0.999 and e['token_ids_identical'] == e['inputs'] and r['token_ids_identical'] == r['inputs']; print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)" "$T/embed.json" "$T/rerank.json"
