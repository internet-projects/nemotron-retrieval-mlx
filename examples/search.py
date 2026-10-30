"""Embed a few documents, search them with a query, then rerank the hits.
    python examples/search.py --embed netprojects/Nemotron-3.5-Embed-1B-mlx-8bit --rerank netprojects/Nemotron-3.5-Rerank-8B-mlx-8bit"""
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nemotron_mlx import Embedder, Reranker

DOCS = ["The Parthenon is a former temple on the Athenian Acropolis, dedicated to the goddess Athena.",
        "Mitochondria generate most of the chemical energy needed to power a cell's biochemical reactions.",
        "The Acropolis Museum in Athens displays finds from the Acropolis, including Parthenon sculptures.",
        "A heat pump moves thermal energy from a cooler space to a warmer one."]

ap = argparse.ArgumentParser()
ap.add_argument("--embed", required=True)
ap.add_argument("--rerank")
a = ap.parse_args()
e = Embedder(a.embed)
D = e.embed_documents(DOCS)                       # L2-normalised; "passage: " prefix added for you
q = "Where can I see the Parthenon sculptures?"
s = e.embed_queries([q]) @ D.T                    # cosine similarity; "query: " prefix added for you
print("embedding:", D.shape, "| 512-d Matryoshka:", e.embed_queries([q], dim=512).shape)
for i in s[0].argsort()[::-1]:
    print(f"  {s[0][i]:.4f}  {DOCS[i][:70]}")
if a.rerank:
    r = Reranker(a.rerank)
    print("reranked:")
    for hit in r.rank(q, DOCS, top_k=3):
        print(f"  {hit['score']:+.3f}  {hit['document'][:70]}")
