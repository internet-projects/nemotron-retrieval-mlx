# Pending upload commands (NOT run - waiting for Paris's approval)

Run on the Mac from ~/local-llms/nemotron-retrieval. Each model folder already holds its README.md (card), LICENSE
(OpenMDW-1.1), NOTICE (and THIRD_PARTY_NOTICES.md for the reranker), config, tokenizer, chat template and weights.

```bash
cd ~/local-llms/nemotron-retrieval
HF=~/local-llms/h2o/v1.4.1/repo/.venv/bin/hf
for n in Nemotron-3.5-Embed-1B-mlx-bf16 Nemotron-3.5-Embed-1B-mlx-8bit \
         Nemotron-3.5-Embed-8B-mlx-8bit Nemotron-3.5-Embed-8B-mlx-4bit \
         Nemotron-3.5-Rerank-8B-mlx-8bit Nemotron-3.5-Rerank-8B-mlx-4bit; do
  $HF repo create netprojects/$n --repo-type model --exist-ok
  $HF upload netprojects/$n models/$n . --repo-type model --commit-message "MLX conversion of nvidia/${n%-mlx-*}-BF16"
done
```

GitHub (the cards' install line points here):

```bash
cd ~/local-llms/nemotron-retrieval/nemotron-retrieval-mlx
gh repo create internet-projects/nemotron-retrieval-mlx --public --source . --remote origin \
   --description "NVIDIA Nemotron 3.5 Embed / Rerank (text and page images) on Apple Silicon with MLX"
git push -u origin master
```

Not included: Nemotron-3.5-Embed-1B-mlx-4bit (SciFact 0.742 vs 0.750 at bf16, only 0.6 GB smaller than 8-bit).
