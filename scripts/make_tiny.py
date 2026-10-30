#!/usr/bin/env python3
"""Test fixture: tiny random Nemotron-3.5-shaped checkpoints (embed and rerank) in the original layout, with the original
tokenizer, processor and chat template, so the port can be checked against the transformers reference without a download.
    python scripts/make_tiny.py --src <folder with an original model's small files> --out DIR [--rerank]"""
import argparse, json, os, shutil
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rerank", action="store_true")
    a = ap.parse_args()
    from safetensors.torch import save_file
    from transformers import AutoConfig, Mistral3Model
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    tc = cfg["text_config"]
    tc.update(hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    vc = cfg["vision_config"]
    vc.update(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, head_dim=16)
    os.makedirs(a.out, exist_ok=True)
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    for f in os.listdir(a.src):
        if f.endswith((".json", ".jinja", ".py")) and f not in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(os.path.join(a.src, f), os.path.join(a.out, f))
    for d in ("1_Pooling", "2_Dense"):
        if os.path.isdir(os.path.join(a.src, d)):
            shutil.copytree(os.path.join(a.src, d), os.path.join(a.out, d), dirs_exist_ok=True)
    torch.manual_seed(1)
    m = Mistral3Model(AutoConfig.from_pretrained(a.out)).float()
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.copy_(torch.randn_like(p) * (0.1 if p.ndim > 1 else 0.2) + (1.0 if p.ndim == 1 and "norm" in n else 0.0))
    sd = {k: v.contiguous() for k, v in m.state_dict().items() if "inv_freq" not in k}
    if a.rerank:
        sd = {"model." + k: v for k, v in sd.items()}
        sd["language_model.score.weight"] = torch.randn(1, 64)
    save_file(sd, os.path.join(a.out, "model-00001-of-00001.safetensors"))
    json.dump({"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in sd}},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"))
    print("tiny model in", a.out, sum(v.numel() for v in sd.values()), "params")


if __name__ == "__main__":
    main()
