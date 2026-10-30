"""Image inputs: the checkpoint's Pixtral vision encoder and Mistral3 projector on MLX (via mlx-vlm), plus NVIDIA's image
preprocessing and image-token layout.

  * preprocessing (as the original PixtralProcessor): RGB, resize so the longest edge is at most `longest_edge` (1,120 px by
    default; the cards allow up to 1,540) and both sides are multiples of 28 px (patch 14 x spatial merge 2), bicubic,
    rescale 1/255, CLIP mean/std;
  * tokens: each "[IMG]" placeholder the chat template writes becomes rows of [IMG] tokens (one per 28 x 28 px block),
    rows separated by [IMG_BREAK], ended by [IMG_END];
  * features: vision encoder (last layer) -> projector norm -> 2 x 2 patch merger -> linear / GELU / linear, placed at the
    [IMG] token positions of the language tower's input embeddings.
The vision encoder and projector are kept at their original precision in every build.
"""
from __future__ import annotations

import base64
import io
import os

import numpy as np


def load_image(x):
    """PIL image, path, raw bytes, base64 string or data URL -> RGB PIL image (http(s) URLs are fetched with transformers)."""
    from PIL import Image
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(x))).convert("RGB")
    if isinstance(x, os.PathLike) or (isinstance(x, str) and os.path.exists(x)):
        with Image.open(x) as im:
            return im.convert("RGB")
    if isinstance(x, str) and x.startswith(("http://", "https://")):
        from transformers.image_utils import load_image as hf_load
        return hf_load(x).convert("RGB")
    if isinstance(x, str):
        if x.startswith("data:"):
            x = x.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(x))).convert("RGB")
    raise ValueError("image must be a PIL image, a path, bytes, a base64 string, a data URL or an http(s) URL")


class Vision:
    """Holds the vision encoder + projector and turns images into language-tower input features."""

    def __init__(self, config, weights, path):
        import json
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_vlm.utils import get_model_and_args, update_module_configs
        from mlx_vlm.models.pixtral.image_processing_pixtral import PixtralImageProcessor
        from mlx_vlm.models.mistral3.mistral3 import Mistral3MultiModalProjector
        from mlx_vlm.models.pixtral.vision import VisionModel
        arch, _ = get_model_and_args(config={"model_type": "mistral3"})
        cfg = json.loads(json.dumps(config))
        cfg.setdefault("text_config", {})["model_type"] = "ministral3"
        mc = update_module_configs(arch.ModelConfig.from_dict(cfg), arch, cfg, ["text", "vision"])

        class _Part(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_tower = VisionModel(mc.vision_config)
                self.multi_modal_projector = Mistral3MultiModalProjector(mc)

        self.mx = mx
        self.part = _Part()
        self.part.load_weights(list(weights.items()), strict=True)
        mx.eval(self.part.parameters())
        self.patch = int(mc.vision_config.patch_size)
        self.merge = int(mc.spatial_merge_size)
        self.feature_layer = mc.vision_feature_layer
        pc = {}
        if os.path.exists(os.path.join(path, "processor_config.json")):
            with open(os.path.join(path, "processor_config.json")) as f:
                pc = json.load(f).get("image_processor", {})
        longest = int(os.environ.get("NEMOTRON_IMAGE_LONGEST_EDGE", (pc.get("size") or {}).get("longest_edge", 1120)))
        self.processor = PixtralImageProcessor(size={"longest_edge": longest}, patch_size=self.patch * self.merge,
                                               image_mean=pc.get("image_mean"), image_std=pc.get("image_std"),
                                               resample=pc.get("resample", 3))
        self.img_token, self.break_token, self.end_token = "[IMG]", "[IMG_BREAK]", "[IMG_END]"

    def prepare(self, image):
        """-> (pixel_values [3, H, W] float32, (H, W), token string for this image)."""
        out = self.processor.preprocess([load_image(image)])
        pv, (h, w) = out["pixel_values"][0], out["image_sizes"][0]
        rows, cols = h // (self.patch * self.merge), w // (self.patch * self.merge)
        s = "".join(self.img_token * cols + (self.break_token if r < rows - 1 else self.end_token) for r in range(rows))
        return pv, (int(h), int(w)), s

    def features(self, pv, size):
        """One image -> [rows * cols, text hidden] features (language-tower dtype is applied by the caller)."""
        mx = self.mx
        h, w = size
        x = mx.array(pv)[None].transpose(0, 2, 3, 1)                  # NCHW -> NHWC
        _, states = self.part.vision_tower(x, output_hidden_states=True, image_sizes=[(h, w)])
        f = states[self.feature_layer][0]                              # [patches, vision hidden]
        proj = self.part.multi_modal_projector
        f = proj.norm(f)
        ph, pw, d, m = h // self.patch, w // self.patch, f.shape[-1], self.merge
        # 2 x 2 patch merger, same order as the original unfold (channel-major within each 2 x 2 block)
        f = f.reshape(ph // m, m, pw // m, m, d).transpose(0, 2, 4, 1, 3).reshape(-1, d * m * m)
        f = proj.patch_merger.merging_layer(f)
        return proj.linear_2(proj.gelu(proj.linear_1(f)))


def vision_weights(weights, vision_config):
    """Original or converted checkpoint keys -> mlx-vlm vision_tower / multi_modal_projector keys (conv weight to MLX layout)."""
    out = {}
    for k, v in weights.items():
        k2 = k[len("model."):] if k.startswith("model.") else k
        if not (k2.startswith("vision_tower.") or k2.startswith("multi_modal_projector.")):
            continue
        if k2.startswith("vision_tower.") and not k2.startswith("vision_tower.vision_model."):
            k2 = "vision_tower.vision_model." + k2[len("vision_tower."):]
        if "position_ids" in k2:
            continue
        if k2.endswith("patch_conv.weight") and v.ndim == 4 and v.shape[1] == vision_config.get("num_channels", 3) and v.shape[-1] != v.shape[1]:
            v = v.transpose(0, 2, 3, 1)
        out[k2] = v
    return out
