"""DINOv2/v3 model implementation for DIET finetuning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from config.models import MODEL_SPECS

import logging
from typing import Dict, Optional

logger = logging.getLogger("dinov3")


class DINOWrapper(nn.Module):
    def __init__(self, model, version):
        super().__init__()
        self.model = model
        self.version = version

    def forward(self, x):
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        outputs = self.model(x)

        if self.version == "v3":
            return outputs.pooler_output
        else:
            return outputs.last_hidden_state[:, 0]


def get_dinov2_model(device, model_size="small"):
    """Create DINOv2 model."""
    return get_dino_model(device, model_size, version="v2")


def get_dinov3_model(device, model_size="small"):
    """Create DINOv3 model."""
    return get_dino_model(device, model_size, version="v3")


def get_dino_model(device, model_size, version="v2"):
    """Create DINOv2 or DINOv3 model."""

    backbone_name = f"dino{version}"

    if backbone_name not in MODEL_SPECS:
        raise ValueError(f"Version {version} not supported")

    spec = MODEL_SPECS[backbone_name]

    # Handle model size and aliases for DINOv3
    if version == "v3" and model_size in ["small", "base", "large"]:
        aliases = {"small": "s16", "base": "b16", "large": "l16"}
        actual_size = aliases[model_size]
    else:
        actual_size = model_size

    if model_size not in spec["sizes"]:
        raise ValueError(f"Model size {model_size} not supported for DINO{version}")

    # Build model name
    if version == "v2":
        model_name = f"facebook/dinov2-{actual_size}"
    else:  # v3
        model_name = f"facebook/dinov3-vit{actual_size}-pretrain-lvd1689m"

    base_model = AutoModel.from_pretrained(model_name)

    # Unfreeze all parameters
    for param in base_model.parameters():
        param.requires_grad = True

    model = DINOWrapper(base_model, version).to(device)
    embedding_dim = spec["embedding_dims"][model_size]

    return model, embedding_dim
