"""Consolidated model configuration for DIET finetuning."""

from typing import Dict, List
from config.base_config import BaseConfig
import dataclasses


# Model architecture specifications - single source of truth
MODEL_SPECS = {
    "resnet50": {
        "sizes": [
            "default",
            "a1_in1k",
            "clip.cc12m",
            "clip.openai",
            "tv2_in1k",
            "gluon_in1k",
            "fb_swsl_ig1b_ft_in1k",
            "ram_in1k",
        ],
        "embedding_dims": {
            "default": 2048,
            "a1_in1k": 2048,
            "clip.cc12m": 2048,
            "clip.openai": 2048,
            "tv2_in1k": 2048,
            "gluon_in1k": 2048,
            "fb_swsl_ig1b_ft_in1k": 2048,
            "ram_in1k": 2048,
        },
    },
    "dinov2": {
        "sizes": ["small", "base", "large", "giant"],
        "embedding_dims": {"small": 384, "base": 768, "large": 1024, "giant": 1536},
    },
    "dinov3": {
        "sizes": [
            "small",
            "base",
            "large",
            "s16",
            "s16plus",
            "b16",
            "l16",
            "h16plus",
            "7b16",
        ],
        "embedding_dims": {
            "small": 384,
            "base": 768,
            "large": 1024,
            "s16": 384,
            "s16plus": 384,
            "b16": 768,
            "l16": 1024,
            "h16plus": 1280,
            "7b16": 4096,
        },
    },
    "mae": {
        "sizes": ["base", "large", "huge"],
        "embedding_dims": {"base": 768, "large": 1024, "huge": 1280},
    },
    "mambavision": {
        "sizes": ["T", "S", "B", "L", "L2", "L3-512", "L3-256"],
        "embedding_dims": {
            "T": 512,
            "S": 768,
            "B": 1024,
            "L": 1280,
            "L2": 1408,
            "L3-512": 1568,
            "L3-256": 1568,
        },
    },
    "ijepa": {
        "sizes": ["h16_1k", "g16_22k", "h14_22k", "h14_1k"],
        "embedding_dims": {
            "h16_1k": 1408,
            "g16_22k": 1408,
            "h14_22k": 1408,
            "h14_1k": 1408,
        },
    },
    "aim": {
        "sizes": ["600M", "1B", "3B", "7B"],
        "embedding_dims": {"600M": 768, "1B": 1024, "3B": 1536, "7B": 2048},
    },
    "simclr": {
        "sizes": ["resnet50-1x", "resnet50-2x", "resnet50-4x"],
        "embedding_dims": {
            "resnet50-1x": 2048,
            "resnet50-2x": 2048,
            "resnet50-4x": 2048,
        },
    },
}

# Fallback embedding dimensions
FALLBACK_EMBEDDING_DIMS = {
    "resnet50": 2048,
    "dinov2": 384,
    "dinov3": 384,
    "mae": 768,
    "mambavision": 512,
    "ijepa": 1408,
    "aim": 768,
    "simclr": 2048,
}

# Performance thresholds for sanity checks
SANITY_CHECK_THRESHOLDS = {
    "dinov2": 0.91,
    "dinov3": 0.91,
    "mae": 0.85,
    "mambavision": 0.85,
    "ijepa": 0.85,
    "aim": 0.75,
    "resnet50": 0.80,
    "simclr": 0.80,
}


@dataclasses.dataclass
class ModelConfig(BaseConfig):
    """Model configuration."""

    backbone_type: str
    model_size: str
    temperature: float = 3.0

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension for this model configuration."""
        return get_model_embedding_dim(self.backbone_type, self.model_size)

    def validate(self) -> None:
        pass


def get_model_embedding_dim(backbone_type: str, model_size: str) -> int:
    """Get embedding dimension for specified model configuration."""
    if backbone_type in MODEL_SPECS:
        if model_size in MODEL_SPECS[backbone_type]["embedding_dims"]:
            return MODEL_SPECS[backbone_type]["embedding_dims"][model_size]

    return FALLBACK_EMBEDDING_DIMS.get(backbone_type, 768)


def get_available_model_sizes(backbone_type: str) -> List[str]:
    """Get available sizes for a specific model backbone."""
    return MODEL_SPECS.get(backbone_type, {}).get("sizes", [])


def get_available_backbones() -> List[str]:
    """Get all available model backbones."""
    return list(MODEL_SPECS.keys())
