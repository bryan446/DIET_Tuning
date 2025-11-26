"""Consolidated data configuration for DIET finetuning."""

from typing import Dict, Tuple, Union
from config.base_config import BaseConfig
import dataclasses


# Dataset statistics - single source of truth
DATASET_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "input_size": 32,
        "is_rgb": True,
        "num_classes": 10,
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "input_size": 32,
        "is_rgb": True,
        "num_classes": 100,
    },
    "food101": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 101,
    },
    "fgvc_aircraft": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 100,
    },
    "pathmnist": {
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 9,
    },
    "chestmnist": {
        "mean": (0.4984,),
        "std": (0.2483,),
        "input_size": 224,
        "is_rgb": False,
        "num_classes": 2,
    },
    "dermamnist": {
        "mean": (0.7634, 0.5423, 0.5698),
        "std": (0.0841, 0.1246, 0.1043),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 7,
    },
    "octmnist": {
        "mean": (0.1778,),
        "std": (0.1316,),
        "input_size": 224,
        "is_rgb": False,
        "num_classes": 4,
    },
    "pneumoniamnist": {
        "mean": (0.5060,),
        "std": (0.2537,),
        "input_size": 224,
        "is_rgb": False,
        "num_classes": 2,
    },
    "retinamnist": {
        "mean": (0.1706, 0.1706, 0.1706),
        "std": (0.1946, 0.1946, 0.1946),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 5,
    },
    "breastmnist": {
        "mean": (0.4846,),
        "std": (0.2522,),
        "input_size": 224,
        "is_rgb": False,
        "num_classes": 2,
    },
    "organamnist": {
        "mean": (0.4996, 0.4996, 0.4996),
        "std": (0.1731, 0.1731, 0.1731),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 11,
    },
    "organcmnist": {
        "mean": (0.4996, 0.4996, 0.4996),
        "std": (0.1731, 0.1731, 0.1731),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 11,
    },
    "organsmnist": {
        "mean": (0.4996, 0.4996, 0.4996),
        "std": (0.1731, 0.1731, 0.1731),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 11,
    },
    "plantnet300k": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "input_size": 224,
        "is_rgb": True,
        "num_classes": 85,
    },
    "galaxy10_decals": {
        "mean": (0.097, 0.097, 0.097),
        "std": (0.174, 0.164, 0.156),
        "input_size": 256,
        "is_rgb": True,
        "num_classes": 10,
    },
    "crop14_balance": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "input_size": 512,
        "is_rgb": True,
        "num_classes": 14,
    },
}

# Default dataset fallback
DEFAULT_DATASET_STATS = {
    "mean": (0.5,),
    "std": (0.5,),
    "input_size": 224,
    "is_rgb": False,
    "num_classes": 10,
}


@dataclasses.dataclass
class DataConfig(BaseConfig):
    """Data configuration."""

    dataset_name: str
    batch_size: int = 32
    limit_data: int = 1000
    num_diet_classes: Union[int, None] = None  # To be set dynamically

    def validate(self) -> None:
        """Validate data configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        if self.limit_data <= 0:
            raise ValueError(f"Limit data must be positive, got {self.limit_data}")

    @property
    def dataset_stats(self) -> Dict:
        """Get dataset statistics."""
        return get_dataset_stats(self.dataset_name)

    @property
    def mean(self) -> Union[Tuple[float, ...], Tuple[float]]:
        """Get dataset mean."""
        return self.dataset_stats["mean"]

    @property
    def std(self) -> Union[Tuple[float, ...], Tuple[float]]:
        """Get dataset std."""
        return self.dataset_stats["std"]

    @property
    def input_size(self) -> int:
        """Get input size."""
        return self.dataset_stats["input_size"]

    @property
    def is_rgb(self) -> bool:
        """Check if dataset is RGB."""
        return self.dataset_stats["is_rgb"]

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return self.dataset_stats["num_classes"]


def get_dataset_stats(dataset_name: str) -> Dict:
    """Get statistics for the specified dataset."""
    return DATASET_STATS.get(dataset_name.lower(), DEFAULT_DATASET_STATS)


def get_available_datasets() -> list:
    """Get all available datasets."""
    return list(DATASET_STATS.keys())
