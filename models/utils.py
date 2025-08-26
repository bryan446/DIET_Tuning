"""Model utility functions shared across training and inference scripts."""

import random
import torch
import numpy as np

from config import DEVICE
from utils.sanity_check import unified_sanity_check

# Model implementations
from models.aim import get_aim_model
from models.dinov2 import get_dinov2_model, get_dinov3_model
from models.ijepa import get_ijepa_model
from models.mae import get_mae_model
from models.mambavision import get_mambavision_model
from models.resnet50 import get_resnet50_model
from models.simclr import get_simclr_model


def set_reproducibility_seeds(seed=42):
    """Set seeds for reproducible results across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set all random seeds to {seed} for reproducibility")


def get_model(backbone_type, model_size, run_sanity_check, use_wandb):
    """Load the appropriate model based on backbone type"""

    if run_sanity_check:
        sanity_results = unified_sanity_check(
            model_type=backbone_type, model_size=model_size, log_to_wandb=use_wandb
        )
        print(sanity_results)

    print(f"Creating {backbone_type}-{model_size} model...")

    if backbone_type == "resnet50":
        return get_resnet50_model(DEVICE, model_size=model_size)

    elif backbone_type == "dinov2":
        return get_dinov2_model(DEVICE, model_size=model_size)

    elif backbone_type == "dinov3":
        return get_dinov3_model(DEVICE, model_size=model_size)

    elif backbone_type == "mae":
        return get_mae_model(DEVICE, model_size=model_size)

    elif backbone_type == "ijepa":
        return get_ijepa_model(DEVICE, model_size=model_size)

    elif backbone_type == "mambavision":
        return get_mambavision_model(DEVICE, model_variant=model_size)

    elif backbone_type == "aim":
        return get_aim_model(DEVICE, model_size=model_size)

    elif backbone_type == "simclr":
        return get_simclr_model(DEVICE, model_size=model_size)

    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
