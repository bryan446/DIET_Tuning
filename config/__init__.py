"""Main configuration module for DIET finetuning.

This module provides a clean, consolidated interface to all configuration
components following DRY principles and single source of truth.
"""

# Re-export main configuration classes
from config.base_config import DEVICE
from config.models import (
    ModelConfig,
    get_model_embedding_dim,
    get_available_model_sizes,
)
from config.data import DataConfig, get_dataset_stats, get_available_datasets
from config.experiment import ExperimentConfig, create_experiment_config_from_args
from config.models import SANITY_CHECK_THRESHOLDS


# Keep legacy functions for existing code
def create_trainer_config(args, dataset_info=None):
    """DEPRECATED: Use create_experiment_config_from_args instead."""
    import warnings

    warnings.warn(
        "create_trainer_config is deprecated. Use create_experiment_config_from_args instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Convert to new format temporarily for backward compatibility
    config = create_experiment_config_from_args(args)

    # Return a simplified dict that matches the old TrainerConfig interface
    return {
        "backbone_type": config.model.backbone_type,
        "model_size": config.model.model_size,
        "num_classes": config.data.num_classes,
        "dataset_name": config.data.dataset_name,
        "batch_size": config.data.batch_size,
        "limit_data": config.data.limit_data,
        "num_diet_classes": config.num_diet_classes,
        "input_size": config.data.input_size,
        "is_rgb": config.data.is_rgb,
        "dataset_mean": config.data.mean,
        "dataset_std": config.data.std,
        "num_epochs": config.training.num_epochs,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "label_smoothing": config.training.label_smoothing,
        "eval_frequency": config.training.eval_frequency,
        "checkpoint_freq": config.training.checkpoint_freq,
        "checkpoint_dir": config.checkpoint_dir,
        "enable_wandb": config.enable_wandb,
    }


def create_experiment_config(args, embedding_dim: int, dataset_info: dict):
    """DEPRECATED: Use create_experiment_config_from_args instead."""
    import warnings

    warnings.warn(
        "create_experiment_config is deprecated. Use ExperimentConfig.to_wandb_config() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    config = create_experiment_config_from_args(args)
    return config.to_wandb_config()


# Main public API
__all__ = [
    # Hardware and global settings
    "DEVICE",
    # Configuration classes
    "ModelConfig",
    "DataConfig",
    "ExperimentConfig",
    # Factory functions
    "create_experiment_config_from_args",
    # Utility functions
    "get_model_embedding_dim",
    "get_available_model_sizes",
    "get_dataset_stats",
    "get_available_datasets",
    # Constants
    "SANITY_CHECK_THRESHOLDS",
    # Legacy compatibility (deprecated)
    "create_trainer_config",
    "create_experiment_config",
]
