"""Consolidated training configuration for DIET finetuning."""

from typing import Dict, Any, Union, Tuple
from config.base_config import BaseConfig
from config.models import ModelConfig
from config.data import DataConfig
import dataclasses


@dataclasses.dataclass
class TrainingConfig(BaseConfig):
    """Training configuration."""

    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.3
    eval_frequency: int = 5
    checkpoint_freq: int = 500
    diet_head_only_epochs: float = 0.05
    num_trained_blocks: int = -1

    # Mixup/CutMix augmentation parameters
    mixup_alpha: float = 1.0
    cutmix_alpha: float = 1.0
    mixup_cutmix_prob: float = 0.8
    mixup_cutmix_switch_prob: float = 0.5

    def validate(self) -> None:
        """Validate training configuration."""
        if self.num_epochs <= 0:
            raise ValueError(f"Epochs must be positive, got {self.num_epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"LR must be positive, got {self.learning_rate}")
        if not 0 <= self.label_smoothing <= 1:
            raise ValueError(
                f"Label smoothing must be in [0,1], got {self.label_smoothing}"
            )
        if self.diet_head_only_epochs < 0:
            raise ValueError(
                f"diet_head_only_epochs must be >= 0, got {self.diet_head_only_epochs}"
            )
        if self.num_trained_blocks < -1:
            raise ValueError(
                f"num_trained_blocks must be >= -1, got {self.num_trained_blocks}"
            )
        if self.mixup_alpha < 0:
            raise ValueError(f"mixup_alpha must be >= 0, got {self.mixup_alpha}")
        if self.cutmix_alpha < 0:
            raise ValueError(f"cutmix_alpha must be >= 0, got {self.cutmix_alpha}")
        if not 0 <= self.mixup_cutmix_prob <= 1:
            raise ValueError(
                f"mixup_cutmix_prob must be in [0,1], got {self.mixup_cutmix_prob}"
            )
        if not 0 <= self.mixup_cutmix_switch_prob <= 1:
            raise ValueError(
                f"mixup_cutmix_switch_prob must be in [0,1], "
                f"got {self.mixup_cutmix_switch_prob}"
            )


@dataclasses.dataclass
class ExperimentConfig(BaseConfig):
    """Complete experiment configuration combining all sub-configs."""

    model: ModelConfig
    data: DataConfig
    training: TrainingConfig

    # Paths
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"

    # Logging
    enable_wandb: bool = True
    wandb_project: str = "DIET-Finetuning"
    wandb_dir: str = "wandb"
    wandb_prefix: str = ""

    # Evaluation
    eval_on_test: bool = False
    resume_from: str = ""

    # Data augmentation
    da_strength: float = 1.0

    def validate(self) -> None:
        """Validate complete experiment configuration."""
        self.model.validate()
        self.data.validate()
        self.training.validate()

    @property
    def num_diet_classes(self) -> int:
        """Calculate number of DIET classes based on data limit and total classes."""
        return min(self.data.limit_data, self.data.num_classes)

    @property
    def is_diet_active(self) -> bool:
        """Check if DIET is active based on label smoothing."""
        return self.training.label_smoothing > 0

    def to_wandb_config(self) -> Dict[str, Any]:
        """Convert to wandb-compatible configuration dict."""
        return {
            # Model parameters
            "backbone_type": self.model.backbone_type,
            "model_size": self.model.model_size,
            "embedding_dim": self.model.embedding_dim,
            # Dataset parameters
            "dataset_name": self.data.dataset_name,
            "num_classes": self.data.num_classes,
            "num_diet_classes": self.num_diet_classes,
            "input_size": self.data.input_size,
            "is_rgb": self.data.is_rgb,
            "batch_size": self.data.batch_size,
            "limit_data": self.data.limit_data,
            # Training parameters
            "num_epochs": self.training.num_epochs,
            "learning_rate": self.training.learning_rate,
            "weight_decay": self.training.weight_decay,
            "label_smoothing": self.training.label_smoothing,
            "eval_frequency": self.training.eval_frequency,
            "checkpoint_freq": self.training.checkpoint_freq,
            "diet_head_only_epochs": self.training.diet_head_only_epochs,
            "num_trained_blocks": self.training.num_trained_blocks,
            # Mixup/CutMix parameters
            "mixup_alpha": self.training.mixup_alpha,
            "cutmix_alpha": self.training.cutmix_alpha,
            "mixup_cutmix_prob": self.training.mixup_cutmix_prob,
            "mixup_cutmix_switch_prob": self.training.mixup_cutmix_switch_prob,
            # Path parameters
            "checkpoint_dir": self.checkpoint_dir,
            "results_dir": self.results_dir,
            # Additional parameters
            "da_strength": self.da_strength,
            "eval_on_test": self.eval_on_test,
            "resume_from": self.resume_from,
            "is_diet_active": self.is_diet_active,
            # Wandb parameters
            "wandb_prefix": self.wandb_prefix,
            "wandb_dir": self.wandb_dir,
            "enable_wandb": self.enable_wandb,
            "wandb_project": self.wandb_project,
        }


def create_experiment_config_from_args(args) -> ExperimentConfig:
    """Create complete experiment configuration from command line arguments."""

    # Create sub-configurations
    model_config = ModelConfig(
        backbone_type=args.backbone,
        model_size=args.model_size,
        temperature=getattr(args, "temperature", 1.0),
    )

    data_config = DataConfig(
        dataset_name=args.dataset,
        batch_size=getattr(args, "batch_size", 32),
        limit_data=getattr(args, "limit_data", 1000),
        num_diet_classes=None,  # This will be computed dynamically (min(limit_data, len(dataset)))
    )

    training_config = TrainingConfig(
        num_epochs=getattr(args, "num_epochs", 30),
        learning_rate=getattr(args, "lr", 1e-4),
        weight_decay=getattr(args, "weight_decay", 0.05),
        label_smoothing=getattr(args, "label_smoothing", 0.3),
        eval_frequency=getattr(args, "eval_frequency", 5),
        checkpoint_freq=getattr(args, "checkpoint_freq", 500),
        diet_head_only_epochs=getattr(args, "diet_head_only_epochs", 0.05),
        num_trained_blocks=getattr(args, "num_trained_blocks", -1),
        mixup_alpha=getattr(args, "mixup_alpha", 1.0),
        cutmix_alpha=getattr(args, "cutmix_alpha", 1.0),
        mixup_cutmix_prob=getattr(args, "mixup_cutmix_prob", 0.8),
        mixup_cutmix_switch_prob=getattr(args, "mixup_cutmix_switch_prob", 0.5),
    )

    # Create complete configuration
    config = ExperimentConfig(
        model=model_config,
        data=data_config,
        training=training_config,
        checkpoint_dir=getattr(args, "checkpoint_dir", "checkpoints"),
        results_dir=getattr(args, "results_dir", "results"),
        enable_wandb=getattr(args, "use_wandb", False),
        wandb_project=getattr(args, "wandb_project", "diet"),
        wandb_dir=getattr(args, "wandb_dir", "wandb"),
        wandb_prefix=getattr(args, "wandb_prefix", ""),
        eval_on_test=getattr(args, "eval_on_test", False),
        resume_from=getattr(args, "resume_from", ""),
        da_strength=getattr(args, "da_strength", 1.0),
    )

    # Validate the complete configuration
    config.validate()

    return config
