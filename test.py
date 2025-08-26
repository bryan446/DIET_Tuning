#!/usr/bin/env python
"""
Inference for DIET finetuning experiments.
"""
# Standard library imports
import os
import argparse
import time

# Third-party imports
import numpy as np
import torch

# Configuration imports
from config import create_experiment_config_from_args, DEVICE

# Data loading
from loaders.data_loader import prepare_data_loaders

# Utility modules
from utils.wandb_logger import (
    init_wandb,
    create_experiment_dashboard,
    log_zero_shot_metrics,
)
from models.utils import set_reproducibility_seeds, get_model
from evaluation.metrics import zero_shot_eval


def test(args):
    """Main inference function"""
    print("\n" + "=" * 70)
    backbone_info = f"{args.backbone.upper()}-{args.model_size}"
    print(f"DIET FINETUNING INFERENCE: {backbone_info} on {args.dataset}")
    print("=" * 70)

    # Basic settings
    device = DEVICE
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading {args.dataset} dataset...")
    train_loader, val_loader, test_loader, dataset_info = prepare_data_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        da_strength=args.da_strength,
        limit_data=np.inf,  # not limiting train data for kNN and LP eval
        root=args.data_root,
    )

    num_classes = dataset_info["num_classes"]
    num_diet_classes = dataset_info["num_diet_classes"]
    print(f"Loaded dataset with {num_classes} classes")
    print(f"Dataset size determines {num_diet_classes} diet classes")

    # Create the backbone model
    net, embedding_dim = get_model(
        args.backbone, args.model_size, args.run_sanity_check, args.use_wandb
    )
    print(f"Created backbone with embedding dimension: {embedding_dim}")

    # Ensure embedding_dim is valid
    if embedding_dim is None:
        raise ValueError("embedding_dim cannot be None")

    checkpoint_path = (
        os.path.join(args.checkpoint_dir, args.resume_from)
        if not os.path.isabs(args.resume_from)
        else args.resume_from
    )

    assert os.path.exists(checkpoint_path)

    # Create experiment configuration
    config = create_experiment_config_from_args(args)

    # Convert to wandb format for logging
    experiment_config = config.to_wandb_config()

    # Initialize wandb if enabled
    run = None
    if args.use_wandb:
        run = init_wandb(experiment_config)

    # initial kNN and LP eval
    print("\n" + "=" * 50)
    print("INITIAL ZERO-SHOT EVALUATION (BEFORE DIET CONTINUED PRETRAINING)")
    print("=" * 50)
    initial_time = time.time()

    initial_results = zero_shot_eval(
        model=net,
        train_loader=train_loader,
        test_loader=(test_loader if args.eval_on_test else val_loader),
        num_classes=num_classes,
        device=device,
        probe_lr=1e-3,
        probe_steps=10000,
    )
    print(f"Initial evaluation completed in {time.time() - initial_time:.2f}s")

    if run is not None:
        log_zero_shot_metrics(run, initial_results, 0)

    print(f"\nLoading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Loading only model, no need to load optimizer + DIET head
    net.load_state_dict(checkpoint["model_state_dict"])

    initial_time = time.time()

    final_results = zero_shot_eval(
        model=net,
        train_loader=train_loader,
        test_loader=(test_loader if args.eval_on_test else val_loader),
        num_classes=num_classes,
        device=device,
        probe_lr=1e-3,
        probe_steps=10000,
    )

    print(f"Final evaluation completed in {time.time() - initial_time:.2f}s")

    if run is not None:
        log_zero_shot_metrics(run, final_results, 1)

    # Create experiment dashboard in wandb
    if args.use_wandb and run is not None:
        create_experiment_dashboard(
            run, None, initial_results, final_results, experiment_config
        )

        # Finish the wandb run
        run.finish()

    return initial_results, final_results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DIET Finetuning Framework")

    # Model arguments
    parser.add_argument(
        "--backbone",
        type=str,
        default="simclr",
        choices=["resnet50", "dinov2", "dinov3", "mae", "mambavision", "ijepa", "aim"],
        help="Backbone model type",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="resnet50-1x",
        help="Model size (depends on backbone type). "
        "DINOv2/v3: small/base/large, MAE: base/large/huge",
    )

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Directory to store dataset files.",
    )
    parser.add_argument(
        "--limit-data",
        type=int,
        default=1000,
        help="Maximum number of training samples",
    )

    # Training arguments
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
        help="Weight decay",
    )
    parser.add_argument(
        "--da-strength",
        type=int,
        default=2,
        help="Data augmentation strength (0-3)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from a checkpoint file (e.g., checkpoint_epoch_25.pt)",
    )

    # DIET arguments
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.3,
        help="Label smoothing strength (0 to disable DIET)",
    )
    parser.add_argument(
        "--diet-head-only-epochs",
        type=float,
        default=0.05,
        help="Fraction of total epochs for DIET-head-only training (freezes backbone). "
        "Set to 0.0 for direct full training.",
    )
    parser.add_argument(
        "--num-trained-blocks",
        type=int,
        default=-1,
        help="Number of transformer blocks to train from the end of the backbone. "
        "Set to -1 to train all blocks, 0 to freeze all blocks, "
        "4 to train last 4 blocks.",
    )

    # Evaluation arguments
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=5,
        help="Run zero-shot evaluation every N epochs",
    )
    parser.add_argument(
        "--eval-on-test",
        action="store_true",
        help="Evaluate on test set instead of validation set",
    )

    # Logging and saving arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-wandb",
        dest="use_wandb",
        action="store_false",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--run-sanity-check",
        action="store_true",
        help="Run the initial k-NN sanity check on CIFAR10.",
    )
    parser.add_argument(
        "--wandb-dir",
        type=str,
        default="wandb",
        help="Directory to save wandb logs",
    )
    parser.add_argument(
        "--wandb-prefix",
        type=str,
        default="DIET",
        help="Prefix for wandb experiment names",
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Set reproducibility seeds
    set_reproducibility_seeds(args.seed)

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.wandb_dir, exist_ok=True)
    os.makedirs(args.data_root, exist_ok=True)

    # Run inference
    test(args)


if __name__ == "__main__":
    main()
