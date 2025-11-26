#!/usr/bin/env python
"""
Main entry point for running DIET finetuning experiments.
"""

# Standard library imports
import os
import argparse

# Third-party imports
import torch

# Configuration imports
from config import create_experiment_config_from_args, DEVICE

# Data loading
from loaders.data_loader import prepare_data_loaders

# Training components
from training.trainer import DIETTrainer

# Utility modules
from utils.wandb_logger import (
    init_wandb,
    create_experiment_dashboard,
    log_model_architecture,
)
from utils.lr_schedule import create_warmup_cosine_scheduler, get_scheduler_info
from models.utils import set_reproducibility_seeds, get_model


def train(args):
    """Main training function"""
    print("\n" + "=" * 70)
    backbone_info = f"{args.backbone.upper()}-{args.model_size}"
    print(f"DIET FINETUNING EXPERIMENT: {backbone_info} on {args.dataset}")
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
        limit_data=args.limit_data,
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

    # Create DIET classification head
    diet_head = torch.nn.Linear(embedding_dim, num_diet_classes, bias=False).to(device)

    # Create optimizer
    print(f"Creating optimizer with lr={args.lr}, weight_decay={args.weight_decay}")
    optimizer = torch.optim.AdamW(
        list(net.parameters()) + list(diet_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Add learning rate scheduler with warmup
    scheduler_info = get_scheduler_info(args.num_epochs, warmup_ratio=0.1)
    print("Creating learning rate scheduler (warmup + cosine annealing)")
    print(
        f"Warmup epochs: {scheduler_info['warmup_epochs']}, "
        f"Total epochs: {scheduler_info['total_epochs']}"
    )

    scheduler = create_warmup_cosine_scheduler(
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        base_lr=args.lr,
        warmup_ratio=0.1,
        eta_min=1e-5,
    )

    # Load from checkpoint if specified
    start_epoch = 0
    if args.resume_from:
        checkpoint_path = (
            os.path.join(args.checkpoint_dir, args.resume_from)
            if not os.path.isabs(args.resume_from)
            else args.resume_from
        )
        if os.path.exists(checkpoint_path):
            print(f"\nLoading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )

            # Load model and optimizer states
            net.load_state_dict(checkpoint["model_state_dict"])
            diet_head.load_state_dict(checkpoint["W_diet_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Get the starting epoch
            start_epoch = checkpoint["epoch"]
            print(f"Resuming from epoch {start_epoch}")

            # Adjust scheduler to the correct epoch
            for _ in range(start_epoch):
                scheduler.step()

            print(
                f"Successfully loaded checkpoint. Current learning rate: "
                f"{scheduler.get_last_lr()[0]:.6f}"
            )
        else:
            print(
                f"Warning: Checkpoint file {checkpoint_path} not found. "
                f"Starting from scratch."
            )

    # Create experiment configuration
    config = create_experiment_config_from_args(args)
    config.data.num_diet_classes = (
        num_diet_classes  # Update diet classes based on dataset
    )

    # Convert to wandb format for logging
    experiment_config = config.to_wandb_config()

    # Add runtime fields
    experiment_config["start_epoch"] = start_epoch

    # Initialize wandb if enabled
    run = None
    if args.use_wandb:
        run = init_wandb(experiment_config)
        log_model_architecture(run, net, diet_head)

    # Create trainer
    trainer = DIETTrainer(
        model=net,
        diet_head=diet_head,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )

    # Choose evaluation loader based on parameter
    eval_loader = test_loader if args.eval_on_test else val_loader
    eval_set_name = "test" if args.eval_on_test else "validation"
    print(f"Using {eval_set_name} set for evaluation")

    # Run the training
    metrics_history, initial_results, final_results = trainer.train(
        train_loader=train_loader,
        test_loader=eval_loader,
        num_epochs=args.num_epochs,
        run=run,
        eval_frequency=args.eval_frequency,
        start_epoch=start_epoch,
    )

    # Create experiment dashboard in wandb
    if args.use_wandb and run is not None:
        create_experiment_dashboard(
            run, metrics_history, initial_results, final_results, experiment_config
        )

        # Finish the wandb run
        run.finish()

    return metrics_history, initial_results, final_results


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

    # Mixup/CutMix augmentation arguments
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=1.0,
        help="Mixup alpha parameter (0.0 to disable mixup)",
    )
    parser.add_argument(
        "--cutmix-alpha",
        type=float,
        default=1.0,
        help="CutMix alpha parameter (0.0 to disable cutmix)",
    )
    parser.add_argument(
        "--mixup-cutmix-prob",
        type=float,
        default=0.0,
        help="Probability of applying mixup/cutmix augmentation",
    )
    parser.add_argument(
        "--mixup-cutmix-switch-prob",
        type=float,
        default=0.0,
        help="Probability of selecting cutmix over mixup when both are active",
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
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="DIET-Finetuning_v4",
        help="Wandb project name",
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

    # Run the training
    train(args)


if __name__ == "__main__":
    main()
