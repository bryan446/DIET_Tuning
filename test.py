#!/usr/bin/env python
"""
Inference for DIET finetuning experiments.
"""

# Standard library imports
import os
import argparse
import shutil
import time

# Third-party imports
import wandb

# Configuration imports
from config import create_experiment_config_from_args, DEVICE

# Data loading
from loaders.data_loader import prepare_data_loaders

# Utility modules
from utils.wandb_logger import (
    create_experiment_dashboard,
    log_inference_metrics_summary_table,
)
from utils.checkpoint_utils import download_final_checkpoint
from models.utils import set_reproducibility_seeds, get_model
from evaluation.metrics import zero_shot_eval


def test(args):
    """Main inference function"""
    print("\n" + "=" * 70)
    print("DIET FINETUNING INFERENCE (backbone and dataset from wandb config)")
    print("=" * 70)

    # Basic settings
    device = DEVICE
    print(f"Using device: {device}")

    # Download checkpoint from wandb if wandb_id is provided
    if args.wandb_id:
        print(f"\nDownloading checkpoint from wandb run: {args.wandb_id}")
        checkpoint_path, epoch, checkpoint, run_config = download_final_checkpoint(
            wandb_id=args.wandb_id,
            target_dir=args.checkpoint_dir,
            return_loaded=True,
            map_location=str(device),
            entity=args.wandb_entity,
            project=args.wandb_project,
        )
        print(f"Downloaded checkpoint from epoch {epoch}: {checkpoint_path}")
        if checkpoint is None:
            raise ValueError("Failed to load checkpoint from wandb")

        # Get required values from wandb config
        required_keys = ["dataset_name", "backbone_type", "model_size"]
        for key in required_keys:
            if key not in run_config:
                raise ValueError(f"{key} not found in wandb run config")

        dataset_name = run_config["dataset_name"]
        backbone_type = run_config["backbone_type"]
        model_size = run_config["model_size"]

        # Update args object with inferred values for config creation
        args.dataset = dataset_name
        args.backbone = backbone_type
        args.model_size = model_size

        print("Inferred from wandb config:")
        print(f"  - Dataset: {dataset_name}")
        print(f"  - Backbone: {backbone_type}")
        print(f"  - Model size: {model_size}")
    else:
        raise ValueError("--wandb-id is required for inference")

    # Load data
    print(f"\nLoading {dataset_name} dataset...")
    train_loader, val_loader, test_loader, dataset_info = prepare_data_loaders(
        dataset_name=dataset_name,
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
        backbone_type, model_size, args.run_sanity_check, args.use_wandb
    )
    print(f"Created {backbone_type} backbone with size {model_size}")
    print(f"Embedding dimension: {embedding_dim}")

    # Ensure embedding_dim is valid
    if embedding_dim is None:
        raise ValueError("embedding_dim cannot be None")

    # Create experiment configuration
    config = create_experiment_config_from_args(args)

    # Convert to wandb format for logging
    experiment_config = config.to_wandb_config()

    # Initialize wandb if enabled - resume the original training run
    run = None
    if args.use_wandb:
        # Resume the original wandb run for inference logging
        print(f"Resuming wandb run: {args.wandb_id}")
        run = wandb.init(
            project=args.wandb_project,
            id=args.wandb_id,
            resume="allow",  # Resume if exists, create if doesn't
            entity=args.wandb_entity,
            dir=args.wandb_dir,
            settings=wandb.Settings(start_method="thread"),
        )
        print(f"Resumed wandb run: {run.name}")

        # Log that we're starting inference
        run.log({"inference_started": True}, commit=False)

    # initial kNN and LP eval
    print("\n" + "=" * 50)
    print("INITIAL ZERO-SHOT EVALUATION (BEFORE DIET CONTINUED PRETRAINING)")
    print("=" * 50)
    initial_time = time.time()

    initial_results = zero_shot_eval(
        model=net,
        train_loader=train_loader,
        test_loader=(test_loader if args.eval_on_test else val_loader),
        device=device,
        probe_lr=1e-3,
        probe_steps=10000,
        store_embeddings=args.store_embeddings,
    )
    print(f"Initial evaluation completed in {time.time() - initial_time:.2f}s")

    # Loading only model, no need to load optimizer + DIET head
    net.load_state_dict(checkpoint["model_state_dict"])

    # Clean up downloaded checkpoint immediately after loading to save disk space
    try:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        # Remove the entire wandb_id subdirectory
        wandb_subdir = f"wandb_{args.wandb_id}"
        if os.path.exists(checkpoint_dir) and wandb_subdir in checkpoint_dir:
            shutil.rmtree(checkpoint_dir)
            print(f"Cleaned up checkpoint directory: {checkpoint_dir}")
    except Exception as e:
        print(f"Warning: Could not clean up checkpoint directory: {e}")

    initial_time = time.time()

    final_results = zero_shot_eval(
        model=net,
        train_loader=train_loader,
        test_loader=(test_loader if args.eval_on_test else val_loader),
        device=device,
        probe_lr=1e-3,
        probe_steps=10000,
        store_embeddings=args.store_embeddings,
    )

    print(f"Final evaluation completed in {time.time() - initial_time:.2f}s")

    # Log inference metrics summary table
    if run is not None:
        log_inference_metrics_summary_table(
            run=run,
            wandb_id=args.wandb_id,
            backbone_type=backbone_type,
            model_size=model_size,
            dataset=dataset_name,
            initial_metrics=initial_results,
            final_metrics=final_results,
        )

    # Create experiment dashboard in wandb
    if args.use_wandb and run is not None:
        create_experiment_dashboard(
            run, None, initial_results, final_results, experiment_config
        )

        # Log that inference is complete
        run.log({"inference_completed": True})

        # Finish the wandb run
        run.finish()
        print("Inference results logged to the original wandb run")

    return initial_results, final_results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DIET Finetuning Framework")

    # Model arguments - now inferred from wandb config
    # (backbone and model-size are extracted from the checkpoint's run config)

    # Dataset arguments
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
        help="Maximum number of training samples for kNN/LP training set",
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
        "--wandb-id",
        type=str,
        default=None,
        help="Wandb run ID to download the final checkpoint from",
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
        "--store-embeddings",
        action="store_true",
        help="Whether to store embeddings from the model",
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
        default="DIET_INFERENCE",
        help="Prefix for wandb experiment names",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="jakobamb",
        help="Wandb entity (username/team) for checkpoint download",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="DIET-Finetuning_v4",
        help="Wandb project name for checkpoint download",
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
