"""Training components for DIET finetuning framework."""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import Optional
from utils.wandb_logger import (
    log_training_metrics,
    log_evaluation_metrics,
    log_zero_shot_metrics,
    save_final_checkpoint,
    log_inference_metrics_summary_table,
)
from evaluation.metrics import zero_shot_eval
from training.metrics_utils import (
    MetricsTracker,
    aggregate_metrics,
)
from config.experiment import ExperimentConfig


class DIETTrainer:
    """Trainer class for DIET finetuning."""

    def __init__(
        self,
        model,
        diet_head,
        device,
        optimizer,
        config: ExperimentConfig,
        scheduler=None,
    ):
        """Initialize the trainer.

        Args:
            model: The backbone model
            diet_head: The DIET linear layer
            device: The device to use
            optimizer: The optimizer
            scheduler: Optional learning rate scheduler
            config: ExperimentConfig object with all training settings
        """
        self.model = model
        self.diet_head = diet_head
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        self.num_diet_classes = config.data.num_diet_classes
        self.label_smoothing = config.training.label_smoothing
        self.checkpoint_dir = config.checkpoint_dir
        self.is_diet_active = config.training.label_smoothing > 0

        self.criterion_diet = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        # Initialize mixup/cutmix transforms
        self._setup_mixup_cutmix()

        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker()

        # For backward compatibility
        self.metrics_history = {
            "train_loss_diet": [],
            "train_loss_probe": [],
            "train_acc": [],
            "test_acc": [],
            "zero_shot_metrics": {},
        }

    def _setup_mixup_cutmix(self):
        """Initialize mixup/cutmix transforms based on configuration."""
        from torchvision.transforms import v2

        # Check if mixup/cutmix should be enabled
        mixup_enabled = self.config.training.mixup_alpha > 0
        cutmix_enabled = self.config.training.cutmix_alpha > 0

        if not (mixup_enabled or cutmix_enabled):
            self.mixup_cutmix_transform = None
            return

        # Create transform list
        transforms = []

        if mixup_enabled:
            transforms.append(
                v2.MixUp(
                    num_classes=self.num_diet_classes,
                    alpha=self.config.training.mixup_alpha,
                )
            )

        if cutmix_enabled:
            transforms.append(
                v2.CutMix(
                    num_classes=self.num_diet_classes,
                    alpha=self.config.training.cutmix_alpha,
                )
            )

        # Create final transform
        if len(transforms) == 1:
            base_transform = transforms[0]
        else:
            # Both transforms active - use RandomChoice with switch probability
            base_transform = v2.RandomChoice(
                transforms,
                p=[
                    1 - self.config.training.mixup_cutmix_switch_prob,
                    self.config.training.mixup_cutmix_switch_prob,
                ],
            )

        # Wrap with probability control
        if self.config.training.mixup_cutmix_prob < 1.0:
            self.mixup_cutmix_transform = v2.RandomApply(
                [base_transform], p=self.config.training.mixup_cutmix_prob
            )
        else:
            self.mixup_cutmix_transform = base_transform

    def _freeze_backbone(self):
        """Freeze backbone model for DIET-only training."""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        # Ensure all BatchNorm layers are in eval mode to prevent running stats updates
        for module in self.model.modules():
            if isinstance(
                module,
                (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
            ):
                module.eval()

    def _unfreeze_backbone(self):
        """Unfreeze backbone model for full training."""
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

    def _apply_selective_block_freezing(self):
        """Apply selective block freezing based on num_trained_blocks config."""
        num_trained_blocks = self.config.training.num_trained_blocks

        if num_trained_blocks == -1:
            # Train all blocks - just unfreeze everything
            self._unfreeze_backbone()
            return

        # Set model to train mode
        self.model.train()

        encoder = None
        layers = None
        model_type = None

        # Check for DINOv2 structure (encoder.layer)
        if hasattr(self.model, "model") and hasattr(self.model.model, "encoder"):
            encoder = self.model.model.encoder
            if hasattr(encoder, "layer"):
                layers = encoder.layer
                model_type = "DINOv2"
        # Check for DINOv3 structure (direct layer attribute)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layer"):
            layers = self.model.model.layer
            model_type = "DINOv3"
        # Check for MAE structure
        elif (
            hasattr(self.model, "model")
            and hasattr(self.model.model, "vit")
            and hasattr(self.model.model.vit, "encoder")
        ):
            encoder = self.model.model.vit.encoder
            if hasattr(encoder, "layer"):
                layers = encoder.layer
                model_type = "MAE"

        if layers is not None:
            total_blocks = len(layers)
            print(f"{model_type} model has {total_blocks} transformer blocks")

            if num_trained_blocks == 0:
                # Freeze all blocks
                for param in self.model.parameters():
                    param.requires_grad = False
                print("Frozen all transformer blocks")
            else:
                # Freeze all parameters first
                for param in self.model.parameters():
                    param.requires_grad = False

                # Unfreeze the last num_trained_blocks
                blocks_to_train = min(num_trained_blocks, total_blocks)
                start_idx = total_blocks - blocks_to_train

                for i in range(start_idx, total_blocks):
                    for param in layers[i].parameters():
                        param.requires_grad = True

                print(
                    f"Training last {blocks_to_train} blocks "
                    f"(blocks {start_idx} to {total_blocks - 1})"
                )
        else:
            # Add diagnostic info to help debug model structure
            print(f"Model structure diagnostic:")
            print(f"  hasattr(self.model, 'model'): {hasattr(self.model, 'model')}")
            if hasattr(self.model, "model"):
                print(
                    f"  hasattr(self.model.model, 'encoder'): {hasattr(self.model.model, 'encoder')}"
                )
                print(
                    f"  hasattr(self.model.model, 'vit'): {hasattr(self.model.model, 'vit')}"
                )
                if hasattr(self.model.model, "vit"):
                    print(
                        f"  hasattr(self.model.model.vit, 'encoder'): {hasattr(self.model.model.vit, 'encoder')}"
                    )
            print(f"  Model type: {type(self.model)}")
            if hasattr(self.model, "model"):
                print(f"  Inner model type: {type(self.model.model)}")

            raise NotImplementedError(
                f"Block freezing (num_trained_blocks={num_trained_blocks}) is not "
                f"implemented for this model architecture. "
                f"Supported models: DINOv2, DINOv3, MAE. "
                f"Use num_trained_blocks=-1 for full backbone training."
            )

    def train(
        self,
        train_loader,
        test_loader,
        num_epochs,
        run=None,
        eval_frequency=5,
        start_epoch=0,
    ):
        """Train the model.

        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            num_epochs: Number of epochs to train
            run: Optional wandb run object
            eval_frequency: How often to run zero-shot evaluation
            start_epoch: Epoch to start training from (for resuming training)

        Returns:
            metrics_history: Dictionary of training metrics
        """
        # Log data augmentation settings
        if self.mixup_cutmix_transform is not None:
            print(
                f"\nData augmentation: mixup (α={self.config.training.mixup_alpha}) "
                f"+ cutmix (α={self.config.training.cutmix_alpha})"
            )
            print(f"Augmentation probability: {self.config.training.mixup_cutmix_prob}")
            switch_prob = self.config.training.mixup_cutmix_switch_prob
            print(f"Cutmix selection probability: {switch_prob}")
        else:
            print("\nData augmentation: standard only (no mixup/cutmix)")

        # Run initial zero-shot evaluation
        print("\n" + "=" * 50)
        print("INITIAL ZERO-SHOT EVALUATION (BEFORE TRAINING)")
        print("=" * 50)
        initial_time = time.time()
        initial_results = zero_shot_eval(
            model=self.model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=self.device,
            probe_lr=1e-3,
            probe_steps=10000,
        )
        print(f"Initial evaluation completed in {time.time() - initial_time:.2f}s")

        # Store initial results
        self.metrics_history["zero_shot_metrics"][0] = initial_results.copy()

        # Log initial results to wandb
        if run is not None:
            log_zero_shot_metrics(run, initial_results, 0)

        # Begin training
        train_start_time = time.time()
        epoch_times = []

        # Count trainable parameters
        num_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        num_trainable_params += sum(
            p.numel() for p in self.diet_head.parameters() if p.requires_grad
        )

        # Determine DIET-head-only phase boundaries
        diet_head_only_epochs = int(
            num_epochs * self.config.training.diet_head_only_epochs
        )
        diet_only_end_epoch = start_epoch + diet_head_only_epochs
        backbone_frozen = False

        # Training loop
        for epoch in range(start_epoch, start_epoch + num_epochs):
            # Start training epoch
            epoch_start = time.time()
            batch_metrics_list = []

            # Handle phase transitions
            diet_only_phase = epoch < diet_only_end_epoch

            if diet_only_phase and not backbone_frozen:
                # Freeze backbone for DIET-only training (one-time operation)
                self._freeze_backbone()
                backbone_frozen = True
                print(
                    f"Epoch {epoch + 1}: Starting DIET-head-only phase "
                    f"(backbone frozen for {diet_head_only_epochs} epochs)"
                )
            elif not diet_only_phase and backbone_frozen:
                # Apply selective block freezing for full training phase
                self._apply_selective_block_freezing()
                backbone_frozen = False
                num_blocks = self.config.training.num_trained_blocks
                if num_blocks == -1:
                    print(
                        f"Epoch {epoch + 1}: Starting full training phase "
                        "(all backbone blocks unfrozen)"
                    )
                elif num_blocks == 0:
                    print(
                        f"Epoch {epoch + 1}: Starting training phase "
                        "(all backbone blocks frozen)"
                    )
                else:
                    print(
                        f"Epoch {epoch + 1}: Starting training phase "
                        f"(training last {num_blocks} backbone blocks)"
                    )

            # Set trainable components to training mode
            self.diet_head.train()
            # Ensure backbone stays in eval mode if frozen
            if backbone_frozen:
                self.model.eval()
                # Extra safety: ensure all BatchNorm layers stay in eval mode
                for module in self.model.modules():
                    if isinstance(
                        module,
                        (
                            torch.nn.BatchNorm1d,
                            torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d,
                        ),
                    ):
                        module.eval()

            print("\n==========================")
            print(
                f"Starting epoch {epoch + 1}/{start_epoch + num_epochs} "
                f"at {time.strftime('%H:%M:%S')}"
            )
            print("==========================\n")
            print("Initializing training loop...")
            print("\n")

            # Iterate through training data (without tqdm)
            for i, batch in enumerate(train_loader):
                assert len(batch) == 3, (
                    f"Batch needs to contain data, label, diet_class, found {batch}"
                )
                x, y, diet_idx = batch

                assert diet_idx is not None, (
                    "DIET class indices are required for training"
                )

                # Send tensors to device
                x = x.to(self.device)
                y = y.to(self.device)
                diet_idx = diet_idx.to(self.device)

                # Ensure y is 1D (flatten if needed)
                if y.dim() > 1:
                    y = y.view(-1)

                if diet_idx.dim() > 1:
                    diet_idx = diet_idx.view(-1)

                # Apply mixup/cutmix if enabled
                if self.mixup_cutmix_transform is not None:
                    x, diet_idx = self.mixup_cutmix_transform(x, diet_idx)

                # Forward pass
                z = self.model(x)  # Original features
                z_norm = F.normalize(z, p=2, dim=1)  # L2 normalize

                # Calculate diet loss
                logits_diet = self.diet_head(z_norm)
                loss_diet = self.criterion_diet(logits_diet, diet_idx)

                loss = loss_diet

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()

                # Calculate gradient norm (only for parameters being trained)
                total_grad_norm = 0.0
                trainable_params = []

                if backbone_frozen:
                    # Only include DIET head during DIET-only phase
                    trainable_params = list(self.diet_head.parameters())
                else:
                    # Include backbone and DIET head when training everything
                    trainable_params = list(self.model.parameters()) + list(
                        self.diet_head.parameters()
                    )

                for p in trainable_params:
                    if p.grad is not None and p.requires_grad:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm**0.5

                # Update parameters
                self.optimizer.step()

                # Track batch metrics
                batch_metrics = {
                    "batch_loss_diet": loss_diet.item(),
                    "batch_grad_norm": total_grad_norm,
                }

                batch_metrics_list.append(batch_metrics)

                # Print batch update every 10 batches
                if i % 10 == 0:
                    # Count currently trainable parameters
                    current_trainable = sum(
                        p.numel() for p in trainable_params if p.requires_grad
                    )
                    phase_description = "DIET-head-only" if backbone_frozen else "full"
                    print(
                        f"Batch {i}: grad_norm={total_grad_norm:.4f} "
                        f"({phase_description} training, {current_trainable} params)"
                    )

            # End of epoch processing
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            print(f"\nEpoch {epoch + 1} completed in {epoch_time:.2f}s\n")

            # Aggregate metrics for the epoch
            epoch_metrics = aggregate_metrics(
                batch_metrics_list,
                prefix="train_",
            )

            # Update metrics tracker
            self.metrics_tracker.update(epoch_metrics)

            # For backward compatibility
            self.metrics_history["train_loss_diet"].append(
                epoch_metrics.get("train_batch_loss_diet", float("nan"))
            )

            # Set unused metrics to NaN
            self.metrics_history["train_loss_probe"].append(float("nan"))
            self.metrics_history["train_acc"].append(float("nan"))

            # Step the learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"Learning rate updated to: {current_lr:.6f}")
            else:
                current_lr = None

            # Log training metrics to wandb
            if run is not None:
                # Create a clean dict for wandb (DIET loss only)
                wandb_metrics = {
                    "diet_loss": epoch_metrics.get("train_batch_loss_diet", 0)
                }

                log_training_metrics(run, wandb_metrics, epoch + 1, current_lr)

            # Print epoch summary
            print(
                f"Epoch {epoch + 1} Metrics - DIET Loss: "
                f"{epoch_metrics.get('train_batch_loss_diet', float('nan')):.4e}"
            )

            # Simple test accuracy placeholder (not used in DIET evaluation)
            test_acc = 0.0  # Placeholder - actual evaluation done via zero-shot
            self.metrics_history["test_acc"].append(test_acc)

            # Log evaluation metrics to wandb
            if run is not None:
                log_evaluation_metrics(run, {"accuracy": test_acc}, epoch + 1)

            # Print epoch summary
            print(f"Epoch {epoch + 1}/{start_epoch + num_epochs} summary:")
            print(
                f"  Train - DIET loss: "
                f"{epoch_metrics.get('train_batch_loss_diet', float('nan')):.4e}"
            )
            print(f"  Test  - Acc: {test_acc:.4f}")

            # Zero-shot evaluation every few epochs
            if (
                epoch + 1
            ) % eval_frequency == 0 or epoch == start_epoch + num_epochs - 1:
                print(f"\nRunning zero-shot evaluation at epoch {epoch + 1}...")

                try:
                    epoch_zero_shot = zero_shot_eval(
                        model=self.model,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        device=self.device,
                        probe_lr=1e-3,
                        probe_steps=10000,
                    )
                    self.metrics_history["zero_shot_metrics"][epoch + 1] = (
                        epoch_zero_shot.copy()
                    )

                    # Log zero-shot metrics to wandb
                    if run is not None:
                        log_zero_shot_metrics(
                            run, epoch_zero_shot, epoch + 1, initial_results
                        )

                except (RuntimeError, ValueError) as e:
                    print(f"Error in zero-shot evaluation: {e}")

            # Note: Checkpoint saving moved to end of training (final checkpoint only)

        # End of training
        training_time = time.time() - train_start_time
        print(f"\nTraining completed in {training_time:.2f}s")

        # Final zero-shot evaluation
        print("\n" + "=" * 50)
        print("FINAL ZERO-SHOT EVALUATION (AFTER TRAINING)")
        print("=" * 50)
        final_time = time.time()
        final_results = zero_shot_eval(
            model=self.model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=self.device,
            probe_lr=1e-3,
            probe_steps=10000,
        )
        print(f"Final evaluation completed in {time.time() - final_time:.2f}s")

        # Log final zero-shot metrics to wandb
        if run is not None:
            log_zero_shot_metrics(
                run, final_results, start_epoch + num_epochs + 1, initial_results
            )

        # Calculate improvements
        improvements = {
            f"improvement_{k}": final_results[k] - initial_results[k]
            for k in initial_results.keys()
        }

        # Print results summary
        print("\n" + "=" * 50)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 50)
        print(f"Number of classes: {self.num_classes}")
        print(
            f"DIET label smoothing: {self.criterion_diet.label_smoothing}"
            + (" (DIET active)" if self.is_diet_active else " (DIET inactive)")
        )
        print("\nZero-shot performance:")
        print("-" * 60)

        print(
            f"{'Metric':<15} {'Initial':<10} {'Final':<10} {'Improvement':<10} {'Relative %':<10}"
        )
        print("-" * 60)
        metrics = list(initial_results.keys())
        for metric in metrics:
            initial = initial_results[metric]
            final = final_results[metric]
            imp = improvements[f"improvement_{metric}"]
            rel_imp = (imp / initial) * 100 if initial > 0 else float("inf")
            print(
                f"{metric:<15} {initial:.4f}     {final:.4f}     {imp:+.4f}     {rel_imp:+.2f}%"
            )

        print("\nCONCLUSION:")
        avg_improvement = np.mean(
            [improvements[f"improvement_{k}"] for k in initial_results.keys()]
        )
        if avg_improvement > 0:
            print(
                f"DIET finetuning {'improved' if self.is_diet_active else 'would likely improve'} zero-shot performance "
                + f"by an average of {avg_improvement:.4f} ({(avg_improvement / np.mean(list(initial_results.values()))) * 100:.2f}%)"
            )
        else:
            print(
                f"DIET finetuning {'did not improve' if self.is_diet_active else 'would likely not improve'} zero-shot performance"
            )

        # Save final checkpoint and summary table
        if run is not None:
            log_inference_metrics_summary_table(
                run,
                run.id,
                self.config.model.backbone_type,
                self.config.model.model_size,
                self.config.data.dataset_name,
                initial_results,
                final_results,
                table_name="inference_metrics_val",
            )
            final_checkpoint_metrics = {
                "final_knn_acc": final_results.get("knn_acc", 0),
                "final_linear_acc": final_results.get("linear_acc", 0),
                "avg_improvement": avg_improvement,
            }
            save_final_checkpoint(
                run,
                self.model,
                self.optimizer,
                self.diet_head,
                start_epoch + num_epochs,
                final_checkpoint_metrics,
                save_dir=self.checkpoint_dir,
            )

        return self.metrics_history, initial_results, final_results

    def create_visualizations(self, tracked_epochs=None):
        """Create visualization plots for training metrics.

        Args:
            tracked_epochs: List of epochs to include in zero-shot plots

        Returns:
            Tuple of (training_fig, zero_shot_fig)
        """
        # Training progress plot
        training_fig = plt.figure(figsize=(15, 5))

        # Plot loss
        plt.subplot(1, 3, 1)
        plt.plot(self.metrics_history["train_loss_diet"], label="DIET Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.grid(True)

        # Plot accuracy
        plt.subplot(1, 3, 2)

        plt.plot(self.metrics_history["test_acc"], label="Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy")
        plt.legend()
        plt.grid(True)

        # Plot zero-shot metrics
        plt.subplot(1, 3, 3)

        # Get initial and final zero-shot metrics
        if tracked_epochs is None:
            tracked_epochs = sorted(self.metrics_history["zero_shot_metrics"].keys())

        if len(tracked_epochs) >= 2:
            initial_epoch = tracked_epochs[0]
            final_epoch = tracked_epochs[-1]

            metrics = list(
                self.metrics_history["zero_shot_metrics"][initial_epoch].keys()
            )
            x = range(len(metrics))
            width = 0.35

            # Plot bar chart comparing initial and final metrics
            plt.bar(
                x,
                [
                    self.metrics_history["zero_shot_metrics"][initial_epoch][m]
                    for m in metrics
                ],
                width,
                label="Initial",
            )
            plt.bar(
                [i + width for i in x],
                [
                    self.metrics_history["zero_shot_metrics"][final_epoch][m]
                    for m in metrics
                ],
                width,
                label="Final",
            )
            plt.xlabel("Metrics")
            plt.ylabel("Score")
            plt.title("Zero-Shot Performance")
            plt.xticks([i + width / 2 for i in x], metrics)
            plt.legend()
            plt.grid(True)

        training_fig.tight_layout()

        # Zero-shot progression plot
        if len(tracked_epochs) >= 2:
            zero_shot_fig = plt.figure(figsize=(15, 10))

            metrics_list = list(
                self.metrics_history["zero_shot_metrics"][tracked_epochs[0]].keys()
            )

            # Plot each metric's progression
            for i, metric in enumerate(metrics_list):
                plt.subplot(2, 2, i + 1)
                values = [
                    self.metrics_history["zero_shot_metrics"][e][metric]
                    for e in tracked_epochs
                ]
                plt.plot(tracked_epochs, values, marker="o", linewidth=2)
                plt.xlabel("Epoch")
                plt.ylabel(f"{metric} Score")
                plt.title(f"Zero-shot {metric} Progression")
                plt.grid(True)

                # Add initial and final values as text annotations
                plt.annotate(
                    f"{values[0]:.4f}",
                    (tracked_epochs[0], values[0]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                )
                plt.annotate(
                    f"{values[-1]:.4f}",
                    (tracked_epochs[-1], values[-1]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                )

            zero_shot_fig.tight_layout()
            zero_shot_fig.suptitle(
                "Zero-shot Metrics Progression During Training", fontsize=16
            )
            zero_shot_fig.subplots_adjust(top=0.9)

            return training_fig, zero_shot_fig

        return training_fig, None


def create_projection_head(in_dim, hidden_dim, out_dim, device):
    """Create projection head for DIET method.

    Args:
        in_dim: Input dimension
        hidden_dim: Hidden dimension
        out_dim: Output dimension
        device: Device to put the projection head on

    Returns:
        Projection head module
    """
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden_dim),
        torch.nn.BatchNorm1d(hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(hidden_dim, out_dim),
    ).to(device)


# Add plot_metrics method to DIETTrainer class
def plot_training_metrics(self, tracked_epochs=None, figsize=(18, 6)):
    """Plot training metrics.

    Args:
        tracked_epochs: List of epochs for zero-shot metrics
        figsize: Figure size tuple

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Use the new plot_metrics utility if metrics_tracker has data
    if self.metrics_tracker.history:
        # Define which metrics to plot
        metrics_to_plot = []

        # Always include DIET loss
        metrics_to_plot.append("train_batch_loss_diet")

        # Include test accuracy and gradient norm
        metrics_to_plot.append("train_batch_grad_norm")

        # Get all metrics data
        metrics_history = self.metrics_tracker.get_all()

        # Add test accuracy from the old metrics_history
        metrics_history["test_acc"] = self.metrics_history["test_acc"]
        metrics_to_plot.append("test_acc")

        # Create plot using the utility function
        from training.metrics_utils import plot_metrics as plot_metrics_util

        return plot_metrics_util(
            metrics_history, metrics_to_plot=metrics_to_plot, figsize=figsize
        )

    # Fallback to legacy plotting
    fig = plt.figure(figsize=figsize)

    # Plot losses
    plt.subplot(1, 3, 1)
    plt.plot(self.metrics_history["train_loss_diet"], label="DIET Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    plt.subplot(1, 3, 2)

    plt.plot(self.metrics_history["test_acc"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    plt.legend()
    plt.grid(True)

    # Plot zero-shot metrics if available
    plt.subplot(1, 3, 3)

    if tracked_epochs is None:
        tracked_epochs = sorted(self.metrics_history["zero_shot_metrics"].keys())

    if len(tracked_epochs) >= 2:
        initial_epoch = tracked_epochs[0]
        final_epoch = tracked_epochs[-1]

        metrics = list(self.metrics_history["zero_shot_metrics"][initial_epoch].keys())
        x = range(len(metrics))
        width = 0.35

        # Plot bar chart comparing initial and final metrics
        plt.bar(
            x,
            [
                self.metrics_history["zero_shot_metrics"][initial_epoch][m]
                for m in metrics
            ],
            width,
            label=f"Epoch {initial_epoch}",
        )
        plt.bar(
            [i + width for i in x],
            [
                self.metrics_history["zero_shot_metrics"][final_epoch][m]
                for m in metrics
            ],
            width,
            label=f"Epoch {final_epoch}",
        )

        plt.xlabel("Metrics")
        plt.ylabel("Score")
        plt.title("Zero-Shot Evaluation")
        plt.xticks([i + width / 2 for i in x], metrics, rotation=45)
        plt.legend()
    else:
        plt.title("Zero-Shot Evaluation\n(Not enough data)")

    plt.tight_layout()
    return fig
