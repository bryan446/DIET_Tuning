"""Logging utilities for DIET finetuning using Weights & Biases."""

import os
import wandb
import torch
import numpy as np
from datetime import datetime
from io import BytesIO

# Import from our metrics module
from evaluation.metrics import (
    create_training_progress_plot,
)


def init_wandb(wandb_cfg):
    """Initialize wandb for experiment tracking

    Args:
        args: Dictionary containing experiment configuration parameters

    Returns:
        run: wandb run object
    """
    # Set wandb data directory to prevent pollution of home directory
    # This controls where artifacts staging and cache are stored
    wandb_dir = wandb_cfg.get("wandb_dir", "wandb")
    os.environ["WANDB_DATA_DIR"] = wandb_dir

    # Create experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_prefix = wandb_cfg.get("wandb_prefix", "DIET")
    experiment_name = (
        f"{wandb_prefix}_{wandb_cfg['backbone_type']}_"
        f"{wandb_cfg['model_size']}_{wandb_cfg['dataset_name']}_{timestamp}"
    )

    # Initialize wandb run
    run = wandb.init(
        project=wandb_cfg["wandb_project"],
        name=experiment_name,
        config=wandb_cfg,
        settings=wandb.Settings(start_method="thread"),
        dir=wandb_dir,
        tags=[
            wandb_cfg["backbone_type"],
            wandb_cfg["model_size"],
            wandb_cfg["dataset_name"],
            "DIET" if wandb_cfg["label_smoothing"] > 0 else "Baseline",
        ],
    )

    print(f"WandB initialized: {run.name}")
    return run


def log_training_metrics(run, metrics_dict, epoch, lr=None):
    """
    Log training metrics to wandb

    Args:
        run: The wandb run object
        metrics_dict: Dictionary containing training metrics
        epoch: Current epoch number
        lr: Current learning rate (optional)
    """
    log_dict = {"epoch": epoch}

    # Add all metrics from the dictionary
    for key, value in metrics_dict.items():
        if key == "diet_loss":
            log_dict["train/diet_loss"] = value
        elif key == "diet_acc":
            log_dict["train/diet_accuracy"] = value
        else:
            log_dict[f"train/{key}"] = value

    # Log learning rate if provided
    if lr is not None:
        log_dict["train/learning_rate"] = lr

    run.log(log_dict, step=epoch)


def log_evaluation_metrics(run, metrics, epoch):
    """Log evaluation metrics to wandb

    Args:
        run: wandb run object
        metrics: Dictionary of evaluation metrics
        epoch: Current epoch number
    """
    log_dict = {"eval/accuracy": metrics["accuracy"], "epoch": epoch}

    # Log metrics to wandb
    run.log(log_dict, step=epoch)


def log_zero_shot_metrics(
    run, metrics, epoch=None, initial_metrics=None, is_inference=False
):
    """Log zero-shot evaluation metrics to wandb

    Args:
        run: wandb run object
        metrics: Dictionary of zero-shot metrics
        epoch: Current epoch number (optional, uses current step if None)
        initial_metrics: Initial zero-shot metrics for comparison (optional)
        is_inference: Whether this is inference logging
    """
    log_dict = {}

    # Only include epoch in log_dict if provided
    if epoch is not None:
        log_dict["epoch"] = epoch

    if is_inference:
        log_key = "zero_shot_inference"
    else:
        log_key = "zero_shot"

    # Log each zero-shot metric
    for metric_name, value in metrics.items():
        log_dict[f"{log_key}/{metric_name}"] = value

        # Log improvements if initial metrics are provided
        if initial_metrics is not None:
            improvement = value - initial_metrics[metric_name]
            relative_improvement = (
                (improvement / initial_metrics[metric_name]) * 100
                if initial_metrics[metric_name] > 0
                else float("inf")
            )
            log_dict[f"{log_key}/{metric_name}_improvement"] = improvement
            log_dict[f"{log_key}/{metric_name}_relative_improvement"] = (
                relative_improvement
            )

    # Calculate average improvement if initial metrics are provided
    if initial_metrics is not None:
        improvements = [metrics[m] - initial_metrics[m] for m in metrics.keys()]
        avg_improvement = np.mean(improvements)
        log_dict[f"{log_key}/average_improvement"] = avg_improvement

    # Log metrics to wandb
    if epoch is not None:
        run.log(log_dict, step=epoch)
    else:
        run.log(log_dict)  # Let wandb use the current step


def log_inference_metrics_summary_table(
    run,
    wandb_id,
    backbone_type,
    model_size,
    dataset,
    initial_metrics,
    final_metrics,
    table_name="inference_metrics_summary_final",
):
    """Log inference metrics as a summary table to wandb

    Args:
        run: wandb run object
        wandb_id: The wandb run ID
        backbone_type: Model backbone type (e.g., dinov2, mae)
        model_size: Model size (e.g., small, base, large)
        dataset: Dataset name
        initial_metrics: Dictionary of initial zero-shot metrics
        final_metrics: Dictionary of final zero-shot metrics
    """
    # Get all metric names from the dictionaries
    metric_names = list(initial_metrics.keys())

    # Create column headers
    columns = ["wandb_id", "backbone_type", "model_size", "dataset"]

    # Add columns for initial metrics
    for metric in metric_names:
        columns.append(f"initial_{metric}")

    # Add columns for final metrics
    for metric in metric_names:
        columns.append(f"final_{metric}")

    # Create the table
    inference_table = wandb.Table(columns=columns)

    # Create the row data
    row_data = [wandb_id, backbone_type, model_size, dataset]

    # Add initial metric values
    for metric in metric_names:
        row_data.append(initial_metrics[metric])

    # Add final metric values
    for metric in metric_names:
        row_data.append(final_metrics[metric])

    # Add the row to the table
    inference_table.add_data(*row_data)

    # Log the table to wandb
    run.log({table_name: inference_table})


def save_model_checkpoint(
    run,
    model,
    optimizer,
    W_diet,
    epoch,
    metrics,
    save_dir="checkpoints",
):
    """Save model checkpoint and log it to wandb

    Args:
        run: wandb run object
        model: The backbone model
        optimizer: Optimizer
        W_diet: DIET linear layer
        epoch: Current epoch number
        metrics: Metrics to determine if this is a best checkpoint
        save_dir: Directory to save checkpoints locally
    """

    # Create checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "W_diet_state_dict": W_diet.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    # Save checkpoint locally
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)

    # Log checkpoint to wandb (frequency is controlled by the caller)
    artifact = wandb.Artifact(
        name=f"model_e{epoch}_{run.id}",
        type="model",
        description=f"Model checkpoint from epoch {epoch}",
    )
    artifact.add_file(checkpoint_path)
    run.log_artifact(artifact)


def save_final_checkpoint(
    run,
    model,
    optimizer,
    W_diet,
    epoch,
    metrics,
    save_dir="checkpoints",
):
    """Save final model checkpoint and log it to wandb

    This function saves only the final checkpoint at the end of training,
    replacing the need for periodic checkpoint saving during training.

    Args:
        run: wandb run object
        model: The backbone model
        optimizer: Optimizer
        W_diet: DIET linear layer
        epoch: Final epoch number
        metrics: Final metrics
        save_dir: Directory to save checkpoints locally
    """

    # Create checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "W_diet_state_dict": W_diet.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    # Save checkpoint locally
    checkpoint_path = os.path.join(save_dir, f"final_checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)

    # Log final checkpoint to wandb
    artifact = wandb.Artifact(
        name=f"final_model_{run.id}",
        type="model",
        description=f"Final model checkpoint from epoch {epoch}",
    )
    artifact.add_file(checkpoint_path)
    run.log_artifact(artifact)

    print(f"Final checkpoint saved and logged to wandb: {checkpoint_path}")


def log_model_architecture(run, model, W_diet):
    """Log model architecture details to wandb

    Args:
        run: wandb run object
        model: The backbone model
        W_diet: DIET linear layer
    """
    # Log architecture as a text table
    architecture_text = "# Model Architecture\n\n"

    # Count parameters
    model_params = sum(p.numel() for p in model.parameters())
    trainable_model_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    W_diet_params = sum(p.numel() for p in W_diet.parameters())
    total_params = model_params + W_diet_params
    total_trainable = trainable_model_params + W_diet_params

    # Add parameter counts
    architecture_text += "## Parameter Counts\n\n"
    architecture_text += (
        "| Component | Total Parameters | Trainable Parameters | % of Total |\n"
    )
    architecture_text += (
        "|-----------|-----------------|----------------------|------------|\n"
    )
    architecture_text += (
        f"| Backbone Model | {model_params:,} | {trainable_model_params:,} | "
        f"{100 * model_params / total_params:.2f}% |\n"
    )
    architecture_text += (
        f"| DIET Head | {W_diet_params:,} | {W_diet_params:,} | "
        f"{100 * W_diet_params / total_params:.2f}% |\n"
    )
    architecture_text += (
        f"| **Total** | **{total_params:,}** | **{total_trainable:,}** | "
        f"**100%** |\n\n"
    )

    # Log the architecture text
    run.log({"model_architecture": wandb.Html(architecture_text)})


def log_figure_to_wandb(run, figure, name):
    """Convert matplotlib figure to wandb Image and log it

    Args:
        run: wandb run object
        figure: Matplotlib figure
        name: Name for the logged figure
    """
    # Save figure to a BytesIO object
    buf = BytesIO()
    figure.savefig(buf, format="png")
    buf.seek(0)

    # Convert BytesIO to PIL Image before logging to wandb
    try:
        from PIL import Image

        img = Image.open(buf)
        run.log({name: wandb.Image(img)})
    except ImportError:
        # If PIL is not available, just log a message and continue
        print(f"Warning: Could not log {name} to wandb - PIL is not installed")


def log_metrics_table(
    run, metrics_history, initial_results=None, final_results=None, enhanced=False
):
    """
    Log metrics tables to W&B with varying levels of detail.

    Args:
        run: The wandb run object
        metrics_history: Dictionary containing metrics history with zero_shot_metrics
        initial_results: Dictionary with initial metrics (optional)
        final_results: Dictionary with final metrics (optional)
        enhanced: Whether to create enhanced tables with trend indicators (optional)
    """
    # Get all tracked epochs and metrics
    tracked_epochs = sorted(metrics_history["zero_shot_metrics"].keys())
    if not tracked_epochs:
        print("No epochs to log in metrics history")
        return

    metrics_list = list(metrics_history["zero_shot_metrics"][tracked_epochs[0]].keys())

    # Create standard metrics table
    columns = ["epoch"]
    for metric in metrics_list:
        columns.extend([f"{metric}", f"{metric}_change", f"{metric}_relative_change"])

    zero_shot_table = wandb.Table(columns=columns)

    # Initial values for calculating changes
    initial_values = metrics_history["zero_shot_metrics"][tracked_epochs[0]]

    # Add data for each epoch
    for epoch in tracked_epochs:
        current_metrics = metrics_history["zero_shot_metrics"][epoch]

        # Create a row for this epoch
        row = [epoch]

        # Add metrics, absolute change and relative change for each metric
        for metric in metrics_list:
            current_value = current_metrics[metric]
            change = current_value - initial_values[metric]
            rel_change = (
                (change / initial_values[metric]) * 100
                if initial_values[metric] > 0
                else float("inf")
            )

            # Add to row: current value, absolute change, relative change
            row.extend([current_value, change, rel_change])

        # Add row to table
        zero_shot_table.add_data(*row)

    # Log the detailed table
    run.log({"zero_shot_metrics_table": zero_shot_table})

    # Create a more user-friendly table for viewing trends
    metric_names = {
        "knn_acc": "K-NN Accuracy",
        "knn_f1": "K-NN F1 Score",
        "knn_roc_auc": "K-NN ROC AUC",
        "linear_acc": "Linear Probe Accuracy",
        "linear_f1": "Linear Probe F1 Score",
        "linear_roc_auc": "Linear Probe ROC AUC",
        "kmeans_ari": "K-Means ARI",
        "kmeans_nmi": "K-Means NMI",
    }

    columns = ["Epoch"]
    for metric in metrics_list:
        # Convert technical metric names to human-readable display names
        if metric in metric_names:
            display_name = metric_names[metric]
        else:
            display_name = metric
        columns.append(display_name)

    friendly_table = wandb.Table(columns=columns)

    # Add a row for each epoch
    for epoch in tracked_epochs:
        row = [epoch]
        for metric in metrics_list:
            value = metrics_history["zero_shot_metrics"][epoch][metric]
            row.append(value)
        friendly_table.add_data(*row)

    # Log the user-friendly table
    run.log({"zero_shot_progression": friendly_table})

    # If initial and final results are provided, create summary tables
    if initial_results and final_results:
        # Basic summary table
        summary_columns = [
            "run",
            "metric",
            "initial",
            "final",
            "change",
            "relative_change",
        ]
        summary_table = wandb.Table(columns=summary_columns)

        for metric in metrics_list:
            initial = initial_results[metric]
            final = final_results[metric]
            change = final - initial
            rel_change = (change / initial) * 100 if initial > 0 else float("inf")

            summary_table.add_data(run.name, metric, initial, final, change, rel_change)

        # Log the summary table
        run.log({"zero_shot_summary_table": summary_table})

        # If enhanced mode is requested, create a more detailed summary
        if enhanced:
            summary_columns = [
                "metric",
                "initial",
                "final",
                "absolute_change",
                "percent_change",
                "trend",
                "significance",
            ]
            enhanced_table = wandb.Table(columns=summary_columns)

            # Add each metric to the summary
            for metric in metrics_list:
                initial = initial_results[metric]
                final = final_results[metric]
                change = final - initial
                percent = (change / initial) * 100 if initial > 0 else 0

                # Create a trend indicator
                trend = "↑" if change > 0 else "↓" if change < 0 else "→"

                # Significance level (arbitrary thresholds)
                if abs(percent) > 20:
                    significance = "High"
                elif abs(percent) > 5:
                    significance = "Medium"
                else:
                    significance = "Low"

                # Add to summary table
                enhanced_table.add_data(
                    metric, initial, final, change, percent, trend, significance
                )

            # Log the enhanced summary table
            run.log({"metrics_final_summary": enhanced_table})


def log_sanity_check_results(run, results_dict, model_type):
    """
    Log sanity check results to W&B tables

    Args:
        run: W&B run object
        results_dict: Dictionary of sanity check results
        model_type: Type of model that was evaluated
    """
    if results_dict is None:
        print(f"No results to log for {model_type}")
        return

    # Extract data
    k_values = results_dict["k_values"]
    accuracies = results_dict["accuracies"]
    best_acc = results_dict["best_acc"]
    best_k = results_dict["best_k"]

    # Create accuracy table by k value
    k_table = wandb.Table(columns=["model", "k_value", "accuracy"])

    # Add data for each k value
    for k, acc in zip(k_values, accuracies):
        k_table.add_data(model_type, k, acc * 100)  # Convert to percentage

    # Log the table
    run.log({f"sanity_check_{model_type}_knn": k_table})

    # Create summary table
    summary_columns = ["model", "best_k_value", "best_accuracy"]
    summary_table = wandb.Table(columns=summary_columns)
    summary_table.add_data(model_type, best_k, best_acc * 100)

    # Log the summary table
    run.log({f"sanity_check_{model_type}_summary": summary_table})


def log_combined_sanity_check_results(run, results_dict):
    """
    Log a combined table of all sanity check results

    Args:
        run: W&B run object
        results_dict: Dictionary mapping model types to their sanity check results
    """
    # Create table for combined results
    combined_table = wandb.Table(
        columns=["model", "method", "best_accuracy", "best_k", "passed_check"]
    )

    # Expected thresholds for each model type
    thresholds = {
        "dinov2": 0.91,
        "mae": 0.85,
        "mambavision": 0.85,
        "ijepa": 0.85,
        "aim": 0.75,
    }

    # Add data for each model
    for model_type, results in results_dict.items():
        if results is None:
            continue

        # Get model's threshold
        threshold = thresholds.get(model_type, 0.85)

        # Add k-NN result
        best_acc = results["best_acc"]
        best_k = results["best_k"]
        passed = best_acc >= threshold

        combined_table.add_data(
            model_type,
            "k-NN",
            best_acc * 100,  # Convert to percentage
            best_k,
            "✓" if passed else "✗",
        )

    # Log the combined table
    run.log({"sanity_check_combined_results": combined_table})


def create_experiment_dashboard(
    run, metrics_history, initial_results, final_results, experiment_config
):
    """
    Create a comprehensive dashboard visualization for the experiment

    Args:
        run: The wandb run object
        metrics_history: Dictionary with metrics history (can be None)
        initial_results: Dictionary with initial zero-shot metrics
        final_results: Dictionary with final zero-shot metrics
        experiment_config: Dictionary with experiment configuration
    """
    # Create main summary as HTML
    backbone_info = (
        f"{experiment_config['backbone_type']} " f"({experiment_config['model_size']})"
    )
    diet_status = "Active" if experiment_config["is_diet_active"] else "Inactive"
    training_samples = experiment_config.get("limit_data", "Full Dataset")

    label_smoothing_val = experiment_config["label_smoothing"]
    num_epochs_val = experiment_config["num_epochs"]
    learning_rate_val = experiment_config["learning_rate"]

    summary_text = f"""
    <h1>DIET Finetuning Experiment Dashboard</h1>
    <h2>Configuration</h2>
    <table>
        <tr><td><b>Model:</b></td><td>{backbone_info}</td></tr>
        <tr><td><b>Dataset:</b></td><td>{experiment_config['dataset_name']}</td></tr>
        <tr><td><b>DIET Status:</b></td><td>{diet_status}</td></tr>
        <tr><td><b>Label Smoothing:</b></td><td>{label_smoothing_val}</td></tr>
        <tr><td><b>Training Samples:</b></td><td>{training_samples}</td></tr>
        <tr><td><b>Training Epochs:</b></td><td>{num_epochs_val}</td></tr>
        <tr><td><b>Learning Rate:</b></td><td>{learning_rate_val}</td></tr>
    </table>
    
    <h2>Performance Summary</h2>
    <table>
        <tr>
            <th>Metric</th><th>Initial</th><th>Final</th>
            <th>Improvement</th><th>Relative %</th>
        </tr>
    """

    # Add metrics to summary
    metrics_list = list(initial_results.keys())
    for metric in metrics_list:
        initial = initial_results[metric]
        final = final_results[metric]
        change = final - initial
        rel_change = (change / initial) * 100 if initial > 0 else float("inf")

        # Add trend indicator
        trend = "⬆️" if change > 0 else "⬇️" if change < 0 else "↔️"

        summary_text += f"""
        <tr>
            <td>{metric}</td>
            <td>{initial:.4f}</td>
            <td>{final:.4f}</td>
            <td>{change:+.4f} {trend}</td>
            <td>{rel_change:+.2f}%</td>
        </tr>
        """

    # Calculate average improvement
    avg_improvement = np.mean(
        [final_results[m] - initial_results[m] for m in metrics_list]
    )
    avg_rel_improvement = (
        avg_improvement / np.mean(list(initial_results.values()))
    ) * 100

    # Close the table and add conclusion
    improvement_text = "improved" if avg_improvement > 0 else "did not improve"
    summary_text += f"""
    </table>
    
    <h2>Conclusion</h2>
    <p>
        DIET finetuning {improvement_text} zero-shot performance by an average of
        {avg_improvement:.4f} ({avg_rel_improvement:.2f}%).
    </p>
    """

    # Log this summary as HTML
    run.log({"experiment_dashboard": wandb.Html(summary_text)})

    # Only create and log plots if metrics_history is available
    if metrics_history is not None:
        # Create and log the combined figures
        # First create the training progress plot
        training_fig = create_training_progress_plot(metrics_history)
        log_figure_to_wandb(run, training_fig, "training_progress")

        # Log tables with full metrics history
        log_metrics_table(
            run, metrics_history, initial_results, final_results, enhanced=True
        )

        # Log individual components
        run.log({"training_progress_final": wandb.Image(training_fig)})
    else:
        # If no metrics history, just log a simple summary table
        summary_columns = [
            "metric",
            "initial",
            "final",
            "absolute_change",
            "percent_change",
        ]
        simple_table = wandb.Table(columns=summary_columns)

        for metric in metrics_list:
            initial = initial_results[metric]
            final = final_results[metric]
            change = final - initial
            percent = (change / initial) * 100 if initial > 0 else 0

            simple_table.add_data(metric, initial, final, change, percent)

        run.log({"zero_shot_summary_simple": simple_table})

    # Always log the experiment summary
    run.log({"experiment_summary": wandb.Html(summary_text)})

    return
