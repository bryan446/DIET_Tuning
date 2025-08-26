"""Evaluation utilities for DIET finetuning framework."""

import numpy as np
import torch
import time
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    f1_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def zero_shot_eval(
    model,
    train_loader,
    test_loader,
    num_classes,
    device,
    probe_lr=1e-3,
    probe_steps=20000,
):
    """Evaluate model using zero-shot methods with proper train/test split.

    Trains on train_loader features and evaluates on test_loader features.
    Single-label datasets only.

    Args:
        net: Model to evaluate
        train_loader: DataLoader for training features
        test_loader: DataLoader for test features
        num_classes: Number of classes
        device: Device to run evaluation on
        probe_lr: Learning rate for linear probe (default: 1e-3)
        probe_steps: Number of training steps for linear probe (default: 50)

    Returns:
        dict: Dictionary of evaluation metrics
    """
    start_time = time.time()
    print("Extracting features for zero-shot evaluation with train/test split...")

    def extract_features(loader, split_name):
        """Extract features from a data loader."""
        features, labels = [], []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Extracting {split_name} features"):
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    x, y, _ = batch
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x, y = batch
                else:
                    raise ValueError("Unexpected batch structure")

                x = x.to(device)
                feat = model(x)

                if isinstance(feat, list):
                    features.extend([f.detach().cpu().numpy() for f in feat])
                else:
                    features.append(feat.detach().cpu().numpy())

                labels.append(y.detach().cpu().numpy())

        features = np.vstack(features)
        labels = np.concatenate(labels, axis=0)

        # Normalize labels to 1D
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(1)

        return features, labels

    # Extract train and test features separately
    train_features, train_labels = extract_features(train_loader, "train")
    test_features, test_labels = extract_features(test_loader, "test")

    print(
        f"Train features: {train_features.shape}, "
        f"Train labels: {train_labels.shape}"
    )
    print(f"Test features: {test_features.shape}, " f"Test labels: {test_labels.shape}")
    print(f"Time: {time.time() - start_time:.2f}s")

    results = {}

    # ---------- k-NN ----------
    from sklearn.preprocessing import normalize

    print("Running k-NN evaluation...")
    t0 = time.time()

    # Dynamically set n_neighbors to avoid error when training set is small
    n_train_samples = train_features.shape[0]
    n_neighbors = min(20, n_train_samples)
    print(
        f"Using {n_neighbors} neighbors (max of 20 or available training samples: {n_train_samples})"
    )

    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors, metric="cosine", weights="distance"
    )
    train_features = normalize(train_features)  # L2
    test_features = normalize(test_features)
    knn.fit(train_features, train_labels)
    knn_pred = knn.predict(test_features)
    knn_proba = knn.predict_proba(test_features)

    results["knn_acc"] = accuracy_score(test_labels, knn_pred)
    results["knn_f1"] = f1_score(
        test_labels, knn_pred, average="macro", zero_division=0
    )

    # ROC AUC calculation - handle binary vs multiclass
    try:
        if num_classes == 2:
            # For binary classification, use positive class probabilities
            positive_proba = np.array(knn_proba)[:, 1]
            results["knn_roc_auc"] = roc_auc_score(test_labels, positive_proba)
        else:
            results["knn_roc_auc"] = roc_auc_score(
                test_labels, knn_proba, multi_class="ovr", average="macro"
            )
    except ValueError as e:
        print(f"Warning: Could not compute ROC AUC for k-NN: {e}")
        results["knn_roc_auc"] = 0.0

    print(f"k-NN accuracy: {results['knn_acc']:.4f}, " f"time: {time.time() - t0:.2f}s")
    print(f"k-NN F1 (macro): {results['knn_f1']:.4f}")
    print(f"k-NN ROC AUC: {results['knn_roc_auc']:.4f}")

    # ---------- Linear probe ----------
    print("Running linear probe evaluation (train on train, test on test)...")
    t0 = time.time()

    out_dim = int(num_classes)
    clf = torch.nn.Linear(train_features.shape[1], out_dim).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=probe_lr)
    crit = torch.nn.CrossEntropyLoss()

    clf.train()
    train_features = normalize(train_features, norm="l2")
    test_features = normalize(test_features, norm="l2")

    Xtr = torch.as_tensor(train_features, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    Xte = torch.as_tensor(test_features, dtype=torch.float32, device=device)

    batch_size = 512
    n_samples = Xtr.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    indices = torch.randperm(n_samples, device=device)

    for step in range(probe_steps):
        # Shuffle indices for each epoch
        if step % n_batches == 0:
            indices = torch.randperm(n_samples, device=device)

        # Get current batch
        batch_idx = step % n_batches
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]

        X_batch = Xtr[batch_indices]
        y_batch = ytr[batch_indices]

        opt.zero_grad()
        loss = crit(clf(X_batch), y_batch)
        loss.backward()
        opt.step()

    clf.eval()
    with torch.no_grad():
        logits = clf(Xte)
        pred = logits.argmax(dim=1).cpu().numpy()
        # Get probabilities for ROC AUC calculation
        probe_proba = torch.softmax(logits, dim=1).cpu().numpy()

    results["linear_acc"] = accuracy_score(test_labels, pred)
    results["linear_f1"] = f1_score(test_labels, pred, average="macro", zero_division=0)

    # ROC AUC calculation for linear probe
    try:
        if num_classes == 2:
            # For binary classification, use positive class probabilities
            positive_proba = probe_proba[:, 1]
            results["linear_roc_auc"] = roc_auc_score(test_labels, positive_proba)
        else:
            results["linear_roc_auc"] = roc_auc_score(
                test_labels, probe_proba, multi_class="ovr", average="macro"
            )
    except ValueError as e:
        print(f"Warning: Could not compute ROC AUC for linear probe: {e}")
        results["linear_roc_auc"] = 0.0

    print(
        f"Linear probe accuracy: {results['linear_acc']:.4f}, "
        f"time: {time.time() - t0:.2f}s"
    )
    print(f"Linear probe F1 (macro): {results['linear_f1']:.4f}")
    print(f"Linear probe ROC AUC: {results['linear_roc_auc']:.4f}")

    print(f"Total zero-shot evaluation time: {time.time() - start_time:.2f}s")
    return results


def evaluate_test_set(net, test_loader, device, W_probe):
    """Evaluate model on test set with probe classifier

    Args:
        net: Model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        W_probe: Probe classifier

    Returns:
        float: Test accuracy
    """
    print("\nStarting evaluation on test set:")
    net.eval()
    with torch.no_grad():
        run_acc_test = []
        for i, batch in enumerate(test_loader):
            # Flexible unpacking: works with (x, y) or (x, y, n)
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    x, y, _ = batch
                elif len(batch) == 2:
                    x, y = batch
                else:
                    print(
                        f"Warning: Unexpected test batch length ({len(batch)}) at batch {i}. Skipping."
                    )
                    continue
            else:
                print(f"Warning: Unexpected test batch type at batch {i}. Skipping.")
                continue

            x = x.to(device)
            y = y.to(device)
            z = net(x)
            logits_probe = W_probe(z)

            # Adjust dimensions if necessary
            if y.dim() != logits_probe.argmax(1).dim():
                y = (
                    y.squeeze()
                    if y.dim() > logits_probe.argmax(1).dim()
                    else y.unsqueeze(0)
                )
            batch_acc = torch.mean((y == logits_probe.argmax(1)).float()).item()
            run_acc_test.append(batch_acc)

            if i % 10 == 0:  # Print only every 10 batches to reduce verbosity
                print(f"Test Batch {i}: Accuracy={batch_acc:.4f}")

        test_acc = np.mean(run_acc_test) if run_acc_test else 0
        print(f"\nOverall Test Accuracy: {test_acc:.4f}")
    return test_acc


def create_zero_shot_progression_plot(metrics_history, tracked_epochs, metrics_list):
    """Create zero-shot metrics progression plot

    Args:
        metrics_history: Dictionary of metrics history
        tracked_epochs: List of epochs to track
        metrics_list: List of metric names to include

    Returns:
        fig: Matplotlib figure
    """
    # Create figure
    fig = Figure(figsize=(15, 10))

    # Plot each metric's progression
    for i, metric in enumerate(metrics_list):
        ax = fig.add_subplot(2, 2, i + 1)
        values = [
            metrics_history["zero_shot_metrics"][e][metric] for e in tracked_epochs
        ]
        ax.plot(tracked_epochs, values, marker="o", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{metric} Score")
        ax.set_title(f"Zero-shot {metric} Progression")
        ax.grid(True)

        # Add initial and final values as text annotations
        ax.annotate(
            f"{values[0]:.4f}",
            (tracked_epochs[0], values[0]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        ax.annotate(
            f"{values[-1]:.4f}",
            (tracked_epochs[-1], values[-1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    fig.tight_layout()
    fig.suptitle("Zero-shot Metrics Progression During Training", fontsize=16)
    fig.subplots_adjust(top=0.9)

    return fig


def create_training_progress_plot(metrics_history):
    """Create training progress plot

    Args:
        metrics_history: Dictionary of metrics history

    Returns:
        fig: Matplotlib figure
    """
    # Create figure
    fig = Figure(figsize=(15, 5))

    # Plot loss
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(metrics_history["train_loss_diet"], label="DIET Loss")
    ax1.plot(metrics_history["train_loss_probe"], label="Probe Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(metrics_history["train_acc"], label="Train Accuracy")
    ax2.plot(metrics_history["test_acc"], label="Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Model Accuracy")
    ax2.legend()
    ax2.grid(True)

    # Plot zero-shot metrics
    ax3 = fig.add_subplot(1, 3, 3)

    # Get initial and final zero-shot metrics
    tracked_epochs = sorted(metrics_history["zero_shot_metrics"].keys())
    if len(tracked_epochs) >= 2:
        initial_epoch = tracked_epochs[0]
        final_epoch = tracked_epochs[-1]

        metrics = list(metrics_history["zero_shot_metrics"][initial_epoch].keys())
        x = range(len(metrics))
        width = 0.35

        # Plot bar chart comparing initial and final metrics
        ax3.bar(
            x,
            [metrics_history["zero_shot_metrics"][initial_epoch][m] for m in metrics],
            width,
            label="Initial",
        )
        ax3.bar(
            [i + width for i in x],
            [metrics_history["zero_shot_metrics"][final_epoch][m] for m in metrics],
            width,
            label="Final",
        )
        ax3.set_xlabel("Metrics")
        ax3.set_ylabel("Score")
        ax3.set_title("Zero-Shot Performance")
        ax3.set_xticks([i + width / 2 for i in x])
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True)

    fig.tight_layout()

    return fig


def calculate_metrics_improvements(initial_metrics, final_metrics):
    """Calculate improvements between initial and final metrics

    Args:
        initial_metrics: Dictionary with initial metric values
        final_metrics: Dictionary with final metric values

    Returns:
        dict: Dictionary with improvements and relative improvements
    """
    improvements = {}

    for metric in initial_metrics:
        initial = initial_metrics[metric]
        final = final_metrics[metric]
        change = final - initial
        relative_change = (change / initial) * 100 if initial > 0 else float("inf")

        improvements[f"improvement_{metric}"] = change
        improvements[f"relative_improvement_{metric}"] = relative_change

    # Calculate average improvement
    avg_improvement = np.mean(
        [improvements[f"improvement_{m}"] for m in initial_metrics.keys()]
    )
    avg_rel_improvement = np.mean(
        [
            improvements[f"relative_improvement_{m}"]
            for m in initial_metrics.keys()
            if improvements[f"relative_improvement_{m}"] != float("inf")
        ]
    )

    improvements["average_improvement"] = avg_improvement
    improvements["average_relative_improvement"] = avg_rel_improvement

    return improvements
