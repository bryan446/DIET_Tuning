import os
import re
import pathlib
from typing import Optional, Tuple
from wandb.apis.public.api import Api

try:
    import torch
except ImportError:
    torch = None  # only needed if you set return_loaded=True

FINAL_ARTIFACT_PREFIX = "final_model_"


def download_final_checkpoint(
    wandb_id: str,
    target_dir: str = "./checkpoints",
    return_loaded: bool = False,
    map_location: str = "cpu",
    entity: str = "jakobamb",
    project: str = "DIET-Finetuning_v3",
) -> Tuple[str, Optional[int], Optional[dict], dict]:
    """
    Download the final checkpoint saved by `save_final_checkpoint(...)`.

    Looks for an artifact named 'final_model_{run.id}' of type 'model',
    finds the file 'final_checkpoint_epoch_{E}.pt' inside, selectively
    downloads only that file into `target_dir`, and (optionally) loads it.

    Args:
        wandb_id: The wandb run ID
        target_dir: Directory to save the downloaded checkpoint
        return_loaded: Whether to load and return the checkpoint dict
        map_location: Device to map tensors to when loading
        entity: Wandb entity (username/team)
        project: Wandb project name

    Returns:
        (local_path, epoch, checkpoint_dict or None, run_config)
    """
    run_path = f"{entity}/{project}/{wandb_id}"
    api = Api()
    run = api.run(run_path)

    # Get the run config
    run_config = run.config

    # Find the matching 'final_model_{run.id}' artifact(s).
    # Use newest last, search backwards.
    candidates = [
        a
        for a in run.logged_artifacts()
        if a.type == "model" and (a.name or "").startswith(FINAL_ARTIFACT_PREFIX)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No model artifacts named '{FINAL_ARTIFACT_PREFIX}*' for run {run_path}."
        )

    chosen = None
    for art in reversed(candidates):
        # Must match exactly this run's id: 'final_model_{run.id}'
        if (art.name or "") == f"{FINAL_ARTIFACT_PREFIX}{run.id}":
            chosen = art
            break
    if chosen is None:
        # Fallback: take newest that starts with the prefix
        # (in case names were duplicated across retries)
        chosen = candidates[-1]

    manifest = getattr(chosen, "manifest", None)
    entries = manifest.entries if (manifest and hasattr(manifest, "entries")) else {}
    if not entries:
        raise FileNotFoundError(f"Artifact '{chosen.name}' has no files.")

    # Find the final checkpoint file inside the artifact
    pat = re.compile(r"final_checkpoint_epoch_(\d+)\.(pt|pth|ckpt|bin)$")
    match_key, epoch = None, None
    for k in entries.keys():
        m = pat.search(os.path.basename(k))
        if m:
            match_key = k
            epoch = int(m.group(1))
            break

    if match_key is None:
        raise FileNotFoundError(
            f"No file like 'final_checkpoint_epoch_{{E}}.pt' "
            f"in artifact '{chosen.name}'."
        )

    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)

    # Download the entire artifact to a unique subdirectory based on wandb_id
    unique_target_dir = os.path.join(target_dir, f"wandb_{wandb_id}")
    artifact_dir = chosen.download(root=unique_target_dir)
    local_path = os.path.join(artifact_dir, match_key)
    local_abs = os.path.abspath(local_path)

    if return_loaded:
        if torch is None:
            raise RuntimeError(
                "PyTorch not available. Install torch or set return_loaded=False."
            )
        ckpt = torch.load(local_abs, map_location=map_location, weights_only=False)
    else:
        ckpt = None

    return local_abs, epoch, ckpt, run_config


# --- usage ---
# path, epoch, ckpt, config = download_final_checkpoint("run_id", return_loaded=True)
# model.load_state_dict(ckpt["model_state_dict"])
# optimizer.load_state_dict(ckpt["optimizer_state_dict"])
# W_diet.load_state_dict(ckpt["W_diet_state_dict"])
# dataset_name = config["dataset_name"]
