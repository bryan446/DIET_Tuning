import logging
from typing import Dict, Optional
import torch
from transformers import AutoModel

import utils.convert_dinov3_vit_to_hf as dinov3_to_hf

logger = logging.getLogger("dinov3_cp_loader")


def load_huggingface_model(model_id: str) -> Dict[str, torch.Tensor]:
    """
    Load a DINOv3 model from HuggingFace Hub and convert it to native DINOv3
    format.

    Args:
        model_id: HuggingFace model identifier (e.g., "facebook/dinov3-vitb16")

    Returns:
        State dict compatible with native DINOv3 models

    Raises:
        ValueError: If model architecture is incompatible
        RuntimeError: If model loading fails
    """
    logger.info(f"Loading HuggingFace model: {model_id}")

    try:
        # Download model files
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        state_dict = model.state_dict()

        # Convert keys to DINOv3 format
        logger.info("Converting HuggingFace keys to DINOv3 format")
        converted_state_dict = _convert_hf_keys_to_dinov3(state_dict)

        logger.info(f"Successfully loaded and converted HuggingFace model {model_id}")
        logger.info(f"Model contains {len(converted_state_dict)} parameters")

        return converted_state_dict

    except Exception as e:
        raise RuntimeError(
            f"Failed to load HuggingFace model {model_id}: {str(e)}"
        ) from e


def _convert_hf_keys_to_dinov3(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Convert HuggingFace DINOv3 state dict keys to native DINOv3 format.

    HF DINOv3 models use different key naming conventions than the native
    implementation. This function translates between the two formats.

    Args:
        state_dict: HuggingFace model state dict

    Returns:
        Converted state dict with DINOv3 native keys
    """
    converted_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        # Convert embeddings layer
        if "embeddings.patch_embeddings.projection" in new_key:
            new_key = new_key.replace(
                "embeddings.patch_embeddings.projection", "patch_embed.proj"
            )
        elif "embeddings.cls_token" in new_key:
            new_key = new_key.replace("embeddings.cls_token", "cls_token")

        # Convert encoder blocks
        if "encoder.layer." in new_key:
            new_key = new_key.replace("encoder.layer.", "blocks.")

        # Convert attention layers
        if ".attention.attention.query" in new_key:
            new_key = new_key.replace(".attention.attention.query", ".attn.qkv")
            # Note: HF splits qkv, but DINOv3 uses fused qkv - handle specially
            # Skip key weights, we'll handle in qkv fusion
        elif ".attention.output.dense" in new_key:
            new_key = new_key.replace(".attention.output.dense", ".attn.proj")

        # Convert layer norms
        if ".layernorm_before" in new_key:
            new_key = new_key.replace(".layernorm_before", ".norm1")
        elif ".layernorm_after" in new_key:
            new_key = new_key.replace(".layernorm_after", ".norm2")

        # Convert MLP layers
        if ".intermediate.dense" in new_key:
            new_key = new_key.replace(".intermediate.dense", ".mlp.fc1")
        elif ".output.dense" in new_key:
            new_key = new_key.replace(".output.dense", ".mlp.fc2")

        converted_state_dict[new_key] = value

    # Handle QKV fusion - combine separate Q, K, V weights into single qkv
    converted_state_dict = _fuse_qkv_weights(state_dict, converted_state_dict)

    return converted_state_dict


def _fuse_qkv_weights(
    original_state_dict: Dict[str, torch.Tensor],
    converted_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Fuse separate Q, K, V weights from HuggingFace format into single qkv
    weights for DINOv3.

    Args:
        original_state_dict: Original HF state dict with separate q, k, v
        converted_state_dict: Partially converted state dict

    Returns:
        State dict with fused qkv weights
    """
    # Find all transformer blocks
    block_indices = set()
    for key in original_state_dict.keys():
        if "encoder.layer." in key:
            parts = key.split(".")
            if "layer" in parts:
                layer_idx = parts[parts.index("layer") + 1]
                try:
                    block_indices.add(int(layer_idx))
                except ValueError:
                    continue

    # Fuse qkv for each block
    for block_idx in block_indices:
        # Define key patterns for q, k, v weights and biases
        base_key = f"encoder.layer.{block_idx}.attention.attention"
        q_weight_key = f"{base_key}.query.weight"
        k_weight_key = f"{base_key}.key.weight"
        v_weight_key = f"{base_key}.value.weight"

        q_bias_key = f"{base_key}.query.bias"
        k_bias_key = f"{base_key}.key.bias"
        v_bias_key = f"{base_key}.value.bias"

        # Check if all qkv weights exist
        qkv_weight_keys = [q_weight_key, k_weight_key, v_weight_key]
        if all(key in original_state_dict for key in qkv_weight_keys):
            # Fuse weights
            q_weight = original_state_dict[q_weight_key]
            k_weight = original_state_dict[k_weight_key]
            v_weight = original_state_dict[v_weight_key]
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            converted_state_dict[f"blocks.{block_idx}.attn.qkv.weight"] = qkv_weight

            # Fuse biases if they exist
            qkv_bias_keys = [q_bias_key, k_bias_key, v_bias_key]
            if all(key in original_state_dict for key in qkv_bias_keys):
                q_bias = original_state_dict[q_bias_key]
                k_bias = original_state_dict[k_bias_key]
                v_bias = original_state_dict[v_bias_key]
                qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                converted_state_dict[f"blocks.{block_idx}.attn.qkv.bias"] = qkv_bias

    return converted_state_dict


def convert_dinov3_to_huggingface(dinov3_state_dict, use_student=False):
    """
    Convert native DINOv3 checkpoints to HuggingFace format.
    """
    key_prefix = "student.backbone" if use_student else "teacher.backbone"

    assert any(k.startswith(key_prefix) for k in dinov3_state_dict.keys()), (
        f"No keys starting with '{key_prefix}' found in state dict. Not a valid DINOv3 training checkpoint?"
    )

    dinov3_state_dict = {
        k.split(f"{key_prefix}.")[1]: v
        for k, v in dinov3_state_dict.items()
        if k.startswith(key_prefix)
    }

    dinov3_state_dict = dinov3_to_hf.split_qkv(dinov3_state_dict)
    original_keys = list(dinov3_state_dict.keys())
    new_keys = dinov3_to_hf.convert_old_keys_to_new_keys(original_keys)

    converted_state_dict = {}

    for key in original_keys:
        new_key = new_keys[key]
        weight_tensor = dinov3_state_dict[key]

        if "bias_mask" in key or "attn.k_proj.bias" in key or "local_cls_norm" in key:
            continue
        if "embeddings.mask_token" in new_key:
            weight_tensor = weight_tensor.unsqueeze(1)
        if "inv_freq" in new_key:
            continue

        converted_state_dict[new_key] = weight_tensor

    return converted_state_dict
