import dataclasses
import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
from openpi.policies import policy_contract as _policy_contract
from openpi.policies import policy_metadata as _policy_metadata
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def _expected_policy_contract(train_config: _config.TrainConfig) -> dict[str, Any] | None:
    data_factory = train_config.data
    state_key = getattr(data_factory, "state_key", None)
    state_dim = getattr(data_factory, "state_dim", None)
    action_key = getattr(data_factory, "action_key", None)
    action_dim = getattr(data_factory, "action_dim", None)
    if state_key is None or state_dim is None or action_key is None or action_dim is None:
        return None
    return _policy_contract.build_policy_contract(
        state_key=str(state_key),
        state_dim=int(state_dim),
        action_key=str(action_key),
        action_dim=int(action_dim),
        control_mode_prompt=getattr(data_factory, "control_mode_prompt", None),
        action_horizon=train_config.model.action_horizon,
    )


def _expected_memory_layout_contract(
    train_config: _config.TrainConfig,
) -> dict[str, Any] | None:
    data_factory = train_config.data
    num_keyframes = getattr(data_factory, "num_keyframes", None)
    keyframe_use_wrist = getattr(
        data_factory, "keyframe_use_wrist_camera", None
    )
    if num_keyframes is None or keyframe_use_wrist is None:
        return None
    return _policy_contract.build_memory_layout_contract(
        num_keyframes=int(num_keyframes),
        keyframe_use_wrist_camera=bool(keyframe_use_wrist),
        current_camera_slots=getattr(data_factory, "current_camera_slots", None),
    )


def _build_metadata(
    train_config: _config.TrainConfig,
    *,
    checkpoint_contract: dict[str, Any] | None = None,
    checkpoint_memory_layout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add the current memory benchmark's typed I/O contract when applicable."""

    metadata = _policy_metadata.build_policy_metadata(
        train_config.policy_metadata, train_config.model
    )
    if checkpoint_memory_layout is not None:
        # The checkpoint, not the CLI-selected config, is authoritative after
        # the exact layout validation in create_trained_policy().
        metadata.update(
            _policy_contract.metadata_from_memory_layout(
                checkpoint_memory_layout
            )
        )
        metadata["memory_layout_contract_source"] = "checkpoint"
    contract = checkpoint_contract or _expected_policy_contract(train_config)
    if contract is None:
        return metadata
    metadata.update(_policy_contract.metadata_from_contract(contract))
    metadata["policy_contract_source"] = (
        "checkpoint" if checkpoint_contract is not None else "config"
    )
    return metadata


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    expected_contract = _expected_policy_contract(train_config)
    expected_memory_layout = _expected_memory_layout_contract(train_config)
    checkpoint_contract = None
    checkpoint_memory_layout = None
    contract_dir = checkpoint_dir / "assets"
    if expected_contract is not None:
        checkpoint_contract = _policy_contract.load_policy_contract(str(contract_dir))
        _policy_contract.validate_policy_contract(
            expected_contract,
            checkpoint_contract,
            source=str(contract_dir / _policy_contract.POLICY_CONTRACT_FILENAME),
        )
    if expected_memory_layout is not None:
        checkpoint_memory_layout = _policy_contract.load_memory_layout_contract(
            str(contract_dir)
        )
        _policy_contract.validate_memory_layout_contract(
            expected_memory_layout,
            checkpoint_memory_layout,
            source=str(
                contract_dir
                / _policy_contract.MEMORY_LAYOUT_CONTRACT_FILENAME
            ),
        )

    # Inference is self-contained in the checkpoint. Building transforms must
    # not load or validate an unrelated stale local assets directory.
    inference_data_factory = dataclasses.replace(
        train_config.data, load_norm_stats=False
    )
    data_config = inference_data_factory.create(
        train_config.assets_dirs, train_config.model
    )
    asset_id = data_config.asset_id
    if checkpoint_contract is not None:
        asset_id = checkpoint_contract.get("normalization", {}).get("asset_id", asset_id)
    if not isinstance(asset_id, str) or not asset_id or pathlib.Path(asset_id).name != asset_id:
        raise ValueError("Checkpoint normalization asset_id must be a single directory name.")
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", asset_id)
    if checkpoint_contract is not None:
        _policy_contract.validate_normalization_binding(
            checkpoint_contract,
            norm_stats,
            normalization_type=(
                "quantile" if data_config.use_quantile_norm else "zscore"
            ),
            asset_id=asset_id,
            source=str(checkpoint_dir / "assets" / asset_id),
        )

    # Only touch the large parameter tree after the cheap schema/stat checks pass.
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)
    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(
            _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
        )

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=_build_metadata(
            train_config,
            checkpoint_contract=checkpoint_contract,
            checkpoint_memory_layout=checkpoint_memory_layout,
        ),
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
