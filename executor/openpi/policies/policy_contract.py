"""Immutable state/action schema stored with memory-policy checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from openpi.policies import dataset_manifest as _dataset_manifest

POLICY_CONTRACT_FILENAME = "policy_contract.json"
POLICY_CONTRACT_SCHEMA_VERSION = 1
MEMORY_LAYOUT_CONTRACT_FILENAME = "memory_layout_contract.json"
MEMORY_LAYOUT_CONTRACT_SCHEMA_VERSION = 1
NORMALIZATION_BINDING_SCHEMA_VERSION = 1
DATASET_MANIFEST_SCHEMA_VERSION = _dataset_manifest.DATASET_MANIFEST_SCHEMA_VERSION

_RUNTIME_CONTRACT_FIELDS = (
    "schema_version",
    "state_key",
    "state_dim",
    "state_representation",
    "state_coordinate_frame",
    "action_key",
    "action_dim",
    "policy_action_space",
    "action_representation",
    "action_coordinate_frame",
    "rotation_representation",
    "rotation_convention",
    "control_mode_prompt",
    "action_horizon",
)

_STATE_CONTRACTS = {
    ("observation.state", 14): {
        "state_representation": (
            "left_absolute_joint6+physical_gripper1+"
            "right_absolute_joint6+physical_gripper1"
        ),
        "state_coordinate_frame": "joint_space",
    },
    ("observation.state_joint", 8): {
        "state_representation": "absolute_joint7+physical_gripper1",
        "state_coordinate_frame": "joint_space",
    },
    ("observation.state", 17): {
        "state_representation": "joint7+physical_gripper1+base_camera_xyz3+rot6d6",
        "state_coordinate_frame": "mixed_joint_and_base_camera",
    },
    ("observation.state", 12): {
        "state_representation": (
            "absolute_ee_xyz3+rot6d_first_two_rows6+gripper_width1+"
            "grasped1+previous_gripper_target1"
        ),
        "state_coordinate_frame": "franka_base",
    },
}

_ACTION_CONTRACTS = {
    ("action", 14): {
        "policy_action_space": "abs_joint",
        "action_representation": (
            "left_absolute_joint6+gripper_command1+"
            "right_absolute_joint6+gripper_command1"
        ),
        "action_coordinate_frame": "joint_space",
        "rotation_representation": "none",
        "rotation_convention": "none",
    },
    ("action", 17): {
        "policy_action_space": "canonical",
        "action_representation": (
            "absolute_joint7+gripper_command1+base_camera_relative_xyz3+relative_rot6d6"
        ),
        "action_coordinate_frame": "mixed_joint_and_base_camera",
        "rotation_representation": "rot6d_first_two_matrix_columns",
        "rotation_convention": "spatial_relative_rotation_expressed_in_base_camera",
    },
    ("action_native", 7): {
        "policy_action_space": "delta_ee",
        "action_representation": "controller_native_delta_xyz3+euler_xyz3+gripper_command1",
        "action_coordinate_frame": "panda_base",
        "rotation_representation": "euler_xyz",
        "rotation_convention": "maniskill_pd_ee_delta_pose_root_aligned_body_rotation",
    },
    ("action_rel_ee", 10): {
        "policy_action_space": "rel_ee",
        "action_representation": (
            "base_camera_relative_xyz3+relative_rot6d6+gripper_command1"
        ),
        "action_coordinate_frame": "base_camera_opencv",
        "rotation_representation": "rot6d_first_two_matrix_columns",
        "rotation_convention": "spatial_relative_rotation_expressed_in_base_camera",
    },
    ("action_abs_joint", 8): {
        "policy_action_space": "abs_joint",
        "action_representation": "absolute_joint7+gripper_command1",
        "action_coordinate_frame": "joint_space",
        "rotation_representation": "none",
        "rotation_convention": "none",
    },
    ("action_delta_joint", 8): {
        "policy_action_space": "delta_joint",
        "action_representation": "relative_joint7+gripper_command1",
        "action_coordinate_frame": "joint_space",
        "rotation_representation": "none",
        "rotation_convention": "none",
    },
    ("action", 10): {
        "policy_action_space": "abs_ee",
        "action_representation": (
            "absolute_ee_target_xyz3+rot6d_first_two_rows6+binary_gripper_target1"
        ),
        "action_coordinate_frame": "franka_base",
        "rotation_representation": "rot6d_first_two_matrix_rows",
        "rotation_convention": "absolute_rotation_expressed_in_franka_base",
    },
}


def build_policy_contract(
    *,
    state_key: str,
    state_dim: int,
    action_key: str,
    action_dim: int,
    control_mode_prompt: str | None,
    action_horizon: int,
) -> dict[str, Any] | None:
    """Build the exact contract for a recognized memory-policy representation."""

    state = _STATE_CONTRACTS.get((str(state_key), int(state_dim)))
    action = _ACTION_CONTRACTS.get((str(action_key), int(action_dim)))
    if state is None or action is None:
        return None
    return {
        "schema_version": POLICY_CONTRACT_SCHEMA_VERSION,
        "state_key": str(state_key),
        "state_dim": int(state_dim),
        **state,
        "action_key": str(action_key),
        "action_dim": int(action_dim),
        **action,
        "control_mode_prompt": control_mode_prompt,
        "action_horizon": int(action_horizon),
    }


def save_policy_contract(directory: str | Path, contract: Mapping[str, Any]) -> Path:
    """Write once, refusing to replace a different contract."""

    path = Path(directory) / POLICY_CONTRACT_FILENAME
    payload = json.dumps(dict(contract), indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = json.loads(path.read_text())
        validate_policy_contract(contract, existing, source=str(path))
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    return path


def load_policy_contract(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / POLICY_CONTRACT_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Policy contract not found at {path}. Legacy checkpoints cannot be served as "
            "the current memory policy, and legacy normalization assets must be recomputed "
            "with the current compute_norm_stats entrypoint."
        )
    contract = json.loads(path.read_text())
    if not isinstance(contract, dict):
        raise ValueError(f"{path}: policy contract must be a JSON object")
    return contract


def validate_policy_contract(
    expected: Mapping[str, Any], actual: Mapping[str, Any], *, source: str
) -> None:
    """Require every expected field to match the checkpoint exactly."""

    errors = [
        f"{key}: checkpoint={actual.get(key)!r}, config={value!r}"
        for key, value in expected.items()
        if actual.get(key) != value
    ]
    if errors:
        raise ValueError(f"Policy contract mismatch in {source}: " + "; ".join(errors))


def build_memory_layout_contract(
    *,
    num_keyframes: int,
    keyframe_use_wrist_camera: bool,
    current_camera_slots: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Describe the visual slot layout that a memory checkpoint was trained with."""

    num_keyframes = int(num_keyframes)
    keyframe_use_wrist_camera = bool(keyframe_use_wrist_camera)
    keyframe_camera_mode = (
        "overview+wrist" if keyframe_use_wrist_camera else "overview"
    )
    keyframe_camera_slots = ["base_0_rgb"]
    if keyframe_use_wrist_camera:
        keyframe_camera_slots.append("left_wrist_0_rgb")
    current_camera_slots = list(
        ("base_0_rgb", "left_wrist_0_rgb")
        if current_camera_slots is None
        else current_camera_slots
    )
    if current_camera_slots == ["base_0_rgb"]:
        current_camera_mode = "overview"
    elif current_camera_slots in (
        ["base_0_rgb", "left_wrist_0_rgb"],
        ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
    ):
        current_camera_mode = "overview+wrist"
    else:
        current_camera_mode = "+".join(current_camera_slots)
    return {
        "schema_version": MEMORY_LAYOUT_CONTRACT_SCHEMA_VERSION,
        "num_keyframes": num_keyframes,
        "current_camera_mode": current_camera_mode,
        "current_camera_slots": current_camera_slots,
        "keyframe_use_wrist_camera": keyframe_use_wrist_camera,
        "keyframe_camera_mode": keyframe_camera_mode,
        "keyframe_camera_slots": keyframe_camera_slots,
    }


def save_memory_layout_contract(
    directory: str | Path, contract: Mapping[str, Any]
) -> Path:
    """Write a checkpoint's immutable memory-image layout."""

    path = Path(directory) / MEMORY_LAYOUT_CONTRACT_FILENAME
    payload = json.dumps(dict(contract), indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = json.loads(path.read_text())
        validate_memory_layout_contract(contract, existing, source=str(path))
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    return path


def load_memory_layout_contract(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / MEMORY_LAYOUT_CONTRACT_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Memory layout contract not found at {path}. This checkpoint cannot prove "
            "which history-camera layout it was trained with; use a checkpoint saved "
            "with the current training code."
        )
    contract = json.loads(path.read_text())
    if not isinstance(contract, dict):
        raise ValueError(f"{path}: memory layout contract must be a JSON object")
    return contract


def validate_memory_layout_contract(
    expected: Mapping[str, Any], actual: Mapping[str, Any], *, source: str
) -> None:
    """Require the checkpoint's current/history image slots to match exactly."""

    errors = [
        f"{key}: checkpoint={actual.get(key)!r}, config={value!r}"
        for key, value in expected.items()
        if actual.get(key) != value
    ]
    if errors:
        raise ValueError(
            f"Memory layout contract mismatch in {source}: " + "; ".join(errors)
        )


def metadata_from_memory_layout(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return handshake fields sourced from a verified checkpoint layout."""

    return {
        "num_keyframes": int(contract["num_keyframes"]),
        "current_camera_mode": str(contract["current_camera_mode"]),
        "keyframe_use_wrist_camera": bool(
            contract["keyframe_use_wrist_camera"]
        ),
        "keyframe_camera_mode": str(contract["keyframe_camera_mode"]),
        "memory_layout_contract_schema_version": int(contract["schema_version"]),
        "memory_layout_contract": dict(contract),
    }


def validate_norm_stats_dimensions(
    norm_stats: Mapping[str, Any] | None,
    contract: Mapping[str, Any],
    *,
    source: str,
) -> None:
    """Reject sliced/padded normalization statistics with the wrong semantic width."""

    if norm_stats is None:
        raise ValueError(f"Normalization statistics are missing from {source}")
    expected = {
        "state": int(contract["state_dim"]),
        "actions": int(contract["action_dim"]),
    }
    errors: list[str] = []
    for key, dimension in expected.items():
        stats = norm_stats.get(key)
        if stats is None:
            errors.append(f"missing {key!r}")
            continue
        for field in ("mean", "std", "q01", "q99"):
            value = getattr(stats, field, None)
            if value is None:
                if field in {"mean", "std"}:
                    errors.append(f"{key}.{field} is missing")
                continue
            shape = np.asarray(value).shape
            if shape != (dimension,):
                errors.append(
                    f"{key}.{field} has shape {shape}, expected {(dimension,)}"
                )
    if errors:
        raise ValueError(f"Normalization-stat contract mismatch in {source}: " + "; ".join(errors))


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def norm_stats_sha256(norm_stats: Mapping[str, Any]) -> str:
    """Hash every normalization value in a dtype-independent canonical form."""

    payload: dict[str, Any] = {}
    for key in sorted(norm_stats):
        stats = norm_stats[key]
        payload[key] = {}
        for field in ("mean", "std", "q01", "q99"):
            value = getattr(stats, field, None)
            payload[key][field] = (
                None
                if value is None
                else np.asarray(value, dtype=np.float64).tolist()
            )
    return _json_sha256(payload)


build_dataset_manifest = _dataset_manifest.build_dataset_manifest


def dataset_manifest_schema_from_contract(contract: Mapping[str, Any]) -> int:
    """Return the manifest schema stored in a normalization binding."""

    binding = contract.get("normalization")
    if not isinstance(binding, Mapping):
        raise ValueError("Normalization binding is missing from policy contract")
    manifest = binding.get("dataset_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("dataset_manifest is missing from normalization binding")
    return int(manifest.get("schema_version", 1))


def bind_normalization(
    contract: Mapping[str, Any],
    norm_stats: Mapping[str, Any],
    *,
    normalization_type: str,
    asset_id: str,
    dataset_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind exact stats and their source dataset to an immutable policy contract."""

    validate_norm_stats_dimensions(norm_stats, contract, source="normalization binding")
    runtime_contract = {key: contract[key] for key in _RUNTIME_CONTRACT_FIELDS}
    normalization = {
        "schema_version": NORMALIZATION_BINDING_SCHEMA_VERSION,
        "normalization_type": str(normalization_type),
        "asset_id": str(asset_id),
        "policy_contract_sha256": _json_sha256(runtime_contract),
        "norm_stats_sha256": norm_stats_sha256(norm_stats),
        "dataset_manifest": dict(dataset_manifest),
        "dataset_manifest_sha256": str(dataset_manifest["manifest_sha256"]),
    }
    return {**runtime_contract, "normalization": normalization}


def validate_normalization_binding(
    contract: Mapping[str, Any],
    norm_stats: Mapping[str, Any] | None,
    *,
    normalization_type: str,
    asset_id: str,
    source: str,
    dataset_manifest: Mapping[str, Any] | None = None,
) -> None:
    """Require exact stats, normalization mode, asset id and optional dataset identity."""

    validate_norm_stats_dimensions(norm_stats, contract, source=source)
    binding = contract.get("normalization")
    if not isinstance(binding, Mapping):
        raise ValueError(f"Normalization binding is missing from {source}")
    runtime_contract = {key: contract[key] for key in _RUNTIME_CONTRACT_FIELDS}
    expected = {
        "schema_version": NORMALIZATION_BINDING_SCHEMA_VERSION,
        "normalization_type": str(normalization_type),
        "asset_id": str(asset_id),
        "policy_contract_sha256": _json_sha256(runtime_contract),
        "norm_stats_sha256": norm_stats_sha256(norm_stats),
    }
    if dataset_manifest is not None:
        expected["dataset_manifest_sha256"] = str(
            dataset_manifest["manifest_sha256"]
        )
    errors = [
        f"{key}: stored={binding.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if binding.get(key) != value
    ]
    stored_manifest = binding.get("dataset_manifest")
    if dataset_manifest is not None and stored_manifest != dict(dataset_manifest):
        errors.append("stored dataset_manifest does not match the current datasets")
    if errors:
        raise ValueError(
            f"Normalization binding mismatch in {source}: " + "; ".join(errors)
        )


def metadata_from_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return JSON-safe handshake fields sourced from a verified checkpoint contract."""

    runtime_contract = {key: contract[key] for key in _RUNTIME_CONTRACT_FIELDS}
    return {
        **runtime_contract,
        "policy_contract_schema_version": int(runtime_contract["schema_version"]),
        "policy_contract": runtime_contract,
    }


__all__ = [
    "DATASET_MANIFEST_SCHEMA_VERSION",
    "MEMORY_LAYOUT_CONTRACT_FILENAME",
    "MEMORY_LAYOUT_CONTRACT_SCHEMA_VERSION",
    "POLICY_CONTRACT_FILENAME",
    "POLICY_CONTRACT_SCHEMA_VERSION",
    "bind_normalization",
    "build_dataset_manifest",
    "build_memory_layout_contract",
    "build_policy_contract",
    "dataset_manifest_schema_from_contract",
    "load_memory_layout_contract",
    "load_policy_contract",
    "metadata_from_contract",
    "metadata_from_memory_layout",
    "norm_stats_sha256",
    "save_memory_layout_contract",
    "save_policy_contract",
    "validate_memory_layout_contract",
    "validate_norm_stats_dimensions",
    "validate_normalization_binding",
    "validate_policy_contract",
]
