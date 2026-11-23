"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tqdm
import tyro

import openpi.models.model as _model
from openpi.policies import policy_contract as _policy_contract
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _compute_memory_lerobot_vector_stats(
    data_config: _config.DataConfig,
    *,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None,
) -> dict[str, normalize.NormStats]:
    """Match the policy loader's numeric batches without decoding unused videos."""

    contract = data_config.policy_contract
    delta_transform = next(
        (
            transform
            for transform in data_config.data_transforms.inputs
            if isinstance(transform, transforms.DeltaActions)
        ),
        None,
    )
    if contract is None:
        raise ValueError(
            "vector-only stats require a policy contract"
        )
    root = os.environ.get("HF_LEROBOT_HOME") or os.environ.get("LEROBOT_HOME")
    if not root:
        raise ValueError("HF_LEROBOT_HOME or LEROBOT_HOME is required")
    state_key = str(contract["state_key"]) if contract is not None else "observation.state"
    action_key = str(contract["action_key"]) if contract is not None else "action"
    mask_key = data_config.policy_action_mask_key
    repo_ids = tuple(data_config.repo_ids) or (str(data_config.repo_id),)
    parquet_files = [
        path
        for repo_id in repo_ids
        for path in sorted((Path(root) / repo_id / "data").rglob("*.parquet"))
    ]
    if not parquet_files:
        raise FileNotFoundError(f"No LeRobot parquet files found under {root}")

    running = {key: normalize.RunningStats() for key in ("state", "actions")}
    state_batch: list[np.ndarray] = []
    action_batch: list[np.ndarray] = []
    accepted = 0
    stop = False
    for path in tqdm.tqdm(parquet_files, desc="Computing vector stats"):
        columns = [state_key, action_key, "episode_index", "frame_index"]
        if mask_key is not None:
            columns.append(str(mask_key))
        table = pq.read_table(path, columns=columns)
        states = np.asarray(table[state_key].to_pylist(), dtype=np.float32)
        actions = np.asarray(table[action_key].to_pylist(), dtype=np.float32)
        masks = (
            np.asarray(table[str(mask_key)].to_numpy(), dtype=bool)
            if mask_key is not None
            else np.ones(table.num_rows, dtype=bool)
        )
        episode_indices = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        frame_indices = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        for episode_index in dict.fromkeys(episode_indices.tolist()):
            rows = np.flatnonzero(episode_indices == episode_index)
            rows = rows[np.argsort(frame_indices[rows], kind="stable")]
            episode_states = states[rows]
            episode_actions = actions[rows]
            for index in np.flatnonzero(masks[rows]):
                if max_frames is not None and accepted >= max_frames:
                    stop = True
                    break
                future = np.minimum(
                    index + np.arange(action_horizon), len(episode_actions) - 1
                )
                state = episode_states[index]
                future_actions = episode_actions[future].copy()
                if delta_transform is not None:
                    future_actions = delta_transform(
                        {"state": state.copy(), "actions": future_actions}
                    )["actions"]
                state_batch.append(state)
                action_batch.append(future_actions)
                accepted += 1
                if len(state_batch) == batch_size:
                    running["state"].update(np.stack(state_batch))
                    running["actions"].update(np.stack(action_batch))
                    state_batch.clear()
                    action_batch.clear()
            if stop:
                break
        if stop:
            break

    # Match torch DataLoader(drop_last=True).
    used = accepted - len(state_batch)
    if used < batch_size:
        raise ValueError(
            f"Only {used} complete policy-action samples; need batch_size={batch_size}"
        )
    print(
        f"Computed vector-only stats from {used} policy-action frames "
        f"({accepted - used} final frames dropped)."
    )
    return {key: value.get_statistics() for key, value in running.items()}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    if num_batches == 0:
        raise ValueError(
            f"num_batches=0 (dataset={len(dataset)} frames, batch_size={batch_size}, "
            f"max_frames={max_frames}). --max-frames must be >= batch_size (and the dataset "
            f"must have >= batch_size frames) or norm stats get no data."
        )
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    # Statistics are recomputed from raw dataset columns. Do not let an old
    # local asset load (or block) before the new dataset is fingerprinted.
    data_factory = dataclasses.replace(config.data, load_norm_stats=False)
    data_config = data_factory.create(config.assets_dirs, config.model)

    if data_config.repo_ids and data_config.policy_contract is not None:
        norm_stats = _compute_memory_lerobot_vector_stats(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            max_frames=max_frames,
        )
        data_loader = None
        num_batches = 0
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    if data_loader is not None:
        keys = ["state", "actions"]
        stats = {key: normalize.RunningStats() for key in keys}
        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
            for key in keys:
                stats[key].update(np.asarray(batch[key]))
        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # Write where training/serving READ them: assets_dir/asset_id (see DataConfigFactory._load_norm_stats
    # and policy_config). asset_id defaults to repo_id, so this is unchanged for the common case but
    # correct when a config sets a custom assets.asset_id alias (otherwise stats land in repo_id/ and
    # the loader can't find them under asset_id/).
    output_path = config.assets_dirs / (data_config.asset_id or data_config.repo_id)
    print(f"Writing stats to: {output_path}")
    if data_config.policy_contract is not None:
        _policy_contract.validate_norm_stats_dimensions(
            norm_stats,
            data_config.policy_contract,
            source="newly computed normalization statistics",
        )
        repo_ids = tuple(data_config.repo_ids) or (str(data_config.repo_id),)
        manifest_schema = _policy_contract.DATASET_MANIFEST_SCHEMA_VERSION
        if output_path.exists():
            manifest_schema = _policy_contract.dataset_manifest_schema_from_contract(
                _policy_contract.load_policy_contract(output_path)
            )
        dataset_manifest = _policy_contract.build_dataset_manifest(
            repo_ids,
            dataset_variant=data_config.dataset_variant,
            schema_version=manifest_schema,
        )
        bound_contract = _policy_contract.bind_normalization(
            data_config.policy_contract,
            norm_stats,
            normalization_type=(
                "quantile" if data_config.use_quantile_norm else "zscore"
            ),
            asset_id=str(data_config.asset_id),
            dataset_manifest=dataset_manifest,
        )
    if data_config.policy_contract is not None and output_path.exists():
        existing_contract = _policy_contract.load_policy_contract(output_path)
        _policy_contract.validate_policy_contract(
            bound_contract,
            existing_contract,
            source=str(output_path / _policy_contract.POLICY_CONTRACT_FILENAME),
        )
        existing_stats = normalize.load(output_path)
        _policy_contract.validate_normalization_binding(
            existing_contract,
            existing_stats,
            normalization_type=(
                "quantile" if data_config.use_quantile_norm else "zscore"
            ),
            asset_id=str(data_config.asset_id),
            source=str(output_path),
            dataset_manifest=dataset_manifest,
        )
        print("Existing normalization asset is identical; leaving it unchanged.")
        return
    normalize.save(output_path, norm_stats)
    if data_config.policy_contract is not None:
        _policy_contract.save_policy_contract(output_path, bound_contract)


if __name__ == "__main__":
    tyro.cli(main)
