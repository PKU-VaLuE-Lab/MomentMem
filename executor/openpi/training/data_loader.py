from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
from openpi.policies import policy_contract as _policy_contract
import openpi.training.config as _config
from openpi.training.keyframe_dataset import KeyframeHistoryDataset
from openpi.training.keyframe_dataset import PolicyActionDataset
from openpi.training.keyframe_dataset import load_keyframe_windows
from openpi.training.keyframe_dataset import load_policy_action_indices
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    seed: int = 0,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    if (
        data_config.policy_contract is not None
        and "normalization" in data_config.policy_contract
    ):
        repo_ids = tuple(data_config.repo_ids) or (repo_id,)
        dataset_manifest = _policy_contract.build_dataset_manifest(
            repo_ids,
            dataset_variant=data_config.dataset_variant,
            schema_version=_policy_contract.dataset_manifest_schema_from_contract(
                data_config.policy_contract
            ),
        )
        _policy_contract.validate_normalization_binding(
            data_config.policy_contract,
            data_config.norm_stats,
            normalization_type=(
                "quantile" if data_config.use_quantile_norm else "zscore"
            ),
            asset_id=str(data_config.asset_id),
            source="current LeRobot training datasets",
            dataset_manifest=dataset_manifest,
        )

    if data_config.repo_ids:
        if data_config.keyframes_path is not None:
            raise ValueError(
                "A single keyframes_path cannot describe multiple LeRobot datasets; "
                "leave it unset so each repo uses <root>/meta/keyframes.json."
            )
        datasets = [
            _create_single_torch_dataset(
                data_config,
                child_repo_id,
                action_horizon,
                keyframes_path=None,
                seed=seed + repo_index,
            )
            for repo_index, child_repo_id in enumerate(data_config.repo_ids)
        ]
        if not datasets:
            raise ValueError("repo_ids must contain at least one LeRobot dataset")
        logging.info(
            "Joint LeRobot training dataset: %d repos, %d policy-action frames: %s",
            len(datasets),
            sum(len(dataset) for dataset in datasets),
            list(data_config.repo_ids),
        )
        return torch.utils.data.ConcatDataset(datasets)

    return _create_single_torch_dataset(
        data_config,
        repo_id,
        action_horizon,
        keyframes_path=data_config.keyframes_path,
        seed=seed,
    )


def _create_single_torch_dataset(
    data_config: _config.DataConfig,
    repo_id: str,
    action_horizon: int,
    *,
    keyframes_path: str | pathlib.Path | None,
    seed: int,
) -> Dataset:
    """Build one repo before concatenation so episode/task indices stay local."""

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        video_backend=os.environ.get("LEROBOT_VIDEO_BACKEND"),
    )
    policy_action_indices = None
    if data_config.policy_action_mask_key is not None:
        policy_action_indices = load_policy_action_indices(
            dataset, data_config.policy_action_mask_key
        )
        if policy_action_indices is None and data_config.require_policy_action_mask:
            raise ValueError(
                f"dataset {repo_id!r} has no required "
                f"{data_config.policy_action_mask_key!r} feature; re-convert it "
                "before training so observation-only actions cannot be sampled"
            )

    # History-keyframe support: attach each frame's past-keyframe images (see KeyframeHistoryDataset).
    if data_config.num_keyframes > 0:
        kf_path = keyframes_path
        if kf_path is None:
            root = getattr(dataset, "root", None) or getattr(dataset_meta, "root", None)
            if root is None:
                raise ValueError("num_keyframes > 0 but keyframes_path is unset and dataset root is unknown.")
            kf_path = pathlib.Path(root) / "meta" / "keyframes.json"
        logging.info(f"Loading keyframe windows from {kf_path}")
        dataset = KeyframeHistoryDataset(
            dataset,
            load_keyframe_windows(kf_path),
            num_keyframes=data_config.num_keyframes,
            base_image_key=data_config.keyframe_base_image_key,
            wrist_image_key=data_config.keyframe_wrist_image_key,
            use_wrist_camera=data_config.keyframe_use_wrist_camera,
            keyframe_point=data_config.keyframe_point,
            keyframe_gamma=data_config.keyframe_gamma,
            non_keyframe_memory_prob=data_config.non_keyframe_memory_prob,
            seed=seed,
        )

    # Apply the row mapping only after the keyframe wrapper has captured the
    # full dataset.  Its internal keyframe lookups must continue to use the
    # original global frame indices, including observation-only frames.
    if policy_action_indices is not None:
        dataset = PolicyActionDataset(dataset, policy_action_indices)

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    # ``TrainConfig.batch_size`` is the effective global batch per optimizer
    # update.  The loader supplies one micro-batch at a time; JAX training uses
    # optax.MultiSteps to accumulate these gradients before updating weights.
    micro_batch_size = config.batch_size // config.gradient_accumulation_steps
    logging.info(
        "effective global batch_size=%d, gradient_accumulation_steps=%d, micro_batch_size=%d",
        config.batch_size,
        config.gradient_accumulation_steps,
        micro_batch_size,
    )

    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=micro_batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(
        data_config, action_horizon, model_config, seed=seed
    )
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        self._epoch = 0

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            sampler = self._data_loader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self._epoch)
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    self._epoch += 1
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)

    def __len__(self) -> int:
        return self._num_batches or len(self._data_loader)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]

    def __len__(self) -> int:
        return len(self._data_loader)

    def set_epoch(self, epoch: int) -> None:
        if not hasattr(self._data_loader, "set_epoch"):
            raise TypeError("This data loader does not support distributed epochs")
        self._data_loader.set_epoch(epoch)
