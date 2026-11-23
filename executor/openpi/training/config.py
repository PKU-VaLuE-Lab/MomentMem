"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
from openpi.policies import policy_contract as _policy_contract
import openpi.policies.memory_policy as memory_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Optional list of LeRobot datasets concatenated after each one receives its
    # own task prompt, policy-action filtering, and keyframe-history wrapper.
    repo_ids: Sequence[str] = ()
    # Human-readable split/variant identity included in the normalization
    # dataset manifest (for example, "standard" versus "visual_ood").
    dataset_variant: str = "unspecified"
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None
    # Exact learned state/action semantics. Saved into every checkpoint and
    # verified before normalization or policy serving.
    policy_contract: dict[str, Any] | None = None
    # Checkpoint-specific current/history image-slot layout. Kept separate from
    # the state/action contract so camera ablations can reuse identical norm stats.
    memory_layout_contract: dict[str, Any] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # History-keyframe inputs. When > 0, the data loader attaches each frame's
    # past keyframe images. Must match the model config's `num_keyframes`.
    num_keyframes: int = 0
    # Whether the history memory contains synchronized wrist views in addition
    # to the overview views. This is independent of the current observation.
    keyframe_use_wrist_camera: bool = True
    # Which frame of each [start, end] memory window to use as the keyframe:
    # "mid"|"start"|"end" (fixed) or "sample" (draw per access from p(k) ∝ keyframe_gamma**k, like the
    # upper Qwen selectors, to match the window-internal frames they fire at online).
    keyframe_point: str = "mid"
    # Geometric discount for keyframe_point="sample" (1.0 = uniform over the window; <1 = early-biased).
    # Default 0.99 mirrors the upper Qwen selectors' early-biased window sampling.
    keyframe_gamma: float = 0.99
    # Independently insert each past frame outside all keyframe windows as a
    # false-positive memory. Zero disables this robustness augmentation.
    non_keyframe_memory_prob: float = 0.0
    # Path to keyframes.json (episode -> mem_boundary_pairs). If None, uses <dataset_root>/meta/keyframes.json.
    keyframes_path: str | None = None
    # LeRobot feature keys for the two cameras used to build keyframe views.
    keyframe_base_image_key: str = "observation.images.image"
    keyframe_wrist_image_key: str = "observation.images.wrist_image"

    # Optional scalar LeRobot feature selecting rows that are valid action
    # targets.  The full episode is retained for keyframe-image lookup, while
    # the training sampler exposes only True rows.
    policy_action_mask_key: str | None = None
    require_policy_action_mask: bool = False



class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None
    # Inference and norm-stat recomputation must not be blocked by unrelated
    # local assets. They explicitly disable this and supply checkpoint/new stats.
    load_norm_stats: tyro.conf.Suppress[bool] = True

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=(
                self._load_norm_stats(
                    epath.Path(self.assets.assets_dir or assets_dirs), asset_id
                )
                if self.load_norm_stats
                else None
            ),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )










@dataclasses.dataclass(frozen=True)
class LeRobotMemoryDataConfig(DataConfigFactory):
    """Data config for memory tasks with history-keyframe image inputs.

    Expects a LeRobot dataset (converted from the maniskill memory tasks) whose camera features
    are `observation.images.image` (base) and `observation.images.wrist_image` (wrist), and whose
    `meta/keyframes.json` holds per-episode `mem_boundary_pairs`. Each training frame is given the
    images at its past keyframe timesteps (see KeyframeHistoryDataset / MemoryInputs).
    """

    # Number of history keyframes (must match the model config's `num_keyframes`).
    num_keyframes: int = 8
    # False makes every history keyframe overview-only. The current observation
    # remains base+wrist, matching the regular pi0.5 observation interface.
    keyframe_use_wrist_camera: bool = True
    # Multiple local/HF LeRobot repo ids for joint training. ``repo_id`` remains
    # the logical asset id used for the combined normalization statistics.
    repo_ids: Sequence[str] = ()
    dataset_variant: str = "legacy"
    # Which frame of each [start, end] window to use as the keyframe: "mid"|"start"|"end" (fixed) or
    # "sample" (per-access draw matching the upper selectors; see keyframe_gamma).
    keyframe_point: str = "mid"
    # Geometric discount for keyframe_point="sample" (1.0 = uniform; <1 = early-biased), mirroring the
    # memorizer window-sampling distribution. Default 0.99 = their early-biased weighting.
    keyframe_gamma: float = 0.99
    # Probability that each past frame outside every keyframe window is also
    # inserted into history, simulating false-positive upper-level predictions.
    non_keyframe_memory_prob: float = 0.0
    # Optional explicit path to keyframes.json (else <dataset_root>/meta/keyframes.json).
    keyframes_path: str | None = None
    # Which LeRobot proprioceptive column becomes pi0.5's discretized state.
    # These legacy defaults keep archived configs loadable; the current nine-task
    # ``pi05_memory`` config explicitly selects observation.state_joint (8-D) and
    # action_rel_ee (10-D) below.
    state_key: str = "observation.state"
    state_dim: int = 17
    # Number of real (unpadded) action dimensions.
    action_dim: int = 17
    # Which LeRobot column to load as the action. The converter stores several action-space
    # representations side by side (see maniskill_env/converter/convert.py):
    #   "action" (canonical, 17d) | "action_native" (7d) |
    #   "action_abs_joint" (8d) | "action_delta_joint" (8d) |
    #   "action_rel_ee" (camera-relative xyz+Rot6D+gripper, 10d)
    # Pick one per training run; `action_dim` must match it:
    # canonical = 17d, native delta-EE = 7d, joint spaces = 8d, rel-EE = 10d.
    action_key: str = "action"
    # pi0.5's action-space marker prepended to the prompt so the model knows which action space to
    # output (it was pretrained co-training joint + end-effector poses, distinguished by this token;
    # see π0.5 paper §action representation). Match the action_key: e.g.
    #   action_abs_joint -> "<control_mode> joint <control_mode> "
    #   action_rel_ee    -> "<control_mode> end effector <control_mode> "
    # Leave None for the default delta-EE setup (no marker). NOTE: exact token format is taken from
    # the paper; verify against PI's released tokenizer/assets if leveraging pretrained heads matters.
    control_mode_prompt: str | None = None
    # Optional fixed prompt if the dataset has no task strings.
    default_prompt: str | None = None
    # Current ManiSkill memory datasets always carry this field.  Requiring it
    # prevents older observed-task conversions from silently training on the
    # autonomous observation phase's no-op controls.
    policy_action_mask_key: str = "policy_action_mask"

    def _validate_vector_columns(self) -> None:
        """Best-effort: assert the selected state/action columns and dimensions."""
        import json
        import os

        if self.repo_id in (None, "fake", tyro.MISSING):
            return
        root = os.environ.get("HF_LEROBOT_HOME") or os.environ.get("LEROBOT_HOME")
        if not root:
            return
        repo_ids = tuple(self.repo_ids) or (str(self.repo_id),)
        for repo_id in repo_ids:
            info_path = pathlib.Path(root) / repo_id / "meta" / "info.json"
            if not info_path.is_file():
                continue  # dataset not locatable here -> LeRobot will report it later
            feats = (json.loads(info_path.read_text()).get("features") or {})
            selections = (
                ("state", self.state_key, self.state_dim),
                ("action", self.action_key, self.action_dim),
            )
            for kind, key, expected_dim in selections:
                if key not in feats:
                    prefix = "observation.state" if kind == "state" else "action"
                    have = sorted(k for k in feats if k.startswith(prefix))
                    raise ValueError(
                        f"Dataset {info_path.parent.parent} has no {kind} column {key!r} "
                        f"(has {have}). Re-convert with the current converter or select an "
                        f"existing {kind} column."
                    )
                dim = next(iter(feats[key].get("shape") or [None]))
                if dim is not None and int(dim) != int(expected_dim):
                    raise ValueError(
                        f"{kind}_dim={expected_dim} but dataset column {key!r} is {dim}d "
                        f"({info_path}). Re-convert with the current schema or fix the config."
                    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Schema guard: the alternate action columns (native/joint/rel_ee)
        # are only present in datasets converted with the current converter. Datasets converted earlier
        # may have a legacy 7-D `action`. Fail fast with a
        # clear message instead of a cryptic KeyError / silent wrong-dim training. Best-effort: only
        # checks when the dataset is locatable under HF_LEROBOT_HOME.
        self._validate_vector_columns()
        if not 0.0 <= self.non_keyframe_memory_prob <= 1.0:
            raise ValueError(
                "non_keyframe_memory_prob must be in [0, 1], got "
                f"{self.non_keyframe_memory_prob}"
            )

        # Slot and camera guards: the data transform and model input spec must
        # define exactly the same history images.
        model_nkf = getattr(model_config, "num_keyframes", None)
        if model_nkf is not None and int(model_nkf) != int(self.num_keyframes):
            raise ValueError(
                f"num_keyframes mismatch: data config has {self.num_keyframes} but model config has "
                f"{model_nkf}. They must be equal."
            )
        model_kf_wrist = bool(
            getattr(model_config, "keyframe_use_wrist_camera", True)
        )
        if model_kf_wrist != bool(self.keyframe_use_wrist_camera):
            raise ValueError(
                "keyframe camera mismatch: data config has "
                f"keyframe_use_wrist_camera={self.keyframe_use_wrist_camera} "
                "but model config has "
                f"keyframe_use_wrist_camera={model_kf_wrist}."
            )

        # Map the LeRobot feature keys to the standard names used by MemoryInputs.
        repack_mapping = {
            "observation/image": "observation.images.image",
            "observation/wrist_image": "observation.images.wrist_image",
            "observation/state": self.state_key,
            "actions": self.action_key,
            # Pass the task prompt through (added by PromptFromLeRobotTask).
            "prompt": "prompt",
            # Keyframe images attached by KeyframeHistoryDataset.
            "keyframe_image_base": "keyframe_image_base",
            "keyframe_mask": "keyframe_mask",
        }
        if self.keyframe_use_wrist_camera:
            repack_mapping["keyframe_image_wrist"] = "keyframe_image_wrist"
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(repack_mapping)
            ]
        )

        # Prepend pi0.5's control_mode marker to the prompt (runs after PromptFromLeRobotTask sets the
        # prompt, before TokenizePrompt). No-op when control_mode_prompt is None (default delta-EE).
        data_inputs = [
            memory_policy.MemoryInputs(
                model_type=model_config.model_type,
                num_keyframes=self.num_keyframes,
                keyframe_use_wrist_camera=self.keyframe_use_wrist_camera,
            )
        ]
        if self.control_mode_prompt:
            data_inputs = [_transforms.PrependPrompt(self.control_mode_prompt), *data_inputs]
        data_transforms = _transforms.Group(
            inputs=data_inputs,
            outputs=[memory_policy.MemoryOutputs(action_dim=self.action_dim)],
        )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        contract = _policy_contract.build_policy_contract(
            state_key=self.state_key,
            state_dim=self.state_dim,
            action_key=self.action_key,
            action_dim=self.action_dim,
            control_mode_prompt=self.control_mode_prompt,
            action_horizon=model_config.action_horizon,
        )
        memory_layout_contract = _policy_contract.build_memory_layout_contract(
            num_keyframes=self.num_keyframes,
            keyframe_use_wrist_camera=self.keyframe_use_wrist_camera,
        )
        base_config = self.create_base_config(assets_dirs, model_config)
        if contract is not None and base_config.norm_stats is not None:
            stats_asset_dir = epath.Path(self.assets.assets_dir or assets_dirs) / str(
                base_config.asset_id
            )
            stats_contract = _policy_contract.load_policy_contract(stats_asset_dir)
            _policy_contract.validate_policy_contract(
                contract,
                stats_contract,
                source=str(
                    stats_asset_dir / _policy_contract.POLICY_CONTRACT_FILENAME
                ),
            )
            _policy_contract.validate_normalization_binding(
                stats_contract,
                base_config.norm_stats,
                normalization_type=(
                    "quantile" if base_config.use_quantile_norm else "zscore"
                ),
                asset_id=str(base_config.asset_id),
                source=str(stats_asset_dir),
            )
            contract = stats_contract
        return dataclasses.replace(
            base_config,
            repo_ids=tuple(self.repo_ids),
            dataset_variant=self.dataset_variant,
            policy_contract=contract,
            memory_layout_contract=memory_layout_contract,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=(self.action_key,),
            prompt_from_task=True,
            num_keyframes=self.num_keyframes,
            keyframe_use_wrist_camera=self.keyframe_use_wrist_camera,
            keyframe_point=self.keyframe_point,
            keyframe_gamma=self.keyframe_gamma,
            non_keyframe_memory_prob=self.non_keyframe_memory_prob,
            keyframes_path=self.keyframes_path,
            policy_action_mask_key=self.policy_action_mask_key,
            require_policy_action_mask=True,
        )






@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of micro-batches whose mean gradient forms one optimizer update.
    # ``batch_size`` remains the effective global batch size, so each data-loader
    # batch has size ``batch_size // gradient_accumulation_steps``.
    gradient_accumulation_steps: int = 1
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))


    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.batch_size % self.gradient_accumulation_steps != 0:
            raise ValueError(
                "batch_size must be divisible by gradient_accumulation_steps"
            )


# Use `get_config` if you need to get a config by name in your code.
MEMORY_BENCHMARK_REPO_IDS = (
    "AO_hidden_pattern",
    "SO_spring_compression",
    "ASO_placement_imitation",
    "AT_drawer_match",
    "ST_flash_beacon",
    "AST_mine_sweeper",
    "AD_scroll_discovery",
    "SD_explore_cube",
    "ASD_seal_checking",
)

def _momentmem_config(name: str, variant: str) -> TrainConfig:
    """Train the executor on keyframes predicted by a frozen memorizer."""
    return TrainConfig(
        name=name,
        project_name="MomentMem",
        assets_base_dir=os.environ.get("MOMENTMEM_ASSETS_PATH", "{assets_path}"),
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, num_keyframes=8,
            keyframe_use_wrist_camera=False,
        ),
        data=LeRobotMemoryDataConfig(
            repo_id=f"momentmem/{variant}",
            repo_ids=MEMORY_BENCHMARK_REPO_IDS,
            dataset_variant=variant,
            assets=AssetsConfig(asset_id=variant),
            num_keyframes=8,
            keyframe_use_wrist_camera=False,
            keyframe_point="start",
            state_key="observation.state_joint", state_dim=8,
            action_key="action_rel_ee", action_dim=10,
            control_mode_prompt="<control_mode> end effector <control_mode> ",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.environ.get("MOMENTMEM_BASE_PARAMS", "gs://openpi-assets/checkpoints/pi05_base/params")
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=60_000),
        checkpoint_base_dir=os.environ.get("MOMENTMEM_CHECKPOINT_PATH", "{executor_output_path}"),
        batch_size=64, fsdp_devices=1, num_workers=4,
        num_train_steps=60_000, save_interval=10_000, keep_period=10_000,
        wandb_enabled=False,
    )


_CONFIGS = [
    _momentmem_config("pi05_memory", "predicted_memory"),
    _momentmem_config("pi05_memory_ood", "visual_ood_predicted_memory"),
]

_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
