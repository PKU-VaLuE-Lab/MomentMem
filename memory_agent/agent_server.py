"""Serve MomentMem over the OpenPI websocket protocol."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OPENPI_CLIENT_SRC = _REPO_ROOT / "executor" / "packages" / "openpi-client" / "src"
_OPENPI_ROOT = _REPO_ROOT / "executor"
_UTILS_ROOT = _REPO_ROOT / "utils"
for _p in (_OPENPI_CLIENT_SRC, _OPENPI_ROOT, _UTILS_ROOT, str(_REPO_ROOT), str(Path(__file__).resolve().parent)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from openpi_client import base_policy as _base_policy  # noqa: E402
from openpi.policies import policy_contract as _policy_contract  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402

from memory_agent import (  # noqa: E402
    ACTION_SPACE_DIMS,
    POLICY_ACTION_DIM,
    POLICY_ACTION_SPACE,
    POLICY_CONTROL_MODE_PROMPT,
    POLICY_STATE_DIM,
    MemoryAgent,
)

try:
    # Map task names to the stable ids used by per-task thresholds.
    from dataset_spec import stable_task_id  # noqa: E402
except Exception:  # pragma: no cover
    stable_task_id = None

logger = logging.getLogger("memory_agent.server")


def _validate_production_contract(agent: MemoryAgent) -> None:
    """Fail before serving if System-1 is not the current nine-task contract."""

    lower_metadata = dict(getattr(agent.vla_policy, "metadata", {}) or {})
    expected_contract = _policy_contract.build_policy_contract(
        state_key="observation.state_joint",
        state_dim=POLICY_STATE_DIM,
        action_key="action_rel_ee",
        action_dim=POLICY_ACTION_DIM,
        control_mode_prompt=POLICY_CONTROL_MODE_PROMPT,
        action_horizon=10,
    )
    expected_lower = _policy_contract.metadata_from_contract(expected_contract)
    expected_lower["policy_contract_source"] = "checkpoint"
    expected_lower["memory_layout_contract_source"] = "checkpoint"
    errors = [
        f"lower metadata {key}={lower_metadata.get(key)!r}, expected {expected!r}"
        for key, expected in expected_lower.items()
        if lower_metadata.get(key) != expected
    ]
    if agent.policy_action_space != POLICY_ACTION_SPACE:
        errors.append(f"policy_action_space={agent.policy_action_space!r}")
    if agent.env_action_space != POLICY_ACTION_SPACE:
        errors.append(f"env_action_space={agent.env_action_space!r}")
    if agent.action_adapter is not None:
        errors.append(f"action_adapter={agent.action_adapter!r}, expected None")
    if errors:
        raise ValueError("Incompatible System-1 policy contract: " + "; ".join(errors))


def _coerce_task_id(raw: Any) -> "int | None":
    """Accept an int task_id, a numeric string, or a task NAME (mapped via stable_task_id())."""
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "ignore")
    if isinstance(raw, str):
        s = raw.strip()
        if s == "":
            return None
        if s.lstrip("-").isdigit():
            return int(s)
        return int(stable_task_id(s))
    arr = np.asarray(raw).reshape(-1)
    if arr.size == 0:
        return None
    if arr.dtype.kind in ("U", "S", "O"):
        return _coerce_task_id(str(arr[0]))
    return int(arr[0])


class AgentServer(_base_policy.BasePolicy):
    """Adapt a stateful :class:`MemoryAgent` to the (stateless) websocket policy interface."""

    def __init__(self, agent: MemoryAgent) -> None:
        self._agent = agent
        self._warned_no_frame = False

    def reset(self) -> None:
        self._agent.reset()

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # Copy the request before removing control fields.
        obs = dict(obs)

        # Episode boundary: client sets obs["reset"]=True on the first frame of a new episode.
        if bool(obs.pop("reset", False)):
            self._agent.reset()

        # Accept either task ids or task names.
        raw_tid = obs.pop("task_id", None)
        raw_name = obs.pop("task_name", None)
        task_id = _coerce_task_id(raw_tid if raw_tid is not None else raw_name)
        prompt = obs.pop("prompt", None)
        # Pass the optional memory instruction only to the VLM route.
        prompt_memory = obs.pop("prompt_memory", None)
        noise = obs.pop("noise", None)
        observe_only = bool(obs.pop("observe_only", False))
        obs.pop("state_representation", None)
        # True env frame index: keeps the System-2 stride on ENV frames, not on requests.
        raw_frame = obs.pop("frame", None)
        env_step = None if raw_frame is None else int(np.asarray(raw_frame).reshape(-1)[0])
        if env_step is None and not self._warned_no_frame:
            logger.warning(
                "Client did not send obs['frame'] (the env step index): retrieval_stride now "
                "counts REQUESTS. If the client executes action chunks (>1 env step per request), "
                "System-2's cadence will NOT match training — send obs['frame'] on normal and "
                "observe-only requests."
            )
            self._warned_no_frame = True

        if observe_only:
            retrieval = self._agent.observe(
                obs,
                prompt=prompt,
                memory=prompt_memory,
                task_id=task_id,
                env_step=env_step,
            )
            return {
                "observe_only": True,
                "num_keyframes": int(self._agent.keyframe_count),
                "retrieval_step": int(retrieval.step_idx),
                "is_keyframe": bool(retrieval.is_keyframe),
                "kf_probability": (
                    float(retrieval.probability) if retrieval.probability is not None else -1.0
                ),
                "kf_processed": bool(retrieval.processed),
            }

        info = self._agent.infer(
            obs,
            prompt=prompt,
            memory=prompt_memory,
            task_id=task_id,
            update_memory=True,
            return_info=True,
            noise=noise,
            env_step=env_step,
        )
        retrieval = info["retrieval"]
        out: Dict[str, Any] = {
            "actions": np.asarray(info["actions"], dtype=np.float32),
            "num_keyframes": int(info["num_keyframes"]),
            "policy_action_space": str(info["policy_action_space"]),
            "env_action_space": str(info["env_action_space"]),
            "action_adapter": str(info["action_adapter"] or "none"),
        }
        # Expose the upper-level retrieval decision for logging/diagnostics (all primitives).
        if retrieval is not None:
            out["is_keyframe"] = bool(retrieval.is_keyframe)
            out["kf_probability"] = (
                float(retrieval.probability) if retrieval.probability is not None else -1.0
            )
            out["kf_processed"] = bool(retrieval.processed)
        return out


class SingleClientPolicyServer(websocket_policy_server.WebsocketPolicyServer):
    """Stateful-policy variant of the openpi websocket server."""

    _active_client = None

    async def _handler(self, websocket):
        if self._active_client is not None:
            logger.warning(
                "Refusing connection from %s: MemoryAgent is stateful and already serving %s.",
                websocket.remote_address, self._active_client,
            )
            await websocket.close(
                code=1013,  # "try again later"
                reason="MemoryAgent server is single-client; another connection is active.",
            )
            return
        self._active_client = websocket.remote_address
        try:
            self._policy.reset()  # a new connection never inherits the previous client's episode
            await super()._handler(websocket)
        finally:
            self._active_client = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("MemoryAgent websocket inference server")
    # VLM mode expects an available remote keyframe server.
    ap.add_argument("--vlm-base-url", default="http://127.0.0.1:8000/v1",
                    help="Base URL of the vLLM OpenAI-compatible server. You may pass .../v1, "
                    ".../v1/chat/completions, or .../v1/completions; multimodal requests are sent "
                    "to the chat-completions endpoint.")
    ap.add_argument("--vlm-model", default="qwen3vl-keyframe",
                    help="Served model name (must match the vLLM server's --served-model-name).")
    ap.add_argument("--vlm-api-key", default="EMPTY", help="API key for the vLLM server (usually a dummy).")
    ap.add_argument("--vlm-max-history", type=int, default=48,
                    help="Cap keyframes sent in the VLM prompt (server --limit-mm-per-prompt safety).")
    ap.add_argument("--vlm-buffer-size", type=int, default=0,
                    help="Number of recent strided timesteps before the current frame to include in "
                    "the VLM prompt. Keyframes inside this buffer are marked there and not repeated "
                    "in the long-term keyframe block.")
    ap.add_argument("--vlm-no-wrist", action="store_true",
                    help="Send only the overview camera when no prompt sidecar is supplied. When a sidecar is "
                    "supplied, this flag must agree with its camera_mode.")
    ap.add_argument("--vlm-add-memory-instruction", action="store_true",
                    help="Append INSTRUCTION_MEMORY (from obs['prompt_memory']) to the VLM prompt only "
                    "(the VLA stays task-only). MUST match how the VLM was trained; a --vlm-prompt-config "
                    "sidecar with add_memory_instruction OVERRIDES this flag.")
    ap.add_argument("--vlm-prompt-config", default=None,
                    help="Path to keyframe_prompt_config.json written at VLM training time "
                    "(add_memory_instruction/buffer_size/max_history/camera_mode/retrieval_stride). When given, these "
                    "OVERRIDE the matching CLI flags so the served prompt structure cannot drift from "
                    "how the VLM was trained; retrieval_stride only fills in when --retrieval-stride is unset.")
    ap.add_argument("--vlm-image-budget", type=int, default=None,
                    help="Max images per request the vLLM server accepts (its --limit-mm-per-prompt "
                    "image cap). The client fails fast with a clear message if a request would exceed it.")
    ap.add_argument("--vlm-image-size", type=int, default=None,
                    help="Resize every image sent to the VLM to this size (square, BILINEAR), matching "
                    "the train-time IMAGE_SIZE. A --vlm-prompt-config sidecar with image_size OVERRIDES "
                    "this. A sidecar image_size=null explicitly selects Qwen dynamic native resolution; "
                    "unset without a sidecar is ambiguous and emits a warning.")
    # System-1 (lower, openpi pi0.5)
    ap.add_argument("--openpi-config", default="pi05_memory",
                    help="openpi TrainConfig name for the lower policy. MUST match how --openpi-checkpoint "
                    "was trained (num_keyframes + keyframe camera mode + action key/dim + norm stats). "
                    "Use pi05_memory for overview-only history. The default "
                    "'pi05_memory' uses 8-D absolute-joint state and 10-D camera-relative EE "
                    "action. Check the startup contract log.")
    ap.add_argument("--openpi-checkpoint", required=True, help="openpi checkpoint dir (the step dir)")
    ap.add_argument("--policy-action-space", default="auto",
                    choices=["auto", POLICY_ACTION_SPACE],
                    help="Action space emitted by the lower openpi policy. 'auto' validates the "
                    "typed 10-D rel_ee metadata from --openpi-config.")
    ap.add_argument("--env-action-space", default="auto",
                    choices=["auto", POLICY_ACTION_SPACE],
                    help="Typed action representation sent to the evaluation client; the current "
                    "benchmark contract is rel_ee.")
    ap.add_argument("--action-adapter", default="auto",
                    choices=["auto", "none"],
                    help="No server-side action conversion is used; auto resolves to none for rel_ee.")
    # runtime
    ap.add_argument("--openpi-pytorch-device", default=None, help="device for a pytorch openpi policy, if used")
    ap.add_argument("--retrieval-stride", type=int, default=None,
                    help="Run the memorizer every Nth frame. "
                    "VLM default: from --vlm-prompt-config's retrieval_stride, else 4 matching the "
                    "legacy 20fps->5fps deployment fallback). VLM training may use temporal-stride "
                    "augmentation; an explicit value here wins over the sidecar.")
    ap.add_argument("--num-keyframes", type=int, default=None,
                    help="keyframe memory size (default: inferred from the openpi config)")
    ap.add_argument("--threshold", type=float, default=None, help="override the default retrieval threshold")
    ap.add_argument("--default-prompt", default=None, help="fallback task prompt if obs carries none")
    ap.add_argument("--allow-missing-wrist", action="store_true",
                    help="zero-fill the wrist image when the client omits observation/wrist_image")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()

    import json as _json

    from vlm_keyframe_client import (
        PROMPT_LAYOUT_VERSION, VLMKeyframeClient, camera_mode_from_config,
        image_size_from_config,
    )  # local import: only needs requests + PIL

    use_wrist = not args.vlm_no_wrist
    cfg = {}
    if args.vlm_prompt_config:
        cfg = _json.loads(Path(args.vlm_prompt_config).read_text())
        # Reject prompt layouts that differ from training.
        layout = cfg.get("prompt_layout")
        if layout is not None and int(layout) != int(PROMPT_LAYOUT_VERSION):
            raise SystemExit(
                f"prompt-config {args.vlm_prompt_config}: prompt_layout={layout} but this code "
                f"builds layout {PROMPT_LAYOUT_VERSION}. Serve this checkpoint with the matching "
                "code revision (or retrain it)."
            )
        if layout is None:
            raise SystemExit(
                f"prompt-config {args.vlm_prompt_config}: missing required prompt_layout."
            )
        # Prefer training-time prompt settings over CLI defaults.
        if "buffer_size" in cfg:
            args.vlm_buffer_size = int(cfg["buffer_size"])
        if "max_history" in cfg:
            args.vlm_max_history = int(cfg["max_history"])
        if "add_memory_instruction" in cfg:
            args.vlm_add_memory_instruction = bool(cfg["add_memory_instruction"])
        try:
            checkpoint_camera_mode = camera_mode_from_config(cfg)
        except ValueError as exc:
            raise SystemExit(f"prompt-config {args.vlm_prompt_config}: {exc}") from exc
        if checkpoint_camera_mode is not None:
            if args.vlm_no_wrist and checkpoint_camera_mode == "overview+wrist":
                raise SystemExit(
                    "--vlm-no-wrist contradicts the dual-camera checkpoint prompt config."
                )
            use_wrist = checkpoint_camera_mode == "overview+wrist"
        try:
            args.vlm_image_size = image_size_from_config(cfg)
        except ValueError as exc:
            raise SystemExit(f"prompt-config {args.vlm_prompt_config}: {exc}") from exc
        # Use the training stride unless explicitly overridden.
        if args.retrieval_stride is None:
            cfg_stride = cfg.get("retrieval_stride")
            if cfg_stride is None and cfg.get("video_fps") and cfg.get("sample_fps"):
                cfg_stride = max(1, int(cfg["video_fps"]) // max(1, int(cfg["sample_fps"])))
            if cfg_stride is not None:
                args.retrieval_stride = int(cfg_stride)
        logger.info(f"Loaded VLM prompt config from {args.vlm_prompt_config}: {cfg}")

    if args.vlm_image_size is None and not cfg:
        logger.warning(
            "No VLM image size configured (--vlm-image-size or sidecar image_size): images go to "
            "the VLM at native camera resolution. This is only correct if that equals the "
            "train-time IMAGE_SIZE — otherwise the vision patch grid differs from training."
        )
    logger.info(
        "VLM System-2 prompt settings (MUST match training): add_memory_instruction=%s "
        "buffer_size=%s camera_mode=%s max_history=%s image_budget=%s image_size=%s retrieval_stride=%s",
        bool(args.vlm_add_memory_instruction), args.vlm_buffer_size,
        "overview+wrist" if use_wrist else "overview",
        args.vlm_max_history, args.vlm_image_budget,
        args.vlm_image_size if args.vlm_image_size is not None else "native(dynamic)",
        args.retrieval_stride if args.retrieval_stride is not None else "4(default)",
    )
    vlm_client = VLMKeyframeClient(
        base_url=args.vlm_base_url,
        model=args.vlm_model,
        api_key=args.vlm_api_key,
        use_wrist=use_wrist,
        max_history=args.vlm_max_history,
        image_budget=args.vlm_image_budget,
        image_size=args.vlm_image_size,
    )

    agent = MemoryAgent(
        vlm_client=vlm_client,
        openpi_config_name=args.openpi_config,
        openpi_checkpoint_dir=args.openpi_checkpoint,
        openpi_pytorch_device=args.openpi_pytorch_device,
        retrieval_stride=args.retrieval_stride,
        num_keyframes=args.num_keyframes,
        threshold=args.threshold,
        default_prompt=args.default_prompt,
        allow_missing_wrist=args.allow_missing_wrist,
        policy_action_space=None if args.policy_action_space == "auto" else args.policy_action_space,
        env_action_space=args.env_action_space,
        action_adapter=args.action_adapter,
        vlm_buffer_size=args.vlm_buffer_size,
        add_memory_instruction=bool(args.vlm_add_memory_instruction),
    )
    try:
        _validate_production_contract(agent)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    policy = AgentServer(agent)

    lower_metadata = dict(agent.vla_policy.metadata)
    contract_metadata = {
        key: lower_metadata[key]
        for key in _policy_contract.metadata_from_contract(
            lower_metadata["policy_contract"]
        )
    }

    metadata = {
        "server": "memory_agent",
        "system2": f"vlm:{args.vlm_model}",
        "system1": args.openpi_config,
        "num_keyframes": agent.num_keyframes,
        "vla_keyframe_camera_mode": (
            "overview+wrist"
            if agent.vla_keyframe_use_wrist_camera
            else "overview"
        ),
        "retrieval_stride": agent.retrieval_stride,
        "policy_action_space": agent.policy_action_space,
        "env_action_space": agent.env_action_space,
        "action_adapter": agent.action_adapter or "none",
        "vlm_buffer_size": agent.vlm_buffer_size,
        "add_memory_instruction": agent.add_memory_instruction,
        "supports_observe": True,
        **contract_metadata,
        "policy_contract_source": "checkpoint",
        "policy_action_dim": ACTION_SPACE_DIMS[agent.policy_action_space],
        "env_action_dim": ACTION_SPACE_DIMS[agent.env_action_space],
    }
    logger.info(f"MemoryAgent ready: {metadata}")
    logger.info(f"Serving on {args.host}:{args.port} (client must set obs['reset']=True at episode start)")

    server = SingleClientPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=metadata
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
