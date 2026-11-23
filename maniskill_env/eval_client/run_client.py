"""Run one v3 task in GT-replay mode or real MemoryAgent policy mode."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from typing import Any

_HARNESS = Path(__file__).resolve().parent
_V3_ROOT = _HARNESS.parent
_TASK_ROOT = _V3_ROOT / "task_env"
for path in (_V3_ROOT, _TASK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from converter.specs import ALL_TASK_NAMES, TASK_SPECS  # noqa: E402
from eval_client.base_task_client import BaseTaskClient  # noqa: E402
from eval_client.protocol import REPLAYABLE_SPACES  # noqa: E402


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_client(task: str, *, visual_ood: bool = False) -> type[BaseTaskClient]:
    task_dir = _TASK_ROOT / task
    for path in (task_dir, _TASK_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    main_path = task_dir / f"{task}.py"
    ood_path = task_dir / f"{task}_OOD.py"
    client_path = task_dir / "task_client.py"
    if not main_path.is_file() or not client_path.is_file():
        raise FileNotFoundError(
            f"task {task} requires {main_path.name} and {client_path.name} in {task_dir}"
        )
    # Use the canonical module name so task_client and the OOD subclass import
    # the already-loaded task rather than registering it a second time.
    _load_file(task, main_path)
    if visual_ood:
        if not ood_path.is_file():
            raise FileNotFoundError(f"task {task} lacks visual OOD module {ood_path}")
        _load_file(f"{task}_OOD", ood_path)
    module = _load_file(f"_v3_client_{task}", client_path)
    classes = [
        value
        for _, value in inspect.getmembers(module, inspect.isclass)
        if issubclass(value, BaseTaskClient)
        and value is not BaseTaskClient
        and value.__module__ == module.__name__
    ]
    standard_env_id = TASK_SPECS[task]["env_id"]
    expected = (
        TASK_SPECS[task]["visual_ood_env_id"] if visual_ood else standard_env_id
    )
    if visual_ood:
        classes[0].ENV_ID = expected
    return classes[0]


def _metadata_env_id(path: Path | None) -> str | None:
    if path is None:
        return None
    source = json.loads(path.read_text())
    return str((source.get("env_info") or {}).get("env_id") or "")


def _json_object(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    path = Path(raw).expanduser()
    try:
        text = path.read_text() if path.is_file() else raw
    except OSError:
        text = raw
    value = json.loads(text)
    return value


def _episode(path: Path | None, index: int, expected_env_id: str):
    if path is None:
        return None, {}, None, None
    source = json.loads(path.read_text())
    env_info = source.get("env_info") or {}
    episodes = source.get("episodes") or []
    episode = episodes[index]
    metadata = episode.get("episode_metadata")
    prompt = env_info.get("instruction_task") or env_info.get("instruction")
    memory = env_info.get("instruction_memory")
    return episode, metadata, prompt, memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("policy", "replay"), default="policy")
    parser.add_argument("--task", choices=ALL_TASK_NAMES, required=True)
    parser.add_argument(
        "--visual-ood",
        action="store_true",
        help="evaluate the task's visual-OOD environment; auto-enabled for OOD metadata",
    )
    parser.add_argument("--meta-json", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--action-space", choices=REPLAYABLE_SPACES, default="delta_ee")
    parser.add_argument("--camera-resolution", type=int, default=224)
    parser.add_argument("--shader", default="default")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--execution-horizon", type=int, default=1)
    parser.add_argument("--api-key")
    parser.add_argument("--open-timeout", type=float, default=60.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--env-kwargs-json")
    parser.add_argument("--prompt")
    parser.add_argument("--memory-prompt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_env_id = _metadata_env_id(args.meta_json)
    ood_env_id = TASK_SPECS[args.task]["visual_ood_env_id"]
    visual_ood = bool(args.visual_ood or metadata_env_id == ood_env_id)
    if args.visual_ood and metadata_env_id not in (None, "", ood_env_id):
        raise ValueError(
            f"--visual-ood conflicts with metadata env_id={metadata_env_id!r}"
        )
    client_class = load_client(args.task, visual_ood=visual_ood)
    episode, episode_metadata, stored_prompt, stored_memory = _episode(
        args.meta_json, args.episode, client_class.ENV_ID
    )
    seed = (
        int(args.seed)
        if args.seed is not None
        else int(episode["episode_seed"])
        if episode is not None
        else 0
    )
    if (
        episode is not None
        and args.seed is not None
        and int(args.seed) != int(episode["episode_seed"])
    ):
        raise ValueError(
            f"--seed={args.seed} conflicts with recorded episode_seed="
            f"{episode['episode_seed']}"
        )
    prompt = args.prompt if args.prompt is not None else stored_prompt
    memory = args.memory_prompt if args.memory_prompt is not None else stored_memory
    client = client_class(
        camera_resolution=args.camera_resolution,
        shader=args.shader,
        env_kwargs=_json_object(args.env_kwargs_json),
    )
    if args.mode == "replay":
        if args.meta_json is None:
            raise ValueError("--mode replay requires --meta-json for exact reset metadata")
        result = client.run_replay(
            action_space=args.action_space,
            seed=seed,
            episode_metadata=episode_metadata,
            prompt=prompt,
            memory_prompt=memory,
            host=args.host,
            port=8770 if args.port is None else args.port,
            max_steps=args.max_steps,
        )
    else:
        result = client.run_policy(
            seed=seed,
            episode_metadata=episode_metadata or None,
            prompt=prompt,
            memory_prompt=memory,
            host=args.host,
            port=8000 if args.port is None else args.port,
            api_key=args.api_key,
            max_steps=args.max_steps,
            execution_horizon=args.execution_horizon,
            open_timeout=args.open_timeout,
            request_timeout=args.request_timeout,
        )
    if args.meta_json is not None:
        result["source_json"] = str(args.meta_json.resolve())
        result["source_episode"] = int(args.episode)
    result["visual_ood"] = visual_ood
    print("RESULT_JSON:" + json.dumps(result, default=str, sort_keys=True), flush=True)
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
