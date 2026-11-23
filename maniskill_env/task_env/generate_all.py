"""Run each v3 task's self-contained ``gen_data.py`` in an isolated process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

TASK_ENV_ROOT = Path(__file__).resolve().parent
MANISKILL_ROOT = TASK_ENV_ROOT.parent
if str(MANISKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(MANISKILL_ROOT))

from generation_contract import (  # noqa: E402
    generation_request,
    generation_request_matches,
    validate_seed_spacing,
    write_generation_request,
)

try:
    from .common import ALL_TASK_NAMES, TASK_NAMES
except ImportError:
    from common import ALL_TASK_NAMES, TASK_NAMES


OBSERVED_TASKS = {
    "AO_hidden_pattern",

    "SO_spring_compression",
    "ASO_placement_imitation",
}


def _is_complete(
    output_dir: Path,
    task_name: str,
    episodes: int,
    *,
    expected_request: dict[str, object],
) -> bool:
    """Return true only for a readable, exactly-sized raw dataset pair."""

    h5_path = output_dir / f"{task_name}.h5"
    json_path = output_dir / f"{task_name}.json"
    if (
        not h5_path.is_file()
        or not json_path.is_file()
        or not generation_request_matches(output_dir, expected_request)
    ):
        return False
    try:
        metadata = json.loads(json_path.read_text())
        if len(metadata.get("episodes", ())) != int(episodes):
            return False
        with h5py.File(h5_path, "r") as h5:
            if len(h5.keys()) != int(episodes):
                return False
            for trajectory in h5.values():
                if "actions" not in trajectory or "policy_action_mask" not in trajectory:
                    return False
                horizon = int(trajectory["actions"].shape[0])
                mask = trajectory["policy_action_mask"][:].astype(bool)
                phase = trajectory["obs/extra/phase_id"][:horizon].reshape(-1)
                expected = (
                    phase == 2
                    if task_name in OBSERVED_TASKS
                    else np.ones(horizon, dtype=bool)
                )
                if mask.shape != (horizon,) or not np.array_equal(mask, expected):
                    return False
            return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--seed-stride", type=int, default=10_000)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--shader", default="default")
    parser.add_argument("--only", nargs="*", choices=ALL_TASK_NAMES)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--visual-ood", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--skip-complete",
        action="store_true",
        help="skip an exact matching request with a readable, exactly-sized H5/JSON pair",
    )
    args = parser.parse_args()
    tasks = tuple(args.only) if args.only else TASK_NAMES
    max_attempts = int(args.max_attempts or (4 * args.episodes + 8))
    try:
        validate_seed_spacing(
            episodes=args.episodes,
            shard_size=args.episodes,
            max_attempts=max_attempts,
            retries=0,
            shard_seed_stride=args.seed_stride,
        )
    except ValueError as error:
        parser.error(str(error))
    if max_attempts < args.episodes:
        parser.error("--max-attempts cannot be smaller than --episodes")
    args.output_root.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, int]] = []
    for task_name in tasks:
        task_dir = TASK_ENV_ROOT / task_name
        generator = task_dir / "gen_data.py"
        if not generator.is_file():
            print(f"[missing] {generator}", file=sys.stderr)
            failures.append((task_name, 2))
            if not args.keep_going:
                break
            continue
        output_dir = args.output_root / task_name
        task_seed = args.base_seed + ALL_TASK_NAMES.index(task_name) * args.seed_stride
        request = generation_request(
            scope="direct-task",
            task=task_name,
            visual_ood=args.visual_ood,
            episodes=args.episodes,
            base_seed=task_seed,
            steps=args.steps,
            max_attempts=max_attempts,
            retries=0,
            shader=args.shader,
            render_backend=None,
            save_video=not args.no_video,
        )
        if args.skip_complete and _is_complete(
            output_dir,
            task_name,
            args.episodes,
            expected_request=request,
        ):
            print(f"[skip-complete] {task_name}: {args.episodes} episodes", flush=True)
            continue
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(task_dir), str(TASK_ENV_ROOT)]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        command = [
            sys.executable,
            str(generator),
            "--episodes",
            str(args.episodes),
            "--base-seed",
            str(task_seed),
            "--steps",
            str(args.steps),
            "--shader",
            str(args.shader),
            "--output-dir",
            str(output_dir.resolve()),
        ]
        if args.max_attempts is not None:
            command.extend(("--max-attempts", str(args.max_attempts)))
        if args.no_video:
            command.append("--no-video")
        if args.visual_ood:
            command.append("--visual-ood")
        print(f"[generate] {task_name}: {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=task_dir, env=env)
        if result.returncode:
            failures.append((task_name, int(result.returncode)))
            if not args.keep_going:
                break
        else:
            write_generation_request(output_dir, request)
    if failures:
        print(
            "generation failures: "
            + ", ".join(f"{task}(rc={code})" for task, code in failures),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
