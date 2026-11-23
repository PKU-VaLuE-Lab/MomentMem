"""Prepare, run, and summarize the reproducible nine-task closed-loop benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from converter.specs import TASK_NAMES  # noqa: E402
from eval_client.connection import MemoryAgentConnection  # noqa: E402
from eval_client.run_client import load_client  # noqa: E402


EVAL_SEED_BASE = 1_000_000_000
EVAL_TASK_SEED_STRIDE = 10_000
POLICY_SEED_BASE = 1_500_000_000
# Keep benchmark randomness in a high seed namespace.
EVAL_SEED_LOWER_BOUND = 1_000_000_000
MAX_SEED_EXCLUSIVE = 2**31
HASHED_SEED_SCHEME = "sha256-v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        + "\n"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=HERE.parent,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        records.append(value)
    return records


def _derive_hashed_seed(
    *, master_seed: int, namespace: str, task: str, trial: int, used: set[int]
) -> int:
    """Derive a stable, unique seed in the benchmark seed namespace."""

    span = MAX_SEED_EXCLUSIVE - EVAL_SEED_LOWER_BOUND
    for nonce in range(MAX_SEED_EXCLUSIVE):
        material = (
            f"memory-based-vla/{HASHED_SEED_SCHEME}/{master_seed}/"
            f"{namespace}/{task}/{trial}/{nonce}"
        ).encode()
        digest = hashlib.sha256(material).digest()
        seed = EVAL_SEED_LOWER_BOUND + (
            int.from_bytes(digest[:8], "big") % span
        )
        if seed not in used:
            used.add(seed)
            return seed
    raise RuntimeError("could not derive a unique evaluation seed")


def _outcome_video_path(
    output: Path, task: str, trial: int, success: bool
) -> Path:
    outcome = "S" if success else "F"
    return output / "videos" / task / f"trial_{trial:03d}-{outcome}.mp4"


def _complete_result_matches_manifest(
    output: Path, row: dict[str, Any], result: dict[str, Any]
) -> bool:
    if result.get("status") != "complete":
        return False
    if result.get("manifest_record") != row:
        return False
    expected_video = _outcome_video_path(
        output,
        str(row["task"]),
        int(row["trial"]),
        bool(result.get("success")),
    ).resolve()
    recorded_video = result.get("video_path")
    if not isinstance(recorded_video, str):
        return False
    return (output / recorded_video).resolve() == expected_video and expected_video.is_file()


def prepare(args: argparse.Namespace) -> int:
    output = args.output_root.resolve()
    visual_ood = bool(args.visual_ood)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "eval_manifest.jsonl"
    config_path = output / "campaign_config.json"
    if manifest.exists() or config_path.exists():
        raise FileExistsError(
            f"refusing to replace an existing campaign manifest under {output}"
        )
    stale_artifacts = [
        path
        for path in (
            output / "results",
            output / "videos",
            output / "errors",
            output / "summary.json",
        )
        if path.exists()
    ]
    if stale_artifacts:
        raise FileExistsError(
            "refusing to prepare a campaign over existing result artifacts: "
            + ", ".join(str(path) for path in stale_artifacts)
        )
    if args.trials_per_task <= 0 or args.trials_per_task > EVAL_TASK_SEED_STRIDE:
        raise ValueError("trials-per-task must be in [1, 10000]")
    hashed_seed_mode = (
        args.episode_seed_master is not None or args.policy_seed_master is not None
    )
    if hashed_seed_mode and (
        args.episode_seed_master is None or args.policy_seed_master is None
    ):
        raise ValueError(
            "episode-seed-master and policy-seed-master must be provided together"
        )

    qwen_root = args.qwen_model.resolve()
    prompt_sidecar = qwen_root / "keyframe_prompt_config.json"
    vla_checkpoint = args.vla_checkpoint.resolve()
    vla_config = str(args.vla_config).strip()
    required = (
        qwen_root / "model.safetensors.index.json",
        prompt_sidecar,
        vla_checkpoint / "assets" / "memory_layout_contract.json",
        vla_checkpoint / "assets" / "policy_contract.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing checkpoint evidence: " + ", ".join(missing))
    prompt_config = json.loads(prompt_sidecar.read_text())
    if (
        prompt_config.get("camera_mode") != "overview"
        or bool(prompt_config.get("add_memory_instruction"))
        or int(prompt_config.get("buffer_size", -1)) != 10
        or int(prompt_config.get("retrieval_stride", -1)) != 4
    ):
        raise ValueError(f"unexpected Qwen prompt contract: {prompt_config}")
    memory_contract_path = vla_checkpoint / "assets" / "memory_layout_contract.json"
    memory_contract = json.loads(memory_contract_path.read_text())
    if (
        memory_contract.get("keyframe_camera_mode") != "overview"
        or bool(memory_contract.get("keyframe_use_wrist_camera"))
    ):
        raise ValueError(f"unexpected VLA memory contract: {memory_contract}")

    records: list[dict[str, Any]] = []
    used_seeds: set[int] = set()
    for task_index, task in enumerate(TASK_NAMES):
        for trial in range(args.trials_per_task):
            if hashed_seed_mode:
                seed = _derive_hashed_seed(
                    master_seed=int(args.episode_seed_master),
                    namespace="episode",
                    task=task,
                    trial=trial,
                    used=used_seeds,
                )
                policy_seed = _derive_hashed_seed(
                    master_seed=int(args.policy_seed_master),
                    namespace="policy-noise",
                    task=task,
                    trial=trial,
                    used=used_seeds,
                )
            else:
                seed = EVAL_SEED_BASE + task_index * EVAL_TASK_SEED_STRIDE + trial
                policy_seed = (
                    POLICY_SEED_BASE + task_index * EVAL_TASK_SEED_STRIDE + trial
                )
            records.append(
                {
                    "record_index": len(records),
                    "task_index": task_index,
                    "task": task,
                    "trial": trial,
                    "episode_seed": seed,
                    "policy_noise_seed": policy_seed,
                    "visual_ood": visual_ood,
                }
            )
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    config = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "repo_commit_at_prepare": _git_commit(),
        "evaluator": "maniskill_env/evaluate_closed_loop.py",
        "evaluator_sha256": _sha256(Path(__file__).resolve()),
        "tasks": list(TASK_NAMES),
        "trials_per_task": int(args.trials_per_task),
        "total_episodes": len(records),
        "visual_ood": visual_ood,
        "simulation": {
            "shader": str(args.shader),
            "render_backend": str(args.render_backend),
        },
        "execution_horizon": int(args.execution_horizon),
        "video": {
            "camera": "base_camera overview RGB",
            "fps": int(args.video_fps),
            "keyframe_hold_seconds": 1.0,
            "keyframe_rendering": (
                "a selected frame replaces its ordinary one-frame display with exactly "
                "one second of frozen copies carrying a 6-pixel RGB-red border"
            ),
        },
        "randomness": {
            "seed_scheme": HASHED_SEED_SCHEME if hashed_seed_mode else "sequential-v1",
            "episode_seed_master": (
                int(args.episode_seed_master) if hashed_seed_mode else None
            ),
            "policy_seed_master": (
                int(args.policy_seed_master) if hashed_seed_mode else None
            ),
            "episode_seed_base": None if hashed_seed_mode else EVAL_SEED_BASE,
            "task_seed_stride": None if hashed_seed_mode else EVAL_TASK_SEED_STRIDE,
            "policy_seed_base": None if hashed_seed_mode else POLICY_SEED_BASE,
            "seed_derivation_material": (
                "memory-based-vla/{scheme}/{master_seed}/{namespace}/"
                "{task}/{trial}/{collision_nonce}; first 64 SHA-256 bits mapped "
                "to [1000000000, 2**31); episode and policy seeds jointly unique"
                if hashed_seed_mode
                else None
            ),
            "evaluation_seed_lower_bound_inclusive": EVAL_SEED_LOWER_BOUND,
            "policy_noise_scheme": (
                "numpy.PCG64(SeedSequence([episode_policy_seed, env_frame])); "
                "standard_normal([10,32],float32) for each VLA infer request"
            ),
        },
        "qwen": {
            "model_root": "{memorizer_path}",
            "model_index_sha256": _sha256(qwen_root / "model.safetensors.index.json"),
            "prompt_sidecar": "keyframe_prompt_config.json",
            "prompt_sidecar_sha256": _sha256(prompt_sidecar),
            "prompt_config": prompt_config,
        },
        "vla": {
            # Visual OOD is an environment-side generalization campaign; it
            # does not silently switch the evaluated policy to OOD training.
            "config": vla_config,
            "checkpoint": "{executor_path}",
            "memory_contract_sha256": _sha256(memory_contract_path),
            "memory_contract": memory_contract,
        },
        "manifest": manifest.name,
        "manifest_sha256": _sha256(manifest),
    }
    _atomic_json(config_path, config)
    print(json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def run(args: argparse.Namespace) -> int:
    output = args.output_root.resolve()
    manifest_path = output / "eval_manifest.jsonl"
    config = json.loads((output / "campaign_config.json").read_text())
    expected_simulation = {
        "shader": str(args.shader),
        "render_backend": str(args.render_backend),
    }
    if config.get("simulation") != expected_simulation:
        raise ValueError(
            f"simulation config mismatch: campaign={config.get('simulation')} "
            f"runtime={expected_simulation}"
        )
    if _sha256(manifest_path) != config["manifest_sha256"]:
        raise ValueError("eval_manifest.jsonl changed after campaign preparation")
    records = _load_manifest(manifest_path)
    visual_ood = bool(config.get("visual_ood", False))
    if args.max_trial_exclusive is not None:
        records = [
            row
            for row in records
            if int(row["trial"]) < args.max_trial_exclusive
        ]
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker-index must be in [0, worker-count)")
    assigned = [
        row
        for row in records
        if int(row["record_index"]) % args.worker_count == args.worker_index
    ]

    # Build the pending queue before ordering it.  Ranking only pending records means
    # a resumed task is not pushed behind hundreds of already-completed episodes.
    completed = 0
    pending: list[dict[str, Any]] = []
    pending_rank_by_task: dict[str, int] = {}
    for row in assigned:
        task = str(row["task"])
        trial = int(row["trial"])
        result_path = output / "results" / task / f"trial_{trial:03d}.json"
        if result_path.is_file():
            prior = json.loads(result_path.read_text())
            if _complete_result_matches_manifest(output, row, prior):
                completed += 1
                continue
        row = dict(row)
        row["_pending_rank"] = pending_rank_by_task.get(task, 0)
        pending_rank_by_task[task] = int(row["_pending_rank"]) + 1
        pending.append(row)

    if args.interleave_tasks:
        task_count = len(config["tasks"])
        task_offset = (args.worker_index + 1) % task_count
        task_stride = args.worker_count % task_count
        # With four workers and nine tasks, successive waves should cover
        # tasks 1-4, then 5-8, then 0-3 instead of making workers converge on
        # the same next task.  Fall back to simple rotation if no inverse exists.
        task_stride_inverse = (
            pow(task_stride, -1, task_count)
            if task_stride and math.gcd(task_stride, task_count) == 1
            else 1
        )
        pending.sort(
            key=lambda row: (
                int(row["_pending_rank"]),
                (
                    (int(row["task_index"]) - task_offset)
                    * task_stride_inverse
                )
                % task_count,
            )
        )

    connection = None
    new_completed = 0
    for row in pending:
        row.pop("_pending_rank", None)
        task = str(row["task"])
        trial = int(row["trial"])
        result_path = output / "results" / task / f"trial_{trial:03d}.json"
        staging_video_path = output / "videos" / task / f"trial_{trial:03d}.mp4"
        if result_path.is_file():
            prior = json.loads(result_path.read_text())
            if _complete_result_matches_manifest(output, row, prior):
                completed += 1
                continue
        last_error = None
        for attempt in range(1, args.max_attempts + 1):
            try:
                if connection is None:
                    connection = MemoryAgentConnection(
                        args.host,
                        args.port,
                        open_timeout=args.open_timeout,
                        request_timeout=args.request_timeout,
                    )
                client = load_client(task, visual_ood=visual_ood)(
                    camera_resolution=224,
                    shader=args.shader,
                    env_kwargs={"render_backend": args.render_backend},
                )
                result = client.run_policy(
                    seed=int(row["episode_seed"]),
                    host=args.host,
                    port=args.port,
                    max_steps=args.max_steps,
                    execution_horizon=int(config["execution_horizon"]),
                    connection=connection,
                    video_path=staging_video_path,
                    video_fps=int(config["video"]["fps"]),
                    keyframe_hold_seconds=float(
                        config["video"]["keyframe_hold_seconds"]
                    ),
                    policy_noise_seed=int(row["policy_noise_seed"]),
                )
                result.update(
                    {
                        "status": "complete",
                        "manifest_record": row,
                        "attempt": attempt,
                        "worker_index": args.worker_index,
                        "completed_at_unix": time.time(),
                    }
                )
                final_video_path = _outcome_video_path(
                    output, task, trial, bool(result["success"])
                )
                final_video_path.parent.mkdir(parents=True, exist_ok=True)
                staging_video_path.replace(final_video_path)
                opposite_video_path = _outcome_video_path(
                    output, task, trial, not bool(result["success"])
                )
                opposite_video_path.unlink(missing_ok=True)
                result["video_path"] = final_video_path.relative_to(output).as_posix()
                _atomic_json(result_path, result)
                (output / "errors" / task / f"trial_{trial:03d}.json").unlink(
                    missing_ok=True
                )
                completed += 1
                new_completed += 1
                print(
                    f"worker={args.worker_index} {completed}/{len(assigned)} "
                    f"task={task} trial={trial} success={bool(result['success'])} "
                    f"steps={result['steps']}",
                    flush=True,
                )
                last_error = None
                break
            except BaseException as error:
                last_error = error
                print(
                    f"worker={args.worker_index} task={task} trial={trial} "
                    f"attempt={attempt} error={type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                if connection is not None:
                    connection.close()
                    connection = None
        if last_error is not None:
            error_path = output / "errors" / task / f"trial_{trial:03d}.json"
            _atomic_json(
                error_path,
                {
                    "status": "error",
                    "manifest_record": row,
                    "worker_index": args.worker_index,
                    "error_type": type(last_error).__name__,
                    "error": str(last_error),
                    "failed_at_unix": time.time(),
                },
            )
            if connection is not None:
                connection.close()
            return 1
        if (
            args.max_new_episodes is not None
            and new_completed >= args.max_new_episodes
        ):
            if connection is not None:
                connection.close()
            print(
                f"worker={args.worker_index} recycling after "
                f"{new_completed} new episodes",
                flush=True,
            )
            return 75
    if connection is not None:
        connection.close()
    return 0


def summarize(args: argparse.Namespace) -> int:
    output = args.output_root.resolve()
    manifest_path = output / "eval_manifest.jsonl"
    config = json.loads((output / "campaign_config.json").read_text())
    if _sha256(manifest_path) != config["manifest_sha256"]:
        raise ValueError("eval_manifest.jsonl changed after campaign preparation")
    records = _load_manifest(manifest_path)
    per_task: dict[str, dict[str, Any]] = {}
    incomplete: list[dict[str, Any]] = []
    for task in TASK_NAMES:
        task_rows = [row for row in records if row["task"] == task]
        results = []
        for row in task_rows:
            path = output / "results" / task / f"trial_{int(row['trial']):03d}.json"
            if not path.is_file():
                incomplete.append(row)
                continue
            value = json.loads(path.read_text())
            if not _complete_result_matches_manifest(output, row, value):
                incomplete.append(row)
                continue
            results.append(value)
        successes = sum(bool(value.get("success")) for value in results)
        per_task[task] = {
            "completed": len(results),
            "expected": len(task_rows),
            "successes": successes,
            "success_rate": successes / len(results) if results else None,
        }
    summary = {
        "schema_version": 1,
        "updated_at_unix": time.time(),
        "complete": not incomplete,
        "completed_episodes": len(records) - len(incomplete),
        "expected_episodes": len(records),
        "incomplete_records": incomplete,
        "per_task": per_task,
    }
    _atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if not incomplete else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--output-root", type=Path, required=True)
    prep.add_argument("--qwen-model", type=Path, required=True)
    prep.add_argument("--vla-checkpoint", type=Path, required=True)
    prep.add_argument(
        "--vla-config",
        default="pi05_memory",
        help="OpenPI TrainConfig used to load and validate the VLA checkpoint.",
    )
    prep.add_argument("--trials-per-task", type=int, default=100)
    prep.add_argument("--episode-seed-master", type=int, default=202608100001)
    prep.add_argument("--policy-seed-master", type=int, default=202608100002)
    prep.add_argument("--execution-horizon", type=int, default=10)
    prep.add_argument("--video-fps", type=int, default=20)
    prep.add_argument("--shader", default="rt")
    prep.add_argument("--render-backend", default="cuda")
    prep.add_argument(
        "--visual-ood",
        action="store_true",
        help="Prepare a campaign using each task's visual-OOD environment.",
    )
    prep.set_defaults(function=prepare)

    execute = sub.add_parser("run")
    execute.add_argument("--output-root", type=Path, required=True)
    execute.add_argument("--worker-index", type=int, required=True)
    execute.add_argument("--worker-count", type=int, required=True)
    execute.add_argument("--host", default="127.0.0.1")
    execute.add_argument("--port", type=int, required=True)
    execute.add_argument("--shader", default="rt")
    execute.add_argument("--render-backend", default="cuda")
    execute.add_argument("--max-steps", type=int, default=2000)
    execute.add_argument(
        "--max-trial-exclusive",
        type=int,
        help=(
            "only evaluate manifest rows whose zero-based trial index is below "
            "this value; useful when resuming a campaign under a smaller protocol"
        ),
    )
    execute.add_argument("--max-attempts", type=int, default=3)
    execute.add_argument("--open-timeout", type=float, default=300.0)
    execute.add_argument("--request-timeout", type=float, default=600.0)
    execute.add_argument(
        "--max-new-episodes",
        type=int,
        help="exit with status 75 after this many new episodes so a supervisor can recycle RT resources",
    )
    execute.add_argument(
        "--interleave-tasks",
        action="store_true",
        help="round-robin pending episodes across tasks, staggered by worker index",
    )
    execute.set_defaults(function=run)

    report = sub.add_parser("summarize")
    report.add_argument("--output-root", type=Path, required=True)
    report.set_defaults(function=summarize)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(parsed.function(parsed))
