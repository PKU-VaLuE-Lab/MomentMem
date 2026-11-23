"""Small provenance and seed-spacing checks shared by dataset generators."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
from typing import Any


REQUEST_FILENAME = "generation_request.json"


def generation_request(
    *,
    scope: str,
    task: str,
    visual_ood: bool,
    episodes: int,
    base_seed: int,
    steps: int,
    max_attempts: int,
    retries: int,
    shader: str,
    render_backend: str | None,
    save_video: bool,
    shard_size: int | None = None,
    seed_stride: int | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "scope": str(scope),
        "task": str(task),
        "visual_ood": bool(visual_ood),
        "episodes": int(episodes),
        "base_seed": int(base_seed),
        "shard_size": None if shard_size is None else int(shard_size),
        "seed_stride": None if seed_stride is None else int(seed_stride),
        "steps": int(steps),
        "max_attempts": int(max_attempts),
        "retries": int(retries),
        "shader": str(shader),
        "render_backend": None if not render_backend else str(render_backend),
        "camera_resolution": [224, 224],
        "save_video": bool(save_video),
    }


def write_generation_request(
    directory: Path, request: Mapping[str, Any]
) -> Path:
    path = Path(directory) / REQUEST_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(request), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    temporary.replace(path)
    return path


def generation_request_matches(
    directory: Path, expected: Mapping[str, Any]
) -> bool:
    try:
        value = json.loads((Path(directory) / REQUEST_FILENAME).read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return value == dict(expected)


def validate_seed_spacing(
    *,
    episodes: int,
    shard_size: int,
    max_attempts: int,
    retries: int,
    shard_seed_stride: int,
    task_seed_stride: int | None = None,
) -> int:
    values = {
        "episodes": episodes,
        "shard_size": shard_size,
        "max_attempts": max_attempts,
        "shard_seed_stride": shard_seed_stride,
    }
    invalid = [name for name, value in values.items() if int(value) <= 0]
    if invalid:
        raise ValueError("these seed-layout values must be positive: " + ", ".join(invalid))
    if int(retries) < 0:
        raise ValueError("retries must be non-negative")

    retry_span = (int(retries) + 1) * int(max_attempts)
    if int(shard_seed_stride) < retry_span:
        raise ValueError(
            f"shard_seed_stride={shard_seed_stride} is smaller than retry span {retry_span}"
        )
    shard_count = (int(episodes) + int(shard_size) - 1) // int(shard_size)
    task_span = (shard_count - 1) * int(shard_seed_stride) + retry_span
    if task_seed_stride is not None and int(task_seed_stride) < task_span:
        raise ValueError(
            f"task_seed_stride={task_seed_stride} is smaller than task span {task_span}"
        )
    return task_span


def seed_intervals_overlap(
    left_starts: Iterable[int], right_starts: Iterable[int], span: int
) -> bool:
    if int(span) <= 0:
        raise ValueError("seed interval span must be positive")
    return any(
        max(int(left), int(right)) < min(int(left) + int(span), int(right) + int(span))
        for left in left_starts
        for right in right_starts
    )
