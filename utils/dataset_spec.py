"""Shared dataset resolver for the System-2 keyframe selectors."""
from __future__ import annotations

import hashlib
import json
import os
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Union

ResolvedLeaf = namedtuple("ResolvedLeaf", ["task_name", "leaf_dir", "dataset_root"])

CANONICAL_TASK_NAMES = (
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


def load_spec(spec: Union[str, dict, "os.PathLike"]) -> Dict[str, List[str]]:
    """Normalize a dict, file, or inline JSON string to ``{dataset_root: tasks}``."""
    if isinstance(spec, dict):
        raw = spec
    elif isinstance(spec, (str, os.PathLike)):
        s = os.fspath(spec)
        if os.path.isfile(s):
            text = Path(s).read_text()
            if s.endswith((".yaml", ".yml")):
                import yaml  # lazy: only needed for YAML spec files
                raw = yaml.safe_load(text)
            else:
                raw = json.loads(text)
        else:
            # treat as an inline JSON string, e.g. '{"/path": ["task_a"]}'
            raw = json.loads(s)
    else:
        raise TypeError(f"unsupported spec type: {type(spec).__name__}")
    out: Dict[str, List[str]] = {}
    for path, tasks in raw.items():
        if isinstance(tasks, str):
            tasks = [tasks]
        out[str(path)] = [str(t) for t in tasks]
    return out


def _dataset_root(path: Union[str, "os.PathLike"]) -> Path:
    raw = os.fspath(path)
    expanded = os.path.expanduser(os.path.expandvars(raw))
    if "$" in expanded:
        raise ValueError(f"dataset root contains an unresolved environment variable: {raw!r}")
    root = Path(expanded).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    return root


def _task_leaf(root: Path, task_name: str) -> Path:
    if task_name not in CANONICAL_TASK_NAMES:
        raise ValueError(f"unknown canonical task name: {task_name!r}")
    leaf = root / "lerobot_format" / task_name
    if not (leaf / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"canonical task leaf is missing meta/info.json: {leaf}"
        )
    return leaf.resolve()


def task_name_from_leaf(leaf_dir: Union[str, "os.PathLike"]) -> str:
    """Return the task name from a canonical LeRobot task leaf."""
    leaf = Path(leaf_dir).resolve()
    task_name = leaf.name
    if task_name not in CANONICAL_TASK_NAMES:
        raise ValueError(f"unknown canonical task name: {task_name!r}")
    if not (leaf / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"canonical task leaf is missing meta/info.json: {leaf}")
    return task_name


def resolve_dataset_root(
    dataset_root: Union[str, "os.PathLike"],
) -> List[ResolvedLeaf]:
    """Resolve every task immediately below ``<dataset-root>/lerobot_format``."""
    root = _dataset_root(dataset_root)
    lerobot_root = root / "lerobot_format"
    if not lerobot_root.is_dir():
        raise FileNotFoundError(f"canonical lerobot_format directory does not exist: {lerobot_root}")
    task_names = sorted(
        path.name
        for path in lerobot_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if not task_names:
        raise ValueError(f"no task directories found under {lerobot_root}")
    return [
        ResolvedLeaf(task_name, str(_task_leaf(root, task_name)), str(root))
        for task_name in task_names
    ]


def resolve_spec(spec: Union[str, dict, "os.PathLike"]) -> List[ResolvedLeaf]:
    """Resolve a spec to ``ResolvedLeaf(task_name, leaf_dir, dataset_root)`` entries."""
    mapping = load_spec(spec)
    resolved: List[ResolvedLeaf] = []
    for dataset_root, tasks in mapping.items():
        root = _dataset_root(dataset_root)
        if tasks == ["*"]:
            leaves = resolve_dataset_root(root)
            tasks = [leaf.task_name for leaf in leaves]
        for task_name in tasks:
            leaf = _task_leaf(root, task_name)
            resolved.append(ResolvedLeaf(task_name, str(leaf), str(root)))
    if not resolved:
        raise ValueError(f"dataset spec resolved to no datasets: {mapping!r}")
    seen = set()
    for r in resolved:
        seen.add(r.leaf_dir)
    return resolved


def stable_task_id(name: str) -> int:
    """Keep task identity stable across dataset subsets and paths."""
    h = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16)
    return 1000 + (h % 1_000_000)


def task_id_map(resolved: List[ResolvedLeaf]) -> Dict[str, int]:
    """task_name -> STABLE task_id (hash of the name). Same name across paths => same id (merged);
    the same name always gets the same id regardless of the rest of the spec."""
    return {name: stable_task_id(name) for name in sorted({r.task_name for r in resolved})}


def resolve_with_task_ids(spec: Union[str, dict, "os.PathLike"]) -> List[tuple]:
    """Return a list of (task_id, task_name, leaf_dir). Same task_name shares task_id (merged)."""
    resolved = resolve_spec(spec)
    tid = task_id_map(resolved)
    return [(tid[r.task_name], r.task_name, r.leaf_dir) for r in resolved]
