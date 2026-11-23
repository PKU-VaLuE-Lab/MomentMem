"""Dependency-free LeRobot dataset manifests with symlink-aware schema v2."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any


DATASET_MANIFEST_SCHEMA_VERSION = 2
KEYFRAME_LABEL_FILES = {
    "meta/keyframes.json",
    "meta/segments.json",
    "meta/vlm_keyframe_generation.json",
}


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_entries(repo_root: Path) -> list[dict[str, Any]]:
    files = [
        path
        for path in sorted(repo_root.rglob("*"))
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(repo_root).parts)
    ]
    return [
        {
            "path": path.relative_to(repo_root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in files
    ]


def _logical_entries(repo_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def visit(directory: Path, logical: Path, ancestors: frozenset[Path]) -> None:
        try:
            resolved = directory.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(f"broken symbolic link in dataset: {directory}") from exc
        if resolved in ancestors:
            raise ValueError(f"symbolic-link directory cycle in dataset: {directory}")
        current_ancestors = ancestors | {resolved}
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            relative = logical / path.name
            if any(part.startswith(".") for part in relative.parts):
                continue
            if path.is_symlink():
                try:
                    target = path.resolve(strict=True)
                except FileNotFoundError as exc:
                    raise ValueError(f"broken symbolic link in dataset: {path}") from exc
                if target.is_dir():
                    visit(path, relative, current_ancestors)
                    continue
                if not target.is_file():
                    raise ValueError(f"unsupported symbolic-link target in dataset: {path}")
            elif path.is_dir():
                visit(path, relative, current_ancestors)
                continue
            elif not path.is_file():
                continue
            entries.append(
                {
                    "path": relative.as_posix(),
                    "size": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )

    visit(repo_root, Path(), frozenset())
    return entries


def _summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "file_count": len(entries),
        "total_bytes": sum(entry["size"] for entry in entries),
        "content_sha256": _json_sha256(entries),
    }


def build_dataset_manifest(
    repo_ids: Sequence[str],
    *,
    dataset_variant: str,
    root: str | Path | None = None,
    schema_version: int = DATASET_MANIFEST_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Fingerprint local LeRobot repositories using the requested schema."""

    if root is None:
        root = os.environ.get("HF_LEROBOT_HOME") or os.environ.get("LEROBOT_HOME")
    if not root:
        raise ValueError(
            "HF_LEROBOT_HOME or LEROBOT_HOME is required to fingerprint training datasets"
        )

    repos: list[dict[str, Any]] = []
    for repo_id in repo_ids:
        repo_root = Path(root) / str(repo_id)
        if not repo_root.is_dir():
            raise FileNotFoundError(f"LeRobot dataset not found for manifest: {repo_root}")
        entries = (
            _legacy_entries(repo_root)
            if schema_version == 1
            else _logical_entries(repo_root)
        )
        if schema_version == 1:
            repos.append({"repo_id": str(repo_id), **_summary(entries)})
            continue
        labels = [entry for entry in entries if entry["path"] in KEYFRAME_LABEL_FILES]
        trajectory = [entry for entry in entries if entry["path"] not in KEYFRAME_LABEL_FILES]
        repos.append(
            {
                "repo_id": str(repo_id),
                "trajectory": _summary(trajectory),
                "keyframe_labels": _summary(labels),
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": int(schema_version),
        "dataset_variant": str(dataset_variant),
        "repo_ids": [str(repo_id) for repo_id in repo_ids],
        "repos": repos,
    }
    if schema_version == DATASET_MANIFEST_SCHEMA_VERSION:
        manifest["trajectory_manifest_sha256"] = _json_sha256(
            [
                {"repo_id": repo["repo_id"], **repo["trajectory"]}
                for repo in repos
            ]
        )
        manifest["keyframe_label_manifest_sha256"] = _json_sha256(
            [
                {"repo_id": repo["repo_id"], **repo["keyframe_labels"]}
                for repo in repos
            ]
        )
    return {**manifest, "manifest_sha256": _json_sha256(manifest)}


__all__ = [
    "DATASET_MANIFEST_SCHEMA_VERSION",
    "KEYFRAME_LABEL_FILES",
    "build_dataset_manifest",
]
