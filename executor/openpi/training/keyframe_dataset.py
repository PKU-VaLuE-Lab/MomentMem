"""Attach past visual keyframes to each frame of a LeRobot dataset."""

from __future__ import annotations

import collections
import json
import logging
import pathlib
from typing import Any, SupportsIndex

import numpy as np
from torch.utils.data import get_worker_info

logger = logging.getLogger("openpi")


def load_policy_action_indices(dataset, key: str) -> np.ndarray | None:
    """Read valid lower-policy row indices without decoding any videos."""

    features = getattr(dataset, "features", {}) or {}
    if key not in features:
        return None
    hf_dataset = getattr(dataset, "hf_dataset", None)
    values = hf_dataset[key]
    mask = np.asarray(
        [bool(np.asarray(value).reshape(-1)[0]) for value in values],
        dtype=bool,
    )
    indices = np.flatnonzero(mask).astype(np.int64, copy=False)
    return indices


class PolicyActionDataset:
    """Expose only rows that carry valid VLA action supervision."""

    def __init__(self, dataset, valid_indices: np.ndarray):
        self._dataset = dataset
        indices = np.asarray(valid_indices, dtype=np.int64).reshape(-1)
        self._indices = indices

    @property
    def valid_indices(self) -> np.ndarray:
        return self._indices.copy()

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        local_index = int(index.__index__())
        if local_index < 0:
            local_index += len(self)
        return dict(self._dataset[int(self._indices[local_index])])


def load_keyframe_windows(path: str | pathlib.Path) -> dict[int, list[tuple[int, int]]]:
    """Load episode_index -> list[(start, end)] windows from a keyframes.json/segments.json."""
    obj = json.loads(pathlib.Path(path).read_text())
    windows: dict[int, list[tuple[int, int]]] = {}
    for ep in obj["episodes"]:
        ep_idx = int(ep["episode_index"])
        pairs = ep.get("mem_boundary_pairs", [])
        windows[ep_idx] = [(int(s), int(e)) for s, e in pairs]
    return windows


def _keyframe_frame(s: int, e: int, mode: str, *, rng: np.random.Generator | None = None, gamma: float = 1.0) -> int:
    if mode == "start":
        return s
    if mode == "end":
        return e
    if mode == "sample":
        # Mirror the upper selectors' window sampling (memorizer): p(k) ∝ gamma**k over
        # k in [0, e-s], frame = s + k. gamma==1 -> uniform over the window; gamma<1 -> early-biased.
        if e <= s or gamma <= 0:
            return s
        if rng is None:
            rng = np.random.default_rng()
        k = np.arange(e - s + 1, dtype=np.float64)
        w = np.power(float(gamma), k)
        p = w / w.sum()
        return int(s + int(rng.choice(len(k), p=p)))
    return (s + e) // 2  # "mid"


class KeyframeHistoryDataset:
    """Wraps a random-access dataset, adding past-keyframe images to each item."""

    def __init__(
        self,
        dataset,
        windows: dict[int, list[tuple[int, int]]],
        *,
        num_keyframes: int,
        base_image_key: str = "observation.images.image",
        wrist_image_key: str = "observation.images.wrist_image",
        use_wrist_camera: bool = True,
        keyframe_point: str = "mid",
        keyframe_gamma: float = 1.0,
        non_keyframe_memory_prob: float = 0.0,
        seed: int = 0,
    ):
        self._dataset = dataset
        self._windows = windows
        self._k = int(num_keyframes)
        self._base_key = base_image_key
        self._wrist_key = wrist_image_key
        self._use_wrist_camera = bool(use_wrist_camera)
        self._point = keyframe_point
        self._gamma = float(keyframe_gamma)
        self._non_keyframe_prob = float(non_keyframe_memory_prob)
        self._seed = int(seed)
        # gidx -> (base, optional wrist), LRU-evicted at _cache_cap so
        # late-episode keyframes remain cached without decoding unused views.
        self._cache: collections.OrderedDict[
            int, tuple[np.ndarray, np.ndarray | None]
        ] = collections.OrderedDict()
        self._cache_cap = 4096
        # A stateful RNG keeps per-access augmentation while remaining reproducible for a fixed
        # training seed, repo order, and worker count.
        self._rng: np.random.Generator | None = None

    def _sample_rng(self) -> np.random.Generator:
        if self._rng is None:
            worker = get_worker_info()
            self._rng = np.random.default_rng(
                [self._seed, 0 if worker is None else worker.id]
            )
        return self._rng

    def __len__(self) -> int:
        return len(self._dataset)

    def _frame_images(self, gidx: int) -> tuple[np.ndarray, np.ndarray | None]:
        cached = self._cache.get(gidx)
        if cached is not None:
            self._cache.move_to_end(gidx)
            return cached
        item = self._dataset[gidx]
        base = np.asarray(item[self._base_key])
        wrist = (
            np.asarray(item[self._wrist_key]) if self._use_wrist_camera else None
        )
        self._cache[gidx] = (base, wrist)
        if len(self._cache) > self._cache_cap:  # bound memory; evict least-recently-used
            self._cache.popitem(last=False)
        return base, wrist

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        item = dict(self._dataset[index])
        ep = int(item["episode_index"])
        frame_index = int(item["frame_index"])
        ep_start_global = int(item["index"]) - frame_index  # global index of this episode's frame 0

        # A keyframe becomes visible once its representative frame has passed (kf < current frame),
        # matching the online flow (the VLM selects a frame mid-window and the VLA sees it from then
        # on) and memorizer's `history = keyframes with k < t` convention. Gating on the window END
        # instead would delay visibility until the whole window completes, creating a train/inference
        # mismatch. Oldest..newest, capped to the most recent K.
        rng = (
            self._sample_rng()
            if self._point == "sample" or self._non_keyframe_prob > 0.0
            else None
        )
        kf_candidates = sorted(
            kf
            for kf in (
                _keyframe_frame(s, e, self._point, rng=rng, gamma=self._gamma)
                for (s, e) in self._windows.get(ep, [])
            )
            if kf < frame_index
        )
        if self._non_keyframe_prob > 0.0 and frame_index > 0:
            # A semantic keyframe window represents one event, so every frame in
            # [start, end] is excluded from false-positive sampling (not merely
            # the representative frame sampled above). Each remaining past frame
            # independently becomes a memory with the configured probability.
            non_keyframe_mask = np.ones((frame_index,), dtype=bool)
            for start, end in self._windows.get(ep, []):
                lo = max(0, start)
                hi = min(frame_index - 1, end)
                if lo <= hi:
                    non_keyframe_mask[lo : hi + 1] = False
            non_keyframes = np.flatnonzero(non_keyframe_mask)
            if len(non_keyframes):
                selected = rng.random(len(non_keyframes)) < self._non_keyframe_prob
                kf_candidates.extend(non_keyframes[selected].tolist())
                kf_candidates.sort()
        kf_frames = kf_candidates[-self._k :]

        base_ref = np.asarray(item[self._base_key])
        base_slots = np.zeros((self._k, *base_ref.shape), dtype=base_ref.dtype)
        wrist_slots = None
        if self._use_wrist_camera:
            wrist_ref = np.asarray(item[self._wrist_key])
            wrist_slots = np.zeros(
                (self._k, *wrist_ref.shape), dtype=wrist_ref.dtype
            )
        mask = np.zeros((self._k,), dtype=bool)
        for j, kf in enumerate(kf_frames):
            b, w = self._frame_images(ep_start_global + int(kf))
            base_slots[j] = b
            if wrist_slots is not None:
                wrist_slots[j] = w
            mask[j] = True

        item["keyframe_image_base"] = base_slots
        if wrist_slots is not None:
            item["keyframe_image_wrist"] = wrist_slots
        item["keyframe_mask"] = mask
        return item
