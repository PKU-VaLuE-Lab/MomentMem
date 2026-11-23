import bisect
import hashlib
import json
import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _decode_frame_with_pyav(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """Decode one RGB frame with PyAV when OpenCV cannot seek an AV1 video."""
    try:
        import av

        with av.open(video_path) as container:
            stream = container.streams.video[0]
            if stream.average_rate is not None and stream.time_base is not None:
                target_pts = int(frame_idx / stream.average_rate / stream.time_base)
                container.seek(target_pts, stream=stream, backward=True, any_frame=False)
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    decoded_idx = round(
                        float(frame.pts * stream.time_base * stream.average_rate)
                    )
                    if decoded_idx == frame_idx:
                        return frame.to_ndarray(format="rgb24")
                    if decoded_idx > frame_idx:
                        break
            else:
                for idx, frame in enumerate(container.decode(stream)):
                    if idx == frame_idx:
                        return frame.to_ndarray(format="rgb24")
                    if idx > frame_idx:
                        break
    except Exception:  # PyAV is an optional fallback; the caller handles decode failure.
        return None
    return None


def _video_codec_name(video_path: str) -> Optional[str]:
    """Read the video codec without decoding frames."""
    try:
        import av

        with av.open(video_path) as container:
            return str(container.streams.video[0].codec_context.name).lower()
    except Exception:  # PyAV is optional; OpenCV remains the default backend.
        return None


@dataclass
class EpisodeContext:
    episode_index: int
    instruction: str
    video_paths: List[str]  # one path per camera, aligned to MomentMemDataset._camera_keys
    orig_len: int
    sub_len: int
    # Original-source frame index for every timestep on this episode's sampled timeline. A fixed
    # stride-4 timeline uses [0, 4, 8, ...]; temporal-stride augmentation uses cumulative random gaps.
    sampled_orig_indices: List[int]
    segments_sub: List[Tuple[int, int]]
    selected_sub: List[int]
    labels: np.ndarray
    loss_mask: np.ndarray
    # Optional mixed teacher/model history produced by Scheduled Sampling.  This affects only
    # what subsequent samples see in their prompt. The accompanying labels/loss masks are
    # recomputed against that same history so the redundancy target remains semantically valid.
    feedback_selected_sub: Optional[List[int]] = None
    feedback_labels: Optional[np.ndarray] = None
    feedback_loss_mask: Optional[np.ndarray] = None


class MomentMemDataset(Dataset):
    """Frame-level dataset for keyframe decision with sparse history input plus an optional
    recent non-current buffer.
    """

    def __init__(
        self,
        dataset_root: str,
        mode: str = "train",
        split_ratio: float = 0.8,
        split_seed: int = 42,
        gamma: float = 0.99,
        video_fps: int = 20,
        sample_fps: Optional[int] = 5,
        transform: Optional[Callable[[Image.Image], Any]] = None,
        include_t0_in_index: bool = True,
        image_size: Optional[int] = 224,
        max_train_episodes: Optional[int] = None,
        max_eval_episodes: Optional[int] = None,
        buffer_size: int = 10,
        max_history: Optional[int] = 48,
        add_memory_instruction: bool = False,
        augment: bool = False,
        aug_brightness: float = 0.3,
        aug_contrast: float = 0.3,
        aug_saturation: float = 0.3,
        aug_hue: float = 0.05,
        aug_crop_scale: float = 0.9,
        allow_blank_frames: bool = False,
        build_supervision: bool = True,
        load_annotations: bool = True,
        camera_mode: str = "auto",
        temporal_stride_augmentation: bool = False,
        temporal_stride_choices: Tuple[int, ...] = (1, 2, 3, 4),
        temporal_stride_probabilities: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
    ) -> None:

        self.dataset_root = os.path.abspath(dataset_root)
        self.mode = mode
        self.split_ratio = split_ratio
        self.split_seed = split_seed
        self.gamma = float(gamma)
        self.video_fps = int(video_fps)
        self.sample_fps = int(sample_fps) if sample_fps is not None else int(video_fps)
        self.transform = transform
        self.include_t0_in_index = include_t0_in_index
        # Working image resolution; None keeps the decoded frame's native dimensions.
        self.image_size = int(image_size) if image_size is not None else None
        # Cap the number of TRAIN episodes per leaf (val split unchanged) — for train-size sweeps.
        self.max_train_episodes = int(max_train_episodes) if max_train_episodes is not None else None
        # Cap the number of VAL episodes per leaf (deterministic subsample) — for fast eval sweeps.
        self.max_eval_episodes = int(max_eval_episodes) if max_eval_episodes is not None else None
        self.buffer_size = int(buffer_size)
        self.max_history = None if max_history is None else int(max_history)
        # Optionally append the dataset's memory instruction to the task.
        self.add_memory_instruction = bool(add_memory_instruction)
        self._warned_missing_memory = False
        # Cache decoded frames reused by many samples.
        self._frame_cache: "OrderedDict[Tuple[str, int], Image.Image]" = OrderedDict()
        self._frame_cache_cap = 4096
        self._pyav_video_paths: set[str] = set()
        self._checked_video_codecs: set[str] = set()
        # Apply one shared augmentation to every image in a sample.
        self.augment = bool(augment)
        self.aug_brightness = float(aug_brightness)
        self.aug_contrast = float(aug_contrast)
        self.aug_saturation = float(aug_saturation)
        self.aug_hue = float(aug_hue)
        self.aug_crop_scale = float(aug_crop_scale)
        # Optionally replace decode failures with blank frames.
        self.allow_blank_frames = bool(allow_blank_frames)
        # Autoregressive evaluation builds memory from model decisions.
        self.build_supervision = bool(build_supervision)
        self.load_annotations = bool(load_annotations)
        self.camera_mode = str(camera_mode).strip().lower()

        self.temporal_stride_augmentation = bool(temporal_stride_augmentation)
        self.temporal_stride_choices = tuple(int(x) for x in temporal_stride_choices)
        raw_stride_probabilities = tuple(float(x) for x in temporal_stride_probabilities)
        probability_sum = sum(raw_stride_probabilities)
        self.temporal_stride_probabilities = tuple(
            weight / probability_sum for weight in raw_stride_probabilities
        )

        if self.video_fps % self.sample_fps != 0:
            raise ValueError(
                f"video_fps ({self.video_fps}) must be divisible by sample_fps ({self.sample_fps})."
            )
        self.stride = self.video_fps // self.sample_fps

        self._episodes_meta = self._load_episodes_meta()
        # {task_text -> memory_text} from meta/tasks.jsonl, for the optional memory instruction.
        self._task_memory_by_text = self._load_task_memory()
        self._segments_by_ep = (
            self._load_segments()
            if self.load_annotations
            else {episode_index: [] for episode_index in self._episodes_meta}
        )
        self._info_meta = self._load_info_meta()
        # Resolve one checkpoint-wide layout: overview only, or overview then wrist.
        self._camera_keys = self._discover_camera_keys()
        self.resolved_camera_mode = (
            "overview+wrist" if len(self._camera_keys) == 2 else "overview"
        )

        self.episodes: List[EpisodeContext] = self._build_episode_contexts()
        self._sampled_orig_indices_by_video_path = {
            video_path: ep.sampled_orig_indices
            for ep in self.episodes
            for video_path in ep.video_paths
        }
        self._offsets: List[int] = self._build_offsets()

        if self.build_supervision:
            self.set_epoch(0)

    def _load_episodes_meta(self) -> Dict[int, Dict[str, Any]]:
        episodes_path = os.path.join(self.dataset_root, "meta", "episodes.jsonl")
        if not os.path.exists(episodes_path):
            raise FileNotFoundError(f"Missing {episodes_path}")

        data: Dict[int, Dict[str, Any]] = {}
        with open(episodes_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ep = int(rec["episode_index"])
                data[ep] = rec
        if not data:
            raise ValueError(f"No episode metadata found in {episodes_path}")
        return data

    def _load_task_memory(self) -> Dict[str, str]:
        """Map task text -> memory text (INSTRUCTION_MEMORY) from meta/tasks.jsonl."""
        tasks_path = os.path.join(self.dataset_root, "meta", "tasks.jsonl")
        mapping: Dict[str, str] = {}
        if not os.path.exists(tasks_path):
            return mapping
        with open(tasks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                task_text = str(rec.get("task", ""))
                mem = rec.get("task_memory")
                if task_text and mem:
                    mapping[task_text] = str(mem)
        return mapping

    def _load_info_meta(self) -> Dict[str, Any]:
        info_path = os.path.join(self.dataset_root, "meta", "info.json")
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Missing {info_path}")
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_segments(self) -> Dict[int, List[Tuple[int, int]]]:
        seg_candidates = [
            os.path.join(self.dataset_root, "meta", "segments.json"),
            os.path.join(self.dataset_root, "meta", "keyframes.json"),
        ]
        seg_path = next((p for p in seg_candidates if os.path.exists(p)), None)
        if seg_path is None:
            raise FileNotFoundError(
                "Missing segment annotation file. Expected meta/segments.json "
                "or meta/keyframes.json."
            )

        with open(seg_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        segments_by_ep: Dict[int, List[Tuple[int, int]]] = {}
        if isinstance(obj, dict) and "episodes" in obj:
            for ep in obj["episodes"]:
                ep_idx = int(ep["episode_index"])
                pairs = ep.get("mem_boundary_pairs", [])
                segments_by_ep[ep_idx] = [(int(s), int(e)) for s, e in pairs]
            return segments_by_ep

        raise ValueError(
            f"Unsupported segment format in {seg_path}. "
            "Expected {'episodes': [{'episode_index':..,'mem_boundary_pairs':..}, ...]}."
        )

    def _discover_camera_keys(self) -> List[str]:
        """Discover videos and resolve the requested one- or two-camera layout."""
        videos_root = Path(self.dataset_root) / "videos" / "chunk-000"
        keys: List[str] = []
        if videos_root.exists():
            keys = sorted(p.name for p in videos_root.iterdir() if p.is_dir())
        if not keys:
            feats = self._info_meta.get("features", {})
            keys = sorted(k for k, v in feats.items() if str(v.get("dtype")) == "video")
        if not keys:
            raise FileNotFoundError(f"No camera/video directories found under {videos_root}.")

        overview = sorted(k for k in keys if "wrist" not in k.lower())
        wrist = sorted(k for k in keys if "wrist" in k.lower())
        if len(overview) != 1:
            raise ValueError(
                f"Expected exactly one overview camera, found {overview} among camera keys {keys}."
            )

        mode = self.camera_mode
        if mode == "overview":
            return [overview[0]]
        if mode == "overview+wrist":
            if len(wrist) != 1:
                raise ValueError(
                    f"camera_mode=overview+wrist requires exactly one wrist camera; "
                    f"found {wrist} among camera keys {keys}."
                )
            return [overview[0], wrist[0]]

        # auto: preserve the old all-camera behavior only when it maps unambiguously to a
        # supported layout. More streams need an explicit future layout, not an accidental prompt.
        if len(wrist) == 0 and len(keys) == 1:
            return [overview[0]]
        if len(wrist) == 1 and len(keys) == 2:
            return [overview[0], wrist[0]]
        raise ValueError(
            f"camera_mode=auto supports one overview camera with zero or one wrist camera; "
            f"found camera keys {keys}. Choose an explicit supported camera layout."
        )

    def _resolve_video_paths(self, ep_idx: int) -> List[str]:
        """One video file per camera key, in self._camera_keys order."""
        chunks_size = int(self._info_meta.get("chunks_size", 1000))
        ep_chunk = ep_idx // chunks_size
        video_pattern = self._info_meta.get("video_path")
        paths: List[str] = []
        for cam in self._camera_keys:
            resolved: Optional[str] = None
            if video_pattern:
                candidate = os.path.join(
                    self.dataset_root,
                    video_pattern.format(
                        episode_chunk=ep_chunk,
                        episode_index=ep_idx,
                        video_key=cam,
                    ),
                )
                if os.path.exists(candidate):
                    resolved = candidate
            if resolved is None:
                # Fallback glob search within this camera's directory.
                cam_dir = Path(self.dataset_root) / "videos" / "chunk-000" / cam
                fallback_name = f"episode_{ep_idx:06d}.mp4"
                for p in cam_dir.rglob(fallback_name):
                    resolved = str(p)
                    break
            if resolved is None:
                raise FileNotFoundError(
                    f"Video not found for episode {ep_idx}, camera '{cam}'."
                )
            paths.append(resolved)
        return paths

    def _get_instruction(self, ep_meta: Dict[str, Any]) -> str:
        tasks = ep_meta.get("tasks", [])
        task_text = str(tasks[0]) if isinstance(tasks, list) and tasks else ""
        if not task_text:
            return ""
        if not self.add_memory_instruction:
            return task_text
        # Use the sole task's memory hint when names differ slightly.
        memory = self._task_memory_by_text.get(task_text)
        if memory is None and len(self._task_memory_by_text) == 1:
            memory = next(iter(self._task_memory_by_text.values()))
        if memory:
            return f"{task_text}\n{memory}"
        # Warn once when the requested memory instruction is unavailable.
        if not self._warned_missing_memory:
            import warnings
            warnings.warn(
                f"add_memory_instruction=True but {self.dataset_root} has no task_memory in "
                "meta/tasks.jsonl (or none matching this episode's task text); the instruction "
                "falls back to TASK-ONLY for this dataset. Regenerate/patch tasks.jsonl with "
                "task_memory if this run is meant to use INSTRUCTION_MEMORY.",
                stacklevel=2,
            )
            self._warned_missing_memory = True
        return task_text

    def _build_episode_contexts(self) -> List[EpisodeContext]:
        all_eps = sorted(
            self._episodes_meta
            if not self.load_annotations
            else set(self._episodes_meta) & set(self._segments_by_ep)
        )
        rng = random.Random(self.split_seed)
        shuffled = all_eps[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        if n <= 1:
            # Reuse a single episode for both splits.
            train_split = val_split = list(shuffled)
        else:
            split_idx = min(max(int(n * self.split_ratio), 1), n - 1)  # keep both splits non-empty
            train_split, val_split = shuffled[:split_idx], shuffled[split_idx:]
        if self.mode == "all":
            # OOD evaluation uses every episode without splitting.
            all_sel = list(shuffled)
            if self.max_eval_episodes is not None:
                all_sel = sorted(all_sel)[: self.max_eval_episodes]
            selected = set(all_sel)
        elif self.mode == "train":
            train_eps = train_split
            if self.max_train_episodes is not None:
                train_eps = train_eps[: self.max_train_episodes]  # fixed val, capped train
            if self.max_eval_episodes is not None:
                # Evaluating ON the training split: deterministic subsample of trained episodes.
                train_eps = sorted(train_eps)[: self.max_eval_episodes]
            selected = set(train_eps)
        else:
            val_eps = val_split
            if self.max_eval_episodes is not None:
                # Deterministic subsample (sorted-by-index head) for fast eval sweeps.
                val_eps = sorted(val_eps)[: self.max_eval_episodes]
            selected = set(val_eps)

        contexts: List[EpisodeContext] = []
        for ep_idx in sorted(selected):
            ep_meta = self._episodes_meta[ep_idx]
            orig_len = int(ep_meta["length"])
            if orig_len <= 0:
                continue

            sampled_orig_indices = self._build_sampled_orig_indices(ep_idx, orig_len)
            sub_len = len(sampled_orig_indices)
            segments_sub = self._map_segments_to_timeline(
                ep_idx, orig_len, sampled_orig_indices
            )
            if segments_sub is None:
                continue

            contexts.append(
                EpisodeContext(
                    episode_index=ep_idx,
                    instruction=self._get_instruction(ep_meta),
                    video_paths=self._resolve_video_paths(ep_idx),
                    orig_len=orig_len,
                    sub_len=sub_len,
                    sampled_orig_indices=sampled_orig_indices,
                    segments_sub=segments_sub,
                    selected_sub=[],
                    labels=np.zeros((sub_len,), dtype=np.float32),
                    loss_mask=np.ones((sub_len,), dtype=np.float32),
                )
            )

        if not contexts:
            raise ValueError("No valid episodes after split/filtering.")
        return contexts

    def _map_segments_to_timeline(
        self,
        ep_idx: int,
        orig_len: int,
        sampled_orig_indices: List[int],
    ) -> Optional[List[Tuple[int, int]]]:
        segments_sub: List[Tuple[int, int]] = []
        for start, end in self._segments_by_ep.get(ep_idx, []):
            if start > end or start >= orig_len:
                return None
            end_clamped = min(end, orig_len - 1)
            s_sub = bisect.bisect_left(sampled_orig_indices, start)
            e_sub = bisect.bisect_right(sampled_orig_indices, end_clamped) - 1
            if s_sub <= e_sub:
                segments_sub.append((s_sub, e_sub))
        return segments_sub

    def _build_sampled_orig_indices(
        self,
        ep_idx: int,
        orig_len: int,
        *,
        epoch: int = 0,
        target_len: Optional[int] = None,
    ) -> List[int]:
        """Build one coherent sampled timeline for an episode."""

        if not self.temporal_stride_augmentation:
            sampled = list(range(0, orig_len, self.stride))
            return sampled

        # A path suffix makes augmentation independent of the absolute mount root.
        root_parts = Path(os.path.normpath(self.dataset_root)).parts
        dataset_identity = "/".join(root_parts[-3:])
        seed_text = f"{self.split_seed}|{dataset_identity}|{int(ep_idx)}"
        if int(epoch) != 0:
            seed_text += f"|{int(epoch)}"
        seed_material = seed_text.encode("utf-8")
        episode_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        rng = random.Random(episode_seed)
        for _ in range(10_000):
            sampled = [0]
            while True:
                gap = rng.choices(
                    self.temporal_stride_choices,
                    weights=self.temporal_stride_probabilities,
                    k=1,
                )[0]
                next_frame = sampled[-1] + int(gap)
                if next_frame >= orig_len:
                    break
                sampled.append(next_frame)
            if target_len is None or len(sampled) == target_len:
                return sampled
        raise RuntimeError(
            f"Unable to sample epoch {epoch} timeline of length {target_len} for episode "
            f"{ep_idx} after 10000 attempts."
        )

    def _build_offsets(self) -> List[int]:
        offsets = [0]
        for ep in self.episodes:
            used_len = ep.sub_len if self.include_t0_in_index else max(ep.sub_len - 1, 0)
            offsets.append(offsets[-1] + used_len)
        return offsets

    def _sample_index_in_segment(self, start: int, end: int, rng: random.Random) -> int:
        if start == end:
            return start
        if self.gamma <= 0:
            return start
        k = np.arange(end - start + 1, dtype=np.float64)
        w = np.power(self.gamma, k)
        p = w / (w.sum() + 1e-12)
        return int(start + rng.choices(range(len(k)), weights=p.tolist(), k=1)[0])

    def _resample_episode_labels(self, ep: EpisodeContext, rng: random.Random) -> None:
        labels = np.zeros((ep.sub_len,), dtype=np.float32)
        loss_mask = np.ones((ep.sub_len,), dtype=np.float32)
        selected = []

        for s, e in ep.segments_sub:
            s_eff = max(0, min(s, ep.sub_len - 1))
            e_eff = max(0, min(e, ep.sub_len - 1))
            if s_eff > e_eff:
                continue

            t_sel = self._sample_index_in_segment(s_eff, e_eff, rng)
            labels[t_sel] = 1.0
            selected.append(t_sel)
            # Counterfactual region before selected frame in the same equivalence segment.
            if t_sel > s_eff:
                loss_mask[s_eff:t_sel] = 0.0

        ep.selected_sub = sorted(set(selected))
        ep.labels = labels
        ep.loss_mask = loss_mask
        # A new counterfactual labeling defines a new sequence.  Never retain predictions made
        # against the previous epoch's labels/history.
        ep.feedback_selected_sub = None
        ep.feedback_labels = None
        ep.feedback_loss_mask = None

    def set_epoch(self, epoch: int) -> None:
        if not self.build_supervision:
            return
        base_seed = self.split_seed + int(epoch) * 1009
        rng = random.Random(base_seed)
        for ep in self.episodes:
            if self.temporal_stride_augmentation:
                sampled = self._build_sampled_orig_indices(
                    ep.episode_index,
                    ep.orig_len,
                    epoch=epoch,
                    target_len=ep.sub_len,
                )
                segments_sub = self._map_segments_to_timeline(
                    ep.episode_index, ep.orig_len, sampled
                )
                ep.sampled_orig_indices[:] = sampled
                ep.segments_sub = segments_sub
            self._resample_episode_labels(ep, rng)

    def __len__(self) -> int:
        return self._offsets[-1]

    def supervised_groups(self) -> Tuple[List[int], List[int], List[int]]:
        """Split supervised samples into positives, hard negatives (inside a memory window), and
        easy negatives.
        """
        pos: List[int] = []
        hard: List[int] = []
        easy: List[int] = []
        for ep_pos, ep in enumerate(self.episodes):
            base = self._offsets[ep_pos]
            used = ep.sub_len if self.include_t0_in_index else max(ep.sub_len - 1, 0)
            windows = ep.segments_sub
            labels, loss_mask = self._supervision_arrays(ep)
            for local in range(used):
                t_sub = local if self.include_t0_in_index else local + 1
                if loss_mask[t_sub] <= 0.0:
                    continue
                gi = base + local
                if labels[t_sub] > 0.5:
                    pos.append(gi)
                elif any(s <= t_sub <= e for (s, e) in windows):
                    hard.append(gi)
                else:
                    easy.append(gi)
        return pos, hard, easy

    def _locate(self, index: int) -> Tuple[int, int]:
        ep_pos = bisect.bisect_right(self._offsets, index) - 1
        local = index - self._offsets[ep_pos]
        t_sub = local if self.include_t0_in_index else local + 1
        return ep_pos, t_sub

    @staticmethod
    def _supervision_arrays(ep: EpisodeContext) -> Tuple[np.ndarray, np.ndarray]:
        """Return labels/mask generated for the exact history shown this epoch."""

        fields = (ep.feedback_selected_sub, ep.feedback_labels, ep.feedback_loss_mask)
        present = [value is not None for value in fields]
        if not any(present):
            return ep.labels, ep.loss_mask
        return ep.feedback_labels, ep.feedback_loss_mask

    def _load_frame_by_sub_idx(self, video_path: str, t_sub: int) -> Image.Image:
        sampled_indices = self._sampled_orig_indices_by_video_path.get(video_path)
        frame_idx = int(sampled_indices[t_sub])
        key = (video_path, frame_idx)
        cached = self._frame_cache.get(key)
        if cached is not None:
            self._frame_cache.move_to_end(key)  # LRU touch
            return cached
        if video_path not in self._checked_video_codecs:
            self._checked_video_codecs.add(video_path)
            if _video_codec_name(video_path) in {"av1", "libdav1d", "libaom-av1"}:
                self._pyav_video_paths.add(video_path)
        ret, frame = False, None
        if video_path not in self._pyav_video_paths:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                self._pyav_video_paths.add(video_path)
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            rgb = _decode_frame_with_pyav(video_path, frame_idx)
            img = Image.fromarray(rgb) if rgb is not None else None
        if img is None:
            if not self.allow_blank_frames:
                raise RuntimeError(
                    f"OpenCV and PyAV failed to decode frame {frame_idx} from {video_path} "
                    "(corrupt/short video?). Pass allow_blank_frames=True to substitute a "
                    "blank frame instead."
                )
            import warnings
            warnings.warn(
                f"OpenCV and PyAV failed to decode frame {frame_idx} from {video_path}; "
                "substituting a blank frame (pass allow_blank_frames=False to fail fast).",
                stacklevel=2,
            )
            if self.image_size is not None:
                blank_size = (self.image_size, self.image_size)
            else:
                cap = cv2.VideoCapture(video_path)
                blank_size = (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
                cap.release()
            return Image.new("RGB", blank_size)
        # Resize images to the model's working resolution.
        if self.image_size is not None and img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        self._frame_cache[key] = img
        if len(self._frame_cache) > self._frame_cache_cap:
            self._frame_cache.popitem(last=False)  # evict LRU
        return img

    def _maybe_transform(self, img: Image.Image) -> Any:
        return self.transform(img) if self.transform is not None else img

    def _sample_aug_params(self):
        """Sample ONE augmentation (color jitter + weak random crop box) for a whole sample."""
        b = random.uniform(max(0.0, 1.0 - self.aug_brightness), 1.0 + self.aug_brightness)
        c = random.uniform(max(0.0, 1.0 - self.aug_contrast), 1.0 + self.aug_contrast)
        s = random.uniform(max(0.0, 1.0 - self.aug_saturation), 1.0 + self.aug_saturation)
        h = random.uniform(-self.aug_hue, self.aug_hue)
        scale = random.uniform(self.aug_crop_scale, 1.0)  # weak crop: keep [scale, 1.0] of each side
        return (b, c, s, h, scale, random.random(), random.random())

    def _apply_aug(self, img: Image.Image, p) -> Image.Image:
        """Apply the shared augmentation params `p` to one PIL image (crop->resize, then jitter)."""
        import torchvision.transforms.functional as TF

        b, c, s, h, scale, dx, dy = p
        w, ht = img.size
        cw, ch = max(1, int(round(w * scale))), max(1, int(round(ht * scale)))
        x0 = int(round((w - cw) * dx))
        y0 = int(round((ht - ch) * dy))
        # weak random crop, then resize back to the original size
        img = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, ht), Image.BILINEAR)
        # color jitter (same factors for every image in the sample)
        img = TF.adjust_brightness(img, b)
        img = TF.adjust_contrast(img, c)
        img = TF.adjust_saturation(img, s)
        img = TF.adjust_hue(img, h)
        return img

    def _load_views_by_sub_idx(self, video_paths: List[str], t_sub: int) -> List[Any]:
        """All camera views for one timestep, in self._camera_keys order (overview, wrist)."""
        return [
            self._maybe_transform(self._load_frame_by_sub_idx(vp, t_sub))
            for vp in video_paths
        ]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ep_pos, t_sub = self._locate(index)
        ep = self.episodes[ep_pos]

        # Exclude buffered keyframes from long-term history.
        if self.buffer_size > 0:
            buffer_sub = list(range(max(0, t_sub - self.buffer_size), t_sub))
        else:
            buffer_sub = []
        buffer_set = set(buffer_sub)
        # Scheduled Sampling may replace the teacher-forced history with a sequence assembled
        # from mixed teacher/model decisions and matching memory-conditioned supervision.
        input_selected_sub = (
            ep.feedback_selected_sub
            if ep.feedback_selected_sub is not None
            else ep.selected_sub
        )
        selected_set = set(input_selected_sub)
        history_sub = [k for k in input_selected_sub if k < t_sub and k not in buffer_set]
        if self.max_history is not None and len(history_sub) > self.max_history:
            history_sub = history_sub[-self.max_history:] if self.max_history > 0 else []
        current_views = self._load_views_by_sub_idx(ep.video_paths, t_sub)
        history_views = [
            self._load_views_by_sub_idx(ep.video_paths, k) for k in history_sub
        ]
        buffer_views = [
            self._load_views_by_sub_idx(ep.video_paths, k) for k in buffer_sub
        ]
        buffer_keyframe_flags = [k in selected_set for k in buffer_sub]

        # Share one augmentation across current, history, and buffer images.
        if self.augment:
            p = self._sample_aug_params()
            current_views = [self._apply_aug(im, p) for im in current_views]
            history_views = [[self._apply_aug(im, p) for im in g] for g in history_views]
            buffer_views = [[self._apply_aug(im, p) for im in g] for g in buffer_views]

        labels, loss_mask = self._supervision_arrays(ep)
        return {
            "instruction": ep.instruction,
            # history_images: list over keyframe timesteps; each item is a list of camera views.
            "history_images": history_views,
            # Buffer entries hold recent views and their keyframe labels.
            "buffer_images": buffer_views,
            "buffer_is_keyframe": buffer_keyframe_flags,
            "buffer_frame_indices_sub": buffer_sub,
            # current_image: list of camera views for the current timestep.
            "current_image": current_views,
            "camera_keys": list(self._camera_keys),
            "num_cameras": len(self._camera_keys),
            "label": torch.tensor(labels[t_sub], dtype=torch.float32),
            "loss_mask": torch.tensor(loss_mask[t_sub], dtype=torch.float32),
            "episode_index": ep.episode_index,
            "frame_index_sub": t_sub,
            "frame_index_orig": int(ep.sampled_orig_indices[t_sub]),
            "history_frame_indices_sub": history_sub,
        }
