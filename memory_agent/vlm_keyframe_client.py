"""HTTP client for the upper-level VLM keyframe selector (a vLLM OpenAI-compatible server)."""
from __future__ import annotations

import base64
import io
import math
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import requests
from PIL import Image

# Reuse the Qwen training and offline inference prompts.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MEMORIZER_ROOT = _REPO_ROOT / "memorizer"
if str(_MEMORIZER_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEMORIZER_ROOT))

from prompts import (  # noqa: E402
    ANSWER_NO,
    ANSWER_YES,
    CHOICES,
    PROMPT_LAYOUT_VERSION,  # noqa: F401 - re-exported for sidecar layout checks
    camera_mode_from_config,  # noqa: F401 - re-exported for sidecar parsing
    build_user_parts,
    image_size_from_config,  # noqa: F401 - re-exported for sidecar parsing
    render_user_content,
    system_prompt,
)


# A keyframe in memory is one timestep's views: (overview, wrist|None).
FramePair = Tuple[np.ndarray, Optional[np.ndarray]]
BufferFrame = Tuple[np.ndarray, Optional[np.ndarray], bool, int]


class VLMKeyframeClient:
    """Query a remote vLLM keyframe model for the "is this a new keyframe?" decision."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "qwen3vl-keyframe",
        api_key: str = "EMPTY",
        use_wrist: bool = True,
        max_history: int | None = 48,
        image_budget: int | None = None,
        image_size: int | None = None,
        timeout: float = 60.0,
        jpeg_quality: int = 90,
    ) -> None:
        self.base_url, self.endpoint = self._resolve_endpoint(base_url)
        self.model = model
        self.api_key = api_key
        self.use_wrist = bool(use_wrist)
        self.num_cameras = 2 if self.use_wrist else 1
        self.max_history = None if max_history is None else int(max_history)
        # Validate requests against the server image limit.
        self.image_budget = None if image_budget is None else int(image_budget)
        # Match the training-time image resolution when configured.
        self.image_size = None if image_size is None else int(image_size)
        self.timeout = float(timeout)
        self.jpeg_quality = int(jpeg_quality)
        # The system prompt's camera-count sentence matches use_wrist (single-camera != "two cameras").
        self.system_text = system_prompt(self.num_cameras)

    def predict(
        self,
        *,
        current_base: np.ndarray,
        current_wrist: Optional[np.ndarray],
        history: Sequence[FramePair],
        buffer: Sequence[BufferFrame] = (),
        prompt: str,
    ) -> float:
        """Return ``P(Yes)`` that the current timestep is a new keyframe, given the memory history."""
        messages = self._build_messages(prompt, history, current_base, current_wrist, buffer=buffer)
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 2,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 20,
            # vLLM extension: constrain decoding to exactly {"Yes", "No"} (top-level JSON field).
            "guided_choice": CHOICES,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = requests.post(self.endpoint, json=body, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return self._parse_p_yes(resp.json())

    # ----------------------------------------------------------------------- #
    @staticmethod
    def _resolve_endpoint(base_url: str) -> tuple[str, str]:
        """Accept either a vLLM base URL or a full completions endpoint."""
        url = str(base_url).rstrip("/")
        if url.endswith("/chat/completions"):
            return url[: -len("/chat/completions")], url
        if url.endswith("/completions"):
            base = url[: -len("/completions")]
            return base, f"{base}/chat/completions"
        return url, f"{url}/chat/completions"

    def _build_messages(
        self,
        prompt: str,
        history: Sequence[FramePair],
        current_base: np.ndarray,
        current_wrist: Optional[np.ndarray],
        buffer: Sequence[BufferFrame] = (),
    ) -> List[dict]:
        history = list(history)
        buffer = list(buffer)
        # Keep recent keyframes within the prompt image limit.
        if self.max_history is not None and len(history) > self.max_history:
            history = history[-self.max_history:] if self.max_history > 0 else []

        num_cameras = 2 if self.use_wrist else 1
        image_items: List[dict] = []
        for base, wrist in history:               # each history keyframe timestep (overview, then wrist)
            image_items.append(self._image_item(base))
            if self.use_wrist:
                image_items.append(self._image_item(wrist if wrist is not None else np.zeros_like(base)))
        for base, wrist, _is_kf, _step in buffer:  # recent buffer timesteps (oldest to newest)
            image_items.append(self._image_item(base))
            if self.use_wrist:
                image_items.append(self._image_item(wrist if wrist is not None else np.zeros_like(base)))
        image_items.append(self._image_item(current_base))   # current timestep views
        if self.use_wrist:
            image_items.append(
                self._image_item(current_wrist if current_wrist is not None else np.zeros_like(current_base))
            )

        user_parts = build_user_parts(
            prompt,
            len(history),
            num_cameras,
            buffer_keyframe_flags=[bool(x[2]) for x in buffer],
            buffer_frame_indices=[int(x[3]) for x in buffer],
        )
        return [
            {"role": "system", "content": [{"type": "text", "text": self.system_text}]},
            {"role": "user", "content": render_user_content(user_parts, image_items)},
        ]

    def _image_item(self, arr: np.ndarray) -> dict:
        return {"type": "image_url", "image_url": {"url": self._to_data_url(arr)}}

    def _to_data_url(self, arr: Any) -> str:
        arr = np.asarray(arr)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if np.issubdtype(arr.dtype, np.floating):
            if arr.size and float(np.nanmax(arr)) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(np.ascontiguousarray(arr)).convert("RGB")
        if self.image_size is not None and img.size != (self.image_size, self.image_size):
            # Use the dataset's training-time resize convention.
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    @staticmethod
    def _parse_p_yes(data: dict) -> float:
        """Recover P(Yes) from the response, looking ONLY at the Yes / No tokens (mirrors qwen
        client).
        """
        choice = data["choices"][0]
        content = (choice.get("message", {}).get("content") or "").strip()
        answer = ANSWER_YES if content.lower().startswith("yes") else ANSWER_NO

        p_yes = 1.0 if answer == ANSWER_YES else 0.0
        try:
            top = choice["logprobs"]["content"][0]["top_logprobs"]
            lp = {t["token"].strip().lower(): t["logprob"] for t in top}
            if "yes" in lp or "no" in lp:
                my = math.exp(lp["yes"]) if "yes" in lp else 0.0
                mn = math.exp(lp["no"]) if "no" in lp else 0.0
                if (my + mn) > 0:
                    p_yes = my / (my + mn)
        except Exception:
            pass
        return float(p_yes)
