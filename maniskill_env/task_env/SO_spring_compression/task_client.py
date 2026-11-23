"""MemoryAgent client declaration for SO_spring_compression."""

import sys
from pathlib import Path

V3_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for path in (V3_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from SO_spring_compression import (  # noqa: E402,F401
    ENV_ID,
    INSTRUCTION_MEMORY,
    INSTRUCTION_TASK,
    TASK_NAME,
)
from eval_client.base_task_client import BaseTaskClient  # noqa: E402


class SOSpringCompressionClient(BaseTaskClient):
    ENV_ID = ENV_ID
    TASK_NAME = TASK_NAME
    INSTRUCTION_TASK = INSTRUCTION_TASK
    INSTRUCTION_MEMORY = INSTRUCTION_MEMORY
    ROBOT_UIDS = "panda"
    MIN_KEYFRAMES = 1
    SUCCESS_INFO_KEY = "evaluation_success"
    EVALUATION_INFO_KEYS = (
        "evaluation_success",
        "evaluation_push_started",
        "evaluation_push_released",
        "evaluation_max_compression",
        "observed_compression",
        "evaluation_max_compression_error",
    )

    def policy_controls_action(self, task_env) -> bool:
        return int(task_env.phase_id[0].item()) == 2


__all__ = ["SOSpringCompressionClient"]
