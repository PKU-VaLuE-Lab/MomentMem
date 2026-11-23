"""MemoryAgent client declaration for AO_hidden_pattern."""

import sys
from pathlib import Path

V3_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for path in (V3_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from AO_hidden_pattern import (  # noqa: E402,F401
    ENV_ID,
    INSTRUCTION_MEMORY,
    INSTRUCTION_TASK,
    TASK_NAME,
)
from eval_client.base_task_client import BaseTaskClient  # noqa: E402


class AOHiddenPatternClient(BaseTaskClient):
    ENV_ID = ENV_ID
    TASK_NAME = TASK_NAME
    INSTRUCTION_TASK = INSTRUCTION_TASK
    INSTRUCTION_MEMORY = INSTRUCTION_MEMORY
    ROBOT_UIDS = "panda"
    MIN_KEYFRAMES = 2

    def policy_controls_action(self, task_env) -> bool:
        return int(task_env.phase_id[0].item()) == 2


__all__ = ["AOHiddenPatternClient"]
