from AD_scroll_discovery import ENV_ID, INSTRUCTION_MEMORY, INSTRUCTION_TASK, TASK_NAME
from eval_client.base_task_client import BaseTaskClient


class ADScrollDiscoveryClient(BaseTaskClient):
    ENV_ID = ENV_ID
    TASK_NAME = TASK_NAME
    INSTRUCTION_TASK = INSTRUCTION_TASK
    INSTRUCTION_MEMORY = INSTRUCTION_MEMORY
    MIN_KEYFRAMES = 3
    SUCCESS_INFO_KEY = "evaluation_success"
    EVALUATION_INFO_KEYS = (
        "evaluation_success",
        "evaluation_stable_steps",
        "stacked_above",
        "paired_scroll_released",
        "wrong_pair_ever_touched",
    )
