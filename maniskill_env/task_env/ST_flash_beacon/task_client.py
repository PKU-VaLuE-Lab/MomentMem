from ST_flash_beacon import ENV_ID, INSTRUCTION_MEMORY, INSTRUCTION_TASK, TASK_NAME
from eval_client.base_task_client import BaseTaskClient


class STFlashBeaconClient(BaseTaskClient):
    ENV_ID = ENV_ID
    TASK_NAME = TASK_NAME
    INSTRUCTION_TASK = INSTRUCTION_TASK
    INSTRUCTION_MEMORY = INSTRUCTION_MEMORY
    MIN_KEYFRAMES = 2
    SUCCESS_INFO_KEY = "evaluation_success"
    EVALUATION_INFO_KEYS = (
        "evaluation_success",
        "evaluation_placements_complete",
        "first_target_position_reached",
        "second_target_position_reached",
    )
