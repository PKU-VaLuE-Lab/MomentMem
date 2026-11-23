from SD_explore_cube import ENV_ID, INSTRUCTION_MEMORY, INSTRUCTION_TASK, TASK_NAME
from eval_client.base_task_client import BaseTaskClient


class SDExploreCubeClient(BaseTaskClient):
    ENV_ID = ENV_ID
    TASK_NAME = TASK_NAME
    INSTRUCTION_TASK = INSTRUCTION_TASK
    INSTRUCTION_MEMORY = INSTRUCTION_MEMORY
    MIN_KEYFRAMES = 3
    SUCCESS_INFO_KEY = "evaluation_success"
    EVALUATION_INFO_KEYS = (
        "evaluation_success",
        "cube_grasped",
        "evaluation_cube_lifted",
        "cube_center_z",
        "cube_bottom_table_clearance",
    )
