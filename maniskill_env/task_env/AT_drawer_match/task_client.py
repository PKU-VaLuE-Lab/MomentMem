from AT_drawer_match import ENV_ID, INSTRUCTION_MEMORY, INSTRUCTION_TASK, TASK_NAME
from eval_client.base_task_client import BaseTaskClient


class ATDrawerMatchClient(BaseTaskClient):
    ENV_ID = ENV_ID
    TASK_NAME = TASK_NAME
    INSTRUCTION_TASK = INSTRUCTION_TASK
    INSTRUCTION_MEMORY = INSTRUCTION_MEMORY
    MIN_KEYFRAMES = 3
