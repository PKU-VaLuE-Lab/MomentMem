"""Generate physical ST_flash_beacon sanity trajectories."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK_ENV_ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(TASK_ENV_ROOT)]

from ST_flash_beacon import ENV_ID, TASK_NAME  # noqa: E402,F401
from ST_flash_beacon_OOD import ENV_ID as VISUAL_OOD_ENV_ID  # noqa: E402,F401
from generation import task_generation_main  # noqa: E402

if __name__ == "__main__":
    task_generation_main(
        task_name=TASK_NAME, env_id=ENV_ID, visual_ood_env_id=VISUAL_OOD_ENV_ID
    )
