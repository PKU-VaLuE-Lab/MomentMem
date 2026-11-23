"""Visual-OOD variant of SD_explore_cube; task mechanics are unchanged."""

from mani_skill.utils.registration import register_env

from SD_explore_cube import SDExploreCube
from common import TASK_OOD_ENV_IDS

ENV_ID = TASK_OOD_ENV_IDS["SD_explore_cube"]


@register_env(ENV_ID, max_episode_steps=1800)
class SDExploreCubeOOD(SDExploreCube):
    VISUAL_OOD = True


__all__ = ["SDExploreCubeOOD", "ENV_ID"]
