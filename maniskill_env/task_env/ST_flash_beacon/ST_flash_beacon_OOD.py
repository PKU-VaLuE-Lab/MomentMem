"""Visual-OOD variant of ST_flash_beacon; task mechanics are unchanged."""

from mani_skill.utils.registration import register_env

from ST_flash_beacon import STFlashBeacon
from common import TASK_OOD_ENV_IDS

ENV_ID = TASK_OOD_ENV_IDS["ST_flash_beacon"]


@register_env(ENV_ID, max_episode_steps=1100)
class STFlashBeaconOOD(STFlashBeacon):
    VISUAL_OOD = True


__all__ = ["STFlashBeaconOOD", "ENV_ID"]
