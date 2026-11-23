"""Visual-OOD variant of ASD_seal_checking; task mechanics are unchanged."""

from mani_skill.utils.registration import register_env

from ASD_seal_checking import ASDSealChecking
from common import TASK_OOD_ENV_IDS

ENV_ID = TASK_OOD_ENV_IDS["ASD_seal_checking"]


@register_env(ENV_ID, max_episode_steps=1700)
class ASDSealCheckingOOD(ASDSealChecking):
    VISUAL_OOD = True


__all__ = ["ASDSealCheckingOOD", "ENV_ID"]
