"""Visual-OOD variant of ASO_placement_imitation; mechanics are unchanged."""

from mani_skill.utils.registration import register_env

from ASO_placement_imitation import ASOPlacementImitation
from common import TASK_OOD_ENV_IDS

ENV_ID = TASK_OOD_ENV_IDS["ASO_placement_imitation"]


@register_env(ENV_ID, max_episode_steps=1300)
class ASOPlacementImitationOOD(ASOPlacementImitation):
    VISUAL_OOD = True


__all__ = ["ASOPlacementImitationOOD", "ENV_ID"]
