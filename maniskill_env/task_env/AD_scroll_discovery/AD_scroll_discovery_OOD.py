"""Visual-OOD variant of AD_scroll_discovery; mechanics are unchanged."""

from mani_skill.utils.registration import register_env

from AD_scroll_discovery import ADScrollDiscovery
from common import TASK_OOD_ENV_IDS

ENV_ID = TASK_OOD_ENV_IDS["AD_scroll_discovery"]


@register_env(ENV_ID, max_episode_steps=1800)
class ADScrollDiscoveryOOD(ADScrollDiscovery):
    VISUAL_OOD = True


__all__ = ["ADScrollDiscoveryOOD", "ENV_ID"]
