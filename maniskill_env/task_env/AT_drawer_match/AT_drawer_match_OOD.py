"""Visual-OOD variant of AT_drawer_match; task mechanics are unchanged."""

from mani_skill.utils.registration import register_env

from AT_drawer_match import ATDrawerMatch
from common import TASK_OOD_ENV_IDS

ENV_ID = TASK_OOD_ENV_IDS["AT_drawer_match"]


@register_env(ENV_ID, max_episode_steps=1300)
class ATDrawerMatchOOD(ATDrawerMatch):
    VISUAL_OOD = True


__all__ = ["ATDrawerMatchOOD", "ENV_ID"]
