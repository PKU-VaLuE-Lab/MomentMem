"""Visual-OOD variant of AO_hidden_pattern; task mechanics are unchanged."""

from mani_skill.utils.registration import register_env

from AO_hidden_pattern import AOHiddenPattern
from common import TASK_OOD_ENV_IDS

ENV_ID = TASK_OOD_ENV_IDS["AO_hidden_pattern"]


@register_env(ENV_ID, max_episode_steps=800)
class AOHiddenPatternOOD(AOHiddenPattern):
    VISUAL_OOD = True


__all__ = ["AOHiddenPatternOOD", "ENV_ID"]
