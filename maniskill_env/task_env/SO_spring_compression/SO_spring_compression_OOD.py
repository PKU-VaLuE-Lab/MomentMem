"""Visual-OOD variant of SO_spring_compression; mechanics are unchanged."""

from mani_skill.utils.registration import register_env

from SO_spring_compression import SOSpringCompression
from common import TASK_OOD_ENV_IDS

ENV_ID = TASK_OOD_ENV_IDS["SO_spring_compression"]


@register_env(ENV_ID, max_episode_steps=700)
class SOSpringCompressionOOD(SOSpringCompression):
    VISUAL_OOD = True


__all__ = ["SOSpringCompressionOOD", "ENV_ID"]
