"""Visual-OOD variant of AST_mine_sweeper; task mechanics are unchanged."""

from mani_skill.utils.registration import register_env

from AST_mine_sweeper import ASTMineSweeper
from common import TASK_OOD_ENV_IDS

ENV_ID = TASK_OOD_ENV_IDS["AST_mine_sweeper"]


@register_env(ENV_ID, max_episode_steps=1500)
class ASTMineSweeperOOD(ASTMineSweeper):
    VISUAL_OOD = True


__all__ = ["ASTMineSweeperOOD", "ENV_ID"]
