"""Self-contained ManiSkill-to-LeRobot converter for the v3 benchmark."""

from .specs import ALL_TASK_NAMES, TASK_NAMES, TASK_SPECS

__all__ = [
    "ALL_TASK_NAMES",
    "TASK_NAMES",
    "TASK_SPECS",
    "convert",
    "register_task_environment",
]


def __getattr__(name):
    if name in {"convert", "register_task_environment"}:
        from .convert import convert, register_task_environment

        return {
            "convert": convert,
            "register_task_environment": register_task_environment,
        }[name]
    raise AttributeError(name)
