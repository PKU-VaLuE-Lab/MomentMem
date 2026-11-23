"""SO_spring_compression task registration."""

from .SO_spring_compression import (
    ENV_ID,
    INSTRUCTION_MEMORY,
    INSTRUCTION_TASK,
    SOSpringCompression,
    TASK_NAME,
)

__all__ = [
    "SOSpringCompression",
    "ENV_ID",
    "TASK_NAME",
    "INSTRUCTION_TASK",
    "INSTRUCTION_MEMORY",
]
