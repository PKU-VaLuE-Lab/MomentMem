"""Dependency-free task registry and canonical output layout."""

from pathlib import Path

TASK_NAMES = (
    "AO_hidden_pattern",
    "SO_spring_compression",
    "ASO_placement_imitation",
    "AT_drawer_match",
    "ST_flash_beacon",
    "AST_mine_sweeper",
    "AD_scroll_discovery",
    "SD_explore_cube",
    "ASD_seal_checking",
)
ALL_TASK_NAMES = TASK_NAMES
TASK_SPECS = {
    name: {
        "module": name,
        "env_id": f"{name}-v3",
        "visual_ood_module": f"{name}_OOD",
        "visual_ood_env_id": f"{name}-visual-ood-v3",
    }
    for name in ALL_TASK_NAMES
}
ENV_ID_TO_TASK = {
    env_id: name
    for name, spec in TASK_SPECS.items()
    for env_id in (spec["env_id"], spec["visual_ood_env_id"])
}


def lerobot_task_path(dataset_root: Path | str, task_name: str) -> Path:
    """Return ``<dataset-root>/lerobot_format/<task-name>`` for a registered task."""
    if task_name not in TASK_SPECS:
        raise ValueError(f"unknown canonical task name: {task_name!r}")
    return Path(dataset_root) / "lerobot_format" / task_name
