"""Numerical contract shared by expert generation and rel-EE adapters."""

from __future__ import annotations

import numpy as np


# One normalized unit corresponds to 0.1 m / 0.1 rad in the pinned Panda
# controller.  A 1e-5 grid therefore changes a command by at most 0.5 microns
# or 0.5 microradians while making camera/Rot6D float32 round trips stable.
NATIVE_ACTION_DECIMALS = 5
REPLAY_QUANTIZED_TASKS = frozenset(
    {
        "SO_spring_compression",
        "AT_drawer_match",

        "AST_mine_sweeper",
    }
)


def quantize_native_arm_action(values: np.ndarray) -> np.ndarray:
    """Round normalized Panda arm commands onto the canonical replay grid."""

    return np.round(
        np.asarray(values, dtype=np.float64), decimals=NATIVE_ACTION_DECIMALS
    ).astype(np.float32)


__all__ = [
    "NATIVE_ACTION_DECIMALS",
    "REPLAY_QUANTIZED_TASKS",
    "quantize_native_arm_action",
]
