"""Geometry and timing for AT_drawer_match."""

CFG = {
    "cabinet_pose": (0.10, 0.18, 0.115),
    "drawer_half_size": (0.085, 0.105, 0.025),
    "drawer_levels": (0.085, 0.0, -0.085),
    "drawer_travel": 0.135,
    # Negative local X points through the drawer opening.  Moving the insert
    # forward makes all 25 high-contrast pattern cells visible when open.
    "card_center_x": -0.030,
    "button_xy": ((-0.18, -0.18), (-0.18, 0.0), (-0.18, 0.18)),
    "button_travel": 0.014,
    "button_threshold": -0.006,
    "drawer_open_threshold": -0.115,
    "drawer_close_threshold": -0.012,
    # The answer press drives the selected upper/middle drawer through its
    # complete physical travel.  The episode terminates on the first frame
    # near the mechanical limit, with no post-opening dwell.
    "answer_drawer_pull": 0.135,
    "answer_drawer_success_threshold": 0.128,
    "open_hold_steps": (24, 34),
    "memory_delay_steps": (16, 24),
}

__all__ = ["CFG"]
