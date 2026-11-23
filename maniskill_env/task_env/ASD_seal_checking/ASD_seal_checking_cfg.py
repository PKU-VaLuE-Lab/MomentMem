"""Configuration for ASD_seal_checking."""

CFG = {
    # Carried-object workspace with visible elbow bend and singularity margin.
    "object_xy_low": (-0.080, -0.205),
    "object_xy_high": (0.080, 0.205),
    "seal_min_separation": 0.205,
    "seal_cube_min_separation": 0.115,
    # A compound stamp silhouette: broad contact foot, narrower centre body,
    # and a small square handle that the gripper actually pinches.
    "seal_base_radius": 0.044,
    "seal_base_half_height": 0.010,
    "seal_stem_radius": 0.019,
    "seal_stem_half_height": 0.014,
    "seal_handle_half_size": (0.012, 0.012, 0.018),
    "seal_mass": 0.11,
    # Large enough to be read clearly at 224 px, while remaining completely
    # covered by the seal's 4.4 cm-radius physical foot before pickup.
    "led_radius": 0.026,
    "led_travel": 0.022,
    "led_emission_strength": 0.22,
    # Fade toward a transparent, non-emissive achromatic neutral, never toward
    # white or a tabletop-specific hue.  The real prismatic disk then lowers
    # below the table during the final part of the fade so no geometry remains.
    "fade_neutral_rgb": (0.35, 0.35, 0.35),
    # End once every candidate color is difficult to read and before disk
    # retraction can increase apparent chroma.
    "keyframe_end_fade_progress": 0.65,
    "keyframe_end_appearance_progress": 0.96,
    "led_retract_start_progress": 0.80,
    # Start the keyframe/fade at the first camera-visible coloured edge, not
    # only after the whole disk is unobstructed.  Across the randomized XY
    # layouts, first visibility occurs at roughly 12--23 mm foot clearance.
    # A 20 mm physical trigger covers every first-visible RGB frame; in the
    # less oblique views it may conservatively include one onset frame before
    # the first coloured pixel rather than omitting memory information.
    "reveal_lift_clearance": 0.020,
    "reveal_uncover_distance": 0.064,
    # Do not consume the cue's fade while the broad stamp foot still obscures
    # most of it.  The keyframe begins at the onset threshold above, but the
    # material remains fully saturated until this second, fully-readable
    # physical threshold is crossed.
    "fade_lift_clearance": 0.050,
    "fade_uncover_distance": 0.070,
    # Camera-visible tray region with robust top-down reach and joint margin.
    "tray_center_x": (-0.220, -0.170),
    "tray_center_y": (-0.115, 0.115),
    "object_tray_clearance_xy": (0.145, 0.185),
    "tray_slots_y": (-0.052, 0.052),
    "tray_floor_top": 0.012,
    "tray_release_clearance": 0.002,
    "seal_release_dwell_steps": 3,
    # Approach the first stamp from home, then transfer directly to the second
    # with enough clearance for the handle and tray walls.
    "initial_stamp_approach_clearance": 0.120,
    "inter_stamp_approach_clearance": 0.060,
    "cube_spawn_min_separation": 0.105,
    # Number of control transitions for the complete colour/alpha fade.  The
    # keyframe deliberately ends earlier, once the cue is too faint to read.
    "reveal_steps": (28, 36),
    "memory_delay_steps": (16, 24),
    "placement_tolerance": 0.045,
    "placement_contact_force_threshold": 0.05,
    "placement_linear_speed_tolerance": 0.03,
    "placement_angular_speed_tolerance": 1.0,
    "placement_stable_steps": 4,
    # The tray floor half extents match the real collision floor.  A seal is
    # in the tray only when its complete circular base fits inside this area.
    "tray_floor_half_xy": (0.083, 0.125),
    "seal_tray_contact_force_threshold": 0.05,
}

__all__ = ["CFG"]
