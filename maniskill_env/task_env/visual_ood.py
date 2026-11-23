"""Shared visual-domain randomization for the nine benchmark OOD variants."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill import ASSET_DIR
from mani_skill.utils.building.actors.ycb import get_ycb_builder
from mani_skill.utils.io_utils import load_json

try:
    from .common import PhysicsMaterial, render_material
except ImportError:  # flat task-local launcher
    from common import PhysicsMaterial, render_material


OOD_SCHEMA_VERSION = 2

YCB_MODEL_INFO_PATH = ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"
YCB_SOURCE_URL = (
    "https://huggingface.co/datasets/haosulab/ManiSkill2/"
    "resolve/main/data/mani_skill2_ycb.zip"
)
YCB_LICENSE_NOTICE = (
    "Derived from the YCB Object and Model Set (CC BY 4.0); preserve source "
    "attribution when redistributing rendered data or processed assets."
)

# Curated compact objects keep their collision geometry outside the benchmark
# manipulation core while providing genuinely textured, scanned household
# clutter. Large tools and long utensils are excluded because their footprint
# could intrude into randomized task layouts. Tabletop and shelf pools are
# deliberately separate: small objects read naturally as nearby clutter,
# while tall packages remain legible in the more distant background.
YCB_TABLETOP_IDS = (
    "002_master_chef_can",
    "005_tomato_soup_can",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "013_apple",
    "014_lemon",
    "015_peach",
    "016_pear",
    "017_orange",
    "018_plum",
    "025_mug",
    "026_sponge",
    "053_mini_soccer_ball",
    "056_tennis_ball",
    "057_racquetball",
    "061_foam_brick",
    "077_rubiks_cube",
)

YCB_REAR_SHELF_IDS = (
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "010_potted_meat_can",
    "021_bleach_cleanser",
    "036_wood_block",
)

_YCB_MODEL_DATABASE: dict[str, Any] | None = None

TABLE_PALETTES = (
    ((0.12, 0.16, 0.19, 1.0), (0.20, 0.27, 0.31, 1.0), "slate_tiles"),
    ((0.72, 0.72, 0.68, 1.0), (0.45, 0.49, 0.50, 1.0), "concrete_panels"),
    ((0.12, 0.31, 0.29, 1.0), (0.08, 0.21, 0.20, 1.0), "green_laminate"),
    ((0.56, 0.66, 0.72, 1.0), (0.20, 0.27, 0.31, 1.0), "blue_ceramic_tiles"),
    ((0.24, 0.25, 0.38, 1.0), (0.38, 0.34, 0.48, 1.0), "studio_laminate"),
)

WALL_PALETTES = (
    ((0.58, 0.66, 0.70, 1.0), (0.18, 0.24, 0.27, 1.0), "blue_grey_lab"),
    ((0.76, 0.72, 0.63, 1.0), (0.35, 0.28, 0.21, 1.0), "warm_workshop"),
    ((0.47, 0.61, 0.53, 1.0), (0.19, 0.31, 0.25, 1.0), "sage_room"),
    ((0.67, 0.58, 0.68, 1.0), (0.29, 0.22, 0.33, 1.0), "mauve_studio"),
    ((0.43, 0.48, 0.58, 1.0), (0.16, 0.19, 0.27, 1.0), "cool_office"),
)

LIGHT_PROFILES = (
    {
        "name": "warm_window",
        "ambient": (0.18, 0.16, 0.13),
        "key_direction": (0.7, 0.35, -1.0),
        "key_color": (1.15, 0.91, 0.68),
        "fill_direction": (-0.55, -0.20, -1.0),
        "fill_color": (0.32, 0.39, 0.48),
        "point_position": (0.10, -0.35, 1.25),
        "point_color": (0.52, 0.40, 0.28),
    },
    {
        "name": "cool_lab",
        "ambient": (0.12, 0.16, 0.21),
        "key_direction": (-0.45, 0.65, -1.0),
        "key_color": (0.69, 0.87, 1.18),
        "fill_direction": (0.55, -0.25, -1.0),
        "fill_color": (0.40, 0.34, 0.29),
        "point_position": (0.35, 0.15, 1.10),
        "point_color": (0.25, 0.34, 0.48),
    },
    {
        "name": "side_lit_studio",
        "ambient": (0.16, 0.13, 0.18),
        "key_direction": (0.15, -0.95, -0.65),
        "key_color": (1.02, 0.76, 0.92),
        "fill_direction": (-0.75, 0.35, -1.0),
        "fill_color": (0.30, 0.46, 0.55),
        "point_position": (-0.40, 0.00, 1.00),
        "point_color": (0.40, 0.28, 0.43),
    },
    {
        "name": "overhead_office",
        "ambient": (0.22, 0.22, 0.20),
        "key_direction": (0.25, 0.20, -1.0),
        "key_color": (0.86, 0.90, 0.82),
        "fill_direction": (-0.65, -0.35, -1.0),
        "fill_color": (0.36, 0.40, 0.48),
        "point_position": (0.00, 0.25, 1.30),
        "point_color": (0.34, 0.36, 0.31),
    },
)


def _visual_box(scene, name: str, half_size: Sequence[float], xyz, color):
    builder = scene.create_actor_builder()
    # Render-only entities disappear from ManiSkill's state registry when the
    # converter recreates an environment with obs_mode="none". A 1-mm static
    # anchor five metres below the scene preserves the actor/state schema in
    # both modes without adding any contact surface near the task.
    builder.add_box_collision(
        pose=sapien.Pose(p=(0.0, 0.0, -5.0)),
        half_size=(0.001, 0.001, 0.001),
        material=PhysicsMaterial().sapien(),
    )
    builder.add_box_visual(
        half_size=np.asarray(half_size, dtype=np.float32),
        material=render_material(color),
    )
    builder.initial_pose = sapien.Pose(p=xyz)
    return builder.build_static(name)


def _static_box(scene, name: str, half_size: Sequence[float], xyz, color):
    hs = np.asarray(half_size, dtype=np.float32)
    builder = scene.create_actor_builder()
    material = PhysicsMaterial(static_friction=0.95, dynamic_friction=0.80)
    builder.add_box_collision(half_size=hs, material=material.sapien())
    builder.add_box_visual(half_size=hs, material=render_material(color))
    builder.initial_pose = sapien.Pose(p=xyz)
    return builder.build_static(name)


def _static_cylinder(scene, name: str, radius: float, half_length: float, xyz, color):
    builder = scene.create_actor_builder()
    material = PhysicsMaterial(static_friction=0.95, dynamic_friction=0.80)
    pose = sapien.Pose(p=xyz, q=euler2quat(0.0, np.pi / 2, 0.0))
    builder.add_cylinder_collision(
        radius=float(radius), half_length=float(half_length), material=material.sapien()
    )
    builder.add_cylinder_visual(
        radius=float(radius), half_length=float(half_length), material=render_material(color)
    )
    builder.initial_pose = pose
    return builder.build_static(name)


def _static_sphere(scene, name: str, radius: float, xyz, color):
    builder = scene.create_actor_builder()
    material = PhysicsMaterial(static_friction=0.95, dynamic_friction=0.80)
    builder.add_sphere_collision(radius=float(radius), material=material.sapien())
    builder.add_sphere_visual(radius=float(radius), material=render_material(color))
    builder.initial_pose = sapien.Pose(p=xyz)
    return builder.build_static(name)


def _ycb_model_database() -> dict[str, Any]:
    global _YCB_MODEL_DATABASE
    if _YCB_MODEL_DATABASE is None:
        if not YCB_MODEL_INFO_PATH.is_file():
            raise FileNotFoundError(
                f"visual OOD requires ManiSkill YCB assets at {YCB_MODEL_INFO_PATH}; "
                "install them with `python -m mani_skill.utils.download_asset ycb -y`"
            )
        _YCB_MODEL_DATABASE = load_json(YCB_MODEL_INFO_PATH)
    return _YCB_MODEL_DATABASE


def _ycb_pose_for_surface(
    model_id: str,
    *,
    center_xy: Sequence[float],
    surface_z: float,
    yaw: float,
) -> tuple[list[float], list[float]]:
    """Place a processed YCB mesh by its true scaled bounding box."""

    metadata = _ycb_model_database()[model_id]
    scale = float(metadata.get("scales", [1.0])[0])
    minimum = np.asarray(metadata["bbox"]["min"], dtype=np.float64) * scale
    maximum = np.asarray(metadata["bbox"]["max"], dtype=np.float64) * scale
    local_center = 0.5 * (minimum + maximum)
    cosine, sine = float(np.cos(yaw)), float(np.sin(yaw))
    rotated_center_xy = np.asarray(
        [
            cosine * local_center[0] - sine * local_center[1],
            sine * local_center[0] + cosine * local_center[1],
        ]
    )
    position = [
        float(center_xy[0] - rotated_center_xy[0]),
        float(center_xy[1] - rotated_center_xy[1]),
        float(surface_z - minimum[2]),
    ]
    quaternion = [float(value) for value in euler2quat(0.0, 0.0, float(yaw))]
    return position, quaternion


def _build_ycb_static(scene, *, name: str, specification: Mapping[str, Any]):
    model_id = str(specification["model_id"])
    builder = get_ycb_builder(scene, model_id, add_collision=True, add_visual=True)
    builder.initial_pose = sapien.Pose(
        p=specification["position"],
        q=specification["quaternion_wxyz"],
    )
    return builder.build_static(name)


def _sample_ycb_assets(rng: np.random.RandomState, layout: int) -> dict[str, Any]:
    table_selected = rng.choice(YCB_TABLETOP_IDS, size=2, replace=False).tolist()
    shelf_selected = rng.choice(YCB_REAR_SHELF_IDS, size=3, replace=False).tolist()
    table_centers = [(-0.365, 0.452), (0.295, 0.452)]
    if layout % 2:
        table_centers.reverse()
    table_specs = []
    for index, (model_id, center) in enumerate(zip(table_selected, table_centers)):
        center = (
            float(center[0] + rng.uniform(-0.012, 0.012)),
            float(center[1] + rng.uniform(-0.006, 0.006)),
        )
        yaw = float(rng.uniform(-np.pi, np.pi))
        position, quaternion = _ycb_pose_for_surface(
            model_id, center_xy=center, surface_z=0.00035, yaw=yaw
        )
        table_specs.append(
            {
                "actor_name": f"ood_ycb_table_{index}",
                "model_id": model_id,
                "position": position,
                "quaternion_wxyz": quaternion,
            }
        )

    shelf_centers = [(-0.39, 0.565), (-0.10, 0.565), (0.27, 0.565)]
    shelf_specs = []
    database = _ycb_model_database()
    for index, (model_id, center) in enumerate(zip(shelf_selected, shelf_centers)):
        metadata = database[model_id]
        scale = float(metadata.get("scales", [1.0])[0])
        extent = (
            np.asarray(metadata["bbox"]["max"], dtype=np.float64)
            - np.asarray(metadata["bbox"]["min"], dtype=np.float64)
        ) * scale
        # Face the object's narrower horizontal dimension into the shallow
        # shelf, then randomly flip its front/back orientation.
        yaw = 0.0 if extent[1] <= extent[0] else np.pi / 2
        yaw += float(rng.randint(2)) * np.pi
        position, quaternion = _ycb_pose_for_surface(
            model_id, center_xy=center, surface_z=0.3245, yaw=yaw
        )
        shelf_specs.append(
            {
                "actor_name": f"ood_ycb_shelf_{index}",
                "model_id": model_id,
                "position": position,
                "quaternion_wxyz": quaternion,
            }
        )
    return {
        "manifest_version": 1,
        "source": "ManiSkill2 processed YCB Object and Model Set",
        "source_url": YCB_SOURCE_URL,
        "license_notice": YCB_LICENSE_NOTICE,
        "asset_pools": {
            "tabletop": "curated_compact_household_objects_v1",
            "rear_shelf": "curated_tall_packaging_v1",
        },
        "tabletop": table_specs,
        "rear_shelf": shelf_specs,
    }


def _metadata_from_options(options: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    episode = (options or {}).get("episode_metadata")
    if not isinstance(episode, Mapping):
        return None
    task = episode.get("task", episode)
    if not isinstance(task, Mapping):
        return None
    value = task.get("visual_ood")
    return value if isinstance(value, Mapping) else None


def _sample_metadata(env, options: Mapping[str, Any] | None) -> dict[str, Any]:
    replay = _metadata_from_options(options)
    if replay is not None:
        metadata = dict(replay)
        return metadata

    seed = int(np.asarray(env._episode_seed).reshape(-1)[0])
    rng = np.random.RandomState(seed ^ 0x5A17C9E3)
    table_id = int(rng.randint(len(TABLE_PALETTES)))
    wall_id = int(rng.randint(len(WALL_PALETTES)))
    light_id = int(rng.randint(len(LIGHT_PROFILES)))
    distractor_id = int(rng.randint(4))
    accent_shift = float(rng.uniform(-0.025, 0.025))
    digital_assets = _sample_ycb_assets(rng, distractor_id)
    return {
        "schema_version": OOD_SCHEMA_VERSION,
        "table_style_id": table_id,
        "table_style": TABLE_PALETTES[table_id][2],
        "wall_style_id": wall_id,
        "wall_style": WALL_PALETTES[wall_id][2],
        "light_profile_id": light_id,
        "light_profile": LIGHT_PROFILES[light_id]["name"],
        "distractor_layout_id": distractor_id,
        "accent_shift": accent_shift,
        "static_distractors": [
            "textured_ycb_tabletop_objects",
            "textured_ycb_rear_shelf_objects",
            "wall_panels",
        ],
        "digital_assets": digital_assets,
    }


def _add_table_finish(env, metadata: Mapping[str, Any]) -> None:
    table_id = int(metadata["table_style_id"])
    base, accent, _ = TABLE_PALETTES[table_id]
    shift = float(metadata.get("accent_shift", 0.0))
    accent = tuple(float(np.clip(c + shift, 0.03, 0.95)) for c in accent[:3]) + (1.0,)
    # Put a collision-free finish just above the original z=0 wooden visual.
    # The original table collision remains the sole support surface, so task
    # heights and contact conditions are unchanged, while the wood is truly
    # occluded instead of covering this OOD finish.
    _visual_box(env.scene, "ood_table_base", (0.605, 0.605, 0.00010), (-0.135, 0.0, 0.00011), base)
    if table_id % 2 == 0:
        for index, y in enumerate(np.linspace(-0.525, 0.525, 8)):
            _visual_box(
                env.scene,
                f"ood_table_strip_{index}",
                (0.603, 0.004, 0.00010),
                (-0.135, float(y), 0.00023),
                accent,
            )
    else:
        for index, x in enumerate(np.linspace(-0.66, 0.39, 7)):
            _visual_box(
                env.scene,
                f"ood_table_joint_{index}",
                (0.003, 0.603, 0.00010),
                (float(x), 0.0, 0.00023),
                accent,
            )
        for index, y in enumerate(np.linspace(-0.525, 0.525, 7)):
            _visual_box(
                env.scene,
                f"ood_table_cross_joint_{index}",
                (0.603, 0.003, 0.00011),
                (-0.135, float(y), 0.00024),
                accent,
            )


def _add_room(env, metadata: Mapping[str, Any]) -> None:
    wall_id = int(metadata["wall_style_id"])
    wall, trim, _ = WALL_PALETTES[wall_id]
    _visual_box(env.scene, "ood_back_wall", (0.92, 0.018, 0.58), (-0.12, 0.70, 0.49), wall)
    _visual_box(env.scene, "ood_wall_baseboard", (0.92, 0.023, 0.030), (-0.12, 0.675, 0.035), trim)
    for index, x in enumerate((-0.51, -0.12, 0.27)):
        panel_color = trim if index != 1 else tuple(0.75 * np.asarray(wall[:3])) + (1.0,)
        _visual_box(
            env.scene,
            f"ood_wall_panel_{index}",
            (0.145, 0.008, 0.12),
            (x, 0.674, 0.60 + 0.035 * (index % 2)),
            panel_color,
        )
    _static_box(env.scene, "ood_rear_shelf", (0.55, 0.050, 0.014), (-0.12, 0.60, 0.31), trim)


def _add_distractors(env, metadata: Mapping[str, Any]) -> None:
    layout = int(metadata["distractor_layout_id"])
    muted = (
        (0.70, 0.24, 0.16, 1.0),
        (0.16, 0.47, 0.58, 1.0),
        (0.37, 0.52, 0.22, 1.0),
        (0.55, 0.34, 0.61, 1.0),
    )
    digital_assets = metadata.get("digital_assets")
    for specification in digital_assets.get("tabletop") or ():
        _build_ycb_static(
            env.scene,
            name=str(specification["actor_name"]),
            specification=specification,
        )
    for specification in digital_assets.get("rear_shelf") or ():
        _build_ycb_static(
            env.scene,
            name=str(specification["actor_name"]),
            specification=specification,
        )

    # The plant is procedural room dressing; principal clutter uses textured
    # YCB geometry.
    c = muted[(layout + 2) % 4]
    _static_cylinder(env.scene, "ood_shelf_pot", 0.034, 0.035, (0.40, 0.565, 0.359), c)
    _static_sphere(env.scene, "ood_shelf_plant_0", 0.036, (0.40, 0.565, 0.425), (0.18, 0.43, 0.20, 1.0))
    _static_sphere(env.scene, "ood_shelf_plant_1", 0.027, (0.365, 0.565, 0.447), (0.12, 0.34, 0.16, 1.0))
    _static_sphere(env.scene, "ood_shelf_plant_2", 0.025, (0.435, 0.565, 0.452), (0.24, 0.52, 0.22, 1.0))


def load_visual_ood_scene(env, options: Mapping[str, Any] | None) -> None:
    metadata = _sample_metadata(env, options)
    env._visual_ood_metadata = metadata
    _add_table_finish(env, metadata)
    _add_room(env, metadata)
    _add_distractors(env, metadata)


def load_visual_ood_lighting(env) -> None:
    metadata = getattr(env, "_visual_ood_metadata", None)
    profile = LIGHT_PROFILES[int(metadata["light_profile_id"])]
    env.scene.set_ambient_light(profile["ambient"])
    env.scene.add_directional_light(
        profile["key_direction"],
        profile["key_color"],
        shadow=bool(env.enable_shadow),
        shadow_scale=3.0,
        shadow_map_size=2048,
    )
    env.scene.add_directional_light(profile["fill_direction"], profile["fill_color"])
    env.scene.add_point_light(
        profile["point_position"], profile["point_color"], shadow=False
    )


__all__ = ["OOD_SCHEMA_VERSION", "load_visual_ood_scene", "load_visual_ood_lighting"]
