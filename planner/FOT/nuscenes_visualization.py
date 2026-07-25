from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nuscenes_fot import (
    NuScenesFOTAdapter,
    NuScenesFOTPlanOutput,
    TrackingBox2D,
)

BEV_BG_COLOR = (20, 24, 34)
BEV_MAP_POLYGON_COLOR = (48, 56, 72)
BEV_MAP_CENTERLINE_COLOR = (120, 138, 168)
BEV_TEXT_COLOR = (224, 232, 246)
BEV_TEXT_SHADOW_COLOR = (0, 0, 0)
BEV_OUTLINE_COLOR = (8, 10, 14)
BEV_EGO_FILL_COLOR = (72, 232, 132)
BEV_EGO_OUTLINE_COLOR = (14, 92, 48)


@dataclass(slots=True)
class VisualizationConfig:
    enabled: bool = True
    bev_output_path: str = "/home/jushuo/Code/zz7-planning/output/nuscenes_fot_bev.png"
    cam_front_output_path: str = "/home/jushuo/Code/zz7-planning/output/nuscenes_fot_cam_front.png"
    image_dataroot: str = ""
    scenario_root: str = ""
    image_source_subdir: str = ""
    case_name: str = ""
    bev_width_px: int = 1000
    bev_height_px: int = 1000
    bev_forward_m: float = 50.0
    bev_backward_m: float = 30.0
    bev_left_m: float = 30.0
    bev_right_m: float = 30.0
    trajectory_dot_radius_px: int = 4
    reference_dot_radius_px: int = 2
    obstacle_dot_radius_px: int = 3
    tracking_box_line_width_px: int = 2
    prediction_line_width_px: int = 2
    map_line_width_px: int = 2
    prediction_dot_radius_px: int = 2
    box_heading_arrow_length_m: float = 1.8
    box_heading_arrow_width_px: int = 2
    id_palette_size: int = 30
    prediction_match_max_distance_m: float = 8.0
    info_font_size: int = 24
    draw_reference_path: bool = True
    draw_obstacles: bool = False
    draw_tracking_boxes: bool = True
    draw_prediction_trajectories: bool = True
    draw_map: bool = True
    draw_ego: bool = True
    map_padding_m: float = 10.0


@dataclass(slots=True)
class VisualizationOutput:
    bev_output_path: Path | None
    cam_front_output_path: Path | None


def _resolve_camera_image_path(
    *,
    nuscenes_root: Path,
    camera_frame: dict,
    config: VisualizationConfig,
) -> Path | None:
    filename = str(camera_frame.get("filename", "") or "").strip()
    basename = Path(filename).name
    channel = "CAM_FRONT"

    image_dataroot = Path(config.image_dataroot) if str(config.image_dataroot).strip() else None
    scenario_root = Path(config.scenario_root) if str(config.scenario_root).strip() else None
    image_source_subdir = str(config.image_source_subdir or "").strip()
    case_name = str(config.case_name or "").strip()

    candidates: list[Path | None] = []
    if filename:
        candidates.append(nuscenes_root / filename)
        candidates.append(image_dataroot / filename if image_dataroot is not None else None)
    if image_source_subdir:
        candidates.extend(
            [
                nuscenes_root / image_source_subdir / channel / basename,
                image_dataroot / image_source_subdir / channel / basename if image_dataroot is not None else None,
            ]
        )
    candidates.extend(
        [
            nuscenes_root / "samples" / channel / basename,
            image_dataroot / "samples" / channel / basename if image_dataroot is not None else None,
            nuscenes_root / basename,
            image_dataroot / basename if image_dataroot is not None else None,
        ]
    )
    if scenario_root is not None:
        if case_name and image_source_subdir:
            candidates.append(scenario_root / case_name / image_source_subdir / channel / basename)
        if image_source_subdir:
            candidates.append(scenario_root / image_source_subdir / channel / basename)
        candidates.append(scenario_root / basename)

    seen = set()
    for candidate in candidates:
        if candidate is None:
            continue
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    if scenario_root is not None and scenario_root.exists() and basename:
        for candidate in scenario_root.rglob(basename):
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
    return None


def save_plan_visualizations(
    adapter: NuScenesFOTAdapter,
    plan_output: NuScenesFOTPlanOutput,
    config: VisualizationConfig,
) -> VisualizationOutput:
    if not config.enabled:
        return VisualizationOutput(bev_output_path=None, cam_front_output_path=None)

    bev_output_path = Path(config.bev_output_path)
    cam_front_output_path = Path(config.cam_front_output_path)
    bev_output_path.parent.mkdir(parents=True, exist_ok=True)
    cam_front_output_path.parent.mkdir(parents=True, exist_ok=True)

    render_bev_plan(adapter, plan_output, config).save(bev_output_path)
    render_cam_front_plan(adapter, plan_output, config).save(cam_front_output_path)

    return VisualizationOutput(
        bev_output_path=bev_output_path,
        cam_front_output_path=cam_front_output_path,
    )


def render_bev_plan(
    adapter: NuScenesFOTAdapter,
    plan_output: NuScenesFOTPlanOutput,
    config: VisualizationConfig,
) -> Image.Image:
    prepared = plan_output.prepared_input
    planning_result = plan_output.planning_result
    ego = prepared.ego_state

    image = Image.new("RGB", (config.bev_width_px, config.bev_height_px), BEV_BG_COLOR)
    draw = ImageDraw.Draw(image)
    id_palette = _build_id_color_palette(config.id_palette_size)
    info_font = _load_font(config.info_font_size)
    id_font = _load_font(max(12, int(config.info_font_size * 0.55)))

    if config.draw_map:
        map_api = adapter.get_map_api(prepared.location)
        radius_m = max(
            config.bev_forward_m,
            config.bev_backward_m,
            config.bev_left_m,
            config.bev_right_m,
        ) + config.map_padding_m
        lane_polygons = map_api.get_lane_polygons_in_radius(
            ego.x,
            ego.y,
            radius_m=radius_m,
        )
        for polygon in lane_polygons:
            _draw_bev_filled_polygon(
                draw,
                config,
                ego.x,
                ego.y,
                ego.yaw,
                [(float(point[0]), float(point[1])) for point in polygon],
                fill=BEV_MAP_POLYGON_COLOR,
            )
        centerlines = map_api.get_centerlines_in_radius(
            ego.x,
            ego.y,
            radius_m=radius_m,
            resolution_meters=1.0,
        )
        for points in centerlines:
            _draw_bev_polyline(
                draw,
                config,
                ego.x,
                ego.y,
                ego.yaw,
                ((float(point[0]), float(point[1])) for point in points),
                color=BEV_MAP_CENTERLINE_COLOR,
                width=config.map_line_width_px,
            )

    if config.draw_reference_path:
        _draw_bev_points(
            draw,
            config,
            ego.x,
            ego.y,
            ego.yaw,
            zip(prepared.reference_waypoint_x, prepared.reference_waypoint_y),
            radius=config.reference_dot_radius_px,
            color=(120, 130, 145),
        )

    if config.draw_tracking_boxes:
        for box in prepared.tracking_boxes:
            color = _color_for_tracking_id(box.tracking_id, id_palette)
            _draw_bev_box_with_heading(
                draw,
                config,
                ego.x,
                ego.y,
                ego.yaw,
                box,
                outline=color,
                width=config.tracking_box_line_width_px,
            )
            center_local = _world_to_ego_local(ego.x, ego.y, ego.yaw, box.center_x, box.center_y)
            center_px = _bev_local_to_pixel(center_local[0], center_local[1], config)
            if center_px is not None:
                draw.text(
                    (center_px[0] + 4, center_px[1] - 10),
                    f"id:{box.tracking_id}",
                    fill=color,
                    font=id_font,
                )

    if config.draw_prediction_trajectories:
        prediction_matches = _match_prediction_trajectories_to_boxes(
            prepared.tracking_boxes,
            prepared.prediction_trajectories,
            max_id_match_distance=config.prediction_match_max_distance_m,
        )
        for match in prediction_matches:
            color = _color_for_tracking_id(match["tracking_id"], id_palette)
            _draw_bev_points(
                draw,
                config,
                ego.x,
                ego.y,
                ego.yaw,
                match["trajectory_points_xy"],
                radius=config.prediction_dot_radius_px,
                color=color,
            )

    if config.draw_obstacles:
        _draw_bev_points(
            draw,
            config,
            ego.x,
            ego.y,
            ego.yaw,
            ((point.x, point.y) for point in prepared.tracking_obstacle_points),
            radius=config.obstacle_dot_radius_px,
            color=(255, 130, 60),
        )
        _draw_bev_points(
            draw,
            config,
            ego.x,
            ego.y,
            ego.yaw,
            ((point.x, point.y) for point in prepared.prediction_obstacle_points),
            radius=config.obstacle_dot_radius_px,
            color=(255, 210, 90),
        )

    if planning_result.best_trajectory is not None:
        _draw_bev_points(
            draw,
            config,
            ego.x,
            ego.y,
            ego.yaw,
            zip(planning_result.best_trajectory.x, planning_result.best_trajectory.y),
            radius=config.trajectory_dot_radius_px,
            color=(0, 255, 80),
        )

    if config.draw_ego:
        _draw_ego_box(draw, config)

    long_cmd, lat_cmd = _infer_motion_command(plan_output)
    _draw_text_block(
        draw,
        14,
        12,
        [
            f"scene: {prepared.scene_name}",
            f"sample: {prepared.sample_token[:8]}...",
            "plan: green dots",
            "tracking: id-colored boxes",
            "prediction: id-colored dots",
            "map: lane polygons + centerlines",
            f"cmd: {long_cmd}, {lat_cmd}",
        ],
        fill=BEV_TEXT_COLOR,
        shadow_fill=BEV_TEXT_SHADOW_COLOR,
        font=info_font,
    )
    return image


def render_cam_front_plan(
    adapter: NuScenesFOTAdapter,
    plan_output: NuScenesFOTPlanOutput,
    config: VisualizationConfig,
) -> Image.Image:
    prepared = plan_output.prepared_input
    planning_result = plan_output.planning_result
    camera_frame = adapter.get_cam_front_frame(prepared.requested_frame_token)
    id_palette = _build_id_color_palette(config.id_palette_size)
    image_path = _resolve_camera_image_path(
        nuscenes_root=adapter.nuscenes_root,
        camera_frame=camera_frame,
        config=config,
    )
    if image_path is not None:
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.new(
            "RGB",
            (int(camera_frame["width"]), int(camera_frame["height"])),
            (15, 15, 15),
        )
    draw = ImageDraw.Draw(image)
    info_font = _load_font(config.info_font_size)
    id_font = _load_font(max(12, int(config.info_font_size * 0.55)))

    if config.draw_reference_path:
        reference_uv = _project_global_points_to_cam(
            np.column_stack(
                [
                    np.asarray(prepared.reference_waypoint_x, dtype=float),
                    np.asarray(prepared.reference_waypoint_y, dtype=float),
                    np.zeros(len(prepared.reference_waypoint_x), dtype=float),
                ]
            ),
            camera_frame,
            image.size,
        )
        _draw_projected_points(
            draw,
            reference_uv,
            radius=max(1, config.reference_dot_radius_px),
            color=(160, 170, 180),
        )

    if config.draw_tracking_boxes:
        for box in prepared.tracking_boxes:
            color = _color_for_tracking_id(box.tracking_id, id_palette)
            corners_xyz = _box_corners_xyz_3d(box)
            corners_uv = _project_global_points_to_cam_keep_order(
                np.asarray(corners_xyz, dtype=float),
                camera_frame,
                image.size,
            )
            _draw_projected_box_3d(
                draw,
                corners_uv,
                color=color,
                width=config.tracking_box_line_width_px,
            )
            valid_uv = [point for point in corners_uv if point is not None]
            if valid_uv:
                min_u = min(point[0] for point in valid_uv)
                min_v = min(point[1] for point in valid_uv)
                draw.text(
                    (min_u, max(0.0, min_v - 12.0)),
                    f"id:{box.tracking_id}",
                    fill=color,
                    font=id_font,
                )

    if config.draw_prediction_trajectories:
        prediction_matches = _match_prediction_trajectories_to_boxes(
            prepared.tracking_boxes,
            prepared.prediction_trajectories,
            max_id_match_distance=config.prediction_match_max_distance_m,
        )
        for match in prediction_matches:
            color = _color_for_tracking_id(match["tracking_id"], id_palette)
            trajectory_uv = _project_global_points_to_cam(
                np.asarray([[x, y, 0.0] for x, y in match["trajectory_points_xy"]], dtype=float),
                camera_frame,
                image.size,
            )
            _draw_projected_points(
                draw,
                trajectory_uv,
                radius=max(1, config.prediction_dot_radius_px),
                color=color,
            )

    if planning_result.best_trajectory is not None:
        trajectory_uv = _project_global_points_to_cam(
            np.column_stack(
                [
                    np.asarray(planning_result.best_trajectory.x, dtype=float),
                    np.asarray(planning_result.best_trajectory.y, dtype=float),
                    np.zeros(len(planning_result.best_trajectory.x), dtype=float),
                ]
            ),
            camera_frame,
            image.size,
        )
        _draw_projected_points(
            draw,
            trajectory_uv,
            radius=config.trajectory_dot_radius_px,
            color=(0, 255, 80),
        )

    if config.draw_obstacles:
        tracking_uv = _project_global_points_to_cam(
            np.asarray(
                [[point.x, point.y, 0.0] for point in prepared.tracking_obstacle_points],
                dtype=float,
            ),
            camera_frame,
            image.size,
        )
        prediction_uv = _project_global_points_to_cam(
            np.asarray(
                [[point.x, point.y, 0.0] for point in prepared.prediction_obstacle_points],
                dtype=float,
            ),
            camera_frame,
            image.size,
        )
        _draw_projected_points(
            draw,
            tracking_uv,
            radius=max(1, config.obstacle_dot_radius_px - 1),
            color=(255, 130, 60),
        )
        _draw_projected_points(
            draw,
            prediction_uv,
            radius=max(1, config.obstacle_dot_radius_px - 1),
            color=(255, 210, 90),
        )

    long_cmd, lat_cmd = _infer_motion_command(plan_output)
    _draw_text_block(
        draw,
        14,
        12,
        [
            "CAM_FRONT",
            f"scene: {prepared.scene_name}",
            "plan: green dots",
            "tracking: id-colored 3D boxes",
            "prediction: id-colored dots",
            f"cmd: {long_cmd}, {lat_cmd}",
            f"image_missing: {image_path is None}",
        ],
        fill=(40, 255, 120),
        font=info_font,
    )
    return image


def _draw_bev_grid(draw: ImageDraw.ImageDraw, config: VisualizationConfig) -> None:
    width = config.bev_width_px
    height = config.bev_height_px
    major_color = (38, 45, 54)
    minor_color = (28, 34, 42)

    for left_m in range(-int(config.bev_right_m), int(config.bev_left_m) + 1, 5):
        x = _bev_x_to_px(float(left_m), config)
        draw.line([(x, 0), (x, height)], fill=major_color if left_m % 10 == 0 else minor_color)

    for forward_m in range(-int(config.bev_backward_m), int(config.bev_forward_m) + 1, 5):
        y = _bev_y_to_px(float(forward_m), config)
        draw.line([(0, y), (width, y)], fill=major_color if forward_m % 10 == 0 else minor_color)


def _draw_bev_points(
    draw: ImageDraw.ImageDraw,
    config: VisualizationConfig,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    points,
    *,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    for point_x, point_y in points:
        forward_m, left_m = _world_to_ego_local(ego_x, ego_y, ego_yaw, point_x, point_y)
        pixel = _bev_local_to_pixel(forward_m, left_m, config)
        if pixel is None:
            continue
        px, py = pixel
        draw.ellipse(
            [(px - radius, py - radius), (px + radius, py + radius)],
            outline=(255, 255, 255),
            fill=color,
        )


def _draw_bev_polyline(
    draw: ImageDraw.ImageDraw,
    config: VisualizationConfig,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    points_xy,
    *,
    color: tuple[int, int, int],
    width: int,
) -> None:
    pixels: list[tuple[int, int]] = []
    for point_x, point_y in points_xy:
        forward_m, left_m = _world_to_ego_local(ego_x, ego_y, ego_yaw, point_x, point_y)
        pixel = _bev_local_to_pixel(forward_m, left_m, config)
        if pixel is not None:
            pixels.append(pixel)
    if len(pixels) >= 2:
        draw.line(pixels, fill=color, width=width)


def _draw_bev_polygon(
    draw: ImageDraw.ImageDraw,
    config: VisualizationConfig,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    points_xy: list[tuple[float, float]],
    *,
    outline: tuple[int, int, int],
    width: int,
) -> None:
    pixels: list[tuple[int, int]] = []
    for point_x, point_y in points_xy:
        forward_m, left_m = _world_to_ego_local(ego_x, ego_y, ego_yaw, point_x, point_y)
        pixel = _bev_local_to_pixel(forward_m, left_m, config)
        if pixel is not None:
            pixels.append(pixel)
    if len(pixels) >= 3:
        draw.line(pixels + [pixels[0]], fill=outline, width=width)


def _draw_bev_filled_polygon(
    draw: ImageDraw.ImageDraw,
    config: VisualizationConfig,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    points_xy,
    *,
    fill: tuple[int, int, int],
) -> None:
    pixels: list[tuple[int, int]] = []
    for point_x, point_y in points_xy:
        forward_m, left_m = _world_to_ego_local(ego_x, ego_y, ego_yaw, point_x, point_y)
        pixel = _bev_local_to_pixel(forward_m, left_m, config)
        if pixel is not None:
            pixels.append(pixel)
    if len(pixels) >= 3:
        draw.polygon(pixels, fill=fill)


def _draw_bev_box_with_heading(
    draw: ImageDraw.ImageDraw,
    config: VisualizationConfig,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    box: TrackingBox2D,
    *,
    outline: tuple[int, int, int],
    width: int,
) -> None:
    corners = _box_corners_xy(box)
    pixels: list[tuple[int, int]] = []
    for point_x, point_y in corners:
        forward_m, left_m = _world_to_ego_local(ego_x, ego_y, ego_yaw, point_x, point_y)
        pixel = _bev_local_to_pixel(forward_m, left_m, config)
        if pixel is None:
            return
        pixels.append(pixel)
    draw.line(pixels + [pixels[0]], fill=BEV_OUTLINE_COLOR, width=width + 2)
    draw.line(pixels + [pixels[0]], fill=outline, width=width)

    front_center_x = 0.5 * (corners[0][0] + corners[1][0])
    front_center_y = 0.5 * (corners[0][1] + corners[1][1])
    tip_x = front_center_x + config.box_heading_arrow_length_m * math.cos(box.yaw)
    tip_y = front_center_y + config.box_heading_arrow_length_m * math.sin(box.yaw)
    p0_local = _world_to_ego_local(ego_x, ego_y, ego_yaw, front_center_x, front_center_y)
    p1_local = _world_to_ego_local(ego_x, ego_y, ego_yaw, tip_x, tip_y)
    p0 = _bev_local_to_pixel(p0_local[0], p0_local[1], config)
    p1 = _bev_local_to_pixel(p1_local[0], p1_local[1], config)
    if p0 is not None and p1 is not None:
        arrow_w = max(2, config.box_heading_arrow_width_px)
        draw.line([p0, p1], fill=BEV_OUTLINE_COLOR, width=arrow_w + 1)
        draw.line([p0, p1], fill=outline, width=arrow_w)


def _draw_ego_box(draw: ImageDraw.ImageDraw, config: VisualizationConfig) -> None:
    ego_length = 4.7
    ego_width = 2.0
    corners_local = [
        (0.5 * ego_length, 0.5 * ego_width),
        (0.5 * ego_length, -0.5 * ego_width),
        (-0.5 * ego_length, -0.5 * ego_width),
        (-0.5 * ego_length, 0.5 * ego_width),
    ]
    pixels: list[tuple[int, int]] = []
    for forward_m, left_m in corners_local:
        pixel = _bev_local_to_pixel(forward_m, left_m, config)
        if pixel is not None:
            pixels.append(pixel)
    if len(pixels) != 4:
        return
    draw.polygon(pixels, fill=BEV_EGO_FILL_COLOR, outline=BEV_EGO_OUTLINE_COLOR)
    center = _bev_local_to_pixel(0.0, 0.0, config)
    tip = _bev_local_to_pixel(0.5 * ego_length + 1.2, 0.0, config)
    if center is not None and tip is not None:
        draw.line([center, tip], fill=BEV_EGO_OUTLINE_COLOR, width=3)


def _draw_text_block(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    lines: list[str],
    *,
    fill: tuple[int, int, int],
    shadow_fill: tuple[int, int, int] | None = None,
    font: ImageFont.ImageFont,
) -> None:
    line_height = int(getattr(font, "size", 16) * 1.2)
    for line in lines:
        if shadow_fill is not None:
            draw.text((x + 1, y + 1), line, fill=shadow_fill, font=font)
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height


def _draw_projected_points(
    draw: ImageDraw.ImageDraw,
    points_uv: list[tuple[float, float]],
    *,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    for u, v in points_uv:
        draw.ellipse(
            [(u - radius, v - radius), (u + radius, v + radius)],
            fill=color,
        )


def _draw_projected_polyline(
    draw: ImageDraw.ImageDraw,
    points_uv: list[tuple[float, float]],
    *,
    color: tuple[int, int, int],
    width: int,
    close: bool,
) -> None:
    if len(points_uv) < 2:
        return
    points = points_uv + [points_uv[0]] if close else points_uv
    draw.line(points, fill=color, width=width)


def _draw_projected_box_3d(
    draw: ImageDraw.ImageDraw,
    corners_uv: list[tuple[float, float] | None],
    *,
    color: tuple[int, int, int],
    width: int,
) -> None:
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for i0, i1 in edges:
        p0 = corners_uv[i0]
        p1 = corners_uv[i1]
        if p0 is None or p1 is None:
            continue
        draw.line([p0, p1], fill=color, width=width)


def _box_corners_xy(box: TrackingBox2D) -> list[tuple[float, float]]:
    half_length = 0.5 * box.length
    half_width = 0.5 * box.width
    cos_yaw = math.cos(box.yaw)
    sin_yaw = math.sin(box.yaw)
    local_corners = [
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    ]
    corners: list[tuple[float, float]] = []
    for local_x, local_y in local_corners:
        global_x = box.center_x + cos_yaw * local_x - sin_yaw * local_y
        global_y = box.center_y + sin_yaw * local_x + cos_yaw * local_y
        corners.append((global_x, global_y))
    return corners


def _box_corners_xyz_3d(box: TrackingBox2D) -> list[tuple[float, float, float]]:
    half_length = 0.5 * box.length
    half_width = 0.5 * box.width
    half_height = 0.5 * box.height
    cos_yaw = math.cos(box.yaw)
    sin_yaw = math.sin(box.yaw)
    local_corners = [
        (half_length, half_width, -half_height),
        (half_length, -half_width, -half_height),
        (-half_length, -half_width, -half_height),
        (-half_length, half_width, -half_height),
        (half_length, half_width, half_height),
        (half_length, -half_width, half_height),
        (-half_length, -half_width, half_height),
        (-half_length, half_width, half_height),
    ]
    corners: list[tuple[float, float, float]] = []
    for local_x, local_y, local_z in local_corners:
        global_x = box.center_x + cos_yaw * local_x - sin_yaw * local_y
        global_y = box.center_y + sin_yaw * local_x + cos_yaw * local_y
        corners.append((global_x, global_y, box.center_z + local_z))
    return corners


def _world_to_ego_local(
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    point_x: float,
    point_y: float,
) -> tuple[float, float]:
    dx = point_x - ego_x
    dy = point_y - ego_y
    forward = math.cos(ego_yaw) * dx + math.sin(ego_yaw) * dy
    left = -math.sin(ego_yaw) * dx + math.cos(ego_yaw) * dy
    return forward, left


def _bev_local_to_pixel(
    forward_m: float,
    left_m: float,
    config: VisualizationConfig,
) -> tuple[int, int] | None:
    if forward_m < -config.bev_backward_m or forward_m > config.bev_forward_m:
        return None
    if left_m < -config.bev_right_m or left_m > config.bev_left_m:
        return None
    return _bev_x_to_px(left_m, config), _bev_y_to_px(forward_m, config)


def _bev_x_to_px(left_m: float, config: VisualizationConfig) -> int:
    total_width_m = config.bev_left_m + config.bev_right_m
    normalized = (config.bev_left_m - left_m) / total_width_m
    return int(round(normalized * config.bev_width_px))


def _bev_y_to_px(forward_m: float, config: VisualizationConfig) -> int:
    total_height_m = config.bev_forward_m + config.bev_backward_m
    normalized = (config.bev_forward_m - forward_m) / total_height_m
    return int(round(normalized * config.bev_height_px))


def _project_global_points_to_cam(
    points_global_xyz: np.ndarray,
    camera_frame: dict,
    image_size: tuple[int, int],
) -> list[tuple[float, float]]:
    if points_global_xyz.size == 0:
        return []

    rotation_ego_global = _quaternion_to_rotation_matrix(camera_frame["ego_pose_rotation"])
    translation_ego_global = np.asarray(camera_frame["ego_pose_translation"], dtype=float)
    rotation_sensor_ego = _quaternion_to_rotation_matrix(camera_frame["calibrated_sensor_rotation"])
    translation_sensor_ego = np.asarray(camera_frame["calibrated_sensor_translation"], dtype=float)
    intrinsic = np.asarray(camera_frame["camera_intrinsic"], dtype=float)

    points_ego = (rotation_ego_global.T @ (points_global_xyz - translation_ego_global).T).T
    points_sensor = (rotation_sensor_ego.T @ (points_ego - translation_sensor_ego).T).T
    valid_depth = points_sensor[:, 2] > 0.5
    if not np.any(valid_depth):
        return []

    points_sensor = points_sensor[valid_depth]
    uvw = (intrinsic @ points_sensor.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]

    image_width, image_height = image_size
    valid_pixels = (
        (uv[:, 0] >= 0.0)
        & (uv[:, 0] < image_width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < image_height)
    )
    return [(float(u), float(v)) for u, v in uv[valid_pixels]]


def _project_global_points_to_cam_keep_order(
    points_global_xyz: np.ndarray,
    camera_frame: dict,
    image_size: tuple[int, int],
) -> list[tuple[float, float] | None]:
    if points_global_xyz.size == 0:
        return []

    rotation_ego_global = _quaternion_to_rotation_matrix(camera_frame["ego_pose_rotation"])
    translation_ego_global = np.asarray(camera_frame["ego_pose_translation"], dtype=float)
    rotation_sensor_ego = _quaternion_to_rotation_matrix(camera_frame["calibrated_sensor_rotation"])
    translation_sensor_ego = np.asarray(camera_frame["calibrated_sensor_translation"], dtype=float)
    intrinsic = np.asarray(camera_frame["camera_intrinsic"], dtype=float)

    points_ego = (rotation_ego_global.T @ (points_global_xyz - translation_ego_global).T).T
    points_sensor = (rotation_sensor_ego.T @ (points_ego - translation_sensor_ego).T).T

    image_width, image_height = image_size
    projected: list[tuple[float, float] | None] = []
    for point in points_sensor:
        if point[2] <= 0.5:
            projected.append(None)
            continue
        uvw = intrinsic @ point
        u = float(uvw[0] / uvw[2])
        v = float(uvw[1] / uvw[2])
        margin = max(image_width, image_height) * 4.0
        if -margin <= u <= image_width + margin and -margin <= v <= image_height + margin:
            projected.append((u, v))
        else:
            projected.append(None)
    return projected


def _load_font(font_size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _infer_motion_command(plan_output: NuScenesFOTPlanOutput) -> tuple[str, str]:
    trajectory = plan_output.planning_result.best_trajectory
    if trajectory is None or len(trajectory.acceleration) == 0 or len(trajectory.d) == 0:
        return "no_plan", "no_plan"

    accel = float(trajectory.acceleration[min(1, len(trajectory.acceleration) - 1)])
    if accel > 0.3:
        long_cmd = "accelerate"
    elif accel < -0.3:
        long_cmd = "brake"
    else:
        long_cmd = "keep_speed"

    lateral_shift = float(trajectory.d[-1] - trajectory.d[0])
    if lateral_shift > 0.8:
        lat_cmd = "lane_change_left"
    elif lateral_shift < -0.8:
        lat_cmd = "lane_change_right"
    else:
        lat_cmd = "keep_lane"
    return long_cmd, lat_cmd


def _build_id_color_palette(size: int) -> list[tuple[int, int, int]]:
    base_palette: list[tuple[int, int, int]] = [
        (255, 87, 51),
        (255, 195, 0),
        (46, 204, 113),
        (52, 152, 219),
        (155, 89, 182),
        (241, 90, 170),
        (26, 188, 156),
        (230, 126, 34),
        (231, 76, 60),
        (127, 140, 141),
        (255, 111, 0),
        (0, 229, 255),
        (168, 255, 4),
        (255, 61, 127),
        (124, 77, 255),
        (0, 200, 83),
    ]
    palette: list[tuple[int, int, int]] = []
    for idx in range(max(1, size)):
        palette.append(base_palette[idx % len(base_palette)])
    return palette


def _tracking_id_to_int(tracking_id: str) -> int:
    try:
        return int(tracking_id)
    except ValueError:
        return sum((idx + 1) * ord(char) for idx, char in enumerate(tracking_id))


def _color_for_tracking_id(
    tracking_id: str | None,
    palette: list[tuple[int, int, int]],
) -> tuple[int, int, int]:
    if tracking_id is None:
        return (255, 210, 90)
    return palette[_tracking_id_to_int(str(tracking_id)) % len(palette)]


def _match_prediction_trajectories_to_boxes(
    tracking_boxes: list[TrackingBox2D],
    prediction_trajectories,
    *,
    max_id_match_distance: float,
) -> list[dict]:
    if not prediction_trajectories:
        return []
    box_by_id = {str(box.tracking_id): box for box in tracking_boxes}
    matched: list[dict] = []
    for trajectory in prediction_trajectories:
        matched_id: str | None = None
        x0 = None
        y0 = None
        if trajectory.points_xy:
            x0, y0 = trajectory.points_xy[0]
        if trajectory.source_track_id is not None:
            candidate = str(trajectory.source_track_id)
            if candidate in box_by_id and x0 is not None and y0 is not None:
                box = box_by_id[candidate]
                dist = math.hypot(box.center_x - x0, box.center_y - y0)
                if dist <= max_id_match_distance:
                    matched_id = candidate
        if matched_id is None and tracking_boxes and x0 is not None and y0 is not None:
            nearest_box = min(
                tracking_boxes,
                key=lambda box: (box.center_x - x0) ** 2 + (box.center_y - y0) ** 2,
            )
            matched_id = str(nearest_box.tracking_id)
        matched.append(
            {
                "tracking_id": matched_id,
                "trajectory_points_xy": trajectory.points_xy,
            }
        )
    return matched


def _quaternion_to_rotation_matrix(quaternion_wxyz: list[float]) -> np.ndarray:
    w, x, y, z = quaternion_wxyz
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
