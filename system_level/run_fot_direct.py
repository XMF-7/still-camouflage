#!/usr/bin/env python3
from __future__ import annotations
"""Run zz7-planning Frenet Optimal Trajectory on zz9 plan_input."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import yaml

from adapter import planner_contract, standard_metadata


EVAL_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = EVAL_ROOT / "config.yaml"
DEFAULT_DT_S = 0.5
DEFAULT_HORIZON_S = 8.0
DETECTION_MODEL_ID_MAP = {
    1: "bevdet",
    2: "bevdepth",
    3: "fastbev",
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return path.resolve()


def _yaw_from_quaternion_wxyz(q: Sequence[float]) -> float:
    w, x, y, z = [float(v) for v in q]
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _xy(value: Any) -> List[float]:
    if isinstance(value, dict):
        if "translation" in value:
            value = value["translation"]
        elif "center_world_m" in value:
            value = value["center_world_m"]
        elif "x_m" in value and "y_m" in value:
            return [float(value["x_m"]), float(value["y_m"])]
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return [float(value[0]), float(value[1])]
    return []


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _cumulative_s(path_xy: np.ndarray) -> np.ndarray:
    if len(path_xy) <= 1:
        return np.zeros((len(path_xy),), dtype=float)
    seg = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _planning_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    planning = cfg.get("planning", {}) if isinstance(cfg.get("planning", {}), dict) else {}
    out = {
        "repo_root": "/home/jushuo/Code/zz7-planning",
        "python": sys.executable,
        "horizon_s": DEFAULT_HORIZON_S,
        "planned_trajectory_sample_interval": DEFAULT_DT_S,
        "route_length_m": 80.0,
        "route_resolution_m": 1.0,
        "closest_lane_radius_m": 20.0,
        "include_tracking_boxes": True,
        "include_prediction_points": True,
        "prediction_step_stride": 1,
        "prediction_lateral_gate_m": 1.8,
        "prediction_s_min_m": 0.0,
        "tracking_box_scale": 1.0,
        "robot_radius": 1.8,
        # If FOT has no feasible candidate, optionally output a hard-brake fallback plan instead of empty trajectory.
        "force_emergency_brake_on_no_solution": False,
        "emergency_decel_mps2": 9.0,
        "emergency_keep_heading": True,
        "prediction_stop_mode": "dynamic",
        "prediction_stop_buffer_m": 4.0,
        "prediction_min_stop_s_m": 2.0,
        "target_speed": 10.0,
        "d_t_s": 1.25,
        "max_speed": 13.88888888888889,
        "max_accel": 5.0,
        "max_curvature": 1.0,
        "dt": 0.25,
        "max_t": 6.0,
        "min_t": 5.0,
        "n_s_sample": 8,
        "max_road_width": 4.0,
        "d_road_w": 0.5,
    }
    for key, value in planning.items():
        if key == "fot" and isinstance(value, dict):
            continue
        out[key] = value
    nested = planning.get("fot", {})
    if isinstance(nested, dict):
        out.update(nested)
    return out


def _tracking_adapter_name(cfg: Dict[str, Any]) -> str:
    tracking_cfg = cfg.get("tracking", {}) if isinstance(cfg.get("tracking", {}), dict) else {}
    model_id = tracking_cfg.get("model_id", None)
    if model_id is not None:
        return {1: "ab3dmot", 2: "centertrack"}.get(int(model_id), str(model_id))
    return str(tracking_cfg.get("adapter", tracking_cfg.get("model", "ab3dmot")) or "ab3dmot").strip().lower()


def _tracking_output_subdir(cfg: Dict[str, Any]) -> str:
    return f"{_legacy_tracking_output_subdir(cfg)}-from-{_detection_output_subdir(cfg)}"


def _legacy_tracking_output_subdir(cfg: Dict[str, Any]) -> str:
    return {"ab3dmot": "track-ab", "centertrack": "track-centertrack"}.get(
        _tracking_adapter_name(cfg),
        f"track-{_tracking_adapter_name(cfg).replace('_', '-')}",
    )


def _detection_model(cfg: Dict[str, Any]) -> str:
    detection_cfg = cfg.get("detection", {}) if isinstance(cfg.get("detection", {}), dict) else {}
    model_id = detection_cfg.get("model_id", None)
    if model_id is not None:
        return DETECTION_MODEL_ID_MAP.get(int(model_id), str(model_id))
    explicit = str(detection_cfg.get("model", detection_cfg.get("detector", "")) or "").strip().lower()
    if explicit:
        return explicit
    config_path = detection_cfg.get("config_path", "")
    if config_path and Path(config_path).exists():
        return str(_load_yaml(Path(config_path)).get("model", "bevdet")).strip().lower()
    return "bevdet"


def _detection_output_subdir(cfg: Dict[str, Any]) -> str:
    return f"det-{_detection_model(cfg).replace('_', '-')}"


def _tracking_json_path(cfg: Dict[str, Any], sample: str, phase: str) -> Path:
    raw = str(cfg.get("input", {}).get("tracking_json", "") or "").strip()
    if raw:
        path = Path(raw).expanduser().resolve()
        if path.is_file():
            return path
        return path / f"{sample}.json"
    return Path(cfg["output"]["dir"]) / sample / _tracking_output_subdir(cfg) / phase / "tracking.json"


def _prediction_xy(plan_input: Dict[str, Any], dt_s: float, horizon_s: float) -> Tuple[str, List[List[float]]]:
    predictions = plan_input.get("target_agent", {}).get("predictions", {})
    if not isinstance(predictions, dict):
        return "", []
    desired_steps = max(1, int(round(float(horizon_s) / float(dt_s))))
    parsed = []
    for key, raw in predictions.items():
        if not isinstance(raw, list):
            continue
        try:
            seconds = float(str(key).rstrip("s"))
        except ValueError:
            seconds = len(raw) * dt_s
        xy = [_xy(item) for item in raw]
        xy = [item for item in xy if item]
        if xy:
            parsed.append((str(key), seconds, xy))
    if not parsed:
        return "", []
    parsed.sort(key=lambda item: (abs(item[1] - horizon_s), -len(item[2])))
    key, _seconds, xy = parsed[0]
    out = xy[:desired_steps]
    if len(out) < desired_steps:
        out.extend([out[-1]] * (desired_steps - len(out)))
    return key, out


def _prediction_obstacles(target_xy: List[List[float]], stride: int) -> List[Any]:
    from frenet_interface import ObstaclePoint

    return [ObstaclePoint(x=float(x), y=float(y)) for x, y in target_xy[:: max(1, int(stride))]]


def _project_points_to_path_sd(points: List[List[float]], path_xy: np.ndarray, path_s: np.ndarray) -> Dict[str, np.ndarray]:
    pts = np.asarray(points, dtype=float).reshape((-1, 2)) if points else np.zeros((0, 2), dtype=float)
    empty = {
        "s": np.zeros((0,), dtype=float),
        "d": np.zeros((0,), dtype=float),
        "distance": np.zeros((0,), dtype=float),
        "segment_index": np.zeros((0,), dtype=int),
    }
    if len(pts) == 0 or len(path_xy) == 0:
        return empty
    if len(path_xy) == 1:
        delta = pts - path_xy[0].reshape((1, 2))
        return {
            "s": np.zeros((len(pts),), dtype=float),
            "d": delta[:, 1].astype(float),
            "distance": np.linalg.norm(delta, axis=1).astype(float),
            "segment_index": np.zeros((len(pts),), dtype=int),
        }

    out_s = np.zeros((len(pts),), dtype=float)
    out_d = np.zeros((len(pts),), dtype=float)
    out_dist = np.zeros((len(pts),), dtype=float)
    out_seg = np.zeros((len(pts),), dtype=int)
    for i, p in enumerate(pts):
        best = (float("inf"), 0.0, 0.0, 0)
        for seg_idx in range(len(path_xy) - 1):
            a = path_xy[seg_idx]
            b = path_xy[seg_idx + 1]
            ab = b - a
            seg_len = float(np.linalg.norm(ab))
            if seg_len <= 1.0e-6:
                continue
            u = float(np.clip(np.dot(p - a, ab) / (seg_len * seg_len), 0.0, 1.0))
            proj = a + u * ab
            delta = p - proj
            dist = float(np.linalg.norm(delta))
            tangent = ab / seg_len
            left_normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
            signed_d = float(np.dot(delta, left_normal))
            s_val = float(path_s[seg_idx] + u * seg_len)
            if dist < best[0]:
                best = (dist, s_val, signed_d, seg_idx)
        out_dist[i], out_s[i], out_d[i], out_seg[i] = best
    return {"s": out_s, "d": out_d, "distance": out_dist, "segment_index": out_seg}


def _filter_prediction_points(
    target_xy: List[List[float]],
    path_xy: np.ndarray,
    path_s: np.ndarray,
    pcfg: Dict[str, Any],
) -> Tuple[List[List[float]], Dict[str, Any]]:
    proj = _project_points_to_path_sd(target_xy, path_xy, path_s)
    s = proj["s"]
    d = proj["d"]
    lateral_gate = float(pcfg.get("prediction_lateral_gate_m", 1.8))
    s_min = float(pcfg.get("prediction_s_min_m", 0.0))
    if len(s) == 0:
        return [], {
            "total_points": 0,
            "used_points": 0,
            "prediction_lateral_gate_m": lateral_gate,
            "prediction_s_min_m": s_min,
            "lateral_abs_min_m": "",
            "lateral_abs_mean_m": "",
            "lateral_abs_max_m": "",
            "used_s_min_m": "",
            "used_s_max_m": "",
        }
    mask = (np.abs(d) <= lateral_gate) & (s >= s_min)
    used = [target_xy[i] for i, keep in enumerate(mask.tolist()) if keep]
    abs_d = np.abs(d)
    used_s = s[mask]
    return used, {
        "total_points": int(len(s)),
        "used_points": int(len(used)),
        "prediction_lateral_gate_m": lateral_gate,
        "prediction_s_min_m": s_min,
        "lateral_abs_min_m": float(np.min(abs_d)),
        "lateral_abs_mean_m": float(np.mean(abs_d)),
        "lateral_abs_max_m": float(np.max(abs_d)),
        "used_s_min_m": float(np.min(used_s)) if len(used_s) else "",
        "used_s_max_m": float(np.max(used_s)) if len(used_s) else "",
    }


def _box_obstacle_points(track: Dict[str, Any], box_scale: float) -> List[Any]:
    from frenet_interface import ObstaclePoint

    center = _xy(track)
    if not center:
        return []
    size = track.get("size", [])
    width = float(size[0]) if isinstance(size, list) and len(size) >= 1 else 2.0
    length = float(size[1]) if isinstance(size, list) and len(size) >= 2 else 4.5
    yaw = _yaw_from_quaternion_wxyz(track.get("rotation", [1.0, 0.0, 0.0, 0.0]))
    half_l = 0.5 * length * float(box_scale)
    half_w = 0.5 * width * float(box_scale)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    offsets = [(0.0, 0.0), (half_l, half_w), (half_l, -half_w), (-half_l, half_w), (-half_l, -half_w)]
    points = []
    for dx, dy in offsets:
        x = center[0] + dx * cos_y - dy * sin_y
        y = center[1] + dx * sin_y + dy * cos_y
        points.append(ObstaclePoint(x=float(x), y=float(y)))
    return points


def _tracking_obstacles(cfg: Dict[str, Any], plan_input: Dict[str, Any], pcfg: Dict[str, Any]) -> Tuple[str, List[Any]]:
    path = _tracking_json_path(cfg, str(plan_input["sample"]), str(plan_input["phase"]))
    if not path.exists():
        return str(path), []
    payload = _load_json(path)
    results = payload.get("results", payload)
    tracks = results.get(str(plan_input["current_sample_token"]), []) if isinstance(results, dict) else []
    obstacles = []
    for track in tracks:
        if isinstance(track, dict):
            obstacles.extend(_box_obstacle_points(track, float(pcfg.get("tracking_box_scale", 1.0))))
    return str(path), obstacles


def _scene_location(nusc: Any, sample_token: str) -> str:
    sample = nusc.get("sample", sample_token)
    scene = nusc.get("scene", sample["scene_token"])
    log = nusc.get("log", scene["log_token"])
    return str(log["location"])


def _estimate_ego_speed(nusc: Any, sample_token: str) -> float:
    sample = nusc.get("sample", sample_token)
    prev_token = str(sample.get("prev", "") or "")
    next_token = str(sample.get("next", "") or "")
    if prev_token:
        a, b = prev_token, sample_token
    elif next_token:
        a, b = sample_token, next_token
    else:
        return 0.0
    pa = _ego_pose(nusc, a)
    pb = _ego_pose(nusc, b)
    dt_s = (pb["timestamp_us"] - pa["timestamp_us"]) / 1_000_000.0
    return _distance([pa["x"], pa["y"]], [pb["x"], pb["y"]]) / dt_s if dt_s > 0.0 else 0.0


def _ego_pose(nusc: Any, sample_token: str) -> Dict[str, Any]:
    sample = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    return {
        "timestamp_us": int(sample["timestamp"]),
        "x": float(ego_pose["translation"][0]),
        "y": float(ego_pose["translation"][1]),
        "yaw": _yaw_from_quaternion_wxyz(ego_pose["rotation"]),
    }


def _build_centerline(cfg: Dict[str, Any], plan_input: Dict[str, Any], nusc: Any, pcfg: Dict[str, Any]) -> Tuple[str, List[str], List[float], List[float]]:
    repo_root = Path(pcfg.get("repo_root", "/home/jushuo/Code/zz7-planning")).resolve()
    sys.path.insert(0, str(repo_root))
    from nuscenes_map import SimpleNuScenesMap

    current = plan_input["ego"]["current"]
    x = float(current["translation"][0])
    y = float(current["translation"][1])
    yaw = float(current.get("yaw_rad", 0.0))
    try:
        map_api = SimpleNuScenesMap(Path(plan_input["dataroot"]).resolve(), _scene_location(nusc, plan_input["current_sample_token"]))
        lane_tokens, centerline = map_api.build_forward_centerline(
            x,
            y,
            yaw,
            route_length_m=float(pcfg.get("route_length_m", 80.0)),
            resolution_meters=float(pcfg.get("route_resolution_m", 1.0)),
            closest_lane_radius=float(pcfg.get("closest_lane_radius_m", 20.0)),
        )
        return "nuscenes_map_forward_centerline", lane_tokens, [float(v) for v in centerline[:, 0]], [float(v) for v in centerline[:, 1]]
    except Exception:
        progress = np.arange(0.0, float(pcfg.get("route_length_m", 80.0)) + 1.0, float(pcfg.get("route_resolution_m", 1.0)))
        xs = x + progress * math.cos(yaw)
        ys = y + progress * math.sin(yaw)
        return "ego_heading_fallback", [], [float(v) for v in xs], [float(v) for v in ys]


def _fot_planner_config(pcfg: Dict[str, Any]) -> Any:
    from frenet_interface import FrenetPlannerConfig, LateralMode, LongitudinalMode

    allowed = set(FrenetPlannerConfig.__dataclass_fields__.keys())
    kwargs = {key: value for key, value in pcfg.items() if key in allowed}
    if "lateral_mode" in kwargs and isinstance(kwargs["lateral_mode"], str):
        kwargs["lateral_mode"] = LateralMode(kwargs["lateral_mode"])
    if "longitudinal_mode" in kwargs and isinstance(kwargs["longitudinal_mode"], str):
        kwargs["longitudinal_mode"] = LongitudinalMode(kwargs["longitudinal_mode"])
    return FrenetPlannerConfig(**kwargs)


def _trajectory_rows(best: Any, start_time_us: int, horizon_s: float) -> List[Dict[str, Any]]:
    if best is None:
        return []
    rows = []
    max_len = min(len(best.t), len(best.x), len(best.y), len(best.yaw))
    for idx in range(max_len):
        t_s = best.t[idx]
        t = float(t_s)
        if t <= 1.0e-6:
            continue
        if t > float(horizon_s) + 1.0e-6:
            break
        rows.append(
            {
                "step": len(rows) + 1,
                "t_s": t,
                "timestamp_us": int(start_time_us + round(t * 1_000_000)),
                "translation": [float(best.x[idx]), float(best.y[idx]), 0.0],
                "yaw_rad": float(best.yaw[idx]),
                "speed_mps": float(best.speed[idx]) if idx < len(best.speed) else "",
                "acceleration_mps2": float(best.acceleration[idx]) if idx < len(best.acceleration) else "",
                "curvature_1pm": float(best.curvature[idx]) if idx < len(best.curvature) else "",
                "frenet_s_m": float(best.s[idx]) if idx < len(best.s) else "",
                "frenet_d_m": float(best.d[idx]) if idx < len(best.d) else "",
            }
        )
    return rows


def _emergency_brake_rows(
    *,
    start_xy: Sequence[float],
    start_yaw: float,
    start_speed: float,
    start_time_us: int,
    horizon_s: float,
    dt_s: float,
    decel_mps2: float,
    keep_heading: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n = max(1, int(round(float(horizon_s) / float(dt_s))))
    x = float(start_xy[0])
    y = float(start_xy[1])
    yaw = float(start_yaw)
    v = max(0.0, float(start_speed))
    a = -abs(float(decel_mps2))
    for i in range(1, n + 1):
        dt = float(dt_s)
        v_next = max(0.0, v + a * dt)
        ds = max(0.0, v * dt + 0.5 * a * dt * dt)
        if keep_heading:
            x += ds * math.cos(yaw)
            y += ds * math.sin(yaw)
        t = i * dt
        rows.append(
            {
                "step": i,
                "t_s": float(t),
                "timestamp_us": int(start_time_us + round(t * 1_000_000)),
                "translation": [float(x), float(y), 0.0],
                "yaw_rad": float(yaw),
                "speed_mps": float(v_next),
                "acceleration_mps2": float(a),
                "curvature_1pm": 0.0,
                "frenet_s_m": "",
                "frenet_d_m": "",
            }
        )
        v = v_next
    return rows


def _metrics(rows: List[Dict[str, Any]], target_xy: List[List[float]], ego_future_gt: List[Dict[str, Any]], best: Any, result: Any, dt_s: float) -> Dict[str, Any]:
    xy = [[float(row["translation"][0]), float(row["translation"][1])] for row in rows]
    distances = [_distance(xy[idx], target_xy[idx]) for idx in range(min(len(xy), len(target_xy)))]
    gt_xy = [_xy(row) for row in ego_future_gt]
    gt_xy = [row for row in gt_xy if row]
    ego_errors = [_distance(xy[idx], gt_xy[idx]) for idx in range(min(len(xy), len(gt_xy)))]
    first_collision_t = ""
    for idx, dist in enumerate(distances):
        if dist <= 3.5:
            first_collision_t = float(rows[idx]["t_s"]) if idx < len(rows) else float((idx + 1) * dt_s)
            break
    speeds = [float(row["speed_mps"]) for row in rows if row.get("speed_mps") != ""]
    accels = [float(row["acceleration_mps2"]) for row in rows if row.get("acceleration_mps2") != ""]
    jerks = [(accels[i] - accels[i - 1]) / dt_s for i in range(1, len(accels))]
    lateral_accels = []
    for row in rows:
        if row.get("speed_mps") != "" and row.get("curvature_1pm") != "":
            lateral_accels.append(abs(float(row["speed_mps"]) ** 2 * float(row["curvature_1pm"])))
    d_values = [abs(float(row["frenet_d_m"])) for row in rows if row.get("frenet_d_m") != ""]
    return {
        "plan_success": bool(rows),
        "trajectory_points": len(rows),
        "ego_ade_to_future_gt_m": float(sum(ego_errors) / len(ego_errors)) if ego_errors else "",
        "ego_fde_to_future_gt_m": float(ego_errors[-1]) if ego_errors else "",
        "min_target_distance_m": min(distances) if distances else "",
        "geometric_collision": first_collision_t != "",
        "first_collision_t_s": first_collision_t,
        "min_longitudinal_accel_mps2": min(accels) if accels else "",
        "max_longitudinal_accel_mps2": max(accels) if accels else "",
        "max_lateral_accel_mps2": max(lateral_accels) if lateral_accels else "",
        "max_jerk_mps3": max(abs(v) for v in jerks) if jerks else "",
        "max_lateral_offset_m": max(d_values) if d_values else "",
        "final_lateral_offset_m": d_values[-1] if d_values else "",
        "trajectory_length_m": sum(_distance(xy[i], xy[i - 1]) for i in range(1, len(xy))) if len(xy) > 1 else 0.0,
        "candidate_cost": float(best.cost) if best is not None else "",
        "candidate_lateral_target_m": float(best.d[-1]) if best is not None and best.d else "",
        "initial_speed_mps": speeds[0] if speeds else "",
        "candidate_counts": dict(getattr(result, "candidate_counts", {}) or {}),
    }


def run_one(config_path: Path, plan_input_path: Path, output_path: Path) -> Path:
    cfg = _load_yaml(config_path)
    plan_input = _load_json(plan_input_path)
    pcfg = _planning_cfg(cfg)
    repo_root = Path(pcfg.get("repo_root", "/home/jushuo/Code/zz7-planning")).resolve()
    sys.path.insert(0, str(repo_root))

    from frenet_interface import CartesianEgoState, LongitudinalMode, plan_once
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(
        version=str(cfg["nuscenes"].get("version", "v1.0-trainval")),
        dataroot=str(Path(cfg["nuscenes"]["dataroot"]).resolve()),
        verbose=False,
    )
    dt_s = float(pcfg.get("planned_trajectory_sample_interval", DEFAULT_DT_S))
    horizon_s = float(pcfg.get("horizon_s", DEFAULT_HORIZON_S))
    pred_key, target_xy = _prediction_xy(plan_input, dt_s, horizon_s)
    centerline_source, lane_tokens, waypoint_x, waypoint_y = _build_centerline(cfg, plan_input, nusc, pcfg)
    path_xy = np.asarray(list(zip(waypoint_x, waypoint_y)), dtype=float)
    path_s = _cumulative_s(path_xy)
    tracking_path, tracking_obstacles = _tracking_obstacles(cfg, plan_input, pcfg) if bool(pcfg.get("include_tracking_boxes", True)) else ("", [])
    filtered_target_xy, pred_filter_info = _filter_prediction_points(target_xy, path_xy, path_s, pcfg)
    prediction_obstacles = (
        _prediction_obstacles(filtered_target_xy, int(pcfg.get("prediction_step_stride", 1)))
        if bool(pcfg.get("include_prediction_points", True))
        else []
    )
    current = plan_input["ego"]["current"]
    ego_state = CartesianEgoState(
        x=float(current["translation"][0]),
        y=float(current["translation"][1]),
        yaw=float(current.get("yaw_rad", 0.0)),
        speed=_estimate_ego_speed(nusc, str(plan_input["current_sample_token"])),
        acceleration=0.0,
        curvature=0.0,
    )
    planner_config = _fot_planner_config(pcfg)
    effective_longitudinal_mode = str(planner_config.longitudinal_mode.value)
    dynamic_stop_s = ""
    used_s_min = pred_filter_info.get("used_s_min_m", "")
    if (
        bool(prediction_obstacles)
        and str(pcfg.get("prediction_stop_mode", "dynamic")).strip().lower() == "dynamic"
        and used_s_min != ""
    ):
        dynamic_stop_s = max(
            float(pcfg.get("prediction_min_stop_s_m", 2.0)),
            float(used_s_min) - float(pcfg.get("prediction_stop_buffer_m", 4.0)),
        )
        planner_config.longitudinal_mode = LongitudinalMode.MERGING_AND_STOPPING
        planner_config.stop_s = float(dynamic_stop_s)
        effective_longitudinal_mode = str(planner_config.longitudinal_mode.value)
    result = plan_once(
        waypoint_x,
        waypoint_y,
        obstacles=tracking_obstacles + prediction_obstacles,
        ego_cartesian_state=ego_state,
        config=planner_config,
    )
    rows = _trajectory_rows(result.best_trajectory, int(current.get("timestamp", 0)), horizon_s)
    used_emergency_fallback = False
    if (not result.has_solution) and bool(pcfg.get("force_emergency_brake_on_no_solution", False)):
        rows = _emergency_brake_rows(
            start_xy=[float(current["translation"][0]), float(current["translation"][1])],
            start_yaw=float(current.get("yaw_rad", 0.0)),
            start_speed=float(ego_state.speed),
            start_time_us=int(current.get("timestamp", 0)),
            horizon_s=float(horizon_s),
            dt_s=float(dt_s),
            decel_mps2=float(pcfg.get("emergency_decel_mps2", 9.0)),
            keep_heading=bool(pcfg.get("emergency_keep_heading", True)),
        )
        used_emergency_fallback = True
    payload = {
        "schema": "zz9.plan_result.v1",
        "metadata": standard_metadata(stage="planning", adapter_name="fot", model_name="PythonRobotics Frenet Optimal Trajectory"),
        "adapter_contract": planner_contract("fot"),
        "sample": plan_input["sample"],
        "phase": plan_input["phase"],
        "adapter": "fot",
        "model_name": "PythonRobotics Frenet Optimal Trajectory",
        "mode": "fot_direct",
        "status": (
            "fot_direct_completed"
            if result.has_solution
            else ("fot_direct_emergency_fallback" if used_emergency_fallback else "fot_direct_no_solution")
        ),
        "native_model_used": True,
        "source_plan_input_json": str(plan_input_path.resolve()),
        "source_tracking_json": tracking_path,
        "uses_tracking_obstacles_in_model": bool(tracking_obstacles),
        "uses_prediction_future_in_model": bool(prediction_obstacles),
        "reads_prediction_future": bool(target_xy),
        "prediction_horizon_key": pred_key,
        "coordinate_frame": "nuscenes_global",
        "feature_source": {
            "centerline_source": centerline_source,
            "reference_lane_tokens": lane_tokens,
            "tracking_obstacle_points": len(tracking_obstacles),
            "prediction_obstacle_points": len(prediction_obstacles),
            "prediction_obstacle_semantics": "static 2D obstacle-point approximation after ego-lane lateral gate; no time dimension",
            "prediction_filter": pred_filter_info,
            "configured_longitudinal_mode": str(pcfg.get("longitudinal_mode", "")),
            "effective_longitudinal_mode": effective_longitudinal_mode,
            "dynamic_prediction_stop_s_m": dynamic_stop_s,
            "used_emergency_brake_fallback": used_emergency_fallback,
        },
        "trajectory": rows,
        "reference_path": [
            {"index": idx, "translation": [float(x), float(y), 0.0]}
            for idx, (x, y) in enumerate(zip(waypoint_x, waypoint_y))
        ],
        "metrics": _metrics(rows, target_xy, plan_input.get("ego", {}).get("future_gt", []), result.best_trajectory, result, dt_s),
    }
    return _write_json(output_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--plan-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run_one(args.config.resolve(), args.plan_input.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
