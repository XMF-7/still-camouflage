#!/usr/bin/env python3
from __future__ import annotations
"""Run a lightweight prediction-conditioned MPC planner on zz9 plan_input.

This adapter is intentionally dependency-light (numpy only) so it can run in
restricted environments without OSQP/CVXOPT.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml

from adapter import planner_contract, standard_metadata


EVAL_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = EVAL_ROOT / "config.yaml"
DEFAULT_DT_S = 0.5
DEFAULT_HORIZON_S = 8.0


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


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


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


def _yaw_from_quaternion_wxyz(q: Sequence[float]) -> float:
    w, x, y, z = [float(v) for v in q]
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _planning_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    planning = cfg.get("planning", {}) if isinstance(cfg.get("planning", {}), dict) else {}
    out = {
        "repo_root": "/home/jushuo/Code/zz7-planning",
        "horizon_s": DEFAULT_HORIZON_S,
        "planned_trajectory_sample_interval": DEFAULT_DT_S,
        "route_length_m": 100.0,
        "route_resolution_m": 1.0,
        "closest_lane_radius_m": 20.0,
        "max_speed_mps": 16.0,
        "desired_speed_mps": 10.0,
        "free_drive_desired_speed_mps": 10.0,
        "max_accel_mps2": 2.5,
        "max_decel_mps2": 9.0,
        "safe_distance_m": 7.0,
        "prediction_lateral_gate_m": 1.8,
        "prediction_s_min_m": 0.0,
        "prediction_weight": 400.0,
        "speed_weight": 2.0,
        "accel_weight": 0.2,
        "jerk_weight": 0.4,
        "grad_iters": 100,
        "grad_lr": 0.08,
        "finite_diff_eps": 0.05,
    }
    for key, value in planning.items():
        if key == "mpc" and isinstance(value, dict):
            continue
        out[key] = value
    nested = planning.get("mpc", {})
    if isinstance(nested, dict):
        out.update(nested)
    return out


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


def _scene_location(nusc: Any, sample_token: str) -> str:
    sample = nusc.get("sample", sample_token)
    scene = nusc.get("scene", sample["scene_token"])
    log = nusc.get("log", scene["log_token"])
    return str(log["location"])


def _build_centerline(cfg: Dict[str, Any], plan_input: Dict[str, Any], nusc: Any, pcfg: Dict[str, Any]) -> Tuple[str, List[str], np.ndarray]:
    current = plan_input["ego"]["current"]
    x = float(current["translation"][0])
    y = float(current["translation"][1])
    yaw = float(current.get("yaw_rad", 0.0))
    try:
        repo_root = Path(pcfg.get("repo_root", "/home/jushuo/Code/zz7-planning")).resolve()
        sys.path.insert(0, str(repo_root))
        from nuscenes_map import SimpleNuScenesMap

        map_api = SimpleNuScenesMap(Path(plan_input["dataroot"]).resolve(), _scene_location(nusc, plan_input["current_sample_token"]))
        lane_tokens, centerline = map_api.build_forward_centerline(
            x,
            y,
            yaw,
            route_length_m=float(pcfg.get("route_length_m", 100.0)),
            resolution_meters=float(pcfg.get("route_resolution_m", 1.0)),
            closest_lane_radius=float(pcfg.get("closest_lane_radius_m", 20.0)),
        )
        return "nuscenes_map_forward_centerline", lane_tokens, np.asarray(centerline[:, :2], dtype=float)
    except Exception as exc:
        progress = np.arange(0.0, float(pcfg.get("route_length_m", 100.0)) + 1.0, float(pcfg.get("route_resolution_m", 1.0)))
        xs = x + progress * math.cos(yaw)
        ys = y + progress * math.sin(yaw)
        return f"ego_heading_fallback:{type(exc).__name__}:{exc}", [], np.stack([xs, ys], axis=-1)


def _cumulative_s(path_xy: np.ndarray) -> np.ndarray:
    if len(path_xy) <= 1:
        return np.zeros((len(path_xy),), dtype=float)
    seg = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _interp_xy(path_xy: np.ndarray, path_s: np.ndarray, s_query: np.ndarray) -> np.ndarray:
    s_clip = np.clip(s_query, 0.0, float(path_s[-1]) if len(path_s) else 0.0)
    x = np.interp(s_clip, path_s, path_xy[:, 0])
    y = np.interp(s_clip, path_s, path_xy[:, 1])
    return np.stack([x, y], axis=-1)


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


def _filter_prediction_projection(proj: Dict[str, np.ndarray], pcfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    s = proj["s"]
    d = proj["d"]
    if len(s) == 0:
        return np.zeros((0,), dtype=float), {
            "total_points": 0,
            "used_points": 0,
            "prediction_lateral_gate_m": float(pcfg.get("prediction_lateral_gate_m", 1.8)),
            "prediction_s_min_m": float(pcfg.get("prediction_s_min_m", 0.0)),
            "lateral_abs_min_m": "",
            "lateral_abs_mean_m": "",
            "lateral_abs_max_m": "",
            "used_s_min_m": "",
            "used_s_max_m": "",
        }
    lateral_gate = float(pcfg.get("prediction_lateral_gate_m", 1.8))
    s_min = float(pcfg.get("prediction_s_min_m", 0.0))
    mask = (np.abs(d) <= lateral_gate) & (s >= s_min)
    used_s = s[mask]
    abs_d = np.abs(d)
    return used_s, {
        "total_points": int(len(s)),
        "used_points": int(len(used_s)),
        "prediction_lateral_gate_m": lateral_gate,
        "prediction_s_min_m": s_min,
        "lateral_abs_min_m": float(np.min(abs_d)),
        "lateral_abs_mean_m": float(np.mean(abs_d)),
        "lateral_abs_max_m": float(np.max(abs_d)),
        "used_s_min_m": float(np.min(used_s)) if len(used_s) else "",
        "used_s_max_m": float(np.max(used_s)) if len(used_s) else "",
    }


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
    sa = nusc.get("sample", a)
    sb = nusc.get("sample", b)
    pda = nusc.get("sample_data", sa["data"]["LIDAR_TOP"])
    pdb = nusc.get("sample_data", sb["data"]["LIDAR_TOP"])
    ea = nusc.get("ego_pose", pda["ego_pose_token"])
    eb = nusc.get("ego_pose", pdb["ego_pose_token"])
    dt = (float(sb["timestamp"]) - float(sa["timestamp"])) / 1_000_000.0
    if dt <= 1e-6:
        return 0.0
    return _distance(ea["translation"], eb["translation"]) / dt


def _rollout_from_accel(a: np.ndarray, s0: float, v0: float, dt: float, v_max: float) -> Tuple[np.ndarray, np.ndarray]:
    n = len(a)
    s = np.zeros((n,), dtype=float)
    v = np.zeros((n,), dtype=float)
    s_prev, v_prev = float(s0), float(v0)
    for k in range(n):
        v_k = float(np.clip(v_prev + a[k] * dt, 0.0, v_max))
        s_k = float(s_prev + v_prev * dt + 0.5 * a[k] * dt * dt)
        s[k] = s_k
        v[k] = v_k
        s_prev, v_prev = s_k, v_k
    return s, v


def _mpc_cost(
    a: np.ndarray,
    s0: float,
    v0: float,
    dt: float,
    v_ref: float,
    v_max: float,
    s_obs: np.ndarray,
    safe_dist: float,
    w_speed: float,
    w_accel: float,
    w_jerk: float,
    w_pred: float,
) -> float:
    s, v = _rollout_from_accel(a, s0, v0, dt, v_max)
    cost = float(w_speed * np.sum((v - v_ref) ** 2) + w_accel * np.sum(a ** 2))
    if len(a) > 1:
        cost += float(w_jerk * np.sum((a[1:] - a[:-1]) ** 2))
    if len(s_obs) > 0:
        count = min(len(s), len(s_obs))
        # Require ego progress to stay behind predicted obstacle by safe_dist.
        violation = np.maximum(0.0, s[:count] - (s_obs[:count] - float(safe_dist)))
        cost += float(w_pred * np.sum(violation ** 2))
    return cost


def _solve_longitudinal_mpc(
    n: int,
    dt: float,
    s0: float,
    v0: float,
    v_ref: float,
    v_max: float,
    a_min: float,
    a_max: float,
    s_obs: np.ndarray,
    safe_dist: float,
    w_speed: float,
    w_accel: float,
    w_jerk: float,
    w_pred: float,
    iters: int,
    lr: float,
    fd_eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    a = np.zeros((n,), dtype=float)
    best_a = a.copy()
    best_cost = _mpc_cost(a, s0, v0, dt, v_ref, v_max, s_obs, safe_dist, w_speed, w_accel, w_jerk, w_pred)

    for _ in range(max(1, int(iters))):
        grad = np.zeros_like(a)
        for j in range(n):
            plus = a.copy()
            minus = a.copy()
            plus[j] += fd_eps
            minus[j] -= fd_eps
            c1 = _mpc_cost(plus, s0, v0, dt, v_ref, v_max, s_obs, safe_dist, w_speed, w_accel, w_jerk, w_pred)
            c2 = _mpc_cost(minus, s0, v0, dt, v_ref, v_max, s_obs, safe_dist, w_speed, w_accel, w_jerk, w_pred)
            grad[j] = (c1 - c2) / (2.0 * fd_eps)
        a = np.clip(a - float(lr) * grad, a_min, a_max)
        cost = _mpc_cost(a, s0, v0, dt, v_ref, v_max, s_obs, safe_dist, w_speed, w_accel, w_jerk, w_pred)
        if cost < best_cost:
            best_cost = cost
            best_a = a.copy()

    s, v = _rollout_from_accel(best_a, s0, v0, dt, v_max)
    info = {
        "iterations": int(iters),
        "best_cost": float(best_cost),
    }
    return s, v, best_a, info


def _trajectory_rows(path_xy: np.ndarray, path_s: np.ndarray, s: np.ndarray, v: np.ndarray, a: np.ndarray, start_time_us: int, dt_s: float) -> List[Dict[str, Any]]:
    xy = _interp_xy(path_xy, path_s, s)
    rows: List[Dict[str, Any]] = []
    for i in range(len(s)):
        if i + 1 < len(s):
            dx = float(xy[i + 1, 0] - xy[i, 0])
            dy = float(xy[i + 1, 1] - xy[i, 1])
        elif i > 0:
            dx = float(xy[i, 0] - xy[i - 1, 0])
            dy = float(xy[i, 1] - xy[i - 1, 1])
        else:
            dx, dy = 1.0, 0.0
        yaw = float(math.atan2(dy, dx))
        rows.append(
            {
                "step": int(i + 1),
                "t_s": float((i + 1) * dt_s),
                "timestamp_us": int(start_time_us + round((i + 1) * dt_s * 1_000_000)),
                "translation": [float(xy[i, 0]), float(xy[i, 1]), 0.0],
                "yaw_rad": yaw,
                "speed_mps": float(v[i]),
                "acceleration_mps2": float(a[i]),
                "frenet_s_m": float(s[i]),
            }
        )
    return rows


def _metrics(rows: List[Dict[str, Any]], target_xy: List[List[float]], ego_future_gt: List[Dict[str, Any]], dt_s: float) -> Dict[str, Any]:
    xy = [[float(row["translation"][0]), float(row["translation"][1])] for row in rows]
    distances = [_distance(xy[idx], target_xy[idx]) for idx in range(min(len(xy), len(target_xy)))]
    gt_xy = [_xy(row) for row in ego_future_gt]
    gt_xy = [row for row in gt_xy if row]
    ego_errors = [_distance(xy[idx], gt_xy[idx]) for idx in range(min(len(xy), len(gt_xy)))]

    first_collision_t = ""
    for idx, dist in enumerate(distances):
        if dist <= 3.5:
            first_collision_t = float((idx + 1) * dt_s)
            break

    accels = [float(row["acceleration_mps2"]) for row in rows if row.get("acceleration_mps2") != ""]
    jerks = [(accels[i] - accels[i - 1]) / dt_s for i in range(1, len(accels))]
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
        "max_jerk_mps3": max(abs(v) for v in jerks) if jerks else "",
        "trajectory_length_m": sum(_distance(xy[i], xy[i - 1]) for i in range(1, len(xy))) if len(xy) > 1 else 0.0,
    }


def run_one(config_path: Path, plan_input_path: Path, output_path: Path) -> Path:
    cfg = _load_yaml(config_path)
    plan_input = _load_json(plan_input_path)
    pcfg = _planning_cfg(cfg)

    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(
        version=str(cfg["nuscenes"].get("version", "v1.0-trainval")),
        dataroot=str(Path(cfg["nuscenes"]["dataroot"]).resolve()),
        verbose=False,
    )

    dt_s = float(pcfg.get("planned_trajectory_sample_interval", DEFAULT_DT_S))
    horizon_s = float(pcfg.get("horizon_s", DEFAULT_HORIZON_S))
    n_steps = max(1, int(round(horizon_s / dt_s)))

    pred_key, target_xy = _prediction_xy(plan_input, dt_s, horizon_s)
    centerline_source, lane_tokens, path_xy = _build_centerline(cfg, plan_input, nusc, pcfg)
    path_s = _cumulative_s(path_xy)

    current = plan_input["ego"]["current"]
    ego_xy = [float(current["translation"][0]), float(current["translation"][1])]
    s0 = float(path_s[int(np.argmin(np.linalg.norm(path_xy - np.asarray(ego_xy).reshape(1, 2), axis=1)))]) if len(path_xy) else 0.0
    v0 = float(_estimate_ego_speed(nusc, str(plan_input["current_sample_token"])))

    pred_projection = _project_points_to_path_sd(target_xy, path_xy, path_s)
    s_obs, pred_filter_info = _filter_prediction_projection(pred_projection, pcfg)
    desired_speed = float(pcfg.get("desired_speed_mps", 10.0))
    if len(s_obs) == 0:
        desired_speed = max(float(pcfg.get("free_drive_desired_speed_mps", desired_speed)), v0)
    s_plan, v_plan, a_plan, opt_info = _solve_longitudinal_mpc(
        n=n_steps,
        dt=dt_s,
        s0=s0,
        v0=v0,
        v_ref=desired_speed,
        v_max=float(pcfg.get("max_speed_mps", 16.0)),
        a_min=-abs(float(pcfg.get("max_decel_mps2", 9.0))),
        a_max=abs(float(pcfg.get("max_accel_mps2", 2.5))),
        s_obs=s_obs,
        safe_dist=float(pcfg.get("safe_distance_m", 7.0)),
        w_speed=float(pcfg.get("speed_weight", 2.0)),
        w_accel=float(pcfg.get("accel_weight", 0.2)),
        w_jerk=float(pcfg.get("jerk_weight", 0.4)),
        w_pred=float(pcfg.get("prediction_weight", 400.0)),
        iters=int(pcfg.get("grad_iters", 100)),
        lr=float(pcfg.get("grad_lr", 0.08)),
        fd_eps=float(pcfg.get("finite_diff_eps", 0.05)),
    )

    rows = _trajectory_rows(path_xy, path_s, s_plan, v_plan, a_plan, int(current.get("timestamp", 0)), dt_s)

    payload = {
        "schema": "zz9.plan_result.v1",
        "metadata": standard_metadata(stage="planning", adapter_name="mpc", model_name="Prediction-conditioned MPC (numpy)"),
        "adapter_contract": planner_contract("mpc"),
        "sample": plan_input["sample"],
        "phase": plan_input["phase"],
        "adapter": "mpc",
        "model_name": "Prediction-conditioned MPC (numpy)",
        "mode": "mpc_direct",
        "status": "mpc_direct_completed",
        "native_model_used": True,
        "source_plan_input_json": str(plan_input_path.resolve()),
        "uses_prediction_future_in_model": bool(len(s_obs) > 0),
        "reads_prediction_future": bool(target_xy),
        "prediction_horizon_key": pred_key,
        "coordinate_frame": "nuscenes_global",
        "feature_source": {
            "centerline_source": centerline_source,
            "reference_lane_tokens": lane_tokens,
            "mpc_solver": "projected_gradient_finite_difference",
            "obstacle_semantics": "prediction points projected to centerline progress constraints after ego-lane lateral gate",
            "prediction_filter": pred_filter_info,
            "effective_desired_speed_mps": desired_speed,
            "optimizer": opt_info,
        },
        "trajectory": rows,
        "reference_path": [
            {"index": idx, "translation": [float(row[0]), float(row[1]), 0.0]}
            for idx, row in enumerate(path_xy)
        ],
        "metrics": _metrics(rows, target_xy, plan_input.get("ego", {}).get("future_gt", []), dt_s),
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
