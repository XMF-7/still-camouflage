#!/usr/bin/env python3
"""Shared adapter contract for zz9 system-level pipeline.

All model-specific adapters should convert their native inputs/outputs through
these helpers before handing data to the next stage. The contract is deliberately
explicit about coordinate frames, units, time axis, and whether a field is used
by the model itself or only by offline metrics.
"""
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence


SCHEMA_VERSION = "zz9.adapter.v1"
GLOBAL_FRAME = "nuscenes_global"
EGO_FRAME = "current_ego"
DISTANCE_UNIT = "meter"
TIME_UNIT = "second"
ANGLE_UNIT = "radian"
DEFAULT_DT_S = 0.5


def _float_list(values: Iterable[Any], length: Optional[int] = None, name: str = "values") -> List[float]:
    out = [float(v) for v in values]
    if length is not None and len(out) != length:
        raise ValueError(f"{name} must have length {length}, got {len(out)}")
    return out


def _optional_float_list(values: Any, length: int, default: Sequence[float], name: str) -> List[float]:
    if values is None or values == "":
        return [float(v) for v in default]
    return _float_list(values, length=length, name=name)


def yaw_from_quaternion_wxyz(rotation_wxyz: Sequence[float]) -> float:
    w, x, y, z = _float_list(rotation_wxyz, length=4, name="rotation_wxyz")
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def make_frame_ref(sample: str, sample_token: str, timestamp_us: Optional[int] = None) -> Dict[str, Any]:
    return {
        "schema": f"{SCHEMA_VERSION}.frame_ref",
        "sample": str(sample),
        "sample_token": str(sample_token),
        "timestamp_us": None if timestamp_us is None else int(timestamp_us),
    }


def make_agent_state(
    *,
    sample_token: str,
    center_world_m: Sequence[float],
    size_wlh_m: Optional[Sequence[float]] = None,
    rotation_wxyz: Optional[Sequence[float]] = None,
    velocity_world_mps: Optional[Sequence[float]] = None,
    agent_id: str = "",
    name: str = "car",
    score: Optional[float] = None,
    source_stage: str = "",
) -> Dict[str, Any]:
    rotation = _optional_float_list(rotation_wxyz, 4, [1.0, 0.0, 0.0, 0.0], "rotation_wxyz")
    state = {
        "schema": f"{SCHEMA_VERSION}.agent_state",
        "sample_token": str(sample_token),
        "agent_id": str(agent_id),
        "name": str(name),
        "source_stage": str(source_stage),
        "coordinate_frame": GLOBAL_FRAME,
        "units": {
            "position": DISTANCE_UNIT,
            "size": DISTANCE_UNIT,
            "velocity": "meter_per_second",
            "yaw": ANGLE_UNIT,
        },
        "center_world_m": _float_list(center_world_m, length=3, name="center_world_m"),
        "size_wlh_m": _optional_float_list(size_wlh_m, 3, [0.0, 0.0, 0.0], "size_wlh_m"),
        "rotation_wxyz": rotation,
        "yaw_rad": yaw_from_quaternion_wxyz(rotation),
        "velocity_world_mps": _optional_float_list(velocity_world_mps, 2, [0.0, 0.0], "velocity_world_mps"),
    }
    if score is not None:
        state["score"] = float(score)
    return state


def agent_state_from_detection_box(box: Dict[str, Any], sample_token: str) -> Dict[str, Any]:
    return make_agent_state(
        sample_token=str(box.get("sample_token", sample_token)),
        center_world_m=box["translation"],
        size_wlh_m=box.get("size"),
        rotation_wxyz=box.get("rotation"),
        velocity_world_mps=box.get("velocity"),
        agent_id=str(box.get("detection_id", "")),
        name=str(box.get("detection_name", "")),
        score=None if box.get("detection_score") is None else float(box.get("detection_score")),
        source_stage="detection",
    )


def agent_state_from_tracking_box(box: Dict[str, Any], sample_token: str) -> Dict[str, Any]:
    return make_agent_state(
        sample_token=str(box.get("sample_token", sample_token)),
        center_world_m=box["translation"],
        size_wlh_m=box.get("size"),
        rotation_wxyz=box.get("rotation"),
        velocity_world_mps=box.get("velocity"),
        agent_id=str(box.get("tracking_id", "")),
        name=str(box.get("tracking_name", "")),
        score=None if box.get("tracking_score") is None else float(box.get("tracking_score")),
        source_stage="tracking",
    )


def agent_state_from_nuscenes_annotation(ann: Dict[str, Any]) -> Dict[str, Any]:
    return make_agent_state(
        sample_token=str(ann["sample_token"]),
        center_world_m=ann["translation"],
        size_wlh_m=ann.get("size"),
        rotation_wxyz=ann.get("rotation"),
        velocity_world_mps=ann.get("velocity", [0.0, 0.0]),
        agent_id=str(ann.get("instance_token", "")),
        name=str(ann.get("category_name", "")),
        source_stage="nuscenes_gt",
    )


def trajectory_from_xy_list(
    *,
    xy: Sequence[Sequence[Any]],
    start_sample_token: str,
    dt_s: float = DEFAULT_DT_S,
    coordinate_frame: str = GLOBAL_FRAME,
    source_stage: str = "",
) -> Dict[str, Any]:
    dt = float(dt_s)
    if dt <= 0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")
    points = []
    for index, point in enumerate(xy):
        xy_point = _float_list(point, length=2, name=f"trajectory[{index}]")
        points.append(
            {
                "step": int(index + 1),
                "t_s": float((index + 1) * dt),
                "x_m": xy_point[0],
                "y_m": xy_point[1],
            }
        )
    return {
        "schema": f"{SCHEMA_VERSION}.trajectory",
        "start_sample_token": str(start_sample_token),
        "coordinate_frame": str(coordinate_frame),
        "source_stage": str(source_stage),
        "dt_s": dt,
        "units": {"position": DISTANCE_UNIT, "time": TIME_UNIT},
        "points": points,
    }


def validate_prediction_payload(payload: Dict[str, Any], frequency_hz: float = 2.0) -> None:
    predictions = payload.get("predictions", {})
    if not isinstance(predictions, dict):
        raise ValueError("prediction payload must contain dict field 'predictions'")
    dt_s = 1.0 / float(frequency_hz)
    for horizon, xy in predictions.items():
        if not isinstance(xy, list):
            raise ValueError(f"predictions[{horizon!r}] must be a list")
        trajectory_from_xy_list(
            xy=xy,
            start_sample_token=str(payload.get("current_sample_token", "")),
            dt_s=dt_s,
            coordinate_frame=GLOBAL_FRAME,
            source_stage="prediction",
        )


def adapter_contract(
    *,
    stage: str,
    adapter_name: str,
    model_name: str,
    native_input: Sequence[str],
    standard_input: Sequence[str],
    standard_output: Sequence[str],
    used_by_model: Sequence[str],
    used_by_metrics: Sequence[str],
    assumptions: Optional[Sequence[str]] = None,
    implemented: bool = True,
) -> Dict[str, Any]:
    return {
        "schema": f"{SCHEMA_VERSION}.contract",
        "stage": str(stage),
        "adapter_name": str(adapter_name),
        "model_name": str(model_name),
        "implemented": bool(implemented),
        "native_input": [str(v) for v in native_input],
        "standard_input": [str(v) for v in standard_input],
        "standard_output": [str(v) for v in standard_output],
        "used_by_model": [str(v) for v in used_by_model],
        "used_by_metrics": [str(v) for v in used_by_metrics],
        "assumptions": [str(v) for v in (assumptions or [])],
        "unit_policy": {
            "position": DISTANCE_UNIT,
            "velocity": "meter_per_second",
            "acceleration": "meter_per_second_squared",
            "time": TIME_UNIT,
            "angle": ANGLE_UNIT,
            "default_dt_s": DEFAULT_DT_S,
        },
        "coordinate_policy": {
            "canonical_world_frame": GLOBAL_FRAME,
            "planner_local_frame": EGO_FRAME,
            "frame_changes_must_be_explicit": True,
        },
    }


def planner_contract(adapter_name: str) -> Dict[str, Any]:
    name = str(adapter_name).lower()
    common_input = [
        "ego history",
        "agent history/current boxes",
        "map",
        "route or fallback lane path",
        "traffic lights defaulted to green/empty when unavailable",
    ]
    if name == "tuplan_pdm":
        return adapter_contract(
            stage="planning",
            adapter_name=name,
            model_name="tuPlan PDM local prediction wrapper",
            native_input=["nuPlan PlannerInput", "nuPlan PlannerInitialization"],
            standard_input=["zz9.plan_input.v1"],
            standard_output=["zz9.plan_result.v1"],
            used_by_model=[
                "ego current state",
                "target/non-ego agent predicted future trajectory",
                "reference path or straight fallback route",
                "vehicle geometry",
            ],
            used_by_metrics=["Trajectron prediction future", "ego planned trajectory", "target/non-ego future boxes"],
            assumptions=[
                "local wrapper consumes Trajectron prediction directly and writes comparable ego trajectory metrics",
                "native tuPlan PDM execution still requires nuPlan map_api, route_roadblock_ids and PlannerInput",
                "traffic light status may be defaulted to green/empty for this experiment",
            ],
            implemented=True,
        )
    if name == "idm":
        return adapter_contract(
            stage="planning",
            adapter_name=name,
            model_name="IDM Planner",
            native_input=["nuPlan PlannerInput", "ego state", "DetectionsTracks current observation", "IDM parameters"],
            standard_input=["zz9.plan_input.v1"],
            standard_output=["zz9.plan_result.v1"],
            used_by_model=[
                "ego current speed/state",
                "target/non-ego current tracked object in DetectionsTracks",
                "reference path or lane centerline",
                "IDM desired speed, minimum gap, headway time and acceleration limits",
            ],
            used_by_metrics=["ego planned trajectory", "target/non-ego agent predicted future trajectory"],
            assumptions=[
                "IDM is a longitudinal car-following model and does not decide lane changes by itself",
                "nuPlan IDM consumes current DetectionsTracks; Trajectron future is attached/kept for metrics but is not used by the official IDM policy",
                "when nuPlan map route is unavailable, the adapter supplies an ego GT/reference path before calling official IDMPlanner",
                "traffic light status is ignored for this experiment",
            ],
            implemented=True,
        )
    if name == "pdm_open":
        return adapter_contract(
            stage="planning",
            adapter_name=name,
            model_name="tuPlan PDM-Open",
            native_input=["PDMFeature: ego_position, ego_velocity, ego_acceleration, planner_centerline"],
            standard_input=["zz9.plan_input.v1", "nuScenes ego pose/map DB", "tuPlan PDM-Open checkpoint"],
            standard_output=["zz9.plan_result.v1"],
            used_by_model=[
                "ego history sampled from nuScenes ego poses",
                "nuScenes map-derived centerline or straight fallback centerline",
                "PDM-Open checkpoint/model weights",
            ],
            used_by_metrics=[
                "ego planned trajectory",
                "ego future GT",
                "target/non-ego agent predicted future trajectory from previous prediction stage",
            ],
            assumptions=[
                "PDM-Open official model does not consume surrounding-agent prediction; prediction is read for downstream metrics/collision checks",
                "nuScenes ego_pose is treated as the ego rear-axle pose; if the upstream pose source is vehicle center, configure pdm_open.nuscenes_ego_pose_reference=center and rear_axle_to_center_m",
                "route roadblocks are replaced by a nuScenes map forward centerline because the current sample is not a nuPlan scenario",
                "direct use of a nuPlan-trained checkpoint on nuScenes has domain gap unless retrained/fine-tuned",
            ],
            implemented=True,
        )
    if name == "fot":
        return adapter_contract(
            stage="planning",
            adapter_name=name,
            model_name="PythonRobotics Frenet Optimal Trajectory",
            native_input=["reference centerline waypoints", "Cartesian ego state", "static 2D obstacle points", "FOT sampling/cost parameters"],
            standard_input=["zz9.plan_input.v1", "zz9 tracking.json", "target prediction from previous prediction stage"],
            standard_output=["zz9.plan_result.v1"],
            used_by_model=[
                "ego current pose and speed estimated from nuScenes ego poses",
                "nuScenes map-derived forward centerline or straight fallback centerline",
                "current tracking boxes converted to obstacle points",
                "target prediction points optionally converted to static obstacle points",
            ],
            used_by_metrics=[
                "ego planned trajectory",
                "ego future GT",
                "target/non-ego agent predicted future trajectory from previous prediction stage",
            ],
            assumptions=[
                "PythonRobotics FOT handles static 2D obstacle points; prediction points have no time dimension after conversion",
                "the planner is optimization/sampling based, not a learned planner",
                "route roadblocks are replaced by a nuScenes map forward centerline because the current sample is not a nuPlan scenario",
            ],
            implemented=True,
        )
    if name == "mpc":
        return adapter_contract(
            stage="planning",
            adapter_name=name,
            model_name="Prediction-conditioned MPC (numpy)",
            native_input=["reference centerline", "ego current state", "target prediction trajectory"],
            standard_input=["zz9.plan_input.v1", "target prediction from previous prediction stage"],
            standard_output=["zz9.plan_result.v1"],
            used_by_model=[
                "ego current pose and speed estimated from nuScenes ego poses",
                "nuScenes map-derived forward centerline or straight fallback centerline",
                "target prediction projected to longitudinal safety constraints",
            ],
            used_by_metrics=[
                "ego planned trajectory",
                "ego future GT",
                "target/non-ego agent predicted future trajectory from previous prediction stage",
            ],
            assumptions=[
                "planner is optimization based and enforces prediction-conditioned longitudinal safety through a soft penalty",
                "current implementation uses a dependency-light projected-gradient solver, not OSQP/CVXOPT",
                "route roadblocks are replaced by a nuScenes map forward centerline because the current sample is not a nuPlan scenario",
            ],
            implemented=True,
        )
    if name in {"plantf", "diffusion_planner"}:
        model_name = {
            "plantf": "PlanTF tracking-history wrapper",
            "diffusion_planner": "Diffusion-Planner tracking-history wrapper",
        }[name]
        return adapter_contract(
            stage="planning",
            adapter_name=name,
            model_name=model_name,
            native_input=["nuPlan PlannerInput", "nuPlan PlannerInitialization", "model checkpoint"],
            standard_input=["zz9.track_plan_input.v1"],
            standard_output=["zz9.plan_result.v1"],
            used_by_model=common_input,
            used_by_metrics=["ego planned trajectory", "target/all-agent tracking history", "constant-velocity extrapolated target future"],
            assumptions=[
                "native network is not executed by the local wrapper unless native_model_used=true in plan_result",
                "attacked tracking history is injected as the source of non-ego agent state",
                "target future is constant-velocity extrapolated from tracking history for local metric computation",
                "traffic light status may be defaulted to green/empty for this experiment",
            ],
            implemented=True,
        )
    if name == "stp3":
        return adapter_contract(
            stage="planning",
            adapter_name=name,
            model_name="ST-P3",
            native_input=["multi-view camera sequence", "nuScenes ego/map labels"],
            standard_input=["zz9 camera/sample inputs"],
            standard_output=["zz9.plan_result.v1"],
            used_by_model=["images", "ego history", "HD map labels"],
            used_by_metrics=["ego planned trajectory", "occupancy or agent boxes"],
            assumptions=["not a downstream consumer of Trajectron prediction JSON"],
            implemented=False,
        )
    if name == "uniad":
        return adapter_contract(
            stage="planning",
            adapter_name=name,
            model_name="UniAD",
            native_input=["multi-view camera queue", "can_bus", "HD map / gt labels"],
            standard_input=["zz9 camera/sample inputs"],
            standard_output=["zz9.plan_result.v1"],
            used_by_model=["images", "tracking queries", "motion/occupancy/planning heads"],
            used_by_metrics=["planning_traj", "future boxes/occupancy"],
            assumptions=["not a downstream consumer of Trajectron prediction JSON without model modification"],
            implemented=False,
        )
    raise ValueError(f"unknown planner adapter: {adapter_name}")


def standard_metadata(stage: str, adapter_name: str, model_name: str = "") -> Dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "stage": str(stage),
        "adapter_name": str(adapter_name),
        "model_name": str(model_name or adapter_name),
        "canonical_coordinate_frame": GLOBAL_FRAME,
        "unit_policy": {
            "distance": DISTANCE_UNIT,
            "time": TIME_UNIT,
            "angle": ANGLE_UNIT,
            "velocity": "meter_per_second",
            "acceleration": "meter_per_second_squared",
        },
    }
