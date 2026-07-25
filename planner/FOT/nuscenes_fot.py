from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from frenet_interface import (
    CartesianEgoState,
    FrenetPlannerConfig,
    FrenetPlanningResult,
    ObstaclePoint,
    plan_once,
)
from nuscenes_map import SimpleNuScenesMap


PredictionMode = Literal["none", "mean", "first", "aligned", "all"]


@dataclass(slots=True)
class NuScenesFOTPaths:
    nuscenes_root: str
    tracking_results_path: str
    prediction_results_path: str | None = None


@dataclass(slots=True)
class NuScenesFOTAdapterConfig:
    route_length_m: float = 80.0
    route_resolution_m: float = 1.0
    closest_lane_radius_m: float = 20.0
    prediction_mode: PredictionMode = "aligned"
    prediction_step_stride: int = 3
    tracking_box_scale: float = 1.0
    include_tracking_boxes: bool = True
    include_prediction_points: bool = False
    prediction_source_dt: float = 0.5
    align_prediction_to_planner_dt: bool = True


@dataclass(slots=True)
class TrackingBox2D:
    tracking_id: str
    center_x: float
    center_y: float
    center_z: float
    length: float
    width: float
    height: float
    yaw: float
    score: float | None = None


@dataclass(slots=True)
class PredictionTrajectory:
    source_track_id: str | None
    points_xy: list[tuple[float, float]]


@dataclass(slots=True)
class NuScenesPreparedFOTInput:
    requested_frame_token: str
    sample_token: str
    scene_name: str
    location: str
    tracking_frame_token: str | None
    reference_lane_tokens: list[str]
    reference_waypoint_x: list[float]
    reference_waypoint_y: list[float]
    ego_state: CartesianEgoState
    tracking_boxes: list[TrackingBox2D]
    prediction_trajectories: list[PredictionTrajectory]
    tracking_obstacle_points: list[ObstaclePoint]
    prediction_obstacle_points: list[ObstaclePoint]

    @property
    def obstacle_points(self) -> list[ObstaclePoint]:
        return self.tracking_obstacle_points + self.prediction_obstacle_points


@dataclass(slots=True)
class NuScenesFOTPlanOutput:
    prepared_input: NuScenesPreparedFOTInput
    planning_result: FrenetPlanningResult


class NuScenesFOTAdapter:
    def __init__(self, paths: NuScenesFOTPaths) -> None:
        self.paths = paths
        self.nuscenes_root = Path(paths.nuscenes_root)
        self.version_root = self.nuscenes_root / "v1.0-trainval"

        self.samples = _load_json(self.version_root / "sample.json")
        self.sample_data = _load_json(self.version_root / "sample_data.json")
        self.scenes = _load_json(self.version_root / "scene.json")
        self.logs = _load_json(self.version_root / "log.json")
        self.ego_poses = _load_json(self.version_root / "ego_pose.json")
        self.calibrated_sensors = _load_json(self.version_root / "calibrated_sensor.json")
        self.sensors = _load_json(self.version_root / "sensor.json")
        self.tracking_results = _load_json(Path(paths.tracking_results_path))["results"]
        self.prediction_payload = (
            _load_json(Path(paths.prediction_results_path))
            if paths.prediction_results_path
            else None
        )

        self.sample_by_token = {record["token"]: record for record in self.samples}
        self.sample_data_by_token = {record["token"]: record for record in self.sample_data}
        self.scene_by_token = {record["token"]: record for record in self.scenes}
        self.scene_by_name = {record["name"]: record for record in self.scenes}
        self.log_by_token = {record["token"]: record for record in self.logs}
        self.ego_pose_by_token = {record["token"]: record for record in self.ego_poses}
        self.calibrated_sensor_by_token = {
            record["token"]: record for record in self.calibrated_sensors
        }
        self.sensor_by_token = {record["token"]: record for record in self.sensors}

        self.channel_by_sample_data_token: dict[str, str] = {}
        self.cam_front_sample_data_tokens_by_sample_token: dict[str, list[str]] = defaultdict(list)
        for record in self.sample_data:
            calibrated = self.calibrated_sensor_by_token[record["calibrated_sensor_token"]]
            sensor = self.sensor_by_token[calibrated["sensor_token"]]
            channel = sensor["channel"]
            self.channel_by_sample_data_token[record["token"]] = channel
            if channel == "CAM_FRONT":
                self.cam_front_sample_data_tokens_by_sample_token[
                    record["sample_token"]
                ].append(record["token"])

        self.sample_token_by_tracking_token: dict[str, str] = {}
        for token in self.tracking_results:
            if token in self.sample_by_token:
                self.sample_token_by_tracking_token[token] = token
            elif token in self.sample_data_by_token:
                self.sample_token_by_tracking_token[token] = self.sample_data_by_token[token]["sample_token"]

        self.predictions_by_sample_token: dict[str, tuple[str, dict]] = {}
        if self.prediction_payload:
            for scene_name, scene_payload in self.prediction_payload["scenes"].items():
                for timestep_payload in scene_payload["timesteps"].values():
                    self.predictions_by_sample_token[timestep_payload["sample_token"]] = (
                        scene_name,
                        timestep_payload,
                    )

        self._map_cache: dict[str, SimpleNuScenesMap] = {}

    def get_cam_front_frame(self, frame_token: str) -> dict:
        if frame_token in self.sample_data_by_token:
            sample_data_token = frame_token
            sample_token = self.sample_data_by_token[frame_token]["sample_token"]
        else:
            sample_token = self.resolve_sample_token(frame_token)
            sample_data_token = self._select_cam_front_sample_data_token(sample_token)

        sample_data_record = self.sample_data_by_token[sample_data_token]
        calibrated_sensor = self.calibrated_sensor_by_token[
            sample_data_record["calibrated_sensor_token"]
        ]
        ego_pose = self.ego_pose_by_token[sample_data_record["ego_pose_token"]]
        return {
            "sample_token": sample_token,
            "sample_data_token": sample_data_token,
            "filename": sample_data_record["filename"],
            "width": sample_data_record["width"],
            "height": sample_data_record["height"],
            "timestamp": sample_data_record["timestamp"],
            "camera_intrinsic": calibrated_sensor["camera_intrinsic"],
            "calibrated_sensor_translation": calibrated_sensor["translation"],
            "calibrated_sensor_rotation": calibrated_sensor["rotation"],
            "ego_pose_translation": ego_pose["translation"],
            "ego_pose_rotation": ego_pose["rotation"],
        }

    def get_scene_sample_tokens(
        self,
        *,
        frame_token: str | None = None,
        scene_name: str | None = None,
    ) -> list[str]:
        if scene_name:
            if scene_name not in self.scene_by_name:
                raise KeyError(f"unknown scene_name: {scene_name}")
            scene_record = self.scene_by_name[scene_name]
        elif frame_token:
            sample_token = self.resolve_sample_token(frame_token)
            scene_record = self.scene_by_token[self.sample_by_token[sample_token]["scene_token"]]
        else:
            raise ValueError("either frame_token or scene_name must be provided")

        sample_tokens: list[str] = []
        sample_token = scene_record["first_sample_token"]
        while sample_token:
            sample_tokens.append(sample_token)
            sample_token = self.sample_by_token[sample_token]["next"]
        return sample_tokens

    def has_cam_front_image(self, frame_token: str) -> bool:
        camera_frame = self.get_cam_front_frame(frame_token)
        image_path = self.nuscenes_root / camera_frame["filename"]
        return image_path.exists()

    def has_tracking_for_sample(self, sample_token: str) -> bool:
        return self._select_tracking_frame_token(sample_token) is not None

    def has_prediction_for_sample(self, sample_token: str) -> bool:
        return sample_token in self.predictions_by_sample_token

    def plan_frame(
        self,
        frame_token: str,
        *,
        planner_config: FrenetPlannerConfig | None = None,
        adapter_config: NuScenesFOTAdapterConfig | None = None,
    ) -> NuScenesFOTPlanOutput:
        effective_planner_config = planner_config or FrenetPlannerConfig()
        prepared_input = self.prepare_input(
            frame_token=frame_token,
            adapter_config=adapter_config,
            planner_dt=effective_planner_config.dt,
        )
        planning_result = plan_once(
            prepared_input.reference_waypoint_x,
            prepared_input.reference_waypoint_y,
            obstacles=prepared_input.obstacle_points,
            ego_cartesian_state=prepared_input.ego_state,
            config=effective_planner_config,
        )
        return NuScenesFOTPlanOutput(
            prepared_input=prepared_input,
            planning_result=planning_result,
        )

    def prepare_input(
        self,
        frame_token: str,
        *,
        adapter_config: NuScenesFOTAdapterConfig | None = None,
        planner_dt: float | None = None,
    ) -> NuScenesPreparedFOTInput:
        adapter_config = adapter_config or NuScenesFOTAdapterConfig()
        sample_token = self.resolve_sample_token(frame_token)
        sample_record = self.sample_by_token[sample_token]
        scene_record = self.scene_by_token[sample_record["scene_token"]]
        scene_name = scene_record["name"]
        location = self.log_by_token[scene_record["log_token"]]["location"]

        ego_state = self._build_ego_state(frame_token)
        map_api = self._get_map(location)
        lane_tokens, centerline = map_api.build_forward_centerline(
            ego_state.x,
            ego_state.y,
            ego_state.yaw,
            route_length_m=adapter_config.route_length_m,
            resolution_meters=adapter_config.route_resolution_m,
            closest_lane_radius=adapter_config.closest_lane_radius_m,
        )

        tracking_frame_token = self._select_tracking_frame_token(sample_token)
        tracking_boxes = (
            self._build_tracking_boxes(
                tracking_frame_token,
                box_scale=adapter_config.tracking_box_scale,
            )
            if adapter_config.include_tracking_boxes
            else []
        )
        prediction_trajectories = self._build_prediction_trajectories(
            sample_token,
            mode=adapter_config.prediction_mode,
            step_stride=adapter_config.prediction_step_stride,
            planner_dt=planner_dt if adapter_config.align_prediction_to_planner_dt else None,
            prediction_source_dt=adapter_config.prediction_source_dt,
        )
        tracking_obstacles = (
            self._tracking_boxes_to_obstacle_points(tracking_boxes)
            if adapter_config.include_tracking_boxes
            else []
        )
        prediction_obstacles = (
            self._prediction_trajectories_to_obstacle_points(
                prediction_trajectories,
            )
            if adapter_config.include_prediction_points
            else []
        )

        return NuScenesPreparedFOTInput(
            requested_frame_token=frame_token,
            sample_token=sample_token,
            scene_name=scene_name,
            location=location,
            tracking_frame_token=tracking_frame_token,
            reference_lane_tokens=lane_tokens,
            reference_waypoint_x=[float(point[0]) for point in centerline],
            reference_waypoint_y=[float(point[1]) for point in centerline],
            ego_state=ego_state,
            tracking_boxes=tracking_boxes,
            prediction_trajectories=prediction_trajectories,
            tracking_obstacle_points=tracking_obstacles,
            prediction_obstacle_points=prediction_obstacles,
        )

    def resolve_sample_token(self, frame_token: str) -> str:
        if frame_token in self.sample_by_token:
            return frame_token
        if frame_token in self.sample_data_by_token:
            return self.sample_data_by_token[frame_token]["sample_token"]
        raise KeyError(f"unknown nuScenes frame token: {frame_token}")

    def _get_map(self, location: str) -> SimpleNuScenesMap:
        if location not in self._map_cache:
            self._map_cache[location] = SimpleNuScenesMap(self.nuscenes_root, location)
        return self._map_cache[location]

    def get_map_api(self, location: str) -> SimpleNuScenesMap:
        return self._get_map(location)

    def _build_ego_state(self, frame_token: str) -> CartesianEgoState:
        if frame_token in self.sample_data_by_token:
            return self._build_ego_state_from_sample_data(frame_token)
        if frame_token in self.sample_by_token:
            return self._build_ego_state_from_sample(frame_token)
        raise KeyError(f"ego state token not found: {frame_token}")

    def _build_ego_state_from_sample(self, sample_token: str) -> CartesianEgoState:
        cam_front_token = self._select_cam_front_sample_data_token(sample_token)
        ego_pose = self.ego_pose_by_token[
            self.sample_data_by_token[cam_front_token]["ego_pose_token"]
        ]
        speed = self._estimate_speed_from_sample(sample_token)
        acceleration = self._estimate_acceleration_from_sample(sample_token)
        return CartesianEgoState(
            x=float(ego_pose["translation"][0]),
            y=float(ego_pose["translation"][1]),
            yaw=_quaternion_to_yaw(ego_pose["rotation"]),
            speed=speed,
            acceleration=acceleration,
            curvature=0.0,
        )

    def _build_ego_state_from_sample_data(self, sample_data_token: str) -> CartesianEgoState:
        sample_data_record = self.sample_data_by_token[sample_data_token]
        ego_pose = self.ego_pose_by_token[sample_data_record["ego_pose_token"]]
        speed = self._estimate_speed_from_sample_data(sample_data_token)
        acceleration = self._estimate_acceleration_from_sample_data(sample_data_token)
        return CartesianEgoState(
            x=float(ego_pose["translation"][0]),
            y=float(ego_pose["translation"][1]),
            yaw=_quaternion_to_yaw(ego_pose["rotation"]),
            speed=speed,
            acceleration=acceleration,
            curvature=0.0,
        )

    def _select_cam_front_sample_data_token(self, sample_token: str) -> str:
        candidates = self.cam_front_sample_data_tokens_by_sample_token.get(sample_token, [])
        if not candidates:
            raise KeyError(f"no CAM_FRONT sample_data found for sample_token={sample_token}")

        keyframes = [
            token for token in candidates if self.sample_data_by_token[token]["is_key_frame"]
        ]
        if keyframes:
            return min(
                keyframes,
                key=lambda token: abs(
                    self.sample_data_by_token[token]["timestamp"]
                    - self.sample_by_token[sample_token]["timestamp"]
                ),
            )
        return min(
            candidates,
            key=lambda token: abs(
                self.sample_data_by_token[token]["timestamp"]
                - self.sample_by_token[sample_token]["timestamp"]
            ),
        )

    def _select_tracking_frame_token(self, sample_token: str) -> str | None:
        candidates = [
            token
            for token in self.tracking_results
            if self.sample_token_by_tracking_token.get(token) == sample_token
        ]
        if not candidates:
            return None

        if sample_token in candidates:
            return sample_token

        sample_timestamp = self.sample_by_token[sample_token]["timestamp"]
        keyframes = [
            token
            for token in candidates
            if token in self.sample_data_by_token and self.sample_data_by_token[token]["is_key_frame"]
        ]
        if keyframes:
            return min(
                keyframes,
                key=lambda token: abs(
                    self.sample_data_by_token[token]["timestamp"] - sample_timestamp
                ),
            )
        return min(
            [token for token in candidates if token in self.sample_data_by_token],
            key=lambda token: abs(
                self.sample_data_by_token[token]["timestamp"] - sample_timestamp
            ),
        ) if any(token in self.sample_data_by_token for token in candidates) else None

    def _build_tracking_boxes(
        self,
        tracking_frame_token: str | None,
        *,
        box_scale: float,
    ) -> list[TrackingBox2D]:
        if tracking_frame_token is None:
            return []

        tracking_boxes: list[TrackingBox2D] = []
        for track in self.tracking_results.get(tracking_frame_token, []):
            tracking_boxes.append(_track_record_to_tracking_box(track, box_scale=box_scale))
        return tracking_boxes

    def _build_prediction_trajectories(
        self,
        sample_token: str,
        *,
        mode: PredictionMode,
        step_stride: int,
        planner_dt: float | None,
        prediction_source_dt: float,
    ) -> list[PredictionTrajectory]:
        if mode == "none":
            return []
        if sample_token not in self.predictions_by_sample_token:
            return []

        _, timestep_payload = self.predictions_by_sample_token[sample_token]
        trajectories_out: list[PredictionTrajectory] = []
        for node_payload in timestep_payload["nodes"].values():
            predictions = node_payload["predictions_global"]
            if not predictions:
                continue

            if mode == "mean":
                mean_prediction = np.mean(np.asarray(predictions, dtype=float), axis=0)
                trajectories = [mean_prediction.tolist()]
            elif mode == "first":
                trajectories = [predictions[0]]
            elif mode == "aligned":
                trajectories = [_select_aligned_prediction_sample(node_payload, predictions)]
            else:
                trajectories = predictions

            for trajectory in trajectories:
                trajectory_points = [
                    (float(point[0]), float(point[1]))
                    for point in trajectory
                ]
                if planner_dt is not None and planner_dt > 0.0 and prediction_source_dt > 0.0:
                    trajectory_points = _resample_xy_trajectory(
                        trajectory_points,
                        src_dt=prediction_source_dt,
                        dst_dt=planner_dt,
                    )
                points_xy = [
                    (float(point[0]), float(point[1]))
                    for point in trajectory_points[:: max(1, step_stride)]
                ]
                if points_xy:
                    trajectories_out.append(
                        PredictionTrajectory(
                            source_track_id=_prediction_source_track_id(node_payload),
                            points_xy=points_xy,
                        )
                    )
        return trajectories_out

    def _tracking_boxes_to_obstacle_points(
        self,
        tracking_boxes: list[TrackingBox2D],
    ) -> list[ObstaclePoint]:
        obstacle_points: list[ObstaclePoint] = []
        for box in tracking_boxes:
            obstacle_points.extend(_tracking_box_to_obstacle_points(box))
        return obstacle_points

    def _prediction_trajectories_to_obstacle_points(
        self,
        prediction_trajectories: list[PredictionTrajectory],
    ) -> list[ObstaclePoint]:
        obstacle_points: list[ObstaclePoint] = []
        for trajectory in prediction_trajectories:
            for x, y in trajectory.points_xy:
                obstacle_points.append(ObstaclePoint(x=x, y=y))
        return obstacle_points

    def _estimate_speed_from_sample(self, sample_token: str) -> float:
        sample_record = self.sample_by_token[sample_token]
        previous_token = sample_record["prev"]
        if not previous_token:
            next_token = sample_record["next"]
            if not next_token:
                return 0.0
            return self._estimate_speed_between_samples(sample_token, next_token)
        return self._estimate_speed_between_samples(previous_token, sample_token)

    def _estimate_acceleration_from_sample(self, sample_token: str) -> float:
        sample_record = self.sample_by_token[sample_token]
        previous_token = sample_record["prev"]
        if not previous_token:
            return 0.0
        previous_sample = self.sample_by_token[previous_token]
        if not previous_sample["prev"]:
            return 0.0

        v0 = self._estimate_speed_between_samples(previous_sample["prev"], previous_token)
        v1 = self._estimate_speed_between_samples(previous_token, sample_token)
        dt = (
            self.sample_by_token[sample_token]["timestamp"]
            - self.sample_by_token[previous_token]["timestamp"]
        ) / 1.0e6
        if dt <= 0.0:
            return 0.0
        return (v1 - v0) / dt

    def _estimate_speed_between_samples(self, token0: str, token1: str) -> float:
        pose0 = self._ego_pose_from_sample(token0)
        pose1 = self._ego_pose_from_sample(token1)
        dt = (
            self.sample_by_token[token1]["timestamp"]
            - self.sample_by_token[token0]["timestamp"]
        ) / 1.0e6
        if dt <= 0.0:
            return 0.0
        dx = pose1["translation"][0] - pose0["translation"][0]
        dy = pose1["translation"][1] - pose0["translation"][1]
        return math.hypot(dx, dy) / dt

    def _ego_pose_from_sample(self, sample_token: str) -> dict:
        cam_front_token = self._select_cam_front_sample_data_token(sample_token)
        ego_pose_token = self.sample_data_by_token[cam_front_token]["ego_pose_token"]
        return self.ego_pose_by_token[ego_pose_token]

    def _estimate_speed_from_sample_data(self, sample_data_token: str) -> float:
        current = self.sample_data_by_token[sample_data_token]
        previous_token = current["prev"]
        if previous_token and previous_token in self.sample_data_by_token:
            previous = self.sample_data_by_token[previous_token]
            return self._estimate_speed_between_sample_data(previous["token"], current["token"])

        next_token = current["next"]
        if next_token and next_token in self.sample_data_by_token:
            return self._estimate_speed_between_sample_data(current["token"], next_token)
        return 0.0

    def _estimate_acceleration_from_sample_data(self, sample_data_token: str) -> float:
        current = self.sample_data_by_token[sample_data_token]
        previous_token = current["prev"]
        if not previous_token or previous_token not in self.sample_data_by_token:
            return 0.0
        previous = self.sample_data_by_token[previous_token]
        previous_previous_token = previous["prev"]
        if not previous_previous_token or previous_previous_token not in self.sample_data_by_token:
            return 0.0

        v0 = self._estimate_speed_between_sample_data(previous_previous_token, previous_token)
        v1 = self._estimate_speed_between_sample_data(previous_token, sample_data_token)
        dt = (current["timestamp"] - previous["timestamp"]) / 1.0e6
        if dt <= 0.0:
            return 0.0
        return (v1 - v0) / dt

    def _estimate_speed_between_sample_data(self, token0: str, token1: str) -> float:
        record0 = self.sample_data_by_token[token0]
        record1 = self.sample_data_by_token[token1]
        pose0 = self.ego_pose_by_token[record0["ego_pose_token"]]
        pose1 = self.ego_pose_by_token[record1["ego_pose_token"]]
        dt = (record1["timestamp"] - record0["timestamp"]) / 1.0e6
        if dt <= 0.0:
            return 0.0
        dx = pose1["translation"][0] - pose0["translation"][0]
        dy = pose1["translation"][1] - pose0["translation"][1]
        return math.hypot(dx, dy) / dt


def _load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def _quaternion_to_yaw(quaternion_wxyz: list[float]) -> float:
    w, x, y, z = quaternion_wxyz
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _track_record_to_tracking_box(
    track_record: dict,
    *,
    box_scale: float,
) -> TrackingBox2D:
    width = float(track_record["size"][0]) * box_scale
    length = float(track_record["size"][1]) * box_scale
    height = float(track_record["size"][2]) if len(track_record["size"]) > 2 else 1.5
    return TrackingBox2D(
        tracking_id=str(track_record.get("tracking_id", "")),
        center_x=float(track_record["translation"][0]),
        center_y=float(track_record["translation"][1]),
        center_z=float(track_record["translation"][2]) if len(track_record["translation"]) > 2 else 0.0,
        length=length,
        width=width,
        height=height,
        yaw=_quaternion_to_yaw(track_record["rotation"]),
        score=float(track_record["tracking_score"]) if "tracking_score" in track_record else None,
    )


def _tracking_box_to_obstacle_points(box: TrackingBox2D) -> list[ObstaclePoint]:
    center_x = box.center_x
    center_y = box.center_y
    width = box.width
    length = box.length
    yaw = box.yaw

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    half_length = 0.5 * length
    half_width = 0.5 * width

    local_points = [
        (0.0, 0.0),
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, half_width),
        (-half_length, -half_width),
    ]
    obstacle_points: list[ObstaclePoint] = []
    for local_x, local_y in local_points:
        global_x = center_x + cos_yaw * local_x - sin_yaw * local_y
        global_y = center_y + sin_yaw * local_x + cos_yaw * local_y
        obstacle_points.append(ObstaclePoint(global_x, global_y))
    return obstacle_points


def _prediction_source_track_id(node_payload: dict) -> str | None:
    if "source_track_id" in node_payload and node_payload["source_track_id"] is not None:
        return str(node_payload["source_track_id"])
    if "instance_token" in node_payload and node_payload["instance_token"] is not None:
        return str(node_payload["instance_token"])
    return None


def _select_aligned_prediction_sample(node_payload: dict, predictions: list) -> list:
    history = node_payload.get("history_global", [])
    if len(history) < 2:
        return predictions[0]

    history_vec = np.asarray(history[-1], dtype=float) - np.asarray(history[-2], dtype=float)
    history_norm = float(np.linalg.norm(history_vec))
    if history_norm < 1.0e-3:
        return predictions[0]
    history_dir = history_vec / history_norm

    best_score = -float("inf")
    best_prediction = predictions[0]
    for prediction in predictions:
        if len(prediction) < 2:
            continue
        pred_vec = np.asarray(prediction[1], dtype=float) - np.asarray(prediction[0], dtype=float)
        pred_norm = float(np.linalg.norm(pred_vec))
        if pred_norm < 1.0e-3:
            score = -1.0
        else:
            pred_dir = pred_vec / pred_norm
            score = float(np.dot(history_dir, pred_dir))
        if score > best_score:
            best_score = score
            best_prediction = prediction
    return best_prediction


def _resample_xy_trajectory(
    points_xy: list[tuple[float, float]],
    *,
    src_dt: float,
    dst_dt: float,
) -> list[tuple[float, float]]:
    if len(points_xy) <= 1:
        return points_xy
    if abs(src_dt - dst_dt) <= 1.0e-6:
        return points_xy

    source = np.asarray(points_xy, dtype=float)
    t_src = np.arange(len(source), dtype=float) * src_dt
    t_end = float(t_src[-1])
    num_samples = max(2, int(math.floor(t_end / dst_dt)) + 1)
    t_dst = np.arange(num_samples, dtype=float) * dst_dt
    if t_dst[-1] < t_end:
        t_dst = np.append(t_dst, t_end)
    x_dst = np.interp(t_dst, t_src, source[:, 0])
    y_dst = np.interp(t_dst, t_src, source[:, 1])
    return [(float(x), float(y)) for x, y in np.column_stack([x_dst, y_dst])]
