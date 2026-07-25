from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from hivt_adapter import HivtPredictor, PredictionConfig, select_history_xy


VEHICLE_NAMES = {
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "motorcycle",
    "bicycle",
}


def yaw_from_quaternion(quaternion_wxyz: List[float]) -> float:
    import math

    w, x, y, z = [float(v) for v in quaternion_wxyz]
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def coarse_category(name: str) -> str:
    name = str(name or "").lower()
    if name in VEHICLE_NAMES or name.startswith("vehicle."):
        return "vehicle"
    if name == "pedestrian" or "pedestrian" in name:
        return "pedestrian"
    return name.split(".", 1)[0]


def squared_distance_xy(left: List[float], right: List[float]) -> float:
    return float((float(left[0]) - float(right[0])) ** 2 + (float(left[1]) - float(right[1])) ** 2)


@dataclass
class NuScenesPaths:
    nuscenes_root: str
    tracking_json: str


def load_json(path: Path) -> Any:
    with path.open("r") as handle:
        return json.load(handle)


class NuScenesIndex:
    def __init__(self, nuscenes_root: Union[str, Path]) -> None:
        self.nuscenes_root = Path(nuscenes_root)
        version_root = self.nuscenes_root / "v1.0-trainval"
        self.samples = load_json(version_root / "sample.json")
        self.sample_data = load_json(version_root / "sample_data.json")
        self.scenes = load_json(version_root / "scene.json")
        self.logs = load_json(version_root / "log.json")
        self.sample_annotations = load_json(version_root / "sample_annotation.json")
        self.instances = load_json(version_root / "instance.json")
        self.categories = load_json(version_root / "category.json")
        self.ego_poses = load_json(version_root / "ego_pose.json")
        self.calibrated_sensors = load_json(version_root / "calibrated_sensor.json")
        self.sensors = load_json(version_root / "sensor.json")
        self.sample_by_token = {record["token"]: record for record in self.samples}
        self.sample_data_by_token = {record["token"]: record for record in self.sample_data}
        self.scene_by_token = {record["token"]: record for record in self.scenes}
        self.log_by_token = {record["token"]: record for record in self.logs}
        self.annotation_by_token = {record["token"]: record for record in self.sample_annotations}
        self.annotations_by_sample_token: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in self.sample_annotations:
            self.annotations_by_sample_token[record["sample_token"]].append(record)
        self.instance_by_token = {record["token"]: record for record in self.instances}
        self.category_by_token = {record["token"]: record for record in self.categories}
        self.ego_pose_by_token = {record["token"]: record for record in self.ego_poses}
        self.calibrated_sensor_by_token = {
            record["token"]: record for record in self.calibrated_sensors
        }
        self.sensor_by_token = {record["token"]: record for record in self.sensors}
        self.cam_front_by_sample_token: Dict[str, List[str]] = defaultdict(list)
        for record in self.sample_data:
            calibrated = self.calibrated_sensor_by_token[record["calibrated_sensor_token"]]
            sensor = self.sensor_by_token[calibrated["sensor_token"]]
            if sensor["channel"] == "CAM_FRONT":
                self.cam_front_by_sample_token[record["sample_token"]].append(record["token"])

    def resolve_sample_token(self, frame_token: str) -> str:
        if frame_token in self.sample_by_token:
            return frame_token
        if frame_token in self.sample_data_by_token:
            return self.sample_data_by_token[frame_token]["sample_token"]
        raise KeyError(f"unknown nuScenes token: {frame_token}")

    def sample_timestamp(self, frame_token: str) -> int:
        if frame_token in self.sample_data_by_token:
            return int(self.sample_data_by_token[frame_token]["timestamp"])
        return int(self.sample_by_token[self.resolve_sample_token(frame_token)]["timestamp"])

    def sample_history_before(
        self,
        cutoff_sample_token: str,
        *,
        history_seconds: float,
        dt: float,
    ) -> List[str]:
        max_frames = max(1, int(float(history_seconds) / float(dt)))
        tokens: List[str] = []
        current = self.sample_by_token[cutoff_sample_token].get("prev") or ""
        while current and len(tokens) < max_frames:
            tokens.append(current)
            current = self.sample_by_token[current].get("prev") or ""
        tokens.reverse()
        return tokens

    def scene_name(self, sample_token: str) -> str:
        scene_token = self.sample_by_token[sample_token]["scene_token"]
        return self.scene_by_token[scene_token]["name"]

    def location(self, sample_token: str) -> str:
        scene_token = self.sample_by_token[sample_token]["scene_token"]
        scene_record = self.scene_by_token[scene_token]
        return self.log_by_token[scene_record["log_token"]]["location"]

    def ego_state(self, frame_token: str) -> Dict[str, float]:
        cam_frame = self.cam_front_frame(frame_token)
        ego = self.ego_pose_by_token[cam_frame["ego_pose_token"]]
        return {
            "x": float(ego["translation"][0]),
            "y": float(ego["translation"][1]),
            "yaw": float(yaw_from_quaternion(ego["rotation"])),
        }

    def cam_front_frame(self, frame_token: str) -> Dict[str, Any]:
        sample_token = self.resolve_sample_token(frame_token)
        candidates = self.cam_front_by_sample_token.get(sample_token, [])
        if not candidates:
            raise KeyError(f"no CAM_FRONT frame for sample_token={sample_token}")
        sample_timestamp = int(self.sample_by_token[sample_token]["timestamp"])
        keyframes = [token for token in candidates if self.sample_data_by_token[token]["is_key_frame"]]
        selected = keyframes if keyframes else candidates
        sample_data_token = min(
            selected,
            key=lambda token: abs(int(self.sample_data_by_token[token]["timestamp"]) - sample_timestamp),
        )
        sample_data = self.sample_data_by_token[sample_data_token]
        calibrated = self.calibrated_sensor_by_token[sample_data["calibrated_sensor_token"]]
        ego_pose = self.ego_pose_by_token[sample_data["ego_pose_token"]]
        return {
            "sample_token": sample_token,
            "sample_data_token": sample_data_token,
            "filename": sample_data["filename"],
            "width": sample_data["width"],
            "height": sample_data["height"],
            "timestamp": sample_data["timestamp"],
            "ego_pose_token": sample_data["ego_pose_token"],
            "camera_intrinsic": calibrated["camera_intrinsic"],
            "calibrated_sensor_translation": calibrated["translation"],
            "calibrated_sensor_rotation": calibrated["rotation"],
            "ego_pose_translation": ego_pose["translation"],
            "ego_pose_rotation": ego_pose["rotation"],
        }

    def annotation_category_name(self, annotation: Dict[str, Any]) -> str:
        instance = self.instance_by_token[annotation["instance_token"]]
        return str(self.category_by_token[instance["category_token"]]["name"])

    def match_annotation(
        self,
        track: Dict[str, Any],
        sample_token: str,
        *,
        max_distance_m: float = 6.0,
    ) -> Optional[Dict[str, Any]]:
        track_category = coarse_category(str(track.get("tracking_name", "")))
        track_xy = track.get("translation") or [0.0, 0.0, 0.0]
        best: Optional[Dict[str, Any]] = None
        best_dist2 = float(max_distance_m) ** 2
        for annotation in self.annotations_by_sample_token.get(sample_token, []):
            if coarse_category(self.annotation_category_name(annotation)) != track_category:
                continue
            dist2 = squared_distance_xy(track_xy, annotation["translation"])
            if dist2 < best_dist2:
                best = annotation
                best_dist2 = dist2
        return best

    def annotation_history_before_input(
        self,
        track: Dict[str, Any],
        *,
        current_sample_token: str,
        cutoff_sample_token: str,
        history_seconds: float,
        dt: float,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        matched = self.match_annotation(track, current_sample_token)
        if matched is None:
            return [], None

        cutoff_timestamp = int(self.sample_by_token[cutoff_sample_token]["timestamp"])
        max_history = max(1, int(float(history_seconds) / float(dt)))
        records: List[Dict[str, Any]] = []
        token = matched.get("prev") or ""
        while token and len(records) < max_history:
            annotation = self.annotation_by_token[token]
            sample_timestamp = int(self.sample_by_token[annotation["sample_token"]]["timestamp"])
            if sample_timestamp < cutoff_timestamp:
                records.append(annotation)
            token = annotation.get("prev") or ""
        records.reverse()
        return records, str(matched["instance_token"])


def current_track_as_history_record(track: Dict[str, Any]) -> Dict[str, Any]:
    return {"translation": track["translation"]}


def tracking_history_until(
    *,
    tracking_results: Dict[str, List[Dict]],
    scene_frame_tokens: List[str],
    current_index: int,
    track_id: str,
    max_records: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for frame_token in scene_frame_tokens[: current_index + 1]:
        box = next(
            (item for item in tracking_results.get(frame_token, []) if str(item.get("tracking_id", "")) == str(track_id)),
            None,
        )
        if box is not None:
            records.append(current_track_as_history_record(box))
    return records[-max(1, int(max_records)) :]


def select_keyframe_tracking_tokens(nusc: NuScenesIndex, tracking_results: Dict[str, List[Dict]]) -> List[str]:
    sample_to_candidates: Dict[str, List[str]] = defaultdict(list)
    for token in tracking_results:
        if token in nusc.sample_by_token:
            sample_to_candidates[token].append(token)
        elif token in nusc.sample_data_by_token:
            sample_to_candidates[nusc.sample_data_by_token[token]["sample_token"]].append(token)

    selected_tokens: List[str] = []
    for sample_token, candidates in sample_to_candidates.items():
        if sample_token in candidates:
            selected_tokens.append(sample_token)
            continue
        keyframes = [
            token
            for token in candidates
            if token in nusc.sample_data_by_token and bool(nusc.sample_data_by_token[token]["is_key_frame"])
        ]
        if keyframes:
            sample_ts = int(nusc.sample_by_token[sample_token]["timestamp"])
            selected = min(
                keyframes,
                key=lambda token: abs(int(nusc.sample_data_by_token[token]["timestamp"]) - sample_ts),
            )
            selected_tokens.append(selected)
    selected_tokens.sort(key=lambda token: nusc.sample_timestamp(token))
    return selected_tokens


def run_prediction(config_path: Union[str, Path]) -> Path:
    config = yaml.safe_load(Path(config_path).read_text())
    paths = config["paths"]
    run_cfg = config["run"]
    pred_cfg = PredictionConfig(**config["prediction"])

    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    nusc = NuScenesIndex(paths["nuscenes_root"])
    tracking_json = load_json(Path(paths["tracking_json"]))
    tracking_results = tracking_json["results"]
    all_frame_tokens = select_keyframe_tracking_tokens(nusc, tracking_results)
    if run_cfg.get("max_frames") is not None:
        all_frame_tokens = all_frame_tokens[: int(run_cfg["max_frames"])]

    grouped_scene_frames: Dict[str, List[str]] = defaultdict(list)
    for frame_token in all_frame_tokens:
        sample_token = nusc.resolve_sample_token(frame_token)
        grouped_scene_frames[nusc.scene_name(sample_token)].append(frame_token)

    scene_filter = set(run_cfg.get("scene_names") or [])
    if scene_filter:
        grouped_scene_frames = {
            scene_name: frames for scene_name, frames in grouped_scene_frames.items() if scene_name in scene_filter
        }

    predictor = HivtPredictor(
        pred_cfg,
        hivt_repo_root=paths.get("hivt_repo_root"),
        hivt_ckpt_path=paths.get("hivt_ckpt_path"),
    )
    prediction_json = {
        "meta": {
            "backend": pred_cfg.backend,
            "pred_steps": pred_cfg.pred_steps,
            "dt": pred_cfg.dt,
            "num_modes": pred_cfg.num_modes,
            "tracking_json_path": str(paths["tracking_json"]),
            "history_source": str(pred_cfg.__dict__.get("history_source", "gt_before_input_plus_current_tracking")),
        },
        "scenes": {},
    }

    for scene_name, scene_frame_tokens in grouped_scene_frames.items():
        scene_payload = {
            "timesteps": {},
        }
        cutoff_sample_token = nusc.resolve_sample_token(scene_frame_tokens[0])
        for timestep_index, frame_token in enumerate(scene_frame_tokens):
            sample_token = nusc.resolve_sample_token(frame_token)
            location = nusc.location(sample_token)
            ego_state = nusc.ego_state(frame_token)
            current_tracks = tracking_results.get(frame_token, [])
            nodes = {}
            history_xy_by_track: Dict[str, List[Tuple[float, float]]] = {}
            history_source = str(getattr(pred_cfg, "history_source", "gt_before_input_plus_current_tracking"))
            max_history_records = max(2, int(float(pred_cfg.history_seconds) / float(pred_cfg.dt)))
            for track in current_tracks:
                track_id = str(track["tracking_id"])
                instance_token = None
                if history_source in {"tracking", "tracking_only", "tracking_history"}:
                    history_until_now = tracking_history_until(
                        tracking_results=tracking_results,
                        scene_frame_tokens=scene_frame_tokens,
                        current_index=timestep_index,
                        track_id=track_id,
                        max_records=max_history_records,
                    )
                    matched = nusc.match_annotation(track, sample_token)
                    if matched is not None:
                        instance_token = str(matched["instance_token"])
                elif history_source in {"tracking_plus_gt_before_input", "gt_plus_tracking"}:
                    gt_history, instance_token = nusc.annotation_history_before_input(
                        track,
                        current_sample_token=sample_token,
                        cutoff_sample_token=cutoff_sample_token,
                        history_seconds=float(pred_cfg.history_seconds),
                        dt=float(pred_cfg.dt),
                    )
                    history_until_now = gt_history + tracking_history_until(
                        tracking_results=tracking_results,
                        scene_frame_tokens=scene_frame_tokens,
                        current_index=timestep_index,
                        track_id=track_id,
                        max_records=max_history_records,
                    )
                else:
                    gt_history, instance_token = nusc.annotation_history_before_input(
                        track,
                        current_sample_token=sample_token,
                        cutoff_sample_token=cutoff_sample_token,
                        history_seconds=float(pred_cfg.history_seconds),
                        dt=float(pred_cfg.dt),
                    )
                    history_until_now = gt_history + [current_track_as_history_record(track)]
                history_xy = select_history_xy(
                    history_until_now,
                    history_seconds=float(pred_cfg.history_seconds),
                    dt=float(pred_cfg.dt),
                )
                history_xy_by_track[track_id] = history_xy
                nodes[track_id] = {
                    "source_track_id": track_id,
                    "matched_instance_token": instance_token,
                    "source_category": track.get("tracking_name", "car"),
                    "history_global": [[x, y] for x, y in history_xy],
                    "predictions_global": [],
                }

            ego_sample_tokens = nusc.sample_history_before(
                cutoff_sample_token,
                history_seconds=float(pred_cfg.history_seconds),
                dt=float(pred_cfg.dt),
            ) + [sample_token]
            ego_history_xy = select_history_xy(
                [
                    {
                        "translation": [state["x"], state["y"], 0.0],
                    }
                    for state in [nusc.ego_state(token) for token in ego_sample_tokens]
                ],
                history_seconds=float(pred_cfg.history_seconds),
                dt=float(pred_cfg.dt),
            )
            prediction_by_track_id = predictor.predict_scene(
                current_tracks=current_tracks,
                history_xy_by_track=history_xy_by_track,
                ego_history_xy=ego_history_xy,
                ego_yaw=float(ego_state["yaw"]),
                location=location,
                nuscenes_root=nusc.nuscenes_root,
            )
            for track_id, node in nodes.items():
                prediction_modes = prediction_by_track_id.get(track_id, [])
                node["predictions_global"] = [
                    [[x, y] for x, y in mode_xy]
                    for mode_xy in prediction_modes
                ]

            scene_payload["timesteps"][str(timestep_index)] = {
                "frame_token": frame_token,
                "sample_token": sample_token,
                "location": location,
                "nodes": nodes,
            }

        prediction_json["scenes"][scene_name] = scene_payload

    prediction_json_path = output_dir / "predictions.json"
    with prediction_json_path.open("w") as handle:
        json.dump(prediction_json, handle, indent=2)
    return prediction_json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="HiVT nuScenes tracking-to-prediction wrapper")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    prediction_json_path = run_prediction(args.config)
    print("prediction_json:", prediction_json_path)


if __name__ == "__main__":
    main()
