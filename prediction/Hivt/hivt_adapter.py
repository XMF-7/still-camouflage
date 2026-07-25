from __future__ import annotations

import math
import pickle
import sys
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from nuscenes_map import SimpleNuScenesMap


DEFAULT_NUSCENES_DEVKIT_ROOT = Path(
    "/home/jushuo/Code/zz6-trajectron/Trajectron-plus-plus/experiments/nuScenes/devkit/python-sdk"
)


@dataclass
class PredictionConfig:
    backend: str = "hivt"  # hivt | cv | ctrv | covernet | mtp
    device: str = "cpu"
    pred_steps: int = 12
    dt: float = 0.2
    history_seconds: float = 2.0
    history_source: str = "tracking"
    use_hivt_if_available: bool = True  # 兼容旧配置
    fallback_mode: str = "none"  # none | cv | ctrv
    num_modes: int = 6  # 输出给 planning 的最大模态数（covernet/mtp）
    hivt_mode_selection: str = "argmax"  # argmax | velocity_aligned | target_yaw_motion_aligned
    hivt_motion_bias_weight: float = 2.0
    hivt_min_history_speed_mps: float = 0.05
    hivt_lateral_dominance_ratio: float = 1.2
    hivt_lateral_residual_gain: float = 0.0
    hivt_suppress_forward_when_lateral: bool = True

    # CTRV
    ctrv_min_yaw_rate: float = 1.0e-3

    # CoverNet
    covernet_model_path: str = ""
    covernet_lattice_path: str = ""
    covernet_devkit_root: str = ""
    covernet_backbone: str = "resnet50"
    covernet_input_shape: Tuple[int, int, int] = (3, 500, 500)
    covernet_source_dt: float = 0.5

    # MTP
    mtp_model_path: str = ""
    mtp_devkit_root: str = ""
    mtp_backbone: str = "resnet50"
    mtp_num_modes: int = 6
    mtp_seconds: float = 6.0
    mtp_frequency_hz: float = 2.0
    mtp_input_shape: Tuple[int, int, int] = (3, 500, 500)


class HivtPredictor:
    def __init__(
        self,
        config: PredictionConfig,
        *,
        hivt_repo_root: Union[str, Path, None] = None,
        hivt_ckpt_path: Union[str, Path, None] = None,
    ) -> None:
        self.config = config
        self.hivt_repo_root = Path(hivt_repo_root) if hivt_repo_root else None
        self.hivt_ckpt_path = Path(hivt_ckpt_path) if hivt_ckpt_path else None

        self._torch = None
        self._TemporalData = None
        self._hivt_model = None

        self._covernet_model = None
        self._covernet_lattice = None
        self._mtp_model = None

    def predict_track(
        self,
        *,
        track_record: Dict,
        history_xy: List[Tuple[float, float]],
    ) -> List[List[Tuple[float, float]]]:
        backend = str(self.config.backend).strip().lower()
        if backend == "hivt":
            return self._run_with_fallback(
                backend=backend,
                track_record=track_record,
                history_xy=history_xy,
                infer_fn=lambda: [self._predict_with_hivt(track_record=track_record, history_xy=history_xy)],
            )
        if backend == "cv":
            return [self._predict_cv(track_record=track_record, history_xy=history_xy)]
        if backend == "ctrv":
            return [self._predict_ctrv(track_record=track_record, history_xy=history_xy)]
        if backend == "covernet":
            return self._run_with_fallback(
                backend=backend,
                track_record=track_record,
                history_xy=history_xy,
                infer_fn=lambda: self._predict_with_covernet(track_record=track_record, history_xy=history_xy),
            )
        if backend == "mtp":
            return self._run_with_fallback(
                backend=backend,
                track_record=track_record,
                history_xy=history_xy,
                infer_fn=lambda: self._predict_with_mtp(track_record=track_record, history_xy=history_xy),
            )
        raise RuntimeError("prediction.backend 只支持 hivt/cv/ctrv/covernet/mtp")

    def predict_scene(
        self,
        *,
        current_tracks: List[Dict],
        history_xy_by_track: Dict[str, List[Tuple[float, float]]],
        ego_history_xy: List[Tuple[float, float]],
        ego_yaw: float,
        location: str,
        nuscenes_root: Union[str, Path],
    ) -> Dict[str, List[List[Tuple[float, float]]]]:
        backend = str(self.config.backend).strip().lower()
        if backend == "hivt":
            return self._run_with_scene_fallback(
                backend=backend,
                current_tracks=current_tracks,
                history_xy_by_track=history_xy_by_track,
                infer_fn=lambda: self._predict_scene_with_hivt(
                    current_tracks=current_tracks,
                    history_xy_by_track=history_xy_by_track,
                    ego_history_xy=ego_history_xy,
                    ego_yaw=float(ego_yaw),
                    location=location,
                    nuscenes_root=nuscenes_root,
                ),
            )

        predictions: Dict[str, List[List[Tuple[float, float]]]] = {}
        for track in current_tracks:
            track_id = str(track["tracking_id"])
            predictions[track_id] = self.predict_track(
                track_record=track,
                history_xy=history_xy_by_track.get(track_id, []),
            )
        return predictions

    def _run_with_fallback(
        self,
        *,
        backend: str,
        track_record: Dict,
        history_xy: List[Tuple[float, float]],
        infer_fn,
    ) -> List[List[Tuple[float, float]]]:
        try:
            return infer_fn()
        except Exception as exc:
            fallback = str(self.config.fallback_mode).strip().lower()
            if fallback == "cv":
                return [self._predict_cv(track_record=track_record, history_xy=history_xy)]
            if fallback == "ctrv":
                return [self._predict_ctrv(track_record=track_record, history_xy=history_xy)]
            raise RuntimeError(f"{backend} prediction failed: {exc}")

    def _run_with_scene_fallback(
        self,
        *,
        backend: str,
        current_tracks: List[Dict],
        history_xy_by_track: Dict[str, List[Tuple[float, float]]],
        infer_fn,
    ) -> Dict[str, List[List[Tuple[float, float]]]]:
        try:
            return infer_fn()
        except Exception as exc:
            fallback = str(self.config.fallback_mode).strip().lower()
            if fallback not in {"cv", "ctrv"}:
                raise RuntimeError(f"{backend} prediction failed: {exc}")
            predictions: Dict[str, List[List[Tuple[float, float]]]] = {}
            for track in current_tracks:
                track_id = str(track["tracking_id"])
                history_xy = history_xy_by_track.get(track_id, [])
                if fallback == "cv":
                    predictions[track_id] = [self._predict_cv(track_record=track, history_xy=history_xy)]
                else:
                    predictions[track_id] = [self._predict_ctrv(track_record=track, history_xy=history_xy)]
            return predictions

    def _load_hivt(self) -> None:
        if self._hivt_model is not None:
            return
        if self.hivt_repo_root is None or self.hivt_ckpt_path is None:
            raise RuntimeError("backend=hivt 需要 paths.hivt_repo_root 和 paths.hivt_ckpt_path")
        if str(self.hivt_repo_root) not in sys.path:
            sys.path.insert(0, str(self.hivt_repo_root))

        import torch  # noqa: WPS433
        from models.hivt import HiVT  # noqa: WPS433
        from utils import TemporalData  # noqa: WPS433

        map_location = "cpu" if self.config.device == "cpu" else None
        model = HiVT.load_from_checkpoint(
            checkpoint_path=str(self.hivt_ckpt_path),
            parallel=True,
            map_location=map_location,
        )
        model.eval()
        model.to(self.config.device)
        self._torch = torch
        self._TemporalData = TemporalData
        self._hivt_model = model

    def _predict_with_hivt(
        self,
        *,
        track_record: Dict,
        history_xy: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        self._load_hivt()
        torch = self._torch
        TemporalData = self._TemporalData
        model = self._hivt_model

        historical_steps = int(model.hparams.historical_steps)
        future_steps_model = int(model.hparams.future_steps)
        future_steps = min(int(self.config.pred_steps), future_steps_model)

        if not history_xy:
            history_xy = [
                (float(track_record["translation"][0]), float(track_record["translation"][1]))
            ]
        history_xy = history_xy[-historical_steps:]
        valid_len = len(history_xy)
        hist_start = historical_steps - valid_len

        positions = torch.zeros((1, historical_steps + future_steps_model, 2), dtype=torch.float32)
        padding_mask = torch.ones((1, historical_steps + future_steps_model), dtype=torch.bool)
        x = torch.zeros((1, historical_steps, 2), dtype=torch.float32)
        bos_mask = torch.zeros((1, historical_steps), dtype=torch.bool)

        for i, (hx, hy) in enumerate(history_xy):
            t = hist_start + i
            positions[0, t, 0] = float(hx)
            positions[0, t, 1] = float(hy)
            padding_mask[0, t] = False
        for t in range(historical_steps, historical_steps + future_steps_model):
            positions[0, t] = positions[0, historical_steps - 1]

        for t in range(1, historical_steps):
            if not bool(padding_mask[0, t]) and not bool(padding_mask[0, t - 1]):
                x[0, t] = positions[0, t] - positions[0, t - 1]
        bos_mask[:, 0] = ~padding_mask[:, 0]
        bos_mask[:, 1:historical_steps] = (
            padding_mask[:, : historical_steps - 1] & ~padding_mask[:, 1:historical_steps]
        )

        rotate_angle = infer_heading_from_history(history_xy)
        rotate_angles = torch.tensor([rotate_angle], dtype=torch.float32)

        edge_index = torch.zeros((2, 0), dtype=torch.long)
        lane_vectors = torch.zeros((0, 2), dtype=torch.float32)
        is_intersections = torch.zeros((0,), dtype=torch.uint8)
        turn_directions = torch.zeros((0,), dtype=torch.uint8)
        traffic_controls = torch.zeros((0,), dtype=torch.uint8)
        lane_actor_index = torch.zeros((2, 0), dtype=torch.long)
        lane_actor_vectors = torch.zeros((0, 2), dtype=torch.float32)

        data = TemporalData(
            x=x,
            positions=positions,
            edge_index=edge_index,
            y=None,
            num_nodes=1,
            padding_mask=padding_mask,
            bos_mask=bos_mask,
            rotate_angles=rotate_angles,
            lane_vectors=lane_vectors,
            is_intersections=is_intersections,
            turn_directions=turn_directions,
            traffic_controls=traffic_controls,
            lane_actor_index=lane_actor_index,
            lane_actor_vectors=lane_actor_vectors,
            seq_id=0,
        ).to(self.config.device)

        with torch.no_grad():
            y_hat, pi = model(data)
        mode_idx = int(torch.argmax(pi[0]).item())
        rel_actor = y_hat[mode_idx, 0, :future_steps, :2]

        c = float(np.cos(rotate_angle))
        s = float(np.sin(rotate_angle))
        rotate_mat = torch.tensor([[c, -s], [s, c]], dtype=torch.float32, device=rel_actor.device)
        rel_scene = rel_actor @ rotate_mat.transpose(0, 1)
        current_xy = positions[0, historical_steps - 1].to(rel_scene.device)
        pred_global = rel_scene + current_xy

        return [(float(p[0].item()), float(p[1].item())) for p in pred_global]

    def _predict_scene_with_hivt(
        self,
        *,
        current_tracks: List[Dict],
        history_xy_by_track: Dict[str, List[Tuple[float, float]]],
        ego_history_xy: List[Tuple[float, float]],
        ego_yaw: float,
        location: str,
        nuscenes_root: Union[str, Path],
    ) -> Dict[str, List[List[Tuple[float, float]]]]:
        if not current_tracks:
            return {}

        self._load_hivt()
        torch = self._torch
        TemporalData = self._TemporalData
        model = self._hivt_model

        historical_steps = int(model.hparams.historical_steps)
        future_steps_model = int(model.hparams.future_steps)
        future_steps = min(int(self.config.pred_steps), future_steps_model)

        if ego_history_xy:
            av_origin_xy = ego_history_xy[-1]
        else:
            av_origin_xy = (
                float(current_tracks[0]["translation"][0]),
                float(current_tracks[0]["translation"][1]),
            )

        av_origin = torch.tensor(av_origin_xy, dtype=torch.float32)
        av_rotate_mat = torch.tensor(
            [
                [float(np.cos(ego_yaw)), -float(np.sin(ego_yaw))],
                [float(np.sin(ego_yaw)), float(np.cos(ego_yaw))],
            ],
            dtype=torch.float32,
        )

        node_items: List[Dict[str, object]] = []
        if ego_history_xy:
            node_items.append(
                {
                    "track_id": "__ego__",
                    "history_xy": ego_history_xy[-historical_steps:],
                    "is_ego": True,
                }
            )
        for track in current_tracks:
            track_id = str(track["tracking_id"])
            history_xy = history_xy_by_track.get(track_id, [])
            if not history_xy:
                history_xy = [
                    (
                        float(track["translation"][0]),
                        float(track["translation"][1]),
                    )
                ]
            node_items.append(
                {
                    "track_id": track_id,
                    "history_xy": history_xy[-historical_steps:],
                    "track_record": track,
                    "is_ego": False,
                }
            )

        num_nodes = len(node_items)
        positions = torch.zeros((num_nodes, historical_steps + future_steps_model, 2), dtype=torch.float32)
        padding_mask = torch.ones((num_nodes, historical_steps + future_steps_model), dtype=torch.bool)
        x = torch.zeros((num_nodes, historical_steps, 2), dtype=torch.float32)
        bos_mask = torch.zeros((num_nodes, historical_steps), dtype=torch.bool)
        rotate_angles = torch.zeros((num_nodes,), dtype=torch.float32)
        current_positions_local = torch.zeros((num_nodes, 2), dtype=torch.float32)

        for node_idx, item in enumerate(node_items):
            history_xy = item["history_xy"]
            valid_len = len(history_xy)
            hist_start = historical_steps - valid_len
            local_history = []
            for i, (hx, hy) in enumerate(history_xy):
                local_xy = world_to_av_frame(
                    (float(hx), float(hy)),
                    origin=av_origin,
                    rotate_mat=av_rotate_mat,
                )
                local_history.append(local_xy)
                t = hist_start + i
                positions[node_idx, t] = local_xy
                padding_mask[node_idx, t] = False
            current_positions_local[node_idx] = positions[node_idx, historical_steps - 1]
            for t in range(historical_steps, historical_steps + future_steps_model):
                positions[node_idx, t] = current_positions_local[node_idx]
            for t in range(1, historical_steps):
                if not bool(padding_mask[node_idx, t]) and not bool(padding_mask[node_idx, t - 1]):
                    x[node_idx, t] = positions[node_idx, t] - positions[node_idx, t - 1]
            bos_mask[node_idx, 0] = ~padding_mask[node_idx, 0]
            bos_mask[node_idx, 1:historical_steps] = (
                padding_mask[node_idx, : historical_steps - 1]
                & ~padding_mask[node_idx, 1:historical_steps]
            )
            rotate_angles[node_idx] = infer_heading_from_local_history(local_history)

        if num_nodes > 1:
            edge_index = torch.tensor(list(permutations(range(num_nodes), 2)), dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        (
            lane_vectors,
            is_intersections,
            turn_directions,
            traffic_controls,
            lane_actor_index,
            lane_actor_vectors,
        ) = self._build_lane_features(
            location=location,
            nuscenes_root=nuscenes_root,
            av_origin=av_origin,
            av_rotate_mat=av_rotate_mat,
            current_positions_local=current_positions_local,
            node_items=node_items,
            radius_m=float(model.hparams.local_radius),
            torch_module=torch,
        )

        data = TemporalData(
            x=x,
            positions=positions,
            edge_index=edge_index,
            y=None,
            num_nodes=num_nodes,
            padding_mask=padding_mask,
            bos_mask=bos_mask,
            rotate_angles=rotate_angles,
            lane_vectors=lane_vectors,
            is_intersections=is_intersections,
            turn_directions=turn_directions,
            traffic_controls=traffic_controls,
            lane_actor_index=lane_actor_index,
            lane_actor_vectors=lane_actor_vectors,
            seq_id=0,
        ).to(self.config.device)

        with torch.no_grad():
            y_hat, pi = model(data)

        predictions: Dict[str, List[List[Tuple[float, float]]]] = {}
        for node_idx, item in enumerate(node_items):
            if bool(item["is_ego"]):
                continue
            actor_rotate_mat = torch.tensor(
                [
                    [
                        float(np.cos(rotate_angles[node_idx].item())),
                        -float(np.sin(rotate_angles[node_idx].item())),
                    ],
                    [
                        float(np.sin(rotate_angles[node_idx].item())),
                        float(np.cos(rotate_angles[node_idx].item())),
                    ],
                ],
                dtype=torch.float32,
                device=y_hat.device,
            )
            all_modes_world = []
            for candidate_idx in range(int(y_hat.size(0))):
                rel_actor = y_hat[candidate_idx, node_idx, :future_steps, :2]
                pred_av = rel_actor @ actor_rotate_mat.transpose(0, 1)
                pred_av = pred_av + current_positions_local[node_idx].to(rel_actor.device)
                pred_world = pred_av @ av_rotate_mat.to(rel_actor.device).transpose(0, 1)
                pred_world = pred_world + av_origin.to(rel_actor.device)
                all_modes_world.append(pred_world)

            mode_idx = self._select_hivt_mode(
                pi_row=pi[node_idx],
                pred_world_modes=all_modes_world,
                current_world_xy=tuple(item["history_xy"][-1]),
                history_xy=item["history_xy"],
                track_record=item.get("track_record"),
            )
            pred_world = all_modes_world[mode_idx]
            pred_world = self._apply_hivt_motion_residual(
                pred_world=pred_world,
                current_world_xy=tuple(item["history_xy"][-1]),
                history_xy=item["history_xy"],
                track_record=item.get("track_record"),
            )
            predictions[str(item["track_id"])] = [[
                (float(point[0].item()), float(point[1].item()))
                for point in pred_world
            ]]

        return predictions


    def _select_hivt_mode(
        self,
        *,
        pi_row,
        pred_world_modes,
        current_world_xy: Tuple[float, float],
        history_xy: List[Tuple[float, float]],
        track_record: Optional[Dict] = None,
    ) -> int:
        base_idx = int(self._torch.argmax(pi_row).item())
        selection = str(self.config.hivt_mode_selection).strip().lower()
        if selection not in {"velocity_aligned", "target_yaw_motion_aligned"}:
            return base_idx

        vx, vy = infer_velocity_from_history(history_xy, dt=float(self.config.dt))
        speed = float(math.hypot(vx, vy))
        if speed < float(self.config.hivt_min_history_speed_mps):
            return base_idx

        if selection == "target_yaw_motion_aligned" and track_record is not None:
            target_yaw = track_yaw_from_record(track_record)
        else:
            target_yaw = None

        if target_yaw is not None:
            forward_axis = np.asarray([math.cos(target_yaw), math.sin(target_yaw)], dtype=np.float32)
            left_axis = np.asarray([-math.sin(target_yaw), math.cos(target_yaw)], dtype=np.float32)
            velocity = np.asarray([vx, vy], dtype=np.float32)
            fwd_speed = float(np.dot(velocity, forward_axis))
            left_speed = float(np.dot(velocity, left_axis))
            if abs(left_speed) >= abs(fwd_speed) * float(self.config.hivt_lateral_dominance_ratio):
                desired_axis = left_axis * (1.0 if left_speed >= 0.0 else -1.0)
            elif abs(fwd_speed) > 1.0e-6:
                desired_axis = forward_axis * (1.0 if fwd_speed >= 0.0 else -1.0)
            else:
                desired_axis = velocity / speed
        else:
            desired_axis = np.asarray([vx / speed, vy / speed], dtype=np.float32)

        current = np.asarray(current_world_xy, dtype=np.float32)
        pi_values = pi_row.detach().cpu().numpy().astype(np.float32)
        base_score = float(pi_values[base_idx])
        best_idx = base_idx
        best_score = base_score
        weight = float(self.config.hivt_motion_bias_weight)

        for idx, pred_world in enumerate(pred_world_modes):
            final_xy = pred_world[-1].detach().cpu().numpy().astype(np.float32)
            displacement = final_xy - current
            aligned_progress = max(0.0, float(np.dot(displacement, desired_axis)))
            score = float(pi_values[idx]) + weight * aligned_progress
            if score > best_score:
                best_score = score
                best_idx = int(idx)
        return best_idx


    def _apply_hivt_motion_residual(
        self,
        *,
        pred_world,
        current_world_xy: Tuple[float, float],
        history_xy: List[Tuple[float, float]],
        track_record: Optional[Dict] = None,
    ):
        gain = float(self.config.hivt_lateral_residual_gain)
        if gain <= 0.0 or track_record is None:
            return pred_world
        target_yaw = track_yaw_from_record(track_record)
        if target_yaw is None:
            return pred_world

        vx, vy = infer_velocity_from_history(history_xy, dt=float(self.config.dt))
        speed = float(math.hypot(vx, vy))
        if speed < float(self.config.hivt_min_history_speed_mps):
            return pred_world

        forward_axis_np = np.asarray([math.cos(target_yaw), math.sin(target_yaw)], dtype=np.float32)
        left_axis_np = np.asarray([-math.sin(target_yaw), math.cos(target_yaw)], dtype=np.float32)
        velocity = np.asarray([vx, vy], dtype=np.float32)
        fwd_speed = float(np.dot(velocity, forward_axis_np))
        left_speed = float(np.dot(velocity, left_axis_np))
        if abs(left_speed) < abs(fwd_speed) * float(self.config.hivt_lateral_dominance_ratio):
            return pred_world

        device = pred_world.device
        dtype = pred_world.dtype
        current = pred_world.new_tensor([float(current_world_xy[0]), float(current_world_xy[1])])
        left_axis = pred_world.new_tensor(left_axis_np)
        forward_axis = pred_world.new_tensor(forward_axis_np)
        steps = pred_world.size(0)
        t = self._torch.arange(1, steps + 1, dtype=dtype, device=device).unsqueeze(1) * float(self.config.dt)
        lateral = left_axis.unsqueeze(0) * (float(left_speed) * gain) * t
        if bool(self.config.hivt_suppress_forward_when_lateral):
            return current.unsqueeze(0) + lateral

        displacement = pred_world - current.unsqueeze(0)
        model_forward = (displacement * forward_axis.unsqueeze(0)).sum(dim=1, keepdim=True) * forward_axis.unsqueeze(0)
        return current.unsqueeze(0) + model_forward + lateral

    def _build_lane_features(
        self,
        *,
        location: str,
        nuscenes_root: Union[str, Path],
        av_origin,
        av_rotate_mat,
        current_positions_local,
        node_items: List[Dict[str, object]],
        radius_m: float,
        torch_module,
    ):
        map_api = SimpleNuScenesMap(nuscenes_root, location)
        unique_centerlines = []
        seen_keys = set()

        for item in node_items:
            history_xy = item["history_xy"]
            if not history_xy:
                continue
            current_xy = history_xy[-1]
            centerlines = map_api.get_centerlines_in_radius(
                float(current_xy[0]),
                float(current_xy[1]),
                radius_m=radius_m,
                resolution_meters=1.0,
            )
            for centerline in centerlines:
                if len(centerline) < 2:
                    continue
                centerline_xy = np.asarray(centerline[:, :2], dtype=np.float32)
                key = (
                    len(centerline_xy),
                    tuple(np.round(centerline_xy[0], 2)),
                    tuple(np.round(centerline_xy[-1], 2)),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                unique_centerlines.append(centerline_xy)

        if not unique_centerlines:
            return (
                torch_module.zeros((0, 2), dtype=torch.float32),
                torch_module.zeros((0,), dtype=torch.uint8),
                torch_module.zeros((0,), dtype=torch.uint8),
                torch_module.zeros((0,), dtype=torch.uint8),
                torch_module.zeros((2, 0), dtype=torch.long),
                torch_module.zeros((0, 2), dtype=torch.float32),
            )

        lane_positions_list = []
        lane_vectors_list = []
        is_intersections_list = []
        turn_directions_list = []
        traffic_controls_list = []

        for centerline_xy in unique_centerlines:
            centerline = torch_module.from_numpy(centerline_xy)
            centerline_local = torch_module.matmul(centerline - av_origin, av_rotate_mat)
            lane_positions_list.append(centerline_local[:-1])
            lane_vectors_list.append(centerline_local[1:] - centerline_local[:-1])
            count = max(0, centerline_local.size(0) - 1)
            is_intersections_list.append(torch_module.zeros((count,), dtype=torch_module.uint8))
            turn_directions_list.append(torch_module.zeros((count,), dtype=torch_module.uint8))
            traffic_controls_list.append(torch_module.zeros((count,), dtype=torch_module.uint8))

        lane_positions = torch_module.cat(lane_positions_list, dim=0)
        lane_vectors = torch_module.cat(lane_vectors_list, dim=0)
        is_intersections = torch_module.cat(is_intersections_list, dim=0)
        turn_directions = torch_module.cat(turn_directions_list, dim=0)
        traffic_controls = torch_module.cat(traffic_controls_list, dim=0)

        num_nodes = int(current_positions_local.size(0))
        lane_actor_index = torch_module.cartesian_prod(
            torch_module.arange(lane_vectors.size(0), dtype=torch_module.long),
            torch_module.arange(num_nodes, dtype=torch_module.long),
        ).t().contiguous()
        lane_actor_vectors = (
            lane_positions.repeat_interleave(num_nodes, dim=0)
            - current_positions_local.repeat(lane_vectors.size(0), 1)
        )
        mask = torch_module.norm(lane_actor_vectors, p=2, dim=-1) < float(radius_m)
        lane_actor_index = lane_actor_index[:, mask]
        lane_actor_vectors = lane_actor_vectors[mask]
        return (
            lane_vectors,
            is_intersections,
            turn_directions,
            traffic_controls,
            lane_actor_index,
            lane_actor_vectors,
        )

    def _predict_cv(
        self,
        *,
        track_record: Dict,
        history_xy: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        vx, vy = infer_velocity_from_history(history_xy, dt=float(self.config.dt))
        start_x = float(track_record["translation"][0])
        start_y = float(track_record["translation"][1])
        trajectory = []
        for step in range(1, int(self.config.pred_steps) + 1):
            t = step * float(self.config.dt)
            trajectory.append((start_x + vx * t, start_y + vy * t))
        return trajectory

    def _predict_ctrv(
        self,
        *,
        track_record: Dict,
        history_xy: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        start_x = float(track_record["translation"][0])
        start_y = float(track_record["translation"][1])
        speed = infer_speed_from_history(history_xy, dt=float(self.config.dt))
        heading = infer_heading_from_history(history_xy)
        track_yaw = track_yaw_from_record(track_record)
        if track_yaw is not None:
            heading = float(track_yaw)

        yaw_rate = infer_yaw_rate_from_history(history_xy, dt=float(self.config.dt))
        min_abs_yaw_rate = float(max(1.0e-6, self.config.ctrv_min_yaw_rate))

        x = start_x
        y = start_y
        theta = heading
        dt = float(self.config.dt)
        trajectory: List[Tuple[float, float]] = []
        for _ in range(int(self.config.pred_steps)):
            if abs(yaw_rate) < min_abs_yaw_rate:
                x += speed * dt * math.cos(theta)
                y += speed * dt * math.sin(theta)
            else:
                next_theta = theta + yaw_rate * dt
                x += (speed / yaw_rate) * (math.sin(next_theta) - math.sin(theta))
                y += (speed / yaw_rate) * (-math.cos(next_theta) + math.cos(theta))
                theta = next_theta
            trajectory.append((float(x), float(y)))
        return trajectory

    def _predict_with_covernet(
        self,
        *,
        track_record: Dict,
        history_xy: List[Tuple[float, float]],
    ) -> List[List[Tuple[float, float]]]:
        model, lattice, torch_module = self._load_covernet_model()

        speed, accel, yaw_rate, heading = infer_kinematics_from_history(
            history_xy=history_xy,
            dt=float(self.config.dt),
            track_record=track_record,
        )
        image_tensor = torch_module.zeros((1, *tuple(self.config.covernet_input_shape)), dtype=torch_module.float32)
        asv = torch_module.tensor([[speed, accel, yaw_rate]], dtype=torch_module.float32)

        model_device = next(model.parameters()).device
        image_tensor = image_tensor.to(model_device)
        asv = asv.to(model_device)

        with torch_module.no_grad():
            logits = model(image_tensor, asv)[0]
            probs = torch_module.softmax(logits, dim=-1)

        top_k = max(1, min(int(self.config.num_modes), int(lattice.shape[0])))
        top_indices = torch_module.topk(probs, k=top_k).indices.detach().cpu().numpy().tolist()

        origin = (float(track_record["translation"][0]), float(track_record["translation"][1]))
        src_dt = float(self.config.covernet_source_dt) if float(self.config.covernet_source_dt) > 0.0 else float(self.config.dt)

        outputs: List[List[Tuple[float, float]]] = []
        for mode_idx in top_indices:
            traj_local = lattice[int(mode_idx)]
            traj_world = agent_local_traj_to_world(
                local_xy=traj_local,
                origin_xy=origin,
                heading_yaw=heading,
            )
            outputs.append(
                resample_trajectory(
                    points_xy=traj_world,
                    src_dt=src_dt,
                    dst_dt=float(self.config.dt),
                    target_steps=int(self.config.pred_steps),
                )
            )
        return outputs

    def _predict_with_mtp(
        self,
        *,
        track_record: Dict,
        history_xy: List[Tuple[float, float]],
    ) -> List[List[Tuple[float, float]]]:
        model, torch_module = self._load_mtp_model()

        speed, accel, yaw_rate, heading = infer_kinematics_from_history(
            history_xy=history_xy,
            dt=float(self.config.dt),
            track_record=track_record,
        )
        image_tensor = torch_module.zeros((1, *tuple(self.config.mtp_input_shape)), dtype=torch_module.float32)
        asv = torch_module.tensor([[speed, accel, yaw_rate]], dtype=torch_module.float32)

        model_device = next(model.parameters()).device
        image_tensor = image_tensor.to(model_device)
        asv = asv.to(model_device)

        with torch_module.no_grad():
            raw = model(image_tensor, asv)[0]

        num_modes = int(model.num_modes)
        if num_modes <= 0:
            raise RuntimeError("MTP model.num_modes is invalid")
        traj_flat = raw[:-num_modes]
        probs = raw[-num_modes:]
        trajectories = traj_flat.reshape(num_modes, -1, 2).detach().cpu().numpy()
        mode_prob = probs.detach().cpu().numpy()

        top_k = max(1, min(int(self.config.num_modes), num_modes))
        top_indices = np.argsort(-mode_prob)[:top_k].tolist()

        origin = (float(track_record["translation"][0]), float(track_record["translation"][1]))
        freq_hz = float(self.config.mtp_frequency_hz)
        if freq_hz <= 0.0:
            raise RuntimeError("prediction.mtp_frequency_hz must be > 0")
        src_dt = 1.0 / freq_hz

        outputs: List[List[Tuple[float, float]]] = []
        for mode_idx in top_indices:
            traj_world = agent_local_traj_to_world(
                local_xy=trajectories[int(mode_idx)],
                origin_xy=origin,
                heading_yaw=heading,
            )
            outputs.append(
                resample_trajectory(
                    points_xy=traj_world,
                    src_dt=src_dt,
                    dst_dt=float(self.config.dt),
                    target_steps=int(self.config.pred_steps),
                )
            )
        return outputs

    def _load_covernet_model(self):
        if self._covernet_model is not None and self._covernet_lattice is not None:
            return self._covernet_model, self._covernet_lattice, self._torch

        self._ensure_torch_loaded()
        torch_module = self._torch
        self._ensure_nuscenes_prediction_devkit(
            str(self.config.covernet_devkit_root).strip() or str(DEFAULT_NUSCENES_DEVKIT_ROOT)
        )

        covernet_model_path = Path(str(self.config.covernet_model_path).strip()) if str(self.config.covernet_model_path).strip() else None
        covernet_lattice_path = Path(str(self.config.covernet_lattice_path).strip()) if str(self.config.covernet_lattice_path).strip() else None
        if covernet_model_path is None or not covernet_model_path.is_file():
            raise FileNotFoundError("backend=covernet 需要有效的 prediction.covernet_model_path")
        if covernet_lattice_path is None or not covernet_lattice_path.is_file():
            raise FileNotFoundError("backend=covernet 需要有效的 prediction.covernet_lattice_path")

        from nuscenes.prediction.models.backbone import ResNetBackbone  # noqa: WPS433
        from nuscenes.prediction.models.covernet import CoverNet  # noqa: WPS433

        lattice = load_covernet_lattice(covernet_lattice_path)
        num_modes = int(lattice.shape[0])
        model = CoverNet(
            ResNetBackbone(str(self.config.covernet_backbone)),
            num_modes=num_modes,
            input_shape=tuple(self.config.covernet_input_shape),
        )
        state_dict = extract_state_dict(torch_module.load(str(covernet_model_path), map_location="cpu"))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(str(self.config.device))

        self._covernet_model = model
        self._covernet_lattice = lattice
        return model, lattice, torch_module

    def _load_mtp_model(self):
        if self._mtp_model is not None:
            return self._mtp_model, self._torch

        self._ensure_torch_loaded()
        torch_module = self._torch
        self._ensure_nuscenes_prediction_devkit(
            str(self.config.mtp_devkit_root).strip() or str(DEFAULT_NUSCENES_DEVKIT_ROOT)
        )

        mtp_model_path = Path(str(self.config.mtp_model_path).strip()) if str(self.config.mtp_model_path).strip() else None
        if mtp_model_path is None or not mtp_model_path.is_file():
            raise FileNotFoundError("backend=mtp 需要有效的 prediction.mtp_model_path")

        from nuscenes.prediction.models.backbone import ResNetBackbone  # noqa: WPS433
        from nuscenes.prediction.models.mtp import MTP  # noqa: WPS433

        model = MTP(
            ResNetBackbone(str(self.config.mtp_backbone)),
            num_modes=int(self.config.mtp_num_modes),
            seconds=float(self.config.mtp_seconds),
            frequency_in_hz=float(self.config.mtp_frequency_hz),
            input_shape=tuple(self.config.mtp_input_shape),
        )
        state_dict = extract_state_dict(torch_module.load(str(mtp_model_path), map_location="cpu"))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(str(self.config.device))

        self._mtp_model = model
        return model, torch_module

    def _ensure_torch_loaded(self) -> None:
        if self._torch is not None:
            return
        import torch  # noqa: WPS433

        self._torch = torch

    def _ensure_nuscenes_prediction_devkit(self, devkit_root: str) -> None:
        try:
            import nuscenes.prediction.models.backbone  # noqa: F401,WPS433
            return
        except Exception:
            pass

        root = Path(devkit_root)
        if not root.is_dir():
            raise FileNotFoundError(
                "找不到 nuScenes prediction devkit 路径。"
                f"请设置 prediction.*_devkit_root，当前: {root}"
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        try:
            import nuscenes.prediction.models.backbone  # noqa: F401,WPS433
        except Exception as exc:
            raise RuntimeError(
                "无法导入 nuscenes prediction 模块。"
                "请检查当前环境是否安装依赖（nuscenes-devkit / opencv-python / torchvision）。"
            ) from exc


def extract_state_dict(payload) -> Dict[str, object]:
    if isinstance(payload, dict):
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            state_dict = payload["state_dict"]
        elif "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
            state_dict = payload["model_state_dict"]
        else:
            state_dict = payload
        if isinstance(state_dict, dict):
            cleaned = {}
            for key, value in state_dict.items():
                clean_key = str(key)
                if clean_key.startswith("module."):
                    clean_key = clean_key[len("module.") :]
                cleaned[clean_key] = value
            return cleaned
    raise RuntimeError("模型权重文件格式无法识别，期望是 state_dict 或其封装字典")


def load_covernet_lattice(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        lattice = np.load(path)
    elif suffix == ".npz":
        payload = np.load(path)
        if not payload.files:
            raise RuntimeError(f"covernet lattice npz is empty: {path}")
        lattice = payload[payload.files[0]]
    elif suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            lattice = pickle.load(handle)
    else:
        lattice = np.load(path, allow_pickle=True)

    lattice_arr = np.asarray(lattice, dtype=np.float32)
    if lattice_arr.ndim != 3 or lattice_arr.shape[-1] != 2:
        raise RuntimeError(
            f"covernet lattice shape must be [num_modes, steps, 2], got {lattice_arr.shape}"
        )
    return lattice_arr


def track_yaw_from_record(track_record: Dict) -> Optional[float]:
    rotation = track_record.get("rotation")
    if not isinstance(rotation, (list, tuple)) or len(rotation) != 4:
        return None
    return yaw_from_quaternion(rotation)


def yaw_from_quaternion(quaternion_wxyz: Union[List[float], Tuple[float, ...]]) -> float:
    w, x, y, z = [float(v) for v in quaternion_wxyz]
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def wrap_angle_pi(angle_rad: float) -> float:
    return float((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)


def infer_heading_from_history(history_xy: List[Tuple[float, float]]) -> float:
    dx, dy, _ = infer_motion_from_history(history_xy)
    return float(np.arctan2(dy, dx + 1.0e-6))


def infer_heading_from_local_history(history_xy: List[np.ndarray]) -> float:
    if len(history_xy) < 2:
        return 0.0
    latest_xy = history_xy[-1]
    for previous_xy in reversed(history_xy[:-1]):
        delta = latest_xy - previous_xy
        dx = float(delta[0])
        dy = float(delta[1])
        if abs(dx) > 1.0e-6 or abs(dy) > 1.0e-6:
            return float(np.arctan2(dy, dx + 1.0e-6))
    return 0.0


def infer_motion_from_history(
    history_xy: List[Tuple[float, float]],
) -> Tuple[float, float, int]:
    if len(history_xy) < 2:
        return 0.0, 0.0, 0

    latest_x, latest_y = history_xy[-1]
    for reverse_index, (prev_x, prev_y) in enumerate(reversed(history_xy[:-1]), start=1):
        dx = float(latest_x - prev_x)
        dy = float(latest_y - prev_y)
        if abs(dx) > 1.0e-6 or abs(dy) > 1.0e-6:
            return dx, dy, reverse_index
    return 0.0, 0.0, 0


def infer_velocity_from_history(
    history_xy: List[Tuple[float, float]],
    *,
    dt: float,
) -> Tuple[float, float]:
    dx, dy, frame_gap = infer_motion_from_history(history_xy)
    if frame_gap <= 0 or dt <= 0:
        return 0.0, 0.0
    total_dt = float(frame_gap) * float(dt)
    return float(dx / total_dt), float(dy / total_dt)


def infer_speed_from_history(
    history_xy: List[Tuple[float, float]],
    *,
    dt: float,
) -> float:
    vx, vy = infer_velocity_from_history(history_xy, dt=dt)
    return float(math.hypot(vx, vy))


def infer_yaw_rate_from_history(
    history_xy: List[Tuple[float, float]],
    *,
    dt: float,
) -> float:
    if dt <= 0.0 or len(history_xy) < 3:
        return 0.0

    headings: List[float] = []
    for idx in range(1, len(history_xy)):
        p0 = history_xy[idx - 1]
        p1 = history_xy[idx]
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        if abs(dx) <= 1.0e-6 and abs(dy) <= 1.0e-6:
            continue
        headings.append(float(math.atan2(dy, dx)))
    if len(headings) < 2:
        return 0.0

    delta = wrap_angle_pi(headings[-1] - headings[-2])
    return float(delta / dt)


def infer_kinematics_from_history(
    *,
    history_xy: List[Tuple[float, float]],
    dt: float,
    track_record: Dict,
) -> Tuple[float, float, float, float]:
    speed_now = infer_speed_from_history(history_xy, dt=dt)
    yaw_rate = infer_yaw_rate_from_history(history_xy, dt=dt)
    heading = infer_heading_from_history(history_xy)
    track_yaw = track_yaw_from_record(track_record)
    if track_yaw is not None:
        heading = float(track_yaw)

    accel = 0.0
    if len(history_xy) >= 3 and dt > 0.0:
        speed_prev = infer_speed_from_history(history_xy[:-1], dt=dt)
        accel = float((speed_now - speed_prev) / dt)
    return float(speed_now), float(accel), float(yaw_rate), float(heading)


def agent_local_to_world(
    local_xy: Tuple[float, float],
    *,
    origin_xy: Tuple[float, float],
    heading_yaw: float,
) -> Tuple[float, float]:
    # nuScenes prediction local frame: x is agent-right, y is agent-forward.
    right = float(local_xy[0])
    forward = float(local_xy[1])
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    dx = math.sin(heading_yaw) * right + math.cos(heading_yaw) * forward
    dy = -math.cos(heading_yaw) * right + math.sin(heading_yaw) * forward
    return float(ox + dx), float(oy + dy)


def agent_local_traj_to_world(
    *,
    local_xy: Union[np.ndarray, List[Tuple[float, float]]],
    origin_xy: Tuple[float, float],
    heading_yaw: float,
) -> List[Tuple[float, float]]:
    arr = np.asarray(local_xy, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise RuntimeError(f"local trajectory must be [N,2], got {arr.shape}")
    return [
        agent_local_to_world((float(point[0]), float(point[1])), origin_xy=origin_xy, heading_yaw=heading_yaw)
        for point in arr
    ]


def resample_trajectory(
    *,
    points_xy: List[Tuple[float, float]],
    src_dt: float,
    dst_dt: float,
    target_steps: int,
) -> List[Tuple[float, float]]:
    if target_steps <= 0:
        return []
    if len(points_xy) <= 1:
        if not points_xy:
            return []
        return [tuple(points_xy[0]) for _ in range(target_steps)]

    src_dt = float(src_dt)
    dst_dt = float(dst_dt)
    if src_dt <= 0.0 or dst_dt <= 0.0:
        return points_xy[:target_steps]

    source = np.asarray(points_xy, dtype=np.float32)
    t_src = np.arange(len(source), dtype=np.float32) * src_dt
    t_dst = np.arange(target_steps, dtype=np.float32) * dst_dt + dst_dt
    t_end = float(t_src[-1])
    t_dst = np.clip(t_dst, 0.0, t_end)
    x_dst = np.interp(t_dst, t_src, source[:, 0])
    y_dst = np.interp(t_dst, t_src, source[:, 1])
    return [(float(x), float(y)) for x, y in np.column_stack([x_dst, y_dst])]


def world_to_av_frame(world_xy: Tuple[float, float], *, origin, rotate_mat):
    point = origin.new_tensor([float(world_xy[0]), float(world_xy[1])])
    return (point - origin) @ rotate_mat


def select_history_xy(
    history_records: List[Dict],
    *,
    history_seconds: float,
    dt: float,
) -> List[Tuple[float, float]]:
    max_frames = max(2, int(np.ceil(history_seconds / dt)))
    selected = history_records[-max_frames:]
    return [
        (float(item["translation"][0]), float(item["translation"][1]))
        for item in selected
    ]
