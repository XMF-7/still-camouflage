from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
import yaml
from match_target_car import _as_path, _load_yaml, _save_yaml, _to_builtin, quaternion_to_rotation_matrix, run_target_matching, sensor_from_global_matrix
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / 'config.yaml'
DEFAULT_NEAR_PLANE_M = 0.1
CAMERA_CHANNELS = ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT')

def _normalize_prompt_points(raw_value: Any) -> List[List[float]]:
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) == 0:
        return []
    first = raw_value[0]
    if isinstance(first, (list, tuple)):
        points: List[List[float]] = []
        for item in raw_value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                points.append([float(item[0]), float(item[1])])
        return points
    if len(raw_value) == 2:
        return [[float(raw_value[0]), float(raw_value[1])]]
    return []

def _normalize_channel_prompt_spec(raw_value: Any) -> Dict[str, List[List[float]]]:
    if isinstance(raw_value, dict):
        return {'positive': _normalize_prompt_points(raw_value.get('positive', [])), 'negative': _normalize_prompt_points(raw_value.get('negative', []))}
    return {'positive': _normalize_prompt_points(raw_value), 'negative': []}

def _resolve_explicit_prompt_spec_map(frame: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, List[List[float]]]], bool]:
    prompt_specs = frame.get('target_prompt_specs_by_channel', None)
    if isinstance(prompt_specs, dict):
        return ({str(channel): _normalize_channel_prompt_spec(value) for channel, value in prompt_specs.items()}, True)
    prompt_points = frame.get('target_prompt_points_by_channel', None)
    if isinstance(prompt_points, dict):
        return ({str(channel): {'positive': _normalize_prompt_points(value), 'negative': []} for channel, value in prompt_points.items()}, True)
    target = frame.get('target', {})
    if isinstance(target, dict):
        prompts_raw = target.get('prompts', None)
        if isinstance(prompts_raw, dict):
            return ({str(channel): _normalize_channel_prompt_spec(value) for channel, value in prompts_raw.items()}, True)
    return ({}, False)

def _camouflage_body_offset_xyz_from_config(config_or_camouflage_cfg: Dict[str, Any]) -> np.ndarray:
    cfg = dict(config_or_camouflage_cfg or {})
    if isinstance(cfg.get('mesh'), dict):
        cfg = dict(cfg.get('mesh', {}))
    if isinstance(cfg.get('camouflage_adjust'), dict):
        cfg = dict(cfg.get('camouflage_adjust', {}))
    if isinstance(cfg.get('camouflage'), dict):
        cfg = dict(cfg.get('camouflage', {}))
    front_m = float(cfg.get('offset_front_m', 0.0))
    left_m = float(cfg.get('offset_left_m', 0.0))
    up_m = float(cfg.get('offset_up_m', 0.0))
    return np.asarray([front_m, left_m, up_m], dtype=np.float32)

def _camouflage_mesh_global_scale_from_config(config_or_camouflage_cfg: Dict[str, Any]) -> float:
    cfg = dict(config_or_camouflage_cfg or {})
    if isinstance(cfg.get('mesh'), dict):
        cfg = dict(cfg.get('mesh', {}))
    if isinstance(cfg.get('camouflage_adjust'), dict):
        cfg = dict(cfg.get('camouflage_adjust', {}))
    if isinstance(cfg.get('camouflage'), dict):
        cfg = dict(cfg.get('camouflage', {}))
    scale = float(cfg.get('mesh_global_scale', 1.0))
    if scale <= 0.0:
        raise ValueError('camouflage.mesh_global_scale must be > 0')
    return scale

def _camouflage_mesh_scale_xyz_from_config(config_or_camouflage_cfg: Dict[str, Any]) -> np.ndarray:
    cfg = dict(config_or_camouflage_cfg or {})
    if isinstance(cfg.get('mesh'), dict):
        cfg = dict(cfg.get('mesh', {}))
    if isinstance(cfg.get('camouflage_adjust'), dict):
        cfg = dict(cfg.get('camouflage_adjust', {}))
    if isinstance(cfg.get('camouflage'), dict):
        cfg = dict(cfg.get('camouflage', {}))
    scale_length = float(cfg.get('scale_length', 1.0))
    scale_width = float(cfg.get('scale_width', 1.0))
    scale_height = float(cfg.get('scale_height', 1.0))
    if scale_length <= 0.0 or scale_width <= 0.0 or scale_height <= 0.0:
        raise ValueError('camouflage mesh per-axis scales must all be > 0')
    return np.asarray([scale_length, scale_width, scale_height], dtype=np.float32)

def _default_obj_to_body_matrix() -> np.ndarray:
    return np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

def _mirror_symmetry_score_for_axis(verts_centered_xyz: np.ndarray, *, mirror_axis: int, sample_count: int=1024, chunk_size: int=128) -> float:
    verts = np.asarray(verts_centered_xyz, dtype=np.float32).reshape(-1, 3)
    if verts.shape[0] == 0:
        return float('inf')
    extent = np.ptp(verts, axis=0).astype(np.float32)
    scale = np.maximum(extent, 0.001).reshape(1, 3)
    verts_norm = verts / scale
    if verts_norm.shape[0] > sample_count:
        sample_idx = np.linspace(0, verts_norm.shape[0] - 1, sample_count, dtype=np.int64)
        sample = verts_norm[sample_idx]
    else:
        sample = verts_norm
    mirrored = sample.copy()
    mirrored[:, mirror_axis] *= -1.0
    best_sq = []
    for start in range(0, mirrored.shape[0], chunk_size):
        query = mirrored[start:start + chunk_size]
        diff = query[:, None, :] - verts_norm[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        best_sq.append(np.min(dist_sq, axis=1))
    all_best_sq = np.concatenate(best_sq, axis=0)
    return float(np.mean(np.sqrt(all_best_sq + 1e-12)))

def _infer_obj_to_body_matrix_from_vertices(verts_xyz: np.ndarray) -> np.ndarray:
    verts = np.asarray(verts_xyz, dtype=np.float32).reshape(-1, 3)
    if verts.shape[0] < 8:
        return _default_obj_to_body_matrix()
    mesh_min = verts.min(axis=0)
    mesh_max = verts.max(axis=0)
    mesh_center = 0.5 * (mesh_min + mesh_max)
    verts_centered = verts - mesh_center.reshape(1, 3)
    extent = (mesh_max - mesh_min).astype(np.float32)
    axis_order = np.argsort(-extent)
    length_axis = int(axis_order[0])
    candidate_axes = [int(axis) for axis in range(3) if int(axis) != length_axis]
    symmetry_scores = {axis: _mirror_symmetry_score_for_axis(verts_centered, mirror_axis=axis) for axis in candidate_axes}
    width_axis = min(candidate_axes, key=lambda axis: symmetry_scores[axis])
    up_axis = int(next((axis for axis in candidate_axes if axis != width_axis)))
    up_positive_span = float(mesh_max[up_axis] - mesh_center[up_axis])
    up_negative_span = float(mesh_center[up_axis] - mesh_min[up_axis])
    up_sign = 1.0 if up_positive_span >= up_negative_span else -1.0
    up_coord = verts_centered[:, up_axis] * up_sign
    length_coord = verts_centered[:, length_axis]
    neg_thresh = float(np.quantile(length_coord, 0.15))
    pos_thresh = float(np.quantile(length_coord, 0.85))
    neg_end = verts_centered[length_coord <= neg_thresh]
    pos_end = verts_centered[length_coord >= pos_thresh]
    if neg_end.shape[0] < 32 or pos_end.shape[0] < 32:
        half_len = 0.5 * float(extent[length_axis])
        end_margin = 0.15 * float(extent[length_axis])
        neg_end = verts_centered[length_coord <= -half_len + end_margin]
        pos_end = verts_centered[length_coord >= half_len - end_margin]

    def _roof_height_score(end_verts: np.ndarray) -> float:
        if end_verts.shape[0] == 0:
            return 0.0
        heights = end_verts[:, up_axis] * up_sign
        top_q = float(np.quantile(heights, 0.9))
        top_slice = heights[heights >= top_q]
        if top_slice.size == 0:
            return float(np.max(heights))
        return float(np.mean(top_slice))
    neg_roof = _roof_height_score(neg_end)
    pos_roof = _roof_height_score(pos_end)
    forward_sign = 1.0 if pos_roof < neg_roof else -1.0
    body_x = np.zeros(3, dtype=np.float32)
    body_x[length_axis] = forward_sign
    body_z = np.zeros(3, dtype=np.float32)
    body_z[up_axis] = up_sign
    body_y = np.cross(body_z, body_x).astype(np.float32)
    if not np.any(np.abs(body_y) > 0.5):
        body_y = np.zeros(3, dtype=np.float32)
        body_y[width_axis] = 1.0
    else:
        body_y = np.sign(body_y) * (np.abs(body_y) > 0.5).astype(np.float32)
    matrix = np.stack([body_x, body_y, body_z], axis=0).astype(np.float32)
    if float(np.linalg.det(matrix)) < 0.0:
        body_y = -body_y
        matrix = np.stack([body_x, body_y, body_z], axis=0).astype(np.float32)
    _validate_obj_to_body_matrix(matrix)
    return matrix

def _validate_obj_to_body_matrix(matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
    should_be_identity = matrix.T @ matrix
    determinant = float(np.linalg.det(matrix))
    if not np.allclose(should_be_identity, np.eye(3, dtype=np.float32), atol=0.0001):
        raise ValueError('obj_to_body_matrix must be orthonormal')
    if not np.isclose(determinant, 1.0, atol=0.0001):
        raise ValueError(f'obj_to_body_matrix must be a proper rotation (det=+1), got det={determinant:.6f}')

def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fp:
        json.dump(_to_builtin(payload), fp, ensure_ascii=False, indent=2)

def _mesh_obj_path_from_config(config: Dict[str, Any]) -> Path:
    mesh_cfg = config.get('mesh', {}) if isinstance(config.get('mesh', {}), dict) else {}
    raw = str(mesh_cfg.get('obj_path', '') or '').strip()
    if not raw:
        raise ValueError('Missing in config.yaml: mesh.obj_path')
    return _as_path(raw)

def _import_torch_and_pytorch3d() -> Tuple[Any, Any]:
    try:
        import torch
        from pytorch3d.io import load_objs_as_meshes
        from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
        from pytorch3d.structures import Meshes
        from pytorch3d.utils import cameras_from_opencv_projection
    except Exception as exc:
        raise ImportError('Cannot import torch/pytorch3d. Install torch and pytorch3d first.') from exc
    return (torch, load_objs_as_meshes, Meshes, MeshRasterizer, RasterizationSettings, cameras_from_opencv_projection)

def _import_nuscenes() -> Any:
    try:
        from nuscenes.nuscenes import NuScenes
    except Exception as exc:
        raise ImportError('Cannot import nuscenes-devkit. Install with: pip install nuscenes-devkit') from exc
    return NuScenes

def _load_binding_or_match(*, config_path: Path, binding_yaml: Optional[Path], temp_dir: Optional[Path], near_plane_m: float, run_match_if_missing: bool) -> Tuple[Dict[str, Any], Path]:
    if binding_yaml is not None:
        binding_path = _as_path(binding_yaml)
        payload = _load_yaml(binding_path)
        return (payload, binding_path)
    if not run_match_if_missing:
        raise FileNotFoundError('binding_yaml is required unless --run-match-if-missing is enabled.')
    binding_payload, _, _ = run_target_matching(config_path=config_path, sequence_yaml=None, near_plane_m=near_plane_m, verbose=True)
    if temp_dir is not None:
        temp_dir = _as_path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        binding_path = temp_dir / 'target-car-binding.yaml'
    _save_yaml(binding_path, binding_payload)
    return (binding_payload, binding_path)

def _vehicle_box_corners_local(size_wlh: List[float]) -> np.ndarray:
    width, length, height = [float(v) for v in size_wlh]
    half_l, half_w, half_h = (0.5 * length, 0.5 * width, 0.5 * height)
    return np.asarray([[half_l, half_w, half_h], [half_l, -half_w, half_h], [-half_l, -half_w, half_h], [-half_l, half_w, half_h], [half_l, half_w, -half_h], [half_l, -half_w, -half_h], [-half_l, -half_w, -half_h], [-half_l, half_w, -half_h]], dtype=np.float32)

def _box_corners_world_from_ann(ann: Dict[str, Any]) -> np.ndarray:
    center_xyz = np.asarray(ann['translation'], dtype=np.float32).reshape(3)
    size_wlh = [float(v) for v in ann['size']]
    rotation_wxyz = [float(v) for v in ann['rotation']]
    local = _vehicle_box_corners_local(size_wlh=size_wlh)
    rot = quaternion_to_rotation_matrix(rotation_wxyz)
    return local @ rot.T + center_xyz.reshape(1, 3)

def _project_world_points(points_world_xyz: np.ndarray, *, sensor_from_global: np.ndarray, camera_intrinsic: np.ndarray, near_plane_m: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_world = np.asarray(points_world_xyz, dtype=np.float32).reshape(-1, 3)
    sensor_from_global = np.asarray(sensor_from_global, dtype=np.float32).reshape(4, 4)
    camera_intrinsic = np.asarray(camera_intrinsic, dtype=np.float32).reshape(3, 3)
    points_world_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)], axis=1)
    points_cam = (sensor_from_global @ points_world_h.T).T[:, :3]
    valid = points_cam[:, 2] > float(near_plane_m)
    points_xy = np.full((points_world.shape[0], 2), np.nan, dtype=np.float32)
    if bool(np.any(valid)):
        projected = (camera_intrinsic @ points_cam[valid].T).T
        points_xy[valid] = projected[:, :2] / projected[:, 2:3]
    return (points_xy, valid, points_cam)

def _bbox_from_points_xy(points_xy: np.ndarray, image_w: int, image_h: int) -> Optional[List[float]]:
    finite = np.all(np.isfinite(points_xy), axis=1)
    if not bool(np.any(finite)):
        return None
    pts = points_xy[finite]
    x1 = float(max(0.0, min(float(image_w - 1), float(np.min(pts[:, 0])))))
    y1 = float(max(0.0, min(float(image_h - 1), float(np.min(pts[:, 1])))))
    x2 = float(max(0.0, min(float(image_w - 1), float(np.max(pts[:, 0])))))
    y2 = float(max(0.0, min(float(image_h - 1), float(np.max(pts[:, 1])))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]

def _default_prompt_offsets_body() -> np.ndarray:
    return np.asarray([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [-0.3, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.3], [0.0, 0.0, 0.5], [0.0, 0.0, 0.7]], dtype=np.float32)

def _project_prompt_points_from_gt(*, center_xyz_gt: np.ndarray, size_wlh: np.ndarray, rotation_wxyz: List[float], sensor_from_global: np.ndarray, camera_intrinsic: np.ndarray, image_w: int, image_h: int, near_plane_m: float) -> Tuple[List[List[float]], Optional[List[float]], List[List[float]], List[List[float]]]:
    center_xyz_gt = np.asarray(center_xyz_gt, dtype=np.float32).reshape(3)
    size_wlh = np.asarray(size_wlh, dtype=np.float32).reshape(3)
    rot_world = quaternion_to_rotation_matrix(rotation_wxyz)
    half_length = max(float(size_wlh[1]) * 0.5 - 0.001, 0.001)
    half_width = max(float(size_wlh[0]) * 0.5 - 0.001, 0.001)
    half_height = max(float(size_wlh[2]) * 0.5 - 0.001, 0.001)
    offsets_body = _default_prompt_offsets_body().copy()
    offsets_body[:, 0] = np.clip(offsets_body[:, 0], -half_length, half_length)
    offsets_body[:, 1] = np.clip(offsets_body[:, 1], -half_width, half_width)
    offsets_body[:, 2] = np.clip(offsets_body[:, 2], -half_height, half_height)
    points_world = offsets_body @ rot_world.T + center_xyz_gt.reshape(1, 3)
    points_xy, valid, _ = _project_world_points(points_world, sensor_from_global=sensor_from_global, camera_intrinsic=camera_intrinsic, near_plane_m=near_plane_m)
    prompt_points_xy: List[List[float]] = []
    prompt_points_body: List[List[float]] = []
    prompt_points_world: List[List[float]] = []
    seen = set()
    for idx in range(points_xy.shape[0]):
        if not bool(valid[idx]):
            continue
        x = float(points_xy[idx, 0])
        y = float(points_xy[idx, 1])
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if not (0.0 <= x <= float(image_w - 1) and 0.0 <= y <= float(image_h - 1)):
            continue
        key = (int(round(x)), int(round(y)))
        if key in seen:
            continue
        seen.add(key)
        prompt_points_xy.append([x, y])
        prompt_points_body.append([float(v) for v in offsets_body[idx]])
        prompt_points_world.append([float(v) for v in points_world[idx]])
    center_xy = prompt_points_xy[0] if prompt_points_xy else None
    return (prompt_points_xy, center_xy, prompt_points_body, prompt_points_world)

def _merge_prompt_points_xy(*point_groups: List[List[float]]) -> List[List[float]]:
    out: List[List[float]] = []
    seen = set()
    for group in point_groups:
        for pt in group:
            if not isinstance(pt, list) or len(pt) != 2:
                continue
            x = float(pt[0])
            y = float(pt[1])
            key = (int(round(x)), int(round(y)))
            if key in seen:
                continue
            seen.add(key)
            out.append([x, y])
    return out

def _bbox_from_binary_mask(mask_u8: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.where(mask_u8 > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    x1 = float(np.min(xs))
    y1 = float(np.min(ys))
    x2 = float(np.max(xs))
    y2 = float(np.max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]

def _resolve_image_path(*, dataroot: Path, image_dataroot: Path, scenario_root: Optional[Path], scenario_roots: Optional[Sequence[Path]]=None, image_source_subdir: str, channel: str, image_name: str, sample_data_filename: str) -> Optional[Path]:
    basename = Path(image_name).name
    sample_rel = Path(sample_data_filename) if str(sample_data_filename).strip() else None
    candidates: List[Path] = []
    if sample_rel is not None:
        candidates.extend([dataroot / sample_rel, image_dataroot / sample_rel])
    candidates.extend([dataroot / image_source_subdir / channel / basename, image_dataroot / image_source_subdir / channel / basename, dataroot / 'samples' / channel / basename, image_dataroot / 'samples' / channel / basename, dataroot / basename, image_dataroot / basename])
    all_scenario_roots: List[Path] = []
    if scenario_root is not None:
        all_scenario_roots.append(scenario_root)
    if scenario_roots:
        all_scenario_roots.extend((Path(path) for path in scenario_roots))
    for root in all_scenario_roots:
        candidates.extend([root / image_source_subdir / channel / basename, root / basename])
    seen = set()
    unique_candidates: List[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)
    for candidate in unique_candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    for root in all_scenario_roots:
        if root.exists():
            recursive_matches = list(root.rglob(basename))
            for candidate in recursive_matches:
                if candidate.exists() and candidate.is_file():
                    return candidate.resolve()
    return None

def _config_value_to_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if ',' in text:
        return [part.strip() for part in text.split(',') if part.strip()]
    return [text]

def _scenario_roots_from_dataset_cfg(dataset_cfg: Dict[str, Any]) -> List[Path]:
    raw_roots = dataset_cfg.get('scenario_roots', dataset_cfg.get('scenario_root', ''))
    roots = _config_value_to_string_list(raw_roots)
    resolved: List[Path] = []
    seen = set()
    for raw in roots:
        path = _as_path(raw)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(path)
    return resolved

def _stable_seed(*parts: str, base_seed: int) -> int:
    content = '|'.join(parts).encode('utf-8')
    digest = hashlib.md5(content).hexdigest()[:8]
    return (int(digest, 16) + int(base_seed)) % (2 ** 32 - 1)

def _generate_camo_pattern(height: int, width: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    base_color = np.asarray([82, 104, 64], dtype=np.uint8)
    canvas[:] = base_color
    palette = [np.asarray([52, 82, 42], dtype=np.uint8), np.asarray([98, 126, 74], dtype=np.uint8), np.asarray([120, 98, 62], dtype=np.uint8)]
    blobs = max(60, int(height * width / 35000))
    for _ in range(blobs):
        color = palette[int(rng.integers(0, len(palette)))]
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        rx = int(rng.integers(max(8, width // 80), max(14, width // 20)))
        ry = int(rng.integers(max(8, height // 80), max(14, height // 20)))
        angle = float(rng.uniform(0.0, 180.0))
        cv2.ellipse(canvas, (cx, cy), (rx, ry), angle, 0.0, 360.0, tuple((int(v) for v in color)), -1)
    canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=2.0, sigmaY=2.0)
    return canvas

def _draw_debug_overlay(*, image_rgb: np.ndarray, raw_camouflage_rgb: np.ndarray, mesh_mask: np.ndarray, gt_bbox_xyxy: Optional[List[float]], mesh_bbox_xyxy: Optional[List[float]], title: str) -> np.ndarray:
    vis = raw_camouflage_rgb.copy()
    if gt_bbox_xyxy is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in gt_bbox_xyxy]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 255, 80), 2)
    if mesh_bbox_xyxy is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in mesh_bbox_xyxy]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 96, 96), 2)
    contour, _ = cv2.findContours((mesh_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contour, -1, (255, 220, 80), 1)
    cv2.putText(vis, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return vis

def run_mesh_projection(*, config_path: Path, binding_yaml: Optional[Path]=None, mesh_obj: Optional[Path]=None, output_dir: Optional[Path]=None, device: str='auto', near_plane_m: float=DEFAULT_NEAR_PLANE_M, run_match_if_missing: bool=False, camo_alpha: float=0.72, camo_seed: int=1234, save_json: bool=False, verbose: bool=True) -> Tuple[Dict[str, Any], Path]:
    config_path = _as_path(config_path)
    config = _load_yaml(config_path)
    mesh_obj_path = _mesh_obj_path_from_config(config) if mesh_obj is None else _as_path(mesh_obj)
    if not mesh_obj_path.exists():
        raise FileNotFoundError(f'Mesh obj not found: {mesh_obj_path}')
    binding_payload, binding_path = _load_binding_or_match(config_path=config_path, binding_yaml=binding_yaml, temp_dir=output_dir / '.tmp_binding' if output_dir is not None else None, near_plane_m=near_plane_m, run_match_if_missing=run_match_if_missing)
    dataset_cfg = config.get('dataset', {}) if isinstance(config, dict) else {}
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
    dataroot = _as_path(str(binding_payload.get('dataroot', dataset_cfg.get('dataroot', ''))))
    image_dataroot = _as_path(str(binding_payload.get('image_dataroot', dataset_cfg.get('image_dataroot', str(dataroot)))))
    scenario_roots = _scenario_roots_from_dataset_cfg(dataset_cfg)
    scenario_root = scenario_roots[0] if scenario_roots else None
    image_source_subdir = str(binding_payload.get('image_source_subdir', dataset_cfg.get('image_source_subdir', 'samples-2'))).strip() or 'samples-2'
    version = str(binding_payload.get('version', 'v1.0-trainval')).strip() or 'v1.0-trainval'
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix='zz3-mesh-proj-')).resolve()
    output_dir = _as_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    NuScenes = _import_nuscenes()
    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
    torch, load_objs_as_meshes, Meshes, MeshRasterizer, RasterizationSettings, cameras_from_opencv_projection = _import_torch_and_pytorch3d()
    if device == 'auto':
        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        torch_device = torch.device(str(device))
    mesh = load_objs_as_meshes([str(mesh_obj_path)], device=torch_device)
    verts_ref = mesh.verts_packed().to(dtype=torch.float32)
    faces_idx = mesh.faces_packed().to(dtype=torch.int64)
    mesh_min = verts_ref.min(dim=0).values
    mesh_max = verts_ref.max(dim=0).values
    mesh_center = 0.5 * (mesh_min + mesh_max)
    obj_to_body_np = _infer_obj_to_body_matrix_from_vertices(verts_ref.detach().cpu().numpy())
    _validate_obj_to_body_matrix(obj_to_body_np)
    obj_to_body = torch.as_tensor(obj_to_body_np, device=torch_device, dtype=torch.float32)
    verts_body_ref = (verts_ref - mesh_center) @ obj_to_body.T
    mesh_extent_body = (verts_body_ref.max(dim=0).values - verts_body_ref.min(dim=0).values).clamp(min=1e-06)
    rasterizer_cache: Dict[Tuple[int, int], Any] = {}
    frames = binding_payload.get('frames', [])
    if not isinstance(frames, list) or not frames:
        raise RuntimeError('Binding payload has no frames')
    projection_records: List[Dict[str, Any]] = []
    rendered_count = 0
    for index, frame in enumerate(frames, start=1):
        if not isinstance(frame, dict):
            continue
        ann_token = str(frame.get('ann_token', '')).strip()
        sample_token = str(frame.get('sample_token', '')).strip()
        frame_id = int(frame.get('frame_id', index))
        sequence_name = str(frame.get('sequence_name', binding_payload.get('sequence_name', 'sequence')) or 'sequence')
        if not ann_token or not sample_token:
            continue
        ann = nusc.get('sample_annotation', ann_token)
        sample = nusc.get('sample', sample_token)
        center_xyz_gt = np.asarray(ann['translation'], dtype=np.float32).reshape(3)
        size_wlh = np.asarray(ann['size'], dtype=np.float32).reshape(3)
        rotation_wxyz = [float(v) for v in ann['rotation']]
        rot_world_np = quaternion_to_rotation_matrix(rotation_wxyz)
        camouflage_body_offset_xyz = _camouflage_body_offset_xyz_from_config(frame)
        mesh_global_scale = _camouflage_mesh_global_scale_from_config(frame)
        mesh_scale_xyz = _camouflage_mesh_scale_xyz_from_config(frame)
        center_xyz = center_xyz_gt + rot_world_np @ camouflage_body_offset_xyz
        target_extent_xyz = torch.as_tensor([float(size_wlh[1]), float(size_wlh[0]), float(size_wlh[2])], device=torch_device, dtype=torch.float32)
        extra_scale_xyz = torch.as_tensor(mesh_scale_xyz, device=torch_device, dtype=torch.float32)
        scale_xyz = target_extent_xyz / mesh_extent_body * extra_scale_xyz * float(mesh_global_scale)
        rot_world = torch.as_tensor(rot_world_np, device=torch_device, dtype=torch.float32)
        verts_aligned = verts_body_ref * scale_xyz
        verts_world = verts_aligned @ rot_world.T + torch.as_tensor(center_xyz, device=torch_device, dtype=torch.float32)
        verts_world_np = verts_world.detach().cpu().numpy()
        gt_box_corners_world = _box_corners_world_from_ann(ann)
        for channel in CAMERA_CHANNELS:
            sample_data_token = str(sample['data'].get(channel, ''))
            if not sample_data_token:
                continue
            sample_data = nusc.get('sample_data', sample_data_token)
            calibrated_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
            ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
            camera_intrinsic = np.asarray(calibrated_sensor['camera_intrinsic'], dtype=np.float32).reshape(3, 3)
            sensor_from_global = sensor_from_global_matrix(ego_pose=ego_pose, calibrated_sensor=calibrated_sensor)
            image_h = int(sample_data.get('height', 0))
            image_w = int(sample_data.get('width', 0))
            if image_h <= 0 or image_w <= 0:
                continue
            gt_xy, gt_valid, _ = _project_world_points(gt_box_corners_world, sensor_from_global=sensor_from_global, camera_intrinsic=camera_intrinsic, near_plane_m=near_plane_m)
            gt_bbox = _bbox_from_points_xy(gt_xy, image_w=image_w, image_h=image_h)
            if gt_bbox is None:
                continue
            center_xy_proj, center_valid, _ = _project_world_points(center_xyz_gt.reshape(1, 3), sensor_from_global=sensor_from_global, camera_intrinsic=camera_intrinsic, near_plane_m=near_plane_m)
            auto_prompt_points_xy, auto_center_xy, auto_prompt_points_body, auto_prompt_points_world = _project_prompt_points_from_gt(center_xyz_gt=center_xyz_gt, size_wlh=size_wlh, rotation_wxyz=rotation_wxyz, sensor_from_global=sensor_from_global, camera_intrinsic=camera_intrinsic, image_w=image_w, image_h=image_h, near_plane_m=near_plane_m)
            explicit_prompt_specs, has_explicit_prompt_specs = _resolve_explicit_prompt_spec_map(frame)
            manual_prompt_points_xy: List[List[float]] = []
            manual_negative_prompt_points_xy: List[List[float]] = []
            if has_explicit_prompt_specs:
                channel_prompt_spec = explicit_prompt_specs.get(channel, None)
                if not isinstance(channel_prompt_spec, dict):
                    continue
                manual_prompt_points_xy = _normalize_prompt_points(channel_prompt_spec.get('positive', []))
                manual_negative_prompt_points_xy = _normalize_prompt_points(channel_prompt_spec.get('negative', []))
                if not manual_prompt_points_xy:
                    continue
            elif isinstance(frame.get('target_prompt_points_by_channel', None), dict):
                manual_prompt_points_xy = [[float(pt[0]), float(pt[1])] for pt in frame.get('target_prompt_points_by_channel', {}).get(channel, []) if isinstance(pt, list) and len(pt) == 2]
            elif isinstance(frame.get('target_prompt_points_xy', None), list):
                manual_prompt_points_xy = [[float(pt[0]), float(pt[1])] for pt in frame.get('target_prompt_points_xy', []) if isinstance(pt, list) and len(pt) == 2]
            combined_prompt_points_xy = _merge_prompt_points_xy(manual_prompt_points_xy, auto_prompt_points_xy)
            if has_explicit_prompt_specs:
                target_center_xy_current = [float(manual_prompt_points_xy[0][0]), float(manual_prompt_points_xy[0][1])]
                prompt_points_xy_current = manual_prompt_points_xy
                prompt_points_body_current = []
                prompt_points_world_current = []
                prompt_source = 'manual_yaml_explicit'
            elif manual_prompt_points_xy and auto_prompt_points_xy:
                target_center_xy_current = [float(manual_prompt_points_xy[0][0]), float(manual_prompt_points_xy[0][1])]
                prompt_points_xy_current = combined_prompt_points_xy
                prompt_points_body_current = auto_prompt_points_body
                prompt_points_world_current = auto_prompt_points_world
                prompt_source = 'manual_yaml+gt_3d_projected'
            elif manual_prompt_points_xy:
                target_center_xy_current = [float(manual_prompt_points_xy[0][0]), float(manual_prompt_points_xy[0][1])]
                prompt_points_xy_current = manual_prompt_points_xy
                prompt_points_body_current = []
                prompt_points_world_current = []
                prompt_source = 'manual_yaml'
            elif auto_prompt_points_xy:
                target_center_xy_current = [float(auto_center_xy[0]), float(auto_center_xy[1])] if auto_center_xy is not None else None
                prompt_points_xy_current = auto_prompt_points_xy
                prompt_points_body_current = auto_prompt_points_body
                prompt_points_world_current = auto_prompt_points_world
                prompt_source = 'gt_3d_projected'
            elif bool(center_valid[0]) and np.all(np.isfinite(center_xy_proj[0])):
                target_center_xy_current = [float(center_xy_proj[0, 0]), float(center_xy_proj[0, 1])]
                prompt_points_xy_current = [target_center_xy_current]
                prompt_points_body_current = [[0.0, 0.0, 0.0]]
                prompt_points_world_current = [[float(v) for v in center_xyz_gt]]
                prompt_source = 'gt_center_projected_fallback'
            else:
                target_center_xy_current = None
                prompt_points_xy_current = []
                prompt_points_body_current = []
                prompt_points_world_current = []
                prompt_source = 'none'
            proj_xy, vert_valid, verts_cam = _project_world_points(verts_world_np, sensor_from_global=sensor_from_global, camera_intrinsic=camera_intrinsic, near_plane_m=near_plane_m)
            key_hw = (image_h, image_w)
            rasterizer = rasterizer_cache.get(key_hw, None)
            if rasterizer is None:
                rast_settings = RasterizationSettings(image_size=(image_h, image_w), blur_radius=0.0, faces_per_pixel=1, perspective_correct=True, bin_size=0, cull_backfaces=True, cull_to_frustum=True)
                rasterizer = MeshRasterizer(raster_settings=rast_settings)
                rasterizer_cache[key_hw] = rasterizer
            R_cv = torch.as_tensor(sensor_from_global[:3, :3], dtype=torch.float32, device=torch_device).unsqueeze(0)
            t_cv = torch.as_tensor(sensor_from_global[:3, 3], dtype=torch.float32, device=torch_device).unsqueeze(0)
            K_cv = torch.as_tensor(camera_intrinsic, dtype=torch.float32, device=torch_device).unsqueeze(0)
            image_size_t = torch.as_tensor([[float(image_h), float(image_w)]], dtype=torch.float32, device=torch_device)
            cameras = cameras_from_opencv_projection(R=R_cv, tvec=t_cv, camera_matrix=K_cv, image_size=image_size_t).to(torch_device)
            mesh_for_raster = Meshes(verts=[verts_world], faces=[faces_idx])
            fragments = rasterizer(mesh_for_raster, cameras=cameras)
            mesh_mask = (fragments.pix_to_face[0, :, :, 0] >= 0).to(torch.uint8).detach().cpu().numpy() * 255
            if int(np.sum(mesh_mask > 0)) == 0:
                continue
            mesh_bbox = _bbox_from_binary_mask(mesh_mask)
            if mesh_bbox is None:
                continue
            image_name = Path(str(sample_data.get('filename', ''))).name
            image_path = _resolve_image_path(dataroot=dataroot, image_dataroot=image_dataroot, scenario_root=scenario_root, scenario_roots=scenario_roots, image_source_subdir=image_source_subdir, channel=channel, image_name=image_name, sample_data_filename=str(sample_data.get('filename', '')))
            if image_path is None:
                continue
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            seed = _stable_seed(sample_token, channel, base_seed=camo_seed)
            camo_pattern = _generate_camo_pattern(image_h, image_w, seed=seed)
            keep = mesh_mask > 0
            raw_camouflage = image_rgb.copy()
            raw_camouflage[keep] = ((1.0 - float(camo_alpha)) * raw_camouflage[keep].astype(np.float32) + float(camo_alpha) * camo_pattern[keep].astype(np.float32)).round().clip(0, 255).astype(np.uint8)
            title = f'frame={frame_id} {channel} ann={ann_token[:8]} valid={int(np.sum(vert_valid))}/{int(vert_valid.shape[0])}'
            debug_overlay = _draw_debug_overlay(image_rgb=image_rgb, raw_camouflage_rgb=raw_camouflage, mesh_mask=mesh_mask, gt_bbox_xyxy=gt_bbox, mesh_bbox_xyxy=mesh_bbox, title=title)
            stem = f'frame_{frame_id:04d}_{channel}_{Path(image_name).stem}'
            raw_path = output_dir / 'raw_camouflage' / f'{stem}.png'
            mask_path = output_dir / 'mesh_masks' / f'{stem}.png'
            overlay_path = output_dir / 'overlays' / f'{stem}.png'
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(raw_path), cv2.cvtColor(raw_camouflage, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(mask_path), mesh_mask)
            cv2.imwrite(str(overlay_path), cv2.cvtColor(debug_overlay, cv2.COLOR_RGB2BGR))
            projection_records.append({'sequence_name': sequence_name, 'frame_id': frame_id, 'sample_token': sample_token, 'scene_token': str(sample.get('scene_token', '')), 'ann_token': ann_token, 'instance_token': str(ann.get('instance_token', '')), 'channel': channel, 'image_name': image_name, 'image_path': str(image_path), 'raw_camouflage_path': str(raw_path), 'mesh_mask_path': str(mask_path), 'overlay_path': str(overlay_path), 'target_center_xy': target_center_xy_current, 'target_prompt_source': prompt_source, 'target_prompt_points_explicit': bool(has_explicit_prompt_specs), 'target_prompt_points_xy': prompt_points_xy_current, 'target_negative_prompt_points_xy': manual_negative_prompt_points_xy, 'target_prompt_points_body_xyz': prompt_points_body_current, 'target_prompt_points_world_xyz': prompt_points_world_current, 'mesh_bbox_xyxy': [float(v) for v in mesh_bbox] if mesh_bbox is not None else None, 'gt_bbox_xyxy': [float(v) for v in gt_bbox] if gt_bbox is not None else None, 'num_vertices_total': int(verts_world_np.shape[0]), 'num_vertices_valid': int(np.sum(vert_valid)), 'zbuffer_enabled': True, 'camera': {'sample_data_token': sample_data_token, 'sample_data_filename': str(sample_data.get('filename', '')), 'width': image_w, 'height': image_h, 'camera_intrinsic': camera_intrinsic.tolist(), 'sensor_from_global': sensor_from_global.tolist()}, 'gt_3d': {'center_xyz': [float(v) for v in center_xyz_gt], 'camouflage_center_xyz': [float(v) for v in center_xyz], 'size_wlh': [float(v) for v in ann['size']], 'rotation_wxyz': [float(v) for v in ann['rotation']], 'camouflage_offset_body_xyz_m': [float(v) for v in camouflage_body_offset_xyz], 'camouflage_mesh_global_scale': float(mesh_global_scale), 'camouflage_mesh_scale_xyz': [float(v) for v in mesh_scale_xyz]}})
            rendered_count += 1
    summary = {'config_path': str(config_path), 'binding_yaml': str(binding_path), 'mesh_obj': str(mesh_obj_path), 'version': version, 'dataroot': str(dataroot), 'image_dataroot': str(image_dataroot), 'image_source_subdir': image_source_subdir, 'device': str(torch_device), 'near_plane_m': float(near_plane_m), 'camo_alpha': float(camo_alpha), 'camouflage_adjust_source': 'sequence_yaml.frames[*].mesh via temporary binding payload', 'mesh_alignment_assumption': 'mesh obj->body axis remap then scale to [length,width,height], then quaternion world rotation', 'obj_to_body_matrix': obj_to_body_np.tolist(), 'rasterization': 'pytorch3d_zbuffer_faces_per_pixel_1_cull_backfaces_true', 'rendered_view_count': rendered_count, 'records': projection_records}
    summary_yaml_path = output_dir / 'mesh_projection_summary.yaml'
    _save_yaml(summary_yaml_path, summary)
    if save_json:
        _save_json(output_dir / 'mesh_projection_summary.json', summary)
    if verbose:
        print(f'[nusc_gt_to_mesh] rendered_views={rendered_count} output={summary_yaml_path}')
    summary_builtin = _to_builtin(summary)
    del rasterizer_cache
    del mesh
    del verts_ref
    del faces_idx
    del verts_body_ref
    del mesh_extent_body
    gc.collect()
    if torch_device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (summary_builtin, summary_yaml_path)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Load mesh with PyTorch3D, align mesh to nuScenes GT 3D bbox, and render/project to all 6 cameras')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH, help='Path to config.yaml')
    parser.add_argument('--binding-yaml', type=Path, default=None, help='Target-binding yaml from match_target_car.py')
    parser.add_argument('--mesh-obj', type=Path, default=None, help='Mesh OBJ path; defaults to mesh.obj_path in config.yaml')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto/cpu/cuda')
    parser.add_argument('--near-plane', type=float, default=DEFAULT_NEAR_PLANE_M, help='Near plane in meters')
    parser.add_argument('--run-match-if-missing', action='store_true', help='If binding yaml is missing, run match_target_car automatically')
    parser.add_argument('--camo-alpha', type=float, default=0.72, help='Camouflage blending alpha')
    parser.add_argument('--camo-seed', type=int, default=1234, help='Camouflage random seed')
    parser.add_argument('--save-json', action='store_true', help='Also save projection summary in JSON')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    _, summary_path = run_mesh_projection(config_path=args.config, binding_yaml=args.binding_yaml, mesh_obj=args.mesh_obj, output_dir=args.output_dir, device=args.device, near_plane_m=float(args.near_plane), run_match_if_missing=bool(args.run_match_if_missing), camo_alpha=float(args.camo_alpha), camo_seed=int(args.camo_seed), save_json=bool(args.save_json), verbose=True)
    print(summary_path)
if __name__ == '__main__':
    main()
