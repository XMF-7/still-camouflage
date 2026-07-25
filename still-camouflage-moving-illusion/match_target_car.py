from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import yaml
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / 'config.yaml'
DEFAULT_NEAR_PLANE_M = 0.1

def _as_path(value: str | Path) -> Path:
    return Path(str(value)).expanduser().resolve()

def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as fp:
        payload = yaml.safe_load(fp)
    if not isinstance(payload, dict):
        raise RuntimeError(f'YAML top-level must be dict: {path}')
    return payload

def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value

def _save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fp:
        yaml.safe_dump(_to_builtin(payload), fp, allow_unicode=False, sort_keys=False)

def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fp:
        json.dump(_to_builtin(payload), fp, ensure_ascii=False, indent=2)

def _bbox_area_xyxy(bbox: List[float]) -> float:
    return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))

def _bbox_center_xy(bbox: List[float]) -> np.ndarray:
    return np.asarray([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float32)

def _bbox_contains_point(bbox: List[float], point_xy: np.ndarray) -> bool:
    return bool(bbox[0] <= point_xy[0] <= bbox[2] and bbox[1] <= point_xy[1] <= bbox[3])

def quaternion_to_rotation_matrix(rotation_wxyz: List[float] | Tuple[float, float, float, float]) -> np.ndarray:
    if len(rotation_wxyz) != 4:
        raise ValueError(f'Quaternion must be [w, x, y, z], got len={len(rotation_wxyz)}')
    w, x, y, z = [float(v) for v in rotation_wxyz]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError('Quaternion norm is too small')
    w, x, y, z = (w / norm, x / norm, y / norm, z / norm)
    return np.asarray([[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)], [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)], [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]], dtype=np.float32)

def yaw_from_quaternion(rotation_wxyz: List[float] | Tuple[float, float, float, float]) -> float:
    w, x, y, z = [float(v) for v in rotation_wxyz]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))

def transform_matrix(translation_xyz: List[float] | Tuple[float, float, float], rotation_wxyz: List[float] | Tuple[float, float, float, float], *, inverse: bool) -> np.ndarray:
    translation = np.asarray(translation_xyz, dtype=np.float32).reshape(3)
    rotation = quaternion_to_rotation_matrix(rotation_wxyz)
    out = np.eye(4, dtype=np.float32)
    if inverse:
        inv_rotation = rotation.T
        out[:3, :3] = inv_rotation
        out[:3, 3] = -inv_rotation @ translation
    else:
        out[:3, :3] = rotation
        out[:3, 3] = translation
    return out

def sensor_from_global_matrix(ego_pose: Dict[str, Any], calibrated_sensor: Dict[str, Any]) -> np.ndarray:
    ego_from_global = transform_matrix(ego_pose['translation'], ego_pose['rotation'], inverse=True)
    sensor_from_ego = transform_matrix(calibrated_sensor['translation'], calibrated_sensor['rotation'], inverse=True)
    return sensor_from_ego @ ego_from_global

def _vehicle_box_corners_local(size_wlh: List[float]) -> np.ndarray:
    width, length, height = [float(v) for v in size_wlh]
    if min(width, length, height) <= 0.0:
        raise ValueError(f'Invalid size_wlh={size_wlh}')
    half_l, half_w, half_h = (0.5 * length, 0.5 * width, 0.5 * height)
    corners = np.asarray([[half_l, half_w, half_h], [half_l, -half_w, half_h], [-half_l, -half_w, half_h], [-half_l, half_w, half_h], [half_l, half_w, -half_h], [half_l, -half_w, -half_h], [-half_l, -half_w, -half_h], [-half_l, half_w, -half_h]], dtype=np.float32)
    return corners

def project_annotation_bbox_to_camera(ann: Dict[str, Any], sample_data: Dict[str, Any], calibrated_sensor: Dict[str, Any], ego_pose: Dict[str, Any], *, near_plane_m: float) -> Optional[List[float]]:
    center_world = np.asarray(ann['translation'], dtype=np.float32).reshape(3)
    size_wlh = [float(v) for v in ann['size']]
    rotation_wxyz = [float(v) for v in ann['rotation']]
    corners_local = _vehicle_box_corners_local(size_wlh=size_wlh)
    rotation = quaternion_to_rotation_matrix(rotation_wxyz)
    corners_world = corners_local @ rotation.T + center_world.reshape(1, 3)
    sensor_from_global = sensor_from_global_matrix(ego_pose=ego_pose, calibrated_sensor=calibrated_sensor)
    corners_world_h = np.concatenate([corners_world, np.ones((8, 1), dtype=np.float32)], axis=1)
    corners_cam = (sensor_from_global @ corners_world_h.T).T[:, :3]
    valid = corners_cam[:, 2] > float(near_plane_m)
    if not bool(np.any(valid)):
        return None
    intrinsic = np.asarray(calibrated_sensor['camera_intrinsic'], dtype=np.float32)
    proj = (intrinsic @ corners_cam[valid].T).T
    xy = proj[:, :2] / proj[:, 2:3]
    image_w = int(sample_data['width'])
    image_h = int(sample_data['height'])
    x1 = float(max(0.0, min(float(image_w - 1), float(np.min(xy[:, 0])))))
    y1 = float(max(0.0, min(float(image_h - 1), float(np.min(xy[:, 1])))))
    x2 = float(max(0.0, min(float(image_w - 1), float(np.max(xy[:, 0])))))
    y2 = float(max(0.0, min(float(image_h - 1), float(np.max(xy[:, 1])))))
    bbox = [x1, y1, x2, y2]
    if _bbox_area_xyxy(bbox) <= 0.0:
        return None
    return bbox

def project_world_point_to_camera(world_xyz: List[float] | Tuple[float, float, float] | np.ndarray, sample_data: Dict[str, Any], calibrated_sensor: Dict[str, Any], ego_pose: Dict[str, Any], *, near_plane_m: float) -> Optional[List[float]]:
    point_world = np.asarray(world_xyz, dtype=np.float32).reshape(1, 3)
    sensor_from_global = sensor_from_global_matrix(ego_pose=ego_pose, calibrated_sensor=calibrated_sensor)
    point_world_h = np.concatenate([point_world, np.ones((1, 1), dtype=np.float32)], axis=1)
    point_cam = (sensor_from_global @ point_world_h.T).T[:, :3]
    if float(point_cam[0, 2]) <= float(near_plane_m):
        return None
    intrinsic = np.asarray(calibrated_sensor['camera_intrinsic'], dtype=np.float32)
    proj = (intrinsic @ point_cam.T).T
    xy = proj[:, :2] / proj[:, 2:3]
    image_w = int(sample_data['width'])
    image_h = int(sample_data['height'])
    x = float(xy[0, 0])
    y = float(xy[0, 1])
    if not (0.0 <= x <= float(image_w - 1) and 0.0 <= y <= float(image_h - 1)):
        return None
    return [x, y]

def backproject_pixel_to_global_ray(pixel_xy: List[float] | Tuple[float, float] | np.ndarray, calibrated_sensor: Dict[str, Any], ego_pose: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    pixel = np.asarray(pixel_xy, dtype=np.float32).reshape(2)
    intrinsic = np.asarray(calibrated_sensor['camera_intrinsic'], dtype=np.float32)
    intrinsic_inv = np.linalg.inv(intrinsic)
    ray_cam = intrinsic_inv @ np.asarray([pixel[0], pixel[1], 1.0], dtype=np.float32)
    ray_cam = ray_cam / max(1e-12, float(np.linalg.norm(ray_cam)))
    global_from_sensor = np.linalg.inv(sensor_from_global_matrix(ego_pose=ego_pose, calibrated_sensor=calibrated_sensor))
    ray_origin_global = global_from_sensor[:3, 3].astype(np.float32)
    ray_direction_global = global_from_sensor[:3, :3] @ ray_cam
    ray_direction_global = ray_direction_global / max(1e-12, float(np.linalg.norm(ray_direction_global)))
    return (ray_origin_global, ray_direction_global.astype(np.float32))

def point_to_ray_distance(point_xyz: List[float] | Tuple[float, float, float] | np.ndarray, ray_origin_xyz: np.ndarray, ray_direction_xyz: np.ndarray) -> Tuple[float, float]:
    point = np.asarray(point_xyz, dtype=np.float32).reshape(3)
    origin = np.asarray(ray_origin_xyz, dtype=np.float32).reshape(3)
    direction = np.asarray(ray_direction_xyz, dtype=np.float32).reshape(3)
    vec = point - origin
    depth_along_ray = float(np.dot(vec, direction))
    closest = origin + depth_along_ray * direction
    ray_distance = float(np.linalg.norm(point - closest))
    return (ray_distance, depth_along_ray)

def _import_nuscenes() -> Any:
    try:
        from nuscenes.nuscenes import NuScenes
    except Exception as exc:
        raise ImportError('Cannot import nuscenes-devkit. Install with: pip install nuscenes-devkit') from exc
    return NuScenes

def resolve_sequence_yaml_from_config(config_path: Path, sequence_yaml: Optional[Path]=None) -> Path:
    if sequence_yaml is not None:
        return _as_path(sequence_yaml)
    cfg = _load_yaml(config_path)
    dataset_cfg = cfg.get('dataset', {}) if isinstance(cfg, dict) else {}
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
    raw_source = str(dataset_cfg.get('image_source_subdir', '')).strip()
    sequence_yaml_value = str(dataset_cfg.get('sequence_yaml', '')).strip()
    source_name = sequence_yaml_value or raw_source
    if ',' in source_name:
        source_name = source_name.split(',', 1)[0].strip()
    if not source_name:
        raise FileNotFoundError('dataset.sequence_yaml / dataset.image_source_subdir must not be empty')
    candidate = Path(source_name)
    if candidate.suffix.lower() in {'.yaml', '.yml'} and candidate.exists():
        return _as_path(candidate)
    legacy_path = config_path.parent / 'data-json' / f'{source_name}.yaml'
    if legacy_path.exists():
        return legacy_path.resolve()
    yaml_root = config_path.parent / 'data-yaml'
    matches = sorted(yaml_root.rglob(f'{source_name}.yaml')) if yaml_root.exists() else []
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise FileNotFoundError('Multiple sequence yaml files matched; set dataset.sequence_yaml to a full path: ' + ' | '.join((str(path) for path in matches)))
    raise FileNotFoundError(f'Sequence yaml not found for: {source_name}')

def default_output_paths(sequence_yaml_path: Path) -> Tuple[Path, Path]:
    parent = sequence_yaml_path.parent
    stem = sequence_yaml_path.stem
    return (parent / f'{stem}-target-car.yaml', parent / f'{stem}-bound.yaml')

def _default_camouflage_adjust() -> Dict[str, float]:
    return {'mesh_global_scale': 1.0, 'scale_length': 1.0, 'scale_width': 1.0, 'scale_height': 1.0, 'offset_front_m': 0.0, 'offset_left_m': 0.0, 'offset_up_m': 0.0}

def _normalize_camouflage_adjust(raw_value: Any) -> Dict[str, float]:
    adjust = _default_camouflage_adjust()
    if isinstance(raw_value, dict):
        for key in list(adjust.keys()):
            if key in raw_value:
                adjust[key] = float(raw_value[key])
    if float(adjust['mesh_global_scale']) <= 0.0:
        raise ValueError('camouflage_adjust.mesh_global_scale must be > 0')
    if float(adjust['scale_length']) <= 0.0:
        raise ValueError('camouflage_adjust.scale_length must be > 0')
    if float(adjust['scale_width']) <= 0.0:
        raise ValueError('camouflage_adjust.scale_width must be > 0')
    if float(adjust['scale_height']) <= 0.0:
        raise ValueError('camouflage_adjust.scale_height must be > 0')
    return adjust

def _extract_camouflage_adjust_map(payload: Any) -> Dict[int, Dict[str, float]]:
    if not isinstance(payload, dict):
        return {}
    frames = payload.get('frames', [])
    if not isinstance(frames, list):
        return {}
    out: Dict[int, Dict[str, float]] = {}
    for index, frame in enumerate(frames, start=1):
        if not isinstance(frame, dict):
            continue
        frame_id = int(frame.get('frame_id', index))
        adjust_raw = frame.get('mesh', None)
        if adjust_raw is None and isinstance(frame.get('target'), dict):
            adjust_raw = frame['target'].get('mesh', None)
        if adjust_raw is None:
            adjust_raw = frame.get('camouflage_adjust', None)
        if adjust_raw is None and isinstance(frame.get('target'), dict):
            adjust_raw = frame['target'].get('camouflage_adjust', None)
        out[frame_id] = _normalize_camouflage_adjust(adjust_raw)
    return out

def _load_existing_camouflage_adjust_map(sequence_yaml_path: Path) -> Dict[int, Dict[str, float]]:
    if not sequence_yaml_path.exists():
        return {}
    try:
        return _extract_camouflage_adjust_map(_load_yaml(sequence_yaml_path))
    except Exception:
        return {}

def _resolve_frame_channel(frame: Dict[str, Any], default_channel: str) -> str:
    images = frame.get('images', {})
    if not isinstance(images, dict):
        images = {}
    target = frame.get('target', {})
    if isinstance(target, dict):
        ch = str(target.get('channel', '')).strip()
        if ch and (not images or ch in images):
            return ch
    if default_channel and (not images or default_channel in images):
        return default_channel
    if images:
        return str(next(iter(images.keys())))
    return default_channel or 'CAM_FRONT'

def _resolve_frame_image_name(frame: Dict[str, Any], channel: str) -> str:
    images = frame.get('images', {})
    if not isinstance(images, dict):
        raise RuntimeError('frame.images must be a dict')
    image_name = str(images.get(channel, '')).strip()
    if image_name:
        return image_name
    if len(images) == 1:
        return str(next(iter(images.values())))
    raise RuntimeError(f'frame has no image for channel={channel}')

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
        positive = _normalize_prompt_points(raw_value.get('positive', []))
        negative = _normalize_prompt_points(raw_value.get('negative', []))
        return {'positive': positive, 'negative': negative}
    positive = _normalize_prompt_points(raw_value)
    return {'positive': positive, 'negative': []}

def _resolve_frame_target_prompt_spec_map(frame: Dict[str, Any], *, default_channel: str='') -> Dict[str, Dict[str, List[List[float]]]]:
    target = frame.get('target', {})
    if not isinstance(target, dict):
        return {}
    prompts_raw = target.get('prompts', None)
    if isinstance(prompts_raw, dict):
        out: Dict[str, Dict[str, List[List[float]]]] = {}
        for channel, value in prompts_raw.items():
            normalized = _normalize_channel_prompt_spec(value)
            if normalized['positive'] or normalized['negative']:
                out[str(channel)] = normalized
        if out:
            return out
    legacy_points = _normalize_prompt_points(target.get('2d_target_xy', target.get('xy', None)))
    if not legacy_points:
        return {}
    channel = str(target.get('channel', '')).strip()
    if not channel:
        images = frame.get('images', {})
        if isinstance(images, dict) and images:
            if default_channel and default_channel in images:
                channel = str(default_channel)
            else:
                channel = str(next(iter(images.keys())))
        else:
            channel = str(default_channel or 'CAM_FRONT')
    return {channel: {'positive': legacy_points, 'negative': []}}

def _resolve_frame_target_prompt_map(frame: Dict[str, Any], *, default_channel: str='') -> Dict[str, List[List[float]]]:
    spec_map = _resolve_frame_target_prompt_spec_map(frame, default_channel=default_channel)
    return {str(channel): list(spec.get('positive', [])) for channel, spec in spec_map.items() if isinstance(spec, dict) and spec.get('positive')}

def _resolve_frame_target_negative_prompt_map(frame: Dict[str, Any], *, default_channel: str='') -> Dict[str, List[List[float]]]:
    spec_map = _resolve_frame_target_prompt_spec_map(frame, default_channel=default_channel)
    return {str(channel): list(spec.get('negative', [])) for channel, spec in spec_map.items() if isinstance(spec, dict) and spec.get('negative')}

def _resolve_frame_target_prompt_points(frame: Dict[str, Any], *, channel: Optional[str]=None, default_channel: str='') -> List[List[float]]:
    prompt_map = _resolve_frame_target_prompt_map(frame, default_channel=default_channel)
    if not prompt_map:
        return []
    if channel:
        return list(prompt_map.get(channel, []))
    first_channel = next(iter(prompt_map.keys()))
    return list(prompt_map.get(first_channel, []))

def _resolve_first_target_point(first_frame: Dict[str, Any], *, channel: str) -> np.ndarray:
    points = _resolve_frame_target_prompt_points(first_frame, channel=channel, default_channel=channel)
    if points:
        return np.asarray(points[0], dtype=np.float32)
    target = first_frame.get('target', {})
    if not isinstance(target, dict):
        raise RuntimeError('first frame must provide target.prompts or target.2d_target_xy')
    point = target.get('2d_target_xy', target.get('xy', None))
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise RuntimeError('first frame target must provide target.prompts or target.2d_target_xy=[x,y]')
    return np.asarray([float(point[0]), float(point[1])], dtype=np.float32)

def _resolve_first_target_prompt_points_by_channel(first_frame: Dict[str, Any], *, default_channel: str) -> Dict[str, List[List[float]]]:
    prompt_map = _resolve_frame_target_prompt_map(first_frame, default_channel=default_channel)
    if prompt_map:
        return {str(channel): list(points) for channel, points in prompt_map.items() if points}
    target = first_frame.get('target', {})
    if not isinstance(target, dict):
        raise RuntimeError('first frame must provide target.prompts or target.2d_target_xy')
    point = target.get('2d_target_xy', target.get('xy', None))
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise RuntimeError('first frame target must provide target.prompts or target.2d_target_xy=[x,y]')
    return {str(default_channel): [[float(point[0]), float(point[1])]]}

def _build_camera_sample_data_lookup(nusc: Any) -> Dict[str, Dict[Any, Any]]:
    by_channel_and_name: Dict[Tuple[str, str], str] = {}
    by_basename: Dict[str, List[str]] = {}
    channel_by_token: Dict[str, str] = {}
    for sample_data in nusc.sample_data:
        calibrated_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        sensor = nusc.get('sensor', calibrated_sensor['sensor_token'])
        if str(sensor.get('modality', '')) != 'camera':
            continue
        token = str(sample_data['token'])
        channel = str(sensor.get('channel', '')).strip()
        filename = Path(str(sample_data.get('filename', ''))).name
        if not filename or not channel:
            continue
        by_channel_and_name[channel, filename] = token
        by_basename.setdefault(filename, []).append(token)
        channel_by_token[token] = channel
    return {'by_channel_and_name': by_channel_and_name, 'by_basename': by_basename, 'channel_by_token': channel_by_token}

def _find_sample_data_token_for_image(lookup: Dict[str, Dict[Any, Any]], *, image_name: str, channel: str) -> str:
    basename = Path(str(image_name)).name
    by_channel_and_name = lookup['by_channel_and_name']
    by_basename = lookup['by_basename']
    channel_by_token = lookup['channel_by_token']
    token = by_channel_and_name.get((channel, basename), '')
    if token:
        return str(token)
    candidates = [str(t) for t in by_basename.get(basename, [])]
    if not candidates:
        raise RuntimeError(f'Cannot locate sample_data for image={basename} channel={channel}')
    if len(candidates) == 1:
        return candidates[0]
    same_channel = [tok for tok in candidates if str(channel_by_token.get(tok, '')) == channel]
    if len(same_channel) == 1:
        return same_channel[0]
    raise RuntimeError(f'Ambiguous sample_data for image={basename} channel={channel}, candidates={candidates}')

def _find_ann_token_by_instance_token(nusc: Any, sample_token: str, instance_token: str) -> Optional[str]:
    sample = nusc.get('sample', sample_token)
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if str(ann.get('instance_token', '')) == str(instance_token):
            return str(ann_token)
    return None

def run_target_matching(*, config_path: Path, sequence_yaml: Optional[Path]=None, near_plane_m: float=DEFAULT_NEAR_PLANE_M, verbose: bool=True) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    config_path = _as_path(config_path)
    sequence_yaml_path = resolve_sequence_yaml_from_config(config_path=config_path, sequence_yaml=sequence_yaml)
    config = _load_yaml(config_path)
    sequence = _load_yaml(sequence_yaml_path)
    dataset_cfg = config.get('dataset', {}) if isinstance(config, dict) else {}
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
    dataroot_raw = str(dataset_cfg.get('dataroot', sequence.get('dataroot', ''))).strip()
    if not dataroot_raw:
        raise RuntimeError('Cannot resolve dataroot from config.dataset.dataroot or sequence.dataroot')
    dataroot = _as_path(dataroot_raw)
    image_dataroot_raw = str(dataset_cfg.get('image_dataroot', dataroot_raw)).strip() or dataroot_raw
    image_dataroot = _as_path(image_dataroot_raw)
    image_source_subdir = sequence_yaml_path.stem
    version = str(sequence.get('version', 'v1.0-trainval')).strip() or 'v1.0-trainval'
    frames = sequence.get('frames', [])
    if not isinstance(frames, list) or not frames:
        raise RuntimeError('sequence.frames must be a non-empty list')
    NuScenes = _import_nuscenes()
    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
    lookup = _build_camera_sample_data_lookup(nusc)
    first_frame = frames[0]
    if not isinstance(first_frame, dict):
        raise RuntimeError('sequence.frames[0] must be dict')
    target_channel = _resolve_frame_channel(first_frame, default_channel='CAM_FRONT')
    first_prompt_points_by_channel = _resolve_first_target_prompt_points_by_channel(first_frame, default_channel=target_channel)
    first_image_name = _resolve_frame_image_name(first_frame, target_channel)
    first_sample_data_token = _find_sample_data_token_for_image(lookup, image_name=first_image_name, channel=target_channel)
    first_sample_data = nusc.get('sample_data', first_sample_data_token)
    first_sample_token = str(first_sample_data['sample_token'])
    first_sample = nusc.get('sample', first_sample_token)
    prompt_views: List[Dict[str, Any]] = []
    for prompt_channel, prompt_points in first_prompt_points_by_channel.items():
        prompt_channel = str(prompt_channel).strip()
        if not prompt_channel or not prompt_points:
            continue
        sample_data_token = ''
        if isinstance(first_sample.get('data', None), dict):
            sample_data_token = str(first_sample['data'].get(prompt_channel, '') or '')
        if not sample_data_token:
            try:
                image_name = _resolve_frame_image_name(first_frame, prompt_channel)
                sample_data_token = _find_sample_data_token_for_image(lookup, image_name=image_name, channel=prompt_channel)
            except Exception:
                sample_data_token = ''
        if not sample_data_token:
            continue
        sample_data = nusc.get('sample_data', sample_data_token)
        calibrated_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
        normalized_points = [np.asarray([float(pt[0]), float(pt[1])], dtype=np.float32) for pt in prompt_points if isinstance(pt, (list, tuple)) and len(pt) == 2]
        if not normalized_points:
            continue
        prompt_views.append({'channel': prompt_channel, 'points_xy': normalized_points, 'sample_data': sample_data, 'calibrated_sensor': calibrated_sensor, 'ego_pose': ego_pose})
    if not prompt_views:
        raise RuntimeError('first frame has no usable positive target prompts for any camera channel')
    primary_prompt_view = next((view for view in prompt_views if str(view['channel']) == str(target_channel)), prompt_views[0])
    target_point_xy = np.asarray(primary_prompt_view['points_xy'][0], dtype=np.float32)
    first_calibrated_sensor = primary_prompt_view['calibrated_sensor']
    first_ego_pose = primary_prompt_view['ego_pose']
    ray_origin_global, ray_direction_global = backproject_pixel_to_global_ray(target_point_xy, calibrated_sensor=first_calibrated_sensor, ego_pose=first_ego_pose)
    candidates: List[Dict[str, Any]] = []
    for ann_token in first_sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        category_name = str(ann.get('category_name', ''))
        if not category_name.startswith('vehicle'):
            continue
        prompt_inside_count = 0
        visible_bbox_count = 0
        center_dist_values: List[float] = []
        area_values: List[float] = []
        first_visible_bbox: Optional[List[float]] = None
        for prompt_view in prompt_views:
            bbox = project_annotation_bbox_to_camera(ann=ann, sample_data=prompt_view['sample_data'], calibrated_sensor=prompt_view['calibrated_sensor'], ego_pose=prompt_view['ego_pose'], near_plane_m=near_plane_m)
            if bbox is None:
                continue
            visible_bbox_count += 1
            if first_visible_bbox is None:
                first_visible_bbox = [float(v) for v in bbox]
            center = _bbox_center_xy(bbox)
            area_values.append(float(_bbox_area_xyxy(bbox)))
            for point_xy in prompt_view['points_xy']:
                if _bbox_contains_point(bbox, point_xy):
                    prompt_inside_count += 1
                center_dist_values.append(float(np.linalg.norm(center - point_xy)))
        if visible_bbox_count <= 0 or first_visible_bbox is None:
            continue
        center_world = [float(v) for v in ann['translation']]
        ray_distance_m, depth_along_ray_m = point_to_ray_distance(center_world, ray_origin_global, ray_direction_global)
        if depth_along_ray_m <= float(near_plane_m):
            continue
        candidates.append({'ann_token': str(ann_token), 'instance_token': str(ann.get('instance_token', '')), 'category_name': category_name, 'bbox': first_visible_bbox, 'inside': bool(prompt_inside_count > 0), 'prompt_inside_count': int(prompt_inside_count), 'visible_bbox_count': int(visible_bbox_count), 'center_dist_px': float(min(center_dist_values)) if center_dist_values else float('inf'), 'area_px2': float(max(area_values)) if area_values else 0.0, 'center_world_xyz': center_world, 'ray_distance_m': float(ray_distance_m), 'depth_along_ray_m': float(depth_along_ray_m)})
    if not candidates:
        raise RuntimeError(f'No visible vehicle candidate in first frame channel={target_channel}')
    max_prompt_inside_count = max((int(row.get('prompt_inside_count', 0)) for row in candidates))
    if max_prompt_inside_count <= 0:
        prompt_debug = {str(view['channel']): [[float(pt[0]), float(pt[1])] for pt in view['points_xy']] for view in prompt_views}
        raise RuntimeError(f'No first-frame positive prompt falls inside any visible vehicle 3D bbox projection. prompt_points_by_channel={prompt_debug}')
    selected = min(candidates, key=lambda row: (-int(row.get('prompt_inside_count', 0)), float(row['ray_distance_m']), float(row['depth_along_ray_m']), float(row['center_dist_px']), -float(row['area_px2'])))
    target_instance_token = str(selected['instance_token'])
    frame_records: List[Dict[str, Any]] = []
    matched_frames = 0
    existing_adjust_map = _load_existing_camouflage_adjust_map(sequence_yaml_path)
    for frame_index, frame in enumerate(frames, start=1):
        if not isinstance(frame, dict):
            raise RuntimeError(f'sequence.frames[{frame_index - 1}] must be dict')
        frame_id = int(frame.get('frame_id', frame_index))
        frame_channel = _resolve_frame_channel(frame, default_channel=target_channel)
        image_name = _resolve_frame_image_name(frame, frame_channel)
        manual_prompt_spec_map = _resolve_frame_target_prompt_spec_map(frame, default_channel=frame_channel)
        manual_prompt_map = {str(channel): list(spec.get('positive', [])) for channel, spec in manual_prompt_spec_map.items() if isinstance(spec, dict) and spec.get('positive')}
        manual_negative_prompt_map = {str(channel): list(spec.get('negative', [])) for channel, spec in manual_prompt_spec_map.items() if isinstance(spec, dict) and spec.get('negative')}
        manual_prompt_points_xy = list(manual_prompt_map.get(frame_channel, []))
        manual_negative_points_xy = list(manual_negative_prompt_map.get(frame_channel, []))
        sample_data_token = _find_sample_data_token_for_image(lookup, image_name=image_name, channel=frame_channel)
        sample_data = nusc.get('sample_data', sample_data_token)
        sample_token = str(sample_data['sample_token'])
        sample = nusc.get('sample', sample_token)
        calibrated_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
        ann_token = _find_ann_token_by_instance_token(nusc=nusc, sample_token=sample_token, instance_token=target_instance_token)
        camera_intrinsic = np.asarray(calibrated_sensor['camera_intrinsic'], dtype=np.float32)
        sensor_from_global = sensor_from_global_matrix(ego_pose=ego_pose, calibrated_sensor=calibrated_sensor)
        global_from_sensor = np.linalg.inv(sensor_from_global)
        record: Dict[str, Any] = {'frame_id': frame_id, 'sample_token': sample_token, 'scene_token': str(sample.get('scene_token', '')), 'timestamp': int(sample.get('timestamp', 0)), 'channel': frame_channel, 'image_name': Path(image_name).name, 'sample_data_token': str(sample_data_token), 'target_instance_token': target_instance_token, 'matched': bool(ann_token), 'ann_token': str(ann_token or ''), 'camera': {'sample_data_filename': str(sample_data.get('filename', '')), 'width': int(sample_data.get('width', 0)), 'height': int(sample_data.get('height', 0)), 'camera_intrinsic': camera_intrinsic.tolist(), 'sensor_from_global': sensor_from_global.tolist(), 'global_from_sensor': global_from_sensor.tolist(), 'ego_pose_token': str(sample_data.get('ego_pose_token', '')), 'calibrated_sensor_token': str(sample_data.get('calibrated_sensor_token', ''))}, 'mesh': existing_adjust_map.get(frame_id, _default_camouflage_adjust())}
        if ann_token:
            ann = nusc.get('sample_annotation', ann_token)
            center_xyz = [float(v) for v in ann['translation']]
            size_wlh = [float(v) for v in ann['size']]
            rotation_wxyz = [float(v) for v in ann['rotation']]
            yaw_rad = yaw_from_quaternion(rotation_wxyz)
            bbox_2d = project_annotation_bbox_to_camera(ann=ann, sample_data=sample_data, calibrated_sensor=calibrated_sensor, ego_pose=ego_pose, near_plane_m=near_plane_m)
            center_2d = project_world_point_to_camera(center_xyz, sample_data=sample_data, calibrated_sensor=calibrated_sensor, ego_pose=ego_pose, near_plane_m=near_plane_m)
            if manual_prompt_points_xy:
                center_prompt_xy = [float(manual_prompt_points_xy[0][0]), float(manual_prompt_points_xy[0][1])]
            elif frame_id == int(frames[0].get('frame_id', 1)) and frame_channel == target_channel:
                center_prompt_xy = [float(target_point_xy[0]), float(target_point_xy[1])]
            elif center_2d is not None:
                center_prompt_xy = [float(center_2d[0]), float(center_2d[1])]
            elif bbox_2d is not None:
                center_from_bbox = _bbox_center_xy(bbox_2d)
                center_prompt_xy = [float(center_from_bbox[0]), float(center_from_bbox[1])]
            else:
                center_prompt_xy = None
            record['gt'] = {'category_name': str(ann.get('category_name', '')), 'center_xyz': center_xyz, 'size_wlh': size_wlh, 'rotation_wxyz': rotation_wxyz, 'yaw_rad': float(yaw_rad)}
            record['bbox_2d_xyxy'] = [float(v) for v in bbox_2d] if bbox_2d is not None else None
            record['target_center_xy'] = center_prompt_xy
            record['target_prompt_points_by_channel'] = manual_prompt_map
            record['target_prompt_specs_by_channel'] = manual_prompt_spec_map
            record['target_prompt_points_xy'] = manual_prompt_points_xy
            record['target_negative_prompt_points_xy'] = manual_negative_points_xy
            matched_frames += 1
        else:
            record['gt'] = None
            record['bbox_2d_xyxy'] = None
            record['target_center_xy'] = None
            record['target_prompt_points_by_channel'] = manual_prompt_map
            record['target_prompt_specs_by_channel'] = manual_prompt_spec_map
            record['target_prompt_points_xy'] = manual_prompt_points_xy
            record['target_negative_prompt_points_xy'] = manual_negative_points_xy
        frame_records.append(record)
    bound_sequence = copy.deepcopy(sequence)
    bound_frames = bound_sequence.get('frames', [])
    if not isinstance(bound_frames, list):
        raise RuntimeError('sequence.frames must be list')
    for idx, record in enumerate(frame_records):
        if idx >= len(bound_frames):
            break
        frame = bound_frames[idx]
        if not isinstance(frame, dict):
            continue
        target = frame.get('target', {})
        if not isinstance(target, dict):
            target = {}
        target['channel'] = str(record['channel'])
        target['instance_token'] = str(target_instance_token)
        target['ann_token'] = str(record.get('ann_token', '') or '')
        target['matched'] = bool(record.get('matched', False))
        gt = record.get('gt', None)
        if isinstance(gt, dict):
            target['global_xy'] = [float(gt['center_xyz'][0]), float(gt['center_xyz'][1])]
            target['gt_center_xyz'] = [float(v) for v in gt['center_xyz']]
            target['gt_size_wlh'] = [float(v) for v in gt['size_wlh']]
            target['gt_yaw_rad'] = float(gt['yaw_rad'])
        else:
            target['global_xy'] = None
        target['target_center_xy'] = record.get('target_center_xy', None)
        target['prompts'] = record.get('target_prompt_specs_by_channel', {})
        target['target_prompt_points_xy'] = record.get('target_prompt_points_xy', [])
        target['target_negative_prompt_points_xy'] = record.get('target_negative_prompt_points_xy', [])
        frame['target'] = target
        frame['mesh'] = _normalize_camouflage_adjust(record.get('mesh', None))
    bound_sequence['target_binding'] = {'instance_token': target_instance_token, 'first_frame_channel': target_channel, 'first_frame_point_xy': [float(target_point_xy[0]), float(target_point_xy[1])], 'first_frame_prompt_points_by_channel': {str(channel): [[float(pt[0]), float(pt[1])] for pt in points] for channel, points in first_prompt_points_by_channel.items()}, 'first_frame_selection_method': 'max_positive_prompts_inside_vehicle_3d_bbox_projection', 'first_frame_selected_prompt_inside_count': int(selected.get('prompt_inside_count', 0)), 'first_frame_prompt_count': int(sum((len(points) for points in first_prompt_points_by_channel.values()))), 'first_frame_ray_origin_xyz': [float(v) for v in ray_origin_global.tolist()], 'first_frame_ray_direction_xyz': [float(v) for v in ray_direction_global.tolist()], 'matched_frames': matched_frames, 'total_frames': len(frame_records)}
    binding_payload: Dict[str, Any] = {'version': version, 'config_path': str(config_path), 'sequence_yaml': str(sequence_yaml_path), 'dataroot': str(dataroot), 'image_dataroot': str(image_dataroot), 'image_source_subdir': image_source_subdir, 'target': {'channel': target_channel, 'first_frame_point_xy': [float(target_point_xy[0]), float(target_point_xy[1])], 'first_frame_prompt_points_by_channel': {str(channel): [[float(pt[0]), float(pt[1])] for pt in points] for channel, points in first_prompt_points_by_channel.items()}, 'first_frame_selection_method': 'max_positive_prompts_inside_vehicle_3d_bbox_projection', 'first_frame_ray_origin_xyz': [float(v) for v in ray_origin_global.tolist()], 'first_frame_ray_direction_xyz': [float(v) for v in ray_direction_global.tolist()], 'instance_token': target_instance_token, 'first_frame_image_name': Path(first_image_name).name, 'first_frame_sample_token': first_sample_token, 'first_frame_ann_token': str(selected['ann_token']), 'first_frame_category_name': str(selected['category_name']), 'first_frame_selected_bbox_xyxy': [float(v) for v in selected['bbox']], 'selection_inside': bool(selected['inside']), 'selection_prompt_inside_count': int(selected.get('prompt_inside_count', 0)), 'selection_prompt_count': int(sum((len(points) for points in first_prompt_points_by_channel.values()))), 'selection_visible_bbox_count': int(selected.get('visible_bbox_count', 0)), 'selection_center_dist_px': float(selected['center_dist_px']), 'selection_area_px2': float(selected['area_px2']), 'selection_ray_distance_m': float(selected['ray_distance_m']), 'selection_depth_along_ray_m': float(selected['depth_along_ray_m'])}, 'matched_frame_count': matched_frames, 'total_frame_count': len(frame_records), 'frames': frame_records}
    if verbose:
        print(f"[match_target_car] instance_token={target_instance_token} matched={matched_frames}/{len(frame_records)} first_frame_ann={selected['ann_token']} prompt_inside={int(selected.get('prompt_inside_count', 0))}/{int(sum((len(points) for points in first_prompt_points_by_channel.values())))}")
    return (_to_builtin(binding_payload), _to_builtin(bound_sequence), sequence_yaml_path)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Match target vehicle from first-frame target.prompts / target.2d_target_xy by 3D prompt ray and bind the same instance across sequence frames')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH, help='Path to config.yaml')
    parser.add_argument('--sequence-yaml', type=Path, default=None, help='Optional explicit sequence yaml path')
    parser.add_argument('--output-binding', type=Path, default=None, help='Output yaml for detailed target binding')
    parser.add_argument('--output-bound-sequence', type=Path, default=None, help='Output yaml with bound target fields')
    parser.add_argument('--save-json', action='store_true', help='Also save JSON copies next to yaml outputs')
    parser.add_argument('--inplace-sequence', action='store_true', help='Overwrite original sequence yaml')
    parser.add_argument('--near-plane', type=float, default=DEFAULT_NEAR_PLANE_M, help='Near plane in meters')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config_path = _as_path(args.config)
    binding_payload, bound_sequence, sequence_yaml_path = run_target_matching(config_path=config_path, sequence_yaml=args.sequence_yaml, near_plane_m=float(args.near_plane), verbose=True)
    output_binding_path = _as_path(args.output_binding) if args.output_binding else None
    if args.inplace_sequence:
        output_bound_sequence_path = sequence_yaml_path
    else:
        output_bound_sequence_path = _as_path(args.output_bound_sequence) if args.output_bound_sequence else None
    if output_binding_path is not None:
        _save_yaml(output_binding_path, binding_payload)
        if args.save_json:
            _save_json(output_binding_path.with_suffix('.json'), binding_payload)
        print(output_binding_path)
    if output_bound_sequence_path is not None:
        _save_yaml(output_bound_sequence_path, bound_sequence)
        if args.save_json:
            _save_json(output_bound_sequence_path.with_suffix('.json'), bound_sequence)
        print(output_bound_sequence_path)
if __name__ == '__main__':
    main()
