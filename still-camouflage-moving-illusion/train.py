from __future__ import annotations
import argparse
import copy
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import json
import math
import pickle
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from eot import FullImageAugmentor, SmallEoTAugmentor
from loss import FrameLossInput, cls_loss, directed_difference_sequence, directed_move_loss, ego_plane_direction_unit_xy, first_frame_min_loss, forward_difference_sequence, forward_move_loss, global_query_rank_loss, lateral_difference_sequence, move_loss, progress_loss, query_identity_loss, rigid_loss
try:
    from loss_stp3_new_shift import compute_stp3_new_loss
except ImportError:

    def compute_stp3_new_loss(*args: Any, **kwargs: Any):
        raise RuntimeError('STP3 new-loss is not in this bundle. Add loss_stp3_new_shift.py or use config.model != stp3.')
from match_target_car import _as_path, _build_camera_sample_data_lookup, _find_sample_data_token_for_image, _load_yaml, _save_yaml, quaternion_to_rotation_matrix, run_target_matching
from model import CAMERA_CHANNELS, BevFormerGradientModel, CameraRecord, FixedQueryMatch, FrameRecord, build_gradient_model, selected_model_cfg, selected_model_name
from nusc_gt_to_mesh import _camouflage_body_offset_xyz_from_config, _camouflage_mesh_global_scale_from_config, _camouflage_mesh_scale_xyz_from_config, _infer_obj_to_body_matrix_from_vertices
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / 'config.yaml'
CHINA_TZ = timezone(timedelta(hours=8))
DEFAULT_SHARED_SAM2_CACHE_ROOT = Path('/home/jushuo/Code/zz0.3-SAM2')
EVAL_MAIN_OUTPUT_DIR = PROJECT_ROOT / 'result' / 'eval-main'
DEFAULT_PRECOMPUTE_CACHE_ROOT = PROJECT_ROOT / 'data-pre'

def _mesh_obj_path_from_config(config: Dict[str, Any]) -> Path:
    mesh_cfg = config.get('mesh', {}) if isinstance(config.get('mesh', {}), dict) else {}
    raw = str(mesh_cfg.get('obj_path', '') or '').strip()
    if not raw:
        raise ValueError('Missing in config.yaml: mesh.obj_path')
    return _as_path(raw)

def _sam2_repo_from_config(config: Dict[str, Any]) -> Path:
    sam2_cfg = config.get('sam2', {}) if isinstance(config.get('sam2', {}), dict) else {}
    raw = str(sam2_cfg.get('repo_root', '') or '').strip()
    if not raw:
        raise ValueError('Missing in config.yaml: sam2.repo_root')
    return _as_path(raw)

def _sam2_checkpoint_from_config(config: Dict[str, Any]) -> Path:
    sam2_cfg = config.get('sam2', {}) if isinstance(config.get('sam2', {}), dict) else {}
    raw = str(sam2_cfg.get('checkpoint_path', '') or '').strip()
    if not raw:
        raise ValueError('Missing in config.yaml: sam2.checkpoint_path')
    return _as_path(raw)

def _sam2_shared_cache_root_from_config(config: Dict[str, Any]) -> Path:
    sam2_cfg = config.get('sam2', {}) if isinstance(config.get('sam2', {}), dict) else {}
    raw = str(sam2_cfg.get('shared_cache_root', '') or '').strip()
    if raw:
        return _as_path(raw)
    return DEFAULT_SHARED_SAM2_CACHE_ROOT.resolve()

def _sam2_cache_record_key(record: Dict[str, Any]) -> str:
    sequence_name = str(record.get('sequence_name', '') or '').strip() or 'sequence'
    frame_id = int(record.get('frame_id', -1))
    sample_token = str(record.get('sample_token', '') or '').strip() or 'sample'
    ann_token = str(record.get('ann_token', '') or '').strip() or 'ann'
    channel = str(record.get('channel', '') or '').strip() or 'channel'
    image_name = str(record.get('image_name', '') or '').strip()
    digest = hashlib.md5('||'.join([sequence_name, str(frame_id), sample_token, ann_token, channel, image_name]).encode('utf-8')).hexdigest()[:12]
    safe_channel = ''.join((ch if ch.isalnum() else '_' for ch in channel))
    return f'{sequence_name}_f{frame_id:04d}_{safe_channel}_{sample_token[:8]}_{digest}'

def _sam2_cache_record_dir(cache_root: Path, record: Dict[str, Any]) -> Path:
    return cache_root / _sam2_cache_record_key(record)

def _load_sam2_summary_from_shared_cache(*, mesh_summary: Dict[str, Any], mesh_summary_path: Path, config: Dict[str, Any], config_path: Path, cache_root: Path) -> Optional[Dict[str, Any]]:
    mesh_records = mesh_summary.get('records', []) if isinstance(mesh_summary, dict) else []
    if not isinstance(mesh_records, list) or not mesh_records:
        return None
    out_records: List[Dict[str, Any]] = []
    for mesh_record in mesh_records:
        if not isinstance(mesh_record, dict):
            return None
        cache_dir = _sam2_cache_record_dir(cache_root, mesh_record)
        sam_mask_path = cache_dir / 'sam_mask.png'
        if not sam_mask_path.exists():
            return None
        final_path = cache_dir / 'final.png'
        panel_path = cache_dir / 'panel.png'
        out_records.append({'sequence_name': str(mesh_record.get('sequence_name', '') or ''), 'frame_id': int(mesh_record.get('frame_id', -1)), 'channel': str(mesh_record.get('channel', '') or ''), 'sample_token': str(mesh_record.get('sample_token', '') or ''), 'ann_token': str(mesh_record.get('ann_token', '') or ''), 'mask_source': 'sam2_instance_mask', 'bbox_used_only_as_prompt': True, 'image_path': str(mesh_record.get('image_path', '') or ''), 'raw_camouflage_path': str(mesh_record.get('raw_camouflage_path', '') or ''), 'mesh_mask_path': str(mesh_record.get('mesh_mask_path', '') or ''), 'sam_mask_path': str(sam_mask_path), 'clipped_mask_path': str(sam_mask_path), 'final_path': str(final_path) if final_path.exists() else '', 'panel_path': str(panel_path) if panel_path.exists() else '', 'prompt_xy': None, 'prompt_points_xy': [], 'box_prompt_xyxy': None, 'positive_point_count': 0, 'negative_point_count': 0, 'sam_meta': {}})
    return {'config_path': str(config_path), 'mesh_summary_yaml': str(mesh_summary_path), 'sam2_repo': str(_sam2_repo_from_config(config)), 'sam2_checkpoint': str(_sam2_checkpoint_from_config(config)), 'sam2_config_name': 'shared_cache', 'device': str(config.get('train', {}).get('device', 'cuda')), 'processed_view_count': len(out_records), 'records': out_records, 'shared_cache_root': str(cache_root), 'shared_cache_hit': True}

def _save_sam2_summary_to_shared_cache(*, sam_summary: Dict[str, Any], cache_root: Path) -> None:
    records = sam_summary.get('records', []) if isinstance(sam_summary, dict) else []
    if not isinstance(records, list):
        return
    cache_root.mkdir(parents=True, exist_ok=True)
    for record in records:
        if not isinstance(record, dict):
            continue
        cache_dir = _sam2_cache_record_dir(cache_root, record)
        cache_dir.mkdir(parents=True, exist_ok=True)
        raw_src = str(record.get('sam_mask_path', '') or '').strip()
        if not raw_src:
            continue
        src_path = _as_path(raw_src)
        if not src_path.exists():
            continue
        shutil.copy2(src_path, cache_dir / 'sam_mask.png')

def _near_plane_from_config(config: Dict[str, Any]) -> float:
    render_cfg = config.get('render', {}) if isinstance(config.get('render', {}), dict) else {}
    return float(render_cfg.get('near_plane_m', 0.1))

def _precompute_cache_enabled_from_config(config: Dict[str, Any]) -> bool:
    train_cfg = config.get('train', {}) if isinstance(config.get('train', {}), dict) else {}
    return bool(train_cfg.get('precompute_cache_enabled', True))

def _precompute_cache_root_from_config(config: Dict[str, Any]) -> Path:
    train_cfg = config.get('train', {}) if isinstance(config.get('train', {}), dict) else {}
    raw = str(train_cfg.get('precompute_cache_root', '') or '').strip()
    if raw:
        return _as_path(raw)
    return DEFAULT_PRECOMPUTE_CACHE_ROOT.resolve()

def _sequence_named_cache_base_dir(cache_root: Path, sequence_yaml_path: Path) -> Path:
    cache_root = _as_path(cache_root)
    stem = str(_as_path(sequence_yaml_path).stem).strip() or 'sequence'
    return cache_root / stem

def _file_signature(path: Path) -> Dict[str, Any]:
    path = _as_path(path)
    stat = path.stat()
    return {'path': str(path.resolve()), 'size': int(stat.st_size), 'mtime_ns': int(stat.st_mtime_ns)}

def _precompute_cache_signature_payload(*, config: Dict[str, Any], config_path: Path, sequence_yaml_paths: Sequence[Path]) -> Dict[str, Any]:
    dataset_cfg = config.get('dataset', {}) if isinstance(config.get('dataset', {}), dict) else {}
    return {'version': 2, 'sequence_yamls': [_file_signature(path) for path in sequence_yaml_paths], 'dataset': {'dataroot': str(_as_path(str(dataset_cfg.get('dataroot', '') or '')).resolve()) if str(dataset_cfg.get('dataroot', '') or '').strip() else '', 'image_dataroot': str(_as_path(str(dataset_cfg.get('image_dataroot', '') or '')).resolve()) if str(dataset_cfg.get('image_dataroot', '') or '').strip() else '', 'scenario_root': str(_scenario_root_for_sequence(config=config, config_path=config_path, sequence_yaml_path=sequence_yaml_paths[0]).resolve()) if sequence_yaml_paths else ''}, 'mesh': {'obj': _file_signature(_mesh_obj_path_from_config(config))}, 'render': {'near_plane_m': float(_near_plane_from_config(config))}, 'sam2': {'repo_root': str(_sam2_repo_from_config(config).resolve()), 'checkpoint': _file_signature(_sam2_checkpoint_from_config(config))}}

def _precompute_cache_dir(*, config: Dict[str, Any], config_path: Path, sequence_yaml_paths: Sequence[Path]) -> Tuple[Path, Dict[str, Any]]:
    if len(sequence_yaml_paths) != 1:
        raise ValueError('Persistent preprocess cache requires exactly one sequence yaml per cache directory')
    cache_root = _precompute_cache_root_from_config(config)
    signature_payload = _precompute_cache_signature_payload(config=config, config_path=config_path, sequence_yaml_paths=sequence_yaml_paths)
    digest = hashlib.md5(json.dumps(signature_payload, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:12]
    base_dir = _sequence_named_cache_base_dir(cache_root, sequence_yaml_paths[0])
    return (base_dir / f'cache-{digest}', signature_payload)

def _precompute_cache_artifact_paths(cache_dir: Path) -> Tuple[Path, Path, Path]:
    binding_path = cache_dir / 'target-car-binding.yaml'
    mesh_summary_path = cache_dir / 'mesh_projection' / 'mesh_projection_summary.yaml'
    sam_summary_path = cache_dir / 'sam2' / 'sam2_mask_summary.yaml'
    return (binding_path, mesh_summary_path, sam_summary_path)

def _cache_signatures_match_ignoring_version(cached_signature: Any, expected_signature: Dict[str, Any]) -> bool:
    if not isinstance(cached_signature, dict):
        return False
    cached = copy.deepcopy(cached_signature)
    expected = copy.deepcopy(expected_signature)
    cached.pop('version', None)
    expected.pop('version', None)
    cached.pop('config_path', None)
    expected.pop('config_path', None)
    return cached == expected

def _summary_record_paths_exist(summary: Dict[str, Any], *, keys: Sequence[str]) -> bool:
    records = summary.get('records', []) if isinstance(summary, dict) else []
    if not isinstance(records, list):
        return False
    for record in records:
        if not isinstance(record, dict):
            return False
        for key in keys:
            raw = str(record.get(key, '') or '').strip()
            if raw and (not _as_path(raw).exists()):
                return False
    return True

def _precompute_cache_is_complete(*, binding_path: Path, mesh_summary_path: Path, sam_summary_path: Path) -> bool:
    if not binding_path.exists() or not mesh_summary_path.exists() or (not sam_summary_path.exists()):
        return False
    try:
        binding_payload = _load_yaml(binding_path)
        mesh_summary = _load_yaml(mesh_summary_path)
        sam_summary = _load_yaml(sam_summary_path)
    except Exception:
        return False
    if not isinstance(binding_payload, dict) or not isinstance(binding_payload.get('frames', []), list):
        return False
    if not _summary_record_paths_exist(mesh_summary, keys=('image_path', 'raw_camouflage_path', 'mesh_mask_path', 'overlay_path')):
        return False
    if not _summary_record_paths_exist(sam_summary, keys=('image_path', 'raw_camouflage_path', 'mesh_mask_path', 'sam_mask_path', 'clipped_mask_path')):
        return False
    return True

def _find_compatible_legacy_precompute_cache(*, current_cache_dir: Path, expected_signature: Dict[str, Any]) -> Optional[Path]:
    base_dir = current_cache_dir.parent
    if not base_dir.exists():
        return None
    candidates = sorted((path for path in base_dir.glob('cache-*') if path.is_dir() and path != current_cache_dir), key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for candidate_dir in candidates:
        cache_meta_path = candidate_dir / 'cache_meta.yaml'
        if not cache_meta_path.exists():
            continue
        try:
            cache_meta = _load_yaml(cache_meta_path)
        except Exception:
            continue
        if not isinstance(cache_meta, dict):
            continue
        if not _cache_signatures_match_ignoring_version(cache_meta.get('signature'), expected_signature):
            continue
        binding_path, mesh_summary_path, sam_summary_path = _precompute_cache_artifact_paths(candidate_dir)
        if _precompute_cache_is_complete(binding_path=binding_path, mesh_summary_path=mesh_summary_path, sam_summary_path=sam_summary_path):
            return candidate_dir
    return None

def _merge_nested_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged

def _resolved_loss_cfg_from_config(config: Dict[str, Any], *, model_name: str) -> Tuple[Dict[str, Any], str]:
    raw_loss_cfg = config.get('loss', {})
    if not isinstance(raw_loss_cfg, dict):
        return ({}, 'loss(<invalid>)')
    if 'by_model' in raw_loss_cfg:
        by_model = raw_loss_cfg.get('by_model', {})
        if not isinstance(by_model, dict):
            raise ValueError('loss.by_model must be a dict with keys bevdet / bevdepth / fastbev / stp3')
        common_cfg = {k: v for k, v in raw_loss_cfg.items() if k != 'by_model'}
        model_override = by_model.get(model_name, {})
        if not isinstance(model_override, dict):
            raise ValueError(f'loss.by_model.{model_name} must be a dict')
        resolved = _merge_nested_dict(common_cfg, model_override)
        return (resolved, f'loss.by_model.{model_name}')
    return (copy.deepcopy(raw_loss_cfg), 'loss')

def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

def _round_float_or_none(value: Optional[float], *, digits: int=2) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), int(digits))

def _format_float_or_none(value: Optional[float], *, digits: int=2) -> Optional[str]:
    if value is None:
        return None
    return f'{float(value):.{int(digits)}f}'
_NUSCENES_VEHICLE_DETECTION_NAMES = {'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'motorcycle', 'bicycle'}

def _write_vehicle_only_results_for_visualization(*, results_path: Path, output_path: Path, vehicle_names: Sequence[str]=tuple(sorted(_NUSCENES_VEHICLE_DETECTION_NAMES))) -> Dict[str, Any]:
    payload = json.loads(results_path.read_text(encoding='utf-8'))
    vehicle_set = {str(name) for name in vehicle_names}
    results = payload.get('results', {})
    if not isinstance(results, dict):
        raise ValueError(f'Invalid nuScenes results payload: missing dict results in {results_path}')
    total_before = 0
    total_after = 0
    filtered_results: Dict[str, List[Dict[str, Any]]] = {}
    for sample_token, detections in results.items():
        if not isinstance(detections, list):
            filtered_results[str(sample_token)] = []
            continue
        total_before += len(detections)
        kept = [det for det in detections if isinstance(det, dict) and str(det.get('detection_name', '')) in vehicle_set]
        total_after += len(kept)
        filtered_results[str(sample_token)] = kept
    filtered_payload = dict(payload)
    filtered_payload['results'] = filtered_results
    _save_json(output_path, filtered_payload)
    return {'vehicle_names': sorted(vehicle_set), 'total_before': int(total_before), 'total_after': int(total_after), 'removed': int(total_before - total_after), 'output_path': str(output_path)}

def _build_visual_name_by_sample_token(*, version: str, dataroot: Path, sample_tokens: Sequence[str]) -> Dict[str, str]:
    try:
        from nuscenes.nuscenes import NuScenes
    except Exception:
        return {}
    tokens = [str(token) for token in sample_tokens if str(token)]
    if not tokens:
        return {}
    token_set = set(tokens)
    token_to_name: Dict[str, str] = {}
    nusc = NuScenes(version=str(version), dataroot=str(dataroot), verbose=False)
    for scene_idx, scene in enumerate(getattr(nusc, 'scene', []), start=1):
        frame_idx = 1
        sample_token = str(scene.get('first_sample_token', '') or '')
        while sample_token:
            if sample_token in token_set:
                token_to_name[sample_token] = f'scene_{scene_idx:04d}_frame_{frame_idx:04d}'
                if len(token_to_name) >= len(token_set):
                    return token_to_name
            sample = nusc.get('sample', sample_token)
            sample_token = str(sample.get('next', '') or '')
            frame_idx += 1
    return token_to_name

def _draw_target_confidence_label_on_plot(*, image_path: Path, label_text: str) -> bool:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return False
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, scale, thickness)
    origin_x = 20
    origin_y = 24 + text_h
    pad = 8
    box_left = max(0, origin_x - pad)
    box_top = max(0, origin_y - text_h - pad)
    box_right = min(int(image.shape[1]) - 1, origin_x + text_w + pad)
    box_bottom = min(int(image.shape[0]) - 1, origin_y + baseline + pad)
    cv2.rectangle(image, (box_left, box_top), (box_right, box_bottom), color=(24, 24, 24), thickness=-1)
    cv2.putText(image, label_text, (origin_x, origin_y), font, scale, (0, 255, 255), thickness, lineType=cv2.LINE_AA)
    return bool(cv2.imwrite(str(image_path), image))

def _annotate_target_confidence_on_visuals(*, config: Dict[str, Any], frames: Sequence[FrameRecord], bev_model: BevFormerGradientModel, results_path: Path, visual_dir: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {'ok': False, 'updated': 0, 'total_frames': 0, 'token_mapped': 0, 'with_target_confidence': 0, 'error': ''}
    if not results_path.exists() or not visual_dir.exists():
        summary['error'] = 'missing_results_or_visual_dir'
        return summary
    try:
        payload = json.loads(results_path.read_text(encoding='utf-8'))
    except Exception as exc:
        summary['error'] = f'results_json_parse_failed:{type(exc).__name__}'
        return summary
    dataset_cfg = config.get('dataset', {})
    model_cfg = _official_model_cfg_from_config(config)
    version = str(dataset_cfg.get('version', 'v1.0-trainval')).strip() or 'v1.0-trainval'
    dataroot = _as_path(str(dataset_cfg.get('dataroot', model_cfg.get('data_root', ''))))
    if not dataroot.exists():
        summary['error'] = f'dataroot_not_found:{dataroot}'
        return summary
    sample_tokens = [str(frame.sample_token) for frame in frames]
    token_to_visual = _build_visual_name_by_sample_token(version=version, dataroot=dataroot, sample_tokens=sample_tokens)
    summary['token_mapped'] = int(len(token_to_visual))
    results_map = payload.get('results', {})
    if not isinstance(results_map, dict):
        summary['error'] = 'results_payload_missing_results_dict'
        return summary
    updated = 0
    with_conf = 0
    total_frames = 0
    for frame in frames:
        total_frames += 1
        sample_token = str(frame.sample_token)
        visual_stem = token_to_visual.get(sample_token)
        if not visual_stem:
            continue
        rows_raw = results_map.get(sample_token, [])
        rows = rows_raw if isinstance(rows_raw, list) else []
        target_name = str(bev_model.frame_target_detection_name(frame))
        target_index = _find_target_replacement_index(rows=rows, target_detection_name=target_name, gt_center_world=frame.gt_center_world)
        target_score: Optional[float] = None
        if target_index >= 0 and target_index < len(rows):
            row = rows[target_index]
            if isinstance(row, dict):
                target_score = float(row.get('detection_score', 0.0))
        if target_score is not None:
            label = f'target car conf: {target_score:.3f}'
            with_conf += 1
        else:
            label = 'target car conf: N/A'
        matched_plots = list(visual_dir.glob(f'{visual_stem}_camera*')) + list(visual_dir.glob(f'{visual_stem}_bev*'))
        if not matched_plots:
            continue
        for plot_path in matched_plots:
            if _draw_target_confidence_label_on_plot(image_path=plot_path, label_text=label):
                updated += 1
    summary['ok'] = True
    summary['updated'] = int(updated)
    summary['total_frames'] = int(total_frames)
    summary['with_target_confidence'] = int(with_conf)
    return summary

def _save_config_snapshot(*, config_path: Path, output_dir: Path) -> Path:
    config_path = _as_path(config_path)
    output_dir = _as_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / 'config.yaml'
    shutil.copyfile(config_path, snapshot_path)
    return snapshot_path

def _append_log_line(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as fp:
        fp.write(message.rstrip() + '\n')

def _print_and_append_log_line(log_path: Path, message: str) -> None:
    print(message)
    _append_log_line(log_path, message)

def _log_clean_reference_coverage(*, training_log_path: Path, prefix: str, frames: Sequence[FrameRecord], clean_detection_refs: Dict[str, Dict[str, torch.Tensor]]) -> None:
    frame_list = list(frames)
    expected_count = len(frame_list)
    missing_frames = [f'{frame.sequence_name}/frame-{frame.frame_id}' for frame in frame_list if frame.cache_key not in clean_detection_refs]
    hit_count = expected_count - len(missing_frames)
    if missing_frames:
        preview = ', '.join(missing_frames[:30])
        if len(missing_frames) > 30:
            preview += f', ... and {len(missing_frames) - 30} more frames'
        line = f'{prefix} clean reference missing for {len(missing_frames)}/{expected_count} frames; excluded from clean-reference motion interpretation: {preview}'
    else:
        line = f'{prefix} clean reference covers all {hit_count}/{expected_count} frames; loss will not fall back to GT due to missing clean refs'
    _print_and_append_log_line(training_log_path, line)

def _append_exception_log(log_path: Path, exc: BaseException, *, prefix: str='[train] run_failed') -> None:
    error_type = type(exc).__name__
    error_text = str(exc)
    oom = isinstance(exc, RuntimeError) and 'out of memory' in error_text.lower()
    if oom:
        _append_log_line(log_path, f'{prefix}: CUDA OOM: {error_text}')
    else:
        _append_log_line(log_path, f'{prefix}: {error_type}: {error_text}')
    tb_text = traceback.format_exc().rstrip()
    if tb_text:
        _append_log_line(log_path, '[train] traceback_start')
        for line in tb_text.splitlines():
            _append_log_line(log_path, line)
        _append_log_line(log_path, '[train] traceback_end')

def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower()

def _release_cuda_memory(*, bev_model: Optional[BevFormerGradientModel]=None, optimizer: Optional[torch.optim.Optimizer]=None) -> None:
    if optimizer is not None:
        try:
            optimizer.zero_grad(set_to_none=True)
        except Exception:
            pass
    if bev_model is not None:
        bev_model.last_bbox_tensor = None
        bev_model.last_cls_tensor = None
        if hasattr(bev_model, 'last_head'):
            bev_model.last_head = None
        if hasattr(bev_model, 'last_heatmap_tensor'):
            bev_model.last_heatmap_tensor = None
        if hasattr(bev_model, 'last_heatmap_grad'):
            bev_model.last_heatmap_grad = None
        if hasattr(bev_model, 'last_query_feature_tensor'):
            bev_model.last_query_feature_tensor = None
        bev_model.last_bbox_grad = None
        bev_model.last_cls_grad = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

def _release_training_runtime(*, bev_model: Optional[BevFormerGradientModel]=None, renderer: Optional['FixedUVTextureRenderer']=None, optimizer: Optional[torch.optim.Optimizer]=None, scaler: Optional[torch.amp.GradScaler]=None) -> None:
    _release_cuda_memory(bev_model=bev_model, optimizer=optimizer)
    if optimizer is not None:
        try:
            optimizer.state.clear()
        except Exception:
            pass
    if scaler is not None:
        try:
            scaler_state = scaler.state_dict()
            scaler_state.clear()
        except Exception:
            pass
    if renderer is not None:
        try:
            renderer.release_gpu()
        except Exception:
            pass
    if bev_model is not None:
        try:
            model = getattr(bev_model, 'model', None)
            if model is not None:
                del model
            bev_model.model = None
            bev_model.last_bbox_tensor = None
            bev_model.last_cls_tensor = None
            if hasattr(bev_model, 'last_head'):
                bev_model.last_head = None
            if hasattr(bev_model, 'last_heatmap_tensor'):
                bev_model.last_heatmap_tensor = None
            if hasattr(bev_model, 'last_heatmap_grad'):
                bev_model.last_heatmap_grad = None
            if hasattr(bev_model, 'last_query_feature_tensor'):
                bev_model.last_query_feature_tensor = None
            bev_model.last_bbox_grad = None
            bev_model.last_cls_grad = None
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

def _autocast_context(*, device: torch.device, enabled: bool, amp_dtype: torch.dtype):
    if not enabled or device.type != 'cuda':
        return nullcontext()
    return torch.autocast(device_type='cuda', dtype=amp_dtype)

def _mean_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum((float(v) for v in values)) / float(len(values)))

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _configure_torch_speed_flags(train_cfg: Dict[str, Any]) -> Dict[str, bool]:
    cudnn_benchmark = bool(train_cfg.get('cudnn_benchmark', False))
    allow_tf32 = bool(train_cfg.get('allow_tf32', False))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        if allow_tf32:
            try:
                torch.set_float32_matmul_precision('high')
            except Exception:
                pass
    return {'cudnn_benchmark': cudnn_benchmark, 'allow_tf32': allow_tf32}

def _china_run_stamp() -> str:
    return datetime.now(CHINA_TZ).strftime('%m%d%H%M')

def _timestamped_output_dir(base_dir: Path) -> Path:
    base_dir = _as_path(base_dir)
    run_dir = base_dir / _china_run_stamp()
    if not run_dir.exists():
        return run_dir
    suffix = 1
    while True:
        candidate = base_dir / f'{_china_run_stamp()}_{suffix:02d}'
        if not candidate.exists():
            return candidate
        suffix += 1

def _sample_named_output_dir(base_dir: Path, sequence_yaml_paths: Sequence[Path]) -> Path:
    base_dir = _as_path(base_dir)
    stems = [str(path.stem).strip() for path in sequence_yaml_paths if str(path.stem).strip()]
    if not stems:
        return _timestamped_output_dir(base_dir)
    run_dir = base_dir
    for stem in stems:
        run_dir = run_dir / stem
    if not run_dir.exists():
        return run_dir
    suffix = 1
    while True:
        candidate = run_dir.parent / f'{run_dir.name}_{suffix:02d}'
        if not candidate.exists():
            return candidate
        suffix += 1

def _resolve_new_output_dir(*, config: Dict[str, Any], config_path: Path, train_cfg: Dict[str, Any]) -> Path:
    output_root = _as_path(str(train_cfg.get('output_dir', config_path.parent / 'result' / 'train')))
    naming_mode = str(train_cfg.get('output_dir_naming', 'time') or 'time').strip().lower()
    if naming_mode == 'sample':
        sequence_yaml_paths = _sequence_yaml_paths_from_config(config, config_path)
        return _sample_named_output_dir(output_root, sequence_yaml_paths)
    if naming_mode != 'time':
        raise ValueError(f'Unsupported train.output_dir_naming: {naming_mode}')
    return _timestamped_output_dir(output_root)

def _checkpoint_dir(output_dir: Path) -> Path:
    return output_dir / 'checkpoints'

def _resolve_resume_ckpt_path(resume_ckpt: Optional[Path]) -> Optional[Path]:
    if resume_ckpt is None:
        return None
    candidate = _as_path(resume_ckpt)
    if candidate.is_file():
        return candidate
    if not candidate.exists() or not candidate.is_dir():
        raise FileNotFoundError(f'Checkpoint path not found: {candidate}')
    search_dirs = [candidate]
    nested_checkpoint_dir = candidate / 'checkpoints'
    if nested_checkpoint_dir.exists():
        search_dirs.insert(0, nested_checkpoint_dir)
    for directory in search_dirs:
        step_ckpts = sorted(list(directory.glob('step-*.pt')) + list(directory.glob('step_*.pt')))
        if step_ckpts:
            return step_ckpts[-1].resolve()
        last_path = directory / 'last.pt'
        if last_path.exists():
            return last_path.resolve()
    nested_best = candidate / 'checkpoints' / 'best.pt'
    if nested_best.exists():
        return nested_best.resolve()
    best_path = candidate / 'best.pt'
    if best_path.exists():
        return best_path.resolve()
    raise FileNotFoundError(f'No checkpoint found under directory: {candidate}')

def _save_training_checkpoint(*, checkpoint_dir: Path, filename: str, step: int, output_dir: Path, config_path: Path, optimizer_name: str, texture_param: torch.nn.Parameter, best_snapshot: torch.Tensor, best_loss: float, best_step: int, history: List[Dict[str, Any]], optimizer: Optional[torch.optim.Optimizer], scaler: Optional[torch.cuda.amp.GradScaler], pgd_anchor: Optional[torch.Tensor], train_eval_history: Optional[List[Dict[str, Any]]]=None, val_history: Optional[List[Dict[str, Any]]]=None) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / str(filename)
    payload: Dict[str, Any] = {'step': int(step), 'output_dir': str(output_dir), 'config_path': str(config_path), 'optimizer_name': str(optimizer_name), 'texture_param': texture_param.detach().cpu(), 'best_snapshot': best_snapshot.detach().cpu(), 'best_loss': float(best_loss), 'best_step': int(best_step), 'history': history, 'train_eval_history': [] if train_eval_history is None else train_eval_history, 'val_history': [] if val_history is None else val_history, 'pgd_anchor': None if pgd_anchor is None else pgd_anchor.detach().cpu()}
    if optimizer is not None:
        payload['optimizer_state'] = optimizer.state_dict()
    if scaler is not None:
        payload['scaler_state'] = scaler.state_dict()
    torch.save(payload, checkpoint_path)
    return checkpoint_path

def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)

def _transform_matrix(translation_xyz: Sequence[float], rotation_wxyz: Sequence[float], *, inverse: bool) -> np.ndarray:
    translation = np.asarray(translation_xyz, dtype=np.float32).reshape(3)
    rotation = quaternion_to_rotation_matrix(rotation_wxyz)
    matrix = np.eye(4, dtype=np.float32)
    if inverse:
        inv_rotation = rotation.T
        matrix[:3, :3] = inv_rotation
        matrix[:3, 3] = -inv_rotation @ translation
    else:
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
    return matrix

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

def _sequence_yaml_root_from_config(config: Dict[str, Any], config_path: Path) -> Path:
    dataset_cfg = config.get('dataset', {}) if isinstance(config.get('dataset', {}), dict) else {}
    raw = str(dataset_cfg.get('sequence_yaml_root', '') or '').strip()
    if raw:
        return _as_path(raw if Path(raw).is_absolute() else config_path.parent / raw)
    return (PROJECT_ROOT / 'data-yaml').resolve()

def _scenario_roots_from_config(config: Dict[str, Any], config_path: Path) -> List[Path]:
    dataset_cfg = config.get('dataset', {}) if isinstance(config.get('dataset', {}), dict) else {}
    raw_roots = dataset_cfg.get('scenario_roots', dataset_cfg.get('scenario_root', ''))
    roots = _config_value_to_string_list(raw_roots)
    if not roots:
        roots = [str(PROJECT_ROOT.parent / 'zz0.1-scenario')]
    resolved: List[Path] = []
    seen = set()
    for raw in roots:
        path = Path(raw)
        resolved_path = _as_path(path if path.is_absolute() else config_path.parent / path)
        key = str(resolved_path)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(resolved_path)
    return resolved

def _scenario_root_from_config(config: Dict[str, Any], config_path: Path) -> Path:
    return _scenario_roots_from_config(config, config_path)[0]

def _scenario_root_for_sequence(*, config: Dict[str, Any], config_path: Path, sequence_yaml_path: Path) -> Path:
    roots = _scenario_roots_from_config(config, config_path)
    case_name, sample_name = _sequence_case_and_name(sequence_yaml_path)
    for root in roots:
        if (root / case_name / sample_name).exists():
            return root
    return roots[0]

def _resolve_sequence_yaml_path_item(*, item: str, config: Dict[str, Any], config_path: Path) -> Path:
    raw = str(item).strip()
    if not raw:
        raise ValueError('Empty sequence item is not allowed')
    direct = Path(raw)
    if direct.suffix.lower() == '.yaml':
        return _as_path(direct if direct.is_absolute() else config_path.parent / direct)
    legacy_candidates = [(config_path.parent / 'data-json' / f'{raw}.yaml').resolve(), (PROJECT_ROOT / 'data-json' / f'{raw}.yaml').resolve()]
    for candidate in legacy_candidates:
        if candidate.exists():
            return candidate
    yaml_root = _sequence_yaml_root_from_config(config, config_path)
    if '/' in raw or '\\' in raw:
        normalized = raw.replace('\\', '/')
        candidate = (yaml_root / f'{normalized}.yaml').resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f'Sequence yaml not found for item={raw}: {candidate}')
    matches = sorted(yaml_root.rglob(f'{raw}.yaml'))
    if not matches:
        raise FileNotFoundError(f'Sequence yaml not found under {yaml_root} for item={raw}')
    if len(matches) > 1:
        match_lines = '\n'.join((str(path.resolve()) for path in matches))
        raise RuntimeError(f'Sequence item={raw} matched multiple yaml files:\n{match_lines}')
    return matches[0].resolve()

def _sequence_yaml_paths_from_dataset_keys(config: Dict[str, Any], config_path: Path, *, sequence_yaml_key: str, image_source_key: str, fallback_image_source: str='', allow_empty: bool=False) -> List[Path]:
    dataset_cfg = config.get('dataset', {}) if isinstance(config.get('dataset', {}), dict) else {}
    explicit_items = _config_value_to_string_list(dataset_cfg.get(sequence_yaml_key, ''))
    source_items = _config_value_to_string_list(dataset_cfg.get(image_source_key, fallback_image_source))
    items = explicit_items if explicit_items else source_items
    if not items:
        if allow_empty:
            return []
        raise ValueError('dataset.sequence_yaml / dataset.image_source_subdir must not be empty')
    return [_resolve_sequence_yaml_path_item(item=item, config=config, config_path=config_path) for item in items]

def _sequence_yaml_paths_from_config(config: Dict[str, Any], config_path: Path) -> List[Path]:
    return _sequence_yaml_paths_from_dataset_keys(config, config_path, sequence_yaml_key='sequence_yaml', image_source_key='image_source_subdir', fallback_image_source='samples-2', allow_empty=False)

def _val_sequence_yaml_paths_from_config(config: Dict[str, Any], config_path: Path) -> List[Path]:
    return _sequence_yaml_paths_from_dataset_keys(config, config_path, sequence_yaml_key='val_sequence_yaml', image_source_key='val_image_source_subdir', fallback_image_source='', allow_empty=True)

def _matched_frame_count_for_groups(frame_groups: Sequence[Sequence[FrameRecord]], fixed_queries: Dict[str, FixedQueryMatch]) -> int:
    return sum((1 for group in frame_groups for frame in group if frame.cache_key in fixed_queries and bool(fixed_queries[frame.cache_key].matched)))

def _progress_pair_count_for_groups(frame_groups: Sequence[Sequence[FrameRecord]], fixed_queries: Dict[str, FixedQueryMatch]) -> int:
    return sum((max(0, sum((1 for frame in group if frame.cache_key in fixed_queries and bool(fixed_queries[frame.cache_key].matched))) - 1) for group in frame_groups))

def _select_train_frame_groups_for_step(*, frame_groups: Sequence[Sequence[FrameRecord]], fixed_queries: Dict[str, FixedQueryMatch], batch_mode: str, samples_per_step: int) -> List[Sequence[FrameRecord]]:
    eligible_groups = [group for group in frame_groups if _matched_frame_count_for_groups([group], fixed_queries) > 0]
    if not eligible_groups:
        return list(frame_groups)
    mode = str(batch_mode).strip().lower()
    if mode not in {'subset', 'mini_batch', 'minibatch', 'mini-batch', 'sample'}:
        return list(eligible_groups)
    count = max(1, int(samples_per_step))
    if count >= len(eligible_groups):
        return list(eligible_groups)
    return random.sample(list(eligible_groups), k=count)

def _frame_group_names(frame_groups: Sequence[Sequence[FrameRecord]]) -> List[str]:
    return [str(group[0].sequence_name) for group in frame_groups if group]

def _sequence_group_key_from_paths(sequence_yaml_paths: Sequence[Path]) -> str:
    stems = [path.stem for path in sequence_yaml_paths]
    if not stems:
        return 'sequence-group'
    if len(stems) == 1:
        return stems[0]
    digest = hashlib.md5('::'.join(stems).encode('utf-8')).hexdigest()[:8]
    return f'group-{digest}'

def _image_source_group_from_config(config: Dict[str, Any], config_path: Path) -> str:
    return _sequence_group_key_from_paths(_sequence_yaml_paths_from_config(config, config_path))

def _sequence_yaml_from_config(config: Dict[str, Any], config_path: Path) -> Path:
    paths = _sequence_yaml_paths_from_config(config, config_path)
    if not paths:
        raise ValueError('No sequence yaml resolved from config')
    return paths[0]

def _sequence_data_root_from_config(config: Dict[str, Any]) -> Path:
    dataset_cfg = config.get('dataset', {}) if isinstance(config.get('dataset', {}), dict) else {}
    model_name = selected_model_name(config)
    model_cfg = selected_model_cfg(config)
    data_root = str(model_cfg.get('data_root', dataset_cfg.get('dataroot', '')) or '').strip()
    if not data_root:
        raise ValueError(f'dataset.dataroot / {model_name}.data_root must not be empty')
    return _as_path(data_root)

def _sequence_case_and_name(sequence_yaml_path: Path) -> Tuple[str, str]:
    return (str(sequence_yaml_path.parent.name), str(sequence_yaml_path.stem))

def _resolve_frame_anchor_channel(frame: Dict[str, Any]) -> str:
    images = frame.get('images', {})
    if not isinstance(images, dict) or not images:
        raise RuntimeError('sequence.frames[*].images must be a non-empty dict')
    if 'CAM_FRONT' in images:
        return 'CAM_FRONT'
    return str(sorted(images.keys())[0])

def _resolve_sequence_sample_tokens(*, sequence_yaml_path: Path, sequence_payload: Dict[str, Any], nusc: Any) -> List[str]:
    lookup = _build_camera_sample_data_lookup(nusc)
    tokens: List[str] = []
    for frame in sequence_payload.get('frames', []):
        if not isinstance(frame, dict):
            raise RuntimeError(f'Invalid frame entry in {sequence_yaml_path}')
        channel = _resolve_frame_anchor_channel(frame)
        images = frame.get('images', {})
        image_name = str(images[channel]).strip()
        sample_data_token = _find_sample_data_token_for_image(lookup, image_name=image_name, channel=channel)
        sample_data = nusc.get('sample_data', sample_data_token)
        tokens.append(str(sample_data['sample_token']))
    if not tokens:
        raise RuntimeError(f'No sample tokens resolved from {sequence_yaml_path}')
    return tokens

def _build_sensor2top_info(*, nusc: Any, sensor_token: str, lidar2ego_translation: Sequence[float], lidar2ego_rotation: Sequence[float], ego2global_translation: Sequence[float], ego2global_rotation: Sequence[float], sensor_type: str, data_path_override: Optional[Path]=None) -> Dict[str, Any]:
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(data_path_override.resolve()) if data_path_override is not None else str(_as_path(nusc.get_sample_data_path(sd_rec['token'])))
    sweep = {'data_path': data_path, 'type': str(sensor_type), 'sample_data_token': str(sd_rec['token']), 'sensor2ego_translation': list(cs_record['translation']), 'sensor2ego_rotation': list(cs_record['rotation']), 'ego2global_translation': list(pose_record['translation']), 'ego2global_rotation': list(pose_record['rotation']), 'timestamp': int(sd_rec['timestamp'])}
    l2e_t = np.asarray(lidar2ego_translation, dtype=np.float32).reshape(3)
    l2e_r_mat = quaternion_to_rotation_matrix(lidar2ego_rotation)
    e2g_t = np.asarray(ego2global_translation, dtype=np.float32).reshape(3)
    e2g_r_mat = quaternion_to_rotation_matrix(ego2global_rotation)
    l2e_t_s = np.asarray(sweep['sensor2ego_translation'], dtype=np.float32).reshape(3)
    l2e_r_s_mat = quaternion_to_rotation_matrix(sweep['sensor2ego_rotation'])
    e2g_t_s = np.asarray(sweep['ego2global_translation'], dtype=np.float32).reshape(3)
    e2g_r_s_mat = quaternion_to_rotation_matrix(sweep['ego2global_rotation'])
    inv_e2g_r_t = np.linalg.inv(e2g_r_mat).T
    inv_l2e_r_t = np.linalg.inv(l2e_r_mat).T
    R = l2e_r_s_mat.T @ e2g_r_s_mat.T @ (inv_e2g_r_t @ inv_l2e_r_t)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (inv_e2g_r_t @ inv_l2e_r_t)
    T -= e2g_t @ (inv_e2g_r_t @ inv_l2e_r_t) + l2e_t @ inv_l2e_r_t
    sweep['sensor2lidar_rotation'] = R.T
    sweep['sensor2lidar_translation'] = T
    return sweep

def _scenario_override_path(*, config: Dict[str, Any], config_path: Path, sequence_yaml_path: Path, sensor_name: str, basename: str) -> Optional[Path]:
    case_name, sample_name = _sequence_case_and_name(sequence_yaml_path)
    for scenario_root in _scenario_roots_from_config(config, config_path):
        candidate = (scenario_root / case_name / sample_name / sensor_name / basename).resolve()
        if candidate.exists():
            return candidate
    return None

def _build_sequence_pkl_from_yaml(*, config: Dict[str, Any], config_path: Path, sequence_yaml_path: Path, output_pkl: Path) -> Path:
    sequence_payload = _load_yaml(sequence_yaml_path)
    version = str(sequence_payload.get('version', 'v1.0-trainval')).strip() or 'v1.0-trainval'
    dataroot = _sequence_data_root_from_config(config)
    try:
        from nuscenes.nuscenes import NuScenes
    except Exception as exc:
        raise ImportError('Generating sequence pkl requires nuscenes-devkit') from exc
    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
    sample_tokens = _resolve_sequence_sample_tokens(sequence_yaml_path=sequence_yaml_path, sequence_payload=sequence_payload, nusc=nusc)
    infos: List[Dict[str, Any]] = []
    total = len(sample_tokens)
    for index, sample_token in enumerate(sample_tokens):
        sample = nusc.get('sample', sample_token)
        lidar_token = str(sample['data']['LIDAR_TOP'])
        lidar_sd = nusc.get('sample_data', lidar_token)
        lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
        lidar_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        lidar_basename = Path(str(nusc.get_sample_data_path(lidar_token))).name
        lidar_path_override = _scenario_override_path(config=config, config_path=config_path, sequence_yaml_path=sequence_yaml_path, sensor_name='LIDAR_TOP', basename=lidar_basename)
        can_bus = np.zeros(18, dtype=np.float32)
        can_bus[0:3] = np.asarray(lidar_pose['translation'], dtype=np.float32)
        can_bus[3:7] = np.asarray(lidar_pose['rotation'], dtype=np.float32)
        info: Dict[str, Any] = {'lidar_path': str((lidar_path_override or _as_path(nusc.get_sample_data_path(lidar_token))).resolve()), 'token': str(sample_token), 'prev': '' if index == 0 else str(sample_tokens[index - 1]), 'next': '' if index == total - 1 else str(sample_tokens[index + 1]), 'can_bus': can_bus, 'frame_idx': int(index), 'sweeps': [], 'cams': {}, 'scene_token': str(sample['scene_token']), 'lidar2ego_translation': list(lidar_cs['translation']), 'lidar2ego_rotation': list(lidar_cs['rotation']), 'ego2global_translation': list(lidar_pose['translation']), 'ego2global_rotation': list(lidar_pose['rotation']), 'timestamp': int(sample['timestamp'])}
        for channel in CAMERA_CHANNELS:
            cam_token = str(sample['data'][channel])
            cam_basename = Path(str(nusc.get_sample_data_path(cam_token))).name
            cam_override = _scenario_override_path(config=config, config_path=config_path, sequence_yaml_path=sequence_yaml_path, sensor_name=channel, basename=cam_basename)
            cam_info = _build_sensor2top_info(nusc=nusc, sensor_token=cam_token, lidar2ego_translation=lidar_cs['translation'], lidar2ego_rotation=lidar_cs['rotation'], ego2global_translation=lidar_pose['translation'], ego2global_rotation=lidar_pose['rotation'], sensor_type=channel, data_path_override=cam_override)
            cam_sd = nusc.get('sample_data', cam_token)
            cam_cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
            cam_info['cam_intrinsic'] = cam_cs['camera_intrinsic']
            info['cams'][channel] = cam_info
        infos.append(info)

    def _build_adjacent_info(sample_token: str) -> Dict[str, Any]:
        sample = nusc.get('sample', sample_token)
        lidar_token = str(sample['data']['LIDAR_TOP'])
        lidar_sd = nusc.get('sample_data', lidar_token)
        lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
        lidar_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        can_bus = np.zeros(18, dtype=np.float32)
        can_bus[0:3] = np.asarray(lidar_pose['translation'], dtype=np.float32)
        can_bus[3:7] = np.asarray(lidar_pose['rotation'], dtype=np.float32)
        adjacent_info: Dict[str, Any] = {'lidar_path': str(_as_path(nusc.get_sample_data_path(lidar_token)).resolve()), 'token': str(sample_token), 'prev': None, 'next': None, 'can_bus': can_bus, 'frame_idx': -1, 'sweeps': [], 'cams': {}, 'scene_token': str(sample['scene_token']), 'lidar2ego_translation': list(lidar_cs['translation']), 'lidar2ego_rotation': list(lidar_cs['rotation']), 'ego2global_translation': list(lidar_pose['translation']), 'ego2global_rotation': list(lidar_pose['rotation']), 'timestamp': int(sample['timestamp'])}
        for channel in CAMERA_CHANNELS:
            cam_token = str(sample['data'][channel])
            cam_basename = Path(str(nusc.get_sample_data_path(cam_token))).name
            cam_path = _as_path(nusc.get_sample_data_path(cam_token))
            if not cam_path.exists():
                for fallback_root in (Path('/home/jushuo/Code/zz0.2-nuScenes'), Path('/home/jushuo/Code/nuScenes')):
                    fallback_path = fallback_root / 'samples' / channel / cam_basename
                    if fallback_path.exists():
                        cam_path = fallback_path.resolve()
                        break
            cam_info = _build_sensor2top_info(nusc=nusc, sensor_token=cam_token, lidar2ego_translation=lidar_cs['translation'], lidar2ego_rotation=lidar_cs['rotation'], ego2global_translation=lidar_pose['translation'], ego2global_rotation=lidar_pose['rotation'], sensor_type=channel, data_path_override=cam_path if cam_path.exists() else None)
            cam_sd = nusc.get('sample_data', cam_token)
            cam_cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
            cam_info['cam_intrinsic'] = cam_cs['camera_intrinsic']
            adjacent_info['cams'][channel] = cam_info
        return adjacent_info

    def _collect_adjacent_infos(sample_token: str, direction: str, max_count: int=6) -> Optional[List[Dict[str, Any]]]:
        rows: List[Dict[str, Any]] = []
        cursor = str(nusc.get('sample', sample_token).get(direction, '') or '')
        while cursor and len(rows) < max_count:
            rows.append(_build_adjacent_info(cursor))
            cursor = str(nusc.get('sample', cursor).get(direction, '') or '')
        return rows if rows else None
    for info in infos:
        token = str(info['token'])
        info['prev'] = _collect_adjacent_infos(token, 'prev')
        info['next'] = _collect_adjacent_infos(token, 'next')
    metadata = {'version': version, 'source_yaml': str(sequence_yaml_path), 'sequence_name': str(sequence_yaml_path.stem), 'case_name': str(sequence_yaml_path.parent.name), 'built_by': 'zz3-3D-camouflage'}
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open('wb') as fp:
        pickle.dump({'infos': infos, 'metadata': metadata}, fp)
    return output_pkl.resolve()

def _ensure_single_sequence_pkl(*, config: Dict[str, Any], config_path: Path, sequence_yaml_path: Path) -> Path:

    def _pkl_has_usable_camera_paths(pkl_path: Path) -> bool:
        try:
            with pkl_path.open('rb') as fp:
                payload = pickle.load(fp)
        except Exception:
            return False
        infos = payload.get('infos', []) if isinstance(payload, dict) else payload
        if not isinstance(infos, list) or not infos:
            return False
        check_count = min(len(infos), 3)
        for info in infos[:check_count]:
            if not isinstance(info, dict):
                return False
            cams = info.get('cams', {})
            if not isinstance(cams, dict) or not cams:
                return False
            for cam in cams.values():
                if not isinstance(cam, dict):
                    return False
                data_path = str(cam.get('data_path', '') or '').strip()
                if not data_path:
                    return False
                if not _as_path(data_path).exists():
                    return False
        return True
    data_root = _sequence_data_root_from_config(config)
    output_pkl = (data_root / f'{sequence_yaml_path.stem}.pkl').resolve()
    if output_pkl.exists():
        if _pkl_has_usable_camera_paths(output_pkl):
            return output_pkl
        print(f'[preprocess] stale pkl path; rebuilding: {output_pkl}')
    return _build_sequence_pkl_from_yaml(config=config, config_path=config_path, sequence_yaml_path=sequence_yaml_path, output_pkl=output_pkl)

def _sequence_pkl_from_paths(config: Dict[str, Any], config_path: Path, sequence_yaml_paths: Sequence[Path], *, explicit_pkl_key: str='sequence_pkl') -> Path:
    dataset_cfg = config.get('dataset', {})
    explicit = str(dataset_cfg.get(explicit_pkl_key, '') or '').strip()
    if explicit:
        return _as_path(explicit if Path(explicit).is_absolute() else config_path.parent / explicit)
    single_paths = [_ensure_single_sequence_pkl(config=config, config_path=config_path, sequence_yaml_path=path) for path in sequence_yaml_paths]
    if len(single_paths) == 1:
        return single_paths[0]
    data_root = _sequence_data_root_from_config(config)
    group_key = _sequence_group_key_from_paths(sequence_yaml_paths)
    combined_pkl = (data_root / f'{group_key}.pkl').resolve()
    if combined_pkl.exists():
        try:
            with combined_pkl.open('rb') as fp:
                existing_payload = pickle.load(fp)
            existing_metadata = existing_payload.get('metadata', {}) if isinstance(existing_payload, dict) else {}
            if isinstance(existing_metadata, dict) and str(existing_metadata.get('version', '')).strip():
                return combined_pkl
        except Exception:
            pass
        try:
            combined_pkl.unlink()
        except FileNotFoundError:
            pass
    combined_infos: List[Dict[str, Any]] = []
    metadata_items: List[Dict[str, Any]] = []
    for path in single_paths:
        with path.open('rb') as fp:
            payload = pickle.load(fp)
        infos = payload['infos'] if isinstance(payload, dict) and isinstance(payload.get('infos'), list) else payload
        if not isinstance(infos, list):
            raise RuntimeError(f'Unsupported per-sequence pkl format: {path}')
        combined_infos.extend(copy.deepcopy(infos))
        if isinstance(payload, dict):
            metadata_items.append(copy.deepcopy(payload.get('metadata', {})))
    combined_version = ''
    for item in metadata_items:
        if isinstance(item, dict):
            maybe_version = str(item.get('version', '') or '').strip()
            if maybe_version:
                combined_version = maybe_version
                break
    if not combined_version:
        dataset_cfg = config.get('dataset', {}) if isinstance(config.get('dataset', {}), dict) else {}
        combined_version = str(dataset_cfg.get('version', 'v1.0-trainval')).strip() or 'v1.0-trainval'
    combined_payload = {'infos': combined_infos, 'metadata': {'version': combined_version, 'built_by': 'zz3-3D-camouflage', 'group_key': group_key, 'source_pkls': [str(path) for path in single_paths], 'source_yamls': [str(path) for path in sequence_yaml_paths], 'items': metadata_items}}
    with combined_pkl.open('wb') as fp:
        pickle.dump(combined_payload, fp)
    return combined_pkl

def _sequence_pkl_from_config(config: Dict[str, Any], config_path: Path) -> Path:
    return _sequence_pkl_from_paths(config, config_path, _sequence_yaml_paths_from_config(config, config_path), explicit_pkl_key='sequence_pkl')

def _summary_cache_is_usable(summary_path: Path, *, required_path_keys: Sequence[str]) -> bool:
    if not summary_path.exists():
        return False
    try:
        summary = _load_yaml(summary_path)
    except Exception:
        return False
    records = summary.get('records', []) if isinstance(summary, dict) else []
    if not isinstance(records, list) or not records:
        return False
    for record in records:
        if not isinstance(record, dict):
            return False
        for key in required_path_keys:
            raw_value = str(record.get(key, '') or '').strip()
            if not raw_value:
                return False
            if not _as_path(raw_value).exists():
                return False
    return True

def _load_sequence_infos(sequence_pkl: Path) -> Dict[str, Dict[str, Any]]:
    with sequence_pkl.open('rb') as fp:
        payload = pickle.load(fp)
    if isinstance(payload, dict) and 'infos' in payload:
        infos = payload['infos']
    elif isinstance(payload, list):
        infos = payload
    else:
        raise RuntimeError(f'Unsupported sequence pkl format: {sequence_pkl}')
    out: Dict[str, Dict[str, Any]] = {}
    for info in infos:
        token = str(info['token'])
        out[token] = info
    return out

def _load_sequence_infos_by_sequence(single_sequence_pkl_map: Dict[str, Path]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for sequence_name, pkl_path in single_sequence_pkl_map.items():
        out[str(sequence_name)] = _load_sequence_infos(pkl_path)
    return out

def _camera_record_from_info(channel: str, cam_info: Dict[str, Any]) -> CameraRecord:
    image_path = _as_path(str(cam_info['data_path']))
    intrinsic = np.asarray(cam_info['cam_intrinsic'], dtype=np.float32).reshape(3, 3)
    sensor2ego_translation = cam_info['sensor2ego_translation']
    sensor2ego_rotation = cam_info['sensor2ego_rotation']
    ego2global_translation = cam_info['ego2global_translation']
    ego2global_rotation = cam_info['ego2global_rotation']
    global_from_ego = _transform_matrix(ego2global_translation, ego2global_rotation, inverse=False)
    ego_from_sensor = _transform_matrix(sensor2ego_translation, sensor2ego_rotation, inverse=False)
    global_from_sensor = global_from_ego @ ego_from_sensor
    sensor_from_global = np.linalg.inv(global_from_sensor).astype(np.float32)
    sensor2lidar_rotation = np.asarray(cam_info['sensor2lidar_rotation'], dtype=np.float32).reshape(3, 3)
    sensor2lidar_translation = np.asarray(cam_info['sensor2lidar_translation'], dtype=np.float32).reshape(3)
    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam = np.eye(4, dtype=np.float32)
    lidar2cam[:3, :3] = lidar2cam_r.T
    lidar2cam[3, :3] = -lidar2cam_t
    viewpad = np.eye(4, dtype=np.float32)
    viewpad[:3, :3] = intrinsic
    lidar2img = (viewpad @ lidar2cam.T).astype(np.float32)
    with Image.open(image_path) as img:
        width, height = img.size
    return CameraRecord(channel=channel, image_path=image_path, width=int(width), height=int(height), camera_intrinsic=intrinsic, lidar2img=lidar2img, lidar2cam=lidar2cam.T.astype(np.float32), sensor_from_global=sensor_from_global, sensor2ego_translation=np.asarray(sensor2ego_translation, dtype=np.float32).reshape(3), sensor2ego_rotation=quaternion_to_rotation_matrix(sensor2ego_rotation), ego2global_translation=np.asarray(ego2global_translation, dtype=np.float32).reshape(3), ego2global_rotation=quaternion_to_rotation_matrix(ego2global_rotation))

def _history_camera_record_from_infos(*, channel: str, current_info: Dict[str, Any], adjacent_info: Dict[str, Any]) -> CameraRecord:
    if channel not in current_info['cams']:
        raise RuntimeError(f'current frame missing camera {channel}')
    if channel not in adjacent_info['cams']:
        raise RuntimeError(f'adjacent frame missing camera {channel}')
    current_cam_info = current_info['cams'][channel]
    adjacent_cam_info = adjacent_info['cams'][channel]
    intrinsic = np.asarray(current_cam_info['cam_intrinsic'], dtype=np.float32).reshape(3, 3)
    egocurr2global = _transform_matrix(current_info['ego2global_translation'], current_info['ego2global_rotation'], inverse=False)
    egoadj2global = _transform_matrix(adjacent_info['ego2global_translation'], adjacent_info['ego2global_rotation'], inverse=False)
    lidar2ego = _transform_matrix(current_info['lidar2ego_translation'], current_info['lidar2ego_rotation'], inverse=False)
    lidaradj2lidarcurr = np.linalg.inv(lidar2ego) @ np.linalg.inv(egocurr2global) @ egoadj2global @ lidar2ego
    transformed_sensor2lidar = np.eye(4, dtype=np.float32)
    transformed_sensor2lidar[:3, :3] = np.asarray(current_cam_info['sensor2lidar_rotation'], dtype=np.float32).reshape(3, 3)
    transformed_sensor2lidar[:3, 3] = np.asarray(current_cam_info['sensor2lidar_translation'], dtype=np.float32).reshape(3)
    transformed_sensor2lidar = (lidaradj2lidarcurr @ transformed_sensor2lidar).astype(np.float32)
    sensor2lidar_rotation = transformed_sensor2lidar[:3, :3]
    sensor2lidar_translation = transformed_sensor2lidar[:3, 3]
    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam = np.eye(4, dtype=np.float32)
    lidar2cam[:3, :3] = lidar2cam_r.T
    lidar2cam[3, :3] = -lidar2cam_t
    viewpad = np.eye(4, dtype=np.float32)
    viewpad[:3, :3] = intrinsic
    lidar2img = (viewpad @ lidar2cam.T).astype(np.float32)
    image_path = _as_path(adjacent_cam_info['data_path'])
    if not image_path.exists():
        fallback_path = _as_path(current_cam_info['data_path'])
        if fallback_path.exists():
            image_path = fallback_path
    with Image.open(image_path) as img:
        width, height = img.size
    sensor2ego_translation = adjacent_cam_info['sensor2ego_translation']
    sensor2ego_rotation = adjacent_cam_info['sensor2ego_rotation']
    ego2global_translation = adjacent_info['ego2global_translation']
    ego2global_rotation = adjacent_info['ego2global_rotation']
    global_from_ego = _transform_matrix(ego2global_translation, ego2global_rotation, inverse=False)
    ego_from_sensor = _transform_matrix(sensor2ego_translation, sensor2ego_rotation, inverse=False)
    global_from_sensor = global_from_ego @ ego_from_sensor
    sensor_from_global = np.linalg.inv(global_from_sensor).astype(np.float32)
    return CameraRecord(channel=channel, image_path=image_path, width=int(width), height=int(height), camera_intrinsic=intrinsic, lidar2img=lidar2img, lidar2cam=lidar2cam.T.astype(np.float32), sensor_from_global=sensor_from_global, sensor2ego_translation=np.asarray(sensor2ego_translation, dtype=np.float32).reshape(3), sensor2ego_rotation=quaternion_to_rotation_matrix(sensor2ego_rotation), ego2global_translation=np.asarray(ego2global_translation, dtype=np.float32).reshape(3), ego2global_rotation=quaternion_to_rotation_matrix(ego2global_rotation))

def _merge_stp3_new_loss_config(loss_cfg: Dict[str, Any], *, config: Dict[str, Any]) -> None:
    stp3_sec = config.get('stp3', {}) if isinstance(config.get('stp3', {}), dict) else {}
    nl = stp3_sec.get('new_loss', {})
    if not isinstance(nl, dict):
        nl = {}
    enabled = bool(nl.get('enabled', False))
    loss_cfg['stp3_new_loss_weight'] = float(nl.get('weight', 1.0)) if enabled else 0.0
    loss_cfg['stp3_new_loss_repulsion_weight'] = float(nl.get('repulsion_weight', 1.0))
    loss_cfg['stp3_new_loss_attraction_weight'] = float(nl.get('attraction_weight', 1.0))
    loss_cfg['stp3_new_loss_shift_lateral_m'] = float(nl.get('shift_lateral_m', 1.0))
    loss_cfg['stp3_new_loss_overlap_act_threshold'] = float(nl.get('overlap_act_threshold', 0.5))
    loss_cfg['stp3_new_loss_use_overlap_refinement'] = bool(nl.get('use_overlap_refinement', True))

def _build_frame_records(binding_payload: Dict[str, Any], info_by_sequence_token: Dict[str, Dict[str, Dict[str, Any]]]) -> List[FrameRecord]:
    frames: List[FrameRecord] = []
    target_instance_token = str(binding_payload.get('target', {}).get('instance_token', '') or '')
    inferred_single_sequence: Optional[str] = None
    if len(info_by_sequence_token) == 1:
        inferred_single_sequence = next(iter(info_by_sequence_token.keys()))
    for binding_frame in binding_payload.get('frames', []):
        sample_token = str(binding_frame['sample_token'])
        raw_seq = binding_frame.get('sequence_name', None)
        payload_seq = binding_payload.get('sequence_name', None)
        if raw_seq:
            sequence_name = str(raw_seq).strip()
        elif payload_seq:
            sequence_name = str(payload_seq).strip()
        elif inferred_single_sequence is not None:
            sequence_name = str(inferred_single_sequence)
        else:
            sequence_name = 'sequence'
        if not sequence_name:
            sequence_name = inferred_single_sequence or 'sequence'
        info = info_by_sequence_token.get(sequence_name, {}).get(sample_token)
        if info is None:
            raise KeyError(f'sample_token={sample_token} missing in sequence pkl for sequence={sequence_name}')
        global_from_ego = _transform_matrix(info['ego2global_translation'], info['ego2global_rotation'], inverse=False)
        ego_from_lidar = _transform_matrix(info['lidar2ego_translation'], info['lidar2ego_rotation'], inverse=False)
        global_from_lidar = global_from_ego @ ego_from_lidar
        lidar_from_global = np.linalg.inv(global_from_lidar).astype(np.float32)
        cameras: Dict[str, CameraRecord] = {}
        for channel in CAMERA_CHANNELS:
            if channel not in info['cams']:
                raise RuntimeError(f'{sample_token} missing camera {channel} in sequence pkl')
            cameras[channel] = _camera_record_from_info(channel, info['cams'][channel])
        history_cameras: List[Dict[str, CameraRecord]] = []
        raw_history = info.get('prev', [])
        if isinstance(raw_history, list):
            for adjacent_info in raw_history:
                if not isinstance(adjacent_info, dict):
                    continue
                history_group: Dict[str, CameraRecord] = {}
                for channel in CAMERA_CHANNELS:
                    history_group[channel] = _history_camera_record_from_infos(channel=channel, current_info=info, adjacent_info=adjacent_info)
                history_cameras.append(history_group)
        gt = binding_frame['gt']
        raw_scene_token = str(binding_frame['scene_token'])
        mask_raw = binding_frame.get('stp3_bev_target_mask_path', binding_frame.get('stp3_bev_target_mask', ''))
        stp3_mask_path = str(mask_raw).strip() if mask_raw is not None else ''
        frame = FrameRecord(sequence_name=sequence_name, frame_id=int(binding_frame['frame_id']), sample_token=sample_token, scene_token=f'{sequence_name}::{raw_scene_token}', timestamp=int(binding_frame['timestamp']), can_bus=np.asarray(info['can_bus'], dtype=np.float32).reshape(-1), lidar_from_global=lidar_from_global, global_from_lidar=global_from_lidar.astype(np.float32), lidar_to_ego_rotation=quaternion_to_rotation_matrix(info['lidar2ego_rotation']), lidar_to_ego_translation=np.asarray(info['lidar2ego_translation'], dtype=np.float32).reshape(3), gt_center_world=np.asarray(gt['center_xyz'], dtype=np.float32).reshape(3), gt_size_wlh=np.asarray(gt['size_wlh'], dtype=np.float32).reshape(3), gt_yaw_world=float(gt['yaw_rad']), gt_category_name=str(gt.get('category_name', '') or ''), target_instance_token=str(binding_frame.get('target_instance_token', target_instance_token) or target_instance_token), stp3_bev_target_mask_path=stp3_mask_path, camouflage_body_offset_xyz=_camouflage_body_offset_xyz_from_config(binding_frame), camouflage_mesh_global_scale=float(_camouflage_mesh_global_scale_from_config(binding_frame)), camouflage_mesh_scale_xyz=np.asarray(_camouflage_mesh_scale_xyz_from_config(binding_frame), dtype=np.float32).reshape(3), cameras=cameras, history_cameras=history_cameras)
        frames.append(frame)
    return frames

def _group_frames_by_sequence_name(frames: Sequence[FrameRecord]) -> List[List[FrameRecord]]:
    groups: List[List[FrameRecord]] = []
    current: List[FrameRecord] = []
    current_name: Optional[str] = None
    for frame in frames:
        if current_name is None or frame.sequence_name != current_name:
            if current:
                groups.append(current)
            current = [frame]
            current_name = frame.sequence_name
        else:
            current.append(frame)
    if current:
        groups.append(current)
    return groups

def _image_to_tensor(path: Path, *, device: torch.device) -> torch.Tensor:
    with Image.open(path) as img:
        arr = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor.to(device=device, dtype=torch.float32)

def _save_tensor_image(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chw = tensor.detach().clamp(0.0, 1.0).cpu()
    arr = (chw.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)

def _save_tensor_image_jpg(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chw = tensor.detach().clamp(0.0, 1.0).cpu()
    arr = (chw.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path, format='JPEG', quality=95)

def _save_tensor_image_png(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chw = tensor.detach().clamp(0.0, 1.0).cpu()
    arr = (chw.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path, format='PNG')

@dataclass
class VisibleView:
    sequence_name: str
    frame_id: int
    sample_token: str
    channel: str
    mesh_mask: torch.Tensor
    sam2_mask: torch.Tensor
    fixed_apply_mask: torch.Tensor
    image_path: Path

    @property
    def cache_key(self) -> str:
        return f'{self.sequence_name}::{self.sample_token}'

@dataclass
class ActiveFrameState:
    frame_input: FrameLossInput
    frame_id: int
    sample_token: str
    query_match: FixedQueryMatch
    bbox_tensor: torch.Tensor
    cls_tensor: torch.Tensor
    heatmap_tensor: Optional[torch.Tensor]
    frame_loss_term: torch.Tensor
    move_lateral_term: torch.Tensor
    move_longitudinal_term: torch.Tensor
    first_frame_min_term: torch.Tensor
    rigid_term: torch.Tensor
    cls_term: torch.Tensor

@dataclass
class AuxiliaryModelRuntime:
    model_name: str
    model: Any
    fixed_queries: Dict[str, FixedQueryMatch]
    clean_detection_refs: Dict[str, Dict[str, torch.Tensor]]
    loss_reference_mode: str
    loss_cfg: Dict[str, Any]
    weight: float
    matched_frame_total: int
    final_decode_match: bool
    use_amp: bool
    amp_dtype_name: str

class FixedUVTextureRenderer:

    def __init__(self, *, mesh_obj_path: Path, frames: Sequence[FrameRecord], sam_summary: Dict[str, Any], texture_cfg: Dict[str, Any], device: torch.device, alpha: float, preload_clean_images_to_device: bool) -> None:
        self.device = device
        self.alpha = float(alpha)
        self.preload_clean_images_to_device = bool(preload_clean_images_to_device)
        try:
            from pytorch3d.io import load_obj
            from pytorch3d.renderer import MeshRasterizer, RasterizationSettings, TexturesUV
            from pytorch3d.structures import Meshes
            from pytorch3d.utils import cameras_from_opencv_projection
        except Exception as exc:
            raise ImportError('torch/pytorch3d are required for train.py') from exc
        self.MeshRasterizer = MeshRasterizer
        self.RasterizationSettings = RasterizationSettings
        self.TexturesUV = TexturesUV
        self.Meshes = Meshes
        self.cameras_from_opencv_projection = cameras_from_opencv_projection
        verts, faces, aux = load_obj(str(mesh_obj_path), load_textures=False)
        self.verts_ref = verts.to(device=device, dtype=torch.float32)
        self.faces_idx = faces.verts_idx.to(device=device, dtype=torch.int64)
        self.verts_uvs = aux.verts_uvs.to(device=device, dtype=torch.float32)
        self.faces_uvs = faces.textures_idx.to(device=device, dtype=torch.int64)
        mesh_center = 0.5 * (self.verts_ref.min(dim=0).values + self.verts_ref.max(dim=0).values)
        obj_to_body_np = _infer_obj_to_body_matrix_from_vertices(self.verts_ref.detach().cpu().numpy())
        obj_to_body = torch.as_tensor(obj_to_body_np, dtype=torch.float32, device=device)
        self.verts_body_ref = (self.verts_ref - mesh_center) @ obj_to_body.T
        self.mesh_extent_body = (self.verts_body_ref.max(dim=0).values - self.verts_body_ref.min(dim=0).values).clamp(min=1e-06)
        self.texture_param, self.texture_anchor = self._build_texture_parameter(texture_cfg)
        self.eot_augmentor = SmallEoTAugmentor(texture_cfg.get('eot', {}), device=device)
        self.image_augmentor = FullImageAugmentor(texture_cfg.get('image_augmentation', {}), device=device)
        self.rasterizer_cache: Dict[Tuple[int, int], Any] = {}
        self.frame_by_key = {frame.cache_key: frame for frame in frames}
        self.projection_final_frame_id_by_sequence: Dict[str, int] = {}
        for frame in frames:
            sequence_name = str(frame.sequence_name)
            self.projection_final_frame_id_by_sequence[sequence_name] = max(int(frame.frame_id), self.projection_final_frame_id_by_sequence.get(sequence_name, int(frame.frame_id)))
        self.clean_images_by_key: Dict[Tuple[str, str], torch.Tensor] = {}
        self.visible_views = self._load_visible_views(sam_summary)
        self.projection_view_mode, self.projection_view_missing_policy, self.projection_target_view_deg_by_sample = self._load_projection_view_config(texture_cfg)
        if self.projection_view_mode == 'grouped':
            offset_count = sum((len(offsets) for offsets in self.projection_target_view_deg_by_sample.values()))
            print(f'[projection_view] mode=grouped_fixed_views samples={len(self.projection_target_view_deg_by_sample)} per_frame_angle_slots={offset_count}')

    def _load_projection_view_config(self, texture_cfg: Dict[str, Any]) -> Tuple[str, str, Dict[str, Dict[int, float]]]:
        projection_cfg = texture_cfg.get('projection_view', {})
        if not isinstance(projection_cfg, dict):
            projection_cfg = {}
        mode = str(projection_cfg.get('mode', 'real') or 'real').strip().lower()
        if mode in {'', 'real', 'gt', 'true', 'self', 'original'}:
            return ('real', 'real', {})
        if mode not in {'grouped', 'fixed_group', 'table', 'xlsx'}:
            raise ValueError(f"camouflage.projection_view.mode must be 'real' or 'grouped', got {mode!r}")
        json_raw = str(projection_cfg.get('grouped_view_json', '') or '').strip()
        if json_raw:
            json_path = _as_path(json_raw if Path(json_raw).is_absolute() else PROJECT_ROOT / json_raw)
        else:
            json_path = PROJECT_ROOT / 'data-yaml' / 'final_frame_viewing_angle_groups.json'
        if not json_path.exists():
            raise FileNotFoundError(f'Grouped projection-view JSON not found: {json_path}. Set camouflage.projection_view.grouped_view_json.')
        payload = json.loads(json_path.read_text(encoding='utf-8'))
        samples = payload.get('samples', {}) if isinstance(payload, dict) else {}
        if not isinstance(samples, dict) or not samples:
            raise ValueError(f'Grouped projection-view JSON has no samples map: {json_path}')
        by_sample: Dict[str, Dict[int, float]] = {}
        for sample_name, row in samples.items():
            if not isinstance(row, dict):
                continue
            angle_value = row.get('target_view_group_deg', row.get('target_bin_deg'))
            frame_angle_raw = row.get('frame_projection_target_view_deg_by_offset', {})
            frame_angles: Dict[int, float] = {}
            if isinstance(frame_angle_raw, dict):
                for offset_key, offset_angle in frame_angle_raw.items():
                    if offset_angle is None:
                        continue
                    try:
                        frame_angles[int(offset_key)] = float(offset_angle)
                    except Exception:
                        continue
            if angle_value is not None and 0 not in frame_angles:
                frame_angles[0] = float(angle_value)
            if not frame_angles:
                continue
            by_sample[str(sample_name)] = frame_angles
        if not by_sample:
            raise ValueError(f'Grouped projection-view JSON contains no target view angles: {json_path}')
        missing_policy_raw = str(projection_cfg.get('missing_policy', 'real') or 'real').strip().lower()
        if missing_policy_raw in {'', 'real', 'fallback', 'fallback_real'}:
            missing_policy = 'real'
        elif missing_policy_raw in {'error', 'raise', 'stop'}:
            missing_policy = 'raise'
        else:
            raise ValueError('camouflage.projection_view.missing_policy only supports real / raise')
        return ('grouped', missing_policy, by_sample)

    def _build_texture_parameter(self, texture_cfg: Dict[str, Any]) -> Tuple[torch.nn.Parameter, torch.Tensor]:
        resolution = texture_cfg.get('resolution', [1024, 1024])
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
            raise ValueError('camouflage.resolution must be [width, height]')
        tex_w = int(resolution[0])
        tex_h = int(resolution[1])
        if tex_w <= 1 or tex_h <= 1:
            raise ValueError('camouflage.resolution must be larger than 1x1')
        init_mode = str(texture_cfg.get('init_mode', 'random')).strip().lower()
        if init_mode == 'image':
            image_path = str(texture_cfg.get('init_image', '') or '').strip()
            if not image_path:
                raise ValueError('camouflage.init_mode=image requires init_image')
            tensor = _image_to_tensor(_as_path(image_path), device=self.device).unsqueeze(0)
            tensor = F.interpolate(tensor, size=(tex_h, tex_w), mode='bilinear', align_corners=False)
        elif init_mode == 'constant':
            rgb = texture_cfg.get('init_rgb', [128, 128, 128])
            arr = np.asarray(rgb, dtype=np.float32).reshape(1, 3, 1, 1) / 255.0
            tensor = torch.from_numpy(np.tile(arr, (1, 1, tex_h, tex_w))).to(device=self.device, dtype=torch.float32)
        else:
            tensor = torch.rand((1, 3, tex_h, tex_w), device=self.device, dtype=torch.float32)
        anchor = tensor.detach().clone()
        eps = 0.0001
        logit_init = torch.logit(tensor.clamp(eps, 1.0 - eps))
        return (torch.nn.Parameter(logit_init), anchor)

    def _load_visible_views(self, sam_summary: Dict[str, Any]) -> List[VisibleView]:
        views: List[VisibleView] = []
        for record in sam_summary.get('records', []):
            if not isinstance(record, dict):
                continue
            sequence_name = str(record.get('sequence_name', '') or '')
            sample_token = str(record.get('sample_token', ''))
            channel = str(record.get('channel', ''))
            frame = self.frame_by_key.get(f'{sequence_name}::{sample_token}') if sequence_name else None
            if frame is None:
                frame = next((item for item in self.frame_by_key.values() if item.sample_token == sample_token), None)
            if frame is None or channel not in frame.cameras:
                continue
            mesh_mask_path = _as_path(record['mesh_mask_path'])
            sam_mask_path = _as_path(record['sam_mask_path'])
            clipped_mask_path = _as_path(record['clipped_mask_path'])
            clean_path = _as_path(record['image_path'])
            mesh_mask = cv2.imread(str(mesh_mask_path), cv2.IMREAD_GRAYSCALE)
            sam_mask = cv2.imread(str(sam_mask_path), cv2.IMREAD_GRAYSCALE)
            clipped_mask = cv2.imread(str(clipped_mask_path), cv2.IMREAD_GRAYSCALE)
            if mesh_mask is None or sam_mask is None or clipped_mask is None:
                raise RuntimeError(f'Cannot read mask png for sample={sample_token} channel={channel}')
            clean_image = _image_to_tensor(clean_path, device=self.device if self.preload_clean_images_to_device else torch.device('cpu'))
            self.clean_images_by_key[frame.cache_key, channel] = clean_image
            views.append(VisibleView(sequence_name=str(frame.sequence_name), frame_id=int(record.get('frame_id', -1)), sample_token=sample_token, channel=channel, mesh_mask=torch.from_numpy(mesh_mask > 0), sam2_mask=torch.from_numpy(sam_mask > 0), fixed_apply_mask=torch.from_numpy(sam_mask > 0), image_path=clean_path))
        return views

    def texture_map(self) -> torch.Tensor:
        maps = torch.sigmoid(self.texture_param)
        return maps.permute(0, 2, 3, 1).contiguous()

    def optimizer_parameters(self) -> List[torch.nn.Parameter]:
        return [self.texture_param]

    def clean_image(self, frame: FrameRecord, channel: str) -> torch.Tensor:
        key = (frame.cache_key, channel)
        image = self.clean_images_by_key.get(key)
        if image is None:
            if channel not in frame.cameras:
                raise KeyError(f'Missing clean image for sample={frame.sample_token} channel={channel}')
            image_path = frame.cameras[channel].image_path
            image = _image_to_tensor(image_path, device=self.device if self.preload_clean_images_to_device else torch.device('cpu'))
            self.clean_images_by_key[key] = image
        if image.device == self.device:
            return image
        return image.to(device=self.device, dtype=torch.float32)

    def _rasterizer(self, height: int, width: int):
        key = (height, width)
        rasterizer = self.rasterizer_cache.get(key)
        if rasterizer is None:
            settings = self.RasterizationSettings(image_size=(height, width), blur_radius=0.0, faces_per_pixel=1, perspective_correct=True, bin_size=0, cull_backfaces=True, cull_to_frustum=True)
            rasterizer = self.MeshRasterizer(raster_settings=settings)
            self.rasterizer_cache[key] = rasterizer
        return rasterizer

    def _ego_origin_world(self, frame: FrameRecord) -> torch.Tensor:
        ego_from_lidar = torch.eye(4, dtype=torch.float32, device=self.device)
        ego_from_lidar[:3, :3] = torch.as_tensor(frame.lidar_to_ego_rotation, dtype=torch.float32, device=self.device)
        ego_from_lidar[:3, 3] = torch.as_tensor(frame.lidar_to_ego_translation, dtype=torch.float32, device=self.device)
        global_from_lidar = torch.as_tensor(frame.global_from_lidar, dtype=torch.float32, device=self.device)
        lidar_from_ego = torch.linalg.inv(ego_from_lidar)
        global_from_ego = global_from_lidar @ lidar_from_ego
        return global_from_ego[:3, 3]

    def _yaw_rotation_world(self, yaw: float) -> torch.Tensor:
        cos_yaw = math.cos(float(yaw))
        sin_yaw = math.sin(float(yaw))
        return torch.as_tensor([[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=self.device)

    @staticmethod
    def _wrap_angle_rad(value: float) -> float:
        return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)

    def _projection_yaw_world(self, frame: FrameRecord) -> float:
        gt_yaw = float(frame.gt_yaw_world)
        if self.projection_view_mode != 'grouped':
            return gt_yaw
        sample_name = str(frame.sequence_name)
        target_view_deg_by_offset = self.projection_target_view_deg_by_sample.get(sample_name)
        if target_view_deg_by_offset is None:
            if self.projection_view_missing_policy == 'real':
                return gt_yaw
            raise KeyError(f'Grouped projection table missing sample={sample_name!r}. Add it to grouped_view_json, or set missing_policy=real to fall back to real views.')
        final_frame_id = self.projection_final_frame_id_by_sequence.get(sample_name, int(frame.frame_id))
        frame_offset_to_final = int(frame.frame_id) - int(final_frame_id)
        target_view_deg = target_view_deg_by_offset.get(frame_offset_to_final)
        if target_view_deg is None:
            if self.projection_view_missing_policy == 'real':
                return gt_yaw
            available_offsets = ', '.join((str(k) for k in sorted(target_view_deg_by_offset)))
            raise KeyError(f'Grouped projection table missing per-frame angles for this sample: sample={sample_name!r} frame={int(frame.frame_id)} final_frame={int(final_frame_id)} offset={frame_offset_to_final} available_offsets=[{available_offsets}].')
        ego_origin_world = self._ego_origin_world(frame)
        center_world = torch.as_tensor(frame.gt_center_world, dtype=torch.float32, device=self.device)
        target_to_ego = ego_origin_world - center_world
        target_to_ego_bearing = math.atan2(float(target_to_ego[1].detach().cpu().item()), float(target_to_ego[0].detach().cpu().item()))
        return self._wrap_angle_rad(target_to_ego_bearing - math.radians(float(target_view_deg)))

    def _sample_geometry3d_eot(self, frame: FrameRecord) -> Optional[Dict[str, float]]:
        ego_origin_world = self._ego_origin_world(frame)
        center_world = torch.as_tensor(frame.gt_center_world, dtype=torch.float32, device=self.device)
        distance_m = float(torch.linalg.norm(center_world - ego_origin_world).item())
        return self.eot_augmentor.sample_geometry3d(distance_m=distance_m)

    def _aligned_mesh(self, frame: FrameRecord, *, geometry3d_eot: Optional[Dict[str, float]]=None):
        return self._aligned_mesh_with_maps(frame, maps_hwc=self.texture_map(), geometry3d_eot=geometry3d_eot)

    def _aligned_mesh_with_maps(self, frame: FrameRecord, *, maps_hwc: torch.Tensor, geometry3d_eot: Optional[Dict[str, float]]=None):
        center_xyz_gt = torch.as_tensor(frame.gt_center_world, dtype=torch.float32, device=self.device)
        size_wlh = torch.as_tensor(frame.gt_size_wlh, dtype=torch.float32, device=self.device)
        base_yaw = float(frame.gt_yaw_world)
        yaw = self._projection_yaw_world(frame)
        target_extent = torch.as_tensor([size_wlh[1], size_wlh[0], size_wlh[2]], dtype=torch.float32, device=self.device)
        frame_scale = float(frame.camouflage_mesh_global_scale)
        frame_scale_xyz = torch.as_tensor(frame.camouflage_mesh_scale_xyz, dtype=torch.float32, device=self.device)
        scale_xyz = target_extent / self.mesh_extent_body * frame_scale_xyz * frame_scale
        rot_world = self._yaw_rotation_world(yaw)
        body_offset_xyz = torch.as_tensor(frame.camouflage_body_offset_xyz, device=self.device, dtype=torch.float32)
        center_rot_world = self._yaw_rotation_world(base_yaw)
        center_xyz = center_xyz_gt + center_rot_world @ body_offset_xyz
        if geometry3d_eot:
            yaw = yaw + math.radians(float(geometry3d_eot.get('yaw_deg', 0.0)))
            rot_world = self._yaw_rotation_world(yaw)
            eot_body_offset = torch.as_tensor([float(geometry3d_eot.get('front_m', 0.0)), float(geometry3d_eot.get('left_m', 0.0)), float(geometry3d_eot.get('up_m', 0.0))], dtype=torch.float32, device=self.device)
            center_xyz = center_xyz + rot_world @ eot_body_offset
            ego_origin_world = self._ego_origin_world(frame)
            radial = center_xyz - ego_origin_world
            radial_norm = torch.linalg.norm(radial).clamp(min=1e-06)
            center_xyz = center_xyz + radial / radial_norm * float(geometry3d_eot.get('depth_delta_m', 0.0))
            scale_xyz = scale_xyz * float(geometry3d_eot.get('scale_mul', 1.0))
            scale_xyz = scale_xyz * torch.as_tensor([float(geometry3d_eot.get('scale_length_mul', geometry3d_eot.get('scale_front_mul', 1.0))), float(geometry3d_eot.get('scale_width_mul', geometry3d_eot.get('scale_left_mul', 1.0))), float(geometry3d_eot.get('scale_height_mul', geometry3d_eot.get('scale_up_mul', 1.0)))], dtype=torch.float32, device=self.device)
        verts_world = self.verts_body_ref * scale_xyz @ rot_world.T + center_xyz
        textures = self.TexturesUV(maps=maps_hwc, faces_uvs=self.faces_uvs.unsqueeze(0), verts_uvs=self.verts_uvs.unsqueeze(0))
        return self.Meshes(verts=[verts_world], faces=[self.faces_idx], textures=textures)

    def _render_channel(self, frame: FrameRecord, channel: str, *, geometry3d_eot: Optional[Dict[str, float]]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._render_channel_with_maps(frame, channel, maps_hwc=self.texture_map(), geometry3d_eot=geometry3d_eot)

    def _render_channel_with_maps(self, frame: FrameRecord, channel: str, *, maps_hwc: torch.Tensor, geometry3d_eot: Optional[Dict[str, float]]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        autocast_ctx = torch.autocast(device_type='cuda', enabled=False) if self.device.type == 'cuda' else nullcontext()
        with autocast_ctx:
            cam = frame.cameras[channel]
            mesh = self._aligned_mesh_with_maps(frame, maps_hwc=maps_hwc, geometry3d_eot=geometry3d_eot)
            rasterizer = self._rasterizer(cam.height, cam.width)
            sensor_from_global = torch.as_tensor(cam.sensor_from_global[:3, :3], dtype=torch.float32, device=self.device).unsqueeze(0)
            tvec = torch.as_tensor(cam.sensor_from_global[:3, 3], dtype=torch.float32, device=self.device).unsqueeze(0)
            camera_matrix = torch.as_tensor(cam.camera_intrinsic, dtype=torch.float32, device=self.device).unsqueeze(0)
            image_size = torch.as_tensor([[float(cam.height), float(cam.width)]], dtype=torch.float32, device=self.device)
            cameras = self.cameras_from_opencv_projection(R=sensor_from_global, tvec=tvec, camera_matrix=camera_matrix, image_size=image_size)
            fragments = rasterizer(mesh, cameras=cameras)
            visible_mask = fragments.pix_to_face[0, :, :, 0] >= 0
            sampled = mesh.sample_textures(fragments)[0, :, :, 0, :]
            rendered = sampled.permute(2, 0, 1).contiguous()
            return (rendered, visible_mask)

    def render_clean_probe_view(self, frame: FrameRecord, channel: str, *, probe_maps: torch.Tensor, fixed_mask: Optional[torch.Tensor], geometry3d_eot: Optional[Dict[str, float]]=None) -> torch.Tensor:
        clean = self.clean_image(frame, channel)
        maps_hwc = probe_maps.permute(0, 2, 3, 1).contiguous()
        rendered, visible_mask = self._render_channel_with_maps(frame, channel, maps_hwc=maps_hwc, geometry3d_eot=geometry3d_eot)
        rendered = rendered * visible_mask.to(dtype=rendered.dtype).unsqueeze(0)
        if fixed_mask is None:
            apply_mask = visible_mask
        else:
            if fixed_mask.device != self.device:
                fixed_mask = fixed_mask.to(device=self.device)
            apply_mask = visible_mask & fixed_mask
        if not bool(apply_mask.any()):
            return clean
        mask = apply_mask.to(dtype=clean.dtype).unsqueeze(0)
        return clean + self.alpha * rendered * mask

    def build_clean_probe_images(self, frame: FrameRecord, *, probe_maps: torch.Tensor) -> Dict[str, torch.Tensor]:
        visible_masks = {(view.cache_key, view.channel): view.fixed_apply_mask for view in self.visible_views}
        images: Dict[str, torch.Tensor] = {}
        for channel in CAMERA_CHANNELS:
            fixed_mask = visible_masks.get((frame.cache_key, channel))
            images[channel] = self.render_clean_probe_view(frame, channel, probe_maps=probe_maps, fixed_mask=fixed_mask)
        return images

    def render_patched_view(self, frame: FrameRecord, channel: str, *, fixed_mask: Optional[torch.Tensor], apply_eot: bool=False, geometry3d_eot: Optional[Dict[str, float]]=None) -> torch.Tensor:
        clean = self.clean_image(frame, channel)
        rendered, visible_mask = self._render_channel(frame, channel, geometry3d_eot=geometry3d_eot)
        rendered = rendered * visible_mask.to(dtype=rendered.dtype).unsqueeze(0)
        effective_mask = visible_mask
        if apply_eot and self.eot_augmentor.enabled:
            rendered, effective_mask = self.eot_augmentor.apply(rendered, effective_mask)
        if fixed_mask is None:
            apply_mask = effective_mask
            foreground_mask_for_aug = apply_mask
        else:
            if fixed_mask.device != self.device:
                fixed_mask = fixed_mask.to(device=self.device)
            apply_mask = effective_mask & fixed_mask
            foreground_mask_for_aug = fixed_mask
        if not bool(apply_mask.any()):
            return clean.clamp(0.0, 1.0)
        rendered = rendered * apply_mask.to(dtype=rendered.dtype).unsqueeze(0)
        mask = apply_mask.to(dtype=clean.dtype).unsqueeze(0)
        blended = (1.0 - self.alpha) * clean + self.alpha * rendered
        patched = clean * (1.0 - mask) + blended * mask
        if apply_eot and self.image_augmentor.enabled:
            patched = self.image_augmentor.apply(patched, foreground_mask=foreground_mask_for_aug)
        return patched.clamp(0.0, 1.0)

    def build_frame_images(self, frame: FrameRecord, *, apply_eot: bool=False) -> Dict[str, torch.Tensor]:
        visible_masks = {(view.cache_key, view.channel): view.fixed_apply_mask for view in self.visible_views}
        images: Dict[str, torch.Tensor] = {}
        geometry3d_eot = self._sample_geometry3d_eot(frame) if apply_eot and self.eot_augmentor.enabled else None
        for channel in CAMERA_CHANNELS:
            fixed_mask = visible_masks.get((frame.cache_key, channel))
            if fixed_mask is None:
                clean = self.clean_image(frame, channel)
                if apply_eot and self.image_augmentor.enabled and self.image_augmentor.apply_to_clean_views:
                    clean = self.image_augmentor.apply(clean, foreground_mask=None)
                images[channel] = clean
            else:
                images[channel] = self.render_patched_view(frame, channel, fixed_mask=fixed_mask, apply_eot=apply_eot, geometry3d_eot=geometry3d_eot)
        return images

    def export_visuals(self, *, output_dir: Path, frames: Sequence[FrameRecord], tag: str) -> None:
        output_dir = _as_path(output_dir)
        mesh_dir = output_dir / 'mesh_mask'
        sam_dir = output_dir / 'sam2_mask'
        final_dir = output_dir / 'patched'
        texture_path = output_dir / f'texture_{tag}.png'
        _save_tensor_image(texture_path, torch.sigmoid(self.texture_param).detach().squeeze(0))
        by_key = {(view.cache_key, view.channel): view for view in self.visible_views}
        for frame in frames:
            frame_images = self.build_frame_images(frame)
            for channel in CAMERA_CHANNELS:
                view = by_key.get((frame.cache_key, channel))
                if view is None:
                    continue
                stem = f'frame_{frame.frame_id:04d}_{channel}_{view.image_path.stem}'
                mesh_mask_u8 = view.mesh_mask.detach().cpu().numpy().astype(np.uint8) * 255
                sam_mask_u8 = view.sam2_mask.detach().cpu().numpy().astype(np.uint8) * 255
                mesh_dir.mkdir(parents=True, exist_ok=True)
                sam_dir.mkdir(parents=True, exist_ok=True)
                final_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mesh_dir / f'{stem}.png'), mesh_mask_u8)
                cv2.imwrite(str(sam_dir / f'{stem}.png'), sam_mask_u8)
                _save_tensor_image(final_dir / f'{stem}.png', frame_images[channel])

    def release_gpu(self) -> None:
        if self.texture_param is not None and isinstance(self.texture_param, torch.nn.Parameter):
            self.texture_param.grad = None
        self.rasterizer_cache.clear()
        self.clean_images_by_key.clear()
        self.visible_views = []
        self.frame_by_key = {}
        self.verts_ref = torch.empty((0, 3), dtype=torch.float32, device='cpu')
        self.faces_idx = torch.empty((0, 3), dtype=torch.int64, device='cpu')
        self.verts_uvs = torch.empty((0, 2), dtype=torch.float32, device='cpu')
        self.faces_uvs = torch.empty((0, 3), dtype=torch.int64, device='cpu')
        self.verts_body_ref = torch.empty((0, 3), dtype=torch.float32, device='cpu')
        self.mesh_extent_body = torch.empty((3,), dtype=torch.float32, device='cpu')
        if isinstance(self.texture_anchor, torch.Tensor):
            self.texture_anchor = self.texture_anchor.detach().cpu()
        if isinstance(self.texture_param, torch.nn.Parameter):
            detached = self.texture_param.detach().cpu()
            self.texture_param = torch.nn.Parameter(detached, requires_grad=False)

def _prepare_precompute_outputs(*, config: Dict[str, Any], config_path: Path, output_root: Path, sequence_yaml_paths: Optional[Sequence[Path]]=None, stage_label: str='train') -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:

    def _combine_binding_payloads(payloads: Sequence[Tuple[Path, Dict[str, Any]]]) -> Dict[str, Any]:
        if not payloads:
            raise RuntimeError('No binding payloads to combine')
        first_yaml, first_payload = payloads[0]
        combined: Dict[str, Any] = {'version': str(first_payload.get('version', 'v1.0-trainval')), 'dataroot': str(first_payload.get('dataroot', config.get('dataset', {}).get('dataroot', ''))), 'image_dataroot': str(first_payload.get('image_dataroot', config.get('dataset', {}).get('image_dataroot', ''))), 'image_source_subdir': _sequence_group_key_from_paths([path for path, _ in payloads]), 'source_sequence_yamls': [str(path) for path, _ in payloads], 'sequence_name': _sequence_group_key_from_paths([path for path, _ in payloads]), 'target': {'instance_token': 'MULTI_SEQUENCE', 'sequence_count': len(payloads)}, 'frames': []}
        for sequence_yaml_path, payload in payloads:
            sequence_name = str(sequence_yaml_path.stem)
            for frame in payload.get('frames', []):
                if not isinstance(frame, dict):
                    continue
                row = copy.deepcopy(frame)
                row['sequence_name'] = sequence_name
                combined['frames'].append(row)
        return combined

    def _combine_summary_payloads(*, summaries: Sequence[Tuple[Path, Dict[str, Any]]], count_key: str) -> Dict[str, Any]:
        if not summaries:
            raise RuntimeError('No per-sequence summaries to combine')
        first_yaml, first_summary = summaries[0]
        combined: Dict[str, Any] = copy.deepcopy(first_summary)
        combined['config_path'] = str(config_path)
        combined['source_sequence_yamls'] = [str(path) for path, _ in summaries]
        combined['sequence_name'] = _sequence_group_key_from_paths([path for path, _ in summaries])
        combined['image_source_subdir'] = _sequence_group_key_from_paths([path for path, _ in summaries])
        combined['records'] = []
        combined[count_key] = 0
        for sequence_yaml_path, summary in summaries:
            sequence_name = str(sequence_yaml_path.stem)
            records = summary.get('records', []) if isinstance(summary, dict) else []
            if not isinstance(records, list):
                continue
            for row in records:
                if not isinstance(row, dict):
                    continue
                merged = copy.deepcopy(row)
                merged['sequence_name'] = sequence_name
                combined['records'].append(merged)
            combined[count_key] = int(combined.get(count_key, 0)) + int(summary.get(count_key, 0))
        return combined

    def _run_stage_and_read_summary(cmd: Sequence[str], expected_summary: Path) -> Dict[str, Any]:
        proc = subprocess.run(list(cmd), cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError('Preprocess failed: \n' + ' '.join((str(part) for part in cmd)) + '\n' + (proc.stdout or '').strip())
        if not expected_summary.exists():
            raise FileNotFoundError(f"Summary file not found after preprocess command: {expected_summary}\n{(proc.stdout or '').strip()}")
        return _load_yaml(expected_summary)

    def _prepare_single_sequence_precompute(sequence_yaml_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        use_persistent_cache = _precompute_cache_enabled_from_config(config)
        if use_persistent_cache:
            cache_dir, cache_signature = _precompute_cache_dir(config=config, config_path=config_path, sequence_yaml_paths=[sequence_yaml_path])
            binding_path, mesh_summary_path, sam_summary_path = _precompute_cache_artifact_paths(cache_dir)
            mesh_output_dir = mesh_summary_path.parent
            sam_output_dir = sam_summary_path.parent
            cache_meta_path = cache_dir / 'cache_meta.yaml'
            if _precompute_cache_is_complete(binding_path=binding_path, mesh_summary_path=mesh_summary_path, sam_summary_path=sam_summary_path):
                print(f'[preprocess] using persistent cache: {cache_dir}')
                return (_load_yaml(binding_path), _load_yaml(mesh_summary_path), _load_yaml(sam_summary_path))
            compatible_cache_dir = _find_compatible_legacy_precompute_cache(current_cache_dir=cache_dir, expected_signature=cache_signature)
            if compatible_cache_dir is not None:
                binding_path, mesh_summary_path, sam_summary_path = _precompute_cache_artifact_paths(compatible_cache_dir)
                print(f'[preprocess] reusing legacy-compatible persistent cache: {compatible_cache_dir} current_key={cache_dir.name}')
                return (_load_yaml(binding_path), _load_yaml(mesh_summary_path), _load_yaml(sam_summary_path))
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            _save_yaml(cache_meta_path, {'cache_dir': str(cache_dir), 'sequence_yaml': str(sequence_yaml_path), 'signature': cache_signature})
        else:
            cache_dir = output_root / f'.precompute_tmp_{stage_label}' / str(sequence_yaml_path.stem)
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            binding_path = cache_dir / 'target-car-binding.yaml'
            mesh_output_dir = cache_dir / 'mesh_projection'
            mesh_summary_path = mesh_output_dir / 'mesh_projection_summary.yaml'
            sam_output_dir = cache_dir / 'sam2'
            sam_summary_path = sam_output_dir / 'sam2_mask_summary.yaml'
        binding_payload, _, _ = run_target_matching(config_path=config_path, sequence_yaml=sequence_yaml_path, near_plane_m=0.1, verbose=True)
        _save_yaml(binding_path, binding_payload)
        mesh_obj_path = _mesh_obj_path_from_config(config)
        train_cfg = config.get('train', {}) if isinstance(config.get('train', {}), dict) else {}
        precompute_python_raw = str(train_cfg.get('precompute_python', '') or '').strip()
        precompute_python = str(_as_path(precompute_python_raw)) if precompute_python_raw else sys.executable
        mesh_summary = _run_stage_and_read_summary([precompute_python, str(PROJECT_ROOT / 'nusc_gt_to_mesh.py'), '--config', str(config_path), '--mesh-obj', str(mesh_obj_path), '--binding-yaml', str(binding_path), '--output-dir', str(mesh_output_dir), '--device', str(config['train'].get('device', 'cuda')), '--near-plane', str(float(config.get('render', {}).get('near_plane_m', 0.1)))], mesh_summary_path)
        if use_persistent_cache:
            print(f'[preprocess] building persistent cache: {cache_dir}')
        else:
            print(f'[preprocess] building temporary preprocess dir: {cache_dir}')
        sam_summary = _run_stage_and_read_summary([precompute_python, str(PROJECT_ROOT / 'sam2_mask.py'), '--config', str(config_path), '--mesh-summary-yaml', str(mesh_summary_path), '--output-dir', str(sam_output_dir), '--sam2-repo', str(_sam2_repo_from_config(config)), '--sam2-checkpoint', str(_sam2_checkpoint_from_config(config)), '--device', str(config['train'].get('device', 'cuda')), '--near-plane', str(_near_plane_from_config(config))], sam_summary_path)
        return (binding_payload, mesh_summary, sam_summary)
    resolved_sequence_yaml_paths = list(sequence_yaml_paths or _sequence_yaml_paths_from_config(config, config_path))
    binding_summaries: List[Tuple[Path, Dict[str, Any]]] = []
    mesh_summaries: List[Tuple[Path, Dict[str, Any]]] = []
    sam_summaries: List[Tuple[Path, Dict[str, Any]]] = []
    for sequence_yaml_path in resolved_sequence_yaml_paths:
        binding_payload, mesh_summary, sam_summary = _prepare_single_sequence_precompute(sequence_yaml_path)
        binding_summaries.append((sequence_yaml_path, binding_payload))
        mesh_summaries.append((sequence_yaml_path, mesh_summary))
        sam_summaries.append((sequence_yaml_path, sam_summary))
    return (_combine_binding_payloads(binding_summaries), _combine_summary_payloads(summaries=mesh_summaries, count_key='rendered_view_count'), _combine_summary_payloads(summaries=sam_summaries, count_key='processed_view_count'))

def _train_config_to_loss_cfg(raw_loss_cfg: Dict[str, Any]) -> Dict[str, Any]:
    loss_cfg = raw_loss_cfg if isinstance(raw_loss_cfg, dict) else {}
    move_cfg = loss_cfg.get('move', {}) if isinstance(loss_cfg.get('move', {}), dict) else {}
    rigid_cfg = loss_cfg.get('rigid', {}) if isinstance(loss_cfg.get('rigid', {}), dict) else {}
    progress_cfg = loss_cfg.get('progress', {}) if isinstance(loss_cfg.get('progress', {}), dict) else {}
    cls_cfg = loss_cfg.get('cls', {}) if isinstance(loss_cfg.get('cls', {}), dict) else {}
    depth_cfg = loss_cfg.get('depth', {}) if isinstance(loss_cfg.get('depth', {}), dict) else {}
    bevdet_cfg = loss_cfg.get('bevdet', {}) if isinstance(loss_cfg.get('bevdet', {}), dict) else {}
    dir_cfg = move_cfg.get('direction', {})
    if not isinstance(dir_cfg, dict):
        dir_cfg = {}
    move_fb = float(dir_cfg.get('front_back_pct', dir_cfg.get('fb', dir_cfg.get('forward', 100.0))))
    move_lr = float(dir_cfg.get('left_right_pct', dir_cfg.get('lr', dir_cfg.get('left', 0.0))))
    progress_step_m = float(progress_cfg.get('step_size_m', progress_cfg.get('stepsize_m', 0.05)))
    return {'progress_weight': float(progress_cfg.get('weight', 1.0)), 'progress_lambda': float(progress_cfg.get('lambda', 1.0)), 'progress_detach_previous': bool(progress_cfg.get('detach_previous', False)), 'progress_step_size_m': progress_step_m, 'progress_step_loss_type': str(progress_cfg.get('step_loss_type', 'l2')), 'first_frame_min_weight': float(progress_cfg.get('first_frame_min', 0.0)), 'first_frame_min_m': float(progress_cfg.get('first_frame_min_m', 0.0)), 'per_frame_min_weight': float(progress_cfg.get('per_frame_min', 0.0)), 'per_frame_min_m': float(progress_cfg.get('per_frame_min_m', 0.0)), 'move_center_weight': float(move_cfg.get('weight', 1.0)), 'move_axis': str(move_cfg.get('axis', 'lateral_y')), 'move_direction_fb_pct': move_fb, 'move_direction_lr_pct': move_lr, 'move_loss_type': str(move_cfg.get('loss_type', 'smooth_l1')), 'rigid_weight': float(rigid_cfg.get('weight', 1.0)), 'rigid_loss_type': str(rigid_cfg.get('loss_type', 'l2')), 'rigid_size_weight': float(rigid_cfg.get('size_weight', 1.0)), 'rigid_yaw_weight': float(rigid_cfg.get('yaw_weight', 1.0)), 'cls_weight': float(cls_cfg.get('weight', 1.0)), 'cls_pos_weight': float(cls_cfg.get('pos_weight', 1.0)), 'cls_neg_weight': float(cls_cfg.get('neg_weight', 1.0)), 'cls_rank_weight': float(cls_cfg.get('rank_weight', 1.0)), 'cls_rank_margin': float(cls_cfg.get('rank_margin', 0.0)), 'cls_global_rank_weight': float(cls_cfg.get('global_rank_weight', 0.0)), 'cls_global_rank_margin': float(cls_cfg.get('global_rank_margin', 0.0)), 'query_identity_weight': float(cls_cfg.get('query_identity_weight', 0.0)), 'query_identity_loss_type': str(cls_cfg.get('query_identity_loss_type', 'cosine')), 'depth_weight': float(depth_cfg.get('weight', 0.0)), 'depth_direction': str(depth_cfg.get('direction', 'far')), 'depth_offset_m': float(depth_cfg.get('offset_m', 2.0)), 'depth_patch_radius': int(depth_cfg.get('patch_radius', 1)), 'depth_loss_type': str(depth_cfg.get('loss_type', 'l1')), 'depth_min_valid_cams': int(depth_cfg.get('min_valid_cams', 1)), 'bevdet_shift_m': float(bevdet_cfg.get('shift_m', loss_cfg.get('shift_m', 1.0))), 'bevdet_old_threshold': float(bevdet_cfg.get('old_threshold', loss_cfg.get('old_threshold', 0.0))), 'bevdet_old_weight': float(bevdet_cfg.get('old_weight', loss_cfg.get('old_weight', 0.5))), 'bevdet_reg_weight': float(bevdet_cfg.get('reg_weight', loss_cfg.get('reg_weight', 0.1))), 'enable_query_losses': True, 'style_weight': 0.0}

def _merge_optimal_into_loss_cfg(loss_cfg: Dict[str, Any], *, config: Dict[str, Any]) -> Dict[str, Any]:
    optimal = config.get('optimal')
    if not isinstance(optimal, dict) or not optimal:
        return loss_cfg
    merged = dict(loss_cfg)
    if 'step_size_m' in optimal or 'stepsize_m' in optimal:
        merged['progress_step_size_m'] = float(optimal.get('step_size_m', optimal.get('stepsize_m', merged.get('progress_step_size_m', 0.05))))
    if 'step_loss_type' in optimal:
        merged['progress_step_loss_type'] = str(optimal['step_loss_type'])
    dir_opt = optimal.get('direction')
    if isinstance(dir_opt, dict) and dir_opt:
        merged['move_direction_fb_pct'] = float(dir_opt.get('front_back_pct', dir_opt.get('fb', merged.get('move_direction_fb_pct', 100.0))))
        merged['move_direction_lr_pct'] = float(dir_opt.get('left_right_pct', dir_opt.get('lr', merged.get('move_direction_lr_pct', 0.0))))
    if 'axis' in optimal:
        merged['move_axis'] = str(optimal['axis']).strip().lower()
    return merged

def _use_directed_move_axis(loss_cfg: Dict[str, Any]) -> bool:
    axis = str(loss_cfg.get('move_axis', 'lateral_y')).strip().lower()
    return axis in {'directed', 'direction', 'custom', 'vec', 'vector'}

def _use_forward_move_axis(loss_cfg: Dict[str, Any]) -> bool:
    axis = str(loss_cfg.get('move_axis', 'lateral_y')).strip().lower()
    return axis in {'forward', 'front', 'forward_x', 'front_x', 'x', '+x'}

def _movement_difference_sequence(frame_inputs: Sequence[FrameLossInput], loss_cfg: Dict[str, Any]) -> List[torch.Tensor]:
    if _use_directed_move_axis(loss_cfg):
        ref0 = frame_inputs[0].pred_center_ego
        u = ego_plane_direction_unit_xy(front_back_pct=float(loss_cfg.get('move_direction_fb_pct', 100.0)), left_right_pct=float(loss_cfg.get('move_direction_lr_pct', 0.0)), device=ref0.device, dtype=ref0.dtype)
        return directed_difference_sequence(frame_inputs, direction_unit_xy=u)
    if _use_forward_move_axis(loss_cfg):
        return forward_difference_sequence(frame_inputs)
    return lateral_difference_sequence(frame_inputs)

def _movement_loss_terms(frame_inputs: Sequence[FrameLossInput], loss_cfg: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    if _use_directed_move_axis(loss_cfg):
        return directed_move_loss(frame_inputs, loss_type=str(loss_cfg.get('move_loss_type', 'smooth_l1')), front_back_pct=float(loss_cfg.get('move_direction_fb_pct', 100.0)), left_right_pct=float(loss_cfg.get('move_direction_lr_pct', 0.0)))
    if _use_forward_move_axis(loss_cfg):
        return forward_move_loss(frame_inputs, loss_type=str(loss_cfg.get('move_loss_type', 'smooth_l1')))
    return move_loss(frame_inputs, loss_type=str(loss_cfg.get('move_loss_type', 'smooth_l1')))

def _move_lateral_loss_weight(loss_cfg: Dict[str, Any], *, is_final_frame: bool) -> float:
    del is_final_frame
    return float(loss_cfg.get('move_center_weight', 1.0))

def _progress_child_loss_weight(loss_cfg: Dict[str, Any], child_key: str) -> float:
    return float(loss_cfg.get('progress_weight', 1.0)) * float(loss_cfg.get(child_key, 0.0))

def _cls_child_loss_weight(loss_cfg: Dict[str, Any], child_key: str) -> float:
    return float(loss_cfg.get('cls_weight', 1.0)) * float(loss_cfg.get(child_key, 0.0))

def _best_gate_passed_from_eval_record(record: Dict[str, Any], *, min_target_confidence: float) -> bool:
    matched_frames = int(record.get('matched_frames', 0))
    expected_frames = int(record.get('target_expected_frames', matched_frames + int(record.get('target_lost_frames', 0))))
    target_lost_frames = int(record.get('target_lost_frames', max(0, expected_frames - matched_frames)))
    min_confidence = float(record.get('target_confidence_min', 0.0))
    return bool(expected_frames > 0 and matched_frames == expected_frames and (target_lost_frames == 0) and (min_confidence >= float(min_target_confidence)))

def _canonical_model_name(raw_model_name: str) -> str:
    return selected_model_name({'model': raw_model_name})

def _auxiliary_model_specs_from_config(config: Dict[str, Any], *, active_model_name: str) -> List[Dict[str, Any]]:
    del active_model_name
    train_cfg = config.get('train', {}) if isinstance(config.get('train', {}), dict) else {}
    raw = train_cfg.get('auxiliary_models', train_cfg.get('ensemble_models', []))
    if raw in (None, '', []):
        return []
    items: List[Any]
    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(',') if part.strip()]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        raise ValueError('train.auxiliary_models must be a string, list, or empty')
    specs: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            model_name = _canonical_model_name(item)
            specs.append({'model': model_name, 'weight': 1.0, 'enabled': True})
            continue
        if not isinstance(item, dict):
            raise ValueError('train.auxiliary_models items must be model name strings or dicts')
        if not bool(item.get('enabled', True)):
            continue
        raw_name = str(item.get('model', item.get('name', '')) or '').strip()
        if not raw_name:
            raise ValueError('train.auxiliary_models dict entry missing model/name')
        model_name = _canonical_model_name(raw_name)
        specs.append({'model': model_name, 'weight': float(item.get('weight', 1.0)), 'enabled': True})
    return specs

def _clear_model_runtime_tensors(bev_model: Any) -> None:
    bev_model.last_bbox_tensor = None
    bev_model.last_cls_tensor = None
    if hasattr(bev_model, 'last_head'):
        bev_model.last_head = None
    if hasattr(bev_model, 'last_heatmap_tensor'):
        bev_model.last_heatmap_tensor = None
    if hasattr(bev_model, 'last_heatmap_grad'):
        bev_model.last_heatmap_grad = None
    if hasattr(bev_model, 'last_query_feature_tensor'):
        bev_model.last_query_feature_tensor = None
    bev_model.last_bbox_grad = None
    bev_model.last_cls_grad = None

def _frame_detection_loss_for_auxiliary_model(*, model_name: str, bev_model: Any, frame: FrameRecord, outs: Any, query_match: FixedQueryMatch, loss_cfg: Dict[str, Any], matched_frames_in_sequence: int, is_final_frame_in_sequence: bool, loss_reference_mode: str, clean_detection_refs: Optional[Dict[str, Dict[str, torch.Tensor]]]=None, training_config_path: Path) -> Tuple[torch.Tensor, Dict[str, float]]:
    prediction = bev_model.target_query_prediction(frame, outs, query_idx=query_match.query_idx)
    if bev_model.last_cls_tensor is None:
        raise RuntimeError(f'auxiliary model={model_name} forward did not expose class hook tensor')
    other_target_logits = _build_query_competition_terms(cls_tensor=bev_model.last_cls_tensor, query_idx=int(query_match.query_idx), target_label=int(prediction.target_label))
    query_feature = _query_feature_for_index(query_feature_tensor=getattr(bev_model, 'last_query_feature_tensor', None), query_idx=int(query_match.query_idx))
    ref_center_ego, ref_size_wlh, ref_yaw = _loss_reference_tensors_for_frame(frame=frame, prediction=prediction, loss_reference_mode=loss_reference_mode, clean_detection_refs=clean_detection_refs)
    frame_input = FrameLossInput(frame_id=frame.frame_id, pred_center_ego=prediction.pred_center_ego, gt_center_ego=prediction.gt_center_ego, pred_size_wlh=prediction.pred_box_lidar[3:6], gt_size_wlh=prediction.gt_box_lidar[3:6], pred_yaw=prediction.pred_box_lidar[6:7], gt_yaw=prediction.gt_box_lidar[6:7], pred_class_logits=prediction.class_logits, target_logit=prediction.target_logit, target_label=prediction.target_label, nearby_target_logits=None, other_target_logits=other_target_logits, other_query_max_logits=None, query_feature=query_feature, clean_query_feature=None, ref_center_ego=ref_center_ego, ref_size_wlh=ref_size_wlh, ref_yaw=ref_yaw)
    move_lateral_term, _move_longitudinal_term, _move_stats = _movement_loss_terms([frame_input], loss_cfg)
    rigid_term, _rigid_stats = rigid_loss([frame_input], loss_type=str(loss_cfg.get('rigid_loss_type', 'l2')), size_weight=float(loss_cfg.get('rigid_size_weight', 1.0)), yaw_weight=float(loss_cfg.get('rigid_yaw_weight', 1.0)))
    cls_term, cls_stats = cls_loss([frame_input], pos_weight=float(loss_cfg.get('cls_pos_weight', 1.0)), neg_weight=float(loss_cfg.get('cls_neg_weight', 1.0)), rank_weight=float(loss_cfg.get('cls_rank_weight', 1.0)), rank_margin=float(loss_cfg.get('cls_rank_margin', 0.0)))
    global_rank_term, _global_rank_stats = global_query_rank_loss([frame_input], margin=float(loss_cfg.get('cls_global_rank_margin', 0.0)))
    query_identity_term = move_lateral_term.new_zeros(())
    lateral_difference = _movement_difference_sequence([frame_input], loss_cfg)[0]
    if matched_frames_in_sequence == 1 and _progress_child_loss_weight(loss_cfg, 'first_frame_min_weight') > 0.0:
        first_frame_term = first_frame_min_loss(lateral_difference, min_m=float(loss_cfg.get('first_frame_min_m', 0.0)))
    else:
        first_frame_term = move_lateral_term.new_zeros(())
    if _progress_child_loss_weight(loss_cfg, 'per_frame_min_weight') > 0.0:
        per_frame_min_term = first_frame_min_loss(lateral_difference, min_m=float(loss_cfg.get('per_frame_min_m', 0.0)))
    else:
        per_frame_min_term = move_lateral_term.new_zeros(())
    move_lateral_weight = _move_lateral_loss_weight(loss_cfg, is_final_frame=bool(is_final_frame_in_sequence))
    depth_term = move_lateral_term.new_zeros(())
    if float(loss_cfg.get('depth_weight', 0.0)) > 0.0 and hasattr(bev_model, 'depth_supervision_loss'):
        depth_term, _depth_stats = bev_model.depth_supervision_loss(frame=frame, direction=str(loss_cfg.get('depth_direction', 'far')), offset_m=float(loss_cfg.get('depth_offset_m', 2.0)), patch_radius=int(loss_cfg.get('depth_patch_radius', 1)), loss_type=str(loss_cfg.get('depth_loss_type', 'l1')), min_valid_cams=int(loss_cfg.get('depth_min_valid_cams', 1)))
    frame_loss_term = move_lateral_term.new_tensor(move_lateral_weight) * move_lateral_term + rigid_term.new_tensor(float(loss_cfg.get('rigid_weight', 1.0))) * rigid_term + cls_term.new_tensor(float(loss_cfg.get('cls_weight', 1.0))) * cls_term + global_rank_term.new_tensor(_cls_child_loss_weight(loss_cfg, 'cls_global_rank_weight')) * global_rank_term + query_identity_term.new_tensor(_cls_child_loss_weight(loss_cfg, 'query_identity_weight')) * query_identity_term + first_frame_term.new_tensor(_progress_child_loss_weight(loss_cfg, 'first_frame_min_weight')) * first_frame_term + per_frame_min_term.new_tensor(_progress_child_loss_weight(loss_cfg, 'per_frame_min_weight')) * per_frame_min_term + depth_term.new_tensor(float(loss_cfg.get('depth_weight', 0.0))) * depth_term
    stp3_nl_stats: Dict[str, float] = {}
    if model_name == 'stp3' and float(loss_cfg.get('stp3_new_loss_weight', 0.0)) > 0.0 and isinstance(outs, dict) and ('segmentation' in outs):
        nw = float(loss_cfg.get('stp3_new_loss_weight', 0.0))
        n_present = int(getattr(getattr(bev_model, 'model', None), 'receptive_field', 1))
        vlog = outs['segmentation'][0, n_present - 1, 1].float()
        cfg = getattr(bev_model, 'cfg', None)
        y_step = float(cfg.LIFT.Y_BOUND[2]) if cfg is not None else 0.5
        stp3_nl_raw, stp3_nl_stats = compute_stp3_new_loss(vlog, mask_path=getattr(frame, 'stp3_bev_target_mask_path', '') or '', y_step_m=y_step, shift_lateral_m=float(loss_cfg.get('stp3_new_loss_shift_lateral_m', 1.0)), repulsion_weight=float(loss_cfg.get('stp3_new_loss_repulsion_weight', 1.0)), attraction_weight=float(loss_cfg.get('stp3_new_loss_attraction_weight', 1.0)), overlap_act_threshold=float(loss_cfg.get('stp3_new_loss_overlap_act_threshold', 0.5)), use_overlap_refinement=bool(loss_cfg.get('stp3_new_loss_use_overlap_refinement', True)), config_path=training_config_path)
        frame_loss_term = frame_loss_term + stp3_nl_raw.new_tensor(nw) * stp3_nl_raw
    return (frame_loss_term, {'loss_model': float(frame_loss_term.detach().item()), 'target_confidence': float(cls_stats.get('target_confidence', 0.0)), 'move_lateral_weight': float(move_lateral_weight), 'weighted_loss_move_lateral': float((move_lateral_term.detach() * move_lateral_term.new_tensor(move_lateral_weight)).item()), **{k: float(v) for k, v in stp3_nl_stats.items()}})

def _backward_auxiliary_model_step(*, runtime: AuxiliaryModelRuntime, frame_groups: Sequence[Sequence[FrameRecord]], renderer: FixedUVTextureRenderer, device: torch.device, use_amp: bool, amp_dtype: torch.dtype, optimizer: Optional[torch.optim.Optimizer], scaler: torch.amp.GradScaler, fixed_conf_threshold: float, fixed_max_center_dist_m: float, fixed_distance_axis: str, fixed_max_cross_axis_dist_m: float, training_config_path: Path) -> Dict[str, Any]:
    matched_frames = 0
    lost_frames = 0
    loss_values: List[float] = []
    confidence_values: List[float] = []
    expected_frames = _matched_frame_count_for_groups(frame_groups, runtime.fixed_queries)
    denom = max(1, int(expected_frames))
    for sequence_frames in frame_groups:
        prev_bev: Optional[torch.Tensor] = None
        prev_scene_token: Optional[str] = None
        prev_abs_pos: Optional[np.ndarray] = None
        prev_abs_angle: Optional[float] = None
        matched_frames_in_sequence = 0
        final_frame_cache_key = str(sequence_frames[-1].cache_key) if sequence_frames else ''
        for frame in sequence_frames:
            query_match = runtime.fixed_queries.get(frame.cache_key)
            retain_grad_for_frame = bool(runtime.final_decode_match or (query_match is not None and query_match.matched))
            with _autocast_context(device=device, enabled=use_amp and runtime.use_amp, amp_dtype=amp_dtype):
                camera_images = renderer.build_frame_images(frame, apply_eot=True)
                outs, prev_bev, prev_abs_pos, prev_abs_angle = runtime.model.forward_frame(frame, camera_images, prev_bev=prev_bev, prev_scene_token=prev_scene_token, prev_abs_pos=prev_abs_pos, prev_abs_angle=prev_abs_angle, retain_grad=retain_grad_for_frame)
            if isinstance(prev_bev, torch.Tensor):
                prev_bev = prev_bev.detach()
            prev_scene_token = frame.scene_token
            if runtime.final_decode_match:
                query_match = runtime.model.match_target_query_from_final_outputs(frame, outs, conf_threshold=float(fixed_conf_threshold), max_center_dist_m=float(fixed_max_center_dist_m), distance_axis=str(fixed_distance_axis), max_cross_axis_dist_m=float(fixed_max_cross_axis_dist_m))
            if query_match is None or not query_match.matched:
                lost_frames += 1
                del camera_images
                del outs
                _clear_model_runtime_tensors(runtime.model)
                continue
            matched_frames += 1
            matched_frames_in_sequence += 1
            frame_loss_term, stats = _frame_detection_loss_for_auxiliary_model(model_name=runtime.model_name, bev_model=runtime.model, frame=frame, outs=outs, query_match=query_match, loss_cfg=runtime.loss_cfg, matched_frames_in_sequence=matched_frames_in_sequence, is_final_frame_in_sequence=str(frame.cache_key) == final_frame_cache_key, loss_reference_mode=runtime.loss_reference_mode, clean_detection_refs=runtime.clean_detection_refs, training_config_path=training_config_path)
            scaled_loss = frame_loss_term * frame_loss_term.new_tensor(float(runtime.weight) / float(denom))
            if optimizer is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            loss_values.append(float(stats.get('loss_model', 0.0)))
            confidence_values.append(float(stats.get('target_confidence', 0.0)))
            del camera_images
            del outs
            _clear_model_runtime_tensors(runtime.model)
    mean_loss = _mean_or_zero(loss_values)
    return {'model': runtime.model_name, 'weight': float(runtime.weight), 'matched_frames': int(matched_frames), 'target_lost_frames': int(lost_frames), 'target_expected_frames': int(expected_frames), 'loss_model': float(mean_loss), 'loss_weighted': float(runtime.weight) * float(mean_loss), 'target_confidence': _mean_or_zero(confidence_values)}

def _log_training_settings(*, training_log_path: Path, config: Dict[str, Any], resolved_loss_cfg: Dict[str, Any], loss_cfg_source: str, config_path: Path, output_dir: Path, sequence_yaml: Sequence[Path], sequence_pkl: Path, seed: int, total_steps: int, start_step: int, log_every: int, checkpoint_every: int, optimizer_name: str, learning_rate: float, weight_decay: float, pgd_epsilon: float, grad_clip_norm: float, use_amp: bool, amp_dtype_name: str, best_min_target_confidence: float, resume_ckpt_path: Optional[Path]) -> None:
    dataset_cfg = config.get('dataset', {}) if isinstance(config.get('dataset', {}), dict) else {}
    model_name = selected_model_name(config)
    model_cfg = selected_model_cfg(config)
    mesh_cfg = config.get('mesh', {}) if isinstance(config.get('mesh', {}), dict) else {}
    sam2_cfg = config.get('sam2', {}) if isinstance(config.get('sam2', {}), dict) else {}
    render_cfg = config.get('render', {}) if isinstance(config.get('render', {}), dict) else {}
    camo_cfg = config.get('camouflage', {}) if isinstance(config.get('camouflage', {}), dict) else {}
    fixed_query_cfg = config.get('fixed_query', {}) if isinstance(config.get('fixed_query', {}), dict) else {}
    loss_cfg = resolved_loss_cfg if isinstance(resolved_loss_cfg, dict) else {}
    optimal_cfg = config.get('optimal', {}) if isinstance(config.get('optimal', {}), dict) else {}
    eff_cfg = _merge_optimal_into_loss_cfg(_train_config_to_loss_cfg(resolved_loss_cfg), config=config)
    move_cfg = loss_cfg.get('move', {}) if isinstance(loss_cfg.get('move', {}), dict) else {}
    progress_cfg = loss_cfg.get('progress', {}) if isinstance(loss_cfg.get('progress', {}), dict) else {}
    rigid_cfg = loss_cfg.get('rigid', {}) if isinstance(loss_cfg.get('rigid', {}), dict) else {}
    cls_cfg = loss_cfg.get('cls', {}) if isinstance(loss_cfg.get('cls', {}), dict) else {}
    depth_cfg = loss_cfg.get('depth', {}) if isinstance(loss_cfg.get('depth', {}), dict) else {}
    style_cfg = loss_cfg.get('style', {}) if isinstance(loss_cfg.get('style', {}), dict) else {}
    train_cfg = config.get('train', {}) if isinstance(config.get('train', {}), dict) else {}
    _append_log_line(training_log_path, '[train] ===== training config summary =====')
    _append_log_line(training_log_path, f'[train] config_path={config_path}')
    _append_log_line(training_log_path, f'[train] output_dir={output_dir}')
    _append_log_line(training_log_path, f"[train] dataset image_source_subdir={dataset_cfg.get('image_source_subdir', '')} dataroot={dataset_cfg.get('dataroot', '')} image_dataroot={dataset_cfg.get('image_dataroot', '')}")
    _append_log_line(training_log_path, f'[train] sequence_yaml_count={len(sequence_yaml)} group_key={_sequence_group_key_from_paths(sequence_yaml)}')
    for idx, path in enumerate(sequence_yaml, start=1):
        _append_log_line(training_log_path, f'[train] sequence_yaml[{idx}]={path}')
    _append_log_line(training_log_path, f'[train] sequence_pkl={sequence_pkl}')
    _append_log_line(training_log_path, f"[train] model={model_name} preset={model_cfg.get('preset', '')} config_path={model_cfg.get('config_path', '')} checkpoint={model_cfg.get('checkpoint_path', '')} workers_per_gpu={model_cfg.get('workers_per_gpu', '')}")
    _append_log_line(training_log_path, f'[train] loss_cfg_source={loss_cfg_source}')
    _append_log_line(training_log_path, f'[train] optimal={json.dumps(optimal_cfg, ensure_ascii=False) if optimal_cfg else "{}"}')
    _append_log_line(training_log_path, f"[train] mesh_obj={mesh_cfg.get('obj_path', '')}")
    _append_log_line(training_log_path, f"[train] sam2_repo={sam2_cfg.get('repo_root', '')} checkpoint={sam2_cfg.get('checkpoint_path', '')}")
    _append_log_line(training_log_path, f"[train] render near_plane_m={render_cfg.get('near_plane_m', 0.1)}")
    _append_log_line(training_log_path, f"[train] texture resolution={camo_cfg.get('resolution', '')} init_mode={camo_cfg.get('init_mode', '')} init_rgb={camo_cfg.get('init_rgb', '')} init_image={camo_cfg.get('init_image', '')} alpha={camo_cfg.get('alpha', 1.0)}")
    _append_log_line(training_log_path, f"[train] fixed_query conf_threshold={fixed_query_cfg.get('conf_threshold', '')} max_center_dist_m={fixed_query_cfg.get('max_center_dist_m', '')} distance_axis={fixed_query_cfg.get('distance_axis', 'lateral_y')} max_cross_axis_dist_m={fixed_query_cfg.get('max_cross_axis_dist_m', fixed_query_cfg.get('max_longitudinal_dist_m', 1.0))} cost_center_xy_weight={fixed_query_cfg.get('cost_center_xy_weight', '')} cost_confidence_weight={fixed_query_cfg.get('cost_confidence_weight', '')} fail_on_query_mismatch={fixed_query_cfg.get('fail_on_query_mismatch', True)}")
    _append_log_line(training_log_path, f"[train] run_args seed={seed} start_step={start_step} total_steps={total_steps} log_every={log_every} checkpoint_every={checkpoint_every} optimizer={optimizer_name} lr={learning_rate} weight_decay={weight_decay} pgd_epsilon={pgd_epsilon} grad_clip_norm={grad_clip_norm} use_amp={use_amp} amp_dtype={amp_dtype_name} loss_reference={train_cfg.get('loss_reference', 'gt')} clean_run={train_cfg.get('clean_run', False)} preload_clean_images_to_device={train_cfg.get('preload_clean_images_to_device', False)} resume_ckpt={(resume_ckpt_path if resume_ckpt_path is not None else '')}")
    _append_log_line(training_log_path, f"[train] progress_loss weight={progress_cfg.get('weight', 1.0)} lambda={progress_cfg.get('lambda', 1.0)} step_size_m={eff_cfg.get('progress_step_size_m', 0.05)} step_loss_type={eff_cfg.get('progress_step_loss_type', 'l2')} detach_previous={progress_cfg.get('detach_previous', False)} first_frame_min={progress_cfg.get('first_frame_min', 0.0)} first_frame_min_m={progress_cfg.get('first_frame_min_m', 0.0)} per_frame_min={progress_cfg.get('per_frame_min', 0.0)} per_frame_min_m={progress_cfg.get('per_frame_min_m', 0.0)}")
    _fb_log = eff_cfg.get('move_direction_fb_pct', '-')
    _lr_log = eff_cfg.get('move_direction_lr_pct', '-')
    _append_log_line(training_log_path, f"[train] move_loss axis={eff_cfg.get('move_axis', 'lateral_y')} direction_fb_lr_pct=({_fb_log},{_lr_log}) weight={move_cfg.get('weight', 1.0)} loss_type={move_cfg.get('loss_type', 'smooth_l1')}")
    _append_log_line(training_log_path, f"[train] rigid_loss weight={rigid_cfg.get('weight', 1.0)} loss_type={rigid_cfg.get('loss_type', 'l2')} size_weight={rigid_cfg.get('size_weight', 1.0)} yaw_weight={rigid_cfg.get('yaw_weight', 1.0)}")
    _append_log_line(training_log_path, f"[train] cls_loss weight={cls_cfg.get('weight', 1.0)} pos_weight={cls_cfg.get('pos_weight', 1.0)} neg_weight={cls_cfg.get('neg_weight', 1.0)} rank_weight={cls_cfg.get('rank_weight', 1.0)} rank_margin={cls_cfg.get('rank_margin', 0.0)} global_rank_weight={cls_cfg.get('global_rank_weight', 0.0)} global_rank_margin={cls_cfg.get('global_rank_margin', 0.0)} query_identity_weight={cls_cfg.get('query_identity_weight', 0.0)} query_identity_loss_type={cls_cfg.get('query_identity_loss_type', 'cosine')} formula=bce(target=1)+bce(max_non_target=0)+softplus(logsumexp(non_target)-target+rank_margin)+softplus(logsumexp(all_other_target)-target+global_margin)+identity(q,clean_q)")
    _append_log_line(training_log_path, f"[train] depth_loss weight={depth_cfg.get('weight', 0.0)} direction={depth_cfg.get('direction', 'far')} offset_m={depth_cfg.get('offset_m', 2.0)} patch_radius={depth_cfg.get('patch_radius', 1)} loss_type={depth_cfg.get('loss_type', 'l1')} min_valid_cams={depth_cfg.get('min_valid_cams', 1)}")
    _append_log_line(training_log_path, f'[train] checkpoint every {checkpoint_every} steps save step-xxxx.pt; target_confidence threshold for logging only={best_min_target_confidence:.3f}')
    _append_log_line(training_log_path, f"[train] style_loss weight={style_cfg.get('weight', 1.0)} tv_weight={style_cfg.get('tv_weight', 0.0)} l2_weight={style_cfg.get('l2_weight', 0.0)} brightness_weight={style_cfg.get('brightness_weight', 0.0)} brightness_target={style_cfg.get('brightness_target', 0.4)} brightness_loss_type={style_cfg.get('brightness_loss_type', 'l2')} nps_weight={style_cfg.get('nps_weight', 0.0)} printable_palette={style_cfg.get('printable_palette', 'default_nuscenes')}")
    eot_summary = camo_cfg.get('eot', {})
    _append_log_line(training_log_path, f'[train] eot_cfg={json.dumps(eot_summary, ensure_ascii=False)}')
    image_aug_summary = camo_cfg.get('image_augmentation', {})
    _append_log_line(training_log_path, f'[train] image_aug_cfg={json.dumps(image_aug_summary, ensure_ascii=False)}')
    _append_log_line(training_log_path, '[train] ===== end training config summary =====')

def _hook_stats_for_frame(query_match: FixedQueryMatch, bbox_tensor: torch.Tensor, cls_tensor: torch.Tensor, *, target_label: int, grad_scale: float=1.0, heatmap_tensor: Optional[torch.Tensor]=None) -> Dict[str, Any]:
    query_idx = int(query_match.query_idx)
    safe_scale = float(grad_scale) if float(grad_scale) > 0.0 else 1.0
    stats: Dict[str, Any] = {}
    if bbox_tensor.grad is not None and cls_tensor.grad is not None:
        bbox_grad_tensor = bbox_tensor.grad[query_idx, 0:3].detach() / safe_scale
        cls_grad_tensor = cls_tensor.grad[query_idx, target_label].detach() / safe_scale
        bbox_grad = bbox_grad_tensor.cpu().tolist()
        cls_grad = float(cls_grad_tensor.cpu().item())
        stats.update({'bbox_center_grad_xyz': [float(v) for v in bbox_grad], 'cls_target_grad': cls_grad})
    if heatmap_tensor is not None and heatmap_tensor.grad is not None:
        grid_u = int(getattr(query_match, 'grid_u', -1))
        grid_v = int(getattr(query_match, 'grid_v', -1))
        if heatmap_tensor.ndim == 4 and 0 <= grid_v < int(heatmap_tensor.shape[2]) and (0 <= grid_u < int(heatmap_tensor.shape[3])):
            heatmap_grad = heatmap_tensor.grad[0, int(target_label), grid_v, grid_u].detach() / safe_scale
            stats['heatmap_target_cell_grad'] = float(heatmap_grad.cpu().item())
            stats['heatmap_target_cell_abs_grad'] = float(heatmap_grad.detach().abs().cpu().item())
    return stats

def _clone_query_match_with_new_idx(query_match: FixedQueryMatch, *, query_idx: int) -> FixedQueryMatch:
    return FixedQueryMatch(sample_token=query_match.sample_token, frame_id=query_match.frame_id, matched=query_match.matched, query_idx=int(query_idx), confidence=float(query_match.confidence), world_distance_m=float(query_match.world_distance_m), match_cost=float(query_match.match_cost), candidate_total=int(query_match.candidate_total), candidate_after_conf=int(query_match.candidate_after_conf), candidate_after_dist=int(query_match.candidate_after_dist), target_world_xy=tuple(query_match.target_world_xy), pred_world_xy=tuple(query_match.pred_world_xy) if query_match.pred_world_xy is not None else None, target_detection_name=str(query_match.target_detection_name), unmatched_reason=str(query_match.unmatched_reason), grid_u=int(query_match.grid_u), grid_v=int(query_match.grid_v), target_ego_xyz=tuple(query_match.target_ego_xyz) if query_match.target_ego_xyz is not None else None)

def _build_unmatched_query_placeholders(frames: Sequence[FrameRecord]) -> Dict[str, FixedQueryMatch]:
    placeholders: Dict[str, FixedQueryMatch] = {}
    for frame in frames:
        placeholders[frame.cache_key] = FixedQueryMatch(sample_token=frame.sample_token, frame_id=frame.frame_id, matched=False, query_idx=-1, confidence=0.0, world_distance_m=float('inf'), match_cost=float('inf'), candidate_total=0, candidate_after_conf=0, candidate_after_dist=0, target_world_xy=(float(frame.gt_center_world[0]), float(frame.gt_center_world[1])), pred_world_xy=None, target_detection_name=str(frame.gt_category_name), unmatched_reason='dynamic_final_decode_query_selection', target_ego_xyz=None)
    return placeholders

def _match_final_decode_queries(*, frames: Sequence[FrameRecord], image_provider: Any, bev_model: BevFormerGradientModel, device: torch.device, use_amp: bool, amp_dtype: torch.dtype, conf_threshold: float, max_center_dist_m: float, distance_axis: str, max_cross_axis_dist_m: float) -> Dict[str, FixedQueryMatch]:
    if not hasattr(bev_model, 'match_target_query_from_final_outputs'):
        raise RuntimeError('final decode fixed target requires match_target_query_from_final_outputs')
    matches: Dict[str, FixedQueryMatch] = {}
    prev_bev: Optional[torch.Tensor] = None
    prev_scene_token: Optional[str] = None
    prev_abs_pos: Optional[np.ndarray] = None
    prev_abs_angle: Optional[float] = None
    with torch.inference_mode():
        for frame in frames:
            with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                camera_images = image_provider(frame)
                outs, prev_bev, prev_abs_pos, prev_abs_angle = bev_model.forward_frame(frame, camera_images, prev_bev=prev_bev, prev_scene_token=prev_scene_token, prev_abs_pos=prev_abs_pos, prev_abs_angle=prev_abs_angle, retain_grad=False)
            if isinstance(prev_bev, torch.Tensor):
                prev_bev = prev_bev.detach()
            prev_scene_token = frame.scene_token
            matches[frame.cache_key] = bev_model.match_target_query_from_final_outputs(frame, outs, conf_threshold=float(conf_threshold), max_center_dist_m=float(max_center_dist_m), distance_axis=str(distance_axis), max_cross_axis_dist_m=float(max_cross_axis_dist_m))
            del outs
            del camera_images
            _clear_model_runtime_tensors(bev_model)
    return matches

def _build_query_competition_terms(*, cls_tensor: torch.Tensor, query_idx: int, target_label: int) -> torch.Tensor:
    if cls_tensor.ndim != 2:
        raise ValueError(f'Expected cls_tensor=[num_query,num_class], got shape={tuple(cls_tensor.shape)}')
    num_query = int(cls_tensor.shape[0])
    if query_idx < 0 or query_idx >= num_query:
        raise IndexError(f'query_idx={query_idx} out of range for cls_tensor with num_query={num_query}')
    keep_mask = torch.ones((num_query,), dtype=torch.bool, device=cls_tensor.device)
    keep_mask[int(query_idx)] = False
    if int(keep_mask.sum().item()) <= 0:
        return cls_tensor.new_zeros((0,))
    other_cls = cls_tensor[keep_mask]
    return other_cls[:, int(target_label)]

def _query_feature_for_index(*, query_feature_tensor: Optional[torch.Tensor], query_idx: int) -> Optional[torch.Tensor]:
    if query_feature_tensor is None:
        return None
    if query_feature_tensor.ndim != 2:
        return None
    if query_idx < 0 or query_idx >= int(query_feature_tensor.shape[0]):
        return None
    return query_feature_tensor[int(query_idx)]

def _collect_clean_query_features(*, frame_groups: Sequence[Sequence[FrameRecord]], renderer: FixedUVTextureRenderer, bev_model: BevFormerGradientModel, fixed_queries: Dict[str, FixedQueryMatch], device: torch.device, use_amp: bool, amp_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    payload: Dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for sequence_frames in frame_groups:
            prev_bev: Optional[torch.Tensor] = None
            prev_scene_token: Optional[str] = None
            prev_abs_pos: Optional[np.ndarray] = None
            prev_abs_angle: Optional[float] = None
            for frame in sequence_frames:
                query_match = fixed_queries.get(frame.cache_key)
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    camera_images = renderer.build_frame_images(frame, apply_eot=False)
                    outs, prev_bev, prev_abs_pos, prev_abs_angle = bev_model.forward_frame(frame, camera_images, prev_bev=prev_bev, prev_scene_token=prev_scene_token, prev_abs_pos=prev_abs_pos, prev_abs_angle=prev_abs_angle, retain_grad=False)
                if isinstance(prev_bev, torch.Tensor):
                    prev_bev = prev_bev.detach()
                prev_scene_token = frame.scene_token
                if query_match is not None and query_match.matched:
                    query_feature = _query_feature_for_index(query_feature_tensor=getattr(bev_model, 'last_query_feature_tensor', None), query_idx=int(query_match.query_idx))
                    if query_feature is not None:
                        payload[frame.cache_key] = query_feature.detach().to(dtype=torch.float32, device='cpu')
                del outs
                del camera_images
                bev_model.last_bbox_tensor = None
                bev_model.last_cls_tensor = None
                if hasattr(bev_model, 'last_head'):
                    bev_model.last_head = None
                if hasattr(bev_model, 'last_heatmap_tensor'):
                    bev_model.last_heatmap_tensor = None
                if hasattr(bev_model, 'last_heatmap_grad'):
                    bev_model.last_heatmap_grad = None
                if hasattr(bev_model, 'last_query_feature_tensor'):
                    bev_model.last_query_feature_tensor = None
                bev_model.last_bbox_grad = None
                bev_model.last_cls_grad = None
    return payload

def _loss_reference_mode_from_config(config: Dict[str, Any]) -> str:
    train_cfg = config.get('train', {}) if isinstance(config.get('train', {}), dict) else {}
    mode = str(train_cfg.get('loss_reference', 'gt') or 'gt').strip().lower()
    if mode not in {'gt', 'clean'}:
        raise ValueError(f'train.loss_reference only supports gt / clean; got={mode!r}')
    return mode

def _clean_detection_cache_path(*, config: Dict[str, Any], config_path: Path, sequence_yaml_path: Path, model_name: str, final_decode_match: bool, fixed_conf_threshold: float, fixed_max_center_dist_m: float, fixed_distance_axis: str, fixed_max_cross_axis_dist_m: float) -> Path:
    cache_dir, _ = _precompute_cache_dir(config=config, config_path=config_path, sequence_yaml_paths=[sequence_yaml_path])
    model_cfg = selected_model_cfg(config)
    clean_cache_version = 2
    signature = {'version': clean_cache_version, 'sequence_yaml': _file_signature(sequence_yaml_path), 'model_name': str(model_name), 'model_cfg': copy.deepcopy(model_cfg), 'final_decode_match': bool(final_decode_match), 'fixed_query': {'conf_threshold': float(fixed_conf_threshold), 'max_center_dist_m': float(fixed_max_center_dist_m), 'distance_axis': str(fixed_distance_axis), 'max_cross_axis_dist_m': float(fixed_max_cross_axis_dist_m)}}
    digest = hashlib.md5(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:12]
    return cache_dir / 'clean_detection' / f'{model_name}-clean-{digest}.yaml'

def _clean_detection_refs_from_payload(payload: Dict[str, Any], *, device: torch.device) -> Dict[str, Dict[str, torch.Tensor]]:
    refs: Dict[str, Dict[str, torch.Tensor]] = {}
    rows = payload.get('frames', []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return refs
    for row in rows:
        if not isinstance(row, dict) or not bool(row.get('matched', False)):
            continue
        cache_key = str(row.get('cache_key', '') or '').strip()
        ref_center = row.get('ref_center_ego', None)
        ref_box = row.get('ref_box_lidar', None)
        if not cache_key or not isinstance(ref_center, list) or (not isinstance(ref_box, list)):
            continue
        refs[cache_key] = {'center_ego': torch.as_tensor(ref_center, device=device, dtype=torch.float32).reshape(3), 'box_lidar': torch.as_tensor(ref_box, device=device, dtype=torch.float32).reshape(-1)}
    return refs

def _collect_clean_detection_reference_payload(*, sequence_name: str, sequence_frames: Sequence[FrameRecord], renderer: FixedUVTextureRenderer, bev_model: BevFormerGradientModel, fixed_queries: Dict[str, FixedQueryMatch], device: torch.device, use_amp: bool, amp_dtype: torch.dtype, final_decode_match: bool, fixed_conf_threshold: float, fixed_max_center_dist_m: float, fixed_distance_axis: str, fixed_max_cross_axis_dist_m: float) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    prev_bev: Optional[torch.Tensor] = None
    prev_scene_token: Optional[str] = None
    prev_abs_pos: Optional[np.ndarray] = None
    prev_abs_angle: Optional[float] = None
    with torch.inference_mode():
        for frame in sequence_frames:
            with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                camera_images = {channel: renderer.clean_image(frame, channel) for channel in CAMERA_CHANNELS}
                outs, prev_bev, prev_abs_pos, prev_abs_angle = bev_model.forward_frame(frame, camera_images, prev_bev=prev_bev, prev_scene_token=prev_scene_token, prev_abs_pos=prev_abs_pos, prev_abs_angle=prev_abs_angle, retain_grad=False)
            if isinstance(prev_bev, torch.Tensor):
                prev_bev = prev_bev.detach()
            prev_scene_token = frame.scene_token
            base_match = fixed_queries.get(frame.cache_key)
            chosen_match = base_match
            reference_source = 'clean_fixed_query'
            if final_decode_match and hasattr(bev_model, 'match_target_query_from_final_outputs'):
                final_match = bev_model.match_target_query_from_final_outputs(frame, outs, conf_threshold=float(fixed_conf_threshold), max_center_dist_m=float(fixed_max_center_dist_m), distance_axis=str(fixed_distance_axis), max_cross_axis_dist_m=float(fixed_max_cross_axis_dist_m))
                if final_match is not None and final_match.matched:
                    chosen_match = final_match
                    reference_source = 'clean_final_decode'
                elif base_match is not None and base_match.matched:
                    reference_source = 'clean_fixed_query_fallback'
            if chosen_match is None or not chosen_match.matched:
                rows.append({'sequence_name': str(sequence_name), 'cache_key': str(frame.cache_key), 'frame_id': int(frame.frame_id), 'sample_token': str(frame.sample_token), 'matched': False, 'reference_source': str(reference_source), 'reason': str(chosen_match.unmatched_reason) if chosen_match is not None else 'missing_clean_query_match'})
                del camera_images
                del outs
                _clear_model_runtime_tensors(bev_model)
                continue
            prediction = bev_model.target_query_prediction(frame, outs, query_idx=int(chosen_match.query_idx))
            rows.append({'sequence_name': str(sequence_name), 'cache_key': str(frame.cache_key), 'frame_id': int(frame.frame_id), 'sample_token': str(frame.sample_token), 'matched': True, 'reference_source': str(reference_source), 'query_idx': int(chosen_match.query_idx), 'target_detection_name': str(prediction.target_detection_name), 'target_confidence': float(torch.sigmoid(prediction.target_logit.detach()).item()), 'ref_center_ego': [float(v) for v in prediction.pred_center_ego.detach().cpu().tolist()], 'ref_box_lidar': [float(v) for v in prediction.pred_box_lidar.detach().cpu().tolist()]})
            del prediction
            del camera_images
            del outs
            _clear_model_runtime_tensors(bev_model)
    return {'sequence_name': str(sequence_name), 'model_name': str(getattr(bev_model, '__class__', type(bev_model)).__name__), 'frames': rows}

def _load_or_compute_clean_detection_references(*, config: Dict[str, Any], config_path: Path, sequence_yaml_paths: Sequence[Path], frame_groups: Sequence[Sequence[FrameRecord]], renderer: FixedUVTextureRenderer, bev_model: BevFormerGradientModel, fixed_queries: Dict[str, FixedQueryMatch], device: torch.device, use_amp: bool, amp_dtype: torch.dtype, model_name: str, final_decode_match: bool, fixed_conf_threshold: float, fixed_max_center_dist_m: float, fixed_distance_axis: str, fixed_max_cross_axis_dist_m: float, training_log_path: Path, split_label: str) -> Dict[str, Dict[str, torch.Tensor]]:
    refs: Dict[str, Dict[str, torch.Tensor]] = {}
    group_by_name = {str(group[0].sequence_name): list(group) for group in frame_groups if group}
    for sequence_yaml_path in sequence_yaml_paths:
        sequence_name = str(sequence_yaml_path.stem)
        sequence_frames = group_by_name.get(sequence_name, [])
        if not sequence_frames:
            continue
        cache_path = _clean_detection_cache_path(config=config, config_path=config_path, sequence_yaml_path=sequence_yaml_path, model_name=model_name, final_decode_match=final_decode_match, fixed_conf_threshold=fixed_conf_threshold, fixed_max_center_dist_m=fixed_max_center_dist_m, fixed_distance_axis=fixed_distance_axis, fixed_max_cross_axis_dist_m=fixed_max_cross_axis_dist_m)
        payload: Optional[Dict[str, Any]] = None
        if cache_path.exists():
            try:
                loaded = _load_yaml(cache_path)
                loaded_rows = loaded.get('frames', []) if isinstance(loaded, dict) else []
                loaded_refs = _clean_detection_refs_from_payload(loaded, device=device)
                if isinstance(loaded_rows, list) and len(loaded_rows) >= len(sequence_frames):
                    payload = loaded
                    refs.update(loaded_refs)
                    _append_log_line(training_log_path, f'[{split_label}] clean_det_cache_hit seq={sequence_name} matched={len(loaded_refs)}/{len(sequence_frames)} path={cache_path}')
            except Exception:
                payload = None
        if payload is None:
            payload = _collect_clean_detection_reference_payload(sequence_name=sequence_name, sequence_frames=sequence_frames, renderer=renderer, bev_model=bev_model, fixed_queries=fixed_queries, device=device, use_amp=use_amp, amp_dtype=amp_dtype, final_decode_match=final_decode_match, fixed_conf_threshold=fixed_conf_threshold, fixed_max_center_dist_m=fixed_max_center_dist_m, fixed_distance_axis=fixed_distance_axis, fixed_max_cross_axis_dist_m=fixed_max_cross_axis_dist_m)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            _save_yaml(cache_path, payload)
            refs.update(_clean_detection_refs_from_payload(payload, device=device))
            _append_log_line(training_log_path, f"[{split_label}] clean_det_cache_saved seq={sequence_name} matched={sum((1 for row in payload.get('frames', []) if isinstance(row, dict) and bool(row.get('matched', False))))}/{len(sequence_frames)} path={cache_path}")
    return refs

def _loss_reference_tensors_for_frame(*, frame: FrameRecord, prediction: Any, loss_reference_mode: str, clean_detection_refs: Optional[Dict[str, Dict[str, torch.Tensor]]]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if str(loss_reference_mode).strip().lower() != 'clean':
        return (None, None, None)
    if not clean_detection_refs:
        return (None, None, None)
    cached = clean_detection_refs.get(frame.cache_key)
    if not isinstance(cached, dict):
        return (None, None, None)
    ref_center = cached.get('center_ego')
    ref_box = cached.get('box_lidar')
    if not isinstance(ref_center, torch.Tensor) or not isinstance(ref_box, torch.Tensor):
        return (None, None, None)
    ref_center_tensor = ref_center.to(device=prediction.pred_center_ego.device, dtype=prediction.pred_center_ego.dtype).reshape(3)
    ref_box_tensor = ref_box.to(device=prediction.pred_box_lidar.device, dtype=prediction.pred_box_lidar.dtype).reshape(-1)
    if int(ref_box_tensor.numel()) < 7:
        return (ref_center_tensor, None, None)
    return (ref_center_tensor, ref_box_tensor[3:6], ref_box_tensor[6:7])

def _log_fixed_query_matches(*, frames: Sequence[FrameRecord], fixed_queries: Dict[str, FixedQueryMatch], training_log_path: Path, prefix: str='[train]') -> None:
    header = f'{prefix} locked target queries before training (GT class per frame):'
    print(header)
    _append_log_line(training_log_path, header)
    for frame in frames:
        query = fixed_queries.get(frame.cache_key)
        if query is None:
            continue
        if not query.matched or query.pred_world_xy is None:
            line = f'{prefix}   seq={frame.sequence_name} frame={frame.frame_id} matched=False reason={query.unmatched_reason} after_conf={query.candidate_after_conf} after_dist={query.candidate_after_dist}'
        else:
            dx = float(query.pred_world_xy[0] - query.target_world_xy[0])
            dy = float(query.pred_world_xy[1] - query.target_world_xy[1])
            line = f'{prefix}   seq={frame.sequence_name} frame={frame.frame_id} query={query.query_idx} target_cls={query.target_detection_name} lock_conf={query.confidence:.4f} pred_xy=({query.pred_world_xy[0]:.3f},{query.pred_world_xy[1]:.3f}) gt_xy=({query.target_world_xy[0]:.3f},{query.target_world_xy[1]:.3f}) diff_xy=({dx:.3f},{dy:.3f})m dist={query.world_distance_m:.3f}m'
        print(line)
        _append_log_line(training_log_path, line)

def _style_printable_palette_from_config(style_cfg: Dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    raw_palette = style_cfg.get('printable_palette', 'default_nuscenes')
    if raw_palette in (None, '', 'default_nuscenes'):
        return None
    if not isinstance(raw_palette, list):
        raise ValueError('loss.style.printable_palette must be an RGB list or default_nuscenes')
    rows: List[List[float]] = []
    for item in raw_palette:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError('each color in loss.style.printable_palette must be [R,G,B]')
        rgb = [float(v) for v in item]
        if max(rgb) > 1.0:
            rgb = [v / 255.0 for v in rgb]
        rows.append(rgb)
    if not rows:
        return None
    return torch.tensor(rows, device=device, dtype=dtype)

def _evaluate_fixed_queries_for_snapshot(*, frames: Sequence[FrameRecord], renderer: FixedUVTextureRenderer, bev_model: BevFormerGradientModel, fixed_queries: Dict[str, FixedQueryMatch], device: torch.device, use_amp: bool, amp_dtype: torch.dtype, final_decode_match: bool=False, final_decode_conf_threshold: float=0.0, final_decode_max_center_dist_m: float=0.0, final_decode_distance_axis: str='lateral_y', final_decode_max_cross_axis_dist_m: float=1.0) -> List[Dict[str, Any]]:
    frame_rows: List[Dict[str, Any]] = []
    prev_bev: Optional[torch.Tensor] = None
    prev_scene_token: Optional[str] = None
    prev_abs_pos: Optional[np.ndarray] = None
    prev_abs_angle: Optional[float] = None
    with torch.inference_mode():
        for frame in frames:
            query_match = fixed_queries.get(frame.cache_key)
            if not final_decode_match and (query_match is None or not query_match.matched):
                continue
            with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                camera_images = renderer.build_frame_images(frame, apply_eot=False)
                outs, prev_bev, prev_abs_pos, prev_abs_angle = bev_model.forward_frame(frame, camera_images, prev_bev=prev_bev, prev_scene_token=prev_scene_token, prev_abs_pos=prev_abs_pos, prev_abs_angle=prev_abs_angle, retain_grad=False)
            if isinstance(prev_bev, torch.Tensor):
                prev_bev = prev_bev.detach()
            prev_scene_token = frame.scene_token
            if final_decode_match:
                if not hasattr(bev_model, 'match_target_query_from_final_outputs'):
                    raise RuntimeError('final_decode_match=true but model lacks match_target_query_from_final_outputs')
                query_match = bev_model.match_target_query_from_final_outputs(frame, outs, conf_threshold=float(final_decode_conf_threshold), max_center_dist_m=float(final_decode_max_center_dist_m), distance_axis=str(final_decode_distance_axis), max_cross_axis_dist_m=float(final_decode_max_cross_axis_dist_m))
            if query_match is None or not query_match.matched:
                if query_match is not None:
                    frame_rows.append({'matched': False, 'sequence_name': str(frame.sequence_name), 'frame_id': int(frame.frame_id), 'sample_token': str(frame.sample_token), 'target_detection_name': str(query_match.target_detection_name), 'unmatched_reason': str(query_match.unmatched_reason), 'candidate_total': int(query_match.candidate_total), 'candidate_after_conf': int(query_match.candidate_after_conf), 'candidate_after_dist': int(query_match.candidate_after_dist)})
                del outs
                del camera_images
                continue
            prediction = bev_model.target_query_prediction(frame, outs, query_idx=query_match.query_idx)
            selected_query_idx = int(getattr(prediction, 'query_idx', query_match.query_idx))
            delta_y = float((prediction.pred_center_ego[1] - prediction.gt_center_ego[1]).detach().item())
            delta_x = float((prediction.pred_center_ego[0] - prediction.gt_center_ego[0]).detach().item())
            size_diff = (prediction.pred_box_lidar[3:6] - prediction.gt_box_lidar[3:6]).detach().abs()
            yaw_diff = torch.atan2(torch.sin(prediction.pred_box_lidar[6:7] - prediction.gt_box_lidar[6:7]), torch.cos(prediction.pred_box_lidar[6:7] - prediction.gt_box_lidar[6:7])).detach().abs()
            yaw_diff_deg = float(math.degrees(float(yaw_diff.item())))
            frame_rows.append({'sequence_name': str(frame.sequence_name), 'frame_id': int(frame.frame_id), 'sample_token': str(frame.sample_token), 'query_idx': int(selected_query_idx), 'delta_y_m': delta_y, 'delta_x_m': delta_x, 'direction': 'left' if delta_y >= 0.0 else 'right', 'shift_abs_m': abs(delta_y), 'target_lateral_move_m': delta_y if float(prediction.gt_center_ego[1].detach().item()) < 0.0 else -delta_y, 'x_direction': 'front' if delta_x >= 0.0 else 'back', 'x_shift_abs_m': abs(delta_x), 'target_detection_name': str(prediction.target_detection_name), 'target_confidence': float(torch.sigmoid(prediction.target_logit.detach()).item()), 'size_diff_mean': float(size_diff.mean().item()), 'yaw_diff_rad': float(yaw_diff.item()), 'yaw_diff_deg': yaw_diff_deg})
            del prediction
            del outs
            del camera_images
            bev_model.last_bbox_tensor = None
            bev_model.last_cls_tensor = None
            if hasattr(bev_model, 'last_head'):
                bev_model.last_head = None
            if hasattr(bev_model, 'last_heatmap_tensor'):
                bev_model.last_heatmap_tensor = None
            if hasattr(bev_model, 'last_heatmap_grad'):
                bev_model.last_heatmap_grad = None
            if hasattr(bev_model, 'last_query_feature_tensor'):
                bev_model.last_query_feature_tensor = None
            bev_model.last_bbox_grad = None
            bev_model.last_cls_grad = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return frame_rows

def _evaluate_dataset_without_grad(*, frame_groups: Sequence[Sequence[FrameRecord]], renderer: FixedUVTextureRenderer, bev_model: BevFormerGradientModel, fixed_queries: Dict[str, FixedQueryMatch], loss_cfg: Dict[str, Any], device: torch.device, use_amp: bool, amp_dtype: torch.dtype, apply_eot: bool, clean_query_features: Optional[Dict[str, torch.Tensor]]=None, loss_reference_mode: str='gt', clean_detection_refs: Optional[Dict[str, Dict[str, torch.Tensor]]]=None, final_decode_match: bool=False, final_decode_conf_threshold: float=0.0, final_decode_max_center_dist_m: float=0.0, final_decode_distance_axis: str='lateral_y', final_decode_max_cross_axis_dist_m: float=1.0) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    enable_query_terms = bool(loss_cfg.get('enable_query_losses', True))
    frame_debug_rows: List[Dict[str, Any]] = []
    matched_frames_step = 0
    target_lost_frames_step = 0
    progress_term_values: List[float] = []
    progress_teacher_values: List[float] = []
    progress_floor_values: List[float] = []
    progress_gain_values: List[float] = []
    move_lateral_term_values: List[float] = []
    move_lateral_weight_values: List[float] = []
    move_lateral_weighted_values: List[float] = []
    move_longitudinal_term_values: List[float] = []
    move_longitudinal_mean_values: List[float] = []
    move_longitudinal_max_values: List[float] = []
    move_longitudinal_hard_values: List[float] = []
    move_longitudinal_hard_excess_values: List[float] = []
    move_longitudinal_hard_excess_max_values: List[float] = []
    lateral_difference_values: List[float] = []
    longitudinal_difference_values: List[float] = []
    longitudinal_difference_max_values: List[float] = []
    first_frame_min_term_values: List[float] = []
    per_frame_min_term_values: List[float] = []
    rigid_term_values: List[float] = []
    rigid_size_loss_values: List[float] = []
    rigid_yaw_loss_values: List[float] = []
    rigid_size_diff_values: List[float] = []
    rigid_yaw_diff_values: List[float] = []
    cls_term_values: List[float] = []
    cls_pos_values: List[float] = []
    cls_neg_values: List[float] = []
    cls_rank_values: List[float] = []
    cls_nearby_values: List[float] = []
    cls_global_rank_values: List[float] = []
    cls_output_proxy_values: List[float] = []
    cls_query_identity_values: List[float] = []
    target_logit_values: List[float] = []
    target_confidence_values: List[float] = []
    noncar_max_conf_values: List[float] = []
    nearby_target_max_conf_values: List[float] = []
    nearby_query_count_values: List[float] = []
    global_other_target_max_conf_values: List[float] = []
    other_query_global_max_conf_values: List[float] = []
    query_identity_cosine_values: List[float] = []
    depth_term_values: List[float] = []
    depth_cam_count_values: List[float] = []
    depth_sample_count_values: List[float] = []
    depth_pred_mean_values: List[float] = []
    depth_gt_mean_values: List[float] = []
    bevdet_loss_values: List[float] = []
    prev_bev: Optional[torch.Tensor] = None
    prev_scene_token: Optional[str] = None
    prev_abs_pos: Optional[np.ndarray] = None
    prev_abs_angle: Optional[float] = None
    with torch.inference_mode():
        for sequence_frames in frame_groups:
            prev_bev = None
            prev_scene_token = None
            prev_abs_pos = None
            prev_abs_angle = None
            sequence_inputs: List[FrameLossInput] = []
            matched_frames_in_sequence = 0
            final_frame_cache_key = str(sequence_frames[-1].cache_key) if sequence_frames else ''
            for frame in sequence_frames:
                query_match = fixed_queries.get(frame.cache_key)
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    camera_images = renderer.build_frame_images(frame, apply_eot=apply_eot)
                    outs, prev_bev, prev_abs_pos, prev_abs_angle = bev_model.forward_frame(frame, camera_images, prev_bev=prev_bev, prev_scene_token=prev_scene_token, prev_abs_pos=prev_abs_pos, prev_abs_angle=prev_abs_angle, retain_grad=False)
                if isinstance(prev_bev, torch.Tensor):
                    prev_bev = prev_bev.detach()
                prev_scene_token = frame.scene_token
                if final_decode_match:
                    if not hasattr(bev_model, 'match_target_query_from_final_outputs'):
                        raise RuntimeError('final_decode_match=true but model lacks match_target_query_from_final_outputs')
                    query_match = bev_model.match_target_query_from_final_outputs(frame, outs, conf_threshold=float(final_decode_conf_threshold), max_center_dist_m=float(final_decode_max_center_dist_m), distance_axis=str(final_decode_distance_axis), max_cross_axis_dist_m=float(final_decode_max_cross_axis_dist_m))
                if query_match is None or not query_match.matched:
                    if final_decode_match and query_match is not None:
                        target_lost_frames_step += 1
                        frame_debug_rows.append({'matched': False, 'sequence_name': str(frame.sequence_name), 'frame_id': int(frame.frame_id), 'sample_token': str(frame.sample_token), 'target_detection_name': str(query_match.target_detection_name), 'unmatched_reason': str(query_match.unmatched_reason), 'candidate_total': int(query_match.candidate_total), 'candidate_after_conf': int(query_match.candidate_after_conf), 'candidate_after_dist': int(query_match.candidate_after_dist)})
                    del camera_images
                    del outs
                    continue
                matched_frames_step += 1
                matched_frames_in_sequence += 1
                prediction = bev_model.target_query_prediction(frame, outs, query_idx=query_match.query_idx)
                if bev_model.last_cls_tensor is None:
                    raise RuntimeError('model forward did not expose class hook tensor')
                other_target_logits = _build_query_competition_terms(cls_tensor=bev_model.last_cls_tensor, query_idx=int(query_match.query_idx), target_label=int(prediction.target_label))
                query_feature = _query_feature_for_index(query_feature_tensor=getattr(bev_model, 'last_query_feature_tensor', None), query_idx=int(query_match.query_idx))
                clean_query_feature = None
                if clean_query_features is not None:
                    cached_feature = clean_query_features.get(frame.cache_key)
                    if cached_feature is not None:
                        clean_query_feature = cached_feature.to(device=prediction.target_logit.device, dtype=torch.float32)
                ref_center_ego, ref_size_wlh, ref_yaw = _loss_reference_tensors_for_frame(frame=frame, prediction=prediction, loss_reference_mode=loss_reference_mode, clean_detection_refs=clean_detection_refs)
                frame_input = FrameLossInput(frame_id=frame.frame_id, pred_center_ego=prediction.pred_center_ego, gt_center_ego=prediction.gt_center_ego, pred_size_wlh=prediction.pred_box_lidar[3:6], gt_size_wlh=prediction.gt_box_lidar[3:6], pred_yaw=prediction.pred_box_lidar[6:7], gt_yaw=prediction.gt_box_lidar[6:7], pred_class_logits=prediction.class_logits, target_logit=prediction.target_logit, target_label=prediction.target_label, nearby_target_logits=None, other_target_logits=other_target_logits, other_query_max_logits=None, query_feature=query_feature, clean_query_feature=clean_query_feature, ref_center_ego=ref_center_ego, ref_size_wlh=ref_size_wlh, ref_yaw=ref_yaw)
                sequence_inputs.append(frame_input)
                display_ref_center = ref_center_ego if ref_center_ego is not None else frame_input.gt_center_ego
                delta_y = float((prediction.pred_center_ego[1] - display_ref_center[1]).detach().item())
                delta_x = float((prediction.pred_center_ego[0] - display_ref_center[0]).detach().item())
                move_lateral_term, move_longitudinal_term, move_stats = _movement_loss_terms([frame_input], loss_cfg)
                rigid_term, rigid_stats = rigid_loss([frame_input], loss_type=str(loss_cfg.get('rigid_loss_type', 'l2')), size_weight=float(loss_cfg.get('rigid_size_weight', 1.0)), yaw_weight=float(loss_cfg.get('rigid_yaw_weight', 1.0)))
                cls_term, cls_stats = cls_loss([frame_input], pos_weight=float(loss_cfg.get('cls_pos_weight', 1.0)), neg_weight=float(loss_cfg.get('cls_neg_weight', 1.0)), rank_weight=float(loss_cfg.get('cls_rank_weight', 1.0)), rank_margin=float(loss_cfg.get('cls_rank_margin', 0.0)))
                global_rank_term, global_rank_stats = global_query_rank_loss([frame_input], margin=float(loss_cfg.get('cls_global_rank_margin', 0.0)))
                query_identity_term, query_identity_stats = query_identity_loss([frame_input], loss_type=str(loss_cfg.get('query_identity_loss_type', 'cosine')))
                lateral_difference = _movement_difference_sequence([frame_input], loss_cfg)[0]
                if matched_frames_in_sequence == 1 and _progress_child_loss_weight(loss_cfg, 'first_frame_min_weight') > 0.0:
                    first_frame_term = first_frame_min_loss(lateral_difference, min_m=float(loss_cfg.get('first_frame_min_m', 0.0)))
                else:
                    first_frame_term = move_lateral_term.new_zeros(())
                if _progress_child_loss_weight(loss_cfg, 'per_frame_min_weight') > 0.0:
                    per_frame_min_term = first_frame_min_loss(lateral_difference, min_m=float(loss_cfg.get('per_frame_min_m', 0.0)))
                else:
                    per_frame_min_term = move_lateral_term.new_zeros(())
                move_lateral_weight = _move_lateral_loss_weight(loss_cfg, is_final_frame=str(frame.cache_key) == final_frame_cache_key)
                depth_stats: Dict[str, float] = {'loss_depth': 0.0, 'depth_cam_count': 0.0, 'depth_sample_count': 0.0, 'depth_pred_mean_m': 0.0, 'depth_gt_mean_m': 0.0}
                if float(loss_cfg.get('depth_weight', 0.0)) > 0.0 and hasattr(bev_model, 'depth_supervision_loss'):
                    _depth_term, depth_stats = bev_model.depth_supervision_loss(frame=frame, direction=str(loss_cfg.get('depth_direction', 'far')), offset_m=float(loss_cfg.get('depth_offset_m', 2.0)), patch_radius=int(loss_cfg.get('depth_patch_radius', 1)), loss_type=str(loss_cfg.get('depth_loss_type', 'l1')), min_valid_cams=int(loss_cfg.get('depth_min_valid_cams', 1)))
                frame_debug_rows.append({'sequence_name': str(frame.sequence_name), 'frame_id': int(frame.frame_id), 'sample_token': str(frame.sample_token), 'reference_mode': str(loss_reference_mode), 'pred_y_m': float(prediction.pred_center_ego[1].detach().item()), 'gt_y_m': float(display_ref_center[1].detach().item()), 'ref_y_m': float(display_ref_center[1].detach().item()), 'true_gt_y_m': float(frame_input.gt_center_ego[1].detach().item()), 'delta_y_m': delta_y, 'direction': 'left' if delta_y >= 0.0 else 'right', 'shift_abs_m': abs(delta_y), 'pred_x_m': float(prediction.pred_center_ego[0].detach().item()), 'gt_x_m': float(display_ref_center[0].detach().item()), 'ref_x_m': float(display_ref_center[0].detach().item()), 'true_gt_x_m': float(frame_input.gt_center_ego[0].detach().item()), 'delta_x_m': delta_x, 'x_direction': 'front' if delta_x >= 0.0 else 'back', 'x_shift_abs_m': abs(delta_x), 'target_detection_name': str(prediction.target_detection_name), 'target_confidence': float(torch.sigmoid(prediction.target_logit.detach()).item()), 'move_lateral_loss': float(move_lateral_term.detach().item()), 'move_lateral_weight': float(move_lateral_weight), 'move_longitudinal_loss': float(move_longitudinal_term.detach().item()), 'move_longitudinal_hard_loss': 0.0, 'xhard_excess_m': 0.0, 'first_frame_min_loss': float(first_frame_term.detach().item()), 'per_frame_min_loss': float(per_frame_min_term.detach().item()), 'difference': float(lateral_difference.detach().item()), 'target_lateral_move_m': float(lateral_difference.detach().item()), 'longitudinal_difference': abs(delta_x), 'rigid_loss': float(rigid_term.detach().item()), 'cls_loss': float(cls_term.detach().item()), 'cls_nearby_loss': 0.0, 'cls_global_rank_loss': float(global_rank_stats.get('loss_global_query_rank', 0.0)), 'cls_output_proxy_loss': 0.0, 'cls_query_identity_loss': float(query_identity_stats.get('loss_query_identity', 0.0)), 'nearby_query_count': 0, 'depth_loss': float(depth_stats.get('loss_depth', 0.0)), 'depth_cam_count': int(round(float(depth_stats.get('depth_cam_count', 0.0)))), 'depth_sample_count': int(round(float(depth_stats.get('depth_sample_count', 0.0)))), 'depth_pred_mean_m': float(depth_stats.get('depth_pred_mean_m', 0.0)), 'depth_gt_mean_m': float(depth_stats.get('depth_gt_mean_m', 0.0))})
                move_lateral_term_values.append(float(move_lateral_term.detach().item()))
                move_lateral_weight_values.append(float(move_lateral_weight))
                move_lateral_weighted_values.append(float((move_lateral_term.detach() * move_lateral_term.new_tensor(move_lateral_weight)).item()))
                move_longitudinal_term_values.append(float(move_longitudinal_term.detach().item()))
                move_longitudinal_mean_values.append(float(move_stats.get('loss_move_longitudinal_mean', 0.0)))
                move_longitudinal_max_values.append(float(move_stats.get('loss_move_longitudinal_max', 0.0)))
                move_longitudinal_hard_values.append(0.0)
                move_longitudinal_hard_excess_values.append(0.0)
                move_longitudinal_hard_excess_max_values.append(0.0)
                lateral_difference_values.append(float(move_stats.get('difference_mean', 0.0)))
                longitudinal_difference_values.append(float(move_stats.get('longitudinal_difference_mean', 0.0)))
                longitudinal_difference_max_values.append(float(move_stats.get('longitudinal_difference_max', 0.0)))
                first_frame_min_term_values.append(float(first_frame_term.detach().item()))
                per_frame_min_term_values.append(float(per_frame_min_term.detach().item()))
                rigid_term_values.append(float(rigid_term.detach().item()))
                rigid_size_loss_values.append(float(rigid_stats.get('loss_rigid_size', 0.0)))
                rigid_yaw_loss_values.append(float(rigid_stats.get('loss_rigid_yaw', 0.0)))
                rigid_size_diff_values.append(float(rigid_stats.get('rigid_size_diff_mean', 0.0)))
                rigid_yaw_diff_values.append(float(rigid_stats.get('rigid_yaw_diff_mean', 0.0)))
                cls_term_values.append(float(cls_term.detach().item()))
                cls_pos_values.append(float(cls_stats.get('loss_cls_pos', 0.0)))
                cls_neg_values.append(float(cls_stats.get('loss_cls_neg', 0.0)))
                cls_rank_values.append(float(cls_stats.get('loss_cls_rank', 0.0)))
                cls_nearby_values.append(float(cls_stats.get('loss_cls_nearby', 0.0)))
                cls_global_rank_values.append(float(global_rank_stats.get('loss_global_query_rank', 0.0)))
                cls_output_proxy_values.append(0.0)
                cls_query_identity_values.append(float(query_identity_stats.get('loss_query_identity', 0.0)))
                target_logit_values.append(float(cls_stats.get('target_logit', 0.0)))
                target_confidence_values.append(float(cls_stats.get('target_confidence', 0.0)))
                noncar_max_conf_values.append(float(cls_stats.get('noncar_max_confidence', 0.0)))
                nearby_target_max_conf_values.append(float(cls_stats.get('nearby_target_max_confidence', 0.0)))
                nearby_query_count_values.append(float(cls_stats.get('nearby_query_count', 0.0)))
                global_other_target_max_conf_values.append(float(global_rank_stats.get('global_other_target_max_confidence', 0.0)))
                other_query_global_max_conf_values.append(0.0)
                query_identity_cosine_values.append(float(query_identity_stats.get('query_identity_cosine', 1.0)))
                depth_term_values.append(float(depth_stats.get('loss_depth', 0.0)))
                depth_cam_count_values.append(float(depth_stats.get('depth_cam_count', 0.0)))
                depth_sample_count_values.append(float(depth_stats.get('depth_sample_count', 0.0)))
                depth_pred_mean_values.append(float(depth_stats.get('depth_pred_mean_m', 0.0)))
                depth_gt_mean_values.append(float(depth_stats.get('depth_gt_mean_m', 0.0)))
                del prediction
                del outs
                del camera_images
                bev_model.last_bbox_tensor = None
                bev_model.last_cls_tensor = None
                if hasattr(bev_model, 'last_head'):
                    bev_model.last_head = None
                if hasattr(bev_model, 'last_heatmap_tensor'):
                    bev_model.last_heatmap_tensor = None
                if hasattr(bev_model, 'last_heatmap_grad'):
                    bev_model.last_heatmap_grad = None
                if hasattr(bev_model, 'last_query_feature_tensor'):
                    bev_model.last_query_feature_tensor = None
            for prev_input, curr_input in zip(sequence_inputs[:-1], sequence_inputs[1:]):
                pair_deviations = _movement_difference_sequence([prev_input, curr_input], loss_cfg)
                progress_term, progress_stats, progress_gains = progress_loss(pair_deviations, step_size_m=float(loss_cfg.get('progress_step_size_m', 0.05)), decay_lambda=float(loss_cfg.get('progress_lambda', 1.0)), detach_previous=bool(loss_cfg.get('progress_detach_previous', False)), loss_type=str(loss_cfg.get('progress_step_loss_type', 'l2')))
                progress_term_values.append(float(progress_term.detach().item()))
                progress_teacher_values.append(float(progress_stats.get('progress_teacher', 0.0)))
                progress_floor_values.append(float(progress_stats.get('progress_floor', 0.0)))
                progress_gain_values.extend([float(v) for v in progress_gains if v is not None])
    weighted_progress_value = float(loss_cfg.get('progress_weight', 1.0)) * _mean_or_zero(progress_term_values)
    weighted_move_lateral_value = _mean_or_zero(move_lateral_weighted_values)
    weighted_first_frame_min_value = _progress_child_loss_weight(loss_cfg, 'first_frame_min_weight') * _mean_or_zero(first_frame_min_term_values)
    weighted_per_frame_min_value = _progress_child_loss_weight(loss_cfg, 'per_frame_min_weight') * _mean_or_zero(per_frame_min_term_values)
    weighted_rigid_value = float(loss_cfg.get('rigid_weight', 1.0)) * _mean_or_zero(rigid_term_values)
    weighted_cls_value = float(loss_cfg.get('cls_weight', 1.0)) * _mean_or_zero(cls_term_values)
    weighted_cls_global_rank_value = _cls_child_loss_weight(loss_cfg, 'cls_global_rank_weight') * _mean_or_zero(cls_global_rank_values)
    weighted_query_identity_value = _cls_child_loss_weight(loss_cfg, 'query_identity_weight') * _mean_or_zero(cls_query_identity_values)
    weighted_depth_value = float(loss_cfg.get('depth_weight', 0.0)) * _mean_or_zero(depth_term_values)
    model_loss_value = weighted_progress_value + weighted_move_lateral_value + weighted_first_frame_min_value + weighted_per_frame_min_value + weighted_rigid_value + weighted_cls_value + weighted_cls_global_rank_value + weighted_query_identity_value + weighted_depth_value
    min_target_confidence_value = min(target_confidence_values) if target_confidence_values else 0.0
    record = {'enable_query_terms': bool(enable_query_terms), 'loss_total': float(model_loss_value), 'loss_model': float(model_loss_value), 'matched_frames': matched_frames_step, 'target_lost_frames': int(target_lost_frames_step), 'target_expected_frames': int(matched_frames_step + target_lost_frames_step), 'target_confidence': _mean_or_zero(target_confidence_values), 'target_confidence_min': float(min_target_confidence_value), 'target_logit': _mean_or_zero(target_logit_values), 'progress_gain_mean': _mean_or_zero(progress_gain_values), 'move_lateral': _mean_or_zero(move_lateral_term_values), 'move_lateral_weight_mean': _mean_or_zero(move_lateral_weight_values), 'weighted_loss_move_lateral': _mean_or_zero(move_lateral_weighted_values), 'move_longitudinal': _mean_or_zero(move_longitudinal_term_values), 'loss_move_longitudinal_mean': _mean_or_zero(move_longitudinal_mean_values), 'loss_move_longitudinal_max': _mean_or_zero(move_longitudinal_max_values), 'loss_progress': _mean_or_zero(progress_term_values), 'loss_move_lateral': _mean_or_zero(move_lateral_term_values), 'loss_move_longitudinal': _mean_or_zero(move_longitudinal_term_values), 'loss_move_longitudinal_hard': _mean_or_zero(move_longitudinal_hard_values), 'longitudinal_hard_excess_mean': _mean_or_zero(move_longitudinal_hard_excess_values), 'longitudinal_hard_excess_max': _mean_or_zero(move_longitudinal_hard_excess_max_values), 'loss_first_frame_min': _mean_or_zero(first_frame_min_term_values), 'loss_per_frame_min': _mean_or_zero(per_frame_min_term_values), 'loss_rigid': _mean_or_zero(rigid_term_values), 'loss_rigid_size': _mean_or_zero(rigid_size_loss_values), 'loss_rigid_yaw': _mean_or_zero(rigid_yaw_loss_values), 'loss_cls': _mean_or_zero(cls_term_values), 'loss_cls_pos': _mean_or_zero(cls_pos_values), 'loss_cls_neg': _mean_or_zero(cls_neg_values), 'loss_cls_rank': _mean_or_zero(cls_rank_values), 'loss_cls_nearby': _mean_or_zero(cls_nearby_values), 'loss_cls_global_rank': _mean_or_zero(cls_global_rank_values), 'loss_cls_output_proxy': _mean_or_zero(cls_output_proxy_values), 'loss_query_identity': _mean_or_zero(cls_query_identity_values), 'loss_depth': _mean_or_zero(depth_term_values), 'depth_cam_count': _mean_or_zero(depth_cam_count_values), 'depth_sample_count': _mean_or_zero(depth_sample_count_values), 'depth_pred_mean_m': _mean_or_zero(depth_pred_mean_values), 'depth_gt_mean_m': _mean_or_zero(depth_gt_mean_values), 'noncar_max_confidence': _mean_or_zero(noncar_max_conf_values), 'nearby_target_max_confidence': _mean_or_zero(nearby_target_max_conf_values), 'nearby_query_count': _mean_or_zero(nearby_query_count_values), 'global_other_target_max_confidence': _mean_or_zero(global_other_target_max_conf_values), 'other_query_global_max_confidence': _mean_or_zero(other_query_global_max_conf_values), 'query_identity_cosine': _mean_or_zero(query_identity_cosine_values), 'progress_teacher': _mean_or_zero(progress_teacher_values), 'progress_floor': _mean_or_zero(progress_floor_values), 'difference_mean': _mean_or_zero(lateral_difference_values), 'longitudinal_difference_mean': _mean_or_zero(longitudinal_difference_values), 'longitudinal_difference_max': _mean_or_zero(longitudinal_difference_max_values), 'rigid_size_diff_mean': _mean_or_zero(rigid_size_diff_values), 'rigid_yaw_diff_mean': _mean_or_zero(rigid_yaw_diff_values)}
    return (record, frame_debug_rows)

def _run_fast_final_decode_val_target_check(*, frame_groups: Sequence[Sequence[FrameRecord]], renderer: FixedUVTextureRenderer, bev_model: BevFormerGradientModel, device: torch.device, use_amp: bool, amp_dtype: torch.dtype, apply_eot: bool, loss_reference_mode: str, clean_detection_refs: Optional[Dict[str, Dict[str, torch.Tensor]]], conf_threshold: float, max_center_dist_m: float, distance_axis: str, max_cross_axis_dist_m: float, export_label: str, training_log_path: Path) -> List[Dict[str, Any]]:
    if not hasattr(bev_model, 'match_target_query_from_final_outputs'):
        raise RuntimeError('final decode val target check requires match_target_query_from_final_outputs')
    records: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for sequence_frames in frame_groups:
            if not sequence_frames:
                continue
            sequence_name = str(sequence_frames[0].sequence_name)
            rows: List[Dict[str, Any]] = []
            matched = 0
            lost = 0
            prev_bev: Optional[torch.Tensor] = None
            prev_scene_token: Optional[str] = None
            prev_abs_pos: Optional[np.ndarray] = None
            prev_abs_angle: Optional[float] = None
            for frame in sequence_frames:
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    camera_images = renderer.build_frame_images(frame, apply_eot=apply_eot)
                    outs, prev_bev, prev_abs_pos, prev_abs_angle = bev_model.forward_frame(frame, camera_images, prev_bev=prev_bev, prev_scene_token=prev_scene_token, prev_abs_pos=prev_abs_pos, prev_abs_angle=prev_abs_angle, retain_grad=False)
                if isinstance(prev_bev, torch.Tensor):
                    prev_bev = prev_bev.detach()
                prev_scene_token = frame.scene_token
                query_match = bev_model.match_target_query_from_final_outputs(frame, outs, conf_threshold=float(conf_threshold), max_center_dist_m=float(max_center_dist_m), distance_axis=str(distance_axis), max_cross_axis_dist_m=float(max_cross_axis_dist_m))
                if query_match is None or not query_match.matched:
                    lost += 1
                    rows.append({'matched': False, 'sequence_name': sequence_name, 'frame_id': int(frame.frame_id), 'sample_token': str(frame.sample_token), 'target_detection_name': str(query_match.target_detection_name) if query_match is not None else str(frame.gt_category_name), 'unmatched_reason': str(query_match.unmatched_reason) if query_match is not None else 'missing_match', 'candidate_total': int(query_match.candidate_total) if query_match is not None else 0, 'candidate_after_conf': int(query_match.candidate_after_conf) if query_match is not None else 0, 'candidate_after_dist': int(query_match.candidate_after_dist) if query_match is not None else 0})
                    del outs
                    del camera_images
                    _clear_model_runtime_tensors(bev_model)
                    continue
                matched += 1
                prediction = bev_model.target_query_prediction(frame, outs, query_idx=int(query_match.query_idx))
                ref_center_ego, _ref_size_wlh, _ref_yaw = _loss_reference_tensors_for_frame(frame=frame, prediction=prediction, loss_reference_mode=loss_reference_mode, clean_detection_refs=clean_detection_refs)
                reference_mode = str(loss_reference_mode).strip().lower()
                reference_source = 'gt'
                if reference_mode == 'clean':
                    if ref_center_ego is None:
                        lost += 1
                        rows.append({'matched': False, 'sequence_name': sequence_name, 'frame_id': int(frame.frame_id), 'sample_token': str(frame.sample_token), 'target_detection_name': str(prediction.target_detection_name), 'unmatched_reason': 'clean_reference_missing', 'candidate_total': int(query_match.candidate_total), 'candidate_after_conf': int(query_match.candidate_after_conf), 'candidate_after_dist': int(query_match.candidate_after_dist), 'reference_source': 'clean_missing'})
                        matched -= 1
                        del prediction
                        del outs
                        del camera_images
                        _clear_model_runtime_tensors(bev_model)
                        continue
                    display_ref_center = ref_center_ego
                    reference_source = 'clean'
                else:
                    display_ref_center = prediction.gt_center_ego
                delta_y = float((prediction.pred_center_ego[1] - display_ref_center[1]).detach().item())
                delta_x = float((prediction.pred_center_ego[0] - display_ref_center[0]).detach().item())
                rows.append({'matched': True, 'sequence_name': sequence_name, 'frame_id': int(frame.frame_id), 'sample_token': str(frame.sample_token), 'query_idx': int(query_match.query_idx), 'target_detection_name': str(prediction.target_detection_name), 'target_confidence': float(torch.sigmoid(prediction.target_logit.detach()).item()), 'direction': 'left' if delta_y >= 0.0 else 'right', 'shift_abs_m': abs(delta_y), 'x_direction': 'front' if delta_x >= 0.0 else 'back', 'x_shift_abs_m': abs(delta_x), 'reference_source': reference_source})
                del prediction
                del outs
                del camera_images
                _clear_model_runtime_tensors(bev_model)
            header = f'[val] official target check {export_label} seq={sequence_name} matched={matched}/{len(sequence_frames)} lost={lost}'
            print(header)
            _append_log_line(training_log_path, header)
            for item in rows:
                if bool(item.get('matched', False)):
                    line = f"[val]   seq={item['sequence_name']} frame={item['frame_id']} cls={item['target_detection_name']} ref={item.get('reference_source', 'gt')} shift={item['direction']}:{item['shift_abs_m']:.4f}m x={item['x_direction']}:{item['x_shift_abs_m']:.4f}m conf={item['target_confidence']:.4f} query={item['query_idx']}"
                else:
                    line = f"[val]   official-target-lost seq={item['sequence_name']} frame={item['frame_id']} reason={item.get('unmatched_reason', '')} candidates={item.get('candidate_total', 0)} after_conf={item.get('candidate_after_conf', 0)} after_dist={item.get('candidate_after_dist', 0)}"
                print(line)
                _append_log_line(training_log_path, line)
            records.append({'step': export_label, 'sequence_name': sequence_name, 'mode': 'fast_final_decode', 'total': int(len(sequence_frames)), 'matched': int(matched), 'lost': int(lost), 'rows': rows})
    return records

def _log_dataset_eval_record(*, prefix: str, step: int, record: Dict[str, Any], frame_debug_rows: Sequence[Dict[str, Any]], training_log_path: Path) -> None:
    enable_query_terms = bool(record.get('enable_query_terms', True))
    expected_frames = int(record.get('target_expected_frames', record.get('matched_frames', 0) + record.get('target_lost_frames', 0)))
    step_header = f"{prefix} step={step:04d} loss={record['loss_total']:.6f} matched={record['matched_frames']}/{expected_frames} lost={int(record.get('target_lost_frames', 0))} conf={record['target_confidence']:.4f} minconf={record['target_confidence_min']:.4f} move={record['loss_move_lateral']:.6f} cross={record['loss_move_longitudinal']:.6f} progress={record['loss_progress']:.6f} first={record['loss_first_frame_min']:.6f} rigid={record['loss_rigid']:.6f} cls={record['loss_cls']:.6f} depth={float(record.get('loss_depth', 0.0)):.6f}"
    print(step_header)
    _append_log_line(training_log_path, step_header)
    subloss_line = f"{prefix}   sub progress_teacher={record['progress_teacher']:.6f} progress_floor={record['progress_floor']:.6f} move_w_mean={float(record.get('move_lateral_weight_mean', 0.0)):.3f} move_weighted={float(record.get('weighted_loss_move_lateral', 0.0)):.6f} cross_mean={record['loss_move_longitudinal_mean']:.6f} cross_max={record['loss_move_longitudinal_max']:.6f} xdrift_mean={record['longitudinal_difference_mean']:.6f} xdrift_max={record['longitudinal_difference_max']:.6f} rigid_size={record['loss_rigid_size']:.6f} rigid_yaw={record['loss_rigid_yaw']:.6f} cls_pos={record['loss_cls_pos']:.6f} cls_neg={record['loss_cls_neg']:.6f} cls_rank={record['loss_cls_rank']:.6f} depth_cams={float(record.get('depth_cam_count', 0.0)):.2f} depth_samples={float(record.get('depth_sample_count', 0.0)):.2f} depth_pred={float(record.get('depth_pred_mean_m', 0.0)):.3f} depth_gt={float(record.get('depth_gt_mean_m', 0.0)):.3f}"
    if enable_query_terms:
        subloss_line = subloss_line + f" cls_global={record['loss_cls_global_rank']:.6f} " + f"qid={record['loss_query_identity']:.6f} " + f"other_t_conf={record['global_other_target_max_confidence']:.6f} " + f"other_any_conf={record['other_query_global_max_confidence']:.6f} " + f"qid_cos={record['query_identity_cosine']:.6f}"
    print(subloss_line)
    _append_log_line(training_log_path, subloss_line)
    for row in frame_debug_rows:
        if not bool(row.get('matched', True)):
            frame_line = f"{prefix}   seq={row['sequence_name']} frame={row['frame_id']} cls={row['target_detection_name']} target_lost=True reason={row.get('unmatched_reason', '')} candidates={int(row.get('candidate_total', 0))} after_conf={int(row.get('candidate_after_conf', 0))} after_dist={int(row.get('candidate_after_dist', 0))}"
            print(frame_line)
            _append_log_line(training_log_path, frame_line)
            continue
        frame_line = f"{prefix}   seq={row['sequence_name']} frame={row['frame_id']} cls={row['target_detection_name']} shift={row['direction']}:{row['shift_abs_m']:.4f}m x={row['x_direction']}:{row['x_shift_abs_m']:.4f}m pred_y={row['pred_y_m']:.4f} ref_y={row.get('ref_y_m', row['gt_y_m']):.4f} pred_x={row['pred_x_m']:.4f} ref_x={row.get('ref_x_m', row['gt_x_m']):.4f} cross={row['move_longitudinal_loss']:.6f} conf={row['target_confidence']:.4f} dloss={float(row.get('depth_loss', 0.0)):.6f}"
        if enable_query_terms:
            frame_line = frame_line + f" cls_global={row.get('cls_global_rank_loss', 0.0):.6f} " + f"qid={row.get('cls_query_identity_loss', 0.0):.6f}"
        print(frame_line)
        _append_log_line(training_log_path, frame_line)

def _format_weighted_loss_formula(*, record: Dict[str, Any], loss_cfg: Dict[str, Any], style_cfg: Dict[str, Any], is_bevdet_model: bool) -> str:

    def _fmt(value: Any) -> str:
        return f'{float(value):.3f}'

    def _add_weighted(parts: List[str], name: str, weight: Any, value: Any) -> None:
        if abs(float(weight)) <= 1e-12:
            return
        parts.append(f'{name}={_fmt(weight)}*{_fmt(value)}')
    parts: List[str] = []
    _add_weighted(parts, 'move', loss_cfg.get('move_center_weight', 1.0), record.get('loss_move_lateral', 0.0))
    _add_weighted(parts, 'progress', loss_cfg.get('progress_weight', 1.0), record.get('loss_progress', 0.0))
    _add_weighted(parts, 'first', _progress_child_loss_weight(loss_cfg, 'first_frame_min_weight'), record.get('loss_first_frame_min', 0.0))
    _add_weighted(parts, 'min', _progress_child_loss_weight(loss_cfg, 'per_frame_min_weight'), record.get('loss_per_frame_min', 0.0))
    _add_weighted(parts, 'rigid', loss_cfg.get('rigid_weight', 1.0), record.get('loss_rigid', 0.0))
    _add_weighted(parts, 'cls', loss_cfg.get('cls_weight', 1.0), record.get('loss_cls', 0.0))
    _add_weighted(parts, 'cls_global', _cls_child_loss_weight(loss_cfg, 'cls_global_rank_weight'), record.get('loss_cls_global_rank', 0.0))
    _add_weighted(parts, 'qid', _cls_child_loss_weight(loss_cfg, 'query_identity_weight'), record.get('loss_query_identity', 0.0))
    _add_weighted(parts, 'depth', loss_cfg.get('depth_weight', 0.0), record.get('loss_depth', 0.0))
    style_weight = float(style_cfg.get('weight', 1.0))
    style_inner_parts: List[str] = []
    if abs(style_weight) > 1e-12:
        _add_weighted(style_inner_parts, 'tv', style_cfg.get('tv_weight', 0.0), record.get('style_tv', 0.0))
        _add_weighted(style_inner_parts, 'l2', style_cfg.get('l2_weight', 0.0), record.get('style_l2', 0.0))
        _add_weighted(style_inner_parts, 'brightness', style_cfg.get('brightness_weight', 0.0), record.get('style_brightness', 0.0))
        _add_weighted(style_inner_parts, 'nps', style_cfg.get('nps_weight', 0.0), record.get('style_nps', 0.0))
    if style_inner_parts:
        parts.append(f'style={_fmt(style_weight)}*(' + ' '.join(style_inner_parts) + ')')
    aux_rows = record.get('auxiliary_models', [])
    if isinstance(aux_rows, list):
        for aux_row in aux_rows:
            if not isinstance(aux_row, dict):
                continue
            aux_weighted = float(aux_row.get('loss_weighted', 0.0))
            if abs(aux_weighted) <= 1e-12:
                continue
            parts.append(f"aux:{aux_row.get('model', '')}={_fmt(aux_row.get('weight', 0.0))}*{_fmt(aux_row.get('loss_model', 0.0))}")
    if parts:
        return f"Total loss={_fmt(record.get('loss_total', 0.0))} " + ' '.join(parts)
    return f"Total loss={_fmt(record.get('loss_total', 0.0))}"

def _finite_float_or_none(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result

def _history_metric_series(records: Sequence[Dict[str, Any]], metric_key: str) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for record in records:
        step = _finite_float_or_none(record.get('step'))
        value = _finite_float_or_none(record.get(metric_key))
        if step is None or value is None:
            continue
        xs.append(step)
        ys.append(value)
    return (xs, ys)

def _eval_history_metric_series(eval_history: Sequence[Dict[str, Any]], metric_key: str) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for entry in eval_history:
        step = _finite_float_or_none(entry.get('step'))
        eval_record = entry.get('eval')
        if step is None or not isinstance(eval_record, dict):
            continue
        value = _finite_float_or_none(eval_record.get(metric_key))
        if value is None:
            continue
        xs.append(step)
        ys.append(value)
    return (xs, ys)

def _val_history_metric_series(val_history: Sequence[Dict[str, Any]], metric_key: str) -> Tuple[List[float], List[float]]:
    return _eval_history_metric_series(val_history, metric_key)

def _final_frame_shift_by_sequence(record: Dict[str, Any]) -> Dict[str, float]:
    rows = record.get('frame_metrics', [])
    if not isinstance(rows, list):
        return {}
    by_sequence: Dict[str, Tuple[int, float]] = {}
    for row in rows:
        if not isinstance(row, dict) or not bool(row.get('matched', True)):
            continue
        shift = _finite_float_or_none(row.get('target_lateral_move_m'))
        if shift is None:
            shift = _finite_float_or_none(row.get('difference'))
        if shift is None:
            continue
        sequence_name = str(row.get('sequence_name', '') or '').strip()
        label = sequence_name if sequence_name else str(row.get('sample_token', '') or 'sample').strip()
        try:
            frame_id = int(row.get('frame_id', -1))
        except (TypeError, ValueError):
            frame_id = -1
        old = by_sequence.get(label)
        if old is None or frame_id >= old[0]:
            by_sequence[label] = (frame_id, shift)
    return {label: shift for label, (_frame_id, shift) in by_sequence.items()}

def _mean_final_frame_shift_series(records: Sequence[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for record in records:
        step = _finite_float_or_none(record.get('step'))
        shifts = _final_frame_shift_by_sequence(record)
        if step is None or not shifts:
            continue
        xs.append(step)
        ys.append(float(sum(shifts.values()) / max(1, len(shifts))))
    return (xs, ys)

def _eval_final_frame_shift_series_by_sequence(eval_history: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Tuple[List[float], List[float]]], Tuple[List[float], List[float]]]:
    by_sequence: Dict[str, Tuple[List[float], List[float]]] = {}
    mean_xs: List[float] = []
    mean_ys: List[float] = []
    for entry in eval_history:
        step = _finite_float_or_none(entry.get('step'))
        eval_record = entry.get('eval')
        if step is None or not isinstance(eval_record, dict):
            continue
        shifts = _final_frame_shift_by_sequence(eval_record)
        if not shifts:
            continue
        mean_xs.append(step)
        mean_ys.append(float(sum(shifts.values()) / max(1, len(shifts))))
        for label, shift in sorted(shifts.items()):
            xs, ys = by_sequence.setdefault(label, ([], []))
            xs.append(step)
            ys.append(shift)
    return (by_sequence, (mean_xs, mean_ys))

def _val_final_frame_shift_series_by_sequence(val_history: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Tuple[List[float], List[float]]], Tuple[List[float], List[float]]]:
    return _eval_final_frame_shift_series_by_sequence(val_history)

def _plot_training_loss_curves(*, history: Sequence[Dict[str, Any]], train_eval_history: Sequence[Dict[str, Any]], val_history: Sequence[Dict[str, Any]], output_path: Path, training_log_path: Path) -> None:
    if bool(getattr(_plot_training_loss_curves, '_disabled', False)):
        return
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = (1500, 950)
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        panels = [(45, 45, 735, 455), (795, 45, 1485, 455), (45, 515, 735, 925), (795, 515, 1485, 925)]

        def _scale_points(xs: Sequence[float], ys: Sequence[float], plot_box: Tuple[int, int, int, int], x_min: float, x_max: float, y_min: float, y_max: float) -> List[Tuple[int, int]]:
            left, top, right, bottom = plot_box
            x_span = max(1e-12, float(x_max) - float(x_min))
            y_span = max(1e-12, float(y_max) - float(y_min))
            return [(int(round(left + (float(x) - x_min) / x_span * (right - left))), int(round(bottom - (float(y) - y_min) / y_span * (bottom - top)))) for x, y in zip(xs, ys)]

        def _draw_panel(*, box: Tuple[int, int, int, int], title: str, ylabel: str, series: Sequence[Tuple[str, Sequence[float], Sequence[float], Tuple[int, int, int]]]) -> None:
            left, top, right, bottom = box
            plot_box = (left + 88, top + 42, right - 18, bottom - 48)
            draw.rectangle([left, top, right, bottom], outline=(205, 205, 205), width=1)
            draw.text((left + 12, top + 10), title, fill=(20, 20, 20))
            draw.text((plot_box[0], bottom - 32), 'step', fill=(80, 80, 80))
            draw.text((left + 12, plot_box[1]), ylabel, fill=(80, 80, 80))
            all_x = [float(x) for _label, xs, _ys, _color in series for x in xs]
            all_y = [float(y) for _label, _xs, ys, _color in series for y in ys]
            if not all_x or not all_y:
                draw.text(((left + right) // 2 - 25, (top + bottom) // 2), 'no data', fill=(120, 120, 120))
                return
            x_min, x_max = (min(all_x), max(all_x))
            y_min, y_max = (min(all_y), max(all_y))
            if abs(x_max - x_min) <= 1e-12:
                x_min -= 1.0
                x_max += 1.0
            if abs(y_max - y_min) <= 1e-12:
                pad = max(1.0, abs(y_max) * 0.1)
                y_min -= pad
                y_max += pad
            else:
                pad = (y_max - y_min) * 0.08
                y_min -= pad
                y_max += pad
            tick_count = 7
            tick_den = float(max(1, tick_count - 1))

            def _format_x_tick(value: float) -> str:
                if abs(value - round(value)) < 1e-06:
                    return f'{value:.0f}'
                if abs(value) < 10.0:
                    return f'{value:.2f}'
                return f'{value:.1f}'

            def _format_y_tick(value: float) -> str:
                abs_value = abs(value)
                if abs_value < 1.0:
                    return f'{value:.4f}'
                if abs_value < 10.0:
                    return f'{value:.3f}'
                if abs_value < 100.0:
                    return f'{value:.2f}'
                if abs_value < 1000.0:
                    return f'{value:.1f}'
                return f'{value:.4g}'
            for idx in range(tick_count):
                gx = int(round(plot_box[0] + idx * (plot_box[2] - plot_box[0]) / tick_den))
                gy = int(round(plot_box[3] - idx * (plot_box[3] - plot_box[1]) / tick_den))
                draw.line([(gx, plot_box[1]), (gx, plot_box[3])], fill=(235, 235, 235))
                draw.line([(plot_box[0], gy), (plot_box[2], gy)], fill=(235, 235, 235))
            if y_min < 0.0 < y_max:
                zero_y = int(round(plot_box[3] - (0.0 - y_min) / (y_max - y_min) * (plot_box[3] - plot_box[1])))
                draw.line([(plot_box[0], zero_y), (plot_box[2], zero_y)], fill=(120, 120, 120), width=2)
            draw.rectangle(plot_box, outline=(80, 80, 80), width=1)
            for idx in range(tick_count):
                tick_value = x_min + idx * (x_max - x_min) / tick_den
                tick_x = int(round(plot_box[0] + idx * (plot_box[2] - plot_box[0]) / tick_den))
                tick_label = _format_x_tick(tick_value)
                try:
                    tick_bbox = draw.textbbox((0, 0), tick_label)
                    tick_width = tick_bbox[2] - tick_bbox[0]
                except Exception:
                    tick_width = len(tick_label) * 6
                draw.line([(tick_x, plot_box[3]), (tick_x, plot_box[3] + 4)], fill=(80, 80, 80))
                label_x = max(plot_box[0], min(plot_box[2] - tick_width, tick_x - tick_width // 2))
                draw.text((label_x, plot_box[3] + 8), tick_label, fill=(90, 90, 90))
            for idx in range(tick_count):
                tick_value = y_min + idx * (y_max - y_min) / tick_den
                tick_y = int(round(plot_box[3] - idx * (plot_box[3] - plot_box[1]) / tick_den))
                tick_label = _format_y_tick(tick_value)
                try:
                    tick_bbox = draw.textbbox((0, 0), tick_label)
                    tick_width = tick_bbox[2] - tick_bbox[0]
                except Exception:
                    tick_width = len(tick_label) * 6
                draw.line([(plot_box[0] - 4, tick_y), (plot_box[0], tick_y)], fill=(80, 80, 80))
                draw.text((plot_box[0] - tick_width - 8, tick_y - 6), tick_label, fill=(90, 90, 90))
            legend_x = plot_box[0] + 8
            legend_y = plot_box[1] + 8
            legend_col_width = 155
            for idx, (label, xs, ys, color) in enumerate(series):
                if not xs or not ys:
                    continue
                points = _scale_points(xs, ys, plot_box, x_min, x_max, y_min, y_max)
                if len(points) == 1:
                    px, py = points[0]
                    draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=color)
                else:
                    draw.line(points, fill=color, width=2)
                lx = legend_x + idx % 3 * legend_col_width
                ly = legend_y + idx // 3 * 16
                if ly < plot_box[3] - 16:
                    draw.line([(lx, ly + 6), (lx + 18, ly + 6)], fill=color, width=2)
                    draw.text((lx + 24, ly), str(label)[:22], fill=(35, 35, 35))

        def _train_val_series(metric_key: str) -> List[Tuple[str, List[float], List[float], Tuple[int, int, int]]]:
            train_batch_x, train_batch_y = _history_metric_series(history, metric_key)
            train_full_x, train_full_y = _eval_history_metric_series(train_eval_history, metric_key)
            val_x, val_y = _eval_history_metric_series(val_history, metric_key)
            return [('train batch', train_batch_x, train_batch_y, (255, 150, 150)), ('train full', train_full_x, train_full_y, (180, 0, 0)), ('val full', val_x, val_y, (0, 150, 0))]
        _draw_panel(box=panels[0], title='Total loss', ylabel='loss', series=_train_val_series('loss_total'))
        _draw_panel(box=panels[1], title='Move loss', ylabel='loss', series=_train_val_series('loss_move_lateral'))
        _draw_panel(box=panels[2], title='Progress loss', ylabel='loss', series=_train_val_series('loss_progress'))
        shift_series: List[Tuple[str, List[float], List[float], Tuple[int, int, int]]] = []
        _train_batch_x, _train_batch_y = _mean_final_frame_shift_series(history)
        train_by_sequence, (train_mean_x, train_mean_y) = _eval_final_frame_shift_series_by_sequence(train_eval_history)
        if train_mean_x:
            shift_series.append(('train full mean', train_mean_x, train_mean_y, (180, 0, 0)))
        val_by_sequence, (val_mean_x, val_mean_y) = _eval_final_frame_shift_series_by_sequence(val_history)
        if val_mean_x:
            shift_series.append(('val final mean', val_mean_x, val_mean_y, (0, 150, 0)))
        palette = [(31, 119, 180), (255, 127, 14), (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207), (44, 160, 44), (214, 39, 40)]
        for idx, (label, (xs, ys)) in enumerate(sorted(val_by_sequence.items())):
            shift_series.append((f'val:{label}', xs, ys, palette[idx % len(palette)]))
        _draw_panel(box=panels[3], title='Final-frame target shift', ylabel='target_move_m', series=shift_series)
        image.save(str(output_path))
    except Exception as exc:
        setattr(_plot_training_loss_curves, '_disabled', True)
        if not bool(getattr(_plot_training_loss_curves, '_warned', False)):
            setattr(_plot_training_loss_curves, '_warned', True)
            _append_log_line(training_log_path, f'[train] loss curve plot failed; skipping further plots: {type(exc).__name__}: {exc}')

def _export_image_tree(*, frames: Sequence[FrameRecord], output_dir: Path, export_subdir: str, image_source_subdir: str, image_provider: Any, lidar_paths_by_sample: Optional[Dict[str, Path]]=None) -> Path:
    image_root = output_dir / export_subdir / 'images'
    sample_root = image_root / image_source_subdir
    for frame in frames:
        frame_images = image_provider(frame)
        for channel in CAMERA_CHANNELS:
            camera = frame.cameras[channel]
            dst_path = _patched_camera_image_path(sample_root, channel, camera)
            _save_tensor_image_png(dst_path, frame_images[channel])
            alias_path = sample_root / channel / camera.image_path.name
            _ensure_lossless_original_name_alias(dst_path, alias_path)
        if lidar_paths_by_sample is not None:
            lidar_path = lidar_paths_by_sample.get(frame.cache_key)
            if lidar_path is not None and lidar_path.exists():
                dst_path = sample_root / 'LIDAR_TOP' / lidar_path.name
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                if not dst_path.exists():
                    try:
                        dst_path.symlink_to(lidar_path)
                    except OSError:
                        shutil.copy2(lidar_path, dst_path)
    return image_root

def _load_sequence_ann_payload(sequence_pkl: Path) -> object:
    with sequence_pkl.open('rb') as fp:
        return pickle.load(fp)

def _build_official_attacked_payload(*, frames: Sequence[FrameRecord], ann_payload_template: object, info_by_token: Dict[str, Dict[str, Any]], patched_paths: Dict[str, Dict[str, str]]) -> object:
    subset_infos: List[Dict[str, Any]] = []
    for frame in frames:
        info = copy.deepcopy(info_by_token[frame.sample_token])
        for channel, patched_path in patched_paths.get(frame.sample_token, {}).items():
            info['cams'][channel]['data_path'] = patched_path
        if 'ann_infos' not in info:
            info['ann_infos'] = (np.zeros((0, 9), dtype=np.float32), np.zeros((0,), dtype=np.int64))
        if 'gt_boxes' not in info:
            info['gt_boxes'] = np.zeros((0, 9), dtype=np.float32)
        if 'gt_names' not in info:
            info['gt_names'] = np.asarray([], dtype=object)
        if 'num_lidar_pts' not in info:
            info['num_lidar_pts'] = np.zeros((0,), dtype=np.int64)
        if 'gt_velocity' not in info:
            info['gt_velocity'] = np.zeros((0, 2), dtype=np.float32)
        if 'valid_flag' not in info:
            info['valid_flag'] = np.zeros((0,), dtype=bool)
        for cam_info in info.get('cams', {}).values():
            if not isinstance(cam_info, dict):
                continue
            if 'cam_intrinsic' in cam_info:
                cam_info['cam_intrinsic'] = np.asarray(cam_info['cam_intrinsic'], dtype=np.float32).reshape(3, 3)
            if 'sensor2lidar_rotation' in cam_info:
                cam_info['sensor2lidar_rotation'] = np.asarray(cam_info['sensor2lidar_rotation'], dtype=np.float32).reshape(3, 3)
            if 'sensor2lidar_translation' in cam_info:
                cam_info['sensor2lidar_translation'] = np.asarray(cam_info['sensor2lidar_translation'], dtype=np.float32).reshape(3)
        subset_infos.append(info)
    if isinstance(ann_payload_template, dict) and 'infos' in ann_payload_template:
        payload = copy.deepcopy(ann_payload_template)
        payload['infos'] = subset_infos
        return payload
    return subset_infos

def _write_official_attacked_ann_file(*, attacked_ann_file: Path, frames: Sequence[FrameRecord], ann_payload_template: object, info_by_token: Dict[str, Dict[str, Any]], patched_paths: Dict[str, Dict[str, str]]) -> None:
    payload = _build_official_attacked_payload(frames=frames, ann_payload_template=ann_payload_template, info_by_token=info_by_token, patched_paths=patched_paths)
    attacked_ann_file.parent.mkdir(parents=True, exist_ok=True)
    with attacked_ann_file.open('wb') as fp:
        pickle.dump(payload, fp)

def _yaw_world_to_quaternion_wxyz(yaw_world: float) -> List[float]:
    half = 0.5 * float(yaw_world)
    return [float(math.cos(half)), 0.0, 0.0, float(math.sin(half))]

def _default_attribute_name_for_detection(detection_name: str) -> str:
    name = str(detection_name or '').strip()
    if name == 'car':
        return 'vehicle.parked'
    if name == 'bus':
        return 'vehicle.stopped'
    if name == 'pedestrian':
        return 'pedestrian.standing'
    if name in {'bicycle', 'motorcycle'}:
        return 'cycle.without_rider'
    return ''

def _build_fixed_query_record(*, frame: FrameRecord, prediction: Any, query_idx: int, attribute_name: Optional[str]=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pred_box_world = prediction.pred_box_world.detach().cpu()
    translation = [float(v) for v in pred_box_world[0:3].tolist()]
    size = [float(v) for v in pred_box_world[3:6].tolist()]
    yaw_world = float(pred_box_world[6].item())
    detection_name = str(prediction.target_detection_name)
    detection_score = float(torch.sigmoid(prediction.target_logit.detach()).item())
    record = {'sample_token': str(frame.sample_token), 'translation': translation, 'size': size, 'rotation': _yaw_world_to_quaternion_wxyz(yaw_world), 'velocity': [0.0, 0.0], 'detection_name': detection_name, 'detection_score': detection_score, 'attribute_name': str(attribute_name if attribute_name is not None else _default_attribute_name_for_detection(detection_name))}
    trace_record = dict(record)
    trace_record['query_idx'] = int(query_idx)
    return (record, trace_record)

def _build_target_movement_metrics(*, frame: FrameRecord, prediction: Any, query_idx: int) -> Dict[str, Any]:
    gt_center_ego = prediction.gt_center_ego.detach().cpu()
    pred_center_ego = prediction.pred_center_ego.detach().cpu()
    gt_x_m = float(gt_center_ego[0].item())
    gt_y_m = float(gt_center_ego[1].item())
    gt_z_m = float(gt_center_ego[2].item())
    pred_x_m = float(pred_center_ego[0].item())
    pred_y_m = float(pred_center_ego[1].item())
    pred_z_m = float(pred_center_ego[2].item())
    gt_lateral_abs_m = abs(gt_y_m)
    pred_lateral_abs_m = abs(pred_y_m)
    moved_toward_y0_m = float(gt_lateral_abs_m - pred_lateral_abs_m)
    moved_toward_y0_ratio = None
    moved_toward_y0_pct = None
    if gt_lateral_abs_m > 1e-06:
        moved_toward_y0_ratio = float(moved_toward_y0_m / gt_lateral_abs_m)
        moved_toward_y0_pct = float(moved_toward_y0_ratio * 100.0)
    return {'sample_token': str(frame.sample_token), 'frame_id': int(frame.frame_id), 'query_idx': int(query_idx), 'gt_center_ego_m': [gt_x_m, gt_y_m, gt_z_m], 'pred_center_ego_m': [pred_x_m, pred_y_m, pred_z_m], 'gt_lateral_offset_to_y0_m': float(gt_lateral_abs_m), 'pred_lateral_offset_to_y0_m': float(pred_lateral_abs_m), 'moved_toward_y0_m': float(moved_toward_y0_m), 'moved_toward_y0_ratio': moved_toward_y0_ratio, 'moved_toward_y0_pct': moved_toward_y0_pct, 'gt_center_world_m': [float(v) for v in frame.gt_center_world.tolist()], 'pred_center_world_m': [float(v) for v in prediction.pred_box_world.detach().cpu()[0:3].tolist()], 'target_detection_name': str(prediction.target_detection_name), 'target_logit': float(prediction.target_logit.detach().cpu().item()), 'target_confidence': float(torch.sigmoid(prediction.target_logit.detach()).cpu().item())}

def _patched_camera_image_path(sample_root: Path, channel: str, camera: CameraRecord) -> Path:
    return sample_root / channel / f'{camera.image_path.stem}.png'

def _ensure_lossless_original_name_alias(png_path: Path, original_path: Path) -> None:
    if original_path == png_path or original_path.exists():
        return
    original_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        original_path.symlink_to(png_path)
    except OSError:
        shutil.copy2(png_path, original_path)

def _find_target_replacement_index(*, rows: Sequence[Dict[str, Any]], target_detection_name: str, gt_center_world: np.ndarray) -> int:
    best_index = -1
    best_dist = float('inf')
    target_name = str(target_detection_name)
    gt_xy = np.asarray(gt_center_world[:2], dtype=np.float32).reshape(2)
    for index, row in enumerate(rows):
        if str(row.get('detection_name', '')) != target_name:
            continue
        translation = row.get('translation', None)
        if not isinstance(translation, (list, tuple)) or len(translation) < 2:
            continue
        row_xy = np.asarray([float(translation[0]), float(translation[1])], dtype=np.float32)
        dist = float(np.linalg.norm(row_xy - gt_xy))
        if dist < best_dist:
            best_dist = dist
            best_index = index
    return best_index

def _collect_fixed_query_override_rows(*, frames: Sequence[FrameRecord], renderer: 'FixedUVTextureRenderer', bev_model: BevFormerGradientModel, fixed_queries: Dict[str, FixedQueryMatch], device: torch.device, use_amp: bool, amp_dtype: torch.dtype) -> Dict[str, Dict[str, Any]]:
    payload: Dict[str, Dict[str, Any]] = {}
    prev_bev: Optional[torch.Tensor] = None
    prev_scene_token: Optional[str] = None
    prev_abs_pos: Optional[np.ndarray] = None
    prev_abs_angle: Optional[float] = None
    with torch.inference_mode():
        for frame in frames:
            query_match = fixed_queries.get(frame.cache_key)
            if query_match is None or not query_match.matched:
                continue
            with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                camera_images = renderer.build_frame_images(frame, apply_eot=False)
                outs, prev_bev, prev_abs_pos, prev_abs_angle = bev_model.forward_frame(frame, camera_images, prev_bev=prev_bev, prev_scene_token=prev_scene_token, prev_abs_pos=prev_abs_pos, prev_abs_angle=prev_abs_angle, retain_grad=False)
            if isinstance(prev_bev, torch.Tensor):
                prev_bev = prev_bev.detach()
            prev_scene_token = frame.scene_token
            prediction = bev_model.target_query_prediction(frame, outs, query_idx=query_match.query_idx)
            selected_query_idx = int(getattr(prediction, 'query_idx', query_match.query_idx))
            fixed_result_row, fixed_trace_row = _build_fixed_query_record(frame=frame, prediction=prediction, query_idx=selected_query_idx)
            movement_metrics = _build_target_movement_metrics(frame=frame, prediction=prediction, query_idx=selected_query_idx)
            payload[frame.cache_key] = {'result_row': fixed_result_row, 'trace_row': fixed_trace_row, 'sequence_name': str(frame.sequence_name), 'sample_token': str(frame.sample_token), 'query_idx': int(selected_query_idx), 'target_detection_name': str(prediction.target_detection_name), 'gt_center_world': [float(v) for v in frame.gt_center_world.tolist()], 'movement_metrics': movement_metrics}
            del prediction
            del outs
            del camera_images
            bev_model.last_bbox_tensor = None
            bev_model.last_cls_tensor = None
            if hasattr(bev_model, 'last_head'):
                bev_model.last_head = None
            if hasattr(bev_model, 'last_heatmap_tensor'):
                bev_model.last_heatmap_tensor = None
            if hasattr(bev_model, 'last_heatmap_grad'):
                bev_model.last_heatmap_grad = None
            if hasattr(bev_model, 'last_query_feature_tensor'):
                bev_model.last_query_feature_tensor = None
            bev_model.last_bbox_grad = None
            bev_model.last_cls_grad = None
    return payload

def _build_fixed_query_payload_from_official(*, official_results_path: Path, official_query_trace_path: Path, fixed_query_overrides: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    official_payload = json.loads(official_results_path.read_text(encoding='utf-8'))
    if official_query_trace_path.exists():
        trace_payload = json.loads(official_query_trace_path.read_text(encoding='utf-8'))
    else:
        trace_payload = {'meta': copy.deepcopy(official_payload.get('meta', {})), 'results': {}}
        official_results = official_payload.get('results', {})
        if isinstance(official_results, dict):
            for sample_token, rows in official_results.items():
                mapped_rows: List[Dict[str, Any]] = []
                if isinstance(rows, list):
                    for row in rows:
                        trace_row = dict(row) if isinstance(row, dict) else {}
                        trace_row['query_idx'] = int(trace_row.get('query_idx', -1))
                        mapped_rows.append(trace_row)
                trace_payload['results'][str(sample_token)] = mapped_rows
    fixed_payload = copy.deepcopy(official_payload)
    fixed_trace_payload = copy.deepcopy(trace_payload)
    for sample_token, override in fixed_query_overrides.items():
        result_row = dict(override['result_row'])
        trace_row = dict(override['trace_row'])
        query_idx = int(override['query_idx'])
        target_detection_name = str(override['target_detection_name'])
        gt_center_world = np.asarray(override['gt_center_world'], dtype=np.float32)
        results_rows = list(fixed_payload.setdefault('results', {}).get(sample_token, []))
        trace_rows = list(fixed_trace_payload.setdefault('results', {}).get(sample_token, []))
        pair_count = min(len(results_rows), len(trace_rows))
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = [(results_rows[idx], trace_rows[idx]) for idx in range(pair_count)]
        for idx in range(pair_count, len(results_rows)):
            pairs.append((results_rows[idx], dict(results_rows[idx])))
        filtered_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        replaced_attribute_name: Optional[str] = None
        for old_result_row, old_trace_row in pairs:
            if int(old_trace_row.get('query_idx', -1)) == query_idx:
                if replaced_attribute_name is None:
                    replaced_attribute_name = str(old_result_row.get('attribute_name', '') or '')
                continue
            filtered_pairs.append((old_result_row, old_trace_row))
        filtered_result_rows = [pair[0] for pair in filtered_pairs]
        replace_index = _find_target_replacement_index(rows=filtered_result_rows, target_detection_name=target_detection_name, gt_center_world=gt_center_world)
        if replace_index >= 0 and replaced_attribute_name is None:
            replaced_attribute_name = str(filtered_result_rows[replace_index].get('attribute_name', '') or '')
        if replaced_attribute_name:
            result_row['attribute_name'] = replaced_attribute_name
            trace_row['attribute_name'] = replaced_attribute_name
        if replace_index >= 0:
            filtered_pairs[replace_index] = (result_row, trace_row)
        else:
            filtered_pairs.append((result_row, trace_row))
        filtered_pairs.sort(key=lambda pair: float(pair[0].get('detection_score', 0.0)), reverse=True)
        fixed_payload['results'][sample_token] = [pair[0] for pair in filtered_pairs]
        fixed_trace_payload['results'][sample_token] = [pair[1] for pair in filtered_pairs]
    return (fixed_payload, fixed_trace_payload)

def _validate_fixed_query_trace_identity(*, frames: Sequence[FrameRecord], fixed_queries: Dict[str, FixedQueryMatch], official_query_trace_path: Path) -> Dict[str, Any]:
    if not official_query_trace_path.exists():
        raise FileNotFoundError(f'Official query trace not found: {official_query_trace_path}')
    trace_payload = json.loads(official_query_trace_path.read_text(encoding='utf-8'))
    trace_results = trace_payload.get('results', {})
    if not isinstance(trace_results, dict):
        raise ValueError("Official query trace payload missing 'results' dict")
    total = 0
    passed = 0
    failed = 0
    missing = 0
    rows: List[Dict[str, Any]] = []
    for frame in frames:
        query_match = fixed_queries.get(frame.cache_key)
        if query_match is None or not query_match.matched:
            continue
        total += 1
        sample_rows = trace_results.get(str(frame.sample_token), [])
        expected_query = int(query_match.query_idx)
        if not isinstance(sample_rows, list) or len(sample_rows) <= 0:
            missing += 1
            failed += 1
            rows.append({'sequence_name': str(frame.sequence_name), 'frame_id': int(frame.frame_id), 'sample_token': str(frame.sample_token), 'expected_query_idx': expected_query, 'top_query_idx': -1, 'top_score': 0.0, 'reason': 'trace_missing_or_empty'})
            continue
        top_row = sample_rows[0] if isinstance(sample_rows[0], dict) else {}
        top_query_idx = int(top_row.get('query_idx', -1))
        top_score = float(top_row.get('detection_score', 0.0))
        if top_query_idx == expected_query:
            passed += 1
        else:
            failed += 1
            rows.append({'sequence_name': str(frame.sequence_name), 'frame_id': int(frame.frame_id), 'sample_token': str(frame.sample_token), 'expected_query_idx': expected_query, 'top_query_idx': top_query_idx, 'top_score': top_score, 'top_detection_name': str(top_row.get('detection_name', '')), 'reason': 'top1_query_mismatch'})
    return {'total': int(total), 'passed': int(passed), 'failed': int(failed), 'missing': int(missing), 'rows': rows}

def _world_center_to_ego_xy(frame: FrameRecord, center_world_xyz: Sequence[float]) -> Tuple[float, float]:
    if len(center_world_xyz) < 2:
        raise ValueError('center_world_xyz must contain at least x/y')
    z_value = float(center_world_xyz[2]) if len(center_world_xyz) >= 3 else float(frame.gt_center_world[2])
    center_world = np.asarray([float(center_world_xyz[0]), float(center_world_xyz[1]), z_value, 1.0], dtype=np.float32)
    center_lidar = (center_world @ frame.lidar_from_global.T)[:3]
    center_ego = center_lidar @ frame.lidar_to_ego_rotation.T + frame.lidar_to_ego_translation
    return (float(center_ego[0]), float(center_ego[1]))

def _axis_projected_distances_ego(*, pred_ego_xy: Tuple[float, float], target_ego_xy: Tuple[float, float], distance_axis: str) -> Tuple[float, float]:
    dx = float(pred_ego_xy[0]) - float(target_ego_xy[0])
    dy = float(pred_ego_xy[1]) - float(target_ego_xy[1])
    axis_name = str(distance_axis).strip().lower()
    if axis_name in {'forward', 'front', 'forward_x', 'front_x', 'x', '+x'}:
        return (abs(dx), abs(dy))
    return (abs(dy), abs(dx))

def _axis_distance_labels(distance_axis: str) -> Tuple[str, str]:
    axis_name = str(distance_axis).strip().lower()
    if axis_name in {'forward', 'front', 'forward_x', 'front_x', 'x', '+x'}:
        return ('longitudinal distance', 'lateral distance')
    return ('lateral distance', 'longitudinal distance')

def _format_optional_distance(value: Optional[float]) -> str:
    if value is None or not math.isfinite(float(value)):
        return 'n/a'
    return f'{float(value):.3f}m'

def _final_decode_lost_reason(*, target_detection_name: str, target_total: int, confident_total: int, valid_translation_total: int, axis_ok_total: int, cross_ok_total: int, conf_threshold: float, max_center_dist_m: float, max_cross_axis_dist_m: float, distance_axis: str, min_axis_distance_m: Optional[float], min_cross_axis_distance_m: Optional[float]) -> str:
    if int(target_total) <= 0:
        return f'no {target_detection_name} candidates in official final outputs'
    if int(confident_total) <= 0:
        return f'all {target_detection_name} candidates below conf_threshold={float(conf_threshold):.2f}'
    if int(valid_translation_total) <= 0:
        return f'{confident_total} confident candidates lack valid 3D centers'
    axis_label, cross_label = _axis_distance_labels(distance_axis)
    reasons: List[str] = []
    if float(max_center_dist_m) > 0.0 and int(axis_ok_total) <= 0:
        reasons.append(f'all candidates exceed {axis_label} threshold {float(max_center_dist_m):.2f}m (min={_format_optional_distance(min_axis_distance_m)})')
    if float(max_cross_axis_dist_m) > 0.0 and int(cross_ok_total) <= 0:
        reasons.append(f'all candidates exceed {cross_label} threshold {float(max_cross_axis_dist_m):.2f}m (min={_format_optional_distance(min_cross_axis_distance_m)})')
    if not reasons:
        reasons.append(f'no candidate satisfies both {axis_label} and {cross_label} gates (min {axis_label}={_format_optional_distance(min_axis_distance_m)}, min {cross_label}={_format_optional_distance(min_cross_axis_distance_m)})')
    return f'{confident_total} confident candidates failed distance filter: ' + '; '.join(reasons)

def _validate_final_decode_target_trace(*, frames: Sequence[FrameRecord], bev_model: BevFormerGradientModel, official_query_trace_path: Path, conf_threshold: float, max_center_dist_m: float, distance_axis: str='lateral_y', max_cross_axis_dist_m: float=1.0, loss_reference_mode: str='gt', clean_detection_refs: Optional[Dict[str, Dict[str, torch.Tensor]]]=None) -> Dict[str, Any]:
    if not official_query_trace_path.exists():
        raise FileNotFoundError(f'Official query trace not found: {official_query_trace_path}')
    trace_payload = json.loads(official_query_trace_path.read_text(encoding='utf-8'))
    trace_results = trace_payload.get('results', {})
    if not isinstance(trace_results, dict):
        raise ValueError("Official query trace payload missing 'results' dict")
    rows: List[Dict[str, Any]] = []
    matched = 0
    lost = 0
    for frame in frames:
        target_detection_name = bev_model.frame_target_detection_name(frame)
        sample_rows = trace_results.get(str(frame.sample_token), [])
        if not isinstance(sample_rows, list):
            sample_rows = []
        target_rows = [row for row in sample_rows if isinstance(row, dict) and str(row.get('detection_name', '')) == target_detection_name]
        confident_rows = [row for row in target_rows if float(row.get('detection_score', 0.0)) >= float(conf_threshold)]
        matched_rows: List[Tuple[Dict[str, Any], float, float, float]] = []
        valid_translation_count = 0
        axis_ok_count = 0
        cross_ok_count = 0
        axis_distances: List[float] = []
        cross_axis_distances: List[float] = []
        target_ego_xy = _world_center_to_ego_xy(frame, frame.gt_center_world.tolist())
        reference_source = 'gt'
        reference_ego_xy: Optional[Tuple[float, float]] = target_ego_xy
        if str(loss_reference_mode).strip().lower() == 'clean':
            reference_source = 'clean_missing'
            reference_ego_xy = None
            cached_ref = clean_detection_refs.get(frame.cache_key) if isinstance(clean_detection_refs, dict) else None
            if isinstance(cached_ref, dict) and isinstance(cached_ref.get('center_ego'), torch.Tensor):
                ref_center = cached_ref['center_ego'].detach().cpu().reshape(-1)
                if int(ref_center.numel()) >= 2:
                    reference_ego_xy = (float(ref_center[0].item()), float(ref_center[1].item()))
                    reference_source = 'clean'
        for row in confident_rows:
            translation = row.get('translation', [])
            if not isinstance(translation, list) or len(translation) < 2:
                continue
            valid_translation_count += 1
            pred_ego_xy = _world_center_to_ego_xy(frame, translation)
            move_axis_dist_m, cross_axis_dist_m = _axis_projected_distances_ego(pred_ego_xy=pred_ego_xy, target_ego_xy=target_ego_xy, distance_axis=distance_axis)
            axis_distances.append(float(move_axis_dist_m))
            cross_axis_distances.append(float(cross_axis_dist_m))
            move_axis_ok = max_center_dist_m <= 0.0 or move_axis_dist_m <= float(max_center_dist_m)
            cross_axis_ok = max_cross_axis_dist_m <= 0.0 or cross_axis_dist_m <= float(max_cross_axis_dist_m)
            if move_axis_ok:
                axis_ok_count += 1
            if cross_axis_ok:
                cross_ok_count += 1
            if move_axis_ok and cross_axis_ok:
                world_distance_m = math.hypot(float(translation[0]) - float(frame.gt_center_world[0]), float(translation[1]) - float(frame.gt_center_world[1]))
                matched_rows.append((row, world_distance_m, move_axis_dist_m, cross_axis_dist_m))
        if not matched_rows:
            clean_projected_distance = abs(float(reference_ego_xy[1])) if reference_ego_xy is not None else None
            min_axis_distance = min(axis_distances) if axis_distances else None
            min_cross_axis_distance = min(cross_axis_distances) if cross_axis_distances else None
            reason = _final_decode_lost_reason(target_detection_name=target_detection_name, target_total=len(target_rows), confident_total=len(confident_rows), valid_translation_total=valid_translation_count, axis_ok_total=axis_ok_count, cross_ok_total=cross_ok_count, conf_threshold=conf_threshold, max_center_dist_m=max_center_dist_m, max_cross_axis_dist_m=max_cross_axis_dist_m, distance_axis=distance_axis, min_axis_distance_m=min_axis_distance, min_cross_axis_distance_m=min_cross_axis_distance)
            lost += 1
            rows.append({'matched': False, 'sequence_name': str(frame.sequence_name), 'frame_id': int(frame.frame_id), 'sample_token': str(frame.sample_token), 'target_detection_name': target_detection_name, 'gt_x_m': float(target_ego_xy[0]), 'gt_y_m': float(target_ego_xy[1]), 'reference_source': reference_source, 'ref_x_m': float(reference_ego_xy[0]) if reference_ego_xy is not None else None, 'ref_y_m': float(reference_ego_xy[1]) if reference_ego_xy is not None else None, 'clean_projected_distance_to_ego_front_line_m': float(clean_projected_distance) if clean_projected_distance is not None else None, 'attacked_projected_distance_to_ego_front_line_m': None, 'attacked_moved_toward_ego_front_line_m': None, 'attacked_moved_toward_ego_front_line_pct': None, 'candidate_total': int(len(target_rows)), 'candidate_after_conf': int(len(confident_rows)), 'candidate_after_dist': 0, 'candidate_after_axis': int(axis_ok_count), 'candidate_after_cross': int(cross_ok_count), 'min_axis_distance_m': min_axis_distance, 'min_cross_axis_distance_m': min_cross_axis_distance, 'reason': reason})
            continue
        best_row, best_distance, best_axis_distance, best_cross_distance = min(matched_rows, key=lambda item: (-float(item[0].get('detection_score', 0.0)), float(item[3]), float(item[2])))
        matched += 1
        best_translation = best_row.get('translation', [])
        best_pred_ego_xy = _world_center_to_ego_xy(frame, best_translation)
        target_ego_xy = _world_center_to_ego_xy(frame, frame.gt_center_world.tolist())
        if reference_ego_xy is None:
            matched -= 1
            lost += 1
            rows.append({'matched': False, 'sequence_name': str(frame.sequence_name), 'frame_id': int(frame.frame_id), 'sample_token': str(frame.sample_token), 'target_detection_name': target_detection_name, 'gt_x_m': float(target_ego_xy[0]), 'gt_y_m': float(target_ego_xy[1]), 'reference_source': reference_source, 'ref_x_m': None, 'ref_y_m': None, 'candidate_total': int(len(target_rows)), 'candidate_after_conf': int(len(confident_rows)), 'candidate_after_dist': int(len(matched_rows)), 'reason': 'clean_reference_missing'})
            continue
        delta_x_m = float(best_pred_ego_xy[0] - reference_ego_xy[0])
        delta_y_m = float(best_pred_ego_xy[1] - reference_ego_xy[1])
        clean_projected_distance = abs(float(reference_ego_xy[1]))
        attacked_projected_distance = abs(float(best_pred_ego_xy[1]))
        moved_toward_ego_front_m = float(clean_projected_distance - attacked_projected_distance)
        moved_toward_ego_front_pct = None
        if clean_projected_distance > 1e-06:
            moved_toward_ego_front_pct = float(moved_toward_ego_front_m / clean_projected_distance * 100.0)
        rows.append({'matched': True, 'sequence_name': str(frame.sequence_name), 'frame_id': int(frame.frame_id), 'sample_token': str(frame.sample_token), 'target_detection_name': target_detection_name, 'query_idx': int(best_row.get('query_idx', -1)), 'score': float(best_row.get('detection_score', 0.0)), 'pred_x_m': float(best_pred_ego_xy[0]), 'pred_y_m': float(best_pred_ego_xy[1]), 'gt_x_m': float(target_ego_xy[0]), 'gt_y_m': float(target_ego_xy[1]), 'reference_source': reference_source, 'ref_x_m': float(reference_ego_xy[0]), 'ref_y_m': float(reference_ego_xy[1]), 'clean_projected_distance_to_ego_front_line_m': float(clean_projected_distance), 'attacked_projected_distance_to_ego_front_line_m': float(attacked_projected_distance), 'attacked_moved_toward_ego_front_line_m': float(moved_toward_ego_front_m), 'attacked_moved_toward_ego_front_line_pct': moved_toward_ego_front_pct, 'delta_x_m': delta_x_m, 'delta_y_m': delta_y_m, 'direction': 'left' if delta_y_m >= 0.0 else 'right', 'shift_abs_m': abs(delta_y_m), 'target_lateral_move_m': delta_y_m if float(reference_ego_xy[1]) < 0.0 else -delta_y_m, 'x_direction': 'front' if delta_x_m >= 0.0 else 'back', 'x_shift_abs_m': abs(delta_x_m), 'distance_m': float(best_distance), 'axis_distance_m': float(best_axis_distance), 'cross_axis_distance_m': float(best_cross_distance), 'candidate_total': int(len(target_rows)), 'candidate_after_conf': int(len(confident_rows)), 'candidate_after_dist': int(len(matched_rows))})
    return {'total': int(len(frames)), 'matched': int(matched), 'lost': int(lost), 'rows': rows}

def _official_model_cfg_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = selected_model_cfg(config)
    if not isinstance(cfg, dict):
        raise ValueError('selected model config must be dict')
    return cfg

def _official_visual_repo_root_from_config(config: Dict[str, Any]) -> Path:
    model_cfg = _official_model_cfg_from_config(config)
    model_repo = _as_path(str(model_cfg.get('repo_root', '')))
    model_visual = model_repo / 'tools' / 'visual.py'
    if model_visual.exists():
        return model_repo
    model_bevdet_visual = model_repo / 'tools' / 'analysis_tools' / 'vis.py'
    if model_bevdet_visual.exists():
        return model_repo
    bev_cfg = config.get('bevformer', {}) if isinstance(config.get('bevformer', {}), dict) else {}
    bev_repo_raw = str(bev_cfg.get('repo_root', '') or '').strip()
    if bev_repo_raw:
        bev_repo = _as_path(bev_repo_raw)
        if (bev_repo / 'tools' / 'visual.py').exists():
            return bev_repo
    raise FileNotFoundError(f'Official visual.py not found under model repo ({model_repo}) or fallback bevformer.repo_root')

def _export_fixed_query_results_and_visuals(*, config: Dict[str, Any], image_source_subdir: str, output_dir: Path, frames: Sequence[FrameRecord], image_root: Path, fixed_query_overrides: Dict[str, Dict[str, Any]], official_results_path: Path, official_query_trace_path: Path, export_subdir: str='fixed_query') -> Tuple[Path, Path]:
    dataset_cfg = config.get('dataset', {})
    model_cfg = _official_model_cfg_from_config(config)
    official_cfg = config.get('official_visual', {})
    score_threshold = float(official_cfg.get('score_threshold', 0.2))
    results_dir = output_dir / export_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'results_nusc.json'
    query_trace_path = results_dir / 'results_nusc_query_trace.json'
    try:
        fixed_payload, fixed_trace_payload = _build_fixed_query_payload_from_official(official_results_path=official_results_path, official_query_trace_path=official_query_trace_path, fixed_query_overrides=fixed_query_overrides)
        _save_json(results_path, fixed_payload)
        _save_json(query_trace_path, fixed_trace_payload)
        if selected_model_name(config) == 'stp3':
            plots_dir = (results_dir / 'plots').resolve()
            plots_dir.mkdir(parents=True, exist_ok=True)
            (plots_dir / 'STP3_PROXY_VISUALIZATION_SKIPPED.txt').write_text('STP3 exposes a differentiable vehicle BEV proxy rather than official 3D detection boxes; fixed-query proxy results are saved in results_nusc.json.\n', encoding='utf-8')
            return (results_path, plots_dir)
        version = str(dataset_cfg.get('version', 'v1.0-trainval')).strip() or 'v1.0-trainval'
        dataroot = _as_path(str(dataset_cfg.get('dataroot', model_cfg.get('data_root', ''))))
        _run_official_visualization(bevformer_repo_root=_official_visual_repo_root_from_config(config), results_path=results_path, output_dir=output_dir, export_subdir=export_subdir, version=version, dataroot=dataroot, image_dataroot=image_root, image_source_subdir=image_source_subdir, score_threshold=score_threshold)
    finally:
        shutil.rmtree(image_root, ignore_errors=True)
    visual_dir = (results_dir / 'plots').resolve()
    if not visual_dir.exists():
        raise FileNotFoundError(f'Fixed-query visualization output not found: {visual_dir}')
    return (results_path, visual_dir)

def _write_official_test_config(*, bev_model: BevFormerGradientModel, attacked_ann_file: Path, config_path: Path) -> None:
    mmcv = bev_model._import_mmcv_cleanly()
    cfg = mmcv.Config.fromfile(str(bev_model.config_path))
    bev_model._import_plugin_module(cfg, config_path=bev_model.config_path)
    cfg.model.pretrained = None
    if 'use_grid_mask' in cfg.model:
        cfg.model.use_grid_mask = False
    if 'video_test_mode' in cfg.model:
        cfg.model.video_test_mode = True
    if bev_model.data_root is not None:
        cfg.data_root = str(bev_model.data_root)
        if 'val' in cfg.data:
            cfg.data.val.data_root = str(bev_model.data_root)
        cfg.data.test.data_root = str(bev_model.data_root)
    cfg.data.workers_per_gpu = bev_model.workers_per_gpu
    cfg.data.test.ann_file = str(attacked_ann_file)
    cfg.data.test.test_mode = True

    def _force_disk_file_client(node: Any) -> None:
        if isinstance(node, dict):
            if 'file_client_args' in node:
                node['file_client_args'] = dict(backend='disk')
            for value in node.values():
                _force_disk_file_client(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                _force_disk_file_client(value)
    _force_disk_file_client(cfg.data.test)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(config_path))

def _run_official_test(*, bev_model: BevFormerGradientModel, attacked_config_path: Path, output_dir: Path, export_subdir: str) -> Path:
    tool_path = bev_model.repo_root / 'tools' / 'test.py'
    if not tool_path.exists():
        raise FileNotFoundError(f'Official test.py not found: {tool_path}')
    results_dir = output_dir / export_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    root_result_path = results_dir / 'results_nusc.json'
    root_query_trace_path = results_dir / 'results_nusc_query_trace.json'
    pts_result_dir = results_dir / 'pts_bbox'
    if root_result_path.exists():
        root_result_path.unlink()
    if root_query_trace_path.exists():
        root_query_trace_path.unlink()
    if pts_result_dir.exists():
        shutil.rmtree(pts_result_dir, ignore_errors=True)
    wrapper_code = '\nimport os\nimport runpy\nimport sys\nimport types\nimport importlib\nimport os.path as osp\n\nraw_args = list(sys.argv[1:])\nlocal_rank = None\nparsed_args = []\nskip_next = False\nfor index, arg in enumerate(raw_args):\n    if skip_next:\n        skip_next = False\n        continue\n    if arg.startswith("--local-rank=") or arg.startswith("--local_rank="):\n        local_rank = arg.split("=", 1)[1]\n        continue\n    if arg in ("--local-rank", "--local_rank"):\n        if index + 1 < len(raw_args):\n            local_rank = raw_args[index + 1]\n            skip_next = True\n        continue\n    parsed_args.append(arg)\n\ntool_path, repo_root, config_path, checkpoint_path, result_dir = parsed_args[:5]\nif local_rank is not None:\n    os.environ["LOCAL_RANK"] = str(local_rank)\nrepo_real = os.path.realpath(repo_root)\nsys.path[:] = [entry for entry in sys.path if os.path.realpath(entry or os.curdir) != repo_real]\nprojects_root = os.path.join(repo_root, "projects")\nfor path_entry in (projects_root, repo_root):\n    if path_entry in sys.path:\n        sys.path.remove(path_entry)\nfor path_entry in (projects_root, repo_root):\n    sys.path.insert(0, path_entry)\nfor module_name in list(sys.modules.keys()):\n    if module_name == "mmdet3d" or module_name.startswith("mmdet3d."):\n        sys.modules.pop(module_name, None)\n    if module_name == "projects" or module_name.startswith("projects."):\n        sys.modules.pop(module_name, None)\n\nimport mmcv\nimport torch\nimport numpy as np\nimport pyquaternion\nmmcv.__version__ = "1.4.0"\ntry:\n    from mmcv.utils import Registry\n    _orig_registry_register_module = Registry._register_module\n\n    def _register_module_duplicate_compat(self, module, module_name=None, force=False):\n        try:\n            return _orig_registry_register_module(self, module, module_name=module_name, force=force)\n        except KeyError as exc:\n            if "is already registered" not in str(exc):\n                raise\n            return _orig_registry_register_module(self, module, module_name=module_name, force=True)\n\n    Registry._register_module = _register_module_duplicate_compat\nexcept Exception:\n    pass\nif "ipdb" not in sys.modules:\n    ipdb_stub = types.ModuleType("ipdb")\n    ipdb_stub.set_trace = lambda *args, **kwargs: None\n    sys.modules["ipdb"] = ipdb_stub\ntry:\n    import numba\n    import numba.errors  # type: ignore\nexcept Exception:\n    try:\n        import numba\n        numba_errors_stub = types.ModuleType("numba.errors")\n        numba_errors_stub.NumbaPerformanceWarning = getattr(numba, "NumbaPerformanceWarning", Warning)\n        sys.modules["numba.errors"] = numba_errors_stub\n    except Exception:\n        pass\n\ndef _install_camera_only_optional_op_stubs(repo_root):\n    class _MissingOpModule(types.ModuleType):\n        def __init__(self, name):\n            super().__init__(name)\n            self.__file__ = f"<optional camera-only op stub: {name}>"\n\n        def __getattr__(self, attr):\n            def _missing(*args, **kwargs):\n                raise RuntimeError(\n                    f"Optional CUDA op {self.__name__}.{attr} is not compiled. "\n                    "This camera-only official test wrapper does not need it."\n                )\n            return _missing\n\n    def _missing_callable(name):\n        def _missing(*args, **kwargs):\n            raise RuntimeError(\n                f"Optional CUDA op {name} is not compiled. "\n                "This camera-only official test wrapper does not need it."\n            )\n        return _missing\n\n    def _ensure_module(name, module=None):\n        existing = sys.modules.get(name)\n        if existing is not None:\n            return existing\n        module = module or _MissingOpModule(name)\n        sys.modules[name] = module\n        return module\n\n    ops_root = types.ModuleType("mmdet3d.ops")\n    ops_root.__path__ = [os.path.join(repo_root, "mmdet3d", "ops")]\n    try:\n        from mmcv.ops import RoIAlign, SigmoidFocalLoss, nms, roi_align, sigmoid_focal_loss\n        ops_root.RoIAlign = RoIAlign\n        ops_root.roi_align = roi_align\n        ops_root.SigmoidFocalLoss = SigmoidFocalLoss\n        ops_root.sigmoid_focal_loss = sigmoid_focal_loss\n        ops_root.nms = nms\n    except Exception:\n        pass\n    for attr_name in (\n        "Voxelization", "voxelization", "DynamicScatter", "dynamic_scatter",\n        "SparseBasicBlock", "SparseBottleneck", "make_sparse_convmodule",\n        "RoIAwarePool3d", "ball_query", "knn", "furthest_point_sample",\n        "furthest_point_sample_with_dist", "three_interpolate", "three_nn",\n        "gather_points", "grouping_operation", "group_points", "GroupAll",\n        "QueryAndGroup", "PointSAModule", "PointSAModuleMSG", "PointFPModule",\n        "points_in_boxes_batch", "points_in_boxes_cpu", "points_in_boxes_gpu",\n        "assign_score_withk", "Points_Sampler", "build_sa_module", "PAConv",\n        "PAConvCUDA", "PAConvSAModuleMSG", "PAConvSAModule",\n        "PAConvCUDASAModule", "PAConvCUDASAModuleMSG",\n    ):\n        setattr(ops_root, attr_name, _missing_callable(f"mmdet3d.ops.{attr_name}"))\n\n    iou3d_cuda = types.ModuleType("mmdet3d.ops.iou3d.iou3d_cuda")\n\n    def _fill_keep(keep, count):\n        if count <= 0:\n            return 0\n        keep[:count] = torch.arange(count, dtype=keep.dtype, device=keep.device)\n        return count\n\n    def _nms_gpu(boxes, keep, thresh, device_id=None):\n        del thresh, device_id\n        return _fill_keep(keep, int(boxes.shape[0]))\n\n    def _boxes_iou_bev_gpu(boxes_a, boxes_b, ans_iou):\n        del boxes_a, boxes_b\n        ans_iou.zero_()\n\n    def _boxes_overlap_bev_gpu(boxes_a, boxes_b, ans_overlap):\n        del boxes_a, boxes_b\n        ans_overlap.zero_()\n\n    iou3d_cuda.nms_gpu = _nms_gpu\n    iou3d_cuda.nms_normal_gpu = _nms_gpu\n    iou3d_cuda.boxes_iou_bev_gpu = _boxes_iou_bev_gpu\n    iou3d_cuda.boxes_overlap_bev_gpu = _boxes_overlap_bev_gpu\n    iou3d_pkg = types.ModuleType("mmdet3d.ops.iou3d")\n    iou3d_pkg.__path__ = [os.path.join(repo_root, "mmdet3d", "ops", "iou3d")]\n    iou3d_pkg.iou3d_cuda = iou3d_cuda\n    iou3d_utils = types.ModuleType("mmdet3d.ops.iou3d.iou3d_utils")\n\n    def _xyxyr_to_xywhr(boxes):\n        converted = boxes.new_zeros(boxes.shape)\n        converted[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5\n        converted[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5\n        converted[:, 2] = boxes[:, 2] - boxes[:, 0]\n        converted[:, 3] = boxes[:, 3] - boxes[:, 1]\n        converted[:, 4] = boxes[:, 4]\n        return converted\n\n    def _nms_indices(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):\n        order = scores.sort(0, descending=True)[1]\n        if pre_maxsize is not None:\n            order = order[:pre_maxsize]\n        try:\n            from mmcv.ops import nms_rotated\n            nms_boxes = _xyxyr_to_xywhr(boxes[order]).float()\n            _kept, kept_local = nms_rotated(nms_boxes, scores[order].float(), float(thresh))\n            order = order[kept_local.to(device=order.device, dtype=torch.long)]\n        except Exception:\n            pass\n        if post_max_size is not None:\n            order = order[:post_max_size]\n        return order.contiguous()\n\n    def _boxes_iou_bev(boxes_a, boxes_b):\n        return boxes_a.new_zeros((int(boxes_a.shape[0]), int(boxes_b.shape[0])))\n\n    iou3d_utils.nms_gpu = _nms_indices\n    iou3d_utils.nms_normal_gpu = _nms_indices\n    iou3d_utils.boxes_iou_bev = _boxes_iou_bev\n    iou3d_pkg.nms_gpu = _nms_indices\n    iou3d_pkg.nms_normal_gpu = _nms_indices\n    iou3d_pkg.boxes_iou_bev = _boxes_iou_bev\n    _ensure_module("mmdet3d.ops", ops_root)\n    _ensure_module("mmdet3d.ops.iou3d", iou3d_pkg)\n    _ensure_module("mmdet3d.ops.iou3d.iou3d_cuda", iou3d_cuda)\n    _ensure_module("mmdet3d.ops.iou3d.iou3d_utils", iou3d_utils)\n\n    roiaware_ext = types.ModuleType("mmdet3d.ops.roiaware_pool3d.roiaware_pool3d_ext")\n\n    def _points_in_boxes_gpu(boxes, points, out):\n        del boxes, points\n        out.fill_(-1)\n\n    def _points_in_boxes_cpu(boxes, points, out):\n        del boxes, points\n        out.zero_()\n\n    roiaware_ext.points_in_boxes_gpu = _points_in_boxes_gpu\n    roiaware_ext.points_in_boxes_batch = _points_in_boxes_gpu\n    roiaware_ext.points_in_boxes_cpu = _points_in_boxes_cpu\n    roiaware_pkg = types.ModuleType("mmdet3d.ops.roiaware_pool3d")\n    roiaware_pkg.__path__ = [os.path.join(repo_root, "mmdet3d", "ops", "roiaware_pool3d")]\n    roiaware_pkg.points_in_boxes_gpu = _missing_callable("mmdet3d.ops.roiaware_pool3d.points_in_boxes_gpu")\n    roiaware_pkg.points_in_boxes_batch = _missing_callable("mmdet3d.ops.roiaware_pool3d.points_in_boxes_batch")\n    roiaware_pkg.points_in_boxes_cpu = _missing_callable("mmdet3d.ops.roiaware_pool3d.points_in_boxes_cpu")\n    roiaware_pkg.RoIAwarePool3d = _missing_callable("mmdet3d.ops.roiaware_pool3d.RoIAwarePool3d")\n    _ensure_module("mmdet3d.ops.roiaware_pool3d", roiaware_pkg)\n    _ensure_module("mmdet3d.ops.roiaware_pool3d.roiaware_pool3d_ext", roiaware_ext)\n\n    spconv_pkg = _MissingOpModule("mmdet3d.ops.spconv")\n    spconv_pkg.__path__ = [os.path.join(repo_root, "mmdet3d", "ops", "spconv")]\n    for attr_name in ("SparseConvTensor", "SparseSequential", "SparseModule", "SubMConv3d", "SparseConv3d", "SparseConv2d", "SparseInverseConv3d"):\n        setattr(spconv_pkg, attr_name, _missing_callable(f"mmdet3d.ops.spconv.{attr_name}"))\n    ops_root.spconv = spconv_pkg\n    _ensure_module("mmdet3d.ops.spconv", spconv_pkg)\n\n    for module_name in (\n        "mmdet3d.ops.ball_query", "mmdet3d.ops.ball_query.ball_query_ext",\n        "mmdet3d.ops.furthest_point_sample", "mmdet3d.ops.furthest_point_sample.furthest_point_sample_ext",\n        "mmdet3d.ops.gather_points", "mmdet3d.ops.gather_points.gather_points_ext",\n        "mmdet3d.ops.group_points", "mmdet3d.ops.group_points.group_points_ext",\n        "mmdet3d.ops.interpolate", "mmdet3d.ops.interpolate.interpolate_ext",\n        "mmdet3d.ops.knn", "mmdet3d.ops.knn.knn_ext",\n        "mmdet3d.ops.paconv", "mmdet3d.ops.paconv.assign_score_withk_ext",\n        "mmdet3d.ops.spconv.sparse_conv_ext",\n        "mmdet3d.ops.voxel", "mmdet3d.ops.voxel.voxel_layer",\n    ):\n        _ensure_module(module_name)\n\n_install_camera_only_optional_op_stubs(repo_root)\nfrom mmdet3d.datasets.nuscenes_dataset import NuScenesDataset, lidar_nusc_box_to_global, output_to_nusc_box\nimport mmdet3d.core.bbox.transforms as bbox_transforms\nfrom mmdet3d.core.bbox.transforms import bbox3d2result as _orig_bbox3d2result\n\nmmcv_file = getattr(mmcv, "__file__", None)\nif not mmcv_file or not hasattr(mmcv, "Config"):\n    mmcv_path = list(getattr(mmcv, "__path__", []) or [])\n    raise RuntimeError(\n        "Abnormal mmcv import: the BEVFormer repo root may appear early on sys.path, "\n        "shadowing the real mmcv with a stub mmcv package."\n        f" mmcv.__file__={mmcv_file!r} mmcv.__path__={mmcv_path!r}"\n    )\n\nif "projects" not in sys.modules:\n    projects_module = types.ModuleType("projects")\n    projects_module.__path__ = [projects_root]\n    sys.modules["projects"] = projects_module\n\ncfg_for_model = mmcv.Config.fromfile(config_path)\nmodel_type = str(getattr(cfg_for_model, "model", {}).get("type", "")).lower()\nhead_class = None\ndetector_module = None\ndenormalize_bbox = None\nNMSFreeCoder = None\nif model_type == "bevformer":\n    denormalize_bbox = importlib.import_module(\n        "projects.mmdet3d_plugin.core.bbox.util"\n    ).denormalize_bbox\n    NMSFreeCoder = importlib.import_module(\n        "projects.mmdet3d_plugin.core.bbox.coders.nms_free_coder"\n    ).NMSFreeCoder\n    head_class = importlib.import_module(\n        "projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_head"\n    ).BEVFormerHead\n    detector_module = importlib.import_module(\n        "projects.mmdet3d_plugin.bevformer.detectors.bevformer"\n    )\nelif model_type.startswith("bevdet"):\n    pass\nelse:\n    try:\n        head_class = importlib.import_module(\n            "projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_head"\n        ).BEVFormerHead\n        detector_module = importlib.import_module(\n            "projects.mmdet3d_plugin.bevformer.detectors.bevformer"\n        )\n    except Exception as exc:\n        raise RuntimeError(f"Unsupported model type for query trace patch: {model_type}") from exc\n\n\ndef _decode_single_with_query(self, cls_scores, bbox_preds):\n    max_num = self.max_num\n    cls_scores = cls_scores.sigmoid()\n    scores, indexs = cls_scores.view(-1).topk(max_num)\n    labels = indexs % self.num_classes\n    bbox_index = torch.div(indexs, self.num_classes, rounding_mode=\'floor\')\n    bbox_preds = bbox_preds[bbox_index]\n\n    final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)\n    final_scores = scores\n    final_preds = labels\n\n    thresh_mask = None\n    if self.score_threshold is not None:\n        thresh_mask = final_scores > self.score_threshold\n        tmp_score = self.score_threshold\n        while thresh_mask.sum() == 0:\n            tmp_score *= 0.9\n            if tmp_score < 0.01:\n                thresh_mask = final_scores > -1\n                break\n            thresh_mask = final_scores >= tmp_score\n\n    if self.post_center_range is not None:\n        post_center_range = torch.tensor(self.post_center_range, device=scores.device)\n        mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(1)\n        mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(1)\n        if self.score_threshold:\n            mask &= thresh_mask\n        return {\n            \'bboxes\': final_box_preds[mask],\n            \'scores\': final_scores[mask],\n            \'labels\': final_preds[mask],\n            \'query_indices\': bbox_index[mask],\n        }\n    raise NotImplementedError(\n        \'Need to reorganize output as a batch, only support post_center_range is not None for now!\'\n    )\n\n\ndef _get_bboxes_with_query(self, preds_dicts, img_metas, rescale=False):\n    preds_dicts = self.bbox_coder.decode(preds_dicts)\n    ret_list = []\n    for i in range(len(preds_dicts)):\n        preds = preds_dicts[i]\n        bboxes = preds[\'bboxes\']\n        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5\n        code_size = bboxes.shape[-1]\n        boxes3d = img_metas[i][\'box_type_3d\'](bboxes, code_size)\n        query_indices = preds.get(\'query_indices\')\n        if query_indices is not None:\n            setattr(boxes3d, \'query_indices\', query_indices.detach().cpu())\n        ret_list.append([boxes3d, preds[\'scores\'], preds[\'labels\']])\n    return ret_list\n\n\ndef _bbox3d2result_with_query(bboxes, scores, labels, attrs=None):\n    result_dict = _orig_bbox3d2result(bboxes, scores, labels, attrs=attrs)\n    query_indices = getattr(bboxes, \'query_indices\', None)\n    if query_indices is not None:\n        result_dict[\'query_indices\'] = query_indices.cpu()\n    return result_dict\n\n\ndef _format_bbox_with_query(self, results, jsonfile_prefix=None):\n    nusc_annos = {}\n    query_trace_annos = {}\n    mapped_class_names = self.CLASSES\n\n    print(\'Converting detection outputs to NuScenes format...\')\n    for sample_id, det in enumerate(mmcv.track_iter_progress(results)):\n        annos = []\n        trace_rows = []\n        with_velocity = bool(getattr(self, "with_velocity", True))\n        boxes = output_to_nusc_box(det, with_velocity)\n        sample_token = self.data_infos[sample_id][\'token\']\n        query_indices = det.get(\'query_indices\')\n        if hasattr(query_indices, \'tolist\'):\n            query_indices = query_indices.tolist()\n        if not isinstance(query_indices, (list, tuple)):\n            query_indices = []\n\n        filtered_pairs = []\n        info = self.data_infos[sample_id]\n        cls_range_map = self.eval_detection_configs.class_range\n        for i, box in enumerate(boxes):\n            query_idx = int(query_indices[i]) if i < len(query_indices) else -1\n            box.rotate(pyquaternion.Quaternion(info[\'lidar2ego_rotation\']))\n            box.translate(np.array(info[\'lidar2ego_translation\']))\n            radius = np.linalg.norm(box.center[:2], 2)\n            det_range = cls_range_map[mapped_class_names[box.label]]\n            if radius > det_range:\n                continue\n            box.rotate(pyquaternion.Quaternion(info[\'ego2global_rotation\']))\n            box.translate(np.array(info[\'ego2global_translation\']))\n            filtered_pairs.append((box, query_idx))\n\n        for box, query_idx in filtered_pairs:\n            name = mapped_class_names[box.label]\n            if (box.velocity[0] ** 2 + box.velocity[1] ** 2) ** 0.5 > 0.2:\n                if name in [\'car\', \'construction_vehicle\', \'bus\', \'truck\', \'trailer\']:\n                    attr = \'vehicle.moving\'\n                elif name in [\'bicycle\', \'motorcycle\']:\n                    attr = \'cycle.with_rider\'\n                else:\n                    attr = NuScenesDataset.DefaultAttribute[name]\n            else:\n                if name in [\'pedestrian\']:\n                    attr = \'pedestrian.standing\'\n                elif name in [\'bus\']:\n                    attr = \'vehicle.stopped\'\n                else:\n                    attr = NuScenesDataset.DefaultAttribute[name]\n\n            anno = dict(\n                sample_token=sample_token,\n                translation=box.center.tolist(),\n                size=box.wlh.tolist(),\n                rotation=box.orientation.elements.tolist(),\n                velocity=box.velocity[:2].tolist(),\n                detection_name=name,\n                detection_score=box.score,\n                attribute_name=attr,\n            )\n            annos.append(anno)\n            trace_anno = dict(anno)\n            trace_anno[\'query_idx\'] = int(query_idx)\n            trace_rows.append(trace_anno)\n        nusc_annos[sample_token] = annos\n        query_trace_annos[sample_token] = trace_rows\n\n    nusc_submissions = {\'meta\': self.modality, \'results\': nusc_annos}\n    trace_payload = {\'meta\': self.modality, \'results\': query_trace_annos}\n    mmcv.mkdir_or_exist(jsonfile_prefix)\n    res_path = osp.join(jsonfile_prefix, \'results_nusc.json\')\n    trace_path = osp.join(jsonfile_prefix, \'results_nusc_query_trace.json\')\n    print(\'Wrote detection results to\', res_path)\n    mmcv.dump(nusc_submissions, res_path)\n    mmcv.dump(trace_payload, trace_path)\n    return res_path\n\n\nif head_class is not None and detector_module is not None and NMSFreeCoder is not None and denormalize_bbox is not None:\n    NMSFreeCoder.decode_single = _decode_single_with_query\n    head_class.get_bboxes = _get_bboxes_with_query\n    bbox_transforms.bbox3d2result = _bbox3d2result_with_query\n    detector_module.bbox3d2result = _bbox3d2result_with_query\n    NuScenesDataset._format_bbox = _format_bbox_with_query\n\nsys.argv = [\n    tool_path,\n    config_path,\n    checkpoint_path,\n    "--format-only",\n    "--launcher",\n    "pytorch",\n    "--eval-options",\n    f"jsonfile_prefix={result_dir}",\n]\nrunpy.run_path(tool_path, run_name="__main__")\n'
    official_log = results_dir / 'official_test.log'
    wrapper_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile('w', suffix='_official_test_runner.py', delete=False, encoding='utf-8') as temp_fp:
            temp_fp.write(wrapper_code)
            wrapper_path = Path(temp_fp.name)
        command = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=1', str(wrapper_path), str(tool_path), str(bev_model.repo_root), str(attacked_config_path), str(bev_model.checkpoint_path), str(results_dir)]
        with official_log.open('w', encoding='utf-8') as fp:
            subprocess.run(command, cwd=str(bev_model.repo_root), stdout=fp, stderr=subprocess.STDOUT, check=True)
    finally:
        if wrapper_path is not None and wrapper_path.exists():
            wrapper_path.unlink()
    pts_result_path = results_dir / 'pts_bbox' / 'results_nusc.json'
    pts_query_trace_path = results_dir / 'pts_bbox' / 'results_nusc_query_trace.json'
    if pts_result_path.exists():
        shutil.copyfile(pts_result_path, root_result_path)
        if pts_query_trace_path.exists():
            shutil.copyfile(pts_query_trace_path, root_query_trace_path)
        shutil.rmtree(results_dir / 'pts_bbox', ignore_errors=True)
        return root_result_path
    if root_result_path.exists():
        return root_result_path
    raise RuntimeError('Official test did not produce results_nusc.json')

def _output_to_nusc_box_compat(det: Dict[str, Any], *, with_velocity: bool) -> List[Any]:
    from mmdet3d.datasets.nuscenes_dataset import output_to_nusc_box
    try:
        return output_to_nusc_box(det, with_velocity)
    except TypeError:
        return output_to_nusc_box(det)

def _nusc_attribute_name(detection_name: str, velocity_xy: Sequence[float]) -> str:
    from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
    name = str(detection_name)
    speed = math.hypot(float(velocity_xy[0]), float(velocity_xy[1])) if len(velocity_xy) >= 2 else 0.0
    if speed > 0.2:
        if name in ['car', 'construction_vehicle', 'bus', 'truck', 'trailer']:
            return 'vehicle.moving'
        if name in ['bicycle', 'motorcycle']:
            return 'cycle.with_rider'
        return NuScenesDataset.DefaultAttribute[name]
    if name == 'pedestrian':
        return 'pedestrian.standing'
    if name == 'bus':
        return 'vehicle.stopped'
    return NuScenesDataset.DefaultAttribute[name]

def _run_official_visualization(*, bevformer_repo_root: Path, results_path: Path, output_dir: Path, export_subdir: str, version: str, dataroot: Path, image_dataroot: Path, image_source_subdir: str, score_threshold: float, bevdet_info_pkl: Optional[Path]=None) -> None:
    visual_tool = bevformer_repo_root / 'tools' / 'visual.py'
    bevdet_visual_tool = bevformer_repo_root / 'tools' / 'analysis_tools' / 'vis.py'
    visual_log = output_dir / export_subdir / 'official_visual.log'
    plots_dir = output_dir / export_subdir / 'plots'
    if plots_dir.exists():
        shutil.rmtree(plots_dir, ignore_errors=True)
    if not visual_tool.exists() and bevdet_visual_tool.exists():
        if bevdet_info_pkl is None or not bevdet_info_pkl.exists():
            raise FileNotFoundError(f'BEVDet official vis requires info pkl: {bevdet_info_pkl}')
        results_dir = output_dir / export_subdir
        results_dir.mkdir(parents=True, exist_ok=True)
        vis_version = 'attacked'
        vis_info_pkl = results_dir / f'bevdetv3-nuscenes_infos_{vis_version}.pkl'
        shutil.copyfile(bevdet_info_pkl, vis_info_pkl)
        vehicle_results_path = results_dir / 'results_nusc_vehicle_visual.json'
        vehicle_filter_summary = _write_vehicle_only_results_for_visualization(results_path=results_path, output_path=vehicle_results_path)
        command = [sys.executable, str(bevdet_visual_tool), str(vehicle_results_path), '--root_path', str(results_dir), '--version', vis_version, '--save_path', str(plots_dir), '--format', 'image', '--vis-frames', '100000', '--vis-thred', str(float(score_threshold))]
        with visual_log.open('w', encoding='utf-8') as fp:
            fp.write('[visual] BEVDet vehicle-only visualization input: ' + json.dumps(vehicle_filter_summary, ensure_ascii=False) + '\n')
            subprocess.run(command, cwd=str(bevformer_repo_root), stdout=fp, stderr=subprocess.STDOUT, check=True)
        return
    lidar_source_dir = image_dataroot / image_source_subdir / 'LIDAR_TOP'
    if not lidar_source_dir.exists():
        lidar_source_dir = dataroot / image_source_subdir / 'LIDAR_TOP'
    samples_dir = dataroot / 'samples'
    lidar_alias_dir = samples_dir / 'LIDAR_TOP'
    sequence_alias_root = dataroot / image_source_subdir
    sequence_lidar_alias_dir = sequence_alias_root / 'LIDAR_TOP'
    created_alias_paths: List[Path] = []
    created_alias_dir = False
    created_sequence_alias_dir = False
    if lidar_source_dir.exists() and (not lidar_alias_dir.exists()):
        samples_dir.mkdir(parents=True, exist_ok=True)
        try:
            lidar_alias_dir.symlink_to(lidar_source_dir, target_is_directory=True)
            created_alias_paths.append(lidar_alias_dir)
        except OSError:
            lidar_alias_dir.mkdir(parents=True, exist_ok=True)
            created_alias_dir = True
            for lidar_file in lidar_source_dir.iterdir():
                alias_file = lidar_alias_dir / lidar_file.name
                if alias_file.exists():
                    continue
                alias_file.symlink_to(lidar_file)
                created_alias_paths.append(alias_file)
    if lidar_source_dir.exists() and image_source_subdir and (not sequence_lidar_alias_dir.exists()):
        sequence_alias_root.mkdir(parents=True, exist_ok=True)
        try:
            sequence_lidar_alias_dir.symlink_to(lidar_source_dir, target_is_directory=True)
            created_alias_paths.append(sequence_lidar_alias_dir)
        except OSError:
            sequence_lidar_alias_dir.mkdir(parents=True, exist_ok=True)
            created_sequence_alias_dir = True
            for lidar_file in lidar_source_dir.iterdir():
                alias_file = sequence_lidar_alias_dir / lidar_file.name
                if alias_file.exists():
                    continue
                alias_file.symlink_to(lidar_file)
                created_alias_paths.append(alias_file)
    if not visual_tool.exists():
        raise FileNotFoundError(f'Official visual.py not found: {visual_tool}')
    command = [sys.executable, str(visual_tool), '--results_path', str(results_path), '--version', str(version), '--dataroot', str(dataroot), '--image-dataroot', str(image_dataroot), '--image-source-subdir', str(image_source_subdir), '--workers', '1', '--score-thr', str(float(score_threshold))]
    try:
        with visual_log.open('w', encoding='utf-8') as fp:
            subprocess.run(command, cwd=str(bevformer_repo_root), stdout=fp, stderr=subprocess.STDOUT, check=True)
    finally:
        for path in reversed(created_alias_paths):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except IsADirectoryError:
                pass
        if created_sequence_alias_dir:
            try:
                sequence_lidar_alias_dir.rmdir()
            except OSError:
                pass
        try:
            if sequence_alias_root.exists() and sequence_alias_root != dataroot and (not any(sequence_alias_root.iterdir())):
                sequence_alias_root.rmdir()
        except OSError:
            pass
        if created_alias_dir:
            try:
                lidar_alias_dir.rmdir()
            except OSError:
                pass

def _export_official_results_and_visuals(*, config: Dict[str, Any], image_source_subdir: str, output_dir: Path, frames: Sequence[FrameRecord], bev_model: BevFormerGradientModel, sequence_pkl: Path, info_by_token: Dict[str, Dict[str, Any]], image_provider: Any, export_subdir: str='official', cleanup_before_test: Optional[Any]=None) -> Tuple[Path, Path]:
    dataset_cfg = config.get('dataset', {})
    model_cfg = _official_model_cfg_from_config(config)
    official_cfg = config.get('official_visual', {})
    score_threshold = float(official_cfg.get('score_threshold', 0.2))
    results_dir = output_dir / export_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    lidar_paths_by_sample = {str(frame.cache_key): _as_path(str(info_by_token[frame.sample_token]['lidar_path'])) for frame in frames if frame.sample_token in info_by_token and info_by_token[frame.sample_token].get('lidar_path')}
    image_root = _export_image_tree(frames=frames, output_dir=output_dir, export_subdir=export_subdir, image_source_subdir=image_source_subdir, image_provider=image_provider, lidar_paths_by_sample=lidar_paths_by_sample)
    selected_name = selected_model_name(config)
    if selected_name == 'stp3':
        try:
            results_path = results_dir / 'results_nusc.json'
            payload = bev_model.official_results_payload(frames, image_provider, score_threshold=score_threshold)
            _save_json(results_path, payload)
            visual_dir = (results_dir / 'plots').resolve()
            visual_dir.mkdir(parents=True, exist_ok=True)
            (visual_dir / 'STP3_PROXY_VISUALIZATION_SKIPPED.txt').write_text('STP3 has no official nuScenes detection head in this integration. The differentiable target proxy is the vehicle BEV probability map; fixed-query proxy rows are exported separately when enabled.\n', encoding='utf-8')
            return (results_path, visual_dir)
        finally:
            shutil.rmtree(image_root, ignore_errors=True)
    attacked_ann_file = results_dir / '.tmp_attacked_infos.pkl'
    attacked_config_path = results_dir / '.tmp_official_test_attacked.py'
    try:
        ann_payload_template = _load_sequence_ann_payload(sequence_pkl)
        patched_paths: Dict[str, Dict[str, str]] = {}
        sample_root = image_root / image_source_subdir
        for frame in frames:
            per_frame: Dict[str, str] = {}
            for channel in CAMERA_CHANNELS:
                camera = frame.cameras[channel]
                patched_path = _patched_camera_image_path(sample_root, channel, camera)
                per_frame[channel] = str(patched_path.resolve())
            patched_paths[frame.sample_token] = per_frame
        _write_official_attacked_ann_file(attacked_ann_file=attacked_ann_file, frames=frames, ann_payload_template=ann_payload_template, info_by_token=info_by_token, patched_paths=patched_paths)
        _write_official_test_config(bev_model=bev_model, attacked_ann_file=attacked_ann_file, config_path=attacked_config_path)
        if cleanup_before_test is not None:
            cleanup_before_test()
        results_path = _run_official_test(bev_model=bev_model, attacked_config_path=attacked_config_path, output_dir=output_dir, export_subdir=export_subdir)
        version = str(dataset_cfg.get('version', 'v1.0-trainval')).strip() or 'v1.0-trainval'
        dataroot = _as_path(str(dataset_cfg.get('dataroot', model_cfg.get('data_root', ''))))
        _run_official_visualization(bevformer_repo_root=_official_visual_repo_root_from_config(config), results_path=results_path, output_dir=output_dir, export_subdir=export_subdir, version=version, dataroot=dataroot, image_dataroot=image_root, image_source_subdir=image_source_subdir, score_threshold=score_threshold, bevdet_info_pkl=attacked_ann_file)
    finally:
        shutil.rmtree(image_root, ignore_errors=True)
        if attacked_ann_file.exists():
            attacked_ann_file.unlink()
        if attacked_config_path.exists():
            attacked_config_path.unlink()
    visual_dir = (results_dir / 'plots').resolve()
    if not visual_dir.exists():
        raise FileNotFoundError(f'Official visualization output not found: {visual_dir}')
    return (results_dir / 'results_nusc.json', visual_dir)

def _export_official_results_and_visuals_from_image_root(*, config: Dict[str, Any], image_source_subdir: str, image_root: Path, output_dir: Path, frames: Sequence[FrameRecord], bev_model: BevFormerGradientModel, sequence_pkl: Path, info_by_token: Dict[str, Dict[str, Any]], export_subdir: str='official', cleanup_before_test: Optional[Any]=None, image_provider: Optional[Any]=None) -> Tuple[Path, Path]:
    dataset_cfg = config.get('dataset', {})
    model_cfg = _official_model_cfg_from_config(config)
    official_cfg = config.get('official_visual', {})
    score_threshold = float(official_cfg.get('score_threshold', 0.2))
    results_dir = output_dir / export_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    selected_name = selected_model_name(config)
    if selected_name == 'stp3':
        if image_provider is None:
            raise ValueError('STP3 proxy export requires image_provider')
        results_path = results_dir / 'results_nusc.json'
        payload = bev_model.official_results_payload(frames, image_provider, score_threshold=score_threshold)
        _save_json(results_path, payload)
        visual_dir = (results_dir / 'plots').resolve()
        visual_dir.mkdir(parents=True, exist_ok=True)
        (visual_dir / 'STP3_PROXY_VISUALIZATION_SKIPPED.txt').write_text('STP3 has no official nuScenes detection head in this integration. The differentiable target proxy is the vehicle BEV probability map; fixed-query proxy rows are exported separately when enabled.\n', encoding='utf-8')
        return (results_path, visual_dir)
    attacked_ann_file = results_dir / '.tmp_attacked_infos.pkl'
    attacked_config_path = results_dir / '.tmp_official_test_attacked.py'
    try:
        ann_payload_template = _load_sequence_ann_payload(sequence_pkl)
        patched_paths: Dict[str, Dict[str, str]] = {}
        sample_root = image_root / image_source_subdir
        for frame in frames:
            per_frame: Dict[str, str] = {}
            for channel in CAMERA_CHANNELS:
                camera = frame.cameras[channel]
                patched_path = _patched_camera_image_path(sample_root, channel, camera)
                per_frame[channel] = str(patched_path.resolve())
            patched_paths[frame.sample_token] = per_frame
        _write_official_attacked_ann_file(attacked_ann_file=attacked_ann_file, frames=frames, ann_payload_template=ann_payload_template, info_by_token=info_by_token, patched_paths=patched_paths)
        _write_official_test_config(bev_model=bev_model, attacked_ann_file=attacked_ann_file, config_path=attacked_config_path)
        if cleanup_before_test is not None:
            cleanup_before_test()
        results_path = _run_official_test(bev_model=bev_model, attacked_config_path=attacked_config_path, output_dir=output_dir, export_subdir=export_subdir)
        version = str(dataset_cfg.get('version', 'v1.0-trainval')).strip() or 'v1.0-trainval'
        dataroot = _as_path(str(dataset_cfg.get('dataroot', model_cfg.get('data_root', ''))))
        _run_official_visualization(bevformer_repo_root=_official_visual_repo_root_from_config(config), results_path=results_path, output_dir=output_dir, export_subdir=export_subdir, version=version, dataroot=dataroot, image_dataroot=image_root, image_source_subdir=image_source_subdir, score_threshold=score_threshold, bevdet_info_pkl=attacked_ann_file)
    finally:
        if attacked_ann_file.exists():
            attacked_ann_file.unlink()
        if attacked_config_path.exists():
            attacked_config_path.unlink()
    visual_dir = (results_dir / 'plots').resolve()
    if not visual_dir.exists():
        raise FileNotFoundError(f'Official visualization output not found: {visual_dir}')
    return (results_dir / 'results_nusc.json', visual_dir)

def _export_val_official_results(*, config: Dict[str, Any], output_dir: Path, val_frame_groups: Sequence[Sequence[FrameRecord]], val_renderer: FixedUVTextureRenderer, bev_model: BevFormerGradientModel, val_single_sequence_pkl_map: Dict[str, Path], val_sequence_pkl: Optional[Path], fallback_sequence_pkl: Path, val_info_by_sequence_token: Dict[str, Dict[str, Dict[str, Any]]], val_info_by_cache_key: Dict[str, Dict[str, Any]], val_fixed_queries: Dict[str, FixedQueryMatch], active_model_name: str, is_bevdet_model: bool, val_use_eot: bool, fixed_conf_threshold: float, fixed_max_center_dist_m: float, fixed_distance_axis: str, fixed_max_cross_axis_dist_m: float, training_log_path: Path, export_subdir_root: str, export_label: str) -> List[Dict[str, Any]]:
    val_records: List[Dict[str, Any]] = []
    val_lidar_paths_by_sample = {str(frame.cache_key): _as_path(str(val_info_by_cache_key[frame.cache_key]['lidar_path'])) for group_frames in val_frame_groups for frame in group_frames if frame.cache_key in val_info_by_cache_key and val_info_by_cache_key[frame.cache_key].get('lidar_path')}
    for group_frames in val_frame_groups:
        sequence_name = str(group_frames[0].sequence_name)
        group_sequence_pkl = val_single_sequence_pkl_map.get(sequence_name, val_sequence_pkl if val_sequence_pkl is not None else fallback_sequence_pkl)
        val_export_subdir = f'{export_subdir_root}/{sequence_name}' if len(val_frame_groups) > 1 else export_subdir_root
        val_image_root = _export_image_tree(frames=group_frames, output_dir=output_dir, export_subdir=val_export_subdir, image_source_subdir=sequence_name, image_provider=lambda frame: val_renderer.build_frame_images(frame, apply_eot=val_use_eot), lidar_paths_by_sample=val_lidar_paths_by_sample)
        try:
            val_results_path, val_visual_dir = _export_official_results_and_visuals_from_image_root(config=config, image_source_subdir=sequence_name, image_root=val_image_root, output_dir=output_dir, frames=group_frames, bev_model=bev_model, sequence_pkl=group_sequence_pkl, info_by_token=val_info_by_sequence_token[sequence_name], export_subdir=val_export_subdir, cleanup_before_test=None, image_provider=lambda frame: val_renderer.build_frame_images(frame, apply_eot=val_use_eot))
        finally:
            shutil.rmtree(val_image_root, ignore_errors=True)
        val_record: Dict[str, Any] = {'step': export_label, 'sequence_name': sequence_name, 'results_path': str(val_results_path), 'visual_dir': str(val_visual_dir)}
        val_ok_line = f'[val] official export done {export_label} seq={sequence_name}: results={val_results_path} visuals={val_visual_dir}'
        print(val_ok_line)
        _append_log_line(training_log_path, val_ok_line)
        val_trace_path = val_results_path.parent / 'results_nusc_query_trace.json'
        if val_trace_path.exists() and (not is_bevdet_model):
            val_trace_check = _validate_fixed_query_trace_identity(frames=group_frames, fixed_queries=val_fixed_queries, official_query_trace_path=val_trace_path)
            val_record.update({'mode': 'official_fixed_query_trace', **val_trace_check})
            val_trace_line = f"[val] official fixed-query check {export_label} seq={sequence_name} passed={val_trace_check['passed']}/{val_trace_check['total']} failed={val_trace_check['failed']} missing={val_trace_check['missing']}"
            print(val_trace_line)
            _append_log_line(training_log_path, val_trace_line)
        else:
            val_record['mode'] = 'official_results_only'
        val_records.append(copy.deepcopy(val_record))
    return val_records

def _export_val_official_current_results(*, config: Dict[str, Any], output_dir: Path, val_frame_groups: Sequence[Sequence[FrameRecord]], val_renderer: FixedUVTextureRenderer, bev_model: BevFormerGradientModel, val_single_sequence_pkl_map: Dict[str, Path], val_sequence_pkl: Optional[Path], fallback_sequence_pkl: Path, val_info_by_sequence_token: Dict[str, Dict[str, Dict[str, Any]]], val_info_by_cache_key: Dict[str, Dict[str, Any]], val_fixed_queries: Dict[str, FixedQueryMatch], active_model_name: str, is_bevdet_model: bool, val_use_eot: bool, fixed_conf_threshold: float, fixed_max_center_dist_m: float, fixed_distance_axis: str, fixed_max_cross_axis_dist_m: float, training_log_path: Path) -> List[Dict[str, Any]]:
    return _export_val_official_results(config=config, output_dir=output_dir, val_frame_groups=val_frame_groups, val_renderer=val_renderer, bev_model=bev_model, val_single_sequence_pkl_map=val_single_sequence_pkl_map, val_sequence_pkl=val_sequence_pkl, fallback_sequence_pkl=fallback_sequence_pkl, val_info_by_sequence_token=val_info_by_sequence_token, val_info_by_cache_key=val_info_by_cache_key, val_fixed_queries=val_fixed_queries, active_model_name=active_model_name, is_bevdet_model=is_bevdet_model, val_use_eot=val_use_eot, fixed_conf_threshold=fixed_conf_threshold, fixed_max_center_dist_m=fixed_max_center_dist_m, fixed_distance_axis=fixed_distance_axis, fixed_max_cross_axis_dist_m=fixed_max_cross_axis_dist_m, training_log_path=training_log_path, export_subdir_root='val_official/current', export_label='current')

def _save_eval_main_report_for_sequence(*, sequence_name: str, source_tag: str, mode: str, rows: Sequence[Dict[str, Any]], output_dir: Path=EVAL_MAIN_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{sequence_name}.json'
    if output_path.exists():
        try:
            payload = json.loads(output_path.read_text(encoding='utf-8'))
        except Exception:
            payload = {}
    else:
        payload = {}
    report_rows: List[Dict[str, Any]] = []
    matched_count = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        matched = bool(row.get('matched', False))
        if matched:
            matched_count += 1
        report_rows.append({'frame_id': int(row.get('frame_id', -1)), 'sample_token': str(row.get('sample_token', '')), 'matched': matched, 'target_detection_name': str(row.get('target_detection_name', '')), 'clean_projected_distance_to_ego_front_line_m': _format_float_or_none(row.get('clean_projected_distance_to_ego_front_line_m')), 'attacked_moved_toward_ego_front_line_m': _format_float_or_none(row.get('attacked_moved_toward_ego_front_line_m')), 'attacked_moved_toward_ego_front_line_pct': _format_float_or_none(row.get('attacked_moved_toward_ego_front_line_pct')), 'reason': str(row.get('reason', '')) if not matched else ''})
    payload['sample'] = str(sequence_name)
    reports = payload.get('reports', {})
    if not isinstance(reports, dict):
        reports = {}
    reports[str(source_tag)] = {'mode': str(mode), 'frame_count': int(len(report_rows)), 'matched_frame_count': int(matched_count), 'frames': report_rows}
    payload['reports'] = reports
    _save_json(output_path, payload)
    return output_path

def run_training(config_path: Path, resume_ckpt: Optional[Path]=None) -> Dict[str, Any]:
    config_path = _as_path(config_path)
    config = _load_yaml(config_path)
    train_cfg = config.get('train', {})
    resume_ckpt_path = _resolve_resume_ckpt_path(resume_ckpt)
    resume_payload: Optional[Dict[str, Any]] = None
    if resume_ckpt_path is not None:
        resume_payload = torch.load(str(resume_ckpt_path), map_location='cpu')
        output_dir = _as_path(str(resume_payload.get('output_dir', resume_ckpt_path.parent.parent)))
    else:
        output_dir = _resolve_new_output_dir(config=config, config_path=config_path, train_cfg=train_cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_sentinel_path = output_dir / '.training_output_alive'
    output_dir_sentinel_path.write_text(datetime.now(CHINA_TZ).isoformat(), encoding='utf-8')
    config_snapshot_path = _save_config_snapshot(config_path=config_path, output_dir=output_dir)
    training_log_path = output_dir / 'training.log'
    _append_log_line(training_log_path, '[train] ===== start =====')
    _append_log_line(training_log_path, f'[train] config_path={config_path}')
    _append_log_line(training_log_path, f'[train] config_snapshot={config_snapshot_path}')
    _append_log_line(training_log_path, f'[train] output_dir={output_dir}')
    _append_log_line(training_log_path, f"[train] resume_ckpt={(resume_ckpt_path if resume_ckpt_path is not None else '')}")
    seed = int(train_cfg.get('seed', 42))
    _set_seed(seed)
    speed_flags = _configure_torch_speed_flags(train_cfg)
    _append_log_line(training_log_path, f"[train] torch_perf cudnn_benchmark={str(speed_flags['cudnn_benchmark']).lower()} allow_tf32={str(speed_flags['allow_tf32']).lower()}")
    sequence_yaml_paths = _sequence_yaml_paths_from_config(config, config_path)
    val_sequence_yaml_paths = _val_sequence_yaml_paths_from_config(config, config_path)
    try:
        _append_log_line(training_log_path, '[train] preprocess: target match / mesh projection / SAM2')
        binding_payload, mesh_summary, sam_summary = _prepare_precompute_outputs(config=config, config_path=config_path, output_root=output_dir, sequence_yaml_paths=sequence_yaml_paths, stage_label='train')
        _append_log_line(training_log_path, '[train] preprocess done')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        single_sequence_pkl_map: Dict[str, Path] = {}
        total_sequence_pkls = len(sequence_yaml_paths)
        for index, path in enumerate(sequence_yaml_paths, start=1):
            _print_and_append_log_line(training_log_path, f'[train] prepare single-sequence pkl {index}/{total_sequence_pkls}: {path.stem}')
            single_sequence_pkl_map[str(path.stem)] = _ensure_single_sequence_pkl(config=config, config_path=config_path, sequence_yaml_path=path)
            _print_and_append_log_line(training_log_path, f'[train] single-sequence pkl done {index}/{total_sequence_pkls}: {single_sequence_pkl_map[str(path.stem)]}')
        _print_and_append_log_line(training_log_path, '[train] merge sequence pkls')
        sequence_pkl = _sequence_pkl_from_config(config, config_path)
        image_source_subdir = _image_source_group_from_config(config, config_path)
        _append_log_line(training_log_path, f'[train] sequence_pkl={sequence_pkl}')
        _print_and_append_log_line(training_log_path, f'[train] merged sequence pkl: {sequence_pkl}')
        info_by_sequence_token = _load_sequence_infos_by_sequence(single_sequence_pkl_map)
        frames = _build_frame_records(binding_payload, info_by_sequence_token)
        frame_groups = _group_frames_by_sequence_name(frames)
        info_by_cache_key = {frame.cache_key: info_by_sequence_token[frame.sequence_name][frame.sample_token] for frame in frames if frame.sequence_name in info_by_sequence_token and frame.sample_token in info_by_sequence_token[frame.sequence_name]}
        _append_log_line(training_log_path, f'[train] num_frames={len(frames)}')
        val_enabled = bool(val_sequence_yaml_paths)
        val_single_sequence_pkl_map: Dict[str, Path] = {}
        val_sequence_pkl: Optional[Path] = None
        val_image_source_subdir = ''
        val_info_by_sequence_token: Dict[str, Dict[str, Dict[str, Any]]] = {}
        val_info_by_cache_key: Dict[str, Dict[str, Any]] = {}
        val_frames: List[FrameRecord] = []
        val_frame_groups: List[List[FrameRecord]] = []
        val_binding_payload: Optional[Dict[str, Any]] = None
        val_sam_summary: Optional[Dict[str, Any]] = None
        if val_enabled:
            _append_log_line(training_log_path, '[val] preprocess: target match / mesh projection / SAM2')
            val_binding_payload, _, val_sam_summary = _prepare_precompute_outputs(config=config, config_path=config_path, output_root=output_dir, sequence_yaml_paths=val_sequence_yaml_paths, stage_label='val')
            _append_log_line(training_log_path, '[val] preprocess done')
            val_single_sequence_pkl_map = {}
            total_val_sequence_pkls = len(val_sequence_yaml_paths)
            for index, path in enumerate(val_sequence_yaml_paths, start=1):
                _print_and_append_log_line(training_log_path, f'[val] prepare single-sequence pkl {index}/{total_val_sequence_pkls}: {path.stem}')
                val_single_sequence_pkl_map[str(path.stem)] = _ensure_single_sequence_pkl(config=config, config_path=config_path, sequence_yaml_path=path)
                _print_and_append_log_line(training_log_path, f'[val] single-sequence pkl done {index}/{total_val_sequence_pkls}: {val_single_sequence_pkl_map[str(path.stem)]}')
            _print_and_append_log_line(training_log_path, '[val] merge sequence pkls')
            val_sequence_pkl = _sequence_pkl_from_paths(config, config_path, val_sequence_yaml_paths, explicit_pkl_key='val_sequence_pkl')
            _print_and_append_log_line(training_log_path, f'[val] merged sequence pkl: {val_sequence_pkl}')
            val_image_source_subdir = _sequence_group_key_from_paths(val_sequence_yaml_paths)
            val_info_by_sequence_token = _load_sequence_infos_by_sequence(val_single_sequence_pkl_map)
            val_frames = _build_frame_records(val_binding_payload, val_info_by_sequence_token)
            val_frame_groups = _group_frames_by_sequence_name(val_frames)
            val_info_by_cache_key = {frame.cache_key: val_info_by_sequence_token[frame.sequence_name][frame.sample_token] for frame in val_frames if frame.sequence_name in val_info_by_sequence_token and frame.sample_token in val_info_by_sequence_token[frame.sequence_name]}
            _append_log_line(training_log_path, f'[val] sequence_pkl={val_sequence_pkl}')
            _append_log_line(training_log_path, f'[val] num_frames={len(val_frames)}')
        device_name = str(train_cfg.get('device', 'cuda')).strip().lower()
        device = torch.device('cuda' if device_name == 'cuda' and torch.cuda.is_available() else 'cpu')
        _append_log_line(training_log_path, f'[train] requested_device={device_name} actual_device={device}')
        if device_name == 'cuda' and device.type != 'cuda':
            raise RuntimeError('config train.device=cuda but cuda is not available')
        _append_log_line(training_log_path, '[train] init renderer')
        renderer = FixedUVTextureRenderer(mesh_obj_path=_mesh_obj_path_from_config(config), frames=frames, sam_summary=sam_summary, texture_cfg=config.get('camouflage', {}), device=device, alpha=float(config.get('camouflage', {}).get('alpha', 1.0)), preload_clean_images_to_device=bool(train_cfg.get('preload_clean_images_to_device', False)))
        val_renderer: Optional[FixedUVTextureRenderer] = None
        if val_enabled:
            if val_sam_summary is None:
                raise RuntimeError('validation enabled but val_sam_summary is missing')
            _append_log_line(training_log_path, '[val] init val renderer')
            val_renderer = FixedUVTextureRenderer(mesh_obj_path=_mesh_obj_path_from_config(config), frames=val_frames, sam_summary=val_sam_summary, texture_cfg=config.get('camouflage', {}), device=device, alpha=float(config.get('camouflage', {}).get('alpha', 1.0)), preload_clean_images_to_device=bool(train_cfg.get('preload_clean_images_to_device', False)))
            val_renderer.texture_param = renderer.texture_param
            val_renderer.texture_anchor = renderer.texture_anchor
        active_model_name = selected_model_name(config)
        is_bevdet_model = active_model_name in {'bevdet', 'bevdepth', 'fastbev'}
        uses_final_decode_match = active_model_name in {'bevdet', 'bevdepth', 'fastbev'}
        resolved_loss_cfg, loss_cfg_source = _resolved_loss_cfg_from_config(config, model_name=active_model_name)
        use_amp_requested = bool(train_cfg.get('use_amp', True)) and device.type == 'cuda'
        amp_dtype_name = str(train_cfg.get('amp_dtype', 'fp16')).strip().lower()
        amp_dtype = torch.float16 if amp_dtype_name in {'fp16', 'float16'} else torch.bfloat16
        use_amp = use_amp_requested
        amp_override_note = ''
        if active_model_name in {'bevdet', 'bevdepth', 'fastbev'} and use_amp and (amp_dtype == torch.float16):
            use_amp = False
            amp_override_note = f'[train] model={active_model_name} with amp_dtype=fp16: disabling amp to avoid GradScaler overflow (use_amp=false)'
        _append_log_line(training_log_path, f'[train] init and load {active_model_name}')
        if amp_override_note:
            _append_log_line(training_log_path, amp_override_note)
        bev_model = build_gradient_model(config=config, device=device_name, use_amp=use_amp, amp_dtype=amp_dtype_name)
        bev_model.build()

        def _clean_image_provider(frame: FrameRecord) -> Dict[str, torch.Tensor]:
            return {channel: renderer.clean_image(frame, channel) for channel in CAMERA_CHANNELS}

        def _initial_patch_image_provider(frame: FrameRecord) -> Dict[str, torch.Tensor]:
            return renderer.build_frame_images(frame, apply_eot=False)
        fixed_query_cfg = config.get('fixed_query', {})
        fixed_conf_threshold = float(fixed_query_cfg.get('conf_threshold', 0.6))
        fixed_max_center_dist_m = float(fixed_query_cfg.get('max_center_dist_m', 1.0))
        fixed_distance_axis = str(fixed_query_cfg.get('distance_axis', 'lateral_y'))
        fixed_max_cross_axis_dist_m = float(fixed_query_cfg.get('max_cross_axis_dist_m', fixed_query_cfg.get('max_longitudinal_dist_m', 1.0)))
        fixed_center_cost_weight = float(fixed_query_cfg.get('cost_center_xy_weight', 1.0))
        fixed_confidence_cost_weight = float(fixed_query_cfg.get('cost_confidence_weight', 0.5))
        loss_reference_mode = _loss_reference_mode_from_config(config)
        if loss_reference_mode == 'clean':
            loss_reference_line = '[train] loss uses clean detections; frames without clean refs fall back to GT'
        else:
            loss_reference_line = '[train] loss uses GT references'
        print(loss_reference_line)
        _append_log_line(training_log_path, loss_reference_line)
        val_fixed_queries: Dict[str, FixedQueryMatch] = {}
        if uses_final_decode_match:
            _append_log_line(training_log_path, f'[train] model={active_model_name}: initial texture then final-decode target lock; train_query_source=final_decode')
            fixed_queries = _match_final_decode_queries(frames=frames, image_provider=_initial_patch_image_provider, bev_model=bev_model, device=device, use_amp=use_amp, amp_dtype=amp_dtype, conf_threshold=fixed_conf_threshold, max_center_dist_m=fixed_max_center_dist_m, distance_axis=fixed_distance_axis, max_cross_axis_dist_m=fixed_max_cross_axis_dist_m)
            matched_queries = [query for query in fixed_queries.values() if query.matched]
            if not matched_queries:
                details = [f'frame={query.frame_id}, reason={query.unmatched_reason}, after_conf={query.candidate_after_conf}, after_dist={query.candidate_after_dist}' for query in fixed_queries.values()]
                no_initial_match_message = f'[train] no {active_model_name} target locked before training; continuing per-step final decode; unmatched frames skipped.\n' + '\n'.join(details)
                print(no_initial_match_message)
                _append_log_line(training_log_path, no_initial_match_message)
            if val_enabled:
                if val_renderer is None:
                    raise RuntimeError('val renderer not initialized')

                def _val_initial_patch_image_provider(frame: FrameRecord) -> Dict[str, torch.Tensor]:
                    return val_renderer.build_frame_images(frame, apply_eot=False)
                _append_log_line(training_log_path, f'[val] initial texture then final-decode lock for {active_model_name}')
                val_fixed_queries = _match_final_decode_queries(frames=val_frames, image_provider=_val_initial_patch_image_provider, bev_model=bev_model, device=device, use_amp=use_amp, amp_dtype=amp_dtype, conf_threshold=fixed_conf_threshold, max_center_dist_m=fixed_max_center_dist_m, distance_axis=fixed_distance_axis, max_cross_axis_dist_m=fixed_max_cross_axis_dist_m)
        else:
            _append_log_line(training_log_path, '[train] lock fixed query')
            fixed_queries = bev_model.match_target_queries(frames, _clean_image_provider, conf_threshold=fixed_conf_threshold, max_center_dist_m=fixed_max_center_dist_m, center_cost_weight=fixed_center_cost_weight, confidence_cost_weight=fixed_confidence_cost_weight)
            matched_queries = [query for query in fixed_queries.values() if query.matched]
            if not matched_queries:
                details = [f'frame={query.frame_id}, reason={query.unmatched_reason}, after_conf={query.candidate_after_conf}, after_dist={query.candidate_after_dist}' for query in fixed_queries.values()]
                raise RuntimeError(f'no target candidate locked before training. Check fixed_query.conf_threshold / fixed_query.max_center_dist_m or adjust {active_model_name} preset.\n' + '\n'.join(details))
        if val_enabled and (not uses_final_decode_match):
            if val_renderer is None:
                raise RuntimeError('val renderer not initialized')

            def _val_clean_image_provider(frame: FrameRecord) -> Dict[str, torch.Tensor]:
                return {channel: val_renderer.clean_image(frame, channel) for channel in CAMERA_CHANNELS}
            _append_log_line(training_log_path, '[val] lock fixed query')
            val_fixed_queries = bev_model.match_target_queries(val_frames, _val_clean_image_provider, conf_threshold=fixed_conf_threshold, max_center_dist_m=fixed_max_center_dist_m, center_cost_weight=fixed_center_cost_weight, confidence_cost_weight=fixed_confidence_cost_weight)
            matched_val_queries = [query for query in val_fixed_queries.values() if query.matched]
            if not matched_val_queries:
                details = [f'frame={query.frame_id}, reason={query.unmatched_reason}, after_conf={query.candidate_after_conf}, after_dist={query.candidate_after_dist}' for query in val_fixed_queries.values()]
                raise RuntimeError('val set: no target candidate locked. Check fixed_query.conf_threshold / fixed_query.max_center_dist_m.\n' + '\n'.join(details))
        train_clean_detection_refs: Dict[str, Dict[str, torch.Tensor]] = {}
        val_clean_detection_refs: Dict[str, Dict[str, torch.Tensor]] = {}
        if loss_reference_mode == 'clean':
            _append_log_line(training_log_path, f'[train] loss_ref=clean; caching {active_model_name} clean detections')
            train_clean_detection_refs = _load_or_compute_clean_detection_references(config=config, config_path=config_path, sequence_yaml_paths=sequence_yaml_paths, frame_groups=frame_groups, renderer=renderer, bev_model=bev_model, fixed_queries=fixed_queries, device=device, use_amp=use_amp, amp_dtype=amp_dtype, model_name=active_model_name, final_decode_match=bool(uses_final_decode_match), fixed_conf_threshold=fixed_conf_threshold, fixed_max_center_dist_m=fixed_max_center_dist_m, fixed_distance_axis=fixed_distance_axis, fixed_max_cross_axis_dist_m=fixed_max_cross_axis_dist_m, training_log_path=training_log_path, split_label='train')
            _append_log_line(training_log_path, f'[train] clean det refs ready={len(train_clean_detection_refs)}/{len(frames)}')
            _log_clean_reference_coverage(training_log_path=training_log_path, prefix='[train]', frames=frames, clean_detection_refs=train_clean_detection_refs)
            if val_enabled and val_renderer is not None:
                _append_log_line(training_log_path, f'[val] loss_ref=clean; caching {active_model_name} clean detections')
                val_clean_detection_refs = _load_or_compute_clean_detection_references(config=config, config_path=config_path, sequence_yaml_paths=val_sequence_yaml_paths, frame_groups=val_frame_groups, renderer=val_renderer, bev_model=bev_model, fixed_queries=val_fixed_queries, device=device, use_amp=use_amp, amp_dtype=amp_dtype, model_name=active_model_name, final_decode_match=bool(uses_final_decode_match), fixed_conf_threshold=fixed_conf_threshold, fixed_max_center_dist_m=fixed_max_center_dist_m, fixed_distance_axis=fixed_distance_axis, fixed_max_cross_axis_dist_m=fixed_max_cross_axis_dist_m, training_log_path=training_log_path, split_label='val')
                _append_log_line(training_log_path, f'[val] clean det refs ready={len(val_clean_detection_refs)}/{len(val_frames)}')
                _log_clean_reference_coverage(training_log_path=training_log_path, prefix='[val]', frames=val_frames, clean_detection_refs=val_clean_detection_refs)
        else:
            _append_log_line(training_log_path, '[train] loss_ref=GT')
        auxiliary_runtimes: List[AuxiliaryModelRuntime] = []
        for aux_spec in _auxiliary_model_specs_from_config(config, active_model_name=active_model_name):
            aux_model_name = str(aux_spec['model'])
            aux_weight = float(aux_spec.get('weight', 1.0))
            if abs(aux_weight) <= 1e-12:
                _append_log_line(training_log_path, f'[aux:{aux_model_name}] weight=0; skip')
                continue
            aux_config = copy.deepcopy(config)
            aux_config['model'] = aux_model_name
            aux_use_amp = use_amp
            aux_amp_note = ''
            if aux_model_name in {'bevdet', 'bevdepth', 'fastbev'} and aux_use_amp and (amp_dtype == torch.float16):
                aux_use_amp = False
                aux_amp_note = f'; amp forced to fp32 for {aux_model_name}'
            _append_log_line(training_log_path, f'[aux:{aux_model_name}] init and load; weight={aux_weight}{aux_amp_note}')
            aux_model = build_gradient_model(config=aux_config, device=device_name, use_amp=aux_use_amp, amp_dtype=amp_dtype_name)
            aux_model.build()
            _append_log_line(training_log_path, f'[aux:{aux_model_name}] record clean target queries')
            aux_fixed_queries = aux_model.match_target_queries(frames, _clean_image_provider, conf_threshold=fixed_conf_threshold, max_center_dist_m=fixed_max_center_dist_m, center_cost_weight=fixed_center_cost_weight, confidence_cost_weight=fixed_confidence_cost_weight)
            aux_matched_queries = [query for query in aux_fixed_queries.values() if query.matched]
            if not aux_matched_queries:
                details = [f'frame={query.frame_id}, reason={query.unmatched_reason}, after_conf={query.candidate_after_conf}, after_dist={query.candidate_after_dist}' for query in aux_fixed_queries.values()]
                raise RuntimeError(f'auxiliary model={aux_model_name}: no target candidate locked. Check fixed_query or model config.\n' + '\n'.join(details))
            aux_resolved_loss_cfg, aux_loss_cfg_source = _resolved_loss_cfg_from_config(aux_config, model_name=aux_model_name)
            aux_clean_detection_refs: Dict[str, Dict[str, torch.Tensor]] = {}
            if loss_reference_mode == 'clean':
                _append_log_line(training_log_path, f'[aux:{aux_model_name}] cache clean detections (loss ref)')
                aux_clean_detection_refs = _load_or_compute_clean_detection_references(config=aux_config, config_path=config_path, sequence_yaml_paths=sequence_yaml_paths, frame_groups=frame_groups, renderer=renderer, bev_model=aux_model, fixed_queries=aux_fixed_queries, device=device, use_amp=bool(aux_use_amp), amp_dtype=amp_dtype, model_name=aux_model_name, final_decode_match=bool(aux_model_name in {'bevdet', 'bevdepth', 'fastbev'}), fixed_conf_threshold=fixed_conf_threshold, fixed_max_center_dist_m=fixed_max_center_dist_m, fixed_distance_axis=fixed_distance_axis, fixed_max_cross_axis_dist_m=fixed_max_cross_axis_dist_m, training_log_path=training_log_path, split_label=f'aux:{aux_model_name}')
                _log_clean_reference_coverage(training_log_path=training_log_path, prefix=f'[aux:{aux_model_name}]', frames=frames, clean_detection_refs=aux_clean_detection_refs)
            aux_loss_cfg_merged = _merge_optimal_into_loss_cfg(_train_config_to_loss_cfg(aux_resolved_loss_cfg), config=aux_config)
            _merge_stp3_new_loss_config(aux_loss_cfg_merged, config=aux_config)
            aux_runtime = AuxiliaryModelRuntime(model_name=aux_model_name, model=aux_model, fixed_queries=aux_fixed_queries, clean_detection_refs=aux_clean_detection_refs, loss_reference_mode=loss_reference_mode, loss_cfg=aux_loss_cfg_merged, weight=aux_weight, matched_frame_total=len(aux_matched_queries), final_decode_match=bool(aux_model_name in {'bevdet', 'bevdepth', 'fastbev'}), use_amp=bool(aux_use_amp), amp_dtype_name=amp_dtype_name)
            auxiliary_runtimes.append(aux_runtime)
            _append_log_line(training_log_path, f'[aux:{aux_model_name}] ready: matched_frames={len(aux_matched_queries)}/{len(frames)} loss_cfg_source={aux_loss_cfg_source} final_decode_match={aux_runtime.final_decode_match}')
        clean_results_by_sequence: Dict[str, str] = {}
        clean_visuals_by_sequence: Dict[str, str] = {}
        clean_query_features: Dict[str, torch.Tensor] = {}
        val_clean_query_features: Dict[str, torch.Tensor] = {}
        raw_loss_cfg = resolved_loss_cfg if isinstance(resolved_loss_cfg, dict) else {}
        raw_cls_cfg = raw_loss_cfg.get('cls', {}) if isinstance(raw_loss_cfg.get('cls', {}), dict) else {}
        if not is_bevdet_model and float(raw_cls_cfg.get('weight', 1.0)) * float(raw_cls_cfg.get('query_identity_weight', 0.0)) > 0.0:
            _append_log_line(training_log_path, '[train] cache clean query features (identity loss)')
            clean_query_features = _collect_clean_query_features(frame_groups=frame_groups, renderer=renderer, bev_model=bev_model, fixed_queries=fixed_queries, device=device, use_amp=use_amp, amp_dtype=amp_dtype)
            _append_log_line(training_log_path, f'[train] clean query features cached={len(clean_query_features)}/{len(matched_queries)}')
            if val_enabled and val_renderer is not None:
                _append_log_line(training_log_path, '[val] cache clean query features (identity loss)')
                val_matched = [query for query in val_fixed_queries.values() if query.matched]
                val_clean_query_features = _collect_clean_query_features(frame_groups=val_frame_groups, renderer=val_renderer, bev_model=bev_model, fixed_queries=val_fixed_queries, device=device, use_amp=use_amp, amp_dtype=amp_dtype)
                _append_log_line(training_log_path, f'[val] clean query features cached={len(val_clean_query_features)}/{len(val_matched)}')
        if bool(train_cfg.get('clean_run', False)):
            _append_log_line(training_log_path, '[train] export pre-train clean official results and visuals')
            for group_frames in frame_groups:
                sequence_name = str(group_frames[0].sequence_name)
                group_sequence_pkl = single_sequence_pkl_map.get(sequence_name, sequence_pkl)
                export_subdir = f'clean/{sequence_name}' if len(frame_groups) > 1 else 'clean'
                clean_results_path, clean_visual_dir = _export_official_results_and_visuals(config=config, image_source_subdir=sequence_name, output_dir=output_dir, frames=group_frames, bev_model=bev_model, sequence_pkl=group_sequence_pkl, info_by_token=info_by_sequence_token[sequence_name], image_provider=_clean_image_provider, export_subdir=export_subdir)
                clean_results_by_sequence[sequence_name] = str(clean_results_path)
                clean_visuals_by_sequence[sequence_name] = str(clean_visual_dir)
                if not is_bevdet_model:
                    clean_overlay_summary = _annotate_target_confidence_on_visuals(config=config, frames=group_frames, bev_model=bev_model, results_path=clean_results_path, visual_dir=clean_visual_dir)
                    _append_log_line(training_log_path, '[train] clean visuals target-conf overlay ' + f"seq={sequence_name} ok={clean_overlay_summary.get('ok', False)} " + f"updated={clean_overlay_summary.get('updated', 0)} " + f"frames={clean_overlay_summary.get('total_frames', 0)} " + f"token_mapped={clean_overlay_summary.get('token_mapped', 0)} " + f"with_conf={clean_overlay_summary.get('with_target_confidence', 0)} " + f"error={clean_overlay_summary.get('error', '')}")
                clean_ok_message = f'[train] clean official results saved seq={sequence_name}: results={clean_results_path} visuals={clean_visual_dir}'
                print(clean_ok_message)
                _append_log_line(training_log_path, clean_ok_message)
        else:
            _append_log_line(training_log_path, '[train] clean_run=false; skip pre-train clean official export')
        matched_frame_total = len(matched_queries)
        best_min_target_confidence = float(config.get('official_visual', {}).get('score_threshold', 0.5))
        total_steps = int(train_cfg.get('steps', 300))
        log_every = max(1, int(train_cfg.get('log_every', 10)))
        batch_mode = str(train_cfg.get('batch_mode', 'full')).strip().lower()
        train_samples_per_step = max(1, int(train_cfg.get('train_samples_per_step', 2)))
        train_eval_every = max(1, int(train_cfg.get('train_eval_every', log_every)))
        val_eval_every = max(1, int(train_cfg.get('val_eval_every', train_eval_every)))
        val_log_every = max(1, int(train_cfg.get('val_log_every', log_every)))
        checkpoint_every = max(1, int(train_cfg.get('checkpoint_every', 10)))
        plot_every_raw = int(train_cfg.get('plot_every', val_eval_every if val_enabled else checkpoint_every))
        plot_every = max(0, plot_every_raw)

        def _should_plot_training_curves(step_index: int) -> bool:
            return bool(plot_every > 0 and (int(step_index) % plot_every == 0 or int(step_index) == 1 or int(step_index) == total_steps))
        backward_chunk_size_raw = int(train_cfg.get('backward_chunk_size', 0))
        if backward_chunk_size_raw <= 0:
            backward_chunk_size = max(1, matched_frame_total)
            backward_mode = 'full_step'
        else:
            backward_chunk_size = max(1, backward_chunk_size_raw)
            backward_mode = 'streaming_frame_chunk'
        optimizer_name = str(train_cfg.get('optimizer', 'adam')).strip().lower()
        base_learning_rate = float(train_cfg.get('lr', 0.03))
        learning_rate = float(base_learning_rate)
        lr_decay_every = max(0, int(train_cfg.get('lr_decay_every', 0)))
        lr_decay_amount = float(train_cfg.get('lr_decay_amount', 0.0))
        lr_min = float(train_cfg.get('lr_min', 0.0))

        def _learning_rate_for_step(step_index: int) -> float:
            if lr_decay_every <= 0 or lr_decay_amount <= 0.0:
                return float(base_learning_rate)
            decay_count = max(0, (int(step_index) - 1) // lr_decay_every)
            return max(float(lr_min), float(base_learning_rate) - float(decay_count) * float(lr_decay_amount))
        weight_decay = float(train_cfg.get('weight_decay', 0.0))
        pgd_epsilon = float(train_cfg.get('pgd_epsilon', 1.0))
        grad_clip_norm = float(train_cfg.get('grad_clip_norm', 0.0))
        if optimizer_name == 'adam':
            optimizer: Optional[torch.optim.Optimizer] = torch.optim.Adam(renderer.optimizer_parameters(), lr=learning_rate, weight_decay=weight_decay)
            pgd_anchor = None
        elif optimizer_name == 'pgd':
            optimizer = None
            pgd_anchor = renderer.texture_param.detach().clone()
        else:
            raise ValueError('train.optimizer only supports adam / pgd')
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp and optimizer_name == 'adam')
        loss_cfg = _merge_optimal_into_loss_cfg(_train_config_to_loss_cfg(resolved_loss_cfg), config=config)
        _merge_stp3_new_loss_config(loss_cfg, config=config)
        style_cfg = resolved_loss_cfg.get('style', {}) if isinstance(resolved_loss_cfg.get('style', {}), dict) else {}
        history: List[Dict[str, Any]] = []
        best_snapshot = renderer.texture_param.detach().clone()
        best_loss = float('inf')
        best_step = 0
        start_step = 1
        last_ckpt_path: Optional[Path] = None
        if resume_payload is not None:
            with torch.no_grad():
                renderer.texture_param.copy_(resume_payload['texture_param'].to(device=device, dtype=renderer.texture_param.dtype))
            loaded_best_snapshot = resume_payload.get('best_snapshot', resume_payload['texture_param'])
            best_snapshot = loaded_best_snapshot.to(device=device, dtype=renderer.texture_param.dtype).detach().clone()
            best_loss = float(resume_payload.get('best_loss', float('inf')))
            best_step = int(resume_payload.get('best_step', resume_payload.get('step', 0)))
            history = list(resume_payload.get('history', []))
            start_step = int(resume_payload.get('step', 0)) + 1
            if optimizer is not None and resume_payload.get('optimizer_name') == optimizer_name and ('optimizer_state' in resume_payload):
                optimizer.load_state_dict(resume_payload['optimizer_state'])
                _optimizer_to_device(optimizer, device)
            if scaler is not None and 'scaler_state' in resume_payload:
                scaler.load_state_dict(resume_payload['scaler_state'])
            if optimizer_name == 'pgd':
                loaded_anchor = resume_payload.get('pgd_anchor')
                if isinstance(loaded_anchor, torch.Tensor):
                    pgd_anchor = loaded_anchor.to(device=device, dtype=renderer.texture_param.dtype)
                else:
                    pgd_anchor = renderer.texture_param.detach().clone()
            if start_step > total_steps:
                raise ValueError(f'Checkpoint step={start_step - 1} already reaches/exceeds train.steps={total_steps}. Increase train.steps in config before resuming.')
        learning_rate = _learning_rate_for_step(start_step)
        if optimizer is not None:
            for group in optimizer.param_groups:
                group['lr'] = learning_rate
        _log_training_settings(training_log_path=training_log_path, config=config, resolved_loss_cfg=resolved_loss_cfg, loss_cfg_source=loss_cfg_source, config_path=config_path, output_dir=output_dir, sequence_yaml=sequence_yaml_paths, sequence_pkl=sequence_pkl, seed=seed, total_steps=total_steps, start_step=start_step, log_every=log_every, checkpoint_every=checkpoint_every, optimizer_name=optimizer_name, learning_rate=learning_rate, weight_decay=weight_decay, pgd_epsilon=pgd_epsilon, grad_clip_norm=grad_clip_norm, use_amp=use_amp, amp_dtype_name=amp_dtype_name, best_min_target_confidence=best_min_target_confidence, resume_ckpt_path=resume_ckpt_path)
        if lr_decay_every > 0 and lr_decay_amount > 0.0:
            _append_log_line(training_log_path, f'[train] lr_schedule=step_decay every={lr_decay_every} amount={lr_decay_amount} min={lr_min} start_lr={learning_rate}')
        else:
            _append_log_line(training_log_path, '[train] lr_schedule=disabled')
        _log_fixed_query_matches(frames=frames, fixed_queries=fixed_queries, training_log_path=training_log_path)
        if val_enabled:
            _log_fixed_query_matches(frames=val_frames, fixed_queries=val_fixed_queries, training_log_path=training_log_path, prefix='[val]')
        for idx, path in enumerate(val_sequence_yaml_paths, start=1):
            _append_log_line(training_log_path, f'[val] sequence_yaml[{idx}]={path}')
        if val_sequence_pkl is not None:
            _append_log_line(training_log_path, f'[val] sequence_pkl={val_sequence_pkl}')
        interrupted = False
        stop_signal: Optional[int] = None
        official_results_path: Optional[Path] = None
        official_visual_dir: Optional[Path] = None
        fixed_query_results_path: Optional[Path] = None
        fixed_query_visual_dir: Optional[Path] = None
        official_results_by_sequence: Dict[str, str] = {}
        official_visuals_by_sequence: Dict[str, str] = {}
        fixed_query_results_by_sequence: Dict[str, str] = {}
        fixed_query_visuals_by_sequence: Dict[str, str] = {}
        fixed_query_trace_check_by_sequence: Dict[str, Dict[str, Any]] = {}
        official_error: str = ''
        fixed_query_error: str = ''
        best_frame_rows: List[Dict[str, Any]] = []
        after_dir = output_dir / 'visuals_after'
        val_history: List[Dict[str, Any]] = []
        train_eval_history: List[Dict[str, Any]] = []
        step_ckpt_paths: List[str] = []
        latest_step_ckpt_path: Optional[Path] = None
        val_use_eot = bool(train_cfg.get('val_use_eot', False))
        loss_plot_path = output_dir / 'loss_curves.png'
        if resume_ckpt_path is not None:
            train_eval_history = list(resume_payload.get('train_eval_history', [])) if resume_payload is not None else []
            val_history = list(resume_payload.get('val_history', [])) if resume_payload is not None else []
            resume_message = f'[train] resume from checkpoint: {resume_ckpt_path} start_step={start_step}'
            print(resume_message)
            _append_log_line(training_log_path, resume_message)
        _append_log_line(training_log_path, f'[train] train hyperparams total_steps={total_steps} log_every={log_every} use_amp={use_amp} amp_dtype={amp_dtype_name} optimizer={optimizer_name} lr={learning_rate} weight_decay={weight_decay} pgd_epsilon={pgd_epsilon} checkpoint_every={checkpoint_every} checkpoint_policy=step_periodic batch_mode={batch_mode} train_samples_per_step={train_samples_per_step} train_eval_every={train_eval_every} val_eval_every={(val_eval_every if val_enabled else 0)} val_log_every={val_log_every} backward_mode={backward_mode} backward_chunk_size_raw={backward_chunk_size_raw} backward_chunk_size_effective={backward_chunk_size}')
        _append_log_line(training_log_path, f'[train] loss_plot_path={loss_plot_path}')
        _append_log_line(training_log_path, f"[train] loss_plot_refresh_interval={(plot_every if plot_every > 0 else 'disabled')}")
        _append_log_line(training_log_path, f'[train] eot_cfg={json.dumps(renderer.eot_augmentor.summary(), ensure_ascii=False)}')
        _append_log_line(training_log_path, f'[train] image_aug_cfg={json.dumps(renderer.image_augmentor.summary(), ensure_ascii=False)}')
        if val_enabled:
            _append_log_line(training_log_path, f"[val] enabled: full no-grad eval every {val_eval_every} train steps (does not affect best/ckpt); val_log_every={val_log_every} val_use_eot={str(val_use_eot).lower()} val_image_source_subdir={config.get('dataset', {}).get('val_image_source_subdir', '')}")
        else:
            _append_log_line(training_log_path, '[val] disabled')
    except Exception as exc:
        _append_exception_log(training_log_path, exc, prefix='[train] init phase failed')
        raise
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _request_stop(signum, _frame) -> None:
        nonlocal stop_signal
        stop_signal = int(signum)
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    output_dir_missing_exit = False
    try:
        for step in range(start_step, total_steps + 1):
            if not output_dir.exists() or not output_dir_sentinel_path.exists():
                output_dir_missing_exit = True
                interrupted = True
                print(f'[train] output_dir missing; exiting: {output_dir}')
                break
            learning_rate = _learning_rate_for_step(step)
            if optimizer is not None:
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            elif renderer.texture_param.grad is not None:
                renderer.texture_param.grad = None
            frame_debug_rows: List[Dict[str, Any]] = []
            grad_by_frame: Dict[str, Any] = {}
            active_state: Optional[ActiveFrameState] = None
            chunk_states: List[ActiveFrameState] = []
            matched_frames_step = 0
            target_lost_frames_step = 0
            progress_term_values: List[float] = []
            progress_teacher_values: List[float] = []
            progress_floor_values: List[float] = []
            progress_gain_values: List[float] = []
            move_lateral_term_values: List[float] = []
            move_lateral_weight_values: List[float] = []
            move_lateral_weighted_values: List[float] = []
            move_longitudinal_term_values: List[float] = []
            move_longitudinal_mean_values: List[float] = []
            move_longitudinal_max_values: List[float] = []
            move_longitudinal_hard_values: List[float] = []
            move_longitudinal_hard_excess_values: List[float] = []
            move_longitudinal_hard_excess_max_values: List[float] = []
            lateral_difference_values: List[float] = []
            longitudinal_difference_values: List[float] = []
            longitudinal_difference_max_values: List[float] = []
            first_frame_min_term_values: List[float] = []
            per_frame_min_term_values: List[float] = []
            rigid_term_values: List[float] = []
            rigid_size_loss_values: List[float] = []
            rigid_yaw_loss_values: List[float] = []
            rigid_size_diff_values: List[float] = []
            rigid_yaw_diff_values: List[float] = []
            cls_term_values: List[float] = []
            cls_pos_values: List[float] = []
            cls_neg_values: List[float] = []
            cls_rank_values: List[float] = []
            cls_nearby_values: List[float] = []
            cls_global_rank_values: List[float] = []
            cls_output_proxy_values: List[float] = []
            cls_query_identity_values: List[float] = []
            target_logit_values: List[float] = []
            target_confidence_values: List[float] = []
            noncar_max_conf_values: List[float] = []
            nearby_target_max_conf_values: List[float] = []
            nearby_query_count_values: List[float] = []
            global_other_target_max_conf_values: List[float] = []
            other_query_global_max_conf_values: List[float] = []
            query_identity_cosine_values: List[float] = []
            depth_term_values: List[float] = []
            stp3_new_loss_weighted_values: List[float] = []
            depth_cam_count_values: List[float] = []
            depth_sample_count_values: List[float] = []
            depth_pred_mean_values: List[float] = []
            depth_gt_mean_values: List[float] = []
            bevdet_loss_values: List[float] = []
            step_frame_groups = _select_train_frame_groups_for_step(frame_groups=frame_groups, fixed_queries=fixed_queries, batch_mode=batch_mode, samples_per_step=train_samples_per_step)
            step_matched_frame_total = _matched_frame_count_for_groups(step_frame_groups, fixed_queries)
            step_progress_pair_total = _progress_pair_count_for_groups(step_frame_groups, fixed_queries)
            step_backward_chunk_size = max(1, step_matched_frame_total) if backward_chunk_size_raw <= 0 else max(1, backward_chunk_size_raw)
            step_sequence_names = _frame_group_names(step_frame_groups)

            def _scaled_loss_term(term: torch.Tensor, weight: float, denom: int) -> torch.Tensor:
                safe_denom = max(1, int(denom))
                return term * term.new_tensor(float(weight) / float(safe_denom))

            def _backward_chunk(loss_term: torch.Tensor, *, retain_graph: bool) -> None:
                if optimizer is not None:
                    scaler.scale(loss_term).backward(retain_graph=retain_graph)
                else:
                    loss_term.backward(retain_graph=retain_graph)

            def _finalize_active_state(state: ActiveFrameState) -> None:
                grad_scale = float(scaler.get_scale()) if optimizer is not None and scaler.is_enabled() else 1.0
                grad_stats = _hook_stats_for_frame(state.query_match, state.bbox_tensor, state.cls_tensor, target_label=state.frame_input.target_label, grad_scale=grad_scale, heatmap_tensor=state.heatmap_tensor)
                if grad_stats:
                    grad_by_frame[f'{state.sample_token}'] = grad_stats

            def _flush_chunk(*, retain_graph: bool) -> None:
                nonlocal active_state, chunk_states
                if not chunk_states:
                    return
                single_loss: Optional[torch.Tensor] = None
                for state in chunk_states:
                    term = _scaled_loss_term(state.frame_loss_term, 1.0, step_matched_frame_total)
                    single_loss = term if single_loss is None else single_loss + term
                progress_loss_sum: Optional[torch.Tensor] = None
                progress_states = ([active_state] if active_state is not None else []) + list(chunk_states)
                for prev_state, curr_state in zip(progress_states[:-1], progress_states[1:]):
                    pair_deviations = _movement_difference_sequence([prev_state.frame_input, curr_state.frame_input], loss_cfg)
                    progress_term, progress_stats, progress_gains = progress_loss(pair_deviations, step_size_m=float(loss_cfg.get('progress_step_size_m', 0.05)), decay_lambda=float(loss_cfg.get('progress_lambda', 1.0)), detach_previous=bool(loss_cfg.get('progress_detach_previous', False)), loss_type=str(loss_cfg.get('progress_step_loss_type', 'l2')))
                    progress_term_values.append(float(progress_term.detach().item()))
                    progress_teacher_values.append(float(progress_stats.get('progress_teacher', 0.0)))
                    progress_floor_values.append(float(progress_stats.get('progress_floor', 0.0)))
                    progress_gain_values.extend([float(v) for v in progress_gains if v is not None])
                    weighted = _scaled_loss_term(progress_term, float(loss_cfg.get('progress_weight', 1.0)), step_progress_pair_total)
                    progress_loss_sum = weighted if progress_loss_sum is None else progress_loss_sum + weighted
                chunk_loss: Optional[torch.Tensor] = single_loss
                if progress_loss_sum is not None:
                    chunk_loss = progress_loss_sum if chunk_loss is None else chunk_loss + progress_loss_sum
                if chunk_loss is not None:
                    _backward_chunk(chunk_loss, retain_graph=retain_graph)
                if active_state is not None:
                    _finalize_active_state(active_state)
                if retain_graph:
                    for state in chunk_states[:-1]:
                        _finalize_active_state(state)
                    active_state = chunk_states[-1]
                else:
                    for state in chunk_states:
                        _finalize_active_state(state)
                    active_state = None
                chunk_states = []
            for sequence_frames in step_frame_groups:
                prev_bev: Optional[torch.Tensor] = None
                prev_scene_token: Optional[str] = None
                prev_abs_pos: Optional[np.ndarray] = None
                prev_abs_angle: Optional[float] = None
                active_state = None
                chunk_states = []
                matched_frames_in_sequence = 0
                final_frame_cache_key = str(sequence_frames[-1].cache_key) if sequence_frames else ''
                sequence_matched_frame_total = len(sequence_frames) if uses_final_decode_match else sum((1 for item in sequence_frames if item.cache_key in fixed_queries and bool(fixed_queries[item.cache_key].matched)))
                for frame in sequence_frames:
                    query_match = fixed_queries.get(frame.cache_key)
                    retain_grad_for_frame = bool(query_match.matched) if query_match is not None else False
                    if uses_final_decode_match:
                        retain_grad_for_frame = True
                    with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                        camera_images = renderer.build_frame_images(frame, apply_eot=True)
                        outs, prev_bev, prev_abs_pos, prev_abs_angle = bev_model.forward_frame(frame, camera_images, prev_bev=prev_bev, prev_scene_token=prev_scene_token, prev_abs_pos=prev_abs_pos, prev_abs_angle=prev_abs_angle, retain_grad=retain_grad_for_frame)
                    if isinstance(prev_bev, torch.Tensor):
                        prev_bev = prev_bev.detach()
                    prev_scene_token = frame.scene_token
                    if is_bevdet_model:
                        query_match = bev_model.match_target_query_from_final_outputs(frame, outs, conf_threshold=fixed_conf_threshold, max_center_dist_m=fixed_max_center_dist_m, distance_axis=fixed_distance_axis, max_cross_axis_dist_m=fixed_max_cross_axis_dist_m)
                    if query_match is None or not query_match.matched:
                        if uses_final_decode_match and query_match is not None:
                            target_lost_frames_step += 1
                            frame_debug_rows.append({'matched': False, 'sequence_name': str(frame.sequence_name), 'frame_id': int(frame.frame_id), 'sample_token': str(frame.sample_token), 'target_detection_name': str(query_match.target_detection_name), 'unmatched_reason': str(query_match.unmatched_reason), 'candidate_total': int(query_match.candidate_total), 'candidate_after_conf': int(query_match.candidate_after_conf), 'candidate_after_dist': int(query_match.candidate_after_dist)})
                        del camera_images
                        del outs
                        continue
                    matched_frames_step += 1
                    matched_frames_in_sequence += 1
                    prediction = bev_model.target_query_prediction(frame, outs, query_idx=query_match.query_idx)
                    if bev_model.last_cls_tensor is None:
                        raise RuntimeError('model forward did not expose class hook tensor')
                    other_target_logits = _build_query_competition_terms(cls_tensor=bev_model.last_cls_tensor, query_idx=int(query_match.query_idx), target_label=int(prediction.target_label))
                    query_feature = _query_feature_for_index(query_feature_tensor=getattr(bev_model, 'last_query_feature_tensor', None), query_idx=int(query_match.query_idx))
                    clean_query_feature = None
                    if clean_query_features is not None:
                        cached_feature = clean_query_features.get(frame.cache_key)
                        if cached_feature is not None:
                            clean_query_feature = cached_feature.to(device=prediction.target_logit.device, dtype=torch.float32)
                    ref_center_ego, ref_size_wlh, ref_yaw = _loss_reference_tensors_for_frame(frame=frame, prediction=prediction, loss_reference_mode=loss_reference_mode, clean_detection_refs=train_clean_detection_refs)
                    frame_input = FrameLossInput(frame_id=frame.frame_id, pred_center_ego=prediction.pred_center_ego, gt_center_ego=prediction.gt_center_ego, pred_size_wlh=prediction.pred_box_lidar[3:6], gt_size_wlh=prediction.gt_box_lidar[3:6], pred_yaw=prediction.pred_box_lidar[6:7], gt_yaw=prediction.gt_box_lidar[6:7], pred_class_logits=prediction.class_logits, target_logit=prediction.target_logit, target_label=prediction.target_label, nearby_target_logits=None, other_target_logits=other_target_logits, other_query_max_logits=None, query_feature=query_feature, clean_query_feature=clean_query_feature, ref_center_ego=ref_center_ego, ref_size_wlh=ref_size_wlh, ref_yaw=ref_yaw)
                    if bev_model.last_bbox_tensor is None or bev_model.last_cls_tensor is None:
                        raise RuntimeError('model forward did not expose hook tensors')
                    display_ref_center = ref_center_ego if ref_center_ego is not None else frame_input.gt_center_ego
                    delta_y = float((prediction.pred_center_ego[1] - display_ref_center[1]).detach().item())
                    delta_x = float((prediction.pred_center_ego[0] - display_ref_center[0]).detach().item())
                    move_lateral_term, move_longitudinal_term, move_stats = _movement_loss_terms([frame_input], loss_cfg)
                    rigid_term, rigid_stats = rigid_loss([frame_input], loss_type=str(loss_cfg.get('rigid_loss_type', 'l2')), size_weight=float(loss_cfg.get('rigid_size_weight', 1.0)), yaw_weight=float(loss_cfg.get('rigid_yaw_weight', 1.0)))
                    cls_term, cls_stats = cls_loss([frame_input], pos_weight=float(loss_cfg.get('cls_pos_weight', 1.0)), neg_weight=float(loss_cfg.get('cls_neg_weight', 1.0)), rank_weight=float(loss_cfg.get('cls_rank_weight', 1.0)), rank_margin=float(loss_cfg.get('cls_rank_margin', 0.0)))
                    global_rank_term, global_rank_stats = global_query_rank_loss([frame_input], margin=float(loss_cfg.get('cls_global_rank_margin', 0.0)))
                    query_identity_term, query_identity_stats = query_identity_loss([frame_input], loss_type=str(loss_cfg.get('query_identity_loss_type', 'cosine')))
                    lateral_difference = _movement_difference_sequence([frame_input], loss_cfg)[0]
                    if matched_frames_in_sequence == 1 and _progress_child_loss_weight(loss_cfg, 'first_frame_min_weight') > 0.0:
                        first_frame_term = first_frame_min_loss(lateral_difference, min_m=float(loss_cfg.get('first_frame_min_m', 0.0)))
                    else:
                        first_frame_term = move_lateral_term.new_zeros(())
                    if _progress_child_loss_weight(loss_cfg, 'per_frame_min_weight') > 0.0:
                        per_frame_min_term = first_frame_min_loss(lateral_difference, min_m=float(loss_cfg.get('per_frame_min_m', 0.0)))
                    else:
                        per_frame_min_term = move_lateral_term.new_zeros(())
                    move_lateral_weight = _move_lateral_loss_weight(loss_cfg, is_final_frame=str(frame.cache_key) == final_frame_cache_key)
                    depth_term = move_lateral_term.new_zeros(())
                    depth_stats: Dict[str, float] = {'loss_depth': 0.0, 'depth_cam_count': 0.0, 'depth_sample_count': 0.0, 'depth_pred_mean_m': 0.0, 'depth_gt_mean_m': 0.0}
                    if float(loss_cfg.get('depth_weight', 0.0)) > 0.0 and hasattr(bev_model, 'depth_supervision_loss'):
                        depth_term, depth_stats = bev_model.depth_supervision_loss(frame=frame, direction=str(loss_cfg.get('depth_direction', 'far')), offset_m=float(loss_cfg.get('depth_offset_m', 2.0)), patch_radius=int(loss_cfg.get('depth_patch_radius', 1)), loss_type=str(loss_cfg.get('depth_loss_type', 'l1')), min_valid_cams=int(loss_cfg.get('depth_min_valid_cams', 1)))
                    stp3_new_loss_weighted_frag = move_lateral_term.new_zeros(())
                    stp3_nl_debug: Dict[str, float] = {'stp3_new_loss_weighted': 0.0, 'stp3_new_loss_repel': 0.0, 'stp3_new_loss_attract': 0.0, 'stp3_new_loss_total_raw': 0.0, 'stp3_new_loss_skip': 0.0, 'stp3_new_loss_shift_cells': 0.0}
                    if active_model_name == 'stp3' and float(loss_cfg.get('stp3_new_loss_weight', 0.0)) > 0.0 and isinstance(outs, dict) and ('segmentation' in outs):
                        nw = float(loss_cfg.get('stp3_new_loss_weight', 0.0))
                        n_pres = int(getattr(bev_model.model, 'receptive_field'))
                        vlog = outs['segmentation'][0, n_pres - 1, 1].float()
                        y_step_m = float(bev_model.cfg.LIFT.Y_BOUND[2])
                        stp3_nl_raw, stp3_nl_dbg = compute_stp3_new_loss(vlog, mask_path=str(getattr(frame, 'stp3_bev_target_mask_path', '') or ''), y_step_m=y_step_m, shift_lateral_m=float(loss_cfg.get('stp3_new_loss_shift_lateral_m', 1.0)), repulsion_weight=float(loss_cfg.get('stp3_new_loss_repulsion_weight', 1.0)), attraction_weight=float(loss_cfg.get('stp3_new_loss_attraction_weight', 1.0)), overlap_act_threshold=float(loss_cfg.get('stp3_new_loss_overlap_act_threshold', 0.5)), use_overlap_refinement=bool(loss_cfg.get('stp3_new_loss_use_overlap_refinement', True)), config_path=config_path)
                        stp3_new_loss_weighted_frag = stp3_nl_raw * stp3_nl_raw.new_tensor(nw)
                        stp3_nl_debug = {'stp3_new_loss_weighted': float(stp3_new_loss_weighted_frag.detach().item()), 'stp3_new_loss_repel': float(stp3_nl_dbg.get('stp3_new_loss_repel', 0.0)), 'stp3_new_loss_attract': float(stp3_nl_dbg.get('stp3_new_loss_attract', 0.0)), 'stp3_new_loss_total_raw': float(stp3_nl_dbg.get('stp3_new_loss_total', 0.0)), 'stp3_new_loss_skip': float(stp3_nl_dbg.get('stp3_new_loss_skip', 0.0)), 'stp3_new_loss_shift_cells': float(stp3_nl_dbg.get('stp3_new_loss_shift_cells', 0.0))}
                        stp3_new_loss_weighted_values.append(float(stp3_new_loss_weighted_frag.detach().item()))
                    else:
                        stp3_new_loss_weighted_values.append(0.0)
                    frame_loss_term = move_lateral_term.new_tensor(move_lateral_weight) * move_lateral_term + rigid_term.new_tensor(float(loss_cfg.get('rigid_weight', 1.0))) * rigid_term + cls_term.new_tensor(float(loss_cfg.get('cls_weight', 1.0))) * cls_term + global_rank_term.new_tensor(_cls_child_loss_weight(loss_cfg, 'cls_global_rank_weight')) * global_rank_term + query_identity_term.new_tensor(_cls_child_loss_weight(loss_cfg, 'query_identity_weight')) * query_identity_term + first_frame_term.new_tensor(_progress_child_loss_weight(loss_cfg, 'first_frame_min_weight')) * first_frame_term + per_frame_min_term.new_tensor(_progress_child_loss_weight(loss_cfg, 'per_frame_min_weight')) * per_frame_min_term + depth_term.new_tensor(float(loss_cfg.get('depth_weight', 0.0))) * depth_term + stp3_new_loss_weighted_frag
                    frame_debug_rows.append({'sequence_name': str(frame.sequence_name), 'frame_id': int(frame.frame_id), 'sample_token': str(frame.sample_token), 'query_idx': int(getattr(prediction, 'query_idx', query_match.query_idx)), 'match_note': str(getattr(query_match, 'unmatched_reason', '')), 'reference_mode': str(loss_reference_mode), 'pred_y_m': float(prediction.pred_center_ego[1].detach().item()), 'gt_y_m': float(display_ref_center[1].detach().item()), 'ref_y_m': float(display_ref_center[1].detach().item()), 'true_gt_y_m': float(frame_input.gt_center_ego[1].detach().item()), 'delta_y_m': delta_y, 'direction': 'left' if delta_y >= 0.0 else 'right', 'shift_abs_m': abs(delta_y), 'pred_x_m': float(prediction.pred_center_ego[0].detach().item()), 'gt_x_m': float(display_ref_center[0].detach().item()), 'ref_x_m': float(display_ref_center[0].detach().item()), 'true_gt_x_m': float(frame_input.gt_center_ego[0].detach().item()), 'delta_x_m': delta_x, 'x_direction': 'front' if delta_x >= 0.0 else 'back', 'x_shift_abs_m': abs(delta_x), 'target_detection_name': str(prediction.target_detection_name), 'target_confidence': float(torch.sigmoid(prediction.target_logit.detach()).item()), 'move_lateral_loss': float(move_lateral_term.detach().item()), 'move_lateral_weight': float(move_lateral_weight), 'move_longitudinal_loss': float(move_longitudinal_term.detach().item()), 'move_longitudinal_hard_loss': 0.0, 'xhard_excess_m': 0.0, 'first_frame_min_loss': float(first_frame_term.detach().item()), 'per_frame_min_loss': float(per_frame_min_term.detach().item()), 'difference': float(lateral_difference.detach().item()), 'target_lateral_move_m': float(lateral_difference.detach().item()), 'longitudinal_difference': abs(delta_x), 'rigid_loss': float(rigid_term.detach().item()), 'cls_loss': float(cls_term.detach().item()), 'cls_nearby_loss': 0.0, 'cls_global_rank_loss': float(global_rank_stats.get('loss_global_query_rank', 0.0)), 'cls_output_proxy_loss': 0.0, 'cls_query_identity_loss': float(query_identity_stats.get('loss_query_identity', 0.0)), 'nearby_query_count': 0, 'depth_loss': float(depth_stats.get('loss_depth', 0.0)), 'depth_cam_count': int(round(float(depth_stats.get('depth_cam_count', 0.0)))), 'depth_sample_count': int(round(float(depth_stats.get('depth_sample_count', 0.0)))), 'depth_pred_mean_m': float(depth_stats.get('depth_pred_mean_m', 0.0)), 'depth_gt_mean_m': float(depth_stats.get('depth_gt_mean_m', 0.0)), 'stp3_new_loss_weighted': float(stp3_nl_debug.get('stp3_new_loss_weighted', 0.0)), 'stp3_new_loss_repel': float(stp3_nl_debug.get('stp3_new_loss_repel', 0.0)), 'stp3_new_loss_attract': float(stp3_nl_debug.get('stp3_new_loss_attract', 0.0)), 'stp3_new_loss_total_raw': float(stp3_nl_debug.get('stp3_new_loss_total_raw', 0.0)), 'stp3_new_loss_skip': float(stp3_nl_debug.get('stp3_new_loss_skip', 0.0)), 'stp3_new_loss_shift_cells': float(stp3_nl_debug.get('stp3_new_loss_shift_cells', 0.0))})
                    state_query_match = query_match
                    if not bool(loss_cfg.get('enable_query_losses', True)):
                        state_query_match = _clone_query_match_with_new_idx(query_match, query_idx=int(prediction.query_idx))
                    current_state = ActiveFrameState(frame_input=frame_input, frame_id=frame.frame_id, sample_token=frame.sample_token, query_match=state_query_match, bbox_tensor=bev_model.last_bbox_tensor, cls_tensor=bev_model.last_cls_tensor, heatmap_tensor=getattr(bev_model, 'last_heatmap_tensor', None), frame_loss_term=frame_loss_term, move_lateral_term=move_lateral_term, move_longitudinal_term=move_longitudinal_term, first_frame_min_term=first_frame_term, rigid_term=rigid_term, cls_term=cls_term)
                    move_lateral_term_values.append(float(move_lateral_term.detach().item()))
                    move_lateral_weight_values.append(float(move_lateral_weight))
                    move_lateral_weighted_values.append(float((move_lateral_term.detach() * move_lateral_term.new_tensor(move_lateral_weight)).item()))
                    move_longitudinal_term_values.append(float(move_longitudinal_term.detach().item()))
                    move_longitudinal_mean_values.append(float(move_stats.get('loss_move_longitudinal_mean', 0.0)))
                    move_longitudinal_max_values.append(float(move_stats.get('loss_move_longitudinal_max', 0.0)))
                    move_longitudinal_hard_values.append(0.0)
                    move_longitudinal_hard_excess_values.append(0.0)
                    move_longitudinal_hard_excess_max_values.append(0.0)
                    lateral_difference_values.append(float(move_stats.get('difference_mean', 0.0)))
                    longitudinal_difference_values.append(float(move_stats.get('longitudinal_difference_mean', 0.0)))
                    longitudinal_difference_max_values.append(float(move_stats.get('longitudinal_difference_max', 0.0)))
                    first_frame_min_term_values.append(float(first_frame_term.detach().item()))
                    per_frame_min_term_values.append(float(per_frame_min_term.detach().item()))
                    rigid_term_values.append(float(rigid_term.detach().item()))
                    rigid_size_loss_values.append(float(rigid_stats.get('loss_rigid_size', 0.0)))
                    rigid_yaw_loss_values.append(float(rigid_stats.get('loss_rigid_yaw', 0.0)))
                    rigid_size_diff_values.append(float(rigid_stats.get('rigid_size_diff_mean', 0.0)))
                    rigid_yaw_diff_values.append(float(rigid_stats.get('rigid_yaw_diff_mean', 0.0)))
                    cls_term_values.append(float(cls_term.detach().item()))
                    cls_pos_values.append(float(cls_stats.get('loss_cls_pos', 0.0)))
                    cls_neg_values.append(float(cls_stats.get('loss_cls_neg', 0.0)))
                    cls_rank_values.append(float(cls_stats.get('loss_cls_rank', 0.0)))
                    cls_nearby_values.append(float(cls_stats.get('loss_cls_nearby', 0.0)))
                    cls_global_rank_values.append(float(global_rank_stats.get('loss_global_query_rank', 0.0)))
                    cls_output_proxy_values.append(0.0)
                    cls_query_identity_values.append(float(query_identity_stats.get('loss_query_identity', 0.0)))
                    target_logit_values.append(float(cls_stats.get('target_logit', 0.0)))
                    target_confidence_values.append(float(cls_stats.get('target_confidence', 0.0)))
                    noncar_max_conf_values.append(float(cls_stats.get('noncar_max_confidence', 0.0)))
                    nearby_target_max_conf_values.append(float(cls_stats.get('nearby_target_max_confidence', 0.0)))
                    nearby_query_count_values.append(float(cls_stats.get('nearby_query_count', 0.0)))
                    global_other_target_max_conf_values.append(float(global_rank_stats.get('global_other_target_max_confidence', 0.0)))
                    other_query_global_max_conf_values.append(0.0)
                    query_identity_cosine_values.append(float(query_identity_stats.get('query_identity_cosine', 1.0)))
                    depth_term_values.append(float(depth_stats.get('loss_depth', 0.0)))
                    depth_cam_count_values.append(float(depth_stats.get('depth_cam_count', 0.0)))
                    depth_sample_count_values.append(float(depth_stats.get('depth_sample_count', 0.0)))
                    depth_pred_mean_values.append(float(depth_stats.get('depth_pred_mean_m', 0.0)))
                    depth_gt_mean_values.append(float(depth_stats.get('depth_gt_mean_m', 0.0)))
                    chunk_states.append(current_state)
                    if len(chunk_states) >= step_backward_chunk_size:
                        _flush_chunk(retain_graph=bool(matched_frames_in_sequence < sequence_matched_frame_total))
                    del camera_images
                    del outs
                    del prediction
                if chunk_states:
                    _flush_chunk(retain_graph=False)
                if active_state is not None:
                    _finalize_active_state(active_state)
                    active_state = None
                chunk_states = []
            if matched_frames_step <= 0:
                fallback_best_step = int(best_step)
                fallback_best_loss = float(best_loss) if math.isfinite(float(best_loss)) else 0.0
                with torch.no_grad():
                    renderer.texture_param.copy_(best_snapshot)
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                elif renderer.texture_param.grad is not None:
                    renderer.texture_param.grad = None
                _clear_model_runtime_tensors(bev_model)
                fallback_record = {'step': int(step), 'skipped': True, 'skip_reason': 'no_target_query_loss', 'fallback_to_checkpoint_snapshot': True, 'fallback_best_step': fallback_best_step, 'fallback_best_loss': fallback_best_loss, 'loss_total': 0.0, 'loss_model': 0.0, 'batch_mode': str(batch_mode), 'batch_sequence_names': step_sequence_names, 'batch_matched_frame_total': int(step_matched_frame_total), 'matched_frames': int(matched_frames_step), 'lr': float(learning_rate), 'target_lost_frames': int(target_lost_frames_step), 'target_expected_frames': int(step_matched_frame_total), 'all_targets_present': False, 'best_conf_gate_passed': False, 'target_confidence': 0.0, 'target_confidence_min': 0.0, 'frame_metrics': frame_debug_rows}
                history.append(fallback_record)
                fallback_line = f'[train] step-{step:04d}: no usable target-query loss; skipped update; reverted to checkpoint step={fallback_best_step} loss={fallback_best_loss:.6f}'
                print(fallback_line)
                _append_log_line(training_log_path, fallback_line)
                if _should_plot_training_curves(step):
                    _plot_training_loss_curves(history=history, train_eval_history=train_eval_history, val_history=val_history, output_path=loss_plot_path, training_log_path=training_log_path)
                if step % log_every == 0 or step == 1 or step == total_steps:
                    for row in frame_debug_rows:
                        if not bool(row.get('matched', True)):
                            frame_line = f"[train]   seq={row['sequence_name']} frame={row['frame_id']} cls={row['target_detection_name']} target_lost=True reason={row.get('unmatched_reason', '')} candidates={int(row.get('candidate_total', 0))} after_conf={int(row.get('candidate_after_conf', 0))} after_dist={int(row.get('candidate_after_dist', 0))}"
                            print(frame_line)
                            _append_log_line(training_log_path, frame_line)
                continue
            _flush_chunk(retain_graph=False)
            auxiliary_step_records: List[Dict[str, Any]] = []
            for aux_runtime in auxiliary_runtimes:
                aux_record = _backward_auxiliary_model_step(runtime=aux_runtime, frame_groups=step_frame_groups, renderer=renderer, device=device, use_amp=use_amp, amp_dtype=amp_dtype, optimizer=optimizer, scaler=scaler, fixed_conf_threshold=fixed_conf_threshold, fixed_max_center_dist_m=fixed_max_center_dist_m, fixed_distance_axis=fixed_distance_axis, fixed_max_cross_axis_dist_m=fixed_max_cross_axis_dist_m, training_config_path=config_path)
                auxiliary_step_records.append(aux_record)
            with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                texture_for_loss = torch.sigmoid(renderer.texture_param)
                style_term = torch.zeros((), device=device, dtype=texture_for_loss.dtype)
                style_tv_raw = torch.zeros((), device=device, dtype=texture_for_loss.dtype)
                style_l2_raw = torch.zeros((), device=device, dtype=texture_for_loss.dtype)
                style_brightness_raw = torch.zeros((), device=device, dtype=texture_for_loss.dtype)
                style_nps_raw = torch.zeros((), device=device, dtype=texture_for_loss.dtype)
                style_weight = float(style_cfg.get('weight', 1.0))
                printable_palette = _style_printable_palette_from_config(style_cfg, device=texture_for_loss.device, dtype=torch.float32)
                if abs(style_weight) > 1e-12 and (float(style_cfg.get('tv_weight', 0.0)) > 0.0 or float(style_cfg.get('l2_weight', 0.0)) > 0.0 or float(style_cfg.get('brightness_weight', 0.0)) > 0.0 or (float(style_cfg.get('nps_weight', 0.0)) > 0.0)):
                    from loss import brightness_loss, l2_anchor_loss, non_printability_score_loss, total_variation_loss
                    if float(style_cfg.get('tv_weight', 0.0)) > 0.0:
                        style_tv_raw = total_variation_loss(texture_for_loss)
                        style_term = style_term + float(style_cfg.get('tv_weight', 0.0)) * style_tv_raw
                    if float(style_cfg.get('l2_weight', 0.0)) > 0.0:
                        style_l2_raw = l2_anchor_loss(texture_for_loss)
                        style_term = style_term + float(style_cfg.get('l2_weight', 0.0)) * style_l2_raw
                    if float(style_cfg.get('brightness_weight', 0.0)) > 0.0:
                        style_brightness_raw = brightness_loss(texture_for_loss, target_brightness=float(style_cfg.get('brightness_target', 0.4)), loss_type=str(style_cfg.get('brightness_loss_type', 'l2')))
                        style_term = style_term + float(style_cfg.get('brightness_weight', 0.0)) * style_brightness_raw
                    if float(style_cfg.get('nps_weight', 0.0)) > 0.0:
                        style_nps_raw = non_printability_score_loss(texture_for_loss, printable_colors=printable_palette)
                        style_term = style_term + float(style_cfg.get('nps_weight', 0.0)) * style_nps_raw
                style_term = style_term * style_weight
            weighted_progress_value = float(loss_cfg.get('progress_weight', 1.0)) * _mean_or_zero(progress_term_values)
            weighted_move_lateral_value = _mean_or_zero(move_lateral_weighted_values)
            weighted_first_frame_min_value = _progress_child_loss_weight(loss_cfg, 'first_frame_min_weight') * _mean_or_zero(first_frame_min_term_values)
            weighted_per_frame_min_value = _progress_child_loss_weight(loss_cfg, 'per_frame_min_weight') * _mean_or_zero(per_frame_min_term_values)
            weighted_rigid_value = float(loss_cfg.get('rigid_weight', 1.0)) * _mean_or_zero(rigid_term_values)
            weighted_cls_value = float(loss_cfg.get('cls_weight', 1.0)) * _mean_or_zero(cls_term_values)
            weighted_cls_global_rank_value = _cls_child_loss_weight(loss_cfg, 'cls_global_rank_weight') * _mean_or_zero(cls_global_rank_values)
            weighted_query_identity_value = _cls_child_loss_weight(loss_cfg, 'query_identity_weight') * _mean_or_zero(cls_query_identity_values)
            weighted_depth_value = float(loss_cfg.get('depth_weight', 0.0)) * _mean_or_zero(depth_term_values)
            weighted_stp3_new_loss_value = _mean_or_zero(stp3_new_loss_weighted_values)
            model_loss_value = weighted_progress_value + weighted_move_lateral_value + weighted_first_frame_min_value + weighted_per_frame_min_value + weighted_rigid_value + weighted_cls_value + weighted_cls_global_rank_value + weighted_query_identity_value + weighted_depth_value + weighted_stp3_new_loss_value
            auxiliary_loss_value = sum((float(row.get('loss_weighted', 0.0)) for row in auxiliary_step_records))
            style_term_value_pre = float(style_term.detach().item())
            loss_total_value_pre = float(model_loss_value + auxiliary_loss_value + style_term_value_pre)
            texture_param_finite = bool(torch.isfinite(renderer.texture_param.detach()).all().item())
            style_term_finite = bool(torch.isfinite(style_term.detach()).all().item())
            if not math.isfinite(loss_total_value_pre) or not math.isfinite(float(model_loss_value)) or (not math.isfinite(float(auxiliary_loss_value))) or (not style_term_finite) or (not texture_param_finite):
                with torch.no_grad():
                    restore_snapshot = best_snapshot
                    if not bool(torch.isfinite(restore_snapshot).all().item()):
                        restore_snapshot = torch.nan_to_num(renderer.texture_param.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                        best_snapshot = restore_snapshot.detach().clone()
                    renderer.texture_param.copy_(restore_snapshot)
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                elif renderer.texture_param.grad is not None:
                    renderer.texture_param.grad = None
                _clear_model_runtime_tensors(bev_model)
                fallback_best_step = int(best_step)
                fallback_best_loss = float(best_loss) if math.isfinite(float(best_loss)) else 0.0
                fallback_record = {'step': int(step), 'skipped': True, 'skip_reason': 'nonfinite_loss', 'fallback_to_checkpoint_snapshot': True, 'fallback_best_step': fallback_best_step, 'fallback_best_loss': fallback_best_loss, 'loss_total': 0.0, 'loss_model': float(model_loss_value) if math.isfinite(float(model_loss_value)) else 0.0, 'loss_auxiliary_weighted': float(auxiliary_loss_value) if math.isfinite(float(auxiliary_loss_value)) else 0.0, 'loss_style_weighted': style_term_value_pre if math.isfinite(style_term_value_pre) else 0.0, 'batch_mode': str(batch_mode), 'batch_sequence_names': step_sequence_names, 'batch_matched_frame_total': int(step_matched_frame_total), 'matched_frames': int(matched_frames_step), 'lr': float(learning_rate), 'target_lost_frames': int(target_lost_frames_step), 'target_expected_frames': int(step_matched_frame_total), 'all_targets_present': False, 'best_conf_gate_passed': False, 'target_confidence': _mean_or_zero(target_confidence_values), 'target_confidence_min': min(target_confidence_values) if target_confidence_values else 0.0, 'frame_metrics': frame_debug_rows}
                history.append(fallback_record)
                fallback_line = f'[train] step-{step:04d}: non-finite loss/texture; skipped update; reverted to checkpoint step={fallback_best_step} loss={fallback_best_loss:.6f}'
                print(fallback_line)
                _append_log_line(training_log_path, fallback_line)
                if _should_plot_training_curves(step):
                    _plot_training_loss_curves(history=history, train_eval_history=train_eval_history, val_history=val_history, output_path=loss_plot_path, training_log_path=training_log_path)
                if step % log_every == 0 or step == 1 or step == total_steps:
                    for row in frame_debug_rows:
                        if not bool(row.get('matched', True)):
                            frame_line = f"[train]   seq={row['sequence_name']} frame={row['frame_id']} cls={row['target_detection_name']} target_lost=True reason={row.get('unmatched_reason', '')} candidates={int(row.get('candidate_total', 0))} after_conf={int(row.get('candidate_after_conf', 0))} after_dist={int(row.get('candidate_after_dist', 0))}"
                            print(frame_line)
                            _append_log_line(training_log_path, frame_line)
                continue
            amp_scale_before = float(scaler.get_scale()) if optimizer is not None and scaler.is_enabled() else 1.0
            amp_scale_after = amp_scale_before
            amp_step_skipped = False
            if style_term.requires_grad:
                if optimizer is not None:
                    scaler.scale(style_term).backward()
                else:
                    style_term.backward()
            evaluated_texture_snapshot = renderer.texture_param.detach().clone()
            if optimizer is not None:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_([renderer.texture_param], max_norm=float(grad_clip_norm))
                texture_grad_norm = 0.0
                texture_grad_finite = True
                if renderer.texture_param.grad is not None:
                    grad_detached = renderer.texture_param.grad.detach()
                    texture_grad_finite = bool(torch.isfinite(grad_detached).all().item())
                    if texture_grad_finite:
                        texture_grad_norm = float(grad_detached.norm().item())
                    else:
                        finite_grad = torch.nan_to_num(grad_detached, nan=0.0, posinf=0.0, neginf=0.0)
                        with torch.no_grad():
                            renderer.texture_param.grad.copy_(finite_grad)
                        texture_grad_norm = float(finite_grad.norm().item())
                scaler.step(optimizer)
                scaler.update()
                if scaler.is_enabled():
                    amp_scale_after = float(scaler.get_scale())
                    amp_step_skipped = amp_scale_after < amp_scale_before
            else:
                if renderer.texture_param.grad is None:
                    raise RuntimeError('PGD update: texture_param.grad is None')
                grad_detached = renderer.texture_param.grad.detach()
                texture_grad_finite = bool(torch.isfinite(grad_detached).all().item())
                texture_grad_norm = float(torch.nan_to_num(grad_detached, nan=0.0, posinf=0.0, neginf=0.0).norm().item())
                with torch.no_grad():
                    renderer.texture_param.add_(-learning_rate * renderer.texture_param.grad.sign())
                    if pgd_anchor is not None and pgd_epsilon > 0.0:
                        delta = (renderer.texture_param - pgd_anchor).clamp(min=-pgd_epsilon, max=pgd_epsilon)
                        renderer.texture_param.copy_(pgd_anchor + delta)
                renderer.texture_param.grad = None
            style_term_value = float(style_term.detach().item())
            min_target_confidence_value = min(target_confidence_values) if target_confidence_values else 0.0
            all_targets_present = bool(step_matched_frame_total > 0 and matched_frames_step == step_matched_frame_total and (target_lost_frames_step == 0))
            best_conf_gate_passed = bool(all_targets_present and min_target_confidence_value >= float(best_min_target_confidence))
            record = {'enable_query_terms': bool(loss_cfg.get('enable_query_losses', True)), 'step': step, 'loss_total': float(model_loss_value + auxiliary_loss_value + style_term_value), 'loss_model': float(model_loss_value), 'loss_stp3_new_mean': float(weighted_stp3_new_loss_value), 'loss_auxiliary_weighted': float(auxiliary_loss_value), 'auxiliary_models': auxiliary_step_records, 'loss_style_weighted': style_term_value, 'batch_mode': str(batch_mode), 'batch_sequence_names': step_sequence_names, 'batch_matched_frame_total': int(step_matched_frame_total), 'full_matched_frame_total': int(matched_frame_total), 'lr': float(learning_rate), 'style_tv': float(style_tv_raw.detach().item()), 'style_l2': float(style_l2_raw.detach().item()), 'style_brightness': float(style_brightness_raw.detach().item()), 'style_freq': 0.0, 'style_nps': float(style_nps_raw.detach().item()), 'matched_frames': matched_frames_step, 'target_lost_frames': int(target_lost_frames_step), 'target_expected_frames': int(step_matched_frame_total), 'all_targets_present': bool(all_targets_present), 'target_confidence': _mean_or_zero(target_confidence_values), 'target_confidence_min': float(min_target_confidence_value), 'best_conf_gate_passed': bool(best_conf_gate_passed), 'target_logit': _mean_or_zero(target_logit_values), 'progress_gain_mean': _mean_or_zero(progress_gain_values), 'move_lateral': _mean_or_zero(move_lateral_term_values), 'move_lateral_weight_mean': _mean_or_zero(move_lateral_weight_values), 'weighted_loss_move_lateral': _mean_or_zero(move_lateral_weighted_values), 'move_longitudinal': _mean_or_zero(move_longitudinal_term_values), 'loss_move_longitudinal_mean': _mean_or_zero(move_longitudinal_mean_values), 'loss_move_longitudinal_max': _mean_or_zero(move_longitudinal_max_values), 'loss_progress': _mean_or_zero(progress_term_values), 'loss_move_lateral': _mean_or_zero(move_lateral_term_values), 'loss_move_longitudinal': _mean_or_zero(move_longitudinal_term_values), 'loss_move_longitudinal_hard': _mean_or_zero(move_longitudinal_hard_values), 'longitudinal_hard_excess_mean': _mean_or_zero(move_longitudinal_hard_excess_values), 'longitudinal_hard_excess_max': _mean_or_zero(move_longitudinal_hard_excess_max_values), 'loss_first_frame_min': _mean_or_zero(first_frame_min_term_values), 'loss_per_frame_min': _mean_or_zero(per_frame_min_term_values), 'loss_rigid': _mean_or_zero(rigid_term_values), 'loss_rigid_size': _mean_or_zero(rigid_size_loss_values), 'loss_rigid_yaw': _mean_or_zero(rigid_yaw_loss_values), 'loss_cls': _mean_or_zero(cls_term_values), 'loss_cls_pos': _mean_or_zero(cls_pos_values), 'loss_cls_neg': _mean_or_zero(cls_neg_values), 'loss_cls_rank': _mean_or_zero(cls_rank_values), 'loss_cls_nearby': _mean_or_zero(cls_nearby_values), 'loss_cls_global_rank': _mean_or_zero(cls_global_rank_values), 'loss_cls_output_proxy': _mean_or_zero(cls_output_proxy_values), 'loss_query_identity': _mean_or_zero(cls_query_identity_values), 'loss_depth': _mean_or_zero(depth_term_values), 'depth_cam_count': _mean_or_zero(depth_cam_count_values), 'depth_sample_count': _mean_or_zero(depth_sample_count_values), 'depth_pred_mean_m': _mean_or_zero(depth_pred_mean_values), 'depth_gt_mean_m': _mean_or_zero(depth_gt_mean_values), 'noncar_max_confidence': _mean_or_zero(noncar_max_conf_values), 'nearby_target_max_confidence': _mean_or_zero(nearby_target_max_conf_values), 'nearby_query_count': _mean_or_zero(nearby_query_count_values), 'global_other_target_max_confidence': _mean_or_zero(global_other_target_max_conf_values), 'other_query_global_max_confidence': _mean_or_zero(other_query_global_max_conf_values), 'query_identity_cosine': _mean_or_zero(query_identity_cosine_values), 'progress_teacher': _mean_or_zero(progress_teacher_values), 'progress_floor': _mean_or_zero(progress_floor_values), 'difference_mean': _mean_or_zero(lateral_difference_values), 'longitudinal_difference_mean': _mean_or_zero(longitudinal_difference_values), 'longitudinal_difference_max': _mean_or_zero(longitudinal_difference_max_values), 'move_loss_type': str(loss_cfg.get('move_loss_type', '')), 'rigid_loss_type': str(loss_cfg.get('rigid_loss_type', '')), 'rigid_size_diff_mean': _mean_or_zero(rigid_size_diff_values), 'rigid_yaw_diff_mean': _mean_or_zero(rigid_yaw_diff_values), 'optimizer': optimizer_name, 'texture_grad_norm': float(texture_grad_norm), 'texture_grad_finite': bool(texture_grad_finite), 'texture_grad_clip_norm': 0.0, 'amp_scale_before': float(amp_scale_before), 'amp_scale_after': float(amp_scale_after), 'amp_step_skipped': bool(amp_step_skipped), 'hook_gradients': grad_by_frame, 'frame_metrics': frame_debug_rows}
            history.append(record)
            train_eval_record: Optional[Dict[str, Any]] = None
            train_eval_frame_debug_rows: List[Dict[str, Any]] = []
            train_should_eval = bool(step % train_eval_every == 0 or step == 1 or step == total_steps)
            if train_should_eval:
                train_eval_record, train_eval_frame_debug_rows = _evaluate_dataset_without_grad(frame_groups=frame_groups, renderer=renderer, bev_model=bev_model, fixed_queries=fixed_queries, loss_cfg=loss_cfg, device=device, use_amp=use_amp, amp_dtype=amp_dtype, apply_eot=False, clean_query_features=clean_query_features, loss_reference_mode=loss_reference_mode, clean_detection_refs=train_clean_detection_refs, final_decode_match=bool(uses_final_decode_match), final_decode_conf_threshold=fixed_conf_threshold, final_decode_max_center_dist_m=fixed_max_center_dist_m, final_decode_distance_axis=fixed_distance_axis, final_decode_max_cross_axis_dist_m=fixed_max_cross_axis_dist_m)
                train_eval_best_gate_passed = _best_gate_passed_from_eval_record(train_eval_record, min_target_confidence=best_min_target_confidence)
                train_eval_record['best_conf_gate_passed'] = bool(train_eval_best_gate_passed)
                train_eval_record['frame_metrics'] = train_eval_frame_debug_rows
                train_eval_history.append({'step': step, 'eval': copy.deepcopy(train_eval_record)})
                record['train_full_loss_total'] = float(train_eval_record['loss_total'])
                record['train_full_best_conf_gate_passed'] = bool(train_eval_best_gate_passed)
                record['train_full_matched_frames'] = int(train_eval_record.get('matched_frames', 0))
                record['train_full_target_expected_frames'] = int(train_eval_record.get('target_expected_frames', 0))
                record['train_full_target_lost_frames'] = int(train_eval_record.get('target_lost_frames', 0))
                record['train_full_target_confidence_min'] = float(train_eval_record.get('target_confidence_min', 0.0))
                if step % log_every == 0 or step == 1 or step == total_steps:
                    _log_dataset_eval_record(prefix='[train-full]', step=step, record=train_eval_record, frame_debug_rows=train_eval_frame_debug_rows, training_log_path=training_log_path)
            val_record: Optional[Dict[str, Any]] = None
            val_frame_debug_rows: List[Dict[str, Any]] = []
            val_should_eval = bool(val_enabled and val_renderer is not None and (step % val_eval_every == 0 or step == 1 or step == total_steps))
            val_should_log = bool(step % val_log_every == 0 or step == 1 or step == total_steps)
            if val_should_eval and val_renderer is not None:
                val_record, val_frame_debug_rows = _evaluate_dataset_without_grad(frame_groups=val_frame_groups, renderer=val_renderer, bev_model=bev_model, fixed_queries=val_fixed_queries, loss_cfg=loss_cfg, device=device, use_amp=use_amp, amp_dtype=amp_dtype, apply_eot=val_use_eot, clean_query_features=val_clean_query_features, loss_reference_mode=loss_reference_mode, clean_detection_refs=val_clean_detection_refs, final_decode_match=bool(uses_final_decode_match), final_decode_conf_threshold=fixed_conf_threshold, final_decode_max_center_dist_m=fixed_max_center_dist_m, final_decode_distance_axis=fixed_distance_axis, final_decode_max_cross_axis_dist_m=fixed_max_cross_axis_dist_m)
                val_best_gate_passed = _best_gate_passed_from_eval_record(val_record, min_target_confidence=best_min_target_confidence)
                val_record['best_conf_gate_passed'] = bool(val_best_gate_passed)
                val_record['frame_metrics'] = val_frame_debug_rows
                val_history.append({'step': step, 'eval': copy.deepcopy(val_record)})
                record['best_selection_source'] = 'disabled'
                record['val_loss_total'] = float(val_record['loss_total'])
                record['val_best_conf_gate_passed'] = bool(val_best_gate_passed)
                record['val_matched_frames'] = int(val_record.get('matched_frames', 0))
                record['val_target_expected_frames'] = int(val_record.get('target_expected_frames', 0))
                record['val_target_lost_frames'] = int(val_record.get('target_lost_frames', 0))
                record['val_target_confidence_min'] = float(val_record.get('target_confidence_min', 0.0))
                if val_should_log:
                    _log_dataset_eval_record(prefix='[val]', step=step, record=val_record, frame_debug_rows=val_frame_debug_rows, training_log_path=training_log_path)
            else:
                record['best_selection_source'] = 'disabled'
            if _should_plot_training_curves(step):
                _plot_training_loss_curves(history=history, train_eval_history=train_eval_history, val_history=val_history, output_path=loss_plot_path, training_log_path=training_log_path)
            if step % checkpoint_every == 0 or step == total_steps:
                best_step = step
                best_loss = float(record.get('loss_total', 0.0))
                best_snapshot = renderer.texture_param.detach().clone()
                latest_step_ckpt_path = _save_training_checkpoint(checkpoint_dir=_checkpoint_dir(output_dir), filename=f'step-{step:04d}.pt', step=step, output_dir=output_dir, config_path=config_path, optimizer_name=optimizer_name, texture_param=renderer.texture_param, best_snapshot=best_snapshot, best_loss=best_loss, best_step=best_step, history=history, optimizer=optimizer, scaler=scaler if optimizer is not None else None, pgd_anchor=pgd_anchor, train_eval_history=train_eval_history, val_history=val_history)
                step_ckpt_paths.append(str(latest_step_ckpt_path))
                _append_log_line(training_log_path, f'[train] step checkpoint saved step={step} path={latest_step_ckpt_path}')
            if step % log_every == 0 or step == 1 or step == total_steps:
                step_line = f"[train] step-{step:04d}: batch={','.join(step_sequence_names)} " + _format_weighted_loss_formula(record=record, loss_cfg=loss_cfg, style_cfg=style_cfg, is_bevdet_model=is_bevdet_model)
                print(step_line)
                _append_log_line(training_log_path, step_line)
                for row in frame_debug_rows:
                    if not bool(row.get('matched', True)):
                        frame_line = f"[train]   seq={row['sequence_name']} frame={row['frame_id']} cls={row['target_detection_name']} target_lost=True reason={row.get('unmatched_reason', '')} candidates={int(row.get('candidate_total', 0))} after_conf={int(row.get('candidate_after_conf', 0))} after_dist={int(row.get('candidate_after_dist', 0))}"
                        print(frame_line)
                        _append_log_line(training_log_path, frame_line)
                        continue
                    frame_line = f"[train]   seq={row['sequence_name']} frame={row['frame_id']} cls={row['target_detection_name']} ref={row.get('reference_mode', 'gt')} shift={row['direction']}:{row['shift_abs_m']:.4f}m x={row['x_direction']}:{row['x_shift_abs_m']:.4f}m conf={row['target_confidence']:.4f} query={row.get('query_idx', -1)}" + (f" note={row.get('match_note', '')}" if row.get('match_note') else '')
                    print(frame_line)
                    _append_log_line(training_log_path, frame_line)
                for aux_row in auxiliary_step_records:
                    aux_line = f"[aux:{aux_row.get('model', '')}] step-{step:04d}: weighted={float(aux_row.get('loss_weighted', 0.0)):.6f} raw={float(aux_row.get('loss_model', 0.0)):.6f} weight={float(aux_row.get('weight', 0.0)):.3f} matched={int(aux_row.get('matched_frames', 0))}/{int(aux_row.get('target_expected_frames', 0))} lost={int(aux_row.get('target_lost_frames', 0))} conf={float(aux_row.get('target_confidence', 0.0)):.4f}"
                    print(aux_line)
                    _append_log_line(training_log_path, aux_line)
    except KeyboardInterrupt:
        interrupted = True
        stop_message = '[train] interrupt: exporting best-texture visuals and official results'
        print(stop_message)
        _append_log_line(training_log_path, stop_message)
    except Exception as exc:
        _append_exception_log(training_log_path, exc, prefix='[train] training phase failed')
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if 'auxiliary_runtimes' in locals():
            for aux_runtime in auxiliary_runtimes:
                try:
                    if not output_dir_missing_exit:
                        _append_log_line(training_log_path, f'[aux:{aux_runtime.model_name}] release training GPU memory')
                    _release_training_runtime(bev_model=aux_runtime.model)
                except Exception:
                    pass
        if output_dir_missing_exit:
            try:
                if 'val_renderer' in locals() and val_renderer is not None:
                    val_renderer.release_gpu()
            except Exception:
                pass
            _release_training_runtime(bev_model=bev_model if 'bev_model' in locals() else None, renderer=renderer if 'renderer' in locals() else None, optimizer=optimizer if 'optimizer' in locals() else None, scaler=scaler if 'scaler' in locals() and optimizer is not None else None)
            current_step_for_exit = int(locals().get('step', max(0, start_step - 1)))
            return {'config_path': str(config_path), 'config_snapshot_path': str(config_snapshot_path), 'model': active_model_name if 'active_model_name' in locals() else '', 'output_dir': str(output_dir), 'sequence_pkl': str(sequence_pkl) if 'sequence_pkl' in locals() else '', 'matched_query_frames': {}, 'latest_checkpoint_step': int(best_step) if 'best_step' in locals() else int(current_step_for_exit), 'latest_checkpoint_loss': float(best_loss) if 'best_loss' in locals() and math.isfinite(float(best_loss)) else 0.0, 'latest_step_ckpt_path': str(latest_step_ckpt_path) if 'latest_step_ckpt_path' in locals() and latest_step_ckpt_path is not None else '', 'step_ckpt_paths': step_ckpt_paths if 'step_ckpt_paths' in locals() else [], 'last_ckpt_path': '', 'optimizer': optimizer_name if 'optimizer_name' in locals() else '', 'lr': float(learning_rate) if 'learning_rate' in locals() else 0.0, 'pgd_epsilon': float(pgd_epsilon) if 'pgd_epsilon' in locals() else 0.0, 'history': history if 'history' in locals() else [], 'train_eval_history': train_eval_history if 'train_eval_history' in locals() else [], 'val_history': val_history if 'val_history' in locals() else [], 'val_enabled': bool(val_sequence_yaml_paths) if 'val_sequence_yaml_paths' in locals() else False, 'val_sequence_pkl': str(val_sequence_pkl) if 'val_sequence_pkl' in locals() and val_sequence_pkl is not None else '', 'current_frame_metrics': [], 'before_visual_dir': '', 'after_visual_dir': str(after_dir) if 'after_dir' in locals() else '', 'interrupted': True, 'stop_signal': None, 'official_results_path': '', 'official_visual_dir': '', 'fixed_query_results_path': '', 'fixed_query_visual_dir': '', 'clean_results_by_sequence': clean_results_by_sequence if 'clean_results_by_sequence' in locals() else {}, 'clean_visuals_by_sequence': clean_visuals_by_sequence if 'clean_visuals_by_sequence' in locals() else {}, 'official_results_by_sequence': {}, 'official_visuals_by_sequence': {}, 'fixed_query_results_by_sequence': {}, 'fixed_query_visuals_by_sequence': {}, 'fixed_query_trace_check_by_sequence': {}, 'official_error': 'output_dir_missing', 'fixed_query_error': '', 'training_log_path': str(training_log_path), 'loss_plot_path': str(loss_plot_path) if 'loss_plot_path' in locals() else ''}
        _release_cuda_memory(bev_model=bev_model if 'bev_model' in locals() else None, optimizer=optimizer if 'optimizer' in locals() else None)
        if 'renderer' in locals():
            current_step_for_last = int(locals().get('step', 0))
            if current_step_for_last <= 0:
                current_step_for_last = max(0, start_step - 1)
            last_ckpt_path = _save_training_checkpoint(checkpoint_dir=_checkpoint_dir(output_dir), filename='last.pt', step=current_step_for_last, output_dir=output_dir, config_path=config_path, optimizer_name=optimizer_name, texture_param=renderer.texture_param, best_snapshot=renderer.texture_param.detach().clone(), best_loss=best_loss, best_step=best_step, history=history, optimizer=optimizer, scaler=scaler if optimizer is not None else None, pgd_anchor=pgd_anchor, train_eval_history=train_eval_history, val_history=val_history)
            _append_log_line(training_log_path, f'[train] last checkpoint saved step={current_step_for_last} path={last_ckpt_path}')
        try:
            best_frame_rows = _evaluate_fixed_queries_for_snapshot(frames=frames, renderer=renderer, bev_model=bev_model, fixed_queries=fixed_queries, device=device, use_amp=use_amp, amp_dtype=amp_dtype, final_decode_match=bool(uses_final_decode_match), final_decode_conf_threshold=fixed_conf_threshold, final_decode_max_center_dist_m=fixed_max_center_dist_m, final_decode_distance_axis=fixed_distance_axis, final_decode_max_cross_axis_dist_m=fixed_max_cross_axis_dist_m)
        except Exception as exc:
            if _is_cuda_oom(exc):
                _append_log_line(training_log_path, '[train] first OOM during texture eval; cleared cache and retrying')
                _release_cuda_memory(bev_model=bev_model, optimizer=optimizer)
                best_frame_rows = _evaluate_fixed_queries_for_snapshot(frames=frames, renderer=renderer, bev_model=bev_model, fixed_queries=fixed_queries, device=device, use_amp=use_amp, amp_dtype=amp_dtype, final_decode_match=bool(uses_final_decode_match), final_decode_conf_threshold=fixed_conf_threshold, final_decode_max_center_dist_m=fixed_max_center_dist_m, final_decode_distance_axis=fixed_distance_axis, final_decode_max_cross_axis_dist_m=fixed_max_cross_axis_dist_m)
            else:
                raise
        current_step_for_eval = int(locals().get('step', max(0, start_step - 1)))
        current_header = f'[train] current texture results step={current_step_for_eval}'
        print(current_header)
        _append_log_line(training_log_path, current_header)
        for row in best_frame_rows:
            if not bool(row.get('matched', True)):
                best_frame_line = f"[train]   current seq={row['sequence_name']} frame={row['frame_id']} cls={row['target_detection_name']} target_lost=True reason={row.get('unmatched_reason', '')} candidates={int(row.get('candidate_total', 0))} after_conf={int(row.get('candidate_after_conf', 0))} after_dist={int(row.get('candidate_after_dist', 0))}"
                print(best_frame_line)
                _append_log_line(training_log_path, best_frame_line)
                continue
            best_frame_line = f"[train]   current seq={row['sequence_name']} frame={row['frame_id']} query={row['query_idx']} cls={row['target_detection_name']} shift={row['direction']}:{row['shift_abs_m']:.4f}m x={row['x_direction']}:{row['x_shift_abs_m']:.4f}m conf={row['target_confidence']:.4f} size_diff={row['size_diff_mean']:.4f} yaw_diff={row.get('yaw_diff_deg', math.degrees(float(row.get('yaw_diff_rad', 0.0)))):.2f}deg"
            print(best_frame_line)
            _append_log_line(training_log_path, best_frame_line)
        try:
            renderer.export_visuals(output_dir=after_dir, frames=frames, tag='after')
            overrides_by_sequence: Dict[str, Dict[str, Dict[str, Any]]] = {}
            fixed_query_overrides = _collect_fixed_query_override_rows(frames=frames, renderer=renderer, bev_model=bev_model, fixed_queries=fixed_queries, device=device, use_amp=use_amp, amp_dtype=amp_dtype)
            for frame_key, payload in fixed_query_overrides.items():
                frame = next((item for item in frames if item.cache_key == frame_key), None)
                if frame is None:
                    continue
                overrides_by_sequence.setdefault(str(frame.sequence_name), {})[str(frame.sample_token)] = payload
            if val_enabled and val_renderer is not None:
                try:
                    if uses_final_decode_match:
                        val_best_records = _run_fast_final_decode_val_target_check(frame_groups=val_frame_groups, renderer=val_renderer, bev_model=bev_model, device=device, use_amp=use_amp, amp_dtype=amp_dtype, apply_eot=val_use_eot, loss_reference_mode=loss_reference_mode, clean_detection_refs=val_clean_detection_refs, conf_threshold=fixed_conf_threshold, max_center_dist_m=fixed_max_center_dist_m, distance_axis=fixed_distance_axis, max_cross_axis_dist_m=fixed_max_cross_axis_dist_m, export_label='current', training_log_path=training_log_path)
                        val_history.append({'step': 'current', 'fast_final_decode': val_best_records})
                    else:
                        _append_log_line(training_log_path, '[val] export current patch official results and visuals')
                        val_best_records = _export_val_official_current_results(config=config, output_dir=output_dir, val_frame_groups=val_frame_groups, val_renderer=val_renderer, bev_model=bev_model, val_single_sequence_pkl_map=val_single_sequence_pkl_map, val_sequence_pkl=val_sequence_pkl, fallback_sequence_pkl=sequence_pkl, val_info_by_sequence_token=val_info_by_sequence_token, val_info_by_cache_key=val_info_by_cache_key, val_fixed_queries=val_fixed_queries, active_model_name=active_model_name, is_bevdet_model=is_bevdet_model, val_use_eot=val_use_eot, fixed_conf_threshold=fixed_conf_threshold, fixed_max_center_dist_m=fixed_max_center_dist_m, fixed_distance_axis=fixed_distance_axis, fixed_max_cross_axis_dist_m=fixed_max_cross_axis_dist_m, training_log_path=training_log_path)
                        val_history.append({'step': 'current', 'official': val_best_records})
                except Exception as exc:
                    val_error = f'{type(exc).__name__}: {exc}'
                    val_history.append({'step': 'current', 'val': [], 'error': val_error})
                    _append_exception_log(training_log_path, exc, prefix='[val] current patch check failed')
                finally:
                    try:
                        val_renderer.release_gpu()
                    except Exception:
                        pass
                    val_renderer = None
                    _release_cuda_memory(bev_model=bev_model, optimizer=optimizer)
            runtime_released_for_official = False
            official_image_roots: Dict[str, Path] = {}
            fixed_query_image_roots: Dict[str, Path] = {}
            lidar_paths_by_sample = {str(frame.cache_key): _as_path(str(info_by_cache_key[frame.cache_key]['lidar_path'])) for frame in frames if frame.cache_key in info_by_cache_key and info_by_cache_key[frame.cache_key].get('lidar_path')}
            for group_frames in frame_groups:
                sequence_name = str(group_frames[0].sequence_name)
                official_export_subdir = f'official/{sequence_name}' if len(frame_groups) > 1 else 'official'
                fixed_export_subdir = f'fixed_query/{sequence_name}' if len(frame_groups) > 1 else 'fixed_query'
                official_image_roots[sequence_name] = _export_image_tree(frames=group_frames, output_dir=output_dir, export_subdir=official_export_subdir, image_source_subdir=sequence_name, image_provider=lambda frame: renderer.build_frame_images(frame, apply_eot=False), lidar_paths_by_sample=lidar_paths_by_sample)
                if not is_bevdet_model:
                    fixed_query_image_roots[sequence_name] = _export_image_tree(frames=group_frames, output_dir=output_dir, export_subdir=fixed_export_subdir, image_source_subdir=sequence_name, image_provider=lambda frame: renderer.build_frame_images(frame, apply_eot=False), lidar_paths_by_sample=lidar_paths_by_sample)

            def _cleanup_before_official_test() -> None:
                nonlocal runtime_released_for_official
                if runtime_released_for_official:
                    return
                _append_log_line(training_log_path, '[train] release training memory before official test')
                _release_training_runtime(bev_model=bev_model, renderer=renderer, optimizer=optimizer, scaler=scaler if optimizer is not None else None)
                runtime_released_for_official = True
            for index, group_frames in enumerate(frame_groups):
                sequence_name = str(group_frames[0].sequence_name)
                group_sequence_pkl = single_sequence_pkl_map.get(sequence_name, sequence_pkl)
                official_export_subdir = f'official/{sequence_name}' if len(frame_groups) > 1 else 'official'
                fixed_export_subdir = f'fixed_query/{sequence_name}' if len(frame_groups) > 1 else 'fixed_query'
                cleanup_fn = _cleanup_before_official_test if index == 0 else None
                official_results_path, official_visual_dir = _export_official_results_and_visuals_from_image_root(config=config, image_source_subdir=sequence_name, image_root=official_image_roots[sequence_name], output_dir=output_dir, frames=group_frames, bev_model=bev_model, sequence_pkl=group_sequence_pkl, info_by_token=info_by_sequence_token[sequence_name], export_subdir=official_export_subdir, cleanup_before_test=cleanup_fn, image_provider=lambda frame: renderer.build_frame_images(frame, apply_eot=False))
                official_results_by_sequence[sequence_name] = str(official_results_path)
                official_visuals_by_sequence[sequence_name] = str(official_visual_dir)
                official_ok_message = f'[train] official results saved seq={sequence_name}: results={official_results_path} visuals={official_visual_dir}'
                print(official_ok_message)
                _append_log_line(training_log_path, official_ok_message)
                official_trace_path = official_results_path.parent / 'results_nusc_query_trace.json'
                if is_bevdet_model:
                    skip_trace_line = f'[train] fixed-query trace check seq={sequence_name} skipped (model={active_model_name} has no transformer query trace; training used final-decode matching)'
                    print(skip_trace_line)
                    _append_log_line(training_log_path, skip_trace_line)
                    continue
                if official_trace_path.exists():
                    trace_check = _validate_fixed_query_trace_identity(frames=group_frames, fixed_queries=fixed_queries, official_query_trace_path=official_trace_path)
                    fixed_query_trace_check_by_sequence[sequence_name] = trace_check
                    trace_line = f"[train] fixed-query trace check seq={sequence_name} passed={trace_check['passed']}/{trace_check['total']} failed={trace_check['failed']} missing={trace_check['missing']}"
                    print(trace_line)
                    _append_log_line(training_log_path, trace_line)
                    for item in trace_check['rows']:
                        mismatch_line = f"[train]   trace-mismatch seq={item['sequence_name']} frame={item['frame_id']} sample={item['sample_token']} expected_q={item['expected_query_idx']} top_q={item['top_query_idx']} top_score={item.get('top_score', 0.0):.4f} reason={item['reason']}"
                        print(mismatch_line)
                        _append_log_line(training_log_path, mismatch_line)
                    if bool(fixed_query_cfg.get('fail_on_query_mismatch', True)) and int(trace_check['failed']) > 0:
                        raise RuntimeError(f"[fixed-query] official top1 query mismatches fixed query: seq={sequence_name} {trace_check['failed']}/{trace_check['total']} frames failed")
                else:
                    skip_trace_line = f'[train] fixed-query trace check seq={sequence_name} skipped (model={active_model_name}, trace_exists={official_trace_path.exists()})'
                    print(skip_trace_line)
                    _append_log_line(training_log_path, skip_trace_line)
                fixed_query_results_path, fixed_query_visual_dir = _export_fixed_query_results_and_visuals(config=config, image_source_subdir=sequence_name, output_dir=output_dir, frames=group_frames, image_root=fixed_query_image_roots[sequence_name], fixed_query_overrides=overrides_by_sequence.get(sequence_name, {}), official_results_path=official_results_path, official_query_trace_path=official_trace_path, export_subdir=fixed_export_subdir)
                fixed_query_results_by_sequence[sequence_name] = str(fixed_query_results_path)
                fixed_query_visuals_by_sequence[sequence_name] = str(fixed_query_visual_dir)
                fixed_query_ok_message = f'[train] fixed-query results saved seq={sequence_name}: results={fixed_query_results_path} visuals={fixed_query_visual_dir}'
                print(fixed_query_ok_message)
                _append_log_line(training_log_path, fixed_query_ok_message)
            if len(frame_groups) == 1:
                official_results_path = Path(official_results_by_sequence[frame_groups[0][0].sequence_name])
                official_visual_dir = Path(official_visuals_by_sequence[frame_groups[0][0].sequence_name])
                if not is_bevdet_model:
                    fixed_query_results_path = Path(fixed_query_results_by_sequence[frame_groups[0][0].sequence_name])
                    fixed_query_visual_dir = Path(fixed_query_visuals_by_sequence[frame_groups[0][0].sequence_name])
        except Exception as exc:
            official_error = f'{type(exc).__name__}: {exc}'
            if '[fixed-query]' in str(exc):
                fixed_query_error = official_error
            if _is_cuda_oom(exc):
                _append_log_line(training_log_path, '[train] export OOM; logged failure; kept step checkpoint')
            else:
                _append_exception_log(training_log_path, exc, prefix='[train] export phase failed')
            shutil.rmtree(output_dir / 'official' / 'images', ignore_errors=True)
            shutil.rmtree(output_dir / 'fixed_query' / 'images', ignore_errors=True)
            official_fail_message = f'[train] official export failed: {official_error}'
            print(official_fail_message)
            _append_log_line(training_log_path, official_fail_message)
    matched_query_frames_summary = {f'{frame.sequence_name}:frame-{frame.frame_id}': {'sequence_name': frame.sequence_name, 'sample_token': frame.sample_token, 'matched': bool(fixed_queries[frame.cache_key].matched), 'query_idx': int(fixed_queries[frame.cache_key].query_idx), 'confidence': float(fixed_queries[frame.cache_key].confidence), 'world_distance_m': float(fixed_queries[frame.cache_key].world_distance_m), 'candidate_total': int(fixed_queries[frame.cache_key].candidate_total), 'candidate_after_conf': int(fixed_queries[frame.cache_key].candidate_after_conf), 'candidate_after_dist': int(fixed_queries[frame.cache_key].candidate_after_dist), 'unmatched_reason': str(fixed_queries[frame.cache_key].unmatched_reason)} for frame in frames}
    summary = {'config_path': str(config_path), 'config_snapshot_path': str(config_snapshot_path), 'model': active_model_name, 'output_dir': str(output_dir), 'sequence_pkl': str(sequence_pkl), 'matched_query_frames': matched_query_frames_summary, 'latest_checkpoint_step': int(best_step), 'latest_checkpoint_loss': float(best_loss) if math.isfinite(float(best_loss)) else 0.0, 'latest_step_ckpt_path': str(latest_step_ckpt_path) if latest_step_ckpt_path is not None else '', 'step_ckpt_paths': step_ckpt_paths, 'last_ckpt_path': str(last_ckpt_path) if last_ckpt_path is not None else '', 'optimizer': optimizer_name, 'lr': float(learning_rate), 'pgd_epsilon': float(pgd_epsilon), 'history': history, 'train_eval_history': train_eval_history, 'val_history': val_history, 'val_enabled': bool(val_sequence_yaml_paths), 'val_sequence_pkl': str(val_sequence_pkl) if val_sequence_pkl is not None else '', 'current_frame_metrics': best_frame_rows, 'before_visual_dir': '', 'after_visual_dir': str(after_dir), 'interrupted': bool(interrupted), 'stop_signal': int(stop_signal) if stop_signal is not None else None, 'official_results_path': str(official_results_path) if official_results_path is not None else '', 'official_visual_dir': str(official_visual_dir) if official_visual_dir is not None else '', 'fixed_query_results_path': str(fixed_query_results_path) if fixed_query_results_path is not None else '', 'fixed_query_visual_dir': str(fixed_query_visual_dir) if fixed_query_visual_dir is not None else '', 'clean_results_by_sequence': clean_results_by_sequence, 'clean_visuals_by_sequence': clean_visuals_by_sequence, 'official_results_by_sequence': official_results_by_sequence, 'official_visuals_by_sequence': official_visuals_by_sequence, 'fixed_query_results_by_sequence': fixed_query_results_by_sequence, 'fixed_query_visuals_by_sequence': fixed_query_visuals_by_sequence, 'fixed_query_trace_check_by_sequence': fixed_query_trace_check_by_sequence, 'official_error': official_error, 'fixed_query_error': fixed_query_error, 'training_log_path': str(training_log_path), 'loss_plot_path': str(loss_plot_path)}
    _append_log_line(training_log_path, f"[train] latest step checkpoint step={summary['latest_checkpoint_step']} loss={summary['latest_checkpoint_loss']:.6f}")
    _append_log_line(training_log_path, f"[train] latest step checkpoint path={summary['latest_step_ckpt_path']}")
    _append_log_line(training_log_path, f"[train] last checkpoint path={summary['last_ckpt_path']}")
    _append_log_line(training_log_path, f"[train] after_train_visual_dir={summary['after_visual_dir']}")
    _append_log_line(training_log_path, f"[train] official_results_path={summary['official_results_path']}")
    _append_log_line(training_log_path, f"[train] official_visual_dir={summary['official_visual_dir']}")
    _append_log_line(training_log_path, f"[train] loss_curve_path={summary['loss_plot_path']}")
    _append_log_line(training_log_path, f"[train] fixed_query_results_path={summary['fixed_query_results_path']}")
    _append_log_line(training_log_path, f"[train] fixed_query_visual_dir={summary['fixed_query_visual_dir']}")
    _append_log_line(training_log_path, f"[val] enabled={str(summary['val_enabled']).lower()}")
    _append_log_line(training_log_path, f"[val] sequence_pkl={summary['val_sequence_pkl']}")
    if clean_results_by_sequence:
        _append_log_line(training_log_path, f'[train] clean_results_by_seq={json.dumps(clean_results_by_sequence, ensure_ascii=False)}')
        _append_log_line(training_log_path, f'[train] clean_visuals_by_seq={json.dumps(clean_visuals_by_sequence, ensure_ascii=False)}')
    if official_results_by_sequence:
        _append_log_line(training_log_path, f'[train] official_results_by_seq={json.dumps(official_results_by_sequence, ensure_ascii=False)}')
        _append_log_line(training_log_path, f'[train] official_visuals_by_seq={json.dumps(official_visuals_by_sequence, ensure_ascii=False)}')
    if fixed_query_results_by_sequence:
        _append_log_line(training_log_path, f'[train] fixed_query_results_by_seq={json.dumps(fixed_query_results_by_sequence, ensure_ascii=False)}')
        _append_log_line(training_log_path, f'[train] fixed_query_visuals_by_seq={json.dumps(fixed_query_visuals_by_sequence, ensure_ascii=False)}')
    if fixed_query_trace_check_by_sequence:
        _append_log_line(training_log_path, f'[train] fixed_query_trace_checks_by_seq={json.dumps(fixed_query_trace_check_by_sequence, ensure_ascii=False)}')
    return summary

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fix pose and scale; optimize camouflage texture only (BEVDet / BEVDepth / FastBEV)')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH, help='Training config YAML path')
    parser.add_argument('--ckpy', '--ckpt', dest='resume_ckpt', type=Path, default=None, help='Resume from checkpoint; pass step-xxxx.pt/last.pt or its directory')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    summary = run_training(args.config, resume_ckpt=args.resume_ckpt)
    print(f"[train] model={summary.get('model', '')}")
    print(f"[train] latest step checkpoint step={summary['latest_checkpoint_step']} loss={summary['latest_checkpoint_loss']:.6f}")
    print(f"[train] latest step checkpoint path={summary.get('latest_step_ckpt_path', '')}")
    print(f"[train] after_train_visual_dir={summary['after_visual_dir']}")
    print(f"[train] official_results_path={summary.get('official_results_path', '')}")
    print(f"[train] official_visual_dir={summary.get('official_visual_dir', '')}")
    print(f"[train] training_log_path={summary.get('training_log_path', '')}")
if __name__ == '__main__':
    main()
