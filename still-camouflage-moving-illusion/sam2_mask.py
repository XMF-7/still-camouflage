from __future__ import annotations
import argparse
import gc
import json
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from match_target_car import _as_path, _load_yaml, _save_yaml, _to_builtin
from nusc_gt_to_mesh import run_mesh_projection
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / 'config.yaml'

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

def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fp:
        json.dump(_to_builtin(payload), fp, ensure_ascii=False, indent=2)

def _load_mesh_summary_or_build(*, config_path: Path, mesh_summary_yaml: Optional[Path], run_mesh_if_missing: bool, near_plane_m: float, device: str) -> Tuple[Dict[str, Any], Path]:
    if mesh_summary_yaml is None:
        if not run_mesh_if_missing:
            raise FileNotFoundError('Mesh summary not provided. Pass --mesh-summary-yaml, or enable --run-mesh-if-missing.')
        summary_path = (Path(tempfile.mkdtemp(prefix='zz3-sam2-mesh-')) / 'mesh_projection_summary.yaml').resolve()
    else:
        summary_path = _as_path(mesh_summary_yaml)
    if summary_path.exists():
        return (_load_yaml(summary_path), summary_path)
    if not run_mesh_if_missing:
        raise FileNotFoundError(f'Mesh summary not found: {summary_path}. Run nusc_gt_to_mesh.py first, or enable --run-mesh-if-missing.')
    _, generated_summary_path = run_mesh_projection(config_path=config_path, binding_yaml=None, output_dir=summary_path.parent, device=device, near_plane_m=near_plane_m, run_match_if_missing=True, save_json=False, verbose=True)
    return (_load_yaml(generated_summary_path), generated_summary_path)

def _import_sam2(repo_root: Path) -> Tuple[Any, Any, Any, Any]:
    repo_root = _as_path(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:
        raise ImportError(f'Cannot import SAM2 from repo={repo_root}. Ensure dependencies are installed in current environment.') from exc
    return (torch, build_sam2, SAM2ImagePredictor, repo_root)

def _config_name_from_checkpoint(ckpt_path: Path) -> str:
    name = ckpt_path.name
    if 'sam2.1_hiera_large' in name:
        return 'configs/sam2.1/sam2.1_hiera_l.yaml'
    if 'sam2.1_hiera_base_plus' in name:
        return 'configs/sam2.1/sam2.1_hiera_b+.yaml'
    if 'sam2.1_hiera_small' in name:
        return 'configs/sam2.1/sam2.1_hiera_s.yaml'
    if 'sam2.1_hiera_tiny' in name:
        return 'configs/sam2.1/sam2.1_hiera_t.yaml'
    if 'sam2_hiera_large' in name:
        return 'configs/sam2/sam2_hiera_l.yaml'
    if 'sam2_hiera_base_plus' in name:
        return 'configs/sam2/sam2_hiera_b+.yaml'
    if 'sam2_hiera_small' in name:
        return 'configs/sam2/sam2_hiera_s.yaml'
    if 'sam2_hiera_tiny' in name:
        return 'configs/sam2/sam2_hiera_t.yaml'
    raise ValueError(f'Cannot infer SAM2 config from checkpoint name: {name}')

def _safe_center_from_bbox(bbox: Optional[List[float]], width: int, height: int) -> Optional[Tuple[float, float]]:
    if bbox is None or len(bbox) != 4:
        return None
    x = float((bbox[0] + bbox[2]) * 0.5)
    y = float((bbox[1] + bbox[3]) * 0.5)
    x = float(np.clip(x, 0.0, max(0.0, width - 1.0)))
    y = float(np.clip(y, 0.0, max(0.0, height - 1.0)))
    return (x, y)

def _normalize_bbox_xyxy(bbox: Optional[List[float]], width: int, height: int) -> Optional[np.ndarray]:
    if bbox is None or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = float(np.clip(min(x1, x2), 0.0, max(0.0, width - 1.0)))
    y1 = float(np.clip(min(y1, y2), 0.0, max(0.0, height - 1.0)))
    x2 = float(np.clip(max(x1, x2), 0.0, max(0.0, width - 1.0)))
    y2 = float(np.clip(max(y1, y2), 0.0, max(0.0, height - 1.0)))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)

def _union_bbox(a: Optional[np.ndarray], b: Optional[np.ndarray], width: int, height: int) -> Optional[np.ndarray]:
    if a is None:
        return b
    if b is None:
        return a
    x1 = float(np.clip(min(float(a[0]), float(b[0])), 0.0, max(0.0, width - 1.0)))
    y1 = float(np.clip(min(float(a[1]), float(b[1])), 0.0, max(0.0, height - 1.0)))
    x2 = float(np.clip(max(float(a[2]), float(b[2])), 0.0, max(0.0, width - 1.0)))
    y2 = float(np.clip(max(float(a[3]), float(b[3])), 0.0, max(0.0, height - 1.0)))
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)

def _expand_bbox(bbox: Optional[np.ndarray], width: int, height: int, ratio: float=0.08) -> Optional[np.ndarray]:
    if bbox is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    pad_x = max(6.0, (x2 - x1) * float(ratio))
    pad_y = max(6.0, (y2 - y1) * float(ratio))
    return np.asarray([float(np.clip(x1 - pad_x, 0.0, max(0.0, width - 1.0))), float(np.clip(y1 - pad_y, 0.0, max(0.0, height - 1.0))), float(np.clip(x2 + pad_x, 0.0, max(0.0, width - 1.0))), float(np.clip(y2 + pad_y, 0.0, max(0.0, height - 1.0)))], dtype=np.float32)

def _dedupe_points(points: List[Tuple[float, float]], width: int, height: int) -> List[Tuple[float, float]]:
    seen = set()
    out: List[Tuple[float, float]] = []
    for x, y in points:
        px = int(np.clip(round(float(x)), 0, width - 1))
        py = int(np.clip(round(float(y)), 0, height - 1))
        key = (px, py)
        if key in seen:
            continue
        seen.add(key)
        out.append((float(px), float(py)))
    return out

def _sample_mesh_positive_points(mesh_mask: np.ndarray, *, prompt_xy: Optional[Tuple[float, float]], bbox_prompt: Optional[np.ndarray]) -> List[Tuple[float, float]]:
    h, w = mesh_mask.shape
    mask_u8 = mesh_mask.astype(np.uint8) * 255
    if int(mask_u8.sum()) <= 0:
        return [prompt_xy] if prompt_xy is not None else []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    interior = cv2.erode(mask_u8, kernel, iterations=1)
    interior_bool = interior > 0
    if int(interior_bool.sum()) <= 0:
        interior_bool = mesh_mask.astype(bool)
    ys, xs = np.where(interior_bool)
    points: List[Tuple[float, float]] = []
    if prompt_xy is not None:
        points.append(prompt_xy)
    if xs.size > 0:
        center_idx = int(xs.size // 2)
        points.append((float(np.mean(xs)), float(np.mean(ys))))
        points.append((float(xs[np.argmin(ys)]), float(ys[np.argmin(ys)])))
        points.append((float(xs[np.argmax(ys)]), float(ys[np.argmax(ys)])))
        points.append((float(xs[np.argmin(xs)]), float(ys[np.argmin(xs)])))
        points.append((float(xs[np.argmax(xs)]), float(ys[np.argmax(xs)])))
        points.append((float(xs[center_idx]), float(ys[center_idx])))
    if bbox_prompt is not None:
        x1, y1, x2, y2 = [float(v) for v in bbox_prompt]
        bbox_candidates = [((x1 + x2) * 0.5, y1 + 0.22 * (y2 - y1)), ((x1 + x2) * 0.5, y1 + 0.5 * (y2 - y1)), ((x1 + x2) * 0.5, y1 + 0.78 * (y2 - y1)), (x1 + 0.28 * (x2 - x1), y1 + 0.55 * (y2 - y1)), (x1 + 0.72 * (x2 - x1), y1 + 0.55 * (y2 - y1))]
        for x, y in bbox_candidates:
            px = int(np.clip(round(x), 0, w - 1))
            py = int(np.clip(round(y), 0, h - 1))
            if interior_bool[py, px]:
                points.append((float(px), float(py)))
    return _dedupe_points(points, w, h)

def _merge_positive_points(*, width: int, height: int, manual_points: List[Tuple[float, float]], auto_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return _dedupe_points(list(manual_points) + list(auto_points), width, height)

def _normalize_prompt_points(raw_value: Any, width: int, height: int) -> List[Tuple[float, float]]:
    if not isinstance(raw_value, list) or not raw_value:
        return []
    points: List[Tuple[float, float]] = []
    for item in raw_value:
        if isinstance(item, list) and len(item) == 2:
            points.append((float(item[0]), float(item[1])))
    return _dedupe_points(points, width, height)

def _sample_negative_points(*, bbox_prompt: Optional[np.ndarray], width: int, height: int) -> List[Tuple[float, float]]:
    if bbox_prompt is None:
        return []
    x1, y1, x2, y2 = [float(v) for v in bbox_prompt]
    pad_x = max(10.0, 0.1 * (x2 - x1))
    pad_y = max(10.0, 0.1 * (y2 - y1))
    candidates = [(x1 - pad_x, y1 - pad_y), ((x1 + x2) * 0.5, y1 - pad_y), (x2 + pad_x, y1 - pad_y), (x1 - pad_x, (y1 + y2) * 0.5), (x2 + pad_x, (y1 + y2) * 0.5), (x1 - pad_x, y2 + pad_y), ((x1 + x2) * 0.5, y2 + pad_y), (x2 + pad_x, y2 + pad_y)]
    return _dedupe_points(candidates, width, height)

def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = int(np.sum(np.logical_and(mask_a, mask_b)))
    union = int(np.sum(np.logical_or(mask_a, mask_b)))
    if union <= 0:
        return 0.0
    return float(inter / union)

def _mask_precision(pred_mask: np.ndarray, ref_mask: np.ndarray) -> float:
    pred_area = int(np.sum(pred_mask))
    if pred_area <= 0:
        return 0.0
    inter = int(np.sum(np.logical_and(pred_mask, ref_mask)))
    return float(inter / pred_area)

def _select_best_mask(masks: np.ndarray, iou_scores: np.ndarray, *, positive_points: List[Tuple[float, float]], negative_points: List[Tuple[float, float]], mesh_mask: np.ndarray, bbox_prompt: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
    if masks.ndim != 3 or masks.shape[0] == 0:
        empty = np.zeros_like(mesh_mask, dtype=bool)
        return (empty, {'reason': 'no_sam_mask'})
    mesh_bool = mesh_mask > 0
    h, w = mesh_bool.shape
    bbox_mask = np.zeros((h, w), dtype=bool)
    if bbox_prompt is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox_prompt]
        bbox_mask[max(0, y1):min(h, y2 + 1), max(0, x1):min(w, x2 + 1)] = True
    mesh_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    mesh_support = cv2.dilate(mesh_bool.astype(np.uint8) * 255, mesh_dilate_kernel, iterations=1) > 0
    best_idx = 0
    best_score = -1000000000.0
    best_mask = None
    best_meta: Dict[str, Any] = {}
    for idx in range(masks.shape[0]):
        m = masks[idx] > 0
        if int(np.sum(m)) == 0:
            continue
        positive_hits = 0.0
        if positive_points:
            hit_count = 0
            for x, y in positive_points:
                px = int(np.clip(round(x), 0, w - 1))
                py = int(np.clip(round(y), 0, h - 1))
                hit_count += 1 if bool(m[py, px]) else 0
            positive_hits = float(hit_count) / float(len(positive_points))
        negative_clean = 1.0
        if negative_points:
            bad_count = 0
            for x, y in negative_points:
                px = int(np.clip(round(x), 0, w - 1))
                py = int(np.clip(round(y), 0, h - 1))
                bad_count += 1 if bool(m[py, px]) else 0
            negative_clean = 1.0 - float(bad_count) / float(len(negative_points))
        overlap_iou = _iou(m, mesh_bool)
        overlap_recall = float(np.sum(np.logical_and(m, mesh_bool))) / max(float(np.sum(mesh_bool)), 1.0)
        mesh_precision = _mask_precision(m, mesh_support)
        bbox_precision = 1.0
        bbox_outside_ratio = 0.0
        if bool(np.any(bbox_mask)):
            bbox_precision = _mask_precision(m, bbox_mask)
            bbox_outside_ratio = 1.0 - bbox_precision
        pred_area = max(float(np.sum(m)), 1.0)
        bbox_area = max(float(np.sum(bbox_mask)), 1.0) if bool(np.any(bbox_mask)) else pred_area
        area_ratio_to_bbox = pred_area / bbox_area
        oversize_penalty = max(0.0, area_ratio_to_bbox - 1.18)
        sam_score = float(iou_scores[idx])
        low_sam_penalty = 6.0 * max(0.0, 0.05 - sam_score)
        tiny_sam_penalty = 18.0 * max(0.0, 0.01 - sam_score)
        score = 3.0 * positive_hits + 1.5 * negative_clean + 2.5 * overlap_iou + 1.5 * overlap_recall + 1.8 * mesh_precision + 1.2 * bbox_precision + 2.5 * sam_score - 2.0 * bbox_outside_ratio - 1.8 * max(0.0, 0.55 - mesh_precision) - 1.5 * oversize_penalty - low_sam_penalty - tiny_sam_penalty
        if score > best_score:
            best_score = score
            best_idx = idx
            best_mask = m
            best_meta = {'candidate_idx': int(idx), 'sam_iou_score': sam_score, 'positive_hits': float(positive_hits), 'negative_clean': float(negative_clean), 'overlap_iou': float(overlap_iou), 'overlap_recall': float(overlap_recall), 'mesh_precision': float(mesh_precision), 'bbox_precision': float(bbox_precision), 'bbox_outside_ratio': float(bbox_outside_ratio), 'area_ratio_to_bbox': float(area_ratio_to_bbox), 'oversize_penalty': float(oversize_penalty), 'low_sam_penalty': float(low_sam_penalty), 'tiny_sam_penalty': float(tiny_sam_penalty), 'combined_score': float(score)}
    if best_mask is None:
        empty = np.zeros_like(mesh_bool, dtype=bool)
        return (empty, {'reason': 'all_empty'})
    return (best_mask.astype(bool), best_meta | {'selected_idx': int(best_idx)})

def _fill_mask_holes(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    mask_u8 = mask.astype(np.uint8) * 255
    h, w = mask_u8.shape
    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood) > 0
    filled = np.logical_or(mask, holes)
    added = int(np.sum(np.logical_and(filled, np.logical_not(mask))))
    return (filled.astype(bool), added)

def _largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(areas)) + 1
    return labels == best_label

def _component_containing_point(mask: np.ndarray, point_xy: Tuple[float, float]) -> Optional[np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return None
    px = int(np.clip(round(point_xy[0]), 0, mask.shape[1] - 1))
    py = int(np.clip(round(point_xy[1]), 0, mask.shape[0] - 1))
    label = int(labels[py, px])
    if label <= 0:
        return None
    return labels == label

def _merge_nearby_components(mask: np.ndarray, *, anchor_mask: np.ndarray, bbox_prompt: Optional[np.ndarray], mesh_mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return (mask.astype(bool), {'kept_components': 1})
    anchor_u8 = anchor_mask.astype(np.uint8)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    anchor_dilated = cv2.dilate(anchor_u8, dilate_kernel, iterations=1) > 0
    mesh_dilated = cv2.dilate(mesh_mask.astype(np.uint8) * 255, dilate_kernel, iterations=1) > 0
    bbox_mask = np.zeros_like(mask, dtype=bool)
    if bbox_prompt is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox_prompt]
        bbox_mask[max(0, y1):min(mask.shape[0], y2 + 1), max(0, x1):min(mask.shape[1], x2 + 1)] = True
    anchor_area = max(int(np.sum(anchor_mask)), 1)
    out = np.zeros_like(mask, dtype=bool)
    kept = 0
    for label in range(1, num_labels):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        if np.array_equal(component, anchor_mask):
            out |= component
            kept += 1
            continue
        overlap_anchor = bool(np.any(np.logical_and(component, anchor_dilated)))
        bbox_precision = _mask_precision(component, bbox_mask) if np.any(bbox_mask) else 1.0
        mesh_precision = _mask_precision(component, mesh_dilated)
        relative_area = float(area) / float(anchor_area)
        if overlap_anchor and bbox_precision >= 0.82 and (mesh_precision >= 0.3) and (relative_area <= 0.35):
            out |= component
            kept += 1
    if kept <= 0:
        return (anchor_mask.astype(bool), {'kept_components': 1, 'fallback_anchor_only': True})
    return (out.astype(bool), {'kept_components': int(kept)})

def _postprocess_sam_silhouette(mask: np.ndarray, *, prompt_xy: Optional[Tuple[float, float]], bbox_prompt: Optional[np.ndarray], mesh_mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    current = mask.astype(bool)
    if int(np.sum(current)) <= 0:
        return (current, {'postprocess': 'empty'})
    if prompt_xy is not None:
        anchor_component = _component_containing_point(current, prompt_xy)
        if anchor_component is None:
            anchor_component = _largest_component(current)
    else:
        anchor_component = _largest_component(current)
    filled, added_pixels = _fill_mask_holes(anchor_component.astype(bool))
    return (filled, {'postprocess': 'anchor_component_fill_holes_only', 'hole_filled_pixels': int(added_pixels), 'kept_components': 1})

def _run_predict(predictor: Any, *, positive_points: List[Tuple[float, float]], negative_points: List[Tuple[float, float]], box_prompt: Optional[np.ndarray], multimask_output: bool, mask_input: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = list(positive_points) + list(negative_points)
    labels = [1] * len(positive_points) + [0] * len(negative_points)
    if points:
        point_coords = np.asarray(points, dtype=np.float32)
        point_labels = np.asarray(labels, dtype=np.int32)
    else:
        point_coords = None
        point_labels = None
    masks, iou_scores, low_res_logits = predictor.predict(point_coords=point_coords, point_labels=point_labels, box=box_prompt, mask_input=mask_input, multimask_output=multimask_output, return_logits=True, normalize_coords=True)
    return (masks, iou_scores, low_res_logits)

def run_sam2_mask(*, config_path: Path, mesh_summary_yaml: Optional[Path]=None, output_dir: Optional[Path]=None, sam2_repo: Optional[Path]=None, sam2_checkpoint: Optional[Path]=None, sam2_device: str='auto', near_plane_m: float=0.1, run_mesh_if_missing: bool=False, save_json: bool=False, verbose: bool=True) -> Tuple[Dict[str, Any], Path]:
    config_path = _as_path(config_path)
    config = _load_yaml(config_path)
    sam2_repo = _sam2_repo_from_config(config) if sam2_repo is None else _as_path(sam2_repo)
    sam2_checkpoint = _sam2_checkpoint_from_config(config) if sam2_checkpoint is None else _as_path(sam2_checkpoint)
    if not sam2_checkpoint.exists():
        raise FileNotFoundError(f'SAM2 checkpoint not found: {sam2_checkpoint}')
    mesh_summary, mesh_summary_path = _load_mesh_summary_or_build(config_path=config_path, mesh_summary_yaml=mesh_summary_yaml, run_mesh_if_missing=run_mesh_if_missing, near_plane_m=near_plane_m, device=sam2_device)
    if output_dir is None:
        output_dir = mesh_summary_path.parent / 'sam2'
    output_dir = _as_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, build_sam2, SAM2ImagePredictor, repo_root = _import_sam2(sam2_repo)
    if sam2_device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(str(sam2_device))
    config_name = _config_name_from_checkpoint(sam2_checkpoint)
    sam_model = build_sam2(config_file=config_name, ckpt_path=str(sam2_checkpoint), device=str(device))
    predictor = SAM2ImagePredictor(sam_model)
    records = mesh_summary.get('records', [])
    if not isinstance(records, list) or not records:
        raise RuntimeError('mesh summary has no records')
    results: List[Dict[str, Any]] = []
    processed = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        image_path = Path(str(record.get('image_path', ''))).expanduser()
        raw_path = Path(str(record.get('raw_camouflage_path', ''))).expanduser()
        mesh_mask_path = Path(str(record.get('mesh_mask_path', ''))).expanduser()
        if not image_path.exists() or not raw_path.exists() or (not mesh_mask_path.exists()):
            continue
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        raw_bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        mesh_mask_u8 = cv2.imread(str(mesh_mask_path), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or raw_bgr is None or mesh_mask_u8 is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        mesh_mask = mesh_mask_u8 > 0
        height, width = image_rgb.shape[:2]
        gt_bbox = _normalize_bbox_xyxy(record.get('gt_bbox_xyxy', None) if isinstance(record.get('gt_bbox_xyxy', None), list) else None, width, height)
        mesh_bbox = _normalize_bbox_xyxy(record.get('mesh_bbox_xyxy', None) if isinstance(record.get('mesh_bbox_xyxy', None), list) else None, width, height)
        manual_prompt_points = _normalize_prompt_points(record.get('target_prompt_points_xy', []), width, height)
        manual_negative_points = _normalize_prompt_points(record.get('target_negative_prompt_points_xy', []), width, height)
        explicit_prompt_points = bool(record.get('target_prompt_points_explicit', False))
        record_prompt_xy = record.get('target_center_xy', None)
        if manual_prompt_points:
            prompt_xy = manual_prompt_points[0]
        elif isinstance(record_prompt_xy, list) and len(record_prompt_xy) == 2:
            prompt_xy = (float(record_prompt_xy[0]), float(record_prompt_xy[1]))
        else:
            prompt_xy = _safe_center_from_bbox(gt_bbox.tolist(), width, height) if gt_bbox is not None else None
        if prompt_xy is None and mesh_bbox is not None:
            prompt_xy = _safe_center_from_bbox(mesh_bbox.tolist(), width, height)
        if explicit_prompt_points:
            box_prompt = None
            positive_points = list(manual_prompt_points)
            negative_points = list(manual_negative_points)
            if not positive_points:
                continue
        else:
            if gt_bbox is not None:
                box_prompt = gt_bbox
            else:
                box_prompt = mesh_bbox
            box_prompt = _expand_bbox(box_prompt, width, height, ratio=0.04)
            auto_positive_points = _sample_mesh_positive_points(mesh_mask, prompt_xy=prompt_xy, bbox_prompt=box_prompt)
            positive_points = _merge_positive_points(width=width, height=height, manual_points=manual_prompt_points, auto_points=auto_positive_points)
            negative_points = _sample_negative_points(bbox_prompt=box_prompt, width=width, height=height)
        predictor.set_image(image_rgb)
        if device.type == 'cuda':
            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        else:
            autocast_ctx = nullcontext()
        with torch.inference_mode(), autocast_ctx:
            masks, iou_scores, low_res_logits = _run_predict(predictor, positive_points=positive_points, negative_points=negative_points, box_prompt=box_prompt, multimask_output=True)
        if not isinstance(masks, np.ndarray) or masks.ndim != 3 or masks.shape[0] == 0:
            continue
        best_mask, best_meta = _select_best_mask(masks, iou_scores, positive_points=positive_points, negative_points=negative_points, mesh_mask=mesh_mask, bbox_prompt=box_prompt)
        selected_idx = int(best_meta.get('selected_idx', -1))
        if isinstance(low_res_logits, np.ndarray) and low_res_logits.ndim == 3 and (0 <= selected_idx < low_res_logits.shape[0]):
            with torch.inference_mode(), autocast_ctx:
                refined_masks, refined_iou_scores, _ = _run_predict(predictor, positive_points=positive_points, negative_points=negative_points, box_prompt=box_prompt, mask_input=low_res_logits[selected_idx:selected_idx + 1], multimask_output=False)
            if isinstance(refined_masks, np.ndarray) and refined_masks.ndim == 3 and (refined_masks.shape[0] > 0):
                refined_mask, refined_meta = _select_best_mask(refined_masks, refined_iou_scores, positive_points=positive_points, negative_points=negative_points, mesh_mask=mesh_mask, bbox_prompt=box_prompt)
                best_mask = refined_mask
                best_meta = {**best_meta, 'refined': True, 'refined_meta': refined_meta}
        clipped_mask, post_meta = _postprocess_sam_silhouette(best_mask, prompt_xy=prompt_xy, bbox_prompt=box_prompt, mesh_mask=mesh_mask)
        best_meta = {**best_meta, 'raw_sam_only': False, 'postprocess_meta': post_meta}
        final_rgb = image_rgb.copy()
        final_rgb[clipped_mask] = raw_rgb[clipped_mask]
        stem = Path(str(record.get('raw_camouflage_path', 'record.png'))).stem
        sam_mask_path = output_dir / 'sam_masks' / f'{stem}.png'
        clipped_mask_path = output_dir / 'clipped_masks' / f'{stem}.png'
        final_path = output_dir / 'clipped_camouflage' / f'{stem}.png'
        panel_path = output_dir / 'panels' / f'{stem}.png'
        sam_mask_path.parent.mkdir(parents=True, exist_ok=True)
        clipped_mask_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sam_mask_path), best_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(clipped_mask_path), clipped_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(final_path), cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR))
        mask_vis = np.zeros((height, width, 3), dtype=np.uint8)
        mask_vis[best_mask] = np.asarray([255, 255, 255], dtype=np.uint8)
        row1 = np.concatenate([image_rgb, raw_rgb], axis=1)
        row2 = np.concatenate([mask_vis, final_rgb], axis=1)
        panel = np.concatenate([row1, row2], axis=0)
        cv2.putText(panel, f"frame={record.get('frame_id', -1)} {record.get('channel', '')}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(panel_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        out_record = {'sequence_name': str(record.get('sequence_name', '') or ''), 'frame_id': int(record.get('frame_id', -1)), 'channel': str(record.get('channel', '')), 'sample_token': str(record.get('sample_token', '')), 'ann_token': str(record.get('ann_token', '')), 'mask_source': 'sam2_instance_mask', 'bbox_used_only_as_prompt': True, 'image_path': str(image_path), 'raw_camouflage_path': str(raw_path), 'mesh_mask_path': str(mesh_mask_path), 'sam_mask_path': str(sam_mask_path), 'clipped_mask_path': str(clipped_mask_path), 'final_path': str(final_path), 'panel_path': str(panel_path), 'prompt_xy': [float(prompt_xy[0]), float(prompt_xy[1])] if prompt_xy is not None else None, 'prompt_points_xy': [[float(x), float(y)] for x, y in positive_points], 'negative_prompt_points_xy': [[float(x), float(y)] for x, y in negative_points], 'box_prompt_xyxy': box_prompt.tolist() if box_prompt is not None else None, 'explicit_prompt_points': explicit_prompt_points, 'positive_point_count': int(len(positive_points)), 'negative_point_count': int(len(negative_points)), 'sam_meta': best_meta}
        results.append(out_record)
        processed += 1
    summary = {'config_path': str(config_path), 'mesh_summary_yaml': str(mesh_summary_path), 'sam2_repo': str(repo_root), 'sam2_checkpoint': str(sam2_checkpoint), 'sam2_config_name': config_name, 'device': str(device), 'processed_view_count': processed, 'records': results}
    summary_path = output_dir / 'sam2_mask_summary.yaml'
    _save_yaml(summary_path, summary)
    if save_json:
        _save_json(output_dir / 'sam2_mask_summary.json', summary)
    if verbose:
        print(f'[sam2_mask] processed_views={processed} output={summary_path}')
    summary_builtin = _to_builtin(summary)
    del predictor
    del sam_model
    gc.collect()
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (summary_builtin, summary_path)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='SAM2 segmentation for target car and clip camouflage outside target mask')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH, help='Path to config.yaml')
    parser.add_argument('--mesh-summary-yaml', type=Path, default=None, help='mesh_projection_summary.yaml')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    parser.add_argument('--sam2-repo', type=Path, default=None, help='Local SAM2 repo; defaults to sam2.repo_root in config.yaml')
    parser.add_argument('--sam2-checkpoint', type=Path, default=None, help='SAM2 checkpoint; defaults to sam2.checkpoint_path in config.yaml')
    parser.add_argument('--device', type=str, default='auto', help='Device for SAM2: auto/cpu/cuda')
    parser.add_argument('--near-plane', type=float, default=0.1, help='Near plane for optional mesh build')
    parser.add_argument('--run-mesh-if-missing', action='store_true', help='If mesh summary missing, run nusc_gt_to_mesh')
    parser.add_argument('--save-json', action='store_true', help='Also save JSON summary')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    _, summary_path = run_sam2_mask(config_path=args.config, mesh_summary_yaml=args.mesh_summary_yaml, output_dir=args.output_dir, sam2_repo=args.sam2_repo, sam2_checkpoint=args.sam2_checkpoint, sam2_device=args.device, near_plane_m=float(args.near_plane), run_mesh_if_missing=bool(args.run_mesh_if_missing), save_json=bool(args.save_json), verbose=True)
    print(summary_path)
if __name__ == '__main__':
    main()
