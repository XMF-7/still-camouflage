from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F

@dataclass
class FrameLossInput:
    frame_id: int
    pred_center_ego: torch.Tensor
    gt_center_ego: torch.Tensor
    pred_size_wlh: torch.Tensor
    gt_size_wlh: torch.Tensor
    pred_yaw: torch.Tensor
    gt_yaw: torch.Tensor
    pred_class_logits: torch.Tensor
    target_logit: torch.Tensor
    target_label: int
    nearby_target_logits: Optional[torch.Tensor] = None
    other_target_logits: Optional[torch.Tensor] = None
    other_query_max_logits: Optional[torch.Tensor] = None
    query_feature: Optional[torch.Tensor] = None
    clean_query_feature: Optional[torch.Tensor] = None
    ref_center_ego: Optional[torch.Tensor] = None
    ref_size_wlh: Optional[torch.Tensor] = None
    ref_yaw: Optional[torch.Tensor] = None

def _reference_center_ego(item: FrameLossInput) -> torch.Tensor:
    return item.ref_center_ego if item.ref_center_ego is not None else item.gt_center_ego

def _reference_size_wlh(item: FrameLossInput) -> torch.Tensor:
    return item.ref_size_wlh if item.ref_size_wlh is not None else item.gt_size_wlh

def _reference_yaw(item: FrameLossInput) -> torch.Tensor:
    return item.ref_yaw if item.ref_yaw is not None else item.gt_yaw

def _distance_loss(pred: torch.Tensor, target: torch.Tensor, *, loss_type: str) -> torch.Tensor:
    mode = str(loss_type).strip().lower()
    if mode in {'smoothl1', 'smooth_l1'}:
        return F.smooth_l1_loss(pred, target, reduction='mean')
    if mode in {'l2', 'mse'}:
        return F.mse_loss(pred, target, reduction='mean')
    if mode in {'l1', 'mae'}:
        return F.l1_loss(pred, target, reduction='mean')
    raise ValueError(f'Unsupported loss_type={loss_type!r}, expected smooth_l1 / l2 / l1')

def total_variation_loss(texture: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(texture[:, :, :, 1:] - texture[:, :, :, :-1]).mean()
    dy = torch.abs(texture[:, :, 1:, :] - texture[:, :, :-1, :]).mean()
    return dx + dy

def l2_anchor_loss(texture: torch.Tensor) -> torch.Tensor:
    return torch.mean((texture - 0.5) ** 2)

def brightness_loss(texture: torch.Tensor, *, target_brightness: float=0.4, loss_type: str='l2') -> torch.Tensor:
    if texture.ndim != 4:
        raise ValueError('texture must be [B, C, H, W]')
    if int(texture.shape[1]) != 3:
        raise ValueError('texture channel dim must be 3 for brightness loss')
    work_texture = texture.to(dtype=torch.float32)
    rgb_to_luma = work_texture.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    brightness_map = torch.sum(work_texture * rgb_to_luma, dim=1, keepdim=True)
    target = brightness_map.new_full(brightness_map.shape, float(target_brightness))
    loss = _distance_loss(brightness_map, target, loss_type=loss_type)
    return loss.to(dtype=texture.dtype)

def frequency_high_frequency_loss(texture: torch.Tensor, *, cutoff_ratio: float=0.25) -> torch.Tensor:
    if texture.ndim != 4:
        raise ValueError('texture must be [B, C, H, W]')
    work_texture = texture.to(dtype=torch.float32)
    height = int(work_texture.shape[-2])
    width = int(work_texture.shape[-1])
    if height <= 1 or width <= 1:
        return texture.new_zeros(())
    cutoff = float(max(0.0, min(1.0, cutoff_ratio)))
    if cutoff >= 1.0:
        return texture.new_zeros(())
    freq_y = torch.fft.fftfreq(height, d=1.0, device=work_texture.device, dtype=torch.float32)
    freq_x = torch.fft.fftfreq(width, d=1.0, device=work_texture.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(freq_y, freq_x, indexing='ij')
    radial = torch.sqrt(grid_x.square() + grid_y.square())
    radial_max = radial.max().clamp(min=torch.finfo(torch.float32).eps)
    radial_norm = radial / radial_max
    high_mask = (radial_norm >= cutoff).to(dtype=torch.float32)
    fft_map = torch.fft.fft2(work_texture, dim=(-2, -1), norm='ortho')
    power = fft_map.real.square() + fft_map.imag.square()
    masked_power = power * high_mask.view(1, 1, height, width)
    denom = high_mask.sum().clamp(min=1.0)
    loss = masked_power.sum() / (work_texture.shape[0] * work_texture.shape[1] * denom)
    return loss.to(dtype=texture.dtype)

def default_nuscenes_printable_palette(device: Optional[torch.device]=None, dtype: torch.dtype=torch.float32) -> torch.Tensor:
    colors_rgb_255 = [[28, 28, 28], [70, 70, 72], [118, 120, 124], [168, 170, 172], [214, 214, 210], [245, 244, 240], [54, 68, 92], [92, 98, 88], [128, 116, 98], [96, 72, 58]]
    return torch.tensor(colors_rgb_255, device=device, dtype=dtype) / 255.0

def non_printability_score_loss(texture: torch.Tensor, *, printable_colors: Optional[torch.Tensor]=None, eps: float=1e-06) -> torch.Tensor:
    if texture.ndim != 4:
        raise ValueError('texture must be [B, C, H, W]')
    if int(texture.shape[1]) != 3:
        raise ValueError('texture channel dim must be 3 for RGB NPS')
    work_texture = texture.to(dtype=torch.float32)
    if printable_colors is None:
        printable_colors = default_nuscenes_printable_palette(device=work_texture.device, dtype=torch.float32)
    else:
        printable_colors = printable_colors.to(device=work_texture.device, dtype=torch.float32)
    if printable_colors.ndim != 2 or int(printable_colors.shape[1]) != 3:
        raise ValueError('printable_colors must be [K, 3]')
    if int(printable_colors.shape[0]) <= 0:
        return texture.new_zeros(())
    pixels = work_texture.permute(0, 2, 3, 1).reshape(-1, 3)
    color_diffs = pixels.unsqueeze(1) - printable_colors.unsqueeze(0)
    color_dist = torch.sqrt(torch.sum(color_diffs.square(), dim=-1) + float(eps))
    nps = torch.prod(color_dist, dim=1).mean()
    return nps.to(dtype=texture.dtype)

def _wrap_angle_diff(angle_diff: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

def _radians_to_degrees(angle_rad: torch.Tensor) -> torch.Tensor:
    return angle_rad * angle_rad.new_tensor(180.0 / math.pi)

def lateral_difference_sequence(frame_inputs: Sequence[FrameLossInput]) -> List[torch.Tensor]:
    values: List[torch.Tensor] = []
    for item in frame_inputs:
        gt_y = _reference_center_ego(item)[1:2]
        pred_y = item.pred_center_ego[1:2]
        direction_to_path = torch.where(gt_y >= 0.0, -torch.ones_like(gt_y), torch.ones_like(gt_y))
        lateral_shift = pred_y - gt_y
        values.append((direction_to_path * lateral_shift).reshape(()))
    return values

def forward_difference_sequence(frame_inputs: Sequence[FrameLossInput]) -> List[torch.Tensor]:
    values: List[torch.Tensor] = []
    for item in frame_inputs:
        gt_x = _reference_center_ego(item)[0:1]
        pred_x = item.pred_center_ego[0:1]
        values.append((pred_x - gt_x).reshape(()))
    return values

def longitudinal_difference_sequence(frame_inputs: Sequence[FrameLossInput]) -> List[torch.Tensor]:
    values: List[torch.Tensor] = []
    for item in frame_inputs:
        gt_x = _reference_center_ego(item)[0:1]
        pred_x = item.pred_center_ego[0:1]
        longitudinal_shift = pred_x - gt_x
        values.append(longitudinal_shift.abs().reshape(()))
    return values

def ego_plane_direction_unit_xy(*, front_back_pct: float, left_right_pct: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Unit direction in ego (nuScenes lidar): +x forward, +y left. Percentages are arbitrary weights and are normalized."""
    raw = torch.tensor([float(front_back_pct), float(left_right_pct)], device=device, dtype=torch.float32)
    n = torch.linalg.norm(raw)
    if float(n.item()) < 1e-08:
        raw = torch.tensor([1.0, 0.0], device=device, dtype=torch.float32)
    else:
        raw = raw / n
    return raw.to(dtype=dtype)

def directed_difference_sequence(frame_inputs: Sequence[FrameLossInput], *, direction_unit_xy: torch.Tensor) -> List[torch.Tensor]:
    u = direction_unit_xy.flatten()[:2].to(dtype=torch.float32)
    u = u / torch.linalg.norm(u).clamp(min=torch.finfo(u.dtype).eps)
    values: List[torch.Tensor] = []
    for item in frame_inputs:
        ref = _reference_center_ego(item)[:2].to(dtype=torch.float32)
        pred = item.pred_center_ego[:2].to(dtype=torch.float32)
        delta = pred - ref
        along = (delta * u.to(device=delta.device)).sum()
        values.append(along.reshape(()))
    return values

def directed_move_loss(frame_inputs: Sequence[FrameLossInput], *, loss_type: str, front_back_pct: float, left_right_pct: float) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    if not frame_inputs:
        zero = torch.zeros(())
        return (zero, zero, {'difference_mean': 0.0, 'longitudinal_difference_mean': 0.0, 'longitudinal_difference_max': 0.0, 'loss_move_longitudinal_mean': 0.0, 'loss_move_longitudinal_max': 0.0})
    ref0 = _reference_center_ego(frame_inputs[0])
    u = ego_plane_direction_unit_xy(front_back_pct=front_back_pct, left_right_pct=left_right_pct, device=ref0.device, dtype=torch.float32)
    along_terms: List[torch.Tensor] = []
    along_vals: List[torch.Tensor] = []
    for item in frame_inputs:
        ref = _reference_center_ego(item)[:2].to(dtype=torch.float32)
        pred = item.pred_center_ego[:2].to(dtype=torch.float32)
        delta = pred - ref
        uu = u.to(device=delta.device, dtype=torch.float32)
        along = (delta * uu).sum()
        along_vals.append(along.reshape(()))
        shortfall = torch.relu(-along).reshape(1)
        along_terms.append(_distance_loss(shortfall, shortfall.new_zeros((1,)), loss_type=loss_type))
    along_mean = torch.stack(along_terms).mean()
    zero_side = along_mean.new_zeros(())
    stats = {'difference_mean': float(torch.stack(along_vals).mean().detach().item()), 'longitudinal_difference_mean': 0.0, 'longitudinal_difference_max': 0.0, 'loss_move_longitudinal_mean': 0.0, 'loss_move_longitudinal_max': 0.0}
    return (along_mean, zero_side, stats)

def progress_loss(progress_values: Sequence[torch.Tensor], *, step_size_m: float, decay_lambda: float, detach_previous: bool=False, loss_type: str='l2') -> Tuple[torch.Tensor, Dict[str, float], List[Optional[float]]]:
    if len(progress_values) <= 1:
        zero = progress_values[0].new_zeros(()) if progress_values else torch.zeros(())
        return (zero, {'progress_teacher': 0.0, 'progress_floor': 0.0, 'progress_step_target_m': float(step_size_m), 'progress_step_err_mean': 0.0}, [])
    s = float(step_size_m)
    losses: List[torch.Tensor] = []
    errs: List[torch.Tensor] = []
    gains: List[Optional[float]] = []
    for pair_index in range(len(progress_values) - 1):
        progress_prev = progress_values[pair_index]
        progress_curr = progress_values[pair_index + 1]
        progress_prev_for_gain = progress_prev.detach() if bool(detach_previous) else progress_prev
        gain = progress_curr - progress_prev_for_gain
        err = gain - gain.new_tensor(s)
        zero = gain.new_zeros((1,))
        term_err = err.reshape(1)
        if str(loss_type).strip().lower() in {'l2', 'mse'}:
            core = term_err.square()
        else:
            core = _distance_loss(term_err, zero, loss_type=str(loss_type))
        decay = progress_prev.new_tensor(float(decay_lambda) ** pair_index)
        losses.append(decay * core.reshape(()))
        errs.append(err.reshape(()))
        gains.append(float(gain.detach().item()))
    total = torch.stack(losses).mean()
    err_mean = float(torch.stack(errs).mean().detach().item()) if errs else 0.0
    stats = {'progress_teacher': float(total.detach().item()), 'progress_floor': 0.0, 'progress_step_target_m': float(s), 'progress_step_err_mean': err_mean}
    return (total, stats, gains)

def move_loss(frame_inputs: Sequence[FrameLossInput], *, loss_type: str) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    if not frame_inputs:
        zero = torch.zeros(())
        return (zero, zero, {'difference_mean': 0.0, 'longitudinal_difference_mean': 0.0, 'longitudinal_difference_max': 0.0, 'loss_move_longitudinal_mean': 0.0, 'loss_move_longitudinal_max': 0.0})
    lateral_terms: List[torch.Tensor] = []
    longitudinal_terms: List[torch.Tensor] = []
    lateral_diffs: List[torch.Tensor] = []
    longitudinal_diffs: List[torch.Tensor] = []
    for item in frame_inputs:
        ref_center = _reference_center_ego(item)
        gt_y = ref_center[1:2]
        pred_y = item.pred_center_ego[1:2]
        lateral_target_shift = -gt_y
        pred_lateral_shift = pred_y - gt_y
        lateral_terms.append(_distance_loss(pred_lateral_shift, lateral_target_shift, loss_type=loss_type))
        gt_x = ref_center[0:1]
        pred_x = item.pred_center_ego[0:1]
        pred_longitudinal_shift = pred_x - gt_x
        longitudinal_terms.append(_distance_loss(pred_longitudinal_shift, torch.zeros_like(pred_longitudinal_shift), loss_type=loss_type))
        direction_to_path = torch.where(gt_y >= 0.0, -torch.ones_like(gt_y), torch.ones_like(gt_y))
        lateral_diffs.append((direction_to_path * pred_lateral_shift).reshape(()))
        longitudinal_diffs.append(pred_longitudinal_shift.abs().reshape(()))
    lateral_mean = torch.stack(lateral_terms).mean()
    longitudinal_term_tensor = torch.stack(longitudinal_terms)
    longitudinal_mean = longitudinal_term_tensor.mean()
    longitudinal_max = longitudinal_term_tensor.max()
    longitudinal_combined = longitudinal_mean + longitudinal_max
    longitudinal_diff_tensor = torch.stack(longitudinal_diffs)
    stats = {'difference_mean': float(torch.stack(lateral_diffs).mean().detach().item()), 'longitudinal_difference_mean': float(longitudinal_diff_tensor.mean().detach().item()), 'longitudinal_difference_max': float(longitudinal_diff_tensor.max().detach().item()), 'loss_move_longitudinal_mean': float(longitudinal_mean.detach().item()), 'loss_move_longitudinal_max': float(longitudinal_max.detach().item())}
    return (lateral_mean, longitudinal_combined, stats)

def forward_move_loss(frame_inputs: Sequence[FrameLossInput], *, loss_type: str) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    del loss_type
    if not frame_inputs:
        zero = torch.zeros(())
        return (zero, zero, {'difference_mean': 0.0, 'longitudinal_difference_mean': 0.0, 'longitudinal_difference_max': 0.0, 'loss_move_longitudinal_mean': 0.0, 'loss_move_longitudinal_max': 0.0})
    move_terms: List[torch.Tensor] = []
    side_keep_terms: List[torch.Tensor] = []
    forward_diffs: List[torch.Tensor] = []
    side_diffs: List[torch.Tensor] = []
    for item in frame_inputs:
        ref_center = _reference_center_ego(item)
        gt_x = ref_center[0:1]
        pred_x = item.pred_center_ego[0:1]
        forward_shift = pred_x - gt_x
        move_terms.append((-forward_shift).reshape(()))
        forward_diffs.append(forward_shift.reshape(()))
        gt_y = ref_center[1:2]
        pred_y = item.pred_center_ego[1:2]
        side_shift = pred_y - gt_y
        side_keep_terms.append(side_shift.square().mean())
        side_diffs.append(side_shift.abs().reshape(()))
    move_mean = torch.stack(move_terms).mean()
    side_keep_tensor = torch.stack(side_keep_terms)
    side_keep_mean = side_keep_tensor.mean()
    side_keep_max = side_keep_tensor.max()
    side_keep_combined = side_keep_mean + side_keep_max
    side_diff_tensor = torch.stack(side_diffs)
    stats = {'difference_mean': float(torch.stack(forward_diffs).mean().detach().item()), 'longitudinal_difference_mean': float(side_diff_tensor.mean().detach().item()), 'longitudinal_difference_max': float(side_diff_tensor.max().detach().item()), 'loss_move_longitudinal_mean': float(side_keep_mean.detach().item()), 'loss_move_longitudinal_max': float(side_keep_max.detach().item())}
    return (move_mean, side_keep_combined, stats)

def first_frame_min_loss(difference: torch.Tensor, *, min_m: float) -> torch.Tensor:
    target_min = difference.new_tensor(float(min_m))
    shortfall = torch.relu(target_min - difference)
    return shortfall.square()

def rigid_loss(frame_inputs: Sequence[FrameLossInput], *, loss_type: str, size_weight: float, yaw_weight: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not frame_inputs:
        zero = torch.zeros(())
        return (zero, {'rigid_size_diff_mean': 0.0, 'rigid_yaw_diff_mean': 0.0, 'loss_rigid_size': 0.0, 'loss_rigid_yaw': 0.0, 'loss_rigid_inner': 0.0, 'rigid_loss_type': str(loss_type), 'rigid_size_weight': float(size_weight), 'rigid_yaw_weight': float(yaw_weight), 'rigid_yaw_unit': 'deg'})
    terms: List[torch.Tensor] = []
    size_diffs: List[torch.Tensor] = []
    yaw_diffs: List[torch.Tensor] = []
    size_terms: List[torch.Tensor] = []
    yaw_terms: List[torch.Tensor] = []
    for item in frame_inputs:
        pred_size = item.pred_size_wlh
        gt_size = _reference_size_wlh(item)
        loss_size = _distance_loss(pred_size, gt_size, loss_type=loss_type)
        yaw_error = _radians_to_degrees(_wrap_angle_diff(item.pred_yaw - _reference_yaw(item))).reshape(1)
        zero_yaw = torch.zeros_like(yaw_error)
        loss_yaw = _distance_loss(yaw_error, zero_yaw, loss_type=loss_type)
        inner = loss_size.new_tensor(float(size_weight)) * loss_size + loss_yaw.new_tensor(float(yaw_weight)) * loss_yaw
        terms.append(inner)
        size_diffs.append((pred_size - gt_size).abs().mean().reshape(()))
        yaw_diffs.append(yaw_error.abs().reshape(()))
        size_terms.append(loss_size.reshape(()))
        yaw_terms.append(loss_yaw.reshape(()))
    total = torch.stack(terms).mean()
    stats = {'rigid_size_diff_mean': float(torch.stack(size_diffs).mean().detach().item()), 'rigid_yaw_diff_mean': float(torch.stack(yaw_diffs).mean().detach().item()), 'loss_rigid_size': float(torch.stack(size_terms).mean().detach().item()), 'loss_rigid_yaw': float(torch.stack(yaw_terms).mean().detach().item()), 'loss_rigid_inner': float(total.detach().item()), 'rigid_loss_type': str(loss_type), 'rigid_size_weight': float(size_weight), 'rigid_yaw_weight': float(yaw_weight), 'rigid_yaw_unit': 'deg'}
    return (total, stats)

def cls_loss(frame_inputs: Sequence[FrameLossInput], *, pos_weight: float=1.0, neg_weight: float=1.0, rank_weight: float=1.0, rank_margin: float=0.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not frame_inputs:
        zero = torch.zeros(())
        return (zero, {'target_logit': 0.0, 'target_confidence': 0.0, 'loss_cls_pos': 0.0, 'loss_cls_neg': 0.0, 'loss_cls_rank': 0.0, 'loss_cls_nearby': 0.0, 'noncar_max_confidence': 0.0, 'nearby_target_max_confidence': 0.0, 'nearby_query_count': 0.0, 'cls_rank_margin': float(rank_margin), 'cls_pos_weight': float(pos_weight), 'cls_neg_weight': float(neg_weight), 'cls_rank_weight': float(rank_weight)})
    pos_terms: List[torch.Tensor] = []
    neg_terms: List[torch.Tensor] = []
    rank_terms: List[torch.Tensor] = []
    target_logits: List[torch.Tensor] = []
    noncar_max_confidences: List[torch.Tensor] = []
    for item in frame_inputs:
        pred_logits = item.pred_class_logits.reshape(-1)
        target_index = int(item.target_label)
        pred_target_logit = item.target_logit.reshape(1)
        loss_pos = F.binary_cross_entropy_with_logits(pred_target_logit, torch.ones_like(pred_target_logit))
        pos_terms.append(loss_pos)
        target_logits.append(item.target_logit.reshape(()))
        noncar_mask = torch.ones_like(pred_logits, dtype=torch.bool)
        if 0 <= target_index < int(pred_logits.numel()):
            noncar_mask[target_index] = False
        noncar_logits = pred_logits[noncar_mask]
        if int(noncar_logits.numel()) > 0:
            max_noncar_logit = noncar_logits.max().reshape(1)
            loss_neg = F.binary_cross_entropy_with_logits(max_noncar_logit, torch.zeros_like(max_noncar_logit))
            noncar_logsumexp = torch.logsumexp(noncar_logits, dim=0, keepdim=True)
            loss_rank = F.softplus(noncar_logsumexp - pred_target_logit + float(rank_margin))
            neg_terms.append(loss_neg)
            rank_terms.append(loss_rank)
            noncar_max_confidences.append(torch.sigmoid(noncar_logits).max().reshape(()))
        else:
            neg_terms.append(loss_pos.new_zeros(()))
            rank_terms.append(loss_pos.new_zeros(()))
            noncar_max_confidences.append(loss_pos.new_zeros(()))
    loss_pos_mean = torch.stack(pos_terms).mean()
    loss_neg_mean = torch.stack(neg_terms).mean()
    loss_rank_mean = torch.stack(rank_terms).mean()
    total = loss_pos_mean.new_tensor(float(pos_weight)) * loss_pos_mean + loss_neg_mean.new_tensor(float(neg_weight)) * loss_neg_mean + loss_rank_mean.new_tensor(float(rank_weight)) * loss_rank_mean
    target_logits_tensor = torch.stack(target_logits)
    stats = {'target_logit': float(target_logits_tensor.mean().detach().item()), 'target_confidence': float(torch.sigmoid(target_logits_tensor).mean().detach().item()), 'loss_cls_pos': float(loss_pos_mean.detach().item()), 'loss_cls_neg': float(loss_neg_mean.detach().item()), 'loss_cls_rank': float(loss_rank_mean.detach().item()), 'loss_cls_nearby': 0.0, 'noncar_max_confidence': float(torch.stack(noncar_max_confidences).mean().detach().item()), 'nearby_target_max_confidence': 0.0, 'nearby_query_count': 0.0, 'cls_rank_margin': float(rank_margin), 'cls_pos_weight': float(pos_weight), 'cls_neg_weight': float(neg_weight), 'cls_rank_weight': float(rank_weight)}
    return (total, stats)

def global_query_rank_loss(frame_inputs: Sequence[FrameLossInput], *, margin: float=0.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not frame_inputs:
        zero = torch.zeros(())
        return (zero, {'loss_global_query_rank': 0.0, 'global_other_target_max_confidence': 0.0})
    terms: List[torch.Tensor] = []
    other_max_confidences: List[torch.Tensor] = []
    for item in frame_inputs:
        pred_target_logit = item.target_logit.reshape(1)
        other_logits = item.other_target_logits
        if other_logits is not None:
            other_logits = other_logits.reshape(-1)
        if other_logits is None or int(other_logits.numel()) <= 0:
            terms.append(pred_target_logit.new_zeros(()))
            other_max_confidences.append(pred_target_logit.new_zeros(()))
            continue
        other_logsumexp = torch.logsumexp(other_logits, dim=0, keepdim=True)
        terms.append(F.softplus(other_logsumexp - pred_target_logit + float(margin)))
        other_max_confidences.append(torch.sigmoid(other_logits).max().reshape(()))
    loss_mean = torch.stack(terms).mean()
    stats = {'loss_global_query_rank': float(loss_mean.detach().item()), 'global_other_target_max_confidence': float(torch.stack(other_max_confidences).mean().detach().item())}
    return (loss_mean, stats)

def query_identity_loss(frame_inputs: Sequence[FrameLossInput], *, loss_type: str='cosine') -> Tuple[torch.Tensor, Dict[str, float]]:
    if not frame_inputs:
        zero = torch.zeros(())
        return (zero, {'loss_query_identity': 0.0, 'query_identity_cosine': 1.0})
    terms: List[torch.Tensor] = []
    cosines: List[torch.Tensor] = []
    for item in frame_inputs:
        query_feature = item.query_feature
        clean_query_feature = item.clean_query_feature
        if query_feature is None or clean_query_feature is None:
            continue
        q = query_feature.reshape(-1).to(dtype=torch.float32)
        qc = clean_query_feature.reshape(-1).to(device=q.device, dtype=torch.float32)
        cosine = F.cosine_similarity(q.unsqueeze(0), qc.unsqueeze(0), dim=1, eps=1e-06).reshape(())
        cosines.append(cosine)
        mode = str(loss_type).strip().lower()
        if mode in {'cos', 'cosine'}:
            term = 1.0 - cosine
        elif mode in {'l2', 'mse'}:
            term = F.mse_loss(q, qc, reduction='mean')
        elif mode in {'l1', 'mae'}:
            term = F.l1_loss(q, qc, reduction='mean')
        else:
            raise ValueError(f'Unsupported query identity loss_type={loss_type!r}; expected cosine/l2/l1')
        terms.append(term.to(dtype=query_feature.dtype))
    if not terms:
        ref = frame_inputs[0].target_logit if frame_inputs else torch.zeros(())
        zero = ref.new_zeros(())
        return (zero, {'loss_query_identity': 0.0, 'query_identity_cosine': 1.0})
    loss_mean = torch.stack(terms).mean()
    cosine_mean = torch.stack(cosines).mean() if cosines else loss_mean.new_tensor(1.0)
    stats = {'loss_query_identity': float(loss_mean.detach().item()), 'query_identity_cosine': float(cosine_mean.detach().item())}
    return (loss_mean, stats)

def style_loss(texture: torch.Tensor, *, printable_colors: Optional[torch.Tensor]=None, brightness_target: float=0.4, brightness_loss_type: str='l2') -> Tuple[torch.Tensor, Dict[str, float]]:
    tv_term = total_variation_loss(texture)
    l2_term = l2_anchor_loss(texture)
    brightness_term = brightness_loss(texture, target_brightness=brightness_target, loss_type=brightness_loss_type)
    nps_term = non_printability_score_loss(texture, printable_colors=printable_colors)
    total = tv_term + l2_term + brightness_term + nps_term
    return (total, {'style_tv': float(tv_term.detach().item()), 'style_l2': float(l2_term.detach().item()), 'style_brightness': float(brightness_term.detach().item()), 'style_freq': 0.0, 'style_nps': float(nps_term.detach().item())})
