from __future__ import annotations
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from .bevdet import BevDetGradientModel
from .bevformer import CAMERA_CHANNELS, FixedQueryMatch, FrameRecord, _as_path, _rotation_matrix_to_quaternion_wxyz
_PRESET_TO_MODULE = {'r50-2key': 'bevdepth.exps.nuscenes.mv.bev_depth_lss_r50_256x704_128x128_24e_2key'}
_PRESET_TO_CKPT = {'r50-2key': 'ckpt/bev_depth_lss_r50_256x704_128x128_24e_2key.pth'}
_PRESET_ALIASES = {'base': 'r50-2key', 'default': 'r50-2key', 'r50': 'r50-2key', '2key': 'r50-2key'}

def _voxel_pooling_train_torch(geom_xyz: torch.Tensor, input_features: torch.Tensor, voxel_num: torch.Tensor) -> torch.Tensor:
    if geom_xyz.ndim != 6 or input_features.ndim != 6:
        raise ValueError(f'Expected geom [B,N,D,H,W,3] and features [B,N,D,H,W,C], got {tuple(geom_xyz.shape)} / {tuple(input_features.shape)}')
    batch_size = int(geom_xyz.shape[0])
    channels = int(input_features.shape[-1])
    voxel_x = int(voxel_num[0].item())
    voxel_y = int(voxel_num[1].item())
    voxel_z = int(voxel_num[2].item())
    out = input_features.new_zeros((batch_size, voxel_y, voxel_x, channels))
    for batch_idx in range(batch_size):
        coords = geom_xyz[batch_idx].reshape(-1, 3).long()
        feats = input_features[batch_idx].reshape(-1, channels)
        keep = (coords[:, 0] >= 0) & (coords[:, 0] < voxel_x) & (coords[:, 1] >= 0) & (coords[:, 1] < voxel_y) & (coords[:, 2] >= 0) & (coords[:, 2] < voxel_z)
        if bool(keep.any()):
            xy_index = coords[keep, 1] * voxel_x + coords[keep, 0]
            flat_out = out[batch_idx].reshape(voxel_y * voxel_x, channels)
            flat_out.index_add_(0, xy_index, feats[keep])
    return out.permute(0, 3, 1, 2).contiguous()

def _voxel_pooling_inference_torch(geom_xyz: torch.Tensor, depth_features: torch.Tensor, context_features: torch.Tensor, voxel_num: torch.Tensor) -> torch.Tensor:
    batch_size, num_cams, num_depth, feat_h, feat_w = [int(v) for v in geom_xyz.shape[:5]]
    channels = int(context_features.shape[1])
    context = context_features.reshape(batch_size, num_cams, channels, feat_h, feat_w)
    input_features = depth_features.unsqueeze(-1) * context.permute(0, 1, 3, 4, 2).unsqueeze(2)
    input_features = input_features.reshape(batch_size, num_cams, num_depth, feat_h, feat_w, channels)
    return _voxel_pooling_train_torch(geom_xyz, input_features.contiguous(), voxel_num)

def _patch_bevdepth_voxel_pooling_fallback() -> None:
    try:
        import bevdepth.layers.backbones.base_lss_fpn as base_lss_fpn
    except Exception:
        return
    needs_train = not callable(getattr(base_lss_fpn, 'voxel_pooling_train', None))
    needs_infer = not callable(getattr(base_lss_fpn, 'voxel_pooling_inference', None))
    if needs_train:
        base_lss_fpn.voxel_pooling_train = _voxel_pooling_train_torch
    if needs_infer:
        base_lss_fpn.voxel_pooling_inference = _voxel_pooling_inference_torch

class BevDepthGradientModel(BevDetGradientModel):

    def __init__(self, *, bevdepth_cfg: Dict[str, Any], device: str='cuda', use_amp: bool=True, amp_dtype: str='fp16'):
        self.repo_root = _as_path(str(bevdepth_cfg.get('repo_root', '')))
        if not self.repo_root.exists():
            raise FileNotFoundError(f'BEVDepth repo_root not found: {self.repo_root}')
        raw_preset = str(bevdepth_cfg.get('preset', 'r50-2key')).strip().lower() or 'r50-2key'
        self.preset = _PRESET_ALIASES.get(raw_preset, raw_preset)
        if self.preset not in _PRESET_TO_MODULE:
            raise ValueError('bevdepth.preset only supports r50-2key')
        raw_checkpoint_path = str(bevdepth_cfg.get('checkpoint_path', '') or '').strip()
        self.config_module = str(bevdepth_cfg.get('config_module', '') or '').strip() or _PRESET_TO_MODULE[self.preset]
        self.checkpoint_path = _as_path(raw_checkpoint_path) if raw_checkpoint_path else (self.repo_root / _PRESET_TO_CKPT[self.preset]).resolve()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f'BEVDepth checkpoint not found: {self.checkpoint_path}')
        self.data_root = _as_path(str(bevdepth_cfg.get('data_root', ''))) if bevdepth_cfg.get('data_root') else None
        self.workers_per_gpu = int(bevdepth_cfg.get('workers_per_gpu', 0))
        self.key_camera_override = str(bevdepth_cfg.get('key_camera', '') or '').strip()
        self.history_fill = str(bevdepth_cfg.get('history_fill', 'repeat_oldest') or 'repeat_oldest').strip().lower()
        requested_device = str(device).strip().lower()
        self.device = torch.device('cuda' if requested_device == 'cuda' and torch.cuda.is_available() else 'cpu')
        if requested_device == 'cuda' and self.device.type != 'cuda':
            raise RuntimeError('Requested cuda for BEVDepth, but cuda is not available')
        self.use_amp = bool(use_amp) and self.device.type == 'cuda'
        amp_dtype_name = str(amp_dtype).strip().lower()
        self.amp_dtype = torch.float16 if amp_dtype_name in {'fp16', 'float16'} else torch.bfloat16
        self.img_mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32, device=self.device).view(3, 1, 1)
        self.img_std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32, device=self.device).view(3, 1, 1)
        self.model = None
        self.lightning_model = None
        self.lidar_box_cls = None
        self.bbox_coder = None
        self.denormalize_bbox_fn = None
        self.class_names: List[str] = []
        self.car_label = -1
        self.cfg = None
        self.data_config: Dict[str, Any] = {}
        self.point_cloud_range: List[float] = []
        self.voxel_size: List[float] = []
        self.out_size_factor = 1
        self.key_camera = 'CAM_FRONT_LEFT'
        self.grid_shape_hw: Tuple[int, int] = (0, 0)
        self.last_bbox_tensor: Optional[torch.Tensor] = None
        self.last_cls_tensor: Optional[torch.Tensor] = None
        self.last_bbox_grad: Optional[torch.Tensor] = None
        self.last_cls_grad: Optional[torch.Tensor] = None
        self.last_heatmap_tensor: Optional[torch.Tensor] = None
        self.last_reg_tensor: Optional[torch.Tensor] = None
        self.last_heatmap_grad: Optional[torch.Tensor] = None
        self.last_reg_grad: Optional[torch.Tensor] = None
        self._prev_frame_for_history: Optional[FrameRecord] = None
        self._prev_images_for_history: Optional[Dict[str, torch.Tensor]] = None

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type='cuda', dtype=self.amp_dtype)

    def build(self) -> None:
        repo_str = str(self.repo_root)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        from importlib import import_module
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        module = import_module(self.config_module)
        lightning_cls = getattr(module, 'BEVDepthLightningModel')
        kwargs: Dict[str, Any] = {}
        if self.data_root is not None:
            kwargs['data_root'] = str(self.data_root)
        lightning_model = lightning_cls(**kwargs)
        _patch_bevdepth_voxel_pooling_fallback()
        ckpt = torch.load(str(self.checkpoint_path), map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = lightning_model.load_state_dict(state_dict, strict=False)
        tolerated_missing = [name for name in missing if name.startswith(('evaluator.', 'model.backbone.frustum'))]
        hard_missing = [name for name in missing if name not in tolerated_missing]
        if hard_missing:
            raise RuntimeError(f'BEVDepth checkpoint missing unexpected keys, first keys={hard_missing[:8]}')
        if unexpected:
            raise RuntimeError(f'BEVDepth checkpoint has unexpected keys, first keys={list(unexpected)[:8]}')
        model = lightning_model.model
        model.pts_bbox_head = model.head
        model.to(self.device)
        model.eval()
        if hasattr(model, 'backbone'):
            model.backbone.training = True
        for param in model.parameters():
            param.requires_grad_(False)
        self.lightning_model = lightning_model
        self.model = model
        self.lidar_box_cls = LiDARInstance3DBoxes
        self.bbox_coder = model.head.bbox_coder
        self.cfg = module
        self.class_names = list(getattr(lightning_model, 'class_names', []))
        self.data_config = dict(getattr(lightning_model, 'ida_aug_conf', {}))
        self.data_config['input_size'] = tuple(self.data_config.get('final_dim', (256, 704)))
        self.point_cloud_range = [float(v) for v in model.head.train_cfg['point_cloud_range']]
        self.voxel_size = [float(v) for v in model.head.train_cfg['voxel_size']]
        self.out_size_factor = int(model.head.train_cfg['out_size_factor'])
        cams = list(self.data_config.get('cams', CAMERA_CHANNELS))
        self.key_camera = self.key_camera_override or (str(cams[0]) if cams else 'CAM_FRONT_LEFT')
        if 'car' in self.class_names:
            self.car_label = self.class_names.index('car')

    def _matrix_sensor2ego(self, frame: FrameRecord, channel: str) -> torch.Tensor:
        return self._camera_sensor2ego_matrix(frame, channel)

    def _matrix_ego2global(self, frame: FrameRecord, channel: str) -> torch.Tensor:
        return self._ego_matrix(frame, channel)

    def _history_frame_and_images(self, frame: FrameRecord, camera_images: Dict[str, torch.Tensor]) -> Tuple[FrameRecord, Dict[str, torch.Tensor]]:
        prev_frame = self._prev_frame_for_history
        prev_images = self._prev_images_for_history
        if prev_frame is not None and prev_images is not None and (prev_frame.scene_token == frame.scene_token):
            return (prev_frame, prev_images)
        detached = {channel: image.detach() for channel, image in camera_images.items()}
        return (frame, detached)

    def _bevdepth_sweep_mats(self, *, key_frame: FrameRecord, sweep_frame: FrameRecord, channel: str) -> Tuple[torch.Tensor, torch.Tensor]:
        disable_amp = torch.autocast(device_type='cuda', enabled=False) if self.device.type == 'cuda' else nullcontext()
        with disable_amp:
            sweepsensor2sweepego = self._matrix_sensor2ego(sweep_frame, channel).to(dtype=torch.float32)
            sweepego2global = self._matrix_ego2global(sweep_frame, channel).to(dtype=torch.float32)
            keyego2global = self._matrix_ego2global(key_frame, channel).to(dtype=torch.float32)
            global2keyego = torch.linalg.inv(keyego2global)
            keysensor2keyego = self._matrix_sensor2ego(key_frame, channel).to(dtype=torch.float32)
            keyego2keysensor = torch.linalg.inv(keysensor2keyego)
            keysensor2sweepsensor = torch.linalg.inv(keyego2keysensor @ global2keyego @ sweepego2global @ sweepsensor2sweepego)
            sweepsensor2keyego = global2keyego @ sweepego2global @ sweepsensor2sweepego
            return (sweepsensor2keyego, keysensor2sweepsensor)

    def prepare_model_inputs(self, frame: FrameRecord, camera_images: Dict[str, torch.Tensor], *, prev_scene_token: Optional[str], prev_abs_pos: Optional[np.ndarray], prev_abs_angle: Optional[float]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, List[Dict[str, Any]], np.ndarray, float]:
        del prev_scene_token, prev_abs_pos, prev_abs_angle
        cams = [str(cam) for cam in self.data_config.get('cams', CAMERA_CHANNELS)]
        history_frame, history_images = self._history_frame_and_images(frame, camera_images)
        sweep_frames = [frame, history_frame]
        sweep_image_maps = [camera_images, history_images]
        sweep_imgs: List[torch.Tensor] = []
        sweep_sensor2ego_mats: List[torch.Tensor] = []
        sweep_intrin_mats: List[torch.Tensor] = []
        sweep_ida_mats: List[torch.Tensor] = []
        sweep_sensor2sensor_mats: List[torch.Tensor] = []
        sweep_timestamps: List[torch.Tensor] = []
        for sweep_frame, image_map in zip(sweep_frames, sweep_image_maps):
            imgs: List[torch.Tensor] = []
            sensor2ego_mats: List[torch.Tensor] = []
            intrin_mats: List[torch.Tensor] = []
            ida_mats: List[torch.Tensor] = []
            sensor2sensor_mats: List[torch.Tensor] = []
            timestamps: List[torch.Tensor] = []
            for channel in cams:
                if channel not in image_map:
                    raise RuntimeError(f'Missing camera image for channel={channel}')
                cam_image = image_map[channel]
                _, height, width = cam_image.shape
                resize, resize_dims, crop = self._sample_augmentation(height=int(height), width=int(width))
                transformed = self._resize_crop_image(cam_image, resize_dims=resize_dims, crop=crop)
                imgs.append(self._normalize_bevdet_image(transformed))
                ida_mat = torch.eye(4, dtype=torch.float32, device=self.device)
                ida_mat[0, 0] = float(resize)
                ida_mat[1, 1] = float(resize)
                ida_mat[0, 3] = -float(crop[0])
                ida_mat[1, 3] = -float(crop[1])
                sweepsensor2keyego, keysensor2sweepsensor = self._bevdepth_sweep_mats(key_frame=frame, sweep_frame=sweep_frame, channel=channel)
                intrin_mat = torch.eye(4, dtype=torch.float32, device=self.device)
                intrin_mat[:3, :3] = torch.as_tensor(sweep_frame.cameras[channel].camera_intrinsic, dtype=torch.float32, device=self.device)
                sensor2ego_mats.append(sweepsensor2keyego)
                intrin_mats.append(intrin_mat)
                ida_mats.append(ida_mat)
                sensor2sensor_mats.append(keysensor2sweepsensor)
                timestamps.append(torch.as_tensor(float(sweep_frame.timestamp), dtype=torch.float32, device=self.device))
            sweep_imgs.append(torch.stack(imgs, dim=0))
            sweep_sensor2ego_mats.append(torch.stack(sensor2ego_mats, dim=0))
            sweep_intrin_mats.append(torch.stack(intrin_mats, dim=0))
            sweep_ida_mats.append(torch.stack(ida_mats, dim=0))
            sweep_sensor2sensor_mats.append(torch.stack(sensor2sensor_mats, dim=0))
            sweep_timestamps.append(torch.stack(timestamps, dim=0))
        imgs_tensor = torch.stack(sweep_imgs, dim=0).unsqueeze(0)
        mats_dict = {'sensor2ego_mats': torch.stack(sweep_sensor2ego_mats, dim=0).unsqueeze(0), 'intrin_mats': torch.stack(sweep_intrin_mats, dim=0).unsqueeze(0), 'ida_mats': torch.stack(sweep_ida_mats, dim=0).unsqueeze(0), 'sensor2sensor_mats': torch.stack(sweep_sensor2sensor_mats, dim=0).unsqueeze(0), 'bda_mat': torch.eye(4, dtype=torch.float32, device=self.device).unsqueeze(0)}
        timestamps_tensor = torch.stack(sweep_timestamps, dim=0).unsqueeze(0)
        meta = {'sample_idx': frame.sample_token, 'sample_token': frame.sample_token, 'token': frame.sample_token, 'scene_token': frame.scene_token, 'box_type_3d': self.lidar_box_cls, 'ego2global_translation': np.asarray(self._global_from_keyego_np(frame)[:3, 3], dtype=np.float32), 'ego2global_rotation': _rotation_matrix_to_quaternion_wxyz(self._global_from_keyego_np(frame)[:3, :3])}
        can_bus = np.asarray(frame.can_bus, dtype=np.float32).reshape(-1)
        abs_pos = can_bus[:3].copy() if can_bus.shape[0] >= 3 else np.zeros((3,), dtype=np.float32)
        abs_angle = float(can_bus[-1]) if can_bus.shape[0] > 0 else 0.0
        return (imgs_tensor, mats_dict, timestamps_tensor, [meta], abs_pos, abs_angle)

    def forward_frame(self, frame: FrameRecord, camera_images: Dict[str, torch.Tensor], *, prev_bev: Optional[torch.Tensor], prev_scene_token: Optional[str], prev_abs_pos: Optional[np.ndarray], prev_abs_angle: Optional[float], retain_grad: bool) -> Tuple[Any, Optional[torch.Tensor], np.ndarray, float]:
        del prev_bev
        if self.model is None:
            raise RuntimeError('BEVDepth model has not been built')
        sweep_imgs, mats_dict, timestamps, img_metas, abs_pos, abs_angle = self.prepare_model_inputs(frame, camera_images, prev_scene_token=prev_scene_token, prev_abs_pos=prev_abs_pos, prev_abs_angle=prev_abs_angle)
        with self._autocast_context():
            outs = self.model(sweep_imgs, mats_dict, timestamps)
        target_label = self.frame_target_label(frame)
        task_id, _local_label, _label_offset = self._target_task_info(target_label)
        preds = self._task_preds(outs, task_id)
        heatmap = preds['heatmap']
        reg = preds['reg']
        self.last_heatmap_tensor = heatmap
        self.last_reg_tensor = reg
        self.last_cls_tensor = self._flatten_task_heatmap(preds)
        self.last_bbox_tensor = self._flatten_task_box_code(preds)
        self.last_heatmap_grad = None
        self.last_reg_grad = None
        self.last_cls_grad = None
        self.last_bbox_grad = None
        if retain_grad:
            if heatmap.requires_grad:
                heatmap.retain_grad()
            if reg.requires_grad:
                reg.retain_grad()
            if self.last_cls_tensor.requires_grad:
                self.last_cls_tensor.retain_grad()
            if self.last_bbox_tensor.requires_grad:
                self.last_bbox_tensor.retain_grad()

                def _save_bbox_grad(grad: torch.Tensor) -> torch.Tensor:
                    self.last_bbox_grad = grad.detach()
                    return grad
                self.last_bbox_tensor.register_hook(_save_bbox_grad)
            if self.last_cls_tensor.requires_grad:

                def _save_cls_grad(grad: torch.Tensor) -> torch.Tensor:
                    self.last_cls_grad = grad.detach()
                    return grad
                self.last_cls_tensor.register_hook(_save_cls_grad)
        self._prev_frame_for_history = frame
        self._prev_images_for_history = {channel: image.detach() for channel, image in camera_images.items()}
        return (outs, None, abs_pos, abs_angle)

    def match_target_queries(self, frames: Sequence[FrameRecord], image_provider: Callable[[FrameRecord], Dict[str, torch.Tensor]], *, conf_threshold: float, max_center_dist_m: float, center_cost_weight: float, confidence_cost_weight: float) -> Dict[str, FixedQueryMatch]:
        return super().match_target_queries(frames, image_provider, conf_threshold=conf_threshold, max_center_dist_m=max_center_dist_m, center_cost_weight=center_cost_weight, confidence_cost_weight=confidence_cost_weight)

    def official_results_payload(self, frames: Sequence[FrameRecord], image_provider: Callable[[FrameRecord], Dict[str, torch.Tensor]], *, score_threshold: float=0.2) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError('BEVDepth model has not been built')
        try:
            from pyquaternion import Quaternion
        except Exception as exc:
            raise ImportError('Official BEVDepth result export requires pyquaternion') from exc
        payload: Dict[str, Any] = {'meta': {'use_lidar': False, 'use_camera': True, 'use_radar': False, 'use_map': False, 'use_external': False}, 'results': {}}
        for frame in frames:
            sweep_imgs, mats_dict, timestamps, img_metas, _, _ = self.prepare_model_inputs(frame, image_provider(frame), prev_scene_token=None, prev_abs_pos=None, prev_abs_angle=None)
            with torch.no_grad():
                with self._autocast_context():
                    outs = self.model(sweep_imgs, mats_dict, timestamps)
                    bbox_list = self.model.get_bboxes(outs, img_metas, rescale=False)
            sample_records: List[Dict[str, Any]] = []
            if bbox_list:
                bboxes, scores, labels = bbox_list[0]
                box_tensor = bboxes.tensor if hasattr(bboxes, 'tensor') else bboxes
                keep = scores >= float(score_threshold)
                kept_boxes = box_tensor[keep].detach().cpu()
                kept_scores = scores[keep].detach().cpu()
                kept_labels = labels[keep].detach().cpu()
                global_quat = Quaternion(_rotation_matrix_to_quaternion_wxyz(self._global_from_keyego_np(frame)[:3, :3]))
                global_trans = np.asarray(self._global_from_keyego_np(frame)[:3, 3], dtype=np.float32)
                for idx, box in enumerate(kept_boxes):
                    label_idx = int(kept_labels[idx].item())
                    detection_name = str(self.class_names[label_idx])
                    center = box[:3].numpy()
                    wlh = box[[4, 3, 5]].numpy()
                    yaw = float(box[6].item())
                    nusc_quat = Quaternion(axis=[0, 0, 1], radians=yaw)
                    nusc_quat = global_quat * nusc_quat
                    center_world = global_quat.rotation_matrix @ center + global_trans
                    sample_records.append({'sample_token': str(frame.sample_token), 'translation': [float(v) for v in center_world.tolist()], 'size': [float(v) for v in wlh.tolist()], 'rotation': [float(v) for v in nusc_quat.elements.tolist()], 'velocity': [0.0, 0.0], 'detection_name': detection_name, 'detection_score': float(kept_scores[idx].item()), 'attribute_name': 'vehicle.parked' if detection_name == 'car' else ''})
            payload['results'][str(frame.sample_token)] = sample_records
        return payload
__all__ = ['CAMERA_CHANNELS', 'BevDepthGradientModel', 'FrameRecord']
