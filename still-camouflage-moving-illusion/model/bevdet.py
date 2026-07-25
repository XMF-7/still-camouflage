from __future__ import annotations
import sys
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from .bevformer import CAMERA_CHANNELS, BevFormerGradientModel, FixedQueryMatch, FrameRecord, QueryPrediction, _as_path, _rotation_matrix_to_quaternion_wxyz
_PRESET_TO_CONFIG = {'r50': 'configs/bevdet/bevdet-r50.py', 'r50-cbgs': 'configs/bevdet/bevdet-r50-cbgs.py', 'r50-4d-cbgs': 'configs/bevdet/bevdet-r50-4d-cbgs.py'}
_PRESET_TO_CKPT = {'r50': 'ckpt/bevdet-r50.pth', 'r50-cbgs': 'ckpt/bevdet-r50-cbgs.pth', 'r50-4d-cbgs': 'ckpt/bevdet-r50-4d-cbgs.pth'}
_PRESET_ALIASES = {'base': 'r50', 'default': 'r50'}

class BevDetGradientModel(BevFormerGradientModel):

    def __init__(self, *, bevdet_cfg: Dict[str, Any], device: str='cuda', use_amp: bool=True, amp_dtype: str='fp16'):
        self.repo_root = _as_path(str(bevdet_cfg.get('repo_root', '')))
        if not self.repo_root.exists():
            raise FileNotFoundError(f'BEVDet repo_root not found: {self.repo_root}')
        raw_preset = str(bevdet_cfg.get('preset', 'r50')).strip().lower() or 'r50'
        preset = _PRESET_ALIASES.get(raw_preset, raw_preset)
        if preset not in _PRESET_TO_CONFIG:
            raise ValueError('bevdet.preset only supports r50/r50-cbgs/r50-4d-cbgs')
        raw_config_path = str(bevdet_cfg.get('config_path', '') or '').strip()
        raw_checkpoint_path = str(bevdet_cfg.get('checkpoint_path', '') or '').strip()
        self.config_path = _as_path(raw_config_path) if raw_config_path else (self.repo_root / _PRESET_TO_CONFIG[preset]).resolve()
        self.checkpoint_path = _as_path(raw_checkpoint_path) if raw_checkpoint_path else (self.repo_root / _PRESET_TO_CKPT[preset]).resolve()
        if not self.config_path.exists():
            raise FileNotFoundError(f'BEVDet config not found: {self.config_path}')
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f'BEVDet checkpoint not found: {self.checkpoint_path}')
        self.data_root = _as_path(str(bevdet_cfg.get('data_root', ''))) if bevdet_cfg.get('data_root') else None
        self.workers_per_gpu = int(bevdet_cfg.get('workers_per_gpu', 0))
        self.key_camera_override = str(bevdet_cfg.get('key_camera', '') or '').strip()
        requested_device = str(device).strip().lower()
        self.device = torch.device('cuda' if requested_device == 'cuda' and torch.cuda.is_available() else 'cpu')
        if requested_device == 'cuda' and self.device.type != 'cuda':
            raise RuntimeError('Requested cuda for BEVDet, but cuda is not available')
        self.use_amp = bool(use_amp) and self.device.type == 'cuda'
        amp_dtype_name = str(amp_dtype).strip().lower()
        self.amp_dtype = torch.float16 if amp_dtype_name in {'fp16', 'float16'} else torch.bfloat16
        self.img_mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32, device=self.device).view(3, 1, 1)
        self.img_std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32, device=self.device).view(3, 1, 1)
        self.model = None
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

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type='cuda', dtype=self.amp_dtype)

    def build(self) -> None:
        mmcv = self._import_mmcv_cleanly()
        repo_str = str(self.repo_root)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        from mmcv.runner import load_checkpoint
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        from mmdet3d.models import build_model
        cfg = mmcv.Config.fromfile(str(self.config_path))
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        if self.data_root is not None:
            cfg.data_root = str(self.data_root)
            for split_name in ('train', 'val', 'test'):
                if split_name in cfg.data and 'data_root' in cfg.data[split_name]:
                    cfg.data[split_name].data_root = str(self.data_root)
        if 'workers_per_gpu' in cfg.data:
            cfg.data.workers_per_gpu = self.workers_per_gpu
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, str(self.checkpoint_path), map_location='cpu')
        model.to(self.device)
        model.eval()
        if hasattr(model, 'grid_mask'):
            model.grid_mask = None
        for param in model.parameters():
            param.requires_grad_(False)
        self.model = model
        self.lidar_box_cls = LiDARInstance3DBoxes
        self.bbox_coder = model.pts_bbox_head.bbox_coder
        self.cfg = cfg
        self.data_config = dict(cfg.get('data_config', {}))
        self.class_names = list(cfg.class_names)
        self.point_cloud_range = [float(v) for v in cfg.point_cloud_range]
        self.voxel_size = [float(v) for v in cfg.voxel_size]
        self.out_size_factor = int(getattr(self.bbox_coder, 'out_size_factor', cfg.model.pts_bbox_head.bbox_coder.out_size_factor))
        cams = list(self.data_config.get('cams', CAMERA_CHANNELS))
        self.key_camera = self.key_camera_override or (str(cams[0]) if cams else 'CAM_FRONT_LEFT')
        if 'car' in self.class_names:
            self.car_label = self.class_names.index('car')

    def _normalize_bevdet_image(self, image_rgb: torch.Tensor) -> torch.Tensor:
        if image_rgb.ndim != 3 or image_rgb.shape[0] != 3:
            raise ValueError('image tensor must be [3, H, W]')
        image = image_rgb[[2, 1, 0], :, :] * 255.0
        return (image - self.img_mean) / self.img_std

    def _sample_augmentation(self, height: int, width: int) -> Tuple[float, Tuple[int, int], Tuple[int, int, int, int]]:
        input_size = self.data_config.get('input_size', (256, 704))
        f_h, f_w = (int(input_size[0]), int(input_size[1]))
        resize = float(f_w) / float(width)
        resize += float(self.data_config.get('resize_test', 0.0))
        resize_dims = (int(width * resize), int(height * resize))
        new_w, new_h = resize_dims
        crop_h = int((1.0 - float(np.mean(self.data_config.get('crop_h', (0.0, 0.0))))) * new_h) - f_h
        crop_w = int(max(0, new_w - f_w) / 2)
        return (resize, resize_dims, (crop_w, crop_h, crop_w + f_w, crop_h + f_h))

    def _resize_crop_image(self, image_rgb: torch.Tensor, *, resize_dims: Tuple[int, int], crop: Tuple[int, int, int, int]) -> torch.Tensor:
        new_w, new_h = resize_dims
        resized = F.interpolate(image_rgb.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
        left, top, right, bottom = crop
        if top < 0 or left < 0 or bottom > new_h or (right > new_w):
            pad_left = max(0, -left)
            pad_top = max(0, -top)
            pad_right = max(0, right - new_w)
            pad_bottom = max(0, bottom - new_h)
            resized = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
            left += pad_left
            right += pad_left
            top += pad_top
            bottom += pad_top
        return resized[:, top:bottom, left:right]

    def _ego_matrix(self, frame: FrameRecord, channel: str) -> torch.Tensor:
        cam = frame.cameras[channel]
        matrix = torch.eye(4, dtype=torch.float32, device=self.device)
        matrix[:3, :3] = torch.as_tensor(cam.ego2global_rotation, dtype=torch.float32, device=self.device)
        matrix[:3, 3] = torch.as_tensor(cam.ego2global_translation, dtype=torch.float32, device=self.device)
        return matrix

    def _camera_sensor2ego_matrix(self, frame: FrameRecord, channel: str) -> torch.Tensor:
        cam = frame.cameras[channel]
        matrix = torch.eye(4, dtype=torch.float32, device=self.device)
        matrix[:3, :3] = torch.as_tensor(cam.sensor2ego_rotation, dtype=torch.float32, device=self.device)
        matrix[:3, 3] = torch.as_tensor(cam.sensor2ego_translation, dtype=torch.float32, device=self.device)
        return matrix

    def _global_from_keyego_np(self, frame: FrameRecord) -> np.ndarray:
        cam = frame.cameras.get(self.key_camera)
        if cam is None:
            cam = frame.cameras[CAMERA_CHANNELS[0]]
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = np.asarray(cam.ego2global_rotation, dtype=np.float32).reshape(3, 3)
        matrix[:3, 3] = np.asarray(cam.ego2global_translation, dtype=np.float32).reshape(3)
        return matrix

    def _gt_center_keyego_tensor(self, frame: FrameRecord) -> torch.Tensor:
        keyego_from_global = np.linalg.inv(self._global_from_keyego_np(frame)).astype(np.float32)
        center_world = np.ones((4,), dtype=np.float32)
        center_world[:3] = np.asarray(frame.gt_center_world, dtype=np.float32).reshape(3)
        center_ego = keyego_from_global @ center_world
        return torch.as_tensor(center_ego[:3], dtype=torch.float32, device=self.device)

    def _gt_box_keyego_tensor(self, frame: FrameRecord) -> torch.Tensor:
        center = self._gt_center_keyego_tensor(frame)
        size = torch.as_tensor(frame.gt_size_wlh, dtype=torch.float32, device=self.device).reshape(3)
        global_from_keyego = self._global_from_keyego_np(frame)
        ego_yaw = math.atan2(float(global_from_keyego[1, 0]), float(global_from_keyego[0, 0]))
        yaw = torch.as_tensor([float(frame.gt_yaw_world) - ego_yaw], dtype=torch.float32, device=self.device)
        return torch.cat([center, size, yaw], dim=0)

    def _target_task_info(self, target_label: int) -> Tuple[int, int, int]:
        if self.model is None:
            raise RuntimeError('BEVDet model has not been built')
        num_classes = list(getattr(self.model.pts_bbox_head, 'num_classes', [len(self.class_names)]))
        offset = 0
        for task_id, count in enumerate(num_classes):
            count_int = int(count)
            if offset <= int(target_label) < offset + count_int:
                return (task_id, int(target_label) - offset, offset)
            offset += count_int
        raise IndexError(f'target_label={target_label} is outside BEVDet task classes={num_classes}')

    def _task_preds(self, outs: Any, task_id: int) -> Dict[str, torch.Tensor]:
        if isinstance(outs, dict):
            return outs
        task_out = outs[int(task_id)]
        if isinstance(task_out, dict):
            return task_out
        if isinstance(task_out, (list, tuple)) and task_out and isinstance(task_out[0], dict):
            return task_out[0]
        raise TypeError(f'Unsupported BEVDet outs structure for task_id={task_id}: {type(task_out)!r}')

    def _grid_from_keyego_xy(self, gt_x: torch.Tensor, gt_y: torch.Tensor, *, feat_h: int, feat_w: int) -> Tuple[int, int, float, float, float, float]:
        if len(self.point_cloud_range) >= 5:
            min_x = float(self.point_cloud_range[0])
            min_y = float(self.point_cloud_range[1])
            max_x = float(self.point_cloud_range[3])
            max_y = float(self.point_cloud_range[4])
        else:
            min_x = min_y = -51.2
            max_x = max_y = 51.2
        range_x = max_x - min_x
        range_y = max_y - min_y
        grid_x = range_x / float(max(1, feat_w))
        grid_y = range_y / float(max(1, feat_h))
        old_w = int(torch.floor((gt_x.detach() - gt_x.new_tensor(min_x)) / gt_x.new_tensor(grid_x)).item())
        old_h = int(torch.floor((gt_y.detach() - gt_y.new_tensor(min_y)) / gt_y.new_tensor(grid_y)).item())
        old_w = max(0, min(int(feat_w) - 1, old_w))
        old_h = max(0, min(int(feat_h) - 1, old_h))
        return (old_h, old_w, min_x, min_y, range_x, range_y)

    def _flatten_task_heatmap(self, preds: Dict[str, torch.Tensor]) -> torch.Tensor:
        heatmap = preds['heatmap']
        if heatmap.ndim != 4 or int(heatmap.shape[0]) != 1:
            raise ValueError(f'Expected BEVDet heatmap [1,C,H,W], got {tuple(heatmap.shape)}')
        return heatmap[0].permute(1, 2, 0).reshape(-1, int(heatmap.shape[1]))

    def _flatten_task_reg(self, preds: Dict[str, torch.Tensor]) -> torch.Tensor:
        reg = preds['reg']
        if reg.ndim != 4 or int(reg.shape[0]) != 1:
            raise ValueError(f'Expected BEVDet reg [1,2,H,W], got {tuple(reg.shape)}')
        return reg[0].permute(1, 2, 0).reshape(-1, int(reg.shape[1]))

    def _flatten_task_box_code(self, preds: Dict[str, torch.Tensor]) -> torch.Tensor:
        heatmap = preds['heatmap']
        if heatmap.ndim != 4 or int(heatmap.shape[0]) != 1:
            raise ValueError(f'Expected BEVDet heatmap [1,C,H,W], got {tuple(heatmap.shape)}')
        feat_h = int(heatmap.shape[-2])
        feat_w = int(heatmap.shape[-1])
        device = heatmap.device
        dtype = heatmap.dtype
        reg = preds['reg'][0].permute(1, 2, 0).reshape(-1, 2)
        height = preds['height'][0].permute(1, 2, 0).reshape(-1, 1)
        dim_raw = preds['dim']
        dim = torch.exp(dim_raw)[0].permute(1, 2, 0).reshape(-1, 3) if getattr(self.model.pts_bbox_head, 'norm_bbox', False) else dim_raw[0].permute(1, 2, 0).reshape(-1, 3)
        rot = preds['rot'][0].permute(1, 2, 0).reshape(-1, 2)
        yaw = torch.atan2(rot[:, 0:1], rot[:, 1:2])
        ys, xs = torch.meshgrid(torch.arange(feat_h, dtype=dtype, device=device), torch.arange(feat_w, dtype=dtype, device=device), indexing='ij')
        xs = xs.reshape(-1, 1) + reg[:, 0:1]
        ys = ys.reshape(-1, 1) + reg[:, 1:2]
        voxel_size = getattr(self.bbox_coder, 'voxel_size', self.voxel_size[:2])
        pc_range = getattr(self.bbox_coder, 'pc_range', self.point_cloud_range[:2])
        out_size_factor = float(getattr(self.bbox_coder, 'out_size_factor', self.out_size_factor))
        xs = xs * out_size_factor * float(voxel_size[0]) + float(pc_range[0])
        ys = ys * out_size_factor * float(voxel_size[1]) + float(pc_range[1])
        parts = [xs, ys, height, dim, yaw]
        if 'vel' in preds:
            vel = preds['vel'][0].permute(1, 2, 0).reshape(-1, int(preds['vel'].shape[1]))
            parts.append(vel)
        return torch.cat(parts, dim=1)

    def _decode_final_detections_with_indices(self, outs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.model is None or self.bbox_coder is None or self.lidar_box_cls is None:
            raise RuntimeError('BEVDet model has not been built')
        from mmdet3d.core.post_processing import nms_bev
        all_boxes: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        all_query_indices: List[torch.Tensor] = []
        label_offset = 0
        head = self.model.pts_bbox_head
        num_classes = list(getattr(head, 'num_classes', [len(self.class_names)]))
        for task_id, class_count in enumerate(num_classes):
            preds = self._task_preds(outs, task_id)
            heatmap = preds['heatmap'].sigmoid()
            scores, inds, clses, _ys, _xs = self.bbox_coder._topk(heatmap, K=int(self.bbox_coder.max_num))
            task_boxes = self._flatten_task_box_code(preds)
            flat_inds = inds.reshape(-1).long()
            boxes = task_boxes[flat_inds]
            scores = scores.reshape(-1)
            labels = clses.reshape(-1).long()
            query_indices = flat_inds
            keep = torch.ones_like(scores, dtype=torch.bool)
            score_threshold = getattr(self.bbox_coder, 'score_threshold', None)
            if score_threshold is not None:
                keep &= scores > float(score_threshold)
            post_center_range = getattr(self.bbox_coder, 'post_center_range', None)
            if post_center_range is not None:
                center_range = torch.as_tensor(post_center_range, dtype=boxes.dtype, device=boxes.device)
                keep &= (boxes[:, :3] >= center_range[:3]).all(dim=1)
                keep &= (boxes[:, :3] <= center_range[3:]).all(dim=1)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            query_indices = query_indices[keep]
            if int(scores.numel()) <= 0:
                label_offset += int(class_count)
                continue
            nms_type = head.test_cfg.get('nms_type')
            if isinstance(nms_type, list):
                nms_type = nms_type[task_id]
            if nms_type == 'circle':
                from mmdet3d.core.post_processing import circle_nms
                centers = boxes[:, [0, 1]]
                circle_boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                selected = torch.as_tensor(circle_nms(circle_boxes.detach().cpu().numpy(), head.test_cfg['min_radius'][task_id], post_max_size=head.test_cfg['post_max_size']), dtype=torch.long, device=boxes.device)
            else:
                default_val = [1.0 for _ in range(len(head.task_heads))]
                factor = head.test_cfg.get('nms_rescale_factor', default_val)[task_id]
                boxes_for_nms_pred = boxes.clone()
                if isinstance(factor, list):
                    for cid, scale in enumerate(factor):
                        boxes_for_nms_pred[labels == int(cid), 3:6] *= float(scale)
                else:
                    boxes_for_nms_pred[:, 3:6] *= float(factor)
                boxes_for_nms = self.lidar_box_cls(boxes_for_nms_pred, self.bbox_coder.code_size).bev
                nms_thresh = head.test_cfg['nms_thr'][task_id] if isinstance(head.test_cfg['nms_thr'], list) else head.test_cfg['nms_thr']
                selected = nms_bev(boxes_for_nms, scores, thresh=nms_thresh, pre_max_size=head.test_cfg['pre_max_size'], post_max_size=head.test_cfg['post_max_size'], xyxyr2xywhr=False)
            selected = selected.to(device=boxes.device, dtype=torch.long)
            if int(selected.numel()) <= 0:
                label_offset += int(class_count)
                continue
            selected_boxes = boxes[selected].clone()
            selected_boxes[:, 2] = selected_boxes[:, 2] - selected_boxes[:, 5] * 0.5
            all_boxes.append(selected_boxes)
            all_scores.append(scores[selected])
            all_labels.append(labels[selected] + int(label_offset))
            all_query_indices.append(query_indices[selected])
            label_offset += int(class_count)
        if not all_boxes:
            device = self.last_heatmap_tensor.device if self.last_heatmap_tensor is not None else self.device
            return (torch.zeros((0, int(getattr(self.bbox_coder, 'code_size', 9))), dtype=torch.float32, device=device), torch.zeros((0,), dtype=torch.float32, device=device), torch.zeros((0,), dtype=torch.long, device=device), torch.zeros((0,), dtype=torch.long, device=device))
        return (torch.cat(all_boxes, dim=0), torch.cat(all_scores, dim=0), torch.cat(all_labels, dim=0), torch.cat(all_query_indices, dim=0))

    def prepare_model_inputs(self, frame: FrameRecord, camera_images: Dict[str, torch.Tensor], *, prev_scene_token: Optional[str], prev_abs_pos: Optional[np.ndarray], prev_abs_angle: Optional[float]) -> Tuple[List[torch.Tensor], List[Dict[str, Any]], np.ndarray, float]:
        del prev_scene_token, prev_abs_pos, prev_abs_angle
        cams = [str(cam) for cam in self.data_config.get('cams', CAMERA_CHANNELS)]
        images: List[torch.Tensor] = []
        sensor2egos: List[torch.Tensor] = []
        ego2globals: List[torch.Tensor] = []
        intrins: List[torch.Tensor] = []
        post_rots: List[torch.Tensor] = []
        post_trans_list: List[torch.Tensor] = []
        for channel in cams:
            if channel not in camera_images:
                raise RuntimeError(f'Missing camera image for channel={channel}')
            cam_image = camera_images[channel]
            _, height, width = cam_image.shape
            resize, resize_dims, crop = self._sample_augmentation(height=int(height), width=int(width))
            transformed = self._resize_crop_image(cam_image, resize_dims=resize_dims, crop=crop)
            images.append(self._normalize_bevdet_image(transformed))
            post_rot = torch.eye(3, dtype=torch.float32, device=self.device)
            post_trans = torch.zeros(3, dtype=torch.float32, device=self.device)
            post_rot[:2, :2] *= float(resize)
            post_trans[:2] -= torch.as_tensor(crop[:2], dtype=torch.float32, device=self.device)
            sensor2egos.append(self._camera_sensor2ego_matrix(frame, channel))
            ego2globals.append(self._ego_matrix(frame, channel))
            intrins.append(torch.as_tensor(frame.cameras[channel].camera_intrinsic, dtype=torch.float32, device=self.device))
            post_rots.append(post_rot)
            post_trans_list.append(post_trans)
        imgs = torch.stack(images, dim=0).unsqueeze(0)
        img_inputs = [imgs, torch.stack(sensor2egos, dim=0).unsqueeze(0), torch.stack(ego2globals, dim=0).unsqueeze(0), torch.stack(intrins, dim=0).unsqueeze(0), torch.stack(post_rots, dim=0).unsqueeze(0), torch.stack(post_trans_list, dim=0).unsqueeze(0), torch.eye(4, dtype=torch.float32, device=self.device).unsqueeze(0)]
        meta = {'sample_idx': frame.sample_token, 'scene_token': frame.scene_token, 'box_type_3d': self.lidar_box_cls, 'pcd_scale_factor': 1.0, 'pcd_horizontal_flip': False, 'pcd_vertical_flip': False}
        can_bus = np.asarray(frame.can_bus, dtype=np.float32).reshape(-1)
        abs_pos = can_bus[:3].copy() if can_bus.shape[0] >= 3 else np.zeros((3,), dtype=np.float32)
        abs_angle = float(can_bus[-1]) if can_bus.shape[0] > 0 else 0.0
        return (img_inputs, [meta], abs_pos, abs_angle)

    def forward_frame(self, frame: FrameRecord, camera_images: Dict[str, torch.Tensor], *, prev_bev: Optional[torch.Tensor], prev_scene_token: Optional[str], prev_abs_pos: Optional[np.ndarray], prev_abs_angle: Optional[float], retain_grad: bool) -> Tuple[Any, Optional[torch.Tensor], np.ndarray, float]:
        del prev_bev
        if self.model is None:
            raise RuntimeError('BEVDet model has not been built')
        img_inputs, img_metas, abs_pos, abs_angle = self.prepare_model_inputs(frame, camera_images, prev_scene_token=prev_scene_token, prev_abs_pos=prev_abs_pos, prev_abs_angle=prev_abs_angle)
        with self._autocast_context():
            img_feats, _, _ = self.model.extract_feat(None, img=img_inputs, img_metas=img_metas)
            outs = self.model.pts_bbox_head(img_feats)
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
        return (outs, None, abs_pos, abs_angle)

    def match_target_queries(self, frames: Sequence[FrameRecord], image_provider: Callable[[FrameRecord], Dict[str, torch.Tensor]], *, conf_threshold: float, max_center_dist_m: float, center_cost_weight: float, confidence_cost_weight: float) -> Dict[str, FixedQueryMatch]:
        del center_cost_weight, confidence_cost_weight
        matches: Dict[str, FixedQueryMatch] = {}
        for frame in frames:
            with torch.no_grad():
                outs, _, _, _ = self.forward_frame(frame, image_provider(frame), prev_bev=None, prev_scene_token=None, prev_abs_pos=None, prev_abs_angle=None, retain_grad=False)
                matches[frame.cache_key] = self.match_target_query_from_final_outputs(frame, outs, conf_threshold=float(conf_threshold), max_center_dist_m=float(max_center_dist_m), distance_axis='lateral_y', max_cross_axis_dist_m=float(max_center_dist_m))
            del outs
        return matches

    def match_target_query_from_final_outputs(self, frame: FrameRecord, outs: Any, *, conf_threshold: float, max_center_dist_m: float, distance_axis: str='lateral_y', max_cross_axis_dist_m: float=1.0) -> FixedQueryMatch:
        target_label = self.frame_target_label(frame)
        target_detection_name = self.frame_target_detection_name(frame)
        boxes, scores, labels, query_indices = self._decode_final_detections_with_indices(outs)
        label_mask = labels == int(target_label)
        boxes = boxes[label_mask]
        scores = scores[label_mask]
        query_indices = query_indices[label_mask]
        total_candidates = int(boxes.shape[0])
        target_world_xy = (float(frame.gt_center_world[0]), float(frame.gt_center_world[1]))
        target_center_ego = self._gt_center_keyego_tensor(frame)
        target_ego_xyz = tuple((float(v) for v in target_center_ego.detach().cpu().tolist()))
        if total_candidates <= 0:
            return FixedQueryMatch(sample_token=frame.sample_token, frame_id=frame.frame_id, matched=False, query_idx=-1, confidence=0.0, world_distance_m=float('inf'), match_cost=float('inf'), candidate_total=0, candidate_after_conf=0, candidate_after_dist=0, target_world_xy=target_world_xy, pred_world_xy=None, target_detection_name=target_detection_name, unmatched_reason=f'BEVDet postprocess has no {target_detection_name} class candidates', target_ego_xyz=target_ego_xyz)
        with torch.no_grad():
            pred_centers_ego = boxes[:, :3]
            deltas = pred_centers_ego[:, :2] - target_center_ego[:2].unsqueeze(0)
            center_distances = torch.linalg.norm(deltas, dim=1)
            conf_keep = scores >= float(conf_threshold)
            after_conf = int(conf_keep.sum().item())
            if after_conf <= 0:
                conf_keep = scores >= scores.max()
                after_conf = int(conf_keep.sum().item())
            if str(distance_axis).strip().lower() == 'forward_x':
                primary = torch.abs(deltas[:, 0])
                cross = torch.abs(deltas[:, 1])
            else:
                primary = torch.abs(deltas[:, 1])
                cross = torch.abs(deltas[:, 0])
            dist_keep = (primary <= float(max_center_dist_m)) & (cross <= float(max_cross_axis_dist_m))
            keep = conf_keep & dist_keep
            after_dist = int(keep.sum().item())
            if after_dist <= 0:
                keep = conf_keep
            kept_idx = torch.nonzero(keep, as_tuple=False).reshape(-1)
            kept_center = center_distances[kept_idx]
            kept_scores = scores[kept_idx].to(kept_center.dtype)
            kept_sort = kept_center - kept_scores * 1e-06
            best_local = int(torch.argmin(kept_sort).item())
            best_det_idx = int(kept_idx[best_local].item())
        best_box = boxes[best_det_idx]
        best_score = float(scores[best_det_idx].detach().item())
        best_query_idx = int(query_indices[best_det_idx].detach().item())
        center_h = np.ones((4,), dtype=np.float32)
        center_h[:3] = best_box[:3].detach().float().cpu().numpy()
        best_world = self._global_from_keyego_np(frame) @ center_h
        pred_world_xy = (float(best_world[0]), float(best_world[1]))
        world_distance_m = math.hypot(pred_world_xy[0] - target_world_xy[0], pred_world_xy[1] - target_world_xy[1])
        return FixedQueryMatch(sample_token=frame.sample_token, frame_id=frame.frame_id, matched=True, query_idx=best_query_idx, confidence=best_score, world_distance_m=world_distance_m, match_cost=float(center_distances[best_det_idx].detach().item()), candidate_total=total_candidates, candidate_after_conf=after_conf, candidate_after_dist=after_dist, target_world_xy=target_world_xy, pred_world_xy=pred_world_xy, target_detection_name=target_detection_name, unmatched_reason='', target_ego_xyz=target_ego_xyz)

    def target_query_prediction(self, frame: FrameRecord, outs: Any, *, query_idx: int) -> QueryPrediction:
        target_label = self.frame_target_label(frame)
        target_detection_name = self.frame_target_detection_name(frame)
        task_id, local_label, _label_offset = self._target_task_info(target_label)
        preds = self._task_preds(outs, task_id)
        heatmap = preds['heatmap']
        feat_h = int(heatmap.shape[-2])
        feat_w = int(heatmap.shape[-1])
        gt_center = self._gt_center_keyego_tensor(frame)
        gt_box = self._gt_box_keyego_tensor(frame)
        old_h, old_w, _min_x, _min_y, _range_x, _range_y = self._grid_from_keyego_xy(gt_center[0], gt_center[1], feat_h=feat_h, feat_w=feat_w)
        if query_idx >= 0:
            old_h = max(0, min(feat_h - 1, int(query_idx) // feat_w))
            old_w = max(0, min(feat_w - 1, int(query_idx) % feat_w))
        flat_idx = old_h * feat_w + old_w
        if self.last_bbox_tensor is None:
            bbox_tensor = self._flatten_task_box_code(preds)
        else:
            bbox_tensor = self.last_bbox_tensor
        pred_box = bbox_tensor[flat_idx].clone()
        if int(pred_box.numel()) >= 6:
            pred_box[2] = pred_box[2] - pred_box[5] * 0.5
        if int(pred_box.numel()) < 7:
            pred_box = torch.cat([pred_box, gt_box[int(pred_box.numel()):7]], dim=0)
        gt_box_like = gt_box.clone()
        flat_heatmap = heatmap[0].permute(1, 2, 0).reshape(-1, int(heatmap.shape[1]))
        bbox_grad_center_xyz = None
        cls_grad_target = None
        bbox_grad_src = self.last_bbox_grad
        if bbox_grad_src is None and self.last_bbox_tensor is not None and (self.last_bbox_tensor.grad is not None):
            bbox_grad_src = self.last_bbox_tensor.grad
        cls_grad_src = self.last_cls_grad
        if cls_grad_src is None and self.last_cls_tensor is not None and (self.last_cls_tensor.grad is not None):
            cls_grad_src = self.last_cls_tensor.grad
        if bbox_grad_src is not None and 0 <= int(flat_idx) < int(bbox_grad_src.shape[0]):
            bbox_grad_center_xyz = bbox_grad_src[flat_idx, 0:3].detach().clone()
        if cls_grad_src is not None and 0 <= int(flat_idx) < int(cls_grad_src.shape[0]):
            if 0 <= int(local_label) < int(cls_grad_src.shape[1]):
                cls_grad_target = cls_grad_src[flat_idx, local_label:local_label + 1].detach().clone()
        if cls_grad_target is None and self.last_heatmap_tensor is not None and (self.last_heatmap_tensor.grad is not None):
            if 0 <= int(local_label) < int(self.last_heatmap_tensor.shape[1]):
                cls_grad_target = self.last_heatmap_tensor.grad[0, local_label, old_h, old_w].reshape(1).detach().clone()
        return QueryPrediction(frame_id=frame.frame_id, sample_token=frame.sample_token, query_idx=int(flat_idx), pred_box_lidar=pred_box, gt_box_lidar=gt_box_like, pred_box_world=pred_box, gt_box_world=gt_box_like, pred_center_ego=pred_box[:3], gt_center_ego=gt_center, pred_corners_ego=torch.zeros((4, 2), dtype=pred_box.dtype, device=pred_box.device), gt_corners_ego=torch.zeros((4, 2), dtype=pred_box.dtype, device=pred_box.device), class_logits=flat_heatmap[flat_idx], target_logit=heatmap[0, local_label, old_h, old_w], target_label=local_label, target_detection_name=target_detection_name, bbox_grad_center_xyz=bbox_grad_center_xyz, cls_grad_target=cls_grad_target)

    def bevdet_heatmap_shift_loss(self, frame: FrameRecord, outs: Any, *, shift_m: float=1.0, old_threshold: float=0.0, old_weight: float=0.5, reg_weight: float=0.1) -> Tuple[torch.Tensor, Dict[str, float], QueryPrediction]:
        target_label = self.frame_target_label(frame)
        task_id, local_label, _label_offset = self._target_task_info(target_label)
        preds = self._task_preds(outs, task_id)
        heatmap_full = preds['heatmap']
        reg_full = preds['reg']
        heatmap = heatmap_full[0, local_label]
        reg = reg_full[0]
        feat_h = int(heatmap.shape[0])
        feat_w = int(heatmap.shape[1])
        gt_center = self._gt_center_keyego_tensor(frame)
        gt_x, gt_y = (gt_center[0], gt_center[1])
        old_h, old_w, min_x, _min_y, range_x, _range_y = self._grid_from_keyego_xy(gt_x, gt_y, feat_h=feat_h, feat_w=feat_w)
        target_x = gt_x - gt_x.new_tensor(float(shift_m))
        grid_size_x = range_x / float(max(1, feat_w))
        target_w = int(torch.floor((target_x.detach() - target_x.new_tensor(min_x)) / target_x.new_tensor(grid_size_x)).item())
        target_w = max(0, min(feat_w - 1, target_w))
        new_logit = heatmap[old_h, target_w]
        old_logit = heatmap[old_h, old_w]
        loss_new = -new_logit
        loss_old = F.relu(old_logit - old_logit.new_tensor(float(old_threshold))).mean()
        pred_x = old_w / float(max(1, feat_w)) * range_x + min_x + reg[0, old_h, old_w]
        loss_reg = (pred_x - target_x).abs()
        loss = loss_new + loss_old.new_tensor(float(old_weight)) * loss_old + loss_reg.new_tensor(float(reg_weight)) * loss_reg
        prediction = self.target_query_prediction(frame, outs, query_idx=old_h * feat_w + old_w)
        stats = {'loss_new': float(loss_new.detach().item()), 'loss_old': float(loss_old.detach().item()), 'loss_reg': float(loss_reg.detach().item()), 'loss_bevdet': float(loss.detach().item()), 'old_logit': float(old_logit.detach().item()), 'new_logit': float(new_logit.detach().item()), 'old_confidence': float(torch.sigmoid(old_logit.detach()).item()), 'new_confidence': float(torch.sigmoid(new_logit.detach()).item()), 'gt_x_m': float(gt_x.detach().item()), 'gt_y_m': float(gt_y.detach().item()), 'target_x_m': float(target_x.detach().item()), 'pred_x_proxy_m': float(pred_x.detach().item()), 'grid_old_h': float(old_h), 'grid_old_w': float(old_w), 'grid_target_w': float(target_w), 'shift_m': float(shift_m)}
        return (loss, stats, prediction)

    def nearby_target_logits(self, frame: FrameRecord, *, query_idx: int, anchor_center_ego: torch.Tensor, target_label: int, radius_m: float) -> torch.Tensor:
        del frame, anchor_center_ego, radius_m
        if self.last_cls_tensor is None or query_idx < 0:
            return torch.zeros((0,), dtype=torch.float32, device=self.device)
        cls_tensor = self.last_cls_tensor
        if int(target_label) < 0 or int(target_label) >= int(cls_tensor.shape[1]):
            return torch.zeros((0,), dtype=cls_tensor.dtype, device=cls_tensor.device)
        keep = torch.ones((int(cls_tensor.shape[0]),), dtype=torch.bool, device=cls_tensor.device)
        if int(query_idx) < int(cls_tensor.shape[0]):
            keep[int(query_idx)] = False
        return cls_tensor[keep, int(target_label)]

    def official_results_payload(self, frames: Sequence[FrameRecord], image_provider: Callable[[FrameRecord], Dict[str, torch.Tensor]], *, score_threshold: float=0.2) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError('BEVDet model has not been built')
        try:
            from pyquaternion import Quaternion
        except Exception as exc:
            raise ImportError('Official BEVDet result export requires pyquaternion') from exc
        payload: Dict[str, Any] = {'meta': {'use_lidar': False, 'use_camera': True, 'use_radar': False, 'use_map': False, 'use_external': False}, 'results': {}}
        for frame in frames:
            img_inputs, img_metas, _, _ = self.prepare_model_inputs(frame, image_provider(frame), prev_scene_token=None, prev_abs_pos=None, prev_abs_angle=None)
            with torch.no_grad():
                with self._autocast_context():
                    img_feats, _, _ = self.model.extract_feat(None, img=img_inputs, img_metas=img_metas)
                    outs = self.model.pts_bbox_head(img_feats)
                    bbox_list = self.model.pts_bbox_head.get_bboxes(outs, img_metas, rescale=False)
            sample_records: List[Dict[str, Any]] = []
            if bbox_list:
                bboxes, scores, labels = bbox_list[0]
                keep = scores >= float(score_threshold)
                kept_boxes = bboxes.tensor[keep].detach().cpu()
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
__all__ = ['CAMERA_CHANNELS', 'BevDetGradientModel', 'FrameRecord']
