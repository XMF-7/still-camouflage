from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple
import torch
import torch.nn.functional as F

def _as_pair(value: Any, default: Tuple[float, float]) -> Tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    if value is None:
        return default
    scalar = float(value)
    return (scalar, scalar)

def _gaussian_kernel_2d(kernel_size: int, sigma: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = max(float(sigma), 1e-06)
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp(min=1e-08)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum().clamp(min=1e-08)
    return kernel_2d

@dataclass
class SmallEoTConfig:
    enabled: bool = True
    geometry3d_prob: float = 0.8
    yaw_perturb_deg: float = 2.0
    translate_front_m: float = 0.1
    translate_left_m: float = 0.1
    translate_up_m: float = 0.03
    depth_ratio: float = 0.05
    scale_ratio_3d: float = 0.03
    geometry_prob: float = 0.8
    translate_px_x: float = 2.0
    translate_px_y: float = 2.0
    scale_delta: float = 0.03
    rotate_deg: float = 2.0
    visible_region_prob: float = 0.6
    mask_erode_px: int = 2
    mask_dilate_px: int = 2
    occ_prob: float = 0.35
    occ_max_frac: float = 0.12
    region_drop_prob: float = 0.25
    region_drop_max_frac: float = 0.08
    photo_prob: float = 0.8
    brightness: float = 0.08
    contrast: float = 0.08
    saturation: float = 0.08
    gamma: float = 0.06
    color_shift: float = 0.04
    noise_prob: float = 0.3
    noise_std: float = 0.01
    blur_prob: float = 0.2
    blur_kernel: int = 3
    blur_sigma: float = 0.8

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any] | None) -> 'SmallEoTConfig':
        payload = dict(cfg or {})
        translate_px_x, translate_px_y = _as_pair(payload.get('translate_px', [2.0, 2.0]), (2.0, 2.0))
        return cls(enabled=bool(payload.get('enabled', True)), geometry3d_prob=float(payload.get('geometry3d_prob', 0.8)), yaw_perturb_deg=float(payload.get('yaw_perturb_deg', 2.0)), translate_front_m=float(payload.get('translate_front_m', 0.1)), translate_left_m=float(payload.get('translate_left_m', 0.1)), translate_up_m=float(payload.get('translate_up_m', 0.03)), depth_ratio=float(payload.get('depth_ratio', 0.05)), scale_ratio_3d=float(payload.get('scale_ratio_3d', 0.03)), geometry_prob=float(payload.get('geometry_prob', 0.8)), translate_px_x=translate_px_x, translate_px_y=translate_px_y, scale_delta=float(payload.get('scale_delta', 0.03)), rotate_deg=float(payload.get('rotate_deg', 2.0)), visible_region_prob=float(payload.get('visible_region_prob', 0.6)), mask_erode_px=int(payload.get('mask_erode_px', 2)), mask_dilate_px=int(payload.get('mask_dilate_px', 2)), occ_prob=float(payload.get('occ_prob', 0.35)), occ_max_frac=float(payload.get('occ_max_frac', 0.12)), region_drop_prob=float(payload.get('region_drop_prob', 0.25)), region_drop_max_frac=float(payload.get('region_drop_max_frac', 0.08)), photo_prob=float(payload.get('photo_prob', 0.8)), brightness=float(payload.get('brightness', 0.08)), contrast=float(payload.get('contrast', 0.08)), saturation=float(payload.get('saturation', 0.08)), gamma=float(payload.get('gamma', 0.06)), color_shift=float(payload.get('color_shift', 0.04)), noise_prob=float(payload.get('noise_prob', 0.3)), noise_std=float(payload.get('noise_std', 0.01)), blur_prob=float(payload.get('blur_prob', 0.2)), blur_kernel=int(payload.get('blur_kernel', 3)), blur_sigma=float(payload.get('blur_sigma', 0.8)))

@dataclass
class FullImageAugmentConfig:
    enabled: bool = False
    photometric_prob: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    gamma: float = 0.0
    color_shift: float = 0.0
    noise_prob: float = 0.0
    noise_std: float = 0.0
    background_noise_prob: float = 0.0
    background_noise_std: float = 0.0
    apply_to_clean_views: bool = False
    background_noise_without_mask: bool = False
    blur_prob: float = 0.0
    blur_kernel: int = 3
    blur_sigma: float = 0.8

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any] | None) -> 'FullImageAugmentConfig':
        payload = dict(cfg or {})
        return cls(enabled=bool(payload.get('enabled', False)), photometric_prob=float(payload.get('photometric_prob', 0.0)), brightness=float(payload.get('brightness', 0.0)), contrast=float(payload.get('contrast', 0.0)), saturation=float(payload.get('saturation', 0.0)), gamma=float(payload.get('gamma', 0.0)), color_shift=float(payload.get('color_shift', 0.0)), noise_prob=float(payload.get('noise_prob', 0.0)), noise_std=float(payload.get('noise_std', 0.0)), background_noise_prob=float(payload.get('background_noise_prob', 0.0)), background_noise_std=float(payload.get('background_noise_std', 0.0)), apply_to_clean_views=bool(payload.get('apply_to_clean_views', False)), background_noise_without_mask=bool(payload.get('background_noise_without_mask', False)), blur_prob=float(payload.get('blur_prob', 0.0)), blur_kernel=int(payload.get('blur_kernel', 3)), blur_sigma=float(payload.get('blur_sigma', 0.8)))

class SmallEoTAugmentor:

    def __init__(self, cfg: Dict[str, Any] | None, *, device: torch.device) -> None:
        self.cfg = SmallEoTConfig.from_dict(cfg)
        self.device = device

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def summary(self) -> Dict[str, Any]:
        return {'enabled': bool(self.cfg.enabled), 'geometry3d_prob': float(self.cfg.geometry3d_prob), 'yaw_perturb_deg': float(self.cfg.yaw_perturb_deg), 'translate_front_m': float(self.cfg.translate_front_m), 'translate_left_m': float(self.cfg.translate_left_m), 'translate_up_m': float(self.cfg.translate_up_m), 'depth_ratio': float(self.cfg.depth_ratio), 'scale_ratio_3d': float(self.cfg.scale_ratio_3d), 'geometry_prob': float(self.cfg.geometry_prob), 'translate_px': [float(self.cfg.translate_px_x), float(self.cfg.translate_px_y)], 'scale_delta': float(self.cfg.scale_delta), 'rotate_deg': float(self.cfg.rotate_deg), 'visible_region_prob': float(self.cfg.visible_region_prob), 'mask_erode_px': int(self.cfg.mask_erode_px), 'mask_dilate_px': int(self.cfg.mask_dilate_px), 'occ_prob': float(self.cfg.occ_prob), 'occ_max_frac': float(self.cfg.occ_max_frac), 'region_drop_prob': float(self.cfg.region_drop_prob), 'region_drop_max_frac': float(self.cfg.region_drop_max_frac), 'photo_prob': float(self.cfg.photo_prob), 'brightness': float(self.cfg.brightness), 'contrast': float(self.cfg.contrast), 'saturation': float(self.cfg.saturation), 'gamma': float(self.cfg.gamma), 'color_shift': float(self.cfg.color_shift), 'noise_prob': float(self.cfg.noise_prob), 'noise_std': float(self.cfg.noise_std), 'blur_prob': float(self.cfg.blur_prob), 'blur_kernel': int(self.cfg.blur_kernel), 'blur_sigma': float(self.cfg.blur_sigma), 'extra_perspective': False}

    def _coin(self, probability: float) -> bool:
        probability = max(0.0, min(1.0, float(probability)))
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        return bool(torch.rand((), device=self.device).item() < probability)

    def _uniform(self, low: float, high: float) -> float:
        low_f = float(low)
        high_f = float(high)
        if abs(high_f - low_f) <= 1e-12:
            return low_f
        return float(torch.empty((), device=self.device).uniform_(low_f, high_f).item())

    def sample_geometry3d(self, *, distance_m: float) -> Dict[str, float] | None:
        if not self.enabled or not self._coin(self.cfg.geometry3d_prob):
            return None
        distance_m = max(float(distance_m), 1e-06)
        depth_delta_m = distance_m * self._uniform(-self.cfg.depth_ratio, self.cfg.depth_ratio)
        scale_mul = max(0.001, 1.0 + self._uniform(-self.cfg.scale_ratio_3d, self.cfg.scale_ratio_3d))
        return {'yaw_deg': self._uniform(-self.cfg.yaw_perturb_deg, self.cfg.yaw_perturb_deg), 'front_m': self._uniform(-self.cfg.translate_front_m, self.cfg.translate_front_m), 'left_m': self._uniform(-self.cfg.translate_left_m, self.cfg.translate_left_m), 'up_m': self._uniform(-self.cfg.translate_up_m, self.cfg.translate_up_m), 'depth_delta_m': depth_delta_m, 'scale_mul': scale_mul}

    def _apply_affine(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, height, width = image.shape
        angle_deg = self._uniform(-self.cfg.rotate_deg, self.cfg.rotate_deg)
        scale = self._uniform(1.0 - self.cfg.scale_delta, 1.0 + self.cfg.scale_delta)
        scale = max(scale, 0.001)
        shift_x = self._uniform(-self.cfg.translate_px_x, self.cfg.translate_px_x)
        shift_y = self._uniform(-self.cfg.translate_px_y, self.cfg.translate_px_y)
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad) / scale
        sin_a = math.sin(angle_rad) / scale
        tx = 2.0 * shift_x / max(float(width), 1.0)
        ty = 2.0 * shift_y / max(float(height), 1.0)
        theta = image.new_tensor([[cos_a, sin_a, tx], [-sin_a, cos_a, ty]]).unsqueeze(0)
        grid = F.affine_grid(theta, size=(1, image.shape[0], height, width), align_corners=False)
        warped_image = F.grid_sample(image.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(0)
        warped_mask = F.grid_sample(mask.unsqueeze(0).unsqueeze(0), grid, mode='nearest', padding_mode='zeros', align_corners=False).squeeze(0).squeeze(0)
        return (warped_image, warped_mask > 0.5)

    def _apply_photometric(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        current = image
        if self.cfg.brightness > 0.0:
            brightness = 1.0 + self._uniform(-self.cfg.brightness, self.cfg.brightness)
            current = current * brightness
        if self.cfg.contrast > 0.0:
            contrast = 1.0 + self._uniform(-self.cfg.contrast, self.cfg.contrast)
            mean = current.mean(dim=(1, 2), keepdim=True)
            current = (current - mean) * contrast + mean
        if self.cfg.saturation > 0.0:
            saturation = 1.0 + self._uniform(-self.cfg.saturation, self.cfg.saturation)
            gray = current[0:1] * 0.2989 + current[1:2] * 0.587 + current[2:3] * 0.114
            current = (current - gray) * saturation + gray
        if self.cfg.gamma > 0.0:
            gamma = max(0.5, 1.0 + self._uniform(-self.cfg.gamma, self.cfg.gamma))
            current = current.clamp(0.0, 1.0).pow(gamma)
        if self.cfg.color_shift > 0.0:
            shift = torch.empty((3, 1, 1), device=current.device, dtype=current.dtype).uniform_(-self.cfg.color_shift, self.cfg.color_shift)
            current = current + shift
        return current * mask.to(dtype=current.dtype).unsqueeze(0)

    def _apply_noise(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.cfg.noise_std <= 0.0:
            return image
        noise = torch.randn_like(image) * float(self.cfg.noise_std)
        return (image + noise) * mask.to(dtype=image.dtype).unsqueeze(0)

    def _apply_blur(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.cfg.blur_kernel <= 1 or self.cfg.blur_sigma <= 0.0:
            return image
        kernel_2d = _gaussian_kernel_2d(self.cfg.blur_kernel, self.cfg.blur_sigma, device=image.device, dtype=image.dtype)
        kernel = kernel_2d.expand(image.shape[0], 1, kernel_2d.shape[0], kernel_2d.shape[1]).contiguous()
        padding = kernel_2d.shape[0] // 2
        blurred = F.conv2d(image.unsqueeze(0), kernel, bias=None, stride=1, padding=padding, groups=image.shape[0]).squeeze(0)
        return blurred * mask.to(dtype=image.dtype).unsqueeze(0)

    def _morph_mask(self, mask: torch.Tensor, radius_px: int, *, dilate: bool) -> torch.Tensor:
        radius_px = max(0, int(radius_px))
        if radius_px <= 0:
            return mask
        kernel = 2 * radius_px + 1
        mask_f = mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        if dilate:
            out = F.max_pool2d(mask_f, kernel_size=kernel, stride=1, padding=radius_px)
            return out.squeeze(0).squeeze(0) > 0.5
        inv = 1.0 - mask_f
        eroded = 1.0 - F.max_pool2d(inv, kernel_size=kernel, stride=1, padding=radius_px)
        return eroded.squeeze(0).squeeze(0) > 0.5

    def _mask_bbox(self, mask: torch.Tensor) -> Tuple[int, int, int, int] | None:
        coords = torch.nonzero(mask, as_tuple=False)
        if coords.numel() == 0:
            return None
        y1 = int(coords[:, 0].min().item())
        y2 = int(coords[:, 0].max().item())
        x1 = int(coords[:, 1].min().item())
        x2 = int(coords[:, 1].max().item())
        return (x1, y1, x2, y2)

    def _apply_visibility_region_eot(self, mask: torch.Tensor) -> torch.Tensor:
        current = mask.to(dtype=torch.bool)
        if not self.enabled or not self._coin(self.cfg.visible_region_prob):
            return current
        original_area = int(current.sum().item())
        if original_area <= 0:
            return current
        if self.cfg.mask_erode_px > 0 and self._coin(0.5):
            current = self._morph_mask(current, self.cfg.mask_erode_px, dilate=False)
        elif self.cfg.mask_dilate_px > 0 and self._coin(0.35):
            current = self._morph_mask(current, self.cfg.mask_dilate_px, dilate=True)
        bbox = self._mask_bbox(current)
        if bbox is not None and self._coin(self.cfg.occ_prob):
            x1, y1, x2, y2 = bbox
            bw = max(1, x2 - x1 + 1)
            bh = max(1, y2 - y1 + 1)
            frac = self._uniform(0.04, self.cfg.occ_max_frac)
            occ_w = max(1, int(round(bw * frac)))
            occ_h = max(1, int(round(bh * frac)))
            ox1 = int(round(self._uniform(float(x1), float(max(x1, x2 - occ_w + 1)))))
            oy1 = int(round(self._uniform(float(y1), float(max(y1, y2 - occ_h + 1)))))
            current[oy1:oy1 + occ_h, ox1:ox1 + occ_w] = False
        if bbox is not None and self._coin(self.cfg.region_drop_prob):
            x1, y1, x2, y2 = bbox
            bw = max(1, x2 - x1 + 1)
            bh = max(1, y2 - y1 + 1)
            frac = self._uniform(0.03, self.cfg.region_drop_max_frac)
            drop_w = max(1, int(round(bw * frac)))
            drop_h = max(1, int(round(bh * frac)))
            dx1 = int(round(self._uniform(float(x1), float(max(x1, x2 - drop_w + 1)))))
            dy1 = int(round(self._uniform(float(y1), float(max(y1, y2 - drop_h + 1)))))
            current[dy1:dy1 + drop_h, dx1:dx1 + drop_w] = False
        kept_area = int(current.sum().item())
        if kept_area <= 0 or kept_area < int(0.18 * original_area):
            return mask.to(dtype=torch.bool)
        return current

    def apply(self, rendered: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if rendered.ndim != 3:
            raise ValueError('rendered must be [3,H,W]')
        if mask.ndim != 2:
            raise ValueError('mask must be [H,W]')
        current_image = rendered
        current_mask = mask.to(device=rendered.device, dtype=torch.bool)
        current_image = current_image * current_mask.to(dtype=current_image.dtype).unsqueeze(0)
        if not self.enabled:
            return (current_image.clamp(0.0, 1.0), current_mask)
        if self._coin(self.cfg.geometry_prob):
            current_image, current_mask = self._apply_affine(current_image, current_mask.to(dtype=current_image.dtype))
        if self._coin(self.cfg.photo_prob):
            current_image = self._apply_photometric(current_image, current_mask)
        if self._coin(self.cfg.noise_prob):
            current_image = self._apply_noise(current_image, current_mask)
        if self._coin(self.cfg.blur_prob):
            current_image = self._apply_blur(current_image, current_mask)
        current_mask = self._apply_visibility_region_eot(current_mask)
        current_image = current_image * current_mask.to(dtype=current_image.dtype).unsqueeze(0)
        return (current_image.clamp(0.0, 1.0), current_mask)

class FullImageAugmentor:

    def __init__(self, cfg: Dict[str, Any] | None, *, device: torch.device) -> None:
        self.cfg = FullImageAugmentConfig.from_dict(cfg)
        self.device = device

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    @property
    def apply_to_clean_views(self) -> bool:
        return bool(self.cfg.apply_to_clean_views)

    def summary(self) -> Dict[str, Any]:
        return {'enabled': bool(self.cfg.enabled), 'photometric_prob': float(self.cfg.photometric_prob), 'brightness': float(self.cfg.brightness), 'contrast': float(self.cfg.contrast), 'saturation': float(self.cfg.saturation), 'gamma': float(self.cfg.gamma), 'color_shift': float(self.cfg.color_shift), 'noise_prob': float(self.cfg.noise_prob), 'noise_std': float(self.cfg.noise_std), 'background_noise_prob': float(self.cfg.background_noise_prob), 'background_noise_std': float(self.cfg.background_noise_std), 'apply_to_clean_views': bool(self.cfg.apply_to_clean_views), 'background_noise_without_mask': bool(self.cfg.background_noise_without_mask), 'blur_prob': float(self.cfg.blur_prob), 'blur_kernel': int(self.cfg.blur_kernel), 'blur_sigma': float(self.cfg.blur_sigma), 'geometry': False}

    def _coin(self, probability: float) -> bool:
        probability = max(0.0, min(1.0, float(probability)))
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        return bool(torch.rand((), device=self.device).item() < probability)

    def _uniform(self, low: float, high: float) -> float:
        low_f = float(low)
        high_f = float(high)
        if abs(high_f - low_f) <= 1e-12:
            return low_f
        return float(torch.empty((), device=self.device).uniform_(low_f, high_f).item())

    def _apply_photometric(self, image: torch.Tensor) -> torch.Tensor:
        current = image
        if self.cfg.brightness > 0.0:
            brightness = 1.0 + self._uniform(-self.cfg.brightness, self.cfg.brightness)
            current = current * brightness
        if self.cfg.contrast > 0.0:
            contrast = 1.0 + self._uniform(-self.cfg.contrast, self.cfg.contrast)
            mean = current.mean(dim=(1, 2), keepdim=True)
            current = (current - mean) * contrast + mean
        if self.cfg.saturation > 0.0:
            saturation = 1.0 + self._uniform(-self.cfg.saturation, self.cfg.saturation)
            gray = current[0:1] * 0.2989 + current[1:2] * 0.587 + current[2:3] * 0.114
            current = (current - gray) * saturation + gray
        if self.cfg.gamma > 0.0:
            gamma = max(0.5, 1.0 + self._uniform(-self.cfg.gamma, self.cfg.gamma))
            current = current.clamp(0.0, 1.0).pow(gamma)
        if self.cfg.color_shift > 0.0:
            shift = torch.empty((3, 1, 1), device=current.device, dtype=current.dtype).uniform_(-self.cfg.color_shift, self.cfg.color_shift)
            current = current + shift
        return current

    def _apply_blur(self, image: torch.Tensor) -> torch.Tensor:
        if self.cfg.blur_kernel <= 1 or self.cfg.blur_sigma <= 0.0:
            return image
        kernel_2d = _gaussian_kernel_2d(self.cfg.blur_kernel, self.cfg.blur_sigma, device=image.device, dtype=image.dtype)
        kernel = kernel_2d.expand(image.shape[0], 1, kernel_2d.shape[0], kernel_2d.shape[1]).contiguous()
        padding = kernel_2d.shape[0] // 2
        return F.conv2d(image.unsqueeze(0), kernel, bias=None, stride=1, padding=padding, groups=image.shape[0]).squeeze(0)

    def apply(self, image: torch.Tensor, *, foreground_mask: torch.Tensor | None=None) -> torch.Tensor:
        if image.ndim != 3:
            raise ValueError('image must be [3,H,W]')
        if not self.enabled:
            return image.clamp(0.0, 1.0)
        current = image
        if self._coin(self.cfg.photometric_prob):
            current = self._apply_photometric(current)
        if self.cfg.noise_std > 0.0 and self._coin(self.cfg.noise_prob):
            current = current + torch.randn_like(current) * float(self.cfg.noise_std)
        if self.cfg.background_noise_std > 0.0 and self._coin(self.cfg.background_noise_prob):
            if foreground_mask is None:
                if self.cfg.background_noise_without_mask:
                    background_mask = torch.ones(current.shape[-2:], device=current.device, dtype=torch.bool)
                else:
                    background_mask = None
            else:
                background_mask = ~foreground_mask.to(device=current.device, dtype=torch.bool)
            if background_mask is not None and bool(background_mask.any()):
                noise = torch.randn_like(current) * float(self.cfg.background_noise_std)
                current = current + noise * background_mask.to(dtype=current.dtype).unsqueeze(0)
        if self._coin(self.cfg.blur_prob):
            current = self._apply_blur(current)
        return current.clamp(0.0, 1.0)
