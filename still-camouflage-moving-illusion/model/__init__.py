from __future__ import annotations
from typing import Any, Dict, Union
from .bevformer import BevFormerGradientModel, CAMERA_CHANNELS, CameraRecord, FixedQueryMatch, FrameRecord
from .bevdet import BevDetGradientModel
from .bevdepth import BevDepthGradientModel
from .fastbev import FastBEVGradientModel
GradientModelType = Union[BevFormerGradientModel, BevDetGradientModel, BevDepthGradientModel, FastBEVGradientModel]

def selected_model_name(config: Dict[str, Any]) -> str:
    raw = str(config.get('model', 'bevdet')).strip().lower()
    if raw in {'bevdet', 'bev_det'}:
        return 'bevdet'
    if raw in {'bevdepth', 'bev_depth'}:
        return 'bevdepth'
    if raw in {'stp3', 'st_p3', 'st-p3'}:
        raise ValueError('config.model=stp3 is not in this minimal bundle (no model/stp3.py). Use bevdet/bevdepth/fastbev.')
    if raw in {'bevformer', 'bev', 'detr3d', 'detr_3d', 'detr'}:
        raise ValueError(f'config.model={raw!r} is not supported in this bundle; use bevdet, bevdepth, or fastbev.')
    if raw in {'fastbev', 'fast_bev', 'fast-bev'}:
        return 'fastbev'
    raise ValueError(f"Unsupported config.model={raw!r}; expected 'bevdet', 'bevdepth', or 'fastbev'")

def selected_model_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    model_name = selected_model_name(config)
    model_cfg = config.get(model_name, {})
    if not isinstance(model_cfg, dict):
        raise ValueError(f'config.{model_name} must be a dict')
    return model_cfg

def build_gradient_model(*, config: Dict[str, Any], device: str='cuda', use_amp: bool=True, amp_dtype: str='fp16') -> GradientModelType:
    model_name = selected_model_name(config)
    if model_name == 'bevdet':
        return BevDetGradientModel(bevdet_cfg=selected_model_cfg(config), device=device, use_amp=use_amp, amp_dtype=amp_dtype)
    if model_name == 'bevdepth':
        return BevDepthGradientModel(bevdepth_cfg=selected_model_cfg(config), device=device, use_amp=use_amp, amp_dtype=amp_dtype)
    if model_name == 'fastbev':
        return FastBEVGradientModel(fastbev_cfg=selected_model_cfg(config), device=device, use_amp=use_amp, amp_dtype=amp_dtype)
    raise RuntimeError(f'Unsupported model {model_name!r}')
__all__ = ['CAMERA_CHANNELS', 'CameraRecord', 'FrameRecord', 'FixedQueryMatch', 'BevFormerGradientModel', 'BevDetGradientModel', 'BevDepthGradientModel', 'FastBEVGradientModel', 'GradientModelType', 'selected_model_name', 'selected_model_cfg', 'build_gradient_model']
