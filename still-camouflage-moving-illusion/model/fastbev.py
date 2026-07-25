from __future__ import annotations
from typing import Any, Dict
from .bevdet import BevDetGradientModel
from .bevformer import _as_path
_PRESET_TO_CONFIG = {'r50-cbgs': 'configs/fastbev/paper/fastbev-r50-cbgs.py', 'r50-cbgs-4d': 'configs/fastbev/paper/fastbev-r50-cbgs-4d.py', 'r101-cbgs-4d-longterm': 'configs/fastbev/paper/fastbev-r101-cbgs-4d-longterm.py'}

class FastBEVGradientModel(BevDetGradientModel):

    def __init__(self, *, fastbev_cfg: Dict[str, Any], device: str='cuda', use_amp: bool=True, amp_dtype: str='fp16'):
        repo_root = _as_path(str(fastbev_cfg.get('repo_root', '')))
        if not repo_root.exists():
            raise FileNotFoundError(f'FastBEV repo_root not found: {repo_root}')
        preset = str(fastbev_cfg.get('preset', 'r50-cbgs')).strip().lower() or 'r50-cbgs'
        if preset not in _PRESET_TO_CONFIG:
            supported = ', '.join(sorted(_PRESET_TO_CONFIG.keys()))
            raise ValueError(f'fastbev.preset unsupported: {preset!r}; supports: {supported}')
        raw_config_path = str(fastbev_cfg.get('config_path', '') or '').strip()
        raw_checkpoint_path = str(fastbev_cfg.get('checkpoint_path', '') or '').strip()
        config_path = _as_path(raw_config_path) if raw_config_path else (repo_root / _PRESET_TO_CONFIG[preset]).resolve()
        checkpoint_path = _as_path(raw_checkpoint_path) if raw_checkpoint_path else None
        if not config_path.exists():
            raise FileNotFoundError(f'FastBEV config not found: {config_path}')
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(f'FastBEV checkpoint not found. Set config.fastbev.checkpoint_path to a valid .pth (preset={preset}, repo_root={repo_root}).')
        bevdet_like_cfg: Dict[str, Any] = {'repo_root': str(repo_root), 'config_path': str(config_path), 'checkpoint_path': str(checkpoint_path), 'data_root': fastbev_cfg.get('data_root', ''), 'workers_per_gpu': int(fastbev_cfg.get('workers_per_gpu', 0)), 'key_camera': str(fastbev_cfg.get('key_camera', '') or ''), 'preset': 'r50'}
        super().__init__(bevdet_cfg=bevdet_like_cfg, device=device, use_amp=use_amp, amp_dtype=amp_dtype)
__all__ = ['FastBEVGradientModel']
