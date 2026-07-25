from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from match_target_car import _as_path, _load_yaml
from nusc_gt_to_mesh import run_mesh_projection
from sam2_mask import run_sam2_mask
from train import _mesh_obj_path_from_config, _near_plane_from_config, _prepare_precompute_outputs, _sam2_checkpoint_from_config, _sam2_repo_from_config, run_training
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONF_PATH = PROJECT_ROOT / 'conf.yaml'
DEFAULT_CONFIG_PATH = PROJECT_ROOT / 'config.yaml'

def _resolve_config_path(user_path: Optional[Path]) -> Path:
    if user_path is not None:
        return _as_path(user_path)
    if DEFAULT_CONF_PATH.exists():
        return DEFAULT_CONF_PATH.resolve()
    return DEFAULT_CONFIG_PATH.resolve()

def _collect_visual_outputs(*, mesh_summary: Dict[str, Any], sam_summary: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    visuals_root = output_root / 'visuals'
    mesh_dir = visuals_root / 'mesh_mask'
    sam_dir = visuals_root / 'sam2_mask'
    final_dir = visuals_root / 'final'
    mesh_dir.mkdir(parents=True, exist_ok=True)
    sam_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    mesh_records = mesh_summary.get('records', []) if isinstance(mesh_summary, dict) else []
    sam_records = sam_summary.get('records', []) if isinstance(sam_summary, dict) else []
    mesh_by_key: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    for row in mesh_records:
        if not isinstance(row, dict):
            continue
        key = (int(row.get('frame_id', -1)), str(row.get('channel', '')), str(row.get('sample_token', '')))
        mesh_by_key[key] = row
    exported_records: List[Dict[str, Any]] = []
    for row in sam_records:
        if not isinstance(row, dict):
            continue
        key = (int(row.get('frame_id', -1)), str(row.get('channel', '')), str(row.get('sample_token', '')))
        mesh_row = mesh_by_key.get(key, {})
        stem = Path(str(row.get('final_path', 'record.png'))).stem
        src_mesh_mask = Path(str(mesh_row.get('mesh_mask_path', ''))).expanduser()
        src_sam_mask = Path(str(row.get('sam_mask_path', ''))).expanduser()
        src_final = Path(str(row.get('final_path', ''))).expanduser()
        if not src_mesh_mask.exists() or not src_sam_mask.exists() or (not src_final.exists()):
            continue
        dst_mesh_mask = mesh_dir / f'{stem}.png'
        dst_sam_mask = sam_dir / f'{stem}.png'
        dst_final = final_dir / f'{stem}.png'
        shutil.copy2(src_mesh_mask, dst_mesh_mask)
        shutil.copy2(src_sam_mask, dst_sam_mask)
        shutil.copy2(src_final, dst_final)
        exported_records.append({'frame_id': int(row.get('frame_id', -1)), 'channel': str(row.get('channel', '')), 'mesh_mask_path': str(dst_mesh_mask), 'sam_mask_path': str(dst_sam_mask), 'final_path': str(dst_final)})
    return {'visuals_root': visuals_root, 'mesh_mask_dir': mesh_dir, 'sam2_mask_dir': sam_dir, 'final_dir': final_dir, 'records': exported_records}

def run_pipeline(*, config_path: Path, output_root: Optional[Path]=None, sam2_repo: Optional[Path], sam2_checkpoint: Optional[Path], device: Optional[str], near_plane_m: Optional[float]) -> Dict[str, Any]:
    config_path = _as_path(config_path)
    config = _load_yaml(config_path)
    if output_root is None:
        output_root = config_path.parent / 'result' / 'pipeline'
    output_root = _as_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_sam2_repo = _sam2_repo_from_config(config) if sam2_repo is None else _as_path(sam2_repo)
    resolved_sam2_checkpoint = _sam2_checkpoint_from_config(config) if sam2_checkpoint is None else _as_path(sam2_checkpoint)
    resolved_device = str(device).strip() if device is not None else str(config.get('train', {}).get('device', 'cuda')).strip()
    resolved_near_plane_m = float(near_plane_m) if near_plane_m is not None else _near_plane_from_config(config)
    train_cfg = config.get('train', {})
    if not isinstance(train_cfg, dict):
        train_cfg = {}
        config['train'] = train_cfg
    render_cfg = config.get('render', {})
    if not isinstance(render_cfg, dict):
        render_cfg = {}
        config['render'] = render_cfg
    sam2_cfg = config.get('sam2', {})
    if not isinstance(sam2_cfg, dict):
        sam2_cfg = {}
        config['sam2'] = sam2_cfg
    train_cfg['device'] = resolved_device
    render_cfg['near_plane_m'] = resolved_near_plane_m
    sam2_cfg['repo_root'] = str(resolved_sam2_repo)
    sam2_cfg['checkpoint_path'] = str(resolved_sam2_checkpoint)
    binding_payload, mesh_summary, sam_summary = _prepare_precompute_outputs(config=config, config_path=config_path, output_root=output_root)
    exported = _collect_visual_outputs(mesh_summary=mesh_summary, sam_summary=sam_summary, output_root=output_root)
    pipeline_summary: Dict[str, Any] = {'config_path': str(config_path), 'output_root': str(output_root), 'sequence_yaml': None, 'binding_yaml': None, 'bound_sequence_yaml': None, 'mesh_summary_yaml': None, 'sam2_summary_yaml': None, 'mesh_mask_dir': str(exported.get('mesh_mask_dir')), 'sam2_mask_dir': str(exported.get('sam2_mask_dir')), 'final_dir': str(exported.get('final_dir')), 'rendered_view_count': int(mesh_summary.get('rendered_view_count', 0)) if isinstance(mesh_summary, dict) else 0, 'sam_processed_view_count': int(sam_summary.get('processed_view_count', 0)) if isinstance(sam_summary, dict) else 0}
    print(f"[main] mesh_mask_dir={exported.get('mesh_mask_dir')}")
    print(f"[main] sam2_mask_dir={exported.get('sam2_mask_dir')}")
    print(f"[main] final_dir={exported.get('final_dir')}")
    return pipeline_summary

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Default entry: run training. Optional visual mode exports only mesh/SAM2/final images.')
    parser.add_argument('--config', type=Path, default=None, help='conf.yaml or config.yaml path')
    parser.add_argument('--ckpt', dest='resume_ckpt', type=Path, default=None, help='Resume from checkpoint; pass step-xxxx.pt/last.pt or its directory')
    parser.add_argument('--mode', type=str, choices=('train', 'visual'), default='train', help='train: run train.py pipeline; visual: only export mesh/SAM2/final images')
    parser.add_argument('--output-root', type=Path, default=None, help='Pipeline output root')
    parser.add_argument('--sam2-repo', type=Path, default=None, help='Local SAM2 repo; defaults to sam2.repo_root in config.yaml')
    parser.add_argument('--sam2-checkpoint', type=Path, default=None, help='SAM2 checkpoint; defaults to sam2.checkpoint_path in config.yaml')
    parser.add_argument('--device', type=str, default=None, help='Device: auto/cpu/cuda; defaults to train.device in config.yaml')
    parser.add_argument('--near-plane', type=float, default=None, help='Near plane in meters; defaults to render.near_plane_m in config.yaml')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config_path = _resolve_config_path(args.config)
    if args.mode == 'train':
        summary = run_training(config_path, resume_ckpt=args.resume_ckpt)
        print(f"[main] train_output_dir={summary.get('output_dir')}")
        print(f"[main] latest_step_ckpt_path={summary.get('latest_step_ckpt_path')}")
        print(f"[main] last_ckpt_path={summary.get('last_ckpt_path')}")
        print(f"[main] before_visual_dir={summary.get('before_visual_dir')}")
        print(f"[main] after_visual_dir={summary.get('after_visual_dir')}")
        print(f"[main] official_results_path={summary.get('official_results_path')}")
        print(f"[main] official_visual_dir={summary.get('official_visual_dir')}")
        if summary.get('clean_results_by_sequence'):
            print(f"[main] clean_results_by_sequence={summary.get('clean_results_by_sequence')}")
            print(f"[main] clean_visuals_by_sequence={summary.get('clean_visuals_by_sequence')}")
        if summary.get('official_results_by_sequence'):
            print(f"[main] official_results_by_sequence={summary.get('official_results_by_sequence')}")
            print(f"[main] official_visuals_by_sequence={summary.get('official_visuals_by_sequence')}")
        print(f"[main] training_log_path={summary.get('training_log_path')}")
        return
    run_pipeline(config_path=config_path, output_root=args.output_root, sam2_repo=args.sam2_repo, sam2_checkpoint=args.sam2_checkpoint, device=args.device, near_plane_m=float(args.near_plane) if args.near_plane is not None else None)
if __name__ == '__main__':
    main()
