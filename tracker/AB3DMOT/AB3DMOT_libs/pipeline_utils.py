import copy
import os
import shutil
from pathlib import Path

import numpy as np
import yaml
try:
    from easydict import EasyDict as edict
except ImportError:
    class edict(dict):
        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError:
                raise AttributeError(item)

        def __setattr__(self, key, value):
            self[key] = value

        def __delattr__(self, item):
            try:
                del self[item]
            except KeyError:
                raise AttributeError(item)

from AB3DMOT_libs.utils import Config


def to_plain_dict(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, dict):
        return {key: to_plain_dict(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [to_plain_dict(value) for value in data]
    return data


def _as_path(value, base_dir):
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_pipeline_config(config_path):
    config_path = Path(config_path).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    official_cfg, _ = Config(str(repo_root / "configs" / "nuScenes.yml"))
    with open(config_path, "r") as file_obj:
        raw_cfg = yaml.safe_load(file_obj) or {}

    cfg = edict(copy.deepcopy(official_cfg))
    cfg.repo_root = str(repo_root)
    cfg.config_path = str(config_path)
    cfg.dataset = raw_cfg.get("dataset", cfg.dataset)
    cfg.split = raw_cfg.get("split", cfg.split)
    tracking_cfg = raw_cfg.get("tracking", {})
    cfg.det_name = raw_cfg.get("det_name", "bevformer")
    cfg.tracker_preset = raw_cfg.get(
        "tracker_preset", tracking_cfg.get("tracker_preset", "centerpoint")
    )

    tracking_category = tracking_cfg.get("category")
    tracking_cat_list = tracking_cfg.get("cat_list")
    if tracking_cat_list is not None:
        cfg.cat_list = tracking_cat_list
    elif tracking_category:
        cfg.cat_list = [tracking_category]
    else:
        cfg.cat_list = list(cfg.cat_list)
    cfg.num_hypo = int(tracking_cfg.get("num_hypo", cfg.num_hypo))
    cfg.score_threshold = float(
        tracking_cfg.get("output_score_threshold", cfg.score_threshold)
    )
    cfg.ego_com = bool(tracking_cfg.get("ego_com", cfg.ego_com))
    cfg.vis = bool(tracking_cfg.get("vis", cfg.vis))
    cfg.affi_pro = bool(tracking_cfg.get("affi_pro", cfg.affi_pro))
    cfg.input_detection_score_threshold = tracking_cfg.get(
        "input_detection_score_threshold"
    )
    if cfg.input_detection_score_threshold is not None:
        cfg.input_detection_score_threshold = float(
            cfg.input_detection_score_threshold
        )
    cfg.tracker_overrides = edict(
        {
            "default": tracking_cfg.get("overrides", {}).get("default", {}) or {},
            "by_category": tracking_cfg.get("overrides", {}).get("by_category", {})
            or {},
        }
    )

    paths_cfg = raw_cfg.get("paths", {})
    cfg.paths = edict()
    bevformer_json = paths_cfg.get("bevformer_json")
    cfg.paths.bevformer_json = (
        str(_as_path(bevformer_json, config_path.parent)) if bevformer_json else ""
    )
    info_pkl = paths_cfg.get("samples_info_pkl", paths_cfg.get("info_pkl"))
    cfg.paths.info_pkl = (
        str(_as_path(info_pkl, config_path.parent)) if info_pkl else ""
    )
    cfg.paths.nuscenes_data_root = str(
        _as_path(
            paths_cfg.get(
                "nuscenes_data_root", repo_root / "data" / "nuScenes" / "data"
            ),
            config_path.parent,
        )
    )
    cfg.paths.nusc_kitti_root = str(
        _as_path(
            paths_cfg.get(
                "nusc_kitti_root", repo_root / "data" / "nuScenes" / "nuKITTI"
            ),
            config_path.parent,
        )
    )
    workspace_dir = paths_cfg.get("workspace_dir", paths_cfg.get("internal_results_root"))
    cfg.paths.workspace_dir = str(
        _as_path(
            workspace_dir or (repo_root / "results" / cfg.dataset),
            config_path.parent,
        )
    )
    cfg.paths.internal_results_root = cfg.paths.workspace_dir
    cfg.paths.output_dir = str(
        _as_path(
            paths_cfg.get("output_dir", repo_root / "result" / "output"),
            config_path.parent,
        )
    )
    cfg.paths.visual_dir = str(
        _as_path(
            paths_cfg.get("visual_dir", repo_root / "result" / "visual"),
            config_path.parent,
        )
    )
    output_json = paths_cfg.get("output_json")
    cfg.paths.output_json = (
        str(_as_path(output_json, config_path.parent)) if output_json else ""
    )
    detection_root = paths_cfg.get("detection_root")
    cfg.paths.detection_root = str(
        _as_path(
            detection_root or (Path(cfg.paths.workspace_dir) / "detection"),
            config_path.parent,
        )
    )
    cfg.paths.official_nuscenes_data = cfg.paths.nuscenes_data_root
    cfg.paths.official_nukitti = cfg.paths.nusc_kitti_root

    prepare_cfg = raw_cfg.get("prepare", {})
    cfg.prepare = edict(
        {
            "relink_official_paths": bool(
                prepare_cfg.get("relink_official_paths", True)
            ),
            "auto_prepare_tracking_data": bool(
                prepare_cfg.get("auto_prepare_tracking_data", False)
            ),
            "auto_prepare_object_data": bool(
                prepare_cfg.get("auto_prepare_object_data", False)
            ),
            "force_reconvert_detection": bool(
                prepare_cfg.get("force_reconvert_detection", False)
            ),
            "force_rerun_tracking": bool(
                prepare_cfg.get("force_rerun_tracking", False)
            ),
            "overwrite_output": bool(prepare_cfg.get("overwrite_output", True)),
            "use_json_subset": bool(prepare_cfg.get("use_json_subset", True)),
            "subset_source": str(prepare_cfg.get("subset_source", "auto")).lower(),
            "sync_workspace_output": bool(
                prepare_cfg.get("sync_workspace_output", False)
            ),
        }
    )

    visual_cfg = raw_cfg.get("visual", {})
    cfg.visualization = edict(
        {
            "enabled": bool(visual_cfg.get("enabled", True)),
            "hypo_index": int(visual_cfg.get("hypo_index", 0)),
            "min_score": visual_cfg.get("min_score"),
            "min_score_by_category": visual_cfg.get("min_score_by_category", {}) or {},
            "color_type": str(visual_cfg.get("color_type", "trk")),
            "render_video": bool(visual_cfg.get("render_video", True)),
            "framerate": int(visual_cfg.get("framerate", 2)),
            "overwrite": bool(visual_cfg.get("overwrite", True)),
        }
    )
    if cfg.visualization.min_score is not None:
        cfg.visualization.min_score = float(cfg.visualization.min_score)

    default_run_name = f"{cfg.det_name}_{cfg.cat_list[0].lower()}_{cfg.split}"
    if raw_cfg.get("run_name"):
        cfg.run_name = raw_cfg.get("run_name")
    elif cfg.paths.output_json:
        cfg.run_name = Path(cfg.paths.output_json).stem
    else:
        cfg.run_name = default_run_name

    if not cfg.paths.output_json:
        cfg.paths.output_json = str(
            (_as_path(cfg.paths.output_dir, config_path.parent) / f"{cfg.run_name}.json")
        )
    cfg.save_root = cfg.paths.workspace_dir

    return cfg


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_symlink(link_path, target_path, allow_replace=False):
    link_path = Path(link_path)
    target_path = Path(target_path)
    ensure_dir(link_path.parent)

    # 官方路径和配置路径相同时，优先把它当真实目录处理。
    if link_path.resolve().as_posix() == target_path.resolve().as_posix():
        if link_path.is_symlink():
            if link_path.exists():
                return
            if not allow_replace:
                raise RuntimeError(
                    f"官方路径 {link_path} 是一个失效软链接。"
                    "请把 prepare.relink_official_paths 设为 true，或者手动删除后重试。"
                )
            link_path.unlink()
        ensure_dir(link_path)
        return

    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink():
            current_target = os.path.realpath(str(link_path))
            desired_target = os.path.realpath(str(target_path))
            if current_target == desired_target:
                return
            if not allow_replace:
                raise RuntimeError(
                    f"官方路径 {link_path} 当前指向 {current_target}，"
                    f"而不是期望的 {desired_target}。请把 "
                    "prepare.relink_official_paths 设为 true，或者修改配置。"
                )
            link_path.unlink()
        else:
            current_target = os.path.realpath(str(link_path))
            desired_target = os.path.realpath(str(target_path))
            if current_target == desired_target:
                return
            raise RuntimeError(
                f"官方路径 {link_path} 已存在且不是软链接。"
                f"如果你想改用 {target_path}，请先手动处理这个目录。"
            )

    link_path.symlink_to(target_path)


def ensure_official_dataset_links(cfg):
    ensure_symlink(
        cfg.paths.official_nuscenes_data,
        cfg.paths.nuscenes_data_root,
        allow_replace=cfg.prepare.relink_official_paths,
    )
    ensure_symlink(
        cfg.paths.official_nukitti,
        cfg.paths.nusc_kitti_root,
        allow_replace=cfg.prepare.relink_official_paths,
    )


def clear_path(path):
    path = Path(path)
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def copytree_overwrite(src, dst):
    src = Path(src)
    dst = Path(dst)
    clear_path(dst)
    ensure_dir(dst.parent)
    shutil.copytree(src, dst)


def visual_threshold_for_category(cfg, category):
    by_category = cfg.visualization.min_score_by_category or {}
    if category in by_category:
        return float(by_category[category])
    if cfg.visualization.min_score is None:
        return None
    return float(cfg.visualization.min_score)
