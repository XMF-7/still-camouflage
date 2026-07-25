# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
#
# This entrypoint keeps the official AB3DMOT tracking code path intact and
# adds a thin config-driven wrapper for BEVFormer nuScenes detection JSON.

from __future__ import print_function

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
XINSHUO_ROOT = REPO_ROOT / "Xinshuo_PyToolbox"
for _path in [REPO_ROOT, XINSHUO_ROOT]:
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")

from AB3DMOT_libs.io import (
    combine_files,
    get_frame_det,
    get_saving_dir,
    load_detection,
    save_affinity,
    save_results,
)
from AB3DMOT_libs.nuscenes_subset import (
    build_subset_context,
    export_subset_tracking_json,
    has_info_subset,
    prepare_subset_detection_inputs,
    prepare_subset_object_correspondence,
    prepare_subset_tracking_data,
)
from AB3DMOT_libs.pipeline_utils import (
    clear_path,
    copytree_overwrite,
    ensure_dir,
    ensure_official_dataset_links,
    load_pipeline_config,
    to_plain_dict,
)
from AB3DMOT_libs.utils import get_subfolder_seq, initialize
from xinshuo_io import load_list_from_folder, mkdir_if_missing
from xinshuo_miscellaneous import get_timestring, print_log


DISTANCE_METRICS = {"dist_3d", "dist_2d", "m_dis"}
SUPPORTED_TRACKER_PRESETS = {"centerpoint", "megvii"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用官方 AB3DMOT 对 BEVFormer 的 nuScenes 检测结果进行跟踪。支持直接提供 detection JSON。"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径。",
    )
    parser.add_argument(
        "--detection_json",
        type=str,
        default=None,
        help="检测结果 JSON 路径（nuScenes 格式）。如果提供，则覆盖 config 中的 bevformer_json 并使用子集模式。",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="输出的 tracking JSON 路径。如果未提供，将基于 detection_json 生成。",
    )
    return parser.parse_args()


def validate_config(cfg):
    if cfg.dataset != "nuScenes":
        raise ValueError("当前封装只支持 dataset=nuScenes。")
    if cfg.tracker_preset not in SUPPORTED_TRACKER_PRESETS:
        raise ValueError(
            f"tracker_preset 只能是 {sorted(SUPPORTED_TRACKER_PRESETS)} 之一，"
            f"当前为 {cfg.tracker_preset}。"
        )
    if cfg.num_hypo != 1:
        raise ValueError(
            "当前仓库配置下无法正常启用官方多假设跟踪，请保持 tracking.num_hypo=1。"
        )

    bevformer_json = Path(cfg.paths.bevformer_json)
    if not bevformer_json.is_file():
        raise FileNotFoundError(
            "paths.bevformer_json 指向的文件不存在："
            f"{cfg.paths.bevformer_json}"
        )
    if getattr(cfg.paths, "info_pkl", ""):
        info_pkl = Path(cfg.paths.info_pkl)
        if not info_pkl.is_file():
            raise FileNotFoundError(
                "paths.samples_info_pkl / paths.info_pkl 指向的文件不存在："
                f"{cfg.paths.info_pkl}"
            )


def set_tracker_metric_bounds(tracker):
    if tracker.metric in DISTANCE_METRICS:
        tracker.max_sim, tracker.min_sim = 0.0, -100.0
    elif tracker.metric in {"iou_2d", "iou_3d"}:
        tracker.max_sim, tracker.min_sim = 1.0, 0.0
    elif tracker.metric in {"giou_2d", "giou_3d"}:
        tracker.max_sim, tracker.min_sim = 1.0, -1.0
    else:
        raise ValueError(f"不支持的 tracker metric 覆盖项：{tracker.metric}")


def apply_tracker_overrides(tracker, cfg, cat, log):
    overrides = {}
    overrides.update(cfg.tracker_overrides.get("default", {}) or {})
    overrides.update(cfg.tracker_overrides.get("by_category", {}).get(cat, {}) or {})
    if not overrides:
        return

    if overrides.get("algm") is not None:
        tracker.algm = overrides["algm"]
    if overrides.get("metric") is not None:
        tracker.metric = overrides["metric"]
    if overrides.get("match_threshold") is not None:
        threshold = float(overrides["match_threshold"])
        if tracker.metric in DISTANCE_METRICS:
            threshold = -abs(threshold)
        tracker.thres = threshold
    elif overrides.get("thres") is not None:
        tracker.thres = float(overrides["thres"])
    if overrides.get("max_age") is not None:
        tracker.max_age = int(overrides["max_age"])
    if overrides.get("min_hits") is not None:
        tracker.min_hits = int(overrides["min_hits"])
    if overrides.get("affi_pro") is not None:
        tracker.affi_process = bool(overrides["affi_pro"])

    set_tracker_metric_bounds(tracker)
    tracker.print_param()
    print_log(
        f"已为类别 {cat} 应用跟踪参数覆盖：{overrides}",
        log=log,
        display=False,
    )


def get_runtime_tracker_cfg(cfg):
    runtime_cfg = load_pipeline_config(cfg.config_path)
    subset_context = getattr(cfg, "subset_context", None)
    if subset_context is not None:
        runtime_cfg.subset_context = subset_context
    runtime_cfg.det_name = cfg.tracker_preset
    return runtime_cfg


def get_sequence_eval_list(cfg, default_seq_eval):
    if has_info_subset(cfg):
        context = build_subset_context(cfg)
        return list(context["scene_names"])
    return list(default_seq_eval)


def ensure_tracking_data(cfg):
    if has_info_subset(cfg):
        print("检测到子集模式，正在按 BEVFormer json / samples 信息准备最小 tracking 数据...")
        prepare_subset_tracking_data(cfg, build_subset_context(cfg))
        return

    nusc_kitti_root = Path(cfg.paths.nusc_kitti_root)
    required_paths = [
        nusc_kitti_root / "tracking" / cfg.split / "calib",
        nusc_kitti_root / "tracking" / cfg.split / "image_02",
        nusc_kitti_root / "tracking" / cfg.split / "oxts",
        nusc_kitti_root / "tracking" / "produced" / "correspondence" / cfg.split,
    ]
    if all(path.exists() for path in required_paths):
        return
    if not cfg.prepare.auto_prepare_tracking_data:
        raise FileNotFoundError(
            "缺少 nuKITTI tracking 数据。请先运行官方脚本 "
            f"`python scripts/nuScenes/export_kitti.py nuscenes_gt2kitti_trk --split {cfg.split}`，"
            "或者把 prepare.auto_prepare_tracking_data 设为 true。"
        )

    from scripts.nuScenes.export_kitti import KittiConverter

    converter = KittiConverter(
        nusc_kitti_root=cfg.paths.nusc_kitti_root,
        data_root=cfg.paths.nuscenes_data_root,
        split=cfg.split,
    )
    converter.nuscenes_gt2kitti_trk()


def ensure_object_correspondence(cfg):
    if has_info_subset(cfg):
        prepare_subset_object_correspondence(cfg, build_subset_context(cfg))
        return

    nusc_kitti_root = Path(cfg.paths.nusc_kitti_root)
    produced_corr = (
        nusc_kitti_root / "object" / "produced" / "correspondence" / f"{cfg.split}.txt"
    )
    if produced_corr.exists():
        return

    source_corr = nusc_kitti_root / "object" / cfg.split / "correspondence.txt"
    if not source_corr.exists():
        if not cfg.prepare.auto_prepare_object_data:
            raise FileNotFoundError(
                "缺少 nuKITTI object correspondence。请先运行官方脚本 "
                "`python scripts/nuScenes/export_kitti.py "
                f"nuscenes_gt2kitti_obj --split {cfg.split}`，"
                "或者把 prepare.auto_prepare_object_data 设为 true。"
            )

        from scripts.nuScenes.export_kitti import KittiConverter

        converter = KittiConverter(
            nusc_kitti_root=cfg.paths.nusc_kitti_root,
            data_root=cfg.paths.nuscenes_data_root,
            split=cfg.split,
        )
        converter.nuscenes_gt2kitti_obj()

    ensure_dir(produced_corr.parent)
    shutil.copy2(source_corr, produced_corr)


def stage_detection_json(cfg):
    source_json = Path(cfg.paths.bevformer_json)
    target_json = (
        Path(cfg.paths.workspace_dir)
        / "staged"
        / "detection"
        / cfg.det_name
        / f"results_{cfg.split}.json"
    )
    ensure_dir(target_json.parent)

    with open(source_json, "r") as file_obj:
        raw_data = json.load(file_obj)

    if isinstance(raw_data, dict) and "results" in raw_data:
        results = raw_data["results"]
        meta = raw_data.get("meta", {})
    elif isinstance(raw_data, dict):
        results = raw_data
        meta = {}
    else:
        raise ValueError(
            "不支持当前的 BEVFormer JSON 格式，应为字典，或者 "
            "`{'results': ...}` 这样的结构。"
        )

    score_threshold = cfg.input_detection_score_threshold
    staged_results = {}
    for sample_token, dets in results.items():
        filtered = []
        for det in dets:
            det_copy = dict(det)
            if "tracking_name" in det_copy and "detection_name" not in det_copy:
                det_copy["detection_name"] = det_copy["tracking_name"]
            if "tracking_score" in det_copy and "detection_score" not in det_copy:
                det_copy["detection_score"] = det_copy["tracking_score"]

            if score_threshold is not None:
                score = det_copy.get("detection_score")
                if score is not None and float(score) < score_threshold:
                    continue
            filtered.append(det_copy)
        staged_results[sample_token] = filtered

    with open(target_json, "w") as file_obj:
        json.dump({"meta": meta, "results": staged_results}, file_obj, indent=2)

    return target_json


def remove_previous_detection_artifacts(cfg):
    detection_root = Path(getattr(cfg.paths, "detection_root", "")) if getattr(cfg.paths, "detection_root", "") else Path(cfg.repo_root) / "data" / cfg.dataset / "detection"
    for path in detection_root.glob(f"{cfg.det_name}_*_{cfg.split}"):
        clear_path(path)

    object_results_root = (
        Path(cfg.paths.nusc_kitti_root)
        / "object"
        / "produced"
        / "results"
        / cfg.split
        / cfg.det_name
    )
    clear_path(object_results_root)


def prepare_detection_inputs(cfg):
    staged_json = stage_detection_json(cfg)
    if has_info_subset(cfg):
        remove_previous_detection_artifacts(cfg)
        print("检测到子集模式，正在只为指定样本生成检测输入...")
        prepare_subset_detection_inputs(cfg, build_subset_context(cfg), staged_json)
        return staged_json

    detection_root = Path(getattr(cfg.paths, "detection_root", "")) if getattr(cfg.paths, "detection_root", "") else Path(cfg.repo_root) / "data" / cfg.dataset / "detection"
    expected_dir = detection_root / f"{cfg.det_name}_{cfg.cat_list[0]}_{cfg.split}"
    expected_ready = expected_dir.exists() and any(expected_dir.glob("*.txt"))
    if expected_ready and not cfg.prepare.force_reconvert_detection:
        return staged_json

    remove_previous_detection_artifacts(cfg)

    from scripts.nuScenes.export_kitti import KittiConverter
    from scripts.pre_processing.convert_det2input import combine_dets

    converter = KittiConverter(
        nusc_kitti_root=cfg.paths.nusc_kitti_root,
        data_root=cfg.paths.nuscenes_data_root,
        result_name=cfg.det_name,
        split=cfg.split,
    )
    converter.nuscenes_obj_result2kitti()
    combine_dets(cfg.dataset, cfg.split, cfg.det_name)
    return staged_json


def result_sha_for_cat(cfg, cat):
    return f"{cfg.det_name}_{cat}_{cfg.split}"


def clear_previous_tracking_results(cfg):
    for cat in cfg.cat_list:
        clear_path(Path(cfg.save_root) / f"{result_sha_for_cat(cfg, cat)}_H{cfg.num_hypo}")
    clear_path(Path(cfg.save_root) / cfg.run_name)


def main_per_cat(cfg, cat, log, id_start):
    result_sha = result_sha_for_cat(cfg, cat)
    det_root_base = Path(getattr(cfg.paths, "detection_root", "")) if getattr(cfg.paths, "detection_root", "") else Path("./data") / cfg.dataset / "detection"
    det_root = det_root_base / result_sha
    subfolder, det_id2str, hw, seq_eval_default, data_root = get_subfolder_seq(
        cfg.dataset, cfg.split
    )
    data_root = str(Path(cfg.paths.nusc_kitti_root)) if getattr(cfg.paths, "nusc_kitti_root", "") else data_root
    seq_eval = get_sequence_eval_list(cfg, seq_eval_default)
    trk_root = os.path.join(data_root, "tracking")
    save_dir = os.path.join(cfg.save_root, f"{result_sha}_H{cfg.num_hypo}")
    mkdir_if_missing(save_dir)

    eval_dir_dict = {}
    for index in range(cfg.num_hypo):
        eval_dir = os.path.join(save_dir, f"data_{index}")
        mkdir_if_missing(eval_dir)
        eval_dir_dict[index] = eval_dir

    seq_count = 0
    total_time, total_frames = 0.0, 0
    last_tracker = None
    runtime_cfg = get_runtime_tracker_cfg(cfg)

    for seq_name in seq_eval:
        seq_file = det_root / f"{seq_name}.txt"
        if not seq_file.is_file():
            continue

        seq_dets, flag = load_detection(str(seq_file))
        if not flag:
            continue

        eval_file_dict, save_trk_dir, affinity_dir, affinity_vis = get_saving_dir(
            eval_dir_dict, seq_name, save_dir, cfg.num_hypo
        )

        tracker, frame_list = initialize(
            runtime_cfg, trk_root, save_dir, subfolder, seq_name, cat, id_start, hw, log
        )
        apply_tracker_overrides(tracker, cfg, cat, log)
        last_tracker = tracker

        min_frame, max_frame = int(frame_list[0]), int(frame_list[-1])
        for frame in range(min_frame, max_frame + 1):
            print_str = (
                f"正在处理 {result_sha} {seq_name}："
                f"序列 {seq_count}/{len(seq_eval)}，帧 {frame}/{max_frame}   \r"
            )
            sys.stdout.write(print_str)
            sys.stdout.flush()

            dets_frame = get_frame_det(seq_dets, frame)
            since = time.time()
            results, affi = tracker.track(dets_frame, frame, seq_name)
            total_time += time.time() - since

            save_affi_file = os.path.join(affinity_dir, f"{frame:06d}.npy")
            save_affi_vis = os.path.join(affinity_vis, f"{frame:06d}.txt")
            if (affi is not None) and (affi.shape[0] + affi.shape[1] > 0):
                np.save(save_affi_file, affi)
                if affi.shape[0] > 0 and affi.shape[1] > 0:
                    save_affinity(affi, save_affi_vis)

            for hypo in range(cfg.num_hypo):
                save_trk_path = os.path.join(save_trk_dir[hypo], f"{frame:06d}.txt")
                with open(save_trk_path, "w") as save_trk_file:
                    for result_tmp in results[hypo]:
                        save_results(
                            result_tmp,
                            save_trk_file,
                            eval_file_dict[hypo],
                            det_id2str,
                            frame,
                            cfg.score_threshold,
                        )

            total_frames += 1
        seq_count += 1

        for index in range(cfg.num_hypo):
            eval_file_dict[index].close()
            id_start = max(id_start, tracker.ID_count[index])

    if last_tracker is None:
        print_log(
            f"{cfg.dataset} {result_sha}：未找到检测文件，已跳过。",
            log=log,
        )
        return id_start

    fps = total_frames / total_time if total_time > 0 else 0.0
    print_log(
        "%s, %25s: 共处理 %5d 帧，耗时 %4.f 秒，速度 %6.1f FPS，当前度量 %s = %.2f"
        % (
            cfg.dataset,
            result_sha,
            total_frames,
            total_time,
            fps,
            last_tracker.metric,
            last_tracker.thres,
        ),
        log=log,
    )
    return id_start


def combine_tracking_results(cfg):
    _, _, _, seq_eval_default, _ = get_subfolder_seq(cfg.dataset, cfg.split)
    seq_list = get_sequence_eval_list(cfg, seq_eval_default)
    subset = [f"{result_sha_for_cat(cfg, cat)}_H{cfg.num_hypo}" for cat in cfg.cat_list]
    run_root = Path(cfg.save_root) / cfg.run_name
    clear_path(run_root)
    ensure_dir(run_root)

    for hypo_index in range(cfg.num_hypo):
        data_suffix = f"_{hypo_index}"

        eval_save_root = run_root / f"data{data_suffix}"
        ensure_dir(eval_save_root)
        for seq_tmp in seq_list:
            file_list = []
            for subset_tmp in subset:
                file_tmp = Path(cfg.save_root) / subset_tmp / f"data{data_suffix}" / f"{seq_tmp}.txt"
                if file_tmp.exists():
                    file_list.append(str(file_tmp))
            if file_list:
                combine_files(file_list, str(eval_save_root / f"{seq_tmp}.txt"))
            else:
                (eval_save_root / f"{seq_tmp}.txt").touch()

        trk_save_root = run_root / f"trk_withid{data_suffix}"
        ensure_dir(trk_save_root)
        for seq_tmp in seq_list:
            frame_names = set()
            for subset_tmp in subset:
                seq_dir = Path(cfg.save_root) / subset_tmp / f"trk_withid{data_suffix}" / seq_tmp
                if not seq_dir.exists():
                    continue
                frame_list, _ = load_list_from_folder(str(seq_dir))
                frame_names.update(Path(frame_file).stem for frame_file in frame_list)

            if not frame_names:
                continue

            save_seq_dir = trk_save_root / seq_tmp
            ensure_dir(save_seq_dir)
            for frame_name in sorted(frame_names):
                file_list = []
                for subset_tmp in subset:
                    frame_file = (
                        Path(cfg.save_root)
                        / subset_tmp
                        / f"trk_withid{data_suffix}"
                        / seq_tmp
                        / f"{frame_name}.txt"
                    )
                    if frame_file.exists():
                        file_list.append(str(frame_file))
                if file_list:
                    combine_files(
                        file_list,
                        str(save_seq_dir / f"{frame_name}.txt"),
                        sort=False,
                    )


def sync_final_output(cfg, config_src):
    internal_run_dir = Path(cfg.save_root) / cfg.run_name
    output_run_dir = Path(cfg.paths.output_dir) / cfg.run_name
    if internal_run_dir.resolve() == output_run_dir.resolve():
        shutil.copy2(config_src, output_run_dir / "config_used.yaml")
        return output_run_dir
    if output_run_dir.exists() and not cfg.prepare.overwrite_output:
        raise FileExistsError(
            f"输出目录已存在：{output_run_dir}。"
            "如果需要覆盖，请把 prepare.overwrite_output 设为 true。"
        )
    copytree_overwrite(internal_run_dir, output_run_dir)
    shutil.copy2(config_src, output_run_dir / "config_used.yaml")
    return output_run_dir


def export_tracking_json(cfg):
    if has_info_subset(cfg):
        print("检测到子集模式，正在导出 subset tracking json...")
        return export_subset_tracking_json(cfg, build_subset_context(cfg))

    from scripts.nuScenes.export_kitti import KittiConverter

    print("正在把官方 AB3DMOT 跟踪结果导出为 nuScenes tracking json...")
    converter = KittiConverter(
        nusc_kitti_root=cfg.paths.nusc_kitti_root,
        data_root=cfg.paths.nuscenes_data_root,
        result_root=cfg.save_root,
        result_name=cfg.run_name,
        split=cfg.split,
    )
    converter.kitti_trk_result2nuscenes()

    exported_json = Path(cfg.save_root) / cfg.run_name / f"results_{cfg.split}.json"
    if not exported_json.is_file():
        raise FileNotFoundError(
            f"官方导出完成后没有找到 json 结果：{exported_json}"
        )
    return exported_json


def maybe_run_visualization(cfg):
    if not cfg.visualization.enabled:
        return
    print("正在生成可视化结果...")
    from visual import run_visualization

    run_visualization(cfg)


def load_source_meta(cfg):
    with open(cfg.paths.bevformer_json, "r") as file_obj:
        raw_data = json.load(file_obj)

    if isinstance(raw_data, dict):
        meta = raw_data.get("meta")
        if isinstance(meta, dict):
            return dict(meta)
    return None


def write_final_json(cfg, exported_json, config_src):
    keep_categories = {str(cat).lower() for cat in cfg.cat_list}
    with open(exported_json, "r") as file_obj:
        tracking_json = json.load(file_obj)

    filtered_results = {}
    for sample_token, objects in tracking_json.get("results", {}).items():
        filtered_results[sample_token] = [
            obj
            for obj in objects
            if str(obj.get("tracking_name", "")).lower() in keep_categories
        ]
    tracking_json["results"] = filtered_results

    source_meta = load_source_meta(cfg)
    if source_meta is not None:
        tracking_json["meta"] = source_meta

    output_json = Path(cfg.paths.output_json)
    ensure_dir(output_json.parent)
    if output_json.exists() and not cfg.prepare.overwrite_output:
        raise FileExistsError(
            f"输出 json 已存在：{output_json}。"
            "如果需要覆盖，请把 prepare.overwrite_output 设为 true。"
        )

    with open(output_json, "w") as file_obj:
        json.dump(tracking_json, file_obj, indent=2)

    config_copy = output_json.with_name(f"{output_json.stem}_config_used.yaml")
    shutil.copy2(config_src, config_copy)
    return output_json


def run_tracking(cfg):
    time_str = get_timestring()
    log_path = Path(cfg.save_root) / "log" / f"log_{time_str}_{cfg.dataset}_{cfg.split}.txt"
    ensure_dir(log_path.parent)
    with open(log_path, "w") as log:
        if isinstance(cfg, dict):
            cfg_to_log = {key: value for key, value in cfg.items() if key != "subset_context"}
        else:
            cfg_to_log = cfg
        print_log(
            yaml.safe_dump(
                to_plain_dict(cfg_to_log), sort_keys=False, allow_unicode=True
            ),
            log,
            display=False,
        )

        id_start = 1
        for cat in cfg.cat_list:
            id_start = main_per_cat(cfg, cat, log, id_start)

        print_log("\n正在合并各类别跟踪结果......", log=log)
        combine_tracking_results(cfg)
        print_log("\n跟踪完成！", log=log)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    cfg = load_pipeline_config(args.config)
    if args.detection_json is not None:
        cfg.paths.bevformer_json = str(Path(args.detection_json).resolve())
        if args.output_json is not None:
            cfg.paths.output_json = str(Path(args.output_json).resolve())
        else:
            det_path = Path(args.detection_json)
            cfg.paths.output_json = str(
                (det_path.parent / f"tracking_{det_path.stem}.json").resolve()
            )
        cfg.prepare.use_json_subset = True
        cfg.prepare.subset_source = "json"
        cfg.prepare.overwrite_output = True
        cfg.prepare.force_reconvert_detection = True
        cfg.prepare.force_rerun_tracking = True
        cfg.visualization.enabled = False
        print(f"使用直接 detection_json 模式：\n  输入: {cfg.paths.bevformer_json}\n  输出: {cfg.paths.output_json}")
    validate_config(cfg)
    ensure_dir(cfg.save_root)
    ensure_dir(Path(cfg.paths.output_json).parent)
    if cfg.prepare.sync_workspace_output:
        ensure_dir(cfg.paths.output_dir)
    ensure_dir(cfg.paths.visual_dir)

    print("正在检查并同步官方数据目录链接...")
    ensure_official_dataset_links(cfg)
    print("正在检查 nuKITTI tracking/object 所需数据...")
    ensure_tracking_data(cfg)
    ensure_object_correspondence(cfg)
    print("正在把 BEVFormer results_nusc.json 转成官方 AB3DMOT 输入...")
    prepare_detection_inputs(cfg)

    if cfg.prepare.force_rerun_tracking:
        print("已启用 force_rerun_tracking，正在清理旧的跟踪结果...")
        clear_previous_tracking_results(cfg)

    print("正在调用官方 AB3DMOT 开始跟踪...")
    run_tracking(cfg)
    exported_json = export_tracking_json(cfg)
    final_json = write_final_json(cfg, exported_json, args.config)
    maybe_run_visualization(cfg)
    if cfg.prepare.sync_workspace_output:
        output_run_dir = sync_final_output(cfg, args.config)
        print(f"官方中间工作目录已同步到：{output_run_dir}")
    print(f"AB3DMOT 跟踪 json 已保存到：{final_json}")


if __name__ == "__main__":
    main()
