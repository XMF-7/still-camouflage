import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
XINSHUO_ROOT = REPO_ROOT / "Xinshuo_PyToolbox"
for _path in [REPO_ROOT, XINSHUO_ROOT]:
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import matplotlib

matplotlib.use("Agg")

from AB3DMOT_libs.kitti_calib import Calibration
from AB3DMOT_libs.kitti_obj import read_label
from AB3DMOT_libs.kitti_trk import Tracklet_3D
from AB3DMOT_libs.nuscenes_subset import build_subset_context, has_info_subset
from AB3DMOT_libs.pipeline_utils import (
    clear_path,
    ensure_dir,
    ensure_official_dataset_links,
    load_pipeline_config,
    visual_threshold_for_category,
)
from AB3DMOT_libs.utils import get_subfolder_seq
from AB3DMOT_libs.vis import vis_image_with_obj
from xinshuo_io import load_list_from_folder
from xinshuo_miscellaneous import print_log
from xinshuo_video import generate_video_from_folder


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用官方可视化代码渲染 AB3DMOT 的跟踪结果。"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径。",
    )
    return parser.parse_args()


def resolve_output_run_dir(cfg):
    output_run_dir = Path(cfg.paths.output_dir) / cfg.run_name
    if output_run_dir.exists():
        return output_run_dir

    internal_run_dir = Path(cfg.save_root) / cfg.run_name
    if internal_run_dir.exists():
        return internal_run_dir

    raise FileNotFoundError(
        f"找不到 {cfg.run_name} 的跟踪输出。"
        f"已检查：{output_run_dir} 和 {internal_run_dir}。"
    )


def resolve_sequence_list(cfg, default_seq_eval):
    if has_info_subset(cfg):
        context = build_subset_context(cfg)
        return list(context["scene_names"])
    return list(default_seq_eval)


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def collect_keep_ids(seq_eval_file, cfg):
    if not seq_eval_file.exists():
        return None

    results = Tracklet_3D(str(seq_eval_file))
    id_scores = {}
    id_category = {}
    for _, frame_data in results.data.items():
        for obj_id, obj in frame_data.items():
            id_scores.setdefault(obj_id, []).append(float(obj.s))
            id_category[obj_id] = obj.type

    keep_ids = set()
    for obj_id, scores in id_scores.items():
        threshold = visual_threshold_for_category(cfg, id_category[obj_id])
        average_score = sum(scores) / float(len(scores))
        if threshold is None or average_score >= threshold:
            keep_ids.add(obj_id)
    return keep_ids


def filter_objects(objects, keep_ids, cfg):
    filtered = []
    for obj in objects:
        if obj.type not in cfg.cat_list:
            continue
        if keep_ids is not None and obj.id not in keep_ids:
            continue
        filtered.append(obj)
    return filtered


def visualize_sequence(
    cfg, output_run_dir, visual_run_dir, seq_name, image_dir, calib_file, hw, log
):
    result_dir = output_run_dir / f"trk_withid_{cfg.visualization.hypo_index}" / seq_name
    eval_file = output_run_dir / f"data_{cfg.visualization.hypo_index}" / f"{seq_name}.txt"
    if not result_dir.exists() and not eval_file.exists():
        return

    save_image_dir = visual_run_dir / "trk_image_vis" / seq_name
    ensure_dir(save_image_dir)
    if cfg.visualization.overwrite:
        clear_path(save_image_dir)
        ensure_dir(save_image_dir)

    keep_ids = collect_keep_ids(eval_file, cfg)
    calib = Calibration(str(calib_file))
    images_list, num_images = load_list_from_folder(str(image_dir))
    print(f"正在可视化序列 {seq_name}，共 {num_images} 帧...")
    print_log(f"序列 {seq_name}，待可视化图片数量：{num_images}", log)

    for count, image_path in enumerate(images_list, start=1):
        image_path = Path(image_path)
        image_index = int(image_path.stem)
        result_file = result_dir / f"{image_index:06d}.txt"
        objects = read_label(str(result_file)) if result_file.exists() else []
        objects = filter_objects(objects, keep_ids, cfg)

        save_path = save_image_dir / f"{image_index:06d}.jpg"
        vis_image_with_obj(
            str(image_path),
            objects,
            [],
            calib,
            hw,
            save_path=str(save_path),
            color_type=cfg.visualization.color_type,
        )
        print(
            f"正在处理序列 {seq_name} 的第 {count}/{num_images} 帧，"
            f"当前绘制目标数：{len(objects)}",
            end="\r",
            flush=True,
        )
        print_log(
            f"正在处理帧 {image_index}，进度 {count}/{num_images}，"
            f"当前绘制目标数：{len(objects)}",
            log,
            display=False,
        )
    print("")

    if cfg.visualization.render_video:
        if not any(save_image_dir.glob("*.jpg")):
            return
        if not ffmpeg_available():
            print("检测到当前环境缺少 ffmpeg/ffprobe，已跳过视频导出，仅保留图片结果。")
            print_log(
                "检测到当前环境缺少 ffmpeg/ffprobe，已跳过视频导出，仅保留图片结果。",
                log,
            )
            return
        video_dir = visual_run_dir / "trk_video_vis"
        ensure_dir(video_dir)
        video_file = video_dir / f"{seq_name}.mp4"
        print(f"正在生成视频：{video_file}")
        generate_video_from_folder(
            str(save_image_dir),
            str(video_file),
            framerate=cfg.visualization.framerate,
        )


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    cfg = load_pipeline_config(args.config)
    run_visualization(cfg)


def run_visualization(cfg):
    if not cfg.visualization.enabled:
        print("visual.enabled 为 false，跳过可视化。")
        return
    ensure_dir(cfg.paths.visual_dir)
    ensure_official_dataset_links(cfg)

    output_run_dir = resolve_output_run_dir(cfg)
    visual_run_dir = Path(cfg.paths.visual_dir) / cfg.run_name
    if cfg.visualization.overwrite:
        clear_path(visual_run_dir)
    ensure_dir(visual_run_dir)
    log_path = visual_run_dir / "visual.log"

    subfolder, _, hw, seq_eval_default, data_root = get_subfolder_seq(
        cfg.dataset, cfg.split
    )
    seq_eval = resolve_sequence_list(cfg, seq_eval_default)
    trk_root = Path(data_root) / "tracking" / subfolder

    with open(log_path, "w") as log:
        print(f"正在从 {output_run_dir} 读取官方跟踪结果...")
        print_log(f"正在从 {output_run_dir} 读取官方跟踪结果...", log)
        for seq_name in seq_eval:
            image_dir = trk_root / "image_02" / seq_name
            calib_file = trk_root / "calib" / f"{seq_name}.txt"
            if not image_dir.exists() or not calib_file.exists():
                continue
            visualize_sequence(
                cfg,
                output_run_dir,
                visual_run_dir,
                seq_name,
                image_dir,
                calib_file,
                hw,
                log,
            )

    print(f"可视化结果已保存到：{visual_run_dir}")


if __name__ == "__main__":
    main()
