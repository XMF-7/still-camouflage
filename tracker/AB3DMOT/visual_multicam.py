import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
XINSHUO_ROOT = REPO_ROOT / "Xinshuo_PyToolbox"
for _path in [REPO_ROOT, XINSHUO_ROOT]:
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")

from PIL import Image
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.utils.kitti import KittiDB

from AB3DMOT_libs.kitti_calib import Calibration, save_calib_file
from AB3DMOT_libs.kitti_obj import Object_3D
from AB3DMOT_libs.nuscenes_subset import (
    _create_kitti_transform,
    _get_frame_sensor_params,
    _get_result_token,
    _nuScenes_transform2KITTI,
    build_subset_context,
)
from AB3DMOT_libs.nuScenes_utils import nuScenes_world2lidar
from AB3DMOT_libs.pipeline_utils import ensure_dir, load_pipeline_config
from AB3DMOT_libs.vis import vis_image_with_obj


CAMERA_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="将 AB3DMOT 的 tracking json 投影到六个 nuScenes 相机视角。"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="AB3DMOT 配置文件路径。",
    )
    parser.add_argument(
        "--tracking-json",
        type=str,
        default="",
        help="可选，显式指定 tracking json；默认读取 config 里的 output_json。",
    )
    parser.add_argument(
        "--info-pkl",
        type=str,
        default="",
        help="可选，显式指定与 tracking json 对齐的 info pkl；默认自动检测或读取 config。",
    )
    return parser.parse_args()


def _load_tracking_results(tracking_json_path):
    with open(tracking_json_path, "r") as file_obj:
        payload = json.load(file_obj)
    results = payload.get("results", {})
    if not isinstance(results, dict) or not results:
        raise ValueError(f"{tracking_json_path} 中没有 results。")
    return payload, results


def _load_info_payload(info_pkl_path):
    import pickle

    with open(info_pkl_path, "rb") as file_obj:
        payload = pickle.load(file_obj)

    infos = payload.get("infos") if isinstance(payload, dict) else payload
    if not infos:
        return None
    return infos


def _detect_matching_info_pkl(cfg, result_tokens):
    configured = str(getattr(cfg.paths, "info_pkl", "") or "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured))

    data_root = Path(cfg.paths.nuscenes_data_root)
    candidates.extend(sorted(data_root.glob("*.pkl")))

    best_path = None
    best_score = (-1, -1)
    result_token_set = set(result_tokens)

    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            infos = _load_info_payload(candidate)
        except Exception:
            continue
        if not infos:
            continue

        overlap = 0
        cam_rich = 0
        for info in infos:
            token = info.get("token")
            if token in result_token_set:
                overlap += 1
                if info.get("cams"):
                    cam_rich += 1
        score = (overlap, cam_rich)
        if score > best_score:
            best_score = score
            best_path = candidate

    if best_path is None or best_score[0] <= 0:
        return None
    return str(best_path.resolve())


def _ensure_subset_context(cfg, result_tokens, override_info_pkl=""):
    if override_info_pkl:
        cfg.paths.info_pkl = str(Path(override_info_pkl).resolve())
    elif not str(getattr(cfg.paths, "info_pkl", "") or "").strip():
        detected = _detect_matching_info_pkl(cfg, result_tokens)
        if detected:
            cfg.paths.info_pkl = detected

    context = build_subset_context(cfg)
    if context is None:
        raise RuntimeError("无法构建 subset context，至少需要 bevformer json 或 info pkl。")
    return context


def _make_track_object(track_item, pose_record, cs_record_lid, p_left_kitti, velo_to_cam_rot, velo_to_cam_trans, r0_rect, image_size):
    box_world = Box(
        track_item["translation"],
        track_item["size"],
        Quaternion(track_item["rotation"]),
        name=str(track_item.get("tracking_name", "car")).capitalize(),
    )
    box_world.score = float(track_item.get("tracking_score", 0.0))
    box_lidar = nuScenes_world2lidar(
        box_world,
        cs_record=cs_record_lid,
        pose_record=pose_record,
    )
    box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
        box_lidar,
        Quaternion(matrix=velo_to_cam_rot),
        velo_to_cam_trans,
        r0_rect,
    )
    bbox_2d = KittiDB.project_kitti_box_to_image(
        box_cam_kitti,
        p_left_kitti,
        imsize=image_size,
    )
    if bbox_2d is None:
        bbox_2d = (-1, -1, -1, -1)
    box_cam_kitti.score = float(track_item.get("tracking_score", 0.0))
    line = KittiDB.box_to_string(
        name=str(track_item.get("tracking_name", "car")).capitalize(),
        box=box_cam_kitti,
        bbox_2d=bbox_2d,
        truncation=0.0,
        occlusion=0,
    )
    obj = Object_3D(line)
    obj.id = int(track_item.get("tracking_id", -1))
    return obj


def _draw_camera_label(image_path, camera_name):
    image = cv2.imread(str(image_path))
    if image is None:
        return
    label = camera_name.replace("CAM_", "")
    font = cv2.FONT_HERSHEY_TRIPLEX
    scale = 0.85
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(label, font, scale, thickness)
    x0, y0 = 20, 18
    cv2.rectangle(image, (x0 - 8, y0 - 4), (x0 + text_w + 8, y0 + text_h + 8), (0, 0, 0), -1)
    cv2.putText(image, label, (x0, y0 + text_h), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.imwrite(str(image_path), image)


def _compose_multicam_grid(image_paths, save_path):
    opened = []
    for camera_name in CAMERA_ORDER:
        image = Image.open(str(image_paths[camera_name])).convert("RGB")
        opened.append((camera_name, image))

    tile_w = max(image.size[0] for _, image in opened)
    tile_h = max(image.size[1] for _, image in opened)
    canvas = Image.new("RGB", (tile_w * 3, tile_h * 2), color=(18, 18, 18))

    for index, (_, image) in enumerate(opened):
        row = index // 3
        col = index % 3
        if image.size != (tile_w, tile_h):
            image = image.resize((tile_w, tile_h))
        canvas.paste(image, (col * tile_w, row * tile_h))

    canvas.save(str(save_path))


def _write_temp_calibration(velo_to_cam_trans, velo_to_cam_rot, r0_rect, p_left_kitti):
    transform = _create_kitti_transform(
        velo_to_cam_trans,
        velo_to_cam_rot,
        r0_rect,
        p_left_kitti,
    )
    temp_file = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    temp_file.close()
    save_calib_file(transform, temp_file.name)
    return temp_file.name


def render_multicam_tracking(cfg, tracking_json_path, override_info_pkl=""):
    _, results = _load_tracking_results(tracking_json_path)
    context = _ensure_subset_context(cfg, results.keys(), override_info_pkl=override_info_pkl)

    visual_root = Path(cfg.paths.visual_dir) / cfg.run_name / "trk_multicam_vis"
    if visual_root.exists():
        shutil.rmtree(str(visual_root))
    ensure_dir(visual_root)

    nusc = context["nusc"]
    per_token = OrderedDict((token, items) for token, items in results.items())
    scene_frame_map = context["scene_frame_map"]

    for result_token, track_items in per_token.items():
        info = context["info_by_result_token"].get(result_token)
        if info is None:
            continue
        scene_name, frame_index = scene_frame_map[result_token]
        scene_root = visual_root / scene_name
        ensure_dir(scene_root)

        single_paths = {}
        for camera_name in CAMERA_ORDER:
            (
                pose_record,
                cs_record_lid,
                cs_record_cam,
                _filename_lid_full,
                filename_cam_full,
            ) = _get_frame_sensor_params(
                nusc,
                info,
                cam_name=camera_name,
                output_file=True,
            )

            cam_info = (info.get("cams") or {}).get(camera_name, {})
            image_rel = cam_info.get("data_path", filename_cam_full)
            image_path = Path(cfg.paths.nuscenes_data_root) / image_rel
            if not image_path.exists():
                image_path = Path(filename_cam_full)
            if not image_path.exists():
                raise FileNotFoundError(f"找不到相机图像：{camera_name} -> {image_rel}")

            with Image.open(str(image_path)) as pil_image:
                width, height = pil_image.size
            image_size = (width, height)

            velo_to_cam_trans, velo_to_cam_rot, r0_rect, p_left_kitti = _nuScenes_transform2KITTI(
                cs_record_lid,
                cs_record_cam,
            )
            calib_path = _write_temp_calibration(
                velo_to_cam_trans,
                velo_to_cam_rot,
                r0_rect,
                p_left_kitti,
            )
            try:
                calib = Calibration(calib_path)
            finally:
                os.unlink(calib_path)

            objects = []
            for track_item in track_items:
                if str(track_item.get("tracking_name", "")).lower() not in {"car", "truck", "bus", "trailer", "bicycle", "motorcycle", "pedestrian"}:
                    continue
                obj = _make_track_object(
                    track_item,
                    pose_record,
                    cs_record_lid,
                    p_left_kitti,
                    velo_to_cam_rot,
                    velo_to_cam_trans,
                    r0_rect,
                    image_size,
                )
                objects.append(obj)

            save_dir = scene_root / camera_name
            ensure_dir(save_dir)
            save_path = save_dir / f"{frame_index:06d}.jpg"
            vis_image_with_obj(
                str(image_path),
                objects,
                [],
                calib,
                {"image": (height, width)},
                save_path=str(save_path),
                color_type=cfg.visualization.color_type,
                text_scale=0.75,
                text_thickness=2,
            )
            _draw_camera_label(save_path, camera_name)
            single_paths[camera_name] = save_path

        multicam_dir = scene_root / "multicam"
        ensure_dir(multicam_dir)
        _compose_multicam_grid(single_paths, multicam_dir / f"{frame_index:06d}.jpg")

    return visual_root


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    cfg = load_pipeline_config(args.config)
    tracking_json_path = args.tracking_json or cfg.paths.output_json
    if not tracking_json_path:
        raise ValueError("未提供 tracking json 路径。")
    visual_root = render_multicam_tracking(
        cfg,
        str(Path(tracking_json_path).resolve()),
        override_info_pkl=args.info_pkl,
    )
    print(f"六视角 tracking 可视化已保存到：{visual_root}")


if __name__ == "__main__":
    main()
