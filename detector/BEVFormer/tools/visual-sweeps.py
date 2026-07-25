import copy
import gc
import multiprocessing as mp
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import box_in_image, view_points
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm


CAMS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]

nusc = None
pseudo_info_by_token = {}
pred_results = None


def _pred_records_for_pseudo_token(
    pseudo_token: str,
    pred_data=None,
    score_thr: float = 0.2,
    detection_name: str | None = None,
):
    if pred_data is None:
        return []
    records = []
    for record in pred_data["results"].get(pseudo_token, []):
        if record["detection_score"] <= score_thr:
            continue
        if detection_name is not None and record["detection_name"] != detection_name:
            continue
        records.append(record)
    return records


def _box_from_record(record: dict):
    box = Box(
        record["translation"],
        record["size"],
        Quaternion(record["rotation"]),
        name=record["detection_name"],
        token="predicted",
    )
    box.score = float(record["detection_score"])
    box.source_record = record
    return box


def _pred_boxes_for_pseudo_token(
    pseudo_token: str,
    pred_data=None,
    score_thr: float = 0.2,
    detection_name: str | None = None,
):
    if pred_data is None:
        return []
    return [
        _box_from_record(record)
        for record in _pred_records_for_pseudo_token(
            pseudo_token,
            pred_data=pred_data,
            score_thr=score_thr,
            detection_name=detection_name,
        )
    ]


def _ego_bev_view():
    return np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _ego_bev_limits(boxes, margin: float = 10.0, min_radius: float = 20.0):
    points_y = [0.0]
    points_x = [0.0]
    for box in boxes:
        corners = box.corners()[:2, :]
        points_x.extend(corners[0, :].tolist())
        points_y.extend(corners[1, :].tolist())

    min_y = min(points_y)
    max_y = max(points_y)
    min_x = min(points_x)
    max_x = max(points_x)
    center_y = 0.5 * (min_y + max_y)
    center_x = 0.5 * (min_x + max_x)
    radius = 0.5 * max(max_y - min_y, max_x - min_x) + margin
    radius = max(radius, min_radius)
    return (
        center_y - radius,
        center_y + radius,
        center_x - radius,
        center_x + radius,
    )


def _color_for_category(category_name: str):
    if category_name == "bicycle":
        return nusc.colormap["vehicle.bicycle"]
    if category_name == "construction_vehicle":
        return nusc.colormap["vehicle.construction"]
    if category_name == "traffic_cone":
        return nusc.colormap["movable_object.trafficcone"]
    for key in nusc.colormap.keys():
        if category_name in key:
            return nusc.colormap[key]
    return [0, 0, 0]


def _predicted_boxes_in_sensor_frame(sample_data_token: str, pred_boxes_global):
    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = nusc.get("sensor", cs_record["sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])
    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record["modality"] != "camera":
        raise ValueError(f"unsupported modality for rendering: {sensor_record['modality']}")

    cam_intrinsic = np.array(cs_record["camera_intrinsic"])
    imsize = (sd_record["width"], sd_record["height"])

    box_list = []
    for box in pred_boxes_global:
        pred_box = copy.deepcopy(box)
        pred_box.translate(-np.array(pose_record["translation"]))
        pred_box.rotate(Quaternion(pose_record["rotation"]).inverse)
        pred_box.translate(-np.array(cs_record["translation"]))
        pred_box.rotate(Quaternion(cs_record["rotation"]).inverse)
        if not box_in_image(pred_box, cam_intrinsic, imsize):
            continue
        box_list.append(pred_box)

    return data_path, box_list, cam_intrinsic


def _project_box_corners_2d(box: Box, camera_intrinsic: np.ndarray):
    corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
    return corners


def _projected_bbox_stats(box: Box, camera_intrinsic: np.ndarray, image_size: tuple[int, int]):
    corners = _project_box_corners_2d(box, camera_intrinsic)
    width, height = image_size
    x_min = float(np.clip(np.min(corners[0, :]), 0, width - 1))
    x_max = float(np.clip(np.max(corners[0, :]), 0, width - 1))
    y_min = float(np.clip(np.min(corners[1, :]), 0, height - 1))
    y_max = float(np.clip(np.max(corners[1, :]), 0, height - 1))
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    anchor_x = x_min
    anchor_y = y_max
    return {
        "center_x": center_x,
        "center_y": center_y,
        "anchor_x": anchor_x,
        "anchor_y": anchor_y,
    }


def _draw_score_label(ax, box: Box, camera_intrinsic: np.ndarray, image_size: tuple[int, int]):
    stats = _projected_bbox_stats(box, camera_intrinsic, image_size)
    score = getattr(box, "score", None)
    if score is None:
        return
    ax.text(
        stats["anchor_x"],
        stats["anchor_y"],
        f"{score:.4f}",
        fontsize=8,
        color="white",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="black", edgecolor="white", linewidth=0.4, alpha=0.70),
        zorder=10,
    )


def _draw_bev_score_label(ax, box: Box):
    score = getattr(box, "score", None)
    if score is None:
        return
    center = np.asarray(box.center, dtype=np.float32)
    ax.text(
        float(center[1]),
        float(center[0]),
        f"{score:.4f}",
        fontsize=8,
        color="white",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="black", edgecolor="white", linewidth=0.4, alpha=0.70),
        zorder=10,
    )


def _boxes_global_to_ego(pseudo_token: str, boxes_global):
    info = pseudo_info_by_token[pseudo_token]
    ego_rotation_inv = Quaternion(info["ego2global_rotation"]).inverse
    ego_translation = np.array(info["ego2global_translation"])

    boxes_ego = []
    for box in boxes_global:
        ego_box = copy.deepcopy(box)
        ego_box.translate(-ego_translation)
        ego_box.rotate(ego_rotation_inv)
        boxes_ego.append(ego_box)
    return boxes_ego


def render_pseudo_frame_data(
    pseudo_token: str, pred_data=None, out_path: str | None = None, verbose: bool = True, score_thr: float = 0.2
):
    info = pseudo_info_by_token[pseudo_token]
    fig, ax = plt.subplots(2, 3, figsize=(24, 9))
    plt.subplots_adjust(wspace=0.07, hspace=0.12)

    for ind, cam in enumerate(CAMS):
        row = 0 if ind < 3 else 1
        col = ind % 3
        sample_data_token = info["cams"][cam]["sample_data_token"]
        boxes = _pred_boxes_for_pseudo_token(pseudo_token, pred_data=pred_data, score_thr=score_thr)
        data_path, boxes_pred, camera_intrinsic = _predicted_boxes_in_sensor_frame(
            sample_data_token, pred_boxes_global=boxes
        )
        data = Image.open(data_path)

        ax[row, col].imshow(data)
        for box in boxes_pred:
            color = np.array(_color_for_category(box.name)) / 255.0
            box.render(ax[row, col], view=camera_intrinsic, normalize=True, colors=(color, color, color))
            _draw_score_label(ax[row, col], box, camera_intrinsic, data.size)

        ax[row, col].set_xlim(0, data.size[0])
        ax[row, col].set_ylim(data.size[1], 0)
        ax[row, col].axis("off")
        delta_ms = info["matched_deltas_ms"][cam]
        ax[row, col].set_title(f"PRED: {cam} (dt={delta_ms:.1f}ms)")

    if out_path is not None:
        plt.savefig(out_path + "_camera", bbox_inches="tight", pad_inches=0.2, dpi=200, facecolor="white")
    if verbose:
        plt.show()

    plt.close()
    gc.collect()


def render_pseudo_frame_bev(
    pseudo_token: str, pred_data=None, out_path: str | None = None, verbose: bool = True, score_thr: float = 0.2
):
    pred_boxes_global = _pred_boxes_for_pseudo_token(
        pseudo_token,
        pred_data=pred_data,
        score_thr=score_thr,
    )
    boxes_pred_ego = _boxes_global_to_ego(pseudo_token, pred_boxes_global)

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.92)
    bev_view = _ego_bev_view()
    pred_color = (0.90, 0.20, 0.20)

    for box in boxes_pred_ego:
        box.render(ax, view=bev_view, colors=(pred_color, pred_color, pred_color))
        _draw_bev_score_label(ax, box)

    y_min, y_max, x_min, x_max = _ego_bev_limits(list(boxes_pred_ego))
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.9)
    ax.set_xlim([y_min, y_max])
    ax.set_ylim([x_min, x_max])
    ax.invert_xaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("y (left)")
    ax.set_ylabel("x (forward)")
    ax.set_title("Ego BEV (front-up): pred(red)")
    ax.grid(True, alpha=0.2)

    if out_path is not None:
        plt.savefig(out_path + "_bev", bbox_inches="tight", pad_inches=0.2, dpi=200, facecolor="white")
    if verbose:
        plt.show()

    plt.close()
    gc.collect()


def _process_token(item):
    pseudo_token, frame_name, out_dir, score_thr = item
    out_path = os.path.join(out_dir, frame_name)
    try:
        render_pseudo_frame_data(
            pseudo_token,
            pred_data=pred_results,
            out_path=out_path,
            verbose=False,
            score_thr=score_thr,
        )
        render_pseudo_frame_bev(
            pseudo_token,
            pred_data=pred_results,
            out_path=out_path,
            verbose=False,
            score_thr=score_thr,
        )
        return None
    except Exception as exc:  # pragma: no cover - best effort logging
        return f"Error rendering {frame_name}: {exc}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render aligned camera-sweeps pseudo-frames with predicted results.")
    parser.add_argument("--results_path", type=str, required=True, help="Path to the results_nusc.json file.")
    parser.add_argument("--ann-file", type=str, required=True, help="Path to the aligned sweeps pkl file.")
    parser.add_argument("--version", type=str, default="v1.0-trainval", help="NuScenes dataset version.")
    parser.add_argument("--dataroot", type=str, default="./data/nuscenes", help="NuScenes dataset root.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    parser.add_argument("--score-thr", type=float, default=0.2, help="Score threshold for rendering predicted boxes.")
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    pred_results = mmcv.load(args.results_path)
    with Path(args.ann_file).open("rb") as fp:
        ann_payload = pickle.load(fp)
    infos = ann_payload["infos"]
    pseudo_info_by_token = {info["token"]: info for info in infos}

    ordered_infos = sorted(infos, key=lambda item: int(item["frame_idx"]))
    ordered_tokens = [info["token"] for info in ordered_infos if info["token"] in pred_results["results"]]
    print(f"Total sorted pseudo frames: {len(ordered_tokens)}")

    base_dir = os.path.dirname(args.results_path)
    out_dir = os.path.join(base_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    work_items = []
    for info in ordered_infos:
        token = info["token"]
        if token not in pred_results["results"]:
            continue
        frame_name = f"frame_{int(info['frame_idx']) + 1:04d}_{Path(info['reference_filename']).stem}"
        work_items.append((token, frame_name, out_dir, args.score_thr))

    if args.workers <= 1:
        for item in tqdm(work_items, total=len(work_items), desc="Rendering pseudo frames"):
            error = _process_token(item)
            if error:
                print(error)
    else:
        with mp.Pool(processes=args.workers) as pool:
            for error in tqdm(pool.imap_unordered(_process_token, work_items), total=len(work_items), desc="Rendering pseudo frames"):
                if error:
                    print(error)
