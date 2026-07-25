#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


CAM_CHANNELS: Sequence[str] = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)
ALL_CHANNELS: Sequence[str] = (*CAM_CHANNELS, "LIDAR_TOP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按照参考相机时间轴和最近邻时间匹配生成 conservative sweeps pkl"
    )
    parser.add_argument(
        "--dataroot",
        type=Path,
        default=Path("/home/jushuo/Code/zz4-bev/nuCarla/BEVFormer/data/nuscenes"),
        help="nuScenes 数据根目录",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-trainval",
        help="nuScenes version",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default="n008-2018-08-31-11-56-46-0400",
        help="目标 scene 名称",
    )
    parser.add_argument(
        "--ref-channel",
        type=str,
        default="CAM_FRONT",
        choices=list(CAM_CHANNELS),
        help="对齐参考相机",
    )
    parser.add_argument(
        "--start-filename",
        type=str,
        default="n008-2018-08-31-11-56-46-0400__CAM_FRONT__1535731408412404",
        help="参考相机起始图像名，可带或不带扩展名",
    )
    parser.add_argument(
        "--end-filename",
        type=str,
        default="n008-2018-08-31-11-56-46-0400__CAM_FRONT__1535731410362404",
        help="参考相机结束图像名，可带或不带扩展名",
    )
    parser.add_argument(
        "--max-delta-ms",
        type=float,
        default=40.0,
        help="跨视角最近邻时间差阈值，单位 ms",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/jushuo/Code/zz4-bev/nuCarla/BEVFormer/data/nuscenes/sweeps_cam_front_aligned_40ms.pkl"),
        help="输出 pkl 路径",
    )
    return parser.parse_args()


def _normalize_filename(name: str, channel: str) -> str:
    suffix = ".pcd.bin" if channel == "LIDAR_TOP" else ".jpg"
    return name if name.endswith(suffix) else f"{name}{suffix}"


def _rotation_matrix(quat: Sequence[float]) -> np.ndarray:
    return Quaternion(quat).rotation_matrix


def _zeros_can_bus() -> np.ndarray:
    return np.zeros(18, dtype=np.float32)


def _relative_sensor_to_lidar(
    nusc: NuScenes,
    sensor_token: str,
    lidar_l2e_t: np.ndarray,
    lidar_l2e_r: np.ndarray,
    lidar_e2g_t: np.ndarray,
    lidar_e2g_r: np.ndarray,
) -> Dict[str, Any]:
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])

    sensor2ego_rotation = np.array(cs_record["rotation"])
    sensor2ego_translation = np.array(cs_record["translation"], dtype=np.float64)
    ego2global_rotation = np.array(pose_record["rotation"])
    ego2global_translation = np.array(pose_record["translation"], dtype=np.float64)

    sensor_l2e_r = _rotation_matrix(sensor2ego_rotation)
    sensor_e2g_r = _rotation_matrix(ego2global_rotation)

    transform_tail = np.linalg.inv(lidar_e2g_r).T @ np.linalg.inv(lidar_l2e_r).T
    rotation = (sensor_l2e_r.T @ sensor_e2g_r.T) @ transform_tail
    translation = (sensor2ego_translation @ sensor_e2g_r.T + ego2global_translation) @ transform_tail
    translation -= (
        lidar_e2g_t @ transform_tail + lidar_l2e_t @ np.linalg.inv(lidar_l2e_r).T
    )

    info = {
        "data_path": str(Path(nusc.get_sample_data_path(sensor_token)).resolve()),
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": sensor2ego_translation.tolist(),
        "sensor2ego_rotation": sensor2ego_rotation.tolist(),
        "ego2global_translation": ego2global_translation.tolist(),
        "ego2global_rotation": ego2global_rotation.tolist(),
        "sensor2lidar_rotation": rotation.T.astype(np.float32),
        "sensor2lidar_translation": translation.astype(np.float32),
        "timestamp": int(sd_rec["timestamp"]),
    }
    if "camera_intrinsic" in cs_record and cs_record["camera_intrinsic"]:
        info["cam_intrinsic"] = np.array(cs_record["camera_intrinsic"], dtype=np.float32)
    return info


def _build_scene_index(nusc: NuScenes, scene_name: str) -> Dict[str, List[Tuple[int, str, str]]]:
    scene_index: Dict[str, List[Tuple[int, str, str]]] = {channel: [] for channel in ALL_CHANNELS}
    for sd in nusc.sample_data:
        filename = Path(sd["filename"]).name
        if not filename.startswith(f"{scene_name}__"):
            continue
        data_path = Path(nusc.get_sample_data_path(sd["token"])).resolve()
        if not data_path.exists():
            continue
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        sensor = nusc.get("sensor", cs["sensor_token"])
        channel = sensor["channel"]
        if channel in scene_index:
            scene_index[channel].append((int(sd["timestamp"]), sd["token"], filename))
    for entries in scene_index.values():
        entries.sort(key=lambda item: item[0])
    return scene_index


def _find_by_filename(entries: Sequence[Tuple[int, str, str]], filename: str) -> Tuple[int, str, str]:
    for item in entries:
        if item[2] == filename:
            return item
    raise FileNotFoundError(f"在参考通道序列中找不到文件: {filename}")


def _nearest_entry(
    entries: Sequence[Tuple[int, str, str]], target_ts: int
) -> Tuple[int, str, str, float]:
    timestamps = [item[0] for item in entries]
    idx = bisect_left(timestamps, target_ts)
    candidates: List[Tuple[int, str, str]] = []
    if idx < len(entries):
        candidates.append(entries[idx])
    if idx > 0:
        candidates.append(entries[idx - 1])
    if not candidates:
        raise RuntimeError("最近邻匹配失败：候选序列为空")
    best = min(candidates, key=lambda item: abs(item[0] - target_ts))
    delta_ms = abs(best[0] - target_ts) / 1000.0
    return best[0], best[1], best[2], delta_ms


def _build_infos(
    nusc: NuScenes,
    scene_index: Dict[str, List[Tuple[int, str, str]]],
    scene_name: str,
    ref_channel: str,
    start_filename: str,
    end_filename: str,
    max_delta_ms: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ref_entries = scene_index[ref_channel]
    start_entry = _find_by_filename(ref_entries, start_filename)
    end_entry = _find_by_filename(ref_entries, end_filename)
    start_ts, end_ts = start_entry[0], end_entry[0]
    if end_ts < start_ts:
        raise ValueError("结束帧时间早于起始帧")

    ref_window = [item for item in ref_entries if start_ts <= item[0] <= end_ts]
    kept_infos: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []

    for frame_idx, (ref_ts, ref_token, ref_name) in enumerate(ref_window):
        matched: Dict[str, Dict[str, Any]] = {}
        deltas_ms: Dict[str, float] = {}
        rejected = False
        reject_reason = ""

        for channel in ALL_CHANNELS:
            entries = scene_index[channel]
            matched_ts, matched_token, matched_name, delta_ms = _nearest_entry(entries, ref_ts)
            deltas_ms[channel] = delta_ms
            if delta_ms > max_delta_ms:
                rejected = True
                reject_reason = (
                    f"{channel} nearest delta {delta_ms:.3f}ms > threshold {max_delta_ms:.3f}ms"
                )
                break
            matched[channel] = {
                "timestamp": matched_ts,
                "token": matched_token,
                "filename": matched_name,
            }

        if rejected:
            dropped.append(
                {
                    "ref_channel": ref_channel,
                    "ref_filename": ref_name,
                    "ref_timestamp": ref_ts,
                    "reason": reject_reason,
                    "deltas_ms": deltas_ms,
                }
            )
            continue

        lidar_token = matched["LIDAR_TOP"]["token"]
        lidar_sd = nusc.get("sample_data", lidar_token)
        lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])

        lidar_l2e_t = np.array(lidar_cs["translation"], dtype=np.float64)
        lidar_l2e_r = _rotation_matrix(lidar_cs["rotation"])
        lidar_e2g_t = np.array(lidar_pose["translation"], dtype=np.float64)
        lidar_e2g_r = _rotation_matrix(lidar_pose["rotation"])

        sample_rec = nusc.get("sample", lidar_sd["sample_token"])
        info = {
            "token": ref_token,
            "timestamp": int(ref_ts),
            "lidar_path": str(Path(nusc.get_sample_data_path(lidar_token)).resolve()),
            "sweeps": [],
            "cams": {},
            "lidar2ego_translation": lidar_l2e_t.astype(np.float32),
            "lidar2ego_rotation": np.array(lidar_cs["rotation"], dtype=np.float32),
            "ego2global_translation": lidar_e2g_t.astype(np.float32),
            "ego2global_rotation": np.array(lidar_pose["rotation"], dtype=np.float32),
            "scene_token": sample_rec["scene_token"],
            "can_bus": _zeros_can_bus(),
            "frame_idx": frame_idx,
            "prev": "",
            "next": "",
            "reference_channel": ref_channel,
            "reference_sample_data_token": ref_token,
            "reference_filename": ref_name,
            "reference_sample_token": sample_rec["token"],
            "matched_deltas_ms": deltas_ms,
        }

        for cam_channel in CAM_CHANNELS:
            cam_token = matched[cam_channel]["token"]
            info["cams"][cam_channel] = _relative_sensor_to_lidar(
                nusc=nusc,
                sensor_token=cam_token,
                lidar_l2e_t=lidar_l2e_t,
                lidar_l2e_r=lidar_l2e_r,
                lidar_e2g_t=lidar_e2g_t,
                lidar_e2g_r=lidar_e2g_r,
            )

        kept_infos.append(info)

    kept_infos.sort(key=lambda item: int(item["timestamp"]))
    for idx, info in enumerate(kept_infos):
        info["frame_idx"] = idx
        info["prev"] = "" if idx == 0 else str(kept_infos[idx - 1]["token"])
        info["next"] = "" if idx == len(kept_infos) - 1 else str(kept_infos[idx + 1]["token"])

    metadata = {
        "version": nusc.version,
        "rebuilt_by": "build_aligned_sweeps_pkl",
        "mode": "camera_sweeps_aligned",
        "scene_name": scene_name,
        "reference_channel": ref_channel,
        "max_delta_ms": float(max_delta_ms),
        "start_filename": start_filename,
        "end_filename": end_filename,
        "num_ref_frames_in_window": len(ref_window),
        "num_infos": len(kept_infos),
        "dropped_count": len(dropped),
        "dropped_frames": dropped,
    }
    return kept_infos, metadata


def main() -> None:
    args = parse_args()
    dataroot = args.dataroot.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    start_filename = _normalize_filename(args.start_filename, args.ref_channel)
    end_filename = _normalize_filename(args.end_filename, args.ref_channel)

    if not dataroot.exists():
        raise FileNotFoundError(f"找不到 dataroot: {dataroot}")

    nusc = NuScenes(version=args.version, dataroot=str(dataroot), verbose=False)
    scene_index = _build_scene_index(nusc=nusc, scene_name=args.scene_name)
    infos, metadata = _build_infos(
        nusc=nusc,
        scene_index=scene_index,
        scene_name=args.scene_name,
        ref_channel=args.ref_channel,
        start_filename=start_filename,
        end_filename=end_filename,
        max_delta_ms=args.max_delta_ms,
    )
    if not infos:
        raise RuntimeError("没有生成任何可用伪帧，请检查输入范围和时间阈值")

    payload = {"infos": infos, "metadata": metadata}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        pickle.dump(payload, fp)

    print(f"wrote {out_path}")
    print(f"num_infos={len(infos)}")
    print(f"dropped={metadata['dropped_count']}")
    if metadata["dropped_frames"]:
        for item in metadata["dropped_frames"]:
            print(
                f"dropped_ref={item['ref_filename']} reason={item['reason']}"
            )


if __name__ == "__main__":
    main()
