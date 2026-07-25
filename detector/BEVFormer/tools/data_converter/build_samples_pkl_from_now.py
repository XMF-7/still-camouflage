#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import pickle
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据 now.pkl 生成一份面向 data/nuscenes/samples 的 pkl"
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/home/jushuo/Code/zz4-bev/nuCarla/BEVFormer/data/nuscenes/now.pkl"),
        help="源 pkl 路径",
    )
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=Path("/home/jushuo/Code/zz4-bev/nuCarla/BEVFormer/data/nuscenes/samples"),
        help="samples 根目录",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/jushuo/Code/zz4-bev/nuCarla/BEVFormer/data/nuscenes/samples.pkl"),
        help="输出 pkl 路径",
    )
    return parser.parse_args()


def load_payload(src_path: Path) -> Dict[str, Any]:
    with src_path.open("rb") as fp:
        payload = pickle.load(fp)
    if not isinstance(payload, dict) or "infos" not in payload:
        raise RuntimeError(f"源 pkl 格式不符合预期: {src_path}")
    if not isinstance(payload["infos"], list):
        raise RuntimeError(f"源 pkl 的 infos 不是列表: {src_path}")
    return payload


def _keyframe_paths_exist(info: Dict[str, Any], samples_root: Path) -> bool:
    lidar_path = Path(str(info.get("lidar_path", ""))).expanduser().resolve()
    if samples_root not in lidar_path.parents or not lidar_path.exists():
        return False

    cams = info.get("cams", {})
    if not isinstance(cams, dict) or not cams:
        return False
    for channel, cam_info in cams.items():
        if not isinstance(cam_info, dict):
            return False
        data_path = Path(str(cam_info.get("data_path", ""))).expanduser().resolve()
        if samples_root not in data_path.parents or not data_path.exists():
            return False
    return True


def _relink_sequence(infos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for info in infos:
        by_scene.setdefault(str(info["scene_token"]), []).append(info)

    normalized: List[Dict[str, Any]] = []
    for scene_infos in by_scene.values():
        scene_infos.sort(key=lambda item: int(item["timestamp"]))
        total = len(scene_infos)
        for index, info in enumerate(scene_infos):
            info["frame_idx"] = index
            info["prev"] = "" if index == 0 else str(scene_infos[index - 1]["token"])
            info["next"] = "" if index == total - 1 else str(scene_infos[index + 1]["token"])
            info["sweeps"] = []
            normalized.append(info)

    normalized.sort(key=lambda item: int(item["timestamp"]))
    return normalized


def build_payload(src_payload: Dict[str, Any], samples_root: Path, src_path: Path) -> Dict[str, Any]:
    infos: List[Dict[str, Any]] = []
    dropped = 0
    for info in src_payload["infos"]:
        if _keyframe_paths_exist(info=info, samples_root=samples_root):
            infos.append(copy.deepcopy(info))
        else:
            dropped += 1
    infos = _relink_sequence(infos)

    metadata = dict(src_payload.get("metadata", {}))
    metadata["rebuilt_by"] = "build_samples_pkl_from_now"
    metadata["source_pkl"] = str(src_path.resolve())
    metadata["samples_root"] = str(samples_root.resolve())
    metadata["num_frames"] = len(infos)
    metadata["dropped_missing_frames"] = dropped
    metadata["mode"] = "samples_only"
    return {"infos": infos, "metadata": metadata}


def main() -> None:
    args = parse_args()
    src_path = args.src.expanduser().resolve()
    samples_root = args.samples_root.expanduser().resolve()
    out_path = args.out.expanduser().resolve()

    if not src_path.exists():
        raise FileNotFoundError(f"找不到源 pkl: {src_path}")
    if not samples_root.exists():
        raise FileNotFoundError(f"找不到 samples 根目录: {samples_root}")

    src_payload = load_payload(src_path=src_path)
    payload = build_payload(src_payload=src_payload, samples_root=samples_root, src_path=src_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        pickle.dump(payload, fp)

    print(f"wrote {out_path}")
    print(f"num_infos={len(payload['infos'])}")
    print(f"samples_root={samples_root}")


if __name__ == "__main__":
    main()
