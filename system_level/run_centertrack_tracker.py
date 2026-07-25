#!/usr/bin/env python3
"""Run CenterTrack's official Tracker association on nuScenes 3D detections.

This adapter deliberately uses CenterTrack's tracker implementation, but not the
CenterTrack image network. BEVDet already produced 3D detections, so we convert
each detection to the minimal fields expected by `utils.tracker.Tracker.step`.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np


CATEGORY_TO_ID = {
    "car": 1,
    "truck": 2,
    "bus": 3,
    "trailer": 4,
    "construction_vehicle": 5,
    "pedestrian": 6,
    "motorcycle": 7,
    "bicycle": 8,
    "traffic_cone": 9,
    "barrier": 10,
}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return path.resolve()


def _load_sample_tokens(sample_info_pkl: Path) -> List[str]:
    with sample_info_pkl.open("rb") as fp:
        payload = pickle.load(fp)
    infos = payload.get("infos") if isinstance(payload, dict) else payload
    if not isinstance(infos, list):
        raise RuntimeError(f"sample info pkl 格式不支持: {sample_info_pkl}")
    infos = sorted(infos, key=lambda item: int(item.get("timestamp", 0)) if isinstance(item, dict) else 0)
    return [str(info.get("token", info.get("sample_token", ""))) for info in infos if isinstance(info, dict) and info.get("token", info.get("sample_token", ""))]


def _box_size_xy(box: Dict[str, Any]) -> List[float]:
    size = box.get("size", [2.0, 4.8, 1.8])
    try:
        width = max(float(size[0]), 0.1)
        length = max(float(size[1]), 0.1)
    except (TypeError, ValueError, IndexError):
        width, length = 2.0, 4.8
    return [width, length]


def _tracking_delta(box: Dict[str, Any], dt_s: float) -> List[float]:
    velocity = box.get("velocity", [0.0, 0.0])
    try:
        vx = float(velocity[0])
        vy = float(velocity[1])
    except (TypeError, ValueError, IndexError):
        vx, vy = 0.0, 0.0
    return [-vx * float(dt_s), -vy * float(dt_s)]


def _to_centertrack_item(box: Dict[str, Any], dt_s: float) -> Dict[str, Any]:
    x, y, _z = [float(v) for v in box["translation"][:3]]
    width, length = _box_size_xy(box)
    name = str(box.get("detection_name", box.get("tracking_name", "car")) or "car")
    return {
        # CenterTrack tracker expects numpy vectors so `ct + tracking` is vector add, not list concat.
        "ct": np.asarray([x, y], dtype=np.float32),
        "bbox": [x - width / 2.0, y - length / 2.0, x + width / 2.0, y + length / 2.0],
        "class": int(CATEGORY_TO_ID.get(name, 1)),
        "score": float(box.get("detection_score", box.get("tracking_score", 0.0)) or 0.0),
        "tracking": np.asarray(_tracking_delta(box, dt_s), dtype=np.float32),
        "_source_box": box,
    }


def _from_centertrack_item(item: Dict[str, Any]) -> Dict[str, Any]:
    box = dict(item.get("_source_box", {}))
    box.pop("detection_score", None)
    box["tracking_id"] = str(item["tracking_id"])
    box["tracking_name"] = str(box.get("detection_name", box.get("tracking_name", "car")) or "car")
    box["tracking_score"] = float(item.get("score", box.get("tracking_score", 0.0)) or 0.0)
    box.setdefault("velocity", [0.0, 0.0])
    return box


def run(
    *,
    repo_root: Path,
    detection_json: Path,
    sample_info_pkl: Path,
    output_json: Path,
    score_threshold: float,
    max_age: int,
    new_thresh: float,
    dt_s: float,
) -> Path:
    lib_root = repo_root / "src" / "lib"
    if str(lib_root) not in sys.path:
        sys.path.insert(0, str(lib_root))
    from utils.tracker import Tracker

    payload = _load_json(detection_json)
    detections = payload.get("results", payload)
    if not isinstance(detections, dict):
        raise RuntimeError(f"detection json 格式不支持: {detection_json}")
    tokens = _load_sample_tokens(sample_info_pkl)
    tracker = Tracker(SimpleNamespace(max_age=int(max_age), new_thresh=float(new_thresh), hungarian=False, public_det=False))
    results: Dict[str, List[Dict[str, Any]]] = {}
    for frame_index, token in enumerate(tokens):
        frame_boxes = []
        for box in detections.get(token, []):
            if not isinstance(box, dict) or "translation" not in box:
                continue
            score = float(box.get("detection_score", 0.0) or 0.0)
            if score < float(score_threshold):
                continue
            frame_boxes.append(_to_centertrack_item(box, dt_s=dt_s))
        if frame_index == 0:
            tracker.init_track(frame_boxes)
            tracked = tracker.tracks
        else:
            tracked = tracker.step(frame_boxes)
        results[token] = [_from_centertrack_item(item) for item in tracked if int(item.get("active", 1)) > 0]

    out_payload = {
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
            "tracker": "centertrack_official_tracker_only",
            "note": "Uses CenterTrack utils.tracker.Tracker association on upstream nuScenes 3D detection boxes; CenterTrack image network is not executed.",
        },
        "results": results,
    }
    return _write_json(output_json, out_payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--detection-json", type=Path, required=True)
    parser.add_argument("--sample-info-pkl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--max-age", type=int, default=3)
    parser.add_argument("--new-thresh", type=float, default=0.3)
    parser.add_argument("--dt-s", type=float, default=0.5)
    args = parser.parse_args()
    print(
        run(
            repo_root=args.repo_root.resolve(),
            detection_json=args.detection_json.resolve(),
            sample_info_pkl=args.sample_info_pkl.resolve(),
            output_json=args.output_json.resolve(),
            score_threshold=args.score_threshold,
            max_age=args.max_age,
            new_thresh=args.new_thresh,
            dt_s=args.dt_s,
        )
    )


if __name__ == "__main__":
    main()
