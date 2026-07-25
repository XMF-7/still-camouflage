#!/usr/bin/env python3

import argparse
import collections
import collections.abc
import json
import math
import pickle
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape as _xml_escape

import numpy as np
import pandas as pd
import torch
import yaml
from pyquaternion import Quaternion


EVAL_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = EVAL_ROOT / "config.yaml"
DEFAULT_INPUT_FRAMES = 3
DEFAULT_TOKEN_SELECTION = "last"
DEFAULT_PREDICTION_HORIZONS_S = [3, 4]
DEFAULT_PREDICTION_FREQUENCY_HZ = 2.0
DEFAULT_HISTORY_STEPS_BEFORE_FIRST_TRACKING = 8
DEFAULT_NUM_SAMPLES = 1
DEFAULT_Z_MODE = True
DEFAULT_GMM_MODE = True
TRACKING_MODEL_ID_MAP = {
    1: "ab3dmot",
    2: "centertrack",
}
TRACKING_SHORT = {
    "ab3dmot": "track-ab",
    "centertrack": "track-centertrack",
}
DETECTION_MODEL_ID_MAP = {
    1: "bevdet",
    2: "bevdepth",
    3: "fastbev",
}
PREDICTION_MODEL_ID_MAP = {
    1: "trajectronpp",
    2: "hivt",
}
EXTERNAL_PREDICTION_ADAPTERS = set()
DEFAULT_TRAJECTRON_REPO_ROOT = "/home/jushuo/Code/zz6-trajectron/Trajectron-plus-plus"
DEFAULT_TRAJECTRON_MODEL_DIR = "/home/jushuo/Code/zz6-trajectron/Trajectron-plus-plus/experiments/nuScenes/models/int_ee_me"

for _name in ("Sequence", "Mapping", "MutableMapping", "Iterable"):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))


class SampleSpec:
    def __init__(
        self,
        name: str,
        path: Path,
        sample_id: str,
        sample_tokens: List[str],
        target_instance_token: str,
        tracking_json_by_phase: Dict[str, Path],
    ) -> None:
        self.name = name
        self.path = path
        self.sample_id = sample_id
        self.sample_tokens = sample_tokens
        self.target_instance_token = target_instance_token
        self.tracking_json_by_phase = tracking_json_by_phase


def _phases(cfg: Dict[str, Any]) -> List[str]:
    del cfg
    return ["clean"]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def _load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _format_template(template: str, sample_name: str, sample_id: str) -> Path:
    return Path(template.format(sample=sample_name, sample_id=sample_id))


def _tracking_json_path(cfg: Dict[str, Any], sample_name: str, phase: str) -> Path:
    return Path(cfg["output"]["dir"]) / sample_name / _tracking_output_subdir(cfg) / phase / "tracking.json"


def _prediction_json_path(cfg: Dict[str, Any], sample_name: str, phase: str) -> Path:
    return Path(cfg["output"]["dir"]) / sample_name / _prediction_output_subdir(cfg) / f"{sample_name}-{phase}.json"


def _tracking_adapter(cfg: Dict[str, Any]) -> str:
    tracking_cfg = cfg.get("tracking", {}) if isinstance(cfg.get("tracking", {}), dict) else {}
    model_id = tracking_cfg.get("model_id", None)
    if model_id is not None:
        try:
            return TRACKING_MODEL_ID_MAP[int(model_id)]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"tracking.model_id={model_id!r} 不支持；可选: {TRACKING_MODEL_ID_MAP}") from exc
    return str(tracking_cfg.get("adapter", tracking_cfg.get("model", "ab3dmot")) or "ab3dmot").strip().lower()


def _tracking_output_subdir(cfg: Dict[str, Any]) -> str:
    return f"{_legacy_tracking_output_subdir(cfg)}-from-{_detection_output_subdir(cfg)}"


def _legacy_tracking_output_subdir(cfg: Dict[str, Any]) -> str:
    adapter = _tracking_adapter(cfg)
    return TRACKING_SHORT.get(adapter, f"track-{adapter.replace('_', '-')}")


def _prediction_adapter(cfg: Dict[str, Any]) -> str:
    prediction_cfg = cfg.get("prediction", {}) if isinstance(cfg.get("prediction", {}), dict) else {}
    model_id = prediction_cfg.get("model_id", None)
    if model_id is not None:
        try:
            return PREDICTION_MODEL_ID_MAP[int(model_id)]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"prediction.model_id={model_id!r} 不支持；可选: {PREDICTION_MODEL_ID_MAP}") from exc
    legacy = cfg.get("trajectron", {}) if isinstance(cfg.get("trajectron", {}), dict) else {}
    return str(prediction_cfg.get("adapter", prediction_cfg.get("model", legacy.get("adapter", "trajectronpp"))) or "trajectronpp").strip().lower()


def _detection_model(cfg: Dict[str, Any]) -> str:
    detection_cfg = cfg.get("detection", {}) if isinstance(cfg.get("detection", {}), dict) else {}
    model_id = detection_cfg.get("model_id", None)
    if model_id is not None:
        try:
            return DETECTION_MODEL_ID_MAP[int(model_id)]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"detection.model_id={model_id!r} 不支持；可选: {DETECTION_MODEL_ID_MAP}") from exc
    explicit = str(detection_cfg.get("model", detection_cfg.get("detector", "")) or "").strip().lower()
    if explicit:
        return explicit
    config_path = detection_cfg.get("config_path", "")
    if config_path and Path(config_path).exists():
        return str(_load_yaml(Path(config_path)).get("model", "bevdet")).strip().lower()
    return "bevdet"


def _detection_source_tag(cfg: Dict[str, Any]) -> str:
    tracking_cfg = cfg.get("tracking", {}) if isinstance(cfg.get("tracking", {}), dict) else {}
    explicit_tag = str(tracking_cfg.get("detection_source_tag", "") or "").strip().lower()
    if explicit_tag:
        return explicit_tag.replace("_", "-")
    detection_input_dir = str(tracking_cfg.get("detection_input_dir", "") or "").strip()
    if detection_input_dir:
        return "external-json"
    return _detection_model(cfg).replace("_", "-")


def _detection_output_subdir(cfg: Dict[str, Any]) -> str:
    return f"det-{_detection_source_tag(cfg)}"


def _prediction_output_subdir(cfg: Dict[str, Any]) -> str:
    return f"pred-{_prediction_adapter(cfg).replace('_', '-')}-from-{_tracking_output_subdir(cfg)}"


def _prediction_cfg(cfg: Dict[str, Any], adapter: str) -> Dict[str, Any]:
    prediction_cfg = cfg.get("prediction", {}) if isinstance(cfg.get("prediction", {}), dict) else {}
    nested = prediction_cfg.get(adapter, {}) if isinstance(prediction_cfg.get(adapter, {}), dict) else {}
    merged = dict(nested)
    return merged


def _sample_items(raw_samples: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_samples, str):
        return [
            {"sample": part.strip()}
            for part in raw_samples.split(",")
            if part.strip()
        ]
    return [
        {"sample": raw} if isinstance(raw, str) else dict(raw)
        for raw in (raw_samples or [])
    ]


def _sample_name(path_or_name: Any) -> str:
    text = str(path_or_name).rstrip("/")
    return Path(text).name if "/" in text else text


def _sample_id(sample_name: str) -> str:
    return sample_name.split("sample-", 1)[1] if sample_name.startswith("sample-") else sample_name


def _scenario_root(cfg: Dict[str, Any]) -> Path:
    return Path(cfg.get("paths", {}).get("scenario_root", "/home/jushuo/Code/zz0.1-scenario")).resolve()


def _scenario_roots(cfg: Dict[str, Any]) -> List[Path]:
    raw = cfg.get("paths", {}).get("scenario_roots", cfg.get("paths", {}).get("scenario_root", "/home/jushuo/Code/zz0.1-scenario"))
    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        items = [str(part).strip() for part in (raw or []) if str(part).strip()]
    return [Path(item).expanduser().resolve() for item in items]


def _resolve_sample_path(cfg: Dict[str, Any], item: Dict[str, Any]) -> Path:
    raw_path = str(item.get("path", "") or "").strip()
    raw_name = str(item.get("sample", item.get("name", "")) or "").strip()
    raw = raw_path or raw_name
    if not raw:
        raise ValueError("samples 里需要写 sample-xxx，或者旧格式 path/name")

    candidate = Path(raw).expanduser()
    if candidate.is_absolute() or "/" in raw or "\\" in raw:
        path = candidate.resolve()
        if not path.exists():
            raise FileNotFoundError(f"sample 路径不存在: {path}")
        return path

    sample_name = _sample_name(raw)
    roots = _scenario_roots(cfg)
    case = str(item.get("case", "") or "").strip()
    if case:
        matches = [(root / case / sample_name).resolve() for root in roots if (root / case / sample_name).exists()]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            tried = "\n".join(str((root / case / sample_name).resolve()) for root in roots)
            raise FileNotFoundError(f"sample 不存在，已尝试:\n{tried}")
        match_lines = "\n".join(str(path) for path in matches)
        raise RuntimeError(f"{sample_name} 在多个 scenario_root 下匹配到 case={case}，请使用绝对 path:\n{match_lines}")

    matches = sorted(path for root in roots for path in root.glob(f"*/{sample_name}") if path.is_dir())
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        root_lines = "\n".join(str(root) for root in roots)
        raise FileNotFoundError(f"在以下 scenario_root 下找不到 {sample_name}:\n{root_lines}")
    match_lines = "\n".join(str(path) for path in matches)
    raise RuntimeError(f"{sample_name} 在 scenario_root 下匹配到多个 case/root，请写 case 字段或绝对 path:\n{match_lines}")


def _add_paths(cfg: Dict[str, Any]) -> Path:
    tcfg = cfg.get("trajectron", {}) if isinstance(cfg.get("trajectron", {}), dict) else {}
    tr_root = Path(tcfg.get("repo_root", DEFAULT_TRAJECTRON_REPO_ROOT)).resolve()
    for path in reversed([
        tr_root / "trajectron",
        tr_root / "experiments" / "nuScenes",
        tr_root / "experiments" / "nuScenes" / "devkit" / "python-sdk",
    ]):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    camo_repo = Path(cfg.get("paths", {}).get("camouflage_repo", "") or "")
    if camo_repo and camo_repo.exists() and str(camo_repo) not in sys.path:
        sys.path.append(str(camo_repo))
    return tr_root


def _info_tokens_from_payload(payload: Any, pkl_path: Path) -> List[str]:
    infos = payload.get("infos") if isinstance(payload, dict) else payload
    if not isinstance(infos, list) or not infos:
        raise RuntimeError(f"pkl 中没有 infos: {pkl_path}")
    infos = sorted(infos, key=lambda item: int(item.get("timestamp", 0)) if isinstance(item, dict) else 0)
    tokens = [str(info.get("token", info.get("sample_token", ""))) for info in infos if isinstance(info, dict)]
    tokens = [token for token in tokens if token]
    if not tokens:
        raise RuntimeError(f"pkl 中没有 sample token: {pkl_path}")
    return tokens


def _load_info_tokens_with_external_python(pkl_path: Path, python_bin: str) -> List[str]:
    if not python_bin:
        raise RuntimeError("需要 Python >= 3.8 才能读取 protocol 5 pkl，但 env.detection_python 为空")
    code = r"""
import json
import pickle
import sys
from pathlib import Path

pkl_path = Path(sys.argv[1])
with pkl_path.open("rb") as fp:
    payload = pickle.load(fp)
infos = payload.get("infos") if isinstance(payload, dict) else payload
if not isinstance(infos, list) or not infos:
    raise RuntimeError(f"pkl has no infos: {pkl_path}")
infos = sorted(infos, key=lambda item: int(item.get("timestamp", 0)) if isinstance(item, dict) else 0)
tokens = [str(info.get("token", info.get("sample_token", ""))) for info in infos if isinstance(info, dict)]
tokens = [token for token in tokens if token]
if not tokens:
    raise RuntimeError(f"pkl has no sample token: {pkl_path}")
print(json.dumps(tokens))
"""
    proc = subprocess.run(
        [str(python_bin), "-c", code, str(pkl_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"读取 protocol 5 pkl 失败: {pkl_path}\n"
            f"python={python_bin}\n"
            f"{proc.stderr.strip()}"
        )
    return [str(token) for token in json.loads(proc.stdout)]


def _load_info_tokens(pkl_root: Path, sample_name: str, cfg: Dict[str, Any]) -> List[str]:
    candidates = [
        pkl_root / f"{sample_name}.pkl",
        pkl_root / f"{sample_name}-official.pkl",
    ]
    pkl_path = next((path for path in candidates if path.exists()), None)
    if pkl_path is None:
        raise FileNotFoundError(f"找不到 {sample_name} 的 pkl，已尝试: {candidates}")
    try:
        with pkl_path.open("rb") as fp:
            payload = pickle.load(fp)
        return _info_tokens_from_payload(payload, pkl_path)
    except ValueError as exc:
        if "unsupported pickle protocol" not in str(exc):
            raise
        python_bin = str(cfg.get("env", {}).get("detection_python", "") or "")
        return _load_info_tokens_with_external_python(pkl_path, python_bin)


def _prediction_horizons_s(cfg: Dict[str, Any]) -> List[int]:
    prediction_cfg = cfg.get("prediction", {}) if isinstance(cfg.get("prediction", {}), dict) else {}
    trajectron_cfg = cfg.get("trajectron", {}) if isinstance(cfg.get("trajectron", {}), dict) else {}
    raw = prediction_cfg.get("horizons_s", trajectron_cfg.get("horizons_s", DEFAULT_PREDICTION_HORIZONS_S))
    return [int(h) for h in raw]


def _prediction_frequency_hz(cfg: Dict[str, Any]) -> float:
    prediction_cfg = cfg.get("prediction", {}) if isinstance(cfg.get("prediction", {}), dict) else {}
    trajectron_cfg = cfg.get("trajectron", {}) if isinstance(cfg.get("trajectron", {}), dict) else {}
    return float(prediction_cfg.get("frequency_hz", trajectron_cfg.get("frequency_hz", DEFAULT_PREDICTION_FREQUENCY_HZ)))


def _history_steps_before_first_tracking(cfg: Dict[str, Any]) -> int:
    prediction_cfg = cfg.get("prediction", {}) if isinstance(cfg.get("prediction", {}), dict) else {}
    trajectron_cfg = cfg.get("trajectron", {}) if isinstance(cfg.get("trajectron", {}), dict) else {}
    return int(
        prediction_cfg.get(
            "history_steps_before_first_tracking",
            trajectron_cfg.get("history_steps_before_first_tracking", DEFAULT_HISTORY_STEPS_BEFORE_FIRST_TRACKING),
        )
    )


def _select_input_tokens(all_tokens: List[str], tracking_cfg: Dict[str, Any]) -> List[str]:
    n = int(tracking_cfg.get("input_frames", DEFAULT_INPUT_FRAMES))
    if len(all_tokens) < n:
        raise RuntimeError(f"sample token 数量不足 {n}: {all_tokens}")
    mode = str(tracking_cfg.get("token_selection", DEFAULT_TOKEN_SELECTION)).lower()
    if mode == "first":
        return all_tokens[:n]
    if mode == "last":
        return all_tokens[-n:]
    raise ValueError("tracking.token_selection 只能是 first / last；需要固定帧时请在 samples.sample_tokens 里显式写")


def _find_latest_binding(data_pre_root: Path, sample_name: str) -> Optional[Path]:
    sample_root = data_pre_root / sample_name
    if not sample_root.exists():
        return None
    candidates = sorted(
        sample_root.glob("cache-*/target-car-binding.yaml"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _find_sequence_yaml(data_yaml_root: Path, sample_name: str, sample_path: Optional[Path] = None) -> Optional[Path]:
    if sample_path is not None:
        case_candidate = data_yaml_root / sample_path.parent.name / f"{sample_name}.yaml"
        if case_candidate.exists():
            return case_candidate
    matches = sorted(data_yaml_root.glob(f"*/{sample_name}.yaml"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"{sample_name} 在 data_yaml_root 下有多个 yaml: {matches}")
    direct = data_yaml_root / f"{sample_name}.yaml"
    return direct if direct.exists() else None


def _resolve_target_instance(cfg: Dict[str, Any], sample_name: str, sample_path: Optional[Path] = None, explicit: str = "") -> str:
    if explicit:
        return explicit
    data_pre_root = Path(cfg["paths"]["data_pre_root"])
    binding = _find_latest_binding(data_pre_root, sample_name)
    if binding is not None:
        payload = _load_yaml(binding)
        token = str(payload.get("target", {}).get("instance_token", "") or "")
        if token:
            return token

    data_yaml_root = Path(cfg["paths"]["data_yaml_root"])
    sequence_yaml = _find_sequence_yaml(data_yaml_root, sample_name, sample_path)
    if sequence_yaml is None:
        raise FileNotFoundError(f"找不到 {sample_name} 的 sequence yaml，也没有可用 target-car-binding.yaml")
    from match_target_car import run_target_matching

    attack_config = Path(cfg.get("paths", {}).get("attack_config", "") or Path(cfg["paths"]["camouflage_repo"]) / "config.yaml")
    binding_payload, _bound_sequence, binding_path = run_target_matching(
        config_path=attack_config,
        sequence_yaml=sequence_yaml,
        verbose=False,
    )
    token = str(binding_payload.get("target", {}).get("instance_token", "") or "")
    if not token:
        raise RuntimeError(f"target matching 没有返回 instance token: {binding_path}")
    return token


def _load_sample_specs(cfg: Dict[str, Any]) -> List[SampleSpec]:
    pkl_root = Path(cfg["paths"]["sample_info_pkl_root"])
    tracking_cfg = cfg["tracking"]
    specs: List[SampleSpec] = []
    for item in _sample_items(cfg.get("samples", [])):
        path = _resolve_sample_path(cfg, item)
        name = str(item.get("name") or _sample_name(path))
        sid = _sample_id(name)
        tokens = [str(token) for token in item.get("sample_tokens", []) if str(token)]
        if not tokens:
            tokens = _select_input_tokens(_load_info_tokens(pkl_root, name, cfg), tracking_cfg)
        target_instance = _resolve_target_instance(
            cfg,
            name,
            path,
            str(item.get("target_instance_token", "") or ""),
        )
        tracking_json_by_phase: Dict[str, Path] = {}
        for phase in _phases(cfg):
            key = f"{phase}_tracking_json"
            tracking_json_by_phase[phase] = Path(item[key]) if item.get(key) else _tracking_json_path(cfg, name, phase)
        specs.append(
            SampleSpec(
                name=name,
                path=path,
                sample_id=sid,
                sample_tokens=tokens,
                target_instance_token=target_instance,
                tracking_json_by_phase=tracking_json_by_phase,
            )
        )
    return specs


def _yaw_from_quaternion(rotation: Sequence[float]) -> float:
    return float(Quaternion(rotation).yaw_pitch_roll[0])


def _annotation_for_instance_at_sample(nusc: Any, instance_token: str, sample_token: str) -> Optional[Dict[str, Any]]:
    sample = nusc.get("sample", sample_token)
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        if str(ann.get("instance_token", "")) == instance_token:
            return ann
    return None


def _record_from_annotation(ann: Dict[str, Any], source: str) -> Dict[str, Any]:
    return {
        "source": source,
        "sample_token": ann["sample_token"],
        "translation": list(ann["translation"]),
        "rotation": list(ann["rotation"]),
        "size": list(ann["size"]),
        "instance_token": ann["instance_token"],
    }


def _record_from_tracking_box(box: Dict[str, Any], sample_token: str) -> Dict[str, Any]:
    return {
        "source": "tracking",
        "sample_token": sample_token,
        "translation": list(box["translation"]),
        "rotation": list(box.get("rotation", [1.0, 0.0, 0.0, 0.0])),
        "size": list(box.get("size", [np.nan, np.nan, np.nan])),
        "tracking_id": str(box.get("tracking_id")),
        "tracking_name": box.get("tracking_name"),
        "tracking_score": box.get("tracking_score"),
    }


def _collect_gt_history(nusc: Any, instance_token: str, first_sample_token: str, max_steps: int) -> List[Dict[str, Any]]:
    first_ann = _annotation_for_instance_at_sample(nusc, instance_token, first_sample_token)
    if first_ann is None:
        raise RuntimeError(f"target instance 不在第一帧 tracking sample 中: {first_sample_token}")
    records: List[Dict[str, Any]] = []
    ann_token = first_ann["prev"]
    while ann_token and len(records) < max_steps:
        ann = nusc.get("sample_annotation", ann_token)
        records.append(_record_from_annotation(ann, "gt_history"))
        ann_token = ann["prev"]
    records.reverse()
    return records


def _collect_gt_records(nusc: Any, instance_token: str, sample_tokens: List[str], source: str) -> List[Dict[str, Any]]:
    records = []
    for sample_token in sample_tokens:
        ann = _annotation_for_instance_at_sample(nusc, instance_token, sample_token)
        if ann is None:
            raise RuntimeError(f"target instance 不在 sample 中，无法取 GT: {sample_token}")
        records.append(_record_from_annotation(ann, source))
    return records


def _collect_future_gt(nusc: Any, instance_token: str, current_sample_token: str, ph: int) -> np.ndarray:
    current = _annotation_for_instance_at_sample(nusc, instance_token, current_sample_token)
    if current is None:
        raise RuntimeError(f"target instance 不在当前 sample 中，无法取未来 GT: {current_sample_token}")
    rows: List[List[float]] = []
    ann_token = current["next"]
    while ann_token and len(rows) < ph:
        ann = nusc.get("sample_annotation", ann_token)
        if str(ann.get("instance_token", "")) != instance_token:
            break
        rows.append([float(ann["translation"][0]), float(ann["translation"][1])])
        ann_token = ann["next"]
    return np.asarray(rows, dtype=float).reshape((-1, 2))


def _tracking_results(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"tracking json 不存在: {path}")
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    results = payload.get("results", payload)
    if not isinstance(results, dict):
        raise ValueError(f"tracking json 格式不支持: {path}")
    return results


def _is_vehicle_box(box: Dict[str, Any]) -> bool:
    name = str(box.get("tracking_name", box.get("detection_name", ""))).lower()
    return name in {"", "car", "vehicle", "truck", "bus", "trailer"} or "vehicle" in name


def _resolve_tracking_id(
    *,
    nusc: Any,
    tracking_results: Dict[str, List[Dict[str, Any]]],
    sample_tokens: List[str],
    instance_token: str,
    max_dist_m: float,
) -> str:
    costs: Dict[str, List[float]] = {}
    for sample_token in sample_tokens:
        ann = _annotation_for_instance_at_sample(nusc, instance_token, sample_token)
        if ann is None:
            continue
        gt_xy = np.asarray(ann["translation"][:2], dtype=float)
        for box in tracking_results.get(sample_token, []):
            if not _is_vehicle_box(box) or "translation" not in box:
                continue
            tid = str(box.get("tracking_id", ""))
            if not tid:
                continue
            det_xy = np.asarray(box["translation"][:2], dtype=float)
            costs.setdefault(tid, []).append(float(np.linalg.norm(det_xy - gt_xy)))
    if not costs:
        raise RuntimeError("无法通过 GT target 和 tracking box 匹配出 tracking_id")
    best_tid, best_values = min(costs.items(), key=lambda item: float(np.mean(item[1])))
    best_dist = float(np.mean(best_values))
    if best_dist > max_dist_m:
        raise RuntimeError(f"最近 tracking_id={best_tid} 平均距离 {best_dist:.3f}m，超过阈值 {max_dist_m:.3f}m")
    return best_tid


def _target_track_binding_path(cfg: Dict[str, Any], sample_name: str, phase: str) -> Path:
    return Path(cfg["output"]["dir"]) / sample_name / _tracking_output_subdir(cfg) / phase / "target-bind.json"


def _write_stage_error(cfg: Dict[str, Any], sample_name: str, phase: str, stage: str, exc: Exception) -> Path:
    path = Path(cfg["output"]["dir"]) / sample_name / "errors" / f"{phase}-{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample": sample_name,
        "phase": phase,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _clear_stage_error(cfg: Dict[str, Any], sample_name: str, phase: str, stage: str) -> None:
    path = Path(cfg["output"]["dir"]) / sample_name / "errors" / f"{phase}-{stage}.json"
    if path.exists():
        path.unlink()


def _compute_target_track_binding(
    *,
    nusc: Any,
    tracking_results: Dict[str, List[Dict[str, Any]]],
    sample: SampleSpec,
    phase: str,
    max_dist_m: float,
) -> Dict[str, Any]:
    costs: Dict[str, List[float]] = {}
    score_map: Dict[str, List[float]] = {}
    frame_hits: Dict[str, set] = {}
    frame_candidates: Dict[str, List[Dict[str, Any]]] = {}
    for sample_token in sample.sample_tokens:
        ann = _annotation_for_instance_at_sample(nusc, sample.target_instance_token, sample_token)
        if ann is None:
            frame_candidates[sample_token] = []
            continue
        gt_xy = np.asarray(ann["translation"][:2], dtype=float)
        rows = []
        for box in tracking_results.get(sample_token, []):
            if not _is_vehicle_box(box) or "translation" not in box:
                continue
            tid = str(box.get("tracking_id", ""))
            if not tid:
                continue
            det_xy = np.asarray(box["translation"][:2], dtype=float)
            dist = float(np.linalg.norm(det_xy - gt_xy))
            costs.setdefault(tid, []).append(dist)
            frame_hits.setdefault(tid, set()).add(sample_token)
            score = box.get("tracking_score", None)
            if score is not None:
                try:
                    score_map.setdefault(tid, []).append(float(score))
                except Exception:
                    pass
            rows.append(
                {
                    "tracking_id": tid,
                    "tracking_center_world": [float(v) for v in box.get("translation", [])],
                    "tracking_score": box.get("tracking_score"),
                    "distance_to_target_gt_m": dist,
                }
            )
        frame_candidates[sample_token] = rows

    if not costs:
        return {
            "sample": sample.name,
            "phase": phase,
            "target_instance_token": sample.target_instance_token,
            "input_sample_tokens": sample.sample_tokens,
            "matched": False,
            "failure_reason": "no vehicle tracking boxes can be compared with target GT",
            "frames": [],
        }

    def _candidate_stat(tid: str, values: List[float]) -> Dict[str, Any]:
        arr = [float(v) for v in values]
        close = [v for v in arr if v <= float(max_dist_m)]
        mean_dist = float(np.mean(arr))
        mean_close = float(np.mean(close)) if close else float("inf")
        max_dist = float(np.max(arr))
        hit_count = int(len(frame_hits.get(tid, set())))
        close_hits = int(len(close))
        scores = score_map.get(tid, [])
        mean_score = float(np.mean(scores)) if scores else float("-inf")
        return {
            "tracking_id": tid,
            "mean_distance_to_target_gt_m": mean_dist,
            "mean_close_distance_m": mean_close,
            "max_distance_to_target_gt_m": max_dist,
            "hit_frame_count": hit_count,
            "close_match_frame_count": close_hits,
            "mean_tracking_score": mean_score,
        }

    candidates = [_candidate_stat(tid, values) for tid, values in costs.items()]
    candidates.sort(
        key=lambda row: (
            -int(row["close_match_frame_count"]),
            -int(row["hit_frame_count"]),
            float(row["mean_close_distance_m"]),
            float(row["mean_distance_to_target_gt_m"]),
            -float(row["mean_tracking_score"]),
            float(row["max_distance_to_target_gt_m"]),
        )
    )
    best = candidates[0]
    best_tid = str(best["tracking_id"])
    best_dist = float(best["mean_distance_to_target_gt_m"])
    frames = []
    for sample_token in sample.sample_tokens:
        ann = _annotation_for_instance_at_sample(nusc, sample.target_instance_token, sample_token)
        gt_center = [float(v) for v in ann["translation"]] if ann is not None else []
        gt_center_ego = (
            _global_xy_to_current_ego(nusc, sample_token, np.asarray([gt_center[:2]], dtype=float))[0].tolist()
            if gt_center
            else []
        )
        best_row = next((row for row in frame_candidates.get(sample_token, []) if row["tracking_id"] == best_tid), None)
        tracking_center = [] if best_row is None else best_row["tracking_center_world"]
        tracking_center_ego = (
            _global_xy_to_current_ego(nusc, sample_token, np.asarray([tracking_center[:2]], dtype=float))[0].tolist()
            if tracking_center
            else []
        )
        delta_x_m = None
        delta_y_m = None
        moved_toward_y0_m = None
        if gt_center_ego and tracking_center_ego:
            delta_x_m = float(tracking_center_ego[0] - gt_center_ego[0])
            delta_y_m = float(tracking_center_ego[1] - gt_center_ego[1])
            direction_to_y0 = -1.0 if float(gt_center_ego[1]) >= 0.0 else 1.0
            moved_toward_y0_m = float(direction_to_y0 * delta_y_m)
        frames.append(
            {
                "sample_token": sample_token,
                "target_gt_center_world": gt_center,
                "target_gt_center_ego": gt_center_ego,
                "tracking_center_world": tracking_center,
                "tracking_center_ego": tracking_center_ego,
                "distance_to_target_gt_m": None if best_row is None else best_row["distance_to_target_gt_m"],
                "delta_x_m": delta_x_m,
                "delta_y_m": delta_y_m,
                "moved_toward_y0_m": moved_toward_y0_m,
                "tracking_score": None if best_row is None else best_row.get("tracking_score"),
                "matched": best_row is not None,
            }
        )

    matched_frames = sum(1 for row in frames if bool(row["matched"]))
    close_match_frames = int(best["close_match_frame_count"])
    within_dist = bool(close_match_frames > 0 and best_dist <= float(max_dist_m))
    overall_matched = bool(within_dist and matched_frames > 0)
    failure_reason = ""
    if not within_dist:
        if close_match_frames <= 0:
            failure_reason = (
                f"no frame within threshold {float(max_dist_m):.3f}m; "
                f"best tracking_id={best_tid} mean distance {best_dist:.3f}m"
            )
        else:
            failure_reason = f"mean distance {best_dist:.3f}m exceeds threshold {float(max_dist_m):.3f}m"
    elif matched_frames <= 0:
        failure_reason = "target tracking id is not observed in selected input frames"

    return {
        "sample": sample.name,
        "phase": phase,
        "target_instance_token": sample.target_instance_token,
        "target_tracking_id": best_tid,
        "input_sample_tokens": sample.sample_tokens,
        "mean_distance_to_target_gt_m": best_dist,
        "max_allowed_distance_m": float(max_dist_m),
        "matched": overall_matched,
        "failure_reason": failure_reason,
        "matched_frame_count": int(matched_frames),
        "close_match_frame_count": close_match_frames,
        "input_frame_count": int(len(frames)),
        "candidate_ranking": candidates[:10],
        "frames": frames,
    }


def _load_or_create_target_track_binding(
    *,
    nusc: Any,
    tracking_results: Dict[str, List[Dict[str, Any]]],
    sample: SampleSpec,
    phase: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    path = _target_track_binding_path(cfg, sample.name, phase)
    payload = _compute_target_track_binding(
        nusc=nusc,
        tracking_results=tracking_results,
        sample=sample,
        phase=phase,
        max_dist_m=float(cfg["tracking"].get("target_match_max_dist_m", 2.0)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return payload


def _collect_tracking_records(
    tracking_results: Dict[str, List[Dict[str, Any]]],
    sample_tokens: List[str],
    tracking_id: str,
) -> List[Dict[str, Any]]:
    records = []
    for sample_token in sample_tokens:
        box = next((row for row in tracking_results.get(sample_token, []) if str(row.get("tracking_id")) == str(tracking_id)), None)
        if box is None:
            continue
        records.append(_record_from_tracking_box(box, sample_token))
    if not records:
        raise RuntimeError(f"tracking_id={tracking_id} 在选定输入帧中不存在")
    return records


def _standardization(cfg: Dict[str, Any] = None) -> Dict[str, Any]:
    velocity_std = 15.0
    if cfg is not None:
        tcfg = cfg.get("trajectron", {}) if isinstance(cfg.get("trajectron", {}), dict) else {}
        velocity_std = float(tcfg.get("velocity_std", 15.0))

    return {
        "PEDESTRIAN": {
            "position": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
            "velocity": {"x": {"mean": 0, "std": 2}, "y": {"mean": 0, "std": 2}},
            "acceleration": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
        },
        "VEHICLE": {
            "position": {"x": {"mean": 0, "std": 80}, "y": {"mean": 0, "std": 80}},
            "velocity": {"x": {"mean": 0, "std": velocity_std}, 
                        "y": {"mean": 0, "std": velocity_std}, 
                        "norm": {"mean": 0, "std": velocity_std}},
            "acceleration": {"x": {"mean": 0, "std": 4}, "y": {"mean": 0, "std": 4}, "norm": {"mean": 0, "std": 4}},
            "heading": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}, "°": {"mean": 0, "std": np.pi}, "d°": {"mean": 0, "std": 1}},
        },
    }


def _vehicle_columns() -> pd.MultiIndex:
    columns = pd.MultiIndex.from_product([["position", "velocity", "acceleration", "heading"], ["x", "y"]])
    columns = columns.append(pd.MultiIndex.from_tuples([("heading", "°"), ("heading", "d°")]))
    return columns.append(pd.MultiIndex.from_product([["velocity", "acceleration"], ["norm"]]))


def _build_vehicle_dataframe(records: List[Dict[str, Any]], origin_xy: np.ndarray, dt: float) -> pd.DataFrame:
    from environment import derivative_of

    xy_global = np.asarray([r["translation"][:2] for r in records], dtype=float)
    xy = xy_global - origin_xy.reshape(1, 2)
    x = xy[:, 0]
    y = xy[:, 1]
    heading = np.unwrap(np.asarray([_yaw_from_quaternion(r["rotation"]) for r in records], dtype=float))
    vx = derivative_of(x, dt)
    vy = derivative_of(y, dt)
    ax = derivative_of(vx, dt)
    ay = derivative_of(vy, dt)
    v = np.stack((vx, vy), axis=-1)
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.0))
    data = {
        ("position", "x"): x,
        ("position", "y"): y,
        ("velocity", "x"): vx,
        ("velocity", "y"): vy,
        ("velocity", "norm"): np.linalg.norm(v, axis=-1),
        ("acceleration", "x"): ax,
        ("acceleration", "y"): ay,
        ("acceleration", "norm"): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
        ("heading", "x"): heading_v[:, 0],
        ("heading", "y"): heading_v[:, 1],
        ("heading", "°"): heading,
        ("heading", "d°"): derivative_of(heading, dt, radian=True),
    }
    return pd.DataFrame(data, columns=_vehicle_columns())


def _ensure_map_json(dataroot: Path, map_name: str) -> None:
    map_json = dataroot / "maps" / f"{map_name}.json"
    expansion = dataroot / "maps" / "expansion" / f"{map_name}.json"
    if not map_json.exists() and expansion.exists():
        try:
            map_json.symlink_to(expansion)
        except OSError:
            shutil.copyfile(str(expansion), str(map_json))


def _build_map(nusc: Any, dataroot: Path, sample_token: str, xy_global: np.ndarray, origin_xy: np.ndarray) -> Dict[str, Any]:
    from environment import GeometricMap
    from nuscenes.map_expansion.map_api import NuScenesMap

    sample = nusc.get("sample", sample_token)
    ns_scene = nusc.get("scene", sample["scene_token"])
    map_name = nusc.get("log", ns_scene["log_token"])["location"]
    _ensure_map_json(dataroot, map_name)
    nusc_map = NuScenesMap(dataroot=str(dataroot), map_name=map_name)
    x_min = float(origin_xy[0])
    y_min = float(origin_xy[1])
    x_max = float(np.ceil(np.max(xy_global[:, 0]) + 50.0))
    y_max = float(np.ceil(np.max(xy_global[:, 1]) + 50.0))
    x_size = x_max - x_min
    y_size = y_max - y_min
    patch_box = (x_min + 0.5 * x_size, y_min + 0.5 * y_size, y_size, x_size)
    canvas_size = (np.round(3 * y_size).astype(int), np.round(3 * x_size).astype(int))
    homography = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    layers = ["lane", "road_segment", "drivable_area", "road_divider", "lane_divider", "stop_line", "ped_crossing", "stop_line", "ped_crossing", "walkway"]
    mask = (nusc_map.get_map_mask(patch_box, 0, layers, canvas_size) * 255.0).astype(np.uint8)
    mask = np.swapaxes(mask, 1, 2)
    return {
        "PEDESTRIAN": GeometricMap(data=np.stack((mask[9], mask[8], np.max(mask[:3], axis=0)), axis=0), homography=homography, description=", ".join(layers)),
        "VEHICLE": GeometricMap(data=np.stack((np.max(mask[:3], axis=0), mask[3], mask[4]), axis=0), homography=homography, description=", ".join(layers)),
    }


def _make_scene(
    nusc: Any,
    dataroot: Path,
    records: List[Dict[str, Any]],
    node_id: str,
    dt: float,
    scene_name: str,
    cfg: Dict[str, Any],
) -> Tuple[Any, Any, Any, np.ndarray]:
    from environment import Environment, Node, Scene

    xy_global = np.asarray([r["translation"][:2] for r in records], dtype=float)
    origin_xy = np.floor(np.min(xy_global, axis=0) - 50.0)
    env = Environment(node_type_list=["VEHICLE", "PEDESTRIAN"], standardization=_standardization(cfg))
    env.attention_radius = {
        (env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): 10.0,
        (env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE): 20.0,
        (env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN): 20.0,
        (env.NodeType.VEHICLE, env.NodeType.VEHICLE): 30.0,
    }
    env.robot_type = env.NodeType.VEHICLE
    scene = Scene(timesteps=len(records), dt=dt, name=scene_name)
    scene.map = _build_map(nusc, dataroot, records[-1]["sample_token"], xy_global, origin_xy)
    node = Node(
        node_type=env.NodeType.VEHICLE,
        node_id=str(node_id),
        data=_build_vehicle_dataframe(records, origin_xy, dt),
        first_timestep=0,
    )
    scene.nodes.append(node)
    env.scenes = [scene]
    return env, scene, node, origin_xy


def _load_trajectron(cfg: Dict[str, Any], env: Any) -> Tuple[Any, Dict[str, Any]]:
    from model import Trajectron
    from model.model_registrar import ModelRegistrar

    tcfg = cfg.get("trajectron", {}) if isinstance(cfg.get("trajectron", {}), dict) else {}
    model_dir = Path(tcfg.get("model_dir", DEFAULT_TRAJECTRON_MODEL_DIR))
    registrar = ModelRegistrar(str(model_dir), "cpu")
    registrar.load_models(int(tcfg.get("checkpoint", 12)))
    with (model_dir / "config.json").open("r", encoding="utf-8") as fp:
        hyp = json.load(fp)
    hyp.update(tcfg.get("overrides", {}) or {})
    trajectron = Trajectron(registrar, hyp, None, "cpu")
    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyp


def _predict_global(
    trajectron: Any,
    env: Any,
    scene: Any,
    node: Any,
    origin_xy: np.ndarray,
    ph: int,
    cfg: Dict[str, Any],
) -> np.ndarray:
    tcfg = cfg["trajectron"]
    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    timestep = scene.timesteps - 1
    with torch.no_grad():
        pred = trajectron.predict(
            scene,
            np.asarray([timestep]),
            ph,
            num_samples=int(tcfg.get("num_samples", DEFAULT_NUM_SAMPLES)),
            min_history_timesteps=1,
            min_future_timesteps=0,
            z_mode=bool(tcfg.get("z_mode", DEFAULT_Z_MODE)),
            gmm_mode=bool(tcfg.get("gmm_mode", DEFAULT_GMM_MODE)),
        )
    pred_local = pred[timestep][node]
    return pred_local + origin_xy.reshape(1, 1, 1, 2)


def _ego_pose_for_sample(nusc: Any, sample_token: str) -> Dict[str, Any]:
    sample = nusc.get("sample", sample_token)
    sample_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    return nusc.get("ego_pose", sample_data["ego_pose_token"])


def _global_xy_to_current_ego(nusc: Any, sample_token: str, xy: np.ndarray) -> np.ndarray:
    from nuscenes.utils.geometry_utils import transform_matrix

    ego_pose = _ego_pose_for_sample(nusc, sample_token)
    global_from_ego = transform_matrix(ego_pose["translation"], Quaternion(ego_pose["rotation"]), inverse=False)
    ego_from_global = np.linalg.inv(global_from_ego)
    pts = np.concatenate([xy, np.zeros((xy.shape[0], 1)), np.ones((xy.shape[0], 1))], axis=1)
    return (ego_from_global @ pts.T).T[:, :2]


def _metrics(nusc: Any, sample_token: str, pred_xy: np.ndarray, gt_xy: np.ndarray) -> Dict[str, Any]:
    count = min(int(pred_xy.shape[0]), int(gt_xy.shape[0]))
    if count <= 0:
        return {"ade": "", "fde": "", "min_de": "", "matched_steps": 0}
    errors = np.linalg.norm(pred_xy[:count] - gt_xy[:count], axis=1)
    return {
        "ade": float(np.mean(errors)),
        "fde": float(errors[-1]),
        "min_de": float(np.min(errors)),
        "matched_steps": count,
    }


def _constant_velocity_xy(records: List[Dict[str, Any]], ph: int) -> np.ndarray:
    xy = np.asarray([row["translation"][:2] for row in records if "translation" in row], dtype=float).reshape((-1, 2))
    if xy.shape[0] == 0:
        raise RuntimeError("CV prediction 需要至少 1 个 observed record")
    if xy.shape[0] == 1:
        step = np.zeros((2,), dtype=float)
    else:
        step = xy[-1] - xy[-2]
    return np.asarray([xy[-1] + step * float(index + 1) for index in range(int(ph))], dtype=float).reshape((-1, 2))


def _evaluate_constant_velocity_phase(
    *,
    nusc: Any,
    sample: SampleSpec,
    phase: str,
    cfg: Dict[str, Any],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    if phase == "clean":
        gt_history = _collect_gt_history(
            nusc,
            sample.target_instance_token,
            sample.sample_tokens[0],
            _history_steps_before_first_tracking(cfg),
        )
        observed = gt_history + _collect_gt_records(nusc, sample.target_instance_token, sample.sample_tokens, "gt_clean")
        tracking_id = sample.target_instance_token
        tracking_json = ""
        target_binding_path = ""
    else:
        tracking = _tracking_results(sample.tracking_json_by_phase[phase])
        target_binding = _load_or_create_target_track_binding(
            nusc=nusc,
            tracking_results=tracking,
            sample=sample,
            phase=phase,
            cfg=cfg,
        )
        if not bool(target_binding.get("matched", False)):
            raise RuntimeError(f"target-track binding 失败: {target_binding.get('failure_reason', '')}")
        tracking_id = str(target_binding["target_tracking_id"])
        gt_history = _collect_gt_history(
            nusc,
            sample.target_instance_token,
            sample.sample_tokens[0],
            _history_steps_before_first_tracking(cfg),
        )
        observed = gt_history + _collect_tracking_records(tracking, sample.sample_tokens, tracking_id)
        tracking_json = str(sample.tracking_json_by_phase[phase])
        target_binding_path = str(_target_track_binding_path(cfg, sample.name, phase))

    metrics_by_ph: Dict[int, Dict[str, Any]] = {}
    pred_json: Dict[str, Any] = {
        "sample": sample.name,
        "phase": phase,
        "prediction_adapter": "constant_velocity",
        "prediction_source": "constant_velocity_from_tracking_or_gt_history",
        "tracking_json": tracking_json,
        "tracking_id": tracking_id,
        "target_track_binding": target_binding_path,
        "target_instance_token": sample.target_instance_token,
        "input_sample_tokens": sample.sample_tokens,
        "current_sample_token": sample.sample_tokens[-1],
        "observed_records": [{"source": r["source"], "sample_token": r["sample_token"], "translation": r["translation"]} for r in observed],
        "prediction_coordinate_frame": "nuscenes_global",
        "prediction_units": {"position": "meter", "time": "second"},
        "frequency_hz": _prediction_frequency_hz(cfg),
        "predictions": {},
        "metrics": {},
    }
    for horizon_s in _prediction_horizons_s(cfg):
        ph = int(round(float(horizon_s) * _prediction_frequency_hz(cfg)))
        pred_xy = _constant_velocity_xy(observed, ph)
        gt_xy = _collect_future_gt(nusc, sample.target_instance_token, sample.sample_tokens[-1], ph)
        metrics_by_ph[int(horizon_s)] = _metrics(nusc, sample.sample_tokens[-1], pred_xy, gt_xy)
        pred_json["predictions"][f"{int(horizon_s)}s"] = pred_xy.tolist()
        pred_json["metrics"][f"{int(horizon_s)}s"] = metrics_by_ph[int(horizon_s)]
    return metrics_by_ph, pred_json


def _phase_label(phase: str) -> str:
    return "干净" if phase == "clean" else "攻击"


def _evaluate_phase(
    *,
    nusc: Any,
    dataroot: Path,
    sample: SampleSpec,
    phase: str,
    cfg: Dict[str, Any],
    trajectron: Optional[Any],
) -> Tuple[Optional[Any], Dict[int, Dict[str, Any]], Dict[str, Any]]:
    adapter = _prediction_adapter(cfg)
    if adapter == "hivt":
        raise RuntimeError(
            "prediction.adapter='hivt' 已纳入可选集合，但当前仓库尚未实现 HiVT 的 nuScenes tracking.json -> native input adapter。"
            "这是当前唯一致命缺口。"
        )
    if adapter != "trajectronpp":
        raise ValueError(f"prediction.adapter={adapter!r} 不能走 Trajectron++ sample adapter")

    tracking = _tracking_results(sample.tracking_json_by_phase[phase])
    target_binding = _load_or_create_target_track_binding(
        nusc=nusc,
        tracking_results=tracking,
        sample=sample,
        phase=phase,
        cfg=cfg,
    )
    if not bool(target_binding.get("matched", False)):
        raise RuntimeError(f"target-track binding 失败: {target_binding.get('failure_reason', '')}")
    tracking_id = str(target_binding["target_tracking_id"])
    gt_history = _collect_gt_history(
        nusc,
        sample.target_instance_token,
        sample.sample_tokens[0],
        _history_steps_before_first_tracking(cfg),
    )
    tracking_records = _collect_tracking_records(tracking, sample.sample_tokens, tracking_id)
    observed = gt_history + tracking_records
    env, scene, node, origin_xy = _make_scene(
        nusc,
        dataroot,
        observed,
        node_id=tracking_id,
        dt=1.0 / _prediction_frequency_hz(cfg),
        scene_name=f"{sample.name}-{phase}",
        cfg=cfg,
    )
    if trajectron is None:
        trajectron, _hyp = _load_trajectron(cfg, env)
    metrics_by_ph: Dict[int, Dict[str, Any]] = {}
    pred_json: Dict[str, Any] = {
        "sample": sample.name,
        "phase": phase,
        "prediction_adapter": "trajectronpp",
        "prediction_source": "trajectronpp_from_tracking_history",
        "tracking_json": str(sample.tracking_json_by_phase[phase]),
        "tracking_id": tracking_id,
        "target_track_binding": str(_target_track_binding_path(cfg, sample.name, phase)),
        "target_instance_token": sample.target_instance_token,
        "input_sample_tokens": sample.sample_tokens,
        "current_sample_token": sample.sample_tokens[-1],
        "observed_records": [{"source": r["source"], "sample_token": r["sample_token"], "translation": r["translation"]} for r in observed],
        "predictions": {},
        "metrics": {},
    }
    for horizon_s in _prediction_horizons_s(cfg):
        ph = int(round(float(horizon_s) * _prediction_frequency_hz(cfg)))
        pred_global = _predict_global(trajectron, env, scene, node, origin_xy, ph, cfg)
        pred_xy = np.asarray(pred_global[0, 0, :, :], dtype=float)
        gt_xy = _collect_future_gt(nusc, sample.target_instance_token, sample.sample_tokens[-1], ph)
        metrics_by_ph[int(horizon_s)] = _metrics(nusc, sample.sample_tokens[-1], pred_xy, gt_xy)
        pred_json["predictions"][f"{int(horizon_s)}s"] = pred_xy.tolist()
        pred_json["metrics"][f"{int(horizon_s)}s"] = metrics_by_ph[int(horizon_s)]
    return trajectron, metrics_by_ph, pred_json


def _write_prediction_json(cfg: Dict[str, Any], payloads: Iterable[Dict[str, Any]]) -> None:
    for payload in payloads:
        sample = payload["sample"]
        phase = payload["phase"]
        path = _prediction_json_path(cfg, sample, phase)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)


def _col_letter(index: int) -> str:
    out = ""
    while index:
        index, rem = divmod(index - 1, 26)
        out = chr(65 + rem) + out
    return out


def _xlsx_cell(value: Any, row_idx: int, col_idx: int) -> str:
    ref = f"{_col_letter(col_idx)}{row_idx}"
    if value is None or value == "":
        return f'<c r="{ref}"/>'
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            return f'<c r="{ref}"><v>{float(value)}</v></c>'
        return f'<c r="{ref}"/>'
    text = _xml_escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _worksheet_xml(rows: List[List[Any]]) -> str:
    body = []
    for r_idx, row in enumerate(rows, start=1):
        cells = "".join(_xlsx_cell(value, r_idx, c_idx) for c_idx, value in enumerate(row, start=1))
        body.append(f'<row r="{r_idx}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(body)}</sheetData>'
        '</worksheet>'
    )


def _write_minimal_xlsx(path: Path, sheets: Dict[str, List[List[Any]]]) -> None:
    sheet_names = list(sheets.keys())
    workbook_sheets = "".join(
        f'<sheet name="{_xml_escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, name in enumerate(sheet_names, start=1)
    )
    workbook_rels = "".join(
        f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        for idx, _name in enumerate(sheet_names, start=1)
    )
    overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for idx, _name in enumerate(sheet_names, start=1)
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            f"{overrides}</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>',
        )
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{workbook_sheets}</sheets></workbook>",
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{workbook_rels}</Relationships>",
        )
        for idx, name in enumerate(sheet_names, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _worksheet_xml(sheets[name]))


def _write_xlsx(path: Path, rows_by_horizon: Dict[int, List[Dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    comments = {
        "样本": "sample-xxx 名称。",
        "当前帧token": "三帧 tracking 输入里的最后一帧，也是 Trajectron 预测起点。",
        "干净ADE": "干净 tracking 替换最近三帧历史后的 ADE。",
        "干净FDE": "干净 tracking 替换最近三帧历史后的 FDE。",
        "干净最小距离": "干净 prediction 未来轨迹到未来 GT 的逐帧 L2 距离最小值。",
        "攻击ADE": "攻击 tracking 替换最近三帧历史后的 ADE。",
        "攻击FDE": "攻击 tracking 替换最近三帧历史后的 FDE。",
        "攻击最小距离": "攻击 prediction 未来轨迹到未来 GT 的逐帧 L2 距离最小值。",
        "状态": "OK 表示该样本两种输入都成功；否则写错误原因。",
    }
    headers = list(comments.keys())
    sheets: Dict[str, List[List[Any]]] = {}
    for horizon_s in sorted(rows_by_horizon):
        sheet_rows = [headers]
        for row in rows_by_horizon[horizon_s]:
            sheet_rows.append([row.get(header, "") for header in headers])
        sheets[f"{horizon_s}s"] = sheet_rows

    sheets["说明"] = [
        ["ADE", "未来每一步 prediction 点到未来 GT 点的 L2 距离平均值。"],
        ["FDE", "最后一个 prediction 点到未来 GT 最后点的 L2 距离。"],
        ["最小距离", "未来每一步 prediction 点到同一时刻未来 GT 点的 L2 距离最小值。"],
        ["历史替换规则", "最近三帧历史都使用对应 clean/attacked tracking；更早历史使用 nuScenes GT；未来 GT 只用于评价。"],
    ]
    _write_minimal_xlsx(path, sheets)


def _external_prediction_command(cfg: Dict[str, Any], adapter: str) -> Dict[str, Any]:
    pcfg = _prediction_cfg(cfg, adapter)
    output_dir = Path(cfg["output"]["dir"]) / f"pred-{adapter}-official"
    python_bin = str(pcfg.get("python") or cfg.get("env", {}).get("prediction_python") or cfg.get("env", {}).get("trajectron_python") or sys.executable)
    if adapter == "lanegcn":
        repo = Path(pcfg.get("repo_root", "/home/jushuo/Code/zz6.5-LaneGCN/LaneGCN")).resolve()
        command = [
            python_bin,
            str(repo / "test.py"),
            "-m",
            str(pcfg.get("model", "lanegcn")),
            "--weight",
            str(pcfg.get("weight", repo / "ckpt" / "36.000.ckpt")),
            "--split",
            str(pcfg.get("split", "val")),
        ]
    elif adapter == "laformer":
        repo = Path(pcfg.get("repo_root", "/home/jushuo/Code/zz6.6-LAformer/LAformer")).resolve()
        command = [
            python_bin,
            str(repo / "src" / "eval.py"),
            "--future_frame_num",
            str(pcfg.get("future_frame_num", 12)),
            "--eval_batch_size",
            str(pcfg.get("eval_batch_size", 128)),
            "--output_dir",
            str(pcfg.get("output_dir", repo / "checkpoints" / "nuScene_k5")),
            "--hidden_size",
            str(pcfg.get("hidden_size", 64)),
            "--train_batch_size",
            str(pcfg.get("train_batch_size", 32)),
            "--lane_loss_weight",
            str(pcfg.get("lane_loss_weight", 0.9)),
            "--topk",
            str(pcfg.get("topk", 2)),
            "--reuse_temp_file",
            "--model_recover_path",
            str(pcfg.get("model_recover_path", repo / "checkpoints" / "nuScene_k5" / "model.50.bin")),
            "--distributed_training",
            str(pcfg.get("distributed_training", 1)),
            "--other_params",
            *[str(v) for v in pcfg.get("other_params", ["step_lane_score", "stage_two", "semantic_lane", "direction", "enhance_global_graph", "subdivide", "new", "laneGCN", "point_level-4-3", "nuscenes", "nuscenes_mode_num=5"])],
            "--do_eval",
        ]
    elif adapter == "hpnet":
        repo = Path(pcfg.get("repo_root", "/home/jushuo/Code/zz6.7-HPNet/HPNet")).resolve()
        dataset = str(pcfg.get("dataset", "argoverse")).lower()
        script = repo / ("HPNet-INTERACTION/test.py" if dataset == "interaction" else "HPNet-Argoverse/test.py")
        command = [
            python_bin,
            str(script),
            "--root",
            str(pcfg.get("data_root", "")),
            "--test_batch_size",
            str(pcfg.get("test_batch_size", 2)),
            "--devices",
            str(pcfg.get("devices", 1)),
            "--ckpt_path",
            str(pcfg.get("ckpt_path", "")),
        ]
    elif adapter == "pgp":
        repo = Path(pcfg.get("repo_root", "/home/jushuo/Code/zz6.8-PGP/PGP")).resolve()
        command = [
            python_bin,
            str(repo / "evaluate.py"),
            "-c",
            str(pcfg.get("config", repo / "configs" / "pgp_gatx2_lvm_traversal.yml")),
            "-r",
            str(pcfg.get("data_root", cfg["nuscenes"]["dataroot"])),
            "-d",
            str(pcfg.get("data_dir", "")),
            "-o",
            str(pcfg.get("output_dir", output_dir)),
            "-w",
            str(pcfg.get("checkpoint", repo / "ckpt" / "PGP_lr-scheduler.tar")),
        ]
    elif adapter == "pip":
        repo = Path(pcfg.get("repo_root", "/home/jushuo/Code/zz6.9-PiP/PiP-Planning-informed-Prediction")).resolve()
        command = [
            python_bin,
            str(repo / "evaluate.py"),
            "--name",
            str(pcfg.get("name", "ngsim_model")),
            "--batch_size",
            str(pcfg.get("batch_size", 64)),
            "--test_set",
            str(pcfg.get("test_set", repo / "datasets" / "NGSIM" / "test.mat")),
        ]
    else:
        raise ValueError(f"未知 prediction adapter: {adapter}")

    compatibility = {
        "lanegcn": {
            "native_input": [
                "Argoverse Forecasting CSV or LaneGCN preprocess .p array",
                "20 history frames and 30 future frames at 10 Hz",
                "Argoverse lane graph fields: ctrs, feats, turn, control, intersect, pre, suc, left, right",
            ],
            "conversion_status": "not_implemented",
            "conversion_blockers": [
                "official modules import argoverse and ArgoverseMap even for the standard test path",
                "zz9 currently has nuScenes global tracks and nuScenesMap, not Argoverse city/lane ids",
                "a correct adapter must synthesize LaneGCN's preprocessed graph dict from nuScenes lanes",
            ],
            "can_be_solved_with_adapter": True,
        },
        "laformer": {
            "native_input": [
                "LAformer nuScenes pickle files produced by src/datascripts/dataloader_nuscenes.py",
                "mapping dict with target history, surrounding agent polylines, lane polylines, labels, origin and rotation",
                "nuScenes prediction challenge instance_sample token list",
            ],
            "conversion_status": "zz9_native_adapter_implemented",
            "conversion_blockers": [
                "official eval.py is only run when prediction.laformer.run_native=true",
                "adapter generates eval.ex_list with zz9 target tracking history; official repo is not modified",
                "native outputs are converted from evalai_submission.json to zz9 pred-traj JSON",
            ],
            "can_be_solved_with_adapter": True,
        },
        "hpnet": {
            "native_input": [
                "Argoverse raw CSV under root/test/data or processed torch_geometric HeteroData .pt",
                "agent fields: visible_mask, position, heading, length, agent_index",
                "lane/centerline heterograph fields and predecessor/successor/adjacent edges",
            ],
            "conversion_status": "not_implemented",
            "conversion_blockers": [
                "checkpoint path is empty until user supplies it",
                "current trajectron++ env is missing torch_geometric/pytorch_lightning",
                "CSV-only conversion would still call ArgoverseMap; a nuScenes adapter must write processed HeteroData directly",
            ],
            "can_be_solved_with_adapter": True,
        },
        "pgp": {
            "native_input": [
                "PGP pickle files in data_dir named instance_sample.pickle",
                "inputs: target_agent_representation, surrounding_agent_representation, map_representation",
                "graph mode fields: lane_node_feats/masks, s_next, edge_type, agent_node_masks, init_node",
            ],
            "conversion_status": "zz9_native_adapter_implemented",
            "conversion_blockers": [
                "official evaluate.py only reports metrics, so zz9 runs the official PGP model modules directly to export predictions",
                "adapter generates NuScenesGraphs-compatible pickles plus stats.pickle from zz9 target tracking history",
                "native outputs are converted to zz9 pred-traj JSON",
            ],
            "can_be_solved_with_adapter": True,
        },
        "pip": {
            "native_input": [
                "NGSIM/highD HDF5 .mat with traj and tracks datasets",
                "highway local X/Y, lane id, lateral/longitudinal maneuver labels, grid neighbor ids",
                "trained_models/<name>/<name>.tar checkpoint layout",
            ],
            "conversion_status": "not_recommended",
            "conversion_blockers": [
                "nuScenes urban scenes do not provide the highway grid/lane/maneuver semantics PiP assumes",
                "synthetic .mat conversion would be a semantic mismatch, not a faithful model adapter",
                "local datasets/trained_models are absent",
            ],
            "can_be_solved_with_adapter": False,
        },
    }[adapter]

    command.extend(str(item) for item in pcfg.get("extra_args", []) or [])
    return {
        "schema": "zz9.prediction_external_command.v1",
        "adapter": adapter,
        "command": command,
        "cwd": str(repo),
        "run_native": bool(pcfg.get("run_native", False)),
        "output_dir": str(output_dir),
        "track_to_native_conversion": False,
        "native_input": compatibility["native_input"],
        "conversion_status": compatibility["conversion_status"],
        "conversion_blockers": compatibility["conversion_blockers"],
        "can_be_solved_with_adapter": compatibility["can_be_solved_with_adapter"],
        "unsupported_reason": "官方入口不能直接消费 zz9 nuScenes tracking.json；需要先写对应 native input adapter。",
    }


def _write_external_prediction_manifest(cfg: Dict[str, Any], adapter: str) -> Path:
    payload = _external_prediction_command(cfg, adapter)
    path = Path(cfg["output"]["dir"]) / f"pred-{adapter}-official" / "command_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _run_external_prediction_adapter(cfg: Dict[str, Any], adapter: str, config_path: Path) -> Path:
    if adapter in {"laformer", "pgp"}:
        pcfg = _prediction_cfg(cfg, adapter)
        python_bin = str(pcfg.get("python") or cfg.get("env", {}).get("prediction_python") or cfg.get("env", {}).get("trajectron_python") or sys.executable)
        command = [
            python_bin,
            EVAL_ROOT / "run_native_prediction_adapter.py",
            "--config",
            config_path,
            "--adapter",
            adapter,
        ]
        subprocess.run([str(item) for item in command], cwd=str(EVAL_ROOT), check=True)
        if not bool(pcfg.get("run_native", False)):
            native_dir = Path(cfg["output"]["dir"]) / f"pred-{adapter}-native"
            raise RuntimeError(
                f"{adapter} native input 已生成: {native_dir}；"
                f"prediction.{adapter}.run_native=false，未运行官方模型，也不会把空 prediction 传给 planner。"
            )
        return Path(cfg["output"]["dir"]) / f"pred-{adapter}-native"
    manifest_path = _write_external_prediction_manifest(cfg, adapter)
    payload = _load_json(manifest_path)
    if payload.get("run_native"):
        subprocess.run(payload["command"], cwd=payload["cwd"], check=True)
        raise RuntimeError(
            f"{adapter} 官方 test/evaluate 已运行，但输出仍是官方格式；"
            "缺少从官方结果到 zz9 pred-traj JSON 的无损转换，不能继续传给 planner。"
        )
    raise RuntimeError(
        f"{adapter} 官方命令已写入 {manifest_path}，默认不运行。{payload['unsupported_reason']}"
    )


def run(config_path: Path) -> Path:
    cfg = _load_yaml(config_path)
    adapter = _prediction_adapter(cfg)
    if adapter in EXTERNAL_PREDICTION_ADAPTERS:
        return _run_external_prediction_adapter(cfg, adapter, config_path)
    if adapter == "trajectronpp":
        _add_paths(cfg)
    elif adapter == "hivt":
        raise RuntimeError(
            "prediction.model_id=2 (HiVT) 当前未实现本地适配。"
            "请先使用 prediction.model_id=1 (Trajectron++)，或补齐 HiVT 适配器后再运行。"
        )
    else:
        raise ValueError(f"prediction.adapter={adapter!r} 暂未接入；可选: {PREDICTION_MODEL_ID_MAP}")
    from nuscenes.nuscenes import NuScenes

    dataroot = Path(cfg["nuscenes"]["dataroot"]).resolve()
    nusc = NuScenes(version=str(cfg["nuscenes"].get("version", "v1.0-trainval")), dataroot=str(dataroot), verbose=False)
    samples = _load_sample_specs(cfg)
    out_dir = Path(cfg["output"]["dir"])
    rows_by_horizon: Dict[int, List[Dict[str, Any]]] = {int(h): [] for h in _prediction_horizons_s(cfg)}
    pred_payloads: List[Dict[str, Any]] = []
    trajectron: Optional[Any] = None

    for sample in samples:
        phase_metrics: Dict[str, Dict[int, Dict[str, Any]]] = {}
        status = "OK"
        for phase in _phases(cfg):
            try:
                trajectron, metrics_by_ph, pred_json = _evaluate_phase(
                    nusc=nusc,
                    dataroot=dataroot,
                    sample=sample,
                    phase=phase,
                    cfg=cfg,
                    trajectron=trajectron,
                )
                phase_metrics[phase] = metrics_by_ph
                pred_payloads.append(pred_json)
                _clear_stage_error(cfg, sample.name, phase, "prediction")
            except Exception as exc:
                status = f"{_phase_label(phase)}失败: {type(exc).__name__}: {exc}"
                _write_stage_error(cfg, sample.name, phase, "prediction", exc)
                phase_metrics[phase] = {}

        for horizon_s in rows_by_horizon:
            for phase in _phases(cfg):
                metrics = phase_metrics.get(phase, {}).get(horizon_s, {})
                rows_by_horizon[horizon_s].append(
                    {
                        "样本": sample.name,
                        "阶段": phase,
                        "当前帧token": sample.sample_tokens[-1],
                        "ADE": metrics.get("ade", ""),
                        "FDE": metrics.get("fde", ""),
                        "最小距离": metrics.get("min_de", ""),
                        "状态": status,
                    }
                )

    _write_prediction_json(cfg, pred_payloads)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
