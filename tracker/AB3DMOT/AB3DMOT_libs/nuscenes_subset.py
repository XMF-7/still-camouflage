import json
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.kitti import KittiDB

from AB3DMOT_libs.kitti_calib import save_calib_file
from AB3DMOT_libs.kitti_obj import Object_3D
from AB3DMOT_libs.kitti_trk import Tracklet_3D
from AB3DMOT_libs.nuScenes2KITTI_helper import kitti_cam2nuScenes_lidar
from AB3DMOT_libs.nuScenes_utils import (
    box_to_trk_sample_result,
    create_nuScenes_box,
    get_sensor_param,
    nuScenes_lidar2world,
    nuScenes_world2lidar,
)
from AB3DMOT_libs.pipeline_utils import clear_path, ensure_dir
from AB3DMOT_libs.utils import get_subfolder_seq


KITTI_TO_NU_LIDAR = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)


class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 兼容不同 numpy 版本在 pickle 中记录的模块路径差异。
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _category_to_tracking_name(category_name):
    tracking_mapping = {
        "vehicle.bicycle": "bicycle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.car": "car",
        "vehicle.motorcycle": "motorcycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "vehicle.trailer": "trailer",
        "vehicle.truck": "truck",
    }
    return tracking_mapping.get(category_name)


def _nuScenes_transform2KITTI(cs_record_lid, cs_record_cam):
    lid_to_ego = transform_matrix(
        cs_record_lid["translation"],
        Quaternion(cs_record_lid["rotation"]),
        inverse=False,
    )
    ego_to_cam = transform_matrix(
        cs_record_cam["translation"],
        Quaternion(cs_record_cam["rotation"]),
        inverse=True,
    )
    lid_to_cam_nuscenes = np.dot(ego_to_cam, lid_to_ego)

    velo_to_cam_kitti = np.dot(
        lid_to_cam_nuscenes, KITTI_TO_NU_LIDAR.transformation_matrix
    )
    velo_to_cam_rot = velo_to_cam_kitti[:3, :3]
    velo_to_cam_trans = velo_to_cam_kitti[:3, 3]

    r0_rect = Quaternion(axis=[1, 0, 0], angle=0)
    p_left_kitti = np.zeros((3, 4))
    p_left_kitti[:3, :3] = cs_record_cam["camera_intrinsic"]
    return velo_to_cam_trans, velo_to_cam_rot, r0_rect, p_left_kitti


def _create_kitti_transform(
    velo_to_cam_trans, velo_to_cam_rot, r0_rect, p_left_kitti
):
    imu_to_velo_kitti = np.zeros((3, 4))
    imu_to_velo_kitti[0, 0] = 1
    imu_to_velo_kitti[1, 1] = 1
    imu_to_velo_kitti[2, 2] = 1

    return OrderedDict(
        {
            "P0": np.zeros((3, 4)),
            "P1": np.zeros((3, 4)),
            "P2": p_left_kitti,
            "P3": np.zeros((3, 4)),
            "R0_rect": r0_rect.rotation_matrix,
            "Tr_velo_to_cam": np.hstack(
                (velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1))
            ),
            "Tr_imu_to_velo": imu_to_velo_kitti,
        }
    )


def _convert_anno_to_kitti(
    nusc,
    anno_token,
    lidar_token,
    instance_token_list,
    velo_to_cam_trans,
    velo_to_cam_rot,
    r0_rect,
    p_left_kitti,
):
    sample_annotation = nusc.get("sample_annotation", anno_token)
    instance_token = sample_annotation["instance_token"]
    if instance_token in instance_token_list:
        track_id = instance_token_list.index(instance_token)
    else:
        track_id = len(instance_token_list)
        instance_token_list.append(instance_token)

    _, box_lidar_nusc, _ = nusc.get_sample_data(
        lidar_token,
        box_vis_level=BoxVisibility.NONE,
        selected_anntokens=[anno_token],
    )
    box_lidar_nusc = box_lidar_nusc[0]

    obj_name = _category_to_tracking_name(sample_annotation["category_name"])
    if obj_name is None:
        return None, -1
    obj_name = obj_name.capitalize()

    box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
        box_lidar_nusc, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect
    )
    box_cam_kitti.score = 1.0
    bbox_2d = KittiDB.project_kitti_box_to_image(
        box_cam_kitti, p_left_kitti, imsize=(1600, 900)
    )
    if bbox_2d is None:
        bbox_2d = (-1, -1, -1, -1)

    output = KittiDB.box_to_string(
        name=obj_name,
        box=box_cam_kitti,
        bbox_2d=bbox_2d,
        truncation=0.0,
        occlusion=0,
    )
    return output, track_id


def has_info_subset(cfg):
    return bool(getattr(cfg.paths, "info_pkl", "")) or bool(
        getattr(cfg.prepare, "use_json_subset", False)
    )


def get_subset_source(cfg):
    source = str(getattr(cfg.prepare, "subset_source", "auto")).lower()
    if source not in {"auto", "pkl", "json"}:
        raise ValueError(
            f"prepare.subset_source 只能是 auto / pkl / json，当前为 {source}。"
        )
    return source


def _load_info_payload(info_pkl_path):
    with open(info_pkl_path, "rb") as file_obj:
        try:
            payload = pickle.load(file_obj)
        except ModuleNotFoundError as exc:
            # 常见于 numpy 版本不一致导致的 `numpy._core` / `numpy.core` 兼容问题。
            if "numpy._core" not in str(exc):
                raise
            file_obj.seek(0)
            payload = _NumpyCompatUnpickler(file_obj).load()

    if isinstance(payload, dict):
        infos = payload.get("infos")
        if infos is None:
            raise ValueError("info pkl 缺少 infos 字段。")
    elif isinstance(payload, list):
        infos = payload
        payload = {"infos": infos}
    else:
        raise ValueError("info pkl 格式不支持，应为 list 或 {'infos': ...}。")

    if not infos:
        raise ValueError("info pkl 里没有任何样本。")

    return payload, infos


def _load_infos_from_bevformer_json(bevformer_json_path):
    with open(bevformer_json_path, "r") as file_obj:
        raw_data = json.load(file_obj)

    if isinstance(raw_data, dict) and "results" in raw_data:
        results = raw_data["results"]
    elif isinstance(raw_data, dict):
        results = raw_data
    else:
        raise ValueError("BEVFormer json 格式不支持，应为 dict 或 {'results': ...}。")

    sample_tokens = list(results.keys())
    if not sample_tokens:
        raise ValueError("BEVFormer json 中没有任何 sample token。")

    payload = {
        "infos": [{"token": sample_token} for sample_token in sample_tokens],
        "metadata": {"source": "bevformer_json"},
    }
    return payload, payload["infos"]


def _resolve_existing_raw_path(path_value, data_root, fallback_rel=None):
    candidate_paths = []

    if path_value:
        path = Path(path_value)
        if path.is_absolute():
            candidate_paths.append(path)
        else:
            candidate_paths.append((Path(data_root) / path).resolve())

    if fallback_rel:
        candidate_paths.append((Path(data_root) / fallback_rel).resolve())

    expanded_candidates = []
    for path in candidate_paths:
        expanded_candidates.append(path)
        path_str = str(path)
        if "/samples/" in path_str:
            expanded_candidates.append(Path(path_str.replace("/samples/", "/sample/")))
        if "/sample/" in path_str:
            expanded_candidates.append(Path(path_str.replace("/sample/", "/samples/")))

    for path in expanded_candidates:
        if path.is_file():
            return path

    basename = Path(fallback_rel or path_value).name
    search_roots = [Path(data_root), Path(data_root).parent]
    for root in search_roots:
        if not root.exists():
            continue
        for match in root.rglob(basename):
            if match.is_file():
                return match

    raise FileNotFoundError("找不到原始数据文件：%s" % (fallback_rel or path_value))


def _get_result_token(info):
    result_token = info.get("_subset_result_token")
    if result_token:
        return result_token

    result_token = info.get("token") or info.get("sample_token")
    if not result_token:
        raise ValueError("info pkl 中存在缺少 token 的样本。")

    info["_subset_result_token"] = result_token
    return result_token


def _token_in_table(nusc, table_name, token):
    return bool(token) and token in nusc._token2ind.get(table_name, {})


def _resolve_sample_token(nusc, info, result_token=None):
    cached = info.get("_subset_sample_token")
    if cached:
        return cached

    result_token = result_token or _get_result_token(info)
    direct_candidates = [
        info.get("reference_sample_token"),
        info.get("sample_token"),
        result_token,
    ]
    for candidate in direct_candidates:
        if _token_in_table(nusc, "sample", candidate):
            info["_subset_sample_token"] = candidate
            return candidate

    sample_data_candidates = [
        result_token,
        info.get("reference_sample_data_token"),
    ]
    for candidate in sample_data_candidates:
        if _token_in_table(nusc, "sample_data", candidate):
            sample_token = nusc.get("sample_data", candidate)["sample_token"]
            info["_subset_sample_token"] = sample_token
            return sample_token

    raise KeyError(
        "无法从 subset info 中解析 sample token，"
        "请检查 token / sample_token / reference_sample_token / "
        "reference_sample_data_token 是否齐全。"
    )


def _build_sensor_records_from_info(info, cam_name):
    cam_info = (info.get("cams") or {}).get(cam_name)
    if not cam_info:
        return None

    required_info_keys = [
        "ego2global_translation",
        "ego2global_rotation",
        "lidar2ego_translation",
        "lidar2ego_rotation",
        "lidar_path",
    ]
    required_cam_keys = [
        "sensor2ego_translation",
        "sensor2ego_rotation",
        "cam_intrinsic",
        "data_path",
    ]
    if any(info.get(key) is None for key in required_info_keys):
        return None
    if any(cam_info.get(key) is None for key in required_cam_keys):
        return None

    pose_record = {
        "translation": list(info["ego2global_translation"]),
        "rotation": list(info["ego2global_rotation"]),
    }
    cs_record_lid = {
        "translation": list(info["lidar2ego_translation"]),
        "rotation": list(info["lidar2ego_rotation"]),
    }
    cs_record_cam = {
        "translation": list(cam_info["sensor2ego_translation"]),
        "rotation": list(cam_info["sensor2ego_rotation"]),
        "camera_intrinsic": np.asarray(cam_info["cam_intrinsic"]),
    }
    return (
        pose_record,
        cs_record_lid,
        cs_record_cam,
        info["lidar_path"],
        cam_info["data_path"],
    )


def _get_frame_sensor_params(nusc, info, cam_name="CAM_FRONT", output_file=False):
    cached = _build_sensor_records_from_info(info, cam_name)
    if cached is not None:
        if output_file:
            return cached
        return cached[:3]

    sample_token = _resolve_sample_token(nusc, info)
    return get_sensor_param(
        nusc,
        sample_token,
        cam_name=cam_name,
        output_file=output_file,
    )


def _get_reference_sample(nusc, info):
    return nusc.get("sample", _resolve_sample_token(nusc, info))


def build_subset_context(cfg):
    if not has_info_subset(cfg):
        return None

    cached = getattr(cfg, "subset_context", None)
    if cached is not None:
        return cached

    subset_source = get_subset_source(cfg)
    info_pkl_path = getattr(cfg.paths, "info_pkl", "")

    if subset_source == "pkl":
        if not info_pkl_path:
            raise ValueError(
                "prepare.subset_source=pkl，但没有提供 paths.samples_info_pkl。"
            )
        payload, infos = _load_info_payload(info_pkl_path)
    elif subset_source == "json":
        payload, infos = _load_infos_from_bevformer_json(cfg.paths.bevformer_json)
    elif info_pkl_path:
        payload, infos = _load_info_payload(info_pkl_path)
    else:
        payload, infos = _load_infos_from_bevformer_json(cfg.paths.bevformer_json)
    version = payload.get("metadata", {}).get("version")
    if not version:
        if cfg.split in {"train", "val", "trainval"}:
            version = "v1.0-trainval"
        else:
            version = "v1.0-test"

    nusc = NuScenes(version=version, dataroot=cfg.paths.nuscenes_data_root, verbose=True)

    scene_to_pairs = OrderedDict()

    for info in infos:
        result_token = _get_result_token(info)
        sample_token = _resolve_sample_token(nusc, info, result_token)
        sample = nusc.get("sample", sample_token)
        scene = nusc.get("scene", sample["scene_token"])
        scene_name = scene["name"]
        timestamp = int(info.get("timestamp") or sample["timestamp"])
        scene_to_pairs.setdefault(scene_name, []).append((timestamp, info))

    scene_to_infos = OrderedDict()
    scene_frame_map = {}
    infos = []
    for scene_name, pairs in scene_to_pairs.items():
        ordered_infos = [info for _, info in sorted(pairs, key=lambda item: item[0])]
        scene_to_infos[scene_name] = ordered_infos
        infos.extend(ordered_infos)
        for frame_index, info in enumerate(ordered_infos):
            result_token = _get_result_token(info)
            scene_frame_map[result_token] = (scene_name, frame_index)

    global_frame_map = OrderedDict(
        (
            _get_result_token(info),
            index,
        )
        for index, info in enumerate(infos)
    )
    info_by_result_token = OrderedDict(
        (_get_result_token(info), info) for info in infos
    )

    context = {
        "payload": payload,
        "infos": infos,
        "info_by_result_token": info_by_result_token,
        "nusc": nusc,
        "scene_to_infos": scene_to_infos,
        "scene_names": list(scene_to_infos.keys()),
        "scene_frame_map": scene_frame_map,
        "global_frame_map": global_frame_map,
    }
    cfg.subset_context = context
    return context


def _copy_camera_image(src_path, dst_path):
    dst_path = Path(dst_path)
    ensure_dir(dst_path.parent)
    image = Image.open(str(src_path))
    image.save(str(dst_path), "PNG")


def _copy_lidar_to_kitti(src_path, dst_path):
    dst_path = Path(dst_path)
    ensure_dir(dst_path.parent)
    pcl = LidarPointCloud.from_file(str(src_path))
    pcl.rotate(KITTI_TO_NU_LIDAR.inverse.rotation_matrix)
    with open(str(dst_path), "wb") as file_obj:
        pcl.points.T.tofile(file_obj)


def prepare_subset_tracking_data(cfg, context):
    nusc = context["nusc"]
    tracking_root = Path(cfg.paths.nusc_kitti_root) / "tracking"
    split_root = tracking_root / cfg.split

    produced_split_file = tracking_root / "produced" / "split" / ("%s.txt" % cfg.split)
    evaluate_file = tracking_root / ("evaluate_tracking.seqmap.%s" % cfg.split)
    produced_corr_root = tracking_root / "produced" / "correspondence" / cfg.split
    clear_path(produced_split_file)
    clear_path(evaluate_file)
    clear_path(produced_corr_root)
    ensure_dir(produced_split_file.parent)
    ensure_dir(evaluate_file.parent)
    ensure_dir(produced_corr_root)

    with open(str(produced_split_file), "w") as split_file, open(
        str(evaluate_file), "w"
    ) as eval_file:
        for scene_name, scene_infos in context["scene_to_infos"].items():
            image_dir = split_root / "image_02" / scene_name
            lidar_dir = split_root / "velodyne" / scene_name
            label_obj_dir = split_root / "label_2_object" / scene_name
            calib_file = split_root / "calib" / ("%s.txt" % scene_name)
            label_file = split_root / "label_02" / ("%s.txt" % scene_name)
            oxts_file = split_root / "oxts" / ("%s.json" % scene_name)
            corr_file = split_root / "correspondence" / ("%s.txt" % scene_name)
            produced_corr_file = (
                tracking_root
                / "produced"
                / "correspondence"
                / cfg.split
                / ("%s.txt" % scene_name)
            )

            for path in [
                image_dir,
                lidar_dir,
                label_obj_dir,
                calib_file,
                label_file,
                oxts_file,
                corr_file,
                produced_corr_file,
            ]:
                clear_path(path)

            ensure_dir(image_dir)
            ensure_dir(lidar_dir)
            ensure_dir(label_obj_dir)
            ensure_dir(calib_file.parent)
            ensure_dir(label_file.parent)
            ensure_dir(oxts_file.parent)
            ensure_dir(corr_file.parent)
            ensure_dir(produced_corr_file.parent)

            split_file.write("%s\n" % scene_name)

            instance_token_list = []
            ego_pose_list = []

            with open(str(corr_file), "w") as corr_obj, open(
                str(produced_corr_file), "w"
            ) as produced_corr_obj, open(str(label_file), "w") as label_obj:
                for frame_index, info in enumerate(scene_infos):
                    result_token = _get_result_token(info)
                    sample = _get_reference_sample(nusc, info)
                    (
                        pose_record,
                        cs_record_lid,
                        cs_record_cam,
                        filename_lid_full,
                        filename_cam_full,
                    ) = _get_frame_sensor_params(
                        nusc, info, cam_name="CAM_FRONT", output_file=True
                    )

                    corr_line = "%06d %s\n" % (frame_index, result_token)
                    corr_obj.write(corr_line)
                    produced_corr_obj.write(corr_line)

                    velo_to_cam_trans, velo_to_cam_rot, r0_rect, p_left_kitti = (
                        _nuScenes_transform2KITTI(cs_record_lid, cs_record_cam)
                    )
                    kitti_transforms = _create_kitti_transform(
                        velo_to_cam_trans, velo_to_cam_rot, r0_rect, p_left_kitti
                    )
                    save_calib_file(kitti_transforms, str(calib_file))

                    cam_info = info.get("cams", {}).get("CAM_FRONT", {})
                    image_src = cam_info.get("data_path", filename_cam_full)
                    lidar_src = info.get("lidar_path", filename_lid_full)
                    image_src = _resolve_existing_raw_path(
                        image_src, cfg.paths.nuscenes_data_root, filename_cam_full
                    )
                    lidar_src = _resolve_existing_raw_path(
                        lidar_src, cfg.paths.nuscenes_data_root, filename_lid_full
                    )

                    _copy_camera_image(
                        image_src, image_dir / ("%06d.png" % frame_index)
                    )
                    _copy_lidar_to_kitti(
                        lidar_src, lidar_dir / ("%06d.bin" % frame_index)
                    )

                    ego_pose = transform_matrix(
                        pose_record["translation"],
                        Quaternion(pose_record["rotation"]),
                        inverse=False,
                    )
                    ego_pose_list.append(ego_pose)

                    label_obj_path = label_obj_dir / ("%06d.txt" % frame_index)
                    with open(str(label_obj_path), "w") as frame_label_obj:
                        for anno_token in sample["anns"]:
                            output, track_id = _convert_anno_to_kitti(
                                nusc,
                                anno_token,
                                sample["data"]["LIDAR_TOP"],
                                instance_token_list,
                                velo_to_cam_trans,
                                velo_to_cam_rot,
                                r0_rect,
                                p_left_kitti,
                            )
                            if output is None:
                                continue
                            frame_label_obj.write(output + "\n")
                            label_obj.write("%d %d %s\n" % (frame_index, track_id, output))

            with open(str(oxts_file), "w") as file_obj:
                json.dump(np.stack(ego_pose_list, axis=0).tolist(), file_obj)

            eval_file.write(
                "%s empty 000000 %06d\n" % (scene_name, len(scene_infos))
            )


def prepare_subset_object_correspondence(cfg, context):
    object_root = Path(cfg.paths.nusc_kitti_root) / "object"
    source_corr = object_root / cfg.split / "correspondence.txt"
    produced_corr = object_root / "produced" / "correspondence" / ("%s.txt" % cfg.split)

    clear_path(source_corr)
    clear_path(produced_corr)
    ensure_dir(source_corr.parent)
    ensure_dir(produced_corr.parent)

    with open(str(source_corr), "w") as source_obj, open(
        str(produced_corr), "w"
    ) as produced_obj:
        for global_index, info in enumerate(context["infos"]):
            result_token = _get_result_token(info)
            line = "%06d %s\n" % (global_index, result_token)
            source_obj.write(line)
            produced_obj.write(line)


def prepare_subset_detection_inputs(cfg, context, staged_json_path):
    with open(str(staged_json_path), "r") as file_obj:
        raw_data = json.load(file_obj)
    results = raw_data.get("results", {})

    _, det_id2str, _, _, _ = get_subfolder_seq(cfg.dataset, cfg.split)
    det_str2id = {name: idx for idx, name in det_id2str.items()}

    nusc = context["nusc"]
    detection_root = Path(getattr(cfg.paths, "detection_root", "")) if getattr(cfg.paths, "detection_root", "") else Path(cfg.repo_root) / "data" / cfg.dataset / "detection"
    object_results_dir = (
        Path(cfg.paths.nusc_kitti_root)
        / "object"
        / "produced"
        / "results"
        / cfg.split
        / cfg.det_name
        / "data"
    )
    clear_path(object_results_dir)
    ensure_dir(object_results_dir)

    scene_file_handles = {}
    try:
        for scene_name in context["scene_names"]:
            scene_file_handles[scene_name] = {}
            for category_name in list(det_id2str.values()) + ["all"]:
                file_path = (
                    detection_root
                    / ("%s_%s_%s" % (cfg.det_name, category_name, cfg.split))
                    / ("%s.txt" % scene_name)
                )
                clear_path(file_path)
                ensure_dir(file_path.parent)
                scene_file_handles[scene_name][category_name] = open(str(file_path), "w")

        for global_index, info in enumerate(context["infos"]):
            result_token = _get_result_token(info)
            scene_name, frame_index = context["scene_frame_map"][result_token]
            pose_record, cs_record_lid, cs_record_cam = _get_frame_sensor_params(
                nusc, info, cam_name="CAM_FRONT"
            )
            velo_to_cam_trans, velo_to_cam_rot, r0_rect, p_left_kitti = (
                _nuScenes_transform2KITTI(cs_record_lid, cs_record_cam)
            )

            object_result_path = object_results_dir / ("%06d.txt" % global_index)
            with open(str(object_result_path), "w") as object_result_file:
                for det in results.get(result_token, []):
                    detection_name = det.get("detection_name")
                    detection_score = det.get("detection_score")
                    if detection_name is None or detection_score is None:
                        continue

                    obj_name = str(detection_name).capitalize()
                    box = Box(
                        det["translation"],
                        det["size"],
                        Quaternion(det["rotation"]),
                        name=obj_name,
                        token=result_token,
                    )
                    box = nuScenes_world2lidar(box, cs_record_lid, pose_record)
                    box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
                        box,
                        Quaternion(matrix=velo_to_cam_rot),
                        velo_to_cam_trans,
                        r0_rect,
                    )
                    bbox_2d = KittiDB.project_kitti_box_to_image(
                        box_cam_kitti, p_left_kitti, imsize=(1600, 900)
                    )
                    if bbox_2d is None:
                        bbox_2d = (-1, -1, -1, -1)
                    box_cam_kitti.score = float(detection_score)
                    result_line = KittiDB.box_to_string(
                        name=obj_name,
                        box=box_cam_kitti,
                        bbox_2d=bbox_2d,
                        truncation=0.0,
                        occlusion=0,
                    )
                    object_result_file.write(result_line + "\n")

                    if obj_name not in det_str2id:
                        continue
                    trk_obj = Object_3D(result_line)
                    trk_line = trk_obj.convert_to_trk_input_str(
                        frame_index, det_str2id[obj_name]
                    )
                    scene_file_handles[scene_name][obj_name].write(trk_line + "\n")
                    scene_file_handles[scene_name]["all"].write(trk_line + "\n")
    finally:
        for scene_handles in scene_file_handles.values():
            for file_obj in scene_handles.values():
                file_obj.close()


def export_subset_tracking_json(cfg, context, meta=None):
    if meta is None:
        meta = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

    tmp_root_dir = Path(cfg.paths.nusc_kitti_root) / "tracking" / cfg.split
    results_dir = Path(cfg.save_root) / cfg.run_name / "data_0"
    corres_dir = Path(cfg.paths.nusc_kitti_root) / "tracking" / "produced" / "correspondence" / cfg.split
    calib_dir = tmp_root_dir / "calib"
    nusc = context["nusc"]

    results = {}
    for info in context["infos"]:
        results[_get_result_token(info)] = []

    for corres_file in sorted(corres_dir.glob("*.txt")):
        seq_name = corres_file.stem
        calib = calib_dir / ("%s.txt" % seq_name)
        result_file = results_dir / ("%s.txt" % seq_name)
        if not calib.is_file() or not result_file.is_file():
            continue

        corr_dict = OrderedDict()
        with open(str(corres_file), "r") as file_obj:
            for raw_line in file_obj:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                frame_index, result_token = raw_line.split(" ", 1)
                corr_dict[frame_index] = result_token

        from AB3DMOT_libs.kitti_calib import Calibration

        calib_obj = Calibration(str(calib))
        tracklet_data = Tracklet_3D(str(result_file)).data

        for frame_index, frame_data in tracklet_data.items():
            result_token = corr_dict.get("%06d" % frame_index)
            if result_token is None:
                continue
            info = context["info_by_result_token"].get(result_token)
            if info is None:
                continue

            pose_record, cs_record_lid, _ = _get_frame_sensor_params(
                nusc, info, cam_name="CAM_FRONT"
            )
            for track_id, obj in frame_data.items():
                box = create_nuScenes_box(obj)
                box = kitti_cam2nuScenes_lidar(box, calib_obj)
                box = nuScenes_lidar2world(box, cs_record_lid, pose_record)
                sample_result = box_to_trk_sample_result(
                    result_token,
                    box,
                    trk_id=track_id,
                )
                results[result_token].append(sample_result)

    submission = {"meta": meta, "results": results}
    submission_path = Path(cfg.save_root) / cfg.run_name / ("results_%s.json" % cfg.split)
    ensure_dir(submission_path.parent)
    with open(str(submission_path), "w") as file_obj:
        json.dump(submission, file_obj, indent=2)
    return submission_path
