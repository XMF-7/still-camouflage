import copy
import os
import mmcv
from nuscenes.nuscenes import NuScenes
from PIL import Image
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.utils.data_classes import Box
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.render import visualize_sample
from nuscenes.eval.detection.utils import category_to_detection_name
import gc


cams = [
  'CAM_FRONT',
  'CAM_FRONT_RIGHT',
  'CAM_BACK_RIGHT',
  'CAM_BACK',
  'CAM_BACK_LEFT',
  'CAM_FRONT_LEFT'
]

IMAGE_DATAROOT = None
IMAGE_SOURCE_SUBDIR = ''


def _resolve_existing_data_path(data_path: str) -> str:
    normalized = os.path.normpath(data_path)
    basename = os.path.basename(normalized)
    parts = normalized.split(os.sep)
    if 'samples' in parts:
        idx = parts.index('samples')
        suffix_parts = parts[idx + 1:]
        root_parts = parts[:idx]
        default_root_dir = os.sep.join(root_parts) if root_parts else os.sep
        root_dir = IMAGE_DATAROOT if IMAGE_DATAROOT else default_root_dir
        candidate_dirs = []
        if IMAGE_SOURCE_SUBDIR:
            candidate_dirs.append(IMAGE_SOURCE_SUBDIR)
        candidate_dirs.extend(['samples-1', 'samples-2', 'samples'])
        for candidate_dir in candidate_dirs:
            candidate = os.path.join(root_dir, candidate_dir, *suffix_parts)
            if os.path.exists(candidate):
                return candidate
        for candidate_dir in candidate_dirs:
            samples_root = os.path.join(root_dir, candidate_dir)
            if os.path.isdir(samples_root):
                for dirpath, _, filenames in os.walk(samples_root):
                    if basename in filenames:
                        return os.path.join(dirpath, basename)
    if os.path.exists(data_path):
        return data_path
    return data_path


def _resolve_relative_filename(filename: str) -> str:
    candidate = _resolve_existing_data_path(os.path.join(IMAGE_DATAROOT or nusc.dataroot, filename))
    root_dir = IMAGE_DATAROOT or nusc.dataroot
    if os.path.exists(candidate):
        return os.path.relpath(candidate, root_dir)
    return filename


def _pred_boxes_for_sample(sample_token: str, pred_data=None, score_thr: float = 0.2):
    if pred_data is None:
        return []
    boxes = []
    for record in pred_data['results'].get(sample_token, []):
        score = float(record.get('detection_score', 0.0))
        if score <= score_thr:
            continue
        box = Box(
            record['translation'],
            record['size'],
            Quaternion(record['rotation']),
            name=record['detection_name'],
            token='predicted',
        )
        box.detection_score = score
        boxes.append(box)
    return boxes


def _draw_box_score(ax, box, camera_intrinsic, image_size):
    score = getattr(box, 'detection_score', None)
    if score is None:
        return
    corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
    width, height = image_size
    finite = np.isfinite(corners).all(axis=0)
    if not np.any(finite):
        return
    xs = np.clip(corners[0, finite], 0, width - 1)
    ys = np.clip(corners[1, finite], 0, height - 1)
    x = float(np.min(xs))
    y = float(np.min(ys))
    ax.text(
        x,
        y,
        f'{float(score):.3f}',
        color='white',
        fontsize=7,
        ha='left',
        va='top',
        bbox=dict(facecolor='black', alpha=0.65, edgecolor='none', pad=1.2),
        zorder=20,
    )


def _bev_limits(boxes, margin: float = 10.0, min_radius: float = 20.0):
    points = [np.array([[0.0, 0.0]], dtype=np.float32)]
    for box in boxes:
        corners = box.corners()[:2, :].T
        points.append(corners.astype(np.float32))
    stacked = np.concatenate(points, axis=0)
    min_xy = stacked.min(axis=0)
    max_xy = stacked.max(axis=0)
    center_xy = 0.5 * (min_xy + max_xy)
    radius = 0.5 * float(np.max(max_xy - min_xy)) + margin
    radius = max(radius, min_radius)
    return (
        center_xy[0] - radius,
        center_xy[0] + radius,
        center_xy[1] - radius,
        center_xy[1] + radius,
    )


def _ego_bev_view():
    # Map ego (x forward, y left) to plot coordinates:
    # horizontal axis stores y, vertical axis stores x.
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


def _gt_boxes_for_sample(sample_token: str):
    sample = nusc.get('sample', sample_token)
    boxes = []
    for ann_token in sample['anns']:
        box = copy.deepcopy(nusc.get_box(ann_token))
        ann_record = nusc.get('sample_annotation', ann_token)
        box.name = ann_record['category_name']
        boxes.append(box)
    return boxes


def _safe_velocity_2d(sample_annotation_token: str):
    velocity = nusc.box_velocity(sample_annotation_token)[:2]
    velocity = np.asarray(velocity, dtype=np.float32)
    if np.any(np.isnan(velocity)):
        return (0.0, 0.0)
    return (float(velocity[0]), float(velocity[1]))


def _gt_eval_boxes_for_sample(sample_token: str) -> EvalBoxes:
    sample = nusc.get('sample', sample_token)
    eval_boxes = EvalBoxes()
    sample_boxes = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        detection_name = category_to_detection_name(ann['category_name'])
        if detection_name is None:
            continue
        sample_boxes.append(
            DetectionBox(
                sample_token=sample_token,
                translation=tuple(ann['translation']),
                size=tuple(ann['size']),
                rotation=tuple(ann['rotation']),
                velocity=_safe_velocity_2d(ann_token),
                num_pts=int(ann['num_lidar_pts'] + ann['num_radar_pts']),
                detection_name=detection_name,
                detection_score=-1.0,
                attribute_name='',
            )
        )
    eval_boxes.add_boxes(sample_token, sample_boxes)
    return eval_boxes


def _pred_eval_boxes_for_sample(sample_token: str, pred_data=None) -> EvalBoxes:
    eval_boxes = EvalBoxes()
    sample_records = pred_data['results'].get(sample_token, []) if pred_data is not None else []
    sample_boxes = []
    for record in sample_records:
        velocity = tuple(list(record.get('velocity', [0.0, 0.0]))[:2])
        if len(velocity) < 2:
            velocity = (0.0, 0.0)
        sample_boxes.append(
            DetectionBox(
                sample_token=sample_token,
                translation=tuple(record['translation']),
                size=tuple(record['size']),
                rotation=tuple(record['rotation']),
                velocity=(float(velocity[0]), float(velocity[1])),
                detection_name=record['detection_name'],
                detection_score=float(record.get('detection_score', 0.0)),
                attribute_name=str(record.get('attribute_name', '') or ''),
            )
        )
    eval_boxes.add_boxes(sample_token, sample_boxes)
    return eval_boxes


def _boxes_global_to_ego(sample_token: str, boxes_global):
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_sd = nusc.get('sample_data', lidar_token)
    pose_record = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    ego_rotation_inv = Quaternion(pose_record['rotation']).inverse
    ego_translation = np.array(pose_record['translation'])

    boxes_ego = []
    for box in boxes_global:
        ego_box = copy.deepcopy(box)
        ego_box.translate(-ego_translation)
        ego_box.rotate(ego_rotation_inv)
        boxes_ego.append(ego_box)
    return boxes_ego


def render_annotation(anntoken: str, margin: float = 10) -> None:
    ann_record = nusc.get('sample_annotation', anntoken)
    sample_record = nusc.get('sample', ann_record['sample_token'])
    assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'

    cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
    all_bboxes, select_cams = [], []
    for cam in cams:
        _, boxes, _ = nusc.get_sample_data(sample_record['data'][cam], box_vis_level=BoxVisibility.ANY,
                                           selected_anntokens=[anntoken])
        if len(boxes) > 0:
            all_bboxes.append(boxes)
            select_cams.append(cam)

    num_cam = len(all_bboxes)
    fig, axes = plt.subplots(1, num_cam + 1, figsize=(18, 9))
    plt.subplots_adjust(wspace=0.08, hspace=0.12)
    select_cams = [sample_record['data'][cam] for cam in select_cams]

    lidar = sample_record['data']['LIDAR_TOP']
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(lidar, selected_anntokens=[anntoken])

    for box in boxes:
        c = np.array(get_color(box.name)) / 255.0
        box.render(axes[0], colors=(c, c, c))
        corners = view_points(boxes[0].corners(), np.eye(4), False)[:2, :]
        axes[0].set_xlim([np.min(corners[0, :]) - margin, np.max(corners[0, :]) + margin])
        axes[0].set_ylim([np.min(corners[1, :]) - margin, np.max(corners[1, :]) + margin])
        axes[0].axis('off')
        axes[0].set_aspect('equal')

    for i in range(1, num_cam + 1):
        cam = select_cams[i - 1]
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam, selected_anntokens=[anntoken])
        im = Image.open(data_path)
        axes[i].imshow(im)
        axes[i].set_title(nusc.get('sample_data', cam)['channel'])
        axes[i].axis('off')
        axes[i].set_aspect('equal')
        for box in boxes:
            c = np.array(get_color(box.name)) / 255.0
            box.render(axes[i], view=camera_intrinsic, normalize=True, colors=(c, c, c))
        axes[i].set_xlim(0, im.size[0])
        axes[i].set_ylim(im.size[1], 0)

    plt.close()
    gc.collect()


def get_color(category_name: str):
    if category_name == 'bicycle':
        return nusc.colormap['vehicle.bicycle']
    elif category_name == 'construction_vehicle':
        return nusc.colormap['vehicle.construction']
    elif category_name == 'traffic_cone':
        return nusc.colormap['movable_object.trafficcone']
    for key in nusc.colormap.keys():
        if category_name in key:
            return nusc.colormap[key]
    return [0, 0, 0]


def get_predicted_data(sample_data_token: str, pred_anns=None):
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    data_path = nusc.get_sample_data_path(sample_data_token)
    data_path = _resolve_existing_data_path(data_path)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    boxes = pred_anns
    box_list = []
    for box in boxes:
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)
        if sensor_record['modality'] == 'camera' and not box_in_image(box, cam_intrinsic, imsize):
            continue
        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def render_sample_data(
    sample_token: str,
    pred_data=None,
    out_path: str = None,
    verbose: bool = True,
    score_thr: float = 0.2,
) -> None:
    sample = nusc.get('sample', sample_token)
    cams = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ]
    _, ax = plt.subplots(4, 3, figsize=(24, 18))
    plt.subplots_adjust(wspace=0.07, hspace=0.12)  # wider vertical margin, same horizontal
    j = 0
    for ind, cam in enumerate(cams):
        sample_data_token = sample['data'][cam]
        sd_record = nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        if sensor_modality == 'camera':
            boxes = _pred_boxes_for_sample(sample_token, pred_data=pred_data, score_thr=score_thr)

            data_path, boxes_pred, camera_intrinsic = get_predicted_data(sample_data_token, pred_anns=boxes)
            _, boxes_gt, _ = nusc.get_sample_data(sample_data_token)
            if ind == 3:
                j += 1
            ind = ind % 3
            data = Image.open(data_path)

            ax[j, ind].imshow(data)
            ax[j + 2, ind].imshow(data)
            for box in boxes_pred:
                c = np.array(get_color(box.name)) / 255.0
                box.render(ax[j, ind], view=camera_intrinsic, normalize=True, colors=(c, c, c))
                _draw_box_score(ax[j, ind], box, camera_intrinsic, data.size)
            for box in boxes_gt:
                c = np.array(get_color(box.name)) / 255.0
                box.render(ax[j + 2, ind], view=camera_intrinsic, normalize=True, colors=(c, c, c))

            ax[j, ind].set_xlim(0, data.size[0])
            ax[j, ind].set_ylim(data.size[1], 0)
            ax[j + 2, ind].set_xlim(0, data.size[0])
            ax[j + 2, ind].set_ylim(data.size[1], 0)
            ax[j, ind].axis('off')
            ax[j, ind].set_title(f'PRED: {sd_record["channel"]}')
            ax[j + 2, ind].axis('off')
            ax[j + 2, ind].set_title(f'GT: {sd_record["channel"]}')
        else:
            raise ValueError("Error: Unknown sensor modality!")

    if out_path is not None:
        plt.savefig(out_path + '_camera', bbox_inches='tight', pad_inches=0.2, dpi=200, facecolor='white')
    if verbose:
        plt.show()

    plt.close()
    gc.collect()


def render_sample_bev(
    sample_token: str,
    pred_data=None,
    out_path: str = None,
    verbose: bool = True,
    score_thr: float = 0.2,
) -> None:
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_record = nusc.get('sample_data', lidar_token)
    lidar_record['filename'] = _resolve_relative_filename(lidar_record['filename'])

    gt_eval_boxes = _gt_eval_boxes_for_sample(sample_token)
    pred_eval_boxes = _pred_eval_boxes_for_sample(sample_token, pred_data=pred_data)
    savepath = None if out_path is None else out_path + '_bev'
    original_dataroot = nusc.dataroot
    if IMAGE_DATAROOT:
        nusc.dataroot = IMAGE_DATAROOT
    try:
        visualize_sample(
            nusc,
            sample_token,
            gt_eval_boxes,
            pred_eval_boxes,
            nsweeps=1,
            conf_th=float(score_thr),
            eval_range=50,
            verbose=verbose,
            display_legend=False,
            savepath=savepath,
        )
    finally:
        nusc.dataroot = original_dataroot
    gc.collect()


if __name__ == '__main__':
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description="Render NuScenes sample data with predicted results.")
    parser.add_argument('--results_path', type=str, help='Path to the results_nusc.json file.')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='NuScenes dataset version.')
    parser.add_argument('--dataroot', type=str, default='./data/nuscenes', help='NuScenes dataset root.')
    parser.add_argument('--image-dataroot', type=str, default='', help='Image root for camera rendering.')
    parser.add_argument('--image-source-subdir', type=str, default='', help='Image subdir such as samples-1 / samples-2.')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers.')
    parser.add_argument('--score-thr', type=float, default=0.2, help='Score threshold for rendering predicted boxes.')
    parser.add_argument('--bev-only', action='store_true', help='Only render official BEV figures.')
    args = parser.parse_args()

    IMAGE_DATAROOT = args.image_dataroot or args.dataroot
    IMAGE_SOURCE_SUBDIR = args.image_source_subdir

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    _original_get_sample_data_path = nusc.get_sample_data_path

    def _patched_get_sample_data_path(sample_data_token: str) -> str:
        return _resolve_existing_data_path(_original_get_sample_data_path(sample_data_token))

    nusc.get_sample_data_path = _patched_get_sample_data_path
    bevformer_results = mmcv.load(args.results_path)
    sample_token_list = list(bevformer_results['results'].keys())

    scene_to_samples = {}
    for scene in nusc.scene:
        first = scene['first_sample_token']
        samples = []
        while first != "":
            sample = nusc.get('sample', first)
            samples.append(first)
            first = sample['next']
        scene_to_samples[scene['token']] = {"name": scene['name'], "samples": samples}

    token_to_name = {}
    for scene_idx, (scene_token, scene_info) in enumerate(scene_to_samples.items(), start=1):
        scene_name = f"scene_{scene_idx:04d}"
        for frame_idx, token in enumerate(scene_info["samples"], start=1):
            token_to_name[token] = f"{scene_name}_frame_{frame_idx:04d}"

    ordered_tokens = [t for t in token_to_name.keys() if t in sample_token_list]
    print(f"Total sorted samples: {len(ordered_tokens)}")

    base_dir = os.path.dirname(args.results_path)
    out_dir = os.path.join(base_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    def process_token(token):
        name = token_to_name[token]
        out_path = os.path.join(out_dir, name)
        try:
            if not args.bev_only:
                render_sample_data(
                    token,
                    pred_data=bevformer_results,
                    out_path=out_path,
                    verbose=False,
                    score_thr=args.score_thr,
                )
            render_sample_bev(
                token,
                pred_data=bevformer_results,
                out_path=out_path,
                verbose=False,
                score_thr=args.score_thr,
            )
        except Exception as e:
            print(f"Error rendering {name}: {e}")

    with mp.Pool(processes=args.workers) as pool:
        list(tqdm(pool.imap_unordered(process_token, ordered_tokens),
                  total=len(ordered_tokens), desc="Rendering samples"))
