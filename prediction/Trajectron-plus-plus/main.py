#!/usr/bin/env python3
"""
Trajectron++ Qualitative Prediction with Custom Tracking Input

Converted from Tracking_Input_Qualitative.ipynb.
All configurations are defined in the CONFIG section at the top of this file.
Just edit the values below and run `python main.py`.

No command line arguments needed.
"""

import sys
import json
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as patches
from pyquaternion import Quaternion

# ==================== CONFIGURATION (edit here) ====================

CONFIG = {
    # Input data
    "tracking_json": Path("/home/jushuo/Code/zz3-3D-camouflage/result/evaluation-0501-bevdet/f-pipeline/sample-001/track-ab-from-det-external-json/attacked/tracking.json"),
    "sample_tokens": [
        "85b8779598cb46f89538faa1d7117404",
        "8584cdee278d4bf89e6620ab02d15a7b",
        "4dfb0052fe944420acddf5d376755dff",
    ],
    "target_tracking_id": "4",                    # tracking_id in your JSON
    "target_instance_token": "ff3415fdc94a49b28c6a0102f0166b31",

    # Prediction settings
    "ph": 6,                                      # prediction horizon (future timesteps)
    "history_steps_before_first_tracking": 8,     # GT history frames before tracking
    "frequency_hz": 2.0,

    # Model
    "model_dir": Path("models/int_ee_me"),
    "checkpoint": 12,                             # corresponds to model_registrar-12.pt

    # nuScenes
    "nuscenes_dataroot": Path("/home/jushuo/Code/zz4-bev/nuCarla/BEVFormer/data/nuscenes"),
    "nuscenes_version": "v1.0-trainval",

    # Output
    "output_dir": Path("qualitative_output/sample-001"),
    "save_prediction_json": True,
    "save_visualization": True,
    "figsize": (14, 12),
    "map_alpha": 0.6,
}

# =================================================================

# Add Trajectron to path
TRAJECTRON_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAJECTRON_ROOT))
sys.path.insert(0, str(TRAJECTRON_ROOT / "trajectron"))

from trajectron.environment import Environment, Scene, Node, GeometricMap, derivative_of
from trajectron.model.model_registrar import ModelRegistrar
from trajectron.visualization.visualization import visualize_prediction
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap


def yaw_from_quaternion(rotation):
    return Quaternion(rotation).yaw_pitch_roll[0]


def annotation_for_instance_at_sample(nusc, instance_token, sample_token):
    sample = nusc.get('sample', sample_token)
    for ann_token in sample.get('anns', []):
        ann = nusc.get('sample_annotation', ann_token)
        if ann['instance_token'] == instance_token:
            return ann
    return None


def record_from_annotation(ann, source):
    return {
        'source': source,
        'sample_token': ann['sample_token'],
        'translation': list(ann['translation']),
        'rotation': list(ann['rotation']),
        'size': list(ann['size']),
        'instance_token': ann['instance_token'],
    }


def record_from_tracking_box(box, sample_token):
    return {
        'source': 'tracking',
        'sample_token': sample_token,
        'translation': list(box.get('translation', [0.0, 0.0, 0.0])),
        'rotation': list(box.get('rotation', [1.0, 0.0, 0.0, 0.0])),
        'size': list(box.get('size', [2.0, 4.5, 1.5])),
        'tracking_id': str(box.get('tracking_id', '')),
        'tracking_name': box.get('tracking_name'),
        'tracking_score': box.get('tracking_score'),
    }


def collect_gt_history_before_first_tracking(nusc, instance_token, first_sample_token, max_steps):
    first_ann = annotation_for_instance_at_sample(nusc, instance_token, first_sample_token)
    if first_ann is None:
        raise RuntimeError(f"Target instance {instance_token} not found at sample {first_sample_token}")

    records = []
    ann_token = first_ann.get('prev')
    while ann_token and len(records) < max_steps:
        ann = nusc.get('sample_annotation', ann_token)
        records.append(record_from_annotation(ann, 'gt_history'))
        ann_token = ann.get('prev')
    records.reverse()
    return records


def tracking_box_by_id(tracking_results, sample_token, tracking_id):
    for box in tracking_results.get(sample_token, []):
        if str(box.get('tracking_id')) == str(tracking_id):
            return box
    return None


def collect_tracking_records(tracking_results, sample_tokens, tracking_id):
    records = []
    for sample_token in sample_tokens:
        box = tracking_box_by_id(tracking_results, sample_token, tracking_id)
        if box is None:
            raise RuntimeError(f"tracking_id {tracking_id} not found at sample {sample_token}")
        records.append(record_from_tracking_box(box, sample_token))
    return records


def build_vehicle_dataframe(records, origin_xy, dt):
    """Build DataFrame exactly as Trajectron expects."""
    xy_global = np.asarray([r['translation'][:2] for r in records], dtype=float)
    xy = xy_global - origin_xy.reshape(1, 2)

    x = xy[:, 0]
    y = xy[:, 1]
    heading = np.unwrap(np.asarray([yaw_from_quaternion(r['rotation']) for r in records], dtype=float))

    vx = derivative_of(x, dt)
    vy = derivative_of(y, dt)
    ax = derivative_of(vx, dt)
    ay = derivative_of(vy, dt)

    v = np.stack((vx, vy), axis=-1)
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1e-6))

    data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
    data_columns = data_columns.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
    data_columns = data_columns.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))

    data_dict = {
        ('position', 'x'): x,
        ('position', 'y'): y,
        ('velocity', 'x'): vx,
        ('velocity', 'y'): vy,
        ('velocity', 'norm'): np.linalg.norm(v, axis=-1),
        ('acceleration', 'x'): ax,
        ('acceleration', 'y'): ay,
        ('acceleration', 'norm'): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
        ('heading', 'x'): heading_v[:, 0],
        ('heading', 'y'): heading_v[:, 1],
        ('heading', '°'): heading,
        ('heading', 'd°'): derivative_of(heading, dt, radian=True),
    }
    return pd.DataFrame(data_dict, columns=data_columns)


def create_custom_scene(nusc, tracking_results, config):
    """Build custom Trajectron Environment from tracking + GT history."""
    gt_history = collect_gt_history_before_first_tracking(
        nusc, config["target_instance_token"], config["sample_tokens"][0], config["history_steps_before_first_tracking"]
    )
    tracking_records = collect_tracking_records(tracking_results, config["sample_tokens"], config["target_tracking_id"])
    observed_records = gt_history + tracking_records

    xy_global = np.asarray([r['translation'][:2] for r in observed_records], dtype=float)
    origin_xy = np.floor(np.min(xy_global, axis=0) - 50.0)

    standardization = {
        'VEHICLE': {
            'position': {'x': {'mean': 0, 'std': 80}, 'y': {'mean': 0, 'std': 80}},
            'velocity': {'x': {'mean': 0, 'std': 15}, 'y': {'mean': 0, 'std': 15}, 'norm': {'mean': 0, 'std': 15}},
            'acceleration': {'x': {'mean': 0, 'std': 4}, 'y': {'mean': 0, 'std': 4}, 'norm': {'mean': 0, 'std': 4}},
            'heading': {
                'x': {'mean': 0, 'std': 1},
                'y': {'mean': 0, 'std': 1},
                '°': {'mean': 0, 'std': np.pi},
                'd°': {'mean': 0, 'std': 1}
            }
        }
    }

    env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization)
    env.attention_radius = {
        (env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): 10.0,
        (env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE): 20.0,
        (env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN): 20.0,
        (env.NodeType.VEHICLE, env.NodeType.VEHICLE): 30.0,
    }
    env.robot_type = env.NodeType.VEHICLE

    # Map
    sample0 = nusc.get('sample', config["sample_tokens"][0])
    scene_info = nusc.get('scene', sample0['scene_token'])
    map_name = nusc.get('log', scene_info['log_token'])['location']
    nusc_map = NuScenesMap(dataroot=str(config["nuscenes_dataroot"]), map_name=map_name)

    x_min, y_min = origin_xy
    x_max = float(np.ceil(np.max(xy_global[:, 0]) + 50))
    y_max = float(np.ceil(np.max(xy_global[:, 1]) + 50))
    patch_box = (x_min + 0.5 * (x_max - x_min), y_min + 0.5 * (y_max - y_min), y_max - y_min, x_max - x_min)
    canvas_size = (int(3 * (y_max - y_min)), int(3 * (x_max - x_min)))
    homography = np.array([[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]])

    layers = ['drivable_area', 'road_segment', 'lane', 'ped_crossing', 'walkway']
    map_mask = (nusc_map.get_map_mask(patch_box, 0, layers, canvas_size) * 255.0).astype(np.uint8)
    map_mask = np.swapaxes(map_mask, 1, 2)

    type_map = {}
    type_map['VEHICLE'] = GeometricMap(
        data=np.stack((np.max(map_mask[:3], axis=0), map_mask[0], map_mask[1]), axis=0),
        homography=homography,
        description='map patch'
    )

    scene = Scene(timesteps=len(observed_records), dt=1.0/config["frequency_hz"], name="custom_tracking_scene")
    scene.map = type_map

    df = build_vehicle_dataframe(observed_records, origin_xy, 1.0/config["frequency_hz"])
    node = Node(
        node_type=env.NodeType.VEHICLE,
        node_id=config["target_tracking_id"],
        data=df,
        obs_length=len(observed_records) - config["ph"],
        pred_length=config["ph"],
        sample_token=config["sample_tokens"][-1]
    )
    scene.nodes = [node]
    env.scenes = [scene]

    print(f"Built scene with {len(observed_records)} timesteps for target {config['target_tracking_id']}")
    return env, origin_xy


def load_model(config):
    model_registrar = ModelRegistrar(config["model_dir"], 0)  # CPU for qualitative
    model_registrar.load_models([config["checkpoint"]])
    model = model_registrar.get_model(config["checkpoint"])
    model.eval()
    print(f"Loaded model: {config['model_dir']} checkpoint {config['checkpoint']}")
    return model


def main():
    config = CONFIG
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Trajectron++ Qualitative with Custom Tracking Input ===")
    print(f"Target: {config['target_tracking_id']} | PH: {config['ph']} | History: {config['history_steps_before_first_tracking']}")

    nusc = NuScenes(version=config["nuscenes_version"], dataroot=str(config["nuscenes_dataroot"]), verbose=False)

    with open(config["tracking_json"], 'r') as f:
        tracking_data = json.load(f)
    tracking_results = tracking_data.get('results', tracking_data)

    env, origin_xy = create_custom_scene(nusc, tracking_results, config)
    model = load_model(config)

    with torch.no_grad():
        predictions = model.predict(env.scenes, ph=config["ph"], batch_size=1, return_robot=False)

    # Visualization
    if config["save_visualization"]:
        fig, ax = plt.subplots(figsize=config["figsize"])
        visualize_prediction(ax, predictions, dt=1.0/config["frequency_hz"], max_hl=8, ph=config["ph"])
        ax.set_title(f"Trajectron++ Qualitative - Target {config['target_tracking_id']}\n"
                     f"PH={config['ph']} | Tracking Input + GT History")
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

        viz_path = output_dir / f"qualitative_{config['target_tracking_id']}_ph{config['ph']}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved: {viz_path}")

    # Save prediction
    if config["save_prediction_json"]:
        pred_path = output_dir / f"prediction_{config['target_tracking_id']}.json"
        with open(pred_path, 'w') as f:
            json.dump({
                "sample": "sample-001",
                "target_tracking_id": config["target_tracking_id"],
                "predictions": predictions,
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
            }, f, default=str, indent=2)
        print(f"Prediction JSON saved: {pred_path}")

    print("Done! Check output directory:", output_dir)


if __name__ == "__main__":
    main()
