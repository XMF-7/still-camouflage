from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


Pose = Tuple[float, float, float]
ArcLinePath = Dict[str, Any]


def principal_value(angle_in_radians: float) -> float:
    interval_min = -math.pi
    two_pi = 2 * math.pi
    return (angle_in_radians - interval_min) % two_pi + interval_min


def compute_segment_sign(arcline_path: ArcLinePath) -> Tuple[int, int, int]:
    shape = arcline_path["shape"]
    segment_sign = [0, 0, 0]
    segment_sign[0] = 1 if shape in ("LRL", "LSL", "LSR") else -1
    if shape == "RLR":
        segment_sign[1] = 1
    elif shape == "LRL":
        segment_sign[1] = -1
    else:
        segment_sign[1] = 0
    segment_sign[2] = 1 if shape in ("LRL", "LSL", "RSL") else -1
    return segment_sign[0], segment_sign[1], segment_sign[2]


def get_transformation_at_step(pose: Pose, step: float) -> Pose:
    theta = pose[2] * step
    ctheta = math.cos(theta)
    stheta = math.sin(theta)
    if abs(pose[2]) < 1.0e-6:
        return pose[0] * step, pose[1] * step, theta
    new_x = (pose[1] * (ctheta - 1.0) + pose[0] * stheta) / pose[2]
    new_y = (pose[0] * (1.0 - ctheta) + pose[1] * stheta) / pose[2]
    return new_x, new_y, theta


def apply_affine_transformation(pose: Pose, transformation: Pose) -> Pose:
    new_x = math.cos(pose[2]) * transformation[0] - math.sin(pose[2]) * transformation[1] + pose[0]
    new_y = math.sin(pose[2]) * transformation[0] + math.cos(pose[2]) * transformation[1] + pose[1]
    new_yaw = principal_value(pose[2] + transformation[2])
    return new_x, new_y, new_yaw


def _get_lie_algebra(
    segment_sign: Tuple[int, int, int],
    radius: float,
) -> List[Tuple[float, float, float]]:
    return [
        (1.0, 0.0, segment_sign[0] / radius),
        (1.0, 0.0, segment_sign[1] / radius),
        (1.0, 0.0, segment_sign[2] / radius),
    ]


def pose_at_length(arcline_path: ArcLinePath, pos: float) -> Pose:
    path_length = sum(arcline_path["segment_length"])
    pos = max(0.0, min(pos, path_length))
    result = arcline_path["start_pose"]
    segment_sign = compute_segment_sign(arcline_path)
    break_points = _get_lie_algebra(segment_sign, arcline_path["radius"])
    for index, break_point in enumerate(break_points):
        length = arcline_path["segment_length"][index]
        if pos <= length:
            transformation = get_transformation_at_step(break_point, pos)
            return apply_affine_transformation(result, transformation)
        transformation = get_transformation_at_step(break_point, length)
        result = apply_affine_transformation(result, transformation)
        pos -= length
    return result


def discretize(arcline_path: ArcLinePath, resolution_meters: float) -> List[Pose]:
    path_length = sum(arcline_path["segment_length"])
    radius = arcline_path["radius"]
    n_points = int(max(math.ceil(path_length / resolution_meters) + 1.5, 2))
    resolution_meters = path_length / (n_points - 1)
    discretization: List[Pose] = []
    cumulative_length = [
        arcline_path["segment_length"][0],
        arcline_path["segment_length"][0] + arcline_path["segment_length"][1],
        path_length + resolution_meters,
    ]
    segment_sign = compute_segment_sign(arcline_path)
    poses = _get_lie_algebra(segment_sign, radius)
    temp_pose = arcline_path["start_pose"]
    pose_index = 0
    pose_s = 0.0
    for step in range(n_points):
        step_along_path = step * resolution_meters
        if step_along_path > cumulative_length[pose_index]:
            temp_pose = pose_at_length(arcline_path, step_along_path)
            pose_s = step_along_path
            pose_index += 1
        transformation = get_transformation_at_step(poses[pose_index], step_along_path - pose_s)
        discretization.append(apply_affine_transformation(temp_pose, transformation))
    return discretization


def discretize_lane(lane: List[ArcLinePath], resolution_meters: float) -> List[Pose]:
    pose_list: List[Pose] = []
    for path in lane:
        pose_list.extend(discretize(path, resolution_meters))
    return pose_list
