from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np

from pythonrobotics.cartesian_frenet_converter import CartesianFrenetConverter
from pythonrobotics import frenet_optimal_trajectory as pr_frenet


class LateralMode(str, Enum):
    HIGH_SPEED = "high_speed"
    LOW_SPEED = "low_speed"


class LongitudinalMode(str, Enum):
    VELOCITY_KEEPING = "velocity_keeping"
    MERGING_AND_STOPPING = "merging_and_stopping"


@dataclass(slots=True)
class FrenetPlannerConfig:
    lateral_mode: LateralMode = LateralMode.HIGH_SPEED
    longitudinal_mode: LongitudinalMode = LongitudinalMode.VELOCITY_KEEPING
    max_speed: float = 50.0 / 3.6
    max_accel: float = 5.0
    max_curvature: float = 1.0
    dt: float = 0.2
    max_t: float = 5.0
    min_t: float = 4.0
    n_s_sample: int = 1
    k_j: float = 0.1
    k_t: float = 0.1
    k_s_dot: float = 1.0
    k_d: float = 1.0
    k_s: float = 1.0
    k_lat: float = 1.0
    k_lon: float = 1.0
    max_road_width: float = 7.0
    d_road_w: float = 1.0
    target_speed: float = 30.0 / 3.6
    d_t_s: float = 5.0 / 3.6
    robot_radius: float = 2.0
    stop_s: float = 25.0
    d_s: float = 2.0
    n_stop_s_sample: int = 4
    projection_coarse_step: float = 0.5
    projection_fine_window: float = 1.0
    projection_fine_step: float = 0.05


@dataclass(slots=True)
class FrenetEgoState:
    s: float
    s_d: float
    s_dd: float = 0.0
    d: float = 0.0
    d_d: float = 0.0
    d_dd: float = 0.0


@dataclass(slots=True)
class CartesianEgoState:
    x: float
    y: float
    yaw: float
    speed: float
    acceleration: float = 0.0
    curvature: float = 0.0


@dataclass(slots=True)
class ObstaclePoint:
    x: float
    y: float


@dataclass(slots=True)
class ReferenceCourse:
    waypoint_x: list[float]
    waypoint_y: list[float]
    sampled_x: list[float]
    sampled_y: list[float]
    sampled_yaw: list[float]
    sampled_curvature: list[float]
    csp: object = field(repr=False)

    @property
    def length(self) -> float:
        return float(self.csp.s[-1])


@dataclass(slots=True)
class PlannedTrajectory:
    t: list[float]
    x: list[float]
    y: list[float]
    yaw: list[float]
    speed: list[float]
    acceleration: list[float]
    curvature: list[float]
    s: list[float]
    s_d: list[float]
    s_dd: list[float]
    d: list[float]
    d_d: list[float]
    d_dd: list[float]
    cost: float


@dataclass(slots=True)
class FrenetPlanningResult:
    reference_course: ReferenceCourse
    input_frenet_state: FrenetEgoState
    best_trajectory: PlannedTrajectory | None
    candidate_counts: dict[str, int]

    @property
    def has_solution(self) -> bool:
        return self.best_trajectory is not None


def build_reference_course(
    waypoint_x: Sequence[float],
    waypoint_y: Sequence[float],
) -> ReferenceCourse:
    if len(waypoint_x) != len(waypoint_y):
        raise ValueError("waypoint_x and waypoint_y must have the same length")
    if len(waypoint_x) < 2:
        raise ValueError("at least two waypoints are required")

    sampled_x, sampled_y, sampled_yaw, sampled_curvature, csp = (
        pr_frenet.generate_target_course(list(waypoint_x), list(waypoint_y))
    )
    return ReferenceCourse(
        waypoint_x=list(waypoint_x),
        waypoint_y=list(waypoint_y),
        sampled_x=sampled_x,
        sampled_y=sampled_y,
        sampled_yaw=sampled_yaw,
        sampled_curvature=sampled_curvature,
        csp=csp,
    )


def project_cartesian_state_to_frenet(
    reference_course: ReferenceCourse,
    ego_state: CartesianEgoState,
    config: FrenetPlannerConfig | None = None,
) -> FrenetEgoState:
    config = config or FrenetPlannerConfig()
    max_valid_s = max(reference_course.length - 1.0e-6, 0.0)
    coarse_step = min(config.projection_coarse_step, max_valid_s) if max_valid_s > 0.0 else 1.0
    coarse_s = np.arange(0.0, max_valid_s + coarse_step, coarse_step)
    coarse_s = np.clip(coarse_s, 0.0, max_valid_s)
    coarse_points = np.array(
        [reference_course.csp.calc_position(float(s)) for s in coarse_s],
        dtype=float,
    )
    coarse_distance = np.sum(
        np.square(coarse_points - np.array([ego_state.x, ego_state.y], dtype=float)),
        axis=1,
    )
    closest_idx = int(np.argmin(coarse_distance))
    coarse_best_s = float(coarse_s[closest_idx])

    fine_s_min = max(0.0, coarse_best_s - config.projection_fine_window)
    fine_s_max = min(max_valid_s, coarse_best_s + config.projection_fine_window)
    fine_s = np.arange(
        fine_s_min,
        fine_s_max + config.projection_fine_step,
        config.projection_fine_step,
    )
    fine_s = np.clip(fine_s, 0.0, max_valid_s)
    fine_points = np.array(
        [reference_course.csp.calc_position(float(s)) for s in fine_s],
        dtype=float,
    )
    fine_distance = np.sum(
        np.square(fine_points - np.array([ego_state.x, ego_state.y], dtype=float)),
        axis=1,
    )
    rs = float(fine_s[int(np.argmin(fine_distance))])

    rx, ry = reference_course.csp.calc_position(rs)
    rtheta = reference_course.csp.calc_yaw(rs)
    rkappa = reference_course.csp.calc_curvature(rs)
    rdkappa = reference_course.csp.calc_curvature_rate(rs)
    s_condition, d_condition = CartesianFrenetConverter.cartesian_to_frenet(
        rs=rs,
        rx=rx,
        ry=ry,
        rtheta=rtheta,
        rkappa=rkappa,
        rdkappa=rdkappa,
        x=ego_state.x,
        y=ego_state.y,
        v=ego_state.speed,
        a=ego_state.acceleration,
        theta=ego_state.yaw,
        kappa=ego_state.curvature,
    )
    return FrenetEgoState(
        s=float(s_condition[0]),
        s_d=float(s_condition[1]),
        s_dd=float(s_condition[2]),
        d=float(d_condition[0]),
        d_d=float(d_condition[1]),
        d_dd=float(d_condition[2]),
    )


def plan_once(
    waypoint_x: Sequence[float],
    waypoint_y: Sequence[float],
    *,
    obstacles: Sequence[ObstaclePoint | tuple[float, float]] = (),
    ego_frenet_state: FrenetEgoState | None = None,
    ego_cartesian_state: CartesianEgoState | None = None,
    config: FrenetPlannerConfig | None = None,
) -> FrenetPlanningResult:
    if (ego_frenet_state is None) == (ego_cartesian_state is None):
        raise ValueError("provide exactly one of ego_frenet_state or ego_cartesian_state")

    config = config or FrenetPlannerConfig()
    reference_course = build_reference_course(waypoint_x, waypoint_y)
    _apply_planner_config(config)

    if ego_frenet_state is None:
        ego_frenet_state = project_cartesian_state_to_frenet(
            reference_course=reference_course,
            ego_state=ego_cartesian_state,
            config=config,
        )

    obstacle_array = _normalize_obstacles(obstacles)
    best_path, path_dict = pr_frenet.frenet_optimal_planning(
        reference_course.csp,
        ego_frenet_state.s,
        ego_frenet_state.s_d,
        ego_frenet_state.s_dd,
        ego_frenet_state.d,
        ego_frenet_state.d_d,
        ego_frenet_state.d_dd,
        obstacle_array,
    )

    return FrenetPlanningResult(
        reference_course=reference_course,
        input_frenet_state=ego_frenet_state,
        best_trajectory=_convert_path(best_path),
        candidate_counts={key: len(value) for key, value in path_dict.items()},
    )


def _apply_planner_config(config: FrenetPlannerConfig) -> None:
    lateral_movement = (
        pr_frenet.LateralMovement.HIGH_SPEED
        if config.lateral_mode == LateralMode.HIGH_SPEED
        else pr_frenet.LateralMovement.LOW_SPEED
    )
    longitudinal_movement = (
        pr_frenet.LongitudinalMovement.VELOCITY_KEEPING
        if config.longitudinal_mode == LongitudinalMode.VELOCITY_KEEPING
        else pr_frenet.LongitudinalMovement.MERGING_AND_STOPPING
    )
    pr_frenet.configure_planner(
        lateral_movement=lateral_movement,
        longitudinal_movement=longitudinal_movement,
        max_speed=config.max_speed,
        max_accel=config.max_accel,
        max_curvature=config.max_curvature,
        dt=config.dt,
        max_t=config.max_t,
        min_t=config.min_t,
        n_s_sample=config.n_s_sample,
        k_j=config.k_j,
        k_t=config.k_t,
        k_s_dot=config.k_s_dot,
        k_d=config.k_d,
        k_s=config.k_s,
        k_lat=config.k_lat,
        k_lon=config.k_lon,
        max_road_width=config.max_road_width,
        d_road_w=config.d_road_w,
        target_speed=config.target_speed,
        d_t_s=config.d_t_s,
        robot_radius=config.robot_radius,
        stop_s=config.stop_s,
        d_s=config.d_s,
        n_stop_s_sample=config.n_stop_s_sample,
    )


def _normalize_obstacles(
    obstacles: Sequence[ObstaclePoint | tuple[float, float]],
) -> np.ndarray:
    if not obstacles:
        return np.empty((0, 2), dtype=float)

    obstacle_xy = []
    for obstacle in obstacles:
        if isinstance(obstacle, ObstaclePoint):
            obstacle_xy.append([obstacle.x, obstacle.y])
        else:
            obstacle_xy.append([float(obstacle[0]), float(obstacle[1])])
    return np.asarray(obstacle_xy, dtype=float)


def _convert_path(path) -> PlannedTrajectory | None:
    if path is None:
        return None
    return PlannedTrajectory(
        t=[float(value) for value in path.t],
        x=[float(value) for value in path.x],
        y=[float(value) for value in path.y],
        yaw=[float(value) for value in path.yaw],
        speed=[float(value) for value in path.v],
        acceleration=[float(value) for value in path.a],
        curvature=[float(value) for value in path.c],
        s=[float(value) for value in path.s],
        s_d=[float(value) for value in path.s_d],
        s_dd=[float(value) for value in path.s_dd],
        d=[float(value) for value in path.d],
        d_d=[float(value) for value in path.d_d],
        d_dd=[float(value) for value in path.d_dd],
        cost=float(path.cf),
    )
