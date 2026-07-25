from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal, Type, TypeVar

from frenet_interface import FrenetPlannerConfig
from nuscenes_fot import (
    NuScenesFOTAdapter,
    NuScenesFOTAdapterConfig,
    NuScenesFOTPaths,
    NuScenesFOTPlanOutput,
)
from nuscenes_visualization import (
    VisualizationConfig,
    VisualizationOutput,
    save_plan_visualizations,
)


T = TypeVar("T")


RunMode = Literal["single", "scene_all"]
FrameFilter = Literal[
    "all",
    "tracking",
    "prediction",
    "tracking_or_prediction",
    "tracking_and_prediction",
]


@dataclass(slots=True)
class RunConfig:
    mode: RunMode = "single"
    frame_token: str | None = None
    scene_name: str | None = None
    frame_filter: FrameFilter = "tracking_or_prediction"
    skip_missing_cam_front_image: bool = True
    output_path: str | None = None
    output_dir: str | None = None
    visualization_dir: str | None = None
    auto_make_bev_video: bool = True
    bev_video_path: str | None = None
    bev_video_fps: float = 2.0
    auto_make_cam_video: bool = True
    cam_video_path: str | None = None
    cam_video_fps: float = 2.0


@dataclass(slots=True)
class RunSingleResult:
    mode: RunMode
    output: NuScenesFOTPlanOutput
    output_path: Path | None
    visualization_output: VisualizationOutput | None


@dataclass(slots=True)
class FrameRunArtifact:
    frame_token: str
    sample_token: str
    has_solution: bool
    output_path: Path | None
    bev_output_path: Path | None
    cam_front_output_path: Path | None


@dataclass(slots=True)
class RunSceneAllResult:
    mode: RunMode
    scene_name: str
    num_frames: int
    num_success: int
    summary_output_path: Path | None
    bev_video_path: Path | None
    cam_video_path: Path | None
    frame_artifacts: list[FrameRunArtifact]


def load_config_file(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML config files"
            ) from exc
        with config_path.open() as handle:
            data = yaml.safe_load(handle)
        return data or {}
    if suffix == ".json":
        with config_path.open() as handle:
            return json.load(handle)
    raise ValueError(f"unsupported config format: {config_path}, use .yaml/.yml or .json")


def run_nuscenes_fot_from_config(
    config_path: str | Path,
) -> RunSingleResult | RunSceneAllResult:
    config = load_config_file(config_path)

    paths = _build_dataclass(NuScenesFOTPaths, config.get("paths", {}))
    adapter = NuScenesFOTAdapter(paths)
    adapter_config = _build_dataclass(
        NuScenesFOTAdapterConfig,
        config.get("adapter", {}),
    )
    planner_config = _build_dataclass(
        FrenetPlannerConfig,
        config.get("planner", {}),
    )

    run_config = _build_dataclass(RunConfig, config.get("run", {}))
    if run_config.mode == "scene_all":
        return _run_scene_all(
            adapter=adapter,
            adapter_config=adapter_config,
            planner_config=planner_config,
            run_config=run_config,
            config=config,
        )
    return _run_single(
        adapter=adapter,
        adapter_config=adapter_config,
        planner_config=planner_config,
        run_config=run_config,
        config=config,
    )


def _run_single(
    *,
    adapter: NuScenesFOTAdapter,
    adapter_config: NuScenesFOTAdapterConfig,
    planner_config: FrenetPlannerConfig,
    run_config: RunConfig,
    config: dict[str, Any],
) -> RunSingleResult:
    if not run_config.frame_token:
        raise ValueError("run.frame_token is required for run.mode=single")

    output = adapter.plan_frame(
        frame_token=run_config.frame_token,
        planner_config=planner_config,
        adapter_config=adapter_config,
    )

    visualization_output: VisualizationOutput | None = None
    if "visualization" in config:
        visualization_config = _build_dataclass(
            VisualizationConfig,
            config.get("visualization", {}),
        )
        if visualization_config.enabled:
            visualization_output = save_plan_visualizations(
                adapter,
                output,
                visualization_config,
            )

    output_path = Path(run_config.output_path) if run_config.output_path else None
    if output_path is not None:
        save_plan_output(output, output_path)

    return RunSingleResult(
        mode="single",
        output=output,
        output_path=output_path,
        visualization_output=visualization_output,
    )


def _run_scene_all(
    *,
    adapter: NuScenesFOTAdapter,
    adapter_config: NuScenesFOTAdapterConfig,
    planner_config: FrenetPlannerConfig,
    run_config: RunConfig,
    config: dict[str, Any],
) -> RunSceneAllResult:
    if not run_config.frame_token and not run_config.scene_name:
        raise ValueError(
            "run.frame_token or run.scene_name is required for run.mode=scene_all"
        )

    sample_tokens = adapter.get_scene_sample_tokens(
        frame_token=run_config.frame_token,
        scene_name=run_config.scene_name,
    )
    sample_tokens = [
        token
        for token in sample_tokens
        if _keep_sample_token(adapter, token, run_config=run_config)
    ]
    if not sample_tokens:
        raise ValueError("no samples matched run.frame_filter / image availability constraints")

    scene_name = adapter.scene_by_token[
        adapter.sample_by_token[sample_tokens[0]]["scene_token"]
    ]["name"]

    summary_output_path = Path(run_config.output_path) if run_config.output_path else None
    output_dir = Path(run_config.output_dir) if run_config.output_dir else None
    visualization_dir = (
        Path(run_config.visualization_dir) if run_config.visualization_dir else None
    )

    visualization_config = (
        _build_dataclass(VisualizationConfig, config.get("visualization", {}))
        if "visualization" in config
        else None
    )

    frame_artifacts: list[FrameRunArtifact] = []
    for index, sample_token in enumerate(sample_tokens):
        output = adapter.plan_frame(
            frame_token=sample_token,
            planner_config=planner_config,
            adapter_config=adapter_config,
        )

        frame_output_path: Path | None = None
        if output_dir is not None:
            frame_output_path = output_dir / f"{index:04d}_{sample_token}.json"
            save_plan_output(output, frame_output_path)

        visualization_output: VisualizationOutput | None = None
        if visualization_config is not None and visualization_config.enabled:
            frame_vis_config = visualization_config
            if visualization_dir is not None:
                frame_vis_config = replace(
                    visualization_config,
                    bev_output_path=str(visualization_dir / f"{index:04d}_{sample_token}_bev.png"),
                    cam_front_output_path=str(
                        visualization_dir / f"{index:04d}_{sample_token}_cam_front.png"
                    ),
                )
            visualization_output = save_plan_visualizations(
                adapter,
                output,
                frame_vis_config,
            )

        frame_artifacts.append(
            FrameRunArtifact(
                frame_token=output.prepared_input.requested_frame_token,
                sample_token=output.prepared_input.sample_token,
                has_solution=output.planning_result.has_solution,
                output_path=frame_output_path,
                bev_output_path=(
                    visualization_output.bev_output_path
                    if visualization_output is not None
                    else None
                ),
                cam_front_output_path=(
                    visualization_output.cam_front_output_path
                    if visualization_output is not None
                    else None
                ),
            )
        )

    num_success = sum(1 for artifact in frame_artifacts if artifact.has_solution)
    bev_video_path: Path | None = None
    if run_config.auto_make_bev_video:
        bev_video_path = _save_image_video_from_artifacts(
            frame_artifacts=frame_artifacts,
            output_kind="bev",
            output_path_override=run_config.bev_video_path,
            fps=float(run_config.bev_video_fps),
            scene_name=scene_name,
            visualization_dir=run_config.visualization_dir,
        )
    cam_video_path: Path | None = None
    if run_config.auto_make_cam_video:
        cam_video_path = _save_image_video_from_artifacts(
            frame_artifacts=frame_artifacts,
            output_kind="cam_front",
            output_path_override=run_config.cam_video_path,
            fps=float(run_config.cam_video_fps),
            scene_name=scene_name,
            visualization_dir=run_config.visualization_dir,
        )
    result = RunSceneAllResult(
        mode="scene_all",
        scene_name=scene_name,
        num_frames=len(frame_artifacts),
        num_success=num_success,
        summary_output_path=summary_output_path,
        bev_video_path=bev_video_path,
        cam_video_path=cam_video_path,
        frame_artifacts=frame_artifacts,
    )
    if summary_output_path is not None:
        save_scene_all_summary(result, summary_output_path)
    return result


def save_plan_output(output: NuScenesFOTPlanOutput, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = plan_output_to_dict(output)
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def plan_output_to_dict(output: NuScenesFOTPlanOutput) -> dict[str, Any]:
    prepared = output.prepared_input
    result = output.planning_result
    long_cmd, lat_cmd = _infer_motion_command_labels(output)
    best_trajectory = asdict(result.best_trajectory) if result.best_trajectory else None
    return {
        "prepared_input": {
            "requested_frame_token": prepared.requested_frame_token,
            "sample_token": prepared.sample_token,
            "scene_name": prepared.scene_name,
            "location": prepared.location,
            "tracking_frame_token": prepared.tracking_frame_token,
            "reference_lane_tokens": list(prepared.reference_lane_tokens),
            "reference_waypoint_x": list(prepared.reference_waypoint_x),
            "reference_waypoint_y": list(prepared.reference_waypoint_y),
            "ego_state": asdict(prepared.ego_state),
            "tracking_boxes": [asdict(box) for box in prepared.tracking_boxes],
            "prediction_trajectories": [
                asdict(trajectory) for trajectory in prepared.prediction_trajectories
            ],
            "tracking_obstacle_points": [asdict(point) for point in prepared.tracking_obstacle_points],
            "prediction_obstacle_points": [asdict(point) for point in prepared.prediction_obstacle_points],
        },
        "planning_result": {
            "has_solution": result.has_solution,
            "candidate_counts": dict(result.candidate_counts),
            "inferred_command": {
                "longitudinal": long_cmd,
                "lateral": lat_cmd,
            },
            "input_frenet_state": asdict(result.input_frenet_state),
            "reference_course_length_m": result.reference_course.length,
            "reference_course_sampled_x": list(result.reference_course.sampled_x),
            "reference_course_sampled_y": list(result.reference_course.sampled_y),
            "reference_course_sampled_yaw": list(result.reference_course.sampled_yaw),
            "reference_course_sampled_curvature": list(result.reference_course.sampled_curvature),
            "best_trajectory": best_trajectory,
        },
    }


def save_scene_all_summary(result: RunSceneAllResult, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": result.mode,
        "scene_name": result.scene_name,
        "num_frames": result.num_frames,
        "num_success": result.num_success,
        "bev_video_path": str(result.bev_video_path) if result.bev_video_path else None,
        "cam_video_path": str(result.cam_video_path) if result.cam_video_path else None,
        "frames": [
            {
                "frame_token": artifact.frame_token,
                "sample_token": artifact.sample_token,
                "has_solution": artifact.has_solution,
                "output_path": str(artifact.output_path) if artifact.output_path else None,
                "bev_output_path": str(artifact.bev_output_path) if artifact.bev_output_path else None,
                "cam_front_output_path": (
                    str(artifact.cam_front_output_path)
                    if artifact.cam_front_output_path
                    else None
                ),
            }
            for artifact in result.frame_artifacts
        ],
    }
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def _save_image_video_from_artifacts(
    *,
    frame_artifacts: list[FrameRunArtifact],
    output_kind: str,
    output_path_override: str | None,
    fps: float,
    scene_name: str,
    visualization_dir: str | None,
) -> Path | None:
    if output_kind == "bev":
        image_paths = [
            Path(artifact.bev_output_path)
            for artifact in frame_artifacts
            if artifact.bev_output_path is not None and Path(artifact.bev_output_path).exists()
        ]
    elif output_kind == "cam_front":
        image_paths = [
            Path(artifact.cam_front_output_path)
            for artifact in frame_artifacts
            if artifact.cam_front_output_path is not None and Path(artifact.cam_front_output_path).exists()
        ]
    else:
        raise ValueError(f"unsupported output_kind={output_kind}")

    if not image_paths:
        return None

    if output_path_override:
        output_path = Path(output_path_override)
    else:
        base_dir = (
            Path(visualization_dir)
            if visualization_dir
            else image_paths[0].parent
        )
        output_path = base_dir / f"{scene_name}_{output_kind}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = max(0.1, float(fps))

    try:
        import imageio.v2 as imageio  # type: ignore

        with imageio.get_writer(str(output_path), fps=fps) as writer:
            for path in image_paths:
                writer.append_data(imageio.imread(path))
        return output_path
    except Exception:
        try:
            from PIL import Image
        except Exception:
            return None
        gif_path = output_path.with_suffix(".gif")
        frames = [Image.open(path).convert("RGB") for path in image_paths]
        if not frames:
            return None
        duration_ms = int(round(1000.0 / fps))
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        for frame in frames:
            frame.close()
        return gif_path


def _keep_sample_token(
    adapter: NuScenesFOTAdapter,
    sample_token: str,
    *,
    run_config: RunConfig,
) -> bool:
    if run_config.skip_missing_cam_front_image and not adapter.has_cam_front_image(sample_token):
        return False

    has_tracking = adapter.has_tracking_for_sample(sample_token)
    has_prediction = adapter.has_prediction_for_sample(sample_token)
    frame_filter = run_config.frame_filter
    if frame_filter == "all":
        return True
    if frame_filter == "tracking":
        return has_tracking
    if frame_filter == "prediction":
        return has_prediction
    if frame_filter == "tracking_or_prediction":
        return has_tracking or has_prediction
    if frame_filter == "tracking_and_prediction":
        return has_tracking and has_prediction
    raise ValueError(f"unsupported frame_filter: {frame_filter}")


def _infer_motion_command_labels(output: NuScenesFOTPlanOutput) -> tuple[str, str]:
    trajectory = output.planning_result.best_trajectory
    if trajectory is None or len(trajectory.acceleration) == 0 or len(trajectory.d) == 0:
        return "no_plan", "no_plan"

    accel = float(trajectory.acceleration[min(1, len(trajectory.acceleration) - 1)])
    if accel > 0.3:
        long_cmd = "accelerate"
    elif accel < -0.3:
        long_cmd = "brake"
    else:
        long_cmd = "keep_speed"

    lateral_shift = float(trajectory.d[-1] - trajectory.d[0])
    if lateral_shift > 0.8:
        lat_cmd = "lane_change_left"
    elif lateral_shift < -0.8:
        lat_cmd = "lane_change_right"
    else:
        lat_cmd = "keep_lane"
    return long_cmd, lat_cmd


def _build_dataclass(cls: Type[T], raw_values: dict[str, Any]) -> T:
    valid_fields = {field.name for field in fields(cls)}
    unknown_fields = sorted(set(raw_values) - valid_fields)
    if unknown_fields:
        raise ValueError(
            f"unknown fields for {cls.__name__}: {', '.join(unknown_fields)}"
        )
    return cls(**raw_values)
