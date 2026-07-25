from __future__ import annotations

import argparse

from nuscenes_fot_config import RunSceneAllResult, RunSingleResult, run_nuscenes_fot_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nuScenes FOT from a YAML/JSON config")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to a .yaml, .yml, or .json config file",
    )
    args = parser.parse_args()

    run_result = run_nuscenes_fot_from_config(args.config)
    if isinstance(run_result, RunSingleResult):
        output = run_result.output
        prepared = output.prepared_input
        result = output.planning_result
        print("mode:", run_result.mode)
        print("scene_name:", prepared.scene_name)
        print("sample_token:", prepared.sample_token)
        print("tracking_frame_token:", prepared.tracking_frame_token)
        print("location:", prepared.location)
        print("num_reference_points:", len(prepared.reference_waypoint_x))
        print("num_tracking_boxes:", len(prepared.tracking_boxes))
        print("num_prediction_trajectories:", len(prepared.prediction_trajectories))
        print("num_tracking_obstacle_points:", len(prepared.tracking_obstacle_points))
        print("num_prediction_obstacle_points:", len(prepared.prediction_obstacle_points))
        print("has_solution:", result.has_solution)
        print("candidate_counts:", result.candidate_counts)
        if result.best_trajectory is not None:
            print("trajectory_points:", len(result.best_trajectory.x))
            print("trajectory_last_xy:", (result.best_trajectory.x[-1], result.best_trajectory.y[-1]))
            print("trajectory_last_speed_mps:", result.best_trajectory.speed[-1])
            print("trajectory_cost:", result.best_trajectory.cost)
        if run_result.output_path is not None:
            print("output_path:", run_result.output_path)
        if run_result.visualization_output is not None:
            print("bev_output_path:", run_result.visualization_output.bev_output_path)
            print("cam_front_output_path:", run_result.visualization_output.cam_front_output_path)
        return

    if isinstance(run_result, RunSceneAllResult):
        print("mode:", run_result.mode)
        print("scene_name:", run_result.scene_name)
        print("num_frames:", run_result.num_frames)
        print("num_success:", run_result.num_success)
        if run_result.summary_output_path is not None:
            print("summary_output_path:", run_result.summary_output_path)
        if run_result.bev_video_path is not None:
            print("bev_video_path:", run_result.bev_video_path)
        if run_result.cam_video_path is not None:
            print("cam_video_path:", run_result.cam_video_path)
        for artifact in run_result.frame_artifacts[:3]:
            print(
                "frame_preview:",
                artifact.sample_token,
                "has_solution=",
                artifact.has_solution,
                "json=",
                artifact.output_path,
                "bev=",
                artifact.bev_output_path,
            )


if __name__ == "__main__":
    main()
