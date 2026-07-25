from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from nuscenes_arcline_path_utils import discretize_lane


class SimpleNuScenesMap:
    def __init__(self, dataroot: str | Path, map_name: str) -> None:
        self.dataroot = Path(dataroot)
        self.map_name = map_name
        map_path = self.dataroot / "maps" / "expansion" / f"{map_name}.json"
        if not map_path.exists():
            raise FileNotFoundError(f"nuScenes map JSON not found: {map_path}")

        with map_path.open() as handle:
            map_data = json.load(handle)

        self._connectivity = map_data["connectivity"]
        self._arcline_path = map_data["arcline_path_3"]
        self._polygon_by_token = {record["token"]: record for record in map_data.get("polygon", [])}
        self._node_xy_by_token = {
            record["token"]: np.asarray([float(record["x"]), float(record["y"])], dtype=float)
            for record in map_data.get("node", [])
        }
        self._lane_tokens = [
            record["token"] for record in map_data["lane"]
        ] + [
            record["token"] for record in map_data["lane_connector"]
        ]
        self._lane_polygon_tokens = [
            record["polygon_token"] for record in map_data["lane"] if record.get("polygon_token")
        ] + [
            record["polygon_token"] for record in map_data["lane_connector"] if record.get("polygon_token")
        ]
        self._discretized_cache: dict[tuple[str, float], np.ndarray] = {}
        self._polygon_cache: dict[str, np.ndarray] = {}

    @property
    def lane_tokens(self) -> list[str]:
        return list(self._lane_tokens)

    def get_outgoing_lane_ids(self, lane_token: str) -> list[str]:
        return list(self._connectivity.get(lane_token, {}).get("outgoing", []))

    def discretize_lane(self, lane_token: str, resolution_meters: float) -> np.ndarray:
        cache_key = (lane_token, resolution_meters)
        if cache_key not in self._discretized_cache:
            discrete = discretize_lane(
                self._arcline_path.get(lane_token, []),
                resolution_meters,
            )
            self._discretized_cache[cache_key] = np.asarray(discrete, dtype=float)
        return self._discretized_cache[cache_key]

    def get_closest_lane(
        self,
        x: float,
        y: float,
        *,
        radius: float = 20.0,
        heading: float | None = None,
        resolution_meters: float = 1.0,
    ) -> str:
        query_xy = np.array([x, y], dtype=float)
        best_lane = ""
        best_score = float("inf")
        radius_sq = radius * radius

        for lane_token in self._lane_tokens:
            points = self.discretize_lane(lane_token, resolution_meters)
            if len(points) == 0:
                continue

            distance_sq = np.sum(np.square(points[:, :2] - query_xy), axis=1)
            nearest_index = int(np.argmin(distance_sq))
            nearest_distance_sq = float(distance_sq[nearest_index])
            if nearest_distance_sq > radius_sq:
                continue

            score = nearest_distance_sq
            if heading is not None:
                lane_heading = float(points[nearest_index, 2])
                score += 4.0 * abs(_angle_diff(lane_heading, heading))

            if score < best_score:
                best_score = score
                best_lane = lane_token

        if not best_lane:
            raise ValueError(
                f"no nearby lane found around ({x:.2f}, {y:.2f}) within {radius:.1f} m"
            )
        return best_lane

    def build_forward_centerline(
        self,
        x: float,
        y: float,
        yaw: float,
        *,
        route_length_m: float = 80.0,
        resolution_meters: float = 1.0,
        closest_lane_radius: float = 20.0,
        max_lane_hops: int = 12,
    ) -> tuple[list[str], np.ndarray]:
        lane_token = self.get_closest_lane(
            x,
            y,
            radius=closest_lane_radius,
            heading=yaw,
            resolution_meters=resolution_meters,
        )
        lane_sequence: list[str] = []
        accumulated_points: list[list[float]] = []
        accumulated_length = 0.0
        visited: set[str] = set()
        query_xy = np.array([x, y], dtype=float)
        current_lane = lane_token

        for _ in range(max_lane_hops):
            if not current_lane or current_lane in visited:
                break

            visited.add(current_lane)
            lane_sequence.append(current_lane)
            points = self.discretize_lane(current_lane, resolution_meters)
            if len(points) == 0:
                break

            if not accumulated_points:
                nearest_index = int(
                    np.argmin(np.sum(np.square(points[:, :2] - query_xy), axis=1))
                )
                points = points[nearest_index:]
            else:
                points = points[1:]

            if len(points) == 0:
                break

            for point in points:
                if accumulated_points:
                    prev_xy = np.array(accumulated_points[-1][:2], dtype=float)
                    accumulated_length += float(
                        np.linalg.norm(point[:2] - prev_xy)
                    )
                accumulated_points.append(point.tolist())
                if accumulated_length >= route_length_m:
                    break

            if accumulated_length >= route_length_m:
                break

            outgoing = self.get_outgoing_lane_ids(current_lane)
            if not outgoing:
                break

            current_heading = float(accumulated_points[-1][2])
            current_lane = min(
                outgoing,
                key=lambda token: abs(
                    _angle_diff(
                        float(self.discretize_lane(token, resolution_meters)[0, 2]),
                        current_heading,
                    )
                ),
            )

        centerline = np.asarray(accumulated_points, dtype=float)
        if len(centerline) < 2:
            raise ValueError("failed to extract a usable forward centerline")
        return lane_sequence, centerline

    def get_centerlines_in_radius(
        self,
        x: float,
        y: float,
        *,
        radius_m: float,
        resolution_meters: float = 1.0,
    ) -> list[np.ndarray]:
        query_xy = np.asarray([x, y], dtype=float)
        radius_sq = float(radius_m * radius_m)
        nearby_centerlines: list[np.ndarray] = []
        for lane_token in self._lane_tokens:
            points = self.discretize_lane(lane_token, resolution_meters)
            if len(points) == 0:
                continue
            distances_sq = np.sum(np.square(points[:, :2] - query_xy), axis=1)
            if float(np.min(distances_sq)) <= radius_sq:
                nearby_centerlines.append(points)
        return nearby_centerlines

    def _polygon_xy(self, polygon_token: str) -> np.ndarray:
        if polygon_token in self._polygon_cache:
            return self._polygon_cache[polygon_token]
        polygon_record = self._polygon_by_token.get(polygon_token)
        if polygon_record is None:
            self._polygon_cache[polygon_token] = np.zeros((0, 2), dtype=float)
            return self._polygon_cache[polygon_token]
        node_tokens = polygon_record.get("exterior_node_tokens", [])
        coords = []
        for token in node_tokens:
            xy = self._node_xy_by_token.get(token)
            if xy is not None:
                coords.append(xy)
        if len(coords) == 0:
            polygon = np.zeros((0, 2), dtype=float)
        else:
            polygon = np.asarray(coords, dtype=float)
        self._polygon_cache[polygon_token] = polygon
        return polygon

    def get_lane_polygons_in_radius(
        self,
        x: float,
        y: float,
        *,
        radius_m: float,
    ) -> list[np.ndarray]:
        query_xy = np.asarray([x, y], dtype=float)
        radius_sq = float(radius_m * radius_m)
        nearby_polygons: list[np.ndarray] = []
        for polygon_token in self._lane_polygon_tokens:
            polygon = self._polygon_xy(polygon_token)
            if len(polygon) == 0:
                continue
            distances_sq = np.sum(np.square(polygon - query_xy), axis=1)
            if float(np.min(distances_sq)) <= radius_sq:
                nearby_polygons.append(polygon)
        return nearby_polygons


def _angle_diff(a: float, b: float) -> float:
    delta = a - b
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta
