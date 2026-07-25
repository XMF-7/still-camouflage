from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from nuscenes_arcline_path_utils import discretize_lane


class SimpleNuScenesMap:
    def __init__(self, dataroot: Union[str, Path], map_name: str) -> None:
        self.dataroot = Path(dataroot)
        map_path = self.dataroot / "maps" / "expansion" / f"{map_name}.json"
        if not map_path.exists():
            raise FileNotFoundError(f"nuScenes map JSON not found: {map_path}")
        with map_path.open("r") as handle:
            map_data = json.load(handle)
        self._arcline_path = map_data["arcline_path_3"]
        self._polygon_by_token = {record["token"]: record for record in map_data.get("polygon", [])}
        self._node_xy_by_token = {
            record["token"]: np.asarray([float(record["x"]), float(record["y"])], dtype=float)
            for record in map_data.get("node", [])
        }
        self._lane_tokens = [record["token"] for record in map_data["lane"]] + [
            record["token"] for record in map_data["lane_connector"]
        ]
        self._lane_polygon_tokens = [
            record["polygon_token"] for record in map_data["lane"] if record.get("polygon_token")
        ] + [
            record["polygon_token"] for record in map_data["lane_connector"] if record.get("polygon_token")
        ]
        self._discretized_cache: Dict[Tuple[str, float], np.ndarray] = {}
        self._polygon_cache: Dict[str, np.ndarray] = {}

    def discretize_lane(self, lane_token: str, resolution_meters: float) -> np.ndarray:
        cache_key = (lane_token, resolution_meters)
        if cache_key not in self._discretized_cache:
            points = discretize_lane(self._arcline_path.get(lane_token, []), resolution_meters)
            self._discretized_cache[cache_key] = np.asarray(points, dtype=float)
        return self._discretized_cache[cache_key]

    def get_centerlines_in_radius(
        self,
        x: float,
        y: float,
        *,
        radius_m: float,
        resolution_meters: float = 1.0,
    ) -> List[np.ndarray]:
        query_xy = np.asarray([x, y], dtype=float)
        radius_sq = float(radius_m * radius_m)
        nearby_centerlines: List[np.ndarray] = []
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
            poly = np.zeros((0, 2), dtype=float)
        else:
            poly = np.asarray(coords, dtype=float)
        self._polygon_cache[polygon_token] = poly
        return poly

    def get_lane_polygons_in_radius(self, x: float, y: float, *, radius_m: float) -> List[np.ndarray]:
        query_xy = np.asarray([x, y], dtype=float)
        radius_sq = float(radius_m * radius_m)
        nearby_polygons: List[np.ndarray] = []
        for polygon_token in self._lane_polygon_tokens:
            polygon_xy = self._polygon_xy(polygon_token)
            if len(polygon_xy) == 0:
                continue
            distances_sq = np.sum(np.square(polygon_xy - query_xy), axis=1)
            if float(np.min(distances_sq)) <= radius_sq:
                nearby_polygons.append(polygon_xy)
        return nearby_polygons
