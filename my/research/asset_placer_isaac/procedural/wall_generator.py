# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""
壁生成モジュール

JSONレイアウトデータから手続き的に壁を生成します。
ドア検出と連携して開口部を持つ壁を生成します。
"""

import math
from typing import Dict, List, Optional, Tuple
from pxr import Gf, Sdf, UsdGeom, UsdShade
import omni.log

from .door_detector import DoorDetector


class WallGenerator:
    """壁を手続き的に生成するクラス"""

    # デフォルトの壁の高さ（メートル単位）
    DEFAULT_WALL_HEIGHT = 2.5  # 2.5m

    # デフォルトの壁の厚み（メートル単位）
    DEFAULT_WALL_THICKNESS = 0.10  # 0.10m = 10cm

    # 開口が端まで食い込むのを防ぐ最小マージン（メートル）
    EDGE_END_MARGIN = 0.0

    def __init__(
        self,
        extract_float_func,
        extract_optional_float_func,
        get_unique_path_func,
        door_detector: DoorDetector,
    ):
        """
        Args:
            extract_float_func: JSONから浮動小数点値を抽出する関数
            extract_optional_float_func: JSONからオプショナルな浮動小数点値を抽出する関数
            get_unique_path_func: 一意なPrimパスを生成する関数
            door_detector: ドア検出器
        """
        self._extract_float = extract_float_func
        self._extract_optional_float = extract_optional_float_func
        self._get_unique_child_path = get_unique_path_func
        self._door_detector = door_detector

    def generate_walls(
        self,
        stage,
        root_prim_path: str,
        area_size_x: float,
        area_size_y: float,
        door_objects: List[Dict[str, object]] = None,
        origin_mode: str = "center",
    ) -> List[str]:
        """
        部屋の4方向の壁を生成します。
        ドアがある場合は開口部を作成します。

        Args:
            stage: USDステージ
            root_prim_path: 親Primのパス
            area_size_x: 部屋のX方向のサイズ（メートル単位）
            area_size_y: 部屋のY方向のサイズ（メートル単位）
            door_objects: ドアオブジェクトのリスト
            origin_mode: "center" or "bottom_left"

        Returns:
            生成された壁Primのパスのリスト
        """
        wall_paths = []

        # 4方向の壁の設定
        # (名前, X位置, Y位置, Xサイズ, Yサイズ, 方向)
        if origin_mode == "bottom_left":
            walls_config = [
                (
                    "Wall_North",
                    area_size_x / 2.0,
                    area_size_y,
                    area_size_x + self.DEFAULT_WALL_THICKNESS,
                    self.DEFAULT_WALL_THICKNESS,
                    "north",
                ),
                (
                    "Wall_South",
                    area_size_x / 2.0,
                    0.0,
                    area_size_x + self.DEFAULT_WALL_THICKNESS,
                    self.DEFAULT_WALL_THICKNESS,
                    "south",
                ),
                (
                    "Wall_East",
                    area_size_x,
                    area_size_y / 2.0,
                    self.DEFAULT_WALL_THICKNESS,
                    area_size_y,
                    "east",
                ),
                (
                    "Wall_West",
                    0.0,
                    area_size_y / 2.0,
                    self.DEFAULT_WALL_THICKNESS,
                    area_size_y,
                    "west",
                ),
            ]
        else:
            walls_config = [
                (
                    "Wall_North",
                    0.0,
                    area_size_y / 2.0,
                    area_size_x + self.DEFAULT_WALL_THICKNESS,
                    self.DEFAULT_WALL_THICKNESS,
                    "north",
                ),
                (
                    "Wall_South",
                    0.0,
                    -area_size_y / 2.0,
                    area_size_x + self.DEFAULT_WALL_THICKNESS,
                    self.DEFAULT_WALL_THICKNESS,
                    "south",
                ),
                (
                    "Wall_East",
                    area_size_x / 2.0,
                    0.0,
                    self.DEFAULT_WALL_THICKNESS,
                    area_size_y,
                    "east",
                ),
                (
                    "Wall_West",
                    -area_size_x / 2.0,
                    0.0,
                    self.DEFAULT_WALL_THICKNESS,
                    area_size_y,
                    "west",
                ),
            ]

        # 各壁を生成
        for wall_name, x, y, size_x, size_y, direction in walls_config:
            # この壁の方向にあるドアを検出
            doors_on_wall = []
            if door_objects:
                doors_on_wall = self._door_detector.find_doors_on_wall(
                    door_objects, direction, area_size_x, area_size_y, origin_mode
                )

            # ドアがある場合は開口部を持つ壁を生成
            if doors_on_wall:
                omni.log.info(f"{wall_name} にドアが検出されました。開口部を作成します。")
                wall_segment_paths: List[str] = []
                try:
                    wall_segment_paths = self._create_wall_segments_with_door_cutouts(
                        stage,
                        root_prim_path,
                        wall_name,
                        x,
                        y,
                        size_x,
                        size_y,
                        direction,
                        doors_on_wall,
                    )
                except Exception as exc:
                    omni.log.error(
                        f"{wall_name} の開口部付き壁生成に失敗しました: {exc}. "
                        "フォールバックとして通常の壁を生成します。"
                    )

                if wall_segment_paths:
                    wall_paths.extend(wall_segment_paths)
                else:
                    wall_path = self._create_single_wall(
                        stage,
                        root_prim_path,
                        wall_name,
                        x,
                        y,
                        size_x,
                        size_y,
                        self.DEFAULT_WALL_HEIGHT,
                        self.DEFAULT_WALL_THICKNESS,
                    )
                    if wall_path:
                        wall_paths.append(wall_path)
            else:
                # ドアがない場合は通常の壁を生成
                wall_path = self._create_single_wall(
                    stage,
                    root_prim_path,
                    wall_name,
                    x,
                    y,
                    size_x,
                    size_y,
                    self.DEFAULT_WALL_HEIGHT,
                    self.DEFAULT_WALL_THICKNESS,
                )
                if wall_path:
                    wall_paths.append(wall_path)

        omni.log.info(f"合計 {len(wall_paths)} 個の壁セグメントを生成しました。")
        return wall_paths

    def generate_walls_from_polygon(
        self,
        stage,
        root_prim_path: str,
        polygon_points: List[object],
        openings: Optional[List[Dict[str, object]]] = None,
        wall_height: Optional[float] = None,
        wall_thickness: Optional[float] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        ポリゴン外周から壁を生成し、開口部（窓/ドア）を切り抜く。
        openings: {type, X, Y, Width, Height, SillHeight} の配列
        """
        points = self._normalize_polygon_points(polygon_points)
        if len(points) < 3:
            omni.log.warn("Room polygon has fewer than 3 valid points.")
            return [], []

        actual_height = wall_height if wall_height is not None else self.DEFAULT_WALL_HEIGHT
        actual_thickness = wall_thickness if wall_thickness is not None else self.DEFAULT_WALL_THICKNESS

        wall_paths: List[str] = []
        window_paths: List[str] = []
        openings = openings or []

        edges = self._build_edges(points)
        edge_opening_map = self._assign_openings_to_edges(openings, edges)

        for edge_index, edge in enumerate(edges):
            p0 = edge["start"]
            unit = edge["unit"]
            edge_len = edge["length"]
            angle_deg = edge["angle"]

            edge_openings = edge_opening_map.get(edge_index, [])
            edge_openings.sort(key=lambda item: item["start"])

            cursor = 0.0
            for idx, opening in enumerate(edge_openings):
                start = max(0.0, opening["start"])
                end = min(edge_len, opening["end"])
                if start - cursor > 0.05:
                    segment_name = f"Wall_Edge{edge_index}_Seg{idx}"
                    path = self._create_oriented_wall_segment(
                        stage,
                        root_prim_path,
                        segment_name,
                        p0,
                        unit,
                        angle_deg,
                        cursor,
                        start,
                        actual_thickness,
                        actual_height,
                        actual_height / 2.0,
                    )
                    if path:
                        wall_paths.append(path)
                cursor = max(cursor, end)

                self._create_opening_segments(
                    stage,
                    root_prim_path,
                    edge_index,
                    idx,
                    p0,
                    unit,
                    angle_deg,
                    opening,
                    actual_thickness,
                    actual_height,
                    wall_paths,
                    window_paths,
                )

            if edge_len - cursor > 0.05:
                segment_name = f"Wall_Edge{edge_index}_Tail"
                path = self._create_oriented_wall_segment(
                    stage,
                    root_prim_path,
                    segment_name,
                    p0,
                    unit,
                    angle_deg,
                    cursor,
                    edge_len,
                    actual_thickness,
                    actual_height,
                    actual_height / 2.0,
                )
                if path:
                    wall_paths.append(path)

        omni.log.info(
            f"Polygon walls generated: walls={len(wall_paths)}, windows={len(window_paths)}"
        )
        return wall_paths, window_paths

    def generate_walls_from_edges(
        self,
        stage,
        root_prim_path: str,
        edges: List[Dict[str, object]],
        openings: Optional[List[Dict[str, object]]] = None,
        wall_height: Optional[float] = None,
        wall_thickness: Optional[float] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        事前に計算済みのエッジ情報から、壁と開口部を生成する。
        edges は _build_edges と同等の辞書構造を想定する。
        openings: {type, X, Y, Width, Height, SillHeight}
        """
        if not edges:
            return [], []

        actual_height = wall_height if wall_height is not None else self.DEFAULT_WALL_HEIGHT
        actual_thickness = wall_thickness if wall_thickness is not None else self.DEFAULT_WALL_THICKNESS

        wall_paths: List[str] = []
        window_paths: List[str] = []
        openings = openings or []

        edge_opening_map = self._assign_openings_to_edges(openings, edges)

        for edge_index, edge in enumerate(edges):
            p0 = edge["start"]
            unit = edge["unit"]
            edge_len = edge["length"]
            angle_deg = edge["angle"]

            edge_openings = edge_opening_map.get(edge_index, [])
            edge_openings.sort(key=lambda item: item["start"])

            cursor = 0.0
            for idx, opening in enumerate(edge_openings):
                start = max(0.0, opening["start"])
                end = min(edge_len, opening["end"])
                if start - cursor > 0.05:
                    segment_name = f"Wall_Edge{edge_index}_Seg{idx}"
                    path = self._create_oriented_wall_segment(
                        stage,
                        root_prim_path,
                        segment_name,
                        p0,
                        unit,
                        angle_deg,
                        cursor,
                        start,
                        actual_thickness,
                        actual_height,
                        actual_height / 2.0,
                    )
                    if path:
                        wall_paths.append(path)
                cursor = max(cursor, end)

                self._create_opening_segments(
                    stage,
                    root_prim_path,
                    edge_index,
                    idx,
                    p0,
                    unit,
                    angle_deg,
                    opening,
                    actual_thickness,
                    actual_height,
                    wall_paths,
                    window_paths,
                )

            if edge_len - cursor > 0.05:
                segment_name = f"Wall_Edge{edge_index}_Tail"
                path = self._create_oriented_wall_segment(
                    stage,
                    root_prim_path,
                    segment_name,
                    p0,
                    unit,
                    angle_deg,
                    cursor,
                    edge_len,
                    actual_thickness,
                    actual_height,
                    actual_height / 2.0,
                )
                if path:
                    wall_paths.append(path)

        return wall_paths, window_paths

    @staticmethod
    def _build_edges(points: List[Tuple[float, float]]) -> List[Dict[str, object]]:
        edges: List[Dict[str, object]] = []
        for edge_index in range(len(points)):
            p0 = points[edge_index]
            p1 = points[(edge_index + 1) % len(points)]
            edge_vec = (p1[0] - p0[0], p1[1] - p0[1])
            edge_len = math.hypot(edge_vec[0], edge_vec[1])
            if edge_len < 1e-4:
                continue
            unit = (edge_vec[0] / edge_len, edge_vec[1] / edge_len)
            angle_deg = math.degrees(math.atan2(unit[1], unit[0]))
            edges.append(
                {
                    "index": edge_index,
                    "start": p0,
                    "end": p1,
                    "unit": unit,
                    "length": edge_len,
                    "angle": angle_deg,
                }
            )
        return edges

    def _assign_openings_to_edges(
        self,
        openings: List[Dict[str, object]],
        edges: List[Dict[str, object]],
        tolerance: float = 0.25,
    ) -> Dict[int, List[Dict[str, object]]]:
        edge_openings: Dict[int, List[Dict[str, object]]] = {i: [] for i in range(len(edges))}
        for opening in openings or []:
            try:
                x = float(opening.get("X", 0.0))
                y = float(opening.get("Y", 0.0))
                width = float(opening.get("Width", 0.0))
                height = float(opening.get("Height", 0.0))
                sill = float(opening.get("SillHeight", 0.0))
            except (TypeError, ValueError):
                continue

            if width <= 0.0:
                continue

            best_edge_idx = None
            best_perp = None
            best_proj = None
            best_len = None

            for idx, edge in enumerate(edges):
                sx, sy = edge["start"]
                ux, uy = edge["unit"]
                edge_len = edge["length"]
                vx = x - sx
                vy = y - sy
                proj = vx * ux + vy * uy
                perp = abs(vx * (-uy) + vy * ux)
                if perp > tolerance:
                    continue
                if proj < -tolerance or proj > edge_len + tolerance:
                    continue
                if best_perp is None or perp < best_perp:
                    best_perp = perp
                    best_edge_idx = idx
                    best_proj = proj
                    best_len = edge_len

            forced_snap = False
            if best_edge_idx is None or best_proj is None or best_len is None:
                closest_idx = None
                closest_proj = None
                closest_len = None
                closest_dist = None
                for idx, edge in enumerate(edges):
                    sx, sy = edge["start"]
                    ux, uy = edge["unit"]
                    edge_len = edge["length"]
                    vx = x - sx
                    vy = y - sy
                    proj = vx * ux + vy * uy
                    proj_clamped = max(0.0, min(edge_len, proj))
                    cx = sx + ux * proj_clamped
                    cy = sy + uy * proj_clamped
                    dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                    if closest_dist is None or dist < closest_dist:
                        closest_dist = dist
                        closest_idx = idx
                        closest_proj = proj_clamped
                        closest_len = edge_len
                if closest_idx is None or closest_proj is None or closest_len is None:
                    continue
                max_snap_distance = max(tolerance * 2.0, width * 0.6)
                if closest_dist is None or closest_dist > max_snap_distance:
                    continue
                best_edge_idx = closest_idx
                best_proj = closest_proj
                best_len = closest_len
                forced_snap = True

            start = best_proj - width / 2.0
            end = best_proj + width / 2.0
            if best_len > 2 * WallGenerator.EDGE_END_MARGIN:
                start = max(start, WallGenerator.EDGE_END_MARGIN)
                end = min(end, best_len - WallGenerator.EDGE_END_MARGIN)
            edge_epsilon = 0.05
            if abs(start) <= edge_epsilon:
                start = 0.0
            if abs(best_len - end) <= edge_epsilon:
                end = best_len
            if end <= 0.0 or start >= best_len:
                continue

            edge_openings[best_edge_idx].append(
                {
                    "type": str(opening.get("type", "window")),
                    "start": max(0.0, start),
                    "end": min(best_len, end),
                    "height": max(0.0, height),
                    "sill": max(0.0, sill),
                    "center_t": max(0.0, min(best_len, best_proj)),
                }
            )
            if forced_snap:
                omni.log.warn(
                    f"Opening forced-snap to edge {best_edge_idx} (dist={closest_dist:.3f})"
                )

        return edge_openings

    @staticmethod
    def _normalize_polygon_points(points: List[object]) -> List[Tuple[float, float]]:
        normalized: List[Tuple[float, float]] = []
        for entry in points or []:
            if isinstance(entry, dict):
                x = entry.get("X", entry.get("x"))
                y = entry.get("Y", entry.get("y"))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                x, y = entry[0], entry[1]
            else:
                continue
            try:
                normalized.append((float(x), float(y)))
            except (TypeError, ValueError):
                continue

        if len(normalized) >= 2:
            first = normalized[0]
            last = normalized[-1]
            if abs(first[0] - last[0]) < 1e-6 and abs(first[1] - last[1]) < 1e-6:
                normalized.pop()
        return normalized

    @staticmethod
    def _find_openings_on_edge(
        openings: List[Dict[str, object]],
        edge_start: Tuple[float, float],
        edge_unit: Tuple[float, float],
        edge_len: float,
        tolerance: float = 0.25,
    ) -> List[Dict[str, object]]:
        edge_openings: List[Dict[str, object]] = []
        ux, uy = edge_unit
        sx, sy = edge_start

        for opening in openings:
            try:
                x = float(opening.get("X", 0.0))
                y = float(opening.get("Y", 0.0))
                width = float(opening.get("Width", 0.0))
                height = float(opening.get("Height", 0.0))
                sill = float(opening.get("SillHeight", 0.0))
            except (TypeError, ValueError):
                continue

            if width <= 0.0:
                continue

            vx = x - sx
            vy = y - sy
            proj = vx * ux + vy * uy
            perp = abs(vx * (-uy) + vy * ux)
            if perp > tolerance:
                continue
            if proj < -tolerance or proj > edge_len + tolerance:
                continue

            start = proj - width / 2.0
            end = proj + width / 2.0
            if edge_len > 2 * WallGenerator.EDGE_END_MARGIN:
                start = max(start, WallGenerator.EDGE_END_MARGIN)
                end = min(end, edge_len - WallGenerator.EDGE_END_MARGIN)
            if end <= 0.0 or start >= edge_len:
                continue

            edge_openings.append(
                {
                    "type": str(opening.get("type", "window")),
                    "start": max(0.0, start),
                    "end": min(edge_len, end),
                    "height": max(0.0, height),
                    "sill": max(0.0, sill),
                    "center_t": max(0.0, min(edge_len, proj)),
                }
            )

        return edge_openings

    def _create_opening_segments(
        self,
        stage,
        root_prim_path: str,
        edge_index: int,
        opening_index: int,
        edge_start: Tuple[float, float],
        edge_unit: Tuple[float, float],
        angle_deg: float,
        opening: Dict[str, object],
        thickness: float,
        wall_height: float,
        wall_paths: List[str],
        window_paths: List[str],
    ) -> None:
        start = opening["start"]
        end = opening["end"]
        open_height = opening["height"]
        sill = opening["sill"]
        opening_type = opening.get("type", "window")

        if start >= end:
            return

        # Door: open from floor to height
        if opening_type == "door":
            if open_height < wall_height - 0.05:
                top_height = wall_height - open_height
                top_center = open_height + top_height / 2.0
                name = f"Wall_Edge{edge_index}_DoorTop{opening_index}"
                path = self._create_oriented_wall_segment(
                    stage,
                    root_prim_path,
                    name,
                    edge_start,
                    edge_unit,
                    angle_deg,
                    start,
                    end,
                    thickness,
                    top_height,
                    top_center,
                )
                if path:
                    wall_paths.append(path)
            return

        # Window: create bottom and top segments
        if sill > 0.05:
            name = f"Wall_Edge{edge_index}_WindowBottom{opening_index}"
            path = self._create_oriented_wall_segment(
                stage,
                root_prim_path,
                name,
                edge_start,
                edge_unit,
                angle_deg,
                start,
                end,
                thickness,
                sill,
                sill / 2.0,
            )
            if path:
                wall_paths.append(path)

        top_start = sill + open_height
        if top_start < wall_height - 0.05:
            top_height = wall_height - top_start
            name = f"Wall_Edge{edge_index}_WindowTop{opening_index}"
            path = self._create_oriented_wall_segment(
                stage,
                root_prim_path,
                name,
                edge_start,
                edge_unit,
                angle_deg,
                start,
                end,
                thickness,
                top_height,
                top_start + top_height / 2.0,
            )
            if path:
                wall_paths.append(path)

        glass_path = self._create_window_glass(
            stage,
            root_prim_path,
            f"Window_Glass_Edge{edge_index}_{opening_index}",
            edge_start,
            edge_unit,
            angle_deg,
            start,
            end,
            max(0.05, open_height),
            max(0.0, sill),
            thickness,
        )
        if glass_path:
            window_paths.append(glass_path)

    def _create_oriented_wall_segment(
        self,
        stage,
        root_prim_path: str,
        name: str,
        edge_start: Tuple[float, float],
        edge_unit: Tuple[float, float],
        angle_deg: float,
        start: float,
        end: float,
        thickness: float,
        height: float,
        z_center: float,
    ) -> Optional[str]:
        length = max(0.0, end - start)
        if length <= 0.05 or height <= 0.01:
            return None

        cx = edge_start[0] + edge_unit[0] * (start + length / 2.0)
        cy = edge_start[1] + edge_unit[1] * (start + length / 2.0)
        path = self._get_unique_child_path(stage, root_prim_path, name)
        cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
        cube.GetSizeAttr().Set(1.0)

        xformable = UsdGeom.Xformable(cube)
        xformable.AddTranslateOp().Set(Gf.Vec3d(cx, cy, z_center))
        xformable.AddRotateZOp().Set(angle_deg)
        xformable.AddScaleOp().Set(Gf.Vec3f(length, thickness, height))
        return path

    def _create_window_glass(
        self,
        stage,
        root_prim_path: str,
        name: str,
        edge_start: Tuple[float, float],
        edge_unit: Tuple[float, float],
        angle_deg: float,
        start: float,
        end: float,
        height: float,
        sill: float,
        wall_thickness: float,
    ) -> Optional[str]:
        length = max(0.0, end - start)
        if length <= 0.05 or height <= 0.05:
            return None

        cx = edge_start[0] + edge_unit[0] * (start + length / 2.0)
        cy = edge_start[1] + edge_unit[1] * (start + length / 2.0)
        z_center = sill + height / 2.0
        glass_thickness = min(0.02, wall_thickness * 0.5)

        path = self._get_unique_child_path(stage, root_prim_path, name)
        cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
        cube.GetSizeAttr().Set(1.0)
        cube.CreateDisplayColorAttr().Set([(0.6, 0.8, 1.0)])
        cube.CreateDisplayOpacityAttr().Set([0.2])

        xformable = UsdGeom.Xformable(cube)
        xformable.AddTranslateOp().Set(Gf.Vec3d(cx, cy, z_center))
        xformable.AddRotateZOp().Set(angle_deg)
        xformable.AddScaleOp().Set(Gf.Vec3f(length, glass_thickness, height))
        try:
            material = self._get_or_create_glass_material(stage, root_prim_path)
            if material:
                UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)
        except Exception as exc:  # pragma: no cover - defensive
            omni.log.warn(f"Failed to bind glass material: {exc}")
        return path

    def _get_or_create_glass_material(self, stage, root_prim_path: str) -> Optional[UsdShade.Material]:
        if not hasattr(self, "_glass_material_cache"):
            self._glass_material_cache = {}
        cache = self._glass_material_cache
        if root_prim_path in cache:
            return cache[root_prim_path]

        materials_path = f"{root_prim_path}/Materials"
        if not stage.GetPrimAtPath(materials_path):
            UsdGeom.Xform.Define(stage, Sdf.Path(materials_path))

        material_path = f"{materials_path}/Glass"
        try:
            material = UsdShade.Material.Define(stage, Sdf.Path(material_path))
            shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{material_path}/Shader"))
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.7, 0.9, 1.0))
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.2)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.05)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5)
            surface_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(surface_output)
        except Exception as exc:
            omni.log.warn(f"Failed to create glass material: {exc}")
            return None

        cache[root_prim_path] = material
        return material

    def _create_single_wall(
        self,
        stage,
        root_prim_path: str,
        wall_name: str,
        x: float,
        y: float,
        size_x: float,
        size_y: float,
        wall_height: float,
        wall_thickness: float,
        custom_z_center: float = None,
        custom_height: float = None,
    ) -> Optional[str]:
        """
        単一の壁を生成します。

        Args:
            stage: USDステージ
            root_prim_path: 親Primのパス
            wall_name: 壁の名前
            x: X位置（メートル単位）
            y: Y位置（メートル単位）
            size_x: Xサイズ（メートル単位）
            size_y: Yサイズ（メートル単位）
            wall_height: 壁の高さ（メートル単位）
            wall_thickness: 壁の厚み（メートル単位）
            custom_z_center: カスタムZ座標の中心（Noneの場合はデフォルト計算）
            custom_height: カスタム高さ（Noneの場合はwall_heightを使用）

        Returns:
            生成された壁PrimのパスまたはNone
        """
        try:
            # 実際に使用する高さを決定
            actual_height = custom_height if custom_height is not None else wall_height

            # Z座標を決定（カスタム値がある場合はそれを使用、なければデフォルト計算）
            if custom_z_center is not None:
                wall_z = custom_z_center
            else:
                # 壁の底面がZ=0になるように配置（中心がwall_height/2）
                wall_z = actual_height / 2.0

            # 壁のPrimパスを生成
            wall_path = self._get_unique_child_path(stage, root_prim_path, wall_name)

            # Cubeとして壁を定義
            cube = UsdGeom.Cube.Define(stage, Sdf.Path(wall_path))

            # Cubeのサイズ属性を設定（1.0メートル単位のキューブのスケールとして）
            cube.GetSizeAttr().Set(1.0)

            # Xformableとしてトランスフォームを適用
            xformable = UsdGeom.Xformable(cube)

            # トランスフォームの適用順序: Translate -> Scale
            translate = Gf.Vec3d(x, y, wall_z)
            scale = Gf.Vec3f(size_x, size_y, actual_height)

            xformable.AddTranslateOp().Set(translate)
            xformable.AddScaleOp().Set(scale)

            omni.log.info(
                f"壁を生成しました: {wall_path} (位置: X={x}, Y={y}, Z={wall_z}, スケール: {size_x}x{size_y}x{actual_height})"
            )
            return wall_path

        except Exception as exc:
            omni.log.error(f"壁の生成に失敗しました: {exc}")
            return None

    def _create_wall_segments_with_door_cutouts(
        self,
        stage,
        root_prim_path: str,
        wall_name: str,
        wall_x: float,
        wall_z: float,
        wall_size_x: float,
        wall_size_z: float,
        wall_direction: str,
        doors_on_wall: List[Dict[str, object]],
    ) -> List[str]:
        """
        ドアの開口部を持つ壁をセグメントとして生成します。

        Args:
            stage: USDステージ
            root_prim_path: 親Primのパス
            wall_name: 壁の名前
            wall_x: 壁のX位置
            wall_z: 壁のZ位置
            wall_size_x: 壁のXサイズ
            wall_size_z: 壁のZサイズ
            wall_direction: 壁の方向
            doors_on_wall: この壁上にあるドアのリスト

        Returns:
            生成された壁セグメントのパスのリスト
        """
        wall_segments = []

        # 今のところ、1つのドアのみをサポート
        if len(doors_on_wall) > 1:
            omni.log.warn(f"{wall_name} に複数のドアがあります。最初のドアのみを考慮します。")

        door = doors_on_wall[0]
        door_name = door.get("object_name", "Door")

        # ドアの位置と寸法を取得
        door_x = self._extract_float(door, "X", 0.0)
        door_y = self._extract_float(door, "Y", 0.0)
        door_height = self._extract_float(door, "Height", 2.1)
        door_height = max(0.0, min(door_height, self.DEFAULT_WALL_HEIGHT))

        # ドアの開口幅を計算（壁の方向とドアの回転を考慮）
        door_opening_width = self._door_detector.calculate_door_opening_width(
            door, wall_direction
        )

        if door_opening_width <= 0.0:
            omni.log.warn(f"{wall_name}: invalid door opening width={door_opening_width}. Fallback to single wall.")
            fallback_path = self._create_single_wall(
                stage,
                root_prim_path,
                wall_name,
                wall_x,
                wall_z,
                wall_size_x,
                wall_size_z,
                self.DEFAULT_WALL_HEIGHT,
                self.DEFAULT_WALL_THICKNESS,
            )
            return [fallback_path] if fallback_path else []

        omni.log.info(
            f"ドア開口部を作成: {door_name}, 開口幅={door_opening_width}, 高さ={door_height}"
        )

        # 壁の方向に応じて、左右のセグメントと上部セグメントを生成
        if wall_direction in ["north", "south"]:
            # X方向に沿った壁（北/南）
            # 壁の左端と右端
            wall_left_edge = wall_x - wall_size_x / 2.0
            wall_right_edge = wall_x + wall_size_x / 2.0

            # ドアの左端と右端
            door_left_edge = door_x - door_opening_width / 2.0
            door_right_edge = door_x + door_opening_width / 2.0
            door_left_edge = max(door_left_edge, wall_left_edge)
            door_right_edge = min(door_right_edge, wall_right_edge)
            door_opening_width = max(0.0, door_right_edge - door_left_edge)

            # 左側のセグメント
            left_segment_width = door_left_edge - wall_left_edge
            if left_segment_width > 0.1:  # 最小幅のチェック
                left_segment_x = wall_left_edge + left_segment_width / 2.0
                left_segment_name = f"{wall_name}_Left"
                left_segment_path = self._create_single_wall(
                    stage,
                    root_prim_path,
                    left_segment_name,
                    left_segment_x,
                    wall_z,
                    left_segment_width,
                    wall_size_z,
                    self.DEFAULT_WALL_HEIGHT,
                    self.DEFAULT_WALL_THICKNESS,
                )
                if left_segment_path:
                    wall_segments.append(left_segment_path)

            # 上部のセグメント（ドアの高さから天井まで）
            if door_height < self.DEFAULT_WALL_HEIGHT:
                top_segment_height = self.DEFAULT_WALL_HEIGHT - door_height
                top_z_center = (door_height + self.DEFAULT_WALL_HEIGHT) / 2.0
                top_segment_name = f"{wall_name}_Top"
                top_segment_x = door_x
                top_segment_path = self._create_single_wall(
                    stage,
                    root_prim_path,
                    top_segment_name,
                    top_segment_x,
                    wall_z,
                    door_opening_width,
                    wall_size_z,
                    self.DEFAULT_WALL_HEIGHT,
                    self.DEFAULT_WALL_THICKNESS,
                    custom_z_center=top_z_center,
                    custom_height=top_segment_height,
                )
                if top_segment_path:
                    wall_segments.append(top_segment_path)

            # 右側のセグメント
            right_segment_width = wall_right_edge - door_right_edge
            if right_segment_width > 0.1:  # 最小幅のチェック
                right_segment_x = door_right_edge + right_segment_width / 2.0
                right_segment_name = f"{wall_name}_Right"
                right_segment_path = self._create_single_wall(
                    stage,
                    root_prim_path,
                    right_segment_name,
                    right_segment_x,
                    wall_z,
                    right_segment_width,
                    wall_size_z,
                    self.DEFAULT_WALL_HEIGHT,
                    self.DEFAULT_WALL_THICKNESS,
                )
                if right_segment_path:
                    wall_segments.append(right_segment_path)

        else:
            # Z方向に沿った壁（東/西）
            # 壁の前端と後端
            wall_front_edge = wall_z - wall_size_z / 2.0
            wall_back_edge = wall_z + wall_size_z / 2.0

            # ドアの前端と後端
            door_front_edge = door_y - door_opening_width / 2.0
            door_back_edge = door_y + door_opening_width / 2.0
            door_front_edge = max(door_front_edge, wall_front_edge)
            door_back_edge = min(door_back_edge, wall_back_edge)
            door_opening_width = max(0.0, door_back_edge - door_front_edge)

            # 前側のセグメント
            front_segment_width = door_front_edge - wall_front_edge
            if front_segment_width > 0.1:  # 最小幅のチェック
                front_segment_z = wall_front_edge + front_segment_width / 2.0
                front_segment_name = f"{wall_name}_Front"
                front_segment_path = self._create_single_wall(
                    stage,
                    root_prim_path,
                    front_segment_name,
                    wall_x,
                    front_segment_z,
                    wall_size_x,
                    front_segment_width,
                    self.DEFAULT_WALL_HEIGHT,
                    self.DEFAULT_WALL_THICKNESS,
                )
                if front_segment_path:
                    wall_segments.append(front_segment_path)

            # 上部のセグメント（ドアの高さから天井まで）
            if door_height < self.DEFAULT_WALL_HEIGHT:
                top_segment_height = self.DEFAULT_WALL_HEIGHT - door_height
                top_z_center = (door_height + self.DEFAULT_WALL_HEIGHT) / 2.0
                top_segment_name = f"{wall_name}_Top"
                top_segment_z = door_y
                top_segment_path = self._create_single_wall(
                    stage,
                    root_prim_path,
                    top_segment_name,
                    wall_x,
                    top_segment_z,
                    wall_size_x,
                    door_opening_width,
                    self.DEFAULT_WALL_HEIGHT,
                    self.DEFAULT_WALL_THICKNESS,
                    custom_z_center=top_z_center,
                    custom_height=top_segment_height,
                )
                if top_segment_path:
                    wall_segments.append(top_segment_path)

            # 後側のセグメント
            back_segment_width = wall_back_edge - door_back_edge
            if back_segment_width > 0.1:  # 最小幅のチェック
                back_segment_z = door_back_edge + back_segment_width / 2.0
                back_segment_name = f"{wall_name}_Back"
                back_segment_path = self._create_single_wall(
                    stage,
                    root_prim_path,
                    back_segment_name,
                    wall_x,
                    back_segment_z,
                    wall_size_x,
                    back_segment_width,
                    self.DEFAULT_WALL_HEIGHT,
                    self.DEFAULT_WALL_THICKNESS,
                )
                if back_segment_path:
                    wall_segments.append(back_segment_path)

        omni.log.info(
            f"{wall_name} を {len(wall_segments)} 個のセグメントに分割しました。"
        )

        if not wall_segments:
            omni.log.warn(f"{wall_name}: no wall segments were created. Fallback to single wall.")
            fallback_path = self._create_single_wall(
                stage,
                root_prim_path,
                wall_name,
                wall_x,
                wall_z,
                wall_size_x,
                wall_size_z,
                self.DEFAULT_WALL_HEIGHT,
                self.DEFAULT_WALL_THICKNESS,
            )
            return [fallback_path] if fallback_path else []

        return wall_segments
