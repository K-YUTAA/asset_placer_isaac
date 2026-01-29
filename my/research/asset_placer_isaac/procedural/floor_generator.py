# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""
床生成モジュール

JSONレイアウトデータから手続き的に床を生成します。
"""

from typing import Dict, List, Optional, Tuple
from pxr import Gf, Sdf, UsdGeom
import omni.log


class FloorGenerator:
    """床を手続き的に生成するクラス"""

    # 床の厚み（メートル単位）
    FLOOR_THICKNESS = 0.1  # 0.1m = 10cm

    def __init__(self, extract_float_func, get_unique_path_func):
        """
        Args:
            extract_float_func: JSONから浮動小数点値を抽出する関数
            get_unique_path_func: 一意なPrimパスを生成する関数
        """
        self._extract_float = extract_float_func
        self._get_unique_child_path = get_unique_path_func

    def generate(
        self,
        stage,
        root_prim_path: str,
        floor_data: Dict[str, object],
    ) -> Optional[str]:
        """
        JSONデータから床を手続き的に生成します。

        Args:
            stage: USDステージ
            root_prim_path: 親Primのパス
            floor_data: JSONの床オブジェクトデータ（Length, Width, X, Y を含む）

        Returns:
            生成された床PrimのパスまたはNone
        """
        try:
            # JSONから寸法と位置を取得（メートル単位）
            length = self._extract_float(floor_data, "Length", 10.0)
            width = self._extract_float(floor_data, "Width", 10.0)
            x = self._extract_float(floor_data, "X", 0.0)
            y = self._extract_float(floor_data, "Y", 0.0)

            # 床のPrimパスを生成
            floor_path = self._get_unique_child_path(stage, root_prim_path, "Floor")

            # Cubeとして床を定義
            cube = UsdGeom.Cube.Define(stage, Sdf.Path(floor_path))

            # Cubeのサイズ属性を設定（1.0メートル単位のキューブのスケールとして）
            cube.GetSizeAttr().Set(1.0)

            # Xformableとしてトランスフォームを適用
            xformable = UsdGeom.Xformable(cube)

            # Z座標: 床の上面がZ=0になるように配置（中心が-floor_thickness/2）
            floor_z = -self.FLOOR_THICKNESS / 2.0

            # トランスフォームの適用順序: Translate -> Scale
            translate = Gf.Vec3d(x, y, floor_z)
            scale = Gf.Vec3f(length, width, self.FLOOR_THICKNESS)

            xformable.AddTranslateOp().Set(translate)
            xformable.AddScaleOp().Set(scale)

            omni.log.info(f"床を生成しました: {floor_path} (スケール: {length}x{width}, 位置: X={x}, Y={y})")
            return floor_path

        except Exception as exc:
            omni.log.error(f"床の生成に失敗しました: {exc}")
            return None

    def generate_polygon(
        self,
        stage,
        root_prim_path: str,
        polygon_points: List[object],
    ) -> Optional[str]:
        """
        ポリゴンの外周点列から床メッシュを生成する。
        pointsは順序付きのXY点列（辞書 or (x, y)）。
        """
        points = self._normalize_polygon_points(polygon_points)
        if len(points) < 3:
            omni.log.warn("Floor polygon has fewer than 3 valid points.")
            return None

        try:
            floor_path = self._get_unique_child_path(stage, root_prim_path, "Floor")
            mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(floor_path))

            mesh_points = [Gf.Vec3f(x, y, 0.0) for x, y in points]
            mesh.CreatePointsAttr(mesh_points)

            face_counts = []
            face_indices = []
            for i in range(1, len(points) - 1):
                face_counts.append(3)
                face_indices.extend([0, i, i + 1])

            mesh.CreateFaceVertexCountsAttr(face_counts)
            mesh.CreateFaceVertexIndicesAttr(face_indices)
            mesh.CreateDoubleSidedAttr(True)

            min_x = min(p[0] for p in points)
            min_y = min(p[1] for p in points)
            max_x = max(p[0] for p in points)
            max_y = max(p[1] for p in points)
            mesh.CreateExtentAttr([Gf.Vec3f(min_x, min_y, 0.0), Gf.Vec3f(max_x, max_y, 0.0)])

            omni.log.info(f"Polygon floor generated: {floor_path} (points={len(points)})")
            return floor_path
        except Exception as exc:
            omni.log.error(f"Failed to generate polygon floor: {exc}")
            return None

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
