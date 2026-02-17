# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""
ドア検出モジュール

壁上のドアを検出し、開口部の計算を行います。
"""

from typing import Dict, List
import omni.log


class DoorDetector:
    """壁上のドアを検出するクラス"""

    # ドア検出の許容誤差（メートル単位）
    DETECTION_TOLERANCE = 0.1  # 0.1m = 10cm

    def __init__(self, extract_float_func, extract_optional_float_func):
        """
        Args:
            extract_float_func: JSONから浮動小数点値を抽出する関数
            extract_optional_float_func: JSONからオプショナルな浮動小数点値を抽出する関数
        """
        self._extract_float = extract_float_func
        self._extract_optional_float = extract_optional_float_func

    def find_doors_on_wall(
        self,
        door_objects: List[Dict[str, object]],
        wall_direction: str,
        area_size_x: float,
        area_size_y: float,
        origin_mode: str = "center",
    ) -> List[Dict[str, object]]:
        """
        指定された壁の方向にあるドアを検出します。

        Args:
            door_objects: ドアオブジェクトのリスト
            wall_direction: 壁の方向 ("north", "south", "east", "west")
            area_size_x: 部屋のX方向のサイズ（メートル単位）
            area_size_y: 部屋のY方向のサイズ（メートル単位）
            origin_mode: "center" or "bottom_left"

        Returns:
            この壁上にあるドアのリスト
        """
        doors_on_wall = []

        omni.log.info(f"=== ドア検出: {wall_direction}側の壁 ===")
        omni.log.info(f"  部屋サイズ: X={area_size_x}, Y={area_size_y}")
        omni.log.info(f"  ドア候補数: {len(door_objects)}")

        for door in door_objects:
            door_name = door.get("object_name", "Unknown")
            door_x = self._extract_float(door, "X", 0.0)
            door_y = self._extract_float(door, "Y", 0.0)

            omni.log.info(f"  ドア '{door_name}': X={door_x}, Y={door_y}")

            # 壁の方向に応じて、ドアが壁上にあるかチェック
            if wall_direction == "north":
                # 北側の壁
                wall_pos = area_size_y if origin_mode == "bottom_left" else area_size_y / 2.0
                distance = abs(door_y - wall_pos)
                omni.log.info(f"    北側壁位置: Y={wall_pos}, 距離: {distance}")
                if distance < self.DETECTION_TOLERANCE:
                    doors_on_wall.append(door)
                    omni.log.info(f"    -> 検出成功！")
            elif wall_direction == "south":
                # 南側の壁
                wall_pos = 0.0 if origin_mode == "bottom_left" else -area_size_y / 2.0
                distance = abs(door_y - wall_pos)
                omni.log.info(f"    南側壁位置: Y={wall_pos}, 距離: {distance}")
                if distance < self.DETECTION_TOLERANCE:
                    doors_on_wall.append(door)
                    omni.log.info(f"    -> 検出成功！")
            elif wall_direction == "east":
                # 東側の壁
                wall_pos = area_size_x if origin_mode == "bottom_left" else area_size_x / 2.0
                distance = abs(door_x - wall_pos)
                omni.log.info(f"    東側壁位置: X={wall_pos}, 距離: {distance}")
                if distance < self.DETECTION_TOLERANCE:
                    doors_on_wall.append(door)
                    omni.log.info(f"    -> 検出成功！")
            elif wall_direction == "west":
                # 西側の壁
                wall_pos = 0.0 if origin_mode == "bottom_left" else -area_size_x / 2.0
                distance = abs(door_x - wall_pos)
                omni.log.info(f"    西側壁位置: X={wall_pos}, 距離: {distance}")
                if distance < self.DETECTION_TOLERANCE:
                    doors_on_wall.append(door)
                    omni.log.info(f"    -> 検出成功！")

        if doors_on_wall:
            omni.log.info(f"=== {wall_direction}側の壁に {len(doors_on_wall)} 個のドアを検出しました ===")

        return doors_on_wall

    def calculate_door_opening_width(
        self,
        door: Dict[str, object],
        wall_direction: str,
    ) -> float:
        """
        ドアの開口幅を計算します。
        壁の方向とドアの回転を考慮して、正しい寸法を返します。

        Args:
            door: ドアオブジェクト
            wall_direction: 壁の方向 ("north", "south", "east", "west")

        Returns:
            開口幅（メートル単位）
        """
        door_length = self._extract_float(door, "Length", 1.0)
        door_width = self._extract_float(door, "Width", 1.0)
        rotation = self._extract_optional_float(door, "rotationZ")

        # 壁の方向に応じて、ドアの位置を壁に沿った座標に変換
        if wall_direction in ["north", "south"]:
            # X方向に沿った壁（北/南）
            # ドアの回転に応じて、壁に沿った方向のサイズを決定
            # rotationZ=0: Length=壁に沿った幅, Width=壁に垂直な厚み
            # rotationZ=90: Width=壁に沿った幅, Length=壁に垂直な厚み
            if rotation is not None and (abs(rotation - 90) < 45 or abs(rotation - 270) < 45):
                # 90度回転: Widthが壁に沿った方向
                door_opening_width = door_width
            else:
                # 0度: Lengthが壁に沿った方向
                door_opening_width = door_length
        else:
            # Y方向に沿った壁（東/西）
            # ドアの回転に応じて、壁に沿った方向のサイズを決定
            if rotation is not None and (abs(rotation - 90) < 45 or abs(rotation - 270) < 45):
                # 90度回転: Lengthが壁に沿った方向
                door_opening_width = door_length
            else:
                # 0度: Widthが壁に沿った方向
                door_opening_width = door_width

        omni.log.info(f"  ドア開口部計算: 方向={wall_direction}, 回転={rotation}, 開口幅={door_opening_width}")

        return door_opening_width
