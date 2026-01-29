# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
床と壁の手続き的生成のテスト

このテストは、以下の機能を検証します：
1. _create_procedural_floor が正しく床を生成する
2. _create_procedural_walls が正しく4つの壁を生成する
3. 床の上面がY=0に配置される
4. 壁の底面がY=0に配置される
"""

import omni.kit.test
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom


class TestProceduralGeneration(omni.kit.test.AsyncTestCase):
    """床と壁の手続き的生成のテストクラス"""

    async def setUp(self):
        """各テストの前に実行される準備処理"""
        # 新しいステージを作成
        await omni.usd.get_context().new_stage_async()
        self.stage = omni.usd.get_context().get_stage()
        self.assertIsNotNone(self.stage)

    async def tearDown(self):
        """各テストの後に実行されるクリーンアップ処理"""
        await omni.usd.get_context().close_stage_async()

    async def test_create_procedural_floor(self):
        """
        _create_procedural_floorが床を正しく生成することをテスト

        期待される動作:
        1. JSONデータから床のサイズと位置を読み取る
        2. 厚み0.1mの床を生成する
        3. 床の上面がY=0に配置される（中心がY=-0.05）
        """
        # Extensionクラスをインポート
        from my.research.asset_placer_isaac.extension import MyExtension
        extension = MyExtension()

        # ルートXformを作成
        root_path = "/TestRoot"
        UsdGeom.Xform.Define(self.stage, Sdf.Path(root_path))

        # JSONデータをシミュレート（メートル単位）
        floor_data = {
            "object_name": "Floor",
            "Length": 10.0,  # 10m
            "Width": 8.0,    # 8m
            "X": 0.0,        # 0m
            "Z": 0.0,        # 0m
        }

        # 床を生成
        floor_path = extension._create_procedural_floor(self.stage, root_path, floor_data)

        # 床が生成されたことを確認
        self.assertIsNotNone(floor_path, "Floor should be generated")
        self.assertTrue(self.stage.GetPrimAtPath(floor_path), f"Floor prim should exist at {floor_path}")

        # 床のプロパティを確認
        floor_prim = self.stage.GetPrimAtPath(floor_path)
        xformable = UsdGeom.Xformable(floor_prim)
        ops = xformable.GetOrderedXformOps()

        # 2つの操作があることを確認: Translate, Scale
        self.assertEqual(len(ops), 2, "Should have exactly 2 xform ops: Translate, Scale")

        # Translate値を確認
        translate_op = ops[0]
        translate_value = translate_op.Get()
        self.assertAlmostEqual(translate_value[0], 0.0, delta=0.01, msg="Floor X should be 0.0")
        self.assertAlmostEqual(translate_value[1], -0.05, delta=0.01, msg="Floor Y should be -0.05 (top at Y=0)")
        self.assertAlmostEqual(translate_value[2], 0.0, delta=0.01, msg="Floor Z should be 0.0")

        # Scale値を確認
        scale_op = ops[1]
        scale_value = scale_op.Get()
        self.assertAlmostEqual(scale_value[0], 10.0, delta=0.01, msg="Floor length should be 10.0m")
        self.assertAlmostEqual(scale_value[1], 0.1, delta=0.01, msg="Floor thickness should be 0.1m")
        self.assertAlmostEqual(scale_value[2], 8.0, delta=0.01, msg="Floor width should be 8.0m")

    async def test_create_procedural_walls(self):
        """
        _create_procedural_wallsが4つの壁を正しく生成することをテスト

        期待される動作:
        1. 部屋のサイズに基づいて4つの壁を生成する
        2. 各壁の底面がY=0に配置される（中心がY=1.25）
        3. 壁の高さが2.5m、厚みが0.15mである
        """
        # Extensionクラスをインポート
        from my.research.asset_placer_isaac.extension import MyExtension
        extension = MyExtension()

        # ルートXformを作成
        root_path = "/TestRoot"
        UsdGeom.Xform.Define(self.stage, Sdf.Path(root_path))

        # 部屋のサイズ（メートル単位）
        area_size_x = 10.0
        area_size_z = 8.0

        # 壁を生成
        wall_paths = extension._create_procedural_walls(
            self.stage, root_path, area_size_x, area_size_z
        )

        # 4つの壁が生成されたことを確認
        self.assertEqual(len(wall_paths), 4, "Should generate exactly 4 walls")

        # すべての壁が存在することを確認
        for wall_path in wall_paths:
            self.assertTrue(self.stage.GetPrimAtPath(wall_path), f"Wall prim should exist at {wall_path}")

        # 各壁のプロパティを確認
        wall_configs = [
            ("Wall_North", 0.0, area_size_z / 2.0, area_size_x + 0.15, 0.15),
            ("Wall_South", 0.0, -area_size_z / 2.0, area_size_x + 0.15, 0.15),
            ("Wall_East", area_size_x / 2.0, 0.0, 0.15, area_size_z),
            ("Wall_West", -area_size_x / 2.0, 0.0, 0.15, area_size_z),
        ]

        for i, (expected_name, expected_x, expected_z, expected_size_x, expected_size_z) in enumerate(wall_configs):
            wall_path = wall_paths[i]
            wall_prim = self.stage.GetPrimAtPath(wall_path)

            # 壁の名前を確認
            self.assertIn(expected_name, wall_path, f"Wall path should contain '{expected_name}'")

            # トランスフォームを確認
            xformable = UsdGeom.Xformable(wall_prim)
            ops = xformable.GetOrderedXformOps()

            # 2つの操作があることを確認: Translate, Scale
            self.assertEqual(len(ops), 2, f"{expected_name} should have exactly 2 xform ops")

            # Translate値を確認
            translate_op = ops[0]
            translate_value = translate_op.Get()
            self.assertAlmostEqual(translate_value[0], expected_x, delta=0.01,
                                   msg=f"{expected_name} X position should be {expected_x}")
            self.assertAlmostEqual(translate_value[1], 1.25, delta=0.01,
                                   msg=f"{expected_name} Y position should be 1.25 (bottom at Y=0)")
            self.assertAlmostEqual(translate_value[2], expected_z, delta=0.01,
                                   msg=f"{expected_name} Z position should be {expected_z}")

            # Scale値を確認
            scale_op = ops[1]
            scale_value = scale_op.Get()
            self.assertAlmostEqual(scale_value[0], expected_size_x, delta=0.01,
                                   msg=f"{expected_name} X size should be {expected_size_x}")
            self.assertAlmostEqual(scale_value[1], 2.5, delta=0.01,
                                   msg=f"{expected_name} height should be 2.5m")
            self.assertAlmostEqual(scale_value[2], expected_size_z, delta=0.01,
                                   msg=f"{expected_name} Z size should be {expected_size_z}")

    async def test_floor_and_walls_integration(self):
        """
        床と壁の統合テスト

        期待される動作:
        1. 床と壁が同じルートPrimの下に生成される
        2. 床の上面と壁の底面が同じY=0に配置される
        """
        # Extensionクラスをインポート
        from my.research.asset_placer_isaac.extension import MyExtension
        extension = MyExtension()

        # ルートXformを作成
        root_path = "/TestRoom"
        UsdGeom.Xform.Define(self.stage, Sdf.Path(root_path))

        # 床データ
        floor_data = {
            "object_name": "Floor",
            "Length": 10.0,
            "Width": 8.0,
            "X": 0.0,
            "Z": 0.0,
        }

        # 床を生成
        floor_path = extension._create_procedural_floor(self.stage, root_path, floor_data)
        self.assertIsNotNone(floor_path, "Floor should be generated")

        # 壁を生成
        wall_paths = extension._create_procedural_walls(self.stage, root_path, 10.0, 8.0)
        self.assertEqual(len(wall_paths), 4, "Should generate 4 walls")

        # すべてのプリムが同じルートの下にあることを確認
        self.assertTrue(floor_path.startswith(root_path), "Floor should be under root prim")
        for wall_path in wall_paths:
            self.assertTrue(wall_path.startswith(root_path), "Wall should be under root prim")

        # 床の上面のY座標を取得
        floor_prim = self.stage.GetPrimAtPath(floor_path)
        floor_xformable = UsdGeom.Xformable(floor_prim)
        floor_ops = floor_xformable.GetOrderedXformOps()
        floor_translate = floor_ops[0].Get()
        floor_scale = floor_ops[1].Get()
        floor_top_y = floor_translate[1] + floor_scale[1] / 2.0

        # 壁の底面のY座標を取得
        wall_prim = self.stage.GetPrimAtPath(wall_paths[0])
        wall_xformable = UsdGeom.Xformable(wall_prim)
        wall_ops = wall_xformable.GetOrderedXformOps()
        wall_translate = wall_ops[0].Get()
        wall_scale = wall_ops[1].Get()
        wall_bottom_y = wall_translate[1] - wall_scale[1] / 2.0

        # 床の上面と壁の底面が同じY=0に配置されていることを確認
        self.assertAlmostEqual(floor_top_y, 0.0, delta=0.01,
                               msg="Floor top should be at Y=0")
        self.assertAlmostEqual(wall_bottom_y, 0.0, delta=0.01,
                               msg="Wall bottom should be at Y=0")
