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
絶対寸法スケーリングのテスト

このテストは、_apply_transform関数が以下の要件を満たすことを検証します：
1. JSONの Length, Width, Height（メートル単位）を読み取る
2. アセットの元のバウンディングボックスサイズを計算する
3. 目標サイズ ÷ 元のサイズ でスケール比を計算する
4. スケール、回転、平行移動の順でトランスフォームを適用する
"""

import omni.kit.test
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom


class TestAbsoluteScaling(omni.kit.test.AsyncTestCase):
    """絶対寸法スケーリングのテストクラス"""

    async def setUp(self):
        """各テストの前に実行される準備処理"""
        # 新しいステージを作成
        await omni.usd.get_context().new_stage_async()
        self.stage = omni.usd.get_context().get_stage()
        self.assertIsNotNone(self.stage)

    async def tearDown(self):
        """各テストの後に実行されるクリーンアップ処理"""
        await omni.usd.get_context().close_stage_async()

    async def test_apply_transform_with_absolute_scaling(self):
        """
        _apply_transformが絶対寸法スケーリングを正しく適用することをテスト

        期待される動作:
        1. 元のバウンディングボックスが (1, 1, 1) mのキューブを作成
        2. JSONデータで Length=2.0m, Height=1.5m, Width=1.0m を指定
        3. 最終的なバウンディングボックスが (2.0, 1.5, 1.0) m になることを確認
        """
        # テスト用のキューブプリミティブを作成（元のサイズ: 1m）
        cube_path = "/TestCube"
        cube_geom = UsdGeom.Cube.Define(self.stage, Sdf.Path(cube_path))
        cube_geom.GetSizeAttr().Set(1.0)  # 1メートルのキューブ

        # キューブのバウンディングボックスを確認（元のサイズ）
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
        original_bbox = bbox_cache.ComputeWorldBound(cube_geom.GetPrim())
        original_range = original_bbox.ComputeAlignedRange()
        original_size = original_range.GetMax() - original_range.GetMin()

        # 元のサイズが約 (1, 1, 1) であることを確認
        self.assertAlmostEqual(original_size[0], 1.0, delta=0.01)
        self.assertAlmostEqual(original_size[1], 1.0, delta=0.01)
        self.assertAlmostEqual(original_size[2], 1.0, delta=0.01)

        # JSONデータをシミュレート（メートル単位）
        # Length=2.0m (X軸), Height=1.5m (Y軸), Width=1.0m (Z軸)
        # Y値は無視され、すべてのモデルの底面がY=0に配置される
        layout_data = {
            "Length": 2.0,     # 2.0 m
            "Height": 1.5,     # 1.5 m
            "Width": 1.0,      # 1.0 m
            "X": 5.0,          # 5.0 m
            "Y": 999.0,        # この値は無視される
            "Z": 3.0,          # 3.0 m
            "rotationY": 45.0  # degrees
        }

        # Extensionクラスをインポート
        from my.research.asset_placer_isaac.extension import MyExtension
        extension = MyExtension()

        # _apply_transformを呼び出し（非同期）
        prim = self.stage.GetPrimAtPath(cube_path)
        await extension._apply_transform(prim, layout_data)

        # トランスフォームが適用された後のバウンディングボックスを計算
        bbox_cache_after = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
        transformed_bbox = bbox_cache_after.ComputeWorldBound(prim)
        transformed_range = transformed_bbox.ComputeAlignedRange()
        transformed_size = transformed_range.GetMax() - transformed_range.GetMin()

        # 期待される最終サイズ（メートル単位）
        expected_x = 2.0  # 2.0m
        expected_y = 1.5  # 1.5m
        expected_z = 1.0  # 1.0m

        # スケーリングが正しく適用されたことを確認
        # 回転が適用されているため、XZ平面のサイズは変わる可能性があるが、
        # Y軸（高さ）は回転の影響を受けないため、正確にチェック可能
        self.assertAlmostEqual(transformed_size[1], expected_y, delta=0.1,
                               msg=f"Height should be {expected_y}m, but got {transformed_size[1]}m")

        # XとZは回転の影響を受けるため、範囲チェック
        # 45度回転の場合、対角線が長くなる
        # 最小でも元のサイズ以上、最大で対角線長以下
        min_xz = min(expected_x, expected_z)
        max_xz = (expected_x**2 + expected_z**2)**0.5

        self.assertGreaterEqual(transformed_size[0], min_xz - 0.1,
                                msg=f"X size should be at least {min_xz}m")
        self.assertLessEqual(transformed_size[0], max_xz + 0.1,
                             msg=f"X size should be at most {max_xz}m")

        # Xformable opsの順序を確認
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()

        # 3つの操作があることを確認: Translate, RotateY, Scale
        self.assertEqual(len(ops), 3, "Should have exactly 3 xform ops: Translate, RotateY, Scale")

        # 操作の順序を確認（追加順序: Translate -> Rotate -> Scale）
        # USDは逆順に適用するため、実際の適用順序は Scale -> Rotate -> Translate
        op_types = [op.GetOpType() for op in ops]
        self.assertEqual(op_types[0], UsdGeom.XformOp.TypeTranslate, "First op should be Translate")
        self.assertEqual(op_types[1], UsdGeom.XformOp.TypeRotateY, "Second op should be RotateY")
        self.assertEqual(op_types[2], UsdGeom.XformOp.TypeScale, "Third op should be Scale")

        # 平行移動値を確認（メートル単位）
        # すべてのモデルの底面がY=0に配置される:
        # bbox_min_y = -0.5 (キューブの中心が原点)
        # y_offset = -0.5 * 1.5 = -0.75
        # final_translate_y = 0.0 - (-0.75) = 0.75
        translate_op = ops[0]
        translate_value = translate_op.Get()
        self.assertAlmostEqual(translate_value[0], 5.0, delta=0.01,
                               msg="Translate X should be 5.0m")
        self.assertAlmostEqual(translate_value[1], 0.75, delta=0.01,
                               msg="Translate Y should be 0.75m (bottom at Y=0)")
        self.assertAlmostEqual(translate_value[2], 3.0, delta=0.01,
                               msg="Translate Z should be 3.0m")

        # 回転値を確認
        rotate_op = ops[1]
        rotate_value = rotate_op.Get()
        self.assertAlmostEqual(rotate_value, 45.0, delta=0.01,
                               msg="RotateY should be 45 degrees")

        # スケール値を確認
        scale_op = ops[2]
        scale_value = scale_op.Get()
        expected_scale_x = 2.0  # 2.0m / 1.0m
        expected_scale_y = 1.5  # 1.5m / 1.0m
        expected_scale_z = 1.0  # 1.0m / 1.0m

        self.assertAlmostEqual(scale_value[0], expected_scale_x, delta=0.01,
                               msg=f"Scale X should be {expected_scale_x}")
        self.assertAlmostEqual(scale_value[1], expected_scale_y, delta=0.01,
                               msg=f"Scale Y should be {expected_scale_y}")
        self.assertAlmostEqual(scale_value[2], expected_scale_z, delta=0.01,
                               msg=f"Scale Z should be {expected_scale_z}")

    async def test_apply_transform_zero_division_safety(self):
        """
        元のバウンディングボックスのサイズが0の場合のゼロ除算回避をテスト

        期待される動作:
        元のサイズが0または非常に小さい場合、スケール比を1.0にする
        """
        # テスト用の空のXformを作成（バウンディングボックスなし）
        xform_path = "/TestXform"
        xform_geom = UsdGeom.Xform.Define(self.stage, Sdf.Path(xform_path))

        # JSONデータ
        layout_data = {
            "Length": 100.0,
            "Height": 100.0,
            "Width": 100.0,
            "X": 0.0,
            "Y": 0.0,
            "Z": 0.0,
        }

        # Extensionクラスをインポート
        from my.research.asset_placer_isaac.extension import MyExtension
        extension = MyExtension()

        # _apply_transformを呼び出し（エラーが発生しないことを確認）
        prim = self.stage.GetPrimAtPath(xform_path)
        try:
            await extension._apply_transform(prim, layout_data)
        except ZeroDivisionError:
            self.fail("_apply_transform raised ZeroDivisionError when original size is zero")

        # Xformable opsを確認
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()

        # スケールopが存在することを確認
        self.assertGreater(len(ops), 0, "Should have at least one xform op")

        scale_op = ops[0]
        scale_value = scale_op.Get()

        # ゼロ除算が発生した場合、スケール値は1.0になるはず
        self.assertIsInstance(scale_value, (Gf.Vec3f, Gf.Vec3d),
                              "Scale value should be a Vec3")

        # すべてのスケール値が有限値であることを確認
        self.assertTrue(all(abs(v) < 1e10 for v in scale_value),
                        "Scale values should be finite")

    async def test_apply_transform_clears_existing_transforms(self):
        """
        既存のトランスフォームがクリアされることをテスト

        期待される動作:
        _apply_transformを呼び出す前に設定されたトランスフォームは
        すべてクリアされ、新しいトランスフォームのみが適用される
        """
        # テスト用のキューブを作成
        cube_path = "/TestCubeWithExistingTransform"
        cube_geom = UsdGeom.Cube.Define(self.stage, Sdf.Path(cube_path))
        cube_geom.GetSizeAttr().Set(1.0)

        # 既存のトランスフォームを設定
        xformable = UsdGeom.Xformable(cube_geom)
        xformable.AddTranslateOp().Set(Gf.Vec3d(10.0, 20.0, 30.0))
        xformable.AddRotateYOp().Set(90.0)
        xformable.AddScaleOp().Set(Gf.Vec3f(5.0, 5.0, 5.0))

        # 既存のトランスフォームがあることを確認
        ops_before = xformable.GetOrderedXformOps()
        self.assertEqual(len(ops_before), 3, "Should have 3 existing xform ops")

        # 新しいJSONデータ
        layout_data = {
            "Length": 200.0,
            "Height": 150.0,
            "Width": 100.0,
            "X": 100.0,
            "Y": 50.0,
            "Z": 200.0,
            "rotationY": 45.0
        }

        # Extensionクラスをインポート
        from my.research.asset_placer_isaac.extension import MyExtension
        extension = MyExtension()

        # _apply_transformを呼び出し（非同期）
        prim = self.stage.GetPrimAtPath(cube_path)
        await extension._apply_transform(prim, layout_data)

        # トランスフォームが新しいものに置き換わっていることを確認
        ops_after = xformable.GetOrderedXformOps()
        self.assertEqual(len(ops_after), 3, "Should have exactly 3 new xform ops")

        # 新しい値が設定されていることを確認
        # 順序: Translate, Rotate, Scale
        translate_value = ops_after[0].Get()
        self.assertAlmostEqual(translate_value[0], 100.0, delta=0.01)  # 100m

        rotate_value = ops_after[1].Get()
        self.assertAlmostEqual(rotate_value, 45.0, delta=0.01)

        scale_value = ops_after[2].Get()
        self.assertAlmostEqual(scale_value[0], 200.0, delta=0.01)  # 200m / 1m = 200
