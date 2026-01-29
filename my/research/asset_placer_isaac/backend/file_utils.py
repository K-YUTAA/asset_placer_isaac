# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
File utility functions for backend module.
"""

import base64
import pathlib
from typing import Optional

import omni.log


def encode_image_to_base64(image_path: str) -> str:
    """画像ファイルをBase64エンコードして返す"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        omni.log.error(f"画像ファイルが見つかりません: {image_path}")
        raise


def read_text_from_file(file_path: str) -> str:
    """テキストファイルを読み込んで内容を返す"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        omni.log.error(f"テキストファイルが見つかりません: {file_path}")
        raise


def build_timestamped_path(
    user_path: Optional[str],
    base_name: str,
    timestamp: str,
    default_dir: str,
    suffix: str,
    tag: str
) -> pathlib.Path:
    """
    タイムスタンプ付きのファイルパスを生成する。

    user_path が:
      - None            : default_dir/base_name_{timestamp}_{tag}{suffix}
      - 既存/新規ディレクトリ: user_path/base_name_{timestamp}_{tag}{suffix}
      - ファイルパス(拡張子あり): 親ディレクトリに stem_{timestamp}{元拡張子} として保存
    """
    if user_path:
        p = pathlib.Path(user_path)
        if p.suffix:  # ファイル名が与えられた場合は {stem}_{timestamp}{suffix}
            parent = p.parent if str(p.parent) not in ("", ".") else pathlib.Path(default_dir)
            parent.mkdir(parents=True, exist_ok=True)
            return parent / f"{p.stem}_{timestamp}{p.suffix}"
        else:  # ディレクトリ指定
            p.mkdir(parents=True, exist_ok=True)
            return p / f"{base_name}_{timestamp}_{tag}{suffix}"
    else:
        outdir = pathlib.Path(default_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir / f"{base_name}_{timestamp}_{tag}{suffix}"
