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
Backend module for AI-based layout generation using OpenAI API.
Refactored from main.py for use as an Omniverse Extension.

This module maintains backward compatibility by re-exporting all functions
that were previously in the monolithic backend.py file.
"""

# ========================================
# Omniverse Kit 組み込み Python へのパッケージ自動インストール
# ========================================
try:
    import omni.kit.pipapi
    omni.kit.pipapi.install("opencv-python")
    omni.kit.pipapi.install("openai")
    omni.kit.pipapi.install("pillow")
except Exception as e:
    print(f"[Warning] Failed to auto-install packages: {e}")

# ========================================
# Import and re-export all public functions
# ========================================

# File utilities
from .file_utils import (
    encode_image_to_base64,
    read_text_from_file,
    build_timestamped_path,
)

# AI processing
from .ai_processing import (
    step1_analyze_image,
    step2_generate_json,
    step3_check_collisions,
)

# Export all functions to maintain backward compatibility
__all__ = [
    # File utilities
    "encode_image_to_base64",
    "read_text_from_file",
    "build_timestamped_path",
    # AI processing
    "step1_analyze_image",
    "step2_generate_json",
    "step3_check_collisions",
]
