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

DEFAULT_PROMPT1_TEXT = (
    "You are an interior layout analyst. Provide a concise summary for layout generation.\n"
    "- Coordinate system: X = right, Y = depth (up in the image), Z = up.\n"
    "- Origin: lower-left inner corner of the room.\n"
    "- Units: meters.\n"
    "- Output format: 6-12 bullet lines max. Each line: object name, count, "
    "approx center (x,y), approx size (LxW), and orientation note if obvious.\n"
    "- No long explanations, no headings, no tables."
)

DEFAULT_PROMPT2_TEXT = (
    "Create a JSON layout description for the room based on the analysis, dimensions text, and floor plan image. "
    "Output MUST be valid JSON only (no prose).\n"
    "Schema:\n"
    "{\n"
    '  "area_name": string,\n'
    '  "area_size_X": number,  # meters\n'
    '  "area_size_Y": number,  # meters\n'
    '  "area_objects_list": [\n'
    "    {\n"
    '      "object_name": string,\n'
    '      "category": string (optional),\n'
    '      "search_prompt": string (optional),\n'
    '      "X": number,  # meters\n'
    '      "Y": number,  # meters\n'
    '      "Length": number,  # meters (X-axis size)\n'
    '      "Width": number,   # meters (Y-axis size)\n'
    '      "Height": number,  # meters (Z-up height)\n'
    '      "rotationZ": number  # degrees: 0/90/180/270\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Notes: Z is up; positions are X/Y only. Use meters everywhere."
)

# UI表示/実呼び出しで共通に使うモデル一覧（ComboBoxの順序 = 保存されるmodel_index）
MODEL_CHOICES = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-5-mini",
    "gpt-5",
    "gpt-5.1",
    "gpt-5.2",
]

ADV_MODEL_CHOICES = ["(Use main model)"] + MODEL_CHOICES + ["gpt-5.2-pro"]
REASONING_EFFORT_CHOICES = ["(default)", "low", "medium", "high", "xhigh"]
TEXT_VERBOSITY_CHOICES = ["(default)", "low", "medium", "high"]
IMAGE_DETAIL_CHOICES = ["(default)", "low", "high"]

# Vector Searchで取得する候補数の上限
VECTOR_SEARCH_LIMIT = 50
