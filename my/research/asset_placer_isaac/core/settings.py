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

import json
import os
from typing import Dict, List, Optional, Set, Tuple

import omni.log


def _find_project_root(start_dir: str, max_depth: int = 12) -> Optional[str]:
    """
    既知のマーカーファイル（repo.toml / main.py）を頼りにプロジェクトルートを推定する。
    見つからない場合は None。
    """
    cur = os.path.abspath(start_dir)
    for _ in range(max_depth):
        if os.path.exists(os.path.join(cur, "repo.toml")) or os.path.exists(os.path.join(cur, "main.py")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def _get_default_json_output_dir(extension_dir: str) -> str:
    """
    JSON保存先のデフォルトを返す。
    - 可能なら <project_root>/json
    - だめなら <extension_dir>/json
    """
    root = _find_project_root(extension_dir)
    if root:
        return os.path.join(root, "json")
    return os.path.join(extension_dir, "json")


class SettingsMixin:
    def _get_project_root_dir(self) -> str:
        root = _find_project_root(self._extension_dir)
        if root:
            return root
        omni.log.warn("Project root not found. Falling back to extension directory for logs.")
        return self._extension_dir

    def _get_log_dir(self) -> str:
        return os.path.join(self._get_project_root_dir(), "log")

    def _load_settings_from_json(self) -> Dict[str, object]:
        """JSONファイルから設定を読み込む"""
        if os.path.exists(self._settings_file):
            try:
                with open(self._settings_file, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                omni.log.info(f"Settings loaded from: {self._settings_file}")
                return settings
            except Exception as e:
                omni.log.error(f"Failed to load settings from JSON: {e}")
                return {}
        else:
            omni.log.info("No settings file found. Using defaults.")
            return {}

    def _save_settings_to_json(self):
        """現在の設定をJSONファイルに保存"""
        try:
            step1_model_index = (
                self._ai_step1_model_combo.model.get_item_value_model().as_int
                if hasattr(self, "_ai_step1_model_combo")
                else self._ai_step1_model_index
            )
            step2_model_index = (
                self._ai_step2_model_combo.model.get_item_value_model().as_int
                if hasattr(self, "_ai_step2_model_combo")
                else self._ai_step2_model_index
            )
            reasoning_effort_index = (
                self._ai_reasoning_effort_combo.model.get_item_value_model().as_int
                if hasattr(self, "_ai_reasoning_effort_combo")
                else self._ai_reasoning_effort_index
            )
            text_verbosity_index = (
                self._ai_text_verbosity_combo.model.get_item_value_model().as_int
                if hasattr(self, "_ai_text_verbosity_combo")
                else self._ai_text_verbosity_index
            )
            image_detail_index = (
                self._ai_image_detail_combo.model.get_item_value_model().as_int
                if hasattr(self, "_ai_image_detail_combo")
                else self._ai_image_detail_index
            )
            max_output_tokens = self._ai_max_output_tokens
            if hasattr(self, "_ai_max_output_tokens_model"):
                raw_tokens = self._ai_max_output_tokens_model.as_string.strip()
                if raw_tokens:
                    try:
                        max_output_tokens = int(float(raw_tokens))
                    except (ValueError, TypeError):
                        max_output_tokens = self._ai_max_output_tokens
            if max_output_tokens < 0:
                max_output_tokens = 0

            max_retries = getattr(self, "_ai_max_retries", 0)
            if hasattr(self, "_ai_max_retries_model"):
                raw_retries = self._ai_max_retries_model.as_string.strip()
                if raw_retries:
                    try:
                        max_retries = int(float(raw_retries))
                    except (ValueError, TypeError):
                        max_retries = getattr(self, "_ai_max_retries", 0)
            if max_retries < 0:
                max_retries = 0

            retry_delay_sec = getattr(self, "_ai_retry_delay_sec", 0.0)
            if hasattr(self, "_ai_retry_delay_model"):
                raw_delay = self._ai_retry_delay_model.as_string.strip()
                if raw_delay:
                    try:
                        retry_delay_sec = float(raw_delay)
                    except (ValueError, TypeError):
                        retry_delay_sec = getattr(self, "_ai_retry_delay_sec", 0.0)
            if retry_delay_sec < 0:
                retry_delay_sec = 0.0

            self._ai_step1_model_index = step1_model_index
            self._ai_step2_model_index = step2_model_index
            self._ai_reasoning_effort_index = reasoning_effort_index
            self._ai_text_verbosity_index = text_verbosity_index
            self._ai_image_detail_index = image_detail_index
            self._ai_max_output_tokens = max_output_tokens
            self._ai_max_retries = max_retries
            self._ai_retry_delay_sec = retry_delay_sec

            settings = {
                "openai_api_key": self._api_key_model.as_string if hasattr(self, "_api_key_model") else "",
                "search_root_url": self._search_root_model.as_string,
                "image_path": self._image_path or "",
                "dimensions_path": self._dimensions_path or "",
                "prompt1_path": self._prompt1_path or "",
                "prompt2_path": self._prompt2_path or "",
                "json_output_dir": self._json_output_dir or "",
                "model_index": self._model_combo.model.get_item_value_model().as_int if hasattr(self, "_model_combo") else 0,
                "ai_step1_model_index": step1_model_index,
                "ai_step2_model_index": step2_model_index,
                "ai_reasoning_effort_index": reasoning_effort_index,
                "ai_text_verbosity_index": text_verbosity_index,
                "ai_image_detail_index": image_detail_index,
                "ai_max_output_tokens": max_output_tokens,
                "ai_max_retries": max_retries,
                "ai_retry_delay_sec": retry_delay_sec,
                "auto_layout_enabled": bool(getattr(self, "_auto_layout_enabled", False)),
                "auto_layout_mode": getattr(self, "_auto_layout_mode", "quick"),
                "auto_layout_file": getattr(self, "_auto_layout_file", ""),
                "debug_bbox_mode": bool(self._debug_bbox_mode.as_bool) if hasattr(self, "_debug_bbox_mode") else False,
                "asset_blacklist": sorted(getattr(self, "_asset_blacklist", set())),
                "asset_blacklist_keys": sorted(getattr(self, "_asset_blacklist_keys", set())),
            }

            with open(self._settings_file, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=4, ensure_ascii=False)

            omni.log.info(f"Settings saved to: {self._settings_file}")
        except Exception as e:
            omni.log.error(f"Failed to save settings to JSON: {e}")

    def _get_json_output_dir(self) -> str:
        if self._json_output_dir:
            return self._json_output_dir
        return _get_default_json_output_dir(self._extension_dir)

    def _load_json_with_fallback(self, filepath: str) -> Optional[dict]:
        """複数エンコーディングでJSONファイル読み込みを試みる。"""
        encodings = ["utf-8", "utf-8-sig", "cp932", "shift_jis"]
        for encoding in encodings:
            try:
                with open(filepath, "r", encoding=encoding) as f:
                    content = f.read()
                data = json.loads(content)
                if encoding != "utf-8":
                    omni.log.info(f"Loaded JSON using encoding '{encoding}'.")
                return data
            except UnicodeDecodeError:
                omni.log.warn(
                    f"Failed to decode '{filepath}' with encoding '{encoding}'. Trying next encoding..."
                )
                continue
            except FileNotFoundError:
                omni.log.error(f"JSON file not found: {filepath}")
                return None
            except json.JSONDecodeError as exc:
                omni.log.error(f"JSON parse error in '{filepath}' (encoding '{encoding}'): {exc}")
                return None

        omni.log.error(f"Unable to decode JSON file '{filepath}' with supported encodings.")
        return None

    def _read_text_with_fallback(self, filepath: str, label: str, required: bool) -> Optional[str]:
        """テキストファイルを複数エンコーディングで読み込む。"""
        if not filepath:
            if required:
                omni.log.error(f"{label} file path is empty.")
            return None

        encodings = ["utf-8", "utf-8-sig", "cp932", "shift_jis"]
        for encoding in encodings:
            try:
                with open(filepath, "r", encoding=encoding) as f:
                    content = f.read()
                if encoding != "utf-8":
                    omni.log.info(f"Loaded {label} using encoding '{encoding}'.")
                return content
            except UnicodeDecodeError:
                omni.log.warn(
                    f"Failed to decode {label} '{filepath}' with encoding '{encoding}'. Trying next encoding..."
                )
                continue
            except FileNotFoundError:
                message = f"{label} file not found: {filepath}"
                if required:
                    omni.log.error(message)
                    return None
                omni.log.warn(message)
                continue
            except Exception as exc:
                message = f"Error reading {label} '{filepath}': {exc}"
                if required:
                    omni.log.error(message)
                    return None
                omni.log.warn(message)
                continue

        message = f"Unable to decode {label} file '{filepath}' with supported encodings."
        if required:
            omni.log.error(message)
        else:
            omni.log.warn(message)
        return None

    def _resolve_prompt_text(
        self,
        explicit_path: Optional[str],
        default_filename: str,
        default_text: str,
        label: str,
    ) -> Tuple[str, str]:
        """プロンプトテキストを解決。ファイルが読めない場合は組み込みデフォルトを使用。"""
        candidate_paths: List[str] = []
        if explicit_path:
            candidate_paths.append(explicit_path)

        candidate_paths.append(default_filename)
        module_dir = os.path.dirname(__file__)
        candidate_paths.append(os.path.join(module_dir, default_filename))
        candidate_paths.append(os.path.join(module_dir, "data", default_filename))
        candidate_paths.append(os.path.join(os.path.dirname(module_dir), default_filename))

        seen: Set[str] = set()
        for path in candidate_paths:
            norm = os.path.abspath(path)
            if norm in seen:
                continue
            seen.add(norm)
            text = self._read_text_with_fallback(norm, label, required=False)
            if text is not None:
                omni.log.info(f"Using {label} from '{norm}'.")
                return text, norm

        omni.log.warn(f"{label} file not provided. Falling back to built-in default text.")
        return default_text, "built-in default"
