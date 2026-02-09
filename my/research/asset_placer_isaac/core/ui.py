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

import asyncio
import base64
import io
import json
import os
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import omni.kit.app
import omni.log
import omni.ui as ui
import omni.usd

from .constants import (
    ADV_MODEL_CHOICES,
    DEFAULT_PROMPT1_TEXT,
    DEFAULT_PROMPT2_TEXT,
    IMAGE_DETAIL_CHOICES,
    JSON_SIZE_MODE_CHOICES,
    MODEL_CHOICES,
    REASONING_EFFORT_CHOICES,
    TEXT_VERBOSITY_CHOICES,
)

SEARCH_TEST_DISPLAY_LIMIT = 10
SEARCH_TEST_THUMB_SIZE = 96


class UIMixin:
    def _setup_japanese_font(self) -> Dict[str, str]:
        """
        譌･譛ｬ隱槭ヵ繧ｩ繝ｳ繝医ｒ險ｭ螳壹☆繧・

        Omniverse Kit 縺ｧ縺ｯ縲√ヵ繧ｩ繝ｳ繝医・繝阪・繧ｸ繝｣繝ｼ縺ｸ縺ｮ逋ｻ骭ｲ縺ｯ荳崎ｦ√・
        繧ｹ繧ｿ繧､繝ｫ霎樊嶌縺ｫ逶ｴ謗･繝輔か繝ｳ繝医ヵ繧｡繧､繝ｫ繝代せ繧呈欠螳壹☆繧後・蜍穂ｽ懊☆繧九・
        """
        try:
            omni.log.info("=== 譌･譛ｬ隱槭ヵ繧ｩ繝ｳ繝医・險ｭ螳壹ｒ髢句ｧ・===")

            # Windows讓呎ｺ悶・譌･譛ｬ隱槭ヵ繧ｩ繝ｳ繝医Μ繧ｹ繝茨ｼ亥━蜈磯・ｼ・
            japanese_fonts = [
                ("MS Gothic", "C:/Windows/Fonts/msgothic.ttc"),
                ("Yu Gothic", "C:/Windows/Fonts/YuGothM.ttc"),
                ("Meiryo", "C:/Windows/Fonts/meiryo.ttc"),
            ]

            for font_name, font_path in japanese_fonts:
                # 繝輔ぃ繧､繝ｫ縺ｮ蟄伜惠遒ｺ隱・
                if not os.path.exists(font_path):
                    omni.log.warn(f"繝輔か繝ｳ繝医ヵ繧｡繧､繝ｫ縺瑚ｦ九▽縺九ｊ縺ｾ縺帙ｓ: {font_path}")
                    continue

                # 繝輔ぃ繧､繝ｫ縺悟ｭ伜惠縺吶ｋ蝣ｴ蜷医√◎縺ｮ繝代せ繧偵せ繧ｿ繧､繝ｫ縺ｨ縺励※霑斐☆
                omni.log.info(f"笨・譌･譛ｬ隱槭ヵ繧ｩ繝ｳ繝・'{font_name}' 繧剃ｽｿ逕ｨ")
                omni.log.info(f"  繝代せ: {font_path}")

                # omni.ui.StringField 縺ｮ style 繝励Ο繝代ユ繧｣縺ｫ貂｡縺吶せ繧ｿ繧､繝ｫ霎樊嶌
                # "Font" 繧ｭ繝ｼ・亥､ｧ譁・ｭ励・F・峨↓繝輔ぃ繧､繝ｫ繝代せ繧呈欠螳・
                style = {
                    "Font": font_path,
                    "font_size": 14
                }

                omni.log.info("笨・繝輔か繝ｳ繝医せ繧ｿ繧､繝ｫ險ｭ螳壼ｮ御ｺ・")
                return style

            # 縺吶∋縺ｦ縺ｮ繝輔か繝ｳ繝医′隕九▽縺九ｉ縺ｪ縺九▲縺溷ｴ蜷・
            omni.log.warn("笞 譌･譛ｬ隱槭ヵ繧ｩ繝ｳ繝医′隕九▽縺九ｊ縺ｾ縺帙ｓ縺ｧ縺励◆")
            omni.log.warn("  繝・ヵ繧ｩ繝ｫ繝医ヵ繧ｩ繝ｳ繝医ｒ菴ｿ逕ｨ縺励∪縺呻ｼ域律譛ｬ隱槭′譁・ｭ怜喧縺代☆繧句庄閭ｽ諤ｧ縺後≠繧翫∪縺呻ｼ・")
            return {}

        except Exception as e:
            omni.log.error(f"笨・譌･譛ｬ隱槭ヵ繧ｩ繝ｳ繝郁ｨｭ螳壻ｸｭ縺ｫ繧ｨ繝ｩ繝ｼ縺檎匱逕・ {e}")
            import traceback
            omni.log.error(f"繧ｹ繧ｿ繝・け繝医Ξ繝ｼ繧ｹ:\n{traceback.format_exc()}")
            return {}

    def _deferred_destroy_window(self, window) -> None:
        """destroys a ui.Window safely on the next UI update frame."""
        if not window:
            return
        app = omni.kit.app.get_app()

        def _destroy_window():
            try:
                window.destroy()
            except Exception as exc:  # pragma: no cover - defensive
                omni.log.warn(f"Failed to destroy window: {exc}")

        if hasattr(app, "next_update_async"):
            async def _wait_then_destroy():
                try:
                    await app.next_update_async()
                except Exception as exc:  # pragma: no cover - defensive
                    omni.log.warn(f"next_update_async failed: {exc}")
                _destroy_window()

            asyncio.ensure_future(_wait_then_destroy())
            return

        if hasattr(app, "next_update"):
            def _on_next_update(_dt):
                _destroy_window()

            app.next_update(_on_next_update)
            return

        asyncio.get_event_loop().call_soon(_destroy_window)

    def _format_preview_source(self, source: str) -> str:
        if not source:
            return "unknown"
        if source == "built-in default":
            return source
        return os.path.basename(source)

    def _to_file_image_url(self, path: str) -> str:
        if not path:
            return ""
        norm = os.path.abspath(path).replace("\\", "/")
        # Ensure a proper file:// URL for local images
        if not norm.startswith("/"):
            norm = f"/{norm}"
        return f"file://{quote(norm, safe='/:')}"

    def _refresh_preview_models(self) -> None:
        prompt1_text, prompt1_source = self._resolve_prompt_text(
            self._prompt1_path, "prompt_1.txt", DEFAULT_PROMPT1_TEXT, "Prompt 1"
        )
        prompt2_text, prompt2_source = self._resolve_prompt_text(
            self._prompt2_path, "prompt_2.txt", DEFAULT_PROMPT2_TEXT, "Prompt 2"
        )
        dims_text = self._read_text_with_fallback(self._dimensions_path, "Dimensions", required=False) or ""

        self._prompt1_preview_model.set_value(prompt1_text)
        self._prompt2_preview_model.set_value(prompt2_text)
        self._dims_preview_model.set_value(dims_text)

        if self._prompt1_preview_label:
            self._prompt1_preview_label.text = f"Prompt 1 Preview ({self._format_preview_source(prompt1_source)})"
        if self._prompt2_preview_label:
            self._prompt2_preview_label.text = f"Prompt 2 Preview ({self._format_preview_source(prompt2_source)})"
        if self._dims_preview_label:
            dims_source = os.path.basename(self._dimensions_path) if self._dimensions_path else "No file selected"
            self._dims_preview_label.text = f"Dimensions Preview ({dims_source})"

    def _update_image_preview(self) -> None:
        if not self._image_preview_label or not self._image_preview_provider:
            return
        if self._image_path and os.path.exists(self._image_path):
            self._image_preview_label.text = f"Image Preview ({os.path.basename(self._image_path)})"
            try:
                from PIL import Image
                image = Image.open(self._image_path).convert("RGBA")
                max_width, max_height = 560, 320
                if image.width > max_width or image.height > max_height:
                    image.thumbnail((max_width, max_height), Image.LANCZOS)
                data = list(image.tobytes())
                self._image_preview_provider.set_bytes_data(data, [image.width, image.height])
                if self._image_preview_frame:
                    self._image_preview_frame.rebuild()
                return
            except Exception as exc:  # pragma: no cover - defensive
                omni.log.warn(f"Failed to load image preview: {exc}")
        else:
            self._image_preview_label.text = "Image Preview (No file selected)"
        try:
            self._image_preview_provider.set_bytes_data([0, 0, 0, 0], [1, 1])
            if self._image_preview_frame:
                self._image_preview_frame.rebuild()
        except Exception as exc:  # pragma: no cover - defensive
            omni.log.warn(f"Failed to clear image preview: {exc}")

    def _update_blacklist_label(self) -> None:
        if hasattr(self, "_blacklist_count_label") and self._blacklist_count_label:
            count = len(getattr(self, "_asset_blacklist", []))
            self._blacklist_count_label.text = f"Blacklisted: {count}"
        if hasattr(self, "_blacklist_window_count_label") and self._blacklist_window_count_label:
            count = len(getattr(self, "_asset_blacklist", []))
            self._blacklist_window_count_label.text = f"Blacklisted: {count}"

    def _resize_blacklist_window(self):
        if not self._blacklist_window or not self._blacklist_scrolling_frame:
            return
        # Header + separator + margins 縺ｮ蛻・ｒ蠑輔＞縺滓ｮ九ｊ繧偵せ繧ｯ繝ｭ繝ｼ繝ｫ鬆伜沺縺ｫ蜈・※繧・
        header_offset = 90
        width_margin = 20
        height = max(120, int(self._blacklist_window.height - header_offset))
        width = max(200, int(self._blacklist_window.width - width_margin))
        self._blacklist_scrolling_frame.height = ui.Pixel(height)
        self._blacklist_scrolling_frame.width = ui.Pixel(width)

    def _resize_json_editor_window(self):
        if not self._json_editor_window or not self._json_editor_assets_frame:
            return
        margin = 28
        fixed_height = 360  # show ~6 items comfortably (DPI-safe)
        width = max(300, int(self._json_editor_window.width - margin))
        self._json_editor_assets_frame.height = ui.Pixel(fixed_height)
        self._json_editor_assets_frame.width = ui.Pixel(width)


    def _populate_blacklist_list(self):
        if not self._blacklist_list_container:
            return
        if not getattr(self, "_asset_blacklist", None):
            ui.Label("No blacklisted assets.")
            return
        for asset_url in sorted(self._asset_blacklist):
            with ui.HStack(height=26):
                url_model = ui.SimpleStringModel(asset_url)
                ui.StringField(model=url_model, read_only=True, height=24, width=ui.Fraction(1))
                ui.Spacer(width=8)
                ui.Button(
                    "Remove",
                    clicked_fn=lambda url=asset_url: self._remove_blacklist_entry(url),
                    width=90,
                    height=24,
                )

    def _rebuild_blacklist_list(self) -> None:
        if self._blacklist_list_container:
            self._blacklist_list_container.clear()
            with self._blacklist_list_container:
                self._populate_blacklist_list()

    def _remove_blacklist_entry(self, asset_url: str) -> None:
        async def _remove_async():
            removed_url = self._remove_asset_from_blacklist(asset_url)
            identity_key = await self._get_asset_identity_key(asset_url)
            removed_key = self._remove_asset_key_from_blacklist(identity_key)
            if removed_url or removed_key:
                self._save_settings_to_json()
                self._update_blacklist_label()
                self._rebuild_blacklist_list()

        asyncio.ensure_future(_remove_async())

    def _clear_blacklist_entries(self) -> None:
        cleared_url = self._clear_asset_blacklist()
        cleared_keys = self._clear_asset_blacklist_keys()
        if cleared_url or cleared_keys:
            self._save_settings_to_json()
            self._update_blacklist_label()
            self._rebuild_blacklist_list()

    def _open_blacklist_window(self):
        if self._blacklist_window:
            existing = self._blacklist_window
            self._blacklist_window = None
            self._deferred_destroy_window(existing)

        self._blacklist_window = ui.Window("Blacklisted Assets", width=720, height=520)
        with self._blacklist_window.frame:
            with ui.VStack(spacing=6, style={"margin": 10}):
                with ui.HStack(height=28):
                    self._blacklist_window_count_label = ui.Label("Blacklisted: 0", width=0)
                    ui.Spacer()
                    ui.Button("Clear All", clicked_fn=self._clear_blacklist_entries, width=100, height=24)
                    ui.Button("Close", clicked_fn=self._close_blacklist_window, width=100, height=24)

                ui.Separator()
                self._blacklist_scrolling_frame = ui.ScrollingFrame(
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                )
                with self._blacklist_scrolling_frame:
                    self._blacklist_list_container = ui.VStack(spacing=6, height=0, style={"margin": 4})

        self._update_blacklist_label()
        self._rebuild_blacklist_list()
        self._resize_blacklist_window()
        self._blacklist_window.set_width_changed_fn(lambda _w: self._resize_blacklist_window())
        self._blacklist_window.set_height_changed_fn(lambda _h: self._resize_blacklist_window())

    def _close_blacklist_window(self):
        if not self._blacklist_window:
            return
        existing = self._blacklist_window
        self._blacklist_window = None
        self._deferred_destroy_window(existing)
        self._blacklist_list_container = None
        self._blacklist_window_count_label = None
        self._blacklist_scrolling_frame = None

    def _set_generated_json_preview(self, json_path: str, layout_json: Dict[str, object]) -> None:
        self._generated_json_path = json_path or ""
        if not json_path:
            self._generated_json_preview_model.set_value("No generated JSON yet.")
            if self._generated_json_label:
                self._generated_json_label.text = "Latest JSON: None"
            return

        try:
            preview_text = json.dumps(layout_json, indent=2, ensure_ascii=False)
        except Exception as exc:  # pragma: no cover - defensive
            omni.log.warn(f"Failed to serialize JSON for preview: {exc}")
            preview_text = str(layout_json)

        self._generated_json_preview_model.set_value(preview_text)
        if self._generated_json_label:
            self._generated_json_label.text = f"Latest JSON: {os.path.basename(json_path)}"

    def _refresh_generated_json_preview_model(self) -> None:
        if not self._generated_json_path:
            self._generated_json_preview_model.set_value("No generated JSON yet.")
            return

        if not os.path.exists(self._generated_json_path):
            self._generated_json_preview_model.set_value(f"File not found: {self._generated_json_path}")
            return

        data = self._load_json_with_fallback(self._generated_json_path)
        if data is None:
            self._generated_json_preview_model.set_value(f"Failed to load JSON: {self._generated_json_path}")
            return

        try:
            self._generated_json_preview_model.set_value(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as exc:  # pragma: no cover - defensive
            omni.log.warn(f"Failed to serialize JSON for preview: {exc}")
            self._generated_json_preview_model.set_value(str(data))

    def _open_generated_json_preview_window(self):
        if not self._generated_json_path:
            omni.log.warn("No generated JSON available for preview.")
            return

        if self._generated_json_preview_window:
            existing = self._generated_json_preview_window
            self._generated_json_preview_window = None
            self._deferred_destroy_window(existing)

        self._generated_json_preview_window = ui.Window("Generated JSON Preview", width=620, height=720)
        with self._generated_json_preview_window.frame:
            with ui.VStack(spacing=8, style={"margin": 10}):
                ui.Label(f"Source: {self._generated_json_path}")
                ui.StringField(
                    model=self._generated_json_preview_model,
                    multiline=True,
                    read_only=True,
                    height=620,
                    style=self._japanese_font_style,
                )
                ui.Spacer(height=8)
                ui.Button("Close", clicked_fn=self._close_generated_json_preview_window, width=120)

        self._refresh_generated_json_preview_model()

    def _close_generated_json_preview_window(self):
        if not self._generated_json_preview_window:
            return
        existing = self._generated_json_preview_window
        self._generated_json_preview_window = None
        self._deferred_destroy_window(existing)

    def _refresh_loaded_json_preview_model(self) -> None:
        if not self._loaded_json_path:
            self._loaded_json_preview_model.set_value("No selected JSON.")
            return
        if not os.path.exists(self._loaded_json_path):
            self._loaded_json_preview_model.set_value(f"File not found: {self._loaded_json_path}")
            return

        data = self._load_json_with_fallback(self._loaded_json_path)
        if data is None:
            self._loaded_json_preview_model.set_value(f"Failed to load JSON: {self._loaded_json_path}")
            return

        try:
            self._loaded_json_preview_model.set_value(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as exc:  # pragma: no cover - defensive
            omni.log.warn(f"Failed to serialize JSON for preview: {exc}")
            self._loaded_json_preview_model.set_value(str(data))

    def _open_loaded_json_preview_window(self):
        if not self._loaded_json_path:
            omni.log.warn("No selected JSON for preview.")
            return

        if self._loaded_json_preview_window:
            existing = self._loaded_json_preview_window
            self._loaded_json_preview_window = None
            self._deferred_destroy_window(existing)

        self._loaded_json_preview_window = ui.Window("Selected JSON Preview", width=620, height=720)
        with self._loaded_json_preview_window.frame:
            with ui.VStack(spacing=8, style={"margin": 10}):
                ui.Label(f"Source: {self._loaded_json_path}")
                ui.StringField(
                    model=self._loaded_json_preview_model,
                    multiline=True,
                    read_only=True,
                    height=620,
                    style=self._japanese_font_style,
                )
                ui.Spacer(height=8)
                ui.Button("Close", clicked_fn=self._close_loaded_json_preview_window, width=120)

        self._refresh_loaded_json_preview_model()

    def _close_loaded_json_preview_window(self):
        if not self._loaded_json_preview_window:
            return
        existing = self._loaded_json_preview_window
        self._loaded_json_preview_window = None
        self._deferred_destroy_window(existing)

    def _open_loaded_json_editor(self):
        if not self._loaded_json_path or not os.path.exists(self._loaded_json_path):
            omni.log.warn("No selected JSON for editing.")
            return
        self._open_json_editor_window()
        self._json_editor_selected_path = self._loaded_json_path
        self._load_json_for_edit(self._loaded_json_path)

    # --- JSON search_prompt editor ---

    def _get_json_editor_dir(self) -> str:
        try:
            return self._get_json_output_dir()
        except Exception:
            return os.path.join(self._extension_dir, "json")

    def _refresh_json_editor_file_list(self) -> List[str]:
        json_dir = self._get_json_editor_dir()
        files: List[str] = []
        if os.path.isdir(json_dir):
            for name in os.listdir(json_dir):
                if name.lower().endswith(".json"):
                    files.append(name)
        files.sort(reverse=True)
        self._json_editor_files = files
        return files

    def _load_json_for_edit(self, json_path: str):
        if not json_path or not os.path.exists(json_path):
            self._json_editor_data = None
            self._json_editor_assets = []
            self._json_editor_assets_key = None
            if self._json_editor_assets_frame:
                self._json_editor_assets_frame.clear()
                with self._json_editor_assets_frame:
                    self._build_json_editor_assets_list()
            return
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._json_editor_data = data
        except Exception as exc:
            omni.log.error(f"Failed to load JSON for edit: {exc}")
            self._json_editor_data = None
            self._json_editor_assets = []
            self._json_editor_assets_key = None
            if self._json_editor_assets_frame:
                self._json_editor_assets_frame.clear()
                with self._json_editor_assets_frame:
                    self._build_json_editor_assets_list()
            return

        assets: List[Dict[str, object]] = []
        assets_key = None
        if isinstance(data, dict):
            if isinstance(data.get("area_objects_list"), list):
                assets = data.get("area_objects_list") or []
                assets_key = "area_objects_list"
            elif isinstance(data.get("objects"), list):
                assets = data.get("objects") or []
                assets_key = "objects"
        elif isinstance(data, list):
            assets = data
            assets_key = None

        self._json_editor_assets = assets
        self._json_editor_assets_key = assets_key
        if self._json_editor_assets_frame:
            self._json_editor_assets_frame.clear()
            with self._json_editor_assets_frame:
                self._build_json_editor_assets_list()

    def _save_json_editor_changes(self):
        if not self._json_editor_selected_path or not self._json_editor_data:
            omni.log.warn("No JSON selected for saving.")
            return
        if self._json_editor_assets and hasattr(self, "_json_editor_asset_models"):
            for idx, model in self._json_editor_asset_models.items():
                if idx < 0 or idx >= len(self._json_editor_assets):
                    continue
                self._json_editor_assets[idx]["search_prompt"] = model.as_string

        try:
            with open(self._json_editor_selected_path, "w", encoding="utf-8") as f:
                json.dump(self._json_editor_data, f, ensure_ascii=False, indent=2)
            omni.log.info(f"JSON saved: {self._json_editor_selected_path}")
        except Exception as exc:
            omni.log.error(f"Failed to save JSON: {exc}")

    def _open_json_editor_window(self):
        if self._json_editor_window:
            existing = self._json_editor_window
            self._json_editor_window = None
            self._deferred_destroy_window(existing)

        self._json_editor_window = ui.Window("Edit JSON Search Prompts", width=860, height=620)
        self._json_editor_file_combo = None
        self._json_editor_files = []
        self._json_editor_selected_path = ""
        self._json_editor_assets = []
        self._json_editor_assets_key = None
        self._json_editor_asset_models = {}
        self._json_editor_assets_frame = None

        json_dir = self._get_json_editor_dir()
        files = self._refresh_json_editor_file_list()

        with self._json_editor_window.frame:
            with ui.VStack(spacing=8, height=0, style={"margin": 10}):
                # Top controls (auto height)
                self._json_editor_top_frame = ui.Frame()
                with self._json_editor_top_frame:
                    with ui.VStack(spacing=6):
                        ui.Label(f"JSON Folder: {json_dir}")
                        with ui.HStack(height=26):
                            ui.Label("JSON File:", width=120)
                            combo_items = files if files else ["(no json files)"]
                            self._json_editor_file_combo = ui.ComboBox(0, *combo_items)
                            ui.Button("Refresh", width=90, clicked_fn=self._on_refresh_json_editor_files)

                        with ui.HStack(height=26):
                            ui.Button("Load", width=90, clicked_fn=self._on_load_json_editor_file)
                            ui.Button("Save", width=90, clicked_fn=self._save_json_editor_changes)
                            ui.Label("Edit search_prompt per asset", width=0)

                ui.Separator()

                # Scroll area (size is adjusted on window resize)
                self._json_editor_assets_frame = ui.ScrollingFrame(height=ui.Pixel(360), width=ui.Fraction(1))
                with self._json_editor_assets_frame:
                    self._build_json_editor_assets_list()

        self._on_load_json_editor_file()
        self._resize_json_editor_window()
        self._json_editor_window.set_width_changed_fn(lambda _w: self._resize_json_editor_window())
        self._json_editor_window.set_height_changed_fn(lambda _h: self._resize_json_editor_window())

    def _on_refresh_json_editor_files(self):
        self._open_json_editor_window()

    def _on_load_json_editor_file(self):
        if not self._json_editor_file_combo or not self._json_editor_files:
            return
        idx = self._json_editor_file_combo.model.get_item_value_model().as_int
        if idx < 0 or idx >= len(self._json_editor_files):
            idx = 0
        filename = self._json_editor_files[idx]
        if filename == "(no json files)":
            return
        json_path = os.path.join(self._get_json_editor_dir(), filename)
        self._json_editor_selected_path = json_path
        self._load_json_for_edit(json_path)

    def _build_json_editor_assets_list(self):
        self._json_editor_asset_models = {}
        if not self._json_editor_assets:
            ui.Label("No assets found in selected JSON.")
            return

        with ui.VStack(spacing=6):
            for idx, asset in enumerate(self._json_editor_assets):
                if not isinstance(asset, dict):
                    continue
                name = str(asset.get("object_name", "") or "")
                category = str(asset.get("category", "") or "")
                prompt = str(asset.get("search_prompt", "") or "")
                model = ui.SimpleStringModel(prompt)
                self._json_editor_asset_models[idx] = model

                with ui.HStack(height=26):
                    ui.Label(name or f"#{idx+1}", width=180)
                    ui.Label(category, width=120)
                    ui.StringField(model=model, width=ui.Fraction(1))
            ui.Spacer(height=12)

    def _open_preview_window(self):
        if self._preview_window:
            existing = self._preview_window
            self._preview_window = None
            self._deferred_destroy_window(existing)

        self._preview_window = ui.Window("Selected File Preview", width=620, height=720)
        with self._preview_window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=8, style={"margin": 10}):
                    self._image_preview_label = ui.Label("Image Preview")
                    self._image_preview_frame = ui.Frame()
                    with self._image_preview_frame:
                        ui.ImageWithProvider(self._image_preview_provider, width=560, height=320)

                    ui.Separator()
                    self._dims_preview_label = ui.Label("Dimensions Preview")
                    ui.StringField(
                        model=self._dims_preview_model,
                        multiline=True,
                        read_only=True,
                        height=100,
                        style=self._japanese_font_style,
                    )

                    ui.Separator()
                    self._prompt1_preview_label = ui.Label("Prompt 1 Preview")
                    ui.StringField(
                        model=self._prompt1_preview_model,
                        multiline=True,
                        read_only=True,
                        height=160,
                        style=self._japanese_font_style,
                    )

                    ui.Separator()
                    self._prompt2_preview_label = ui.Label("Prompt 2 Preview")
                    ui.StringField(
                        model=self._prompt2_preview_model,
                        multiline=True,
                        read_only=True,
                        height=160,
                        style=self._japanese_font_style,
                    )

                    ui.Spacer(height=8)
                    ui.Button("Close", clicked_fn=self._close_preview_window, width=120)

        self._refresh_preview_models()
        self._update_image_preview()

    def _close_preview_window(self):
        if not self._preview_window:
            return
        existing = self._preview_window
        self._preview_window = None
        self._deferred_destroy_window(existing)
        self._image_preview_widget = None
        self._image_preview_frame = None
        self._image_preview_label = None
        self._prompt1_preview_label = None
        self._prompt2_preview_label = None
        self._dims_preview_label = None

    def _open_search_test_window(self):
        if self._search_test_window:
            existing = self._search_test_window
            self._search_test_window = None
            self._deferred_destroy_window(existing)

        if not self._search_test_query_model:
            self._search_test_query_model = ui.SimpleStringModel("")

        self._search_test_window = ui.Window("USD Search Tester", width=720, height=640)
        with self._search_test_window.frame:
            with ui.VStack(spacing=8, style={"margin": 10}):
                ui.Label("Vector Search Tester")
                with ui.HStack(height=26):
                    ui.Label("Query:", width=60)
                    ui.StringField(model=self._search_test_query_model, width=ui.Fraction(1))
                    ui.Button("Search", clicked_fn=self._on_search_test_click, width=100)

                self._search_test_status_label = ui.Label("Ready")
                ui.Separator()

                with ui.ScrollingFrame(
                    height=440,
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                ):
                    self._search_test_results_frame = ui.Frame()
                    self._search_test_results_frame.set_build_fn(self._build_search_test_results_frame)

                ui.Spacer(height=6)
                ui.Button("Close", clicked_fn=self._close_search_test_window, width=120)

        self._set_search_test_status("Ready")
        self._rebuild_search_test_results()

    def _close_search_test_window(self):
        if not self._search_test_window:
            return
        existing = self._search_test_window
        self._search_test_window = None
        self._deferred_destroy_window(existing)
        if self._search_test_task and not self._search_test_task.done():
            self._search_test_task.cancel()
        self._search_test_task = None
        self._search_test_results_frame = None
        self._search_test_status_label = None
        self._search_test_result_providers = []

    def _set_search_test_status(self, message: str) -> None:
        if self._search_test_status_label:
            self._search_test_status_label.text = message

    def _on_search_test_click(self) -> None:
        query = self._search_test_query_model.as_string.strip() if self._search_test_query_model else ""
        if not query:
            self._set_search_test_status("Query is empty.")
            return

        if self._search_test_task and not self._search_test_task.done():
            self._search_test_task.cancel()

        self._set_search_test_status("Searching...")
        self._search_test_results = []
        self._rebuild_search_test_results()
        self._search_test_task = asyncio.ensure_future(self._run_search_test(query))

    async def _run_search_test(self, query: str) -> None:
        try:
            results = await self._search_assets_for_query(query, target_valid=10, initial_limit=10, max_limit=200)
        except Exception as exc:
            omni.log.warn(f"Search test failed: {exc}")
            self._set_search_test_status("Search failed.")
            return

        self._search_test_results = results or []
        total_results = len(self._search_test_results)
        if total_results > SEARCH_TEST_DISPLAY_LIMIT:
            self._set_search_test_status(
                f"Results: {total_results} (showing {SEARCH_TEST_DISPLAY_LIMIT})"
            )
        else:
            self._set_search_test_status(f"Results: {total_results}")
        self._rebuild_search_test_results()

    def _build_search_test_provider(self, image_b64: Optional[str]) -> Optional[ui.ByteImageProvider]:
        if not image_b64:
            return None
        try:
            from PIL import Image

            raw = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(raw)).convert("RGBA")
            max_size = SEARCH_TEST_THUMB_SIZE
            if image.width > max_size or image.height > max_size:
                image.thumbnail((max_size, max_size), Image.LANCZOS)
            provider = ui.ByteImageProvider()
            provider.set_bytes_data(list(image.tobytes()), [image.width, image.height])
            return provider
        except Exception as exc:  # pragma: no cover - defensive
            omni.log.warn(f"Failed to decode search thumbnail: {exc}")
            return None

    def _blacklist_search_result(self, asset_url: str, identity_key: Optional[str]) -> None:
        added_url = self._add_asset_to_blacklist(asset_url)
        added_key = self._add_asset_key_to_blacklist(identity_key)
        if added_url or added_key:
            self._save_settings_to_json()
            self._update_blacklist_label()
        self._rebuild_search_test_results()

    async def _search_assets_for_query(
        self,
        query: str,
        target_valid: int = 10,
        initial_limit: int = 10,
        max_limit: int = 200,
    ) -> List[Dict[str, object]]:
        """
        繝・せ繝育畑: 繝吶け繧ｿ繝ｼ讀懃ｴ｢縺ｧ繧ｵ繝繝阪う繝ｫ莉倥″縺ｮ邨先棡荳隕ｧ繧貞叙蠕励☆繧九・
        繝悶Λ繝・け繝ｪ繧ｹ繝・蜷御ｸ諤ｧ繧ｭ繝ｼ縺ｧ繝輔ぅ繝ｫ繧ｿ縺励∵怏蜉ｹ蛟呵｣懊′ target_valid 莉ｶ縺ｫ驕斐☆繧九∪縺ｧ蜀肴､懃ｴ｢縺吶ｋ縲・
        """
        import requests
        import json

        if not query:
            return []

        api_url = "http://192.168.11.65:30080/search"
        api_basic_auth = "omniverse:tsukuverse"
        usd_extensions = (".usd", ".usda", ".usdc", ".usdz")

        excluded_urls = set()
        for url in getattr(self, "_asset_blacklist", set()):
            if url:
                excluded_urls.add(self._normalize_asset_url(str(url)))
        excluded_keys = set(getattr(self, "_asset_blacklist_keys", set()))

        def _extract_hits(result: object) -> List[object]:
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                if "results" in result:
                    return result.get("results", [])
                if "hits" in result:
                    hits = result.get("hits")
                    if isinstance(hits, dict) and "hits" in hits:
                        return hits.get("hits", [])
                    if isinstance(hits, list):
                        return hits
                if "items" in result:
                    return result.get("items", [])
            return []

        def _coerce_size(value) -> Optional[int]:
            try:
                size_val = int(value)
                return size_val if size_val >= 0 else None
            except (TypeError, ValueError):
                return None

        def _extract_size(data: object) -> Optional[int]:
            if not isinstance(data, dict):
                return None
            for key in ("size", "file_size", "fileSize", "bytes", "file_size_bytes", "content_length"):
                if key in data:
                    size_val = _coerce_size(data.get(key))
                    if size_val is not None:
                        return size_val
            nested = data.get("stat")
            if isinstance(nested, dict):
                for key in ("size", "file_size", "bytes"):
                    if key in nested:
                        size_val = _coerce_size(nested.get(key))
                        if size_val is not None:
                            return size_val
            return None

        def _extract_url(item_data: object) -> Optional[str]:
            if not isinstance(item_data, dict):
                return None
            for key in ("url", "uri", "path", "file_path", "asset_path"):
                val = item_data.get(key)
                if isinstance(val, str) and val:
                    return val
            return None

        limit = max(1, int(initial_limit))
        prev_total_hits = -1
        loop = asyncio.get_event_loop()

        while True:
            payload = {
                "vector_queries": [
                    {
                        "field_name": "clip-embedding.embedding",
                        "query_type": "text",
                        "query": query,
                    }
                ],
                "return_metadata": True,
                "return_images": True,
                "limit": limit,
                "file_extension_include": "usd,usda,usdc,usdz",
                "file_extension_exclude": "png,jpg,jpeg",
            }

            def sync_request():
                user, password = api_basic_auth.split(":", 1)
                response = requests.post(api_url, json=payload, auth=(user, password), timeout=60)
                response.raise_for_status()
                return response.json()

            result = await loop.run_in_executor(None, sync_request)
            items = _extract_hits(result)
            total_hits = len(items)
            if total_hits == 0:
                return []

            candidates = []
            for item in items:
                item_data = item.get("_source", item) if isinstance(item, dict) else {}
                url = _extract_url(item_data)
                if not url:
                    continue
                if not url.lower().endswith(usd_extensions):
                    continue
                size_val = _extract_size(item_data) or _extract_size(item)
                score = 0.0
                if isinstance(item, dict):
                    score = item.get("score") or item.get("_score") or item_data.get("score", 0.0)
                image_b64 = None
                if isinstance(item, dict):
                    image_b64 = item.get("image") or item_data.get("image")
                candidates.append(
                    {
                        "url": url,
                        "score": score,
                        "image_b64": image_b64,
                        "size_bytes": size_val,
                    }
                )

            filtered: List[Dict[str, object]] = []
            seen_keys = set()
            for cand in candidates:
                normalized_url = self._normalize_asset_url(str(cand.get("url", "")))
                if not normalized_url:
                    continue
                if normalized_url in excluded_urls:
                    continue

                size_bytes = cand.get("size_bytes")
                if size_bytes is None:
                    size_bytes = await self._get_asset_size(normalized_url)
                identity_key = self._build_asset_identity_key(normalized_url, size_bytes)

                if identity_key and identity_key in excluded_keys:
                    continue
                if identity_key and identity_key in seen_keys:
                    continue
                if identity_key:
                    seen_keys.add(identity_key)

                try:
                    score_val = float(cand.get("score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    score_val = 0.0

                filtered.append(
                    {
                        "url": normalized_url,
                        "score": score_val,
                        "image_b64": cand.get("image_b64"),
                        "identity_key": identity_key,
                        "size_bytes": size_bytes,
                    }
                )

            if len(filtered) >= target_valid:
                return filtered
            if limit >= max_limit:
                omni.log.info(
                    f"[Vector Search Test] Reached max limit ({max_limit}). Returning {len(filtered)} results."
                )
                return filtered
            if total_hits <= prev_total_hits:
                omni.log.info(
                    f"[Vector Search Test] No additional hits (total={total_hits}). Returning {len(filtered)} results."
                )
                return filtered

            prev_total_hits = total_hits
            limit = min(limit * 2, max_limit)
            omni.log.info(f"[Vector Search Test] Increasing limit to {limit} for more results.")

    def _build_search_test_results_frame(self):
        self._search_test_result_providers = []
        results = self._search_test_results or []
        if not results:
            ui.Label("No results yet.")
            return

        excluded_keys = set(getattr(self, "_asset_blacklist_keys", set()))
        shown = 0
        display_results = results[:SEARCH_TEST_DISPLAY_LIMIT]
        row_height = SEARCH_TEST_THUMB_SIZE + 36
        with ui.VStack(spacing=6):
            for result in display_results:
                url = result.get("url", "") if isinstance(result, dict) else ""
                if not url:
                    continue
                if self._is_asset_blacklisted(url):
                    continue
                identity_key = result.get("identity_key") if isinstance(result, dict) else None
                if identity_key and identity_key in excluded_keys:
                    continue

                shown += 1
                try:
                    score = float(result.get("score", 0) or 0)
                except (TypeError, ValueError):
                    score = 0.0
                with ui.HStack(height=row_height):
                    provider = self._build_search_test_provider(result.get("image_b64"))
                    if provider:
                        self._search_test_result_providers.append(provider)
                        ui.ImageWithProvider(
                            provider,
                            width=SEARCH_TEST_THUMB_SIZE,
                            height=SEARCH_TEST_THUMB_SIZE,
                        )
                    else:
                        ui.Label(
                            "No Image",
                            width=SEARCH_TEST_THUMB_SIZE,
                            height=SEARCH_TEST_THUMB_SIZE,
                        )

                    with ui.VStack(spacing=2, height=0, width=ui.Fraction(1)):
                        ui.Label(os.path.basename(url))
                        url_model = ui.SimpleStringModel(url)
                        ui.StringField(model=url_model, read_only=True, height=22, width=ui.Fraction(1))
                        if identity_key:
                            ui.Label(f"Key: {identity_key}", height=18)

                    ui.Label(f"{score:.2f}", width=60)
                    ui.Button(
                        "Blacklist",
                        clicked_fn=lambda u=url, k=identity_key: self._blacklist_search_result(u, k),
                        width=90,
                        height=24,
                    )

            if shown == 0:
                ui.Label("No results available after blacklist filtering.")

    def _rebuild_search_test_results(self) -> None:
        if not self._search_test_results_frame:
            return
        self._search_test_results_frame.rebuild()

    def _open_replacement_candidates_window(self) -> None:
        if self._replacement_window:
            existing = self._replacement_window
            self._replacement_window = None
            self._deferred_destroy_window(existing)

        context = omni.usd.get_context()
        selection = context.get_selection()
        prim_paths = selection.get_selected_prim_paths()
        if not prim_paths:
            omni.log.warn("No prim selected.")
            return

        prim_path = prim_paths[0]
        stage = context.get_stage()
        prim = stage.GetPrimAtPath(prim_path) if stage else None
        if not prim or not prim.IsValid():
            omni.log.warn(f"Invalid prim selected: {prim_path}")
            return

        ref_prim = self._get_reference_prim(prim)
        asset_url = self._get_asset_url_from_prim(ref_prim)
        if not asset_url:
            omni.log.warn(f"Selected prim has no asset reference: {prim_path}")
            return

        meta = self._get_search_metadata_from_prim(ref_prim)
        search_query = meta.get("search_query") or self._build_search_query_from_object(meta)
        if not search_query:
            omni.log.warn("Unable to determine search query for selected asset.")
            return

        self._replacement_prim_path = str(ref_prim.GetPath())
        self._replacement_asset_url = asset_url
        self._replacement_query = search_query

        self._replacement_window = ui.Window("Replacement Candidates", width=920, height=680)
        with self._replacement_window.frame:
            with ui.VStack(spacing=8, style={"margin": 10}):
                ui.Label("Replacement Candidates")
                ui.Label(f"Prim: {self._replacement_prim_path}", width=0)
                ui.Label(f"Query: {self._replacement_query}", width=0)
                ui.Label(f"Current asset: {self._replacement_asset_url}", width=0)

                with ui.HStack(height=26):
                    ui.Button("Refresh", clicked_fn=self._on_refresh_replacement_candidates, width=100)
                    self._replacement_status_label = ui.Label("Searching...", width=0)

                ui.Separator()

                with ui.ScrollingFrame(
                    height=480,
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                ):
                    self._replacement_results_frame = ui.Frame()
                    self._replacement_results_frame.set_build_fn(self._build_replacement_candidates_frame)

                ui.Spacer(height=6)
                ui.Button("Close", clicked_fn=self._close_replacement_candidates_window, width=120)

        self._replacement_results = []
        self._rebuild_replacement_candidates_frame()
        if self._replacement_task and not self._replacement_task.done():
            self._replacement_task.cancel()
        self._replacement_task = asyncio.ensure_future(
            self._run_replacement_candidates_search(self._replacement_query)
        )

    def _close_replacement_candidates_window(self) -> None:
        if not self._replacement_window:
            return
        existing = self._replacement_window
        self._replacement_window = None
        self._deferred_destroy_window(existing)
        if self._replacement_task and not self._replacement_task.done():
            self._replacement_task.cancel()
        self._replacement_task = None
        self._replacement_results_frame = None
        self._replacement_status_label = None
        self._replacement_result_providers = []

    def _set_replacement_status(self, message: str) -> None:
        if self._replacement_status_label:
            self._replacement_status_label.text = message

    def _on_refresh_replacement_candidates(self) -> None:
        query = str(getattr(self, "_replacement_query", "") or "").strip()
        if not query:
            self._set_replacement_status("Query is empty.")
            return

        if self._replacement_task and not self._replacement_task.done():
            self._replacement_task.cancel()

        self._set_replacement_status("Searching...")
        self._replacement_results = []
        self._rebuild_replacement_candidates_frame()
        self._replacement_task = asyncio.ensure_future(self._run_replacement_candidates_search(query))

    async def _run_replacement_candidates_search(self, query: str) -> None:
        try:
            results = await self._search_assets_for_query(query, target_valid=20, initial_limit=20, max_limit=200)
        except Exception as exc:
            omni.log.warn(f"Replacement candidate search failed: {exc}")
            self._set_replacement_status("Search failed.")
            return

        filtered = results or []
        root_note = ""
        search_root = self._search_root_model.as_string if self._search_root_model else ""
        search_root = search_root.strip()
        if search_root:
            normalized_root = search_root if search_root.endswith("/") else f"{search_root}/"
            root_filtered = [
                result
                for result in filtered
                if str(result.get("url", "")).startswith(normalized_root)
            ]
            if root_filtered:
                filtered = root_filtered
            else:
                root_note = " (Search Root had no matches)"

        self._replacement_results = filtered or []
        total_results = len(self._replacement_results)
        if total_results:
            self._set_replacement_status(f"Results: {total_results}{root_note}")
        else:
            self._set_replacement_status("No results.")
        self._rebuild_replacement_candidates_frame()

    def _on_replace_candidate_click(self, replacement_url: str) -> None:
        prim_path = self._replacement_prim_path or self._selected_prim_path
        if not prim_path:
            omni.log.warn("No prim target for replacement.")
            return

        async def _do_replace():
            self._set_replacement_status("Replacing...")
            try:
                await self._replace_selected_asset_with_url(prim_path, replacement_url, add_to_blacklist=False)
            except Exception as exc:  # pragma: no cover - defensive
                omni.log.warn(f"Replacement failed: {exc}")
                self._set_replacement_status("Replace failed.")
                return

            self._replacement_asset_url = replacement_url
            self._asset_url_model.set_value(replacement_url)
            offset = self._get_asset_rotation_offset(replacement_url)
            self._asset_offset_model.set_value(str(offset))
            self._set_replacement_status("Replaced.")
            self._rebuild_replacement_candidates_frame()

        asyncio.ensure_future(_do_replace())

    def _build_replacement_candidates_frame(self):
        self._replacement_result_providers = []
        results = self._replacement_results or []
        if not results:
            ui.Label("No results yet.")
            return

        excluded_keys = set(getattr(self, "_asset_blacklist_keys", set()))
        current_norm = self._normalize_asset_url(self._replacement_asset_url) if self._replacement_asset_url else ""
        row_height = SEARCH_TEST_THUMB_SIZE + 36
        shown = 0
        with ui.VStack(spacing=6):
            for idx, result in enumerate(results, start=1):
                url = result.get("url", "") if isinstance(result, dict) else ""
                if not url:
                    continue
                identity_key = result.get("identity_key") if isinstance(result, dict) else None
                is_blacklisted = False
                if self._is_asset_blacklisted(url):
                    is_blacklisted = True
                if identity_key and identity_key in excluded_keys:
                    is_blacklisted = True

                try:
                    score = float(result.get("score", 0) or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                is_current = self._normalize_asset_url(url) == current_norm if current_norm else False
                label_text = f"#{idx} {url}"
                if is_current:
                    label_text += " (current)"
                if is_blacklisted:
                    label_text += " [blacklisted]"

                shown += 1
                with ui.HStack(height=row_height, spacing=8):
                    with ui.VStack(width=SEARCH_TEST_THUMB_SIZE, spacing=4):
                        provider = self._build_search_test_provider(result.get("image_b64"))
                        if provider:
                            self._replacement_result_providers.append(provider)
                            ui.ImageWithProvider(
                                provider,
                                width=SEARCH_TEST_THUMB_SIZE,
                                height=SEARCH_TEST_THUMB_SIZE,
                            )
                        else:
                            ui.Label(
                                "No Image",
                                width=SEARCH_TEST_THUMB_SIZE,
                                height=SEARCH_TEST_THUMB_SIZE,
                            )
                        ui.Button(
                            "Replace",
                            clicked_fn=lambda u=url: self._on_replace_candidate_click(u),
                            width=SEARCH_TEST_THUMB_SIZE,
                        )

                    with ui.VStack(width=ui.Fraction(1)):
                        ui.Label(label_text, width=0)
                        ui.Label(f"Score: {score:.4f}", width=0)

            if shown == 0:
                ui.Label("No displayable results (all filtered or missing URLs).", width=0)

    def _rebuild_replacement_candidates_frame(self) -> None:
        if not self._replacement_results_frame:
            return
        self._replacement_results_frame.rebuild()

    def _switch_tab(self, tab_index: int):
        """繧ｿ繝悶ｒ蛻・ｊ譖ｿ縺医ｋ"""
        self._active_tab = tab_index
        self._ai_status_label = None
        self._ai_tokens_label = None
        self._generate_button = None
        self._cancel_ai_button = None
        self._json_size_mode_combo = None
        self._tab_container.clear()
        with self._tab_container:
            self._build_tab_content()

    def _build_tab_content(self):
        """迴ｾ蝨ｨ縺ｮ繧｢繧ｯ繝・ぅ繝悶ち繝悶・繧ｳ繝ｳ繝・Φ繝・ｒ讒狗ｯ峨☆繧・"""
        if self._active_tab == 0:
            # 繧ｿ繝・1: "Generate from Image"
            with ui.VStack(spacing=5, height=0):
                ui.Label("Inputs")

                # 逕ｻ蜒上/蟇ｸ豕輔ヵ繧｡繧､繝ｫ驕ｸ謚・
                with ui.HStack(height=25, spacing=12):
                    with ui.HStack(width=ui.Fraction(1), spacing=6):
                        ui.Button("Select Image File...", clicked_fn=self._on_select_image_click, width=150)
                        image_label_text = os.path.basename(self._image_path) if self._image_path else "No file selected"
                        self._image_label = ui.Label(image_label_text, width=0)
                    with ui.HStack(width=ui.Fraction(1), spacing=6):
                        ui.Button("Select Dimensions File...", clicked_fn=self._on_select_dims_click, width=170)
                        dims_label_text = (
                            os.path.basename(self._dimensions_path) if self._dimensions_path else "No file selected"
                        )
                        self._dims_label = ui.Label(dims_label_text, width=0)

                # 繝励Ο繝ｳ繝励ヨ驕ｸ謚橸ｼ医が繝励す繝ｧ繝ｳ・・
                with ui.HStack(height=25, spacing=12):
                    with ui.HStack(width=ui.Fraction(1), spacing=6):
                        ui.Button("Select Prompt 1 (optional)...", clicked_fn=self._on_select_prompt1_click, width=170)
                        prompt1_label_text = (
                            os.path.basename(self._prompt1_path) if self._prompt1_path else "Using default"
                        )
                        self._prompt1_label = ui.Label(prompt1_label_text, width=0)
                    with ui.HStack(width=ui.Fraction(1), spacing=6):
                        ui.Button("Select Prompt 2 (optional)...", clicked_fn=self._on_select_prompt2_click, width=170)
                        prompt2_label_text = (
                            os.path.basename(self._prompt2_path) if self._prompt2_path else "Using default"
                        )
                        self._prompt2_label = ui.Label(prompt2_label_text, width=0)

                with ui.HStack(height=25):
                    ui.Button("Preview Inputs...", clicked_fn=self._open_preview_window, width=150)
                    ui.Label("Preview selected image/prompts/dimensions", width=0)

                ui.Spacer(height=8)
                ui.Separator()
                ui.Label("AI Generation & Search")

                # 繝｢繝・Ν驕ｸ謚・
                # 繧､繝ｳ繝・ャ繧ｯ繧ｹ縺ｮ遽・峇繝√ぉ繝・け・・ODEL_CHOICES縺ｮ遽・峇蜀・ｼ・
                model_index = self._saved_model_index if 0 <= self._saved_model_index < len(MODEL_CHOICES) else 0
                with ui.HStack(height=25):
                    ui.Label("AI Model:", width=150)
                    self._model_combo = ui.ComboBox(model_index, *MODEL_CHOICES)
                    # 繝｢繝・Ν螟画峩譎ゅ↓JSON繝輔ぃ繧､繝ｫ縺ｫ菫晏ｭ・
                    self._model_combo.model.get_item_value_model().add_value_changed_fn(
                        lambda m: self._save_settings_to_json()
                    )
                    ui.Button("AI Settings...", clicked_fn=self._open_ai_settings_window, width=120)

                # API繧ｭ繝ｼ蜈･蜉・
                with ui.HStack(height=25):
                    ui.Label("OpenAI API Key:", width=150)
                    ui.StringField(model=self._api_key_model, password_mode=True)

                # USD Search 繝ｫ繝ｼ繝・
                with ui.HStack(height=25):
                    ui.Label("Search Root URL:", width=150)
                    ui.StringField(model=self._search_root_model)

                with ui.HStack(height=25):
                    ui.Button("Search Tester...", clicked_fn=self._open_search_test_window, width=150)
                    ui.Label("Search preview", width=0)

                self._build_asset_orientation_section(show_attach=True, show_replace_candidates=True)

                ui.Spacer(height=10)

                with ui.HStack(height=26, spacing=4):
                    self._generate_button = ui.Button(
                        "Generate JSON & Scene (AI)",
                        clicked_fn=self._on_generate_json_click,
                        width=220,
                        height=26,
                    )
                    with ui.HStack(spacing=2, width=145):
                        size_mode_index = getattr(self, "_json_size_mode_index", 1)
                        if size_mode_index < 0 or size_mode_index >= len(JSON_SIZE_MODE_CHOICES):
                            size_mode_index = 1 if len(JSON_SIZE_MODE_CHOICES) > 1 else 0
                        ui.Label("size_mode:", width=60)
                        self._json_size_mode_combo = ui.ComboBox(
                            size_mode_index, *JSON_SIZE_MODE_CHOICES, width=80
                        )
                        self._json_size_mode_combo.model.get_item_value_model().add_value_changed_fn(
                            lambda m: self._save_settings_to_json()
                        )
                    ui.Spacer(width=10)
                    ui.CheckBox(model=self._require_approval, width=18)
                    ui.Label("Require approval after image analysis", width=0)
                with ui.HStack(height=24):
                    self._cancel_ai_button = ui.Button("Cancel AI", clicked_fn=self._on_cancel_ai_click, width=120, height=24)
                    self._cancel_ai_button.visible = False
                    status_text = getattr(self, "_ai_status_text", "AI Status: Idle")
                    self._ai_status_label = ui.Label(status_text, width=0)
                with ui.HStack(height=20):
                    tokens_text = getattr(self, "_ai_tokens_text", "Tokens: -")
                    self._ai_tokens_label = ui.Label(tokens_text, width=0)
                ui.Separator()
                ui.Label("JSON Files")
                ui.Label("Generated JSON")
                with ui.HStack(height=25):
                    ui.Button("Preview Generated JSON...", clicked_fn=self._open_generated_json_preview_window, width=190)
                    label_text = (
                        f"Latest JSON: {os.path.basename(self._generated_json_path)}"
                        if self._generated_json_path
                        else "Latest JSON: None"
                    )
                    self._generated_json_label = ui.Label(label_text, width=0)

                ui.Spacer(height=6)
                ui.Label("Selected JSON")
                with ui.HStack(height=25):
                    ui.Button("Select JSON File...", clicked_fn=self._on_load_json_click, width=190)
                    loaded_label_text = (
                        f"Selected JSON: {os.path.basename(self._loaded_json_path)}"
                        if self._loaded_json_path
                        else "Selected JSON: None"
                    )
                    self._loaded_json_label = ui.Label(loaded_label_text, width=0)
                with ui.HStack(height=26):
                    ui.Button("Preview Selected JSON...", clicked_fn=self._open_loaded_json_preview_window, width=190)
                    ui.Button("Edit Selected JSON...", clicked_fn=self._open_loaded_json_editor, width=190)
                    ui.Label("Preview or edit selected JSON", width=0)
                ui.Spacer(height=4)
                ui.Label("Placement (Selected JSON)")
                with ui.HStack(height=25):
                    ui.Button("Generate Scene from JSON", clicked_fn=self._on_generate_from_json_click, width=190)
                    ui.Label("Generate scene using selected JSON", width=0)
                with ui.HStack(height=25):
                    ui.Button("Place BBoxes from JSON (Debug)", clicked_fn=self._on_place_bbox_from_json_click, width=190)
                    ui.Label("Place debug bboxes using selected JSON", width=0)

                ui.Spacer(height=10)

                # --- LLM蛻・梵邨先棡陦ｨ遉ｺ谺・---
                ui.Label("AI Analysis Result:")
                self._analysis_text_field = ui.StringField(
                    model=self._analysis_text_model,
                    multiline=True,
                    read_only=True,
                    height=300,
                    style=self._japanese_font_style  # 譌･譛ｬ隱槭ヵ繧ｩ繝ｳ繝磯←逕ｨ
                )

                ui.Spacer(height=5)
                ui.Spacer()  # 繝懊ち繝ｳ繧ｨ繝ｪ繧｢繧剃ｸ矩Κ縺ｫ蝗ｺ螳・
                ui.Separator()

                # 謇ｿ隱・諡貞凄繝懊ち繝ｳ・域怙蛻昴・髱櫁｡ｨ遉ｺ・・
                with ui.HStack(height=34, visible=False) as self._approval_buttons_container:
                    ui.Button("笨・Approve & Continue", clicked_fn=self._on_approve_click, width=0, height=30)
                    ui.Spacer(width=10)
                    ui.Button("笨・Reject & Add Context", clicked_fn=self._on_reject_click, width=0, height=30)

                ui.Spacer(height=5)

    # --- "Load from File" 繧ｿ繝悶・繧ｳ繝ｼ繝ｫ繝舌ャ繧ｯ ---

    def _open_ai_settings_window(self):
        if self._ai_settings_window:
            existing = self._ai_settings_window
            self._ai_settings_window = None
            self._deferred_destroy_window(existing)

        self._ai_settings_window = ui.Window("AI Advanced Settings", width=560, height=580)
        with self._ai_settings_window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=8, style={"margin": 10}):
                    ui.Label("AI Advanced Settings")

                    step1_index = self._ai_step1_model_index
                    if step1_index < 0 or step1_index >= len(ADV_MODEL_CHOICES):
                        step1_index = 0
                    step2_index = self._ai_step2_model_index
                    if step2_index < 0 or step2_index >= len(ADV_MODEL_CHOICES):
                        step2_index = 0
                    effort_index = self._ai_reasoning_effort_index
                    if effort_index < 0 or effort_index >= len(REASONING_EFFORT_CHOICES):
                        effort_index = 0
                    verbosity_index = self._ai_text_verbosity_index
                    if verbosity_index < 0 or verbosity_index >= len(TEXT_VERBOSITY_CHOICES):
                        verbosity_index = 0
                    detail_index = self._ai_image_detail_index
                    if detail_index < 0 or detail_index >= len(IMAGE_DETAIL_CHOICES):
                        detail_index = 0

                    with ui.HStack(height=25):
                        ui.Label("Step 1 Model:", width=180)
                        self._ai_step1_model_combo = ui.ComboBox(step1_index, *ADV_MODEL_CHOICES)
                        self._ai_step1_model_combo.model.get_item_value_model().add_value_changed_fn(
                            lambda m: self._save_settings_to_json()
                        )

                    with ui.HStack(height=25):
                        ui.Label("Step 2 Model:", width=180)
                        self._ai_step2_model_combo = ui.ComboBox(step2_index, *ADV_MODEL_CHOICES)
                        self._ai_step2_model_combo.model.get_item_value_model().add_value_changed_fn(
                            lambda m: self._save_settings_to_json()
                        )

                    with ui.HStack(height=25):
                        ui.Label("Reasoning Effort:", width=180)
                        self._ai_reasoning_effort_combo = ui.ComboBox(effort_index, *REASONING_EFFORT_CHOICES)
                        self._ai_reasoning_effort_combo.model.get_item_value_model().add_value_changed_fn(
                            lambda m: self._save_settings_to_json()
                        )

                    with ui.HStack(height=25):
                        ui.Label("Text Verbosity:", width=180)
                        self._ai_text_verbosity_combo = ui.ComboBox(verbosity_index, *TEXT_VERBOSITY_CHOICES)
                        self._ai_text_verbosity_combo.model.get_item_value_model().add_value_changed_fn(
                            lambda m: self._save_settings_to_json()
                        )

                    with ui.HStack(height=25):
                        ui.Label("Image Detail:", width=180)
                        self._ai_image_detail_combo = ui.ComboBox(detail_index, *IMAGE_DETAIL_CHOICES)
                        self._ai_image_detail_combo.model.get_item_value_model().add_value_changed_fn(
                            lambda m: self._save_settings_to_json()
                        )

                    with ui.HStack(height=25):
                        ui.Label("Max Output Tokens:", width=180)
                        ui.StringField(model=self._ai_max_output_tokens_model)
                        self._ai_max_output_tokens_model.add_value_changed_fn(
                            lambda m: self._save_settings_to_json()
                        )

                    with ui.HStack(height=25):
                        ui.Label("Max Retries:", width=180)
                        ui.StringField(model=self._ai_max_retries_model)
                        self._ai_max_retries_model.add_value_changed_fn(
                            lambda m: self._save_settings_to_json()
                        )

                    with ui.HStack(height=25):
                        ui.Label("Retry Delay (sec):", width=180)
                        ui.StringField(model=self._ai_retry_delay_model)
                        self._ai_retry_delay_model.add_value_changed_fn(
                            lambda m: self._save_settings_to_json()
                        )

                    ui.Spacer(height=8)
                    ui.Button("Close", clicked_fn=self._close_ai_settings_window, width=120)

    def _close_ai_settings_window(self):
        if not self._ai_settings_window:
            return
        existing = self._ai_settings_window
        self._ai_settings_window = None
        self._deferred_destroy_window(existing)
        self._ai_step1_model_combo = None
        self._ai_step2_model_combo = None
        self._ai_reasoning_effort_combo = None
        self._ai_text_verbosity_combo = None
        self._ai_image_detail_combo = None

    def _build_asset_orientation_section(
        self,
        show_attach: bool = False,
        show_replace_candidates: bool = False,
    ):
        ui.Separator()
        ui.Label("Asset Orientation Offset (per asset_url)")
        with ui.HStack(height=25, width=ui.Fraction(1)):
            ui.Button("Use Selected Prim", clicked_fn=self._on_use_selected_prim_click, width=150)
            ui.StringField(model=self._asset_url_model, read_only=True, width=ui.Fraction(1))
        if show_attach:
            with ui.HStack(height=25):
                ui.Button("Attach Metadata", clicked_fn=self._on_attach_metadata_selected_prim_click, width=150)
                ui.Label("Attach placement metadata to selected prim", width=0)
        with ui.HStack(height=25):
            ui.Label("Rotation Offset (deg):", width=150)
            ui.StringField(model=self._asset_offset_model)
        with ui.HStack(height=25):
            ui.Button("-90", clicked_fn=lambda: self._nudge_asset_offset(-90), width=60)
            ui.Button("+90", clicked_fn=lambda: self._nudge_asset_offset(90), width=60)
            ui.Button("+180", clicked_fn=lambda: self._nudge_asset_offset(180), width=70)
            ui.Button("Save & Apply", clicked_fn=self._on_save_asset_offset_click, width=120)

        with ui.HStack(height=25):
            ui.Button("Replace Only", clicked_fn=self._on_replace_selected_asset_click, width=150)
            ui.Label("Replace without blacklist", width=0)

        if show_replace_candidates:
            with ui.HStack(height=25):
                ui.Button(
                    "Show Candidates...",
                    clicked_fn=self._open_replacement_candidates_window,
                    width=150,
                )
                ui.Label("Browse ranked replacements with thumbnails", width=0)

        with ui.HStack(height=25):
            ui.Button("Blacklist & Replace", clicked_fn=self._on_blacklist_selected_asset_click, width=150)
            self._blacklist_count_label = ui.Label("Blacklisted: 0", width=0)
            ui.Button("Manage...", clicked_fn=self._open_blacklist_window, width=90)
        self._update_blacklist_label()

