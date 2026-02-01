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
import os
from typing import Dict, Optional

import carb
import omni.ext
import omni.log
import omni.ui as ui

from .commands import CommandsMixin
from .constants import (
    ADV_MODEL_CHOICES,
    DEFAULT_PROMPT1_TEXT,
    DEFAULT_PROMPT2_TEXT,
    IMAGE_DETAIL_CHOICES,
    MODEL_CHOICES,
    REASONING_EFFORT_CHOICES,
    TEXT_VERBOSITY_CHOICES,
    VECTOR_SEARCH_LIMIT,
)
from .handlers import HandlersMixin
from .settings import SettingsMixin
from .state import StateMixin
from .ui import UIMixin
from ..procedural import DoorDetector, FloorGenerator, WallGenerator


# Functions and vars are available to other extensions as usual in python:
# `my.research.asset_placer.some_public_function(x)`
def some_public_function(x: int):
    """This is a public function that can be called from other extensions."""
    omni.log.info(
        f"[my.research.asset_placer] some_public_function was called with {x}"
    )
    return x**x


class MyExtension(
    SettingsMixin, StateMixin, UIMixin, HandlersMixin, CommandsMixin, omni.ext.IExt
):
    """USD Search Placer Extension with Tabbed UI."""

    def on_startup(self, _ext_id):
        omni.log.info("[my.research.asset_placer] Extension startup")

        # --- 設定ファイルのパス ---
        self._extension_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        self._settings_file = os.path.join(
            self._extension_dir, "extension_settings.json"
        )

        # --- JSONファイルから設定を読み込む ---
        saved_settings = self._load_settings_from_json()

        # --- APIキーの永続化（JSON優先） ---
        # 環境変数からAPIキーを取得（JSONより低優先）
        env_api_key = os.environ.get("OPENAI_API_KEY", "")

        # JSONファイル > 環境変数 の優先順位で読み込み
        saved_key = saved_settings.get("openai_api_key", "")
        initial_api_key = saved_key if saved_key else env_api_key

        self._api_key_model = ui.SimpleStringModel(initial_api_key)
        self._api_key_model.add_value_changed_fn(
            lambda m: self._save_settings_to_json()
        )

        # どこから読み込んだかログに記録
        if saved_key:
            omni.log.info("OpenAI API Key loaded from extension_settings.json")
        elif env_api_key:
            omni.log.info(
                "OpenAI API Key loaded from OPENAI_API_KEY environment variable"
            )

        # --- Search Root URLの永続化（JSONファイル使用） ---
        self._search_root_model = ui.SimpleStringModel(
            saved_settings.get(
                "search_root_url",
                "omniverse://192.168.11.65/Users/tsukuba1/MyAssets/Office/",
            )
        )
        self._search_root_model.add_value_changed_fn(
            lambda m: self._save_settings_to_json()
        )

        # --- ファイルパスの永続化（JSONファイル使用） ---
        self._image_path = saved_settings.get("image_path", "")
        self._dimensions_path = saved_settings.get("dimensions_path", "")
        self._prompt1_path = saved_settings.get("prompt1_path", "")
        self._prompt2_path = saved_settings.get("prompt2_path", "")
        self._json_output_dir = saved_settings.get("json_output_dir", "")
        self._saved_model_index = saved_settings.get("model_index", 0)
        self._ai_step1_model_index = saved_settings.get("ai_step1_model_index", 0)
        self._ai_step2_model_index = saved_settings.get("ai_step2_model_index", 0)
        self._ai_reasoning_effort_index = saved_settings.get(
            "ai_reasoning_effort_index", 0
        )
        self._ai_text_verbosity_index = saved_settings.get("ai_text_verbosity_index", 0)
        self._ai_image_detail_index = saved_settings.get("ai_image_detail_index", 0)
        self._ai_max_output_tokens = saved_settings.get("ai_max_output_tokens", 16000)
        self._ai_settings_window = None
        self._ai_max_output_tokens_model = ui.SimpleStringModel(
            str(self._ai_max_output_tokens)
        )
        self._ai_max_retries = saved_settings.get("ai_max_retries", 1)
        self._ai_retry_delay_sec = saved_settings.get("ai_retry_delay_sec", 1.0)
        self._ai_max_retries_model = ui.SimpleStringModel(str(self._ai_max_retries))
        self._ai_retry_delay_model = ui.SimpleStringModel(str(self._ai_retry_delay_sec))

        # --- Auto layout load (startup) ---
        self._auto_layout_enabled = saved_settings.get("auto_layout_enabled", True)
        self._auto_layout_mode = saved_settings.get("auto_layout_mode", "quick")
        self._auto_layout_file = saved_settings.get("auto_layout_file", "")
        self._auto_layout_task: Optional[asyncio.Task] = None

        # --- Asset rotation offset cache (per asset_url) ---
        self._rotation_offsets_file = os.path.join(
            self._extension_dir, "asset_rotation_offsets.json"
        )
        self._asset_rotation_offsets = self._load_rotation_offsets()
        self._asset_url_model = ui.SimpleStringModel("")
        self._asset_offset_model = ui.SimpleStringModel("0")
        self._selected_prim_path = ""
        raw_blacklist = saved_settings.get("asset_blacklist", [])
        if isinstance(raw_blacklist, (str, bytes)):
            raw_blacklist = [raw_blacklist]
        if not isinstance(raw_blacklist, (list, tuple, set)):
            raw_blacklist = []
        self._asset_blacklist = {
            self._normalize_asset_url(str(url)) for url in raw_blacklist if url
        }
        raw_blacklist_keys = saved_settings.get("asset_blacklist_keys", [])
        if isinstance(raw_blacklist_keys, (str, bytes)):
            raw_blacklist_keys = [raw_blacklist_keys]
        if not isinstance(raw_blacklist_keys, (list, tuple, set)):
            raw_blacklist_keys = []
        self._asset_blacklist_keys = {str(key) for key in raw_blacklist_keys if key}
        self._asset_size_cache = {}
        self._replacement_history_by_prim: Dict[str, Dict[str, list]] = {}
        self._replacement_history_limit = 20
        self._blacklist_count_label = None
        self._blacklist_window = None
        self._blacklist_list_container = None
        self._blacklist_window_count_label = None
        self._blacklist_scrolling_frame = None

        # --- LLM分析結果表示用モデル ---
        self._analysis_text_model = ui.SimpleStringModel("")
        self._ai_status_text = "AI Status: Idle"
        self._ai_tokens_text = "Tokens: -"
        self._ai_status_label = None
        self._ai_tokens_label = None
        self._generate_button = None
        self._cancel_ai_button = None
        self._ai_task: Optional[asyncio.Task] = None

        # --- 生成JSONプレビュー用モデル ---
        self._generated_json_path = ""
        self._generated_json_preview_window = None
        self._generated_json_preview_model = ui.SimpleStringModel("")
        self._generated_json_label = None
        self._loaded_json_path = ""
        self._loaded_json_preview_window = None
        self._loaded_json_preview_model = ui.SimpleStringModel("")

        # --- プレビュー用モデル ---
        self._preview_window = None
        self._image_preview_widget = None
        self._image_preview_frame = None
        self._image_preview_provider = ui.ByteImageProvider()
        self._image_preview_label = None
        self._prompt1_preview_label = None
        self._prompt2_preview_label = None
        self._dims_preview_label = None
        self._prompt1_preview_model = ui.SimpleStringModel("")
        self._prompt2_preview_model = ui.SimpleStringModel("")
        self._dims_preview_model = ui.SimpleStringModel("")

        # --- JSON editor window ---
        self._json_editor_window = None
        self._json_editor_file_combo = None
        self._json_editor_files = []
        self._json_editor_selected_path = ""
        self._json_editor_data = None
        self._json_editor_assets = []
        self._json_editor_assets_key = None
        self._json_editor_asset_models = {}
        self._json_editor_assets_frame = None

        # --- USD search test window ---
        self._search_test_window = None
        self._search_test_query_model = ui.SimpleStringModel("")
        self._search_test_status_label = None
        self._search_test_results_frame = None
        self._search_test_results = []
        self._search_test_result_providers = []
        self._search_test_task: Optional[asyncio.Task] = None

        # --- Replacement candidates window ---
        self._replacement_window = None
        self._replacement_status_label = None
        self._replacement_results_frame = None
        self._replacement_results = []
        self._replacement_result_providers = []
        self._replacement_task: Optional[asyncio.Task] = None
        self._replacement_prim_path = ""
        self._replacement_asset_url = ""
        self._replacement_query = ""

        # --- 承認ワークフロー用の状態管理 ---
        self._require_approval = ui.SimpleBoolModel(False)  # 承認ステップを挟むかどうか
        self._analysis_result = None  # Step 1の分析結果を保持
        self._approval_pending = False  # 承認待ち状態
        self._additional_context = ""  # 拒否時の追加コンテキスト
        self._context_popup = None  # 修正指示ポップアップウィンドウ

        self._layout_json = None  # 最終的に使用するJSONデータを保持する変数
        self._file_picker = None
        self._active_tab = 0  # 0: Generate from Image, 1: Details
        self._search_task: Optional[asyncio.Task] = None

        # アセットURLごとのupAxisキャッシュ
        self._asset_up_axis_cache: Dict[str, str] = {}

        # --- 日本語フォントの設定 ---
        self._japanese_font_style = self._setup_japanese_font()

        # --- 手続き的生成モジュールの初期化 ---
        self._door_detector = DoorDetector(
            self._extract_float, self._extract_optional_float
        )
        self._floor_generator = FloorGenerator(
            self._extract_float, self._get_unique_child_path
        )
        self._wall_generator = WallGenerator(
            self._extract_float,
            self._extract_optional_float,
            self._get_unique_child_path,
            self._door_detector,
        )

        # --- ウィンドウの作成 ---
        self._window = ui.Window("USD Search Placer", width=500, height=650)

        with self._window.frame:
            # メインの縦積みレイアウト
            with ui.VStack(spacing=5):
                # タブボタン
                with ui.HStack(height=30):
                    self._tab_btn_generate = ui.Button(
                        "Generate from Image", clicked_fn=lambda: self._switch_tab(0)
                    )
                    self._tab_btn_details = ui.Button(
                        "Details", clicked_fn=lambda: self._switch_tab(1)
                    )

                ui.Separator()

                # タブコンテンツ用のコンテナ
                self._tab_container = ui.VStack(spacing=5, height=0)
                with self._tab_container:
                    self._build_tab_content()

                # メインVStackの最後にスペーサーを追加（これがないとUI全体が下に寄る）
                ui.Spacer()

        self._refresh_preview_models()
        self._apply_default_view_settings()

        if self._auto_layout_enabled:
            self._auto_layout_task = asyncio.ensure_future(self._auto_load_layout())

    def _apply_default_view_settings(self) -> None:
        settings = carb.settings.get_settings()
        if not settings:
            return
        settings.set("/app/viewport/grid/enabled", True)
        settings.set("/persistent/app/viewport/grid/enabled", True)
        settings.set(
            "/exts/omni.kit.viewport.menubar.lighting/defaultRig", "Grey_Studio"
        )

        try:
            import omni.kit.actions.core

            action_registry = omni.kit.actions.core.get_action_registry()
            action_registry.execute_action(
                "omni.kit.viewport.menubar.lighting",
                "set_lighting_mode_rig",
                "Grey_Studio",
            )
        except Exception as exc:
            omni.log.warn(f"Failed to apply Grey Studio lighting rig: {exc}")

    def on_shutdown(self):
        omni.log.info("[my.research.asset_placer] Extension shutdown")
        if self._ai_task and not self._ai_task.done():
            self._ai_task.cancel()
        self._ai_task = None
        if self._search_task and not self._search_task.done():
            self._search_task.cancel()
        self._search_task = None
        if self._search_test_task and not self._search_test_task.done():
            self._search_test_task.cancel()
        self._search_test_task = None
        if self._replacement_task and not self._replacement_task.done():
            self._replacement_task.cancel()
        self._replacement_task = None
        if self._auto_layout_task and not self._auto_layout_task.done():
            self._auto_layout_task.cancel()
        self._auto_layout_task = None
        if hasattr(self, "_context_popup") and self._context_popup:
            self._context_popup.destroy()
            self._context_popup = None
        if self._window:
            self._window.destroy()
            self._window = None
        if self._ai_settings_window:
            self._ai_settings_window.destroy()
            self._ai_settings_window = None
        if self._preview_window:
            self._preview_window.destroy()
            self._preview_window = None
        if self._blacklist_window:
            self._blacklist_window.destroy()
            self._blacklist_window = None

    async def _auto_load_layout(self) -> None:
        """起動時にレイアウトを自動復元する。"""
        try:
            from omni.kit.quicklayout import QuickLayout
        except Exception as exc:  # pragma: no cover - optional dependency
            omni.log.warn(f"QuickLayout is not available: {exc}")
            return

        try:
            app = omni.kit.app.get_app()
            for _ in range(3):
                await app.next_update_async()

            mode = str(self._auto_layout_mode or "quick").lower()
            if mode == "file":
                layout_file = str(self._auto_layout_file or "").strip()
                if not layout_file:
                    omni.log.warn("Auto layout enabled but no layout file set.")
                    return
                if not os.path.exists(layout_file):
                    omni.log.warn(f"Layout file not found: {layout_file}")
                    return
                QuickLayout.load_file(layout_file, keep_windows_open=False)
                omni.log.info(f"Loaded layout file: {layout_file}")
            else:
                QuickLayout.quick_load(None, None)
                omni.log.info("Loaded quick layout.")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            omni.log.warn(f"Auto layout load failed: {exc}")
        if self._generated_json_preview_window:
            self._generated_json_preview_window.destroy()
            self._generated_json_preview_window = None
        if self._loaded_json_preview_window:
            self._loaded_json_preview_window.destroy()
            self._loaded_json_preview_window = None
        if self._search_test_window:
            self._search_test_window.destroy()
            self._search_test_window = None
        if self._replacement_window:
            self._replacement_window.destroy()
            self._replacement_window = None
        if self._file_picker:
            self._deferred_destroy_picker()
