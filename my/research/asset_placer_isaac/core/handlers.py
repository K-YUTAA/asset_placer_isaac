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
from typing import Optional

import omni.kit.app
import omni.kit.window.filepicker as fp
import omni.log
import omni.ui as ui
import omni.usd


class HandlersMixin:
    def _deferred_destroy_picker(self):
        """destroys the file picker safely on the next UI update frame."""
        if not self._file_picker:
            return
        dialog = self._file_picker
        self._file_picker = None
        app = omni.kit.app.get_app()

        def _destroy_dialog():
            try:
                dialog.destroy()
            except Exception as exc:  # pragma: no cover - defensive
                omni.log.warn(f"Failed to destroy file picker dialog: {exc}")

        if hasattr(app, "next_update_async"):
            async def _wait_then_destroy():
                try:
                    await app.next_update_async()
                except Exception as exc:  # pragma: no cover - defensive
                    omni.log.warn(f"next_update_async failed: {exc}")
                _destroy_dialog()

            asyncio.ensure_future(_wait_then_destroy())
            return

        if hasattr(app, "next_update"):
            def _on_next_update(_dt):
                _destroy_dialog()

            app.next_update(_on_next_update)
            return

        asyncio.get_event_loop().call_soon(_destroy_dialog)

    def _on_use_selected_prim_click(self):
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

        asset_url = self._get_asset_url_from_prim(prim)
        if not asset_url:
            omni.log.warn(f"Selected prim has no asset reference: {prim_path}")
            return

        self._selected_prim_path = prim_path
        self._asset_url_model.set_value(asset_url)
        offset = self._get_asset_rotation_offset(asset_url)
        self._asset_offset_model.set_value(str(offset))

    def _nudge_asset_offset(self, delta: int):
        current = self._parse_rotation_offset(self._asset_offset_model.as_string)
        new_value = (current + delta) % 360
        self._asset_offset_model.set_value(str(new_value))

    def _on_save_asset_offset_click(self):
        asset_url = self._asset_url_model.as_string.strip()
        if not asset_url:
            omni.log.warn("Asset URL is empty. Use 'Use Selected Prim' first.")
            return

        offset = self._parse_rotation_offset(self._asset_offset_model.as_string)
        self._asset_rotation_offsets[asset_url] = offset
        self._save_rotation_offsets()
        omni.log.info(f"Saved rotation offset {offset} for asset: {asset_url}")

        stage = omni.usd.get_context().get_stage()
        if not stage:
            omni.log.warn("No USD stage available. Rotation offset saved but not applied.")
            return

        asyncio.ensure_future(self._apply_rotation_offset_to_matching_assets(stage, asset_url))

    def _on_attach_metadata_selected_prim_click(self):
        context = omni.usd.get_context()
        selection = context.get_selection()
        prim_paths = selection.get_selected_prim_paths()
        if not prim_paths:
            omni.log.warn("No prim selected for metadata attachment.")
            return

        prim_path = prim_paths[0]
        stage = context.get_stage()
        prim = stage.GetPrimAtPath(prim_path) if stage else None
        if not prim or not prim.IsValid():
            omni.log.warn(f"Invalid prim selected: {prim_path}")
            return

        asset_url = self._get_asset_url_from_prim(prim)
        if not asset_url:
            omni.log.warn(f"Selected prim has no asset reference: {prim_path}")

        if self._attach_placement_metadata_from_prim(prim, asset_url):
            omni.log.info(f"Placement metadata attached for '{prim_path}'")
        else:
            omni.log.warn(f"Failed to attach placement metadata for '{prim_path}'")

    def _on_blacklist_selected_asset_click(self):
        context = omni.usd.get_context()
        selection = context.get_selection()
        prim_paths = selection.get_selected_prim_paths()
        if not prim_paths:
            omni.log.warn("No prim selected for blacklisting.")
            return

        prim_path = prim_paths[0]
        asyncio.ensure_future(self._blacklist_and_replace_selected_asset(prim_path))

    def _on_replace_selected_asset_click(self):
        context = omni.usd.get_context()
        selection = context.get_selection()
        prim_paths = selection.get_selected_prim_paths()
        if not prim_paths:
            omni.log.warn("No prim selected for replacement.")
            return

        prim_path = prim_paths[0]
        asyncio.ensure_future(self._replace_selected_asset(prim_path))

    def _on_load_json_click(self):
        """ "Select JSON File..." ボタンがクリックされた (旧 _on_button_click) """
        omni.log.info("Select JSON button clicked. Showing File Picker...")

        if self._file_picker:
            self._deferred_destroy_picker()  # 毎回新しく作る

        def json_handler(*args, **kwargs):
            omni.log.info(f"JSON handler called with args={args}, kwargs={kwargs}")
            if len(args) == 2:
                filename, dirname = args[0], args[1]
                full_path = os.path.join(dirname, filename)
                selections = [full_path]
            else:
                omni.log.error(f"Unexpected arguments: args={args}")
                return
            self._on_file_picker_selection(selections)

        self._file_picker = fp.FilePickerDialog(
            "Select JSON File",
            allow_multi_selection=False,
            apply_button_label="Open",
            file_extension_options=[("*.json", "JSON File")],
            click_apply_handler=json_handler,
            click_cancel_handler=lambda *args: self._deferred_destroy_picker(),
        )
        self._file_picker.show()

    def _on_file_picker_selection(self, selections: list):
        """ファイルピッカーでJSONが選択された"""
        if self._file_picker:
            self._deferred_destroy_picker()

        if not selections or len(selections) == 0:
            omni.log.info("No file selected. Canceled.")
            return

        filepath = selections[0]
        omni.log.info(f"Selected file: {filepath}")
        self._loaded_json_path = filepath
        if hasattr(self, "_loaded_json_label") and self._loaded_json_label:
            self._loaded_json_label.text = f"Selected JSON: {os.path.basename(filepath)}"

        layout_data = self._load_json_with_fallback(filepath)
        if layout_data is None:
            self._layout_json = None
            return

        self._layout_json = layout_data
        omni.log.info("JSON file loaded successfully")
        omni.log.info(f"Layout data: {self._layout_json}")

        # Load only; placement uses separate button.

    def _on_generate_from_json_click(self):
        if self._search_task and not self._search_task.done():
            self._search_task.cancel()

        layout_data = None

        if getattr(self, "_loaded_json_path", "") and os.path.exists(self._loaded_json_path):
            layout_data = self._load_json_with_fallback(self._loaded_json_path)
            if layout_data is None:
                omni.log.warn("Failed to load JSON for placement.")
                return
            self._layout_json = layout_data
        elif getattr(self, "_layout_json", None):
            layout_data = self._layout_json

        if layout_data is None:
            omni.log.warn("No JSON selected. Use 'Select JSON File...' first.")
            return

        omni.log.info("Starting placement from selected JSON...")
        self._start_asset_search(layout_data)

    def _on_place_bbox_from_json_click(self):
        if self._search_task and not self._search_task.done():
            self._search_task.cancel()

        layout_data = None

        if getattr(self, "_loaded_json_path", "") and os.path.exists(self._loaded_json_path):
            layout_data = self._load_json_with_fallback(self._loaded_json_path)
            if layout_data is None:
                omni.log.warn("Failed to load JSON for bbox placement.")
                return
            self._layout_json = layout_data
        elif getattr(self, "_layout_json", None):
            layout_data = self._layout_json

        if layout_data is None:
            omni.log.warn("No JSON selected. Use 'Select JSON File...' first.")
            return

        omni.log.info("Placing debug bounding boxes from selected JSON...")
        self._search_task = asyncio.ensure_future(self._place_debug_bboxes(layout_data))

    def _on_select_image_click(self):
        """画像ファイル選択ボタンのコールバック"""
        omni.log.info("Image file picker opened")
        if self._file_picker:
            self._deferred_destroy_picker()

        def image_handler(*args, **kwargs):
            """画像ファイルピッカーのハンドラー"""
            omni.log.info(f"Image handler called with args={args}, kwargs={kwargs}")
            # FilePickerDialog は (filename, dirname) の2つの引数を渡す
            if len(args) == 2:
                filename, dirname = args[0], args[1]
                # selections を構築
                import os
                full_path = os.path.join(dirname, filename)
                selections = [full_path]
            else:
                omni.log.error(f"Unexpected arguments: args={args}")
                return

            self._on_image_selected(selections)

        self._file_picker = fp.FilePickerDialog(
            "Select Image File",
            allow_multi_selection=False,
            apply_button_label="Open",
            file_extension_options=[
                ("*.jpg", "JPEG Image"),
                ("*.jpeg", "JPEG Image"),
                ("*.png", "PNG Image"),
                ("*.bmp", "BMP Image"),
            ],
            click_apply_handler=image_handler,
            click_cancel_handler=lambda *args: self._deferred_destroy_picker(),
        )
        self._file_picker.show()

    def _on_image_selected(self, selections: list):
        """画像ファイルが選択された"""
        if self._file_picker:
            self._deferred_destroy_picker()

        if selections and len(selections) > 0:
            self._image_path = selections[0]
            file_basename = os.path.basename(self._image_path)
            self._image_label.text = file_basename
            # JSONファイルに保存
            self._save_settings_to_json()
            self._update_image_preview()
            omni.log.info(f"Image file selected: {self._image_path}")

    def _on_select_dims_click(self):
        """寸法ファイル選択ボタンのコールバック"""
        omni.log.info("Dimensions file picker opened")
        if self._file_picker:
            self._deferred_destroy_picker()

        def dims_handler(*args, **kwargs):
            omni.log.info(f"Dims handler called with args={args}, kwargs={kwargs}")
            if len(args) == 2:
                filename, dirname = args[0], args[1]
                import os
                full_path = os.path.join(dirname, filename)
                selections = [full_path]
            else:
                omni.log.error(f"Unexpected arguments: args={args}")
                return
            self._on_dims_selected(selections)

        self._file_picker = fp.FilePickerDialog(
            "Select Dimensions File",
            allow_multi_selection=False,
            apply_button_label="Open",
            file_extension_options=[("*.txt", "Text File")],
            click_apply_handler=dims_handler,
            click_cancel_handler=lambda *args: self._deferred_destroy_picker(),
        )
        self._file_picker.show()

    def _on_dims_selected(self, selections: list):
        """寸法ファイルが選択された"""
        if self._file_picker:
            self._deferred_destroy_picker()

        if selections and len(selections) > 0:
            self._dimensions_path = selections[0]
            file_basename = os.path.basename(self._dimensions_path)
            self._dims_label.text = file_basename
            # JSONファイルに保存
            self._save_settings_to_json()
            self._refresh_preview_models()
            omni.log.info(f"Dimensions file selected: {self._dimensions_path}")

    def _on_select_prompt1_click(self):
        """プロンプト1ファイル選択ボタンのコールバック"""
        omni.log.info("Prompt 1 file picker opened")
        if self._file_picker:
            self._deferred_destroy_picker()

        def prompt1_handler(*args, **kwargs):
            omni.log.info(f"Prompt1 handler called with args={args}, kwargs={kwargs}")
            if len(args) == 2:
                filename, dirname = args[0], args[1]
                import os
                full_path = os.path.join(dirname, filename)
                selections = [full_path]
            else:
                omni.log.error(f"Unexpected arguments: args={args}")
                return
            self._on_prompt1_selected(selections)

        self._file_picker = fp.FilePickerDialog(
            "Select Prompt 1 File",
            allow_multi_selection=False,
            apply_button_label="Open",
            file_extension_options=[("*.txt", "Text File")],
            click_apply_handler=prompt1_handler,
            click_cancel_handler=lambda *args: self._deferred_destroy_picker(),
        )
        self._file_picker.show()

    def _on_prompt1_selected(self, selections: list):
        """プロンプト1ファイルが選択された"""
        if self._file_picker:
            self._deferred_destroy_picker()

        if selections and len(selections) > 0:
            self._prompt1_path = selections[0]
            file_basename = os.path.basename(self._prompt1_path)
            self._prompt1_label.text = file_basename
            # JSONファイルに保存
            self._save_settings_to_json()
            self._refresh_preview_models()
            omni.log.info(f"Prompt 1 file selected: {self._prompt1_path}")

    def _on_select_prompt2_click(self):
        """プロンプト2ファイル選択ボタンのコールバック"""
        omni.log.info("Prompt 2 file picker opened")
        if self._file_picker:
            self._deferred_destroy_picker()

        def prompt2_handler(*args, **kwargs):
            omni.log.info(f"Prompt2 handler called with args={args}, kwargs={kwargs}")
            if len(args) == 2:
                filename, dirname = args[0], args[1]
                import os
                full_path = os.path.join(dirname, filename)
                selections = [full_path]
            else:
                omni.log.error(f"Unexpected arguments: args={args}")
                return
            self._on_prompt2_selected(selections)

        self._file_picker = fp.FilePickerDialog(
            "Select Prompt 2 File",
            allow_multi_selection=False,
            apply_button_label="Open",
            file_extension_options=[("*.txt", "Text File")],
            click_apply_handler=prompt2_handler,
            click_cancel_handler=lambda *args: self._deferred_destroy_picker(),
        )
        self._file_picker.show()

    def _on_prompt2_selected(self, selections: list):
        """プロンプト2ファイルが選択された"""
        if self._file_picker:
            self._deferred_destroy_picker()

        if selections and len(selections) > 0:
            self._prompt2_path = selections[0]
            file_basename = os.path.basename(self._prompt2_path)
            self._prompt2_label.text = file_basename
            # JSONファイルに保存
            self._save_settings_to_json()
            self._refresh_preview_models()
            omni.log.info(f"Prompt 2 file selected: {self._prompt2_path}")

    def _on_generate_json_click(self):
        """Generate JSON (AI) ボタンのコールバック（非同期タスクを起動）"""
        omni.log.info("Starting AI JSON generation (async)...")

        # 既にAI処理中なら何もしない
        if self._ai_task and not self._ai_task.done():
            omni.log.warn("AI generation is already running.")
            return

        # 既存の配置タスクがあればキャンセル
        if self._search_task and not self._search_task.done():
            self._search_task.cancel()

        # UIを "生成中..." に更新
        self._analysis_text_model.as_string = "Generating... (Step 1: Analyzing image...)"
        self._set_ai_status("AI Status: Step 1 (Analyzing image)")
        self._set_ai_tokens(None, None)
        self._set_ai_busy(True)

        # 非同期でAI生成タスクを実行
        self._ai_task = asyncio.ensure_future(self._do_ai_generation())

    def _on_approve_click(self):
        """承認ボタンのコールバック"""
        if not self._approval_pending or not self._analysis_result:
            omni.log.warn("No pending approval or analysis result")
            return

        omni.log.info("User approved the analysis. Continuing to Step 2...")

        # 承認ボタンを非表示
        self._approval_buttons_container.visible = False
        self._approval_pending = False

        # 追加コンテキストをクリア
        self._additional_context = ""

        # Step 2以降を実行
        self._set_ai_status("AI Status: Step 2 (Generating JSON)")
        self._set_ai_busy(True)
        self._ai_task = asyncio.ensure_future(self._do_step2_and_placement())

    def _on_reject_click(self):
        """拒否ボタンのコールバック - 修正指示ポップアップを表示"""
        omni.log.info("User rejected the analysis. Opening context popup...")

        # 承認ボタンを非表示
        self._approval_buttons_container.visible = False

        # 次のフレームでポップアップを表示（イベント中の UI 操作を避けるため）
        async def _delayed_show_popup():
            await omni.kit.app.get_app().next_update_async()
            self._show_context_popup()

        asyncio.ensure_future(_delayed_show_popup())

    def _show_context_popup(self):
        """修正指示入力ポップアップを表示"""
        omni.log.info("Creating context input popup window...")

        # 既存のポップアップがある場合は閉じる（この関数は次のフレームで呼ばれるので安全）
        if self._context_popup:
            try:
                omni.log.info("Destroying existing popup before creating new one...")
                self._context_popup.destroy()
                omni.log.info("Existing popup destroyed successfully")
            except Exception as e:
                omni.log.warn(f"Failed to destroy existing popup: {e}")
            self._context_popup = None

        # ポップアップウィンドウを作成
        self._context_popup = ui.Window("Add Context for Re-analysis", width=500, height=300)

        with self._context_popup.frame:
            with ui.VStack(spacing=10, style={"margin": 10}):
                ui.Label("Please provide additional context or corrections:")
                ui.Spacer(height=5)

                # 修正指示入力フィールド
                self._context_input_model = ui.SimpleStringModel(self._additional_context)
                ui.StringField(
                    model=self._context_input_model,
                    multiline=True,
                    height=150,
                    style=self._japanese_font_style
                )

                ui.Spacer(height=10)

                # ボタン
                with ui.HStack(height=30, spacing=10):
                    ui.Button("✓ Resubmit", clicked_fn=self._on_resubmit_click, width=0, height=30)
                    ui.Button("✗ Cancel", clicked_fn=self._on_cancel_context_click, width=0, height=30)

        omni.log.info("Context popup window created successfully")

    def _on_resubmit_click(self):
        """
        再送信ボタンのコールバック - 修正指示を追加して再実行

        このメソッドは同期的に呼ばれるので、非同期タスクを
        asyncio.ensure_future() でスケジュールする必要がある
        """
        try:
            omni.log.info("="*80)
            omni.log.info("=== RESUBMIT BUTTON CLICKED ===")
            omni.log.info("="*80)

            # 1. 入力テキストを取得
            if not hasattr(self, "_context_input_model") or not self._context_input_model:
                omni.log.error("✗ Context input model not found!")
                return

            self._additional_context = self._context_input_model.as_string.strip()

            if not self._additional_context:
                omni.log.warn("⚠ User provided empty context")
            else:
                omni.log.info("✓ User provided context:")
                omni.log.info(f"  Length: {len(self._additional_context)} characters")
                omni.log.info(f"  Preview: {self._additional_context[:200]}...")

            # 2. ポップアップを次のフレームで破棄（イベント中の destroy は禁止されているため）
            if self._context_popup:
                async def _delayed_destroy_popup():
                    """次のフレームでポップアップを破棄する"""
                    await omni.kit.app.get_app().next_update_async()
                    if self._context_popup:
                        try:
                            self._context_popup.destroy()
                            self._context_popup = None
                            omni.log.info("✓ Popup window destroyed (delayed)")
                        except Exception as e:
                            omni.log.error(f"✗ Failed to destroy popup: {e}")

                omni.log.info("Scheduling popup window destruction for next frame...")
                asyncio.ensure_future(_delayed_destroy_popup())

            # 3. UIの状態をリセット
            omni.log.info("Resetting UI state...")
            self._approval_pending = False
            self._approval_buttons_container.visible = False  # 承認ボタンを非表示
            self._analysis_text_model.as_string = (
                "Resubmitting with additional context...\n\n"
                "(Step 1: Analyzing image with new instructions...)"
            )
            omni.log.info("✓ UI state reset")

            # 4. 既存のタスクをキャンセル
            if self._ai_task and not self._ai_task.done():
                omni.log.info("Cancelling existing task...")
                try:
                    self._ai_task.cancel()
                    self._ai_task = None
                    omni.log.info("Existing task cancelled")
                except Exception as e:
                    omni.log.error(f"Failed to cancel task: {e}")

            # 5. 新しい非同期タスクを開始
            omni.log.info("Starting new AI generation task...")
            omni.log.info("  Additional context will be appended to prompt")
            omni.log.info(f"  Context: '{self._additional_context[:100]}...'")

            # asyncio.ensure_future でタスクをスケジュール
            self._set_ai_status("AI Status: Step 1 (Analyzing image)")
            self._set_ai_tokens(None, None)
            self._set_ai_busy(True)
            self._ai_task = asyncio.ensure_future(self._do_ai_generation())

            omni.log.info(f"✓ Task scheduled: {self._ai_task}")
            omni.log.info("="*80)

        except Exception as e:
            omni.log.error(f"✗ CRITICAL ERROR in _on_resubmit_click: {e}")
            import traceback
            omni.log.error(f"Stack trace:\n{traceback.format_exc()}")
            # エラーをUIに表示
            if hasattr(self, "_analysis_text_model"):
                self._analysis_text_model.as_string = f"Error in resubmit: {e}"

    def _on_cancel_context_click(self):
        """キャンセルボタンのコールバック"""
        omni.log.info("Cancel button clicked in context popup")

        # ポップアップを次のフレームで破棄（イベント中の destroy は禁止されているため）
        if self._context_popup:
            async def _delayed_destroy_popup():
                """次のフレームでポップアップを破棄する"""
                await omni.kit.app.get_app().next_update_async()
                if self._context_popup:
                    try:
                        self._context_popup.destroy()
                        self._context_popup = None
                        omni.log.info("Context popup cancelled and destroyed (delayed)")
                    except Exception as e:
                        omni.log.error(f"Failed to destroy popup on cancel: {e}")

            omni.log.info("Scheduling popup destruction for next frame...")
            asyncio.ensure_future(_delayed_destroy_popup())

        # 承認ボタンを再表示
        if hasattr(self, "_approval_buttons_container"):
            self._approval_buttons_container.visible = True
            omni.log.info("Approval buttons restored")

    def _on_cancel_ai_click(self):
        """AI処理のキャンセル"""
        if self._ai_task and not self._ai_task.done():
            omni.log.info("Cancelling AI task...")
            self._ai_task.cancel()
            self._set_ai_status("AI Status: Cancelled")
            self._set_ai_busy(False)
            return

        if self._approval_pending:
            omni.log.info("Clearing pending approval state.")
            self._approval_pending = False
            if hasattr(self, "_approval_buttons_container"):
                self._approval_buttons_container.visible = False
            self._analysis_result = None
            self._set_ai_status("AI Status: Cancelled")
            self._set_ai_busy(False)

    def _set_ai_status(self, message: str):
        self._ai_status_text = message
        if hasattr(self, "_ai_status_label") and self._ai_status_label:
            self._ai_status_label.text = message

    def _set_ai_tokens(self, step1_stats: Optional[dict], step2_stats: Optional[dict]):
        def _format_stats(label: str, stats: Optional[dict]) -> str:
            if not stats:
                return f"{label}: -"
            prompt = stats.get("prompt_tokens", 0)
            completion = stats.get("completion_tokens", 0)
            total = stats.get("total_tokens", 0)
            return f"{label} in={prompt} out={completion} total={total}"

        parts = [_format_stats("Step1", step1_stats), _format_stats("Step2", step2_stats)]
        tokens_text = "Tokens: " + " | ".join(parts)
        self._ai_tokens_text = tokens_text
        if hasattr(self, "_ai_tokens_label") and self._ai_tokens_label:
            self._ai_tokens_label.text = tokens_text

    def _set_ai_busy(self, busy: bool):
        if hasattr(self, "_generate_button") and self._generate_button:
            self._generate_button.enabled = not busy
        if hasattr(self, "_cancel_ai_button") and self._cancel_ai_button:
            self._cancel_ai_button.visible = busy
