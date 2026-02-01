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
import datetime
import json
import math
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import omni.kit.app
import omni.log
import omni.usd
from pxr import Gf, Sdf, Tf, Usd, UsdGeom

from .. import backend as be
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


class CommandsMixin:
    def _get_ai_runtime_options(self, base_model_name: str) -> Tuple[str, str, Dict[str, object]]:
        step1_combo = getattr(self, "_ai_step1_model_combo", None)
        step2_combo = getattr(self, "_ai_step2_model_combo", None)
        effort_combo = getattr(self, "_ai_reasoning_effort_combo", None)
        verbosity_combo = getattr(self, "_ai_text_verbosity_combo", None)
        detail_combo = getattr(self, "_ai_image_detail_combo", None)

        step1_index = step1_combo.model.get_item_value_model().as_int if step1_combo else self._ai_step1_model_index
        step2_index = step2_combo.model.get_item_value_model().as_int if step2_combo else self._ai_step2_model_index
        effort_index = effort_combo.model.get_item_value_model().as_int if effort_combo else self._ai_reasoning_effort_index
        verbosity_index = (
            verbosity_combo.model.get_item_value_model().as_int if verbosity_combo else self._ai_text_verbosity_index
        )
        detail_index = detail_combo.model.get_item_value_model().as_int if detail_combo else self._ai_image_detail_index

        step1_model_name = base_model_name
        step2_model_name = base_model_name
        if 0 < step1_index < len(ADV_MODEL_CHOICES):
            step1_model_name = ADV_MODEL_CHOICES[step1_index]
        if 0 < step2_index < len(ADV_MODEL_CHOICES):
            step2_model_name = ADV_MODEL_CHOICES[step2_index]

        reasoning_effort = None
        if 0 < effort_index < len(REASONING_EFFORT_CHOICES):
            reasoning_effort = REASONING_EFFORT_CHOICES[effort_index]

        text_verbosity = None
        if 0 < verbosity_index < len(TEXT_VERBOSITY_CHOICES):
            text_verbosity = TEXT_VERBOSITY_CHOICES[verbosity_index]

        image_detail = None
        if 0 < detail_index < len(IMAGE_DETAIL_CHOICES):
            image_detail = IMAGE_DETAIL_CHOICES[detail_index]

        max_output_tokens = self._ai_max_output_tokens
        if hasattr(self, "_ai_max_output_tokens_model"):
            raw_tokens = self._ai_max_output_tokens_model.as_string.strip()
            if raw_tokens:
                try:
                    max_output_tokens = int(float(raw_tokens))
                except (ValueError, TypeError):
                    max_output_tokens = self._ai_max_output_tokens
        if max_output_tokens <= 0:
            max_output_tokens = None

        return step1_model_name, step2_model_name, {
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens,
            "text_verbosity": text_verbosity,
            "image_detail": image_detail,
        }

    def _get_replacement_history(self, prim_path: str) -> Dict[str, List[str]]:
        if not hasattr(self, "_replacement_history_by_prim") or self._replacement_history_by_prim is None:
            self._replacement_history_by_prim = {}
        history = self._replacement_history_by_prim.get(prim_path)
        if not history:
            history = {"urls": [], "keys": []}
            self._replacement_history_by_prim[prim_path] = history
        return history


    def _append_replacement_history(
        self, prim_path: str, asset_url: Optional[str], identity_key: Optional[str]
    ) -> None:
        limit = int(getattr(self, "_replacement_history_limit", 20))
        if limit <= 0:
            return
        history = self._get_replacement_history(prim_path)
        if asset_url:
            normalized = self._normalize_asset_url(asset_url)
            if normalized and normalized not in history["urls"]:
                history["urls"].append(normalized)
                if len(history["urls"]) > limit:
                    history["urls"] = history["urls"][-limit:]
        if identity_key and identity_key not in history["keys"]:
            history["keys"].append(identity_key)
            if len(history["keys"]) > limit:
                history["keys"] = history["keys"][-limit:]


    async def _select_next_history_url(
        self, history_urls: List[str], current_url: str
    ) -> Optional[str]:
        if not history_urls or len(history_urls) < 2:
            return None
        try:
            start_idx = history_urls.index(current_url)
        except ValueError:
            start_idx = -1

        total = len(history_urls)
        for offset in range(1, total + 1):
            candidate = history_urls[(start_idx + offset) % total]
            if candidate == current_url:
                continue
            if self._is_asset_blacklisted(candidate):
                continue
            candidate_key = await self._get_asset_identity_key(candidate)
            if candidate_key and candidate_key in getattr(self, "_asset_blacklist_keys", set()):
                continue
            return candidate
        return None


    async def _do_ai_generation(self):
        """
        （非同期）AI生成のメインロジック

        このメソッドは asyncio.ensure_future() によって非同期タスクとして実行される。
        Resubmit時には、self._additional_context に追加コンテキストが設定されている。
        """
        try:
            omni.log.info("+"*80)
            omni.log.info("*** _do_ai_generation() STARTED ***")
            omni.log.info("+"*80)
            omni.log.info(f"  Image path: {self._image_path}")
            omni.log.info(f"  Dimensions path: {self._dimensions_path}")
            omni.log.info(f"  Additional context: {len(self._additional_context) if self._additional_context else 0} chars")

            # --- 1. 入力チェック ---
            if not self._image_path:
                omni.log.error("✗ Image file not selected")
                self._analysis_text_model.as_string = "Error: Please select an image file first."
                return

            if not self._dimensions_path:
                omni.log.error("Please select a dimensions file first")
                self._analysis_text_model.as_string = "Error: Please select a dimensions file first."
                return

            # APIキーを取得（起動時に環境変数から読み込み済み）
            api_key = self._api_key_model.as_string.strip()

            if not api_key:
                omni.log.error(
                    "OpenAI API Key is not set. Set the OPENAI_API_KEY environment variable or enter it in the UI."
                )
                self._analysis_text_model.as_string = (
                    "Error: OpenAI API Key is not set. Set the OPENAI_API_KEY environment variable or enter it in the UI."
                )
                return

            # プロンプトとディメンションの読み込み
            prompt1_text, prompt1_source = self._resolve_prompt_text(
                self._prompt1_path, "prompt_1.txt", DEFAULT_PROMPT1_TEXT, "Prompt 1"
            )
            prompt2_text, prompt2_source = self._resolve_prompt_text(
                self._prompt2_path, "prompt_2.txt", DEFAULT_PROMPT2_TEXT, "Prompt 2"
            )
            dimensions_text = self._read_text_with_fallback(self._dimensions_path, "Dimensions", required=True)

            if dimensions_text is None:
                self._analysis_text_model.as_string = "Error: Failed to load dimensions file."
                return

            # モデル名の取得
            model_index = self._model_combo.model.get_item_value_model().as_int
            if model_index < 0 or model_index >= len(MODEL_CHOICES):
                model_index = 0
            model_name = MODEL_CHOICES[model_index]
            step1_model_name, step2_model_name, ai_overrides = self._get_ai_runtime_options(model_name)

            omni.log.info(f"Using Step 1 model: {step1_model_name}")
            omni.log.info(f"Using Step 2 model: {step2_model_name}")
            omni.log.info(f"Image: {self._image_path}")
            omni.log.info(f"Dimensions: {self._dimensions_path}")
            omni.log.info(f"Prompt 1 source: {prompt1_source}")
            omni.log.info(f"Prompt 2 source: {prompt2_source}")

            # 画像のBase64エンコード
            image_base64 = be.encode_image_to_base64(self._image_path)
            if not image_base64:
                omni.log.error("Failed to encode image to base64")
                self._analysis_text_model.as_string = "Error: Failed to encode image to base64."
                return

            if not prompt1_text or not prompt2_text:
                omni.log.error("Failed to load prompt text. Aborting generation.")
                self._analysis_text_model.as_string = "Error: Failed to load prompt text."
                return

            # 修正指示がある場合はprompt1に追加
            if self._additional_context:
                omni.log.info("="*60)
                omni.log.info("【RESUBMISSION WITH ADDITIONAL CONTEXT】")
                omni.log.info(f"Context length: {len(self._additional_context)} characters")
                omni.log.info(f"Context preview: {self._additional_context[:200]}...")
                omni.log.info("="*60)
                prompt1_text += f"\n\n【ユーザーからの追加指示・修正】\n{self._additional_context}"
                # コンテキスト適用後にクリア（次回の実行で混ざらないように）
                # self._additional_context = ""  # これはやらない：承認ステップで使うかもしれない

            # --- 2. ステップ1: 画像分析 (非同期呼び出し) ---
            omni.log.info("=== Step 1: Analyzing image ===")
            analysis_text, step1_stats = await be.step1_analyze_image(
                image_base64,
                prompt1_text,
                dimensions_text,
                step1_model_name,
                api_key,
                **ai_overrides,
            )

            if not analysis_text:
                omni.log.error("Failed to analyze image")
                self._analysis_text_model.as_string = "Error: Failed to analyze image."
                return

            # --- ▼ UIに結果を反映 ▼ ---
            step1_stats_text = f"\\n\\n━━━ Step 1 Stats ━━━\\n⏱ Time: {step1_stats['time']:.2f}s | 📥 Prompt: {step1_stats['prompt_tokens']:,} | 📤 Completion: {step1_stats['completion_tokens']:,} | 📊 Total: {step1_stats['total_tokens']:,}"
            self._analysis_text_model.as_string = f"Step 1 (Analysis) Complete:{step1_stats_text}\\n\\n{analysis_text}"
            self._set_ai_tokens(step1_stats, None)
            self._set_ai_status("AI Status: Step 1 (Analysis) Complete")
            omni.log.info(f"Analysis completed in {step1_stats['time']:.2f}s")
            omni.log.info(f"Analysis result:\n{analysis_text}")

            # 承認モードの場合はここで停止
            if self._require_approval.as_bool:
                omni.log.info("Approval mode enabled. Waiting for user approval...")
                self._analysis_text_model.as_string += (
                    "\n\n⏸️ Waiting for approval. Click 'Approve' to continue or 'Reject' to add context."
                )

                # 分析結果を保存（承認後に使用）
                self._analysis_result = {
                    "analysis_text": analysis_text,
                    "step1_stats": step1_stats,
                    "image_base64": image_base64,
                    "prompt2_text": prompt2_text,
                    "dimensions_text": dimensions_text,
                    "model_name": model_name,
                    "step2_model_name": step2_model_name,
                    "ai_overrides": ai_overrides,
                    "api_key": api_key
                }

                # 承認ボタンを表示
                self._approval_buttons_container.visible = True
                self._approval_pending = True
                return  # ここで一旦停止

            # 承認モードがオフの場合は続行
            omni.log.info("Approval mode disabled. Continuing to Step 2...")

            # --- 3. ステップ2: JSON生成 (非同期呼び出し) ---
            omni.log.info("=== Step 2: Generating JSON ===")
            self._analysis_text_model.as_string += "\n\nGenerating... (Step 2: Creating JSON...)"

            layout_json, step2_stats = await be.step2_generate_json(
                analysis_text,
                dimensions_text,
                image_base64,
                prompt2_text,
                step2_model_name,
                api_key,
                **ai_overrides,
            )

            if not layout_json:
                omni.log.error("Failed to generate JSON")
                self._analysis_text_model.as_string += "\nError: Failed to generate JSON."
                return

            self._analysis_text_model.as_string += "\nJSON Generation Complete."
            self._set_ai_tokens(step1_stats, step2_stats)
            self._set_ai_status("AI Status: Step 2 (JSON Generation) Complete")
            omni.log.info(f"JSON generation completed in {step2_stats['time']:.2f}s")

            # JSONをファイルに保存
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
            base_name = os.path.splitext(os.path.basename(self._image_path))[0]
            model_name_safe = step2_model_name.replace("/", "-").replace(":", "-")
            json_filename = f"{base_name}_layout_{model_name_safe}_{timestamp}.json"

            out_dir = self._get_json_output_dir()
            os.makedirs(out_dir, exist_ok=True)
            json_path = os.path.join(out_dir, json_filename)

            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(layout_json, f, indent=4, ensure_ascii=False)
                omni.log.info(f"JSON saved to: {json_path}")
                self._analysis_text_model.as_string += f"\nJSON saved to: {json_path}"
                self._set_generated_json_preview(json_path, layout_json)
            except Exception as e:
                omni.log.error(f"Failed to save JSON: {e}")
                self._analysis_text_model.as_string += f"\nError: Failed to save JSON: {e}"
                return

            # 統計情報の出力
            total_time = step1_stats["time"] + step2_stats["time"]
            total_prompt_tokens = step1_stats["prompt_tokens"] + step2_stats["prompt_tokens"]
            total_completion_tokens = step1_stats["completion_tokens"] + step2_stats["completion_tokens"]
            total_tokens = step1_stats["total_tokens"] + step2_stats["total_tokens"]

            omni.log.info("="*60)
            omni.log.info("【Total Statistics (Step 1 + Step 2)】")
            omni.log.info("="*60)
            omni.log.info(f"Total processing time: {total_time:.2f}s")
            omni.log.info(f"Total prompt tokens: {total_prompt_tokens:,}")
            omni.log.info(f"Total completion tokens: {total_completion_tokens:,}")
            omni.log.info(f"Total tokens: {total_tokens:,}")
            omni.log.info("="*60)

            # --- 4. ステップ3: 衝突検出 ---
            omni.log.info("=== Step 3: Checking collisions ===")
            total_stats_text = f"\\n\\n━━━ Total Stats (Step 1 + Step 2) ━━━\\n⏱ Time: {total_time:.2f}s | 📥 Prompt: {total_prompt_tokens:,} | 📤 Completion: {total_completion_tokens:,} | 📊 Total: {total_tokens:,}"
            self._analysis_text_model.as_string += total_stats_text
            self._analysis_text_model.as_string += "\n\nChecking collisions..."
            collision_results = be.step3_check_collisions(layout_json)

            # ログを保存
            log_path = self._save_generation_log(
                step1_stats=step1_stats,
                step2_stats=step2_stats,
                analysis_text=analysis_text,
                layout_json=layout_json,
                collision_results=collision_results,
                model_name=model_name,
                step2_model_name=step2_model_name,
                ai_overrides=ai_overrides,
                image_path=self._image_path,
                prompt1_text=prompt1_text,
                prompt2_text=prompt2_text,
                dimensions_text=dimensions_text
            )
            if log_path:
                self._analysis_text_model.as_string += f"\nDetailed log saved to: {log_path}"

            # --- 5. ステップ4/5: アセット配置 ---
            self._layout_json = layout_json
            self._analysis_text_model.as_string += "\nStarting asset placement..."

            # _start_asset_searchはすでに非同期タスクとして動作
            self._start_asset_search(self._layout_json)

            self._analysis_text_model.as_string += "\n\nProcess finished."
            omni.log.info("AI JSON generation completed successfully!")

        except asyncio.CancelledError:
            omni.log.warn("AI Generation task was cancelled.")
            self._analysis_text_model.as_string = "Task Cancelled."
            raise
        except Exception as e:
            omni.log.error(f"AI Generation failed: {e}")
            self._analysis_text_model.as_string = f"Error: {e}"
        finally:
            self._set_ai_busy(False)
            self._ai_task = None

    async def _do_step2_and_placement(self):
        """Step 2（JSON生成）以降を実行"""
        try:
            if not self._analysis_result:
                omni.log.error("No analysis result available")
                return

            # 保存した分析結果を取得
            analysis_text = self._analysis_result["analysis_text"]
            step1_stats = self._analysis_result["step1_stats"]
            image_base64 = self._analysis_result["image_base64"]
            prompt2_text = self._analysis_result["prompt2_text"]
            dimensions_text = self._analysis_result["dimensions_text"]
            model_name = self._analysis_result["model_name"]
            step2_model_name = self._analysis_result.get("step2_model_name", model_name)
            ai_overrides = self._analysis_result.get("ai_overrides", {})
            api_key = self._analysis_result["api_key"]

            # --- ステップ2: JSON生成 (非同期呼び出し) ---
            omni.log.info("=== Step 2: Generating JSON ===")
            self._analysis_text_model.as_string += "\n\nGenerating... (Step 2: Creating JSON...)"

            layout_json, step2_stats = await be.step2_generate_json(
                analysis_text,
                dimensions_text,
                image_base64,
                prompt2_text,
                step2_model_name,
                api_key,
                **ai_overrides,
            )

            if not layout_json:
                omni.log.error("Failed to generate JSON")
                self._analysis_text_model.as_string += "\nError: Failed to generate JSON."
                return

            self._analysis_text_model.as_string += "\nJSON Generation Complete."
            self._set_ai_tokens(self._analysis_result["step1_stats"], step2_stats)
            self._set_ai_status("AI Status: Step 2 (JSON Generation) Complete")
            omni.log.info(f"JSON generation completed in {step2_stats['time']:.2f}s")

            # JSONをファイルに保存
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
            base_name = os.path.splitext(os.path.basename(self._image_path))[0]
            model_name_safe = step2_model_name.replace("/", "-").replace(":", "-")
            json_filename = f"{base_name}_layout_{model_name_safe}_{timestamp}.json"

            out_dir = self._get_json_output_dir()
            os.makedirs(out_dir, exist_ok=True)
            json_path = os.path.join(out_dir, json_filename)

            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(layout_json, f, indent=4, ensure_ascii=False)
                omni.log.info(f"JSON saved to: {json_path}")
                self._analysis_text_model.as_string += f"\nJSON saved to: {json_path}"
                self._set_generated_json_preview(json_path, layout_json)
            except Exception as e:
                omni.log.error(f"Failed to save JSON: {e}")
                self._analysis_text_model.as_string += f"\nError: Failed to save JSON: {e}"
                return

            # 統計情報の出力
            total_time = step1_stats["time"] + step2_stats["time"]
            total_prompt_tokens = step1_stats["prompt_tokens"] + step2_stats["prompt_tokens"]
            total_completion_tokens = step1_stats["completion_tokens"] + step2_stats["completion_tokens"]
            total_tokens = step1_stats["total_tokens"] + step2_stats["total_tokens"]

            omni.log.info("="*60)
            omni.log.info("【Total Statistics (Step 1 + Step 2)】")
            omni.log.info("="*60)
            omni.log.info(f"Total processing time: {total_time:.2f}s")
            omni.log.info(f"Total prompt tokens: {total_prompt_tokens:,}")
            omni.log.info(f"Total completion tokens: {total_completion_tokens:,}")
            omni.log.info(f"Total tokens: {total_tokens:,}")
            omni.log.info("="*60)

            # --- ステップ3: 衝突検出 ---
            omni.log.info("=== Step 3: Checking collisions ===")
            total_stats_text = f"\\n\\n━━━ Total Stats (Step 1 + Step 2) ━━━\\n⏱ Time: {total_time:.2f}s | 📥 Prompt: {total_prompt_tokens:,} | 📤 Completion: {total_completion_tokens:,} | 📊 Total: {total_tokens:,}"
            self._analysis_text_model.as_string += total_stats_text
            # --- ステップ3: 衝突検出 ---
            omni.log.info("=== Step 3: Checking collisions ===")
            collision_results = be.step3_check_collisions(layout_json)

            # ログを保存
            log_path = self._save_generation_log(
                step1_stats=step1_stats,
                step2_stats=step2_stats,
                analysis_text=analysis_text,
                layout_json=layout_json,
                collision_results=collision_results,
                model_name=model_name,
                step2_model_name=step2_model_name,
                ai_overrides=ai_overrides,
                image_path=self._image_path,
                prompt1_text=prompt2_text, # ※このケースではStep2から開始
                prompt2_text=prompt2_text,
                dimensions_text=dimensions_text,
                tag="step2_only"
            )
            if log_path:
                self._analysis_text_model.as_string += f"\nDetailed log saved to: {log_path}"

            # --- ステップ4以降: アセット検索と配置 ---
            self._layout_json = layout_json
            self._analysis_text_model.as_string += "\n\nStarting asset placement..."
            self._start_asset_search(self._layout_json)
            self._start_asset_search(self._layout_json)

            self._analysis_text_model.as_string += "\n\nProcess finished."
            omni.log.info("Step 2-5 completed successfully!")

        except asyncio.CancelledError:
            omni.log.warn("Step 2-5 task was cancelled.")
            self._analysis_text_model.as_string += "\nTask Cancelled."
            raise
        except Exception as e:
            omni.log.error(f"Step 2-5 failed: {e}")
            self._analysis_text_model.as_string += f"\nError: {e}"
        finally:
            self._set_ai_busy(False)
            self._ai_task = None

    # --- フェーズ4 (共通の開始点) ---
    def _start_asset_search(self, layout_data: dict):
        """JSONデータをもとにUSD Searchを開始する (フェーズ4)"""
        if layout_data is None:
            omni.log.error("No layout data to process.")
            return

        # ステージをZ-Up / メートル単位に設定
        stage = omni.usd.get_context().get_stage()
        if stage:
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            omni.log.info("Stage configured: Z-Up, 1.0 meters per unit")

        omni.log.info("="*50)
        omni.log.info("STARTING PHASE 4: USD Search")
        omni.log.info("Layout data received:")
        omni.log.info(f"{layout_data}")
        omni.log.info("="*50)

        if self._search_task and not self._search_task.done():
            self._search_task.cancel()

        # レイアウトデータを非同期タスク用にコピー
        layout_copy = json.loads(json.dumps(layout_data))
        self._search_task = asyncio.ensure_future(self._search_and_place_assets(layout_copy))

    def _extract_layout_objects(self, layout_data) -> List[Dict[str, object]]:
        """レイアウトデータから配置対象のオブジェクトリストを抽出する。"""
        if isinstance(layout_data, list):
            return [obj for obj in layout_data if isinstance(obj, dict)]

        if not isinstance(layout_data, dict):
            return []

        candidate_keys = [
            "area_objects_list",
            "objects",
            "items",
            "object_list",
            "layout_objects",
        ]
        for key in candidate_keys:
            value = layout_data.get(key)
            if isinstance(value, list):
                return [obj for obj in value if isinstance(obj, dict)]

        # 一部のJSONでは layout_data["layout"]["objects"] のようにネストされることがある
        for value in layout_data.values():
            if isinstance(value, list):
                filtered = [obj for obj in value if isinstance(obj, dict)]
                if filtered:
                    return filtered
            elif isinstance(value, dict):
                nested = self._extract_layout_objects(value)
                if nested:
                    return nested

        return []

    
    def _place_debug_bboxes(
        self,
        layout_data: Dict[str, object],
        root_name: str = "DebugBBoxes",
        clear_existing: bool = True,
    ) -> None:
        """JSON?????????????????(BBox)??????"""
        if not layout_data:
            omni.log.warn("No layout data provided for debug bboxes.")
            return

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            omni.log.error("No USD stage is available. Open or create a stage first.")
            return

        # ????????
        root_token = self._sanitize_identifier(root_name)
        root_path = f"/{root_token}"
        if clear_existing:
            existing = stage.GetPrimAtPath(root_path)
            if existing and existing.IsValid():
                stage.RemovePrim(root_path)

        UsdGeom.Xform.Define(stage, Sdf.Path(root_path))

        objects = self._extract_layout_objects(layout_data)
        if not objects:
            omni.log.warn("Layout data does not contain placeable objects for debug bboxes.")
            return

        # ?????????????
        category_colors = {
            "bed": Gf.Vec3f(0.7, 0.5, 0.9),
            "sofa": Gf.Vec3f(0.5, 0.7, 0.9),
            "table": Gf.Vec3f(0.8, 0.7, 0.4),
            "chair": Gf.Vec3f(0.6, 0.8, 0.6),
            "door": Gf.Vec3f(0.9, 0.6, 0.4),
            "window": Gf.Vec3f(0.4, 0.8, 0.9),
            "wall": Gf.Vec3f(0.7, 0.7, 0.7),
            "floor": Gf.Vec3f(0.5, 0.5, 0.5),
        }

        placed = 0
        skipped = 0
        for index, obj in enumerate(objects, start=1):
            name = str(obj.get("object_name", "") or obj.get("name", "") or f"item_{index}")
            category = str(obj.get("category", "") or "").strip().lower()

            length = self._extract_float(obj, "Length", 0.0)
            width = self._extract_float(obj, "Width", 0.0)
            height = self._extract_float(obj, "Height", 0.0)

            if length <= 0.0 or width <= 0.0 or height <= 0.0:
                omni.log.warn(
                    f"[DebugBBox] Skipping '{name}': invalid size (L={length}, W={width}, H={height})"
                )
                skipped += 1
                continue

            x = self._extract_float(obj, "X", 0.0)
            y = self._extract_float(obj, "Y", 0.0)
            z = self._extract_float(obj, "Z", 0.0)

            rotation = self._extract_optional_float_by_keys(
                obj,
                [
                    "rotationZ",
                    "RotationZ",
                    "rotation_z",
                    "Rotation_Z",
                    "rotationY",
                    "RotationY",
                    "rotation_y",
                    "Rotation_Y",
                    "rotation",
                    "Rotation",
                ],
            )
            rotation = rotation if rotation is not None else 0.0

            token = self._sanitize_identifier(name)
            prim_path = self._get_unique_child_path(stage, root_path, token)
            cube = UsdGeom.Cube.Define(stage, prim_path)
            cube.CreateSizeAttr(1.0)

            # ???
            color = category_colors.get(category, Gf.Vec3f(0.8, 0.8, 0.8))
            gprim = UsdGeom.Gprim(cube.GetPrim())
            gprim.CreateDisplayColorAttr([color])

            # ????????
            xformable = UsdGeom.Xformable(cube.GetPrim())
            translate_op = xformable.AddTranslateOp()
            rotate_op = xformable.AddRotateXYZOp()
            scale_op = xformable.AddScaleOp()

            translate_op.Set(Gf.Vec3f(x, y, z + height * 0.5))
            rotate_op.Set(Gf.Vec3f(0.0, 0.0, rotation))
            scale_op.Set(Gf.Vec3f(length, width, height))

            placed += 1

        omni.log.info(
            f"[DebugBBox] Placed {placed} boxes under '{root_path}'. Skipped={skipped}."
        )

    async def _search_and_place_assets(self, layout_data) -> None:
            """USD Search を用いてアセット検索と配置を行う非同期タスク。"""
            placed = 0
            skipped = 0
    
            try:
                search_root = self._search_root_model.as_string if self._search_root_model else ""
                search_root = search_root.strip()
                if not search_root:
                    omni.log.error("Search root URL is empty. Set the 'Search Root URL' field before running placement.")
                    return
    
                if not search_root.startswith("omniverse://"):
                    omni.log.error(f"Search root must start with 'omniverse://'. Current value: {search_root}")
                    return
    
                normalized_root = search_root if search_root.endswith("/") else f"{search_root}/"
    
                stage = omni.usd.get_context().get_stage()
                if stage is None:
                    omni.log.error("No USD stage is available. Open or create a stage before placing assets.")
                    return
    
                area_name = layout_data.get("area_name") if isinstance(layout_data, dict) else None
                root_prim_path = self._get_or_create_root_prim(stage, area_name)
    
                objects = self._extract_layout_objects(layout_data)
                if not objects:
                    omni.log.warn(
                        "Layout data does not contain placeable objects. Expected keys such as "
                        "'area_objects_list', 'objects', or a top-level list of entries."
                    )
                    return
                omni.log.info(f"Found {len(objects)} candidate objects for placement.")
    
                # ドアのリストを収集（壁の切り抜き用）
                door_objects = []
    
                for index, obj in enumerate(objects):
                    name = str(obj.get("object_name", "") or "").strip()
                    category = str(obj.get("category", "") or "").strip()
                    category_lower = category.lower()
                    search_prompt = str(obj.get("search_prompt", "") or "").strip()
    
                    if not (name or category or search_prompt):
                        omni.log.warn(f"Skipping object #{index + 1}: missing 'object_name', 'category', and 'search_prompt'.")
                        skipped += 1
                        continue
    
                    # 床の場合は手続き的に生成
                    if name.lower() == "floor" or category_lower == "floor":
                        omni.log.info(f"[Procedural] Generating floor ({index + 1}/{len(objects)})")
                        floor_path = self._create_procedural_floor(stage, root_prim_path, obj)
                        if floor_path:
                            omni.log.info(f"Generated floor at {floor_path}")
                            placed += 1
                        else:
                            skipped += 1
                        continue
    
                    # ドアの場合はリストに追加（後で壁の切り抜きに使用）
                    if category_lower == "door" or "door" in name.lower():
                        door_objects.append(obj)
                        omni.log.info(f"[Door] Detected door '{name}' at X={obj.get('X', 0)}, Y={obj.get('Y', 0)}")
    
                    # 検索クエリを生成
                    search_query = self._build_search_query_from_object(obj)
    
                    omni.log.info(
                        f"[Search] Querying '{search_query}' "
                        f"(Original name: '{name}', category: '{category}') "
                        f"({index + 1}/{len(objects)})"
                    )
                    try:
                        # ベクトルベースのセマンティック検索APIを使用
                        # 直接USDファイルのURLが返されるため、再帰的検索は不要
                        asset_url = await self._semantic_search_asset(search_query, normalized_root)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        omni.log.error(f"Vector search failed for '{name}': {exc}")
                        skipped += 1
                        continue
    
                    if not asset_url:
                        omni.log.warn(f"No matching USD asset found for '{name}' (searched as '{search_query}').")
                        skipped += 1
                        continue
    
                    prim_path = await self._reference_asset(stage, root_prim_path, name, asset_url, obj)
                    if prim_path:
                        omni.log.info(f"Placed '{name}' at {prim_path}")
                        placed += 1
                    else:
                        skipped += 1
    
                    await asyncio.sleep(0)
    
                # すべてのオブジェクト配置が完了した後、壁を生成
                omni.log.info("=== Generating procedural walls ===")
                wall_paths: List[str] = []
                window_paths: List[str] = []
    
                rooms = layout_data.get("rooms") or []
                outer_polygon = layout_data.get("outer_polygon")
                room_polygon = layout_data.get("room_polygon")
                openings: List[Dict[str, object]] = []
    
                if rooms:
                    for room in rooms:
                        openings.extend(room.get("openings") or [])
                if isinstance(layout_data.get("openings"), list):
                    openings.extend(layout_data.get("openings") or [])
    
                openings = self._dedupe_openings(openings)
                interior_openings = [
                    op for op in openings if str(op.get("type", "") or "").lower() == "door"
                ]
                used_polygon_walls = False
    
                if rooms or outer_polygon or room_polygon:
                    try:
                        outer_edges: List[Dict[str, object]] = []
                        if outer_polygon:
                            outer_points = self._wall_generator._normalize_polygon_points(outer_polygon)
                            if len(outer_points) >= 3:
                                outer_edges = self._wall_generator._build_edges(outer_points)
                            if outer_edges:
                                wall_chunk, window_chunk = self._wall_generator.generate_walls_from_polygon(
                                    stage, root_prim_path, outer_polygon, openings
                                )
                                wall_paths.extend(wall_chunk)
                                window_paths.extend(window_chunk)
    
                        if rooms:
                            room_edges: List[Dict[str, object]] = []
                            for room in rooms:
                                polygon = room.get("room_polygon")
                                if not polygon:
                                    continue
                                room_points = self._wall_generator._normalize_polygon_points(polygon)
                                if len(room_points) < 3:
                                    continue
                                room_edges.extend(self._wall_generator._build_edges(room_points))
    
                            if room_edges:
                                merged_edges = self._merge_axis_aligned_edges(room_edges)
                                if outer_edges:
                                    merged_edges = self._filter_edges_against_outer(
                                        merged_edges, outer_edges
                                    )
                                if merged_edges:
                                    wall_chunk, window_chunk = self._wall_generator.generate_walls_from_edges(
                                        stage, root_prim_path, merged_edges, interior_openings
                                    )
                                    wall_paths.extend(wall_chunk)
                                    window_paths.extend(window_chunk)
    
                        elif room_polygon and not outer_polygon:
                            wall_chunk, window_chunk = self._wall_generator.generate_walls_from_polygon(
                                stage, root_prim_path, room_polygon, openings
                            )
                            wall_paths.extend(wall_chunk)
                            window_paths.extend(window_chunk)
    
                        used_polygon_walls = bool(wall_paths or window_paths)
                    except Exception as exc:
                        omni.log.warn(
                            f"Polygon wall generation failed, falling back to rectangular walls: {exc}"
                        )
                        used_polygon_walls = False
    
                if not used_polygon_walls:
                    area_size_x = layout_data.get("area_size_X")
                    area_size_y = layout_data.get("area_size_Y")
                    origin_mode = "center"
    
                    if isinstance(area_size_x, (int, float)) and isinstance(area_size_y, (int, float)):
                        # Heuristic: if all object coords are inside [0, size], treat origin as bottom-left.
                        try:
                            xs = [float(obj.get("X")) for obj in objects if obj.get("X") is not None]
                            ys = [float(obj.get("Y")) for obj in objects if obj.get("Y") is not None]
                            if xs and ys:
                                min_x, max_x = min(xs), max(xs)
                                min_y, max_y = min(ys), max(ys)
                                tol = 1e-3
                                if (
                                    min_x >= -tol
                                    and min_y >= -tol
                                    and max_x <= float(area_size_x) + tol
                                    and max_y <= float(area_size_y) + tol
                                ):
                                    origin_mode = "bottom_left"
                        except Exception:
                            origin_mode = "center"
    
                        wall_paths = self._create_procedural_walls(
                            stage,
                            root_prim_path,
                            float(area_size_x),
                            float(area_size_y),
                            door_objects,
                            origin_mode,
                        )
                    else:
                        omni.log.warn(
                            "Cannot generate walls: area_size_X or area_size_Y not found in layout data. "
                            f"area_size_X={area_size_x}, area_size_Y={area_size_y}"
                        )
    
                placed += len(wall_paths) + len(window_paths)
                omni.log.info(
                    f"Generated {len(wall_paths)} wall segments and {len(window_paths)} window panes"
                )
                omni.log.info(f"USD Search placement finished. Placed={placed}, Skipped={skipped}")
            except asyncio.CancelledError:
                omni.log.warn("USD Search placement cancelled.")
                raise
            except Exception as exc:
                omni.log.error(f"Unexpected error during USD Search placement: {exc}")
            finally:
                self._search_task = None
    
    async def _blacklist_and_replace_selected_asset(self, prim_path: str) -> None:
        await self._replace_selected_asset(prim_path, add_to_blacklist=True)

    async def _replace_selected_asset(self, prim_path: str, add_to_blacklist: bool = False) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            omni.log.error("No USD stage is available. Open or create a stage before replacing assets.")
            return

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            omni.log.warn(f"Invalid prim selected: {prim_path}")
            return

        ref_prim = self._get_reference_prim(prim)
        asset_url = self._get_asset_url_from_prim(ref_prim)
        if not asset_url:
            omni.log.warn(f"Selected prim has no asset reference: {prim_path}")
            return

        identity_key = await self._get_asset_identity_key(asset_url)
        if add_to_blacklist:
            added_url = self._add_asset_to_blacklist(asset_url)
            added_key = self._add_asset_key_to_blacklist(identity_key)
            if added_url or added_key:
                self._save_settings_to_json()
                self._update_blacklist_label()
                self._rebuild_blacklist_list()
                omni.log.info(
                    f"Blacklisted asset: {self._normalize_asset_url(asset_url)} (key={identity_key})"
                )
            else:
                omni.log.info(
                    f"Asset already blacklisted: {self._normalize_asset_url(asset_url)} (key={identity_key})"
                )

        data = self._build_data_from_prim_metadata(ref_prim)
        if not data:
            omni.log.warn("Selected prim has no placement metadata. Cannot replace asset in-place.")
            return

        meta = self._get_search_metadata_from_prim(ref_prim)
        search_query = meta.get("search_query") or self._build_search_query_from_object(meta)
        if not search_query:
            omni.log.warn("Unable to determine search query for selected asset.")
            return

        search_root = self._search_root_model.as_string if self._search_root_model else ""
        search_root = search_root.strip()
        if not search_root:
            omni.log.error("Search root URL is empty. Set the 'Search Root URL' field before replacing assets.")
            return
        if not search_root.startswith("omniverse://"):
            omni.log.error(f"Search root must start with 'omniverse://'. Current value: {search_root}")
            return
        normalized_root = search_root if search_root.endswith("/") else f"{search_root}/"

        replacement_url = await self._semantic_search_asset(
            search_query,
            normalized_root,
            exclude_urls={asset_url},
            exclude_keys={identity_key} if identity_key else None,
        )
        if not replacement_url:
            omni.log.warn(f"No replacement asset found for query '{search_query}'.")
            return

        omni.log.info(
            f"Replacing asset '{asset_url}' with '{replacement_url}' for prim '{ref_prim.GetPath()}'"
        )
        references = ref_prim.GetReferences()
        references.ClearReferences()
        references.AddReference(replacement_url)
        await self._apply_transform(ref_prim, data, replacement_url)


    async def _replace_selected_asset_with_url(
        self,
        prim_path: str,
        replacement_url: str,
        add_to_blacklist: bool = False,
    ) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            omni.log.error("No USD stage is available. Open or create a stage before replacing assets.")
            return

        if not replacement_url:
            omni.log.warn("Replacement URL is empty. Abort replace.")
            return

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            omni.log.warn(f"Invalid prim selected: {prim_path}")
            return

        ref_prim = self._get_reference_prim(prim)
        asset_url = self._get_asset_url_from_prim(ref_prim)
        if not asset_url:
            omni.log.warn(f"Selected prim has no asset reference: {prim_path}")
            return

        replacement_url = str(replacement_url).strip()
        if not replacement_url:
            omni.log.warn("Replacement URL is empty after normalization.")
            return

        history_key = str(ref_prim.GetPath())
        identity_key = await self._get_asset_identity_key(asset_url)
        self._append_replacement_history(history_key, asset_url, identity_key)
        if add_to_blacklist:
            added_url = self._add_asset_to_blacklist(asset_url)
            added_key = self._add_asset_key_to_blacklist(identity_key)
            if added_url or added_key:
                self._save_settings_to_json()
                self._update_blacklist_label()
                self._rebuild_blacklist_list()
                omni.log.info(
                    f"Blacklisted asset: {self._normalize_asset_url(asset_url)} (key={identity_key})"
                )
            else:
                omni.log.info(
                    f"Asset already blacklisted: {self._normalize_asset_url(asset_url)} (key={identity_key})"
                )

        data = self._build_data_from_prim_metadata(ref_prim)
        if not data:
            omni.log.warn("Selected prim has no placement metadata. Cannot replace asset in-place.")
            return

        if self._normalize_asset_url(replacement_url) == self._normalize_asset_url(asset_url):
            omni.log.info("Replacement URL matches current asset. Skipping replace.")
            return

        omni.log.info(
            f"Replacing asset '{asset_url}' with '{replacement_url}' for prim '{ref_prim.GetPath()}'"
        )
        references = ref_prim.GetReferences()
        references.ClearReferences()
        references.AddReference(replacement_url)
        await self._apply_transform(ref_prim, data, replacement_url)
        replacement_key = await self._get_asset_identity_key(replacement_url)
        self._append_replacement_history(history_key, replacement_url, replacement_key)

    def _get_or_create_root_prim(self, stage, area_name: Optional[str]) -> str:
        """配置先のルートXformを取得または作成する。"""
        token = self._sanitize_identifier(area_name or "AssetPlacer")
        root_path = f"/{token}"
        if not stage.GetPrimAtPath(root_path):
            UsdGeom.Xform.Define(stage, Sdf.Path(root_path))
        return root_path

    async def _reference_asset(
        self,
        stage,
        root_prim_path: str,
        object_name: str,
        asset_url: str,
        object_data: Dict[str, object],
    ) -> Optional[str]:
        """アセットを参照として配置し、トランスフォームを適用する。"""
        try:
            child_token = self._sanitize_identifier(object_name or "Asset")
            prim_path = self._get_unique_child_path(stage, root_prim_path, child_token)
            prim = stage.DefinePrim(prim_path, "Xform")
            references = prim.GetReferences()
            references.ClearReferences()
            references.AddReference(asset_url)
            await self._apply_transform(prim, object_data, asset_url)
            return prim_path
        except Exception as exc:
            omni.log.error(f"Failed to reference asset '{asset_url}' for '{object_name}': {exc}")
            return None

    def _get_unique_child_path(self, stage, parent_path: str, child_token: str) -> str:
        """同名Primが存在する場合は連番を付与してユニークなパスを返す。"""
        parent_sdf = Sdf.Path(parent_path)
        candidate = parent_sdf.AppendChild(child_token)
        suffix = 1
        while stage.GetPrimAtPath(candidate):
            candidate = parent_sdf.AppendChild(f"{child_token}_{suffix}")
            suffix += 1
        return str(candidate)

    async def _apply_transform(self, prim, data: Dict[str, object], asset_url: Optional[str] = None) -> None:
        """
        レイアウト情報から取得したトランスフォームをPrimに設定する。

        JSONの寸法（メートル単位）を絶対的な最終寸法として扱い、
        アセットの元のサイズに基づいてスケール比を計算して適用する。
        """
        # アセットのロード完了を待機（最大10フレーム）
        # 参照アセットのジオメトリがロードされるまで、BBoxは正しく計算されない
        max_retries = 10
        time_code = Usd.TimeCode.Default()

        # 参照先のupAxisを取得（キャッシュ付き）。asset_urlが未指定の場合は参照を調査。
        if not asset_url:
            asset_url = self._get_asset_url_from_prim(prim)

        up_axis = self._get_asset_up_axis(asset_url) if asset_url else "Z"
        omni.log.info(f"[UpAxis] Using upAxis='{up_axis}' for prim '{prim.GetPath()}', asset='{asset_url}'")

        for attempt in range(max_retries):
            # 1フレーム待機してUSDがアセットをロードする時間を与える
            await omni.kit.app.get_app().next_update_async()

            # BBoxが有効かチェック
            bbox_cache = UsdGeom.BBoxCache(time_code, ["default"])
            bbox = bbox_cache.ComputeWorldBound(prim)
            bbox_range = bbox.ComputeAlignedRange()
            size_vec = bbox_range.GetMax() - bbox_range.GetMin()

            # サイズが非ゼロ（ジオメトリがロードされた）なら待機完了
            if size_vec[0] > 1e-6 or size_vec[1] > 1e-6 or size_vec[2] > 1e-6:
                omni.log.info(f"Asset geometry loaded after {attempt + 1} frame(s)")
                break
        else:
            # 最大試行回数に達してもロードされない場合は警告
            omni.log.warn(f"Asset geometry may not be fully loaded after {max_retries} frames")

        xformable = UsdGeom.Xformable(prim)

        # ステップ1: 既存トランスフォームのクリア
        try:
            for op in xformable.GetOrderedXformOps():
                prim.RemoveProperty(op.GetAttr().GetName())
            if hasattr(xformable, "ClearXformOpOrder"):
                xformable.ClearXformOpOrder()
            else:
                xformable.SetXformOpOrder([])
        except Exception as exc:  # pragma: no cover - defensive cleanup
            omni.log.warn(f"Failed clearing existing xform ops on '{prim.GetPath()}': {exc}")

        # ステップ2: 「元のサイズ」の計算
        # 待機処理で既に計算したBBoxを再利用
        # （念のため再計算して最新の状態を取得）
        bbox_cache = UsdGeom.BBoxCache(time_code, ["default"])
        bbox = bbox_cache.ComputeWorldBound(prim)
        bbox_range = bbox.ComputeAlignedRange()

        # 元のサイズ（メートル単位）を取得
        original_size_vec = bbox_range.GetMax() - bbox_range.GetMin()
        original_size_x = original_size_vec[0]
        original_size_y = original_size_vec[1]
        original_size_z = original_size_vec[2]

        omni.log.info(
            f"Original asset size (m): X={original_size_x:.4f}, Y={original_size_y:.4f}, Z={original_size_z:.4f}"
        )

        # ステップ3: 回転を先に取得（Z軸周り）
        rotation = self._extract_optional_float_by_keys(
            data,
            [
                "rotationZ",
                "RotationZ",
                "rotation_z",
                "Rotation_Z",
                "rotationY",
                "RotationY",
                "rotation_y",
                "Rotation_Y",
                "rotation",
                "Rotation",
            ],
        )

        base_rotation = rotation if rotation is not None else 0.0
        rotation_offset = self._get_asset_rotation_offset(asset_url)
        effective_rotation = (base_rotation + rotation_offset) % 360.0
        if rotation_offset:
            omni.log.info(
                f"[Transform] rotation base={base_rotation} deg, offset={rotation_offset} deg, "
                f"effective={effective_rotation} deg"
            )

        # ステップ4: 「目標のサイズ」の読み取り（Z-Up座標系）
        # JSONからLength, Height, Width（メートル単位）を読み取り
        json_length = self._extract_float(data, "Length", 0.0)
        json_height = self._extract_float(data, "Height", 0.0)
        json_width = self._extract_float(data, "Width", 0.0)
        object_name = str(data.get("object_name", "") or "")
        category = str(data.get("category", "") or "")
        category_lower = category.lower()
        is_door = category_lower == "door" or "door" in object_name.lower()


        # Z-Up axis mapping
        # 0/180 deg: Length->X, Width->Y
        # 90/270 deg: swap Length/Width (Width->X, Length->Y)
        if is_door:
            # Door sizing: map width/thickness to the asset's horizontal axes (no rotation-based swap).
            door_width = max(json_length, json_width)
            door_thickness = min(json_length, json_width)
            if door_width <= 0.0 and json_length > 0.0:
                door_width = json_length
            if door_thickness <= 0.0 and json_width > 0.0:
                door_thickness = json_width

            if up_axis == "Y":
                # Horizontal axes are X and Z when Y is up.
                if original_size_x >= original_size_z:
                    target_size_x = door_width
                    target_size_y = door_thickness  # target_y maps to local Z in _compute_scale_with_up_axis
                else:
                    target_size_x = door_thickness
                    target_size_y = door_width
            else:
                # Z-up: horizontal axes are X and Y.
                if original_size_x >= original_size_y:
                    target_size_x = door_width
                    target_size_y = door_thickness
                else:
                    target_size_x = door_thickness
                    target_size_y = door_width

            omni.log.info(
                f"[Transform] Door sizing: width={door_width:.4f}, thickness={door_thickness:.4f}, "
                f"mapped_to=({target_size_x:.4f}, {target_size_y:.4f})"
            )
        else:
            rotation_for_scale = effective_rotation
            if abs(rotation_for_scale - 90) < 45 or abs(rotation_for_scale - 270) < 45:
                target_size_x = json_width   # Width -> X
                target_size_y = json_length  # Length -> Y
                omni.log.info(
                    "Rotation "
                    f"{rotation_for_scale} deg detected: swapping Length/Width mapping (Width->X, Length->Y)"
                )
            else:
                target_size_x = json_length  # Length -> X
                target_size_y = json_width   # Width -> Y


        target_size_z = json_height  # Height → Z軸（常に同じ）

        # ステップ5: 「スケール比」の計算（upAxisに応じてY/Zを入れ替え）
        scale_x, scale_y, scale_z = self._compute_scale_with_up_axis(
            up_axis,
            (original_size_x, original_size_y, original_size_z),
            (target_size_x, target_size_y, target_size_z),
        )
        final_scale = Gf.Vec3f(scale_x, scale_y, scale_z)

        omni.log.info(f"[Transform] up_axis={up_axis}, scale=({scale_x:.4f}, {scale_y:.4f}, {scale_z:.4f})")
        omni.log.info(
            f"[Transform] Target size (m): X={target_size_x:.4f}, Y={target_size_y:.4f}, Z={target_size_z:.4f}"
        )

        # ステップ6: XformOpsの追加（適用順: Scale -> RotateX(必要時) -> RotateZ -> Translate）
        # 追加順は逆（Translate -> RotateZ -> RotateX -> Scale）
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))

        rotate_z_op = xformable.AddRotateZOp()
        rotate_z_op.Set(effective_rotation)

        if up_axis == "Y":
            rotate_x_op = xformable.AddRotateXOp()
            rotate_x_op.Set(90.0)

        scale_op = xformable.AddScaleOp()
        scale_op.Set(final_scale)

        # ステップ7: 回転+スケール後のBBoxを取得し、床合わせのためのTranslate Zを決定
        bbox_cache2 = UsdGeom.BBoxCache(time_code, ["default"])
        bbox2 = bbox_cache2.ComputeWorldBound(prim)
        bbox_range2 = bbox2.ComputeAlignedRange()
        min_after = bbox_range2.GetMin()
        max_after = bbox_range2.GetMax()
        min_z_after_rot_scale = min_after[2]
        center_x_after = (min_after[0] + max_after[0]) * 0.5
        center_y_after = (min_after[1] + max_after[1]) * 0.5

        # ステップ8: JSON座標を適用（Zは床合わせで上書き）
        x = self._extract_float(data, "X", 0.0)
        y = self._extract_float(data, "Y", 0.0)
        # object_name/category/is_door are extracted above
        if is_door and (abs(center_x_after) > 1e-4 or abs(center_y_after) > 1e-4):
            omni.log.info(
                f"[Transform] Door center align: center_offset=({center_x_after:.4f}, {center_y_after:.4f})"
            )
            x -= center_x_after
            y -= center_y_after
        search_prompt = str(data.get("search_prompt", "") or "")
        search_query = self._build_search_query_from_object({
            "object_name": object_name,
            "category": category,
            "search_prompt": search_prompt,
        })

        self._store_placement_metadata(
            prim,
            asset_url,
            base_rotation,
            json_length,
            json_width,
            json_height,
            x,
            y,
            object_name=object_name,
            category=category,
            search_prompt=search_prompt,
            search_query=search_query,
        )
        translate_z = -min_z_after_rot_scale
        translate_op.Set(Gf.Vec3d(x, y, translate_z))

        omni.log.info(
            f"[Transform] bbox_min_after_rot_scale={min_z_after_rot_scale:.4f}, "
            f"final_translate=({x:.4f}, {y:.4f}, {translate_z:.4f})"
        )

    def _select_asset_url(
        self,
        response,
        host: str,
        object_name: str,
    ) -> Optional[str]:
        """USD Searchの結果から最適なアセットURLを選択する。"""
        paths = getattr(response, "paths", None)
        if not paths:
            omni.log.warn(f"No paths found in search response for '{object_name}'")
            return None

        normalized_target = self._normalize_name(object_name)
        candidates: List[str] = []
        for path in paths:
            uri = getattr(path, "uri", None)
            if not uri:
                continue
            candidate = f"omniverse://{host}{uri}"
            if not candidate.lower().endswith(".usd"):
                continue
            candidates.append(candidate)

        if not candidates:
            omni.log.warn(f"No USD candidates found for '{object_name}' (searched {len(paths)} paths)")
            return None

        omni.log.info(f"Found {len(candidates)} USD candidates for '{object_name}':")
        for candidate in candidates:
            omni.log.info(f"  - {candidate}")

        # Step 1: 完全一致を試みる
        omni.log.info(f"Trying exact match for normalized target: '{normalized_target}'")
        for candidate in candidates:
            base = candidate.split("/")[-1]
            normalized_base = self._normalize_name(base.split(".")[0])
            omni.log.info(f"  Comparing '{normalized_base}' == '{normalized_target}'")
            if normalized_base == normalized_target:
                omni.log.info(f"  -> Exact match found: {candidate}")
                return candidate

        # Step 2: 部分一致を試みる（クエリの各単語がファイル名と一致するか）
        omni.log.info(f"Trying partial match with query words: {object_name.lower().split()}")
        query_words = object_name.lower().split()
        for candidate in candidates:
            base = candidate.split("/")[-1]
            normalized_base = self._normalize_name(base.split(".")[0])
            for word in query_words:
                normalized_word = self._normalize_name(word)
                omni.log.info(f"  Checking if '{normalized_word}' matches '{normalized_base}'")
                # 単語がファイル名と完全一致、またはファイル名に含まれる場合
                if normalized_word == normalized_base or normalized_word in normalized_base:
                    omni.log.info(f"  -> Partial match found: {candidate}")
                    return candidate

        # Step 3: フォールバック（最初の候補を返す）
        omni.log.info(f"No match found, using first candidate: {candidates[0]}")
        return candidates[0]

    @staticmethod
    def _sanitize_identifier(name: str) -> str:
        token = Tf.MakeValidIdentifier(name) if name else ""
        return token or "Item"

    @staticmethod
    def _normalize_name(value: str) -> str:
        return "".join(ch for ch in value.lower() if ch.isalnum())

    def _extract_float(self, data: Dict[str, object], key: str, default: float = 0.0) -> float:
        value = self._get_value_case_insensitive(data, key)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            omni.log.warn(f"Invalid numeric value '{value}' for key '{key}'. Using default {default}.")
            return default

    def _extract_optional_float(self, data: Dict[str, object], key: str) -> Optional[float]:
        value = self._get_value_case_insensitive(data, key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            omni.log.warn(f"Invalid numeric value '{value}' for key '{key}'.")
            return None

    def _extract_optional_float_by_keys(self, data: Dict[str, object], keys: List[str]) -> Optional[float]:
        for key in keys:
            value = self._extract_optional_float(data, key)
            if value is not None:
                return value
        return None

    @staticmethod
    def _get_value_case_insensitive(data: Dict[str, object], key: str) -> Optional[object]:
        if key in data:
            return data[key]
        lowered = key.lower()
        for existing_key, value in data.items():
            if isinstance(existing_key, str) and existing_key.lower() == lowered:
                return value
        return None

    def _create_procedural_floor(
        self,
        stage,
        root_prim_path: str,
        object_data: Dict[str, object],
    ) -> Optional[str]:
        """
        JSONデータから床を手続き的に生成します。

        Args:
            stage: USDステージ
            root_prim_path: 親Primのパス
            object_data: JSONオブジェクトデータ（Length, Width, X, Z を含む）

        Returns:
            生成された床PrimのパスまたはNone
        """
        return self._floor_generator.generate(stage, root_prim_path, object_data)

    def _create_procedural_walls(
        self,
        stage,
        root_prim_path: str,
        area_size_x: float,
        area_size_y: float,
        door_objects: List[Dict[str, object]] = None,
        origin_mode: str = "center",
    ) -> List[str]:
        """
        部屋のサイズに基づいて4つの壁を手続き的に生成します。
        ドアの位置がある場合は、その部分を切り抜きます。

        Args:
            stage: USDステージ
            root_prim_path: 親Primのパス
            area_size_x: 部屋のX方向のサイズ（メートル単位）
            area_size_y: 部屋のY方向のサイズ（メートル単位）
            door_objects: ドアオブジェクトのリスト（切り抜き用）
            origin_mode: "center" or "bottom_left"

        Returns:
            生成された壁Primのパスのリスト
        """
        return self._wall_generator.generate_walls(
            stage, root_prim_path, area_size_x, area_size_y, door_objects, origin_mode
        )

    @staticmethod
    def _dedupe_openings(
        openings: List[Dict[str, object]], tol: float = 1e-3
    ) -> List[Dict[str, object]]:
        deduped: List[Dict[str, object]] = []
        seen = set()
        for opening in openings or []:
            try:
                x = float(opening.get("X", 0.0))
                y = float(opening.get("Y", 0.0))
                width = float(opening.get("Width", 0.0))
                height = float(opening.get("Height", 0.0))
                sill = float(opening.get("SillHeight", 0.0))
            except (TypeError, ValueError):
                deduped.append(opening)
                continue
            opening_type = str(opening.get("type", "") or "").lower()
            key = (
                opening_type,
                round(x / tol) * tol,
                round(y / tol) * tol,
                round(width / tol) * tol,
                round(height / tol) * tol,
                round(sill / tol) * tol,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(opening)
        return deduped

    @staticmethod
    def _edge_key(
        p0: Tuple[float, float], p1: Tuple[float, float], tol: float = 1e-3
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        def quantize(point: Tuple[float, float]) -> Tuple[float, float]:
            return (round(point[0] / tol) * tol, round(point[1] / tol) * tol)

        a = quantize(p0)
        b = quantize(p1)
        return tuple(sorted((a, b)))

    def _collect_room_edges(
        self, rooms: List[Dict[str, object]], tol: float = 1e-3
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
        edge_map: Dict[Tuple[Tuple[float, float], Tuple[float, float]], Dict[str, object]] = {}
        for room in rooms or []:
            polygon = room.get("room_polygon")
            points = self._wall_generator._normalize_polygon_points(polygon)
            if len(points) < 3:
                continue
            edges = self._wall_generator._build_edges(points)
            for edge in edges:
                key = self._edge_key(edge["start"], edge["end"], tol)
                if key in edge_map:
                    edge_map[key]["count"] += 1
                else:
                    edge_map[key] = {"edge": edge, "count": 1}

        all_edges = [entry["edge"] for entry in edge_map.values()]
        interior_edges = [entry["edge"] for entry in edge_map.values() if entry["count"] > 1]
        exterior_edges = [entry["edge"] for entry in edge_map.values() if entry["count"] == 1]
        return all_edges, interior_edges, exterior_edges

    @staticmethod
    def _edge_axis_signature(
        edge: Dict[str, object], tol: float = 1e-4
    ) -> Optional[Tuple[str, float, float, float]]:
        p0 = edge.get("start")
        p1 = edge.get("end")
        if not p0 or not p1:
            return None
        x0, y0 = p0
        x1, y1 = p1
        if abs(x0 - x1) < tol:
            x = round(x0 / tol) * tol
            y_min, y_max = sorted([y0, y1])
            return ("v", x, y_min, y_max)
        if abs(y0 - y1) < tol:
            y = round(y0 / tol) * tol
            x_min, x_max = sorted([x0, x1])
            return ("h", y, x_min, x_max)
        return None

    @staticmethod
    def _build_edge_from_points(
        p0: Tuple[float, float], p1: Tuple[float, float]
    ) -> Optional[Dict[str, object]]:
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            return None
        unit = (dx / length, dy / length)
        angle = math.degrees(math.atan2(dy, dx))
        return {
            "start": (float(p0[0]), float(p0[1])),
            "end": (float(p1[0]), float(p1[1])),
            "unit": unit,
            "length": float(length),
            "angle": float(angle),
        }

    def _merge_axis_aligned_edges(
        self, edges: List[Dict[str, object]], tol: float = 1e-4
    ) -> List[Dict[str, object]]:
        from collections import defaultdict

        horiz = defaultdict(list)
        vert = defaultdict(list)
        others: List[Dict[str, object]] = []

        for edge in edges or []:
            sig = self._edge_axis_signature(edge, tol)
            if not sig:
                others.append(edge)
                continue
            axis, coord, s0, s1 = sig
            if axis == "h":
                horiz[coord].append((s0, s1))
            else:
                vert[coord].append((s0, s1))

        def merge_ranges(ranges: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
            ranges = sorted(ranges, key=lambda r: r[0])
            merged: List[List[float]] = []
            for start, end in ranges:
                if not merged or start > merged[-1][1] + tol:
                    merged.append([start, end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)
            return [(r[0], r[1]) for r in merged]

        merged_edges: List[Dict[str, object]] = []
        for y, ranges in horiz.items():
            for x0, x1 in merge_ranges(ranges):
                edge = self._build_edge_from_points((x0, y), (x1, y))
                if edge:
                    merged_edges.append(edge)

        for x, ranges in vert.items():
            for y0, y1 in merge_ranges(ranges):
                edge = self._build_edge_from_points((x, y0), (x, y1))
                if edge:
                    merged_edges.append(edge)

        merged_edges.extend(others)
        return merged_edges

    @staticmethod
    def _edge_overlaps_any(
        sig: Tuple[str, float, float, float],
        outer_sigs: List[Tuple[str, float, float, float]],
        tol: float = 1e-4,
    ) -> bool:
        axis, coord, s0, s1 = sig
        for osig in outer_sigs:
            if osig[0] != axis:
                continue
            if abs(osig[1] - coord) > tol:
                continue
            o0, o1 = osig[2], osig[3]
            if min(s1, o1) - max(s0, o0) > tol:
                return True
        return False

    def _filter_edges_against_outer(
        self,
        edges: List[Dict[str, object]],
        outer_edges: List[Dict[str, object]],
        tol: float = 1e-4,
    ) -> List[Dict[str, object]]:
        outer_sigs = [self._edge_axis_signature(edge, tol) for edge in outer_edges or []]
        outer_sigs = [sig for sig in outer_sigs if sig]
        if not outer_sigs:
            return edges

        filtered: List[Dict[str, object]] = []
        for edge in edges or []:
            sig = self._edge_axis_signature(edge, tol)
            if sig and self._edge_overlaps_any(sig, outer_sigs, tol):
                continue
            filtered.append(edge)
        return filtered

    def _normalize_furniture_name(self, json_name: str) -> str:
        """
        AIが生成したJSONの家具名を、USD Searchが理解できる
        正規化された検索クエリに変換します。
        """

        # 1. 既知の名前をマッピング
        #    (この辞書は必要に応じて拡張してください)
        name_mapping = {
            # ベッド
            "accessible_master_bed": "bed",
            "left_side_access_bed": "bed",
            "master_bed": "bed",
            # 冷蔵庫
            "compact_refrigerator": "refrigerator",
            "tall_refrigerator": "refrigerator",
            # 椅子
            "support_chair": "chair",
            "office_chair": "chair",
            "upper_corner_chair": "chair",
            "transfer_chair": "chair",
            # テレビ
            "wall_mounted_tv": "television",
            "left_wall_wallmount_tv": "television",
            "slim_wall_tv": "television",
            "wall_tv": "television",
            # 引き出し・収納
            "accessible_drawer": "drawer",
            "lower_corner_drawer": "drawer",
            "low_drawer": "drawer",
            # ドア
            "main_entrance_door": "door",
            "entrance_door": "door",
            "inward_swing_door": "door",
            "swing_door": "door",
            # その他
            "office_table_1": "office table",
            "office_cabinet": "cabinet",
            "office_lamp": "lamp",
            "office_printer": "printer",
            "single_spiral_notepad": "notepad",
        }

        normalized_name = json_name.lower().replace("_", " ")

        # マッピング辞書（キーを小文字化・スペース化して）で検索
        for key, value in name_mapping.items():
            if key.replace("_", " ") in normalized_name:
                omni.log.info(f"Normalized '{json_name}' -> '{value}' (using mapping)")
                return value

        # 2. マッピングがない場合のフォールバック
        #    アンダースコアを除去して最後の単語を使用
        words = normalized_name.split()
        if words:
            fallback_query = words[-1]
            omni.log.info(f"Normalized '{json_name}' -> '{fallback_query}' (using fallback)")
            return fallback_query

        omni.log.warn(f"Could not normalize '{json_name}', using original.")
        return json_name

    def _build_search_query_from_object(self, obj: Dict[str, object]) -> str:
        """
        JSONオブジェクト1件からセマンティック検索用のクエリ文字列を組み立てる。

        優先順:
        1) search_prompt があればそれをベースに使用
        2) search_prompt が空で object_name があれば _normalize_furniture_name で正規化
        3) どちらも無ければ汎用プレースホルダを使う
        category があれば先頭に付与する。
        """
        object_name = str(obj.get("object_name", "") or "").strip()
        category = str(obj.get("category", "") or "").strip()
        search_prompt = str(obj.get("search_prompt", "") or "").strip()

        if search_prompt:
            base_text = search_prompt
        elif object_name:
            base_text = self._normalize_furniture_name(object_name)
        else:
            base_text = "generic furniture object"

        if category:
            query = f"{category}, {base_text}"
        else:
            query = base_text

        omni.log.info(
            f"[SearchQuery] category='{category}', name='{object_name}', "
            f"search_prompt_len={len(search_prompt)}, query='{query}'"
        )
        return query

    @staticmethod
    def _compute_scale_with_up_axis(
        up_axis: str,
        original_sizes: Tuple[float, float, float],
        target_sizes: Tuple[float, float, float],
        min_size: float = 1e-6,
    ) -> Tuple[float, float, float]:
        """
        upAxisに応じたスケール比を計算する。

        Z-upステージでの想定:
        - Z-upアセット: (X→X, Y→Y, Z→Z)
        - Y-upアセット: RotateX(+90) を適用してZ-upに補正するため、(X→X, Z→Y, Y→Z) を考慮
        """
        original_x, original_y, original_z = original_sizes
        target_x, target_y, target_z = target_sizes

        def safe_div(num: float, denom: float) -> float:
            if abs(denom) > min_size and num > 0:
                return num / denom
            return 1.0

        if up_axis == "Y":
            # RotateX(+90) により local Y -> world Z, local Z -> world Y
            scale_x = safe_div(target_x, original_x)
            scale_y = safe_div(target_z, original_y)
            scale_z = safe_div(target_y, original_z)
        else:
            scale_x = safe_div(target_x, original_x)
            scale_y = safe_div(target_y, original_y)
            scale_z = safe_div(target_z, original_z)

        return scale_x, scale_y, scale_z

    def _get_asset_up_axis(self, asset_url: str) -> str:
        """
        アセットのupAxisを取得し、キャッシュする。未知・取得失敗時は'Y'を返す。
        """
        if not asset_url:
            omni.log.warn("[UpAxis] asset_url is empty. Fallback to 'Y'.")
            return "Y"

        if not hasattr(self, "_asset_up_axis_cache"):
            self._asset_up_axis_cache = {}

        cached = self._asset_up_axis_cache.get(asset_url)
        if cached:
            omni.log.info(f"[UpAxis] Cache hit: {asset_url} -> {cached}")
            return cached

        up_axis = "Y"
        try:
            stage = Usd.Stage.Open(asset_url)
            if not stage:
                omni.log.warn(f"[UpAxis] Failed to open stage for {asset_url}. Fallback to 'Y'.")
            else:
                axis_token = UsdGeom.GetStageUpAxis(stage)
                up_axis = "Z" if axis_token == UsdGeom.Tokens.z else "Y"
                omni.log.info(f"[UpAxis] Retrieved upAxis='{up_axis}' for {asset_url}")
        except Exception as exc:
            omni.log.warn(f"[UpAxis] Error while getting upAxis for {asset_url}: {exc}. Fallback to 'Y'.")

        self._asset_up_axis_cache[asset_url] = up_axis
        return up_axis

    async def _get_asset_size(self, asset_url: str) -> Optional[int]:
        if not asset_url:
            return None
        normalized = self._normalize_asset_url(asset_url)
        cache = getattr(self, "_asset_size_cache", None)
        if cache is None:
            self._asset_size_cache = {}
            cache = self._asset_size_cache
        if normalized in cache:
            return cache[normalized]
        try:
            import omni.client
            result, entry = await omni.client.stat_async(normalized)
            if result == omni.client.Result.OK and entry:
                size = int(entry.size) if entry.size is not None else None
            else:
                size = None
        except Exception as exc:
            omni.log.warn(f"[AssetSize] Failed to stat '{normalized}': {exc}")
            size = None
        cache[normalized] = size
        return size

    async def _get_asset_identity_key(
        self, asset_url: str, size_bytes: Optional[int] = None
    ) -> Optional[str]:
        if not asset_url:
            return None
        if size_bytes is None:
            size_bytes = await self._get_asset_size(asset_url)
        return self._build_asset_identity_key(asset_url, size_bytes)

    async def _semantic_search_asset(
        self,
        query: str,
        search_root: str,
        exclude_urls: Optional[set] = None,
        exclude_keys: Optional[set] = None,
    ) -> Optional[str]:
        """
        外部セマンティック検索API (Vector Search) を使用してアセットを検索します。
        GETではなくPOSTを使用し、CLIP埋め込みベクトルによる直接検索を行います。

        Args:
            query: 検索クエリ (例: "A comfortable chair")
            search_root: 検索ルートURL（今回は使用しないが互換性のため残す）
        Returns:
            アセットのOmniverse URL、または見つからない場合はNone
        """
        import requests
        import asyncio
        import json

        # エンドポイント設定
        api_url = "http://192.168.11.65:30080/search"
        api_basic_auth = "omniverse:tsukuverse"

        # --- Vector Search用ペイロード構築 ---
        payload = {
            "vector_queries": [
                {
                    "field_name": "clip-embedding.embedding",
                    "query_type": "text",
                    "query": query,
                }
            ],
            "return_metadata": True,
            "return_images": False,
            "limit": VECTOR_SEARCH_LIMIT,
            "file_extension_include": "usd,usda,usdc,usdz",
            "file_extension_exclude": "png,jpg,jpeg",
        }

        omni.log.info(f"[Vector Search] POST Request to: {api_url}")
        omni.log.info(f"  Payload: {json.dumps(payload, ensure_ascii=False)}")

        try:
            # 同期処理を別スレッドで実行（UIをブロックしない）
            loop = asyncio.get_event_loop()

            def sync_request():
                user, password = api_basic_auth.split(":", 1)
                response = requests.post(api_url, json=payload, auth=(user, password), timeout=60)
                response.raise_for_status()
                return response.json()

            result = await loop.run_in_executor(None, sync_request)

            omni.log.info("[Vector Search] API Response received")
            omni.log.info(f"  Response: {json.dumps(result, ensure_ascii=False)[:500]}")

            # --- レスポンスからUSDパスを抽出 ---
            items = []
            if isinstance(result, list):
                items = result
            elif isinstance(result, dict):
                if "results" in result:
                    items = result["results"]
                elif "hits" in result:
                    if isinstance(result["hits"], dict) and "hits" in result["hits"]:
                        items = result["hits"]["hits"]
                elif "items" in result:
                    items = result["items"]

            if not items:
                omni.log.warn(f"[Vector Search] No results found for '{query}'")
                return None

            usd_extensions = (".usd", ".usda", ".usdc", ".usdz")
            excluded_urls = set()
            for url in getattr(self, "_asset_blacklist", set()):
                if url:
                    excluded_urls.add(self._normalize_asset_url(str(url)))
            if exclude_urls:
                for url in exclude_urls:
                    if url:
                        excluded_urls.add(self._normalize_asset_url(str(url)))

            excluded_keys = set(getattr(self, "_asset_blacklist_keys", set()))
            if exclude_keys:
                for key in exclude_keys:
                    if key:
                        excluded_keys.add(str(key))
            seen_keys = set()

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

            candidates = []
            for item in items:
                item_data = item.get("_source", item) if isinstance(item, dict) else {}
                candidate_keys = ["uri", "url", "path", "file_path", "asset_path"]
                size_val = _extract_size(item_data) or _extract_size(item)
                for key in candidate_keys:
                    if isinstance(item_data, dict) and key in item_data:
                        val = item_data[key]
                        if val and isinstance(val, str):
                            if val.lower().endswith(usd_extensions):
                                candidates.append((val, size_val))
                                break
                            omni.log.info(f"[Vector Search] Skipping non-USD file: {val}")

            if not candidates:
                omni.log.warn(f"[Vector Search] No USD files found in results for '{query}'")
                return None

            for candidate_url, candidate_size in candidates:
                normalized_url = self._normalize_asset_url(candidate_url)
                if normalized_url in excluded_urls:
                    omni.log.info(f"[Vector Search] Skipping blacklisted asset: {normalized_url}")
                    continue

                size_bytes = candidate_size
                if size_bytes is None:
                    size_bytes = await self._get_asset_size(normalized_url)

                identity_key = self._build_asset_identity_key(normalized_url, size_bytes)
                if identity_key and identity_key in excluded_keys:
                    omni.log.info(
                        f"[Vector Search] Skipping blacklisted key: {identity_key} ({normalized_url})"
                    )
                    continue
                if identity_key and identity_key in seen_keys:
                    continue
                if identity_key:
                    seen_keys.add(identity_key)

                omni.log.info(f"[Vector Search] Final Asset URL: {normalized_url}")
                return normalized_url

            omni.log.warn(f"[Vector Search] No USD files found after filtering for '{query}'")
            return None

        except requests.exceptions.RequestException as exc:
            omni.log.error(f"[Vector Search] HTTP request failed: {exc}")
            return None
        except Exception as exc:
            omni.log.error(f"[Vector Search] Unexpected error: {exc}")
            import traceback
            omni.log.error(f"Stack trace:\n{traceback.format_exc()}")
            return None

    def _save_generation_log(
        self,
        step1_stats: Dict[str, Any],
        step2_stats: Dict[str, Any],
        analysis_text: str,
        layout_json: Dict[str, Any],
        collision_results: List[Tuple[str, str, float]],
        model_name: str,
        step2_model_name: str,
        ai_overrides: Dict[str, Any],
        image_path: str,
        prompt1_text: str,
        prompt2_text: str,
        dimensions_text: str,
        tag: str = "full"
    ) -> Optional[str]:
        """AI生成の全行程をログファイルに保存する"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            log_dir = self._get_log_dir()
            os.makedirs(log_dir, exist_ok=True)
            
            log_filename = f"{timestamp}_{tag}.log"
            log_path = os.path.join(log_dir, log_filename)

            total_time = (step1_stats.get("time", 0) or 0) + (step2_stats.get("time", 0) or 0)
            total_prompt = (step1_stats.get("prompt_tokens", 0) or 0) + (step2_stats.get("prompt_tokens", 0) or 0)
            total_completion = (step1_stats.get("completion_tokens", 0) or 0) + (step2_stats.get("completion_tokens", 0) or 0)
            total_tokens = (step1_stats.get("total_tokens", 0) or 0) + (step2_stats.get("total_tokens", 0) or 0)

            lines = []
            lines.append("AI Layout Generation Log")
            lines.append(f"timestamp: {timestamp}")
            lines.append("")
            lines.append("[AI Settings]")
            lines.append(f"step1_model: {model_name}")
            lines.append(f"step2_model: {step2_model_name}")
            for k, v in ai_overrides.items():
                lines.append(f"{k}: {v}")
            lines.append("")
            
            lines.append("[Step 1 Stats]")
            lines.append(f"time_sec: {step1_stats.get('time', 0)}")
            lines.append(f"input_tokens: {step1_stats.get('prompt_tokens', 0)}")
            lines.append(f"output_tokens: {step1_stats.get('completion_tokens', 0)}")
            lines.append(f"total_tokens: {step1_stats.get('total_tokens', 0)}")
            lines.append("")

            lines.append("[Step 2 Stats]")
            lines.append(f"time_sec: {step2_stats.get('time', 0)}")
            lines.append(f"input_tokens: {step2_stats.get('prompt_tokens', 0)}")
            lines.append(f"output_tokens: {step2_stats.get('completion_tokens', 0)}")
            lines.append(f"total_tokens: {step2_stats.get('total_tokens', 0)}")
            lines.append("")

            lines.append("[Total Stats]")
            lines.append(f"time_sec: {total_time}")
            lines.append(f"input_tokens: {total_prompt}")
            lines.append(f"output_tokens: {total_completion}")
            lines.append(f"total_tokens: {total_tokens}")
            lines.append("")

            lines.append("[Analysis Text]")
            lines.append(analysis_text)
            lines.append("")

            lines.append("[AABB Collisions]")
            if collision_results:
                for name1, name2, volume in collision_results:
                    lines.append(f"- {name1} vs {name2}: overlap_volume={volume}")
            else:
                lines.append("No collisions detected.")
            lines.append("")

            lines.append("[Room Size]")
            lines.append(f"area_size_X: {layout_json.get('area_size_X', 'N/A')}")
            lines.append(f"area_size_Y: {layout_json.get('area_size_Y', 'N/A')}")
            lines.append("")

            lines.append("[Inputs]")
            lines.append(f"image_path: {image_path}")
            lines.append("")

            lines.append("[dimensions.txt]")
            lines.append(dimensions_text)
            lines.append("")

            lines.append("[prompt1]")
            lines.append(prompt1_text)
            lines.append("")

            lines.append("[prompt2]")
            lines.append(prompt2_text)
            lines.append("")

            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            
            omni.log.info(f"Detailed generation log saved to: {log_path}")
            return log_path

        except Exception as e:
            omni.log.error(f"Failed to save generation log: {e}")
            import traceback
            omni.log.error(traceback.format_exc())
            return None
