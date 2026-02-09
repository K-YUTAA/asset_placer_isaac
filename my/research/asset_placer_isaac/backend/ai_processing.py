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
AI processing functions for layout generation using OpenAI API.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Tuple, Optional

import omni.log
import httpx

# タイムアウト設定: Reasoning モデルは長時間かかるため20分に設定
# デフォルトの10分ではタイムアウトしてリトライが発生する
DEFAULT_API_TIMEOUT = 1200.0  # 20分

DEFAULT_REASONING_EFFORT = "high"
DEFAULT_TEXT_VERBOSITY = "high"
DEFAULT_MAX_OUTPUT_TOKENS = 32000
DEFAULT_IMAGE_DETAIL = "high"


def _get_usage_value(usage: Any, key: str, default: Any = None) -> Any:
    if usage is None:
        return default
    if isinstance(usage, dict):
        return usage.get(key, default)
    return getattr(usage, key, default)


def _get_usage_tokens(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "source": "none"}

    input_tokens = _get_usage_value(usage, "input_tokens", None)
    output_tokens = _get_usage_value(usage, "output_tokens", None)
    total_tokens = _get_usage_value(usage, "total_tokens", None)

    if input_tokens is None and output_tokens is None:
        input_tokens = _get_usage_value(usage, "prompt_tokens", 0)
        output_tokens = _get_usage_value(usage, "completion_tokens", 0)
        total_tokens = _get_usage_value(usage, "total_tokens", input_tokens + output_tokens)
        return {
            "input_tokens": input_tokens or 0,
            "output_tokens": output_tokens or 0,
            "total_tokens": total_tokens or 0,
            "source": "prompt/completion",
        }

    if total_tokens is None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return {
        "input_tokens": input_tokens or 0,
        "output_tokens": output_tokens or 0,
        "total_tokens": total_tokens or 0,
        "source": "input/output",
    }


def _get_reasoning_tokens(usage: Any) -> Optional[int]:
    details = _get_usage_value(usage, "output_tokens_details", None)
    if details is None:
        details = _get_usage_value(usage, "completion_tokens_details", None)
    if details is None:
        return None

    if isinstance(details, dict):
        return details.get("reasoning_tokens")
    return getattr(details, "reasoning_tokens", None)


def _log_response_details(response: Any) -> None:
    model_used = getattr(response, "model", None)
    omni.log.info(f"model_used={model_used}")

    usage = getattr(response, "usage", None)
    if usage is None:
        omni.log.info("usage=None")
        return

    tokens = _get_usage_tokens(usage)
    if tokens["source"] == "prompt/completion":
        omni.log.info(
            f"usage: prompt={tokens['input_tokens']}, completion={tokens['output_tokens']}, total={tokens['total_tokens']}"
        )
    else:
        omni.log.info(
            f"usage: input={tokens['input_tokens']}, output={tokens['output_tokens']}, total={tokens['total_tokens']}"
        )

    details = _get_usage_value(usage, "output_tokens_details", None)
    if details is None:
        details = _get_usage_value(usage, "completion_tokens_details", None)
    if details is not None:
        reasoning_tokens = _get_reasoning_tokens(usage)
        omni.log.info(f"output_tokens_details={details}")
        if reasoning_tokens is not None:
            omni.log.info(f"reasoning_tokens={reasoning_tokens}")


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        return getattr(message, "content", "") or ""

    return ""


def _get_finish_reason(response: Any) -> Optional[str]:
    """
    レスポンスから finish_reason または status を取得する。

    Responses API は 'status' フィールド (completed/incomplete) を使用し、
    Chat Completions API は 'choices[0].finish_reason' (stop/length) を使用する。
    """
    # Responses API: status フィールドをチェック
    status = getattr(response, "status", None)
    if status:
        return status

    # Chat Completions API: choices[0].finish_reason をチェック
    choices = getattr(response, "choices", None)
    if choices and len(choices) > 0:
        return getattr(choices[0], "finish_reason", None)

    return None


def _is_response_truncated(finish_reason: Optional[str]) -> bool:
    """
    finish_reason/status がトランケーションを示すかどうかを判定する。

    - 'length': Chat Completions API でトークン制限に達した
    - 'incomplete': Responses API で出力が不完全
    """
    if finish_reason is None:
        return False
    return finish_reason in ("length", "incomplete")


async def _stream_response_with_logging(stream, step_name: str) -> Tuple[str, Any]:
    """
    ストリーミングレスポンスを処理し、進捗をリアルタイムでログ出力する。

    Args:
        stream: OpenAI streaming response object
        step_name: ログに表示するステップ名 (例: "Step 1", "Step 2")

    Returns:
        (output_text, final_response): 生成されたテキストと最終レスポンスオブジェクト
    """
    output_text = ""
    reasoning_summary = ""
    final_response = None
    last_log_len = 0
    log_interval = 100  # 100文字ごとにログ出力

    omni.log.info(f"[{step_name}] 🚀 Streaming started...")

    async for event in stream:
        event_type = getattr(event, "type", None)

        # 全てのイベントに対して、もし usage が含まれていれば取得を試みる
        # (ストリーミングAPIでは時折、途中のイベントや複数のイベントに usage が分散・上書きされることがあるため)
        if hasattr(event, "response") and event.response:
            event_usage = getattr(event.response, "usage", None)
            if event_usage:
                if final_response is None:
                    final_response = event.response
                else:
                    # usage が更新されている場合は反映
                    final_response.usage = event_usage

        if event_type == "response.output_text.delta":
            # テキスト生成の進捗
            delta = getattr(event, "delta", "")
            output_text += delta
            # 一定間隔でログ出力
            if len(output_text) - last_log_len >= log_interval:
                omni.log.info(f"[{step_name}] 📝 Generating... ({len(output_text):,} chars)")
                last_log_len = len(output_text)

        elif event_type == "response.reasoning_summary_text.delta":
            # 推論サマリ（reasoningモデル用）
            delta = getattr(event, "delta", "")
            reasoning_summary += delta
            # 推論サマリはリアルタイムで表示（最初の100文字）
            if len(reasoning_summary) <= 100:
                omni.log.info(f"[{step_name}] 🧠 Reasoning: {delta}")

        elif event_type == "response.function_call_arguments.delta":
            # ツール/関数呼び出しの進行
            omni.log.info(f"[{step_name}] 🔧 Tool call arguments streaming...")

        elif event_type == "response.output_item.added":
            # 新しい出力項目が追加された
            item = getattr(event, "item", None)
            item_type = getattr(item, "type", None) if item else None
            if item_type:
                omni.log.info(f"[{step_name}] 📦 Output item added: {item_type}")

        elif event_type == "response.in_progress":
            omni.log.info(f"[{step_name}] ⏳ Response in progress...")

        elif event_type in ("response.completed", "response.done"):
            # 生成完了 - response オブジェクトを取得
            event_resp = getattr(event, "response", None)
            if event_resp:
                final_response = event_resp
            
            omni.log.info(f"[{step_name}] ✅ Generation complete ({len(output_text):,} chars)")

            # usage情報をログ出力（デバッグ用）
            if final_response:
                usage = getattr(final_response, "usage", None)
                if usage:
                    omni.log.info(f"[{step_name}] 📊 Usage data captured: {usage}")
                else:
                    omni.log.warn(f"[{step_name}] ⚠️ usage data is still None at completion")
            else:
                omni.log.warn(f"[{step_name}] ⚠️ final_response is still None at completion")

            # 最終的な推論サマリがある場合は表示
            if reasoning_summary:
                summary_preview = reasoning_summary[:200] + "..." if len(reasoning_summary) > 200 else reasoning_summary
                omni.log.info(f"[{step_name}] 🧠 Reasoning summary: {summary_preview}")

    # 万が一 final_response が None の場合、usage ゼロのモックを作成して返す
    if final_response is None:
        omni.log.warn(f"[{step_name}] ⚠️ Creating mock response as final_response was None")
        class MockResponse:
            def __init__(self):
                self.usage = None
                self.model = "unknown"
        final_response = MockResponse()

    return output_text, final_response


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _build_responses_input(
    prompt_text: str,
    image_base64: str,
    image_detail: Optional[str] = None,
) -> List[Dict[str, Any]]:
    detail = image_detail or DEFAULT_IMAGE_DETAIL
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_text},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image_base64}",
                    "detail": detail,
                },
            ],
        }
    ]


async def step1_analyze_image(
    image_base64: str,
    prompt1_text: str,
    dimensions_text: str,
    model_name: str,
    api_key: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    text_verbosity: Optional[str] = None,
    image_detail: Optional[str] = None,
    max_retries: Optional[int] = None,
    retry_delay_sec: Optional[float] = None,
    non_interactive: bool = True,  # Extension ???? True
) -> Tuple[str, Dict[str, Any]]:
    """
    ????1: ?????????????????

    Args:
        image_base64: Base64?????????????
        prompt1_text: ????????????
        dimensions_text: ????????
        model_name: ????OpenAI????
        api_key: OpenAI API???None?????????????
        non_interactive: ??????????Extension ???? True?

    Returns:
        (analysis_text, stats): ?????????????
    """
    # Lazy import
    from openai import AsyncOpenAI

    try:
        client = AsyncOpenAI(api_key=api_key, timeout=DEFAULT_API_TIMEOUT) if api_key else AsyncOpenAI(timeout=DEFAULT_API_TIMEOUT)
    except Exception as e:
        omni.log.error(f"OpenAI APIクライアントの初期化に失敗: {e}")
        raise RuntimeError("OpenAI clientの初期化に失敗。APIキーを確認してください")

    omni.log.info(f"--- ????1: ??????????? (???: {model_name}) ---")

    # ????????
    total_time = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    initial_prompt_text = f"{prompt1_text}\n\n--- ????????? ---\n{dimensions_text}"

    effective_reasoning_effort = reasoning_effort or DEFAULT_REASONING_EFFORT
    effective_max_output_tokens = (
        max_output_tokens if isinstance(max_output_tokens, int) and max_output_tokens > 0 else DEFAULT_MAX_OUTPUT_TOKENS
    )
    effective_text_verbosity = text_verbosity or DEFAULT_TEXT_VERBOSITY
    effective_image_detail = image_detail or DEFAULT_IMAGE_DETAIL

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": initial_prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": effective_image_detail}},
            ],
        }
    ]
    responses_input = _build_responses_input(initial_prompt_text, image_base64, effective_image_detail)

    retry_limit = int(max_retries) if isinstance(max_retries, (int, float)) else 0
    if retry_limit < 0:
        retry_limit = 0
    retry_delay = float(retry_delay_sec) if isinstance(retry_delay_sec, (int, float)) else 0.0
    if retry_delay < 0:
        retry_delay = 0.0
    attempts = retry_limit + 1

    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            start_time = time.time()
            response = None
            response_text = ""
            used_api = "responses"

            if hasattr(client, "responses"):
                try:
                    # ストリーミングモードでAPIを呼び出し、リアルタイムでログ出力
                    stream = await client.responses.create(
                        model=model_name,
                        input=responses_input,
                        reasoning={"effort": effective_reasoning_effort, "summary": "auto"},
                        text={"verbosity": effective_text_verbosity},
                        max_output_tokens=effective_max_output_tokens,
                        stream=True,
                    )
                    response_text, response = await _stream_response_with_logging(stream, "Step 1")

                    # Responses API が成功したが output_text が空の場合をチェック
                    if response and not response_text:
                        finish_reason = _get_finish_reason(response)
                        if _is_response_truncated(finish_reason):
                            reasoning_tokens = _get_reasoning_tokens(getattr(response, "usage", None))
                            omni.log.error(
                                f"Responses API returned empty output due to truncation. "
                                f"finish_reason={finish_reason}, reasoning_tokens={reasoning_tokens}. "
                                f"Increase max_output_tokens (current: {effective_max_output_tokens})."
                            )
                            raise RuntimeError(
                                f"Reasoning exhausted token limit before generating output. "
                                f"finish_reason={finish_reason}. Increase max_output_tokens."
                            )
                except RuntimeError:
                    raise
                except Exception as e:
                    omni.log.warn(f"Responses API failed, falling back to chat.completions: {e}")
                    response = None
                    response_text = ""

            if not response_text:
                used_api = "chat.completions"
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    reasoning_effort=effective_reasoning_effort,
                )
                response_text = _extract_response_text(response)

            # finish_reason をログに記録
            finish_reason = _get_finish_reason(response)
            omni.log.info(f"finish_reason={finish_reason}")
            if _is_response_truncated(finish_reason):
                omni.log.warn(
                    f"Response was truncated (finish_reason={finish_reason}). "
                    f"Output may be incomplete. Consider increasing max_output_tokens."
                )

            end_time = time.time()
            _log_response_details(response)
            omni.log.info(f"OpenAI API used: {used_api}")

            # ????????? 
            if not response_text:
                raise RuntimeError("API?????????????????????")

            processing_time = end_time - start_time
            total_time += processing_time
            omni.log.info(f"--- ????1: ????: {processing_time:.2f}? ---")

            usage = getattr(response, "usage", None)
            if usage is not None:
                tokens = _get_usage_tokens(usage)
                total_prompt_tokens += tokens["input_tokens"]
                total_completion_tokens += tokens["output_tokens"]
                total_tokens += tokens["total_tokens"]
                if tokens["source"] == "prompt/completion":
                    omni.log.info(
                        f"--- ?????: ?????={tokens['input_tokens']}, ??={tokens['output_tokens']}, ??={tokens['total_tokens']} ---"
                    )
                else:
                    omni.log.info(
                        f"--- ?????: ??={tokens['input_tokens']}, ??={tokens['output_tokens']}, ??={tokens['total_tokens']} ---"
                    )

            analysis_text = response_text
            omni.log.info(f"??????\n{analysis_text}")

            stats = {
                "time": total_time,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
            }
            return analysis_text, stats

        except asyncio.CancelledError:
            raise
        except Exception as e:
            last_error = e
            omni.log.warn(f"Step 1 failed (attempt {attempt}/{attempts}): {e}")
            if attempt < attempts and retry_delay > 0:
                await asyncio.sleep(retry_delay)
            continue

    omni.log.error(f"API????????????????: {last_error}")
    raise last_error if last_error else RuntimeError("Step 1 failed without a captured exception.")


async def step2_generate_json(
    analysis_text: str,
    dimensions_text: str,
    image_base64: str,
    prompt2_base_text: str,
    model_name: str,
    api_key: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    text_verbosity: Optional[str] = None,
    image_detail: Optional[str] = None,
    max_retries: Optional[int] = None,
    retry_delay_sec: Optional[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    ????2: JSON???????????????

    Args:
        analysis_text: ????1????????????
        dimensions_text: ????????
        image_base64: Base64?????????????
        prompt2_base_text: JSON????????????
        model_name: ????OpenAI????
        api_key: OpenAI API???None?????????????

    Returns:
        (layout_json, stats): ??????????JSON?????
    """
    # Lazy import
    from openai import AsyncOpenAI

    try:
        client = AsyncOpenAI(api_key=api_key, timeout=DEFAULT_API_TIMEOUT) if api_key else AsyncOpenAI(timeout=DEFAULT_API_TIMEOUT)
    except Exception as e:
        omni.log.error(f"OpenAI APIクライアントの初期化に失敗: {e}")
        raise RuntimeError("OpenAI clientの初期化に失敗。APIキーを確認してください")

    omni.log.info(f"--- ????2: JSON????????? (???: {model_name}) ---")

    final_prompt = f"""
{prompt2_base_text}

--- Additional Context from Previous Step ---
The following is a detailed analysis of the furniture layout from the image. Use this as a guide for placement and orientation.
{analysis_text}

--- Dimension Data for This Request ---
{dimensions_text}
"""

    effective_reasoning_effort = reasoning_effort or DEFAULT_REASONING_EFFORT
    effective_max_output_tokens = (
        max_output_tokens if isinstance(max_output_tokens, int) and max_output_tokens > 0 else DEFAULT_MAX_OUTPUT_TOKENS
    )
    effective_text_verbosity = text_verbosity or DEFAULT_TEXT_VERBOSITY
    effective_image_detail = image_detail or DEFAULT_IMAGE_DETAIL

    responses_input = _build_responses_input(final_prompt, image_base64, effective_image_detail)

    retry_limit = int(max_retries) if isinstance(max_retries, (int, float)) else 0
    if retry_limit < 0:
        retry_limit = 0
    retry_delay = float(retry_delay_sec) if isinstance(retry_delay_sec, (int, float)) else 0.0
    if retry_delay < 0:
        retry_delay = 0.0
    attempts = retry_limit + 1

    total_time = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            start_time = time.time()
            response = None
            response_text = ""
            used_api = "responses"

            if hasattr(client, "responses"):
                try:
                    # ストリーミングモードでAPIを呼び出し、リアルタイムでログ出力
                    stream = await client.responses.create(
                        model=model_name,
                        input=responses_input,
                        reasoning={"effort": effective_reasoning_effort, "summary": "auto"},
                        text={"verbosity": effective_text_verbosity},
                        max_output_tokens=effective_max_output_tokens,
                        stream=True,
                    )
                    response_text, response = await _stream_response_with_logging(stream, "Step 2")

                    # Responses API が成功したが output_text が空の場合をチェック
                    if response and not response_text:
                        finish_reason = _get_finish_reason(response)
                        if _is_response_truncated(finish_reason):
                            # トランケーションを検出: 二重課金を防ぐため、フォールバックせずエラー
                            reasoning_tokens = _get_reasoning_tokens(getattr(response, "usage", None))
                            omni.log.error(
                                f"Responses API returned empty output due to truncation. "
                                f"finish_reason={finish_reason}, reasoning_tokens={reasoning_tokens}. "
                                f"Increase max_output_tokens (current: {effective_max_output_tokens})."
                            )
                            raise RuntimeError(
                                f"Reasoning exhausted token limit before generating output. "
                                f"finish_reason={finish_reason}. Increase max_output_tokens."
                            )
                except RuntimeError:
                    # トランケーションエラーは再スローする（フォールバックしない）
                    raise
                except Exception as e:
                    omni.log.warn(f"Responses API failed, falling back to chat.completions: {e}")
                    response = None
                    response_text = ""

            if not response_text:
                used_api = "chat.completions"
                response = await client.chat.completions.create(
                    model=model_name,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": final_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}",
                                        "detail": effective_image_detail,
                                    },
                                },
                            ],
                        }
                    ],
                    reasoning_effort=effective_reasoning_effort,
                )
                response_text = _extract_response_text(response)

            # finish_reason をログに記録
            finish_reason = _get_finish_reason(response)
            omni.log.info(f"finish_reason={finish_reason}")
            if _is_response_truncated(finish_reason):
                omni.log.warn(
                    f"Response was truncated (finish_reason={finish_reason}). "
                    f"Output may be incomplete. Consider increasing max_output_tokens."
                )

            end_time = time.time()
            _log_response_details(response)
            omni.log.info(f"OpenAI API used: {used_api}")

            processing_time = end_time - start_time
            total_time += processing_time
            omni.log.info(f"--- ????2: ????: {processing_time:.2f}? ---")

            usage = getattr(response, "usage", None)
            if usage is not None:
                tokens = _get_usage_tokens(usage)
                total_prompt_tokens += tokens["input_tokens"]
                total_completion_tokens += tokens["output_tokens"]
                total_tokens += tokens["total_tokens"]
                if tokens["source"] == "prompt/completion":
                    omni.log.info(
                        f"--- ?????: ?????={tokens['input_tokens']}, ??={tokens['output_tokens']}, ??={tokens['total_tokens']} ---"
                    )
                else:
                    omni.log.info(
                        f"--- ?????: ??={tokens['input_tokens']}, ??={tokens['output_tokens']}, ??={tokens['total_tokens']} ---"
                    )

            if not response_text:
                raise RuntimeError("API?????JSON??????????????")

            json_output_str = response_text
            omni.log.info("--- JSON?????????? ---")

            stats = {
                "time": total_time,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
            }

            return _extract_json_from_text(json_output_str), stats

        except asyncio.CancelledError:
            raise
        except Exception as e:
            last_error = e
            omni.log.warn(f"Step 2 failed (attempt {attempt}/{attempts}): {e}")
            if attempt < attempts and retry_delay > 0:
                await asyncio.sleep(retry_delay)
            continue

    omni.log.error(f"API????????????????: {last_error}")
    raise last_error if last_error else RuntimeError("Step 2 failed without a captured exception.")

def step3_check_collisions(layout: Dict[str, Any]) -> List[Tuple[str, str, float]]:
    """
    ステップ3: オブジェクト間の衝突検出を開始します

    Axis-Aligned Bounding Box (AABB) の衝突判定ロジックを使用して
    オブジェクト間の衝突を検出し、重複体積を計算する。

    Args:
        layout: レイアウトJSON

    Returns:
        colliding_pairs: 衝突ペアのリスト [(name1, name2, overlap_volume), ...]
    """
    omni.log.info("--- ステップ3: オブジェクト間の衝突検出を開始します ---")

    objects = layout.get("area_objects_list", [])
    colliding_pairs = []

    # 各オブジェクトのバウンディングボックスを計算しておく
    object_bounds = []
    for obj in objects:
        name = obj.get("object_name", "Unnamed")
        # "Floor" は衝突対象から除外
        if name.lower() == "floor":
            continue
        try:
            cx = float(obj.get("X", 0.0))
            cz = float(obj.get("Z", 0.0))
            length = float(obj.get("Length", 1.0))  # x-axis
            height = float(obj.get("Height", 1.0))  # y-axis
            width = float(obj.get("Width", 1.0))    # z-axis

            rotation_z = float(obj.get("rotationZ", 0.0))
            size_mode = obj.get("size_mode") or layout.get("size_mode") or "world"
            if isinstance(size_mode, str):
                size_mode = size_mode.lower()
            else:
                size_mode = "world"
            if size_mode not in ("world", "local"):
                size_mode = "world"
            if size_mode == "local":
                rot = int(round(rotation_z)) % 360
                if rot in (90, 270):
                    length, width = width, length

            # 中心座標から最小・最大座標を計算
            # Y座標は床面(0)から高さまで
            min_x, max_x = cx - length / 2, cx + length / 2
            min_y, max_y = 0, height
            min_z, max_z = cz - width / 2, cz + width / 2

            object_bounds.append({
                "name": name,
                "min_x": min_x, "max_x": max_x,
                "min_y": min_y, "max_y": max_y,
                "min_z": min_z, "max_z": max_z,
            })
        except (ValueError, TypeError) as e:
            omni.log.warn(f"オブジェクト '{name}' の寸法データが無効です。スキップします。エラー: {e}")
            continue

    # すべてのオブジェクトのペアを比較
    for i in range(len(object_bounds)):
        for j in range(i + 1, len(object_bounds)):
            obj1 = object_bounds[i]
            obj2 = object_bounds[j]

            # X, Y, Z軸それぞれで重なりをチェック
            overlap_x = (obj1["min_x"] < obj2["max_x"]) and (obj1["max_x"] > obj2["min_x"])
            overlap_y = (obj1["min_y"] < obj2["max_y"]) and (obj1["max_y"] > obj2["min_y"])
            overlap_z = (obj1["min_z"] < obj2["max_z"]) and (obj1["max_z"] > obj2["min_z"])

            # すべての軸で重なっていれば衝突している
            if overlap_x and overlap_y and overlap_z:
                # 重複部分の体積を計算
                overlap_length = max(0, min(obj1["max_x"], obj2["max_x"]) - max(obj1["min_x"], obj2["min_x"]))
                overlap_height = max(0, min(obj1["max_y"], obj2["max_y"]) - max(obj1["min_y"], obj2["min_y"]))
                overlap_width = max(0, min(obj1["max_z"], obj2["max_z"]) - max(obj1["min_z"], obj2["min_z"]))

                overlap_volume = overlap_length * overlap_height * overlap_width

                if overlap_volume > 0:
                    colliding_pairs.append((obj1["name"], obj2["name"], overlap_volume))

    if not colliding_pairs:
        omni.log.info("--- 衝突は検出されませんでした。 ---")
    else:
        omni.log.warn("--- 衝突が検出されました ---")
        total_overlap_volume = 0
        for name1, name2, volume in colliding_pairs:
            # 単位がcmなので、立法センチメートルで表示
            omni.log.warn(f"  - '{name1}' と '{name2}' が衝突しています。(重複体積: {volume:,.2f} cm³)")
            total_overlap_volume += volume
        omni.log.warn(f"合計重複体積: {total_overlap_volume:,.2f} cm³")

    return colliding_pairs
