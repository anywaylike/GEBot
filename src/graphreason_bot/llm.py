"""
graphreason_bot/llm.py
===============
大语言模型调用封装（API key 从环境变量读取）
支持 GPT-4o、DeepSeek-V4-Flash、GLM-4.7
"""
import inspect
import re
import time
from typing import Tuple, Optional
from graphreason_bot.config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, GPT_MODEL,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    DEEPSEEK_REASONING_EFFORT, DEEPSEEK_THINKING_ENABLED,
    ZHIPUAI_API_KEY, ZHIPUAI_MODEL,
    DEFAULT_TEMPERATURE, DEFAULT_MAX_RETRIES, DEFAULT_RETRY_SLEEP,
    LLM_TIMEOUT,
)


class LLMNonRetryableError(RuntimeError):
    """API error that requires user action rather than automatic retry."""


def _is_non_retryable_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    message = str(error).lower()
    return (
        status_code == 402
        or "insufficient balance" in message
        or "insufficient_balance" in message
    )


def _call_deepseek(client, model: str, prompt: str):
    """按官方 V4-Pro 参数调用，并兼容较旧的 OpenAI Python SDK。"""
    extra_body = {}
    if DEEPSEEK_THINKING_ENABLED:
        extra_body["thinking"] = {"type": "enabled"}

    request_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    # 新版 SDK 支持顶层 reasoning_effort；旧版 SDK 用 extra_body
    # 合并到相同的 HTTP JSON 请求体。
    parameters = inspect.signature(
        client.chat.completions.create
    ).parameters
    if "reasoning_effort" in parameters:
        request_kwargs["reasoning_effort"] = DEEPSEEK_REASONING_EFFORT
    else:
        extra_body["reasoning_effort"] = DEEPSEEK_REASONING_EFFORT
    if extra_body:
        request_kwargs["extra_body"] = extra_body

    return client.chat.completions.create(**request_kwargs)


def _get_client(model_name: str):
    """按模型名返回 OpenAI-compatible client"""
    from openai import OpenAI

    if model_name == "GPT":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set")
        return OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            timeout=LLM_TIMEOUT,
        ), GPT_MODEL
    elif model_name == "DeepSeek":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY or JENIYA_API_KEY is not set")
        return OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            timeout=LLM_TIMEOUT,
        ), DEEPSEEK_MODEL
    elif model_name == "ZhipuAI":
        if not ZHIPUAI_API_KEY:
            raise ValueError("ZHIPUAI_API_KEY is not set")
        from zhipuai import ZhipuAI as ZhipuAIClient
        client = ZhipuAIClient(api_key=ZHIPUAI_API_KEY)
        return client, ZHIPUAI_MODEL
    else:
        # Any other model name is treated as an OpenAI-compatible model id
        # served by OPENAI_BASE_URL, e.g. "gpt-5.4-mini".
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY or JENIYA_API_KEY is not set")
        return OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            timeout=LLM_TIMEOUT,
        ), model_name


def _call_llm(prompt: str, model_name: str,
              temperature: float = DEFAULT_TEMPERATURE,
              max_retries: int = DEFAULT_MAX_RETRIES,
              retry_sleep: float = DEFAULT_RETRY_SLEEP) -> Optional[str]:
    """调用 LLM，含重试机制"""
    client, model = _get_client(model_name)

    for attempt in range(1, max_retries + 1):
        try:
            if model_name == "ZhipuAI":
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    stream=False,
                )

            if response is None or not hasattr(response, "choices") or not response.choices:
                print(f"[LLM] Empty response (attempt {attempt}/{max_retries})")
                if attempt < max_retries:
                    time.sleep(retry_sleep)
                continue

            answer = response.choices[0].message.content
            if answer and answer.strip():
                return answer.strip()

        except Exception as e:
            print(f"[LLM] Error ({model_name}, attempt {attempt}/{max_retries}): {e}")
            if _is_non_retryable_error(e):
                raise LLMNonRetryableError(
                    f"{model_name} API rejected the request because the "
                    "account balance is insufficient. Recharge the account "
                    "before resuming."
                ) from e
            if attempt < max_retries:
                time.sleep(retry_sleep)

    return None


def extract_label_and_reason(response_text: str) -> Tuple[str, str]:
    """
    从 LLM 响应中提取 label 和 reasoning。
    支持多种输出格式。
    """
    response_text = response_text.strip()
    lines = response_text.splitlines()
    label = "unknown"
    reasoning = ""

    label_mapping = {
        "a": "human", "human": "human",
        "b": "bot", "bot": "bot",
    }

    def clean_reasoning(lines):
        return [line for line in lines
                if line.strip().lower() not in {"reasoning:", "reasoning"}
                and line.strip()]

    for i, line in enumerate(lines):
        lowered = line.lower().strip().replace("**", "").replace("`", "")

        # Label: bot / Label: human
        match_label = re.match(r'^label\s*[:\-]?\s*(a|b|human|bot)', lowered)
        if match_label:
            val = match_label.group(1)
            label = label_mapping.get(val, "unknown")
            reasoning = "\n".join(clean_reasoning(lines[i + 1:]))
            return label, reasoning

        # A: human / B: bot
        match_ab = re.match(r'^(a|b)\s*[:\-]?\s*(human|bot)', lowered)
        if match_ab:
            tag = match_ab.group(1)
            meaning = match_ab.group(2)
            label = label_mapping.get(meaning or tag, "unknown")
            reasoning = "\n".join(clean_reasoning(lines[i + 1:]))
            return label, reasoning

        # Predicted Label: human
        match_pred = re.match(r'^predicted\s+label\s*[:\-]?\s*(human|bot)', lowered)
        if match_pred:
            label = label_mapping.get(match_pred.group(1), "unknown")
            reasoning = "\n".join(clean_reasoning(lines[i + 1:]))
            return label, reasoning

    # Fallback: keyword search
    for line in lines:
        lowered = line.lower()
        if "bot" in lowered and "human" not in lowered:
            return "bot", "\n".join(clean_reasoning(lines))
        if "human" in lowered and "bot" not in lowered:
            return "human", "\n".join(clean_reasoning(lines))

    return "unknown", "\n".join(clean_reasoning(lines))


def detect(prompt: str, model_name: str) -> Tuple[str, str]:
    """
    完整的检测流程：调用 LLM + 解析结果。

    Returns:
        (reasoning, label)
    """
    response = _call_llm(prompt, model_name)
    if response is None:
        return "", "unknown"
    label, reasoning = extract_label_and_reason(response)
    return reasoning, label
