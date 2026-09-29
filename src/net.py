"""HTTP JSON POST 统一封装：指数退避 + 抖动重试。

背景：LLM / Embedding / Rerank 三处 API 封装原先各写一份 `_RETRY=2` +
固定 `time.sleep(1.5)`。固定间隔在 429 频控下几乎必然连续失败，且不区分
「可重试」（429/5xx/超时）与「重试无意义」（401/400 参数错）。

用法:
    from net import post_json
    data = post_json(url, payload, api_key, label="Embedding")
"""

from __future__ import annotations

import json
import random
import time
import urllib.request

DEFAULT_TIMEOUT = 120
DEFAULT_RETRIES = 4          # 总尝试次数 = 1 + retries
_BASE_DELAY = 1.0            # 首次退避秒数
_MAX_DELAY = 30.0

# 4xx 中值得重试的：408 请求超时、429 限流；其余 4xx 直接抛
_RETRYABLE_4XX = {408, 429}


def _retry_after(e: Exception) -> float | None:
    """429/503 若带 Retry-After 头则优先采用服务端建议的等待时间。"""
    hdr = getattr(e, "headers", None)
    if not hdr:
        return None
    v = hdr.get("Retry-After")
    if not v:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def post_json(url: str, payload: dict, api_key: str,
              timeout: float = DEFAULT_TIMEOUT,
              retries: int = DEFAULT_RETRIES,
              label: str = "API", quiet: bool = False) -> dict:
    """POST JSON 并返回解析后的 dict；失败按指数退避 + 抖动重试。

    - 可重试：网络异常、超时、408/429、5xx
    - 不重试：400/401/403/404 等参数或鉴权错误（重试只会浪费配额）
    """
    body = json.dumps(payload).encode("utf-8")
    last_err: Exception | None = None

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last_err = e
            code = getattr(e, "code", None)
            if code is not None and 400 <= code < 500 and code not in _RETRYABLE_4XX:
                raise
            if attempt >= retries:
                break
            wait = _retry_after(e)
            if wait is None:
                wait = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
                wait *= 0.5 + random.random()   # 抖动，避免多进程同步重试
            if not quiet:
                tag = f" {code}" if code else ""
                print(f"  [{label}] 第 {attempt + 1} 次失败"
                      f"（{type(e).__name__}{tag}），{wait:.1f}s 后重试 …",
                      flush=True)
            time.sleep(wait)

    raise RuntimeError(f"{label} 失败（已重试 {retries} 次）: {last_err}")
