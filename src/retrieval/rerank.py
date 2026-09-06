"""重排 API 封装（baseline-4 用）：SiliconFlow BAAI/bge-reranker-v2-m3。

rerank(query, documents) -> scores（与 documents 对齐），调用方自行排序。
用法: from retrieval.rerank import rerank
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
from config import RERANK  # noqa: E402

_TIMEOUT = 120
_RETRY = 2


def rerank(query: str, documents: list[str]) -> list[float]:
    """返回与 documents 等长的相关性分数（越高越相关）。"""
    if not documents:
        return []
    url = f"{RERANK.base}/rerank"
    payload = {"model": RERANK.model, "query": query, "documents": documents}
    body = json.dumps(payload).encode("utf-8")
    last_err: Exception | None = None
    for _ in range(_RETRY + 1):
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {RERANK.key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            n = len(documents)
            scores = [0.0] * n
            for it in data.get("results", []):
                idx, sc = it["index"], it.get("relevance_score", 0.0)
                if 0 <= idx < n:
                    scores[idx] = sc
            return scores
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5)
    raise RuntimeError(f"rerank API 失败: {last_err}")
