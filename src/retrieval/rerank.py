"""重排 API 封装（baseline-4 用）：SiliconFlow BAAI/bge-reranker-v2-m3。

rerank(query, documents) -> scores（与 documents 对齐），调用方自行排序。
- 重试: 统一走 net.post_json（指数退避 + 抖动）
- 缓存: 以 sha1(query + documents) 为键落盘 data/cache/rerank_cache.json，
        同一 (问题, 候选集) 重复评测零 API 调用

用法: from retrieval.rerank import rerank
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
from config import RERANK  # noqa: E402
from net import post_json  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = _ROOT / "data" / "cache"
CACHE_FILE = CACHE_DIR / "rerank_cache.json"

_cache: dict[str, list[float]] | None = None


def _key(query: str, documents: list[str]) -> str:
    h = hashlib.sha1()
    h.update(query.encode("utf-8"))
    for d in documents:
        h.update(b"\x00")
        h.update(d.encode("utf-8"))
    return h.hexdigest()


def _load_cache() -> dict[str, list[float]]:
    global _cache
    if _cache is None:
        if CACHE_FILE.exists():
            try:
                _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 —— 缓存损坏则丢弃重建
                _cache = {}
        else:
            _cache = {}
    return _cache


def _save_cache(cache: dict[str, list[float]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")


def rerank(query: str, documents: list[str], quiet: bool = True) -> list[float]:
    """返回与 documents 等长的相关性分数（越高越相关）。"""
    if not documents:
        return []

    cache = _load_cache()
    ck = _key(query, documents)
    if ck in cache and len(cache[ck]) == len(documents):
        return cache[ck]

    data = post_json(f"{RERANK.base}/rerank",
                     {"model": RERANK.model, "query": query,
                      "documents": documents},
                     RERANK.key, label="Rerank", quiet=quiet)

    n = len(documents)
    scores = [0.0] * n
    for it in data.get("results", []):
        idx, sc = it["index"], it.get("relevance_score", 0.0)
        if 0 <= idx < n:
            scores[idx] = sc

    cache[ck] = scores
    _save_cache(cache)
    return scores
