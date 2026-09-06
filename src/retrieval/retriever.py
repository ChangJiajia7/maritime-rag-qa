"""在线检索组件（serve 侧）——生成问答链路用，算法与离线评测同源。

用法:
    from retrieval.retriever import Retriever
    r = Retriever(method="hybrid", alpha=0.7, rerank=False)   # 惰性加载索引
    hits = r.retrieve("客船能否同时载运危险货物", k=5)
    # hits: list[(score, Chunk)]，chunk.source_loc 直接可用于引用

method: bm25 | dense | hybrid（dense/hybrid 首次调用会加载/构建嵌入缓存）
rerank: True 时候选扩至 20 并交 bge-reranker-v2-m3 重排取 k
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from retrieval.bm25 import BM25Index, load_corpus  # noqa: E402


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class Retriever:
    def __init__(self, method: str = "hybrid", alpha: float = 0.7,
                 rerank: bool = False):
        if method not in ("bm25", "dense", "hybrid"):
            raise ValueError(method)
        self.method, self.alpha, self.rerank = method, alpha, rerank
        self.chunks = load_corpus()
        self._bm = None
        self._den = None

    # —— 惰性加载 ——
    def _bm25(self) -> BM25Index:
        if self._bm is None:
            self._bm = BM25Index(self.chunks)
        return self._bm

    def _dense(self):
        if self._den is None:
            from retrieval.dense import DenseIndex, load_or_build
            self._den = DenseIndex(self.chunks, load_or_build(self.chunks, quiet=True))
        return self._den

    # —— 检索 ——
    def _scores(self, query: str, q_vec=None) -> np.ndarray:
        if self.method == "bm25":
            return self._bm25().score_all(query)
        if self.method == "dense":
            return self._dense().sims_all(q_vec)
        return self.alpha * _minmax(self._dense().sims_all(q_vec)) \
            + (1 - self.alpha) * _minmax(self._bm25().score_all(query))

    def retrieve(self, query: str, k: int = 5) -> list[tuple[float, object]]:
        q_vec = None
        if self.method != "bm25":
            from retrieval.dense import embed_texts
            q_vec = embed_texts([query], quiet=True)[0]
        scores = self._scores(query, q_vec)

        cand_k = 20 if self.rerank else k
        order = np.argsort(-scores)[:cand_k]
        cands = [(float(scores[i]), self.chunks[i]) for i in order.tolist()]

        if self.rerank and cands:
            from retrieval.rerank import rerank as rerank_api
            sc = rerank_api(query, [c[1].text for c in cands])
            cands = sorted(zip(sc, [c[1] for c in cands]), key=lambda x: -x[0])[:k]

        return [(float(s), c) for s, c in cands]
