"""条款级稠密向量检索（baseline-2：纯向量，全 API 无本地模型）。

- 嵌入: SiliconFlow BAAI/bge-m3（config.EMBED，urllib POST /embeddings）
- 语料: data/law_*.jsonl 全部条款块（与 bm25.load_corpus 同序）
- 存储: 归一化向量落盘 data/cache/dense_emb.npy + .keys.json（可重建，不入 git）
- 检索: 余弦相似度（归一化后即点积）top-k

用法:
    from retrieval.dense import DenseIndex, load_or_build
    idx = load_or_build(chunks)          # 缓存命中则免 API
    hits = idx.search("客船能否同时载运危险货物", k=5)
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
from config import EMBED  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = _ROOT / "data" / "cache"
CACHE_NPY = CACHE_DIR / "dense_emb.npy"
CACHE_KEYS = CACHE_DIR / "dense_emb.keys.json"

_BATCH = 24          # 每请求最多文本数（硅基流动 bge-m3 兼容上限内）
_TIMEOUT = 120
_RETRY = 2


@dataclass
class DenseHit:
    doc: str
    article_no: int
    chapter: str
    score: float
    snippet: str


def _post_embeddings(texts: list[str]) -> list[list[float]]:
    """调用一次 /embeddings（输入非空校验由上层保证）。"""
    url = f"{EMBED.base}/embeddings"
    payload = {"model": EMBED.model, "input": texts, "encoding_format": "float"}
    body = json.dumps(payload).encode("utf-8")
    last_err: Exception | None = None
    for _ in range(_RETRY + 1):
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {EMBED.key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            items = sorted(data["data"], key=lambda d: d["index"])
            return [it["embedding"] for it in items]
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5)
    raise RuntimeError(f"embedding API 失败: {last_err}")


def embed_texts(texts: list[str], quiet: bool = False) -> np.ndarray:
    """批量嵌入（分批），返回 float32 矩阵，行已 L2 归一化。"""
    out: list[np.ndarray] = []
    for i in range(0, len(texts), _BATCH):
        batch = texts[i:i + _BATCH]
        if not quiet:
            print(f"  embed [{i+1}-{i+len(batch)}/{len(texts)}] ...", flush=True)
        vecs = _post_embeddings(batch)
        m = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        out.append(m / np.maximum(norms, 1e-9))
    return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)


def _chunk_keys(chunks) -> list[str]:
    return [f"{c.doc}|{c.article_no}" for c in chunks]


def load_or_build(chunks, force: bool = False, quiet: bool = False) -> np.ndarray:
    """条款块 → 归一化嵌入矩阵；缓存命中（keys 完全一致）则直接读盘。"""
    keys = _chunk_keys(chunks)
    if not force and CACHE_NPY.exists() and CACHE_KEYS.exists():
        cached = json.loads(CACHE_KEYS.read_text(encoding="utf-8"))
        if cached == keys:
            m = np.load(CACHE_NPY)
            if not quiet:
                print(f"嵌入缓存命中: {m.shape[0]} 条（{CACHE_NPY.name}）")
            return m
        if not quiet:
            print("语料有变，重建嵌入缓存 …")
    texts = [c.text for c in chunks]
    m = embed_texts(texts, quiet=quiet)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_NPY, m)
    CACHE_KEYS.write_text(json.dumps(keys, ensure_ascii=False), encoding="utf-8")
    if not quiet:
        print(f"嵌入完成并缓存: {m.shape[0]} 条 × {m.shape[1]} 维 → {CACHE_NPY.name}")
    return m


class DenseIndex:
    """BGE-M3 余弦检索（矩阵已归一化，sim = 点积）。"""

    def __init__(self, chunks, emb_matrix: np.ndarray, top_k: int = 5):
        self.chunks = chunks
        self.matrix = emb_matrix
        self.top_k = top_k

    def _embed_query(self, query: str) -> np.ndarray:
        vec = embed_texts([query], quiet=True)[0]
        return vec.reshape(1, -1)

    def search(self, query: str, k: int | None = None) -> list[DenseHit]:
        k = k or self.top_k
        q = self._embed_query(query)
        sims = self.matrix @ q.T  # (N,1)
        order = np.argsort(-sims[:, 0])[:k]
        out: list[DenseHit] = []
        for i in order:
            c = self.chunks[i]
            out.append(DenseHit(doc=c.doc, article_no=c.article_no, chapter=c.chapter,
                                score=float(sims[i, 0]), snippet=c.text[:70].replace("\n", " ")))
        return out

    def sims_all(self, q_vec: "np.ndarray") -> "np.ndarray":
        """全库余弦相似度（长度 N），供混合融合（hybrid）使用。"""
        return (self.matrix @ q_vec.reshape(1, -1).T)[:, 0]

    def search_by_vector(self, q_vec: np.ndarray, k: int | None = None) -> list[DenseHit]:
        """已算好的 query 向量直接检索（评测批量复用）。"""
        k = k or self.top_k
        sims = self.sims_all(q_vec)
        order = np.argsort(-sims)[:k]
        out: list[DenseHit] = []
        for i in order.tolist():
            c = self.chunks[i]
            out.append(DenseHit(doc=c.doc, article_no=c.article_no, chapter=c.chapter,
                                score=float(sims[i]), snippet=c.text[:70].replace("\n", " ")))
        return out


if __name__ == "__main__":
    # 快速自检：索引规模 + 一条示例查询
    from retrieval.bm25 import load_corpus
    chunks = load_corpus()
    mat = load_or_build(chunks)
    idx = DenseIndex(chunks, mat)
    for q in ("客船能否同时载运乘客和危险货物", "船员遣返费用由谁承担"):
        print(f"\nQ: {q}")
        for h in idx.search(q, k=3):
            print(f"  {h.score:.4f} {h.doc[:8]} 第{h.article_no}条 | {h.snippet[:40]}")
