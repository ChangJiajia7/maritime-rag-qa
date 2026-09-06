"""条款级 BM25 稀疏检索（baseline-1，零向量模型依赖）。

- 语料: data/law_*.jsonl 全部条款块（doc / article_no / chapter / text）
- 分词: jieba（词典加载在首次 import 时完成）
- 检索: Okapi BM25（k1=1.5, b=0.75），整段条款 text 为一个"文档"
- 用途: 与后续 向量(BGE-M3)/混合/rerank 链路对比的下限基线

用法:
    from retrieval.bm25 import load_corpus, BM25Index
    idx = BM25Index(load_corpus())
    hits = idx.search("客船能否同时载运危险货物", k=5)
    # hits = [Hit(doc=..., article_no=..., chapter=..., score=..., snippet=...), ...]
"""

from __future__ import annotations

import glob
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import jieba

_ROOT = Path(__file__).resolve().parents[2]  # 项目根
_K1, _B = 1.5, 0.75

# 纯噪声 token（法规文本里的框架词，不参与匹配）
_STOP = set("的 了 与 和 或 及 在 对 于 是 应当 依照 按照 违反 本条 前款 下列 规定 有关 其 他 该 项 目 款 至 自 起 中 上 下 内 外 由 被 向 从 为 以 者 等 之 而 并 不 得 有 无 任何 以及 按照 办法 情况 情形 活动 行为 人员 单位".split())


@dataclass
class Chunk:
    doc: str
    doc_version: str
    chapter: str
    article_no: int
    text: str

    @property
    def key(self) -> tuple[str, int]:
        return (self.doc, self.article_no)


@dataclass
class Hit:
    doc: str
    article_no: int
    chapter: str
    score: float
    snippet: str


def _norm_text(t: str) -> str:
    """只保留中文/数字/字母，规整空白（条号前缀本身参与索引，便于命中溯源）。"""
    t = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", " ", t)
    return re.sub(r"\s+", " ", t)


def tokenize(text: str) -> list[str]:
    toks = []
    for w in jieba.lcut(_norm_text(text)):
        w = w.strip().lower()
        if len(w) >= 2 and w not in _STOP and not w.isdigit():  # 纯数字保留? 见下行
            toks.append(w)
        elif w.isdigit() and len(w) <= 5:  # 短数字（60/1000/175000）是法条定位高信号
            toks.append(w)
    return toks


def load_corpus(raw_dir: str | None = None) -> list[Chunk]:
    """读入全部 data/law_*.jsonl，返回条款块列表（按文件内顺序）。"""
    if raw_dir is None:
        raw_dir = str(_ROOT / "data")
    chunks: list[Chunk] = []
    for fp in sorted(glob.glob(str(Path(raw_dir) / "law_*.jsonl"))):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                chunks.append(Chunk(
                    doc=r["doc"], doc_version=r.get("doc_version", ""),
                    chapter=r.get("chapter", ""), article_no=r["article_no"],
                    text=r["text"],
                ))
    return chunks


class BM25Index:
    """Okapi BM25。N≈593 条款块，直接全量打分即可。"""

    def __init__(self, chunks: list[Chunk], k1: float = _K1, b: float = _B):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.n = len(chunks)
        self.doc_tokens: list[list[str]] = [tokenize(c.text) for c in chunks]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avg_len = sum(self.doc_lens) / max(self.n, 1)
        # df / idf
        df: dict[str, int] = {}
        for toks in self.doc_tokens:
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log((self.n - d + 0.5) / (d + 0.5) + 1.0) for w, d in df.items()}

    def score_all(self, query: str) -> "np.ndarray":
        """全库 BM25 分（长度 N），供混合融合（hybrid）使用。"""
        import numpy as np
        q_toks = [w for w in tokenize(query) if w in self.idf]
        scores = np.zeros(self.n, dtype=np.float64)
        if not q_toks:
            return scores
        q_counts: dict[str, int] = {}
        for w in q_toks:
            q_counts[w] = q_counts.get(w, 0) + 1
        for i, toks in enumerate(self.doc_tokens):
            dl = self.doc_lens[i]
            if dl == 0:
                continue
            tf: dict[str, int] = {}
            for w in toks:
                tf[w] = tf.get(w, 0) + 1
            acc = 0.0
            for w, qf in q_counts.items():
                f = tf.get(w, 0)
                if not f:
                    continue
                acc += self.idf[w] * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
                )
            scores[i] = acc
        return scores

    def search(self, query: str, k: int = 5) -> list[Hit]:
        import numpy as np
        scores = self.score_all(query)
        ranked = np.argsort(-scores)
        out: list[Hit] = []
        for i in ranked.tolist():
            if scores[i] <= 0:
                break
            c = self.chunks[i]
            out.append(Hit(doc=c.doc, article_no=c.article_no, chapter=c.chapter,
                           score=float(scores[i]), snippet=c.text[:70].replace("\n", " ")))
            if len(out) >= k:
                break
        return out
