"""检索基线统一评测：recall@1/3/5 + MRR@5。

用法（项目根）:
    python src/retrieval/eval_retrieval.py --method bm25                     # baseline-1
    python src/retrieval/eval_retrieval.py --method dense                    # baseline-2
    python src/retrieval/eval_retrieval.py --method hybrid [--alpha 0.5]     # baseline-3
    python src/retrieval/eval_retrieval.py --method hybrid --rerank          # baseline-4
    ... 任一加 --verbose 输出每问 top5 明细

融合: dense 与 bm25 分数各自 min-max 归一化后按 α 加权（α=向量权重）。
rerank: 候选扩到 top20 后交 bge-reranker-v2-m3 API 重排取 top5。

命中判定: golden = (source_doc, golden_articles)；top-k 出现同 doc 且
article_no ∈ golden 即命中。
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from retrieval.bm25 import BM25Index, load_corpus  # noqa: E402

EVAL_CSV = _ROOT / "eval" / "qa_eval_set.csv"


def parse_golden(s: str) -> set[int]:
    return {int(x) for x in s.split(",") if x.strip().isdigit()}


def _load_test_rows() -> list[dict]:
    with open(EVAL_CSV, encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r["split"] == "test"]


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def run(method: str = "dense", alpha: float = 0.5, rerank: bool = False,
        verbose: bool = False) -> int:
    t0 = time.time()
    chunks = load_corpus()
    rows = _load_test_rows()
    n = len(rows)
    tag = method + ("+rerank" if rerank else "")
    print(f"方法: {tag} (α={alpha}) | 索引 {len(chunks)} 条款块 | test {n} 条")

    # —— 构造各检索器的全库打分器 ——
    bm = BM25Index(chunks) if method in ("bm25", "hybrid") else None
    if method in ("dense", "hybrid"):
        from retrieval.dense import DenseIndex, embed_queries, load_or_build
        mat = load_or_build(chunks)
        den = DenseIndex(chunks, mat)
        print("批量嵌入 question（带持久化缓存）…")
        q_vecs = embed_queries([r["question"] for r in rows])
    else:
        den, q_vecs = None, None

    hits_at = {1: 0, 3: 0, 5: 0}
    rr_sum = 0.0
    miss: list[str] = []
    detail: list[tuple] = []

    def topk_hits(i: int, k: int) -> list:
        """按 method 取该问前 k 名命中块（rerank 前置候选用 k=20）。"""
        q = rows[i]["question"]
        if method == "bm25":
            scores = bm.score_all(q)
        elif method == "dense":
            scores = den.sims_all(q_vecs[i])
        else:  # hybrid
            scores = alpha * _minmax(den.sims_all(q_vecs[i])) \
                + (1 - alpha) * _minmax(bm.score_all(q))
        order = np.argsort(-scores)[:k]
        out = []
        for j in order.tolist():
            c = chunks[j]
            out.append((float(scores[j]), c))
        return out

    for i, r in enumerate(rows):
        qid = r["id"]
        doc, golden = r["source_doc"], parse_golden(r["golden_articles"])
        q = r["question"]

        cands = topk_hits(i, 20 if rerank else 5)
        if rerank and cands:
            from retrieval.rerank import rerank as rerank_api
            docs = [c[1].text for c in cands]
            sc = rerank_api(q, docs)
            cands = sorted(zip(sc, [c[1] for c in cands]), key=lambda x: -x[0])[:5]

        rank = None
        for p, (sc, c) in enumerate(cands, start=1):
            if c.doc == doc and c.article_no in golden:
                rank = p
                break
        for k in hits_at:
            if rank is not None and rank <= k:
                hits_at[k] += 1
        rr_sum += 1.0 / rank if rank else 0.0
        if rank is None:
            miss.append(qid)
        detail.append((qid, q[:26], doc, sorted(golden), rank,
                       [f"{c.doc[:6]}~{c.article_no}" for _, c in cands]))

    dt = time.time() - t0
    print(f"recall@1 = {hits_at[1]}/{n} = {hits_at[1]/n:.3f}")
    print(f"recall@3 = {hits_at[3]}/{n} = {hits_at[3]/n:.3f}")
    print(f"recall@5 = {hits_at[5]}/{n} = {hits_at[5]/n:.3f}")
    print(f"MRR@5   = {rr_sum/n:.3f}")
    print(f"top5 未命中: {len(miss)} 条: {miss} | 耗时 {dt:.1f}s")

    if verbose:
        print("\n--- 每问 top5 明细 ---")
        for qid, q, doc, golden, rank, top5 in detail:
            mark = "OK" if rank is not None else "XX"
            print(f"{qid} {mark} golden={doc[:6]}~{golden} rank={rank} | {q}")
            print(f"      top5: {top5}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="dense", choices=["bm25", "dense", "hybrid"])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    return run(method=args.method, alpha=args.alpha, rerank=args.rerank,
               verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
