"""BM25 baseline 离线评测（baseline-1）：recall@1/3/5 + MRR@5。

用法（项目根目录执行）:
    python src/retrieval/eval_bm25.py              # 汇总指标
    python src/retrieval/eval_bm25.py --verbose    # 附带每问 top-5 命中明细

命中判定: 一条 QA 的 golden = (source_doc, golden_articles 集合)；
检索 top-k 中出现同 doc 且 article_no ∈ golden 即视为命中。
golden_articles 解析: "8" -> {8}; "37,38" -> {37,38}。

输出: 汇总行（可直接贴入 experiments/log.md 的 A 表）。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))  # 允许从项目根直接运行

from retrieval.bm25 import BM25Index, load_corpus

EVAL_CSV = _ROOT / "eval" / "qa_eval_set.csv"


def parse_golden(s: str) -> set[int]:
    return {int(x) for x in s.split(",") if x.strip().isdigit()}


def main() -> int:
    verbose = "--verbose" in sys.argv
    idx = BM25Index(load_corpus())
    print(f"索引: {idx.n} 个条款块 | 词典 {len(idx.idf)} 词")

    rows = []
    with open(EVAL_CSV, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] != "test":
                continue
            rows.append(r)

    hits_at = {1: 0, 3: 0, 5: 0}
    rr_sum = 0.0
    miss_ids: list[str] = []
    detail: list[tuple] = []

    for r in rows:
        qid, q, doc, golden = r["id"], r["question"], r["source_doc"], parse_golden(r["golden_articles"])
        hits = idx.search(q, k=5)
        rank = None
        for i, h in enumerate(hits, start=1):
            if h.doc == doc and h.article_no in golden:
                rank = i
                break
        for k in hits_at:
            if rank is not None and rank <= k:
                hits_at[k] += 1
        rr = 1.0 / rank if rank is not None else 0.0
        rr_sum += rr
        if rank is None:
            miss_ids.append(qid)
        detail.append((qid, q[:24], doc, sorted(golden), rank, [f"{h.doc[:6]}~{h.article_no}" for h in hits]))

    n = len(rows)
    print(f"评估集 test: {n} 条")
    for k in (1, 3, 5):
        print(f"recall@{k} = {hits_at[k]}/{n} = {hits_at[k]/n:.3f}")
    print(f"MRR@5   = {rr_sum/n:.3f}")
    print(f"top5 未命中: {len(miss_ids)} 条: {miss_ids}")

    if verbose:
        print("\n--- 每问 top5 明细 ---")
        for qid, q, doc, golden, rank, top5 in detail:
            mark = "✓" if rank is not None else "✗"
            print(f"{qid} {mark} golden={doc[:6]}~{golden} rank={rank} | Q: {q}")
            print(f"      top5: {top5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
