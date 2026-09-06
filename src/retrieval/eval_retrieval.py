"""检索基线统一评测：recall@1/3/5 + MRR@5。

用法（项目根）:
    python src/retrieval/eval_retrieval.py --method bm25            # baseline-1
    python src/retrieval/eval_retrieval.py --method dense           # baseline-2（默认）
    python src/retrieval/eval_retrieval.py --method bm25 --verbose  # 每问 top5 明细

命中判定: golden = (source_doc, golden_articles)；检索 top-k 出现同 doc 且
article_no ∈ golden 即命中。golden_articles 解析: "8"->{8}; "37,38"->{37,38}。
dense 模式把 40 条 question 批量嵌入一次，再逐问做余弦检索。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from retrieval.bm25 import BM25Index, load_corpus  # noqa: E402

EVAL_CSV = _ROOT / "eval" / "qa_eval_set.csv"


def parse_golden(s: str) -> set[int]:
    return {int(x) for x in s.split(",") if x.strip().isdigit()}


def _load_test_rows() -> list[dict]:
    with open(EVAL_CSV, encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r["split"] == "test"]


def run(method: str = "dense", verbose: bool = False) -> int:
    chunks = load_corpus()
    rows = _load_test_rows()
    n = len(rows)
    print(f"方法: {method} | 索引 {len(chunks)} 条款块 | 评估集 test {n} 条")

    hits_at = {1: 0, 3: 0, 5: 0}
    rr_sum = 0.0
    miss: list[str] = []
    detail: list[tuple] = []

    if method == "bm25":
        idx = BM25Index(chunks)
        def top5(i: int) -> list:
            return idx.search(rows[i]["question"], k=5)
    else:  # dense
        from retrieval.dense import DenseIndex, embed_texts, load_or_build
        mat = load_or_build(chunks)
        idx = DenseIndex(chunks, mat)
        print("批量嵌入 question（1 次 API 调用）…")
        q_vecs = embed_texts([r["question"] for r in rows], quiet=False)
        def top5(i: int) -> list:
            return idx.search_by_vector(q_vecs[i], k=5)

    for i, r in enumerate(rows):
        qid = r["id"]
        doc, golden = r["source_doc"], parse_golden(r["golden_articles"])
        hits = top5(i)
        rank = next((p for p, h in enumerate(hits, start=1)
                     if h.doc == doc and h.article_no in golden), None)
        for k in hits_at:
            if rank is not None and rank <= k:
                hits_at[k] += 1
        rr_sum += 1.0 / rank if rank else 0.0
        if rank is None:
            miss.append(qid)
        detail.append((qid, r["question"][:26], doc, sorted(golden), rank,
                       [f"{h.doc[:6]}~{h.article_no}" for h in hits]))

    print(f"recall@1 = {hits_at[1]}/{n} = {hits_at[1]/n:.3f}")
    print(f"recall@3 = {hits_at[3]}/{n} = {hits_at[3]/n:.3f}")
    print(f"recall@5 = {hits_at[5]}/{n} = {hits_at[5]/n:.3f}")
    print(f"MRR@5   = {rr_sum/n:.3f}")
    print(f"top5 未命中: {len(miss)} 条: {miss}")

    if verbose:
        print("\n--- 每问 top5 明细 ---")
        for qid, q, doc, golden, rank, top5h in detail:
            mark = "OK" if rank is not None else "XX"
            print(f"{qid} {mark} golden={doc[:6]}~{golden} rank={rank} | {q}")
            print(f"      top5: {top5h}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="dense", choices=["bm25", "dense"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    return run(method=args.method, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
