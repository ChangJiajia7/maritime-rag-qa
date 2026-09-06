"""生成侧评测：G1（无引用约束对照）vs G2（RAG + 强制引用约束）。

流程:
  1. 对 40 条 test：G1 = 纯 LLM 直接答；G2 = 检索 top5 + 条文约束答
  2. LLM judge 逐条批改（correct / fabricated）——fabricated 口径 v2：
     只记「与标准答案冲突」或「凭空编造具体事实（数字/期限/机关/义务归属）」；
     「补充了检索条文中的其它真实相关条款」不算幻觉（有 [n] 引用可核验）。
  3. 汇总 正确率 / 幻觉率 / G2 引用命中率；原始输出落 experiments/g_outputs.json

用法（项目根，真实调用 DeepSeek）:
    python src/generation/eval_generation.py            # 全流程（生成+判，~4 分钟）
    python src/generation/eval_generation.py --rejudge  # 只重判 g_outputs.json（改口径后复用）
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "src" / "generation"))
sys.path.insert(0, str(_ROOT / "src" / "retrieval"))

from qa import LegalQA, _post_chat, ask_no_context  # noqa: E402

EVAL_CSV = _ROOT / "eval" / "qa_eval_set.csv"
OUT_JSON = _ROOT / "experiments" / "g_outputs.json"
_CITE_RE = re.compile(r"\[(\d{1,2})\]")
_JSON_RE = re.compile(r"\{.*\}", re.S)

JUDGE_SYSTEM_V2 = (
    "你是严格的海事法规批改员。每次给定【问题】【标准答案】【候选回答】，只输出 JSON："
    '{"correct":0或1,"fabricated":0或1,"reason":"一句话"}\n'
    "判定规则：\n"
    "- correct=1：候选回答与标准答案要点一致（允许同义改写、允许更口语化）。\n"
    "- fabricated=1（真幻觉）：候选回答包含与标准答案冲突的事实，或凭空编造具体内容"
    "（标准答案里没有、候选也无法给出出处支撑的数字/期限/机关/处罚/义务归属）。\n"
    "- 候选回答补充了与问题相关、表述有据的其它内容，只要不与标准答案冲突、不属明显编造，"
    "fabricated 记为 0（这是更全面的回答，不是幻觉）。\n"
    "- 候选回答给出不同但同样正确的依据条款时不影响 correct。"
)


def judge(q: str, golden: str, cand: str) -> dict:
    resp = _post_chat([
        {"role": "system", "content": JUDGE_SYSTEM_V2},
        {"role": "user",
         "content": f"【问题】{q}\n【标准答案】{golden}\n【候选回答】{cand}"},
    ], temperature=0.0, max_tokens=200)
    m = _JSON_RE.search(resp)
    if not m:
        return {"correct": None, "fabricated": None, "reason": f"parse fail: {resp[:80]}"}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"correct": None, "fabricated": None, "reason": f"json fail: {resp[:80]}"}


def g2_cite_hit(resp: dict, doc: str, golden: set[int]) -> bool:
    by_n = {h["n"]: h for h in resp["hits"]}
    cited = [by_n[int(n)] for n in _CITE_RE.findall(resp["answer"]) if int(n) in by_n]
    return any(h["doc"] == doc and h["article_no"] in golden for h in cited)


def summarize(outputs: list[dict], rows_map: dict) -> dict:
    n = len(outputs)
    g1_c = sum(o["j1"].get("correct") == 1 for o in outputs)
    g1_f = sum(o["j1"].get("fabricated") == 1 for o in outputs)
    g2_c = sum(o["j2"].get("correct") == 1 for o in outputs)
    g2_f = sum(o["j2"].get("fabricated") == 1 for o in outputs)
    cite_hit = sum(o["cite_hit"] for o in outputs)
    s = {
        "n": n,
        "g1_correct": f"{g1_c}/{n}={g1_c/n:.3f}", "g1_fabricated": f"{g1_f}/{n}={g1_f/n:.3f}",
        "g2_correct": f"{g2_c}/{n}={g2_c/n:.3f}", "g2_fabricated": f"{g2_f}/{n}={g2_f/n:.3f}",
        "g2_cite_hit": f"{cite_hit}/{n}={cite_hit/n:.3f}",
    }
    print("=" * 46)
    print(f"G1 无引用约束 : 正确率 {s['g1_correct']} | 幻觉率 {s['g1_fabricated']}")
    print(f"G2 RAG+引用  : 正确率 {s['g2_correct']} | 幻觉率 {s['g2_fabricated']}")
    print(f"G2 引用命中率: {s['g2_cite_hit']}（回答引用的条款含 golden 条款）")
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rejudge", action="store_true", help="只重判 g_outputs.json（不重新生成）")
    args = ap.parse_args()

    rows = []
    with open(EVAL_CSV, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test":
                rows.append(r)

    if args.rejudge:
        outputs = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        print(f"重判 {len(outputs)} 条（口径 v2；cite_hit 为程序化判定，保留原值）…")
        t0 = time.time()
        rowmap = {r["id"]: r for r in rows}
        for i, o in enumerate(outputs, start=1):
            r = rowmap[o["id"]]
            o["j1"] = judge(o["q"], r["answer"], o["g1"])
            o["j2"] = judge(o["q"], r["answer"], o["g2"])
            if i % 10 == 0 or i == len(outputs):
                print(f"  {i}/{len(outputs)} 已跑 {time.time()-t0:.0f}s", flush=True)
        OUT_JSON.write_text(json.dumps(outputs, ensure_ascii=False, indent=1), encoding="utf-8")
        summarize(outputs, rows)
        return 0

    # —— 全流程：生成 + 判定 ——
    qa = LegalQA()
    t0 = time.time()
    outputs = []
    print(f"共 {len(rows)} 条，逐条跑 G1(无约束)+G2(RAG约束)+judge …", flush=True)
    for i, r in enumerate(rows, start=1):
        doc = r["source_doc"]
        golden = {int(x) for x in r["golden_articles"].split(",") if x.strip().isdigit()}
        g1 = ask_no_context(r["question"])
        g2 = qa.ask(r["question"])
        outputs.append({"id": r["id"], "q": r["question"], "g1": g1["answer"],
                        "g2": g2["answer"], "g2_hits": g2["hits"],
                        "j1": judge(r["question"], r["answer"], g1["answer"]),
                        "j2": judge(r["question"], r["answer"], g2["answer"]),
                        "cite_hit": g2_cite_hit(g2, doc, golden)})
        if i % 5 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} 已跑 {time.time()-t0:.0f}s", flush=True)
    OUT_JSON.write_text(json.dumps(outputs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n原始输出已存 {OUT_JSON.name}，总耗时 {time.time()-t0:.0f}s")
    summarize(outputs, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
