"""法规文本 → 条款级结构化分块（条款感知 chunking）。

输入: 清洗后的法规 txt（每段一行，条款行以 "第X条" 开头）
输出: JSONL —— 每条 = 一个"章"节点 + 若干"条款"节点，保留 source_loc 供引用溯源。

用法:
    python ingest/chunk_law.py <law.txt> <out.jsonl>

每条记录结构:
    {
      "doc": "海上交通安全法",
      "doc_version": "2021修订",
      "chapter": "第一章　总则",          # 章归属（无章则空）
      "article_no": 1,                    # 阿拉伯数字条款号
      "article_cn": "第一条",
      "text": "第一条 为了加强...",        # 条款正文（含所有款/项）
      "source_loc": "海上交通安全法(2021修订) 第一条"
    }
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# 中文数字转阿拉伯（1-999 够用）
_CN = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
       "六": 6, "七": 7, "八": 8, "九": 9}
_ART_RE = re.compile(r"^第([一二三四五六七八九十百零]+)条")
_CH_RE = re.compile(r"^第([一二三四五六七八九十百零]+)章")


def cn2int(s: str) -> int:
    s = s.replace("零", "")
    if not s:
        return 0
    if "百" in s:
        h, _, rest = s.partition("百")
        return (_CN[h] if h else 1) * 100 + cn2int(rest)
    if "十" in s:
        pre, _, post = s.partition("十")
        return (_CN[pre] if pre else 1) * 10 + (_CN[post] if post else 0)
    return _CN.get(s, 0)


def split_by_articles(lines: list[str], doc: str, doc_version: str) -> list[dict]:
    """按条款边界切分。条款正文 = 条款行 + 其后非条款非章的续行（款/项）。"""
    records: list[dict] = []
    chapter = ""
    cur = None  # 正在累积的条款

    def flush():
        nonlocal cur
        if cur is not None:
            records.append(cur)
            cur = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        m_ch = _CH_RE.match(line)
        if m_ch:
            flush()
            chapter = line
            continue
        m_art = _ART_RE.match(line)
        if m_art:
            flush()
            cn_no = m_art.group(1)
            cur = {
                "doc": doc,
                "doc_version": doc_version,
                "chapter": chapter,
                "article_no": cn2int(cn_no),
                "article_cn": f"第{cn_no}条",
                "text": line,
                "source_loc": f"{doc}({doc_version}) {chapter} {f'第{cn_no}条'}",
            }
            continue
        # 续行（同条目的款/项）
        if cur is not None:
            cur["text"] += "\n" + line
    flush()
    return records


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    in_path, out_path = sys.argv[1], sys.argv[2]
    text = Path(in_path).read_text(encoding="utf-8")
    lines = text.splitlines()

    # 文档标题取前几行作为 doc 名（去空白）
    doc = next((l.strip() for l in lines[:5] if l.strip()), Path(in_path).stem)
    doc_version = "2021修订" if "2021" in Path(in_path).stem else ""

    records = split_by_articles(lines, doc, doc_version)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    nums = [r["article_no"] for r in records if r["article_no"]]
    missing = [i for i in range(1, max(nums) + 1) if i not in nums]
    print(f"输出: {out_path}  条款块数: {len(records)}")
    print(f"条款编号范围: {min(nums)}–{max(nums)}  缺失: {missing if missing else '无'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
