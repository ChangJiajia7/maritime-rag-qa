"""法规文本 → 条款级结构化分块（条款感知 chunking，支持 章/节/条 三级）。

输入: 清洗后的法规 txt（每段一行；章="第X章"、节="第X节"、条款="第X条" 各自成行）
输出: JSONL —— 每条 = 一个条款节点，保留 章/节 归属与 source_loc 供引用溯源。

用法:
    python ingest/chunk_law.py <law.txt> <out.jsonl> [doc名] [版本]

每条记录结构:
    {
      "doc": "中华人民共和国海商法",
      "doc_version": "2025修订",
      "chapter": "第二章　船舶",            # 章归属（无章则空）
      "section": "第一节　船舶所有权",       # 节归属（无节则空）
      "article_no": 7,                       # 阿拉伯数字条款号
      "article_cn": "第七条",
      "text": "第七条 船舶所有权，是指...",   # 条款正文（含所有款/项）
      "source_loc": "中华人民共和国海商法(2025修订) 第二章 第一节 第七条"
    }

doc 名/版本自动推断规则：
- 显式传第 3/4 参数则用之
- 否则查内置文件名映射表（law_mst_2021 / law_maritime_commerce_2025 …）
- 版本号默认取文件名中的 4 位年份 + "修订"
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
_SEC_RE = re.compile(r"^第([一二三四五六七八九十百零]+)节")

# 文件名(去扩展名) → 法规正式名
_NAME_MAP = {
    "law_mst_2021": "中华人民共和国海上交通安全法",
    "law_maritime_commerce_2025": "中华人民共和国海商法",
}


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


def infer_meta(in_path: str) -> tuple[str, str]:
    """从文件名推断 doc 名与版本；无法匹配则回退 stem。"""
    stem = Path(in_path).stem
    doc = _NAME_MAP.get(stem, stem)
    m_year = re.search(r"(20\d{2})", stem)
    version = f"{m_year.group(1)}修订" if m_year else ""
    return doc, version


def split_by_articles(lines: list[str], doc: str, doc_version: str) -> list[dict]:
    """按条款边界切分；章/节标题作为归属信息，不并入条款正文。"""
    records: list[dict] = []
    chapter, section = "", ""
    cur: dict | None = None

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
            section = ""  # 换章重置节
            continue
        m_sec = _SEC_RE.match(line)
        if m_sec:
            flush()
            section = line
            continue
        m_art = _ART_RE.match(line)
        if m_art:
            flush()
            cn_no = m_art.group(1)
            loc = f"{doc}({doc_version})"
            if chapter:
                loc += f" {chapter}"
            if section:
                loc += f" {section}"
            loc += f" {f'第{cn_no}条'}"
            cur = {
                "doc": doc,
                "doc_version": doc_version,
                "chapter": chapter,
                "section": section,
                "article_no": cn2int(cn_no),
                "article_cn": f"第{cn_no}条",
                "text": line,
                "source_loc": loc,
            }
            continue
        # 续行（同条目的款/项）
        if cur is not None:
            cur["text"] += "\n" + line
    flush()
    return records


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    in_path, out_path = sys.argv[1], sys.argv[2]
    doc = sys.argv[3] if len(sys.argv) > 3 else None
    doc_version = sys.argv[4] if len(sys.argv) > 4 else None

    lines = Path(in_path).read_text(encoding="utf-8").splitlines()
    if doc is None or doc_version is None:
        auto_doc, auto_ver = infer_meta(in_path)
        doc = doc or auto_doc
        doc_version = doc_version if doc_version is not None else auto_ver

    records = split_by_articles(lines, doc, doc_version)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    nums = [r["article_no"] for r in records if r["article_no"]]
    missing = [i for i in range(1, max(nums) + 1) if i not in nums]
    n_sec = len({r["section"] for r in records if r["section"]})
    print(f"输出: {out_path}  条款块数: {len(records)}  节数: {n_sec}")
    print(f"doc={doc} 版本={doc_version or '(空)'}")
    print(f"条款编号范围: {min(nums)}–{max(nums)}  缺失: {missing if missing else '无'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
