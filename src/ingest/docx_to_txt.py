"""docx → 纯文本清洗（法规语料专用）。

输入: 官方 docx（含 第X条 结构化标题）
输出: 清洗后 txt，保留条款/款/项结构，供后续条款感知分块使用。

用法:
    python ingest/docx_to_txt.py <in.docx> <out.txt>

要点:
- 用标准库 zipfile 解 word/document.xml，不依赖 python-docx
- 段落按 <w:p> 输出，<w:tab> -> 空格，超长行不合并（保留原文分段）
- 抽取后人工抽检条款边界（见 SOURCES.md 纪律）
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def extract_paragraphs(docx_path: str) -> list[str]:
    """返回 docx 全部段落文本（按文档顺序）。"""
    with zipfile.ZipFile(docx_path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    paras: list[str] = []
    for p in root.iter(f"{{{NS['w']}}}p"):
        parts: list[str] = []
        for node in p.iter():
            tag = node.tag.split("}")[-1]
            if tag == "t" and node.text:
                parts.append(node.text)
            elif tag == "tab":
                parts.append(" ")
            elif tag == "br":
                parts.append("\n")
        text = "".join(parts).strip()
        paras.append(text)
    return paras


def clean_law_text(paras: list[str]) -> str:
    """轻度清洗：去空段/URL 污染行（官方 docx 常带来源网页注脚）。"""
    url_re = re.compile(r"^https?://\S+$")
    out: list[str] = []
    for line in paras:
        if not line:
            continue
        if url_re.match(line.strip()):
            continue  # 纯 URL 行（页面注脚）—— 检索噪声，剔除
        out.append(line)
    return "\n".join(out)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    in_path, out_path = sys.argv[1], sys.argv[2]
    paras = extract_paragraphs(in_path)
    text = clean_law_text(paras)
    Path(out_path).write_text(text, encoding="utf-8")

    n_articles = len(re.findall(r"^第[一二三四五六七八九十百千]+条", text, re.M))
    print(f"段落数: {len(paras)}  输出: {out_path}")
    print(f"识别条款数(粗略): {n_articles}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
