# -*- coding: utf-8 -*-
"""阶段 0 出门判据：把本脚本的 4 个 TODO 全部实现，跑通《海上交通安全法》PDF 清洗。

跟学提示（周计划三铁律第 2 条：先自己动手再求助）：
  1. 先读 pdfplumber 官方文档 Quickstart（约 10 分钟）；
  2. 自己填 TODO，报错自己查；
  3. 实在卡住 30 分钟以上再搜索/问 AI，且明确要求"给提示，不要给完整代码"。

运行：python scripts/clean_pdf.py
输入：data/raw/*.pdf
输出：data/processed/<同名>.txt（一行一条，"第X条"开头的条文独立成行）
"""
from pathlib import Path

import pdfplumber  # pip install pdfplumber

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"


def extract_pages(pdf_path: Path) -> list[str]:
    """TODO 1：逐页提取文本，返回"每页一个字符串"的列表。

    提示：with pdfplumber.open(pdf_path) as pdf: 里 for page in pdf.pages:
    注意 page.extract_text() 可能返回 None（扫描页/空白页），要兜底成 ""。
    """
    raise NotImplementedError("TODO 1")


def clean_text(text: str) -> str:
    """TODO 2：基础清洗——去掉页眉页脚噪声、压缩连续空行与多余空格。

    提示：str.splitlines() + 逐行判断 + "\\n".join()；
    动手前先打印 2-3 页原文，观察噪声长什么样（页码？水印字？换行断裂？），再决定删什么。
    """
    raise NotImplementedError("TODO 2")


def split_articles(text: str) -> list[str]:
    """TODO 3：按"第X条"切分成条文列表（为后续分块与引用溯源做准备）。

    提示：import re；正则参考 r"第[一二三四五六七八九十百零]+条"；
    自检标准：切出的条文总数应与 PDF 目录/正文里的条文总数一致。
    """
    raise NotImplementedError("TODO 3")


def main() -> None:
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"[!] {RAW_DIR} 下没有 PDF。请先按 data/SOURCES.md 取数并放入原始文件。")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for pdf in pdfs:
        print(f"[*] 处理 {pdf.name}")
        pages = extract_pages(pdf)           # TODO 1
        text = clean_text("\n".join(pages))  # TODO 2
        articles = split_articles(text)      # TODO 3

        # TODO 4：把 articles 写入 OUT_DIR / (pdf.stem + ".txt")，一行一条；
        # 并打印统计：页数(len(pages))、总字符数(len(text))、条文数(len(articles))。
        # 这三个数要同步写进 data/SOURCES.md 的"覆盖范围"列。
        raise NotImplementedError("TODO 4")

    print("[✓] 完成。下一步：目视抽查 3 条条文与原文一致（防清洗丢内容）。")


if __name__ == "__main__":
    main()
