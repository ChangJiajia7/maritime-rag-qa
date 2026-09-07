"""政府门户法规页 html → 纯文本清洗（法规语料专用，与 docx/epub 清洗并列）。

背景: 维基文库(zh)/司法部法规库搜索接口不可达时，政府门户转载页（来源多为中国人大网，
     如 gd.gov.cn 政策库、gov.cn 公报）是现行法规全文的可靠来源。2026-09-07 航道法
     入库即用此法（见 data/SOURCES.md #5）。

页面结构假设（gd.gov.cn 实证）:
- 正文在 <p> 段落中；大条文常被拆成多个 <p>（续段不带条号，靠 chunk_law 状态机合并，不丢）
- 页头/页脚噪声（电话/版权/分享）多为 <div> 文本，不在正文 <p> 流内或位于正文前
- 页面可能含"目录"（第一条之前的章标题重复段）与正文章标题（紧邻第一条前）

用法:
    python ingest/gov_html_to_txt.py <in.html> <out.txt> [正文起始段号]

输出: 清洗后 txt（段落行），正文从正文章标题/第一条起，到最后一个"第X条"段落止。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_P_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_ART_RE = re.compile(r"^第([一二三四五六七八九十百零]+)条")
_CH_RE = re.compile(r"^第([一二三四五六七八九十百零]+)章")
# 页脚/导航噪声关键词（正文末条后的续段吸收时用于截断）
_FOOT_KW = ("版权", "备案", "分享", "打印", "纠错", "上一篇", "下一篇", "主办",
            "承办", "电话", "邮箱", "微信", "微博", "返回顶部", "站点地图", "友情链接")


def extract_paras(html: str) -> list[str]:
    """全页 <p> → 文本段落（去 tag、全角空格改半角、去空）。"""
    out = []
    for raw in _P_RE.findall(html):
        p = _TAG_RE.sub("", raw).replace("\u3000", " ").replace("\xa0", " ").strip()
        if p and len(p) > 1:
            out.append(p)
    return out


def locate_body(paras: list[str]) -> tuple[int, int]:
    """定位正文 [start, end)。
    start = 第一个'第X条'段；若其前一段是章标题(第一章)则并入（正文章标题）。
    end   = 最后一个'第X条'段之后，再向后吸收紧邻的续段（大条文被 <p> 拆出的
            末条续段），遇章标题/下一个条/页脚噪声关键词即停。"""
    art_idx = [i for i, p in enumerate(paras) if _ART_RE.match(p)]
    if not art_idx:
        raise ValueError("未找到任何'第X条'段落，可能非法规正文页")
    start = art_idx[0]
    if start > 0 and _CH_RE.match(paras[start - 1]):
        start -= 1
    end = art_idx[-1] + 1
    while end < len(paras):
        p = paras[end]
        if _ART_RE.match(p) or _CH_RE.match(p):
            break
        if any(k in p for k in _FOOT_KW):
            break
        end += 1
    return start, end


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    in_path, out_path = sys.argv[1], sys.argv[2]
    html = Path(in_path).read_text(encoding="utf-8", errors="ignore")
    paras = extract_paras(html)
    start, end = locate_body(paras)
    body = paras[start:end]
    arts = [p for p in body if _ART_RE.match(p)]
    chs = [p for p in body if _CH_RE.match(p)]
    Path(out_path).write_text("\n".join(body), encoding="utf-8")
    print(f"输出: {out_path}")
    print(f"正文段落 {start}-{end}（共 {len(body)} 段）| 条款数: {len(arts)} | 章标题数: {len(chs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
