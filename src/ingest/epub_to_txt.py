"""epub → 纯文本清洗（法规语料专用，Wikisource 等单正文文件 epub）。

输入: epub（正文为单个 XHTML，含 h2=章 / h3=节 / p=条款）
输出: 清洗后 txt，按文档顺序保留 章/节/条款 层级，供 chunk_law.py 复用。

用法:
    python ingest/epub_to_txt.py <in.epub> <out.txt>

要点:
- 标准库 zipfile 读正文 XHTML；正则按 h1-h3 标题与 p 段落还原顺序
- 条款行可能 "第X条正文" 同行（无空格），也可能条款在 p 内跨行
- 维基文库尾部版权声明段（"本作品来自中华人民共和国法律…"）会被剔除
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

_XHTML_RE = re.compile(r'<[^>]+>')
_ENT = {"&nbsp;": " ", "&#160;": " ", "&amp;": "&", "&lt;": "<",
        "&gt;": ">", "&quot;": '"', "&#39;": "'"}

def _clean_ent(t: str) -> str:
    for k, v in _ENT.items():
        t = t.replace(k, v)
    return t


def _find_body_xhtml(z: zipfile.ZipFile) -> str:
    """定位正文 XHTML：优先条款密度最高者（法规正文特征），spine 首项仅作次选。

    背景：部分 epub 的 OPF spine 首项是 title.xhtml（标题页），正文在
    c0_*.xhtml 等文件；条款文本量最大、含"第X条"最多的文件才是正文。
    """
    cands = [n for n in z.namelist() if n.endswith(".xhtml")]
    # 主选：含条款行最多的文件
    best, best_cnt = "", 0
    for c in cands:
        try:
            t = z.read(c).decode("utf-8", errors="ignore")
            cnt = len(re.findall(r"第[一二三四五六七八九十百零]+条", t))
            # 权重：条款越多越像正文；标题页这类导航页通常是 0
            if cnt > best_cnt:
                best, best_cnt = c, cnt
        except Exception:
            continue
    if best_cnt >= 30:  # 法规正文通常上百条，30 为保守阈值
        return best
    # 次选：OPF spine 第一项
    opf_name = next((n for n in z.namelist() if n.endswith(".opf")), None)
    if opf_name:
        opf = z.read(opf_name).decode("utf-8", errors="ignore")
        m_spine = re.search(r"<spine[^>]*>(.*?)</spine>", opf, re.S)
        if m_spine:
            first_idref = re.search(
                r'itemref[^>]*idref="([^"]+)"', m_spine.group(1))
            if first_idref:
                m_mani = re.search(
                    rf'<item[^>]*id="{re.escape(first_idref.group(1))}"[^>]*href="([^"]+)"',
                    opf)
                if m_mani:
                    href = m_mani.group(1)
                    base = opf_name.rsplit("/", 1)[0]
                    full = f"{base}/{href}" if base else href
                    if full in z.namelist():
                        return full
    return best if best else ""


def extract_to_lines(z: zipfile.ZipFile, xhtml_path: str) -> list[str]:
    """按文档顺序输出行：章/节/条款各占一行，正文续行合并不做（交给下游）。"""
    raw = z.read(xhtml_path).decode("utf-8", errors="ignore")
    # 去掉 license/页脚 div（维基文库版权横幅）
    raw = re.sub(r'<div[^>]*class="[^"]*(?:license|licensetpl|footer|nav)[^"]*"[^>]*>.*?</div>',
                 "", raw, flags=re.S | re.I)
    # 顺序扫描 body 内的 h1-h3 与 p
    body = re.search(r"<body[^>]*>(.*?)</body>", raw, re.S)
    body = body.group(1) if body else raw
    tokens = re.findall(
        r'<(h[1-3]|p)[^>]*>(.*?)</\1>', body, re.S)
    out: list[str] = []
    for tag, content in tokens:
        t = _clean_ent(_XHTML_RE.sub("", content)).strip()
        t = re.sub(r"[ \t\u3000]+", " ", t)
        if not t:
            continue
        # 维基文库版权尾注剔除
        if "本作品来自" in t or t.startswith("Public domain"):
            continue
        out.append(t)
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    in_path, out_path = sys.argv[1], sys.argv[2]
    with zipfile.ZipFile(in_path) as z:
        xhtml_path = _find_body_xhtml(z)
        if not xhtml_path:
            print("未找到正文 XHTML，退出")
            return 1
        print(f"正文文件: {xhtml_path}")
        lines = extract_to_lines(z, xhtml_path)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    n_art = len([l for l in lines if re.match(r"^第[一二三四五六七八九十百零]+条", l)])
    n_ch = len([l for l in lines if re.match(r"^第[一二三四五六七八九十百零]+章", l)])
    print(f"行数: {len(lines)}  条款行: {n_art}  章行: {n_ch}  输出: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
