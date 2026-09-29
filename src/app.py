"""海事法规 RAG 问答 —— Gradio Web 界面 v2（海事深蓝+金 品牌版）。

链路: 641 条款块(5部法规) → hybrid 检索 top5(α=0.7) → DeepSeek-V4-Flash(硅基流动)
      → 强制 [n] 引用 → 右侧引用卡片（可折叠，出处可核验）。
v2 定制（对应复试讲稿 §25-28' 可讲）：
  - 品牌顶栏：深蓝渐变横幅 + 金色点缀 + 5 部法规徽章（不署名，保持中立）。
  - 交互：多轮上下文轮数可调(0/1/2)；回答尾注带来源分布徽章(如 船员条例×3·海商法×2)；
    示例问题按法规分组，点按填入回车发送（dataset.click 尽力自动发送，失败降级为填入）；
    引用面板 = 原生 <details> 折叠卡片，首条展开、其余收起。
用法: python src/app.py   （浏览器打开 http://127.0.0.1:7860）
"""

from __future__ import annotations

import html
import re
import sys
import time
from collections import Counter
from pathlib import Path

import gradio as gr

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "generation"))

from config import MIX_ALPHA, TOP_K  # noqa: E402
from qa import LegalQA  # noqa: E402

# 线上最优链路：hybrid α=0.7（无 rerank 评测最优），top5
# α / top_k 统一由 config 提供（.env 可覆盖），不再硬编码
qa = LegalQA(method="hybrid", alpha=MIX_ALPHA, rerank=False, k=TOP_K)

_CITE_RE = re.compile(r"\[(\d{1,2})\]")
_META_RE = re.compile(r"\n\n_（检索.*?）_$")  # 尾注（来源徽章+耗时），喂回 LLM 前剥离

# doc 全名 -> 徽章短名（对应 data/*.jsonl 的 doc 字段）
DOC_SHORT = {
    "中华人民共和国海商法": "海商法",
    "中华人民共和国海上交通安全法": "海交法",
    "中华人民共和国内河交通安全管理条例": "内河条例",
    "中华人民共和国船员条例": "船员条例",
}


def _short(doc: str) -> str:
    if doc in DOC_SHORT:
        return DOC_SHORT[doc]
    return doc.replace("中华人民共和国", "")[:8] or doc


# 示例问题按法分类 —— 每组均经混合检索实测 top1 命中该法（_ex_classify_tmp 结果）
EXAMPLES = {
    "海商法": [
        "两船碰撞互有过错时，赔偿责任如何分配？",
        "同一船舶设立两个以上船舶抵押权，清偿顺序如何确定？",
        "船舶优先权包括哪些项目？",
    ],
    "海交法": [
        "客船能否同时载运乘客和危险货物？",
        "船舶进出港口需要办理什么手续？",
        "海上交通事故发生后，当事人应当向谁报告？",
    ],
    "内河条例": [
        "内河航道内禁止哪些行为？",
        "在内河通航水域举行大型群众性活动需要什么手续？",
        "船舶在内河航行，需要遵守哪些航行规则？",
    ],
    "船员条例": [
        "船员的遣返费用由谁承担？",
        "船员在船工作期间，用人单位应当为其缴纳什么保险？",
        "船员发证机构是哪级海事管理机构？",
    ],
    "航道法": [
        "禁止危害航道通航安全的行为有哪些？",
        "在航道保护范围内采砂需要遵守什么规定？",
        "在航道内倾倒垃圾会有什么法律后果？",
    ],
}

PLACEHOLDER = ("输入海事法规问题（回车发送）。支持追问，如先问「船员遣返费用由谁承担？」"
               "再问「包括哪些内容？」")
REFS_PLACEHOLDER = '<div class="wb-placeholder">提问后此处显示引用条文（可折叠）与出处。</div>'

THEME = gr.themes.Soft(primary_hue=gr.themes.colors.blue,
                       neutral_hue=gr.themes.colors.gray)

# 深蓝+金 海事风；明暗双主题用 gradio 原生 CSS 变量取文字/背景色
CSS = """
.wb-banner{background:linear-gradient(120deg,#0a2747 0%,#14508a 60%,#0e3a66 100%);
  border-radius:14px;border-left:5px solid #d8b04c;padding:18px 24px 14px;margin-bottom:6px}
.wb-banner h1{margin:0;font-size:1.32rem;color:#ffffff;letter-spacing:.5px}
.wb-banner .sub{color:#ecdda6;margin:6px 0 0;font-size:.9rem;line-height:1.6}
.wb-banner .laws{margin-top:10px}
.wb-badge{display:inline-block;background:rgba(255,255,255,.13);
  border:1px solid rgba(216,176,76,.6);color:#f6ecc9;border-radius:999px;
  padding:2px 11px;font-size:.76rem;margin:2px 6px 2px 0}
.wb-refcard details{background:var(--block-background-fill);
  border:1px solid var(--border-color-primary);border-left:3px solid #b9973f;
  border-radius:8px;padding:4px 11px;margin:6px 0}
.wb-refcard summary{cursor:pointer;color:var(--body-text-color);
  font-size:.86rem;line-height:1.6;font-weight:600}
.wb-refcard summary b{color:#b9973f}
.wb-refcard .body{margin:4px 2px 6px;padding-top:6px;
  border-top:1px dashed var(--border-color-primary);
  color:var(--body-text-color);font-size:.84rem;line-height:1.65;white-space:pre-wrap}
.wb-placeholder{color:var(--body-text-color);opacity:.62;font-size:.9rem}
.wb-foot{color:var(--body-text-color);opacity:.55;font-size:.78rem;
  text-align:center;margin:12px 0 2px}
"""


def _source_stats(hits: list[dict]) -> str:
    """回答尾注的来源分布徽章文案：'船员条例×3 · 海商法×2'。"""
    cnt = Counter(h["doc"] for h in hits)
    parts = [f"{_short(d)}×{c}" for d, c in cnt.most_common()]
    return " · ".join(parts) if parts else "—"


def _refs_html(answer: str, hits: list[dict]) -> str:
    """把 answer 中的 [n] 展开为可折叠引用卡片（<details>，首条展开）。"""
    used = [int(x) for x in _CITE_RE.findall(answer)]
    used = [n for n in used if 1 <= n <= len(hits)]
    if not used:
        return ('<div class="wb-placeholder">回答未引用检索条文'
                '（可能直接说明“未规定”或拒绝作答）。</div>')
    parts = ['<div class="wb-refcard">']
    for i, n in enumerate(sorted(set(used))):
        h = hits[n - 1]
        loc = html.escape(h["source_loc"] or f"{h['doc']} 第{h['article_no']}条")
        body = html.escape(h["text"])
        open_ = " open" if i == 0 else ""
        parts.append(
            f'<details{open_}><summary>[{n}] <b>{html.escape(_short(h["doc"]))}</b>'
            f" · {loc}</summary><div class=\"body\">{body}</div></details>"
        )
    parts.append("</div>")
    return "".join(parts)


def _content_str(content) -> str:
    """Chatbot 消息 content 兼容纯 str 与 gradio6 结构化列表
    [{'text':..., 'type':'text'}, ...]（messages 格式回传为后者）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for seg in content:
            if isinstance(seg, str):
                parts.append(seg)
            elif isinstance(seg, dict):
                parts.append(str(seg.get("text") or seg.get("content") or ""))
        return "".join(parts)
    return "" if content is None else str(content)


def _to_llm_history(history: list, rounds: int) -> list[dict] | None:
    """界面历史 -> 喂 LLM 的消息（content 统一转 str 并剥尾注；rounds=0 返回 None）。"""
    if rounds <= 0:
        return None
    out = []
    for msg in history:
        if msg.get("role") not in ("user", "assistant"):
            continue
        content = _META_RE.sub("", _content_str(msg.get("content")))
        if content.strip():
            out.append({"role": msg["role"], "content": content.strip()})
    msgs = out[-(rounds * 2):]
    return msgs or None


def answer(question: str, history: list, rounds: str) -> tuple:
    history = list(history or [])
    history.append({"role": "user", "content": question})
    if not question.strip():
        history.append({"role": "assistant", "content": "请输入问题。"})
        return history, "", REFS_PLACEHOLDER
    t0 = time.time()
    try:
        resp = qa.ask(question.strip(),
                      history_msgs=_to_llm_history(history[:-1], int(rounds or 2)))
        dt = time.time() - t0
        footer = (f"\n\n_（检索 641 条款块 · top5：{_source_stats(resp['hits'])}"
                  f" · 用时 {dt:.1f}s）_")
        history.append({"role": "assistant",
                        "content": resp["answer"] + footer})
        return history, "", _refs_html(resp["answer"], resp["hits"])
    except Exception as e:  # noqa: BLE001 —— 界面兜底显示，同时打印 traceback 供服务端诊断
        import traceback
        traceback.print_exc()
        history.append({"role": "assistant",
                        "content": f"请求失败（{type(e).__name__}）：{e}"
                                   "\n\n请检查 .env 的 API 配置后重试。"})
        return history, "", REFS_PLACEHOLDER


def clear_all() -> tuple:
    return [], REFS_PLACEHOLDER


def build() -> gr.Blocks:
    with gr.Blocks(title="海事法规 RAG 问答") as demo:
        # —— 品牌顶栏（不署名，保持中立） ——
        gr.HTML(
            '<div class="wb-banner">'
            "<h1>海事法规 RAG 问答</h1>"
            '<div class="sub">条款级检索 · 强制引用溯源 · 答案逐条可核验。'
            "回答只依据检索到的条文，句末 [n] 对应右侧出处。</div>"
            '<div class="laws">'
            '<span class="wb-badge">海商法 310 条</span>'
            '<span class="wb-badge">海交法 122 条</span>'
            '<span class="wb-badge">内河条例 95 条</span>'
            '<span class="wb-badge">船员条例 66 条</span>'
            '<span class="wb-badge">航道法 48 条</span>'
            '<span class="wb-badge">合计 641 条款</span>'
            "</div></div>"
        )
        chatbot = gr.Chatbot(height=400, show_label=False, autoscroll=True,
                             placeholder="输入问题开始（支持多轮追问）")
        with gr.Row():
            with gr.Column(scale=7):
                msg = gr.Textbox(placeholder=PLACEHOLDER, show_label=False,
                                 autofocus=True, lines=1)
                with gr.Row():
                    clear_btn = gr.Button("清空对话", size="sm", variant="secondary")
                    rounds_dd = gr.Dropdown(
                        choices=[("仅当前问题", "0"), ("最近 1 轮", "1"),
                                 ("最近 2 轮", "2")],
                        value="2", label="多轮上下文", min_width=150)
                gr.Markdown("**示例问题**（按法规分类，点按填入后回车发送）")
                ex_list: list = []
                # 5 组拆两行防横向溢出
                for row_laws in (("海商法", "海交法", "内河条例"),
                                 ("船员条例", "航道法")):
                    with gr.Row():
                        for law in row_laws:
                            with gr.Column(min_width=140):
                                ex = gr.Examples(label=law,
                                                 examples=EXAMPLES[law],
                                                 inputs=msg,
                                                 examples_per_page=3)
                                ex_list.append(ex)
            with gr.Column(scale=5):
                refs = gr.HTML(REFS_PLACEHOLDER)
        # 示例点按自动问答（须在两栏组件都创建后再挂接）；若事件未触发，
        # 问题已填入输入框，回车即可
        for ex in ex_list:
            ex.dataset.click(answer, [msg, chatbot, rounds_dd],
                             [chatbot, msg, refs])
        gr.HTML('<div class="wb-foot">条款级检索（BM25+BGE-M3 混合 α=0.7）· '
                "引用约束生成（DeepSeek-V4-Flash）· 数据口径见 data/SOURCES.md"
                "</div>")
        msg.submit(answer, [msg, chatbot, rounds_dd], [chatbot, msg, refs])
        clear_btn.click(clear_all, None, [chatbot, refs])
    return demo


if __name__ == "__main__":
    print("预热检索索引（首次较慢）…")
    t0 = time.time()
    try:
        qa.retriever.retrieve("预热", k=1)
        print(f"索引就绪 {time.time()-t0:.1f}s，启动界面 …")
    except Exception as e:  # noqa: BLE001
        print(f"预热失败（不影响启动，提问时再试）: {e}")
    build().queue().launch(server_name="127.0.0.1", server_port=7860,
                           theme=THEME, css=CSS, show_error=True)
