"""海事法规 RAG 问答 —— Gradio Web 界面（演示/交付形态）。

链路: 593 条款块(4部法规) → hybrid 检索 top5(α=0.7) → DeepSeek-V4-Flash(硅基流动)
      → 强制 [n] 引用 → 界面右侧展示引用条文出处，可核验。

用法: python src/app.py   （浏览器打开 http://127.0.0.1:7860）
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import gradio as gr

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "generation"))

from qa import LegalQA  # noqa: E402

# 线上最优链路：hybrid α=0.7（无 rerank 评测最优），top5
qa = LegalQA(method="hybrid", alpha=0.7, rerank=False, k=5)

_CITE_RE = re.compile(r"\[(\d{1,2})\]")

EXAMPLES = [
    "客船能否同时载运乘客和危险货物？",
    "船员的遣返费用由谁承担？",
    "两船碰撞互有过错时，赔偿责任如何分配？",
    "同一船舶设立两个以上船舶抵押权，清偿顺序如何确定？",
    "内河航道内禁止哪些行为？",
    "船长在航行中死亡，由谁代理其职务？",
]

PLACEHOLDER = "输入海事法规问题（回车发送），例如：船员适任证书被吊销后多久不得重新申请？"


def _ref_markdown(answer: str, hits: list[dict]) -> str:
    """把 answer 中的 [n] 展开为引用清单（source_loc + 条文摘录）。"""
    used = [int(x) for x in _CITE_RE.findall(answer)]
    used = [n for n in used if 1 <= n <= len(hits)]
    if not used:
        return "回答未引用检索条文（可能直接说明'未规定'或拒绝作答）。"
    lines = ["**引用条文（出处可核验）**", ""]
    for n in sorted(set(used)):
        h = hits[n - 1]
        body = h["text"].replace("\n", " ")
        lines.append(f"**{n}.** `{h['source_loc']}`")
        lines.append(f"> {body[:170]}{'…' if len(body) > 170 else ''}")
        lines.append("")
    return "\n".join(lines)


def answer(question: str, history: list) -> tuple:
    history = list(history or [])
    history.append({"role": "user", "content": question})
    if not question.strip():
        history.append({"role": "assistant", "content": "请输入问题。"})
        return history, "", "提问后此处显示引用条文与出处。"
    t0 = time.time()
    try:
        resp = qa.ask(question.strip())
        dt = time.time() - t0
        content = resp["answer"] + f"\n\n_（检索 593 条款块 top5 · 用时 {dt:.1f}s）_"
        history.append({"role": "assistant", "content": content})
        return history, "", _ref_markdown(resp["answer"], resp["hits"])
    except Exception as e:  # noqa: BLE001 —— 界面要兜底显示
        history.append({"role": "assistant",
                        "content": f"请求失败（{type(e).__name__}）：{e}\n\n请检查 .env 的 API 配置后重试。"})
        return history, "", "请求失败，引用面板暂无可显示内容。"


def build() -> gr.Blocks:
    with gr.Blocks(title="海事法规 RAG 问答") as demo:
        gr.Markdown(
            "# 海事法规 RAG 问答\n"
            "基于 4 部现行海事法规（**海商法 310 · 海交法 122 · 内河条例 95 · 船员条例 66 = 593 条**）"
            "的条款级检索问答。回答**只依据检索到的条文**，句末 `[n]` 对应右侧引用出处，可逐条核验。"
        )
        chatbot = gr.Chatbot(height=430, show_label=False, autoscroll=True)
        with gr.Row():
            with gr.Column(scale=5):
                msg = gr.Textbox(placeholder=PLACEHOLDER, show_label=False,
                                 autofocus=True, lines=1)
                msg.submit(answer, [msg, chatbot], [chatbot, msg])
                gr.Examples(examples=EXAMPLES, inputs=msg, label="示例问题")
            with gr.Column(scale=4):
                refs = gr.Markdown("提问后此处显示引用条文与出处。")
    return demo


if __name__ == "__main__":
    print("预热检索索引（首次较慢）…")
    t0 = time.time()
    try:
        qa.retriever.retrieve("预热", k=1)
        print(f"索引就绪 {time.time()-t0:.1f}s，启动界面 …")
    except Exception as e:  # noqa: BLE001
        print(f"预热失败（不影响启动，提问时再试）: {e}")
    build().queue().launch(server_name="127.0.0.1", server_port=7860, show_error=True)
