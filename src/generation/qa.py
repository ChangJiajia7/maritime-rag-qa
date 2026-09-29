"""生成侧 RAG 问答（G2 链路）：检索上下文 + 强制引用约束 + DeepSeek 生成。

用法（交互冒烟）:
    python src/generation/qa.py "客船能否同时载运乘客和危险货物？"

返回 dict: {answer, hits:[{source_loc, article_no, doc}...]}
- answer 中的 [n] 标记 = hits 列表下标（1-based）的引用
- citations() 把 [n] 展开为 source_loc 字符串，供界面/校验直接使用
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "src" / "retrieval"))

from config import LLM  # noqa: E402
from net import post_json  # noqa: E402
from retrieval.retriever import Retriever  # noqa: E402

_CITE_RE = re.compile(r"\[(\d{1,2})\]")

SYSTEM_PROMPT = (
    "你是海事法规问答助手。回答规则：\n"
    "1. 只能依据【参考条文】作答，参考条文按 [编号] 列出；引用时在对应句子末尾加 [编号]。\n"
    "2. 严禁编造参考条文中没有的内容、数字、期限或出处；条文未规定时如实说明'相关条文中未规定'。\n"
    "3. 若问题与海事法规无关或参考条文无法回答，直接说明原因，不要强行作答。\n"
    "4. 回答用简体中文，先给结论再给依据，控制在 200 字以内。"
)


def _post_chat(messages: list[dict], temperature: float = 0.1,
               max_tokens: int = 700) -> str:
    data = post_json(f"{LLM.base}/chat/completions",
                     {"model": LLM.model, "messages": messages,
                      "temperature": temperature, "max_tokens": max_tokens},
                     LLM.key, label="LLM")
    return data["choices"][0]["message"]["content"].strip()


class LegalQA:
    """带检索的问答（G2）。"""

    def __init__(self, method: str = "hybrid", alpha: float = 0.7,
                 rerank: bool = False, k: int = 5):
        self.retriever = Retriever(method=method, alpha=alpha, rerank=rerank)
        self.k = k

    def _context(self, question: str) -> list[dict]:
        """检索 → [{"n":1, "source_loc":..., "article_no":..., "text":...}]。"""
        hits = self.retriever.retrieve(question, k=self.k)
        return [
            {"n": i + 1, "source_loc": c.source_loc or f"{c.doc} 第{c.article_no}条",
             "article_no": c.article_no, "doc": c.doc, "text": c.text}
            for i, (_, c) in enumerate(hits)
        ]

    def ask(self, question: str, history_msgs: list[dict] | None = None,
            temperature: float = 0.1) -> dict:
        """带检索的问答（G2）。history_msgs = 最近对话 [{"role","content"}]，
        作上下文让 LLM 理解追问指代；参考条文始终在本轮最后给出。
        追问查询扩展：本轮问题过短（疑似指代）时，拼接最近一次 user 问题再检索。"""
        # —— 查询扩展：短追问拼上轮问题，保证检索不丢主题 ——
        query = question.strip()
        if history_msgs and len(query) < 12:
            last_user = next((h["content"] for h in reversed(list(history_msgs))
                              if h.get("role") == "user"), "")
            if last_user:
                query = f"{last_user[:60]} {query}"
        ctx = self._context(query)
        blocks = []
        for it in ctx:
            blocks.append(f"[{it['n']}] {it['source_loc']}\n{it['text']}")
        user_msg = (f"【参考条文】\n" + "\n\n".join(blocks)
                    + f"\n\n【问题】{question}")

        msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history_msgs:
            for h in list(history_msgs)[-4:]:  # 保留最近 2 轮问答
                content = (h.get("content") or "").strip()
                if len(content) > 400:
                    content = content[:400] + "…"
                msgs.append({"role": h["role"], "content": content})
        msgs.append({"role": "user", "content": user_msg})
        answer = _post_chat(msgs, temperature=temperature)
        return {"question": question, "answer": answer,
                "hits": [{"n": it["n"], "source_loc": it["source_loc"],
                          "article_no": it["article_no"], "doc": it["doc"],
                          "text": it["text"]}
                         for it in ctx]}

    def citations(self, resp: dict) -> list[str]:
        """把 answer 中的 [n] 展开成 source_loc（引用后校验的输入）。"""
        locs = {h["n"]: h["source_loc"] for h in resp["hits"]}
        used = {int(m) for m in _CITE_RE.findall(resp["answer"])}
        return [locs[n] for n in sorted(used) if n in locs]


def ask_no_context(question: str, temperature: float = 0.1) -> dict:
    """无参考条文的对照组（G1）：纯靠模型知识回答，不约束出处。"""
    answer = _post_chat([
        {"role": "system",
         "content": "你是海事法规问答助手。请用简体中文回答下面的问题，先给结论再给依据，控制在 200 字以内。"},
        {"role": "user", "content": question},
    ], temperature=temperature)
    return {"question": question, "answer": answer, "hits": []}


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "客船能否同时载运乘客和危险货物？"
    qa = LegalQA()
    t0 = time.time()
    r = qa.ask(q)
    print(f"Q: {q}\n耗时 {time.time()-t0:.1f}s\n\nA: {r['answer']}\n")
    print("引用展开:")
    for loc in qa.citations(r):
        print("  -", loc)
