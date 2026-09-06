"""生成侧 RAG 问答（G2 链路）：检索上下文 + 强制引用约束 + DeepSeek 生成。

用法（交互冒烟）:
    python src/generation/qa.py "客船能否同时载运乘客和危险货物？"

返回 dict: {answer, hits:[{source_loc, article_no, doc}...]}
- answer 中的 [n] 标记 = hits 列表下标（1-based）的引用
- citations() 把 [n] 展开为 source_loc 字符串，供界面/校验直接使用
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "src" / "retrieval"))

from config import LLM  # noqa: E402
from retrieval.retriever import Retriever  # noqa: E402

_TIMEOUT = 120
_RETRY = 2
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
    url = f"{LLM.base}/chat/completions"
    payload = {"model": LLM.model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    body = json.dumps(payload).encode("utf-8")
    last_err: Exception | None = None
    for _ in range(_RETRY + 1):
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {LLM.key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5)
    raise RuntimeError(f"LLM API 失败: {last_err}")


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

    def ask(self, question: str, temperature: float = 0.1) -> dict:
        ctx = self._context(question)
        blocks = []
        for it in ctx:
            blocks.append(f"[{it['n']}] {it['source_loc']}\n{it['text']}")
        user_msg = (f"【参考条文】\n" + "\n\n".join(blocks)
                    + f"\n\n【问题】{question}")
        answer = _post_chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ], temperature=temperature)
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
