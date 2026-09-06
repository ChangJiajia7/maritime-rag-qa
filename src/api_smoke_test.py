"""API 三件套连通性自检（P0 验收）。

用法：
  1. 复制 .env.example 为 .env 并填入真实 key
  2. python src/api_smoke_test.py

逐个验证 LLM / Embedding / Rerank 是否可用，输出诊断与耗时。
纯标准库实现（urllib），无需额外依赖。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

from config import EMBED, LLM, RERANK, missing_keys

TIMEOUT = 60


def _post(url: str, key: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _check(name: str, base: str, key: str, path: str, payload: dict) -> bool:
    if not (base and key):
        print(f"[{name}] 未配置，跳过")
        return False
    url = f"{base}{path}"
    print(f"[{name}] {url}")
    t0 = time.time()
    try:
        data = _post(url, key, payload)
        dt = time.time() - t0
        print(f"[{name}] OK  {dt:.1f}s")
        return True
    except Exception as e:  # noqa: BLE001 —— 自检要看到原始错误
        print(f"[{name}] FAIL  {type(e).__name__}: {e}")
        return False


def main() -> int:
    miss = missing_keys()
    if miss:
        print(f"缺失配置项: {miss}")
        print("请先复制 .env.example 为 .env 并填入真实 key")
        return 1

    print("=" * 50)
    ok = []

    # ① LLM：最小 chat 补全
    ok.append(_check(
        "LLM", LLM.base, LLM.key, "/chat/completions",
        {"model": LLM.model, "messages": [{"role": "user", "content": "回复两个字：就绪"}], "max_tokens": 10},
    ))

    # ② Embedding：两条中文法规句
    ok.append(_check(
        "EMBED", EMBED.base, EMBED.key, "/embeddings",
        {"model": EMBED.model,
         "input": ["船舶夜间航行应当显示航行灯", "船长负责船舶的安全管理"]},
    ))

    # ③ Rerank：一对 query-doc
    ok.append(_check(
        "RERANK", RERANK.base, RERANK.key, "/rerank",
        {"model": RERANK.model,
         "query": "船舶夜间应当显示什么号灯",
         "documents": ["船舶夜间航行应当显示航行灯", "今天天气晴朗适合出海"]},
    ))

    print("=" * 50)
    n_ok = sum(ok)
    n_cfg = 3 - len(miss)
    print(f"通过 {n_ok}/{n_cfg}")
    return 0 if n_ok == n_cfg else 2


if __name__ == "__main__":
    sys.exit(main())
