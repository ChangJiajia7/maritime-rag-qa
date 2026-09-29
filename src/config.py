"""配置加载：.env -> 常量。

全 API 三件套（LLM / Embedding / Rerank）各自独立配置，互不绑定，
便于按供应商可达性与性价比自由组合。.env 不入 git（见 .gitignore），
本模块在仓库根目录加载 .env，若缺文件则回退到环境变量/示例默认值并告警。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = _ROOT / ".env") -> None:
    """极简 .env 解析（无需 python-dotenv 依赖）。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:  # 已存在的环境变量优先
            os.environ[k] = v


_load_dotenv()


class _API(NamedTuple):
    base: str
    key: str
    model: str


def _api(prefix: str, default_model: str) -> _API:
    return _API(
        base=os.getenv(f"{prefix}_API_BASE", "").rstrip("/"),
        key=os.getenv(f"{prefix}_API_KEY", ""),
        model=os.getenv(f"{prefix}_MODEL", default_model),
    )


LLM = _api("LLM", "deepseek-chat")
EMBED = _api("EMBED", "BAAI/bge-m3")
RERANK = _api("RERANK", "BAAI/bge-reranker-v2-m3")

# —— 检索运行参数（app.py / Retriever 实际消费）——
# 注意：分块按「条」切（见 ingest/chunk_law.py），不使用 token 长度参数，
# 故不存在 CHUNK_SIZE / CHUNK_OVERLAP —— 历史遗留的 512/64 已删除，避免误导。
TOP_K = int(os.getenv("TOP_K", "5"))
MIX_ALPHA = float(os.getenv("MIX_ALPHA", "0.7"))  # 见 log A3；注入面包屑后 α=0.9 更优（log A16）

# —— 检索文本的层级面包屑注入模式 ——
# 法条脱离章节就成了孤儿（如罚则条款"违反本规定…处以罚款"看不到适用的行为条款）。
# 注入章节归属让向量与 BM25 同时捕获主题。仅用于建索引；展示/引用仍用原 text。
#   0 = 不注入（原文）
#   1 = 【法规 > 章 > 第X条】+ 原文
#   2 = 【章 > 第X条】+ 原文（去掉法规名，避免法名共享词面带来稀疏侧干扰）
INDEX_BREADCRUMB = int(os.getenv("INDEX_BREADCRUMB", "1"))

# 本地路径
DATA_RAW = _ROOT / "data" / "raw"


def missing_keys() -> list[str]:
    """返回未配置的项名，用于启动自检。"""
    missing = []
    for name, api in (("LLM", LLM), ("EMBED", EMBED), ("RERANK", RERANK)):
        if not api.key or not api.base:
            missing.append(name)
    return missing


def summary() -> str:
    lines = [
        f"LLM   : {LLM.model} @ {LLM.base or '(未配置)'}",
        f"EMBED : {EMBED.model} @ {EMBED.base or '(未配置)'}",
        f"RERANK: {RERANK.model} @ {RERANK.base or '(未配置)'}",
        f"检索  : 条款级分块（按「条」，非 token 窗口）  top_k={TOP_K} alpha={MIX_ALPHA}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
    miss = missing_keys()
    print(f"缺失配置: {miss if miss else '无，全部就绪'}")
