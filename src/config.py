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

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
TOP_K = int(os.getenv("TOP_K", "5"))
MIX_ALPHA = float(os.getenv("MIX_ALPHA", "0.5"))

# 本地路径
DATA_RAW = _ROOT / "data" / "raw"
CHROMA_DIR = _ROOT / "chroma"


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
        f"chunk={CHUNK_SIZE}/{CHUNK_OVERLAP} top_k={TOP_K} alpha={MIX_ALPHA}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
    miss = missing_keys()
    print(f"缺失配置: {miss if miss else '无，全部就绪'}")
