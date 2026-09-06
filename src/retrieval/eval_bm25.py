"""BM25 评测入口（baseline-1）——薄壳，逻辑见 eval_retrieval.run。

用法: python src/retrieval/eval_bm25.py [--verbose]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_retrieval import run  # noqa: E402


def main() -> int:
    verbose = "--verbose" in sys.argv
    return run(method="bm25", verbose=verbose)


if __name__ == "__main__":
    sys.exit(main())
