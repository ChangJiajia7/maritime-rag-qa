# src/ 模块规划（对应讲稿 §13–20'）

| 模块 | 职责 | 状态 |
|---|---|---|
| `ingest/` | PDF 清洗 → **条款感知分块**（按条款边界切，非固定窗口）→ Chroma 入库 | 🚧 TODO |
| `retrieval/` | 向量(BGE-M3) + BM25 混合（权重 α）→ bge-reranker 重排；离线评测脚本出 recall@k / MRR | 🚧 TODO |
| `generation/` | Qwen2.5-7B（API 优先 / Ollama 兜底）+ 强制出处提示词 + 引用后校验 + Gradio 前端 | 🚧 TODO |
| `finetune/` | bge-small-zh-v1.5 + MultipleNegativesRankingLoss(InfoNCE) 微调；微调前后对比报告 | 🚧 TODO |

> 开发纪律（沿用个人工程原则）：一次只改一个模块，跑通再下一个；每步增量提交；能简不繁——先跑通 baseline-1（纯 BM25）再谈混合。
