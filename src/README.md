# src/ 模块规划（对应讲稿 §13–25'，见 docs/复试30分钟讲稿_20260906.md）

| 模块 | 职责 | 状态 |
|---|---|---|
| `ingest/` | 法规 docx/epub/html 清洗 → **条款感知分块** → JSONL（章/节/条 + source_loc） | ✅ 5 部法规 641 条（海商法310/海交法122/内河条例95/船员条例66/航道法48），台账 data/SOURCES.md |
| `retrieval/` | BM25 + 向量(BGE-M3) + 混合(α) + rerank；离线评测 recall@k / MRR | ✅ 四级基线跑通，最优 h@α0.5+rerank：recall@1=.975/MRR=.975（experiments/log.md） |
| `generation/` | DeepSeek(硅基流动 V4-Flash) + 强制出处提示词 + [n] 引用 + 后校验 | ✅ qa.py 问答链路 / eval_generation.py G1G2（幻觉率 .30→0）/ Gradio UI src/app.py |
| `app.py` | Web 问答界面（演示交付形态） | ✅ `python src/app.py` → http://127.0.0.1:7860 |
| `finetune/` | bge-small-zh-v1.5 + MultipleNegativesRankingLoss 微调；前后对比 | 🚧 TODO（需先补标 train split） |

> 开发纪律（沿用个人工程原则）：一次只改一个模块，跑通再下一个；每步增量提交；能简不繁——先跑通 baseline-1（纯 BM25）再谈混合。
