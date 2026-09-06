# maritime-rag-qa · 海事法规 / 航海通告 RAG 问答系统

> 项目一（主力）· 2026.10 → 2027.02 · 对标宁博 MaRAG（Ocean Engineering 344, 2026）思路的复现 + 改进
> 状态：🚧 脚手架阶段（2026-09-05 创建）

面向海事法规 / 航海通告的中文检索增强问答系统：**答案强制引用条款出处（溯源反幻觉）**，混合检索（向量 + BM25）+ 重排，并在自建海事 QA 评估集上用 InfoNCE 微调嵌入模型，与四级 baseline 做量化对比。

## 30 秒亮点

- **引用溯源**：每个答案附条款出处，幻觉可核查、可回溯
- **领域微调**：`bge-small-zh-v1.5` + MultipleNegativesRankingLoss（InfoNCE），微调前后 recall@k / MRR 对比
- **完整实验链**：分块 × top-k × 混合权重 × rerank 四级 baseline，全部可复现（单一真相表 `experiments/log.md`）

## 数据来源（全部公开，合规声明）

| 语料 | 来源 | 状态 |
|---|---|---|
| 《海上交通安全法》等国内法规 | 全国人大网 / 交通运输部公开文本 | TODO |
| 航海通告 | 中国海事局官网公开 PDF | TODO |
| SOLAS / MARPOL 参考资料 | IMO 公开摘要 | TODO |

> 合规红线：只用公开文档；`data/raw/` 不入 git（版权），仓库只放清洗脚本与样例。
> 逐条溯源记录见 [data/SOURCES.md](data/SOURCES.md)（五要素：来源/接口/参数/时间/字段）。

## 技术栈与选型理由（讲稿 §8–13' 素材，"为什么"列做实验时填）

| 组件 | 选型 | 为什么（TODO） |
|---|---|---|
| 框架 | LlamaIndex | |
| 嵌入 | BGE-M3（检索）/ bge-small-zh-v1.5（微调实验） | |
| 向量库 | Chroma | 嵌入式零运维，轻量外抛 |
| 混合检索 | BM25 (rank-bm25) + 向量，权重 α | 法规条文术语需精确匹配？TODO |
| 重排 | bge-reranker-v2-m3 | |
| 生成 | Qwen2.5-7B（API 优先，Ollama 量化兜底） | 为什么不全量微调 LLM：TODO |
| 前端 | Gradio | |

## 快速开始

```bash
pip install -r requirements.txt
python -m src.ingest.run          # TODO：PDF 清洗 → 条款感知分块 → 入库
python -m src.retrieval.eval      # TODO：四级 baseline + recall@k / MRR
python -m src.finetune.train_bge  # TODO：InfoNCE 微调 + 前后对比
python -m src.generation.app      # TODO：Gradio 问答（强制出处）
```

## 实验记录

所有数字的单一真相：[experiments/log.md](experiments/log.md)（检索网格 A / 嵌入微调 B / 引用约束 C / 踩坑 D）。

## 目录结构

```
maritime-rag-qa/
├── data/            # 语料落位 + SOURCES.md 溯源台账（raw 不入库）
├── eval/            # 80–100 条 QA 评估集（train/test 拆分，出处到条款）
├── src/
│   ├── ingest/      # PDF 清洗、条款感知分块
│   ├── retrieval/   # 混合检索 + 重排 + 离线评测
│   ├── generation/  # Qwen 生成 + 引用约束 + Gradio
│   └── finetune/    # bge-small-zh InfoNCE 微调
├── experiments/     # 实验记录（讲稿所有数字的唯一出处）
├── docs/            # 30 分钟讲稿骨架
└── requirements.txt
```

## 验收清单（= 30 分钟讲稿提纲）

- [ ] 数据来源与清洗可溯（`data/SOURCES.md` 全量五要素）
- [ ] 模型选型理由（本表"为什么"列全部填实）
- [ ] 损失函数：InfoNCE 微调实验 + 前后对比（`experiments/log.md` B）
- [ ] 调参过程：分块 × top-k × α × rerank 网格（A）
- [ ] baseline 对比：BM25 / 纯向量 / 混合 / 混合+rerank + 引用约束幻觉率（A、C）
- [ ] 30 分钟讲稿（`docs/讲稿骨架.md`）
- [ ] 演示 GIF / 录屏（无网可放）

## License

MIT（见 [LICENSE](LICENSE)）
