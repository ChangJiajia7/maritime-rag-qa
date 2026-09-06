# maritime-rag-qa · 海事法规 RAG 问答系统

> 考研复试作品集项目一 · 对标宁博 MaRAG（Ocean Engineering 344, 2026）的复现 + 改进
> 状态：✅ **v0 全链路闭环**（2026-09-06）——语料 → 检索四级基线 → 生成引用约束 → Web 界面

面向海事法规的中文检索增强问答系统：**答案强制引用条款出处（溯源反幻觉）**。语料按条款级分块（非固定窗口），混合检索（自研 BM25 + BGE-M3 向量）可选重排，生成端只依据检索条文作答并带 `[n]` 出处标记；检索与生成均有自建评测集的量化结果，全部数字单一真相于 `experiments/log.md`。

## 现状快照（数字均来自真实实验）

| 环节 | 结果 |
|---|---|
| 语料 | 4 部现行海事法规 · **593 条款块**（条款级分块，source_loc 精确到 法/章/节/条） |
| 检索最优 | 混合 α=0.5 + rerank：**recall@1 = 0.975 / MRR = 0.975**（40 题 test） |
| 检索线上 | 混合 α=0.7（无 rerank）：recall@1 = 0.925 / MRR = 0.950 |
| 生成 | G2 RAG+引用约束：正确率 **1.000** / 幻觉率 **0.000**（对照 G1 无约束 0.725 / 0.300） |
| 界面 | Gradio Web 问答（右侧引用出处面板可核验） |

## 设计亮点（30 秒版）

1. **条款级分块**：每条法规按「第 X 条」边界切开（非 512 字固定窗口），每条块保留章/节归属与 `source_loc`——这是引用溯源与反幻觉的地基。
2. **引用约束生成**：System prompt 强制「只能依据参考条文 + 句末 [n] 标注 + 未规定须明说」，幻觉率实验归零（见实验 C）。
3. **轻量自研而非框架堆叠**：不用 LlamaIndex / Chroma / rank-bm25 包——检索管道全部自研（jieba + Okapi BM25、条款状态机分块、urllib 直连 API）。593 条款的小语料不需要向量库抽象，且每一行打分逻辑可向导师讲透。
4. **评测闭环**：40 题固定考卷（每题标 golden 条款）→ 换方法重测同一套题 → recall@k / MRR / 幻觉率全部可比。
5. **全 API 无本地模型**：嵌入/重排/生成全走硅基流动，向量缓存可重建，`.env` 三件套独立配置。

## 架构

```
4 部现行法规（docx/epub 官方文本）
        │ 清洗：去网页注脚 URL / 版权横幅 / 空段
        ▼
   条款感知分块（章/节/条状态机）──► data/*.jsonl（593 条款块，含 source_loc）
        │                              ▲
        │                              │ 引用溯源
        ▼                              │
   评估集 40 条 test QA（golden 条款）──┤
        │                              │
        ▼                              │
   检索评测 recall@1/3/5 · MRR ────────┤
    ├─ BM25（自研 jieba+Okapi）
    ├─ 向量（BGE-M3 API，1024d，本地缓存）
    ├─ 混合（α 加权）→ 最优 α=0.5
    └─ + rerank（bge-reranker-v2-m3）→ MRR .975
        │
        ▼
   生成：检索 top5 → 引用编号条文块 → DeepSeek-V4-Flash（硅基流动）
        │       强制 [n] 出处 + 禁编造
        ▼
   Gradio Web UI（右侧引用出处面板，出处可核验）
```

## 技术栈（真实实现）

| 层 | 实现 | 备注 |
|---|---|---|
| 语料解析 | 标准库 zipfile 解 docx/epub | 无 python-docx 依赖 |
| 分块 | 自研条款感知（章/节/条 + 中文数字转换） | `src/ingest/chunk_law.py` |
| BM25 | 自研 Okapi BM25（k1=1.5, b=0.75）+ jieba | `src/retrieval/bm25.py` |
| 向量 | SiliconFlow `BAAI/bge-m3`（1024d） | `src/retrieval/dense.py`，缓存 `data/cache/` |
| 重排 | SiliconFlow `BAAI/bge-reranker-v2-m3` | `src/retrieval/rerank.py` |
| 生成 | SiliconFlow `deepseek-ai/DeepSeek-V4-Flash` | OpenAI 兼容 `/v1/chat/completions` |
| 评测 | 自研 recall@k/MRR + LLM judge 幻觉率 | `src/retrieval/eval_retrieval.py` / `src/generation/eval_generation.py` |
| 界面 | Gradio 6.x | `src/app.py` |

## 数据与合规

- 语料溯源台账 [data/SOURCES.md](data/SOURCES.md)：每条记录五要素（来源 URL/查询参数/取数时间/落位/版本备注），`data/raw/` 不入 git。
- **只引现行有效版本**（版本陷阱已规避并记录）：海商法 2025 修订(310 条，2026-05-01 施行) / 海交法 2021(122) / 内河条例 2019 第三次修订(95) / 船员条例 2023 第七次修订(66)。
- `.env`（真实 key）不入库；`.env.example` 为占位符，可安全推 GitHub。

## 评测结果速查（完整见 [experiments/log.md](experiments/log.md)）

**A. 检索网格（40 题 test，条款级分块）**

| 基线 | recall@1 | recall@3 | recall@5 | MRR@5 |
|---|---|---|---|---|
| 1 纯 BM25 | 0.775 | 0.900 | 1.000 | 0.859 |
| 2 纯向量 | 0.875 | 0.925 | 0.925 | 0.900 |
| 3 混合 α=0.5 | 0.900 | 0.975 | 0.975 | 0.938 |
| 4 混合 + rerank | **0.975** | 0.975 | 0.975 | **0.975** |

α 扫描（无 rerank）：0.3→.890 / 0.5→.938 / **0.7→.950** / 0.9→.938。

**C. 生成（40 题 test，judge 口径 v2）**

| 组 | 链路 | 正确率 | 幻觉率 | 引用命中 |
|---|---|---|---|---|
| G1 | 纯 LLM 知识直答 | 0.725 | 0.300 | — |
| G2 | RAG + 强制出处 + 后校验 | **1.000** | **0.000** | 0.975 |

**D. 踩坑记录 4 条**（讲稿素材）：条款归属张冠李戴（43 vs 44 条）｜纯向量列举式盲区 → 混合必要性｜judge 口径把「有据补充」误判幻觉 → v2 修正｜028「罚则 vs 行为条款」检索极限与引用兜底实证。

## 快速开始

```bash
# 0. 环境（Windows）
py -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 1. 配置三件套 API key（LLM/EMBED/RERANK 各填一套；嵌入与重排用硅基流动）
cp .env.example .env
python src/api_smoke_test.py                    # 三件套连通自检 3/3

# 2. 数据（已入库可直接用；重建走 src/ingest/*.py）
ls data/*.jsonl                                 # 4 部法规 593 条款块

# 3. 检索四级基线评测（每次换 --method / --alpha / --rerank）
python src/retrieval/eval_retrieval.py --method bm25
python src/retrieval/eval_retrieval.py --method dense             # 首次嵌入较慢，之后走缓存
python src/retrieval/eval_retrieval.py --method hybrid --alpha 0.5
python src/retrieval/eval_retrieval.py --method hybrid --alpha 0.5 --rerank

# 4. 生成侧 G1/G2 对照（真实调用 LLM，约 4 分钟）
python src/generation/eval_generation.py

# 5. CLI 单问 / Web 界面
python src/generation/qa.py "客船能否同时载运乘客和危险货物？"
python src/app.py                                # → http://127.0.0.1:7860
```

## 目录结构

```
maritime-rag-qa/
├── data/            # 4 部法规 JSONL（条款块）+ SOURCES.md 溯源台账（raw/、cache/ 不入库）
├── eval/            # 评估集 qa_eval_set.csv（40 条 test，golden 条款机器可查）
├── src/
│   ├── ingest/      # docx/epub → txt → 条款感知 JSONL
│   ├── retrieval/   # bm25 / dense / rerank / retriever / eval_retrieval
│   ├── generation/  # qa.py（引用约束问答）/ eval_generation.py（G1G2）
│   └── app.py       # Gradio Web 问答界面
├── experiments/     # 实验单一真相 log.md + g_outputs.json（G 原始输出可复核）
└── .env.example     # 占位符模板（无真实 key）
```

## 已知局限与下一步

- **028 盲区**：检索无法区分「罚则条款 vs 行为条款」答非所问（题面与罚则用词重叠）；生成侧引用约束下仍能基于罚则给出正确事实——已作检索极限记录，下一步可做检索→生成联动校验。
- **范围**：现覆盖 4 部国内现行法规；航海通告、SOLAS/MARPOL、裁判文书等语料未纳入（MaRAG 对标的扩展方向）。
- **微调（experiments/log.md B 区 TODO）**：全 API 架构 + 检索已达 MRR .975，嵌入微调收益天花板明显；需先补标 train split，作为方法论扩展而非线上提升手段。
- **模型口径**：G 系列数字用 DeepSeek 官方 `deepseek-chat` 得出（`.env.bak-deepseek-20260906` 可回退）；当前默认硅基流动 `deepseek-ai/DeepSeek-V4-Flash`，重跑 `eval_generation.py` 即可复现新口径。
- **评测集**：test 40 条已冻结；train split 待标（微调前完成）。

## License

MIT（见 [LICENSE](LICENSE)）
