# maritime-rag-qa · 海事法规 RAG 问答系统

> 考研复试作品集项目一 · 对标宁博 MaRAG（Ocean Engineering 344, 2026）的复现 + 改进
> 状态：✅ **v0 全链路闭环**（2026-09-06）——语料 → 检索四级基线 → 生成引用约束 → Web 界面

面向海事法规的中文检索增强问答系统：**答案强制引用条款出处（溯源反幻觉）**。语料按条款级分块（非固定窗口），混合检索（自研 BM25 + BGE-M3 向量）可选重排，生成端只依据检索条文作答并带 `[n]` 出处标记；检索与生成均有自建评测集的量化结果，全部数字单一真相于 `experiments/log.md`。

## 现状快照（数字均来自真实实验）

| 环节 | 结果 |
|---|---|
| 语料 | 5 部现行海事法规 · **641 条款块**（条款级分块，source_loc 精确到 法/章/节/条） |
| **结构注入** | 索引文本注入「法规 > 章 > 第X条」归属：纯向量 recall@5 **0.925 → 1.000**、recall@1 **0.875 → 0.950**，内河条例列举式盲区**整类消除**（3 道漏召题同升 rank 1，其余 35 题排名不动；log A2 节） |
| 检索最优 | 混合 **α=0.9**（无 rerank）：recall@1 = 0.975 / recall@5 = **1.000** / MRR = **0.988**（40 题 test） |
| 检索线上 | 混合 α=0.7（无 rerank）：recall@1 = 0.950 / recall@5 = 1.000 / MRR = 0.975 |
| 重排（rerank） | 注入**之前**有效（recall@1 0.875→0.975）；注入**之后不再受益**（recall@5 反降至 0.975，延迟 17.9s vs 0.8s）→ **已从线上链路下线** |
| 生成 | G2 RAG+引用约束：正确率 **1.000** / 幻觉率 **0.000**（对照 G1 无约束 0.725 / 0.300） |
| 界面 | Gradio Web 问答（多轮追问：保留最近 2 轮上下文 + 短追问自动拼上轮问题再检索；右侧引用出处面板可核验） |

## 设计亮点（30 秒版）

1. **条款级分块**：每条法规按「第 X 条」边界切开（非 512 字固定窗口），每条块保留章/节归属与 `source_loc`——这是引用溯源与反幻觉的地基。
2. **章节结构注入**（本项目最有价值的一次改动）：法条脱离章节就成了孤儿——罚则条款只写"违反本条例的规定……处以罚款"，看不到它对应的行为条款。把「法规 > 章 > 第X条」作为面包屑**注入索引文本**（`src/retrieval/bm25.py:index_text()`，仅用于建索引，展示与引用仍用原文）后，纯向量 recall@5 从 0.925 升到 **1.000**，3 道内河条例列举式漏召题**同升 rank 1**，其余 35 题排名不动（整类消除，非单题过拟合）。对照实验还发现：只注入章名、不注入法规名，效果与完全不注入一致——**法规名才是决定性成分**。详见 log.md **A2 节**。
3. **引用约束生成**：System prompt 强制「只能依据参考条文 + 句末 [n] 标注 + 未规定须明说」，幻觉率实验归零（见实验 C）。
4. **轻量自研而非框架堆叠**：不用 LlamaIndex / Chroma / rank-bm25 包——检索管道全部自研（jieba + Okapi BM25、条款状态机分块、urllib 直连 API）。641 条款的小语料不需要向量库抽象，且每一行打分逻辑可向导师讲透。
5. **评测闭环**：40 题固定考卷（每题标 golden 条款）→ 换方法重测同一套题 → recall@k / MRR / 幻觉率全部可比。
6. **全 API 无本地模型**：嵌入/重排/生成全走硅基流动，向量缓存可重建，`.env` 三件套独立配置。
7. **多轮追问不丢主题**：短问题（<12 字，如"包括哪些内容？"）自动拼接上一轮问题再检索，防指代漂移；保留最近 2 轮对话作 LLM 上下文（commit `9271f4f`，实测见 log.md D 表末条）。
8. **查询与重排结果持久化缓存**：同一 query 的嵌入、同一 (问题, 候选集) 的重排得分落盘复用，重复评测零 API 调用（α 网格扫描从「每次重调」降到秒级：rerank 版 18.7s → 0.8s）。

## 架构

```
5 部现行法规（docx/epub/政府门户官方文本）
        │ 清洗：去网页注脚 URL / 版权横幅 / 空段
        ▼
   条款感知分块（章/节/条状态机）──► data/*.jsonl（641 条款块，含 source_loc）
        │                              ▲
        │ 建索引时注入面包屑             │ 引用溯源（展示仍用原文）
        │ 【法规 > 章 > 第X条】+ 正文    │
        ▼                              │
   评估集 40 条 test QA（golden 条款）──┤
        │                              │
        ▼                              │
   检索评测 recall@1/3/5 · MRR ────────┤
    ├─ BM25（自研 jieba+Okapi）
    ├─ 向量（BGE-M3 API，1024d，本地缓存）
    ├─ 混合（min-max 归一化后 α 加权）→ 最优 α=0.9，MRR .988
    └─ + rerank（bge-reranker-v2-m3）→ 注入前有效；注入后不再受益，已下线
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
| HTTP 调用 | 统一封装：指数退避 + 抖动 + `Retry-After`；4xx 仅重试 408/429 | `src/net.py` |
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

**A. 检索网格（40 题 test，641 条款块）**

注入**前**（`INDEX_BREADCRUMB=0`，log A6–A12）：

| 基线 | recall@1 | recall@3 | recall@5 | MRR@5 | top5 未中 |
|---|---|---|---|---|---|
| 1 纯 BM25 | 0.750 | 0.900 | 0.975 | 0.841 | 028 |
| 2 纯向量 | 0.875 | 0.925 | 0.925 | 0.900 | 025 / 028 / 030 |
| 3 混合 α=0.5 | 0.875 | 0.975 | 0.975 | 0.925 | 028 |
| 4 混合 α=0.5 + rerank | 0.975 | 0.975 | 0.975 | 0.975 | 028 |

注入**后**（`INDEX_BREADCRUMB=1`，log A13–A18）——**当前线上配置**：

| 基线 | recall@1 | recall@3 | recall@5 | MRR@5 | top5 未中 |
|---|---|---|---|---|---|
| 1 纯 BM25 | 0.775 | 0.875 | 0.975 | 0.847 | 028 |
| 2 纯向量 | 0.950 | **1.000** | **1.000** | 0.975 | — |
| 3 混合 α=0.7 | 0.950 | 1.000 | 1.000 | 0.975 | — |
| 4 混合 **α=0.9** | **0.975** | 1.000 | 1.000 | **0.988** | — |
| 5 混合 α=0.5 + rerank | 0.975 | 0.975 | 0.975 | 0.975 | 028 |

α 扫描（注入后）：0.3→.906 / 0.5→.944 / 0.7→.975 / **0.9→.988**。
（注入前同一次扫描：0.3→.877 / 0.5→.925 / 0.7→.950 / 0.9→.938——**参数没变，索引变了，最优点从 0.7 移到 0.9**。）

> **口径提示**：log.md 的 A1（BM25）与 A3（α 扫描）记录于 **593 块**语料（4 部法规，扩《航道法》前），
> 上表为 **641 块**口径复测；A2 / A4 未标块数，但复测与 641 块逐位一致。引用 A1 / A3 时须注明口径。

**C. 生成（40 题 test，judge 口径 v2）**

| 组 | 链路 | 正确率 | 幻觉率 | 引用命中 |
|---|---|---|---|---|
| G1 | 纯 LLM 知识直答 | 0.725 | 0.300 | — |
| G2 | RAG + 强制出处 + 后校验 | **1.000** | **0.000** | 0.975 |

**D. 踩坑记录 5 条**（讲稿素材）：条款归属张冠李戴（43 vs 44 条）｜纯向量列举式盲区 → 混合必要性｜judge 口径把「有据补充」误判幻觉 → v2 修正｜028「罚则 vs 行为条款」——**查明根因是索引丢了章节归属，注入后整类修复**｜多轮短追问指代漂移 → 拼接上轮问题。

## 快速开始

```bash
# 0. 环境（Windows）
# 日常运行：轻量环境（gradio + jieba + numpy，与 src/ 实际 import 一致）
py -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 重型环境（已隔离到 .venv-heavy）：torch/pandas/pypdf 等旧体积依赖仅在
# 微调嵌入模型、原型探索时启用——.venv-heavy\Scripts\activate 后手动装对应包。

# 1. 配置三件套 API key（LLM/EMBED/RERANK 各填一套；嵌入与重排用硅基流动）
cp .env.example .env
python src/api_smoke_test.py                    # 三件套连通自检 3/3

# 2. 数据（已入库可直接用；重建走 src/ingest/*.py）
ls data/*.jsonl                                 # 5 部法规 641 条款块

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
├── data/            # 5 部法规 JSONL（条款块）+ SOURCES.md 溯源台账（raw/、cache/ 不入库）
├── eval/            # 评估集 qa_eval_set.csv（40 条 test，golden 条款机器可查）
├── src/
│   ├── ingest/      # docx/epub → txt → 条款感知 JSONL
│   ├── retrieval/   # bm25 / dense / rerank / retriever / eval_retrieval
│   ├── generation/  # qa.py（引用约束问答）/ eval_generation.py（G1G2）
│   ├── net.py       # HTTP 调用统一封装（指数退避重试）
│   └── app.py       # Gradio Web 问答界面
├── experiments/     # 实验单一真相 log.md + g_outputs.json（G 原始输出可复核）
└── .env.example     # 占位符模板（无真实 key）
```

## 已知局限与下一步

- **028 已修复（原「检索极限」）**：曾表现为「罚则条款 vs 行为条款」无法区分。查明根因是**索引丢了章节归属**——行为条款第 27 条脱离「第三章 航行」后失去主题锚点，被它的处罚版（第 74 条，同词复述）和句式更像的《航道法》第 35 条一起挤出前五。注入面包屑后第 27 条升至 **rank 1**，3 道同因漏召题整类修复（log A2 节）。
- **超参在同一套 test 上选定**（已知方法学软肋，答辩须主动说明）：α 网格扫描与「是否注入章节结构」的比对都在 40 题 test 上完成，严格说属数据污染。影响可控的依据：α 在 0.5 / 0.7 / 0.9 上是 0.944 / 0.975 / 0.988，**趋势单调、没有反转**，且有原理（纯向量召回已满分，权重越高越好）；结构注入带来的是**整类提升**（3 题同因同解、35 题排名完全不动），不是单题过拟合。规范做法是扩题后重划 train/val/test、调参只在 val 上做。
- **评测集题型单一**：现有 40 题**全为单跳事实题**（条款查找 19 / 数字定位 10 / 多条件交叉 7 / 术语消歧 3 / 罚则匹配 1），**跨条多跳 0 题、OOD 拒答 0 题**。下一步补 A 表跨条多跳（golden 用条款集合 + set-recall）与 G 表 C3 越界拒答子集。
- **范围**：现覆盖 5 部国内现行法规；航海通告、SOLAS/MARPOL、裁判文书等语料未纳入（MaRAG 对标的扩展方向）。
- **微调（experiments/log.md B 区 TODO）**：检索侧已达 recall@5 全中、MRR .988，嵌入微调收益天花板明显；且需先补标 train split——作为方法论扩展而非线上提升手段，优先级低于扩题与评测规范化。
- **模型口径**：G 系列数字用 DeepSeek 官方 `deepseek-chat` 得出（`.env.bak-deepseek-20260906` 可回退）；当前默认硅基流动 `deepseek-ai/DeepSeek-V4-Flash`，重跑 `eval_generation.py` 即可复现新口径。
- **评测集**：test 40 条已冻结（只追不加改）；train split 待标（微调前完成）。
- **样本量口径**：G 系列（正确率 1.000 / 幻觉率 0.000）与检索数字均基于 40 题 × 641 条款的小样本，结果偏乐观；扩语料或换题面后指标可能回落，数字解读须带此口径（复试答辩建议主动说明）。

## License

MIT（见 [LICENSE](LICENSE)）
