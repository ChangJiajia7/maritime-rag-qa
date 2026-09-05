# maritime-rag-qa · 海事法规与航海通告 RAG 问答系统

> 考研复试项目一（主力）｜周期 2026.10 → 2027.02 完成并开源
> 对标：宁博《MaRAG: a knowledge graph and retrieval-augmented framework for maritime accident analysis》（Ocean Engineering 344, 2026）——本项目在海事 RAG 思路上做复现与改进，后期加入海事实体消歧（知识图谱）扩展。

## 项目简介

面向海事法规（《海上交通安全法》等）、航海通告等**公开文档**的中文 RAG 问答系统：

- 混合检索（向量 + 关键词 BM25）+ 重排
- 答案强制引用溯源（条款出处），抑制幻觉
- 自建中文海事 QA 评估集，量化评测 recall@k / MRR
- 嵌入模型领域微调（InfoNCE 对比学习），报告微调前后对比

## 目录结构

```
maritime-rag-qa/
├── data/
│   ├── raw/            # 原始公开文档（PDF），体积大不入库（见 .gitignore）
│   ├── processed/      # 清洗后的结构化文本（每条一行）
│   └── SOURCES.md      # 数据来源台账（每份文档一行，合规红线）
├── scripts/
│   ├── clean_pdf.py    # 阶段0：PDF 清洗（4 个 TODO，自己实现）
│   └── requirements.txt
├── eval/
│   └── qa_eval_set.csv # QA 评估集（id/question/answer/source_doc/source_page/split）
├── app/                # 阶段2：Gradio 问答前端
├── notebooks/          # 实验笔记本
├── results/
│   └── EXPERIMENTS.md  # 实验记录（调参过程留痕）
├── docs/
│   └── 讲稿-30min-骨架.md
└── README.md
```

## 路线图

- [ ] 阶段0（W1-W3）：环境与工具，PDF 清洗脚本
- [ ] 阶段1（W4-W7）：数据采集与清洗（`data/` + `SOURCES.md` 全量溯源）
- [ ] 阶段2（W8-W15）：RAG 管道 MVP（混合检索 + 引用溯源 + Gradio）
- [ ] 阶段3（W18-W21）：嵌入微调 + recall@k/MRR 对比表
- [ ] 阶段4（W23-W24）：README/实验表/讲稿定稿，正式开源
- [ ] 可选 stretch：海事实体消歧（知识图谱，时间盒 3h，超时砍）

## 数据来源声明（合规红线）

只使用**公开**海事文档，逐条登记在 [data/SOURCES.md](data/SOURCES.md)；不爬取有版权、需登录或内部接口的数据；开源前清理敏感信息。

## 快速开始

```bash
# 1. 克隆后创建虚拟环境（Python ≥ 3.10）
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 2. 安装依赖
pip install -r scripts/requirements.txt

# 3. 按 data/SOURCES.md 收集公开文档，放入 data/raw/

# 4. 跑通清洗脚本（阶段0 判据）
python scripts/clean_pdf.py
```

## 实验记录

所有调参过程登记在 [results/EXPERIMENTS.md](results/EXPERIMENTS.md)——讲稿「调参过程」环节直接引用本表。

## License

[MIT](LICENSE)
