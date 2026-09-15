# FinAgent：金融多源问答 Agent

FinAgent 是一个面向基金结构化数据、公司披露文档和公开网页信息的本地 Agent 原型。它将问题拆分为原子需求，按需调用 SQL、RAG、Web 或本地计算，并保留来源、页码、任务状态和运行 Trace，便于人工核查。

> 项目用于本地研究与演示，不构成投资建议、审计意见或生产级金融服务。所有数值结论都应回到原始数据、披露文件或权威来源复核。

## Public Repository

This repository contains the core implementation of FinAgent, including:

- multi-source Agent orchestration
- RAG retrieval and reranking
- Text-to-SQL pipeline
- Tool Calling
- context/session management
- tracing and evaluation

Large financial datasets, original documents, generated embeddings,
FAISS indexes and runtime traces are intentionally excluded.

The public test suite contains deterministic/mock-based tests and currently passes:

```text
444 passed
```

Full financial SQL / RAG / Web / LLM execution requires users to
provide their own data assets and API configuration.

## 能力概览

- **任务级编排**：结合时间语义、来源限制和工具必要性，将复合问题拆为可独立执行的需求；依赖任务按拓扑顺序传递结果与证据。
- **金融 RAG**：PDF 处理、文档范围隔离、Dense/BM25 双路召回、RRF 融合、重排，以及文档/页码/chunk 级证据追踪。
- **Text-to-SQL**：基金实体对齐、常见金融指标语义规划、只读 SQLite 执行、超时和返回行数限制。
- **网页研究**：对需要时效性的信息进行搜索与抓取，保存 URL、抓取时间、来源层级和疑似提示注入标记。
- **确定性金额计算**：金额证据结构化为 `currency`、`display_unit`、`multiplier`、`base_value`；元/万元/亿元先统一倍率，人民币与美元缺少汇率及日期时拒绝混算。加减乘除、比值和增长率由本地 `Decimal` 工具执行，而非由模型输出结果。
- **可观测性与会话**：FastAPI/SSE 接口、可选 Redis 会话状态、Trace、运行摘要和本地调试页面。

## 架构

```text
用户问题
  │
  ├─ 需求拆解 / 时间解析 / 来源规划
  │       │
  │       ├─ SQL：结构化基金数据 ─┐
  │       ├─ RAG：披露文档与页码 ─┼─ 统一 Evidence
  │       ├─ Web：公开时效信息 ──┤
  │       └─ Local：确定性计算 ──┘
  │
  └─ 引用、覆盖与金额语义校验 → 最终回答 + Trace
```

## 快速开始

### 1. 环境

项目使用 Conda 环境 `finagent`。以下命令均在项目根目录执行：

```powershell
conda create -n finagent python=3.11
conda run -n finagent python -m pip install -r requirements.txt
```

> 如果你已具备项目环境，不需要重新创建。项目中的 Python 命令应始终通过 `conda run -n finagent` 执行。

### 2. 配置外部服务

将 [`.env.example`](.env.example) 复制为 `.env`，再填写实际值。`.env` 已被 Git 忽略，切勿提交密钥。

```powershell
Copy-Item .env.example .env
```

完整执行需要兼容的模型服务、相应认证配置和可访问的模型名。文档向量化、重排和网页搜索也需要用户自行配置对应服务；Redis 未启动时，多轮会话能力会降级。

### 3. 检查本地资产

完整金融数据库因体积及数据授权原因不随公开仓库发布。

RAG 索引为运行时生成资产，不随公开仓库发布。文档目录元数据保留在 `data/catalog/`；如需完整 SQL 或 RAG 功能，请提供自有授权数据，并按“重建文档索引”执行分阶段脚本。不要把未授权的报告、认证信息或含敏感信息的 Trace 上传到公开仓库。

### 4. 启动服务

```powershell
conda run -n finagent python -m uvicorn agent:app --host 127.0.0.1 --port 8000
```

服务启动后，向 `POST /` 发送请求。接口以 SSE 返回 `Message` 与 `Ping` 事件：

```powershell
Invoke-WebRequest -Method Post http://127.0.0.1:8000/ `
  -ContentType 'application/json' `
  -Body '{"question":"只用本地数据库，查询基金代码 000001 的单位净值。","session_id":"demo-001"}'
```

可选：设置 `DEBUG_UI_ENABLED=1`、`TRACE_ENABLED=1` 后访问 `http://127.0.0.1:8000/ui` 查看本地调试页面。该页面没有鉴权，仅限本机可信环境使用。

## 可演示的问题

| 链路 | 示例问题 | 核查点 |
|---|---|---|
| SQL | `只用本地数据库，查询基金代码 000001 在指定日期的资产净值。` | 日期、字段、原始 rows |
| RAG | `根据健帆招股书说明关键材料供应商与供货风险。` | 文档范围、页码、引用 |
| 多轮 | 先询问一家公司的业务，再问“它的供应风险呢？” | 指代与会话上下文 |
| 金额计算 | `将两条已查询金额相减，按万元展示。` | 币种、倍率、公式、输入证据 |
| 失败处理 | `只用本地数据库查询不存在日期的数据。` | 不编造，不越权回退 |

## 确定性金额计算

对金额计算类问题，模型最多提交受限计算计划（引用哪条证据、选择何种运算、展示单位）；用户可见的计算结果只由本地计算器使用 Python `Decimal` 生成：

```text
1 亿元人民币 − 2500 万元人民币
= 100,000,000 − 25,000,000
= 7500 万元人民币
```

若混合人民币和美元而缺少汇率及汇率日期，系统会明确报告无法执行，而不会猜测汇率或输出伪精确数字。

## 重建文档索引

多文档索引分为三个显式阶段；请使用自己的授权 PDF 和 API 密钥。

```powershell
# A：单份 PDF → chunks JSONL
conda run -n finagent python scripts/stage_a_pdf_to_chunks.py --input <report.pdf> --output <chunks.jsonl>

# B：chunks → 可复用 embedding artifact
conda run -n finagent python scripts/stage_b_chunks_to_embeddings.py --input <chunks.jsonl> --output-dir <embedding-dir>

# C：基础索引 + embedding artifacts → 多文档 FAISS 索引
conda run -n finagent python scripts/stage_c_build_multi_index.py --artifact <embedding-dir> --output-dir outputs/index/multi_doc_v1
```

## 测试

```powershell
# 金额语义与确定性计算
conda run -n finagent python -m pytest tests/test_financial_amounts.py tests/test_financial_calculator.py -q

# 全量回归（可能需要本地数据库、索引及相关服务）
conda run -n finagent python -m pytest tests -q
```

## 目录说明

```text
agent.py / agent_loop.py       FastAPI、SSE 与应用入口
orchestration/                 需求规划、执行、证据、会话、Trace、确定性计算
text2sql/                      SQL 语义规划、实体对齐与校验
rag/                           文档处理、向量索引、BM25、RRF、重排
tools/                         SQL 与文档检索工具封装
scripts/                       索引构建与本地检查脚本
tests/                         单元与集成测试
docs/                          架构审阅与人工评估文档
```

## 评估与边界

- 历史检索实验用于定位召回与排序问题，不能等同于端到端答案正确率。
- 完整答案需人工核对主体、日期、指标口径、单位、来源与计算过程；建议依照 [人工评估手册](docs/MANUAL_EVALUATION.md) 记录。
- 当前面向本机演示。SSE 在完整答案产生后再输出分块；取消传播、跨进程会话一致性、公开部署鉴权与抓取安全边界仍需在上线前完善。
- 不要将项目描述为多 Agent、生产级高并发服务、自动投资决策系统或已完成大规模人工验收。

## 相关文档

- [架构审阅与改进清单](docs/ARCHITECTURE_REVIEW.md)
- [人工评估手册](docs/MANUAL_EVALUATION.md)

## License

当前仓库尚未声明开源许可证。公开发布前请补充许可证，并确认数据集、PDF、模型服务及第三方依赖的授权范围。
