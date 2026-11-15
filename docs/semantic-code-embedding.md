# 本地代码语义 embedding 与检索对比

## 实现

代码检索支持三种模式：

| 模式 | 行为 |
|---|---|
| `auto`（默认） | 使用本地已校验的代码模型；缺失、校验失败或初始化失败时警告并回退 hash，报告保留原因 |
| `semantic` | 严格使用代码模型；不可用即失败，评测不允许隐式降级 |
| `hash` | 使用原有 384 维词项哈希，不加载权重、不联网 |

语义模型为 [Jina 官方代码 embedding](https://huggingface.co/jinaai/jina-embeddings-v2-base-code)，
模型卡说明其使用代码与自然语言配对训练，支持英语及 C++ 等编程语言。这里不把它当作已经验证
的中文检索模型，也不宣称它是当前最优模型。本轮用它建立可复现的本地语义基线。

固定 revision `516f4baf13dec4ddddda8631e019b5737c8bc250`，使用官方 `onnx/model_quantized.onnx`
（约 162 MB）及 tokenizer。下载和加载均检查固定 SHA-256；只加载数据资产，不执行模型仓库的
Python 代码。沿用模型卡的 attention-mask mean pooling 和 L2 归一化，输出 768 维向量。
ONNX Runtime 使用 CPU，默认 4 线程、batch 4、512 token；超长片段会截断并记录数量。

模型与 tokenizer 哈希、revision、池化版本、维度、token 上限共同构成 embedding identity。
语义 collection 使用 identity 后缀，与旧 hash collection 分开；缓存版本也包含 identity。
更改模型配置必须为新空间建索引，不能复用旧向量。旧版默认 384 维 hash collection 仍可读取。
推理中途出错直接失败，不在已经开始的索引写入或查询中切换向量空间。

索引与查询不联网、不调用付费模型服务，源码仅在本地处理。模型下载是单独的显式命令。
Agent 按 workspace 和 embedding 配置缓存模型；下载/替换权重后应重启现有 Agent 进程。
外部 workspace 的路径、私有目录、大小限制及只读权限继续生效。
本轮只改变代码检索 encoder，独立的只读回答缓存仍沿用原实现。

## 使用

```bash
uv sync --frozen --extra rag --extra dev
uv run python scripts/download_code_embedding.py
uv run python scripts/index_codebase.py --embedding semantic --query "prevent integral windup"

# 外部仓库使用同一模型，但 collection 和缓存仍按 workspace 隔离
uv run python scripts/index_codebase.py --workspace examples/external_cpp --embedding semantic --query "joint error"

# 不加载语义模型的离线基线
uv run python scripts/index_codebase.py --embedding hash
```

可通过 `RAG_EMBEDDING`、`RAG_EMBEDDING_MODEL_DIR`、`RAG_EMBEDDING_MAX_TOKENS`、
`RAG_EMBEDDING_THREADS` 配置。CLI 的 `--embedding`、`--embedding-model-dir`、
`--embedding-max-tokens` 覆盖对应环境值。模型目录和 token 上限在建索引与 Agent 会话中应一致。
第一次切换到语义模型时，需要显式建立语义索引；工具不会误读原 hash 索引。

## 对比协议

```bash
uv run python scripts/run_rag_eval.py --compare \
  --persist-dir .agent_state/semantic-comparison/chroma \
  --output .agent_state/semantic-comparison/python --rebuild

uv run python scripts/run_rag_eval.py --compare \
  --workspace examples/external_cpp --artifact-root .agent_state/semantic-cpp \
  --manifest benchmarks/rag_cpp_smoke_cases.json \
  --output .agent_state/semantic-comparison/cpp --rebuild
```

五组为关键词、hash vector、hash hybrid、semantic vector、semantic hybrid。
关键词组跳过向量查询与 query embedding；hybrid 沿用既有 RRF 权重，不针对结果调参。
各组使用相同切块与标注，检索缓存关闭；运行器核对源码 revision，期间变化即拒绝比较。
语义组固定严格模式。记录模型/权重身份、查询集哈希、索引规模、各题名次、Hit/Recall/MRR、
查询延迟、模型加载与同步耗时、截断计数。未完成运行不会标成 `complete`。

Python 基准为原有 10 个英文查询，每题只有一个目标，因而 Hit@K 与 Recall@K 相同；
不是独立保留集。C++ 公开 smoke 只有 4 个文件和 4 条查询，Top-5 天然存在天花板，仅验证
外部仓库链路，不能据此宣称真实缺陷定位或修复成功率提高。延迟为单次 CPU 测量，包含源码
新鲜度检查及首轮初始化开销，不是稳定吞吐基准。索引耗时须结合 `chunks_updated` 判断冷/热同步。

## 本轮结果

2026-11-12，Windows / ONNX Runtime 1.26.0 / CPU 4 线程，默认 512 token。
原有 Python 基准共 10 条查询、545 个片段，源码 revision `42851ee509188036`。
[紧凑结果及逐题名次](experiments/rag-semantic-python-20261112.json)：

| 策略 | Hit@1 | Hit@3 | Hit@5 | MRR@5 | 平均查询 ms |
|---|---:|---:|---:|---:|---:|
| 关键词 | 70% | 90% | 100% | 0.8250 | 160.435 |
| Hash vector | 40% | 80% | 80% | 0.5500 | 52.421 |
| Hash hybrid | 60% | 90% | 100% | 0.7533 | 145.529 |
| Semantic vector | 60% | 90% | 100% | 0.7533 | 98.224 |
| Semantic hybrid | 80% | 100% | 100% | 0.8833 | 192.044 |

语义模型首轮同步更新 545 个片段耗时 143.270 秒，hash 为 0.733 秒；模型/客户端加载分别为
约 0.85 / 0.89 秒。语义索引有 62 个片段触及 512-token 截断。语义混合的 Top-1 比关键词
多命中 1 题，比旧 hash 混合多命中 2 题；关键词已经达到 Top-5 全命中，主要差别是排名。
语义向量也找回了 hash vector 在 Top-5 中遗漏的两题。语义混合延迟和首次建索引成本更高，
这组结果支持继续评测，不能从 10 条已知查询推导普遍提升或修复成功率收益。

外部公开 C++ smoke 共 4 条查询、4 个文件/片段。
[结果及逐题名次](experiments/rag-semantic-cpp-20261112.json)：

| 策略 | Hit@1 | Hit@5 | MRR@5 |
|---|---:|---:|---:|
| 关键词 | 50% | 75% | 0.6250 |
| Hash vector | 25% | 100% | 0.5417 |
| Hash hybrid | 50% | 100% | 0.6875 |
| Semantic vector | 100% | 100% | 1.0000 |
| Semantic hybrid | 50% | 100% | 0.7083 |

语义向量将自然语言描述的 stdin 驱动程序排到第一，但现有 RRF 融合受到关键词排名影响，
该题降到第三；角度差实现也由第一降到第二。保留这一负结果：混合检索并非总优于语义向量，
当前不据这个极小 smoke 调整融合权重。下一步应在独立 C++ 检索开发/保留集上验证路由、
融合与长片段策略。现有 M7.1 静态文档四组协议和结果没有改动。

原始分组报告（含检索片段）保存在 `.agent_state/semantic-comparison/`，模型资产也在忽略目录，
仅紧凑指标、名次与复现元数据入库；本轮无付费 API 调用。

## 工程验证

2026-11-13 增量：Agent 工具与持久会话已开放三种检索策略，新增目标文件互不重叠的
C++ 开发/公开保留集，真实模型五组对比及局限见 [检索策略与评测](rag-strategy-selection.md)。

2026-11-13 增量：文档索引已接入实际 tokenizer 预算切块和父片段上下文，见
[Token 切块协议与覆盖验证](rag-token-chunking.md)。上表及 62/545 截断统计为修改前的历史
结果；新语义 collection 使用独立版本，不沿用旧截断向量。源码快照标识与索引布局标识分开记录。

测试覆盖：显式 hash 不加载模型、缺失/损坏权重与严格模式、池化和 padding mask、归一化、
截断统计、关键词不执行向量查询、旧 hash collection 兼容、不同 encoder 空间隔离、
评测强制严格语义模式与源码变化拒绝，以及本机真实代码模型推理 smoke。
完整离线回归：**414 passed、1 skipped、46 deselected**，报告
`.agent_state/semantic-full.xml`。跳过 Windows 上的 POSIX 权限用例，排除需要真实模型 API、
C++/容器/仿真等独立环境的标记测试；本地 ONNX 真实模型 smoke 实际执行并通过。
全项目 Ruff、mypy（80 个源码文件）、`uv lock --check --offline` 和 diff whitespace 检查通过。
CI 离线任务补充安装 rag extra；CI 不隐式下载大模型，未预置权重时真实 encoder smoke 明确跳过。
