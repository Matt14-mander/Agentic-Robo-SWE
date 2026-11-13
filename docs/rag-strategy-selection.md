# Agent 检索策略与独立 C++ 评测

## 配置和选择

`search_code_knowledge` 新增可选 `strategy`，Agent 可在一次调用中选择：

| 值 | 行为 | 适合的查询 |
|---|---|---|
| `lexical` | 关键词检索，不进行查询向量推理 | 标识符、已有代码术语 |
| `vector` | 使用当前 embedding 后端进行向量检索 | 行为描述、语义近似表达 |
| `hybrid` | 融合关键词与向量排名 | 同时有术语和概念证据 |

优先级是：单次调用 `strategy` > 已保存的会话 `rag_strategy` > 环境变量 `RAG_STRATEGY`
> `hybrid`。模型 schema 只接受这三个值，内部 RunnableConfig 不暴露给模型。未提供新参数
的旧调用继续工作。命中结果的标题标记实际使用的策略；原有路径过滤、top_k、workspace 隔离和
只读工具权限保持有效，结果缓存按策略分别保存。

```bash
# 固定会话默认值，Agent 仍可在单次调用中覆盖。
uv run python scripts/run_agent.py "定位传感器数据处理逻辑" \
  --workspace /path/to/repo --artifact-root /path/to/state \
  --read-only --rag-strategy lexical --thread-id inspect-sensors

# 恢复时省略策略参数，会沿用 checkpoint；显式提供可更新默认值。
uv run python scripts/run_agent.py --thread-id inspect-sensors --continue \
  --workspace /path/to/repo --artifact-root /path/to/state --rag-strategy vector
```

`.env` 可设置 `RAG_STRATEGY=hybrid`，用于新会话默认值。CLI 将选择保存到 AgentState；
直接使用图时可在初始 state 中设置 `rag_strategy`。旧 state 没有该字段时，planner 在
首次执行时解析环境默认值并保存。`--status` 显示已保存的选择。恢复会话不会因为环境变量
变化而自动改换策略；`--resume approve|deny` 同样支持显式更新，并保留原审批流程。

planner 提示词说明如何按证据选择策略，不会自动执行三次查询。默认值进入提示词指纹，
因此只读语义响应缓存也区分不同会话策略。ToolNode 使用调用局部配置传递默认值，避免
不同会话通过修改全局环境变量互相影响；工具结果与调用参数进入既有修复轨迹。

`vector` 不等同于必然启用真实语义模型：后端仍由 `RAG_EMBEDDING` 决定，`auto` 可以退回
hash。需要严格语义检索时设置 `RAG_EMBEDDING=semantic` 并准备固定模型资产。`lexical`
跳过查询 embedding 推理，但打开配置好的索引仍需相应后端依赖和资产。

## 独立 C++ 评测协议

新增 [独立 C++ v1 数据集](../benchmarks/retrieval_cpp/README.md)：20 个头文件，开发集和
公开保留集各 16 题，各含 8 个标识符查询和 8 个行为描述查询。两组目标文件互不重叠，
在同一个完整 corpus 中检索；包括角度直接相减/周期归一化、拥有/借用资源、加和校验/CRC、
阻塞队列/原子环形队列等相近实现。查询和标签存放在 corpus 外，禁止被索引。

语料和查询在首轮运行前固定，开发集结果未用于调整权重或默认策略；随后运行保留集。
五组分别是关键词、hash vector、hash hybrid、semantic vector、semantic hybrid，沿用
已有排名算法和 Top-1/3/5 指标，关闭检索缓存。严格语义组失败时报告失败，不冒充 hash
结果。记录模型 identity、源码 revision、逐文件/manifest 指纹、逐题排名、索引成本和
平均查询延迟，评测过程中输入变化会使报告标为 incomplete。

```bash
uv run python scripts/run_cpp_retrieval_eval.py --split dev --hash-only
uv run python scripts/run_cpp_retrieval_eval.py --split dev --output .agent_state/cpp-retrieval/dev
uv run python scripts/run_cpp_retrieval_eval.py --split heldout --output .agent_state/cpp-retrieval/heldout
```

## 2026-11-13 结果

固定源码 revision：`5277449d95821e6e`。本机实际执行了固定 Jina ONNX 模型，两组均为
20 个片段，未触发输入截断；没有调用付费模型 API。

| 检索组 | 开发集 Hit@1 | 保留集 Hit@1 | 保留集 Hit@5 | 保留集 MRR@5 | 保留集平均查询 ms |
|---|---:|---:|---:|---:|---:|
| 关键词 | 75.00% | 68.75% | 81.25% | 0.7344 | 14.457 |
| Hash vector | 56.25% | 37.50% | 68.75% | 0.4760 | 14.616 |
| Hash hybrid | 75.00% | 62.50% | 87.50% | 0.7312 | 19.703 |
| Semantic vector | 100.00% | 87.50% | 100.00% | 0.9375 | 34.888 |
| Semantic hybrid | 100.00% | 87.50% | 100.00% | 0.9375 | 40.026 |

[开发集报告](experiments/rag-cpp-independent-dev-20261113.json)和
[公开保留集报告](experiments/rag-cpp-independent-heldout-20261113.json)保存全部指纹与逐题排名。
原始候选片段和独立 Chroma 库保存在 `.agent_state/cpp-retrieval-20261113/`。

保留集的 8 个标识符查询，关键词全部 Top-1 命中；概念查询则只有 3/8 Top-1。中文概念查询
缺乏与英文代码共用的词项，是关键词漏召回的重要来源。真实语义两组在概念查询上均为 6/8
Top-1，但失误不同：融合把 TokenBucket 请求限流从第 2 提升到第 1，同时把无锁环形队列
从第 1 降到第 2；CRC 描述在两组均为第 2。Hash hybrid 也有把正确 Top-1 排名降低的情况。
本轮因此保留 `hybrid` 默认值与既有融合权重，开放显式选择，不引入基于小样本的自动路由。

这是受控、公开、可重复的文件级检索评测。短头文件、标识符查询占一半、小 corpus 以及
中英文查询分布都限制外推；保留集并非私有盲测。大型真实仓库、精确行级证据、修复成功率
和规模性能仍需单独验证。若据公开保留集调参，应新增另一组保留数据再验收。

## 工程验证

回归覆盖策略优先级、schema 与非法输入、结果缓存分区、并发会话配置隔离、checkpoint
默认值、CLI 新建/继续/审批恢复覆盖、以及审批暂停时更新策略后仍能正确恢复。独立评测测试
验证目标文件/查询不重叠、标签不进入 corpus、目标存在、输入变化拒绝完成，以及不加载
语义模型的 hash-only smoke。20 个头文件通过 C++17 语法检查；编译不属于检索运行器流程。

2026-11-13：新增 27 个回归用例；完整离线测试 463 passed、1 skipped（Windows 上的
POSIX 权限契约）、46 deselected（e2e、cpp、isolation、simulation、autodiff_codegen、
pinocchio、crocoddyl），报告 `.agent_state/rag-strategy-full.xml`。全项目 Ruff、mypy
（82 个源码文件）及 diff whitespace 检查通过。并发策略隔离用例预先初始化既有工具注册表，
测试范围是调用配置隔离，不包含注册表的并发冷启动。
