# 增量变更检测与关键词倒排检索

本轮针对大仓库热查询中重复读取 Chroma 全表、哈希全部源码和重新分词全部片段的开销。
检索策略、BM25 参数、符号/路径权重、父片段去重和 RRF 融合规则保持不变。

## 查询与更新路径

1. `open_code_index` 为持久化 collection 加上共享文件锁和 generation sidecar。
   首次查询或其他新版进程写入后，读取一次完整片段集合，构建内存倒排项和文件映射。
   generation 是随机版本标记，不以片段数量判断变化；同数量替换或缩小索引范围也能失效。
2. 热查询轮询已索引路径的 size、mtime、change time、设备/文件身份和文件类型。
   POSIX 使用 ctime，Windows 使用 `GetFileInformationByHandleEx(FileBasicInfo).ChangeTime`，
   不将 Windows 创建时间当成修改检测信号。`write_patch` 同时通知当前进程中的索引。
   Windows 字段定义见 [Microsoft FILE_BASIC_INFO](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_basic_info)。
3. 仅对变更文件读取内容、计算哈希、重新切块和更新 embedding。变化路径的词频、文档长度、
   文档频率集合、符号/路径倒排项与父片段上下文同步更新；删除或清空文件会清理旧项。
   关键词检索只评分查询词 postings 的候选并排序，路径范围内的 BM25 统计仍与原算法一致。
4. 返回前重新读取、哈希每个命中文件，校验其与建立结果时的源码快照一致，包括缓存命中、
   向量/混合检索和父片段上下文。变化则先刷新再重查，最多三轮；持续变化、不可读、路径
   越界、非法输入和后端失败均返回错误，不退回旧结果。稳定读取也检查读取前后的元数据。

`SourceDetector` 的状态属于每个 `CodeIndex`；索引的进程内锁保护派生状态，共享文件锁
串行化同一持久化目录的查询/同步（该目录内各 collection 共享锁）。后端写入前后都原子替换 generation，部分写入失败或
进程中止也会使其他读者重载。每个变更文件的内存状态在所有后端操作成功后才发布；失败
保留重新检查条件。这里的 generation 用于失效，不替代独立索引同步恢复 PR 的事务日志。

跨进程测试还复现了 Chroma 1.5.9 的读者状态问题：同数量替换后 SQLite 元数据已更新，
旧进程的 ANN 查询仍可能漏掉新向量。generation 带独立的进程标识；观察到其他进程写入时，
通过公开 `Client.close()` 关闭并重新打开本项目持有的共享客户端，重新获取 collection。
同目录的多个 `CodeIndex` 共用一个受管理客户端，随之换用新读者；没有清空 Chroma 全局缓存
或删除索引。最低 Chroma 版本更新为 1.5.9。跨进程三策略与同进程多索引读者均有回归覆盖。

源码哈希继续决定 source revision，片段 revision 决定缓存失效；只修改时间戳但内容未变
不会重新 embedding。倒排索引不另存持久化副本，重开进程从 Chroma 重建，避免双份索引
持久化一致性问题。路径范围统计缓存最多 64 项，片段变化时清空。

## 配置与边界

默认 `RAG_FRESHNESS=incremental`。对变更时间语义不可靠的网络盘、特殊文件系统或需要
每次查询检查所有内容的场景，设置 `RAG_FRESHNESS=strict`；此模式全量读取/哈希已索引
文件，但仍复用倒排项。无法取得可靠的 Windows ChangeTime 时，该文件自动退回内容检查。
配置在索引打开时读取，修改后需重启 Agent；`CodeIndex(..., strict_freshness=True)` 可直接配置。

- 热查询仍是 **O(已索引文件数) 的元数据轮询**，没有文件系统 watcher；本轮消除全量内容
  读取、Chroma 全表传输和重新分词，不宣称全查询 O(变更数)。高频词可命中所有片段，
  仍要评分/排序全部候选。内存随语料词项与 postings 增长，冷启动仍需全量构建。
- 新文件、重命名后的新路径以及删除/清空后恢复的文件仍需显式 `sync`，查询不扩大索引范围。
  首次查询、索引范围改变、跨进程写入后需要重载；自定义未包装 collection 为兼容直接写入，
  保守地每次读取全表，但相同片段不再分词。
- 所有并发写入者须使用新版项目索引入口或 `TrackedCollection`。直接调用底层 Chroma、
  旧版写入者及手工删除 generation 不受该协调协议保护；混合版本部署应先停止旧进程。
  同一进程也不应另开未受管理的 Chroma 客户端引用同目录，否则公开 close 无法释放最后一个读者。
  文件锁依赖存储的锁/原子替换语义，建议 artifacts 位于本地磁盘。
- 返回校验保证已检查的命中片段与当时读取的源码一致，不是整个文件系统的原子快照。
  特殊操作若能隐藏全部变更元数据，默认模式仍会检查已有命中，但可能漏掉新相关的非命中文件；
  此时应使用 strict。源码在最后一次校验后仍可能被其他程序修改，执行修复前仍须读当前文件。
- 路径过滤的向量检索仍沿用原候选扩展逻辑，本轮没有优化其全 collection 查询路径。
  离线 hash 与真实语义 embedding 共用变更检测，性能报告仅测 hash，不推断模型推理收益。

## 回归与复现

新增 `tests/test_incremental_retrieval.py`，覆盖原扫描评分等价（插入、更新、删除、路径前缀、
符号/路径独占命中、无词文本）、热查询工作计数、等长且恢复 mtime 的外部修改、原子替换、
删除、scope 改变、显式写入通知、缓存内容复核、查询竞态、连续写入失败关闭、读取权限错误、
严格/不支持变更 token 的回退、部分写入失败后重试、跨进程同数量 scope 替换、持久化重开
以及 token 子片段和父上下文刷新。测试以结果/工作计数断言，不设置机器相关的耗时阈值。

最终验证：新增 **38** 个用例；RAG 定向回归 **107 passed、1 deselected**；完整离线回归
**501 passed、1 skipped、46 deselected**。跳过项为 Windows 的 POSIX 权限契约，排除项为
e2e、cpp、isolation、simulation、autodiff_codegen、pinocchio、crocoddyl。完整报告为
`.agent_state/incremental-release.xml`。全项目 Ruff、mypy（85 个源码文件）、离线锁文件检查
与 diff whitespace 检查通过。没有把这些结果视为真实模型修复成功率或原生重依赖执行验收。

```bash
uv run pytest -q tests/test_incremental_retrieval.py tests/test_rag.py tests/test_workspace_rag.py \
  tests/test_token_chunking.py tests/test_code_embedding.py -m "not cpp"
uv run python scripts/run_cpp_retrieval_eval.py --split dev --hash-only \
  --output .agent_state/incremental-cpp-dev
uv run python scripts/run_cpp_retrieval_eval.py --split heldout --hash-only \
  --output .agent_state/incremental-cpp-heldout
uv run python scripts/benchmark_rag_incremental.py --files 200 1000 --queries 8 \
  --baseline-ref 175c11b --output docs/experiments/rag-incremental-performance-20261113.json
```

性能脚本直接加载本地 Git 基准提交 `175c11b` 的完整 `CodeIndex`，与当前实现使用相同源码、
独立 Chroma 数据库和离线 hash。仅针对可信本地 Git 版本使用该选项；需要本地保有该提交。
两档语料分别 200/1,000 个文件及片段，每文件 120 行，约 2.12/10.62 MB。查询含唯一 marker
与高频 `torque`，比较冷启动、未缓存热查询、缓存命中、公共词和单文件修改。两个版本均做
源码刷新，新版另有返回前内容校验。每阶段断言排序、评分和源码一致，并检查修改后旧词消失。
临时库保存在 `.agent_state/incremental-benchmarks/`，不写入项目源码或目标仓库。

## 2026-11-13 实测

原始报告：[性能数据](experiments/rag-incremental-performance-20261113.json)、
[独立 C++ 离线 smoke](experiments/rag-incremental-cpp-smoke-20261113.json)。
Windows 11、Python 3.13.2、Chroma 1.5.9 持久化库、离线 hash；性能复跑时未并行执行测试。
这里是 OS 文件缓存已暖的数据，不能用来估算冷磁盘吞吐。每个热阶段 8 次查询，p95 仅作
描述性统计；冷启动和单文件修改各一次，不作统计显著性主张。

| 文件/片段数 | 场景 | 原实现 p50 / p95（ms） | 增量实现 p50 / p95（ms） |
|---:|---|---:|---:|
| 200 | 未缓存，唯一词 | 472.008 / 484.493 | 168.096 / 179.441 |
| 200 | 缓存命中 | 199.186 / 215.156 | 154.088 / 174.488 |
| 200 | 未缓存，高频词 | 466.995 / 489.562 | 183.729 / 191.139 |
| 1,000 | 未缓存，唯一词 | 2,397.072 / 2,436.971 | 792.987 / 849.153 |
| 1,000 | 缓存命中 | 976.933 / 1,026.045 | 791.250 / 823.188 |
| 1,000 | 未缓存，高频词 | 2,377.328 / 2,451.014 | 821.197 / 859.930 |

1,000 文件的未缓存唯一词查询 p50 约 **3.02 倍加速**。8 次查询的源码内容读取从 8,000 次
降为 8 次（84,992,000 → 84,992 字节），Chroma 全量读取从 16 次降为 0，重新分词片段从
8,000 降为 0。新版仍有 8,016 次元数据检查，包含返回前稳定读取的前后检查。
高频词返回 5 个文件，8 次查询需读 40 个命中文件；候选评分仍可能覆盖全部片段。

单文件修改在 1,000 文件档为 2,504.178 → 904.397 ms；新版只重新分词 1 个片段，读取
2 次源码（更新与返回校验），不读取完整 collection。所有对比的内容、排名和评分相同。

代价同样保留：首次查询从 499.942 → 1,090.656 ms（200 文件）、
2,458.190 → 5,274.119 ms（1,000 文件），因为需要构建倒排状态并完成冷启动源码检查。
纯缓存命中改善远小于未缓存查询，Windows 路径安全校验与元数据轮询仍占明显开销。
这些合成语料结论不等同于百万文件生产仓库或语义模型性能；下一步可测本地 watcher/
变更日志与轮询兜底，并单独度量倒排项的峰值内存和多读者锁争用。

独立 C++ 开发/公开保留集各 16 题，三组 hash-only 指标均与上轮报告相同：保留集 lexical
Hit@1/Hit@5 为 68.75%/81.25%，hash vector 为 37.50%/68.75%，hash hybrid 为
62.50%/87.50%。源码 revision 仍为 `5277449d95821e6e`，未重跑真实语义模型评测或调用付费 LLM。
