# M7.1 结构化修复轨迹

更新：2026-10-28。此项为 M7.1 实验观测设施，给 M8 的 RepairSkill 提炼准备证据；
不是技能提炼器或技能验证器。[2026-10-29 首轮真实四组实验](experiments/m71-live-20261029.md)
已使用该轨迹设施完成，结果与原始记录的开发证据边界仍分开报告。
四组运行器与共享失败分类已接入，见 [公平消融运行器](phase71-fair-ablations.md)。

## 使用

现有公开 benchmark 可显式启用记录，输出位于报告目录的 `traces/<独立运行目录>/`：

```bash
uv run python scripts/run_benchmark.py --manifest benchmarks/cpp_professional_cases.json --case cpp_scalar_adl --repair-trace
```

案例 ID 以 `--list` 输出为准。运行 Agent 会调用配置的付费模型；记录与检查本身不调用模型。
正式 M7.1 对照组必须统一启用记录、模型和预算。`focused/general` 仍不等同于四组消融。

外部仓库会话：

```bash
uv run python scripts/run_agent.py "定位并修复缺陷" --workspace /path/to/repo --artifact-root /path/to/artifacts --cpp-tools --trusted-local --repair-trace
uv run python scripts/inspect_repair_trace.py /path/to/artifacts/repair_traces/<segment-id>
```

容器执行可将 `--trusted-local` 换为 `--execution-image <pinned-image>`。
CLI 会打印轨迹目录；恢复时仍需提供 `--repair-trace` 和原工作区参数。
每次执行/恢复生成新 segment，以 `metadata.thread_id` 关联；不会覆盖旧文件。
`--status` 不创建记录，启用记录不改变审批流程或模型可见上下文。
Python 调用可用 `use_repair_trace(directory, source, metadata)` 包裹 graph.invoke/stream；
目录必须不存在，树级源码根与轨迹目录不能重叠。

## 产物与契约（schema_version = 1）

| 文件 | 内容 |
|---|---|
| `manifest.json` | development scope、run_id、源码根/范围、环境及其摘要、调用方提供的任务/预算/运行元信息 |
| `events.jsonl` | 带时间、连续序号、节点 span、循环步和源码指纹的追加事件，每条写入后 flush/fsync |
| `sources/<sha256>.json` | 相对路径、可执行位和文件内容 SHA-256；不是源码副本 |

环境记录采用允许列表：Python、平台、模型标识、执行后端、固定容器镜像或 trusted-local 标志、
调用方提供的工具链指纹。不采集环境变量全集、凭据或模型 thinking/reasoning 块。
2026-10-29 起，`agent_message.provider_response` 仅允许保存响应中的
`model_name/model/system_fingerprint/finish_reason`，用于识别请求别名与实际服务模型的区别；
其他供应商响应元数据仍不导出。`usage` 保留供应商提供的输入缓存明细。
环境摘要只代表已记录字段，不等价于完整可复现环境；模型别名变化及本地未固定依赖仍需实验协议约束。

事件分为：

- `session_started/session_closed`：记录段生命周期、耗时与记录开销；closed 不表示任务通过。
- `node_started/node_completed/node_error/node_interrupted`：操作边界；完成节点记录前后源码身份。
- `agent_message`：公开文本陈述与供应商提供的 token usage；不自动认定为已证实的修复假设。
- `tool_requested/tool_result`：调用 ID、工具、参数、输出、返回/错误/拒绝/跳过状态。
- `development_validation`：现有公开 benchmark gate 的报告值与原始输出，通过 span 关联源码边界。

`cpp_build` 的结构化 `passed/stage/exit_code` 记录为 build observation，不能当作全部回归通过。
普通 `execute_python` 返回的日志保留为原始输出，不通过任意 `passed=True` 字符串推断修复通过。
公开 gate 的 `reported_passed` 仍是已有 gate 的报告值；记录层没有赋予它独立隐藏 oracle 的权威。

工具批次保留原并发行为。结果属于 `node_batch`，消息顺序不是实际执行顺序；前后源码相同
也不证明执行途中从未发生修改。只有后续验证协议能确认某项验证覆盖特定不可变候选。
源码变化会产生不同 manifest；旧证据不自动迁移到新源码。当前记录层不改变已有 benchmark 路由。

外部仓库使用与 M7.0 `source_digest` 相同的有界、排除私有目录的源码枚举及哈希算法。
旧 fixture 仅绑定目标文件，`source_scope=fixture_file`，不能据此宣称完整仓库已经绑定。
源码读取失败记为 unavailable，不能当作“源码未变”。源码 manifest 只验证内容身份，
完整复现还需要保存对应仓库 revision/补丁、依赖与构建锁。

## 完整性与后续技能准入

读取器校验 schema、development scope、run_id、连续序号及源码 manifest 摘要。
进程终止时保留已完成的 JSONL 行；最后一条未写完整的行标为 partial_tail，未完成节点显式列出。
普通异常保留 node_error；审批暂停记 node_interrupted。写入失败会使本次尝试失败，不能静默丢轨迹。
文件摘要不是抗恶意篡改签名；轨迹是操作者控制的实验产物，不对同宿主不可信代码提供安全保证。

文本、容器层级和集合长度有上限，截断带标志；凭据形状的字典键会脱敏，但不能保证任意
源码/工具自由文本不含敏感信息。因此轨迹不自动公开、不自动索引，也不直接提供给模型检索。
读取器始终返回 `skill_eligible=false`：记录存在不等于满足技能准入。

M8 的提炼/验证器需要另外检查：完整性、适用条件、任务划分、源码/环境一致性、独立开发回归、
反例与失败模式、负迁移、版本与撤销。截断或不完整轨迹须补充证据后再考虑准入。
最终隐藏评测入口未接入记录层，oracle 输出不作为技能材料。benchmark 父进程的最终判分
仍写入原报告；旧公开 benchmark 的下一轮公开修复提示可能包含其开发反馈，不能冒称隐藏测试。

## 当前实现与待办

已实现图节点记录、公开 gate 事件、外部 CLI 开关、benchmark 子进程独立记录和报告关联、
有界载荷、源码 manifest、离线读取检查。图状态/checkpoint 不存放轨迹正文。

后续进展：四组运行器与共享失败分类已实现，分类附在工具和开发验证事件中；
节点完成事件同时保存运行器的 `failure` 与 `stop_reason`，保留预算耗尽和策略拒绝证据。
`model_identity` 元信息允许实验运行器明确标记 scripted 模型，避免将替身用量标为真实模型用量。
待办：真实模型四组重复轮次与完整 17 题；进一步细化编译/导数/动力学/仿真结构化指标适配；
候选级不可变执行证据；反例缩减；RepairSkill schema、提炼、验证、检索与撤销；数据集划分和迁移实验。
初始轨迹实现验收未启动付费批跑；后续首轮真实实验已单独记录，尚不宣称真实缺陷迁移收益。

## 本轮验证记录

2026-10-28，Windows 项目 `.venv`：

- 离线回归：`pytest -m "not e2e and not cpp and not isolation"`，337 passed、1 skipped、26 deselected。
  其中新增 9 项轨迹测试覆盖真实图/工具调度、开关前后行为一致、审批恢复与拒绝、预算跳过、
  公开 gate、异常与截断尾行、字段脱敏与 manifest 校验、源码缺失、报告关联和实际 worker 子进程。
- `ruff check .` 与 `mypy src scripts/run_agent.py scripts/run_benchmark.py scripts/inspect_repair_trace.py` 通过。
- 实际读取离线修复轨迹：26 条事件、3 个工具结果、1 次源码变化，无未结束节点；用量来自测试替身，
  不代表真实模型表现。C++ 构建返回值在新增测试中使用替身，本轮未重跑原生编译或 Docker 验收。

首次全套运行受当前沙箱系统临时目录权限影响；将测试进程的 TEMP/TMP 和 pytest basetemp
设到项目 `.agent_state` 下后通过，未修改业务代码绕过权限。跳过项为 Windows 不适用的
POSIX 可执行位测试；付费模型与原生/容器专项被显式排除。
