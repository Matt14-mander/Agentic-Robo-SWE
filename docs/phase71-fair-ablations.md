# M7.1 四组公平消融与失败分类

更新：2026-10-28。运行器、分组能力边界、统一失败分类及结构化轨迹已实现。
首个适配器覆盖 `benchmarks/cpp_robotics_cases.json` 的 8 个公开 C++ 合成题；
不把公开校准成绩称为隐藏测试通过率，不把 scripted smoke 称为模型修复实验。

## 四组定义

| arm | 源码工具 | 领域说明 | 专业诊断 | 开发验证反馈 |
|---|---|---|---|---|
| `base` | 读取目标、唯一匹配修改目标 | 无 | 无 | 无 |
| `docs` | 同上 | 固定 `ablation_domain_notes.md` | 无 | 无 |
| `tools` | 同上 | 同上 | 固定探针编译、Scalar/AD 静态诊断 | 无 |
| `feedback` | 同上 | 同上 | 同上 | 公开数值验证器 |

这是逐项累加设计。四组使用相同任务文字、起始源码、模型配置、上下文保留策略和预算。
工具 schema 与系统提示词摘要记录在 worker 结果和 trace manifest 中。
固定领域说明不是动态 RAG；本轮不同时引入检索、技能记忆、不同模型或搜索策略。

旧 `focused/general` 入口继续用于原 benchmark，不能替代这四组。新入口使用专门的图：
模型可见工具只有表中的能力；调度层再次验证工具名和预算。没有通用 Python 执行、
仓库遍历、任意路径读写、自动 validator gate 或最终判分后的反馈重试。
专业编译使用固定 probe，不向模型提供 shell 命令或可编辑构建脚本。

初始 8 题已给出精确目标，因此基础组只需要源码读写。此基线不能外推为“带任意 shell 的
通用 SWE Agent”。CMake/clangd 的真实仓库工具和其他重依赖题将在后续适配器中单独验收。

## 运行方式

无需模型调用的真实 C++ smoke：

```bash
uv run python scripts/run_ablation.py --smoke --trusted-local --output .agent_state/ablations/smoke-01
```

该模式使用明确标记的确定性模型替身，按已知缺陷执行测试补丁；编译、运行和独立 Python
数值判分是真实执行。替身 token 数仅用于测试预算记账，不代表模型成本。
先对 broken/reference 做实际校准；然后 8 题 × 4 组运行。每次重新编译，不共享构建缓存。
可通过 `--case cpp_sample_period` 选择一个任务；未知或重复 ID 会被拒绝。

后续真实模型实验（会产生模型费用）：

```bash
uv run python scripts/run_ablation.py --live --trusted-local --repeats 3 --seed 17 --max-steps 8 --max-tools 12 --max-tokens 16000 --timeout 300 --output .agent_state/ablations/live-01
```

输出目录必须不存在，防止覆写上次结果。`--compiler` 可以指定 g++ 的路径。
默认不会自行选择 live 模式；必须指定 `--smoke` 或 `--live`。
这是显式 trusted-local 的公开开发实验，候选代码在宿主执行，不是安全隔离沙箱。
现有独立容器 oracle 不会因这个适配器出现而自动覆盖这 8 道公开题。

## 公平性与预算

- 开始前保存 `protocol.json`，固定 case 清单与 fixture 哈希、预算、重复次数、seed、模型标识、
  编译器版本、probe/oracle/领域说明摘要。每个 case/repeat/arm 使用独立工作区和产物目录。
- 任务顺序由 seed 固定，组顺序按任务和 repeat 轮转；保留真实执行顺序，减少固定组序的缓存偏差。
- broken 必须呈现候选缺陷，reference 必须通过。环境缺失或校准无效时四组全部阻断，
  不调用模型，也不把它们计入有效任务分母。参考解只由操作者校准流程使用。
- 模型调用次数与工具调用次数执行上限；工具批次超过剩余预算会整批拒绝。
  不在耗尽后免费追加总结调用、隐藏工具或额外验证重试。
- 总 token 基于供应商 usage 检查；缺失或不一致的 usage 视为无效证据。
  每次输出有剩余 token 上限，但输入 token 在调用返回后才获知，因此总 token 是
  **调用边界检查的上限**，不是绝不超支的计费保证。超出后拒绝待执行工具，报告预算耗尽，
  不能把超预算修复计为合规成功。墙钟由外部 worker 超时边界控制。
- 四组均保存修复轨迹，记录开销计入尝试墙钟。调用方预算、trace 开销与最终判分耗时的
  边界必须一起报告；模型别名相同不等于模型服务版本永不变化。

最终判分在 worker 退出、trace 关闭后，由同一个公开验证实现重新运行；结果只写操作者报告。
D 组的开发反馈与本适配器最终判分使用同一套公开开发样例，因此这能验证运行协议，
不能证明隐藏集泛化。进入真实修复结论前必须建立独立开发/保留划分并接入 M7.0 的隔离最终评测。

## 产物与指标

- `protocol.json`：固定实验协议及 `protocol_id`。
- `calibration/<case>/`：broken/reference、编译日志、原生观察及校准报告。
- `runs/<case>-<repeat>-<arm>/`：独立源码、request、worker 结果、轨迹、最终判分、补丁与源码指纹。
- `report.json`：逐次原子更新；中断保留已有结果和运行状态。

每组报告计划/已完成数量、校准有效次数、成功次数和失败类别计数。有效校准后的运行错误、
超时及预算耗尽算端到端失败，同时在类别计数中单列；没有悄悄从分母删掉。
live 的汇总按 case 平均重复实验结果，并按 case 做 bootstrap 区间，避免把重复运行当作新题。
小样本区间仍不能说明跨仓库泛化；scripted smoke 不生成模型质量区间。

## 统一失败分类 v1

`agent.failures` 提供共享的 `category/stage/reason/schema_version`，用于四组报告、
旧 benchmark 验证结果、结构化工具/开发验证轨迹和仿真诊断。

| category | 判据 |
|---|---|
| `none` | 当前操作明确通过；构建通过仍仅代表构建 |
| `environment` | 明确的能力或运行环境缺失 |
| `invalid_baseline` | broken/reference 校准不符合协议 |
| `compile_error` / `runtime_error` | 明确失败的编译/原生执行阶段 |
| `numerical_mismatch` / `derivative_mismatch` | 数值正确性/稠密或稀疏导数阶段失败 |
| `compatibility_error` | AD 兼容性阶段拒绝，不等同于实测导数错误 |
| `safety_violation` / `behavioral_mismatch` | 结构化安全事件/跟踪或控制环行为失败 |
| `performance_regression` | 显式性能门禁未达到要求 |
| `timeout` / `budget_exhausted` | 执行超时/协议预算耗尽 |
| `invalid_evidence` | 非法 verdict、损坏报告、缺失用量等协议证据问题 |
| `tool_error` / `model_error` | 工具调度或模型调用失败 |
| `unclassified` | 证据不足或旧验证器没有足够的结构化字段 |

分类依据明确 stage/结构化字段，不通过自由文本关键词猜原因。编译报错中的 `error` 字段
不会自动归为环境故障；无法区分 report/build 的旧复合阶段保留未知。旧 benchmark 的执行
策略与基线接受规则没有改成新实验协议；严格消融应使用新入口。

## 完成边界

已实现四组运行器、真实原生公开 8 题适配、失败分类、预算/权限约束、轨迹与增量报告。
仍需：真实模型重复实验、17 题重依赖适配器、独立隐藏集、真实仓库任务、精细领域指标扩展。
这些工作属于 M7.1 剩余实验和 M7.2；RepairSkill 自动提炼/准入仍属于 M8。

## 2026-10-28 本地验收

- Windows 项目 `.venv` 离线回归：374 passed、1 skipped、26 deselected。
  排除 live e2e、C++/容器专项；跳过 Windows 不适用的 POSIX 可执行位用例。
  测试进程 TEMP/TMP 与 basetemp 位于项目 `.agent_state`，避免沙箱系统临时目录权限问题。
- 最后补充的轨迹/预算回归：46 passed，确认预算与策略失败同时出现在节点完成事件和运行报告中。
- `ruff check .`、`mypy src scripts/run_ablation.py`、`git diff --check` 通过。
- g++ 13.2.0 原生 smoke：8 题 broken/reference 校准有效，四组共 32 次运行全部通过。
  32 份轨迹共有 896 条事件；审计确认首尾源码指纹、协议标识和预算计数一致，
  只有 `feedback` 组出现开发验证事件，无未结束节点或截断尾行。
- 父进程预算复核及最终判分源码绑定加入后，另跑 `cpp_scalar_adl` 四组原生 smoke，4 次通过。
  最终判分另记录耗时与 `evaluated_source`，判分期间源码变化会拒绝接受结果。
- 本轮没有模型 API 调用。smoke 的确定性修复和 token 替身不能用来证明四组的质量差异。

本地产物（忽略目录，不提交二进制）：

- `.agent_state/ablation-smoke-eight-v1/report.json`、`protocol.json`、`audit.json`：8×4 原生 smoke 与轨迹审计。
- `.agent_state/ablation-smoke-final-audit/report.json`：最终父进程审计逻辑的单题四组原生复测。

CI 的 `cpp-domain-checks` 已增加不调用模型的 8×4 原生 smoke，并保存协议、结果及轨迹。
CI 配置已更新，本轮未触发或宣称远端 CI 通过。
