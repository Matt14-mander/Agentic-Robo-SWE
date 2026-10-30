# M7.1：17 题适配、独立输入划分与原生异常证据

更新：2026-10-30（运行始于 10-29）。本轮不调用模型 API；采用现有四组能力递增设计、
8 次模型调用 / 12 次工具 / 16k token / 300 秒尝试预算，先验收离线协议和真实原生 smoke。
旧 `--suite robotics --split public` 入口与历史实验数据保留，新实验使用独立协议 ID。

## 17 题适配

| 子集 | 题数 | 真正执行的检查 |
|---|---:|---|
| Robotics 单头文件 | 8 | 原生 C++ 观察 + 独立 Python 方程；Scalar 题含 Dual 导数 |
| C++ reference | 1 | `robot_step` 原生观察 + 操作者 Python 方程 |
| Dense AD/CodeGen | 4 | 原生模板、CppAD、生成库、解析导数与有限差分；复用 Dense/Sparse 报告门禁 |
| Sparse Jacobian | 2 | 同上，另检查 COO 结构零、缺失及多余依赖 |
| Pinocchio RNEA | 2 | Pinocchio 原生、解析 Jacobian、CppAD、CodeGen 与有限差分的一致性 |

适配器位于 `src/agent/ablation_suite.py`。所有题仍只开放 `candidate.hpp` 读写；
tools 组增加固定编译探针与 AD 静态诊断，feedback 组再增加开发验证。
RNEA 本轮评价数值与导数正确性；现有 harness 的性能/控制环数据不作为本轮通过条件，
不混入性能优化结论。重依赖版本按安装 marker、固定 commit 和必要头文件核对，
缺失则在校准阶段阻断四组，不启动模型或 smoke 替身。

RNEA 适配器使用固定 Pinocchio 安装导出的 `pkg-config` 编译/链接参数，核对版本与文件摘要，
并保留原 CMake 使用的 Boost MPL 限制与 CppAD 宏。省去每次重复 CMake 配置，但每次仍重新
编译候选与原 harness，不跨尝试复用二进制。构建记录保存完整 argv 和导出参数；
其他 Domain Pack 的原 CMake 入口不变。

确定性 smoke 使用操作者参考实现进行补丁替换，真实编译、执行和判分；
参考实现不会出现在 live 请求、提示词或工具中。smoke 成绩与替身 token 不是模型能力/成本。

## 开发与保留输入

`independent-inputs-v1` 为每题生成两个确定性但不重叠的输入集合：

- Robotics、C++ reference、RNEA：每侧 24 个输入；Dense/Sparse：每侧 32 个输入。
- 开发与保留使用不同 seed；协议记录每侧样本数和完整输入摘要，开始前检查没有交集。
  两侧均不复用旧固定回归/边界点；旧公开验证器继续保留原回归/边界覆盖。
- broken/reference 必须分别在两侧校准。源码或依赖缺失、参考解失败、报告损坏均不得计为校准成功。
- worker 只调用 `development` 验证；父进程在轨迹关闭后调用 `heldout` 最终判分，
  最终结果不反馈给 Agent，不用于追加修复尝试。
- 四组共享同一题的开发/保留分布、源码、编译条件、预算和最终评价。各次构建独立，
  不跨组共用构建缓存；依赖目录可共享只读安装。

新适配器的 harness 统一使用 `-O0`（重依赖 CMake 同时保留 `-DNDEBUG`），与轻量原生适配器
一致，减少模板优化阶段占用修复预算。生成库的既有编译设置不变。`build_policy` 及相关源码
摘要写入协议；这不是性能基准，不能将时间结果与旧 Release 优化实验直接比较。

这里的“保留”是**输入集合与调用路径隔离**：题目、生成器和 seed 仍在公开仓库，
trusted-local 候选与验证器不具备 OS 安全隔离，Dense/RNEA 也复用了同进程参考计算。
因此协议标记 `public_synthetic_disjoint_inputs_not_secure_hidden`，不宣称完成 M7.0 的
私有容器 oracle 接入，也不把它称为真实缺陷隐藏测试。技能来源/准入/最终测试按仓库或
缺陷家族拆分仍属于 M7.2/M8；本轮只解决输入重叠与最终反馈进入修复环路的问题。

## 原生退出与历史异常

`ProcessResult` 新增 `native_returncode` 与 `termination`，保留原有 `returncode` 兼容字段。
监督器超时/输出限制产生的 124 与程序自行返回 124 可以区分。
原生适配器保存 `execution.json`，包含退出码、Windows 十六进制状态或 POSIX signal、
终止原因、stdout/stderr、耗时、输出是否存在、字节数、行数及 SHA-256。
数值判分失败、非零退出或输出缺行时也保留进程证据；空 stderr 不再变成空错误原因。

对上一轮 `cpp_anti_windup` 参考程序的排查发现：

- Windows Application Event 1000 的应用路径与原校准程序完全一致，时间为
  **2026-10-29 19:11:44 +08:00**，异常码 **0xC0000025**，故障模块 `ntdll.dll`。
- 微软将该码定义为 `STATUS_NONCONTINUABLE_EXCEPTION`，即不可继续的异常。
  [官方状态码定义](https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-erref/596a1078-e883-4972-9bbc-49e60bebca55)
- 原观察文件仅 19/28 行，与异常终止相符。这能确认不是普通数值判分失败；
  没有崩溃转储/堆栈，不能从 `ntdll.dll` 或状态码进一步归因到某个源码、驱动或工具链缺陷。
- 新编译参考程序连续 20 次 fresh-process 复测均退出 0、输出 28 行并通过独立判分，
  未复现历史异常；不据此声称根因已修复。原报告与后续证据分开保存。

[可提交的复测与崩溃事件摘要](experiments/m71-native-stability-20261029.json)保存原文件摘要、
匹配的相对应用路径、模块版本、异常码及 20 次进程收据。

复测入口（不自动重试掩盖失败）：

```bash
uv run python scripts/check_native_stability.py --output .agent_state/native-stability-new --runs 20
```

## 运行与复核

完整 17 题、独立输入、无模型调用：

```bash
uv run python scripts/run_ablation.py --smoke --trusted-local --suite professional --split independent --output .agent_state/professional-smoke-new
```

默认编译器仍为 `g++`，兼容旧入口；RNEA 的 300 秒预算复验使用了下述固定 Clang 14 C++
驱动文件和 Linux 可写构建目录。按本轮已准备的环境运行时，追加
`--compiler .agent_state/toolchains/clang14/install/usr/lib/llvm-14/bin/clang++`。
不同编译器/文件系统的耗时不可直接比较；正式模型实验前应先在拟用环境完成全套 smoke。

仅选择题目可以重复使用 `--case`。默认读取项目的 `.agent_state/domain_deps/`；
已有固定版本 Linux 安装可以通过 `--dependency-root /absolute/path/to/domain_deps` 指定。
该目录只供探测/编译读取，所有构建与运行产物仍放在本轮输出目录。依赖路径和 marker 摘要
进入协议身份；不能把不同依赖/平台的 smoke 当作同一环境的模型消融合并。

输出包括协议、双侧校准、逐次 worker/trace/final、构建阶段日志和原生执行收据。
离线复核仍可使用 `scripts/analyze_ablation.py`；Linux 原生源码执行位指纹应在 Linux 中审计，
避免跨 Windows/WSL 的权限位表示差异导致错误比较。

CI 新增手动触发的 `m71-professional-smoke.yml`，使用固定 Clang 14 前端和已有脚本固定的依赖版本执行
完整 17×4 原生 smoke、轨迹审计并上传 JSON/源码证据；不调用模型 API。
本轮只新增配置，没有触发远端 CI。日常离线 CI 自动覆盖新增划分与权限回归测试。

## 本轮验收记录

各行独立报告，跨平台结果不合并为同一环境的消融成功率。全部使用 scripted smoke，
没有真实模型调用，也没有模型费用或技能迁移收益结论。

| 验收 | 已完成结果 | 可复核证据 |
|---|---|---|
| Windows / GCC 13.2，17 题入口 | 9 道可运行题 × 4 组 = 36 次通过；8 道 Linux 依赖题的 32 个槽位在校准前环境阻断 | [Windows 审计摘要](experiments/m71-professional-windows-smoke-20261029.json) |
| C++ reference 最终签名复验 | `const Scalar& u` 参考实现，双侧校准通过，四组 4/4；单独保留，不与上行重复计数 | [最终复验摘要](experiments/m71-reference-final-smoke-20261029.json) |
| Linux / GCC 11.4，最终版本轻量题 | 9 题 × 4 组 = 36/36；双侧校准、36 份轨迹及 1,008 条事件审计通过 | [Linux 轻量审计摘要](experiments/ablation-light-linux-o0-20261029-audit.json) |
| Linux / GCC 11.4，AD/Sparse | 6 题 × 4 组 = 24/24；24 份轨迹、672 条事件审计通过。所属 8 题批次在 RNEA 校准期间中断，另 8 个槽位未执行 | [中断批次的已完成尝试审计](experiments/ablation-heavy-linux-o0-20261029-audit.json) |
| Linux / Clang 14，RNEA 直接构建 | 2 题 × 4 组 = 8/8；双侧校准、8 份轨迹及 224 条事件审计通过 | [RNEA 最终审计摘要](experiments/agentic-robo-swe-m71-rnea-direct-20261030.json) |
| anti_windup 参考程序稳定性 | 20/20 fresh-process 退出 0，28 行完整输出且数值通过 | [退出与事件摘要](experiments/m71-native-stability-20261029.json) |

按题目去重，三个 Linux 验收子集覆盖 **17 题 × 4 组的 68 条路径**，共 1,904 条轨迹事件，
所有已执行尝试通过审计与最终数值门禁。各批次冻结的适配器版本、编译器、文件系统与协议 ID
分别保留；这是分阶段实现覆盖，不是一次完整同环境实验。GCC 子集最长尝试约 187.0 秒，
RNEA 直接构建子集最长约 **184.4 秒**，均未放宽 300 秒 worker 预算。

最终完整离线回归：`pytest -q -m 'not e2e and not cpp and not isolation'` 为 **385 passed、
1 skipped、26 deselected**；包括 17×4 路由/轨迹、两侧输入无交集、开发报告不能冒充保留报告、
缺依赖先阻断及原生退出/超时/部分输出等回归。原生 AD/RNEA 验收另行报告，不以这些模拟测试代替。
随后增加中断批次审计及负例，相关测试子集再次 55 项通过；RNEA 直接构建新增两项依赖版本与
候选重新编译回归，最终相关子集 **57/57** 通过。一次复验曾在 Windows 原子替换
`report.json` 时遇到 `WinError 5`（56 passed、1 failed），换独立临时目录复核通过；
保留该环境性异常，不将第一次运行算为全通过。Ruff 与 Mypy（81 个源码文件）通过。

Windows 主批次 36 份有效轨迹、1,008 条事件均通过源码/用量/闭合状态审计；
缺 Linux 能力的 32 个槽位没有启动 worker。C++ reference 最终复验另有 4 份轨迹、112 条事件。
轻量 Windows 验收早于重依赖构建策略最终冻结，不能与后续 Linux 协议 ID 混用。

Linux 重依赖最终 smoke 使用 WSL Ubuntu 22.04 / GCC 11.4，以及已有固定版本只读依赖；
构建、运行产物和临时文件位于当前工作区可写目录。本轮未重新安装 AD/Pinocchio 依赖，
也未修复/重挂载只读根文件系统。
前期用于排查构建耗时的 Release、依赖路径及 tmpfs 预检均在模型尝试前中断；前两次日志保留在
`.agent_state/ablation-heavy-linux-20261029`、
`.agent_state/ablation-heavy-linux-native-deps-20261029`。早期
`/dev/shm/agentic-robo-swe-m71-preflight-20261029` 临时目录已不存在，无法完整回溯原始日志；
不计为通过。
固定 `-O0` 的 GCC 批次独立写入 `.agent_state/ablation-heavy-linux-o0-20261029`。
同一 Linux 工具链下的九道轻量题最终复验单独写入 `.agent_state/ablation-light-linux-o0-20261029`。
两批次部分重叠运行；墙钟记录用于预算核对，不是专用主机上的性能排名。

GCC 的首道 RNEA 开发侧校准中，错误实现被数值门禁拒绝、参考实现通过，原生退出均为 0。
但两次配置/编译分别约为 **61.4+325.8 秒**和 **60.3+302.3 秒**，单次构建已超过 300 秒
worker 预算，编译进程观测内存约 6.9 GB。因此在保留侧校准开始时中止 GCC 批次，
不追加预算，也不把未执行的 RNEA 槽位算成修复失败或通过。

`analyze_ablation.py --allow-incomplete` 仅允许显式审计 `interrupted` 批次的已保存结果；
输出 `experiment_complete=false`、`missing_attempts` 和 `observed_attempts_only`，
仍检查源码、轨迹与用量，不允许将其合并成校准补跑成功。默认入口继续拒绝不完整实验。

随后从系统已配置 Ubuntu 源下载固定 `1:14.0.0-1ubuntu1.1` 的 `clang-14` 与
`libclang-common-14-dev`（共约 6 MB），仅解包到项目 `.agent_state/toolchains/clang14`，
复用系统已有 LLVM/Clang 14 共享库，没有修改系统软件安装。两道 RNEA 的 Clang 试验写入
`.agent_state/ablation-rnea-clang14-20261029`，与 GCC 使用不同协议/环境身份；四组权限、
输入和预算不变。它与 GCC 子集只能共同说明适配器覆盖，不能拼成单环境模型消融成绩。
[Clang 包版本和 SHA-256](experiments/m71-clang14-toolchain-20261029.json)单独保存。

跨 NTFS 的 Clang 首次配置/构建约 **62.8+182.1 秒**；Linux 内存目录的 CMake 路径首次约
**83.7+81.2 秒**，仍会让 feedback 的两次构建挤占原预算。这两次预检都在修复尝试前停止。
其中 Clang/CMake 的 `/dev/shm/agentic-robo-swe-m71-rnea-clang14-20261029` 目录随后也已不存在；
所列时长来自当时读取的构建收据，不能替代完整原始日志或用于验收计数。
最终 RNEA 采用上述直接编译入口，在 `/dev/shm/agentic-robo-swe-m71-rnea-direct-20261030`
进行独立批次，省去固定构建配置开销；已完成全部 8 次运行，并在 Linux 环境退出前将原始轨迹、
源码、观察值、编译命令和 JSON 证据归档回
`.agent_state/agentic-robo-swe-m71-rnea-direct-20261030.tar.gz`。
[归档校验记录](experiments/m71-rnea-artifact-20261030.json)包含 SHA-256、327 个条目、
8 份轨迹、18 份原生运行收据和双侧校准摘要。归档不包含可重建的二进制/对象文件。

原始 JSON、轨迹与原生构建产物保存在 `.agent_state/`；文档中的摘要保留报告、协议和源码摘要，
可用于复核来源，但不替代原始构建日志。完整 17 题真实模型重复实验、私有隔离 oracle、
按缺陷家族/仓库划分任务集及 RepairSkill 准入尚未执行。

下一步 P0：选定同一 Linux 文件系统与 Clang 14 驱动，对正式实验环境进行完整 17 题 smoke
并冻结配置；接入私有隔离最终评测。随后再做真实模型重复轮次。当前公开输入划分及
scripted 轨迹不能直接晋升为已验证 RepairSkill。
