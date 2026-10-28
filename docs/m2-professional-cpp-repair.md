# M2.0：专业 C++ 源码修复评测起步

状态（2026-10-26）：首批 8 个 C++17 任务、原生正反校准、统一运行入口与可审计报告已实现。
本机 Windows/GCC 13.2/CMake 3.30 已运行全部 8 对 broken/reference 原生校验。
这些是评测基础设施与题目有效性证据，**尚未运行付费 LLM 修复实验，不能报告 Agent 成功率**。

## 定位与数据边界

目标是验证 Agent 能否修改机器人 C++ 源码并通过独立判定，而非重复测量数值库速度。
第一批是人工构造、公开可读的单头文件任务（synthetic_controlled / public development），
代表常见专业错误模式，不是来自真实仓库的问题，不是保留集，也不能声称覆盖大型工程。
校准参考解位于 pack 的 harness/reference.hpp；只供验证器正反自检，修复实验禁止读取、复制或修改参考解。
这条限制依赖实验协议，**尚未实现 OS 级隐藏参考解或防作弊隔离**，因此不能用结果证明未知缺陷泛化能力。

| 任务 ID | 故障模式 | 独立检查 |
|---|---|---|
| cpp_scalar_adl | Scalar 转为 double 丢失导数 | double 值 + 双方向 forward-mode 导数 |
| cpp_jacobian_layout | 行/列主序索引错位 | 非对称 3×2 J*v |
| cpp_quaternion_order | xyzw/wxyz 约定错位 | 缩放/符号等价四元数与向量旋转 |
| cpp_wrench_frame | 参考原点转换的叉乘符号错误 | 旋转后的力与原点平移力矩 |
| cpp_sample_period | 整数除法把采样周期变为零 | 多频率恒加速度精确积分 |
| cpp_trajectory_scaling | 归一化时间缺少二阶链式缩放 | 不同持续时间、端点与内部导数 |
| cpp_anti_windup | 只夹紧输出、不阻止积分饱和 | 正负饱和、恢复、正常积分与最终状态 |
| cpp_joint_wrap | 连续关节跨界误差未取最短角 | 正负 pi 两侧与多圈输入 |

每题使用 4 个固定场景和 24 个固定随机输入；每题 seed=24680+场景索引。
Python 方程独立计算预期结果，C++ 只返回实际观察，不自报 passed。
统一容差 abs=1e-9、rel=1e-8；检查输出行数、形状和有限性，保存首个失败输入。
自定义 Dual 仅用于 Scalar 类型传播校准，不等同于已验证 CppAD/CppADCodeGen。
当前 8 题的原始错误均可编译；编译失败视为环境/构建错误，后续编译缺陷任务需要单独定义合法失败阶段。

## 两层任务清单

- `benchmarks/cpp_robotics_cases.json`：新增 8 题，仅需 CMake/C++17，本地 Windows/Linux 可运行。
- `benchmarks/cpp_professional_cases.json`：组合新增 8 题与已有 9 个 C++/AD/Sparse/RNEA 任务，共 17 题。
  AD 与 RNEA 子集要求 WSL/Linux 已安装固定版本依赖；不能在缺少能力时计为修复失败或通过。

组合清单的 includes 路径相对项目根目录；拒绝路径越界、循环引用和重复 case ID。
不会把 RAG 查询、MPC 性能采样或正反校准次数计入修复题目数。

## 无 LLM 的前置校验

```bash
uv run python scripts/run_benchmark.py --manifest benchmarks/cpp_professional_cases.json --list
uv run python scripts/check_cpp_robotics_fixtures.py
uv run pytest -q tests/test_cpp_robotics.py -m cpp
```

校准要求每个错误头文件失败且没有基础设施错误，参考修复通过。
脚本不覆盖实验 workspace；报告保存于 benchmarks/results/cpp-fixture-check-<uuid>/report.json，
每个 validator 调用另保留源哈希、oracle 哈希、工具链、构建和原生执行诊断。
该检查没有原生能力时必须失败，不能用 SKIPPED 冒充完成。

## Agent 修复实验：先小后大

先对单题做一次 smoke，确认模型、执行后端和官方门禁工作正常：

```bash
uv run python scripts/run_benchmark.py --manifest benchmarks/cpp_robotics_cases.json --case cpp_scalar_adl --strategy focused --repair-attempts 1 --max-loops 8 --tool-budget 12 --task-timeout 300
```

再运行第一批，每题独立重置并重复 3 次：

```bash
uv run python scripts/run_benchmark.py --manifest benchmarks/cpp_robotics_cases.json --repeats 3 --strategy focused --repair-attempts 1 --max-loops 8 --tool-budget 12 --task-timeout 300
```

完整 17 题必须先检查 Linux 固定依赖和错误基线，再运行；建议任务超时 600 秒，配置更改必须记录：

```bash
uv run python scripts/run_benchmark.py --manifest benchmarks/cpp_professional_cases.json --validate-fixtures
uv run python scripts/run_benchmark.py --manifest benchmarks/cpp_professional_cases.json --repeats 3 --strategy focused --repair-attempts 1 --max-loops 8 --tool-budget 12 --task-timeout 600
```

`--validate-fixtures` 可能按能力跳过不支持的题目，务必检查输出；正式运行入口会拒绝缺少能力的任务。
首次构建冷启动和缓存命中耗时不同，编译耗时不能直接当作 Agent 推理效率差异。
不自动启动批量付费 API 请求；先决定模型与预算，再执行修复实验。

## 指标与审计

沿用单题超时、实时进度、逐题落盘、checkpoint、独立最终验证和 false-positive 统计；新增：

- `source_diff`、修改前后 SHA-256：每次独立运行保存最终补丁，不依赖会被下一次覆盖的 workspace。
- `case_catalog` 与 SHA-256、`model_identity`：保存案例定义、fixture 哈希、模型标识及既有预算/工具链信息。
- `grouped.category/difficulty/domain_pack`：成功率、误判数、超时数、带错误运行数、平均 token/工具调用。
- `baseline_error`、`invalid_baselines`：环境错误或已通过的原始题目不启动 LLM 尝试，保存为无效基线；对比前必须为零。

当前成功率是“完成运行中通过官方验证的比例”，不是 pass@k；重复运行不是新题目，
报告中的预算合规统计为全部尝试的累计调用数。补丁只记录最终文本差异，尚未保存完整操作轨迹。
公开开发集、少量合成任务和参考解可见性限制，应随任何成功率一起披露。

## 后续里程碑

与 [Phase 7–9 开发计划](phase7-verification-driven-development.md) 对齐：先完成 M7.0
工作区/语义/隔离基础，再开展 M7.1 公平消融和 M7.2 真实仓库试点。当前公开 M2 fixture
仍保持原入口；新的独立评测协议不会自动把旧公开题转换成隐藏集。

1. M2.0：完成 8 题 smoke/3 次重复基线，按失败类别复盘，而非优先扩大题目数量。
2. M2.1：扩至 20–30 个专业任务，增加多文件、Eigen、模板编译错误、真实缺陷来源与独立保留任务；
   先验证参考修复，再冻结评测输入/协议，避免同题变体泄漏。
3. M2.2：按相同任务/模型/预算对照基础能力、文档 RAG、专业工具和 Validator。
   当前 focused 与 general 同时改变探索策略，不能直接作为严格 RAG 消融；四组消融尚未实现。
4. 根据失败分布进入 C++ 语义理解：compile_commands、类型/符号/引用、模板与跨文件上下文。

尚未实现：隐藏测试隔离、首次编译成功率事件采集、完整四组消融、真实仓库大规模修复和生产机器人验证。
