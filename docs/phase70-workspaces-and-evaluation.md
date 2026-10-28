# M7.0：外部工作区、C++ 工具与独立评测

路线与后续验收见 [Phase 7–9 开发计划](phase7-verification-driven-development.md)。

## 1. 工作区契约

`agent.workspace.Workspace` 将三个概念分开：

- Harness 安装根：内置 Domain Pack、依赖与旧 fixture benchmark 的根目录。
- `--workspace`：待修复仓库；模型读取、搜索和补丁路径只相对此目录。
- `--artifact-root`：checkpoint、构建与运行产物；必须在目标仓库之外且不包含目标仓库。
  默认位于 Harness 的 `.agent_state/workspaces/<目标路径哈希>/`。

ContextVar 传递会话上下文，工具线程继承同一上下文，不修改全局 PROJECT_ROOT。
checkpoint 保存目标根、产物根和执行模式绑定，恢复必须提供相同参数。

```bash
uv run python scripts/run_agent.py "解释这个仓库的控制器结构" \
  --workspace /path/to/robot-repo --artifact-root /path/to/run-state \
  --read-only --thread-id robot-inspect

uv run python scripts/run_agent.py "定位并修复连续关节跨界错误" \
  --workspace /path/to/robot-repo --artifact-root /path/to/run-state \
  --trusted-local --cpp-tools --hitl --thread-id robot-repair
```

`--trusted-local` 显式允许仓库代码/CMake/clangd 在宿主运行，不提供宿主文件隔离。
容器模式使用 `--execution-image sha256:<完整镜像ID>`，两者互斥；外部执行未配置时拒绝。
旧的无 `--workspace` 调用保留原行为。外部会话不复用 Harness 的 Python RAG/回答缓存或
内置 fixture Domain Pack。工具拒绝目标外路径、符号链接/junction、`.git`、`.env*`、
`.agent_state` 等私有元数据；调度端再次检查工具权限，不能靠伪造调用绕过只读模式。

## 2. C++ 工具

当前支持根 CMakeLists、Ninja、单一 Debug 构建树，数据库位于
`<artifact-root>/cpp-build/compile_commands.json`。

| 工具 | 行为 |
|---|---|
| `cpp_build(target, configure_only)` | 固定参数配置 CMake、导出数据库、可选构建；每阶段最多 180 秒 |
| `cpp_compile_commands()` | 检查并返回编译条目，不直接执行条目 command |
| `cpp_semantic_query(path, operation, line, column)` | clangd 定义、引用、hover 类型、诊断 |

无 LLM 入口：

```bash
uv run python scripts/run_cpp_tools.py build \
  --workspace examples/external_cpp --trusted-local --target controller
uv run python scripts/run_cpp_tools.py commands \
  --workspace examples/external_cpp --trusted-local
uv run python scripts/run_cpp_tools.py definition \
  --workspace examples/external_cpp --trusted-local --path main.cpp --line 9 --column 22
```

输入为 1-based Unicode 字符位置；输出为 LSP 的 0-based UTF-16 坐标。
每次重新建立 clangd 会话，最多解析 32 个编译单元并返回覆盖范围，引用不宣称全库完备。
查询进程预算 45 秒（包括协议写入），单文件最多 1 MB、协议帧最多 2 MB，超时/缺工具
显式报错。只读模式不开放会执行进程的 C++ 工具。
容器 CMake/clangd 共享 `/workspace` 和 `/build` 映射，源文件经快照过滤后只读挂载。
更换执行后端或镜像必须使用新产物目录；项目依赖需预装在镜像，运行期间不联网安装。

## 3. 独立最终评测

```text
私有 oracle prepare → cases
候选 runner + 只读源码快照 + cases → observations
私有 oracle judge + cases + observations → passed / checks
```

runner 与 oracle 必须是不同的固定镜像 ID/digest。runner 与模型执行镜像不得包含 oracle、
参考解或预期答案；镜像作者必须审计内容，两个不同哈希本身不能证明没有复制同一秘密。

候选容器无网络、无宿主环境凭据、无 Docker socket、非 root、只读根文件系统和源码，
有独立 tmpfs 与 CPU/内存/PID/输出/时间限制，结束后清理本次命名容器。
oracle 没有候选源码挂载，不能 import/运行候选代码，只按数据协议处理观察值；
领域适配器仍需验证数据形状、数值和语义。

最终评分是操作者入口，未注册为模型工具。私有 spec 和 raw artifacts 必须位于候选目录
之外；结果只公开状态、检查数、指纹和产物位置，不向修复环路泄露隐藏输入。
拒绝候选自报 passed、数量不符、重复 JSON key、NaN/Infinity、异常退出；
缺 Docker 或镜像直接失败，不退回本地。

结果绑定源码、规格 SHA-256 和 runner/oracle 镜像 ID。评测中源码变化导致失败；
`receipt_matches()` 检查历史通过报告是否仍对应当前源码和规格。
旧 M1/M2 图内消息文本判定不在本轮替换，旧 benchmark 保留独立最终复验；统一证据协议
列入 M8.0。目录内 `.env*` 与常见私有目录会过滤，但操作者仍应准备不含秘密的候选仓库。

## 4. 公开示例与原生验收

`examples/external_cpp` 是故意缺少 angle wrapping 的多文件控制器；`examples/isolation`
提供 runner/oracle。示例完全公开，只验证架构，不是隐藏集或真实机器人安全证明。

Linux/WSL 显式构建（构建需网络，运行禁网）：

```bash
docker build -t robo-phase70-runner -f examples/isolation/runner.Dockerfile examples/isolation
docker build -t robo-phase70-oracle -f examples/isolation/oracle.Dockerfile examples/isolation
export ROBO_TEST_RUNNER_IMAGE=$(docker image inspect robo-phase70-runner --format '{{.Id}}')
export ROBO_TEST_ORACLE_IMAGE=$(docker image inspect robo-phase70-oracle --format '{{.Id}}')
uv run pytest -q tests/test_repository_integration.py
```

基础镜像标签仅用于开发示例；正式任务应冻结 base digest、软件依赖和镜像构建清单。
执行仅接受已经存在的完整镜像 ID/digest，使用 `--pull=never`。

操作者在候选目录外保存 spec，替换两个真实镜像 ID：

```json
{
  "schema_version": 1,
  "task_id": "public-joint-wrap-demo",
  "runner_image": "sha256:<runner完整ID>",
  "oracle_image": "sha256:<oracle完整ID>",
  "runner_command": ["python3", "/opt/robo/runner.py"],
  "oracle_command": ["python3", "/opt/robo/oracle.py"],
  "timeout_seconds": 120
}
```

```bash
uv run python scripts/evaluate_repository.py \
  --workspace /path/to/candidate --spec /path/to/private/spec.json \
  --output /path/to/private/results
```

退出码 0=通过，1=判定完成但未通过，2=协议/环境/执行错误。POSIX 每次私有运行目录为
0700；Windows 私有目录 ACL 由操作者配置。本地可信模式不适用于含私有材料的非可信执行。

## 5. 验收记录

2026-10-28，本轮完成：

- 最终 Windows 离线回归：328 passed、1 skipped、26 deselected（排除 e2e、cpp、isolation）。
  跳过的是 POSIX 可执行权限测试，已在 Linux 通过。原有领域、路由、缓存和持久化回归通过；
  ruff、mypy 与 diff whitespace 检查通过。记录：`.agent_state/phase70-full.junit.xml`。
- WSL Ubuntu 22.04 / Python 3.14.7 / GCC 11.4 / clangd 14：原生多文件 CMake +
  clangd 集成首轮 1 passed，覆盖定义、跨文件引用、类型与编译错误诊断。
  最新代码另完成原生 C++、工作区、协议与子进程边界合计 25 passed；结果保存为
  `.agent_state/phase70-linux-native.junit.xml`。
- WSL Docker 29.1.3：实际容器集成 5 passed，覆盖容器 C++ 工具、broken/reference
  独立判分与旧证据失效、隔离访问、伪造 verdict 拒绝、超时销毁。
- 新增 CI `workspace-isolation-checks`，显式构建标准 Dockerfile 镜像并运行上述协议和
  原生容器用例。CI 配置已添加，尚未在远程 GitHub 上触发。

本机 Docker Hub 拉取超时，实际验收镜像使用现有 WSL 的公开系统二进制/头文件组装本地
base，再分别只复制 runner.py 或 oracle.py；未复制工作区、宿主 home、凭据或私有材料。
这验证了运行隔离和协议，没有验证标准 Dockerfile 的联网构建；后者交由上述 CI/可联网环境。
镜像 ID、验收命令与原始输出保留在本次本地 `.agent_state/` 和 WSL 测试产物中。

最终复测环境阻塞：后续 WSL 启动时 `/dev/sdc` 根文件系统显示
`ro,relatime,discard,errors=remount-ro,data=ordered`，Docker 日志为
`chmod /var/lib/docker: read-only file system`。未执行磁盘修复或强制重挂载，也不推断故障原因。
五项容器通过记录早于最后的快照可执行权限/目录过滤、Python 3.11 reparse 检测和 LSP
写入超时防护调整；这些增量通过 Windows 定向回归及上述 Linux 原生 25 项复验
（使用项目内临时目录，禁用 pytest 文件描述符捕获），但完整容器重验需先恢复 WSL 可写状态。
正式环境验收不得省略这一步，也不能用测试跳过代替通过。

没有调用付费 LLM，没有重新测量 Phase 6 性能，也没有将此公开示例算作 M2 隐藏集。
