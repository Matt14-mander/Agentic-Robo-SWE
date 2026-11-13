# 外部 C++ workspace RAG

## 设计与边界

外部仓库检索复用 `CodeIndex` 的可切换 embedding、BM25 风格评分、RRF 融合和
版本化缓存，通过显式 `Workspace` 绑定替换 Harness 的源码发现、读取与切块。

- 索引存储于 `workspace.artifacts/rag/<root-identity>/chroma/`，不写入目标仓库。
  路径身份和 collection 元数据绑定目标根目录；共用 artifacts 的两个仓库仍分开存储。
  Agent 的索引对象缓存按完整 Workspace 分开，跨 workspace 使用旧对象会被拒绝。
- 默认扫描目标仓库；可用 `--root` 多次指定子目录或文件。复用 workspace 的私有路径、
  符号链接/junction、普通文件及 20,000 文件 / 100 MB 限制；单个索引文本上限 1 MB。
- 支持 `.c/.cc/.cpp/.cxx/.h/.hh/.hpp/.hxx/.ipp/.tpp/.inl`、`.md/.markdown`、
  `.cmake` 和 `CMakeLists.txt`。不自动读取系统头文件、编译数据库引用的外部路径或远程文档。
  输入必须为 UTF-8 文本；超限或二进制输入会报错。当前沿用 workspace 固定排除目录，
  不解析 `.gitignore`，需要缩小范围时使用显式 `--root`。
- 基础版按 120 行、15 行重叠切块，保存相对路径、行号、类型和内容哈希。
  C++ 片段使用 `<file>` 符号标识，明确不代表 AST 函数边界或 clangd 符号解析。
  Harness 的 Python AST 切块保持不变。
  使用语义 embedding 时，超 token 预算的窗口继续拆成子片段，命中后带回父窗口上下文；
  子片段带精确行列范围，同一父窗口不重复占用结果名额。详见 [Token 切块](rag-token-chunking.md)。
- 建索引是显式操作；查询自动刷新已索引文件，失败时不返回旧缓存。
  新文件以及删除/清空后重新加入的文件需要再次同步。切换源码根时，sync 以本次指定范围为准。
  热查询通过文件变更元数据做增量检测，关键词使用倒排索引；返回前复核命中文件内容，
  包括缓存命中与父片段上下文。跨进程通过 collection 文件锁和 generation 标记失效，
  具体边界、严格模式及性能数据见 [增量检索](rag-incremental-retrieval.md)。

## 与 clangd 的分工

Agent 可通过工具的 `strategy` 参数选择 lexical/vector/hybrid，也可通过 `--rag-strategy`
设置会话默认值；不会改变 workspace 边界或自动启用 C++ 执行权限。独立的 20 文件、32 查询
公开评测及配置示例见 [检索策略与评测](rag-strategy-selection.md)。

1. 未知实现位置或概念查询：`search_code_knowledge` 返回当前 workspace 的候选片段。
2. 精确字符串：grep；已知路径：`read_file_chunk`，确认实际符号行列。
3. 已开启 C++ 工具且完成 `cpp_build(configure_only=true)`：通过
   `cpp_semantic_query` 获取定义、引用、hover 类型或诊断。

RAG 不执行仓库代码、不触发 CMake、不启动 clangd，因此无需 `--trusted-local` 或执行镜像，
可以在只读会话使用。clangd 仍沿用现有执行授权、编译数据库与 1-based Unicode 输入位置约定。
检索片段起始行不是符号位置；不得直接把它当作 clangd 查询坐标。编译器诊断和验证器结果
仍是修复正确性的证据，检索分数不是正确性概率。

## 使用

```bash
uv sync --extra rag
uv run python scripts/index_codebase.py --workspace /path/to/repo \
  --root src --root include --query "joint torque saturation"

# 自定义 artifacts 时，建索引和 Agent 会话使用同一个目录
uv run python scripts/index_codebase.py --workspace /path/to/repo \
  --artifact-root /path/to/artifacts --query "controller" --strategy lexical
```

省略 `--root` 时索引整个目标仓库内支持的文件；显式指定的根必须存在。
外部模式不接受 `--persist-dir`，存储位置由 Workspace 决定，防止覆盖 Harness 索引。
`search_code_knowledge` 根据当前会话自动选择索引，无需模型传递 workspace 路径。
未建立索引时返回建索引提示；安装缺少 Chroma 时返回 optional dependency 提示。

## 验收

2026-11-12，本轮完成：

- 相关回归：87 passed、2 skipped，覆盖 Python RAG、C++ workspace 检索、缓存、工具调度、
  workspace 边界、现有 C++ 协议及持久化。两个跳过项为 Windows 上的 POSIX 权限测试，
  以及缺少原生 clangd/CMake/Ninja 的衔接集成测试。
- 完整离线回归：405 passed、1 skipped、46 deselected，排除 e2e、cpp、isolation、
  simulation、autodiff_codegen、pinocchio、crocoddyl 标记；报告位于
  `.agent_state/workspace-rag-full.xml`。此结果不包含原生 C++ 或真实模型能力验收。
- 真实 Chroma 用例通过：同一个 artifacts 下的两个仓库分别存储，重新打开索引后自动识别
  文件修改，另一仓库不受影响；collection 绑定不符时拒绝读取。
- 公开 `examples/external_cpp` CLI smoke：4 files / 4 chunks；`joint error` 查询返回
  `controller.cpp`、`controller.hpp` 和 `main.cpp` 等候选，重复查询命中缓存。
  产物目录 `.agent_state/workspace-rag-cli-smoke`，未调用模型或执行目标仓库代码。
- 全项目 Ruff 与 mypy（79 个源码文件）通过；diff whitespace 检查通过。
- 已增加 RAG → 读取实际符号位置 → clangd definition 的原生/容器集成测试，并接入
  `workspace-isolation-checks` CI。当前 Windows 仅发现 g++，缺 clangd/CMake/Ninja；
  本机跳过真实 clangd 衔接，未声称通过。CI 配置已更新，尚未触发远程运行。

复现离线回归：

```bash
uv run pytest -q -m "not e2e and not cpp and not isolation and not simulation and not autodiff_codegen and not pinocchio and not crocoddyl"
uv run pytest -q tests/test_workspace_rag.py
```

后续已接入 [本地代码语义 embedding 与离线回退](semantic-code-embedding.md)，可用同一
workspace 和 `--embedding semantic` 单独建立语义 collection。C++ AST 切块、自动 clangd
图扩展和新一轮真实模型消融仍待后续验证，不改变既有 M7.1 四组实验结论。
