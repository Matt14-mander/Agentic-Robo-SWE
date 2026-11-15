# Token 长度切块与父上下文

## 行为

原 Python AST 片段和 C++ 120 行窗口仍作为父片段。语义 embedding 使用模型自带的
tokenizer 对父片段计数，预算包括特殊 token；不再假定固定行数能放入 512 token。
未超预算的父片段直接索引，超预算的片段切成多个子片段，默认重叠不超过 32 token
（小预算时不超过可用源码预算的 1/8）。优先沿换行切分，单行过长时沿字符边界继续切分。

切块使用独立、关闭 padding 和 truncation 的 tokenizer。每个实际源码子串都会重新编码
检查长度，处理子串边界改变分词的情况；源码直接切片，不使用 decode 重新生成代码。
切分不会丢掉父片段中的字符，支持 Unicode 和长单行。新增子片段保留父 ID、父内容指纹、
符号、路径及 1-based 行列范围（结束位置包含在内）。

Hashing 离线回退没有模型 token 上限，继续使用原有父片段，不加载 tokenizer 或模型。
本轮解决文档索引截断；超过配置上限的查询文本仍沿用原模型截断行为。

## 检索与上下文

向量检索在子片段上评分；关键词检索在实际索引片段上评分。各自先按父 ID 去重，再做
混合排名，同一父片段只占一个结果名额。向量候选中的重复子片段过多时，扩大候选数，
直到获得足够的不同父片段或遍历完索引。混合检索保留向量命中的子片段作为展示证据。

`SearchResult.content` 是命中的子片段，`parent_context` 提供完整父片段及其行号；
原短片段/Hashing 结果保持原内容，`parent_context` 为 None。父上下文不是整个文件，
也不保证是完整长函数：它仍受原 AST/行窗口切块的边界约束。

父内容不在每个 Chroma 子片段中重复保存，避免长单行导致存储量成倍膨胀。索引维护父片段
映射；重新打开后从经过 workspace 安全校验的当前源码恢复。子片段的索引指纹包含父内容
指纹，所以只修改父上下文也会刷新元数据并使缓存失效。源码不可读时不返回旧上下文。

Agent 工具先显示命中子片段，再在每条结果原有 4,000 字符的源码预算内显示父上下文。
空间不足时父内容明确标注为节选并给出完整行号范围，供 `read_file_chunk` 补读；不会用
父片段开头替换命中片段，导致尾部证据再次消失。结构化 SearchResult 仍保留完整父上下文。

## 索引版本与评测

语义 encoder identity 加入 `token-children-v1`，为新切块策略创建隔离的 collection，
避免混用旧截断向量。升级后需对目标 roots 运行一次现有 sync 命令。旧 collection 不删除，
Hashing collection 不变；不应把旧 collection 的向量直接复制到新命名空间。

```bash
uv run python scripts/index_codebase.py --embedding semantic --root src
```

`retrieval_manifest.source_revision` 基于未拆分的父片段，用于验证 Hashing 与语义检索
读取同一源码快照；`index_revision` 基于实际索引片段，`parent_count` 和 `chunk_count`
分别记录父片段数和实际索引数。不同切块布局不再被误判为源码发生变化。

## 验证

对固定源码提交 `8d880db` 的 `src/`，使用原模型 tokenizer：545 个父片段中有 62 个超过
512 token。拆分后为 636 个实际索引片段，超预算数量为 0，最长恰好 512 token。
完整本地模型编码结果见 [固定源码覆盖报告](experiments/rag-token-coverage-20261113.json)。
本机使用固定 ONNX 模型完成 636 个片段编码，`truncated_texts=0`，耗时 157.923 秒；
这次时间只包含编码，不包含 Chroma 写入，不与旧的整轮索引耗时直接比较。
这验证了覆盖与长度约束，不代表新的检索准确率或修复成功率结论；旧检索评测表保留为历史结果。

回归覆盖特殊 token 的预算边界、连续长行、换行、缩进、Unicode、所有字符覆盖、尾部检索、
父片段去重、上下文更新导致的缓存失效、旧片段重切分、C++ workspace 更新/删除、Chroma
重新打开，以及固定本地真实模型对子片段编码不产生额外截断。CI 使用离线构造的 tokenizer
执行必需回归；没有本地权重时真实模型用例明确跳过，不自动下载。

2026-11-13：新增 22 个用例，相关回归 83 passed、1 deselected（原生 C++）；完整离线
回归 436 passed、1 skipped（Windows 上的 POSIX 权限契约）、46 deselected（e2e、cpp、
isolation、simulation、autodiff_codegen、pinocchio、crocoddyl）。本机真实模型用例实际执行；
全项目 Ruff、mypy（81 个源码文件）和 diff whitespace 检查通过。
完整测试报告位于 `.agent_state/token-chunk-full.xml`。

```bash
uv run pytest -q tests/test_token_chunking.py tests/test_code_embedding.py tests/test_rag.py \
  tests/test_workspace_rag.py tests/test_rag_chunking.py tests/test_rag_evaluation.py -m "not cpp"
```
