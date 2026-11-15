# 索引同步失败恢复

## 问题与处理

旧版 `CodeIndex.sync()` 在计算新 embedding 前删除旧片段，`rebuild=True` 还会先清空
collection。函数改名后如果 embedding 失败，旧片段消失、新片段未写入；自动刷新依赖的
源码路径也随旧片段消失，导致后续查询持续返回空结果。

新的同步顺序：

1. 检查 workspace 并发现、切分本次指定范围的源码。
2. 保存待完成任务（源码根绑定、roots、rebuild），独立于片段元数据。
3. 分批计算全部待更新片段的向量，完成后才修改 collection。
4. 分批 upsert 新片段；全部成功后才删除不再属于本次范围的旧 ID。
5. 清除待完成记录，发布新 revision，使旧检索缓存失效。

embedding 在任意批次失败时，原有索引内容保持不变。写入或清理中途失败时，collection
可能暂时包含部分新旧片段，但待完成记录仍在。下一次查询（包括 `use_cache=False`）先按
原 roots 和 rebuild 重新同步，成功后才刷新源码和检索；恢复仍失败则报错，不返回缓存或
部分结果。恢复使用磁盘上的当前源码，而非失败时的源码快照。

## 持久化与兼容

`open_code_index()` 在 Chroma 存储目录内为每个 collection ID 创建独立的
`sync-<identity>.json`。临时文件写入并 flush/fsync 后原子替换正式记录；提交完成后删除。
不同 workspace 和 collection 不共享记录；复用 collection 名称创建新 collection 时，也
不会沿用已删除 collection 的任务。记录校验版本、字段和源码根绑定；损坏时查询报错，
显式重新执行 sync 可以覆盖损坏记录并恢复。

直接构造 `CodeIndex(collection)` 时，待完成任务默认只保存在内存；需要重启恢复的调用方
应使用 `open_code_index()`，或显式传入唯一的 `sync_state_path`。旧索引无需迁移；没有
待完成记录时继续沿用已有源码刷新逻辑。已被旧版失败同步彻底清空、且没有恢复记录的索引
仍需要手工运行一次 sync，本次修复无法推断已经丢失的历史索引范围。

`rebuild=True` 仍重新计算所有片段，但不预先清空 collection。`IndexStats.deleted` 现在
仅统计实际删除的过期 ID；原 ID 被重新编码/upsert 不计为删除。

## 边界

- 这是可重试协议，不是 Chroma 多批次 ACID 事务或完整旧版本快照回滚。
- 同一个 CodeIndex 实例通过锁串行化 sync 和 search。仍要求每个 collection 单写者，
  不保证多个进程或多个实例并发同步时的事务隔离；直接读取底层 collection 绕过恢复检查。
- 全部待更新向量需暂存内存，内存开销随本轮更新片段数增长。
- 源码读取/切块在持久化任务前失败时，不修改已有 collection；修复输入后重新执行 sync。

## 回归验证

2026-11-13：新增 27 个恢复测试，相关 RAG 测试合计 79 passed、1 deselected（原生 C++）。
两条索引清空回归在旧实现上均失败，修复后通过。完整离线测试为 432 passed、1 skipped、
46 deselected（排除 e2e、cpp、isolation、simulation、autodiff_codegen、pinocchio、
crocoddyl）；结果保存在 `.agent_state/sync-recovery-full.xml`。全项目 Ruff 和 mypy
（80 个源码文件）通过。恢复测试已加入安装 RAG 依赖的 workspace CI 任务。

故障注入覆盖普通同步和 rebuild、首批及后续批次推理失败、首次及部分 upsert 失败、
删除前后失败、恢复记录写入/清理失败、首次建索引失败、同进程及重新打开后的自动恢复、
持续故障下缓存与直接检索均拒绝返回结果。真实 Chroma 测试验证部分写入后的 reopen、
范围切换和 workspace 隔离。

```bash
uv run pytest -q tests/test_rag_sync_recovery.py tests/test_rag.py \
  tests/test_workspace_rag.py tests/test_rag_evaluation.py tests/test_rag_chunking.py -m "not cpp"
```
