"""Tool 模块 —— 暴露 @tool 装饰的可调用集合, 给 ToolNode 使用。

新代码请用 ``from agent.tools import ALL_TOOLS``;
Phase 1 老测试通过 ``agent.tools.file_ops.read_file`` 仍能工作 (向后兼容)。
"""

from agent.tools.exec_ops import execute_python
from agent.tools.read_ops import grep_codebase, list_dir, read_file_chunk
from agent.tools.write_ops import write_patch

# 注册顺序决定 LLM system prompt 里列出的顺序, 把"探索类"放前面、"写操作"放后面,
# 引导 LLM 先看清再动手。
ALL_TOOLS = [
    list_dir,           # 探索类工具放前面, 帮助 LLM 先了解环境
    read_file_chunk,    # 也是探索类工具, 让 LLM 能查看文件内容
    grep_codebase,      # 也是探索类工具, 让 LLM 能搜索代码库
    execute_python,     # 执行类工具, 让 LLM 能运行 Python 代码 (如测试脚本)
    write_patch,        # 写操作类工具放最后, 因为它是"动手改代码", 需要 LLM 确认好前面信息后再用它, 减少误用的风险
]

# M1.2 benchmark tasks already provide an exact file and validator. Removing
# discovery tools from the bound schema prevents redundant repository traversal.
FOCUSED_BENCHMARK_TOOLS = [
    read_file_chunk,
    execute_python,
    write_patch,
]

__all__ = [
    "ALL_TOOLS",
    "FOCUSED_BENCHMARK_TOOLS",
    "execute_python",
    "grep_codebase",
    "list_dir",
    "read_file_chunk",
    "write_patch",
]
