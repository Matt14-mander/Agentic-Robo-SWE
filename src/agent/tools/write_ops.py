"""写类工具 —— 让 LLM 能修改代码。

设计借鉴 Claude Code Edit / SWE-agent edit:
- 走 exact-string-match 搜索替换 (不是 unified diff), 实现简单且 LLM 容错好。
- 强制 ``old_string`` 在文件中**唯一**出现, 否则报错 —— 强迫 LLM 提供足够上下文,
  避免歧义改坏代码 (这是 SWE-agent ACI 的核心 lesson)。
- 写入前后给出文件大小 / 差异行数, 让 LLM 校验改动幅度。
"""

from __future__ import annotations

from langchain_core.tools import tool

from agent.tools._paths import relpath_for_display, resolve_within_root


@tool
def write_patch(path: str, old_string: str, new_string: str) -> str:
    """在仓库内文件中执行**一次**精确字符串替换。

    ``old_string`` 必须在文件中**唯一**出现; 出现 0 次或 2 次以上都会失败,
    此时请扩大 old_string 的上下文 (多包几行前后) 直到唯一定位。

    Args:
        path: 相对仓库根目录的路径。
        old_string: 要被替换的文本 (必须唯一)。
        new_string: 新文本。
    Returns:
        成功时返回 "OK: <path>  -<n_old_lines>/+<n_new_lines> lines";
        失败时返回 "ERROR: ..."。
    """
    try:
        p = resolve_within_root(path)
    except ValueError as e:
        return f"ERROR: {e}"
    if not p.exists():
        return f"ERROR: File not found: {path}"
    if not p.is_file():
        return f"ERROR: Not a regular file: {path}"
    if old_string == new_string:
        return "ERROR: old_string and new_string are identical; no-op refused."
    if not old_string:
        return "ERROR: old_string is empty; refusing (use a non-empty anchor)."

    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"ERROR: Cannot decode {path} as UTF-8 (binary file?)"

    count = text.count(old_string)
    if count == 0:
        return (
            f"ERROR: old_string not found in {path}. "
            "Tip: include enough surrounding lines so the snippet matches verbatim."
        )
    if count > 1:
        return (
            f"ERROR: old_string matches {count} places in {path}; must be unique. "
            "Tip: extend the snippet with more surrounding context to disambiguate."
        )

    new_text = text.replace(old_string, new_string, 1)
    p.write_text(new_text, encoding="utf-8")

    n_old = old_string.count("\n") + 1
    n_new = new_string.count("\n") + 1
    return f"OK: {relpath_for_display(p)}  -{n_old}/+{n_new} lines"
