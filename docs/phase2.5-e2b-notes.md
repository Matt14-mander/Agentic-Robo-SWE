# Phase 2.5 深挖笔记 —— E2B 集成的三个不显然设计

面向: 已经跑通 Phase 2.5、想理解 `src/agent/tools/exec_ops.py` 里"为什么这么写"的读者。

---

## 一、E2B `Execution` dataclass 细节

对应代码: `src/agent/tools/exec_ops.py:182-193`。

```python
stdout = "".join(execution.logs.stdout)
stderr = "".join(execution.logs.stderr)
returncode = 1 if execution.error else 0
if execution.error:
    stderr += (
        f"\n[E2B error] {execution.error.name}: {execution.error.value}\n"
        f"{execution.error.traceback}"
    )
```

### `Execution` 对象的完整字段

`sbx.run_code(...)` 返回的 `Execution` 是 E2B 定义的 dataclass, 关键字段:

| 字段 | 类型 | 说明 |
|------|------|------|
| `.logs.stdout` | `list[str]` | **每个元素是一行**(通常含 `\n`), 不是完整字符串 |
| `.logs.stderr` | `list[str]` | 同上 |
| `.error` | `Optional[Error]` | `None` 表示无异常; 非空则含 `name` / `value` / `traceback` |
| `.results` | `list[Result]` | 富输出(matplotlib 图、HTML、DataFrame 表格等) —— **我们没用** |
| `.text` | `str` | 最后一个表达式的求值结果(Jupyter cell 尾行返回值) |
| `.execution_count` | `int` | Jupyter 风格递增计数器 |

### 为什么 `logs.stdout` 是 `list[str]` 而不是 `str`?

因为 E2B 底层是 **Jupyter Kernel 协议**: 内核每输出一行就发一个 `stream` 消息, SDK 把每条消息按顺序追加进 list。**这是 IO 流的原始形态**, 尚未合并。

所以我们要 `"".join(execution.logs.stdout)` 手动合并 —— 不是我们的输出格式怪, 是 E2B 保留了流式语义。

**副作用**: 如果代码没换行(比如 `print("a", end="")`), 一条 list 元素可能不含 `\n`; `"".join` 恰好正确处理, 无需 `"\n".join`。

### `error` 字段的三段式结构

E2B 抓 Python 异常时会拆成三块, 模拟 IPython 的 `ultratb`:

```python
execution.error.name       # "ValueError"
execution.error.value      # "invalid literal for int() with base 10: 'foo'"
execution.error.traceback  # 完整 Traceback, 含 "  File ..." 行
```

**为什么拼到 stderr 尾部**? 因为 local backend 的 subprocess 返回的 stderr 里天然就有完整 traceback。**我们把两条路径的输出结构对齐**, 让 LLM 看到的错误格式一致, 切 backend 不用改 prompt。

### 为什么 `returncode = 1 if execution.error else 0`

E2B **没有** `returncode` 概念(Jupyter Kernel 里不存在这个东西)。但我们要保持与 local backend 的 `_format_output` 接口一致(`exit_code:` 那一行), 只好合成一个二值: **有异常就是 1, 否则就是 0**。这是一个"为了统一接口而做的假信号", 在多 backend 抽象里是常见模式。

---

## 二、为什么不用 context manager

对应代码: `src/agent/tools/exec_ops.py:147-180`。

```python
try:
    sbx = Sandbox.create(api_key=api_key, timeout=300)
except Exception as e:
    return f"ERROR: failed to create E2B sandbox: {type(e).__name__}: {e}"

try:
    if needs_project:
        ...upload...
    execution = sbx.run_code(...)
except Exception as e:
    return f"ERROR: E2B run_code failed: {type(e).__name__}: {e}"
finally:
    try:
        sbx.kill()
    except Exception:
        pass
```

Pythonic 写法应该是 `with Sandbox.create(...) as sbx:`。为什么没这么写?

### 原因 1: `create()` 失败时**不需要清理**, 普通 `with` 会把两件事绑死

`with X() as ...:` 的语义是"构造成功后必然清理"。如果 `Sandbox.create` 本身抛异常, 没有实例可以清理 —— 但 `with` 语法要求你先绑名字。所以你会写:

```python
sbx = Sandbox.create(...)   # 这里抛异常, 完全没进入 with
with sbx:
    ...
```

这跟我们现在的写法**一模一样**, `with` 完全没占便宜。

### 原因 2: 我们要**吞掉清理错误**, `with` 让这变难

`sbx.kill()` 可能因网络抖动、E2B 平台故障失败。这时如果 `run_code` 已经成功返回结果, 我们**希望这个 kill 失败静默处理** —— 不能因为清理失败就丢弃已拿到的执行结果。

```python
finally:
    try:
        sbx.kill()
    except Exception:
        pass          # 明确的语义: 用完了、清理失败不影响返回值
```

如果用 `with`, `__exit__` 抛异常会向上传播, 可能把 `run_code` 的返回值扔掉, 或者**遮盖原始异常**(Python 会打印 "During handling of the above exception, another exception occurred")。用 try/finally 显式控制这个语义。

### 原因 3: `e2b-code-interpreter` 的 `Sandbox` 支持 `with`, 但**语义与我们要的不同**

E2B SDK 的 `Sandbox` 确实实现了 `__enter__` / `__exit__`, 但:

- `Sandbox.create(...)` 是**同步构造 + 已启动**的 factory method, 直接用 `with` 需要把两步耦合
- SDK 早期版本(0.0.x)曾多次调整 sandbox 生命周期 API(有过 `kill` vs `close` vs `destroy` 的命名折腾) —— 用 try/finally + `try: sbx.kill() except: pass` 是版本迁移最稳的写法

### 原因 4: **语义解耦**

三段结构对读者非常清楚:

```
try: create           → 失败? 直接返回 ERROR, 无需清理
try: use              → 上传 + 执行, 失败也进入 finally
finally: cleanup      → 无论成功失败都跑
```

用 `with` 会把 create 和 cleanup 绑在同一层缩进, 读代码时"创建可能失败但不影响清理"的意图反而变模糊。

### 什么时候会想切回 `with`?

如果实现 **sandbox 池化 / 复用**(Phase 3+ 可能做): 一个 sandbox 服务多次工具调用, 那时 sandbox 是**长期资源**, 用 `contextlib.ExitStack` 组合管理会更清爽。当前每次调用一个新 sandbox, `create() → use → kill()` 是**短事务模式**, try/finally 反而更贴合。

---

## 三、按需上传 regex 的取舍

对应代码: `src/agent/tools/exec_ops.py:40-43` 和 `:144`。

```python
_PROJECT_IMPORT_RE = re.compile(
    r"^(?:from|import)\s+(agent|tests)(?:[.\s]|$)", re.MULTILINE
)

needs_project = bool(_PROJECT_IMPORT_RE.search(code))
```

### 逐字段拆解

| 片段 | 作用 |
|------|------|
| `^` + `re.MULTILINE` | 匹配**任意行**的开头(不加 MULTILINE 只匹配整段开头) |
| `(?:from\|import)` | 非捕获组, 匹配 `from X` / `import X` 两种句法 |
| `\s+` | 至少一个空白, 保证 `fromagent` 这种"疑似前缀"不误伤 |
| `(agent\|tests)` | 命名空间白名单(顺便记录到 group 1, 虽然我们没用) |
| `(?:[.\s]\|$)` | 后面必须是 **`.`(`import agent.state`)、空白(`import agent as a`)或行尾** —— 防止 `agent_helper` 这种前缀相同的误命中 |

### 具体行为

| 输入 | 命中? | 备注 |
|------|------|------|
| `import agent` | ✓ | 顶层导入 |
| `from agent.state import X` | ✓ | 典型场景 |
| `import agents_lib` | ✗ | 前缀命名空间保护生效 |
| `    from agent.state import X` | **✗ 漏检** | ^ 不允许行首空白 |
| `# from agent.state import X` | ✗ | 注释开头是 `#` 不是 `from` |
| 多行字符串里独立一行以 `from agent` 开头 | **✓ 误检** | 无法从字面看出这是字符串内容 |
| `__import__("agent.state")` | ✗ | 动态导入不认 |
| `importlib.import_module("agent.state")` | ✗ | 同上 |

### 三个取舍的坐标

**Trade-off 1: False positive vs False negative**

规则设计得**严格**(`^` 强制行首、`agent` 白名单、后接 `[.\s]` 边界), 意味着:

- **False negative**(该上传没上传) —— **可自愈**: LLM 在 E2B 输出里看到 `ModuleNotFoundError: No module named 'agent'`, 下轮会调整 code 结构(把 import 提到顶层)
- **False positive**(不该上传却上传了) —— **静默浪费**: 每次多花 ~1s 网络时间, 用户看不到、LLM 看不到, 无从纠正

**结论**: 可自愈的错误比静默浪费更好。所以**倾向严格** —— 宁可漏放上传、由错误信息触发调整。

**Trade-off 2: 正则 vs AST**

真正正确的做法是 AST 解析:

```python
import ast
try:
    tree = ast.parse(code)
    needs_project = any(
        (isinstance(n, ast.Import) and any(a.name.split(".")[0] in ("agent", "tests") for a in n.names)) or
        (isinstance(n, ast.ImportFrom) and n.module and n.module.split(".")[0] in ("agent", "tests"))
        for n in ast.walk(tree)
    )
except SyntaxError:
    needs_project = False
```

优点: 覆盖缩进 import、语法完全准确、能识别 relative import、零字符串误检。

缺点: LLM 生成的代码**可能有语法错误**, `ast.parse` 抛异常怎么办? 兜底 `False` 会漏检; 兜底 `True` 就等于每次都上传。

我们选正则的**唯一理由**: **LLM 的 code 允许语法错误**(比如它写了半截、留了错行让沙盒报错来找 bug)。这时 AST 崩, 正则还能给出合理判断。这是"LLM 生成代码"这个特殊上下文才成立的取舍。

**Trade-off 3: 白名单粒度**

规则里写死了 `(agent|tests)` 两个顶层包名。如果项目将来加一个 `robotics_lib/` 需要上传, 正则得手工改。可以做成配置:

```python
_PROJECT_PACKAGES = ("agent", "tests")
_PROJECT_IMPORT_RE = re.compile(
    rf"^(?:from|import)\s+({'|'.join(_PROJECT_PACKAGES)})(?:[.\s]|$)",
    re.MULTILINE,
)
```

暂时没做, 是因为 Phase 2.5 阶段项目结构还在动。**过早参数化 = 过度设计**。

### 学习要点

**给 LLM agent 做"启发式短路优化", 标准是"错误可自愈"**:

| 场景 | 处理方式 |
|------|---------|
| 判错造成静默副作用(浪费时间、多花钱、留垃圾) | 严格标准, 宁漏勿冤 |
| 判错造成明显错误(LLM 收到清晰报错) | 宽松标准, 让 LLM 自己纠正 |
| 判错会导致数据损坏 / 不可逆操作 | 绝不启发式, 必须精确 |

按需上传属于第二类 —— 严格 regex + 让 error message 承担剩余覆盖。这是 agent 工具设计里一个反复出现的模式。

---

## 快速对照表(源码 anchor)

| 代码位置 | 学习点 |
|---------|--------|
| `exec_ops.py:41-43` | 严格 regex 的字段拆解 |
| `exec_ops.py:144` | needs_project 的单点决策 |
| `exec_ops.py:147-150` | create 与 use 分离的第一段 try |
| `exec_ops.py:175-180` | finally + 嵌套 try 的静默清理 |
| `exec_ops.py:182-193` | logs.stdout 拼接 + error 三段展开 |
| `exec_ops.py:185` | 无 returncode 的合成 |
