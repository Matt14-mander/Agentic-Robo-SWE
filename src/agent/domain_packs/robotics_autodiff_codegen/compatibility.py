"""Conservative source diagnostics for common C++ AD incompatibilities."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from agent.tools._paths import relpath_for_display, resolve_within_root


@dataclass(frozen=True)
class CompatibilityFinding:
    rule: str
    severity: str
    line: int
    message: str
    excerpt: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_RULES: tuple[tuple[str, str, re.Pattern[str], str], ...] = (
    (
        "hardcoded_scalar",
        "error",
        re.compile(r"\b(?:const\s+)?double\s+[A-Za-z_]\w*|(?:array|vector)\s*<\s*double\b"),
        "Model data uses double instead of the template Scalar type.",
    ),
    (
        "eigen_scalar_loss",
        "error",
        re.compile(r"Eigen::(?:Vector\d+[fd]|Matrix\s*<\s*double\b)"),
        "Eigen storage fixes the scalar to double and breaks AD propagation.",
    ),
    (
        "unsafe_value_conversion",
        "error",
        re.compile(r"static_cast\s*<\s*double\s*>|\b(?:Value|value)\s*\("),
        "Converting an active value to double detaches it from the AD graph.",
    ),
    (
        "data_dependent_branch",
        "warning",
        re.compile(r"\b(?:if|while)\s*\([^\n]*(?:x|q|state|input)\s*\["),
        "A value-dependent branch can record only one branch on the tape.",
    ),
    (
        "dynamic_dimension",
        "warning",
        re.compile(r"Eigen::Dynamic|\.resize\s*\("),
        "Dynamic dimensions require an explicit, stable recording-time contract.",
    ),
    (
        "unsupported_black_box",
        "warning",
        re.compile(r"\bstd::(?:erf|fmod|remainder|frexp|modf)\s*\("),
        "This black-box math call needs an AD-compatible replacement or atomic operation.",
    ),
)


def inspect_source(path: str | Path) -> dict[str, object]:
    source = resolve_within_root(str(path))
    if not source.is_file():
        raise FileNotFoundError(f"Source file not found: {source}")
    findings: list[CompatibilityFinding] = []
    in_block_comment = False
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line, in_block_comment = _code_only(raw_line, in_block_comment)
        if not line.strip():
            continue
        for rule, severity, pattern, message in _RULES:
            if pattern.search(line):
                findings.append(CompatibilityFinding(
                    rule=rule,
                    severity=severity,
                    line=line_number,
                    message=message,
                    excerpt=line.strip()[:240],
                ))
    errors = sum(item.severity == "error" for item in findings)
    warnings = sum(item.severity == "warning" for item in findings)
    return {
        "passed": errors == 0,
        "source": relpath_for_display(source),
        "errors": errors,
        "warnings": warnings,
        "findings": [item.to_dict() for item in findings],
    }


def _code_only(line: str, in_block_comment: bool) -> tuple[str, bool]:
    output = ""
    cursor = 0
    while cursor < len(line):
        if in_block_comment:
            end = line.find("*/", cursor)
            if end < 0:
                return output, True
            cursor = end + 2
            in_block_comment = False
            continue
        start = line.find("/*", cursor)
        single = line.find("//", cursor)
        if single >= 0 and (start < 0 or single < start):
            output += line[cursor:single]
            return output, False
        if start < 0:
            output += line[cursor:]
            return output, False
        output += line[cursor:start]
        cursor = start + 2
        in_block_comment = True
    return output, in_block_comment
