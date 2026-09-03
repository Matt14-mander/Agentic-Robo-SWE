"""Agent-facing diagnostics for the Phase 6.1 autodiff workflow."""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.tools import tool

from agent.domain_packs.robotics_autodiff_codegen.compatibility import inspect_source
from agent.domain_packs.robotics_autodiff_codegen.dependencies import inspect_dependencies
from agent.domain_packs.robotics_autodiff_codegen.validators import (
    validate_autodiff_codegen,
    validate_sparse_codegen,
)


@tool
def inspect_ad_compatibility(source_path: str) -> str:
    """Find deterministic C++ Scalar, branch, dimension, and black-box AD risks."""
    try:
        result = inspect_source(source_path)
    except (OSError, ValueError) as exc:
        result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def inspect_autodiff_codegen_environment() -> str:
    """Inspect pinned CppAD/CppADCodeGen and Linux runtime compilation capabilities."""
    return json.dumps(inspect_dependencies(), ensure_ascii=False, indent=2)


@tool
def validate_autodiff_codegen_model(
    source_path: str, model_contract: Literal["coupled", "triangular"] = "coupled"
) -> str:
    """Run Dense+Sparse gates; use triangular for the sparse_codegen benchmark manifest."""
    validator = validate_sparse_codegen if model_contract == "triangular" else validate_autodiff_codegen
    return json.dumps(validator(source_path), ensure_ascii=False, indent=2)


AUTODIFF_CODEGEN_TOOLS = (
    inspect_ad_compatibility,
    inspect_autodiff_codegen_environment,
    validate_autodiff_codegen_model,
)
