"""Agent-facing diagnostics for the Phase 6.1 autodiff workflow."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from agent.domain_packs.robotics_autodiff_codegen.compatibility import inspect_source
from agent.domain_packs.robotics_autodiff_codegen.dependencies import inspect_dependencies
from agent.domain_packs.robotics_autodiff_codegen.validators import validate_autodiff_codegen


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
def validate_autodiff_codegen_model(source_path: str) -> str:
    """Run the official output and finite-difference Jacobian gate for a source model."""
    return json.dumps(validate_autodiff_codegen(source_path), ensure_ascii=False, indent=2)


AUTODIFF_CODEGEN_TOOLS = (
    inspect_ad_compatibility,
    inspect_autodiff_codegen_environment,
    validate_autodiff_codegen_model,
)
