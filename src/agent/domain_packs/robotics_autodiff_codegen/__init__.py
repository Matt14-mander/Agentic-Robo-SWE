"""Factory for the Phase 6.1 robotics autodiff and code generation pack."""

from pathlib import Path

from agent.domain.loader import load_domain_manifest
from agent.domain.models import DomainPack
from agent.domain_packs.robotics_autodiff_codegen.constants import PACK_ID, PACK_VERSION
from agent.domain_packs.robotics_autodiff_codegen.dependencies import inspect_dependencies
from agent.domain_packs.robotics_autodiff_codegen.tools import AUTODIFF_CODEGEN_TOOLS
from agent.domain_packs.robotics_autodiff_codegen.validators import validate_autodiff_codegen


_PACK_ROOT = Path(__file__).resolve().parent


def create_pack() -> DomainPack:
    manifest = load_domain_manifest(_PACK_ROOT / "pack.toml")
    if (manifest.id, manifest.version) != (PACK_ID, PACK_VERSION):
        raise ValueError("Autodiff CodeGen pack manifest identity is inconsistent")
    return DomainPack(
        manifest=manifest,
        tools=AUTODIFF_CODEGEN_TOOLS,
        validators={
            "robotics_autodiff_codegen.output_and_jacobian": validate_autodiff_codegen,
        },
        prompt_fragment=(_PACK_ROOT / "prompt.md").read_text(encoding="utf-8"),
        capability_probe=inspect_dependencies,
        source=_PACK_ROOT,
    )
