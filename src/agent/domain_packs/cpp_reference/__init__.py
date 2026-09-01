"""Factory for the built-in Phase 6.0 C++ reference Domain Pack."""

from pathlib import Path

from agent.domain.loader import load_domain_manifest
from agent.domain.models import DomainPack
from agent.domain_packs.cpp_reference.constants import PACK_ID, PACK_VERSION
from agent.domain.toolchain import inspect_toolchain
from agent.domain_packs.cpp_reference.tools import CPP_REFERENCE_TOOLS
from agent.domain_packs.cpp_reference.validators import validate_cpp_reference


_PACK_ROOT = Path(__file__).resolve().parent


def create_pack() -> DomainPack:
    manifest = load_domain_manifest(_PACK_ROOT / "pack.toml")
    if (manifest.id, manifest.version) != (PACK_ID, PACK_VERSION):
        raise ValueError("C++ reference pack manifest identity is inconsistent")
    prompt = (_PACK_ROOT / "prompt.md").read_text(encoding="utf-8")
    return DomainPack(
        manifest=manifest,
        tools=CPP_REFERENCE_TOOLS,
        validators={"cpp_reference.output_equivalence": validate_cpp_reference},
        prompt_fragment=prompt,
        capability_probe=lambda: inspect_toolchain().to_dict(),
        source=_PACK_ROOT,
    )
