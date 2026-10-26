"""Portable professional-pattern C++ repair tasks, independent of heavy native libraries."""
from functools import partial
from pathlib import Path

from agent.domain.loader import load_domain_manifest
from agent.domain.models import DomainPack
from agent.domain.toolchain import inspect_toolchain
from agent.domain_packs.cpp_robotics.spec import CASE_NAMES
from agent.domain_packs.cpp_robotics.validators import validate_robotics_cpp


def create_pack() -> DomainPack:
    root = Path(__file__).resolve().parent
    return DomainPack(
        manifest=load_domain_manifest(root / "pack.toml"),
        validators={f"cpp_robotics.{name}": partial(validate_robotics_cpp, scenario=name)
                    for name in CASE_NAMES},
        prompt_fragment=(root / "prompt.md").read_text(encoding="utf-8"),
        capability_probe=lambda: inspect_toolchain().to_dict(), source=root,
    )
