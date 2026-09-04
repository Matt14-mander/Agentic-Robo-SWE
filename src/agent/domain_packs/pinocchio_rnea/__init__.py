"""Optional Pinocchio pack; importing it never imports/installs native dependencies."""

import json

from langchain_core.tools import tool

from agent.domain.loader import load_domain_manifest
from agent.domain.models import DomainPack
from agent.domain_packs.pinocchio_rnea.dependencies import inspect_dependencies
from agent.domain_packs.pinocchio_rnea.spec import PACK_ROOT, model_contract
from agent.domain_packs.pinocchio_rnea.validators import validate_codegen_benefit, validate_rnea


@tool
def inspect_pinocchio_rnea_environment() -> str:
    """Read fixed robot layout and local pinned Pinocchio capabilities without installation."""
    return json.dumps({"environment": inspect_dependencies(), "model": model_contract()}, indent=2)


@tool
def validate_pinocchio_rnea_model(source_path: str) -> str:
    """Compile a Scalar-generic candidate_rnea wrapper; measure numerical validity and latency."""
    return json.dumps(validate_rnea(source_path), ensure_ascii=False, indent=2)


def create_pack() -> DomainPack:
    return DomainPack(
        manifest=load_domain_manifest(PACK_ROOT / "pack.toml"),
        tools=(inspect_pinocchio_rnea_environment, validate_pinocchio_rnea_model),
        validators={"pinocchio_rnea.correctness": validate_rnea,
                    "pinocchio_rnea.codegen_benefit": validate_codegen_benefit},
        prompt_fragment=(PACK_ROOT / "prompt.md").read_text(encoding="utf-8"),
        capability_probe=inspect_dependencies, source=PACK_ROOT,
    )
