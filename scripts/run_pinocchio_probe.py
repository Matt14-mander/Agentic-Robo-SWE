"""Run the correct fixed-URDF RNEA probe without an LLM or editing a benchmark workspace."""

import argparse
import json

from agent.domain_packs.pinocchio_rnea.spec import PACK_ROOT
from agent.domain_packs.pinocchio_rnea.validators import validate_rnea


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-codegen-benefit", action="store_true",
                        help="Fail unless measurements establish a >=10%% median CodeGen benefit")
    args = parser.parse_args()
    result = validate_rnea(str(PACK_ROOT / "harness/reference.hpp"),
                           require_benefit=args.require_codegen_benefit)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
