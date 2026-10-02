"""Run the Phase 6.4a deterministic Crocoddyl solver-chain profile."""

import json

from agent.domain_packs.crocoddyl_mpc.validators import validate_crocoddyl_mpc


if __name__ == "__main__":
    result = validate_crocoddyl_mpc()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.get("passed") is True else 1)
