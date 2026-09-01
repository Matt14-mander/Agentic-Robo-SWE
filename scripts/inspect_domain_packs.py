"""List installed Domain Packs, their tools, validators and local capabilities."""

from __future__ import annotations

import argparse
import json

from agent.domain.registry import get_domain_registry


def _payload() -> dict[str, object]:
    packs = []
    for pack in get_domain_registry().all_packs():
        capabilities = dict(pack.capability_probe()) if pack.capability_probe else {}
        packs.append({
            "id": pack.id,
            "version": pack.version,
            "description": pack.manifest.description,
            "tools": [tool.name for tool in pack.tools],
            "validators": sorted(pack.validators),
            "required_executables": list(pack.manifest.required_executables),
            "optional_executables": list(pack.manifest.optional_executables),
            "local_only": pack.manifest.local_only,
            "capabilities": capabilities,
        })
    return {"schema_version": 1, "packs": packs}


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect installed Agentic-Robo-SWE Domain Packs")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()
    payload = _payload()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    for pack in payload["packs"]:
        capabilities = pack["capabilities"]
        status = "ready" if capabilities.get("available", True) else "missing capabilities"
        print(f"{pack['id']} {pack['version']} — {status}")
        print(f"  tools: {', '.join(pack['tools'])}")
        print(f"  validators: {', '.join(pack['validators'])}")
        if capabilities.get("missing"):
            print(f"  missing: {', '.join(capabilities['missing'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

