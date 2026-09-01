"""Strict TOML loader for built-in Domain Pack manifests."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from agent.domain.models import DomainManifest


_PACK_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def load_domain_manifest(path: str | Path) -> DomainManifest:
    manifest_path = Path(path)
    data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = data.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"Unsupported Domain Pack schema_version: {schema_version!r}")
    pack_id = str(data.get("id", ""))
    if not _PACK_ID_RE.fullmatch(pack_id):
        raise ValueError(f"Invalid Domain Pack id: {pack_id!r}")
    version = str(data.get("version", "")).strip()
    description = str(data.get("description", "")).strip()
    if not version or not description:
        raise ValueError("Domain Pack version and description are required")
    return DomainManifest(
        schema_version=1,
        id=pack_id,
        version=version,
        description=description,
        required_executables=_string_tuple(data, "required_executables"),
        optional_executables=_string_tuple(data, "optional_executables"),
        local_only=bool(data.get("local_only", True)),
    )


def _string_tuple(data: dict[str, object], key: str) -> tuple[str, ...]:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"Domain Pack {key} must be a list of non-empty strings")
    return tuple(value)

