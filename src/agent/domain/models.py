"""Contracts shared by the core harness and optional robotics domain packs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping


ValidatorResult = dict[str, object]
DomainValidator = Callable[[str], ValidatorResult]
CapabilityProbe = Callable[[], Mapping[str, object]]


@dataclass(frozen=True)
class DomainManifest:
    schema_version: int
    id: str
    version: str
    description: str
    required_executables: tuple[str, ...] = ()
    optional_executables: tuple[str, ...] = ()
    local_only: bool = True


@dataclass(frozen=True)
class ArtifactPolicy:
    root: str = ".agent_state/domain_artifacts"
    preserve_on_failure: bool = True
    cache_successful_builds: bool = True


@dataclass(frozen=True)
class DomainPack:
    manifest: DomainManifest
    tools: tuple[Any, ...] = ()
    validators: Mapping[str, DomainValidator] = field(default_factory=dict)
    prompt_fragment: str = ""
    artifact_policy: ArtifactPolicy = field(default_factory=ArtifactPolicy)
    capability_probe: CapabilityProbe | None = None
    source: Path | None = None

    @property
    def id(self) -> str:
        return self.manifest.id

    @property
    def version(self) -> str:
        return self.manifest.version
