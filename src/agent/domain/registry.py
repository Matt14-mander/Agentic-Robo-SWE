"""Domain Pack registry with deterministic conflict detection."""

from __future__ import annotations

from collections.abc import Iterable

from agent.domain.models import DomainPack, DomainValidator


class DomainRegistry:
    def __init__(self) -> None:
        self._packs: dict[str, DomainPack] = {}
        self._tool_owners: dict[str, str] = {}
        self._validators: dict[str, tuple[str, DomainValidator]] = {}

    def register(self, pack: DomainPack) -> None:
        if pack.id in self._packs:
            raise ValueError(f"Domain Pack already registered: {pack.id}")
        tool_names = [str(getattr(tool, "name", "")) for tool in pack.tools]
        if any(not name for name in tool_names):
            raise ValueError(f"Domain Pack {pack.id} contains an unnamed tool")
        if len(tool_names) != len(set(tool_names)):
            raise ValueError(f"Domain Pack {pack.id} contains duplicate tool names")
        conflicts = sorted(name for name in tool_names if name in self._tool_owners)
        if conflicts:
            raise ValueError(f"Domain tool name conflict: {', '.join(conflicts)}")
        for validator_id in pack.validators:
            expected_prefix = f"{pack.id}."
            if not validator_id.startswith(expected_prefix):
                raise ValueError(
                    f"Domain validator {validator_id!r} must start with {expected_prefix!r}"
                )
            if validator_id in self._validators:
                raise ValueError(f"Domain validator already registered: {validator_id}")
        self._packs[pack.id] = pack
        self._tool_owners.update({name: pack.id for name in tool_names})
        self._validators.update(
            {validator_id: (pack.id, validator) for validator_id, validator in pack.validators.items()}
        )

    def get(self, pack_id: str) -> DomainPack:
        try:
            return self._packs[pack_id]
        except KeyError as exc:
            raise KeyError(f"Unknown Domain Pack: {pack_id}") from exc

    def resolve(self, pack_ids: Iterable[str]) -> tuple[DomainPack, ...]:
        resolved: list[DomainPack] = []
        seen: set[str] = set()
        for pack_id in pack_ids:
            if pack_id in seen:
                continue
            resolved.append(self.get(pack_id))
            seen.add(pack_id)
        return tuple(resolved)

    def validator(self, validator_id: str) -> DomainValidator | None:
        entry = self._validators.get(validator_id)
        return entry[1] if entry else None

    def all_packs(self) -> tuple[DomainPack, ...]:
        return tuple(self._packs.values())


_REGISTRY = DomainRegistry()
_BUILTINS_LOADED = False


def get_domain_registry() -> DomainRegistry:
    global _BUILTINS_LOADED
    if not _BUILTINS_LOADED:
        from agent.domain_packs.cpp_reference import create_pack
        from agent.domain_packs.pinocchio_rnea import create_pack as create_pinocchio_pack
        from agent.domain_packs.robotics_autodiff_codegen import (
            create_pack as create_autodiff_codegen_pack,
        )

        _REGISTRY.register(create_pack())
        _REGISTRY.register(create_autodiff_codegen_pack())
        _REGISTRY.register(create_pinocchio_pack())
        _BUILTINS_LOADED = True
    return _REGISTRY
