"""Resolve pack-specific tools, prompts and metadata from Agent state."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from agent.domain.models import DomainPack
from agent.domain.registry import get_domain_registry


def active_domain_packs(pack_ids: Iterable[str]) -> tuple[DomainPack, ...]:
    return get_domain_registry().resolve(pack_ids)


def domain_prompt_fragment(pack_ids: Iterable[str]) -> str:
    fragments = [pack.prompt_fragment.strip() for pack in active_domain_packs(pack_ids)]
    fragments = [fragment for fragment in fragments if fragment]
    if not fragments:
        return ""
    return "\n\nDomain Pack workflow:\n" + "\n\n".join(fragments) + "\n"


def append_domain_tools(base_tools: Sequence[Any], pack_ids: Iterable[str]) -> list[Any]:
    tools = list(base_tools)
    names = {str(getattr(tool, "name", "")) for tool in tools}
    for pack in active_domain_packs(pack_ids):
        for tool in pack.tools:
            name = str(getattr(tool, "name", ""))
            if name in names:
                raise ValueError(f"Domain tool conflicts with core tool: {name}")
            tools.append(tool)
            names.add(name)
    return tools
