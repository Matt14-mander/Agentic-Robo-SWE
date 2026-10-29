"""Operator-owned development traces; never an input to the planner or hidden oracle.

Record node boundaries instead of serializing concurrent ToolNode calls. Source
identity belongs to the whole batch, not to an invented ordering of its tools.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field

from agent.workspace import source_files

_SECRET_KEY = re.compile(r"(^|_)(api_key|token|password|secret|authorization|credential)s?$", re.I)
_USAGE_KEYS = {"input_tokens", "output_tokens", "total_tokens"}
_ACTIVE: ContextVar[RepairTrace | None] = ContextVar("repair_trace", default=None)
_SPAN: ContextVar[str | None] = ContextVar("repair_trace_span", default=None)


class SourceIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["available", "unavailable"]
    digest: str | None = None
    manifest: str | None = None
    error: str | None = None


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    scope: Literal["development"] = "development"
    run_id: str
    sequence: int = Field(ge=1)
    timestamp: str
    kind: Literal[
        "session_started", "session_closed", "node_started", "node_completed",
        "node_error", "node_interrupted", "agent_message", "tool_requested",
        "tool_result", "development_validation",
    ]
    span_id: str | None = None
    node: str | None = None
    loop_step: int | None = None
    source: SourceIdentity | None = None
    data: dict[str, Any] = Field(default_factory=dict)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True).encode()).hexdigest()


def bounded(value: Any, depth: int = 0) -> Any:
    """Bound untrusted payloads and redact credential-shaped keys, not arbitrary prose.

    Truncation is explicit; these payloads must not silently become complete skills.
    Token usage keys (input_tokens, etc.) are retained.
    """
    if depth > 6:
        return {"truncated": True, "reason": "depth"}
    if isinstance(value, str):
        if len(value) <= 12000:
            return value
        return {"text": value[:12000], "truncated": True, "original_chars": len(value),
                "sha256": hashlib.sha256(value.encode()).hexdigest()}
    if isinstance(value, dict):
        result = {str(k): "[REDACTED]" if _SECRET_KEY.search(str(k)) and not
                  (str(k) in _USAGE_KEYS and isinstance(v, int)) else bounded(v, depth + 1)
                  for k, v in list(value.items())[:100]}
        if len(value) > 100:
            result["_truncated_items"] = len(value) - 100
        return result
    if isinstance(value, (tuple, list)):
        items = [bounded(item, depth + 1) for item in value[:100]]
        if len(value) > 100:
            items.append({"truncated_items": len(value) - 100})
        return items
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return bounded(str(value), depth + 1)


def _public_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    # Do not export provider thinking/reasoning blocks or response metadata.
    if isinstance(content, list):
        return "\n".join(item if isinstance(item, str) else str(item.get("text", ""))
                         for item in content
                         if isinstance(item, str) or
                         isinstance(item, dict) and item.get("type") == "text")
    return ""


class RepairTrace:
    def __init__(self, directory: Path, source: Path, metadata: dict[str, Any]):
        self.directory = directory.resolve()
        self.source = source.resolve(strict=True)
        if self.source.is_dir() and (self.directory.is_relative_to(self.source) or
                                     self.source.is_relative_to(self.directory)):
            raise ValueError("Trace directory and source tree must not overlap")
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "sources").mkdir()
        self.run_id = uuid.uuid4().hex
        self.sequence = 0
        self.lock = threading.Lock()
        self.started = time.perf_counter()
        self.recording_seconds = 0.0
        from agent.config import get_model_identity
        from agent.workspace import current_workspace
        workspace = current_workspace()
        environment = {
            "python": platform.python_version(), "platform": platform.platform(),
            "model": metadata.get("model_identity") or get_model_identity(),
            "execution_image": workspace.execution_image if workspace else None,
            "trusted_local": workspace.trusted_local if workspace else None,
            "executor_backend": os.getenv("EXECUTOR_BACKEND", "local"),
            "toolchain_fingerprint": metadata.get("toolchain_fingerprint"),
        }
        manifest = {"schema_version": 1, "scope": "development", "run_id": self.run_id,
                    "source_root": str(self.source),
                    "source_scope": "tree" if self.source.is_dir() else "fixture_file",
                    "environment": environment, "environment_id": _digest(environment),
                    "metadata": bounded(metadata)}
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self.stream = (self.directory / "events.jsonl").open("x", encoding="utf-8")

    def snapshot(self) -> SourceIdentity:
        start = time.perf_counter()
        try:
            paths = source_files(self.source) if self.source.is_dir() else [self.source]
            entries = []
            for path in paths:
                if path.is_symlink() or not path.is_file():
                    raise ValueError("Non-regular source")
                info = path.stat()
                if info.st_size > 100_000_000:
                    raise ValueError("Source exceeds snapshot bound")
                name = path.relative_to(self.source).as_posix() if self.source.is_dir() else path.name
                entries.append((name, bool(info.st_mode & 0o111),
                                hashlib.sha256(path.read_bytes()).hexdigest()))
            # Same algorithm as workspace.source_digest for external repository trees.
            digest = hashlib.sha256(json.dumps(entries, ensure_ascii=True).encode()).hexdigest()
            relative = f"sources/{digest}.json"
            destination = self.directory / relative
            if not destination.exists():
                destination.write_text(json.dumps(entries, ensure_ascii=True), encoding="utf-8")
            return SourceIdentity(status="available", digest=digest, manifest=relative)
        except (OSError, ValueError) as exc:
            return SourceIdentity(status="unavailable", error=type(exc).__name__)
        finally:
            self.recording_seconds += time.perf_counter() - start

    def emit(self, kind: Any, *, source: SourceIdentity | None = None,
             span_id: str | None = None, node: str | None = None,
             loop_step: int | None = None, **data: Any) -> None:
        start = time.perf_counter()
        with self.lock:
            self.sequence += 1
            event = TraceEvent(run_id=self.run_id, sequence=self.sequence,
                               timestamp=datetime.now(UTC).isoformat(), kind=kind,
                               source=source, span_id=span_id, node=node,
                               loop_step=loop_step, data=bounded(data))
            self.stream.write(event.model_dump_json() + "\n")
            self.stream.flush()
            os.fsync(self.stream.fileno())
        self.recording_seconds += time.perf_counter() - start

    def messages(self, messages: list[Any], *, node: str, span_id: str,
                 loop_step: int, source: SourceIdentity) -> None:
        common: dict[str, Any] = {
            "node": node, "span_id": span_id, "loop_step": loop_step, "source": source}
        for message in messages:
            if isinstance(message, AIMessage):
                self.emit("agent_message", **common, text=_public_text(message.content),
                          usage=getattr(message, "usage_metadata", None),
                          provider_response={key: message.response_metadata[key] for key in
                                             ("model_name", "model", "system_fingerprint", "finish_reason")
                                             if key in message.response_metadata},
                          epistemic_status="agent_statement_not_verified")
                for call in message.tool_calls:
                    self.emit("tool_requested", **common, tool_call_id=call["id"],
                              tool=call["name"], arguments=call["args"])
            elif isinstance(message, ToolMessage):
                text = _public_text(message.content)
                status = "returned"
                if text.startswith("DENIED"):
                    status = "denied"
                elif text.startswith("SKIPPED"):
                    status = "skipped"
                elif message.status == "error" or text.startswith("ERROR:"):
                    status = "error"
                observation = None
                if message.name == "cpp_build":
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, dict) and isinstance(parsed.get("passed"), bool):
                            observation = {"kind": "build", "reported_passed": parsed["passed"],
                                           "stage": parsed.get("stage"),
                                           "exit_code": parsed.get("exit_code")}
                    except (ValueError, TypeError):
                        pass
                self.emit("tool_result", **common, tool_call_id=message.tool_call_id,
                          tool=message.name, status=status, output=text,
                          observation=observation, source_attribution="node_batch",
                          failure=_tool_failure(message.name, text, status),
                          epistemic_status="tool_observation_not_repair_acceptance")
            elif isinstance(message, HumanMessage) and node == "benchmark_validate":
                # Deterministic gate output, not a fresh user instruction.
                self.emit("tool_result", **common, tool="benchmark_gate", output=message.content,
                          status="returned", source_attribution="node_batch")


@contextmanager
def use_repair_trace(directory: str | Path, source: str | Path,
                     metadata: dict[str, Any]) -> Iterator[RepairTrace]:
    """Create an exclusive trace segment. Resume creates a new segment, linked by thread ID."""
    trace = RepairTrace(Path(directory), Path(source), metadata)
    token = _ACTIVE.set(trace)
    outcome = "closed"
    try:
        trace.emit("session_started", source=trace.snapshot())
        yield trace
    except BaseException:
        outcome = "error"
        raise
    finally:
        try:
            trace.emit("session_closed", source=trace.snapshot(), status=outcome,
                       elapsed_seconds=time.perf_counter() - trace.started,
                       recording_seconds=trace.recording_seconds)
        finally:
            _ACTIVE.reset(token)
            trace.stream.close()


def _tool_failure(name: str | None, output: str, status: str) -> dict[str, Any] | None:
    from agent.failures import Failure, classify
    if status == "error":
        return Failure("tool_error", "tool", "tool returned an error").to_dict()
    if name in {"cpp_build", "compile_candidate_cpp", "validate_development"}:
        try:
            payload = json.loads(output)
            if isinstance(payload, dict):
                return classify(payload).to_dict()
        except ValueError:
            return Failure("invalid_evidence", "protocol", "malformed structured tool output").to_dict()
    return None


def record_development_validation(validator: str, output: str, reported_passed: bool,
                                  *, payload: dict[str, Any] | None = None) -> None:
    trace = _ACTIVE.get()
    if trace:
        from agent.failures import classify
        trace.emit("development_validation", span_id=_SPAN.get(), validator=validator,
                   output=output, reported_passed=reported_passed,
                   failure=classify(payload or {"passed": reported_passed}).to_dict(),
                   authority="public_benchmark_gate", source_attribution="enclosing_node_batch")


def trace_node(name: str, function: Callable, *, with_config: bool = False) -> Callable:
    def wrapped(state, config: RunnableConfig):
        trace = _ACTIVE.get()
        if trace is None:
            return function(state, config) if with_config else function(state)
        span_id = uuid.uuid4().hex
        loop_step = int(state.get("loop_step", 0))
        before = trace.snapshot()
        common: dict[str, Any] = {"span_id": span_id, "node": name, "loop_step": loop_step}
        trace.emit("node_started", **common, source=before)
        token = _SPAN.set(span_id)
        start = time.perf_counter()
        try:
            result = function(state, config) if with_config else function(state)
        except BaseException as exc:
            from langgraph.errors import GraphInterrupt
            trace.emit("node_interrupted" if isinstance(exc, GraphInterrupt) else "node_error",
                       **common, source=trace.snapshot(), error_type=type(exc).__name__,
                       elapsed_seconds=time.perf_counter() - start)
            raise
        finally:
            _SPAN.reset(token)
        elapsed = time.perf_counter() - start
        after = trace.snapshot()
        trace.messages(result.get("messages", []), **common, source=after)
        trace.emit("node_completed", **common, source=after,
                   source_before=before.model_dump(),
                   failure=result.get("failure"), stop_reason=result.get("stop"),
                   source_changed=(before.digest != after.digest) if
                   before.status == after.status == "available" else None,
                   elapsed_seconds=elapsed)
        return result
    return wrapped


def inspect_trace(directory: str | Path) -> dict[str, Any]:
    """Validate an event prefix; a killed worker's partial final line is not evidence."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or manifest["scope"] != "development":
        raise ValueError("Unsupported trace manifest")
    events: list[TraceEvent] = []
    checked_sources: set[str] = set()
    partial_tail = False
    with (directory / "events.jsonl").open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                partial_tail = True
                break
            event = TraceEvent.model_validate_json(line)
            if event.run_id != manifest["run_id"] or event.sequence != len(events) + 1:
                raise ValueError("Trace run/sequence mismatch")
            identities = [event.source]
            if event.data.get("source_before"):
                identities.append(SourceIdentity.model_validate(event.data["source_before"]))
            for identity in identities:
                if identity is None or identity.status != "available":
                    continue
                if not identity.digest or not re.fullmatch(r"[a-f0-9]{64}", identity.digest):
                    raise ValueError("Invalid source digest")
                if identity.manifest != f"sources/{identity.digest}.json":
                    raise ValueError("Invalid source manifest path")
                if identity.digest in checked_sources:
                    continue
                entries = json.loads((directory / identity.manifest).read_text(encoding="utf-8"))
                actual = hashlib.sha256(json.dumps(entries, ensure_ascii=True).encode()).hexdigest()
                if actual != identity.digest:
                    raise ValueError("Source manifest digest mismatch")
                checked_sources.add(identity.digest)
            events.append(event)
    started = {e.span_id for e in events if e.kind == "node_started"}
    finished = {e.span_id for e in events if e.kind in {
        "node_completed", "node_error", "node_interrupted"}}
    usages = [e.data.get("usage") for e in events if e.kind == "agent_message"]
    usage = {key: sum(item[key] for item in usages if isinstance(item, dict) and
                     isinstance(item.get(key), int)) for key in sorted(_USAGE_KEYS)}
    closing = events[-1].data if events and events[-1].kind == "session_closed" else {}
    return {"schema_version": 1, "run_id": manifest["run_id"], "scope": "development",
            "events": len(events), "partial_tail": partial_tail,
            "session_closed": bool(events and events[-1].kind == "session_closed"),
            "unfinished_spans": sorted(started - finished),
            "node_errors": sum(e.kind == "node_error" for e in events),
            "interrupted_nodes": sum(e.kind == "node_interrupted" for e in events),
            "reported_usage": usage, "usage_missing": sum(item is None for item in usages),
            "elapsed_seconds": closing.get("elapsed_seconds"),
            "recording_seconds": closing.get("recording_seconds"),
            "source_changes": sum(e.kind == "node_completed" and
                                  e.data.get("source_changed") is True for e in events),
            "source_unavailable": sum(e.source is not None and
                                      e.source.status == "unavailable" for e in events),
            "tool_results": sum(e.kind == "tool_result" for e in events),
            "development_validations": sum(e.kind == "development_validation" for e in events),
            "skill_eligible": False}
