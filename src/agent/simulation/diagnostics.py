"""Failure diagnosis and standalone HTML rendering for Phase 5.3."""

from __future__ import annotations

import html
import math
from dataclasses import asdict, dataclass
from typing import Sequence

from agent.simulation.robustness import RobustnessReport, RobustnessTrial
from agent.simulation.trajectory_tracking import TrajectorySample


@dataclass(frozen=True)
class TrialDiagnosis:
    seed: int
    status: str
    category: str
    primary_failure: str | None
    first_failure_time: float | None
    first_safety_event: dict[str, object] | None
    peak_tracking_error: float | None
    peak_tracking_error_time: float | None
    trace_samples: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def diagnose_trial(trial: RobustnessTrial) -> TrialDiagnosis:
    """Summarize the earliest event and worst observed tracking error."""
    event = trial.metrics.first_safety_event
    trace = trial.metrics.trace or ()
    peak_sample = max(trace, key=_sample_max_error) if trace else None
    primary_failure = trial.failures[0] if trial.failures else None
    category = _failure_category(primary_failure, event.kind if event else None)
    first_failure_time = (
        event.time_seconds
        if event is not None
        else trace[-1].time_seconds if primary_failure and trace else None
    )
    return TrialDiagnosis(
        seed=trial.scenario.seed,
        status="pass" if trial.passed else "fail",
        category=category,
        primary_failure=primary_failure,
        first_failure_time=first_failure_time,
        first_safety_event=asdict(event) if event else None,
        peak_tracking_error=_sample_max_error(peak_sample) if peak_sample else None,
        peak_tracking_error_time=peak_sample.time_seconds if peak_sample else None,
        trace_samples=len(trace),
    )


def render_diagnostic_html(report: RobustnessReport) -> str:
    """Render a dependency-free report with one diagnostic section per seed."""
    sections = "".join(_render_trial(trial) for trial in report.trials)
    status = "PASS" if report.passed else "FAIL"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agentic-Robo-SWE robustness diagnostics</title>
<style>
:root {{ color-scheme: light dark; --bg:#f7f8fa; --surface:#fff; --fg:#17202a;
  --muted:#667085; --border:#d0d5dd; --s1:#2563eb; --s2:#d97706; --s3:#059669;
  --s4:#7c3aed; --danger:#dc2626; --grid:#e4e7ec; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#111827; --surface:#1f2937;
  --fg:#f3f4f6; --muted:#a7b0bf; --border:#4b5563; --s1:#60a5fa; --s2:#fbbf24;
  --s3:#34d399; --s4:#c4b5fd; --danger:#f87171; --grid:#374151; }} }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--fg);
  font:14px/1.5 system-ui,sans-serif; }} main {{ max-width:1100px; margin:auto; padding:24px; }}
h1,h2,h3 {{ font-weight:600; }} .summary,.trial {{ background:var(--surface);
  border:1px solid var(--border); border-radius:10px; padding:18px; margin-bottom:18px; }}
.metrics {{ display:flex; flex-wrap:wrap; gap:18px; color:var(--muted); }}
.metrics strong {{ color:var(--fg); }} .failure {{ color:var(--danger); }}
.chart {{ width:100%; overflow:hidden; margin-top:12px; }} svg {{ width:100%; height:auto;
  display:block; }} .frame,.grid {{ fill:none; stroke:var(--grid); }}
.axis {{ fill:var(--muted); font-size:11px; }} .line {{ fill:none; stroke-width:2; }}
.s1 {{ stroke:var(--s1); }} .s2 {{ stroke:var(--s2); }} .s3 {{ stroke:var(--s3); }}
.s4 {{ stroke:var(--s4); }} .target {{ stroke-dasharray:6 4; opacity:.8; }}
.limit {{ stroke:var(--danger); stroke-dasharray:4 4; opacity:.7; }}
.event {{ stroke:var(--danger); stroke-width:1.5; }} .legend {{ color:var(--muted); }}
.legend span {{ margin-right:16px; white-space:nowrap; }} code {{ color:var(--fg); }}
@media (max-width:600px) {{ main {{ padding:12px; }} .summary,.trial {{ padding:12px; }} }}
</style>
</head>
<body><main>
<h1>Phase 5.3 robustness diagnostics</h1>
<section class="summary">
  <h2>Suite {status}</h2>
  <div class="metrics">
    <span><strong>{report.passed_trials}/{report.total_trials}</strong> trials passed</span>
    <span><strong>{report.pass_rate:.1%}</strong> pass rate</span>
    <span><strong>{report.p95_tracking_rmse:.5f}</strong> P95 tracking RMSE</span>
    <span><strong>{report.safety_violations}</strong> safety violations</span>
    <span><strong>{report.robustness_score:.2f}</strong> score</span>
  </div>
</section>
{sections}
</main></body></html>
"""


def _render_trial(trial: RobustnessTrial) -> str:
    diagnosis = diagnose_trial(trial)
    trace = trial.metrics.trace or ()
    status = "PASS" if trial.passed else "FAIL"
    failure = html.escape(diagnosis.primary_failure or "No validation failure")
    event_text = (
        f"first event at {diagnosis.first_failure_time:.3f}s"
        if diagnosis.first_failure_time is not None
        else "no failure event"
    )
    if not trace:
        charts = "<p>No trace captured. Re-run without <code>--no-trace</code>.</p>"
    else:
        charts = (
            '<div class="legend"><span>solid: actual</span><span>dashed: target/limit</span>'
            '<span>red vertical: first safety event</span></div>'
            + _position_chart(trace, trial)
            + _command_chart(trace, trial)
            + _error_chart(trace, trial)
        )
    failure_class = " failure" if not trial.passed else ""
    return f"""
<section class="trial">
  <h2>Seed {trial.scenario.seed} — {status}</h2>
  <div class="metrics">
    <span>category <strong>{html.escape(diagnosis.category)}</strong></span>
    <span>trace samples <strong>{diagnosis.trace_samples}</strong></span>
    <span>{html.escape(event_text)}</span>
  </div>
  <p class="{failure_class.strip()}">{failure}</p>
  {charts}
</section>
"""


def _position_chart(trace: Sequence[TrajectorySample], trial: RobustnessTrial) -> str:
    return _line_chart(
        "Joint position tracking",
        "position (rad)",
        trace,
        (
            ("shoulder actual", [sample.positions[0] for sample in trace], "s1"),
            ("shoulder target", [sample.target_positions[0] for sample in trace], "s1 target"),
            ("elbow actual", [sample.positions[1] for sample in trace], "s2"),
            ("elbow target", [sample.target_positions[1] for sample in trace], "s2 target"),
        ),
        event_time=_event_time(trial),
    )


def _command_chart(trace: Sequence[TrajectorySample], trial: RobustnessTrial) -> str:
    return _line_chart(
        "Raw controller commands",
        "torque (N·m)",
        trace,
        (
            ("shoulder", [sample.raw_commands[0] for sample in trace], "s1"),
            ("elbow", [sample.raw_commands[1] for sample in trace], "s2"),
        ),
        references=(-10.0, -8.0, 8.0, 10.0),
        event_time=_event_time(trial),
    )


def _error_chart(trace: Sequence[TrajectorySample], trial: RobustnessTrial) -> str:
    return _line_chart(
        "Joint tracking error",
        "error (rad)",
        trace,
        (
            ("shoulder", [sample.joint_errors[0] for sample in trace], "s3"),
            ("elbow", [sample.joint_errors[1] for sample in trace], "s4"),
        ),
        references=(0.0,),
        event_time=_event_time(trial),
    )


def _line_chart(
    title: str,
    y_label: str,
    trace: Sequence[TrajectorySample],
    series: Sequence[tuple[str, Sequence[float], str]],
    *,
    references: Sequence[float] = (),
    event_time: float | None = None,
) -> str:
    width, height = 960.0, 250.0
    left, right, top, bottom = 66.0, 18.0, 30.0, 38.0
    times = [sample.time_seconds for sample in trace]
    x_min, x_max = 0.0, max(times)
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0
    values = [value for _, points, _ in series for value in points] + list(references)
    finite_values = [value for value in values if math.isfinite(value)] or [-1.0, 1.0]
    y_min, y_max = min(finite_values), max(finite_values)
    padding = max((y_max - y_min) * 0.08, 0.02)
    y_min, y_max = y_min - padding, y_max + padding

    def x_coord(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def y_coord(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * (height - top - bottom)

    grid: list[str] = []
    for index in range(5):
        fraction = index / 4
        x_value = x_min + fraction * (x_max - x_min)
        x = x_coord(x_value)
        grid.append(
            f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
            f'y2="{height - bottom}"/><text class="axis" x="{x:.2f}" y="{height - 12}" '
            f'text-anchor="middle">{x_value:.2f}</text>'
        )
        y_value = y_max - fraction * (y_max - y_min)
        y = y_coord(y_value)
        grid.append(
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" '
            f'y2="{y:.2f}"/><text class="axis" x="{left - 8}" y="{y + 4:.2f}" '
            f'text-anchor="end">{y_value:.2f}</text>'
        )
    paths: list[str] = []
    for label, points, css_class in series:
        finite_points = [
            (time, value)
            for time, value in zip(times, points, strict=True)
            if math.isfinite(value)
        ]
        commands = " ".join(
            f"{'M' if index == 0 else 'L'} {x_coord(time):.2f} {y_coord(value):.2f}"
            for index, (time, value) in enumerate(finite_points)
        )
        paths.append(
            f'<path class="line {css_class}" d="{commands}"><title>{html.escape(label)}</title></path>'
        )
    reference_lines = "".join(
        f'<line class="limit" x1="{left}" y1="{y_coord(value):.2f}" '
        f'x2="{width - right}" y2="{y_coord(value):.2f}"/>'
        for value in references
    )
    event_line = ""
    if event_time is not None and x_min <= event_time <= x_max:
        x = x_coord(event_time)
        event_line = (
            f'<line class="event" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
            f'y2="{height - bottom}"/>'
        )
    legend = " · ".join(html.escape(label) for label, _, _ in series)
    return f"""
<div class="chart">
  <h3>{html.escape(title)}</h3>
  <div class="legend">{legend}</div>
  <svg viewBox="0 0 {width:.0f} {height:.0f}" role="img"
       aria-label="{html.escape(title)} over time">
    <rect class="frame" x="{left}" y="{top}" width="{width-left-right}" height="{height-top-bottom}"/>
    {''.join(grid)}{reference_lines}{''.join(paths)}{event_line}
    <text class="axis" x="{(left + width - right) / 2:.2f}" y="{height - 1}" text-anchor="middle">time (s)</text>
    <text class="axis" transform="translate(14 {(top + height - bottom) / 2:.2f}) rotate(-90)"
          text-anchor="middle">{html.escape(y_label)}</text>
  </svg>
</div>
"""


def _sample_max_error(sample: TrajectorySample) -> float:
    return max(abs(value) for value in sample.joint_errors)


def _event_time(trial: RobustnessTrial) -> float | None:
    event = trial.metrics.first_safety_event
    return event.time_seconds if event else None


def _failure_category(primary_failure: str | None, event_kind: str | None) -> str:
    if event_kind:
        return event_kind
    if primary_failure is None:
        return "none"
    lowered = primary_failure.lower()
    if any(word in lowered for word in ("tracking", "final error", "tail rmse", "velocity")):
        return "tracking"
    return "validation"
