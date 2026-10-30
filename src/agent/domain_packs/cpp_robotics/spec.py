"""Independent deterministic inputs and robotics equations; never execute candidate Python."""
from __future__ import annotations

import math
import random

CASE_NAMES = ("scalar_adl", "jacobian_layout", "quaternion_order", "wrench_frame",
              "sample_period", "trajectory_scaling", "anti_windup", "joint_wrap")
VERSION = "0.1.0"
SEED = 24680


def samples(scenario: str, *, seed: int = SEED, include_fixed: bool = True) -> list[list[float]]:
    index = CASE_NAMES.index(scenario)
    rng = random.Random(seed + index)
    if scenario == "scalar_adl":
        fixed = [[0., 0.], [1., -.5], [-1., .8], [math.pi / 2, 0.]]
        extra = [[rng.uniform(-2, 2), rng.uniform(-2, 2)] for _ in range(24)]
    elif scenario == "jacobian_layout":
        fixed = [[1., 2., 3., 4., 5., 6., 1., -1.], [1., 0., 0., 1., 2., -2., 0., 1.],
                 [0.] * 8, [-1., 2., -3., 4., -5., 6., -1., .5]]
        extra = [[rng.uniform(-3, 3) for _ in range(8)] for _ in range(24)]
    elif scenario == "quaternion_order":
        fixed = [[0., 0., 0., 1., 1., 2., 3.], [0., 0., 2., 2., 1., 0., 0.],
                 [1., 0., 0., 0., 0., 1., 0.], [0., 0., -2., -2., 1., 0., 0.]]
        extra = [[rng.uniform(-2, 2) for _ in range(7)] for _ in range(24)]
    elif scenario == "wrench_frame":
        fixed = [[0., 1., 0., 0., 0., 2., 0., 0., 0., 0.],
                 [math.pi / 2, 0., 0., 0., 1., 0., 0., 0., 1., 0.],
                 [0.] * 10, [-.8, -.2, .7, 1., 2., -3., 4., .1, -.2, .3]]
        extra = [[rng.uniform(-2, 2) for _ in range(10)] for _ in range(24)]
    elif scenario == "sample_period":
        fixed = [[1., 2., 3., 1000.], [0., -1., 2., 50.], [2., 0., 0., 100.], [0., 0., 1., 10.]]
        extra = [[rng.uniform(-2, 2) for _ in range(3)] + [float(rng.choice([10, 50, 100, 1000]))]
                 for _ in range(24)]
    elif scenario == "trajectory_scaling":
        fixed = [[0., 1., 2., 0.], [0., 1., 2., 2.], [1., -2., .5, .1], [-1., 3., 4., 1.]]
        extra = [[rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(.2, 5), rng.uniform(-1, 6)]
                 for _ in range(24)]
    elif scenario == "anti_windup":
        fixed = [[.1, 1., 1., 1.] + [10.] * 20 + [-.5] * 5,
                 [.1, 1., 1., 1.] + [-10.] * 20 + [.5] * 5,
                 [.2, 2., .5, .8] + [.1] * 25, [.05, 3., 1., .5] + [0.] * 25]
        extra = [[rng.uniform(.01, .2), rng.uniform(.5, 3), 1., .8]
                 + [rng.uniform(-8, 8) for _ in range(25)] for _ in range(24)]
    else:
        fixed = [[3., -3.], [-3., 3.], [0., 0.], [7., -6.]]
        extra = [[rng.uniform(-20, 20), rng.uniform(-20, 20)] for _ in range(24)]
    return (fixed if include_fixed else []) + extra


def expected(scenario: str, x: list[float]) -> list[float]:
    if scenario == "scalar_adl":
        q, v = x
        return [math.sin(q) + q*v, math.cos(q) + v, q]
    if scenario == "jacobian_layout":
        return [x[2*r]*x[6] + x[2*r+1]*x[7] for r in range(3)]
    if scenario == "quaternion_order":
        norm = math.sqrt(sum(t*t for t in x[:4]))
        u = [t / norm for t in x[:3]]
        w, vector = x[3] / norm, x[4:]
        dot = sum(a*b for a, b in zip(u, vector, strict=True))
        cross = [u[1]*vector[2]-u[2]*vector[1], u[2]*vector[0]-u[0]*vector[2], u[0]*vector[1]-u[1]*vector[0]]
        return [(2*w*w-1)*vector[i] + 2*u[i]*dot + 2*w*cross[i] for i in range(3)]
    if scenario == "wrench_frame":
        c, s = math.cos(x[0]), math.sin(x[0])
        def rotate(v: list[float]) -> list[float]:
            return [c*v[0]-s*v[1], s*v[0]+c*v[1], v[2]]
        p, force, moment = x[1:4], rotate(x[4:7]), rotate(x[7:10])
        cross = [p[1]*force[2]-p[2]*force[1], p[2]*force[0]-p[0]*force[2], p[0]*force[1]-p[1]*force[0]]
        return force + [a+b for a, b in zip(moment, cross, strict=True)]
    if scenario == "sample_period":
        q, v, a, hz = x
        dt = 1. / hz
        return [q + v*dt + .5*a*dt*dt, v+a*dt]
    if scenario == "trajectory_scaling":
        start, goal, duration, time = x
        s = min(1., max(0., time / duration))
        d = goal-start
        return [start+d*(10*s**3-15*s**4+6*s**5),
                d*(30*s**2-60*s**3+30*s**4)/duration,
                d*(60*s-180*s**2+120*s**3)/duration**2]
    if scenario == "anti_windup":
        dt, limit, kp, ki = x[:4]
        integral, outputs = 0., []
        for error in x[4:]:
            candidate = integral + error*dt
            raw = kp*error + ki*candidate
            if abs(raw) <= limit or (raw > limit and error < 0) or (raw < -limit and error > 0):
                integral = candidate
            outputs.append(max(-limit, min(limit, kp*error + ki*integral)))
        return outputs + [integral]
    if scenario == "joint_wrap":
        return [math.atan2(math.sin(x[1]-x[0]), math.cos(x[1]-x[0]))]
    raise ValueError(f"Unknown robotics scenario: {scenario}")
