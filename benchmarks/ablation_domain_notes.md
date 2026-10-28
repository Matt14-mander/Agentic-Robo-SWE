# Robotics C++ development notes

Preserve the declared scalar type throughout a templated numerical computation.
AD scalar types can define mathematical overloads in their own namespace; unqualified
calls with a standard fallback allow argument-dependent lookup. Explicit conversion
to a passive scalar can remove derivatives even when values remain plausible.

Check storage order, vector dimensions, quaternion component order and active/passive
rotation conventions against the interface contract. A change of reference point
for a wrench affects the moment as well as the force frame. Keep units explicit:
frequency, period and normalized trajectory time are distinct quantities.

For saturation and constrained control, inspect both output limits and internal
state evolution. Continuous joint displacement requires a consistent wrap convention.
Compilation and heuristic source diagnostics do not establish numerical correctness.
Validate any repair using the evidence available in the current session.
