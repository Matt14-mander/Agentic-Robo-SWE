"""PUBLIC demo oracle. Production holdouts belong only in a private oracle image."""
import json
import math
import random
import sys

payload = json.load(sys.stdin)
if sys.argv[1] == "prepare":
    rng = random.Random(72103)
    cases = [[3.1, -3.1], [-3.1, 3.1], [0.1, 0.2]]
    cases += [[rng.uniform(-20, 20), rng.uniform(-20, 20)] for _ in range(29)]
    print(json.dumps({"cases": cases}))
elif sys.argv[1] == "judge":
    cases, observed = payload["cases"], payload["observations"]
    passed = len(cases) == len(observed) and all(
        type(value) in (int, float) and math.isfinite(value) and
        math.isclose(value, math.remainder(target - actual, 2 * math.pi), abs_tol=1e-10, rel_tol=1e-10)
        for (target, actual), value in zip(cases, observed)
    )
    print(json.dumps({"passed": passed, "checks": len(cases)}))
else:
    raise SystemExit("unknown phase")
