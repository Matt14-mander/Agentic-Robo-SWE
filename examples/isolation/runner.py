"""Trusted example adapter: build candidate, collect numbers; do not grade."""
import json
import subprocess
import sys

payload = json.load(sys.stdin)
subprocess.run(["cmake", "-S", "/candidate", "-B", "/tmp/build", "-G", "Ninja"],
               check=True, stdout=sys.stderr, timeout=60)
subprocess.run(["cmake", "--build", "/tmp/build", "--parallel", "2"],
               check=True, stdout=sys.stderr, timeout=60)
inputs = "".join(f"{target} {actual}\n" for target, actual in payload["cases"])
result = subprocess.run(["/tmp/build/controller"], input=inputs, text=True,
                        stdout=subprocess.PIPE, check=True, timeout=10)
observed = [float(line) for line in result.stdout.splitlines()]
print(json.dumps({"observations": observed}, allow_nan=False))
