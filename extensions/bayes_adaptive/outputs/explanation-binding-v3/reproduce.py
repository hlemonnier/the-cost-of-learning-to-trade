"""Command-level receipt-gate rejection artifact; does not score candidates.

Fixtures claiming success are synthetic, not real numerical acceptance. A
deliberately absent candidate directory stops matched fixtures immediately after
the receipt gate. The original authenticated cache is loaded by each command.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
HERE = OUT.parents[1]
FAMILY = HERE / "cache/v2/w9-b17-gh25"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


family = json.loads((FAMILY / "complete.json").read_text())
base = {"scope": "Synthetic diagnostic fixture; not actual refinement acceptance",
        "passes_predeclared_rule": True,
        "selected_artifact_sha256": family["artifact_sha256"],
        "selected_resolution": family["specification"],
        "source": family["source"], "build": family["build"]}
cases = [("matched_fixture_reaches_candidate_gate", deepcopy(base), "FileNotFoundError")]
for key, name in (("selected_artifact_sha256", "selected_artifact"),
                  ("selected_resolution", "selected_specification"),
                  ("source", "solver_source"), ("build", "numerical_build")):
    changed = deepcopy(base)
    changed[key] = "wrong diagnostic value"
    cases.append(("different_" + name, changed,
                  "numerical acceptance does not authenticate selected family: " + name))
    missing = deepcopy(base)
    del missing[key]
    cases.append(("missing_" + name, missing,
                  "numerical acceptance does not authenticate selected family: " + name))
for name, value in (("nonboolean_flag", "true"), ("missing_flag", None)):
    changed = deepcopy(base)
    if value is None:
        del changed["passes_predeclared_rule"]
    else:
        changed["passes_predeclared_rule"] = value
    cases.append((name, changed, "numerical receipt requires a Boolean acceptance flag"))
unresolved = deepcopy(base)
unresolved["passes_predeclared_rule"] = False
unresolved["selected_artifact_sha256"] = "Unresolved diagnostic; not an acceptance claim"
cases.append(("unresolved_fixture_reaches_candidate_gate", unresolved, "FileNotFoundError"))
missing_candidates = OUT / "deliberately_absent_candidates"
if missing_candidates.exists():
    raise RuntimeError("The rejection sentinel must remain absent")
results = []
for name, fixture, expected in cases:
    receipt = OUT / (name + ".json")
    receipt.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    output = OUT / (name + "-analysis")
    if output.exists():
        raise RuntimeError("No analysis directory may exist in gate-only verification")
    command = [sys.executable, str(HERE / "explain.py"), "analyze",
               "--candidates", str(missing_candidates), "--family", str(FAMILY),
               "--numerical-checks", str(receipt), "--output", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    passed = (completed.returncode != 0 and expected in completed.stderr
              and not output.exists())
    results.append({"case": name, "fixture_sha256": sha(receipt), "command": command,
                    "expected_error": expected, "returncode": completed.returncode,
                    "analysis_output_created": output.exists(), "passed": passed,
                    "stderr_last_lines": completed.stderr.splitlines()[-4:]})
record = {"scope": "Receipt binding only; synthetic successes are not numerical acceptance",
          "producer_sha256": sha(HERE / "explain.py"),
          "reproduce_sha256": sha(Path(__file__)),
          "family_artifact_sha256": family["artifact_sha256"],
          "cases": results, "passed": all(item["passed"] for item in results)}
(OUT / "verification.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps({"cases": len(results), "passed": record["passed"]}))
if not record["passed"]:
    raise SystemExit(1)
