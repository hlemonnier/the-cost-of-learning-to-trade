"""E2E equivalence of horizon-major execution to the preserved policy-major run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(HERE)]
from solver import SolverSpec, solve_family


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError("Use fresh comparison outputs")
    args.out.mkdir(parents=True)
    family = solve_family(SolverSpec(weight_points=9, belief_points=17,
                                    quadrature_points=15, belief_geometry="endpoint_sine"), args.cache)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "TRADE_LEARNING_CORE_ROOT": str(ROOT)}
    old = HERE / "revisions/run_policy_major.py"
    command = ("import importlib.util,sys;from pathlib import Path;"
               "old=Path(sys.argv.pop(1)).resolve();ext=Path(sys.argv.pop(1)).resolve();"
               "sys.path.insert(0,str(ext));"
               "spec=importlib.util.spec_from_file_location('legacy_policy_major',old);"
               "m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);"
               "m.HERE=ext;m.main()")
    for name, start in (("policy_major", [sys.executable, "-c", command, str(old), str(HERE)]),
                        ("horizon_major", [sys.executable, str(HERE / "run.py")])):
        with (args.out / (name + ".txt")).open("w") as log:
            subprocess.run(start + ["--profile", "smoke", "--tables", str(args.cache),
                                   "--out", str(args.out / name)], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    def files(folder):
        return {str(p.relative_to(folder)): digest(p) for p in folder.rglob("*")
                if p.is_file() and p.name not in ("manifest.json", "timings.csv")}
    reference, current = files(args.out / "policy_major"), files(args.out / "horizon_major")
    if reference != current:
        raise AssertionError({"differing": [name for name in sorted(set(reference) | set(current))
                                           if reference.get(name) != current.get(name)]})
    receipt = {"passed": True, "scope": "720-record end-to-end schedule equivalence; no final outcomes",
               "legacy_run_sha256": digest(old), "current_run_sha256": digest(HERE / "run.py"),
               "solver_sha256": digest(HERE / "solver.py"), "table_artifact_sha256": family.metadata["artifact_sha256"],
               "identical_nontiming_files": reference, "count": len(reference),
               "excluded": ["manifest.json: distinct runner provenance and elapsed time", "timings.csv: execution timing"]}
    (args.out / "verification.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "identical_nontiming_files": len(reference)}))


if __name__ == "__main__":
    main()
