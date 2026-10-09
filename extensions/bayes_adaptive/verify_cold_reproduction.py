"""Cold selected-family rebuild and independent end-to-end smoke reconciliation.

All output caches are newly created. Content-equivalent tables retain their own
metadata identity; this command never restamps an artifact or changes the study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from verify_extension import authenticate_grid_geometry, canonical_hash


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def run(args):
    started = time.perf_counter()
    core, ext, reference, cache, out = (path.resolve() for path in
                                      (args.root, args.extension, args.reference_smoke, args.cache, args.out))
    require(not cache.exists() and not out.exists(), "Cold cache and evidence destination must both be new")
    require((core / "src/trade_learning/model.py").is_file(), "Extracted original core is missing")
    prior = read(reference / "manifest.json")
    require(prior["profile"] == "smoke" and prior["status"] == "complete" and
            prior["policy_records"] == 720, "An authenticated completed reference smoke is required")
    require(prior.get("numerical_acceptance_sha256") is None and
            prior.get("numerical_contract", {}).get("version") == 2,
            "Cold reference must be a development smoke under the frozen numerical contract")
    for name, expected in prior["files"].items():
        require(sha(reference / name) == expected, f"Reference smoke changed: {name}")
    for group, directory in (("core", core), ("extension", ext)):
        for name, expected in prior["inputs"][group].items():
            require(sha(directory / name) == expected, f"Supplied {group} source differs from the reference: {name}")
    out.mkdir(parents=True)
    record = {"passed": False, "cold_cache_existed_at_start": False,
              "reference_manifest_sha256": sha(reference / "manifest.json"),
              "reference_table_artifact_sha256": prior["table_artifact_sha256"],
              "verifier_sha256": sha(Path(__file__)), "commands": [],
              "scope": "Independent cold build on one numerical stack; same seeds, not additional research observations"}
    env = dict(os.environ, TRADE_LEARNING_CORE_ROOT=str(core), PYTHONPATH=str(core / "src"))

    def execute(label, command):
        print(json.dumps({"phase": label, "command": command}), flush=True)
        command_start = time.perf_counter()
        with (out / f"{label}.txt").open("w") as log:
            result = subprocess.run(command, cwd=core.parent, env=env, stdout=log, stderr=subprocess.STDOUT)
        record["commands"].append({"label": label, "command": command,
                                   "returncode": result.returncode,
                                   "seconds": time.perf_counter()-command_start,
                                   "log_sha256": sha(out / f"{label}.txt")})
        require(result.returncode == 0, f"{label} failed with exit {result.returncode}; see preserved log")

    try:
        spec = prior["table_specification"]
        require(spec["theta"] == .35 and spec["kappas"] == [.002, .1] and spec["qmax"] == 2 and
                spec["horizon"] == 30 and spec["known_belief_points"] == 321 and
                spec["known_quadrature_points"] == 161 and
                spec.get("belief_geometry") in ("uniform", "endpoint_sine"),
                "Unsupported selected-family model or missing V3 geometry")
        execute("cold_build", [sys.executable, str(ext / "solver.py"), "solve", "--directory", str(cache),
                               "--weight-points", str(spec["weight_points"]), "--belief-points", str(spec["belief_points"]),
                               "--quadrature-points", str(spec["quadrature_points"]), "--horizon", str(spec["horizon"]),
                               "--belief-geometry", spec["belief_geometry"]])
        complete = read(cache / "complete.json")
        record["cold_table_artifact_sha256"] = complete["artifact_sha256"]
        axes = authenticate_grid_geometry(complete)
        record["cold_grid_geometry_sha256"] = complete["grid_geometry"]["axes_sha256"]
        expected_source_files = {
            "solver.py": prior["inputs"]["extension"]["solver.py"],
            **{f"trade_learning.{module}": prior["inputs"]["core"][f"src/trade_learning/{module}.py"]
               for module in ("control", "filtering", "model", "numerics")},
        }
        require(complete["source"] == {"files": expected_source_files,
                                       "sha256": canonical_hash(expected_source_files)} and
                complete["specification"] == spec and complete["build"] == prior["build"] and
                complete["numerical_contract"] == prior["numerical_contract"] and
                complete["arrays"] == prior["table_arrays"] and
                len(axes) == 3, "Cold table is not exactly equivalent on this build")
        for item in complete["arrays"]:
            require(sha(cache / item["file"]) == item["sha256"], f"Cold array content changed: {item['file']}")
        record["arrays_identical"] = True
        record["array_count"] = len(complete["arrays"])
        record["array_bytes"] = sum(item["bytes"] for item in complete["arrays"])
        write(out / "cold_complete.json", complete)
        execute("cold_smoke", [sys.executable, str(ext / "run.py"), "--profile", "smoke",
                               "--tables", str(cache), "--out", str(out / "smoke")])
        command = [sys.executable, str(ext / "verify_extension.py"), "--root", str(core), "--extension", str(ext),
                   "--study", str(out / "smoke"), "--profile", "smoke", "--tables", str(cache),
                   "--candidates", str(args.candidates.resolve()), "--original-zip", str(args.original_zip.resolve()),
                   "--out", str(out / "independent_smoke.json")]
        execute("independent_smoke", command)
        checked = read(out / "independent_smoke.json")
        require(checked["passed"] is True and checked["rows"] == 720 and
                checked["policy_decisions_recomputed"] == 21600, "Complete independent smoke replay missing")
        current = read(out / "smoke/manifest.json")
        require(current["inputs"] == prior["inputs"] and current["build"] == prior["build"] and
                current["protocol"] == prior["protocol"] and set(current["files"]) == set(prior["files"]),
                "Cold/reference smoke source, build, protocol or output inventory differs")
        compared = {}
        for name in prior["files"]:
            if name == "timings.csv":
                continue
            actual = sha(out / "smoke" / name)
            require(actual == prior["files"][name] == current["files"][name], f"Cold economic input/output differs: {name}")
            compared[name] = actual
        record.update({"passed": True, "smoke_economic_files_identical": True,
                       "compared_files": compared, "policy_records": 720, "paired_market_trajectories": 48,
                       "independently_replayed_decisions": 21600,
                       "independent_verifier_sha256": checked["verifier_sha256"],
                       "cold_manifest_sha256": sha(out / "smoke/manifest.json"),
                       "independent_smoke_sha256": sha(out / "independent_smoke.json"),
                       "identity_note": "Artifact metadata may differ through measured solve time; all eleven array records and actual bytes must match"})
    except Exception as error:
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        record["seconds"] = time.perf_counter()-started
        write(out / "verification.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "extension", "reference-smoke", "cache", "out", "candidates", "original-zip"):
        parser.add_argument("--"+name, type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("passed", "array_count", "policy_records", "seconds")}))


if __name__ == "__main__":
    main()
