"""Cold-process resource profile for one canonical practical active control table.

Run from repository root:
  .venv/bin/python verification/profile_resources.py \
    --artifact outputs/control_cache/reaudit_canonical_cold_20260924 \
    --output outputs/verification/reaudit/resource_envelope.json

The child solves directly, then serializes. It never loads an existing table.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SETTINGS = {"theta": .35, "kappa": .02, "horizon": 300, "qmax": 5,
            "belief_points": 641, "quadrature_points": 161, "mode": "active"}


def allocated_and_logical_bytes(path: Path) -> dict:
    files = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            stat = item.stat()
            files.append({"path": str(item.relative_to(ROOT)), "logical_bytes": stat.st_size,
                          "allocated_bytes": stat.st_blocks * 512})
    return {"files": files,
            "logical_bytes": sum(x["logical_bytes"] for x in files),
            "allocated_bytes": sum(x["allocated_bytes"] for x in files)}


def child(artifact: Path, child_result: Path) -> None:
    from trade_learning.control import save_control, solve_control

    if artifact.exists():
        raise FileExistsError(f"Cold artifact already exists: {artifact}")
    started = time.perf_counter()
    solution = solve_control(**SETTINGS)
    solve_elapsed = time.perf_counter() - started
    if (solution.theta, solution.kappa, solution.horizon, solution.qmax,
        len(solution.beliefs), solution.quadrature_points, solution.mode) != (
        .35, .02, 300, 5, 641, 161, "active"):
        raise AssertionError("Returned solution has noncanonical settings")
    save_control(artifact, solution)
    child_result.write_text(json.dumps({"settings": SETTINGS, "solver_metadata": solution.metadata,
                                        "solve_elapsed_seconds": solve_elapsed,
                                        "solve_and_save_elapsed_seconds": time.perf_counter()-started},
                                       indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    output = args.output.resolve()
    if args.child:
        if args.child_result is None:
            raise ValueError("Child result path missing")
        child(artifact, args.child_result)
        return
    if artifact.exists():
        raise FileExistsError(f"Fresh output directory required: {artifact}")
    output.parent.mkdir(parents=True, exist_ok=True)
    child_result = output.with_name(output.stem + "_child.json")
    log = output.with_name(output.stem + "_child.log")
    if child_result.exists() or log.exists():
        raise FileExistsError("Fresh child result and log paths required")
    command = [sys.executable, str(Path(__file__).resolve()), "--artifact", str(artifact),
               "--output", str(output), "--child", "--child-result", str(child_result)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    import numpy
    import scipy
    recorded = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "working_directory": str(ROOT), "command": command,
                "environment_override": {"PYTHONPATH": env["PYTHONPATH"]},
                "settings": SETTINGS,
                "host": {"platform": platform.platform(), "machine": platform.machine(),
                         "python": sys.version, "numpy": numpy.__version__,
                         "scipy": scipy.__version__, "cpu_count": os.cpu_count()},
                "scope": "one new Python process: direct solve_control then save_control; no cache load"}
    if sys.platform == "darwin":
        recorded["host"]["physical_memory_bytes"] = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True).strip())
    start = time.perf_counter()
    sampled_max = 0
    samples = 0
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        while True:
            try:
                sampled_kib = int(subprocess.check_output(
                    ["ps", "-o", "rss=", "-p", str(process.pid)], text=True,
                    stderr=subprocess.DEVNULL).strip())
                sampled_max = max(sampled_max, sampled_kib * 1024)
                samples += 1
            except (subprocess.CalledProcessError, ValueError):
                pass
            waited, status, usage = os.wait4(process.pid, os.WNOHANG)
            if waited:
                break
            time.sleep(.05)
    process.returncode = os.waitstatus_to_exitcode(status)
    recorded.update({"finished_utc": datetime.now(timezone.utc).isoformat(),
                     "elapsed_seconds": time.perf_counter()-start,
                     "child_exit_code": process.returncode,
                     "peak_process_rss_bytes": usage.ru_maxrss if sys.platform == "darwin"
                                               else usage.ru_maxrss * 1024,
                     "peak_process_rss_source": "wait4 rusage.ru_maxrss",
                     "peak_process_rss_unit_conversion": "bytes on Darwin; KiB multiplied by 1024 on Linux",
                     "sampled_max_rss_bytes": sampled_max, "rss_samples": samples,
                     "rss_sample_method": "ps rss KiB at about 50 ms intervals; may miss a brief peak",
                     "child_log": str(log.relative_to(ROOT))})
    if process.returncode != 0 or not child_result.exists() or not (artifact / "metadata.json").exists():
        recorded["passed"] = False
        output.write_text(json.dumps(recorded, indent=2) + "\n")
        raise RuntimeError(f"Cold resource profile failed; see {log}")
    from trade_learning.control import load_control

    loaded = load_control(artifact, mmap_mode="r")
    if (loaded.theta, loaded.kappa, loaded.horizon, loaded.qmax,
        len(loaded.beliefs), loaded.quadrature_points, loaded.mode) != (
        .35, .02, 300, 5, 641, 161, "active"):
        raise AssertionError("Serialized cold control artifact has noncanonical settings")
    recorded["artifact"] = str(artifact.relative_to(ROOT))
    recorded["artifact_bytes"] = allocated_and_logical_bytes(artifact)
    recorded["artifact_record"] = {
        key: loaded.artifact[key] for key in
        ("format_version", "specification", "key", "source", "build", "artifact_sha256")}
    recorded["child_result"] = json.loads(child_result.read_text())
    recorded["child_result_path"] = str(child_result.relative_to(ROOT))
    recorded["validation"] = "load_control validated complete artifact and authenticated arrays"
    recorded["passed"] = True
    output.write_text(json.dumps(recorded, indent=2) + "\n")
    print(json.dumps({"output": str(output), "passed": True,
                      "peak_process_rss_bytes": recorded["peak_process_rss_bytes"],
                      "artifact_logical_bytes": recorded["artifact_bytes"]["logical_bytes"],
                      "elapsed_seconds": recorded["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
