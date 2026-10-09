"""Operational supervision of the frozen endpoint-geometry numerical refinement.

Stdout/stderr persist outside the Codex process session. This wrapper records
observed process RSS, elapsed time, available disk and the actual child exit.
It changes no solver equations, configuration, environment or cache identity.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
EXT = ROOT / "extensions/bayes_adaptive"
HERE = Path(__file__).resolve().parent
OUT = EXT / "outputs/numerical-endpoint"
SUMMARY = OUT / "resources_summary.json"


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    solver_hash = hashlib.sha256((EXT / "solver.py").read_bytes()).hexdigest()
    if solver_hash != "8fdf88cae1b6b74ebd738e8b07942c15fd99042e2fd4f80f112660fc77f9e1ff":
        raise RuntimeError("Frozen solver changed before detached launch")
    process_rows = subprocess.check_output(["ps", "-Ao", "pid=,command="], text=True)
    if any("/refine_geometry.py --output" in row for row in process_rows.splitlines()):
        raise RuntimeError("A refinement process already exists; no duplicate launch")
    command = [str(ROOT / ".venv/bin/python"), str(EXT / "refine_geometry.py"),
               "--output", str(OUT), "--cache", str(EXT / "cache/v3_endpoint")]
    started = time.monotonic()
    with (OUT / "progress.jsonl").open("a", buffering=1) as log:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, close_fds=True)
        awake = subprocess.Popen(["caffeinate", "-i", "-w", str(child.pid)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        receipt = {"started_utc": utc(), "worker_pid": child.pid,
                   "command": command, "solver_sha256": solver_hash,
                   "exit_code": None, "process_running": True,
                   "launch_mode": "Supervisor detached with start_new_session=True, stdin /dev/null and file stdout/stderr"}
        write_json(HERE / "execution_receipt.json", receipt)
        summary = {"sessions": []}
        session = {"pid": child.pid, "started_utc": receipt["started_utc"],
                   "process_running": True, "sampled_peak_rss_bytes": 0,
                   "lowest_observed_disk_available_bytes": shutil.disk_usage(ROOT).free,
                   "samples": 0}
        summary["sessions"].append(session)
        with (OUT / "resources.jsonl").open("a", buffering=1) as resources:
            while child.poll() is None:
                observed = subprocess.run(["ps", "-p", str(child.pid), "-o", "rss=,%cpu=,etime="],
                                          text=True, capture_output=True, check=False).stdout.strip()
                free = shutil.disk_usage(ROOT).free
                if observed:
                    rss, cpu, elapsed = observed.split(maxsplit=2)
                    sample = {"utc": utc(), "pid": child.pid, "rss_bytes": int(rss)*1024,
                              "cpu_percent": float(cpu), "process_elapsed": elapsed,
                              "disk_available_bytes": free}
                    resources.write(json.dumps(sample, sort_keys=True) + "\n")
                    session["samples"] += 1
                    session["sampled_peak_rss_bytes"] = max(session["sampled_peak_rss_bytes"], sample["rss_bytes"])
                    session["lowest_observed_disk_available_bytes"] = min(session["lowest_observed_disk_available_bytes"], free)
                summary.update(active_pid=child.pid, process_running=True,
                               sampled_peak_rss_bytes=max(s["sampled_peak_rss_bytes"] for s in summary["sessions"]),
                               lowest_observed_disk_available_bytes=min(s["lowest_observed_disk_available_bytes"] for s in summary["sessions"]))
                write_json(SUMMARY, summary)
                time.sleep(15)
        code = child.wait()
    session.update(process_running=False, completed_utc=utc(), exit_code=code,
                   wall_seconds=time.monotonic()-started)
    summary.update(active_pid=None, process_running=False)
    write_json(SUMMARY, summary)
    receipt.update(process_running=False, completed_utc=utc(), exit_code=code,
                   wall_seconds=session["wall_seconds"])
    write_json(HERE / "execution_receipt.json", receipt)
    return code


if __name__ == "__main__":
    sys.exit(main())
