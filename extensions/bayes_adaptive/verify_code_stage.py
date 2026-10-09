"""Bounded command-level rejection checks for the theory/code checkpoint.

Uses a complete coarse smoke and copied scratch tables. It never generates the
final economic sample, changes the protocol, or treats this as final science.
Failure cases were declared in validation_failure_modes.md and
packaging_failure_modes.md before the corresponding code changes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def check(value, message):
    if not value:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "extension", "study", "cold-tables", "original-zip", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    root, ext, study, tables, original, out = (p.resolve() for p in
        (args.root, args.extension, args.study, args.cold_tables, args.original_zip, args.out))
    check(not out.exists(), "Verification output must be new")
    out.mkdir(parents=True)
    complete = json.loads((tables / "complete.json").read_text())
    check(complete["specification"]["weight_points"] == 9 and
          complete["specification"]["belief_points"] == 17, "Only the bounded coarse family is allowed")
    cases = []

    def command(name, cmd, expected):
        result = subprocess.run(cmd, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out / f"{name}.txt").write_text(result.stdout)
        passed = result.returncode != 0 and expected in result.stdout
        cases.append({"case": name, "passed": passed, "command": cmd,
                      "returncode": result.returncode, "expected_error": expected,
                      "log_sha256": sha(out / f"{name}.txt")})
        check(passed, f"Unexpected result for {name}; inspect its command log")

    base = [sys.executable, str(ext / "verify_extension.py"), "--root", str(root), "--extension", str(ext),
            "--study", str(study), "--profile", "smoke"]
    receipt = {"passed": False, "scope": "Bounded code-stage E2E; no final numerical/economic acceptance",
               "source_sha256": sha(Path(__file__)), "independent_verifier_sha256": sha(ext / "verify_extension.py"),
               "study_manifest_sha256": sha(study / "manifest.json"), "cases": cases}
    try:
        command("new_metadata_requires_optin", base+["--tables", str(tables)], "requires explicit opt-in")
        command("optin_requires_tables", base+["--allow-array-equivalent-build"], "requires --tables")
        with tempfile.TemporaryDirectory(prefix="bayes-code-copied-cache-") as temporary:
            copied = Path(temporary) / "tables"
            shutil.copytree(tables, copied)
            metadata_path = copied / "complete.json"
            mutations = {
                "wrong_format": lambda x: x.update(format_version=1),
                "different_source": lambda x: x["source"]["files"].update({"solver.py": "0"*64}),
                "different_build": lambda x: x["build"].update(python="deliberately different"),
                "different_specification": lambda x: x["specification"].update(belief_points=19),
                "different_retained_actions": lambda x: x.update(retain_q=["bayes"]),
                "different_array_record": lambda x: x["arrays"][0].update(dtype="float32"),
            }
            for name, mutate in mutations.items():
                changed = copy.deepcopy(complete)
                mutate(changed)
                changed["artifact_sha256"] = canonical({k: v for k, v in changed.items() if k != "artifact_sha256"})
                metadata_path.write_text(json.dumps(changed))
                command(name, base+["--tables", str(copied), "--allow-array-equivalent-build"], "VerificationError")
            changed = copy.deepcopy(complete)
            changed["artifact_sha256"] = "0"*64
            metadata_path.write_text(json.dumps(changed))
            command("false_artifact_signature", base+["--tables", str(copied), "--allow-array-equivalent-build"], "manifest identity mismatch")
            metadata_path.write_text(json.dumps(complete))
            array = np.load(copied / "q-bayes.npy", mmap_mode="r+")
            array[1, 2, 1, 100, 0] += .125
            array.flush()
            del array
            command("changed_actual_array", base+["--tables", str(copied), "--allow-array-equivalent-build"], "Missing or changed control array")
            changed = copy.deepcopy(complete)
            for item in changed["arrays"]:
                if item["file"] == "q-bayes.npy":
                    item["sha256"] = sha(copied / item["file"])
            changed["artifact_sha256"] = canonical({k: v for k, v in changed.items() if k != "artifact_sha256"})
            metadata_path.write_text(json.dumps(changed))
            command("changed_and_rehashed_array", base+["--tables", str(copied), "--allow-array-equivalent-build"], "Control arrays differ")

        numerical = ext / "outputs/numerical-amended/numerical_checks.json"
        rejected_economic = out / "must_not_create_economic"
        command("unresolved_numerics_block_economics", [sys.executable, str(ext / "run.py"), "--profile", "full",
                "--tables", str(tables), "--numerical-checks", str(numerical), "--out", str(rejected_economic)],
                "Numerical resolution is not accepted")
        check(not rejected_economic.exists(), "Rejected numerical evidence created economic outputs")
        command("smoke_cannot_be_final_report", [sys.executable, str(ext / "build_report.py"), "--study", str(study),
                "--numerical-checks", str(numerical), "--validation", str(ext / "outputs/independent_smoke_v2_current_source.json"),
                "--explanation", str(ext / "outputs/code_stage/explanation_pipeline"),
                "--cold-validation", str(ext / "outputs/code_stage/cold_reproduction/verification.json"),
                "--out", str(out / "must_not_create_final_report")], "Report requires the complete prespecified full campaign")
        include = out / "incomplete_include_fixture.json"
        include.write_text(json.dumps({"format_version": 1, "files": [{"source": "extensions/bayes_adaptive/README.md",
            "target": "extensions/bayes_adaptive/README.md", "bytes": (ext/"README.md").stat().st_size,
            "sha256": sha(ext/"README.md")}]}))
        command("incomplete_science_cannot_be_final_package", [sys.executable, str(ext / "package_review.py"),
                "--original", str(original), "--include", str(include), "--output", str(out / "must_not_create_final.zip")],
                "Required final science, validation or reviewer navigation is absent")

        # Real archive transport, deliberately minimal and labelled as a fixture.
        with tempfile.TemporaryDirectory(prefix="bayes-code-transport-") as temporary:
            fixture = Path(temporary) / "fixture.zip"
            payload = {"START_HERE.md": b"Development transport fixture; not a submission.\n",
                       "verify_review.py": (ext/"verify_review.py").read_bytes(),
                       "extensions/bayes_adaptive/README.md": b"Development fixture.\n",
                       "extensions/bayes_adaptive/REPRODUCING.md": b"Code-stage checksum fixture.\n"}
            files = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in payload.items()}
            files[original.name] = {"bytes": original.stat().st_size, "sha256": sha(original)}
            manifest = {"format_version": 1, "original_zip_sha256": sha(original),
                        "research_core_snapshot": "0c13a03ec83212ca1c752959ce5cfdd7588b63c3", "files": files,
                        "purpose": "bounded development transport fixture"}
            def make_fixture(change=None):
                with zipfile.ZipFile(fixture, "w", compression=zipfile.ZIP_STORED) as archive:
                    archive.write(original, original.name)
                    for name, data in payload.items():
                        archive.writestr(name, data[:-1]+b"x" if change == "content" and name == "START_HERE.md" else data)
                    archive.writestr("review_manifest.json", json.dumps(manifest))
                    if change == "traversal":
                        archive.writestr("../outside.txt", b"must reject")
                    if change == "extra":
                        archive.writestr("extra.txt", b"must reject")
            make_fixture()
            good = subprocess.run([sys.executable, str(ext/"verify_review.py"), "--archive", str(fixture)],
                                  text=True, capture_output=True)
            check(good.returncode == 0, "Valid development transport did not authenticate")
            receipt["valid_transport"] = json.loads(good.stdout)
            for fault, expected in (("content", "Content mismatch"), ("traversal", "Unsafe ZIP member"),
                                    ("extra", "ZIP payload differs")):
                make_fixture(fault)
                command("transport_"+fault, [sys.executable, str(ext/"verify_review.py"), "--archive", str(fixture)], expected)
        for item in complete["arrays"]:
            check(sha(tables/item["file"]) == item["sha256"], "Scratch checks changed the original cold cache")
        receipt.update(passed=True, rejection_cases=len(cases), original_cache_unchanged=True)
    finally:
        (out / "verification.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"passed": receipt["passed"], "rejection_cases": len(cases)}))


if __name__ == "__main__":
    main()
