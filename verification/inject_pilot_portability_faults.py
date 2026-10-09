"""End-to-end fault probes for the public-pilot verifier in an isolated fixture.

The fixture copies only the two public pilot files and campaign metadata; it
symlinks read-only production source. Canonical campaign files are never edited.
This file was written before the corresponding verifier hardening.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = ("nan_cash", "nan_executed_price", "finite_unexecuted_price",
         "changed_stored_winner", "config_environment_order")
PILOTS = ("03_01", "08_02")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(path: Path, case: str) -> None:
    (path / "verification").mkdir(parents=True)
    (path / "configs").mkdir()
    (path / "outputs/full/pilots").mkdir(parents=True)
    (path / "src").symlink_to(ROOT / "src", target_is_directory=True)
    shutil.copy2(ROOT / "verification/verify_pilot_portability.py", path / "verification")
    shutil.copy2(ROOT / "configs/protocol.json", path / "configs")
    for filename in ("manifest.json", "pilot_fits.json"):
        shutil.copy2(ROOT / "outputs/full" / filename, path / "outputs/full")
    for name in PILOTS:
        shutil.copy2(ROOT / "outputs/full/pilots" / f"{name}.npz", path / "outputs/full/pilots")
    if case == "nan_cash":
        sys.path.insert(0, str(ROOT / "src"))
        from trade_learning.run import pilot_hash
        file = path / "outputs/full/pilots/03_01.npz"
        with np.load(file, allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
        arrays["cash"][0, 0] = np.nan
        np.savez_compressed(file, **arrays)
        new_hash = pilot_hash(arrays)
        manifest_path = path / "outputs/full/manifest.json"
        fit_path = path / "outputs/full/pilot_fits.json"
        manifest = json.loads(manifest_path.read_text())
        fits = json.loads(fit_path.read_text())
        manifest["streams"][31]["pilot_data_hash"] = new_hash
        fits[31]["pilot_data_hash"] = new_hash
        manifest_path.write_text(json.dumps(manifest))
        fit_path.write_text(json.dumps(fits))
    elif case == "changed_stored_winner":
        fit_path = path / "outputs/full/pilot_fits.json"
        fits = json.loads(fit_path.read_text())
        for index in (31, 82):
            weights = [-2.5] * 9
            weights[0] = -2.0 - 5e-10
            weights[1] = -2.0
            fits[index]["log_weights"] = weights
        fit_path.write_text(json.dumps(fits))
    elif case == "config_environment_order":
        config = path / "configs/protocol.json"
        raw = json.loads(config.read_text())
        raw["kappa_grid"] = list(reversed(raw["kappa_grid"]))
        config.write_text(json.dumps(raw))


def child(case: str, path: Path) -> None:
    target = path / "verification/verify_pilot_portability.py"
    spec = importlib.util.spec_from_file_location("fault_verifier", target)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if case in ("nan_cash", "nan_executed_price", "finite_unexecuted_price"):
        real_generate = module.generate_uniform_pilot

        def generated(**kwargs):
            arrays = {key: np.array(value, copy=True)
                      for key, value in real_generate(**kwargs).as_dict().items()}
            if kwargs["seed"] == json.loads((path / "outputs/full/manifest.json").read_text())["streams"][31]["pilot_seed"]:
                if case == "nan_cash":
                    arrays["cash"][0, 0] = np.nan
                elif case == "nan_executed_price":
                    position = tuple(np.argwhere(np.isfinite(arrays["execution_prices"]))[0])
                    arrays["execution_prices"][position] = np.nan
                else:
                    position = tuple(np.argwhere(np.isnan(arrays["execution_prices"]))[0])
                    arrays["execution_prices"][position] = 0.0
            return SimpleNamespace(as_dict=lambda: arrays)

        module.generate_uniform_pilot = generated
    if case == "changed_stored_winner":
        real_fit = module.fit_grid_posterior

        def near_tie(*args, **kwargs):
            actual = real_fit(*args, **kwargs)
            weights = np.array([-2.5] * 9)
            weights[0] = -2.0
            weights[1] = -2.0 - 5e-10
            return SimpleNamespace(log_weights=weights,
                                   log_likelihood=actual.log_likelihood,
                                   episode_log_likelihood=actual.episode_log_likelihood)

        module.fit_grid_posterior = near_tie
    sys.argv = [str(target), "--label", "fault"]
    module.main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("before", "after"))
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    if args.case:
        child(args.case, args.fixture)
        return
    if not args.phase:
        parser.error("--phase is required unless --case is supplied")
    canonical = {name: digest(ROOT / "outputs/full/pilots" / f"{name}.npz") for name in PILOTS}
    results = []
    (ROOT / "tmp").mkdir(exist_ok=True)
    for case in CASES:
        with tempfile.TemporaryDirectory(prefix="pilot-portability-fault-", dir=ROOT / "tmp") as directory:
            path = Path(directory)
            fixture(path, case)
            run = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                  "--case", case, "--fixture", str(path)],
                                 cwd=path, text=True, capture_output=True)
            artifact = path / "outputs/verification/reaudit/pilot_portability_fault.json"
            payload = json.loads(artifact.read_text()) if artifact.exists() else None
            results.append({"case": case, "exit_code": run.returncode,
                            "checker_passed": None if payload is None else payload["passed"],
                            "stderr_tail": run.stderr.strip().splitlines()[-2:],
                            "stdout_tail": run.stdout.strip().splitlines()[-7:]})
    assert canonical == {name: digest(ROOT / "outputs/full/pilots" / f"{name}.npz") for name in PILOTS}
    expected_before = {"nan_cash": 0, "nan_executed_price": 1,
                       "finite_unexecuted_price": 1, "changed_stored_winner": 0,
                       "config_environment_order": 0}
    expected = expected_before if args.phase == "before" else dict.fromkeys(CASES, 1)
    ok = all((r["exit_code"] == expected[r["case"]]) for r in results)
    receipt = {"phase": args.phase, "expectation_met": ok,
               "canonical_archive_sha256": canonical, "results": results}
    out = ROOT / "outputs/verification/reaudit" / f"pilot_portability_fault_injection_{args.phase}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
