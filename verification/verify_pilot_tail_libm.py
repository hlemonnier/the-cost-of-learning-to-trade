"""Reconstruct the two Linux-vs-macOS normal-tail draws from exact PCG bits.

This is a focused diagnostic companion to verify_pilot_portability.py. Its
failure modes and numerical interpretation are recorded in
pilot_portability_failure_modes.md. Run once on each platform and compare JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOR_R = 3.6541528853610087963519472518
NOR_INV_R = 0.27366123732975827203338247596
CASES = ((3, 1, 49, 163, 15162), (8, 2, 35, 252, 10964))


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/verification/reaudit")
    args = parser.parse_args()
    require(args.label.isidentifier(), "label must be an identifier")
    manifest_path = ROOT / "outputs/full/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest["protocol"]
    records = []
    for environment, pilot, episode, period, raw_index in CASES:
        name = f"{environment:02d}_{pilot:02d}"
        stream = manifest["streams"][10 * environment + pilot]
        seed = int(np.random.SeedSequence([11, environment, pilot, protocol["master_seed"]])
                   .generate_state(1, dtype=np.uint64)[0])
        require(stream["pilot_seed"] == seed, f"{name}: seed mismatch")
        z_seed = np.random.SeedSequence(seed).spawn(6)[2]
        raw = int(np.random.PCG64(z_seed).random_raw(raw_index + 1)[raw_index])
        u = float((raw >> 11) / 2**53)
        log1p_value = math.log1p(-u)
        tail = NOR_R - NOR_INV_R * log1p_value
        z = float(np.random.default_rng(z_seed).standard_normal((100, 300))[episode, period])
        require(z == -tail, f"{name}: raw bits do not explain the tail normal")
        original_path = ROOT / "outputs/full/pilots" / f"{name}.npz"
        with np.load(original_path, allow_pickle=False) as data:
            signal = int(data["signal"][episode, period])
            original_return = float(data["return_"][episode, period])
        regenerated_return = float(.03 * signal + .3 * z)
        require(signal == 0, f"{name}: unexpected signal in focused draw")
        require(abs(regenerated_return - original_return) <= 2.220446049250313e-16,
                f"{name}: difference exceeds one ULP at observed magnitude")
        records.append({"pilot": name, "episode": episode, "period": period,
                        "raw_index_in_z_component": raw_index, "raw_uint64": raw,
                        "uniform_hex": u.hex(), "log1p_minus_uniform_hex": log1p_value.hex(),
                        "tail_magnitude_hex": tail.hex(), "regenerated_z_hex": z.hex(),
                        "signal": signal, "original_return_hex": original_return.hex(),
                        "regenerated_return_hex": regenerated_return.hex(),
                        "original_archive_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
                        "return_ulp_steps": int(abs(np.float64(original_return).view(np.uint64).item()
                                                   - np.float64(regenerated_return).view(np.uint64).item()))})
    receipt = {"runtime": {"python": platform.python_version(), "numpy": np.__version__,
                            "system": platform.system(), "machine": platform.machine()},
               "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               "numpy_source": "https://github.com/numpy/numpy/blob/v2.3.5/numpy/random/src/distributions/distributions.c",
               "constant_source": "https://github.com/numpy/numpy/blob/v2.3.5/numpy/random/src/distributions/ziggurat_constants.h",
               "passed": True, "cases": records}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"pilot_tail_libm_{args.label}.json"
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"path": str(path), "cases": records}, indent=2))


if __name__ == "__main__":
    main()
