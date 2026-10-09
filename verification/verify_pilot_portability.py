"""Regenerate public pilots and compare raw arrays across numerical builds.

Run with the intended Python interpreter, e.g.
  .venv/bin/python verification/verify_pilot_portability.py --all --label supported
  python3 verification/verify_pilot_portability.py --all --label alternate

Failure contract: verification/pilot_portability_failure_modes.md.
The original public archives and stored fits are read-only inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trade_learning.environment import generate_exogenous, generate_uniform_pilot  # noqa: E402
from trade_learning.filtering import fit_grid_posterior  # noqa: E402
from trade_learning.policies import CANDIDATE_THETA, CANDIDATE_KAPPA  # noqa: E402
from trade_learning.protocol import load_configuration, resolve_protocol, protocol_fingerprint  # noqa: E402
from trade_learning.run import pilot_hash, seed_for  # noqa: E402

PUBLIC_FIELDS = (
    "price", "signal", "cash", "inventory", "actions", "action_probability",
    "return_", "submitted", "fills", "execution_prices", "executed_quantities", "fees",
)
DISCRETE = {"signal", "inventory", "actions", "submitted", "fills", "executed_quantities"}
ABS_LIMIT = {
    "price": 1e-9, "cash": 1e-9, "execution_prices": 1e-9,
    "return_": 1e-12, "fees": 1e-12, "action_probability": 1e-14,
}
FIT_LIMIT = 1e-9
DETAILED = {(3, 1), (8, 2)}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def field_hash(array: np.ndarray) -> str:
    return sha_bytes(np.ascontiguousarray(array).tobytes())


def ulp_max(left: np.ndarray, right: np.ndarray) -> int:
    """Maximum representable-float64 steps, monotone through negative values."""
    if left.size == 0:
        return 0
    aa = np.ascontiguousarray(left, dtype=np.float64).view(np.uint64)
    bb = np.ascontiguousarray(right, dtype=np.float64).view(np.uint64)
    sign = np.uint64(0x8000000000000000)
    oa = np.where((aa & sign) != 0, ~aa, aa ^ sign)
    ob = np.where((bb & sign) != 0, ~bb, bb ^ sign)
    distance = np.maximum(oa, ob) - np.minimum(oa, ob)
    return int(distance.max())


def compare_field(name: str, original: np.ndarray, regenerated: np.ndarray,
                  original_executed: np.ndarray | None = None,
                  regenerated_executed: np.ndarray | None = None) -> dict:
    require(original.shape == regenerated.shape, f"{name}: shape mismatch")
    require(original.dtype == regenerated.dtype, f"{name}: dtype mismatch")
    base = {"shape": list(original.shape), "dtype": str(original.dtype),
            "original_sha256": field_hash(original), "regenerated_sha256": field_hash(regenerated)}
    if name in DISCRETE:
        n = int(np.count_nonzero(original != regenerated))
        base.update(differing=n, exact=n == 0, passed=n == 0,
                    first_differing_indices=np.argwhere(original != regenerated)[:5].tolist())
        return base
    if name == "execution_prices":
        require(original_executed is not None and regenerated_executed is not None,
                "execution prices require execution masks")
        require(original_executed.shape == original.shape and regenerated_executed.shape == regenerated.shape,
                "execution price and quantity shapes differ")
        expected_original_nan = original_executed == 0
        expected_regenerated_nan = regenerated_executed == 0
    else:
        require(original_executed is None and regenerated_executed is None,
                "execution masks supplied for another field")
        expected_original_nan = np.zeros(original.shape, dtype=bool)
        expected_regenerated_nan = np.zeros(regenerated.shape, dtype=bool)
    nan_original = np.isnan(original)
    nan_regenerated = np.isnan(regenerated)
    nan_mismatch = int(np.count_nonzero(nan_original != nan_regenerated))
    invalid_original_nan_mask = int(np.count_nonzero(nan_original != expected_original_nan))
    invalid_regenerated_nan_mask = int(np.count_nonzero(nan_regenerated != expected_regenerated_nan))
    nonfinite = int(np.count_nonzero(~np.isfinite(original[~expected_original_nan]))
                    + np.count_nonzero(~np.isfinite(regenerated[~expected_regenerated_nan])))
    finite = np.isfinite(original) & np.isfinite(regenerated)
    a, b = original[finite], regenerated[finite]
    delta = np.abs(a - b)
    maximum = float(delta.max()) if delta.size else 0.0
    n = int(np.count_nonzero(a != b))
    limit = ABS_LIMIT[name]
    positions = np.argwhere(finite & (original != regenerated))[:5]
    examples = [{"index": point.tolist(),
                 "original_hex": float(original[tuple(point)]).hex(),
                 "regenerated_hex": float(regenerated[tuple(point)]).hex()}
                for point in positions]
    base.update(differing=n, max_abs=maximum, max_ulp=ulp_max(a, b),
                first_differences=examples,
                nan_mask_mismatch=nan_mismatch,
                invalid_original_nan_mask=invalid_original_nan_mask,
                invalid_regenerated_nan_mask=invalid_regenerated_nan_mask,
                nonfinite=nonfinite,
                absolute_limit=limit,
                passed=(nan_mismatch == 0 and invalid_original_nan_mask == 0
                        and invalid_regenerated_nan_mask == 0 and nonfinite == 0
                        and maximum <= limit))
    return base


def verify_fit(original: dict, generated: dict, stored: dict) -> dict:
    # All three fits use the public arrays only; no private evaluator state.
    fit_original = fit_grid_posterior(original, CANDIDATE_THETA, CANDIDATE_KAPPA)
    fit_generated = fit_grid_posterior(generated, CANDIDATE_THETA, CANDIDATE_KAPPA)
    saved = np.asarray(stored["log_weights"], dtype=np.float64)
    require(saved.shape == fit_original.log_weights.shape, "stored fit shape mismatch")
    require(np.array_equal(np.asarray(stored["candidate_theta"]), CANDIDATE_THETA), "theta order mismatch")
    require(np.array_equal(np.asarray(stored["candidate_kappa"]), CANDIDATE_KAPPA), "kappa order mismatch")
    delta_saved = fit_original.log_weights - saved
    delta_regenerated = fit_generated.log_weights - fit_original.log_weights
    winner = int(np.argmax(fit_original.log_weights))
    stored_winner = int(np.argmax(saved))
    return {
        "original_public_fit_log_likelihood": fit_original.log_likelihood.tolist(),
        "original_public_episode_log_likelihood": fit_original.episode_log_likelihood.tolist(),
        "stored_vs_original_log_weight_delta": delta_saved.tolist(),
        "regenerated_vs_original_log_weight_delta": delta_regenerated.tolist(),
        "stored_max_abs": float(np.max(np.abs(delta_saved))),
        "regenerated_max_abs": float(np.max(np.abs(delta_regenerated))),
        "winning_model_index_original": winner,
        "winning_model_index_stored": stored_winner,
        "winning_model_index_regenerated": int(np.argmax(fit_generated.log_weights)),
        "log_weight_absolute_limit": FIT_LIMIT,
        "passed": (np.isfinite(saved).all() and np.isfinite(fit_original.log_weights).all()
                   and np.isfinite(fit_generated.log_weights).all()
                   and float(np.max(np.abs(delta_saved))) <= FIT_LIMIT
                   and float(np.max(np.abs(delta_regenerated))) <= FIT_LIMIT
                   and winner == stored_winner
                   and winner == int(np.argmax(fit_generated.log_weights))),
    }


def export_original_hex(original: dict, destination: Path) -> dict:
    """Plain-text public observations, one episode/period per line for future review."""
    columns = ("episode", "period", "price_hex", "signal", "cash_hex", "inventory",
               "actions", "action_probability_hex", "return_hex", "submitted_bid",
               "submitted_ask", "fills_bid", "fills_ask", "execution_prices_bid_hex",
               "execution_prices_ask_hex", "executed_quantities_bid", "executed_quantities_ask",
               "fees_hex")
    def hx(value):
        return float(value).hex()
    with destination.open("w") as out:
        out.write("\t".join(columns) + "\n")
        for episode in range(100):
            for period in range(301):
                row = [str(episode), str(period), hx(original["price"][episode, period]),
                       str(int(original["signal"][episode, period])),
                       hx(original["cash"][episode, period]),
                       str(int(original["inventory"][episode, period]))]
                if period < 300:
                    row.extend([str(int(original["actions"][episode, period])),
                        hx(original["action_probability"][episode, period]),
                        hx(original["return_"][episode, period])])
                    row.extend(str(int(v)) for v in original["submitted"][episode, period])
                    row.extend(str(int(v)) for v in original["fills"][episode, period])
                    row.extend(hx(v) for v in original["execution_prices"][episode, period])
                    row.extend(str(int(v)) for v in original["executed_quantities"][episode, period])
                    row.append(hx(original["fees"][episode, period]))
                else:
                    row.extend([""] * 12)
                out.write("\t".join(row) + "\n")
    # The line count and digest bind this reviewable text to the JSON receipt.
    with destination.open() as check:
        require(sum(1 for _ in check) == 30101, "hex export row count mismatch")
    return {"path": str(destination.relative_to(ROOT)),
            "sha256": sha_bytes(destination.read_bytes()), "data_rows": 30100}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="regenerate all 90; default is detailed two only")
    parser.add_argument("--label", required=True, help="short runtime label for output files")
    parser.add_argument("--export-original-hex", action="store_true",
                        help="write public arrays of the two focus pilots as plain hex-float TSV")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/verification/reaudit")
    args = parser.parse_args()
    require(args.label.isidentifier(), "label must be an identifier")

    manifest_path = ROOT / "outputs/full/manifest.json"
    fits_path = ROOT / "outputs/full/pilot_fits.json"
    manifest = json.loads(manifest_path.read_text())
    fits = json.loads(fits_path.read_text())
    source_files = sorted((ROOT / "src/trade_learning").glob("*.py"))
    source_hash = sha_bytes(b"".join(p.name.encode() + p.read_bytes() for p in source_files))
    require(source_hash == manifest["code_hash"], "current production source differs from campaign")
    config_path = ROOT / "configs/protocol.json"
    require(sha_bytes(config_path.read_bytes()) == manifest["configuration_sha256"],
            "configuration bytes differ from campaign")
    configured_protocol = resolve_protocol(load_configuration(config_path), "full")
    protocol = manifest["protocol"]
    require(protocol == configured_protocol, "campaign protocol differs from validated configuration")
    require(protocol_fingerprint(protocol) == manifest["protocol_sha256"],
            "campaign protocol fingerprint mismatch")
    require((protocol["horizon"], protocol["qmax"], protocol["pilots"], protocol["pilot_episodes"]) ==
            (300, 5, 10, 100), "unexpected pilot protocol")
    environments = [(float(t), float(k)) for t, k in protocol["environments"]]
    require(len(environments) == 9 and len(manifest["streams"]) == len(fits) == 90,
            "missing environment, stream, or fit")
    ids = [(e, p) for e in range(9) for p in range(10)] if args.all else sorted(DETAILED)
    records = []
    for e, p in ids:
        index = 10 * e + p
        stream = manifest["streams"][index]
        stored_fit = fits[index]
        theta, kappa = environments[e]
        name = f"{e:02d}_{p:02d}"
        require(stream["pilot"] == stored_fit["pilot"] == p, f"{name}: pilot index mismatch")
        require(stream["pilot_seed"] == seed_for(protocol["master_seed"], 11, e, p),
                f"{name}: pilot seed mismatch")
        require(stream["behavior_seed"] == seed_for(protocol["master_seed"], 12, e, p),
                f"{name}: behavior seed mismatch")
        original_path = ROOT / "outputs/full/pilots" / f"{name}.npz"
        original_zip_hash = sha_bytes(original_path.read_bytes())
        with np.load(original_path, allow_pickle=False) as archive:
            require(set(archive.files) == set(PUBLIC_FIELDS), f"{name}: public field set mismatch")
            original = {key: archive[key] for key in PUBLIC_FIELDS}
        hex_export = None
        if args.export_original_hex and (e, p) in DETAILED:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            hex_export = export_original_hex(original, args.out_dir / f"pilot_original_{name}_hex.tsv")
        generated = generate_uniform_pilot(
            episodes=100, horizon=300, theta=theta, kappa=kappa,
            seed=stream["pilot_seed"], policy_seed=stream["behavior_seed"], qmax=5,
        ).as_dict()
        require(set(generated) == set(PUBLIC_FIELDS), f"{name}: regenerated field set mismatch")
        original_digest = pilot_hash(original)
        regenerated_digest = pilot_hash(generated)
        require(original_digest == stream["pilot_data_hash"] == stored_fit["pilot_data_hash"],
                f"{name}: original archive does not match campaign")
        fields = {key: compare_field(
            key, original[key], generated[key],
            original["executed_quantities"] if key == "execution_prices" else None,
            generated["executed_quantities"] if key == "execution_prices" else None,
        ) for key in PUBLIC_FIELDS}
        return_diagnostics = []
        if (e, p) in DETAILED and fields["return_"]["differing"]:
            tapes = generate_exogenous(100, 300, theta, kappa, stream["pilot_seed"])
            for example in fields["return_"]["first_differences"]:
                episode, period = example["index"]
                return_diagnostics.append({"index": [episode, period],
                    "regenerated_z_hex": float(tapes.z[episode, period]).hex(),
                    "signal": int(tapes.x[episode, period])})
        fit = verify_fit(original, generated, stored_fit) if (e, p) in DETAILED else None
        passed = all(item["passed"] for item in fields.values()) and (fit is None or fit["passed"])
        records.append({"pilot": name, "environment": stream["environment"],
                        "original_archive_sha256": original_zip_hash,
                        "original_public_hash": original_digest,
                        "regenerated_public_hash": regenerated_digest,
                        "public_hash_equal": original_digest == regenerated_digest,
                        "fields": fields, "return_diagnostics": return_diagnostics,
                        "fit": fit, "hex_export": hex_export, "passed": passed})
        print(f"{name}: {'PASS' if passed else 'FAIL'} "
              f"public_hash={'same' if original_digest == regenerated_digest else 'different'}", flush=True)

    result = {
        "scope": "local original archives versus deterministic regeneration; no reviewer scratch arrays",
        "failure_contract": "verification/pilot_portability_failure_modes.md",
        "checker_contract_version": 2,
        "checker_sha256": sha_bytes(Path(__file__).read_bytes()),
        "source_sha256": source_hash,
        "configuration_sha256": sha_bytes(config_path.read_bytes()),
        "manifest_sha256": sha_bytes(manifest_path.read_bytes()),
        "pilot_fits_sha256": sha_bytes(fits_path.read_bytes()),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "pandas": pd.__version__,
                    "system": platform.system(), "machine": platform.machine()},
        "absolute_limits": ABS_LIMIT, "fit_log_weight_absolute_limit": FIT_LIMIT,
        "count": len(records), "passed_count": sum(r["passed"] for r in records),
        "matching_public_hashes": sum(r["public_hash_equal"] for r in records),
        "passed": all(r["passed"] for r in records), "pilots": records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_dir / f"pilot_portability_{args.label}"
    prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with prefix.with_suffix(".tsv").open("w") as out:
        out.write("pilot\tfield\tdtype\tshape\toriginal_sha256\tregenerated_sha256\tdiffering\tmax_abs\tmax_ulp\tpassed\n")
        for record in records:
            for key, field in record["fields"].items():
                out.write("\t".join(map(str, [record["pilot"], key, field["dtype"],
                    "x".join(map(str, field["shape"])), field["original_sha256"],
                    field["regenerated_sha256"], field["differing"], field.get("max_abs", ""),
                    field.get("max_ulp", ""), field["passed"]])) + "\n")
    print(json.dumps({key: result[key] for key in
          ("count", "passed_count", "matching_public_hashes", "passed")}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
