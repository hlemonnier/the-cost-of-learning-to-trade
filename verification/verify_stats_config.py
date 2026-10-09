"""Pre-implementation acceptance experiment for AUD-03 and AUD-04.

Consumes original complete artifacts read-only, injects meaningful boundary
failures, and optionally runs a tiny fresh-process cold-cache campaign.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from trade_learning import run
from trade_learning.statistics import clustered_estimate, stratified_estimate, summarize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/verification/audit_corrections"


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(run.clean_json(value), indent=2, allow_nan=False) + "\n")


def raw_configuration():
    return json.loads((ROOT / "configs/protocol.json").read_text())


def resolved_with_raw(raw, profile="full"):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        (path / "configs").mkdir()
        (path / "configs/protocol.json").write_text(raw if isinstance(raw, str) else json.dumps(raw))
        with patch.object(run, "ROOT", path):
            return run.make_protocol(profile)


def numeric_difference(actual, expected, path="root"):
    """Compare all pre-existing fields, keeping added audit metadata separate."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(expected) <= set(actual), (path, set(expected) - set(actual))
        return max((numeric_difference(actual[k], v, f"{path}.{k}") for k, v in expected.items()), default=0.)
    if isinstance(expected, list):
        assert len(actual) == len(expected), path
        return max((numeric_difference(a, e, f"{path}[{i}]") for i, (a, e) in enumerate(zip(actual, expected))), default=0.)
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert isinstance(actual, (int, float)) and math.isfinite(actual), (path, actual)
        error = abs(actual - expected)
        assert error <= 1e-12, (path, actual, expected, error)
        return error
    assert actual == expected, (path, actual, expected)
    return 0.


def capture_before(original):
    one = pd.DataFrame({"pilot": [0, 0, 0], "episode": [0, 1, 2], "v": [0., 10., 20.]})
    record = {"scope": "Preserved pre-correction probes; these are defects, not acceptance results"}
    for name, call in [
        ("single_pilot", lambda: clustered_estimate(one, "v")),
        ("implicit_population", lambda: stratified_estimate(pd.DataFrame({
            "environment": ["one"]*4, "pilot": [0, 0, 1, 1], "episode": [0, 1, 0, 1], "v": [1., 2., 3., 4.]}), "v")),
    ]:
        try:
            record[name] = {"accepted": True, "result": call()}
        except (TypeError, ValueError) as exc:
            record[name] = {"accepted": False, "error": str(exc)}
    for field, value in [("theta_grid", [.27]), ("kappa_grid", [.04]), ("policies", ["abstain"]), ("primary_baseline", "taker")]:
        raw = raw_configuration(); raw[field] = value
        try:
            record[field] = {"accepted": True, "resolved": resolved_with_raw(raw)}
        except (TypeError, ValueError) as exc:
            record[field] = {"accepted": False, "error": str(exc)}
    manifest = json.loads((original / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["code_hash"] == "96ccb4b6e1086d78416c62cd40851b24b2d07990bd971d3581cc0864263c91f7", "Use the preserved original campaign"
    assert hashlib.sha256((original / "episodes.csv.gz").read_bytes()).hexdigest() == manifest["csv_sha256"]
    data = pd.read_csv(original / "episodes.csv.gz")
    saved = json.loads((original / "summary.json").read_text())
    computed = run.clean_json(summarize(data, manifest["protocol"]))
    record["original_summary_max_numeric_difference"] = numeric_difference(computed, saved)
    record["original_rows"] = len(data)
    record["original_csv_sha256"] = hashlib.sha256((original / "episodes.csv.gz").read_bytes()).hexdigest()
    save("stats_config_before.json", record)
    print(json.dumps({"artifact": str(OUT / 'stats_config_before.json'), "original_rows": len(data)}))


def verify(original, smoke_run):
    from trade_learning.protocol import validate_protocol, validate_inference_support, protocol_fingerprint
    rejected = []

    def reject(name, call):
        try:
            call()
        except (ValueError, TypeError) as exc:
            rejected.append({"case": name, "exception": type(exc).__name__, "message": str(exc)})
        else:
            raise AssertionError(f"Malformed evidence/configuration was accepted: {name}")

    valid = pd.DataFrame({"pilot": [0, 0, 1, 1], "episode": [0, 1, 0, 1], "v": [0., 2., 3., 5.]})
    reject("single_pilot_nonconstant", lambda: clustered_estimate(pd.DataFrame({"pilot": [0]*3, "episode": [0, 1, 2], "v": [0., 10., 20.]}), "v"))
    reject("empty", lambda: clustered_estimate(valid.iloc[:0], "v"))
    reject("one_episode_per_pilot", lambda: clustered_estimate(valid.iloc[[0, 2]], "v"))
    reject("unbalanced_pilots", lambda: clustered_estimate(valid.iloc[:-1], "v"))
    reject("duplicate_pilot_episode", lambda: clustered_estimate(pd.concat([valid, valid.iloc[[0]]]), "v"))
    reject("missing_episode_identifier", lambda: clustered_estimate(valid.drop(columns="episode"), "v"))
    for label, value in [("nan", np.nan), ("positive_infinity", np.inf), ("negative_infinity", -np.inf), ("overflow", 1e308)]:
        bad = valid.copy(); bad["v"] = value
        reject(label, lambda bad=bad: clustered_estimate(bad, "v"))
    for label, value in [("text_outcome", "1.0"), ("complex_outcome", 1+1j), ("boolean_outcome", True)]:
        bad = valid.copy(); bad["v"] = value
        reject(label, lambda bad=bad: clustered_estimate(bad, "v"))
    for alpha in (0, 1, -.1, np.nan, np.inf):
        reject(f"invalid_alpha_{alpha}", lambda alpha=alpha: clustered_estimate(valid, "v", alpha))
    for identifier, value in [("pilot", .5), ("episode", -1), ("pilot", np.nan)]:
        bad = valid.astype({identifier: float}); bad.loc[0, identifier] = value
        reject(f"invalid_{identifier}_{value}", lambda bad=bad: clustered_estimate(bad, "v"))
    bad = valid.copy(); bad["environment"] = ["a", "b", "a", "b"]
    reject("mixed_cluster_strata", lambda: clustered_estimate(bad, "v"))
    reject("undeclared_stratified_population", lambda: stratified_estimate(bad, "v"))
    # Zero observed variance from two valid pilots remains legitimate; no floor.
    zero = valid.copy(); zero["v"] = 2.
    assert clustered_estimate(zero, "v")["se"] == 0

    full_protocol = run.make_protocol("full")
    smoke_protocol = run.make_protocol("smoke")
    assert len(full_protocol["environments"]) == 9
    assert smoke_protocol["environments"] == [[.35, .02]]
    assert smoke_protocol["deployment_population"] != full_protocol["deployment_population"]
    assert len(smoke_protocol["theta_grid"])*len(smoke_protocol["kappa_grid"]) == 9
    validate_inference_support(full_protocol, run.CANDIDATE_THETA, run.CANDIDATE_KAPPA, run.POLICIES)
    wrong_theta = run.CANDIDATE_THETA.copy(); wrong_theta[0] = .27
    reject("actual_inference_support_mismatch", lambda: validate_inference_support(full_protocol, wrong_theta, run.CANDIDATE_KAPPA, run.POLICIES))
    reject("actual_policy_implementation_mismatch", lambda: validate_inference_support(full_protocol, run.CANDIDATE_THETA, run.CANDIDATE_KAPPA, ("abstain",)))

    changes = {"theta_grid": [.27], "kappa_grid": [.04], "policies": ["abstain"],
               "primary_policy": "independent", "primary_baseline": "taker", "deployment_threshold": .01,
               "deployment_population": "an invented market", "variants": ["nominal"], "horizon": 301,
               "qmax": 4, "pilots": 1, "pilot_episodes": 99, "test_episodes": 99,
               "version": 999, "parameter_prior": "oracle", "online_updates": "across episodes",
               "algorithm_selection": "select after outcomes", "mispelled_field": 123,
               "master_seed": True, "belief_points": 0, "quadrature_points": 1.5}
    for field, value in changes.items():
        raw = raw_configuration(); raw[field] = value
        for profile in ("full", "smoke"):
            reject(f"config_{profile}_{field}", lambda raw=raw, profile=profile: resolved_with_raw(raw, profile))
    for field in ("policies", "theta_grid", "primary_baseline"):
        raw = raw_configuration(); del raw[field]
        reject(f"missing_config_{field}", lambda raw=raw: resolved_with_raw(raw))
    for field in ("theta_grid", "policies", "variants"):
        raw = raw_configuration(); raw[field] = list(reversed(raw[field]))
        reject(f"reordered_{field}", lambda raw=raw: resolved_with_raw(raw))
    raw = raw_configuration(); raw["theta_grid"] = [.15, .35, .35]
    reject("duplicate_grid", lambda: resolved_with_raw(raw))
    duplicate_json = json.dumps(raw_configuration())[:-1] + ', "policies": ["abstain"]}'
    reject("duplicate_json_key", lambda: resolved_with_raw(duplicate_json))
    raw_json = json.dumps(raw_configuration()).replace('"master_seed": 260924138', '"master_seed": NaN')
    if 'NaN' not in raw_json:
        raw_json = json.dumps({**raw_configuration(), "master_seed": float('nan')})
    reject("nonfinite_json_literal", lambda: resolved_with_raw(raw_json))
    reject("unknown_profile", lambda: run.make_protocol("invented"))
    for bp, qp in [(0, None), (True, None), (1.5, None), (None, 0), (None, -2)]:
        reject(f"bad_resolution_{bp}_{qp}", lambda bp=bp, qp=qp: run.make_protocol("full", bp, qp))
    numerical = run.make_protocol("full", 41, 23)
    assert numerical["belief_points"] == 41 and numerical["quadrature_points"] == 23 and numerical["belief_overrides"] == {}
    for field, value in [("environments", [[.15, .002]]), ("policies", ["abstain"]),
                         ("primary_baseline", "taker"), ("deployment_population", "wrong"),
                         ("pilots", 1), ("profile", "bad")]:
        bad = deepcopy(full_protocol); bad[field] = value
        reject(f"resolved_{field}", lambda bad=bad: validate_protocol(bad))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)/"must_not_exist"
            with patch.object(run, "TableBank", side_effect=AssertionError("table constructed before validation")), patch.object(run, "generate_exogenous", side_effect=AssertionError("simulation before validation")):
                reject(f"execute_preflight_{field}", lambda bad=bad: run.execute(bad, destination, Path(directory)/"cache"))
            assert not destination.exists()

    manifest = json.loads((original / "manifest.json").read_text())
    original_data = pd.read_csv(original / "episodes.csv.gz")
    assert manifest["status"] == "complete"
    assert manifest["code_hash"] == "96ccb4b6e1086d78416c62cd40851b24b2d07990bd971d3581cc0864263c91f7", "Use the preserved original campaign"
    assert hashlib.sha256((original / "episodes.csv.gz").read_bytes()).hexdigest() == manifest["csv_sha256"]
    original_summary = json.loads((original / "summary.json").read_text())
    new_summary = run.clean_json(summarize(original_data, manifest["protocol"]))
    maximum = numeric_difference(new_summary, original_summary)
    reject("missing_declared_environment", lambda: summarize(original_data[original_data.environment != 'theta0.15_kappa0.002'], manifest["protocol"]))
    pivot = original_data[original_data.variant == "nominal"].pivot(index=["environment", "pilot", "episode"], columns="policy", values="objective")
    delta = (pivot.active-pivot.noinfo).rename("difference").reset_index()
    primary = stratified_estimate(delta, "difference", protocol=manifest["protocol"])
    assert abs(primary["mean"]-original_summary["primary_comparison"]["mean"]) <= 1e-12
    reject("missing_stratified_environment", lambda: stratified_estimate(delta[delta.environment != 'theta0.15_kappa0.002'], "difference", protocol=manifest["protocol"]))
    smoke_path = ROOT/"outputs/verification/pipeline/first/episodes.csv.gz"
    data = pd.read_csv(smoke_path)
    smoke_summary = summarize(data, smoke_protocol)
    assert smoke_summary["primary_comparison"]["decision"] == "smoke_only_no_deployment_decision"
    for label, bad in [
        ("missing_comparator", data[data.policy != "noinfo"]),
        ("missing_variant", data[data.variant != "fixed_duration"]),
        ("missing_pilot", data[data.pilot != 1]),
        ("missing_episode", data.iloc[:-1]),
        ("duplicate_paired_key", pd.concat([data, data.iloc[[0]]], ignore_index=True)),
        ("duplicate_replacing_missing", pd.concat([data.iloc[:-1], data.iloc[[0]]], ignore_index=True)),
    ]:
        reject(label, lambda bad=bad: summarize(bad, smoke_protocol))
    for col, value in [("objective", np.nan), ("pnl", np.inf), ("inventory_penalty", -np.inf),
                       ("theta", .15), ("kappa", .10), ("environment", "wrong"), ("policy", "unknown"),
                       ("pilot", 2), ("episode", 16), ("pilot_data_hash", None), ("starting_posterior_hash", None)]:
        bad = data.copy(); bad.loc[0, col] = value
        reject(f"campaign_{col}_{value}", lambda bad=bad: summarize(bad, smoke_protocol))
    # Every always-defined ledger/execution metric must fail closed. Ratios and
    # learner-only monitoring retain their intentional unavailable values.
    economic_fields = [
        "market_spread_cost", "liquidation_spread", "liquidation_fees",
        "directional_pnl", "execution_exposure_pnl", "inventory_sq_sum",
        "inventory_post_sq_sum", "inventory_abs_sum", "max_abs_inventory",
        "passive_fills", "market_fills", "terminal_inventory", "terminal_units",
        "risk_penalty", "reconciled_pnl", "terminal_inventory_after_liquidation",
        "mean_abs_inventory", "bid_fill_count", "ask_fill_count", "max_quote_gap",
        "submitted_periods", "submitted_sides_x-1", "submitted_sides_x0", "submitted_sides_x1",
    ]
    for field in economic_fields:
        bad = data.astype({field: float}).copy(); bad.loc[0, field] = np.nan
        reject(f"campaign_nonfinite_{field}", lambda bad=bad: summarize(bad, smoke_protocol))
        reject(f"campaign_missing_{field}", lambda field=field: summarize(data.drop(columns=field), smoke_protocol))
    for field in ("objective", "episode", "policy", "pilot_data_hash"):
        reject(f"campaign_missing_column_{field}", lambda field=field: summarize(data.drop(columns=field), smoke_protocol))
    bad = data.copy(); bad.loc[0, "pilot_data_hash"] = "0"*64
    reject("unmatched_pilot_identity", lambda: summarize(bad, smoke_protocol))
    bad = data.copy(); bad["pilot_data_hash"] = "0"*64
    reject("reused_pilot_disguised_as_independent", lambda: summarize(bad, smoke_protocol))
    bad = data.copy(); first = bad.index[bad.policy == "active"][0]; bad.loc[first, "starting_posterior_hash"] = "0"*64
    reject("unmatched_initial_learning_state", lambda: summarize(bad, smoke_protocol))

    extra = {}
    if smoke_run:
        destination = OUT/"stats_config_smoke"
        with tempfile.TemporaryDirectory(prefix="trade-learning-stats-config-cache-") as cache:
            subprocess.run([sys.executable, "-m", "trade_learning.run", "--profile", "smoke", "--out", str(destination), "--cache", cache], cwd=ROOT, check=True)
        m = json.loads((destination/"manifest.json").read_text())
        actual = pd.read_csv(destination/"episodes.csv.gz")
        assert m["status"] == "complete" and len(actual) == 672
        assert m["protocol_sha256"] == protocol_fingerprint(m["protocol"])
        assert m["control_artifacts"] and m["numerical_contract"]
        assert m["protocol"]["deployment_population"] == smoke_protocol["deployment_population"]
        assert set(actual.environment) == {"theta0.35_kappa0.02"}
        extra = {"smoke_manifest": str(destination/"manifest.json"), "control_fingerprints": len(m["control_artifacts"])}
    artifact = {"passed": True, "scope": "AUD-03/AUD-04 failure probes and original raw-summary reconciliation; no full simulation",
                "rejected_cases": rejected, "rejected_count": len(rejected),
                "canonical_rows": len(original_data), "canonical_max_numeric_difference": maximum,
                "original_csv_sha256": hashlib.sha256((original/"episodes.csv.gz").read_bytes()).hexdigest(),
                "original_source_sha256": manifest["code_hash"],
                "resolved_smoke": smoke_protocol, "resolved_full": full_protocol,
                "source_hash": run.source_hash(), **extra}
    save("stats_config_verification.json", artifact)
    print(json.dumps({k: artifact[k] for k in ["passed", "rejected_count", "canonical_rows", "canonical_max_numeric_difference"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True,
                        help="outputs/full from preserved tag audit/original-c9d1d4c")
    parser.add_argument("--capture-before", action="store_true")
    parser.add_argument("--smoke-run", action="store_true")
    args = parser.parse_args()
    capture_before(args.original) if args.capture_before else verify(args.original, args.smoke_run)
