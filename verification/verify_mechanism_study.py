"""Independent end-to-end reader for the frozen matched-mechanism study.

This verifier imports neither the study runner nor trade_learning production modules.
It regenerates exogenous tapes from the public seed, replays every recorded action
through the assignment's cash-flow equations, checks the public decision traces,
and recalculates paired episode-level inference.  Failure cases were declared in
verification/mechanism_failure_modes.md before this executable was implemented.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
from scipy.special import expit, log_ndtr, logsumexp, ndtri
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path("configs/mechanism_study.json")
CONFIG_SHA256 = "d2c6f2edd919e7ff0f08b7a09ea05ec109234f6f4ef17c2eaef392aae04d942a"
PROTOCOL_SHA256 = "773cdeca929f732d6c3662a46a9e2cb1f5c25698f8ed4d2dfcd4b357d2755364"
FAILURE_CATALOG_SHA256 = "a13c9156df0d831e5e7b69492e530ca72605408a39bd6e79b89a8e7029c4a529"
POLICIES = (
    "active", "noinfo", "no_forecast", "blind_regime", "inventory_off",
    "myopic", "full_information",
)
CONTRASTS = (
    ("forecast", "noinfo", "no_forecast"),
    ("current_inference", "noinfo", "blind_regime"),
    ("inventory_continuation", "noinfo", "inventory_off"),
    ("future_information", "active", "noinfo"),
)
ARTIFACTS = (
    "episodes.csv.gz", "actions.npz", "traces.csv.gz", "numerical_checks.json",
    "summary.json",
)
TAPE_KEYS = ("x", "h", "z", "u_bid", "u_ask")
TABLE_KEYS = ("active", "noinfo", "no_forecast", "inventory_off", "full_information")
ARRAY_KEYS = ("beliefs", "values", "q_values")
TRACE_COLUMNS = (
    "policy", "episode", "t", "inventory", "signal", "price", "cash", "belief",
    "decision_belief", "action", "return_", "bid_fill", "ask_fill", "next_inventory",
    "next_cash",
)
REQUIRED_ROW_COLUMNS = (
    "policy", "episode", "objective", "pnl", "inventory_penalty", "inventory_sq_sum",
    "max_abs_inventory", "terminal_inventory_before_liquidation",
    "terminal_liquidation_cost", "reconciliation_error",
)
K = np.array([[.75, .20, .05], [.10, .80, .10], [.05, .20, .75]], dtype=float)
BID_DEPTH = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, -2, -2], dtype=np.int8)
ASK_DEPTH = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1, -2, -2], dtype=np.int8)
SHA_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
NUMERICAL_CONTRACT = {
    "version": 2,
    "score_dtype": "float64",
    "tie_absolute_tolerance": 1e-12,
    "tie_relative_tolerance": 0.0,
    "tie_action": "lowest admissible action ID within absolute tolerance of maximum",
    "bellman_value": "Q of the action selected by the same tie rule",
    "invalid_action": "negative infinity; no decision at remaining horizon zero",
}
GIT_CHECKOUT_SCOPE = "current checkout commit plus exact source hashes"
GIT_EXPORT_SCOPE = "source export without Git metadata; exact source hashes only"


class VerificationError(ValueError):
    """A study artifact contradicts the predeclared protocol or raw evidence."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(record: object) -> str:
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def json_read(path: Path) -> dict:
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise VerificationError(f"Invalid JSON {path}: {exc}") from exc
    require(isinstance(result, dict), f"Expected JSON object: {path}")
    return result


def number(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"Invalid {label}: {value!r}") from exc
    require(math.isfinite(result), f"Nonfinite {label}")
    return result


def integer(value: object, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"Invalid {label}: {value!r}") from exc
    require(str(result) == str(value), f"Non-integer {label}: {value!r}")
    return result


def close(actual: float, expected: float, label: str, tolerance: float = 2e-8) -> None:
    require(abs(actual - expected) <= tolerance, f"{label}: {actual!r} versus {expected!r}")


def frozen_config(root: Path) -> dict:
    path = root / CONFIG_PATH
    require(sha(path) == CONFIG_SHA256, "Frozen mechanism-study configuration bytes changed")
    require(sha(root / "report/mechanism_study_protocol.md") == PROTOCOL_SHA256,
            "Predeclared study protocol bytes changed")
    require(sha(root / "verification/mechanism_failure_modes.md") == FAILURE_CATALOG_SHA256,
            "Pre-implementation failure catalogue bytes changed")
    config = json_read(path)
    require(config.get("version") == 1, "Wrong mechanism-study version")
    for key, expected in (("theta", .35), ("kappa", .02), ("horizon", 30),
                          ("qmax", 2), ("episodes", 20000), ("seed", 260924506),
                          ("belief_points", 321), ("quadrature_points", 161),
                          ("alpha", .05), ("family_size", 4), ("trace_episodes", 12)):
        require(config.get(key) == expected, f"Frozen setting changed: {key}")
    require(config.get("policies") == list(POLICIES), "Wrong frozen policy order")
    require(config.get("contrasts") == [dict(mechanism=m, policy=p, baseline=b)
                                       for m, p, b in CONTRASTS], "Wrong frozen contrasts")
    require(config.get("refinement") == dict(initial=.001, sup=.005, robust_gap=.001,
                                             robust_disagreement=.005), "Wrong refinement contract")
    return config


def check_manifest(root: Path, study: Path, config: dict) -> tuple[dict, dict, dict]:
    manifest = json_read(study / "manifest.json")
    require(manifest.get("status") == "complete", "Study manifest is incomplete")
    require(manifest.get("protocol") == config, "Manifest protocol differs from frozen configuration")
    require(manifest.get("numerical_contract") == NUMERICAL_CONTRACT,
            "Manifest numerical decision contract differs")
    require(manifest.get("cold_control_build") is True
            and manifest.get("shared_control_cache") is False,
            "Study was not declared as an isolated cold control build")
    resolution = manifest.get("selected_resolution")
    require(isinstance(resolution, list) and len(resolution) == 2
            and all(type(value) is int for value in resolution),
            "Malformed selected numerical resolution")
    beliefs, quadrature = resolution
    require(beliefs in (321, 641, 1281, 2561, 5121)
            and quadrature in (161, 321, 641, 1281),
            "Selected resolution violates the frozen refinement sequence")

    source_files = sorted((root / "src/trade_learning").glob("*.py"))
    source_aggregate = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes()
                                                 for p in source_files)).hexdigest()
    canonical_manifest = json_read(root / "outputs/full/manifest.json")
    require(canonical_manifest.get("status") == "complete", "Canonical campaign is incomplete")
    require(source_aggregate == canonical_manifest.get("code_hash")
            == manifest.get("canonical_source_sha256"),
            "Supplemental study does not match frozen canonical production source")
    canonical_raw = sha(root / "outputs/full/episodes.csv.gz")
    require(canonical_raw == canonical_manifest.get("csv_sha256")
            == manifest.get("canonical_raw_sha256"),
            "Supplemental study does not match canonical raw campaign")
    freeze_commit = manifest.get("protocol_freeze_commit")
    execution_commit = manifest.get("execution_commit")
    git_scope = manifest.get("execution_git_scope")
    require(isinstance(freeze_commit, str) and re.fullmatch(r"[0-9a-f]{4,40}", freeze_commit),
            "Malformed frozen-protocol commit label")
    require(git_scope in (GIT_CHECKOUT_SCOPE, GIT_EXPORT_SCOPE),
            "Missing or unsupported execution Git scope")
    if execution_commit is None:
        require(git_scope == GIT_EXPORT_SCOPE,
                "Null execution commit requires explicit source-export scope")
    else:
        require(isinstance(execution_commit, str)
                and re.fullmatch(r"[0-9a-f]{40}", execution_commit)
                and git_scope == GIT_CHECKOUT_SCOPE,
                "Execution commit/scope do not describe a Git checkout")
    # Check the repository's own .git entry before invoking Git. A Git command
    # from an unpacked bundle could otherwise silently resolve an ancestor repo.
    if (root / ".git").exists():
        require(git_scope == GIT_CHECKOUT_SCOPE and execution_commit is not None,
                "Git checkout is present but study claims a source export")
        for revision in (freeze_commit, execution_commit):
            resolved = subprocess.run(["git", "cat-file", "-t", revision], cwd=root,
                                      text=True, capture_output=True, check=False)
            require(resolved.returncode == 0 and resolved.stdout.strip() == "commit",
                    f"Unresolvable study Git revision: {revision}")
        frozen_config = subprocess.run(["git", "show", f"{freeze_commit}:{CONFIG_PATH}"],
                                       cwd=root, capture_output=True, check=False)
        require(frozen_config.returncode == 0
                and hashlib.sha256(frozen_config.stdout).hexdigest() == CONFIG_SHA256,
                "Freeze commit does not contain the declared configuration")
        git_validation = {"history_available": True, "commits_verified": True,
                          "scope": "Own repository Git objects validated; exact current source bytes also checked"}
    else:
        git_validation = {"history_available": False, "commits_verified": False,
                          "scope": "Git history unavailable in exported archive; exact frozen protocol, production source and canonical raw bytes verified"}
    inputs = manifest.get("input_sha256")
    require(isinstance(inputs, dict), "Missing input_sha256 map")
    required = {str(CONFIG_PATH), "report/mechanism_study_protocol.md",
                "verification/mechanism_failure_modes.md", "verification/mechanism_study.py",
                "verification/verify_control.py"}
    required.update(str(p.relative_to(root)) for p in (root / "src/trade_learning").glob("*.py"))
    require(required <= set(inputs), f"Missing source inputs: {sorted(required - set(inputs))}")
    for relative, expected in inputs.items():
        require(isinstance(relative, str) and isinstance(expected, str) and SHA_PATTERN.fullmatch(expected),
                "Malformed input hash entry")
        candidate = (root / relative).resolve()
        require(candidate.is_relative_to(root.resolve()) and candidate.is_file(),
                f"Unsafe or missing source input: {relative}")
        require(sha(candidate) == expected, f"Stale source input: {relative}")

    files = manifest.get("files")
    require(isinstance(files, dict) and set(files) == set(ARTIFACTS),
            "Manifest files must list exactly the five study artifacts")
    for name, expected in files.items():
        require(isinstance(expected, str) and SHA_PATTERN.fullmatch(expected),
                f"Invalid file digest: {name}")
        require((study / name).is_file() and sha(study / name) == expected,
                f"Missing or stale artifact: {name}")

    build = manifest.get("build")
    require(isinstance(build, dict), "Missing numerical build fingerprint")
    build_sha = build.get("sha256")
    require(isinstance(build_sha, str) and SHA_PATTERN.fullmatch(build_sha), "Malformed build digest")
    require(canonical_hash({k: v for k, v in build.items() if k != "sha256"}) == build_sha,
            "Numerical build fingerprint hash does not verify")
    require(isinstance(build.get("numerical_binaries"), dict)
            and len(build["numerical_binaries"]) >= 4, "Incomplete numerical binary fingerprint")
    for digest in build["numerical_binaries"].values():
        require(isinstance(digest, str) and SHA_PATTERN.fullmatch(digest),
                "Malformed numerical binary digest")

    tables = manifest.get("table_fingerprints")
    require(isinstance(tables, dict) and set(tables) == set(TABLE_KEYS),
            "Wrong control table fingerprint set")
    for mode, arrays in tables.items():
        require(isinstance(arrays, dict) and set(arrays) == set(ARRAY_KEYS),
                f"Incomplete table fingerprint: {mode}")
        for name, item in arrays.items():
            require(isinstance(item, dict) and {"sha256", "shape", "dtype"} <= set(item),
                    f"Malformed table array fingerprint: {mode}/{name}")
            require(isinstance(item["sha256"], str) and SHA_PATTERN.fullmatch(item["sha256"]),
                    f"Malformed table array digest: {mode}/{name}")
            require(isinstance(item["shape"], list) and len(item["shape"]) >= 1
                    and all(type(v) is int and v > 0 for v in item["shape"]),
                    f"Malformed table array shape: {mode}/{name}")
            require(item["dtype"] in ("float64", "<f8"),
                    f"Unexpected table array dtype: {mode}/{name}")
        model_beliefs = 2 if mode == "full_information" else beliefs
        shapes = {"beliefs": [model_beliefs],
                  "values": [config["horizon"] + 1, 2 * config["qmax"] + 1, 3, model_beliefs],
                  "q_values": [config["horizon"] + 1, 2 * config["qmax"] + 1, 3,
                               model_beliefs, 11]}
        for name, expected_shape in shapes.items():
            require(arrays[name]["shape"] == expected_shape,
                    f"Table shape differs from selected resolution: {mode}/{name}")
    tape_hashes = manifest.get("tape_sha256")
    require(isinstance(tape_hashes, dict) and set(tape_hashes) == set(TAPE_KEYS),
            "Wrong exogenous tape hash set")
    require(all(isinstance(x, str) and SHA_PATTERN.fullmatch(x) for x in tape_hashes.values()),
            "Malformed exogenous tape digest")
    numerical = json_read(study / "numerical_checks.json")
    require(numerical.get("passed") is True, "Producer numerical checks did not pass")
    check_numerical(numerical, config, resolution)
    return manifest, numerical, git_validation


def check_numerical(numerical: dict, config: dict, resolution: list[int]) -> None:
    """Recalculate recorded acceptance logic; do not accept producer pass flags alone."""
    structural = numerical.get("structural")
    require(isinstance(structural, dict) and structural.get("passed") is True,
            "Missing passed structural calculation")
    structural_limits = {
        "max_inventory_deletion_error": 2e-11,
        "max_one_step_drift_error": 2e-12,
        "synthetic_continuation_error": 2e-11,
        "inventory_independent_continuation_error": 2e-11,
    }
    for key, limit in structural_limits.items():
        value = number(structural.get(key), f"structural {key}")
        require(0 <= value < limit, f"Structural {key} exceeds its declared limit")
    require(integer(structural.get("independent_inventory_cases"), "inventory cases") > 0,
            "No independent inventory enumeration recorded")
    require(structural.get("exact_final_step_inventory_agreement") is True
            and structural.get("inventory_off_before_final_step_equals_immediate_reward_rule") is True,
            "Inventory continuation invariants failed")

    records = numerical.get("refinement")
    modes = tuple(mode for mode in TABLE_KEYS if mode != "full_information")
    require(isinstance(records, list) and len(records) >= 2 and len(records) % 2 == 0,
            "Refinement records must contain both axes at each stage")
    stage = [config["belief_points"], config["quadrature_points"]]
    horizon_list = list(range(1, config["horizon"] + 1))
    tolerance = config["refinement"]
    for index in range(0, len(records), 2):
        axis_pass = {}
        for offset, axis in enumerate(("belief", "quadrature")):
            record = records[index + offset]
            require(isinstance(record, dict) and record.get("axis") == axis,
                    f"Refinement axis missing/out of order at stage {index // 2}")
            coarse = ([(stage[0] + 1) // 2, stage[1]] if axis == "belief"
                      else [stage[0], (stage[1] + 1) // 2])
            require(record.get("coarse") == coarse and record.get("fine") == stage,
                    f"Refinement setting sequence differs at {axis}/{stage}")
            comparisons = record.get("comparisons")
            require(isinstance(comparisons, dict) and set(comparisons) == set(modes),
                    f"Refinement omits a required mode at {axis}/{stage}")
            mode_pass = []
            for mode in modes:
                comparison = comparisons[mode]
                require(isinstance(comparison, dict)
                        and comparison.get("coarse") == coarse
                        and comparison.get("fine") == stage
                        and comparison.get("numerical_contract") == NUMERICAL_CONTRACT
                        and comparison.get("remaining_horizons_checked") == horizon_list,
                        f"Refinement mode metadata mismatch: {axis}/{mode}")
                per_horizon = comparison.get("per_horizon")
                require(isinstance(per_horizon, list) and len(per_horizon) == config["horizon"],
                        f"Refinement horizon coverage incomplete: {axis}/{mode}")
                validated = []
                for period, row in enumerate(per_horizon, 1):
                    require(isinstance(row, dict) and row.get("remaining") == period,
                            f"Refinement skipped horizon {period}: {axis}/{mode}")
                    initial = number(row.get("initial_value_change"), "refinement initial drift")
                    sup = number(row.get("maximum_common_state_value_change"), "refinement state drift")
                    robust = number(row.get("robust_action_disagreement_fraction"), "refinement disagreement")
                    loss = number(row.get("maximum_action_loss_under_fine_scores"), "refinement loss")
                    require(min(initial, sup, robust, loss) >= 0 and robust <= 1,
                            f"Invalid refinement value: {axis}/{mode}/{period}")
                    states = integer(row.get("states"), "refinement states")
                    robust_states = integer(row.get("robust_states"), "refinement robust states")
                    disagreements = integer(row.get("action_disagreements"), "refinement action disagreements")
                    robust_disagreements = integer(row.get("robust_action_disagreements"),
                                                   "refinement robust disagreements")
                    require(states > 0 and 0 <= robust_states <= states
                            and 0 <= disagreements <= states
                            and 0 <= robust_disagreements <= min(disagreements, robust_states),
                            f"Impossible refinement state counts: {axis}/{mode}/{period}")
                    close(robust, robust_disagreements / max(1, robust_states),
                          f"Refinement fraction: {axis}/{mode}/{period}", 1e-13)
                    passing = (initial <= tolerance["initial"]
                               and sup <= tolerance["sup"]
                               and robust <= tolerance["robust_disagreement"])
                    require(row.get("passes_predeclared_rule") is passing,
                            f"Incorrect horizon acceptance flag: {axis}/{mode}/{period}")
                    validated.append((initial, sup, robust, loss, states, robust_states,
                                      disagreements, robust_disagreements, passing))
                maxima = {
                    "maximum_initial_state_value_change_over_horizons": max(x[0] for x in validated),
                    "maximum_common_state_value_change": max(x[1] for x in validated),
                    "maximum_per_horizon_robust_action_disagreement_fraction": max(x[2] for x in validated),
                    "maximum_action_loss_under_fine_scores": max(x[3] for x in validated),
                    "initial_value_change": validated[-1][0],
                    "raw_action_disagreement_fraction": sum(x[6] for x in validated) / sum(x[4] for x in validated),
                    "robust_action_disagreement_fraction": sum(x[7] for x in validated) / max(1, sum(x[5] for x in validated)),
                }
                for key, expected in maxima.items():
                    close(number(comparison.get(key), f"aggregate {key}"), expected,
                          f"Wrong refinement aggregate {axis}/{mode}/{key}", 1e-12)
                accepted = all(x[8] for x in validated)
                require(comparison.get("passes_predeclared_rule") is accepted,
                        f"Incorrect mode acceptance flag: {axis}/{mode}")
                mode_pass.append(accepted)
            axis_pass[axis] = all(mode_pass)
            require(record.get("passed") is axis_pass[axis],
                    f"Incorrect axis acceptance flag: {axis}/{stage}")
        final_stage = index + 2 == len(records)
        if final_stage:
            require(all(axis_pass.values()) and stage == resolution,
                    "Last numerical refinement stage is not accepted at selected resolution")
        else:
            require(not all(axis_pass.values()), "Unnecessary refinement stage after both axes passed")
            stage = [stage[0] if axis_pass["belief"] else 2 * stage[0] - 1,
                     stage[1] if axis_pass["quadrature"] else 2 * stage[1] - 1]

    same_state = numerical.get("same_state_comparisons")
    require(isinstance(same_state, list) and len(same_state) == 4
            and {x.get("mechanism") for x in same_state} ==
            {"forecast", "inventory_continuation", "future_information", "current_inference"},
            "Missing one of four controlled decision comparisons")
    for item in same_state:
        require(integer(item.get("states"), "same-state count") > 0,
                "Empty same-state comparison")
        require(number(item.get("maximum_enabled_score_loss"), "same-state score loss") >= 0,
                "Negative same-state decision loss")
    witness = numerical.get("fixed_active_continuation_feedback_witness")
    require(isinstance(witness, dict) and integer(witness.get("states"), "feedback states") > 0,
            "Missing fixed-active-continuation witness")
    require(number(witness.get("minimum_feedback_bonus"), "minimum feedback bonus") >= -1e-10,
            "Feedback witness violates approximate information dominance")


def regenerate_tapes(config: dict) -> dict[str, np.ndarray]:
    """Independent transcription of the public RNG and transition specification."""
    n, horizon, seed = (config[k] for k in ("episodes", "horizon", "seed"))
    spawned = np.random.SeedSequence(seed).spawn(6)
    rx, rh, rz, rb, ra, _phase = (np.random.default_rng(s) for s in spawned)
    x = np.zeros((n, horizon + 1), dtype=np.int8)
    uniforms = rx.random((n, horizon))
    cumulative = np.cumsum(K, axis=1)
    for period in range(horizon):
        x[:, period + 1] = np.sum(uniforms[:, period, None] > cumulative[x[:, period] + 1], axis=1) - 1
    h0 = 2 * rh.integers(0, 2, size=n, dtype=np.int8) - 1
    h = np.empty((n, horizon), dtype=np.int8)
    h[:, 0] = h0
    flips = np.where(rh.random((n, horizon - 1)) < config["kappa"], -1, 1)
    h[:, 1:] = h0[:, None] * np.cumprod(flips, axis=1)
    return {"x": x, "h": h, "z": rz.standard_normal((n, horizon)),
            "u_bid": rb.standard_normal((n, horizon)),
            "u_ask": ra.standard_normal((n, horizon))}


def check_tapes(tapes: dict, manifest: dict, config: dict) -> None:
    n, horizon = config["episodes"], config["horizon"]
    for name in TAPE_KEYS:
        array = tapes[name]
        expected_shape = (n, horizon + 1) if name == "x" else (n, horizon)
        require(array.shape == expected_shape, f"Wrong regenerated tape shape: {name}")
        actual = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        require(actual == manifest["tape_sha256"][name], f"Exogenous tape differs: {name}")
    require(np.all(tapes["x"][:, 0] == 0) and np.isin(tapes["h"], (-1, 1)).all(),
            "Regenerated tape violates initial conditions")


def read_episodes(study: Path, config: dict) -> tuple[dict[str, dict[str, np.ndarray]], int]:
    n = config["episodes"]
    numeric = tuple(x for x in REQUIRED_ROW_COLUMNS if x not in ("policy", "episode"))
    records = {p: {key: np.full(n, np.nan) for key in numeric} for p in POLICIES}
    seen = {p: np.zeros(n, dtype=bool) for p in POLICIES}
    rows = 0
    with gzip.open(study / "episodes.csv.gz", "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and set(REQUIRED_ROW_COLUMNS) <= set(reader.fieldnames),
                "Missing outcome columns")
        for row in reader:
            rows += 1
            policy = row["policy"]
            require(policy in records, f"Unknown outcome policy: {policy}")
            episode = integer(row["episode"], "episode ID")
            require(0 <= episode < n, "Episode ID outside frozen budget")
            require(not seen[policy][episode], f"Duplicate outcome ID: {policy}/{episode}")
            seen[policy][episode] = True
            for key in numeric:
                records[policy][key][episode] = number(row[key], f"{policy}/{episode}/{key}")
            item = records[policy]
            penalty = item["inventory_penalty"][episode]
            qsq = item["inventory_sq_sum"][episode]
            qend = item["terminal_inventory_before_liquidation"][episode]
            close(penalty, .002 * qsq, f"Penalty identity {policy}/{episode}", 2e-9)
            close(item["objective"][episode], item["pnl"][episode] - penalty,
                  f"Objective identity {policy}/{episode}", 2e-8)
            close(item["terminal_liquidation_cost"][episode], .027 * abs(qend),
                  f"Terminal liquidation {policy}/{episode}", 2e-9)
            require(qend == math.trunc(qend) and abs(qend) <= 2,
                    f"Impossible terminal inventory: {policy}/{episode}")
            require(0 <= item["max_abs_inventory"][episode] <= 2,
                    f"Inventory bound violated in outcome: {policy}/{episode}")
            require(abs(item["reconciliation_error"][episode]) <= 2e-8,
                    f"Ledger reconciliation failed: {policy}/{episode}")
    require(rows == n * len(POLICIES), f"Incomplete outcome budget: {rows}")
    require(all(np.all(mask) for mask in seen.values()), "Missing policy/episode outcome IDs")
    return records, rows


def read_actions(study: Path, config: dict) -> dict[str, np.ndarray]:
    with np.load(study / "actions.npz", allow_pickle=False) as data:
        require(set(data.files) == set(POLICIES), "Action arrays do not match frozen policies")
        actions = {policy: np.array(data[policy], copy=True) for policy in POLICIES}
    for policy, array in actions.items():
        require(array.dtype == np.int8, f"Action array dtype is not int8: {policy}")
        require(array.shape == (config["episodes"], config["horizon"]),
                f"Action array shape differs: {policy}")
        require(np.all((array >= 0) & (array <= 10)), f"Invalid action ID: {policy}")
    return actions


def replay_actions(tapes: dict, actions: dict[str, np.ndarray], records: dict,
                   config: dict) -> tuple[dict, float]:
    """Replay every action with a separate cash/inventory implementation."""
    n, horizon = config["episodes"], config["horizon"]
    x, h, z = (tapes[k] for k in ("x", "h", "z"))
    returns = .03 * x[:, :-1] + .30 * z
    prices = np.empty((n, horizon + 1), dtype=np.float64)
    prices[:, 0] = 1000.
    for period in range(horizon):
        prices[:, period + 1] = prices[:, period] + returns[:, period]
    rho = config["theta"] * h
    scale = np.sqrt(1 - rho**2)
    potential = {}
    for side, noise, side_name in ((1, tapes["u_bid"], "bid"),
                                   (-1, tapes["u_ask"], "ask")):
        latent = rho * side * z + scale * noise
        for depth in (0, 1):
            threshold = ndtri(expit(-.3 - .7 * depth - .2 * side * x[:, :-1]))
            potential[(side_name, depth)] = latent <= threshold

    trace_expectations = {}
    max_error = 0.
    for policy in POLICIES:
        q = np.zeros(n, dtype=np.int16)
        cash = np.zeros(n, dtype=np.float64)
        qsq = np.zeros(n, dtype=np.float64)
        maxq = np.zeros(n, dtype=np.int16)
        trace_expectations[policy] = {}
        for period in range(horizon):
            selected = actions[policy][:, period]
            bid = BID_DEPTH[selected]
            ask = ASK_DEPTH[selected]
            passive = selected < 9
            bid_on = passive & (bid >= 0)
            ask_on = passive & (ask >= 0)
            buy = selected == 9
            sell = selected == 10
            require(not np.any((q == 2) & (bid_on | buy)),
                    f"Unsafe bid/market buy action: {policy}/{period}")
            require(not np.any((q == -2) & (ask_on | sell)),
                    f"Unsafe ask/market sell action: {policy}/{period}")
            bid_fill = bid_on & np.where(bid == 0, potential[("bid", 0)][:, period],
                                          potential[("bid", 1)][:, period])
            ask_fill = ask_on & np.where(ask == 0, potential[("ask", 0)][:, period],
                                          potential[("ask", 1)][:, period])
            before_q = q.copy()
            before_cash = cash.copy()
            price = prices[:, period]
            cash += bid_fill * (-price + .025 + .025 * np.maximum(bid, 0) - .001)
            cash += ask_fill * (price + .025 + .025 * np.maximum(ask, 0) - .001)
            cash += buy * (-price - .027) + sell * (price - .027)
            q += bid_fill.astype(np.int16) - ask_fill.astype(np.int16)
            q += buy.astype(np.int16) - sell.astype(np.int16)
            require(np.all(abs(q) <= 2), f"Realized inventory outside bound: {policy}/{period}")
            qsq += before_q.astype(float)**2
            np.maximum(maxq, abs(q), out=maxq)
            for episode in range(config["trace_episodes"]):
                trace_expectations[policy][episode, period] = dict(
                    inventory=int(before_q[episode]), signal=int(x[episode, period]),
                    price=float(price[episode]), cash=float(before_cash[episode]),
                    action=int(selected[episode]), return_=float(returns[episode, period]),
                    bid_fill=int(bid_fill[episode]) if bid_on[episode] else -1,
                    ask_fill=int(ask_fill[episode]) if ask_on[episode] else -1,
                    next_inventory=int(q[episode]), next_cash=float(cash[episode]),
                )
        final_q = q.astype(float)
        liquidation = .027 * abs(final_q)
        pnl = cash + final_q * prices[:, horizon] - liquidation
        objective = pnl - .002 * qsq
        expected = dict(pnl=pnl, objective=objective, inventory_penalty=.002 * qsq,
                        inventory_sq_sum=qsq, max_abs_inventory=maxq,
                        terminal_inventory_before_liquidation=final_q,
                        terminal_liquidation_cost=liquidation)
        for key, recomputed in expected.items():
            supplied = records[policy][key]
            error = float(np.max(np.abs(supplied - recomputed)))
            tolerance = 2e-8 if key in ("pnl", "objective") else 2e-9
            require(error <= tolerance, f"Independent ledger replay mismatch: {policy}/{key} {error}")
            max_error = max(max_error, error)
    return trace_expectations, max_error


def read_traces(study: Path, config: dict, tapes: dict, expectations: dict) -> int:
    keys = {(policy, episode, period) for policy in POLICIES
            for episode in range(config["trace_episodes"])
            for period in range(config["horizon"])}
    beliefs = {(policy, episode): .5 for policy in POLICIES
               for episode in range(config["trace_episodes"])}
    rows = {}
    with gzip.open(study / "traces.csv.gz", "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and set(TRACE_COLUMNS) <= set(reader.fieldnames),
                "Trace columns differ from frozen schema")
        for row in reader:
            key = (row["policy"], integer(row["episode"], "trace episode"),
                   integer(row["t"], "trace period"))
            require(key in keys, f"Unexpected trace key: {key}")
            require(key not in rows, f"Duplicate trace key: {key}")
            rows[key] = row
    require(set(rows) == keys, "Missing trace decision rows")
    for policy in POLICIES:
        for episode in range(config["trace_episodes"]):
            for period in range(config["horizon"]):
                row = rows[policy, episode, period]
                expected = expectations[policy][episode, period]
                for column in ("inventory", "signal", "action", "bid_fill", "ask_fill", "next_inventory"):
                    require(integer(row[column], f"trace {column}") == expected[column],
                            f"Trace {column} disagrees with independent replay: {policy}/{episode}/{period}")
                for column in ("price", "cash", "return_", "next_cash"):
                    close(number(row[column], f"trace {column}"), expected[column],
                          f"Trace {column} disagrees with independent replay: {policy}/{episode}/{period}")
                belief = beliefs[policy, episode]
                close(number(row["belief"], "trace belief"), belief,
                      f"Trace filtered belief: {policy}/{episode}/{period}", 2e-9)
                decision = (.5 if policy == "blind_regime" else
                            (float(tapes["h"][episode, period] == 1)
                             if policy == "full_information" else belief))
                close(number(row["decision_belief"], "trace decision belief"), decision,
                      f"Trace decision belief: {policy}/{episode}/{period}", 2e-9)
                z = (expected["return_"] - .03 * expected["signal"]) / .30
                loglike = []
                for hidden in (-1, 1):
                    value = 0.
                    for side, depth, fill in ((1, BID_DEPTH[expected["action"]], expected["bid_fill"]),
                                              (-1, ASK_DEPTH[expected["action"]], expected["ask_fill"])):
                        if depth < 0:
                            continue
                        marginal = expit(-.3 - .7 * depth - .2 * side * expected["signal"])
                        standardized = (ndtri(marginal) - hidden * config["theta"] * side * z) / math.sqrt(1 - config["theta"]**2)
                        value += float(log_ndtr(standardized if fill else -standardized))
                    loglike.append(value)
                prior = np.array([1 - belief, belief])
                logposterior = np.log(prior) + loglike
                posterior = math.exp(logposterior[1] - float(logsumexp(logposterior)))
                beliefs[policy, episode] = config["kappa"] + (1 - 2 * config["kappa"]) * posterior
    return len(rows)


def check_summary(study: Path, records: dict, config: dict) -> dict:
    summary = json_read(study / "summary.json")
    policies = summary.get("policies")
    contrasts = summary.get("contrasts")
    require(isinstance(policies, list) and len(policies) == len(POLICIES),
            "Summary policy list is incomplete")
    require(isinstance(contrasts, list) and len(contrasts) == len(CONTRASTS),
            "Summary contrast list is incomplete")
    require([entry.get("policy") for entry in policies] == list(POLICIES),
            "Summary policy order differs from frozen protocol")
    n, alpha = config["episodes"], config["alpha"]
    policy_results = {}
    for entry in policies:
        policy = entry["policy"]
        objective = records[policy]["objective"]
        mean = float(objective.mean())
        se = float(objective.std(ddof=1) / math.sqrt(n))
        pnl_mean = float(records[policy]["pnl"].mean())
        for key, value in (("mean", mean), ("se", se), ("pnl_mean", pnl_mean)):
            close(number(entry.get(key), f"summary {policy}/{key}"), value,
                  f"Summary policy {policy}/{key}", 5e-10)
        policy_results[policy] = dict(mean=mean, se=se, pnl_mean=pnl_mean)
    expected_results = []
    for index, (mechanism, policy, baseline) in enumerate(CONTRASTS):
        entry = contrasts[index]
        require((entry.get("mechanism"), entry.get("policy"), entry.get("baseline"))
                == (mechanism, policy, baseline), f"Wrong contrast identity or order: {index}")
        difference = records[policy]["objective"] - records[baseline]["objective"]
        mean = float(difference.mean())
        se = float(difference.std(ddof=1) / math.sqrt(n))
        df = n - 1
        ordinary = float(student_t.ppf(1 - alpha / 2, df))
        simultaneous = float(student_t.ppf(1 - alpha / (2 * config["family_size"]), df))
        expected = dict(mechanism=mechanism, policy=policy, baseline=baseline,
                        mean=mean, se=se, low=mean - ordinary * se,
                        high=mean + ordinary * se,
                        simultaneous_low=mean - simultaneous * se,
                        simultaneous_high=mean + simultaneous * se,
                        df=df, n=n)
        for key in ("mean", "se", "low", "high", "simultaneous_low", "simultaneous_high"):
            close(number(entry.get(key), f"summary {mechanism}/{key}"), expected[key],
                  f"Summary contrast {mechanism}/{key}", 5e-10)
        for key in ("df", "n"):
            require(integer(entry.get(key), f"summary {mechanism}/{key}") == expected[key],
                    f"Summary contrast {mechanism}/{key} differs")
        expected_results.append(expected)
    return {"policies": policy_results, "contrasts": expected_results}


def verify(root: Path, study: Path) -> dict:
    root, study = root.resolve(), study.resolve()
    require(root.is_dir() and study.is_dir(), "Root/study directory missing")
    config = frozen_config(root)
    manifest, numerical, git_validation = check_manifest(root, study, config)
    tapes = regenerate_tapes(config)
    check_tapes(tapes, manifest, config)
    records, row_count = read_episodes(study, config)
    actions = read_actions(study, config)
    expectations, max_ledger_error = replay_actions(tapes, actions, records, config)
    trace_rows = read_traces(study, config, tapes, expectations)
    statistics = check_summary(study, records, config)
    return {
        "passed": True,
        "scope": "Independent raw reader, regenerated paired tapes, action-level cash ledger and paired episode inference; no production simulation/statistics imports",
        "verifier_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(root / CONFIG_PATH),
        "manifest_sha256": sha(study / "manifest.json"),
        "source_inputs_checked": len(manifest["input_sha256"]),
        "file_sha256": {name: sha(study / name) for name in ARTIFACTS},
        "numerical_build_sha256": manifest["build"]["sha256"],
        "validation_git_scope": git_validation,
        "numerical_checks_reported_passed": numerical["passed"],
        "numerical_table_array_scope": "Fingerprint metadata checked; arrays are not embedded here for independent rehashing",
        "tape_sha256": manifest["tape_sha256"],
        "policies": len(POLICIES), "episodes_per_policy": config["episodes"],
        "outcome_rows": row_count, "trace_rows": trace_rows,
        "maximum_independent_ledger_error": max_ledger_error,
        "recomputed": statistics,
    }


def rewrite_outcomes(path: Path, mutate) -> None:
    source = path.with_name("episodes.original.gz")
    path.replace(source)
    try:
        with gzip.open(source, "rt", newline="") as original, gzip.open(path, "wt", newline="") as changed:
            reader = csv.DictReader(original)
            writer = csv.DictWriter(changed, fieldnames=reader.fieldnames)
            writer.writeheader()
            for index, row in enumerate(reader):
                for candidate in mutate(index, row):
                    writer.writerow(candidate)
    finally:
        source.unlink(missing_ok=True)


def fault_probes(root: Path, study: Path) -> list[dict]:
    """Mutate copied artifacts; keep the accepted study and current tree untouched."""
    cases = []
    names = ("wrong_protocol", "stale_source_hash", "stale_artifact_hash", "missing_id",
             "duplicate_id", "nonfinite_outcome", "changed_scalar_ledger", "corrupt_action_data",
             "changed_trace_state", "wrong_contrast")
    with tempfile.TemporaryDirectory(prefix="mechanism-study-faults-") as directory:
        temporary = Path(directory)
        for case in names:
            fixture = temporary / case
            fixture.mkdir()
            for name in ("manifest.json",) + ARTIFACTS:
                shutil.copy2(study / name, fixture / name)
            manifest = json_read(fixture / "manifest.json")
            if case == "wrong_protocol":
                manifest["protocol"]["seed"] += 1
            elif case == "stale_source_hash":
                manifest["input_sha256"]["verification/mechanism_study.py"] = "0" * 64
            elif case == "stale_artifact_hash":
                with (fixture / "actions.npz").open("ab") as stream:
                    stream.write(b"stale")
            elif case in ("missing_id", "duplicate_id", "nonfinite_outcome", "changed_scalar_ledger"):
                def mutate(index, row):
                    if index != 0:
                        return (row,)
                    if case == "missing_id":
                        return ()
                    if case == "duplicate_id":
                        return (row, row)
                    if case == "nonfinite_outcome":
                        row["objective"] = "nan"
                    else:
                        row["pnl"] = str(float(row["pnl"]) + .5)
                    return (row,)
                rewrite_outcomes(fixture / "episodes.csv.gz", mutate)
                manifest["files"]["episodes.csv.gz"] = sha(fixture / "episodes.csv.gz")
            elif case == "corrupt_action_data":
                with np.load(fixture / "actions.npz", allow_pickle=False) as data:
                    arrays = {name: np.array(data[name], copy=True) for name in data.files}
                arrays["active"][500, 0] = 99
                np.savez_compressed(fixture / "actions.npz", **arrays)
                manifest["files"]["actions.npz"] = sha(fixture / "actions.npz")
            elif case == "changed_trace_state":
                source = fixture / "traces.csv.gz"
                original = source.with_name("traces.original.gz")
                source.replace(original)
                try:
                    with gzip.open(original, "rt", newline="") as read_stream, gzip.open(source, "wt", newline="") as write_stream:
                        reader = csv.DictReader(read_stream)
                        writer = csv.DictWriter(write_stream, fieldnames=reader.fieldnames)
                        writer.writeheader()
                        for index, row in enumerate(reader):
                            if index == 0:
                                row["next_cash"] = str(float(row["next_cash"]) + .5)
                            writer.writerow(row)
                finally:
                    original.unlink(missing_ok=True)
                manifest["files"]["traces.csv.gz"] = sha(source)
            elif case == "wrong_contrast":
                summary = json_read(fixture / "summary.json")
                summary["contrasts"][0]["mean"] += 1.
                (fixture / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                manifest["files"]["summary.json"] = sha(fixture / "summary.json")
            (fixture / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            try:
                verify(root, fixture)
            except (VerificationError, OSError, ValueError, KeyError, EOFError) as exc:
                cases.append({"case": case, "rejected": True, "reason": str(exc)[:300]})
            else:
                raise VerificationError(f"Fault probe was accepted: {case}")
    return cases


def verify_repeat(root: Path, first: Path, repeat: Path) -> dict:
    require(first.resolve() != repeat.resolve(), "Cold repeats must use different directories")
    second_receipt = verify(root, repeat)
    first_manifest = json_read(first / "manifest.json")
    second_manifest = json_read(repeat / "manifest.json")
    for key in ("input_sha256", "build", "table_fingerprints", "tape_sha256",
                "numerical_contract", "selected_resolution", "protocol"):
        require(first_manifest[key] == second_manifest[key],
                f"Cold-repeat manifest identity differs: {key}")
    require(first_manifest["cold_control_build"] is second_manifest["cold_control_build"] is True
            and first_manifest["shared_control_cache"] is second_manifest["shared_control_cache"] is False,
            "Cold repeats did not certify independent control builds")
    require(first_manifest["files"] == second_manifest["files"],
            "Cold-repeat artifact content hashes differ")
    return {"passed": True, "first_manifest_sha256": sha(first / "manifest.json"),
            "repeat_manifest_sha256": sha(repeat / "manifest.json"),
            "exact_five_file_content": True,
            "matching_input_build_table_tape_and_numerical_contract": True,
            "first_file_sha256": first_manifest["files"],
            "repeat_file_sha256": second_receipt["file_sha256"],
            "scope": "Two separate study directories each passed independent raw/action/statistical verification on this numerical build"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--study", type=Path, default=Path("outputs/verification/reaudit/mechanism_study"))
    parser.add_argument("--repeat", type=Path,
                        help="Independently verify a second cold run and require exact artifact content")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fault-probes", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    study = (root / args.study).resolve() if not args.study.is_absolute() else args.study.resolve()
    result = verify(root, study)
    if args.repeat:
        repeat = (root / args.repeat).resolve() if not args.repeat.is_absolute() else args.repeat.resolve()
        result["repeat"] = verify_repeat(root, study, repeat)
    if args.fault_probes:
        result["fault_probes"] = fault_probes(root, study)
        result["fault_probes_rejected"] = len(result["fault_probes"])
    if args.out:
        destination = args.out if args.out.is_absolute() else root / args.out
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
