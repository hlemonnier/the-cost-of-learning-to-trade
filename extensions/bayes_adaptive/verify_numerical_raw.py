"""Read-only, independent reconciliation of saved joint numerical refinements.

See NUMERICAL_VALIDATION.md for failure modes written before this implementation.
The production solver and its interpolation/comparison routines are not imported.
Exit 0 means that the selected saved rows agree with authenticated raw arrays;
it does not mean that the unchanged numerical convergence gate passed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from verify_extension import (CONFIG_SHA256, NUMERICAL_CONTRACT,
                              VerificationError, canonical_hash, frozen_config,
                              need, numerical_design, read_json, sha,
                              verify_numerical_probes)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VALUE_KINDS = ("bayes", "reveal1", "reveal2", "reveal3", "no_feedback", "frozen_model")
Q_KINDS = ("bayes", "no_feedback", "frozen_model")
FLOAT_ATOL = 1e-10


def _source_identity(core_root: Path) -> dict:
    files = {"solver.py": sha(HERE / "solver.py")}
    for name in ("control", "filtering", "model", "numerics"):
        files[f"trade_learning.{name}"] = sha(core_root / "src" / "trade_learning" / f"{name}.py")
    return {"files": files, "sha256": canonical_hash(files)}


def _core_root(explicit: Path | None) -> Path:
    candidate = explicit or Path(os.environ.get("TRADE_LEARNING_CORE_ROOT", ROOT))
    candidate = candidate.resolve()
    if explicit is None and not (candidate / "src/trade_learning/model.py").is_file():
        candidate = candidate / "Trade Learning_Hugo_Lemonnier"
    need((candidate / "src/trade_learning/model.py").is_file(),
         f"Original extracted core source is missing: {candidate}")
    return candidate


def _expected_spec(config: dict, grid: list[int], quadrature: int,
                   geometry: str) -> dict:
    return {"theta": config["theta"], "kappas": config["kappas"],
            "horizon": config["horizon"], "qmax": config["qmax"],
            "weight_points": grid[0], "belief_points": grid[1],
            "quadrature_points": quadrature,
            "known_belief_points": config["known_reference"]["belief_points"],
            "known_quadrature_points": config["known_reference"]["quadrature_points"],
            "belief_geometry": geometry}


def _check_ladder(receipt: dict, config: dict, numeric_config: dict,
                  geometry: str) -> tuple[list[dict], dict]:
    pairs = receipt.get("comparisons")
    need(isinstance(pairs, list) and len(pairs) >= 2 and len(pairs) % 2 == 0,
         "Receipt has no complete pair of independent refinement axes")
    grid_levels = numeric_config["joint_grid_levels"]
    quadrature_levels = numeric_config["quadrature_levels"]
    gi = hi = 1
    for offset in range(0, len(pairs), 2):
        pair = pairs[offset:offset + 2]
        need([item.get("axis") for item in pair] == ["belief", "quadrature"],
             f"Pair {offset // 2} omits or reorders an independent axis")
        need(gi < len(grid_levels) and hi < len(quadrature_levels),
             "Receipt continued after exhausting a refinement axis")
        fine = _expected_spec(config, grid_levels[gi], quadrature_levels[hi], geometry)
        for axis, item in zip(("belief", "quadrature"), pair):
            coarse = (_expected_spec(config, grid_levels[gi - 1], quadrature_levels[hi], geometry)
                      if axis == "belief" else
                      _expected_spec(config, grid_levels[gi], quadrature_levels[hi - 1], geometry))
            need(item.get("fine_specification") == fine and
                 item.get("coarse_specification") == coarse,
                 f"Pair {offset // 2} {axis} skips a level or changes another setting")
            need(item.get("all_horizons") == list(range(1, config["horizon"] + 1))
                 and item.get("probe_count") == 2581,
                 f"Pair {offset // 2} {axis} does not cover the fixed probes/horizons")
            rows = item.get("by_horizon_policy")
            need(isinstance(rows, list) and len(rows) == config["horizon"] * len(VALUE_KINDS),
                 f"Pair {offset // 2} {axis} has incomplete policy/horizon rows")
            need(isinstance(item.get("passes_predeclared_rule"), bool),
                 f"Pair {offset // 2} {axis} lacks a boolean pass flag")
        if all(item["passes_predeclared_rule"] for item in pair):
            need(offset == len(pairs) - 2, "Receipt continued after its first passing pair")
        else:
            gi += int(not pair[0]["passes_predeclared_rule"])
            hi += int(not pair[1]["passes_predeclared_rule"])
    last = pairs[-1]["fine_specification"]
    need(receipt.get("last_attempted_resolution") == last,
         "Last-attempted specification differs from final saved comparison")
    selected = receipt.get("selected_resolution")
    if selected is not None:
        need(selected == last and all(item["passes_predeclared_rule"] for item in pairs[-2:]),
             "Selected resolution was not the first jointly passing setting")
    if receipt.get("passes_predeclared_rule") is True:
        need(selected is not None and receipt.get("known_reference", {}).get("passes_predeclared_rule") is True,
             "Receipt declares acceptance without joint and known-model gates")
    else:
        need(receipt.get("passes_predeclared_rule") is False,
             "Numerical acceptance flag must be boolean")
    return pairs, last


def _coordinates(joint: np.ndarray) -> np.ndarray:
    need(joint.ndim == 2 and joint.shape[1] == 4 and np.isfinite(joint).all()
         and (joint >= 0).all(), "Invalid joint probe probabilities")
    mass = joint.sum(axis=1)
    need(np.max(np.abs(mass - 1.)) <= 1e-10, "Joint probes do not sum to one")
    p = joint / mass[:, None]
    w0 = p[:, 0] + p[:, 1]
    w1 = p[:, 2] + p[:, 3]
    b0 = np.divide(p[:, 1], w0, out=np.full(len(p), .5), where=w0 > 0)
    b1 = np.divide(p[:, 3], w1, out=np.full(len(p), .5), where=w1 > 0)
    return np.column_stack((w0, b0, b1))


def _interpolation_plan(joint: np.ndarray,
                        axes: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Locate points with independent axis searches, not solver floor indexing."""
    coords = _coordinates(joint)
    counts = tuple(len(axis) for axis in axes)
    lows = []
    fractions = []
    for column, (count, axis) in enumerate(zip(counts, axes)):
        low = np.clip(np.searchsorted(axis, coords[:, column], side="right") - 1, 0, count - 2)
        fraction = (coords[:, column] - axis[low]) / (axis[low + 1] - axis[low])
        need(np.min(fraction) >= -1e-13 and np.max(fraction) <= 1 + 1e-13,
             "Interpolation point lies outside its grid cell")
        lows.append(low)
        fractions.append(fraction)
    corners = []
    for dw in (0, 1):
        for db0 in (0, 1):
            for db1 in (0, 1):
                col = np.ravel_multi_index((lows[0] + dw, lows[1] + db0, lows[2] + db1), counts)
                weight = np.ones(len(coords))
                for flag, fraction in zip((dw, db0, db1), fractions):
                    weight *= fraction if flag else 1 - fraction
                corners.append((col, weight))
    weights = np.stack([weight for _, weight in corners])
    need(np.min(weights) >= -1e-13 and np.max(np.abs(weights.sum(axis=0) - 1.)) < 1e-13,
         "Interpolation weights are negative or fail to sum to one")
    return tuple(corners)


def _verified_axes(complete: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Authenticate physical V3 axes from their specification and manifest."""
    spec = complete.get("specification")
    need(complete.get("format_version") == 3 and isinstance(spec, dict),
         "Historical or malformed numerical family cannot enter a V3 raw audit")
    geometry = spec.get("belief_geometry")
    need(geometry in ("uniform", "endpoint_sine"), "Unsupported numerical geometry")
    weight_count, belief_count = spec.get("weight_points"), spec.get("belief_points")
    need(all(isinstance(count, int) and not isinstance(count, bool) and count >= 2
             for count in (weight_count, belief_count)), "Invalid physical-axis lengths")
    weight = np.linspace(0., 1., weight_count)
    u = np.linspace(0., 1., belief_count)
    if geometry == "uniform":
        belief = u
    else:
        belief = np.sin(.5 * np.pi * u) ** 2
        belief[belief_count // 2:] = 1. - belief[:(belief_count + 1) // 2][::-1]
        belief[0], belief[-1] = 0., 1.
        if belief_count % 2:
            belief[belief_count // 2] = .5
    axes = (weight, belief, belief.copy())
    need(all(np.all(np.diff(axis) > 0.) and axis[0] == 0. and axis[-1] == 1.
             for axis in axes), "Physical axes are not increasing with exact endpoints")
    axis_values = {name: axis.tolist() for name, axis in zip(
        ("weight0", "conditional_plus0", "conditional_plus1"), axes)}
    expected = {
        "name": geometry,
        "axis_order": ["weight0", "conditional_plus0", "conditional_plus1"],
        "axis_generation": ("uniform-weight_symmetric-sine-squared-beliefs-v1"
                            if geometry == "endpoint_sine" else "uniform-linspace-v1"),
        "interpolation": ("physical-interval-search-trilinear-v1"
                          if geometry == "endpoint_sine" else "uniform-scaled-trilinear-v1"),
        "axes": axis_values,
        "axes_sha256": canonical_hash(axis_values),
    }
    need(complete.get("grid_geometry") == expected,
         "Recorded physical axes differ from independently reconstructed geometry")
    return axes


def _legal_actions(inventory: np.ndarray, qmax: int) -> np.ndarray:
    q = np.asarray(inventory)
    need(q.ndim == 1 and np.isfinite(q).all() and np.equal(q, np.floor(q)).all()
         and (np.abs(q) <= qmax).all(), "Invalid probe inventory")
    # Passive action IDs are the Cartesian product of bid,ask depths (-1,0,1).
    bid = np.repeat(np.array([-1, 0, 1]), 3)
    ask = np.tile(np.array([-1, 0, 1]), 3)
    legal = np.ones((len(q), 11), dtype=bool)
    legal[:, :9] &= ~((q[:, None] >= qmax) & (bid >= 0))
    legal[:, :9] &= ~((q[:, None] <= -qmax) & (ask >= 0))
    legal[:, 9] = q < qmax
    legal[:, 10] = q > -qmax
    need(np.all(np.sum(legal, axis=1) >= 2), "A probe has fewer than two admissible actions")
    return legal


def _load_family(cache_root: Path, spec: dict, source: dict, build: dict) -> tuple[dict, dict, int, tuple]:
    name = f"w{spec['weight_points']}-b{spec['belief_points']}-gh{spec['quadrature_points']}"
    directory = cache_root / name
    complete = read_json(directory / "complete.json")
    need(complete.get("specification") == spec and complete.get("source") == source
         and complete.get("build") == build,
         f"Stale or wrong-source numerical family: {directory}")
    axes = _verified_axes(complete)
    need(complete.get("numerical_contract") == NUMERICAL_CONTRACT and
         complete.get("retain_q") == list(Q_KINDS),
         f"Numerical contract or retained Q arrays changed: {directory}")
    signature = complete.get("artifact_sha256")
    need(signature == canonical_hash({key: value for key, value in complete.items()
                                      if key != "artifact_sha256"}),
         f"Family manifest identity mismatch: {directory}")
    records = complete.get("arrays")
    need(isinstance(records, list), f"Missing family array manifest: {directory}")
    expected_names = ({f"value-{kind}.npy" for kind in VALUE_KINDS}
                      | {f"q-{kind}.npy" for kind in Q_KINDS}
                      | {"known-q-0.npy", "known-q-1.npy"})
    need({item.get("file") for item in records if isinstance(item, dict)} == expected_names
         and len(records) == len(expected_names),
         f"Family array manifest is incomplete or duplicated: {directory}")
    loaded = {}
    bytes_checked = 0
    for item in records:
        filename = item["file"]
        need(isinstance(filename, str) and Path(filename).name == filename,
             "Unsafe array path in family manifest")
        path = directory / filename
        need(path.is_file() and path.stat().st_size == item.get("bytes") and
             sha(path) == item.get("sha256"), f"Array bytes differ from manifest: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        need(array.dtype == np.float64 and str(array.dtype) == item.get("dtype")
             and list(array.shape) == item.get("shape"),
             f"Array shape/dtype differs from manifest: {path}")
        shape = ([spec["horizon"] + 1, 2 * spec["qmax"] + 1, 3]
                 + ([spec["known_belief_points"]] if filename.startswith("known-q-")
                    else [spec["weight_points"] * spec["belief_points"] ** 2]))
        if filename.startswith(("q-", "known-q-")):
            shape.append(11)
        need(list(array.shape) == shape, f"Unexpected control-array shape: {path}")
        loaded[filename] = array
        bytes_checked += item["bytes"]
    need({path.name for path in directory.iterdir() if path.is_file()} ==
         expected_names | {"complete.json"},
         f"Numerical family contains missing or unrecorded files: {directory}")
    return loaded, complete, bytes_checked, axes


def _values(table: np.ndarray, remaining: int, inventory: np.ndarray,
            signal: np.ndarray, corners: tuple, qmax: int) -> np.ndarray:
    result = np.zeros(len(inventory))
    for column, weight in corners:
        block = table[remaining, inventory + qmax, signal + 1, column]
        need(np.isfinite(block).all(), "Nonfinite value at a numerical probe corner")
        result += weight * block
    need(np.isfinite(result).all(), "Interpolated value is nonfinite")
    return result


def _scores(table: np.ndarray, remaining: int, inventory: np.ndarray,
            signal: np.ndarray, corners: tuple, legal: np.ndarray, qmax: int) -> np.ndarray:
    result = np.zeros((len(inventory), 11))
    for column, weight in corners:
        block = table[remaining, inventory + qmax, signal + 1, column]
        need(np.isfinite(block[legal]).all() and np.isneginf(block[~legal]).all(),
             "Retained Q array has an invalid legal-action mask or nonfinite score")
        result += weight[:, None] * np.where(legal, block, 0.)
    return np.where(legal, result, -np.inf)


def _stable_actions(scores: np.ndarray, legal: np.ndarray) -> np.ndarray:
    best = np.max(scores, axis=1, keepdims=True)
    eligible = legal & ((best - scores) <= 1e-12)
    need(np.all(np.any(eligible, axis=1)), "Stable action selection has no legal candidate")
    return np.argmax(eligible, axis=1)


def _close(actual: object, reported: object, description: str) -> None:
    need(isinstance(reported, (int, float)) and not isinstance(reported, bool)
         and np.isfinite(reported) and abs(float(actual) - float(reported)) <= FLOAT_ATOL,
         f"{description}: raw {actual!r} differs from receipt {reported!r}")


def _audit_comparison(item: dict, coarse_arrays: dict, fine_arrays: dict,
                      coarse_corners: tuple, fine_corners: tuple,
                      coarse_initial: tuple, fine_initial: tuple,
                      probes: dict, thresholds: dict, pair_index: int) -> dict:
    q = probes["inventory"].astype(np.int64)
    x = probes["signal"].astype(np.int64)
    need(np.isin(x, (-1, 0, 1)).all(), "Invalid probe signal")
    qmax = item["fine_specification"]["qmax"]
    legal = _legal_actions(q, qmax)
    iq = np.zeros(5, dtype=np.int64)
    ix = np.zeros(5, dtype=np.int64)
    rows = item["by_horizon_policy"]
    maxima = {"max_initial_value_change": 0., "max_probe_value_change": 0.,
              "max_finite_q_change": 0., "max_robust_disagreement_fraction": 0.}
    checked = 0
    for remaining in range(1, item["fine_specification"]["horizon"] + 1):
        for kind in VALUE_KINDS:
            row = rows[checked]
            need(row.get("remaining") == remaining and row.get("kind") == kind,
                 f"Pair {pair_index} {item['axis']} row order or coverage changed")
            filename = f"value-{kind}.npy"
            va = _values(coarse_arrays[filename], remaining, q, x, coarse_corners, qmax)
            vb = _values(fine_arrays[filename], remaining, q, x, fine_corners, qmax)
            ia = _values(coarse_arrays[filename], remaining, iq, ix, coarse_initial, qmax)
            ib = _values(fine_arrays[filename], remaining, iq, ix, fine_initial, qmax)
            metrics = {"initial_value_change": float(np.max(np.abs(ia - ib))),
                       "probe_value_change": float(np.max(np.abs(va - vb)))}
            if kind in Q_KINDS:
                filename = f"q-{kind}.npy"
                qa = _scores(coarse_arrays[filename], remaining, q, x, coarse_corners, legal, qmax)
                qb = _scores(fine_arrays[filename], remaining, q, x, fine_corners, legal, qmax)
                finite_q_change = float(np.max(np.abs(qa[legal] - qb[legal])))
                ordered = np.sort(qb, axis=1)
                robust = (ordered[:, -1] - ordered[:, -2]) > thresholds["robust_gap"]
                denominator = int(np.sum(robust))
                disagreements = int(np.sum(robust &
                                           (_stable_actions(qa, legal) != _stable_actions(qb, legal))))
                fraction = disagreements / denominator if denominator else 0.
                metrics.update(finite_q_change=finite_q_change,
                               robust_disagreement_fraction=fraction)
                need(row.get("robust_probe_count") == denominator and
                     row.get("robust_disagreements") == disagreements,
                     f"Pair {pair_index} {item['axis']} {kind} horizon {remaining} robust counts differ")
            else:
                need("finite_q_change" not in row and "robust_disagreement_fraction" not in row,
                     "Value-only revelation row claims a retained-Q check")
            for field, result in metrics.items():
                _close(result, row.get(field),
                       f"Pair {pair_index} {item['axis']} {kind} horizon {remaining} {field}")
                maxima["max_" + field] = max(maxima["max_" + field], result)
            checked += 1
    need(set(item.get("summary", {})) == set(maxima), "Numerical summary fields changed")
    for field, result in maxima.items():
        _close(result, item["summary"][field], f"Pair {pair_index} {item['axis']} {field}")
    passes = (maxima["max_initial_value_change"] <= thresholds["initial"] and
              maxima["max_probe_value_change"] <= thresholds["probe_sup"] and
              maxima["max_finite_q_change"] <= thresholds["probe_sup"] and
              maxima["max_robust_disagreement_fraction"] <= thresholds["robust_disagreement"])
    need(item["passes_predeclared_rule"] is bool(passes),
         f"Pair {pair_index} {item['axis']} threshold flag differs from raw arrays")
    return {"pair_index": pair_index, "axis": item["axis"],
            "coarse_artifact_sha256": item["coarse_artifact_sha256"],
            "fine_artifact_sha256": item["fine_artifact_sha256"],
            "horizon_policy_rows_reconciled": checked,
            "maxima_from_raw_arrays": maxima,
            "passes_unchanged_joint_gate": bool(passes)}


def audit(receipt_path: Path, cache_root: Path, pair_choice: str, axis_choice: str,
          core_root: Path | None = None) -> dict:
    receipt_path = receipt_path.resolve()
    cache_root = cache_root.resolve()
    core_root = _core_root(core_root)
    receipt_hash = sha(receipt_path)
    receipt = read_json(receipt_path)
    statistical = frozen_config(HERE)
    numeric = numerical_design(HERE, statistical)
    numeric_hash = sha(HERE / "numerical_config.json")
    protocol_hash = receipt.get("protocol_config_sha256")
    need(protocol_hash in (CONFIG_SHA256, numeric_hash), "Numerical receipt has an unknown protocol hash")
    active = statistical if protocol_hash == CONFIG_SHA256 else numeric
    geometry = receipt.get("belief_geometry")
    need(geometry in ("uniform", "endpoint_sine"),
         "Numerical receipt omits the explicit V3 physical geometry")
    if geometry == "endpoint_sine":
        need(protocol_hash == numeric_hash and
             receipt.get("geometry_amendment_sha256") == sha(HERE / "GEOMETRY_AMENDMENT.md") and
             receipt.get("geometry_wrapper_sha256") == sha(HERE / "refine_geometry.py"),
             "Endpoint receipt does not bind the pre-outcome geometry amendment")
    else:
        need("geometry_amendment_sha256" not in receipt and
             "geometry_wrapper_sha256" not in receipt,
             "Uniform receipt claims endpoint geometry evidence")
    need(receipt.get("thresholds") == statistical["refinement"] and
         receipt.get("certified_bound") is False,
         "Numerical thresholds or certification claim changed")
    build = receipt.get("build")
    need(isinstance(build, dict) and build.get("sha256") == canonical_hash({
        key: value for key, value in build.items() if key != "sha256"}),
        "Numerical build fingerprint is malformed")
    need(receipt.get("source") == _source_identity(core_root),
         "Numerical receipt source is stale or from another extracted core")
    need(receipt.get("probe_design") == {"boundary_count": 405,
         "random_count": statistical["refinement"]["random_probes"],
         "reflection_count": 128}, "Numerical probe design changed")
    if protocol_hash == numeric_hash and (
            geometry == "endpoint_sine" or receipt.get("selected_resolution") is not None):
        need(receipt.get("numerical_config_sha256") == numeric_hash and
             receipt.get("statistical_protocol_sha256") == CONFIG_SHA256 and
             receipt.get("numerical_amendment_sha256") == sha(HERE / "NUMERICAL_AMENDMENT.md") and
             receipt.get("numerical_wrapper_sha256") == sha(HERE / "refine_numerical.py"),
             "Amended receipt does not bind its pre-outcome numerical amendment")
    probes_path = receipt_path.parent / "numerical_probes.npz"
    need(probes_path.is_file() and sha(probes_path) == receipt.get("probe_sha256"),
         "Numerical probe file does not match receipt")
    count = verify_numerical_probes(probes_path, statistical)
    need(count == 2581, "Unexpected probe count")
    with np.load(probes_path, allow_pickle=False) as raw:
        probes = {name: raw[name] for name in raw.files}
    pairs, last_spec = _check_ladder(receipt, statistical, active, geometry)
    pair_count = len(pairs) // 2
    if pair_choice == "all":
        indices = list(range(pair_count))
    elif pair_choice == "last":
        indices = [pair_count - 1]
    else:
        index = int(pair_choice)
        need(0 <= index < pair_count, f"Pair index must be in [0,{pair_count - 1}]")
        indices = [index]
    axes = ("belief", "quadrature") if axis_choice == "both" else (axis_choice,)
    selected = [(index, pairs[2 * index + (axis == "quadrature")])
                for index in indices for axis in axes]
    bytes_authenticated = 0
    checked = []
    for pair_index, item in selected:
        family_cache = {}
        family_identities = {}
        families = []
        plans = []
        initial_plans = []
        for position in ("coarse", "fine"):
            spec = item[f"{position}_specification"]
            key = (spec["weight_points"], spec["belief_points"],
                   spec["quadrature_points"], spec["belief_geometry"])
            if key not in family_cache:
                arrays, complete, byte_count, physical_axes = _load_family(
                    cache_root, spec, receipt["source"], receipt["build"])
                family_cache[key] = (arrays, physical_axes)
                family_identities[key] = complete["artifact_sha256"]
                bytes_authenticated += byte_count
            need(family_identities[key] == item[f"{position}_artifact_sha256"],
                 f"Pair {pair_index} {item['axis']} references a different {position} artifact")
            arrays, physical_axes = family_cache[key]
            families.append(arrays)
            plans.append(_interpolation_plan(probes["joint"], physical_axes))
            initial_plans.append(_interpolation_plan(probes["initial_joint"], physical_axes))
        checked.append(_audit_comparison(item, *families, *plans, *initial_plans,
                                         probes, statistical["refinement"], pair_index))
        family_cache.clear()
        families.clear()
        plans.clear()
        initial_plans.clear()
    need(sha(receipt_path) == receipt_hash, "Numerical receipt changed during raw audit")
    complete_coverage = len(selected) == len(pairs)
    last_pair_raw_pass = None
    if pair_count - 1 in indices and axis_choice == "both":
        last_pair_raw_pass = all(row["passes_unchanged_joint_gate"]
                                 for row in checked if row["pair_index"] == pair_count - 1)
    return {"audit_status": "raw_rows_reconciled", "scope": "joint saved-array comparisons only",
            "verifier_sha256": sha(Path(__file__)),
            "numerical_receipt": str(receipt_path), "numerical_receipt_sha256": receipt_hash,
            "statistical_protocol_sha256": CONFIG_SHA256,
            "numerical_config_sha256": numeric_hash,
            "receipt_protocol_config_sha256": protocol_hash,
            "belief_geometry": geometry,
            "current_source_identity": receipt["source"],
            "core_root": str(core_root),
            "receipt_build_sha256": receipt["build"].get("sha256"),
            "probe_sha256": receipt["probe_sha256"], "probe_count": count,
            "authenticated_array_bytes": bytes_authenticated,
            "audited_comparisons": checked,
            "all_saved_joint_comparisons_audited": complete_coverage,
            "last_pair_joint_gate_passed_if_audited": last_pair_raw_pass,
            "receipt_selected_resolution": receipt.get("selected_resolution"),
            "receipt_last_attempted_resolution": last_spec,
            "receipt_claims_numerical_acceptance": receipt.get("passes_predeclared_rule"),
            "raw_reconciliation_is_numerical_acceptance": False,
            "note": "A reconciled unresolved gate remains unresolved. This audit does not independently recompute scalar known-model references or certify a mathematical error bound."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numerical-checks", type=Path, required=True,
                        help="V3 numerical receipt for the exact family cache")
    parser.add_argument("--cache", type=Path, required=True,
                        help="V3 family-cache root named by each comparison specification")
    parser.add_argument("--root", type=Path,
                        help="Original core root containing src/trade_learning (needed for extracted review ZIPs)")
    parser.add_argument("--pair", default="last",
                        help="0-based pair index, 'last' (default), or 'all'")
    parser.add_argument("--axis", choices=("belief", "quadrature", "both"), default="both")
    args = parser.parse_args()
    try:
        result = audit(args.numerical_checks, args.cache, args.pair, args.axis, args.root)
    except (VerificationError, OSError, ValueError, KeyError, IndexError) as exc:
        print(json.dumps({"audit_status": "failed", "error": str(exc)}), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
