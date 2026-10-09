"""Independent E2E reader and mathematical oracle for the Bayes extension.

This file deliberately does not import the production market, filtering, ledger
or statistics modules. Failure modes were recorded in validation_failure_modes.md
before this verifier was written. A result is accepted only from raw evidence;
the producer's summary flags are insufficient.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from collections import OrderedDict

import numpy as np
from scipy.special import expit, ndtr, ndtri, roots_hermitenorm
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'verification'))
from publication_identity import recorded_core_hash, recorded_digest
K = np.array([[.75, .20, .05], [.10, .80, .10], [.05, .20, .75]])
BID = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, -2, -2], dtype=np.int8)
ASK = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1, -2, -2], dtype=np.int8)
THETA = .35
KAPPAS = (.002, .10)
TERMINAL_COST = .027
REAL_TOL = 3e-8
CONFIG_SHA256 = "8681e5a8c48f25967f95a4b7d42b76531cbc4e8e00d013114c3e6efe93aee462"
CORE_SOURCE_HASH = "cabebb53081506b722f4f9232e49a5171585217b7712e4c7a2940149aa6a34eb"
PROTOCOL_COMMIT = "8db62080ef4e5170413d01b7f5fdac33bfd2b1e5"
NUMERICAL_CONTRACT = {
    "version": 2, "score_dtype": "float64", "tie_absolute_tolerance": 1e-12,
    "tie_relative_tolerance": 0.0,
    "tie_action": "lowest admissible action ID within absolute tolerance of maximum",
    "bellman_value": "Q of the action selected by the same tie rule",
    "invalid_action": "negative infinity; no decision at remaining horizon zero",
}


class VerificationError(ValueError):
    """Raw extension evidence contradicts its frozen protocol."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise VerificationError(f"Invalid JSON {path}: {exc}") from exc
    need(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def grid_axes(specification: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild the declared physical axes without importing the control solver."""
    name = specification.get("belief_geometry")
    need(name in ("uniform", "endpoint_sine"), "Unknown belief-grid geometry")
    weight_count = specification.get("weight_points")
    belief_count = specification.get("belief_points")
    need(all(isinstance(count, int) and not isinstance(count, bool) and count >= 2
             for count in (weight_count, belief_count)), "Invalid belief-grid axis sizes")
    weight = np.linspace(0., 1., weight_count)
    position = np.linspace(0., 1., belief_count)
    if name == "uniform":
        belief = position
    else:
        belief = np.sin(.5 * np.pi * position) ** 2
        belief[belief_count // 2:] = 1. - belief[:(belief_count + 1) // 2][::-1]
        belief[0], belief[-1] = 0., 1.
        if belief_count % 2:
            belief[belief_count // 2] = .5
    need(np.all(np.diff(weight) > 0.) and np.all(np.diff(belief) > 0.)
         and weight[0] == belief[0] == 0.
         and weight[-1] == belief[-1] == 1.,
         "Belief-grid physical axes are not strictly increasing with exact endpoints")
    return weight, belief, belief.copy()


def grid_cells(specification: dict, coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Locate joint coordinates in physical intervals, independently of solver code."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    need(coordinates.ndim == 2 and coordinates.shape[1] == 3
         and np.isfinite(coordinates).all()
         and np.all(coordinates >= -1e-13) and np.all(coordinates <= 1. + 1e-13),
         "Invalid physical joint-grid query coordinates")
    axes = grid_axes(specification)
    lows, fractions = [], []
    for axis, column in zip(axes, np.clip(coordinates, 0., 1.).T):
        low = np.clip(np.searchsorted(axis, column, side="right") - 1, 0, len(axis) - 2)
        fraction = (column - axis[low]) / (axis[low + 1] - axis[low])
        need(np.all(fraction >= -1e-13) and np.all(fraction <= 1. + 1e-13),
             "Physical interpolation escaped its enclosing interval")
        lows.append(low)
        fractions.append(fraction)
    return np.stack(lows, axis=1), np.stack(fractions, axis=1)


def authenticate_grid_geometry(complete: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Require V3's full axis record to equal independently reconstructed axes."""
    need(complete.get("format_version") == 3,
         "Expected V3 belief-geometry artifact; historical cache cannot be reused")
    spec = complete.get("specification")
    need(isinstance(spec, dict), "Missing control-grid specification")
    axes = grid_axes(spec)
    axis_values = {key: value.tolist() for key, value in zip(
        ("weight0", "conditional_plus0", "conditional_plus1"), axes)}
    endpoint = spec["belief_geometry"] == "endpoint_sine"
    expected = {
        "name": spec["belief_geometry"],
        "axis_order": ["weight0", "conditional_plus0", "conditional_plus1"],
        "axis_generation": ("uniform-weight_symmetric-sine-squared-beliefs-v1"
                            if endpoint else "uniform-linspace-v1"),
        "interpolation": ("physical-interval-search-trilinear-v1"
                          if endpoint else "uniform-scaled-trilinear-v1"),
        "axes": axis_values,
        "axes_sha256": canonical_hash(axis_values),
    }
    need(complete.get("grid_geometry") == expected,
         "Control-grid geometry/physical axes differ from declared specification")
    return axes


def frozen_config(extension: Path) -> dict:
    path = extension / "config.json"
    need(path.is_file() and sha(path) == CONFIG_SHA256,
         "Frozen Bayes extension configuration changed")
    config = read_json(path)
    need(config.get("version") == 1 and config.get("theta") == THETA
         and config.get("kappas") == list(KAPPAS)
         and config.get("model_prior") == [.5, .5],
         "Wrong frozen model family")
    need(config.get("pilot_budgets") == [0, 1, 5]
         and config.get("policies") == ["bayes", "weighted_q", "no_feedback",
                                        "frozen_model", "known_parameter"],
         "Wrong frozen pilot/policy design")
    return config


def numerical_design(extension: Path, statistical: dict) -> dict:
    """Keep the frozen statistical design and its one-grid count amendment."""
    amended = read_json(extension / "numerical_config.json")
    expected = {**statistical,
                "joint_grid_levels": statistical["joint_grid_levels"] + [[49, 97, 97]]}
    need(amended == expected and (extension / "NUMERICAL_AMENDMENT.md").is_file()
         and (extension / "refine_numerical.py").is_file(),
         "Numerical-only amendment is missing or changes the statistical design")
    return amended


def expected_output_files(config: dict, replicates: int) -> set[str]:
    fixed = {"episodes.csv.gz", "traces.csv.gz", "planning_values.csv",
             "timings.csv", "pilot_records.json", "diagnostics.csv", "summary.json"}
    for model in (0, 1):
        for replicate in range(replicates):
            suffix = f"m{model}_r{replicate}.npz"
            fixed.update(f"{directory}/{suffix}" for directory in
                         ("pilots", "tapes", "actions"))
    return fixed


def verify_table_artifact(table_dir: Path, manifest: dict, config: dict,
                          numeric_config: dict, *,
                          allow_array_equivalent_build: bool = False) -> dict:
    complete = read_json(table_dir / "complete.json")
    need(set(complete) == {"format_version", "specification", "retain_q", "source",
                           "build", "numerical_contract", "grid_geometry", "metadata", "arrays",
                           "artifact_sha256"}
         and isinstance(complete.get("metadata"), dict),
         "Control artifact has missing or unexpected metadata fields")
    authenticate_grid_geometry(complete)
    expected_source_files = {
        "solver.py": manifest["inputs"]["extension"]["solver.py"],
        **{f"trade_learning.{module}": manifest["inputs"]["core"][f"src/trade_learning/{module}.py"]
           for module in ("control", "filtering", "model", "numerics")},
    }
    need(complete.get("source") == {"files": expected_source_files,
                                        "sha256": canonical_hash(expected_source_files)}
         and complete.get("build") == manifest.get("build")
         and complete.get("numerical_contract") == NUMERICAL_CONTRACT
         and complete.get("retain_q") == ["bayes", "no_feedback", "frozen_model"],
         "Control artifact source/build/precision provenance is incomplete or stale")
    signature = complete.get("artifact_sha256")
    original_signature = manifest.get("table_artifact_sha256")
    need(signature == canonical_hash({k: v for k, v in complete.items()
                                      if k != "artifact_sha256"}),
         "Control artifact manifest identity mismatch")
    need(isinstance(original_signature, str) and len(original_signature) == 64,
         "Study lacks original evaluated control-artifact identity")
    need(allow_array_equivalent_build or signature == original_signature,
         "Supplied control artifact differs from exact evaluated artifact; cold-array equivalence requires explicit opt-in")
    need(complete.get("arrays") == manifest.get("table_arrays"),
         "Control arrays differ from evaluated table identity")
    need(complete.get("specification") == manifest.get("table_specification"),
         "Evaluated control specification mismatch")
    names = set()
    total_bytes = 0
    finite_cells = 0
    legal_q_cells = 0
    for item in complete["arrays"]:
        need(isinstance(item, dict) and set(item) == {"file", "shape", "dtype", "sha256", "bytes"},
             "Malformed control array record")
        name = item["file"]
        need(isinstance(name, str) and name.endswith(".npy")
             and Path(name).name == name and name not in names,
             "Unsafe or duplicated control array path")
        names.add(name)
        path = table_dir / name
        need(path.is_file() and path.stat().st_size == item["bytes"]
             and sha(path) == item["sha256"],
             f"Missing or changed control array: {name}")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        need(list(value.shape) == item["shape"] and str(value.dtype) == item["dtype"]
             and value.dtype == np.float64,
             f"Wrong control array shape/dtype: {name}")
        is_value = name.startswith("value-")
        is_q = name.startswith("q-") or name.startswith("known-q-")
        need(is_value or is_q, f"Unexpected numerical array kind: {name}")
        for remaining in range(value.shape[0]):
            for qi, q in enumerate(range(-config["qmax"], config["qmax"] + 1)):
                slab = value[remaining, qi]
                if is_value:
                    need(np.isfinite(slab).all(),
                         f"Nonfinite value-table cell: {name}, n={remaining}, q={q}")
                    finite_cells += slab.size
                elif remaining == 0:
                    need(np.isneginf(slab).all(),
                         f"Nonterminal Q entries at horizon zero: {name}, q={q}")
                else:
                    legal = legal_actions(q, config["qmax"])
                    need(np.isfinite(slab[..., legal]).all()
                         and np.isneginf(slab[..., ~legal]).all(),
                         f"Wrong finite/illegal Q mask: {name}, n={remaining}, q={q}")
                    legal_q_cells += slab[..., legal].size
        total_bytes += item["bytes"]
    spec = complete["specification"]
    expected = {f"value-{kind}.npy" for kind in
                ("bayes", "reveal1", "reveal2", "reveal3", "no_feedback", "frozen_model")}
    expected.update(f"q-{kind}.npy" for kind in complete["retain_q"])
    expected.update({"known-q-0.npy", "known-q-1.npy"})
    need(names == expected, "Incomplete numerical control table set")
    need(spec["theta"] == THETA and spec["kappas"] == list(KAPPAS)
         and spec["horizon"] == 30 and spec["qmax"] == 2,
         "Wrong numerical table model")
    grid_size = spec["weight_points"] * spec["belief_points"] ** 2
    value_shape = [spec["horizon"] + 1, 2 * spec["qmax"] + 1, 3, grid_size]
    known_shape = [spec["horizon"] + 1, 2 * spec["qmax"] + 1, 3,
                   spec["known_belief_points"], 11]
    for item in complete["arrays"]:
        name = item["file"]
        correct_shape = (known_shape if name.startswith("known-q-") else
                         value_shape if name.startswith("value-") else value_shape + [11])
        need(item["shape"] == correct_shape,
             f"Malformed control array axes: {name}")
    need([spec["weight_points"], spec["belief_points"], spec["belief_points"]]
         in numeric_config["joint_grid_levels"],
         "Control belief grid is outside predeclared ladder")
    need(spec["quadrature_points"] in numeric_config["quadrature_levels"],
         "Control quadrature is outside predeclared ladder")
    actual = {path.name for path in table_dir.iterdir() if path.is_file()}
    need(actual == names | {"complete.json"},
         "Control directory contains missing or unrecorded files")
    return {"artifact_sha256": signature,
            "original_evaluated_artifact_sha256": original_signature,
            "supplied_artifact_sha256": signature,
            "identity_mode": ("exact_artifact" if signature == original_signature
                              else "same_build_exact_arrays"),
            "equivalent_array_option_requested": allow_array_equivalent_build,
            "arrays_checked": len(names),
            "grid_geometry_sha256": complete["grid_geometry"]["axes_sha256"],
            "array_bytes": total_bytes, "finite_value_cells": finite_cells,
            "finite_legal_q_cells": legal_q_cells, "specification": spec}


def _nonnegative(value: object, label: str) -> float:
    result = number(value, label)
    need(result >= 0, f"Negative numerical refinement diagnostic: {label}")
    return result


def _product_joint(coordinates: np.ndarray) -> np.ndarray:
    w, b0, b1 = coordinates.T
    return np.column_stack((w * (1 - b0), w * b0,
                            (1 - w) * (1 - b1), (1 - w) * b1))


def verify_numerical_probes(path: Path, config: dict) -> int:
    """Recreate the fixed probe population without importing the solver."""
    random_count = config["refinement"]["random_probes"]
    rng = np.random.default_rng(np.random.SeedSequence([
        config["root_seed"], config["seed_namespaces"]["numerical_probes"], 0, 0]))
    random_joint = rng.dirichlet(np.ones(4), size=random_count)
    qmax = config["qmax"]
    random_q = rng.integers(-qmax, qmax + 1, size=random_count)
    random_x = rng.integers(-1, 2, size=random_count)
    corners = np.stack(np.meshgrid([0., .5, 1.], [0., .5, 1.], [0., .5, 1.],
                                   indexing="ij"), axis=-1).reshape(-1, 3)
    boundary = np.tile(_product_joint(corners), (3 * (2 * qmax + 1), 1))
    boundary_q = np.repeat(np.arange(-qmax, qmax + 1), 3 * len(corners))
    boundary_x = np.tile(np.repeat(np.arange(-1, 2), len(corners)), 2 * qmax + 1)
    reflected = min(128, random_count)
    expected = {
        "joint": np.concatenate((random_joint, boundary, random_joint[:reflected])),
        "inventory": np.concatenate((random_q, boundary_q, -random_q[:reflected])),
        "signal": np.concatenate((random_x, boundary_x, -random_x[:reflected])),
        "initial_joint": _product_joint(np.column_stack((
            np.array([0., .25, .5, .75, 1.]), np.full(5, .5), np.full(5, .5)))),
    }
    with np.load(path, allow_pickle=False) as actual:
        need(set(actual.files) == set(expected), "Numerical probe arrays are incomplete")
        for key, value in expected.items():
            need(actual[key].shape == value.shape and np.array_equal(actual[key], value),
                 f"Numerical probe values/seeds changed: {key}")
    return len(expected["joint"])


def verify_numerical_acceptance(numerical_path: Path, config: dict, numeric_config: dict,
                                manifest: dict, table_dir: Path,
                                core_root: Path, extension: Path) -> dict:
    """Re-evaluate the frozen numerical gate from every exported horizon row."""
    numerical = read_json(numerical_path)
    numeric_hash = sha(extension / "numerical_config.json")
    amendment = numerical.get("protocol_config_sha256") == numeric_hash
    active_numeric_config = numeric_config if amendment else config
    geometry = numerical.get("belief_geometry")
    need(geometry in ("uniform", "endpoint_sine"),
         "Numerical receipt omits the explicit V3 grid geometry")
    if geometry == "endpoint_sine":
        need(amendment and numerical.get("geometry_amendment_sha256") ==
             sha(extension / "GEOMETRY_AMENDMENT.md")
             and numerical.get("geometry_wrapper_sha256") ==
             sha(extension / "refine_geometry.py"),
             "Endpoint numerical receipt lacks the pre-outcome geometry amendment")
    else:
        need("geometry_amendment_sha256" not in numerical
             and "geometry_wrapper_sha256" not in numerical,
             "Uniform numerical receipt claims endpoint-amendment evidence")
    need(numerical.get("protocol_config_sha256") in (CONFIG_SHA256, numeric_hash)
         and numerical.get("certified_bound") is False,
         "Numerical acceptance contradicts frozen protocol/certification scope")
    if amendment:
        need(numerical.get("numerical_config_sha256") == numeric_hash
             and numerical.get("statistical_protocol_sha256") == CONFIG_SHA256
             and numerical.get("numerical_amendment_sha256") ==
             sha(extension / "NUMERICAL_AMENDMENT.md")
             and numerical.get("numerical_wrapper_sha256") ==
             sha(extension / "refine_numerical.py"),
             "Numerical acceptance does not bind the explicit pre-evaluation amendment")
    need(numerical.get("thresholds") == config["refinement"],
         "Numerical refinement thresholds changed")
    need(numerical.get("probe_design") ==
         {"boundary_count": 405, "random_count": config["refinement"]["random_probes"],
          "reflection_count": 128},
         "Numerical probe design differs from predeclared design")
    probes_path = numerical_path.parent / "numerical_probes.npz"
    need(probes_path.is_file() and sha(probes_path) == numerical.get("probe_sha256"),
         "Numerical probe set is missing/stale")
    probe_count = verify_numerical_probes(probes_path, config)
    expected_source = {
        "solver.py": sha(extension / "solver.py"),
        "trade_learning.control": sha(core_root / "src/trade_learning/control.py"),
        "trade_learning.filtering": sha(core_root / "src/trade_learning/filtering.py"),
        "trade_learning.model": sha(core_root / "src/trade_learning/model.py"),
        "trade_learning.numerics": sha(core_root / "src/trade_learning/numerics.py"),
    }
    source = numerical.get("source")
    need(isinstance(source, dict) and source.get("files") == expected_source
         and source.get("sha256") == canonical_hash(expected_source),
         "Numerical result source is stale or incomplete")
    table_record = read_json(table_dir / "complete.json")
    authenticate_grid_geometry(table_record)
    need(table_record["grid_geometry"]["name"] == geometry,
         "Numerical receipt geometry differs from selected physical axes")
    need(numerical.get("build") == table_record.get("build") == manifest.get("build"),
         "Numerical/table/evaluation build fingerprints differ")
    selection = numerical.get("selected_resolution")
    need(selection == manifest.get("table_specification") == table_record.get("specification"),
         "Selected numerical resolution differs from evaluated table")
    need(numerical.get("selected_artifact_sha256") ==
         manifest.get("table_artifact_sha256"),
         "Numerical acceptance differs from the original evaluated artifact")

    comparisons = numerical.get("comparisons")
    need(isinstance(comparisons, list) and len(comparisons) >= 2
         and len(comparisons) % 2 == 0,
         "Numerical refinement must retain both independent axes at each stage")
    kinds = ("bayes", "reveal1", "reveal2", "reveal3", "no_feedback", "frozen_model")
    action_kinds = {"bayes", "no_feedback", "frozen_model"}
    checked_rows = 0
    grid_index = 1
    gh_index = 1
    need(numerical.get("selection_rule") ==
         "start second levels, independently compare preceding levels, advance only failing axes",
         "Numerical refinement selection rule changed")
    def expected_spec(grid: list[int], quadrature: int) -> dict:
        return {"theta": config["theta"], "kappas": config["kappas"],
                "horizon": config["horizon"], "qmax": config["qmax"],
                "weight_points": grid[0], "belief_points": grid[1],
                "quadrature_points": quadrature,
                "known_belief_points": config["known_reference"]["belief_points"],
                "known_quadrature_points": config["known_reference"]["quadrature_points"],
                "belief_geometry": geometry}
    for pair_start in range(0, len(comparisons), 2):
        need([comparisons[pair_start + i].get("axis") for i in (0, 1)] ==
             ["belief", "quadrature"], "Numerical refinement axis order changed")
        for comparison in comparisons[pair_start:pair_start + 2]:
            axis = comparison["axis"]
            coarse, fine = comparison.get("coarse_specification"), comparison.get("fine_specification")
            need(isinstance(coarse, dict) and isinstance(fine, dict),
                 "Missing numerical refinement specifications")
            grid = active_numeric_config["joint_grid_levels"]
            gh = active_numeric_config["quadrature_levels"]
            exact_fine = expected_spec(grid[grid_index], gh[gh_index])
            exact_coarse = (expected_spec(grid[grid_index - 1], gh[gh_index])
                            if axis == "belief" else
                            expected_spec(grid[grid_index], gh[gh_index - 1]))
            need(fine == exact_fine and coarse == exact_coarse,
                 "Numerical refinement skipped a level or changed an unrelated specification")
            need([fine["weight_points"], fine["belief_points"], fine["belief_points"]]
                 in active_numeric_config["joint_grid_levels"]
                 and fine["quadrature_points"] in active_numeric_config["quadrature_levels"],
                 "Numerical fine setting outside frozen ladder")
            if axis == "belief":
                need(coarse["quadrature_points"] == fine["quadrature_points"]
                     and (coarse["weight_points"], coarse["belief_points"]) !=
                     (fine["weight_points"], fine["belief_points"]),
                     "Belief refinement changed quadrature or no grid dimension")
            else:
                need(coarse["weight_points"] == fine["weight_points"]
                     and coarse["belief_points"] == fine["belief_points"]
                     and coarse["quadrature_points"] != fine["quadrature_points"],
                     "Quadrature refinement changed belief grid")
            need(comparison.get("all_horizons") == list(range(1, config["horizon"] + 1))
                 and comparison.get("probe_count") == probe_count,
                 "Numerical comparison misses horizons/probes")
            rows = comparison.get("by_horizon_policy")
            need(isinstance(rows, list) and len(rows) == config["horizon"] * len(kinds),
                 "Incomplete numerical comparison policy/horizon rows")
            maxima = {"max_initial_value_change": 0., "max_probe_value_change": 0.,
                      "max_finite_q_change": 0., "max_robust_disagreement_fraction": 0.}
            for index, row in enumerate(rows):
                need(row.get("remaining") == 1 + index // len(kinds)
                     and row.get("kind") == kinds[index % len(kinds)],
                     "Numerical refinement row order/coverage differs")
                for field, summary_field in (("initial_value_change", "max_initial_value_change"),
                                             ("probe_value_change", "max_probe_value_change")):
                    maxima[summary_field] = max(maxima[summary_field],
                                                _nonnegative(row.get(field), field))
                if row["kind"] in action_kinds:
                    maxima["max_finite_q_change"] = max(
                        maxima["max_finite_q_change"],
                        _nonnegative(row.get("finite_q_change"), "finite_q_change"))
                    fraction = _nonnegative(row.get("robust_disagreement_fraction"),
                                            "robust_disagreement_fraction")
                    maxima["max_robust_disagreement_fraction"] = max(
                        maxima["max_robust_disagreement_fraction"], fraction)
                    count = integer(row.get("robust_disagreements"), "robust disagreements")
                    size = integer(row.get("robust_probe_count"), "robust probe count")
                    need(0 <= count <= size <= 2581 and size > 0
                         and abs(fraction - count / size) < 1e-12,
                         "Numerical robust action disagreement count/fraction mismatch")
                else:
                    need("finite_q_change" not in row and "robust_disagreement_fraction" not in row,
                         "Revelation-only value row unexpectedly claims a policy Q check")
                checked_rows += 1
            reported = comparison.get("summary")
            need(isinstance(reported, dict) and set(reported) == set(maxima),
                 "Malformed numerical comparison maxima")
            for field, expected in maxima.items():
                close(number(reported[field], field), expected,
                      f"Numerical comparison maximum {field}", 1e-12)
            threshold = config["refinement"]
            passed = (maxima["max_initial_value_change"] <= threshold["initial"]
                      and maxima["max_probe_value_change"] <= threshold["probe_sup"]
                      and maxima["max_finite_q_change"] <= threshold["probe_sup"]
                      and maxima["max_robust_disagreement_fraction"] <= threshold["robust_disagreement"])
            need(comparison.get("passes_predeclared_rule") is passed,
                 "Numerical comparison pass flag disagrees with frozen thresholds")
        belief_pass = comparisons[pair_start]["passes_predeclared_rule"]
        quadrature_pass = comparisons[pair_start + 1]["passes_predeclared_rule"]
        if belief_pass and quadrature_pass:
            need(pair_start == len(comparisons) - 2,
                 "Numerical search continued after first jointly passing setting")
        else:
            grid_index += int(not belief_pass)
            gh_index += int(not quadrature_pass)
            need(grid_index < len(active_numeric_config["joint_grid_levels"])
                 and gh_index < len(active_numeric_config["quadrature_levels"]),
                 "Numerical acceptance proceeded beyond an exhausted axis")
    last_pair = comparisons[-2:]
    need(last_pair[0]["fine_specification"] == selection
         and last_pair[1]["fine_specification"] == selection,
         "Selected resolution is not the last two-axis comparison")
    known = numerical.get("known_reference")
    need(isinstance(known, dict) and known.get("passes_predeclared_rule") is True
         and known.get("source") == source and known.get("build") == numerical.get("build"),
         "Known-model scalar reference is missing, stale, or failed")
    models = known.get("models")
    need(isinstance(models, list) and len(models) == 2,
         "Known-reference model coverage is incomplete")
    known_rows = 0
    for model_index, model in enumerate(models):
        need(model.get("model_index") == model_index
             and model.get("kappa") == KAPPAS[model_index]
             and model.get("resolutions") == [[321, 161], [641, 161], [321, 321]],
             "Known-reference model or resolution ladder changed")
        known_comparisons = model.get("comparisons")
        need(isinstance(known_comparisons, list) and len(known_comparisons) == 2
             and [item.get("axis") for item in known_comparisons] == ["belief", "quadrature"],
             "Known-reference independent axes are missing")
        for comparison in known_comparisons:
            rows = comparison.get("by_horizon")
            need(isinstance(rows, list) and len(rows) == config["horizon"],
                 "Known-reference horizon coverage is incomplete")
            maxima = {field: 0. for field in
                      ("initial_value_change", "probe_value_change", "finite_q_change",
                       "robust_disagreement_fraction")}
            for index, row in enumerate(rows):
                need(row.get("remaining") == index + 1,
                     "Known-reference horizon row missing or reordered")
                for field in maxima:
                    maxima[field] = max(maxima[field], _nonnegative(row.get(field), field))
                count = integer(row.get("robust_disagreements"), "known robust disagreements")
                size = integer(row.get("robust_probe_count"), "known robust probe count")
                need(0 <= count <= size <= 2581 and size > 0
                     and abs(row["robust_disagreement_fraction"] - count / size) < 1e-12,
                     "Known-reference disagreement fraction/count mismatch")
                known_rows += 1
            reported = comparison.get("summary")
            need(isinstance(reported, dict) and set(reported) == set(maxima),
                 "Known-reference maxima are malformed")
            for field, expected in maxima.items():
                close(number(reported[field], field), expected,
                      f"Known-reference maximum {field}", 1e-12)
            threshold = config["refinement"]
            passed = (maxima["initial_value_change"] <= threshold["initial"]
                      and maxima["probe_value_change"] <= threshold["probe_sup"]
                      and maxima["finite_q_change"] <= threshold["probe_sup"]
                      and maxima["robust_disagreement_fraction"] <= threshold["robust_disagreement"])
            need(comparison.get("passes_predeclared_rule") is passed and passed,
                 "Known-reference axis fails frozen acceptance thresholds")
    need(numerical.get("passes_predeclared_rule") is True
         and all(item.get("passes_predeclared_rule") is True for item in last_pair),
         "Full numerical acceptance was declared without both passing axes")
    need(numerical.get("status") == "resolved_on_prespecified_probes",
         "Numerical acceptance has no resolved status")
    return {"passed": True, "comparisons": len(comparisons),
            "horizon_policy_rows_checked": checked_rows,
            "known_reference_rows_checked": known_rows,
            "independent_probe_states_recreated": probe_count,
            "selected_resolution": selection,
            "belief_geometry": geometry,
            "selected_grid_axes_sha256": table_record["grid_geometry"]["axes_sha256"],
            "selected_artifact_sha256": manifest["table_artifact_sha256"],
            "supplied_array_equivalent_artifact_sha256": table_record["artifact_sha256"],
            "certified_bound": False}


def verify_manifest(core_root: Path, extension: Path, study: Path,
                    config: dict, *, profile: str, tables: Path | None = None,
                    numerical_checks: Path | None = None,
                    allow_array_equivalent_build: bool = False) -> tuple[dict, dict, dict, dict]:
    need(not allow_array_equivalent_build or tables is not None,
         "Array-equivalent cold build requires a supplied control-table directory")
    manifest = read_json(study / "manifest.json")
    numeric_config = numerical_design(extension, config)
    need(manifest.get("status") == "complete" and manifest.get("profile") == profile,
         "Study is incomplete or has wrong profile")
    need(manifest.get("protocol") == config
         and manifest.get("protocol_sha256") == CONFIG_SHA256
         and manifest.get("protocol_freeze_commit") == PROTOCOL_COMMIT,
         "Study protocol does not match frozen configuration")
    replicates = config["replicates_per_model"] if profile == "full" else 2
    episodes = config["episodes_per_replicate"] if profile == "full" else 12
    need(manifest.get("replicates_per_model") == replicates
         and manifest.get("episodes_per_replicate") == episodes,
         "Study replicate/episode budgets differ from profile")
    expected_rows = 2 * replicates * episodes * len(config["pilot_budgets"]) * len(config["policies"])
    need(manifest.get("policy_records") == expected_rows
         and manifest.get("paired_market_trajectories") == 2 * replicates * episodes,
         "Study count metadata disagrees with Cartesian product")
    need(manifest.get("numerical_contract") == NUMERICAL_CONTRACT,
         "Numerical tie/precision contract changed")

    core_paths = sorted((core_root / "src/trade_learning").glob("*.py"))
    need(core_paths, "Original core source is unavailable")
    aggregate = hashlib.sha256(b"".join(path.name.encode() + path.read_bytes()
                                      for path in core_paths)).hexdigest()
    recorded = aggregate != manifest.get("core_source_hash")
    if recorded:
        need(recorded_core_hash(core_root) == CORE_SOURCE_HASH == manifest.get("core_source_hash"),
             "Recorded scientific source differs from publication migration")
    source_inputs = manifest.get("inputs")
    need(isinstance(source_inputs, dict) and set(source_inputs) == {"core", "extension"},
         "Incomplete source-input identity")
    expected_core = {str(path.relative_to(core_root)):
                     (recorded_digest(core_root, str(path.relative_to(core_root))) if recorded else sha(path))
                     for path in core_paths}
    expected_extension = {name: (recorded_digest(core_root, str((extension / name).relative_to(core_root)))
                                 if recorded else sha(extension / name)) for name in
                          ("run.py", "solver.py", "config.json", "PROTOCOL.md",
                           "NUMERICAL_AMENDMENT.md", "numerical_config.json",
                           "refine_numerical.py", "GEOMETRY_AMENDMENT.md",
                           "refine_geometry.py")}
    need(source_inputs["core"] == expected_core
         and source_inputs["extension"] == expected_extension,
         "Study source bytes differ from currently supplied source")
    files = manifest.get("files")
    expected = expected_output_files(config, replicates)
    need(isinstance(files, dict) and set(files) == expected,
         f"Study file inventory differs: missing={sorted(expected - set(files or {}))[:5]}")
    for relative, digest in files.items():
        path = (study / relative).resolve()
        need(path.is_relative_to(study.resolve()) and path.is_file()
             and isinstance(digest, str) and len(digest) == 64 and sha(path) == digest,
             f"Missing or stale study artifact: {relative}")
    actual = {str(path.relative_to(study)) for path in study.rglob("*") if path.is_file()}
    need(actual == expected | {"manifest.json"},
         "Study contains unmanifested or missing output files")
    build = manifest.get("build")
    need(isinstance(build, dict) and build.get("sha256") ==
         canonical_hash({key: value for key, value in build.items() if key != "sha256"}),
         "Numerical build fingerprint is malformed")
    table_receipt = {"checked": False}
    if tables is not None:
        table_receipt = {"checked": True, **verify_table_artifact(
            tables, manifest, config, numeric_config,
            allow_array_equivalent_build=allow_array_equivalent_build)}
    elif profile == "full":
        raise VerificationError("Full study requires the evaluated control-table directory")
    numerical_receipt = {"checked": False}
    if profile == "full":
        need(numerical_checks is not None and numerical_checks.is_file(),
             "Full study requires numerical acceptance evidence")
        need(sha(numerical_checks) == manifest.get("numerical_acceptance_sha256"),
             "Stale numerical acceptance evidence")
        numerical = read_json(numerical_checks)
        need(numerical.get("passes_predeclared_rule") is True,
             "Predeclared numerical refinement rule did not pass")
        numerical_receipt = verify_numerical_acceptance(
            numerical_checks, config, numeric_config, manifest, tables, core_root, extension)
    else:
        need(manifest.get("numerical_acceptance_sha256") is None,
             "Smoke study unexpectedly claims full numerical acceptance")
    effective = {**config, "replicates_per_model": replicates,
                 "episodes_per_replicate": episodes}
    return manifest, effective, table_receipt, numerical_receipt


def number(value: object, label: str) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise VerificationError(f"Invalid {label}: {value!r}") from exc
    need(math.isfinite(result), f"Nonfinite {label}")
    return result


def integer(value: object, label: str) -> int:
    try:
        result = int(value)
    except (ValueError, TypeError) as exc:
        raise VerificationError(f"Invalid {label}: {value!r}") from exc
    need(str(result) == str(value), f"Non-integer {label}: {value!r}")
    return result


def close(actual: float, expected: float, label: str, tol: float = REAL_TOL) -> None:
    need(math.isfinite(actual) and math.isfinite(expected)
         and abs(actual - expected) <= tol,
         f"{label}: {actual!r} versus {expected!r} (tol={tol:g})")


def legal_actions(q: int, qmax: int) -> np.ndarray:
    need(type(q) is int and -qmax <= q <= qmax, "Invalid inventory for legality")
    valid = np.ones(11, dtype=bool)
    if q == qmax:
        valid[:9] &= BID[:9] < 0
        valid[9] = False
    if q == -qmax:
        valid[:9] &= ASK[:9] < 0
        valid[10] = False
    return valid


def gh(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = roots_hermitenorm(order)
    return nodes, weights / math.sqrt(2 * math.pi)


def transition_matrix(kappas: tuple[float, float] = KAPPAS) -> np.ndarray:
    result = np.zeros((4, 4), dtype=float)
    for model, kappa in enumerate(kappas):
        i = 2 * model
        result[i:i + 2, i:i + 2] = ((1 - kappa, kappa), (kappa, 1 - kappa))
    return result


def selected_outcomes(action: int) -> list[tuple[int, int]]:
    """All possible selected-fill pairs, with -1 for each unsubmitted side."""
    need(0 <= action <= 10, "Invalid action in observation oracle")
    bid = (-1, 0, 1) if BID[action] < 0 else (0, 1)
    ask = (-1, 0, 1) if ASK[action] < 0 else (0, 1)
    bid = (-1,) if BID[action] < 0 else bid
    ask = (-1,) if ASK[action] < 0 else ask
    return [(b, a) for b in bid for a in ask]


def likelihood(x: int, z: np.ndarray, action: int, fills: tuple[int, int],
               theta: float = THETA) -> np.ndarray:
    """P(selected fills | z, model, hidden sign); columns are the four states."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    need(x in (-1, 0, 1) and np.isfinite(z).all(), "Invalid likelihood input")
    need(fills in selected_outcomes(action), "Invalid selected-fill mask")
    hidden = np.array([-1., 1., -1., 1.])
    result = np.ones((len(z), 4), dtype=float)
    for side, depth, fill in ((1, int(BID[action]), fills[0]),
                              (-1, int(ASK[action]), fills[1])):
        if depth < 0:
            continue
        p = expit(-.3 - .7 * depth - .2 * side * x)
        threshold = ndtri(p)
        standardized = (threshold - theta * side * z[:, None] * hidden[None, :]) / math.sqrt(1 - theta * theta)
        result *= ndtr(standardized if fill == 1 else -standardized)
    need(np.isfinite(result).all() and np.all((0 <= result) & (result <= 1)),
         "Nonfinite or invalid selected likelihood")
    return result


def posterior_next(joint: np.ndarray, x: int, z: float, action: int,
                   fills: tuple[int, int], kappas: tuple[float, float] = KAPPAS) -> np.ndarray:
    joint = np.asarray(joint, dtype=float)
    need(joint.shape == (4,) and np.isfinite(joint).all()
         and np.all(joint >= 0) and abs(float(joint.sum()) - 1) < 1e-12,
         "Invalid joint prior")
    weighted = joint * likelihood(x, np.array([z]), action, fills)[0]
    evidence = float(weighted.sum())
    need(evidence > 0 and math.isfinite(evidence), "Impossible selected observation")
    answer = (weighted @ transition_matrix(kappas)) / evidence
    need(np.isfinite(answer).all() and np.all(answer >= 0)
         and abs(float(answer.sum()) - 1) < 1e-12, "Invalid joint posterior")
    return answer


def increment(q: int, x: int, z: np.ndarray, action: int,
              fills: tuple[int, int]) -> tuple[np.ndarray, int]:
    """One-period change in marked wealth less pre-decision inventory penalty."""
    r = .03 * x + .30 * z
    gain = q * r - .002 * q * q
    dq = 0
    if action >= 9:
        side = 1 if action == 9 else -1
        return gain + side * r - .027, side
    for side, depth, filled in ((1, int(BID[action]), fills[0]),
                                (-1, int(ASK[action]), fills[1])):
        if depth >= 0 and filled == 1:
            gain = gain + .025 + .025 * depth - .001 + side * r
            dq += side
    return gain, dq


def one_step_state_q(q: int, x: int, qmax: int, order: int = 161) -> np.ndarray:
    """Direct Gaussian integral of each terminal Q, one column per hidden state."""
    nodes, weights = gh(order)
    result = np.full((11, 4), -np.inf)
    for action in np.flatnonzero(legal_actions(q, qmax)):
        state_values = np.zeros(4)
        for fills in selected_outcomes(int(action)):
            like = likelihood(x, nodes, int(action), fills)
            gain, dq = increment(q, x, nodes, int(action), fills)
            value = gain - TERMINAL_COST * abs(q + dq)
            state_values += np.sum(weights[:, None] * like * value[:, None], axis=0)
        result[action] = state_values
    return result


def direct_q1(q: int, x: int, joint: np.ndarray, qmax: int,
              order: int = 161) -> np.ndarray:
    result = np.full(11, -np.inf)
    valid = legal_actions(q, qmax)
    result[valid] = one_step_state_q(q, x, qmax, order)[valid] @ np.asarray(joint, dtype=float)
    return result


def direct_q2(q: int, x: int, joint: np.ndarray, qmax: int,
              outer_order: int = 81, inner_order: int = 161) -> np.ndarray:
    """Direct two-decision Bellman quadrature, with exact V1 belief geometry."""
    joint = np.asarray(joint, dtype=float)
    need(joint.shape == (4,) and np.isfinite(joint).all()
         and np.all(joint >= 0) and abs(float(joint.sum()) - 1) < 1e-12,
         "Invalid joint prior")
    nodes, weights = gh(outer_order)
    matrices = {(qq, xx): one_step_state_q(qq, xx, qmax, inner_order)
                for qq in range(-qmax, qmax + 1) for xx in (-1, 0, 1)}
    transition = transition_matrix()
    result = np.full(11, -np.inf)
    for action in np.flatnonzero(legal_actions(q, qmax)):
        total = 0.
        for fills in selected_outcomes(int(action)):
            like = likelihood(x, nodes, int(action), fills)
            weighted = like * joint[None, :]
            mass = weighted.sum(axis=1)
            need(np.all(mass > 0), "Nonpositive quadrature observation mass")
            next_joint = (weighted @ transition) / mass[:, None]
            gain, dq = increment(q, x, nodes, int(action), fills)
            continuation = np.zeros(len(nodes))
            for xnext in (-1, 0, 1):
                valid = legal_actions(q + dq, qmax)
                scores = next_joint @ matrices[q + dq, xnext][valid].T
                continuation += K[x + 1, xnext + 1] * np.max(scores, axis=1)
            total += float(np.dot(weights * mass, gain + continuation))
        result[action] = total
    return result


def math_self_consistency() -> dict:
    """Check identities that require no producer implementation or artifact."""
    max_mass_error = 0.
    max_cold_weight_drift = 0.
    max_noquote_drift = 0.
    cold = np.full(4, .25)
    for x in (-1, 0, 1):
        for action in range(11):
            for z in (-2.1, 0., 1.7):
                total = sum(likelihood(x, np.array([z]), action, fills)[0]
                            for fills in selected_outcomes(action))
                max_mass_error = max(max_mass_error, float(np.max(abs(total - 1))))
                for fills in selected_outcomes(action):
                    updated = posterior_next(cold, x, z, action, fills)
                    max_cold_weight_drift = max(max_cold_weight_drift,
                                                abs(float(updated[:2].sum()) - .5))
                    if action in (0, 9, 10):
                        forecast = cold @ transition_matrix()
                        max_noquote_drift = max(max_noquote_drift,
                                                float(np.max(abs(updated - forecast))))
    need(max_mass_error < 2e-15, "Independent selected-outcome mass error")
    need(max_cold_weight_drift < 2e-15, "Cold model weights changed after first observation")
    need(max_noquote_drift < 2e-15, "No-feedback action updated belief")
    swap = {i: next(j for j in range(9)
                    if BID[j] == ASK[i] and ASK[j] == BID[i]) for i in range(9)}
    swap.update({9: 10, 10: 9})
    asymmetric = np.array([.12, .38, .31, .19])
    direct_left = direct_q1(1, 1, asymmetric, 2)
    direct_right = direct_q1(-1, -1, asymmetric, 2)
    symmetry_error = max(abs(float(direct_left[a] - direct_right[swap[a]]))
                         for a in range(11) if np.isfinite(direct_left[a]))
    need(symmetry_error < 3e-13, "Independent one-step sign symmetry failed")
    return {"mass_error": max_mass_error,
            "cold_first_step_model_weight_drift": max_cold_weight_drift,
            "no_feedback_belief_error": max_noquote_drift,
            "one_step_sign_symmetry_error": symmetry_error,
            "scope": "Independent four-state selected likelihood and timing identities"}


def solver_math_checks(core_root: Path, extension: Path) -> dict:
    """E2E comparison of a fresh small solver with independent direct quadrature."""
    sys.path.insert(0, str(core_root / "src"))
    sys.path.insert(0, str(extension))
    from solver import SolverSpec, solve_family, update_joint

    target = np.array([.15, .35, .35, .15])
    cold = np.full(4, .25)
    singleton = np.array([.1, .9, 0., 0.])
    posterior_error = 0.
    for joint in (target, cold, singleton):
        for x in (-1, 0, 1):
            for action in (0, 1, 4, 8, 9, 10):
                for fills in selected_outcomes(action):
                    z = .43
                    expected = posterior_next(joint, x, z, action, fills)
                    got = update_joint(joint, np.array([x]),
                                       np.array([.03 * x + .30 * z]),
                                       np.array([action]), np.array([fills]))
                    posterior_error = max(posterior_error,
                                          float(np.max(abs(expected - got))))
    need(posterior_error < 2e-13, "Solver selected-feedback update differs from oracle")

    one_step_error = 0.
    two_step_errors = []
    singleton_known_error = None
    for grid in ((5, 9, 15), (9, 17, 25)):
        spec = SolverSpec(horizon=2, qmax=2,
                          weight_points=grid[0], belief_points=grid[1],
                          quadrature_points=grid[2],
                          known_belief_points=65, known_quadrature_points=41)
        with tempfile.TemporaryDirectory(prefix="bayes-math-cold-") as temporary:
            family = solve_family(spec, directory=Path(temporary), retain_q=("bayes",))
            for joint in (target, cold, singleton):
                for q, x in ((0, 0), (2, 1), (-2, -1)):
                    got = family.q_values("bayes", 1, q, x, joint)
                    expected = direct_q1(q, x, joint, 2)
                    valid = legal_actions(q, 2)
                    one_step_error = max(one_step_error,
                                         float(np.max(abs(got[valid] - expected[valid]))))
                    need(np.isneginf(got[~valid]).all(),
                         "Solver assigns finite Q to inadmissible action")
            got2 = family.q_values("bayes", 2, 0, 0, target)
            expected2 = direct_q2(0, 0, target, 2)
            valid = legal_actions(0, 2)
            two_step_errors.append(float(np.max(abs(got2[valid] - expected2[valid]))))
            need(int(np.argmax(got2)) == int(np.argmax(expected2)),
                 "Short-horizon action differs from direct quadrature")
            if grid == (9, 17, 25):
                bayes_singleton = family.q_values("bayes", 2, 0, 0, singleton)
                known_singleton = family.known_q_values(0, 2, 0, 0, singleton)
                singleton_known_error = float(np.max(abs(bayes_singleton - known_singleton)))
    need(one_step_error < 2e-12, "Solver one-step Bellman values differ from direct oracle")
    need(two_step_errors[0] < .002 and two_step_errors[1] < .0005
         and two_step_errors[1] < two_step_errors[0],
         "Short-horizon direct-quadrature error did not refine")
    need(singleton_known_error is not None and singleton_known_error < .001,
         "Zero-model-uncertainty reduction disagrees with known-model reference")
    structural = solver_structural_checks(SolverSpec, solve_family)
    return {"posterior_max_error": posterior_error,
            "one_step_max_finite_q_error": one_step_error,
            "two_step_max_finite_q_error_by_grid": two_step_errors,
            "singleton_known_model_q_error": singleton_known_error,
            "structural_reductions": structural,
            "solver_source_sha256": sha(extension / "solver.py"),
            "core_model_sha256": sha(core_root / "src/trade_learning/model.py"),
            "grids": [[5, 9, 15], [9, 17, 25]],
            "scope": "Fresh separate small-grid solver builds versus direct Gaussian enumeration; empirical approximation checks, not certified bounds"}


def solver_structural_checks(SolverSpec, solve_family) -> dict:
    """Fresh whole-family duplicate-model and zero-information reductions."""
    joints = np.array([[.12, .38, .31, .19], [.35, .05, .12, .48],
                       [.05, .15, .60, .20], [.25, .25, .25, .25]])
    duplicate_errors = []
    weighted_error = 0.
    for weight_points, belief_points, quadrature_points in ((5, 9, 15), (9, 17, 25)):
        spec = SolverSpec(kappas=(.02, .02), horizon=5,
                          weight_points=weight_points, belief_points=belief_points,
                          quadrature_points=quadrature_points,
                          known_belief_points=65, known_quadrature_points=41)
        with tempfile.TemporaryDirectory(prefix="bayes-duplicate-model-") as temporary:
            family = solve_family(spec, directory=Path(temporary))
            need(np.array_equal(family.known_tables[0], family.known_tables[1]),
                 "Identical-kappa known-model tables differ")
            family_error = 0.
            for remaining in range(1, 6):
                for q, x in ((0, 0), (2, 1), (-2, -1)):
                    for joint in joints:
                        marginal_plus = joint[1] + joint[3]
                        collapsed = np.array([1 - marginal_plus, marginal_plus, 0., 0.])
                        bayes = family.q_values("bayes", remaining, q, x, joint)
                        known = family.known_q_values(0, remaining, q, x, collapsed)
                        need(np.array_equal(np.isfinite(bayes), np.isfinite(known)),
                             "Duplicate-model reduction changed action legality")
                        finite = np.isfinite(bayes)
                        family_error = max(family_error,
                                           float(np.max(abs(bayes[finite] - known[finite]))))
                    for weight in (.2, .5, .8):
                        for hidden_plus in (.35, .65):
                            same_conditional = np.array([
                                weight * (1 - hidden_plus), weight * hidden_plus,
                                (1 - weight) * (1 - hidden_plus),
                                (1 - weight) * hidden_plus])
                            weighted = family.q_values(
                                "weighted_q", remaining, q, x, same_conditional)
                            known = family.known_q_values(
                                0, remaining, q, x, same_conditional)
                            need(np.array_equal(np.isfinite(weighted), np.isfinite(known)),
                                 "Duplicate-model weighted Q changed action legality")
                            finite = np.isfinite(weighted)
                            weighted_error = max(weighted_error,
                                                 float(np.max(abs(weighted[finite] - known[finite]))))
            duplicate_errors.append(family_error)
    need(duplicate_errors[1] < .002 and duplicate_errors[1] < duplicate_errors[0]
         and weighted_error < 2e-13,
         "Identical-kappa model reduction failed to refine toward known-model control")

    spec = SolverSpec(theta=0., horizon=5, weight_points=5, belief_points=9,
                      quadrature_points=15, known_belief_points=65,
                      known_quadrature_points=41)
    zero_value_error = 0.
    zero_q_error = 0.
    with tempfile.TemporaryDirectory(prefix="bayes-zero-theta-") as temporary:
        family = solve_family(spec, directory=Path(temporary))
        for mode in ("no_feedback", "frozen_model"):
            bayes_v, mode_v = family.value_tables["bayes"], family.value_tables[mode]
            zero_value_error = max(zero_value_error,
                                   float(np.max(abs(bayes_v - mode_v))))
            bayes_q, mode_q = family.action_tables["bayes"], family.action_tables[mode]
            need(np.array_equal(np.isfinite(bayes_q), np.isfinite(mode_q)),
                 f"Theta-zero {mode} changed admissible Q mask")
            finite = np.isfinite(bayes_q)
            zero_q_error = max(zero_q_error,
                               float(np.max(abs(bayes_q[finite] - mode_q[finite]))))
    need(zero_value_error < 2e-13 and zero_q_error < 2e-13,
         "Theta-zero belief interventions differ from Bayes recursion")
    return {"duplicate_kappa": [.02, .02], "duplicate_model_horizon": 5,
            "bayes_to_marginal_known_max_q_error_by_grid": duplicate_errors,
            "equal_conditional_weighted_q_max_error": weighted_error,
            "theta_zero_horizon": 5, "theta_zero_full_value_table_max_error": zero_value_error,
            "theta_zero_full_legal_q_table_max_error": zero_q_error,
            "scope": "Fresh whole-family structural reductions; agreement is numerical, with no exact bound claim"}


def seed_entropy(config: dict, namespace: str, model: int, replicate: int,
                 profile: str) -> list[int]:
    need(profile in ("full", "smoke"), "Unknown study profile")
    values = [config["root_seed"], config["seed_namespaces"][namespace], model, replicate]
    return values if profile == "full" else [config["root_seed"], 510, *values[1:]]


def generated_tapes(config: dict, model: int, replicate: int,
                    *, pilot: bool = False, episodes: int | None = None,
                    profile: str = "full") -> dict[str, np.ndarray]:
    """Reproduce independently spawned evaluator innovations and hidden paths."""
    n = (5 if pilot else config["episodes_per_replicate"]) if episodes is None else episodes
    horizon = config["horizon"]
    namespace = "pilot_market" if pilot else "evaluation"
    seed = np.random.SeedSequence(seed_entropy(config, namespace, model, replicate, profile))
    rx, rh, rz, rb, ra, _ = (np.random.default_rng(item) for item in seed.spawn(6))
    x = np.zeros((n, horizon + 1), dtype=np.int8)
    uniform = rx.random((n, horizon))
    cumulative = np.cumsum(K, axis=1)
    for t in range(horizon):
        x[:, t + 1] = np.sum(uniform[:, t, None] > cumulative[x[:, t] + 1], axis=1) - 1
    initial = 2 * rh.integers(0, 2, size=n, dtype=np.int8) - 1
    hidden = np.empty((n, horizon), dtype=np.int8)
    hidden[:, 0] = initial
    signs = np.where(rh.random((n, horizon - 1)) < config["kappas"][model], -1, 1)
    hidden[:, 1:] = initial[:, None] * np.cumprod(signs, axis=1)
    return {"x": x, "h": hidden, "z": rz.standard_normal((n, horizon)),
            "u_bid": rb.standard_normal((n, horizon)),
            "u_ask": ra.standard_normal((n, horizon))}


def read_tapes(config: dict, study: Path, model: int, replicate: int,
               *, pilot: bool = False, episodes: int | None = None,
               profile: str = "full") -> dict[str, np.ndarray]:
    expected = generated_tapes(config, model, replicate, pilot=pilot,
                               episodes=episodes, profile=profile)
    if pilot:
        return expected
    path = study / "tapes" / f"m{model}_r{replicate}.npz"
    need(path.is_file(), f"Missing evaluation tape: {path}")
    with np.load(path, allow_pickle=False) as archive:
        need(set(archive.files) == set(expected), f"Wrong evaluation tape keys: {path}")
        actual = {key: np.array(archive[key], copy=True) for key in expected}
    for key in expected:
        need(actual[key].shape == expected[key].shape, f"Wrong tape shape {path}/{key}")
        need(actual[key].dtype == expected[key].dtype, f"Wrong tape dtype {path}/{key}")
        need(np.array_equal(actual[key], expected[key]),
             f"Tape does not match independent seed regeneration: {path}/{key}")
        if key in ("z", "u_bid", "u_ask"):
            need(np.isfinite(actual[key]).all(), f"Nonfinite tape {path}/{key}")
    return actual


def _expected_potential(tape: dict, theta: float) -> dict[tuple[int, int], np.ndarray]:
    x, h, z = tape["x"][:, :-1], tape["h"], tape["z"]
    scale = math.sqrt(1 - theta * theta)
    result = {}
    for side, noise in ((1, tape["u_bid"]), (-1, tape["u_ask"])):
        latent = h * theta * side * z + scale * noise
        for depth in (0, 1):
            threshold = ndtri(expit(-.3 - .7 * depth - .2 * side * x))
            result[side, depth] = latent <= threshold
    return result


def replay(tape: dict, actions: np.ndarray, config: dict,
           *, trace_episodes: int = 0) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Independent action-ledger replay for every recorded episode and tick."""
    n, horizon = tape["z"].shape
    need(actions.shape == (n, horizon) and actions.dtype == np.int8,
         "Action array shape/dtype mismatch")
    need(np.all((0 <= actions) & (actions <= 10)), "Action ID outside 0..10")
    qmax = config["qmax"]
    x, z = tape["x"][:, :-1], tape["z"]
    returns = .03 * x + .30 * z
    potential = _expected_potential(tape, config["theta"])
    q = np.zeros(n, dtype=np.int16)
    cash = np.zeros(n, dtype=float)
    price = np.full(n, 1000., dtype=float)
    names = ("spread_capture", "passive_fees", "market_fees", "market_spread_cost",
             "directional_pnl", "execution_exposure_pnl", "inventory_sq_sum",
             "inventory_post_sq_sum", "inventory_abs_sum", "max_abs_inventory",
             "passive_fills", "market_fills", "directional_exposure",
             "execution_selection", "market_innovation")
    metrics = {name: np.zeros(n, dtype=float) for name in names}
    metrics["action_counts"] = np.zeros((n, 11), dtype=np.int32)
    metrics["fill_counts"] = np.zeros((n, 2), dtype=np.int32)
    trace = {name: np.empty((trace_episodes, horizon), dtype=float) for name in
             ("inventory", "signal", "price", "cash", "action", "return_",
              "bid_fill", "ask_fill", "next_inventory", "next_cash", "next_price")}
    decision_inventory = np.empty((n, horizon), dtype=np.int16)
    for t in range(horizon):
        action = actions[:, t]
        bid_depth, ask_depth = BID[action], ASK[action]
        bid_on, ask_on = bid_depth >= 0, ask_depth >= 0
        buy, sell = action == 9, action == 10
        need(not np.any((q == qmax) & (bid_on | buy)),
             f"Unsafe buy/quote at upper boundary, t={t}")
        need(not np.any((q == -qmax) & (ask_on | sell)),
             f"Unsafe sell/quote at lower boundary, t={t}")
        bid_fill = bid_on & np.where(bid_depth == 0, potential[1, 0][:, t],
                                     potential[1, 1][:, t])
        ask_fill = ask_on & np.where(ask_depth == 0, potential[-1, 0][:, t],
                                     potential[-1, 1][:, t])
        passive_count = bid_fill.astype(np.int16) + ask_fill.astype(np.int16)
        q_before = q.copy()
        decision_inventory[:, t] = q_before
        cash_before = cash.copy()
        r = returns[:, t]
        passive_dq = bid_fill.astype(np.int16) - ask_fill.astype(np.int16)
        market_dq = buy.astype(np.int16) - sell.astype(np.int16)
        dq = passive_dq + market_dq
        bid_distance = .025 + .025 * np.maximum(bid_depth, 0)
        ask_distance = .025 + .025 * np.maximum(ask_depth, 0)
        cash += bid_fill * (-price + bid_distance - .001)
        cash += ask_fill * (price + ask_distance - .001)
        cash += buy * (-price - .027) + sell * (price - .027)
        q += dq
        need(np.all(abs(q) <= qmax), f"Realized inventory bound violated at t={t}")
        metrics["spread_capture"] += bid_fill * bid_distance + ask_fill * ask_distance
        metrics["passive_fees"] += passive_count * .001
        metrics["market_fees"] += (buy + sell) * .002
        metrics["market_spread_cost"] += (buy + sell) * .025
        metrics["directional_pnl"] += q_before * r
        metrics["execution_exposure_pnl"] += dq * r
        metrics["inventory_sq_sum"] += q_before.astype(float) ** 2
        metrics["inventory_post_sq_sum"] += q.astype(float) ** 2
        metrics["inventory_abs_sum"] += abs(q_before)
        np.maximum(metrics["max_abs_inventory"], abs(q), out=metrics["max_abs_inventory"])
        metrics["passive_fills"] += passive_count
        metrics["market_fills"] += buy + sell
        metrics["directional_exposure"] += q_before * r + dq * .03 * x[:, t]
        metrics["execution_selection"] += passive_dq * .30 * z[:, t]
        metrics["market_innovation"] += market_dq * .30 * z[:, t]
        metrics["fill_counts"][:, 0] += bid_fill
        metrics["fill_counts"][:, 1] += ask_fill
        metrics["action_counts"][np.arange(n), action] += 1
        if trace_episodes:
            sl = slice(0, trace_episodes)
            for key, value in (("inventory", q_before), ("signal", x[:, t]),
                               ("price", price), ("cash", cash_before),
                               ("action", action), ("return_", r),
                               ("bid_fill", np.where(bid_on, bid_fill, -1)),
                               ("ask_fill", np.where(ask_on, ask_fill, -1)),
                               ("next_inventory", q), ("next_cash", cash),
                               ("next_price", price + r)):
                trace[key][:, t] = value[sl]
        price += r
    terminal_q = q.astype(float)
    terminal_units = abs(terminal_q)
    liquidation = terminal_units * .027
    pnl = cash + terminal_q * price - liquidation
    reconciled = (metrics["spread_capture"] - metrics["passive_fees"]
                  - metrics["market_fees"] - metrics["market_spread_cost"]
                  + metrics["directional_pnl"] + metrics["execution_exposure_pnl"]
                  - liquidation)
    metrics.update(terminal_inventory=terminal_q,
                   terminal_units=terminal_units,
                   liquidation_cost=liquidation,
                   liquidation_fees=terminal_units * .002,
                   liquidation_spread=terminal_units * .025,
                   pnl=pnl,
                   risk_penalty=.002 * metrics["inventory_sq_sum"],
                   inventory_penalty=.002 * metrics["inventory_sq_sum"],
                   objective=pnl - .002 * metrics["inventory_sq_sum"],
                   reconciled_pnl=reconciled,
                   reconciliation_error=pnl - reconciled,
                   turnover=metrics["passive_fills"] + metrics["market_fills"] + terminal_units,
                   terminal_inventory_after_liquidation=np.zeros(n))
    trace["all_inventory"] = decision_inventory
    return metrics, trace


def filter_batch(tape: dict, actions: np.ndarray, initial_joint: np.ndarray,
                 config: dict, true_model: int,
                 *, trace_episodes: int = 0, policy: str | None = None,
                 decision_inventory: np.ndarray | None = None,
                 tables: dict[str, np.ndarray] | None = None,
                 table_specification: dict | None = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Independent four-state public posterior for every evaluation episode."""
    n, horizon = actions.shape
    joint = np.broadcast_to(initial_joint, (n, 4)).copy()
    x, z = tape["x"][:, :-1], tape["z"]
    h = np.array([-1., 1., -1., 1.])
    trans = transition_matrix(tuple(config["kappas"]))
    entropy_sum = np.zeros(n)
    true_weight_sum = np.zeros(n)
    before = np.empty((trace_episodes, horizon, 4))
    after = np.empty_like(before)
    for t in range(horizon):
        if tables is not None:
            need(policy is not None and decision_inventory is not None
                 and table_specification is not None,
                 "Decision replay needs policy, inventory and table specification")
            expected_actions = independent_actions(
                policy, horizon - t, decision_inventory[:, t], x[:, t], joint,
                true_model, tables, table_specification)
            if not np.array_equal(expected_actions, actions[:, t]):
                first = int(np.flatnonzero(expected_actions != actions[:, t])[0])
                raise VerificationError(
                    f"Policy action differs from independent public-state/Q-table decision "
                    f"policy={policy} model={true_model} episode={first} t={t} "
                    f"expected={int(expected_actions[first])} recorded={int(actions[first, t])}")
        if trace_episodes:
            before[:, t] = joint[:trace_episodes]
        selected = actions[:, t]
        like = np.ones((n, 4))
        for side, depths, noise in ((1, BID[selected], tape["u_bid"][:, t]),
                                     (-1, ASK[selected], tape["u_ask"][:, t])):
            active = depths >= 0
            if not np.any(active):
                continue
            threshold = ndtri(expit(-.3 - .7 * np.maximum(depths, 0)
                                     - .2 * side * x[:, t]))
            standardized = (threshold[:, None] - config["theta"] * side *
                            z[:, t, None] * h[None, :]) / math.sqrt(1 - config["theta"]**2)
            latent = tape["h"][:, t] * config["theta"] * side * z[:, t] + \
                     math.sqrt(1 - config["theta"]**2) * noise
            observed_fill = latent <= threshold
            side_like = ndtr(np.where(observed_fill[:, None], standardized, -standardized))
            like *= np.where(active[:, None], side_like, 1.)
        updated = joint * like
        mass = updated.sum(axis=1)
        need(np.all(np.isfinite(mass)) and np.all(mass > 0),
             "Nonpositive public joint likelihood in evaluation")
        joint = (updated @ trans) / mass[:, None]
        need(np.isfinite(joint).all() and np.max(abs(joint.sum(axis=1) - 1)) < 1e-12,
             "Independent public filter lost probability mass")
        if trace_episodes:
            after[:, t] = joint[:trace_episodes]
        weights = joint.reshape(n, 2, 2).sum(axis=2)
        entropy_sum -= np.sum(np.where(weights > 0, weights * np.log(weights), 0.), axis=1)
        true_weight_sum += weights[:, true_model]
    initial_weights = initial_joint.reshape(2, 2).sum(axis=1)
    final_weights = joint.reshape(n, 2, 2).sum(axis=2)
    metrics = {"model_entropy_initial": np.full(n, -np.sum(initial_weights * np.log(initial_weights))),
               "model_entropy_final": -np.sum(np.where(final_weights > 0,
                                                        final_weights * np.log(final_weights), 0.), axis=1),
               "true_model_probability_final": final_weights[:, true_model],
               "model_entropy_period_mean": entropy_sum / horizon,
               "true_model_probability_period_mean": true_weight_sum / horizon}
    return metrics, {"before_joint": before, "after_joint": after}


def fit_pilot_model_weights(pilot: dict[str, np.ndarray], budget: int,
                            config: dict) -> np.ndarray:
    """Forward likelihood for a fixed M across reset pilot episodes."""
    need(0 <= budget <= 5, "Invalid pilot budget")
    if budget == 0:
        return np.array([.5, .5])
    model_loglik = np.zeros(2)
    horizon = config["horizon"]
    for episode in range(budget):
        for model in (0, 1):
            current = np.array([.5, .5])
            for t in range(horizon):
                action = int(pilot["actions"][episode, t])
                fills = tuple(int(v) for v in pilot["fills"][episode, t])
                r = float(pilot["return_"][episode, t])
                x = int(pilot["signal"][episode, t])
                z = (r - .03 * x) / .30
                ll = likelihood(x, np.array([z]), action, fills)[0, 2 * model:2 * model + 2]
                weighted = current * ll
                evidence = float(weighted.sum())
                need(evidence > 0 and math.isfinite(evidence), "Invalid pilot selected likelihood")
                model_loglik[model] += math.log(evidence)
                current = (weighted / evidence) @ transition_matrix()[2 * model:2 * model + 2,
                                                                         2 * model:2 * model + 2]
    shift = float(model_loglik.max())
    weights = np.exp(model_loglik - shift)
    return weights / weights.sum()


def verify_pilot(config: dict, study: Path, model: int, replicate: int,
                 *, profile: str = "full") -> dict:
    """Rebuild all five public pilots, including uniform safe behavior."""
    path = study / "pilots" / f"m{model}_r{replicate}.npz"
    need(path.is_file(), f"Missing public pilot file: {path}")
    fields = {"price", "signal", "cash", "inventory", "actions",
              "action_probability", "return_", "submitted", "fills",
              "execution_prices", "executed_quantities", "fees"}
    with np.load(path, allow_pickle=False) as archive:
        need(set(archive.files) == fields, f"Wrong public pilot fields: {path}")
        pilot = {name: np.array(archive[name], copy=True) for name in fields}
    n, horizon = 5, config["horizon"]
    state_shape = (n, horizon + 1)
    tick_shape = (n, horizon)
    for name in ("price", "signal", "cash", "inventory"):
        need(pilot[name].shape == state_shape, f"Pilot state shape {path}/{name}")
    for name in ("actions", "action_probability", "return_", "fees"):
        need(pilot[name].shape == tick_shape, f"Pilot tick shape {path}/{name}")
    for name in ("submitted", "fills", "execution_prices", "executed_quantities"):
        need(pilot[name].shape == tick_shape + (2,), f"Pilot side shape {path}/{name}")
    need(pilot["actions"].dtype == np.int8, f"Pilot action dtype {path}")
    tape = generated_tapes(config, model, replicate, pilot=True, episodes=n, profile=profile)
    metrics, trace = replay(tape, pilot["actions"], config, trace_episodes=n)
    need(np.array_equal(pilot["signal"], tape["x"]), f"Pilot signal tape mismatch {path}")
    for field, before, after in (("price", "price", "next_price"),
                                 ("cash", "cash", "next_cash"),
                                 ("inventory", "inventory", "next_inventory")):
        need(np.max(abs(pilot[field][:, :-1] - trace[before])) < REAL_TOL,
             f"Pilot predecision state mismatch {path}/{field}")
        need(np.max(abs(pilot[field][:, 1:] - trace[after])) < REAL_TOL,
             f"Pilot next state mismatch {path}/{field}")
    need(np.max(abs(pilot["return_"] - trace["return_"])) < 1e-14,
         f"Pilot return mismatch {path}")
    need(np.all(np.isfinite(pilot["action_probability"])) and
         np.all((pilot["action_probability"] > 0) & (pilot["action_probability"] <= 1)),
         f"Invalid pilot action probabilities {path}")
    rng = np.random.default_rng(np.random.SeedSequence(
        seed_entropy(config, "pilot_actions", model, replicate, profile)))
    for t in range(horizon):
        q = trace["inventory"][:, t].astype(int)
        admissible = np.stack([legal_actions(int(v), config["qmax"]) for v in q])
        count = admissible.sum(axis=1)
        rank = np.floor(rng.random(n) * count).astype(int)
        expected = np.argmax(np.cumsum(admissible, axis=1) > rank[:, None], axis=1)
        need(np.array_equal(pilot["actions"][:, t], expected),
             f"Pilot action not generated by frozen uniform stream: {path}/t={t}")
        need(np.max(abs(pilot["action_probability"][:, t] - 1 / count)) < 1e-15,
             f"Pilot behavior probability mismatch: {path}/t={t}")
    selected = np.stack((BID[pilot["actions"]] >= 0,
                         ASK[pilot["actions"]] >= 0), axis=-1)
    need(np.array_equal(pilot["submitted"], selected),
         f"Pilot selected-side mask mismatch: {path}")
    replay_fills = np.stack((trace["bid_fill"], trace["ask_fill"]), axis=-1)
    need(np.array_equal(pilot["fills"], replay_fills), f"Pilot selected fills mismatch: {path}")
    fills = pilot["fills"]
    action = pilot["actions"]
    expected_qty = np.stack((np.where(fills[:, :, 0] == 1, 1, 0) + (action == 9),
                             np.where(fills[:, :, 1] == 1, 1, 0) + (action == 10)), axis=-1)
    need(np.array_equal(pilot["executed_quantities"], expected_qty),
         f"Pilot executed quantities mismatch: {path}")
    expected_price = np.full((n, horizon, 2), np.nan)
    before_price = trace["price"]
    for side_index, side, depth in ((0, 1, BID[action]), (1, -1, ASK[action])):
        passive_fill = fills[:, :, side_index] == 1
        expected_price[:, :, side_index][passive_fill] = (before_price - side *
            (.025 + .025 * depth))[passive_fill]
    expected_price[:, :, 0][action == 9] = (before_price + .025)[action == 9]
    expected_price[:, :, 1][action == 10] = (before_price - .025)[action == 10]
    need(np.array_equal(np.isnan(pilot["execution_prices"]), np.isnan(expected_price)),
         f"Pilot execution-price missing mask mismatch: {path}")
    observed_prices = np.isfinite(expected_price)
    need(np.max(abs(pilot["execution_prices"][observed_prices]
                    - expected_price[observed_prices]), initial=0.) < REAL_TOL,
         f"Pilot execution prices mismatch: {path}")
    expected_fees = .001 * np.sum(fills == 1, axis=-1) + .002 * (action >= 9)
    need(np.max(abs(pilot["fees"] - expected_fees)) < 1e-15,
         f"Pilot execution fees mismatch: {path}")
    need(np.max(abs(metrics["reconciliation_error"])) < REAL_TOL,
         f"Pilot ledger reconciliation failed: {path}")
    weights = {budget: fit_pilot_model_weights(pilot, budget, config)
               for budget in config["pilot_budgets"]}
    return {"pilot_sha256": sha(path), "model_weights": weights,
            "episodes": n, "ticks": n * horizon}


def read_actions(config: dict, study: Path, model: int, replicate: int) -> dict[str, np.ndarray]:
    path = study / "actions" / f"m{model}_r{replicate}.npz"
    need(path.is_file(), f"Missing action bundle: {path}")
    expected = {f"b{budget}_{policy}" for budget in config["pilot_budgets"]
                for policy in config["policies"]}
    with np.load(path, allow_pickle=False) as archive:
        need(set(archive.files) == expected, f"Wrong action bundle keys: {path}")
        result = {key: np.array(archive[key], copy=True) for key in expected}
    for key, actions in result.items():
        need(actions.shape == (config["episodes_per_replicate"], config["horizon"]),
             f"Wrong action shape: {path}/{key}")
        need(actions.dtype == np.int8 and np.all((actions >= 0) & (actions <= 10)),
             f"Wrong action dtype or ID: {path}/{key}")
    reference = result["b0_known_parameter"]
    for budget in config["pilot_budgets"][1:]:
        need(np.array_equal(reference, result[f"b{budget}_known_parameter"]),
             f"Privileged known-parameter actions changed with pilot budget: {path}")
    return result


def read_decision_tables(table_dir: Path, specification: dict) -> dict[str, np.ndarray]:
    names = ("q-bayes", "q-no_feedback", "q-frozen_model", "known-q-0", "known-q-1")
    tables = {name: np.load(table_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
              for name in names}
    n, qmax = specification["horizon"], specification["qmax"]
    grid = specification["weight_points"] * specification["belief_points"]**2
    for name in names:
        belief_axis = specification["known_belief_points"] if name.startswith("known") else grid
        need(tables[name].shape == (n + 1, 2 * qmax + 1, 3, belief_axis, 11)
             and tables[name].dtype == np.float64,
             f"Wrong authenticated decision-table shape: {name}")
    return tables


def _legal_batch(q: np.ndarray, qmax: int) -> np.ndarray:
    legal = np.ones((len(q), 11), dtype=bool)
    legal[:, :9] &= ~((q[:, None] == qmax) & (BID[None, :9] >= 0))
    legal[:, :9] &= ~((q[:, None] == -qmax) & (ASK[None, :9] >= 0))
    legal[:, 9] = q < qmax
    legal[:, 10] = q > -qmax
    return legal


def _known_scores(table: np.ndarray, remaining: int, q: np.ndarray, x: np.ndarray,
                  belief: np.ndarray, qmax: int, legal: np.ndarray) -> np.ndarray:
    size = table.shape[3]
    position = np.clip(belief, 0., 1.) * (size - 1)
    lower = np.minimum(np.floor(position).astype(np.int64), size - 2)
    weight = position - lower
    start = table[remaining, q + qmax, x + 1, lower]
    end = table[remaining, q + qmax, x + 1, lower + 1]
    start = np.where(legal, start, 0.)
    end = np.where(legal, end, 0.)
    return np.where(legal, start + (end - start) * weight[:, None], -np.inf)


def _scalar_planning_value(table: np.ndarray, remaining: int, q: int, x: int,
                           joint: np.ndarray, specification: dict) -> float:
    """Independently interpolate a nodal V table at one public joint belief."""
    w = float(joint[0] + joint[1])
    other = float(joint[2] + joint[3])
    b0 = float(joint[1] / w) if w > 0 else .5
    b1 = float(joint[3] / other) if other > 0 else .5
    bp = specification["belief_points"]
    lower, frac = grid_cells(specification, np.array([[w, b0, b1]]))
    lower, frac = lower[0], frac[0]
    result = 0.
    for dw in (0, 1):
        for db0 in (0, 1):
            for db1 in (0, 1):
                column = ((lower[0] + dw) * bp + lower[1] + db0) * bp + lower[2] + db1
                coefficient = ((frac[0] if dw else 1 - frac[0])
                               * (frac[1] if db0 else 1 - frac[1])
                               * (frac[2] if db1 else 1 - frac[2]))
                result += float(coefficient * table[remaining, q + specification["qmax"], x + 1, column])
    return result


def independent_actions(kind: str, remaining: int, q: np.ndarray, x: np.ndarray,
                        joint: np.ndarray, true_model: int,
                        tables: dict[str, np.ndarray], specification: dict) -> np.ndarray:
    """Read Q arrays with separately implemented interpolation and stable ties."""
    n = len(q)
    need(q.shape == x.shape == (n,) and joint.shape == (n, 4),
         "Decision state shape mismatch")
    qmax = specification["qmax"]
    legal = _legal_batch(q, qmax)
    need(np.all(legal.any(axis=1)), "No admissible decision action")
    w = joint[:, 0] + joint[:, 1]
    other = joint[:, 2] + joint[:, 3]
    b0 = np.divide(joint[:, 1], w, out=np.full(n, .5), where=w > 0)
    b1 = np.divide(joint[:, 3], other, out=np.full(n, .5), where=other > 0)
    if kind in ("weighted_q", "known_parameter"):
        known0 = _known_scores(tables["known-q-0"], remaining, q, x, b0, qmax, legal)
        known1 = _known_scores(tables["known-q-1"], remaining, q, x, b1, qmax, legal)
        if kind == "known_parameter":
            scores = known0 if true_model == 0 else known1
        else:
            scores = np.where(legal, w[:, None] * np.where(legal, known0, 0.)
                              + other[:, None] * np.where(legal, known1, 0.), -np.inf)
    else:
        need(kind in ("bayes", "no_feedback", "frozen_model"),
             "Unknown decision policy")
        bp = specification["belief_points"]
        coords = np.stack((w, b0, b1), axis=1)
        lower, frac = grid_cells(specification, coords)
        scores = np.zeros((n, 11), dtype=float)
        table = tables[f"q-{kind}"]
        for dw in (0, 1):
            for db0 in (0, 1):
                for db1 in (0, 1):
                    column = ((lower[:, 0] + dw) * bp + lower[:, 1] + db0) * bp + lower[:, 2] + db1
                    coefficient = ((frac[:, 0] if dw else 1 - frac[:, 0])
                                   * (frac[:, 1] if db0 else 1 - frac[:, 1])
                                   * (frac[:, 2] if db1 else 1 - frac[:, 2]))
                    block = table[remaining, q + qmax, x + 1, column]
                    scores += coefficient[:, None] * np.where(legal, block, 0.)
        scores = np.where(legal, scores, -np.inf)
    need(np.isfinite(scores[legal]).all() and np.isneginf(scores[~legal]).all(),
         "Decision-table interpolation produced illegal/nonfinite scores")
    maximum = np.max(scores, axis=1)
    candidates = legal & ((maximum[:, None] - scores) <= 1e-12)
    need(np.all(candidates.any(axis=1)), "No stable-tie action candidate")
    return np.argmax(candidates, axis=1).astype(np.int8)


def _metrics_row_check(row: dict, metrics: dict, episode: int,
                       label: str, maximum_errors: dict[str, float]) -> None:
    for key, array in metrics.items():
        if key in ("action_counts", "fill_counts"):
            continue
        need(key in row, f"Missing scalar ledger column: {key}")
        observed = number(row[key], f"{label}/{key}")
        expected = float(array[episode])
        difference = abs(observed - expected)
        maximum_errors[key] = max(maximum_errors.get(key, 0.), difference)
        need(difference <= REAL_TOL, f"Independent ledger mismatch {label}/{key}: {difference:g}")
    for action in range(11):
        key = f"action_{action}"
        need(key in row, f"Missing action count column: {key}")
        observed = integer(row[key], f"{label}/{key}")
        need(observed == int(metrics["action_counts"][episode, action]),
             f"Action count disagrees with action array: {label}/{key}")
    for key, side in (("bid_fill_count", 0), ("ask_fill_count", 1)):
        if key in row:
            need(integer(row[key], f"{label}/{key}") == int(metrics["fill_counts"][episode, side]),
                 f"Side fill count disagrees with action replay: {label}/{key}")


def verify_episodes(config: dict, study: Path, *, profile: str = "full",
                    table_dir: Path | None = None,
                    table_specification: dict | None = None) -> tuple[np.ndarray, dict]:
    """Stream every outcome row and compare with an independent full action replay."""
    n = config["episodes_per_replicate"]
    rep_count = config["replicates_per_model"]
    budgets = config["pilot_budgets"]
    policies = config["policies"]
    shape = (2, rep_count, len(budgets), len(policies), n, 2)
    outcomes = np.full(shape, np.nan)
    path = study / "episodes.csv.gz"
    need(path.is_file(), f"Missing raw episode data: {path}")
    errors: dict[str, float] = {}
    pilot_hashes = []
    tape_hashes = []
    action_hashes = []
    trace_cache = {}
    count = 0
    decisions_checked = 0
    decision_tables = (read_decision_tables(table_dir, table_specification)
                       if table_dir is not None and table_specification is not None else None)
    with gzip.open(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        expected_columns = {"model_index", "replicate", "episode", "budget", "policy",
                            "objective", "pnl", "inventory_penalty", "terminal_inventory"}
        expected_columns.update(f"action_{a}" for a in range(11))
        need(reader.fieldnames is not None
             and len(reader.fieldnames) == len(set(reader.fieldnames))
             and expected_columns <= set(reader.fieldnames),
             "Missing or duplicated raw episode columns")
        for model in (0, 1):
            for replicate in range(rep_count):
                pilot = verify_pilot(config, study, model, replicate, profile=profile)
                pilot_hashes.append(pilot["pilot_sha256"])
                tape = read_tapes(config, study, model, replicate, profile=profile)
                tape_hashes.append(sha(study / "tapes" / f"m{model}_r{replicate}.npz"))
                actions = read_actions(config, study, model, replicate)
                action_hashes.append(sha(study / "actions" / f"m{model}_r{replicate}.npz"))
                for budget_index, budget in enumerate(budgets):
                    weights = pilot["model_weights"][budget]
                    initial_joint = np.repeat(weights / 2, 2)
                    for policy_index, policy in enumerate(policies):
                        key = f"b{budget}_{policy}"
                        metrics, trace = replay(tape, actions[key], config,
                                                trace_episodes=config["trace_episodes_per_replicate"])
                        filter_metrics, filter_trace = filter_batch(
                            tape, actions[key], initial_joint, config, model,
                            trace_episodes=config["trace_episodes_per_replicate"],
                            policy=policy, decision_inventory=trace["all_inventory"],
                            tables=decision_tables, table_specification=table_specification)
                        trace.pop("all_inventory")
                        if decision_tables is not None:
                            decisions_checked += n * config["horizon"]
                        trace_cache[model, replicate, budget, policy] = (
                            trace, filter_trace, initial_joint)
                        for episode in range(n):
                            row = next(reader, None)
                            need(row is not None, "Incomplete raw episode Cartesian product")
                            label = f"m{model}/r{replicate}/b{budget}/{policy}/e{episode}"
                            for name, expected in (("model_index", model),
                                                   ("replicate", replicate),
                                                   ("budget", budget), ("episode", episode)):
                                need(integer(row[name], f"{label}/{name}") == expected,
                                     f"Wrong or unpaired raw episode key: {label}/{name}")
                            need(row["policy"] == policy,
                                 f"Wrong or unpaired raw episode policy: {label}")
                            _metrics_row_check(row, metrics, episode, label, errors)
                            for field, values in filter_metrics.items():
                                need(field in row, f"Missing public posterior diagnostic: {field}")
                                observed = number(row[field], f"{label}/{field}")
                                difference = abs(observed - float(values[episode]))
                                errors[field] = max(errors.get(field, 0.), difference)
                                need(difference <= REAL_TOL,
                                     f"Public posterior diagnostic mismatch {label}/{field}: {difference:g}")
                            outcomes[model, replicate, budget_index, policy_index, episode, 0] = number(
                                row["objective"], f"{label}/objective")
                            outcomes[model, replicate, budget_index, policy_index, episode, 1] = number(
                                row["pnl"], f"{label}/pnl")
                            count += 1
        need(next(reader, None) is None, "Extra raw episode row beyond declared design")
    need(np.isfinite(outcomes).all(), "Incomplete or nonfinite outcome array")
    need(len(set(pilot_hashes)) == 2 * rep_count, "Pilot files reused across independent replicates")
    need(len(set(tape_hashes)) == 2 * rep_count, "Evaluation tapes reused across independent replicates")
    return outcomes, {"rows": count, "pilot_sha256": pilot_hashes,
                      "tape_sha256": tape_hashes, "action_sha256": action_hashes,
                      "maximum_ledger_errors": errors, "trace_cache": trace_cache,
                      "policy_decisions_recomputed": decisions_checked}


def verify_traces(config: dict, study: Path, episode_receipt: dict) -> dict:
    """Check every exported public before/after posterior and decision state."""
    path = study / "traces.csv.gz"
    need(path.is_file(), f"Missing public trace file: {path}")
    count = 0
    max_error = 0.
    integer_fields = ("inventory", "signal", "action", "bid_fill", "ask_fill",
                      "next_inventory")
    float_fields = ("price", "cash", "return_", "next_cash")
    with gzip.open(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        needed = {"model_index", "replicate", "episode", "budget", "policy", "t"}
        needed.update(integer_fields)
        needed.update(float_fields)
        needed.update(f"p{j}" for j in range(4))
        needed.update(f"next_p{j}" for j in range(4))
        needed.update(f"decision_p{j}" for j in range(4))
        need(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames))
             and needed <= set(reader.fieldnames), "Incomplete or duplicate public trace columns")
        for model in (0, 1):
            for replicate in range(config["replicates_per_model"]):
                for budget in config["pilot_budgets"]:
                    for policy in config["policies"]:
                        trace, filtered, initial_joint = episode_receipt["trace_cache"][
                            model, replicate, budget, policy]
                        for t in range(config["horizon"]):
                            for episode in range(config["trace_episodes_per_replicate"]):
                                row = next(reader, None)
                                need(row is not None, "Incomplete public trace Cartesian product")
                                label = f"m{model}/r{replicate}/b{budget}/{policy}/e{episode}/t{t}"
                                for field, expected in (("model_index", model), ("replicate", replicate),
                                                        ("budget", budget), ("episode", episode), ("t", t)):
                                    need(integer(row[field], f"trace {label}/{field}") == expected,
                                         f"Wrong public trace key: {label}/{field}")
                                need(row["policy"] == policy, f"Wrong public trace policy: {label}")
                                for field in integer_fields:
                                    need(integer(row[field], f"trace {label}/{field}") ==
                                         int(trace[field][episode, t]),
                                         f"Public trace ledger mismatch: {label}/{field}")
                                for field in float_fields:
                                    difference = abs(number(row[field], f"trace {label}/{field}")
                                                     - float(trace[field][episode, t]))
                                    max_error = max(max_error, difference)
                                    need(difference <= REAL_TOL,
                                         f"Public trace ledger mismatch: {label}/{field}")
                                prior = filtered["before_joint"][episode, t]
                                after = filtered["after_joint"][episode, t]
                                if t == 0:
                                    need(np.max(abs(prior - initial_joint)) < 1e-12,
                                         f"Evaluation H prior not reset after pilot: {label}")
                                decision = prior.copy()
                                if policy == "known_parameter":
                                    weight = float(prior[2 * model:2 * model + 2].sum())
                                    need(weight > 0, f"Zero true-model mass in privileged trace: {label}")
                                    decision[:] = 0.
                                    decision[2 * model:2 * model + 2] = (
                                        prior[2 * model:2 * model + 2] / weight)
                                for prefix, expected in (("p", prior), ("next_p", after),
                                                         ("decision_p", decision)):
                                    actual = np.array([number(row[f"{prefix}{j}"],
                                                              f"trace {label}/{prefix}{j}") for j in range(4)])
                                    need(np.all(actual >= 0) and abs(float(actual.sum()) - 1) < 1e-11,
                                         f"Invalid trace posterior simplex: {label}/{prefix}")
                                    error = float(np.max(abs(actual - expected)))
                                    max_error = max(max_error, error)
                                    need(error <= 2e-10,
                                         f"Trace posterior/decision mismatch {label}/{prefix}: {error:g}")
                                count += 1
        need(next(reader, None) is None, "Extra public trace row beyond declared design")
    expected_count = (2 * config["replicates_per_model"] * len(config["pilot_budgets"])
                      * len(config["policies"]) * config["trace_episodes_per_replicate"]
                      * config["horizon"])
    need(count == expected_count, "Wrong public trace row budget")
    return {"rows": count, "maximum_trace_error": max_error,
            "scope": "Independent action/ledger/filter/privileged-decision replay for every saved trace"}


def paired_welch(outcomes: np.ndarray, config: dict, budget: int,
                 policy: str, baseline: str, metric: str, alpha: float,
                 family_size: int = 1) -> dict:
    """Equal-model mixture of paired replicate means with Welch pilot variance."""
    bi = config["pilot_budgets"].index(budget)
    pi = config["policies"].index(policy)
    qi = config["policies"].index(baseline)
    mi = config["metrics"].index(metric)
    differences = outcomes[:, :, bi, pi, :, mi] - outcomes[:, :, bi, qi, :, mi]
    need(np.isfinite(differences).all(), "Nonfinite paired episode difference")
    replicate_means = differences.mean(axis=2)
    r = config["replicates_per_model"]
    need(r >= 2 and replicate_means.shape == (2, r), "Incomplete independent replicates")
    model_means = replicate_means.mean(axis=1)
    sample_variances = replicate_means.var(axis=1, ddof=1)
    components = sample_variances / (4 * r)
    variance = float(components.sum())
    mean = float(model_means.mean())
    se = math.sqrt(variance)
    denominator = float(np.sum(components**2 / (r - 1)))
    df = variance * variance / denominator if denominator > 0 else float(2 * r - 2)
    critical = float(student_t.ppf(1 - alpha / 2, df))
    adjusted = float(student_t.ppf(1 - alpha / (2 * family_size), df))
    result = {"budget": budget, "policy": policy, "baseline": baseline,
              "metric": metric, "mean": mean, "se": se, "df": df,
              "low": mean - critical * se, "high": mean + critical * se,
              "simultaneous_low": mean - adjusted * se,
              "simultaneous_high": mean + adjusted * se,
              "stratum_means": model_means.tolist(),
              "replicate_mean_sample_variances": sample_variances.tolist(),
              "independent_replicates_per_model": r,
              "episodes_per_replicate": config["episodes_per_replicate"],
              "alpha": alpha, "zero_empirical_variance": bool(variance == 0)}
    need(all(math.isfinite(result[key]) for key in ("mean", "se", "df", "low", "high")),
         "Nonfinite paired Welch result")
    return result


def recalculate_statistics(outcomes: np.ndarray, config: dict) -> dict:
    primary = config["primary"]
    primary_result = paired_welch(outcomes, config, primary["budget"],
                                  primary["policy"], primary["baseline"],
                                  primary["metric"], primary["alpha"])
    primary_result["economic_threshold"] = primary["economic_threshold"]
    primary_result["threshold_passed"] = bool(
        primary_result["low"] > primary["economic_threshold"])
    family = []
    for budget in config["pilot_budgets"]:
        for metric in config["metrics"]:
            for policy, baseline in config["contrasts"]:
                family.append(paired_welch(outcomes, config, budget, policy, baseline,
                                           metric, primary["alpha"],
                                           config["supplementary_family_size"]))
    need(len(family) == config["supplementary_family_size"],
         "Incomplete supplementary contrast family")
    policy_means = []
    for budget_index, budget in enumerate(config["pilot_budgets"]):
        for metric_index, metric in enumerate(config["metrics"]):
            for policy_index, policy in enumerate(config["policies"]):
                raw = outcomes[:, :, budget_index, policy_index, :, metric_index]
                replicate_means = raw.mean(axis=2)
                r = config["replicates_per_model"]
                means = replicate_means.mean(axis=1)
                sample_variances = replicate_means.var(axis=1, ddof=1)
                components = sample_variances / (4 * r)
                variance = float(components.sum())
                denominator = float(np.sum(components**2 / (r - 1)))
                df = variance * variance / denominator if denominator > 0 else float(2 * r - 2)
                se = math.sqrt(variance)
                mean = float(means.mean())
                critical = float(student_t.ppf(1 - primary["alpha"] / 2, df))
                policy_means.append({"budget": budget, "policy": policy, "metric": metric,
                                     "mean": mean, "se": se, "df": df,
                                     "low": mean - critical * se, "high": mean + critical * se,
                                     "simultaneous_low": mean - critical * se,
                                     "simultaneous_high": mean + critical * se,
                                     "stratum_means": means.tolist(),
                                     "independent_replicates_per_model": r,
                                     "zero_empirical_variance": bool(variance == 0)})
    return {"primary": primary_result, "family": family,
            "policy_means": policy_means,
            "scope": "Paired episode deltas averaged within independent pilot/evaluation replicates; equal true-model mixture; empirical Student/Welch approximation"}


def _compare_estimate(exported: dict, recomputed: dict, label: str) -> None:
    keys = ("budget", "policy", "metric", "mean", "se", "df", "low", "high",
            "simultaneous_low", "simultaneous_high", "stratum_means",
            "independent_replicates_per_model", "zero_empirical_variance")
    if "baseline" in recomputed:
        keys += ("baseline",)
    for key in keys:
        need(key in exported, f"Missing exported inference field: {label}/{key}")
        expected = recomputed[key]
        actual = exported[key]
        if isinstance(expected, bool) or isinstance(expected, str) or isinstance(expected, int):
            need(actual == expected and type(actual) is type(expected),
                 f"Wrong inference identity: {label}/{key}")
        elif isinstance(expected, list):
            need(isinstance(actual, list) and len(actual) == len(expected),
                 f"Wrong inference stratum shape: {label}/{key}")
            for index, value in enumerate(expected):
                close(number(actual[index], f"{label}/{key}/{index}"), value,
                      f"Inference mismatch {label}/{key}/{index}", 2e-9)
        else:
            close(number(actual, f"{label}/{key}"), expected,
                  f"Inference mismatch {label}/{key}", 2e-9)


def verify_summary(config: dict, study: Path, outcomes: np.ndarray) -> dict:
    summary = read_json(study / "summary.json")
    expected_rows = int(np.prod(outcomes.shape[:-1]))
    need(summary.get("rows") == expected_rows, "Summary row count differs from complete raw design")
    paired = 2 * config["replicates_per_model"] * config["episodes_per_replicate"]
    need(summary.get("paired_market_trajectories") == paired,
         "Summary paired-market count differs")
    need(summary.get("independent_pilot_replicates") == 2 * config["replicates_per_model"],
         "Summary independent-pilot count differs")
    need(summary.get("supplementary_family_size") == config["supplementary_family_size"],
         "Wrong summary multiplicity family")
    recomputed = recalculate_statistics(outcomes, config)
    contrasts = summary.get("contrasts")
    means = summary.get("policy_means")
    need(isinstance(contrasts, list) and len(contrasts) == len(recomputed["family"]),
         "Incomplete summary contrast family")
    need(isinstance(means, list) and len(means) == len(recomputed["policy_means"]),
         "Incomplete summary policy means")
    for index, (record, expected) in enumerate(zip(contrasts, recomputed["family"])):
        need(isinstance(record, dict), f"Malformed contrast record {index}")
        _compare_estimate(record, expected, f"contrast/{index}")
    for index, (record, expected) in enumerate(zip(means, recomputed["policy_means"])):
        need(isinstance(record, dict), f"Malformed policy mean record {index}")
        _compare_estimate(record, expected, f"policy_mean/{index}")
    primary = summary.get("primary")
    need(isinstance(primary, dict), "Missing primary comparison")
    reference = next(record for record in recomputed["family"]
                     if all(record[key] == config["primary"][key]
                            for key in ("budget", "policy", "baseline", "metric")))
    _compare_estimate(primary, reference, "primary")
    close(number(summary.get("economic_threshold"), "summary threshold"),
          config["primary"]["economic_threshold"], "Summary economic threshold", 0.)
    need(summary.get("primary_empirical_lower_exceeds_threshold") is
         (reference["low"] > config["primary"]["economic_threshold"]),
         "Primary threshold decision does not follow paired interval")
    return {"contrasts_checked": len(contrasts), "policy_means_checked": len(means),
            "primary": reference,
            "primary_threshold_passed": reference["low"] > config["primary"]["economic_threshold"],
            "uncertainty_scope": "Empirical paired pilot-aware Welch intervals; no numerical/global/evolving-market guarantee"}


def verify_pilot_records(config: dict, study: Path, *, profile: str) -> dict:
    report = read_json(study / "pilot_records.json")
    need(report.get("profile") == profile and isinstance(report.get("records"), list),
         "Invalid pilot record profile/list")
    expected_count = 2 * config["replicates_per_model"]
    need(len(report["records"]) == expected_count, "Incomplete pilot record budget")
    max_weight_error = 0.
    for index, row in enumerate(report["records"]):
        model, replicate = divmod(index, config["replicates_per_model"])
        need(isinstance(row, dict) and row.get("model_index") == model
             and row.get("replicate") == replicate,
             "Wrong pilot record order or identity")
        for field, namespace in (("pilot_seed", "pilot_market"),
                                 ("pilot_policy_seed", "pilot_actions"),
                                 ("evaluation_seed", "evaluation")):
            need(row.get(field) == seed_entropy(config, namespace, model, replicate, profile),
                 f"Wrong pilot/evaluation stream identity: {model}/{replicate}/{field}")
        pilot = verify_pilot(config, study, model, replicate, profile=profile)
        weights = row.get("model_weights")
        need(isinstance(weights, dict) and set(weights) ==
             {str(b) for b in config["pilot_budgets"]},
             "Incomplete pilot budget weights")
        for budget in config["pilot_budgets"]:
            actual = np.asarray(weights[str(budget)], dtype=float)
            expected = pilot["model_weights"][budget]
            need(actual.shape == (2,) and np.isfinite(actual).all()
                 and np.all(actual >= 0) and abs(float(actual.sum()) - 1) < 1e-12,
                 "Invalid recorded pilot model weights")
            error = float(np.max(abs(actual - expected)))
            max_weight_error = max(max_weight_error, error)
            need(error < 2e-12, "Pilot model fit differs from independent forward likelihood")
    return {"pilot_records": expected_count, "maximum_model_weight_error": max_weight_error,
            "scope": "Independent regeneration of behavior/market seeds and nested 0/1/5 pilot posterior"}


def _csv_records(path: Path, required: set[str]) -> list[dict]:
    need(path.is_file(), f"Missing CSV evidence: {path}")
    with path.open("rt", newline="") as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames))
             and required <= set(reader.fieldnames), f"Missing/duplicated CSV fields: {path}")
        return list(reader)


def verify_secondary_tables(config: dict, study: Path,
                            outcomes: np.ndarray, *, table_dir: Path | None = None,
                            table_specification: dict | None = None) -> dict:
    rows = _csv_records(study / "planning_values.csv",
                        {"model_index", "replicate", "budget", "weight_model0",
                         "initial_model_entropy", "revelation_0", "revelation_1_direct",
                         "known_true_model", "bayes", "reveal1", "reveal2",
                         "reveal3", "no_feedback", "frozen_model"})
    expected_plans = 2 * config["replicates_per_model"] * len(config["pilot_budgets"])
    need(len(rows) == expected_plans, "Incomplete planning-value rows")
    max_reveal1_interp_gap = 0.
    max_numerical_hierarchy_violation = 0.
    max_planning_table_error = 0.
    planning_tables = None
    if table_dir is not None:
        need(table_specification is not None, "Missing authenticated planning-table specification")
        kinds = ("bayes", "reveal1", "reveal2", "reveal3", "no_feedback", "frozen_model")
        planning_tables = {kind: np.load(table_dir / f"value-{kind}.npy",
                                         mmap_mode="r", allow_pickle=False)
                           for kind in kinds}
        planning_tables.update({f"known-q-{model}": np.load(
            table_dir / f"known-q-{model}.npy", mmap_mode="r", allow_pickle=False)
            for model in (0, 1)})
        spec = table_specification
        expected_shape = (spec["horizon"] + 1, 2 * spec["qmax"] + 1, 3,
                          spec["weight_points"] * spec["belief_points"] ** 2)
        need(all(planning_tables[kind].shape == expected_shape for kind in kinds),
             "Authenticated planning value-table shape mismatch")
    for index, row in enumerate(rows):
        model = index // (config["replicates_per_model"] * len(config["pilot_budgets"]))
        replicate = (index // len(config["pilot_budgets"])) % config["replicates_per_model"]
        budget = config["pilot_budgets"][index % len(config["pilot_budgets"])]
        label = f"planning/m{model}/r{replicate}/b{budget}"
        for field, expected in (("model_index", model), ("replicate", replicate),
                                ("budget", budget)):
            need(integer(row[field], f"{label}/{field}") == expected,
                 f"Wrong planning-value key: {label}/{field}")
        pilot = verify_pilot(config, study, model, replicate,
                             profile="full" if config["replicates_per_model"] == 30 else "smoke")
        weights = pilot["model_weights"][budget]
        close(number(row["weight_model0"], f"{label}/weight_model0"),
              float(weights[0]), f"Planning model weight {label}", 2e-12)
        entropy = float(-np.sum(weights * np.log(weights)))
        close(number(row["initial_model_entropy"], f"{label}/initial_model_entropy"),
              entropy, f"Planning initial entropy {label}", 2e-12)
        values = {field: number(row[field], f"{label}/{field}") for field in
                  ("revelation_0", "revelation_1_direct", "known_true_model",
                   "bayes", "reveal1", "reveal2", "reveal3", "no_feedback", "frozen_model")}
        if planning_tables is not None:
            joint = np.repeat(weights / 2, 2)
            legal = _legal_batch(np.array([0], dtype=np.int64), spec["qmax"])
            q = np.array([0], dtype=np.int64)
            x = np.array([0], dtype=np.int64)
            k0 = _known_scores(planning_tables["known-q-0"], spec["horizon"], q, x,
                               np.array([.5]), spec["qmax"], legal)[0]
            k1 = _known_scores(planning_tables["known-q-1"], spec["horizon"], q, x,
                               np.array([.5]), spec["qmax"], legal)[0]
            mix = np.where(legal[0], weights[0] * np.where(legal[0], k0, 0.)
                           + weights[1] * np.where(legal[0], k1, 0.), -np.inf)
            expected_values = {
                "revelation_0": float(weights[0] * np.max(k0) + weights[1] * np.max(k1)),
                "revelation_1_direct": float(np.max(mix)),
                "known_true_model": float(np.max(k0 if model == 0 else k1)),
            }
            expected_values.update({kind: _scalar_planning_value(
                planning_tables[kind], spec["horizon"], 0, 0, joint, spec)
                for kind in kinds})
            for field, expected_value in expected_values.items():
                difference = abs(values[field] - expected_value)
                max_planning_table_error = max(max_planning_table_error, difference)
                close(values[field], expected_value,
                      f"Independent planning-table value {label}/{field}", 1e-10)
        need(values["revelation_0"] + 1e-9 >= values["revelation_1_direct"],
             f"Known-parameter-entry value below weighted-Q relaxation: {label}")
        max_reveal1_interp_gap = max(max_reveal1_interp_gap,
                                     abs(values["reveal1"] - values["revelation_1_direct"]))
        chain = [values[key] for key in
                 ("revelation_0", "revelation_1_direct", "reveal2", "reveal3", "bayes")]
        max_numerical_hierarchy_violation = max(
            max_numerical_hierarchy_violation,
            *(max(chain[j + 1] - chain[j], 0.) for j in range(len(chain) - 1)))
    timing = _csv_records(study / "timings.csv",
                          {"model_index", "replicate", "budget", "policy",
                           "decisions", "choice_seconds"})
    expected_timing = expected_plans * len(config["policies"])
    need(len(timing) == expected_timing, "Incomplete policy compute timing rows")
    for index, row in enumerate(timing):
        model = index // (config["replicates_per_model"] * len(config["pilot_budgets"])
                          * len(config["policies"]))
        replicate = (index // (len(config["pilot_budgets"]) * len(config["policies"]))) % config["replicates_per_model"]
        budget = config["pilot_budgets"][(index // len(config["policies"])) % len(config["pilot_budgets"])]
        policy = config["policies"][index % len(config["policies"])]
        need(integer(row["model_index"], "timing model") == model
             and integer(row["replicate"], "timing replicate") == replicate
             and integer(row["budget"], "timing budget") == budget
             and row["policy"] == policy,
             "Wrong compute-timing key/order")
        need(integer(row["decisions"], "timing decisions") ==
             config["episodes_per_replicate"] * config["horizon"],
             "Wrong compute-timing decision count")
        need(number(row["choice_seconds"], "choice_seconds") >= 0,
             "Negative compute time")
    diagnostics = _csv_records(study / "diagnostics.csv",
                               {"model_index", "budget", "policy", "objective", "pnl"})
    expected_diagnostics = 2 * len(config["pilot_budgets"]) * len(config["policies"])
    need(len(diagnostics) == expected_diagnostics, "Incomplete diagnostics grouping")
    indexed = {}
    for row in diagnostics:
        key = (integer(row["model_index"], "diagnostic model"),
               integer(row["budget"], "diagnostic budget"), row["policy"])
        need(key not in indexed, "Duplicate diagnostic grouping key")
        indexed[key] = row
    for budget_index, budget in enumerate(config["pilot_budgets"]):
        for policy_index, policy in enumerate(config["policies"]):
            for model in (0, 1):
                need((model, budget, policy) in indexed,
                     "Missing diagnostic grouping key")
                row = indexed[model, budget, policy]
                for metric_index, metric in enumerate(config["metrics"]):
                    expected = float(outcomes[model, :, budget_index, policy_index, :, metric_index].mean())
                    close(number(row[metric], f"diagnostic/{metric}"), expected,
                          f"Diagnostic mean mismatch {budget}/{policy}/{model}/{metric}", 2e-9)
    return {"planning_rows": len(rows), "timing_rows": len(timing),
            "diagnostic_rows": len(diagnostics),
            "planning_table_rows_replayed": len(rows) if planning_tables is not None else 0,
            "max_planning_table_error": max_planning_table_error,
            "max_reveal1_interpolation_gap": max_reveal1_interp_gap,
            "max_numerical_hierarchy_violation": max_numerical_hierarchy_violation,
            "numerical_scope": "Planning and revelation values replayed from authenticated tables when supplied; exact hierarchy reversals are reported, not claimed to be certified bounds"}


def verify_explanation_candidates(core_root: Path, extension: Path,
                                   candidate_dir: Path, config: dict) -> dict:
    """Replay every reached explanatory prefix from public selected feedback."""
    candidate_dir = candidate_dir.resolve()
    manifest = read_json(candidate_dir / "manifest.json")
    signature = manifest.get("artifact_sha256")
    need(signature == canonical_hash({k: v for k, v in manifest.items()
                                      if k != "artifact_sha256"}),
         "Explanatory candidate manifest signature mismatch")
    need(manifest.get("format_version") == 1
         and manifest.get("kind") == "explanatory_candidates"
         and manifest.get("design_version") == 1
         and manifest.get("config_sha256") == CONFIG_SHA256
         and manifest.get("protocol_commit") == PROTOCOL_COMMIT
         and manifest.get("numerical_contract") == NUMERICAL_CONTRACT,
         "Explanatory candidate design/protocol changed")
    expected_sources = {name: sha(path) for name, path in {
        "explain.py": extension / "explain.py",
        "explain_failure_modes.md": extension / "explain_failure_modes.md",
        "THEORY.md": extension / "THEORY.md",
        "solver.py": extension / "solver.py",
        **{f"trade_learning/{module}.py": core_root / "src/trade_learning" / f"{module}.py"
           for module in ("model", "filtering", "environment", "numerics", "control")},
    }.items()}
    need(manifest.get("source") == {"files": expected_sources,
                                    "sha256": canonical_hash(expected_sources)},
         "Explanatory candidate source identity is stale")
    build = manifest.get("build")
    need(isinstance(build, dict) and build.get("sha256") ==
         canonical_hash({k: v for k, v in build.items() if k != "sha256"}),
         "Explanatory candidate build fingerprint is malformed")
    seeds = [{"model_index_evaluator_only": model,
              "market": [config["root_seed"], 410, 0, model],
              "action": [config["root_seed"], 410, 1, model]}
             for model in (0, 1)]
    need(manifest.get("seeds") == seeds and manifest.get("episodes_per_model") == 1000
         and manifest.get("reached_states") == 60000
         and manifest.get("probe_count") == 2581
         and manifest.get("arbitrary_states") == 77430
         and manifest.get("total_states") == 137430,
         "Explanatory candidate seeds/counts changed")
    files = manifest.get("files")
    expected_names = {"exploration_public.npz", "candidate_states.npz"}
    need(isinstance(files, list) and len(files) == 2
         and {item.get("file") for item in files} == expected_names,
         "Explanatory candidate file inventory is incomplete")
    need({p.name for p in candidate_dir.iterdir() if p.is_file()} ==
         expected_names | {"manifest.json"},
         "Explanatory candidate directory has unrecorded files")
    archive_arrays = {}
    for record in files:
        path = candidate_dir / record["file"]
        need(path.is_file() and path.stat().st_size == record.get("bytes")
             and sha(path) == record.get("sha256"),
             f"Explanatory candidate file changed: {path.name}")
        with np.load(path, allow_pickle=False) as archive:
            need(set(archive.files) == set(record.get("arrays", {})),
                 f"Explanatory candidate arrays missing/duplicated: {path.name}")
            for name in archive.files:
                array = archive[name]
                need(record["arrays"][name] ==
                     {"shape": list(array.shape), "dtype": str(array.dtype)},
                     f"Explanatory candidate shape/dtype changed: {path.name}/{name}")
            archive_arrays[path.name] = {name: np.array(archive[name], copy=True)
                                         for name in archive.files}
    public = archive_arrays["exploration_public.npz"]
    states = archive_arrays["candidate_states.npz"]
    need(set(public) == {"signal", "return_", "actions", "fills", "inventory", "joint"}
         and set(states) == {"candidate_id", "source", "remaining", "inventory",
                             "signal", "joint", "model_index", "episode", "t", "probe_index"},
         "Explanatory public/candidate schema changed")
    need(public["signal"].shape == (2, 1000, 31)
         and public["return_"].shape == (2, 1000, 30)
         and public["actions"].shape == (2, 1000, 30)
         and public["fills"].shape == (2, 1000, 30, 2)
         and public["inventory"].shape == (2, 1000, 30)
         and public["joint"].shape == (2, 1000, 30, 4),
         "Explanatory public-prefix dimensions changed")
    count = 137430
    need(all(array.shape == ((count, 4) if key == "joint" else (count,))
             for key, array in states.items()),
         "Explanatory candidate dimensions do not align")
    need(np.array_equal(states["candidate_id"], np.arange(count))
         and np.array_equal(states["source"], np.r_[np.zeros(60000, dtype=np.int8),
                                                   np.ones(77430, dtype=np.int8)]),
         "Explanatory candidate IDs or source order changed")
    need(np.isfinite(states["joint"]).all() and np.all(states["joint"] >= 0)
         and np.max(abs(states["joint"].sum(axis=1) - 1)) < 1e-12
         and np.isin(states["signal"], (-1, 0, 1)).all()
         and np.all(abs(states["inventory"]) <= config["qmax"]),
         "Explanatory candidate beliefs or public states invalid")
    need(np.array_equal(states["remaining"][:60000],
                        np.tile(np.arange(30, 0, -1), 2000))
         and np.array_equal(states["model_index"][:60000],
                            np.repeat(np.arange(2), 30000))
         and np.array_equal(states["episode"][:60000],
                            np.tile(np.repeat(np.arange(1000), 30), 2))
         and np.array_equal(states["t"][:60000], np.tile(np.arange(30), 2000))
         and np.all(states["probe_index"][:60000] == -1),
         "Reached explanatory provenance or horizon is false")
    for key in ("inventory", "joint"):
        need(np.array_equal(states[key][:60000], public[key].reshape(-1, *public[key].shape[3:])),
             f"Reached candidate {key} differs from saved public prefix")
    need(np.array_equal(states["signal"][:60000], public["signal"][:, :, :-1].reshape(-1)),
         "Reached candidate signal differs from saved public prefix")
    need(np.all(states["model_index"][60000:] == -1)
         and np.all(states["episode"][60000:] == -1)
         and np.all(states["t"][60000:] == -1)
         and np.array_equal(states["remaining"][60000:], np.repeat(np.arange(1, 31), 2581))
         and np.array_equal(states["probe_index"][60000:], np.tile(np.arange(2581), 30)),
         "Arbitrary numerical probes claim false episode reachability")
    max_posterior_error = 0.
    for model in (0, 1):
        q = np.zeros(1000, dtype=np.int64)
        joint = np.full((1000, 4), .25)
        need(np.all(public["signal"][model, :, 0] == 0),
             "Explanatory cold public signal did not reset")
        for t in range(30):
            observed_q = public["inventory"][model, :, t]
            observed_joint = public["joint"][model, :, t]
            need(np.array_equal(observed_q, q),
                 f"Explanatory inventory path disagrees at model={model}, t={t}")
            error = float(np.max(abs(observed_joint - joint)))
            max_posterior_error = max(max_posterior_error, error)
            need(error < 4e-13,
                 f"Explanatory public posterior differs from independent filter: model={model}, t={t}")
            action = public["actions"][model, :, t]
            x = public["signal"][model, :, t]
            fills = public["fills"][model, :, t]
            returns = public["return_"][model, :, t]
            need(np.isin(x, (-1, 0, 1)).all() and np.isfinite(returns).all()
                 and np.all((action >= 0) & (action <= 10)),
                 "Explanatory public action, signal or return invalid")
            legal = _legal_batch(q, config["qmax"])
            need(np.all(legal[np.arange(1000), action]),
                 "Explanatory uniform exploration selected an unsafe action")
            active_bid = BID[action] >= 0
            active_ask = ASK[action] >= 0
            need(np.all(np.where(active_bid, np.isin(fills[:, 0], (0, 1)), fills[:, 0] == -1))
                 and np.all(np.where(active_ask, np.isin(fills[:, 1], (0, 1)), fills[:, 1] == -1)),
                 "Explanatory selected-fill/missing-side markers are invalid")
            z = (returns - .03 * x) / .30
            hidden = np.array([-1., 1., -1., 1.])
            like = np.ones((1000, 4))
            for side, depth, active, fill in ((1, BID[action], active_bid, fills[:, 0]),
                                             (-1, ASK[action], active_ask, fills[:, 1])):
                threshold = ndtri(expit(-.3 - .7 * np.maximum(depth, 0) - .2 * side * x))
                standardized = (threshold[:, None] - THETA * side * z[:, None] * hidden[None, :]) / math.sqrt(1 - THETA**2)
                side_like = ndtr(np.where(fill[:, None] == 1, standardized, -standardized))
                like *= np.where(active[:, None], side_like, 1.)
            weighted = joint * like
            evidence = weighted.sum(axis=1)
            need(np.isfinite(evidence).all() and np.all(evidence > 0),
                 "Explanatory public likelihood is nonpositive")
            joint = (weighted @ transition_matrix()) / evidence[:, None]
            q += np.where(action == 9, 1, np.where(action == 10, -1, 0))
            q += (active_bid & (fills[:, 0] == 1)).astype(int)
            q -= (active_ask & (fills[:, 1] == 1)).astype(int)
            need(np.all(abs(q) <= config["qmax"]),
                 "Explanatory inventory escaped the admissible region")
    return {"checked": True, "artifact_sha256": signature,
            "public_decisions_replayed": 60000, "candidate_states_checked": count,
            "max_public_posterior_error": max_posterior_error,
            "scope": "Every reached pre-decision belief and inventory independently replayed from saved selected public feedback"}


def verify(core_root: Path, extension: Path, study: Path, *, profile: str,
           tables: Path | None = None, numerical_checks: Path | None = None,
           original_zip: Path | None = None, check_solver: bool = False,
           candidates: Path | None = None,
           allow_array_equivalent_build: bool = False) -> dict:
    core_root, extension, study = (path.resolve() for path in
                                   (core_root, extension, study))
    need(core_root.is_dir() and extension.is_dir() and study.is_dir(),
         "Core/extension/study directory missing")
    config = frozen_config(extension)
    manifest, effective, table_receipt, numerical_receipt = verify_manifest(
        core_root, extension, study, config, profile=profile,
        tables=tables.resolve() if tables else None,
        numerical_checks=numerical_checks.resolve() if numerical_checks else None,
        allow_array_equivalent_build=allow_array_equivalent_build)
    outcomes, episode_receipt = verify_episodes(
        effective, study, profile=profile,
        table_dir=tables.resolve() if tables else None,
        table_specification=manifest["table_specification"] if tables else None)
    traces = verify_traces(effective, study, episode_receipt)
    summary = verify_summary(effective, study, outcomes)
    pilots = verify_pilot_records(effective, study, profile=profile)
    secondary = verify_secondary_tables(
        effective, study, outcomes, table_dir=tables.resolve() if tables else None,
        table_specification=manifest["table_specification"] if tables else None)
    original = {"checked": False}
    if original_zip is not None:
        original_zip = original_zip.resolve()
        need(original_zip.is_file() and sha(original_zip) == config["original_zip_sha256"],
             "Frozen original baseline archive changed")
        original = {"checked": True, "sha256": sha(original_zip),
                    "bytes": original_zip.stat().st_size}
    math = math_self_consistency()
    solver_check = solver_math_checks(core_root, extension) if check_solver else {"checked": False}
    candidate_check = (verify_explanation_candidates(core_root, extension,
                       candidates, config) if candidates else {"checked": False})
    return {"passed": True, "profile": profile,
            "scope": "Independent full raw/action/filter/ledger/paired-statistics replay; no producer runner/statistics imports",
            "verifier_sha256": sha(Path(__file__)),
            "validation_failure_modes_sha256": sha(extension / "validation_failure_modes.md"),
            "config_sha256": sha(extension / "config.json"),
            "manifest_sha256": sha(study / "manifest.json"),
            "study_file_count": len(manifest["files"]),
            "rows": episode_receipt["rows"],
            "policy_decisions_recomputed": episode_receipt["policy_decisions_recomputed"],
            "trace_rows": traces["rows"],
            "pilot_records": pilots["pilot_records"],
            "paired_market_trajectories": manifest["paired_market_trajectories"],
            "maximum_ledger_errors": episode_receipt["maximum_ledger_errors"],
            "maximum_trace_error": traces["maximum_trace_error"],
            "maximum_pilot_model_weight_error": pilots["maximum_model_weight_error"],
            "recomputed_inference": summary,
            "table_identity_mode": table_receipt.get("identity_mode", "not_checked"),
            "original_evaluated_table_artifact_sha256": manifest["table_artifact_sha256"],
            "supplied_table_artifact_sha256": table_receipt.get("supplied_artifact_sha256"),
            "secondary_tables": secondary,
            "table_artifact": table_receipt,
            "numerical_acceptance": numerical_receipt,
            "original_zip": original,
            "math": math, "solver_short_horizon": solver_check,
            "explanatory_candidates": candidate_check,
            "limits": "Sampling intervals are empirical pilot-aware Welch intervals; solver refinement does not certify a 0.002 optimality gap or exact upper bound"}


def _rewrite_csv_gz(path: Path, mutate) -> None:
    original = path.with_name(path.name + ".original")
    path.replace(original)
    try:
        with gzip.open(original, "rt", newline="") as input_stream, \
             gzip.open(path, "wt", newline="") as output_stream:
            reader = csv.DictReader(input_stream)
            writer = csv.DictWriter(output_stream, fieldnames=reader.fieldnames)
            writer.writeheader()
            for index, row in enumerate(reader):
                for candidate in mutate(index, row):
                    writer.writerow(candidate)
    finally:
        original.unlink(missing_ok=True)


def fault_probes(core_root: Path, extension: Path, study: Path, *,
                 profile: str = "smoke", tables: Path | None = None,
                 numerical_checks: Path | None = None,
                 allow_array_equivalent_build: bool = False) -> list[dict]:
    """Alter copies of an accepted smoke export; every case must fail closed."""
    need(profile == "smoke", "Fault probes use a bounded complete smoke export")
    cases = ("missing_file", "stale_manifest_hash", "missing_row", "duplicate_row",
             "unpaired_key", "nonfinite_outcome", "changed_terminal_pnl",
             "changed_legal_action", "invalid_action_id", "changed_tape",
             "reused_pilot", "changed_selected_pilot_fill", "wrong_trace_posterior",
             "wrong_summary_contrast", "wrong_source_hash")
    receipts = []
    with tempfile.TemporaryDirectory(prefix="bayes-extension-faults-") as temporary:
        location = Path(temporary)
        for case in cases:
            fixture = location / case
            shutil.copytree(study, fixture)
            manifest = read_json(fixture / "manifest.json")
            touched = None
            if case == "missing_file":
                (fixture / "tapes/m0_r0.npz").unlink()
            elif case == "stale_manifest_hash":
                with (fixture / "episodes.csv.gz").open("ab") as stream:
                    stream.write(b"tamper")
            elif case in ("missing_row", "duplicate_row", "unpaired_key",
                          "nonfinite_outcome", "changed_terminal_pnl"):
                def mutate(index, row):
                    if index:
                        return (row,)
                    if case == "missing_row":
                        return ()
                    if case == "duplicate_row":
                        return (row, row)
                    if case == "unpaired_key":
                        row["budget"] = "1"
                    elif case == "nonfinite_outcome":
                        row["objective"] = "nan"
                    elif case == "changed_terminal_pnl":
                        row["pnl"] = str(float(row["pnl"]) + .5)
                    return (row,)
                touched = "episodes.csv.gz"
                _rewrite_csv_gz(fixture / touched, mutate)
            elif case in ("changed_legal_action", "invalid_action_id"):
                touched = "actions/m0_r0.npz"
                with np.load(fixture / touched, allow_pickle=False) as archive:
                    arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
                key = "b0_bayes"
                arrays[key][0, 0] = (1 if arrays[key][0, 0] != 1 else 2)
                if case == "invalid_action_id":
                    arrays[key][0, 0] = 99
                np.savez_compressed(fixture / touched, **arrays)
            elif case == "changed_tape":
                touched = "tapes/m0_r0.npz"
                with np.load(fixture / touched, allow_pickle=False) as archive:
                    arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
                arrays["z"][0, 0] += .5
                np.savez_compressed(fixture / touched, **arrays)
            elif case == "reused_pilot":
                touched = "pilots/m0_r1.npz"
                shutil.copy2(fixture / "pilots/m0_r0.npz", fixture / touched)
            elif case == "changed_selected_pilot_fill":
                touched = "pilots/m0_r0.npz"
                with np.load(fixture / touched, allow_pickle=False) as archive:
                    arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
                submitted = np.argwhere(arrays["submitted"])
                need(len(submitted) > 0, "Fault setup lacks submitted pilot side")
                i, t, side = submitted[0]
                arrays["fills"][i, t, side] = 1 - arrays["fills"][i, t, side]
                np.savez_compressed(fixture / touched, **arrays)
            elif case == "wrong_trace_posterior":
                touched = "traces.csv.gz"
                def mutate(index, row):
                    if index == 0:
                        row["p0"] = str(float(row["p0"]) + .1)
                    return (row,)
                _rewrite_csv_gz(fixture / touched, mutate)
            elif case == "wrong_summary_contrast":
                touched = "summary.json"
                summary = read_json(fixture / touched)
                summary["contrasts"][0]["mean"] += 1.
                (fixture / touched).write_text(json.dumps(summary, indent=2) + "\n")
            elif case == "wrong_source_hash":
                manifest["inputs"]["core"]["src/trade_learning/model.py"] = "0" * 64
            if touched is not None:
                manifest["files"][touched] = sha(fixture / touched)
            (fixture / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            try:
                # The accepted baseline already authenticated the multi-GiB
                # table arrays; only the legal-action probe repeats the full
                # table check to reach independent action-choice recomputation.
                selected_tables = tables if case == "changed_legal_action" else None
                verify(core_root, extension, fixture, profile=profile, tables=selected_tables,
                       numerical_checks=numerical_checks, check_solver=False,
                       allow_array_equivalent_build=(allow_array_equivalent_build
                                                     and selected_tables is not None))
            except (VerificationError, ValueError, OSError, EOFError, KeyError) as exc:
                receipts.append({"case": case, "rejected": True, "reason": str(exc)[:240]})
            else:
                raise VerificationError(f"Fault probe was accepted: {case}")
    return receipts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-only", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="Root of the unchanged original Trade Learning source tree")
    parser.add_argument("--extension", type=Path, default=HERE,
                        help="Root of the separate Bayes extension source")
    parser.add_argument("--study", type=Path,
                        help="Directory containing raw Bayes extension evidence")
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--tables", type=Path,
                        help="Exact numerical control directory used by the study")
    parser.add_argument("--allow-array-equivalent-build", action="store_true",
                        help="Accept a same-build cold family only when all 11 authenticated numerical arrays exactly match the evaluated family; retain both artifact identities")
    parser.add_argument("--numerical-checks", type=Path,
                        help="Recorded numerical acceptance JSON used by a full study")
    parser.add_argument("--original-zip", type=Path,
                        help="Original frozen 188-file baseline archive for unchanged-byte check")
    parser.add_argument("--solver-checks", action="store_true",
                        help="Build two small fresh control grids and compare direct quadrature")
    parser.add_argument("--candidates", type=Path,
                        help="Separate fixed explanatory-candidate public-prefix evidence")
    parser.add_argument("--candidate-only", action="store_true",
                        help="Replay explanatory public prefixes without a study")
    parser.add_argument("--fault-probes", action="store_true",
                        help="Use complete smoke evidence to demonstrate fail-closed rejection")
    parser.add_argument("--out", type=Path,
                        help="Write a repeatable machine-readable independent receipt")
    args = parser.parse_args()
    need(not args.allow_array_equivalent_build or args.tables is not None,
         "--allow-array-equivalent-build requires --tables")
    if args.candidate_only:
        need(args.candidates is not None, "--candidate-only requires --candidates")
        result = {"passed": True, "explanatory_candidates":
                  verify_explanation_candidates(args.root.resolve(), args.extension.resolve(),
                                                args.candidates, frozen_config(args.extension))}
    elif args.math_only:
        result = {"passed": True, "math": math_self_consistency()}
        if args.solver_checks:
            result["solver_short_horizon"] = solver_math_checks(
                args.root.resolve(), args.extension.resolve())
    else:
        need(args.study is not None, "--study is required for raw E2E validation")
        result = verify(args.root, args.extension, args.study, profile=args.profile,
                        tables=args.tables, numerical_checks=args.numerical_checks,
                        original_zip=args.original_zip, check_solver=args.solver_checks,
                        candidates=args.candidates,
                        allow_array_equivalent_build=args.allow_array_equivalent_build)
        if args.fault_probes:
            result["fault_probes"] = fault_probes(
                args.root, args.extension, args.study, profile=args.profile,
                tables=args.tables, numerical_checks=args.numerical_checks,
                allow_array_equivalent_build=args.allow_array_equivalent_build)
            result["fault_probes_rejected"] = len(result["fault_probes"])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
