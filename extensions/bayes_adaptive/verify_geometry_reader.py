"""Bounded E2E check of independent V3 readers against real cold control tables.

This checks implementation and provenance at a short horizon. It never declares
the frozen 30-horizon numerical gate or the economic study accepted.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

from verify_extension import (VerificationError, authenticate_grid_geometry,
                              canonical_hash, frozen_config, independent_actions,
                              read_json, sha, verify_numerical_probes)
from verify_numerical_raw import (_core_root, _interpolation_plan, _legal_actions,
                                  _scores, _source_identity, _values, _verified_axes)


HERE = Path(__file__).resolve().parent
Q_KINDS = ("bayes", "no_feedback", "frozen_model")
EXPECTED_ARRAYS = ({f"value-{kind}.npy" for kind in
                    ("bayes", "reveal1", "reveal2", "reveal3", "no_feedback", "frozen_model")}
                   | {f"q-{kind}.npy" for kind in Q_KINDS}
                   | {"known-q-0.npy", "known-q-1.npy"})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _family(directory: Path, expected_source: dict) -> tuple[dict, dict]:
    complete = read_json(directory / "complete.json")
    require(complete.get("source") == expected_source,
            f"Family has stale source: {directory}")
    require(complete.get("artifact_sha256") == canonical_hash({
        key: value for key, value in complete.items() if key != "artifact_sha256"}),
        f"Family manifest digest changed: {directory}")
    axes1, axes2 = authenticate_grid_geometry(complete), _verified_axes(complete)
    require(all(np.array_equal(left, right) for left, right in zip(axes1, axes2)),
            "Independent physical-axis readers disagree")
    records = complete.get("arrays")
    require(isinstance(records, list) and len(records) == 11 and
            {item.get("file") for item in records if isinstance(item, dict)} == EXPECTED_ARRAYS,
            "Family lacks the eleven fixed numerical arrays")
    require({path.name for path in directory.iterdir() if path.is_file()} ==
            EXPECTED_ARRAYS | {"complete.json"}, "Family contains unrecorded files")
    arrays = {}
    for item in records:
        name = item["file"]
        path = directory / name
        require(path.is_file() and path.stat().st_size == item["bytes"] and
                sha(path) == item["sha256"], f"Array content changed: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        require(array.dtype == np.float64 and list(array.shape) == item["shape"] and
                str(array.dtype) == item["dtype"], f"Array shape/dtype changed: {path}")
        arrays[path.stem] = array
    return complete, arrays


def _rejections(complete: dict) -> list[str]:
    cases = {}
    changed = deepcopy(complete)
    changed["grid_geometry"]["axes"]["conditional_plus0"][1] += .0001
    cases["altered_axis"] = changed
    changed = deepcopy(complete)
    changed["grid_geometry"]["axes_sha256"] = "0" * 64
    cases["altered_axes_digest"] = changed
    changed = deepcopy(complete)
    changed["grid_geometry"]["interpolation"] = "sine-parameter-interpolation"
    cases["wrong_interpolation"] = changed
    changed = deepcopy(complete)
    changed["specification"]["belief_geometry"] = "uniform"
    cases["mixed_geometry"] = changed
    changed = deepcopy(complete)
    changed["format_version"] = 2
    cases["historical_format"] = changed
    rejected = []
    for label, item in cases.items():
        for verifier in (authenticate_grid_geometry, _verified_axes):
            try:
                verifier(item)
            except (VerificationError, ValueError):
                pass
            else:
                raise ValueError(f"{label} passed a physical-axis reader")
        rejected.append(label)
    return rejected


def run(first: Path, cold: Path, probes: Path, core: Path, out: Path) -> dict:
    require(not out.exists(), "Independent E2E receipt path must be fresh")
    first, cold, probes, core = (path.resolve() for path in (first, cold, probes, core))
    require(first != cold, "Cold family must be a separate completed directory")
    config = frozen_config(HERE)
    require(verify_numerical_probes(probes, config) == 2581,
            "Frozen numerical probe count changed")
    source = _source_identity(core)
    initial, arrays = _family(first, source)
    rebuilt, _ = _family(cold, source)
    specification = initial["specification"]
    require(specification["belief_geometry"] == "endpoint_sine" and
            specification["theta"] == config["theta"] and
            specification["kappas"] == config["kappas"] and
            specification["qmax"] == config["qmax"] and
            initial["build"] == rebuilt["build"] and
            initial["numerical_contract"] == rebuilt["numerical_contract"] and
            initial["arrays"] == rebuilt["arrays"] and
            specification == rebuilt["specification"] and
            initial["grid_geometry"] == rebuilt["grid_geometry"],
            "Cold endpoint family is not exactly array-equivalent on this build")
    require(specification["horizon"] < config["horizon"],
            "Bounded E2E may not be mistaken for the full frozen horizon")
    with np.load(probes, allow_pickle=False) as raw:
        q = np.array(raw["inventory"], dtype=np.int64)
        x = np.array(raw["signal"], dtype=np.int64)
        joint = np.array(raw["joint"], dtype=np.float64)
    tables = {name: arrays[name] for name in
              ("q-bayes", "q-no_feedback", "q-frozen_model", "known-q-0", "known-q-1")}
    sys.path.insert(0, str(core / "src"))
    sys.path.insert(0, str(HERE))
    from trade_learning.numerics import stable_argmax
    from solver import load_family

    family = load_family(first)
    decisions = []
    corners = _interpolation_plan(joint, _verified_axes(initial))
    legal = _legal_actions(q, specification["qmax"])
    maximum_raw_value_error = 0.
    maximum_raw_q_error = 0.
    for remaining in range(1, specification["horizon"] + 1):
        for kind in ("bayes", "reveal1", "reveal2", "reveal3", "no_feedback", "frozen_model"):
            actual_value = _values(arrays[f"value-{kind}"], remaining, q, x,
                                   corners, specification["qmax"])
            producer_value = family.values(kind, remaining, q, x, joint)
            error = float(np.max(np.abs(actual_value - producer_value)))
            maximum_raw_value_error = max(maximum_raw_value_error, error)
            require(error <= 1e-10,
                    f"Raw physical value interpolation differs: {kind}, horizon {remaining}")
        for kind in Q_KINDS + ("weighted_q",):
            actual = independent_actions(kind, remaining, q, x, joint, 0,
                                         tables, specification)
            producer = family.actions(kind, remaining, q, x, joint)
            require(np.array_equal(actual, producer),
                    f"Independent action replay differs: {kind}, horizon {remaining}")
            decisions.append(actual)
            if kind in Q_KINDS:
                actual_q = _scores(arrays[f"q-{kind}"], remaining, q, x,
                                   corners, legal, specification["qmax"])
                producer_q = family.q_values(kind, remaining, q, x, joint)
                error = float(np.max(np.abs(actual_q[legal] - producer_q[legal])))
                maximum_raw_q_error = max(maximum_raw_q_error, error)
                require(error <= 1e-10,
                        f"Raw physical Q interpolation differs: {kind}, horizon {remaining}")
        for model in (0, 1):
            actual = independent_actions("known_parameter", remaining, q, x, joint,
                                         model, tables, specification)
            producer = stable_argmax(family.known_q_values(
                model, remaining, q, x, joint)).astype(np.int8)
            require(np.array_equal(actual, producer),
                    f"Independent known-model action replay differs: model {model}, horizon {remaining}")
            decisions.append(actual)
    decision_bytes = np.stack(decisions).tobytes()
    import hashlib
    result = {
        "passed": True,
        "scope": "Short-horizon cold V3 endpoint family and independent physical-state action replay; no 30-horizon convergence claim",
        "verifier_sha256": sha(Path(__file__)),
        "independent_extension_verifier_sha256": sha(HERE / "verify_extension.py"),
        "independent_raw_verifier_sha256": sha(HERE / "verify_numerical_raw.py"),
        "first_manifest_sha256": sha(first / "complete.json"),
        "cold_manifest_sha256": sha(cold / "complete.json"),
        "first_artifact_sha256": initial["artifact_sha256"],
        "cold_artifact_sha256": rebuilt["artifact_sha256"],
        "axes_sha256": initial["grid_geometry"]["axes_sha256"],
        "probe_sha256": sha(probes), "probe_count": len(q),
        "specification": specification,
        "array_records_identical": True, "array_count": 11,
        "all_array_bytes_authenticated": True,
        "value_modes_interpolated": 6,
        "q_modes_interpolated": 3,
        "maximum_independent_raw_value_error": maximum_raw_value_error,
        "maximum_independent_raw_q_error": maximum_raw_q_error,
        "horizon_policy_probe_decisions_replayed": len(decisions) * len(q),
        "decision_sha256": hashlib.sha256(decision_bytes).hexdigest(),
        "geometry_faults_rejected_by_both_readers": _rejections(initial),
        "reproduction_command": [sys.executable, str(Path(__file__).resolve()),
                                 "--first", str(first), "--cold", str(cold),
                                 "--probes", str(probes), "--root", str(core),
                                 "--out", str(out)],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--cold", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.first, args.cold, args.probes, _core_root(args.root), args.out)
    print(json.dumps({"passed": result["passed"],
                      "decisions_replayed": result["horizon_policy_probe_decisions_replayed"],
                      "out": str(args.out.resolve())}))


if __name__ == "__main__":
    main()
