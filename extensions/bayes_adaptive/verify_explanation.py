"""Independent raw reader for every fixed direct explanatory case.

This replays actions, rewards, current-state information and same-continuation
Bellman backups from authenticated V3 arrays. It does not import explain.py or
the production solver. See EXPLANATION_VERIFICATION.md for prior failure modes.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import expit, ndtri, xlogy

from verify_extension import (BID, ASK, K, NUMERICAL_CONTRACT, PROTOCOL_COMMIT,
                              CONFIG_SHA256, VerificationError,
                              authenticate_grid_geometry, canonical_hash,
                              frozen_config, gh, legal_actions, likelihood, need,
                              read_json, selected_outcomes, sha, verify_numerical_probes,
                              verify_explanation_candidates)
from verify_numerical_raw import (_core_root, _interpolation_plan, _legal_actions, _scores,
                                  _source_identity, _values, _verified_axes)


HERE = Path(__file__).resolve().parent
SCORE_NAMES = ("table_bayes", "direct_bayes", "same_continuation_frozen_model",
               "same_continuation_no_feedback", "weighted_q")
VALUE_KINDS = ("bayes", "reveal1", "reveal2", "reveal3", "no_feedback", "frozen_model")
Q_KINDS = ("bayes", "no_feedback", "frozen_model")
EXPECTED_ARRAYS = ({f"value-{name}.npy" for name in VALUE_KINDS}
                   | {f"q-{name}.npy" for name in Q_KINDS}
                   | {"known-q-0.npy", "known-q-1.npy"})
EXPECTED_FILES = {"EXPLANATION.md", "direct_candidates.jsonl", "direct_scores.csv",
                  "error_envelopes.json", "explanation.json", "fixed_state_checks.json",
                  "traversal.csv"}
FLOAT_ATOL = 3e-9


def _close(actual: float, reported: object, label: str, tolerance: float = FLOAT_ATOL) -> float:
    need(isinstance(reported, (int, float)) and not isinstance(reported, bool)
         and math.isfinite(float(reported)) and math.isfinite(float(actual)),
         f"Nonfinite or missing explanation quantity: {label}")
    error = abs(float(actual) - float(reported))
    need(error <= tolerance, f"Explanation {label} differs from raw calculation by {error:g}")
    return error


def _vector(values: np.ndarray, reported: object, legal: np.ndarray,
            label: str, tolerance: float = FLOAT_ATOL) -> float:
    need(isinstance(reported, list) and len(reported) == 11,
         f"Explanation {label} has the wrong action-vector shape")
    observed = []
    for action, value in enumerate(reported):
        if legal[action]:
            need(isinstance(value, (int, float)) and not isinstance(value, bool)
                 and math.isfinite(float(value)),
                 f"Missing finite legal explanation score: {label}/{action}")
            observed.append(float(value))
        else:
            need(value is None, f"Illegal explanation action is not null: {label}/{action}")
            observed.append(np.nan)
    error = float(np.max(np.abs(values[legal] - np.asarray(observed)[legal])))
    need(error <= tolerance, f"Explanation {label} differs from raw array by {error:g}")
    return error


def _family(directory: Path, expected_source: dict) -> dict:
    directory = directory.resolve()
    record = read_json(directory / "complete.json")
    need(record.get("source") == expected_source and
         record.get("artifact_sha256") == canonical_hash({
             key: value for key, value in record.items() if key != "artifact_sha256"}) and
         record.get("numerical_contract") == NUMERICAL_CONTRACT and
         record.get("retain_q") == list(Q_KINDS),
         f"Explanatory control artifact source/contract/digest changed: {directory}")
    axes1, axes2 = authenticate_grid_geometry(record), _verified_axes(record)
    need(all(np.array_equal(a, b) for a, b in zip(axes1, axes2)),
         "Independent explanatory physical axes disagree")
    spec = record["specification"]
    records = record.get("arrays")
    need(isinstance(records, list) and len(records) == 11 and
         {item.get("file") for item in records if isinstance(item, dict)} == EXPECTED_ARRAYS and
         {path.name for path in directory.iterdir() if path.is_file()} ==
         EXPECTED_ARRAYS | {"complete.json"},
         f"Incomplete explanatory family: {directory}")
    arrays = {}
    grid_size = spec["weight_points"] * spec["belief_points"] ** 2
    for item in records:
        filename = item["file"]
        path = directory / filename
        need(path.is_file() and path.stat().st_size == item.get("bytes") and
             sha(path) == item.get("sha256"), f"Explanatory array bytes changed: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        shape = ([spec["horizon"] + 1, 2 * spec["qmax"] + 1, 3]
                 + ([spec["known_belief_points"]] if filename.startswith("known-q-")
                    else [grid_size]))
        if filename.startswith(("q-", "known-q-")):
            shape.append(11)
        need(array.dtype == np.float64 and str(array.dtype) == item.get("dtype")
             and list(array.shape) == shape == item.get("shape"),
             f"Explanatory array shape/dtype changed: {path}")
        arrays[path.stem] = array
    return {"directory": directory, "record": record, "spec": spec,
            "axes": axes1, "arrays": arrays}


def _analysis_manifest(analysis: Path, candidates: Path,
                       families: dict[str, dict], config: dict) -> tuple[dict, dict]:
    manifest = read_json(analysis / "manifest.json")
    need(manifest.get("artifact_sha256") == canonical_hash({
        key: value for key, value in manifest.items() if key != "artifact_sha256"}),
        "Explanatory analysis manifest digest changed")
    candidate_manifest = read_json(candidates / "manifest.json")
    need(manifest.get("format_version") == 1 and
         manifest.get("kind") == "explanatory_analysis" and
         manifest.get("design_version") == 1 and
         manifest.get("config_sha256") == CONFIG_SHA256 and
         manifest.get("protocol_commit") == PROTOCOL_COMMIT and
         manifest.get("source") == candidate_manifest.get("source") and
         manifest.get("build") == candidate_manifest.get("build") and
         manifest.get("numerical_contract") == NUMERICAL_CONTRACT and
         manifest.get("candidate_artifact_sha256") == candidate_manifest.get("artifact_sha256"),
         "Explanatory analysis source/build/candidate identity changed")
    family_specs = [{**family["spec"], "label": label,
                     "artifact_sha256": family["record"]["artifact_sha256"]}
                    for label, family in families.items()]
    need(manifest.get("family_specifications") == family_specs and
         all(family["record"]["build"] == manifest["build"] for family in families.values()),
         "Explanatory family list or numerical build differs")
    records = manifest.get("files")
    need(isinstance(records, list) and len(records) == len(EXPECTED_FILES) and
         {item.get("file") for item in records if isinstance(item, dict)} == EXPECTED_FILES and
         {path.name for path in analysis.iterdir() if path.is_file()} ==
         EXPECTED_FILES | {"manifest.json"},
         "Explanatory output inventory is missing or duplicated")
    for item in records:
        path = analysis / item["file"]
        need(path.is_file() and path.stat().st_size == item.get("bytes") and
             sha(path) == item.get("sha256"),
             f"Explanatory output changed: {path}")
    explanation = read_json(analysis / "explanation.json")
    need(explanation.get("design_version") == 1 and
         explanation.get("candidate_manifest_sha256") == candidate_manifest["artifact_sha256"] and
         explanation.get("family_specifications") == family_specs and
         explanation.get("certified_bound") is False and
         explanation.get("comparison_families_supplied") == len(families) - 1,
         "Explanatory summary family/design identity changed")
    need(manifest.get("fixed_state_checks_passed") is True and
         explanation.get("fixed_state_checks") == read_json(analysis / "fixed_state_checks.json") and
         explanation["fixed_state_checks"].get("passed") is True,
         "Explanatory fixed-state summary differs from its authenticated record")
    return manifest, explanation


def _arbitrary_states(states: dict, config: dict, probes: Path) -> int:
    """Bind all arbitrary candidates to the frozen seed population and saved probes."""
    count = verify_numerical_probes(probes, config)
    with np.load(probes, allow_pickle=False) as saved:
        first = slice(60000, 60000 + count)
        for key in ("joint", "inventory", "signal"):
            need(np.array_equal(states[key][first], saved[key]) and
                 np.array_equal(states[key][60000:], np.tile(
                     saved[key], (config["horizon"], 1) if key == "joint" else config["horizon"])),
                 f"Arbitrary explanatory {key} differs from frozen numerical probes")
    need(count == 2581, "Explanatory numerical probe population changed")
    return count


def _coordinates(joint: np.ndarray) -> np.ndarray:
    p = np.asarray(joint, dtype=np.float64)
    w = p[..., 0] + p[..., 1]
    other = p[..., 2] + p[..., 3]
    b0 = np.divide(p[..., 1], w, out=np.full_like(w, .5), where=w > 0)
    b1 = np.divide(p[..., 3], other, out=np.full_like(w, .5), where=other > 0)
    return np.stack((w, b0, b1), axis=-1)


def _joint_from_coordinates(coords: np.ndarray) -> np.ndarray:
    w, b0, b1 = np.moveaxis(np.asarray(coords), -1, 0)
    return np.stack((w * (1. - b0), w * b0,
                     (1. - w) * (1. - b1), (1. - w) * b1), axis=-1)


def _legal(q: int, qmax: int) -> np.ndarray:
    return legal_actions(int(q), qmax)


def _known_scores(table: np.ndarray, remaining: int, q: int, x: int,
                  belief: float, qmax: int, legal: np.ndarray) -> np.ndarray:
    count = table.shape[3]
    position = float(np.clip(belief, 0., 1.)) * (count - 1)
    lo = min(int(math.floor(position)), count - 2)
    fraction = position - lo
    before = np.asarray(table[remaining, q + qmax, x + 1, lo])
    after = np.asarray(table[remaining, q + qmax, x + 1, lo + 1])
    return np.where(legal, (1. - fraction) * np.where(legal, before, 0.)
                    + fraction * np.where(legal, after, 0.), -np.inf)


def _table_scores(family: dict, kind: str, remaining: int, q: int,
                  x: int, joint: np.ndarray, legal: np.ndarray) -> np.ndarray:
    coords = _coordinates(joint)
    if kind == "weighted_q":
        known0 = _known_scores(family["arrays"]["known-q-0"], remaining,
                               q, x, float(coords[1]), family["spec"]["qmax"], legal)
        known1 = _known_scores(family["arrays"]["known-q-1"], remaining,
                               q, x, float(coords[2]), family["spec"]["qmax"], legal)
        return np.where(legal, coords[0] * np.where(legal, known0, 0.)
                        + (1. - coords[0]) * np.where(legal, known1, 0.), -np.inf)
    corners = _interpolation_plan(np.asarray(joint)[None, :], family["axes"])
    result = _scores(family["arrays"][f"q-{kind}"], remaining,
                     np.array([q]), np.array([x]), corners,
                     legal[None, :], family["spec"]["qmax"])
    return result[0]


def _reward(q: int, x: int, joint: np.ndarray, legal: np.ndarray,
            theta: float) -> np.ndarray:
    """Analytic selected-execution expectation, independent of producer code."""
    sign_mean = float(joint[1] + joint[3] - joint[0] - joint[2])
    result = np.full(11, -np.inf)
    base = .03 * q * x - .002 * q * q
    for action in np.flatnonzero(legal):
        value = base
        if action >= 9:
            side = 1 if action == 9 else -1
            value += side * .03 * x - .025 - .002
        else:
            for side, depth in ((1, int(BID[action])), (-1, int(ASK[action]))):
                if depth < 0:
                    continue
                probability = float(expit(-.3 - .7 * depth - .2 * side * x))
                threshold = float(ndtri(probability))
                density = math.exp(-.5 * threshold * threshold) / math.sqrt(2 * math.pi)
                value += probability * (.025 + .025 * depth - .001 + side * .03 * x)
                value -= .30 * theta * sign_mean * density
        result[action] = value
    return result


def _actions_and_gaps(scores: np.ndarray, legal: np.ndarray) -> tuple[int, float]:
    need(np.isfinite(scores[legal]).all() and np.isneginf(scores[~legal]).all(),
         "Explanatory action scores have the wrong legal mask")
    best = float(np.max(scores))
    candidates = legal & ((best - scores) <= 1e-12)
    need(np.any(candidates), "No stable legal explanatory action")
    action = int(np.argmax(candidates))
    ordered = np.sort(scores)
    return action, float(ordered[-1] - ordered[-2])


@lru_cache(maxsize=16)
def _normal_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    return gh(order)


def _information(signal: int, joint: np.ndarray, theta: float) -> dict[str, np.ndarray]:
    """Enumerate selected feedback under an independent four-state emission law."""
    nodes, weights = _normal_rule(241)
    prior = joint.reshape(2, 2)
    model_prior = prior.sum(axis=1)
    regime_prior = prior.sum(axis=0)
    result = {name: np.zeros(11) for name in
              ("model_information", "conditional_regime_information",
               "regime_information", "mass", "chain_identity_residual")}
    for action in range(11):
        mass_sum = np.zeros(len(nodes))
        model_information = np.zeros(len(nodes))
        conditional_information = np.zeros(len(nodes))
        regime_information = np.zeros(len(nodes))
        for fills in selected_outcomes(action):
            emission = likelihood(signal, nodes, action, fills, theta).reshape(-1, 2, 2)
            joint_mass = prior[None, :, :] * emission
            by_model = joint_mass.sum(axis=2)
            by_regime = joint_mass.sum(axis=1)
            mass = joint_mass.sum(axis=(1, 2))
            mass_sum += mass
            with np.errstate(divide="ignore", invalid="ignore"):
                model_denominator = model_prior[None, :] * mass[:, None]
                regime_denominator = regime_prior[None, :] * mass[:, None]
                cond_denominator = np.divide(
                    prior[None, :, :] * by_model[:, :, None],
                    model_prior[None, :, None],
                    out=np.zeros_like(joint_mass), where=model_prior[None, :, None] > 0.)
                model_information += np.sum(xlogy(by_model, by_model) -
                                            xlogy(by_model, model_denominator), axis=1)
                regime_information += np.sum(xlogy(by_regime, by_regime) -
                                             xlogy(by_regime, regime_denominator), axis=1)
                conditional_information += np.sum(xlogy(joint_mass, joint_mass) -
                                                  xlogy(joint_mass, cond_denominator), axis=(1, 2))
        need(np.max(np.abs(mass_sum - 1.)) <= 2e-13,
             "Selected observation outcomes fail to partition probability mass")
        result["mass"][action] = float(weights @ mass_sum)
        result["model_information"][action] = max(float(weights @ model_information), 0.)
        result["conditional_regime_information"][action] = max(
            float(weights @ conditional_information), 0.)
        result["regime_information"][action] = max(float(weights @ regime_information), 0.)
        result["chain_identity_residual"][action] = (
            result["regime_information"][action]
            - result["model_information"][action]
            - result["conditional_regime_information"][action])
    return result


def _next_joint(joint: np.ndarray, emissions: np.ndarray,
                kappas: tuple[float, float], mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Condition current H, optionally suppress feedback, then predict next H."""
    prior = joint.reshape(2, 2)
    previous_weights = prior.sum(axis=1)
    previous_beliefs = np.divide(prior[:, 1], previous_weights,
                                 out=np.full(2, .5), where=previous_weights > 0.)
    mass_by_state = emissions * joint[None, :]
    mass = mass_by_state.sum(axis=1)
    if mode == "no_feedback":
        model_weights = np.broadcast_to(previous_weights, (len(mass), 2))
        conditional = np.broadcast_to(previous_beliefs, (len(mass), 2))
    else:
        state = mass_by_state.reshape(-1, 2, 2)
        by_model = state.sum(axis=2)
        conditional = np.divide(state[:, :, 1], by_model,
                                out=np.broadcast_to(previous_beliefs, by_model.shape).copy(),
                                where=by_model > 0.)
        if mode == "frozen_model":
            model_weights = np.broadcast_to(previous_weights, by_model.shape)
        else:
            model_weights = np.divide(by_model, mass[:, None],
                                      out=np.broadcast_to(previous_weights, by_model.shape).copy(),
                                      where=mass[:, None] > 0.)
    k = np.asarray(kappas, dtype=float)
    predicted = k[None, :] + (1. - 2. * k[None, :]) * conditional
    next_joint = np.stack((model_weights * (1. - predicted),
                           model_weights * predicted), axis=-1).reshape(-1, 4)
    need(np.isfinite(next_joint).all() and np.all(next_joint >= -1e-14) and
         np.max(np.abs(next_joint.sum(axis=1) - 1.)) < 2e-13,
         "Independent next joint belief left the simplex")
    return mass, next_joint


def _backup(family: dict, remaining: int, q: int, x: int, joint: np.ndarray,
            reward: np.ndarray, mode: str) -> np.ndarray:
    """Direct legal action scores with one common BA continuation for all modes."""
    need(mode in ("bayes", "frozen_model", "no_feedback"),
         "Unknown explanatory update intervention")
    spec = family["spec"]
    legal = _legal(q, spec["qmax"])
    nodes, weights = _normal_rule(spec["quadrature_points"])
    table = family["arrays"]["value-bayes"]
    scores = np.full(11, -np.inf)
    for action in np.flatnonzero(legal):
        score = float(reward[action])
        for fills in selected_outcomes(int(action)):
            emissions = likelihood(x, nodes, int(action), fills, spec["theta"])
            probability, next_joint = _next_joint(
                joint, emissions, tuple(spec["kappas"]), mode)
            dq = ((1 if action == 9 else -1) if action >= 9 else
                  (int(fills[0] == 1) - int(fills[1] == 1)))
            next_inventory = np.full(len(nodes), q + dq, dtype=np.int64)
            corners = _interpolation_plan(next_joint, family["axes"])
            future = np.zeros(len(nodes))
            for next_signal in (-1, 0, 1):
                values = _values(table, remaining - 1, next_inventory,
                                 np.full(len(nodes), next_signal, dtype=np.int64),
                                 corners, spec["qmax"])
                future += K[x + 1, next_signal + 1] * values
            score += float(np.dot(weights * probability, future))
        scores[action] = score
    return scores


def _direct_scores(family: dict, remaining: int, q: int, x: int,
                   joint: np.ndarray, reward: np.ndarray) -> dict[str, np.ndarray]:
    legal = _legal(q, family["spec"]["qmax"])
    return {
        "table_bayes": _table_scores(family, "bayes", remaining, q, x, joint, legal),
        "direct_bayes": _backup(family, remaining, q, x, joint, reward, "bayes"),
        "same_continuation_frozen_model": _backup(
            family, remaining, q, x, joint, reward, "frozen_model"),
        "same_continuation_no_feedback": _backup(
            family, remaining, q, x, joint, reward, "no_feedback"),
        "weighted_q": _table_scores(family, "weighted_q", remaining, q, x, joint, legal),
    }


def _batch_table_scores(family: dict, kind: str, remaining: int,
                        q: np.ndarray, x: np.ndarray, joint: np.ndarray) -> np.ndarray:
    legal = _legal_actions(q, family["spec"]["qmax"])
    qmax = family["spec"]["qmax"]
    if kind == "bayes":
        corners = _interpolation_plan(joint, family["axes"])
        return _scores(family["arrays"]["q-bayes"], remaining, q, x,
                       corners, legal, qmax)
    need(kind == "weighted_q", "Unknown explanatory screening policy")
    coords = _coordinates(joint)
    known = []
    for model in (0, 1):
        table = family["arrays"][f"known-q-{model}"]
        count = table.shape[3]
        position = np.clip(coords[:, 1 + model], 0., 1.) * (count - 1)
        lo = np.minimum(np.floor(position).astype(np.int64), count - 2)
        fraction = position - lo
        before = np.where(legal, table[remaining, q + qmax, x + 1, lo], 0.)
        after = np.where(legal, table[remaining, q + qmax, x + 1, lo + 1], 0.)
        known.append(before + (after - before) * fraction[:, None])
    return np.where(legal, coords[:, 0, None] * known[0]
                    + (1. - coords[:, 0, None]) * known[1], -np.inf)


def _screen(family: dict, states: dict, traversal: Path,
            summary: list[dict]) -> tuple[np.ndarray, dict, int]:
    count = len(states["candidate_id"])
    result = {name: np.empty(count, dtype=np.int8 if name.endswith("action") else np.float64)
              for name in ("bayes_action", "weighted_q_action", "bayes_gap", "weighted_q_gap")}
    for remaining in range(1, family["spec"]["horizon"] + 1):
        selected = np.flatnonzero(states["remaining"] == remaining)
        for start in range(0, len(selected), 2048):
            ids = selected[start:start + 2048]
            q, x, p = (states[key][ids] for key in ("inventory", "signal", "joint"))
            legal = _legal_actions(q, family["spec"]["qmax"])
            for kind in ("bayes", "weighted_q"):
                scores = _batch_table_scores(family, kind, remaining, q, x, p)
                need(np.isfinite(scores[legal]).all() and np.isneginf(scores[~legal]).all(),
                     "Nonfinite/illegal table-screen score")
                best = np.max(scores, axis=1)
                choices = legal & ((best[:, None] - scores) <= 1e-12)
                ordered = np.sort(scores, axis=1)
                result[kind + "_action"][ids] = np.argmax(choices, axis=1)
                result[kind + "_gap"][ids] = ordered[:, -1] - ordered[:, -2]
    shortlist, summaries = [], []
    for source, source_name in ((0, "uniform_exploration_reached"),
                                (1, "arbitrary_numerical_probe")):
        candidates = np.flatnonzero((states["source"] == source) &
                                   (result["bayes_action"] != result["weighted_q_action"]))
        quality = np.minimum(result["bayes_gap"][candidates],
                             result["weighted_q_gap"][candidates])
        ordered = candidates[np.lexsort((candidates, -quality))]
        unique, seen = [], set()
        for candidate in ordered:
            key = (int(states["remaining"][candidate]),
                   int(states["inventory"][candidate]),
                   int(states["signal"][candidate]),
                   states["joint"][candidate].tobytes())
            if key not in seen:
                seen.add(key)
                unique.append(int(candidate))
        chosen = unique[:256]
        shortlist.extend(chosen)
        summaries.append({"source": source_name,
                          "screened": int(np.sum(states["source"] == source)),
                          "action_disagreements": len(candidates),
                          "unique_disagreements": len(unique),
                          "directly_examined": len(chosen), "direct_limit": 256})
    need(summary == summaries, "Explanatory traversal summary differs from raw Q screen")
    examined = np.zeros(count, dtype=bool)
    examined[shortlist] = True
    with traversal.open(newline="") as stream:
        reader = csv.DictReader(stream)
        expected_columns = {"candidate_id", "source", "remaining", "inventory", "signal",
                            "p0", "p1", "p2", "p3", "model_index_evaluator_only",
                            "episode", "t", "probe_index", "bayes_action",
                            "weighted_q_action", "bayes_gap", "weighted_q_gap",
                            "directly_examined"}
        need(reader.fieldnames is not None and set(reader.fieldnames) == expected_columns
             and len(reader.fieldnames) == len(expected_columns),
             "Explanatory traversal CSV columns changed")
        checked = 0
        for i, row in enumerate(reader):
            need(i < count and int(row["candidate_id"]) == i,
                 "Explanatory traversal IDs are missing/reordered")
            source_name = ("uniform_exploration_reached" if states["source"][i] == 0
                           else "arbitrary_numerical_probe")
            need(row["source"] == source_name and
                 row["directly_examined"] == str(bool(examined[i])),
                 f"Explanatory traversal source/selection changed at {i}")
            for key, state_key in (("remaining", "remaining"), ("inventory", "inventory"),
                                   ("signal", "signal"), ("episode", "episode"),
                                   ("t", "t"), ("probe_index", "probe_index"),
                                   ("model_index_evaluator_only", "model_index"),
                                   ("bayes_action", "bayes_action"),
                                   ("weighted_q_action", "weighted_q_action")):
                target = result[state_key][i] if state_key in result else states[state_key][i]
                need(int(row[key]) == int(target),
                     f"Explanatory traversal state/action changed: {i}/{key}")
            for j in range(4):
                _close(float(states["joint"][i, j]), float(row[f"p{j}"]),
                       f"traversal/{i}/p{j}", 1e-15)
            for key in ("bayes_gap", "weighted_q_gap"):
                _close(float(result[key][i]), float(row[key]),
                       f"traversal/{i}/{key}", 2e-12)
            checked += 1
        need(checked == count, "Explanatory traversal omits candidate states")
    return np.asarray(shortlist, dtype=np.int64), result, checked


def _full_vector(actual: np.ndarray, reported: object, label: str,
                 tolerance: float = FLOAT_ATOL) -> float:
    need(isinstance(reported, list) and len(reported) == 11 and
         all(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(float(value)) for value in reported),
         f"Explanatory information vector malformed: {label}")
    error = float(np.max(np.abs(actual - np.asarray(reported, dtype=float))))
    need(error <= tolerance, f"Explanatory information differs: {label}, {error:g}")
    return error


def _csv_scores(path: Path) -> dict[tuple[int, str, int], dict]:
    columns = {"candidate_id", "family", "action", "legal", "immediate_expected_reward",
               "one_period_q", *SCORE_NAMES, "enabled_continuation",
               "frozen_weight_continuation", "model_update_score_increment",
               "model_information", "conditional_regime_information", "regime_information"}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames is not None and set(reader.fieldnames) == columns
             and len(reader.fieldnames) == len(columns),
             "Direct-score CSV columns changed")
        rows = {}
        for row in reader:
            key = (int(row["candidate_id"]), row["family"], int(row["action"]))
            need(key not in rows and 0 <= key[2] <= 10,
                 "Direct-score CSV has duplicated or invalid action rows")
            rows[key] = row
    return rows


def _csv_number(row: dict, field: str, expected: float | None,
                label: str) -> float:
    value = row[field]
    if expected is None:
        need(value == "", f"Illegal direct-score cell is not null: {label}/{field}")
        return 0.
    need(value != "", f"Missing legal direct-score cell: {label}/{field}")
    return _close(expected, float(value), f"{label}/{field}")


def _direct_cases(analysis: Path, states: dict, shortlist: np.ndarray,
                  families: dict[str, dict], explanation: dict,
                  config: dict) -> dict:
    with (analysis / "direct_candidates.jsonl").open() as stream:
        entries = [json.loads(line) for line in stream if line.strip()]
    need(len(entries) == len(shortlist) and
         [entry.get("candidate_id") for entry in entries] == shortlist.tolist(),
         "Direct candidates omit or reorder the frozen shortlist")
    csv_rows = _csv_scores(analysis / "direct_scores.csv")
    expected_csv_rows = len(entries) * len(families) * 11
    need(len(csv_rows) == expected_csv_rows,
         "Direct-score CSV omits a family or action")
    maximum = {key: 0. for key in ("score", "reward", "one_period_q",
                                    "information", "csv", "gap")}
    qualifiers = 0
    stable_count = 0
    winner_candidates = []
    for position, (candidate, entry) in enumerate(zip(shortlist, entries)):
        candidate = int(candidate)
        q = int(states["inventory"][candidate])
        x = int(states["signal"][candidate])
        remaining = int(states["remaining"][candidate])
        joint = np.asarray(states["joint"][candidate], dtype=float)
        legal = _legal(q, config["qmax"])
        source = int(states["source"][candidate])
        source_name = ("uniform_exploration_reached" if source == 0
                       else "arbitrary_numerical_probe")
        need(entry.get("source") == source_name and
             entry.get("remaining") == remaining and
             entry.get("inventory") == q and entry.get("signal") == x and
             entry.get("model_index_evaluator_only") == int(states["model_index"][candidate]) and
             entry.get("episode") == int(states["episode"][candidate]) and
             entry.get("t") == int(states["t"][candidate]) and
             entry.get("probe_index") == int(states["probe_index"][candidate]) and
             entry.get("legal_actions") == legal.tolist() and
             entry.get("information_quadrature_points") == 241,
             f"Direct case provenance/design changed: {candidate}")
        need(isinstance(entry.get("joint"), list) and len(entry["joint"]) == 4 and
             max(abs(np.asarray(entry["joint"]) - joint)) <= 1e-15 and
             isinstance(entry.get("coordinates"), list) and len(entry["coordinates"]) == 3 and
             max(abs(np.asarray(entry["coordinates"]) - _coordinates(joint))) <= 1e-14,
             f"Direct case physical state changed: {candidate}")
        reachability = ("Recorded public prefix under uniform safe exploration; no scored-policy prevalence claim"
                        if source == 0 else
                        "Arbitrary joint prior; reachability from the initial prior is not established")
        need(entry.get("reachability") == reachability,
             f"Direct case reachability claim changed: {candidate}")
        reward = _reward(q, x, joint, legal, config["theta"])
        maximum["reward"] = max(maximum["reward"], _vector(
            reward, entry.get("immediate_expected_reward"), legal,
            f"case/{candidate}/immediate_reward"))
        information = _information(x, joint, config["theta"])
        need(isinstance(entry.get("information"), dict) and
             set(entry["information"]) == set(information),
             f"Direct case information inventory changed: {candidate}")
        for key, values in information.items():
            maximum["information"] = max(maximum["information"], _full_vector(
                values, entry["information"][key], f"case/{candidate}/{key}"))
        one_period = _backup(families["selected"], 1, q, x, joint, reward, "bayes")
        maximum["one_period_q"] = max(maximum["one_period_q"], _vector(
            one_period, entry.get("one_period_q"), legal,
            f"case/{candidate}/one_period_q"))
        need(isinstance(entry.get("families"), dict) and
             set(entry["families"]) == set(families),
             f"Direct case family inventory changed: {candidate}")
        computed = {}
        qualifications = {}
        for label, family in families.items():
            record = entry["families"][label]
            need(record.get("artifact_sha256") == family["record"]["artifact_sha256"] and
                 record.get("specification") == family["spec"] and
                 isinstance(record.get("scores"), dict) and
                 set(record["scores"]) == set(SCORE_NAMES),
                 f"Direct case family identity/scores changed: {candidate}/{label}")
            scores = _direct_scores(family, remaining, q, x, joint, reward)
            computed[label] = scores
            actions, gaps = {}, {}
            for name, values in scores.items():
                maximum["score"] = max(maximum["score"], _vector(
                    values, record["scores"][name], legal,
                    f"case/{candidate}/{label}/{name}"))
                action, gap = _actions_and_gaps(values, legal)
                actions[name], gaps[name] = action, gap
                need(record.get("actions", {}).get(name) == action,
                     f"Direct case action changed: {candidate}/{label}/{name}")
                maximum["gap"] = max(maximum["gap"], _close(
                    gap, record.get("gaps", {}).get(name),
                    f"case/{candidate}/{label}/{name}/gap"))
            need(set(record.get("actions", {})) == set(SCORE_NAMES) and
                 set(record.get("gaps", {})) == set(SCORE_NAMES),
                 f"Direct case action/gap inventory changed: {candidate}/{label}")
            checks = {
                "direct_matches_deployed_bayes": actions["direct_bayes"] == actions["table_bayes"],
                "bayes_differs_from_weighted_q": actions["direct_bayes"] != actions["weighted_q"],
                "frozen_update_reverses_to_weighted_q": actions["same_continuation_frozen_model"] == actions["weighted_q"],
                "bayes_margin_at_least_001": gaps["direct_bayes"] >= .001,
                "frozen_margin_at_least_001": gaps["same_continuation_frozen_model"] >= .001,
                "weighted_q_margin_at_least_001": gaps["weighted_q"] >= .001,
                "positive_model_information_for_bayes_action": information["model_information"][actions["direct_bayes"]] > 1e-10,
            }
            qualifies = all(checks.values())
            qualifications[label] = qualifies
            minimum_margin = min(gaps[name] for name in
                                 ("direct_bayes", "same_continuation_frozen_model", "weighted_q"))
            need(record.get("qualification_checks") == checks and
                 record.get("qualifies") is qualifies,
                 f"Direct case qualification changed: {candidate}/{label}")
            _close(minimum_margin, record.get("minimum_direct_margin"),
                   f"case/{candidate}/{label}/minimum_margin")
            model_increment = np.full(11, -np.inf)
            model_increment[legal] = (scores["direct_bayes"][legal]
                                      - scores["same_continuation_frozen_model"][legal])
            maximum["score"] = max(maximum["score"], _vector(
                model_increment, record.get("model_update_score_increment"), legal,
                f"case/{candidate}/{label}/model_increment"))
            for action in range(11):
                row = csv_rows.get((candidate, label, action))
                need(row is not None and row["legal"] == str(bool(legal[action])),
                     f"Direct-score CSV row/key/legality changed: {candidate}/{label}/{action}")
                row_label = f"csv/{candidate}/{label}/{action}"
                fields = {"immediate_expected_reward": reward[action],
                          "one_period_q": one_period[action],
                          **{name: values[action] for name, values in scores.items()},
                          "enabled_continuation": (scores["direct_bayes"][action] - reward[action]
                                                   if legal[action] else -np.inf),
                          "frozen_weight_continuation": (
                              scores["same_continuation_frozen_model"][action] - reward[action]
                              if legal[action] else -np.inf),
                          "model_update_score_increment": model_increment[action],
                          **{name: information[name][action] for name in
                             ("model_information", "conditional_regime_information",
                              "regime_information")}}
                for name, value in fields.items():
                    expected = None if (not legal[action] and name not in
                                        ("model_information", "conditional_regime_information",
                                         "regime_information")) else float(value)
                    maximum["csv"] = max(maximum["csv"], _csv_number(
                        row, name, expected, row_label))
        selected_scores = computed["selected"]
        for label, scores in computed.items():
            record = entry["families"][label]
            difference = {name: float(np.max(np.abs(scores[name][legal]
                                                    - selected_scores[name][legal])))
                          for name in SCORE_NAMES}
            need(set(record.get("max_finite_score_change_from_selected", {})) ==
                 set(SCORE_NAMES),
                 f"Direct case cross-family score inventory changed: {candidate}/{label}")
            for name, value in difference.items():
                _close(value, record["max_finite_score_change_from_selected"][name],
                       f"case/{candidate}/{label}/{name}/cross_family")
        stable = (len(families) > 1 and all(qualifications.values()))
        if stable:
            for label in families:
                if label == "selected":
                    continue
                for name in ("direct_bayes", "same_continuation_frozen_model", "weighted_q"):
                    stable &= (entry["families"][label]["actions"][name] ==
                               entry["families"]["selected"]["actions"][name])
        need(entry.get("refinement_stable_witness") is bool(stable),
             f"Direct case stability changed: {candidate}")
        qualifiers += int(qualifications["selected"])
        stable_count += int(stable)
        if stable:
            winner_candidates.append(position)
    need(explanation.get("direct_candidates") == len(entries) and
         explanation.get("selected_resolution_qualifiers") == qualifiers and
         explanation.get("refinement_stable_witnesses") == stable_count and
         explanation.get("refinement_stability_status") == (
             "checked_on_supplied_families" if len(families) > 1 else
             "unresolved_no_comparison_family"),
         "Explanatory summary counts or stability status changed")
    if not entries:
        need(explanation.get("winner") is None, "Explanatory result invents a winner")
        winner_id = None
    else:
        available = winner_candidates if winner_candidates else list(range(len(entries)))
        winner_position = sorted(available, key=lambda pos: (
            int(states["source"][shortlist[pos]]),
            -float(entries[pos]["families"]["selected"]["minimum_direct_margin"]),
            int(shortlist[pos])))[0]
        winner_id = int(shortlist[winner_position])
        winner = explanation.get("winner")
        need(isinstance(winner, dict) and winner.get("candidate_id") == winner_id and
             winner.get("selection_status") == (
                 "refinement_stable_witness" if winner_candidates else
                 "strongest_checked_nonqualifying_case"),
             "Explanatory winner differs from the fixed selection rule")
        original = dict(winner)
        original.pop("selection_status")
        original.pop("illustrative_posteriors")
        need(original == entries[winner_position],
             "Explanatory winner details differ from its direct-case record")
    return {"direct_cases": len(entries), "direct_family_cases": len(entries) * len(families),
            "direct_legal_scores_recomputed": sum(
                len(families) * 5 * int(np.sum(_legal(int(states["inventory"][idx]), config["qmax"])))
                for idx in shortlist),
            "csv_action_rows_checked": expected_csv_rows,
            "selected_qualifiers": qualifiers, "refinement_stable_witnesses": stable_count,
            "winner_candidate_id": winner_id, "maximum_absolute_errors": maximum}


def _fixed_checks(states: dict, families: dict[str, dict], analysis: Path,
                  explanation: dict, config: dict) -> dict:
    """Recalculate every equal-conditional and terminal-suppression case."""
    points, inventories, signals, horizons = [], [], [], []
    for remaining in (1, 2, 5, 15, 30):
        for q, x in ((-2, -1), (0, 0), (2, 1)):
            for weight in (0., .5, 1.):
                for belief in (.2, .5, .8):
                    points.append(_joint_from_coordinates(np.array([weight, belief, belief])))
                    inventories.append(q)
                    signals.append(x)
                    horizons.append(remaining)
    equal_count = len(points)
    for index in range(16):
        candidate = 60000 + index
        points.append(np.asarray(states["joint"][candidate]))
        inventories.append(int(states["inventory"][candidate]))
        signals.append(int(states["signal"][candidate]))
        horizons.append(1)
    information = [_information(int(x), np.asarray(joint), config["theta"])
                   for x, joint in zip(signals, points)]
    model_info = max(float(item["model_information"].max())
                     for item in information[:equal_count])
    chain_error = max(float(np.max(np.abs(item["chain_identity_residual"])))
                      for item in information)
    mass_error = max(float(np.max(np.abs(item["mass"] - 1.)))
                     for item in information)
    observed = read_json(analysis / "fixed_state_checks.json")
    need(isinstance(observed.get("by_family"), list) and
         len(observed["by_family"]) == len(families),
         "Fixed-state family inventory changed")
    max_report_error = 0.
    family_records = []
    for (label, family), reported in zip(families.items(), observed["by_family"]):
        equal_error = 0.
        final_error = 0.
        for index, (joint, q, x, remaining) in enumerate(zip(
                points, inventories, signals, horizons)):
            legal = _legal(q, config["qmax"])
            reward = _reward(q, x, joint, legal, config["theta"])
            enabled = _backup(family, remaining, q, x, joint, reward, "bayes")
            frozen = _backup(family, remaining, q, x, joint, reward, "frozen_model")
            if index < equal_count:
                equal_error = max(equal_error, float(np.max(np.abs(
                    enabled[legal] - frozen[legal]))))
            if remaining == 1:
                suppressed = _backup(family, remaining, q, x, joint,
                                     reward, "no_feedback")
                final_error = max(final_error,
                                  float(np.max(np.abs(enabled[legal] - frozen[legal]))),
                                  float(np.max(np.abs(enabled[legal] - suppressed[legal]))))
        computed = {
            "family": label, "states": len(points),
            "equal_conditional_states": equal_count,
            "max_equal_conditional_frozen_difference": equal_error,
            "max_final_horizon_suppression_difference": final_error,
            "max_equal_conditional_model_information": model_info,
            "max_information_chain_residual": chain_error,
            "max_mass_error": mass_error,
        }
        computed["passed"] = (equal_error <= 1e-10 and final_error <= 1e-10
                              and model_info <= 1e-10 and chain_error <= 1e-10
                              and mass_error <= 1e-11)
        need(set(reported) == set(computed),
             f"Fixed-state check fields changed: {label}")
        for key, value in computed.items():
            if isinstance(value, float):
                max_report_error = max(max_report_error, _close(
                    value, reported[key], f"fixed/{label}/{key}", 3e-9))
            else:
                need(reported[key] == value,
                     f"Fixed-state check identity/flag changed: {label}/{key}")
        family_records.append(computed)
    need(observed.get("passed") is True and
         explanation.get("fixed_state_checks") == observed and
         all(item["passed"] for item in family_records),
         "Fixed-state identities did not all pass independent recomputation")
    return {"states_per_family": len(points), "equal_conditional_states": equal_count,
            "family_state_checks": len(points) * len(families),
            "maximum_absolute_report_error": max_report_error,
            "passed": True}


def _illustrative_posteriors(winner: dict | None, config: dict) -> dict:
    if winner is None:
        return {"branches_checked": 0, "maximum_absolute_error": 0.}
    record = winner.get("illustrative_posteriors")
    need(isinstance(record, dict) and isinstance(record.get("branches"), list) and
         record.get("scope") == (
             "fixed hypothetical public observations; no realized PnL selection "
             "or probability mass at these z values"),
         "Illustrative posterior branch inventory changed")
    selected = winner["families"]["selected"]["actions"]
    actions = sorted({selected["direct_bayes"], selected["weighted_q"]})
    expected = []
    for action in actions:
        submitted = [side for side, depth in enumerate((BID[action], ASK[action]))
                     if depth >= 0]
        for z in (-2., -1., 0., 1., 2.):
            for bits in range(1 << len(submitted)):
                fills = [-1, -1]
                for bit, side in enumerate(submitted):
                    fills[side] = (bits >> bit) & 1
                expected.append((action, z, tuple(fills)))
    need(len(record["branches"]) == len(expected),
         "Illustrative posterior branches are missing or duplicated")
    maximum = 0.
    joint = np.asarray(winner["joint"], dtype=float)
    for reported, (action, z, fills) in zip(record["branches"], expected):
        need(reported.get("action") == action and reported.get("z") == z and
             reported.get("selected_fills") == list(fills),
             "Illustrative posterior branch ordering/observation changed")
        emissions = likelihood(winner["signal"], np.array([z]), action,
                               fills, config["theta"])
        for mode, key, coordinate_key in (
                ("bayes", "actual_next_joint", "actual_next_coordinates"),
                ("frozen_model", "frozen_weight_next_joint", "frozen_next_coordinates")):
            _, predicted = _next_joint(joint, emissions, tuple(config["kappas"]), mode)
            target = np.asarray(reported.get(key))
            coordinates = np.asarray(reported.get(coordinate_key))
            need(target.shape == (4,) and coordinates.shape == (3,),
                 "Illustrative posterior branch has invalid shape")
            maximum = max(maximum, float(np.max(np.abs(predicted[0] - target))),
                          float(np.max(np.abs(_coordinates(predicted[0]) - coordinates))))
    need(maximum <= 3e-12,
         f"Illustrative posterior branches differ from independent update: {maximum:g}")
    return {"branches_checked": len(expected), "maximum_absolute_error": maximum}


def _numerical_binding(explanation: dict, selected: dict,
                       numerical_path: Path | None) -> dict:
    reported = explanation.get("numerical_selection_evidence")
    if reported is None:
        need(numerical_path is None,
             "A numerical receipt was supplied but not recorded in the analysis")
        return {"receipt_sha256": None, "selected_artifact_sha256":
                selected["record"]["artifact_sha256"],
                "passes_predeclared_rule": False,
                "binding_status": "unresolved_no_receipt"}
    need(numerical_path is not None,
         "Analysis claims numerical evidence without a supplied receipt")
    numerical_path = numerical_path.resolve()
    loaded = read_json(numerical_path)
    need(reported.get("sha256") == sha(numerical_path) and
         reported.get("selected_family_artifact_sha256") ==
         selected["record"]["artifact_sha256"] and
         type(loaded.get("passes_predeclared_rule")) is bool and
         reported.get("passes_predeclared_rule") is loaded["passes_predeclared_rule"] and
         reported.get("selected_resolution") == loaded.get("selected_resolution") and
         reported.get("selected_artifact_sha256") == loaded.get("selected_artifact_sha256") and
         reported.get("certified_bound") is False,
         "Explanatory numerical receipt bytes/selected-family link changed")
    passed = loaded["passes_predeclared_rule"]
    if passed:
        checks = {
            "selected_artifact": loaded.get("selected_artifact_sha256") ==
                                 selected["record"]["artifact_sha256"],
            "selected_specification": loaded.get("selected_resolution") == selected["spec"],
            "solver_source": loaded.get("source") == selected["record"]["source"],
            "numerical_build": loaded.get("build") == selected["record"]["build"],
        }
        need(all(checks.values()) and reported.get("binding_checks") == checks and
             reported.get("acceptance_bound_to_selected_family") is True and
             reported.get("binding_status") == "accepted_family_authenticated" and
             loaded.get("belief_geometry") == selected["spec"]["belief_geometry"],
             "Passing numerical receipt does not authenticate this selected family")
    else:
        need(reported.get("binding_checks") is None and
             reported.get("acceptance_bound_to_selected_family") is False and
             reported.get("binding_status") == "unresolved_receipt_not_acceptance",
             "Nonpassing numerical receipt is represented as accepted")
    return {"receipt_sha256": sha(numerical_path),
            "selected_artifact_sha256": selected["record"]["artifact_sha256"],
            "passes_predeclared_rule": passed,
            "binding_status": reported["binding_status"]}


def _compare_nested(actual: object, reported: object, label: str) -> float:
    if isinstance(actual, dict):
        need(isinstance(reported, dict) and set(reported) == set(actual),
             f"Explanatory envelope fields differ: {label}")
        return max((_compare_nested(value, reported[key], f"{label}/{key}")
                    for key, value in actual.items()), default=0.)
    if isinstance(actual, list):
        need(isinstance(reported, list) and len(reported) == len(actual),
             f"Explanatory envelope row count differs: {label}")
        return max((_compare_nested(value, got, f"{label}/{i}")
                    for i, (value, got) in enumerate(zip(actual, reported))), default=0.)
    if isinstance(actual, float):
        return _close(actual, reported, label, 3e-10)
    need(type(reported) is type(actual) and reported == actual,
         f"Explanatory envelope identity/flag differs: {label}")
    return 0.


def _envelopes(analysis: Path, families: dict[str, dict], config: dict) -> dict:
    """Independently integrate Gaussian W1 and evaluate all physical-cell allowances."""
    reported = read_json(analysis / "error_envelopes.json")
    theta = config["theta"]
    ctheta = theta / math.sqrt(1. - theta * theta) / math.sqrt(2. * math.pi)
    rules, levels = {}, []
    for family in families.values():
        spec = family["spec"]
        order = spec["quadrature_points"]
        if str(order) not in rules:
            nodes, raw_weights = gh(order)
            original_mass = float(raw_weights.sum())
            weights = raw_weights / original_mass
            cuts = np.r_[0., np.cumsum(weights)]
            cuts[-1] = 1.
            limits = ndtri(cuts)
            phi = lambda value: math.exp(-.5 * value * value) / math.sqrt(2. * math.pi)
            terms = []
            for low, high, node in zip(limits[:-1], limits[1:], nodes):
                middle = min(max(float(node), float(low)), float(high))
                left = quad(lambda value: (node - value) * phi(value),
                            float(low), middle, epsabs=1e-12, epsrel=1e-12)[0]
                right = quad(lambda value: (value - node) * phi(value),
                             middle, float(high), epsabs=1e-12, epsrel=1e-12)[0]
                terms.append(left + right)
            rules[str(order)] = {
                "distance": float(sum(terms)), "original_mass": original_mass,
                "normalization_correction": 1. - original_mass,
                "discrete_mean": float(weights @ nodes),
                "discrete_second_moment": float(weights @ (nodes * nodes)),
                "terms": terms, "certified_bound": False,
                "arithmetic": "float64 CDF/inverse-CDF evaluation, no outward-rounded intervals",
            }
        geometry = family["record"]["grid_geometry"]
        widths = {name: float(np.max(np.diff(axis)))
                  for name, axis in zip(geometry["axis_order"], family["axes"])}
        delta = widths["weight0"] + max(widths["conditional_plus0"],
                                         widths["conditional_plus1"])
        distance = rules[str(order)]["distance"]
        rows = []
        for n in range(config["horizon"] + 1):
            sum_d = sum((.231 * k + .054 / 2. for k in range(n)), 0.)
            integration = 4. * ctheta * distance * sum_d
            nodal = sum_d * (delta + 4. * ctheta * distance)
            final_interp = 0. if n == 0 else (.231 * n + .054 / 2.) * delta
            rows.append({"remaining": n, "sum_D_continuations": sum_d,
                         "one_sided_allowance": integration,
                         "two_sided_nodal_allowance": nodal,
                         "additional_final_off_grid_interpolation": final_interp,
                         "two_sided_off_grid_allowance": nodal + final_interp})
        levels.append({**spec, "label": next(label for label, value in families.items()
                                          if value is family),
                       "artifact_sha256": family["record"]["artifact_sha256"],
                       "grid_geometry": geometry, "max_cell_widths": widths,
                       "delta_grid_definition": (
                           "maximum physical weight width plus maximum conditional-belief width"),
                       "delta_grid": delta, "normal_transport_distance": distance,
                       "by_horizon": rows, "certified_bound": False})
    expected = {
        "theory": "THEORY.md equations (27)-(32)",
        "stage_expected_reward_envelope": .231,
        "terminal_payoff_span": .054,
        "c_theta": ctheta,
        "quantity_scope": "BA table; known-model/revelation terminal errors require their own allowance",
        "certified_bound": False,
        "solver_rounding_allowance_included": False,
        "excludes": ["outward-rounded special functions and quadrature parameters",
                     "solver floating-point error",
                     "approximate maximizing/tie arithmetic error"],
        "interpretation": "broad mathematical allowances evaluated numerically; not an economic near-optimality certificate",
        "rules": rules, "levels": levels,
    }
    error = _compare_nested(expected, reported, "envelope")
    return {"families": len(levels),
            "horizon_rows_recomputed": sum(len(level["by_horizon"]) for level in levels),
            "quadrature_rules_integrated": len(rules),
            "maximum_absolute_report_error": error,
            "passed": True}


def verify(core_root: Path, candidates: Path, analysis: Path,
           selected: Path, comparisons: list[Path], probes: Path,
           numerical_checks: Path | None) -> dict:
    config = frozen_config(HERE)
    need(len(comparisons) == 2,
         "Full explanatory validation requires both comparison families")
    core_root, candidates, analysis, probes = (
        path.resolve() for path in (core_root, candidates, analysis, probes))
    candidate_result = verify_explanation_candidates(
        core_root, HERE, candidates, config)
    with np.load(candidates / "candidate_states.npz", allow_pickle=False) as archive:
        states = {key: np.array(archive[key], copy=True) for key in archive.files}
    probe_count = _arbitrary_states(states, config, probes)
    expected_source = _source_identity(core_root)
    families = {"selected": _family(selected, expected_source)}
    for i, directory in enumerate(comparisons, start=1):
        families[f"comparison_{i}"] = _family(directory, expected_source)
    need(len({item["record"]["artifact_sha256"] for item in families.values()}) == 3,
         "Explanatory comparison families are not distinct")
    for label, family in families.items():
        spec = family["spec"]
        need(spec.get("theta") == config["theta"] and
             spec.get("kappas") == config["kappas"] and
             spec.get("horizon") == config["horizon"] and
             spec.get("qmax") == config["qmax"] and
             spec.get("known_belief_points") ==
             config["known_reference"]["belief_points"] and
             spec.get("known_quadrature_points") ==
             config["known_reference"]["quadrature_points"] and
             spec.get("belief_geometry") == "endpoint_sine",
             f"Explanatory family differs from frozen endpoint market: {label}")
    manifest, explanation = _analysis_manifest(analysis, candidates, families, config)
    numerical_binding = _numerical_binding(
        explanation, families["selected"], numerical_checks)
    envelope = _envelopes(analysis, families, config)
    shortlist, _, traversal_count = _screen(
        families["selected"], states, analysis / "traversal.csv",
        explanation["search"])
    direct = _direct_cases(analysis, states, shortlist, families,
                           explanation, config)
    fixed = _fixed_checks(states, families, analysis, explanation, config)
    illustrative = _illustrative_posteriors(explanation.get("winner"), config)
    need(traversal_count == 137430 and probe_count == 2581 and
         direct["direct_cases"] == len(shortlist) and
         direct["direct_family_cases"] == 3 * len(shortlist) and
         direct["csv_action_rows_checked"] == 33 * len(shortlist) and
         fixed["states_per_family"] == 151 and fixed["family_state_checks"] == 453,
         "Independent explanatory audit did not cover the full declared scope")
    return {
        "schema_version": 1,
        "validation_completion": True,
        "scope": "full_endpoint_explanation_independent_raw_recomputation",
        "certified_bound": False,
        "source": {"verifier_sha256": sha(Path(__file__)),
                   "candidate_reader_sha256": sha(HERE / "verify_extension.py"),
                   "physical_reader_sha256": sha(HERE / "verify_numerical_raw.py"),
                   "solver_source": expected_source},
        "candidate": {"manifest_sha256": sha(candidates / "manifest.json"),
                      "artifact_sha256": candidate_result["artifact_sha256"],
                      "states": 137430, "public_decisions_replayed":
                      candidate_result["public_decisions_replayed"],
                      "frozen_numerical_probe_sha256": sha(probes),
                      "frozen_numerical_probe_count": probe_count},
        "analysis": {"manifest_sha256": sha(analysis / "manifest.json"),
                     "artifact_sha256": manifest["artifact_sha256"],
                     "file_records": manifest["files"],
                     "traversal_rows_recomputed": traversal_count,
                     "direct": direct,
                     "fixed_state": fixed,
                     "illustrative_posteriors": illustrative,
                     "error_envelopes": envelope},
        "families": {label: {
            "artifact_sha256": family["record"]["artifact_sha256"],
            "manifest_sha256": sha(family["directory"] / "complete.json"),
            "specification": family["spec"],
            "grid_axes_sha256": family["record"]["grid_geometry"]["axes_sha256"],
            "array_records": family["record"]["arrays"],
        } for label, family in families.items()},
        "numerical_acceptance_binding": numerical_binding,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Core source root (or set TRADE_LEARNING_CORE_ROOT)")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, action="append", required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--numerical-checks", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(_core_root(args.root), args.candidates, args.analysis,
                    args.selected, args.comparison, args.probes,
                    args.numerical_checks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"validation_completion": result["validation_completion"],
                      "analysis_artifact_sha256": result["analysis"]["artifact_sha256"],
                      "direct_cases": result["analysis"]["direct"]["direct_cases"],
                      "output": str(args.out.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
