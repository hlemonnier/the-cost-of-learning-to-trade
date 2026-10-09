"""Prespecified action-level update-dependence analysis; never starts a solver.

Read explain_failure_modes.md before use. All exploratory streams and selection
rules were fixed before generation. Ordinary floating calculations below are
not interval-arithmetic certificates for any computed optimality bound.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
CORE = Path(os.environ.get("TRADE_LEARNING_CORE_ROOT", REPOSITORY)).resolve()
if not (CORE / "src/trade_learning/model.py").is_file():
    CORE = REPOSITORY / "Trade Learning_Hugo_Lemonnier"
if not (CORE / "src/trade_learning/model.py").is_file():
    raise RuntimeError("Set TRADE_LEARNING_CORE_ROOT to the unchanged extracted core package")
sys.path.insert(0, str(CORE / "src"))
sys.path.insert(0, str(HERE))

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri, roots_hermitenorm, xlogy

from trade_learning.environment import BatchedEnvironment, generate_exogenous
from trade_learning.model import ACTIONS, CP, CT, DEPTH, H, K, LAMBDA, MU, SIGMA, admissible, fill_probability
from trade_learning.numerics import build_fingerprint, canonical_hash, file_sha256, numerical_contract, stable_argmax
from solver import (coordinates_to_joint, expected_rewards, joint_to_coordinates,
                    load_family, numerical_probes, update_joint, SolverSpec)

CONFIG_SHA256 = "8681e5a8c48f25967f95a4b7d42b76531cbc4e8e00d013114c3e6efe93aee462"
PROTOCOL_COMMIT = "8db62080ef4e5170413d01b7f5fdac33bfd2b1e5"
DESIGN_VERSION = 1
EPISODES_PER_MODEL = 1000
DIRECT_LIMIT_PER_SOURCE = 256
ROBUST_GAP = .001
INFORMATION_THRESHOLD = 1e-10
INFORMATION_QUADRATURE = 241
STAGE_ENVELOPE = .231
TERMINAL_SPAN = .054
SOURCE_NAMES = {0: "uniform_exploration_reached", 1: "arbitrary_numerical_probe"}
SCORE_NAMES = ("table_bayes", "direct_bayes", "same_continuation_frozen_model",
               "same_continuation_no_feedback", "weighted_q")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")


def _new_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _config(path):
    path = Path(path)
    if file_sha256(path) != CONFIG_SHA256:
        raise ValueError("configuration differs from the prespecified extension")
    value = json.loads(path.read_text())
    if (value["root_seed"], value["seed_namespaces"]["explanatory_probes"],
        value["horizon"], value["qmax"], value["theta"], value["kappas"]) != (
            260925508, 410, 30, 2, .35, [.002, .1]):
        raise ValueError("unsupported explanatory market or stream contract")
    return value


def _source_identity():
    paths = {"explain.py": Path(__file__), "explain_failure_modes.md": HERE / "explain_failure_modes.md",
             "THEORY.md": HERE / "THEORY.md", "solver.py": HERE / "solver.py"}
    for name in ("model.py", "filtering.py", "environment.py", "numerics.py", "control.py"):
        paths["trade_learning/" + name] = CORE / "src/trade_learning" / name
    files = {name: file_sha256(path) for name, path in paths.items()}
    return {"files": files, "sha256": canonical_hash(files)}


def _array_record(path):
    with np.load(path, allow_pickle=False) as arrays:
        structure = {name: {"shape": list(arrays[name].shape), "dtype": str(arrays[name].dtype)}
                     for name in arrays.files}
    return {"file": Path(path).name, "bytes": Path(path).stat().st_size,
            "sha256": file_sha256(path), "arrays": structure}


def _manifest(directory, kind, payload):
    record = {"format_version": 1, "kind": kind, "design_version": DESIGN_VERSION,
              "protocol_commit": PROTOCOL_COMMIT, "config_sha256": CONFIG_SHA256,
              "source": _source_identity(), "build": build_fingerprint(),
              "numerical_contract": numerical_contract(),
              "created_utc": datetime.now(timezone.utc).isoformat(), **payload}
    record["artifact_sha256"] = canonical_hash(record)
    _write_json(Path(directory) / "manifest.json", record)
    return record


def _checked_joint(joint):
    p = np.asarray(joint, dtype=float)
    if p.ndim != 2 or p.shape[1] != 4 or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("joint must be a finite nonnegative (states,4) array")
    if not np.allclose(p.sum(axis=1), 1., atol=1e-12, rtol=0.):
        raise ValueError("joint state probabilities must sum to one")
    return p


def analytic_rewards(inventory, signal, joint, theta=.35, qmax=2):
    """Independent formula (6)-(7), with explicit legal-action masking."""
    p = _checked_joint(joint)
    q, x = np.asarray(inventory), np.asarray(signal)
    if q.shape != (len(p),) or x.shape != q.shape or not np.isin(x, (-1, 0, 1)).all():
        raise ValueError("public state dimensions or signal invalid")
    legal = admissible(q, qmax)
    sign_mean = p[:, 1] + p[:, 3] - p[:, 0] - p[:, 2]
    reward = np.full((len(p), 11), -np.inf)
    base = q * MU * x - LAMBDA * q * q
    for action in range(11):
        value = base.astype(float).copy()
        if action >= 9:
            value += (1 if action == 9 else -1) * MU * x - H - CT
        else:
            for side, depth in zip((1, -1), ACTIONS[action]):
                if depth < 0:
                    continue
                prob = fill_probability(x, side, depth)
                threshold = ndtri(prob)
                density = np.exp(-threshold * threshold / 2) / math.sqrt(2 * math.pi)
                value += prob * (H + DEPTH * depth - CP + side * MU * x)
                value -= SIGMA * theta * sign_mean * density
        reward[:, action] = np.where(legal[:, action], value, -np.inf)
    return reward


def information_scores(signal, joint, theta=.35, quadrature_points=INFORMATION_QUADRATURE):
    """Current-state information only; no model label, reward bonus or transition.

    MI(model;O), MI(H;O|model), and MI(H;O) use independently enumerated selected
    outcomes. All actions are scored, irrespective of inventory admissibility.
    The common-emission identity is MI(H;O)=MI(model;O)+MI(H;O|model).
    """
    p = _checked_joint(joint)
    x = np.asarray(signal)
    if x.shape != (len(p),) or not np.isin(x, (-1, 0, 1)).all():
        raise ValueError("signal dimensions/values invalid")
    if not np.isfinite(theta) or not 0 <= theta < 1:
        raise ValueError("theta must be in [0,1)")
    if not isinstance(quadrature_points, int) or quadrature_points < 1:
        raise ValueError("quadrature_points must be positive integer")
    nodes, weights = roots_hermitenorm(quadrature_points)
    weights = weights / weights.sum()
    fields = {name: np.zeros((len(p), 11)) for name in (
        "model_information", "conditional_regime_information", "regime_information", "mass")}
    for current_x in (-1, 0, 1):
        take = np.flatnonzero(x == current_x)
        if not len(take):
            continue
        pp = p[take].reshape(-1, 2, 2)
        model_prior = pp.sum(axis=2)
        hidden_prior = pp.sum(axis=1)
        for action in range(11):
            sides = [] if action >= 9 else [(s, int(k)) for s, k in zip((1, -1), ACTIONS[action]) if k >= 0]
            mi_m = np.zeros((len(take), len(nodes)))
            mi_cond = np.zeros_like(mi_m)
            mi_h = np.zeros_like(mi_m)
            mass_sum = np.zeros_like(mi_m)
            for bits in range(1 << len(sides)):
                likelihood = np.ones((2, len(nodes)))
                for side_index, (side, depth) in enumerate(sides):
                    threshold = ndtri(fill_probability(current_x, side, depth))
                    fill_sign = 1 if (bits >> side_index) & 1 else -1
                    for hi, hidden in enumerate((-1, 1)):
                        likelihood[hi] *= ndtr(fill_sign * (threshold - hidden * theta * side * nodes)
                                                 / math.sqrt(1 - theta * theta))
                joint_mass = pp[:, :, :, None] * likelihood[None, None, :, :]
                model_mass = joint_mass.sum(axis=2)
                hidden_mass = joint_mass.sum(axis=1)
                mass = model_mass.sum(axis=1)
                conditional_mass = np.divide(model_mass, model_prior[:, :, None],
                                             out=np.zeros_like(model_mass), where=model_prior[:, :, None] > 0)
                mi_m += np.sum(xlogy(model_mass, model_mass)
                               - xlogy(model_mass, model_prior[:, :, None])
                               - xlogy(model_mass, mass[:, None, :]), axis=1)
                mi_cond += np.sum(xlogy(joint_mass, likelihood[None, None, :, :])
                                  - xlogy(joint_mass, conditional_mass[:, :, None, :]), axis=(1, 2))
                mi_h += np.sum(xlogy(hidden_mass, likelihood[None, :, :])
                               - xlogy(hidden_mass, mass[:, None, :]), axis=1)
                mass_sum += mass
            for key, array in (("model_information", mi_m), ("conditional_regime_information", mi_cond),
                               ("regime_information", mi_h), ("mass", mass_sum)):
                values = array @ weights
                if not np.isfinite(values).all():
                    raise ArithmeticError("information enumeration produced a nonfinite value")
                if key != "mass":
                    if np.min(values) < -1e-11:
                        raise ArithmeticError("information gain is materially negative")
                    values = np.maximum(values, 0.)
                fields[key][take, action] = values
    fields["chain_identity_residual"] = (fields["regime_information"] - fields["model_information"]
                                         - fields["conditional_regime_information"])
    return fields


def normal_transport_distance(nodes, weights):
    """Equation (30), evaluated in float64; normalization is explicitly recorded."""
    nodes, weights = np.asarray(nodes, float), np.asarray(weights, float)
    if (nodes.ndim != 1 or weights.shape != nodes.shape or not len(nodes)
            or not np.isfinite(nodes).all() or not np.isfinite(weights).all()
            or np.any(weights <= 0) or np.any(np.diff(nodes) <= 0)):
        raise ValueError("strictly increasing finite nodes and positive finite weights required")
    original_mass = float(weights.sum())
    if original_mass <= 0 or abs(original_mass - 1.) > 1e-10:
        raise ValueError("quadrature weights must already sum to one within 1e-10")
    weights = weights / original_mass
    cuts = np.r_[0., np.cumsum(weights)]
    cuts[-1] = 1.
    if np.any(cuts < -1e-15) or np.any(cuts > 1 + 1e-15):
        raise ArithmeticError("invalid cumulative quadrature mass")
    cuts = np.clip(cuts, 0., 1.)
    bounds = ndtri(cuts)
    normal_density = lambda z: math.exp(-.5 * z * z) / math.sqrt(2 * math.pi) if math.isfinite(z) else 0.
    terms = []
    for a, b, node in zip(bounds[:-1], bounds[1:], nodes):
        middle = min(max(float(node), float(a)), float(b))
        value = node * (ndtr(middle) - ndtr(a)) + normal_density(middle) - normal_density(float(a))
        value += normal_density(middle) - normal_density(float(b)) - node * (ndtr(b) - ndtr(middle))
        if value < -1e-13:
            raise ArithmeticError("negative Gaussian transport interval")
        terms.append(max(float(value), 0.))
    return {"distance": float(sum(terms)), "original_mass": original_mass,
            "normalization_correction": 1. - original_mass,
            "discrete_mean": float(weights @ nodes), "discrete_second_moment": float(weights @ (nodes * nodes)),
            "terms": terms, "certified_bound": False,
            "arithmetic": "float64 CDF/inverse-CDF evaluation, no outward-rounded intervals"}


def error_envelopes(config, specifications=None):
    """Evaluate broad allowances using physical maximum cell widths.

    Uniform records retain their original arithmetic and metadata. Endpoint
    meshes require their actual widest cells, not average spacing or widths in
    the sine parameter. These float64 evaluations remain uncertified.
    """
    from solver import grid_axes, grid_geometry_identity

    levels = specifications or [
        {"weight_points": w, "belief_points": b, "quadrature_points": gh}
        for w, b, _ in config["joint_grid_levels"] for gh in config["quadrature_levels"]]
    rules = {}
    records = []
    theta, horizon = float(config["theta"]), int(config["horizon"])
    ctheta = theta / math.sqrt(1 - theta * theta) / math.sqrt(2 * math.pi)
    for level in levels:
        geometry = level.get("belief_geometry", "uniform")
        if geometry not in ("uniform", "endpoint_sine"):
            raise ValueError("belief_geometry must be uniform or endpoint_sine")
        wp, bp, gh = (int(level[key]) for key in ("weight_points", "belief_points", "quadrature_points"))
        if min(wp, bp) < 2 or gh < 1:
            raise ValueError("invalid grid or quadrature size")
        if gh not in rules:
            nodes, weights = roots_hermitenorm(gh)
            rules[gh] = normal_transport_distance(nodes, weights / math.sqrt(2 * math.pi))
        distance = rules[gh]["distance"]
        geometry_evidence = {}
        if geometry == "uniform":
            delta = 1 / (wp - 1) + 1 / (bp - 1)
        else:
            spec = SolverSpec(weight_points=wp, belief_points=bp,
                              quadrature_points=gh, belief_geometry=geometry)
            identity = grid_geometry_identity(spec)
            widths = {name: float(np.max(np.diff(axis)))
                      for name, axis in zip(identity["axis_order"], grid_axes(spec))}
            delta = widths["weight0"] + max(widths["conditional_plus0"], widths["conditional_plus1"])
            geometry_evidence = {"grid_geometry": identity, "max_cell_widths": widths,
                                 "delta_grid_definition": "maximum physical weight width plus maximum conditional-belief width"}
        rows = []
        for n in range(horizon + 1):
            sum_d = STAGE_ENVELOPE * n * (n - 1) / 2 + TERMINAL_SPAN / 2 * n
            integration = 4 * ctheta * distance * sum_d
            nodal = sum_d * (delta + 4 * ctheta * distance)
            final_interp = 0. if n == 0 else (STAGE_ENVELOPE * n + TERMINAL_SPAN / 2) * delta
            rows.append({"remaining": n, "sum_D_continuations": sum_d,
                         "one_sided_allowance": integration, "two_sided_nodal_allowance": nodal,
                         "additional_final_off_grid_interpolation": final_interp,
                         "two_sided_off_grid_allowance": nodal + final_interp})
        records.append({**level, **geometry_evidence, "delta_grid": delta, "normal_transport_distance": distance,
                        "by_horizon": rows, "certified_bound": False})
    return {"theory": "THEORY.md equations (27)-(32)", "stage_expected_reward_envelope": STAGE_ENVELOPE,
            "terminal_payoff_span": TERMINAL_SPAN, "c_theta": ctheta,
            "quantity_scope": "BA table; known-model/revelation terminal errors require their own allowance",
            "certified_bound": False, "solver_rounding_allowance_included": False,
            "excludes": ["outward-rounded special functions and quadrature parameters", "solver floating-point error",
                         "approximate maximizing/tie arithmetic error"],
            "interpretation": "broad mathematical allowances evaluated numerically; not an economic near-optimality certificate",
            "rules": rules, "levels": records}


def prepare(config_path, output):
    config = _config(config_path)
    out = _new_directory(output)
    horizon, episodes = config["horizon"], EPISODES_PER_MODEL
    public = {"signal": np.empty((2, episodes, horizon + 1), np.int8),
              "return_": np.empty((2, episodes, horizon)),
              "actions": np.empty((2, episodes, horizon), np.int8),
              "fills": np.empty((2, episodes, horizon, 2), np.int8),
              "inventory": np.empty((2, episodes, horizon), np.int8),
              "joint": np.empty((2, episodes, horizon, 4))}
    seeds = []
    for model_index, kappa in enumerate(config["kappas"]):
        market_seed = [config["root_seed"], 410, 0, model_index]
        action_seed = [config["root_seed"], 410, 1, model_index]
        seeds.append({"model_index_evaluator_only": model_index, "market": market_seed, "action": action_seed})
        tape = generate_exogenous(episodes, horizon, config["theta"], kappa, market_seed)
        environment = BatchedEnvironment(tape, qmax=config["qmax"])
        rng = np.random.default_rng(np.random.SeedSequence(action_seed))
        joint = np.full((episodes, 4), .25)
        for t in range(horizon):
            observed = environment.observe()
            legal = observed.admissible_actions
            counts = legal.sum(axis=1)
            rank = np.floor(rng.random(episodes) * counts).astype(int)
            action = np.argmax(np.cumsum(legal, axis=1) > rank[:, None], axis=1).astype(np.int8)
            public["signal"][model_index, :, t] = observed.signal
            public["inventory"][model_index, :, t] = observed.inventory
            public["joint"][model_index, :, t] = joint
            feedback = environment.step(action)
            public["actions"][model_index, :, t] = action
            public["return_"][model_index, :, t] = feedback.return_
            public["fills"][model_index, :, t] = feedback.fills
            joint = update_joint(joint, feedback.signal, feedback.return_, action, feedback.fills,
                                 theta=config["theta"], kappas=tuple(config["kappas"]))
        public["signal"][model_index, :, horizon] = feedback.next_observation.signal
    probes = numerical_probes(config)
    reached_count, probe_count = 2 * episodes * horizon, len(probes["joint"])
    arbitrary_count = horizon * probe_count
    count = reached_count + arbitrary_count
    states = {
        "candidate_id": np.arange(count, dtype=np.int64),
        "source": np.r_[np.zeros(reached_count, np.int8), np.ones(arbitrary_count, np.int8)],
        "remaining": np.r_[np.tile(np.arange(horizon, 0, -1), 2 * episodes),
                            np.repeat(np.arange(1, horizon + 1), probe_count)].astype(np.int8),
        "inventory": np.r_[public["inventory"].reshape(-1), np.tile(probes["inventory"], horizon)].astype(np.int8),
        "signal": np.r_[public["signal"][:, :, :-1].reshape(-1), np.tile(probes["signal"], horizon)].astype(np.int8),
        "joint": np.concatenate((public["joint"].reshape(-1, 4), np.tile(probes["joint"], (horizon, 1)))),
        "model_index": np.r_[np.repeat(np.arange(2), episodes * horizon), np.full(arbitrary_count, -1)].astype(np.int8),
        "episode": np.r_[np.tile(np.repeat(np.arange(episodes), horizon), 2), np.full(arbitrary_count, -1)].astype(np.int16),
        "t": np.r_[np.tile(np.arange(horizon), 2 * episodes), np.full(arbitrary_count, -1)].astype(np.int8),
        "probe_index": np.r_[np.full(reached_count, -1), np.tile(np.arange(probe_count), horizon)].astype(np.int32),
    }
    _validate_states(states, config)
    np.savez_compressed(out / "exploration_public.npz", **public)
    np.savez_compressed(out / "candidate_states.npz", **states)
    return _manifest(out, "explanatory_candidates", {
        "seeds": seeds, "episodes_per_model": episodes, "reached_states": reached_count,
        "arbitrary_states": arbitrary_count, "total_states": count, "probe_count": probe_count,
        "source_names": SOURCE_NAMES, "model_labels": "evaluator provenance only; excluded from policy inputs",
        "files": [_array_record(out / name) for name in ("exploration_public.npz", "candidate_states.npz")]})


def _validate_states(states, config):
    required = {"candidate_id", "source", "remaining", "inventory", "signal", "joint",
                "model_index", "episode", "t", "probe_index"}
    if set(states) != required:
        raise ValueError("candidate array names differ from the schema")
    p = _checked_joint(states["joint"])
    count = len(p)
    if any(np.asarray(value).shape != ((count, 4) if key == "joint" else (count,)) for key, value in states.items()):
        raise ValueError("candidate dimensions do not align")
    if not np.array_equal(states["candidate_id"], np.arange(count)):
        raise ValueError("candidate IDs must be the complete unique ordered sequence")
    if not np.isin(states["source"], (0, 1)).all() or not np.isin(states["signal"], (-1, 0, 1)).all():
        raise ValueError("invalid candidate source or signal")
    admissible(states["inventory"], config["qmax"])
    if np.any((states["remaining"] < 1) | (states["remaining"] > config["horizon"])):
        raise ValueError("candidate horizon invalid")
    reached = states["source"] == 0
    if (np.sum(reached) != 2 * EPISODES_PER_MODEL * config["horizon"]
            or not np.isin(states["model_index"][reached], (0, 1)).all()
            or np.any((states["episode"][reached] < 0) | (states["episode"][reached] >= EPISODES_PER_MODEL))
            or not np.array_equal(states["remaining"][reached], config["horizon"] - states["t"][reached])):
        raise ValueError("reached-state counts or temporal provenance invalid")
    if (np.any(states["t"][~reached] != -1) or np.any(states["episode"][~reached] != -1)
            or np.any(states["model_index"][~reached] != -1)):
        raise ValueError("arbitrary states must not claim episode reachability")


def load_candidates(directory, config):
    directory = Path(directory).resolve()
    record = json.loads((directory / "manifest.json").read_text())
    signature = record.pop("artifact_sha256", None)
    if signature != canonical_hash(record):
        raise ValueError("candidate manifest identity mismatch")
    record["artifact_sha256"] = signature
    if record.get("kind") != "explanatory_candidates" or record.get("config_sha256") != CONFIG_SHA256:
        raise ValueError("wrong candidate artifact kind/configuration")
    if record.get("source") != _source_identity() or record.get("build") != build_fingerprint():
        raise ValueError("candidate source/build identity differs; reproduce into a new directory")
    if {item["file"] for item in record["files"]} != {"exploration_public.npz", "candidate_states.npz"}:
        raise ValueError("candidate artifact has missing or unexpected files")
    for item in record["files"]:
        path = directory / item["file"]
        if path.parent != directory or not path.is_file() or _array_record(path) != item:
            raise ValueError("candidate file identity or structure mismatch")
    with np.load(directory / "candidate_states.npz", allow_pickle=False) as loaded:
        states = {key: loaded[key] for key in loaded.files}
    _validate_states(states, config)
    return states, record


def _family_contract(family, config):
    if (family.spec.theta != config["theta"] or list(family.spec.kappas) != config["kappas"]
            or family.spec.horizon != config["horizon"] or family.spec.qmax != config["qmax"]):
        raise ValueError("family differs from the fixed explanatory market")


def _actions_and_gaps(scores, inventory, qmax=2):
    scores = np.asarray(scores)
    legal = admissible(inventory, qmax)
    if scores.shape != legal.shape or not np.array_equal(np.isfinite(scores), legal):
        raise ArithmeticError("legal action scores must be finite and illegal scores nonfinite")
    if np.any(~legal & ~np.isneginf(scores)):
        raise ArithmeticError("illegal action score must be negative infinity")
    chosen = stable_argmax(scores)
    ordered = np.sort(scores, axis=1)
    return chosen, ordered[:, -1] - ordered[:, -2]


def _screen(family, states):
    count = len(states["candidate_id"])
    result = {name: np.empty(count, dtype=np.int8 if name.endswith("action") else float)
              for name in ("bayes_action", "weighted_q_action", "bayes_gap", "weighted_q_gap")}
    for remaining in range(1, family.spec.horizon + 1):
        selected = np.flatnonzero(states["remaining"] == remaining)
        for start in range(0, len(selected), 2048):
            ids = selected[start:start + 2048]
            q, x, p = (states[key][ids] for key in ("inventory", "signal", "joint"))
            for kind in ("bayes", "weighted_q"):
                actions, gaps = _actions_and_gaps(family.q_values(kind, remaining, q, x, p), q)
                result[kind + "_action"][ids] = actions
                result[kind + "_gap"][ids] = gaps
    shortlist = []
    summaries = []
    for source in (0, 1):
        candidates = np.flatnonzero((states["source"] == source)
                                   & (result["bayes_action"] != result["weighted_q_action"]))
        quality = np.minimum(result["bayes_gap"][candidates], result["weighted_q_gap"][candidates])
        ordered = candidates[np.lexsort((candidates, -quality))]
        unique, seen = [], set()
        for candidate in ordered:
            state_key = (int(states["remaining"][candidate]), int(states["inventory"][candidate]),
                         int(states["signal"][candidate]), states["joint"][candidate].tobytes())
            if state_key not in seen:
                seen.add(state_key)
                unique.append(int(candidate))
        chosen = unique[:DIRECT_LIMIT_PER_SOURCE]
        shortlist.extend(chosen)
        summaries.append({"source": SOURCE_NAMES[source], "screened": int(np.sum(states["source"] == source)),
                          "action_disagreements": len(candidates), "unique_disagreements": len(unique),
                          "directly_examined": len(chosen), "direct_limit": DIRECT_LIMIT_PER_SOURCE})
    return result, np.asarray(shortlist, dtype=np.int64), summaries


def _direct_scores(family, states, ids):
    values = {name: np.empty((len(ids), 11)) for name in SCORE_NAMES}
    for remaining in sorted(set(states["remaining"][ids].tolist())):
        positions = np.flatnonzero(states["remaining"][ids] == remaining)
        for start in range(0, len(positions), 64):
            pos = positions[start:start + 64]
            subset = ids[pos]
            q, x, p = (states[key][subset] for key in ("inventory", "signal", "joint"))
            values["table_bayes"][pos] = family.q_values("bayes", remaining, q, x, p)
            values["weighted_q"][pos] = family.q_values("weighted_q", remaining, q, x, p)
            for name, mode in (("direct_bayes", "bayes"),
                               ("same_continuation_frozen_model", "frozen_model"),
                               ("same_continuation_no_feedback", "no_feedback")):
                values[name][pos] = family.bellman_q("bayes", remaining, q, x, p, update_mode=mode)
    actions, gaps = {}, {}
    for name, scores in values.items():
        actions[name], gaps[name] = _actions_and_gaps(scores, states["inventory"][ids])
    return {"scores": values, "actions": actions, "gaps": gaps}


def _qualifications(bundle, model_information):
    actions, gaps = bundle["actions"], bundle["gaps"]
    count = len(actions["direct_bayes"])
    checks = {
        "direct_matches_deployed_bayes": actions["direct_bayes"] == actions["table_bayes"],
        "bayes_differs_from_weighted_q": actions["direct_bayes"] != actions["weighted_q"],
        "frozen_update_reverses_to_weighted_q": actions["same_continuation_frozen_model"] == actions["weighted_q"],
        "bayes_margin_at_least_001": gaps["direct_bayes"] >= ROBUST_GAP,
        "frozen_margin_at_least_001": gaps["same_continuation_frozen_model"] >= ROBUST_GAP,
        "weighted_q_margin_at_least_001": gaps["weighted_q"] >= ROBUST_GAP,
        "positive_model_information_for_bayes_action": model_information[np.arange(count), actions["direct_bayes"]] > INFORMATION_THRESHOLD,
    }
    return {"checks": checks, "qualifies": np.logical_and.reduce(list(checks.values())),
            "minimum_direct_margin": np.minimum.reduce([
                gaps["direct_bayes"], gaps["same_continuation_frozen_model"], gaps["weighted_q"]])}


def _provenance(states, candidate):
    return {"candidate_id": int(candidate), "source": SOURCE_NAMES[int(states["source"][candidate])],
            "remaining": int(states["remaining"][candidate]), "inventory": int(states["inventory"][candidate]),
            "signal": int(states["signal"][candidate]), "joint": states["joint"][candidate].tolist(),
            "coordinates": joint_to_coordinates(states["joint"][candidate]).tolist(),
            "model_index_evaluator_only": int(states["model_index"][candidate]),
            "episode": int(states["episode"][candidate]), "t": int(states["t"][candidate]),
            "probe_index": int(states["probe_index"][candidate]),
            "reachability": ("Recorded public prefix under uniform safe exploration; no scored-policy prevalence claim"
                             if states["source"][candidate] == 0 else
                             "Arbitrary joint prior; reachability from the initial prior is not established")}


def _illustrative_posteriors(state, actions, config):
    """Fixed z illustrations, not probabilities assigned to a discretized market."""
    records = []
    x, p = state["signal"], np.asarray(state["joint"])
    for action in sorted(set(map(int, actions))):
        submitted = np.array([False, False]) if action >= 9 else ACTIONS[action] >= 0
        side_indices = np.flatnonzero(submitted)
        for z in (-2., -1., 0., 1., 2.):
            for bits in range(1 << len(side_indices)):
                fills = np.full(2, -1, dtype=np.int8)
                for bit, side_index in enumerate(side_indices):
                    fills[side_index] = (bits >> bit) & 1
                common = dict(joint=p[None, :], signal=np.array([x]),
                              return_=np.array([MU * x + SIGMA * z]), actions=np.array([action]),
                              fills=fills[None, :], theta=config["theta"], kappas=tuple(config["kappas"]))
                actual = update_joint(**common)[0]
                frozen = update_joint(**common, mode="frozen_model")[0]
                records.append({"action": action, "z": z, "selected_fills": fills.tolist(),
                                "actual_next_joint": actual.tolist(), "frozen_weight_next_joint": frozen.tolist(),
                                "actual_next_coordinates": joint_to_coordinates(actual).tolist(),
                                "frozen_next_coordinates": joint_to_coordinates(frozen).tolist()})
    return {"scope": "fixed hypothetical public observations; no realized PnL selection or probability mass at these z values",
            "branches": records}


def _fixed_state_checks(families, config):
    points, q, x, remaining = [], [], [], []
    for n in (1, 2, 5, 15, 30):
        for inventory, signal in ((-2, -1), (0, 0), (2, 1)):
            for w in (0., .5, 1.):
                for b in (.2, .5, .8):
                    points.append(coordinates_to_joint(np.array([w, b, b])))
                    q.append(inventory); x.append(signal); remaining.append(n)
    equal_count = len(points)
    probes = numerical_probes(config)
    for i in range(16):
        points.append(probes["joint"][i]); q.append(probes["inventory"][i])
        x.append(probes["signal"][i]); remaining.append(1)
    states = {"joint": np.asarray(points), "inventory": np.asarray(q), "signal": np.asarray(x),
              "remaining": np.asarray(remaining)}
    info = information_scores(states["signal"], states["joint"])
    records = []
    for label, family in families:
        scores = _direct_scores(family, states, np.arange(len(points)))
        legal = admissible(states["inventory"], config["qmax"])
        enabled = scores["scores"]["direct_bayes"]
        frozen = scores["scores"]["same_continuation_frozen_model"]
        no_feedback = scores["scores"]["same_continuation_no_feedback"]
        equal_mask = legal.copy(); equal_mask[equal_count:] = False
        final_mask = legal & (states["remaining"] == 1)[:, None]
        fm_error = float(np.max(np.abs(enabled[equal_mask] - frozen[equal_mask])))
        final_error = max(float(np.max(np.abs(enabled[final_mask] - mode[final_mask])))
                          for mode in (frozen, no_feedback))
        entry = {"family": label, "states": len(points), "equal_conditional_states": equal_count,
                 "max_equal_conditional_frozen_difference": fm_error,
                 "max_final_horizon_suppression_difference": final_error,
                 "max_equal_conditional_model_information": float(info["model_information"][:equal_count].max()),
                 "max_information_chain_residual": float(np.abs(info["chain_identity_residual"]).max()),
                 "max_mass_error": float(np.abs(info["mass"] - 1.).max())}
        entry["passed"] = (fm_error <= 1e-10 and final_error <= 1e-10
                           and entry["max_equal_conditional_model_information"] <= 1e-10
                           and entry["max_information_chain_residual"] <= 1e-10
                           and entry["max_mass_error"] <= 1e-11)
        records.append(entry)
    return {"by_family": records, "passed": all(item["passed"] for item in records)}


def _numerical_selection_evidence(numerical_checks, selected):
    """Bind a claimed numerical pass before any candidate analysis or output."""
    if numerical_checks is None:
        return None
    path = Path(numerical_checks).resolve()
    receipt_bytes = path.read_bytes()
    loaded = json.loads(receipt_bytes)
    if not isinstance(loaded, dict) or type(loaded.get("passes_predeclared_rule")) is not bool:
        raise ValueError("numerical receipt requires a Boolean acceptance flag")
    passed = loaded["passes_predeclared_rule"]
    bindings = None
    if passed:
        bindings = {
            "selected_artifact": loaded.get("selected_artifact_sha256") == selected.metadata["artifact_sha256"],
            "selected_specification": loaded.get("selected_resolution") == _jsonable(asdict(selected.spec)),
            "solver_source": loaded.get("source") == selected.metadata["source"],
            "numerical_build": loaded.get("build") == selected.metadata["build"],
        }
        failed = [name for name, matches in bindings.items() if not matches]
        if failed:
            raise ValueError("numerical acceptance does not authenticate selected family: " + ", ".join(failed))
    return {"file": str(path), "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "passes_predeclared_rule": passed,
            "selected_resolution": loaded.get("selected_resolution"),
            "selected_artifact_sha256": loaded.get("selected_artifact_sha256"),
            "selected_family_artifact_sha256": selected.metadata["artifact_sha256"],
            "acceptance_bound_to_selected_family": passed,
            "binding_checks": bindings,
            "binding_status": "accepted_family_authenticated" if passed else "unresolved_receipt_not_acceptance",
            "certified_bound": False}


def analyze(config_path, candidates_directory, family_directory, comparison_directories, output, numerical_checks=None):
    started = time.perf_counter()
    config = _config(config_path)
    selected = load_family(family_directory)
    _family_contract(selected, config)
    numerical_record = _numerical_selection_evidence(numerical_checks, selected)
    states, candidate_manifest = load_candidates(candidates_directory, config)
    families = [("selected", selected)]
    for i, directory in enumerate(comparison_directories):
        family = load_family(directory)
        _family_contract(family, config)
        if any(family.metadata["artifact_sha256"] == other.metadata["artifact_sha256"] for _, other in families):
            raise ValueError("comparison families must be distinct authenticated artifacts")
        families.append((f"comparison_{i + 1}", family))
    out = _new_directory(output)
    screen, ids, traversal_summary = _screen(selected, states)
    examined = np.zeros(len(states["candidate_id"]), dtype=bool); examined[ids] = True
    columns = ["candidate_id", "source", "remaining", "inventory", "signal", "p0", "p1", "p2", "p3",
               "model_index_evaluator_only", "episode", "t", "probe_index", "bayes_action", "weighted_q_action",
               "bayes_gap", "weighted_q_gap", "directly_examined"]
    with (out / "traversal.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for i in range(len(states["candidate_id"])):
            writer.writerow({"candidate_id": i, "source": SOURCE_NAMES[int(states["source"][i])],
                             **{key: int(states[key][i]) for key in ("remaining", "inventory", "signal", "episode", "t", "probe_index")},
                             **{f"p{k}": float(states["joint"][i, k]) for k in range(4)},
                             "model_index_evaluator_only": int(states["model_index"][i]),
                             **{key: screen[key][i].item() for key in screen}, "directly_examined": bool(examined[i])})
    fixed_checks = _fixed_state_checks(families, config)
    _write_json(out / "fixed_state_checks.json", fixed_checks)
    rewards = analytic_rewards(states["inventory"][ids], states["signal"][ids], states["joint"][ids])
    info = information_scores(states["signal"][ids], states["joint"][ids])
    bundles, qualifications = {}, {}
    for label, family in families:
        bundles[label] = _direct_scores(family, states, ids)
        qualifications[label] = _qualifications(bundles[label], info["model_information"])
    stable = qualifications["selected"]["qualifies"].copy()
    if len(families) == 1:
        stable[:] = False
    for label, _ in families[1:]:
        stable &= qualifications[label]["qualifies"]
        for name in ("direct_bayes", "same_continuation_frozen_model", "weighted_q"):
            stable &= bundles[label]["actions"][name] == bundles["selected"]["actions"][name]
    legal = admissible(states["inventory"][ids], config["qmax"])
    one_period = np.empty_like(rewards)
    for start in range(0, len(ids), 64):
        subset = ids[start:start + 64]
        one_period[start:start + 64] = selected.bellman_q(
            "bayes", 1, states["inventory"][subset], states["signal"][subset], states["joint"][subset])
    all_entries = []
    with (out / "direct_candidates.jsonl").open("w") as json_stream, (out / "direct_scores.csv").open("w", newline="") as csv_stream:
        score_columns = ["candidate_id", "family", "action", "legal", "immediate_expected_reward", "one_period_q",
                         *SCORE_NAMES, "enabled_continuation", "frozen_weight_continuation", "model_update_score_increment",
                         "model_information", "conditional_regime_information", "regime_information"]
        writer = csv.DictWriter(csv_stream, fieldnames=score_columns); writer.writeheader()
        for pos, candidate in enumerate(ids):
            entry = {**_provenance(states, candidate), "legal_actions": legal[pos].tolist(),
                     "immediate_expected_reward": rewards[pos].tolist(), "one_period_q": one_period[pos].tolist(),
                     "information_quadrature_points": INFORMATION_QUADRATURE,
                     "information": {key: value[pos].tolist() for key, value in info.items()},
                     "refinement_stable_witness": bool(stable[pos]), "families": {}}
            for label, family in families:
                bundle, qualification = bundles[label], qualifications[label]
                model_increment = np.full(11, np.nan)
                model_increment[legal[pos]] = (bundle["scores"]["direct_bayes"][pos, legal[pos]]
                                               - bundle["scores"]["same_continuation_frozen_model"][pos, legal[pos]])
                differences = {name: float(np.max(np.abs(bundle["scores"][name][pos, legal[pos]]
                                                         - bundles["selected"]["scores"][name][pos, legal[pos]])))
                               for name in SCORE_NAMES}
                entry["families"][label] = {
                    "artifact_sha256": family.metadata["artifact_sha256"], "specification": asdict(family.spec),
                    "scores": {name: value[pos].tolist() for name, value in bundle["scores"].items()},
                    "actions": {name: int(value[pos]) for name, value in bundle["actions"].items()},
                    "gaps": {name: float(value[pos]) for name, value in bundle["gaps"].items()},
                    "qualification_checks": {name: bool(value[pos]) for name, value in qualification["checks"].items()},
                    "qualifies": bool(qualification["qualifies"][pos]),
                    "minimum_direct_margin": float(qualification["minimum_direct_margin"][pos]),
                    "model_update_score_increment": model_increment.tolist(),
                    "max_finite_score_change_from_selected": differences}
                for action in range(11):
                    enabled_continuation = frozen_continuation = increment = None
                    if legal[pos, action]:
                        enabled_continuation = bundle["scores"]["direct_bayes"][pos, action] - rewards[pos, action]
                        frozen_continuation = bundle["scores"]["same_continuation_frozen_model"][pos, action] - rewards[pos, action]
                        increment = enabled_continuation - frozen_continuation
                    writer.writerow(_jsonable({"candidate_id": int(candidate), "family": label, "action": action,
                                               "legal": bool(legal[pos, action]), "immediate_expected_reward": rewards[pos, action],
                                               "one_period_q": one_period[pos, action],
                                               **{name: value[pos, action] for name, value in bundle["scores"].items()},
                                               "enabled_continuation": enabled_continuation,
                                               "frozen_weight_continuation": frozen_continuation,
                                               "model_update_score_increment": increment,
                                               **{name: info[name][pos, action] for name in (
                                                   "model_information", "conditional_regime_information", "regime_information")}}))
            all_entries.append(entry)
            json_stream.write(json.dumps(_jsonable(entry), sort_keys=True, allow_nan=False) + "\n")
    winner = None
    if len(ids):
        available = np.flatnonzero(stable) if np.any(stable) else np.arange(len(ids))
        winner_position = sorted(available, key=lambda pos: (
            int(states["source"][ids[pos]]), -qualifications["selected"]["minimum_direct_margin"][pos], int(ids[pos])))[0]
        winner = all_entries[winner_position]
        chosen = bundles["selected"]["actions"]
        winner["illustrative_posteriors"] = _illustrative_posteriors(
            winner, [chosen["direct_bayes"][winner_position], chosen["weighted_q"][winner_position]], config)
        winner["selection_status"] = "refinement_stable_witness" if stable[winner_position] else "strongest_checked_nonqualifying_case"
    specifications = [{**asdict(family.spec), "label": label, "artifact_sha256": family.metadata["artifact_sha256"]}
                      for label, family in families]
    envelope = error_envelopes(config, specifications)
    _write_json(out / "error_envelopes.json", envelope)
    if numerical_record is not None and file_sha256(numerical_record["file"]) != numerical_record["sha256"]:
        raise ValueError("numerical receipt changed during explanatory analysis")
    record = {"design_version": DESIGN_VERSION, "search": traversal_summary,
              "candidate_manifest_sha256": candidate_manifest["artifact_sha256"],
              "direct_candidates": len(ids), "selected_resolution_qualifiers": int(qualifications["selected"]["qualifies"].sum()),
              "refinement_stable_witnesses": int(stable.sum()),
              "comparison_families_supplied": len(families) - 1,
              "refinement_stability_status": "checked_on_supplied_families" if len(families) > 1 else "unresolved_no_comparison_family",
              "fixed_state_checks": fixed_checks, "numerical_selection_evidence": numerical_record,
              "winner": winner, "family_specifications": specifications, "certified_bound": False,
              "interpretation": "sensitivity to marginal-model posterior updates in planning; not a clean causal removal of model information",
              "search_limit": "Direct checks examine the fixed top 256 disagreements per source after table screening; null does not exclude other states",
              "elapsed_seconds": time.perf_counter() - started}
    _write_json(out / "explanation.json", record)
    _write_explanation(out / "EXPLANATION.md", record, envelope)
    files = [{"file": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
             for path in sorted(out.iterdir()) if path.is_file()]
    manifest = _manifest(out, "explanatory_analysis", {"candidate_artifact_sha256": candidate_manifest["artifact_sha256"],
                                                    "family_specifications": specifications, "files": files,
                                                    "fixed_state_checks_passed": fixed_checks["passed"]})
    if not fixed_checks["passed"]:
        raise ArithmeticError("fixed-state explanatory checks failed; adverse artifact was retained")
    return manifest


def _write_explanation(path, record, envelopes):
    total = sum(item["screened"] for item in record["search"])
    stable = record["refinement_stable_witnesses"]
    lines = ["# Fixed-state update-dependence analysis", "",
             f"The fixed search screened {total:,} candidate states and directly examined {record['direct_candidates']} disagreements.",
             f"It found **{stable} refinement-stable qualifying witnesses** under the prespecified .001 action-margin rule.", "",
             "These are numerical action diagnostics. The intervention freezes marginal model weights while retaining conditional regime updates; it is not a coherent deletion of only model information.", "",
             "| Candidate source | Screened | BA/WQ disagreements | Distinct disagreements | Direct checks |",
             "|---|---:|---:|---:|---:|"]
    for row in record["search"]:
        lines.append(f"| {row['source']} | {row['screened']} | {row['action_disagreements']} | {row['unique_disagreements']} | {row['directly_examined']} |")
    lines += ["", record["search_limit"] + ".", "",
              "Cross-family stability: " + record["refinement_stability_status"] + ". No reported action margin is an interval-certified true Bellman gap."]
    winner = record["winner"]
    if winner:
        selected = winner["families"]["selected"]
        actions = selected["actions"]
        lines += ["", f"## Retained case {winner['candidate_id']}", "",
                  f"Status: **{winner['selection_status']}**. {winner['reachability']}.", "",
                  f"State: n={winner['remaining']}, q={winner['inventory']}, x={winner['signal']}, joint p={winner['joint']}.", "",
                  "| Quantity | BA choice | Weighted-Q choice |", "|---|---:|---:|"]
        for name, values in (("Action ID", None), ("Immediate expected reward", winner["immediate_expected_reward"]),
                             ("One-period score including liquidation", winner["one_period_q"]),
                             ("Model information (nats)", winner["information"]["model_information"]),
                             ("Conditional regime information (nats)", winner["information"]["conditional_regime_information"])):
            ba, wq = actions["direct_bayes"], actions["weighted_q"]
            a, b = (ba, wq) if values is None else (values[ba], values[wq])
            lines.append(f"| {name} | {a:.10g} | {b:.10g} |")
        lines += ["", f"The same-BA-continuation frozen-weight action is {actions['same_continuation_frozen_model']}. Complete legal-action scores and every failed qualification are retained in direct_candidates.jsonl and direct_scores.csv."]
        if not stable:
            lines += ["", "This retained case does not satisfy the complete witness criterion. The fixed search therefore supplies no qualified action-level demonstration; it does not rule out smaller effects or witnesses omitted by the declared screen/cap."]
    else:
        lines += ["", "No table-screened BA/weighted-Q disagreement was available for a direct witness. This is a negative result for the specified candidate set."]
    final = envelopes["levels"][0]["by_horizon"][-1]
    lines += ["", "## Numerical scope", "",
              f"For the selected family, the broad theorem allowance at T=30 is {final['one_sided_allowance']:.6g} objective units on the one-sided BA comparison, {final['two_sided_nodal_allowance']:.6g} on grid nodes, and {final['two_sided_off_grid_allowance']:.6g} for a final off-grid value query.",
              "These are ordinary floating evaluations of conservative mathematical formulas, with no solver-rounding or interval-arithmetic certificate. They do not establish an economic optimality gap of .002. Observed cross-cache score changes are a separate empirical diagnostic.", "",
              "Cold/equal-conditional-belief and terminal suppression identities: " + ("passed." if record["fixed_state_checks"]["passed"] else "FAILED; retain the adverse receipt."), "",
              "All reached states come from separate namespace-410 public exploration, with no final-evaluation PnL used in selection. Parent campaign results are needed to assess the population economic contrast."]
    Path(path).write_text("\n".join(lines) + "\n")


def diagnostics(config_path, output):
    """Small E2E mathematical artifact and declared rejection probes; no solve."""
    config = _config(config_path)
    out = _new_directory(output)
    p = coordinates_to_joint(np.array([[.5, .5, .5], [.2, .3, .3], [0., .2, .7],
                                       [1., .3, .9], [.5, .1, .9], [.3, .85, .2]]))
    x = np.array([0, -1, 1, 0, -1, 1]); q = np.array([0, -2, 2, 0, -1, 1])
    info = information_scores(x, p)
    reward = analytic_rewards(q, x, p)
    spec = SolverSpec()
    reference = expected_rewards(spec, joint_to_coordinates(p))[q + 2, x + 1, np.arange(len(p))]
    legal = admissible(q, 2)
    n1 = normal_transport_distance(np.array([0.]), np.array([1.]))
    nodes, weights = roots_hermitenorm(3); weights /= math.sqrt(2 * math.pi)
    transport = normal_transport_distance(nodes, weights)
    cuts = ndtri(np.r_[0., np.cumsum(weights)[:-1], 1.])
    independent_transport = sum(quad(lambda z, node=node: abs(z - node) * np.exp(-z*z/2) / math.sqrt(2*math.pi),
                                     float(a), float(b), epsabs=1e-11)[0]
                                for a, b, node in zip(cuts[:-1], cuts[1:], nodes))
    checks = {
        "one_node_transport_error": abs(n1["distance"] - math.sqrt(2 / math.pi)),
        "three_node_independent_integration_error": abs(transport["distance"] - independent_transport),
        "analytic_reward_reference_error": float(np.max(np.abs(reward[legal] - reference[legal]))),
        "information_mass_error": float(np.max(np.abs(info["mass"] - 1))),
        "information_chain_error": float(np.max(np.abs(info["chain_identity_residual"]))),
        "equal_conditional_model_information": float(info["model_information"][:2].max()),
        "singleton_model_information": float(info["model_information"][2:4].max()),
        "no_quote_market_information": float(max(info[key][:, [0, 9, 10]].max() for key in (
            "model_information", "conditional_regime_information", "regime_information"))),
        "correlated_belief_quote_model_information": float(info["model_information"][4:, 4].min()),
    }
    faults = [
        ("non_normalized_joint", lambda: information_scores(np.array([0]), np.full((1, 4), .3))),
        ("invalid_public_signal", lambda: information_scores(np.array([2]), np.full((1, 4), .25))),
        ("negative_quadrature_weight", lambda: normal_transport_distance(np.array([-1., 1.]), np.array([-.1, 1.1]))),
        ("unsorted_quadrature_nodes", lambda: normal_transport_distance(np.array([1., -1.]), np.array([.5, .5]))),
        ("unnormalized_quadrature", lambda: normal_transport_distance(np.array([0.]), np.array([2.]))),
        ("unsafe_inventory", lambda: analytic_rewards(np.array([3]), np.array([0]), np.full((1, 4), .25))),
    ]
    rejections = []
    for name, operation in faults:
        try:
            operation()
        except (ValueError, ArithmeticError) as error:
            rejections.append({"fault": name, "rejected": True, "reason": str(error)})
        else:
            rejections.append({"fault": name, "rejected": False})
    passed = (all(value <= 1e-9 for key, value in checks.items() if key != "correlated_belief_quote_model_information")
              and checks["correlated_belief_quote_model_information"] > INFORMATION_THRESHOLD
              and all(row["rejected"] for row in rejections))
    _write_json(out / "diagnostics.json", {"checks": checks, "rejections": rejections, "passed": passed,
                                          "certified_bound": False, "information": info, "reward": reward})
    _write_json(out / "error_envelopes.json", error_envelopes(config))
    manifest = _manifest(out, "explanation_diagnostics", {
        "passed": passed, "files": [{"file": name, "bytes": (out / name).stat().st_size,
                                      "sha256": file_sha256(out / name)}
                                     for name in ("diagnostics.json", "error_envelopes.json")]})
    if not passed:
        raise ArithmeticError("explanatory mathematical diagnostics failed; receipt retained")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "diagnostics", "envelopes"):
        child = sub.add_parser(command)
        child.add_argument("--output", type=Path, required=True)
    child = sub.add_parser("analyze")
    child.add_argument("--candidates", type=Path, required=True)
    child.add_argument("--family", type=Path, required=True)
    child.add_argument("--comparison", action="append", type=Path, default=[])
    child.add_argument("--numerical-checks", type=Path)
    child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.config, args.output)
    elif args.command == "diagnostics":
        result = diagnostics(args.config, args.output)
    elif args.command == "envelopes":
        out = _new_directory(args.output)
        _write_json(out / "error_envelopes.json", error_envelopes(_config(args.config)))
        result = {"output": str(out), "certified_bound": False}
    else:
        result = analyze(args.config, args.candidates, args.family, args.comparison, args.output, args.numerical_checks)
    print(json.dumps({key: result.get(key) for key in (
        "kind", "artifact_sha256", "passed", "total_states", "output", "fixed_state_checks_passed")
                      if key in result}, sort_keys=True))


if __name__ == "__main__":
    main()
