"""Prespecified paired pilot/evaluation campaign for the Bayes extension.

Failure modes and protocol were recorded at 8db6208 before this implementation.
The independent verifier imports none of this producer's inference/statistics.
All large control tables are supplied separately as authenticated artifacts.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
CORE = Path(os.environ.get("TRADE_LEARNING_CORE_ROOT", REPOSITORY)).resolve()
if not (CORE / "src/trade_learning/model.py").is_file():
    raise RuntimeError("Set TRADE_LEARNING_CORE_ROOT to the research source root")
sys.path.insert(0, str(CORE / "src"))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import t as student_t

from trade_learning.environment import BatchedEnvironment, generate_exogenous, generate_uniform_pilot
from trade_learning.filtering import filter_step, fit_grid_posterior
from trade_learning.numerics import build_fingerprint, file_sha256, numerical_contract, stable_argmax
from solver import load_family

CONFIG_SHA256 = "8681e5a8c48f25967f95a4b7d42b76531cbc4e8e00d013114c3e6efe93aee462"
PROTOCOL_COMMIT = "8db62080ef4e5170413d01b7f5fdac33bfd2b1e5"
CORE_SOURCE_HASH = "bfe4d3aebcd0f34a3d7fdced84bcc36b0e10103f24313da4f1dff573e1a868a1"
FEASIBLE = ("bayes", "weighted_q", "no_feedback", "frozen_model")
KEYS = ["model_index", "replicate", "episode", "budget", "policy"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_identity():
    core_files = sorted((CORE / "src/trade_learning").glob("*.py"))
    aggregate = hashlib.sha256(b"".join(p.name.encode()+p.read_bytes() for p in core_files)).hexdigest()
    require(aggregate == CORE_SOURCE_HASH, "The verified original production source changed")
    return {"core": {str(p.relative_to(CORE)): file_sha256(p) for p in core_files},
            "extension": {name: file_sha256(HERE / name) for name in
                          ("run.py", "solver.py", "config.json", "PROTOCOL.md",
                           "NUMERICAL_AMENDMENT.md", "numerical_config.json", "refine_numerical.py",
                           "GEOMETRY_AMENDMENT.md", "refine_geometry.py")}}


def read_protocol():
    require(file_sha256(HERE / "config.json") == CONFIG_SHA256,
            "Unsupported change to the frozen extension configuration")
    return json.loads((HERE / "config.json").read_text())


def validate_numerical_protocol(numerical, config):
    geometry = numerical.get("belief_geometry", "uniform")
    require(geometry in ("uniform", "endpoint_sine"), "Unsupported belief geometry")
    require(numerical.get("selected_resolution", {}).get("belief_geometry", "uniform") == geometry,
            "Selected geometry differs from the numerical acceptance receipt")
    if geometry == "endpoint_sine":
        require(numerical.get("geometry_amendment_sha256") == file_sha256(HERE / "GEOMETRY_AMENDMENT.md") and
                numerical.get("geometry_wrapper_sha256") == file_sha256(HERE / "refine_geometry.py"),
                "Endpoint geometry requires its explicit authenticated amendment")
    if numerical.get("protocol_config_sha256") == CONFIG_SHA256:
        require(geometry == "uniform", "Endpoint geometry must use the explicitly amended numerical ladder")
        return
    path = HERE / "numerical_config.json"
    amended = json.loads(path.read_text())
    expected = {**config, "joint_grid_levels": config["joint_grid_levels"] + [[49, 97, 97]]}
    require(amended == expected, "Unsupported numerical-only amendment")
    digest = file_sha256(path)
    require(numerical.get("protocol_config_sha256") == digest and
            numerical.get("numerical_config_sha256") == digest and
            numerical.get("statistical_protocol_sha256") == CONFIG_SHA256 and
            numerical.get("numerical_amendment_sha256") == file_sha256(HERE / "NUMERICAL_AMENDMENT.md") and
            numerical.get("numerical_wrapper_sha256") == file_sha256(HERE / "refine_numerical.py"),
            "Numerical receipt does not bind the explicit pre-evaluation amendment")


def stream_seed(config, profile, namespace, model, replicate):
    values = [config["root_seed"], namespace, model, replicate]
    return values if profile == "full" else [config["root_seed"], 510, namespace, model, replicate]


class Posterior:
    """Exact public-data filter; fresh H prior in every evaluation episode."""
    def __init__(self, initial_log_weights, episodes, config):
        self.log_weights = np.broadcast_to(initial_log_weights, (episodes, 2)).copy()
        self.beliefs = np.full((episodes, 2), .5)
        self.theta = np.repeat(config["theta"], 2)
        self.kappas = np.array(config["kappas"])

    @property
    def joint(self):
        weights = np.exp(self.log_weights)
        return np.stack((weights*(1-self.beliefs), weights*self.beliefs), axis=-1).reshape(-1, 4)

    def update(self, feedback):
        self.beliefs, self.log_weights, _ = filter_step(
            self.beliefs, self.log_weights, feedback.signal, feedback.return_,
            feedback.action, feedback.fills, self.theta, self.kappas)


class FeasiblePolicy:
    def __init__(self, kind, family, initial_log_weights, episodes, config):
        require(kind in FEASIBLE, "Unknown feasible policy")
        self.kind, self.family = kind, family
        self.posterior = Posterior(initial_log_weights, episodes, config)
        self.choice_seconds = 0.

    @property
    def decision_joint(self):
        return self.posterior.joint

    def choose(self, observation):
        started = time.perf_counter()
        result = self.family.actions(self.kind, observation.horizon-observation.t,
                                     observation.inventory, observation.signal, self.decision_joint)
        self.choice_seconds += time.perf_counter()-started
        return result


class KnownParameterReference:
    """Only this evaluator reference receives a model label; no hidden regime."""
    def __init__(self, model, family, initial_log_weights, episodes, config):
        self.model, self.family = model, family
        self.posterior = Posterior(initial_log_weights, episodes, config)
        self.choice_seconds = 0.

    @property
    def decision_joint(self):
        result = np.zeros_like(self.posterior.joint)
        belief = self.posterior.beliefs[:, self.model]
        result[:, 2*self.model] = 1-belief
        result[:, 2*self.model+1] = belief
        return result

    def choose(self, observation):
        started = time.perf_counter()
        scores = self.family.known_q_values(self.model, observation.horizon-observation.t,
                                           observation.inventory, observation.signal,
                                           self.decision_joint)
        result = stable_argmax(scores, observation.admissible_actions, axis=1).astype(np.int8)
        self.choice_seconds += time.perf_counter()-started
        return result


def mixture_estimate(pilot_means, alpha=.05, family_size=1):
    """Equal two-model mixture of independent replicate means, Welch interval."""
    a = np.asarray(pilot_means, dtype=float)
    require(a.ndim == 2 and a.shape[0] == 2 and a.shape[1] >= 2 and np.isfinite(a).all(),
            "Inference requires both model strata and at least two finite independent replicates")
    n = a.shape[1]
    components = a.var(axis=1, ddof=1)/(4*n)
    variance = float(components.sum())
    df = float(variance**2 / np.sum(components**2/(n-1))) if variance > 0 else float(2*(n-1))
    se, mean = float(np.sqrt(variance)), float(a.mean())
    radius = float(student_t.ppf(1-alpha/2, df))*se
    adjusted = float(student_t.ppf(1-alpha/(2*family_size), df))*se
    return {"mean": mean, "se": se, "df": df, "low": mean-radius, "high": mean+radius,
            "simultaneous_low": mean-adjusted, "simultaneous_high": mean+adjusted,
            "stratum_means": a.mean(axis=1).tolist(), "independent_replicates_per_model": n,
            "zero_empirical_variance": bool(variance == 0)}


def summarize(out, config, replicates, episodes):
    use = KEYS + ["objective", "pnl", "inventory_penalty", "inventory_sq_sum",
                  "passive_fees", "market_fees", "liquidation_cost", "model_entropy_final",
                  "true_model_probability_final", "model_entropy_initial"] + [f"action_{a}" for a in range(11)]
    frame = pd.read_csv(out / "episodes.csv.gz", usecols=use, float_precision="round_trip")
    require(not frame.duplicated(KEYS).any(), "Duplicate or broken paired outcome keys")
    count = 2*replicates*episodes*len(config["pilot_budgets"])*len(config["policies"])
    require(len(frame) == count, "Incomplete outcome population")
    require(np.isfinite(frame.drop(columns="policy").to_numpy()).all(), "Nonfinite economic output")
    expected = pd.MultiIndex.from_product([range(2), range(replicates), range(episodes),
                                          config["pilot_budgets"], config["policies"]], names=KEYS)
    indexed = frame.set_index(KEYS)
    require(indexed.index.sort_values().equals(expected.sort_values()), "Missing model/replicate/budget/policy/episode")
    contrasts, means = [], []
    for budget in config["pilot_budgets"]:
        for metric in config["metrics"]:
            pivot = frame[frame.budget == budget].pivot(index=["model_index", "replicate", "episode"],
                                                      columns="policy", values=metric).sort_index()
            require(list(pivot.index.names) == ["model_index", "replicate", "episode"] and
                    len(pivot) == 2*replicates*episodes and not pivot.isna().any().any(), "Broken policy pairing")
            for policy, baseline in config["contrasts"]:
                delta = (pivot[policy]-pivot[baseline]).groupby(level=[0, 1]).mean().to_numpy().reshape(2, replicates)
                estimate = mixture_estimate(delta, config["primary"]["alpha"], config["supplementary_family_size"])
                contrasts.append({"budget": budget, "policy": policy, "baseline": baseline,
                                  "metric": metric, **estimate})
            for policy in config["policies"]:
                pm = pivot[policy].groupby(level=[0, 1]).mean().to_numpy().reshape(2, replicates)
                means.append({"budget": budget, "policy": policy, "metric": metric,
                              **mixture_estimate(pm, config["primary"]["alpha"])})
    primary = next(r for r in contrasts if all(r[k] == config["primary"][k] for k in
                                              ("budget", "policy", "baseline", "metric")))
    threshold = config["primary"]["economic_threshold"]
    summary = {"rows": len(frame), "paired_market_trajectories": 2*replicates*episodes,
               "independent_pilot_replicates": 2*replicates, "primary": primary,
               "economic_threshold": threshold,
               "primary_empirical_lower_exceeds_threshold": primary["low"] > threshold,
               "contrasts": contrasts, "policy_means": means,
               "supplementary_family_size": config["supplementary_family_size"],
               "uncertainty": "Paired episode differences averaged within independent pilots; equal two-model mixture; empirical Welch intervals; 24 simultaneous supplementary contrasts",
               "scope": "Same market trajectories are reused across budgets and policies. Numerical approximation uncertainty is separate from these sampling intervals."}
    diagnostics = frame.groupby(["budget", "policy", "model_index"], sort=True).mean(numeric_only=True).reset_index()
    diagnostics.to_csv(out / "diagnostics.csv", index=False, float_format="%.17g")
    write_json(out / "summary.json", summary)
    return summary


def pilot_and_plan(config, family, model, replicate, profile, out):
    ns = config["seed_namespaces"]
    seed = stream_seed(config, profile, ns["pilot_market"], model, replicate)
    policy_seed = stream_seed(config, profile, ns["pilot_actions"], model, replicate)
    pilot = generate_uniform_pilot(episodes=max(config["pilot_budgets"]), horizon=config["horizon"],
                                  theta=config["theta"], kappa=config["kappas"][model], seed=seed,
                                  policy_seed=policy_seed, qmax=config["qmax"])
    arrays = pilot.as_dict()
    np.savez_compressed(out / "pilots" / f"m{model}_r{replicate}.npz", **arrays)
    fits, plans = {}, []
    for budget in config["pilot_budgets"]:
        fit = np.log(np.array(config["model_prior"])) if budget == 0 else fit_grid_posterior(
            {k: v[:budget] for k, v in arrays.items()}, [config["theta"]]*2, config["kappas"]).log_weights
        fits[budget] = fit
        weights = np.exp(fit)
        joint = np.repeat(weights/2, 2)
        n = config["horizon"]
        qj = [family.known_q_values(j, n, 0, 0, joint) for j in range(2)]
        u0 = sum(weights[j]*qj[j][stable_argmax(qj[j])] for j in range(2))
        weighted = family.q_values("weighted_q", n, 0, 0, joint)
        row = {"model_index": model, "replicate": replicate, "budget": budget,
               "weight_model0": float(weights[0]), "initial_model_entropy": float(-xlogy(weights, weights).sum()),
               "revelation_0": float(u0), "revelation_1_direct": float(weighted[stable_argmax(weighted)]),
               "known_true_model": float(qj[model][stable_argmax(qj[model])])}
        row.update({kind: float(family.values(kind, n, 0, 0, joint)) for kind in
                    ("bayes", "reveal1", "reveal2", "reveal3", "no_feedback", "frozen_model")})
        plans.append(row)
    return fits, plans, {"model_index": model, "replicate": replicate,
                        "pilot_seed": seed, "pilot_policy_seed": policy_seed,
                        "model_weights": {str(b): np.exp(f).tolist() for b, f in fits.items()}}


def evaluate(config, family, out, profile):
    replicates = config["replicates_per_model"] if profile == "full" else 2
    episodes = config["episodes_per_replicate"] if profile == "full" else 12
    horizon = config["horizon"]
    for name in ("pilots", "tapes", "actions"):
        (out / name).mkdir()
    traces, model_records, planning, timings = [], [], [], []
    first = True
    with (out / "episodes.csv.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as stream:
                for model in range(2):
                    for replicate in range(replicates):
                        fit, plans, record = pilot_and_plan(config, family, model, replicate, profile, out)
                        planning.extend(plans)
                        seed = stream_seed(config, profile, config["seed_namespaces"]["evaluation"], model, replicate)
                        tapes = generate_exogenous(episodes, horizon, config["theta"], config["kappas"][model], seed)
                        np.savez_compressed(out / "tapes" / f"m{model}_r{replicate}.npz",
                                            **{k: getattr(tapes, k) for k in ("x", "h", "z", "u_bid", "u_ask")})
                        record["evaluation_seed"] = seed
                        model_records.append(record)
                        action_sets, tasks = {}, []
                        for budget in config["pilot_budgets"]:
                            for policy in config["policies"]:
                                controller = (KnownParameterReference(model, family, fit[budget], episodes, config)
                                              if policy == "known_parameter" else
                                              FeasiblePolicy(policy, family, fit[budget], episodes, config))
                                prior_weights = np.exp(fit[budget])
                                tasks.append({"budget": budget, "policy": policy, "controller": controller,
                                              "env": BatchedEnvironment(tapes, config["qmax"]),
                                              "actions": np.empty((episodes, horizon), dtype=np.int8),
                                              "entropy_sum": np.zeros(episodes),
                                              "true_probability_sum": np.zeros(episodes), "traces": [],
                                              "initial_entropy": float(-xlogy(prior_weights, prior_weights).sum())})
                        # Each controller retains its own state and selected feedback.
                        # Interleave only the schedule, reusing the same horizon's
                        # table pages before advancing to the next horizon.
                        for t in range(horizon):
                            for task in tasks:
                                budget, policy = task["budget"], task["policy"]
                                controller, env, actions = task["controller"], task["env"], task["actions"]
                                obs = env.observe()
                                before = controller.posterior.joint
                                decision = controller.decision_joint
                                actions[:, t] = controller.choose(obs)
                                feedback = env.step(actions[:, t])
                                controller.posterior.update(feedback)
                                after = controller.posterior.joint
                                weights = np.exp(controller.posterior.log_weights)
                                task["entropy_sum"] -= xlogy(weights, weights).sum(axis=1)
                                task["true_probability_sum"] += weights[:, model]
                                for i in range(min(episodes, config["trace_episodes_per_replicate"])):
                                    row = {"model_index": model, "replicate": replicate, "episode": i,
                                           "budget": budget, "policy": policy, "t": t,
                                           "inventory": int(obs.inventory[i]), "signal": int(obs.signal[i]),
                                           "price": float(obs.price[i]), "cash": float(obs.cash[i]),
                                           "action": int(actions[i, t]), "return_": float(feedback.return_[i]),
                                           "bid_fill": int(feedback.fills[i, 0]), "ask_fill": int(feedback.fills[i, 1]),
                                           "next_inventory": int(feedback.next_observation.inventory[i]),
                                           "next_cash": float(feedback.next_observation.cash[i])}
                                    row.update({f"p{j}": float(before[i, j]) for j in range(4)})
                                    row.update({f"next_p{j}": float(after[i, j]) for j in range(4)})
                                    row.update({f"decision_p{j}": float(decision[i, j]) for j in range(4)})
                                    task["traces"].append(row)
                        # Serialize in the unchanged budget/policy/time/episode order.
                        for task in tasks:
                            budget, policy = task["budget"], task["policy"]
                            controller, env = task["controller"], task["env"]
                            weights = np.exp(controller.posterior.log_weights)
                            metrics = env.finalize()
                            require(np.max(np.abs(metrics["reconciliation_error"])) < 1e-8, "Ledger failed")
                            require(np.all(metrics["terminal_inventory_after_liquidation"] == 0), "Unliquidated terminal inventory")
                            scalar = {k: v for k, v in metrics.items() if v.ndim == 1}
                            scalar.update({f"action_{a}": metrics["action_counts"][:, a] for a in range(11)})
                            scalar.update(model_entropy_initial=np.full(episodes, task["initial_entropy"]),
                                          model_entropy_final=-xlogy(weights, weights).sum(axis=1),
                                          true_model_probability_final=weights[:, model],
                                          model_entropy_period_mean=task["entropy_sum"]/horizon,
                                          true_model_probability_period_mean=task["true_probability_sum"]/horizon)
                            frame = pd.DataFrame(scalar)
                            for k, v in reversed(list(zip(KEYS, (model, replicate, np.arange(episodes), budget, policy)))):
                                frame.insert(0, k, v)
                            require(np.isfinite(frame.drop(columns="policy").to_numpy()).all(), "Nonfinite outcome")
                            frame.to_csv(stream, index=False, header=first, float_format="%.17g")
                            first = False
                            action_sets[f"b{budget}_{policy}"] = task["actions"]
                            traces.extend(task["traces"])
                            timings.append({"model_index": model, "replicate": replicate, "budget": budget,
                                            "policy": policy, "decisions": episodes*horizon,
                                            "choice_seconds": controller.choice_seconds})
                        np.savez_compressed(out / "actions" / f"m{model}_r{replicate}.npz", **action_sets)
                        print(json.dumps({"stage": "paired_evaluation", "model": model,
                                          "replicate": replicate, "replicates": replicates}), flush=True)
    pd.DataFrame(traces).to_csv(out / "traces.csv.gz", index=False, float_format="%.17g",
                               compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(planning).to_csv(out / "planning_values.csv", index=False, float_format="%.17g")
    pd.DataFrame(timings).to_csv(out / "timings.csv", index=False, float_format="%.17g")
    write_json(out / "pilot_records.json", {"profile": profile, "records": model_records})
    return summarize(out, config, replicates, episodes), replicates, episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--numerical-checks", type=Path)
    args = parser.parse_args()
    config = read_protocol()
    require(not args.out.exists(), "Evaluation output must be a new directory")
    if args.profile == "full":
        require(args.numerical_checks is not None, "Full evaluation requires recorded numerical acceptance")
        acceptance_hash = file_sha256(args.numerical_checks)
        numerical = json.loads(args.numerical_checks.read_text())
        require(numerical.get("passes_predeclared_rule") is True,
                "Numerical resolution is not accepted; record a separate explicit fallback protocol before evaluation")
    else:
        numerical = None
        acceptance_hash = None
    before = source_identity()
    started = time.perf_counter()
    family = load_family(args.tables)
    require(family.spec.theta == config["theta"] and list(family.spec.kappas) == config["kappas"] and
            family.spec.horizon == config["horizon"] and family.spec.qmax == config["qmax"], "Wrong control family")
    if numerical is not None:
        # A success flag alone cannot authorize a different cache or specification.
        validate_numerical_protocol(numerical, config)
        specification = json.loads(json.dumps(asdict(family.spec)))
        require(numerical.get("selected_resolution") == specification and
                numerical.get("selected_artifact_sha256") == family.metadata["artifact_sha256"] and
                numerical.get("source") == family.metadata["source"] and
                numerical.get("build") == family.metadata["build"],
                "Numerical acceptance does not authenticate this control family and protocol")
    args.out.mkdir(parents=True)
    summary, replicates, episodes = evaluate(config, family, args.out, args.profile)
    require(before == source_identity(), "Source/configuration changed during evaluation")
    if numerical is not None:
        require(file_sha256(args.numerical_checks) == acceptance_hash,
                "Numerical acceptance changed during evaluation")
    files = {str(p.relative_to(args.out)): file_sha256(p) for p in sorted(args.out.rglob("*")) if p.is_file()}
    manifest = {"status": "complete", "profile": args.profile, "protocol": config,
                "protocol_sha256": CONFIG_SHA256, "protocol_freeze_commit": PROTOCOL_COMMIT,
                "inputs": before, "core_source_hash": CORE_SOURCE_HASH,
                "replicates_per_model": replicates, "episodes_per_replicate": episodes,
                "policy_records": summary["rows"], "paired_market_trajectories": 2*replicates*episodes,
                "files": files, "table_specification": asdict(family.spec),
                "table_artifact_sha256": family.metadata["artifact_sha256"],
                "table_arrays": family.metadata["arrays"],
                "numerical_acceptance_sha256": acceptance_hash,
                "build": build_fingerprint(), "numerical_contract": numerical_contract(),
                "python": sys.version, "platform": platform.platform(), "seconds": time.perf_counter()-started,
                "scope": "New separate paired synthetic research; source/arrays authenticated. Numerical error is distinct from empirical sampling uncertainty."}
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "profile": args.profile, "rows": summary["rows"],
                      "out": str(args.out), "seconds": manifest["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
