"""Independent end-to-end evidence for the stationary learning-cost theorem.

Written before the theory implementation; failure cases are in
report/theory_appendix.md. Run from project root:
    PYTHONPATH=src .venv/bin/python verification/verify_theory.py

Produces outputs/theory/verification.json, constants.json, actions.csv and
identification.json. Independent quadrature below is intentionally expressed
as the full selected-feedback experiment, including all fill subsets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit, log_ndtr, ndtri

from trade_learning.theory import (
    action_table,
    conditional_log_likelihood,
    reduction_constants,
    simulate_identification,
)


ROOT = Path(__file__).resolve().parents[1]
ACTIONS = [(b, a) for b in (-1, 0, 1) for a in (-1, 0, 1)] + [(-2, -2), (-2, -2)]


def independent_integrals(theta: float, action: int, order: int) -> dict[str, float]:
    """Full feedback enumeration, not a sum of per-side KLs or moment identity."""
    nodes, weights = hermgauss(order)
    z, w = np.sqrt(2.0) * nodes, weights / np.sqrt(np.pi)
    if action == 0:
        return {"reward_bad": 0.0, "reward_good": 0.0, "kl": 0.0, "mass_error": 0.0, "fill_law_error": 0.0}
    if action >= 9:
        sign = 1 if action == 9 else -1
        gain = np.dot(w, sign * .30 * z - 2 * .027)
        return {"reward_bad": float(gain), "reward_good": float(gain), "kl": 0.0, "mass_error": 0.0, "fill_law_error": 0.0}
    bid, ask = ACTIONS[action]
    sums = np.zeros(4)  # bad reward, good reward, bad-to-good KL, reverse KL
    masses = np.zeros(2)
    fill_law_error = 0.0
    for fb in ((0, 1) if bid >= 0 else (-1,)):
        for fa in ((0, 1) if ask >= 0 else (-1,)):
            lp = np.zeros_like(z)
            lm = np.zeros_like(z)
            gain = np.zeros_like(z)
            dq = 0
            for depth, sign, fill in ((bid, 1, fb), (ask, -1, fa)):
                if depth < 0:
                    continue
                threshold = ndtri(expit(-.3 - .7 * depth))
                vp = (threshold - theta * sign * z) / np.sqrt(1 - theta ** 2)
                vm = (threshold + theta * sign * z) / np.sqrt(1 - theta ** 2)
                lp += log_ndtr(vp if fill else -vp)
                lm += log_ndtr(vm if fill else -vm)
                gain += fill * (.025 + .025 * depth - .001)
                dq += sign * fill
            gain += dq * .30 * z - abs(dq) * .027
            pp, pm = np.exp(lp), np.exp(lm)
            sums += [np.dot(w, pp * gain), np.dot(w, pm * gain), np.dot(w, pp * (lp - lm)), np.dot(w, pm * (lm - lp))]
            mp, mm = np.dot(w, pp), np.dot(w, pm)
            masses += [mp, mm]
            fill_law_error = max(fill_law_error, abs(mp - mm))
    return {"reward_bad": float(sums[0]), "reward_good": float(sums[1]), "kl": float(sums[2]), "reverse_kl": float(sums[3]), "mass_error": float(np.max(np.abs(masses - 1))), "fill_law_error": float(fill_law_error)}


def raw_action_experiments(theta: float, samples: int, seed: int) -> list[dict]:
    """Actual potential-normal fills and signed cash ledger, independent of theory."""
    rng = np.random.default_rng(seed)
    z, ub, ua = rng.standard_normal((3, samples))
    result = []
    for regime in (1, -1):
        vb = regime * theta * z + np.sqrt(1 - theta ** 2) * ub
        va = -regime * theta * z + np.sqrt(1 - theta ** 2) * ua
        fills_b = [vb <= ndtri(expit(-.3 - .7 * k)) for k in (0, 1)]
        fills_a = [va <= ndtri(expit(-.3 - .7 * k)) for k in (0, 1)]
        assert np.all(~fills_b[1] | fills_b[0])
        assert np.all(~fills_a[1] | fills_a[0])
        for aid, (bid, ask) in enumerate(ACTIONS):
            cash = np.zeros(samples)
            inventory = np.zeros(samples)
            if aid < 9:
                if bid >= 0:
                    f = fills_b[bid]
                    cash += f * (-1000.0 + .025 + .025 * bid - .001)
                    inventory += f
                if ask >= 0:
                    f = fills_a[ask]
                    cash += f * (1000.0 + .025 + .025 * ask - .001)
                    inventory -= f
            else:
                inventory += 1 if aid == 9 else -1
                cash -= (1 if aid == 9 else -1) * 1000.0 + .027
            pnl = cash + inventory * (1000.0 + .30 * z) - .027 * np.abs(inventory)
            result.append({"regime": regime, "action": aid, "mean": float(np.mean(pnl)), "se": float(np.std(pnl, ddof=1) / np.sqrt(samples))})
    return result


def write_json(path: Path, obj: dict | list) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/theory")
    parser.add_argument("--samples", type=int, default=250_000)
    parser.add_argument("--identification-repeats", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=2026092403)
    args = parser.parse_args()
    if args.samples < 1000 or args.identification_repeats < 1000:
        raise ValueError("Use at least 1000 independent repetitions for Monte Carlo checks.")
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    constants = reduction_constants(theta=.35, delta=.05)
    rows = action_table(theta=.35)
    assert len(rows) == 11 and [r["action"] for r in rows] == list(range(11))
    checks = {}
    maximum = {"reward_error": 0.0, "kl_error": 0.0, "refinement_error": 0.0, "fill_law_error": 0.0}
    for row in rows:
        low = independent_integrals(.35, row["action"], 80)
        high = independent_integrals(.35, row["action"], 160)
        maximum["reward_error"] = max(maximum["reward_error"], abs(row["reward_bad"] - high["reward_bad"]), abs(row["reward_good"] - high["reward_good"]))
        maximum["kl_error"] = max(maximum["kl_error"], abs(row["kl_bad_good"] - high["kl"]))
        maximum["refinement_error"] = max(maximum["refinement_error"], *(abs(high[k] - low[k]) for k in ("reward_bad", "reward_good", "kl")))
        maximum["fill_law_error"] = max(maximum["fill_law_error"], high["fill_law_error"])
        assert high["mass_error"] < 1e-12
        assert abs(high.get("reverse_kl", high["kl"]) - high["kl"]) < 1e-12
        assert row["kl_bad_good"] <= constants["continuous_side_kl"] * row["sides"] + 1e-12
        if row["sides"]:
            assert row["reward_bad"] <= -constants["minimum_gross_cost_per_side"] * row["sides"] + 1e-12
            assert row["kl_bad_good"] > 0
        elif row["action"] == 0:
            assert row["reward_bad"] == row["reward_good"] == row["kl_bad_good"] == 0
        else:
            assert abs(row["reward_bad"] + .054) < 1e-12
    assert all(value < 2e-10 for value in maximum.values()), maximum
    checks["independent_quadrature"] = maximum
    assert constants["optimal_bad_action"] == 0
    assert constants["optimal_good_action"] != 0
    assert constants["economic_bound_exact_action"] >= constants["economic_bound_analytic"] > 0
    assert constants["side_bound_exact"] >= constants["side_bound_gaussian"] > 0
    checks["economic_and_information_inequalities"] = True

    # Tail likelihoods, absent feedback and sign reversal are exact properties.
    for z in (-40., -10., 0., 10., 40.):
        for x in (-1, 0, 1):
            for action, depths in enumerate(ACTIONS[:9]):
                fb_values = (0, 1) if depths[0] >= 0 else (-1,)
                fa_values = (0, 1) if depths[1] >= 0 else (-1,)
                for fb in fb_values:
                    for fa in fa_values:
                        lp = conditional_log_likelihood(.35, 1, x, z, depths, (fb, fa))
                        lm = conditional_log_likelihood(.35, -1, x, -z, depths, (fb, fa))
                        assert np.isfinite(lp) and abs(lp - lm) < 1e-12
    for invalid_delta in (0., .5, 1.):
        try:
            reduction_constants(delta=invalid_delta)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid accuracy level was accepted")
    try:
        reduction_constants(theta=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Unidentifiable theta=0 was accepted for a learning bound")
    checks["tail_likelihood_and_invalid_assumptions"] = True

    raw = raw_action_experiments(.35, args.samples, args.seed)
    max_zscore = 0.0
    for observed in raw:
        expected = rows[observed["action"]]["reward_bad" if observed["regime"] == 1 else "reward_good"]
        zscore = abs(observed["mean"] - expected) / max(observed["se"], 1e-14)
        max_zscore = max(max_zscore, zscore)
        assert zscore < 6.5, (observed, expected, zscore)
    checks["raw_noise_ledger"] = {"samples_per_regime": args.samples, "maximum_standardized_error": max_zscore, "simultaneous_tolerance_standard_errors": 6.5, "results": raw}

    identification = simulate_identification(theta=.35, delta=.05, repeats=args.identification_repeats, seed=args.seed + 1)
    for env in identification["environments"]:
        assert env["error_rate"] + 4 * env["error_se"] < .05
        assert abs(env["mean_log_likelihood_ratio"] - env["expected_kl_from_counts"]) < 6.5 * env["kl_martingale_se"] + 1e-12
        assert env["mean_submitted_sides"] > constants["side_bound_exact"]
        if env["regime"] == 1:
            assert env["expected_regret_from_counts"] >= constants["economic_bound_exact_action"]
            assert abs(env["mean_pnl"] + env["expected_regret_from_counts"]) < 6.5 * env["pnl_martingale_se"] + 1e-12
    checks["adaptive_identification"] = True

    write_json(args.output / "constants.json", constants)
    write_json(args.output / "identification.json", identification)
    with (args.output / "actions.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    files = [ROOT / "src/trade_learning/theory.py", Path(__file__), ROOT / "BENCHMARK.md"]
    artifact = {"status": "passed", "scope": "Independent stationary-reduction and exact-likelihood evidence; not final policy evaluation", "seed": args.seed, "python": platform.python_version(), "numpy": np.__version__, "runtime_seconds": time.perf_counter() - started, "checks": checks, "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    write_json(args.output / "verification.json", artifact)
    print(json.dumps({"status": artifact["status"], "runtime_seconds": artifact["runtime_seconds"], "constants": constants, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
