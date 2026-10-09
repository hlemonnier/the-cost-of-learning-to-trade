"""Pre-implementation end-to-end environment/filter acceptance specification.

This is an executable acceptance harness, not a post-implementation unit suite.
Failures and numerical tolerances were enumerated before model implementation.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtr, ndtri, logsumexp

from trade_learning.model import ACTIONS, K, H, DEPTH, CP, CT, LAMBDA, MU, SIGMA, admissible, fill_probability
from trade_learning.environment import (
    ExogenousBatch, BatchedEnvironment, generate_exogenous,
    generate_uniform_pilot, potential_fills,
)
from trade_learning.filtering import observation_log_likelihood, filter_step, fit_grid_posterior


def close(actual, expected, atol=1e-9):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)


def forced(episodes=1, horizon=1, z=.5, ub=-100., ua=-100.):
    return ExogenousBatch(
        x=np.zeros((episodes, horizon + 1), dtype=np.int8),
        h=np.ones((episodes, horizon), dtype=np.int8),
        z=np.full((episodes, horizon), z),
        u_bid=np.full((episodes, horizon), ub),
        u_ask=np.full((episodes, horizon), ua),
        theta=.35, kappa=.02, variant="nominal",
    )


def check_actions_and_ledger():
    expected_actions = np.asarray(list(itertools.product((-1, 0, 1), repeat=2)) + [(-2, -2)] * 2)
    np.testing.assert_array_equal(ACTIONS, expected_actions)
    for qmax in (1, 2, 5):
        for q in range(-qmax, qmax + 1):
            expected = []
            for a, depths in enumerate(expected_actions):
                if a < 9:
                    choices = [(0, 1) if d >= 0 else (0,) for d in depths]
                    safe = all(abs(q + fb - fa) <= qmax for fb, fa in itertools.product(*choices))
                else:
                    safe = abs(q + (1 if a == 9 else -1)) <= qmax
                expected.append(safe)
            np.testing.assert_array_equal(admissible(q, qmax), expected)

    # Every action traverses a complete episode with deterministic selected fills.
    actions = np.arange(11)
    env = BatchedEnvironment(forced(11), qmax=5)
    before = env.observe()
    feedback = env.step(actions)
    expected_q = np.zeros(11)
    expected_c = np.zeros(11)
    expected_spread = np.zeros(11)
    expected_pf = np.zeros(11)
    for a, (bid, ask) in enumerate(expected_actions):
        if a < 9:
            for side, depth in ((1, bid), (-1, ask)):
                if depth >= 0:
                    expected_q[a] += side
                    expected_c[a] += -side * 1000 + .025 + .025 * depth - .001
                    expected_spread[a] += .025 + .025 * depth
                    expected_pf[a] += .001
        else:
            side = 1 if a == 9 else -1
            expected_q[a] = side
            expected_c[a] = -side * 1000 - .025 - .002
    close(feedback.next_observation.inventory, expected_q)
    close(feedback.next_observation.cash, expected_c)
    close(feedback.next_observation.price, 1000 + .3 * .5)
    close(feedback.fees, expected_pf + (actions >= 9) * .002)
    metrics = env.finalize()
    expected_pnl = expected_c + expected_q * (1000 + .3 * .5) - abs(expected_q) * .027
    close(metrics["pnl"], expected_pnl)
    close(metrics["objective"], expected_pnl)  # q_0 is zero even when q_1 is not.
    close(metrics["inventory_sq_sum"], 0)
    close(metrics["spread_capture"], expected_spread)
    close(metrics["reconciled_pnl"], expected_pnl)
    close(metrics["directional_exposure"], 0)
    close(metrics["execution_selection"], np.where(actions < 9, expected_q * .15, 0))
    close(metrics["market_innovation"], np.where(actions >= 9, expected_q * .15, 0))
    close(env.observe().inventory, 0)
    close(env.observe().cash, expected_pnl)
    for operation in (lambda: env.finalize(), lambda: env.step(actions)):
        try:
            operation()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Finished episode operation was accepted")
    # Submitted non-fill and missing feedback remain distinct; no fee on non-fill.
    env = BatchedEnvironment(forced(2, ub=100., ua=100.))
    f = env.step(np.array([3, 9]))
    np.testing.assert_array_equal(f.fills, [[0, -1], [-1, -1]])
    close(f.fees, [0, .002])
    # Boundary safety, abstention exposure, pre-decision penalties, no partial step.
    env = BatchedEnvironment(forced(horizon=6, z=1.))
    for _ in range(5):
        env.step(np.array([9]))
    pre = env.observe()
    for action in (3, 4, 9, -1, 11, 1.5):
        try:
            env.step(np.array([action]))
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("Invalid or unsafe action was accepted")
        close(env.observe().cash, pre.cash)
        close(env.observe().inventory, pre.inventory)
        assert env.observe().t == 5
    f = env.step(np.array([0]))
    close(f.next_observation.cash, pre.cash)
    close(f.next_observation.inventory, 5)
    m = env.finalize()
    close(m["inventory_sq_sum"], 0 + 1 + 4 + 9 + 16 + 25)
    close(m["risk_penalty"], .002 * 55)
    close(m["objective"], m["pnl"] - .002 * 55)
    close(m["liquidation_cost"], 5 * .027)
    close(m["reconciliation_error"], 0)
    return {"actions_exhaustively_checked": 11 * (3 + 5 + 11), "forced_trajectory_actions": 11}


def check_public_boundary_and_pilot():
    tape = generate_exogenous(100, 300, .65, .02, seed=117)
    same = generate_exogenous(100, 300, .65, .02, seed=117)
    other = generate_exogenous(100, 300, .65, .02, seed=118)
    for name in ("x", "h", "z", "u_bid", "u_ask"):
        np.testing.assert_array_equal(getattr(tape, name), getattr(same, name))
        assert not np.array_equal(getattr(tape, name), getattr(other, name))
    env = BatchedEnvironment(tape)
    obs = env.observe()
    assert {f.name for f in dataclasses.fields(obs)} == {
        "t", "horizon", "qmax", "price", "signal", "cash", "inventory", "admissible_actions"}
    assert not obs.inventory.flags.writeable
    # A caller can force its own copy writable but cannot mutate the evaluator.
    obs.inventory.setflags(write=True)
    obs.inventory[:] = 4
    close(env.observe().inventory, 0)
    try:
        obs.t = 5
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("Public record is mutable")
    private_changed = forced(100, 300, z=20., ub=20.)
    other_obs = BatchedEnvironment(private_changed).observe()
    for name in ("price", "signal", "cash", "inventory", "admissible_actions"):
        np.testing.assert_array_equal(getattr(env.observe(), name), getattr(other_obs, name))
    pilot = generate_uniform_pilot(episodes=100, horizon=300, theta=.65, kappa=.02,
                                   seed=117, policy_seed=119)
    data = pilot.as_dict()
    forbidden = ("theta", "kappa", "seed", "regime", "hidden", "u_bid", "u_ask", "counterfactual")
    assert not any(word in key.lower() for key in data for word in forbidden)
    assert all(not value.flags.writeable for value in data.values())
    assert pilot.actions.shape == (100, 300) and pilot.signal.shape == (100, 301)
    close(pilot.price[:, 0], 1000)
    close(pilot.inventory[:, 0], 0)
    close(pilot.cash[:, 0], 0)
    close(pilot.signal[:, 0], 0)
    mask = admissible(pilot.inventory[:, :-1], 5)
    close(pilot.action_probability, 1 / mask.sum(axis=-1))
    assert np.all(np.take_along_axis(mask, pilot.actions[..., None], axis=-1))
    expected_submitted = (ACTIONS[pilot.actions] >= 0) & (pilot.actions[..., None] < 9)
    np.testing.assert_array_equal(pilot.submitted, expected_submitted)
    np.testing.assert_array_equal(pilot.fills == -1, ~expected_submitted)
    assert np.all(abs(pilot.inventory) <= 5)
    # Full independent trajectory ledger, including action/quantity attribution.
    q = np.zeros(100)
    cash = np.zeros(100)
    wealth_terms = np.zeros(100)
    direction, selection, taker_noise = (np.zeros(100) for _ in range(3))
    max_action_z = 0.
    for t in range(300):
        oldq = q.copy()
        dp = pilot.return_[:, t]
        a = pilot.actions[:, t]
        period_cost = np.zeros(100)
        passive_dq = np.zeros(100)
        market_dq = np.zeros(100)
        for side_index, side in enumerate((1, -1)):
            depth = ACTIONS[a, side_index]
            fill = np.where(depth >= 0, pilot.fills[:, t, side_index], 0)
            quantity = side * fill
            cash += fill * (-side * pilot.price[:, t] + .025 + depth * .025 - .001)
            q += quantity
            passive_dq += quantity
            period_cost += fill * (.025 + depth * .025 - .001)
        for action, side in ((9, 1), (10, -1)):
            selected = a == action
            cash[selected] += -side * pilot.price[selected, t] - .027
            q[selected] += side
            market_dq[selected] += side
            period_cost[selected] -= .027
        wealth_terms += period_cost + oldq * dp + (q - oldq) * dp
        direction += oldq * dp + (q - oldq) * .03 * pilot.signal[:, t]
        selection += passive_dq * (dp - .03 * pilot.signal[:, t])
        taker_noise += market_dq * (dp - .03 * pilot.signal[:, t])
        env.step(a)
        close(pilot.inventory[:, t + 1], q)
        close(pilot.cash[:, t + 1], cash, atol=2e-8)
        close(pilot.price[:, t + 1], pilot.price[:, t] + dp)
    close(wealth_terms, cash + q * pilot.price[:, -1], atol=2e-8)
    ledger = env.finalize()
    close(ledger["directional_exposure"], direction)
    close(ledger["execution_selection"], selection)
    close(ledger["market_innovation"], taker_noise)
    close(ledger["directional_pnl"] + ledger["execution_exposure_pnl"], direction + selection + taker_noise)
    close(ledger["reconciliation_error"], 0, atol=2e-8)
    for q0 in range(-5, 6):
        loc = pilot.inventory[:, :-1] == q0
        n = int(loc.sum())
        valid = admissible(q0, 5)
        p = 1 / valid.sum()
        counts = np.bincount(pilot.actions[loc], minlength=11)
        if n:
            z = np.abs(counts[valid] - n * p) / np.sqrt(n * p * (1 - p))
            max_action_z = max(max_action_z, float(z.max()))
            assert np.all(z < 7.)
    return pilot, {"pilot_rows": 30000, "max_uniform_action_z": max_action_z,
                   "max_reconciliation_error": float(np.max(abs(wealth_terms - cash - q * pilot.price[:, -1])))}


def check_conditional_distributions():
    records = []
    # Tolerance = eight estimated standard errors plus a 0.001 floor for fills.
    # Independent tapes are used for every grid/variant combination.
    index = 0
    for theta, kappa, variant in itertools.product((.15, .35, .65), (.002, .02, .10),
                                                  ("nominal", "state_dependence", "fixed_duration")):
        tape = generate_exogenous(1200, 100, theta, kappa, seed=5100 + index, variant=variant)
        index += 1
        fills = potential_fills(tape)
        assert np.all(fills[..., 1] <= fills[..., 0])
        assert np.all(tape.x[:, 0] == 0)
        assert abs(np.mean(tape.h[:, 0] == 1) - .5) < 8 * np.sqrt(.25 / 1200)
        corr = np.corrcoef(np.stack([tape.z.ravel(), tape.u_bid.ravel(), tape.u_ask.ravel()]))
        assert np.max(abs(corr - np.eye(3))) < .025
        if variant == "fixed_duration":
            for row in tape.h:
                switches = np.flatnonzero(row[1:] != row[:-1]) + 1
                assert len(switches) >= 1
                assert np.all(np.diff(switches) == 50)
                assert switches[0] <= 50
        else:
            n = tape.h[:, 1:].size
            observed = np.mean(tape.h[:, 1:] != tape.h[:, :-1])
            assert abs(observed - kappa) < 8 * np.sqrt(kappa * (1 - kappa) / n) + .0001
        max_fill_z = 0.
        max_execution_moment_z = 0.
        for x in (-1, 0, 1):
            selected = tape.x[:, :-1] == x
            z = tape.z[selected]
            n = len(z)
            assert n > 15000
            assert abs(np.mean(z)) < 8 / np.sqrt(n)
            assert abs(np.var(z) - 1) < 8 * np.sqrt(2 / n)
            counts = np.bincount(tape.x[:, 1:][selected] + 1, minlength=3)
            for j in range(3):
                p = K[x + 1, j]
                assert abs(counts[j] / n - p) < 8 * np.sqrt(p * (1 - p) / n) + .0005
            for side_index, side in enumerate((1, -1)):
                for depth in (0, 1):
                    p = float(fill_probability(x, side, depth))
                    obs = fills[..., side_index, depth][selected].mean()
                    se = np.sqrt(p * (1 - p) / n)
                    max_fill_z = max(max_fill_z, abs(obs - p) / se)
                    assert abs(obs - p) < 8 * se + .001
                    for hidden in (-1, 1):
                        loc = selected & (tape.h == hidden)
                        rho = hidden * theta * (1 + .25 * x if variant == "state_dependence" else 1)
                        expected_moment = -rho * side * np.exp(-.5 * ndtri(p)**2) / np.sqrt(2 * np.pi)
                        v = fills[..., side_index, depth][loc] * tape.z[loc]
                        se_m = v.std(ddof=1) / np.sqrt(len(v))
                        max_execution_moment_z = max(max_execution_moment_z, abs(v.mean() - expected_moment) / se_m)
                        assert abs(v.mean() - expected_moment) < 8 * se_m + .001
        # Depth-specific same-return dependence: compare empirical simultaneous
        # fills to an independent Gaussian quadrature calculation.
        nodes, weights = np.polynomial.hermite.hermgauss(80)
        nodes *= np.sqrt(2)
        weights /= np.sqrt(np.pi)
        selected = tape.x[:, :-1] == 0
        near = float(fill_probability(0, 1, 0))
        threshold = ndtri(near)
        probs_b = ndtr((threshold - theta * nodes) / np.sqrt(1 - theta**2))
        probs_a = ndtr((threshold + theta * nodes) / np.sqrt(1 - theta**2))
        joint = float(weights @ (probs_b * probs_a))
        observed_joint = np.mean(fills[..., 0, 0][selected] * fills[..., 1, 0][selected])
        se_joint = np.sqrt(joint * (1 - joint) / selected.sum())
        assert abs(observed_joint - joint) < 8 * se_joint + .001
        records.append({"theta": theta, "kappa": kappa, "variant": variant,
                        "periods": 120000, "max_fill_standard_error_units": max_fill_z,
                        "max_execution_moment_standard_error_units": max_execution_moment_z,
                        "joint_fill_observed": float(observed_joint), "joint_fill_expected": joint})
    return records


def independent_path_likelihood(x, r, a, f, theta, kappa):
    """Enumerate all 2**T hidden paths without using the production filter."""
    total = 0.
    for hidden in itertools.product((-1, 1), repeat=len(x)):
        weight = .5
        for t, ht in enumerate(hidden):
            if t:
                weight *= 1 - kappa if ht == hidden[t - 1] else kappa
            z = (r[t] - .03 * x[t]) / .30
            if a[t] < 9:
                for si, s in enumerate((1, -1)):
                    depth = int(ACTIONS[a[t], si])
                    if depth >= 0:
                        marginal = 1 / (1 + np.exp(.3 + .7 * depth + .2 * s * x[t]))
                        p = ndtr((ndtri(marginal) - ht * theta * s * z) / np.sqrt(1 - theta**2))
                        weight *= p if f[t, si] == 1 else 1 - p
        total += weight
    return total


def check_filter_and_calibration(pilot):
    x = np.array([[0, 1, -1, 0, 1, 0]])
    r = np.array([[.27, -.42, .66, .13, -.22, .09]])
    a = np.array([[4, 3, 0, 8, 2, 9]])
    f = np.array([[[1, 0], [0, -1], [-1, -1], [0, 1], [-1, 1], [-1, -1]]])
    theta = np.array([.15, .35, .65])
    kappa = np.array([.10, .02, .002])
    fitted = fit_grid_posterior({"signal": x, "return_": r, "actions": a, "fills": f}, theta, kappa)
    exact = np.array([independent_path_likelihood(x[0], r[0], a[0], f[0], th, kp)
                      for th, kp in zip(theta, kappa)])
    close(fitted.log_likelihood, np.log(exact), atol=3e-13)
    close(fitted.weights, exact / exact.sum(), atol=3e-13)
    b = np.full((1, 3), .5)
    lw = np.full((1, 3), -np.log(3))
    total = 0.
    for t in range(6):
        oldb = b.copy()
        b, lw, ll = filter_step(b, lw, x[:, t], r[:, t], a[:, t], f[:, t], theta, kappa)
        total += ll[0]
        if a[0, t] in (0, 9, 10):
            close(b, kappa + (1 - 2 * kappa) * oldb)
    close(total, np.log(exact.mean()), atol=3e-13)
    close(np.exp(lw[0]), fitted.weights, atol=3e-13)
    missing = observation_log_likelihood(np.array([0]), np.array([100.]), np.array([0]),
                                        np.array([[-1, -1]]), theta)
    close(missing, 0)
    extreme = observation_log_likelihood(np.array([0, 0]), np.array([-100., 100.]),
                                        np.array([4, 4]), np.array([[1, 1], [0, 0]]), theta)
    assert np.isfinite(extreme).all()
    b, lw, ll = filter_step(np.full((2, 3), .5), np.broadcast_to([-2000., -1000., 0.], (2, 3)),
                           np.array([0, 0]), np.array([-100., 100.]), np.array([4, 4]),
                           np.array([[1, 1], [0, 0]]), theta, kappa)
    assert np.isfinite(lw).all() and np.isfinite(ll).all()
    close(logsumexp(lw, axis=1), 0, atol=1e-10)
    # Compare known-parameter predicted beliefs with hidden truth only here.
    tape = generate_exogenous(100, 300, .65, .02, seed=117)
    belief = np.full((100, 1), .5)
    logw = np.zeros((100, 1))
    beliefs, truth = [], []
    score = 0.
    for t in range(300):
        beliefs.append(belief[:, 0].copy())
        truth.append((tape.h[:, t] == 1).astype(float))
        belief, logw, ll = filter_step(belief, logw, pilot.signal[:, t], pilot.return_[:, t],
                                      pilot.actions[:, t], pilot.fills[:, t], np.array([.65]), np.array([.02]))
        score += float(ll.sum())
    beliefs, truth = np.asarray(beliefs), np.asarray(truth)
    brier = np.mean((beliefs - truth)**2)
    assert brier < .20
    calibration = []
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        loc = (beliefs >= lo) & (beliefs < hi)
        if loc.any():
            calibration.append({"low": float(lo), "high": float(hi), "n": int(loc.sum()),
                                "mean_prediction": float(beliefs[loc].mean()),
                                "frequency": float(truth[loc].mean())})
    gridtheta = np.repeat([.15, .35, .65], 3)
    gridkappa = np.tile([.002, .02, .10], 3)
    gridfit = fit_grid_posterior(pilot, gridtheta, gridkappa)
    assert gridfit.weights[7] > .95
    return {"enumeration_max_log_likelihood_error": float(np.max(abs(fitted.log_likelihood - np.log(exact)))),
            "known_parameter_predictive_brier": float(brier), "uninformed_brier": .25,
            "conditional_fill_log_score_per_period": score / 30000,
            "calibration": calibration, "pilot_grid_posterior": gridfit.weights.tolist(),
            "pilot_grid_log_posterior": gridfit.log_weights.tolist()}


def main():
    started = time.perf_counter()
    checks = {"ledger": check_actions_and_ledger()}
    pilot, checks["public_and_pilot"] = check_public_boundary_and_pilot()
    checks["conditional_distributions"] = check_conditional_distributions()
    checks["filter"] = check_filter_and_calibration(pilot)
    checks["status"] = "PASS"
    checks["seconds"] = time.perf_counter() - started
    checks["command"] = "PYTHONPATH=src .venv/bin/python verification/verify_environment.py"
    paths = [Path(__file__), *[Path("src/trade_learning") / f for f in ("model.py", "environment.py", "filtering.py")]]
    checks["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    output = Path("outputs/environment/verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({"status": checks["status"], "seconds": checks["seconds"], "artifact": str(output)}))


if __name__ == "__main__":
    main()
