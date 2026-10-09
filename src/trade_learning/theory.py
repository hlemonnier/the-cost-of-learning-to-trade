"""Exact-integral constants and evidence for a stationary cost-of-learning bound.

This module is independent of the simulator and controller. It implements the
one-period-reset, constant-sign experiment in report/theory_appendix.md, not a
trading policy for the full benchmark. Floating-point integration illustrates
the analytic proof and is not a certified interval-arithmetic calculation.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import product
import math

import numpy as np
from scipy.integrate import quad
from scipy.special import expit, log_ndtr, ndtr, ndtri


_H = .025
_DEPTH = .025
_CP = .001
_CT = .002
_SIGMA = .30
_LIQ = _H + _CT
_ACTIONS = tuple(product((-1, 0, 1), repeat=2)) + ((-2, -2), (-2, -2))
_SQRT_TWO_PI = math.sqrt(2 * math.pi)


def _phi(z: float) -> float:
    return math.exp(-z * z / 2) / _SQRT_TWO_PI


def _theta_check(theta: float) -> None:
    if not np.isfinite(theta) or not 0 <= theta < 1:
        raise ValueError("theta must be finite and in [0, 1)")


def _integrate(function) -> tuple[float, float]:
    value, error = quad(function, -np.inf, np.inf, epsabs=2e-12, epsrel=2e-12, limit=300)
    return float(value), float(error)


def conditional_log_likelihood(theta: float, regime: int, x: int, z, depths, fills):
    """Log of the conditional selected-feedback mass given observed z and H.

    The common Gaussian return density is omitted; it cancels in every regime
    comparison. Missing sides must have a negative depth and fill=-1. A
    submitted side must have depth 0/1 and fill 0/1. z may be an array.
    """
    _theta_check(theta)
    if regime not in (-1, 1) or x not in (-1, 0, 1):
        raise ValueError("regime must be -1/+1 and x must be -1/0/+1")
    if len(depths) != 2 or len(fills) != 2:
        raise ValueError("Exactly two side positions are required")
    z = np.asarray(z, dtype=float)
    result = np.zeros_like(z)
    scale = math.sqrt(1 - theta * theta)
    for side, depth, fill in zip((1, -1), depths, fills):
        if depth < 0:
            if np.any(np.asarray(fill) != -1):
                raise ValueError("Unsubmitted sides must be marked missing (-1)")
            continue
        if depth not in (0, 1) or np.any(~np.isin(fill, (0, 1))):
            raise ValueError("Submitted depth must be 0/1 and fill must be 0/1")
        a = ndtri(expit(-.3 - .7 * depth - .2 * side * x))
        v = (a - regime * theta * side * z) / scale
        result += np.where(np.asarray(fill) == 1, log_ndtr(v), log_ndtr(-v))
    return float(result) if result.ndim == 0 else result


@lru_cache(maxsize=64)
def side_information(theta: float, depth: int) -> tuple[float, float]:
    """KL(Q_bad || Q_good) per submitted side at x=0, with error estimate."""
    _theta_check(theta)
    if depth not in (0, 1):
        raise ValueError("depth must be 0 or 1")
    if theta == 0:
        return 0.0, 0.0
    a = ndtri(expit(-.3 - .7 * depth))
    scale = math.sqrt(1 - theta * theta)

    def integrand(z):
        v0, v1 = (a - theta * z) / scale, (a + theta * z) / scale
        lp0, lp1 = log_ndtr(v0), log_ndtr(v1)
        lq0, lq1 = log_ndtr(-v0), log_ndtr(-v1)
        divergence = math.exp(lp0) * (lp0 - lp1) + math.exp(lq0) * (lq0 - lq1)
        return _phi(z) * divergence

    return _integrate(integrand)


@lru_cache(maxsize=128)
def joint_fill_probability(theta: float, bid_depth: int, ask_depth: int) -> tuple[float, float]:
    """Unconditional both-fill probability, identical under H=+1 and H=-1."""
    _theta_check(theta)
    if bid_depth not in (0, 1) or ask_depth not in (0, 1):
        raise ValueError("Both submitted depths must be 0 or 1")
    ab = ndtri(expit(-.3 - .7 * bid_depth))
    aa = ndtri(expit(-.3 - .7 * ask_depth))
    scale = math.sqrt(1 - theta * theta)
    return _integrate(lambda z: _phi(z) * ndtr((ab - theta * z) / scale) * ndtr((aa + theta * z) / scale))


def action_table(theta: float = .35) -> list[dict]:
    """All eleven actions with exact-formula rewards and numerical KL integrals."""
    _theta_check(theta)
    p = [float(expit(-.3 - .7 * k)) for k in (0, 1)]
    density = [_phi(float(ndtri(v))) for v in p]
    information = [side_information(theta, k)[0] for k in (0, 1)]
    rows = []
    for action, (bid, ask) in enumerate(_ACTIONS):
        depths = (bid, ask) if action < 9 else ()
        sides = sum(k >= 0 for k in depths)
        both = joint_fill_probability(theta, bid, ask)[0] if action < 9 and bid >= 0 and ask >= 0 else 0.0
        liquidation_units = sum(p[k] for k in depths if k >= 0) - 2 * both
        spread_net_fees = sum((_H + k * _DEPTH - _CP) * p[k] for k in depths if k >= 0)
        selection_magnitude = _SIGMA * theta * sum(density[k] for k in depths if k >= 0)
        kl = sum(information[k] for k in depths if k >= 0)
        if action >= 9:
            bad_reward = good_reward = -2 * _LIQ
            liquidation_units = 1.0
            name = "market_buy" if action == 9 else "market_sell"
        else:
            bad_reward = spread_net_fees - selection_magnitude - _LIQ * liquidation_units
            good_reward = spread_net_fees + selection_magnitude - _LIQ * liquidation_units
            name = "abstain" if action == 0 else f"bid_{bid}_ask_{ask}"
        rows.append({"action": action, "name": name, "bid_depth": bid, "ask_depth": ask, "sides": sides, "reward_bad": float(bad_reward), "reward_good": float(good_reward), "kl_bad_good": float(kl), "expected_terminal_units": float(liquidation_units), "both_fill_probability": float(both), "cost_per_nat": float(-bad_reward / kl) if kl else None})
    return rows


@lru_cache(maxsize=64)
def feedback_affinity(theta: float, action: int = 8) -> tuple[float, float]:
    """Hellinger affinity integral for one selected action and observed return."""
    _theta_check(theta)
    if action not in range(11):
        raise ValueError("Invalid action")
    if action == 0 or action >= 9:
        return 1.0, 0.0
    depths = _ACTIONS[action]
    scale = math.sqrt(1 - theta * theta)

    def integrand(z):
        log_affinity = 0.0
        for side, depth in zip((1, -1), depths):
            if depth < 0:
                continue
            a = ndtri(expit(-.3 - .7 * depth))
            v0, v1 = (a - theta * side * z) / scale, (a + theta * side * z) / scale
            log_affinity += float(np.logaddexp(.5 * (log_ndtr(v0) + log_ndtr(v1)), .5 * (log_ndtr(-v0) + log_ndtr(-v1))))
        return _phi(z) * math.exp(log_affinity)

    return _integrate(integrand)


def reduction_constants(theta: float = .35, delta: float = .05) -> dict:
    """Compute exact-formula bounds for a reduction with all informative actions costly.

    Raises if the requested theta does not satisfy the simple positive
    pre-liquidation cost assumption used by the analytic economic proof.
    """
    _theta_check(theta)
    if theta == 0 or not 0 < delta < .5:
        raise ValueError("Need theta>0 and 0<delta<1/2 for identification")
    p = [float(expit(-.3 - .7 * k)) for k in (0, 1)]
    thresholds = [float(ndtri(v)) for v in p]
    side_costs = [_SIGMA * theta * _phi(thresholds[k]) - (_H + k * _DEPTH - _CP) * p[k] for k in (0, 1)]
    minimum_cost = min(side_costs)
    if minimum_cost <= 0:
        raise ValueError("This theta does not satisfy the strictly costly-feedback assumption")
    values_errors = [side_information(theta, k) for k in (0, 1)]
    side_kl = [ve[0] for ve in values_errors]
    gaussian_kl = 2 * theta * theta / (1 - theta * theta)
    beta = (1 - 2 * delta) * math.log((1 - delta) / delta)
    rows = action_table(theta)
    informative = [row for row in rows if row["sides"]]
    cheapest = min(informative, key=lambda row: row["cost_per_nat"])
    eta = cheapest["cost_per_nat"]
    affinity, affinity_error = feedback_affinity(theta, 8)
    return {"theta": theta, "delta": delta, "regime_bad": 1, "regime_good": -1, "fixed_signal": 0, "liquidation_cost_per_unit": _LIQ, "fill_probabilities": p, "thresholds": thresholds, "gross_cost_per_side_by_depth": side_costs, "minimum_gross_cost_per_side": minimum_cost, "continuous_side_kl": gaussian_kl, "side_kl_by_depth": side_kl, "maximum_side_kl": max(side_kl), "testing_kl_requirement": beta, "side_bound_gaussian": beta / gaussian_kl, "side_bound_exact": beta / max(side_kl), "period_bound_gaussian": beta / (2 * gaussian_kl), "period_bound_exact": beta / (2 * max(side_kl)), "minimum_integer_periods_exact": math.ceil(beta / (2 * max(side_kl))), "economic_bound_analytic": minimum_cost * beta / gaussian_kl, "economic_bound_exact_sides": minimum_cost * beta / max(side_kl), "minimum_cost_per_nat": eta, "minimum_cost_per_nat_action": cheapest["action"], "economic_bound_exact_action": eta * beta, "optimal_bad_action": max(rows, key=lambda row: row["reward_bad"])["action"], "optimal_good_action": max(rows, key=lambda row: row["reward_good"])["action"], "optimal_good_reward": max(row["reward_good"] for row in rows), "deep_two_sided_affinity": affinity, "quadrature_error_estimates": {"side_kl": [ve[1] for ve in values_errors], "affinity": affinity_error}, "numerical_status": "Exact integral formulas evaluated by adaptive quadrature; error estimates are not certified interval bounds."}


def simulate_identification(theta: float = .35, delta: float = .05, repeats: int = 40_000, seed: int = 2026092404) -> dict:
    """Stopped deep-two-sided likelihood-ratio test, using raw potential normals.

    Quote both deep sides while |log(dP_bad/dP_good)| < log(2/delta),
    then abstain for the remainder of a fixed number of reset opportunities.
    A finite-horizon sign decision completes the procedure. The risk bound
    exp(-boundary)+affinity**horizon follows from a stopped likelihood-ratio
    martingale and a fixed-horizon Chernoff inequality, derived in the appendix.
    The affinity used to choose the illustrative horizon is numerical.
    """
    constants = reduction_constants(theta, delta)
    affinity = constants["deep_two_sided_affinity"]
    # Extra two periods and a small numerical buffer protect against ordinary
    # floating-point integration discrepancies, without claiming certification.
    numerical_affinity_upper = affinity + max(1e-12, 10 * constants["quadrature_error_estimates"]["affinity"])
    horizon = math.ceil(math.log(delta / 2) / math.log(numerical_affinity_upper)) + 2
    boundary = math.log(2 / delta)
    outcomes = []
    rows = action_table(theta)
    quote_row = rows[8]
    child_seeds = np.random.SeedSequence(seed).spawn(2)
    threshold = ndtri(expit(-1.0))
    scale = math.sqrt(1 - theta * theta)
    for regime, sequence in zip((1, -1), child_seeds):
        rng = np.random.default_rng(sequence)
        llr = np.zeros(repeats)
        pnl = np.zeros(repeats)
        counts = np.zeros(repeats, dtype=np.int64)
        stopped = np.zeros(repeats, dtype=bool)
        for _ in range(horizon):
            selected = ~stopped
            n_active = int(np.sum(selected))
            if n_active == 0:
                break
            z, ub, ua = rng.standard_normal((3, n_active))
            fb = (regime * theta * z + scale * ub <= threshold).astype(np.int8)
            fa = (-regime * theta * z + scale * ua <= threshold).astype(np.int8)
            lp = conditional_log_likelihood(theta, 1, 0, z, (1, 1), (fb, fa))
            lm = conditional_log_likelihood(theta, -1, 0, z, (1, 1), (fb, fa))
            llr[selected] += lp - lm
            dq = fb - fa
            pnl[selected] += (fb + fa) * (.05 - .001) + dq * .30 * z - np.abs(dq) * _LIQ
            counts[selected] += 1
            stopped |= np.abs(llr) >= boundary
        wrong = llr < 0 if regime == 1 else llr >= 0
        rate = float(np.mean(wrong))
        actual_llr = regime * llr
        kl_compensator = counts * quote_row["kl_bad_good"]
        reward_mean = quote_row["reward_bad"] if regime == 1 else quote_row["reward_good"]
        pnl_compensator = counts * reward_mean
        result = {"regime": regime, "repeats": repeats, "error_rate": rate, "error_se": math.sqrt(rate * (1 - rate) / repeats), "mean_quoted_periods": float(np.mean(counts)), "mean_submitted_sides": float(2 * np.mean(counts)), "horizon_censoring_rate": float(np.mean(~stopped)), "mean_log_likelihood_ratio": float(np.mean(actual_llr)), "expected_kl_from_counts": float(np.mean(kl_compensator)), "kl_martingale_se": float(np.std(actual_llr - kl_compensator, ddof=1) / math.sqrt(repeats)), "mean_pnl": float(np.mean(pnl)), "pnl_se": float(np.std(pnl, ddof=1) / math.sqrt(repeats)), "expected_pnl_from_counts": float(np.mean(pnl_compensator)), "pnl_martingale_se": float(np.std(pnl - pnl_compensator, ddof=1) / math.sqrt(repeats))}
        result["expected_regret_from_counts"] = float(-np.mean(pnl_compensator)) if regime == 1 else float(horizon * constants["optimal_good_reward"] - np.mean(pnl_compensator))
        outcomes.append(result)
    return {"policy": "Quote both deep sides until the log-likelihood-ratio hits +/-log(2/delta), then abstain; classify at the fixed horizon", "theta": theta, "delta": delta, "seed": seed, "boundary": boundary, "horizon_opportunities": horizon, "affinity": affinity, "risk_bound_formula": "exp(-boundary) + affinity**horizon", "risk_bound_numerical": math.exp(-boundary) + affinity ** horizon, "scope_note": "An identification experiment, not a proposed profit-maximising policy. Risk formula is analytic; numeric affinity is quadrature, not a certificate.", "environments": outcomes}
