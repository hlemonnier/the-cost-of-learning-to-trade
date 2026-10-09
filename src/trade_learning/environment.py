"""Batched evaluator, an explicit ledger, and isolated public policy records.

The evaluator owns exogenous tapes. A policy receives only PublicObservation
and PublicFeedback, never BatchedEnvironment or ExogenousBatch. Read-only
defensive copies enforce memory separation; Python is not a security sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping
import numpy as np
from scipy.special import ndtri

from .model import K, ACTIONS, H, DEPTH, CP, CT, LAMBDA, MU, SIGMA, admissible, fill_probability


def _frozen_array(value, dtype=None):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class ExogenousBatch:
    """Evaluator-only tape, independent of any action or policy randomisation."""
    x: np.ndarray
    h: np.ndarray
    z: np.ndarray
    u_bid: np.ndarray
    u_ask: np.ndarray
    theta: float
    kappa: float
    variant: str = "nominal"

    def __post_init__(self):
        if self.variant not in ("nominal", "state_dependence", "fixed_duration"):
            raise ValueError("unknown environment variant")
        if not (0 <= self.theta < 1) or not (0 <= self.kappa <= 1):
            raise ValueError("invalid environment parameters")
        if self.variant == "state_dependence" and self.theta * 1.25 >= 1:
            raise ValueError("state-dependent correlation must have absolute value below one")
        for name in ("x", "h", "z", "u_bid", "u_ask"):
            object.__setattr__(self, name, _frozen_array(getattr(self, name)))
        if self.z.ndim != 2 or min(self.z.shape) < 1:
            raise ValueError("tapes require positive episode and horizon dimensions")
        n, horizon = self.z.shape
        if self.x.shape != (n, horizon + 1):
            raise ValueError("signal tape must include both episode endpoints")
        if any(getattr(self, name).shape != self.z.shape for name in ("h", "u_bid", "u_ask")):
            raise ValueError("hidden and innovation tapes must have matching dimensions")
        if not np.all(np.isin(self.x, (-1, 0, 1))) or not np.all(np.isin(self.h, (-1, 1))):
            raise ValueError("invalid signal or hidden sign")
        if not np.all(self.x[:, 0] == 0):
            raise ValueError("all episodes must start with public signal zero")
        if not all(np.isfinite(getattr(self, name)).all() for name in ("z", "u_bid", "u_ask")):
            raise ValueError("innovations must be finite")

    @property
    def episodes(self):
        return self.z.shape[0]

    @property
    def horizon(self):
        return self.z.shape[1]


def generate_exogenous(episodes, horizon, theta, kappa, seed, variant="nominal"):
    """Generate evaluator tapes using separate deterministic component streams.

    The same tapes can be reused across policies for paired evaluation. Initial
    regimes and geometric flips share their regime RNG; signal, return, two
    side noises and fixed-duration phase each have distinct spawned streams.
    """
    if int(episodes) != episodes or int(horizon) != horizon or episodes < 1 or horizon < 1:
        raise ValueError("episodes and horizon must be positive integers")
    episodes, horizon = int(episodes), int(horizon)
    if not 0 <= theta < 1 or not 0 <= kappa <= 1:
        raise ValueError("invalid environment parameters")
    if variant not in ("nominal", "state_dependence", "fixed_duration"):
        raise ValueError("unknown environment variant")
    root = np.random.SeedSequence(seed)
    rx, rh, rz, rb, ra, rd = (np.random.default_rng(s) for s in root.spawn(6))
    x = np.zeros((episodes, horizon + 1), dtype=np.int8)
    x_uniform = rx.random((episodes, horizon))
    cumulative = np.cumsum(K, axis=1)
    for t in range(horizon):
        x[:, t + 1] = np.sum(x_uniform[:, t, None] > cumulative[x[:, t] + 1], axis=1) - 1
    h0 = 2 * rh.integers(0, 2, size=episodes, dtype=np.int8) - 1
    if variant == "fixed_duration":
        phase = rd.integers(0, 50, size=episodes)
        parity = ((np.arange(horizon)[None, :] + phase[:, None]) // 50) % 2
        hidden = (h0[:, None] * (1 - 2 * parity)).astype(np.int8)
    else:
        hidden = np.empty((episodes, horizon), dtype=np.int8)
        hidden[:, 0] = h0
        flips = np.where(rh.random((episodes, horizon - 1)) < kappa, -1, 1)
        hidden[:, 1:] = h0[:, None] * np.cumprod(flips, axis=1)
    return ExogenousBatch(x=x, h=hidden, z=rz.standard_normal((episodes, horizon)),
                          u_bid=rb.standard_normal((episodes, horizon)),
                          u_ask=ra.standard_normal((episodes, horizon)),
                          theta=float(theta), kappa=float(kappa), variant=variant)


def potential_fills(tapes: ExogenousBatch):
    """Evaluator-only counterfactual array (episode,period,side,depth)."""
    x = tapes.x[:, :-1]
    rho = tapes.h * tapes.theta
    if tapes.variant == "state_dependence":
        rho = rho * (1 + .25 * x)
    residual_scale = np.sqrt(1 - rho**2)
    result = np.empty(tapes.z.shape + (2, 2), dtype=bool)
    for side_index, (side, noise) in enumerate(((1, tapes.u_bid), (-1, tapes.u_ask))):
        v = rho * side * tapes.z + residual_scale * noise
        for depth in (0, 1):
            result[..., side_index, depth] = v <= ndtri(fill_probability(x, side, depth))
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PublicObservation:
    t: int
    horizon: int
    qmax: int
    price: np.ndarray
    signal: np.ndarray
    cash: np.ndarray
    inventory: np.ndarray
    admissible_actions: np.ndarray


@dataclass(frozen=True, slots=True)
class PublicFeedback:
    t: int
    action: np.ndarray
    signal: np.ndarray
    return_: np.ndarray
    submitted: np.ndarray
    fills: np.ndarray
    execution_prices: np.ndarray
    executed_quantities: np.ndarray
    fees: np.ndarray
    next_observation: PublicObservation


class BatchedEnvironment:
    """Evaluator-owned batched ledger with atomic legality checks before steps."""

    def __init__(self, tapes: ExogenousBatch, qmax=5):
        if int(qmax) != qmax or qmax < 1:
            raise ValueError("qmax must be a positive integer")
        self._tapes = tapes
        self.qmax = int(qmax)
        self.horizon = tapes.horizon
        self.episodes = tapes.episodes
        self._t = 0
        self._finalized = False
        self._price = np.full(self.episodes, 1000.)
        self._cash = np.zeros(self.episodes)
        self._q = np.zeros(self.episodes, dtype=np.int16)
        names = ("spread_capture", "passive_fees", "market_fees", "market_spread_cost",
                 "directional_pnl", "execution_exposure_pnl", "inventory_sq_sum",
                 "inventory_post_sq_sum", "inventory_abs_sum", "max_abs_inventory",
                 "passive_fills", "market_fills", "directional_exposure",
                 "execution_selection", "market_innovation")
        self._metrics = {name: np.zeros(self.episodes) for name in names}
        self._action_counts = np.zeros((self.episodes, 11), dtype=np.int32)
        self._fill_counts = np.zeros((self.episodes, 2), dtype=np.int32)

    def observe(self):
        """Return a detached pre-decision snapshot; no latent/seed metadata."""
        return PublicObservation(t=self._t, horizon=self.horizon, qmax=self.qmax,
                                 price=_frozen_array(self._price),
                                 signal=_frozen_array(self._tapes.x[:, self._t]),
                                 cash=_frozen_array(self._cash),
                                 inventory=_frozen_array(self._q),
                                 admissible_actions=_frozen_array(admissible(self._q, self.qmax)))

    def step(self, action_ids):
        if self._finalized or self._t >= self.horizon:
            raise RuntimeError("episode has no remaining action periods")
        actions = np.asarray(action_ids)
        if actions.ndim == 0:
            actions = np.full(self.episodes, actions.item())
        if actions.shape != (self.episodes,):
            raise ValueError("one action per episode is required")
        if not np.issubdtype(actions.dtype, np.integer):
            raise TypeError("action IDs must have an integer dtype")
        if np.any(actions < 0) or np.any(actions > 10):
            raise ValueError("action ID outside 0..10")
        mask = admissible(self._q, self.qmax)
        if not np.all(mask[np.arange(self.episodes), actions]):
            raise ValueError("an action permits a fill subset outside the inventory bound")
        t = self._t
        x = self._tapes.x[:, t]
        z = self._tapes.z[:, t]
        rho = self._tapes.h[:, t] * self._tapes.theta
        if self._tapes.variant == "state_dependence":
            rho = rho * (1 + .25 * x)
        residual = np.sqrt(1 - rho**2)
        depths = ACTIONS[actions]
        submitted = depths >= 0
        fills = np.full((self.episodes, 2), -1, dtype=np.int8)
        quantities = np.zeros((self.episodes, 2), dtype=np.int8)
        prices = np.full((self.episodes, 2), np.nan)
        fees = np.zeros(self.episodes)
        delta_cash = np.zeros(self.episodes)
        delta_q = np.zeros(self.episodes, dtype=np.int16)
        passive_delta_q = np.zeros(self.episodes, dtype=np.int16)
        market_delta_q = np.zeros(self.episodes, dtype=np.int16)
        metrics = self._metrics
        for side_index, (side, noise) in enumerate(((1, self._tapes.u_bid[:, t]),
                                                  (-1, self._tapes.u_ask[:, t]))):
            depth = depths[:, side_index]
            is_submitted = submitted[:, side_index]
            threshold = ndtri(fill_probability(x, side, np.maximum(depth, 0)))
            realized = (rho * side * z + residual * noise <= threshold) & is_submitted
            fills[is_submitted, side_index] = realized[is_submitted].astype(np.int8)
            quantities[:, side_index] = realized
            distance = H + DEPTH * depth
            delta_cash += realized * (-side * self._price + distance - CP)
            delta_q += side * realized
            passive_delta_q += side * realized
            prices[realized, side_index] = (self._price - side * distance)[realized]
            metrics["spread_capture"] += realized * distance
            metrics["passive_fees"] += realized * CP
            metrics["passive_fills"] += realized
            self._fill_counts[:, side_index] += realized
            fees += realized * CP
        for action, side, side_index in ((9, 1, 0), (10, -1, 1)):
            selected = actions == action
            delta_cash += selected * (-side * self._price - H - CT)
            delta_q += side * selected
            market_delta_q += side * selected
            quantities[selected, side_index] = 1
            prices[selected, side_index] = self._price[selected] + side * H
            metrics["market_fees"] += selected * CT
            metrics["market_spread_cost"] += selected * H
            metrics["market_fills"] += selected
            fees += selected * CT
        return_ = MU * x + SIGMA * z
        metrics["directional_pnl"] += self._q * return_
        metrics["execution_exposure_pnl"] += delta_q * return_
        metrics["directional_exposure"] += self._q * return_ + delta_q * MU * x
        metrics["execution_selection"] += passive_delta_q * SIGMA * z
        metrics["market_innovation"] += market_delta_q * SIGMA * z
        metrics["inventory_sq_sum"] += self._q.astype(float)**2
        metrics["inventory_abs_sum"] += abs(self._q)
        self._cash += delta_cash
        self._q += delta_q
        self._price += return_
        metrics["inventory_post_sq_sum"] += self._q.astype(float)**2
        np.maximum(metrics["max_abs_inventory"], abs(self._q), out=metrics["max_abs_inventory"])
        self._action_counts[np.arange(self.episodes), actions] += 1
        self._t += 1
        return PublicFeedback(t=t, action=_frozen_array(actions), signal=_frozen_array(x),
                              return_=_frozen_array(return_), submitted=_frozen_array(submitted),
                              fills=_frozen_array(fills), execution_prices=_frozen_array(prices),
                              executed_quantities=_frozen_array(quantities), fees=_frozen_array(fees),
                              next_observation=self.observe())

    def finalize(self):
        """Liquidate once after the last period; return detached episode metrics."""
        if self._finalized:
            raise RuntimeError("terminal liquidation has already been applied")
        if self._t != self.horizon:
            raise RuntimeError("cannot finalize an unfinished episode")
        metrics = self._metrics
        terminal_q = self._q.copy()
        terminal_units = abs(terminal_q)
        liquidation = terminal_units * (H + CT)
        pnl = self._cash + self._q * self._price - liquidation
        reconciled = (metrics["spread_capture"] - metrics["passive_fees"]
                      - metrics["market_fees"] - metrics["market_spread_cost"]
                      + metrics["directional_pnl"] + metrics["execution_exposure_pnl"] - liquidation)
        metrics.update(terminal_inventory=terminal_q, terminal_units=terminal_units,
                       liquidation_cost=liquidation, liquidation_fees=terminal_units * CT,
                       liquidation_spread=terminal_units * H, pnl=pnl,
                       risk_penalty=LAMBDA * metrics["inventory_sq_sum"],
                       inventory_penalty=LAMBDA * metrics["inventory_sq_sum"],
                       objective=pnl - LAMBDA * metrics["inventory_sq_sum"],
                       reconciled_pnl=reconciled, reconciliation_error=pnl - reconciled,
                       turnover=metrics["passive_fills"] + metrics["market_fills"] + terminal_units,
                       action_counts=self._action_counts, fill_counts=self._fill_counts,
                       terminal_inventory_after_liquidation=np.zeros(self.episodes, dtype=np.int16))
        self._cash = pnl.copy()
        self._q[:] = 0
        self._finalized = True
        return {name: _frozen_array(value) for name, value in metrics.items()}


@dataclass(frozen=True, slots=True)
class PilotData:
    """Only public historical observations, with states at both endpoints."""
    price: np.ndarray
    signal: np.ndarray
    cash: np.ndarray
    inventory: np.ndarray
    actions: np.ndarray
    action_probability: np.ndarray
    return_: np.ndarray
    submitted: np.ndarray
    fills: np.ndarray
    execution_prices: np.ndarray
    executed_quantities: np.ndarray
    fees: np.ndarray

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self, field.name, _frozen_array(getattr(self, field.name)))

    def as_dict(self):
        """Return detached read-only arrays, suitable for np.savez_compressed."""
        return {field.name: _frozen_array(getattr(self, field.name)) for field in fields(self)}

    to_dict = as_dict


def generate_uniform_pilot(*, episodes=100, horizon=300, theta, kappa, seed, policy_seed, qmax=5):
    """Generate the prescribed nominal pilot without returning any evaluator data.

    Every decision is uniform over currently safe action IDs. Its exact
    behaviour probability is stored; each independent episode resets all public
    state and the regime prior. Explicit independent policy randomisation is
    separate from the spawned exogenous component streams.
    """
    tapes = generate_exogenous(episodes, horizon, theta, kappa, seed, variant="nominal")
    env = BatchedEnvironment(tapes, qmax=qmax)
    rng = np.random.default_rng(policy_seed)
    data = {name: np.empty((episodes, horizon + 1), dtype=dtype)
            for name, dtype in (("price", float), ("signal", np.int8), ("cash", float),
                                ("inventory", np.int16))}
    data.update(actions=np.empty((episodes, horizon), dtype=np.int8),
                action_probability=np.empty((episodes, horizon)), return_=np.empty((episodes, horizon)),
                submitted=np.empty((episodes, horizon, 2), dtype=bool),
                fills=np.empty((episodes, horizon, 2), dtype=np.int8),
                execution_prices=np.empty((episodes, horizon, 2)),
                executed_quantities=np.empty((episodes, horizon, 2), dtype=np.int8),
                fees=np.empty((episodes, horizon)))
    obs = env.observe()
    for name in ("price", "signal", "cash", "inventory"):
        data[name][:, 0] = getattr(obs, name)
    for t in range(horizon):
        count = obs.admissible_actions.sum(axis=1)
        # Selecting a rank in the ordered valid IDs handles boundaries exactly.
        rank = np.floor(rng.random(episodes) * count).astype(int)
        cumulative = np.cumsum(obs.admissible_actions, axis=1)
        action = np.argmax(cumulative > rank[:, None], axis=1).astype(np.int8)
        feedback = env.step(action)
        data["actions"][:, t] = action
        data["action_probability"][:, t] = 1 / count
        for name in ("return_", "submitted", "fills", "execution_prices", "executed_quantities", "fees"):
            data[name][:, t] = getattr(feedback, name)
        obs = feedback.next_observation
        for name in ("price", "signal", "cash", "inventory"):
            data[name][:, t + 1] = getattr(obs, name)
    env.finalize()
    return PilotData(**data)
