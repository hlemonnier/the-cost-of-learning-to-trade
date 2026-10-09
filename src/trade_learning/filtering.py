"""Exact finite-grid HMM likelihood from selected public execution feedback.

All returned observation log scores condition on observed returns: the common
Gaussian return density is omitted because it carries no regime/model evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from collections.abc import Mapping
import numpy as np
from scipy.special import log_ndtr, ndtri, logsumexp

from .model import ACTIONS, MU, SIGMA, fill_probability


def _candidate_vectors(candidate_theta, candidate_kappa=None):
    theta = np.atleast_1d(np.asarray(candidate_theta, dtype=float))
    if theta.ndim != 1 or theta.size < 1 or not np.isfinite(theta).all() or np.any(theta < 0) or np.any(theta >= 1):
        raise ValueError("candidate theta must be a nonempty vector in [0,1)")
    if candidate_kappa is None:
        return theta
    kappa = np.atleast_1d(np.asarray(candidate_kappa, dtype=float))
    if kappa.shape != theta.shape or not np.isfinite(kappa).all() or np.any(kappa < 0) or np.any(kappa > 1):
        raise ValueError("one candidate kappa in [0,1] is required per theta")
    return theta, kappa


def observation_log_likelihood(signal, return_, actions, fills, candidate_theta):
    """Log P(selected fills | observed return, current hidden sign, action).

    Output shape is (episodes,models,2), hidden signs ordered (-1,+1).
    Side outcomes are independent *conditional on return and hidden sign*.
    They are mixed jointly, not separately, over the uncertain hidden sign.
    Unsubmitted sides must be encoded -1; a submitted non-fill is evidence 0.
    """
    theta = _candidate_vectors(candidate_theta)
    x = np.atleast_1d(np.asarray(signal))
    r = np.atleast_1d(np.asarray(return_, dtype=float))
    actions = np.atleast_1d(np.asarray(actions))
    fills = np.asarray(fills)
    if x.ndim != 1 or r.shape != x.shape or actions.shape != x.shape or fills.shape != x.shape + (2,):
        raise ValueError("public feedback dimensions do not match")
    if not np.isin(x, (-1, 0, 1)).all() or not np.isfinite(r).all():
        raise ValueError("invalid public signal or return")
    if not np.issubdtype(actions.dtype, np.integer) or np.any(actions < 0) or np.any(actions > 10):
        raise ValueError("invalid action IDs")
    depths = ACTIONS[actions]
    submitted = depths >= 0
    if not np.all(fills[~submitted] == -1) or not np.isin(fills[submitted], (0, 1)).all():
        raise ValueError("fill masks must distinguish unsubmitted -1 from observed 0/1")
    z = (r - MU * x) / SIGMA
    rho = theta[:, None] * np.array([-1., 1.])[None, :]
    scale = np.sqrt(1 - theta**2)[None, :, None]
    log_likelihood = np.zeros((len(x), len(theta), 2))
    for side_index, side in enumerate((1, -1)):
        threshold = ndtri(fill_probability(x, side, np.maximum(depths[:, side_index], 0)))
        standardized = (threshold[:, None, None] - z[:, None, None] * side * rho[None, :, :]) / scale
        signed = np.where(fills[:, side_index, None, None] == 1, standardized, -standardized)
        contribution = log_ndtr(signed)
        log_likelihood += np.where(submitted[:, side_index, None, None], contribution, 0.)
    return log_likelihood


def _hidden_update(beliefs, log_likelihood, kappa):
    with np.errstate(divide="ignore", invalid="ignore"):
        log_prior = np.stack((np.log1p(-beliefs), np.log(beliefs)), axis=-1)
    log_joint = log_prior + log_likelihood
    log_evidence = logsumexp(log_joint, axis=-1)
    current_posterior = np.exp(log_joint[..., 1] - log_evidence)
    next_beliefs = kappa[None, :] + (1 - 2 * kappa[None, :]) * current_posterior
    return np.clip(next_beliefs, 0., 1.), log_evidence


def filter_step(beliefs, log_weights, signal, return_, actions, fills, candidate_theta, candidate_kappa):
    """Update model weights and predict each model's next-period regime belief.

    `beliefs[e,m]` is the pre-decision P(H_t=+1) conditional on model m.
    `log_weights[e,m]` can be extremely small finite values; retaining log
    weights between periods prevents irreversible floating-point model removal.
    Returned log score is log P(selected fills | observed return,past,action).
    No passive feedback makes this score zero and only predicts the regime.
    """
    theta, kappa = _candidate_vectors(candidate_theta, candidate_kappa)
    beliefs = np.asarray(beliefs, dtype=float)
    log_weights = np.asarray(log_weights, dtype=float)
    if beliefs.ndim != 2 or beliefs.shape[1] != theta.size or log_weights.shape != beliefs.shape:
        raise ValueError("beliefs and log weights must have shape (episodes,models)")
    if not np.isfinite(beliefs).all() or np.any(beliefs < 0) or np.any(beliefs > 1):
        raise ValueError("regime beliefs must be finite probabilities")
    if np.isnan(log_weights).any() or np.isposinf(log_weights).any() or not np.isfinite(logsumexp(log_weights, axis=1)).all():
        raise ValueError("each episode needs a finite, nonzero parameter prior")
    log_likelihood = observation_log_likelihood(signal, return_, actions, fills, theta)
    if log_likelihood.shape[:2] != beliefs.shape:
        raise ValueError("feedback episode count differs from filter state")
    next_beliefs, model_log_evidence = _hidden_update(beliefs, log_likelihood, kappa)
    prior = log_weights - logsumexp(log_weights, axis=1, keepdims=True)
    joint = prior + model_log_evidence
    predictive = logsumexp(joint, axis=1)
    return next_beliefs, joint - predictive[:, None], predictive


@dataclass(frozen=True, slots=True)
class GridFit:
    candidate_theta: np.ndarray
    candidate_kappa: np.ndarray
    log_weights: np.ndarray
    log_likelihood: np.ndarray
    episode_log_likelihood: np.ndarray

    def __post_init__(self):
        for field in fields(self):
            value = np.array(getattr(self, field.name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, field.name, value)

    @property
    def weights(self):
        result = np.exp(self.log_weights)
        result.setflags(write=False)
        return result

    def as_dict(self):
        return {field.name: np.array(getattr(self, field.name), copy=True) for field in fields(self)}

    to_dict = as_dict


def fit_grid_posterior(pilot, candidate_theta, candidate_kappa, prior_log_weights=None):
    """Fit all pilot episodes exactly within the declared finite model family.

    Takes a public PilotData object or mapping with signal, return_, actions,
    fills. Episodes reset hidden belief to one half; no evaluator labels enter.
    A uniform candidate prior is used unless an explicit log prior is given.
    Public signals can include T+1 endpoints or the T pre-decision values.
    """
    theta, kappa = _candidate_vectors(candidate_theta, candidate_kappa)
    take = pilot.__getitem__ if isinstance(pilot, Mapping) else lambda key: getattr(pilot, key)
    signal, returns, actions, fills = (np.asarray(take(key)) for key in ("signal", "return_", "actions", "fills"))
    if actions.ndim != 2 or returns.shape != actions.shape or fills.shape != actions.shape + (2,):
        raise ValueError("pilot actions, returns and fills must have episode/time dimensions")
    episodes, horizon = actions.shape
    if episodes < 1 or horizon < 1 or signal.shape not in ((episodes, horizon), (episodes, horizon + 1)):
        raise ValueError("invalid pilot signal endpoints or empty pilot")
    beliefs = np.full((episodes, len(theta)), .5)
    episode_ll = np.zeros_like(beliefs)
    for t in range(horizon):
        observation_ll = observation_log_likelihood(signal[:, t], returns[:, t], actions[:, t], fills[:, t], theta)
        beliefs, evidence = _hidden_update(beliefs, observation_ll, kappa)
        episode_ll += evidence
    log_likelihood = episode_ll.sum(axis=0)
    prior = np.full(len(theta), -np.log(len(theta))) if prior_log_weights is None else np.asarray(prior_log_weights, dtype=float)
    if prior.shape != theta.shape or np.isnan(prior).any() or np.isposinf(prior).any() or not np.isfinite(logsumexp(prior)):
        raise ValueError("invalid candidate model prior")
    joint = prior + log_likelihood
    log_weights = joint - logsumexp(joint)
    return GridFit(candidate_theta=theta, candidate_kappa=kappa, log_weights=log_weights,
                   log_likelihood=log_likelihood, episode_log_likelihood=episode_ll)
