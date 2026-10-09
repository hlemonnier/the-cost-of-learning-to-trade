"""Public-data policies and a separately labelled privileged reference.

The nine model parameters are a declared public prior, never a hidden environment
label. Posterior-weighted control is an approximation: future parameter learning
is not optimized. Only the active/noinfo treatment of future regime feedback differs.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from .filtering import filter_step
from .numerics import stable_argmax, control_key, manifest_fingerprint


CANDIDATE_THETA = np.repeat(np.array([.15, .35, .65]), 3)
CANDIDATE_KAPPA = np.tile(np.array([.002, .02, .10]), 3)
POLICIES = ('abstain', 'taker', 'independent', 'myopic', 'noinfo',
            'active', 'full_information')


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@dataclass
class Table:
    q_values: np.ndarray
    beliefs: np.ndarray
    metadata: dict


class TableBank:
    """Disk-backed tables computed identically for every public support point."""

    def __init__(self, directory, horizon, qmax, belief_points, quadrature_points,
                 belief_overrides=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.horizon, self.qmax = horizon, qmax
        self.belief_points, self.quadrature_points = belief_points, quadrature_points
        self.belief_overrides = dict(belief_overrides or {})
        self.tables = {}
        self._fingerprints = {}
        self.computation = []

    def table(self, theta, kappa, mode):
        from .control import solve_control, save_control, load_control
        points = self.belief_points
        if mode in {'active', 'noinfo'}:
            points = self.belief_overrides.get(f'{theta:g},{kappa:g}', points)
        signature = dict(theta=float(theta), kappa=float(kappa), mode=mode,
                         horizon=self.horizon, qmax=self.qmax,
                         belief_points=points,
                         quadrature_points=self.quadrature_points)
        key = control_key(signature)
        if key in self.tables:
            return self.tables[key]
        path = self.directory / key
        if not path.exists():
            solution = solve_control(**signature)
            save_control(path, solution, metadata_name='complete.json', specification=signature)
            del solution
        solution = load_control(path, metadata_name='complete.json', expected_specification=signature)
        metadata = {**signature, **deepcopy(solution.artifact)}
        result = Table(solution.q_values, solution.beliefs, metadata)
        self.tables[key] = result
        self._fingerprints[key] = manifest_fingerprint(solution.artifact)
        self.computation.append(metadata)
        return result

    def fingerprints(self):
        """Deterministic content/build/source identity for each evaluated table."""
        return deepcopy([self._fingerprints[key] for key in sorted(self._fingerprints)])

    def verify_loaded_tables(self):
        """Rehash and revalidate disk artifacts after evaluation, without repair."""
        from .control import load_control, CacheIntegrityError
        for key, before in self._fingerprints.items():
            solution = load_control(self.directory/key, metadata_name='complete.json',
                                    expected_specification=before['specification'])
            if manifest_fingerprint(solution.artifact) != before:
                raise CacheIntegrityError(f'Evaluated control artifact changed: {key}')
        return self.fingerprints()

    def learner_tables(self, mode):
        return [self.table(th, ka, mode) for th, ka in
                zip(CANDIDATE_THETA, CANDIDATE_KAPPA)]


def interpolate_scores(table, remaining, inventory, signal, beliefs, qmax):
    """Interpolates finite action values; legality is imposed by the public mask."""
    grid = table.beliefs
    if not isinstance(remaining, (int, np.integer)) or not 1 <= remaining < table.q_values.shape[0]:
        raise ValueError('remaining must identify a positive solved horizon')
    beliefs = np.asarray(beliefs)
    if not np.all(np.isfinite(beliefs)):
        raise ValueError('Beliefs must be finite')
    beliefs = np.clip(beliefs, grid[0], grid[-1])
    upper = np.clip(np.searchsorted(grid, beliefs, side='right'), 1, len(grid)-1)
    lower = upper-1
    frac = (beliefs-grid[lower])/(grid[upper]-grid[lower])
    low = table.q_values[remaining, inventory+qmax, signal+1, lower, :]
    high = table.q_values[remaining, inventory+qmax, signal+1, upper, :]
    # 0 * -inf is undefined at endpoints. Invalid actions will be masked below.
    if np.any(np.isnan(low) | np.isposinf(low) | np.isnan(high) | np.isposinf(high)):
        raise ValueError('Invalid nonfinite interpolation score')
    if not np.array_equal(np.isneginf(low), np.isneginf(high)):
        raise ValueError('Action admissibility changes across belief interpolation endpoints')
    low = np.where(np.isneginf(low), 0., low)
    high = np.where(np.isneginf(high), 0., high)
    return low+(high-low)*frac[:, None]


class GridPolicy:
    def __init__(self, mode, fit, episodes, bank):
        if mode not in {'active', 'noinfo', 'myopic'}:
            raise ValueError(mode)
        self.mode, self.bank = mode, bank
        self.tables = bank.learner_tables('active' if mode == 'myopic' else mode)
        self.initial_log_weights = np.array(fit.log_weights, dtype=float, copy=True)
        self.log_weights = np.broadcast_to(self.initial_log_weights,
                                           (episodes, len(CANDIDATE_THETA))).copy()
        self.beliefs = np.full_like(self.log_weights, .5)
        self.starting_posterior_hash = array_hash(self.initial_log_weights)
        self.predictive_loglik = np.zeros(episodes)

    def predict_regime(self):
        return np.sum(np.exp(self.log_weights)*self.beliefs, axis=1)

    def predict_rho(self):
        return np.sum(np.exp(self.log_weights)*CANDIDATE_THETA[None, :]
                      *(2*self.beliefs-1), axis=1)

    def choose(self, observation):
        if not 0 <= observation.t < observation.horizon:
            raise ValueError('Cannot choose an action after the terminal horizon')
        remaining = 1 if self.mode == 'myopic' else observation.horizon-observation.t
        weights = np.exp(self.log_weights)
        scores = np.zeros((len(observation.inventory), 11))
        for m, table in enumerate(self.tables):
            scores += weights[:, m, None]*interpolate_scores(
                table, remaining, observation.inventory, observation.signal,
                self.beliefs[:, m], self.bank.qmax)
        scores[~observation.admissible_actions] = -np.inf
        return stable_argmax(scores, observation.admissible_actions, axis=1).astype(np.int8)

    def update(self, feedback):
        self.beliefs, self.log_weights, self.predictive_loglik = filter_step(
            self.beliefs, self.log_weights, feedback.signal, feedback.return_,
            feedback.action, feedback.fills, CANDIDATE_THETA, CANDIDATE_KAPPA)


class FixedPolicy:
    def __init__(self, mode, episodes, bank):
        if mode not in {'abstain', 'taker', 'independent'}:
            raise ValueError(mode)
        self.mode, self.bank, self.episodes = mode, bank, episodes
        self.table = None if mode == 'abstain' else bank.table(0., .02, mode)
        self.starting_posterior_hash = 'not_data_dependent'
        self.predictive_loglik = np.zeros(episodes)

    def choose(self, observation):
        if not 0 <= observation.t < observation.horizon:
            raise ValueError('Cannot choose an action after the terminal horizon')
        if self.mode == 'abstain':
            return np.zeros(self.episodes, dtype=np.int8)
        scores = interpolate_scores(self.table, observation.horizon-observation.t,
                                    observation.inventory, observation.signal,
                                    np.full(self.episodes, .5), self.bank.qmax)
        scores[~observation.admissible_actions] = -np.inf
        valid = observation.admissible_actions.copy()
        if self.mode == 'taker':
            scores[:, 1:9] = -np.inf
            valid[:, 1:9] = False
        return stable_argmax(scores, valid, axis=1).astype(np.int8)

    def update(self, feedback):
        pass

    def predict_rho(self):
        return np.zeros(self.episodes)


class FullInformationReference:
    """Evaluator-only benchmark: explicitly receives current H, never future H."""

    def __init__(self, theta, kappa, bank):
        self.table = bank.table(theta, kappa, 'full')
        self.bank = bank
        self.starting_posterior_hash = 'privileged_reference'

    def choose(self, observation, current_regime):
        remaining = observation.horizon-observation.t
        if not 1 <= remaining < self.table.q_values.shape[0]:
            raise ValueError('remaining must identify a positive solved horizon')
        indices = (np.asarray(current_regime) > 0).astype(int)
        scores = np.array(self.table.q_values[remaining,
                          observation.inventory+self.bank.qmax,
                          observation.signal+1, indices, :], copy=True)
        scores[~observation.admissible_actions] = -np.inf
        return stable_argmax(scores, observation.admissible_actions, axis=1).astype(np.int8)

    def update(self, feedback):
        pass
