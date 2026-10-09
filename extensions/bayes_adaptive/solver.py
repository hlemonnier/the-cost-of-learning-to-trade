"""Finite-horizon control on a genuine joint model/regime belief.

The four joint states are (M0,-1), (M0,+1), (M1,-1), (M1,+1).
Product coordinates (weight0, conditional_plus0, conditional_plus1) are only a
numerical representation. Bayes control never chooses a separate action for
each model. See SOLVER_DESIGN.md for failure modes declared before coding.

Gauss-Hermite integration and trilinear belief interpolation are approximations;
the resulting model-revelation values are NOT certified numerical bounds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.special import logsumexp, ndtr, ndtri, roots_hermitenorm

from trade_learning.control import solve_control
from trade_learning.filtering import observation_log_likelihood
from trade_learning.model import ACTIONS, CP, CT, DEPTH, H, K, LAMBDA, MU, SIGMA, admissible, fill_probability
from trade_learning.numerics import build_fingerprint, canonical_hash, file_sha256, numerical_contract, stable_argmax


FORMAT_VERSION = 3
BELIEF_GEOMETRIES = ("uniform", "endpoint_sine")
KINDS = ("bayes", "reveal1", "reveal2", "reveal3", "no_feedback", "frozen_model")
UPDATE_MODES = ("bayes", "no_feedback", "frozen_model")
DEFAULT_RETAIN_Q = ("bayes", "no_feedback", "frozen_model")


@dataclass(frozen=True)
class SolverSpec:
    theta: float = .35
    kappas: tuple[float, float] = (.002, .10)
    horizon: int = 30
    qmax: int = 2
    weight_points: int = 9
    belief_points: int = 17
    quadrature_points: int = 15
    known_belief_points: int = 321
    known_quadrature_points: int = 161
    belief_geometry: str = "uniform"

    def __post_init__(self):
        if not np.isfinite(self.theta) or not 0 <= self.theta < 1:
            raise ValueError("theta must be finite in [0,1)")
        kappas = tuple(float(k) for k in self.kappas)
        if len(kappas) != 2 or not np.isfinite(kappas).all() or min(kappas) < 0 or max(kappas) > 1:
            raise ValueError("exactly two finite transition probabilities are required")
        object.__setattr__(self, "kappas", kappas)
        if self.belief_geometry not in BELIEF_GEOMETRIES:
            raise ValueError(f"belief_geometry must be one of {BELIEF_GEOMETRIES}")
        for name, lower in (("horizon", 0), ("qmax", 1), ("weight_points", 2),
                            ("belief_points", 2), ("quadrature_points", 1),
                            ("known_belief_points", 2), ("known_quadrature_points", 1)):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")

    @property
    def grid_shape(self):
        return (self.weight_points, self.belief_points, self.belief_points)

    @property
    def grid_size(self):
        return int(np.prod(self.grid_shape))


def coordinates_to_joint(coordinates):
    """Convert (...,3) product coordinates to (...,4) joint probabilities."""
    c = np.asarray(coordinates, dtype=np.float64)
    if c.ndim < 1 or c.shape[-1] != 3 or not np.isfinite(c).all() or np.any((c < 0) | (c > 1)):
        raise ValueError("coordinates must have last axis 3 and lie in [0,1]")
    w, b0, b1 = np.moveaxis(c, -1, 0)
    return np.stack((w*(1-b0), w*b0, (1-w)*(1-b1), (1-w)*b1), axis=-1)


def _validated_joint(joint):
    p = np.asarray(joint, dtype=np.float64)
    if p.ndim < 1 or p.shape[-1] != 4 or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("joint belief must be a finite nonnegative array with last axis 4")
    mass = p.sum(axis=-1, keepdims=True)
    if np.any(np.abs(mass-1) > 1e-10):
        raise ValueError("joint probabilities must sum to one")
    return p/mass


def joint_to_coordinates(joint):
    """Canonical .5 conditional probability is used for a zero-weight model."""
    p = _validated_joint(joint)
    w = p[..., 0]+p[..., 1]
    w1 = p[..., 2]+p[..., 3]
    b0 = np.divide(p[..., 1], w, out=np.full_like(w, .5), where=w > 0)
    b1 = np.divide(p[..., 3], w1, out=np.full_like(w1, .5), where=w1 > 0)
    return np.stack((w, b0, b1), axis=-1)


def update_joint(joint, signal, return_, actions, fills, theta=.35, kappas=(.002, .10), mode="bayes"):
    """Condition on current selected feedback, then predict the next regime.

    Inputs are batched public data; no simulator model/regime label is accepted.
    ``frozen_model`` retains prior marginal model weights but updates both
    conditional hidden beliefs. ``no_feedback`` only predicts hidden regimes.
    These two are artificial planning updates, not the evaluator's filter.
    """
    if mode not in UPDATE_MODES:
        raise ValueError(f"unknown update mode {mode}")
    checked = SolverSpec(theta=theta, kappas=kappas, horizon=0)
    p = _validated_joint(joint)
    original_shape = p.shape
    p = p.reshape(-1, 4)
    ll = observation_log_likelihood(np.asarray(signal).reshape(-1), np.asarray(return_).reshape(-1),
                                    np.asarray(actions).reshape(-1), np.asarray(fills).reshape(-1, 2),
                                    np.repeat(checked.theta, 2))
    if len(ll) != len(p):
        raise ValueError("feedback count must equal joint belief count")
    coords = joint_to_coordinates(p)
    if mode == "no_feedback":
        b = coords[:, 1:]
        weights = np.stack((coords[:, 0], 1-coords[:, 0]), axis=1)
    else:
        with np.errstate(divide="ignore"):
            lp = np.log(p.reshape(-1, 2, 2))
        log_posterior = lp+ll
        log_mass = logsumexp(log_posterior, axis=(1, 2))
        if not np.isfinite(log_mass).all():
            raise ArithmeticError("observation has zero probability under the joint prior")
        posterior = np.exp(log_posterior-log_mass[:, None, None])
        updated_weights = posterior.sum(axis=2)
        b = np.divide(posterior[:, :, 1], updated_weights,
                      out=np.array(coords[:, 1:], copy=True), where=updated_weights > 0)
        weights = updated_weights if mode == "bayes" else np.stack((coords[:, 0], 1-coords[:, 0]), axis=1)
    k = np.asarray(checked.kappas)
    b = k[None, :]+(1-2*k[None, :])*b
    result = np.stack((weights*(1-b), weights*b), axis=-1).reshape(original_shape)
    return result/result.sum(axis=-1, keepdims=True)


@lru_cache(maxsize=64)
def grid_axes(spec):
    """Immutable physical probability axes; known-model grids are unchanged."""
    weight = np.linspace(0, 1, spec.weight_points)
    belief = np.linspace(0, 1, spec.belief_points)
    if spec.belief_geometry == "endpoint_sine":
        belief = np.sin(.5*np.pi*belief)**2
        count = spec.belief_points
        belief[count//2:] = 1-belief[:(count+1)//2][::-1]
        belief[0], belief[-1] = 0., 1.
        if count % 2:
            belief[count//2] = .5
    for axis in (weight, belief):
        if not np.all(np.diff(axis) > 0) or axis[0] != 0. or axis[-1] != 1.:
            raise ArithmeticError("belief grid must have increasing physical nodes and exact endpoints")
        axis.flags.writeable = False
    return weight, belief, belief


def grid_geometry_identity(spec):
    """Authenticate exact physical nodes as well as their generation recipe."""
    names = ["weight0", "conditional_plus0", "conditional_plus1"]
    axes = {name: axis.tolist() for name, axis in zip(names, grid_axes(spec))}
    endpoint = spec.belief_geometry == "endpoint_sine"
    return {"name": spec.belief_geometry, "axis_order": names,
            "axis_generation": ("uniform-weight_symmetric-sine-squared-beliefs-v1" if endpoint else "uniform-linspace-v1"),
            "interpolation": ("physical-interval-search-trilinear-v1" if endpoint else "uniform-scaled-trilinear-v1"),
            "axes": axes, "axes_sha256": canonical_hash(axes)}


def grid_coordinates(spec):
    axes = grid_axes(spec)
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _corners(coordinates, spec):
    """Return eight nonnegative product-grid interpolation coefficients."""
    c = np.asarray(coordinates, dtype=np.float64)
    if spec.belief_geometry == "uniform":
        # Retain v2 arithmetic exactly for a full-array regression comparison.
        scaled = np.clip(c, 0., 1.)*(np.array(spec.grid_shape)-1)
        low = np.minimum(np.floor(scaled).astype(np.int64), np.array(spec.grid_shape)-2)
        frac = scaled-low
    else:
        physical = np.clip(c, 0., 1.)
        cells, fractions = [], []
        for index, axis in enumerate(grid_axes(spec)):
            position = physical[..., index]
            cell = np.clip(np.searchsorted(axis, position, side="right")-1, 0, len(axis)-2)
            cells.append(cell)
            fractions.append((position-axis[cell])/(axis[cell+1]-axis[cell]))
        low, frac = np.stack(cells, axis=-1), np.stack(fractions, axis=-1)
    for dw in (0, 1):
        for db0 in (0, 1):
            for db1 in (0, 1):
                col = ((low[..., 0]+dw)*spec.belief_points+low[..., 1]+db0)*spec.belief_points+low[..., 2]+db1
                weight = (frac[..., 0] if dw else 1-frac[..., 0])
                weight = weight*(frac[..., 1] if db0 else 1-frac[..., 1])
                weight = weight*(frac[..., 2] if db1 else 1-frac[..., 2])
                yield col, weight


def _observation_outcomes(x, action, theta, nodes):
    """Visible outcome tuples (dq,L_minus,L_plus), conditional on z."""
    if action >= 9:
        yield (1 if action == 9 else -1), np.ones_like(nodes), np.ones_like(nodes)
        return
    submitted = [(s, int(k)) for s, k in zip((1, -1), ACTIONS[action]) if k >= 0]
    scale = np.sqrt(1-theta*theta)
    thresholds = [float(ndtri(fill_probability(x, s, k))) for s, k in submitted]
    for bits in range(1 << len(submitted)):
        lm, lp = np.ones_like(nodes), np.ones_like(nodes)
        dq = 0
        for j, ((side, _), threshold) in enumerate(zip(submitted, thresholds)):
            fill = (bits >> j) & 1
            direction = 1 if fill else -1
            dq += side*fill
            lm *= ndtr(direction*(threshold+theta*side*nodes)/scale)
            lp *= ndtr(direction*(threshold-theta*side*nodes)/scale)
        yield dq, lm, lp


def _outcome_kernel(coordinates, spec, lm, lp, quadrature_weights, mode, chunk_size=1024):
    """Integrate one visible outcome; kernels are shared across inventory."""
    data_parts, indices_parts, pointers = [], [], [0]
    k0, k1 = spec.kappas
    for start in range(0, len(coordinates), chunk_size):
        c = coordinates[start:start+chunk_size]
        w, b0, b1 = [c[:, k, None] for k in range(3)]
        m0 = (1-b0)*lm+b0*lp
        m1 = (1-b1)*lm+b1*lp
        mixture = w*m0+(1-w)*m1
        if mode == "no_feedback":
            mass = (mixture @ quadrature_weights)[:, None]
            target = np.concatenate((w, k0+(1-2*k0)*b0, k1+(1-2*k1)*b1), axis=1)[:, None, :]
        else:
            bp0 = np.divide(b0*lp, m0, out=np.broadcast_to(b0, m0.shape).copy(), where=m0 > 0)
            bp1 = np.divide(b1*lp, m1, out=np.broadcast_to(b1, m1.shape).copy(), where=m1 > 0)
            wp = np.divide(w*m0, mixture, out=np.broadcast_to(w, mixture.shape).copy(), where=mixture > 0)
            if mode == "frozen_model":
                wp = np.broadcast_to(w, mixture.shape)
            target = np.stack((wp, k0+(1-2*k0)*bp0, k1+(1-2*k1)*bp1), axis=-1)
            mass = mixture*quadrature_weights
        rows = np.broadcast_to(np.arange(len(c))[:, None], mass.shape).ravel()
        row_parts, col_parts, weight_parts = [], [], []
        for cols, coeff in _corners(target, spec):
            weights = (mass*coeff).ravel()
            keep = weights != 0
            row_parts.append(rows[keep])
            col_parts.append(cols.ravel()[keep])
            weight_parts.append(weights[keep])
        block = sparse.coo_matrix((np.concatenate(weight_parts),
                                   (np.concatenate(row_parts), np.concatenate(col_parts))),
                                  shape=(len(c), spec.grid_size)).tocsr()
        block.eliminate_zeros()
        data_parts.append(block.data)
        indices_parts.append(block.indices)
        pointers.extend((block.indptr[1:]+pointers[-1]).tolist())
    return sparse.csr_matrix((np.concatenate(data_parts), np.concatenate(indices_parts),
                              np.asarray(pointers, dtype=np.int64)), shape=(spec.grid_size, spec.grid_size))


def _save_kernel(path, kernel):
    path.mkdir(parents=True, exist_ok=False)
    for name in ("data", "indices", "indptr"):
        np.save(path/f"{name}.npy", getattr(kernel, name), allow_pickle=False)


def _load_kernel(path, size):
    arrays = [np.load(path/f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in ("data", "indices", "indptr")]
    return sparse.csr_matrix(tuple(arrays), shape=(size, size), copy=False)


class BeliefOperator:
    """Outcome kernels on belief only, with public inventory transitions."""
    def __init__(self, spec, mode, directory=None, progress=None):
        if mode not in UPDATE_MODES:
            raise ValueError(mode)
        self.spec, self.mode = spec, mode
        self.coordinates = grid_coordinates(spec)
        self.valid = admissible(np.arange(-spec.qmax, spec.qmax+1), spec.qmax)
        self.kernels = {}
        self.metadata = {"mode": mode, "kernels": [], "max_transition_mass_error": 0.}
        self.directory = Path(directory) if directory is not None else None
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=False)
        nodes, weights = roots_hermitenorm(spec.quadrature_points)
        weights = weights/np.sqrt(2*np.pi)
        started = time.perf_counter()
        for xi, x in enumerate((-1, 0, 1)):
            for action in range(11):
                grouped = {}
                for dq, lm, lp in _observation_outcomes(x, action, spec.theta, nodes):
                    kernel = _outcome_kernel(self.coordinates, spec, lm, lp, weights, mode)
                    grouped[dq] = kernel if dq not in grouped else grouped[dq]+kernel
                total_mass = np.zeros(spec.grid_size)
                for dq, kernel in sorted(grouped.items()):
                    total_mass += np.asarray(kernel.sum(axis=1)).ravel()
                    key = (xi, action, dq)
                    record = {"xi": xi, "action": action, "dq": dq, "nonzeros": int(kernel.nnz),
                              "bytes": int(kernel.data.nbytes+kernel.indices.nbytes+kernel.indptr.nbytes)}
                    if self.directory is not None:
                        path = self.directory/f"x{xi}-a{action}-dq{dq}"
                        _save_kernel(path, kernel)
                        self.kernels[key] = _load_kernel(path, spec.grid_size)
                    else:
                        self.kernels[key] = kernel
                    self.metadata["kernels"].append(record)
                error = float(np.max(np.abs(total_mass-1)))
                self.metadata["max_transition_mass_error"] = max(error, self.metadata["max_transition_mass_error"])
                if error > 2e-11:
                    raise ArithmeticError(f"{mode} x={x} action={action} transition mass error {error}")
            if progress is not None:
                progress({"phase": "operator", "mode": mode, "signal": x,
                          "elapsed_seconds": time.perf_counter()-started})
        self.metadata["build_seconds"] = time.perf_counter()-started
        self.metadata["total_bytes"] = sum(k["bytes"] for k in self.metadata["kernels"])
        self.metadata["total_nonzeros"] = sum(k["nonzeros"] for k in self.metadata["kernels"])

    def apply(self, continuation, rewards):
        """One common-action Bellman Q step; optional trailing family axis."""
        spec = self.spec
        v = np.asarray(continuation)
        single = v.ndim == 3
        if single:
            v = v[..., None]
        if v.shape[:3] != (2*spec.qmax+1, 3, spec.grid_size):
            raise ValueError("continuation dimensions do not match operator")
        expected = np.einsum("xy,qybk->qxbk", K, v, optimize=True)
        result = np.broadcast_to(rewards[..., None], rewards.shape+(v.shape[-1],)).copy()
        for (xi, action, dq), kernel in self.kernels.items():
            qi = np.flatnonzero(self.valid[:, action])
            if np.any((qi+dq < 0) | (qi+dq > 2*spec.qmax)):
                raise AssertionError("unsafe inventory outcome")
            next_values = expected[qi+dq, xi].transpose(1, 0, 2).reshape(spec.grid_size, -1)
            product = (kernel @ next_values).reshape(spec.grid_size, len(qi), v.shape[-1]).transpose(1, 0, 2)
            result[qi, xi, :, action, :] += product
        return result[..., 0] if single else result


def expected_rewards(spec, coordinates=None):
    """Analytic marked-wealth increment less the pre-decision inventory risk."""
    c = grid_coordinates(spec) if coordinates is None else np.asarray(coordinates)
    mean_sign = c[:, 0]*(2*c[:, 1]-1)+(1-c[:, 0])*(2*c[:, 2]-1)
    inventory = np.arange(-spec.qmax, spec.qmax+1)
    valid = admissible(inventory, spec.qmax)
    result = np.full((len(inventory), 3, len(c), 11), -np.inf)
    for qi, q in enumerate(inventory):
        for xi, x in enumerate((-1, 0, 1)):
            base = q*MU*x-LAMBDA*q*q
            for action in np.flatnonzero(valid[qi]):
                if action >= 9:
                    side = 1 if action == 9 else -1
                    result[qi, xi, :, action] = base+side*MU*x-H-CT
                else:
                    reward = np.full(len(c), base)
                    for side, depth in zip((1, -1), ACTIONS[action]):
                        if depth < 0:
                            continue
                        p = float(fill_probability(x, side, int(depth)))
                        threshold = ndtri(p)
                        density = np.exp(-.5*threshold**2)/np.sqrt(2*np.pi)
                        reward += p*(H+int(depth)*DEPTH-CP+side*MU*x)
                        reward -= SIGMA*spec.theta*mean_sign*density
                    result[qi, xi, :, action] = reward
    return result


def _select_values(qvalues):
    actions = stable_argmax(qvalues, axis=-1)
    return np.take_along_axis(qvalues, actions[..., None], axis=-1)[..., 0]


def _scalar_scores(table, remaining, inventory, signal, belief, qmax):
    """Interpolate the immutable known-model hidden-regime Q table."""
    count = table.shape[3]
    position = np.clip(belief, 0, 1)*(count-1)
    lo = np.minimum(np.floor(position).astype(np.int64), count-2)
    fraction = position-lo
    a = table[remaining, inventory+qmax, signal+1, lo]
    b = table[remaining, inventory+qmax, signal+1, lo+1]
    legal = admissible(inventory, qmax)
    a, b = np.where(legal, a, 0.), np.where(legal, b, 0.)
    return np.where(legal, a+(b-a)*fraction[..., None], -np.inf)


def _scalar_values(table, remaining, inventory, signal, belief, qmax):
    if remaining == 0:
        return -(H+CT)*np.abs(inventory)
    count = table.shape[3]
    position = np.clip(belief, 0, 1)*(count-1)
    lo = np.minimum(np.floor(position).astype(np.int64), count-2)
    fraction = position-lo
    a = _select_values(table[remaining, inventory+qmax, signal+1, lo])
    b = _select_values(table[remaining, inventory+qmax, signal+1, lo+1])
    return a+(b-a)*fraction


def _inputs(spec, remaining, inventory, signal, joint, allow_zero=False):
    minimum = 0 if allow_zero else 1
    if isinstance(remaining, (bool, np.bool_)) or not isinstance(remaining, (int, np.integer)) or not minimum <= remaining <= spec.horizon:
        raise ValueError(f"remaining must be an integer in [{minimum},{spec.horizon}]")
    p = _validated_joint(joint)
    q, x, _ = np.broadcast_arrays(np.asarray(inventory), np.asarray(signal), p[..., 0])
    if not np.isfinite(q).all() or np.any(q != np.floor(q)) or np.any(np.abs(q) > spec.qmax):
        raise ValueError("inventory is outside the configured integer range")
    if not np.isin(x, (-1, 0, 1)).all():
        raise ValueError("signal must be -1, 0 or 1")
    p = np.broadcast_to(p, q.shape+(4,))
    return q.astype(np.int64), x.astype(np.int64), joint_to_coordinates(p)


class ControlFamily:
    def __init__(self, spec, value_tables, action_tables, known_tables, directory, metadata, temporary=None):
        self.spec = spec
        self.value_tables = value_tables
        self.action_tables = action_tables
        self.known_tables = known_tables
        self.directory = Path(directory)
        self.metadata = metadata
        self._temporary = temporary

    def known_q_values(self, model_index, remaining, inventory, signal, joint):
        """Privileged model identity reference; hidden regime remains uncertain."""
        if model_index not in (0, 1):
            raise ValueError("model_index must be 0 or 1")
        q, x, c = _inputs(self.spec, remaining, inventory, signal, joint)
        return _scalar_scores(self.known_tables[model_index], remaining, q, x, c[..., 1+model_index], self.spec.qmax)

    def q_values(self, kind, remaining, inventory, signal, joint):
        """Batched interpolated feasible-policy action scores.

        ``reveal1`` uses the independent high-resolution known-model tables.
        Kinds without retained Q arrays can be evaluated with ``bellman_q``.
        """
        if kind == "weighted_q":
            kind = "reveal1"
        q, x, c = _inputs(self.spec, remaining, inventory, signal, joint)
        legal = admissible(q, self.spec.qmax)
        if kind == "reveal1":
            a = _scalar_scores(self.known_tables[0], remaining, q, x, c[..., 1], self.spec.qmax)
            b = _scalar_scores(self.known_tables[1], remaining, q, x, c[..., 2], self.spec.qmax)
            return np.where(legal, c[..., 0, None]*np.where(legal, a, 0.)+(1-c[..., 0, None])*np.where(legal, b, 0.), -np.inf)
        if kind not in self.action_tables:
            if kind not in KINDS:
                raise ValueError(f"unknown policy kind {kind}")
            return self.bellman_q(kind, remaining, inventory, signal, joint)
        result = np.zeros(q.shape+(11,))
        for col, weight in _corners(c, self.spec):
            block = self.action_tables[kind][remaining, q+self.spec.qmax, x+1, col]
            result += weight[..., None]*np.where(legal, block, 0.)
        return np.where(legal, result, -np.inf)

    def actions(self, kind, remaining, inventory, signal, joint):
        return stable_argmax(self.q_values(kind, remaining, inventory, signal, joint)).astype(np.int8)

    def values(self, kind, remaining, inventory, signal, joint):
        """Interpolate a Bellman table, not the realized value of its policy."""
        if kind == "weighted_q":
            kind = "reveal1"
        if kind not in self.value_tables:
            raise ValueError(f"unknown value kind {kind}")
        q, x, c = _inputs(self.spec, remaining, inventory, signal, joint, allow_zero=True)
        result = np.zeros(q.shape)
        for col, weight in _corners(c, self.spec):
            result += weight*self.value_tables[kind][remaining, q+self.spec.qmax, x+1, col]
        return result

    def revelation_value(self, delay, remaining, inventory, signal, joint):
        """Direct U0/U1 and interpolated U2/U3 numerical relaxed objectives.

        U0 interpolates each known-model V before posterior weighting. U1
        maximizes the direct weighted known-model Q at the supplied state. This
        differs off-grid from interpolating the nodal ``value-reveal1`` array.
        """
        if delay not in (0, 1, 2, 3):
            raise ValueError("supported revelation delays are 0,1,2,3")
        q, x, c = _inputs(self.spec, remaining, inventory, signal, joint, allow_zero=True)
        if remaining == 0:
            return -(H+CT)*np.abs(q)
        if delay == 0:
            a = _scalar_values(self.known_tables[0], remaining, q, x, c[..., 1], self.spec.qmax)
            b = _scalar_values(self.known_tables[1], remaining, q, x, c[..., 2], self.spec.qmax)
            return c[..., 0]*a+(1-c[..., 0])*b
        if delay == 1:
            return _select_values(self.q_values("reveal1", remaining, q, x, joint))
        return self.values(f"reveal{delay}", remaining, q, x, joint)

    def bellman_q(self, kind, remaining, inventory, signal, joint, *, update_mode=None):
        """Direct quadrature scoring against the stored continuation.

        This is intended for sparse diagnostics, not bulk policy evaluation.
        Supplying update_mode changes only this step's posterior map and retains
        the indicated policy's continuation, reward and observation law.
        """
        if kind == "weighted_q":
            kind = "reveal1"
        if kind not in KINDS:
            raise ValueError(kind)
        if kind == "reveal1" and update_mode is None:
            return self.q_values(kind, remaining, inventory, signal, joint)
        q, x, c = _inputs(self.spec, remaining, inventory, signal, joint)
        shape = q.shape
        q, x, c = q.ravel(), x.ravel(), c.reshape(-1, 3)
        mode = update_mode or (kind if kind in ("no_feedback", "frozen_model") else "bayes")
        if mode not in UPDATE_MODES:
            raise ValueError(mode)
        if kind == "reveal1":
            raise ValueError("one-step suppression with a revelation continuation is not implemented")
        continuation_kind = {"reveal2": "reveal1", "reveal3": "reveal2"}.get(kind, kind)
        nodes, weights = roots_hermitenorm(self.spec.quadrature_points)
        weights = weights/np.sqrt(2*np.pi)
        result = np.full((len(q), 11), -np.inf)
        valid = admissible(q, self.spec.qmax)
        p = coordinates_to_joint(c)
        k0, k1 = self.spec.kappas
        for xi, current_x in enumerate((-1, 0, 1)):
            for action in range(11):
                selected = np.flatnonzero((x == current_x) & valid[:, action])
                if not len(selected):
                    continue
                cq = q[selected]
                cc = c[selected]
                rewards = expected_rewards(self.spec, cc)[cq+self.spec.qmax, xi, np.arange(len(selected)), action]
                score = rewards.copy()
                w, b0, b1 = [cc[:, i, None] for i in range(3)]
                for dq, lm, lp in _observation_outcomes(current_x, action, self.spec.theta, nodes):
                    m0, m1 = (1-b0)*lm+b0*lp, (1-b1)*lm+b1*lp
                    mixture = w*m0+(1-w)*m1
                    if mode == "no_feedback":
                        target = np.concatenate((w, k0+(1-2*k0)*b0, k1+(1-2*k1)*b1), axis=1)[:, None, :]
                        target = np.broadcast_to(target, (len(selected), len(nodes), 3))
                    else:
                        bp0 = np.divide(b0*lp, m0, out=np.broadcast_to(b0, m0.shape).copy(), where=m0 > 0)
                        bp1 = np.divide(b1*lp, m1, out=np.broadcast_to(b1, m1.shape).copy(), where=m1 > 0)
                        wp = np.divide(w*m0, mixture, out=np.broadcast_to(w, mixture.shape).copy(), where=mixture > 0)
                        if mode == "frozen_model":
                            wp = np.broadcast_to(w, mixture.shape)
                        target = np.stack((wp, k0+(1-2*k0)*bp0, k1+(1-2*k1)*bp1), axis=-1)
                    continuation = np.zeros(mixture.shape)
                    target_joint = coordinates_to_joint(target)
                    for next_xi, next_x in enumerate((-1, 0, 1)):
                        continuation += K[xi, next_xi]*self.values(continuation_kind, remaining-1,
                                                                 cq[:, None]+dq, next_x, target_joint)
                    score += (mixture*continuation) @ weights
                result[selected, action] = score
        return result.reshape(shape+(11,))


def _source_identity():
    import trade_learning.control
    import trade_learning.filtering
    import trade_learning.model
    import trade_learning.numerics
    paths = {"solver.py": Path(__file__)}
    for module in (trade_learning.control, trade_learning.filtering, trade_learning.model, trade_learning.numerics):
        paths[module.__name__] = Path(module.__file__)
    files = {name: file_sha256(path) for name, path in paths.items()}
    return {"files": files, "sha256": canonical_hash(files)}


def _new_array(directory, name, shape, fill=None):
    array = np.lib.format.open_memmap(directory/f"{name}.npy", mode="w+", dtype=np.float64, shape=shape)
    if fill is not None:
        array[...] = fill
    return array


def _array_record(path):
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {"file": path.name, "shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": file_sha256(path), "bytes": path.stat().st_size}


def _flush_completed_q(table):
    """Persist a completed Q slice and release dispensable mapped pages.

    The Bellman recursion reads V continuations, never a preceding retained Q
    slice. Advisory release does not close the mapping or change array content.
    A platform without madvise still performs the required flush.
    """
    table.flush()
    try:
        import mmap
        table._mmap.madvise(mmap.MADV_DONTNEED)
        return True
    except (AttributeError, OSError):
        return False


def solve_family(spec=SolverSpec(), directory=None, retain_q=DEFAULT_RETAIN_Q,
                 progress: Callable[[dict], None] | None = None, keep_operators=False):
    """Build all six full-horizon recursions in an extension-only artifact.

    An existing complete directory is strictly validated and loaded. Existing
    partial/stale artifacts are rejected instead of silently repaired. A new
    temporary directory is used if no directory is supplied.
    """
    if not isinstance(spec, SolverSpec):
        raise TypeError("spec must be a SolverSpec")
    retain_q = tuple(retain_q)
    if len(set(retain_q)) != len(retain_q) or not set(retain_q).issubset(set(KINDS)-{"reveal1"}):
        raise ValueError("retain_q must contain unique supported joint-table kinds")
    temporary = None
    if directory is None:
        temporary = tempfile.TemporaryDirectory(prefix="trade-learning-bayes-")
        destination = Path(temporary.name)
    else:
        destination = Path(directory)
        if destination.exists():
            if (destination/"complete.json").exists():
                return load_family(destination, expected_spec=spec, expected_retain_q=retain_q)
            if any(destination.iterdir()):
                raise ValueError(f"partial control directory must be investigated or explicitly replaced: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source = _source_identity()
    build = build_fingerprint()
    shape = (spec.horizon+1, 2*spec.qmax+1, 3, spec.grid_size)
    terminal = -(H+CT)*np.abs(np.arange(-spec.qmax, spec.qmax+1))[:, None, None]
    values = {kind: _new_array(destination, f"value-{kind}", shape) for kind in KINDS}
    # Every positive-horizon slice is assigned in full before authentication.
    # Avoid dirtying many gigabytes of future Q slices before they are needed.
    qtables = {kind: _new_array(destination, f"q-{kind}", shape+(11,)) for kind in retain_q}
    for table in qtables.values():
        table[0] = -np.inf
    for table in values.values():
        table[0] = terminal
    known = []
    metadata = {"approximation": "Float64 trilinear product-belief interpolation and Gauss-Hermite continuation",
                "implementation_revision": "v3: authenticated physical belief geometry and flushed Q horizons; same Bellman equations",
                "certified_bound": False, "state_order": ["M0,H-", "M0,H+", "M1,H-", "M1,H+"],
                "flatten_order": "C order: weight0, conditional_plus0, conditional_plus1",
                "no_feedback_semantics": "prediction-only beliefs in planning; evaluator uses full exact filtering",
                "frozen_model_semantics": "freeze marginal model weights, update conditional regimes in planning; artificial intervention",
                "storage": {"format": "float64 NPY memory maps", "completed_q_horizon_flushes": 0,
                            "advisory_q_release_successes": 0},
                "operators": [], "known_models": []}
    for model, kappa in enumerate(spec.kappas):
        solution = solve_control(spec.theta, kappa, spec.horizon, spec.qmax,
                                 spec.known_belief_points, spec.known_quadrature_points, "active")
        table = _new_array(destination, f"known-q-{model}", solution.q_values.shape)
        table[...] = solution.q_values
        table.flush()
        known.append(table)
        metadata["known_models"].append(solution.metadata)
        del solution
        if progress is not None:
            progress({"phase": "known_model", "model": model, "elapsed_seconds": time.perf_counter()-started})
    family = ControlFamily(spec, values, qtables, known, destination, metadata, temporary)
    coordinates = grid_coordinates(spec)
    joint = coordinates_to_joint(coordinates)
    rewards = expected_rewards(spec, coordinates)
    for remaining in range(1, spec.horizon+1):
        for qi, q in enumerate(range(-spec.qmax, spec.qmax+1)):
            for xi, x in enumerate((-1, 0, 1)):
                values["reveal1"][remaining, qi, xi] = _select_values(family.q_values("reveal1", remaining, q, x, joint))
    for mode in UPDATE_MODES:
        operator_path = destination/f"operators-{mode}"
        operator = BeliefOperator(spec, mode, operator_path, progress)
        metadata["operators"].append(operator.metadata)
        kinds = ("bayes", "reveal2", "reveal3") if mode == "bayes" else (mode,)
        for remaining in range(1, spec.horizon+1):
            # All three continuations use remaining-1 and are already ready.
            # A single sparse pass avoids reading a disk-backed kernel thrice.
            if mode == "bayes":
                continuation = np.stack([values[kind][remaining-1]
                                         for kind in ("bayes", "reveal1", "reveal2")], axis=-1)
                combined_scores = operator.apply(continuation, rewards)
                del continuation
            else:
                combined_scores = operator.apply(values[mode][remaining-1], rewards)[..., None]
            for index, kind in enumerate(kinds):
                scores = combined_scores[..., index]
                values[kind][remaining] = _select_values(scores)
                if kind in qtables:
                    qtables[kind][remaining] = scores
                    released = _flush_completed_q(qtables[kind])
                    metadata["storage"]["completed_q_horizon_flushes"] += 1
                    metadata["storage"]["advisory_q_release_successes"] += int(released)
            del scores, combined_scores
            if progress is not None:
                progress({"phase": "bellman", "mode": mode, "remaining": remaining,
                          "elapsed_seconds": time.perf_counter()-started})
        del operator
        gc.collect()
        if not keep_operators:
            import shutil
            shutil.rmtree(operator_path)
    for table in list(values.values())+list(qtables.values())+known:
        table.flush()
    if _source_identity() != source:
        raise RuntimeError("source changed during numerical solve; refusing to authenticate artifact")
    metadata["solve_seconds"] = time.perf_counter()-started
    record = {"format_version": FORMAT_VERSION, "specification": asdict(spec), "retain_q": list(retain_q),
              "grid_geometry": grid_geometry_identity(spec),
              "source": source, "build": build, "numerical_contract": numerical_contract(),
              "metadata": metadata, "arrays": [_array_record(path) for path in sorted(destination.glob("*.npy"))]}
    record["artifact_sha256"] = canonical_hash(record)
    (destination/"complete.json").write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False)+"\n")
    family.metadata = record
    return family


def load_family(directory, *, expected_spec=None, expected_retain_q=None, verify_hashes=True):
    """Authenticate complete source/build/spec/array identity before reuse."""
    directory = Path(directory)
    record = json.loads((directory/"complete.json").read_text())
    signature = record.pop("artifact_sha256", None)
    if signature != canonical_hash(record) or record.get("format_version") != FORMAT_VERSION:
        raise ValueError("invalid Bayes control manifest identity/version")
    record["artifact_sha256"] = signature
    spec = SolverSpec(**record["specification"])
    if record.get("grid_geometry") != grid_geometry_identity(spec):
        raise ValueError("Bayes control physical grid geometry differs from its specification")
    if expected_spec is not None and spec != expected_spec:
        raise ValueError("Bayes control specification mismatch")
    if expected_retain_q is not None and list(expected_retain_q) != record["retain_q"]:
        raise ValueError("retained Q-table contract mismatch")
    if record["source"] != _source_identity() or record["build"] != build_fingerprint():
        raise ValueError("Bayes control source/build differs from this process")
    if record["numerical_contract"] != numerical_contract():
        raise ValueError("Bayes control tie/precision contract differs")
    arrays = {}
    for item in record["arrays"]:
        path = directory/item["file"]
        if path.parent != directory or not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError("invalid or missing Bayes control array")
        if verify_hashes and file_sha256(path) != item["sha256"]:
            raise ValueError(f"Bayes control array hash mismatch: {path.name}")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(value.shape) != item["shape"] or str(value.dtype) != item["dtype"] or value.dtype != np.float64:
            raise ValueError(f"Bayes control array shape/dtype mismatch: {path.name}")
        arrays[path.stem] = value
    expected_names = {f"value-{kind}" for kind in KINDS} | {f"q-{kind}" for kind in record["retain_q"]} | {"known-q-0", "known-q-1"}
    if set(arrays) != expected_names:
        raise ValueError("missing, duplicate or unexpected Bayes control tables")
    shape = (spec.horizon+1, 2*spec.qmax+1, 3, spec.grid_size)
    if any(arrays[f"value-{kind}"].shape != shape for kind in KINDS):
        raise ValueError("value-table axes violate solver specification")
    if any(arrays[f"q-{kind}"].shape != shape+(11,) for kind in record["retain_q"]):
        raise ValueError("Q-table axes violate solver specification")
    known_shape = (spec.horizon+1, 2*spec.qmax+1, 3, spec.known_belief_points, 11)
    if any(arrays[f"known-q-{model}"].shape != known_shape for model in (0, 1)):
        raise ValueError("known-model table axes violate solver specification")
    return ControlFamily(spec, {kind: arrays[f"value-{kind}"] for kind in KINDS},
                         {kind: arrays[f"q-{kind}"] for kind in record["retain_q"]},
                         [arrays[f"known-q-{model}"] for model in (0, 1)], directory, record)


def numerical_probes(config):
    """Fixed public-belief probes, independent of all economic outcomes."""
    count = int(config["refinement"]["random_probes"])
    rng = np.random.default_rng(np.random.SeedSequence([
        config["root_seed"], config["seed_namespaces"]["numerical_probes"], 0, 0]))
    random_joint = rng.dirichlet(np.ones(4), size=count)
    qmax = int(config["qmax"])
    random_q = rng.integers(-qmax, qmax+1, size=count)
    random_x = rng.integers(-1, 2, size=count)
    corners = np.stack(np.meshgrid([0., .5, 1.], [0., .5, 1.], [0., .5, 1.], indexing="ij"), axis=-1).reshape(-1, 3)
    boundary_joint = np.tile(coordinates_to_joint(corners), (3*(2*qmax+1), 1))
    boundary_q = np.repeat(np.arange(-qmax, qmax+1), 3*len(corners))
    boundary_x = np.tile(np.repeat(np.arange(-1, 2), len(corners)), 2*qmax+1)
    mirror_count = min(128, count)
    joint = np.concatenate((random_joint, boundary_joint, random_joint[:mirror_count]))
    inventory = np.concatenate((random_q, boundary_q, -random_q[:mirror_count]))
    signal = np.concatenate((random_x, boundary_x, -random_x[:mirror_count]))
    initial = coordinates_to_joint(np.column_stack((np.array([0., .25, .5, .75, 1.]), np.full(5, .5), np.full(5, .5))))
    return {"joint": joint, "inventory": inventory, "signal": signal,
            "initial_joint": initial, "random_count": count, "boundary_count": len(boundary_joint),
            "reflection_count": mirror_count}


def compare_families(coarse, fine, probes, thresholds):
    """All-horizon pointwise refinement; no policy-economic observations."""
    if coarse.spec.horizon != fine.spec.horizon or coarse.spec.qmax != fine.spec.qmax:
        raise ValueError("refinement families must share the horizon and inventory contract")
    q, x, p = (probes[key] for key in ("inventory", "signal", "joint"))
    initial = probes["initial_joint"]
    records = []
    overall = {"max_initial_value_change": 0., "max_probe_value_change": 0.,
               "max_finite_q_change": 0., "max_robust_disagreement_fraction": 0.}
    for remaining in range(1, fine.spec.horizon+1):
        for kind in KINDS:
            initial_change = float(np.max(np.abs(coarse.values(kind, remaining, 0, 0, initial)-fine.values(kind, remaining, 0, 0, initial))))
            value_change = float(np.max(np.abs(coarse.values(kind, remaining, q, x, p)-fine.values(kind, remaining, q, x, p))))
            entry = {"remaining": remaining, "kind": kind, "initial_value_change": initial_change,
                     "probe_value_change": value_change}
            overall["max_initial_value_change"] = max(overall["max_initial_value_change"], initial_change)
            overall["max_probe_value_change"] = max(overall["max_probe_value_change"], value_change)
            if kind in DEFAULT_RETAIN_Q:
                qa = coarse.q_values(kind, remaining, q, x, p)
                qb = fine.q_values(kind, remaining, q, x, p)
                finite = np.isfinite(qb)
                if not np.array_equal(finite, np.isfinite(qa)):
                    raise ArithmeticError("refinement changed action admissibility")
                qchange = float(np.max(np.abs(qa[finite]-qb[finite])))
                ca, cb = stable_argmax(qa), stable_argmax(qb)
                ordered = np.sort(qb, axis=1)
                robust = ordered[:, -1]-ordered[:, -2] > thresholds["robust_gap"]
                denominator = int(np.sum(robust))
                numerator = int(np.sum(robust & (ca != cb)))
                disagreement = numerator/denominator if denominator else 0.
                entry.update(finite_q_change=qchange, robust_disagreements=numerator,
                             robust_probe_count=denominator, robust_disagreement_fraction=disagreement)
                overall["max_finite_q_change"] = max(overall["max_finite_q_change"], qchange)
                overall["max_robust_disagreement_fraction"] = max(overall["max_robust_disagreement_fraction"], disagreement)
            records.append(entry)
    passes = (overall["max_initial_value_change"] <= thresholds["initial"]
              and overall["max_probe_value_change"] <= thresholds["probe_sup"]
              and overall["max_finite_q_change"] <= thresholds["probe_sup"]
              and overall["max_robust_disagreement_fraction"] <= thresholds["robust_disagreement"])
    return {"coarse_specification": asdict(coarse.spec), "fine_specification": asdict(fine.spec),
            "coarse_artifact_sha256": coarse.metadata["artifact_sha256"],
            "fine_artifact_sha256": fine.metadata["artifact_sha256"],
            "probe_count": len(p), "all_horizons": list(range(1, fine.spec.horizon+1)),
            "quantities": {"values": list(KINDS), "action_values": list(DEFAULT_RETAIN_Q),
                           "reveal1_value": "interpolated nodal relaxation, not direct off-grid weighted-Q maximum",
                           "disagreement_aggregation": "maximum over each policy and remaining horizon"},
            "summary": overall, "passes_predeclared_rule": bool(passes), "by_horizon_policy": records}


def known_reference_refinement(config, probes, output_directory, progress=None):
    """Independently vary the scalar known-model belief and quadrature axes."""
    reference = config["known_reference"]
    resolutions = ((reference["belief_points"], reference["quadrature_points"]),
                   (reference["refined_belief_points"], reference["quadrature_points"]),
                   (reference["belief_points"], reference["refined_quadrature_points"]))
    coords = joint_to_coordinates(probes["joint"])
    q, x = probes["inventory"], probes["signal"]
    models = []
    thresholds = config["refinement"]
    for model, kappa in enumerate(config["kappas"]):
        solutions = []
        for beliefs, quadrature in resolutions:
            solution = solve_control(config["theta"], kappa, config["horizon"], config["qmax"], beliefs, quadrature, "active")
            solutions.append(solution)
            if progress is not None:
                progress({"phase": "known_refinement", "model": model, "belief_points": beliefs,
                          "quadrature_points": quadrature, "solve_seconds": solution.metadata["solve_seconds"]})
        comparisons = []
        for axis, fine in zip(("belief", "quadrature"), solutions[1:]):
            base = solutions[0]
            by_horizon = []
            for remaining in range(1, config["horizon"]+1):
                qa = _scalar_scores(base.q_values, remaining, q, x, coords[:, model+1], config["qmax"])
                qb = _scalar_scores(fine.q_values, remaining, q, x, coords[:, model+1], config["qmax"])
                finite = np.isfinite(qb)
                va = _scalar_values(base.q_values, remaining, q, x, coords[:, model+1], config["qmax"])
                vb = _scalar_values(fine.q_values, remaining, q, x, coords[:, model+1], config["qmax"])
                ia = _scalar_values(base.q_values, remaining, np.array(0), np.array(0), np.array(.5), config["qmax"])
                ib = _scalar_values(fine.q_values, remaining, np.array(0), np.array(0), np.array(.5), config["qmax"])
                ordered = np.sort(qb, axis=1)
                robust = ordered[:, -1]-ordered[:, -2] > thresholds["robust_gap"]
                denominator = int(np.sum(robust))
                disagreement = int(np.sum(robust & (stable_argmax(qa) != stable_argmax(qb))))
                by_horizon.append({"remaining": remaining, "initial_value_change": float(abs(ia-ib)),
                                   "probe_value_change": float(np.max(np.abs(va-vb))),
                                   "finite_q_change": float(np.max(np.abs(qa[finite]-qb[finite]))),
                                   "robust_disagreements": disagreement, "robust_probe_count": denominator,
                                   "robust_disagreement_fraction": disagreement/denominator if denominator else 0.})
            summary = {key: max(row[key] for row in by_horizon) for key in
                       ("initial_value_change", "probe_value_change", "finite_q_change", "robust_disagreement_fraction")}
            passes = (summary["initial_value_change"] <= thresholds["initial"]
                      and summary["probe_value_change"] <= thresholds["probe_sup"]
                      and summary["finite_q_change"] <= thresholds["probe_sup"]
                      and summary["robust_disagreement_fraction"] <= thresholds["robust_disagreement"])
            comparisons.append({"axis": axis, "summary": summary, "passes_predeclared_rule": bool(passes),
                                "by_horizon": by_horizon})
        models.append({"model_index": model, "kappa": kappa, "resolutions": resolutions,
                       "comparisons": comparisons})
    result = {"models": models, "passes_predeclared_rule": all(c["passes_predeclared_rule"] for m in models for c in m["comparisons"]),
              "source": _source_identity(), "build": build_fingerprint()}
    (Path(output_directory)/"known_reference_checks.json").write_text(json.dumps(result, sort_keys=True, indent=2)+"\n")
    return result


def run_numerical_refinement(config_path, output_directory, cache_directory, progress=None, *, belief_geometry="uniform"):
    """Frozen coordinate-refinement rule; retains every failed comparison.

    Start at the second grid and quadrature levels. At each iteration vary each
    axis independently against its immediately preceding level. Advance only
    failing axes. The returned fine setting must pass both comparisons. Exhausting
    a failing axis is explicitly unresolved, never an economic-evaluation pass.
    """
    if belief_geometry not in BELIEF_GEOMETRIES:
        raise ValueError(f"belief_geometry must be one of {BELIEF_GEOMETRIES}")
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_directory)
    cache.mkdir(parents=True, exist_ok=True)
    probes = numerical_probes(config)
    probes_path = output/"numerical_probes.npz"
    np.savez_compressed(probes_path, **{key: value for key, value in probes.items() if isinstance(value, np.ndarray)})
    source = _source_identity()
    record = {"protocol_config_sha256": file_sha256(config_path), "source": source, "build": build_fingerprint(),
              "belief_geometry": belief_geometry,
              "probe_sha256": file_sha256(probes_path), "probe_design": {key: value for key, value in probes.items() if not isinstance(value, np.ndarray)},
              "selection_rule": "start second levels, independently compare preceding levels, advance only failing axes",
              "thresholds": config["refinement"], "comparisons": [], "selected_resolution": None,
              "passes_predeclared_rule": False, "certified_bound": False}

    def write_record():
        (output/"numerical_checks.json").write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False)+"\n")

    record["known_reference"] = known_reference_refinement(config, probes, output, progress)
    write_record()
    grid_levels, quadrature_levels = config["joint_grid_levels"], config["quadrature_levels"]
    gi, hi = 1, 1
    built = {}

    def get_family(grid_index, quadrature_index):
        key = (grid_index, quadrature_index)
        if key in built:
            return built[key]
        grid = grid_levels[grid_index]
        if grid[1] != grid[2]:
            raise ValueError("solver requires equal conditional-regime grid counts")
        spec = SolverSpec(theta=config["theta"], kappas=tuple(config["kappas"]), horizon=config["horizon"], qmax=config["qmax"],
                          weight_points=grid[0], belief_points=grid[1], quadrature_points=quadrature_levels[quadrature_index],
                          known_belief_points=config["known_reference"]["belief_points"],
                          known_quadrature_points=config["known_reference"]["quadrature_points"],
                          belief_geometry=belief_geometry)
        directory = cache/f"w{spec.weight_points}-b{spec.belief_points}-gh{spec.quadrature_points}"
        family = solve_family(spec, directory, progress=progress)
        built[key] = family
        if progress is not None:
            progress({"phase": "family_complete", "specification": asdict(spec), "directory": str(directory),
                      "solve_seconds": family.metadata["metadata"]["solve_seconds"],
                      "artifact_sha256": family.metadata["artifact_sha256"]})
        return family

    while True:
        fine = get_family(gi, hi)
        grid_coarse = get_family(gi-1, hi)
        quadrature_coarse = get_family(gi, hi-1)
        grid_comparison = compare_families(grid_coarse, fine, probes, config["refinement"])
        quadrature_comparison = compare_families(quadrature_coarse, fine, probes, config["refinement"])
        grid_comparison["axis"] = "belief"
        quadrature_comparison["axis"] = "quadrature"
        record["comparisons"].extend((grid_comparison, quadrature_comparison))
        grid_pass = grid_comparison["passes_predeclared_rule"]
        quadrature_pass = quadrature_comparison["passes_predeclared_rule"]
        record["last_attempted_resolution"] = asdict(fine.spec)
        if progress is not None:
            progress({"phase": "refinement_comparison", "grid_index": gi, "quadrature_index": hi,
                      "belief": grid_comparison["summary"], "belief_pass": grid_pass,
                      "quadrature": quadrature_comparison["summary"], "quadrature_pass": quadrature_pass})
        write_record()
        if grid_pass and quadrature_pass:
            record["selected_resolution"] = asdict(fine.spec)
            record["selected_directory"] = str(fine.directory.resolve())
            record["selected_artifact_sha256"] = fine.metadata["artifact_sha256"]
            record["selected_grid_geometry"] = fine.metadata["grid_geometry"]
            record["passes_predeclared_rule"] = record["known_reference"]["passes_predeclared_rule"]
            record["status"] = "resolved_on_prespecified_probes" if record["passes_predeclared_rule"] else "known_reference_unresolved"
            break
        exhausted = ((not grid_pass and gi+1 == len(grid_levels)) or
                     (not quadrature_pass and hi+1 == len(quadrature_levels)))
        if exhausted:
            record["status"] = "unresolved_at_prespecified_axis_limit"
            record["unresolved_axes"] = [axis for axis, passed in (("belief", grid_pass), ("quadrature", quadrature_pass)) if not passed]
            break
        gi += int(not grid_pass)
        hi += int(not quadrature_pass)
        # Preserve disk artifacts while limiting live array mappings.
        built.clear()
        del fine, grid_coarse, quadrature_coarse
        gc.collect()
    if _source_identity() != source:
        raise RuntimeError("solver source changed during refinement")
    write_record()
    return record


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    refine = subparsers.add_parser("refine", help="run the frozen all-horizon numerical refinement")
    refine.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    refine.add_argument("--output", type=Path, required=True)
    refine.add_argument("--cache", type=Path, required=True)
    refine.add_argument("--belief-geometry", choices=BELIEF_GEOMETRIES, default="uniform")
    solve = subparsers.add_parser("solve", help="solve a numerical family without economic evaluation")
    solve.add_argument("--directory", type=Path, required=True)
    solve.add_argument("--weight-points", type=int, default=17)
    solve.add_argument("--belief-points", type=int, default=33)
    solve.add_argument("--quadrature-points", type=int, default=25)
    solve.add_argument("--horizon", type=int, default=30)
    solve.add_argument("--belief-geometry", choices=BELIEF_GEOMETRIES, default="uniform")
    args = parser.parse_args()
    progress = lambda event: print(json.dumps(event, sort_keys=True), flush=True)
    if args.command == "refine":
        result = run_numerical_refinement(args.config, args.output, args.cache, progress,
                                          belief_geometry=args.belief_geometry)
        progress({"phase": "numerical_checks_complete", "status": result["status"],
                  "passes_predeclared_rule": result["passes_predeclared_rule"],
                  "selected_resolution": result["selected_resolution"]})
    else:
        family = solve_family(SolverSpec(weight_points=args.weight_points, belief_points=args.belief_points,
                                         quadrature_points=args.quadrature_points, horizon=args.horizon,
                                         belief_geometry=args.belief_geometry),
                              directory=args.directory, progress=progress)
        progress({"phase": "complete", "artifact_sha256": family.metadata["artifact_sha256"],
                  "directory": str(family.directory)})


if __name__ == "__main__":
    main()
