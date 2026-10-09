"""Finite-horizon control with scalar Bayesian beliefs and checked quadrature.

The exact model is integrated numerically: this is an approximation to its
continuous-belief Bellman recursion, not a certified optimum. Verification and
failure modes were specified before implementation in verification/verify_control.py
and report/control_notes.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from scipy import sparse
from scipy.special import ndtr, ndtri, roots_hermitenorm

from .model import ACTIONS, CP, CT, DEPTH, H, K, LAMBDA, MU, SIGMA, admissible, fill_probability
from .numerics import (CONTROL_FORMAT_VERSION, stable_argmax, numerical_contract,
                       source_fingerprint, build_fingerprint, file_sha256,
                       artifact_fingerprint, control_key)


MODES = frozenset(("active", "noinfo", "independent", "taker", "full"))


@dataclass
class ControlSolution:
    theta: float
    kappa: float
    horizon: int
    qmax: int
    quadrature_points: int
    mode: str
    beliefs: np.ndarray
    values: np.ndarray
    q_values: np.ndarray
    metadata: dict
    artifact: dict | None = None

    def save(self, path):
        save_control(path, self)


def _observation_outcomes(x, action, theta, nodes):
    """Return (inventory change,L_minus(z),L_plus(z)) for every visible outcome."""
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
            filled = (bits >> j) & 1
            dq += side*filled
            direction = 1 if filled else -1
            # Phi(-u), rather than 1-Phi(u), preserves small non-fill likelihoods.
            lm *= ndtr(direction*(threshold+theta*side*nodes)/scale)
            lp *= ndtr(direction*(threshold-theta*side*nodes)/scale)
        yield dq, lm, lp


def _belief_kernel(beliefs, kappa, lm, lp, weights, active):
    """Sparse outcome-weighted transition on belief; rows need not sum to one."""
    count = len(beliefs)
    b = beliefs[:, None]
    mixture = (1-b)*lm[None, :]+b*lp[None, :]
    if active:
        posterior = np.broadcast_to(b, mixture.shape).copy()
        np.divide(b*lp[None, :], mixture, out=posterior, where=mixture > 0)
        target = np.clip(kappa+(1-2*kappa)*posterior, 0, 1)
        mass = mixture*weights[None, :]
    else:
        target = np.clip(kappa+(1-2*kappa)*b, 0, 1)
        mass = (mixture @ weights)[:, None]
    position = target*(count-1)
    low = np.floor(position).astype(np.int64)
    high = np.minimum(low+1, count-1)
    upper = position-low
    rows = np.broadcast_to(np.arange(count)[:, None], mass.shape).ravel()
    cols = np.concatenate((low.ravel(), high.ravel()))
    data = np.concatenate(((mass*(1-upper)).ravel(), (mass*upper).ravel()))
    kernel = sparse.coo_matrix((data, (np.tile(rows, 2), cols)), shape=(count, count)).tocsr()
    kernel.eliminate_zeros()
    return kernel


def _expected_rewards(theta, qmax, beliefs, valid):
    """Analytic expected marked-wealth increment minus pre-decision risk."""
    inventory = np.arange(-qmax, qmax+1)
    rewards = np.full((len(inventory), 3, len(beliefs), 11), -np.inf)
    mean_regime = 2*beliefs-1
    for qi, q in enumerate(inventory):
        for xi, x in enumerate((-1, 0, 1)):
            base = q*MU*x-LAMBDA*q*q
            for action in np.flatnonzero(valid[qi]):
                if action >= 9:
                    side = 1 if action == 9 else -1
                    rewards[qi, xi, :, action] = base+side*MU*x-H-CT
                else:
                    reward = np.full(len(beliefs), base, dtype=np.float64)
                    for side, depth in zip((1, -1), ACTIONS[action]):
                        if depth < 0:
                            continue
                        p = float(fill_probability(x, side, int(depth)))
                        threshold = ndtri(p)
                        density = np.exp(-0.5*threshold*threshold)/np.sqrt(2*np.pi)
                        reward += p*(H+depth*DEPTH-CP+side*MU*x)
                        reward -= SIGMA*mean_regime*theta*density
                    rewards[qi, xi, :, action] = reward
    return rewards


def _build_operator(theta, kappa, qmax, beliefs, quadrature_points, mode):
    """Kernel acts on V averaged over next signal, avoiding threefold duplication."""
    started = time.perf_counter()
    effective_theta = 0.0 if mode == "independent" else theta
    nodes, weights = roots_hermitenorm(quadrature_points)
    weights = weights/np.sqrt(2*np.pi)
    nb = len(beliefs)
    nq = 2*qmax+1
    valid = admissible(np.arange(-qmax, qmax+1), qmax).copy()
    if mode == "taker":
        valid[:, 1:9] = False
    rewards = _expected_rewards(effective_theta, qmax, beliefs, valid)
    row_pieces, col_pieces, data_pieces = [], [], []
    for xi, x in enumerate((-1, 0, 1)):
        for action in range(11):
            qs = np.flatnonzero(valid[:, action])
            if not len(qs):
                continue
            for dq, lm, lp in _observation_outcomes(x, action, effective_theta, nodes):
                kernel = _belief_kernel(beliefs, kappa, lm, lp, weights, mode == "active")
                coo = kernel.tocoo()
                for qi in qs:
                    qj = qi+dq
                    if not 0 <= qj < nq:
                        raise AssertionError("Admissibility mask admitted unsafe fill outcome")
                    row_pieces.append((qi*3+xi)*nb*11+coo.row*11+action)
                    col_pieces.append((qj*3+xi)*nb+coo.col)
                    data_pieces.append(coo.data)
    nstate = nq*3*nb
    transition = sparse.coo_matrix(
        (np.concatenate(data_pieces), (np.concatenate(row_pieces), np.concatenate(col_pieces))),
        shape=(nstate*11, nstate), dtype=np.float64,
    ).tocsr()
    transition.eliminate_zeros()
    mass = np.asarray(transition.sum(axis=1)).reshape(nq, 3, nb, 11)
    expected_mass = np.broadcast_to(valid[:, None, None, :], mass.shape)
    mass_error = float(np.max(np.abs(mass-expected_mass)))
    if mass_error > 1e-11:
        raise ArithmeticError(f"Transition kernel mass error {mass_error:g}")
    return transition, rewards, {
        "operator_build_seconds": time.perf_counter()-started,
        "transition_nonzeros": int(transition.nnz),
        "transition_storage_bytes": int(transition.data.nbytes+transition.indices.nbytes
                                        +transition.indptr.nbytes),
        "max_transition_mass_error": mass_error,
        "effective_planning_theta": effective_theta,
    }


def _bellman_step(transition, rewards, continuation):
    # K rows are current x; columns are next x. Preserve (q,current x,b) order.
    expected_next_signal = np.einsum("xy,qyb->qxb", K, continuation, optimize=True)
    return rewards+(transition @ expected_next_signal.ravel()).reshape(rewards.shape)


def solve_control(theta, kappa, horizon, qmax, belief_points, quadrature_points, mode="active"):
    """Solve the declared finite-horizon approximate Bellman problem.

    q_values[n,q+qmax,x+1,bindex,a] uses periods remaining n; invalid actions
    and the unused q_values[0] are -inf. Full mode always uses beliefs [0,1],
    interpreted as observed current regimes, with exact next-regime averaging.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown control mode {mode!r}; expected one of {sorted(MODES)}")
    if not (np.isfinite(theta) and 0 <= theta < 1):
        raise ValueError("theta must lie in [0,1)")
    if not (np.isfinite(kappa) and 0 <= kappa <= 1):
        raise ValueError("kappa must lie in [0,1]")
    for name, value, minimum in (("horizon", horizon, 0), ("qmax", qmax, 1),
                                  ("belief_points", belief_points, 2),
                                  ("quadrature_points", quadrature_points, 1)):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    started = time.perf_counter()
    beliefs = np.linspace(0, 1, 2 if mode == "full" else belief_points, dtype=np.float64)
    shape = (horizon+1, 2*qmax+1, 3, len(beliefs))
    values = np.empty(shape, dtype=np.float64)
    q_values = np.full((*shape, 11), -np.inf, dtype=np.float64)
    values[0] = -(H+CT)*np.abs(np.arange(-qmax, qmax+1))[:, None, None]
    transition, rewards, metadata = _build_operator(theta, kappa, qmax, beliefs, quadrature_points, mode)
    for remaining in range(1, horizon+1):
        q_values[remaining] = _bellman_step(transition, rewards, values[remaining-1])
        chosen = stable_argmax(q_values[remaining])
        values[remaining] = np.take_along_axis(q_values[remaining], chosen[..., None], -1)[..., 0]
    metadata.update({
        "solve_seconds": time.perf_counter()-started,
        "table_storage_bytes": int(values.nbytes+q_values.nbytes+beliefs.nbytes),
        "approximation": "Gauss-Hermite continuation integration; uniform linear belief interpolation",
        "reward_integration": "analytic expected marked-wealth increment",
        "tie_breaking": numerical_contract(),
        "requested_belief_points": int(belief_points),
        "noinfo_semantics": "prediction-only belief in planning; runtime filtering external",
        "full_semantics": "current regime observed, next regime averaged; no future innovation access",
    })
    return ControlSolution(float(theta), float(kappa), int(horizon), int(qmax),
                           int(quadrature_points), mode, beliefs, values, q_values, metadata)


def feedback_suppressed_q(solution, remaining):
    """Same active continuation, changing only feedback conditioning this step."""
    if solution.mode != "active":
        raise ValueError("The fixed continuation must come from an active solution")
    if not isinstance(remaining, (int, np.integer)) or not 1 <= remaining <= solution.horizon:
        raise ValueError("remaining must identify a positive solved horizon")
    transition, rewards, _ = _build_operator(solution.theta, solution.kappa, solution.qmax,
                                             solution.beliefs, solution.quadrature_points, "noinfo")
    return _bellman_step(transition, rewards, solution.values[remaining-1])


class CacheIntegrityError(ValueError):
    """A control artifact is incomplete, stale, corrupt, or inconsistent."""


def _specification(solution):
    return {'theta': solution.theta, 'kappa': solution.kappa, 'horizon': solution.horizon,
            'qmax': solution.qmax, 'mode': solution.mode,
            'belief_points': solution.metadata.get('requested_belief_points', len(solution.beliefs)),
            'quadrature_points': solution.quadrature_points}


def _checked_mask(specification):
    if set(specification) != {'theta', 'kappa', 'horizon', 'qmax', 'mode',
                              'belief_points', 'quadrature_points'}:
        raise CacheIntegrityError('Unexpected control specification fields')
    if specification['mode'] not in MODES:
        raise CacheIntegrityError('Unknown control mode')
    for name, minimum in (('horizon', 0), ('qmax', 1), ('belief_points', 2), ('quadrature_points', 1)):
        value = specification[name]
        if type(value) is not int or value < minimum:
            raise CacheIntegrityError(f'Invalid specification {name}')
    for name, upper, closed in (('theta', 1., False), ('kappa', 1., True)):
        value = specification[name]
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0 or (value > upper if closed else value >= upper):
            raise CacheIntegrityError(f'Invalid specification {name}')
    qmax = specification['qmax']
    mask = admissible(np.arange(-qmax, qmax+1), qmax).copy()
    if specification['mode'] == 'taker':
        mask[:, 1:9] = False
    return mask


def _check_arrays(specification, arrays):
    valid = _checked_mask(specification)
    nb = 2 if specification['mode'] == 'full' else specification['belief_points']
    shape = (specification['horizon']+1, 2*specification['qmax']+1, 3, nb)
    shapes = {'beliefs': (nb,), 'values': shape, 'q_values': (*shape, 11)}
    for name, array in arrays.items():
        if array.shape != shapes[name] or array.dtype != np.dtype('float64') or not array.flags.c_contiguous:
            raise CacheIntegrityError(f'{name}: incorrect shape, dtype, or memory layout')
    if not np.array_equal(arrays['beliefs'], np.linspace(0., 1., nb)):
        raise CacheIntegrityError('Beliefs must equal the declared uniform [0,1] grid')
    if not np.all(np.isneginf(arrays['q_values'][0])):
        raise CacheIntegrityError('Terminal Q rows must contain only -infinity')
    terminal = -(H+CT)*np.abs(np.arange(-specification['qmax'], specification['qmax']+1))[:, None, None]
    if not np.array_equal(arrays['values'][0], np.broadcast_to(terminal, shape[1:])):
        raise CacheIntegrityError('Incorrect terminal liquidation values')
    mask = np.broadcast_to(valid[:, None, None, :], shapes['q_values'][1:])
    # Bounded per-horizon working memory, even for large memory-mapped tables.
    for n in range(1, specification['horizon']+1):
        q = arrays['q_values'][n]
        if not np.all(np.isfinite(q[mask])) or not np.all(np.isneginf(q[~mask])):
            raise CacheIntegrityError(f'Q legality/finite/sentinel structure failed at horizon {n}')
        selected = stable_argmax(q)
        values = np.take_along_axis(q, selected[..., None], -1)[..., 0]
        if not np.array_equal(arrays['values'][n], values):
            raise CacheIntegrityError(f'V does not equal selected-action Q at horizon {n}')
    return hashlib.sha256(np.ascontiguousarray(valid).tobytes()).hexdigest()


def save_control(path, solution, *, metadata_name='metadata.json', specification=None):
    """Publish a complete, content-bound artifact atomically; never replace one.

    Rebuilding an existing artifact requires an explicit new destination or an
    operator's deliberate removal of the old destination. No implicit repair.
    """
    path = Path(path)
    if path.exists():
        raise CacheIntegrityError(f'Artifact already exists: {path}; explicitly choose a new cache or remove this entry')
    spec = dict(specification or _specification(solution))
    if spec != _specification(solution):
        raise CacheIntegrityError('Save specification does not match solved settings')
    arrays = {name: getattr(solution, name) for name in ('beliefs', 'values', 'q_values')}
    mask_hash = _check_arrays(spec, arrays)
    record = {'format_version': CONTROL_FORMAT_VERSION, 'specification': spec,
              'key': control_key(spec), 'numerical_contract': numerical_contract(),
              'source': source_fingerprint(), 'build': build_fingerprint(),
              'admissibility_sha256': mask_hash, 'solver': solution.metadata,
              'seconds': solution.metadata['solve_seconds'], 'arrays': {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{path.name}-', dir=path.parent))
    try:
        for name, array in arrays.items():
            target = temporary/f'{name}.npy'
            np.save(target, array, allow_pickle=False)
            record['arrays'][name] = {'shape': list(array.shape), 'dtype': array.dtype.str,
                                     'bytes': target.stat().st_size, 'sha256': file_sha256(target)}
        record['artifact_sha256'] = artifact_fingerprint(record)
        (temporary/metadata_name).write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        if path.exists():
            raise CacheIntegrityError(f'Artifact appeared during construction: {path}')
        os.rename(temporary, path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return record


def load_control(path, mmap_mode='r', *, metadata_name='metadata.json', expected_specification=None):
    """Fail closed on stale provenance, byte corruption, or semantic mismatch."""
    path = Path(path)
    if mmap_mode not in ('r', None):
        raise CacheIntegrityError('Control artifacts may only be loaded read-only')
    try:
        record = json.loads((path/metadata_name).read_text())
        if record['format_version'] != CONTROL_FORMAT_VERSION:
            raise CacheIntegrityError('Unsupported/stale control-table format')
        spec = record['specification']
        _checked_mask(spec)
        if expected_specification is not None and spec != expected_specification:
            raise CacheIntegrityError('Cache settings do not match requested settings')
        if record['key'] != control_key(spec):
            raise CacheIntegrityError('Cache logical key does not match settings')
        for name, expected in (('numerical_contract', numerical_contract()),
                               ('source', source_fingerprint()), ('build', build_fingerprint())):
            if record[name] != expected:
                raise CacheIntegrityError(f'Stale or modified {name} provenance')
        if record['artifact_sha256'] != artifact_fingerprint(record):
            raise CacheIntegrityError('Artifact metadata digest mismatch')
        if set(record['arrays']) != {'beliefs', 'values', 'q_values'}:
            raise CacheIntegrityError('Missing or unexpected array declarations')
        arrays = {}
        for name, declared in record['arrays'].items():
            file = path/f'{name}.npy'
            if file.stat().st_size != declared['bytes'] or file_sha256(file) != declared['sha256']:
                raise CacheIntegrityError(f'{name}: byte digest or size mismatch')
            array = np.load(file, mmap_mode=mmap_mode, allow_pickle=False)
            if list(array.shape) != declared['shape'] or array.dtype.str != declared['dtype']:
                raise CacheIntegrityError(f'{name}: array/header metadata mismatch')
            array.setflags(write=False)
            arrays[name] = array
        if _check_arrays(spec, arrays) != record['admissibility_sha256']:
            raise CacheIntegrityError('Admissibility-mask digest mismatch')
        args = {key: value for key, value in spec.items() if key != 'belief_points'}
        return ControlSolution(**args, **arrays, metadata=record['solver'], artifact=record)
    except CacheIntegrityError:
        raise
    except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
        raise CacheIntegrityError(f'Invalid or incomplete control artifact {path}: {exc}') from exc
