"""Frozen matched reduced-model study; no canonical source changes or warm caches.

Design and failure cases precede this implementation at protocol commit ac7b70b.
The independent command-line verifier reconstructs every ledger and contrast.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import ndtr, ndtri
from scipy.stats import t as student_t

from trade_learning.control import solve_control, _build_operator, _bellman_step
from trade_learning.environment import BatchedEnvironment, generate_exogenous
from trade_learning.filtering import filter_step
from trade_learning.model import ACTIONS, K, MU, admissible, fill_probability
from trade_learning.numerics import stable_argmax, numerical_contract, build_fingerprint, file_sha256
from trade_learning.policies import interpolate_scores
from verify_control import compare_solutions, feedback_check

ROOT = Path(__file__).resolve().parents[1]
CONFIG_SHA256 = 'd2c6f2edd919e7ff0f08b7a09ea05ec109234f6f4ef17c2eaef392aae04d942a'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def array_record(value):
    return {'sha256': array_hash(value), 'shape': list(value.shape), 'dtype': str(value.dtype)}


def load_protocol(path):
    # This is a frozen benchmark, not a generalized configuration interface.
    require(file_sha256(path) == CONFIG_SHA256,
            'Unsupported configuration edit: this study requires its frozen exact JSON')
    return json.loads(path.read_text())


def execution_revision():
    """Exported review bundles have source hashes but intentionally no .git."""
    if not (ROOT / '.git').exists():
        return None, 'source export without Git metadata; exact source hashes only'
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    return revision, 'current checkout commit plus exact source hashes'


def drift_rewards(qmax, beliefs, valid):
    result = np.zeros((2*qmax+1, 3, len(beliefs), 11))
    for qi, q in enumerate(range(-qmax, qmax+1)):
        for xi, x in enumerate((-1, 0, 1)):
            for action in np.flatnonzero(valid[qi]):
                dq = (1 if action == 9 else -1) if action >= 9 else sum(
                    side * float(fill_probability(x, side, int(depth)))
                    for side, depth in zip((1, -1), ACTIONS[action]) if depth >= 0)
                result[qi, xi, :, action] = MU*x*(q+dq)
    return result


def fixed_inventory_continuation(values, beliefs, kappa):
    target = kappa+(1-2*kappa)*beliefs
    # Do not change X, K or the continuation function; only keep its q argument.
    predicted = np.stack([np.interp(target, beliefs, row)
                          for row in values.reshape(-1, len(beliefs))]).reshape(values.shape)
    return np.einsum('xy,qyb->qxb', K, predicted, optimize=True)


def build_tables(protocol, belief_points, quadrature_points):
    args = (protocol['theta'], protocol['kappa'], protocol['horizon'], protocol['qmax'],
            belief_points, quadrature_points)
    active, noinfo = [solve_control(*args, mode=mode) for mode in ('active', 'noinfo')]
    transition, rewards, metadata = _build_operator(
        noinfo.theta, noinfo.kappa, noinfo.qmax, noinfo.beliefs, quadrature_points, 'noinfo')
    valid = admissible(np.arange(-noinfo.qmax, noinfo.qmax+1), noinfo.qmax)
    drift = drift_rewards(noinfo.qmax, noinfo.beliefs, valid)
    changed_rewards = rewards-drift
    values = np.empty_like(noinfo.values)
    scores = np.full_like(noinfo.q_values, -np.inf)
    values[0] = noinfo.values[0]
    for m in range(1, noinfo.horizon+1):
        scores[m] = _bellman_step(transition, changed_rewards, values[m-1])
        selected = stable_argmax(scores[m])
        values[m] = np.take_along_axis(scores[m], selected[..., None], -1)[..., 0]
    no_forecast = replace(noinfo, mode='no_forecast', values=values, q_values=scores,
                          metadata={**metadata, 'interpretation': 'zero predictive drift in planning only'})
    inventory_scores = np.full_like(noinfo.q_values, -np.inf)
    inventory_values = np.empty_like(noinfo.values)
    inventory_values[0] = noinfo.values[0]
    inventory_scores[1] = noinfo.q_values[1]
    inventory_values[1] = noinfo.values[1]
    for m in range(2, noinfo.horizon+1):
        common = fixed_inventory_continuation(noinfo.values[m-1], noinfo.beliefs, noinfo.kappa)
        inventory_scores[m] = rewards+common[..., None]
        selected = stable_argmax(inventory_scores[m])
        inventory_values[m] = np.take_along_axis(inventory_scores[m], selected[..., None], -1)[..., 0]
    inventory_off = replace(noinfo, mode='inventory_off', values=inventory_values,
                            q_values=inventory_scores,
                            metadata={'interpretation': 'maximum ablated decision score, not policy value'})
    return {'active': active, 'noinfo': noinfo, 'no_forecast': no_forecast,
            'inventory_off': inventory_off}


def refine(protocol, out):
    b, g = protocol['belief_points'], protocol['quadrature_points']
    records = []
    while True:
        tables = build_tables(protocol, b, g)
        axis_pass = {}
        for axis, settings in (('belief', ((b+1)//2, g)), ('quadrature', (b, (g+1)//2))):
            coarse = build_tables(protocol, *settings)
            comparisons = {}
            for mode in tables:
                comparisons[mode] = compare_solutions(coarse[mode], tables[mode],
                    protocol['refinement']['initial'], protocol['refinement']['sup'],
                    protocol['refinement']['robust_gap'], protocol['refinement']['robust_disagreement'])
            passed = all(item['passes_predeclared_rule'] for item in comparisons.values())
            records.append({'axis': axis, 'coarse': list(settings), 'fine': [b, g],
                            'comparisons': comparisons, 'passed': passed})
            axis_pass[axis] = passed
            del coarse
            write(out / 'refinement_progress.json', records)
            print(json.dumps({'stage': 'refinement', 'axis': axis, 'settings': [b, g], 'passed': passed}), flush=True)
        if all(axis_pass.values()):
            break
        del tables
        b = 2*b-1 if not axis_pass['belief'] else b
        g = 2*g-1 if not axis_pass['quadrature'] else g
        require(b <= 5121 and g <= 1281, 'Numerical refinement remains unresolved; retain failures')
    tables['full_information'] = solve_control(protocol['theta'], protocol['kappa'],
                    protocol['horizon'], protocol['qmax'], b, g, 'full')
    return tables, records, [b, g]


def independent_outcome_probabilities(x, action, theta):
    """Scalar Gaussian integration, independent of production kernel assembly."""
    if action >= 9:
        return [(1 if action == 9 else -1, 1.)]
    bid, ask = [(action // 3)-1, (action % 3)-1]
    sides = [(s, k) for s, k in ((1, bid), (-1, ask)) if k >= 0]
    results = []
    for bits in range(1 << len(sides)):
        dq = sum(s*((bits >> j) & 1) for j, (s, _) in enumerate(sides))
        def integrand(z):
            p = math.exp(-z*z/2)/math.sqrt(2*math.pi)
            for j, (side, depth) in enumerate(sides):
                threshold = ndtri(1/(1+math.exp(.3+.7*depth+.2*side*x)))
                direction = 1 if (bits >> j) & 1 else -1
                p *= ndtr(direction*(threshold-theta*side*z)/math.sqrt(1-theta**2))
            return p
        probability = quad(integrand, -11, 11, epsabs=1e-12, epsrel=1e-12)[0]
        results.append((dq, probability))
    return results


def structural_checks(tables):
    noinfo, off, forecast = (tables[k] for k in ('noinfo', 'inventory_off', 'no_forecast'))
    qmax, beliefs = noinfo.qmax, noinfo.beliefs
    valid = admissible(np.arange(-qmax, qmax+1), qmax)
    mask = np.broadcast_to(valid[:, None, None, :], noinfo.q_values.shape[1:])
    for name, table in tables.items():
        own_mask = np.broadcast_to(valid[:, None, None, :], table.q_values.shape[1:])
        require(table.q_values.dtype == np.float64, 'Non-float64 table')
        require(np.array_equal(table.beliefs, np.linspace(0, 1, len(table.beliefs))), 'Malformed grid')
        require(np.isneginf(table.q_values[0]).all(), 'Nonempty terminal Q')
        terminal = -.027*np.abs(np.arange(-qmax, qmax+1))[:, None, None]
        # Decimal .027 and the production sum .025+.002 differ by one rounding
        # step; this independent decimal check is mathematical, not byte identity.
        require(np.max(np.abs(table.values[0]-terminal)) < 1e-15, 'Terminal value mismatch')
        for m in range(1, table.horizon+1):
            require(np.isfinite(table.q_values[m][own_mask]).all() and
                    np.isneginf(table.q_values[m][~own_mask]).all(), 'Illegal table entries')
    require(np.array_equal(off.q_values[1], noinfo.q_values[1]), 'Inventory ablation altered final decision')
    require(np.array_equal(forecast.values[0], noinfo.values[0]), 'Forecast ablation altered liquidation')
    transition, rewards, _ = _build_operator(noinfo.theta, noinfo.kappa, qmax, beliefs,
                                             noinfo.quadrature_points, 'noinfo')
    drift = drift_rewards(qmax, beliefs, valid)
    drift_error = 0.
    for qi, q in enumerate(range(-qmax, qmax+1)):
        for xi, x in enumerate((-1, 0, 1)):
            for a in np.flatnonzero(valid[qi]):
                if a >= 9:
                    mean_q = q+(1 if a == 9 else -1)
                else:
                    mean_q = float(q)
                    for side, depth in ((1, a//3-1), (-1, a%3-1)):
                        if depth >= 0:
                            mean_q += side/(1+math.exp(.3+.7*depth+.2*side*x))
                expected = .03*x*mean_q
                difference = noinfo.q_values[1, qi, xi, :, a]-forecast.q_values[1, qi, xi, :, a]
                drift_error = max(drift_error, float(np.max(np.abs(difference-expected))))
    require(drift_error < 2e-12, 'Predictive drift deletion differs from exact cash-flow formula')
    max_inventory_error = 0.
    cases = 0
    probabilities = {(x, a): independent_outcome_probabilities(x, a, noinfo.theta)
                     for x in (-1, 0, 1) for a in range(11)}
    for m in (2, 17, 30):
        for qi, q in enumerate(range(-qmax, qmax+1)):
            for xi, x in enumerate((-1, 0, 1)):
                for b in (0., .125, .5, .75, 1.):
                    bi = int(round(b*(len(beliefs)-1)))
                    predicted = noinfo.kappa+(1-2*noinfo.kappa)*beliefs[bi]
                    def continuation(index):
                        return sum(K[xi, yi]*np.interp(predicted, beliefs, noinfo.values[m-1, index, yi])
                                   for yi in range(3))
                    base = continuation(qi)
                    for a in np.flatnonzero(valid[qi]):
                        deletion = sum(prob*(continuation(qi+dq)-base)
                                       for dq, prob in probabilities[(x, a)])
                        actual = noinfo.q_values[m, qi, xi, bi, a]-off.q_values[m, qi, xi, bi, a]
                        max_inventory_error = max(max_inventory_error, abs(float(actual-deletion)))
                        cases += 1
    require(max_inventory_error < 2e-11, 'Inventory score deletion fails independent outcome enumeration')
    # A synthetic continuation checks the removed operator independently of a solved V.
    q = np.arange(-qmax, qmax+1)[:, None, None]
    x = np.arange(-1, 2)[None, :, None]
    synthetic = .1*q*q+.03*q*x+.02*q*beliefs[None, None, :]+.05*x*beliefs[None, None, :]
    actual = np.zeros_like(rewards)
    computed = _bellman_step(transition, rewards, synthetic)
    actual[mask] = computed[mask]-rewards[mask]
    fixed = fixed_inventory_continuation(synthetic, beliefs, noinfo.kappa)
    synthetic_error = 0.
    for qi in range(2*qmax+1):
        for xi, xv in enumerate((-1, 0, 1)):
            for a in np.flatnonzero(valid[qi]):
                expected = sum(prob*fixed[qi+dq, xi] for dq, prob in probabilities[(xv, a)])
                synthetic_error = max(synthetic_error, float(np.max(np.abs(actual[qi, xi, :, a]-expected))))
    require(synthetic_error < 2e-11, 'Synthetic continuation kernel mismatch')
    independent = np.broadcast_to(.05*x+.02*beliefs[None, None, :], synthetic.shape)
    true_scores = _bellman_step(transition, rewards, independent)
    suppressed = rewards+fixed_inventory_continuation(independent, beliefs, noinfo.kappa)[..., None]
    zero_error = float(np.max(np.abs(true_scores[mask]-suppressed[mask])))
    require(zero_error < 2e-11, 'Inventory-independent continuation changed action scores')
    # Inventory-off maximizes immediate reward until the true last decision.
    immediate = stable_argmax(rewards)
    for m in range(2, noinfo.horizon+1):
        require(np.array_equal(stable_argmax(off.q_values[m]), immediate), 'Inventory-off selection drifted')
    return {'passed': True, 'independent_inventory_cases': cases,
            'max_inventory_deletion_error': max_inventory_error,
            'max_one_step_drift_error': drift_error,
            'synthetic_continuation_error': synthetic_error,
            'inventory_independent_continuation_error': zero_error,
            'exact_final_step_inventory_agreement': True,
            'inventory_off_before_final_step_equals_immediate_reward_rule': True}


def same_state_comparisons(tables):
    base = tables['noinfo']
    records = []
    pairs = [('forecast', base, tables['no_forecast']),
             ('inventory_continuation', base, tables['inventory_off']),
             ('future_information', tables['active'], base),
             ('current_inference', base, base)]
    for channel, enabled, disabled in pairs:
        count = meaningful = 0
        witness = None
        maximum_loss = -1.
        for m in range(1, base.horizon+1):
            scores = enabled.q_values[m]
            other = disabled.q_values[m]
            if channel == 'current_inference':
                other = np.broadcast_to(other[:, :, len(base.beliefs)//2:len(base.beliefs)//2+1], other.shape)
            on, off = stable_argmax(scores), stable_argmax(other)
            loss = np.max(scores, axis=-1)-np.take_along_axis(scores, off[..., None], -1)[..., 0]
            changed = on != off
            count += int(changed.sum())
            meaningful += int((changed & (loss > .001)).sum())
            loc = np.unravel_index(np.argmax(loss), loss.shape)
            if float(loss[loc]) > maximum_loss:
                maximum_loss = float(loss[loc])
                qi, xi, bi = loc
                witness = {'remaining': m, 'inventory': int(qi)-base.qmax, 'signal': int(xi)-1,
                           'belief': float(base.beliefs[bi]), 'enabled_action': int(on[loc]),
                           'disabled_action': int(off[loc]), 'loss_in_enabled_decision_score': maximum_loss}
        records.append({'mechanism': channel, 'states': base.horizon*base.values[0].size,
                        'different_actions': count, 'different_actions_with_score_loss_above_001': meaningful,
                        'maximum_enabled_score_loss': maximum_loss, 'illustration': witness,
                        'scope': 'All grid states/horizons; selected illustration is not a statistical test or true-policy value gap'})
    return records


def evaluate(protocol, tables, out):
    n, horizon, qmax = protocol['episodes'], protocol['horizon'], protocol['qmax']
    tapes = generate_exogenous(n, horizon, protocol['theta'], protocol['kappa'], protocol['seed'])
    theta, kappa = np.array([protocol['theta']]), np.array([protocol['kappa']])
    outcomes, action_arrays, frames, traces, policy_records = {}, {}, [], [], []
    for policy in protocol['policies']:
        table = tables['noinfo' if policy in ('blind_regime', 'myopic') else policy]
        env = BatchedEnvironment(tapes, qmax)
        belief, log_weights = np.full((n, 1), .5), np.zeros((n, 1))
        actions = np.empty((n, horizon), dtype=np.int8)
        for t in range(horizon):
            obs = env.observe()
            decision_belief = belief[:, 0]
            if policy == 'blind_regime':
                decision_belief = np.full(n, .5)
            elif policy == 'full_information':
                decision_belief = (tapes.h[:, t] == 1).astype(float)
            scores = interpolate_scores(table, 1 if policy == 'myopic' else horizon-t,
                                         obs.inventory, obs.signal, decision_belief, qmax)
            actions[:, t] = stable_argmax(scores, obs.admissible_actions, axis=1).astype(np.int8)
            feedback = env.step(actions[:, t])
            for i in range(protocol['trace_episodes']):
                traces.append({'policy': policy, 'episode': i, 't': t,
                    'inventory': int(obs.inventory[i]), 'signal': int(obs.signal[i]),
                    'price': float(obs.price[i]), 'cash': float(obs.cash[i]),
                    'belief': float(belief[i, 0]), 'decision_belief': float(decision_belief[i]),
                    'action': int(actions[i, t]), 'return_': float(feedback.return_[i]),
                    'bid_fill': int(feedback.fills[i, 0]), 'ask_fill': int(feedback.fills[i, 1]),
                    'next_inventory': int(feedback.next_observation.inventory[i]),
                    'next_cash': float(feedback.next_observation.cash[i])})
            # Shadow filtering in the blind/reference cases cannot change their routed belief.
            belief, log_weights, _ = filter_step(belief, log_weights, feedback.signal,
                    feedback.return_, feedback.action, feedback.fills, theta, kappa)
        metrics = env.finalize()
        require(float(np.max(np.abs(metrics['reconciliation_error']))) < 1e-8, 'Cash ledger mismatch')
        require(np.all(metrics['terminal_inventory_after_liquidation'] == 0), 'Unliquidated terminal inventory')
        scalar = {key: value for key, value in metrics.items() if value.ndim == 1}
        scalar['terminal_inventory_before_liquidation'] = metrics['terminal_inventory']
        scalar['terminal_liquidation_cost'] = metrics['liquidation_cost']
        frame = pd.DataFrame(scalar)
        frame.insert(0, 'episode', np.arange(n))
        frame.insert(0, 'policy', policy)
        require(np.isfinite(frame.drop(columns='policy').to_numpy()).all(), 'Nonfinite episode outcome')
        frames.append(frame)
        outcomes[policy] = metrics['objective']
        action_arrays[policy] = actions
        record = {'policy': policy, 'mean': float(metrics['objective'].mean()),
                  'se': float(metrics['objective'].std(ddof=1)/np.sqrt(n)),
                  'pnl_mean': float(metrics['pnl'].mean()),
                  'inventory_sq_mean_per_period': float(metrics['inventory_sq_sum'].mean()/horizon),
                  'passive_fees_mean': float(metrics['passive_fees'].mean()),
                  'market_fees_mean': float(metrics['market_fees'].mean()),
                  'liquidation_cost_mean': float(metrics['liquidation_cost'].mean()),
                  'max_reconciliation_error': float(np.max(np.abs(metrics['reconciliation_error']))),
                  'action_frequencies': (np.bincount(actions.ravel(), minlength=11)/(n*horizon)).tolist()}
        if policy in ('active', 'full_information', 'blind_regime'):
            planning = tables['noinfo' if policy == 'blind_regime' else policy]
            values = planning.values[horizon, qmax, 1]
            value = float(values.mean()) if policy == 'full_information' else float(np.interp(.5, planning.beliefs, values))
            record['planner_value'] = value
            record['planning_error_in_mc_se'] = abs(record['mean']-value)/record['se']
            require(record['planning_error_in_mc_se'] < 4, 'Matched planning/rollout calibration failed')
        policy_records.append(record)
        print(json.dumps({'stage': 'rollout', **record}), flush=True)
    pd.concat(frames, ignore_index=True).to_csv(out / 'episodes.csv.gz', index=False,
                         float_format='%.17g', compression={'method': 'gzip', 'mtime': 0})
    pd.DataFrame(traces).to_csv(out / 'traces.csv.gz', index=False,
                         float_format='%.17g', compression={'method': 'gzip', 'mtime': 0})
    np.savez_compressed(out / 'actions.npz', **action_arrays)
    contrasts = []
    for comparison in protocol['contrasts']:
        delta = outcomes[comparison['policy']]-outcomes[comparison['baseline']]
        mean, se = float(delta.mean()), float(delta.std(ddof=1)/np.sqrt(n))
        ordinary = float(student_t.ppf(1-protocol['alpha']/2, n-1))*se
        simultaneous = float(student_t.ppf(1-protocol['alpha']/(2*protocol['family_size']), n-1))*se
        contrasts.append({**comparison, 'mean': mean, 'se': se, 'low': mean-ordinary,
                          'high': mean+ordinary, 'simultaneous_low': mean-simultaneous,
                          'simultaneous_high': mean+simultaneous, 'df': n-1, 'n': n})
    write(out / 'summary.json', {'policies': policy_records, 'contrasts': contrasts,
        'episodes': n, 'rows': n*len(protocol['policies']), 'family_size': protocol['family_size'],
        'uncertainty': 'Paired-episode Student intervals; simultaneous Bonferroni family of four; approximate Monte Carlo coverage',
        'scope': 'New fixed reduced-model policy interventions; nonadditive; canonical experiment unchanged'})
    return {name: array_hash(getattr(tapes, name)) for name in ('x', 'h', 'z', 'u_bid', 'u_ask')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/mechanism_study.json')
    args = parser.parse_args()
    started = time.perf_counter()
    protocol = load_protocol(args.config)
    require(not args.out.exists() or not any(args.out.iterdir()), 'Output directory must be new or empty')
    args.out.mkdir(parents=True, exist_ok=True)
    canonical = json.loads((ROOT / 'outputs/full/manifest.json').read_text())
    source = sorted((ROOT / 'src/trade_learning').glob('*.py'))
    source_hash = hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in source)).hexdigest()
    require(source_hash == canonical['code_hash'], 'Canonical production source changed')
    inputs = source + [Path(__file__), ROOT / 'verification/verify_control.py', args.config,
                      ROOT / 'report/mechanism_study_protocol.md', ROOT / 'verification/mechanism_failure_modes.md']
    input_sha = {str(p.resolve().relative_to(ROOT)): file_sha256(p) for p in inputs}
    write(args.out / 'manifest.json', {'status': 'running', 'protocol': protocol, 'input_sha256': input_sha})
    tables, refinements, settings = refine(protocol, args.out)
    structural = structural_checks(tables)
    checks = {'passed': True, 'structural': structural, 'refinement': refinements,
              'same_state_comparisons': same_state_comparisons(tables),
              'fixed_active_continuation_feedback_witness': feedback_check(tables['active'])}
    write(args.out / 'numerical_checks.json', checks)
    tapes = evaluate(protocol, tables, args.out)
    require(input_sha == {path: file_sha256(ROOT / path) for path in input_sha}, 'Inputs changed during study')
    table_hashes = {mode: {name: array_record(getattr(table, name))
                         for name in ('beliefs', 'values', 'q_values')} for mode, table in tables.items()}
    filenames = ('episodes.csv.gz', 'actions.npz', 'traces.csv.gz', 'numerical_checks.json', 'summary.json')
    revision, git_scope = execution_revision()
    manifest = {'status': 'complete', 'protocol': protocol, 'input_sha256': input_sha,
        'protocol_freeze_commit': 'ac7b70b',
        'execution_commit': revision, 'execution_git_scope': git_scope,
        'canonical_source_sha256': source_hash, 'canonical_raw_sha256': canonical['csv_sha256'],
        'numerical_contract': numerical_contract(), 'build': build_fingerprint(),
        'cold_control_build': True, 'shared_control_cache': False, 'selected_resolution': settings,
        'table_fingerprints': table_hashes, 'tape_sha256': tapes,
        'files': {name: file_sha256(args.out / name) for name in filenames},
        'seconds': time.perf_counter()-started,
        'scope': 'Supplemental implementation and evaluation; final independent verifier and PDF build are separate'}
    write(args.out / 'manifest.json', manifest)
    (args.out / 'refinement_progress.json').unlink()
    print(json.dumps({'status': 'complete', 'seconds': manifest['seconds'], 'out': str(args.out)}), flush=True)


if __name__ == '__main__':
    main()
