"""Documented entry point for synthetic pilot fitting and independent evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.special import ndtri

from .environment import BatchedEnvironment, generate_exogenous, generate_uniform_pilot
from .filtering import fit_grid_posterior
from .model import ACTIONS, fill_probability, MU, SIGMA
from .policies import (CANDIDATE_THETA, CANDIDATE_KAPPA, POLICIES, TableBank,
                       GridPolicy, FixedPolicy, FullInformationReference)
from .statistics import summarize
from .protocol import (CONFIGURATION_CONTRACT, load_configuration, resolve_protocol,
                       validate_protocol, validate_inference_support, protocol_fingerprint)


ROOT = Path(__file__).resolve().parents[2]


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), indent=2, allow_nan=False)+'\n')


def source_hash():
    files = sorted((ROOT / 'src' / 'trade_learning').glob('*.py'))
    return hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in files)).hexdigest()


def seed_for(master, namespace, environment, pilot):
    # Component IDs precede the fixed root, following NumPy's SeedSequence guidance.
    return int(np.random.SeedSequence([namespace, environment, pilot, master]).generate_state(
        1, dtype=np.uint64)[0])


def pilot_hash(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        h.update(key.encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def make_protocol(profile, belief_points=None, quadrature_points=None):
    raw = load_configuration(ROOT / 'configs' / 'protocol.json')
    result = resolve_protocol(raw, profile, belief_points, quadrature_points)
    validate_inference_support(result, CANDIDATE_THETA, CANDIDATE_KAPPA, POLICIES)
    return result


def evaluate_batch(policy_name, fit, tapes, bank, theta, kappa):
    count, horizon = tapes.z.shape
    env = BatchedEnvironment(tapes, qmax=bank.qmax)
    if policy_name in {'active', 'noinfo', 'myopic'}:
        policy = GridPolicy(policy_name, fit, count, bank)
    elif policy_name == 'full_information':
        policy = FullInformationReference(theta, kappa, bank)
    else:
        policy = FixedPolicy(policy_name, count, bank)
    quote_gap = np.zeros(count, dtype=int)
    max_quote_gap = np.zeros(count, dtype=int)
    brier = np.zeros(count)
    wrong_confident = np.zeros(count)
    switch_error = np.zeros(count)
    switch_count = np.zeros(count)
    switch_age = np.full(count, horizon+1, dtype=int)
    logscore_sum = np.zeros(count)
    submitted_periods = np.zeros(count)
    submitted_sides = np.zeros(count)
    score_sum = np.zeros((count, 3))
    side_counts_by_signal = np.zeros((count, 3))
    sign = np.array([1, -1])
    learning = isinstance(policy, GridPolicy)
    for t in range(horizon):
        observation = env.observe()
        if learning:
            prior_belief = policy.predict_regime()
            prediction_rho = policy.predict_rho()
            # Labels used only for retrospective evaluator diagnostics.
            actual = (tapes.h[:, t] > 0).astype(float)
            brier += (prior_belief-actual)**2
            wrong_confident += (((prior_belief > .9) & (actual == 0))
                                | ((prior_belief < .1) & (actual == 1)))
            if t > 0:
                switch_age = np.where(tapes.h[:, t] != tapes.h[:, t-1], 0, switch_age+1)
            recent = switch_age < 10
            switch_error += recent*((prior_belief >= .5) != (actual == 1))
            switch_count += recent
        else:
            prediction_rho = np.zeros(count)
        if policy_name == 'full_information':
            actions = policy.choose(observation, tapes.h[:, t])
        else:
            actions = policy.choose(observation)
        feedback = env.step(actions)
        policy.update(feedback)
        quote = feedback.submitted.any(axis=1)
        quote_gap = np.where(quote, 0, quote_gap+1)
        max_quote_gap = np.maximum(max_quote_gap, quote_gap)
        sides = feedback.submitted.sum(axis=1)
        submitted_periods += quote
        submitted_sides += sides
        if learning:
            logscore_sum += policy.predictive_loglik
        # Public martingale diagnostic: execution/return association, conditional
        # on past evidence. Baseline models have rho=0; no hidden label enters it.
        z = (feedback.return_-MU*feedback.signal)/SIGMA
        depths = np.maximum(ACTIONS[actions], 0)
        probabilities = fill_probability(feedback.signal[:, None], sign[None, :], depths)
        threshold = ndtri(probabilities)
        phi = np.exp(-.5*threshold**2)/np.sqrt(2*np.pi)
        selected = np.where(feedback.submitted, feedback.fills, 0)
        residual = (selected*sign[None, :]).sum(axis=1)*z
        residual += prediction_rho*(phi*feedback.submitted).sum(axis=1)
        for xidx in range(3):
            cell = feedback.signal == xidx-1
            score_sum[:, xidx] += cell*residual
            side_counts_by_signal[:, xidx] += cell*sides
    metrics = env.finalize()
    result = {k: np.asarray(v) for k, v in metrics.items() if np.asarray(v).ndim == 1}
    result['mean_abs_inventory'] = metrics['inventory_abs_sum']/horizon
    result['bid_fill_count'] = metrics['fill_counts'][:, 0]
    result['ask_fill_count'] = metrics['fill_counts'][:, 1]
    if policy_name == 'full_information':
        # This diagnostic was defined for public learner predictions. Do not
        # mislabel rho=0 residuals as a calibration check of a privileged policy.
        score_sum[:] = np.nan
    counts = metrics['action_counts']
    for a in range(11):
        result[f'action_{a}'] = counts[:, a]
    result.update(max_quote_gap=max_quote_gap,
                  submitted_periods=submitted_periods, submitted_sides=submitted_sides,
                  conditional_logscore=np.divide(logscore_sum, submitted_periods,
                      out=np.full(count, np.nan), where=(submitted_periods > 0) & learning),
                  regime_brier=brier/horizon if learning else np.full(count, np.nan),
                  wrong_confident_fraction=wrong_confident/horizon if learning else np.full(count, np.nan),
                  post_switch_error_count=switch_error if learning else np.full(count, np.nan),
                  post_switch_window_periods=switch_count if learning else np.full(count, np.nan),
                  post_switch_error_10=np.divide(switch_error, switch_count,
                      out=np.full(count, np.nan), where=(switch_count > 0) & learning),
                  score_residual_per_side=np.divide(score_sum.sum(axis=1), submitted_sides,
                      out=np.full(count, np.nan), where=submitted_sides > 0))
    for xidx in range(3):
        result[f'score_sum_x{xidx-1}'] = score_sum[:, xidx]
        result[f'submitted_sides_x{xidx-1}'] = side_counts_by_signal[:, xidx]
    if learning:
        weights = np.exp(policy.log_weights)
        result['final_parameter_entropy'] = -np.sum(weights*policy.log_weights, axis=1)
    else:
        result['final_parameter_entropy'] = np.full(count, np.nan)
    result['starting_posterior_hash'] = [policy.starting_posterior_hash]*count
    return pd.DataFrame(result)


def execute(protocol, out, cache):
    # Validate direct callers before filesystem writes, table construction, or
    # simulator sampling. A valid CLI path alone is not an execution contract.
    protocol = validate_protocol(protocol)
    validate_inference_support(protocol, CANDIDATE_THETA, CANDIDATE_KAPPA, POLICIES)
    configuration_path = ROOT / 'configs' / 'protocol.json'
    load_configuration(configuration_path)
    configuration_hash = hashlib.sha256(configuration_path.read_bytes()).hexdigest()
    resolved_hash = protocol_fingerprint(protocol)
    from .numerics import numerical_contract

    started = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'pilots').mkdir(exist_ok=True)
    (out / 'batches').mkdir(exist_ok=True)
    code_hash = source_hash()
    specification_hash = hashlib.sha256((ROOT / 'BENCHMARK.md').read_bytes()).hexdigest()
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        revision = 'uncommitted'
    manifest = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(),
                    protocol=protocol, code_hash=code_hash, git_revision=revision,
                    configuration_contract=CONFIGURATION_CONTRACT,
                    configuration_sha256=configuration_hash, protocol_sha256=resolved_hash,
                    numerical_contract=numerical_contract(),
                    specification_hash=specification_hash,
                    hardware=dict(platform=platform.platform(), machine=platform.machine(),
                                  processor=platform.processor(), cpu_count=os.cpu_count()),
                    software={p: importlib.metadata.version(p) for p in
                              ['numpy', 'scipy', 'pandas', 'matplotlib']},
                    python=platform.python_version(), streams=[],
                    learner_boundary='Public pilot and public observation/feedback only; labels are evaluator-only.',
                    reference_boundary='Current hidden H and nominal parameters only, not future innovations.',
                    final_episode_reset='Fresh pilot posterior and H prior=0.5 in every episode.',
                    timing_note='Computation is not additional environment observations.')
    write_json(out / 'manifest.json', manifest)
    bank = TableBank(cache, protocol['horizon'], protocol['qmax'],
                     protocol['belief_points'], protocol['quadrature_points'],
                     protocol.get('belief_overrides'))
    # Materialize all public-support tables before revealing final evaluation results.
    for mode in ['active', 'noinfo']:
        for th, ka in zip(CANDIDATE_THETA, CANDIDATE_KAPPA):
            table = bank.table(th, ka, mode)
            print(f'table {mode} theta={th:g} kappa={ka:g} ready '
                  f'({table.metadata["seconds"]:.2f}s original computation)', flush=True)
    bank.table(0., .02, 'independent')
    bank.table(0., .02, 'taker')
    for th, ka in protocol['environments']:
        bank.table(th, ka, 'full')
    initial_control_artifacts = bank.fingerprints()
    manifest['control_artifacts'] = initial_control_artifacts
    write_json(out / 'manifest.json', manifest)
    all_frames, fit_records = [], []
    for env_index, (theta, kappa) in enumerate(protocol['environments']):
        env_name = f'theta{theta:g}_kappa{kappa:g}'
        for replicate in range(protocol['pilots']):
            pilot_seed = seed_for(protocol['master_seed'], 11, env_index, replicate)
            behavior_seed = seed_for(protocol['master_seed'], 12, env_index, replicate)
            pilot = generate_uniform_pilot(episodes=protocol['pilot_episodes'],
                       horizon=protocol['horizon'], theta=theta, kappa=kappa,
                       seed=pilot_seed, policy_seed=behavior_seed, qmax=protocol['qmax'])
            public_arrays = pilot.as_dict()
            fingerprint = pilot_hash(public_arrays)
            np.savez_compressed(out / 'pilots' / f'{env_index:02d}_{replicate:02d}.npz', **public_arrays)
            fit = fit_grid_posterior(pilot, CANDIDATE_THETA, CANDIDATE_KAPPA)
            fit_records.append(dict(environment=env_name, pilot=replicate,
                            pilot_data_hash=fingerprint,
                            log_weights=fit.log_weights.tolist(), weights=fit.weights.tolist(),
                            candidate_theta=CANDIDATE_THETA.tolist(), candidate_kappa=CANDIDATE_KAPPA.tolist(),
                            pilot_episodes=protocol['pilot_episodes'], horizon=protocol['horizon']))
            stream_record = dict(environment=env_name, pilot=replicate,
                                 pilot_seed=pilot_seed, behavior_seed=behavior_seed,
                                 pilot_data_hash=fingerprint, final={})
            for variant_index, variant in enumerate(protocol['variants']):
                final_seed = seed_for(protocol['master_seed'], 21+variant_index, env_index, replicate)
                stream_record['final'][variant] = final_seed
                tapes = generate_exogenous(protocol['test_episodes'], protocol['horizon'],
                            theta, kappa, seed=final_seed, variant=variant)
                for policy_name in protocol['policies']:
                    frame = evaluate_batch(policy_name, fit, tapes, bank, theta, kappa)
                    frame['environment'], frame['theta'], frame['kappa'] = env_name, theta, kappa
                    frame['pilot'], frame['variant'], frame['policy'] = replicate, variant, policy_name
                    frame['episode'] = np.arange(protocol['test_episodes'])
                    frame['pilot_data_hash'] = fingerprint
                    all_frames.append(frame)
            manifest['streams'].append(stream_record)
            partial = pd.concat(all_frames[-len(protocol['policies'])*len(protocol['variants']):], ignore_index=True)
            partial.to_csv(out / 'batches' / f'{env_index:02d}_{replicate:02d}.csv.gz', index=False,
                           compression={'method': 'gzip', 'mtime': 0})
            print(f'completed {env_name} pilot {replicate+1}/{protocol["pilots"]}; '
                  f'{time.perf_counter()-started:.1f}s elapsed', flush=True)
            write_json(out / 'manifest.json', manifest)
    data = pd.concat(all_frames, ignore_index=True)
    data.to_csv(out / 'episodes.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    write_json(out / 'pilot_fits.json', fit_records)
    write_json(out / 'summary.json', summarize(data, protocol))
    if source_hash() != code_hash:
        raise RuntimeError('Production source changed during evaluation; results must be rerun after freezing.')
    if hashlib.sha256(configuration_path.read_bytes()).hexdigest() != configuration_hash:
        raise RuntimeError('Configuration file changed during evaluation; results cannot be marked complete.')
    if protocol_fingerprint(protocol) != resolved_hash:
        raise RuntimeError('Resolved protocol changed during evaluation; results cannot be marked complete.')
    final_control_artifacts = bank.verify_loaded_tables()
    if final_control_artifacts != initial_control_artifacts:
        raise RuntimeError('Evaluated control artifact fingerprints changed during evaluation.')
    manifest.update(status='complete', rows=len(data), wall_seconds=time.perf_counter()-started,
                    control_computation=bank.computation,
                    control_artifacts=final_control_artifacts,
                    csv_sha256=hashlib.sha256((out / 'episodes.csv.gz').read_bytes()).hexdigest(),
                    completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({'status': 'complete', 'episodes': len(data),
                       'seconds': manifest['wall_seconds'], 'output': str(out)}), flush=True)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['smoke', 'full'], default='smoke')
    parser.add_argument('--out', default='outputs/smoke')
    parser.add_argument('--cache', default='outputs/control_cache')
    parser.add_argument('--belief-points', type=int)
    parser.add_argument('--quadrature-points', type=int)
    args = parser.parse_args()
    execute(make_protocol(args.profile, args.belief_points, args.quadrature_points), args.out, args.cache)


if __name__ == '__main__':
    main()
