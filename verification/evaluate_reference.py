"""Independent rollout validation of reduced Bellman values and runtime controls.

Fixed before the first run: 20,000 paired independent episodes, seed 260924301,
T=30,qmax=2,theta=.35,kappa=.02,B321,GH161. Failure cases include wrong remaining
horizon, wrong prior/transition, comparing a replanning noinfo rollout to its
non-learning planner value, and mistaking Monte Carlo error for grid error.
The active and full predicted values must agree with rollout within 4 MC SE.
The comparisons retain observed failures and are not used to tune a policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from trade_learning.environment import BatchedEnvironment, generate_exogenous
from trade_learning.filtering import filter_step
from trade_learning.policies import TableBank, interpolate_scores, FullInformationReference
from trade_learning.run import write_json
from trade_learning.numerics import stable_argmax, numerical_contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('outputs/verification/audit_corrections/rollout'))
    parser.add_argument('--cache', type=Path, default=Path('outputs/control_cache/reduced_reference'))
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    episodes = 20000
    bank = TableBank(args.cache, 30, 2, 321, 161)
    tapes = generate_exogenous(episodes, 30, .35, .02, 260924301)
    outcomes, planned, records = {}, {}, []
    for mode in ['active', 'noinfo', 'myopic', 'full']:
        env = BatchedEnvironment(tapes, 2)
        table = bank.table(.35, .02, 'active' if mode == 'myopic' else mode)
        beliefs = np.full((episodes, 1), .5)
        log_weights = np.zeros_like(beliefs)
        if mode == 'full':
            oracle = FullInformationReference(.35, .02, bank)
        for t in range(30):
            obs = env.observe()
            if mode == 'full':
                action = oracle.choose(obs, tapes.h[:, t])
            else:
                scores = interpolate_scores(table, 1 if mode == 'myopic' else 30-t,
                                             obs.inventory, obs.signal, beliefs[:, 0], 2)
                scores[~obs.admissible_actions] = -np.inf
                action = stable_argmax(scores, obs.admissible_actions, axis=1).astype(np.int8)
            feedback = env.step(action)
            beliefs, log_weights, _ = filter_step(beliefs, log_weights,
                        feedback.signal, feedback.return_, feedback.action, feedback.fills,
                        np.array([.35]), np.array([.02]))
        metrics = env.finalize()
        outcomes[mode] = metrics['objective']
        record = dict(policy=mode, episodes=episodes,
                      objective_mean=float(np.mean(metrics['objective'])),
                      objective_mc_se=float(np.std(metrics['objective'], ddof=1)/np.sqrt(episodes)),
                      pnl_mean=float(np.mean(metrics['pnl'])))
        if mode in {'active', 'full', 'noinfo'}:
            n = 30
            first_q = table.q_values[n, 2, 1]
            chosen = stable_argmax(first_q)
            first_values = np.take_along_axis(first_q, chosen[..., None], -1)[..., 0]
            value = float(np.mean(first_values)) if mode == 'full' else float(first_values[160])
            planned[mode] = value
            record['planner_initial_value'] = value
            if mode in {'active', 'full'}:
                error = abs(record['objective_mean']-value)
                record['value_error_in_mc_se'] = error/record['objective_mc_se']
                assert record['value_error_in_mc_se'] < 4, record
        records.append(record)
    differences = []
    for other in ['noinfo', 'myopic', 'full']:
        delta = outcomes['active']-outcomes[other]
        mean, se = float(delta.mean()), float(delta.std(ddof=1)/np.sqrt(episodes))
        critical = student_t.ppf(.975, episodes-1)
        differences.append(dict(baseline=other, mean=mean, mc_se=se,
                                low=float(mean-critical*se), high=float(mean+critical*se)))
    pd.DataFrame(outcomes).to_csv(out / 'paired_objectives.csv.gz', index=False,
                                 compression={'method':'gzip', 'mtime':0})
    fingerprints = bank.verify_loaded_tables()
    artifact = dict(passed=True, seed=260924301, episodes=episodes,
                    numerical_contract=numerical_contract(), control_artifacts=fingerprints, settings=
                    dict(horizon=30, qmax=2, theta=.35, kappa=.02, belief_points=321, quadrature_points=161),
                    policies=records, paired_differences=differences,
                    note='No-information planner value excludes future learning; its evaluated runtime policy filters and replans.')
    write_json(out / 'verification.json', artifact)
    print(json.dumps(artifact, indent=2))


if __name__ == '__main__':
    main()
