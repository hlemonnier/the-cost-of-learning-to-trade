"""Reconcile the prespecified two-stack reduced replay; never tune from its result.

The failure/scope contract is in numerical_failure_modes.md. CSV row order is
episode identity here: evaluate_reference.py emits the same fixed seed order.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(primary, secondary):
    folders = [Path(primary), Path(secondary)]
    records = [json.loads((p/'verification.json').read_text()) for p in folders]
    frames = [pd.read_csv(p/'paired_objectives.csv.gz') for p in folders]
    for record in records:
        require(record['passed'], 'Incomplete or failed reduced verification')
    for key in ['seed', 'episodes', 'settings', 'numerical_contract']:
        require(records[0][key] == records[1][key], f'Mismatched {key}')
    n = records[0]['episodes']
    require(n == 20000 and records[0]['seed'] == 260924301, 'Undeclared replay budget/seed')
    require(list(frames[0]) == list(frames[1]) == ['active', 'noinfo', 'myopic', 'full'], 'Policy mismatch')
    require(all(len(f) == n and np.isfinite(f.to_numpy()).all() for f in frames), 'Incomplete or nonfinite outcomes')
    registries = [{a['key']: a for a in r['control_artifacts']} for r in records]
    require(len(registries[0]) == len(registries[1]) == 3 and set(registries[0]) == set(registries[1]),
            'Missing or different table specifications')
    for key in registries[0]:
        a,b = [r[key] for r in registries]
        for field in ['specification', 'source', 'numerical_contract']:
            require(a[field] == b[field], f'Non-build difference in table {field}')
    builds = []
    for registry in registries:
        require(len({a['build']['sha256'] for a in registry.values()}) == 1, 'Mixed builds within a run')
        builds.append(next(iter(registry.values()))['build'])
    require(builds[0]['sha256'] != builds[1]['sha256'], 'Not two different numerical builds')
    differences = []
    for policy in frames[0]:
        delta = frames[1][policy]-frames[0][policy]
        differences.append(dict(policy=policy, primary_mean=float(frames[0][policy].mean()),
                                secondary_mean=float(frames[1][policy].mean()), mean_change=float(delta.mean()),
                                max_absolute_episode_difference=float(delta.abs().max()),
                                differing_episodes=int((delta != 0).sum()),
                                differences_exceeding_1e_12=int((delta.abs() > 1e-12).sum())))
    inputs = []
    for folder,build in zip(folders,builds):
        inputs.append(dict(directory=str(folder),
                           build={k:build[k] for k in ['python','numpy','scipy','system','machine','sha256']},
                           sha256={f:hashlib.sha256((folder/f).read_bytes()).hexdigest()
                                   for f in ['verification.json','paired_objectives.csv.gz']}))
    return dict(comparison_completed=True, exact_episode_equality=all(x['differing_episodes']==0 for x in differences),
                scope='Two local macOS numerical stacks; no claim of universal platform invariance',
                seed=records[0]['seed'], episodes=n, settings=records[0]['settings'],
                numerical_contract=records[0]['numerical_contract'],
                control_source_sha256=next(iter(registries[0].values()))['source']['sha256'],
                inputs=inputs, policies=differences)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--primary',type=Path,default=Path('outputs/reference/rollout'))
    parser.add_argument('--secondary',type=Path,default=Path('outputs/verification/audit_corrections/cross_stack'))
    parser.add_argument('--out',type=Path,default=Path('outputs/verification/audit_corrections/cross_stack_comparison.json'))
    args=parser.parse_args()
    result=compare(args.primary,args.secondary)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['comparison_completed','exact_episode_equality','episodes','policies']},indent=2))
