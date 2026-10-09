"""Prewritten end-to-end acceptance test, before policy/evaluation implementation.

Failure cases: incomplete grids, wrong pilot/test budgets, non-reproducible RNG,
cross-episode training, leaking evaluator metadata into starting posterior,
unmatched pilots, incorrect reward reconciliation, missing stress or policy,
nonzero abstention, absent reproducibility metadata, and incomplete summaries.
This uses fresh subprocesses and retains verifiable CSV/JSON artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='outputs/verification/pipeline')
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    # Cold caches are deliberately separate and are never shipped as evidence.
    with tempfile.TemporaryDirectory(prefix='trade-learning-cold-reproduction-') as caches:
        for name in ['first', 'repeat']:
            cache = Path(caches) / name
            assert not cache.exists()
            subprocess.run([sys.executable, '-m', 'trade_learning.run', '--profile', 'smoke',
                            '--cache', str(cache), '--out', str(root / name)], check=True)
    paths = [root / name for name in ['first', 'repeat']]
    frames = [pd.read_csv(p / 'episodes.csv.gz') for p in paths]
    sort = ['environment', 'pilot', 'variant', 'policy', 'episode']
    frames = [f.sort_values(sort).reset_index(drop=True) for f in frames]
    pd.testing.assert_frame_equal(frames[0], frames[1], check_exact=True)
    data = frames[0]
    manifest = json.loads((paths[0] / 'manifest.json').read_text())
    repeated_manifest = json.loads((paths[1] / 'manifest.json').read_text())
    assert manifest['control_artifacts'] == repeated_manifest['control_artifacts']
    assert len(manifest['control_artifacts']) == 21
    assert manifest['numerical_contract'] == repeated_manifest['numerical_contract']
    assert manifest['configuration_sha256'] == repeated_manifest['configuration_sha256']
    assert manifest['protocol_sha256'] == repeated_manifest['protocol_sha256']
    protocol = manifest['protocol']
    expected_policies = {'abstain', 'taker', 'independent', 'myopic',
                         'noinfo', 'active', 'full_information'}
    assert set(data.policy) == expected_policies
    assert set(data.variant) == {'nominal', 'state_dependence', 'fixed_duration'}
    expected_rows = (len(protocol['environments']) * protocol['pilots']
                     * protocol['test_episodes'] * 3 * len(expected_policies))
    assert len(data) == expected_rows
    counts = data.groupby(['environment', 'pilot', 'variant', 'policy']).size()
    assert (counts == protocol['test_episodes']).all()
    assert data[['pnl', 'objective', 'inventory_penalty']].notna().all().all()
    assert np.isfinite(data[['pnl', 'objective']]).all().all()
    assert data.reconciliation_error.abs().max() < 1e-8
    assert np.allclose(data.objective, data.pnl-data.inventory_penalty, atol=1e-10)
    assert (data.max_abs_inventory <= protocol['qmax']).all()
    assert (data.loc[data.policy == 'abstain', 'pnl'] == 0).all()
    assert (data.loc[data.policy == 'abstain', 'objective'] == 0).all()
    assert (data.terminal_inventory_after_liquidation == 0).all()
    for _, group in data[data.policy.isin(['myopic', 'noinfo', 'active'])].groupby(
            ['environment', 'pilot']):
        assert group.starting_posterior_hash.nunique() == 1
        assert group.pilot_data_hash.nunique() == 1
    assert manifest['status'] == 'complete'
    assert manifest['code_hash'] and manifest['specification_hash']
    assert manifest['software'] and manifest['hardware']
    assert (paths[0] / 'summary.json').is_file()
    summary = json.loads((paths[0] / 'summary.json').read_text())
    assert 'primary_comparison' in summary and 'uncertainty_method' in summary
    result = {'passed': True, 'rows': len(data), 'profile': 'smoke',
              'exact_repeat': True,
              'separate_cold_caches': True,
              'control_content_and_build_fingerprints_match': True,
              'control_tables_per_run': len(manifest['control_artifacts']),
              'source_hashes': [manifest['code_hash'], repeated_manifest['code_hash']],
              'max_reconciliation_error': float(data.reconciliation_error.abs().max()),
              'artifacts': [str(p) for p in paths],
              'scope': 'Fresh-process small end-to-end integration; not full evaluation.'}
    (root / 'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
