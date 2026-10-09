#!/bin/zsh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHONPATH=extensions/bayes_adaptive .venv/bin/python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
import copy
import json
import numpy as np

from explain import (analytic_rewards, error_envelopes, information_scores,
                     _illustrative_posteriors as producer_illustrative)
from solver import load_family
from verify_explanation import (_arbitrary_states, _backup, _envelopes,
                                _family, _illustrative_posteriors, _information,
                                _legal, _reward)
from verify_extension import VerificationError, frozen_config, sha
from verify_numerical_raw import _source_identity

ext = Path('extensions/bayes_adaptive')
out = ext / 'outputs/explanation-independent-bounded'
family_path = ext / 'cache/geometry-e2e-2/endpoint-first'
probe_path = ext / 'outputs/numerical-endpoint/numerical_probes.npz'
candidate_path = ext / 'outputs/explanation-candidates-endpoint/candidate_states.npz'
config = frozen_config(ext)
family = _family(family_path, _source_identity(Path('.')))
producer = load_family(family_path)
with np.load(probe_path, allow_pickle=False) as archive:
    probes = {name: archive[name] for name in archive.files}
with np.load(candidate_path, allow_pickle=False) as archive:
    states = {name: archive[name] for name in archive.files}
probe_count = _arbitrary_states(states, config, probe_path)
maximum = {'reward': 0., 'direct_q': 0., 'information': 0., 'terminal_suppression': 0.}
for index, remaining in enumerate((1, 2, 5, 7)):
    q = int(probes['inventory'][index])
    x = int(probes['signal'][index])
    joint = np.asarray(probes['joint'][index])
    legal = _legal(q, config['qmax'])
    reward = _reward(q, x, joint, legal, config['theta'])
    published_reward = analytic_rewards(np.array([q]), np.array([x]), joint[None, :])[0]
    maximum['reward'] = max(maximum['reward'], float(np.max(np.abs(
        reward[legal] - published_reward[legal]))))
    info = _information(x, joint, config['theta'])
    published_info = information_scores(np.array([x]), joint[None, :])
    for name, values in info.items():
        maximum['information'] = max(maximum['information'], float(np.max(np.abs(
            values - published_info[name][0]))))
    independent = {}
    for mode in ('bayes', 'frozen_model', 'no_feedback'):
        independent[mode] = _backup(family, remaining, q, x, joint, reward, mode)
        published = producer.bellman_q('bayes', remaining, np.array([q]),
                                       np.array([x]), joint[None, :],
                                       update_mode=mode)[0]
        maximum['direct_q'] = max(maximum['direct_q'], float(np.max(np.abs(
            independent[mode][legal] - published[legal]))))
    if remaining == 1:
        for mode in ('frozen_model', 'no_feedback'):
            maximum['terminal_suppression'] = max(
                maximum['terminal_suppression'], float(np.max(np.abs(
                    independent['bayes'][legal] - independent[mode][legal]))))
state = {'joint': probes['joint'][0].tolist(), 'signal': int(probes['signal'][0]),
         'families': {'selected': {'actions': {'direct_bayes': 1, 'weighted_q': 4}}}}
state['illustrative_posteriors'] = producer_illustrative(state, [1, 4], config)
illustrative = _illustrative_posteriors(state, config)
specs = [{**family['spec'], 'label': 'selected',
          'artifact_sha256': family['record']['artifact_sha256']}]
envelope_source = error_envelopes(config, specs)
with TemporaryDirectory() as directory:
    path = Path(directory)
    envelope_file = path / 'error_envelopes.json'
    envelope_file.write_text(json.dumps(envelope_source))
    envelope = _envelopes(path, {'selected': family}, config)
    rejected = []
    for name in ('changed_delta_grid', 'false_certification'):
        faulty = copy.deepcopy(envelope_source)
        if name == 'changed_delta_grid':
            faulty['levels'][0]['delta_grid'] -= .01
        else:
            faulty['certified_bound'] = True
        envelope_file.write_text(json.dumps(faulty))
        try:
            _envelopes(path, {'selected': family}, config)
        except VerificationError:
            rejected.append(name)
receipt = {
    'scope': 'bounded_development_e2e_only; no h30 explanation acceptance',
    'verifier_sha256': sha(ext / 'verify_explanation.py'),
    'reproduction_script_sha256': sha(out / 'reproduction.sh'),
    'family_manifest_sha256': sha(family_path / 'complete.json'),
    'family_artifact_sha256': family['record']['artifact_sha256'],
    'frozen_probe_sha256': sha(probe_path),
    'candidate_state_sha256': sha(candidate_path),
    'probe_count_bound_to_candidates': probe_count,
    'states_checked': 4,
    'bellman_modes_per_state': 3,
    'maximum_absolute_errors': maximum,
    'illustrative_posteriors': illustrative,
    'error_envelopes': envelope,
    'mutations_rejected': rejected,
    'passed': (maximum['reward'] < 1e-10 and maximum['direct_q'] < 1e-10
               and maximum['information'] < 1e-10
               and maximum['terminal_suppression'] < 1e-10
               and illustrative['maximum_absolute_error'] < 1e-10
               and envelope['passed'] and len(rejected) == 2),
}
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
print(json.dumps({'passed': receipt['passed'], 'receipt': str(out / 'receipt.json')},
                 sort_keys=True))
if not receipt['passed']:
    raise SystemExit(1)
PY
