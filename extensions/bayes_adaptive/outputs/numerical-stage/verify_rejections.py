"""Bounded CLI rejection checks for the independent numerical raw auditor.

Failure modes were recorded in NUMERICAL_VALIDATION.md before implementation.
All changes are confined to temporary receipt/metadata copies; array bytes stay read-only.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
EXT = ROOT/'extensions/bayes_adaptive'
HERE = Path(__file__).resolve().parent
VERIFIER = EXT/'verify_numerical_raw.py'
REFERENCE = EXT/'outputs/numerical-amended/numerical_checks.json'
CACHE = EXT/'cache/v2'

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def main():
    original = json.loads(REFERENCE.read_text())
    cases = []
    record = {'passed': False, 'verifier_sha256': sha(VERIFIER), 'harness_sha256': sha(Path(__file__)),
              'reference_receipt_sha256': sha(REFERENCE), 'cases': cases,
              'scope': 'CLI rejection of changed numerical receipt or cache metadata; no economic outcomes'}
    def run(name, changed, cache=CACHE, expected=''):
        with tempfile.TemporaryDirectory(prefix='bayes-numeric-reject-') as temporary:
            out = Path(temporary)
            (out/'numerical_checks.json').write_text(json.dumps(changed))
            (out/'numerical_probes.npz').symlink_to(REFERENCE.parent/'numerical_probes.npz')
            result = subprocess.run([sys.executable,str(VERIFIER),'--numerical-checks',str(out/'numerical_checks.json'),
                                     '--cache',str(cache),'--pair','0','--axis','belief'],
                                    capture_output=True,text=True,cwd=ROOT)
            passed = result.returncode != 0 and expected in result.stderr
            cases.append({'case':name,'rejected':passed,'returncode':result.returncode,'stderr':result.stderr.strip()})
            if not passed:
                raise AssertionError(name)
    try:
        for name, mutate, expected in [
            ('changed_threshold',lambda r:r['thresholds'].update(probe_sup=.5),'thresholds'),
            ('missing_horizon',lambda r:r['comparisons'][0]['by_horizon_policy'].pop(),'incomplete policy'),
            ('changed_grid_axis',lambda r:r['comparisons'][0]['fine_specification'].update(belief_points=35),'changes another setting'),
            ('unresolved_declared_accepted',lambda r:r.update(passes_predeclared_rule=True),'without joint and known-model'),
            ('changed_probe_identity',lambda r:r.update(probe_sha256='0'*64),'probe file does not match'),
            ('changed_producer_source',lambda r:r['source'].update(sha256='0'*64),'source is stale'),
        ]:
            changed=deepcopy(original);mutate(changed);run(name,changed,expected=expected)
        with tempfile.TemporaryDirectory(prefix='bayes-numeric-array-metadata-') as temporary:
            cache=Path(temporary);name='w9-b17-gh25';family=cache/name;family.mkdir()
            for file in (CACHE/name).glob('*.npy'):
                (family/file.name).symlink_to(file)
            complete=json.loads((CACHE/name/'complete.json').read_text())
            complete['arrays'][0]['dtype']='float32'
            complete.pop('artifact_sha256');complete['artifact_sha256']=canonical(complete)
            (family/'complete.json').write_text(json.dumps(complete))
            changed=deepcopy(original)
            changed['comparisons'][0]['coarse_artifact_sha256']=complete['artifact_sha256']
            run('false_declared_dtype_with_valid_manifest_hash',changed,cache,expected='shape/dtype')
        record['passed']=True
        record['rejection_cases']=len(cases)
        assert sha(REFERENCE)==record['reference_receipt_sha256']
    finally:
        (HERE/'independent_rejections.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'passed':True,'cases':len(cases)}))

if __name__=='__main__':
    main()
