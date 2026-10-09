"""Prewritten bounded end-to-end acceptance for AUD-01/AUD-02."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np

from trade_learning.control import solve_control, load_control, save_control, CacheIntegrityError
from trade_learning.numerics import stable_argmax, numerical_contract, artifact_fingerprint
from trade_learning.policies import TableBank, Table, GridPolicy, FixedPolicy, FullInformationReference, interpolate_scores
from trade_learning.model import admissible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('outputs/verification/audit_corrections/numerical_contract'))
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    # Corruption fixtures are disposable cache bytes, never research deliverables.
    with tempfile.TemporaryDirectory(prefix='trade-learning-numerical-contract-') as directory:
        verify(args.out, Path(directory))


def verify(out, scratch):
    checks={}; rejected={}

    def check(name,condition):
        if not bool(condition): raise AssertionError(name)
        checks[name]=True

    def fails(name,call):
        try: call()
        except (ValueError,CacheIntegrityError) as exc:
            rejected[name]=str(exc); checks[name]=True
        else: raise AssertionError(f'{name}: malformed input was accepted')

    check('exact_lowest_id',stable_argmax(np.array([1.,1.,0.]))==0)
    check('sub_tolerance_perturbation',stable_argmax(np.array([1.,1.+5e-13,0.]))==0)
    check('outside_tolerance',stable_argmax(np.array([1.,1.+2e-12,0.]))==1)
    check('absolute_not_relative',stable_argmax(np.array([1e6,1e6+1e-7]))==1)
    check('legality_precedes_tie',stable_argmax(np.array([9.,1.,1.]),np.array([False,True,True]))==1)
    fails('all_actions_illegal',lambda:stable_argmax(np.array([-np.inf,-np.inf])))
    fails('legal_nan',lambda:stable_argmax(np.array([0.,np.nan])))
    fails('legal_positive_infinity',lambda:stable_argmax(np.array([0.,np.inf])))
    s=solve_control(.35,.02,4,2,9,15,'active')
    q=s.q_values[1,1,1,-1].copy()  # q=-1,x=0,b=1: q0 abstain and q9 buy tie.
    check('myopic_mathematical_tie',abs(q[0]-q[9])<1e-12 and stable_argmax(q)==0)
    for j,delta in enumerate([-4e-13,0.,4e-13]):
        altered=q.copy(); altered[9]+=delta
        check(f'myopic_perturbation_{j}',stable_argmax(altered)==0)
    symmetric=s.q_values[1,2,1,4].copy(); mask=np.zeros(11,dtype=bool); mask[[5,7]]=True
    check('symmetric_passive_tie',abs(symmetric[5]-symmetric[7])<1e-12 and stable_argmax(symmetric,mask)==5)
    chosen=stable_argmax(s.q_values[1:])
    check('bellman_uses_chosen_action',np.array_equal(s.values[1:],np.take_along_axis(s.q_values[1:],chosen[...,None],-1)[...,0]))

    # Same legal tiny-score tie through every deployed policy class.
    fixture=np.full((2,5,3,9,11),-np.inf)
    for qi in range(5):
        fixture[1,qi,...,admissible(qi-2,2)]=-1.
    fixture[1,1,:,:,0]=0.; fixture[1,1,:,:,9]=4e-13
    table=Table(fixture,np.linspace(0,1,9),{'mode':'active','horizon':1})
    class StaticBank:
        horizon=1; qmax=2
        def table(self,*args): return table
        def learner_tables(self,*args): return [table]*9
    obs=SimpleNamespace(t=0,horizon=1,inventory=np.array([-1]),signal=np.array([0]),admissible_actions=admissible(np.array([-1]),2))
    fit=SimpleNamespace(log_weights=np.full(9,-np.log(9)))
    for mode in ('active','noinfo','myopic'):
        check('runtime_'+mode,GridPolicy(mode,fit,1,StaticBank()).choose(obs)[0]==0)
    for mode in ('taker','independent'):
        check('runtime_'+mode,FixedPolicy(mode,1,StaticBank()).choose(obs)[0]==0)
    check('runtime_full',FullInformationReference(.35,.02,StaticBank()).choose(obs,np.array([-1]))[0]==0)
    fails('remaining_zero',lambda:interpolate_scores(table,0,obs.inventory,obs.signal,np.array([.5]),2))
    fails('remaining_too_large',lambda:interpolate_scores(table,2,obs.inventory,obs.signal,np.array([.5]),2))
    standalone=scratch/'standalone'; save_control(standalone,s)
    restored=load_control(standalone)
    check('standalone_round_trip',np.array_equal(restored.values,s.values) and np.array_equal(restored.q_values,s.q_values))
    check('standalone_read_only',not restored.q_values.flags.writeable)

    original=scratch/'valid'
    bank=TableBank(original,4,2,9,15); table=bank.table(.35,.02,'active')
    key=table.metadata['key']; path=original/key
    before=bank.fingerprints()
    check('complete_fingerprints',len(before)==1 and set(before[0]['arrays'])=={'q_values','values','beliefs'})
    check('end_revalidation',bank.verify_loaded_tables()==before)
    second=TableBank(scratch/'cold_repeat',4,2,9,15); second.table(.35,.02,'active')
    check('cold_content_repeat',second.fingerprints()==before)

    def update_manifest(directory):
        p=directory/'complete.json'; m=json.loads(p.read_text())
        for name,rec in m['arrays'].items():
            file=directory/f'{name}.npy'; a=np.load(file,allow_pickle=False)
            rec.update(shape=list(a.shape),dtype=a.dtype.str,bytes=file.stat().st_size,
                       sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        m['artifact_sha256']=artifact_fingerprint(m); p.write_text(json.dumps(m))

    def case(name,mutator,resign=False):
        directory=scratch/name/key; shutil.copytree(path,directory)
        mutator(directory)
        if resign: update_manifest(directory)
        fails(name,lambda:TableBank(directory.parent,4,2,9,15).table(.35,.02,'active'))

    def array_change(directory,name,change):
        file=directory/f'{name}.npy'; a=np.load(file,allow_pickle=False); altered=change(a)
        np.save(file,a if altered is None else altered,allow_pickle=False)

    def q_set(index,value):
        return lambda d:array_change(d,'q_values',lambda a:a.__setitem__(index,value))

    case('finite_corruption',q_set((1,2,1,4,0),.12345))
    case('nan_even_with_matching_hash',q_set((1,2,1,4,0),np.nan),True)
    case('wrong_dtype',lambda d:array_change(d,'q_values',lambda a:a.astype(np.float32)),True)
    case('wrong_shape',lambda d:array_change(d,'q_values',lambda a:a[:-1]),True)
    case('nonuniform_beliefs',lambda d:array_change(d,'beliefs',lambda a:a.__setitem__(4,.5001)),True)
    case('illegal_finite_score',q_set((1,4,1,4,9),0.),True)
    case('missing_legal_score',q_set((1,2,1,4,0),-np.inf),True)
    case('terminal_q_sentinel',q_set((0,2,1,4,0),0.),True)
    case('terminal_value',lambda d:array_change(d,'values',lambda a:a.__setitem__((0,2,1,4),1.)),True)
    case('bellman_value',lambda d:array_change(d,'values',lambda a:a.__setitem__((2,2,1,4),1.)),True)

    def metadata_change(directory,change):
        file=directory/'complete.json'; m=json.loads(file.read_text()); change(m)
        m['artifact_sha256']=artifact_fingerprint(m); file.write_text(json.dumps(m))
    case('stale_build',lambda d:metadata_change(d,lambda m:m['build'].__setitem__('sha256','0'*64)))
    case('stale_source',lambda d:metadata_change(d,lambda m:m['source'].__setitem__('sha256','0'*64)))
    case('stale_contract',lambda d:metadata_change(d,lambda m:m['numerical_contract'].__setitem__('tie_absolute_tolerance',1e-4)))
    case('wrong_settings',lambda d:metadata_change(d,lambda m:m['specification'].__setitem__('theta',.15)))
    case('missing_completion',lambda d:(d/'complete.json').unlink())
    case('missing_array',lambda d:(d/'values.npy').unlink())
    case('malformed_metadata',lambda d:(d/'complete.json').write_text('{'))
    case('stale_format',lambda d:metadata_change(d,lambda m:m.__setitem__('format_version',1)))
    warm=scratch/'warm'/key; shutil.copytree(path,warm)
    warmbank=TableBank(warm.parent,4,2,9,15); warmbank.table(.35,.02,'active')
    q_set((1,2,1,4,0),.12345)(warm)
    fails('post_load_mutation',warmbank.verify_loaded_tables)
    record={'passed':True,'numerical_contract':numerical_contract(),'checks':checks,
            'rejections':rejected,'control_artifacts':before,
            'scope':'Bounded supported-stack decisions and cold-table integrity; no full campaign or cross-stack claim.'}
    (out/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'passed':True,'checks':len(checks),'rejection_probes':len(rejected),'output':str(out/'verification.json')},indent=2))


if __name__=='__main__': main()
