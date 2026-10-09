"""Independent full raw-artifact reconciliation. See campaign_failure_modes.md.

Deliberately imports no trade_learning production/statistics functions. Supports the
immutable reviewed snapshot and the corrected full campaign on the same tapes.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import zipfile
import numpy as np
import pandas as pd
from scipy.stats import t
from publication_identity import recorded_core_hash, recorded_digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pilot_digest(arrays):
    digest = hashlib.sha256()
    for key, array in sorted(arrays.items()):
        value = np.ascontiguousarray(array)
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def pilot_estimate(group, column, alpha=.05):
    # Explicit per-pilot arrays, independent of production groupby implementation.
    samples = [g[column].to_numpy() for _, g in group.groupby('pilot', sort=True)]
    require(len(samples) >= 2 and all(len(x) >= 2 for x in samples), 'Unidentified uncertainty')
    require(all(np.isfinite(x).all() for x in samples), 'Nonfinite inference input')
    means = np.array([x.mean() for x in samples])
    n = len(means)
    variance = float(np.sum((means-means.mean())**2)/(n-1))
    within = float(np.mean([np.var(x, ddof=1)/len(x) for x in samples]))
    se = float(np.sqrt(variance/n))
    delta = float(t.ppf(1-alpha/2, n-1)*se)
    return dict(mean=float(means.mean()), se=se, low=float(means.mean()-delta),
                high=float(means.mean()+delta), df=n-1, pilots=n, episodes=len(group),
                observed_pilot_mean_variance=variance, within_pilot_mean_variance=within,
                estimated_between_pilot_variance=max(0., variance-within))


def mixture(group, column):
    cells = [pilot_estimate(g, column) for _, g in group.groupby('environment', sort=True)]
    require(len(cells) == 9, 'Missing declared mixture stratum')
    mean = float(np.mean([v['mean'] for v in cells]))
    terms = np.array([v['se']**2/81 for v in cells])
    variance = float(terms.sum())
    denominator = sum(a*a/v['df'] for a,v in zip(terms,cells))
    df = variance**2/denominator if denominator else 1.
    se = np.sqrt(variance)
    return dict(mean=mean,se=float(se),df=float(df),low=float(mean-t.ppf(.975,df)*se),
                high=float(mean+t.ppf(.975,df)*se),one_sided_lower=float(mean-t.ppf(.95,df)*se),
                environments=9,pilots=sum(v['pilots'] for v in cells),episodes=len(group))


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def audit_corrected_provenance(root, manifest):
    """Check the complete recorded registry without importing production code.

    Actual cache bytes were rehashed by the run. They are deliberately excluded
    from Git; this independently checks the manifest's source/settings/build and
    content-digest bindings, not a new physical read of those excluded caches.
    """
    require(manifest['configuration_contract']=='fixed-trade-learning-benchmark-v2','Missing corrected configuration contract')
    require(sha(root/'configs/protocol.json')==manifest['configuration_sha256'],'Configuration bytes mismatch')
    require(canonical(manifest['protocol'])==manifest['protocol_sha256'],'Resolved protocol mismatch')
    for key,value in read(root/'configs/protocol.json').items():
        require(manifest['protocol'][key]==value, f'Canonical report protocol/config mismatch: {key}')
    contract=manifest['numerical_contract']
    require(contract['version']==2 and contract['score_dtype']=='float64'
            and contract['tie_absolute_tolerance']==1e-12 and contract['tie_relative_tolerance']==0,
            'Incorrect corrected numerical contract')
    sources={name:sha(root/'src/trade_learning'/name) for name in ['control.py','model.py','numerics.py','policies.py']}
    expected_source={'files':sources,'sha256':canonical(sources)}
    protocol=manifest['protocol']; expected=[]
    for mode in ['active','noinfo','full']:
        for th,ka in protocol['environments']:
            b=protocol['belief_overrides'].get(f'{th:g},{ka:g}',protocol['belief_points']) if mode!='full' else protocol['belief_points']
            expected.append(dict(theta=th,kappa=ka,mode=mode,horizon=300,qmax=5,
                                 belief_points=b,quadrature_points=protocol['quadrature_points']))
    for mode in ['independent','taker']:
        expected.append(dict(theta=0.,kappa=.02,mode=mode,horizon=300,qmax=5,
                             belief_points=protocol['belief_points'],quadrature_points=protocol['quadrature_points']))
    records=manifest['control_artifacts']
    require(len(records)==29 and len({r['key'] for r in records})==29,'Missing or duplicate control fingerprints')
    require({canonical(r['specification']) for r in records}=={canonical(s) for s in expected},'Control specification coverage')
    builds=set()
    for record in records:
        spec=record['specification']
        require(record['source']==expected_source and record['numerical_contract']==contract,'Control source or tie rule mismatch')
        build=dict(record['build']); build_sha=build.pop('sha256')
        require(canonical(build)==build_sha,'Numerical build digest mismatch');builds.add(build_sha)
        require(build['python']==manifest['python'] and build['numpy']==manifest['software']['numpy']
                and build['scipy']==manifest['software']['scipy'] and build['machine']==manifest['hardware']['machine'],
                'Run/build software mismatch')
        expected_key='v2-'+canonical({'specification':spec,'format_version':2})[:24]
        require(record['key']==expected_key,'Control key mismatch')
        require(set(record['arrays'])=={'beliefs','values','q_values'},'Control array declarations')
        b=2 if spec['mode']=='full' else spec['belief_points']
        shapes={'beliefs':[b],'values':[301,11,3,b],'q_values':[301,11,3,b,11]}
        for name, item in record['arrays'].items():
            require(item['shape']==shapes[name] and np.dtype(item['dtype'])==np.dtype('float64'),'Control array structure')
            require(item['bytes']>=8*int(np.prod(shapes[name])) and len(item['sha256'])==64,'Control array digest/size')
        inventory=np.arange(-5,6)
        mask=np.ones((11,11),dtype=bool)
        for action,(bid,ask) in enumerate(itertools.product([-1,0,1],repeat=2)):
            mask[:,action]=((bid<0)|(inventory<5))&((ask<0)|(inventory>-5))
        mask[:,9]=inventory<5;mask[:,10]=inventory>-5
        if spec['mode']=='taker':mask[:,1:9]=False
        payload={key:record[key] for key in ['key','specification','numerical_contract','source','build','arrays']}
        payload.update(format_version=2,admissibility_sha256=hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest())
        require(canonical(payload)==record['artifact_sha256'],'Artifact fingerprint does not bind recorded content/provenance')
    require(len(builds)==1,'Inconsistent numerical builds')
    return dict(checked=True,tables=29,configuration_sha256=manifest['configuration_sha256'],
                protocol_sha256=manifest['protocol_sha256'],control_source_sha256=expected_source['sha256'],
                numerical_build_sha256=next(iter(builds)),
                physical_cache_bytes='Verified by run before/after evaluation; excluded from repository')


def audit(root: Path, check_package=False, check_exports=True):
    path = root/'outputs/full'
    manifest, summary = read(path/'manifest.json'), read(path/'summary.json')
    protocol = manifest['protocol']
    require(manifest['status']=='complete', 'Incomplete run')
    source = sorted((root/'src/trade_learning').glob('*.py'))
    code_hash = hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in source)).hexdigest()
    if code_hash != manifest['code_hash']:
        require(recorded_core_hash(root)==manifest['code_hash'], 'Recorded source hash mismatch')
    specification_sha = sha(root/'BENCHMARK.md')
    if specification_sha != manifest['specification_hash']:
        specification_sha = recorded_digest(root, 'BENCHMARK.md')
    require(specification_sha==manifest['specification_hash'], 'Specification mismatch')
    require(sha(path/'episodes.csv.gz')==manifest['csv_sha256'], 'Raw CSV hash mismatch')
    provenance = audit_corrected_provenance(root,manifest) if (root/'src/trade_learning/numerics.py').exists() else {'legacy_snapshot': True}
    frame = pd.read_csv(path/'episodes.csv.gz')
    policies=['abstain','taker','independent','myopic','noinfo','active','full_information']
    variants=['nominal','state_dependence','fixed_duration']
    environments=[f'theta{th:g}_kappa{ka:g}' for th,ka in itertools.product([.15,.35,.65],[.002,.02,.1])]
    require(protocol['policies']==policies and protocol['variants']==variants, 'Advertised policy/variant mismatch')
    require(protocol['environments']==[[th,ka] for th,ka in itertools.product([.15,.35,.65],[.002,.02,.1])], 'Advertised grid mismatch')
    for key,value in [('horizon',300),('qmax',5),('pilots',10),('pilot_episodes',100),('test_episodes',100)]:
        require(protocol[key]==value, f'Wrong {key}')
    keys=['environment','pilot','variant','policy','episode']
    require(len(frame)==manifest['rows']==189000 and not frame.duplicated(keys).any(), 'Record budget or duplicate key')
    expected=set(itertools.product(environments,range(10),variants,policies))
    groups=frame.groupby(keys[:-1],sort=True)
    require(set(groups.groups)==expected, 'Incomplete or extra declared cells')
    require(all(len(g)==100 and set(g.episode)==set(range(100)) for _,g in groups), 'Episode budget/identity mismatch')
    require(len(frame[['environment','pilot','variant','episode']].drop_duplicates())==27000, 'Pairing mismatch')
    economic=['pnl','objective','inventory_penalty','spread_capture','passive_fees','market_fees',
              'market_spread_cost','liquidation_cost','directional_exposure','execution_selection','market_innovation']
    require(np.isfinite(frame[economic]).all().all(), 'Nonfinite economics')
    require((frame.max_abs_inventory<=5).all() and (frame.terminal_inventory_after_liquidation==0).all(), 'Inventory violation')
    require((frame[[f'action_{a}' for a in range(11)]].sum(axis=1)==300).all(), 'Action counts')
    ledger=frame.spread_capture-frame.passive_fees-frame.market_fees-frame.market_spread_cost-frame.liquidation_cost+frame.directional_exposure+frame.execution_selection+frame.market_innovation
    ledger_error=float(np.max(np.abs(ledger-frame.pnl)))
    require(ledger_error<1e-8 and frame.reconciliation_error.abs().max()<1e-8, 'Ledger mismatch')
    require(np.max(np.abs(frame.pnl-frame.inventory_penalty-frame.objective))<1e-10, 'Objective mismatch')
    require((frame.loc[frame.policy=='abstain',['pnl','objective']]==0).all().all(), 'Abstention mismatch')
    fits=read(path/'pilot_fits.json'); streams=manifest['streams']; ids=[]; pilot_hashes={}
    require(len(fits)==len(streams)==90, 'Pilot/stream record budget')
    for i,(stream,fit) in enumerate(zip(streams,fits)):
        env,pilot=divmod(i,10)
        require((stream['environment'],stream['pilot'])==(environments[env],pilot), 'Stream identity')
        file=path/f'pilots/{env:02d}_{pilot:02d}.npz'
        with np.load(file,allow_pickle=False) as a:
            arrays={k:a[k] for k in a.files}
        require(arrays['actions'].shape==(100,300),'Pilot budget')
        require(not set(arrays)&{'h','z','theta','kappa','u_bid','u_ask'},'Hidden pilot labels')
        digest=pilot_digest(arrays)
        require(digest==stream['pilot_data_hash']==fit['pilot_data_hash'],'Pilot fingerprint')
        group=frame[(frame.environment==environments[env])&(frame.pilot==pilot)]
        require(set(group.pilot_data_hash)=={digest},'Unpaired pilot data')
        learning=group[group.policy.isin(['active','noinfo','myopic'])]
        require(learning.starting_posterior_hash.nunique()==1,'Unpaired starting posterior')
        ids.extend([stream['pilot_seed'],stream['behavior_seed'],*stream['final'].values()])
        pilot_hashes[str(file.relative_to(root))]=sha(file)
    require(len(ids)==len(set(ids))==450,'Nonindependent random stream IDs')
    differences=[]
    def compare(expected,actual,label):
        for key,value in expected.items():
            error=abs(float(actual[key])-float(value))
            require(error<=1e-10, f'{label}.{key}: {error}')
            differences.append(error)
    require(len(summary['per_environment'])==189 and len(summary['paired_differences'])==162,'Summary cell count')
    require({(r['variant'],r['environment'],r['policy']) for r in summary['per_environment']}
            ==set(itertools.product(variants,environments,policies)), 'Summary cell identities')
    require({(r['variant'],r['environment'],r['baseline']) for r in summary['paired_differences']}
            ==set(itertools.product(variants,environments,[p for p in policies if p!='active'])),
            'Contrast identities')
    for record in summary['per_environment']:
        g=frame[(frame.variant==record['variant'])&(frame.environment==record['environment'])&(frame.policy==record['policy'])]
        for column in ['objective','pnl']:
            compare(pilot_estimate(g,column),record[column],'cell '+column)
    pivot=frame.pivot(index=['variant','environment','pilot','episode'],columns='policy',values='objective')
    for record in summary['paired_differences']:
        delta=(pivot.active-pivot[record['baseline']]).rename('difference').reset_index()
        g=delta[(delta.variant==record['variant'])&(delta.environment==record['environment'])]
        compare(pilot_estimate(g,'difference'),record['paired'],'paired')
        compare(pilot_estimate(g,'difference',.05/162),record['family_adjusted'],'adjusted')
    delta=(pivot.active-pivot.noinfo).rename('difference').reset_index()
    primary=mixture(delta[delta.variant=='nominal'],'difference')
    compare(primary,summary['primary_comparison'],'primary')
    for record in summary['overall']:
        g=frame[(frame.variant==record['variant'])&(frame.policy==record['policy'])]
        for column in ['objective','pnl']:
            compare(mixture(g,column),record[column],'overall '+column)
    require(len(summary['overall'])==21 and
            {(r['variant'],r['policy']) for r in summary['overall']}==set(itertools.product(variants,policies)),
            'Mixture identities')
    pnl=mixture(frame[(frame.variant=='nominal')&(frame.policy=='active')],'pnl')
    compare(pnl,summary['primary_comparison']['active_net_pnl'],'primary pnl')
    decision=(primary['one_sided_lower']>.05 and pnl['one_sided_lower']>0)
    require(summary['primary_comparison']['decision'].startswith('prefer_active')==decision,'Deployment decision mismatch')
    export=root/'outputs/tables/all_economic_metrics.csv'
    export_checked=0
    if check_exports:
        require(export.exists(), 'Missing requested economic export')
        table=pd.read_csv(export)
        require(len(table)==189 and not table.duplicated(['variant','environment','policy']).any(),'Economic export cell count/duplicates')
        require(set(map(tuple,table[['variant','environment','policy']].to_numpy()))==set(itertools.product(variants,environments,policies)),'Economic export cell identities')
        for row in table.to_dict('records'):
            g=frame[(frame.variant==row['variant'])&(frame.environment==row['environment'])&(frame.policy==row['policy'])]
            expected_export={}
            for column in ['objective','pnl']:
                expected_export.update({column+'_'+key:value for key,value in pilot_estimate(g,column).items()})
            switch_periods=g.post_switch_window_periods.sum()
            expected_export['pooled_post_switch_error_10']=(g.post_switch_error_count.sum()/switch_periods if switch_periods>0 else np.nan)
            sides=g.submitted_sides.sum()
            expected_export['pooled_score_residual_per_side']=(g[[f'score_sum_x{x}' for x in [-1,0,1]]].sum().sum()/sides
                if row['policy'] in ['active','noinfo','myopic'] and sides>0 else g.score_residual_per_side.mean())
            for x in [-1,0,1]:
                denominator=g[f'submitted_sides_x{x}'].sum()
                expected_export[f'pooled_public_residual_x{x}']=(g[f'score_sum_x{x}'].sum(min_count=1)/denominator if denominator>0 else np.nan)
            for action in range(11):
                expected_export[f'action_frequency_{action}']=g[f'action_{action}'].mean()/300
            quantile=float(np.quantile(g.pnl,.05))
            expected_export.update(pnl_5pct_quantile=quantile,pnl_lower_5pct_mean=float(g.loc[g.pnl<=quantile,'pnl'].mean()),
                                   loss_probability=float((g.pnl<0).mean()))
            for name,value in expected_export.items():
                actual=row[name]
                require((pd.isna(value) and pd.isna(actual)) or abs(value-actual)<1e-10, 'Economic exported statistic '+name)
                export_checked+=1
            for name,value in row.items():
                if name.startswith('mean_') and name[5:] in frame.columns:
                    actual=float(g[name[5:]].mean())
                    require((pd.isna(value) and pd.isna(actual)) or abs(value-actual)<1e-10,'Economic export '+name)
                    export_checked+=1
    text_export_checked=0
    if check_exports and not provenance.get('legacy_snapshot',False):
        policy_keys=['variant','environment','pilot','policy']
        moments=pd.read_csv(root/'outputs/tables/pilot_objective_moments.csv')
        require(len(moments)==1890 and not moments.duplicated(policy_keys).any(),'Pilot moment coverage')
        raw_groups=dict(tuple(frame.groupby(policy_keys,sort=True)))
        require(set(map(tuple,moments[policy_keys].to_numpy()))==set(raw_groups),'Pilot moment identities')
        for row in moments.to_dict('records'):
            group=raw_groups[tuple(row[k] for k in policy_keys)]
            require(row['episodes']==len(group)==100,'Pilot moment budget')
            for name in ['objective','pnl']:
                values=group[name].to_numpy()
                for suffix,expected_value in [('mean',np.mean(values)),('variance',np.var(values,ddof=1))]:
                    require(abs(row[name+'_'+suffix]-expected_value)<1e-10,'Pilot moment '+name+' '+suffix)
                    text_export_checked+=1
        pair_keys=['variant','environment','pilot','baseline']
        moments=pd.read_csv(root/'outputs/tables/pilot_paired_moments.csv')
        require(len(moments)==1620 and not moments.duplicated(pair_keys).any(),'Paired moment coverage')
        require(set(map(tuple,moments[pair_keys].to_numpy()))==set(itertools.product(variants,environments,range(10),[p for p in policies if p!='active'])),
                'Paired moment identities')
        for row in moments.to_dict('records'):
            group=pivot.xs(tuple(row[k] for k in pair_keys[:3]),level=pair_keys[:3])
            values=(group.active-group[row['baseline']]).to_numpy()
            require(row['episodes']==len(values)==100,'Paired moment budget')
            for name,expected_value in [('mean',np.mean(values)),('variance',np.var(values,ddof=1))]:
                require(abs(row[name]-expected_value)<1e-10,'Paired moment '+name)
                text_export_checked+=1
        primary_rows=pd.read_csv(root/'outputs/tables/paired_primary_episodes.csv')
        require(np.isfinite(primary_rows[['active','noinfo','active_minus_noinfo']].to_numpy()).all(),'Nonfinite primary text outcome')
        primary_keys=['environment','pilot','episode']
        require(len(primary_rows)==9000 and not primary_rows.duplicated(primary_keys).any(),'Primary text row coverage')
        original_primary=pivot.xs('nominal',level='variant')[['active','noinfo']]
        reported_primary=primary_rows.set_index(primary_keys)
        require(set(original_primary.index)==set(reported_primary.index),'Primary text row identities')
        for name in ['active','noinfo']:
            require(np.max(np.abs(original_primary[name]-reported_primary[name]))<1e-10,'Primary text outcome '+name)
        require(np.max(np.abs(reported_primary.active-reported_primary.noinfo-reported_primary.active_minus_noinfo))<1e-10,'Primary text difference')
        text_export_checked+=27000
    archive_result=None
    if check_package:
        package=root/'output/Trade Learning_Part_2_Review_Package.zip'
        spec=read(root/'output/review_package_manifest.json')
        package_status=read(root/'output/review_package_verification.json')
        require(package_status['passed'] and package_status['sha256']==sha(package)
                and package_status['bytes']==package.stat().st_size,'Package status identity mismatch')
        archives=spec.get('archives',{'primary':'output/Trade Learning_Part_2_Review_Package.zip'})
        if 'archives' in spec:
            require(set(archives)==set(package_status.get('archives',{})), 'Incomplete review-bundle status')
            require(all(record.get('archive') in archives for record in spec['files']), 'Unassigned archive member')
        archive_hashes={}
        for name,relative in archives.items():
            current_archive=root/relative
            archive_hashes[name]=sha(current_archive)
            if 'archives' in spec:
                status=package_status['archives'][name]
                require(status['path']==relative and status['sha256']==archive_hashes[name]
                        and status['bytes']==current_archive.stat().st_size,'Archive status identity '+name)
            members=[record for record in spec['files'] if record.get('archive','primary')==name]
            with zipfile.ZipFile(current_archive) as z:
                require(z.testzip() is None,'Broken ZIP')
                names=['Trade Learning/'+record['path'] for record in members]
                names.append('Trade Learning/output/review_package_manifest.json')
                require(len(z.namelist())==len(names) and set(z.namelist())==set(names),'Archive membership mismatch')
                require(json.loads(z.read(names[-1]))==spec,'Embedded package inventory differs')
                for record in members:
                    data=z.read('Trade Learning/'+record['path'])
                    require(len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256'],'Archived file hash '+record['path'])
                    require(sha(root/record['path'])==record['sha256'],'Archive differs from current file '+record['path'])
        archive_result=dict(sha256=sha(package),files=len(spec['files']),all_file_hashes_match=True,
                            all_current_tree_hashes_match=True,exact_membership=True,external_status_matches=True,
                            archives_sha256=archive_hashes)
    return dict(passed=True,kind='independent raw-file reconciliation, not independent full simulation',
                source_hash=manifest['code_hash'], published_source_hash=code_hash,
                source_migration_checked=(code_hash!=manifest['code_hash']), csv_sha256=sha(path/'episodes.csv.gz'),rows=len(frame),cells=189,
                pilot_datasets=90,paired_exogenous_episodes=27000,unique_stream_ids=450,
                primary=primary,all_162_contrasts_recomputed=True,all_189_cell_intervals_recomputed=True,
                all_42_mixture_estimates_recomputed=True,maximum_summary_numeric_error=max(differences),
                maximum_ledger_attribution_error=ledger_error,economic_export_values_checked=export_checked,
                github_text_export_values_checked=text_export_checked,
                pilot_binary_sha256=pilot_hashes,archive=archive_result,control_provenance=provenance,
                runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--package',action='store_true')
    p.add_argument('--skip-exports',action='store_true',help='Audit raw campaign before regenerating report exports')
    a=p.parse_args()
    result=audit(a.root.resolve(),a.package,check_exports=not a.skip_exports)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='pilot_binary_sha256'},indent=2))

if __name__=='__main__':
    main()
