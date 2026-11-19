"""Direct timestamp arithmetic and independent aggregation for reported bc results."""
import math
from pathlib import Path
import re
import subprocess
import numpy as np

from global_policy_inputs import OUT,BB,SEEDS,ARMS,LOADS,INFO,read,save,bundle,demand
from global_policy_plan import check,model_path
from six_step_learning import gz_read,sha


def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-7),(a,b)


def raw_flow_times(flows,end):
    values={}
    for row in flows:
        start=row['origin_s'];done=row['yard_done_s']
        if start is None or start>end:continue
        finish=min(done,end) if done is not None else end
        assert finish>=start-1e-6
        keys=(row['flow'],'TRUCK' if row['flow'].startswith('GATE_') else 'VESSEL','ALL_ORDERS')
        for key in keys:values.setdefault(key,[]).append((max(0.,finish-start),done is None or done>end))
    return {key:dict(n=len(v),mean_s=math.fsum(x for x,_ in v)/len(v),unfinished=sum(u for _,u in v)) for key,v in values.items()}


def check_group(rows,expected):
    assert len(rows)==expected['cases']
    close(math.fsum(r['result']['phi_krw'] for r in rows)/len(rows),expected['cost'])
    close(math.fsum(r['result']['c_vessel'] for r in rows)/len(rows),expected['vessel_cost'])
    close(math.fsum(r['completion']['at26']['ALL_ORDERS']['unfinished'] for r in rows)/len(rows),expected['unfinished26_mean'])
    for flow,d in expected['flows'].items():
        cells=[r['completion']['at24'][flow] for r in rows]
        close(math.fsum(c['mean_s'] for c in cells)/len(cells)/60,d['mean_min'])
        close(math.fsum(c['total_s'] for c in cells)/sum(c['n'] for c in cells)/60,d['order_weighted_min'])
        for h in (24,26):
            close(math.fsum(r['completion'][f'at{h}'][flow]['unfinished'] for r in rows)/len(rows),d[f'unfinished{h}_mean'])


def choose(rows,label):
    arm,stage=('zero','initial') if label=='initial' else (label,'rule' if label=='SF_SPT' else 'final')
    return [r for r in rows if r['case']['arm']==arm and r['case']['stage']==stage]


def verify_bootstrap(rows,obj):
    grids=[]
    for label in (obj['baseline'],obj['candidate']):
        selected=choose(rows,label);grid=[]
        for day in range(3):
            cells=[]
            for seed in SEEDS:
                cells.append([r['result']['phi_krw'] for r in selected if r['case']['index']==day and
                    (label=='SF_SPT' or r['case']['seed']==seed)])
            assert all(len(c)==3 for c in cells)
            grid.append([math.fsum(c)/3 for c in cells])
        grids.append(grid)
    rng=np.random.default_rng(10089900);diffs=[];ratios=[]
    for _ in range(10000):
        dates=rng.integers(0,3,3);models=rng.integers(0,3,3)
        a,b=[math.fsum(grid[d][m] for d in dates for m in models)/9 for grid in grids]
        diffs.append(b-a);ratios.append(100*(b/a-1))
    for field,values in [('cost_difference_ci95',diffs),('change_pct_ci95',ratios)]:
        for a,b in zip(np.quantile(values,[.025,.975]),obj[field]):close(a,b)


def main():
    cfg=check();summary=read(OUT/'summary.json');audit=read(OUT/'audit.json');assert audit['passed']
    execution=read(OUT/'post-training-execution.json')
    recovered=not execution['passed']
    if recovered:
        assert execution['stages'][-1]['stage']=='report' and execution['stages'][-1]['returncode']==1
        assert all(stage['returncode']==0 for stage in execution['stages'][:-1])
        recovery=read(OUT/'post-processing-recovery.json');assert recovery['passed']
        assert recovery['failed_record_sha256']==sha(OUT/recovery['preserved_failure'])
        assert recovery['script_after_sha256']==sha(Path('scripts/v7/report_global_policy.py'))
        assert not recovery['experiment_restarted'] and not recovery['training_evaluation_or_criteria_changed']
        assert recovery['visual_review']=='passed'
        for name,digest in recovery['artifact_hashes'].items():assert sha(Path(name))==digest
    else:assert all(stage['returncode']==0 for stage in execution['stages'])
    curves={(c['split'],c['index']) for c in cfg['train']+cfg['development']+cfg['holdout']}
    wiring_checks=0
    for split,index in sorted(curves):
        for load in LOADS:
            case=bundle().case(split,index,load)
            assert case['load']==load and case['demand_profile']==demand(split,index)
            assert case['information_profile']==read(INFO)
            wiring_checks+=1
    manifest=read(BB/'source-manifest.json');assert len(manifest)==600
    for entry in manifest:
        path=Path('오더데이터')/entry['path']
        assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],path
    print('input wiring and 600 original file hashes passed',flush=True)
    rows=[];count=0;flow_count=0
    for i,c in enumerate(cfg['evaluation_cases']):
        raw=gz_read(OUT/'evaluation'/f'{i:03d}.json.gz');assert raw['case']==c
        orders=gz_read(OUT/'orders'/f'{i:03d}.json.gz');flows=gz_read(OUT/'flow-orders'/f'{i:03d}.json.gz')
        count+=len(orders);flow_count+=len(flows);assert len(orders)==c['load']
        times=[]
        for r in orders:
            a=r['times']['gate_in_s'];done=r['times']['job_done_s']
            assert a is not None and a<=86400
            times.append(max(0.,(min(done,86400) if done is not None else 86400)-a))
        close(math.fsum(times)/len(times),raw['result']['mean_gate_to_job_done_s'])
        for h in (24,26):
            calculated=raw_flow_times(flows,h*3600.)
            for f,d in calculated.items():
                reference=raw['completion'][f'at{h}'][f]
                assert d['n']==reference['n'] and d['unfinished']==reference['unfinished'];close(d['mean_s'],reference['mean_s'])
                assert raw['completion']['at24'][f]['n']==raw['completion']['at26'][f]['n']
        if c['split']=='holdout':rows.append(raw)
        if (i+1)%30==0:print('independent timestamp checks:',i+1,'/ 180',flush=True)
    groups={}
    for label in ('initial','zero','global','SF_SPT'):
        selected=choose(rows,label)
        assert len(selected)==(9 if label=='SF_SPT' else 27)
        expected=summary['groups'][label]
        check_group(selected,expected)
        groups[label]=expected
    for axis in ('seed','load'):
        for key,obj in summary[f'by_{axis}'].items():
            subset=[r for r in rows if r['case'][axis]==int(key) or (axis=='seed' and r['case']['arm']=='SF_SPT')]
            for label,expected in obj['groups'].items():check_group(choose(subset,label),expected)
    for a,b,key in [('zero','global','paired_change'),('SF_SPT','global','vs_rule')]:
        close(100*(groups[b]['cost']/groups[a]['cost']-1),summary[key]['cost_pct'])
        for f in groups[a]['flows']:
            close(100*(groups[b]['flows'][f]['mean_min']/groups[a]['flows'][f]['mean_min']-1),summary[key]['flow_pct'][f])
    for key,obj in summary['uncertainty'].items():
        close(math.fsum(v for row in obj['paired_cost_matrix'] for v in row)/9,obj['cost_difference'])
        verify_bootstrap(rows,obj)
    assert audit['training']['updates']==audit['training']['weights_optimizer_replay_updates']==36
    assert audit['training']['targets']==73728 and audit['evaluation']['cases']==180
    assert audit['evaluation']['repeats']==2 and audit['tests']==19
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.name not in
        ('verification.json','publication.json','current-gates.json','current-gates.json.sha256','dashboard-audit.json','dashboard-audit.json.sha256')
        and not p.name.endswith('.log')]
    artifacts={p.as_posix():sha(p) for p in paths}
    for p in Path('scripts/v7').glob('*global_policy*.py'):artifacts[p.as_posix()]=sha(p)
    for name in ('test_global_policy_exploration.py','test_global_value_context.py','test_confirmed_demand.py'):
        p=Path('tests/v7')/name;artifacts[p.as_posix()]=sha(p)
    p=BB/'source-manifest.json';artifacts[p.as_posix()]=sha(p)
    report=(OUT/'README.md').read_text(encoding='utf-8')
    assert len(report.splitlines())<=200
    for label in groups:
        assert f'{groups[label]["flows"]["TRUCK"]["mean_min"]:.3f}' in report
        assert f'{groups[label]["flows"]["VESSEL"]["mean_min"]:.3f}' in report
    assert f'{summary["paired_change"]["cost_pct"]:+.2f}%' in report
    save(OUT/'verification.json',dict(passed=True,tests=19,evaluations=180,repeats=2,
        original_files_unchanged=len(manifest),original_manifest_sha256=sha(BB/'source-manifest.json'),
        independent_input_wiring_checks=wiring_checks,
        report_only_recovery_verified=recovered,
        evaluation_truck_rows=count,evaluation_flow_rows=flow_count,training_truck_rows=audit['training']['orders'],
        sample_targets=73728,diagnostic_targets=55296,optimizer_updates_replayed=36,source_files=len(read(OUT/'campaign.json')['sources']),
        artifact_hashes=artifacts,summary_sha256=sha(OUT/'summary.json'),report_sha256=sha(OUT/'README.md'),
        policy_selection='final update 5 in every model; no prediction-error veto',
        scope='Independent direct timestamps, fixed cohort means, final model comparisons and stored-experience replay',
        limitations=['No full training-trajectory replay','Synthetic-family evaluation, not real-terminal confirmation']))
    print('independent verification passed:',count,'truck rows and',flow_count,'all-flow rows',flush=True)


if __name__=='__main__':main()
