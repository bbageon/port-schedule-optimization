"""Independent interval totals from the flow ledger and original visit records."""
from collections import defaultdict
from datetime import datetime
import csv
import math
import statistics as st
import subprocess

from inbound_realism_common import BASE, RAW, SOURCE, OUT, cases, read, save, sha
from gate_time_provenance import CONFIRMED, load_evidence


def equal(a,b):
    assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(a,b)


def main():
    sim=read(OUT/'simulation.json');real=read(OUT/'real-records.json');choice=read(OUT/'choices.json')
    published=read(SOURCE/'publication.json')['artifact_hashes'];source_checks={}
    for analysis in (sim,choice):
        for path,want in analysis['sources'].items():
            p=SOURCE/path;got=sha(p)
            assert got==want==published[p.relative_to(BASE).as_posix()]
            source_checks[p.relative_to(BASE).as_posix()]=got
    lookup={(r['case']['arm'],r['case']['stage'],r['case']['seed'],r['case']['index'],r['case']['load'],r['flow']):r for r in sim['per_case']}
    checked_rows=0;checked_intervals=0
    for i,c in enumerate(cases()):
        if c['split']!='holdout' or c['mode']!='argmax':continue
        p=SOURCE/'flow-orders'/f'{i:03d}.json.gz';source_checks[p.relative_to(BASE).as_posix()]=sha(p)
        assert source_checks[p.relative_to(BASE).as_posix()]==published[p.relative_to(BASE).as_posix()]
        flows=read(p)
        for flow in ('GATE_IN','GATE_OUT'):
            rows=[r for r in flows if r['flow']==flow];checked_rows+=len(rows)
            wanted=lookup[c['arm'],c['stage'],c['seed'],c['index'],c['load'],flow]
            for hour in (24,26):
                end=hour*3600
                durations=[(min(end,r['finish_s'] if r['finish_s'] is not None else end)-r['origin_s'])/60 for r in rows]
                equal(sum(durations)/len(durations),wanted[f'at{hour}']['means']['total'])
                assert sum(r['finish_s'] is None or r['finish_s']>end for r in rows)==wanted[f'at{hour}']['unfinished']
                checked_intervals+=len(rows)
    proof=load_evidence((RAW/'기사/_raw').glob('*.gz'));full=defaultdict(list)
    for p in sorted((RAW/'기사/방문').glob('*.csv')):
        with p.open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                if proof.classify(r) not in CONFIRMED:continue
                if not all(r.get(k) for k in ('gate_in_time','enter_block_time','job_done_time')):continue
                a,b,c=(datetime.fromisoformat(r[k]) for k in ('gate_in_time','enter_block_time','job_done_time'))
                if a<=b<=c:full[r['in_out']].append(((b-a).total_seconds()/60,(c-b).total_seconds()/60,(c-a).total_seconds()/60))
    for flow,x in full.items():
        want=real['groups'][flow]['matched_triplets'];assert len(x)==want['n']
        for j,name in enumerate(('gate_to_block','block_to_done','gate_to_done')):
            values=[r[j] for r in x];equal(sum(values)/len(values),want[name]['mean']);equal(st.median(values),want[name]['median'])
        assert all(v['within_rounding'] for v in real['groups'][flow]['derived_column_error_min'].values())
    total=next(r for r in choice['groups'] if r['seed']==r['load']==0)
    for key in ('seed','load'):
        cells=[r for r in choice['groups'] if r[key]!=0 and r['load' if key=='seed' else 'seed']==0]
        assert sum(r['n'] for r in cells)==total['n']
        for arm in ('initial','zero','global'):
            for metric in ('inbound_conditional','inbound_argmax','outbound_argmax'):
                equal(sum(r['n']*r[arm][metric] for r in cells)/total['n'],total[arm][metric])
    assert choice['states']==55296 and choice['counts']['mixed_single']==total['n']
    expected_fields=real['raw_timing_fields']
    assert 'jobDoneTime' in expected_fields and not any('start' in x.lower() for x in expected_fields)
    # The raw data remain untouched, including every file not used directly in a statistic.
    for r in real['raw_files']:assert sha(RAW/r['path'])==r['sha256']
    scripts={p.relative_to(BASE).as_posix():sha(p) for p in (BASE/'scripts/v7').glob('*inbound_realism*.py')}
    save('verification.json',dict(passed=True,task='YR-332-bd',source_bc_commit='fc133978d9f705355155364b5f93810d2cf1b5c8',
        analysis_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        tracked_status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True,encoding='utf-8'),
        independent_truck_rows=checked_rows,intervals_at_two_horizons=checked_intervals,
        real_triplets={k:len(v) for k,v in full.items()},same_state_samples=choice['states'],mixed_single_states=total['n'],
        raw_unchanged_files=len(real['raw_files']),source_hashes=source_checks,analysis_scripts=scripts,
        artifact_hashes={n:sha(OUT/n) for n in ('simulation.json','real-records.json','choices.json')},
        scope='Independent numeric totals and unchanged sources; not field semantics or a causal real-world trial'))
    print('verification passed',checked_rows,'truck rows;',sum(map(len,full.values())),'real triplets;',choice['states'],'saved choice states',flush=True)


if __name__=='__main__':main()
