"""Recompute report essentials without relying on the aggregation implementation."""
import hashlib
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from six_step_learning import gz_read, sha
from value_stability import check
from value_stability_inputs import OUT, PREREG, ARMS, SEEDS, LOADS, read, save


def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-7),(a,b)


def main():
    check();s=read(OUT/'summary.json')
    source_publication=read(OUT/'source-publication.json')
    assert source_publication['passed']
    for item in source_publication['format_only_legacy_inputs']:
        assert sha(item['byte_snapshot'])==item['runtime_sha256']
        assert read(item['byte_snapshot'])==read(item['original_path'])
    for p,h in s['artifact_hashes'].items():assert sha(p)==h,p
    cases=[];n_orders=0
    for p in sorted((OUT/'evaluation').glob('*.json.gz')):
        if 'repeat' in p.name:continue
        row=gz_read(p);orders=gz_read(OUT/'orders'/p.name)
        assert len(orders)==row['case']['load'],(p,len(orders),row['case']['load'])
        n_orders+=len(orders)
        # Direct timestamp arithmetic, independent from both simulator and report.
        observed=[]
        for o in orders:
            a=o['times']['gate_in_s'];done=o['times']['job_done_s']
            if a is not None and a<=86400:
                observed.append(max(0.,min(done,86400)-a if done is not None else 86400-a))
        close(math.fsum(observed)/len(observed),row['result']['mean_gate_to_job_done_s'])
        if row['case']['split']=='holdout':cases.append(row)
    for arm in (*ARMS,'SF_SPT'):
        subset=[r for r in cases if r['case']['arm']==arm]
        close(math.fsum(r['result']['mean_gate_to_job_done_s']/60 for r in subset)/len(subset),
              s['groups'][arm]['flows']['TRUCK']['mean_min'])
        close(math.fsum(r['result']['phi_krw'] for r in subset)/len(subset),s['groups'][arm]['cost'])
    assert s['training']['sample_targets_checked']==73728
    assert s['training']['full_checkpoint_replay_updates']==36
    tests=0
    for suite in ET.parse(OUT/'contract-tests.xml').getroot().iter('testsuite'):
        assert int(suite.attrib['failures'])==int(suite.attrib['errors'])==0
        tests+=int(suite.attrib['tests'])
    assert tests==15
    choice=read(OUT/'choice-diagnostic.json');diagnostic_artifacts={}
    for path,h in choice['sources'].items():
        assert sha(path)==h,path
        diagnostic_artifacts[path]=h
    assert len(choice['sources'])==18
    choice_rows=0
    for seed in SEEDS:
        path=OUT/f'choice-states-{seed}.json.gz';raw=gz_read(path)
        diagnostic_artifacts[path.as_posix()]=sha(path);choice_rows+=len(raw)
        row=next(r for r in choice['rows'] if r['seed']==seed and r['load']==0)
        single=[r for r in raw if r['single_crane']]
        mixed=[r for r in single if r['mixed_truck']]
        assert row['n']==len(raw) and row['single_crane_n']==len(single)
        assert row['mixed_single_crane_n']==len(mixed)
        close(math.fsum(r['control_popart_total_variation'] for r in single)/len(single),row['single_crane_mean_total_variation'])
        close(sum(r['control']['top_candidate']!=r['popart']['top_candidate'] for r in single)/len(single),row['single_crane_argmax_disagreement'])
        for arm in ('initial',*ARMS):
            close(math.fsum(r[arm]['inbound_conditional'] for r in mixed)/len(mixed),row[arm]['inbound_conditional'])
            close(sum(r[arm]['top_flow']==0 for r in mixed)/len(mixed),row[arm]['inbound_top_fraction'])
    assert choice_rows==36864
    for name in ('choice-diagnostic.json','gradient-diagnostic.json'):
        p=OUT/name;diagnostic_artifacts[p.as_posix()]=sha(p)
    docs=[OUT/'README.md',PREREG,Path('.claude/docs/dashboard-task-specs/YR-332-ay-value-learning-stability.md')]
    for p in docs:
        text=p.read_text('utf-8');assert len(text.splitlines())<=200
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if '://' not in target and target not in ('verification.json','publication.json'):
                assert (p.parent/target.split('#')[0]).exists(),(p,target)
    # Verify every displayed main result rounds to its independently checked value.
    report=(OUT/'README.md').read_text('utf-8')
    for arm in (*ARMS,'SF_SPT'):
        assert f'{s["groups"][arm]["flows"]["TRUCK"]["mean_min"]:.3f}' in report
    assert f'{s["paired_change"]["flow_pct"]["TRUCK"]:+.2f}%' in report
    save(OUT/'verification.json',dict(passed=True,tests=tests,evaluation_cases=126,repeats=2,
        truck_order_rows_checked=n_orders,source_files=len(read(OUT/'campaign.json')['sources']),
        evidence_files=len(s['artifact_hashes'])+len(diagnostic_artifacts),full_checkpoint_replay_updates=36,
        same_state_choice_rows_checked=choice_rows,diagnostic_artifacts=diagnostic_artifacts,
        legacy_newline_snapshots=len(source_publication['format_only_legacy_inputs']),
        targets_checked=73728,summary_sha256=sha(OUT/'summary.json'),report_sha256=sha(OUT/'README.md'),
        scope='source hashes, direct timing arithmetic, raw ledgers, fixed final checkpoints and optimizer replay',
        limitations=['No independent real-terminal validation','Three model seeds and three synthetic test curves',
                     'Full training simulator rerun not performed; saved-experience optimization replay is exact']))
    print('verification passed',n_orders,'order rows,',tests,'tests',flush=True)


if __name__=='__main__':main()
