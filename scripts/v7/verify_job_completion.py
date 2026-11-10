"""YR-332-af: separate work completion from gate exit on identical trajectories."""
import argparse
from dataclasses import asdict
import math
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import audit_admission_ledger as ledger
import verify_reservation_contract as reservation
from benchmark_default_eval import save
from six_step_learning import gz_read, gz_save, sha
from yard_rl.integrated.repro import repro_stamp
from yard_rl.v7.evaluation_probe import digest
from yard_rl.v7.schema.lifecycle import TRUCK_TIME_CONTRACT

OUT = Path('outputs/reports/yr332_job_completion')
SPEC = Path('.claude/docs/dashboard-task-specs/YR-332-af-job-completion-objective.md')
LOADS = (3500,7500,15000)
SEED = 9915101
CASES = [(n,False) for n in LOADS]+[(7500,True)]


def raw_metrics(rows, end=86400.):
    """Independent arithmetic from raw timestamps; never call production helpers."""
    by_flow = {}
    for flow in ('ALL','GATE_IN','GATE_OUT'):
        group = [r for r in rows if flow=='ALL' or r['flow']==flow]
        work, full, post = [], [], []
        censored = exit_censored = pending = 0
        for r in group:
            a,c,o = (r['times'][k] for k in ('gate_in_s','job_done_s','gate_out_s'))
            if a is None or a>end:
                continue
            w = max(0.,(end if c is None else min(c,end))-a)
            f = max(0.,(end if o is None else min(o,end))-a)
            work.append(w);full.append(f)
            unfinished = c is None or c>end
            unexited = o is None or o>end
            censored += unfinished;exit_censored += unexited
            if not unfinished:
                post.append(max(0.,(end if o is None else min(o,end))-c))
                pending += unexited
        n=len(work)
        mean = lambda x: math.fsum(x)/len(x) if x else 0.
        q = lambda p: sorted(work)[min(n-1,int(p*n))] if n else 0.
        assert math.isclose(math.fsum(full),math.fsum(work)+math.fsum(post),abs_tol=1e-6)
        by_flow[flow] = dict(n_trucks=n,n_job_censored=censored,n_job_done=n-censored,
            n_censored=exit_censored,n_exit_pending=pending,
            mean_gate_to_job_done_s=mean(work),mean_turn_time_s=mean(full),
            p50_gate_to_job_done_s=q(.50),p90_gate_to_job_done_s=q(.90),p99_gate_to_job_done_s=q(.99),
            job_over_ratio=sum(x>3600 for x in work)/n if n else 0.,
            post_job_gate_s=math.fsum(post),mean_job_done_to_gate_out_s=mean(post),
            truck_s=math.fsum(work),
            c_wait=math.fsum(40000*x/3600+40000*max(0.,x-3600)/3600 for x in work),
            exit_based_wait=math.fsum(40000*x/3600+40000*max(0.,x-3600)/3600 for x in full))
    return by_flow


def assert_metrics(metrics, result):
    for k,v in metrics['ALL'].items():
        if k not in ('truck_s','exit_based_wait'):
            assert math.isclose(v,result[k],rel_tol=1e-11,abs_tol=1e-6),(k,v,result[k])
    assert result['truck_time_contract']==TRUCK_TIME_CONTRACT
    assert math.isclose(result['phi_krw'],sum(result[k] for k in ('c_wait','c_move','c_rehandle','c_vessel')),
                        rel_tol=1e-12)


def filename(load,repeat=False):
    return f'{load}'+('-repeat' if repeat else '')+'.json.gz'


def evaluate(load,repeat=False):
    import torch
    import yard_rl.v7.stage.episode as episode
    name=filename(load,repeat)
    for sub in ('evaluation','orders'):
        if (OUT/sub/name).exists():
            raise FileExistsError('Preserve existing evidence: '+str(OUT/sub/name))
    dirty=subprocess.check_output(['git','status','--short','--untracked-files=no'],encoding='utf-8').strip()
    if dirty:
        raise RuntimeError('Commit tracked changes before collecting evidence: '+dirty)
    torch.set_num_threads(1)
    config=dict(load=load,seed=SEED,appointment_sigma_s=600,dispatcher='SF_SPT',candidate_cap=None,
        decision_probe_shortcircuit=True,training=False,official_end_s=86400,engine_end_s=93600,repeat=repeat,
        truck_time_contract=TRUCK_TIME_CONTRACT)
    paths=[Path(__file__).resolve().relative_to(Path.cwd()),SPEC,
           *[Path('scripts/v7')/p for p in ('audit_admission_ledger.py','evaluate_classical_rules.py',
              'evaluate_arrival_order.py','verify_reservation_contract.py','benchmark_default_eval.py','six_step_learning.py')],
           *sorted(Path('src/yard_rl/v7').rglob('*.py'))]
    evidence=dict(repro=repro_stamp(experiment='YR-332-af',seeds={'input':[SEED]},params=config,
                                   prereg=SPEC.as_posix()),sources={p.as_posix():sha(p) for p in paths})
    original,announcer=ledger.install_announcer()
    probe=reservation.ReservationProbe(announcer)
    start=time.perf_counter()
    try:
        result=episode.run_episode(load=load,seed=SEED,appointment_sigma_s=600,dispatcher='SF_SPT',
            candidate_cap=None,decision_probe_shortcircuit=True,evaluation_probe=probe)
    finally:
        episode.V3Announcer=original
    result=result.as_dict()
    metrics=raw_metrics(probe.rows)
    assert_metrics(metrics,result)
    oldname=reservation.filename('evaluate',load,600)
    old=gz_read(reservation.OUT/'evaluation'/oldname)
    assert probe.rows==gz_read(reservation.OUT/'orders'/oldname)
    for k in reservation.HASHES:
        assert probe.result[k]==old['probe'][k],k
    for k,v in old['result'].items():
        if k not in ('phi_krw','c_wait'):
            assert result[k]==v,(k,result[k],v)
    evidence.update(config=config,elapsed_s=time.perf_counter()-start,result=result,probe=probe.result,
        metrics_by_flow=metrics,orders_sha256=digest(probe.rows),
        old_evaluation_path=(reservation.OUT/'evaluation'/oldname).as_posix(),
        old_evaluation_sha256=sha(reservation.OUT/'evaluation'/oldname),identical_physics=True)
    if repeat:
        first=gz_read(OUT/'evaluation'/filename(load))
        assert first['result']==result
        assert first['orders_sha256']==evidence['orders_sha256']
        assert all(first['probe'][k]==probe.result[k] for k in reservation.HASHES)
        evidence['exact_repeat']=True
    gz_save(OUT/'evaluation'/name,evidence)
    gz_save(OUT/'orders'/name,probe.rows)
    print(f'{name}: job_mean_min={metrics["ALL"]["mean_gate_to_job_done_s"]/60:.6f}, '
          f'job_unfinished={metrics["ALL"]["n_job_censored"]}, identical_physics=True',flush=True)


def verify():
    from yard_rl.v7.reward.phi import terminal_cost_krw
    from yard_rl.v7.schema.record import ExecutionRecord
    archived=[]; artifacts=[]; inputs={}
    for p in sorted((reservation.OUT/'orders').glob('*.json.gz')):
        rows=gz_read(p);raw=raw_metrics(rows)
        # Read all rows, including completion after the observation window.
        records={r['job_id']:ExecutionRecord(r['job_id'],**r['times']) for r in rows}
        b=terminal_cost_krw(records,end_s=86400.).as_dict()
        assert_metrics(raw,b)
        old=gz_read(reservation.OUT/'evaluation'/p.name)
        assert math.isclose(raw['ALL']['exit_based_wait'],old['result']['c_wait'],rel_tol=1e-11)
        archived.append(dict(path=p.as_posix(),metrics_by_flow=raw))
        inputs[p.as_posix()]=sha(p)
        inputs[(reservation.OUT/'evaluation'/p.name).as_posix()]=sha(reservation.OUT/'evaluation'/p.name)
    for load,repeat in CASES:
        name=filename(load,repeat);p=OUT/'evaluation'/name; artifact=gz_read(p)
        rows=gz_read(OUT/'orders'/name);raw=raw_metrics(rows)
        assert_metrics(raw,artifact['result'])
        assert artifact['identical_physics']
        assert digest(rows)==artifact['orders_sha256']
        oldpath=Path(artifact['old_evaluation_path'])
        assert sha(oldpath)==artifact['old_evaluation_sha256']
        old=gz_read(oldpath)
        assert all(old['probe'][k]==artifact['probe'][k] for k in reservation.HASHES)
        assert rows==gz_read(reservation.OUT/'orders'/oldpath.name)
        artifacts.append(artifact)
        inputs[p.as_posix()]=sha(p);inputs[(OUT/'orders'/name).as_posix()]=sha(OUT/'orders'/name)
    first,again=artifacts[1],artifacts[-1]
    assert first['result']==again['result'] and first['orders_sha256']==again['orders_sha256']
    source_check=reservation.verify_sources(artifacts)
    tests={}
    for p in sorted(OUT.glob('tests-*.xml')):
        for tc in ET.parse(p).iter('testcase'):
            key=tc.attrib.get('classname','')+'::'+tc.attrib['name']
            tests[key]=not any(tc.find(k) is not None for k in ('failure','error','skipped'))
        inputs[p.as_posix()]=sha(p)
    # Explicit pre-existing whole-world divergence, reproduced in clean a2595188.
    known='tests.v7.test_gpu_month::test_day_phi_matches_v5'
    baseline=OUT/'baseline-known-failures.xml'
    old_failures={tc.attrib.get('classname','')+'::'+tc.attrib['name']
        for tc in ET.parse(baseline).iter('testcase') if tc.find('failure') is not None}
    assert known in old_failures and tests.get(known) is False
    passed={k:v for k,v in tests.items() if k!=known}
    assert passed and all(passed.values()),[k for k,v in passed.items() if not v]
    inputs[baseline.as_posix()]=sha(baseline)
    inputs[(OUT/'baseline-check.json').as_posix()]=sha(OUT/'baseline-check.json')
    # Endpoint correction does not establish a new learned policy benefit.
    report=dict(passed=True,truck_time_contract=TRUCK_TIME_CONTRACT,training=False,
        archived_runs=len(archived),archived_orders=sum(x['metrics_by_flow']['ALL']['n_trucks'] for x in archived),
        evaluation_runs=len(artifacts),admitted=sum(x['result']['admitted'] for x in artifacts),
        exact_repeat=True,identical_physics_all_runs=True,source_check=source_check,
        tests_passed=len(passed),known_preexisting_failures=[known],
        array_month_physical_equivalence_verified=False,inputs=inputs,archived=archived,
        current=[dict(load=x['config']['load'],metrics_by_flow=x['metrics_by_flow']) for x in artifacts[:3]],
        normalization='existing frozen exit-based scales retained; no refit',
        conclusion='Accounting contract verified; changes are metric definition effects, not learning gains.')
    save(OUT/'verification.json',report)
    print({k:v for k,v in report.items() if k not in ('inputs','archived','current')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('evaluate','verify'))
    parser.add_argument('--load',type=int,choices=LOADS,default=7500)
    parser.add_argument('--repeat',action='store_true')
    args=parser.parse_args()
    if args.mode=='evaluate':evaluate(args.load,args.repeat)
    else:verify()
