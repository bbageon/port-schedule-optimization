"""YR-332-m: observe rejected admissions and 24/26-hour lifecycle accounting."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import evaluate_classical_rules as classical
from evaluate_arrival_order import HASHES
from six_step_learning import gz_read, gz_save, load_net, sha
from benchmark_default_eval import save
from yard_rl.v7.evaluation_probe import digest

OUT=Path('outputs/reports/yr332_admission_audit')
PREREG=Path('docs/research/v7-discharge/admission-ledger-prereg.md')
SEEDS=(9913501,9913502,9913503)
LOADS=(3500,7500,15000)
CASES=[(a,0,n,0) for n in LOADS for a in classical.RULES]+[
    (a,s,n,0) for n in LOADS for a in ('flow','linear') for s in SEEDS]+[
    ('default_reference',9913320,3500,0),('default_optimized',9913320,3500,0),
    ('flow',9913501,15000,1)]
WORKERS=6
FIELDS=('gate_in_s','block_in_s','service_start_s','job_done_s','gate_out_s')


def name(case):
    a,s,n,r=case
    return f'{a}-{s}-{n}'+('-repeat' if r else '')


def path_for(case):
    return OUT/'evaluation'/(name(case)+'.json.gz')


def source_path(case):
    a,s,n,_=case
    if a.startswith('default_'):
        return Path('outputs/reports/yr332_default_eval/campaign')/f'{n}-{s}-{a[8:]}.json.gz'
    if a=='linear':
        return Path('outputs/reports/yr332_count_transform/evaluation')/f'linear-{s}-final-{n}.json.gz'
    return classical.OUT/'evaluation'/f'{a}-{s}-{n}.json.gz'


def checkpoint(case):
    a,s,_,_=case
    if a=='flow':return Path(f'outputs/reports/yr332_six_steps/learning/model-{s}/flow-2.pt')
    if a=='linear':return Path(f'outputs/reports/yr332_count_transform/models/linear-{s}-final.pt')
    return None


def source_hashes():
    paths=[Path(__file__),PREREG,Path('scripts/v7/report_admission_ledger.py'),
        Path('scripts/v7/evaluate_classical_rules.py'),Path('scripts/v7/evaluate_arrival_order.py'),
        Path('scripts/v7/six_step_learning.py'),Path('scripts/v7/benchmark_default_eval.py'),
        *Path('src/yard_rl/v7').rglob('*.py')]
    return {p.resolve().relative_to(Path.cwd()).as_posix():sha(p) for p in paths}


def capacity_state(mbt,bid):
    sim=mbt.blocks[bid]; g=sim.profile.block
    pending=Counter(j.flow.value for j in sim.jobs.values() if j.status.value=='PLANNED')
    physical=g.bay_count*g.row_count*g.tier_max
    used=len(sim.stacks.containers)
    inbound=pending.get('GATE_IN',0); discharge=pending.get('VESSEL_DISCHARGE',0)
    reserved=mbt._reserved_inbound[bid]
    free=physical-used-inbound-discharge-reserved
    assert free==mbt.free_slots(bid)
    return dict(physical=physical,used=used,physical_free=physical-used,pending_inbound=inbound,
        pending_discharge=discharge,reserved=reserved,free=free,margin=mbt.capacity_margin,
        free_without_discharge=free+discharge)


def project_records(mbt,records,end_s):
    """Reuse the real bridge on an isolated record copy; do not advance the world."""
    from yard_rl.v7.stage.bridge import MarketBridge
    bridge=object.__new__(MarketBridge)
    bridge.records=copy.deepcopy(records)
    bridge._sync(mbt,end_s)
    return bridge.records


def lifecycle(row,end_s):
    a,b,s,c,o=[row.get(k) for k in FIELDS]
    present=lambda value:value is not None and value<=end_s+1e-9
    if not present(a):return 'NOT_ENTERED'
    if present(o):return 'EXITED'
    if present(c):return 'EXIT_TRAVEL'
    if present(s):return 'SERVICE'
    if present(b):return 'WAITING'
    return 'INBOUND_TRAVEL'


def censored_seconds(row,end_s):
    a,o=row['gate_in_s'],row['gate_out_s']
    if a is None or a>end_s:return None
    return max(0.,min(o,end_s)-a if o is not None else end_s-a)


def reason_group(reason):
    if reason in ('TAIL','NO_TARGET','CONTAINER_ID_CHANGED'):return reason
    if '용량 부족' in reason:return 'CAPACITY'
    if '반출 대상 부재' in reason:return 'MISSING_CONTAINER'
    if '투입시각 무효' in reason:return 'INVALID_TIME'
    return 'OTHER'


def install_announcer():
    import yard_rl.v7.stage.episode as episode
    base=episode.V3Announcer
    class RecordingAnnouncer(base):
        instances=[]
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.visited=Counter()
            self.review_epochs=[]
            self.instances.append(self)
        def review(self,mbt,t):
            self.review_epochs.append(float(t))
            for entry in self.by_epoch.get(round(t,6),[]):self.visited[entry['job_id']]+=1
            return super().review(mbt,t)
    episode.V3Announcer=RecordingAnnouncer
    return base,RecordingAnnouncer


class LedgerProbe(classical.StudyProbe):
    def __init__(self,announcer):
        super().__init__()
        self.announcer=announcer
        self.attempts=[]

    def start(self,built,mbt):
        super().start(built,mbt)
        self.schedule=copy.deepcopy(built['schedule'])
        self.initial_capacity={bid:capacity_state(mbt,bid) for bid in sorted(mbt.blocks)}
        original=mbt.admit_external_job
        from yard_rl.v7.world.integrated.multiblock import TransferError
        def observed(bid,job,*,gate_in_s,travel_s):
            row=dict(job_id=job.job_id,block=bid,flow=job.flow.value,
                review_s=mbt.blocks[bid].clock,gate_in_s=gate_in_s,travel_s=travel_s)
            try:
                value=original(bid,job,gate_in_s=gate_in_s,travel_s=travel_s)
            except TransferError as exc:
                row.update(accepted=False,reason=str(exc),capacity=capacity_state(mbt,bid))
                self.attempts.append(row)
                raise
            row['accepted']=True
            self.attempts.append(row)
            return value
        mbt.admit_external_job=observed

    def finish(self,mbt,records,episode):
        super().finish(mbt,records,episode)
        ann=self.announcer.instances[-1]
        before=digest({k:asdict(v) for k,v in records.items()})
        projected24=project_records(mbt,records,86400.)
        projected26=project_records(mbt,records,93600.)
        assert digest({k:asdict(v) for k,v in projected24.items()})==before
        assert digest({k:asdict(v) for k,v in records.items()})==before
        skipped={r['job_id']:r for r in ann.skips}
        assert len(skipped)==ann.n_skipped
        accepted={r['job_id'] for r in self.attempts if r['accepted']}
        assert len(accepted)==ann.n_admitted==episode.admitted
        attempts={r['job_id']:r for r in self.attempts}
        assert len(attempts)==len(self.attempts)
        assert not accepted.intersection(skipped)
        ledger={jid:(bid,tt) for bid,sim in mbt.blocks.items() for jid,tt in sim.time_ledger.records.items()}
        jobs={jid:(bid,j) for bid,sim in mbt.blocks.items() for jid,j in sim.jobs.items()}
        rows=[]
        for entry in self.schedule:
            jid=entry['job_id']; r24=projected24[jid]; r26=projected26[jid]
            raw={k:getattr(ledger[jid][1],v) if jid in ledger else None
                 for k,v in zip(FIELDS,('gate_in','block_arrival','service_start','job_done','gate_out'))}
            reason=skipped.get(jid,{}).get('reason')
            row=dict(job_id=jid,flow=entry['flow'],block=entry['block'],
                scheduled_gate_s=entry['arrival_s'],notice_s=entry['arrival_s']-entry['lead_s'],
                travel_s=entry['travel_s'],admitted=jid in accepted,review_count=ann.visited[jid],
                reason=reason,reason_group=reason_group(reason) if reason else None,
                attempt=attempts.get(jid),times=raw,
                records24={k:getattr(r24,k) for k in FIELDS},records26={k:getattr(r26,k) for k in FIELDS},
                engine_status=jobs[jid][1].status.value if jid in jobs else None)
            row['state24']=lifecycle(raw,86400.);row['state26']=lifecycle(raw,93600.)
            for end,label in ((86400.,'records24'),(93600.,'records26')):
                assert row[label]=={k:v if v is not None and v<=end+1e-9 else None for k,v in raw.items()},jid
            if jid in accepted:
                stamps=[v for v in raw.values() if v is not None]
                assert all(a<=b+1e-6 for a,b in zip(stamps,stamps[1:])),jid
            rows.append(row)
        n_entered=sum(r['times']['gate_in_s'] is not None and r['times']['gate_in_s']<=86400 for r in rows)
        n_censored=sum(r['admitted'] and r['state24']!='EXITED' for r in rows)
        assert n_entered==episode.admitted==episode.breakdown['n_trucks']
        assert n_censored==episode.breakdown['n_censored']
        mean=sum(censored_seconds(r['times'],86400.) or 0. for r in rows)/n_entered
        assert math.isclose(mean,episode.breakdown['mean_turn_time_s'],rel_tol=1e-11)
        unreviewed=[r['job_id'] for r in rows if not r['review_count']]
        assert len(rows)==ann.n_admitted+ann.n_skipped+len(unreviewed)
        assert all(n==1 for n in ann.visited.values())
        self.rows=rows
        vessels=[dict(block=bid,job_id=jid,flow=j.flow.value,status=j.status.value,
                      service_start=j.service_start,service_end=j.service_end)
                 for jid,(bid,j) in jobs.items() if j.flow.value.startswith('VESSEL_')]
        self.result['ledger_audit']=dict(input_orders=len(rows),admitted=ann.n_admitted,skipped=ann.n_skipped,
            unreviewed=unreviewed,reason_counts=dict(Counter(r['reason_group'] for r in rows if r['reason'])),
            initial_capacity=self.initial_capacity,
            final_capacity={bid:capacity_state(mbt,bid) for bid in sorted(mbt.blocks)},
            states24=dict(Counter(r['flow']+'/'+r['state24'] for r in rows)),
            states26=dict(Counter(r['flow']+'/'+r['state26'] for r in rows)),
            vessel_jobs=vessels,last_review_s=max(ann.review_epochs),
            observed_review_once=True,original_records_unchanged=True,
            projection24_exact=True,projection26_matches_engine=True)
        # The inherited legacy name refers to the official 24h records, not 26h exits.
        op=self.result['operational']
        op['gate_exit_censored_24h']=op.pop('trucks_unfinished')


def evaluate(index,smoke=False):
    import torch
    import yard_rl.v7.stage.episode as episode
    from yard_rl.v7.crane.policy import CraneNet
    torch.set_num_threads(1)
    case=CASES[index];a,s,n,repeat=case
    cp=checkpoint(case);net=load_net(cp) if cp else None
    default=a.startswith('default_')
    if default:
        torch.manual_seed(s+2);net=CraneNet().eval()
    seed=s if default else 9913900+10*LOADS.index(n)
    if smoke:seed=9914409
    original,ann=install_announcer();probe=LedgerProbe(ann)
    started=time.perf_counter()
    try:
        result=episode.run_episode(load=80 if smoke else n,seed=seed,
            dispatcher=classical.RULES.get(a,'RL_CRANE'),crane_net=net,evaluation_probe=probe,
            candidate_cap=12 if default else None,evaluation_mode=a[8:] if default else 'optimized',
            decision_probe_shortcircuit=not default)
    finally:episode.V3Announcer=original
    assert result.policy_exceptions==result.rollout_worlds==result.traded_edges==0
    out=dict(case=case,input_seed=seed,wall_seconds=time.perf_counter()-started,
        result=result.as_dict(),probe=probe.result,checkpoint=str(cp) if cp else None,
        checkpoint_sha256=sha(cp) if cp else None,threads=1)
    if not smoke:
        old=gz_read(source_path(case))
        assert out['result']==old['result'],('result',case)
        assert out['probe']['metadata']==old['probe']['metadata']
        for k in HASHES:assert out['probe'][k]==old['probe'][k],(case,k)
        out['source']=source_path(case).as_posix();out['source_sha256']=sha(source_path(case))
        out['exact_replay']=True
    dest=OUT/'smoke.json.gz' if smoke else path_for(case)
    gz_save(dest,out)
    gz_save(OUT/('smoke-orders.json.gz' if smoke else f'orders/{name(case)}.json.gz'),probe.rows)
    print(f'{name(case)}: admitted={result.admitted}; skipped={probe.result["ledger_audit"]["skipped"]}; exact={not smoke}',flush=True)


def prepare():
    from yard_rl.integrated.repro import repro_stamp
    assert not classical.git('status','--short','--untracked-files=no'),'commit before campaign'
    paths=set(source_path(c) for c in CASES)|{checkpoint(c) for c in CASES if checkpoint(c)}
    config=dict(cases=CASES,workers=WORKERS,torch_threads=1,official_end_s=86400,engine_end_s=93600,
        training=False,policy_changed=False,admission_changed=False,measurement_only=True)
    data=dict(task='YR-332-m',sources=source_hashes(),configuration=config,
        input_sha256={p.as_posix():sha(p) for p in sorted(paths)},
        repro=repro_stamp(experiment='YR-332-m',seeds=dict(input=[9913900,9913910,9913920,9913320],
            model=list(SEEDS)),params=config,prereg=PREREG.as_posix()))
    save(OUT/'campaign.json',data)


def campaign():
    prepare()
    def one(i):
        log=OUT/'logs'/f'{i}.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w',encoding='utf-8') as f:
            p=subprocess.run([sys.executable,__file__,'--operation','one','--index',str(i)],stdout=f,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if p.returncode:raise RuntimeError(log.read_text(encoding='utf-8')[-3000:])
        return i
    order=list(range(len(CASES)));random.Random(9914410).shuffle(order);done=[]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for future in as_completed([pool.submit(one,i) for i in order]):
            done.append(future.result());save(OUT/'progress.json',dict(completed=done,total=len(CASES)))
            print(f'ledger: {len(done)}/{len(CASES)}',flush=True)
    from report_admission_ledger import report
    report()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operation',choices=('campaign','one','smoke'),default='campaign')
    p.add_argument('--index',type=int,default=0)
    args=p.parse_args()
    if args.operation=='campaign':campaign()
    else:evaluate(args.index,smoke=args.operation=='smoke')
