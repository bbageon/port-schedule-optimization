"""Frozen argmax evaluation and separately labelled stochastic value validation."""
from __future__ import annotations

import copy
import time

from six_step_learning import gz_save, gz_read, sha
from value_stability_inputs import (OUT, SEEDS, ARMS, SCALE, LOADS, RUNTIME, INFO,
                                   read, save, demand, input_case)
from yard_rl.v7.evaluation_probe import digest


def cases():
    rows = []
    for seed in SEEDS:
        for i, load in enumerate(LOADS):
            rows.append(dict(**input_case('selection', i, load), arm='control', seed=seed,
                             stage='initial', mode='argmax'))
    for arm in ARMS:
        for seed in SEEDS:
            for stage in ('update-2', 'final'):
                for i, load in enumerate(LOADS):
                    rows.append(dict(**input_case('selection', i, load), arm=arm, seed=seed,
                                     stage=stage, mode='argmax'))
            for i, load in enumerate(LOADS):
                rows.append(dict(**input_case('selection', i, load), arm=arm, seed=seed,
                                 stage='final', mode='stochastic'))
            for i in range(3):
                for load in LOADS:
                    rows.append(dict(**input_case('holdout', i, load), arm=arm, seed=seed,
                                     stage='final', mode='argmax'))
    for i in range(3):
        for load in LOADS:
            rows.append(dict(**input_case('holdout', i, load), arm='SF_SPT', seed=0,
                             stage='rule', mode='argmax'))
    return rows


def rollout(case, net=None):
    import torch
    import audit_admission_ledger as ledger
    import yard_rl.v7.stage.episode as episode
    from horizon_metrics import FlowProbe, aggregate
    from verify_job_completion import raw_metrics, assert_metrics
    from yard_rl.v7.crane.on_policy import OnPolicyResolver
    torch.set_num_threads(1)
    stochastic = case['mode']=='stochastic'
    resolver = (OnPolicyResolver(net, stochastic=stochastic, collect=stochastic,
                seed=9956000+case['index']+100*LOADS.index(case['load'])) if net is not None else None)
    original, announcer = ledger.install_announcer()
    class Probe(FlowProbe):
        def finish(self, mbt, records, ep):
            for sim in mbt.blocks.values(): sim.check_invariants()
            self.visits = dict(mbt.visit_stats)
            return super().finish(mbt, records, ep)
    probe = Probe(announcer); started = time.perf_counter()
    try:
        ep = episode.run_episode(load=case['load'], seed=case['input_seed'],
            dispatcher='RL_CRANE' if net is not None else case['arm'], crane_net=net,
            joint_resolver=resolver, information_profile=read(INFO),
            demand_profile=demand(case['split'], case['index']), evaluation_probe=probe, **RUNTIME)
    finally:
        episode.V3Announcer = original
    result = ep.as_dict(); metrics = raw_metrics(probe.rows); assert_metrics(metrics,result)
    flows = copy.deepcopy(probe.flow_rows)
    for row in flows:
        if row['flow'].startswith('GATE_'): row['finish_s']=row['yard_done_s']
    data = dict(case=case, result=result, metrics=metrics, probe=probe.result, visits=probe.visits,
        completion={f'at{h}':aggregate(flows,h*3600.) for h in (24,26)},
        orders_sha256=digest(probe.rows), schedule_sha256=digest(probe.schedule),
        seconds=time.perf_counter()-started)
    assert ep.admitted==case['load'] and ep.policy_exceptions==ep.rollout_worlds==ep.traded_edges==0
    assert probe.result['ledger_audit']['skipped']==0 and all(r['review_count']==1 for r in probe.rows)
    assert probe.visits['late_releases']==probe.visits['late_exits']==0
    if resolver:
        data['on_policy'] = dict(decisions=resolver.decisions, nontrivial=resolver.nontrivial,
            max_candidates=resolver.max_candidates, oracle_calls=resolver.oracle_calls,
            latency_seconds=dict(count=len(resolver.latencies), total=sum(resolver.latencies)))
        if stochastic: data['accounting']=resolver.finish(ep.phi_krw,SCALE)
    return data, probe.rows, flows, resolver


def evaluate(i, repeat=False):
    import torch
    from resched_ppo import encode_sample
    from value_stability import check, folder, restore, value_diagnostic
    check(); case=cases()[i]; name=f'{i:03d}'+('-repeat' if repeat else '')
    path=OUT/'evaluation'/f'{name}.json.gz'
    if path.exists():
        assert gz_read(path)['case']==case
        print(f'existing eval {name}',flush=True); return
    net=None; cp=None
    if case['arm'] in ARMS:
        cp=folder(case['arm'],case['seed'])/f'{case["stage"]}.pt'
        net, _, _=restore(cp)
    data, orders, flows, resolver=rollout(case,net)
    data['checkpoint_sha256']=sha(cp) if cp else None
    if case['mode']=='stochastic':
        rng=torch.Generator().manual_seed(9956100+case['index'])
        indices=torch.randperm(len(resolver.samples),generator=rng)[:2048].tolist()
        selected=[resolver.samples[j] for j in indices]
        diagnostic=value_diagnostic(net,selected,[case]*len(selected))
        data['value_diagnostic']={k:v for k,v in diagnostic.items() if k not in ('values','action_logps')}
        gz_save(OUT/'validation-samples'/f'{name}.json.gz',dict(indices=indices,
            samples=[encode_sample(s) for s in selected],diagnostic=diagnostic))
    for sub, value in [('evaluation',data),('orders',orders),('flow-orders',flows)]:
        gz_save(OUT/sub/f'{name}.json.gz',value)
    if repeat:
        prior=gz_read(OUT/'evaluation'/f'{i:03d}.json.gz')
        for key in ('candidates_sha256','choices_sha256','events_sha256','jobs_sha256','records_sha256','result_sha256'):
            assert prior['probe'][key]==data['probe'][key],key
        assert prior['orders_sha256']==data['orders_sha256']
    print(f'eval {name} {case["arm"]} {case["seed"]} {case["stage"]} {case["split"]} {case["index"]} load={case["load"]} {data["result"]["mean_gate_to_job_done_s"]/60:.3f}min {data["seconds"]:.1f}s',flush=True)


def smoke():
    import torch
    from value_stability import checkpoint, folder, value_diagnostic
    from yard_rl.v7.crane.value_normalization import NormalizedValueNet, fit
    torch.set_num_threads(1); torch.manual_seed(9954001)
    initial=NormalizedValueNet().eval()
    case=dict(**input_case('train',12,3500),arm='control',seed=9954001,stage='smoke',mode='stochastic')
    data, _, _, resolver=rollout(case,initial)
    samples=resolver.samples[:256]; rows=[]
    for arm in ARMS:
        net=copy.deepcopy(initial)
        optimizer=torch.optim.Adam(net.parameters(),lr=.0003)
        before=value_diagnostic(net,samples,[case]*len(samples))
        result=fit(net,optimizer,samples,arm=arm,seed=9954003,max_samples=256,epochs=1,batch_size=64)
        after=value_diagnostic(net,samples,[case]*len(samples))
        rows.append(dict(arm=arm,optimization=result,before_rmse=before['rmse'],after_rmse=after['rmse']))
    save(OUT/'smoke.json',dict(passed=True,case=case,rollout=data,updates=rows,
                              excluded_from_campaign=True))
    print(f'smoke passed, rollout={data["seconds"]:.1f}s',flush=True)
