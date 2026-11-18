"""Record-only and critic-context trajectories share exactly the same world path."""
import copy
import time
from dataclasses import asdict

from global_value_inputs import (RUNTIME, SCALE, INFO, demand, read)


def rollout(case, net=None, *, mode='off', stochastic=False, collect=False, action_seed=0):
    import torch
    import audit_admission_ledger as ledger
    import yard_rl.v7.stage.episode as episode
    from horizon_metrics import FlowProbe, aggregate
    from verify_job_completion import raw_metrics, assert_metrics
    from yard_rl.v7.evaluation_probe import digest
    from yard_rl.v7.crane.on_policy import OnPolicyResolver
    torch.set_num_threads(1)
    resolver = (OnPolicyResolver(net, stochastic=stochastic, collect=collect,
                                seed=action_seed, global_context_mode=mode) if net is not None else None)
    original, announcer = ledger.install_announcer()
    class Probe(FlowProbe):
        def finish(self, mbt, records, ep):
            for sim in mbt.blocks.values(): sim.check_invariants()
            self.visits = dict(mbt.visit_stats)
            return super().finish(mbt, records, ep)
    probe = Probe(announcer); started = time.perf_counter()
    try:
        ep = episode.run_episode(load=case['load'], seed=case['input_seed'],
            dispatcher='RL_CRANE' if net is not None else 'SF_SPT', crane_net=net,
            joint_resolver=resolver, information_profile=read(INFO),
            demand_profile=demand(case['split'], case['index']), evaluation_probe=probe, **RUNTIME)
    finally:
        episode.V3Announcer = original
    result = ep.as_dict(); metrics = raw_metrics(probe.rows); assert_metrics(metrics, result)
    flows = copy.deepcopy(probe.flow_rows)
    for row in flows:
        if row['flow'].startswith('GATE_'): row['finish_s'] = row['yard_done_s']
    data = dict(case=case, result=result, metrics=metrics, probe=probe.result, visits=probe.visits,
        completion={f'at{h}':aggregate(flows, h*3600.) for h in (24,26)},
        orders_sha256=digest(probe.rows), schedule_sha256=digest(probe.schedule),
        seconds=time.perf_counter()-started)
    assert ep.admitted == case['load'] and ep.policy_exceptions == ep.rollout_worlds == ep.traded_edges == 0
    assert probe.result['ledger_audit']['skipped'] == 0 and all(r['review_count']==1 for r in probe.rows)
    assert probe.visits['late_releases'] == probe.visits['late_exits'] == 0
    if resolver:
        data['on_policy'] = dict(decisions=resolver.decisions, nontrivial=resolver.nontrivial,
            max_candidates=resolver.max_candidates, oracle_calls=resolver.oracle_calls)
        if collect: data['accounting'] = resolver.finish(ep.phi_krw, SCALE)
        snapshots = [asdict(s) for s in resolver.global_snapshots]
        data['global_audit'] = dict(mode=mode, snapshots=snapshots,
            decision_observations=len(resolver.global_decisions),
            observation_digest=digest(resolver.global_decisions),
            feature_min=min((o['feature'] for o in resolver.global_decisions), default=0),
            feature_max=max((o['feature'] for o in resolver.global_decisions), default=0),
            rng_state_sha256=digest(resolver.rng.get_state().tolist()))
    return data, probe.rows, flows, resolver
