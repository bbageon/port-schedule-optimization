from pathlib import Path
from itertools import permutations,product
import random

import pytest


@pytest.fixture
def reach(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts/v7'))
    import action_reachability
    return action_reachability


def item(crane,tier,token,single_ok=True):
    return dict(crane=crane,tier=list(tier),token=token,single_ok=single_ok)


def brute(items,compatible):
    tiers=sorted({tuple(r['tier']) for r in items})
    groups=[[i for i,r in enumerate(items) if tuple(r['tier'])==tier] for tier in tiers]
    found=set()
    for ordered in product(*(permutations(group) for group in groups)):
        selected=[];cranes=set();tokens=set()
        for i in [i for group in ordered for i in group]:
            r=items[i]
            if r['crane'] in cranes or (r['token'] is not None and r['token'] in tokens):continue
            ok=r['single_ok'] if not selected else compatible(selected[0],i)
            if ok:
                selected.append(i);cranes.add(r['crane'])
                if r['token'] is not None:tokens.add(r['token'])
        found.update(selected)
    return found


def test_prepare_cannot_overtake_available_service_on_one_crane(reach):
    items=[item('a',(1,0),'service'),item('a',(1,1),'prepare')]
    result=reach.reachable(items,1,lambda a,b:True)
    assert not result['reachable']
    reach.replay_certificate(items,1,result,lambda a,b:True)


def test_other_crane_can_block_service_and_make_prepare_reachable(reach):
    items=[item('a',(1,0),'service'),item('a',(1,1),'prepare'),item('b',(1,0),'other')]
    compatible=lambda a,b:{a,b}!={0,2}
    result=reach.reachable(items,1,compatible)
    assert result['reachable'] and result['witness']==[2,1]
    reach.replay_certificate(items,1,result,compatible)


def test_same_job_cannot_be_used_by_two_cranes(reach):
    items=[item('a',(1,1),'same'),item('b',(0,0),'same')]
    assert not reach.reachable(items,0,lambda a,b:True)['reachable']


def test_exact_search_matches_all_permutations_on_small_random_systems(reach):
    rng=random.Random(9914500)
    for _ in range(120):
        items=[item(rng.choice(('a','b')),rng.choice(((0,0),(1,0),(1,1),(1,2))),
                    rng.choice(('j0','j1','j2',None)),rng.random()>.15) for _ in range(5)]
        pairs={(i,j):rng.random()>.35 for i in range(5) for j in range(i+1,5)}
        compatible=lambda a,b:pairs[tuple(sorted((a,b)))]
        expected=brute(items,compatible)
        for target in range(5):
            result=reach.reachable(items,target,compatible)
            assert result['reachable']==(target in expected)
            reach.replay_certificate(items,target,result,compatible)


def test_three_cranes_are_not_silently_classified(reach):
    with pytest.raises(ValueError):
        reach.reachable([item(c,(1,0),c) for c in 'abc'],0,lambda a,b:True)


def test_rejected_candidate_is_not_revisited_after_another_crane_accepts(reach):
    items=[item('a',(1,0),'service',False),item('a',(1,1),'prepare'),item('b',(1,0),'other')]
    result=reach.reachable(items,1,lambda a,b:True)
    assert result['reachable'] and result['rejected_prefix']==[0]
    reach.replay_certificate(items,1,result,lambda a,b:True)


def test_real_resolver_witness_force_and_feature_ties_are_kept_separate(reach):
    import torch
    from test_candidate_recall import state,generate
    from yard_rl.v7.crane.policy import CraneNet,CranePolicy
    from yard_rl.v7.crane.features import crane_features
    from yard_rl.v7.world.integrated.baselines import ResolverPolicy
    import audit_action_contract as audit
    torch.set_num_threads(1);torch.manual_seed(9914502)
    sim,dp=state();gb=generate(sim,dp,None)
    pref=CranePolicy(CraneNet(feature_mode='feasible'))
    assignment=ResolverPolicy(pref).decide(sim,dp,gb);cid=dp.crane_ids[0]
    picked=assignment[cid]
    alt=next(g for g in gb[cid].items if g.feasible and g.job_ref and g.job_ref.job_id!=picked.job_ref.job_id)
    samples=[]
    for action,g,phi in (('PICKED',picked,100.),('ALT',alt,90.)):
        row=crane_features(sim,cid,g,feature_mode='feasible',include_flow=True)
        samples.append(dict(row=row[:9],is_store=bool(row[9]),decision_key=[round(sim.clock,6),'Y01',cid],
            job=g.job_ref.job_id,action=action,phi=phi,target=(phi-95.)/10.))
    probe=audit.ContractProbe(dict(samples=samples),dict(samples=samples))
    probe.block_names={id(sim):'Y01'}
    before=repr([(j.job_id,j.status,j.service_start,j.service_end) for j in sim.jobs.values()])
    probe.before_select(sim,dp,gb,pref)
    assert len(probe.rows)==2
    assert all(r['proof']['reachable'] and r['forced_selected'] for r in probe.rows)
    assert all(r['score_witnesses']['independent']['target_selected'] for r in probe.rows)
    assert before==repr([(j.job_id,j.status,j.service_start,j.service_end) for j in sim.jobs.values()])
    assert ResolverPolicy(pref).decide(sim,dp,gb)==assignment
