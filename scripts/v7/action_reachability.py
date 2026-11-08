"""Exact two-crane greedy reachability under fixed priority tiers.

Scores may vary independently within a tier. This is an upper bound on a shared
neural scorer: impossible here means impossible for that scorer too. A witness
does not establish that a particular finite neural network can realize it.
"""


def reachable(items,target,compatible):
    cranes={r['crane'] for r in items}
    if len(cranes)>2:raise ValueError('proof is limited to at most two deciding cranes')
    singles=[i for i,r in enumerate(items) if r['single_ok']]
    if not singles:return dict(reachable=False,firsts=[],certificate=[],reason='NO_FEASIBLE_FIRST')
    earliest=min(tuple(items[i]['tier']) for i in singles)
    firsts=[i for i in singles if tuple(items[i]['tier'])==earliest]
    t=items[target];certificate=[]
    for first in firsts:
        f=items[first]
        if first==target:
            return dict(reachable=True,firsts=firsts,witness=[target],rejected_prefix=[],certificate=certificate)
        if t['tier']<f['tier']:
            certificate.append(dict(first=first,reason='TARGET_ALREADY_REJECTED'));continue
        if f['crane']==t['crane']:
            certificate.append(dict(first=first,reason='CRANE_ALREADY_ASSIGNED'));continue
        if t['token'] is not None and t['token']==f['token']:
            certificate.append(dict(first=first,reason='TOKEN_ALREADY_ASSIGNED'));continue
        if not compatible(first,target):
            certificate.append(dict(first=first,reason='JOINT_CONFLICT'));continue
        blocked=None
        for i,r in enumerate(items):
            if r['crane']!=t['crane'] or tuple(r['tier'])>=tuple(t['tier']):continue
            # Earlier tiers have already been rejected. In the first tier,
            # individually infeasible pairs can be placed before the first accept.
            if r['tier']<f['tier'] or (r['tier']==f['tier'] and not r['single_ok']):continue
            if r['token'] is not None and r['token']==f['token']:continue
            if compatible(first,i):
                blocked=i;break
        if blocked is None:
            prefix=[i for i,r in enumerate(items) if i!=target and r['tier']==f['tier'] and not r['single_ok']]
            return dict(reachable=True,firsts=firsts,witness=[first,target],rejected_prefix=prefix,certificate=certificate)
        certificate.append(dict(first=first,reason='HIGHER_TIER_REMAINS',blocker=blocked))
    return dict(reachable=False,firsts=firsts,certificate=certificate,reason='ALL_FIRST_CHOICES_BLOCKED')


def replay_certificate(items,target,result,compatible):
    """Validate saved proof coverage and every blocking reason independently."""
    feasible=[i for i,r in enumerate(items) if r['single_ok']]
    if not feasible:
        assert not result['reachable'] and result['firsts']==[]
        return
    firsts={i for i in feasible if items[i]['tier']==min(items[j]['tier'] for j in feasible)}
    assert set(result['firsts'])==firsts
    t=items[target]
    if result['reachable']:
        witness=result['witness'];f=witness[0]
        assert f in firsts
        if len(witness)==1:assert f==target;return
        assert witness==[f,target] and items[f]['crane']!=t['crane']
        assert t['token'] is None or t['token']!=items[f]['token']
        assert compatible(f,target)
        for i,r in enumerate(items):
            if r['crane']==t['crane'] and r['tier']<t['tier']:
                if r['tier']<items[f]['tier'] or (r['tier']==items[f]['tier'] and not r['single_ok']):continue
                assert (r['token'] is not None and r['token']==items[f]['token']) or not compatible(f,i)
        return
    assert {r['first'] for r in result['certificate']}==firsts
    for row in result['certificate']:
        f=items[row['first']]
        if row['reason']=='TARGET_ALREADY_REJECTED':assert t['tier']<f['tier'] and not t['single_ok']
        elif row['reason']=='CRANE_ALREADY_ASSIGNED':assert f['crane']==t['crane'] and row['first']!=target
        elif row['reason']=='TOKEN_ALREADY_ASSIGNED':assert t['token'] is not None and f['token']==t['token']
        elif row['reason']=='JOINT_CONFLICT':assert not compatible(row['first'],target)
        elif row['reason']=='HIGHER_TIER_REMAINS':
            b=items[row['blocker']]
            assert b['crane']==t['crane'] and b['tier']<t['tier']
            assert b['tier']>f['tier'] or (b['tier']==f['tier'] and b['single_ok'])
            assert b['token'] is None or b['token']!=f['token']
            assert compatible(row['first'],row['blocker'])
        else:raise AssertionError(row)
