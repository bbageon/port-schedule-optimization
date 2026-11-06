"""Frozen-model FCFS evaluation with exact checks before reusing prior evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

from six_step_learning import gz_read, gz_save, load_net, sha

PRIOR = Path('outputs/reports/yr332_six_steps/learning')
OUT = Path('outputs/reports/yr332_arrival_eval')
PREREG = Path('docs/research/v7-discharge/arrival-order-prereg.md')
LOADS = (3500, 7500, 15000)
SEEDS = (9913501, 9913502, 9913503)
ARMS = ('initial', 'pointwise', 'pairwise', 'coverage', 'flow')
HASHES = ('candidates_sha256', 'choices_sha256', 'events_sha256', 'jobs_sha256',
          'records_sha256', 'result_sha256')
# Registration fixes these checks before inspecting FCFS outcomes.
CASES = [('rule_all', 0, n, 0) for n in LOADS] + [
    ('initial', 9913501, 3500, 0), ('pointwise', 9913502, 7500, 0),
    ('pairwise', 9913503, 15000, 0), ('coverage', 9913501, 7500, 0),
    ('flow', 9913503, 15000, 0),
] + [('fcfs_arrival', 0, n, 0) for n in LOADS] + [('fcfs_arrival', 0, 3500, 1)]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def git(*args):
    return subprocess.check_output(['git', *args], encoding='utf-8').strip()


def name(case):
    a, m, n, rep = case
    return f'{a}-{m}-{n}' + ('-repeat' if rep else '')


def path_for(case):
    return OUT/'evaluation'/(name(case)+'.json.gz')


def source_hashes():
    files = [Path(__file__), PREREG, *Path('src/yard_rl/v7').rglob('*.py')]
    return {p.as_posix(): sha(p) for p in files}


def prepare():
    assert not git('status', '--short', '--untracked-files=no'), 'commit before running'
    prior_manifest = json.loads((PRIOR/'campaign-stage-4.json').read_text(encoding='utf-8'))
    changed = [p for p, h in prior_manifest['sources'].items() if sha(p) != h]
    assert changed == ['src/yard_rl/v7/dispatch.py'], changed
    old = sorted((PRIOR/'evaluation').glob('*.json.gz'))
    assert len(old) == 51
    published_hashes = {}
    for stage in (4, 5, 6):
        report = json.loads((PRIOR/f'stage{stage}-summary.json').read_text(encoding='utf-8'))
        published_hashes.update(report['artifact_sha256'])
    old_hashes, models = {}, {}
    for p in old:
        h = sha(p)
        assert published_hashes[p.as_posix()] == h, p
        old_hashes[p.as_posix()] = h
        d = gz_read(p)
        assert d['parameters'] == dict(candidate_cap=12 if d['arm']=='rule_cap12' else None,
            shared_slots=False, existence_scan=True, teacher=False)
        if d['checkpoint']:
            ck = Path(d['checkpoint'])
            assert sha(ck) == d['checkpoint_sha256'], ck
            models[ck.as_posix()] = sha(ck)
    assert len(models) == 15
    return dict(task='YR-332-u', commit=git('rev-parse', 'HEAD'), dirty=False,
        sources=source_hashes(), prior_evaluation_sha256=old_hashes, model_sha256=models,
        prior_source_changes=changed, cases=CASES, completed=[], workers=2,
        configuration=dict(blocks=21, cranes=42, observation_s=86400, drain_s=7200,
            initial_fill=.65, loads=LOADS, input_seeds=(9913900,9913910,9913920),
            candidate_cap=None, decision_probe_shortcircuit=True,
            reuse_rehandle_slots=False, reallocation=False, training=False,
            threads_per_case=1, model_selection='all four methods, final checkpoints',
            cost_contract='unchanged v7 KRW four terms'),
        scope='bounded reevaluation; prior cases reused only after exact replay checks')


def evaluate(index):
    import torch
    from yard_rl.v7.evaluation_probe import EvaluationProbe
    from yard_rl.v7.stage.episode import run_episode
    from yard_rl.v7.stage.exec_config import DEFAULT_EXEC_CONFIG
    torch.set_num_threads(1)
    case = CASES[index]
    arm, model_seed, load, repeat = case
    is_rule = arm in ('rule_all', 'fcfs_arrival')
    model = None if is_rule else PRIOR/f'model-{model_seed}'/(
        'initial.pt' if arm=='initial' else f'{arm}-2.pt')
    net = None if model is None else load_net(model)
    dispatcher = 'FCFS_ARRIVAL' if arm=='fcfs_arrival' else 'SF_SPT' if is_rule else 'RL_CRANE'
    seed = 9913900 + 10*LOADS.index(load)
    probe = EvaluationProbe()
    start = time.perf_counter()
    r = run_episode(load=load, seed=seed, dispatcher=dispatcher, crane_net=net,
        candidate_cap=None, evaluation_probe=probe, reuse_rehandle_slots=False,
        decision_probe_shortcircuit=True)
    assert r.policy_exceptions == 0 and r.rollout_worlds == 0
    assert r.traded_edges == 0 and r.n_space == 0 and r.n_time == 0
    d = dict(arm=arm, model_seed=None if is_rule else model_seed, load=load, seed=seed,
        repeat=repeat, dispatcher=dispatcher, python=sys.version, torch=torch.__version__,
        device='cpu', wall_seconds=time.perf_counter()-start,
        parameters=dict(candidate_cap=None, shared_slots=False, existence_scan=True, teacher=False),
        exec_config=asdict(DEFAULT_EXEC_CONFIG), result=r.as_dict(), probe=probe.result,
        checkpoint=None if model is None else model.as_posix(),
        checkpoint_sha256=None if model is None else sha(model))
    gz_save(path_for(case), d)
    print(f'{name(case)}: cost={r.phi_krw:.3f}; turn_s={r.breakdown["mean_turn_time_s"]:.3f}', flush=True)


def replay_equal(a, b):
    assert a['load']==b['load'] and a['seed']==b['seed']
    assert a['parameters']==b['parameters']
    assert a['checkpoint_sha256']==b['checkpoint_sha256']
    assert a['result']==b['result'], 'outcome differs'
    assert a['probe']['metadata']==b['probe']['metadata'], 'input differs'
    for key in HASHES + ('flow_status_counts',):
        assert a['probe'][key]==b['probe'][key], key


def campaign():
    marker = OUT/'campaign.json'
    stamp = prepare()
    if marker.exists():
        old = json.loads(marker.read_text(encoding='utf-8'))
        for k in ('sources','prior_evaluation_sha256','model_sha256','commit'):
            assert old[k]==stamp[k], f'{k} changed'
        stamp = old
    save(marker, stamp)
    def run(i):
        case = CASES[i]
        print('START '+name(case), flush=True)
        log = OUT/'logs'/(name(case)+'.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable, __file__, '--operation','one','--index',str(i)],
                stdout=f, stderr=subprocess.STDOUT, check=True)
        d = gz_read(path_for(case))
        if case[0] != 'fcfs_arrival':
            replay_equal(d, gz_read(PRIOR/'evaluation'/(name(case)+'.json.gz')))
        return i, d
    order = [8, 0] + [i for i in range(len(CASES)) if i not in (8, 0)]
    pending = iter(i for i in order if name(CASES[i]) not in stamp['completed'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = set()
        for _ in range(2):
            i = next(pending, None)
            if i is not None: futures.add(pool.submit(run,i))
        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for f in done:
                i, d = f.result()
                stamp['completed'].append(name(CASES[i]))
                stamp.setdefault('result_sha256',{})[path_for(CASES[i]).as_posix()] = sha(path_for(CASES[i]))
                save(marker,stamp)
                print(f'DONE {len(stamp["completed"])}/{len(CASES)} {name(CASES[i])} '
                      f'{d["wall_seconds"]:.1f}s',flush=True)
            for _ in done:
                i = next(pending,None)
                if i is not None: futures.add(pool.submit(run,i))
    assert source_hashes()==stamp['sources']
    verify()


def vessel_done(d, flow=None):
    return sum(n for k,n in d['probe']['flow_status_counts'].items()
        if k.endswith('/DONE') and (k.startswith(flow+'/') if flow else k.startswith('VESSEL_')))


def compare(a,b):
    ar,br = a['result'],b['result']
    return dict(cost_change_pct=100*(br['phi_krw']/ar['phi_krw']-1),
        turn_change_pct=100*(br['mean_turn_time_s']/ar['mean_turn_time_s']-1),
        turn_before_min=ar['mean_turn_time_s']/60, turn_after_min=br['mean_turn_time_s']/60,
        admitted_change=br['admitted']-ar['admitted'],
        censored_24h_change=br['n_censored']-ar['n_censored'],
        vessel_done_26h_change=vessel_done(b)-vessel_done(a),
        discharge_done_change=vessel_done(b,'VESSEL_DISCHARGE')-vessel_done(a,'VESSEL_DISCHARGE'),
        loading_done_change=vessel_done(b,'VESSEL_LOAD')-vessel_done(a,'VESSEL_LOAD'))


def verify():
    import statistics
    stamp = json.loads((OUT/'campaign.json').read_text(encoding='utf-8'))
    assert set(stamp['completed'])=={name(c) for c in CASES}
    assert source_hashes()==stamp['sources']
    for section in ('prior_evaluation_sha256','model_sha256','result_sha256'):
        for p,h in stamp[section].items():
            assert sha(p)==h, p
    current = {tuple(c):gz_read(path_for(c)) for c in CASES}
    for c,d in current.items():
        assert d['result']['policy_exceptions']==0
        if c[0]!='fcfs_arrival':
            replay_equal(d,gz_read(PRIOR/'evaluation'/(name(c)+'.json.gz')))
    replay_equal(current[('fcfs_arrival',0,3500,0)],current[('fcfs_arrival',0,3500,1)])
    records=[]
    for load in LOADS:
        fcfs=current[('fcfs_arrival',0,load,0)]
        spt=current[('rule_all',0,load,0)]
        assert fcfs['probe']['metadata']==spt['probe']['metadata']
        records.append(dict(arm='rule_all',model_seed=None,load=load,baseline='fcfs_arrival',**compare(fcfs,spt)))
        for arm in ARMS:
            for model_seed in SEEDS:
                p=PRIOR/'evaluation'/f'{arm}-{model_seed}-{load}.json.gz'
                d=gz_read(p)
                assert d['probe']['metadata']==fcfs['probe']['metadata'],p
                for base in (fcfs,spt):
                    records.append(dict(arm=arm,model_seed=model_seed,load=load,
                        baseline=base['arm'],**compare(base,d)))
    averages=[]
    for arm,baseline in sorted({(r['arm'],r['baseline']) for r in records}):
        all_rows=[r for r in records if r['arm']==arm and r['baseline']==baseline]
        for load in (None,*LOADS):
            rows=[r for r in all_rows if load is None or r['load']==load]
            row=dict(arm=arm,baseline=baseline,load=load,n=len(rows))
            row.update({k:statistics.mean(r[k] for r in rows) for k in rows[0]
                if k not in ('arm','model_seed','load','baseline')})
            row.update(cost_wins=sum(r['cost_change_pct']<0 for r in rows),
                turn_wins=sum(r['turn_change_pct']<0 for r in rows),
                admission_drop=sum(r['admitted_change']<0 for r in rows),
                unfinished_increase=sum(r['censored_24h_change']>0 for r in rows),
                vessel_drop=sum(r['vessel_done_26h_change']<0 for r in rows))
            averages.append(row)
    save(OUT/'summary.json',dict(comparisons=records,averages=averages,
        fresh_evaluations=12,exact_prior_replays=8,exact_fcfs_repeats=1,
        prior_evaluations_verified=51,models_verified=15,independent_input_days=3,
        learning_methods=4,trained_models=12,retraining=False,
        limitations=['same three previously examined days',
          'stored policies trained for three updates; no new convergence claim',
          'all 51 prior artifacts verified; eight exact trajectory replay checks',
          'cost/physics and admission accounting limitations unchanged',
          'not an observed Busan terminal operating policy']))
    print('VERIFIED: 8 exact prior replays; FCFS repeat exact; 51 prior artifacts, 15 models',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operation',choices=('campaign','one','verify'),default='campaign')
    p.add_argument('--index',type=int)
    args=p.parse_args()
    if args.operation=='one': evaluate(args.index)
    elif args.operation=='verify': verify()
    else: campaign()
