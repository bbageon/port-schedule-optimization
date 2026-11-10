"""Fixed-budget completion-objective training and classical-rule evaluation."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from six_step_learning import checkpoint, gz_read, gz_save, labelset, load_net, sha
from benchmark_default_eval import save

OUT = Path('outputs/reports/yr332_completion_learning')
PROFILE = Path('outputs/reports/yr332_real_order_information/profile.json')
PREREG = Path('.claude/docs/dashboard-task-specs/YR-332-ag-job-completion-learning-evaluation.md')
SEEDS = (9917101, 9917102, 9917103)
LOADS = (3500, 7500, 15000)
RULES = ('SF_SPT','FIFO','LIFO','SPT','NEAREST','LWKR','RANDOM')
RUNTIME = dict(candidate_cap=None, reuse_rehandle_slots=True,
               decision_probe_shortcircuit=True, preserve_external_demand=True)
CASES = [(rule,0,load,day,0) for rule in RULES for load in LOADS for day in range(3)]
CASES += [(arm,seed,load,day,0) for arm in ('initial','trained')
          for seed in SEEDS for load in LOADS for day in range(3)]
CASES += [('SF_SPT',0,7500,0,1),('trained',SEEDS[0],7500,0,1)]


def profile():
    return json.loads(PROFILE.read_text(encoding='utf-8'))


def folder(seed):
    return OUT/f'model-{seed}'


def sources():
    paths = [Path(__file__), PREREG, PROFILE, *Path('src/yard_rl/v7').rglob('*.py'),
             *[Path('scripts/v7')/p for p in ('six_step_learning.py','horizon_metrics.py',
                  'audit_admission_ledger.py','verify_job_completion.py','benchmark_default_eval.py')]]
    return {p.as_posix(): sha(p) for p in paths}


def check():
    m=json.loads((OUT/'campaign.json').read_text(encoding='utf-8'))
    for p,h in m['sources'].items():
        assert sha(p)==h, ('campaign source changed',p)


def prepare():
    from yard_rl.integrated.repro import repro_stamp
    assert not subprocess.check_output(['git','status','--short','--untracked-files=no']).strip()
    config=dict(model_seeds=SEEDS, loads=LOADS, updates=6, pairs_per_update=24,
        optimizer_steps_per_update=50, horizon_s=10800, behavior='frozen initial policy',
        rule_dispatchers=RULES, cases=CASES, runtime=RUNTIME, profile_sha256=sha(PROFILE),
        training_seed_formula='9917200 + 100*model_index + 10*iteration',
        evaluation_seed_formula='9917600 + 100*day + 10*load_index',
        primary_seconds=86400, engine_seconds=93600, alternative_contract='reachable',
        claim_scope='user-authorized diagnostic on real-data-calibrated information and synthetic physics',
        baseline='SF_SPT', decision='deterministic argmax through unchanged feasible resolver')
    assert not (OUT/'campaign.json').exists()
    save(OUT/'campaign.json',dict(configuration=config,sources=sources(),
        repro=repro_stamp(experiment='YR-332-ag', seeds=dict(models=list(SEEDS),
            training=[9917200+100*m+10*i for m in range(3) for i in range(6)],
            evaluation=[9917600+100*d+10*l for d in range(3) for l in range(3)]),
            params=config,prereg=PREREG.as_posix())))


def train(seed):
    import torch
    from yard_rl.v7.crane.policy import CraneNet, add_store_input
    from yard_rl.v7.crane.fit import CraneTrainer
    from yard_rl.v7.stage.episode import run_episode
    from yard_rl.v7.stage.rollout import RolloutBudget
    torch.set_num_threads(1);check();torch.manual_seed(seed)
    behavior=add_store_input(CraneNet(feature_mode='feasible'))
    net=copy.deepcopy(behavior);trainer=CraneTrainer(net,loss_kind='pairwise')
    initial=folder(seed)/'initial.pt'
    assert not initial.exists()
    checkpoint(initial,behavior,information_profile_sha256=sha(PROFILE),model_seed=seed)
    rows=[]
    for it,load in enumerate(LOADS*2):
        start=time.perf_counter();input_seed=9917200+100*SEEDS.index(seed)+10*it
        ep=run_episode(load=load,seed=input_seed,dispatcher='RL_CRANE',crane_net=behavior,
            information_profile=profile(),crane_budget=RolloutBudget(max_labels=24,identity_checks=0),
            horizon_s=10800.,workers=2,crane_sampling='coverage',crane_alternative_contract='reachable',**RUNTIME)
        data=dict(model_seed=seed,iteration=it,input_seed=input_seed,load=load,
            result=ep.as_dict(),samples=[asdict(s) for s in ep.crane_labels],seconds=time.perf_counter()-start)
        path=folder(seed)/f'labels-{it}.json.gz';gz_save(path,data)
        c=ep.crane_stats
        assert ep.policy_exceptions==c['hook_errors']==c['factual_mismatch']==c['force_failed']==0
        assert ep.admitted==load and len(ep.crane_labels)>=4
        assert c['sampled']==c['no_witness']+c['labeled']
        fit=trainer.fit(labelset(data),seed=input_seed);assert fit.steps==50
        cp=folder(seed)/f'final-{it}.pt'
        checkpoint(cp,net,trainer,model_seed=seed,iteration=it,corpus_sha256=sha(path),
                   information_profile_sha256=sha(PROFILE))
        rows.append(dict(iteration=it,load=load,input_seed=input_seed,corpus=path.as_posix(),
            corpus_sha256=sha(path),checkpoint=cp.as_posix(),checkpoint_sha256=sha(cp),
            fit=asdict(fit),stats=c,seconds=time.perf_counter()-start))
        save(folder(seed)/'training.json',dict(rows=rows,model_config=net.model_config(),torch=torch.__version__))
        print(f'model={seed} update={it+1}/6 pairs={c["labeled"]} seconds={rows[-1]["seconds"]:.1f}',flush=True)


def evaluate(index, smoke=False):
    import torch
    import audit_admission_ledger as ledger
    import yard_rl.v7.stage.episode as episode
    from horizon_metrics import FlowProbe, aggregate
    from verify_job_completion import raw_metrics, assert_metrics
    from yard_rl.v7.evaluation_probe import digest
    torch.set_num_threads(1)
    if not smoke:check()
    arm,seed,load,day,repeat=CASES[index]
    input_seed=9917600+100*day+10*LOADS.index(load)
    cp=folder(seed)/('initial.pt' if arm=='initial' else 'final-5.pt') if seed else None
    if smoke:load,input_seed=80,9917999
    original,ann=ledger.install_announcer();probe=FlowProbe(ann)
    start=time.perf_counter()
    try:
        ep=episode.run_episode(load=load,seed=input_seed,
            dispatcher='RL_CRANE' if seed else arm,crane_net=load_net(cp) if cp else None,
            information_profile=profile(),evaluation_probe=probe,**RUNTIME)
    finally:episode.V3Announcer=original
    result=ep.as_dict();metrics=raw_metrics(probe.rows);assert_metrics(metrics,result)
    flow_rows=copy.deepcopy(probe.flow_rows)
    for r in flow_rows:
        if r['flow'].startswith('GATE_'):r['finish_s']=r['yard_done_s']
    completion={f'at{h}':aggregate(flow_rows,h*3600.) for h in (24,26)}
    name='smoke' if smoke else '-'.join(map(str,CASES[index]))
    data=dict(case=CASES[index],input_seed=input_seed,profile_sha256=sha(PROFILE),
        result=result,probe=probe.result,metrics_by_flow=metrics,completion_by_flow=completion,
        orders_sha256=digest(probe.rows),checkpoint_sha256=sha(cp) if cp else None,
        seconds=time.perf_counter()-start)
    for sub,value in [('evaluation',data),('orders',probe.rows),('flow-orders',flow_rows)]:
        path=OUT/sub/(name+'.json.gz');assert not path.exists();gz_save(path,value)
    assert ep.admitted==load and ep.policy_exceptions==ep.rollout_worlds==ep.traded_edges==0
    assert probe.result['ledger_audit']['skipped']==0
    assert all(r['review_count']==1 for r in probe.rows)
    assert all(abs(r['attempt']['review_s']-r['notice_s'])<=1e-6 for r in probe.rows)
    print(f'eval {index} {name}: work={result["mean_gate_to_job_done_s"]/60:.3f}min {data["seconds"]:.1f}s',flush=True)


def smoke():
    import torch
    from yard_rl.v7.crane.policy import CraneNet,add_store_input
    from yard_rl.v7.crane.fit import CraneTrainer
    from yard_rl.v7.stage.episode import run_episode
    from yard_rl.v7.stage.rollout import RolloutBudget
    torch.set_num_threads(1);torch.manual_seed(9917999)
    net=add_store_input(CraneNet(feature_mode='feasible'))
    ep=run_episode(load=300,seed=9917999,dispatcher='RL_CRANE',crane_net=net,
        information_profile=profile(),crane_budget=RolloutBudget(max_labels=4,identity_checks=0),
        horizon_s=10800.,workers=1,crane_sampling='coverage',**RUNTIME)
    data=dict(result=ep.as_dict(),samples=[asdict(s) for s in ep.crane_labels])
    gz_save(OUT/'smoke-labels.json.gz',data)
    c=ep.crane_stats
    assert ep.policy_exceptions==c['hook_errors']==c['factual_mismatch']==c['force_failed']==0
    assert c['labeled']>0
    fit=CraneTrainer(net,loss_kind='pairwise').fit(labelset(data),seed=9917999)
    save(OUT/'smoke-fit.json',asdict(fit));evaluate(0,smoke=True)


def run_many(mode):
    values=SEEDS if mode=='train' else range(len(CASES))
    def run(value):
        log=OUT/'logs'/f'{mode}-{value}.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w',encoding='utf-8') as handle:
            p=subprocess.run([sys.executable,__file__,mode,str(value)],stdout=handle,stderr=subprocess.STDOUT)
        print(f'{mode} {value}: exit={p.returncode}',flush=True)
        if p.returncode:raise RuntimeError(log.read_text(encoding='utf-8'))
    with ThreadPoolExecutor(max_workers=3 if mode=='train' else 4) as pool:
        for job in as_completed([pool.submit(run,v) for v in values]):job.result()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','smoke','train','evaluate','train-all','evaluate-all'))
    parser.add_argument('value',nargs='?',type=int);args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='smoke':smoke()
    elif args.command.endswith('-all'):run_many(args.command.split('-')[0])
    elif args.command=='train':train(args.value)
    else:evaluate(args.value)
