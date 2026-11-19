"""Fixed bc budget and immutable source manifest; predictive error is reporting only."""
import subprocess
from pathlib import Path

from global_policy_inputs import (OUT,BB,PREREG,INFO,SEEDS,ARMS,SCALE,RUNTIME,LOADS,
    read,save,checkpoint_path,training_cases,development_cases,holdout_cases)
from six_step_learning import sha
from yard_rl.integrated.repro import repro_stamp


def model_path(arm, seed):return OUT/'policy'/f'{arm}-{seed}'


def eval_cases():
    rows=[]
    for seed in SEEDS:
        for c in development_cases()+holdout_cases():
            rows.append(dict(**c,arm='zero',seed=seed,stage='initial',mode='argmax'))
    for arm in ARMS:
        for seed in SEEDS:
            for stage in ('update-1','update-3','final'):
                for c in development_cases():rows.append(dict(**c,arm=arm,seed=seed,stage=stage,mode='argmax'))
            for c in holdout_cases():rows.append(dict(**c,arm=arm,seed=seed,stage='final',mode='argmax'))
    for c in holdout_cases():rows.append(dict(**c,arm='SF_SPT',seed=0,stage='rule',mode='argmax'))
    for arm,stage in [('zero','initial'),('zero','final'),('global','final')]:
        for seed in SEEDS:
            for c in development_cases():rows.append(dict(**c,arm=arm,seed=seed,stage=stage,mode='stochastic'))
    assert len(rows)==180
    return rows


def repeat_indices():
    return [next(i for i,c in enumerate(eval_cases()) if c['arm']==arm and c['stage']=='final'
        and c['seed']==SEEDS[0] and c['split']=='holdout' and c['index']==0 and c['load']==7500) for arm in ARMS]


def prepare():
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no']).strip()
    assert not (OUT/'campaign.json').exists()
    assert read(OUT/'input-preflight.json')['passed'] and read(OUT/'smoke.json')['passed']
    tracked=subprocess.check_output(['git','ls-files','-z','src/yard_rl/v7','src/yard_rl/integrated/repro.py','scripts/v7']).decode('utf-8')
    paths=[Path(p) for p in tracked.split('\0') if p.endswith('.py')]
    paths += [PREREG,INFO,BB/'bundle.json',BB/'demand-base.json',BB/'confirmed/curve-fit.json',BB/'confirmed/daily.json',
              OUT/'bank.json',OUT/'bundle.json',OUT/'split-audit.json',*[checkpoint_path(s) for s in SEEDS]]
    cfg=dict(task='YR-332-bc',arms=list(ARMS),model_seeds=list(SEEDS),train=training_cases(),
        development=development_cases(),holdout=holdout_cases(),evaluation_cases=eval_cases(),repeat_indices=repeat_indices(),
        updates=6,days_per_update=2,max_samples=2048,epochs=4,batch_size=64,lr=.0003,
        clip=.2,value_coefficient=.5,entropy_coefficient=.01,gradient_clip=1.,return_scale=SCALE,discount=1.,
        observation_s=86400,engine_s=93600,runtime=RUNTIME,workers=6,torch_threads=1,
        prediction_error_mode='report_only',checkpoint_selection='last update only; no adaptive stopping or selection',
        initialization='paired ay final weights and PopArt moments; expanded zero column; fresh Adam',
        seed_bases=dict(bank=10084000,action=10085000,stochastic_evaluation=10086000,
                        sample=10087000,update=10088000,smoke=10089000,bootstrap=10089900),
        uncertainty=dict(method='crossed day/model cluster bootstrap after load averaging',repeats=10000),
        authorization='Explicit user request on 2026-11-19; preserve realism rejection; bounded synthetic exploration')
    stamp=repro_stamp(experiment='YR-332-bc',seeds=dict(model=list(SEEDS),
        input=sorted({c['input_seed'] for c in cfg['train']+cfg['development']+cfg['holdout']}),
        action=[10085000+100*k+j for k in range(3) for j in range(12)],
        sampling=[10087000+100*k+i for k in range(3) for i in range(6)],
        updates=[10088000+100*k+i for k in range(3) for i in range(6)],
        stochastic_evaluation=[10086000+100*k+10*i+i for k in range(3) for i in range(3)],
        diagnostic_sampling=[10087000+100*k+10*i+i for k in range(3) for i in range(3)],
        curve_bank=[10084000],smoke_actions=[10089000],smoke_updates=[10089001],bootstrap=[10089900]),params=cfg,prereg=PREREG.as_posix())
    save(OUT/'campaign.json',dict(configuration=cfg,sources={p.as_posix():sha(p) for p in paths},repro=stamp,
        commit=stamp['code']['git_head'],clean_tracked_tree=True))


def check():
    obj=read(OUT/'campaign.json')
    for p,h in obj['sources'].items():assert sha(p)==h,('Frozen source changed',p)
    cfg=obj['configuration']
    assert cfg['prediction_error_mode']=='report_only'
    return cfg


if __name__=='__main__':prepare()
