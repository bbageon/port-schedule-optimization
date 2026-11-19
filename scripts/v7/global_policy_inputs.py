"""YR-332-bc: frozen new synthetic inputs using the confirmed bb calibration."""
from pathlib import Path
from functools import lru_cache
import os

from confirmed_demand_support import read, save, bind
from global_value_inputs import checkpoint_path, SEEDS, SCALE, RUNTIME, ARMS
from six_step_learning import sha
from yard_rl.v7.day_shape import build_bank
from yard_rl.v7.day_mix import add_daily_mix
from yard_rl.v7.input_bundle import InputBundle, SCHEMA
from yard_rl.v7.evaluation_probe import digest

OUT = Path('outputs/reports/yr332_global_policy')
BB = Path('outputs/reports/yr332_confirmed_demand')
PREREG = Path('docs/research/v7-discharge/global-policy-prereg.md')
LOADS = (3500, 7500, 15000)
INFO = Path('outputs/reports/yr332_gate_time_provenance/profile-confirmed.json')


@lru_cache(maxsize=1)
def bundle():
    return InputBundle(OUT/'bundle.json')


def demand(split, index):
    return bundle().case(split, index, 3500)['demand_profile']


def input_case(split, index, load):
    kw = bundle().case(split, index, load)
    return dict(split=split, index=index, load=load, input_seed=kw['seed'])


def training_cases():
    return [input_case('train', i, LOADS[i % 3]) for i in range(12)]


def development_cases():
    return [input_case('selection', i, load) for i, load in enumerate(LOADS)]


def holdout_cases():
    return [input_case('holdout', i, load) for i in range(3) for load in LOADS]


def prepare_bank():
    assert not (OUT/'bank.json').exists()
    InputBundle(BB/'bundle.json')
    old = read(BB/'bank.json')
    bank = build_bank(read(BB/'confirmed/curve-fit.json')['fits'], seed=10084000,
        sizes=dict(train=13, selection=3, holdout=3), basis=old['basis_hour_masses'], family=old['family'])
    bank = add_daily_mix(bank, read(BB/'confirmed/daily.json')['per_date'])
    bank['initial_stock_mode'] = 'preserve'
    prior_paths = [BB/'bank.json', *[Path('outputs/reports')/n/'bank.json' for n in
        ('yr332_synthetic_inputs', 'yr332_value_stability', 'yr332_global_value_context', 'yr332_resched_ppo')]]
    prior = {r['shape_sha256'] for p in prior_paths if p.exists()
             for rows in read(p)['splits'].values() for r in rows}
    hashes = [r['shape_sha256'] for rows in bank['splits'].values() for r in rows]
    assert len(set(hashes)) == 19 and not (set(hashes) & prior)
    for k, split in enumerate(('train','selection','holdout')):
        for i, r in enumerate(bank['splits'][split]):r['input_seed'] = 10084100+100*k+i
    save(OUT/'bank.json', bank)
    def ref(p):return dict(path=Path(os.path.relpath(p, OUT)).as_posix(), sha256=sha(p))
    save(OUT/'bundle.json', dict(schema=SCHEMA, task='YR-332-bc',
        assets=dict(information=ref(INFO), bank=ref(OUT/'bank.json'), demand=ref(BB/'demand-base.json')),
        scope='New synthetic draws with common source-confirmed calibration; not unseen real dates'))
    save(OUT/'split-audit.json', dict(new_curves=19, previous_curves=len(prior), overlap=0,
        calibration_shared=True, previous_banks={p.as_posix():sha(p) for p in prior_paths if p.exists()}))


def preflight():
    from audit_day_shapes import check_stage
    from evaluate_visit_links import rule_checks
    from yard_rl.v7.stage.orders import build_stage
    from yard_rl.v7.world.integrated.profiles import build_h21_profile
    cases = training_cases()+[input_case('selection',i,l) for i in range(3) for l in LOADS]
    cases += holdout_cases()+[input_case('train',12,3500)]
    rows=[]
    for c in cases:
        built=build_stage(profile=build_h21_profile(), **bundle().case(c['split'],c['index'],c['load']))
        rules=rule_checks(built);assert not any(rules.values()),rules
        rows.append(dict(case=c,audit=check_stage(built,c['load']),rules=rules,
            scenarios_sha256=digest(built['scenarios']),vessels_sha256=digest(built['vessel_schedule'])))
    assert len(rows)==31
    save(OUT/'input-preflight.json',dict(passed=True,cases=rows,rejected=0,bundle_sha256=sha(OUT/'bundle.json')))


def rollout(case, net=None, *, mode='off', stochastic=False, collect=False, action_seed=0):
    import global_value_rollout as common
    with bind(common, INFO=INFO, demand=demand, RUNTIME=RUNTIME):
        data,orders,flows,resolver=common.rollout(case,net,mode=mode,stochastic=stochastic,
                                               collect=collect,action_seed=action_seed)
    if collect:data['cost_epochs']=resolver.costs
    return data,orders,flows,resolver


if __name__=='__main__':
    import sys
    {'bank':prepare_bank,'preflight':preflight}[sys.argv[1]]()
