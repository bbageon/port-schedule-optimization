"""Frozen new dates for an offline information test, followed conditionally by PPO."""
from pathlib import Path

from value_stability_inputs import (FIT, OLD_BANK, MIX, MINUTES, BASE, INFO, SCALE,
                                   SEEDS, RUNTIME, LOADS, read, save)
from six_step_learning import sha
from yard_rl.v7.day_shape import build_bank, demand_for_shape
from yard_rl.v7.day_mix import add_daily_mix
from yard_rl.v7.evaluation_probe import digest

OUT = Path('outputs/reports/yr332_global_value_context')
PREREG = Path('docs/research/v7-discharge/global-value-context-prereg.md')
AY = Path('outputs/reports/yr332_value_stability')
ARMS = ('zero', 'global')


def checkpoint_path(seed):
    return AY/f'popart-{seed}'/'final.pt'


def prepare_bank():
    assert not (OUT/'bank.json').exists()
    old, fitted = read(OLD_BANK), read(FIT)
    bank = build_bank(fitted['fits'], seed=9964000, sizes=dict(train=13, selection=3, holdout=3),
                      basis=old['basis_hour_masses'], family=old['family'])
    bank = add_daily_mix(bank, read(MIX)['day_variation']['per_date'])
    bank['initial_stock_mode'] = 'preserve'
    prior = [old, read(AY/'bank.json')]
    old_hashes = {r['shape_sha256'] for b in prior for rows in b['splits'].values() for r in rows}
    hashes = [r['shape_sha256'] for rows in bank['splits'].values() for r in rows]
    assert len(set(hashes)) == len(hashes) and not (set(hashes) & old_hashes)
    for k, name in enumerate(('train', 'selection', 'holdout')):
        for i, row in enumerate(bank['splits'][name]):
            row['input_seed'] = 9964200 + k*100 + i
    save(OUT/'bank.json', bank)
    save(OUT/'split-audit.json', dict(new_curves=len(hashes), overlap_with_previous=0,
        previous_curves=len(old_hashes), calibration_shared=True,
        scope='New synthetic draws; independent dates only within the calibrated generator family'))


def demand(split, index):
    row = read(OUT/'bank.json')['splits'][split][index]
    p = demand_for_shape(read(BASE)['profiles']['observed_visit_links'], row['hour_masses'],
                         source=dict(experiment='YR-332-az', split=split, index=index))
    p['gate_out_share'] = row['gate_out_share']
    cal = read(MINUTES); assert cal['passed']
    p['within_hour'] = dict(method='correlated_minutes', parameters=cal['parameters'],
                            source=dict(calibration_sha256=sha(MINUTES), unit='physical visits'))
    return p


def input_case(split, index, load):
    row = read(OUT/'bank.json')['splits'][split][index]
    return dict(split=split, index=index, load=load, input_seed=row['input_seed'])


def training_cases():
    return [input_case('train', i, LOADS[i % 3]) for i in range(12)]


def validation_cases():
    return [input_case('selection', i, load) for i in range(3) for load in LOADS]


def holdout_cases():
    return [input_case('holdout', i, load) for i in range(3) for load in LOADS]


def case_name(case):
    return f'{case["split"]}-{case["index"]:02d}-{case["load"]}'


def preflight():
    from audit_day_shapes import check_stage
    from evaluate_visit_links import rule_checks
    from yard_rl.v7.stage.orders import build_stage
    from yard_rl.v7.world.integrated.profiles import build_h21_profile
    cases = training_cases() + validation_cases() + holdout_cases() + [input_case('train',12,3500)]
    rows = []
    for c in cases:
        built = build_stage(load=c['load'], seed=c['input_seed'], profile=build_h21_profile(),
            information_profile=read(INFO), demand_profile=demand(c['split'], c['index']),
            initial_stock_mode='preserve')
        rules = rule_checks(built); assert not any(rules.values()), rules
        rows.append(dict(case=c, audit=check_stage(built, c['load']), rules=rules,
            scenarios_sha256=digest(built['scenarios']), vessels_sha256=digest(built['vessel_schedule'])))
    save(OUT/'input-preflight.json', dict(passed=True, cases=rows, rejected=0))


if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'bank': prepare_bank()
    elif sys.argv[1] == 'preflight': preflight()
    else: raise ValueError(sys.argv[1])
