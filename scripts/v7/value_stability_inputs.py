"""New draws from the unchanged synthetic family; no outcome-dependent filtering."""
from pathlib import Path
import json

from audit_day_shapes import BASE, INFO, check_stage
from completion_learning import LOADS, RUNTIME
from six_step_learning import sha
from yard_rl.v7.day_shape import build_bank, demand_for_shape
from yard_rl.v7.day_mix import add_daily_mix
from yard_rl.v7.evaluation_probe import digest

OUT = Path('outputs/reports/yr332_value_stability')
PREREG = Path('docs/research/v7-discharge/value-stability-prereg.md')
FIT = Path('outputs/reports/yr332_day_shapes/calibration-periodic_linear_24.json')
OLD_BANK = Path('outputs/reports/yr332_synthetic_inputs/bank.json')
MIX = Path('outputs/reports/yr332_day_flow_volume/measurement.json')
MINUTES = Path('outputs/reports/yr332_synthetic_inputs/multiscale-minute-calibration.json')
SEEDS = (9951101, 9951102, 9951103)
ARMS = ('control', 'popart')
SCALE = 368293633.7020551


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                    encoding='utf-8', newline='\n')


def prepare_bank():
    assert not (OUT/'bank.json').exists()
    old, fitted = read(OLD_BANK), read(FIT)
    bank = build_bank(fitted['fits'], seed=9954000, sizes=dict(train=13, selection=3, holdout=3),
                      basis=old['basis_hour_masses'], family=old['family'])
    bank = add_daily_mix(bank, read(MIX)['day_variation']['per_date'])
    bank['initial_stock_mode'] = 'preserve'
    hashes = [r['shape_sha256'] for rows in bank['splits'].values() for r in rows]
    old_hashes = {r['shape_sha256'] for rows in old['splits'].values() for r in rows}
    assert len(set(hashes)) == len(hashes) and not (set(hashes) & old_hashes)
    for k, rows in enumerate(bank['splits'].values()):
        for i, row in enumerate(rows):
            row['input_seed'] = 9954200 + k*100 + i
    save(OUT/'bank.json', bank)
    save(OUT/'split-audit.json', dict(new_shapes=len(hashes), overlap_with_previous=0,
        development_calibration_shared=True, old_bank_sha256=sha(OLD_BANK),
        scope='new synthetic draws only; not independent real dates or terminals'))


def demand(split, index):
    row = read(OUT/'bank.json')['splits'][split][index]
    p = demand_for_shape(read(BASE)['profiles']['observed_visit_links'], row['hour_masses'],
                         source=dict(experiment='YR-332-ay', split=split, index=index))
    p['gate_out_share'] = row['gate_out_share']
    cal = read(MINUTES)
    assert cal['passed']
    p['within_hour'] = dict(method='correlated_minutes', parameters=cal['parameters'],
                            source=dict(calibration_sha256=sha(MINUTES), unit='physical visits'))
    return p


def input_case(split, index, load):
    row = read(OUT/'bank.json')['splits'][split][index]
    return dict(split=split, index=index, load=load, input_seed=row['input_seed'])


def trajectories():
    return [input_case('train', i, LOADS[i % 3]) for i in range(12)]


def preflight():
    from evaluate_visit_links import rule_checks
    from yard_rl.v7.stage.orders import build_stage
    from yard_rl.v7.world.integrated.profiles import build_h21_profile
    cases = trajectories() + [input_case('train', 12, 3500)]
    cases += [input_case('selection', i, load) for i, load in enumerate(LOADS)]
    cases += [input_case('holdout', i, load) for i in range(3) for load in LOADS]
    rows = []
    for case in cases:
        built = build_stage(load=case['load'], seed=case['input_seed'], profile=build_h21_profile(),
            information_profile=read(INFO), demand_profile=demand(case['split'], case['index']),
            initial_stock_mode='preserve')
        rules = rule_checks(built)
        assert not any(rules.values()), rules
        rows.append(dict(case=case, audit=check_stage(built, case['load']), rules=rules,
                         scenarios_sha256=digest(built['scenarios']), vessels_sha256=digest(built['vessel_schedule'])))
    save(OUT/'input-preflight.json', dict(passed=True, cases=rows, rejected=0,
         scope='input conservation and visit constraints, not external realism validation'))


if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'bank': prepare_bank()
    elif sys.argv[1] == 'preflight': preflight()
    else: raise ValueError(sys.argv[1])
