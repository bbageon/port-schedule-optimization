"""Paired single-axis and complete-bundle input checks; no policy is executed."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from pathlib import Path
import time

from audit_day_shapes import check_stage
from confirmed_demand_support import (BASE,OUT,BA,PREREG,OLD_BANK,OLD_DEMAND,OLD_MINUTES,
    ARMS,LOADS,read,save,sha,make_cases)
from evaluate_visit_links import input_stats,rule_checks
from yard_rl.integrated.repro import repro_stamp
from yard_rl.v7.day_shape import demand_for_shape
from yard_rl.v7.evaluation_probe import digest
from yard_rl.v7.input_bundle import InputBundle
from yard_rl.v7.information_contract import notice_time
from yard_rl.v7.stage.orders import build_stage,orders_from_schedule
from yard_rl.v7.world.integrated.profiles import build_h21_profile

SCHEDULE_FIELDS = {'arrival_s','gate_in_s','reserved_s','lead_s','visit'}
PROFILE_FIELDS = dict(legacy=set(),curve={'hour_counts'},mix={'gate_out_share'},
    minutes={'within_hour'},links={'visit_link'},all={'hour_counts','gate_out_share','within_hour','visit_link'})


class Conditions:
    def __init__(self, root):
        self.root = Path(root)
        self.old_bank,self.new_bank = read(OLD_BANK),read(self.root/'bank.json')
        self.base = read(OLD_DEMAND)['profiles']['observed_visit_links']
        self.old_minutes = read(OLD_MINUTES/'multiscale-minute-calibration.json')['parameters']
        self.new_minutes = read(self.root/'confirmed/multiscale-minute-calibration.json')['parameters']
        self.new_links = read(self.root/'confirmed/visit-link.json')
        self.info = read(BA/'profile-confirmed.json')
        self.bundle = InputBundle(self.root/'bundle.json')

    def profile(self, case):
        split,index,arm = case['split'],case['index'],case['arm']
        if arm not in ARMS:raise ValueError('unknown input comparison arm')
        old,new = self.old_bank['splits'][split][index],self.new_bank['splits'][split][index]
        p = demand_for_shape(self.base,(new if arm in ('curve','all') else old)['hour_masses'],
            source=dict(task='YR-332-bb',split=split,index=index,arm=arm))
        p['gate_out_share'] = (new if arm in ('mix','all') else old)['gate_out_share']
        p['within_hour'] = dict(method='correlated_minutes',parameters=copy.deepcopy(
            self.new_minutes if arm in ('minutes','all') else self.old_minutes))
        if arm in ('links','all'):p['visit_link'] = copy.deepcopy(self.new_links)
        return p

    def one(self, case):
        p = self.profile(case)
        if case['arm'] == 'all':
            kwargs = self.bundle.case(case['split'],case['index'],case['load'])
            q = kwargs['demand_profile']
            assert kwargs['information_profile'] == self.info and kwargs['seed'] == case['seed']
            assert all(p[k] == q[k] for k in ('hour_counts','gate_out_share','visit_link'))
            assert p['within_hour']['parameters'] == q['within_hour']['parameters']
        else:
            kwargs = dict(load=case['load'],seed=case['seed'],information_profile=self.info,
                          demand_profile=p,initial_stock_mode='preserve')
        built = build_stage(profile=build_h21_profile(),**kwargs)
        audit, rules = check_stage(built,case['load']),rule_checks(built)
        assert not any(rules.values()),rules
        orders,records = orders_from_schedule(built)
        assert len(orders) == len(records) == case['load']
        assert all(0 <= notice_time(r) <= r['arrival_s'] for r in built['schedule'])
        assert all(r['eta_offset_s'] == self.info['flows'][r['flow']]['eta_offset_s'] for r in built['schedule'])
        result = dict(case=case,stage=audit,rules=rules,statistics=input_stats(built),
            scenarios_sha256=digest(built['scenarios']),vessels_sha256=digest(built['vessel_schedule']),
            visits_sha256=digest([(r['job_id'],r.get('visit')) for r in built['schedule']]),
            input_stock_total=sum(len(s.containers) for s in built['scenarios'].values()),
            information_profile_sha256=sha(BA/'profile-confirmed.json'),
            demand_sha256=digest(kwargs['demand_profile']),stock=built['initial_stock'])
        return result,built['schedule'],p


def compare(before, after):
    b,old,old_profile = before
    a,new,new_profile = after
    arm = a['case']['arm']
    differences = {k for k in old_profile if old_profile[k] != new_profile[k]}-{'source'}
    assert differences <= PROFILE_FIELDS[arm],(arm,differences)
    assert [r['job_id'] for r in old] == [r['job_id'] for r in new]
    changed = Counter()
    for x,y in zip(old,new):
        for k in set(x)|set(y):
            if x.get(k) != y.get(k):changed[k] += 1
    checks = dict(profile_single_axis=True,orders_preserved=a['stage']['orders'] == b['stage']['orders'])
    if arm in ('curve','minutes','links'):
        checks.update(unchanged_stock=a['scenarios_sha256'] == b['scenarios_sha256'],
            unchanged_vessels=a['vessels_sha256'] == b['vessels_sha256'],
            unchanged_order_attributes=set(changed) <= SCHEDULE_FIELDS)
    if arm in ('minutes','links'):
        checks['unchanged_hour_counts'] = a['stage']['hour_counts'] == b['stage']['hour_counts']
    if arm == 'minutes':
        checks['unchanged_visit_links'] = a['visits_sha256'] == b['visits_sha256']
    assert all(checks.values()),(arm,checks)
    n = len(old)
    shifts = [abs(x['arrival_s']-y['arrival_s']) for x,y in zip(old,new)]
    return dict(case=a['case'],checks=checks,changed_fields=dict(changed),changed_profile_fields=sorted(differences),
        visit_count_change=a['statistics']['visits']-b['statistics']['visits'],
        multi_order_share_change=a['statistics']['multi_order_share']-b['statistics']['multi_order_share'],
        mean_absolute_arrival_shift_s=sum(shifts)/n,max_absolute_arrival_shift_s=max(shifts),
        hourly_tvd=.5*sum(abs(x-y) for x,y in zip(a['stage']['hour_counts'],b['stage']['hour_counts']))/n,
        flow_count_changes={f:a['stage']['flow_counts'].get(f,0)-b['stage']['flow_counts'].get(f,0) for f in ('GATE_IN','GATE_OUT')})


def run(root,out):
    out = Path(out)
    if (out/'input-check.json').exists():raise FileExistsError('Use a new output directory')
    conditions = Conditions(root)
    cases = make_cases(conditions.new_bank)
    assert len(cases) == 102
    configuration = dict(cases=cases,arms=list(ARMS),loads=list(LOADS),repeated_cases=6,policy_execution=False,
        bundle_path=Path(root/'bundle.json').relative_to(BASE).as_posix(),bundle_sha256=sha(root/'bundle.json'))
    repro = repro_stamp(experiment='YR-332-bb-inputs',seeds=dict(input=sorted({r['seed'] for r in cases})),
        params=configuration,prereg=PREREG)
    records,pairs,repeats = [],[],[]
    started = time.perf_counter(); previous = None
    for case in cases:
        t0 = time.perf_counter()
        result = conditions.one(case)
        records.append(result[0])
        if case['arm'] == 'legacy':previous = result
        elif case['split'] == 'train' and case['index'] < 3:
            assert previous[0]['case']['load'] == case['load'] and previous[0]['case']['index'] == case['index']
            pairs.append(compare(previous,result))
        if case['split'] == 'train' and case['index'] == 0 and case['load'] == 3500:
            replay = conditions.one(case)
            repeats.append(dict(case=case,exact=result == replay))
            assert repeats[-1]['exact']
        save(out/'input-progress.json',dict(done=len(records),total=len(cases),last=case,seconds=time.perf_counter()-started))
        print(f"input {len(records)}/{len(cases)} {case['split']}:{case['index']} {case['load']} {case['arm']} {time.perf_counter()-t0:.1f}s",flush=True)
    assert len(pairs) == 45 and len(repeats) == 6 and all(r['exact'] for r in repeats)
    result = dict(status='PASS',configuration=configuration,repro=repro,records=records,pairs=pairs,repeats=repeats,
        cases=len(records),generated_orders=sum(r['case']['load'] for r in records),
        replay_orders=sum(r['case']['load'] for r in repeats),seconds=time.perf_counter()-started,
        interpretation='Input contracts and single-axis changes; no scheduling performance or learning claim')
    save(out/'input-check.json',result)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=OUT)
    p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();run(a.root.resolve(),a.out.resolve())
