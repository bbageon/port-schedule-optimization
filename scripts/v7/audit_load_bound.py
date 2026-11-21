"""YR-332-i: check finite-fleet LOAD bounds against executable event traces."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from yard_rl.integrated.repro import repro_stamp
from yard_rl.v7.world.domain.enums import ContainerSize, JobFlow, LoadStatus
from yard_rl.v7.world.domain.models import Container, Job
from yard_rl.v7.world.integrated.fixtures import build_integrated_profile
from yard_rl.v7.world.integrated.scenario import InjectedEvent, TerminalScenario
from yard_rl.v7.world.integrated.scenario_gen import phys_min_completion_s
from yard_rl.v7.world.integrated.vessel import VesselPlan, VesselProcess, VesselWorkType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/v7'))
from test_discharge_backpressure import finish
from audit_transport_calibration import TimingSimulator

PREREG = Path('docs/research/v7-discharge/load-bound-check-prereg.md')
CADENCE = 3600. / 27.5
CASES = [dict(units=u, moves=m) for u in (1, 3) for m in (1, 2, 3, 4, 8, 32)]
CASES += [dict(units=u, moves=8, ships=2, delay=d) for u in (1, 3) for d in (0, 1200)]
CASES += [dict(units=3, moves=m, start=600.) for m in (1, 3)]


def fixture(*, units=3, moves=2, ships=1, delay=0, start=0.):
    profile = build_integrated_profile()
    profile = replace(profile, cranes=(profile.cranes[0],),
                      transfer=replace(profile.transfer, n_units=units, move_time_s=470.))
    containers, jobs, vessels = {}, [], []
    for i in range(ships):
        vid = f'V{i}'
        vessels.append(VesselProcess(vid, VesselWorkType.LOAD,
            VesselPlan(start, 30000., None, 31000., moves, CADENCE)))
        for j in range(moves):
            cid = f'{vid}-C{j:03}'
            containers[cid] = Container(cid, ContainerSize.FT40, LoadStatus.FULL,
                                        profile.block.block_id, i * moves + j + 1, 1, 1)
            jobs.append(Job(f'{vid}-{j:03}', JobFlow.VESSEL_LOAD,
                            start + j * CADENCE, None, None,
                            target_container=cid, vessel_id=vid))
    events = [] if not delay else [InjectedEvent(0., 'EQUIPMENT_DOWN', 'YC-A'),
                                   InjectedEvent(float(delay), 'EQUIPMENT_UP', 'YC-A')]
    return profile, TerminalScenario('load-bound-probe', 0, 32000., 0.,
                                      containers, jobs, vessels, events)


def run_case(**config):
    profile, scenario = fixture(**config)
    sim = TimingSimulator(profile, scenario, yard_handover_cap=profile.transfer.n_units)
    finish(sim)
    sim.check_invariants()
    assert sim.unfinished_backlog() == sim.transfer.waiting_count() == 0
    vessels = []
    for vid, vessel in sim.vessels.items():
        targets = [scenario.containers[j.target_container] for j in scenario.jobs if j.vessel_id == vid]
        args = dict(work=VesselWorkType.LOAD, start_s=vessel.plan.planned_start_s,
                    moves=vessel.plan.total_moves, cadence_s=CADENCE)
        actual = vessel.truth.actual_completion_s
        assert actual is not None
        vessels.append(dict(id=vid, actual_completion_s=actual,
            with_targets_s=phys_min_completion_s(profile, load_targets=targets, **args),
            without_targets_s=phys_min_completion_s(profile, **args)))
    return dict(config=config, profile=asdict(profile), events=sim.timing,
        event_hash=sim.event_stream_hash(), vessels=vessels, cost_raw=sim.cost.episode_raw(),
        jobs=[dict(id=j.job_id, status=j.status.value, start_s=j.service_start, end_s=j.service_end)
              for j in sim.jobs.values()], remaining=sim.unfinished_backlog())


def physical(run):
    value = json.loads(json.dumps(run))
    for vessel in value['vessels']:
        vessel.pop('with_targets_s')
        vessel.pop('without_targets_s')
    return value


def input_snapshot(path):
    from audit_day_shapes import check_stage
    from global_policy_inputs import bundle, holdout_cases
    from yard_rl.v7.evaluation_probe import digest
    from yard_rl.v7.stage.orders import build_stage
    from yard_rl.v7.world.integrated.profiles import build_h21_profile
    reference_path = Path('outputs/reports/yr332_global_policy/input-preflight.json')
    refs = {tuple(r['case'][k] for k in ('split', 'index', 'load')): r
            for r in json.loads(reference_path.read_text(encoding='utf-8'))['cases']}
    rows = []
    for case in holdout_cases():
        built = build_stage(profile=build_h21_profile(),
                            **bundle().case(case['split'], case['index'], case['load']))
        ref = refs[tuple(case[k] for k in ('split', 'index', 'load'))]
        scenarios = {key: asdict(sc) for key, sc in built['scenarios'].items()}
        bounds = []
        for bid, sc in scenarios.items():
            for vessel in sc['vessels']:
                bounds.append(dict(block=bid, id=vessel['vessel_id'],
                    value=vessel['plan'].pop('phys_min_completion_s')))
        row = dict(case=case, audit=check_stage(built, case['load']),
            scenarios_sha256=digest(built['scenarios']), execution_scenarios_sha256=digest(scenarios),
            schedule_sha256=digest(built['schedule']), vessels_sha256=digest(built['vessel_schedule']),
            bounds=bounds, reference=ref)
        assert row['audit'] == ref['audit']
        rows.append(row)
    path.write_text(json.dumps(dict(rows=rows, source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'], text=True).strip()), ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--before', type=Path)
    parser.add_argument('--inputs-out', type=Path)
    args = parser.parse_args()
    assert not subprocess.check_output(['git', 'status', '--short', '--untracked-files=no'],
                                       encoding='utf-8').strip()
    sources = [Path(__file__), PREREG, Path('scripts/v7/audit_transport_calibration.py'),
               Path('tests/v7/test_discharge_backpressure.py'),
               *Path('src/yard_rl/v7').rglob('*.py')]
    result = dict(task='YR-332-i', sources={p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in sources},
        repro=repro_stamp(experiment='YR-332-i-load-bound', seeds=dict(fixture=[0]),
            params=dict(cases=CASES, repeat=CASES[-1], cadence_s=CADENCE,
                        horizon_s=32000., transfer_s=470., training=False), prereg=str(PREREG)))
    runs = [run_case(**c) for c in CASES]
    repeat = run_case(**CASES[-1])
    assert repeat == runs[-1]
    violations = [dict(case=i, vessel=v['id'], mode=mode, actual_s=v['actual_completion_s'],
                       bound_s=v[mode], excess_s=v[mode]-v['actual_completion_s'])
                  for i, r in enumerate(runs) for v in r['vessels']
                  for mode in ('with_targets_s', 'without_targets_s')
                  if v[mode] > v['actual_completion_s'] + 1e-8]
    result.update(runs=runs, repeat=repeat, violations=violations, repeat_identical=True)
    if args.before:
        previous = json.loads(args.before.read_text(encoding='utf-8'))
        assert [physical(r) for r in runs] == [physical(r) for r in previous['runs']]
        result['physical_traces_unchanged'] = True
        assert not violations, violations
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')
    if args.inputs_out:
        input_snapshot(args.inputs_out)
    print(json.dumps(dict(runs=len(runs)+1, violations=violations, repeat_identical=True)))


if __name__ == '__main__':
    main()
