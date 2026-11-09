"""Record the existing vessel transport timing contract, without recalibration."""
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from benchmark_default_eval import save
from six_step_learning import sha
from yard_rl.integrated.repro import repro_stamp
from yard_rl.v7.stage.orders import build_stage
from yard_rl.v7.world.integrated.profiles import build_h21_profile
from yard_rl.v7.world.integrated.scenario_gen import phys_min_completion_s
from yard_rl.v7.world.integrated.transfer import TransferFleet
from yard_rl.v7.world.integrated.vessel import VesselWorkType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tests/v7'))
from test_discharge_backpressure import discharge_fixture, finish, TracedSimulator

BASE = Path('outputs/reports/yr332_discharge_calibration')
OUT = BASE/'corrected'
PREREG = Path('docs/research/v7-discharge/discharge-bound-correction-prereg.md')
CASES = [(1, n, 0) for n in (1, 2, 3)] + [
    (ships, 8, delay) for ships in (1, 2) for delay in (0, 1200)]
CADENCE = 3600./27.5


class TimingSimulator(TracedSimulator):
    def reset(self):
        super().reset()
        self.timing = []

    def state(self):
        return dict(pipeline=dict(self._discharge_pipeline),
                    busy_until=list(self.transfer.busy_until),
                    pending=list(self.transfer.pending))

    def _handle(self, event):
        before = self.state()
        super()._handle(event)
        if event.kind_name in ('STS_MOVE', 'TRANSFER_ARRIVE', 'VESSEL_RELEASED',
                               'JOB_COMPLETED', 'EQUIPMENT_DOWN', 'EQUIPMENT_UP'):
            self.timing.append(dict(time_s=self.clock, kind=event.kind_name,
                                    subject=event.payload, before=before, after=self.state()))


def run(ships, moves, delay):
    h21 = build_h21_profile()
    profile, scenario = discharge_fixture(ships=ships, moves=moves, yard_delay=delay)
    profile = replace(profile, transfer=h21.transfer)
    for vessel in scenario.vessels:
        vessel.plan = replace(vessel.plan, sts_move_interval_s=CADENCE)
    sim = TimingSimulator(profile, scenario, yard_handover_cap=profile.transfer.n_units)
    finish(sim)
    sim.check_invariants()
    results = dict(config=dict(ships=ships, moves_per_ship=moves, yard_delay_s=delay,
                              fleet=asdict(profile.transfer), cadence_s=CADENCE),
        events=sim.timing, event_hash=sim.event_stream_hash(),
        jobs=[dict(id=j.job_id, status=j.status.value, start_s=j.service_start,
                   end_s=j.service_end) for j in sim.jobs.values()],
        vessels=[dict(id=v.vessel_id, actual_completion_s=v.truth.actual_completion_s,
            declared_lower_bound_s=phys_min_completion_s(profile, work=VesselWorkType.DISCHARGE,
                start_s=0., moves=moves, cadence_s=CADENCE),
            idle_s=v.sts_wait_accum_s) for v in sim.vessels.values()],
        peak_pipeline=sim.peak, remaining=sim.unfinished_backlog(),
        final_pipeline=sum(sim._discharge_pipeline.values()))
    assert results['remaining'] == results['final_pipeline'] == 0
    assert 0 < results['peak_pipeline'] <= profile.transfer.n_units
    return results


def compare_daily_inputs():
    before = json.loads((BASE/'stage-before.json').read_text(encoding='utf-8'))
    comparisons = []
    for previous in before['rows']:
        load, seed = previous['load'], previous['seed']
        built = build_stage(load=load, seed=seed, profile=build_h21_profile())
        plans, scenarios = [], {}
        for bid, sc in built['scenarios'].items():
            item = asdict(sc)
            item['meta'].pop('vessel_deadline_achievable', None)
            for vessel in item['vessels']:
                vessel['plan'].pop('phys_min_completion_s', None)
                plans.append(dict(block=bid, id=vessel['vessel_id'], plan=vessel['plan']))
            scenarios[bid] = item
        payload = dict(scenarios=scenarios, schedule=built['schedule'],
                       fleet=[asdict(v) for v in built['fleet']])
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        assert digest == previous['execution_inputs_sha256']
        assert plans == previous['plans']
        comparisons.append(dict(load=load, seed=seed, plans=len(plans),
                                execution_inputs_sha256=digest, identical=True))
    return comparisons


def main():
    assert not subprocess.check_output(
        ['git', 'status', '--short', '--untracked-files=no'], encoding='utf-8').strip()
    paths = [Path(__file__), PREREG, Path('tests/v7/test_discharge_backpressure.py'),
             Path('src/yard_rl/integrated/repro.py'),
             Path('configs/terminals/dgt_armg.yaml'),
             *Path('src/yard_rl/v7').rglob('*.py')]
    manifest = dict(task='YR-332-i', sources={p.as_posix():sha(p) for p in paths},
        repro=repro_stamp(experiment='YR-332-i-existing-contract', seeds=dict(fixture=[0]),
            params=dict(cases=CASES, repeat=CASES[-1], cadence_s=CADENCE,
                        profile='H21 transfer on single-crane causal fixture', training=False),
            prereg=PREREG.as_posix()))
    save(OUT/'timing-manifest.json', manifest)
    fleet = TransferFleet('probe', 'YT', n_units=1, move_time_s=470.)
    arrival = fleet.request(0., 'V0')
    release = fleet.busy_until[0]
    second_arrival = fleet.request(arrival, 'V1')
    fleet_probe = dict(first_request_s=0., first_arrival_s=arrival,
                      first_unit_release_s=release, second_arrival_s=second_arrival)
    assert arrival == release == 470. and second_arrival == 940.
    runs = [run(*case) for case in CASES]
    repeat = run(*CASES[-1])
    assert repeat == runs[-1]
    save(OUT/'timing-raw.json', dict(fleet_probe=fleet_probe, runs=runs, repeat=repeat))
    previous = json.loads((BASE/'timing-raw.json').read_text(encoding='utf-8'))
    for old, new in zip(previous['runs']+[previous['repeat']], runs+[repeat], strict=True):
        old, new = copy.deepcopy(old), copy.deepcopy(new)
        for result in (old, new):
            for vessel in result['vessels']:
                vessel.pop('declared_lower_bound_s')
        assert old == new, 'Timing correction changed physical execution'
    violations = [dict(case=r['config'], vessel=v) for r in runs for v in r['vessels']
                  if v['actual_completion_s'] < v['declared_lower_bound_s']-1e-8]
    arrivals = [e for r in runs for e in r['events'] if e['kind']=='TRANSFER_ARRIVE']
    releases = [e for r in runs for e in r['events'] if
                sum(e['after']['pipeline'].values()) < sum(e['before']['pipeline'].values())]
    assert all(e['after']['pipeline']==e['before']['pipeline'] for e in arrivals)
    assert all(e['kind']=='JOB_COMPLETED' for e in releases)
    summary = dict(task='YR-332-i', run_count=8, repeat_identical=True,
        fleet_probe=fleet_probe, arrival_count=len(arrivals), pipeline_release_count=len(releases),
        arrival_does_not_release_pipeline=True, pipeline_released_at_final_stack=True,
        lower_bound_counterexamples=violations, invariant_checks='PASS',
        calibration_status='UNCONFIRMED', parameters_changed=False,
        lower_bound_formula_changed=True, physical_trace_matches_before=True,
        daily_input_comparisons=compare_daily_inputs(),
        missing=['active fleet allocation', 'loaded and empty travel times',
                 'container handoff and vehicle release condition'],
        raw_sha256=sha(OUT/'timing-raw.json'))
    assert not violations, violations
    save(OUT/'timing-summary.json', summary)
    print(json.dumps(dict(runs=8, repeat_identical=True, arrivals=len(arrivals),
                         releases=len(releases), bound_counterexamples=len(violations)), indent=2))


if __name__ == '__main__':
    main()
