"""Paired, frozen-rule validation of the all-reserved synthetic information contract."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import io
import inspect
import math
from pathlib import Path
import statistics
import subprocess
import time
import xml.etree.ElementTree as ET

import audit_admission_ledger as ledger
from benchmark_default_eval import save
from six_step_learning import gz_read, gz_save, sha
from yard_rl.integrated.repro import repro_stamp
from yard_rl.v7.evaluation_probe import digest

OUT = Path('outputs/reports/yr332_reservation_contract')
SPEC = Path('.claude/docs/dashboard-task-specs/YR-332-ad-all-reserved-synthetic-contract.md')
LOADS = (3500, 7500, 15000)
SEEDS = (9915101, 9915102, 9915103)
HASHES = ('candidates_sha256', 'choices_sha256', 'events_sha256',
          'jobs_sha256', 'records_sha256')


def stamp(config):
    dirty = subprocess.check_output(['git', 'status', '--short', '--untracked-files=no'],
                                    text=True, encoding='utf-8').strip()
    if dirty:
        raise RuntimeError('Commit tracked changes before collecting evidence: ' + dirty)
    paths = [Path(__file__).resolve().relative_to(Path.cwd()), SPEC,
             Path(ledger.__file__).resolve().relative_to(Path.cwd()),
             *sorted(Path('src/yard_rl/v7').rglob('*.py'))]
    return dict(repro=repro_stamp(experiment='YR-332-ad', seeds={'input': list(SEEDS)},
                                  params=config, prereg=SPEC.as_posix()),
                sources={p.as_posix(): sha(p) for p in paths})


class ReservationProbe(ledger.LedgerProbe):
    def finish(self, mbt, records, episode):
        super().finish(mbt, records, episode)
        entries = {e['job_id']: e for e in self.schedule}
        jobs = {jid: job for sim in mbt.blocks.values() for jid, job in sim.jobs.items()}
        for row in self.rows:
            e, j = entries[row['job_id']], jobs[row['job_id']]
            reserve = float(e.get('reserved_s', e['arrival_s']))
            assert math.isfinite(reserve)
            assert j.appointment_gate_time == reserve
            assert j.notified_gate_in_s == reserve
            assert j.provided_eta == reserve + e['travel_base_s']
            assert row['times']['gate_in_s'] == e['arrival_s']
            assert row['times']['block_in_s'] == e['arrival_s'] + e['travel_s']
            row.update(reserved_s=reserve, expected_block_s=j.provided_eta,
                       appointment_error_s=e['arrival_s'] - reserve)


def filename(mode, load, sigma=0, repeat=False):
    suffix = '-repeat' if repeat else ''
    return f'{mode}-{load}-sigma{sigma:g}{suffix}.json.gz'


def evaluate(mode, load, sigma, repeat):
    import torch
    import yard_rl.v7.stage.episode as episode
    if mode == 'baseline' and 'appointment_sigma_s' in inspect.signature(episode.run_episode).parameters:
        raise RuntimeError('Baseline must run at the pre-change commit abce89a5; '
                           'current defaults describe the new reservation contract')
    name = filename(mode, load, sigma, repeat)
    for target in (OUT / 'evaluation' / name, OUT / 'orders' / name):
        if target.exists():
            raise FileExistsError(f'Preserve existing evidence: {target}')
    torch.set_num_threads(1)
    config = dict(mode=mode, load=load, seed=SEEDS[0], appointment_sigma_s=sigma,
                  dispatcher='SF_SPT', candidate_cap=None, training=False,
                  official_end_s=86400, engine_end_s=93600, repeat=repeat)
    evidence = stamp(config)
    original, announcer = ledger.install_announcer()
    probe = ReservationProbe(announcer)
    kwargs = {} if mode == 'baseline' else {'appointment_sigma_s': sigma}
    start = time.perf_counter()
    try:
        result = episode.run_episode(load=load, seed=SEEDS[0], dispatcher='SF_SPT',
            candidate_cap=None, decision_probe_shortcircuit=True,
            evaluation_probe=probe, **kwargs)
    finally:
        episode.V3Announcer = original
    assert result.admitted == load
    assert result.policy_exceptions == result.traded_edges == result.rollout_worlds == 0
    assert probe.result['ledger_audit']['skipped'] == 0
    evidence.update(config=config, elapsed_s=time.perf_counter() - start,
                    result=result.as_dict(), probe=probe.result,
                    orders_sha256=digest(probe.rows))
    if mode == 'evaluate' and sigma == 0:
        old = gz_read(OUT / 'evaluation' / filename('baseline', load))
        normalized = copy.deepcopy(result.as_dict())
        normalized['environment'].pop('reservation_contract')
        assert normalized == old['result']
        assert all(probe.result[k] == old['probe'][k] for k in HASHES)
        assert probe.rows == gz_read(OUT / 'orders' / filename('baseline', load))
        evidence['zero_error_exact_replay'] = True
    if repeat:
        first = gz_read(OUT / 'evaluation' / filename(mode, load, sigma))
        assert evidence['result'] == first['result']
        assert evidence['orders_sha256'] == first['orders_sha256']
        assert all(probe.result[k] == first['probe'][k] for k in HASHES)
        evidence['exact_repeat'] = True
    gz_save(OUT / 'evaluation' / name, evidence)
    gz_save(OUT / 'orders' / name, probe.rows)
    print(f'{name}: admitted={result.admitted}, mean_turn_s='
          f'{result.breakdown["mean_turn_time_s"]:.6f}, cost={result.phi_krw:.6f}', flush=True)


def generate():
    from yard_rl.v7.stage.orders import build_stage, orders_from_schedule
    from yard_rl.v7.world.integrated.profiles import build_h21_profile
    if (OUT / 'generation.json').exists():
        raise FileExistsError('Preserve existing generation evidence; use a separate checkout')
    config = dict(loads=LOADS, seeds=SEEDS, sigmas_s=(0, 600, 1800),
                  actual_arrival_curve_preserved=True, all_orders_reserved=True)
    evidence, results = stamp(config), []
    profile = build_h21_profile()
    for load in LOADS:
        for seed in SEEDS:
            baseline = build_stage(load=load, seed=seed, profile=profile, appointment_sigma_s=0)
            original_schedule = baseline['schedule']
            physical = digest([{k: v for k, v in e.items()
                                if k not in ('reserved_s', 'reservation_schema')}
                               for e in original_schedule])
            for sigma in config['sigmas_s']:
                built = build_stage(load=load, seed=seed, profile=profile, appointment_sigma_s=sigma)
                again = build_stage(load=load, seed=seed, profile=profile, appointment_sigma_s=sigma)
                assert digest(built['schedule']) == digest(again['schedule'])
                assert physical == digest([{k: v for k, v in e.items()
                                            if k not in ('reserved_s', 'reservation_schema')}
                                           for e in built['schedule']])
                assert digest(built['vessel_schedule']) == digest(baseline['vessel_schedule'])
                assert digest(built['scenarios']) == digest(baseline['scenarios'])
                orders, records = orders_from_schedule(built)
                assert len(orders) == len(records) == load
                errors = [e['arrival_s'] - e['reserved_s'] for e in built['schedule']]
                assert all(math.isfinite(o.in_out_reserve_s) for o in orders.values())
                assert all(r.gate_in_s is None and r.block_in_s is None for r in records.values())
                assert all(abs(x) <= 2 * sigma + 1e-8 for x in errors)
                if sigma:
                    assert min(errors) < 0 < max(errors)
                else:
                    assert set(errors) == {0}
                results.append(dict(load=load, seed=seed, sigma_s=sigma, orders=len(orders),
                    schedule_sha256=digest(built['schedule']), physical_sha256=physical,
                    mean_error_s=statistics.mean(errors), mean_abs_error_s=statistics.mean(map(abs, errors)),
                    min_error_s=min(errors), max_error_s=max(errors),
                    reserved_before_window=sum(e['reserved_s'] < 0 for e in built['schedule']),
                    reserved_after_window=sum(e['reserved_s'] >= 86400 for e in built['schedule']),
                    reserved_before_notice=sum(e['reserved_s'] < e['arrival_s'] - e['lead_s']
                                               for e in built['schedule']),
                    equal_to_actual=sum(x == 0 for x in errors)))
    evidence.update(configuration=config, checks=results, passed=True,
                    unique_physical_orders=sum(LOADS) * len(SEEDS),
                    generated_orders=sum(r['orders'] for r in results))
    save(OUT / 'generation.json', evidence)
    print('generation: 27 conditions / 234000 order rows, repeated exactly', flush=True)


def verify_sources(artifacts):
    """Compare recorded bytes with each clean run's pushed commit, not current HEAD."""
    expected = {}
    for artifact in artifacts:
        code = artifact['repro']['code']
        assert code['git_dirty'] is False
        commit = code['git_head']
        subprocess.run(['git', 'merge-base', '--is-ancestor', commit,
                        'origin/강화학습-판매'], check=True, capture_output=True)
        for name, file_hash in artifact['sources'].items():
            path = Path(name)
            rel = path.resolve().relative_to(Path.cwd()).as_posix()
            key = f'{commit}:{rel}'
            assert key not in expected or expected[key] == file_hash
            expected[key] = file_hash
    keys = list(expected)
    process = subprocess.run(['git', 'cat-file', '--batch'],
        input=('\n'.join(keys) + '\n').encode('utf-8'), capture_output=True, check=True)
    stream = io.BytesIO(process.stdout)
    crlf = 0
    for key in keys:
        header = stream.readline().decode().split()
        assert len(header) == 3 and header[1] == 'blob', (key, header)
        value = stream.read(int(header[2]))
        assert stream.read(1) == b'\n'
        if hashlib.sha256(value).hexdigest() != expected[key]:
            # Git stores text as LF; the existing Windows checkout can use CRLF.
            assert hashlib.sha256(value.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest() == expected[key], key
            crlf += 1
    assert not stream.read()
    return dict(passed=True, commit_file_pairs=len(keys), windows_crlf_matches=crlf,
                commits=sorted({k.split(':', 1)[0] for k in keys}))


def verify_tests():
    """Later targeted reruns supersede the corrected fixture's initial failure."""
    outcomes, histories = {}, []
    for name in ('tests-contract.xml', 'tests-stage.xml', 'tests-compatibility.xml',
                 'tests-array-legacy.xml'):
        path = OUT / name
        root = ET.parse(path).getroot()
        for case in root.iter('testcase'):
            key = f"{case.attrib.get('classname')}::{case.attrib['name']}"
            status = 'failed' if case.find('failure') is not None or case.find('error') is not None else (
                'skipped' if case.find('skipped') is not None else 'passed')
            if status != 'passed':
                histories.append(dict(test=key, status=status, file=name))
            outcomes[key] = status
    assert all(v == 'passed' for v in outcomes.values()), outcomes
    return dict(unique_passed=len(outcomes), earlier_nonpassing=histories,
                excluded='v5 generate_fixed_seed CLI integration requires os.sched_getaffinity; '
                         'unavailable on Windows. Direct v7 fixed-seed serialization is covered.',
                files={name: sha(OUT / name) for name in (
                    'tests-contract.xml', 'tests-stage.xml', 'tests-compatibility.xml', 'tests-array-legacy.xml')})


def verify():
    import json
    generation = json.loads((OUT / 'generation.json').read_text(encoding='utf-8'))
    assert generation['passed']
    rows, paths, artifacts = [], [OUT / 'generation.json'], [generation]
    for load in LOADS:
        pair = {}
        for mode, sigma in (('baseline', 0), ('evaluate', 0), ('evaluate', 600)):
            p = OUT / 'evaluation' / filename(mode, load, sigma)
            data = gz_read(p)
            artifacts.append(data)
            q = OUT / 'orders' / p.name
            orders = gz_read(q)
            assert digest(orders) == data['orders_sha256']
            assert len(orders) == data['result']['admitted'] == load
            assert all(r['admitted'] and r['review_count'] == 1 for r in orders)
            assert all(r['times']['gate_in_s'] == r['scheduled_gate_s'] for r in orders)
            # Independent recomputation: reservation time never enters truck dwell.
            mean_s = sum((min(86400, r['times']['gate_out_s'])
                          if r['times']['gate_out_s'] is not None else 86400)
                         - r['times']['gate_in_s'] for r in orders) / load
            assert math.isclose(mean_s, data['result']['mean_turn_time_s'], rel_tol=1e-11)
            cost = sum(data['result'][k] for k in ('c_wait', 'c_move', 'c_rehandle', 'c_vessel'))
            assert math.isclose(cost, data['result']['phi_krw'], rel_tol=1e-12)
            censored = sum(r['times']['gate_out_s'] is None or r['times']['gate_out_s'] > 86400
                           for r in orders)
            assert censored == data['result']['n_censored']
            paths.extend((p, q))
            pair[f'{mode}-{sigma}'] = data
        assert pair['evaluate-0']['zero_error_exact_replay']
        normalized = copy.deepcopy(pair['evaluate-0']['result'])
        normalized['environment'].pop('reservation_contract')
        assert normalized == pair['baseline-0']['result']
        assert all(pair['evaluate-0']['probe'][k] == pair['baseline-0']['probe'][k] for k in HASHES)
        assert pair['evaluate-0']['orders_sha256'] == pair['baseline-0']['orders_sha256']
        before, after = pair['evaluate-0']['result'], pair['evaluate-600']['result']
        rows.append(dict(load=load, admitted=after['admitted'],
            mean_turn_min_before=before['mean_turn_time_s']/60,
            mean_turn_min_after=after['mean_turn_time_s']/60,
            relative_change_pct=100*(after['mean_turn_time_s'] / before['mean_turn_time_s']-1),
            cost_before=before['phi_krw'], cost_after=after['phi_krw'],
            censored_before=before['n_censored'], censored_after=after['n_censored']))
    repeat = OUT / 'evaluation' / filename('evaluate', 7500, 600, True)
    repeated = gz_read(repeat)
    first = gz_read(OUT / 'evaluation' / filename('evaluate', 7500, 600))
    assert repeated['exact_repeat'] and repeated['result'] == first['result']
    assert all(repeated['probe'][k] == first['probe'][k] for k in HASHES)
    assert repeated['orders_sha256'] == first['orders_sha256']
    artifacts.append(repeated)
    paths.extend((repeat, OUT / 'orders' / repeat.name))
    source_audit = verify_sources(artifacts)
    test_audit = verify_tests()
    save(OUT / 'verification.json', dict(passed=True, evaluation_runs=10,
        generation_conditions=27, unique_physical_orders=generation['unique_physical_orders'],
        comparisons=rows, source_check=source_audit, tests=test_audit,
        inputs={p.as_posix(): sha(p) for p in paths}, training=False,
        conclusion='Synthetic reservation contract verified; empirical calibration and policy benefit unproven.'))
    print(rows, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation', choices=('baseline', 'evaluate', 'generate', 'verify'), required=True)
    parser.add_argument('--load', type=int, choices=LOADS, default=7500)
    parser.add_argument('--sigma', type=float, default=600)
    parser.add_argument('--repeat', action='store_true')
    args = parser.parse_args()
    if args.operation == 'generate':
        generate()
    elif args.operation == 'verify':
        verify()
    else:
        evaluate(args.operation, args.load, 0 if args.operation == 'baseline' else args.sigma, args.repeat)
