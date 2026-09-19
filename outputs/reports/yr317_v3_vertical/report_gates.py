"""Record partial vertical qualification honestly; preserve every original result."""
import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from yard_rl.experiments.gate_harness import (GateOutcome, GateStatus,
    ResearchGateReport, attach_common_gates, audit_dashboard, combine_reliability,
    judge_claim_alignment, judge_runtime_evidence, judge_scenario_validity)

OUT = Path(__file__).resolve().parent
SPEC = '.claude/docs/dashboard-task-specs/YR-317-h-v3-review-basic-layouts.md'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tests(path):
    suites = list(ET.parse(path).getroot().iter('testsuite'))
    return {k: sum(int(s.get(k, 0)) for s in suites)
            for k in ('tests', 'failures', 'errors', 'skipped')}


def source_matches(manifest):
    runner = hashlib.sha256(subprocess.check_output(['git', 'show', manifest['source_commit']+
        ':scripts/v3/qualify_vertical_environment.py'], cwd=ROOT)).hexdigest()
    raw = subprocess.check_output(['git', 'archive', manifest['source_commit'],
        'src/yard_rl/v3', 'configs'], cwd=ROOT)
    found = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive:
            if not member.isfile():
                continue
            name, data = member.name, archive.extractfile(member).read()
            if name.startswith('configs/') or name.endswith('.py'):
                if Path(name).suffix in {'.py', '.yaml', '.yml', '.json', '.csv', '.md'}:
                    data = data.replace(b'\r\n', b'\n')
                key = name.replace('src/yard_rl/v3/', 'v3/', 1)
                found[key] = hashlib.sha256(data).hexdigest()
    return runner == manifest['runner_sha256'] and found == manifest['runtime']['files']


def local_fix_evidence(checks, artifacts, source_commit, evidence_commit):
    probe = OUT/'deadlock-probe-f2533ee'
    replay = read(probe/'candidate-fix-replay.json')
    final_vertical, final_legacy = (tests(OUT/n) for n in
        ('tests-after-candidate-fix.xml', 'legacy-after-candidate-fix-tests.xml'))
    artifacts.extend(OUT/n for n in ('tests-after-candidate-fix.xml',
        'legacy-after-candidate-fix-tests.xml', 'legacy-after-candidate-fix-equivalence.json'))
    artifacts += [probe/'candidate-fix-replay.json', probe/'replay-candidate-fix.py']
    identity = replay['identity']
    checks['fix.source_commit'] = identity['frozen_source_commit'] == source_commit
    checks['fix.replay_script'] = sha(probe/'replay-candidate-fix.py') == identity['diagnostic_script_sha256']
    for relative, field, digest in (
            ('src/yard_rl/v3/layouts/candidates.py', 'candidate_file', 'candidate_sha256'),
            ('src/yard_rl/v3/stage/episode.py', 'rule_policy_file', 'rule_policy_file_sha256')):
        path = ROOT/relative
        artifacts.append(path)
        committed = subprocess.run(['git', 'show', evidence_commit+':'+relative],
            cwd=ROOT, capture_output=True, check=False)
        checks['fix.'+field] = (identity[field].endswith('/'+relative) and committed.returncode == 0
            and hashlib.sha256(committed.stdout).hexdigest() == identity[digest]
            and hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == identity[digest])
    checks['fix.tests_no_failures'] = all(v[k] == 0 for v in (final_vertical, final_legacy)
        for k in ('errors', 'failures', 'skipped'))
    checks['fix.replay_scope'] = replay['passed'] is True and set(replay['blocks']) == {'Y05', 'Y14'}
    counts = {'final_vertical_checks': final_vertical['tests'], 'final_legacy_checks': final_legacy['tests']}
    for block, row in replay['blocks'].items():
        snapshot = probe/(block+'-snapshot.pkl.gz')
        artifacts.append(snapshot)
        checks['fix.'+block+'.snapshot'] = sha(snapshot) == row['snapshot_sha256']
        checks['fix.'+block+'.same_initial'] = row['original']['initial'] == row['fixed']['initial']
        checks['fix.'+block+'.regression'] = (row['regression_passed'] is True
            and row['original']['newly_completed'] == 0 and row['fixed']['newly_completed'] == 10)
        for variant in ('original', 'fixed'):
            run = row[variant]
            checks['fix.'+block+'.'+variant] = (run['policy_exceptions'] == 0
                and run['invariant_checks'] > 0 and run['original_snapshot_not_mutated'] is True
                and run['newly_completed'] == run['final']['yard_done']-run['initial']['yard_done'])
            counts[block+'_'+variant+'_newly_completed'] = run['newly_completed']
    equivalence = read(OUT/'legacy-after-candidate-fix-equivalence.json')
    checks['fix.legacy_equivalence'] = (equivalence['passed'] is True
        and equivalence['all_requested_checks_passed'] is True and bool(equivalence['checks'])
        and all(v is True for v in equivalence['checks'].values())
        and sha(OUT/equivalence['baseline_receipt']['path']) == equivalence['baseline_receipt']['raw_sha256'])
    for label, row in equivalence['artifacts'].items():
        result, metadata = OUT/row['path'], OUT/row['source_manifest_path']
        artifacts += [result, metadata]
        checks['fix.legacy_'+label] = (sha(result) == row['compressed_sha256']
            and hashlib.sha256(gzip.decompress(result.read_bytes())).hexdigest() == row['uncompressed_sha256']
            and sha(metadata) == row['source_manifest_sha256'])
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-commit', required=True)
    parser.add_argument('--remote-ref', required=True)
    parser.add_argument('--out', type=Path, default=OUT/'qualification-gates.json')
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Use a new gate output; original evidence is never overwritten')
    run = OUT/'run-f2533ee'
    manifest = read(run/'manifest.json')
    summary = read(run/'summary.json') if (run/'summary.json').exists() else None
    arms = [read(run/a/'qualification.json') for a in manifest['arms']
            if (run/a/'qualification.json').exists()]
    progress, launch = read(run/'progress.json'), read(OUT/'qualification-launch.json')
    vertical, legacy = tests(OUT/'tests-final.xml'), tests(OUT/'legacy-focused-tests.xml')
    checks = {}
    artifacts = [OUT/'tests-final.xml', OUT/'legacy-focused-tests.xml',
        OUT/'legacy-equivalence.json', OUT/'qualification-launch.json',
        run/'manifest.json', run/'progress.json',
        run/'environment-spec.json', run/'canonical-input.json.gz', run/'balanced-vessels.json.gz']
    artifacts += [p for p in (run/'summary.json', run/'failure.json',
        run/'evidence-index.json', OUT/'evidence-index.json') if p.exists()]
    paired = (progress['state'] == 'completed' and summary is not None
        and summary['passed'] is True and summary['same_input_across_arms'] is True
        and len(arms) == 2 and summary['arms'] == arms and all(a['passed'] for a in arms))
    failed_as_recorded = (progress['state'] == 'failed' and summary is None
        and (run/'failure.json').exists() and bool(arms) and arms[-1]['passed'] is False
        and [a['arm'] for a in arms] == manifest['arms'][:len(arms)])
    checks['terminal_state_matches_records'] = paired or failed_as_recorded
    checks['source_and_runner_match_commit'] = source_matches(manifest)
    checks['launch_matches_source'] = launch['source_commit'] == manifest['source_commit']
    checks['canonical_bundle'] = sha(run/'canonical-input.json.gz') == manifest['canonical_bundle_sha256']
    checks['balanced_bundle'] = sha(run/'balanced-vessels.json.gz') == manifest['balanced_vessels_sha256']
    outcomes = {}
    for row in arms:
        folder = run/row['arm']
        saved = read(folder/'qualification.json')
        result_path = folder/'result.json.gz' if (folder/'result.json.gz').exists() else folder/'result.json'
        result_bytes = gzip.decompress(result_path.read_bytes()) if result_path.suffix == '.gz' else result_path.read_bytes()
        result = json.loads(result_bytes.decode('utf8'))
        artifacts += [folder/'qualification.json', folder/'manifest.json', result_path]
        checks[row['arm']+'.record_identity'] = (saved == row and bool(saved['checks'])
            and all(type(v) is bool for v in saved['checks'].values())
            and saved['passed'] == all(saved['checks'].values())
            and hashlib.sha256(result_bytes).hexdigest() == saved['result_sha256'])
        request, vessel = result['request_summary'], result['vessel_work_summary']
        outcomes[row['arm']] = dict(trucks_completed=request['states'].get('COMPLETED', 0),
            vessel_requested=vessel['requested_moves'], vessel_completed=vessel['completed_yard_jobs'],
            vessel_remaining=vessel['outstanding_yard_jobs'], passed=saved['passed'],
            failed_checks=[k for k, v in saved['checks'].items() if not v])
        checks[row['arm']+'.summary_identity'] = (saved['request_summary'] == request
            and saved['vessel_work_summary'] == vessel
            and saved['checks']['all_vessel_work_completed'] == vessel['all_requested_work_completed']
            and saved['checks']['all_requests_completed'] == (request['states'].get('COMPLETED', 0) == sum(manifest['counts']))
            and vessel['requested_moves'] == vessel['completed_yard_jobs'] + vessel['outstanding_yard_jobs']
                + vessel['unadmitted_moves'] + vessel['unaccounted_yard_jobs'])
        checks[row['arm']+'.recording'] = all(saved['checks'][k] is True for k in
            ('request_recording', 'admission_counts', 'vessel_recording', 'canonical_input_matches',
             'frozen_networks', 'runtime_unchanged', 'checkpoint_unchanged'))
        checks[row['arm']+'.cost_ledger'] = math.isclose(result['request_summary']['accounted_wait_krw'],
            math.fsum(d['c_wait'] for d in result['days']), rel_tol=1e-10, abs_tol=1e-4)
        for name, expected in saved['raw_artifacts'].items():
            if Path(name).name != name:
                raise ValueError('Unexpected raw artifact path')
            artifacts.append(folder/name)
            checks[row['arm']+'.'+name] = sha(folder/name) == expected
    checks['tests_no_failures'] = all(v[k] == 0 for v in (vertical, legacy)
        for k in ('errors', 'failures', 'skipped'))
    equivalence = read(OUT/'legacy-equivalence.json')
    checks['legacy_equivalence'] = (equivalence['passed'] is True
        and bool(equivalence['checks']) and all(v is True for v in equivalence['checks'].values()))
    fix_counts = local_fix_evidence(checks, artifacts, manifest['source_commit'], args.evidence_commit)
    errors = [name for name, passed in checks.items() if not passed]
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in artifacts}
    stamp = dict(code=dict(git_head=manifest['source_commit'], git_dirty=manifest['git_dirty']),
        seeds={'diagnostic': [manifest['seed']], 'days': [d['seed'] for d in manifest['days']]},
        prereg=SPEC, params={'qualification_manifest': manifest,
            'scope': 'Manifest-derived provenance; implementation protocol, not performance preregistration'})
    runtime = judge_runtime_evidence(stamp, artifact_hashes=hashes, root=ROOT)
    if errors:
        runtime = GateOutcome('runtime_evidence', GateStatus.FAIL,
            'Qualification artifacts or checks do not reconcile.', tuple(errors)+runtime.reasons, runtime.evidence)
    board = audit_dashboard(ROOT, task_id='YR-317-h', expected_state='in-progress', spec_path=SPEC,
        evidence_paths=list(hashes), evidence_commits=[args.evidence_commit],
        remote_ref=args.remote_ref, pin_commit=args.evidence_commit)
    alignment = judge_claim_alignment(
        {'vertical_checks': 55, 'legacy_checks': 38, 'qualification_arms': 1,
         'paired_fixture_completed': 0, 'trucks_completed': 120, 'vessel_completed': 4469,
         'vessel_remaining': 1269, 'training_runs': 0,
         'final_vertical_checks': 63, 'final_legacy_checks': 38,
         'Y05_original_newly_completed': 0, 'Y05_fixed_newly_completed': 10,
         'Y14_original_newly_completed': 0, 'Y14_fixed_newly_completed': 10},
        {'vertical_checks': vertical['tests']-vertical['failures']-vertical['errors']-vertical['skipped'],
         'legacy_checks': legacy['tests']-legacy['failures']-legacy['errors']-legacy['skipped'],
         'qualification_arms': len(arms), 'paired_fixture_completed': int(paired),
         'trucks_completed': sum(o['trucks_completed'] for o in outcomes.values()),
         'vessel_completed': sum(o['vessel_completed'] for o in outcomes.values()),
         'vessel_remaining': sum(o['vessel_remaining'] for o in outcomes.values()),
         'training_runs': manifest['new_training_runs'], **fix_counts})
    reliability = combine_reliability(runtime, board, alignment)
    internal = {name: bool(arms) and all(a['checks'][key] is True for a in arms) for name, key in (
        ('event_time_order', 'request_recording'), ('physical_constraints', 'physical_invariants_enabled'),
        ('fixture_work_completion', 'all_vessel_work_completed'))}
    internal['ledger_conservation'] = not errors and all(a['checks']['container_flow_passed']
        and a['checks']['vessel_recording'] for a in arms)
    scenario = judge_scenario_validity(internal_checks=internal, flow_checks={}, anchors={},
        continuous_operation=False, request_real_terminal_claim=False, root=ROOT)
    scenario = GateOutcome(scenario.name, scenario.status, scenario.summary,
        scenario.reasons+('Matched horizontal-environment qualification is not included.',
            'Recorded vessel deadline floor omits the full end-transfer workload; feasibility is not certified.'), scenario.evidence)
    performance = GateOutcome('performance', GateStatus.INCONCLUSIVE,
        'One implementation diagnostic trajectory does not test policy superiority.',
        ('No independent performance sample or prespecified cost inference was performed.',),
        {'independent_performance_runs': manifest['independent_performance_runs'], 'cost_inference': False})
    payload = attach_common_gates(dict(schema='yr317.vertical.partial-gates.v1', task='YR-317-h',
        scope='Partial vertical implementation qualification; H/V validation remains unfinished',
        physical_fixture_passed=paired and not errors, finished_paired_fixture=paired,
        local_candidate_fix=fix_counts, local_fix_is_not_whole_horizon_qualification=True,
        outcomes=outcomes, checks=checks, evidence_commit=args.evidence_commit, original_results_modified=False),
        ResearchGateReport(performance, reliability, scenario))
    with args.out.open('x', encoding='utf8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: payload['common_gates'][k]['status']
                     for k in ('performance', 'reliability', 'scenario_validity')}))


if __name__ == '__main__':
    main()
