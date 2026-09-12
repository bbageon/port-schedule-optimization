"""Package YR-306 identity audit evidence; never run or repair a simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    args = parser.parse_args()
    root = args.root.resolve()
    report = root / 'outputs/reports/yr306_container_identity'
    run = root / 'outputs/v5/yr306-identity-preflight-30d'

    def read(path):
        return json.loads(path.read_text(encoding='utf-8'))

    def write(name, value):
        (report / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n',
                                   encoding='utf-8')

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    before, after = (read(report / name) for name in ('baseline-frozen.json', 'after-frozen.json'))
    manifest, status, contract = (read(run / name) for name in
                                  ('manifest.json', 'status.json', 'container_contract.json'))
    tests = {}
    cases = {}
    for name in ('targeted-tests.xml', 'contract-and-clone-tests.xml', 'full-tests.xml',
                 'fixed-admission-tests.xml'):
        xml = ET.parse(report / name).getroot()
        suites = list(xml.iter('testsuite'))
        tests[name] = {key: sum(int(s.get(key, '0')) for s in suites)
                       for key in ('tests', 'failures', 'errors', 'skipped')}
        tests[name]['seconds'] = sum(float(s.get('time', '0')) for s in suites)
        tests[name]['all_collected_passed'] = all(tests[name][key] == 0
                                                 for key in ('failures', 'errors', 'skipped'))
        cases[name] = {(c.get('classname'), c.get('name')) for c in xml.iter('testcase')}
    # This is a report of completed checks, not permission to hide an incomplete run.
    if (tests['full-tests.xml']['tests'] != 249
            or tests['fixed-admission-tests.xml']['tests'] != 9 or not all(
                t['all_collected_passed'] for t in tests.values())):
        raise RuntimeError('Inspect incomplete or failed tests before packaging the report')

    checks = {
        'baseline_clean': before['code']['git_dirty'] is False,
        'after_clean': after['code']['git_dirty'] is False,
        'same_audit_script': before['audit_script_sha256'] == after['audit_script_sha256'],
        'same_seed_and_daily_workloads': before['seed'] == after['seed'] == manifest['seed']
            and before['days'] == after['days'] == manifest['days'],
        'same_six_order_fields': before['schema_fields_v3'] == before['schema_fields_v5']
            == after['schema_fields_v3'] == after['schema_fields_v5']
            and len(after['schema_fields_v5']) == 6,
        'all_original_public_ids_unlinked': before['identity_mismatches'] == before['truck_orders'],
        'all_new_public_ids_linked': after['identity_mismatches'] == 0,
        'new_initial_ids_unique': after['initial_count'] == after['initial_distinct_ids'],
        'same_fixed_source_at_entrypoint': manifest['code'] | {'pid': after['code']['pid']}
            == after['code'],
        'entrypoint_audit_matches_static_audit': contract == after['container_contract'],
        'bad_input_rejected_before_world_and_training': status['state'] == 'failed'
            and status['error'].startswith('ContainerContractError:')
            and status['time_s'] is None and status['last_policy_boundary_s'] is None
            and status['updates'] == status['completed_days'] == status['admitted'] == 0,
        'no_final_model': not (run / 'final.pt').exists(),
        'failure_checkpoint_intact': digest(run / status['failed_checkpoint']['path'])
            == status['failed_checkpoint']['sha256'],
        'all_258_cases_covered': len(cases['full-tests.xml'] | cases['fixed-admission-tests.xml']) == 258,
    }
    if not all(checks.values()):
        raise RuntimeError(f'Evidence mismatch: {[key for key, ok in checks.items() if not ok]}')
    for name in ('manifest.json', 'status.json', 'container_contract.json', 'events.jsonl'):
        shutil.copyfile(run / name, report / ('preflight-' + name))
    shutil.copyfile(root / 'outputs/v5/yr306-target-probe/diagnosis.json',
                    report / 'original-failure-diagnosis.json')
    commit = after['code']['git_head']
    tree = subprocess.check_output(['git', 'ls-tree', '-r', commit, '--', 'tests/v5'], cwd=root)
    extra_test = 'tests/v5/test_fixed_vessel_admission.py'
    extra_commit = subprocess.check_output(
        ['git', 'log', '-1', '--format=%H', '--', extra_test], cwd=root, text=True).strip()
    if not extra_commit:
        raise RuntimeError('Commit the additional tests before publishing evidence')
    files = ['baseline-frozen.json', 'after-frozen.json', 'preflight-manifest.json',
             'preflight-status.json', 'preflight-container_contract.json', 'preflight-events.jsonl',
             'original-failure-diagnosis.json', *tests]
    value = {
        'task': 'YR-306', 'scope': 'fixed-container identity repair and fail-closed input audit',
        'claim_scope': 'NO_PERFORMANCE_CLAIM',
        'preregistration_commit': '83d7670fa12bb9f20f369512d2edfeb924ce3816',
        'preregistration': '.claude/docs/dashboard-task-specs/YR-306-container-identity-audit.md',
        'source_before': before['code'], 'source_after': after['code'],
        'test_source_commit': commit,
        'test_tree_listing_sha256': hashlib.sha256(tree).hexdigest(),
        'additional_test_source_commit': extra_commit,
        'additional_test_source_path': extra_test,
        'additional_test_source_sha256_lf': hashlib.sha256(
            (root / extra_test).read_text(encoding='utf-8').encode()).hexdigest(),
        'cpu_affinity': [23], 'maximum_cpu_cores': 1,
        'tests_overlap_do_not_sum': tests,
        'test_fixture_scope': 'Small fixed-cargo fixtures; one day of vessel work. NOT original 30-day repair.',
        'checks': checks, 'identity_repair_verified': True,
        'original_30_day_input_valid': False, 'long_training_restarted': False,
        'violations_remaining': contract['violations'],
        'unresolved': [
            'Original daily generator reuses departed container targets without recorded re-entry.',
            'Original vessel LOAD plans have quantities but no fixed container manifests.',
            'Fixed incoming-to-outgoing cargo chains and physical readiness are not implemented end-to-end.',
            'Subsequent pickup/load must follow the actual block after inbound relocation.',
            'Static identity validity does not prove physical feasibility or policy performance.',
        ],
        'artifact_hash_mode': 'raw bytes; report-local gitattributes prevents newline conversion',
        'artifacts_sha256': {name: digest(report / name) for name in files},
    }
    from yard_rl.experiments.gate_harness import (
        attach_common_gates, judge_claim_alignment, report_from_dict,
    )
    reported = dict(trucks=177500, mismatch_before=177500, mismatch_after=0,
                    initial_distinct_ids=13608, duplicate_exits=88679, missing_sources=8,
                    unbound_vessel_load_streams=108, unbound_vessel_load_moves=44670, updates=0)
    raw = dict(trucks=after['truck_orders'], mismatch_before=before['identity_mismatches'],
               mismatch_after=after['identity_mismatches'], initial_distinct_ids=after['initial_distinct_ids'],
               updates=status['updates'], **contract['violations'])
    alignment = judge_claim_alignment(reported_values=reported, raw_values=raw)
    value['claim_alignment'] = alignment.as_dict()
    if alignment.status.value != 'PASS':
        raise RuntimeError('Reported counts disagree with source evidence')
    value = attach_common_gates(value, report_from_dict(read(report / 'gate-after-identity.json')))
    write('verification.json', value)
    print(json.dumps({'checks': checks, 'tests': tests, 'long_training_restarted': False},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
