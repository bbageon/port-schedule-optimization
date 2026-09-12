"""Package two independently generated seed bundles; never train or simulate."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from yard_rl.experiments.gate_harness import (
    attach_common_gates, judge_claim_alignment, report_from_dict,
)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--repeat', type=Path, required=True)
    args = parser.parse_args()
    root, repeat = args.report, args.repeat
    first = root / 'seed-9900306'
    a, b = read(first/'audit.json'), read(repeat/'audit.json')
    ma, mb = read(first/'manifest.json'), read(repeat/'manifest.json')
    before = read(first/'baseline-audit.json')
    ah, bh = digest(first/'seed-data.json.gz'), digest(repeat/'seed-data.json.gz')
    tests, cases = {}, set()
    for name in ('verified-tests.xml', 'final-focused-tests.xml'):
        xml = ET.parse(root/name).getroot()
        suites = list(xml.iter('testsuite'))
        tests[name] = {k: sum(int(s.get(k, '0')) for s in suites)
                       for k in ('tests', 'failures', 'errors', 'skipped')}
        tests[name]['seconds'] = sum(float(s.get('time', '0')) for s in suites)
        cases.update((c.get('classname'), c.get('name')) for c in xml.iter('testcase'))
    checks = dict(
        both_clean=a['code']['git_dirty'] is b['code']['git_dirty'] is False,
        same_code=a['code']['git_head'] == b['code']['git_head']
            and a['code']['source_sha256'] == b['code']['source_sha256'],
        same_seed=a['seed'] == b['seed'] == 9900306,
        same_days=ma['days'] == mb['days'] and len(ma['days']) == 30,
        same_rules=ma['rules'] == mb['rules'] and ma['rules_sha256_lf'] == mb['rules_sha256_lf'],
        same_original_trucks=a['raw_schedule_sha256'] == b['raw_schedule_sha256'],
        same_original_vessels=a['raw_vessels_sha256'] == b['raw_vessels_sha256'],
        same_initial_cargo=a['initial_inventory_sha256'] == b['initial_inventory_sha256'],
        both_input_valid=a['audit']['passed'] is b['audit']['passed'] is True,
        equal_audits=a['audit'] == b['audit'],
        data_files_byte_identical=ah == bh == a['artifact']['sha256'] == b['artifact']['sha256'],
        no_unregistered_changes=all(a['checks'].values()) and all(b['checks'].values()),
        one_core=ma['cpus'] == mb['cpus'] == [23],
        no_production_simulation_or_learning=all(r['simulation_steps'] == r['learning_updates'] == 0 for r in (a,b)),
        all_selected_tests_passed=all(t[k] == 0 for t in tests.values() for k in ('failures','errors','skipped')),
        unique_tests=len(cases) == 147,
    )
    if not all(checks.values()):
        raise RuntimeError(f'Evidence mismatch: {checks}')
    reported = dict(trucks=177500, sources=161371, exits=146963, unassigned=14408,
                    duplicate_exits=0, missing_sources=0, unbound_vessel_load_moves=0,
                    normalized_initial_block_labels=13608, unique_tests=147)
    au = a['audit']
    raw = dict(trucks=au['truck_orders'], sources=au['sources'], exits=au['exits'],
               unassigned=au['ending_unassigned_sources'], unique_tests=len(cases),
               normalized_initial_block_labels=a['normalized_initial_block_labels'],
               **{k: au['contract']['violations'].get(k, 0) for k in
                  ('duplicate_exits', 'missing_sources', 'unbound_vessel_load_moves')})
    alignment = judge_claim_alignment(reported_values=reported, raw_values=raw)
    if alignment.status.value != 'PASS':
        raise RuntimeError(alignment.as_dict())
    for name in ('audit.json', 'manifest.json', 'baseline-audit.json'):
        target = root / ('repeat-' + name)
        if target.exists():
            raise FileExistsError(target)
        shutil.copyfile(repeat/name, target)
    files = ['seed-9900306/'+n for n in
             ('audit.json', 'manifest.json', 'baseline-audit.json', 'seed-data.json.gz')]
    files += ['repeat-'+n for n in ('audit.json','manifest.json','baseline-audit.json')]
    files += list(tests) + ['development-tests.xml', 'focused-tests.xml']
    value = dict(task='YR-306', scope='regenerated fixed-identity input, not runtime acceptance',
        source_code=a['code'], seed=9900306, rule_manifest=ma,
        checks=checks, tests_overlap_do_not_sum=tests, unique_passing_tests=len(cases),
        earlier_failed_tests='development-tests.xml and focused-tests.xml retained; repaired by final code',
        tests_scope='Generator, contract, fixed admission, PPO integration/continuous fixtures and frozen engine clone. Not all v5 tests.',
        old_violations=before['violations'], new_audit=au,
        changed_outbound_size_metadata=a['changed_outbound_size_metadata'],
        normalized_initial_block_labels=a['normalized_initial_block_labels'],
        bundle_sha256=ah, generation_seconds=[a['elapsed_s'], b['elapsed_s']],
        generation_simulation_steps=0, generation_learning_updates=0,
        runtime_default_changed=False, ready_for_training=False,
        claim_alignment=alignment.as_dict(), artifacts_sha256={f:digest(root/f) for f in files})
    value = attach_common_gates(value, report_from_dict(read(root/'gate-after-seed.json')))
    write(root/'verification.json', value)
    print(json.dumps(dict(checks=checks, raw_counts=raw, bundle_sha256=ah), indent=2))


if __name__ == '__main__':
    main()
