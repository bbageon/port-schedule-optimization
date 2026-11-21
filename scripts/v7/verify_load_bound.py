"""Independently verify saved load-bound probes, input impact and regression scope."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/reports/yr332_load_bound'


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def digest(path):
    # Evidence text uses Git's LF representation; original source stamps stay intact.
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def source_check(record):
    commit = record['repro']['code']['git_head']
    assert record['repro']['code']['git_dirty'] is False
    archive = subprocess.check_output(['git', 'archive', '--format=zip', commit,
                                       'src', 'scripts/v7', 'tests/v7',
                                       'docs/research/v7-discharge'])
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for name, expected in record['sources'].items():
            p = Path(name)
            name = p.relative_to(ROOT).as_posix() if p.is_absolute() else p.as_posix()
            data = z.read(name)
            assert expected in {hashlib.sha256(data).hexdigest(),
                                hashlib.sha256(data.replace(b'\n', b'\r\n')).hexdigest()}, name
    return dict(commit=commit, clean=True, files=len(record['sources']))


def physical(record):
    copy = json.loads(json.dumps(record))
    for vessel in copy['vessels']:
        del vessel['with_targets_s'], vessel['without_targets_s']
    return copy


def regression(name):
    root = ET.parse(OUT / name).getroot()
    tests = root.findall('.//testcase')
    failures = sorted((t.attrib['classname'].split('.')[-1], t.attrib['name'])
                      for t in tests if t.find('failure') is not None)
    assert not root.findall('.//error')
    return dict(tests=len(tests), passed=len(tests)-len(failures), failures=failures)


def main():
    before, after = read('before-full.json'), read('after.json')
    assert before['violations'] == read('before.json')['violations']
    assert len(before['runs']) == len(after['runs']) == 18
    assert len(before['violations']) == 5 and not after['violations']
    for record in (before, after):
        assert record['repeat_identical'] and record['repeat'] == record['runs'][-1]
    assert [physical(r) for r in before['runs']] == [physical(r) for r in after['runs']]
    for run in after['runs']:
        assert run['remaining'] == 0
        for vessel in run['vessels']:
            for mode in ('with_targets_s', 'without_targets_s'):
                assert vessel[mode] <= vessel['actual_completion_s'] + 1e-8

    old_inputs, new_inputs = read('inputs-before.json'), read('inputs-after.json')
    assert len(old_inputs['rows']) == len(new_inputs['rows']) == 9
    assert old_inputs['source_commit'] == before['repro']['code']['git_head']
    assert new_inputs['source_commit'] == after['repro']['code']['git_head']
    changes, vessel_count = [], 0
    for old, new in zip(old_inputs['rows'], new_inputs['rows']):
        for key in ('case', 'audit', 'schedule_sha256', 'vessels_sha256',
                    'execution_scenarios_sha256', 'reference'):
            assert old[key] == new[key], (old['case'], key)
        for key in ('audit', 'scenarios_sha256', 'vessels_sha256'):
            assert old[key] == old['reference'][key]
        assert len(old['bounds']) == len(new['bounds'])
        vessel_count += len(old['bounds'])
        for x, y in zip(old['bounds'], new['bounds']):
            assert (x['block'], x['id']) == (y['block'], y['id'])
            assert y['value'] <= x['value'] + 1e-8
            if x['value'] != y['value']:
                assert x['id'].startswith('V-LOAD')
                changes.append(dict(case=old['case'], block=x['block'], vessel=x['id'],
                                    old_s=x['value'], new_s=y['value'],
                                    change_s=y['value']-x['value']))

    baseline = regression('regression-baseline-unrelated.xml')
    final = regression('regression-verified.xml')
    assert len(baseline['failures']) == 5
    assert final['failures'] == baseline['failures']
    expected_fails = {'test_world_clone', 'test_month_stage'}
    assert {x[0] for x in final['failures']} == expected_fails
    artifacts = [OUT / name for name in (
        'authorization.json', 'before.json', 'before-full.json', 'after.json',
        'inputs-before.json', 'inputs-after.json', 'regression-before.xml',
        'regression-focused.xml', 'regression-final.xml', 'regression-verified.xml',
        'baseline-check.json', 'regression-baseline-unrelated.xml',
        'baseline-setup-attempt.json', 'regression-baseline-setup-attempt.xml')]
    result = dict(task='YR-332-i', source_checks=[source_check(before), source_check(after)],
        fixed_conditions=18, runs_with_repeat=19,
        violations_before=5, violations_after=0,
        max_original_excess_s=max(v['excess_s'] for v in before['violations']),
        physical_traces_and_costs_unchanged=True, exact_repeat=True,
        input_cases=9, input_vessel_plans=vessel_count,
        execution_inputs_deadlines_and_departures_unchanged=True,
        bounds_changed=len(changes), bound_changes=changes,
        baseline_regression=baseline, final_regression=final,
        new_regression_failures=0, all_regression_tests_passed=False,
        relaxed_event_combinations=4*2*3*6,
        training=False, external_transport_calibrated=False,
        artifacts_sha256_lf={p.name: digest(p) for p in sorted(artifacts)},
        verifier_sha256_lf=digest(Path(__file__)))
    (OUT / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n',
                                          encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('bound_changes', 'artifacts_sha256_lf')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
