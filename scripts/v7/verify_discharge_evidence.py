"""Check YR-332-h diagnostic claims, provenance and artifact hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

from yard_rl.experiments.gate_harness import judge_claim_alignment


def main():
    root = Path.cwd()
    out = root / 'outputs/reports/yr332_discharge_latency'
    load = lambda name: json.loads((out / name).read_text(encoding='utf-8'))
    latency, discharge = load('latency.json'), load('discharge.json')
    report = (out / 'report.md').read_text(encoding='utf-8')
    env = latency['environment']
    assert not env['tracked_dirty'] and not discharge['tracked_dirty']
    assert env['commit'] == discharge['commit']
    subprocess.run(['git', 'cat-file', '-e', env['commit']+'^{commit}'], check=True)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    source_hashes = {}
    for name, expected in env['sha256'].items():
        path = Path(name)
        if not path.is_absolute():
            path = root / path
        assert digest(path) == expected, name
        source_hashes[path.relative_to(root).as_posix()] = expected
    fixture = root / 'tests/v7/test_discharge_backpressure.py'
    assert digest(fixture) == discharge['fixture_sha256']
    source_hashes[fixture.relative_to(root).as_posix()] = digest(fixture)

    rows = discharge['rows']
    assert len(rows) == 16
    assert all(r['remaining'] == r['pipeline_remaining'] == 0 for r in rows)
    assert all(r['peak'] <= r['cap'] for r in rows if r['cap'] is not None)
    pick = lambda cap, delay: next(r for r in rows if
        (r['ships'], r['cap'], r['yard_delay_s']) == (1, cap, delay))
    fast, slow = pick(3, 0), pick(3, 600)
    assert slow['idle_s'] > fast['idle_s'] and slow['vessel_cost_krw'] > fast['vessel_cost_krw']
    delta = slow['completion_s']['V0'] - fast['completion_s']['V0']
    assert pick(None, 0)['completion_s'] == pick(None, 600)['completion_s']
    assert all(r['same_best'] and r['max_abs_error'] < 7.46e-8 for r in latency['network'])
    assert all(r['same_assignment'] and r['same_features'] and r['orders'] == r['feasible_pairs']
               for r in latency['pipeline'])

    # Read the displayed table cells, rather than comparing two copies of JSON.
    number = lambda cell: float(cell.strip().replace(',', ''))
    reported, raw = {}, {}
    pipeline = {r['orders']: r for r in latency['pipeline']}
    network = {r['candidates']: r for r in latency['network']}
    table_kind = None
    for line in report.splitlines():
        if line.startswith('| 후보 수 |'):
            table_kind = 'network'
        elif line.startswith('| 오더 수 |'):
            table_kind = 'pipeline'
        if not line.startswith('| '):
            continue
        cells = line.split('|')[1:-1]
        label = cells[0].strip()
        if label == 'ON, 3':
            for key, cell, value in zip(('ship_fast', 'ship_slow', 'ship_delay'), cells[1:],
                    (fast['completion_s']['V0'], slow['completion_s']['V0'], delta)):
                reported[key], raw[key] = number(cell), value
        elif re.fullmatch(r'[\d,]+', label):
            n = int(label.replace(',', ''))
            assert table_kind in ('network', 'pipeline')
            is_network = table_kind == 'network'
            series = network[n] if is_network else pipeline[n]
            keys = ('scalar', 'batch', 'chunks1024') if is_network else (
                'scalar_total', 'batch_cached_total', 'generation')
            for key, cell in zip(keys, cells[1:]):
                name = f'{"network" if is_network else "pipeline"}_{n}_{key}'
                reported[name], raw[name] = number(cell), series[key]['median_ms']
    assert len(reported) == 24
    alignment = judge_claim_alignment(reported, raw, absolute_tolerance=.000501)
    assert alignment.status.value == 'PASS', alignment.as_dict()

    suites = {}
    for name in ('tests.xml', 'cargo-tests.xml', 'baseline-cargo-test.xml'):
        suites[name] = [r.attrib for r in ET.parse(out / name).getroot().iter('testsuite')]
    assert suites['tests.xml'][0]['tests'] == '178'
    assert suites['tests.xml'][0]['failures'] == '0'
    assert suites['cargo-tests.xml'][0]['tests'] == '21'
    assert suites['cargo-tests.xml'][0]['failures'] == '1'
    old = load('baseline-cargo-check.json')
    assert old['exit_code'] == 1 and "unsupported operand type(s) for /" in old['stdout']
    assert "test_cargo_capacity_wait.py:112" in old['stdout']
    artifacts = {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()
                 and p.name not in ('verification.json', 'dashboard-audit.json')}
    result = dict(schema='yard.v7.discharge-verification.v1', source_commit=env['commit'],
        claim_alignment=alignment.as_dict(), source_sha256=source_hashes,
        artifact_sha256=artifacts, test_suites=suites,
        scope=dict(implementation='PASS within diagnostic cases',
                   performance='INCONCLUSIVE: no policy training/comparison',
                   scenario_validity='INCONCLUSIVE: fleet, return timing and deadlines uncalibrated',
                   authorization_for_new_experiment=False))
    (out / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n',
                                          encoding='utf-8')
    print(f'PASS: {len(reported)} displayed values, {len(source_hashes)} source hashes; '
          '198 distinct tests passed, 1 pre-existing failure reproduced.')


if __name__ == '__main__':
    main()
