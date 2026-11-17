"""Verify report claims and, after publication, commit/board/evidence byte alignment."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import xml.etree.ElementTree as ET

from ppo_credit_audit import OUT, SOURCE, KINDS, SEEDS, sha, read, save
from ppo_credit_summary import records

SPEC = Path('.claude/docs/dashboard-task-specs/YR-332-ax-ppo-credit-diagnosis.md')
AY = Path('.claude/docs/dashboard-task-specs/YR-332-ay-value-learning-stability.md')
HISTORY = Path('.claude/docs/strategy-history/2026-11-17-YR-332-ax-ppo-credit.md')
CONTROL = [Path('.gitattributes'), SPEC, AY, HISTORY, Path('.claude/docs/strategy-history/README.md'),
           *[Path('.claude/Dashboard')/p for p in ('README.md', 'backlog.md', 'done.md')]]


def local():
    manifest = read(OUT/'manifest.json')
    for group in ('input_hashes', 'analyzer_hashes'):
        for path, digest in manifest[group].items():
            assert sha(path) == digest, path
    summary, detail = read(OUT/'summary.json'), read(OUT/'choice-detail.json')
    assert detail['script_sha256'] == sha('scripts/v7/ppo_credit_choice_detail.py')
    body = (OUT/'README.md').read_text('utf-8')
    tf = {r['group']['load']: r['metrics'] for r in summary['tables']['kind_load'] if r['group']['kind'] == 'transformer'}
    bins = {(r['group']['load'], r['group']['time_bin']): r['metrics']
            for r in summary['tables']['kind_load_time_bin'] if r['group']['kind'] == 'transformer'}
    checked_tables = 0
    for load, m in tf.items():
        line = f'| {load:,}건 | '+' | '.join(f'{100*bins[load, t]["positive_advantage_fraction"]:.1f}%' for t in range(4))+' |'
        assert line in body, line
        line = f'| Transformer · {load:,} | {m["pre"]["rmse"]:.3f} | {m["post"]["rmse"]:.3f} |'
        assert line in body, line
        sat = m['saturation']
        rates = [sat['candidate_fraction'], *[sat['cap_counts'][i]/sat['flow_counts'][i] for i in range(2)],
                 sat['states_with_any']/m['n']]
        line = f'| {load:,}건 | '+' | '.join(f'{100*r:.1f}%' for r in rates)+' |'
        assert line in body, line
        checked_tables += 3
    for row in detail['rows']:
        if row['kind'] != 'transformer' or row['load'] != 0:
            continue
        a, b = row['initial'], row['final']
        line = (f'| {row["seed"]} | {row["mixed_single_crane_n"]:,} | '
                f'{100*a["mixed_single_crane_inbound_probability"]:.2f}% → {100*b["mixed_single_crane_inbound_probability"]:.2f}% | '
                f'{100*a["mixed_single_crane_top_inbound_fraction"]:.2f}% → {100*b["mixed_single_crane_top_inbound_fraction"]:.2f}% |')
        assert line in body, line
        checked_tables += 1
    raw = records()
    early = [r for r in raw if r['kind'] == 'transformer' and r['load'] == 15000 and r['hour'] < 12]
    assert len(early) == 5838 and sum(r['advantage'] > 0 for r in early) == 6
    for kind in KINDS:
        for seed in SEEDS:
            parts = [r for r in detail['rows'] if r['kind'] == kind and r['seed'] == seed]
            whole = next(r for r in parts if r['load'] == 0)
            assert whole['n'] == 12288
            for field in ('n', 'single_crane_n', 'mixed_single_crane_n'):
                assert sum(r[field] for r in parts if r['load']) == whole[field]
    suites = list(ET.parse(OUT/'tests.xml').getroot().iter('testsuite'))
    count = sum(int(s.attrib['tests']) for s in suites)
    failed = sum(int(s.attrib['failures'])+int(s.attrib['errors']) for s in suites)
    assert count == 6 and failed == 0
    for path in [*CONTROL, OUT/'README.md', Path('docs/research/v7-discharge/ppo-credit-analysis-plan.md')]:
        if path.suffix == '.md':
            assert len(path.read_text('utf-8').splitlines()) <= 200, path
    for task, state, spec in [('YR-332-ax', 'done', SPEC), ('YR-332-ay', 'backlog', AY)]:
        found = []
        for name in ('backlog', 'ready', 'in-progress', 'done', 'cancelled'):
            found.extend([name for line in (Path('.claude/Dashboard')/f'{name}.md').read_text('utf-8').splitlines()
                          if line.startswith(f'| {task} |')])
        assert found == [state], (task, found)
        assert f'**상태**: {state}' in spec.read_text('utf-8')
    result = dict(status='PASS', checked_table_rows=checked_tables, source_files_unchanged=300,
        tests=count, failed=failed, supplemental_partition_checks=True, ax_ay_rows_unique=True,
        markdown_under_200_lines=True, source_manifest_sha256=sha(OUT/'manifest.json'),
        supplemental_sha256=sha(OUT/'choice-detail.json'), report_sha256=sha(OUT/'README.md'),
        supplementary_analyzer_sha256=sha('scripts/v7/ppo_credit_choice_detail.py'),
        plotter_sha256=sha('scripts/v7/plot_ppo_credit.py'), verifier_sha256=sha(__file__))
    save(OUT/'report-verification.json', result)
    old = read(SOURCE/'current-gates.json')
    save(OUT/'research-status.json', dict(scope='Read-only diagnosis; original experimental gates carried unchanged',
        source_gate=(SOURCE/'current-gates.json').as_posix(), source_gate_sha256=sha(SOURCE/'current-gates.json'),
        original_status={k: old['common_gates'][k]['status'] for k in ('performance', 'reliability', 'scenario_validity')},
        diagnostic_record_checks='PASS', new_performance_claim=False, new_simulation=False, new_training=False,
        next_task='YR-332-ay', next_task_started=False))
    print(json.dumps(result, ensure_ascii=False), flush=True)


def publication(commit):
    from yard_rl.experiments.gate_harness import audit_dashboard
    paths = [*CONTROL, *Path('scripts/v7').glob('ppo_credit_*.py'), Path('scripts/v7/plot_ppo_credit.py'),
             Path(__file__).resolve().relative_to(Path.cwd()), Path('tests/test_ppo_credit_diagnostics.py'),
             Path('docs/research/v7-discharge/ppo-credit-analysis-plan.md'),
             *[p for p in OUT.iterdir() if p.is_file() and p.name not in ('publication.json', 'dashboard-audit.json')]]
    expected = {p.as_posix(): sha(p) for p in paths}
    # Include immutable source bytes; original commit hashes alone do not guarantee the current index retained them.
    expected.update(read(OUT/'manifest.json')['input_hashes'])
    requests = ''.join(commit+':'+p+'\n' for p in expected).encode('utf-8')
    raw = subprocess.check_output(['git', 'cat-file', '--batch'], input=requests)
    at = 0
    for path, digest in expected.items():
        end = raw.index(b'\n', at)
        header = raw[at:end].split()
        assert len(header) == 3, (path, header)
        size = int(header[2])
        assert hashlib.sha256(raw[end+1:end+1+size]).hexdigest() == digest, path
        at = end+size+2
    board = audit_dashboard('.', task_id='YR-332-ax', expected_state='done', spec_path=SPEC,
        evidence_paths=[OUT/'README.md', OUT/'manifest.json', OUT/'summary.json', OUT/'verification.json',
                        OUT/'report-verification.json', OUT/'research-status.json'],
        evidence_commits=[commit], remote_ref='origin/강화학습-판매', pin_commit=commit)
    assert board.status.value == 'PASS', board.as_dict()
    save(OUT/'dashboard-audit.json', board.as_dict())
    result = dict(status='PASS', evidence_commit=commit, files_byte_identical=len(expected),
                  remote_ref='origin/강화학습-판매', dashboard_status='PASS',
                  explanation='Published analysis and original inputs retain measured bytes; no new training claim.')
    save(OUT/'publication.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.commit:
        publication(args.commit)
    else:
        local()
