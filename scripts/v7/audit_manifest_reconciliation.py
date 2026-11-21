"""Trace every unregistered physical-code change before updating clone expectations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = 'd7c4a3b18fc0ecc7588845f585dd3fd00e0a2add'
MANIFEST = 'tests/v7/v6-copy-manifest.json'
MAPPING = {
    'engine.py': ('YR-332-al', ['4fb69d6e']),
    'multiblock.py': ('YR-332-al', ['4fb69d6e', '6848ab00']),
    'order_schema.py': ('YR-332-an', ['31100a15']),
    'time_contract.py': ('YR-332-al', ['4fb69d6e']),
}
SPECS = {
    'YR-332-al': '.claude/docs/dashboard-task-specs/YR-332-al-multi-order-visit-link.md',
    'YR-332-an': '.claude/docs/dashboard-task-specs/YR-332-an-schedule-gate-time-field.md',
}
REPORTS = {'YR-332-al': 'outputs/reports/yr332_visit_links',
           'YR-332-an': 'outputs/reports/yr332_bnct_time_field'}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def normalized_sha(data):
    return hashlib.sha256((data.decode('utf-8').replace('\r\n', '\n').rstrip()+'\n').encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expect-mismatches', type=int, required=True)
    args = parser.parse_args()
    assert not git('status', '--short', '--untracked-files=no').strip()
    old = json.loads(git('show', BASE+':'+MANIFEST))
    current = json.loads((ROOT/MANIFEST).read_text(encoding='utf-8'))
    assert current['world_sha256'] == old['world_sha256']
    assert current['base_commit'] == old['base_commit']
    assert not git('diff', BASE, '--', 'src').strip(), 'Runtime code changed'
    rows = []
    for name, (task, short_commits) in MAPPING.items():
        rel = 'world/integrated/'+name
        path = 'src/yard_rl/v7/'+rel
        prior = old['v7_world_overrides'].get(rel, {})
        expected = prior.get('sha256', old['world_sha256'][rel])
        history = git('log', '--format=%H', BASE, '--', path).decode().splitlines()
        base = next(c for c in history if normalized_sha(git('show', c+':'+path)) == expected)
        changes = git('log', '--reverse', '--format=%H', base+'..'+BASE, '--', path).decode().splitlines()
        assert changes == [git('rev-parse', c).decode().strip() for c in short_commits]
        actual = normalized_sha((ROOT/path).read_bytes())
        assert normalized_sha(git('show', changes[-1]+':'+path)) == actual
        evidence = {}
        for p in (SPECS[task], REPORTS[task]+'/README.md', REPORTS[task]+'/verification.json',
                  REPORTS[task]+'/authorization.json'):
            raw = git('show', BASE+':'+p)
            assert raw == (ROOT/p).read_bytes().replace(b'\r\n', b'\n')
            evidence[p] = hashlib.sha256(raw).hexdigest()
        verdict = json.loads((ROOT/REPORTS[task]/'verification.json').read_text(encoding='utf-8'))
        assert verdict['status'] == 'PASS'
        assert json.loads((ROOT/REPORTS[task]/'authorization.json').read_text(encoding='utf-8'))['allowed']
        for c in changes:
            subprocess.run(['git', 'merge-base', '--is-ancestor', c, 'origin/강화학습-판매'],
                           cwd=ROOT, check=True)
            assert task in git('show', c+':'+SPECS[task]).decode('utf-8')
        if task == 'YR-332-al':
            assert normalized_sha(git('show', verdict['source_commit']+':'+path)) == actual
        diff = git('diff', base, BASE, '--', path)
        rows.append(dict(path=path, relative=rel, task=task, prior_expected_sha256=expected,
                         prior_commit=base, changes=changes, current_sha256=actual,
                         delta_sha256=hashlib.sha256(diff).hexdigest(),
                         evidence_sha256=evidence,
                         matches_current_manifest=actual == current['v7_world_overrides'].get(rel, {}).get(
                             'sha256', current['world_sha256'][rel])))
    mismatches = [rel for rel in current['world_sha256'] if normalized_sha(
        (ROOT/'src/yard_rl/v7'/rel).read_bytes()) != current['v7_world_overrides'].get(rel, {}).get(
            'sha256', current['world_sha256'][rel])]
    assert len(mismatches) == args.expect_mismatches, mismatches
    out = dict(task='YR-332-be', source_commit=git('rev-parse', 'HEAD').decode().strip(),
               git_dirty=False, baseline=BASE, runtime_source_unchanged=True,
               source_tree=git('rev-parse', BASE+':src').decode().strip(),
               baseline_manifest_preserved=True, audited_files=rows, mismatches=mismatches,
               manifest_sha256=hashlib.sha256((ROOT/MANIFEST).read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(files=len(rows), changes=sum(len(r['changes']) for r in rows),
                         mismatches=mismatches, runtime_unchanged=True)))


if __name__ == '__main__':
    main()
