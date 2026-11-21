"""Verify the saved monthly plan diagnosis without changing input or policy."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/reports/yr332_month_inventory'


def main():
    path = OUT / 'diagnosis.json'
    d = json.loads(path.read_text(encoding='utf-8'))
    assert d['repeat_identical'] and d['repro']['code']['git_dirty'] is False
    commit = d['repro']['code']['git_head']
    archive = subprocess.check_output(['git','archive','--format=zip',commit,
                                      'src','scripts/v7','.claude/docs/dashboard-task-specs'])
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for name, sha in d['sources'].items():
            f = Path(name)
            name = f.relative_to(ROOT).as_posix() if f.is_absolute() else f.as_posix()
            raw = z.read(name)
            assert sha in (hashlib.sha256(raw).hexdigest(),
                           hashlib.sha256(raw.replace(b'\n',b'\r\n')).hexdigest()), name
    for variant, data in d['variants'].items():
        assert len(data['rows']) == 30*21
        old, totals = dict(data['initial']), {}
        for row in data['rows']:
            b, day = row['block'], row['day']
            assert row['initial_sod'] == old[b]
            old[b] += row['GATE_IN'] + row['DISCHARGE'] - row['GATE_OUT'] - row['LOAD']
            assert old[b] == row['planned_eod']
            totals[day] = totals.get(day,0)+old[b]
        assert list(totals.values()) == data['daily_totals']
        for f in ('GATE_IN','GATE_OUT','DISCHARGE','LOAD'):
            assert sum(r[f] for r in data['rows']) == data['summary']['flow_totals'][f]
    nominal = d['variants']['original_test']['summary']
    a = d['variants']['actual_initial_same_plan']
    guarded = d['variants']['actual_initial_guarded_plan']
    assert (nominal['first_day_total'],nominal['minimum_total']) == (20148,9544)
    assert a == guarded and not d['changed_streams']
    negative = [r for r in a['rows'] if r['planned_eod'] < 0]
    assert all(r['LOAD'] == r['DISCHARGE'] == 0 for r in negative)
    last = [r for r in a['rows'] if r['day']==29]
    assert min(r['planned_eod'] for r in last) >= 0
    summary = a['summary']
    result = dict(task='YR-332-bf',verification_passed=True,source_commit=commit,
        verified_source_files=len(d['sources']),verified_balance_rows=30*21*3,
        input_repeat_identical=True,orders=d['orders'],
        nominal_test_reproduced=True,actual_initial_total=summary['initial_total'],
        actual_first_day_total=summary['first_day_total'],actual_minimum_total=summary['minimum_total'],
        actual_final_total=summary['last_day_total'],
        negative_block_days=len(negative),affected_blocks=sorted({r['block'] for r in negative}),
        worst_planned_balance=min(r['planned_eod'] for r in negative),
        first_negative=dict(day_one_based=negative[0]['day']+1,block=negative[0]['block'],
                            balance=negative[0]['planned_eod']),
        month_end_minimum_block=min(r['planned_eod'] for r in last),
        existing_end_guard_changes=0,maximum_planned_block=summary['maximum_block'],
        capacity_per_block=d['capacity_per_block'],actual_90pct_guard=summary['original_90pct_guard'],
        minimum_vs_initial=summary['minimum_total']/summary['initial_total'],
        minimum_vs_first_day=summary['first_day_retention'],flow_totals=summary['flow_totals'],
        interpretation='Plan balance only; no physical negative inventory or actual service-time claim.',
        training=False,policy_run=False,
        artifact_hashes={path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()},
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
