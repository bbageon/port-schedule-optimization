"""Pin the completed demand correction to verified raw results and Dashboard."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import subprocess
import xml.etree.ElementTree as ET

from benchmark_default_eval import save
from six_step_learning import gz_read, sha
from evaluate_demand_preservation import OUT, CASES, name
from yard_rl.experiments.gate_harness import (
    GateOutcome, GateStatus, attach_common_gates, audit_dashboard,
    judge_claim_alignment, judge_runtime_evidence, report_from_dict)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finalize(pin):
    manifest, checked = read(OUT/'campaign.json'), read(OUT/'verification.json')
    assert checked['summary_sha256'] == sha(OUT/'summary.json')
    assert checked['report_sha256'] == sha(OUT/'README.md')
    rows_by_group = {}
    for case in CASES:
        if case[-1]:
            continue
        mode, arm, seed, load, _ = case
        rows = gz_read(OUT/'orders'/(name(case)+'.json.gz'))
        run = gz_read(OUT/'evaluation'/(name(case)+'.json.gz'))
        missing = sum(r['times']['gate_in_s'] is None for r in rows)
        pending = []
        for end in (86400.,93600.):
            pending.append(sum(r['times']['gate_in_s'] is not None and
                (r['times']['gate_out_s'] is None or r['times']['gate_out_s'] > end) for r in rows))
        total = 0.
        for row in rows:
            t = row['times']
            if t['gate_in_s'] is not None:
                exit_time = 86400. if t['gate_out_s'] is None else min(t['gate_out_s'],86400.)
                total += exit_time - t['gate_in_s']
        raw = [missing,*pending,total/(len(rows)-missing)/60.,run['result']['phi_krw']]
        rows_by_group.setdefault((arm,load,mode),[]).append(raw)
    reported, actual = {}, {}
    for line in (OUT/'README.md').read_text(encoding='utf-8').splitlines():
        cells = [c.strip() for c in line.split('|')][1:-1]
        if len(cells)!=8 or cells[2] not in ('legacy','preserved'):
            continue
        key = (cells[0],int(cells[1]),cells[2])
        raw = rows_by_group[key]
        for i, value in enumerate(cells[3:]):
            label = '/'.join(map(str,key))+'/'+str(i)
            reported[label] = float(value)
            actual[label] = statistics.mean(r[i] for r in raw)
    assert len(reported) == len(rows_by_group)*5
    claims = asdict(judge_claim_alignment(reported,actual,absolute_tolerance=.00500001))
    assert claims['status']=='PASS',claims
    hashes = {**manifest['sources'],**manifest['input_sha256'],**checked['artifact_sha256'],
              str(OUT/'summary.json'):sha(OUT/'summary.json'),
              str(OUT/'README.md'):sha(OUT/'README.md'),
              str(OUT/'preflight.xml'):sha(OUT/'preflight.xml'),
              str(OUT/'world-copy.xml'):sha(OUT/'world-copy.xml'),
              'tests/v7/test_world_clone.py':sha('tests/v7/test_world_clone.py'),
              'tests/v7/v6-copy-manifest.json':sha('tests/v7/v6-copy-manifest.json')}
    runtime = asdict(judge_runtime_evidence(manifest['repro'],artifact_hashes=hashes,root='.'))
    assert runtime['status']=='PASS',runtime
    tests = ET.parse(OUT/'preflight.xml').find('testsuite').attrib
    assert int(tests['failures'])==int(tests['errors'])==0 and int(tests['tests'])==63
    clone_tests = ET.parse(OUT/'world-copy.xml').find('testsuite').attrib
    assert int(clone_tests['failures'])==int(clone_tests['errors'])==0 and int(clone_tests['tests'])==66
    upstream = subprocess.check_output(['git','rev-parse','--abbrev-ref','@{upstream}'],encoding='utf-8').strip()
    board = asdict(audit_dashboard('.',task_id='YR-332-aa',expected_state='done',
        spec_path='.claude/docs/dashboard-task-specs/YR-332-aa-rejected-demand-preservation.md',
        evidence_paths=[str(OUT/p) for p in ('campaign.json','summary.json','verification.json','README.md')],
        evidence_commits=[pin],remote_ref=upstream,pin_commit=pin))
    assert board['status']=='PASS',board
    evidence = dict(runtime=runtime,dashboard=board,claim_alignment=claims,
                    test_count=129,verifier_sha256=sha(__file__))
    gates = dict(
        performance=GateOutcome('performance',GateStatus.FAIL,
            '정책 채택 미통과 유지; 주문 보존은 환경 보정이며 재학습 효과 실험이 아님').as_dict(),
        reliability=GateOutcome('reliability',GateStatus.PASS,
            '실행 코드·전건 장부·보고 표·작업 상태·원격 커밋 일치',evidence=evidence).as_dict(),
        scenario_validity=GateOutcome('scenario_validity',GateStatus.FAIL,
            '주문 누락은 복구; 본선 운반·예약 정보·교사 대안의 현실성 보정은 남음',
            reasons=['YR-332-i vessel transport/time calibration',
                     'YR-332-f appointment versus actual arrival',
                     'YR-332-ab executable teacher alternatives']).as_dict())
    verdict = dict(task='YR-332-aa',order_preservation='PASS',operational_adoption='NOT_PASSED',
                   dashboard_pin=pin,summary_sha256=sha(OUT/'summary.json'),common_gates=gates)
    save(OUT/'final-dashboard-audit.json',board)
    save(OUT/'final-verdict.json',attach_common_gates(verdict,report_from_dict(verdict)))
    print(f'YR-332-aa: {len(reported)} report numbers, 129 tests, runtime and Dashboard PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pin',required=True)
    finalize(parser.parse_args().pin)
