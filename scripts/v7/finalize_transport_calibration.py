"""Verify timing traces, historical source bytes, and the limited correction claim."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from benchmark_default_eval import save
from six_step_learning import sha
from yard_rl.experiments.gate_harness import (
    GateOutcome, GateStatus, attach_common_gates, audit_dashboard,
    judge_claim_alignment, judge_runtime_evidence, report_from_dict)

OUT = Path('outputs/reports/yr332_discharge_calibration')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def historical_sources(manifest):
    head = manifest['repro']['code']['git_head']
    assert manifest['repro']['code']['git_dirty'] is False
    paths = [Path(p).resolve().relative_to(Path.cwd()).as_posix() for p in manifest['sources']]
    data = subprocess.check_output(['git','cat-file','--batch'],
        input=('\n'.join(head+':'+p for p in paths)+'\n').encode())
    offset, line_endings = 0, []
    for path, expected in zip(paths, manifest['sources'].values(), strict=True):
        end = data.index(b'\n', offset)
        size = int(data[offset:end].split()[-1])
        blob = data[end+1:end+1+size]
        offset = end+2+size
        if hashlib.sha256(blob).hexdigest() != expected:
            assert hashlib.sha256(blob.replace(b'\n', b'\r\n')).hexdigest()==expected, path
            line_endings.append(path)
    return dict(commit=head, checked=len(paths), windows_line_endings=line_endings)


def verify():
    old_manifest = read(OUT/'timing-manifest.json')
    old, new = read(OUT/'timing-raw.json'), read(OUT/'corrected/timing-raw.json')
    summary = read(OUT/'corrected/timing-summary.json')
    assert sha(OUT/'corrected/timing-raw.json') == summary['raw_sha256']
    assert new['repeat'] == new['runs'][-1] and old['repeat'] == old['runs'][-1]
    for before, after in zip(old['runs']+[old['repeat']],new['runs']+[new['repeat']],strict=True):
        before, after = copy.deepcopy(before), copy.deepcopy(after)
        for result in (before, after):
            for vessel in result['vessels']:
                vessel.pop('declared_lower_bound_s')
        assert before == after
    old_counter = sum(v['actual_completion_s']+1e-8 < v['declared_lower_bound_s']
                      for r in old['runs'] for v in r['vessels'])
    new_counter = sum(v['actual_completion_s']+1e-8 < v['declared_lower_bound_s']
                      for r in new['runs'] for v in r['vessels'])
    assert old_counter == 3 and new_counter == 0
    assert all(r['identical'] for r in summary['daily_input_comparisons'])
    tests = ET.parse(OUT/'bound-final.xml').find('testsuite').attrib
    assert int(tests['tests']) == 138 and int(tests['failures']) == int(tests['errors']) == 0
    return dict(historical_sources=historical_sources(old_manifest),
                old_counterexamples=old_counter, new_counterexamples=new_counter,
                paired_runs=8, tests=138, actual=old, corrected=new)


def report():
    checked = verify()
    lines = ['# YR-332-i — 본선 시간 계약 점검·양하 하한 보정', '',
        '시간 진단 8회씩 전후 비교. 실제 작업 사건·완료·유휴는 같고, 잘못된 양하 하한 반례는 3→0이다.',
        '관련 회귀 138건 통과. 재학습·운반 대수·주행시간 변경은 없다.', '',
        '| 상자 수 | 기존 하한(초) | 실제 본선 완료(초) | 수정 하한(초) |',
        '|---|---:|---:|---:|']
    for old, new in zip(checked['actual']['runs'][:3], checked['corrected']['runs'][:3]):
        a, b = old['vessels'][0], new['vessels'][0]
        lines.append(f"| {old['config']['moves_per_ship']} | {a['declared_lower_bound_s']:.3f} | "
                     f"{a['actual_completion_s']:.3f} | {b['declared_lower_bound_s']:.3f} |")
    lines += ['', '처음 1~3개 상자는 차량 귀환 전에 배에서 내릴 수 있다. 기존 식은 오래 운전했을 때의',
        '평균 공급 간격을 처음부터 적용해 실제 완료보다 늦은 값을 최소 시간이라고 불렀다.',
        '공유 버퍼 상태를 받지 않는 함수에서 양하는 안벽 작업 간격의 하한만 쓰도록 고쳤다.',
        '하한은 필요조건이며 그 시각까지 마칠 수 있다는 보증이 아니다. 적하 추정식은 그대로다.', '',
        '3,500/7,500/15,000건의 기존 하루 입력은 주문·물리 입력·본선 계획 30건의 마감·출항 시각이',
        '전후 동일했다. 비교에서 분리한 것은 하한 진단값과 버전 표기다. 새 정책 성능 비교가 아니다.', '',
        '## 아직 해결되지 않은 부분', '',
        '- 현재 화물은 요청 470초 뒤 도착하고 차량도 같은 순간 재사용 가능해진다.',
        '- 양하 파이프라인은 도착 때가 아니라 야드 최종 적재 때 해제된다(도착·해제 각 54건 확인).',
        '- 블록당 3대·470초는 실측 보정값으로 확인되지 않았다. 공식 장비 보유량은 동시 투입량이 아니다.',
        '- 실제 운반차 투입·편도/귀환 시간·상자 인수 조건 자료가 필요하다. 현실성 전체 통과는 보류한다.', '',
        '연구 정책·보상을 바꿔 좋은 결과가 나오도록 맞춘 실험이 아니다. 다음은 같은 i 작업에서',
        '자료를 받아 운반·인계 계약을 확정하고, 이후 ab의 실행 가능한 교사 대안·c의 학습을 비교한다.', '',
        '첫 확장 회귀의 2건은 차량 1대로 16상자를 나르는 시험이 8,000초에서 잘렸기 때문이었다.',
        '시험 관측창만 16,000초로 늘려 전건 완료를 확인했고, 다시 실행한 138건은 모두 통과했다.',
        '운영 평가창은 바꾸지 않았다. 최초 실패 기록도 보존했다.', '',
        '[출처 검토](../../../docs/research/v7-discharge/transport-calibration-review.md) ·',
        '[비교 계약](../../../docs/research/v7-discharge/discharge-bound-correction-prereg.md) ·',
        '[원자료 전](timing-raw.json) · [원자료 후](corrected/timing-raw.json) ·',
        '[검증](verification.json) · [회귀](bound-final.xml)']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    checked.pop('actual'); checked.pop('corrected')
    save(OUT/'verification.json', dict(**checked, report_sha256=sha(OUT/'README.md')))


def finalize(pin):
    verify()
    verified = read(OUT/'verification.json')
    assert verified['report_sha256'] == sha(OUT/'README.md')
    manifest = read(OUT/'corrected/timing-manifest.json')
    artifacts = {p.as_posix():sha(p) for p in OUT.rglob('*') if p.is_file()
                 and p.name not in ('final-verdict.json','final-dashboard-audit.json')}
    artifacts.update(manifest['sources'])
    artifacts['tests/v7/test_vessel_completion_bound.py'] = sha('tests/v7/test_vessel_completion_bound.py')
    runtime = asdict(judge_runtime_evidence(manifest['repro'],artifact_hashes=artifacts,root='.'))
    assert runtime['status']=='PASS',runtime
    actual, reported = {}, {}
    old, new = read(OUT/'timing-raw.json')['runs'], read(OUT/'corrected/timing-raw.json')['runs']
    for line in (OUT/'README.md').read_text(encoding='utf-8').splitlines():
        cells = [c.strip() for c in line.split('|')][1:-1]
        if len(cells)!=4 or not cells[0].isdigit():
            continue
        idx = int(cells[0])-1
        values = [old[idx]['vessels'][0]['declared_lower_bound_s'],
                  old[idx]['vessels'][0]['actual_completion_s'],
                  new[idx]['vessels'][0]['declared_lower_bound_s']]
        for k,(value,truth) in enumerate(zip(cells[1:], values, strict=True)):
            label = f'{idx+1}/{k}'
            reported[label],actual[label] = float(value),truth
    assert len(actual)==9
    claims = asdict(judge_claim_alignment(reported,actual,absolute_tolerance=.00051))
    assert claims['status']=='PASS',claims
    upstream = subprocess.check_output(['git','rev-parse','--abbrev-ref','@{upstream}'],encoding='utf-8').strip()
    board = asdict(audit_dashboard('.',task_id='YR-332-i',expected_state='in-progress',
        spec_path='.claude/docs/dashboard-task-specs/YR-332-i-discharge-calibration.md',
        evidence_paths=[str(OUT/p) for p in ('verification.json','README.md','corrected/timing-summary.json')],
        evidence_commits=[pin],remote_ref=upstream,pin_commit=pin))
    assert board['status']=='PASS',board
    gates = dict(performance=GateOutcome('performance',GateStatus.FAIL,
        '정책 채택 미통과 유지; 이번 결과는 시간 계약 보정이며 재학습 아님').as_dict(),
        reliability=GateOutcome('reliability',GateStatus.PASS,'기록·코드·원자료·원격 상태 일치',
            evidence=dict(runtime=runtime,dashboard=board,claim_alignment=claims)).as_dict(),
        scenario_validity=GateOutcome('scenario_validity',GateStatus.FAIL,
            '양하 하한 오류 수정; 운반·인계 실측 근거 및 나머지 환경 계약 미확정',
            reasons=['YR-332-i active transport allocation and timing data missing',
                     'YR-332-i achievable deadlines and LOAD approximation unconfirmed',
                     'YR-332-f appointment realism', 'YR-332-ab teacher action contract']).as_dict())
    verdict = dict(task='YR-332-i',discharge_bound_correction='PASS',
                   calibration='UNCONFIRMED',training_run=False,common_gates=gates)
    save(OUT/'final-dashboard-audit.json',board)
    save(OUT/'final-verdict.json',attach_common_gates(verdict,report_from_dict(verdict)))
    print('Timing correction verified; 138 tests; calibration remains unconfirmed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pin')
    args=p.parse_args()
    finalize(args.pin) if args.pin else report()
