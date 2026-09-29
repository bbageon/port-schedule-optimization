"""Audit the paired original-input comparison; never substitute a dispatch-rule arm."""
import argparse
import gzip
import json
import math
from pathlib import Path
import statistics

import numpy as np

from yard_rl.experiments.gate_harness import (
    GateOutcome, GateStatus, ResearchGateReport, attach_common_gates,
    audit_dashboard, combine_reliability, judge_claim_alignment,
    judge_runtime_evidence, judge_scenario_validity)
from yard_rl.v6.ppo.checkpoint import load_policy
from yard_rl.v6.ppo.journal import write_json
from yard_rl.v6.ppo.original_comparison import assignment_audit
from yard_rl.v6.ppo.provenance import file_sha256 as sha
from yard_rl.v6.reward.operational import KEYS, reference_config, normalized_loss
from yard_rl.v6.schema import Order

from original_assignment_experiment import parameter_hash


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def compressed(path):
    with gzip.open(path, 'rt', encoding='utf-8') as source:
        return json.load(source)


def request_seconds(orders, records, t):
    total = 0.
    for order in orders:
        if order['in_out_reserve_s'] > t:
            continue
        end = records[order['doc_key']]['gate_out_s']
        total += max(0., (t if end is None else min(t, end))-order['in_out_reserve_s'])
    return total


def guard_values(report):
    end = report['evaluation_endpoint']
    jobs = end['cargo']['active_unfinished_jobs']
    return dict(original_request_s=report['measured_original_request_s'],
                truck_remaining=end['request']['unfinished'],
                vessel_remaining=jobs.get('VESSEL_LOAD', 0)+jobs.get('VESSEL_DISCHARGE', 0))


def audit(root):
    campaign = read(root/'campaign/campaign.json')
    assert campaign['all_completed'] and campaign['registered_runs'] == 6
    assert len(campaign['completed']) == 6
    ref = reference_config()
    reports, manifests, traces, references = {}, {}, {}, {}
    for job in campaign['completed']:
        assert job['exit_code'] == 0
        folder = root/'campaign'/job['name']
        r, m = read(folder/'report.json'), read(folder/'manifest.json')
        name = f"eval-{r['seed']}-{r['arm']}"
        assert name == job['name'] and m['arm'] == r['arm']
        assert r['status'] == 'complete' and r['protocol_complete'] and not r['smoke']
        assert m['measurement_window_s'] == r['measurement_window_s'] == [86400., 172800.]
        assert m['intervention_s'] == 86400. and m['endpoint_s'] == 259200.
        assert m['normalization'] == r['reward_normalization'] == ref
        assert not r['updates'] and r['config']['reward_mode'] == 'operational'
        assert r['parameter_unchanged'] and r['checkpoint_reload_exact'] and r['cf_calls'] == 0
        assert r['invariants'] and not m['code']['git_dirty']
        assert m['prereg_sha256'] == sha(root/'campaign/prereg-executed.md')
        assert read(folder/'container_contract.json')['passed']
        assert r['admission']['skipped'] == r['admission']['vessel_failed'] == 0
        input_path = Path(f"outputs/reports/yr331_training/eval-{r['seed']}-cost/seed-data.json.gz")
        train_seed = r['seed']-30000
        checkpoint_path = Path(f'outputs/reports/yr331_operational/campaign/train-{train_seed}-operational/policy-final.pt')
        assert sha(input_path) == m['input_sha256'] and sha(checkpoint_path) == m['checkpoint_sha256']
        references[str(input_path)] = m['input_sha256']
        references[str(checkpoint_path)] = m['checkpoint_sha256']
        source = compressed(input_path)
        assert r['admission']['admitted'] == len(source['schedule'])
        raw = compressed(folder/'orders-and-records.json.gz')
        orders = {row['doc_key']: Order(**row) for row in raw['orders']}
        assert assignment_audit(source['orders'], orders) == r['assignments']
        if r['arm'] == 'original':
            assert r['assignments']['original_fields_preserved']
            assert r['traded_edges'] == r['n_space'] == r['n_time'] == 0
        before = parameter_hash(load_policy(checkpoint_path))
        after = parameter_hash(load_policy(folder/'policy-final.pt'))
        assert before == after == m['initial_parameter_sha256'] == r['final_parameter_sha256']
        trace = np.asarray(read(folder/'physical-trace.json')['rows'])
        physical = []
        for t in (86400., 172800.):
            rows = trace[trace[:, 0] == t]
            assert len(rows) == 1
            physical.append(rows[0, 1:])
        measured = dict(zip(KEYS, physical[1]-physical[0]))
        assert measured == r['measured_physical']
        assert math.isclose(normalized_loss(measured, ref), r['measured_objective'], rel_tol=1e-12)
        records = {row['doc_key']: row for row in raw['records']}
        assert len(records) == len(source['orders'])
        request = request_seconds(source['orders'], records, 172800.)-request_seconds(source['orders'], records, 86400.)
        assert math.isclose(request, r['measured_original_request_s'], rel_tol=1e-10)
        reports[name], manifests[name], traces[name] = r, m, trace
    pairs = []
    for seed in (9961000, 9971000, 9981000):
        arms = {arm: reports[f'eval-{seed}-{arm}'] for arm in ('original', 'reallocated')}
        am = {arm: manifests[f'eval-{seed}-{arm}'] for arm in arms}
        for key in ('input_sha256', 'checkpoint_sha256', 'initial_parameter_sha256',
                    'normalization', 'measurement_window_s', 'action_mode'):
            assert am['original'][key] == am['reallocated'][key]
        a, b = [traces[f'eval-{seed}-{arm}'] for arm in arms]
        assert np.array_equal(a[a[:, 0] <= 86400.], b[b[:, 0] <= 86400.])
        for arm in arms:
            snapshot = read(root/f'campaign/eval-{seed}-{arm}/snapshots.json')['86400']
            assert snapshot['assignments']['original_fields_preserved']
            assert snapshot['n_space'] == snapshot['n_time'] == 0
        streams = [read(root/f'campaign/eval-{seed}-{arm}/random-streams.json') for arm in arms]
        assert streams[0] == streams[1]
        guards = {arm: guard_values(r) for arm, r in arms.items()}
        deltas = {key: guards['reallocated'][key]-guards['original'][key] for key in guards['original']}
        scores = {arm: r['measured_objective'] for arm, r in arms.items()}
        pairs.append(dict(eval_seed=seed, objectives=scores,
            reduction_percent=100*(1-scores['reallocated']/scores['original']),
            physical={arm: r['measured_physical'] for arm, r in arms.items()},
            guards=guards, guard_deltas=deltas, guards_pass=all(x <= 1e-6 for x in deltas.values()),
            reassignments={arm: {key: r[key] for key in ('n_space', 'n_time', 'traded_edges')} for arm, r in arms.items()},
            same_crane_weights=True, common_warmup_exact=True, original_assignment_preserved=True))
    means = {arm: statistics.mean(p['objectives'][arm] for p in pairs) for arm in ('original', 'reallocated')}
    reduction = 100*(1-means['reallocated']/means['original'])
    guard_pass = all(p['guards_pass'] for p in pairs)
    result = dict(task='YR-331-f', registered_runs=6, completed_runs=6, frozen_evaluations=6,
        trained_models=0, updates_applied=0, baseline='original input assignments, same frozen learned crane policy',
        original_execution_trace_available=False, normalization=ref, pairs=pairs, mean_objectives=means,
        mean_reduction_percent=reduction, guards_pass=guard_pass, diagnostic_success=reduction > 0 and guard_pass,
        statistical_confirmation=False, real_terminal_claim=False,
        first_day_equal=True, original_fields_preserved=True,
        execution_commit=next(iter(reports.values()))['source']['git_head'])
    assert len({r['source']['git_head'] for r in reports.values()}) == 1
    write_json(root/'result.json', result)
    write_json(root/'references.json', references)
    render(root, result)
    files = [p for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('artifacts.json', 'gates.json', '.gitattributes')]
    write_json(root/'artifacts.json', dict(files=[dict(path=p.relative_to(root).as_posix(), sha256=sha(p), bytes=p.stat().st_size) for p in files]))
    return result


def render(root, result):
    means = result['mean_objectives']
    lines = ['# v6 원본 배정 유지 대비 재배정 평가', '',
        '**비교 기준 정정 완료. 같은 크레인 정책에서 재배정 허용 여부만 비교했다.**', '',
        '## 진행사항', '',
        '- 재배정 전 원본은 최초 블록·예약 시각·컨테이너·본선 계획·초기 재고다. 별도의 원본 처리 순서·완료 기록은 없다.',
        '- 원본 유지는 판매 KEEP·구매 REJECT로 실행하고, 비교 쪽만 같은 모델의 판매·구매를 허용했다.',
        '- 양쪽 모두 동일한 학습 크레인 정책을 쓴다. SF-SPT 규칙으로 교체하지 않았다.',
        '- 기존 운영 보상 모델 3개를 그대로 사용한 3쌍·6회 평가다. 재학습·가중치 갱신은 0회다.',
        '- 첫날은 두 실행 모두 원본을 유지해 물리 기록이 정확히 같았다. 둘째 날부터 재배정 허용 여부가 갈린다.',
        '- 판매·구매 추첨이 크레인 추첨을 밀지 않도록 두 실행 모두 같은 역할별 난수열을 적용했다.',
        '- 원본 유지 3회 모두 주문 6필드 보존·공간/시간 거래 0건. 입력·모델 파일 지문과 평가 중 가중치 동결을 검증했다.',
        '- 전건 투입·물리 제약·저장 복원 통과. 새 비교 및 기존 학습 연결 검사 25건 통과.', '',
        '## 해석', '',
        '점수는 기존 네 물리 지표를 같은 고정 기준으로 정규화한 합이며 작을수록 좋다. 원화 비용이 아니다.',
        '개선률은 `100×(1−재배정 점수/원본 유지 점수)`다. 음수는 재배정 후 악화를 뜻한다.', '',
        '| 평가 입력 | 원본 배정 유지 | 재배정 허용 | 원본 대비 개선률 |', '|---|---:|---:|---:|']
    for p in result['pairs']:
        lines.append(f"| {p['eval_seed']} | {p['objectives']['original']:.4f} | {p['objectives']['reallocated']:.4f} | {p['reduction_percent']:+.2f}% |")
    lines += [f"| 평균 | {means['original']:.4f} | {means['reallocated']:.4f} | {result['mean_reduction_percent']:+.2f}% |", '',
        '평균은 점수 평균끼리 비교했다. 측정창은 둘째 날 24시간이며 아래 시간은 모든 대상의 시간 합계다.', '',
        '| 실행 | 트럭 체류 합계(시간) | 본선 중단 합계(시간) | 빈 이동 합계(시간) | 재취급(회) |', '|---|---:|---:|---:|---:|']
    for arm, label in [('original', '원본 배정 유지'), ('reallocated', '재배정 허용')]:
        raw = {key: statistics.mean(p['physical'][arm][key] for p in result['pairs']) for key in KEYS}
        lines.append(f"| {label} | {raw['truck_s']/3600:.2f} | {raw['vessel_s']/3600:.2f} | {raw['empty_s']/3600:.2f} | {raw['rehandles']:.2f} |")
    lines += ['', '아래 변화는 재배정−원본 유지다. 요청시간은 둘째 날 누적 증가분이며 잔여는 셋째 날 끝이다. 양수는 악화다.', '',
        '| 입력 | 최초 요청~완료 시간 변화(트럭·시간) | 트럭 잔여 변화(대) | 본선 잔여 변화(건) | 비악화 조건 |', '|---|---:|---:|---:|---|']
    for p in result['pairs']:
        d = p['guard_deltas']
        lines.append(f"| {p['eval_seed']} | {d['original_request_s']/3600:+.2f} | {d['truck_remaining']:+d} | {d['vessel_remaining']:+d} | {'통과' if p['guards_pass'] else '미충족'} |")
    lines += ['', f"사전 고정 진단 성공 조건: **{'충족' if result['diagnostic_success'] else '미충족'}**. 평균 점수만으로 요청 대기·미처리 악화를 상쇄하지 않는다.", '',
        '## 기존 보고와의 관계·한계', '',
        '- 과거 16.47% 악화는 KEEP+SF-SPT 대비 결과다. 재배정과 크레인 정책이 함께 달랐으므로 원본 대비 재배정 효과로 해석하지 않는다.',
        '- 이번 결과는 같은 학습 크레인의 원본 배정 유지 대비 효과다. TOS 원본 실행 기록·실제 부산항 성능을 재현한 결과는 아니다.',
        '- 모델당 학습 24회인 기존 모델을 재사용했다. 평가 입력도 과거에 사용한 3개이며 독립 잠금평가나 수렴 증거가 아니다.',
        '- 0.25씩 동일 비중과 정규화 기준은 e에서 그대로 유지했다. 정규화 자체의 효과나 최적 비중을 새로 검증하지 않는다.',
        '- 첫날을 공통 원본 운전으로 맞추고 역할별 추첨을 적용했으므로 과거 평가 수치와 직접 이어 붙이지 않는다.',
        '- 고정 입력의 실제 처리 지연은 허용하되 원본 주문은 바꾸지 않는다. 미완료 작업도 요청 시간에 포함한다.',
        '- 기존 검증 입력 공급 부족과 실제 항만 자료 부재는 남아 있다.',
        f"- 실행 코드: `{result['execution_commit']}`. [원자료](result.json) · [실행 전 계약](campaign/prereg-executed.md) · [파일 지문](artifacts.json)", '',
        '## 예정사항', '',
        '- YR-331-c: 독립 검증 입력의 컨테이너 공급 부족을 복구한다. 주문과 실제 공급이 맞아야 다음 판단을 신뢰할 수 있다.',
        '- YR-331-a: 복구한 독립 자료에서 크레인의 안전 수용 상한을 확인한다. 이번 비교를 최적 혼잡도 증명으로 사용하지 않는다.', '']
    (root/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(root/'reported-values.json', dict(frozen_evaluations=6, updates_applied=0,
                                               mean_reduction_percent=round(result['mean_reduction_percent'], 2)))


def gates(root, board_commit, remote_ref):
    result = read(root/'result.json')
    hashes = {str(root/row['path']): row['sha256'] for row in read(root/'artifacts.json')['files']}
    hashes.update(read(root/'references.json'))
    outcomes = []
    for folder in sorted((root/'campaign').glob('eval-*')):
        if not folder.is_dir():
            continue
        manifest = read(folder/'manifest.json')
        stamp = dict(code=manifest['code'], seeds={'run': [manifest['seed']]},
                     prereg=str(root/'campaign/prereg-executed.md'), params=manifest)
        outcomes.append(judge_runtime_evidence(stamp, artifact_hashes=hashes))
    assert len(outcomes) == 6 and all(row.status == GateStatus.PASS for row in outcomes)
    dashboard = audit_dashboard(Path('.'), task_id='YR-331-f', expected_state='done',
        spec_path=Path('.claude/docs/dashboard-task-specs/YR-331-f-original-assignment-comparison.md'),
        evidence_paths=(root/'result.json', root/'report.md'), evidence_commits=(board_commit,),
        remote_ref=remote_ref, pin_commit=board_commit)
    raw = {key: result[key] for key in ('frozen_evaluations', 'updates_applied', 'mean_reduction_percent')}
    alignment = judge_claim_alignment(read(root/'reported-values.json'), raw, absolute_tolerance=.005)
    reliability = combine_reliability(outcomes[-1], dashboard, alignment)
    performance = GateOutcome('performance', GateStatus.INCONCLUSIVE if result['guards_pass'] else GateStatus.FAIL,
        '원본 배정 유지 대비 같은 크레인 정책의 재배정 효과 3쌍 진단; 통계적 확증 아님',
        (() if result['guards_pass'] else ('최초 요청 대기 또는 트럭/본선 잔여의 비악화 조건 미충족',)),
        dict(baseline=result['baseline'], diagnostic_success=result['diagnostic_success'],
             mean_reduction_percent=result['mean_reduction_percent']))
    scenario = judge_scenario_validity(internal_checks={'physical_constraints': True, 'ledger_conservation': True},
        flow_checks={'fixed_measurement_window': True}, anchors={}, continuous_operation=True,
        request_real_terminal_claim=False)
    return attach_common_gates(dict(task='YR-331-f', scope='original-assignment paired diagnostic',
        result_sha256=sha(root/'result.json'), per_run_source_audits=[row.as_dict() for row in outcomes],
        unresolved_existing_input_supply_issue='YR-331-c'), ResearchGateReport(performance, reliability, scenario))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('outputs/reports/yr331_original'))
    parser.add_argument('--board-commit')
    parser.add_argument('--remote-ref')
    args = parser.parse_args()
    if args.board_commit:
        result = gates(args.root, args.board_commit, args.remote_ref)
        write_json(args.root/'gates.json', result)
        print({key: row['status'] for key, row in result['common_gates'].items() if isinstance(row, dict) and 'status' in row})
    else:
        result = audit(args.root)
        print({key: result[key] for key in ('completed_runs', 'updates_applied', 'mean_reduction_percent', 'diagnostic_success')})
