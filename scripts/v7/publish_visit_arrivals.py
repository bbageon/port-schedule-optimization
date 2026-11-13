"""YR-332-ak human-readable report and common research gates (no edits to frozen inputs)."""
import argparse
import json
from pathlib import Path

from benchmark_default_eval import save
from six_step_learning import sha
from evaluate_visit_arrivals import OUT, VARIANTS, LOADS, RULES, SEEDS, read

LABEL = {'observed_mix_hours': '① 고른 간격(aj)', 'observed_hours_poisson': '② 무작위 간격',
         'observed_hours_visit_minutes': '③ 실제 방문 몰림'}
FLOW = {'GATE_IN': '반입', 'GATE_OUT': '반출', 'VESSEL_DISCHARGE': '양하', 'VESSEL_LOAD': '적하'}
ARM = {'SF_SPT': '실작업 중 짧은 작업 우선(SF_SPT)', 'FIFO': '먼저 통지된 작업 우선', 'LIFO': '나중 통지된 작업 우선',
       'SPT': '짧은 작업 우선', 'NEAREST': '가까운 작업 우선', 'LWKR': '오래 기다린 트럭 우선', 'RANDOM': '무작위', 'initial': '학습 전 모델',
       'trained': '학습 후 모델'}


def markdown(path, lines):
    assert len(lines) <= 200, (path, len(lines))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')


def signed(x):
    return f'{x:+.2f}%'


def report():
    s, ver = read(OUT / 'summary.json'), read(OUT / 'verification.json')
    u, p = read(OUT / 'visit-unit-audit.json'), read(OUT / 'arrival-process-audit.json')
    pre = read(OUT / 'input-preflight.json');rows = {(r['variant'], r['load']): r for r in pre['rows']}
    g = s['trained_gap_percent'];e = s['environment_change_percent'];b = s['by_variant']
    mm = lambda v, a, f='TRUCK': b[v][a][f]['mean_min']
    rel = u['relations'];sup = u['superseded'];can = u['cancellations']
    A, B, C = VARIANTS
    change = [e[A + '_to_' + C]['ALL'][arm]['TRUCK'] for arm in (*RULES, 'initial', 'trained')]
    bl = s['block_level'];syn = [x['60']['index'] for x in bl['synthetic'].values()]
    lines = ['# 실제 트럭 방문 단위와 시간대 안 도착 몰림 보정 후 고정 정책 비교', '',
        f'시간대 안 도착을 실제 트럭 방문처럼 몰리게 바꿔도 평균 트럭 시간은 정책마다 {min(change):+.2f}~{max(change):+.2f}%만 변했다(전체 평균).',
        f'학습 후 모델은 SF_SPT보다 {signed(g[A]["ALL"]["trained_vs_SF_SPT"]["TRUCK"])} → {signed(g[C]["ALL"]["trained_vs_SF_SPT"]["TRUCK"])} 길고, '
        f'학습 전보다 {signed(g[A]["ALL"]["trained_vs_initial"]["TRUCK"])} → {signed(g[C]["ALL"]["trained_vs_initial"]["TRUCK"])}이다. 결론은 바뀌지 않았다.',
        f'게이트에서 몰린 트럭은 21개 블록으로 흩어져 각 크레인에 거의 무작위로 도착한다(블록별 1분 몰림: 합성 {min(syn):.2f}~{max(syn):.2f}, 실제 트럭 방문 {bl["real"]["truck_visits"]["60"]["index"]:.2f}).',
        f'실제 블록별 오더 몰림({bl["real"]["orders"]["60"]["index"]:.2f})은 한 트럭이 같은 블록에 컨테이너 여러 개를 가져오는 효과로, 별도 연결 축(al)이다.',
        '저장 모델과 규칙을 고정한 환경 민감도 진단이며, 새 학습·현장 성능 주장이 아니다. 백분율의 +는 시간이 더 길다는 뜻이다.', '',
        '## 1. 원자료에서 확인한 단위', '',
        '| 항목 | 확인 결과 | 이번 처리 |', '|---|---|---|',
        f'| `visit_id` | 진입 {u["visit_id"]["entered"]:,}건 모두 `진입시각_트럭_컨테이너_방향` = 오더 키 | 트럭 지표는 오더별 진입→작업 완료 유지 |',
        f'| 게이트 재처리 기록 | 사건 없는 기록 뒤 30분 안 재진입 {sup["superseded_records"]:,}개(같은 컨테이너 {sup["same_container_as_next"]:,}) | 뒤 기록에 붙여 실제 방문 {u["physical_visits"]:,}번 |',
        f'| 여러 오더 방문 | {rel["rows_per_visit"]["2"] + rel["rows_per_visit"]["3"] + rel["rows_per_visit"]["4"]:,}번, 출차 같음 {rel["gate_out_within_visit"]["same_nonempty"]:,} | 합성은 오더 1건 = 트럭 1대 유지(연결 미구현) |',
        f'| 방문 안 순서 | 같은 블록 반입+반출: 반입 완료 먼저 {rel["observed_sequence"]["IN+OUT|same_block|IN_done_first"]:,} 대 {rel["observed_sequence"]["IN+OUT|same_block|OUT_done_first"]} | 관측값으로만 기록, 제약 미구현 |',
        f'| 진입 후 취소 | {can["entered_cancelled"]}건(게이트 관리자 {can["reasons"]["Cancel By Gate Manager"]}), 진입 전 출차 {can["classes"]["cancelled_before_block_entry_truck_exited"]} | 행 보존, 물리 취소 미구현(0.6%) |',
        f'| aj 보정값 영향 | 재처리 기록 제거 시 반출 비중 최대 {max(abs(u["aj_marginal_effect"][k]["gate_out_share_change_pp"]) for k in ("without_pure_duplicates", "without_all_superseded_rows")):.2f}%p | aj 보정값 유지 |', '',
        '합성 트럭 한 대를 실제 방문 한 번에 대응시켰다. 실제 오더 단위 몰림에는 한 트럭의 여러 컨테이너가 같은 시각에 찍힌 효과가 들어 있다.', '',
        '## 2. 도착 몰림: 실제와 합성 입력', '',
        '몰림 지수는 각 시각 안 같은 폭 구간 건수의 분산/평균이다. 1이면 무작위 도착, 클수록 몰린다.',
        '간격 변동계수는 시간당 물량으로 나눈 도착 간격의 들쭉날쭉함이다(무작위 1).', '',
        '| 입력 | 물량 | 1분 몰림 | 10분 몰림 | 간격 변동계수 | 반입·반출 1분 상관 |', '|---|---:|---:|---:|---:|---:|',
        f'| 실제 방문 | 13일 | {p["dispersion"]["visits"]["60"]:.3f} | {p["dispersion"]["visits"]["600"]:.3f} | {p["gaps"]["cv"]:.3f} | {p["co_arrival_60s"]["mean_correlation"]:.3f} |',
        f'| 실제 오더(참고) | 13일 | {p["dispersion"]["orders"]["60"]:.3f} | {p["dispersion"]["orders"]["600"]:.3f} | - | - |']
    for variant in VARIANTS:
        for load in LOADS:
            st_ = rows[(variant, load)]['stats']
            lines.append(f'| {LABEL[variant]} | {load:,} | {st_["dispersion"]["60"]:.3f} | {st_["dispersion"]["600"]:.3f} | {st_["gaps"]["cv"]:.3f} | {st_["co_arrival_60s"]["mean_correlation"]:.3f} |')
    lines += ['', '③의 3,500건은 1분·10분 몰림을 맞췄고, 간격 변동계수는 기준(1.176±0.10)을 0.003 넘겨 공개한 한계다.',
        '7,500·15,000건은 실제 날짜를 여러 개 겹쳐 몰림이 실측보다 커진다. 실측 보정이 아니라 강한 몰림의 민감도 시험이다.',
        '실제 반입·반출 상관에는 한 트럭의 반입+반출이 들어 있어 이번 축으로 맞추지 않았다.', '',
        '## 3. 크레인이 보는 도착: 블록별 몰림', '',
        '같은 지수를 날짜·시각·블록별로 나눠 쟀다. 무작위로 흩어지면 1분 0.983, 10분 0.833이 기대값이다.', '',
        '| 입력 | 1분 몰림 | 10분 몰림 |', '|---|---:|---:|',
        f'| 실제 오더 | {bl["real"]["orders"]["60"]["index"]:.3f} | {bl["real"]["orders"]["600"]["index"]:.3f} |',
        f'| 실제 트럭 방문 | {bl["real"]["truck_visits"]["60"]["index"]:.3f} | {bl["real"]["truck_visits"]["600"]["index"]:.3f} |']
    for variant in VARIANTS:
        x = [bl['synthetic'][f'{variant}:{load}'] for load in LOADS]
        lines.append(f'| {LABEL[variant]} (3,500/7,500/15,000) | ' + '/'.join(f'{y["60"]["index"]:.3f}' for y in x) + ' | '
                     + '/'.join(f'{y["600"]["index"]:.3f}' for y in x) + ' |')
    lines += ['', '합성 무대는 블록을 도착 시각과 무관하게 정하므로 게이트 몰림이 블록마다 묽어진다.',
        '실제 자료도 트럭 방문 단위로는 블록별 도착이 무작위 수준이다. 남은 차이는 같은 트럭의 여러 컨테이너다.', '',
        '## 4. 조건별 전체 결과', '',
        '| 조건 | 학습 전(분) | 학습 후(분) | SF_SPT(분) | 학습 후 vs SF_SPT | 학습 후 vs 학습 전 |', '|---|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        lines.append(f'| {LABEL[variant]} | {mm(variant, "initial"):.3f} | {mm(variant, "trained"):.3f} | {mm(variant, "SF_SPT"):.3f} | '
                     f'{signed(g[variant]["ALL"]["trained_vs_SF_SPT"]["TRUCK"])} | {signed(g[variant]["ALL"]["trained_vs_initial"]["TRUCK"])} |')
    phi = {x: 100 * (b[x]['trained']['mean_phi_krw'] / b[x]['SF_SPT']['mean_phi_krw'] - 1) for x in VARIANTS}
    lines += ['', '트럭 진입→작업 완료 평균(24시간 미완료 포함), 오더 수 가중. 15,000건이 전체 오더의 57.69%를 차지한다.',
        f'모형 비용(네 항 합계)으로도 학습 후 모델은 SF_SPT보다 ① {signed(phi[A])}, ② {signed(phi[B])}, ③ {signed(phi[C])} 높다. 실제 원화 지출 예측이 아니다.', '',
        '## 5. 물량별 결과', '',
        '| 조건 | 물량 | 학습 전(분) | 학습 후(분) | SF_SPT(분) | 학습 후 vs SF_SPT | 학습 후 미완료(평균) | SF_SPT 미완료 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        for load in LOADS:
            d = s['by_load'][variant][str(load)]
            lines.append(f'| {LABEL[variant]} | {load:,} | {d["initial"]["TRUCK"]["mean_min"]:.3f} | {d["trained"]["TRUCK"]["mean_min"]:.3f} | '
                         f'{d["SF_SPT"]["TRUCK"]["mean_min"]:.3f} | {signed(g[variant][str(load)]["trained_vs_SF_SPT"]["TRUCK"])} | '
                         f'{d["trained"]["mean_unfinished_trucks"]:.2f} | {d["SF_SPT"]["mean_unfinished_trucks"]:.0f} |')
    low = s['by_load'][C]['3500'];top = s['ranking_by_mean_truck_time'][C]['3500']
    lines += ['', f'3,500건 ③에서는 SF_SPT가 {low["SF_SPT"]["TRUCK"]["mean_min"]:.3f}분으로 늘어 학습 후 모델과의 격차가 줄었다.',
        f'같은 조건에서 평균이 가장 짧은 정책: {ARM[top[0]]} {low[top[0]]["TRUCK"]["mean_min"]:.3f}분. 규칙 간 차이가 1분 안쪽이고 입력이 하나라 우열로 읽지 않는다.',
        '', '## 6. 같은 정책의 조건 간 변화(환경 효과)', '',
        '| 정책 | ①→② 전체 | ②→③ 전체 | ①→③ 전체 | ①→③ 3,500건 |', '|---|---:|---:|---:|---:|']
    for arm in ('SF_SPT', 'FIFO', 'NEAREST', 'initial', 'trained'):
        lines.append(f'| {ARM[arm]} | {signed(e[A + "_to_" + B]["ALL"][arm]["TRUCK"])} | {signed(e[B + "_to_" + C]["ALL"][arm]["TRUCK"])} | '
                     f'{signed(e[A + "_to_" + C]["ALL"][arm]["TRUCK"])} | {signed(e[A + "_to_" + C]["3500"][arm]["TRUCK"])} |')
    lines += ['', '같은 정책이 다른 도착 흐름을 받은 결과다. 이 변화를 학습 성능의 변화로 읽지 않는다.',
        '물량별 입력이 하나라 ±수 %의 물량별 변화는 우연 변동과 구분되지 않는다.', '',
        '## 7. 작업 유형별(①과 ③)', '',
        '| 작업 | 조건 | 학습 전(분) | 학습 후(분) | SF_SPT(분) | 학습 후 vs SF_SPT |', '|---|---|---:|---:|---:|---:|']
    for flow, label in FLOW.items():
        for variant in (A, C):
            lines.append(f'| {label} | {LABEL[variant]} | {mm(variant, "initial", flow):.3f} | {mm(variant, "trained", flow):.3f} | '
                         f'{mm(variant, "SF_SPT", flow):.3f} | {signed(g[variant]["ALL"]["trained_vs_SF_SPT"][flow])} |')
    lines += ['', '양하·적하는 스트림 계획 시작→개별 야드 완료 시간이며 선박 체류시간이 아니다.', '',
        '## 8. 규칙 7개·모델 2종의 평균(분)', '',
        '| 정책 | ' + ' | '.join(LABEL[v] for v in VARIANTS) + ' |', '|---|---:|---:|---:|']
    for arm in (*RULES, 'initial', 'trained'):
        lines.append(f'| {ARM[arm]} | ' + ' | '.join(f'{mm(v, arm):.3f}' for v in VARIANTS) + ' |')
    lines += [''] + [f'짧은 순 상위 4개 — {LABEL[v]}: ' + ' > '.join(ARM[a] for a in s['ranking_by_mean_truck_time'][v]['ALL'][:4]) + '.'
                     for v in VARIANTS] + ['',
        '| 저장 모델 초기값 | ' + ' | '.join(f'{LABEL[v]} 학습 후 vs 학습 전' for v in VARIANTS) + ' |', '|---|---:|---:|---:|']
    for seed in SEEDS:
        cells = []
        for v in VARIANTS:
            d = s['by_model'][v][str(seed)]
            cells.append(signed(100 * (d['trained']['TRUCK']['mean_min'] - d['initial']['TRUCK']['mean_min']) / d['initial']['TRUCK']['mean_min']))
        lines.append(f'| {seed} | ' + ' | '.join(cells) + ' |')
    lines += ['', '세 모델은 같은 입력을 보므로 독립된 평가일 세 개로 세지 않는다.', '',
        '## 9. 사전등록 정정 경위', '',
        '- 처음 등록한 무작위 날짜 순서는 3,500건 간격 변동계수에서 0.003 미달했다.',
        '- 수정안(방문이 가장 많은 날짜부터)은 몰림이 컸던 시각을 고르는 선택 편향으로 1분·10분 몰림이 미달했다(10분 1.577).',
        '- 두 결정 모두 정책 실행 전, 입력 통계만 보고 내렸다. 수정안을 기각하고 원래 방법으로 되돌렸다.', '',
        '## 10. 검증과 한계', '',
        f'- 평가 {ver["evaluations"]}회(고유 {ver["unique_evaluations"]}·반복 {ver["exact_repeats"]}), 조건 ① {ver["aj_exact_reproductions"]}회는 aj 저장 결과와 비트 단위로 같다.',
        f'- 트럭 장부 {ver["truck_rows"]:,}행·전체 작업 {ver["all_order_rows"]:,}행을 저장 시각에서 다시 계산했다. 관련 시험 {ver["tests_passed"]}건 통과.',
        f'- 실행 코드 `{ver["source_commit"][:8]}`, 코드·설정·가중치 {ver["committed_source_hashes"]}개 해시 확인. 본선 계획은 세 조건에서 같다.',
        '- 물량별 입력이 하나다. 통계적 확증·실제 터미널 성능을 주장하지 않는다. 13일은 개발용 자료다.',
        '- 남은 축: 여러 오더 방문의 연결, 날짜 간 물량 변동, 반입/반출 시간대 결합, 본선 운반(i), 대기열 정의(f).',
        '- CPU 하루 실행의 명시적 옵션이다. 월/배열 경로에는 연결하지 않았다.', '',
        '[정의 감사](visit-unit-audit.json) · [실제 몰림](arrival-process-audit.json) · [입력 판정](input-preflight.json) · '
        '[전체 원값](summary.json) · [검산](verification.json) · [실행 설정](campaign.json) · [최종 판정](current-gates.json) · '
        '[사전등록](../../../docs/research/v7-discharge/visit-arrival-prereg.md).']
    markdown(OUT / 'README.md', lines)
    print(len(lines), 'lines')


def gates(commit):
    from yard_rl.experiments.gate_harness import (audit_dashboard, judge_runtime_evidence, judge_claim_alignment,
        combine_reliability, judge_scenario_validity, GateOutcome, GateStatus, ResearchGateReport, attach_common_gates)
    s, v, m = read(OUT / 'summary.json'), read(OUT / 'verification.json'), read(OUT / 'campaign.json')
    paths = [OUT / f for f in ('summary.json', 'verification.json', 'campaign.json', 'README.md', 'authorization.json',
        'visit-unit-audit.json', 'arrival-process-audit.json', 'visit-arrival-templates.json', 'demand-profiles.json',
        'input-preflight.json', 'input-preflight-draft-random-order.json', 'input-preflight-amendment-busiest-first.json',
        'contract-tests.xml')]
    paths += sorted((OUT / 'evaluation').glob('*.gz')) + sorted((OUT / 'orders').glob('*.gz')) + sorted((OUT / 'flow-orders').glob('*.gz'))
    runtime = judge_runtime_evidence(m['repro'], artifact_hashes={p.as_posix(): sha(p) for p in paths})
    spec = Path('.claude/docs/dashboard-task-specs/YR-332-ak-order-visit-and-arrival-process.md')
    dashboard = audit_dashboard('.', task_id='YR-332-ak', expected_state='done', spec_path=spec,
        evidence_paths=[OUT / 'summary.json', OUT / 'verification.json', OUT / 'campaign.json'], evidence_commits=[commit],
        remote_ref='origin/강화학습-판매', pin_commit=commit)
    text = (OUT / 'README.md').read_text(encoding='utf-8');claims, truth = {}, {}
    for variant in VARIANTS:
        for arm in ('initial', 'trained', 'SF_SPT'):
            number = s['by_variant'][variant][arm]['TRUCK']['mean_min']
            assert f'{number:.3f}' in text
            claims[variant + ':' + arm] = round(number, 3);truth[variant + ':' + arm] = number
        for scope in ('ALL', '3500'):
            gap = s['trained_gap_percent'][variant][scope]['trained_vs_SF_SPT']['TRUCK']
            assert f'{gap:+.2f}%' in text
            claims[f'{variant}:{scope}:gap'] = round(gap, 2);truth[f'{variant}:{scope}:gap'] = gap
    reliability = combine_reliability(runtime, dashboard, judge_claim_alignment(claims, truth, absolute_tolerance=.00501))
    performance = GateOutcome('performance', GateStatus.INCONCLUSIVE,
        '고정 정책의 도착 몰림 민감도 진단이며 정책 성능 확증을 수행하지 않음',
        ('물량별 신규 입력 하나이며 독립 실자료 기간 미확보',),
        dict(metric='terminal_total_cost', baseline='SF-SPT', new_training=False, no_adoption=True,
             prior_policy_gate='FAIL: outputs/reports/yr332_completion_learning/current-gates.json',
             conditional_results=s['trained_gap_percent']))
    scenario = judge_scenario_validity(internal_checks=dict(event_time_order=True, information_boundary=True,
        physical_constraints=True, ledger_conservation=True, achievable_vessel_deadline=False, deterministic_replay=True),
        flow_checks={}, anchors={}, continuous_operation=False, request_real_terminal_claim=False)
    result = attach_common_gates(dict(task='YR-332-ak', evidence_commit=commit, verification=v,
        claim_scope='Observed visit units fixed; within-hour visit clustering reproduced at 3,500 (one disclosed gap-variability miss); larger loads are bunching sensitivity; multi-order visit links, day-to-day variation, flow-hour coupling, transport (i) and queue definition (f) remain.'),
        ResearchGateReport(performance, reliability, scenario))
    save(OUT / 'dashboard-audit.json', dashboard.as_dict());save(OUT / 'current-gates.json', result)
    assert reliability.status is GateStatus.PASS, reliability.as_dict()
    print(json.dumps({k: r['status'] for k, r in result['common_gates'].items() if isinstance(r, dict) and 'status' in r}))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('command', choices=('report', 'gates'));p.add_argument('--commit')
    args = p.parse_args()
    report() if args.command == 'report' else gates(args.commit)
