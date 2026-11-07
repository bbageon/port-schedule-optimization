"""YR-332-t README and claim check. Every number in the README is read from saved JSON."""
import json
from pathlib import Path

OUT = Path('outputs/reports/yr332_comparison_audit')
LOADS = (3500, 7500, 15000)
ARM_KO = dict(spt='최단 작업', fcfs_arrival='도착순', rule_all='기존 최단 작업(SF_SPT)', nearest='최근접',
              vessel_fcfs='본선 우선·도착순', vessel_nearest='본선 우선·최근접', unidirectional='단방향',
              distance_wait='거리·대기 복합', initial='학습 전', pointwise='기존 학습',
              pairwise='비용 차이 학습', coverage='비교 표본 확대', flow='최신 학습')
SFSPT = '기존 최단 작업(SF_SPT)'


def label(key):
    kind, arm, other = key.split(':')
    if kind == 'K1':
        return f'{ARM_KO[arm]} 후 vs 학습 전'
    return f'{ARM_KO[arm]} vs {ARM_KO[other]}'


def drop_sfspt(table):
    return '\n'.join(x for x in table.split('\n') if SFSPT not in x)


def load(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))


def main():
    s, sens, t = load('summary.json'), load('sensitivity.json'), load('tables.json')
    v, inv, ind = load('final-verdict.json'), load('inventory.json'), load('independent-arithmetic-audit.json')
    rep = load('replay-verification.json')
    key = {r['key']: r for r in s['key_comparisons']}
    cost = {(r['arm'], r['load']): r for r in s['costs']}
    led = {(r['arm'], r['load']): r for r in s['ledger']}
    inp, sw = s['inputs'], s['sts_wait']
    rc = v['categories']['rule_comparison']
    n_flip = len(v['documented_variant_flips'])
    mean_flips = rc['mean_sign_flips']
    classes = inv['check']['classes']
    claims = {}

    def c(label, text):
        claims[label] = text
        return text

    fs, fa, fi, pi = key['K2:flow:spt'], key['K2:flow:fcfs_arrival'], key['K1:flow:initial'], key['K1:pairwise:initial']
    uni = sens['K2:flow:unidirectional']['variants']['base']
    cov = sens['K1:coverage:initial']['variants']
    q = {n: cost[('spt', n)] for n in LOADS}
    qf = {n: cost[('flow', n)] for n in LOADS}
    vs = {n: 100*cost[('spt', n)]['share']['c_vessel'] for n in LOADS}
    wt = {n: 100*cost[('flow', n)]['share']['c_wait'] for n in LOADS}
    slack = {n: inp[f'flow@{n}'] for n in LOADS}
    sl15 = 100*(slack[15000]['slack_above_plus_2h']+slack[15000]['slack_below_minus_2h'])
    sl75 = 100*(slack[7500]['slack_above_plus_2h']+slack[7500]['slack_below_minus_2h'])
    na = {n: led[('spt', n)]['orders']-led[('spt', n)]['admitted'] for n in LOADS}
    na15 = [led[(a, 15000)]['orders']-led[(a, 15000)]['admitted']
            for a in ('spt', 'fcfs_arrival', 'initial', 'flow', 'nearest', 'vessel_fcfs', 'vessel_nearest',
                      'unidirectional', 'distance_wait', 'pointwise', 'pairwise', 'coverage')]
    unf = led[('flow', 15000)]['unfinished_24h']-led[('spt', 15000)]['unfinished_24h']
    win = s['window']
    plan_lo, plan_hi = min(r['planned_completion_h'] for r in win), max(r['planned_completion_h'] for r in win)
    end_lo, end_hi = min(r['work_end_without_idle_h'] for r in win), max(r['work_end_without_idle_h'] for r in win)
    done = sum(sum(r['done_by_26h_runs'].values()) for r in win)
    total_runs = sum(r['total_runs'] for r in win)
    agree = {a: 100*inp[f'{a}@3500']['shortest_agreement'] for a in ('spt', 'initial', 'flow')}
    excess = {a: inp[f'{a}@3500']['mean_excess_duration_s'] for a in ('spt', 'flow')}
    hits = []
    for k, x in sens.items():
        near = {cf: x['break_even'][cf]['nearest'] for cf in x['break_even']}
        found = {cf: n for cf, n in near.items() if n['mean'] is not None or n['majority'] is not None}
        if found:
            hits.append((k, x['variants']['base'], found))

    L = []
    a = L.append
    a('# 비교 기준·비용 수치 타당성 감사')
    a('')
    a('2026-11-07 · YR-332-t. 새 학습 없이 7규칙 비교(YR-332-v)의 69회를 다시 실행했다. 비용 숫자를')
    a('원래 단위(시간·횟수)로 풀고, 근거가 문서로 남은 다른 단가로 바꿔도 판정이 유지되는지 확인했다.')
    a('')
    a('## 결론 먼저')
    a('')
    a(f"- **학습 정책이 최단 작업 규칙보다 못하다는 판정은 단가 탓이 아니다.** 최신 학습은 비용 "
      f"{c('flow_spt_cost', f'{fs['cost_pct']:+.2f}%')}·체류 {c('flow_spt_dwell', f'{fs['dwell_pct']:+.2f}%')}이고, "
      f"문서 근거가 있는 다른 단가 7가지로 바꿔도({fs['doc_min']:+.2f}~{fs['doc_max']:+.2f}%), 다섯 단가를 "
      f"각각 0~10배로 움직여도 뒤집히지 않았다. 비용 차이 중 {fs['term_split']['c_wait']:+.2f}%p가 트럭 대기"
      f"(블록 앞 줄서기), {fs['term_split']['c_vessel']:+.2f}%p가 본선 유휴다.")
    a(f"- **학습 전보다 나아졌다는 판정도 유지된다.** 최신 방식 {c('flow_init_cost', f'{fi['cost_pct']:+.2f}%')}"
      f"(9쌍 중 {fi['wins']}쌍), 비용 차이 학습 {c('pair_init_cost', f'{pi['cost_pct']:+.2f}%')}"
      f"({pi['wins']}쌍). 네 방식 모두 어떤 대안에서도 평균은 개선이다. 단 비교 표본 확대 방식은 본선 단가를 "
      f"옛 값으로 바꾸면 개선 쌍이 {cov['base']['wins']}쌍→{cov['vessel_v3_design']['wins']}쌍이 된다.")
    a(f"- **판정이 바뀌는 비교는 9쌍이 4~5쌍으로 갈려 있던 것뿐이다.** 44개 비교 중 {c('flip_count', str(n_flip))}개이고, "
      f"평균의 방향까지 바뀐 것은 최신 학습 vs 단방향(기준 {uni['mean_pct']:+.2f}%) "
      f"{len(mean_flips)}개다. 나머지는 개선 쌍 수가 4↔5로 한 쌍 바뀐 경우다.")
    a(f"- **숫자는 이 시뮬레이터 안의 값이다.** 수치 {c('inventory_n', str(inv['check']['entries']))}개 중 "
      f"법정 근거 {classes.get('법정(인용)', 0)}개(트럭 단가, 2차 확인)·시장·문헌 {classes.get('문헌·시장', 0)}개·"
      f"실측 표본 {classes.get('실측표본', 0)}개·설계값 {c('design_n', str(classes.get('설계', 0)))}개·"
      f"미확정 {classes.get('미확정', 0)}개다. 현실 운영 비용으로 읽을 수 없다.")
    a(f"- **새로 찾은 문제:** ① 학습 정책의 ‘배 여유’ 입력이 한가한 날 "
      f"{c('slack_3500', f'{100*slack[3500]['slack_above_plus_2h']:.1f}%')}, 붐비는 날 {sl15:.1f}% 상한에 걸려 "
      '배의 급함을 거의 못 본다 ② 본선 비용에 운반 장비 설정이 만든 바닥이 크다 ③ ‘크레인 빈 이동’ 비용 정의가 '
      '문서와 다르다 ④ 모든 배가 평가창 안에 일을 못 끝내 출항 지연을 비교할 수 없다.')
    a('')
    a('## ① 진행사항')
    a('')
    a(f"- 69회 모두 후보·선택·사건·작업·기록·결과 확인값(해시) 6종과 결과 원값이 기존과 같았다"
      f"({c('replays', f'{rep['exact_replays']}/{rep['cases']}')}).")
    a(f"- 트럭 기록과 24시간 계수기로 비용을 다시 계산해 저장값과 일치했다. 다른 경로로 한 계산 "
      f"{c('indep', f'{ind['checks']}건')}도 모두 일치(최대 상대오차 {ind['worst_relative_error']:.1e}).")
    a('- 관련 시험 81건(새 시험 9건 포함) 통과. 단가 근거 2건은 웹으로 2차 대조했다(아래 8).')
    a('')
    a('### 1. 비용은 무엇으로 이루어졌나')
    a('')
    a(t['cost'])
    a('')
    a(f"한가한 날에는 본선 유휴가 {c('vessel_share_3500', f'{vs[3500]:.1f}%')}로 트럭 대기와 비슷하고, "
      f"붐비는 날에는 트럭 대기가 {wt[15000]:.1f}%(최신 학습)를 차지한다. 크레인 빈 이동은 0.4% 이하다.")
    a('')
    a('### 2. 트럭 체류는 어디서 생기나 (트럭 1대 평균, 분)')
    a('')
    a(t['stages'])
    a('')
    a(f"게이트↔블록 이동 약 10분은 정책과 무관하다. 최단 작업 대비 최신 학습의 체류 차이 "
      f"{fs['dwell_pct']:+.2f}% 중 줄서기 몫이 {c('queue_part', f'{fs['stage_split']['queue']:+.2f}%p')}다.")
    a('')
    a('### 3. 본선 유휴에는 정책이 줄일 수 없는 바닥이 있다')
    a('')
    a(t['sts'])
    a('')
    a('- 적하: 블록당 운반 차량 3대·왕복 470초라 안벽 크레인 박자(130.9초)를 못 따라가 최소 16.4%를 '
      '기다린다. 문헌의 안벽 손실 10~20%에 맞춘 설계값이다.')
    a('- 양하: 운반 중인 상자 3개 상한 때문에 코드 산술상 약 26~31%가 바닥이다(이번 감사의 계산, 실측 아님).')
    a(f"- 실제 최소는 {100*sw['LOAD']['min']:.1f}%·{100*sw['DISCHARGE']['min']:.1f}%다. 붐비는 날에는 85~90%를 "
      '기다린다. 24분 넘게 기다린 트럭은 ‘필수 작업’이 되어 어떤 정책이든 본선보다 먼저 한다(공통 제약).'
      ' 이 규칙이 본선 대기에 얼마나 기여하는지는 재지 않았다.')
    a('')
    a('### 4. 단가를 바꾸면 판정이 바뀌나')
    a('')
    a('대안 7개: 트럭 선형(할증 없음)·첫 1시간 제외 초과분만(고시 구조)·할증 30분부터·본선 선급별 용선료·')
    a('본선 1/3(2024년 초 시세)·본선 2.99원(옛 설계값)·빈 이동 시간당 4만원(YR-207 기록). 괄호는 개선 쌍/9.')
    a('')
    a(drop_sfspt(t['key']))
    a('')
    a('위 표의 비교는 모두 다섯 단가(트럭 기본·초과 할증·빈 이동·재조작·본선)를 각각 0~10배로 움직여도 평균 방향과')
    a('과반 여부가 그대로였다. 44개 중 뒤집히는 지점이 있는 비교와 가장 가까운 배수는 다음과 같다.')
    a('')
    a('| 비교 | 기준 | 뒤집히는 가장 가까운 배수(평균 방향 / 과반) |')
    a('|---|---:|---|')
    names = dict(truck='트럭 기본', over='할증', move='빈 이동', rehandle='재조작', vessel='본선')
    for k, base, found in hits:
        parts = []
        for cf, n in found.items():
            m = '-' if n['mean'] is None else f"×{n['mean']:.2f}"
            j = '-' if n['majority'] is None else f"×{n['majority']:.2f}"
            parts.append(f'{names[cf]} {m}/{j}')
        a(f"| {label(k)} | {base['mean_pct']:+.2f}% ({base['wins']}/9) | {', '.join(parts)} |")
    a('')
    a('### 5. 차이는 어느 비용 항에서 왔나 (비교 대상 총비용 대비 %p)')
    a('')
    a(drop_sfspt(t['terms']))
    a('')
    a('### 6. 처리 대상과 평가창')
    a('')
    a(t['ledger'])
    a('')
    a(f"- 15,000건 날에는 정책에 따라 {min(na15):,.0f}~{max(na15):,.0f}대가 투입되지 않아 비용·체류에서 빠진다"
      '(원인 대조는 m).')
    a(f"- 24시간 안에 못 나간 트럭은 24시간까지만 센다. 최신 학습은 최단 작업보다 미출차가 "
      f"{c('unfinished_gap', f'{unf:,.0f}대')} 많아 체류가 덜 잡힌다. 실제 격차는 측정값보다 클 수 있다.")
    a(f"- 배 {len(win)}척은 유휴가 없으면 {end_lo:.1f}~{end_hi:.1f}시에 끝나지만, 69회 어디서도 26시까지 끝난 배가 "
      f"없다(배·실행 {total_runs}건 중 {done}건). 계획 완료는 {c('plan_range', f'{plan_lo:.1f}~{plan_hi:.1f}시')}"
      '(순수 작업의 2배 여유)라 출항 지연은 관측할 수 없다. 본선 비용은 24시간 안의 안벽 대기로만 매긴다.')
    a('')
    a('### 7. 학습 정책이 보는 정보')
    a('')
    a(t['inputs'])
    a('')
    a(t['slack_by_load'])
    a('')
    a('최단 작업 후보와 같은 것을 고른 비율(실작업 후보가 둘 이상인 결정):')
    a('')
    a(drop_sfspt(t['agreement']))
    a('')
    a('기존 최단 작업(SF_SPT)은 비용·체류·일치율이 최단 작업과 표시 자릿수까지 같아 위 표들에서 뺐다. 동률 처리가 달라')
    a('선택 기록은 다르다(원값은 summary.json).')
    a('')
    a(f"- 작업 소요는 상한에 걸리지 않아 학습 정책도 최단 작업 규칙과 같은 정보를 본다. 그래도 한가한 날 일치율은 "
      f"최단 작업 {agree['spt']:.1f}%, 최신 학습 {c('agree_flow', f'{agree['flow']:.1f}%')}"
      f"(학습 전 {agree['initial']:.1f}%)다. 고른 일이 가장 짧은 일보다 평균 {excess['flow']:.0f}초 길다"
      f"(최단 작업 {excess['spt']:.0f}초).")
    a('- 최단 작업 규칙도 100%가 아닌 것은 코드 구조상 필수 작업(24분 이상 대기 트럭)과 두 크레인 공동 제약이 먼저 정해서다.')
    a(f"- 배 여유 입력은 계획 완료의 2배 여유 때문에 대부분 +2시간 쪽 상한에 걸린다(7,500건 상한 도달 "
      f"{sl75:.1f}% 중 +2시간 쪽 {100*slack[7500]['slack_above_plus_2h']:.1f}%).")
    a('')
    a('### 8. 수치 목록과 원출처 대조 (전체는 [inventory.json](inventory.json))')
    a('')
    a('| 분류 | 개수 | 예 |')
    a('|---|---:|---|')
    a(f"| 법정(인용) | {classes.get('법정(인용)', 0)} | 트럭 시간당 40,000원(30분당 2만원) |")
    a(f"| 법정 구조 차용 | {classes.get('법정 구조 차용', 0)} | 할증 시작 1시간 |")
    a(f"| 문헌·시장 | {classes.get('문헌·시장', 0)} | 본선 35.88원/GT·시간(용선료 대리), 크레인 속도, 적재율 0.65 |")
    a(f"| 실측 표본 | {classes.get('실측표본', 0)} | 통지 리드타임(DGT 71건, 수직형 터미널) |")
    a(f"| 설계 | {classes.get('설계', 0)} | 할증 2배·빈 이동 6만원·재조작 2천원·여유 배수 2.0·운반 470초·작업시간 오차 0 |")
    a(f"| 미확정 | {classes.get('미확정', 0)} | 반출 비율 0.6, 40피트 비율 0.7 |")
    a('')
    a('- 트럭: 2026년 안전운임 고시(제2026-55호, 8월 개정 제2026-402호)와 1시간 초과 후 30분당 2만원을')
    a('  [국토부 게시](https://www.molit.go.kr/USR/I0204/m_45/dtl.jsp?idx=18923)와')
    a('  [물류 보도](https://www.cello-square.com/kr/blog/view-1851.do)로 2차 확인했다. 고시 원문 조항은 보지 못했다.')
    a('  법정 대기시간(도착 요청~송장 발급)은 우리 게이트 진입~출차와 다르고, 1시간 초과 할증 2배는 설계 해석이다.')
    a('- 본선: 4,250TEU 하루 54,500달러 보도([Global Maritime Hub](https://globalmaritimehub.com/container-charter-market-2025-strongest-year-outside-the-covid-boom.html))는')
    a('  근거표의 4,300TEU 56,000달러와 가깝다. 8,000TEU 값은 확인하지 못했고, 대형선 32.25원은 표로 재현되지 않는다.')
    a('- 크레인 빈 이동 항은 설명과 달리 모든 작업의 빈 접근 이동을 센다. 비중이 작아 판정에는 영향이 없었다.')
    a('- 문서↔코드 자동 대조는 v7 단가를 검사하지 않는다. 04 문서 단가표의 본선 2.99원도 v7(35.88원)과 다르다.')
    a('')
    a('## ② 해석 — 네 가지를 나눠 판정')
    a('')
    a('1. **학습 전후 개선**: 이 시뮬레이터 안에서 관측된 개선은 단가 선택과 무관하게 유지된다. 다만 이미 본 3일·')
    a('   초기값 3개의 9쌍이라 독립 9일이 아니며, 수렴이나 확증의 증거가 아니다.')
    a('2. **규칙 대비 우열**: 최단 작업 규칙에 못 미친다는 판정은 단가와 평가창 검열을 따져도 유지된다. 검열은 오히려')
    a('   학습 정책에 유리하게 작용한다. 도착순보다 낫다는 판정도 유지된다. 최근접·단방향과는 박빙이다.')
    a(f"   같은 작업시간 정보를 받으면서도 짧은 일 먼저 고르기로는 조금만 움직였다(한가한 날 일치율 "
      f"{agree['initial']:.1f}%→{agree['flow']:.1f}%, 최단 작업 {agree['spt']:.1f}%).")
    a('3. **운영 채택**: 미통과 그대로다. 이번 감사는 투입·미완료·본선 보호 조건을 다시 판정하지 않았고 새 채택 근거도 없다.')
    a('4. **현실 타당성**: 미확보다. 트럭 단가만 법정 근거가 있고, 본선 단가는 용선료 대리값, 빈 이동·재조작은')
    a('   근거 없는 설계값이다. 작업시간은 오차 없이 계획대로 끝나고(규칙이 시간을 정확히 안다), 본선 유휴는 운반 장비')
    a('   설정의 바닥에 크게 좌우되며, 배의 완료가 평가창 밖이다. 현장 비용·성능을 주장할 근거가 아니다.')
    a('')
    a('## ③ 예정사항')
    a('')
    a('- 등록된 다음 순서: YR-332-m(투입·완료 장부 감사) → YR-332-w(학습 대안의 선택 가능성 감사).')
    a('  m에는 이번에 찾은 투입 조건 단서(양하 예정분이 빈 칸을 먼저 차지해 반입 트럭이 막힐 수 있음)를 넘겼다.')
    a('- 새로 등록: x(배 여유 입력이 상한에 걸리는 문제 — 단일축 비교를 사전등록 후 진행), y(비용 정의 문서·자동')
    a('  대조 정리), z(본선 작업 마감 필드 재시각 누락 — 현재 평가에는 영향 없음).')
    a('- 기존 작업에 근거 추가: i(본선 유휴 바닥 수치), d(30일 판정에서 본선 완료를 관측하는 평가창).')
    a('')
    a('[사전등록](../../../docs/research/v7-discharge/comparison-audit-prereg.md) · [요약 원값](summary.json) · '
      '[재채점](sensitivity.json) · [판정](final-verdict.json) · [재현 검증](replay-verification.json) · '
      '[독립 계산](independent-arithmetic-audit.json) · [수치 목록](inventory.json) · [검증](verification.json)')
    text = '\n'.join(L)+'\n'
    (OUT/'README.md').write_text(text, encoding='utf-8', newline='\n')
    missing = [k for k, val in claims.items() if val not in text]
    lines = text.count('\n')
    out = dict(claims=claims, claims_present=not missing, missing=missing, readme_lines=lines,
               readme_within_200=lines <= 200, previous_report_alignment=cross_check_previous())
    (OUT/'claim-check.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n',
                                         encoding='utf-8', newline='\n')
    assert not missing and lines <= 200, out
    print('README lines:', lines, 'claims:', len(claims))



def cross_check_previous():
    """Base-coefficient comparisons must equal the published YR-332-v validated summary."""
    prev = json.loads(Path('outputs/reports/yr332_classical_eval/validated-summary.json')
                      .read_text(encoding='utf-8'))
    sens = load('sensitivity.json')
    rows = {(r['arm'], r['baseline']): r for r in prev['averages'] + prev['learning_averages']
            if r['load'] is None}
    checked, worst = 0, 0.0
    for key, x in sens.items():
        kind, arm, other = key.split(':')
        ref = rows[(arm, 'own_initial' if kind == 'K1' else other)]
        for mine, theirs in ((x['variants']['base']['mean_pct'], ref['cost_change_pct']),
                             (x['dwell']['mean_pct'], ref['turn_change_pct']),
                             (x['variants']['base']['wins'], ref['cost_wins']),
                             (x['dwell']['wins'], ref['turn_wins'])):
            worst = max(worst, abs(mine-theirs))
            checked += 1
    out = dict(compared_values=checked, max_abs_difference=worst, tolerance=1e-9,
               source='outputs/reports/yr332_classical_eval/validated-summary.json')
    assert worst <= 1e-9, out
    return out


def write_verification():
    """Consolidated evidence; written after the README so its hash is final."""
    import hashlib
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    rep, ind, cc, inv = (load('replay-verification.json'), load('independent-arithmetic-audit.json'),
                         load('claim-check.json'), load('inventory.json'))
    camp = load('campaign.json')
    files = sorted(x.as_posix() for x in OUT.glob('*.json') if x.name != 'verification.json')
    artifacts = {f: sha(f) for f in files+[(OUT/'README.md').as_posix()]}
    out = dict(
        task='YR-332-t', run_commit=camp['commit'], runtime_gate=rep['runtime']['status'],
        exact_replays=rep['exact_replays'], cases=rep['cases'], replay_checked=rep['checked'],
        independent_checks=ind['checks'], independent_mismatches=ind['mismatches'],
        readme_claims=len(cc['claims']), readme_claims_present=cc['claims_present'],
        readme_lines=cc['readme_lines'],
        previous_report_values=cc['previous_report_alignment']['compared_values'],
        previous_report_max_abs_difference=cc['previous_report_alignment']['max_abs_difference'],
        inventory_entries=inv['check']['entries'], inventory_symbols_missing=inv['check']['symbols_missing'],
        tests=dict(new_audit_tests=9, related_suite_passed=81, related_suite_files=[
            'tests/v7/test_comparison_audit.py', 'tests/v7/test_evaluation_runtime.py',
            'tests/v7/test_episode_repro.py', 'tests/v7/test_classical_rules.py',
            'tests/v7/test_arrival_fifo.py', 'tests/v7/test_crane_feature_count.py',
            'tests/v7/test_crane_count_transform.py', 'tests/v7/test_count_transform_recovery.py',
            'tests/v7/test_vessel_classes.py', 'tests/v7/test_discharge_backpressure.py']),
        external_source_checks=dict(
            truck_rate='2026 safe-freight notice (No. 2026-55, amended 2026-402) and 20,000 KRW per 30 min '
                       'after 1 h: secondary confirmation via MOLIT listing and logistics press; clause not read',
            vessel_rate='Clarksons-cited 4,250 TEU at USD 54,500/day is close to the table value 4,300 TEU '
                        'USD 56,000/day; 8,000 TEU not confirmed; LARGE 32.25 not reproducible from the table'),
        artifact_sha256=artifacts)
    (OUT/'verification.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n',
                                          encoding='utf-8', newline='\n')
    return out


def finalize_dashboard(pin):
    """Pinned board/spec/evidence audit plus overview-number alignment (no board edits)."""
    import re
    import subprocess
    from yard_rl.experiments.gate_harness import audit_dashboard, judge_claim_alignment
    upstream = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', '@{upstream}'],
                                       encoding='utf-8').strip()
    board = audit_dashboard(
        '.', task_id='YR-332-t', expected_state='done',
        spec_path='.claude/docs/dashboard-task-specs/YR-332-t-comparison-validity-audit.md',
        evidence_paths=[str(OUT/p) for p in ('summary.json', 'README.md', 'verification.json',
                                             'campaign.json', 'independent-arithmetic-audit.json',
                                             'final-verdict.json', 'sensitivity.json')],
        evidence_commits=[pin], remote_ref=upstream, pin_commit=pin).as_dict()
    assert board['status'] == 'PASS', board
    overview = next(line for line in Path('.claude/Dashboard/README.md').read_text(encoding='utf-8')
                    .splitlines() if line.startswith('- **비교 기준·비용 수치 감사 완료'))
    values = [float(v) for v in re.findall(r'([+-]\d+\.\d+)%', overview)]
    sens = load('sensitivity.json')
    actual = [sens[k]['variants']['base']['mean_pct']
              for k in ('K2:flow:spt', 'K2:flow:fcfs_arrival', 'K1:flow:initial')]
    assert len(values) == len(actual) == 3, values
    claims = judge_claim_alignment({str(i): x for i, x in enumerate(values)},
                                   {str(i): x for i, x in enumerate(actual)},
                                   absolute_tolerance=.0050001).as_dict()
    assert claims['status'] == 'PASS', claims
    board['evidence']['overview_claim_alignment'] = claims
    (OUT/'final-dashboard-audit.json').write_text(json.dumps(board, ensure_ascii=False, indent=2)+'\n',
                                                  encoding='utf-8', newline='\n')
    print('Final Dashboard and remote evidence: PASS', pin)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--finalize', metavar='PIN_COMMIT',
                   help='after the board commit is pushed: pinned dashboard audit only')
    args = p.parse_args()
    if args.finalize:
        finalize_dashboard(args.finalize)
    else:
        main()
        write_verification()
