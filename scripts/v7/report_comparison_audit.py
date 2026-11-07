"""YR-332-t analysis of the exact re-runs: cost terms, rescoring, break-even, window.

Pure functions are importable for tests. All comparisons and coefficient variants are
the preregistered ones (docs/research/v7-discharge/comparison-audit-prereg.md).
"""
import argparse
import json
import math
from pathlib import Path
import statistics

import audit_comparison_validity as run
from six_step_learning import gz_read, sha

OUT = run.OUT
LOADS, SEEDS, ARMS, RULES = run.LOADS, run.SEEDS, run.ARMS, run.RULES
RULE_KEYS = (*RULES, 'rule_all')
LEARNED = ARMS[1:]
BASE = dict(truck=40_000.0, over=40_000.0, threshold=3600, move=60_000.0, rehandle=2_000.0,
            vessel=35.88)
#: Same charter table as the implemented median (reference doc, YR-307), by GT.
VESSEL_BY_GT = {50_000.0: 45.78, 100_000.0: 35.88, 150_000.0: 32.25}
#: (A) documented alternatives, one element changed at a time.
VARIANTS = {
    'base': {},
    'truck_linear': dict(over=0.0),
    'truck_legal_excess': dict(truck=0.0),
    'truck_threshold_1800': dict(threshold=1800),
    'vessel_by_class': dict(vessel='by_class'),
    'vessel_2024_level': dict(vessel=35.88/3),
    'vessel_v3_design': dict(vessel=2.99),
    'yc_move_yr207': dict(move=40_000.0),
}
#: (B) linear coefficients scanned from 0 to 10x for break-even points.
BREAK_EVEN = ('truck', 'over', 'move', 'rehandle', 'vessel')
SCAN_MAX = 10.0


def units(d):
    """Native quantities behind the four official terms (24-hour window)."""
    a = d['probe']['audit']
    t = a['trucks']
    return dict(truck_h=t['sum_tt']/3600, over3600_h=t['over_3600']/3600,
                over1800_h=t['over_1800']/3600, move_h=a['yc_empty_travel_s']/3600,
                rehandles=float(a['rehandles']), gt_h=a['vessel_gt_idle_hours'],
                gt_h_by_class=sum(VESSEL_BY_GT[float(s['gt'])]*s['gt']*s['idle_s']/3600
                                  for s in a['ships_24h'].values()))


def cost(u, **change):
    """Total cost of one fixed trajectory with one coefficient element changed."""
    c = dict(BASE, **change)
    over = u['over3600_h'] if c['threshold'] == 3600 else u['over1800_h']
    vessel = u['gt_h_by_class'] if c['vessel'] == 'by_class' else c['vessel']*u['gt_h']
    return (c['truck']*u['truck_h']+c['over']*over+c['move']*u['move_h']
            + c['rehandle']*u['rehandles']+vessel)


def pair_changes(costs_a, costs_b):
    """Per-pair percent change of a over b; mean with equal weights and wins (a lower)."""
    changes = [100.0*(a/b-1.0) for a, b in zip(costs_a, costs_b)]
    return dict(mean_pct=math.fsum(changes)/len(changes), wins=sum(x < 0 for x in changes),
                n=len(changes), pairs=changes)


def comparisons(data):
    """K1 (each learned arm vs own initial) and K2 (each arm vs each rule), 9 pairs each."""
    out = {}
    for arm in LEARNED:
        out[f'K1:{arm}:initial'] = [((arm, s, n, 0), ('initial', s, n, 0))
                                    for n in LOADS for s in SEEDS]
    for arm in ARMS:
        for rule in RULE_KEYS:
            out[f'K2:{arm}:{rule}'] = [((arm, s, n, 0), (rule, 0, n, 0))
                                       for n in LOADS for s in SEEDS]
    return out


def verdict(row):
    return dict(better=row['mean_pct'] < 0, majority=row['wins'] >= 5)


def rescore(data, pairs, **change):
    u = {c: units(d) for c, d in data.items()}
    return pair_changes([cost(u[a], **change) for a, _ in pairs],
                        [cost(u[b], **change) for _, b in pairs])


def break_even(ua, ub, key, base_value=None, scan_max=SCAN_MAX, steps=2000):
    """Coefficient multipliers in [0, scan_max] where the mean sign or majority flips."""
    base_value = BASE[key] if base_value is None else base_value

    def summary(mult):
        return pair_changes([cost(u, **{key: base_value*mult}) for u in ua],
                            [cost(u, **{key: base_value*mult}) for u in ub])
    ref = verdict(summary(1.0))
    grid = [scan_max*i/steps for i in range(steps+1)]
    flips = dict(mean=[], majority=[])
    previous = None
    for m in grid:
        v = verdict(summary(m))
        if previous is not None:
            for kind, field in (('mean', 'better'), ('majority', 'majority')):
                if v[field] != previous[1][field]:
                    lo, hi = previous[0], m
                    for _ in range(50):
                        mid = (lo+hi)/2
                        if verdict(summary(mid))[field] == previous[1][field]:
                            lo = mid
                        else:
                            hi = mid
                    flips[kind].append((lo+hi)/2)
        previous = (m, v)
    nearest = {k: min(v, key=lambda m: abs(math.log(max(m, 1e-12)))) if v else None
               for k, v in flips.items()}
    return dict(reference=ref, flips=flips, nearest=nearest)


def term_split(da, db):
    """Percent change of a over b split by cost term, relative to b's total."""
    ra, rb = da['result'], db['result']
    return {k: 100.0*(ra[k]-rb[k])/rb['phi_krw']
            for k in ('c_wait', 'c_move', 'c_rehandle', 'c_vessel')}


def stage_split_pct(da, db):
    """Percent change of mean turn time of a over b split by lifecycle stage."""
    ta, tb = da['probe']['audit']['trucks'], db['probe']['audit']['trucks']
    mean_b = tb['sum_tt']/tb['n']
    return {k: 100.0*(ta['stages'][k]/ta['n']-tb['stages'][k]/tb['n'])/mean_b
            for k in ('travel_in', 'queue', 'service', 'exit')}


def mean_dict(rows):
    return {k: statistics.mean(r[k] for r in rows) for k in rows[0]}


def load_all():
    return {c: gz_read(run.path_for(c)) for c in run.CASES}


TERMS = ('c_wait', 'c_move', 'c_rehandle', 'c_vessel')
STAGES = ('travel_in', 'queue', 'service', 'exit')


def runs_of(data, arm, load):
    if arm in RULE_KEYS:
        return [data[(arm, 0, load, 0)]]
    return [data[(arm, s, load, 0)] for s in SEEDS]


def vessel_by_flow(d):
    """Official vessel cost split by the flow of each STS stream (sums to c_vessel)."""
    a = d['probe']['audit']
    out = {}
    for s in a['streams_24h']:
        gt = a['ships_24h'][s['ship']]['gt']
        out[s['flow']] = out.get(s['flow'], 0.0)+BASE['vessel']*gt*s['sts_wait_s']/s['sts']/3600
    return out


def sts_wait_share(d):
    """Average share of berthed time inside the 24h window that each STS stream waited."""
    a = d['probe']['audit']
    start = {v['vessel_id']: max(0.0, v['berth_start_s']) for v in a['fleet']}
    out = {}
    for s in a['streams_24h']:
        out.setdefault(s['flow'], []).append(s['sts_wait_s']/(run.END_S-start[s['ship']]))
    return {k: statistics.mean(v) for k, v in out.items()}


def sts_wait_distribution(data):
    """Per-stream STS waiting share over all runs; the load floor 16.4% is documented."""
    out = {}
    for d in data.values():
        a = d['probe']['audit']
        start = {v['vessel_id']: max(0.0, v['berth_start_s']) for v in a['fleet']}
        for s in a['streams_24h']:
            out.setdefault(s['flow'], []).append(s['sts_wait_s']/(run.END_S-start[s['ship']]))
    return {k: dict(streams=len(v), min=min(v), median=statistics.median(v),
                    mean=statistics.mean(v), max=max(v)) for k, v in out.items()}


def cost_table(data):
    rows = []
    for load in LOADS:
        for arm in (*RULE_KEYS, *ARMS):
            ds = runs_of(data, arm, load)

            def avg(f):
                return statistics.mean(f(d) for d in ds)
            t = lambda d: d['probe']['audit']['trucks']
            rows.append(dict(
                arm=arm, load=load, runs=len(ds), phi_krw=avg(lambda d: d['result']['phi_krw']),
                krw={k: avg(lambda d, k=k: d['result'][k]) for k in TERMS},
                share={k: avg(lambda d, k=k: d['result'][k]/d['result']['phi_krw']) for k in TERMS},
                native=dict(
                    truck_hours=avg(lambda d: units(d)['truck_h']),
                    truck_over_1h_hours=avg(lambda d: units(d)['over3600_h']),
                    yc_empty_hours=avg(lambda d: units(d)['move_h']),
                    rehandles=avg(lambda d: units(d)['rehandles']),
                    vessel_idle_ship_hours=avg(lambda d: sum(
                        s['idle_s'] for s in d['probe']['audit']['ships_24h'].values())/3600),
                    vessel_gt_hours=avg(lambda d: units(d)['gt_h'])),
                surcharge_share_of_truck=avg(lambda d: BASE['over']*units(d)['over3600_h']
                                             / d['result']['c_wait']),
                mean_turn_min=avg(lambda d: d['result']['mean_turn_time_s']/60),
                stage_min={k: avg(lambda d, k=k: t(d)['stages'][k]/t(d)['n']/60) for k in STAGES},
                stage_share={k: avg(lambda d, k=k: t(d)['stages'][k]/t(d)['sum_tt']) for k in STAGES},
                flow_stage_min={f: {k: avg(lambda d, f=f, k=k: t(d)['by_flow'][f][k]
                                           / t(d)['by_flow'][f]['n']/60) for k in STAGES}
                                for f in ('GATE_IN', 'GATE_OUT')},
                vessel_by_flow_krw={f: avg(lambda d, f=f: vessel_by_flow(d).get(f, 0.0))
                                    for f in ('DISCHARGE', 'LOAD')},
                sts_wait_share={f: avg(lambda d, f=f: sts_wait_share(d).get(f, 0.0))
                                for f in ('DISCHARGE', 'LOAD')},
                structural_vessel_krw=avg(lambda d: d['result']['c_vessel_structural'])))
    return rows


def ledger_table(data):
    """Denominators side by side with the totals; cause recovery belongs to YR-332-m."""
    rows = []
    for load in LOADS:
        for arm in (*RULE_KEYS, *ARMS):
            ds = runs_of(data, arm, load)
            avg = lambda f: statistics.mean(f(d) for d in ds)
            op = lambda d: d['probe']['operational']
            rows.append(dict(
                arm=arm, load=load,
                orders=avg(lambda d: d['probe']['audit']['orders']),
                admitted=avg(lambda d: d['result']['admitted']),
                gated_in_24h=avg(lambda d: d['result']['n_trucks']),
                not_admitted=avg(lambda d: d['probe']['audit']['orders']-d['result']['admitted']),
                unfinished_24h=avg(lambda d: d['result']['n_censored']),
                vessels_unfinished_26h=avg(lambda d: op(d)['vessel_unfinished']),
                vessel_moves_remaining_26h=avg(lambda d: sum(s['remaining_moves']
                                                             for s in op(d)['streams'])),
                vessel_delay_26h_lower_bound_h=avg(lambda d: op(d)['vessel_delay_s']/3600)))
    return rows


def window_table(data):
    rows = []
    for load in LOADS:
        d = data[('spt', 0, load, 0)]
        ships = {r['ship']: r for r in d['probe']['operational']['ships']}
        for v in d['probe']['audit']['fleet']:
            plan = ships[v['vessel_id']]['planned_s']
            done = {arm: sum(ships_done(x, v['vessel_id']) for x in runs_of(data, arm, load))
                    for arm in (*RULE_KEYS, *ARMS)}
            rows.append(dict(load=load, ship=v['vessel_id'], vessel_class=v['vessel_class'],
                             gt=v['gt'], berth_start_h=v['berth_start_s']/3600,
                             work_h=v['work_s']/3600,
                             work_end_without_idle_h=(max(0.0, v['berth_start_s'])+v['work_s'])/3600,
                             planned_completion_h=plan/3600,
                             planned_after_26h=plan > run.END_S+7200,
                             done_by_26h_runs=done, total_runs=sum(len(runs_of(data, a, load))
                                                                    for a in done)))
    return rows


def ships_done(d, ship):
    return next(r['done'] for r in d['probe']['operational']['ships'] if r['ship'] == ship)


def input_table(data):
    """Pooled counts per family ('rules' = all eight rules) and per arm, also per load."""
    rows = {}
    for (arm, seed, load, _), d in data.items():
        a = d['probe']['audit']
        family = 'rules' if arm in RULE_KEYS else arm
        for group in {family, arm, f'{family}@{load}', f'{arm}@{load}'}:
            acc = rows.setdefault(group, {})
            for k, v in {**a['policy_inputs'], **a['shortest_job_agreement']}.items():
                acc[k] = acc.get(k, 0)+v
            for k, v in a['input_maxima'].items():
                acc['max_'+k] = max(acc.get('max_'+k, v), v)
    out = {}
    for family, acc in rows.items():
        out[family] = dict(
            crane_decisions=acc['crane_decisions'], serve_candidates=acc['serve_candidates'],
            over_clip_rate={k: acc.get(k+'_over_clip', 0)/acc[k+'_n'] for k in run.CLIPS
                            if acc.get(k+'_n')},
            slack_above_plus_2h=acc.get('slack_above_plus_clip', 0)/acc['slack_abs_s_n'],
            slack_below_minus_2h=acc.get('slack_below_minus_clip', 0)/acc['slack_abs_s_n'],
            waiting_over_18_rate=acc['waiting_count_over_18']/acc['crane_decisions'],
            maxima={k[4:]: v for k, v in acc.items() if k.startswith('max_')},
            shortest_agreement=acc['chose_shortest']/acc['serve_choices_with_alternatives'],
            serve_choices_with_alternatives=acc['serve_choices_with_alternatives'],
            mean_excess_duration_s=acc['excess_duration_s']/acc['serve_choices_with_alternatives'])
    return out


def independent_check(data):
    """Second path: rebuild sums from stored truck columns, and costs from units."""
    checks = mismatches = 0
    worst = 0.0
    for case, d in data.items():
        cols = gz_read(run.trucks_path(case))
        a, r = d['probe']['audit'], d['result']
        sum_tt = over1 = over05 = 0.0
        n = 0
        for gi, go in zip(cols['gate_in_s'], cols['gate_out_s']):
            if gi is None:
                continue
            end = run.END_S if go is None else min(go, run.END_S)
            tt = max(0.0, end-gi)
            n += 1
            sum_tt += tt
            over1 += max(0.0, tt-3600)
            over05 += max(0.0, tt-1800)
        pairs = [(sum_tt, a['trucks']['sum_tt']), (over1, a['trucks']['over_3600']),
                 (over05, a['trucks']['over_1800']),
                 ((BASE['truck']*sum_tt+BASE['over']*over1)/3600, r['c_wait']),
                 (sum(vessel_by_flow(d).values()), r['c_vessel']),
                 (cost(units(d)), r['phi_krw'])]
        for got, want in pairs:
            checks += 1
            rel = abs(got-want)/max(1.0, abs(want))
            worst = max(worst, rel)
            mismatches += rel > 1e-6
        checks += 1
        mismatches += n != r['n_trucks']
    return dict(checks=checks, mismatches=mismatches, worst_relative_error=worst,
                tolerance=1e-6, note='truck columns are rounded to 1 ms')


def analyze(data):
    groups = comparisons(data)
    sens = {}
    for key, pairs in groups.items():
        rows = {v: rescore(data, pairs, **change) for v, change in VARIANTS.items()}
        base = verdict(rows['base'])
        flips = [v for v, r in rows.items() if verdict(r) != base]
        detail = {v: [f for f in ('better', 'majority') if verdict(rows[v])[f] != base[f]]
                  for v in flips}
        dwell = pair_changes([data[a]['result']['mean_turn_time_s'] for a, _ in pairs],
                             [data[b]['result']['mean_turn_time_s'] for _, b in pairs])
        ua = [units(data[a]) for a, _ in pairs]
        ub = [units(data[b]) for _, b in pairs]
        sens[key] = dict(
            variants={v: {k: r[k] for k in ('mean_pct', 'wins', 'n')} for v, r in rows.items()},
            base_pairs=rows['base']['pairs'], base_verdict=base, documented_variant_flips=flips,
            flip_detail=detail,
            dwell={k: dwell[k] for k in ('mean_pct', 'wins', 'n')},
            break_even={k: break_even(ua, ub, k) for k in BREAK_EVEN},
            term_split_pct=mean_dict([term_split(data[a], data[b]) for a, b in pairs]),
            stage_split_pct=mean_dict([stage_split_pct(data[a], data[b]) for a, b in pairs]))
    return sens


SRC = 'src/yard_rl/v7/'
#: (group, name, value, unit, file, symbol present in file, claimed basis, class, note).
#: Classes: 법정(인용) / 문헌·시장 / 실측표본 / 설계 / 미확정. Values marked live are read from code.
def inventory():
    from yard_rl.v7.reward import krw
    from yard_rl.v7.crane import features as f
    from yard_rl.v7.stage import vessels as vs
    from yard_rl.v7.world.integrated import profiles, scenario_gen, terminal_stream as ts
    cls = {c[0]: c for c in krw.VESSEL_CLASSES}
    obs = ts.OBS_24H
    stream_params = ts.TerminalStreamParams(load_4h=3500)  # load does not change these defaults
    rows = [
        ('비용', '트럭 시간가치', krw.KRW_TRUCK_HOUR, '원/트럭·시간', 'reward/krw.py', 'KRW_TRUCK_HOUR',
         '2026 안전운임 고시 항만부두 대기료 30분당 20,000원', '법정(인용)',
         '고시와 30분당 2만원·1시간 기준은 국토부 게시·물류 보도로 2차 확인, 조항 원문 미대조. '
         '법정 대기시간(도착 요청~송장 발급)과 게이트 진입~출차는 다르다.'),
        ('비용', '1시간 초과 할증', krw.KRW_TRUCK_OVER_HOUR, '원/트럭·시간', 'reward/krw.py',
         'KRW_TRUCK_OVER_HOUR', '고시의 1시간 초과분 청구를 실제 비용 2배로 해석', '설계',
         'YR-207 결정 기록의 목적식은 할증 없는 선형이다.'),
        ('비용', '할증 시작', krw.KRW_OVERTIME_START_S, '초', 'reward/krw.py', 'KRW_OVERTIME_START_S',
         '고시의 첫 1시간', '법정 구조 차용', '무대 트럭 SLA 1,800초와 다르다(04 문서 주석).'),
        ('비용', 'YC 빈 주행', krw.KRW_YC_MOVE_HOUR, '원/시간', 'reward/krw.py', 'KRW_YC_MOVE_HOUR',
         '전력·정비·운영 환산', '설계',
         '계산 근거 없음. YR-207 기록은 40,000원. 실제 측정량은 재조작·재배치로 생긴 추가 이동이 아니라 '
         '모든 작업의 빈 갠트리 접근 이동이다.'),
        ('비용', '재조작', krw.KRW_REHANDLE_EACH, '원/회', 'reward/krw.py', 'KRW_REHANDLE_EACH',
         '추가 lift 2분 × YC 분당 단가', '설계', '사전 재조작·적하 파내기도 센다.'),
        ('비용', '본선 유휴', krw.KRW_VESSEL_GT_HOUR, '원/GT·시간', 'reward/krw.py',
         'KRW_VESSEL_GT_HOUR', '2025 1년 정기용선료 보간 중앙값(환율 1,350원/달러)', '문헌·시장',
         '용선료는 지연 손실 자체가 아닌 대리값. 4,250TEU 일 54,500달러 보도는 표와 근접, 8,000TEU 값은 '
         '미확인. 대형 32.25원은 표의 세 점으로 재현되지 않는다(선형 외삽 29.85원). 선급별 값 대신 '
         '중앙값 하나를 모든 배에 쓴다.'),
        ('관측창', '관측 24시간(워밍업 2시간 포함)', obs.observe_s, '초', 'world/integrated/terminal_stream.py',
         'OBS_24H', '사전등록 §E', '설계', '워밍업 2시간도 비용에 들어간다(0시부터 계산).'),
        ('관측창', '정리 시간', ts.DIURNAL_DRAIN_S, '초', 'world/integrated/terminal_stream.py',
         'DIURNAL_DRAIN_S', '하루 끝 도착분 완료 자리', '설계',
         '비용에는 안 들어가고 출입 기록도 24시간까지만 갱신된다.'),
        ('관측창', '본선 계획 완료 여유 배수', scenario_gen.TerminalGenParams().vessel_deadline_mult,
         '배', 'world/integrated/scenario_gen.py',
         'vessel_deadline_mult', '기본 2.0(넉넉)', '설계',
         '계획 완료 = 시작 + 2×순수 작업시간 → 평가창 밖. 정책의 본선 여유 입력이 늘 상한에 걸린다.'),
        ('트럭', '반출 비율', stream_params.gate_out_share, '비율',
         'world/integrated/terminal_stream.py', 'gate_out_share', '표기 없음', '미확정', ''),
        ('트럭', '40피트 비율', stream_params.size_mix_ft40, '비율',
         'world/integrated/terminal_stream.py', 'size_mix_ft40', '표기 없음', '미확정', ''),
        ('트럭', '하루 도착 곡선', '10·15·21시 봉우리, 야간 38%', '-',
         'world/integrated/terminal_stream.py', 'DIURNAL_PEAKS', '사전등록·사용자 확인', '설계', ''),
        ('트럭', '통지 리드타임', 'DGT 표본 분포', '-', 'stage/dgt_lead.py', 'DGT',
         'DGT 방문 표본 중 통지 있는 71건', '실측표본', '수직형 터미널 1시간 표본이다.'),
        ('트럭', '예약 대비 실제 도착 오차', 0.0, '초', 'stage/orders.py', 'arrival_s',
         '예약 = 실제 도착', '설계', '실자료는 중앙 93분 늦음(YR-332-f에서 다룬다).'),
        ('트럭', '게이트→블록 이동', '190~410(평균 300)', '초', 'world/integrated/yard_layout.py',
         'gate_to_block_s', '사용자 결정 5분·합성 배치', '설계', ''),
        ('트럭', '블록→게이트 이동', stream_params.exit_travel_mu_s, '초(평균)',
         'world/integrated/terminal_stream.py', 'exit_travel_mu_s', 'assumed', '설계', '표준편차 12%.'),
        ('본선', '선급 GT', '50,000/100,000/150,000', 'GT', 'reward/krw.py', 'VESSEL_CLASSES',
         '사용자 결정 2026-08-20(02b)', '설계', ''),
        ('본선', '선급별 안벽 크레인 수', '/'.join(str(krw.VESSEL_STS_STREAMS[k]) for k in
                                         ('SMALL', 'MEDIUM', 'LARGE')), '대', 'reward/krw.py',
         'VESSEL_STS_STREAMS', '실측 참조(스트림당 25~30 moves/h)', '문헌·시장', ''),
        ('본선', '하루 선단', '소2·중1·대 30%', '척', 'stage/vessels.py', 'DAY_FLEET', '02b §3', '설계', ''),
        ('본선', '안벽 크레인 생산성', vs.STREAM_MOVES_PER_H, 'moves/시간', 'stage/vessels.py',
         'STREAM_MOVES_PER_H', '25~30 moves/h 중앙값', '문헌·시장', ''),
        ('운반', '블록당 운반 차량', 3, '대', 'world/integrated/profiles.py', 'n_units=3',
         'AGV 60대/23블록 공개값 유도', '문헌·시장', ''),
        ('운반', '운반 왕복', profiles.YT_ROUND_TRIP_S, '초', 'world/integrated/profiles.py',
         'YT_ROUND_TRIP_S', '안벽 손실 10~20% 문헌에 맞춘 재보정', '설계',
         '적하 안벽 대기 하한 16.4%를 만든다. 양하는 인계 상한 3박스와 함께 코드 산술상 약 31% 하한.'),
        ('운반', '양하 인계 상한·안벽 버퍼', 3, '박스', 'stage/episode.py', 'yard_handover_cap',
         'YR-332-h 집계 근사', '설계', ''),
        ('야드', '블록 수·블록당 크레인', '21·2', '-', 'world/integrated/yard_layout.py',
         'TERMINAL_BLOCKS', 'HJNC 21블록 규모 앵커', '설계', '두 크레인은 역할 구분 없는 근사.'),
        ('야드', '초기 적재율', stream_params.fill_ratio, '비율',
         'world/integrated/terminal_stream.py', 'fill_ratio', '문헌 범위 0.60~0.75 중심', '문헌·시장', ''),
        ('야드', '크레인 속도(갠트리·트롤리·호이스트)', '4.0·1.0·0.58/1.17', 'm/s',
         'world/sim/travel_time.py', 'gantry', 'Kalmar ASC 등 장비 자료', '문헌·시장',
         '축 동작을 순서대로 더하고 겹치지 않는다(가정).'),
        ('야드', '작업시간 오차', 0.0, '초', 'world/integrated/engine.py', 'duration_s',
         '계획 시간 = 실제 시간', '설계', '규칙과 학습 정책 모두 실제와 같은 계획 시간을 본다.'),
        ('야드', '크레인 안전 간격', 2.0, '베이', 'world/integrated/reservation.py', 'safety',
         'assumed', '설계', ''),
        ('정책 입력', '누적 대기 상한', f._WAIT_REF_H, '시간', 'crane/features.py', '_WAIT_REF_H',
         '할증 시작의 2배', '설계', ''),
        ('정책 입력', '본선 여유 상한', f._SLACK_CLIP_H, '±시간', 'crane/features.py', '_SLACK_CLIP_H',
         '배 크기 상수라 자름', '설계', '이번 관측에서 실제 값이 늘 이 범위 밖이었다(분석 3).'),
        ('정책 입력', '소요·빈 이동·재조작 눈금', '3600초·200m·30회에서 1', '-', 'crane/features.py',
         'duration_s', 'v4 그대로', '설계', ''),
        ('정책 입력', '대기 트럭 수 눈금', 18, '대', 'crane/features.py', '18.0', 'v4 그대로', '설계',
         'YR-332-s에서 상한 제거를 비교했고 채택하지 않았다.'),
        ('공통 제약', '필수 작업 문턱', '누적 대기 1,440초(=0.8×1,800)', '-',
         'world/integrated/candidates.py', 'mandatory_wait_frac', 'SLA 가정', '설계',
         '규칙과 학습 정책 모두 이 작업을 먼저 한다.'),
    ]
    keys = ('group', 'name', 'value', 'unit', 'file', 'symbol', 'claimed_basis', 'class', 'note')
    return [dict(zip(keys, (*r[:4], SRC+r[4], *r[5:]))) for r in rows]


def check_inventory(rows):
    missing = [r['name'] for r in rows if r['symbol'] not in Path(r['file']).read_text(encoding='utf-8')]
    return dict(entries=len(rows), symbols_missing=missing,
                classes={c: sum(r['class'] == c for r in rows) for c in sorted({r['class'] for r in rows})})


KEY = (('K1:flow:initial', '최신 학습 후 vs 같은 모델 학습 전'),
       ('K1:pairwise:initial', '비용 차이 학습 후 vs 같은 모델 학습 전'),
       ('K2:flow:spt', '최신 학습 vs 최단 작업'),
       ('K2:flow:rule_all', '최신 학습 vs 기존 최단 작업(SF_SPT)'),
       ('K2:flow:nearest', '최신 학습 vs 최근접'),
       ('K2:flow:fcfs_arrival', '최신 학습 vs 도착순'),
       ('K2:pairwise:spt', '비용 차이 학습 vs 최단 작업'),
       ('K2:initial:spt', '학습 전 모델 vs 최단 작업'))
COEF_KO = dict(truck='트럭 기본', over='1시간 초과 할증', move='크레인 빈 이동', rehandle='재조작',
               vessel='본선 유휴')


def fmt_mult(m):
    return '없음(0~10배)' if m is None else f'×{m:.2f}'


def key_rows(sens):
    rows = []
    for key, label in KEY:
        s = sens[key]
        docs = {v: r['mean_pct'] for v, r in s['variants'].items() if v != 'base'}
        rows.append(dict(key=key, label=label, cost_pct=s['variants']['base']['mean_pct'],
                         wins=s['variants']['base']['wins'], dwell_pct=s['dwell']['mean_pct'],
                         dwell_wins=s['dwell']['wins'], flips=s['documented_variant_flips'],
                         doc_min=min(docs.values()), doc_max=max(docs.values()),
                         nearest={k: s['break_even'][k]['nearest']['mean'] for k in BREAK_EVEN},
                         nearest_majority={k: s['break_even'][k]['nearest']['majority']
                                           for k in BREAK_EVEN},
                         term_split=s['term_split_pct'], stage_split=s['stage_split_pct']))
    return rows


def final_verdict(sens, summary, inv_check, indep, replay):
    learned = {k: v for k, v in sens.items() if k.startswith('K1:')}
    rules = {k: v for k, v in sens.items() if k.startswith('K2:')}
    flips = {k: v['flip_detail'] for k, v in sens.items() if v['documented_variant_flips']}
    k1_better = all(v['base_verdict']['better'] for v in learned.values())
    k1_mean_robust = not any('better' in f for k, d in flips.items() if k.startswith('K1:')
                             for f in d.values())
    k2_mean_flips = sorted(k for k, d in flips.items() if k.startswith('K2:')
                           and any('better' in f for f in d.values()))
    flow_rules = {k.split(':')[2]: dict(cost_pct=v['variants']['base']['mean_pct'],
                                        wins=v['variants']['base']['wins'],
                                        dwell_pct=v['dwell']['mean_pct'],
                                        flips=v['documented_variant_flips'])
                  for k, v in rules.items() if k.startswith('K2:flow:')}
    slack = summary['inputs']['flow']['slack_above_plus_2h']+summary['inputs']['flow']['slack_below_minus_2h']
    return dict(
        task='YR-332-t', exact_replays=replay['exact_replays'], cases=replay['cases'],
        independent_checks=indep['checks'], independent_mismatches=indep['mismatches'],
        documented_variant_flips=flips,
        categories=dict(
            learning_before_after=dict(
                status=('MEAN_IMPROVEMENT_SIGN_ROBUST_TO_DOCUMENTED_COSTS'
                        if k1_better and k1_mean_robust else 'MEAN_SIGN_COST_SENSITIVE'),
                majority_changes={k: d for k, d in flips.items() if k.startswith('K1:')},
                arms={k.split(':')[1]: dict(cost_pct=v['variants']['base']['mean_pct'],
                                            wins=v['variants']['base']['wins'],
                                            dwell_pct=v['dwell']['mean_pct'],
                                            flips=v['documented_variant_flips'])
                      for k, v in learned.items()},
                limits='three previously seen input days; 9 pairs are not independent days'),
            rule_comparison=dict(
                status=('VERDICTS_ROBUST_TO_DOCUMENTED_COSTS'
                        if not any(k.startswith('K2:') for k in flips) else 'SOME_VERDICTS_FLIP'),
                mean_sign_flips=k2_mean_flips,
                changes={k: d for k, d in flips.items() if k.startswith('K2:')},
                latest_vs_rules=flow_rules),
            operational_adoption=dict(
                status='NOT_PASSED_UNCHANGED',
                reason='YR-332-v/s protective conditions (admission, unfinished, vessel yard '
                       'completion) were not re-judged; this audit adds no adoption evidence'),
            real_world_validity=dict(
                status='NOT_ESTABLISHED',
                inventory_classes=inv_check['classes'],
                vessel_slack_input_saturated_share=slack,
                reasons=['only the truck rate has a legal basis (secondary check, clause unread)',
                         'vessel rate is a charter-rate proxy with partly unverified points',
                         'crane move and rehandle rates are design values without derivation',
                         'service times are exact (no noise) and uncalibrated',
                         'vessel completion falls outside the 24/26 h window'])),
        gates=dict(performance='UNCHANGED_NO_NEW_CLAIM', reliability='REPLAY_PASS_DOC_MISMATCHES',
                   scenario_validity='NOT_PASSED_UNCHANGED'))


ARM_KO = dict(spt='최단 작업', fcfs_arrival='도착순', rule_all='기존 최단 작업(SF_SPT)',
              nearest='최근접', vessel_fcfs='본선 우선·도착순', vessel_nearest='본선 우선·최근접',
              unidirectional='단방향', distance_wait='거리·대기 복합', initial='학습 전',
              pointwise='기존 학습', pairwise='비용 차이 학습', coverage='비교 표본 확대',
              flow='작업 방향 추가(최신)')
VARIANT_KO = dict(truck_linear='트럭 선형', truck_legal_excess='트럭 초과분만',
                  truck_threshold_1800='할증 30분부터', vessel_by_class='본선 선급별',
                  vessel_2024_level='본선 1/3', vessel_v3_design='본선 2.99원',
                  yc_move_yr207='빈 이동 4만원')


def tables(summary, sens):
    """Markdown tables for the README; every number comes from the saved JSON."""
    cost = {(r['arm'], r['load']): r for r in summary['costs']}
    led = {(r['arm'], r['load']): r for r in summary['ledger']}
    out = {}
    lines = ['| 물량 | 방식 | 총비용(백만원) | 트럭 대기 | 빈 이동 | 재조작 | 본선 유휴 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for load in LOADS:
        for arm in ('spt', 'fcfs_arrival', 'flow'):
            r = cost[(arm, load)]
            lines.append(f"| {load:,} | {ARM_KO[arm]} | {r['phi_krw']/1e6:,.1f} | "
                         + ' | '.join(f"{100*r['share'][k]:.1f}%" for k in TERMS)+' |')
    out['cost'] = '\n'.join(lines)
    lines = ['| 물량 | 방식 | 게이트→블록 | 블록 줄서기 | 작업 | 블록→게이트 | 합계 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for load in LOADS:
        for arm in ('spt', 'flow'):
            r = cost[(arm, load)]
            lines.append(f"| {load:,} | {ARM_KO[arm]} | "
                         + ' | '.join(f"{r['stage_min'][k]:.2f}" for k in STAGES)
                         + f" | {r['mean_turn_min']:.2f} |")
    out['stages'] = '\n'.join(lines)
    sw = summary['sts_wait']
    out['sts'] = '\n'.join([
        '| 안벽 크레인 흐름 | 설계상 하한 | 관측 최소 | 관측 중앙 | 관측 최대 | 스트림 수 |',
        '|---|---:|---:|---:|---:|---:|',
        f"| 적하(싣기) | 16.4%(문서) | {100*sw['LOAD']['min']:.1f}% | {100*sw['LOAD']['median']:.1f}% | "
        f"{100*sw['LOAD']['max']:.1f}% | {sw['LOAD']['streams']} |",
        f"| 양하(내리기) | 약 26~31%(코드 산술) | {100*sw['DISCHARGE']['min']:.1f}% | "
        f"{100*sw['DISCHARGE']['median']:.1f}% | {100*sw['DISCHARGE']['max']:.1f}% | "
        f"{sw['DISCHARGE']['streams']} |"])
    lines = ['| 비교 | 비용 변화 | 체류 변화 | 대안 7개 중 판정 변화 | 대안별 비용 변화 범위 |',
             '|---|---:|---:|---:|---:|']
    for r in summary['key_comparisons']:
        lines.append(f"| {r['label']} | {r['cost_pct']:+.2f}% ({r['wins']}/9) | "
                     f"{r['dwell_pct']:+.2f}% ({r['dwell_wins']}/9) | {len(r['flips'])} | "
                     f"{r['doc_min']:+.2f}% ~ {r['doc_max']:+.2f}% |")
    out['key'] = '\n'.join(lines)
    lines = ['| 비교 | '+' | '.join(COEF_KO[k] for k in BREAK_EVEN)+' |',
             '|---|'+'---:|'*len(BREAK_EVEN)]
    for r in summary['key_comparisons']:
        lines.append(f"| {r['label']} | "+' | '.join(fmt_mult(r['nearest'][k]) for k in BREAK_EVEN)+' |')
    out['break_even'] = '\n'.join(lines)
    lines = ['| 비교 | 트럭 대기 | 빈 이동 | 재조작 | 본선 유휴 | 합계 |', '|---|---:|---:|---:|---:|---:|']
    for r in summary['key_comparisons']:
        t = r['term_split']
        lines.append(f"| {r['label']} | "+' | '.join(f"{t[k]:+.2f}" for k in TERMS)
                     + f" | {sum(t.values()):+.2f} |")
    out['terms'] = '\n'.join(lines)
    lines = ['| 비교 | 게이트→블록 | 블록 줄서기 | 작업 | 블록→게이트 | 합계 |', '|---|---:|---:|---:|---:|---:|']
    for r in summary['key_comparisons']:
        t = r['stage_split']
        lines.append(f"| {r['label']} | "+' | '.join(f"{t[k]:+.2f}" for k in STAGES)
                     + f" | {sum(t.values()):+.2f} |")
    out['stage_split'] = '\n'.join(lines)
    lines = ['| 물량 | 방식 | 주문 | 투입 | 24시간 미출차 | 26시간 미완료 본선 | 26시간 본선 잔여 작업 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for load in LOADS:
        for arm in ('spt', 'fcfs_arrival', 'initial', 'flow'):
            r = led[(arm, load)]
            lines.append(f"| {load:,} | {ARM_KO[arm]} | {r['orders']:,.0f} | {r['admitted']:,.2f} | "
                         f"{r['unfinished_24h']:,.2f} | {r['vessels_unfinished_26h']:.2f} | "
                         f"{r['vessel_moves_remaining_26h']:,.1f} |")
    out['ledger'] = '\n'.join(lines)
    lines = ['| 물량 | 배 | 선급 | 접안 시작(시) | 순수 작업(시간) | 유휴 없을 때 끝(시) | 계획 완료(시) | 26시간 완료 실행 |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for r in summary['window']:
        done = sum(r['done_by_26h_runs'].values())
        lines.append(f"| {r['load']:,} | {r['ship']} | {r['vessel_class']} | {r['berth_start_h']:.1f} | "
                     f"{r['work_h']:.1f} | {r['work_end_without_idle_h']:.1f} | "
                     f"{r['planned_completion_h']:.1f} | {done}/{r['total_runs']} |")
    out['window'] = '\n'.join(lines)
    inp = summary['inputs']
    names = dict(duration_s='작업 소요(1시간)', empty_gantry_m='빈 이동 거리(200m)',
                 rehandles='재조작 수(30회)', cum_wait_s='트럭 누적 대기(2시간)',
                 slack_abs_s='본선 여유(±2시간)')
    lines = ['| 정책 입력(상한) | 규칙 실행 중 | 학습 전 | 최신 학습 |', '|---|---:|---:|---:|']
    for k, label in names.items():
        lines.append(f"| {label} | " + ' | '.join(
            f"{100*inp[f]['over_clip_rate'].get(k, 0.0):.2f}%" for f in ('rules', 'initial', 'flow'))+' |')
    lines.append('| 대기 트럭 수(18대) | '+' | '.join(f"{100*inp[f]['waiting_over_18_rate']:.2f}%"
                                                   for f in ('rules', 'initial', 'flow'))+' |')
    out['inputs'] = '\n'.join(lines)
    lines = ['| 물량 | 본선 여유가 +2시간 초과 | −2시간 미만 | 상한 안 |', '|---|---:|---:|---:|']
    for load in LOADS:
        r = inp[f'flow@{load}']
        inside = 1-r['slack_above_plus_2h']-r['slack_below_minus_2h']
        lines.append(f"| {load:,} | {100*r['slack_above_plus_2h']:.1f}% | "
                     f"{100*r['slack_below_minus_2h']:.1f}% | {100*inside:.1f}% |")
    out['slack_by_load'] = '\n'.join(lines)
    lines = ['| 방식 | 3,500 | 7,500 | 15,000 | 전체 |', '|---|---:|---:|---:|---:|']
    for arm in ('spt', 'rule_all', 'fcfs_arrival', 'nearest', 'initial', 'pairwise', 'flow'):
        lines.append(f"| {ARM_KO[arm]} | "+' | '.join(
            f"{100*inp[f'{arm}@{n}']['shortest_agreement']:.1f}%" for n in LOADS)
            + f" | {100*inp[arm]['shortest_agreement']:.1f}% |")
    out['agreement'] = '\n'.join(lines)
    return out


def main():
    data = load_all()
    assert len(data) == 69
    sens = analyze(data)
    inv = inventory()
    summary = dict(
        cases=len(data), costs=cost_table(data), ledger=ledger_table(data),
        window=window_table(data), inputs=input_table(data),
        sts_wait=sts_wait_distribution(data), sts_load_floor_documented=0.164,
        documented_variants=VARIANTS, vessel_rate_by_gt=VESSEL_BY_GT, break_even_scan=[0, SCAN_MAX])
    summary['key_comparisons'] = key_rows(sens)
    inv_check = check_inventory(inv)
    indep = independent_check(data)
    replay = json.loads((OUT/'replay-verification.json').read_text(encoding='utf-8'))
    run.save(OUT/'sensitivity.json', sens)
    run.save(OUT/'summary.json', summary)
    run.save(OUT/'inventory.json', dict(entries=inv, check=inv_check))
    run.save(OUT/'independent-arithmetic-audit.json', indep)
    run.save(OUT/'final-verdict.json', final_verdict(sens, summary, inv_check, indep, replay))
    run.save(OUT/'tables.json', tables(summary, sens))
    print('sensitivity comparisons:', len(sens), 'inventory:', len(inv),
          'independent checks:', indep['checks'], 'mismatches:', indep['mismatches'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operation', choices=('analyze',), default='analyze')
    p.parse_args()
    main()
