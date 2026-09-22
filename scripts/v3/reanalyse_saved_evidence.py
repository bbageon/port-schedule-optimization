"""Re-analyse the completed 80-run evidence without running any simulation (YR-317-l).

Every number here comes from files already written by the independent evaluation, the
block-only addition, and the stall scan. Nothing is re-simulated, no model is loaded and
no evidence file is modified. The secondary bootstrap uses its own declared seed so it is
never confused with the registered analyses (9,900,721 / 9,900,723).

Outputs land in outputs/reports/yr317_v3_reanalysis/ as one JSON per question plus a
summary.json and a human-readable results.md.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/reports/yr317_v3_reanalysis'
COMBINED = ROOT / 'outputs/reports/yr317_v3_block_only/run-de630ec-combined/month-results.json'
SCAN = ROOT / 'outputs/reports/yr317_v3_stall_diagnosis/scan-80.json'
PRIMARY_RUN = ROOT / 'outputs/reports/yr317_v3_independent_eval/run-7e2fb14/months'
ADDON_RUNS = (ROOT / 'outputs/reports/yr317_v3_block_only/run-de630ec/months',
              ROOT / 'outputs/reports/yr317_v3_block_only/run-ea9164b/months')
ARMS = ('NO_REALLOC', 'RL', 'RL_TIME', 'RL_SPACE')
ARM_KR = {'NO_REALLOC': '재배정없음', 'RL': '전체정책', 'RL_TIME': '시간정책', 'RL_SPACE': '블록정책'}
SET_KR = {'all_20': '전체 20시드', 'deadlock_free': '교착 없는 집합'}
LEARNED = ('RL', 'RL_TIME', 'RL_SPACE')
BOOTSTRAP_SEED = 9_900_725          # secondary analyses only; registered runs use ...721/...723
BOOTSTRAP_DRAWS = 20_000
THRESHOLDS_H = (None, 24.0, 12.0, 6.0, 3.0, 1.0)   # None = no filter


# ----------------------------------------------------------------- loading
def load_months():
    rows = json.loads(COMBINED.read_text(encoding='utf-8'))
    by = collections.defaultdict(dict)
    for r in rows:
        by[r['seed']][r['arm']] = r
    complete = {s: v for s, v in by.items() if set(v) == set(ARMS)}
    if len(complete) != len(by):
        raise SystemExit(f'incomplete seeds: {sorted(set(by) - set(complete))}')
    return dict(sorted(complete.items()))


def load_stalls():
    runs = json.loads(SCAN.read_text(encoding='utf-8'))['runs']
    out = {}
    for value in runs.values():
        month = value['month']
        unresolved = [b['longest_h'] for b in value['stalled_blocks'].values() if b['unresolved_at_end']]
        out[(month['seed'], month['arm'])] = max(unresolved, default=0.0)
    return out


def daily_path(seed, arm):
    direct = PRIMARY_RUN / str(seed) / arm / 'daily-final.jsonl'
    if direct.is_file():
        return direct
    for base in ADDON_RUNS:
        candidate = base / str(seed) / arm / 'daily-final.jsonl'
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'daily-final.jsonl for {seed}/{arm}')


def load_daily(seed, arm):
    return [json.loads(line) for line in daily_path(seed, arm).read_text(encoding='utf-8').splitlines() if line.strip()]


# ----------------------------------------------------------------- helpers
def kept_seeds(months, stalls, threshold_h, arms=ARMS):
    """Seeds whose every listed arm stayed below the unresolved-stall threshold."""
    if threshold_h is None:
        return list(months)
    return [s for s in months if all(stalls[(s, a)] < threshold_h for a in arms)]


def bootstrap_ci(values, level, seed):
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choice(values) for _ in range(n)) for _ in range(BOOTSTRAP_DRAWS))
    lo = means[int((1 - level) / 2 * BOOTSTRAP_DRAWS)]
    hi = means[min(BOOTSTRAP_DRAWS - 1, int((1 + level) / 2 * BOOTSTRAP_DRAWS) - 1)]
    return [lo, hi]


def trimmed_mean(values, fraction=0.10):
    ordered = sorted(values)
    cut = int(len(ordered) * fraction)
    core = ordered[cut:len(ordered) - cut] or ordered
    return statistics.fmean(core)


def pair_stats(months, seeds, arm, key='cost_28d_krw', level=0.975, seed=BOOTSTRAP_SEED):
    """Baseline minus `arm`: positive means the policy is cheaper."""
    diffs = [months[s]['NO_REALLOC'][key] - months[s][arm][key] for s in seeds]
    pct = [100 * d / months[s]['NO_REALLOC'][key] for d, s in zip(diffs, seeds)]
    ratios = [months[s][arm][key] / months[s]['NO_REALLOC'][key] for s in seeds]
    base_sum = sum(months[s]['NO_REALLOC'][key] for s in seeds)
    return dict(
        n=len(seeds), wins=sum(1 for d in diffs if d > 0),
        mean_saving_krw=statistics.fmean(diffs) if diffs else None,
        median_saving_pct=statistics.median(pct) if pct else None,
        mean_saving_pct=statistics.fmean(pct) if pct else None,
        ratio_of_sums_pct=100 * sum(diffs) / base_sum if base_sum else None,
        interval_level=level, interval_krw=bootstrap_ci(diffs, level, seed),
        trimmed_mean_saving_krw=trimmed_mean(diffs) if diffs else None,
        median_log_cost_ratio=statistics.median(math.log(r) for r in ratios) if ratios else None,
        per_seed_saving_pct={str(s): round(p, 3) for s, p in zip(seeds, pct)},
    )


# ----------------------------------------------------------------- questions
def q_covariate_balance(months, stalls):
    """Are the months removed by the stall filter different in their INPUTS?"""
    keep = kept_seeds(months, stalls, 24.0)
    drop = [s for s in months if s not in keep]

    def describe(group):
        requested = [months[s]['NO_REALLOC']['requested'] for s in group]
        loads = collections.Counter()
        vessel, heavy_days = [], []
        for s in group:
            daily = load_daily(s, 'NO_REALLOC')
            for d in daily:
                loads[d['load']] += 1
            heavy_days.append(sum(1 for d in daily if d['load'] >= 12500))
            vessel.append(months[s]['NO_REALLOC']['unfinished_vessel_jobs'])
        return dict(
            seeds=group, n=len(group),
            requested_mean=statistics.fmean(requested), requested_median=statistics.median(requested),
            heavy_days_mean=statistics.fmean(heavy_days),
            load_day_counts={str(k): v for k, v in sorted(loads.items())},
            baseline_cost_28d_median_krw=statistics.median(months[s]['NO_REALLOC']['cost_28d_krw'] for s in group),
            baseline_unfinished_vessel_median=statistics.median(vessel),
        )

    excluded, retained = describe(drop), describe(keep)
    pooled = statistics.pstdev([months[s]['NO_REALLOC']['requested'] for s in months]) or 1.0
    return dict(
        question='Do the seeds removed by the 24h unresolved-stall filter differ in inputs fixed before any policy ran?',
        excluded=excluded, retained=retained,
        requested_difference_pct=100 * (excluded['requested_mean'] - retained['requested_mean']) / retained['requested_mean'],
        requested_standardised_mean_difference=(excluded['requested_mean'] - retained['requested_mean']) / pooled,
        busiest_three_seeds=[s for _, s in sorted(((months[s]['NO_REALLOC']['requested'], s) for s in months), reverse=True)[:3]],
        verdict='필터가 더 바쁜 달을 걷어낸다 — 중립적 제외가 아니라 한계로 적어야 한다. / The filter removes the busier months; report it as a limitation, not a neutral exclusion.',
    )


def q_action_substitution(months, stalls):
    """Does enabling the spatial axis crowd out the temporal actions that pay?"""
    out = {}
    for label, seeds in (('all_20', list(months)), ('deadlock_free', kept_seeds(months, stalls, 24.0))):
        base = sum(months[s]['NO_REALLOC']['cost_28d_krw'] for s in seeds)
        rows = {}
        for arm in LEARNED:
            temporal = sum(months[s][arm]['temporal'] for s in seeds)
            spatial = sum(months[s][arm]['spatial'] for s in seeds)
            saving = base - sum(months[s][arm]['cost_28d_krw'] for s in seeds)
            rows[arm] = dict(temporal=temporal, spatial=spatial, total_actions=temporal + spatial,
                             saving_krw=saving,
                             krw_per_1000_actions=(1000 * saving / (temporal + spatial)) if temporal + spatial else None)
        rows['substitution_full_vs_time_only'] = dict(
            temporal_given_up=rows['RL_TIME']['temporal'] - rows['RL']['temporal'],
            spatial_added=rows['RL']['spatial'] - rows['RL_TIME']['spatial'],
            saving_lost_krw=rows['RL_TIME']['saving_krw'] - rows['RL']['saving_krw'])
        out[label] = dict(n=len(seeds), arms=rows)
    return dict(
        question='When the spatial axis is enabled, how many temporal actions are given up and what does that cost?',
        results=out,
        verdict='전체정책은 이득을 내던 시간 변경을 블록 변경으로 바꾸고 절감을 잃는다 — 공간 축 부정 결과의 기전이다. / Full trades paying temporal actions for spatial ones and loses the saving.')


def q_completion_normalised(months, stalls):
    """Is the cheaper cost just the result of finishing less work?"""
    out = {}
    for label, seeds in (('all_20', list(months)), ('deadlock_free', kept_seeds(months, stalls, 24.0))):
        rows = {}
        for arm in ARMS:
            cost = sum(months[s][arm]['cost_28d_krw'] for s in seeds)
            completed = sum(months[s][arm]['completed'] for s in seeds)
            requested = sum(months[s][arm]['requested'] for s in seeds)
            rows[arm] = dict(cost_krw=cost, completed=completed, requested=requested,
                             censored=sum(months[s][arm]['censored'] for s in seeds),
                             unbound=sum(months[s][arm]['unbound_at_cutoff'] for s in seeds),
                             unfinished_vessel=sum(months[s][arm]['unfinished_vessel_jobs'] for s in seeds),
                             completion_rate_pct=100 * completed / requested,
                             cost_per_completed_krw=cost / completed)
        base = rows['NO_REALLOC']['cost_per_completed_krw']
        for arm in LEARNED:
            per_seed_wins = sum(
                1 for s in seeds
                if months[s][arm]['cost_28d_krw'] / months[s][arm]['completed']
                < months[s]['NO_REALLOC']['cost_28d_krw'] / months[s]['NO_REALLOC']['completed'])
            rows[arm]['saving_per_completed_pct'] = 100 * (base - rows[arm]['cost_per_completed_krw']) / base
            rows[arm]['wins_per_completed'] = f'{per_seed_wins}/{len(seeds)}'
        out[label] = dict(n=len(seeds), arms=rows)
    return dict(
        question='Normalising by trucks actually completed, is the time-only saving still there?',
        results=out,
        verdict='시간정책은 같은 비율의 수요를 끝내고 완료 1대당 더 싸다 — 절감이 미완료 작업의 착시가 아니다. / Time-only completes the same share of demand and is cheaper per completed truck.')


def q_window(months, stalls):
    """Does the 28-day measurement window create the saving?"""
    out = {}
    for label, seeds in (('all_20', list(months)), ('deadlock_free', kept_seeds(months, stalls, 24.0))):
        out[label] = {arm: {w: pair_stats(months, seeds, arm, key=k)
                            for w, k in (('28d', 'cost_28d_krw'), ('30d', 'cost_30d_krw'))}
                      for arm in LEARNED}
        out[label]['n'] = len(seeds)
    return dict(
        question='Deferred trucks could be pushed past a 28-day cut-off; does the 30-day window remove the saving?',
        results=out,
        verdict='두 창에서 결론이 같다 — 창 선택이 자유도가 되지 않도록 둘 다 싣는다. / Conclusions agree on both windows; report both.')


def q_threshold(months, stalls):
    """How sensitive is the picture to where the stall filter is drawn?"""
    rows = []
    for threshold in THRESHOLDS_H:
        seeds = kept_seeds(months, stalls, threshold)
        entry = dict(threshold_h=threshold, n=len(seeds), excluded=[s for s in months if s not in seeds])
        for arm in LEARNED:
            entry[arm] = pair_stats(months, seeds, arm) if seeds else None
        rows.append(entry)
    return dict(
        question='Does the conclusion depend on the stall threshold?',
        declared='어느 문턱도 헤드라인으로 고르지 않는다. 등록된 무필터 행이 주판정으로 남는다. / No threshold is selected as the headline; the registered unfiltered row stays primary.',
        rows=rows)


def q_breakeven(months, stalls, deferral_hours=None):
    """How large an external rescheduling charge would erase the time-only saving?"""
    out = {}
    for label, seeds in (('all_20', list(months)), ('deadlock_free', kept_seeds(months, stalls, 24.0))):
        base = sum(months[s]['NO_REALLOC']['cost_28d_krw'] for s in seeds)
        saving = base - sum(months[s]['RL_TIME']['cost_28d_krw'] for s in seeds)
        changes = sum(months[s]['RL_TIME']['temporal'] for s in seeds)
        entry = dict(n=len(seeds), saving_krw=saving, time_changes=changes,
                     breakeven_krw_per_change=saving / changes if changes else None)
        if deferral_hours and label in deferral_hours:
            hours = deferral_hours[label]['total_deferral_hours']
            entry.update(total_deferral_hours=hours,
                         breakeven_krw_per_deferred_hour=saving / hours if hours else None,
                         mean_deferral_minutes=deferral_hours[label]['mean_deferral_minutes'])
        out[label] = entry
    return dict(
        question='Reviewer 1-2: external rescheduling cost is excluded. Above what charge does the benefit vanish?',
        results=out,
        caveat='상한값일 뿐이다 — 모형 절감을 변경된 예약 건수로 나눈 값이며 운송사가 거절하지 않는다고 가정한다. / Upper bound only; assumes no carrier declines.')


def measure_deferrals(months, stalls):
    """Sum the actual deferral applied by the time-only policy (reads request ledgers)."""
    groups = {'all_20': list(months), 'deadlock_free': kept_seeds(months, stalls, 24.0)}
    per_seed = {}
    for seed in months:
        path = None
        for base in (PRIMARY_RUN,) + ADDON_RUNS:
            candidate = base / str(seed) / 'RL_TIME' / 'requests.jsonl.gz'
            if candidate.is_file():
                path = candidate
                break
        if path is None:
            raise FileNotFoundError(f'requests ledger for {seed}/RL_TIME')
        total_s, changed = 0.0, 0
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                final = row.get('final_reserved_arrival_s')
                if final is None:
                    continue
                delta = final - row['requested_arrival_s']
                if delta > 1.0:
                    total_s += delta
                    changed += 1
        per_seed[seed] = dict(total_deferral_hours=total_s / 3600, changed=changed)
        print(json.dumps({'seed': seed, **per_seed[seed]}), flush=True)
    out = {}
    for label, seeds in groups.items():
        hours = sum(per_seed[s]['total_deferral_hours'] for s in seeds)
        changed = sum(per_seed[s]['changed'] for s in seeds)
        out[label] = dict(total_deferral_hours=hours, changed=changed,
                          mean_deferral_minutes=60 * hours / changed if changed else None)
    return dict(per_seed={str(k): v for k, v in per_seed.items()}, groups=out)


# ----------------------------------------------------------------- report
def write_report(results):
    cov, act = results['covariate_balance'], results['action_substitution']
    comp, win = results['completion_normalised'], results['window']
    thr, brk = results['threshold_sensitivity'], results['breakeven_rescheduling']
    lines = ['# 저장 기록 재분석 (시뮬레이션 0시간)', '',
             '완료된 80건의 저장 증거만 다시 계산했다. 새 실행·새 학습 0건.',
             f'보조 재표집 시드 {BOOTSTRAP_SEED:,} (등록 분석의 9,900,721 / 9,900,723 과 구분), 재표집 {BOOTSTRAP_DRAWS:,}회.', '',
             '## 1. 교착 제외 필터의 입력 공변량 균형', '',
             '| 집합 | 시드 수 | 평균 요청 트럭 | 혼잡일(12,500대 이상) 평균 | 기준정책 28일 비용 중앙 |', '|---|---|---|---|---|']
    for name, block in (('제외', cov['excluded']), ('잔존', cov['retained'])):
        lines.append(f"| {name} | {block['n']} | {block['requested_mean']:,.0f}대 | {block['heavy_days_mean']:.1f}일 | "
                     f"{block['baseline_cost_28d_median_krw']/1e8:,.1f}억 |")
    lines += ['', f"제외 집합이 요청 기준 **{cov['requested_difference_pct']:+.1f}%** 더 붐빈다 "
                  f"(표준화 평균차 {cov['requested_standardised_mean_difference']:+.2f}). "
                  f"가장 붐비는 시드 셋 {', '.join(f'{s:,}' for s in cov['busiest_three_seeds'])}.", '',
              f"→ {cov['verdict']}", '',
              '## 2. 행동 대체 — 공간 축을 켜면 무엇을 포기하나 (교착 없는 집합)', '',
              '| 정책 | 시간 변경 | 블록 변경 | 기준 대비 절감 | 행동 1,000건당 |', '|---|---|---|---|---|']
    free = act['results']['deadlock_free']['arms']
    for arm in LEARNED:
        row = free[arm]
        lines.append(f"| {ARM_KR[arm]} | {row['temporal']:,} | {row['spatial']:,} | {row['saving_krw']/1e8:+,.1f}억 | "
                     f"{row['krw_per_1000_actions']/1e8:+.3f}억 |")
    sub = free['substitution_full_vs_time_only']
    lines += ['', f"전체정책은 시간정책보다 시간 변경을 **{sub['temporal_given_up']:,}건 포기**하고 "
                  f"블록 변경 {sub['spatial_added']:,}건을 더하며, 그 대가로 절감 {abs(sub['saving_lost_krw'])/1e8:,.1f}억을 잃는다.", '',
              '## 3. 완료 대수로 보정한 비용 (교착 없는 집합)', '',
              '| 정책 | 완료율 | 완료 1대당 비용 | 기준 대비 | 대당 기준 이긴 달 |', '|---|---|---|---|---|']
    cf = comp['results']['deadlock_free']['arms']
    lines.append(f"| 재배정없음 | {cf['NO_REALLOC']['completion_rate_pct']:.2f}% | {cf['NO_REALLOC']['cost_per_completed_krw']:,.0f}원 | — | — |")
    for arm in LEARNED:
        row = cf[arm]
        lines.append(f"| {ARM_KR[arm]} | {row['completion_rate_pct']:.2f}% | {row['cost_per_completed_krw']:,.0f}원 | "
                     f"{row['saving_per_completed_pct']:+.1f}% | {row['wins_per_completed']} |")
    lines += ['', f"→ {comp['verdict']}", '', '## 4. 28일 창과 30일 창', '',
              '| 집합 | 정책 | 28일 승/중앙 | 30일 승/중앙 |', '|---|---|---|---|']
    for label in ('all_20', 'deadlock_free'):
        block = win['results'][label]
        for arm in LEARNED:
            a, b = block[arm]['28d'], block[arm]['30d']
            lines.append(f"| {SET_KR.get(label, label)} | {ARM_KR[arm]} | {a['wins']}/{a['n']} · {a['median_saving_pct']:+.2f}% | "
                         f"{b['wins']}/{b['n']} · {b['median_saving_pct']:+.2f}% |")
    lines += ['', '## 5. 교착 문턱 민감도 (어느 문턱도 헤드라인으로 고르지 않는다)', '',
              '| 문턱 | 남은 시드 | ' + ' | '.join(ARM_KR[a] + ' 승/중앙' for a in LEARNED) + ' |', '|---|---|---|---|---|']
    for row in thr['rows']:
        name = '무필터' if row['threshold_h'] is None else f"{row['threshold_h']:.0f}시간"
        cells = []
        for arm in LEARNED:
            v = row[arm]
            cells.append(f"{v['wins']}/{v['n']} · {v['median_saving_pct']:+.1f}%" if v else '—')
        lines.append(f"| {name} | {row['n']} | " + ' | '.join(cells) + ' |')
    lines += ['', '## 6. 손익분기 재예약 비용 (리뷰 1-2 부분 대응)', '']
    for label, row in brk['results'].items():
        name = SET_KR.get(label, label)
        if row['saving_krw'] <= 0:
            lines.append(f"- {name}: 모형 절감이 음수({row['saving_krw']/1e8:,.1f}억)라 손익분기가 정의되지 않는다 "
                         f"— 교착 실행이 비용을 지배하기 때문이며, 아래 교착 없는 집합으로만 계산한다.")
            continue
        extra = ''
        if row.get('breakeven_krw_per_deferred_hour'):
            extra = (f", 지연 1시간당 **{row['breakeven_krw_per_deferred_hour']:,.0f}원**"
                     f" (평균 지연 {row['mean_deferral_minutes']:.1f}분)")
        lines.append(f"- {name}: 절감 {row['saving_krw']/1e8:,.1f}억 ÷ 시간 변경 {row['time_changes']:,}건 → "
                     f"변경 1건당 **{row['breakeven_krw_per_change']:,.0f}원**{extra} 이상이면 이득이 사라진다.")
    lines += ['', f"→ {brk['caveat']}", '']
    (OUT / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', action='store_true', help='also read the 20 request ledgers for deferral hours (slow)')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    months, stalls = load_months(), load_stalls()

    deferrals = None
    if args.ledger:
        deferrals = measure_deferrals(months, stalls)
        (OUT / 'deferral-measurement.json').write_text(json.dumps(deferrals, ensure_ascii=False, indent=1), encoding='utf-8')
    elif (OUT / 'deferral-measurement.json').is_file():
        deferrals = json.loads((OUT / 'deferral-measurement.json').read_text(encoding='utf-8'))

    results = dict(
        scope='Re-analysis of saved evidence only; no simulation, no training, no evidence file modified.',
        bootstrap=dict(seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS,
                       note='Secondary analyses. The registered analyses keep seeds 9,900,721 and 9,900,723.'),
        covariate_balance=q_covariate_balance(months, stalls),
        action_substitution=q_action_substitution(months, stalls),
        completion_normalised=q_completion_normalised(months, stalls),
        window=q_window(months, stalls),
        threshold_sensitivity=q_threshold(months, stalls),
        breakeven_rescheduling=q_breakeven(months, stalls, deferrals['groups'] if deferrals else None),
    )
    for name, value in results.items():
        if isinstance(value, dict) and 'question' in value:
            (OUT / f"{name.replace('_', '-')}.json").write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf-8')
    (OUT / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    write_report(results)
    print(json.dumps(dict(seeds=len(months), outputs=sorted(p.name for p in OUT.iterdir())), ensure_ascii=False))


if __name__ == '__main__':
    main()
