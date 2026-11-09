"""Independently audit all K corpora, weights and raw order rows, then summarize."""
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
import statistics as st

from benchmark_default_eval import save
from six_step_learning import gz_read, sha, load_net, labelset
from horizon_sweep import OUT, SOURCE, SEEDS, LOADS, KS, CASES, name, folder, common_choices, check_sources
from yard_rl.integrated.evalkit import paired
from yard_rl.v7.evaluation_probe import digest

ARMS = ('initial', 'k1', 'k2', 'k3', 'k6', 'sf_spt')
METRICS = ('TRUCK', 'GATE_IN', 'GATE_OUT', 'VESSEL_DISCHARGE', 'VESSEL_LOAD', 'ALL_ORDERS')
LABELS = dict(TRUCK='전체 트럭', GATE_IN='반입', GATE_OUT='반출',
              VESSEL_DISCHARGE='양하', VESSEL_LOAD='적하', ALL_ORDERS='모든 오더 혼합')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-6), (a, b)


def independent_times(rows, end):
    """Recount stored events without calling the measurement implementation."""
    result = {}
    for group in (*METRICS, 'VESSEL'):
        selected = [r for r in rows if r['origin_s'] <= end and
            (r['flow'] == group or group == 'ALL_ORDERS' or
             group == 'TRUCK' and r['flow'].startswith('GATE_') or
             group == 'VESSEL' and r['flow'].startswith('VESSEL_'))]
        if not selected:
            continue
        total, unfinished = 0., 0
        ready_values = []
        for r in selected:
            finish = r['finish_s']
            if finish is None or finish > end:
                total += end-r['origin_s']
                unfinished += 1
            else:
                total += finish-r['origin_s']
            if r['ready_s'] is not None and r['ready_s'] <= end:
                finish = r['yard_done_s']
                ready_values.append((end if finish is None or finish > end else finish)-r['ready_s'])
        result[group] = dict(n=len(selected), total_s=total, mean_s=total/len(selected),
            unfinished=unfinished, not_ready=len(selected)-len(ready_values),
            ready_n=len(ready_values), ready_mean_s=st.fmean(ready_values) if ready_values else None)
    return result


def verify_training():
    import torch
    from yard_rl.v7.crane.fit import CraneTrainer
    torch.set_num_threads(1)
    artifacts, results = {}, []
    for k in KS:
        totals = Counter()
        windows, gaps = [], []
        for seed in SEEDS:
            initial = SOURCE/f'model-{seed}'/'initial.pt'
            net = load_net(initial)
            trainer = CraneTrainer(net, loss_kind='pairwise')
            base = SOURCE/f'model-{seed}' if k == 3 else folder(k, seed)
            stored = read(base/'training.json')['rows']
            for it in range(6):
                cp = base/(f'reachable-{it}.pt' if k == 3 else f'final-{it}.pt')
                path = base/(f'labels-reachable-{it}.json.gz' if k == 3 else f'labels-{it}.json.gz')
                data = gz_read(path)
                reference = gz_read(SOURCE/f'model-{seed}'/f'labels-reachable-{it}.json.gz')
                assert common_choices(data) == common_choices(reference)
                assert data['input_seed'] == 9914800+100*SEEDS.index(seed)+10*it
                assert data['load'] == LOADS[it % 3] and data['behaviour_sha256'] == sha(initial)
                record, = [r for r in stored if r['iteration'] == it and
                           (k != 3 or r['contract'] == 'reachable')]
                assert record['corpus_sha256'] == sha(path)
                c = data['result']['crane']
                assert data['result']['policy_exceptions'] == c['force_failed'] == c['factual_mismatch'] == c['hook_errors'] == 0
                assert len(data['samples']) == 2*c['labeled']
                assert c['sampled'] == c['labeled']+c['no_witness']
                for a, b, pair, old_pair in zip(data['samples'][::2], data['samples'][1::2],
                        c['comparisons'], reference['result']['crane']['comparisons'], strict=True):
                    assert a['phi'] == pair['phi_factual'] and b['phi'] == pair['phi_alt']
                    assert a['decision_key'] == b['decision_key'] == pair['key']
                    assert pair['factual_ok'] and pair['forced_ok']
                    assert pair['requested_end_s'] == pair['t']+3600.*k
                    assert pair['cost_end_s'] == min(pair['requested_end_s'], 93600.)
                    close(pair['effective_horizon_s'], pair['cost_end_s']-pair['t'])
                    windows.append(pair['effective_horizon_s'])
                    gap = pair['phi_alt']-pair['phi_factual']
                    gaps.append(gap)
                    old_gap = old_pair['phi_alt']-old_pair['phi_factual']
                    totals['sign_flip_vs_k3'] += gap*old_gap < 0
                    totals['zero_raw_gap'] += gap == 0
                    totals['truncated'] += pair['cost_end_s'] < pair['requested_end_s']
                fit = trainer.fit(labelset(data), seed=data['input_seed'])
                assert fit.steps == 50 and asdict(fit) == record['fit']
                ck = torch.load(cp, weights_only=True)
                assert all(torch.equal(value, ck['crane'][key]) for key, value in net.state_dict().items())
                # Verify optimizer moments as well as predictions/weights.
                current = trainer.opt.state_dict()
                assert current['param_groups'] == ck['opt']['param_groups']
                for param, states in current['state'].items():
                    for key, value in states.items():
                        other = ck['opt']['state'][param][key]
                        assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
                totals['exact_refits'] += 1
                for key in ('sampled', 'labeled', 'no_witness', 'zero', 'worlds'):
                    totals[key] += c[key]
                artifacts[path.as_posix()], artifacts[cp.as_posix()] = sha(path), sha(cp)
            artifacts[(base/'training.json').as_posix()] = sha(base/'training.json')
        results.append(dict(k=k, totals=dict(totals), effective_horizon_s=dict(
            min=min(windows), max=max(windows), mean=st.fmean(windows)),
            median_absolute_cost_gap=st.median(map(abs, gaps))))
        print(f'K={k}: 18 exact weight/optimizer refits, same questions, bounded costs', flush=True)
    return results, artifacts


def verify_evaluations():
    from evaluate_arrival_order import HASHES
    artifacts, inputs, cohorts, values = {}, {}, {}, {}
    truck_rows = flow_rows = old_replays = 0
    for case in CASES:
        arm, seed, load, day, repeat = case
        p = OUT/'evaluation'/f'{name(case)}.json.gz'
        q = OUT/'orders'/p.name
        f = OUT/'flow-orders'/p.name
        data, trucks, rows = gz_read(p), gz_read(q), gz_read(f)
        assert data['case'] == list(case) and data['input_seed'] == 9915900+100*day+10*LOADS.index(load)
        assert len(trucks) == data['result']['admitted'] == load
        assert data['result']['policy_exceptions'] == data['result']['rollout_worlds'] == data['result']['traded_edges'] == 0
        assert data['probe']['ledger_audit']['skipped'] == 0
        assert len(rows) == load+data['probe']['metadata']['initial_vessel_jobs']
        assert len({r['job_id'] for r in rows}) == len(rows)
        row_map = {r['job_id']: r for r in rows}
        for truck in trucks:
            assert truck['admitted'] and truck['reason'] is None and truck['review_count'] == 1
            t, r = truck['times'], row_map[truck['job_id']]
            assert r['flow'] == truck['flow'] and r['origin_s'] == t['gate_in_s'] == truck['scheduled_gate_s']
            assert r['ready_s'] == t['block_in_s'] == t['gate_in_s']+truck['travel_s']
            assert r['finish_s'] == t['gate_out_s'] and r['yard_done_s'] == t['job_done_s']
        for r in rows:
            stamps = [r[k] for k in ('origin_s', 'ready_s', 'service_start_s', 'yard_done_s', 'finish_s') if r[k] is not None]
            assert all(a <= b+1e-6 for a, b in zip(stamps, stamps[1:])), r
        cohort = [{k: r[k] for k in ('job_id', 'block', 'flow', 'origin_s', 'planned_release_s', 'stream')} for r in rows]
        timing = data['probe']['flow_timing']
        assert digest(cohort) == timing['cohort_sha256'] and digest(rows) == timing['rows_sha256']
        assert cohorts.setdefault((load, day), digest(cohort)) == digest(cohort)
        identity = data['probe']['metadata']['input_sha256']
        assert inputs.setdefault((load, day), identity) == identity
        m = dict(cost=data['result']['phi_krw'])
        for end, suffix in ((86400., '24'), (93600., '26')):
            independent = independent_times(rows, end)
            stored = timing['at'+suffix]
            assert independent.keys() == stored.keys()
            for group, a in independent.items():
                for key, value in a.items():
                    if value is None:
                        assert stored[group][key] is None
                    else:
                        close(value, stored[group][key])
                for key, value in a.items():
                    m[f'{group}_{key}_{suffix}'] = value
        close(m['TRUCK_mean_s_24'], data['result']['mean_turn_time_s'])
        assert m['TRUCK_unfinished_24'] == data['result']['n_censored']
        for key in ('vessel_unfinished', 'vessel_idle_s', 'vessel_delay_s'):
            m[key] = data['probe']['operational'][key]
        if data['checkpoint']:
            assert sha(data['checkpoint']) == data['checkpoint_sha256']
        if arm in ('initial', 'k3', 'sf_spt'):
            old_arm = 'reachable-5' if arm == 'k3' else arm
            old = SOURCE/'evaluation'/f'{old_arm}-{seed}-{load}-{day}-0.json.gz'
            previous = gz_read(old)
            assert previous['result'] == data['result']
            assert all(previous['probe'][k] == data['probe'][k] for k in HASHES)
            assert trucks == gz_read(SOURCE/'orders'/old.name)
            old_replays += 1
        if repeat:
            original = (*case[:-1], 0)
            old = gz_read(OUT/'evaluation'/f'{name(original)}.json.gz')
            assert old['result'] == data['result']
            assert all(old['probe'][k] == data['probe'][k] for k in HASHES)
            assert old['probe']['flow_timing'] == timing
            assert rows == gz_read(OUT/'flow-orders'/f'{name(original)}.json.gz')
            assert trucks == gz_read(OUT/'orders'/f'{name(original)}.json.gz')
        values[case] = m
        truck_rows += len(trucks)
        flow_rows += len(rows)
        for path in (p, q, f):
            artifacts[path.as_posix()] = sha(path)
    assert len(inputs) == len(cohorts) == 9 and old_replays == 63
    return values, artifacts, dict(runs=len(CASES), truck_rows=truck_rows, all_order_rows=flow_rows,
        independent_inputs=9, previous_runs_exact=old_replays, new_repeats_exact=3,
        admitted_all=True, fixed_cohort=True)


def summarize(values, training, verified):
    groups = defaultdict(list)
    for (arm, seed, load, day, repeat), v in values.items():
        if not repeat:
            groups[(arm, load, day)].append(v)
    daily = {key: {metric: st.fmean(r[metric] for r in rows)
                  for metric in rows[0] if all(r[metric] is not None for r in rows)}
             for key, rows in groups.items()}
    comparisons, by_load, aggregate, guards, per_seed = [], [], [], {}, []
    for arm in ARMS:
        day_rows = [v for (a, _, _), v in daily.items() if a == arm]
        aggregate.append(dict(arm=arm, **{key: st.fmean(v[key] for v in day_rows)
            for key in day_rows[0] if all(key in v for v in day_rows)}))
        for load in LOADS:
            chosen = [daily[(arm, load, day)] for day in range(3)]
            by_load.append(dict(arm=arm, load=load, **{key: st.fmean(v[key] for v in chosen)
                for key in chosen[0] if all(key in v for v in chosen)}))
    for k in KS:
        arm = f'k{k}'
        final = [v for (a, _, _, _, rep), v in values.items() if a == arm and not rep]
        guards[arm] = dict(completion=all(v['TRUCK_unfinished_26'] == v['vessel_unfinished'] ==
                v['VESSEL_unfinished_26'] == 0 for v in final),
            backlog_zero=all(v['ALL_ORDERS_unfinished_26'] == 0 for v in final),
            physical_valid=True,
            vessel_protection=all(values[(arm, s, l, d, 0)][m] <= values[('sf_spt', 0, l, d, 0)][m]+1e-6
                for s in SEEDS for l in LOADS for d in range(3)
                for m in ('vessel_unfinished', 'vessel_idle_s', 'vessel_delay_s', 'VESSEL_unfinished_26')))
        for control in ('initial', 'k3', 'sf_spt'):
            for metric in ('cost', *[f'{m}_mean_s_{end}' for end in ('24', '26') for m in METRICS]):
                x = [daily[(arm, l, d)][metric] for l in LOADS for d in range(3)]
                y = [daily[(control, l, d)][metric] for l in LOADS for d in range(3)]
                change = [100*(a-b)/b for a, b in zip(x, y)]
                reduction = [-v for v in change]
                signed = asdict(paired(change, delta_interest=1.))
                reduced = asdict(paired(reduction, delta_interest=1.))
                # evalkit's label assumes negative is good; our reduction is positive-good.
                reduced['label'] = signed['label']
                reduced['sign_convention'] = 'positive is improvement'
                row = dict(k=k, control=control, metric=metric, candidate_mean=st.fmean(x),
                    control_mean=st.fmean(y), reduction_percent=reduced,
                    candidate_minus_control_percent=signed,
                    paired_raw=asdict(paired([a-b for a, b in zip(x, y)])))
                if metric != 'cost':
                    count_key = metric.replace('_mean_s_', '_n_')
                    counts = [daily[(arm, l, d)][count_key] for l in LOADS for d in range(3)]
                    row['pooled_order_time_reduction_percent'] = 100*(1-sum(a*n for a, n in zip(x, counts))/sum(b*n for b, n in zip(y, counts)))
                comparisons.append(row)
            for seed in SEEDS:
                a = [values[(arm, seed, l, d, 0)]['TRUCK_mean_s_24'] for l in LOADS for d in range(3)]
                b = [values[(control, 0 if control == 'sf_spt' else seed, l, d, 0)]['TRUCK_mean_s_24'] for l in LOADS for d in range(3)]
                per_seed.append(dict(k=k, control=control, model_seed=seed,
                    truck_reduction_percent=st.fmean(100*(y-x)/y for x, y in zip(a, b))))
    return dict(**verified, training=training, aggregate=aggregate, by_load=by_load,
        comparisons=comparisons, guards=guards, per_seed=per_seed,
        sign_convention='positive reduction is shorter; negative reduction is worse',
        scope='fixed synthetic horizon diagnostic; no independent holdout or actual port efficacy claim')


def write_report(summary):
    table = {r['arm']: r for r in summary['aggregate']}
    comp = lambda k, m, control='initial', end='24': next(r for r in summary['comparisons']
        if r['k'] == k and r['control'] == control and r['metric'] == f'{m}_mean_s_{end}')
    lines = ['# K=1·2·3·6시간 학습 비교', '',
        'K는 선택 뒤 미래 비용을 비교하는 시간이다. 하루 평가 길이와 정책 구조는 동일하다.',
        '초기값 3개·6회 갱신·9개 입력. K3 학습 18회는 기존 검증 자료 재사용, 나머지 54회는 신규 실행.',
        f'평가 {summary["runs"]}회(144회+반복 3회), 주문 기록 {summary["all_order_rows"]:,}건을 전수 검산했다.', '',
        '## 시간 정의', '',
        '- 전체 트럭·반입·반출: 실제 게이트 진입→출차.',
        '- 양하·적하: 해당 본선 작업 시작 예정시각→개별 야드 작업 완료. 선박 체류시간은 아니다.',
        '- 모든 오더 혼합: 위 두 정의를 주문 수로 가중한 참고 지표. 트럭 체류시간과 구분한다.',
        '- 24시간까지 관측하며 미완료·미도착 주문도 그때까지의 시간을 포함한다. 완주 평균은 아니다.',
        '- 감소율=(학습 전−학습 후)/학습 전. **양수는 개선, 음수는 악화**다.',
        '- 초기값 셋을 먼저 평균하고 입력 9개의 평균/감소율을 같은 비중으로 평균했다.', '',
        '## 업무별 평균 시간과 감소율', '',
        '단위: 분. 괄호는 학습 전 대비 감소율이다.', '',
        '| 구분 | 학습 전 | K1 | K2 | K3 | K6 |', '|---|---:|---:|---:|---:|---:|']
    for metric in METRICS:
        cells = [f'{table["initial"][metric+"_mean_s_24"]/60:.3f}']
        for k in KS:
            cells.append(f'{table[f"k{k}"][metric+"_mean_s_24"]/60:.3f} ({comp(k, metric)["reduction_percent"]["mean"]:+.3f}%)')
        lines.append('| '+LABELS[metric]+' | '+' | '.join(cells)+' |')
    lines += ['', '평균 방식 때문에 표의 평균 분 수끼리 계산한 비율과 입력별 감소율 평균은 조금 다를 수 있다.', '',
        '## 부하별 전체 트럭 평균', '',
        '| 하루 주문 | 학습 전 | K1 | K2 | K3 | K6 | 작업 규칙 |', '|---|---:|---:|---:|---:|---:|---:|']
    for load in LOADS:
        t = {r['arm']: r for r in summary['by_load'] if r['load'] == load}
        lines.append(f'| {load} | '+' | '.join(f'{t[a]["TRUCK_mean_s_24"]/60:.3f}' for a in ARMS)+' |')
    lines += ['', '작업 규칙은 SF-SPT: 실행 가능한 실작업 중 예상 작업시간이 짧은 작업을 우선한다.', '',
        '## 비교 기준과 불확실성', '',
        '| K | 학습 전 대비 감소율 [95% 구간] | K3 대비 | 규칙 대비 | 26시간 관측·학습 전 대비 |',
        '|---|---:|---:|---:|---:|']
    for k in KS:
        r = comp(k, 'TRUCK')['reduction_percent']
        lines.append(f'| {k} | {r["mean"]:+.3f}% [{r["ci_lo"]:+.3f}, {r["ci_hi"]:+.3f}] | '+
            f'{comp(k, "TRUCK", "k3")["reduction_percent"]["mean"]:+.3f}% | '+
            f'{comp(k, "TRUCK", "sf_spt")["reduction_percent"]["mean"]:+.3f}% | '+
            f'{comp(k, "TRUCK", end="26")["reduction_percent"]["mean"]:+.3f}% |')
    lines += ['', '구간은 같은 입력끼리 비교한 9개 차이의 불확실성을 표시한다. 네 K 탐색에 대한 다중비교 보정 구간은 아니다.',
        '이미 사용했던 평가 입력이며 독립 잠금 확증으로 간주하지 않는다. 초기값별 결과는 summary.json에 있다.', '',
        '## 24시간 미완료와 미도착', '',
        '| 정책 | 트럭 미출차 | 양하 미완료 | 양하 미도착 | 적하 미완료 |', '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        r = table[arm]
        lines.append(f'| {arm} | '+' | '.join(f'{r[key]:.3f}' for key in (
            'TRUCK_unfinished_24', 'VESSEL_DISCHARGE_unfinished_24',
            'VESSEL_DISCHARGE_not_ready_24', 'VESSEL_LOAD_unfinished_24'))+' |')
    lines += ['', '건수는 여러 실행의 평균이라 소수다. 본선 미도착을 제외한 야드 대기 평균으로 주 판정을 바꾸지 않았다.', '',
        '## 학습 신호와 실제 창', '',
        '| K | 비교 쌍 | 비용 차이 0 | K3과 우열 반전 | 종료 경계에서 잘린 쌍 | 실제 평균 창(시간) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary['training']:
        c = r['totals']
        lines.append(f'| {r["k"]} | {c["labeled"]} | {c["zero_raw_gap"]} | {c["sign_flip_vs_k3"]} | '+
                     f'{c["truncated"]} | {r["effective_horizon_s"]["mean"]/3600:.3f} |')
    lines += ['', '모든 K의 행동 궤적·표집 질문·대안·입력은 정확히 같았다. 다른 것은 비교 종료시각과 거기서 생긴 비용 차이다.',
        '26시간 환경 종료를 넘는 비용은 포함하지 않았다. 각 K의 가중치와 최적화기 상태 18개씩, 총 72개를 다시 학습해 일치 확인했다.', '',
        '## 해석 범위와 예정사항', '',
        '이번 실험은 고정된 합성 환경에서 K의 영향을 측정했다. 가장 좋은 수치 하나만으로 K를 채택하거나 실항만 최적화를 주장하지 않는다.',
        '본선 보호·미완료·규칙 대비 비용을 함께 판정한다. 실측 운반·예약 보정 미확정은 유지한다.',
        '등록된 YR-332-i는 실제 운반차 투입·이동·인계 조건 보정, YR-332-f는 예약과 실제 도착 차이 보정이다.',
        '실제 항만에 대한 주장을 하려면 그 자료 근거가 필요하다.', '',
        '[사전등록](../../../docs/research/v7-discharge/horizon-sweep-prereg.md) · [원값](summary.json) · [검증](verification.json)']
    assert len(lines) <= 200
    (OUT/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')


def report():
    check_sources()
    training, artifacts = verify_training()
    values, more, verified = verify_evaluations()
    artifacts.update(more)
    summary = summarize(values, training, verified)
    save(OUT/'summary.json', summary)
    write_report(summary)
    save(OUT/'verification.json', dict(status='PASS', **verified, exact_refits=72,
        source_hashes_verified=True, artifact_sha256=artifacts,
        summary_sha256=sha(OUT/'summary.json'), report_sha256=sha(OUT/'README.md')))
    print(f'Verified {len(CASES)} runs, {verified["all_order_rows"]} rows, 72 refits.', flush=True)


if __name__ == '__main__':
    report()
