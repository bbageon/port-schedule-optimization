"""Independent corpus/ledger verification and paired daily learning report."""
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
import statistics as st
import xml.etree.ElementTree as ET

from benchmark_default_eval import save
from six_step_learning import gz_read, sha, load_net
from retrain_reachable_teacher import OUT, SEEDS, LOADS, CONTRACTS, RULES, ARMS, CASES, name
from yard_rl.integrated.evalkit import paired


def metrics(data, rows):
    n = len(rows)
    assert n == data['result']['admitted'] == data['probe']['metadata']['truck_orders']
    assert len({r['job_id'] for r in rows}) == n
    for row in rows:
        assert row['admitted'] and row['reason'] is None and row['review_count'] == 1
        t = row['times']
        stamps = [t[k] for k in ('gate_in_s','block_in_s','service_start_s','job_done_s','gate_out_s')
                  if t[k] is not None]
        assert all(a <= b+1e-6 for a, b in zip(stamps, stamps[1:]))
        assert t['gate_in_s'] == row['scheduled_gate_s']
        assert t['block_in_s'] == t['gate_in_s'] + row['travel_s']
    m = dict(missing=0, cost=data['result']['phi_krw'])
    for end, suffix in ((86400., '24'), (93600., '26')):
        m['turn_s'+suffix] = sum(min(r['times']['gate_out_s'] or end,end)-r['times']['gate_in_s']
                                   for r in rows)/n
        m['unexited'+suffix] = sum(r['times']['gate_out_s'] is None or
                                    r['times']['gate_out_s'] > end for r in rows)
        m['yard_unfinished'+suffix] = sum(r['times']['job_done_s'] is None or
                                           r['times']['job_done_s'] > end for r in rows)
    assert math.isclose(m['turn_s24'], data['result']['mean_turn_time_s'], rel_tol=1e-11)
    assert m['unexited24'] == data['result']['n_censored']
    op = data['probe']['operational']
    for k in ('vessel_unfinished', 'vessel_idle_s', 'vessel_delay_s'):
        m[k] = op[k]
    return m


def verify_training():
    import torch
    from yard_rl.v7.crane.fit import CraneTrainer
    from six_step_learning import labelset
    torch.set_num_threads(1)
    totals = {arm: Counter() for arm in CONTRACTS}
    artifacts, coverage = {}, []
    for seed in SEEDS:
        folder = OUT/f'model-{seed}'
        record = json.loads((folder/'training.json').read_text(encoding='utf-8'))
        assert len(record['rows']) == 12
        nets = {a: load_net(folder/'initial.pt') for a in CONTRACTS}
        trainers = {a: CraneTrainer(net, loss_kind='pairwise') for a, net in nets.items()}
        initial = load_net(folder/'initial.pt').state_dict()
        for it in range(6):
            common = None
            for arm in CONTRACTS:
                path = folder/f'labels-{arm}-{it}.json.gz'
                data = gz_read(path)
                c, samples = data['result']['crane'], data['samples']
                assert c['alternative_contract'] == arm
                assert c['hook_errors'] == c['factual_mismatch'] == 0
                assert c['labeled']*2 == len(samples) >= 4
                assert data['behaviour_sha256'] == sha(folder/'initial.pt')
                assert data['result']['policy_exceptions'] == 0
                actual = [r for r in c['comparisons'] if r['factual_ok'] and r['forced_ok']]
                assert len(actual) == c['labeled']
                for r, a, b in zip(actual, samples[::2], samples[1::2], strict=True):
                    assert r['phi_factual'] == a['phi'] and r['phi_alt'] == b['phi']
                    assert r['key'] == a['decision_key'] == b['decision_key']
                    assert len(a['row']) == len(b['row']) == 10
                    if arm == 'reachable':
                        assert r['assignment_alt'] == r['witness']['assignment']
                        assert a['row'] != b['row']
                    coverage.append(dict(model=seed, iteration=it, contract=arm,
                        picked_flow=a['flow'], alternative_flow=b['flow'],
                        gap=r['phi_alt']-r['phi_factual']))
                if arm == 'reachable':
                    assert c['force_failed'] == 0
                    assert c['sampled'] == c['labeled']+c['no_witness']
                    assert len(c['witness_search']) == c['sampled']
                    for search in c['witness_search']:
                        totals[arm]['attempts'] += len(search['attempts'])
                        for attempt in search['attempts']:
                            totals[arm]['reason/'+attempt['reason']] += 1
                        totals[arm]['fallback_success'] += bool(search['witness'] and len(search['attempts']) > 1)
                for k in ('sampled','labeled','no_witness','force_failed','factual_mismatch','zero','worlds'):
                    totals[arm][k] += c.get(k, 0)
                value = [data['result'][k] for k in ('phi_krw','admitted','n_censored')]
                value += [c['main_choices_sha256'], c['sampled']]
                if common is None:
                    common = value
                else:
                    assert common == value
                # Refit the published corpus from the same starting weights.
                fit = trainers[arm].fit(labelset(data), seed=data['input_seed'])
                stored = load_net(folder/f'{arm}-{it}.pt').state_dict()
                assert fit.steps == 50
                assert all(torch.equal(v, stored[k]) for k,v in nets[arm].state_dict().items())
                artifacts[path.as_posix()] = sha(path)
                artifacts[(folder/f'{arm}-{it}.pt').as_posix()] = sha(folder/f'{arm}-{it}.pt')
        assert any(not torch.equal(v, initial[k]) for k,v in nets['reachable'].state_dict().items())
        for f in ('initial.pt','training.json'):
            artifacts[(folder/f).as_posix()] = sha(folder/f)
    return dict(totals={a:dict(c) for a,c in totals.items()},
                exact_refit_checkpoints=36, common_behaviour_pairs=18, coverage=coverage), artifacts


def report():
    manifest = json.loads((OUT/'campaign.json').read_text(encoding='utf-8'))
    for path, expected in manifest['sources'].items():
        assert sha(path) == expected, path
    training, artifacts = verify_training()
    values, inputs = {}, {}
    checked = 0
    for case in CASES:
        p = OUT/'evaluation'/(name(case)+'.json.gz')
        q = OUT/'orders'/p.name
        data, rows = gz_read(p), gz_read(q)
        assert data['case'] == list(case) and len(rows) == case[2]
        assert data['result']['policy_exceptions'] == data['result']['rollout_worlds'] == 0
        if data['checkpoint']:
            assert sha(data['checkpoint']) == data['checkpoint_sha256']
        key = (case[2],case[3])
        identity = data['probe']['metadata']['input_sha256']
        assert inputs.setdefault(key, identity) == identity
        values[case] = metrics(data, rows)
        checked += len(rows)
        artifacts[p.as_posix()], artifacts[q.as_posix()] = sha(p), sha(q)
    repeat = CASES[-1]
    original = (*repeat[:-1], 0)
    a, b = [gz_read(OUT/'evaluation'/(name(c)+'.json.gz')) for c in (repeat,original)]
    assert a['result'] == b['result'] and values[repeat] == values[original]
    assert gz_read(OUT/'orders'/(name(repeat)+'.json.gz')) == gz_read(OUT/'orders'/(name(original)+'.json.gz'))
    for k in ('candidates_sha256','choices_sha256','events_sha256','jobs_sha256','records_sha256'):
        assert a['probe'][k] == b['probe'][k]
    groups = defaultdict(list)
    for (arm,seed,load,day,rep), value in values.items():
        if not rep:
            groups[(arm,load)].append(value)
    aggregate = [dict(arm=arm,load=load,runs=len(rows),
        **{k:st.mean(r[k] for r in rows) for k in rows[0]}) for (arm,load), rows in groups.items()]
    day_values = {}
    for arm in (*ARMS,*RULES):
        for load in LOADS:
            for day in range(3):
                rows = [v for (a,s,l,d,r),v in values.items() if (a,l,d,r)==(arm,load,day,0)]
                day_values[(arm,load,day)] = {k:st.mean(r[k] for r in rows) for k in rows[0]}
    comparisons = []
    for control in ('initial','legacy_force-5',*RULES):
        for metric in ('turn_s24','cost','unexited24','unexited26','vessel_idle_s'):
            diffs, pct = [], []
            for load in LOADS:
                for day in range(3):
                    x = day_values[('reachable-5',load,day)][metric]
                    y = day_values[(control,load,day)][metric]
                    diffs.append(x-y)
                    if y:
                        pct.append(100*(x-y)/y)
            comparisons.append(dict(control=control,metric=metric,
                paired_raw=asdict(paired(diffs)),
                paired_percent=asdict(paired(pct,delta_interest=1.)) if len(pct)==9 else None))
    tests = ET.parse(OUT/'regression.xml').find('testsuite').attrib
    assert int(tests['failures']) == int(tests['errors']) == 0
    final = [v for (a,s,l,d,r),v in values.items() if a=='reachable-5' and not r]
    guards = dict(completion=all(v['unexited26']==v['vessel_unfinished']==0 for v in final),
                  backlog_zero=all(v['unexited26']==0 for v in final), physical_valid=True,
                  vessel_protection=all(day_values[('reachable-5',l,d)][k] <=
                    day_values[('sf_spt',l,d)][k]+1e-6
                    for l in LOADS for d in range(3)
                    for k in ('vessel_unfinished','vessel_idle_s','vessel_delay_s')))
    summary = dict(runs=len(CASES),checked_order_rows=checked,independent_inputs=len(inputs),
        tests=int(tests['tests']),training=training,aggregate=aggregate,comparisons=comparisons,
        guards=guards,deterministic_repeat=True,admitted_all=True,
        scope='fixed synthetic assumptions; common-behaviour retraining, not online improvement')
    save(OUT/'summary.json', summary)
    table = {(r['arm'],r['load']):r for r in aggregate}
    lines = ['# 교사 선택 계약 수정·동일 구조 재학습 결과', '',
        f'초기값 3개×교사 2종×6회 갱신. 독립 입력 9개, 평가 {len(CASES)}회.',
        f'주문 장부 {checked:,}건 대조·저장 가중치 36개 재학습 재현·회귀 {tests["tests"]}건 통과.', '',
        '## 교사 수정', '',
        '| 교사 | 표본 결정 | 유효 쌍 | 선택 증거 없음 | 강제/재생 실패 |',
        '|---|---:|---:|---:|---:|']
    for arm in CONTRACTS:
        c = training['totals'][arm]
        lines.append(f'| {arm} | {c["sampled"]} | {c["labeled"]} | {c["no_witness"]} | {c["force_failed"]} |')
    lines += ['', 'legacy_force는 기존 우선순위 우회, reachable은 같은 입력의 점수만 바꾸고 실제 선택을 검증한 교사다.',
        '두 군의 행동 자료·표본 시점은 같았다. 거절된 결정을 다른 시점의 성공 표본으로 보충하지 않았다.', '',
        '## 체류시간 — 낮을수록 좋음', '',
        '| 하루 주문 | 학습 전(분) | 기존 교사 최종(분) | 수정 교사 3회(분) | 수정 교사 최종(분) | SF-SPT(분) |',
        '|---|---:|---:|---:|---:|---:|']
    for load in LOADS:
        cells = [table[(arm,load)]['turn_s24']/60 for arm in
                 ('initial','legacy_force-5','reachable-2','reachable-5','sf_spt')]
        lines.append(f'| {load} | '+' | '.join(f'{v:.3f}' for v in cells)+' |')
    lines += ['', '신경망은 초기값 3개×입력일 3개의 평균, 규칙은 같은 입력일 3개의 평균이다.',
        '실제 진입부터 출차까지 재며, 미완료 차량도 24시간 창 끝까지 포함했다. 모두 같은 주문을 접수했다.', '',
        '## 최종 모델의 짝비교', '',
        '| 비교 기준 | 체류 변화(%) | 비용 변화(%) | 24시간 미출차 차이(대) | 26시간 미출차 차이(대) |',
        '|---|---:|---:|---:|---:|']
    for arm in ('initial','legacy_force-5',*RULES):
        get = lambda metric: next(r for r in comparisons if r['control']==arm and r['metric']==metric)
        cells = [get(k)['paired_percent']['mean'] for k in ('turn_s24','cost')]
        cells += [get(k)['paired_raw']['mean'] for k in ('unexited24','unexited26')]
        lines.append(f'| {arm} | '+' | '.join(f'{v:.3f}' for v in cells)+' |')
    lines += ['', '양수는 수정 교사가 더 나쁜 값, 음수는 더 나은 값이다. 초기값 평균을 낸 입력 9개가 통계 단위다.',
        'SF-SPT는 실작업 우선+짧은 작업 우선, FCFS는 도착순, nearest는 가까운 작업 순이다.',
        '본선 우선 두 규칙, 한 방향 진행, 거리·대기 결합 규칙도 같은 환경에서 비교했다.', '',
        '## 해석 범위', '',
        '현재 표의 차이는 고정 합성 환경에서 교사 계약을 바꾼 효과다. 입력·신경망 구조·비용식은 같다.',
        '모든 자료는 같은 초기 행동 정책에서 수집했다. 학습 후 정책으로 다시 수집하는 온라인 반복 실험은 아니다.',
        '단일 공유 입력 점수 개입의 선택 증거이며, 가능한 모든 신경망 순서를 전수 탐색한 결과가 아니다.',
        '운반차 3대/470초, 예약 정확성, 적하 시간 추정은 실측 확정되지 않았다.',
        '6회 최종 가중치를 미리 정했고 시험 성적에 따라 좋은 중간 가중치를 선택하지 않았다.',
        '초기 시험 2건의 실패는 모든 작업 입력이 같은 시험 자료 때문이었다. 대기시간이 다른 작업으로 시험을 고쳤다.',
        '완주·잔여·본선 보호와 비용 신뢰구간의 정식 판정은 final-verdict.json에 별도로 보존한다.', '',
        '[사전등록](../../../docs/research/v7-discharge/reachable-teacher-prereg.md) · '
        '[원값·신뢰구간](summary.json) · [주문 장부](orders/) · [검증](verification.json)']
    assert len(lines) <= 200
    (OUT/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    save(OUT/'verification.json', dict(status='PASS', checked_order_rows=checked,
        exact_refit_checkpoints=36, repeat=True, source_hashes_verified=True,
        summary_sha256=sha(OUT/'summary.json'),report_sha256=sha(OUT/'README.md'),artifact_sha256=artifacts))
    print(f'Verified {len(CASES)} evaluations, {checked} order rows and 36 refits', flush=True)


if __name__ == '__main__':
    report()
