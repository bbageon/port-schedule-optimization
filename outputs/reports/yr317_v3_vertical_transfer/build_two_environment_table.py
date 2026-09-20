"""Two-environment, same-seed table for YR-317-h (descriptive, no inference).

Reuses the completed legacy shared-layout results of seed 20,000,000 from the
independent evaluation (run-7e2fb14) and pairs them with the vertical-end
results (run-8c24c80/main). Reuse is allowed only when completion, recording
audit, canonical input fingerprints and the frozen checkpoint identity match.
Costs are the final request-day attributed values stored in result.json
``days`` (identical to daily-final.jsonl), not the provisional in-day
observations of days.jsonl.
"""
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SEED = 20_000_000
ARMS = ('NO_REALLOC', 'RL_TIME')
LEGACY = ROOT / 'outputs/reports/yr317_v3_independent_eval/run-7e2fb14'
VERTICAL = OUT / 'run-8c24c80/main'
LARGE_RAW = ('result.json', 'requests.jsonl.gz', 'operating-state.jsonl.gz',
             'container-links.jsonl.gz', 'outcome-audit.json')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arm_dir(env, arm):
    return (LEGACY / 'months' / str(SEED) / arm) if env == 'legacy_shared' else (VERTICAL / arm)


def check_reusable(env, arm, expected_input, expected_ckpt):
    d = arm_dir(env, arm)
    completion = load(d / 'completion.json')
    result = load(d / 'result.json')
    audit = load(d / 'independent-audit.json')
    problems = []
    if completion.get('state') != 'completed':
        problems.append('completion state is not completed')
    if not completion.get('audit', {}).get('passed') or not audit.get('passed'):
        problems.append('recording audit did not pass')
    if completion.get('input_hashes') != expected_input:
        problems.append('canonical input fingerprints differ from the vertical run')
    if completion.get('result_sha256') != sha(d / 'result.json'):
        problems.append('result.json fingerprint changed after completion')
    if result.get('claim_eligible') is not False or result.get('rollout_calls', 0) != 0:
        problems.append('frozen-policy guard failed')
    if result.get('seed') != SEED:
        problems.append('unexpected seed')
    if not all(v is True for v in result.get('recording_checks', {}).values()):
        problems.append('recording checks contain a failure')
    # legacy results carry the checkpoint identity in the run's launch.json; vertical results carry it inline
    ckpt = expected_ckpt if env == 'legacy_shared' else result.get('checkpoint_sha256')
    if ckpt != expected_ckpt:
        problems.append('checkpoint identity differs')
    days = result['days']
    if [x['index'] for x in days] != list(range(30)):
        problems.append('30 daily records are required')
    if any(not math.isfinite(x['phi_krw']) or x['phi_krw'] < 0 for x in days):
        problems.append('non-finite or negative daily cost')
    return completion, result, problems


def summarise(result):
    days = result['days']
    mid = [x for x in days if 0 < x['index'] < 29]
    def tot(rows, key):
        return math.fsum(x[key] for x in rows)
    trucks_mid = sum(x['n_trucks'] for x in mid)
    rs = result['request_summary']
    vw = result['vessel_work_summary']
    return dict(
        cost_28d_krw={k: tot(mid, k) for k in ('phi_krw', 'c_wait', 'c_move', 'c_rehandle', 'c_vessel')},
        cost_30d_krw=tot(days, 'phi_krw'),
        requests=dict(requested=rs['requested'], completed=rs['states'].get('COMPLETED', 0),
                      unfinished_at_cutoff=rs['states'].get('CENSORED', 0),
                      unbound_at_cutoff=rs.get('unbound_jobs_at_end')),
        vessel_yard_jobs=dict(requested=vw['requested_moves'], completed=vw['completed_yard_jobs'],
                              outstanding=vw['outstanding_yard_jobs']),
        actions=dict(time_changes=result.get('time', 0), block_changes=result.get('space', 0),
                     traded=result.get('traded', 0)),
        turn_time_h=dict(mean_28d=math.fsum(x['mean_turn_time_s'] * x['n_trucks'] for x in mid) / trucks_mid / 3600,
                         p90_median_of_days_28d=sorted(x['p90_turn_time_s'] for x in mid)[len(mid) // 2] / 3600),
        trucks_28d=trucks_mid,
        elapsed_h=result['elapsed_s'] / 3600,
    )


def main():
    vertical_cfg = load(OUT / 'config.json')
    legacy_launch = load(LEGACY / 'launch.json')
    expected_ckpt = legacy_launch['checkpoint_sha256']
    v_ref = load(VERTICAL / 'RL_TIME/completion.json')
    expected_input = v_ref['input_hashes']
    pair = load(VERTICAL / 'pair-summary.json')
    if not pair.get('passed') or pair.get('seed') != SEED:
        raise SystemExit('vertical pair summary is not a passed main-seed summary')

    envs, problems, raw_hashes = {}, {}, {}
    for env in ('legacy_shared', 'vertical_end'):
        envs[env] = {}
        for arm in ARMS:
            completion, result, probs = check_reusable(env, arm, expected_input, expected_ckpt)
            problems[f'{env}/{arm}'] = probs
            envs[env][arm] = summarise(result)
            envs[env][arm]['completed_at'] = completion['at']
            envs[env][arm]['result_sha256'] = completion['result_sha256']
            if env == 'vertical_end':
                raw_hashes[arm] = {name: sha(VERTICAL / arm / name) for name in LARGE_RAW}
    for env in envs:
        base, time = envs[env]['NO_REALLOC'], envs[env]['RL_TIME']
        for scope, key in (('28d', 'cost_28d_krw'), ('30d', 'cost_30d_krw')):
            ca = base[key]['phi_krw'] if scope == '28d' else base[key]
            cb = time[key]['phi_krw'] if scope == '28d' else time[key]
            envs[env][f'time_minus_baseline_{scope}'] = dict(
                saving_krw=ca - cb, saving_pct=100 * (ca - cb) / ca)
    # cross-check against the supervisor's own pair summary
    v = envs['vertical_end']
    pm = pair['costs']['measurement_days']
    consistent = (abs(v['NO_REALLOC']['cost_28d_krw']['phi_krw'] - pm['baseline_cost_krw']) < 1
                  and abs(v['RL_TIME']['cost_28d_krw']['phi_krw'] - pm['time_only_cost_krw']) < 1)
    all_ok = consistent and not any(problems.values())
    out = dict(
        passed=all_ok, seed=SEED, policies=list(ARMS),
        measurement='request-day attributed final cost (result.json days == daily-final.jsonl); '
                    'middle 28 days primary, 30 days supplementary; provisional days.jsonl not used',
        reuse_conditions=dict(expected_input=expected_input, checkpoint_sha256=expected_ckpt,
                              legacy_source_commit=load(LEGACY / 'months' / str(SEED) / 'RL_TIME/result.json')['repro']['code']['git_head'],
                              vertical_source_commit=vertical_cfg['source_commit'], problems=problems),
        pair_summary_consistent=consistent,
        environments=envs,
        vertical_large_raw_sha256=raw_hashes,
        claim_scope='Single predefined seed; descriptive only. Not an independent replication, '
                    'not a horizontal-vs-vertical causal comparison, not a field result.',
    )
    (OUT / 'two-environment-comparison.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')

    def eok(x):
        return f'{x / 1e8:,.1f}'
    lines = ['# 같은 시드(20,000,000) 두 환경 결과표', '',
             f'검사 통과: {all_ok}. 단위: 억원, 28일(2~29일차) 주분석. 시드 하나의 서술이며 통계 추론이 아니다.', '',
             '| 환경 | 정책 | 28일 비용 | 트럭대기 | 크레인이동 | 재조작 | 본선 | 30일 비용 | 시간 변경 | 미완료 트럭 | 미결 트럭 | 본선 야드 잔여 | 평균 턴타임(h, 미완료 체류 포함) |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for env, label in (('legacy_shared', '기존 공유형'), ('vertical_end', '수직 끝단형')):
        for arm in ARMS:
            s = envs[env][arm]
            c = s['cost_28d_krw']
            lines.append(f"| {label} | {arm} | {eok(c['phi_krw'])} | {eok(c['c_wait'])} | {eok(c['c_move'])} | "
                         f"{eok(c['c_rehandle'])} | {eok(c['c_vessel'])} | {eok(s['cost_30d_krw'])} | "
                         f"{s['actions']['time_changes']:,} | {s['requests']['unfinished_at_cutoff']:,} | "
                         f"{s['requests']['unbound_at_cutoff']:,} | {s['vessel_yard_jobs']['outstanding']:,} | "
                         f"{s['turn_time_h']['mean_28d']:.2f} |")
        d = envs[env]['time_minus_baseline_28d']
        lines.append(f"| {label} | 시간−기준 | {eok(-d['saving_krw'])} ({-d['saving_pct']:+.2f}%) | | | | | "
                     f"{eok(-envs[env]['time_minus_baseline_30d']['saving_krw'])} ({-envs[env]['time_minus_baseline_30d']['saving_pct']:+.2f}%) | | | | | |")
    lines += ['', '음수 = 시간정책이 더 쌈. 미완료 트럭 = 30일+2시간 종료 시점에 반출을 못 마친 요청.',
              '기존 공유형 결과는 독립 평가 run-7e2fb14의 같은 시드 완료본을 지문 검증 후 재사용했다.']
    (OUT / 'two-environment-comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(passed=all_ok, problems=problems, consistent=consistent), ensure_ascii=False))
    for env in envs:
        for scope in ('28d', '30d'):
            print(env, scope, envs[env][f'time_minus_baseline_{scope}'])


if __name__ == '__main__':
    main()
