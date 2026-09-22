"""YR-319 — 가중치는 얼린 채 **행동만 추첨으로** 고르는 30일 평가.

[[YR-306]] 비교는 한 번에 두 가지를 바꿨다 — **가중치를 얼렸고**(학습 중단)
**행동을 최고점으로만 골랐다**(argmax). 그래서 규칙 대비 172배가 어느 쪽 탓인지
귀속이 안 됐다. 이 스크립트는 **가중치만 얼리고 고르는 방식은 학습 때 그대로**
(확률 추첨) 두어 그 둘을 가른다.

바꾸는 것은 `sample_actions` 하나뿐이다. 시드·30일 명단·화물·체크포인트·측정창은
[[YR-306]] 원본과 **같은 값을 원본 manifest 에서 읽어** 쓴다.

■ 사전 중단규칙 (결과를 보기 전에 못박는다)
  어느 날이든 **그날 코호트 평균 턴타임**(게이트아웃−게이트인, 그날 끝에서 검열)이
  같은 날 규칙 팔의 **3배를 넘으면 즉시 중단**하고 `diverged` 로 기록한다.
  근거 — 한 달을 완주한 추첨 실행은 30일 내내 최대 **1.95배**였고, 무너진 두 실행은
  각각 **1일차·3일차**에 이 선을 넘었다(최대 19.2·20.7배). 건강한 실행은 안 죽이고
  무너지는 실행은 사흘 안에 끊는다. 131시간짜리 낭비가 사라진다.

■ 이 실행이 **하지 않는** 것
  성능 주장. 학습에 쓴 명단을 다시 쓰는 **진단**이고, 미완료 작업이 0이 아니며,
  시드 9900306 은 이미 전부 열람한 진단 대역이다.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from yard_rl.v5.ppo.checkpoint import load_policy
from yard_rl.v5.ppo.journal import RunJournal, write_json
from yard_rl.v5.ppo.provenance import code_stamp, file_sha256
from yard_rl.v5.ppo.runtime import PPOConfig, PPORuntime
from yard_rl.v5.reward.counterfactual import rollout_calls
from yard_rl.v5.stage.cargo_input import restore_input
from yard_rl.v5.stage.month import DAY_S, DayPlan
from yard_rl.v5.stage.month_run import run_month
from yard_rl.v5.stage.seed_bundle import load_seed_bundle

CHECKPOINTS = ('final', 'initial')

#: ★사전등록 — 규칙 팔의 **날짜별 코호트 평균 턴타임(초)**. 중단규칙의 기준선이다.
#: 출처 `outputs/v5/yr306-reference-comparison/rule/month_result.json` 의 `live[].mean_turn_time_s`.
RULE_SOURCE_SHA256 = '036365aa1a8b84659b80f98c3c006926bd73531a21fc7ca6440a9d18ef85db07'
RULE_DAILY_TURN_S = (
    1161.124049, 5902.904688, 1605.893929, 1300.983441, 5648.654119,
    2288.707306, 1566.300059, 5850.534664, 2190.616151, 1465.121862,
    2070.836599, 1141.514473, 1157.917424, 1591.768865, 1990.894833,
    1324.432489, 1416.639559, 1294.664582, 1380.828388, 1394.548371,
    1999.347424, 6180.13841, 2749.030753, 1658.610893, 1886.944609,
    2084.487413, 1791.548282, 1609.654168, 1615.408513, 1489.838887,
)
#: 규칙 팔 대비 몇 배까지 봐주나. 건강한 실행 실측 최대 1.95배 · 무너진 실행 19.2배.
ABORT_RATIO = 3.0


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def policy_digest(policy):
    digest = hashlib.sha256()
    for key, value in sorted(policy.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class DivergentAbort(RuntimeError):
    """사전 중단규칙이 걸렸다 — 실패가 아니라 **예정된 조기 종료**다."""

    def __init__(self, day, observed, reference):
        self.day, self.observed, self.reference = day, observed, reference
        super().__init__(f'day {day}: turn time {observed:.0f}s '
                         f'> {ABORT_RATIO}x rule {reference:.0f}s')


class SamplingJournal(RunJournal):
    """[[YR-306]] 평가 일지와 같은 모양 — 다만 정책이 추첨으로 고른다."""

    def __init__(self, output, days, manifest, expected_digest):
        self.expected_digest = expected_digest
        self.last_cost = 0.0
        super().__init__(output, days, manifest)

    def event(self, kind, row):
        if kind == 'start':
            row = {**row, 'learning_days': 0, 'evaluation_days': len(self.days)}
        super().event(kind, row)

    def boundary(self, runtime):
        t = runtime.time_s
        if self.last_time == t:
            return
        self.last_time = t
        require(not runtime.training and not runtime.updates and not runtime.buffer,
                'Frozen evaluation collected learning data')
        require(runtime.sample_actions, 'This evaluation must sample, not argmax')
        if t > 0 and t % DAY_S == 0 and t <= len(self.days) * DAY_S:
            require(policy_digest(runtime.policy) == self.expected_digest, 'Policy changed')
            for sim in runtime.mbt.blocks.values():
                sim.check_invariants()
            day = self.days[int(t // DAY_S) - 1]
            row = dict(day=day.index + 1, load=day.load, time_s=t, training=False,
                       measurement_day=0 < day.index < len(self.days) - 1,
                       interval_cost_krw=runtime.cost_krw - self.last_cost,
                       cost_krw=runtime.cost_krw, cost_breakdown=runtime.cost_breakdown,
                       cargo=runtime.mbt.cargo_report(), updates=0,
                       policy_digest=self.expected_digest, physical_invariants_checked=True,
                       roles=dict(runtime.role_counts), crane_actions=dict(runtime.crane_actions))
            self.daily.append(row)
            self.last_cost = runtime.cost_krw
            write_json(self.output / 'days.json', self.daily)
            self.event('day', row)
        if t % 3600 == 0:
            self.save_admissions()
            write_json(self.output / 'status.json',
                       dict(state='running', phase='frozen-sampling', time_s=t,
                            completed_days=len(self.daily), updates=0,
                            cost_krw=runtime.cost_krw,
                            admitted=self.admissions['admitted'],
                            skipped=self.admissions['skipped'],
                            wall_seconds=time.perf_counter() - self.started))


def dump_ledger(bridge, path):
    """거래 한 건마다 종류(공간/시간)·성공여부를 남긴다.

    [[YR-306]] 은 합계(`n_space`·`n_time`·`txn_failed`)만 저장해서 **실패 8,236건이
    공간인지 시간인지 가를 수 없었다.** 다리는 이미 줄마다 쌓고 있었고 파일로 쓰는
    줄만 없었다.
    """
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        for row in bridge.ledger:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    counts: dict = {}
    for row in bridge.ledger:
        key = (row['kind'], bool(row['ok']))
        counts[key] = counts.get(key, 0) + 1
    return {f"{kind}_{'ok' if ok else 'failed'}": n for (kind, ok), n in sorted(counts.items())}


def run(*, original_run, seed_bundle, checkpoint_name, output, rule_month_result):
    original_run, seed_bundle, output = map(Path, (original_run, seed_bundle, output))
    require(checkpoint_name in CHECKPOINTS, f'Checkpoint must be one of {CHECKPOINTS}')
    require(not output.exists(), 'Refusing to overwrite an evaluation')
    torch.set_num_threads(1)
    require(len(os.sched_getaffinity(0)) == 1, 'Evaluation must use one CPU')

    # 사전등록한 기준선이 실제 규칙 팔 산출물과 같은 파일인지 확인한다.
    require(file_sha256(rule_month_result) == RULE_SOURCE_SHA256,
            'Preregistered rule baseline does not match its source file')
    baseline = [row['mean_turn_time_s']
                for row in read(rule_month_result)['live']]
    require([round(v, 6) for v in baseline] == list(RULE_DAILY_TURN_S),
            'Preregistered rule baseline drifted from its source')

    stamp = code_stamp()
    original = read(original_run / 'manifest.json')
    original_report = read(original_run / 'report.json')
    require(original_report['state'] == 'completed', 'Original training must be complete')
    # ★원본과 **같은 파이썬/토치**여야 한다. 소스 해시는 이 작업이 runtime.py 를
    #   고쳤으므로 일부러 **같기를 요구하지 않고 양쪽을 다 기록**한다.
    for key in ('python', 'torch', 'numpy'):
        require(stamp[key] == original['code'][key], f'Original {key} changed')

    originals = {p.name: file_sha256(p) for p in original_run.iterdir() if p.is_file()}
    document, _ = load_seed_bundle(seed_bundle, expected_sha256=original['fixed_seed']['sha256'])
    days = [DayPlan(d['index'], d['load'], d['label'], d['seed'], d['t0'], d['n_days'])
            for d in original['days']]
    require(len(days) == len(RULE_DAILY_TURN_S), 'Baseline length must match the month')
    restore_input(document, seed=original['seed'], days=days, lead_mode='DIST')

    checkpoint = original_run / f'{checkpoint_name}.pt'
    reference_key = f'{checkpoint_name}_checkpoint'
    require(file_sha256(checkpoint) == original_report[reference_key]['sha256'],
            'Checkpoint changed')
    torch.manual_seed(original['seed'])
    policy = load_policy(checkpoint)
    torch.manual_seed(original['seed'])
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    digest = policy_digest(policy)

    manifest = dict(
        task='YR-319', purpose='DIAGNOSTIC_ONLY; isolate action selection from weight freezing',
        checkpoint_arm=checkpoint_name, code=stamp, source_original=original['code'],
        seed=original['seed'], days=original['days'], ppo=original['ppo'], training=False,
        action_mode='sample', compared_against='YR-306 argmax run with the same checkpoint',
        checkpoint=dict(path=str(checkpoint.resolve()),
                        sha256=originals[f'{checkpoint_name}.pt'], observer_only=False),
        policy_digest=digest,
        fixed_seed=dict(path=str(seed_bundle.resolve()), sha256=original['fixed_seed']['sha256']),
        measurement_window_s=[DAY_S, (len(days) - 1) * DAY_S],
        input_scope='training-input reuse, not held-out generalization',
        abort_rule=dict(metric='daily cohort mean turn time (gate-out minus gate-in)',
                        ratio=ABORT_RATIO, baseline='YR-306 rule arm, same day',
                        baseline_sha256=RULE_SOURCE_SHA256,
                        evidence='healthy sampling run peaked at 1.95x; broken argmax runs '
                                 'crossed 3x on day 1 and day 3'),
        claim_scope='NO_PERFORMANCE_CLAIM',
        original_files_sha256=originals)

    journal = SamplingJournal(output, days, manifest, digest)
    runtime = PPORuntime(policy, config=PPOConfig(**original['ppo']), seed=original['seed'],
                         training=False, sample_actions=True, on_boundary=journal.boundary)
    before_cf = rollout_calls()
    aborted = None

    def on_day(report):
        journal.day_report(report)
        index = report.index
        if index < len(baseline) and report.mean_turn_time_s > ABORT_RATIO * baseline[index]:
            raise DivergentAbort(index + 1, report.mean_turn_time_s, baseline[index])

    result = None
    try:
        result = run_month(seed=original['seed'], days=days, ppo=runtime, seed_data=document,
                           on_admission=journal.admission, on_day=on_day,
                           on_container_contract=journal.container_contract)
    except DivergentAbort as abort:
        aborted = dict(day=abort.day, observed_turn_time_s=abort.observed,
                       rule_turn_time_s=abort.reference, ratio=abort.observed / abort.reference)
    except Exception as error:      # 세계 구축 실패 등 — 증거를 남기고 다시 던진다.
        journal.fail(error, runtime)
        raise

    # 세계가 붙기 전에 끝났으면 장부도 보고서도 만들 수 없다.
    require(runtime.bound, 'Run ended before the world was bound')
    trades = dump_ledger(runtime.bridge, output / 'trade-ledger.jsonl.gz')
    if result is not None:
        write_json(output / 'month_result.json', asdict(result))

    checks = dict(
        frozen_policy=policy_digest(policy) == digest,
        no_updates=not runtime.updates and runtime.learning_intervals == 0,
        no_counterfactual=rollout_calls() == before_cf,
        sampled_actions=runtime.sample_actions,
        original_files_preserved=all(file_sha256(original_run / name) == value
                                     for name, value in originals.items()),
        seed_preserved=file_sha256(seed_bundle) == original['fixed_seed']['sha256'])
    if result is not None:
        checks.update(
            all_trucks=result.admitted == len(document['schedule']) and result.skipped == 0,
            all_days=len(journal.daily) == len(days),
            no_policy_exceptions=result.policy_exceptions == 0,
            fixed_end=runtime.time_s == len(days) * DAY_S + 7200)
    for sim in runtime.mbt.blocks.values():
        sim.check_invariants()
    checks['physical_invariants'] = True

    state = 'diverged' if aborted else ('completed' if all(checks.values()) else 'invalid')
    measurement = math.fsum(row['interval_cost_krw'] for row in journal.daily
                            if row['measurement_day'])
    report = {**runtime.report(), 'state': state, 'task': 'YR-319',
              'checkpoint_arm': checkpoint_name, 'action_mode': 'sample',
              'training': False, 'claim_scope': 'NO_PERFORMANCE_CLAIM',
              'checks': checks, 'aborted': aborted, 'trade_ledger': trades,
              'measurement_cost_krw': measurement, 'cargo': runtime.mbt.cargo_report(),
              'completed_days': len(journal.daily), 'code': stamp,
              f'{checkpoint_name}_checkpoint': dict(path=str(checkpoint),
                                                    sha256=originals[f'{checkpoint_name}.pt'])}
    write_json(output / 'report.json', report)
    write_json(output / 'status.json',
               dict(state=state, completed_days=len(journal.daily),
                    aborted=aborted, cost_krw=runtime.cost_krw,
                    wall_seconds=time.perf_counter() - journal.started))
    journal.event('finished', dict(state=state, aborted=aborted,
                                   completed_days=len(journal.daily)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-run', required=True)
    parser.add_argument('--seed-bundle', required=True)
    parser.add_argument('--checkpoint', choices=CHECKPOINTS, default='final')
    parser.add_argument('--rule-month-result', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = run(original_run=args.original_run, seed_bundle=args.seed_bundle,
                 checkpoint_name=args.checkpoint, output=args.output,
                 rule_month_result=args.rule_month_result)
    print(json.dumps({k: report[k] for k in ('state', 'completed_days', 'aborted',
                                             'cost_krw', 'trade_ledger')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
