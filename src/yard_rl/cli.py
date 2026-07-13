"""CLI — Exp 예비 PoC 파이프라인.

python -m yard_rl.cli run-exp1   [--train N] [--epochs K] [--eval M] [--quick]
python -m yard_rl.cli run-matrix [--train N] [--epochs K] [--eval M] [--quick]

공통 순서: train 시나리오 생성 → FIFO 로 bucket·Scale fit(고정) → Q-learning 학습
→ (val greedy 확인) → test seeds paired 평가 → 리포트.
run-matrix 는 Exp-1/2/3A/3B/3C 조건별 QL 을 각각 학습해 동일 test 일에 비교한다.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .domain.enums import ControlScope, InformationLevel, PriorityRule
from .experiments.report import build_matrix_report, build_report
from .experiments.runner import (TEST_SEED0, TRAIN_SEED0, VAL_SEED0, PolicySpec,
                                 check_seed_bands, evaluate_paired,
                                 fit_buckets_and_scales, make_scenarios, run_episode)
from .envs.rewards import CostConfig
from .envs.yard_env import YardEnv
from .io.profile_loader import load_profile
from .io.scenario_gen import GenParams
from .policies.baselines import FixedRulePolicy, baseline_policies
from .policies.q_learning import QLearningAgent, QLearningConfig, train

DEFAULT_PROFILE = "configs/terminals/poc_single_crane.yaml"
DEFAULT_COST = "configs/costs/won_cost_v1.yaml"

# 실험 matrix — 정보수준 × 행동범위 (실험설계안 §3, 02 §3)
EXP_CONDITIONS = [
    ("QL_EXP1", InformationLevel.BLOCK_ARRIVAL, ControlScope.SEQUENCE_ONLY),
    ("QL_EXP2", InformationLevel.GATE_IN, ControlScope.SEQUENCE_ONLY),
    ("QL_EXP3A", InformationLevel.PRE_ADVICE, ControlScope.SEQUENCE_ONLY),
    ("QL_EXP3B", InformationLevel.PRE_ADVICE, ControlScope.PLUS_POSITIONING),
    ("QL_EXP3C", InformationLevel.PRE_ADVICE, ControlScope.PLUS_PRE_REHANDLE),
]

LADDER = [
    ("QL_EXP1", "QL_EXP2", "Exp-1→2 정보시점(게이트) 효과 (H1)"),
    ("QL_EXP2", "QL_EXP3A", "Exp-2→3A 사전정보 효과 (H2, 동일 행동공간)"),
    ("QL_EXP3A", "QL_EXP3B", "3A→3B +포지셔닝 효과"),
    ("QL_EXP3B", "QL_EXP3C", "3B→3C +선재조작 효과"),
]


def _prepare(n_train: int, n_eval: int, profile_path: str, out_dir: str):
    check_seed_bands(n_train, 4, n_eval)
    profile = load_profile(profile_path)
    params = GenParams()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[prep] train {n_train} / test {n_eval} 시나리오 생성 + FIFO fit")
    train_scs = make_scenarios(profile, TRAIN_SEED0, n_train, params)
    test_scs = make_scenarios(profile, TEST_SEED0, n_eval, params)
    buckets, reward = fit_buckets_and_scales(profile, train_scs,
                                             InformationLevel.BLOCK_ARRIVAL)
    buckets.save(out / "buckets.json")
    reward.save(out / "reward_scales.json")
    return profile, params, out, train_scs, test_scs, buckets, reward


def _train_agent(name, level, scope, profile, train_scs, buckets, reward,
                 epochs, out: Path) -> QLearningAgent:
    print(f"[train] {name} (level={level.value}, scope={scope.value})")
    env = YardEnv(profile, info_level=level, control_scope=scope,
                  bucket_cfg=buckets, reward_cfg=reward)
    agent = QLearningAgent(QLearningConfig(), seed=0, policy_name=name)
    train(agent, env, train_scs, epochs=epochs)
    agent.table.save(out / f"qtable_{name}.json")
    return agent


def _baseline_specs() -> list[PolicySpec]:
    # Baseline 은 현행 방식 정보수준(블록 도착 이후)·sequence_only 로 고정
    return [PolicySpec(p.name, p) for p in baseline_policies()]


def run_exp1(n_train, epochs, n_eval, profile_path, out_dir) -> Path:
    t0 = time.time()
    (profile, params, out, train_scs, test_scs,
     buckets, reward) = _prepare(n_train, n_eval, profile_path, out_dir)
    name, level, scope = EXP_CONDITIONS[0]
    agent = _train_agent(name, level, scope, profile, train_scs, buckets, reward,
                         epochs, out)
    (out / "train_log.json").write_text(json.dumps(agent.train_log, indent=1),
                                        encoding="utf-8")
    val_scs = make_scenarios(profile, VAL_SEED0, 4, params)
    for pname, pol in [("LONGEST_WAIT", FixedRulePolicy(PriorityRule.LONGEST_WAIT)),
                       (name, agent)]:
        envv = YardEnv(profile, info_level=level, bucket_cfg=buckets, reward_cfg=reward)
        ws = [run_episode(pol, envv, sc).metrics["mean_wait_min"] for sc in val_scs]
        print(f"    val mean_wait_min {pname}: " + ", ".join(f"{w:.1f}" for w in ws))
    specs = _baseline_specs() + [PolicySpec(name, agent, level, scope)]
    results = evaluate_paired(specs, profile, test_scs, buckets=buckets,
                              reward=reward, check_invariants=True)
    meta = _meta(profile, params, n_train, epochs, n_eval,
                 실험="Exp-1 (정보=블록 도착 이후, sequence_only, 단일 YC)")
    path = build_report(results, baseline="FIFO", meta=meta, out_dir=out)
    print(f"완료 ({time.time() - t0:.1f}s) → {path}")
    return path


def run_exp1_cost(n_train, epochs, n_eval, profile_path, out_dir, cost_path) -> Path:
    """YR-025: 목적함수를 원화비용 argmin 으로 재정의한 Exp-1 재실험.

    QL_EXP1(정규화 Core Cost, 대조군) vs QL_EXP1_COST(reward=-C_won) 를 같은
    train 시나리오로 학습해 동일 test 일에 paired 비교. C_won 은 5개 KPI 에
    선형이라 학습 파이프라인은 RewardConfig 치환만으로 재사용된다.
    """
    t0 = time.time()
    (profile, params, out, train_scs, test_scs,
     buckets, reward) = _prepare(n_train, n_eval, profile_path, out_dir)
    cost = CostConfig.load(cost_path)
    cost_reward = cost.to_reward_config()
    cost_reward.save(out / "cost_reward_scales.json")
    name, level, scope = EXP_CONDITIONS[0]
    agent_norm = _train_agent(name, level, scope, profile, train_scs, buckets,
                              reward, epochs, out)
    print(f"[train] QL_EXP1_COST (reward=-C_won, {cost.cost_id})")
    env = YardEnv(profile, info_level=level, control_scope=scope,
                  bucket_cfg=buckets, reward_cfg=cost_reward)
    agent_cost = QLearningAgent(QLearningConfig(), seed=0,
                                policy_name="QL_EXP1_COST")
    train(agent_cost, env, train_scs, epochs=epochs)
    agent_cost.table.save(out / "qtable_QL_EXP1_COST.json")
    (out / "train_log_cost.json").write_text(
        json.dumps(agent_cost.train_log, indent=1), encoding="utf-8")
    specs = (_baseline_specs()
             + [PolicySpec(name, agent_norm, level, scope),
                PolicySpec("QL_EXP1_COST", agent_cost, level, scope)])
    results = evaluate_paired(specs, profile, test_scs, buckets=buckets,
                              reward=reward, check_invariants=True)
    for rs in results.values():  # 총비용(만원) 지표 주입 — 전 정책 공통 산식
        for r in rs:
            r.metrics["total_cost_manwon"] = cost.cost_of_metrics(r.metrics)
    meta = _meta(profile, params, n_train, epochs, n_eval,
                 실험="Exp-1 비용 argmin (YR-025) — QL_EXP1_COST(reward=-C_won) vs "
                    "QL_EXP1(정규화) vs baseline",
                 목적함수=f"{cost.cost_id} (assumed) — 대기 {cost.truck_wait_krw_per_hour:,.0f}₩/h·"
                     f"tail 할증 {cost.tail_extra_krw_per_hour:,.0f}₩/h(안전운임 anchor, 30분 proxy)·"
                     f"이동 {cost.gantry_move_krw_per_km:,.0f}₩/km·"
                     f"재조작 {cost.rehandle_krw_per_move:,.0f}₩·"
                     f"본선 {cost.vessel_delay_krw_per_hour:,.0f}₩/h")
    path = build_report(results, baseline="FIFO", meta=meta, out_dir=out,
                        extra_metrics=("total_cost_manwon",),
                        ql_name="QL_EXP1_COST")
    print(f"완료 ({time.time() - t0:.1f}s) → {path}")
    return path


def run_matrix(n_train, epochs, n_eval, profile_path, out_dir) -> Path:
    t0 = time.time()
    (profile, params, out, train_scs, test_scs,
     buckets, reward) = _prepare(n_train, n_eval, profile_path, out_dir)
    specs = _baseline_specs()
    for name, level, scope in EXP_CONDITIONS:
        agent = _train_agent(name, level, scope, profile, train_scs, buckets, reward,
                             epochs, out)
        specs.append(PolicySpec(name, agent, level, scope))
    print(f"[eval] test paired 평가 ({len(specs)} 조건 × {n_eval} seeds, invariant ON)")
    results = evaluate_paired(specs, profile, test_scs, buckets=buckets,
                              reward=reward, check_invariants=True)
    meta = _meta(profile, params, n_train, epochs, n_eval,
                 실험="Exp-1/2/3A/3B/3C matrix — 정보시점 × 행동범위",
                 ETA="합성 제공 ETA = 실제도착 ± U(0,300s) (EMPIRICAL 대응, assumed)")
    path = build_matrix_report(results, baseline="FIFO", ladder=LADDER,
                               meta=meta, out_dir=out)
    print(f"완료 ({time.time() - t0:.1f}s) → {path}")
    return path


def _meta(profile, params, n_train, epochs, n_eval, **extra) -> dict:
    meta = dict(extra)
    meta.update({
        "프로파일": f"{profile.terminal_id} (assumed={profile.assumed})",
        "시나리오": f"합성 v1 — {params}",
        "seeds": f"train {TRAIN_SEED0}..{TRAIN_SEED0 + n_train - 1} ×{epochs}epoch / "
                 f"test {TEST_SEED0}..{TEST_SEED0 + n_eval - 1} (paired)",
        "reward": "정규화 Core Cost (탄소 미포함), w=(1,.3,.1,.1,.3) assumed, "
                  "Scale=train FIFO fit 고정",
        "주의": "CURRENT_RULE 미확보 — 휴리스틱 대비 비교만 유효",
    })
    return meta


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="yard_rl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd, help_ in [("run-exp1", "Exp-1 예비 PoC"), ("run-matrix", "Exp-1~3C 종합"),
                       ("run-exp1-cost", "Exp-1 원화비용 argmin (YR-025)")]:
        p = sub.add_parser(cmd, help=help_)
        p.add_argument("--train", type=int, default=30)
        p.add_argument("--epochs", type=int, default=4)
        p.add_argument("--eval", type=int, default=12)
        p.add_argument("--profile", default=DEFAULT_PROFILE)
        p.add_argument("--out", default=None)
        p.add_argument("--quick", action="store_true")
        if cmd == "run-exp1-cost":
            p.add_argument("--cost", default=DEFAULT_COST)
    args = ap.parse_args(argv)
    if args.quick:
        args.train, args.epochs, args.eval = 6, 1, 4
    out = args.out or {"run-exp1": "outputs/reports/exp1",
                       "run-matrix": "outputs/reports/exp_matrix",
                       "run-exp1-cost": "outputs/reports/exp1_cost"}[args.cmd]
    if args.cmd == "run-exp1":
        run_exp1(args.train, args.epochs, args.eval, args.profile, out)
    elif args.cmd == "run-exp1-cost":
        run_exp1_cost(args.train, args.epochs, args.eval, args.profile, out, args.cost)
    else:
        run_matrix(args.train, args.epochs, args.eval, args.profile, out)


if __name__ == "__main__":
    main()
