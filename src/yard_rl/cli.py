"""CLI — Exp 예비 PoC 파이프라인.

python -m yard_rl.cli run-exp1   [--train N] [--epochs K] [--eval M] [--quick]
python -m yard_rl.cli run-exp1-direct-costq [--train 1000] [--val 30] [--eval 100]
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
from .envs.observations import BucketConfig
from .envs.rewards import CostConfig
from .envs.yard_env import YardEnv
from .experiments.recorder import record_episode
from .experiments.direct_job_runner import (
    DEFAULT_DIRECT_PROFILE, DirectExperimentConfig, quick_direct_config,
    run_direct_job_experiment,
)
from .io.profile_loader import load_profile
from .policies.q_learning import QTable
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


def record_replay(profile_path: str, exp_dir: str, policy_name: str, seed: int,
                  out_dir: str) -> Path:
    """YR-015-a: 학습 산출물(exp_dir 의 qtable/buckets)로 replay 를 기록.

    policy_name: baseline rule 이름(FIFO 등) 또는 exp_dir 의 qtable_<NAME>.json.
    시나리오는 실험과 동일한 GenParams·seed 로 재생성 (결정론 재현).
    """
    profile = load_profile(profile_path)
    exp = Path(exp_dir)
    buckets = BucketConfig.load(exp / "buckets.json")
    scenario = make_scenarios(profile, seed, 1, GenParams())[0]
    rule_names = {r.name for r in PriorityRule}
    if policy_name in rule_names:
        policy = FixedRulePolicy(PriorityRule[policy_name])
    else:
        qpath = exp / f"qtable_{policy_name}.json"
        agent = QLearningAgent(QLearningConfig(), seed=0, policy_name=policy_name)
        agent.table = QTable.load(qpath, agent.cfg.n_actions)
        policy = agent
    level, scope = InformationLevel.BLOCK_ARRIVAL, ControlScope.SEQUENCE_ONLY
    env = YardEnv(profile, info_level=level, control_scope=scope,
                  bucket_cfg=buckets, check_invariants=True)
    run_id = f"{profile.terminal_id}_{policy_name}_seed{seed}"
    path = record_episode(policy, env, scenario, run_id=run_id,
                          policy_name=policy_name, out_dir=out_dir)
    m = json.loads(path.read_text(encoding="utf-8"))["manifest"]["final_metrics"]
    print(f"[replay] {run_id}: {m['n_decisions']:.0f} decisions, "
          f"mean_wait {m['mean_wait_min']:.1f}min → {path}")
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
    pd = sub.add_parser(
        "run-exp1-direct-costq",
        help="Exp-1 외부트럭 Direct-Job Cost-Q (YR-027)",
    )
    pd.add_argument("--train", type=int, default=1_000)
    pd.add_argument("--val", type=int, default=30)
    pd.add_argument("--eval", type=int, default=100)
    pd.add_argument("--checkpoint", type=int, default=50)
    pd.add_argument("--n-external", type=int, default=100)
    pd.add_argument("--profile", default=DEFAULT_DIRECT_PROFILE)
    pd.add_argument("--out", default="outputs/reports/exp1_direct_costq_minimal_hjnc")
    pd.add_argument("--quick", action="store_true")
    pw = sub.add_parser("run-wtail", help="YR-018 w_tail × 학습예산 grid")
    pw.add_argument("--train", type=int, default=30)
    pw.add_argument("--epochs-list", default="4,10",
                    help="쉼표 구분 학습예산 축 (수렴진단 요건: 예산 동반)")
    pw.add_argument("--weights", default="0,0.1,0.3,1.0")
    pw.add_argument("--eval", type=int, default=12)
    pw.add_argument("--profile", default=DEFAULT_PROFILE)
    pw.add_argument("--out", default="outputs/reports/wtail_grid")
    pw.add_argument("--quick", action="store_true")
    pv = sub.add_parser("run-costq-v1final",
                        help="YR-030-b v1 최종안 상태 + greedy-prior + γ grid")
    pv.add_argument("--profile", default=DEFAULT_DIRECT_PROFILE)
    pv.add_argument("--out", default="outputs/reports/costq_v1final_hjnc")
    pv.add_argument("--quick", action="store_true")
    pcd = sub.add_parser("run-candidate-dqn",
                         help="YR-039 통합 Candidate DQN/DDQN/Dueling 3-arm")
    pcd.add_argument("--out", default="outputs/reports/candidate_dqn_poc")
    pcd.add_argument("--quick", action="store_true")
    pog = sub.add_parser("run-oracle-gap",
                         help="YR-031 Oracle 상한 측정 (전지적 beam vs greedy)")
    pog.add_argument("--profile", default=DEFAULT_DIRECT_PROFILE)
    pog.add_argument("--out", default="outputs/reports/oracle_gap_hjnc")
    pog.add_argument("--quick", action="store_true")
    pdn = sub.add_parser("run-delta-net",
                         help="YR-012 잔차 연속-feature Δ 학습 (함수근사, 사전등록)")
    pdn.add_argument("--profile", default=DEFAULT_DIRECT_PROFILE)
    pdn.add_argument("--out", default="outputs/reports/residual_delta_hjnc")
    pdn.add_argument("--quick", action="store_true")
    pop = sub.add_parser("run-oracle-pattern",
                         help="YR-031-b Oracle 개선 패턴 분석 (H-A/H-B, 사전등록)")
    pop.add_argument("--profile", default=DEFAULT_DIRECT_PROFILE)
    pop.add_argument("--out", default="outputs/reports/oracle_pattern_hjnc")
    pop.add_argument("--quick", action="store_true")
    pds = sub.add_parser("run-delta-stable",
                         help="YR-012-b Δ-net 안정화 (replay+target net, 사전등록)")
    pds.add_argument("--profile", default=DEFAULT_DIRECT_PROFILE)
    pds.add_argument("--out", default="outputs/reports/residual_delta_stable_hjnc")
    pds.add_argument("--quick", action="store_true")
    psf = sub.add_parser("run-delta-setfeat",
                         help="YR-012-c Δ-net feature 14→22 집합맥락 (사전등록)")
    psf.add_argument("--profile", default=DEFAULT_DIRECT_PROFILE)
    psf.add_argument("--out", default="outputs/reports/residual_setfeat_hjnc")
    psf.add_argument("--quick", action="store_true")
    pcs = sub.add_parser("run-setfeat-select",
                         help="YR-033 checkpoint 선택 프로토콜 보완 (사전등록)")
    pcs.add_argument("--profile", default=DEFAULT_DIRECT_PROFILE)
    pcs.add_argument("--out", default="outputs/reports/setfeat_selection_hjnc")
    pcs.add_argument("--quick", action="store_true")
    prc = sub.add_parser("run-costq-residual",
                         help="YR-030-c Greedy 기반 잔차 Cost-Q 3-arm (사전등록)")
    prc.add_argument("--profile", default=DEFAULT_DIRECT_PROFILE)
    prc.add_argument("--out", default="outputs/reports/costq_residual_hjnc")
    prc.add_argument("--quick", action="store_true")
    pa = sub.add_parser("run-costq-ablation",
                        help="YR-028 coverage ablation (checkpoint 규칙 vs 상태 vs 예산)")
    pa.add_argument("--profile", default=DEFAULT_DIRECT_PROFILE)
    pa.add_argument("--out", default="outputs/reports/costq_coverage_ablation_hjnc")
    pa.add_argument("--quick", action="store_true")
    pr = sub.add_parser("record-replay", help="replay 기록 (YR-015-a, UI 용)")
    pr.add_argument("--profile", default=DEFAULT_PROFILE)
    pr.add_argument("--exp-dir", required=True,
                    help="buckets.json·qtable_*.json 이 있는 실험 산출물 디렉토리")
    pr.add_argument("--policy", default="QL_EXP1",
                    help="baseline rule 이름 또는 qtable_<NAME>.json 의 NAME")
    pr.add_argument("--seed", type=int, default=TEST_SEED0)
    pr.add_argument("--out", default="outputs/replays")
    args = ap.parse_args(argv)
    if args.cmd == "run-candidate-dqn":
        from .experiments.candidate_dqn_experiment import (
            CandidateDqnConfig, quick_candidate_dqn_config, run_candidate_dqn)
        cfg = (quick_candidate_dqn_config() if args.quick
               else CandidateDqnConfig())
        run_candidate_dqn(args.out, cfg)
        return
    if args.cmd == "run-oracle-gap":
        from .experiments.oracle_gap import (OracleGapConfig,
                                             quick_oracle_config,
                                             run_oracle_gap)
        cfg = quick_oracle_config() if args.quick else OracleGapConfig()
        run_oracle_gap(args.profile, args.out, cfg)
        return
    if args.cmd == "run-delta-net":
        from .experiments.residual_delta_experiment import (DeltaExpConfig,
                                                            quick_delta_config,
                                                            run_delta_experiment)
        cfg = quick_delta_config() if args.quick else DeltaExpConfig()
        run_delta_experiment(args.profile, args.out, cfg)
        return
    if args.cmd == "run-oracle-pattern":
        from .experiments.oracle_pattern import (PatternConfig,
                                                 quick_pattern_config,
                                                 run_oracle_pattern)
        cfg = quick_pattern_config() if args.quick else PatternConfig()
        run_oracle_pattern(args.profile, args.out, cfg)
        return
    if args.cmd == "run-delta-stable":
        from .experiments.residual_delta_stable import (StableExpConfig,
                                                        quick_stable_config,
                                                        run_stable_experiment)
        cfg = quick_stable_config() if args.quick else StableExpConfig()
        run_stable_experiment(args.profile, args.out, cfg)
        return
    if args.cmd == "run-delta-setfeat":
        from .experiments.residual_setfeat_experiment import (SetFeatConfig,
                                                             quick_setfeat_config,
                                                             run_setfeat_experiment)
        cfg = quick_setfeat_config() if args.quick else SetFeatConfig()
        run_setfeat_experiment(args.profile, args.out, cfg)
        return
    if args.cmd == "run-setfeat-select":
        from .experiments.setfeat_selection import (SelectConfig,
                                                    quick_select_config,
                                                    run_selection_experiment)
        cfg = quick_select_config() if args.quick else SelectConfig()
        run_selection_experiment(args.profile, args.out, cfg)
        return
    if args.cmd == "run-costq-residual":
        from .experiments.residual_costq import (ResidualConfig,
                                                 quick_residual_config,
                                                 run_residual_experiment)
        cfg = quick_residual_config() if args.quick else ResidualConfig()
        run_residual_experiment(args.profile, args.out, cfg)
        return
    if args.cmd == "run-costq-v1final":
        from .experiments.state_v1_final import (V1FinalConfig,
                                                 quick_v1final_config,
                                                 run_v1_final_experiment)
        cfg = quick_v1final_config() if args.quick else V1FinalConfig()
        run_v1_final_experiment(args.profile, args.out, cfg)
        return
    if args.cmd == "run-costq-ablation":
        from .experiments.coverage_ablation import (AblationConfig,
                                                    quick_ablation_config,
                                                    run_coverage_ablation)
        cfg = quick_ablation_config() if args.quick else AblationConfig()
        run_coverage_ablation(args.profile, args.out, cfg)
        return
    if args.cmd == "run-wtail":
        from .experiments.wtail_grid import run_wtail_grid
        if args.quick:
            args.train, args.eval, args.epochs_list = 6, 4, "1"
        run_wtail_grid(profile_path=args.profile, out_dir=args.out,
                       n_train=args.train, n_eval=args.eval,
                       epochs_list=tuple(int(x) for x in args.epochs_list.split(",")),
                       weights=tuple(float(x) for x in args.weights.split(",")))
        return
    if args.cmd == "record-replay":
        record_replay(args.profile, args.exp_dir, args.policy, args.seed, args.out)
        return
    if args.cmd == "run-exp1-direct-costq":
        cfg = (quick_direct_config() if args.quick else DirectExperimentConfig(
            train_episodes=args.train,
            validation_episodes=args.val,
            test_episodes=args.eval,
            checkpoint_every=args.checkpoint,
            n_external=args.n_external,
        ))
        run_direct_job_experiment(args.profile, args.out, cfg)
        return
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
