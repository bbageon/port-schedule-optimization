"""CLI — Exp-1 예비 PoC 파이프라인.

python -m yard_rl.cli run-exp1 [--train N] [--epochs K] [--eval M] [--quick]

순서: train 시나리오 생성 → FIFO 로 bucket·Scale fit(고정) → Q-learning 학습
→ validation greedy 확인 → test seeds paired 평가(Baseline 4종 + QL) → 리포트.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .domain.enums import InformationLevel, PriorityRule
from .experiments.report import build_report
from .experiments.runner import (TEST_SEED0, TRAIN_SEED0, VAL_SEED0, check_seed_bands,
                                 evaluate_paired, fit_buckets_and_scales, make_scenarios,
                                 run_episode)
from .envs.yard_env import YardEnv
from .io.profile_loader import load_profile
from .io.scenario_gen import GenParams
from .policies.baselines import FixedRulePolicy, baseline_policies
from .policies.q_learning import QLearningAgent, QLearningConfig, train

DEFAULT_PROFILE = "configs/terminals/poc_single_crane.yaml"


def run_exp1(n_train: int, epochs: int, n_eval: int, profile_path: str,
             out_dir: str) -> Path:
    t0 = time.time()
    check_seed_bands(n_train, 4, n_eval)  # train-on-test 침범 가드
    profile = load_profile(profile_path)
    level = InformationLevel.BLOCK_ARRIVAL  # Exp-1: 블록 도착 이후 정보만
    params = GenParams()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] train 시나리오 {n_train}개 생성 (seed {TRAIN_SEED0}+)")
    train_scs = make_scenarios(profile, TRAIN_SEED0, n_train, params)

    print("[2/5] FIFO train 실행 → bucket·reward Scale fit (이후 고정)")
    buckets, reward = fit_buckets_and_scales(profile, train_scs, level)
    buckets.save(out / "buckets.json")
    reward.save(out / "reward_scales.json")

    print(f"[3/5] Tabular Q-learning 학습: {n_train}ep × {epochs}epoch")
    env_train = YardEnv(profile, info_level=level, bucket_cfg=buckets, reward_cfg=reward)
    agent = QLearningAgent(QLearningConfig(), seed=0)
    train(agent, env_train, train_scs, epochs=epochs)
    agent.table.save(out / "qtable.json")
    (out / "train_log.json").write_text(json.dumps(agent.train_log, indent=1),
                                        encoding="utf-8")

    print("[4/5] validation greedy 확인 (seed %d+)" % VAL_SEED0)
    val_scs = make_scenarios(profile, VAL_SEED0, 4, params)
    for name, pol in [("LONGEST_WAIT", FixedRulePolicy(PriorityRule.LONGEST_WAIT)),
                      ("QL_EXP1", agent)]:
        envv = YardEnv(profile, info_level=level, bucket_cfg=buckets, reward_cfg=reward)
        ws = [run_episode(pol, envv, sc).metrics["mean_wait_min"] for sc in val_scs]
        print(f"    val mean_wait_min {name}: " + ", ".join(f"{w:.1f}" for w in ws))

    print(f"[5/5] test paired 평가 (seed {TEST_SEED0}+, {n_eval} seeds, invariant ON)")
    test_scs = make_scenarios(profile, TEST_SEED0, n_eval, params)
    policies = baseline_policies() + [agent]
    results = evaluate_paired(policies, profile, test_scs, level=level,
                              buckets=buckets, reward=reward, check_invariants=True)

    meta = {
        "실험": "Exp-1 (정보=블록 도착 이후, control=sequence_only, 단일 YC)",
        "프로파일": f"{profile.terminal_id} (assumed={profile.assumed})",
        "시나리오": f"합성 v1 — {params}",
        "seeds": f"train {TRAIN_SEED0}..{TRAIN_SEED0 + n_train - 1} ×{epochs}epoch / "
                 f"val {VAL_SEED0}.. / test {TEST_SEED0}..{TEST_SEED0 + n_eval - 1} (paired)",
        "Q-learning": f"{agent.cfg} / 방문상태 {len(agent.table.q)}",
        "reward": "정규화 Core Cost (탄소 미포함), w=(1,.3,.1,.1,.3) assumed",
        "주의": "CURRENT_RULE 미확보 — 휴리스틱 대비 비교만 유효",
    }
    path = build_report(results, baseline="FIFO", meta=meta, out_dir=out)
    print(f"완료 ({time.time() - t0:.1f}s) → {path}")
    return path


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="yard_rl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("run-exp1", help="Exp-1 예비 PoC 전체 파이프라인")
    p1.add_argument("--train", type=int, default=30, help="train 시나리오 수")
    p1.add_argument("--epochs", type=int, default=4)
    p1.add_argument("--eval", type=int, default=12, help="test paired seeds")
    p1.add_argument("--profile", default=DEFAULT_PROFILE)
    p1.add_argument("--out", default="outputs/reports/exp1")
    p1.add_argument("--quick", action="store_true", help="빠른 스모크 (train 6, epochs 1, eval 4)")
    args = ap.parse_args(argv)
    if args.cmd == "run-exp1":
        if args.quick:
            args.train, args.epochs, args.eval = 6, 1, 4
        run_exp1(args.train, args.epochs, args.eval, args.profile, args.out)


if __name__ == "__main__":
    main()
