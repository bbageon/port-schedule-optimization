"""배열 세계(v6) 학습 진입점 — v5 `python -m yard_rl.v6.ppo.continuous` 의 대응 ([[YR-327]] 조각 8).

■ 무엇을 하나
  30일 무대를 **배열 세계**로 세우고(`gpu/month.to_month_world`), 60초 검토 경계마다
  Φ(비용) 차분을 보상으로 받아 학습 정책망(37칸 tanh)을 PPO 로 갱신한다.
  v5 와 같은 일을 하는 배선이고, 새 물리·새 산술은 하나도 없다 (`gpu/train.py` 머리말).

■ 범위 (조각 8) — **이 진입점으로 연구 결과를 주장하지 않는다** (`claim_scope=NO_PERFORMANCE_CLAIM`)
  · 행동: 기본은 v5 와 같은 규칙 `sample_actions = training` 이다 (v5 `runtime.py:72-73`) — 즉 학습이면
    **추첨**(`action_mode: sample-all-days` · 배열판 자기 난수 `jax.random`), `--eval` 이면 최고점(argmax).
    `--no-sample` 로 학습 중에도 최고점을 강제할 수 있는데 그것은 **동등성 대조용 설정**이고
    [[YR-324]] 가 "정책을 망친다" 고 판정한 설정이다 — 학습 런에 쓰지 마라.
  · 시장(공간·시간 판매)은 열지 않는다 = v5 `arm='NO_REALLOC'` 과 같은 세계. 보고의 시장 계수기·
    `roles['seller']` 는 **null**(묻지 않았다)이고 `market: "unported"` 가 붙는다.
    ⚠️ 그래서 지금 돌고 있는 연구 팔(`RL_SPACE` — 월 공간거래 3만 건)은 **이 진입점으로 못 돌린다**.
  · `seed_data`(고정 화물 = `CargoTerminal`) 갈래는 범위 밖이다 (`MonthScopeError`) —
    ⚠️ 그런데 **연구선이 쓰는 갈래가 그 갈래다** (`ppo/workload_experiment.py` 는 언제나 seed_data 를
    넘긴다). 즉 배열 학습은 지금 연구선 설정으로 실행 불가다.
  · 증거 배선(v5 형식 체크포인트·날별 장부·완주 게이트 6종·코드 도장)은 아직 없다 — `.npz` 가중치와
    `report.json` 뿐이라 v5 평가·재개 도구가 읽지 못한다.

■ 쓰기
    PYTHONPATH=src python scripts/v6/train_v6.py --output outputs/v6/train-01 --days 3 --debug-load 30
    PYTHONPATH=src python scripts/v6/train_v6.py --output … --ckpt outputs/v5/yr302-final-train/policy.pt
    앞에서 끊기: `--e1`. **`--e0>0` 은 거절한다** — 프로세스를 넘겨 이어 돌릴 수 없기 때문이다
    (`gpu/train.train_month` 의 `box` 는 같은 프로세스 안에서만 산다. 전에는 경고 없이 새 무대를 t=0 에
    세운 뒤 앞 e0 에폭을 건너뛰어 '완료' 로 보고했다 — 아무 결정도 하지 않은 세계였다).
    진짜 이어 돌리기에는 `MB.save_run`/`load_run` + `MonthResult.state`·`MonthTape`·`AdamState`·난수
    상태를 함께 절이는 손잡이가 필요하다.

  ⚠️ **속도** — 경계마다 Φ·본선 유휴를 호스트에서 읽는다. 21블록·30일(43,201 경계)은 시간이 많이 걸린다.
     속도 판정(손익분기)은 기록·갱신을 끈 굴리기 경로로 따로 잰다: `scripts/v6/bench_breakeven.py`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy  # noqa: F401  ★numpy 를 torch 보다 먼저 (OMP #15 — dump_ground_truth.py 머리말)

try:
    import jax as _jax
    _jax.config.update("jax_enable_x64", True)     # ★배열을 만들기 전에 float64 를 켠다
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_BAND = 9_900_000


def make_plan(seed: int, n_days: int, load=None):
    """v5 `ppo/continuous.make_plan` 과 같은 규칙 (같은 검사·같은 날 시드)."""
    from yard_rl.v6.stage.month import DAY_S, DayPlan, plan_month
    if isinstance(seed, bool) or not isinstance(seed, int) or seed // 100_000 * 100_000 != DIAGNOSTIC_BAND:
        raise ValueError("진단 시드 [9900000, 9999999] 안의 정수를 쓰라")
    if not 3 <= int(n_days) <= 30:
        raise ValueError("이어진 런은 3~30일이다")
    if load is not None:
        return [DayPlan(i, int(load), "fixed-load-debug", seed + 1000 * (i + 1), i * DAY_S, n_days)
                for i in range(int(n_days))]
    return plan_month(seed, n_days=int(n_days))


def load_net(ckpt: str | None, *, net_seed: int, hidden: int, reward_scale_krw=None, reward_mode=None):
    """체크포인트가 있으면 그 가중치, 없으면 **고정 시드 무작위 초기화** (torch 로 만들어 같은 수를 싣는다)."""
    from yard_rl.v6.gpu import v5net as VN
    if ckpt:
        from yard_rl.v6.ppo.checkpoint import load_policy
        pol = load_policy(ckpt)
        from yard_rl.v6.ppo.runtime import PPOConfig, check_reward_contract
        mode = reward_mode or ('legacy-krw' if reward_scale_krw is not None else 'operational')
        check_reward_contract(pol, PPOConfig(reward_mode=mode, reward_scale_krw=reward_scale_krw))
        sd = {k: v.detach().cpu().numpy() for k, v in pol.state_dict().items()}
        return VN.load_v5_params(sd), f"ckpt:{Path(ckpt).stem}"
    import torch
    from yard_rl.v6.ppo.model import BlockPolicy
    torch.manual_seed(int(net_seed))
    pol = BlockPolicy(hidden=int(hidden))
    sd = {k: v.detach().cpu().numpy() for k, v in pol.state_dict().items()}
    return VN.load_v5_params(sd), f"seed{net_seed}h{hidden}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True, help="새 디렉터리 (덮어쓰지 않는다)")
    ap.add_argument("--seed", type=int, default=9_900_306)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--debug-load", type=int, default=None, help="하루 트럭 수를 못박는다 (배선 시험용)")
    ap.add_argument("--blocks", default=None, help="블록을 줄인다 (쉼표 구분 — 시험용)")
    ap.add_argument("--cap-moves", type=int, default=None, help="본선 한 척 물량 상한 (시험용)")
    ap.add_argument("--ckpt", default=None, help="v5 체크포인트 (없으면 고정 시드 무작위 초기화)")
    ap.add_argument("--reward-scale-krw", type=float, help="과거 결과 재현용 명시적 눈금; 기본은 고정 기준 자료")
    ap.add_argument('--reward-mode', choices=['operational', 'legacy-krw'], help='기본: 시간·횟수 직접 정규화')
    ap.add_argument("--net-seed", type=int, default=20_260_927)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--e0", type=int, default=0,
                    help="이어 돌리기 자리 — 같은 프로세스가 아니면 거절한다 (위 ⚠️)")
    ap.add_argument("--e1", type=int, default=None)
    ap.add_argument("--eval", action="store_true", help="갱신 없이 굴린다 (training=False)")
    ap.add_argument("--sample", dest="sample", action="store_true", default=None,
                    help="행동을 추첨으로 뽑는다 (기본: 학습이면 켬 — v5 와 같은 규칙)")
    ap.add_argument("--no-sample", dest="sample", action="store_false",
                    help="최고점(argmax) 수집 — 동등성 대조용. 학습 런에 쓰면 YR-324 의 실패 설정이다")
    ap.add_argument("--sample-seed", type=int, default=None, help="추첨 난수 씨 (기본: 무대 시드)")
    ap.add_argument("--stop-s", type=float, default=None, help="이 시각에 절단 (v5 DebugStop)")
    ap.add_argument("--cmax", type=int, default=256)
    ap.add_argument("--no-learning-window", action="store_true",
                    help="학습창을 걸지 않는다 (기본은 v5 처럼 첫날·마지막날을 연결용으로 뺀다)")
    args = ap.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        ap.error(f"이미 있는 디렉터리다 — 새 곳을 고르라: {out}")

    from yard_rl.v6.gpu import train as TR
    from yard_rl.v6.stage.month import DAY_S

    days = make_plan(args.seed, args.days, args.debug_load)
    reward_mode = args.reward_mode or ('legacy-krw' if args.reward_scale_krw is not None else 'operational')
    net, net_tag = load_net(args.ckpt, net_seed=args.net_seed, hidden=args.hidden,
                            reward_scale_krw=args.reward_scale_krw, reward_mode=reward_mode)
    window = None if args.no_learning_window else (DAY_S, (len(days) - 1) * DAY_S)
    training = not args.eval
    #: v5 `PPORuntime.__init__` 의 규칙 그대로 — 안 주면 `sample_actions = bool(training)`
    sample = training if args.sample is None else bool(args.sample)
    tcfg = TR.TrainConfig(training=training, stop_s=args.stop_s, cmax=int(args.cmax),
                          reward_scale_krw=args.reward_scale_krw, reward_mode=reward_mode,
                          learning_window_s=window, sample_actions=sample,
                          sample_seed=args.sample_seed)
    blocks = tuple(b.strip() for b in args.blocks.split(",")) if args.blocks else None
    updates: list = []
    day_rows: list = []
    t0 = time.perf_counter()
    ts, rep = TR.train_month(seed=args.seed, days=days, net_params=net, tcfg=tcfg,
                             blocks=blocks, cap_moves=args.cap_moves,
                             e0=args.e0, e1=args.e1,
                             on_update=lambda r: (updates.append(r),
                                                  print(json.dumps({"update": r}, default=str),
                                                        flush=True)),
                             on_boundary=lambda run, d, t: day_rows.append({"day": d, "t": t}))
    wall = time.perf_counter() - t0
    out.mkdir(parents=True, exist_ok=False)
    #: ★완주 게이트 — 비트·위반·소진이 0 이 아니면 **실패로 끝낸다** (전에는 언제나 'completed' 였다)
    bad = {"flags": list(rep.get("flags") or ()), "violation": rep.get("violation"),
           "exhausted": rep.get("exhausted"), "truncated": rep.get("truncated")}
    failed = bool(bad["flags"]) or bool(rep.get("violation")) or bool(rep.get("exhausted"))
    rep.update(generation="v6-array", net=net_tag, seed=args.seed, n_days=len(days),
               learning_window_s=window, wall_seconds=wall, eval=bool(args.eval),
               e0=args.e0, e1=args.e1, day_boundaries=day_rows,
               claim_scope="NO_PERFORMANCE_CLAIM", gate=bad, gate_passed=not failed,
               scope_notes=[
                   "시장 미이식 — 연구 팔(RL_SPACE 등) 실행 불가 · roles.seller/buyer 는 null(묻지 않았다)",
                   "고정 화물(CargoTerminal) 갈래 미이식 — 연구선 시드 묶음으로 실행 불가",
                   ("추첨 수집(action_mode=sample-all-days) — 표본은 v5 와 다르다(재현 불가 난수열)"
                    if sample else
                    "탐색 없음(argmax 수집) — v5 학습 설정(sample-all-days)과 다름 · YR-324 실패 설정"),
                   "증거 배선 미완 — v5 형식 체크포인트·날별 장부·완주 게이트 6종·코드 도장 없음"])
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str)
                                     + "\n", encoding="utf-8")
    np_out = {k: numpy.asarray(v) for k, v in
              zip(("w1", "b1", "w2", "b2", "w_actor", "b_actor", "w_critic", "b_critic"),
                  ts.params.net)}
    numpy.savez(out / "policy_final.npz", **np_out)
    print(json.dumps({"state": "failed" if failed else "completed",
                      "gate": bad, "action_mode": rep.get("action_mode"),
                      "wall_seconds": round(wall, 1),
                      "updates": len(updates), "intervals": rep.get("intervals"),
                      "cost_krw": rep.get("cost_krw"), "output": str(out)},
                     ensure_ascii=False), flush=True)
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
