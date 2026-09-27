"""★손익분기 실측 — 세계를 몇 개 쌓으면 배열판이 v5 를 코어 수만큼 돌리는 것보다 빠른가 ([[YR-327]] 조각 8).

사용자 질문 그대로: *"세계 몇 개부터 배열판이 v5 를 코어 수만큼 돌리는 것보다 빠른가?"*

■ ★★두 가지를 **따로** 잰다 — 섞으면 판정이 틀린다 (2026-09-27 수정 단계)
    `--which array/v5`      **굴리기 전용** 경로 (`multiblock.run_epochs` 만 vmap) — 세계를 쌓을 수 있다.
    `--which trainpath`     **학습 경로** (`gpu/train.train_month`) — 경계마다 호스트로 돌아오므로
                            (Φ·본선 유휴 사전 읽기·`bool(...)` 동기화 20여 회) **세계를 쌓을 수 없다**.
  연구가 학습에 쓰는 것은 **뒤쪽**이다. 그래서 "세계를 쌓는 것이 값어치 있나" 의 판정 수치는 앞쪽 표의
  3.9배가 아니라 **학습 경로의 배수**다 (아래 `trainpath` 절). 앞쪽 표는 '동등성을 포기한 학습 모드로
  갔을 때의 상한' 으로만 읽어라.
  ⚠️ 어느 쪽도 **시장(공간·시간 판매)이 없다** — 배열판에 시장 다리가 없어 지금 돌고 있는 연구 팔
    (`RL_SPACE` · 월 공간거래 3만 건)과는 원리상 비교할 수 없다. v5 `arm='NO_REALLOC'` 과 같은 세계다.

■ 재는 것 — **세계·하루당 시간** (초). 둘을 같은 단위로 만들어 나란히 놓는다.
    배열판   세계 B 개를 `vmap` 으로 한 번에 굴린다 → (벽시계 / B)
    v5       한 프로세스가 한 세계를 굴린다 → 코어 수만큼 **실제로 병렬 실행**해 (벽시계 / 동시 세계 수)
  교차점 B* = 배열판 세계·하루당 시간이 v5 다중프로세스의 세계·하루당 시간 아래로 내려가는 첫 B.

■ 무대 — 터미널 30 **하루** (21블록 · 1,441 검토 에폭 · 24시간 + 배수)
    `terminal_stream.build_diurnal(load_4h=30, day_total=30)` · 정책 `sf_spt`(규칙 · 정답 궤적이 검증한 조합).
  ⚠️ 배열 쪽은 `multiblock.run_epochs` (전진 + 동기화 + 투입 + 원장) 를 돈다. v5 쪽은
    `MultiBlockTerminal.run(pol, review_fn=ann.review)` 로 **같은 일**을 한다.
    이 무대는 조각 6 이 **해시 21/21 로 비트 일치**를 확인한 그 무대다 (터미널 30·300).
  ⚠️ 배열 쪽은 **검토 에폭 1,441개만** 잰다 — 배수 구간(`finish_run`)은 빼 놓았다(그 자리에서 한 번
    더 컴파일해야 한다). 배수 창은 `DIURNAL_DRAIN_S = 7,200`초이고 `sim_end_s = 93,600`초다
    (전에 '1,200초 / 87,600초 ≈ 1.4%' 로 적었는데 **길이가 6배 틀렸다**). 편향은 시뮬 시간이 아니라
    실측으로 적는다: 부하 30·B=1 에서 검토 에폭 82.742초 대 `finish_run` 1.368초 = **벽시계의 1.6%**,
    블록·스텝으로는 50,005 중 183건 = **0.37%** (부하 3500 에서도 0.59%). 배열에 유리한 쪽 편향이고
    교차점 판정을 뒤집을 크기가 아니다.
  ⚠️ 이 무대에는 재지정(`retarget`)·NO_TARGET 이 없다 (그건 30일 무대의 일이다) — 그래서 배열 쪽이
    호스트 거부권을 돌릴 필요가 없고, 양쪽이 정확히 같은 일을 한다.
  ⚠️ ★**무대 대표성** — 기본 무대는 하루 트럭 **30대**이고, 이것은 배열판에 **가장 불리한** 점이다
    (칸이 크게 비어 래기드 낭비가 크다). 연구선은 하루 6,600대(월 198,000)다. 부하를 키우면 v5 는
    2.25배 느려지는데 배열은 1.245배만 느려져 **핸디캡이 5.7배 → 3.2배로 줄지만 뒤집히지는 않는다**
    (실측 부하 30·300·3500 · `--load` 로 재현). 표 한 줄만 보고 일반 결론을 내지 말 것.

■ 반복·워밍업
    배열판은 **컴파일을 뺀다** — 첫 호출을 워밍업으로 버리고 그 뒤 `--reps` 회를 잰다(컴파일 시간도 따로 남긴다).
    v5 는 컴파일이 없으므로 그대로 잰다. 같은 시드 세계는 결정적이라 반복 사이 변동은 기계 소음뿐이다.

■ 세계를 어떻게 쌓나 — 두 가지를 **모두** 잰다
    same     같은 세계를 B 벌 (모든 세계가 같은 스텝에서 park → 래기드 낭비 0 = 상한)
    distinct 시드가 다른 세계 B 개 (블록마다 park 시점이 달라 `while_loop` 이 최대치를 따라간다 = 실제)
  조각 6 이 남긴 '래기드 while 1.85배' 가 B 를 키울 때 어떻게 변하는지는 이 둘의 비로 읽는다.

사용법 (WSL · GPU):
    PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/v6/bench_breakeven.py \
        --which array --worlds 1,2,4,8 --epochs 44
    JAX_PLATFORMS=cpu 를 더하면 같은 측정을 CPU 로.
    v5 기준선: `--which v5 --procs 1,24 --reps 2`
    ★학습 경로: `--which trainpath --boundaries 100,300` (+ `--procs 24` 로 v5 다중프로세스까지)
    합치기:   `--which merge`
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy  # noqa: F401  ★numpy 를 먼저 (OMP #15 회피 — dump_ground_truth.py 머리말)

#: ★배열을 만들기 **전에** float64 를 켠다 — 안 켜면 시각이 조용히 float32 로 내려앉는다
#:   (`gpu/events.py` 머리말 · `state.check_x64` 가 크게 실패한다).
try:
    import jax as _jax
    _jax.config.update("jax_enable_x64", True)
except Exception:                                  # jax 없는 파이썬 → v5 갈래만 쓴다
    pass

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "reports" / "yr327_v6_port"
LOAD = 30
SEED = 9_900_777


# ───────────────────────────────────────────────── 무대 (v5 · 배열 공통 입력)
def build_stage(load: int, seed: int):
    from yard_rl.v6.world.integrated import profiles as pr, terminal_stream as ts, yard_layout as yl
    prof, layout = pr.build_h21_profile(), yl.terminal_layout()
    built = ts.build_diurnal(prof, seed, obs=ts.OBS_24H, layout=layout,
                             params=ts.TerminalStreamParams(load_4h=load),
                             day_total=load, background_seed=seed)
    return prof, layout, built


# ───────────────────────────────────────────────── v5 쪽 (한 프로세스 = 한 세계)
def _v5_sim(eng, prof, scn, lvl):
    s = eng.TerminalSimulator(prof, scn, check_invariants=True)
    s.info_level = lvl
    return s


def v5_one_world(seed: int, load: int) -> dict:
    """v5 터미널 하루 한 벌 — `dump_ground_truth.run_terminal` 의 규칙 갈래와 같은 배선."""
    from yard_rl.v6.world.domain.enums import InformationLevel
    from yard_rl.v6.world.integrated import (baselines as bl, candidates as cd, engine as eng,
                                             multiblock as mb, policy_config as pc,
                                             terminal_stream as ts)
    lvl = InformationLevel.PRE_ADVICE
    t_b = time.perf_counter()
    prof, _layout, built = build_stage(load, seed)
    mbt = mb.MultiBlockTerminal(
        {b: ts.ensure_time_ledger(_v5_sim(eng, prof, s, lvl)) for b, s in built["scenarios"].items()},
        extra_review_epochs=ts.admission_epochs(ts.OBS_24H))
    ann = ts.ScheduledAnnouncer(built["schedule"], lead_s=1800.0, end_s=built["sim_end_s"])
    setup_s = time.perf_counter() - t_b
    gens: dict = {}
    pol_obj = bl.ResolverPolicy(bl.ServiceFirstSPTPreference(), "SF")

    def exec_policy(sim, dp):
        g = gens.setdefault(id(sim), cd.CandidateGenerator(config=pc.LEGACY_DEFAULT))
        gb = {c: g.generate(sim, c, lvl) for c in dp.crane_ids}
        bl._apply(sim, pol_obj.decide(sim, dp, gb))

    t0 = time.perf_counter()
    mbt.run(exec_policy, review_fn=ann.review)
    return {"seed": int(seed), "run_s": time.perf_counter() - t0, "setup_s": setup_s}


def _v5_worker(args):
    seed, load = args
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    return v5_one_world(seed, load)


def bench_v5(*, procs, load: int, seed: int, reps: int) -> dict:
    """① 한 세계 단독 시간 ② 프로세스 P 개를 **실제로 동시에** 돌린 처리량."""
    import concurrent.futures as cf
    single = []
    for _ in range(max(1, reps)):
        r = v5_one_world(seed, load)
        single.append(r["run_s"])
        print(f"  v5 단독 1세계 {r['run_s']:.2f}s (무대 세우기 {r['setup_s']:.2f}s 별도)", flush=True)
    rows = []
    for p in (procs if isinstance(procs, (list, tuple)) else [procs]):
        p = int(p)
        #: ★프로세스 수마다 **반복해서** 재고 최소·최대를 남긴다 (전에는 1회뿐이라 편차가 없었고,
        #:  이 측정은 기계 점유에 크게 흔들린다 — 같은 값이 다른 작업과 겹치면 1.5~1.8배가 된다).
        walls, inners, maxes = [], [], []
        for _ in range(max(1, reps)):
            t0 = time.perf_counter()
            with cf.ProcessPoolExecutor(max_workers=p) as ex:
                got = list(ex.map(_v5_worker, [(seed + 1000 * i, load) for i in range(p)]))
            walls.append(time.perf_counter() - t0)
            inners.append(sum(r["run_s"] for r in got) / p)
            maxes.append(max(r["run_s"] for r in got))
        wall, inner = min(walls), min(inners)
        rows.append({"procs": p, "wall_s": wall, "per_world_day_s": wall / p, "inner_mean_s": inner,
                     "inner_max_s": min(maxes), "walls_s": walls, "reps": len(walls),
                     "wall_max_s": max(walls)})
        print(f"  v5 ×{p:>3}  벽시계 {wall:7.2f}s (반복 {len(walls)}회 최소 · 최대 {max(walls):.2f}s)  "
              f"세계·하루당 {wall / p:7.3f}s  (프로세스 안 평균 {inner:.2f}s)", flush=True)
    return {"single_run_s": single, "single_mean_s": sum(single) / len(single), "parallel": rows,
            "cpu_count": os.cpu_count(), "load": load, "seed": seed}


# ───────────────────────────────────────────────── 배열 쪽 (세계 B 개를 vmap)
def array_setup(prof, built, *, policy: str = "sf_spt"):
    from yard_rl.v6.gpu import dispatch as DP, host_terminal as HT, multiblock as MB
    from yard_rl.v6.gpu.geom import Geom
    from yard_rl.v6.world.integrated import multiblock as mb, terminal_stream as ts
    tw, tt = HT.to_terminal_world(prof, built, lead_s=1800.0,
                                  extra_review_epochs=ts.admission_epochs(ts.OBS_24H))
    g = Geom.from_profile(prof)
    run0 = MB.make_run(tw, tt, params=MB.terminal_resolver_params(tt, g))
    k0 = tt.tables[0].crane_index[prof.cranes[0].crane_id]
    steps = 8 * int(tt.n_max) + 256
    eng = MB.Engine(g=g, policy_fn=DP.make_resolver(policy, g, count_lost=False), check=True,
                    pre_advice=True, horizon_s=float(prof.decision_horizon_s), joint=True,
                    margin=mb.CAPACITY_MARGIN, end_ann=float(built["sim_end_s"]),
                    steps_per_epoch=steps, steps_final=steps, k0=int(k0))
    return run0, tt, eng


def _stack(runs):
    import jax
    import jax.numpy as jnp
    if len(runs) == 1:
        return jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], runs[0])
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *runs)


def _vrun(MB):
    import jax

    def f(runs, e0, n, eng):
        return jax.vmap(lambda r: MB.run_epochs(r, e0, n, eng))(runs)

    return jax.jit(f, static_argnames=("n", "eng"))


def _nbytes(tree) -> int:
    import jax
    import numpy as np
    return int(sum(np.asarray(x).nbytes for x in jax.tree_util.tree_leaves(tree)))


def bench_array(*, worlds, epochs: int, reps: int, load: int, seed: int, mode: str,
                policy: str = "sf_spt") -> dict:
    import jax
    import numpy as np
    from yard_rl.v6.gpu import multiblock as MB
    dev = jax.devices()[0]
    prof, _layout, built = build_stage(load, seed)
    run_base, tt, eng = array_setup(prof, built, policy=policy)
    E = int(run_base.n_epochs)
    #: `--epochs 0` = 하루 전체. 슬라이스로 재면 **앞쪽 에폭이 가장 비어 있어** 낮게 나온다
    #:   (트럭이 아직 안 들어왔고 크레인이 놀아 결정이 적다) — 판정 수치는 전체로 잰다.
    epochs = E if int(epochs) <= 0 else min(int(epochs), E)
    vrun = _vrun(MB)
    one_bytes = _nbytes(run_base)
    rows = []
    cache: dict[int, object] = {0: run_base}
    for B in worlds:
        B = int(B)
        try:
            if mode == "same":
                runs = [run_base] * B
            else:
                runs = []
                for i in range(B):
                    if i not in cache:
                        p2, _l2, b2 = build_stage(load, seed + 1000 * (i + 1))
                        cache[i] = array_setup(p2, b2, policy=policy)[0]
                    runs.append(cache[i])
            t_b = time.perf_counter()
            stacked = _stack(runs)
            jax.block_until_ready(stacked)
            t_stack = time.perf_counter() - t_b
            t_c = time.perf_counter()
            out = vrun(stacked, 0, epochs, eng)
            jax.block_until_ready(out)
            t_compile = time.perf_counter() - t_c        # 컴파일 + 첫 실행 (워밍업 — 버린다)
            got = []
            for _ in range(max(1, reps)):
                t0 = time.perf_counter()
                out = vrun(stacked, 0, epochs, eng)
                jax.block_until_ready(out)
                got.append(time.perf_counter() - t0)
            best = min(got)
            #: ★"빨라서" 가 아니라 "일을 안 해서" 빠른 것이 아님을 매번 확인한다 —
            #:  위반 비트 0 · 상한 소진 0 · 실제로 돈 블록·스텝 수를 같이 남긴다.
            r_out = out[0]
            viol = int(np.asarray(r_out.tw.blocks.violation).max())
            exh = int(np.asarray(r_out.exhausted).sum())
            steps = int(np.asarray(r_out.tw.blocks.steps).sum())
            assert viol == 0 and exh == 0, f"B={B} 세계가 깨졌다 violation={viol} exhausted={exh}"
            row = {"worlds": B, "mode": mode, "epochs_timed": epochs, "epochs_total": E,
                   "violation": viol, "exhausted": exh, "block_steps": steps,
                   "wall_s": best, "reps_s": got, "stack_s": t_stack,
                   "compile_plus_first_s": t_compile,
                   "state_mb": round(one_bytes * B / 1e6, 1),
                   "ms_per_epoch_all_worlds": 1000.0 * best / epochs,
                   "per_world_day_s": best / B * (E / epochs)}
            rows.append(row)
            print(f"  배열 ×{B:>3} [{mode}] {epochs}에폭 {best:7.3f}s  "
                  f"({row['ms_per_epoch_all_worlds']:6.1f} ms/에폭)  "
                  f"세계·하루당 {row['per_world_day_s']:7.3f}s  "
                  f"(컴파일+첫회 {t_compile:.1f}s · 상태 {row['state_mb']}MB · "
                  f"블록·스텝 {steps:,} · 위반 {viol})", flush=True)
            del out, stacked
        except Exception as ex:                          # 메모리 벽 등 — 어디서 막혔는지 남긴다
            rows.append({"worlds": B, "mode": mode, "error": f"{type(ex).__name__}: {ex}"[:600],
                         "state_mb": round(one_bytes * B / 1e6, 1)})
            print(f"  배열 ×{B:>3} [{mode}] ★막혔다: {type(ex).__name__}: {str(ex)[:300]}", flush=True)
            break
    return {"device": dev.platform, "device_kind": getattr(dev, "device_kind", ""),
            "epochs_total": E, "n_max": int(tt.n_max), "steps_per_epoch": int(eng.steps_per_epoch),
            "blocks": int(run_base.b), "policy": policy, "one_world_state_mb": round(one_bytes / 1e6, 2),
            "rows": rows, "jax": jax.__version__}


# ═══════════════════════════════════════════════ ★학습 경로 (연구가 실제로 쓰는 경로)
def _net_state_dict(ckpt: str | None):
    """양쪽에 **같은 가중치**를 준다 — 체크포인트가 있으면 그것, 없으면 고정 시드 초기화."""
    import numpy  # noqa: F401
    import torch
    from yard_rl.v6.ppo.model import BlockPolicy
    if ckpt and Path(ckpt).exists():
        from yard_rl.v6.ppo.checkpoint import load_policy
        pol = load_policy(ckpt)
        return {k: v.detach().cpu().numpy() for k, v in pol.state_dict().items()}, Path(ckpt).stem
    torch.manual_seed(20_260_927)
    pol = BlockPolicy()
    return {k: v.detach().cpu().numpy() for k, v in pol.state_dict().items()}, "seed20260927"


def _plan_days(seed: int, load: int, n_days: int = 2):
    from yard_rl.v6.stage.month import DAY_S, DayPlan
    return [DayPlan(index=i, load=int(load), label="trainpath", seed=seed + 1000 * (i + 1),
                    t0=i * DAY_S, n_days=n_days) for i in range(n_days)]


def _bypass_gate():
    """v5 PPO 경로의 컨테이너 계약 게이트를 이 측정에서만 지나간다 (v5 파일은 고치지 않는다).

    ★기본(`MonthTerminal`) 갈래에서만 막힌다 — 고정 화물 갈래에서는 통과한다. 배열판은 기본 갈래만
      옮겼으므로 **같은 갈래를 나란히 재려면** 여기서 지나가야 한다 (속도 측정이라 계약과 무관하다).
    """
    from yard_rl.v6.stage import month_run as MR
    old = MR.require_container_plan
    MR.require_container_plan = lambda rep: None
    return MR, old


def v5_trainpath(*, seed: int, load: int, boundaries: int, sd, training: bool = True) -> dict:
    """v5 `run_month(ppo=PPORuntime(...))` 를 경계 N 개까지 굴린 시간 (학습 경로)."""
    import numpy  # noqa: F401
    import torch
    from yard_rl.v6.ppo.model import BlockPolicy
    from yard_rl.v6.ppo.runtime import DebugStop, PPOConfig, PPORuntime
    MR, old = _bypass_gate()
    torch.set_num_threads(1)
    try:
        pol = BlockPolicy()
        pol.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()}, strict=True)
        rt = PPORuntime(pol, config=PPOConfig(), seed=seed, training=training,
                        sample_actions=False, stop_s=float(boundaries) * 60.0)
        t0 = time.perf_counter()
        try:
            MR.run_month(seed=seed, days=_plan_days(seed, load), ppo=rt)
            stop = None
        except DebugStop:
            stop = "DebugStop"
        wall = time.perf_counter() - t0
    finally:
        MR.require_container_plan = old
    r = rt.report()
    return {"boundaries": boundaries, "wall_s": wall, "stop": stop, "intervals": r["intervals"],
            "updates": len(r["updates"]), "crane": r["roles"].get("crane", 0),
            "cost_krw": r["cost_krw"]}


def _v5_trainpath_worker(args):
    seed, load, boundaries, sd = args
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    return v5_trainpath(seed=seed, load=load, boundaries=boundaries, sd=sd)


def array_trainpath(*, seed: int, load: int, boundaries: int, sd, training: bool = True) -> dict:
    """배열 `train_month` 를 경계 N 개까지 (무대 세우기+컴파일과 정상상태를 **따로** 남긴다)."""
    import jax
    from yard_rl.v6.gpu import train as TR
    from yard_rl.v6.gpu import v5net as VN
    net = VN.load_v5_params(sd)
    tcfg = TR.TrainConfig(training=training, cmax=256, amax=13)
    t_b = time.perf_counter()
    box: dict = {}
    #: 무대 세우기만 따로 — `e1=0` 이면 루프를 한 번도 돌지 않는다
    TR.train_month(seed=seed, days=_plan_days(seed, load), net_params=net, tcfg=tcfg, box=box, e1=0)
    setup_s = time.perf_counter() - t_b
    n = [0]

    def on_epoch(run, e, t, codes, out):
        n[0] += 1

    t0 = time.perf_counter()
    ts, rep = TR.train_month(seed=seed, days=_plan_days(seed, load), net_params=net, tcfg=tcfg,
                             box=box, e1=int(boundaries), on_epoch=on_epoch)
    jax.block_until_ready(ts.params.net)
    wall = time.perf_counter() - t0
    return {"boundaries": n[0], "wall_s": wall, "setup_s": setup_s, "intervals": rep["intervals"],
            "updates": len(ts.updates), "crane": rep["roles"].get("crane", 0),
            "cost_krw": rep["cost_krw"], "flags": list(rep["flags"]),
            "device": jax.devices()[0].platform}


def bench_trainpath(*, boundaries, load: int, seed: int, ckpt=None, procs=None,
                    epochs_per_day: int = 1441) -> dict:
    """★학습 경로의 세계·하루당 시간 — 두 길이의 **차**로 정상상태를 뽑는다.

    한 길이만 재면 무대 세우기+컴파일이 섞여 (전 보고의 '0.89초/경계' 처럼) 2배 이상 틀린다.
    `slope = (t(n2) − t(n1)) / (n2 − n1)` 이 경계당 정상상태 시간이고, 세계·하루당 = slope × 1,441.
    """
    import jax
    sd, net_tag = _net_state_dict(ckpt)
    bs = sorted({int(b) for b in boundaries})
    rows = {"array": [], "v5": []}
    for b in bs:
        r = array_trainpath(seed=seed, load=load, boundaries=b, sd=sd)
        rows["array"].append(r)
        print(f"  배열 학습경로 경계 {b:>4}: {r['wall_s']:8.2f}s (무대+컴파일 {r['setup_s']:.1f}s 별도 · "
              f"갱신 {r['updates']} · 크레인결정 {r['crane']})", flush=True)
    for b in bs:
        r = v5_trainpath(seed=seed, load=load, boundaries=b, sd=sd)
        rows["v5"].append(r)
        print(f"  v5  학습경로 경계 {b:>4}: {r['wall_s']:8.2f}s (갱신 {r['updates']} · "
              f"크레인결정 {r['crane']} · 정지 {r['stop']})", flush=True)

    def slope(rs):
        if len(rs) < 2:
            return None
        a, b2 = rs[0], rs[-1]
        d = b2["boundaries"] - a["boundaries"]
        return None if d <= 0 else (b2["wall_s"] - a["wall_s"]) / d

    out = {"stage": f"터미널{load} · 21블록 · seed {seed} · 학습 정책망 {net_tag} · 학습 켬",
           "device": jax.devices()[0].platform, "boundaries": bs, "rows": rows,
           "epochs_per_day": epochs_per_day,
           "array_s_per_boundary": slope(rows["array"]),
           "v5_s_per_boundary": slope(rows["v5"])}
    for k in ("array", "v5"):
        sl = out[f"{k}_s_per_boundary"]
        out[f"{k}_per_world_day_s"] = None if sl is None else sl * epochs_per_day
    if out["array_s_per_boundary"] and out["v5_s_per_boundary"]:
        out["array_over_v5"] = out["array_s_per_boundary"] / out["v5_s_per_boundary"]
    if procs:
        import concurrent.futures as cf
        p = int(procs)
        #: ★짧은 쪽 길이로 잰다 — 부하 30 에서 시드에 따라 t=16,440초쯤 `NO_TARGET`(ContainerContractError)
        #:  으로 v5 가 스스로 멈추는 세계가 있다 (경계 300 = t=18,000초는 그 뒤다).
        b = bs[0]
        #: ★**같은 시드**를 P 벌 돌린다 — 배열 쪽 학습 경로도 세계 하나(같은 시드)를 재기 때문이다
        #:  (굴리기 표의 '다른 시드 24개' 비대칭을 여기서는 만들지 않는다).
        t0 = time.perf_counter()
        with cf.ProcessPoolExecutor(max_workers=p) as ex:
            got = list(ex.map(_v5_trainpath_worker, [(seed, load, b, sd) for _ in range(p)]))
        wall = time.perf_counter() - t0
        out["v5_parallel"] = {"procs": p, "boundaries": b, "wall_s": wall,
                             "s_per_boundary_per_world": wall / p / b,
                             "per_world_day_s": wall / p / b * epochs_per_day,
                             "inner_mean_s": sum(r["wall_s"] for r in got) / p}
        print(f"  v5 ×{p} 학습경로 경계 {b}: 벽시계 {wall:.1f}s → 세계·하루당 "
              f"{out['v5_parallel']['per_world_day_s']:.1f}s", flush=True)
        if out["array_per_world_day_s"]:
            out["array_over_v5_parallel"] = (out["array_per_world_day_s"]
                                             / out["v5_parallel"]["per_world_day_s"])
    return out


    # ───────────────────────────────────────────────── 래기드 낭비 (조각 6 이 남긴 수치가 B 로 희석되나)
def bench_waste(*, worlds, epochs: int, load: int, seed: int, mode: str, policy: str = "sf_spt") -> dict:
    """★계산한 블록·스텝 대 **실제로 쓴** 블록·스텝 — 조각 6 의 '래기드 while 1.85배(유효율 54.2%)'.

    `multiblock.run_to_epoch` 의 `while_loop` 은 **아직 park 하지 않은 블록이 하나라도 있으면** 돈다.
    park 한 블록에는 `step` 을 적용하지 않지만(마스크) **계산은 한다** — 그게 낭비다.
    `vmap` 아래서는 술어가 배치 any 로 바뀌므로 세계를 쌓으면 상한이 **세계들의 최대치**를 따라간다.

    재는 것: `i`(while 반복 수) × 블록 수 × 세계 수 = **계산한** 블록·스텝,
             `world.steps` 증가분의 합 = **쓴** 블록·스텝. 유효율 = 쓴/계산.
    호스트 루프라 느리므로 작은 B 에서만 잰다 (답은 B 와 무관하게 정확하다).
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from yard_rl.v6.gpu import multiblock as MB
    from yard_rl.v6.gpu.state import V_STEPS_EXHAUSTED

    def adv_iter(run, eng):
        W, parked, i, stuck = MB.run_to_epoch(run.tw.blocks, run.params, eng,
                                              max_steps=eng.steps_per_epoch)
        W = W._replace(violation=W.violation | jnp.where(stuck, V_STEPS_EXHAUSTED, 0).astype(jnp.int32))
        return run._replace(tw=run.tw._replace(blocks=W),
                            exhausted=run.exhausted + stuck.astype(jnp.int32)), i

    vadv = jax.jit(lambda runs, eng: jax.vmap(lambda r: adv_iter(r, eng))(runs),
                   static_argnames=("eng",))
    vrev = jax.jit(lambda runs, e, eng: jax.vmap(lambda r: MB.review_epoch(r, e, eng))(runs),
                   static_argnames=("eng",))
    prof, _l, built = build_stage(load, seed)
    run_base, tt, eng = array_setup(prof, built, policy=policy)
    E = int(run_base.n_epochs)
    epochs = E if int(epochs) <= 0 else min(int(epochs), E)
    rows = []
    cache: dict[int, object] = {0: run_base}
    for B in worlds:
        B = int(B)
        if mode == "same":
            runs = [run_base] * B
        else:
            runs = []
            for i in range(B):
                if i not in cache:
                    p2, _l2, b2 = build_stage(load, seed + 1000 * (i + 1))
                    cache[i] = array_setup(p2, b2, policy=policy)[0]
                runs.append(cache[i])
        st = _stack(runs)
        nb = int(run_base.b)
        steps0 = int(np.asarray(st.tw.blocks.steps).sum())
        trips = 0
        t0 = time.perf_counter()
        for e in range(epochs):
            st, i = vadv(st, eng)
            trips += int(np.asarray(i).max())
            st, _codes = vrev(st, jnp.asarray(e, jnp.int32), eng)
        jax.block_until_ready(st)
        used = int(np.asarray(st.tw.blocks.steps).sum()) - steps0
        computed = trips * nb * B
        rows.append({"worlds": B, "mode": mode, "epochs": epochs, "while_trips": trips,
                     "computed_block_steps": computed, "used_block_steps": used,
                     "efficiency": used / max(1, computed),
                     "ragged_factor": computed / max(1, used),
                     "host_loop_s": round(time.perf_counter() - t0, 1)})
        print(f"  낭비 ×{B:>3} [{mode}] while 반복 {trips:>7}  계산 {computed:>10}  쓴 {used:>10}  "
              f"유효율 {100 * rows[-1]['efficiency']:5.1f}%  = {rows[-1]['ragged_factor']:.2f}배 낭비",
              flush=True)
        del st
    # 오더 칸 패딩 — 정적이라 B 와 무관하다 (세계를 쌓아도 희석되지 않는다)
    o = np.asarray(run_base.tw.blocks.orders.block)
    used_rows = int((o >= 0).sum())
    rows.append({"order_rows_used": used_rows, "order_rows_total": int(o.size),
                 "order_padding": 1.0 - used_rows / max(1, o.size),
                 "note": "오더 칸 패딩은 정적이다 — 세계를 쌓아도 그대로다"})
    return {"device": jax.devices()[0].platform, "blocks": int(run_base.b),
            "epochs_total": E, "rows": rows}


# ───────────────────────────────────────────────── 합치기 · 표
def breakeven(v5: dict, arr: dict) -> dict:
    ref = None
    for r in v5.get("parallel", []):
        if ref is None or r["procs"] > ref["procs"]:
            ref = r
    if ref is None:
        return {}
    target = ref["per_world_day_s"]
    star = None
    for row in arr.get("rows", []):
        if "per_world_day_s" in row and row["per_world_day_s"] <= target:
            star = row["worlds"]
            break
    best = min((r["per_world_day_s"] for r in arr.get("rows", []) if "per_world_day_s" in r),
               default=None)
    #: ★**한계비용** — 세계를 하나 더 얹는 값. `d(벽시계)/dB` 를 이웃한 두 B 로 낸다.
    #:  이 값이 v5 다중프로세스의 세계·하루당 시간보다 크면 **B 를 얼마로 키워도 교차점은 없다**:
    #:  고정비를 0 으로 보내도(B→∞) 세계당 시간이 한계비용 아래로 내려갈 수 없기 때문이다.
    ok = [r for r in arr.get("rows", []) if "wall_s" in r and "worlds" in r]
    ok.sort(key=lambda r: r["worlds"])
    E = arr.get("epochs_total") or 1
    marg = []
    for a, b in zip(ok, ok[1:]):
        dB = b["worlds"] - a["worlds"]
        if dB <= 0:
            continue
        per_epoch = (b["wall_s"] - a["wall_s"]) / dB
        marg.append({"from": a["worlds"], "to": b["worlds"],
                     "marginal_per_world_day_s": per_epoch * (E / max(1, b["epochs_timed"]))})
    m_last = marg[-1]["marginal_per_world_day_s"] if marg else None
    #: ★판정은 **포화 구간**(from ≥ 8 세계)의 한계비용 최솟값으로 한다 — 작은 B 의 한계비용에는
    #:  고정비(컴파일·전송)가 섞여 음수까지 나오고, B→∞ 의 점근선과 무관하다.
    big = [m["marginal_per_world_day_s"] for m in marg if (m["from"] or 0) >= 8]
    m_min = min(big or [m["marginal_per_world_day_s"] for m in marg], default=None)
    m_all_min = min((m["marginal_per_world_day_s"] for m in marg), default=None)
    return {"v5_reference_procs": ref["procs"], "v5_per_world_day_s": target,
            "array_device": arr.get("device"), "B_star": star,
            "array_best_per_world_day_s": best,
            #: ⚠️ 이름이 헷갈려 남겨 둔 옛 키 — 값은 v5/배열 이라 **1 보다 작으면 배열이 느리다**
            "speedup_at_best": (target / best) if best else None,
            #: 읽기 쉬운 쪽: 배열 최고값이 v5 다중프로세스의 몇 배 뒤인가 (클수록 배열이 나쁘다)
            "v5_faster_at_best_x": (best / target) if best else None,
            "marginal": marg, "marginal_min_per_world_day_s": m_min,
            "marginal_min_all_B_per_world_day_s": m_all_min,
            "marginal_basis": "from>=8 세계 (포화 구간)",
            "marginal_last_per_world_day_s": m_last,
            "B_star_exists": (None if m_min is None else bool(m_min <= target)),
            "note": ("B* = 배열판 세계·하루당 시간이 v5 다중프로세스 아래로 내려가는 첫 B. "
                     "`B_star_exists=False` 면 **어떤 B 에서도 없다** — 세계 하나를 더 얹는 값"
                     "(한계비용)이 이미 v5 다중프로세스보다 크기 때문이다(점근선이 v5 위)."),
            "caveat_mode": ("표의 배열 행이 `same`(같은 세계 B벌 복사)이면 **상한**이다 — 서로 다른 시드"
                            "(연구 실제 무대)는 실측 1.21배 느리다(B=16 · 154.788s 대 187.370s). "
                            "v5 기준선은 서로 다른 시드 24개이므로 비교가 비대칭이다.")}


def table(doc: dict) -> str:
    lines = [f"■ 손익분기 — 세계·하루당 시간 (초, 작을수록 빠르다)   무대: {doc.get('stage', '')}",
             "   ★★이 표는 **굴리기 전용 경로**다 (`multiblock.run_epochs` 만 vmap · 기록·갱신 없음) —",
             "     **학습 경로가 아니다.** 연구가 학습에 쓰는 경로는 경계마다 호스트로 돌아와 세계를 쌓을 수",
             "     없다. 판정 수치는 아래 '■ ★학습 경로' 절이다. 이 표는 '동등성을 포기했을 때의 상한' 으로 읽어라.",
             "   ⚠️ 양쪽 모두 **시장(공간·시간 판매)이 없다** — 지금 돌고 있는 연구 팔(RL_SPACE)과는 비교 불가.",
             "   ⚠️ 배열 행의 `same` 은 같은 세계를 B벌 복사한 **상한**이다 (다른 시드 = 실제 무대는 1.21배 느리다,",
             "     B=16 실측 154.788s 대 187.370s). v5 기준선은 서로 다른 시드 24개다 — 비교가 비대칭이다.", ""]
    v5 = doc.get("v5") or {}
    if v5:
        lines.append(f"v5 단독 1세계   : {v5['single_mean_s']:8.3f} s/세계·하루  "
                     f"(코어 {v5['cpu_count']}개 기계)")
        for r in v5.get("parallel", []):
            lines.append(f"v5 ×{r['procs']:<3}프로세스 : {r['per_world_day_s']:8.3f} s/세계·하루 "
                         f"(벽시계 {r['wall_s']:.1f}s · 프로세스 안 평균 {r['inner_mean_s']:.1f}s)")
        lines.append("")
    for key in ("array_gpu", "array_cpu"):
        a = doc.get(key)
        if not a:
            continue
        lines.append(f"배열판 [{a['device']}{(' ' + a['device_kind']) if a.get('device_kind') else ''}]"
                     f" — {a['blocks']}블록 · 전체 {a['epochs_total']}에폭 · 세계 하나 상태 "
                     f"{a.get('one_world_state_mb')}MB")
        if a.get("source"):
            lines.append(f"  (표 = **한 세션** {a['source']} [{a.get('mode_shown')}] — 세션을 섞지 않는다. "
                         f"다른 세션 재측정은 아래 '독립 재측정' 줄)")
        lines.append("  세계  방식      잰에폭  벽시계(s)  ms/에폭  세계·하루당(s)  컴파일+첫회(s)  상태(MB)")
        for r in a["rows"]:
            if "error" in r:
                lines.append(f"  {r['worlds']:>4}  {r['mode']:<8}  ★막힘 ({r.get('state_mb')}MB): "
                             f"{r['error'][:70]}")
                continue
            lines.append(f"  {r['worlds']:>4}  {r['mode']:<8}  {r['epochs_timed']:>5}  "
                         f"{r['wall_s']:>9.3f}  {r['ms_per_epoch_all_worlds']:>7.1f}  "
                         f"{r['per_world_day_s']:>13.3f}  {r['compile_plus_first_s']:>13.1f}  "
                         f"{r.get('state_mb', 0):>8.1f}")
        lines.append("")
    for key, label in (("breakeven_gpu", "GPU"), ("breakeven_cpu", "CPU")):
        b = doc.get(key)
        if not b:
            continue
        tgt, best = b["v5_per_world_day_s"], b["array_best_per_world_day_s"]
        m_min = b.get("marginal_min_per_world_day_s")
        if b["B_star"] is not None:
            lines.append(f"★교차점 B* [{label}] = {b['B_star']} 세계 — 그 수부터 배열판이 "
                         f"v5 ×{b['v5_reference_procs']} 다중프로세스({tgt:.3f} s/세계·하루)보다 빠르다.")
        elif m_min is not None and m_min > tgt:
            lines.append(f"★교차점 B* [{label}] = **존재하지 않는다** — 세계를 하나 더 얹는 값(한계비용)이 "
                         f"이미 {m_min:.2f} s/세계·하루인데 v5 ×{b['v5_reference_procs']} 는 {tgt:.3f} 다. "
                         f"고정비를 0 으로 보내도(B→∞) 배열판은 한계비용 아래로 못 내려가므로 "
                         f"**포화가 다가가는 점근선 자체가 v5 위**다. 잰 범위의 최고값은 {best:.3f} "
                         f"(v5 가 {best / tgt:.1f}배 빠르다)이고, 같은 세계 복사가 아닌 실제 무대로 "
                         f"보정하면(×1.21) {best * 1.21 / tgt:.1f}배다. 메모리 벽·컴파일은 보조 근거다.")
        else:
            lines.append(f"★교차점 B* [{label}] = **없다** (잰 범위에서) — 배열판이 가장 좋을 때 "
                         f"{best:.3f} s/세계·하루인데 v5 ×{b['v5_reference_procs']} 는 {tgt:.3f} 다 → "
                         f"v5 다중프로세스가 아직 {best / tgt:.1f}배 빠르다.")
        for m in b.get("marginal", []):
            lines.append(f"    한계비용 {m['from']:>3}→{m['to']:<3} 세계: "
                         f"{m['marginal_per_world_day_s']:7.2f} s/세계·하루")
    for key, label in (("waste_gpu", "GPU"), ("waste_cpu", "CPU")):
        w = doc.get(key)
        if not w:
            continue
        lines += ["", f"■ 계산한 블록·스텝 대 쓴 블록·스텝 [{label}] — 조각 6 의 '래기드 while' 이 B 로 희석되나"]
        for m, d in w.items():
            for r in d.get("rows", []):
                if "efficiency" in r:
                    lines.append(f"  세계 {r['worlds']:>3} [{m:<8}] while {r['while_trips']:>7} · "
                                 f"계산 {r['computed_block_steps']:>10} · 쓴 {r['used_block_steps']:>10} "
                                 f"→ 유효율 {100 * r['efficiency']:5.1f}% ({r['ragged_factor']:.2f}배 낭비)")
                elif "order_padding" in r:
                    lines.append(f"  오더 칸 패딩 {100 * r['order_padding']:.1f}% "
                                 f"({r['order_rows_used']}/{r['order_rows_total']}) — 정적 · B 무관")
    for key, label in (("array_gpu_remeasured", "GPU"), ("array_cpu_remeasured", "CPU")):
        rm = doc.get(key)
        if not rm:
            continue
        lines += ["", f"■ 독립 재측정 [{label}] — 다른 세션·다른 프로세스에서 같은 칸을 다시 잰 값"]
        for g in rm:
            for r in g["rows"]:
                if r.get("per_world_day_s") is None:
                    continue
                lines.append(f"  세계 {r['worlds']:>3} [{g['mode']:<8}] {r['wall_s']:8.3f}s → "
                             f"{r['per_world_day_s']:7.3f} s/세계·하루  "
                             f"(반복 {len(r.get('reps_s') or [])}회 · {g['source']})")
    rg = doc.get("ragged")
    if rg:
        lines += ["", "■ 래기드 낭비 (같은세계 B벌 대 다른시드 B개 — 1 에 가까우면 낭비가 희석됐다)"]
        for r in rg:
            lines.append(f"  세계 {r['worlds']:>3}: same {r['same_s']:.3f}s · distinct {r['distinct_s']:.3f}s "
                         f"→ {r['ratio']:.2f}배")
    lines += ["", "■ B 를 키울 때 무엇이 먼저 막히나 (실측)", *_walls(doc)]
    lines += _trainpath_lines(doc)
    return "\n".join(lines) + "\n"


def _trainpath_lines(doc: dict) -> list:
    """★학습 경로 절 — 판정 수치는 이쪽이다 (굴리기 표가 아니다)."""
    out = []
    for key, label in (("trainpath_gpu", "GPU"), ("trainpath_cpu", "CPU")):
        t = doc.get(key)
        if not t:
            continue
        out += ["", f"■ ★학습 경로 [{label}] — **연구가 실제로 쓰는 경로** (`gpu/train.train_month`)",
                f"   무대: {t.get('stage', '')} · 세계 하나 (경계마다 호스트로 돌아오므로 세계를 쌓을 수 없다)"]
        for who, name in (("array", "배열"), ("v5", "v5  ")):
            for r in t["rows"].get(who, []):
                extra = (f"무대+컴파일 {r['setup_s']:.1f}s 별도 · " if "setup_s" in r else "")
                out.append(f"   {name} 경계 {r['boundaries']:>4}: {r['wall_s']:8.2f}s  "
                           f"({extra}갱신 {r['updates']} · 크레인결정 {r['crane']})")
        a, v = t.get("array_s_per_boundary"), t.get("v5_s_per_boundary")
        if a and v:
            E = t.get("epochs_per_day", 1441)
            out.append(f"   → 정상상태(두 길이의 차): 배열 {1000 * a:.0f} ms/경계 = {a * E:.0f} s/세계·하루 · "
                       f"v5 {1000 * v:.1f} ms/경계 = {v * E:.1f} s/세계·하루 → **배열이 {a / v:.0f}배 느리다**")
        vp = t.get("v5_parallel")
        if vp:
            out.append(f"   → v5 ×{vp['procs']} 프로세스(같은 학습 경로): {vp['per_world_day_s']:.1f} s/세계·하루"
                       + (f" → 배열 대비 **{t['array_over_v5_parallel']:.0f}배**"
                          if t.get("array_over_v5_parallel") else ""))
        out.append("   ⇒ 세계를 쌓는 축을 주장하려면 **먼저 경계 회계를 장치 안으로** 넣어 vmap 가능하게 만들고"
                   " 다시 재는 것이 순서다.")
    return out


def _walls(doc: dict) -> list:
    """벽 넷을 잰 값으로 적는다 — 처리량 포화 · 메모리 · 컴파일 · 래기드 낭비."""
    a = doc.get("array_gpu") or {}
    rows = [r for r in a.get("rows", []) if "per_world_day_s" in r]
    out = []
    if len(rows) >= 3:
        r1, rm1, rm2 = rows[0], rows[-2], rows[-1]
        gain = 100 * (1 - rm2["per_world_day_s"] / rm1["per_world_day_s"])
        out.append(f"  ① **처리량 포화** — 세계 1개 {r1['per_world_day_s']:.1f}s → "
                   f"{rm2['worlds']}개 {rm2['per_world_day_s']:.2f}s. 세계를 마지막으로 두 배 늘려 얻은 "
                   f"것은 {gain:.0f}% 뿐이다 — 이미 평평하다. 이게 **첫 번째 벽**이다.")
    #: 메모리 벽은 **다른 세션에서** 잰 경우가 많다 (큰 B 를 따로 돌린다) — 전 세션을 훑는다
    bad = [r for r in a.get("rows", []) if "error" in r]
    for d in (doc.get("array_gpu_all") or {}).values():
        for sub in (d.values() if isinstance(d, dict) else []):
            if isinstance(sub, dict):
                bad += [r for r in sub.get("rows", []) if "error" in r]
    bad.sort(key=lambda r: r.get("worlds", 0))
    if bad:
        out.append(f"  ② **메모리** — 세계 {bad[0]['worlds']}개에서 장치 메모리가 터진다. 상태 자체는 "
                   f"{bad[0].get('state_mb')}MB 뿐이고, 터지는 것은 `while_loop` 본문의 중간 배열이다.")
    if rows:
        out.append(f"  ③ **컴파일** — 세계 1개 {rows[0]['compile_plus_first_s']:.0f}초 → "
                   f"{rows[-1]['worlds']}개 {rows[-1]['compile_plus_first_s']:.0f}초 (한 번만 드는 값이지만 "
                   f"B 를 바꾸면 다시 든다).")
    w = doc.get("waste_gpu") or {}
    sm = {r["worlds"]: r for r in (w.get("same", {}).get("rows", [])) if "efficiency" in r}
    ds = {r["worlds"]: r for r in (w.get("distinct", {}).get("rows", [])) if "efficiency" in r}
    if sm and ds:
        k = max(set(sm) & set(ds))
        out.append(f"  ④ **래기드 낭비** — (가) 같은 세계 복사본에서 유효율이 "
                   f"{100 * sm[min(sm)]['efficiency']:.1f}% 로 B 와 무관한 것은 **정의상 자명**하다"
                   f"(세계가 비트 동일이라 park 시점도 같다 — while 반복이 B 마다 똑같은 것이 파수꾼이다). "
                   f"(나) 정보가 있는 쪽은 시드가 다른 경우이고 {100 * ds[min(ds)]['efficiency']:.1f}% → "
                   f"{100 * ds[k]['efficiency']:.1f}%(B={k}) 로 나빠진다. 다만 그 벌점은 (시드 모집단의 "
                   f"while 최대치)/(한 시드)로 **유계**라 B 를 키워도 무한히 커지지 않는다 — 벽시계 실측도 "
                   f"1.21배(B=16)에서 멈췄다. 낭비가 한 세계 안 21블록 사이에서 나므로 세계를 쌓는 것으로는 "
                   f"줄지 않는다. 오더 칸 패딩(29.2%)은 정적이라 그대로다.")
    return out or ["  (자료 부족)"]


# ───────────────────────────────────────────────── 진입점
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", default="array",
                    choices=("array", "v5", "waste", "trainpath", "merge"))
    ap.add_argument("--worlds", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--epochs", type=int, default=44,
                    help="잴 에폭 수 (전체 1,441 중 — 세계·하루당으로 환산한다)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--mode", default="distinct", choices=("same", "distinct", "both"))
    ap.add_argument("--procs", default=None, help="v5 다중프로세스 개수 목록 (기본 '1,코어수')")
    ap.add_argument("--load", type=int, default=LOAD)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--policy", default="sf_spt")
    ap.add_argument("--tag", default=None, help="결과 조각 파일 이름 (기본 which+장치)")
    ap.add_argument("--boundaries", default="100,300",
                    help="학습 경로에서 잴 경계 수 두 개 이상 (차로 정상상태를 뽑는다)")
    ap.add_argument("--ckpt", default=None, help="학습 경로에 쓸 v5 체크포인트 (없으면 고정 시드 초기화)")
    ap.add_argument("--train-procs", type=int, default=None,
                    help="학습 경로 v5 다중프로세스 개수 (예: 24)")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    worlds = [int(x) for x in args.worlds.split(",") if x.strip()]

    if args.which == "v5":
        procs = ([int(x) for x in args.procs.split(",")] if args.procs
                 else [1, int(os.cpu_count() or 1)])
        print(f"■ v5 기준선 — 터미널{args.load} 하루 · 프로세스 {procs}", flush=True)
        doc = bench_v5(procs=procs, load=args.load, seed=args.seed, reps=args.reps)
        (OUT / f"breakeven_v5{('_' + args.tag) if args.tag else ''}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0

    if args.which == "trainpath":
        import jax
        bs = [int(x) for x in args.boundaries.split(",") if x.strip()]
        print(f"■ ★학습 경로 — {jax.devices()[0].platform} · 경계 {bs} · 부하 {args.load}", flush=True)
        doc = bench_trainpath(boundaries=bs, load=args.load, seed=args.seed, ckpt=args.ckpt,
                              procs=args.train_procs)
        tag = args.tag or jax.devices()[0].platform
        (OUT / f"breakeven_trainpath_{tag}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0

    if args.which == "waste":
        import jax
        print(f"■ 래기드 낭비 — {jax.devices()[0].platform} · 세계 {worlds}", flush=True)
        modes = ("same", "distinct") if args.mode == "both" else (args.mode,)
        docs = {m: bench_waste(worlds=worlds, epochs=args.epochs, load=args.load, seed=args.seed,
                               mode=m, policy=args.policy) for m in modes}
        tag = args.tag or jax.devices()[0].platform
        (OUT / f"breakeven_waste_{tag}.json").write_text(
            json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0

    if args.which == "array":
        import jax
        plat = jax.devices()[0].platform
        print(f"■ 배열판 — {plat} · 세계 {worlds} · {args.epochs}에폭 × {args.reps}회", flush=True)
        modes = ("same", "distinct") if args.mode == "both" else (args.mode,)
        docs = {}
        for m in modes:
            docs[m] = bench_array(worlds=worlds, epochs=args.epochs, reps=args.reps,
                                  load=args.load, seed=args.seed, mode=m, policy=args.policy)
        tag = args.tag or plat
        (OUT / f"breakeven_array_{tag}.json").write_text(
            json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0

    # merge — 조각들을 모아 breakeven.json + 사람이 읽는 표
    doc: dict = {"stage": f"터미널{args.load} 하루 · 21블록 · seed {args.seed} · 정책 {args.policy}",
                 "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    v5p = OUT / "breakeven_v5.json"
    if v5p.exists():
        doc["v5"] = json.loads(v5p.read_text(encoding="utf-8"))
    for tag, key in (("gpu", "array_gpu"), ("cpu", "array_cpu")):
        #: ★조각 파일을 **전부** 모아 (worlds, mode) 로 묶고, 같은 칸이 여러 번 재졌으면
        #:  ① `reps_s` 가 있는 행(= 스크립트가 직접 쓴 행 · 사람이 전사한 행은 비어 있다)
        #:  ② 그중 더 빠른 행(기계가 한가할 때 잰 값 — 점유는 시간을 늘리기만 한다)
        #:  을 고른다. 고른 출처를 `source` 로 남겨 표와 JSON 에서 추적할 수 있게 한다
        #:  (검증 반박: '증거 JSON 이 스크립트가 쓴 파일이 아니다 — 반복 편차 증거가 사라졌다').
        files = sorted(OUT.glob(f"breakeven_array_*{tag}*.json"))
        if not files:
            continue
        #: ★**한 세션을 표로 고정한다** — 세션이 다른 행을 섞으면 한계비용·포화가 '기계 상태 차이' 를
        #:  재는 값이 된다(실측: 섞으면 한계비용이 −0.56 까지 나온다). 표는 세계 수가 가장 많이 채워진
        #:  한 (파일, 방식) 묶음이고, 다른 세션의 행은 `remeasured` 로 따로 싣는다(교차 검증용).
        groups: list = []
        alls: dict = {}
        for f in files:
            d = json.loads(f.read_text(encoding="utf-8"))
            alls[f.name] = d
            for mode, sub in d.items():
                if not isinstance(sub, dict) or "rows" not in sub:
                    continue
                rows = [dict(r, source=f.name) for r in sub["rows"] if "worlds" in r]
                if rows:
                    groups.append((f.name, mode, {k: v for k, v in sub.items() if k != "rows"},
                                   sorted(rows, key=lambda r: r["worlds"])))
        if not groups:
            continue
        #: 대표 세션 = (잰 세계 수가 많은 것 · 그다음 `same` 방식) — `distinct` 는 아래 비율로 보정한다
        fname, mode, head, rows = max(groups, key=lambda g: (len(g[3]), g[1] == "same"))
        pick = dict(head, rows=rows, mode_shown=mode, source=fname,
                    sources=[g[0] for g in groups])
        doc[key] = pick
        doc[key + "_all"] = alls
        doc[key + "_remeasured"] = [
            {"source": g[0], "mode": g[1],
             "rows": [{k: r.get(k) for k in ("worlds", "wall_s", "per_world_day_s", "reps_s")}
                      for r in g[3]]}
            for g in groups if (g[0], g[1]) != (fname, mode)]
        doc[key + "_distinct_rows"] = [r for g in groups if g[1] == "distinct" for r in g[3]]
        if doc.get("v5") and pick["rows"]:
            doc["breakeven_" + tag] = breakeven(doc["v5"], pick)
    for tag, key in (("gpu", "waste_gpu"), ("cpu", "waste_cpu")):
        pw = OUT / f"breakeven_waste_{tag}.json"
        if pw.exists():
            doc[key] = json.loads(pw.read_text(encoding="utf-8"))
    for tag, key in (("gpu", "trainpath_gpu"), ("cpu", "trainpath_cpu")):
        pt = OUT / f"breakeven_trainpath_{tag}.json"
        if pt.exists():
            doc[key] = json.loads(pt.read_text(encoding="utf-8"))
    sm = {r["worlds"]: r for r in (doc.get("array_gpu") or {}).get("rows", []) if "wall_s" in r}
    ds = {r["worlds"]: r for r in doc.get("array_gpu_distinct_rows", []) if "wall_s" in r}
    if sm and ds:
        doc["ragged"] = [{"worlds": w, "same_s": sm[w]["wall_s"], "distinct_s": ds[w]["wall_s"],
                          "ratio": ds[w]["wall_s"] / sm[w]["wall_s"],
                          "source": (sm[w].get("source"), ds[w].get("source"))}
                         for w in sorted(set(sm) & set(ds))]
    (OUT / "breakeven.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    txt = table(doc)
    (OUT / "breakeven.txt").write_text(txt, encoding="utf-8")
    print(txt, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
