"""조각 8 통합 동등성 — 배열 학습 드라이버가 v5 와 같은 답을 내는가 ([[YR-327]]).

기대값을 손으로 적지 않는다 — **v5 정본을 실제로 불러** 같은 입력의 답을 받아 비교한다.

■ 층 — **무엇이 갱신되는 상태에서 잰 것인지**를 층 이름에 박아 둔다 (2026-09-27 수정)
  (0) 기록 채널이 세계를 바꾸지 않는가            — `record=True` 세계와 `record=False` 세계의 잎 비트
  (1) 배치 테이프 합치기 == 낱개 `select_record`   — `gpu/train.merge_rec` 대 `ppo_runtime.select_record`
  (A) **갱신 산술만** (엔진 없음): v5 `run_debug` 의 **실제** 구간 테이프 → v5 `ppo.update` 와 배열
      `ppo_update` — 미니배치별 손실·KL·노름 rtol 1e-4 · 미니배치 수·조기중단 `==` · 파라미터 atol 1e-5
  (S) **접합부**: 배열 세계가 스스로 모은 `IntervalRow` → `train.intervals_to_batch` 가
      v5 테이프를 `ppo_buffer.pack_intervals` 로 담은 것과 잎·dtype·GAE 까지 같은가
  (B) **얼린 정책 굴리기**(`training=False` = 갱신 0회) · 엔진 붙여 경계 120개 — 경계 Φ ·
      **경계마다** 결정 계수기 · 보고 정수
  (C) **얼린 정책 굴리기**(`training=False`) · v5 가 스스로 멈출 때까지 (접두사 대조)
  (D) ★**학습 켠 닫힌 고리** (`training=True` — 수집 → 버퍼 → GAE → 갱신 → **갱신된 망으로 다음 수집**):
      경계 Φ · 경계마다 계수기 · **결정 낱개**(역할·시각·선택 색인·후보 수) · 갱신 보고 · 최종 파라미터.
      v5 쪽 시장 선택자가 버퍼에 담는 판매자·구매자 Choice 를 **떼어낸 대조 전용 무대**다 (아래 ⚠️ 시장).
  (E) **추첨(sample) 수집** — v5 와 표본은 못 맞춘다(규칙 ⑥). 분포·재현성·자기일관성으로 본다.
  (F) **배열 단독 완주** — 날 경계를 넘기고 `finish` 까지 (v5 는 이 경계를 못 넘는다 · 아래 ⚠️ 갈래)

■ ⚠️ 이 파일의 v5 는 **원코드 그대로의 v5 가 아니다** (모든 층에 걸린 조건)
  ① `stage/container_contract.require_container_plan` 을 몽키패치로 지나간다 (`_bypass_container_gate`).
     기본(`MonthTerminal`) 갈래에서는 그 게이트가 **막고**(실측 부하 60·150·300 전부
     `unbound_vessel_load_streams`), 고정 화물(`CargoTerminal`) 갈래에서는 **통과한다**(실측
     부하 60·300 `passed=True`). 배열판은 기본 갈래만 옮겼으므로 여기서 지나가는 것이고,
     **지나간 세계는 v5 가 "학습해서는 안 된다" 고 판정한 입력**이다.
  ② v5 PPO 경로는 시장에도 정책을 묻고 그 결정을 **학습 버퍼에 담는다** (`month_run.py:301`).
     실측으로 v5 첫 갱신의 micro_actions 147 = 크레인 81 + **판매자 66** 이다 — 배열판에 시장 다리가
     없으므로 (D) 층은 v5 쪽 판매자·구매자 Choice 를 버퍼에서 떼어낸 무대에서만 성립한다.
  ③ 그래서 **"v6 로 30일 학습을 재현할 수 있다" 는 이 파일의 결론이 아니다.** 여기서 받치는 것은
     '얼린 정책 굴리기' · '갱신 산술' · '시장을 떼어낸 닫힌 고리' 세 가지다.

■ 시간 예산
  (A)(B)(C)(D) 는 v5 torch 망을 결정마다 실제로 굴려 대조하므로 느리다. 환경변수로 규모를 줄인다:
    TRAIN_EQUIV_SECONDS   (B) 층이 굴릴 시뮬 초 (기본 7200 = 경계 120)
    TRAIN_EQUIV_D_SECONDS (D) 층이 굴릴 시뮬 초 (기본 2400 = 경계 40)
    TRAIN_EQUIV_D_ROLLOUT (D) 층 `rollout_intervals` (기본 10 → 경계 40 에서 갱신 4회)
    TRAIN_EQUIV_DAYS      (C) 층 날수 (기본 3)
    TRAIN_EQUIV_SKIP      "A,S,B,C,D,E,F" 중 건너뛸 층
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import sys
import time

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                                     # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_ROOT, "src"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from yard_rl.v6.gpu import dispatch as DP                                   # noqa: E402
from yard_rl.v6.gpu import month as MO                                      # noqa: E402
from yard_rl.v6.gpu import multiblock as MB                                 # noqa: E402
from yard_rl.v6.gpu import ppo_buffer as PB                                 # noqa: E402
from yard_rl.v6.gpu import ppo_runtime as PR                                # noqa: E402
from yard_rl.v6.gpu import ppo_update as PU                                 # noqa: E402
from yard_rl.v6.gpu import train as TR                                      # noqa: E402
from yard_rl.v6.gpu import v5net as VN                                      # noqa: E402
from yard_rl.v6.stage import month as v5m                                   # noqa: E402

F = jnp.float64
SEED = 9_900_302
CKPT = os.path.join(_ROOT, "outputs", "v5", "yr302-final-train", "policy.pt")
REPORT: dict = {}
_SKIP = {s.strip().upper() for s in os.environ.get("TRAIN_EQUIV_SKIP", "").split(",") if s.strip()}


# ───────────────────────────────────────────────── 공통 도구
def _plan(n_days: int, load: int):
    return [v5m.DayPlan(index=i, load=load, label="train-equiv", seed=SEED + 1_000 * (i + 1),
                        t0=i * v5m.DAY_S, n_days=n_days) for i in range(n_days)]


def _state_dict():
    """실제 학습 체크포인트 `yr302-final-train` (없으면 고정 시드 무작위 망)."""
    if os.path.exists(CKPT):
        import numpy  # noqa: F401  ★numpy 를 torch 보다 먼저 (OMP #15)
        from yard_rl.v6.ppo.checkpoint import load_policy
        pol = load_policy(CKPT)
        return {k: v.detach().cpu().numpy() for k, v in pol.state_dict().items()}, "ckpt"
    import numpy  # noqa: F401
    import torch
    from yard_rl.v6.ppo.model import BlockPolicy
    torch.manual_seed(20_260_927)
    pol = BlockPolicy()
    return {k: v.detach().cpu().numpy() for k, v in pol.state_dict().items()}, "seed20260927"


def _eq(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind == "f":
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def _tree_diff(x, y):
    """잎마다 비트 대조 (NaN 은 같은 자리면 같다). 갈린 잎의 (경로, 앞 6개) 목록."""
    px, _ = jax.tree_util.tree_flatten_with_path(x)
    py, _ = jax.tree_util.tree_flatten_with_path(y)
    assert len(px) == len(py), f"pytree 구조가 다르다 {len(px)} vs {len(py)}"
    return [(jax.tree_util.keystr(k), np.ravel(np.asarray(a))[:6], np.ravel(np.asarray(b))[:6])
            for (k, a), (_, b) in zip(px, py) if not _eq(a, b)]


def _block_hash(run, b: int) -> str:
    """블록 하나의 상태 해시 — 날마다 v5·배열을 한 줄로 대조한다."""
    W = run.tw.blocks
    h = hashlib.sha256()
    for name in ("clock", "steps"):
        h.update(np.asarray(getattr(W, name))[b].tobytes())
    for name in ("queue_area_s", "tail_area_s", "loaded_gantry_m", "empty_gantry_m",
                 "rehandles", "pre_rehandles", "completed_ext", "completed_vessel",
                 "vessel_delay_s", "positioning"):
        h.update(np.asarray(getattr(W.kpi, name))[b].tobytes())
    return h.hexdigest()[:16]


KINDS = ("SERVE", "PRE_REHANDLE", "REPOSITION", "WAIT")


def _array_epoch_probe(box: dict, rows: list):
    """`train_month(on_epoch=…)` 자리 — 경계마다 (시각·Φ·크레인 계수기·역할 계수·갱신 수) 를 남긴다.

    ★총계만 맞대면 갈림을 가린다 — 실측으로 확인했다: 학습을 켠 재실행에서 총 크레인 결정이
      164 == 164 로 같은데도 세계는 갈려 있었다(PRE_REHANDLE 2 대 0). 그래서 **경계마다** 본다.
    """
    def on_epoch(run, e, t, codes, out):
        ts = box["ts"]
        rows.append(dict(t=round(float(t), 6), cost=float(out.cost),
                         acts=[int(x) for x in np.asarray(ts.st.crane_actions)],
                         crane=int(np.asarray(ts.st.role_counts)[2]),
                         updates=int(np.asarray(ts.st.updates)),
                         buffered=len(ts.buffer)))
    return on_epoch


def _cmp_boundaries(rt5, rowsA: list, *, rel_tol: float = 1e-9) -> dict:
    """경계마다 Φ 와 **결정 계수기**를 맞댄다. 처음 갈린 지점을 돌려준다 (숨기지 않는다)."""
    n = min(len(rt5.bounds), len(rowsA))
    worst, first_bad, first_count_bad = 0.0, None, None
    for i in range(n):
        t5, c5 = rt5.bounds[i]
        ra = rowsA[i]
        assert abs(t5 - ra["t"]) < 1e-6, f"경계 {i} 시각이 다르다 v5 {t5} vs 배열 {ra['t']}"
        rel = abs(ra["cost"] - c5) / max(1.0, abs(c5))
        worst = max(worst, rel)
        if rel > rel_tol and first_bad is None:
            first_bad = dict(index=i, t=t5, v5=c5, array=ra["cost"], rel=rel)
        _t, roles5, acts5 = rt5.snaps[i]
        a5 = [int(acts5.get(k, 0)) for k in KINDS]
        if (a5 != ra["acts"] or int(roles5.get("crane", 0)) != ra["crane"]) and first_count_bad is None:
            first_count_bad = dict(index=i, t=t5, v5_acts=a5, array_acts=ra["acts"],
                                   v5_crane=int(roles5.get("crane", 0)), array_crane=ra["crane"])
    return dict(n=n, v5_bounds=len(rt5.bounds), array_bounds=len(rowsA), worst_rel=worst,
                first_bad=first_bad, first_count_bad=first_count_bad)


def _decisions_of_tape(choices, *, role: int = 2) -> list:
    """`PendingTape`(한 구간) → 결정 낱개 목록 `(블록, 시각, 선택 색인, 후보 수)`."""
    n = np.asarray(choices.n)
    tm, ac, na, rl = (np.asarray(choices.time_s), np.asarray(choices.action),
                      np.asarray(choices.n_actions), np.asarray(choices.role))
    out = []
    for b in range(n.shape[0]):
        for c in range(int(n[b])):
            if int(rl[b, c]) != role:
                continue
            out.append((int(b), round(float(tm[b, c]), 6), int(ac[b, c]), int(na[b, c])))
    return out


def _cmp_decisions(v5_selects: list, arr: list) -> dict:
    """결정을 **낱개로** 맞댄다 — `(블록, 시각)` 으로 묶어 그 안을 정렬해 비교한다.

    같은 시각·같은 블록에 두 크레인이 결정하면 v5 는 `dp.crane_ids`(사전순) 순서, 배열은 크레인 번호
    순서로 담는다 — 순서가 다를 수 있으므로 묶음 안에서만 정렬한다. 그래도 '어떤 결정이 몇 개' 는
    낱개로 대조된다 (총계 대조가 가리는 갈림을 여기서 잡는다).
    """
    def group(rows):
        g: dict = {}
        for r in rows:
            g.setdefault((r[0], r[1]), []).append(tuple(r[2:]))
        return {k: sorted(v) for k, v in g.items()}
    g5 = group([(b, t, a, nn) for (_role, t, b, a, nn) in v5_selects])
    ga = group(arr)
    keys = sorted(set(g5) | set(ga))
    bad = []
    for k in keys:
        if g5.get(k) != ga.get(k):
            bad.append(dict(block=k[0], t=k[1], v5=g5.get(k), array=ga.get(k)))
    return dict(v5=len(v5_selects), array=len(arr), groups=len(keys), n_bad=len(bad),
                first_bad=(bad[0] if bad else None), bad=bad[:5])


# ═══════════════════════════════════════════════ (0) 기록 채널이 세계를 바꾸지 않는가
def test_00_recording_channel_does_not_change_the_world():
    """`engine_step`/`multiblock` 의 기록 채널을 켜도 궤적이 **잎 비트까지** 같아야 한다.

    켜면 `VF.features` 가 한 번 더 돌고 while_loop 캐리에 테이프가 실린다 — 둘 다 세계를 읽기만 한다.
    """
    sd, tag = _state_dict()
    net = VN.load_v5_params(sd)
    days = _plan(2, 150)
    got = {}
    for rec in (True, False):
        run, tt, lay, eng, params, prof, g, ex = TR.month_setup(
            seed=SEED, days=days, net_params=net, blocks=("Y01", "Y04"), cap_moves=24, record=rec)
        cfg = dataclasses.replace(TR.TrainConfig(n_blocks=int(run.b)), training=False).runtime()
        ts = TR.new_train_state(params, TR.TrainConfig(n_blocks=int(run.b)), seed=SEED)
        slots, grid = MO.gate_out_slots(run, lay), MO.epoch_grid(run)
        run, _rows, _op = MO.open_day(run, tt, lay, 0, seed=SEED, profile=prof, t0=0.0)
        if rec:
            tape_fn = TR.make_tape_fn(cfg)
            tcfg = TR.TrainConfig(n_blocks=int(run.b))
            for e in range(120):
                run, _c, _s, t, ts = TR.collect_epoch(run, tt, lay, e, eng, ts, cfg,
                                                      tape_fn=tape_fn, slots=slots, grid=grid)
                #: ★경계를 같이 부른다 — 부르지 않으면 `started` 가 거짓이라 v5 가 던지는 자리와 같아져
                #:  (`F_DECISION_NO_BOUNDARY`) 결정이 기록도 계수도 되지 않는다. 경계는 세계를 읽기만 하므로
                #:  아래의 잎 비트 대조는 그대로 뜻이 있다.
                TR.boundary_at(run, lay, prof, g, t, ts, cfg, tcfg,
                               crane_order=ex["crane_order"])
            got["rec_state"] = ts.st
        else:
            for e in range(120):
                run, _c, _s = MO.month_epoch(run, tt, lay, e, eng, slots=slots, grid=grid)
        got["rec" if rec else "plain"] = run
    bad = _tree_diff(got["rec"].tw.blocks, got["plain"].tw.blocks)
    assert not bad, f"기록을 켜니 세계가 갈렸다 ({len(bad)} 잎): {bad[:3]}"
    bad2 = _tree_diff(got["rec"].tw.ledger, got["plain"].tw.ledger)
    assert not bad2, f"기록을 켜니 전역 원장이 갈렸다: {bad2[:3]}"
    asked = int(np.asarray(got["rec_state"].crane_actions).sum())
    assert asked > 0, "120 에폭에서 크레인 결정이 한 번도 없었다 — 대조가 공허하다"
    REPORT["(0) 기록채널"] = dict(net=tag, epochs=120, crane_asks=asked,
                                 leaves=len(jax.tree_util.tree_leaves(got["rec"].tw.blocks)))


# ═══════════════════════════════════════════════ (1) 배치 합치기 == 낱개 select_record
def test_01_merge_matches_select_record():
    """`train.merge_rec` (블록축 한꺼번에) 가 `ppo_runtime.select_record` 반복과 **잎 비트까지** 같은가.

    같은 무대를 두 길로 기록한다: ① `merge_rec` ② 블록·크레인 순서대로 `select_record`.
    창 밖·경계 전·마스크 빈 칸·칸 넘침까지 일부러 섞어 넣는다.
    """
    B, K, A, D, C = 3, 2, 5, PR.ROLES and 37, 4
    cfg = PR.RuntimeConfig(n_blocks=B, cmax=C, amax=A, input_dim=D)
    rng = np.random.default_rng(4242)

    def rec_of(step: int):
        o = np.array([[step % 2 == 0, True], [True, False], [step > 1, step > 1]], bool)
        rows = rng.normal(size=(B, K, A, D))
        mask = rng.random((B, K, A)) > 0.35
        if step == 2:                                 # 마스크가 전부 거짓인 결정 하나 (v5 는 던진다)
            mask[0, 0, :] = False
        nact = np.full((B, K), A, np.int32)
        if step == 3:
            nact[1, 0] = A + 2                        # 후보 칸 넘침
        return TR.DecisionRec(open=jnp.asarray(o), rows=jnp.asarray(rows, F),
                              mask=jnp.asarray(mask),
                              action=jnp.asarray(rng.integers(0, A, size=(B, K)), jnp.int32),
                              log_prob=jnp.asarray(rng.normal(size=(B, K)), F),
                              n_actions=jnp.asarray(nact), kind=jnp.asarray(
                                  rng.integers(0, 4, size=(B, K)), jnp.int32),
                              time_s=jnp.asarray([100.0, 100.0, 100.0], F))

    steps = [rec_of(i) for i in range(6)]
    st0 = PR.new_state(cfg)
    # ★첫 경계를 지난 상태 (started=True · 구간 시작 60초 · 학습창 없음 = 늘 수집)
    st0 = st0._replace(started=jnp.ones((), bool), time_s=jnp.asarray(60.0, F))
    dec = jnp.ones((B,), bool)
    act = jnp.ones((B,), bool)

    carry = TR.new_tape_carry(st0, cfg)
    for rec in steps:
        carry = TR.merge_rec(carry, rec, dec, act, cfg)

    st = st0
    for rec in steps:                                 # 낱개 — 블록 바깥, 크레인 안쪽 (merge 와 같은 순서)
        for k in range(K):
            for b in range(B):
                if not bool(rec.open[b, k]):
                    continue
                st, out = PR.select_record(
                    st, cfg, role=TR.ROLE_CRANE, block=b, t=rec.time_s[b], rows=rec.rows[b, k],
                    mask=rec.mask[b, k], action=rec.action[b, k], log_prob=rec.log_prob[b, k],
                    n_actions=rec.n_actions[b, k])
                #: v5 는 `select` 가 던진 자리에서는 `crane_actions` 도 안 센다 (`ppo/crane.py:81-82`)
                if not int(out.flags) & TR._RAISE_BITS:
                    st = PR.count_crane_action(st, rec.kind[b, k])
    bad = _tree_diff(carry.pending, st.pending)
    assert not bad, f"테이프가 갈렸다: {bad[:4]}"
    assert _eq(carry.role_counts, st.role_counts), (carry.role_counts, st.role_counts)
    assert _eq(carry.crane_actions, st.crane_actions), (carry.crane_actions, st.crane_actions)
    assert int(carry.flags) == int(st.flags), (PR.flag_names(int(carry.flags)),
                                              PR.flag_names(int(st.flags)))
    fl = PR.flag_names(int(carry.flags))
    assert {"MASK_EMPTY", "ACTION_OVERFLOW", "PENDING_OVERFLOW"} <= set(fl), fl
    REPORT["(1) 합치기"] = dict(steps=len(steps), flags=list(fl),
                               recorded=int(np.asarray(carry.pending.n).sum()),
                               overflow=int(np.asarray(carry.pending.overflow).sum()))


# ═══════════════════════════════════════════════ (A) 엔진 없이 — 실제 v5 구간 테이프로 갱신 대조
def _bypass_container_gate():
    """★v5 `run_month(ppo=…)` 의 **컨테이너 계약 게이트**를 이 시험에서만 지나간다.

    ★정정 (2026-09-27 수정 단계 — 전에는 "어떤 부하·어떤 날수에서도 통과하지 않는다" 고 적었다):
      막히는 것은 **기본(`MonthTerminal`) 갈래**뿐이다. 실측 —
        · 기본 갈래  : 부하 60 `{duplicate_exits:1, unbound_vessel_load_moves:3952,
                        unbound_vessel_load_streams:10}` · 부하 300 `{…:9}` → **실패**
        · 고정 화물(`CargoTerminal`, `seed_data` 를 주는 갈래) : 부하 60·300 모두 **passed=True**
      즉 v5 PPO 경로는 **연구선 설정(고정 화물)에서는 잘 돈다** (실측: 부하 300·3일·추첨으로
      t=90,000초까지 완주 · 날 경계 86,400 통과 · 갱신 25회). 우리가 게이트를 지나가는 이유는
      **배열판이 기본 갈래만 옮겼기 때문**이고, 그래서 **여기서 지나간 세계는 v5 가 "학습해서는
      안 된다" 고 판정한 입력**이다 (본선 적하 물량 3,952건이 야드 상자에 묶이지 않은 세계).
      게이트 자체는 컨테이너 신원 부기에 관한 것이고 PPO 산술·엔진 물리와 무관하다.
    v5 파일은 고치지 않는다 (몽키패치는 이 시험 안에서만 되돌린다).
    """
    from yard_rl.v6.stage import month_run as MR
    old = MR.require_container_plan
    MR.require_container_plan = lambda rep: None
    return MR, old


def _v5_tap_run(seconds: float, load: int, config):
    """v5 `ppo/run.run_debug` 의 몸통 — 첫 `_update` 직전 상태를 통째로 떠 온다.

    v5 파일은 수정하지 않는다. `PPORuntime` 을 **상속**해 `_update` 를 가로챈다 (시험 파일 안).
    """
    import numpy  # noqa: F401
    import torch
    from yard_rl.v6.ppo.model import BlockPolicy
    from yard_rl.v6.ppo.runtime import DebugStop, PPORuntime
    MR, old_gate = _bypass_container_gate()
    run_month = MR.run_month

    class Tap(PPORuntime):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.taps: list = []

        def _update(self, bootstrap):
            if self.training and self.buffer and not self.taps:
                self.taps.append(dict(
                    intervals=copy.deepcopy(self.buffer), bootstrap=np.asarray(bootstrap).copy(),
                    state_dict={k: v.detach().cpu().numpy().copy()
                                for k, v in self.policy.state_dict().items()},
                    rng_state=copy.deepcopy(self.rng.bit_generator.state),
                    n_opt_state=len(self.optimizer.state_dict().get("state", {}))))
            super()._update(bootstrap)

    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    policy = BlockPolicy()
    rt = Tap(policy, config=config, seed=SEED, training=True, stop_s=seconds)
    days = [v5m.DayPlan(index=i, load=load, label="v5-debug", seed=SEED + 1000 * (i + 1),
                        t0=i * v5m.DAY_S, n_days=2) for i in range(2)]
    try:
        run_month(seed=SEED, days=days, ppo=rt)
    except DebugStop:
        pass
    finally:
        MR.require_container_plan = old_gate
    return rt


@pytest.mark.skipif("A" in _SKIP, reason="TRAIN_EQUIV_SKIP")
def test_A_update_matches_v5_on_real_rollout():
    """★A층 — v5 가 **실제로 모은** 구간 60개로 갱신을 두 판에서 돌려 비교한다."""
    import numpy  # noqa: F401
    import torch
    from yard_rl.v6.ppo.model import BlockPolicy
    from yard_rl.v6.ppo.runtime import PPOConfig
    from yard_rl.v6.ppo.update import update as v5_update

    load = int(os.environ.get("TRAIN_EQUIV_A_LOAD", "3500"))   # plan_month 의 보통 날 부하
    config = PPOConfig()
    t0 = time.perf_counter()
    rt = _v5_tap_run(float(config.rollout_intervals) * config.time_unit_s + 1.0, load, config)
    assert rt.taps, "v5 가 첫 갱신에 닿지 못했다 — 초를 늘려라"
    tap = rt.taps[0]
    t_collect = time.perf_counter() - t0
    iv, boot, sd = tap["intervals"], tap["bootstrap"], tap["state_dict"]
    assert tap["n_opt_state"] == 0, "첫 갱신인데 옵티마이저 상태가 비어 있지 않다"

    # ── v5 판: 같은 가중치·같은 난수 상태에서 `update` 한 번
    torch.set_num_threads(1)
    pol5 = BlockPolicy()
    pol5.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()}, strict=True)
    opt5 = torch.optim.Adam(pol5.parameters(), lr=config.learning_rate)
    rng5 = np.random.default_rng(SEED)
    rng5.bit_generator.state = copy.deepcopy(tap["rng_state"])
    rep5 = v5_update(pol5, opt5, iv, boot, config, rng5)
    after5 = {k: v.detach().cpu().numpy().copy() for k, v in pol5.state_dict().items()}

    # ── 배열 판: 같은 구간을 `pack_intervals` 로 담고 `ppo_update`
    batch, over = PB.pack_intervals(iv)
    assert not over.any, f"칸이 모자라 결정을 못 담았다 {over}"
    assert np.asarray(batch.values).dtype == np.float32, "v5 values 는 float32 여야 한다"
    net = VN.load_v5_params(sd)
    adam = PU.init_adam(net)
    rngA = np.random.default_rng(SEED)
    rngA.bit_generator.state = copy.deepcopy(tap["rng_state"])
    hyper = PU.hyper_from_v5(config)
    adv, ret = PB.gae(batch, boot, gamma=config.gamma, lam=config.gae_lambda,
                      time_unit_s=config.time_unit_s)
    tape, adv_e, ret_e = TR.batch_to_tape(batch, adv, ret)
    n_entries = int(batch.n_intervals) * batch.n_blocks
    orders = PU.draw_orders(rngA, n_entries=n_entries, hyper=hyper)
    netA, adamA, out = PU.ppo_update(net, adam, tape, adv_e, ret_e, orders, hyper)
    repA = PU.as_v5_report(out, n_intervals=int(batch.n_intervals))

    # ── 정수·불리언은 정확히
    for key in ("minibatches", "early_stopped", "intervals", "block_samples",
                "active_block_samples", "micro_actions"):
        assert repA[key] == rep5[key], f"{key}: 배열 {repA[key]} vs v5 {rep5[key]}"
    # ── 손실·KL·노름
    assert abs(repA["max_kl"] - rep5["max_kl"]) < 2e-7, (repA["max_kl"], rep5["max_kl"])
    np.testing.assert_allclose(repA["loss"], rep5["loss"], rtol=1e-4)
    np.testing.assert_allclose(repA["max_grad_norm_before_clip"],
                               rep5["max_grad_norm_before_clip"], rtol=1e-4)
    # ── 최종 파라미터 (actor.bias 는 식별 불가 — ppo_update 담당의 기록 참조)
    arr = VN.to_v5_state_dict(netA)
    worst = {}
    for k, v in after5.items():
        if k == "actor.bias":
            worst[k] = float(np.max(np.abs(arr[k] - v)))
            continue
        d = float(np.max(np.abs(arr[k] - v)))
        worst[k] = d
        assert d < 1e-5, f"{k} 최종 파라미터 차 {d:.3g} ≥ 1e-5"
    REPORT["(A) 갱신"] = dict(intervals=int(batch.n_intervals), blocks=batch.n_blocks,
                             cmax=batch.c_max, amax=batch.a_max,
                             micro_actions=repA["micro_actions"], minibatches=repA["minibatches"],
                             early_stopped=repA["early_stopped"],
                             kl=(repA["max_kl"], rep5["max_kl"]),
                             loss=(repA["loss"], rep5["loss"]),
                             worst_param={k: round(v, 12) for k, v in worst.items()},
                             collect_s=round(t_collect, 1))


# ═══════════════════════════════════════════════ (B) 엔진 붙여 — 같은 세계를 v5·배열로
def _v5_month_with_ppo(days, *, sd, seconds, training=False, tolerate=False, config=None,
                       detach_market=False):
    """v5 `run_month(ppo=PPORuntime(...))` 기본 갈래. 경계마다 (t, cost)·계수기 사진·결정 낱개를 남긴다.

    ⚠️ v5 PPO 경로는 **시장을 연다** (`market.seller.selector = market.buyer.selector = ppo.select`,
       `month_run.py:301`) — `arm` 을 바꾸면 `run_month` 가 거절한다 (month_run.py:224).
       배열판에는 시장 다리가 없다(조각 8 범위 밖)므로:
         · 얼린 정책((B)(C))에서는 거래가 0 건인 창이면 **세계가 같다** (시험이 `traded_edges == 0` 단언).
         · 그러나 **학습을 켜면** 그 시장 결정이 v5 버퍼에 들어가 갱신이 달라진다 (실측 44.9%).
           그래서 `detach_market=True` 는 시장 결정을 **v5 와 같은 답으로 그대로 내되** 방금 담은
           `Choice` 만 버퍼에서 빼낸다 → 갱신 표본이 크레인만 남아 배열판과 같은 무대가 된다.
           (v5 파일은 고치지 않는다 — `PPORuntime` 을 상속해 `select` 를 감싼다.)
    """
    import numpy  # noqa: F401
    import torch
    from yard_rl.v6.ppo.model import BlockPolicy
    from yard_rl.v6.ppo.runtime import DebugStop, PPOConfig, PPORuntime
    MR, old_gate = _bypass_container_gate()
    run_month = MR.run_month

    torch.set_num_threads(1)
    policy = BlockPolicy()
    policy.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()}, strict=True)

    class Tap(PPORuntime):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.bounds: list = []
            self.snaps: list = []
            self.selects: list = []          # 크레인 결정 낱개 (역할·시각·블록 색인·선택·후보 수)
            self.market_selects: list = []   # 시장 결정 낱개 (배열판에 대응물이 없다)
            self.detached = 0                # 버퍼에서 빼낸 시장 Choice 수

        def select(self, role, bid, t, rows, mask=None):
            lst, n0 = None, 0
            if (detach_market and role != "crane" and self.time_s is not None
                    and self.collecting_at(self.time_s) and hasattr(self, "pending")):
                lst = self.pending[self.index[bid]]
                n0 = len(lst)
            idx = super().select(role, bid, t, rows, mask=mask)
            if lst is not None and len(lst) > n0:
                #: ★시장 Choice 만 빼낸다 — 결정 자체(세계에 주는 답)는 v5 그대로다
                self.detached += len(lst) - n0
                del lst[n0:]
            row = (role, round(float(t), 6), self.index[bid], int(idx), int(len(rows)))
            (self.selects if role == "crane" else self.market_selects).append(row)
            return idx

        def boundary(self, t, *, terminated=False, final=False):
            #: ★`try/finally` — `super().boundary` 는 절단 자리에서 `DebugStop` 을 던지고 나간다.
            #:  전에는 그 뒤 줄에서 기록했으므로 **마지막 경계가 빠졌다**(실측 v5 120 대 배열 121).
            #:  절단·마지막 갱신이 걸리는 바로 그 경계라 대조에서 빠지면 안 된다.
            ok = False
            try:
                super().boundary(t, terminated=terminated, final=final)
                ok = True
            except DebugStop:
                ok = True                    # 절단은 정상 종료 — 그 경계까지 기록한다
                raise
            finally:
                if ok:
                    self.bounds.append((round(float(t), 6), float(self.cost_krw)))
                    self.snaps.append((round(float(t), 6), dict(self.role_counts),
                                       dict(self.crane_actions)))

    rt = Tap(policy, config=config or PPOConfig(), seed=SEED, training=training,
             sample_actions=False, stop_s=seconds)
    stop = None
    try:
        run_month(seed=SEED, days=days, ppo=rt)
    except DebugStop:
        stop = "DebugStop"
    except BaseException as ex:                       # ★v5 가 스스로 멈춘 자리를 기록한다
        if not tolerate:
            raise
        stop = f"{type(ex).__name__}: {str(ex)[:120]}"
    finally:
        MR.require_container_plan = old_gate
    return (rt, stop) if tolerate else rt


@pytest.mark.skipif("B" in _SKIP, reason="TRAIN_EQUIV_SKIP")
def test_B_frozen_rollout_boundaries_match_v5():
    """★B층 — **얼린 정책 굴리기**(`training=False` = 갱신 0회). 경계 Φ·**경계별 계수기**·보고 정수.

    ⚠️ 이 층은 '학습 루프가 같다' 를 보이지 않는다 — 갱신이 한 번도 일어나지 않는 상태의 대조다.
       '수집 → 갱신 → 갱신된 망으로 계속' 의 합성은 (D) 층이 본다.
    """
    seconds = float(os.environ.get("TRAIN_EQUIV_SECONDS", "7200"))   # 명세의 경계 120개
    sd, tag = _state_dict()
    days = _plan(2, int(os.environ.get("TRAIN_EQUIV_B_LOAD", "3500")))
    t0 = time.perf_counter()
    rt5 = _v5_month_with_ppo(days, sd=sd, seconds=seconds, training=False)
    assert rt5.report()["traded_edges"] == 0, "v5 가 이 창에서 거래를 했다 — 배열판에 시장이 없어 대조 불가"
    t_v5 = time.perf_counter() - t0

    net = VN.load_v5_params(sd)
    tcfg = TR.TrainConfig(training=False, stop_s=seconds, cmax=256, amax=13)
    box: dict = {}
    rowsA: list = []
    t1 = time.perf_counter()
    ts, rep = TR.train_month(seed=SEED, days=days, net_params=net, tcfg=tcfg, box=box,
                             on_epoch=_array_epoch_probe(box, rowsA))
    t_arr = time.perf_counter() - t1

    cmp = _cmp_boundaries(rt5, rowsA)
    r5 = rt5.report()
    REPORT["(B) 얼린 굴리기"] = dict(net=tag, seconds=seconds, training=False, updates=len(ts.updates),
                                **cmp, v5_selects=len(rt5.selects),
                                v5_market_selects=len(rt5.market_selects),
                                array_role_counts=[int(x) for x in np.asarray(ts.st.role_counts)],
                                array_crane=[int(x) for x in np.asarray(ts.st.crane_actions)],
                                v5_s=round(t_v5, 1), array_s=round(t_arr, 1),
                                v5_report={k: r5[k] for k in
                                           ("intervals", "roles", "crane_actions", "cost_krw")},
                                array_report={k: rep[k] for k in
                                              ("intervals", "roles", "crane_actions", "cost_krw")})
    assert cmp["n"] >= 10, f"경계가 너무 적다 v5 {len(rt5.bounds)} 배열 {len(rowsA)}"
    #: ★v5 Tap 이 마지막 경계까지 기록하므로 개수가 같아야 한다 (try/finally 고침 전에는 120 대 121 이었다)
    assert cmp["v5_bounds"] == cmp["array_bounds"], (cmp["v5_bounds"], cmp["array_bounds"])
    assert cmp["first_bad"] is None, f"경계 Φ 가 처음 갈린 지점: {cmp['first_bad']}"
    assert cmp["first_count_bad"] is None, f"경계별 결정 계수기가 처음 갈린 지점: {cmp['first_count_bad']}"
    assert ts.updates == [], "얼린 굴리기인데 갱신이 일어났다"
    assert rep["intervals"] == r5["intervals"], (rep["intervals"], r5["intervals"])
    #: ★`roles` 는 **크레인 칸만** 본다 — v5 PPO 경로는 시장(seller/buyer)에게도 정책을 묻고
    #:  배열판에는 시장 다리가 없다 (조각 8 범위 밖 · `_v5_month_with_ppo` 머리말). 거래가 0 건이라
    #:  세계는 같지만 '정책에 물은 횟수' 는 그만큼 다르다 — 배열 보고는 그 자리를 `null` 로 남긴다.
    assert rep["roles"].get("crane", 0) == r5["roles"].get("crane", 0), (rep["roles"], r5["roles"])
    assert rep["roles"]["seller"] is None and rep["roles"]["buyer"] is None, rep["roles"]
    assert rep["market"] == "unported" and rep["traded_edges"] is None, rep["market"]
    assert rep["crane_actions"] == r5["crane_actions"], (rep["crane_actions"], r5["crane_actions"])
    # Compare the direct physical ledgers and normalized reward, not only KRW diagnostics.
    from yard_rl.v6.reward.operational import KEYS
    np.testing.assert_allclose([rep['physical_totals'][k] for k in KEYS],
                               [r5['physical_totals'][k] for k in KEYS], rtol=1e-10)
    assert rep['team_reward'] == pytest.approx(r5['team_reward'], rel=1e-10)
    assert r5["roles"].get("seller", 0) > 0, "v5 가 판매자에게 묻지 않았다 — 대조 전제가 바뀌었다"


# ═══════════════════════════════════════════════ (C) 30일 — 날마다 블록 해시·Φ
@pytest.mark.skipif("C" in _SKIP, reason="TRAIN_EQUIV_SKIP")
def test_C_frozen_rollout_prefix_until_v5_stops():
    """★C층 — **얼린 정책 굴리기**를 v5 가 스스로 멈출 때까지 (접두사 대조 · 하루~13.5시간).

    ⚠️ 이 층은 **'30일 무대 대조' 가 아니다**. 이름 그대로 *v5 가 멈춘 시각까지의 접두사* 다:
      ① v5 `run_month(ppo=…)` **기본(MonthTerminal) 갈래**는 트럭 하나라도 못 들어오면
         `ContainerContractError` 로 스스로 멈춘다 (month_run.py:519-523). 실측(2026-09-27 · 실제
         체크포인트): 부하 30·60·150·300·600 전부 첫날 안에 `NO_TARGET` 으로 멈춘다 (가장 멀리 가는
         부하 30 이 t=48,780초 = 13.5시간 · 경계 **814** · 크레인 결정 **1,083**).
         ★그러나 이것은 **갈래를 고른 탓**이다 — 고정 화물(`CargoTerminal`) 갈래에서는 v5 가 날 경계를
         넘겨 완주한다 (실측 부하 300·3일·추첨으로 t=90,000초 · 갱신 25회). 그 갈래는 배열판에 없다.
      ② 하루가 1,441 검토 경계이고 배열판은 경계마다 호스트로 돌아온다 (실측 CPU 0.39초/경계 정상상태 ·
         21블록) — 21블록·30일 전 규모 대조는 어느 쪽으로도 이 시험 안에서 불가능하다.
      ③ 날 경계를 **넘긴** 대조는 `tests/v6/test_gpu_month.py`(규칙 정책 · 2~6블록 · 2~3일)와
         아래 (F) 층(배열 단독 완주)이 나눠 받친다. 학습 정책망 + 21블록 + 날 넘김의 조합은 미검증이다.

    명시로 켠다: `TRAIN_EQUIV_FULL=1` 또는 `TRAIN_EQUIV_DAYS=<날수>`.
    """
    if not (os.environ.get("TRAIN_EQUIV_FULL") or os.environ.get("TRAIN_EQUIV_DAYS")):
        pytest.skip("C층은 명시로 켠다 — TRAIN_EQUIV_FULL=1 또는 TRAIN_EQUIV_DAYS=<날수>")
    n_days = int(os.environ.get("TRAIN_EQUIV_DAYS", "3"))
    load = int(os.environ.get("TRAIN_EQUIV_C_LOAD", "30"))   # v5 가 가장 멀리 가는 부하 (아래 ⚠️)
    sd, tag = _state_dict()
    days = _plan(n_days, load)
    t0 = time.perf_counter()
    rt5, stop = _v5_month_with_ppo(days, sd=sd, seconds=None, training=False, tolerate=True)
    t_v5 = time.perf_counter() - t0
    r5 = rt5.report()
    assert r5["traded_edges"] == 0, "v5 가 거래를 했다 — 배열판에 시장이 없어 대조 불가"
    assert rt5.bounds, "v5 경계가 하나도 없다"
    t_last = rt5.bounds[-1][0]
    n_cross = sum(1 for t, _c in rt5.bounds if t >= v5m.DAY_S - 1e-9)
    net = VN.load_v5_params(sd)
    tcfg = TR.TrainConfig(training=False, cmax=256, amax=13, stop_s=t_last)
    box: dict = {}
    rowsA: list = []
    dayA: list = []
    t1 = time.perf_counter()
    ts, rep = TR.train_month(seed=SEED, days=days, net_params=net, tcfg=tcfg, box=box,
                             on_epoch=_array_epoch_probe(box, rowsA),
                             on_boundary=lambda run, d, t: dayA.append((int(d), float(t))))
    t_arr = time.perf_counter() - t1
    cmp = _cmp_boundaries(rt5, rowsA)
    REPORT["(C) 얼린 접두사"] = dict(net=tag, days=n_days, load=load, v5_stop=stop, training=False,
                                 v5_last_t=t_last, crossed_day1=n_cross, **cmp,
                                 array_day_boundaries=dayA,
                                 v5_s=round(t_v5, 1), array_s=round(t_arr, 1),
                                 v5_crane=r5["roles"].get("crane", 0),
                                 array_crane=rep["roles"].get("crane", 0),
                                 v5_crane_actions=r5["crane_actions"],
                                 array_crane_actions=rep["crane_actions"])
    #: ★규모는 '경계 수를 세는' 것이 아니라 **v5 가 간 곳까지 배열이 전부 따라왔는가** 로 단언한다
    #:  (전에는 `n >= 500` 이라 배열이 일찍 멈춰 규모가 줄어도 통과했다).
    assert cmp["v5_bounds"] == cmp["array_bounds"] == cmp["n"], (
        f"경계 수가 다르다 v5 {cmp['v5_bounds']} 배열 {cmp['array_bounds']} — v5 는 {t_last}s 에 멈췄다 ({stop})")
    assert cmp["n"] >= 500, f"대조한 경계가 {cmp['n']} 개뿐이다 (v5 정지 {stop})"
    assert cmp["first_bad"] is None, f"경계 Φ 가 처음 갈린 지점: {cmp['first_bad']}"
    assert cmp["first_count_bad"] is None, f"경계별 결정 계수기가 처음 갈린 지점: {cmp['first_count_bad']}"
    #: 마지막 경계의 계수기 사진도 한 번 더 (보고용 · 위 경계별 대조가 이미 이것을 포함한다)
    _t5, roles5, acts5 = rt5.snaps[cmp["n"] - 1]
    REPORT["(C) 얼린 접두사"]["v5_snapshot"] = dict(t=_t5, roles=roles5, crane_actions=acts5)
    assert rep["roles"].get("crane", 0) == roles5.get("crane", 0), (rep["roles"], roles5)
    assert rep["crane_actions"] == acts5, (rep["crane_actions"], acts5)


# ═══════════════════════════════════════════════ (D) ★학습 켠 닫힌 고리 (수집→갱신→갱신된 망)
@pytest.mark.skipif("D" in _SKIP, reason="TRAIN_EQUIV_SKIP")
def test_D_closed_loop_with_updates_matches_v5():
    """★D층 — **학습을 켜고** 같은 무대를 v5·배열로 돌려 경계마다 맞댄다.

    이 층이 없으면 조각 8 의 본체('수집 → 버퍼 → GAE → 갱신 → **갱신된 망으로 다음 수집**')가
    한 번도 대조되지 않는다. (B)(C) 는 `training=False` 라 갱신이 0회, (A) 는 v5 가 모아 준 테이프로
    갱신 산술만 본다.

    ⚠️ **대조 전용 무대다** — v5 쪽 시장(판매자·구매자) 결정을 학습 버퍼에서 떼어냈다
       (`detach_market=True`). 떼어내지 않으면 v5 버퍼에 판매자 Choice 가 들어가 갱신이 달라진다:
       실측으로 v5 첫 갱신 micro_actions 147 = 크레인 81 + 판매자 66 (배열 81) → **표본의 44.9%** 차이.
       세계에 주는 답은 v5 그대로이므로 '같은 세계' 는 유지된다.
    갱신을 여러 번 밟으려고 `rollout_intervals` 를 줄인다 (v5 `PPOConfig` 의 정당한 손잡이다) —
    경계 40 · 구간 10 이면 갱신 4회다. 난수열(미니배치 순열)은 양쪽 모두 `default_rng(SEED)` 다.
    """
    import numpy  # noqa: F401
    from yard_rl.v6.ppo.runtime import PPOConfig
    seconds = float(os.environ.get("TRAIN_EQUIV_D_SECONDS", "2400"))
    rollout = int(os.environ.get("TRAIN_EQUIV_D_ROLLOUT", "10"))
    load = int(os.environ.get("TRAIN_EQUIV_D_LOAD", "3500"))
    sd, tag = _state_dict()
    days = _plan(2, load)
    config = PPOConfig(rollout_intervals=rollout)

    t0 = time.perf_counter()
    rt5 = _v5_month_with_ppo(days, sd=sd, seconds=seconds, training=True, config=config,
                             detach_market=True)
    t_v5 = time.perf_counter() - t0
    r5 = rt5.report()
    assert r5["traded_edges"] == 0, "v5 가 거래를 했다 — 배열판에 시장이 없어 대조 불가"
    assert len(rt5.updates) >= 2, f"v5 갱신이 {len(rt5.updates)}회뿐 — 초·구간 수를 조절하라"
    assert rt5.market_selects, "v5 가 시장에 묻지 않았다 — detach 대조의 전제가 바뀌었다"
    after5 = {k: v.detach().cpu().numpy().copy() for k, v in rt5.policy.state_dict().items()}

    net = VN.load_v5_params(sd)
    tcfg = TR.TrainConfig.from_v5(config, n_blocks=21, cmax=256, amax=13, training=True,
                                  stop_s=seconds)
    box: dict = {}
    rowsA: list = []
    decA: list = []
    upA: list = []
    probe = _array_epoch_probe(box, rowsA)

    def on_epoch(run, e, t, codes, out):
        probe(run, e, t, codes, out)
        if bool(out.advanced):
            decA.extend(_decisions_of_tape(out.interval.choices))

    t1 = time.perf_counter()
    ts, rep = TR.train_month(seed=SEED, days=days, net_params=net, tcfg=tcfg, box=box,
                             on_epoch=on_epoch, on_update=lambda r: upA.append(dict(r)))
    t_arr = time.perf_counter() - t1

    cmp = _cmp_boundaries(rt5, rowsA)
    dec = _cmp_decisions(rt5.selects, decA)
    ups = []
    for j in range(min(len(rt5.updates), len(upA))):
        u5, ua = dict(rt5.updates[j]), upA[j]
        ups.append(dict(index=j + 1,
                        intervals=(u5["intervals"], ua["intervals"]),
                        minibatches=(u5["minibatches"], ua["minibatches"]),
                        early=(u5["early_stopped"], ua["early_stopped"]),
                        micro=(u5["micro_actions"], ua["micro_actions"]),
                        kl_abs=abs(u5["max_kl"] - ua["max_kl"]),
                        loss_rel=abs(u5["loss"] - ua["loss"]) / max(1e-30, abs(u5["loss"]))))
    arr_sd = VN.to_v5_state_dict(ts.params.net)
    worst_param = {k: float(np.max(np.abs(arr_sd[k] - v))) for k, v in after5.items()}
    REPORT["(D) 닫힌 고리"] = dict(net=tag, seconds=seconds, rollout_intervals=rollout, load=load,
                                training=True, detached_market_choices=rt5.detached,
                                v5_updates=len(rt5.updates), array_updates=len(upA), **cmp,
                                decisions=dec, updates=ups,
                                worst_param={k: round(v, 12) for k, v in worst_param.items()},
                                v5_s=round(t_v5, 1), array_s=round(t_arr, 1))

    assert cmp["v5_bounds"] == cmp["array_bounds"], (cmp["v5_bounds"], cmp["array_bounds"])
    assert cmp["first_bad"] is None, f"경계 Φ 가 처음 갈린 지점: {cmp['first_bad']}"
    assert cmp["first_count_bad"] is None, f"경계별 결정 계수기가 처음 갈린 지점: {cmp['first_count_bad']}"
    assert dec["n_bad"] == 0, f"결정 낱개가 갈렸다 ({dec['n_bad']}건): {dec['first_bad']}"
    assert dec["array"] > 0 and dec["v5"] == dec["array"], (dec["v5"], dec["array"])
    assert len(upA) == len(rt5.updates), (len(upA), len(rt5.updates))
    for u in ups:
        assert u["intervals"][0] == u["intervals"][1], u
        assert u["minibatches"][0] == u["minibatches"][1], u
        assert u["early"][0] == u["early"][1], u
        #: ★여기가 시장 누락이 드러나던 자리다 — 크레인만 담기면 양쪽 micro_actions 가 같아야 한다
        assert u["micro"][0] == u["micro"][1], u
        assert u["kl_abs"] < 2e-6, u
        assert u["loss_rel"] < 1e-3, u
    for k, v in worst_param.items():
        if k == "actor.bias":            # 식별 불가 칸 ((A) 층 머리말)
            continue
        assert v < 1e-4, f"{k} 최종 파라미터 차 {v:.3g} >= 1e-4 (갱신 {len(upA)}회 뒤)"


# ═══════════════════════════════════════════════ (E) 추첨(sample) 수집 — 분포·재현성·자기일관성
@pytest.mark.skipif("E" in _SKIP, reason="TRAIN_EQUIV_SKIP")
def test_E_sampling_is_reproducible_and_self_consistent():
    """★E층 — 추첨 수집(`TrainConfig.sample_actions`) 이 **돌고 · 재현되고 · 기록이 자기일관적**인가.

    v5 와 같은 **표본**은 원리상 낼 수 없다 (`torch.multinomial` 난수열 · 규칙 ⑥). 그래서 보는 것은:
      ① 분포 — `VN.sample_action` 의 빈도가 그 확률과 맞는가 (큰 표본)
      ② 재현성 — 같은 `sample_seed` 면 런이 같고, 다른 씨면 달라진다
      ③ 자기일관성 — 기록된 `log_prob` 이 `log_probs(scores, mask)[action]` 과 **비트 일치**
      ④ 탐색이 실제로 있나 — 최고점 수집과 결정이 달라지는 자리가 있는가
    (v5 와의 분포 일치 자체는 `tests/v6/test_gpu_v5net.py::test_distribution_matches_torch_categorical`
     이 torch 분포와 맞대 이미 지킨다.)
    """
    from yard_rl.v6.gpu import v5net as VN2
    # ── (1) 분포: 고정 점수·마스크에서 20,000 번 뽑아 확률과 맞춘다
    rng = np.random.default_rng(777)
    scores = jnp.asarray(rng.normal(size=(7,)) * 1.5, F)
    mask = jnp.asarray([True, True, False, True, True, False, True])
    pr = np.asarray(VN2.probs(scores, mask))
    keys = jax.random.split(jax.random.PRNGKey(20_260_927), 20_000)
    drawn = np.asarray(jax.vmap(lambda k: VN2.sample_action(k, scores, mask))(keys))
    assert set(np.unique(drawn)) <= {0, 1, 3, 4, 6}, f"마스크 밖을 뽑았다 {np.unique(drawn)}"
    freq = np.bincount(drawn, minlength=7) / len(drawn)
    worst_z = 0.0
    for i in np.where(pr > 0)[0]:
        sd_i = float(np.sqrt(pr[i] * (1 - pr[i]) / len(drawn)))
        worst_z = max(worst_z, abs(freq[i] - pr[i]) / max(sd_i, 1e-12))
    assert worst_z < 5.0, f"빈도가 확률과 어긋난다 (최대 {worst_z:.2f} 시그마)"

    # ── (2)(3)(4) 학습 루프에서 — **갱신 전에 끊는다** (`e1` 로 20 에폭만; `stop_s` 로 끊으면 v5 규약대로
    #    그 경계에서 갱신이 한 번 일어나 `runtime.py:198` 망이 바뀌고 로그확률 되짚기가 성립하지 않는다)
    sd, tag = _state_dict()
    net = VN.load_v5_params(sd)
    days = _plan(2, int(os.environ.get("TRAIN_EQUIV_E_LOAD", "3500")))
    n_ep = int(os.environ.get("TRAIN_EQUIV_E_EPOCHS", "40"))
    got = {}
    for label, seed_s, sample in (("s1", 1234, True), ("s1b", 1234, True), ("s2", 5678, True),
                                  ("greedy", None, False)):
        tcfg = TR.TrainConfig(training=True, cmax=256, amax=13,
                              sample_actions=sample, sample_seed=seed_s)
        box: dict = {}
        rows: list = []
        dec: list = []
        tapes: list = []
        probe = _array_epoch_probe(box, rows)

        def on_epoch(run, e, t, codes, out, _d=dec, _p=probe, _t=tapes):
            _p(run, e, t, codes, out)
            if bool(out.advanced):
                _d.extend(_decisions_of_tape(out.interval.choices))
                _t.append(out.interval.choices)

        ts, rep = TR.train_month(seed=SEED, days=days, net_params=net, tcfg=tcfg, box=box,
                                 blocks=("Y01", "Y04"), cap_moves=24, e1=n_ep, on_epoch=on_epoch)
        got[label] = dict(rows=rows, dec=dec, rep=rep, ts=ts, tapes=tapes,
                          mode=rep.get("action_mode"))
    assert got["s1"]["mode"] == "sample-all-days" and got["greedy"]["mode"] == "argmax"
    assert got["s1"]["dec"], "짧은 무대에서 결정이 하나도 없었다 — 대조가 공허하다"
    assert not got["s1"]["ts"].updates, "갱신이 일어났다 — 로그확률 되짚기가 성립하지 않는다"
    # (2) 재현성
    assert got["s1"]["dec"] == got["s1b"]["dec"], "같은 씨인데 결정이 달라졌다"
    assert [r["cost"] for r in got["s1"]["rows"]] == [r["cost"] for r in got["s1b"]["rows"]]
    diff_seed = sum(1 for a, b in zip(got["s1"]["dec"], got["s2"]["dec"]) if a != b)
    diff_greedy = sum(1 for a, b in zip(got["s1"]["dec"], got["greedy"]["dec"]) if a != b)
    # (3) 자기일관성 — 기록된 로그확률이 기록된 행렬·마스크·행동의 로그확률과 같은가
    checked, worst = 0, 0.0
    for label in ("s1", "greedy"):
        for pend in got[label]["tapes"]:
            n = np.asarray(pend.n)
            for b in range(n.shape[0]):
                for c in range(int(n[b])):
                    a = int(np.asarray(pend.action)[b, c])
                    rows_bc = jnp.asarray(np.asarray(pend.rows)[b, c], F)
                    mask_bc = jnp.asarray(np.asarray(pend.mask)[b, c])
                    lp = float(VN2.log_prob_of(VN2.actor_scores(net, rows_bc), mask_bc, a))
                    worst = max(worst, abs(lp - float(np.asarray(pend.log_prob)[b, c])))
                    checked += 1
    REPORT["(E) 추첨"] = dict(net=tag, worst_freq_sigma=round(worst_z, 2),
                             decisions=len(got["s1"]["dec"]),
                             same_seed_identical=True, diff_seed_changed=diff_seed,
                             vs_greedy_changed=diff_greedy, logprob_checked=checked,
                             logprob_worst_abs=worst,
                             sample_report={k: got["s1"]["rep"].get(k)
                                            for k in ("action_mode", "sample_seed", "market")})
    assert checked > 0 and worst == 0.0, f"기록된 로그확률이 되짚은 값과 다르다 (최대 {worst:.3g})"
    assert diff_seed > 0 or diff_greedy > 0, "추첨이 결정을 바꾸지 않았다 — 탐색이 없다"


# ═══════════════════════════════════════════════ (F) 배열 단독 완주 — 날 경계·finish·마지막 갱신
@pytest.mark.skipif("F" in _SKIP, reason="TRAIN_EQUIV_SKIP")
def test_F_array_alone_crosses_a_day_and_finishes():
    """★F층 — 배열 학습 루프가 **날 경계를 넘고 `finish` 까지** 간다 (v5 대조 없이 자기 일관성).

    왜 v5 대조가 아닌가: v5 PPO 경로(기본 갈래)는 첫날 안에 `ContainerContractError` 로 멈춘다
    ((C) 층 머리말). 그래서 '날 경계를 넘긴 학습 루프' 는 **배열 단독으로**만 밟을 수 있다.
    여기서 처음 실행되는 갈래: `open_day(day=1)`(어제 야드 위에 오늘 배) · `month.close_day_at` 2회 ·
    `MB.finish_run_jit`(배수 구간) · `PR.finish` · 마지막 갱신 · 날별 확정 Φ.
    작은 무대(2블록·부하 60·2일)로 잡는다 — 그래도 2,881 검토 경계다.
    """
    sd, tag = _state_dict()
    net = VN.load_v5_params(sd)
    days = _plan(2, int(os.environ.get("TRAIN_EQUIV_F_LOAD", "60")))
    tcfg = TR.TrainConfig(training=True, cmax=256, amax=13, sample_actions=False)
    dayA: list = []
    upA: list = []
    t0 = time.perf_counter()
    ts, rep = TR.train_month(seed=SEED, days=days, net_params=net, tcfg=tcfg,
                             blocks=("Y01", "Y04"), cap_moves=24,
                             on_boundary=lambda run, d, t: dayA.append((int(d), round(float(t), 3))),
                             on_update=lambda r: upA.append(dict(r)))
    wall = time.perf_counter() - t0
    m = rep["month"]
    REPORT["(F) 단독 완주"] = dict(net=tag, epochs=m["n_epochs"], day_boundaries=dayA,
                                updates=len(upA), intervals=rep["intervals"],
                                flags=list(rep["flags"]), violation=rep["violation"],
                                exhausted=rep["exhausted"], truncated=rep["truncated"],
                                days=[{k: d[k] for k in ("index", "provisional", "phi_krw",
                                                         "n_trucks")} for d in m["days"]],
                                admitted=m["admitted"], skipped=m["skipped"],
                                cost_krw=rep["cost_krw"], wall_s=round(wall, 1))
    assert dayA == [(0, 0.0), (1, 86400.0)], f"날 경계가 다르다 {dayA}"
    assert len(m["days"]) == 2 and all(not d["provisional"] for d in m["days"]), m["days"]
    assert not rep["flags"], f"위반 비트가 켜졌다 {rep['flags']}"
    assert rep["violation"] == 0 and rep["exhausted"] == 0, (rep["violation"], rep["exhausted"])
    assert len(upA) >= 2, f"갱신이 {len(upA)}회뿐이다 — 마지막 갱신 갈래가 안 돌았다"
    assert m["n_epochs"] >= 2881, m["n_epochs"]


# ═══════════════════════════════════════════════ (S) 접합부 — 배열이 모은 구간이 갱신에 닿는 길
@pytest.mark.skipif("S" in _SKIP, reason="TRAIN_EQUIV_SKIP")
def test_S_intervals_to_batch_matches_pack_intervals():
    """★S층 — `train.intervals_to_batch`(배열이 스스로 모은 `IntervalRow`) 가
    `ppo_buffer.pack_intervals`(v5 `Interval`) 와 **같은 배치**를 내는가.

    이 접합부는 어떤 시험도 부르지 않았다 ((A) 층은 v5 테이프를 `pack_intervals` 로 담는다). 두 모듈
    담당이 '칸 이름이 세 곳 다르다'(n<->n_choices · time_s<->choice_time_s · n_actions<->n_cands)고
    명시하며 통합자에게 넘긴 자리다. 같은 v5 테이프를 두 길로 담아 잎·dtype·GAE 까지 맞댄다.
    """
    import test_gpu_ppo_runtime as R
    load = int(os.environ.get("TRAIN_EQUIV_S_LOAD", "300"))
    dur = float(os.environ.get("TRAIN_EQUIV_S_DUR", "21600"))
    with pytest.MonkeyPatch.context() as mp:
        R._fixed_cargo_patch(mp)
        stage = R._run_v5(load=load, duration_s=dur, snap_s=3600.0)
    cfg = R._cfg_of(stage)
    iv5 = stage["intervals"]
    brows = [e for k, e in stage["events"] if k == "B"]
    costs = np.asarray(R._array_phi(brows).total)

    st = PR.new_state(cfg)
    rows: list = []
    bi = 0
    for kind, ev in stage["events"]:
        if kind == "S":
            st, _o = PR.select_record(st, cfg, role=R.ROLE_CODE[ev["role"]], block=ev["block"],
                                      t=ev["t"], rows=R._pad_rows(ev["x"]),
                                      mask=R._pad_mask(ev["mask"]), action=ev["action"],
                                      log_prob=ev["log_prob"], n_actions=ev["n"])
            continue
        cost = jnp.asarray(costs[bi], F)
        bi += 1
        if ev["final"]:
            st, out = PR.finish(st, cfg, jnp.asarray(ev["t"], F), cost,
                                jnp.asarray(ev["states"], F), jnp.asarray(ev["values"], F),
                                terminated=ev["terminated"])
        else:
            st, out = PR.boundary(st, cfg, jnp.asarray(ev["t"], F), cost,
                                  jnp.asarray(ev["states"], F), jnp.asarray(ev["values"], F),
                                  terminated=ev["terminated"], final=False)
        if bool(out.advanced) and bool(out.interval.valid):
            rows.append(out.interval)
        if not bool(out.ignored):
            st = PR.refresh_values(st, jnp.asarray(ev["values"], F))
    assert len(rows) == len(iv5), f"구간 수가 다르다 배열 {len(rows)} vs v5 {len(iv5)}"

    b5, over = PB.pack_intervals(iv5, c_max=cfg.cmax, a_max=cfg.amax)
    assert not over.any, f"칸이 모자라 결정을 못 담았다 {over}"
    ba = TR.intervals_to_batch(rows)
    #: ★`rows` 잎은 **패딩 칸을 지우지 않는다** (`PendingTape` 머리말 — 경계마다 (B,cmax,amax,37) 을
    #:  0 으로 다시 쓰면 11MB/경계다). 그래서 잎 전체 비트 대조에서 그 하나만 뺀다. 대신 아래에서
    #:  **쓰는 칸**(`n_choices`·`n_cands` 안쪽)을 낱개로 맞댄다 — 학습이 읽는 칸은 그것뿐이다
    #:  (빈 칸의 `mask` 는 전부 거짓이라 후보가 없다).
    bad = [b for b in _tree_diff(ba, b5) if b[0] != ".rows"]
    nch, nc = np.asarray(b5.n_choices), np.asarray(b5.n_cands)
    ra, r5 = np.asarray(ba.rows), np.asarray(b5.rows)
    ma, m5 = np.asarray(ba.mask), np.asarray(b5.mask)
    used_bad, cells = [], 0
    for i in range(nch.shape[0]):
        for b in range(nch.shape[1]):
            for c in range(int(nch[i, b])):
                k = min(int(nc[i, b, c]), cfg.amax)
                cells += k
                if not np.array_equal(ra[i, b, c, :k], r5[i, b, c, :k]):
                    used_bad.append(("rows", i, b, c))
                if not np.array_equal(ma[i, b, c, :k], m5[i, b, c, :k]):
                    used_bad.append(("mask", i, b, c))
    adv_a, ret_a = PB.gae(ba, brows[-1]["values"], gamma=0.999, lam=0.95, time_unit_s=60.0)
    adv_5, ret_5 = PB.gae(b5, brows[-1]["values"], gamma=0.999, lam=0.95, time_unit_s=60.0)
    gae_same = bool(np.array_equal(np.asarray(adv_a), np.asarray(adv_5))
                    and np.array_equal(np.asarray(ret_a), np.asarray(ret_5)))
    REPORT["(S) 접합부"] = dict(load=load, duration_s=dur, intervals=len(rows),
                              blocks=int(nch.shape[1]), cmax=cfg.cmax, amax=cfg.amax,
                              choices=int(nch.sum()), compared_cand_rows=cells,
                              leaf_diff=[b[0] for b in bad], used_diff=used_bad[:5],
                              gae_identical=gae_same,
                              values_dtype=str(np.asarray(ba.values).dtype))
    assert not used_bad, f"쓰는 칸이 갈렸다: {used_bad[:5]}"
    assert not bad, f"잎이 갈렸다: {[b[0] for b in bad][:4]}"
    assert gae_same, "GAE 출력이 갈렸다"
    assert int(nch.sum()) > 0, "결정이 하나도 없었다 — 대조가 공허하다"


# ═══════════════════════════════════════════════ 보고
def test_zz_report(capsys):
    with capsys.disabled():
        print("\n■ 조각 8 통합 동등성 보고")
        for k, v in REPORT.items():
            print(f"  {k}: {json.dumps(v, ensure_ascii=False, default=str)[:900]}")
    out = os.path.join(_ROOT, "outputs", "v6", "verify")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "train_equiv_report.json")
    #: ★**합쳐서** 쓴다 — 층을 골라 돌린 실행이 다른 층의 증거를 지우면 안 된다 (검증 반박: "(C) 가
    #:  없다 — 뒤에 A·B 실행이 덮어썼다"). 층마다 잰 시각을 붙여 언제 것인지 드러낸다.
    old: dict = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                old = json.load(f)
        except Exception:
            old = {}
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    merged = {k: v for k, v in old.items() if k != "_meta"}
    for k, v in REPORT.items():
        merged[k] = dict(v, measured_at=stamp) if isinstance(v, dict) else v
    meta = dict(old.get("_meta") or {})
    meta[stamp] = dict(layers=sorted(REPORT), backend=jax.default_backend(),
                       skipped=sorted(_SKIP), full=bool(os.environ.get("TRAIN_EQUIV_FULL")))
    merged["_meta"] = meta
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1, default=str)
