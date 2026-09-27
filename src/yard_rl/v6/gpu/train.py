"""조각 8 통합 — **배열 세계 학습 드라이버** ([[YR-327]]).

v5 정본 대응: `ppo/continuous.py:main` → `stage/month_run.run_month(ppo=PPORuntime(...))`.
이 파일은 조각 1~7 과 조각 8 모듈 넷(`ppo_runtime` · `ppo_buffer` · `ppo_update` · `month`)을
**하나의 학습 루프**로 엮는다. 새 물리·새 산술은 하나도 없다 — 배선과 기록만 한다.

■ 무엇이 어디서 오나 (재구현 금지 규약)
    세계 한 스텝        `engine_step.step` (조각 1~4)
    블록 묶음·검토 시각  `multiblock.run_to_epoch` · `review_epoch` (조각 6)
    30일 무대           `month.to_month_world` · `open_day` · `no_target_skips` · `MonthTape` (조각 8 모듈)
    후보·마스크·순차조건부 `cands3` · `v5cond.sequential_conditional` (조각 3·7)
    특징 37칸·블록 요약  `v5feat` (조각 7)
    망 순전파·행동        `v5net` (조각 7)
    Φ(비용)             `phi.terminal_cost_krw` (조각 5)
    학습 회계(경계·구간)  `ppo_runtime` (조각 8 모듈)
    버퍼·GAE            `ppo_buffer` (조각 8 모듈)
    PPO 갱신            `ppo_update` (조각 8 모듈)

■ ★기록 채널 — 학습은 '무엇을 골랐나' 만으로는 못 한다
  갱신에는 결정마다 **후보 행렬 (A,37) · 마스크 (A,) · 고른 번호 · 그때의 로그확률**이 필요하다.
  그 값들은 `policy_fn` 안(jit 안)에서 만들어지고 사라진다. 그래서 조각 8 은 `engine_step` 에
  **선택형 기록 채널**을 냈다 (`DecideOut.rec` · `StepTrace.rec`, 기본 `None` = 빈 pytree):
      · 정책이 네 값 `(choice, lost, flags, rec)` 을 돌려주면 그 `rec` 가 `StepTrace.rec` 로 나온다.
      · `multiblock.run_to_epoch(tape=…, tape_fn=…)` 이 while_loop 캐리에 테이프를 함께 실어
        스텝마다 쌓는다 → 에폭이 끝나면 `PendingTape` (B,cmax,…) 한 벌이 손에 있다.
      · 기록을 안 쓰는 경로(조각 1~7 시험 692건)는 `rec=None` 이라 **잎이 하나도 없다** —
        계산도 메모리도 늘지 않는다.

■ 동등성 축 (조각 7 규약 그대로) · 탐색
  **대조**는 최고점(argmax)에서만 한다 — 추첨(`torch.multinomial`)은 난수열을 재현할 수 없어 v5 와
  같은 표본을 낼 수 없다 (규칙 ⑥). 그러나 **학습에는 탐색이 필요하다**: v5 `PPORuntime` 은
  `sample_actions = bool(training)` 이고 v5 정본 드라이버는 매니페스트에 `action_mode:
  "sample-all-days"` 를 박아 둔다 (`ppo/continuous.py:62`). 그래서 배열판도 `TrainConfig.sample_actions`
  로 **자기 난수(jax.random)** 추첨 갈래를 낸다 (키 규약: `dispatch.V5NetParams.sample_key`).
  · 추첨 갈래의 검증은 '표본 일치' 가 아니라 **분포 수준**이다 — 같은 상태에서 로그확률·엔트로피·확률이
    v5 와 같은가(`tests/v6/test_gpu_v5net.py`) + 뽑은 빈도가 그 확률과 맞는가(카이제곱) + 같은 씨면
    런이 재현되나 (`tests/v6/test_gpu_train_equiv.py` (E) 층).
  · 그러므로 **`sample_actions=False` 로 잰 동등성은 '같은 알고리즘 설정의 동등성' 이 아니다** —
    v5 의 실제 학습 설정은 추첨이고, 최고점 수집은 이 저장소가 이미 실패로 판정한 설정이다
    ([[YR-324]] "최고점 선택이 정책을 망친다"). 보고에는 `action_mode` 를 반드시 싣는다.

■ 알려진 한계 (조각 8 통합 실측 · 2026-09-27 수정 단계에서 갱신)
  ① 정책망 층은 **비트 일치가 불가능**하다 (torch tanh·MKL GEMM vs XLA — 우리 코드 밖).
     조각 7 실측 갈림 확률 4.7e-06/결정. 30일 대조는 '해시 일치' 가 아니라 '첫 갈린 지점까지의
     접두사 일치' 로 판정한다.
  ② `run_month` 의 **`CargoTerminal` 갈래(`seed_data` 를 줄 때, `stage/month_run.py:276-281`)는
     이번 범위 밖**이다 (전역 큐 4단 키·원격 본선 인계·끝의 재고 등식). 기본 `MonthTerminal`
     갈래만 옮겼고, `month.to_month_world(seed_data=…)` 는 `MonthScopeError` 로 거절한다.
     ⚠️ ★**연구선이 쓰는 갈래가 바로 그 갈래다** — `ppo/workload_experiment.py:74-97` 은 시드 묶음이
     없어도 `build_fixed_seed` 로 문서를 만들어 **언제나** `seed_data` 를 넘긴다. 즉 **배열 학습은
     지금 연구선 설정(고정 화물)으로는 실행할 수 없다**. 반대로 기본 갈래는 v5 자신의 컨테이너 계약
     게이트(`stage/container_contract.require_container_plan`)가 막는다 — 대조 시험이 그 게이트를
     몽키패치로 지나가는 이유이고, **지나간 세계는 v5 가 학습을 거부하는 입력**이다 (고정 화물 갈래에서는
     같은 게이트가 통과한다 — 실측 부하 60·300 모두 `passed=True`).
  ③ 배수 구간(마지막 날 뒤 2시간)에는 검토 경계가 없어 한 구간이 7,200초가 되고 결정이 140건까지
     쌓인다 — `cfg.cmax` 를 그만큼 잡아야 한다 (`F_PENDING_OVERFLOW` 로 크게 실패한다).
  ④ 경계마다 Φ·본선 유휴를 **호스트**에서 읽는다 (`month_vessel_idle` 은 파이썬 사전 순서를 지켜야
     비트가 맞는다). 30일 43,201 경계에서 이것이 속도 벽이다 — 손익분기 측정은 기록·갱신을 끈
     굴리기 경로(`scripts/v6/bench_breakeven.py`)로 따로 한다.
  ⑤ 시장(`stage/market.py`·`bridge.py`)은 옮기지 않았다 — v5 `arm='NO_REALLOC'` 과 같은 세계다.
     `report()` 의 시장 계수기 네 칸과 `roles['seller']`·`roles['buyer']` 는 **`None`**(묻지 않았다)이고
     `market: "unported"` 가 붙는다 (0 = '안 팔기를 골랐다' 와 구별한다).
     ⚠️ ★이것이 **학습 동등성의 구멍**이다 (실측): v5 PPO 경로는 `month_run.py:301` 이 판매자·구매자
     선택자까지 `ppo.select` 에 물려 **그 결정을 학습 버퍼에 담는다**. 학습을 켜고 같은 무대를 돌리면
     v5 첫 갱신의 micro_actions 147 = 크레인 81 + 판매자 66 인데 배열은 81(크레인만)이다 —
     **학습 표본의 44.9% 가 빠진다**. 그래서 갱신 4 경계 뒤(t=3,900s)부터 결정이 갈린다.
     대조를 성립시키려면 v5 쪽 시장 선택자를 버퍼에서 떼어낸 **대조 전용 무대**가 필요하다
     (`tests/v6/test_gpu_train_equiv.py` (D) 층이 그것이다). 시장 이식 전에는
     "v6 로 30일 학습을 재현할 수 있다" 고 쓰지 않는다.
  ⑥ 위반 비트는 호스트로 돌아온 자리에서 **크게 실패한다** (`PR.raise_on_flags`) — 결정 칸 넘침
     (`F_PENDING_OVERFLOW`)·마스크 빈 칸 등이 조용히 지나가면 잘린 테이프로 학습하고도 '완료' 가 된다.
  ⑦ `e0>0` 인데 이어 돌릴 `box` 가 없으면 거절한다 — 프로세스를 넘겨 이어 돌리는 손잡이는 아직 없다.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from . import dispatch as DP
from . import engine_step as ES
from . import month as MO
from . import multiblock as MB
from . import phi as PH
from . import state as STATE
from . import ppo_buffer as PB
from . import ppo_runtime as PR
from . import ppo_update as PU
from . import v5cond as VC
from . import v5feat as VF
from . import v5net as VN
from .events import EMPTY_ID, EMPTY_TIME, TIME_DTYPE
from .geom import Geom
from .state import V_STEPS_EXHAUSTED

F = TIME_DTYPE
#: v5 `ppo/model.ROLES.index("crane")` — 크레인 결정의 역할 번호
ROLE_CRANE = PR.ROLES.index("crane")

#: ★`multiblock.Engine.params_axes` — 학습 정책 파라미터의 vmap 축 (망·end_s 는 공유, reserve_s 만 블록별)
PARAMS_AXES = DP.V5NetParams(net=None, end_s=None, reserve_s=0, sample_key=None)
#: 추첨 수집일 때 — `sample_key` 는 **블록마다 한 벌** (B,2) 이라 축 0 (`V5NetParams` 머리말)
PARAMS_AXES_SAMPLE = DP.V5NetParams(net=None, end_s=None, reserve_s=0, sample_key=0)

__all__ = ["PARAMS_AXES", "PARAMS_AXES_SAMPLE", "UNASKED_ROLES", "DecisionRec", "make_recording_policy", "TapeCarry", "new_tape_carry", "merge_rec",
           "make_tape_fn", "TrainConfig", "TrainState", "reserve_table", "block_rows_all",
           "states_values", "boundary_orders", "cost_at", "intervals_to_batch", "batch_to_tape",
           "apply_update", "new_train_state", "collect_epoch", "boundary_at", "month_setup",
           "train_month", "report"]


# ═══════════════════════════════════════════════ ① 기록 정책 (수집용)
class DecisionRec(NamedTuple):
    """결정 한 번의 **학습 기록** — 크레인마다 한 줄. v5 `ppo/buffer.Choice` 의 배열판.

    `open` 이 거짓인 크레인 줄은 v5 에 **아예 없다** (v5 는 `dp.crane_ids` 만 순회한다) — 테이프에
    담지 않는다. 빈 칸 규약은 `gpu/` 관례대로 정수 −1 · 시각 +inf · 마스크 False · 행렬 0.0.
    """

    open: jnp.ndarray       # (K,) bool   이번 결정에서 물은 크레인
    rows: jnp.ndarray       # (K,A,37) f64  `encode(rows, "crane")` — 행 i = candidate_id i
    mask: jnp.ndarray       # (K,A) bool  `ppo/crane.joint_mask`
    action: jnp.ndarray     # (K,) int32  고른 items 색인 (−1 = 안 물음)
    log_prob: jnp.ndarray   # (K,) f64    `dist.log_prob(action)`
    n_actions: jnp.ndarray  # (K,) int32  v5 `len(rows)` = items 길이
    kind: jnp.ndarray       # (K,) int32  고른 후보의 종류 PK_* (보고용 계수기)
    time_s: jnp.ndarray     # () f64      결정 시각 (`sim.now`)


def make_recording_policy(g: Geom, *, k_max: int = VC.K_MAX, crane_order=None, v5_cast: bool = True,
                          sample: bool = False):
    """`dispatch.make_v5net_policy` 와 **같은 결정**을 내면서 학습 기록까지 내는 정책.

    돌려주는 `policy_fn(params, world, c3, fl, pr, open_)` 은 **네 값**
    `(choice (K,), lost (K,), flags (), rec: DecisionRec)` 을 준다. `engine_step.decide_joint` 가
    네 번째를 `DecideOut.rec` 에 그대로 싣는다.

    ■ 기록을 어떻게 얻나 — **같은 순수 함수를 같은 입력으로 다시** 부른다 (규약: 재구현 금지)
      `sequential_conditional` 이 이미 `SeqOut.mask`(크레인 k 가 본 joint_mask)·`item`(고른 색인)·
      `prior`(직전 물은 크레인)·`choice`(최종 선택 열)를 준다. 남은 것은 (A,37) 후보 행렬과 로그확률뿐이고,
      둘 다 `VF.features` + `VN` 으로 **결정적으로** 다시 만들 수 있다.
      ★`out.choice[prior[k]]` 가 '단계 k 시점의 sel[prior_k]' 와 같은 이유: `prior_k < k` 이고 sel 의
        각 칸은 단계 k 에 **한 번** 써지므로, 최종 sel 의 그 칸은 이미 확정된 값이다.
      비용: `VF.features` 한 번이 더 돈다 (정책 비용의 10~15% — 90% 는 `joint_mask` 의 dry_run 이다).

    ■ `sample=True` — **추첨 수집** (탐색). v5 `sample_actions=True` 에 대응하고 v5 정본 드라이버
      (`ppo/continuous.py:62` `action_mode: "sample-all-days"`)가 쓰는 설정이다. 표본은 v5 와 다르다
      (`torch.multinomial` 난수열은 재현 불가 — 규칙 ⑥). 로그확률은 **고른 행동의 것**이라
      추첨이든 최고점이든 기록이 자기일관적이다. 키 규약은 `dispatch.V5NetParams.sample_key` 머리말.
    """
    bind = DP.make_v5net_pick(g, k_max=k_max, crane_order=crane_order, v5_cast=v5_cast,
                              sample=sample)
    I = VC.item_max(k_max)

    def policy_fn(params, world, c3, fl, pr, open_):
        K = world.k
        block = VF.block_row(world, g, end_s=params.end_s, reserve_s=params.reserve_s,
                             crane_order=crane_order)              # crane.py:70 — 결정당 한 번
        out = VC.sequential_conditional(world, fl, pr, open_, bind(params, world, fl, pr, block), g,
                                        k_max=k_max)

        def one(k):
            pk = out.prior[k]
            pk_i = jnp.clip(pk, 0, K - 1)
            pcol = jnp.where(pk >= 0, out.choice[pk_i], jnp.int32(EMPTY_ID))
            pkind, pbay = VF.prior_from_choice(fl, pk_i, pcol)
            fo = VF.features(world, g, fl, pr, block=block,
                             prior_kind=jnp.full((K,), pkind, jnp.int32),
                             prior_end_bay=jnp.full((K,), pbay, F), c_max=I, role="crane")
            x = fo.x[k]                                            # (I,37)
            xin = jnp.asarray(VF.as_net_input(x), F) if v5_cast else x
            a = out.item[k]
            lp = VN.log_prob_of(VN.actor_scores(params.net, xin), out.mask[k], a)
            col = out.choice[k]
            kind = jnp.where(col >= 0, fl.kind[k, jnp.clip(col, 0, fl.kind.shape[1] - 1)],
                             jnp.int32(EMPTY_ID)).astype(jnp.int32)
            return xin, out.mask[k], a, lp, fo.n_items[k], kind

        rows, mask, action, logp, n_act, kind = jax.vmap(one)(jnp.arange(K, dtype=jnp.int32))
        rec = DecisionRec(open=jnp.asarray(open_, bool), rows=rows, mask=mask,
                          action=action.astype(jnp.int32), log_prob=logp,
                          n_actions=n_act.astype(jnp.int32), kind=kind, time_s=world.clock)
        return out.choice, jnp.zeros((K,), bool), out.flags, rec

    def rec_zeros(world):
        """결정이 없던 스텝의 빈 기록 — `lax.cond` 두 갈래의 pytree 구조를 맞추려면 필요하다."""
        K = world.k
        return DecisionRec(open=jnp.zeros((K,), bool), rows=jnp.zeros((K, I, VF.INPUT_DIM), F),
                           mask=jnp.zeros((K, I), bool), action=jnp.full((K,), EMPTY_ID, jnp.int32),
                           log_prob=jnp.zeros((K,), F), n_actions=jnp.zeros((K,), jnp.int32),
                           kind=jnp.full((K,), EMPTY_ID, jnp.int32), time_s=world.clock)

    policy_fn.rec_zeros = rec_zeros
    policy_fn.amax = I
    policy_fn.sample = bool(sample)
    policy_fn.__name__ = "v5net_record" + ("_sample" if sample else "")
    return policy_fn


# ═══════════════════════════════════════════════ ② 테이프 합치기 (배치 append)
class TapeCarry(NamedTuple):
    """에폭 하나를 굴리는 동안 while_loop 캐리에 함께 타는 학습 테이프.

    `started`·`time_s`·`collecting` 은 **에폭 동안 상수**다 (구간 시작 시각으로 판정 — v5 는
    `collecting_at(self.time_s)` 를 쓴다, `ppo/runtime.py:137`).
    """

    pending: PR.PendingTape
    role_counts: jnp.ndarray     # (4,) int32
    crane_actions: jnp.ndarray   # (4,) int32
    flags: jnp.ndarray           # () int32
    started: jnp.ndarray         # () bool
    time_s: jnp.ndarray          # () f64   구간 시작 시각
    collecting: jnp.ndarray      # () bool  `collecting_at(cfg, time_s)`


def new_tape_carry(st: PR.RuntimeState, cfg: PR.RuntimeConfig) -> TapeCarry:
    return TapeCarry(pending=st.pending, role_counts=st.role_counts,
                     crane_actions=st.crane_actions, flags=jnp.zeros((), jnp.int32),
                     started=st.started, time_s=st.time_s,
                     collecting=PR.collecting_at(cfg, st.time_s))


_RAISE_BITS = (PR.F_DECISION_TIME_BAD | PR.F_DECISION_NO_BOUNDARY
               | PR.F_DECISION_EARLY | PR.F_MASK_EMPTY)


def merge_rec(c: TapeCarry, rec: DecisionRec, decided, active, cfg: PR.RuntimeConfig) -> TapeCarry:
    """스텝 하나의 결정 기록을 `PendingTape` 에 **크레인 순서대로** 쌓는다.

    `ppo_runtime.select_record` 를 블록축으로 한꺼번에 한 것이다 — 같은 검사·같은 비트·같은 계수기
    (시험 `test_merge_matches_select_record` 가 낱개 `select_record` 반복과 잎 비트까지 대조한다).
    `rec` 잎은 앞에 (B,) 가 붙어 있다 (vmap 결과). `decided`·`active` 는 (B,) bool.
    """
    B = int(c.pending.n.shape[0])
    A, D = int(cfg.amax), int(cfg.input_dim)
    K = int(rec.open.shape[1])
    bidx = jnp.arange(B, dtype=jnp.int32)
    ask0 = jnp.asarray(active, bool) & jnp.asarray(decided, bool)
    t = jnp.asarray(rec.time_s, F)
    bad_t = ~jnp.isfinite(t) | (t < 0)
    early = c.started & (t < c.time_s - PR.EPS_TIME)

    def body(cc: TapeCarry, k):
        ask = ask0 & rec.open[:, k]
        rows = jnp.asarray(rec.rows[:, k], F)
        mask = jnp.asarray(rec.mask[:, k], bool)
        nact = rec.n_actions[:, k].astype(jnp.int32)
        new = (jnp.where(ask & bad_t, PR.F_DECISION_TIME_BAD, 0)
               | jnp.where(ask & ~cc.started, PR.F_DECISION_NO_BOUNDARY, 0)
               | jnp.where(ask & early, PR.F_DECISION_EARLY, 0)
               | jnp.where(ask & ~jnp.any(mask, axis=-1), PR.F_MASK_EMPTY, 0)
               | jnp.where(ask & (nact > A), PR.F_ACTION_OVERFLOW, 0)).astype(jnp.int32)
        raised = (new & _RAISE_BITS) != 0
        coll = ask & cc.collecting & cc.started & ~raised
        p = cc.pending
        slot = p.n
        room = slot < cfg.cmax
        w = coll & room
        sl = jnp.minimum(slot, cfg.cmax - 1)

        def put(arr, val):
            cur = arr[bidx, sl]
            m = jnp.reshape(w, (B,) + (1,) * (cur.ndim - 1))
            return arr.at[bidx, sl].set(jnp.where(m, val, cur))

        p2 = p._replace(
            n=p.n + w.astype(jnp.int32),
            role=put(p.role, jnp.full((B,), ROLE_CRANE, jnp.int32)),
            time_s=put(p.time_s, t),
            rows=put(p.rows, jnp.where(jnp.reshape(w, (B, 1, 1)), rows,
                                       jnp.zeros((B, A, D), F))),
            mask=put(p.mask, mask),
            n_actions=put(p.n_actions, nact),
            action=put(p.action, rec.action[:, k].astype(jnp.int32)),
            log_prob=put(p.log_prob, jnp.asarray(rec.log_prob[:, k], F)),
            overflow=p.overflow + (coll & ~room).astype(jnp.int32))
        #: 칸이 모자라 못 담은 결정 — v5 에는 대응물이 없다 (파이썬 리스트라 상한이 없다).
        #:  조용히 버리지 않고 `select_record` 와 **같은 비트**로 크게 알린다.
        new = new | jnp.where(coll & ~room, PR.F_PENDING_OVERFLOW, 0).astype(jnp.int32)
        # v5 145행 `role_counts[role] += 1` — 담겼든 안 담겼든, **던지지 않았을 때만**
        #: ★x64 에서 `jnp.sum` 은 누산기를 int64 로 올린다 — int32 칸에 흩으면 FutureWarning(뒤에는 오류)
        counted = jnp.sum((ask & ~raised).astype(jnp.int32), dtype=jnp.int32)
        #: v5 `ppo/crane.py:82` `crane_actions[name] += 1` 은 `select` 가 **돌아온 뒤**다 — 던진 자리에서는
        #:  세지 않는다 (마스크 전부 거짓 등). `role_counts` 와 같은 조건이다.
        kk = jnp.clip(rec.kind[:, k], 0, PR.N_KINDS - 1)
        ca = cc.crane_actions.at[kk].add(jnp.where(ask & ~raised, 1, 0).astype(jnp.int32))
        return cc._replace(pending=p2,
                           role_counts=cc.role_counts.at[ROLE_CRANE].add(counted),
                           crane_actions=ca,
                           flags=cc.flags | jnp.bitwise_or.reduce(new)), None

    out, _ = lax.scan(body, c, jnp.arange(K, dtype=jnp.int32))
    return out


def make_tape_fn(cfg: PR.RuntimeConfig):
    """`multiblock.run_to_epoch(tape_fn=…)` 자리에 들어가는 함수 — 같은 cfg 면 같은 객체."""
    def tape_fn(carry: TapeCarry, tr, active):
        return merge_rec(carry, tr.rec, tr.decided, active, cfg)
    return tape_fn


# ═══════════════════════════════════════════════ ③ 상태·가치·비용 (경계가 쓰는 재료)
def reserve_table(run: MB.TerminalRun) -> jnp.ndarray:
    """(B,N) f64 — 오더 행마다 `Order.in_out_reserve_s`(= `round(도착예정, 3)`), 트럭이 아니면 +inf.

    v5 `features/block.py:92,122` 가 읽는 **통지된 예정**이다 (실현 게이트인이 아니다 —
    `dump_ground_truth.net_terminal_inputs` 머리말의 정보 경계). 명단이 정적이라 런당 한 번 만든다.
    """
    owner = np.asarray(run.tw.ledger.owner)
    row = np.asarray(run.tw.ledger.row)
    arr = np.asarray(run.tw.sched.arrival_s)
    B, N = np.asarray(run.tw.blocks.orders.block).shape
    out = np.full((B, N), np.inf)
    for s in range(int(run.tw.sched.s)):
        b, r = int(owner[s]), int(row[s])
        if 0 <= b < B and 0 <= r < N:
            out[b, r] = round(float(arr[s]), 3)         # stage/orders.py:88
    return jnp.asarray(out, F)


def block_rows_all(run: MB.TerminalRun, g: Geom, *, end_s, reserve_s, crane_order=None) -> jnp.ndarray:
    """(B,8) f64 — 블록마다 `v5feat.block_row` 한 줄 (v5 `PPORuntime.states_at` 의 앞 절반)."""
    f = partial(VF.block_row, g=g, end_s=jnp.asarray(end_s, F), crane_order=crane_order)
    return jax.vmap(lambda w, r: f(w, reserve_s=r))(run.tw.blocks, reserve_s)


def states_values(params, run: MB.TerminalRun, g: Geom, *, end_s, reserve_s, crane_order=None,
                  v5_cast: bool = True):
    """v5 `states_at(t)` + `policy.value(states)` → `(states (B,37), values (B,))`.

    `v5_cast` 는 v5 `encode` 가 특징을 float32 로 깎는 것(`ppo/model.py:14`)을 재현한다.
    """
    rows = block_rows_all(run, g, end_s=end_s, reserve_s=reserve_s, crane_order=crane_order)
    st = VF.state_rows(rows)
    x = jnp.asarray(VF.as_net_input(st), F) if v5_cast else st
    return st, VN.state_values(params.net, x)


def boundary_orders(run: MB.TerminalRun, t):
    """Φ 가 읽는 **전체 트럭** 오더 배열 (S,) — `month.day_orders` 의 전체판 (날 마스크 없음).

    v5 `PPORuntime.read_cost(t)` 는 `bridge.records` 전부를 읽고, `bridge._sync` 는 **값 ≤ t 인
    단계만** 찍는다 (bridge.py:109-118). 행 순서는 명단 순서 = v5 `records` 삽입 순서다.
    `day_orders` 와 달리 순수 jax 다 — 경계마다 불리므로 파이썬 루프를 쓸 수 없다.
    """
    L, o = run.tw.ledger, run.tw.blocks.orders
    B, N = o.block.shape
    t = jnp.asarray(t, F)
    oc = jnp.clip(L.owner, 0, B - 1)
    rc = jnp.clip(L.row, 0, N - 1)
    a_in = jnp.where(L.registered & (L.a_gate_in <= t + 1e-9), L.a_gate_in, jnp.asarray(EMPTY_TIME, F))
    go = o.gate_out_s[oc, rc]
    g_out = jnp.where(L.registered & (go <= t + 1e-9), go, jnp.asarray(EMPTY_TIME, F))
    S = int(L.registered.shape[0])
    flat = STATE.empty_orders(S)._replace(block=jnp.zeros((S,), jnp.int32),
                                      is_external=jnp.ones((S,), bool),
                                      gate_in_s=a_in, gate_out_s=g_out)
    return flat


def cost_at(run: MB.TerminalRun, lay, profile, t, *, archive=()) -> PR.CostOut:
    """v5 `PPORuntime.read_cost(t)` — 터미널 Φ 한 개 (원). 본선 유휴는 **호스트 순서**를 지킨다."""
    idle = MO.month_vessel_idle(run, lay, archive=tuple(archive))
    ships = list(idle)
    gt = jnp.asarray([idle[k][0] for k in ships] or [0.0], F)
    il = jnp.asarray([idle[k][1] for k in ships] or [0.0], F)
    vm = jnp.asarray([True] * len(ships) or [False])
    return PR.read_cost(boundary_orders(run, t), jnp.asarray(t, F), vessel_gt=gt, vessel_idle_s=il,
                        vessel_mask=vm, yc_extra_move_s=MO.yc_empty_travel_s(run, profile),
                        rehandles=MO.rehandles_of(run))


# ═══════════════════════════════════════════════ ④ 구간 → 갱신 입력
def intervals_to_batch(rows, *, values_f32: bool = True, feat_f32: bool = True) -> PB.IntervalBatch:
    """`ppo_runtime.IntervalRow` R 개 → `ppo_buffer.IntervalBatch` (칸 이름만 옮긴다).

    ★`values_f32` — v5 `Interval.values` 는 **torch float32 망의 출력**이고 `ppo/buffer.gae` 의
      delta 가 그래서 float32 로 계산된다 (`ppo_buffer` 담당의 '★명세는 … 실제 v5 는 delta 를
      float32 로 계산한다' 항목). 배열판도 그 갈래를 타야 같은 답이 나오므로 기본이 True 다.
      전부 f64 로 올리면 40/40 무대에서 |Δ| 1.5e-07 까지 갈린다.
    """
    if not rows:
        raise ValueError("구간이 하나도 없다 — v5 `_update` 는 buffer 가 비면 아무것도 안 한다")
    ch = [r.choices for r in rows]
    st = jnp.stack([jnp.asarray(r.states, jnp.float32 if feat_f32 else F) for r in rows])
    vl = jnp.stack([jnp.asarray(r.values, jnp.float32 if values_f32 else F) for r in rows])
    return PB.IntervalBatch(
        start_s=jnp.stack([jnp.asarray(r.start_s, F) for r in rows]),
        end_s=jnp.stack([jnp.asarray(r.end_s, F) for r in rows]),
        reward=jnp.stack([jnp.asarray(r.reward, F) for r in rows]),
        terminated=jnp.stack([jnp.asarray(r.terminated, bool) for r in rows]),
        values=vl, states=st,
        n_choices=jnp.stack([c.n for c in ch]),
        action=jnp.stack([c.action for c in ch]),
        log_prob=jnp.stack([c.log_prob for c in ch]),
        role=jnp.stack([c.role for c in ch]),
        choice_time_s=jnp.stack([c.time_s for c in ch]),
        n_cands=jnp.stack([c.n_actions for c in ch]),
        rows=jnp.stack([jnp.asarray(c.rows, jnp.float32 if feat_f32 else F) for c in ch]),
        mask=jnp.stack([c.mask for c in ch]),
        n_intervals=jnp.asarray(len(rows), jnp.int32))


def batch_to_tape(batch: PB.IntervalBatch, adv, returns):
    """`IntervalBatch` (R,B,…) → `ppo_update.Tape` (E,…) · 이득·수익도 (E,) 로 눕힌다.

    ★`[:n]` 슬라이스가 **필수**다 — v5 `entries` 는 실제 구간만이고 미니배치 순열 길이도 그 값이다
      (`ppo_update.Tape` 머리말).
    """
    n = int(batch.n_intervals)
    B, C, A, D = batch.n_blocks, batch.c_max, batch.a_max, batch.feat_dim
    e = n * B
    tape = PU.Tape(states=batch.states[:n].reshape(e, D),
                   rows=batch.rows[:n].reshape(e, C, A, D),
                   mask=batch.mask[:n].reshape(e, C, A),
                   action=batch.action[:n].reshape(e, C),
                   old_logp=batch.log_prob[:n].reshape(e, C),
                   n_choices=batch.n_choices[:n].reshape(e))
    return tape, jnp.asarray(adv)[:n].reshape(e), jnp.asarray(returns)[:n].reshape(e)


_gae_jit = jax.jit(PB.gae, static_argnames=("gamma", "lam", "time_unit_s"))
_update_jit = jax.jit(PU.ppo_update, static_argnames=("hyper",))


def apply_update(net, adam, rows, bootstrap, *, hyper: PU.PPOHyper, rng, gamma: float,
                 lam: float, time_unit_s: float, values_f32: bool = True):
    """v5 `ppo/update.update(policy, optimizer, buffer, bootstrap, config, rng)` 한 번.

    `net` 은 **망 가중치**(`v5net.V5PolicyParams`) 다 — `dispatch.V5NetParams`(망 + 터미널 층 값 둘)
    가 아니다. 순서를 v5 그대로 지킨다: GAE → 검사 → `entries` 순 호스트 순열 → 미니배치 scan → 보고.
    반환 `(net', adam', report dict, UpdateOut)`.
    """
    batch = intervals_to_batch(rows, values_f32=values_f32)
    audit = PB.gae_audit(batch, bootstrap, gamma=gamma, lam=lam, time_unit_s=time_unit_s)
    PB.raise_on_audit(audit)                       # v5 와 같은 순서·같은 메시지의 ValueError
    adv, ret = _gae_jit(batch, bootstrap, gamma=gamma, lam=lam, time_unit_s=time_unit_s)
    tape, adv_e, ret_e = batch_to_tape(batch, adv, ret)
    n_entries = int(batch.n_intervals) * batch.n_blocks
    before = rng.bit_generator.state
    orders = PU.draw_orders(rng, n_entries=n_entries, hyper=hyper)
    net2, adam2, out = _update_jit(net, adam, tape, adv_e, ret_e, orders, hyper)
    bits = int(out.violations)
    if bits:
        raise FloatingPointError(f"PPO 갱신 위반 비트 {bits} = {PU.violation_names(bits)}")
    # v5 는 조기중단으로 남은 epoch 의 순열을 **안 뽑는다** — 난수열을 그만큼 되감는다
    PU.rewind_orders_rng(rng, before, n_entries=n_entries,
                         epochs_entered=int(out.epochs_entered))
    return net2, adam2, PU.as_v5_report(out, n_intervals=int(batch.n_intervals)), out


# ═══════════════════════════════════════════════ ⑤ 드라이버
@dataclass(frozen=True)
class TrainConfig:
    """학습 한 번의 정적 설정 — `PPOConfig` + `PPORuntime` 인자 + 배열 칸 크기."""

    n_blocks: int = 21
    cmax: int = 256            #: ★배수 구간(7,200초)에 결정 140건이 쌓인다 — 60초 격자는 39건 (머리말 ③)
    amax: int = 13             #: 크레인 후보 칸 = `v5cond.item_max(k_max)` = k_max + 1
    rollout_intervals: int = 60
    epochs: int = 2
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    time_unit_s: float = 60.0
    reward_scale_krw: float = 1_000_000.0
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    training: bool = True
    stop_s: float | None = None
    learning_window_s: tuple | None = None
    #: ★행동을 추첨으로 뽑나 (v5 `PPORuntime.sample_actions`). **동등성 대조는 False 에서만** 뜻이 있다
    #:  (규칙 ⑥ — v5 `torch.multinomial` 난수열은 재현 불가). 학습에는 탐색이 필요하므로 실제 학습 런은
    #:  True 로 쓴다 — v5 정본도 `action_mode: "sample-all-days"` 다.
    sample_actions: bool = False
    #: 추첨 난수 씨 (블록마다 `fold_in` 으로 갈라 쓴다). None 이면 무대 시드를 쓴다.
    sample_seed: int | None = None

    def runtime(self) -> PR.RuntimeConfig:
        return PR.RuntimeConfig(n_blocks=self.n_blocks, cmax=self.cmax, amax=self.amax,
                                input_dim=VF.INPUT_DIM, rollout_intervals=self.rollout_intervals,
                                reward_scale_krw=self.reward_scale_krw, training=self.training,
                                stop_s=self.stop_s, learning_window_s=self.learning_window_s)

    def hyper(self) -> PU.PPOHyper:
        return PU.PPOHyper(epochs=self.epochs, minibatch_size=self.minibatch_size,
                           learning_rate=self.learning_rate, clip=self.clip,
                           value_coef=self.value_coef, entropy_coef=self.entropy_coef,
                           max_grad_norm=self.max_grad_norm, target_kl=self.target_kl)

    @classmethod
    def from_v5(cls, config, *, n_blocks=21, cmax=256, amax=13, training=True, stop_s=None,
                learning_window_s=None, sample_actions=False, sample_seed=None) -> "TrainConfig":
        """v5 `PPOConfig` 를 오리처럼 읽는다 (torch 를 들이지 않는다)."""
        return cls(n_blocks=n_blocks, cmax=cmax, amax=amax,
                   rollout_intervals=int(config.rollout_intervals), epochs=int(config.epochs),
                   minibatch_size=int(config.minibatch_size),
                   learning_rate=float(config.learning_rate), gamma=float(config.gamma),
                   gae_lambda=float(config.gae_lambda), time_unit_s=float(config.time_unit_s),
                   reward_scale_krw=float(config.reward_scale_krw), clip=float(config.clip),
                   value_coef=float(config.value_coef), entropy_coef=float(config.entropy_coef),
                   max_grad_norm=float(config.max_grad_norm), target_kl=float(config.target_kl),
                   training=bool(training), stop_s=stop_s, learning_window_s=learning_window_s,
                   sample_actions=bool(sample_actions), sample_seed=sample_seed)


@dataclass
class TrainState:
    """학습 런 하나의 **가변** 상태 — v5 `PPORuntime` 의 인스턴스 칸에 대응."""

    st: PR.RuntimeState
    params: object                 #: `dispatch.V5NetParams` (망 + end_s + reserve_s)
    adam: PU.AdamState
    buffer: list = dataclasses.field(default_factory=list)   #: `IntervalRow` 목록 (v5 `self.buffer`)
    updates: list = dataclasses.field(default_factory=list)
    rng: object = None             #: `np.random.default_rng(seed)` (미니배치 순열 — v5 `self.rng`)
    cost_breakdown: dict = dataclasses.field(default_factory=dict)


def new_train_state(params, cfg: TrainConfig, *, seed: int) -> TrainState:
    return TrainState(st=PR.new_state(cfg.runtime()), params=params,
                      adam=PU.init_adam(params.net), rng=np.random.default_rng(seed))


def _advance_recording(run: MB.TerminalRun, eng: MB.Engine, carry: TapeCarry, tape_fn):
    """`month._advance_to_epoch` 의 기록판 — 전 블록을 다음 검토 시각까지 굴리며 테이프를 쌓는다."""
    W, parked, _, stuck, carry2 = MB.run_to_epoch(run.tw.blocks, run.params, eng,
                                                  max_steps=eng.steps_per_epoch,
                                                  tape=carry, tape_fn=tape_fn)
    W = W._replace(violation=W.violation | jnp.where(stuck, V_STEPS_EXHAUSTED, 0).astype(jnp.int32))
    return run._replace(tw=run.tw._replace(blocks=W),
                        exhausted=run.exhausted + stuck.astype(jnp.int32)), carry2


_advance_rec_jit = jax.jit(_advance_recording, static_argnames=("eng", "tape_fn"))


def collect_epoch(run: MB.TerminalRun, tt, lay, e: int, eng: MB.Engine, ts: TrainState,
                  cfg: PR.RuntimeConfig, *, tape_fn, slots=None, grid=None):
    """검토 시각 하나 — `month.month_epoch` 와 **같은 순서**에 학습 기록을 얹는다.

    전 블록 전진(기록) → `_sync_locks` → 투입(NO_TARGET 거부권) → 원장 등록.
    반환 `(run, codes, skips, t, TrainState)`. 경계(`boundary_at`) 는 부르는 쪽이 이어서 한다 —
    v5 `review(m, t)` 가 `ann.review` **뒤에** `ppo.boundary(t)` 를 부르는 순서 그대로다.
    """
    #: ★갱신으로 망이 바뀌었을 수 있다 — 엔진이 읽는 `run.params` 를 지금 상태로 맞춘다
    run = run._replace(params=ts.params)
    carry = new_tape_carry(ts.st, cfg)
    run, carry = _advance_rec_jit(run, eng, carry, tape_fn)
    grid = MO.epoch_grid(run) if grid is None else grid
    t = float(grid[0][int(e)])
    slot = int(grid[1][int(e)])
    S = int(run.tw.sched.s)
    if slots is not None and slot not in slots:
        blocked, skips = np.zeros((S,), bool), []
    else:
        blocked, skips = MO.no_target_skips(run, tt, lay, e, t)
    run, codes = MO._review_jit(run, jnp.asarray(e, jnp.int32), jnp.asarray(blocked), eng)
    ts.st = ts.st._replace(pending=carry.pending, role_counts=carry.role_counts,
                           crane_actions=carry.crane_actions, flags=ts.st.flags | carry.flags)
    #: ★결정 칸 넘침·마스크 빈 칸 등은 **여기서 크게 실패한다** — 비트만 켜고 지나가면 잘린 테이프로
    #:  학습하고도 '완료' 로 보고된다 (`ppo_runtime.FATAL_BITS` 머리말).
    PR.raise_on_flags(ts.st.flags, where=f"collect_epoch e={int(e)}")
    return run, codes, skips, t, ts


def boundary_at(run: MB.TerminalRun, lay, profile, g: Geom, t: float, ts: TrainState,
                cfg: PR.RuntimeConfig, tcfg: TrainConfig, *, archive=(), crane_order=None,
                terminated: bool = False, final: bool = False, on_update=None):
    """v5 `PPORuntime.boundary(t)` 한 번 — 상태·가치·비용을 먼저 만들고 `ppo_runtime.boundary` 를 부른다.

    순서 (`ppo_runtime` 머리말 '쓰는 순서'): ① 상태·가치 → ② 비용 → ③ `boundary` →
    ④ `do_update` 면 갱신 → ⑤ `refresh_values`(갱신 뒤 망으로 다시) → ⑥ `stop` 이면 절단.
    반환 `(BoundaryOut, TrainState)`.
    """
    states, values = states_values(ts.params, run, g, end_s=lay.month_s,
                                   reserve_s=ts.params.reserve_s, crane_order=crane_order)
    cost = cost_at(run, lay, profile, t, archive=archive)
    ts.cost_breakdown = cost.phi.as_dict()
    st2, out = PR.boundary(ts.st, cfg, jnp.asarray(t, F), cost.total, states, values,
                           terminated=terminated, final=final)
    ts.st = st2
    PR.raise_on_flags(ts.st.flags, where=f"boundary t={float(t):g}")
    if bool(out.advanced) and bool(out.interval.valid):
        ts.buffer.append(out.interval)
    if bool(out.do_update) and tcfg.training and ts.buffer:
        net2, ts.adam, rep, _raw = apply_update(
            ts.params.net, ts.adam, ts.buffer, out.bootstrap,
            hyper=tcfg.hyper(), rng=ts.rng, gamma=tcfg.gamma, lam=tcfg.gae_lambda,
            time_unit_s=tcfg.time_unit_s)
        ts.params = ts.params._replace(net=net2)
        rep.update(index=len(ts.updates) + 1, time_s=float(t), cost_krw=float(ts.st.cost_krw))
        ts.updates.append(rep)
        ts.buffer.clear()
        #: ★`PR.boundary` 가 이미 `updates` 를 올렸다 — 여기서 또 올리면 두 번 센다
        if on_update is not None:
            on_update(rep)
        # ★가치는 갱신 **뒤** 망으로 다시 모은다 (v5 runtime.py:206-208)
        _s2, v2 = states_values(ts.params, run, g, end_s=lay.month_s,
                                reserve_s=ts.params.reserve_s, crane_order=crane_order)
        ts.st = PR.refresh_values(ts.st, v2)
    return out, ts


# ═══════════════════════════════════════════════ ⑥ 무대 세우기 · 30일 학습
def month_setup(*, seed: int, days, load=None, blocks=None, cap_moves=None, state_dict=None,
                net_params=None, k_max: int = VC.K_MAX, n_max=None, q_cap=None, log_cap=None,
                record: bool = True, check: bool = True, sample: bool = False, sample_seed=None):
    """v5 `stage/month_run.run_month` 기본 갈래의 무대 — 배열판 (조각 8 모듈 `month.to_month_world`).

    `record=True` 면 학습 기록 정책(`make_recording_policy`), False 면 조각 7 의
    `dispatch.make_v5net_policy` (기록 없음 = 평가·속도 측정용).
    반환 `(run, tt, lay, eng, params, prof, g, extras)`.
    """
    from ..stage import month as v5m
    from ..world.integrated.profiles import build_h21_profile
    from ..world.integrated.terminal_stream import OBS_24H
    from ..world.integrated.yard_layout import terminal_layout
    prof = build_h21_profile()
    layout = terminal_layout() if blocks is None else terminal_layout().subset(tuple(blocks))
    days = list(days)
    built = v5m.build_month(seed, days=days, profile=prof, layout=layout, lead_mode="DIST")
    v_by_day = v5m.plan_month_vessels(days, layout, obs=OBS_24H,
                                      truck_net=v5m.truck_net_by_block(built["schedule"]))
    if cap_moves is not None:
        v_by_day = {d: [dict(r, moves=min(int(r["moves"]), int(cap_moves))) for r in rows]
                    for d, rows in v_by_day.items()}
    tw, tt, lay = MO.to_month_world(prof, built, v_by_day, days=days, seed=seed, layout=layout,
                                    n_max=n_max, q_cap=q_cap, log_cap=log_cap)
    g = Geom.from_profile(prof)
    run = MO.make_month_run(tw, tt, g)
    k0 = tt.tables[0].crane_index[prof.cranes[0].crane_id]
    crane_order = _crane_order(tt, prof)
    if sample and not record:
        raise ValueError("추첨 수집은 기록 정책에만 있다 — record=True 로 부르라")
    pol = (make_recording_policy(g, k_max=k_max, crane_order=crane_order, sample=sample) if record
           else DP.make_v5net_policy(g, k_max=k_max, crane_order=crane_order))
    eng = MO.month_engine(tt, g, policy_fn=pol, check=check,
                          horizon_s=float(prof.decision_horizon_s), k0=int(k0))
    #: ★가중치는 전 블록 공유, `reserve_s` 만 블록마다 — vmap 이 망을 B 벌 복제하지 않게 축을 못박는다
    eng = dataclasses.replace(eng, params_axes=PARAMS_AXES_SAMPLE if sample else PARAMS_AXES)
    net = (VN.load_v5_params(state_dict) if state_dict is not None else net_params)
    if net is None:
        raise ValueError("학습 정책망 가중치가 필요하다 — state_dict 또는 net_params")
    #: ★`end_s` 는 v5 `MarketBridge.end_s` = **`month_s`** 다 (`stage/episode.py:171` ← `month_run.py:294`
    #:  `episode_end_s=month_s`). `sim_end_s`(배수 2시간 포함)가 아니다 — 블록 요약 칸 6(`clock_frac`)이
    #:  `min(1, t/end_s)` 라 분모를 틀리면 그 칸이 조용히 달라진다.
    #: 추첨 키는 **블록마다 한 벌** — 같은 스텝에 있는 두 블록이 같은 표본 자리를 쓰지 않게 한다
    skey = None
    if sample:
        base = jax.random.PRNGKey(int(seed if sample_seed is None else sample_seed))
        skey = jax.random.split(base, int(run.b))
    params = DP.V5NetParams(net=net, end_s=jnp.asarray(lay.month_s, F),
                            reserve_s=reserve_table(run), sample_key=skey)
    run = run._replace(params=params)
    return run, tt, lay, eng, params, prof, g, {"built": built, "v_by_day": v_by_day,
                                                "days": days, "layout": layout,
                                                "crane_order": crane_order, "policy_fn": pol}


def _crane_order(tt, prof) -> tuple:
    """v5 `sim.profile.cranes` 나열 순서 → 배열 크레인 번호 (블록 요약 칸 2 의 보정합 순서)."""
    idx = tt.tables[0].crane_index
    return tuple(int(idx[c.crane_id]) for c in prof.cranes)


class MonthScopeError(MO.MonthScopeError):
    """`CargoTerminal` 갈래는 범위 밖 (머리말 ②)."""


def train_month(*, seed: int, days, state_dict=None, net_params=None, tcfg: TrainConfig | None = None,
                e0: int = 0, e1: int | None = None, blocks=None, cap_moves=None, n_max=None,
                q_cap=None, log_cap=None, on_update=None, on_boundary=None, on_epoch=None,
                seed_data=None, check: bool = True, box: dict | None = None):
    """★배열판 30일 학습 — v5 `ppo/continuous.run_continuous` 의 몸통에 대응.

    v5 `review(m, t)` 순서 (month_run.py:513-539) 를 그대로 지킨다:
      ① 투입(`ann.review`) → ② `ppo.boundary(t)` → ③ 계수기 사진 → ④ 날 경계면 어제 닫기·오늘 배 붙이기
    끝에 `finish_run` · `ppo.finish(sim_end)` (v5 541-548행).

    `box` 를 주면 무대·상태를 그 사전에 담아 **이어 돌릴 수 있다** (`e0/e1` 로 쪼갤 때 같은 box 를 준다).
    반환 `(TrainState, dict 보고)`.
    """
    if seed_data is not None:
        raise MonthScopeError("`seed_data`(CargoTerminal 갈래)는 조각 8 범위 밖이다 — 머리말 ②")
    tcfg = tcfg or TrainConfig()
    #: ★`e0>0` 인데 이어 돌릴 상태(box)가 없으면 **크게 실패한다** (2026-09-27 통합·수정).
    #:  전에는 새 무대를 t=0 에 세운 뒤 앞 e0 에폭을 건너뛰어, 아무 결정도 하지 않은 세계를 '완료' 로
    #:  보고했다 (검증 반박 '`--e0` 가 프로세스를 넘으면 조용히 다른 세계를 만든다').
    if int(e0) > 0 and (box is None or "run" not in box):
        raise ValueError(
            f"e0={int(e0)} 인데 이어 돌릴 무대(box)가 없다 — 같은 프로세스에서 앞 구간을 먼저 돌려 "
            "같은 box 를 넘기라. 프로세스를 넘겨 이어 돌리는 손잡이는 아직 없다 (절인 상태 필요).")
    box = {} if box is None else box
    if "run" not in box:
        run, tt, lay, eng, params, prof, g, ex = month_setup(
            seed=seed, days=days, blocks=blocks, cap_moves=cap_moves, state_dict=state_dict,
            net_params=net_params, n_max=n_max, q_cap=q_cap, log_cap=log_cap, record=True,
            check=check, sample=bool(tcfg.sample_actions), sample_seed=tcfg.sample_seed)
        cfg = dataclasses.replace(tcfg, n_blocks=int(run.b)).runtime()
        ts = new_train_state(params, dataclasses.replace(tcfg, n_blocks=int(run.b)), seed=seed)
        box.update(run=run, tt=tt, lay=lay, eng=eng, prof=prof, g=g, ex=ex, cfg=cfg, ts=ts,
                   tape=MO.MonthTape(lay, prof), res=MO.MonthResult(),
                   slots=MO.gate_out_slots(run, lay), grid=MO.epoch_grid(run),
                   tape_fn=make_tape_fn(cfg), tcfg=dataclasses.replace(tcfg, n_blocks=int(run.b)))
    run, tt, lay, eng = box["run"], box["tt"], box["lay"], box["eng"]
    prof, g, cfg, ts = box["prof"], box["g"], box["cfg"], box["ts"]
    tcfg2, tape, res = box["tcfg"], box["tape"], box["res"]
    if res.state is None:
        res.state = {"day": 0, "snap": 0.0, "opened": (0, 0, 0),
                     "pruned_seen": np.zeros(np.asarray(run.tw.blocks.orders.status).shape, bool),
                     "retired_seen": np.zeros(np.asarray(run.tw.blocks.vessels.alive).shape, bool)}
    state = res.state
    E = run.n_epochs
    e1 = E if e1 is None else int(e1)
    n_days = len(lay.days)
    stopped = False
    for e in range(int(e0), int(e1)):
        run, codes, skips, t, ts = collect_epoch(run, tt, lay, e, eng, ts, cfg,
                                                 tape_fn=box["tape_fn"], slots=box["slots"],
                                                 grid=box["grid"])
        res.truck_skips += skips
        res.n_epochs += 1
        out, ts = boundary_at(run, lay, prof, g, t, ts, cfg, tcfg2, archive=tape.archive,
                             crane_order=box["ex"]["crane_order"], on_update=on_update)
        if on_epoch is not None:
            on_epoch(run, e, t, codes, out)
        if bool(out.stop):
            #: v5 `DebugStop` — 절단이고 엔진 실패가 아니다 (runtime.py:210)
            ts.st = ts.st._replace(truncated=jnp.ones((), bool))
            stopped = True
            box["run"] = run
            break
        if t >= state["snap"]:
            tape.snap(run, t)
            state["snap"] = t + MO.SNAP_S
        while state["day"] < n_days and t >= lay.days[state["day"]].t0 - 1e-9:
            d = lay.days[state["day"]]
            tape.snap(run, d.t0)
            if on_boundary is not None:
                on_boundary(run, int(d.index), float(d.t0))
            if state["day"] > 0:
                _close_day(run, tt, lay, tape, res, state, lay.days[state["day"] - 1], d.t0)
            run, rows, opened = MO.open_day(run, tt, lay, int(d.index), seed=seed, profile=prof,
                                           deadline_mult=MO.VESSEL_DEADLINE_MULT, t0=d.t0)
            res.vessel_admissions += rows
            state["opened"] = opened
            state["day"] += 1
            state["snap"] = t + MO.SNAP_S
        box["run"] = run
    if not stopped and e1 >= E:
        run = MB.finish_run_jit(run, eng)
        box["run"] = run
        # ★순서가 v5 다 (month_run.py:541-548): `ppo.finish(sim_end)` **먼저**, 그 다음 `tape.snap(month_s)`
        #   · 마지막 날 닫기. 거꾸로 하면 `close_day` 의 `retire_audit` 이 `tape.archive` 에 이름을 더해
        #   `month_vessel_idle` 의 **덧셈 순서**가 바뀌고 본선 유휴 합의 마지막 비트가 갈린다.
        cost = cost_at(run, lay, prof, lay.sim_end_s, archive=tape.archive)
        states, values = states_values(ts.params, run, g, end_s=lay.month_s,
                                       reserve_s=ts.params.reserve_s,
                                       crane_order=box["ex"]["crane_order"])
        st2, out = PR.finish(ts.st, cfg, jnp.asarray(lay.sim_end_s, F), cost.total, states, values)
        ts.st = st2
        ts.cost_breakdown = cost.phi.as_dict()
        tape.snap(run, lay.month_s)
        _close_day(run, tt, lay, tape, res, state, lay.days[-1], lay.month_s)
        if bool(out.advanced) and bool(out.interval.valid):
            ts.buffer.append(out.interval)
        if bool(out.do_update) and tcfg2.training and ts.buffer:
            net2, ts.adam, rep, _ = apply_update(
                ts.params.net, ts.adam, ts.buffer, out.bootstrap, hyper=tcfg2.hyper(), rng=ts.rng,
                gamma=tcfg2.gamma, lam=tcfg2.gae_lambda, time_unit_s=tcfg2.time_unit_s)
            ts.params = ts.params._replace(net=net2)
            rep.update(index=len(ts.updates) + 1, time_s=float(lay.sim_end_s),
                       cost_krw=float(ts.st.cost_krw))
            ts.updates.append(rep)
            ts.buffer.clear()
            if on_update is not None:
                on_update(rep)
        for d in lay.days:
            phi = MO.day_phi(run, lay, int(d.index), end_s=lay.sim_end_s, tape=tape,
                             t0=d.t0, t1=min(d.t1, lay.month_s))
            live = dict(res.live[int(d.index)])
            live.update(MO._phi_cols(phi), provisional=False)
            res.days.append(live)
        res.admitted = int(np.asarray(run.tw.ledger.registered).sum())
        res.skipped = len(res.truck_skips)
    return ts, report(ts, cfg, month=res, box=box, tcfg=tcfg2, seed=seed)


def _close_day(run, tt, lay, tape, res, state, d, t: float) -> None:
    """어제 닫기 — `month.close_day_at` **그 함수를 그대로 부른다**.

    ★전에는 여기에 같은 부기를 **복제**해 두었다 (`month.run_month` 안 `close_day`). 한쪽만 고치면
      조용히 갈리므로 (검증 반박) 조각 8 수정 단계에서 `month.py` 가 모듈 수준 함수로 내주게 하고
      이 자리는 얇은 껍데기로 남겼다.
    """
    MO.close_day_at(run, tt, lay, tape, res, state, d, t)


#: 배열판에 다리가 없어 **묻지도 않은** 역할 — 보고에서 `null` 로 드러낸다 (0 과 구별한다)
UNASKED_ROLES = ("seller", "buyer")


def report(ts: TrainState, cfg: PR.RuntimeConfig, *, month=None, box=None, tcfg=None,
           seed=None) -> dict:
    """v5 `PPORuntime.report()` 와 같은 키 (+ 배열판 표시).

    ★시장 계수기는 **0 이 아니라 `None`** 이다 (2026-09-27 통합·수정). 0 으로 적으면 "정책이 안 팔기를
      골랐다"(실제로 있는 결과 — yr331 규칙 팔이 거래 0) 와 "시장이 없어서 물어보지도 않았다" 가 구별되지
      않아, 이 보고를 읽는 사람·판정 스크립트가 배열 학습을 '거래를 안 한 강화학습 실행' 으로 오해한다.
      같은 이유로 `roles` 에도 `{"seller": null, "buyer": null}` 을 남긴다 (`PR.report` 는 0 인 칸을 빼므로
      '묻지 않았다' 가 사라진다). `market: "unported"` 가 그 사실의 이름표다.
    """
    d = PR.report(ts.st, cfg, cost_breakdown=ts.cost_breakdown, bridge=None, updates=ts.updates)
    d["market"] = "unported"          # 시장(판매자·구매자·중개·매칭)이 배열판에 없다 — 머리말 ⑤
    d["roles"] = {**{r: None for r in UNASKED_ROLES}, **d["roles"]}
    if tcfg is not None:
        #: v5 정본 매니페스트의 `action_mode` 와 같은 이름 (`ppo/continuous.py:62`)
        d["action_mode"] = "sample-all-days" if bool(tcfg.sample_actions) else "argmax"
        #: 실제로 쓴 씨를 적는다 — `sample_seed` 를 안 주면 **무대 시드**를 쓴다 (`month_setup`)
        eff = tcfg.sample_seed if tcfg.sample_seed is not None else seed
        d["sample_seed"] = (None if not tcfg.sample_actions or eff is None else int(eff))
    if month is not None:
        d["month"] = {"admitted": month.admitted, "skipped": month.skipped,
                      "n_epochs": month.n_epochs, "days": month.days, "live": month.live,
                      "vessel_admissions": month.vessel_admissions,
                      "truck_skips": month.truck_skips}
    if box is not None and "run" in box:
        run = box["run"]
        d["violation"] = int(np.asarray(run.tw.blocks.violation).max())
        d["exhausted"] = int(np.asarray(run.exhausted).sum())
    return d
