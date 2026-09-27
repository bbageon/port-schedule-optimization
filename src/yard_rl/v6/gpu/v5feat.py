"""v5 학습 정책망이 먹는 **37칸 특징 벡터**를 배열로 ([[YR-327]] 조각 7 · key=feat).

■ 결론부터 — 37칸은 어디서 왔나
  v5 의 학습 정책(`ppo/model.py:BlockPolicy`)은 입력 폭이 `INPUT_DIM = RAW_DIM(32) + 역할 4 + 1 = 37`
  이고, 한 줄(행)이 후보 하나다. 그 한 줄은 **세 군데서 이어 붙여** 만들어진다:

      features/block.py:block_features(n_cands=None)   → 앞 8칸  (블록 요약 · 후보수 칸 없음)
      ppo/crane.py:candidate_row                       → 그 뒤 16칸 (후보 7 + 종류 4 + 직전 크레인 5)
      ppo/model.py:encode(rows, "crane")               → 24칸을 32칸으로 **0 채움** + 역할 표시 5칸

  즉 실제로 값이 들어가는 칸은 **24개**이고, 24~31 은 구조적 0(패딩), 32~36 은 역할 표시다.
  `encode` 는 `torch.float32` 로 캐스팅하므로 **망이 보는 값은 float32** 다 — 이 모듈은
  캐스팅 **전**의 float64 를 내고, `as_net_input` 이 float32 로 내린다(둘 다 시험한다).

  ⚠️ `world/contract/vectors.py`·`contract/schema.py` 의 `FeatureVector`/`SCHEMA`(그룹 global·yc·
  candidate·queue·vessel · `norm_ref`·`clip_lo/clip_hi`·`assumed_default`)는 **학습 경로가 쓰지 않는다**
  — 쓰는 곳은 `integrated/adapter.py` 하나뿐이고 `ppo/` 와 `actors/` 는 한 번도 부르지 않는다
  (2026-09-26 grep 확인). 그래서 정규화·클립·기본값은 그 스키마가 아니라 **위 세 파일에 박힌
  나눗셈 상수와 `min`/`max`** 가 정본이다. 조각 7 명세가 가리킨 자리와 다르므로 여기 적어 둔다.

■ ★37칸의 정의와 순서 (정본 · `FEATURE_NAMES` 와 같은 순서)

    칸  이름                        v5 출처                        식 (정규화·클립 포함)
    ── 블록 요약 8칸 (features/block.py:137-160 · n_cands=None 이라 ⑦ 후보수 칸이 빠진 8차원) ──
     0  block_inside_10             block.py:148 · 58-68           블록 안 전부 ÷ 10        (게이트인 ≤ t < 게이트아웃)
     1  block_pipeline_10           block.py:149 · 71-81           오는 중 ÷ 10             (게이트인 ≤ t < 블록도착)
     2  block_crane_backlog_h       block.py:150 · 28-30           Σ max(0, 크레인여유시각 − t) ÷ 3600
     3  block_occupancy             block.py:151 · 33-35           야드 상자 수 ÷ max(1, 칸·열·단)
     4  block_vessel_slack_h_clip2  block.py:152 · 38-41           max(−2, min(2, 최소 본선여유 ÷ 3600))
     5  block_announced_soon_10     block.py:153 · 84-95           t < 통지예정 ≤ t+1800 인 오더 수 ÷ 10
     6  block_clock_frac            block.py:155                   min(1, t ÷ max(1, end_s))
     7  block_waiting_10            block.py:156 · 44-55           줄 선 대수 ÷ 10          (블록도착 ≤ t < 작업시작)
    ── 후보 16칸 (ppo/crane.py:39-56) ──
     8  cand_kind_serve             crane.py:44                    종류 == SERVE            (1.0 / 0.0)
     9  cand_kind_pre_rehandle      crane.py:44                    종류 == PRE_REHANDLE
    10  cand_kind_reposition        crane.py:44                    종류 == REPOSITION
    11  cand_kind_wait              crane.py:44                    종류 == WAIT
    12  cand_is_vessel              crane.py:45                    후보의 작업참조가 본선연계인가
    13  cand_is_external            crane.py:46                    후보의 작업참조가 외부트럭인가
    14  cand_duration_h             crane.py:47                    계획 소요 ÷ 3600          (계획 없으면 0)
    15  cand_cum_wait_h             crane.py:48 · engine.py:258-265 누적대기 ÷ 3600
    16  cand_empty_gantry_100m      crane.py:49                    계획 빈주행 ÷ 100
    17  cand_rehandles_10           crane.py:50                    계획 재조작 수 ÷ 10
    18  cand_end_bay_100            crane.py:51                    계획 도착 칸 ÷ 100
    19  prior_kind_serve            crane.py:53-54                 **직전 크레인이 고른** 종류 == SERVE
    20  prior_kind_pre_rehandle     crane.py:54
    21  prior_kind_reposition       crane.py:54
    22  prior_kind_wait             crane.py:54
    23  prior_end_bay_100           crane.py:55                    직전 크레인 계획 도착 칸 ÷ 100
    ── 구조적 0 패딩 8칸 (model.py:19-20 — 행 폭 24 < RAW_DIM 32) ──
    24..31 pad_0..pad_7                                            항상 0.0
    ── 역할 표시 5칸 (model.py:21-26) ──
    32  role_seller                 model.py:9,21                  크레인 행에서는 0.0
    33  role_buyer                                                  0.0
    34  role_crane                                                  **1.0**
    35  role_state                                                  0.0 (`states_at` 의 상태 행만 1.0)
    36  buyer_first_row             model.py:26                     Buyer 의 BUY 행 표시 — 크레인 행은 0.0

  ★숨은 함정 셋 (v5 코드를 그대로 읽어야 보인다)
  ① **PRE_REHANDLE 의 작업참조는 `is_external=True`·`is_vessel=False` 다 — 단, `vessel_prep=False` 일 때만**
     (candidates.py:308-311) — 그 작업이 본선이든 트럭이든. 그래서 칸 13 은 PRE 후보에서 **항상 1.0**,
     칸 12 는 **항상 0.0** 이다. SERVE 만 오더의 실제 값을 읽는다.
     ⚠️ v5 `_pre_rehandle` 에는 **둘째 분기**가 있다 — `CandidateGenerator(vessel_prep=True)` 면
        `iter_vessel_prep_jobs` 루프(candidates.py:319-332 · YR-088 본선판 ETA)가 `is_vessel=True`·
        `is_external=False` 인 PRE_REHANDLE 후보를 낸다. 그 설정에서는 v5 가 칸 12·13 에 (1,0) 을 주는데
        이 파일은 (0,1) 을 준다 — 해당 후보마다 두 칸이 틀린다.
        지금은 안 밟힌다: `vessel_prep` 기본값 False(candidates.py:228) 이고 `ppo/crane.py:73` 이
        `CandidateGenerator(config=LEGACY_DEFAULT)` 로만 만든다. `ExecPolicyConfig` 플래그가 아니라 **생성기
        생성 인자**라 README 의 'policy_config 플래그 미이식' 문구에도 안 걸린다.
        그래서 `cand_rows`·`features` 가 `vessel_prep` 을 인자로 받아 **True 면 크게 실패**한다
        (조용히 틀린 특징을 내보내지 않는다). 배열 후보 생성기(`cands3`)도 그 후보를 만들지 않는다.
  ② REPOSITION 의 작업참조 이름은 `"REPO:<크레인>:<칸>"` 이라 `sim.jobs` 에 없다(candidates.py:413)
     → 칸 15 누적대기는 0.0, 칸 12·13 도 0.0.
  ③ 칸 14·16·17·18 은 **계획이 없으면 0.0** 이다 — WAIT 과 "필수인데 계획 실패(PLAN_FAILED)" SERVE 가
     그 경우다(candidates.py:290-295). 계획 실패 후보는 목록에 **오르므로** 행이 실재한다.

■ 무엇을 안 하나 — 블록 요약의 두 입력은 **바깥에서 받는다**
  `block_features` 는 블록 세계만 보지 않는다. 다음 둘은 터미널(다중블록) 층의 값이라
  `block_row` 의 인자로 받는다 (기본값을 숨겨 두면 조용히 틀린 0 이 나온다):
    `end_s`      = `MarketBridge.end_s` (에피소드 끝) — `world.end_s`(블록 평가창)와 **다르다**
    `reserve_s`  = 오더별 `Order.in_out_reserve_s` (= `round(도착예정, 3)`, stage/orders.py:88)
                   그 블록 소속이 아니거나 오더가 없으면 +inf
  나머지 여섯 칸은 전부 `BlockWorld` 안에 있다.

■ 같은 답을 내기 위한 규칙 (exact.py 규약)
  · 실수는 전부 float64. **상수 나눗셈은 이 파일의 `_div`** — `exact.div_const` 는 분모가 스칼라라
    배열 분자에서 역수 곱으로 바뀐다(3.0÷10 이 0.3 대신 0.30000000000000004). `_div` 머리말 참조.
  · 크레인 여유 합은 v5 가 `sum(제너레이터)` 라 **파이썬 3.12 보정합**이다 → `exact.sum_python`.
    더하는 순서는 `sim.profile.cranes` 순서이므로 `crane_order` 로 받는다(기본은 배열 순).
  · 본선 여유 `pc − now − 남은수·간격` 은 곱을 `mul_exact` 로 실체화한 뒤 뺀다(FMA 방어).
  · 값이 비유한(NaN·inf)이면 v5 `encode` 가 예외를 던지므로, 안 실린 행은 **0 으로 지운다**.

■ 서명 (통합 단계가 그대로 쓴다 — 아래 `__all__`)
    block_row(world, g, *, end_s, reserve_s, crane_order=None) -> (8,) f64
    features(world, g, flat, pruned, *, block, prior_kind, prior_end_bay,
             c_max=None, role="crane") -> FeatOut(x (K,c_max,37) f64 · mask (K,c_max) · n_items (K,) · overflow ())
    cand_rows(world, g, flat, *, prior_kind, prior_end_bay) -> (K, C, 16) f64      # 열 순서 (candidate_id 아님)
    encode_rows(rows, role) -> (…, 37) f64                                         # model.py:encode 의 배열판
    state_rows(block_rows) -> (Bq, 37) f64                                         # runtime.states_at
    prior_from_choice(flat, k, col) -> (kind int32, end_bay f64)                    # 직전 크레인 선택 → 칸 19~23 재료
    as_net_input(x) -> float32                                                      # 망이 실제로 보는 값
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from .events import EMPTY_ID, EMPTY_TIME, TIME_DTYPE
from .exact import div_const, mul_exact, sum_python
from .geom import Geom
from .state import PK_PRE_REHANDLE, PK_REPOSITION, PK_SERVE, PK_WAIT, BlockWorld
from .travel import div_exact

__all__ = ["INPUT_DIM", "RAW_DIM", "BLOCK_DIM", "CAND_DIM", "CRANE_ROW_DIM", "ROLES", "KIND_ORDER",
           "FEATURE_NAMES", "IDX", "NO_PRIOR", "ANNOUNCE_HORIZON_S", "VESSEL_SLACK_CAP_S",
           "FeatOut", "block_row", "cand_rows", "features", "encode_rows", "state_rows",
           "prior_from_choice", "as_net_input"]

F = TIME_DTYPE

# ───────────────────────────────────────────────── 폭과 이름 (ppo/model.py:8-10)
RAW_DIM = 32                        # model.py:8
ROLES = ("seller", "buyer", "crane", "state")    # model.py:9 — 순서가 곧 칸 번호
INPUT_DIM = RAW_DIM + len(ROLES) + 1             # model.py:10 = 37
BLOCK_DIM = 8                       # features/block.py:BLOCK_DIM_BUYER — n_cands=None 판
CAND_DIM = 16                       # ppo/crane.py:candidate_row 의 뒤쪽
CRANE_ROW_DIM = BLOCK_DIM + CAND_DIM             # = 24 (패딩 전 실제 폭)

#: 종류 원핫 순서 — ppo/crane.py:16-17 `KINDS`
KIND_ORDER = (PK_SERVE, PK_PRE_REHANDLE, PK_REPOSITION, PK_WAIT)

#: 직전 크레인이 없다 (첫 크레인)
NO_PRIOR = EMPTY_ID

#: features/block.py:25 — "곧 올 통지분" 창
ANNOUNCE_HORIZON_S = 1800.0
#: features/block.py:41 — 본선 여유의 상한 대체값 (2시간)
VESSEL_SLACK_CAP_S = 2.0 * 3600.0

FEATURE_NAMES: tuple[str, ...] = (
    # 블록 요약 8 (features/block.py:147-157)
    "block_inside_10", "block_pipeline_10", "block_crane_backlog_h", "block_occupancy",
    "block_vessel_slack_h_clip2", "block_announced_soon_10", "block_clock_frac", "block_waiting_10",
    # 후보 16 (ppo/crane.py:44-55)
    "cand_kind_serve", "cand_kind_pre_rehandle", "cand_kind_reposition", "cand_kind_wait",
    "cand_is_vessel", "cand_is_external", "cand_duration_h", "cand_cum_wait_h",
    "cand_empty_gantry_100m", "cand_rehandles_10", "cand_end_bay_100",
    "prior_kind_serve", "prior_kind_pre_rehandle", "prior_kind_reposition", "prior_kind_wait",
    "prior_end_bay_100",
    # 구조적 0 패딩 8 (model.py:19-20)
    "pad_0", "pad_1", "pad_2", "pad_3", "pad_4", "pad_5", "pad_6", "pad_7",
    # 역할 표시 5 (model.py:21-26)
    "role_seller", "role_buyer", "role_crane", "role_state", "buyer_first_row",
)
assert len(FEATURE_NAMES) == INPUT_DIM, "37칸 이름표가 입력 폭과 어긋났다"

#: 이름 → 칸 번호 (시험·보고가 "몇 번째 칸이 갈렸나" 를 사람 말로 적을 때 쓴다)
IDX: dict[str, int] = {n: i for i, n in enumerate(FEATURE_NAMES)}


# ───────────────────────────────────────────────── ★상수 나눗셈 (함정 하나 더)
def _div(a, c: float):
    """`a / c` (c 는 파이썬 상수) — **배열이어도** IEEE 나눗셈 그대로. 파이썬 `a / c` 와 비트 동일.

    ★2026-09-26: 이 함수는 이제 `exact.div_const` **그 자체**다 (사본 없음). 조각 7 이 찾은 함정
      ("`div_const` 의 0차원 분모는 장벽 뒤에서 다시 브로드캐스트돼 역수 곱이 된다 — 배열 3.0÷10 이
      0.3 대신 0.30000000000000004") 을 공용 함수에서 고쳤으므로, 여기서 우회하지 않고 그대로 부른다.
      `cand_rehandles_10`(재조작 3건 ÷ 10) 이 그 한 비트 때문에 갈렸던 자리다.
    """
    return div_const(jnp.asarray(a, F), float(c), dtype=F)


# ───────────────────────────────────────────────── 블록 요약 8칸
def block_row(world: BlockWorld, g: Geom, *, end_s, reserve_s, crane_order=None) -> jnp.ndarray:
    """`features/block.py:block_features(mbt, bid, t, n_cands=None, …)` 의 배열판 → (8,) float64.

    `end_s`      () f64 — `MarketBridge.end_s`. **`world.end_s` 가 아니다** (블록 평가창과 다른 값).
    `reserve_s`  (N,) f64 — 오더별 `Order.in_out_reserve_s`(그 블록 소속만, 아니면 +inf).
    `crane_order` 정적 순열 — `sim.profile.cranes` 의 나열 순서를 배열 크레인 번호로 옮긴 것.
                 v5 가 그 순서로 `sum()` 하므로 보정합의 마지막 비트가 여기에 달렸다. None = 배열 순.

    ■ 왜 다섯 칸이 `gate_in_s`·`block_in_s`·`service_s`·`gate_out_s` 로 충분한가
      v5 는 `ExecutionRecord`(터미널이 보낸 사실)를 읽고, 그 기록은 `MarketBridge._sync` 가
      **`값 ≤ t` 인 것만** 찍는다 (bridge.py:79-89). 배열 세계의 네 열은 사건이 일어날 때 채워지고
      그 전에는 +inf 다 — `+inf > t` 가 곧 "아직 안 찍혔다" 라서 두 표현이 같은 답을 낸다.
      기록 자체가 없는 본선 작업은 `gate_in_s = +inf` 로 자연히 빠진다(장부 등록이 외부트럭 한정).
    """
    o, cr = world.orders, world.cranes
    t = world.clock
    present = o.block >= 0                                       # sim.jobs 에 있는 오더
    reserve_s = jnp.asarray(reserve_s, F)

    # ① 블록 안 전부 (block.py:58-68) · ② 오는 중 (71-81) · ⑨ 줄 선 대수 (44-55)
    entered = present & (o.gate_in_s <= t)
    inside = entered & (o.gate_out_s > t)
    pipeline = entered & (o.block_in_s > t)
    waiting = present & (o.block_in_s <= t) & (o.service_s > t)

    # ③ 크레인 여유 합 (block.py:28-30) — v5 는 `sum(제너레이터)` = 파이썬 3.12 보정합
    K = int(cr.available_at.shape[0])
    order = tuple(range(K)) if crane_order is None else tuple(int(k) for k in crane_order)
    if sorted(order) != list(range(K)):
        raise ValueError(f"crane_order 는 0..{K - 1} 의 순열이어야 한다 — 받은 값 {order}")
    backlog = sum_python(jnp.stack([jnp.maximum(0.0, cr.available_at[k] - t) for k in order]))

    # ④ 점유율 (block.py:33-35) — 분모는 정적 상수라 파이썬에서 미리 곱한다
    cap = float(max(1, int(g.bay_count) * int(g.row_count) * int(g.tier_max)))
    occupancy = _div(jnp.sum(world.conts.c_alive).astype(F), cap)

    # ⑤ 본선 여유 (block.py:38-41 · integrated/vessel.py:56-64)
    v = world.vessels
    on = v.alive & (v.planned_completion_s < EMPTY_TIME) & ~v.done
    rem = jnp.where(v.started, jnp.maximum(0, v.remaining), v.total_moves).astype(F)
    rem_s = mul_exact(rem, v.cadence_s)                          # 파이썬 `rem * 간격` 과 같은 한 번 반올림
    slack = (v.planned_completion_s - t) - rem_s                 # 좌결합 — vessel.py:64 그대로
    best = jnp.minimum(jnp.min(jnp.where(on, slack, EMPTY_TIME), initial=jnp.asarray(EMPTY_TIME, F)),
                       jnp.asarray(VESSEL_SLACK_CAP_S, F))       # min([…] + [2·3600])
    vessel_slack = jnp.maximum(-2.0, jnp.minimum(2.0, _div(best, 3600.0)))

    # ⑥ 곧 올 통지분 (block.py:84-95) — 통지된 예정만, 실현 게이트인은 안 읽는다
    soon = present & (t < reserve_s) & (reserve_s <= t + jnp.asarray(ANNOUNCE_HORIZON_S, F))

    # ⑧ 지금 몇 시인가 (block.py:155) — 분모가 상수로 접혀도 역수 곱이 되지 않게 장벽 뒤에 둔다
    den = lax.optimization_barrier(jnp.maximum(jnp.asarray(1.0, F), jnp.asarray(end_s, F)))
    clock_frac = jnp.minimum(1.0, div_exact(t, den))

    def per10(mask):
        return _div(jnp.sum(mask).astype(F), 10.0)

    return jnp.stack([per10(inside), per10(pipeline), _div(backlog, 3600.0), occupancy,
                      vessel_slack, per10(soon), clock_frac, per10(waiting)])


# ───────────────────────────────────────────────── 후보 16칸
def cand_rows(world: BlockWorld, g: Geom, flat, *, prior_kind, prior_end_bay,
              vessel_prep: bool = False) -> jnp.ndarray:
    """`ppo/crane.py:candidate_row` 의 뒤 16칸 → (K, C, 16) float64. 열 순서는 `flat` 의 열 순서다.

    `prior_kind`    (K,) int32 — 직전 크레인이 고른 후보의 종류 `PK_*`. `NO_PRIOR`(-1) = 직전 없음.
    `prior_end_bay` (K,) f64   — 그 후보 계획의 도착 칸. **NaN = 계획 없음**(WAIT) 또는 직전 없음.
    `vessel_prep`   v5 `CandidateGenerator(vessel_prep=...)` 값. **True 는 미이식이라 거절한다** (머리말 함정 ①)
                    — 그 설정에서는 PRE_REHANDLE 이 `is_vessel=True`·`is_external=False` 도 될 수 있어
                    칸 12·13 이 조용히 틀린다.
    """
    if vessel_prep:
        raise NotImplementedError(
            "CandidateGenerator(vessel_prep=True) 는 미이식이다 — 그 설정의 PRE_REHANDLE 은 "
            "is_vessel=True·is_external=False 인 후보를 내는데(candidates.py:319-332) 이 파일은 "
            "(0,1) 로 굳어 있어 칸 12·13 이 틀린다. 조용히 틀리지 않게 여기서 크게 실패한다.")
    o = world.orders
    t = world.clock
    kind = flat.kind
    job = flat.job
    ji = jnp.maximum(job, 0)                                     # -1 을 0 으로 눌러 gather (마스크로 지운다)
    has_job = job >= 0
    is_serve = kind == PK_SERVE
    is_pre = kind == PK_PRE_REHANDLE

    # 칸 8~11 — 종류 원핫 (crane.py:44)
    onehot = [(kind == k).astype(F) for k in KIND_ORDER]

    # 칸 12·13 — 작업참조의 깃발. ★PRE 는 is_vessel=False·is_external=True (candidates.py:308-311)
    #   — vessel_prep=False 일 때만. True 는 위에서 거절한다 (머리말 함정 ①)
    ov = has_job & o.is_vessel[ji]
    oe = has_job & o.is_external[ji]
    is_vessel = (is_serve & ov).astype(F)
    is_external = ((is_serve & oe) | is_pre).astype(F)

    # 칸 15 — 누적대기 (engine.py:258-265). 안쪽에서 `j.is_external_truck` 을 또 보므로 오더 깃발과 AND
    arrived = o.is_external & (o.block_in_s < EMPTY_TIME) & (o.block_in_s <= t)
    cum = jnp.where(arrived, t - o.block_in_s, 0.0)[ji]
    cum_on = (is_serve | is_pre) & oe
    cum_h = _div(jnp.where(cum_on, cum, 0.0), 3600.0)

    # 칸 14·16·17·18 — 계획 값. 계획이 없으면 0.0 (crane.py:47-51 의 `0.0 if plan is None else …`)
    ok = flat.plan_ok
    dur = _div(jnp.where(ok, flat.dur, 0.0), 3600.0)
    emp = _div(jnp.where(ok, flat.empty_m, 0.0), 100.0)
    reh = _div(jnp.where(ok, flat.rehandles, 0).astype(F), 10.0)
    eb = _div(jnp.where(ok, flat.end_bay, 0.0), 100.0)

    # 칸 19~23 — 직전 크레인 (crane.py:52-55). 크레인마다 다르므로 (K,1) 로 브로드캐스트
    pk = jnp.asarray(prior_kind, jnp.int32)[:, None]
    peb = jnp.asarray(prior_end_bay, F)[:, None]
    has_prior = pk >= 0
    prior_hot = [((pk == k) & has_prior).astype(F) for k in KIND_ORDER]
    # NaN 을 **나누기 전에** 지운다 — 산출에 비유한 값이 한 칸도 남지 않게 (v5 encode 가 거부한다)
    prior_eb = _div(jnp.where(has_prior & ~jnp.isnan(peb), peb, 0.0), 100.0)

    K, C = kind.shape
    cols = (onehot + [is_vessel, is_external, dur, cum_h, emp, reh, eb]
            + [jnp.broadcast_to(c, (K, C)) for c in prior_hot]
            + [jnp.broadcast_to(prior_eb, (K, C))])
    assert len(cols) == CAND_DIM
    return jnp.stack(cols, axis=-1)


# ───────────────────────────────────────────────── encode (ppo/model.py:13-27)
def encode_rows(rows: jnp.ndarray, role: str) -> jnp.ndarray:
    """`ppo/model.py:encode` 의 배열판 — 마지막 축을 32칸으로 0 채운 뒤 역할 칸을 세운다.

    `rows` (…, D) D ≤ 32. 돌려주는 값 (…, 37) float64. `role="buyer"` 는 v5 가 BUY/REJECT 두 행을
    요구하고 첫 행에만 칸 36 을 세우므로 **여기서는 지원하지 않는다**(크레인·상태 행만; 부르면 예외).
    """
    if role not in ROLES:
        raise ValueError(f"알 수 없는 역할 {role!r} — {ROLES} 중 하나")
    if role == "buyer":
        raise NotImplementedError("Buyer 행은 첫 행 표시(칸 36)가 필요해 별도 함수로 둔다")
    rows = jnp.asarray(rows, F)
    d = int(rows.shape[-1])
    if d > RAW_DIM:
        raise ValueError(f"행 폭 {d} 가 RAW_DIM={RAW_DIM} 을 넘는다 (model.py:15)")
    pad = jnp.zeros(rows.shape[:-1] + (RAW_DIM - d,), F)
    tail = jnp.asarray([1.0 if i == ROLES.index(role) else 0.0 for i in range(len(ROLES))] + [0.0], F)
    tail = jnp.broadcast_to(tail, rows.shape[:-1] + (len(ROLES) + 1,))
    return jnp.concatenate([rows, pad, tail], axis=-1)


def state_rows(block_rows: jnp.ndarray) -> jnp.ndarray:
    """`ppo/runtime.py:112-113 states_at` — 블록마다 블록 요약 한 줄, 역할 `state`. (Bq, 8) → (Bq, 37)."""
    return encode_rows(block_rows, "state")


# ───────────────────────────────────────────────── 후보 순서로 모으기
class FeatOut(NamedTuple):
    """`features` 의 산출 — 행 순서가 v5 `generate().items` 순서(= `candidate_id`)와 같다."""

    x: jnp.ndarray          # (K, c_max, 37) f64  — 안 실린 행은 전부 0.0
    mask: jnp.ndarray       # (K, c_max) bool     — v5 items 에 실린 행 (pad_mask)
    n_items: jnp.ndarray    # (K,) int32          — len(items) = 추린 수 + WAIT 1
    overflow: jnp.ndarray   # () int32            — candidate_id ≥ c_max 로 잘린 행 수 (0 이어야 정상)


def features(world: BlockWorld, g: Geom, flat, pruned, *, block, prior_kind, prior_end_bay,
             c_max: int | None = None, role: str = "crane", vessel_prep: bool = False) -> FeatOut:
    """결정 한 번의 **(K, c_max, 37)** 특징 — 행 i 가 `candidate_id == i` 인 후보다.

    `flat`   `cands3.flat_view(...)` · `pruned` `cands3.prune(...)`
    `block`  (8,) f64 — `block_row(...)`. 한 결정에 한 번 계산해 전 크레인이 공유한다
             (v5 도 크레인 루프 **앞**에서 한 번 부른다 — ppo/crane.py:70).
    `c_max`  정적 폭. None 이면 `flat` 의 열 수(손실 불가). v5 는 보통 12(=k_max) + WAIT 1 이지만
             필수 후보가 예산을 넘으면 더 길어질 수 있어(candidates.py:505-512) 잘린 수를 `overflow` 로 센다.

    ⚠️ **전 크레인 K 줄을 다 낸다.** v5 는 이번 결정에 부른 크레인(`dp.crane_ids`)만 계산하므로,
       통합자는 `world.decision.pending` 으로 걸러 쓴다. 직전 크레인 칸(19~23)은 **크레인 사전순
       바로 앞 크레인의 선택**이라 순차 루프(`lax.scan`) 안에서 `prior_from_choice` 로 갱신해야 한다 —
       한 번에 K 줄을 뽑으려면 앞 크레인의 선택이 이미 정해져 있어야 한다.
    """
    K, C = flat.kind.shape
    width = C if c_max is None else int(c_max)
    rows = cand_rows(world, g, flat, prior_kind=prior_kind, prior_end_bay=prior_end_bay,
                     vessel_prep=vessel_prep)
    blk = jnp.broadcast_to(jnp.asarray(block, F), (K, C, BLOCK_DIM))
    row24 = jnp.concatenate([blk, rows], axis=-1)                # crane.py:44 `list(block_row) + …`

    keep = pruned.keep
    cid = pruned.candidate_id
    fits = keep & (cid >= 0) & (cid < width)
    row24 = jnp.where(fits[:, :, None], row24, 0.0)              # 안 실린 행은 지운다 (비유한 값 차단)

    dst = jnp.where(fits, cid, width)                            # width 번째 = 버리는 칸
    ki = jnp.broadcast_to(jnp.arange(K, dtype=jnp.int32)[:, None], (K, C))
    packed = jnp.zeros((K, width + 1, CRANE_ROW_DIM), F).at[ki, dst].set(row24)[:, :width]

    n_items = (pruned.n_kept + 1).astype(jnp.int32)
    # 폭이 모자라면 (overflow > 0) 마스크가 없는 행을 참이라 말하지 않게 잘라 둔다 — 호출자는 overflow 로 판정한다
    mask = jnp.arange(width, dtype=jnp.int32)[None, :] < jnp.minimum(n_items, width)[:, None]
    overflow = jnp.sum(keep & (cid >= width)).astype(jnp.int32)
    # 안 실린 행은 역할 칸까지 0 으로 지운다 — v5 에는 그 행이 **아예 없다**
    x = jnp.where(mask[:, :, None], encode_rows(packed, role), 0.0)
    return FeatOut(x=x, mask=mask, n_items=n_items, overflow=overflow)


def prior_from_choice(flat, k, col):
    """직전 크레인의 선택(열 번호) → 칸 19~23 의 재료 `(kind int32, end_bay f64)`.

    `col < 0` 이면 직전 크레인이 없다 → `(NO_PRIOR, NaN)`. 계획이 없는 후보(WAIT)는 `end_bay = NaN`.
    """
    ci = jnp.maximum(jnp.asarray(col, jnp.int32), 0)
    on = jnp.asarray(col, jnp.int32) >= 0
    kind = jnp.where(on, flat.kind[k, ci], jnp.int32(NO_PRIOR)).astype(jnp.int32)
    eb = jnp.where(on & flat.plan_ok[k, ci], flat.end_bay[k, ci], jnp.nan)
    return kind, eb


def as_net_input(x: jnp.ndarray) -> jnp.ndarray:
    """망이 실제로 보는 값 — `ppo/model.py:14 torch.as_tensor(rows, dtype=torch.float32)` 와 같은 캐스팅."""
    return jnp.asarray(x, jnp.float32)
