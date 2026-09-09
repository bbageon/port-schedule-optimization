"""블록별 "오늘 48개 30분 슬롯" 계획 특징표 — 공개 정보만으로 만드는 하루 지도.

■ 왜 만드는가
지금 판매 정책(transfer_head.block_features)이 보는 것은 **한 시점의 스냅샷**이다 —
지금 안에 몇 대가 있고, 앞으로 30분에 몇 대가 통지됐는가. 그래서 "지금은 한가하지만
두 시간 뒤에 피크가 온다"·"이 블록은 오후에 본선이 붙는다" 를 표현할 방법이 없었다.
이 모듈은 하루(86,400초)를 30분 × 48칸으로 자르고, 칸마다 **무엇이 오기로 되어 있고
그때 얼마나 처리할 수 있는가**를 적은 계획표 [48, F] 를 만든다.

■ 정보 경계 (엄수 — 이 모듈의 존재 조건)
계획표는 "지금 공개적으로 알려진 것"만 담는다. 실현 미래값(`actual_gate_in` 이 아직
미래인 것·`actual_block_arrival`)은 **한 줄도 읽지 않는다**. 미래 구간의 트럭은 오직
①`notified_gate_in_s`(통지·재예약 시각, 없으면 `appointment_gate_time` — 접근자
`time_sell.notified_gate_in`) ②공개 예측 도착(`estimated_block_arrival`/`provided_eta`)
③본선 계획(`VesselPlan` — planned_start_s·total_moves·sts_move_interval_s·
planned_completion_s) 으로만 센다. 현재 상태(작업 status·재고·크레인 수·free_slots)는
"지금 관측"이라 공개다 — block_congestion·sell_review 가 쓰는 것과 같은 경계다.

■ 정규화
각 특징을 O(1) 규모로 만드는 제수를 **상수로 노출**한다(`SLOT_DIVISORS`, 순서는
`SLOT_FEATURES` 와 1:1). 극단 상태에서도 신경망 입력이 튀지 않도록 마지막에
[`CLIP_LO`, `CLIP_HI`] 로 자른다 — 원값이 필요하면 `raw=True` 로 부른다.

■ 동결 계약
`SLOT_FEATURES` 의 **순서**가 계약이다. 항목을 끼워 넣으면 이미 학습된 가중치의 의미가
통째로 바뀐다 — 추가는 반드시 **뒤에만** 붙인다.

■ 지금 한계 (정직 고지)
· 4·7·8번은 현 계약에서 48칸이 **모두 같은 값**이다(크레인 수·현재 여유칸·누적 이송은
  시간에 따라 변하는 계획값이 아니다). 열로 남겨 두는 이유는 계약 안정성이다.
· 3번은 지금 있는 미완 작업이 결정론 배수로 줄어드는 것만 본다 — 앞으로 올 트럭이
  만들 backlog 는 1·2번이 따로 알려 준다(이중 계상 금지).
· 1·2번은 이미 지나간 칸도 "그 칸에 통지됐던 대수"로 남는다 — 이 표는 남은 일의 예보가
  아니라 **하루 계획의 기록**이다.
"""
from __future__ import annotations

from .block_congestion import SVC_REF_S
from .cost_curve_v2 import GATE_BLOCK_MEAN_S
from .time_sell import notified_gate_in

# ------------------------------------------------------------------ 격자 계약
SLOT_S = 1800.0                  # 슬롯 길이 30분
N_SLOTS = 48                     # 24시간 = 30분 × 48
DAY_S = SLOT_S * N_SLOTS         # 86,400초 (관측계약 OBS_24H 의 하루)

# 특징 이름 — **순서 동결**(아래 _rows_for_block 의 열 순서와 1:1)
SLOT_FEATURES = (
    "notified_in",          # 1. 그 슬롯에 통지된 반입(GATE_IN) 트럭 수
    "notified_out",         # 2. 그 슬롯에 통지된 반출(GATE_OUT) 트럭 수
    "backlog_start",        # 3. 슬롯 시작 시점의 예상 미완 작업 수 (이미 존재하는 것만)
    "crane_capacity",       # 4. 크레인 예상 처리 가능량 (크레인 수 × 슬롯 ÷ 기준 서비스)
    "vessel_moves",         # 5. 그 슬롯에 걸치는 본선 작업량 (계획 기준 moves)
    "vessel_deadline_slack",  # 6. 본선 마감까지 여유(초) — 없으면 큰 값
    "free_slots",           # 7. 블록 장치 여유칸
    "accepted_transfers",   # 8. 이 블록이 지금까지 수락한 이송 수(누적)
    "slot_phase",           # 9. (슬롯 시작 − now)/86400 — 지난 슬롯은 음수
)
N_FEATURES = len(SLOT_FEATURES)

# ------------------------------------------------------------------ 정규화 제수 (동결)
DIV_TRUCKS = 20.0        # 슬롯당 트럭 수 — 30분 20대면 사실상 만원(2크레인 처리량 상한급)
DIV_BACKLOG = 20.0       # 미완 작업 수 — 위와 같은 잣대(같은 단위끼리 같은 제수)
DIV_CRANE_CAP = 20.0     # 2크레인 × 1800초 ÷ 180초 = 20건 → 기준 무대에서 정확히 1.0
DIV_VESSEL_MOVES = 20.0  # 본선 moves — 트럭 1대와 크레인 1건이 맞먹으므로 같은 제수
DIV_SLACK_S = 3600.0     # 마감 여유는 "시간" 단위로 읽는다
DIV_FREE_SLOTS = 1000.0  # 물리 용량 24×10×6=1,440 → 대략 [0, 1.44] (transfer_head 와 동일 관례)
DIV_TRANSFERS = 20.0     # 누적 수락 이송 — 하루 20건이면 이미 많이 받은 블록
DIV_PHASE_S = DAY_S      # 과거/미래 표식은 하루로 나눠 [−1, 1]
NO_DEADLINE_S = DAY_S    # 본선 마감이 없을 때 쓰는 "큰 값"(=하루) — 급하지 않다는 뜻

SLOT_DIVISORS = (DIV_TRUCKS, DIV_TRUCKS, DIV_BACKLOG, DIV_CRANE_CAP, DIV_VESSEL_MOVES,
                 DIV_SLACK_S, DIV_FREE_SLOTS, DIV_TRANSFERS, DIV_PHASE_S)

CLIP_LO, CLIP_HI = -2.0, 5.0   # 정규화 후 안전 범위 — 극단 상태에서도 입력이 튀지 않게

# 블록에 도착했으나 아직 끝나지 않은 작업 = backlog (현재 관측 — 공개)
_BACKLOG_STATES = ("RELEASED", "WAITING", "ASSIGNED", "RUNNING")
_TRUCK_FLOWS = ("GATE_IN", "GATE_OUT")


# ------------------------------------------------------------------ 격자 유틸
def slot_index(t: float) -> int:
    """시각(초) → 슬롯 번호 0~47.

    하루 밖(음수·배수 구간 t ≥ 86,400)은 양 끝 슬롯으로 잘라 붙인다 — 계획표는 하루짜리
    고정 격자이고, 배수 구간은 측정 대상이 아니기 때문이다(terminal_stream 의 drain).
    집계에서 하루 밖 항목을 **버려야 하는** 곳은 `_day_slot` 을 쓴다(끝 칸 부풀림 방지).
    """
    return max(0, min(N_SLOTS - 1, int(t // SLOT_S)))


def slot_start(i: int) -> float:
    """슬롯 번호 → 시작 시각(초)."""
    if not 0 <= int(i) < N_SLOTS:
        raise IndexError(f"슬롯 번호는 0~{N_SLOTS - 1}: {i}")
    return float(int(i)) * SLOT_S


def _day_slot(t: float | None) -> int | None:
    """하루 격자 안이면 슬롯 번호, 밖이면 None (배수 구간을 47번 칸에 얹지 않기 위함)."""
    if t is None:
        return None
    if t < 0.0 or t >= DAY_S:
        return None
    return int(t // SLOT_S)


def public_block_eta(job) -> float | None:
    """정책이 읽어도 되는 **공개 예측 블록도착 시각**.

    ①생성기가 통지 기반으로 만든 예측(`estimated_block_arrival`) ②그 별칭
    (`provided_eta`) ③둘 다 없으면 통지 gate-in + 기대 주행(계약 중심값 300초).
    실현 `actual_block_arrival` 은 쓰지 않는다 — 미래 구간에서 그것을 읽으면 누출이다.
    `or` 대신 None 검사를 쓰는 이유: 0.0(하루 시작 도착)이 falsy 라 조용히 밀려난다.
    """
    for field in ("estimated_block_arrival", "provided_eta"):
        v = getattr(job, field, None)
        if v is not None:
            return float(v)
    gi = notified_gate_in(job)
    return None if gi is None else float(gi) + GATE_BLOCK_MEAN_S


# ------------------------------------------------------------------ 원장 1회 순회
def _scan_ledger(mbt, blocks) -> dict[str, dict]:
    """전역 원장을 **1회만** 훑어 블록별 (통지 반입/반출 슬롯 히스토그램, 수락 이송 수).

    호출자가 블록마다 3,600건 원장을 다시 훑으면 21블록 × 1,440 epoch 에서 비용이
    폭발한다 — terminal_slot_plan 은 이 함수를 한 번 부르고 나눠 쓴다.
    """
    acc = {b: {"in": [0.0] * N_SLOTS, "out": [0.0] * N_SLOTS, "recv": 0.0}
           for b in blocks}
    for jid, rec in mbt.ledger.records.items():
        # 수락 이송: 마지막 이송의 목적지가 이 블록이면 "받은 것" (이송 이력은 공개 원장)
        if rec.transfer_history:
            dst = rec.transfer_history[-1][1]
            if dst in acc:
                acc[dst]["recv"] += 1.0
        if rec.owner not in acc or rec.flow not in _TRUCK_FLOWS:
            continue                      # 본선 연계 작업은 특징 5(계획 moves)가 담당
        sim = mbt.blocks.get(rec.owner)
        job = sim.jobs.get(jid) if sim is not None else None
        if job is None:
            continue
        # 칸은 **공개 예측 블록도착** 기준으로 정한다(그 시각에 블록이 일을 받는다).
        # 이미 처리된 트럭도 자기 칸에 그대로 남는다 — 계획의 기록이지 잔여량이 아니다.
        i = _day_slot(public_block_eta(job))
        if i is None:
            continue
        acc[rec.owner]["in" if rec.flow == "GATE_IN" else "out"][i] += 1.0

    # ★YR-171-A 정보 계약 — 공개 예약 장부가 붙어 있으면 반입/반출 칸은 **장부가 정본**.
    # 통지된 것만 세는 위 경로는 미래 칸이 0.4~5.6% 만 채워진다(yr171_horizon_probe).
    # 장부는 하루 명단 전체를 담으므로 "안 보이는 것"과 "한가한 것"이 구분된다.
    # 수락 이송(recv)은 **현재 관측**이라 장부가 아니라 원장에서 그대로 온다.
    from .day_plan import get as _day_plan_get
    plan = _day_plan_get(mbt)
    if plan is not None:
        hist = plan.slot_hist()
        for b in acc:
            h = hist.get(b)
            acc[b]["in"] = h["in"] if h else [0.0] * N_SLOTS
            acc[b]["out"] = h["out"] if h else [0.0] * N_SLOTS
    return acc


# ------------------------------------------------------------------ 슬롯별 원료
def _vessel_moves_per_slot(sim) -> list[float]:
    """본선 계획 moves 를 슬롯에 배분 — cadence 등간격 가정(계획서가 주는 유일한 형태).

    구간 [planned_start, planned_start + total_moves × cadence] 과 슬롯의 겹침 길이를
    cadence 로 나눈다. 실현 진행(remaining_moves)이 아니라 **계획**만 쓴다.
    """
    out = [0.0] * N_SLOTS
    for v in getattr(sim, "vessels", {}).values():
        plan = getattr(v, "plan", None)
        if plan is None:
            continue
        iv = float(getattr(plan, "sts_move_interval_s", 0.0) or 0.0)
        moves = int(getattr(plan, "total_moves", 0) or 0)
        if iv <= 0.0 or moves <= 0:
            continue
        s = float(plan.planned_start_s)
        e = s + moves * iv
        for i in range(N_SLOTS):
            a = slot_start(i)
            ov = min(a + SLOT_S, e) - max(a, s)
            if ov > 0.0:
                out[i] += ov / iv
    return out


def _deadline_slack_at(sim, at_s: float) -> float:
    """`at_s` 기준 **가장 임박한** 본선 계획 마감까지 남은 초. 없으면 NO_DEADLINE_S.

    이미 지난 마감(미완 본선)은 음수로 남긴다 — "늦고 있다"는 정보 자체가 신호다.
    끝난 본선(`done`)은 제외한다(현재 관측 — 공개 상태값).
    """
    best = None
    for v in getattr(sim, "vessels", {}).values():
        if getattr(v, "done", False):
            continue
        pc = getattr(getattr(v, "plan", None), "planned_completion_s", None)
        if pc is None:
            continue                       # 완료시각 결측(SYMPTOM) — 마감 압박 미상
        rem = float(pc) - at_s
        if best is None or rem < best:
            best = rem
    return NO_DEADLINE_S if best is None else best


def _rows_for_block(mbt, bid: str, now: float, hist: dict,
                    raw: bool) -> list[list[float]]:
    """블록 하나의 [48, F] 표 — 원료 조립 + 정규화."""
    sim = mbt.blocks[bid]
    n_cranes = max(1, len(sim.profile.cranes))
    # (3) 지금 블록 안에 남아 있는 미완 작업 — "이미 존재하는 것만" (통지분은 특징 1·2)
    backlog_now = float(sum(1 for j in sim.jobs.values()
                            if j.status.name in _BACKLOG_STATES))
    drain_per_s = n_cranes / SVC_REF_S            # 기준 서비스시간 기준 결정론 배수
    # (4) 슬롯당 처리 가능량 — 현 계약에서는 전 슬롯 동일(크레인 수가 변하지 않는다).
    #     그래도 슬롯별 열로 둔다: 나중에 크레인 가용시간이 시간에 따라 달라져도 계약 불변.
    capacity = n_cranes * SLOT_S / SVC_REF_S
    moves = _vessel_moves_per_slot(sim)
    free = float(mbt.free_slots(bid))             # (7) 현재 여유칸 (예약분 차감 포함)
    recv = hist["recv"]                           # (8) 누적 수락 이송
    rows: list[list[float]] = []
    for i in range(N_SLOTS):
        a = slot_start(i)
        # 지난 슬롯은 되돌아볼 이력이 없으므로 현재 backlog 를 그대로 둔다(elapsed≥0 clamp).
        elapsed = max(0.0, a - now)
        vals = [hist["in"][i], hist["out"][i],
                max(0.0, backlog_now - elapsed * drain_per_s),
                capacity, moves[i], _deadline_slack_at(sim, a),
                free, recv, a - now]
        if not raw:
            vals = [_clip(v / d) for v, d in zip(vals, SLOT_DIVISORS)]
        rows.append(vals)
    return rows


def _clip(x: float) -> float:
    return max(CLIP_LO, min(CLIP_HI, x))


# ------------------------------------------------------------------ 공개 API
def block_slot_plan(mbt, bid: str, now: float, *,
                    raw: bool = False) -> list[list[float]]:
    """블록 하나의 [48, F] 계획표. 공개 정보만.

    raw=True 면 정규화·clip 없이 원단위(대수·초)로 돌려준다 — 감사·디버깅용.
    """
    if bid not in mbt.blocks:
        raise KeyError(f"터미널에 없는 블록: {bid}")
    hist = _scan_ledger(mbt, (bid,))[bid]
    return _rows_for_block(mbt, bid, float(now), hist, raw)


def terminal_slot_plan(mbt, now: float, *,
                       raw: bool = False) -> dict[str, list[list[float]]]:
    """전 블록 계획표 — 원장 1회 순회로 21블록을 한꺼번에 만든다."""
    now = float(now)
    hists = _scan_ledger(mbt, tuple(mbt.blocks))
    return {b: _rows_for_block(mbt, b, now, hists[b], raw) for b in mbt.blocks}


# ------------------------------------------------------------------ 감사 보조
def slot_plan_stats(plan: list[list[float]]) -> dict[str, tuple[float, float, float]]:
    """특징별 (최소, 최대, 합) — shape·범위 점검용. 계획표를 눈으로 읽을 때 쓴다."""
    if len(plan) != N_SLOTS or any(len(r) != N_FEATURES for r in plan):
        raise ValueError(f"계획표 shape 위반 — ({N_SLOTS}, {N_FEATURES}) 이어야 함")
    out = {}
    for k, name in enumerate(SLOT_FEATURES):
        col = [r[k] for r in plan]
        out[name] = (min(col), max(col), sum(col))
    return out
