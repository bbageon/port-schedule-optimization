"""거래별 공로(difference reward) 계산 — "그 거래가 터미널 전체 비용을 얼마나 바꿨나".

■ 왜 필요한가 (문제)
현 보상은 구간 전역 증분비용이라 **누가 그 변화를 만들었는지**를 구분하지 못한다.
21개 블록이 같은 60초 격자에서 동시에 제안을 넣으므로, 한 블록의 좋은 판매가 다른
블록의 나쁜 판매에 묻히거나 그 반대가 된다(신용 배분 문제). 판매한 source 에게
"네 거래가 없었다면 터미널 비용이 어떻게 됐을까"를 돌려주면 신호가 자기 행동에
붙는다.

■ 무엇을 계산하나 (정의)
    D_i = C(offer i 를 빼고 다시 매칭한 총비용) − C(전체 매칭 총비용)
D_i > 0 = 그 offer 가 터미널 비용을 **줄였다**(없었으면 더 비쌌다).
핵심은 "빼고 **다시** 매칭"이라는 점이다 — offer i 가 없으면 그 자리(빈 슬롯)를 다른
offer 가 대신 쓸 수 있고, 그만큼 i 의 공로는 자기 비용(−c_i)보다 작아진다. 자기 비용을
그대로 공로로 주면 "내가 아니어도 누군가 챙겼을 이득"까지 자기 몫으로 챙기게 된다.

■ 매칭 규칙 (sell_review.UnifiedSellOrchestrator.review 와 동일)
매 반복에서 남은 (제안 × 좌표) 전 조합 중 비용 최소 쌍을 고르고, 그 offer 를 제거한다
(작업 1회 이송 상한 = offer 당 좌표 1개). 공간 좌표를 고르면 그 블록의 잔여 용량을 1
줄이고, 여유가 `capacity_margin` 이하로 떨어진 좌표는 이후 후보에서 뺀다. KEEP·TIME 은
블록 간 이동이 아니므로 용량을 쓰지 않는다. 동점은 (cost, offer_key, coord) 사전순으로
깬다 — 제안 목록의 **순서와 무관**한 결과(순열 불변)를 보장하기 위함이다.

■ 순수 함수 계약 (반사실 계산의 전제)
`matching` 의 입력은 **좌표 비용표뿐**이다 — mbt·엔진·시뮬 상태를 일절 건드리지 않는다.
반사실(counterfactual)은 같은 표 위에서 offer 만 빼고 다시 푸는 것이므로, 시뮬레이션을
되감지 않고도 "없었다면"을 계산할 수 있다.

■ 한계 (정직 고지)
resolver 본체는 배정 1건마다 가상 대기열 q 를 갱신하고 **다음 반복의 비용을 재계산**
한다(볼록 비용). 여기서는 비용표가 동결된 1벌이라, 그 재계산분만큼 근사다. 다만
전체 매칭과 반사실 매칭이 **같은 표**를 쓰므로 두 값의 차이(D_i)는 같은 기준 위에서
비교된다. 표를 매 반복 재계산하려면 비용 함수를 콜백으로 받아야 하고, 그러면 순수
함수가 아니게 되어 반사실 비용이 폭증한다(offer 수 × 반복 수 × 좌표 수) — 동결 표는
그 절충이다.

■ 정보 경계: 이 모듈은 좌표 비용표만 받는다. 표를 만드는 쪽(`freeze_coord_costs`)은
resolver 와 같은 공개 정보(통지 gate-in·공개 ETA·내부 대수·통지 pipeline)만 읽는다.
"""
from __future__ import annotations

KEEP_COORD = "KEEP"
TIME_COORD = "TIME"
# 용량을 쓰지 않는 좌표 — KEEP(현 계획 유지)·TIME(같은 블록 +Δ 이연)은 블록 간 이동이
# 아니라서 수신 블록의 슬롯을 차지하지 않는다.
FREE_COORDS = frozenset({KEEP_COORD, TIME_COORD})


# ------------------------------------------------------------------ 결정론 보조
def _tie_key(x):
    """동점 파훼용 정규화 키.

    offer_key 는 호출부에 따라 문자열(job_id)일 수도, 튜플((src, jid))일 수도 있다.
    형이 섞이면 `<` 비교가 TypeError 로 깨져 결과가 호출부에 좌우되므로, (형 등급,
    값) 로 감싸 **항상 비교 가능**하게 만든다. 같은 형끼리는 사전순이 그대로 보존된다.
    """
    if isinstance(x, tuple):
        return (3, tuple(_tie_key(e) for e in x))
    if isinstance(x, str):
        return (2, x)
    if isinstance(x, (int, float)):
        return (1, float(x))
    return (4, repr(x))


def _validate(coord_costs_by_offer: dict, free_slots: dict) -> None:
    """좌표가 공간 좌표인데 free_slots 에 없으면 즉시 실격 (fail-fast).

    조용히 "용량 무한"으로 취급하면 용량 가드가 사라진 채 다른 실험이 된다 —
    resolver 의 fail-closed 정신을 그대로 따른다.
    """
    for okey, rows in coord_costs_by_offer.items():
        for cost, coord in rows:
            if coord in FREE_COORDS or coord in free_slots:
                continue
            raise KeyError(
                f"좌표 {coord!r}(offer {okey!r})의 용량을 알 수 없다 — "
                f"free_slots 에 넣거나 FREE_COORDS({sorted(FREE_COORDS)})를 쓰라")


def _available(coord: str, free_slots: dict, capacity_margin: int,
               used: dict) -> bool:
    """그 좌표에 1대를 더 얹을 수 있나 — resolver 와 동일 판정.

    resolver: `free_slots(dst) - vcap[dst] <= capacity_margin` 이면 후보 제외.
    """
    if coord in FREE_COORDS:
        return True
    return free_slots[coord] - used.get(coord, 0) > capacity_margin


# ------------------------------------------------------------------ 순수 매칭
def matching(coord_costs_by_offer: dict, *, free_slots: dict,
             capacity_margin: int, exclude: set | None = None) -> list[tuple]:
    """resolver 와 동일 규칙의 **순수 함수** 매칭.

    Parameters
    ----------
    coord_costs_by_offer : {offer_key: [(cost, coord), ...]}
        offer 별 대안 좌표의 순비용표. 음수 = 이득. coord 는 수신 블록 id 또는
        "KEEP"/"TIME".
    free_slots : {block_id: int}
        블록별 수신 가능 슬롯 (mbt.free_slots(b) 의 스냅샷).
    capacity_margin : int
        만재 직전 이송 금지 여유 (mbt.capacity_margin).
    exclude : set | None
        빼고 다시 매칭할 offer_key 집합 — 반사실(counterfactual)용.

    Returns
    -------
    [(offer_key, coord, cost), ...]  — **배정된 순서대로**(전역 최소 쌍 순).
        배정 가능한 좌표가 하나도 없는 offer 는 결과에서 빠진다(=아무것도 안 함,
        총비용 기여 0). resolver 도 같은 상황에서 그 제안을 그대로 흘려보낸다.
    """
    exclude = set() if exclude is None else set(exclude)
    _validate(coord_costs_by_offer, free_slots)
    remaining = {k: list(v) for k, v in coord_costs_by_offer.items()
                 if k not in exclude}
    used: dict[str, int] = {}          # 가상 배정 수 (resolver 의 vcap 과 같은 역할)
    result: list[tuple] = []
    while remaining:
        best = None                    # (cost, _tie(offer_key), _tie(coord)) 최소
        best_pick = None               # (offer_key, coord, cost)
        for okey, rows in remaining.items():
            for cost, coord in rows:
                if not _available(coord, free_slots, capacity_margin, used):
                    continue
                key = (float(cost), _tie_key(okey), _tie_key(coord))
                if best is None or key < best:
                    best, best_pick = key, (okey, coord, float(cost))
        if best_pick is None:
            break                      # 남은 offer 전부 배정 불가 — 용량은 늘지 않는다
        okey, coord, cost = best_pick
        del remaining[okey]            # offer 당 좌표 1개 (작업 1회 이송 상한)
        result.append((okey, coord, cost))
        if coord not in FREE_COORDS:
            used[coord] = used.get(coord, 0) + 1
    return result


def total_cost(assignment: list[tuple]) -> float:
    """매칭 결과의 총비용 — 미배정 offer 는 0 기여."""
    return sum(c for _, _, c in assignment)


# ------------------------------------------------------------------ 차이 공로
def difference_credit(coord_costs_by_offer: dict, *, free_slots: dict,
                      capacity_margin: int) -> dict:
    """offer 별 차이 공로 D_i.

        D_i = C(offer i 를 뺀 매칭) − C(전체 매칭)

    양수 = 그 offer 가 터미널 비용을 줄였다. 다른 offer 가 그 빈 슬롯을 대신 쓰는
    효과까지 반영된다(빼고 **다시** 매칭하므로) — 대체 가능한 이득은 공로에서 빠진다.

    Returns
    -------
    {offer_key: D_i}  — 입력 표의 모든 offer 에 대해 값을 낸다(미배정 offer 는 0).
    """
    base = matching(coord_costs_by_offer, free_slots=free_slots,
                    capacity_margin=capacity_margin)
    base_total = total_cost(base)
    credit: dict = {}
    for okey in coord_costs_by_offer:
        cf = matching(coord_costs_by_offer, free_slots=free_slots,
                      capacity_margin=capacity_margin, exclude={okey})
        credit[okey] = total_cost(cf) - base_total
    return credit


# ------------------------------------------------------------------ 표 동결 보조
def freeze_coord_costs(orchestrator, mbt, t: float,
                       offers: list[tuple[str, str, str]]) -> tuple[dict, dict]:
    """실제 resolver 의 좌표 비용표를 **동결 시점 1회** 로 떠서 위 함수들에 넘긴다.

    resolver 와 같은 함수(`_coord_costs`)를 그대로 호출하므로 비용 정의가 갈라지지
    않는다. 가상 상태 q 는 resolver 의 동결 관측(내부 대수 + 통지 pipeline)과 동일하고,
    vcap 은 빈 상태(=배정 전) 다 — 이 시점의 표가 "동결 1벌"이다.

    Returns: (coord_costs_by_offer, free_slots)
        offer_key = (src, jid) 튜플. free_slots = 매칭에 등장하는 블록의 슬롯 스냅샷.
    """
    from .sell_review import block_inside, block_pipeline   # 지연 import (단위시험 경량)
    q = {b: float(block_inside(mbt.blocks[b], t) + block_pipeline(mbt, b, t))
         for b in mbt.blocks}
    vcap: dict[str, int] = {}
    table: dict = {}
    for src, jid, flow in offers:
        table[(src, jid)] = list(
            orchestrator._coord_costs(mbt, src, jid, flow, t, q, vcap))
    free = {b: int(mbt.free_slots(b)) for b in mbt.blocks}
    return table, free


# ================================================================== 단위 시험
if __name__ == "__main__":
    def _show(name, got, want=None):
        ok = "OK " if (want is None or got == want) else "FAIL"
        print(f"[{ok}] {name}: {got}" + (f"  (기대 {want})" if want is not None else ""))
        assert want is None or got == want, name

    print("=" * 70)
    print("① 매칭이 최소비용 좌표를 고르는가 (용량 여유 충분)")
    # offer 3개 — 각자 자기 최소 좌표를 가져갈 수 있어야 한다.
    t1 = {
        "j1": [(0.0, "KEEP"), (-1.0, "Y02"), (-0.4, "Y03")],
        "j2": [(0.0, "KEEP"), (-0.7, "Y02"), (-0.9, "Y03")],
        "j3": [(0.0, "KEEP"), (-0.2, "Y02"), (-0.1, "Y03"), (-0.5, "TIME")],
    }
    fs1 = {"Y02": 10, "Y03": 10}
    m1 = matching(t1, free_slots=fs1, capacity_margin=2)
    _show("배정(전역 최소 쌍 순)", m1,
          [("j1", "Y02", -1.0), ("j2", "Y03", -0.9), ("j3", "TIME", -0.5)])
    _show("총비용", round(total_cost(m1), 6), -2.4)

    print("-" * 70)
    print("①-b 순열 불변 — 입력 dict 순서를 바꿔도 같은 결과인가")
    t1b = {k: t1[k] for k in ("j3", "j1", "j2")}
    m1b = matching(t1b, free_slots=fs1, capacity_margin=2)
    _show("순서 뒤집은 표의 배정 집합", sorted(m1b), sorted(m1))

    print("-" * 70)
    print("①-c 동점 파훼 — 같은 비용이면 (cost, offer_key, coord) 사전순")
    t1c = {"jB": [(-1.0, "Y02"), (0.0, "KEEP")],
           "jA": [(-1.0, "Y02"), (0.0, "KEEP")]}
    m1c = matching(t1c, free_slots={"Y02": 3}, capacity_margin=2)   # 여유 3-0>2 → 1대만
    _show("먼저 잡는 offer", m1c[0][0], "jA")

    print("=" * 70)
    print("② 용량 1인 블록에 둘이 몰리면 하나만 배정되는가")
    # free=3, margin=2 → 여유 1대분. 둘 다 Y05 를 원하지만 한 대만 들어간다.
    t2 = {"jx": [(0.0, "KEEP"), (-3.0, "Y05")],
          "jy": [(0.0, "KEEP"), (-2.5, "Y05")]}
    m2 = matching(t2, free_slots={"Y05": 3}, capacity_margin=2)
    _show("배정", m2, [("jx", "Y05", -3.0), ("jy", "KEEP", 0.0)])
    n_y05 = sum(1 for _, c, _ in m2 if c == "Y05")
    _show("Y05 배정 수", n_y05, 1)

    print("=" * 70)
    print("③ 모든 좌표가 KEEP(0) 보다 비싸면 전부 KEEP 인가")
    t3 = {"ja": [(0.0, "KEEP"), (0.4, "Y02"), (1.2, "Y03"), (0.9, "TIME")],
          "jb": [(0.0, "KEEP"), (0.1, "Y02"), (2.0, "Y03")]}
    m3 = matching(t3, free_slots={"Y02": 10, "Y03": 10}, capacity_margin=2)
    _show("배정", sorted(m3), [("ja", "KEEP", 0.0), ("jb", "KEEP", 0.0)])
    _show("총비용", total_cost(m3), 0.0)

    print("=" * 70)
    print("④ 차이 공로가 '없었으면 남이 그 슬롯을 썼다'를 반영하는가")
    # jx 는 자기 이득 3.0 이지만, 없었으면 jy 가 같은 슬롯에서 2.5 를 챙겼을 것이다.
    #   전체:  jx→Y05(-3.0), jy→KEEP(0)          총 -3.0
    #   jx 제외: jy→Y05(-2.5)                     총 -2.5
    #   D_jx = -2.5 - (-3.0) = +0.5  ≪ 자기 이득 3.0
    d4 = difference_credit(t2, free_slots={"Y05": 3}, capacity_margin=2)
    _show("D(jx) — 대체 가능분을 뺀 순공로", round(d4["jx"], 6), 0.5)
    _show("자기 비용 기준의 순진한 공로", 3.0)
    _show("D(jy) — 어차피 못 쓴 offer", round(d4["jy"], 6), 0.0)

    print("-" * 70)
    print("④-b 용량이 넉넉하면 공로 = 자기 이득 (대체 효과 없음)")
    d4b = difference_credit(t2, free_slots={"Y05": 10}, capacity_margin=2)
    _show("D(jx)", round(d4b["jx"], 6), 3.0)
    _show("D(jy)", round(d4b["jy"], 6), 2.5)

    print("-" * 70)
    print("④-c 손해 거래는 음수 공로 — allow_keep 이 없으면 강제 배정된다")
    t4c = {"jz": [(0.8, "Y07")], "jw": [(0.0, "KEEP")]}
    d4c = difference_credit(t4c, free_slots={"Y07": 10}, capacity_margin=2)
    _show("D(jz)", round(d4c["jz"], 6), -0.8)

    print("=" * 70)
    print("⑤ fail-fast — 용량을 모르는 좌표는 즉시 실격")
    try:
        matching({"j": [(0.0, "Y99")]}, free_slots={}, capacity_margin=2)
        _show("KeyError 발생", False, True)
    except KeyError as e:
        _show("KeyError 발생", True, True)
        print(f"      메시지: {str(e)[:80]}")

    print("=" * 70)
    print("모든 단위 시험 통과")
