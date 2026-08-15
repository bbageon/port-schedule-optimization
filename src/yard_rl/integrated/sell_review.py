"""YR-151/161 — 판매 검토 orchestration: 사전 통지(lead) 창에서 공간·시간 판매 결정.

■ 구조 (사용자 확정 축: 반입 = 공간 · 반출 = 시간 · 양하/적하 = 제외)

    투입 통지(lead>0)          진입(gate-in)            진출(gate-out)
        │◄────── 판매 결정 창 ──────►│
        │  반입: KEEP / SELL(목적지 블록 변경 — pre_gate 원자 이송)
        │  반출: KEEP / DEFER(재예약 "다른 시각에 와 주세요" — 전원 수락 가정)
        └─ 검토는 60초 admission epoch 위에서 함께 돈다 (CompositeReview)

■ 왜 lead 가 필요한가
walk-in(lead 0)은 투입 즉시 gate-in 이라 **진입 전 결정 창이 없다** — 공간·시간 판매
모두 후보 0. `ANNOUNCE_LEAD_S`(30분 = YR-151 판매 검토 창과 동일)를 두면 통지에서
진입까지의 구간이 결정 창이 된다.

■ 고정 WIP 와의 상호작용 (0B 사전등록 시 동결 필수 — 정직 고지)
lead>0 이면 컨트롤러 목표(내부+pipeline = L)에서 **물리적 내부 대수는 L 보다 작아진다**
(진행 중인 통지분이 pipeline 에 잠김 — 대략 이탈률×lead 만큼). 0B 사전등록은
①"확약 총량 = L" 해석을 받아들이거나 ②L 을 그만큼 올려 동결하거나를 **결과 열람 전에**
정해야 한다. 여기서는 배선만 제공하고 해석을 정하지 않는다.

■ 정책 계층
`SellPolicy` 인터페이스가 결정을 내리고 orchestrator 는 실행·원장 기록만 한다.
- `KeepAll`     — 기준선 K (판매 없음)
- `CalcGreedy`  — 계산 기준선 자리(결정론) — PPO 없이 배선을 끝까지 돌리기 위한 최소
                  규칙. **성능 주장에 쓰는 Q30 계산 기준선은 별도 동결본으로 등록한다.**
- (예정) YR-151 PPO TransferHead 가 같은 인터페이스로 꽂힌다.
■ 테스트 유예 (빌드 우선 지시 2026-08-08) — 디버깅 국면에서 계약 테스트 몰아서.
"""
from __future__ import annotations

from .pre_gate import MAX_TRANSFERS, WINDOW_S, iter_pre_gate_candidates
from .time_sell import (DEFER_DELTA_S, MAX_ENTRY_DEFERRALS,
                        iter_time_sell_candidates, try_time_sell)
from .yard_layout import YardLayout

ANNOUNCE_LEAD_S = 1_800.0     # 사전 통지 = 판매 검토 창 30분 (YR-151 계약과 동일)


def block_inside(sim, t: float) -> int:
    """블록의 현재 내부 외부트럭 수 (A ≤ t < O) — 공개 관측값."""
    tl = getattr(sim, "time_ledger", None)
    if tl is None:
        return 0
    return sum(1 for r in tl.records.values()
               if r.gate_in <= t + 1e-9 and (r.gate_out is None or r.gate_out > t))


# ------------------------------------------------------------------ 정책 계층
class KeepAll:
    """기준선 K — 아무것도 팔지 않는다."""

    def decide_space(self, mbt, src: str, cands: list, t: float) -> str | None:
        return None

    def decide_time(self, mbt, src: str, cands: list, t: float) -> str | None:
        return None


class CalcGreedy:
    """계산 기준선 **자리**(결정론) — 배선 검증용 최소 규칙.

    자기 블록 내부 대수가 타 블록 최소보다 `pressure_gap` 이상 크면:
      공간(반입): 공개 ETA 가 가장 이른 후보 1건 OFFER
      시간(반출): 공개 ETA 가 가장 이른 후보 1건 재예약
    성능 판정용 Q30 계산 기준선은 이 클래스가 아니라 별도 동결본으로 등록해야 한다.
    """

    def __init__(self, pressure_gap: int = 3):
        self.pressure_gap = pressure_gap

    def _pressed(self, mbt, src: str, t: float) -> bool:
        mine = block_inside(mbt.blocks[src], t)
        others = [block_inside(s, t) for b, s in mbt.blocks.items() if b != src]
        return bool(others) and mine - min(others) >= self.pressure_gap

    def decide_space(self, mbt, src: str, cands: list, t: float) -> str | None:
        if not cands or not self._pressed(mbt, src, t):
            return None
        return min(cands, key=lambda c: (c[1], c[0]))[0]

    def decide_time(self, mbt, src: str, cands: list, t: float) -> str | None:
        if not cands or not self._pressed(mbt, src, t):
            return None
        return min(cands, key=lambda c: (c[1], c[0]))[0]


def block_pipeline(mbt, bid: str, t: float) -> int:
    """블록으로 오기로 **통지된**(미진입) 트럭 수 — 공개 정보. 이송되면 owner 가 옮겨간다.

    정보경계 정정(감사 치명 6): 실현 `a_gate_in` 대신 **공개 통지 시각**을 읽는다 —
    준수오차 도입 시 누출 방지. 통지값이 없는 구작업만 원장값으로 fallback.
    """
    from .time_sell import notified_gate_in
    n = 0
    for jid, rec in mbt.ledger.records.items():
        if rec.owner != bid:
            continue
        sim = mbt.blocks.get(bid)
        j = sim.jobs.get(jid) if sim is not None else None
        gi = notified_gate_in(j) if j is not None else rec.a_gate_in
        if gi is not None and gi > t + 1e-9:
            n += 1
    return n


# 60초 격자 가드(감사 치명 5) — 공용 유틸로 이동(time_grid.py, 2026-08-09 감사:
# 투입 계층도 같은 가드가 필요한데 생성기가 판매 계층에 의존하면 안 된다).
# 기존 import 경로(sell_review.on_grid) 호환을 위해 재노출한다.
from .time_grid import on_grid  # noqa: E402  (재노출 — 기존 사용처 호환)


# ------------------------------------------------------------------ 중앙 matching
class PreGateResolver:
    """공간 판매의 **중앙 동시 matching** — 전 블록 제안을 모아 한 번에 배정한다.

    ★재설계 2회(사용자 지적 2026-08-09):
    1차 결함 — 블록별 결정 즉시 확정: 이름 순서가 결과에 영향(정보 비대칭). → 동결 수집.
    2차 결함 — matching 내부의 "혼잡 소스 우선" 줄 세우기: 비용 구조 **밖의** 순서
    규칙이라 결과가 그 규칙에 좌우됐다(사실상 순차 배정). → **전역 한계비용 최소 쌍
    반복 선택**으로 교체: 매 반복에서 남은 (제안 × 수신처) 전 조합 중 비용이 가장 싼
    쌍을 고른다. 외부 우선순위 규칙이 없고, **결과는 제안 목록의 순서와 무관**하다
    (순열 불변 — 검증 부채 테스트 항목). 부하 비용이 볼록(대수 증가)이므로 이 반복
    선택은 "총 한계비용 최소 배정"과 일치하는 표준 구조다.

    절차: ① 동결 관측 1회(가상 원장 = 내부 + 통지 pipeline) ② 가상 원장 위에서
    **순수 matching 완성**(엔진 미접촉) ③ 완성된 배정을 엔진 원자 확정(fail-closed —
    stale·용량·상한 최종 검증, 실패 건만 KEEP).

    ★비용 통화 matching(사용자 지적 2026-08-09 — "대수 +1/−1 만으로는 안 된다"):
    kf(비용계약 적합)를 주면 matching 기준이 **비용 단위**가 된다. 배정 때 가상
    대기열 상태를 +1/−1 갱신하고, 다음 반복의 비용을 그 상태에서 **재계산**한다 —
    비용은 더하고 빼는 장부가 아니라 혼잡 상태의 **볼록 함수**라서, 같은 1대라도
    붐비는 블록에 얹으면 한계비용이 더 크게 뛴다(대수 비교는 이를 놓친다).
    kf 없이 쓰면 (대수, 주행) proxy 로 동작한다(배선 확인용 — 성능 판정 금지).
    """

    def __init__(self, layout: YardLayout, kf=None):
        self.layout = layout
        self.kf = kf

    def _pair_key(self, mbt, src: str, jid: str, dst: str, q: dict):
        """(제안, 수신처) 쌍의 한계비용 key — kf 있으면 비용 단위, 없으면 대수 proxy."""
        delta = self.layout.pre_gate_route_delta_s(src, dst)
        if self.kf is None:
            return (q[dst], delta, jid, dst)
        from .sell_gain import cost_given_queue
        rec = mbt.ledger.records[jid]
        j = mbt.blocks[src].jobs[jid]
        eta = getattr(j, "estimated_block_arrival", None) or j.provided_eta
        # 이 트럭을 dst 에 얹는 한계비용 − src 에서 덜어지는 한계비용 (둘 다 가상 상태)
        add = cost_given_queue(mbt.blocks[dst], gate_in_s=rec.a_gate_in,
                               eta_s=eta + delta, queue=q[dst], kf=self.kf)
        relief = cost_given_queue(mbt.blocks[src], gate_in_s=rec.a_gate_in,
                                  eta_s=eta, queue=q[src], kf=self.kf)
        net_cost = add + delta / 3600.0 - relief      # 작을수록(음수일수록) 좋은 배정
        return (net_cost, jid, dst)

    def match_all(self, mbt, t: float,
                  offers: list[tuple[str, str]]) -> list[dict]:
        # ── ① 동결 관측 1회 — matching 은 이 가상 대기열 상태 위에서만 진행된다
        q = {b: float(block_inside(mbt.blocks[b], t) + block_pipeline(mbt, b, t))
             for b in mbt.blocks}
        # ── ② 순수 matching: 전역 한계비용 최소 쌍을 반복 선택 (엔진 미접촉·순열 불변)
        remaining = list(offers)
        assignment: list[tuple[str, str, str]] = []          # (src, jid, dst)
        while remaining:
            best = None                     # (key, idx, dst)
            for idx, (src, jid) in enumerate(remaining):
                for dst in mbt.blocks:
                    if dst == src:
                        continue
                    key = self._pair_key(mbt, src, jid, dst, q)
                    if best is None or key < best[0]:
                        best = (key, idx, dst)
            _, idx, dst = best
            src, jid = remaining.pop(idx)
            assignment.append((src, jid, dst))
            q[dst] += 1.0                   # 상태 갱신 — 비용은 다음 반복에서 재계산된다
            q[src] = max(0.0, q[src] - 1.0)
        # ── ③ 완성된 배정을 원자 확정 — 실패 건만 KEEP (엔진 fail-closed)
        results = []
        for src, jid, dst in assignment:
            ok = mbt.try_pre_gate_transfer(
                jid, dst, travel_s=self.layout.gate_to_block_s(dst),
                route_delta_s=self.layout.pre_gate_route_delta_s(src, dst))
            results.append({"src": src, "job_id": jid,
                            "dst": dst if ok else None, "committed": ok})
        return results


# ------------------------------------------------------------------ orchestrator
class SellReviewOrchestrator:
    """epoch 마다 블록별로 공간(반입)·시간(반출) 판매를 검토·실행하고 전량 기록한다."""

    def __init__(self, policy, layout: YardLayout, *, kf=None,
                 enable_space: bool = True, enable_time: bool = True,
                 window_s: float = WINDOW_S, defer_delta_s: float = DEFER_DELTA_S):
        self.policy = policy
        self.resolver = PreGateResolver(layout, kf=kf)   # kf = 비용 통화 matching (권장)
        self.enable_space = enable_space
        self.enable_time = enable_time
        self.window_s = window_s
        self.defer_delta_s = defer_delta_s
        self.ledger: list[dict] = []
        self.n_space = 0
        self.n_time = 0

    def review(self, mbt, t: float) -> None:
        """★목표 구조(사용자 확정 2026-08-09): 동결 → 전 블록 수집 → 일괄 matching → 확정.

        ① 결정 단계에서는 **commit 이 0건**이다 — epoch 동안 시뮬 시간이 흐르지 않으므로
           "무변이 = 전 21블록이 같은 스냅샷을 본다"가 보장된다(순회 순서 무의미해짐).
        ② 공간 제안은 중앙 matching 이 한 번에 배정한다(우선순위 = 혼잡 필요 기반).
        ③ 시간 이연은 경합 자원이 없어 matching 이 불필요 — 동결 결정을 그대로 집행.
        """
        # ── ① 동결 관측 위에서 전 블록 결정 수집 (commit 없음)
        space_offers: list[tuple[str, str]] = []
        time_picks: list[tuple[str, str]] = []
        for src in sorted(mbt.blocks):          # 순서 무관 — 이 단계는 아무것도 바꾸지 않는다
            if self.enable_space:
                cands = iter_pre_gate_candidates(mbt, src, horizon_s=self.window_s,
                                                 max_transfers=MAX_TRANSFERS)
                if cands:
                    pick = self.policy.decide_space(mbt, src, cands, t)
                    if pick is not None:
                        space_offers.append((src, pick))
            if self.enable_time:
                cands = iter_time_sell_candidates(mbt, src, horizon_s=self.window_s,
                                                  max_deferrals=MAX_ENTRY_DEFERRALS)
                if cands:
                    pick = self.policy.decide_time(mbt, src, cands, t)
                    if pick is not None:
                        time_picks.append((src, pick))
        # ── ② 공간: 중앙 일괄 matching → 원자 확정
        if space_offers:
            for r in self.resolver.match_all(mbt, t, space_offers):
                self.ledger.append({"t": t, "axis": "SPACE", "src": r["src"],
                                    "job_id": r["job_id"], "dst": r["dst"],
                                    "decision": "SELL" if r["committed"]
                                    else "KEEP_TXN_FAIL"})
                if r["committed"]:
                    self.n_space += 1
        # ── ③ 시간: 동결 결정 집행 (블록 내 재예약 — 블록 간 경합 없음)
        for src, jid in time_picks:
            ok = try_time_sell(mbt, jid, delta_s=self.defer_delta_s,
                               max_deferrals=MAX_ENTRY_DEFERRALS, t=t)
            self.ledger.append({"t": t, "axis": "TIME", "src": src, "job_id": jid,
                                "decision": "DEFER" if ok else "KEEP_TXN_FAIL"})
            if ok:
                self.n_time += 1


class CompositeReview:
    """여러 review 콜러블을 순서대로 실행 — 투입 컨트롤러 다음에 판매 검토가 돈다."""

    def __init__(self, *reviews):
        self.reviews = [r for r in reviews if r is not None]

    def review(self, mbt, t: float) -> None:
        for r in self.reviews:
            r.review(mbt, t) if hasattr(r, "review") else r(mbt, t)


# ================================================================== B-(b) 통합 (2026-08-09)
class UnifiedSellOrchestrator:
    """★확정 구조 B-(b): PPO = 계획 진단("무엇을 덜어낼까") · resolver = 좌표 선택.

    ① 동결 수집 — 블록마다 반입(공간 자격)·반출(시간 자격) 후보의 **합집합**을 정책에
       주고 KEEP/OFFER 1건을 받는다(commit 0건 — 전 블록 같은 스냅샷).
    ② 중앙 matching — 제안마다 대안 좌표를 **한 저울(비용 통화)** 에 올린다:
         반입 제안: {다른 블록 20곳(공간)} ∪ {+Δ 이연(시간)}
         반출 제안: {+Δ 이연(시간)}  (공간 불가 — 컨테이너 위치 고정)
       전역 한계비용 최소 쌍 반복 선택·가상 상태 갱신·볼록 재계산·순열 불변.
    ③ 원자 확정 — 공간은 pre_gate 이송, 시간은 재예약. 실패 건만 KEEP.

    비용 통화(kf)가 **필수**다 — 축을 한 저울에 올리려면 단일 통화가 전제이므로
    proxy(대수) 모드를 허용하지 않는다(fail-fast).
    """

    def __init__(self, policy, layout: YardLayout, kf, *,
                 window_s: float = WINDOW_S, defer_delta_s: float = DEFER_DELTA_S,
                 grid_s: float = 60.0, allow_keep_coord: bool = True,
                 dry_run: bool = False,
                 time_slots: bool = False, buy_net=None):
        if kf is None:
            raise ValueError("UnifiedSellOrchestrator 는 비용 통화(kf)가 필수 — "
                             "축 비교는 단일 통화 위에서만 성립한다")
        # ★shadow 재정의(감사 2026-08-09): 즉시-KEEP 반환은 resolver 를 안 거쳐 견적·
        # matching·version 검사가 검증되지 않는다 — shadow 정책은 **dry_run resolver**
        # 와만 짝지을 수 있다(수집→저울→matching→용량까지 실제로 흐르고 원자 확정만
        # 생략). 짝이 어긋나면 조용히 다른 실험이 되므로 생성자에서 즉시 실격.
        if getattr(policy, "mode", None) == "shadow" and not dry_run:
            raise ValueError("shadow 정책은 dry_run=True resolver 와만 결합 가능 — "
                             "아니면 shadow 제안이 실제 commit 되어 환경을 오염시킨다")
        self.policy = policy
        self.layout = layout
        self.kf = kf
        self.window_s = window_s
        self.defer_delta_s = defer_delta_s
        # ★YR-171-C 시간 좌표 계약. 기본 False = 고정 +15분 한 칸(구 계약 보존).
        # True 면 오늘의 30분 슬롯 48칸에서 고른다 — 하루 공개 예약 장부가 붙어 있어야
        # 미래 칸이 비어 보이지 않는다(YR-171-A). 붙지 않았으면 즉시 실격.
        self.time_slots = bool(time_slots)
        self.buy_net = buy_net               # None = 결정론 계산식으로 견적
        self._plan_cache = None              # (t, order, index, plans) — epoch 공유
        self.grid_s = grid_s                 # 60초 계약 — 격자 밖 epoch 에서는 닫음
        self.allow_keep_coord = allow_keep_coord   # 현 좌표 유지 ΔJ=0 (0B 동결 대상)
        self.dry_run = dry_run
        self.ledger: list[dict] = []
        self.n_space = 0
        self.n_time = 0

    # ---- ① 동결 수집
    def _collect(self, mbt, t: float) -> list[tuple[str, str, str]]:
        offers = []                                   # (src, jid, flow)
        for src in sorted(mbt.blocks):
            space = iter_pre_gate_candidates(mbt, src, horizon_s=self.window_s,
                                             max_transfers=MAX_TRANSFERS)
            time_ = iter_time_sell_candidates(mbt, src, horizon_s=self.window_s,
                                              max_deferrals=MAX_ENTRY_DEFERRALS)
            cands = ([(jid, eta, "GATE_IN") for jid, eta in space]
                     + [(jid, eta, "GATE_OUT") for jid, eta in time_])
            if not cands:
                continue
            pick = self.policy.decide(mbt, src, cands, t)
            if pick is not None:
                flow = next(f for j, _, f in cands if j == pick)
                offers.append((src, pick, flow))
        return offers

    # ---- ② 축 저울: 제안 1건의 대안 좌표별 순비용 (가상 상태 q 위에서)
    def _require_day_plan(self, mbt) -> None:
        """48칸 모드는 하루 공개 예약 장부가 **필수**다 — 없으면 즉시 실격.

        장부 없이 48칸을 열면 미래 칸이 0.4~5.6% 만 채워져(yr171_horizon_probe)
        "먼 슬롯일수록 한가하다"를 배운다. 조용히 나쁜 실험이 되므로 fail-closed.
        """
        from .day_plan import get as _day_plan_get
        if _day_plan_get(mbt) is None:
            raise ValueError(
                "time_slots=True 는 day_plan(하루 공개 예약 장부)이 필수 — "
                "장부 없이 48칸을 열면 미래 칸이 비어 보여 먼 슬롯을 무조건 고른다")

    def _coord_costs(self, mbt, src: str, jid: str, flow: str, t: float,
                     q: dict, vcap: dict) -> list[tuple[float, str]]:
        from .sell_gain import cost_given_queue
        from .time_sell import notified_gate_in
        j = mbt.blocks[src].jobs[jid]
        gi = notified_gate_in(j)                       # 공개 통지 시각 (누출 금지 — 감사 6)
        eta = getattr(j, "estimated_block_arrival", None) or j.provided_eta
        relief = cost_given_queue(mbt.blocks[src], gate_in_s=gi,
                                  eta_s=eta, queue=q[src], kf=self.kf)
        out: list[tuple[float, str]] = []
        if self.allow_keep_coord:
            # 현 좌표 유지 ΔJ=0 — 모든 대안이 손해면 resolver 가 KEEP 을 고른다
            # (감사 구조 지적: 나쁜 SELL 강제 배정 방지. S vs Q30 오염 소지가 있어
            #  0B 사전등록에서 이 플래그를 동결해야 한다 — 양군 동일 적용이 조건).
            out.append((0.0, "KEEP"))
        if flow == "GATE_IN":                          # 공간 대안 — 용량은 matching 에서 검사
            for dst in mbt.blocks:
                if dst == src:
                    continue
                # 감사 구조 지적: matching 단계에서 용량을 본다 — 가상 배정분 포함
                if mbt.free_slots(dst) - vcap.get(dst, 0) <= mbt.capacity_margin:
                    continue
                delta = self.layout.pre_gate_route_delta_s(src, dst)
                add = cost_given_queue(mbt.blocks[dst], gate_in_s=gi,
                                       eta_s=eta + delta, queue=q[dst], kf=self.kf)
                out.append((add + delta / 3600.0 - relief, dst))
        # 시간 대안 — 반입·반출 공통. 이연 중 대기열 배수는 결정론 proxy.
        sim = mbt.blocks[src]
        n_cranes = max(1, len(sim.profile.cranes))
        from .block_congestion import SVC_REF_S
        if not self.time_slots:
            # 구 계약: 고정 +Δ 한 칸 (YR-171 이전)
            d = self.defer_delta_s
            q_late = max(0.0, q[src] - d * n_cranes / SVC_REF_S)
            add_t = cost_given_queue(sim, gate_in_s=gi + d, eta_s=eta + d,
                                     queue=q_late, kf=self.kf)
            out.append((add_t + d / 3600.0 - relief, "TIME"))  # 외부 대기 = 동일 단가
            return out
        # ★YR-171-C: 오늘의 30분 슬롯 중에서 고른다. 좌표 = "TIME@k" (절대 칸 번호).
        # 값은 BUY 견적망이 있으면 그 예측을, 없으면 같은 결정론 proxy 를 쓴다.
        from .slot_plan import N_SLOTS, SLOT_S
        cur = int(gi // SLOT_S)
        quotes = self._buy_quotes(mbt, src, jid, t) if self.buy_net else None
        for k in range(cur + 1, N_SLOTS):
            d = k * SLOT_S - gi                     # 그 칸 시작으로 재예약
            if d <= 0.0:
                continue
            if quotes is not None:
                add_t = float(quotes[k])
            else:
                q_late = max(0.0, q[src] - d * n_cranes / SVC_REF_S)
                add_t = cost_given_queue(sim, gate_in_s=gi + d, eta_s=eta + d,
                                         queue=q_late, kf=self.kf)
            out.append((add_t + d / 3600.0 - relief, f"TIME@{k}"))
        return out

    def _buy_quotes(self, mbt, src: str, jid: str, t: float):
        """BUY 견적망으로 (이 작업, src 블록, 48칸) 예상 부담을 한 번에 받는다.

        시간 판매는 **블록이 그대로**이므로 src 행만 쓴다. 계획표는 epoch 마다 한 번만
        만들어 `_plan_cache` 로 공유한다 — 블록마다 다시 만들면 비용이 폭발한다.
        """
        import torch
        from .buy_estimator import job_feature_batch
        cache = self._plan_cache
        if cache is None or cache[0] != t:
            from .slot_plan import terminal_slot_plan
            plan = terminal_slot_plan(mbt, t)
            order = sorted(plan)
            cache = (t, order, {b: i for i, b in enumerate(order)},
                     torch.tensor([plan[b] for b in order], dtype=torch.float32))
            self._plan_cache = cache
        _t, _order, bidx, plans = cache
        with torch.no_grad():
            jobs = job_feature_batch(mbt, [(src, jid)], t)
            out = self.buy_net(plans, jobs)          # (1, B, 48)
        return out[0, bidx[src]].tolist()

    def review(self, mbt, t: float) -> None:
        if not on_grid(t, self.grid_s):
            return                     # 60초 계약 — gate-in 시각 epoch 에서는 열지 않는다
        if self.time_slots:
            self._require_day_plan(mbt)
        offers = self._collect(mbt, t)
        if not offers:
            return
        # ---- ② 중앙 matching: 전역 한계비용 최소 (제안 × 좌표) 쌍 반복 선택
        q = {b: float(block_inside(mbt.blocks[b], t) + block_pipeline(mbt, b, t))
             for b in mbt.blocks}
        vcap: dict[str, int] = {}                          # 가상 배정 수 (용량 검사용)
        remaining = list(offers)
        # (src, jid, flow, coord, delta_j) — delta_j = 선택 좌표의 계산 순비용(음수 =
        # 이득). 0B 신호 판정의 원료라 원장에 함께 박제한다(기록 확장 — 행동 불변).
        assignment: list[tuple[str, str, str, str, float]] = []
        while remaining:
            best = None
            for idx, (src, jid, flow) in enumerate(remaining):
                for cost, coord in self._coord_costs(mbt, src, jid, flow, t, q, vcap):
                    key = (cost, jid, coord)
                    if best is None or key < best[0]:
                        best = (key, idx, coord)
            if best is None:
                break
            (cost, _, _), idx, coord = best
            src, jid, flow = remaining.pop(idx)
            assignment.append((src, jid, flow, coord, cost))
            # 시간 좌표는 "TIME"(구 계약) 또는 "TIME@k"(48칸) — 블록이 안 바뀐다
            if coord not in ("TIME", "KEEP") and not coord.startswith("TIME@"):
                q[coord] += 1.0
                q[src] = max(0.0, q[src] - 1.0)
                vcap[coord] = vcap.get(coord, 0) + 1
            # 시간: 블록 간 이동 없음 — 가상 상태 불변(이연 배수는 좌표 비용에 반영)
        # ---- ③ 원자 확정
        for src, jid, flow, coord, dj in assignment:
            if coord == "KEEP":
                self.ledger.append({"t": t, "axis": "KEEP", "src": src, "job_id": jid,
                                    "flow": flow, "decision": "RESOLVER_KEEP",
                                    "delta_j": round(dj, 6)})
                continue
            if self.dry_run:
                # shadow dry-run — 견적·matching·용량 검사를 전부 통과한 would-commit
                # 을 기록만 하고 확정을 생략한다(소유권·시간 장부·난수열 불변).
                is_time = coord == "TIME" or coord.startswith("TIME@")
                self.ledger.append({"t": t,
                                    "axis": "TIME" if is_time else "SPACE",
                                    "src": src, "job_id": jid, "flow": flow,
                                    "dst": None if is_time else coord,
                                    "decision": "DRY_WOULD_COMMIT",
                                    "delta_j": round(dj, 6)})
                continue
            if coord == "TIME" or coord.startswith("TIME@"):
                if coord == "TIME":
                    delta = self.defer_delta_s
                    slot = None
                else:
                    # "TIME@k" — k 번 칸 시작으로 재예약. 이연량은 지금 예약에서 유도.
                    from .slot_plan import SLOT_S
                    from .time_sell import notified_gate_in
                    slot = int(coord.split("@", 1)[1])
                    gi_now = notified_gate_in(mbt.blocks[src].jobs[jid])
                    delta = slot * SLOT_S - gi_now
                ok = (delta > 0.0) and try_time_sell(
                    mbt, jid, delta_s=delta,
                    max_deferrals=MAX_ENTRY_DEFERRALS, t=t)
                self.ledger.append({"t": t, "axis": "TIME", "src": src, "job_id": jid,
                                    "flow": flow, "slot": slot,
                                    "defer_s": round(delta, 3),
                                    "decision": "DEFER" if ok else "KEEP_TXN_FAIL",
                                    "delta_j": round(dj, 6)})
                self.n_time += 1 if ok else 0
            else:
                ok = mbt.try_pre_gate_transfer(
                    jid, coord, travel_s=self.layout.gate_to_block_s(coord),
                    route_delta_s=self.layout.pre_gate_route_delta_s(src, coord))
                self.ledger.append({"t": t, "axis": "SPACE", "src": src, "job_id": jid,
                                    "flow": flow, "dst": coord if ok else None,
                                    "decision": "SELL" if ok else "KEEP_TXN_FAIL",
                                    "delta_j": round(dj, 6)})
                self.n_space += 1 if ok else 0


class KeepAllUnified:
    """기준선 K (통합 인터페이스) — 아무것도 덜어내지 않는다."""

    def decide(self, mbt, src: str, cands: list, t: float) -> str | None:
        return None
