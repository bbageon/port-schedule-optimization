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
    """블록으로 오기로 **통지된**(미진입) 트럭 수 — 공개 정보. 이송되면 owner 가 옮겨간다."""
    return sum(1 for rec in mbt.ledger.records.values()
               if rec.owner == bid and rec.a_gate_in is not None
               and rec.a_gate_in > t + 1e-9)


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
                               max_deferrals=MAX_ENTRY_DEFERRALS)
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
                 window_s: float = WINDOW_S, defer_delta_s: float = DEFER_DELTA_S):
        if kf is None:
            raise ValueError("UnifiedSellOrchestrator 는 비용 통화(kf)가 필수 — "
                             "축 비교는 단일 통화 위에서만 성립한다")
        self.policy = policy
        self.layout = layout
        self.kf = kf
        self.window_s = window_s
        self.defer_delta_s = defer_delta_s
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
    def _coord_costs(self, mbt, src: str, jid: str, flow: str, t: float,
                     q: dict) -> list[tuple[float, str]]:
        from .sell_gain import cost_given_queue
        rec = mbt.ledger.records[jid]
        j = mbt.blocks[src].jobs[jid]
        eta = getattr(j, "estimated_block_arrival", None) or j.provided_eta
        relief = cost_given_queue(mbt.blocks[src], gate_in_s=rec.a_gate_in,
                                  eta_s=eta, queue=q[src], kf=self.kf)
        out: list[tuple[float, str]] = []
        if flow == "GATE_IN":                          # 공간 대안 20곳
            for dst in mbt.blocks:
                if dst == src:
                    continue
                delta = self.layout.pre_gate_route_delta_s(src, dst)
                add = cost_given_queue(mbt.blocks[dst], gate_in_s=rec.a_gate_in,
                                       eta_s=eta + delta, queue=q[dst], kf=self.kf)
                out.append((add + delta / 3600.0 - relief, dst))
        # 시간 대안 (+Δ) — 반입·반출 공통. 이연 중 대기열 배수는 결정론 proxy.
        d = self.defer_delta_s
        sim = mbt.blocks[src]
        n_cranes = max(1, len(sim.profile.cranes))
        from .block_congestion import SVC_REF_S
        q_late = max(0.0, q[src] - d * n_cranes / SVC_REF_S)
        add_t = cost_given_queue(sim, gate_in_s=rec.a_gate_in + d, eta_s=eta + d,
                                 queue=q_late, kf=self.kf)
        out.append((add_t + d / 3600.0 - relief, "TIME"))   # 외부 대기 = 동일 단가
        return out

    def review(self, mbt, t: float) -> None:
        offers = self._collect(mbt, t)
        if not offers:
            return
        # ---- ② 중앙 matching: 전역 한계비용 최소 (제안 × 좌표) 쌍 반복 선택
        q = {b: float(block_inside(mbt.blocks[b], t) + block_pipeline(mbt, b, t))
             for b in mbt.blocks}
        remaining = list(offers)
        assignment: list[tuple[str, str, str, str]] = []    # (src, jid, flow, coord)
        while remaining:
            best = None
            for idx, (src, jid, flow) in enumerate(remaining):
                for cost, coord in self._coord_costs(mbt, src, jid, flow, t, q):
                    key = (cost, jid, coord)
                    if best is None or key < best[0]:
                        best = (key, idx, coord)
            _, idx, coord = best
            src, jid, flow = remaining.pop(idx)
            assignment.append((src, jid, flow, coord))
            if coord != "TIME":                       # 공간: 가상 상태 이동
                q[coord] += 1.0
                q[src] = max(0.0, q[src] - 1.0)
            # 시간: 블록 간 이동 없음 — 가상 상태 불변(이연 배수는 좌표 비용에 반영)
        # ---- ③ 원자 확정
        for src, jid, flow, coord in assignment:
            if coord == "TIME":
                ok = try_time_sell(mbt, jid, delta_s=self.defer_delta_s,
                                   max_deferrals=MAX_ENTRY_DEFERRALS)
                self.ledger.append({"t": t, "axis": "TIME", "src": src, "job_id": jid,
                                    "flow": flow,
                                    "decision": "DEFER" if ok else "KEEP_TXN_FAIL"})
                self.n_time += 1 if ok else 0
            else:
                ok = mbt.try_pre_gate_transfer(
                    jid, coord, travel_s=self.layout.gate_to_block_s(coord),
                    route_delta_s=self.layout.pre_gate_route_delta_s(src, coord))
                self.ledger.append({"t": t, "axis": "SPACE", "src": src, "job_id": jid,
                                    "flow": flow, "dst": coord if ok else None,
                                    "decision": "SELL" if ok else "KEEP_TXN_FAIL"})
                self.n_space += 1 if ok else 0


class KeepAllUnified:
    """기준선 K (통합 인터페이스) — 아무것도 덜어내지 않는다."""

    def decide(self, mbt, src: str, cands: list, t: float) -> str | None:
        return None
