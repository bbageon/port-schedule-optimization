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


# ------------------------------------------------------------------ 수신 블록 선정
class PreGateResolver:
    """공간 판매의 수신 블록 선정 — 결정론: (수신 내부 대수, 진입 전 주행 차이, 이름) 최소.

    YR-151 계약대로 최종 선택권은 resolver 에 있다. 부담 근사는 **공개 관측값**(내부
    대수)만 쓴다 — 실현 미래값 미열람. commit 은 엔진 원자 이송(try_pre_gate_transfer).
    """

    def __init__(self, layout: YardLayout):
        self.layout = layout

    def place(self, mbt, src: str, job_id: str, t: float) -> str | None:
        best = None
        for dst in sorted(b for b in mbt.blocks if b != src):
            key = (block_inside(mbt.blocks[dst], t),
                   self.layout.pre_gate_route_delta_s(src, dst), dst)
            if best is None or key < best[0]:
                best = (key, dst)
        if best is None:
            return None
        dst = best[1]
        ok = mbt.try_pre_gate_transfer(
            job_id, dst, travel_s=self.layout.gate_to_block_s(dst),
            route_delta_s=self.layout.pre_gate_route_delta_s(src, dst))
        return dst if ok else None


# ------------------------------------------------------------------ orchestrator
class SellReviewOrchestrator:
    """epoch 마다 블록별로 공간(반입)·시간(반출) 판매를 검토·실행하고 전량 기록한다."""

    def __init__(self, policy, layout: YardLayout, *,
                 enable_space: bool = True, enable_time: bool = True,
                 window_s: float = WINDOW_S, defer_delta_s: float = DEFER_DELTA_S):
        self.policy = policy
        self.resolver = PreGateResolver(layout)
        self.enable_space = enable_space
        self.enable_time = enable_time
        self.window_s = window_s
        self.defer_delta_s = defer_delta_s
        self.ledger: list[dict] = []
        self.n_space = 0
        self.n_time = 0

    def review(self, mbt, t: float) -> None:
        for src in sorted(mbt.blocks):
            if self.enable_space:
                cands = iter_pre_gate_candidates(mbt, src, horizon_s=self.window_s,
                                                 max_transfers=MAX_TRANSFERS)
                if cands:
                    pick = self.policy.decide_space(mbt, src, cands, t)
                    if pick is not None:
                        dst = self.resolver.place(mbt, src, pick, t)
                        self.ledger.append({"t": t, "axis": "SPACE", "src": src,
                                            "job_id": pick, "dst": dst,
                                            "decision": "SELL" if dst else "KEEP_TXN_FAIL"})
                        if dst:
                            self.n_space += 1
            if self.enable_time:
                cands = iter_time_sell_candidates(mbt, src, horizon_s=self.window_s,
                                                  max_deferrals=MAX_ENTRY_DEFERRALS)
                if cands:
                    pick = self.policy.decide_time(mbt, src, cands, t)
                    if pick is not None:
                        ok = try_time_sell(mbt, pick, delta_s=self.defer_delta_s,
                                           max_deferrals=MAX_ENTRY_DEFERRALS)
                        self.ledger.append({"t": t, "axis": "TIME", "src": src,
                                            "job_id": pick,
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
