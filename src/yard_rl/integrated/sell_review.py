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
    """공간 판매의 **중앙 일괄 matching** — 전 블록 제안을 모아 한 번에 배정한다.

    ★재설계(사용자 지적 2026-08-09): 구판은 블록별로 결정 즉시 확정해서 **블록 이름
    순서가 결과에 영향**을 줬다(앞 블록의 확정을 뒤 블록이 관측 — epoch 안 정보 비대칭).
    기존 post-gate 계산 resolver(transfer_quote)가 이미 "수집→정렬→일괄 matching"인데
    새 계층이 그 패턴을 어긴 구현 결함이었다. 이제:

      ① 동결 관측 위에서 전 제안 수집(결정 단계 commit 0건)
      ② 우선순위 = **혼잡 필요 기반**(소스 부하 내림차순 → 작업 id) — 이름 순서 아님
      ③ 가상 원장(내부+통지 pipeline)을 갱신하며 수신 배정 — 같은 epoch 안 정합
      ④ 엔진 원자 확정(fail-closed) — stale·용량·상한은 엔진이 최종 검증

    부담 근사는 공개 관측값(내부 대수 + 통지 pipeline)만 쓴다 — 실현 미래값 미열람.
    """

    def __init__(self, layout: YardLayout):
        self.layout = layout

    def match_all(self, mbt, t: float,
                  offers: list[tuple[str, str]]) -> list[dict]:
        # 동결 관측 1회 — 이후 배정은 이 가상 원장 위에서만 갱신된다
        load = {b: block_inside(mbt.blocks[b], t) + block_pipeline(mbt, b, t)
                for b in mbt.blocks}
        order = sorted(offers, key=lambda o: (-load[o[0]], o[1]))
        results = []
        for src, jid in order:
            best = None
            for dst in sorted(b for b in mbt.blocks if b != src):
                key = (load[dst], self.layout.pre_gate_route_delta_s(src, dst), dst)
                if best is None or key < best[0]:
                    best = (key, dst)
            dst = best[1]
            ok = mbt.try_pre_gate_transfer(
                jid, dst, travel_s=self.layout.gate_to_block_s(dst),
                route_delta_s=self.layout.pre_gate_route_delta_s(src, dst))
            if ok:
                load[dst] += 1              # 가상 반영 — 다음 배정이 같은 epoch 안 정합
                load[src] -= 1
            results.append({"src": src, "job_id": jid,
                            "dst": dst if ok else None, "committed": ok})
        return results


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
