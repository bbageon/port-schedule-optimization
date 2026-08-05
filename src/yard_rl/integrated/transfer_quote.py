"""블록 간 판매 발의·수신부담 견적·결정론 확정 — YR-133 1차 (event-only·반입·1건/epoch).

계약 (spec YR-133 1차 동결):
- source: 그 epoch gate-in 재배정 가능 후보 중 **top-1 OFFER**(OutRelief 최대·tie jid 순).
  OutRelief = v2 예측 KEEP 비용 j_truck(Ô_src+bias, A, D_T, κ_T) — 공개 정보만.
- receiver: 허용 블록(타 블록 전부) 각각 **InBurden** = 가상 도착(A+이동+우회) 기준
  수신 큐 proxy Ô_dst 의 j_truck — 최소 InBurden·tie 블록명순. 예측 결측 = NO_BID.
- resolver: NetGain = OutRelief − InBurden − route/3600 − gain_margin > 0 ∧ 본선 가드
  → MultiBlockTerminal.try_transfer, **epoch당 최대 1건**. 실패·결측·가드 = KEEP.
- quote 는 발행 epoch 전용(이월 금지 — 만료 원천 차단). **이송/작업 ≤ max_transfers**.
- 견적 원장 전량 기록(감사 가능). 결정론: 블록·후보 정렬 순회, 무작위는 travel_fn 주입.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cost_curve_v2 import (GATE_BLOCK_MEAN_S, SVC_REF_S, KappaFit, j_truck,
                            observed_gate_in, predict_gate_out, truck_target_s)

ROUTE_S_DEFAULT = 180.0
# ★정정(2026-08-04, 1차 파일럿 원장): yr099 승계값 0.5 는 "에피소드 전체 비용 차" 단위
# 맥락이라 "작업 1건 예측 비용"(중앙 0.36~0.40) 견적에서는 구조적 발화 불가.
# 정정 도출 = κ_T 예측오차 1σ (383.7s ≈ 0.107 비용단위) — 예측오차 이내 이득 무시
# (churn 방지 목적 동일·파일럿 원장 수치에 맞춘 값 아님). gain_margin=None 이면 κ 유도.
GAIN_MARGIN_DEFAULT = None
VESSEL_SLACK_MIN_DEFAULT = 0.0     # yr105 본선 보호 경계 승계 (의미 경계·튜닝 아님)


@dataclass(frozen=True)
class TransferOffer:
    job_id: str
    src: str
    out_relief: float              # KEEP 예상 비용 (내보내면 덜어질 몫)
    version: int
    t: float                       # 발행 epoch — 이 시각 밖에서는 무효


@dataclass(frozen=True)
class ReceiveQuote:
    job_id: str
    dst: str
    in_burden: float               # 받으면 생길 예상 비용
    version: int
    t: float


def predict_keep_cost(sim, jid: str, kf: KappaFit) -> float | None:
    """source 블록에 남을 때의 v2 예측 트럭 비용 — 결측 시 None (fail-closed)."""
    j = sim.jobs.get(jid)
    if j is None:
        return None
    a = observed_gate_in(sim, j)
    o_hat = predict_gate_out(sim, jid)
    if a is None or o_hat is None:
        return None
    return j_truck(o_hat + kf.bias_t_s, a, truck_target_s(sim, a), kf.kappa_t_s)


def predict_move_cost(src_sim, dst_sim, jid: str, kf: KappaFit, *,
                      travel_s: float, route_s: float) -> float | None:
    """수신 블록으로 이송됐을 때의 v2 예측 트럭 비용 — 공개 정보만.

    가상 도착 t_arr = A + 이동 + 우회. Ô_dst = predict_gate_out 의 미도착 분기와 같은
    큐 proxy (수신 크레인 잔여 + 대기 트럭 수 × 기준 서비스). 실현(actual) 미열람."""
    j = src_sim.jobs.get(jid)
    if j is None:
        return None
    a = observed_gate_in(src_sim, j)
    if a is None:
        return None
    t_arr = a + travel_s + route_s
    crane_ids = [c.crane_id for c in dst_sim.profile.cranes]
    now = dst_sim.now
    run_rem = sum(max(0.0, dst_sim.fleet.get(c).state.available_at - now)
                  for c in crane_ids)
    ahead = sum(1 for o in dst_sim.jobs.values()
                if getattr(o, "is_external_truck", False) and o.status.name == "WAITING")
    t0 = max(now, t_arr)
    wait_pred = (run_rem + SVC_REF_S * ahead) / max(1, len(crane_ids))
    o_hat = t0 + wait_pred + SVC_REF_S + GATE_BLOCK_MEAN_S
    # D_T 는 원 게이트 진입 기준 — 이송해도 불변 (spec: gate 를 origin 으로)
    return j_truck(o_hat + kf.bias_t_s, a, truck_target_s(src_sim, a), kf.kappa_t_s)


class TransferQuoteResolver:
    """결정론 블록 간 확정자 — MultiBlockTerminal.run 의 review_fn 으로 주입."""

    def __init__(self, kf: KappaFit, *, travel_fn, vessel_slack_fn=None,
                 route_s: float = ROUTE_S_DEFAULT,
                 gain_margin: float | None = GAIN_MARGIN_DEFAULT,
                 vessel_slack_min: float = VESSEL_SLACK_MIN_DEFAULT,
                 max_transfers: int = 1):
        self.kf = kf
        self.travel_fn = travel_fn                  # (src, jid) -> travel_s (하네스 결정론 rng)
        self.vessel_slack_fn = vessel_slack_fn      # (sim) -> float | None (본선 가드)
        self.route_s = route_s
        # None = κ_T 1σ 유도 (예측오차 이내 이득 무시 — 정정 도출, 상수 튜닝 아님)
        self.gain_margin = (kf.kappa_t_s / 3600.0 if gain_margin is None
                            else gain_margin)
        self.vessel_slack_min = vessel_slack_min
        self.max_transfers = max_transfers
        self.ledger: list[dict] = []                # 견적 원장 (감사 가능·epoch 전용 quote)
        self.n_transferred = 0

    def review(self, mbt, t: float) -> None:
        """★YR-149 선결 정렬: 소스 이름순 첫 양수가 아니라 **같은 epoch 허용 제안 전체
        중 최대 NetGain 1건**을 고르고, commit 직전 quote 발행 version 과 현재 version
        일치를 직접 검사한다 (불일치 = KEEP_STALE_VERSION)."""
        proposals = []                              # (net, offer, quote, travel)
        for src in sorted(mbt.blocks):
            src_sim = mbt.blocks[src]
            # 본선 가드 — 소스 LOAD 최소 slack 이 경계 미만이면 이 블록 발의 금지
            if self.vessel_slack_fn is not None:
                sl = self.vessel_slack_fn(src_sim)
                if sl is not None and sl < self.vessel_slack_min:
                    self.ledger.append({"t": t, "src": src, "decision": "BLOCKED_VESSEL",
                                        "slack_s": sl})
                    continue
            cands = sorted(
                jid for jid, rec in mbt.ledger.records.items()
                if rec.owner == src and rec.reassignable
                and rec.transfer_count < self.max_transfers      # 이송/작업 상한 mask
                and rec.a_gate_in is not None and abs(rec.a_gate_in - t) <= 1e-6)
            if not cands:
                continue
            # ① source top-1 OFFER — OutRelief 최대 (tie: jid 사전순 = 정렬 순회로 보장)
            offers = []
            for jid in cands:
                relief = predict_keep_cost(src_sim, jid, self.kf)
                if relief is None:
                    self.ledger.append({"t": t, "src": src, "job_id": jid,
                                        "decision": "KEEP_NO_PREDICT"})
                    continue
                offers.append(TransferOffer(jid, src, relief,
                                            mbt.ledger.records[jid].version, t))
            if not offers:
                continue
            offer = max(offers, key=lambda o: (o.out_relief, o.job_id))
            # ② receiver 견적 — 허용 블록 = 타 블록 전부, 최소 InBurden (tie 블록명순)
            quotes = []
            for dst in sorted(b for b in mbt.blocks if b != src):
                travel = self.travel_fn(src, offer.job_id)
                burden = predict_move_cost(src_sim, mbt.blocks[dst], offer.job_id,
                                           self.kf, travel_s=travel,
                                           route_s=self.route_s)
                if burden is None:
                    self.ledger.append({"t": t, "src": src, "dst": dst,
                                        "job_id": offer.job_id, "decision": "NO_BID"})
                    continue
                quotes.append((ReceiveQuote(offer.job_id, dst, burden,
                                            offer.version, t), travel))
            if not quotes:
                continue
            quote, travel = min(quotes, key=lambda q: (q[0].in_burden, q[0].dst))
            net = (offer.out_relief - quote.in_burden
                   - self.route_s / 3600.0 - self.gain_margin)
            rec = {"t": t, "src": src, "dst": quote.dst, "job_id": offer.job_id,
                   "out_relief": offer.out_relief, "in_burden": quote.in_burden,
                   "net_gain": net, "version": offer.version}
            if net <= 0.0:
                self.ledger.append({**rec, "decision": "KEEP"})
                continue
            proposals.append((net, offer, quote, travel, rec))
        if not proposals:
            return
        # ③ 결정론 확정 — 전 제안 중 최대 NetGain 1건 (tie: 작업 id 사전순)
        proposals.sort(key=lambda p: (-p[0], p[1].job_id))
        for i, (net, offer, quote, travel, rec) in enumerate(proposals):
            if i > 0:                               # epoch당 최대 1건 — 잔여는 KEEP 기록
                self.ledger.append({**rec, "decision": "KEEP_NOT_SELECTED"})
                continue
            cur_ver = mbt.ledger.records[offer.job_id].version
            if cur_ver != offer.version:            # quote 발행 vs 확정 version 직접 검사
                self.ledger.append({**rec, "decision": "KEEP_STALE_VERSION",
                                    "version_at_commit": cur_ver})
                continue
            ok = mbt.try_transfer(offer.job_id, quote.dst,
                                  route_s=self.route_s, travel_s=travel)
            self.ledger.append({**rec,
                                "decision": "TRANSFER" if ok else "KEEP_TXN_FAIL"})
            if ok:
                self.n_transferred += 1
