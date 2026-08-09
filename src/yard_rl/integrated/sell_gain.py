"""공간·시간 판매를 관통하는 단일 기준 — 순이득(NetGain) (사용자 질문 답 2026-08-08).

■ 기준 (양 축 공통)
    순이득 = J(현 계획) − J(대안 계획) − 전환 실비용 − 안전여유   > 0 이면 판다
    · 반입(공간): 대안 = (다른 블록 b', 같은 시각) · 전환비용 = 주행 차이(음수 가능)
    · 반출(시간): 대안 = (같은 블록, t+Δ)          · 전환비용 = 기사 외부 대기 Δ

작업의 계획은 (블록, 시각) 좌표쌍이고, 반입은 블록 좌표·반출은 시각 좌표를 바꿀 수
있을 뿐 — "바꿔서 이득이 남는가"는 같은 식이다.

■ 왜 이득이 존재하는가 — 비용의 볼록성
v2 트럭 비용은 대기에 **볼록**(SLA 꼬리 누진)이다. 같은 총 대기라도 몰리면 더 비싸므로
부하를 펴면(공간이든 시간이든) 총비용이 내려간다. 비용이 선형이라면 시간 판매는 원리상
이득이 없다(미룬 만큼 기사가 밖에서 기다리므로) — 볼록성이 이 기준의 존재 이유다.

■ 단위·여유 통일 (YR-133 정정 원칙 승계)
비용단위 = 트럭 1시간(기존 transfer_quote 와 동일 — route_s/3600). 기사 외부 대기도
**터미널 안 대기와 같은 단가**로 계상한다(단가를 낮추면 비용 은닉 통로가 된다).
안전여유 = κ_T 1σ(≈0.107 비용단위) — 예측오차 이내 이득 무시(churn 방지).

■ 정보 경계
전부 공개 정보만: 통지된 gate-in·공개 ETA·현재 내부 대수·**통지된** 미래 도착.
미래 대기열 예측은 결정론 배수(drain) + 통지 도착만으로 만든다 — 실현 미래값 미열람.

■ 지위: Q30 계열 계산 기준선의 **골격**이다. 성능 판정에 쓰려면 0B 사전등록에서
Δ·후보수·동결값을 결과 열람 전에 동결해야 한다. 테스트 유예(빌드 우선 지시).
"""
from __future__ import annotations

from .block_congestion import SVC_REF_S
from .cost_curve_v2 import GATE_BLOCK_MEAN_S, KappaFit, j_truck, truck_target_s
from .time_sell import DEFER_DELTA_S
from .yard_layout import YardLayout


# ------------------------------------------------------------------ 공개정보 예측
def public_queue_at(mbt, bid: str, at_s: float) -> float:
    """시각 at 의 예상 내부 트럭 수 — 공개 정보만의 결정론 예측.

    지금 내부 + 통지된 도착(공개) − 서비스 배수(크레인 수/기준 서비스시간 × 경과).
    실현 완료·실현 도착은 쓰지 않는다.
    """
    sim = mbt.blocks[bid]
    now = mbt.now
    tl = getattr(sim, "time_ledger", None)
    inside = announced = 0
    if tl is not None:
        for r in tl.records.values():
            if r.gate_in <= now + 1e-9:
                if r.gate_out is None or r.gate_out > now:
                    inside += 1
            elif r.gate_in <= at_s:
                announced += 1                       # 통지 도착 — 공개 정보
    n_cranes = max(1, len(sim.profile.cranes))
    drain = (at_s - now) * n_cranes / SVC_REF_S
    return max(0.0, inside + announced - drain)


def cost_given_queue(sim, *, gate_in_s: float, eta_s: float, queue: float,
                     kf: KappaFit) -> float:
    """대기열 상태를 **인자로 받아** v2 예상 비용을 계산한다 (볼록 j_truck 환산).

    matching 의 가상 원장이 상태(+1/−1)를 갱신하면 비용은 이 함수로 **재계산**된다 —
    비용은 더하고 빼는 장부가 아니라 혼잡 상태의 볼록 함수이므로, 같은 트럭 1대라도
    붐비는 블록에 얹으면 한계비용이 더 크게 뛴다(그 차이가 판매 이득의 원천).
    """
    n_cranes = max(1, len(sim.profile.cranes))
    wait_pred = max(0.0, queue) * SVC_REF_S / n_cranes
    o_hat = eta_s + wait_pred + SVC_REF_S + GATE_BLOCK_MEAN_S
    return j_truck(o_hat + kf.bias_t_s, gate_in_s, truck_target_s(sim, gate_in_s),
                   kf.kappa_t_s)


def expected_cost(mbt, bid: str, *, gate_in_s: float, eta_s: float,
                  kf: KappaFit) -> float:
    """공개정보 기준, 이 트럭이 (블록 bid, 도착 eta)로 갈 때의 v2 예상 비용.

    transfer_quote.predict_move_cost 와 같은 큐 proxy 골격 — 볼록 비용 j_truck 로 환산.
    """
    return cost_given_queue(mbt.blocks[bid], gate_in_s=gate_in_s, eta_s=eta_s,
                            queue=public_queue_at(mbt, bid, eta_s), kf=kf)


# ------------------------------------------------------------------ 통일 순이득
class UnifiedNetGain:
    """SellPolicy 구현 — 두 축 모두 순이득 > 0 인 후보 중 최대 1건을 고른다."""

    def __init__(self, kf: KappaFit, layout: YardLayout, *,
                 defer_delta_s: float = DEFER_DELTA_S,
                 gain_margin: float | None = None):
        self.kf = kf
        self.layout = layout
        self.defer_delta_s = defer_delta_s
        # None = κ_T 1σ 유도(YR-133 정정 — 상수 튜닝 아님)
        self.gain_margin = (kf.kappa_t_s / 3600.0 if gain_margin is None
                            else gain_margin)
        self.ledger: list[dict] = []

    # ---- 반입(공간): J(현 블록) − min_{b'}[J(b') + 주행차/3600] − 여유
    def decide_space(self, mbt, src: str, cands: list, t: float) -> str | None:
        best = None
        for jid, eta in cands:
            rec = mbt.ledger.records[jid]
            j_keep = expected_cost(mbt, src, gate_in_s=rec.a_gate_in, eta_s=eta,
                                   kf=self.kf)
            for dst in sorted(b for b in mbt.blocks if b != src):
                delta = self.layout.pre_gate_route_delta_s(src, dst)
                eta2 = eta + delta
                j_alt = (expected_cost(mbt, dst, gate_in_s=rec.a_gate_in,
                                       eta_s=eta2, kf=self.kf)
                         + delta / 3600.0)
                net = j_keep - j_alt - self.gain_margin
                if net > 0 and (best is None or net > best[0]):
                    best = (net, jid, dst)
        if best is None:
            return None
        self.ledger.append({"t": t, "axis": "SPACE", "src": src, "job_id": best[1],
                            "net_gain": round(best[0], 6), "suggest_dst": best[2]})
        return best[1]

    # ---- 반출(시간): J(지금 도착) − [J(Δ 뒤 도착) + Δ/3600] − 여유
    def decide_time(self, mbt, src: str, cands: list, t: float) -> str | None:
        best = None
        d = self.defer_delta_s
        for jid, eta in cands:
            rec = mbt.ledger.records[jid]
            j_now = expected_cost(mbt, src, gate_in_s=rec.a_gate_in, eta_s=eta,
                                  kf=self.kf)
            j_late = (expected_cost(mbt, src, gate_in_s=rec.a_gate_in + d,
                                    eta_s=eta + d, kf=self.kf)
                      + d / 3600.0)                 # 기사 외부 대기 — 동일 단가(은닉 금지)
            net = j_now - j_late - self.gain_margin
            if net > 0 and (best is None or net > best[0]):
                best = (net, jid)
        if best is None:
            return None
        self.ledger.append({"t": t, "axis": "TIME", "src": src, "job_id": best[1],
                            "net_gain": round(best[0], 6),
                            "defer_delta_s": d})
        return best[1]
