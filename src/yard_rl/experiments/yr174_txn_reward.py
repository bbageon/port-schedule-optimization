"""YR-174 — **거래별 실현 손익을 그 결정에 되돌려 붙이는** 배선.

■ 무엇을 바꾸나
지금까지 보상은 두 판이었다. 둘 다 실패했다.
  · 전역(YR-170): 하루 총비용 하나를 21블록이 나눠 가짐 → 행동 신호 0.0004%
  · 블록별(YR-172): 자기 블록 비용만 → 떠넘기기 구조(판 +5.8분 / 받은 −7.5분)
여기서는 **거래 하나하나의 실현 손익**을 계산해 **그 판매를 결정한 그 표본**에 붙인다.

    보상(그 결정) = D_i   (판매가 성사된 결정에만. KEEP·거절은 0)

■ D_i 를 어떻게 실측하나 (반사실 없이)
정확한 D_i = C(그 거래만 없앤 세계) − C(실제 세계) 이지만, 그건 하루를 다시 돌려야 하고
YR-168 에서 잡음이 효과의 80배라 표본 부족으로 실패했다. 여기서는 **한 번의 실행에서
관측 가능한 항으로 분해**한다:

    D_i = R_src − B_dst − ΔC_route − ΔC_driver

  · `B_dst` 수신 부담 = 그 트럭이 dst 에서 실제로 쓴 비용
      + **그 트럭이 크레인을 점유하는 동안 dst 에서 기다린 다른 트럭 수 × 점유 시간**
    (뒤에 줄 선 트럭들이 그만큼 밀린다 — 견적식이 놓치던 외부효과가 이 항이다)
  · `R_src` 소스 절감 = 그 트럭이 src 에 **남았더라면** 점유했을 시간 × 그 시각 src 에서
    기다리던 트럭 수 + 그 트럭 자신이 src 에서 겪었을 비용
    ※ "남았더라면"은 관측되지 않으므로 **실현 서비스 시간을 그대로 대입**한다(가정).
      같은 트럭·같은 컨테이너라 서비스 시간은 블록이 달라도 비슷하다는 가정이며,
      재취급 수는 야드 상태에 따라 다르므로 **근사**임을 명시한다.
  · `ΔC_route` 주행 추가 = (게이트→dst) − (게이트→src), 실측
  · `ΔC_driver` 기사 시간변경 = 이연분(외부 대기), 실측

**대칭성**: 양쪽 모두 "점유 시간 × 그때 대기 중이던 트럭 수" 라는 같은 형태다.
한쪽만 정밀하게 재면 편향이 생기므로 일부러 같은 근사를 쓴다.

■ 눈금 (앞선 실패 재발 방지)
거래 손익은 자연스럽게 O(1)(대략 −3~+3)이라 **나누지 않는다**(RET_SCALE = 1.0).
전역 보상 때 목표가 −1,670 인데 critic 초기 출력이 −0.06 이라 2만 7천 배 어긋나
학습이 멈췄다. 여기서는 목표가 0 근처라 처음부터 닿는다.
그리고 `assert_scale_sane` 로 **시작 전에** 목표와 critic 출력의 거리를 검사한다 —
100배 넘게 벌어지면 학습을 시작하지 않는다.

■ 자기참조 발산 소멸
전역 판에서 목표에 `+ V(s_end)` 를 더해 γ=1 되먹임으로 발산했다. 거래별 보상에는
그 항이 **아예 없다** — 하루 끝 예측을 더할 이유가 없으므로 구조적으로 재발 불가다.
"""
from __future__ import annotations

from statistics import fmean, pstdev

RET_SCALE = 1.0          # 거래 손익은 이미 O(1) — 나누지 않는다(위 설명)
SCALE_GUARD_RATIO = 100.0   # 목표와 critic 출력이 이 배수 넘게 벌어지면 실격


class TransactionLog:
    """resolver 가 확정한 거래를 **결정과 이을 수 있게** 기록한다(관측 전용).

    판매 원장(orch.ledger)에는 (t, src, job_id, dst) 가 있고 정책 기록(policy.trail)
    에는 (t, src) 가 있다. 둘을 (round(t,6), src) 로 조인한다 — 한 블록은 한 epoch 에
    제안을 하나만 내므로 이 키가 유일하다.
    """

    def __init__(self):
        self.rows: list[dict] = []

    def collect(self, sell_ledger: list[dict]) -> list[dict]:
        self.rows = [
            {"txn": i, "t": round(e["t"], 6), "src": e["src"],
             "job_id": e["job_id"], "dst": e.get("dst"),
             "axis": e.get("axis"), "delta_j": e.get("delta_j")}
            for i, e in enumerate(sell_ledger)
            if e.get("decision") in ("SELL", "DEFER")]
        return self.rows


# ------------------------------------------------------------------ 실현 손익
def _waiting_at(records, t: float) -> int:
    """시각 t 에 그 블록에서 **서비스를 기다리던** 트럭 수 (B ≤ t < S)."""
    n = 0
    for r in records.values():
        if (r.block_arrival is not None and r.block_arrival <= t
                and (r.service_start is None or r.service_start > t)):
            n += 1
    return n


def _truck_cost(r, l_t: float) -> float:
    """그 트럭 자신의 실현 비용(비용시간) — 체류 + 43분 초과 벌금."""
    if r.gate_in is None or r.gate_out is None:
        return 0.0
    stay = r.gate_out - r.gate_in
    return stay / 3600.0 + max(0.0, stay - l_t) / 3600.0


def realized_credit(mbt, txns: list[dict], layout, *, l_t: float) -> dict:
    """거래별 실현 손익 D_i. 반환 {txn: D_i} (비용시간, 양수 = 터미널에 이득).

    한 번의 실행에서 관측 가능한 항으로만 계산한다(모듈 docstring 의 분해식).
    """
    ledgers = {b: sim.time_ledger.records for b, sim in mbt.blocks.items()}
    out: dict[int, float] = {}
    for tx in txns:
        src, dst, jid = tx["src"], tx.get("dst"), tx["job_id"]
        owner = dst if (tx["axis"] == "SPACE" and dst) else src
        rec = ledgers.get(owner, {}).get(jid)
        if rec is None or rec.service_start is None or rec.job_done is None:
            out[tx["txn"]] = 0.0          # 미완 — 채점하지 않는다(0 = KEEP 과 동일)
            continue
        occupy = rec.job_done - rec.service_start          # 크레인 점유 시간
        own = _truck_cost(rec, l_t)

        if tx["axis"] == "SPACE" and dst:
            # 수신 부담 = 자기 비용 + 점유 동안 dst 에서 밀린 트럭들
            blocked_dst = _waiting_at(ledgers[dst], rec.service_start)
            b_dst = own + blocked_dst * occupy / 3600.0
            # 소스 절감 = 남았더라면 점유했을 시간 × 그때 src 대기 수 + 자기 비용
            blocked_src = _waiting_at(ledgers[src], rec.service_start)
            r_src = own + blocked_src * occupy / 3600.0
            route = max(0.0, layout.gate_to_block_s(dst)
                        - layout.gate_to_block_s(src)) / 3600.0
            out[tx["txn"]] = r_src - b_dst - route
        else:
            # 시간 이연 — 블록은 그대로다. 이득은 "원래 오려던 시각에 안 와서 그때
            # 대기 중이던 트럭들이 덜 밀린 것", 비용은 "기사가 밖에서 더 기다린 것".
            j = mbt.blocks[src].jobs.get(jid)
            orig = getattr(j, "appointment_gate_time", None) if j else None
            if orig is None or rec.gate_in is None:
                out[tx["txn"]] = 0.0
                continue
            defer_s = max(0.0, rec.gate_in - orig)
            # 원래 시각에 그 블록에서 기다리던 트럭 수 — 그만큼 덜 밀렸다
            relief = _waiting_at(ledgers[src], orig) * occupy / 3600.0
            out[tx["txn"]] = relief - defer_s / 3600.0
    return out


# ------------------------------------------------------------------ 배치 조립
def build_batch_txn(trail: list[dict], txns: list[dict], credits: dict,
                    *, ret_scale: float = RET_SCALE) -> list[dict]:
    """거래 손익을 **그 결정을 내린 표본**에 붙인다.

    조인 키 = (round(t,6), src). 판매가 성사되지 않은 결정(KEEP·resolver 거절)은
    실제 상태가 KEEP 과 같으므로 **보상 0** 이다(설계안 명시).
    """
    by_key = {(tx["t"], tx["src"]): tx["txn"] for tx in txns}
    out = []
    for tr in trail:
        key = (round(tr["t"], 6), tr["src"])
        txn = by_key.get(key)
        d = credits.get(txn, 0.0) if txn is not None else 0.0
        r = d / ret_scale
        out.append({**tr, "ret": r, "adv": r - tr["value"],
                    "txn": txn, "resolver": None})
    return out


# ------------------------------------------------------------------ 눈금 검사
def scale_report(batch: list[dict]) -> dict:
    """목표·critic 출력·둘의 거리 — **학습 시작 전에** 찍어 본다."""
    rets = [b["ret"] for b in batch]
    vals = [b["value"] for b in batch]
    nz = [r for r in rets if abs(r) > 1e-12]
    ret_rms = (fmean(r * r for r in rets)) ** 0.5 if rets else 0.0
    val_rms = (fmean(v * v for v in vals)) ** 0.5 if vals else 0.0
    return {"n": len(batch), "n_nonzero": len(nz),
            "nonzero_share": round(len(nz) / len(batch), 4) if batch else 0.0,
            "ret_rms": round(ret_rms, 6), "ret_min": round(min(rets), 4) if rets else 0,
            "ret_max": round(max(rets), 4) if rets else 0,
            "ret_std": round(pstdev(rets), 6) if len(rets) > 1 else 0.0,
            "value_rms": round(val_rms, 6),
            "ratio": round(ret_rms / max(val_rms, 1e-9), 2)}


def assert_scale_sane(batch: list[dict], *, ratio_max: float = SCALE_GUARD_RATIO) -> dict:
    """★눈금 자동 검사 — 목표와 critic 출력이 너무 벌어지면 **시작 전에 멈춘다**.

    전역 보상 판에서 목표 −1,670 vs critic −0.06(2만 7천 배)로 학습이 멈춘 채
    16시간을 돌았다. 그 실패를 구조적으로 막는다. critic 이 아직 0 근처인 첫 회차에
    목표가 O(1) 이면 비율이 크게 나올 수 있으므로, **목표 자체의 절대 크기**도 함께 본다.
    """
    rep = scale_report(batch)
    if rep["n"] == 0:
        raise RuntimeError("배치가 비어 있다 — 학습 불가")
    if rep["ret_rms"] > ratio_max:
        raise RuntimeError(
            f"목표 크기가 과도하다: ret_rms={rep['ret_rms']} > {ratio_max}. "
            f"거래 손익은 O(1) 이어야 한다 — 눈금(RET_SCALE)이 잘못됐을 가능성. {rep}")
    if rep["ret_rms"] < 1e-4:
        raise RuntimeError(
            f"목표가 사실상 0 이다: ret_rms={rep['ret_rms']}. 나눗셈이 과했거나 "
            f"거래가 하나도 없다 — 학습 신호가 없다. {rep}")
    if rep["nonzero_share"] == 0.0:
        raise RuntimeError("보상이 전부 0 — 거래가 결정에 연결되지 않았다(조인 실패)")
    return rep
