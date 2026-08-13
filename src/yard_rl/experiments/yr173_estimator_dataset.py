"""YR-173 — 견적 모델용 **데이터 적재** (사용자 지시 2026-08-13: "일단 쌓아둬").

목적: 나중에 "이 트럭을 이 블록·이 시간에 넣으면 부담이 얼마인가"를 **예측하는 모델**
(지도학습)이 필요해질 수 있다. 그때 학습 데이터를 새로 만들지 않도록, 실험을 돌리는
김에 **(결정 시점 상황 → 실현 결과)** 짝을 남긴다.

왜 지도학습인가: 지금 막힌 것은 강화학습(정답 없음)이지만, 견적은 **정답이 있다** —
시뮬레이션을 굴려보면 실제 값이 나온다. 예측 대 실제를 직접 대조할 수 있어 검증이
훨씬 쉽다. 960개 후보(수신 20 × 슬롯 48)를 한 번에 물어보는 것도 GPU 가 잘한다.

**★ 데이터의 지위 (사용자 명시 2026-08-13)**
이 자료는 **시뮬레이션 산출물이지 실제 터미널 운영 기록이 아니다.** 여기서 학습한
모델은 "이 시뮬레이터를 흉내내는 모델"이며, 실제 항만 성능을 예측한다고 주장할 수
없다. 산출물 헤더에 이 사실을 함께 적는다.

수집은 **읽기 전용**이다 — 결정에 개입하지 않고 관측만 한다. 기본 꺼짐.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..integrated.sell_review import block_inside, block_pipeline
from .yr139_blockq_v4_ppo import SLA_ANCHOR_S, phi_v2

OUT = Path("outputs/reports/yr173_estimator_dataset")
DATA_NOTE = ("시뮬레이션 산출 데이터 — 실제 터미널 운영 기록이 아니다. "
             "여기서 학습한 모델은 이 시뮬레이터의 근사이며 실제 항만 성능 주장 불가.")


class DecisionRecorder:
    """resolver 결정 시점의 **공개 상황**을 기록한다(읽기 전용).

    review 사슬에 얹어 매 60초 격자마다 블록 상태를 찍고, 에피소드 끝에서
    판매 원장·시간 장부와 조인해 **실현 결과**를 붙인다.
    """

    def __init__(self):
        self.snap: dict[float, dict] = {}      # t -> {block: 상태}

    def review(self, mbt, t: float) -> None:
        k = round(t, 6)
        if k in self.snap:
            return
        self.snap[k] = {
            b: {"inside": block_inside(sim, t),
                "pipeline": block_pipeline(mbt, b, t),
                "free": mbt.free_slots(b),
                "phi": round(phi_v2(sim, t), 4)}
            for b, sim in mbt.blocks.items()}

    # ------------------------------------------------------------------ 조인
    def rows(self, mbt, sell_ledger: list[dict], layout, end_s: float,
             sla_s: float) -> list[dict]:
        """결정 1건 = 표본 1개. 입력(결정 시점 공개상황) + 정답(실현 결과)."""
        l_t = SLA_ANCHOR_S + sla_s
        phi_end = {b: phi_v2(sim, end_s) for b, sim in mbt.blocks.items()}
        rec_by_job = {}
        for bid, sim in mbt.blocks.items():
            for jid, r in sim.time_ledger.records.items():
                rec_by_job[jid] = (bid, r)
        out = []
        for e in sell_ledger:
            if e.get("decision") not in ("SELL", "RESOLVER_KEEP", "DEFER"):
                continue
            t = round(e["t"], 6)
            st = self.snap.get(t)
            if st is None:
                continue
            src, dst = e["src"], e.get("dst")
            owner_rec = rec_by_job.get(e["job_id"])
            realized = None
            if owner_rec is not None:
                own_b, r = owner_rec
                if r.gate_in is not None and r.gate_out is not None:
                    stay = r.gate_out - r.gate_in
                    realized = {
                        "owner_block": own_b, "stay_s": round(stay, 1),
                        "penalty_s": round(max(0.0, stay - l_t), 1),
                        "cost_hours": round(stay / 3600.0
                                            + max(0.0, stay - l_t) / 3600.0, 5)}
            out.append({
                # --- 입력 (결정 시점 공개 정보만) ---
                "t": t, "hour": int(t // 3600), "src": src, "dst": dst,
                "axis": e.get("axis"), "flow": e.get("flow"),
                "decision": e["decision"],
                "src_state": st.get(src),
                "dst_state": st.get(dst) if dst else None,
                "travel_src_s": layout.gate_to_block_s(src),
                "travel_dst_s": layout.gate_to_block_s(dst) if dst else None,
                "terminal_inside": sum(v["inside"] for v in st.values()),
                # --- 견적(현행 계산식이 예측한 값) ---
                "delta_j": e.get("delta_j"),
                # --- 정답(실현) ---
                "realized": realized,
                "src_phi_delta": (round(phi_end[src] - st[src]["phi"], 4)
                                  if src in st else None),
                "dst_phi_delta": (round(phi_end[dst] - st[dst]["phi"], 4)
                                  if dst and dst in st else None),
            })
        return out


def dump(rows: list[dict], tag: str, *, seed: int, stamp: dict | None = None) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"decisions_{tag}_s{seed}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_header": True, "note": DATA_NOTE, "seed": seed,
                            "tag": tag, "n_rows": len(rows),
                            "stamp": stamp or {}}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def collect(seed: int, *, tag: str = "base") -> dict:
    """수집 전용 1 에피소드 — 판매는 현행 계산 기준선이 결정(정책 학습 없음)."""
    from ..integrated.policy_config import ADOPTED_C0_GUARD
    from .yr151_transfer_ppo import load_kf
    from .yr170_sell_ppo_diurnal import run_episode_diurnal
    from .yr170_axis_probe import AxisPolicy
    from ..integrated.terminal_stream import OBS_24H

    rec = DecisionRecorder()
    ep = run_episode_diurnal(seed, AxisPolicy("both"), load_kf(),
                             exec_head="sf", exec_config=None,
                             _return_mbt=True, _extra_review=rec.review)
    mbt = ep.pop("_mbt")
    from ..integrated.yard_layout import terminal_layout
    layout = terminal_layout()
    sla = next(iter(mbt.blocks.values())).profile.long_wait_sla_s
    rows = rec.rows(mbt, ep["sell_ledger"], layout, OBS_24H.observe_s, sla)
    p = dump(rows, tag, seed=seed)
    res = {"seed": seed, "tag": tag, "n_rows": len(rows),
           "n_sell": sum(1 for r in rows if r["decision"] == "SELL"),
           "n_keep": sum(1 for r in rows if r["decision"] == "RESOLVER_KEEP"),
           "n_defer": sum(1 for r in rows if r["decision"] == "DEFER"),
           "path": str(p), "note": DATA_NOTE}
    print(json.dumps(res, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=9_100_000)
    ap.add_argument("--tag", default="base")
    a = ap.parse_args()
    collect(a.seed, tag=a.tag)
    print("DONE")
