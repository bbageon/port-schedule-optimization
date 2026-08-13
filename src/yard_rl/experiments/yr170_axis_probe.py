"""YR-170 축 분리 진단 — **공간 판매 vs 시간 판매, 어느 쪽이 손해를 내는가**.

42차 피드백이 지적한 지점: 시간 판매(재예약)는 지금 **고정 +15분**이고 기사 외부
대기가 **터미널 내 대기와 같은 단가**로 계상된다(phi_terminal:196). 그러면 이연은
"밖에서 15분 + 안에서 (15분 − 절약분)"이라, **절약분이 15분을 넘을 때만** 이득이다.
그건 볼록 구간(혼잡 첨두)을 정확히 피해갈 때만 생기는데 **고를 수가 없다**.

실측 배경: 학습 전 정책의 하루 판매가 공간 985 / **시간 2,635** 로 시간축이 2.7배다.
전체 손해 +21% 중 어느 축이 얼마인지 아직 안 갈랐다.

여기서 SF 집행(에피소드 87초 — 채택 헤드의 1/6)으로 **4조건**을 돌려 가른다.
성능 판정 아님 — 어느 축을 먼저 고칠지 정하는 진단이다.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .yr151_transfer_ppo import load_kf
from .yr170_sell_ppo_diurnal import run_episode_diurnal

OUT = Path("outputs/reports/yr170_axis_probe")
SEED = 8_400_000


class AxisPolicy:
    """지명 규칙은 결정론(공개 ETA 최소) — 다만 **축을 제한**한다.

    axis='both'  : 반입·반출 전부 지명 (현행)
    axis='space' : 반입(GATE_IN)만 → 좌표는 타 블록 이동
    axis='time'  : 반출(GATE_OUT)만 → 좌표는 시간 이연
    axis='none'  : 아무것도 지명 안 함 (전건 KEEP 기준선)
    """

    mode = "live"

    def __init__(self, axis: str):
        self.axis = axis
        self.trail: list[dict] = []

    def decide(self, mbt, src: str, cands: list, t: float) -> str | None:
        if self.axis == "none" or not cands:
            return None
        if self.axis == "space":
            c = [x for x in cands if x[2] == "GATE_IN"]
        elif self.axis == "time":
            c = [x for x in cands if x[2] == "GATE_OUT"]
        else:
            c = cands
        return min(c, key=lambda x: (x[1], x[0]))[0] if c else None


def run(exec_head: str = "sf") -> dict:
    rows = []
    for axis in ("none", "space", "time", "both"):
        t0 = time.perf_counter()
        ep = run_episode_diurnal(SEED, AxisPolicy(axis), load_kf(),
                                 exec_head=exec_head, exec_config=None)
        rows.append({"axis": axis, "phi_final": round(ep["phi_final"], 1),
                     "n_space": ep["n_space"], "n_time": ep["n_time"],
                     "elapsed_s": round(time.perf_counter() - t0, 1)})
    base = rows[0]["phi_final"]
    for r in rows:
        r["vs_keep"] = round(r["phi_final"] - base, 1)
        r["vs_keep_pct"] = round((r["phi_final"] / base - 1) * 100, 2)
        n = r["n_space"] + r["n_time"]
        r["per_sale_min"] = round(r["vs_keep"] * 60 / n, 2) if n else None
    res = {"probe": "axis_isolation", "exec_head": exec_head, "seed": SEED,
           "rows": rows,
           "note": "SF 집행 기준 진단(성능 판정 아님). vs_keep 양수 = 기준선보다 나쁨. "
                   "per_sale_min = 판매 1건당 평균 손익(분, 양수 = 손해)."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"axis_{exec_head}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec-head", default="sf", choices=["sf", "adopted"])
    a = ap.parse_args()
    run(a.exec_head)
    print("DONE")
