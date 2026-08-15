"""YR-171-A 선행 실측 — **현 계약에서 48칸 계획표가 실제로 얼마나 비어 있는가**.

■ 왜 재는가
YR-174 판정 뒤 다음 축은 "시간 좌표를 +15분 1칸에서 48칸으로 연다" 이다. 그런데
`ScheduledAnnouncer` 는 트럭을 **도착 30분 전**에만 투입한다(`ANNOUNCE_LEAD_S=1800`
= 슬롯 1칸). 그러면 `slot_plan` 이 만드는 [48, F] 계획표에서 **미래 칸은 거의 전부 0**
일 것이다 — 통지된 트럭이 없으니까.

그 상태로 48칸을 열면 정책은 "먼 슬롯일수록 한가하다" 를 배운다. 실제로 한가한 게
아니라 **아직 안 보이는 것**인데도. 이건 최적화가 아니라 계측 결함이다.

이 하네스는 그 주장을 **숫자로 확정**한다(성능 판정 아님 — 관측 전용):
  · 각 시각에서 미래 48칸 중 통지 트럭이 있는 칸이 몇 칸인가
  · 그 칸들이 실제로는 몇 대를 받게 되는가(사후 정답과 대조)

■ 정보 경계
사후 정답(`schedule` 원본)은 **비교용으로만** 쓴다. 계획표 생성 경로에는 넣지 않는다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("outputs/reports/yr171_horizon")
SPEC = ".claude/docs/dashboard-task-specs/YR-171-time-sale-slot-contract.md"
PROBE_HOURS = (4.0, 8.0, 11.0, 15.0, 20.0)     # 야간·오전·피크·피크·저녁


class _Probe:
    """관측 전용 훅 — 지정 시각에 계획표 점유를 찍는다(결정에 개입 없음)."""

    def __init__(self, times: list[float]):
        self.want = sorted(times)
        self.rows: list[dict] = []
        self._i = 0

    def review(self, mbt, t: float) -> None:
        if self._i >= len(self.want) or t < self.want[self._i]:
            return
        self._i += 1
        from ..integrated.slot_plan import (N_SLOTS, SLOT_FEATURES, SLOT_S,
                                            terminal_slot_plan)
        i_in = SLOT_FEATURES.index("notified_in")
        i_out = SLOT_FEATURES.index("notified_out")
        plan = terminal_slot_plan(mbt, t, raw=True)
        cur = int(t // SLOT_S)
        fut_nonzero, fut_total, per_block_fut = 0, 0, []
        for _bid, tbl in plan.items():
            n = sum(1 for k in range(cur + 1, N_SLOTS)
                    if tbl[k][i_in] + tbl[k][i_out] > 0)
            per_block_fut.append(n)
            fut_nonzero += n
            fut_total += max(0, N_SLOTS - cur - 1)
        self.rows.append({
            "t": t, "hour": round(t / 3600.0, 2), "cur_slot": cur,
            "future_slots_per_block": max(0, N_SLOTS - cur - 1),
            "future_slots_with_trucks": fut_nonzero,
            "future_slots_total": fut_total,
            "share_nonempty": round(fut_nonzero / fut_total, 4) if fut_total else None,
            "per_block_min": min(per_block_fut), "per_block_max": max(per_block_fut)})


def run(seed: int = 9_000_000) -> Path:
    from ..integrated.policy_config import ADOPTED_C0_GUARD
    from ..integrated.repro import code_dirty, repro_stamp
    from ..integrated.slot_plan import N_SLOTS, SLOT_S
    from ..integrated.terminal_stream import OBS_24H, build_diurnal
    from .yr151_transfer_ppo import load_kf
    from .yr170_sell_ppo_diurnal import KeepAllTrail, run_episode_diurnal
    from ..integrated.sell_review import ANNOUNCE_LEAD_S

    probe = _Probe([h * 3600.0 for h in PROBE_HOURS])
    run_episode_diurnal(seed, KeepAllTrail(), load_kf(),
                        exec_config=ADOPTED_C0_GUARD, _extra_review=probe.review)

    # 사후 정답 — 그 날 실제로 각 슬롯에 몇 대가 오기로 되어 있었나(비교 전용).
    # run_episode_diurnal 과 **같은 인자**로 다시 만든다(명단은 결정론).
    from ..integrated.terminal_stream import (DIURNAL_DAY_TOTAL,
                                              TerminalStreamParams)
    from ..integrated.yard_layout import terminal_layout
    from ..integrated.profiles import build_h21_profile
    built = build_diurnal(build_h21_profile(), seed, obs=OBS_24H,
                          layout=terminal_layout(),
                          params=TerminalStreamParams(load_4h=DIURNAL_DAY_TOTAL),
                          background_seed=seed)
    truth = [0] * N_SLOTS
    for e in built["schedule"]:
        k = int(e["arrival_s"] // SLOT_S)
        if 0 <= k < N_SLOTS:
            truth[k] += 1
    n_slots_with_trucks = sum(1 for v in truth if v > 0)

    res = {"experiment": "YR-171-A 선행 실측 — 48칸 계획표의 미래 가시성",
           "kind": "observation_only", "seed": seed,
           "announce_lead_s": ANNOUNCE_LEAD_S, "slot_s": SLOT_S, "n_slots": N_SLOTS,
           "lead_in_slots": round(ANNOUNCE_LEAD_S / SLOT_S, 3),
           "probe_rows": probe.rows,
           "truth_slots_with_trucks": n_slots_with_trucks,
           "truth_per_slot": truth,
           "note": "share_nonempty = 미래 슬롯 중 통지된 트럭이 하나라도 있는 칸의 비율. "
                   "사후 정답은 비교 전용이며 계획표 생성 경로에 넣지 않는다.",
           "code_dirty": bool(code_dirty()),
           "stamp": repro_stamp(
               experiment="YR-171-A 선행 실측 — 48칸 계획표의 미래 가시성",
               seeds={"episode": [seed]},
               params={"probe_hours": list(PROBE_HOURS),
                       "ANNOUNCE_LEAD_S": ANNOUNCE_LEAD_S,
                       "observation": OBS_24H.as_dict()},
               prereg=SPEC)}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "visibility.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                 encoding="utf-8")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=9_000_000)
    p = run(ap.parse_args().seed)
    d = json.loads(p.read_text(encoding="utf-8"))
    print(f"통지 선행 = {d['lead_in_slots']} 슬롯 (슬롯 {d['slot_s']}초 × {d['n_slots']}칸)")
    for r in d["probe_rows"]:
        print(f"  {r['hour']:>5.1f}시  현재 {r['cur_slot']:>2}번 칸  "
              f"미래 {r['future_slots_per_block']:>2}칸/블록  "
              f"트럭 보이는 칸 {r['future_slots_with_trucks']:>3}/{r['future_slots_total']:>4} "
              f"({r['share_nonempty']:.2%})  블록당 {r['per_block_min']}~{r['per_block_max']}칸")
    print(f"사후 정답: 하루 48칸 중 실제로 트럭이 오는 칸 = {d['truth_slots_with_trucks']}칸")
    print("DONE", p)
