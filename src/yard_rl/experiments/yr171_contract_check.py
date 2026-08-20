"""YR-171-A 계약 전환 확인 — ①구 계약 불변 ②새 계약이 실제로 칸을 채우는가.

■ ① 회귀 (가장 중요)
`day_plan_public=False` 일 때 결과가 **한 자리도** 달라지면 안 된다. YR-174 판정
(전건 KEEP 대비 −84.49)이 조용히 바뀌면 그 판정을 다시 못 읽는다. 이미 박제된
`eval_vs_keep.json` 의 같은 (팔, 날) Φ 와 대조한다.

■ ② 가시성 (전환의 목적)
`day_plan_public=True` 일 때 48칸 계획표의 미래 칸이 실제로 채워지는지 잰다.
선행 실측(`yr171_horizon_probe`)에서 0.4~5.6% 였다. 성능 주장은 하지 않는다
(171-A 사다리 규약) — **정보가 열렸는지**만 본다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("outputs/reports/yr171_horizon")
EVAL = Path("outputs/reports/yr174_txn_reward/eval_vs_keep.json")
SPEC = ".claude/docs/dashboard-task-specs/YR-171-time-sale-slot-contract.md"
PROBE_HOURS = (4.0, 8.0, 11.0, 15.0, 20.0)


class _Vis:
    """관측 전용 — 지정 시각에 미래 칸 점유율을 찍는다."""

    def __init__(self, times):
        self.want = sorted(times)
        self.rows: list[dict] = []
        self._i = 0

    def review(self, mbt, t: float) -> None:
        if self._i >= len(self.want) or t < self.want[self._i]:
            return
        self._i += 1
        from ..integrated.day_plan import get as day_plan_get
        from ..integrated.slot_plan import (N_SLOTS, SLOT_FEATURES, SLOT_S,
                                            terminal_slot_plan)
        i_in = SLOT_FEATURES.index("notified_in")
        i_out = SLOT_FEATURES.index("notified_out")
        plan = terminal_slot_plan(mbt, t, raw=True)
        cur = int(t // SLOT_S)
        nz = tot = 0
        for tbl in plan.values():
            for k in range(cur + 1, N_SLOTS):
                tot += 1
                nz += 1 if tbl[k][i_in] + tbl[k][i_out] > 0 else 0
        dp = day_plan_get(mbt)
        self.rows.append({
            "hour": round(t / 3600.0, 2), "cur_slot": cur,
            "future_slots_with_trucks": nz, "future_slots_total": tot,
            "share_nonempty": round(nz / tot, 4) if tot else None,
            "plan_version": None if dp is None else dp.plan_version})


def _episode(day: int, *, public: bool, ckpt: str | None, probe=None) -> dict:
    import torch
    # ★박제값(eval_vs_keep.json)은 자식 프로세스에서 스레드 1개로 돌았다. 스레드 수가
    # 다르면 집행 헤드 신경망의 부동소수점 축약 순서가 달라져 집행 결정이 갈릴 수 있다.
    # 회귀 대조는 **같은 조건**에서 해야 의미가 있다.
    torch.set_num_threads(1)
    from ..integrated.policy_config import ADOPTED_C0_GUARD
    from ..v1.ppo_policy import PpoSellPolicy, TransferActor, TransferCritic
    from ..integrated.yard_layout import terminal_layout
    from .yr151_transfer_ppo import load_kf
    from .yr170_sell_ppo_diurnal import KeepAllTrail, run_episode_diurnal
    if ckpt is None:
        pol = KeepAllTrail()
    else:
        a, c = TransferActor(), TransferCritic()
        st = torch.load(ckpt, map_location="cpu", weights_only=True)
        a.load_state_dict(st["actor"])
        c.load_state_dict(st["critic"])
        pol = PpoSellPolicy(a, c, mode="live", sample=False,
                            layout=terminal_layout())
    ep = run_episode_diurnal(day, pol, load_kf(), exec_config=ADOPTED_C0_GUARD,
                             day_plan_public=public,
                             _extra_review=(probe.review if probe else None))
    return {"phi_final": round(ep["phi_final"], 4), "n_space": ep["n_space"],
            "n_time": ep["n_time"], "admitted": ep["admitted"]}


def run(day: int = 9_000_000, *, skip_visibility: bool = False) -> Path:
    from ..integrated.repro import code_dirty, repro_stamp
    from ..integrated.terminal_stream import OBS_24H

    prev = json.loads(EVAL.read_text(encoding="utf-8"))
    want = {(r["arm"], r["day"]): r for r in prev["rows"]}
    net = "outputs/reports/yr174_txn_reward/ppo_s8400000/net.pt"

    # ① 회귀 — 구 계약(public=False)이 박제값과 한 자리도 다르면 안 된다
    reg = []
    for arm, ck in (("K", None), ("P9_s8400000", net)):
        got = _episode(day, public=False, ckpt=ck)
        exp = want[(arm, day)]
        same = all(got[k] == exp[k]
                   for k in ("phi_final", "n_space", "n_time", "admitted"))
        reg.append({"arm": arm, "day": day, "expected": {
            k: exp[k] for k in ("phi_final", "n_space", "n_time", "admitted")},
            "got": got, "identical": same})

    # ② 가시성 — 새 계약(public=True)에서 미래 칸이 채워지는가
    if skip_visibility:
        old = json.loads((OUT / "contract_check.json").read_text(encoding="utf-8"))
        vis_rows, new_ep = old["visibility_new_contract"], old["new_contract_keep_episode"]
    else:
        vis = _Vis([h * 3600.0 for h in PROBE_HOURS])
        new_ep = _episode(day, public=True, ckpt=None, probe=vis)
        vis_rows = vis.rows

    res = {"experiment": "YR-171-A 계약 전환 확인", "kind": "regression+observation",
           "day": day,
           "regression_old_contract": reg,
           "all_identical": all(r["identical"] for r in reg),
           "visibility_new_contract": vis_rows,
           "new_contract_keep_episode": new_ep,
           "note": "성능 주장 없음(171-A 사다리 규약). 회귀는 구 계약 보존, "
                   "가시성은 정보가 열렸는지만 본다.",
           "code_dirty": bool(code_dirty()),
           "stamp": repro_stamp(
               experiment="YR-171-A 계약 전환 확인",
               seeds={"day": [day]},
               params={"probe_hours": list(PROBE_HOURS),
                       "observation": OBS_24H.as_dict()},
               prereg=SPEC)}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "contract_check.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                 encoding="utf-8")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, default=9_000_000)
    ap.add_argument("--skip-visibility", action="store_true")
    _a = ap.parse_args()
    p = run(_a.day, skip_visibility=_a.skip_visibility)
    d = json.loads(p.read_text(encoding="utf-8"))
    print("① 구 계약 회귀 (박제값과 대조)")
    for r in d["regression_old_contract"]:
        print(f"   {r['arm']:>12}  기대 Φ {r['expected']['phi_final']:>9.2f}  "
              f"실측 Φ {r['got']['phi_final']:>9.2f}  "
              f"{'동일' if r['identical'] else '★불일치★'}")
    print(f"   → 전부 동일: {d['all_identical']}")
    print("② 새 계약 가시성 (미래 칸 중 트럭이 보이는 비율)")
    for r in d["visibility_new_contract"]:
        print(f"   {r['hour']:>5.1f}시  {r['future_slots_with_trucks']:>5}/"
              f"{r['future_slots_total']:>5}  ({r['share_nonempty']:.2%})  "
              f"plan_version={r['plan_version']}")
    print("DONE", p)
