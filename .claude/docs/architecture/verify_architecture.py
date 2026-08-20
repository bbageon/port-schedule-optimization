"""아키텍처 문서 ↔ 코드 교차검증.

    PYTHONPATH=src python .claude/docs/architecture/verify_architecture.py

■ ★설계 원칙 — 값은 **문서에 한 벌만** 둔다
기대값을 스크립트에 박아두면 복사본이 셋(코드·문서·스크립트)이 되고, 문서만
낡아도 아무도 못 잡는다. 그래서 이 스크립트는 **문서의 ```contract 블록을 읽어**
코드와 대조한다.

    아키텍처가 바뀌면 → **문서의 contract 블록만** 고친다. 스크립트는 안 고친다.
    스크립트를 고치는 경우 → **새로운 종류의 검사**를 추가할 때뿐이다.

■ 검사 종류
  ① 계약값     문서 contract 블록 ↔ 코드 (아래 GETTERS 가 대응을 정의)
  ② 정보 경계  정책 경로가 실현 미래값을 읽지 않는가 (값이 아니라 규칙)
  ③ 알려진 결함 문서가 "결함"이라 적은 것이 아직 그대로인가
                — 고쳐졌으면 **문서를 갱신하라**고 알린다

■ 새 계약값을 추가하려면
  1) 해당 문서의 ```contract 블록에 `key = value` 한 줄
  2) 아래 GETTERS 에 `"key": lambda: <코드에서 읽는 식>` 한 줄
  키가 GETTERS 에 없으면 **미배선**으로 보고된다(조용히 넘어가지 않는다).
"""
from __future__ import annotations

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "src" / "yard_rl"


# ── 계약 키 → 코드에서 읽는 법 ------------------------------------------------
def _getters():
    from yard_rl.integrated import sell_q as Q, sell_review as R, time_sell as T
    from yard_rl.integrated.cost_curve_v2 import RHO_VESSEL_V2
    from yard_rl.integrated.profiles import build_h21_profile
    from yard_rl.integrated.sell_q import SellQNet
    from yard_rl.integrated.terminal_stream import DIURNAL_DAY_TOTAL, OBS_24H
    from yard_rl.integrated.yard_layout import terminal_layout
    p, l = build_h21_profile(), terminal_layout()
    g = p.block

    def crane_params():
        from yard_rl.experiments.yr151_transfer_ppo import load_adopted_execution_head
        net, _ = load_adopted_execution_head()
        return sum(x.numel() for x in net.parameters())

    return {
        # 02 무대
        "blocks": lambda: len(l.ids),
        "cranes_per_block": lambda: len(p.cranes),
        "block_slots": lambda: g.bay_count * g.row_count * g.tier_max,
        "observe_s": lambda: OBS_24H.observe_s,
        "day_total": lambda: DIURNAL_DAY_TOTAL,
        "truck_sla_s": lambda: p.long_wait_sla_s,
        "rho_vessel": lambda: RHO_VESSEL_V2,
        "gate_nearest_s": lambda: round(l.gate_to_block_s(l.ids[0])),
        "gate_farthest_s": lambda: round(l.gate_to_block_s(l.ids[-1])),
        # 03 결정층
        "announce_lead_s": lambda: R.ANNOUNCE_LEAD_S,
        "window_s": lambda: R.WINDOW_S,
        "defer_delta_s": lambda: T.DEFER_DELTA_S,
        "max_entry_deferrals": lambda: T.MAX_ENTRY_DEFERRALS,
        "max_transfers": lambda: R.MAX_TRANSFERS,
        "q_row_dim": lambda: Q.Q_ROW_DIM,
        "block_dim": lambda: Q.BLOCK_DIM,
        "coord_dim": lambda: Q.COORD_DIM,
        "keep_q": lambda: Q.KEEP_Q,
        "sell_q_params": lambda: sum(x.numel() for x in SellQNet().parameters()),
        "crane_net_params": crane_params,
        # 01 오더 스키마 — **목표값**. 확정 작업이 끝나면 일치해야 한다.
        "order_fields_target": _order_field_count,
        "record_fields_target": _record_field_count,
    }


def _order_field_count():
    """스케줄 항목의 키 수 — 확정 후 6(오더)이 목표."""
    from yard_rl.integrated.profiles import build_h21_profile
    from yard_rl.integrated.terminal_stream import (DIURNAL_DAY_TOTAL, OBS_24H,
                                                    TerminalStreamParams,
                                                    build_diurnal)
    from yard_rl.integrated.yard_layout import terminal_layout
    b = build_diurnal(build_h21_profile(), 1, obs=OBS_24H,
                      layout=terminal_layout(),
                      params=TerminalStreamParams(load_4h=DIURNAL_DAY_TOTAL),
                      background_seed=1)
    RECORD = {"blockInTime", "serviceStartTime", "jobDoneTime", "gateOutTime",
              "prevConLoc", "moveLoc", "conSwapReason",
              "block_previous", "block_worked", "swap_reason"}
    return len([k for k in b["schedule"][0] if k not in RECORD])


def _record_field_count():
    """기록 필드 수 — 확정 후 8(gateIn·blockIn·serviceStart·jobDone·gateOut
    + prevConLoc·moveLoc·conSwapReason)이 목표."""
    from yard_rl.integrated.time_contract import TruckTimes
    import dataclasses
    return len(dataclasses.fields(TruckTimes)) + 3


def _parse_contracts():
    """문서의 ```contract 블록 → {(문서, 키): 값}."""
    out = {}
    for md in sorted(HERE.glob("*.md")):
        for blk in re.findall(r"```contract\n(.*?)```", md.read_text(encoding="utf-8"), re.S):
            for line in blk.splitlines():
                line = line.split("#", 1)[0].strip()
                if not line or "=" not in line:
                    continue
                k, v = (x.strip() for x in line.split("=", 1))
                out[(md.name, k)] = float(v) if "." in v else int(v)
    return out


def main() -> int:
    ok, bad, miss, note = [], [], [], []
    contracts = _parse_contracts()
    getters = _getters()

    todo = []
    for (doc, key), want in sorted(contracts.items()):
        f = getters.get(key)
        if f is None:
            miss.append(f"{doc} · {key} — GETTERS 에 배선 없음")
            continue
        got = f()
        line = f"{key:<20} 문서 {want!r:>10} · 코드 {got!r:>10}   ({doc})"
        if key.endswith("_target"):
            # ★목표값 — 아직 안 한 작업이지 불일치가 아니다.
            (ok if got == want else todo).append(
                line + ("" if got == want else "   ← 미완"))
        else:
            (ok if got == want else bad).append(line)

    # ── ② 정보 경계 (값이 아니라 규칙)
    LEAK = ("actual_gate_in", "actual_block_arrival", "actual_completion_s")
    for f in ("integrated/transfer_head.py", "integrated/sell_q.py",
              "integrated/sell_gain.py"):
        body = "\n".join(ln for ln in (SRC / f).read_text(encoding="utf-8").splitlines()
                         if not ln.lstrip().startswith("#"))
        hit = [k for k in LEAK if k in body]
        (bad if hit else ok).append(
            f"{'정보경계':<20} {f} — {'★누출 ' + str(hit) if hit else '깨끗'}")

    # ── ③ 문서가 "결함"이라 적은 것이 아직 그대로인가
    th = (SRC / "integrated/transfer_head.py").read_text(encoding="utf-8")
    m = re.search(r"return \[inside / 10\.0.*?\]", th, re.S)
    feat = m.group(0) if m else ""
    note.append(f"[결함] 정책 특징에 시각 없음 — "
                f"{'★고쳐짐 → 문서 갱신 필요' if ('t /' in feat or 't/' in feat) else '그대로'}")
    cc = (SRC / "integrated/cost_curve_v2.py").read_text(encoding="utf-8")
    note.append(f"[결함] 재조작 비용항 없음 — "
                f"{'★고쳐짐 → 문서 갱신 필요' if 'rehandle' in cc else '그대로'}")
    n_old = sum(len(re.findall(r'\["(job_id|block|arrival_s)"\]',
                               f.read_text(encoding="utf-8", errors="ignore")))
                for f in SRC.rglob("*.py"))
    note.append(f"[진행] 구 키 소비처 {n_old}곳 (스키마 확정 후 0 이 목표)")

    print("═" * 68)
    print(f"  계약 {len(contracts)}개  ·  일치 {len(ok)}  불일치 {len(bad)}  "
          f"미배선 {len(miss)}  진행중 {len(todo)}")
    print("═" * 68)
    for s in ok:
        print("  ✅", s)
    for s in bad:
        print("  ❌", s)
    for s in miss:
        print("  ⚠️ ", s)
    if todo:
        print()
        print("  ── 목표값 (아직 안 한 작업 · 불일치 아님)")
        for s in todo:
            print("  ⏳", s)
    print()
    for s in note:
        print("  •", s)
    print()
    if bad or miss:
        print("★불일치/미배선이 있다. 문서와 코드 중 어느 쪽이 옳은지 판정하고 맞춘다.")
        return 1
    print("문서와 코드가 일치한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
