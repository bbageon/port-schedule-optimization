"""아키텍처 문서 ↔ 코드 교차검증.

문서(`.claude/docs/architecture/*.md`)에 적힌 수치가 **지금 코드와 맞는지** 본다.
불일치가 나오면 **둘 중 하나가 틀린 것**이고, 어느 쪽인지 판정해야 한다.

    PYTHONPATH=src python .claude/docs/architecture/verify_architecture.py

■ 무엇을 검사하나
  ① 무대 상수 — 블록·크레인·슬롯·관측창·물량·SLA·ρ
  ② 판매 계약 — 리드·창·격자·이연량·상한
  ③ 신경망 차원 — Q 행 21 = 7+6+8
  ④ 정보 경계 — 정책 경로가 실현 미래값을 읽지 않는가
  ⑤ 문서가 지적한 **알려진 결함**이 아직 그대로인가 (고쳐졌으면 문서를 고쳐야 함)
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC = ROOT / "src" / "yard_rl"
ok, bad, note = [], [], []


def chk(label, got, want):
    (ok if got == want else bad).append(f"{label}: 문서 {want!r} · 코드 {got!r}")


def main() -> int:
    # ── ① 무대
    from yard_rl.integrated.cost_curve_v2 import RHO_VESSEL_V2
    from yard_rl.integrated.profiles import build_h21_profile
    from yard_rl.integrated.terminal_stream import DIURNAL_DAY_TOTAL, OBS_24H
    from yard_rl.integrated.yard_layout import terminal_layout
    p, l = build_h21_profile(), terminal_layout()
    g = p.block
    chk("블록 수", len(l.ids), 21)
    chk("블록당 크레인", len(p.cranes), 2)
    chk("블록 슬롯", g.bay_count * g.row_count * g.tier_max, 1440)
    chk("관측창(초)", OBS_24H.observe_s, 86400.0)
    chk("하루 물량", DIURNAL_DAY_TOTAL, 3600)
    chk("트럭 SLA(초)", p.long_wait_sla_s, 1800.0)
    chk("본선 가중 ρ", RHO_VESSEL_V2, 10.0)
    chk("게이트→최근접(초)", round(l.gate_to_block_s(l.ids[0])), 190)
    chk("게이트→최원거리(초)", round(l.gate_to_block_s(l.ids[-1])), 410)

    # ── ② 판매 계약
    from yard_rl.integrated import sell_review as R, time_sell as T
    chk("통지 리드(초)", R.ANNOUNCE_LEAD_S, 1800.0)
    chk("검토 창(초)", R.WINDOW_S, 1800.0)
    chk("이연량(초)", T.DEFER_DELTA_S, 900.0)
    chk("이연 상한", T.MAX_ENTRY_DEFERRALS, 1)
    chk("이송 상한", R.MAX_TRANSFERS, 1)

    # ── ③ 신경망
    from yard_rl.integrated import sell_q as Q
    chk("Q 행 차원", Q.Q_ROW_DIM, 21)
    chk("블록 특징", Q.BLOCK_DIM, 7)
    chk("좌표 특징", Q.COORD_DIM, 8)
    chk("KEEP 기준점", Q.KEEP_Q, 0.0)
    from yard_rl.integrated.sell_q import SellQNet
    n = sum(x.numel() for x in SellQNet().parameters())
    chk("판매 Q 파라미터", n, 5633)

    # ── ④ 정보 경계 — 정책 경로에 실현 미래값이 없어야 한다
    LEAK = ("actual_gate_in", "actual_block_arrival", "actual_completion_s")
    for f in ("integrated/transfer_head.py", "integrated/sell_q.py",
              "integrated/sell_gain.py"):
        txt = (SRC / f).read_text(encoding="utf-8")
        body = "\n".join(ln for ln in txt.splitlines()
                         if not ln.lstrip().startswith("#"))
        hit = [k for k in LEAK if k in body]
        (bad if hit else ok).append(
            f"정보경계 {f}: {'★누출 ' + str(hit) if hit else '깨끗'}")

    # ── ⑤ 문서가 지적한 알려진 결함 — 아직 그대로인가
    th = (SRC / "integrated/transfer_head.py").read_text(encoding="utf-8")
    m = re.search(r"return \[inside / 10\.0.*?\]", th, re.S)
    feat = m.group(0) if m else ""
    has_time = ("t /" in feat) or ("t/" in feat)
    note.append(f"[결함] 정책 특징에 시각 없음: {'고쳐짐 → 문서 갱신 필요' if has_time else '그대로(문서 일치)'}")

    cc = (SRC / "integrated/cost_curve_v2.py").read_text(encoding="utf-8")
    note.append(f"[결함] 재조작 비용항 없음: "
                f"{'고쳐짐 → 문서 갱신 필요' if 'rehandle' in cc else '그대로(문서 일치)'}")

    old_keys = 0
    for f in SRC.rglob("*.py"):
        t = f.read_text(encoding="utf-8", errors="ignore")
        old_keys += len(re.findall(r'\["(job_id|block|arrival_s)"\]', t))
    note.append(f"[진행] 구 키 소비처 {old_keys}곳 (스키마 확정 후 0 이 목표)")

    # ── 보고
    print("═" * 62)
    print(f"  일치 {len(ok)}  ·  불일치 {len(bad)}")
    print("═" * 62)
    for s in ok:
        print("  ✅", s)
    if bad:
        print()
        for s in bad:
            print("  ❌", s)
    print()
    for s in note:
        print("  •", s)
    print()
    if bad:
        print("★불일치가 있다. 문서와 코드 중 어느 쪽이 옳은지 판정하고 둘을 맞춘다.")
        return 1
    print("문서와 코드가 일치한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
