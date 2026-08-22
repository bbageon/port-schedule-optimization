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

■ `_target` 접미
  **v3 목표값**이다. 코드가 아직 안 따라온 것이므로 불일치(❌)가 아니라 진행중(⏳)
  으로 분류한다. 리팩토링이 끝나면 ✅ 로 바뀐다.

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
        "quay_axis_s": _quay_axis_s,
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
        # ── v3 목표값 (`_target`) — 아직 안 한 작업이지 불일치가 아니다.
        # ★2026-08-22: v3 패키지가 생겨 배선을 `yard_rl.v3` 로 옮겼다. v2(`integrated`)
        #   는 애초에 이 목표를 만족할 의무가 없다 — 재야 할 것은 **v3 가 문서를
        #   따라왔는가** 다. 다만 **무대 축**(리드 분포·부하 3수준·선급 물리)은 세대
        #   공용이라 `integrated` 를 계속 본다. v3 는 무대를 복제하지 않는다.
        # 00 라이프사이클 — 다섯 단계가 다 표현되는가
        "lifecycle_stages_target": _lifecycle_stage_count,
        # 01 오더 스키마 (축 ①)
        "order_fields_target": _order_field_count,
        "record_fields_target": _record_field_count,
        # 02 무대 (축 ①) — 무대는 공용이라 `integrated` 를 본다
        "lead_time_dist_target": _lead_time_is_dist,
        "load_levels_target": _load_level_count,
        # 03 결정층 (축 ③④)
        "seller_buyer_target": _seller_buyer_independent,
        "block_dim_target": _v3_block_dim,
        # 04 비용과 보상 (축 ②) — 원화 네 항
        "cost_terms_target": _cost_term_count,
        "krw_truck_hour_target": _krw_truck_hour,
        "vessel_classes_target": _vessel_class_count,
        # 04b 학습 잣대 — 반사실 지평
        "counterfactual_h_s_target": _counterfactual_h_s,
        # 05 정보 경계 (축 ④)
        "policy_has_clock_target": lambda: int(_has_clock()),
        "policy_waiting_def_target": _policy_waiting_def,
        # 06b 대조군 — 부분요인 셀 수
        "eval_cells_total": _eval_cells_total,
    }


V3 = SRC / "v3"


def _has_v3() -> bool:
    """v3 패키지가 생겼는가. 없으면 게터는 v2 를 본다(이행기 호환)."""
    return (V3 / "__init__.py").exists()


def _block_features_src() -> str:
    """정책이 블록을 어떻게 보는지 — **그 세대의** 특징 함수 본문.

    v3 가 있으면 v3 것을 읽는다. v2 의 `transfer_head` 는 v3 목표를 만족할 의무가
    없으므로 그걸로 v3 를 채점하면 영원히 ⏳ 로 남는다.
    """
    if _has_v3():
        return (V3 / "features/block.py").read_text(encoding="utf-8")
    th = (SRC / "integrated/transfer_head.py").read_text(encoding="utf-8")
    m = re.search(r"return \[inside / 10\.0.*?\]", th, re.S)
    return m.group(0) if m else ""


def _has_clock() -> bool:
    """정책 블록 특징에 시각이 들어갔는가."""
    f = _block_features_src()
    return "t /" in f or "t/" in f or "clock" in f


def _seller_buyer_independent() -> int:
    """Seller 와 Buyer 가 **각자 망을 갖고 독립 판단**하는가.

    망을 공유하면 "독립 행위자" 가 아니라 이름만 둘인 하나다 — 03 §1.
    """
    if not _has_v3():
        return 0
    try:
        from yard_rl.v3.actors import Buyer, BuyerNet, Seller, SellerNet
    except Exception:
        return 0
    return int(SellerNet is not BuyerNet and Seller is not Buyer)


def _v3_block_dim() -> int:
    """정책이 블록에서 읽는 특징 수 — 목표 9 (v2 의 7 + 시각 + 줄 선 대수)."""
    if not _has_v3():
        from yard_rl.integrated import sell_q as Q2
        return Q2.BLOCK_DIM
    from yard_rl.v3.features import BLOCK_DIM
    return int(BLOCK_DIM)


def _eval_cells_total() -> int:
    """부분요인 판정 셀 수 — 겹침을 뺀 **고유** 런이다."""
    if not _has_v3():
        return 0
    from yard_rl.v3.eval import plan_cells
    return len(plan_cells())


def _quay_axis_s() -> int:
    """게이트↔안벽 축 전체(초) — 안벽을 축 반대 끝에 대칭으로 둔 값.

    `190 + 410 = 600` 으로 **기존 레이아웃에서 유도**되므로 새 자유변수가 아니다.
    코드에 `quay_to_block_s` 가 생기면 그 값을 쓰고, 없으면 유도값을 돌려준다
    (지금은 안벽 개념이 없어 유도값 = 목표값이므로 ✅ 로 잡힌다).
    """
    # ★무대도 v3 사본을 본다 (사용자 지시 2026-08-22 — v3 는 밖을 임포트하지 않는다)
    if _has_v3():
        from yard_rl.v3.world.integrated.yard_layout import terminal_layout
    else:
        from yard_rl.integrated.yard_layout import terminal_layout
    lay = terminal_layout()
    q = getattr(lay, "quay_to_block_s", None)
    if q is not None:
        return round(q(lay.ids[0]) + lay.gate_to_block_s(lay.ids[0]))
    return round(lay.gate_to_block_s(lay.ids[0]) + lay.gate_to_block_s(lay.ids[-1]))


def _load_level_count() -> int:
    """실험이 도는 하루 물량 수준의 수 — 목표 3 (3,500·5,000·7,500).

    사용자 결정 2026-08-20: 셋 다 실제로 발생할 수 있는 오더 건수이므로 **셋 전부**
    에서 판정하고 전부 통과해야 한다. 지금은 단일 상수라 1 이다.
    """
    from yard_rl.v3.world.integrated import terminal_stream as TS
    levels = getattr(TS, "DIURNAL_LOAD_LEVELS", None)
    if not levels:
        return 1
    if _has_v3():
        # ★무대가 선언한 수준과 **판정이 실제로 도는 수준**이 같아야 한다.
        from yard_rl.v3.eval import LOAD_LEVELS
        if tuple(LOAD_LEVELS) != tuple(levels):
            return 0
    return len(levels)


def _lead_time_is_dist() -> int:
    """통지 리드타임이 트럭마다 다른가 (0=30분 고정)."""
    from yard_rl.v3.world.integrated import terminal_stream as TS
    if not (hasattr(TS, "sample_lead_s") and hasattr(TS, "LEAD_TIME_DIST")):
        return 0
    # ★상수 하나로는 분포가 아니다 — 분위가 실제로 다른 값을 내는지 본다.
    lo, mid, hi = (TS.sample_lead_s(q) for q in (0.25, 0.50, 0.75))
    return int(lo < mid < hi)


def _cost_term_count() -> int:
    """Φ 를 이루는 비용 항의 수 — 목표 4 (대기·YC 이동·재취급·본선).

    현행은 트럭·본선 둘뿐이다. YC 추가 이동과 재취급은 비용항이 없다
    (엔진에 동작은 있으나 Φ 에 안 들어간다).
    """
    if _has_v3():
        src = ((V3 / "reward/krw.py").read_text(encoding="utf-8")
               + (V3 / "reward/phi.py").read_text(encoding="utf-8"))
    else:
        src = (SRC / "integrated/cost_curve_v2.py").read_text(encoding="utf-8")
    have = [
        any(k in src for k in ("truck_wait_krw", "c_wait", "j_truck")),
        any(k in src for k in ("yc_move_krw", "c_move", "crane_move")),
        any(k in src for k in ("rehandle_krw", "c_rehandle", "rehandle")),
        any(k in src for k in ("vessel_idle_krw", "c_vessel", "j_vessel",
                               "RHO_VESSEL")),
    ]
    return sum(1 for h in have if h)


def _krw_truck_hour() -> int:
    """트럭 시간가치(원/트럭·시간) — 목표 40,000 (2026 안전운임 고시).

    현행은 비용시간 단위라 원화 상수가 없다 → 0.
    """
    if _has_v3():
        from yard_rl.v3.reward import KRW_TRUCK_HOUR
        return int(KRW_TRUCK_HOUR)
    from yard_rl.integrated import cost_curve_v2 as C
    for name in ("KRW_TRUCK_HOUR", "V_TRUCK_KRW_H", "TRUCK_KRW_PER_HOUR"):
        v = getattr(C, name, None)
        if v is not None:
            return int(v)
    return 0


def _vessel_class_count() -> int:
    """본선 선급 수 — 목표 3 (50k/100k/150k GT · 2.99원/GT·시간).

    현행은 전 본선이 동등하고 rho 가 10.0 단일값이라 1 이다.
    """
    from yard_rl.v3.world.integrated import vessel as V
    stage = getattr(V, "VESSEL_CLASSES", None) or getattr(V, "GT_CLASSES", None)
    if not stage:
        return 1
    if not _has_v3():
        return len(stage)
    # ★무대(물리)와 보상(원화)이 **같은 표**를 봐야 한다. 어긋나면 0 = 실패.
    from yard_rl.v3.reward import VESSEL_CLASSES as krw
    if tuple((c.name, c.gt, c.teu) for c in stage) != tuple(krw):
        return 0
    return len(stage)


def _counterfactual_h_s() -> float:
    """반사실 rollout 지평(초). 미구현이면 0."""
    if _has_v3():
        import yard_rl.v3 as V3P
        return float(getattr(V3P, "CF_HORIZON_S", 0.0))
    try:
        from yard_rl.experiments import yr204_counterfactual as CF  # type: ignore
    except Exception:
        return 0.0
    return float(getattr(CF, "CF_HORIZON_S", 0.0))


def _policy_waiting_def() -> int:
    """정책이 '줄 선 대수'를 보는가 (0=inside 총 대수로 대체)."""
    f = _block_features_src()
    return int("waiting" in f or "queued" in f)


def _lifecycle_stage_count() -> int:
    """라이프사이클 다섯 단계 중 **코드에 시각 필드가 있는** 것의 수.

    코피노 수신 → 게이트 진입 → 블록 진입 → 작업 완료 → 게이트 아웃 (사용자 확정
    2026-08-20). `service_start` 는 터미널이 전송하지 않으므로 단계가 아니다.
    현재 4 — `copinoNoticeTime` 이 없어 리드타임 축이 성립하지 않는다.
    """
    import dataclasses

    if _has_v3():
        from yard_rl.v3.schema import ExecutionRecord, Order, Stage
        from yard_rl.v3.schema.lifecycle import STAGE_FIELD
        owners = {f.name for f in dataclasses.fields(Order)}
        owners |= {f.name for f in dataclasses.fields(ExecutionRecord)}
        # 단계마다 **시각을 담을 필드가 실제로 있는가**. 이름만 세지 않는다.
        return sum(1 for st in Stage if STAGE_FIELD.get(st) in owners)

    from yard_rl.integrated.time_contract import TruckTimes
    tt = {f.name for f in dataclasses.fields(TruckTimes)}
    src = (SRC / "integrated/order_schema.py").read_text(encoding="utf-8")
    have = {
        "copinoNotice": ("copino" in src or "notice_s" in src),
        "gateIn": "gate_in" in tt,
        "blockIn": "block_arrival" in tt,
        "jobDone": "job_done" in tt,
        "gateOut": "gate_out" in tt,
    }
    return sum(1 for v in have.values() if v)


def _order_field_count():
    """스케줄 항목의 키 수 — 확정 후 6(오더)이 목표."""
    if _has_v3():
        import dataclasses

        from yard_rl.v3.schema import Order
        return len(dataclasses.fields(Order))

    from yard_rl.integrated.profiles import build_h21_profile
    from yard_rl.integrated.terminal_stream import (DIURNAL_DAY_TOTAL, OBS_24H,
                                                    TerminalStreamParams,
                                                    build_diurnal)
    from yard_rl.integrated.yard_layout import terminal_layout
    b = build_diurnal(build_h21_profile(), 1, obs=OBS_24H,
                      layout=terminal_layout(),
                      params=TerminalStreamParams(load_4h=DIURNAL_DAY_TOTAL),
                      background_seed=1)
    RECORD = {"gateInTime", "blockInTime", "serviceStartTime", "jobDoneTime",
              "gateOutTime", "prevConLoc", "conSwapReason",
              "block_previous", "block_worked", "swap_reason"}
    return len([k for k in b["schedule"][0] if k not in RECORD])


def _record_field_count():
    """기록 필드 수 — 목표 7 = 시각 5(gateIn·blockIn·serviceStart·jobDone·gateOut)
    + 교체 2(prevConLoc·conSwapReason). moveLoc 은 두지 않는다(2026-08-20).

    ★교체 2 는 **`record_swap()` 이 실제로 불릴 때만** 센다. 필드가 선언만 돼
    있고 호출부가 없으면 기록이 남지 않으므로 "있다"고 셀 수 없다.
    """
    import dataclasses

    if _has_v3():
        from yard_rl.v3.schema import ExecutionRecord, TRANSMITTED_FIELDS
        have = {f.name for f in dataclasses.fields(ExecutionRecord)}
        n = sum(1 for name in TRANSMITTED_FIELDS if name in have)
        callers = sum(f.read_text(encoding="utf-8", errors="ignore").count("record_swap(")
                      for f in V3.rglob("*.py") if f.name != "record.py")
        return n if callers else n - 2      # 호출부가 없으면 교체 2 는 못 센다

    from yard_rl.integrated.time_contract import TruckTimes
    callers = sum(f.read_text(encoding="utf-8", errors="ignore").count("record_swap(")
                  for f in SRC.rglob("*.py") if f.name != "order_schema.py")
    return len(dataclasses.fields(TruckTimes)) + (2 if callers else 0)


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
    files = ["integrated/transfer_head.py", "integrated/sell_q.py",
             "integrated/sell_gain.py"]
    if _has_v3():
        # v3 정책 경로 — 학생(Seller·Buyer)과 그 입력. 교사(counterfactual)는 뺀다:
        # 교사는 **일부러** 미래를 굴리는 쪽이라 여기서 잡으면 오탐이다.
        files += ["v3/actors/seller.py", "v3/actors/buyer.py",
                  "v3/actors/nets.py", "v3/features/block.py",
                  "v3/features/candidate.py",
                  # ★무대 결합부 — 엔진 장부에는 미래가 이미 적혀 있어 여기가
                  #   가장 새기 쉽다([[YR-215]]).
                  "v3/stage/bridge.py", "v3/stage/orders.py",
                  "v3/stage/episode.py"]
    for f in files:
        body = "\n".join(ln for ln in (SRC / f).read_text(encoding="utf-8").splitlines()
                         if not ln.lstrip().startswith("#"))
        hit = [k for k in LEAK if k in body]
        (bad if hit else ok).append(
            f"{'정보경계':<20} {f} — {'★누출 ' + str(hit) if hit else '깨끗'}")

    # ── ③ 문서가 "결함"이라 적은 것이 아직 그대로인가
    # ★결함은 **세대별로** 읽는다. v2 에 남아 있는 것이 정상이고(그 세대는 그렇게
    #   판정됐다), v3 에서 고쳐졌는지가 관심사다. 둘 다 그대로면 아직 안 한 일이다.
    v2_head = (SRC / "integrated/transfer_head.py").read_text(encoding="utf-8")
    m = re.search(r"return \[inside / 10\.0.*?\]", v2_head, re.S)
    v2_body = m.group(0) if m else ""          # ★특징 함수 본문만 — 파일 전체는 오탐
    v2_clock = "t /" in v2_body or "clock" in v2_body
    note.append(f"[결함] 정책 특징에 시각 없음 — "
                f"v2 {'고침' if v2_clock else '그대로(정상)'} · "
                f"v3 {'✅ 고침' if _has_clock() else '★아직'}")
    v2_cc = (SRC / "integrated/cost_curve_v2.py").read_text(encoding="utf-8")
    v3_cc = (V3 / "reward/krw.py").read_text(encoding="utf-8") if _has_v3() else ""
    note.append(f"[결함] 재조작 비용항 없음 — "
                f"v2 {'고침' if 'rehandle' in v2_cc else '그대로(정상)'} · "
                f"v3 {'✅ 고침' if 'rehandle' in v3_cc else '★아직'}")
    note.append(f"[세대] 목표값 배선 대상 — {'v3' if _has_v3() else 'v2(integrated)'}")
    # ★v3 세계 사본은 세지 않는다 — 원본과 같은 파일이라 두 번 세면 수치가 부푼다.
    n_old = sum(len(re.findall(r'\["(job_id|block|arrival_s)"\]',
                               f.read_text(encoding="utf-8", errors="ignore")))
                for f in SRC.rglob("*.py") if "v3/world/" not in f.as_posix())
    note.append(f"[진행] 구 키 소비처 {n_old}곳 (스키마 확정 후 0 이 목표 · 사본 제외)")

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
