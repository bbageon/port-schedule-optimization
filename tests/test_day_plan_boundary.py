"""YR-171-A 정보 경계 기계 검증 — 공개 예약 장부가 미래를 훔쳐보지 않는가.

정보 계약을 바꾸는 변경(30분 통지 → 하루 명단 공개)이므로, "정말 공개 정보만
쓰는가"를 **사람이 읽어서**가 아니라 기계로 확인한다. 통과하지 못하면 이 계약으로
낸 어떤 성능 수치도 읽을 가치가 없다.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from yard_rl.integrated.day_plan import DayPlan, DayPlanError, attach, get
from yard_rl.integrated.slot_plan import DAY_S, N_SLOTS, SLOT_S

SRC = Path(__file__).resolve().parents[1] / "src" / "yard_rl" / "integrated" / "day_plan.py"

# 실현값 — 장부가 한 번도 읽어서는 안 되는 이름
FORBIDDEN = (
    "actual_gate_in", "actual_block_arrival", "a_gate_in", "a_block_arrival",
    "service_start", "service_end", "job_done", "rehandles", "remaining_moves",
    "actual_completion_s", "truth",
)
# 엔진 접근 경로 — 장부는 자기 사본만 본다
FORBIDDEN_ENGINE = ("blocks", "jobs", "ledger", "vessels", "sim")


def _schedule(n=6, *, block="B01"):
    return [{"job_id": f"{block}:D-{i:05d}", "block": block,
             "arrival_s": 600.0 + i * 1800.0,
             "flow": "GATE_IN" if i % 2 == 0 else "GATE_OUT"}
            for i in range(n)]


# ---------------------------------------------------------------- 정적 검사
def test_source_never_names_realized_values():
    """소스에 실현값 이름이 **한 번도** 나오지 않는다 (주석 제외 — 식별자 기준)."""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
    hit = sorted(n for n in FORBIDDEN if n in names)
    assert not hit, f"장부가 실현값을 참조한다: {hit}"


def test_source_never_touches_engine():
    """엔진 객체 속성 접근이 없다 — `attach`/`get` 의 day_plan 부착만 예외."""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ENGINE:
            bad.append(node.attr)
    assert not bad, f"장부가 엔진을 읽는다: {sorted(set(bad))}"


# ---------------------------------------------------------------- 내용 검사
def test_plan_is_stable_without_reschedule():
    """재예약이 없으면 장부는 **언제 물어봐도 같다** — 실현 사건에 반응하지 않는다."""
    p = DayPlan.from_schedule(_schedule())
    a, v = p.slot_hist(), p.plan_version
    assert p.slot_hist() == a and p.plan_version == v


def test_reschedule_bumps_version_and_moves_one_truck():
    p = DayPlan.from_schedule(_schedule(n=2))
    jid = "B01:D-00000"
    before = p.slot_hist()["B01"]["in"][:]
    v0 = p.plan_version
    p.reschedule(jid, p.gate_in(jid) + 4 * SLOT_S)
    assert p.plan_version == v0 + 1
    after = p.slot_hist()["B01"]["in"]
    assert sum(after) == sum(before), "재예약은 대수를 바꾸지 않는다"
    assert after != before, "재예약했는데 칸이 그대로다"


def test_appointment_is_immutable_under_reschedule():
    """최초 예약은 안 바뀐다 — 기사 외부 대기의 원점이라 바뀌면 비용이 은닉된다."""
    p = DayPlan.from_schedule(_schedule(n=1))
    jid = "B01:D-00000"
    a0 = p.appointment(jid)
    p.reschedule(jid, a0 + SLOT_S)
    assert p.appointment(jid) == a0
    assert p.gate_in(jid) == a0 + SLOT_S


def test_cross_day_reschedule_refused():
    """날짜를 넘기는 재예약은 거부(명세: 날짜 변경 금지)."""
    p = DayPlan.from_schedule(_schedule(n=1))
    jid = "B01:D-00000"
    with pytest.raises(DayPlanError):
        p.reschedule(jid, DAY_S + 1.0)
    with pytest.raises(DayPlanError):
        p.reschedule(jid, -1.0)


def test_unknown_job_refused():
    p = DayPlan.from_schedule(_schedule(n=1))
    with pytest.raises(DayPlanError):
        p.reschedule("없는:작업", 100.0)


def test_duplicate_job_id_refused():
    s = _schedule(n=1) * 2
    with pytest.raises(DayPlanError):
        DayPlan.from_schedule(s)


# ---------------------------------------------------------------- 만료 검사
def test_stale_quote_is_hard_failure():
    """버전이 오른 뒤 옛 견적을 쓰면 조용히 넘어가지 않고 **터진다**."""
    p = DayPlan.from_schedule(_schedule(n=2))
    stamp = p.stamp()
    p.check_fresh(stamp)                       # 아직 유효
    p.reschedule("B01:D-00000", 7200.0)
    with pytest.raises(DayPlanError):
        p.check_fresh(stamp)


# ---------------------------------------------------------------- 부착 검사
class _FakeMbt:
    pass


def test_attach_once_only():
    m = _FakeMbt()
    assert get(m) is None, "안 붙인 상태는 None (구 계약)"
    attach(m, _schedule(n=1))
    assert get(m) is not None
    with pytest.raises(DayPlanError):
        attach(m, _schedule(n=1))


def test_hist_shape_and_grid():
    """48칸이고, 하루 격자 밖 예약은 어느 칸에도 얹지 않는다."""
    s = _schedule(n=1)
    s[0]["arrival_s"] = DAY_S + 5000.0          # 배수 구간
    p = DayPlan.from_schedule(s)
    assert p.slot_hist() == {}, "격자 밖 예약이 칸에 얹혔다"
    s[0]["arrival_s"] = 0.0
    p = DayPlan.from_schedule(s)
    h = p.slot_hist()["B01"]
    assert len(h["in"]) == N_SLOTS and len(h["out"]) == N_SLOTS
    assert sum(h["in"]) == 1.0
