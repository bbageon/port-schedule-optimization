"""YR-146 계약 고정 — 대기 허가증 v2 (25차 필수 테스트 7종)."""
from __future__ import annotations

from yard_rl.experiments.yr090_dense_vessel import BASE
from yard_rl.experiments.yr146_deploy_guard import IV_KEYS, _episode_guard, classify_wait


def _c(**kw):
    base = dict(has_progress=True, interference=False, has_escape=False,
                deadline_pressure=False, triggered=False,
                prev_untriggered_same_snap=False)
    base.update(kw)
    return classify_wait(**base)


def test_future_eta_recheck_no_intervention():
    """① 600초 점검 재개방인데 관측 trigger 가 미래 → 불개입."""
    assert _c(triggered=True, prev_untriggered_same_snap=True) == "ALLOW"


def test_eta_update_no_intervention():
    """② ETA 갱신(새 trigger 미래 이동) → 불개입 (trigger 존재 경로)."""
    assert _c(triggered=True) == "ALLOW"


def test_no_target_repeat_intervenes():
    """③⑤ 실제 사건 처리 뒤에도 무진전 + 기다릴 대상 없는 반복 대기 → 개입."""
    assert _c(triggered=False, prev_untriggered_same_snap=True) == "IV_NO_TARGET"
    assert _c(triggered=False, prev_untriggered_same_snap=False) == "ALLOW"  # 최초 1회 허용


def test_unrelated_event_does_not_reset():
    """④ 스냅샷 = (완료 수·실작업 후보 수) — 무관 이벤트는 두 값을 안 바꿔 반복 감지 유지.
    (스냅샷 불변 = prev_untriggered_same_snap True 경로 그대로 개입.)"""
    assert _c(triggered=False, prev_untriggered_same_snap=True) == "IV_NO_TARGET"


def test_deadline_pressure_intervenes():
    """⑥(f) 마감 압박 — 잔여 작업·진행 가능·종료 임박 → 개입."""
    assert _c(deadline_pressure=True, triggered=True) == "IV_DEADLINE"


def test_interference_escape():
    """⑥(g) 진행 전무 ∧ 간섭 → 최소 탈출 · 간섭 아니면 구조적 대기 불간섭."""
    assert _c(has_progress=False, interference=True, has_escape=True) == "IV_ESCAPE"
    assert _c(has_progress=False, interference=False, has_escape=True) == "ALLOW"


def test_guard_passthrough_when_no_intervention():
    """⑦ 개입 0 인 판에서 ON 은 OFF 와 완전 동일 (불간섭 계약·통합)."""
    cell, seed, arm, ts = "mid-tight", BASE["mid-tight"] + 16, "c1", 221_000
    off = _episode_guard(cell, seed, arm, ts, guard_on=False)
    on = _episode_guard(cell, seed, arm, ts, guard_on=True)
    g = on.pop("guard")
    on.pop("guard_permits", None)                 # 원장(기록 전용)은 비교 제외
    assert g["dec"] > 0
    if sum(g[k] for k in IV_KEYS) == 0:
        assert on == off
