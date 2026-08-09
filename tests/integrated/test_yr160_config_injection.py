"""YR-160 본체 1단계 — 행동 정의 플래그의 **주입** 배선 (2026-08-09).

후보 생성기가 전역 대신 주입받은 `ExecPolicyConfig` 를 읽는다. 미주입(None)은
과도기 전역 경로(기존 실험 골든 불변). 주입이 전역을 이겨야 "전역 깜빡함 → 몰래
기각 정책" 사고가 구조적으로 차단된다.
"""
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.integrated.policy_config import (ADOPTED_C0_GUARD, LEGACY_DEFAULT,
                                              applied)


def test_injected_config_beats_globals():
    g_inj = CandidateGenerator(config=LEGACY_DEFAULT)
    g_glob = CandidateGenerator()
    with applied(ADOPTED_C0_GUARD):
        assert g_inj._flags() == ("WAIT", False, False, False)       # 주입 우선
        assert g_glob._flags() == ("DEFER_ALL", True, False, True)   # 미주입 = 전역
    assert g_glob._flags() == ("WAIT", False, False, False)          # 원상복구 확인


def test_adopted_injection_matches_guard_values():
    g = CandidateGenerator(config=ADOPTED_C0_GUARD)
    assert g._flags() == ("DEFER_ALL", True, False, True)
