"""NEW 계층목적 (YR-071, 사용자 승인 조건 2) — 사전식(lexicographic) 4-tier 비교.

OLD 실패 기전 (term_diagnosis 실측): 총비용의 93% 가 간섭+차선혼잡 2항 공동지배 →
탐색기가 트럭 대기를 희생하며 페널티 항을 깎는 퇴화. 교정은 가중 재조정이 아니라
**항 간 비교 방식의 계층화**다: 하위 tier 가 아무리 좋아도 상위 tier 열세를 뒤집을
수 없다 ("하위 운영비용이 좋아져도 트럭 대기가 나빠지는 계획은 선택 불가").

tier-①(안전·미완료·backlog)은 비교 항이 아니라 구조 보장이다 — 안전·물리는
resolver mask, 완료는 mandatory+마감 강제 (YR-061 실측: 미완료 페널티 무발동).
에피소드 수준 guard(완료율·backlog)는 실험 판정이 담당한다.

항 내부의 정규화·가중 상수는 OLD(assumed_default)와 동일하게 유지한다 — 바꾸는
것은 항 "간" 우선순위뿐이므로, tier 안에서의 순서는 기존 상수의 몫이다.
동결 근거: strategy-history/2026-07-19-YR-071-목적재정렬-G0-prereg.md §2.
"""
from __future__ import annotations

TIER_A = ("truck_wait",)                       # 1차: 트럭 누적대기
TIER_B = ("long_wait",)                        # 2차: 긴 대기 (SLA 초과)
TIER_C = ("vessel_delay", "depart_delay")      # 3차: 본선 마감·지연
_CLASSIFIED = frozenset(TIER_A + TIER_B + TIER_C)


def hierarchy_key(terms: dict[str, float]) -> tuple[float, float, float, float]:
    """창 누적 항별 기여(dict) → 사전식 비교 key. 작을수록 좋다 (min 비교).

    미분류 항(신규 항 포함)은 전부 tier-D(하위 운영비용)로 합산 — 누락 방지 안전망.
    반올림 9자리는 기존 JointRolloutGreedy tie-break 관행과 동일 (부동소수 결정론).
    """
    a = sum(terms.get(t, 0.0) for t in TIER_A)
    b = sum(terms.get(t, 0.0) for t in TIER_B)
    c = sum(terms.get(t, 0.0) for t in TIER_C)
    d = sum(v for k, v in terms.items() if k not in _CLASSIFIED)
    return (round(a, 9), round(b, 9), round(c, 9), round(d, 9))
