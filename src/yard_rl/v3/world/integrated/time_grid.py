"""공용 60초 격자 판정 — 판매 검토와 WIP 투입이 공유하는 시간 유틸.

엔진의 review epoch 은 "60초 격자 + 모든 gate-in 시각"의 합집합이다. 격자 가드 없이
review 콜백을 쓰면 검토·투입 주기가 60초 계약을 벗어난다(실측: 300초 창에서 6회가
아니라 13회 — 감사 치명 5). 판매 쪽은 sell_review 가 이 가드를 먼저 도입했고,
투입 쪽(WipAdmissionController)이 빠져 있던 것을 2026-08-09 감사가 지적했다 —
생성기 계층(terminal_stream)이 판매 계층(sell_review)에 의존하지 않도록 여기로 분리.
"""
from __future__ import annotations


def on_grid(t: float, grid_s: float = 60.0) -> bool:
    """t 가 grid_s 격자 시각인가 — 부동소수점 오차 1e-6 허용."""
    r = t % grid_s
    return r < 1e-6 or grid_s - r < 1e-6
