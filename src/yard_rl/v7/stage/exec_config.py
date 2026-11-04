"""v7 크레인 행동 범위 — 무엇을 고를 수 있나 ([[YR-332-a]] · YR-332 확정 결정 6).

    실작업(SERVE)        ✅  트럭·본선 일감을 한다
    파내기(PRE_REHANDLE) ✅  곧 올 트럭의 상자 위를 미리 치운다
    쉬기(WAIT)           ✅  할 일이 없을 때의 안전판 — 못 뺀다
    교착 탈출 빈 이동     ✅  크레인 둘이 서로 막혔을 때 비켜선다 (안전 기능)
    미리 자리잡는 빈 이동  ❌  ([[YR-148]] 채택 결정 — `safety_only`)

`safety_only=True` 는 `CandidateGenerator._reposition` 에서 **탈출 목표만** 후보로 낸다
(`world/integrated/candidates.py`). 물리 세계(`world/`)는 v6 사본 그대로이고, 여기서는
기존 스위치를 **켜기만** 한다.

`wait_mode="WAIT"` — 이연(DEFER) 대기 후보는 안 낸다. 쉬기는 그냥 쉬기다.
"""
from __future__ import annotations

from ..world.integrated.policy_config import LEGACY_DEFAULT, ExecPolicyConfig

#: v7 기본 — 크레인 결정의 행동 범위.
V7_CRANE_ONLY = ExecPolicyConfig(
    name="V7_CRANE_ONLY",
    wait_mode="WAIT",
    safety_only=True,
    bound_repo=False,
    prepo_one_shot=False,
    forbid_strategic_wait=False,
)

#: v6 까지의 행동 범위(미리 자리잡는 빈 이동 포함) — 비교·회귀용으로만 남긴다.
V6_LEGACY = LEGACY_DEFAULT

DEFAULT_EXEC_CONFIG = V7_CRANE_ONLY
