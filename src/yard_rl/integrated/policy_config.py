"""실행 정책 구성의 **불변 정의 + 원자적 적용/복구** (YR-160 부분 이행, 2026-08-09).

■ 왜
행동 정의가 `candidates.py` 모듈 전역 플래그라서, 실험마다 전역을 갈아끼우고 되돌리는
코드가 10여 곳에 흩어져 있다(중간점검 최대 발견). 한 실험이 복구를 빼먹으면 다른
실험이 조용히 다른 정책으로 돈다. 이 모듈은 그 위험을 두 단계로 줄인다:

  ① **불변 구성 객체**(frozen dataclass) — "채택 구성"이 코드에서 이름을 갖는다.
  ② **원자적 적용/복구**(context manager) — 예외가 나도 반드시 원상복구된다.

■ 한계 (정직 고지)
전역 자체를 없애고 생성기에 주입하는 **YR-160 본체는 아직 잔여**다(전 골든 실험 회귀
필요). 이 계층은 그 전까지의 안전장치이며, 병렬 프로세스 안에서 서로 다른 구성을
동시에 쓰는 것은 여전히 불가능하다(전역이므로).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass

from . import candidates as cand_mod


@dataclass(frozen=True)
class ExecPolicyConfig:
    """크레인 행동 정의 플래그 묶음 — 값 자체가 정책 신원의 일부다."""

    wait_mode: str = "WAIT"
    safety_only: bool = False
    bound_repo: bool = False
    prepo_one_shot: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


# 채택 배포 구성 (YR-148 판정: C0 + 유한 DEFER + one-shot — 대기 허가증은 정책 래퍼 몫)
ADOPTED_C0_GUARD = ExecPolicyConfig(wait_mode="DEFER_ALL", safety_only=True,
                                    bound_repo=False, prepo_one_shot=True)
# 구판 기본값 (골든 재현용 — 이 구성은 기각된 축을 포함하므로 성능 판정 금지)
LEGACY_DEFAULT = ExecPolicyConfig()


@contextmanager
def applied(cfg: ExecPolicyConfig):
    """구성을 전역에 적용하고, 예외 여부와 무관하게 원상복구한다 (원자적)."""
    prev = (cand_mod.WAIT_MODE, cand_mod.SAFETY_ONLY,
            cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT)
    cand_mod.WAIT_MODE = cfg.wait_mode
    cand_mod.SAFETY_ONLY = cfg.safety_only
    cand_mod.BOUND_REPO = cfg.bound_repo
    cand_mod.PREPO_ONE_SHOT = cfg.prepo_one_shot
    try:
        yield cfg
    finally:
        (cand_mod.WAIT_MODE, cand_mod.SAFETY_ONLY,
         cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT) = prev


def current() -> ExecPolicyConfig:
    """지금 전역에 걸려 있는 구성 — 스탬프·감사용 스냅샷."""
    return ExecPolicyConfig(wait_mode=cand_mod.WAIT_MODE,
                            safety_only=cand_mod.SAFETY_ONLY,
                            bound_repo=cand_mod.BOUND_REPO,
                            prepo_one_shot=cand_mod.PREPO_ONE_SHOT)
