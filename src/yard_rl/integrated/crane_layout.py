"""블록당 크레인 수 구성 컴포넌트 — 크레인 개수를 선택 가능한 다이얼로 모듈화.

환경 구성 3축의 하나: **터미널(어디)** × **혼잡도(얼마나)** × **크레인 수(몇 대)** → 시나리오.
`crane_layout(n).apply(profile)` 가 프로파일의 크레인을 N대로 바꾼다.

★정직성 게이트 (두 겹):
  1. **정책 호환**: 채택 FT(student_ft.pt)는 **2크레인 슬롯·214입력 고정**이라 N≠2 는 그대로
     못 쓴다 (policy_compatible=False). N≠2 는 baseline(SF-SPT 등)만 실행 가능 — FT 는 재학습
     필요(가변 크레인 = YR-081).
  2. **용량 충실**: 현 엔진은 다크레인 용량 스케일링이 **미충실**하다 — 고정 이송 fleet·단일
     인계행·bay 부하불균형으로 N↑ 가 처리량↑ 이 아니다(실측: split N=3 이 N=2 보다 대기 큼).
     실 다크레인(인계 zone·크레인별 이송·교차 handoff)은 YR-081 구조확장.

따라서 faithful=True 는 **N=2·shared**(FT 학습·보정 구성)뿐. N≠2 는 실행되지만 warnings 동반,
성능·용량 주장 금지.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .profile import IntegratedProfile

_PARTITIONS = ("shared", "split")


@dataclass(frozen=True)
class CraneLayout:
    """블록당 크레인 수 + bay 분담 방식.

    shared: 전 크레인이 블록 전 bay 담당(현 2크레인 구성 — 비통과 간섭 resolver 몫).
    split : bay 를 크레인 수로 등분해 각자 전담(간섭↓·부하불균형↑, N>2 권장이나 여전히 미충실).
    """

    n_cranes: int
    bay_partition: str = "shared"

    def __post_init__(self) -> None:
        if self.n_cranes < 1:
            raise ValueError("n_cranes 는 1 이상")
        if self.bay_partition not in _PARTITIONS:
            raise ValueError(f"bay_partition 은 {_PARTITIONS} 중: {self.bay_partition!r}")

    @property
    def policy_compatible(self) -> bool:
        """채택 FT(2크레인 슬롯·214입력)를 무재학습으로 쓸 수 있는가."""
        return self.n_cranes == 2 and self.bay_partition == "shared"

    @property
    def faithful(self) -> bool:
        """엔진이 이 구성을 충실히 실행하는가 (= FT 보정 구성 N=2·shared 뿐)."""
        return self.policy_compatible

    def warnings(self) -> tuple[str, ...]:
        if self.faithful:
            return ()
        w = [f"FT 채택정책은 2크레인 슬롯 고정 — N={self.n_cranes} 은 무재학습 불가. "
             "baseline(SF-SPT 등)만 실행, FT 는 재학습 필요(가변 크레인 = YR-081)."]
        if self.n_cranes == 1:
            w.append("단일 크레인 = 공동경합 없음 — FT 의 2크레인 협조구조 부적용.")
        elif self.n_cranes > 2:
            w.append("현 엔진은 다크레인 용량 스케일링 미충실 — 고정 이송 fleet·단일 인계행·"
                     "bay 부하불균형으로 N↑ 가 처리량↑ 아님(실측 확인). 실 다크레인은 YR-081.")
        return tuple(w)

    def _ranges(self, bay_count: int) -> list[tuple[int, int]]:
        if self.bay_partition == "shared":
            return [(1, bay_count)] * self.n_cranes
        out = []
        for i in range(self.n_cranes):
            lo = i * bay_count // self.n_cranes + 1
            hi = (i + 1) * bay_count // self.n_cranes
            out.append((lo, max(lo, hi)))
        return out

    def apply(self, profile: IntegratedProfile) -> IntegratedProfile:
        """프로파일의 크레인을 N대로 교체 (N=2·shared·기존 2크레인 프로파일이면 identity)."""
        if (self.n_cranes == len(profile.cranes) and self.bay_partition == "shared"):
            return profile                         # 현 구성 보존 (FT golden-safe)
        base = profile.cranes[0]
        ranges = self._ranges(profile.block.bay_count)
        cranes = tuple(
            replace(base, crane_id=f"YC-{i + 1}", service_bay_min=lo, service_bay_max=hi)
            for i, (lo, hi) in enumerate(ranges))
        return replace(profile, cranes=cranes)


def crane_layout(n_cranes: int = 2, bay_partition: str = "shared") -> CraneLayout:
    """블록당 크레인 수 선택 → CraneLayout.

    예) crane_layout(2)                       # 현 구성 (FT 호환·faithful)
        crane_layout(1)                       # 단일 크레인 (baseline 전용)
        crane_layout(3, bay_partition="split") # 3크레인 bay 분담 (실험·미충실)
    """
    return CraneLayout(n_cranes=n_cranes, bay_partition=bay_partition)
