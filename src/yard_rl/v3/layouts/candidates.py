"""후보 예산을 **실행 가능한 일**에 먼저 준다 (옵트인).

기본 생성기는 점수만 보고 예산을 채우므로, 점수 높은 **실행 불가능** 후보가 12칸을
다 먹으면 실행 가능한 후보가 통째로 잘린다. 처음에는 크레인 역할이 나뉜 배치에서
먼저 드러나(YR-317-h) 이 모듈에 만들었는데, [[YR-317-j]] 진단에서 **기본 야드도 같은
결함으로 멈춘다**는 것이 확인됐다.

실측(시드 21,000,000 · 블록 Y17 · 재배정 없음): 간섭 교착에서 탈출용 REPOSITION 3건이
전부 실행 가능했는데도 실행 불가능한 SERVE 11건에 밀려 사라졌고, 크레인은 22.6일 동안
WAIT 만 골랐다. 예산을 200으로 늘리면 같은 3건이 그대로 나타난다 — 생성이 아니라
**가지치기**가 원인이다.

⚠️ **여기 두는 이유**: `v3/world/` 는 v1 엔진의 **사본**이고 확장 3파일을 뺀 전부가
바이트 동일해야 한다(`tests/v3/test_world_clone.py`). 그 규약을 깨지 않으려면 이
파생 생성기는 사본 바깥, 즉 v3 전용 모듈인 여기에 있어야 한다.

점수·마스크·정준 동률 규칙은 건드리지 않는다. 바뀌는 것은 비필수 후보의 정렬에
`feasible` 을 1순위 키로 넣는 것과, 그래도 실행 가능한 후보가 하나도 안 남으면 가장
좋은 것 하나를 더 싣는 것뿐이다. 기본 생성기는 그대로라 기존 결과는 재현된다.
"""
from __future__ import annotations

from ..world.integrated.candidates import CandidateGenerator, GenCandidate


class FeasibleFirstCandidateGenerator(CandidateGenerator):
    """Preserve mandatory work and at least one available feasible candidate."""

    def _prune(self, raw) -> list[GenCandidate]:
        # WAIT is appended by inherited generate(), outside this work budget.
        budget = self.k_max - 1
        mandatory = [candidate for candidate in raw if candidate.mandatory]
        rest = [candidate for candidate in raw if not candidate.mandatory]
        rest.sort(key=lambda candidate: (
            not candidate.feasible, -candidate.score,
        ) + self._order_key(candidate))
        kept = mandatory + rest[:max(0, budget - len(mandatory))]

        # Mandatory overflow must not hide every executable action. Candidate
        # sets already permit overflow to preserve mandatory work; add just one
        # best feasible action when necessary, without changing scores or masks.
        if not any(candidate.feasible for candidate in kept):
            best_feasible = next((candidate for candidate in rest
                                  if candidate.feasible), None)
            if best_feasible is not None:
                kept.append(best_feasible)
        return kept


#: 원래 이름 — 수직형 배치 실행기와 그 시험이 import 를 바꾸지 않도록 남긴다.
FeasibleCandidateGenerator = FeasibleFirstCandidateGenerator

__all__ = ['FeasibleCandidateGenerator', 'FeasibleFirstCandidateGenerator']
