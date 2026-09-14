# YR-317 — 원고에 반영할 정정 문안

2026-09-14. 투고본 식별 전 검토 가능한 문안. `submission/`과 `submissionv2/`는 아직 수정하지 않았다.
실행하지 않은 반복 실험이나 사용성 평가를 완료했다고 서술하지 않는다.

## 통계 단위 — YR-317-a

> The original evaluation used one continuous 30-day trajectory per condition,
> with outcomes summarized over its middle 28 days. Because the terminal state
> carries over between days, these daily outcomes are dependent. We therefore
> treat the daily comparisons as descriptive and do not interpret their sign-test
> or signed-rank p-values as evidence from independent replications. A complete
> independently generated monthly trajectory is the unit of replication for
> subsequent evaluation.

뜻: 기존 28일을 독립 표본 28개로 세지 않는다. 독립 월 반복을 실제로 마친 뒤 그 방법·결과를 별도 문단에 넣는다.

## 초기 모형·학습 일수 — YR-317-a

> The early checkpoint was saved after the first day's parameter updates; it
> was not an untrained initialization. Its comparison with the final checkpoint
> measures the observed difference between these two training stages, rather than
> improvement over a zero-update model. The original implementation could update
> the networks on boundary days when labels were available. The exclusion of the
> first and last days applied to outcome summaries and should not be read as an
> exclusion from parameter updates. Future training runs record the initialization,
> initialization seed, actual update days, and optimization-step counts separately.

뜻: 과거 실험의 학습 효과를 소급해서 복원하지 않는다. 새로운 비교에는 동일 학습 실행에서 저장한 진짜 초기값을 사용한다.

## 순위 설명 — YR-317-c의 검증을 앞둔 수학적 범위

> In the implemented seller teacher, each sampled reallocation action is compared
> with keeping the current assignment at the same decision state. If their costs
> are C(a) and K, the pair-centered target for the reallocation is proportional to
> [C(a) − K]/2. Under an identical reference state and continuation, these ideal
> reallocation targets preserve the ordering of the corresponding costs. However,
> the KEEP target is proportional to [K − C(a)]/2 and depends on the alternative
> sampled. This algebra does not establish the accuracy of learned rankings across
> unseen candidates or the threshold for choosing KEEP; these require separate
> validation.

뜻: 후보 간 이상적인 목표 순서는 설명할 수 있지만, 학습된 점수의 정확성과 변경 여부까지 증명한 것은 아니다.

## 거절권과 수락망 — YR-317-d의 비교 정의

> The no-veto variant forces acceptance but retains the acceptance network's
> predicted score in the resolver's ordering. It therefore isolates the veto
> decision, rather than removing the entire acceptance network. A full-network
> ablation must also replace this learned ordering with a prespecified ordering
> independent of that network, while retaining the simulator's existing validity checks.

뜻: 거절 제거와 망 전체 제거를 한 실험으로 취급하지 않는다. 추가 비교 결과가 나오기 전에는 이 문단이 방법 정의다.

## 예약·변경 부담 — YR-317-b, 사용자 결정

> The simulator does not fully represent appointment-slot capacity or the economic
> burden of rescheduling for carriers. Deployment would require a feasibility
> constraint layer to exclude changes that cannot be executed given notice periods,
> vehicle movements, existing assignments, and appointment capacity. Incorporating
> rescheduling and external-waiting costs, and evaluating net benefits under these
> constraints, are left to future work. The reported reductions concern the costs
> currently represented in the simulator and do not establish deployment-level
> net savings.

뜻: 물리적으로 실행할 수 있는지와 비용을 감수할 만한지를 구분한다. 리뷰어의 현실 제약 실험 요구를 전부 충족했다고 답하지 않는다.

## 사용자·업무 맥락 — YR-317-f

> A prospective use context is decision support for terminal operations planners
> and carrier dispatchers. Terminal planners would review proposed yard and arrival
> changes alongside the current operating state. Carrier dispatchers would assess
> whether an arrival change is compatible with vehicle schedules, and drivers
> would receive confirmed instructions. The software Buyer represents a destination
> policy in the simulator; it is not a model of human consent. Human approval,
> explanation, and operational-system integration are proposed deployment functions,
> rather than implemented or evaluated capabilities of this study.

뜻: 대상 사용자·역할을 명시하되 실제 사용자 연구나 사람의 동의를 검증했다고 쓰지 않는다.
