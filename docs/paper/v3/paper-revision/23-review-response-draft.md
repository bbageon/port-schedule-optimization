# 전체 리뷰 답변서 초안과 원고 반영 지도

2026-09-18. 대상: [통합 개정 원고](main-integrated.tex). **실험 결과가 모두 확보되기 전의 답변 초안이며 최종 제출본이 아니다.**
아래 완료형은 원고의 설명·주장 수정에만 사용한다. 진행 중인 20시드·4정책·80개 실행의 성능 수치는 넣지 않았다.
원고 위치는 9/18 통합 PDF 12쪽의 실제 절 번호·시작 페이지와 대조했다. 뒤에 실험 표를 넣으면 다시 갱신한다.
기존 제출 원고와 부분 개정 원고는 보존한다. 새로운 실험의 완료나 제출 전 추가 실험 일정을 약속하지 않는다.

## English response draft

We thank the reviewers for identifying gaps between the proposed architecture and the evidence supporting its claims. The revision distinguishes historical single-trajectory observations, changes to the explanatory text, the independent evaluation currently in progress, and validation that remains unavailable. The historical performance tables are retained as descriptive results under their original experimental conditions. They are not presented as independent replications or evidence of deployment readiness.

### Reviewer 1, comment 1 — independent replications and dependent days

**Concern.** The 28 reported days belong to one continuous trajectory and do not constitute independent tests.

**Response.** We agree. We have withdrawn the use of day-level tests and day-resampled intervals as evidence of independent-run statistical reliability. The revised evaluation protocol treats one continuous month as the independent unit and retains the dependence between its days. It specifies 20 independently initialized environments, with the same input realization paired across the baseline, Full, Time-only, and Block-only policies. Every run uses a separate engine and process, fixed learned weights, and no additional training or exploration. The 28-day paper window and the complete 30-day window are distinguished; the existing two-hour closing interval is also disclosed.

The protocol resamples 20 paired months 20,000 times. In the original three-policy comparison family, baseline-minus-Full and Time-only-minus-Full are the two primary 28-day comparisons, each with a 97.5% interval; auxiliary comparisons use descriptive 95% intervals. Block-only was registered as a later addition after the original evaluation had started. Its baseline-minus-Block-only and Block-only-minus-Full comparisons form a separate two-comparison family, with 97.5% intervals for the 28-day window and descriptive 95% intervals for the 30-day window. We do not describe all four policies as having been registered together initially or claim that the separate families provide one combined family-wide error guarantee.

We also disclose an environmental change: `PRESERVE` retains declared demand instead of excluding requests when immediate admission conditions fail, while `COUNT_BALANCED` adjusts selected vessel loading/discharging directions to address the identified quantity imbalance. Truck arrival curves, daily volumes, and initial inventory are retained, and the revised vessel plan is shared across policies. This is a fixed-model evaluation under the corrected admission/supply contract, not an exact repetition of the original environment or a retraining result. Unfinished work remains an outcome and is not a reason to remove a seed.

**Evidence and location.** E1–E3; §4.1 (p. 7), §5.1 (p. 8), and §5.4 (p. 10).

**Result.** The 80 runs are complete (Table 3, §5.4). Time-only was cheaper than no reallocation in 16/20 months (median 12.7%); Full in 4/20 and Block-only in 3/20; every pre-registered mean-difference interval includes zero. Nine runs in seven seeds, including one baseline run, ended in a simulator crane deadlock (two idle cranes trapped within the safety gap at a block end with no move-aside action) that inflates monthly cost 5–10 fold. With pairs containing an unresolved deadlock of at least 24 h removed by a mechanical criterion, Time-only is cheaper in 15/15 months with a 97.5% interval of [+1.27, +2.03] billion KRW, while Full and Block-only are significantly more expensive. Per-seed costs: [32-독립시드별-결과표.md](32-독립시드별-결과표.md); deadlock diagnosis: [31-정지-결함-진단.md](31-정지-결함-진단.md).

**Outstanding.** The deadlock is a simulator defect that must be repaired before the affected months can be re-run; the deadlock-free rows are a validity filter, not a registered analysis. Unfinished work remains higher under every learned policy.

### Reviewer 1, comment 2 — appointment limits and rescheduling costs

**Concern.** Unbounded appointment slots and omitted carrier costs may overstate the practical benefit of time changes.

**Response.** We have clarified the distinction between modeled terminal costs and executable transport schedules. Block-capacity checks are applied, but appointment-slot quotas remain unbounded in the reported evaluations. The temporal candidate range does not establish carrier availability, an operational minimum notice period, or compatibility with subsequent trips. The present objective measures truck time from actual gate entry to exit or the evaluation cutoff; it does not add outside-gate waiting, rescheduling charges, or subsequent-trip disruption. The code audit confirms that recording a deferral does not mean its external cost is included in the objective.

The revision cites truck appointment and collaborative scheduling literature and identifies booking quotas, carrier-approved change windows, notice requirements, and cargo/vessel deadlines as deployment conditions requiring terminal and carrier data. Their integration into an operational feasibility layer and the associated economic evaluation are described as future work. The reported terminal-cost reduction is therefore not presented as verified net savings for terminal operators and carriers together.

**Evidence and location.** E4; related work, §3.2 (p. 4), §4 (p. 6), and §6 (p. 11).

**Outstanding.** This is a partial response: no experiment with realistic appointment quotas or added rescheduling charges is supplied. Benefit retention and a break-even change charge remain unestablished.

### Reviewer 1, comment 3 — pairwise labels and candidate ranking

**Concern.** Scores learned from separate action comparisons need not reliably rank the complete candidate set.

**Response.** We have clarified the actual labeling mechanism. The proposal learner compares KEEP with a selected change, rather than relying on an arbitrary collection of unrelated action pairs. For a common state, exogenous trajectory, continuation policy, and horizon, let K denote the KEEP cost and S a positive scaling constant. The ideal target for a changed candidate a is [C(a) − K]/(2S), which preserves the cost ordering among changed candidates sharing that reference. This conditional property does not guarantee the ranking produced by an approximating network.

The KEEP target is [K − C(a)]/(2S) and therefore depends on the alternative used in its comparison. We now distinguish ranking changed candidates, deciding between KEEP and a change, acceptance errors, and the ordering of competing proposals. The commitment procedure sorts consented proposals using predicted acceptance costs; under resource contention, that ordering can affect the committed set and is not merely a harmless tie-breaking convention.

**Evidence and location.** E5; §3.3 (p. 4), §3.2 (p. 4), §5.3 (p. 9), and §5.4 (p. 10).

**Outstanding.** Full-candidate validation against simulated costs, including selection regret, rank agreement, missed beneficial changes, and harmful changes, has not been completed. Existing small counterfactual wiring checks do not establish candidate-ranking accuracy. The explanatory correction is not presented as a substitute for that empirical validation.

### Reviewer 1, comment 4a — contribution of spatial adjustment

**Concern.** The historical Full result is close to Time-only, while Block-only increases aggregate cost.

**Response.** We have retained the original ablation results and made the load-dependent interpretation explicit. The historical low-volume and high-volume groups suggest different roles for spatial and temporal changes, but do not establish a universal low-load/high-load rule or consistent superiority of the joint model. Planned daily volume is distinguished from realized congestion because queues and unfinished work carry over from earlier days. The descriptive load-group comparison does not treat its days as independent samples.

The independent protocol now includes Block-only alongside the other three policies. Full versus Time-only tests the incremental effect of permitting spatial actions in the fixed model; Block-only versus the baseline tests spatial changes in isolation. These restricted policies share the same learned weights and are not separately optimized policies. The later addition and its separate statistical comparison family are disclosed above.

**Evidence and location.** E1, E2, and E6; §4.1 (p. 7), §5.2 (p. 9), and §5.4 (p. 10).

**Result.** Spatial adjustment does not replicate: Full was cheaper than the baseline in 4/20 months and Block-only in 3/20, and with deadlocked pairs removed both are significantly more expensive (Table 3). Block changes also left request targets unbound in most seeds under the corrected contract. The manuscript now reports this as a negative result and makes time-only adjustment the main claim.

**Outstanding.** Whether the unbound targets stem from the block-change policy or from the request-preservation contract is not yet diagnosed; no claim that spatial adjustment improves total cost is made.

### Reviewer 1, comment 4b — contribution of the acceptance network

**Concern.** A separate acceptance network is presented as part of the architecture without sufficient evidence of its additional value.

**Response.** We have separated the proposed architectural role from the evidence needed to justify it. The acceptance mechanism both permits or rejects a proposal and supplies scores used to order competing proposals. Turning off the veto while retaining those scores is not equivalent to removing the acceptance network. The current four-policy independent comparison retains the acceptance mechanism in the learned policies and therefore does not isolate its contribution. The first-iteration comparison is also not an acceptance-network ablation.

**Evidence and location.** E5 and E6; §3.3 (p. 4), §3.2 (p. 4), §5.3 (p. 9), and §5.4 (p. 10).

**Outstanding.** An independent multi-seed comparison isolating the veto and a complete network-removal comparison are unavailable. Earlier request-accounting diagnostics containing a no-veto condition are not substituted for this independent component evaluation.

### Reviewer 1, comment 5a — concentrated savings and robustness

**Concern.** Most historical savings arise on a few congested days; demand patterns, costs, and horizons require broader examination.

**Response.** We have retained the concentration of savings and the existing controlled variation in congested-day frequency as descriptive evidence, while making their single-realization scope explicit. The current independent protocol preserves the original synthetic arrival-curve construction and records daily demand, starting backlog, queues by block, unfinished yard work, turnaround times, action shares, and costs. These observations support interpretation of differing policy trajectories, but are not new independent daily replications.

We distinguish three different quantities that had to remain explicit: the 60-second policy review interval, the three-hour counterfactual observation horizon, and the continuous monthly evaluation period. The 28-day and 30-day totals are two windows from the same run, not a completed robustness test at different counterfactual horizons or longer operating horizons. Low recorded costs are interpreted together with work left unfinished at the cutoff.

**Evidence and location.** E1, E3, E6, and E7; §4 (p. 6), §4.1 (p. 7), §5.2 (p. 9), §5.4 (p. 10), and §6 (p. 11).

**Outstanding.** The independent evaluation is pending. New peak-width, cost-weight, counterfactual-horizon, and extended-operation performance comparisons are not reported as completed. A mathematical inspection of the arrival curve is not a simulation-cost sensitivity result.

### Reviewer 1, comment 5b — online processing time and reproducibility

**Concern.** Implementation details and measured online latency are needed to assess reproducibility and real-time use.

**Response.** We have distinguished the online decision path from offline counterfactual simulation and from the simulator's advancement of physical events. The absence of online counterfactual calls does not establish a millisecond response time. Reproduction records identify the execution source, input and checkpoint hashes, runtime settings, seeds, and saved outcomes. The revised protocol also specifies fixed weights, per-policy input matching, conservation checks, and preservation of unfinished work.

The scope of the missing timing evidence is stated explicitly: candidate construction, proposal inference, acceptance inference, conflict checks, and commitment must be measured as an end-to-end decision cycle, with component timings reported without double-counting nested calls. Whole-month elapsed time and CPU bottleneck samples are not offered as substitutes for this measurement. Communication and human response times remain outside the implemented policy path.

**Evidence and location.** E1, E3, and E8; §4.1 (p. 7), §5.4 (p. 10), and §6 (p. 11).

**Outstanding.** A measured end-to-end online latency distribution and its hardware-qualified relation to the review interval remain unavailable. Reproduction metadata alone does not validate real-time deployment.

### Reviewer 2 — synthetic environment and a single realization

**Concern.** Reliance on a synthetic environment and one 30-day realization limits the evidence for feasibility.

**Response.** We agree and distinguish repeatability within the simulator from validity at an operating terminal. The independent protocol addresses realization-specific effects using 20 environments with matched policy inputs, but its final results are pending and it remains synthetic. We explicitly disclose the corrected admission and vessel-supply contract and its difference from the original training and evaluation conditions. Neither record validity nor quantity balance establishes sustainable operation or field feasibility.

We also clarify what the policy consumes. The networks receive order records and yard-state summaries only (Sect. 3.3); no layout geometry enters them, so the method needs no layout-specific inputs. We therefore do not claim validation of parallel or perpendicular terminals. As a robustness check we trained and evaluated the same procedure in a second synthetic environment with different transfer and crane dynamics; there, the environment-specific time-only policy was cheaper than no reallocation on all five measurement days of one seed while unfinished work increased, and that environment is uncalibrated. We report this only as evidence that the learning procedure adapts to different dynamics, not as a layout comparison. Replication of a particular operating terminal is outside the present scope. The architecture is presented as a simulation-based decision-support concept with unresolved external-validity limitations.

**Evidence and location.** E1–E3 and E9; §4 (p. 6), §4.1 (p. 7), §5.4 (p. 10), and §6 (p. 11).

**Outstanding.** The complete independent results and calibration of the simulator against operating-terminal measurements (turn time, crane utilisation, throughput) are not supplied by this draft. Additional seeds alone would not resolve the absence of field evidence.

### Reviewer 3 — users, decision making, and HCI context

**Concern.** The target users, their operational role, and the interaction context are unclear.

**Response.** We have added an intended-use section positioning the architecture as a recommendation component alongside a terminal operating system. The proposed roles are terminal yard planners and operations staff who review or revise assignments, carrier dispatchers who assess time changes against transport commitments, and drivers who receive confirmed instructions and report execution difficulties. A concrete proposed workflow links operator review, dispatcher coordination, and communication of the final instruction. Human response deadlines are operational design decisions and are not identified with the algorithm's 60-second review interval.

The section cites Busan New Port truck-waiting research, Korean TOS research, and the World Bank/IAPH digitalization report that documents the Busan Port Authority's port community system (vehicle booking, transshipment shuttle, and integrated information services) as application context. These sources do not establish that our system has been integrated into those services or determine their actual approval permissions. Agreement between the proposal and acceptance networks is explicitly distinguished from human consent. The present evidence concerns the computational decision-support component, not interface usability or human acceptance behavior.

**Evidence and location.** E4 and E10; §3.5 (p. 5) and §6 (p. 11).

**Outstanding.** No interface deployment, TOS/port-community-system integration, user study, or improvement in human decision making is claimed. The added context responds to the requested clarification but does not establish an empirical HCI contribution.

## 한글 상태·증거 지도

아래 E 번호는 이 답변서 안의 근거 묶음이다. YR 번호는 저장소 작업표에 등록된 작업 식별자다.

| 근거 | 실제 자료 | 확인 범위와 현재 한계 |
|---|---|---|
| E1 | [독립 반복 계약](20-독립반복-실행계약.md), [원래 사전등록](../../../../outputs/reports/yr317_v3_independent_eval/prereg.md) | 월 단위 통계·고정 가중치·동일 입력·잔여 보존의 실행 계약. 최종 성능 확증은 미완료 |
| E2 | [Block-only 추가](22-Block-only-독립반복-추가.md), [추가 사전등록](../../../../outputs/reports/yr317_v3_block_only/prereg.md) | 나중에 추가한 20개 비교와 별도 비교군의 통계 규칙. 네 정책의 최초 동시 등록으로 쓰지 않음 |
| E3 | [요청 보존](15-요청보존-보정검증.md), [공급 보정](17-공급계획-연결검증.md), [일별 관측](18-독립반복-일별관측.md), [검사 오류·복구](21-독립반복-검사오류와복구.md) | 요청·물리 연결·비용 기록과 환경 변경 근거. 모든 작업 완료·현장 적용성 통과와 구별 |
| E4 | [제약 문헌](02-현실제약-문헌검토.md), [기존 부분 답변](11-리뷰1-2-리뷰3-답변초안.md), [비용 경로 확인](12-시간변경비용-코드확인.md) | 제약·비용 범위 설명은 반영. 현실 제약하 성능·손익분기 단가 실증은 미완료 |
| E5 | [후보 순위 전략](03-후보순위-검증전략.md), [후보 순위 작업 명세](../../../../.claude/docs/dashboard-task-specs/YR-317-c-v3-review-ranking-validation.md) | 공통 KEEP 조건과 남는 오류를 구분. 전체 후보 선택 손실·순위 정확성은 미측정 |
| E6 | [기존 부하별 근거](01-독립반복-부하별효과.md), [기존 성능표](08-성능비교표.md) | 과거 단일 실행의 성능과 별도 진단을 조건별로 보존. 현재 독립 실험 결과로 전용하지 않음 |
| E7 | [피크 폭 검토](19-피크폭-비용민감도-검토.md) | 도착곡선의 수학적 성질과 비용 민감도 설계. 폭을 바꾼 비용 실험은 미실행 |
| E8 | [온라인 시간·재현성 명세](../../../../.claude/docs/dashboard-task-specs/YR-317-e-v3-review-runtime-reproduction.md), [원자료 검사·통계 코드](../../../../scripts/v3/independent_eval_checks.py) | 실행 메타데이터·검산·집계 코드 존재. 전체 온라인 처리시간 분포는 미측정 |
| E9 | [수평·수직 정의](04-수평-수직환경-정의.md), [기본 구조 작업 명세](../../../../.claude/docs/dashboard-task-specs/YR-317-h-v3-review-basic-layouts.md) | 두 구조의 검증 설계. 구현·학습·구조별 실증 완료로 표현하지 않음 |
| E10 | [TOS·올컨e 사용 맥락](05-TOS-올컨e-사용맥락.md), [문헌 목록](06-문헌목록.md), [부분 원고 반영](10-문헌-사용맥락-원고반영.md) | 문헌 인용과 제안 사용자 흐름 반영. 실제 연동·사용자 효과 검증은 아님 |

## 최종 제출 전 대조할 항목

- 통합 원고에 실제 들어간 문장과 위의 완료형을 대조한다. 빠진 수정은 답변을 먼저 완료형으로 유지하지 않는다.
- 현재 위치 표시는 9/18 PDF와 일치한다. 최종 실험 표를 추가한 뒤 절 번호·페이지를 다시 대조한다.
- 진행 중인 독립 실험의 최종 표·신뢰구간이 없으면 해당 대응을 완료로 바꾸지 않는다. 중간 성능으로 결과 칸을 채우지 않는다.
- 오래된 원고의 성능 수치는 당시 환경의 기술 통계로 보존하고, 수정된 접수·공급 환경의 확증 결과와 분리한다.
- 후보 순위·수락망 분리 효과·온라인 시간·두 구조 실증·현실 제약하 편익은 미완료 상태를 그대로 표시한다.
- 리뷰 1-2는 부분 대응, 리뷰 3은 사용 맥락 설명의 보완이다. 전체 리뷰 요구를 모두 해결했다고 결론내리지 않는다.
