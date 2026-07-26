# YR-098 — 통제권한 감사 (authority-null)

> **수정 아닌 측정.** 블록 Q 가 타이밍만으로 딸 수 있는 본선 여지가 **어디(적하만)에 얼마나** 있는지 CI로 확정. [[YR-097]]·[[YR-090]]을 LOAD 전용으로 스코핑하는 선결 게이트.
> 상위 index: [backlog.md](../../Dashboard/backlog.md).

## 왜 (근거)

적대검증 워크플로(2026-07-26)의 **통제권한** 렌즈가 정직한 null("블록 Q 는 타이밍만 → 본선 여지 없음")을 코드로 검증해 **절반만 참**임을 발견:
- **양하** — 레버 0. `yard_handover_cap=None`([engine.py:65](../../../src/yard_rl/integrated/engine.py#L65))이라 크레인 적재 지연이 STS 에 역압력 안 감. sts_wait 전 시드 0. 상위층(TOS/YT) 소유.
- **적하** — 레버 실재. 굶주림당 released 적하 job 4~5건 대기 중 크레인이 트럭 서빙. **학습 0인 한 줄 규칙(VesselFirst)이 전 셀 numeraire 로 SF 를 이김**(high-tight 113.8→105.8).

→ 본선 fix 를 **적하에만** 걸어야 "통제불가 양하에 신호" 4번째 실패를 사전 제거. 그 스코핑을 숫자로 확정한다.

## 무엇을 (측정 설계)

학습기 **무변경**(Q·손실·인코딩·보상 0 수정). 고정 resolver preference 4팔을 기존 `run_joint_episode`에 통과:
- **SF**(기준) · **VesselFirst**(무조건 본선 우선) · **LoadOnlyFirst**(적하만 tier0) · **ConditionalVesselFirst**(`flow_margin_s<0`일 때만 tier0).
- 각 (cell,arm,seed): `total_cost`·`berth_overrun_min`·`mean_wait`·`completion`·`term_contrib` 수집.
- `sim.vessels` 순회로 `max(0, actual−planned)`를 **load/disch 버킷 분해**.
- `transfer.n_units∈{3,6,12,24}` 오버라이드 **YT 스윕**(적하 병목이 YT-bound 아님 재확인).
- `_paired_ci`로 (VF−SF)·(Cond−SF)·(Cond−VF)를 `total_cost`·`berth` 양쪽 CI.

새 파일 `yr098_authority_null.py`. 재사용: `yr080e.VesselFirstServe`·`baselines.run_joint_episode`·`yr071._paired_ci`·`yr088.CELLS/BASE/RC`. 신규 소형 2클래스(`BaselinePreference.rank` 확장): `LoadOnlyFirstServe`·`ConditionalVesselFirstServe`.

## 판정 게이트 (조건)

- **null 반증 게이트 = 최선 규칙팔 − SF** (무조건 VF−SF 아님). VesselFirst 의 트럭 손상이 적하 여지가 작지만 실재하는 셀에서 **거짓 NO-GO** 를 내 본선 라인을 잘못 폐기하는 것을 방지. Cond−SF 를 이미 수집하므로 게이트 정의만 이렇게.
- **GO 의 의미 명시**: "행동공간에 여지 존재"이지 "학습가능성"이 아님. 감사는 곧 fix 가 아니라 **다음 단일메커니즘 최적화(YR-097·YR-090)의 스코핑 신호**.

## 산출

`outputs/reports/yr098_authority_null/{results.json, rows.jsonl, report.md}` — 셀별 load/disch 분해 berth·규칙팔별 numeraire CI·YT 스윕 불변성.
