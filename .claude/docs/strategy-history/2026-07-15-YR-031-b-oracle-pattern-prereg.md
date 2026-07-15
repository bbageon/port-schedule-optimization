# YR-031-b — Oracle 개선 순서 패턴 분석 사전등록

> 기록일: 2026-07-15 · 상태: **사전등록 — 본 실행 전 동결** · 사용자 승인: "YR-031-b 반영하고 착수해봐"
> 상위: [YR-031 결과](2026-07-14-YR-031-oracle-gap-prereg.md) · 선행: [YR-012-b 결과](2026-07-15-YR-012-b-delta-stable-prereg.md)

## 1. 결정 경위

1. YR-012-b 로 학습 절차 축(초기화→잔차 구조→해상도→안정화)이 **전부 소진** —
   실정책 최강은 여전히 YR-012 online Δ-net (+0.083/+0.107). oracle 상금
   하한 +0.182분은 실재 (YR-031, 혼잡일 편중).
2. **사용자 가설 (2026-07-15 원문 요지)**: "state 는 충분한데 액션이 job 을
   선택하는 단순 구조라 보상 체계가 단순해지고, 결국 패턴 속에서 불확실성과
   최적화할 수 있는 포인트를 못 찾아낸 게 아닐까?" — 논의를 거쳐 두 개의
   검증 가능한 명명 가설로 확정:
   - **H-A (feature 예측가능성)**: oracle 의 이탈 시점("greedy 를 벗어나야 할
     순간")은 결정 시점에 관측 가능한 인과적 feature 로 예측된다.
     참 → 그 feature 가 Δ-net 의 다음 입력 (feature 부족이 병목).
   - **H-B (조합 의존성)**: 이탈 선택은 후보쌍 비교만으로 재구성되지 않고
     후보 집합 맥락에 의존한다. 참 → 후보 독립 스코어링을 넘는 구조
     (집합 인코딩 또는 결정시 탐색) 필요.
3. 보상 단순성은 용의자가 아님을 사전 확인 (YR-018·YR-025 기각 + 비용 항등식)
   — 본 실험은 보상이 아니라 **정보와 표현 구조**를 겨냥한다.

## 2. 방법 (재학습 없음 — 분석 실험)

| 단계 | 내용 |
|---|---|
| oracle 재현 | YR-031 beam 알고리즘 그대로 (`beam_day_with_trace` — 값 반환 로직 비트 동일, 궤적만 추가 반환. 동률 시 greedy 궤적 = 이탈 최소 보수성). W=12, 동일 test band 160000+100 |
| 이탈 추출 | oracle 최적 궤적을 리플레이 — 매 결정에서 greedy(IMMEDIATE_COST_GREEDY) 선택과 대조. 이탈 = 선택 불일치 |
| 결정 feature (H-A 입력, 22dim) | 전역 5 (진행률·크레인·대기·최장·30분+) + 미래 4 + **집합 맥락 8** (후보 수·service min/mean/max·reach min/mean·반출 비율·짧은작업 비율) + greedy 선택 후보 5 — 전부 결정 시점 관측 (인과적) |
| H-A 판정 | 로지스틱 (torch, day-그룹 5-fold CV, 표준화 fold 내, class weight) 로 "이 결정에서 이탈?" AUC. **≥0.75 SUPPORTED / 0.60~0.75 PARTIAL / <0.60 REJECTED** |
| H-B 판정 | 이탈 이벤트만: oracle-greedy 후보쌍 차이 5dim(P 모델) vs +맥락 17dim(S 모델) — 부호 뒤집기 증강 이진화, 동일 CV. **S−P AUC 이득 ≥0.05 → SUPPORTED**. 이탈 <30건 또는 <5일이면 INSUFFICIENT_DATA |
| 기술 통계 | 이탈 해부: anti-SPT(더 긴 작업 선택)·장기대기 구제·포지셔닝(더 먼 작업)·방향 전환 비율, 즉시 희생 중앙값, 혼잡 3분위 층화·상금 집중도 |

## 3. 검토한 대안과 기각 사유

| 대안 | 기각 사유 |
|---|---|
| YR-031 산출물 재사용 (재실행 없이) | per-day 요약만 저장됨 — 궤적 미보존. beam 재실행 불가피 (46분, 수용) |
| 저장된 이탈로 Δ-net 즉시 재학습 (모방학습) | 판정 없이 처방부터 하는 격 — H-A/H-B 판정이 먼저 (feature 인지 구조인지에 따라 처방이 다름) |
| 트리 기반 분류기 (GBM 등) | 신규 의존성. 로지스틱은 torch 로 충분하고 가중치가 feature 중요도로 직접 해석됨 |
| 이탈의 개별 기여도 분해 (counterfactual per-event) | 이벤트당 재시뮬 비용 폭발 — day 수준 improvement 와 상관으로 대체 (기술 통계) |

## 4. 한계 (사전 명시)

- beam 이 찾은 이탈은 **하나의** 개선 경로 — 동일 개선을 주는 다른 경로 존재
  가능 (이탈 패턴은 충분조건의 표본이지 필요조건이 아님).
- H-A 의 AUC 는 "시점 예측가능성"이지 회수 가능 상금의 크기가 아님 — 회수량
  추정은 판정 후 별도 실험 (feature 반영 Δ-net).
- 부호 증강 이진화(H-B)는 쌍별 비교의 대칭 가정 — 표준 관행이나 근사임.
- 합성 + assumed 프로파일 — 실운영 주장 불가.

## 5. 비목표

Δ-net 재학습·feature 반영 (판정 후 별도 사전등록) · beam 폭 확대 · 다른 band ·
YR-031 상금 재측정 (부산물로 재현되나 판정 변경 없음).

## 6. 산출물

`outputs/reports/oracle_pattern_hjnc/` — divergence_events.json (이벤트 원자료,
후속 feature 설계 재료)·per_day.json·oracle_pattern_results.json·
oracle_pattern_report.md. 구현: `experiments/oracle_pattern.py`·
`oracle_gap.beam_day_with_trace` (비트 동일 리팩터)·CLI `run-oracle-pattern [--quick]`.
