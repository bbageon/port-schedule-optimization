# YR-030 — 계열 2 (Direct-Job Cost-Q) baseline 승격 결정

> 기록일: 2026-07-14 · 상태: **방향 결정 (사용자) — 구현 전**
> 관련: [YR-027 v1](2026-07-13-YR-027-exp1-direct-job-cost-q.md) · [v2 최소상태](2026-07-13-YR-027-exp1-direct-job-cost-q-minimal-state.md)

## 1. 결정

신규 RL 실험의 baseline 정책 구조를 **계열 2 (Direct-Job Cost-Q: 후보 단위
(GlobalState, CandidateFeature) 스칼라 비용 argmin)** 로 지정한다. 이후 실험은 이
골격 위에서 **상태 feature·파라미터·학습설정을 추가**하며 진행한다.

계열 1 (rule-선택 tabular QL)은 **폐기가 아니라 동결**: Exp-1~3·YR-018·YR-023 의
PoC 증거(H1 예비지지, w_tail negative 등)는 그대로 보존하되, 신규 실험 라인에서
제외한다.

## 2. 근거 (사용자 결정 논리 + 데이터 뒷받침)

1. **계열 1의 기준 모호성**: rule-선택은 상태마다 "어떤 행동을 왜 골라야 하는지"의
   트레이드오프가 없거나 학습 신호로 보이지 않는다 — YR-018 grid 에서 w_tail 0→1
   전 구간 인접 유의차 0건이 실증. rule 어휘 자체가 정책 표현력의 상한이며, 새
   능력마다 rule 수작업 추가가 필요하다.
2. **계열 2의 기준 명시성**: 매 결정이 "실행가능 후보들의 예상 비용 비교"라는
   단일 명시 기준 — 후보 간 트레이드오프가 결정 단위에서 직접 표현된다.
3. **함수근사 정합**: 구현계획 02 §5.2 의 함수근사 관측(candidate_features[K,F])과
   구조가 동일 — DQN/PPO(YR-012) 전환 시 테이블→네트워크 교체만으로 이어진다.
4. **coverage 통제법 확보**: YR-027 v2 가 최소상태로 fallback 55.04%→0.01% 를
   실증 — 남은 병목은 커버리지가 아니라 **상태 표현력** (v2 가 과빈곤해 열세).

## 3. 실험 축 — "파라미터·학습설정 추가"의 구체화

| 축 | 내용 | 근거·주의 |
|---|---|---|
| 상태 v3 | v1(~102만 서명, fallback 55%)과 v2(2+3 feature, 열세) 사이 중간지점 — wait bucket·reach 등 선별 재도입 | 수렴진단 지표(fallback/thin) 동반 필수 |
| 학습설정 | α=n^-p (p grid), ε 스케줄, checkpoint 선택 정책, train 에피소드 수 | YR-028 ablation 이 선행 (v1 실패 원인 분리) |
| 제약 | SLA 임박 시 후보 필터 (P95 보호) | YR-029 재편 — 보상형 페널티는 폐기 (기준 모호, YR-018 기제) |
| 평가 계약 | paired·locked test·guardrail(P95/completion/backlog)·shortest-service 등 강 휴리스틱 비교군 유지 | YR-027 계약 승계 |

## 4. board 반영

- **YR-030** (umbrella) 신설 · **YR-028** 을 1차 선행 ablation 으로 연결.
- **YR-029** 를 계열 2 후보 필터로 재편 (보상형 후보 폐기 사유 박제).
- **YR-020** 🟠→🟡 — "함수근사 전환조건 판단재료" 역할은 본 결정으로 종결,
  Exp-2/3 열세의 학술적 원인 규명 가치로만 유지.
