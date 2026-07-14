# 전략 히스토리

> 실험 전략의 의사결정 시점별 원문을 보존한다. 현재 구현 상태는 Dashboard가 원본이며,
> 이 폴더의 문서는 설계 스냅샷이므로 구현·검증 완료를 뜻하지 않는다.

| 기록일 | ID | 상태 | 핵심 결정 | 문서 |
|---|---|---|---|---|
| 2026-07-13 | YR-027 | 구현·평가 완료, primary 미통과 | Direct-Job Cost-Q가 shortest-service보다 평균 +0.039분, fallback 55.0%로 coverage 부족 | [전략·결과](2026-07-13-YR-027-exp1-direct-job-cost-q.md) |
| 2026-07-13 | YR-027 v2 | 평가 완료 — coverage 통과, primary 미통과 | 최소상태로 fallback 0.01% 달성했으나 순수 Cost-Q 순서가 shortest-service 열세 (+1.195분) | [v2 최소상태](2026-07-13-YR-027-exp1-direct-job-cost-q-minimal-state.md) |
| 2026-07-14 | YR-030 | 방향 결정 (사용자) — 구현 전 | 계열 2(Direct-Job Cost-Q)를 실험 baseline 으로 승격, 상태 v3·학습설정·후보필터 확장 실험. 계열 1은 PoC 증거로 동결 | [전략](2026-07-14-YR-030-series2-baseline-pivot.md) |
