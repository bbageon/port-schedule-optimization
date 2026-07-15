# YR-031-b — Oracle 개선 순서 패턴 분석

> ⚠ 가정 프로파일 + 합성 시나리오. oracle(전지적 beam)이 greedy 를 이긴
> 결정들의 구조 분석 — 사용자 가설 H-A(feature 예측가능)·H-B(조합 의존) 판정.

- **H-A (이탈 시점이 관측 feature 로 예측되는가)**: AUC **0.852** (임계 0.75) → **SUPPORTED** (n=10000, 이탈 416건)
- **H-B (이탈 선택이 집합 맥락에 의존하는가)**: 쌍별 AUC 0.993 → +집합맥락 0.993 (이득 +0.000, 임계 0.05) → **REJECTED**

## 이탈의 해부 (divergence taxonomy)

- 이탈률: 결정의 4.2%
- oracle 이 **더 긴 작업**을 고른 비율: 90% (anti-SPT)
- oracle 이 더 오래 기다린 트럭을 고른 비율: 50%
- oracle 이 더 먼 작업을 고른 비율: 39% (포지셔닝형)
- 방향 전환(반입↔반출) 비율: 3%
- 즉시 희생 중앙값: 0.0038분/결정

## 혼잡도 층화 (일 greedy_mean 3분위)

- 일평균 이탈 수: 한산 3.8 · 중간 4.2 · 혼잡 4.5
- 상금의 혼잡 상위 1/3 집중도: 50%

*생성: yard_rl.experiments.oracle_pattern — 원자료 oracle_pattern_results.json·divergence_events.json*