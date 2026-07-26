# YR-099 MVP G1 — t=0 반입 재배정 headroom 판정 (2026-07-27)

## 설정

2블록(A=high-tight+본선압박 · B=mid-loose 여유), local policy=SF 고정, t=0 review 1회,
epoch당 transfer ≤1, 전수 quote(Gain=OutRelief−InBurden−Route−Margin), 8 paired seeds.
엔진 무변경(시나리오 수준 원자 이동) — G0 소유권·보존·결정론·정보안전 테스트 6 통과.

## 결과

| arm | Δterminal(C−A0) | 부호일치(예측 Gain vs 실현) |
|---|---|---|
| **C (공개정보 예측-quote)** | **+3.08 [−4.55, +10.71] — 상금 없음** | **2/8** |
| 오라클 (참-시나리오 quote, 진단 전용·배포 금지) | **−22.41 [−32.12, −12.70] — 8/8** | 8/8 |

## 판정 (정직)

1. **재배정 상금은 크게 실재한다** — 오라클 8/8 시드 개선, 평균 −22.4 numeraire
   (A0 총비용의 ~16%). 블록 간 배정은 블록 내 타이밍 여지(VF −2.84)의 **~8배** 크기.
2. **그러나 공개정보 단일 예측궤적 quote 는 그 상금을 못 잡는다** — 예측세계 marginal
   (7~36)이 실현세계와 무상관(2/8). 원인 = marginal 카오스가 아니라 **예측↔실현 궤적
   발산**(berth 33× 분기 민감도): 오라클 성공이 카오스 가설을 기각.
3. spec 재검토 조건 ③("scalar quote 부호·순위 반복 오판") **발동** — 단 처방은 quote
   강건화(ensemble: 공개 ETA 오차분포에서 K개 예측표본 추출·marginal 평균)이 1순위,
   중앙 joint scorer 는 그 다음. → 후속 YR-101 등록.
4. 한계(정직): t=0 review 1회·반입만(양하 cross-block 은 엔진 브리지 필요)·GAIN_MARGIN=0.5
   에서 8/8 transfer 실행(margin 이 노이즈 quote 를 못 거름 — ensemble 전 margin 상향 무의미).

원자료: results.json(예측-quote) · results_oracle.json(오라클)
