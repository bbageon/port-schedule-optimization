# YR-101 — ensemble quote (공개 오차분포 K표본 평균 marginal)

- **Epic**: RL / **Priority**: 🟠 / **등록**: 2026-07-27 / **착수**: 2026-07-27 (사용자 지시)
- **상위**: [[YR-099]] transfer — G1 재검토조건③ 발동의 1순위 처방
- **근거**: [YR-099-G1 판정](../../outputs/reports/yr099_transfer_mvp/report.md) — 오라클
  −22.41 [−32.1,−12.7] 8/8 = 상금 실재(총비용 ~16%), 단일 예측궤적 quote 2/8 무상관.
  원인 = 예측↔실현 궤적 발산(berth 33× 분기 민감) — marginal 카오스 아님(오라클이 기각).

## 처방 (무엇)

단일 예측 시나리오 대신 **공개 오차분포에서 K개 예측 시나리오를 추출**해 marginal 을 평균:

```text
표본 k: 실제 게이트진입 := 예약 + clip(N(0, 준수σ), ±2σ)   ← 생성기와 같은 분포族
        실제 블록도착 := 진입 + trunc_normal(주행 μ, σfrac)
        (전용 RNG 스트림 "ens-*:{seed}:{k}" — 실현 draw 열과 완전 분리 = 누출 0)
OutRelief(A,j) = mean_k [J_k(A with j) − J_k(A without j)]
InBurden(B,j)  = mean_k [J_k(B with j′) − J_k(B base)]
Gain = OutRelief − InBurden − Route − Margin,  부가지표 pos_frac(표본 중 Gain>0 비율)
```

분포 파라미터(준수 σ·주행 μ·σfrac)는 터미널이 아는 운영 통계 = 공개정보 등급
(YR-087 현실형 rollout 의 K표본 관례와 동일). 실현 draw 는 절대 미참조.

## 사전등록 (결과 미열람 동결 — 2026-07-27)

- **설정**: K=5 · GAIN_MARGIN=0.5(기존 동일) · epoch당 transfer ≤1 · 같은 8 paired seeds·
  셀(BASE 833000/834000, high-tight+mid-loose) — G1·오라클과 직접 비교 가능.
- **pick 규칙**: eligible 전수에서 **mean-Gain 최대 & mean-Gain > MARGIN** (pos_frac 은
  보고용 부가지표 — 선택 규칙에 미사용, 사후 분석만).
- **주판정**: ① 부호일치(예측 mean-Gain>0 ↔ 실현 d<0)가 단일-quote 2/8 대비 개선되나
  ② G1 d_total CI — **상한<0 이면 "상금 회수 개시"** / CI 0 포함이면 "미달(단 부호일치
  개선 여부 별도 보고)" / 평균>0 이면 실패. ③ 오라클 대비 회수율 = mean_d/−22.41 보고.
- **guard**: 완주·G0 불변식(소유권·보존) 유지. KEEP 도 정상 결과.
- **한계 고정**: t=0 review·반입만·SF 고정 — G1 MVP 와 동일 (변인 = quote 방식 하나).
- 실패 시 다음 순서(spec): 중앙 joint scorer 검토. K=11 민감도는 compute 여유 시 부가.

## 산출

`outputs/reports/yr099_transfer_mvp/results_ens_K5.json` + report.md 갱신 + board 판정.
