# 본선 인지 STATE 재학습 결과 — flow_margin (itc-v5) (2026-07-24)

> 실험: 반응형 RL(yr088_joint_rl)에 **flow_margin 을 STATE에 추가**(SCHEMA itc-v4→itc-v5)하고
> **같은 리워드·seed·200 episode**로 재학습 → 본선(berth)이 개선되는가. 리워드는 안 바꿈
> (v6 리워드축은 이미 실패). 즉 **"더 나은 본선 STATE가 학습을 여는가"**의 단독 검정.

## 1. 배경 — 축이 갈렸다

- **리워드축(v6)**: sts_wait 지배 재조정 → **실패**(berth 전 셀 악화·건전깨짐·미완주). 대리신호 악용.
- **STATE축(이번)**: flow_margin(STS 굶/막힘 여유 요약)을 관측에 추가. 리워드 불변.

## 2. 결과 (test, RL vs SF, berth=선석초과 분)

| 셀 | itc-v4 RL/SF | **itc-v5 RL/SF** | Δberth(v4→v5) |
|---|---|---|---|
| mid-loose | 27.4 / 29.4 | **11.8 / 29.4** | **−15.6** |
| high-loose | 68.9 / 62.1 | 68.4 / 62.1 | −0.5 |
| mid-tight | 108.8 / 109.6 | **102.1 / 109.6** | −6.7 |
| high-tight | 145.2 / 161.4 | **148.8 / 161.4** | +3.6 |

- **평균 berth: RL 87.6 → 82.8분** (SF 90.6). itc-v5 RL 이 **SF 를 berth 에서 3/4 셀 + 평균으로 이김**.
- **트럭 유지**: 평균대기 RL 8.51 (SF 9.04) — 본선 개선이 트럭 희생 아님. 전 셀 **healthy·완주 True**.

## 3. 판정 — STATE축은 (모듈하게) 효과 있음

**flow_margin 을 STATE에 넣으니 본선(berth)이 개선됐다** — 리워드로는 못 하던 걸 **더 나은 관측**이
부분적으로 열었다. 리워드축(v6 실패) vs STATE축(이번 개선)의 대비가 **"본선은 보상이 아니라 관측/표현
문제"** 라는 이번 세션 가설을 지지한다.

**정직한 한계 (과대해석 금지)**:
- **모듈함**: 평균 −4.8분(~5%). "본선 완전 해결" 아님. mid-loose 만 크게(−15.6), high-loose 평평,
  high-tight 오히려 +3.6.
- **통계 미확증**: 셀당 6 test seed 평균·**paired CI 없음**·방향 혼재(3↓1↑). 확증은 seed 확대·CI 필요.
- **조기 정체**: best_ep=20 (v4·v5 공통) — 20→200 episode 추가 학습이 val 개선 없음. RL 이 일찍 정체.
- **lookahead 잔여**: 개선이 모듈한 건 본선이 **관측 + 미래계획** 둘 다 필요하다는 앞선 결론과 정합 —
  STATE 개선이 절반을 열고, 나머지(경합 미래)는 여전히 rollout/credit 축.

## 4. 함의·다음

- **확인된 것**: 본선 심화의 유효 레버 = **관측(STATE) 표현** > 리워드 shaping. flow_margin 채택 가치 있음.
- **미확증**: 개선폭의 통계 유의성(paired CI·더 많은 seed). high-loose 무개선·high-tight 악화 원인.
- **다음 축**: ① 개선 확증(seed↑·CI) ② 남은 lookahead 축(위험시 rollout=하이브리드·n-step credit)
  ③ per-candidate 본선효과(vessel_prep 후보와 결합) ④ 본선판 ETA 선제준비 결합.
- 산출물: `outputs/reports/yr088_rl_itcv5/`(rl_net.pt·results.json, 미커밋 — 재생성 가능).
  전 결과 **문헌 보정 시뮬 조건**.
