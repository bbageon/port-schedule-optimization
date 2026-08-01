# YR-139 — BlockQ-v4-A: 중앙 공동후보 PPO (학습방식 단일축)
> 상태: **done (2026-08-01 — ★공식 기각·G0 최초 완전 통과·REPO 단일 모드 → v4-B 갈림길)**
> 근거: [v4 전환 기록](../strategy-history/2026-08-01-BlockQ-v4-PPO-전환.md) — YR-138
> 분포 이동 판정의 직격 처방 (자기 궤적 + 실비용 직접 최적화).

## 계약 ([하네스](../../../src/yard_rl/experiments/yr139_blockq_v4_ppo.py) docstring 이 동결 정본)

- **유일 변경 = 학습방식** (Q 회귀 → 중앙 PPO). 상태 인코딩(250)·후보 생성·안전
  resolver·계약 물리·비용계약 전부 현행 그대로. 전략 WAIT 제외(FORBID 유지).
- Actor = 공동후보 행 → logit → mask 후 softmax. Critic = V(상태 구획 174차원).
- 보상 = −ΔΦ (구간 실비용). **등식 테스트**: Σ 구간비용 = Φ(end)−Φ(0) = 평가 총비용
  (v2 실현 hard·end 검열 — 출문 end 밖은 end 검열로 정의 통일·고지).
- γ=1 · GAE(λ=0.95, Critic 잡음용) · clip 0.2 · lr 3e-4 · 엔트로피 0.01 — 표준 앵커
  (PPO 원 논문 기본값 계열·튜닝 아님). Advantage 정규화. 선택 없이 **최종 정책 사용**
  (체크포인트 고르기로 인한 누출·교락 제거).
- 학습: 3 초기화 × 60 iteration × 8 에피소드(4셀 혼합·train 회전 시드) = 480 ep/초기화.
- 평가(동결): 미열람 대역 BASE+2600..2602 (12 ep) — PPO(3 초기화) vs SF 짝지어 완주.
  판정: ①완주 100%·backlog 0 ②3 중 ≥2 초기화에서 v2 실현 총비용 짝 평균 < 0 (방향 —
  유의성은 잠금평가 몫) ③WAIT·REPO 장악 0. **신호 없으면 PPO 트랙도 중단.**
- v4-B(조건부): REPO 장악으로만 실패 시 PREPOSITION(job 연결·만료) 단일축.

## Evidence
사전동결: (커밋 예정) · 결과: outputs/reports/yr139_blockq_v4_ppo/


## 판정 결과 (2026-08-01)

- G0 ✓ 전 판(완주 100%·backlog 0 — 학습 정책 최초). 비용 +15.2/+26.6/+15.7 (방향 0/3) ✗.
  REPO 장악 29/36 ✗ (WAIT 0). A→O +7.6/+14.0/+7.6분.
- 갈림길: 조항 A(중단) vs 조항 B(REPO 장악 → v4-B PREPOSITION 단일축, 사전 등록됨).
  권고 = v4-B 1회 — 실패 시 트랙 중단 확정. 2600 대역 열람됨.
- [report](../../../outputs/reports/yr139_blockq_v4_ppo/report.md)
