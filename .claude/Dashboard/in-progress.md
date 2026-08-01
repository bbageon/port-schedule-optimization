# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).
>
> **사후 비용계약 변경(2026-07-30)**: YR-135의 현재 라벨은 계단형 트럭비용·본선 33의
> v1 계약이다. 결과는 V/A 구조 진단으로만 보존하며 새 정책 채택 근거로 승계하지 않는다.
> 구조가 통과하면 YR-136 점증비용 v2로 신규 라벨을 생성해야 한다.

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-139 | RL | **BlockQ-v4-A — 중앙 공동후보 PPO (학습방식 단일축)** | 🔴 | 2026-08-01 | **13차 피드백**: v3 종결(분포 이동) → v4 = 자기 궤적 + 실비용 직접 최적화. Actor(공동후보 softmax·mask 선제거)+Critic(V 174차원), 보상 = −ΔΦ·**등식 테스트(Σ 구간비용 = 평가 총비용)**·γ=1·GAE λ0.95·clip 0.2(표준 앵커)·최종 정책 사용(선택 누출 제거). 60 iter × 8 ep × 3 초기화. 판정: 완주 100%∧backlog 0 ∧ ≥2/3 초기화 비용 감소 방향 ∧ 장악 0 — **신호 없으면 PPO 트랙도 중단**. YR-133·124 보류 · [spec](../docs/dashboard-task-specs/YR-139-blockq-v4-ppo.md)·[전환 기록](../docs/strategy-history/2026-08-01-BlockQ-v4-PPO-전환.md) |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
