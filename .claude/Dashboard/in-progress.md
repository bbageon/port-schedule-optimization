# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).
>
> **사후 비용계약 변경(2026-07-30)**: YR-135의 현재 라벨은 계단형 트럭비용·본선 33의
> v1 계약이다. 결과는 V/A 구조 진단으로만 보존하며 새 정책 채택 근거로 승계하지 않는다.
> 구조가 통과하면 YR-136 점증비용 v2로 신규 라벨을 생성해야 한다.

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-140 | RL | **v4-A-fix — PPO 단위 계약 정정 + 신규 시드 재실행** | 🔴 | 2026-08-02 | **14차 피드백**: GAE 단위 혼합(원 단위 r + 1/20 V) 확정 — 실험 성립 조건 수리(튜닝 아님)가 조항 A/B 보다 선행. 수정 = 단위 통일 하나, 단위 테스트 2건(가치 완전 예측 → advantage 0 / 2행동 학습 방향) + 등식 3 = 5 통과. 평가 = 신규 미열람 BASE+2900..2902, 판정 동일. 분기: REPO 장악 재현 시에만 v4-B(구속적 PREPOSITION) 1회 → 실패 시 트랙 중단 확정 · [spec](../docs/dashboard-task-specs/YR-140-ppo-unit-fix.md)·[하네스](../../src/yard_rl/experiments/yr139_blockq_v4_ppo.py) |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
