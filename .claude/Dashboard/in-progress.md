# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).
>
> **사후 비용계약 변경(2026-07-30)**: YR-135의 현재 라벨은 계단형 트럭비용·본선 33의
> v1 계약이다. 결과는 V/A 구조 진단으로만 보존하며 새 정책 채택 근거로 승계하지 않는다.
> 구조가 통과하면 YR-136 점증비용 v2로 신규 라벨을 생성해야 한다.

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-141 | RL | **v4-B — 구속적 PREPOSITION 단일축 (확장 판정 7항)** | 🔴 | 2026-08-02 | **15차 확장 기준으로 동결**: 목표 = 재배치 감소만이 아니라 **트럭 이득 보존 + 본선·이동 손실 방지**. 유일 변경 = BOUND_REPO(결속 PREPO:<jid>:<bay>·근접 소멸·만료 내재·탈출 분리 — opt-in, 회귀 39 테스트 통과). 비교군 SF/v4-A(재사용)/v4-B(신규 학습), 미열람 3200 대역+실현 지문. 판정 J1 완주 ∧ J2 장악0 ∧ J3 vs SF ≥2/3 ∧ J4 B−A v2 ≤0 ∧ J5 본선 비열등 ∧ J6 v1 비열등 ∧ J7 반복·만료 이동 0. 성공 → 잠금평가 · [spec](../docs/dashboard-task-specs/YR-141-bound-preposition.md)·[하네스](../../src/yard_rl/experiments/yr141_bound_prepo.py) |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
