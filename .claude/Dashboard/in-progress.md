# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).
>
> **사후 비용계약 변경(2026-07-30)**: YR-135의 현재 라벨은 계단형 트럭비용·본선 33의
> v1 계약이다. 결과는 V/A 구조 진단으로만 보존하며 새 정책 채택 근거로 승계하지 않는다.
> 구조가 통과하면 YR-136 점증비용 v2로 신규 라벨을 생성해야 한다.

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-137 | RL | **v2 비용계약 정렬 보완 + clean 재적합 + 맞춤 대조군 (9차 피드백)** | 🔴 | 2026-07-31 | **YR-136 하향의 보완 4묶음**: ①라벨·평가 정렬 — 미래 gate-in 미열람(✓수정: observed_gate_in — PLANNED 80대 사전 기록 누출 실측)·예측 softplus/실현 hard 분리(✓함수 추가)·본선 후보-인과 조건부(paired 반사실 v2 판 — 착수 동결)·검열 일치 ②clean 재적합 — **게이트→블록 실측 161.7~258.3s ≠ 계약 180~420s: **(a) 확정(10차 피드백 권고 수용)** — 기반 기설 발견(YR-103 플래그·예약 필드), 켜서 재적합** → 미열람 시드·본선 ≥32척 재적합 ③같은 라벨로 3망 대조군(회귀/순위/V·A+보조) ④확장 판정 등록(동점 top-1·에피소드 후회 CI·총비용·완주·backlog·WAIT/REPO 재퇴화). YR-133 연기 · [spec](../docs/dashboard-task-specs/YR-137-v2-label-alignment.md) |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
