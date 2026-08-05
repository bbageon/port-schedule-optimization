# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).
>
> **사후 비용계약 변경(2026-07-30)**: YR-135의 현재 라벨은 계단형 트럭비용·본선 33의
> v1 계약이다. 결과는 V/A 구조 진단으로만 보존하며 새 정책 채택 근거로 승계하지 않는다.
> 구조가 통과하면 YR-136 점증비용 v2로 신규 라벨을 생성해야 한다.

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-149 | RL | 견적·평가 정렬 → 파급 원인 진단 → 이송 효과 확증 (1단계 착수) | P0 | 2026-08-05 | [spec](../docs/dashboard-task-specs/YR-149-quote-refine-confirm.md) · 범위 정정(2026-08-05) 반영: TOS 연동 제외·0.107 = 기능 보정값. 선결 정렬 구현 + 1단계 동결 `cad0e9a` → **1단계 완료**: 리플레이 20런 재현 성립, 잔차 소스 +0.54/수신 −0.51/**본선 +0.66 = 지배 축**(동결 규칙) — 수신 과소 가설 기각·소스 과대 경향 병기. **2단계 = 본선 항 배선**(vessel_cost.compare_completion_cost — 설계돼 있던 미배선 API) 설계·동결 대기 · 1단계 판정 `ed50db0` |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
