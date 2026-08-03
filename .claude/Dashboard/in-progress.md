# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).
>
> **사후 비용계약 변경(2026-07-30)**: YR-135의 현재 라벨은 계단형 트럭비용·본선 33의
> v1 계약이다. 결과는 V/A 구조 진단으로만 보존하며 새 정책 채택 근거로 승계하지 않는다.
> 구조가 통과하면 YR-136 점증비용 v2로 신규 라벨을 생성해야 한다.

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-147 | RL | WAIT-WAIT 정책 최적화 — 1단계 기준 측정 (하드 마스크 없음·유한 DEFER 는 2단계) | P0 | 2026-08-03 | [spec](../docs/dashboard-task-specs/YR-147-wait-wait-policy-optimization.md) · **1단계 기준선 완료 (★21차 정정 반영)**: 공동결정 중 진행 가능 전원대기 발생률 17.4/7.5/2.7%(합산 9.53%)·엄격 D_wait<0 33/53/13%·표본 비대표(high-tight 0)·깊은 정지 8건=2에피소드(해결 주장은 가설) — 진단 단계이며 성능 개선 아님. 다음 = 2단계 **A/B/C 3군 분해**(무기한 WAIT/전량 유한 DEFER/trigger 조건부 발행 — B−A·C−B 효과 분리) 사전등록 |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
