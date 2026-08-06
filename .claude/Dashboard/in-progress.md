# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).
>
> **사후 비용계약 변경(2026-07-30)**: YR-135의 현재 라벨은 계단형 트럭비용·본선 33의
> v1 계약이다. 결과는 V/A 구조 진단으로만 보존하며 새 정책 채택 근거로 승계하지 않는다.
> 구조가 통과하면 YR-136 점증비용 v2로 신규 라벨을 생성해야 한다.

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-150 | Exp | **H-21 21블록 지속 유입 자격 — 0·1단계 통과, 현실성 게이트는 FAIL** | 🔴 | 2026-08-06 | **1단계 + 감사 정정 5건 반영(`3f7275b`)**: 자격 **10종** 전 항 통과 — master stream 정확 분할·관측시간까지 유입·5분 스냅샷·본선 분산·장부 보존·상태 사후분류·**블록별 스냅샷**·**프로파일 H-21 YT**·**내부타당성 6종**·정책 예외 0 · [report](../../outputs/reports/yr150_h21_pilot/report.md). **감사 정정**: ①H-21 이 **AGV** 프로파일로 돌고 있었다 → 중립 `build_h21_profile()`(YT, 역학 수치 동일) ②"4시간 도착 수 = L 정확히 일치"는 **철회** — L 은 명목 **예약** 도착강도이고 실제 gate-in 은 예약오차로 다르다(명목 76 vs 실제 74) ③"모든 블록이 늘 한가"는 합계만으로 미증명이었다 → 블록별 스냅샷으로 **실측**: 어느 순간에도 21블록 중 **10~16개가 완전히 빔**, 블록 최대 WIP 2~3대 ④상태 **CLEAR/BUSY/OVERLOADED** 3구간화(BUSY 임계 = 자유흐름 A→O 778.2초 × 1.5, 계약값 유도라 관측이 임계를 못 움직임 — 실측 혼잡비 1.03~1.09) ⑤부하 효과는 **인과 비교가 아니다**(부하마다 시드 상이·각 1시드·균등 배분만·본선이 6블록에만). **★현실성 게이트 실제 제출 = FAIL**: `gate_to_block_time` 시뮬 190~410초가 문헌 120~300초 **밖**, 나머지 4종 **미수집**(근거를 지어내지 않음) → [YR-158](../docs/dashboard-task-specs/YR-158-external-anchor-sourcing.md). 부하 확장은 [YR-157](../docs/dashboard-task-specs/YR-157-h21-load-band-extension.md). **0단계 기록(유지)**: 실행 `5be10ad`→evidence `f37414d`, N=3 계약 9종 통과·목적지별 주행 110/220초·동시 확정 2 epoch(강제 발화 계약검사) · [report](../../outputs/reports/yr150_nblock_contract/report.md). 전체 회귀 **770 통과·4 skip / 기존 YR-111 실패 1건** · [spec](../docs/dashboard-task-specs/YR-150-continuous-inflow-steady-state.md) · **금지**: 정책 성능 비교 |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
