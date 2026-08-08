# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).
>
> **사후 비용계약 변경(2026-07-30)**: YR-135의 현재 라벨은 계단형 트럭비용·본선 33의
> v1 계약이다. 결과는 V/A 구조 진단으로만 보존하며 새 정책 채택 근거로 승계하지 않는다.
> 구조가 통과하면 YR-136 점증비용 v2로 신규 라벨을 생성해야 한다.

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-150 | Exp | **H-21 21블록 — ★4차 재정의: 고정 재공량(WIP) 계약 전환, 재자격 미판정** | 🔴 | 2026-08-06 | **★부하 정의 전환(사용자 결정 2026-08-08)**: L∈{50..150} = 터미널 안에 **유지하는** 외부트럭 대수(고정 WIP). 초기 채움 L 대 + 나간 만큼 60초 주기 교체 투입. 구현 = `admit_external_job`(런 중 투입 수술·fail-closed) + `build_fixed_wip`(pool 사전 추첨·결정론) + `WipAdmissionController` + `yr150_h21_wip_pilot`(자격 검사 W1~W8). smoke 실측: WIP 60 유지구간 min=max=60·건너뜀 0·불변식 OK·혼잡비 1.204. **공정성 한계 박제**: 빠른 정책일수록 트럭을 더 받아 정책별 처리 물량이 달라짐 → 판정은 **같은 재공량에서 처리량+시간당 v2 비용 공동 판정**만 허용(에피소드 총비용 단독 금지). walk-in(lead 0)이라 **PRE_GATE 창 없음** — 0B 전 lead>0 설계 선결. [결정](../docs/strategy-history/2026-08-08-YR-150-고정WIP-4차재정의-사용자결정.md) · **구 유입량 계약 이력**: 0단계(N=3 계약 9종, evidence `f37414d`)·1단계(자격 10종, `3f7275b`) 통과 보존 — [0단계](../../outputs/reports/yr150_nblock_contract/report.md) · [1단계](../../outputs/reports/yr150_h21_pilot/report.md). **★본선 물량 재보정(2026-08-08)**: 척당 15건은 2블록 시절 값의 무비판 승계 — HJNC 확인 처리량(231만 TEU/년)에서 터미널 본선 작업률 145~170 moves/h 를 유도해 12 process×120 moves(계획 161/h·앵커 안·양하 6/적하 6)로 동결, 앵커 등록부에 vessel_workload 승격. smoke: WIP 60 유지 그대로·혼잡비 1.20→1.405(본선 경쟁 실화). **고정 WIP 재자격은 미판정**(`yr150_h21_wip_pilot` 공식 실행 필요). 현실성 게이트 FAIL 지속(YR-158) · [spec](../docs/dashboard-task-specs/YR-150-continuous-inflow-steady-state.md) · **금지**: 정책 성능 비교 |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
