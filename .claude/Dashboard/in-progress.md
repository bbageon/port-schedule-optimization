# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).
>
> **사후 비용계약 변경(2026-07-30)**: YR-135의 현재 라벨은 계단형 트럭비용·본선 33의
> v1 계약이다. 결과는 V/A 구조 진단으로만 보존하며 새 정책 채택 근거로 승계하지 않는다.
> 구조가 통과하면 YR-136 점증비용 v2로 신규 라벨을 생성해야 한다.

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-150 | Exp | **21블록 터미널 전체 지속 유입 자격시험 — 0단계 통과, 다음 1단계 파일럿** | 🔴 | 2026-08-06 | [spec](../docs/dashboard-task-specs/YR-150-continuous-inflow-steady-state.md) · 게이트 인가 `allowed=true`(REMEDIATION·target=`scenario_validity`, gate `cd6f373`). **0단계 완료(`5be10ad`)**: N=3 계약 9종 전 항 통과 — 목적지별 주행 matrix **비영**(110·220초, 구판은 180초 상수)·최소(부담+주행) 목적지 선택 69/69·**동시 확정 2 epoch**(터미널 1건 상한 아님)·소스당 1건·용량 fail-closed·rollback 복원·결정론·게이트 진입 전 재배정 주행 차이 **±110/±220초(0 아님)**. 앵커 보존(평균 300초·범위 190~410초). 골든 보존(전체 회귀 742 통과, 1 실패는 기존 YR-111 동일 수치) · [report](../../outputs/reports/yr150_nblock_contract/report.md). **0단계는 현실성 게이트를 닫지 않는다** — 다음은 **1단계 21블록 자격 파일럿**(터미널 master stream 을 배분벡터 `p` 로 21블록 배분·관측시간 종료·5~10분 스냅샷·상태 사후분류). **금지**: 성능 비교·정책 군 비교 |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
