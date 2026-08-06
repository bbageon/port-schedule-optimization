# 🟢 In Progress

> 현재 진행 중. **한 번에 1개** 권장. [index](README.md) · 인접: [ready](ready.md) → 여기 → [done](done.md).
>
> **사후 비용계약 변경(2026-07-30)**: YR-135의 현재 라벨은 계단형 트럭비용·본선 33의
> v1 계약이다. 결과는 V/A 구조 진단으로만 보존하며 새 정책 채택 근거로 승계하지 않는다.
> 구조가 통과하면 YR-136 점증비용 v2로 신규 라벨을 생성해야 한다.

| ID | Epic | Title | Priority | 착수 | Note |
|---|---|---|---|---|---|
| YR-150 | Exp | **H-21 수평 공유형 21블록 지속 유입 자격 — 0단계 통과, 다음 1단계** | 🔴 | 2026-08-06 | [spec](../docs/dashboard-task-specs/YR-150-continuous-inflow-steady-state.md) · 게이트 인가 `allowed=true`(REMEDIATION·target=`scenario_validity`, gate `cd6f373`). **0단계 실행 기준 `5be10ad` → 판정·evidence `f37414d`**: N=3 계약 9종 전 항 통과 — 목적지별 주행 matrix 비영(110·220초)·최소(부담+주행) 목적지 69/69·동시 확정 2 epoch(**강제 발화 계약검사, 자연 성능 아님**)·소스당 1건·용량 fail-closed·rollback·결정론·PRE_GATE 주행 차이 ±110/±220초. 판정 당시 전체 회귀 742 통과/기존 실패 1건, **현재 HEAD 직접 재검증 753 통과·4 skip/동일 YR-111 실패 1건** · [report](../../outputs/reports/yr150_nblock_contract/report.md). 다음은 **H-21 21블록 자격 파일럿**(terminal master stream·관측시간 종료·스냅샷·사후 상태분류). 현재 route는 1차원 합성이고 실제 수평 터미널 재현이 아니다. V-21 역할분리형은 YR-083 후 별도 자격·성능으로 분리한다. **금지**: 이 단계의 성능·정책 비교 |

---

운영: 시작 시 [ready.md](ready.md) 에서 pull. 종료(commit) 시 [done.md](done.md) 로 이동 + commit 링크 박제.
