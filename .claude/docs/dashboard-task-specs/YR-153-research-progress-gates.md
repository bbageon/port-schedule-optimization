# YR-153 — 연구 진행 3대 게이트 하네스

**상태**: **done (2026-08-05 — 구현·적대감사 통과, commit `88352ef`)**

## 사용자 요구

다음 단계 전 세 가지를 우선 확인한다.

1. 규칙 기준선보다 성능이 증가했는가?
2. Dashboard의 주장과 실행 코드·원자료가 일치하는가?
3. 시드 데이터가 항만에서 가능한 물리·정보·운송흐름인가?

미통과가 있으면 그 원인을 직접 해결하는 작업만 허용한다. 세 항목이 모두 통과하면 관련 없는
새 가설을 만들지 않고 연구 목표의 확증·잠금평가까지 빠르게 진행한다.

## 구현 범위

- 공통 `PASS / FAIL / INCONCLUSIVE` 자료형
- 원정밀도 신뢰구간·검정력·하드 가드를 쓰는 성능 게이트
- 실행 commit·dirty·절대 시드·설정·artifact hash 검사
- Dashboard ID 유일성·spec 상태·증거 경로·원격 commit 및 **각 증거 파일↔commit 연결** 감사 CLI
- 물리·정보·장부·지속유입 안정성 + 관측/시뮬레이션 수치범위·근거파일 hash 현실성 게이트
- 세 게이트 상태에 따른 다음 작업 허용기(`OBJECTIVE / REMEDIATION / NEW_HYPOTHESIS`)
- 저장 판정 JSON의 hash·원격 commit을 검증하고 `PASS` 근거를 재계산한 뒤 미통과 한 축만 허용하는
  `authorize-next` CLI(종료코드 2면 착수 금지)
- negative result를 먼저 저장하고 다음 주장만 막는 결과 JSON 계약

## 판정 계약

- 성능 `PASS`: 후보가 규칙보다 사전 최소개선량 이상 좋다는 신뢰구간 + 보호 가드 통과
- 신뢰성 `PASS`: 실행 스탬프와 commit 뒤 Dashboard 감사 모두 통과
- 현실성 `PASS`: 요청한 주장 범위의 내부·흐름·수치 앵커·근거파일 검사가 통과. 특정 터미널
  실성능 주장은 실제 운영 이력 파일까지 있어야 통과
- 하나라도 미통과면 `REMEDIATION_ONLY`
- 전부 통과하면 `ADVANCE_TO_OBJECTIVE`

## 현재 연구 상태를 이 계약으로 다시 읽기

- 성능: **INCONCLUSIVE** — 지속 유입 환경에서 현재 배포 정책의 SF-SPT 대비 터미널 총비용 잠금평가가 없음.
- 신뢰성: **FAIL** — YR-149의 완료 링크는 맞지만 실행 스탬프가 `git_dirty=true`이고 전체 설정이 비어 있음. 공용 판정의 반올림 결함도 이 작업에서 수정.
- 현실성: **INCONCLUSIVE** — 문헌 보정 유한 물량 시드는 내부적으로 가능하지만, 지속 유입 정상상태와 실제 운영 이력은 아직 미검증.

따라서 현재 허용 순서는 YR-151 0A(신뢰성) → YR-150(현실성) → YR-151 0B(성능)뿐이다.

## 완료 조건

- 새 하네스 계약 테스트 통과
- 기존 evalkit 경계 판정 원정밀도 회귀 테스트 통과
- 전역 규칙·skill·결정이력 반영
- 이 row의 Dashboard↔spec↔evidence 감사 통과
- 범위가 명확한 commit과 origin push

## Evidence

- [현재 3대 게이트 스냅샷](../../../outputs/reports/yr153_research_gates/current_gate.json)
- [쉬운 설명 보고서](../../../outputs/reports/yr153_research_gates/report.md)
- 구현: `src/yard_rl/experiments/gate_harness.py`
- 테스트: `tests/integrated/test_yr153_research_gate.py`
- 구현·원자료 commit: `88352ef` (origin/master push 확인)
- 검증: targeted 48 passed, 독립 적대감사 critical/major 0
- gate 파일 sha256: `9935624b5f71951af03fc42667c7df05d8a07c3d46ec7288f5ea9bb354351620`
