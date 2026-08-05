# Skill: 연구 진행 3대 게이트

> 트리거: 판정 실험 결과를 저장할 때 / Dashboard 완료 처리 뒤 / 다음 연구 작업을 착수할 때.
> 구현: `yard_rl.experiments.gate_harness`.

## 목적

다음 세 질문에 답하지 않은 채 새 가설로 이동하는 일을 막는다.

1. **성능** — 현재 후보가 규칙 기준선보다 터미널 비용을 줄였는가?
2. **신뢰성** — Dashboard, 실행 코드, 시드·설정, 원자료와 보고 수치가 일치하는가?
3. **현실성** — 시드가 물리·정보시점·운송흐름을 지키며, 주장 범위에 맞는 자료 근거가 있는가?

## 1. 실험 실행 직후

1. 원자료와 결과 JSON을 **먼저 저장**한다. 실패·미확정도 지우지 않는다.
2. `repro_stamp()`로 commit, dirty 여부, 절대 시드, 전체 설정, 사전등록을 기록한다.
3. 결과·체크포인트·설정 파일의 sha256을 기록한다.
4. `judge_performance()`는 **terminal total cost 후보−SF-SPT**의 반올림 전 신뢰구간을 사용한다.
5. 완주·backlog=0·본선 보호·물리 유효성 네 가드는 하나라도 미수집이면 실패한다.
6. `judge_scenario_validity()`로 내부타당성과 지속운영 타당성을 나눠 검사한다.
7. `judge_claim_alignment()`로 보고서·Dashboard 핵심 수치를 결과 JSON 원값과 대조한다.
8. `attach_common_gates()`로 결과 JSON에 판정을 붙인다.

성능은 세 값만 허용한다.

- `PASS`: 신뢰구간 전체가 사전 최소개선량을 넘고 가드도 통과.
- `FAIL`: 충분한 표본에서 최소개선 미달 또는 하드 가드 위반.
- `INCONCLUSIVE`: 표본 부족이나 신뢰구간 교차로 아직 판단 불가.

## 2. commit·Dashboard 갱신 뒤

다음 명령으로 row·spec·증거 경로·commit을 대조한다.

```powershell
$env:PYTHONPATH='src'
python -m yard_rl.experiments.gate_harness audit-dashboard `
  --task YR-NNN --state done `
  --spec .claude/docs/dashboard-task-specs/YR-NNN-name.md `
  --evidence-path outputs/reports/.../result.json `
  --evidence-commit <commit> --remote-ref origin/master
```

검사는 다음을 fail-closed로 본다.

- ID가 다섯 상태 파일 중 정확히 한 곳에 존재
- board 상태와 spec의 `**상태**` 일치
- 코드·보고서·원자료 경로 존재
- evidence commit 존재 및 원격 branch 반영
- 실행 시점 코드가 실제 존재하는 clean commit이며 절대 시드·빈 값이 아닌 전체 설정이 완전
- 사전등록 파일과 artifact가 존재하며 계산한 sha256이 기록값과 일치

## 3. 현실성 판정

내부타당성 필수 항목:

- 사건 순서 `A≤B≤S≤C≤O`, 미래정보 누출 없음
- 크레인 안전·비통과·적재·용량 제약
- 작업 소유권·시간·비용 장부 보존
- 달성 가능한 본선 마감, 결정론 재현

지속운영 성능시험 추가 항목:

- 평가창 끝까지 외생 도착 지속
- warm-up 제외 후 고정 측정창 사용
- 실현 상태를 CLEAR/BUSY/OVERLOADED로 분류하고, 도착률·완료율과 재공량·backlog 기울기가
  그 분류와 일치하는지 검사한다(의도된 과부하를 정상상태 실패로 오인하지 않음).
- 부하는 입력 수치가 아니라 **실현된 상태**로 분류

외부 보정 앵커는 게이트→블록 시간, 초기 장치율, 트럭 도착률, 크레인 서비스시간, 본선 작업량을
모두 요구한다. 각 항목은 단순 `true`나 URL 문자열이 아니라 **관측 범위·시뮬레이션 범위·로컬
근거 파일·그 파일의 sha256**을 함께 기록하고, 시뮬레이션 범위가 관측 범위 안인지 기계 검사한다.
근거 파일은 `yard_rl.external_anchor.v1` JSON이며 지표명·단위·관측범위·출처 제목·위치를 담는다.
특정 터미널 실성능을 주장할 때는 `yard_rl.operational_trace.v1` 형식의 익명 운영 이력 30건 이상과
sha256이 필요하다. 각 행은 작업 ID·유형·블록과 gate-in→block-in→완료→gate-out 사건시각을 담는다.

공개자료·문헌 보정만 있으면 허용 주장은 `문헌 보정 시뮬레이션 방법론`까지다. 실제 운영
이력이 없으면 특정 터미널 실성능 주장은 자동 차단한다.

## 4. 다음 작업 허용 규칙

Dashboard row를 `in-progress`로 옮기기 전에 저장된 현재 판정으로 착수 허가를 받는다.

```powershell
$env:PYTHONPATH='src'
python -m yard_rl.experiments.gate_harness authorize-next `
  --gate-file outputs/reports/yr153_research_gates/current_gate.json `
  --gate-sha256 <sha256> --gate-commit <commit> --remote-ref origin/master `
  --task YR-NNN --spec .claude/docs/dashboard-task-specs/YR-NNN-name.md `
  --kind REMEDIATION --target reliability
```

하네스는 gate JSON 자체가 해당 원격 commit에 들어 있고 현재 hash가 같은지 확인한 뒤, 저장된 `PASS`의
원자료 경로를 다시 판정한다. 명령이 종료코드 2를 내면 착수하지 않는다. 보정은 한 번에 미통과 게이트 **정확히 하나**만 지정하며,
순서는 **신뢰성 → 현실성 → 성능**이다. 앞 게이트가 PASS가 아니면 뒤 단계는 열리지 않는다.

- 세 게이트 모두 `PASS`: 연구 목표의 확증·잠금평가로 바로 진행한다. 관련 없는 새 가설 금지.
- 하나라도 `FAIL/INCONCLUSIVE`: 그 게이트를 직접 해결하는 `REMEDIATION`만 허용한다.
- 성능 `INCONCLUSIVE`: 사전 고정 표본 확증을 한 번 수행한다. 결과를 보고 표본을 추가하지 않는다.
- 시나리오 `FAIL`: 정책을 바꾸지 않고 환경·장부·정보 계약만 고친다.
- 충분한 표본의 성능 `FAIL`이어도 특정 손실 기제가 독립 시드에서 재현되지 않으면 임의 가설을
  만들지 않고 해당 범위를 닫는다.
- 새 가설은 관측된 실패 기제를 직접 겨냥하고, 한 번에 한 축만 바꿀 때만 등록한다.
