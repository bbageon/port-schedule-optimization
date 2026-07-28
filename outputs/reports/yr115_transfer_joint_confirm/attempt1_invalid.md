# YR-115 v1 blind pilot 무효 기록

- 실행 HEAD: `ac6cfed`
- 표본: 신규 16쌍, 평균·CI·raw row는 저장·출력하지 않음
- 공개된 값: 계획 표본 수 `n=238`
- 판정: **무효 — 확증 진입 금지**

독립 사전결과 감사가 파일럿 직후 두 계약 누락을 확인했다.

1. 실제 import되는 일부 experiment/env/policy 모듈이 source digest 밖이었다.
2. 미출문 차량도 `평가종료-A` 검열값으로 A→O 표본에 포함돼 기존 count guard가 누락을
   탐지하지 못했다.

`power_note.json`은 실패 경위를 재현하기 위한 evidence로만 보존한다. 그 안의 표본 계획은
사용하지 않으며 v1의 32개 실현은 v2 pilot·confirm에서 모두 제외한다.
