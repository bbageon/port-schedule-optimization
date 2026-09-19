# 수직형 구현 검증 증거

2026-09-19, YR-317-h. 수직형 코드와 국소 정지 수정의 검증이며 두 구조의 성능 비교가 아니다.

- [전체 구현·한계](../../../docs/paper/v3/paper-revision/26-수직형-구현검증.md).
- 초기 코드 `f2533ee`: 경로·역할·입력·본선 마감 55검사, 기존 환경 38검사 통과.
- 초기 2일 진단은 트럭 120대와 본선 작업 5,738회를 투입했다. 트럭은 전부 완료했으나 본선은 1,269회 남아 자격 검사가 실패했다. 시간 정책은 시작하지 않았다.
- [실패 분석](initial-diagnostic-analysis.md), [원 판정](run-f2533ee/NO_REALLOC/qualification.json), [압축 원본 결과](run-f2533ee/NO_REALLOC/result.json.gz), [압축/원본 지문](run-f2533ee/evidence-index.json).
- Y02의 전체 운반 완료 하한은 최소 63.03시간으로 진단 종료 50시간을 넘는다. 이 용량 문제와 아래 후보 결함을 구분한다.
- [정체 상태 재현](deadlock-probe-f2533ee/summary.json), [후보 결함](deadlock-probe-f2533ee/candidate-diagnosis.json): 실행 가능한 후보를 전부 버려 대기만 남기는 문제를 확인했다.
- [수정 재생](deadlock-probe-f2533ee/candidate-fix-replay.json): 같은 상태·같은 배차로 1시간 동안 두 블록 모두 기존 0건→수정 10건 완료. 정책 예외·물리 제약 위반 0.
- 수정 후 [새 구조 63검사](tests-after-candidate-fix.xml)와 [기존 환경 38검사](legacy-after-candidate-fix-tests.xml), 총 101검사 통과. 학습 분기 검사 1개는 범위 밖으로 제외했다.
- [최종 기본 경로 동일성](legacy-after-candidate-fix-equivalence.json): 기존 1일·21대 결과 및 관측 289행과 정확히 일치한다.

원본 실패·이전 검사 기록은 보존한다. 국소 수정 결과를 2일 전체 완료나 정책 성능으로 바꾸어 보고하지 않는다.
합성 도착곡선·기존 독립 평가의 고정 코드·입력·모델은 바꾸지 않았고 새 학습은 없다.
후속 h 작업은 두 구조의 공통 이송·후보 조건 및 종료 시 잔여 작업의 보고 기준을 맞추는 일이다.
