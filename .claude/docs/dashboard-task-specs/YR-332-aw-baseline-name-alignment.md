# YR-332-aw — 기존 비교 규칙 이름·설명 정합 확인

**상태**: backlog
등록: 2026-11-17, av의 적용 코드 대조 중 발견. 보정 대상: reliability(설명과 실행 코드의 일치).

## 무엇·왜

- 현재 실제 경로 `v7.dispatch.make_preference`에서 `FIFO`는 `release_time`으로 먼저 통지된 작업을 고른다.
  실제 관측된 블록 도착순은 별도 코드명 `FCFS_ARRIVAL`이다.
- `LWKR`는 `LongestWait` 클래스이며 `-sim.cum_wait(job_id)`로 트럭 최장 대기 우선이다.
  남은 작업량 우선(통상적 LWKR 의미)과 다르다. 본선은 누적 트럭 대기가0이라 후순위가 된다.
- 앞 합성 입력·학습 보고서에서 이를 대기 등록순·남은 작업량 우선으로 설명한 부분을 확인했다.
  원자료 수치·실행 알고리즘이 바뀐 것은 아니지만 연구 방법 서술을 정정해야 한다.
- 이번av 보고서는 실제 동작에 맞춰 작성한다. 본 작업은 이전 문서의 영향 범위를 확인한다.

## 범위

1. 저장 실행의 dispatcher→factory→class→정렬키를 commit별로 확인한다. 원본 world 사본의
   `FIFOPreference`는 미래정보 경고가 있지만 현재 `v7.dispatch.FIFO`와 다른 경로이므로 섞지 않는다.
2. `scripts/v7/report_synthetic_pipeline.py`와 생성된 기존 보고서·Dashboard·연구 설명에서 잘못된 정의를 찾는다.
3. 과거 사전등록·원자료·해시는 덮어쓰지 않는다. 정정 문서와 참조를 추가하고 바뀐 문서의 게이트를 갱신한다.
4. 문헌 대응이나 비교 결론에 영향이 있으면 원 주장과 영향 범위를 분리해 보고한다.

## 비목표·완료 조건

- 정책 순서·점수·기본 dispatcher를 바꾸지 않는다. 새 학습·새 규칙 평가도 이 작업에 포함하지 않는다.
- 이름을 바꾸어 과거 결과가 다른 알고리즘의 결과인 것처럼 보이게 하지 않는다.
- 명칭–실제 정렬키–적용 파일 매핑, 원 수치 불변 확인, 정정 문서·보드·근거 정합 뒤 commit/push.

증거: `src/yard_rl/v7/dispatch.py`의 `FIFO`, `LongestWait`, `RULE_BASES`, `make_preference`.
발견 시점 source commit은 `cc1a1e23`이며 av의206개 source-byte 검사에 포함된다.
