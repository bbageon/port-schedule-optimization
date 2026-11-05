# 기본 하루 평가 경로 — 속도 최적화 적용

2026-11-05 · YR-332-l. 기존 `run_episode`에 평가 대조 설정과 선택적 계측을 연결했다.

## 사용

```python
from yard_rl.v7.stage.episode import run_episode
from yard_rl.v7.evaluation_probe import EvaluationProbe

probe = EvaluationProbe()
result = run_episode(load=7500, seed=9913320, dispatcher="RL_CRANE",
                     evaluation_mode="optimized", evaluation_probe=probe)
```

- `optimized`가 기본이다. 상태별 목록·계획/슬롯 재사용·공통 정보·일괄 추론을 사용한다.
- `reference`는 개별 추론·전체 오더 탐색·계획 재사용 끔의 평가 대조다.
- `batch`는 일괄/공통 정보만 켠 중간 대조다. 이번 기본 환경 캠페인은 reference/optimized만 비교했다.
- 기본 후보 예산 12와 필수 후보 초과 보존, 신경망 입력·보상·최종 안전 검사는 양쪽 같다.
- `crane_net`을 지정하면 같은 모델을 재사용한다. 없으면 기존대로 `seed+2` 초기값을 쓴다.
- 대조 설정이나 관측자를 교사와 함께 켜면 명시적으로 거절한다. 기본 학습 경로의 계약은 유지한다.

## 측정 경계

`probe.result`는 후보 생성·선택/공동 안전 검사·전체 실행 시간과 원시 결정 시간을 제공한다.
결정 시간에는 실제 작업 적용과 관측자 자체 계산을 넣지 않는다.
전체 실행 시간에는 관측자 기록 비용이 포함되므로 기록 없는 운영 시간과 같다고 주장하지 않는다.
후보 계획·선택·사건·완료 시각·오더 장부·운영 비용의 해시로 같은 결과인지 대조한다.
관측자 실패는 호출자에게 전달하며, 작업을 조용히 쉬기로 바꾸지 않는다.

21블록의 42크레인은 시뮬레이션 사건 순서대로 처리한다. 여러 외부 요청을 동시에 받는 서버가 아니다.
프로세스 최대 메모리는 파이썬/PyTorch·상태·캐시·계측 기록을 모두 포함한다.
계측은 상태를 읽기만 하며 네트워크 업데이트와 비교용 미래 시뮬레이션을 실행하지 않는다.

## 근거

- [사전등록](default-eval-prereg.md)
- `scripts/v7/benchmark_default_eval.py`: 새 프로세스에서 기본 하루 실행을 9쌍 비교.
- `scripts/v7/summarize_default_eval.py`: 원자료 대조·압축 보존·보고서 생성.
- `tests/v7/test_evaluation_runtime.py`: 실제 실행 정책 대조, 관측 실패 전파, 교사 계약 분리.
- 결과: `outputs/reports/yr332_default_eval/report.md`.
