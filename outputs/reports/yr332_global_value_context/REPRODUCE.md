# 재현 절차

이 실험의 학습·입력·시뮬레이터 소스는 `campaign.json`의 commit과 sources에 고정돼 있다.
예측 진단을 통과한 경우에만 PPO를 실행한다. 결과를 보고 기준·횟수를 바꾸지 않는다.
`summary.json`의 policy가 null이면 이번 증거에는 새 PPO 성능 비교가 없다.

## 환경과 코드

1. 별도 작업 디렉터리에서 `campaign.json`의 commit을 체크아웃한다.
2. 정확한 줄바꿈까지 복원하려면 `source-snapshot.zip`을 그 별도 디렉터리에 푼다.
   현재 작업 중인 저장소에 덮어쓰지 않는다. 파일별 SHA256은 source-publication.json에 있다.
3. Python과 라이브러리는 runtime-environment.json을 따른다. 실제 실행은 Windows yr331-env CPU다.
4. PowerShell에서 `PYTHONPATH=src;scripts/v7`, `PYTHONDONTWRITEBYTECODE=1`을 설정한다.
5. 기존 출력은 보존한다. 전체 재실행은 별도 복제본에서 새 출력 경로로 진행한다.
   준비 도구는 기존 bank/campaign/model 파일을 덮어쓰지 않고 실패하도록 설계됐다.

## 저장 자료로 학습 재현

```powershell
$env:PYTHONPATH='src;scripts/v7'
$env:PYTHONDONTWRITEBYTECODE='1'
python scripts/v7/global_value_experiment.py prediction-replay --workers 6
```

이 명령은 저장된 공통 경험의 표현을 사용해 비용망의 전체200회 학습을 다시 계산한다.
각 최종 가중치와 Adam 상태가 정확히 같아야 한다. 시뮬레이션 전체를 다시 돌렸다는 뜻은 아니다.
PPO가 실행된 결과에서만 `policy-replay`를 추가로 사용한다.

## 새 경험부터 실행하는 순서

- 코드·사전등록·bank를 고정하고 `global_value_inputs.py preflight`로 공급·방문 제약을 확인한다.
- `global_value_experiment.py smoke`로 기존 경로와 기록만 추가한 경로가 같은지 검사한다.
- clean tracked commit에서 `global_value_experiment.py prepare`를 실행해 소스·모델·입력을 동결한다.
- `collect` → `embed` → `predict` → `judge` 순서로 실행한다. 각각 명령 인수는 같은 스크립트의 mode다.
- prediction-summary.json의 passed가 true일 때만 `train` → `evaluate`를 실행한다.
- `common-repeat`, `prediction-replay`, `verify_global_value_wiring.py`로 기록·학습·전체 배선을 재현한다.
- PPO를 실행했다면 `policy-replay`, `evaluation-repeat`도 수행한다.
- `audit_global_value.py`는 오더 시각·대기 스냅샷·예측 오차를 저장 원자료에서 다시 계산한다.
- `report_global_value.py`와 `plot_global_value.py`는 그 판정에서 보고서와 그림을 만든다.

입력·행동·표본·학습 난수, 반복 횟수와 통과 기준은 사전등록 및 campaign.json에 있다.
학습 코드와 분석 코드는 서로 다른 커밋일 수 있다. 후자는 publication.json의 해시로 고정한다.

## 지표 해석

- 공통 경험의 예측 오차와 실제 정책 운용 결과를 구분한다. 비용 예측망만 학습하면 선택망은 바뀌지 않는다.
- 트럭 시간은 게이트 진입→작업완료, 본선 시간은 계획된 묶음 시작→개별 야드 작업완료다.
- 미완료도24시간 관측 종료까지 경과시간을 포함한다. 26시간 미완료 개수도 따로 보고한다.
- 동일 날짜의 모든 결정은 같은 분할에 속한다. 검증 결정들을 학습 자료로 되돌려 넣지 않는다.
- 새로운 합성 날짜이며 같은 실측 보정 가족이다. 현실성·현장 성능 확증과 구분한다.
