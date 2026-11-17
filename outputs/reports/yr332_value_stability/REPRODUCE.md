# 검산과 재현 범위

- 학습·평가 핵심 소스는 `campaign.json`의 commit과 215개 파일 해시로 고정했다.
- 실제 패키지·실행 환경은 `execution-environment.json`, 원격 소스 대조는 `source-publication.json`에 있다.
- 모든 비교군의 입력과 최종 모델 선택은 결과를 보기 전에 고정했다.
- 모델당 12개 합성 날짜를 썼다. 6개 모델이 같은 날짜를 공유하므로 72개의 독립 날짜가 아니다.

## 원자료 검산

아래 명령은 저장된 오더 시각·표적·파일 해시를 검산한다. 연구 환경의 Python 경로를 사용한다.

```powershell
$env:PYTHONPATH='src;scripts/v7'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
& '..\yr331-env\Scripts\python.exe' scripts/v7/verify_value_stability.py
& '..\yr331-env\Scripts\python.exe' -m pytest tests/v7/test_on_policy.py tests/test_value_normalization.py -q
```

**새 checkout의 줄바꿈 차이**: 구형 `outputs/reports/yr332_day_shapes/calibration-periodic_linear_24.json`
한 개는 실행 당시 CRLF, Git에는 LF로 저장되어 있었다. JSON 내용은 정확히 같음을 확인했다.
해시 검산에서 이 파일만 다르면 **별도 검산 작업사본에서** 다음 원래 바이트 사본을 복원한다.
기존 연구 파일이나 다른 사람의 작업을 덮어쓰지 않는다.

```powershell
Copy-Item -LiteralPath 'outputs/reports/yr332_value_stability/frozen-sources/calibration-periodic_linear_24.json' -Destination 'outputs/reports/yr332_day_shapes/calibration-periodic_linear_24.json'
```

## 학습 재실행의 의미

- `initial.pt`, `update-0.pt`~`update-5.pt`, `final.pt`에 신경망·Adam·정규화 통계·Torch 난수를 저장했다.
- 이번 검산은 저장된 경험을 읽어 매 갱신 직전 상태에서 다시 학습하고, 다음 체크포인트의
  신경망과 Adam 상태가 정확히 같아지는지 확인한다. 총 36갱신이다.
- 이 검산은 학습용 터미널 시뮬레이션 전체를 다시 돌린 것과 다르다.
- 별도 작업사본에서 `value_stability.py replay --workers 6`으로 재실행할 수 있다.
  `replay.json`의 실행 시간은 다시 기록되므로 원본 발행 해시는 그 전에 검산한다.
- 실제 터미널이나 다른 하드웨어의 재현까지 입증하지 않았다. 평가의 결정적 재실행은 두 경우다.
- 새 학습의 `train` 명령은 기존 초기 체크포인트가 있으면 중단한다. 새 실험은 출력과 등록을 분리한다.

음수 행동 신호는 두 군 모두 유지한다. 정규화 통계는 학습 표적에만 갱신하며 검증·시험 자료는 넣지 않는다.
