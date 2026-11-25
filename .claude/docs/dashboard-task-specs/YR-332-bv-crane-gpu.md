# YR-332-bv · 신경망 학습·추론 장치 선택

- **상태**: backlog
- 사용자 요청 2026-11-25. 현재 BP 모델의 신경망 구조를 확인하고 GPU(그래픽 연산 장치)로 실행 시간을 줄인다.
- BP 세 번째 갱신 모델 및 바로 다음 물결의 실제 결정 1,024개로 CPU 1스레드, CPU 16스레드, RTX 5090을 비교한다. 같은 64건씩 4반복, 세 번의 중앙값이다. 추론은 시뮬레이터가 한 번에 보내는 한 상태의 점수와 선택을 재현한다.
- 수치 검사 항목은 마스크가 있는 선택의 확률·기울기, 점수 보존, 체크포인트 재개, 고정 선택과 반복 학습 차이다. 완료된 BP 기록과 예정시각 실험의 최초 등록 기록을 보존한다.
- 장치 선택은 `scripts/v7/short_window_train.py`의 `--train-device cpu|cuda`, `--inference-device cpu|cuda`이다. CUDA를 요청한 경로는 GPU를 쓸 수 있는 PyTorch가 필요하다.
- [측정 코드](../../../scripts/v7/benchmark_short_window_device.py) · [확률 및 기울기 검증](../../../tests/v7/test_crane_devices.py)
