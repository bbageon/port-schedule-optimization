# 전역 대기 정보 정책 비교 재현

실행 설정은 `campaign.json`과 `docs/research/v7-discharge/global-policy-prereg.md`에 고정했다.
저장된 원파일을 보존한다. 아래 학습·평가 명령은 같은 경로에 기존 파일이 있으면 실패하도록 만들었다.
전체 재실행은 별도 checkout에서 결과 폴더의 생성 산출물만 별도로 보관한 후 수행한다.
원래 원자료·보정값·과거 체크포인트는 수정하지 않는다.

## 환경

- Python 3.13.9, PyTorch 2.14.0+cpu. Python은 campaign, 라이브러리는 runtime-environment에 기록한다.
- Windows의 `../yr331-env/Scripts/python.exe`를 사용했다. CUDA는 사용하지 않았다.
- 한 번에 최대 6개 실행, 실행별 PyTorch/OMP/MKL 스레드 1개다.
- 그림만 기존 Anaconda Python의 Matplotlib 3.10.6으로 생성했다. 학습 환경에는 패키지를 추가하지 않았다.

```powershell
$env:PYTHONPATH='src;scripts/v7'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
```

아래 `python`은 위 환경의 Python 실행 파일을 뜻한다.

## 새 실행 순서

```text
python scripts/v7/global_policy_inputs.py bank
python scripts/v7/global_policy_inputs.py preflight
python scripts/v7/global_policy_train.py smoke
python scripts/v7/global_policy_plan.py
python scripts/v7/global_policy_train.py train --workers 6
python scripts/v7/global_policy_train.py replay --workers 6
python scripts/v7/global_policy_eval.py evaluate --workers 6
python scripts/v7/global_policy_eval.py repeat --workers 6
```

`plan`은 추적 파일이 수정되지 않은 커밋에서만 실행한다. 원 실행은 입력·연결 검사 결과도 먼저 커밋했다.
bank의 19개 곡선 중 학습 12개·개발 3개·최종 3개를 사용하며 남은 학습 1개는 연결 검사 전용이다.
최종 시험 입력은 6개 모델의 학습이 모두 끝난 후 평가한다. 중간 모델은 개발 결과만 기록한다.
예측 오차가 커져도 정해진 학습량·평가량을 변경하지 않는다.

## 저장 결과 검산

```text
python scripts/v7/audit_global_policy.py
python scripts/v7/analyze_global_policy.py
python scripts/v7/report_global_policy.py
python scripts/v7/plot_global_policy.py
python scripts/v7/verify_global_policy.py
```

위 그림 명령의 `python`은 Matplotlib이 있는 별도 실행 파일을 사용한다.
원 실행 경로는 `C:/Users/geonu/anaconda3/python.exe`다. 그림 코드는 저장 JSON만 읽는다.
`finish_global_policy.py --plot-python <그림용 Python>`은 재현→평가→2재실행→검산→집계→보고→그림의 같은 순서를 묶는다.

`replay`는 저장 경험에서 가중치·Adam 상태·진단값이 정확히 재현되는지 확인한다.
전체 학습 환경 72일을 다시 돌린다는 뜻은 아니다. 평가 환경 재실행은 사전에 고정한 2개다.
`audit`는 모든 학습·평가 장부, 전역 관측 시각, 표적 비용, 초기 모델, 최종 모델, 입력·코호트를 대조한다.
`verify`는 시각 원파일에서 평균과 미완료를 별도로 계산하고 보고서 수치와 결과 JSON을 대조한다.
반올림은 표시할 때만 한다. 모든 결과의 기준/변경은 `zero`→`global`이다.

원 실행에서는 보고서의 미완료 건수 표를 출력할 때 필드 이름의 문자열 치환 오류가 났다.
`post-training-execution.json`에 그 실패를 보존했다. 학습·평가·검산·집계는 이미 성공한 상태였다.
출력 코드만 수정해 보고서·그림을 다시 생성했으며 `post-processing-recovery.json`에 파일 해시와 검토를 남겼다.
실험·학습 예산·평가 기준은 바꾸지 않았다. `verify`는 실패 기록과 이 복구의 연결도 확인한다.

## 코드·게시 근거

`source-snapshot.zip`은 실행 당시 실제 파일 바이트다. 기존 CRLF 차이는 source-publication에 구분한다.
고정된 코드·입력 해시는 `global_policy_plan.check()`가 매번 확인한다.
최종 산출물·Dashboard·명세를 커밋하고 origin에 올린 뒤 아래 명령으로 공통 판정을 발행한다.

```text
python scripts/v7/publish_global_policy.py source
python scripts/v7/publish_global_policy.py publish --commit <원격에 올라간 결과 커밋>
```

`source`는 산출물 게시 커밋 전에 한 번만 실행한다. `publish`는 그 이후 실행하며 판정 파일만 별도 커밋한다.
원본 시각 확인 600개 파일의 불변 확인은 별도 verification의 source 보존 기록을 함께 참고한다.
기본 정책은 바꾸지 않으며 운반·대기 정의·고물량 실측 확인을 대신하지 않는다.
