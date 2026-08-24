"""RUDDER — **어느 시각대**가 비용 차이를 만들었나 ([[YR-223]]).

■ 이 꾸러미는 v3 를 **재는 도구**다 — v3 의 경쟁자가 아니다
  v3 가 v1·v2 를 통째로 복제한 이유는 **비교 대상**이었기 때문이다. 남의 코드를
  빌려 쓰면 그쪽이 바뀔 때 이쪽이 조용히 같이 바뀌어 비교가 오염된다.

  RUDDER 는 반대다. **v3 정책이 실제로 만든 궤적**을 재야 한다. 복제하면 사본을
  재게 되고, v3 가 바뀌어도 옛 v3 를 계속 재는 셈이라 **측정이 틀린다.**
  그래서 v3 를 **읽기 전용 재료로 가져다 쓰되, 한 줄도 고치지 않는다.**

  ⚠️ 이 규칙은 시험이 지킨다 — `tests/rudder/test_no_v3_mutation.py`.

■ 조립은 v3 것을 그대로 쓴다
  창을 굴리는 러너는 새로 쓰지만, 시장·다리·실행정책은 v3 의 `_Ctx` 가 만든다.
  다르게 조립하면 **v3 아닌 것의 궤적**을 재게 된다. 동일성은 시험이 검사한다 —
  `tests/rudder/test_runner_identity.py`.

■ 무엇을 만드나
      tape       1분 epoch 토큰 (무엇을 보는가)
      runner     창 하나를 굴리며 토큰을 남긴다 (정책 / 안팔기 짝)
      collect    창을 여러 개 · 프로세스로 나눠 모은다
      model      소형 LSTM (causal)
      train      held-out 20% 학습
      contrib    기여도 = 연속 예측값의 차이
      gates      자격시험 A~F

■ 무엇을 하지 않나
  **정책 학습에 기여도를 넣지 않는다.** 이 꾸러미는 자격시험까지다. 통과한 뒤
  표본 추출기로 쓰는 것이 [[YR-223]] 4단계, 정책 연결은 [[YR-224]] 다.
"""
from .runner import Window, collect_day, intervene, run_window, snapshot_at
from .tape import ACTION_FEATURES, ACTION_IDX, DIM, FEATURES, EpochTape, Token

__all__ = ["FEATURES", "ACTION_FEATURES", "ACTION_IDX", "DIM", "Token", "EpochTape",
           "Window", "collect_day", "run_window", "intervene", "snapshot_at"]
