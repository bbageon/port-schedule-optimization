"""v3 가 굴리는 세계 — **v3 소유의 사본**이다.

사용자 지시 2026-08-22: *"기존 코드를 임포트하지말고 v3 폴더에 새로운 코드를
작성해야해."* 그 전 지시(2026-08-20)도 같다 — *"v1 은 v1 만, v2 는 v2 만,
v3 는 v3 만 있어야해."*

■ 무엇이 들어 있나 (62 파일 · 10,666줄)
      integrated/   엔진 조정자·무대 생성기·비용곡선·배치·배정기
      sim/          이산사건 엔진 코어·적재·이동시간·KPI
      domain/       모델·열거형·검증
      contract/     상태·전이·직렬화 계약
      io/           프로파일·시나리오 적재

  원본은 `src/yard_rl/{integrated,sim,domain,contract,io}` 이고, 그 트리의 임포트가
  **전부 상대경로**라 구조만 그대로 옮기면 한 줄도 안 고치고 돈다. 손댈 데가 없으니
  복제 과정에서 오류가 끼어들 자리도 없다.

■ v3 전용 확장 — 원본과 **여기만** 다르다
  | 파일 | 무엇 | 왜 v3 것인가 |
  |---|---|---|
  | `integrated/yard_layout.py` | `quay_axis_s` · `quay_to_block_s` · `yt_round_trip_s` | 안벽 축은 v3 무대 구성 (02-무대 §1) |
  | `integrated/vessel.py` | `VesselClass` · `VESSEL_CLASSES` · `PORT_TIME_TABLE` · `port_time_s` | 선급 3종은 v3 축 (02b-본선) |
  | `integrated/terminal_stream.py` | `DIURNAL_LOAD_LEVELS` · `LEAD_TIME_DIST` · `sample_lead_s` | 부하 3수준·리드 분포는 v3 축 (02-무대 §2-1·§4) |

■ ⚠️ 사본의 대가 — 알고 쓴다
  · v2 엔진에 결함이 고쳐져도 **여기로 오지 않는다.** 반대도 마찬가지다.
  · 세대 간 비교(v2 vs v3)는 두 사본이 **물리적으로 같다는 가정** 위에 선다.
    그래서 위 표의 세 파일 말고는 원본과 **바이트 동일**해야 하고,
    `tests/v4/test_world_clone.py`(바이트 동일)와
    `tests/v4/test_world_equivalence.py`(같은 하루를 굴려 수치 일치)가 검사한다.

■ 여기를 고치지 않는다
  v3 가 무대를 바꿔야 하면 **위 표에 줄을 추가**하고 그 근거를 적는다.
  조용히 고치면 "v2 와 같은 무대" 라는 전제가 깨지고 짝비교가 무효가 된다.
"""
