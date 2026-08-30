"""그림이 원자료를 드는 방법 — 한 곳에 모은다.

    from figdata import month

    month().LOAD_WEIGHTS
    month().plan_month(seed)

■ 왜 패키지로 import 하지 않는가
  `from yard_rl.v3.stage.month import ...` 는 패키지 `__init__` 을 타고
  `stage → bridge → actors → buyer → torch` 까지 끌고 온다. 그림은 학습을 하지
  않으므로 그 무게를 질 이유가 없고, **torch 가 없는 환경에서도 그림은 그려져야
  한다** (논문 그림을 다시 뽑는 사람이 학습 환경까지 갖출 필요는 없다).

  그래서 `month.py` **한 파일만** 경로로 직접 든다. 규칙 자체는 그 파일에서 오므로
  값이 갈릴 여지가 없다 — 부하 가중치도, 날짜별 부하 추첨도 같은 원본이다.

■ 왜 `month.py` 만인가
  다른 원자료(`terminal_stream` 의 도착 곡선 상수 등)는 패키지로 들어도 torch 를
  건드리지 않는다. 문제가 되는 것은 `stage/__init__` 하나뿐이라 거기만 우회한다.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

_month = None


def month():
    """`src/yard_rl/v3/stage/month.py` 를 패키지 없이 든다 (한 번만 읽는다)."""
    global _month
    if _month is None:
        path = ROOT / "src/yard_rl/v3/stage/month.py"
        spec = importlib.util.spec_from_file_location("_month_plan", path)
        mod = importlib.util.module_from_spec(spec)
        # dataclass 가 자기 모듈을 sys.modules 에서 되찾으므로 먼저 등록한다.
        sys.modules["_month_plan"] = mod
        spec.loader.exec_module(mod)
        _month = mod
    return _month
