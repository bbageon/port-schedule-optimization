"""pytest 플러그인 — 시험 모듈의 `REPORT`(dict) 를 조각 실행 사이에 이어 준다 ([[YR-327]]).

■ 왜 필요한가
  `tests/v6/test_gpu_*.py` 몇 개는 파일 끝에 **전체 집계 시험**(`zz_report`·`ties_were_exercised`·
  `escape_paths` 등)을 둔다 — 앞 시험들이 모듈 전역 `REPORT` 에 쌓은 통계(동률 건수·탈출 발화 수·
  거절 사유 분포)가 실제로 시험됐는지 단언한다. 파일을 **조각내 따로 돌리면** REPORT 가 비어 집계
  시험이 거짓 실패한다. 이 플러그인이 조각마다 REPORT 를 JSON 으로 남기고(`REPORT_TAG`), 집계
  시험을 돌릴 때 병합본을 미리 채워 넣는다(`REPORT_LOAD`).

■ 언제 쓰나
  WSL 세션이 짧게 끊기는 환경(2026-09-25 이 기계: 배포판이 부팅 ~88초 뒤 종료)에서
  `scripts/v6/verify_chunked.sh` 가 쓴다. 정상 환경에서는 파일을 통째로 돌리면 되고 이 플러그인은
  필요 없다.

■ 사용
    PYTHONPATH=src:tests/v6 REPORT_TAG=<이름> [REPORT_DIR=<폴더>] pytest ... -p report_dump
    PYTHONPATH=src:tests/v6 REPORT_LOAD=<merged.json>            pytest ... -p report_dump -k zz_report
  `REPORT_DIR` 기본값은 이 파일의 폴더 — 저장소를 더럽히지 않게 `outputs/v6/verify/` 를 권한다.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _report_modules() -> dict:
    out = {}
    for name, mod in list(sys.modules.items()):
        rep = getattr(mod, "REPORT", None)
        if isinstance(rep, dict) and ("test_gpu" in name or name.startswith("_tg")):
            out[name] = rep
    return out


def _fix_keys(d):
    """JSON 은 dict 키를 문자열로 만든다 — 정수 키(거절 코드 등)만 되돌린다."""
    if isinstance(d, dict):
        return {(int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): _fix_keys(v)
                for k, v in d.items()}
    if isinstance(d, list):
        return [_fix_keys(x) for x in d]
    return d


def pytest_collection_finish(session):
    path = os.environ.get("REPORT_LOAD")
    if not path or not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        merged = json.load(f)
    for name, rep in _report_modules().items():
        base = name.split(".")[-1]
        for key, data in merged.items():
            if key.split(".")[-1] != base:
                continue
            for k, v in data.items():
                if isinstance(v, dict):
                    rep.setdefault(k, {}).update(_fix_keys(v))
                else:
                    rep[k] = v


def pytest_sessionfinish(session, exitstatus):
    tag = os.environ.get("REPORT_TAG")
    if not tag:
        return
    out = {name: rep for name, rep in _report_modules().items() if rep}

    def default(o):
        try:
            return float(o)
        except Exception:
            return str(o)

    d = os.environ.get("REPORT_DIR", _HERE)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"report_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, default=default, ensure_ascii=False)
