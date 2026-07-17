"""계약 버전 동결 — 필드/단위/TOK/직렬화 변경은 SCHEMA_VERSION bump 을 강제 (YR-035).

golden 이 깨지면: (의도된 계약 변경이면) SCHEMA_VERSION 을 올리고 golden 을 재생성,
(의도치 않은 변경이면) 회귀를 수정한다. 스키마 기술자와 전이 fixture 를 분리 동결한다.
"""
import json
from pathlib import Path

from yard_rl.contract import (SCHEMA_VERSION, build_minimal_transition, dumps)
from yard_rl.contract.schema import schema_descriptor

_HERE = Path(__file__).parent
# 파일명을 버전에서 유도 — bump 시 구버전 golden 이 조용히 재사용되지 않고 부재로 즉시 발화.
_SCHEMA_GOLDEN = _HERE / f"golden_schema_{SCHEMA_VERSION}.json"
_TRANS_GOLDEN = _HERE / f"golden_transition_{SCHEMA_VERSION}.json"


def test_schema_frozen():
    """스키마 구조(필드·단위·TOK·ablation) 동결 — 값과 무관."""
    want = json.dumps(schema_descriptor(), sort_keys=True, ensure_ascii=False, indent=1)
    got = _SCHEMA_GOLDEN.read_text(encoding="utf-8").rstrip("\n")
    assert want == got, "SCHEMA 가 golden 과 불일치 — 계약 변경이면 SCHEMA_VERSION bump + golden 재생성"


def test_transition_serialization_frozen():
    """전이 직렬화 bit 동결 — fixture·serialize 회귀 감지."""
    want = dumps(build_minimal_transition())
    got = _TRANS_GOLDEN.read_text(encoding="utf-8").rstrip("\n")
    assert want == got


def test_golden_version_matches():
    desc = json.loads(_SCHEMA_GOLDEN.read_text(encoding="utf-8"))
    assert desc["version"] == SCHEMA_VERSION
    assert _TRANS_GOLDEN.name.endswith(f"{SCHEMA_VERSION}.json")
