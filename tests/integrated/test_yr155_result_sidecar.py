"""YR-155 — 결과 파일의 해시는 파일 밖(sidecar)에 둔다.

결과 JSON 안에 자기 해시를 적으면 **적는 순간 내용이 바뀌어** 기록값이
덧쓰기 전 파일의 해시가 된다. 검증하는 쪽이 쓸 수 없는 값이다
(2026-08-06 YR-151 0A 실측: 기록 `4862ae71…` vs 실제 `c287a9d5…`).
원리상 자기검증은 불가능하므로 규약 자체를 바꾼다.
"""
from __future__ import annotations

import hashlib
import json

from yard_rl.integrated.repro import (SIDECAR_SUFFIX, sha256_file,
                                      verify_result, write_result)


def test_sidecar_matches_the_written_file(tmp_path):
    p = tmp_path / "r.json"
    digest = write_result(p, {"a": 1, "b": "한글", "c": [1.5, None]})
    side = p.with_name(p.name + SIDECAR_SUFFIX)

    assert side.is_file()
    assert side.read_text(encoding="utf-8").strip() == digest
    # **자기검증이 실제로 성립한다** — 구 규약이 못 하던 것
    assert digest == sha256_file(p)
    assert digest == hashlib.sha256(p.read_bytes()).hexdigest()
    assert verify_result(p) is True


def test_payload_does_not_carry_its_own_hash(tmp_path):
    """새 규약은 결과 안에 self_sha256 을 넣지 않는다."""
    p = tmp_path / "r.json"
    write_result(p, {"verdict": "PASS"})
    assert "self_sha256" not in json.loads(p.read_text(encoding="utf-8"))


def test_tampering_is_detected(tmp_path):
    """탐지력 — 파일이 한 글자만 바뀌어도 잡는다."""
    p = tmp_path / "r.json"
    write_result(p, {"verdict": "PASS"})
    assert verify_result(p) is True
    p.write_text(p.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert verify_result(p) is False


def test_missing_sidecar_is_unknown_not_pass(tmp_path):
    """구 산출물은 sidecar 가 없다 — False(위반)가 아니라 None(미상)이다."""
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"self_sha256": "옛 규약"}), encoding="utf-8")
    assert verify_result(p) is None
    assert verify_result(tmp_path / "없는파일.json") is None


def test_zero_a_writer_uses_sidecar():
    """YR-151 0A 산출부가 새 규약을 쓰는지 — 구 2회 덧쓰기 흔적이 없어야 한다.

    모듈을 import 하면 torch 의존이 딸려오므로(Windows 미설치) **소스를
    텍스트로** 읽는다. 검사 대상은 코드 문자열이라 이것으로 충분하다.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "src" / "yard_rl" /
           "experiments" / "yr151_pre_gate_0a.py").read_text(encoding="utf-8")
    assert "write_result(p, res)" in src
    assert 'res["self_sha256"]' not in src
