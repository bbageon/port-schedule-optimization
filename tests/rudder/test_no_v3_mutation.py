"""★rudder 는 v3 를 **재기만** 한다 — 고치지 않는다 ([[YR-223]]).

v3 를 복제하지 않고 가져다 쓰는 대신 지켜야 하는 계약이다. 이 시험이 깨지면
"우리가 재는 것이 v3 다" 라는 주장이 무너진다.
"""
from __future__ import annotations

import pathlib
import re

import pytest

RUDDER = pathlib.Path(__file__).resolve().parents[2] / "src" / "yard_rl" / "rudder"

#: v3 를 밖에서 고치는 흔한 수법들.
FORBIDDEN = (
    (re.compile(r"^\s*(?:yard_rl\.)?v3[\w.]*\s*="), "v3 모듈 속성에 대입"),
    (re.compile(r"\bsetattr\s*\(\s*(?:yard_rl\.)?v3"), "setattr 로 v3 변경"),
    (re.compile(r"\bmonkeypatch\b"), "monkeypatch"),
    (re.compile(r"\bimportlib\.reload\b"), "모듈 재적재"),
)


def test_source_never_writes_into_v3():
    bad = []
    for f in sorted(RUDDER.rglob("*.py")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for pat, why in FORBIDDEN:
                if pat.search(line):
                    bad.append(f"{f.name}:{i} {why} — {line.strip()}")
    assert not bad, "rudder 가 v3 를 고치려 한다:\n" + "\n".join(bad)


def test_v3_tree_is_not_written_at_runtime(tmp_path):
    """창을 굴려도 v3 소스 파일이 안 바뀐다."""
    v3 = RUDDER.parent / "v3"
    before = {f: f.stat().st_mtime_ns for f in v3.rglob("*.py")}
    from yard_rl.rudder.runner import build_ctx, run_branch
    ctx, mbt, orders, records, _b, _o = build_ctx(load=200, seed=9_900_811)
    run_branch(ctx, mbt=mbt, orders=orders, records=records, decided=set(),
               t0=3600.0, horizon_s=900.0, freeze=False, record=True)
    after = {f: f.stat().st_mtime_ns for f in v3.rglob("*.py")}
    assert before == after, "v3 파일이 실행 중에 바뀌었다"


def test_v3_episode_unaffected_by_rudder_run():
    """★rudder 를 돌린 **뒤에도** v3 에피소드가 같은 값을 낸다.

    전역 난수·모듈 상태가 새면 여기서 갈린다.
    """
    from yard_rl.rudder.runner import build_ctx, run_branch
    from yard_rl.v3.reward import reset_rollout_calls
    from yard_rl.v3.stage import run_episode

    reset_rollout_calls()
    a = run_episode(load=300, arm="RL", seed=9_900_812).as_dict()

    ctx, mbt, orders, records, _b, _o = build_ctx(load=200, seed=9_900_811)
    run_branch(ctx, mbt=mbt, orders=orders, records=records, decided=set(),
               t0=3600.0, horizon_s=900.0, freeze=False, record=True)

    reset_rollout_calls()
    b = run_episode(load=300, arm="RL", seed=9_900_812).as_dict()
    assert a == b, "rudder 실행이 v3 에피소드 결과를 바꿨다"
