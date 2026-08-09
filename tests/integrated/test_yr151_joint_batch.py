"""YR-151 — 공동 기록(joint) → 학습 배치 연결 계약 (33차 감사 ④, 2026-08-09).

구판 build_batch 는 개별 trail + Φ 표본만 써서 "공동 기록 미연결" 지적 대상이었다.
정정 계약: 결정 시점 Φ 는 공동 기록의 phi_pre 에서 읽고, resolver 결과를 결정에
join 해 배치에 박제하며, 결정 epoch 이 공동 기록에 없으면 즉시 실격한다.
"""
import pytest

from yard_rl.experiments.yr151_transfer_ppo import build_batch


def _trail(t=60.0, picked="J1"):
    return [{"t": t, "src": "A", "rows": None, "critic_in": None, "action": 1,
             "logp": -0.5, "value": 2.0, "n_cands": 1, "picked": picked}]


def test_batch_reads_phi_and_joins_resolver_from_joint():
    joint = [{"t": 60.0, "phi_pre": 10.0, "phi_next": 12.0, "decisions": [],
              "resolver": [{"axis": "SPACE", "src": "A", "job_id": "J1",
                            "decision": "SELL", "delta_j": -0.4, "dst": "B"}],
              "done": False}]
    b = build_batch(_trail(), joint, phi_final=15.0, v_end={"A": 1.0})
    assert b[0]["ret"] == pytest.approx(-(15.0 - 10.0) + 1.0)
    assert b[0]["adv"] == pytest.approx(b[0]["ret"] - 2.0)
    assert b[0]["resolver"]["decision"] == "SELL"
    assert b[0]["resolver"]["dst"] == "B"


def test_batch_keep_decision_has_no_resolver_join():
    joint = [{"t": 60.0, "phi_pre": 10.0, "phi_next": 11.0, "decisions": [],
              "resolver": [], "done": False}]
    b = build_batch(_trail(picked=None), joint, phi_final=12.0)
    assert b[0]["resolver"] is None


def test_batch_fails_closed_when_epoch_missing_from_joint():
    joint = [{"t": 120.0, "phi_pre": 10.0, "phi_next": 11.0, "decisions": [],
              "resolver": [], "done": False}]
    with pytest.raises(RuntimeError):
        build_batch(_trail(t=60.0), joint, phi_final=12.0)
