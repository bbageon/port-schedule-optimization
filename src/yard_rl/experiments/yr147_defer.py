"""YR-147 2단계 — A/B/C 학습·계측 (사전등록 동결: spec YR-147 §2단계 구현 계약).

A = 현재 무기한 WAIT (YR-145 B2 체크포인트 재사용 — 학습 없음)
B = DEFER_ALL (후보 삭제 없이 전 대기 유한화, 만료 now+600s)
C = DEFER_TRIGGER (관측 trigger 있을 때만 전략적 DEFER — 부재 시 구조 fallback 전용)
유일 변경 = 대기 행동 의미. 결속+one-shot(B2 계약)·보상·상태·PPO 불변.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..integrated import candidates as cand_mod
from .yr139_blockq_v4_ppo import train_one

OUT = Path("outputs/reports/yr147_defer")
ARM_WAIT_MODE = {"b": "DEFER_ALL", "c": "DEFER_TRIGGER"}


def train(ts: int, arm: str):
    prev = cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT, cand_mod.WAIT_MODE
    cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT = True, True     # B2 계약 유지
    cand_mod.WAIT_MODE = ARM_WAIT_MODE[arm]
    try:
        return train_one(ts, out_root=OUT / arm)
    finally:
        (cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT, cand_mod.WAIT_MODE) = prev


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, required=True)
    ap.add_argument("--arm", choices=("b", "c"), required=True)
    a = ap.parse_args()
    train(a.train, a.arm)
    print("DONE")
