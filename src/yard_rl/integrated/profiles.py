"""통합 터미널 프로파일 빌더 (YR-042) — YAML 근거 조립.

DGT 근사: dgt_armg.yaml(치수·ARMG 속도, 문헌 보정 v2) + dgt_public_topology.yaml
(블록당 2기·AGV 60대 공개근거) 를 IntegratedProfile 로 조립한다.
근사 한계 (명시): 육/해측 역할 고정·AGV 스케줄 연동은 미반영 — 크레인 2기가
동일 스펙으로 전 블록을 공유한다 (정식 DGT 는 별도 패키지).
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..contract.state import LaneGraph
from ..io.profile_loader import load_profile
from .profile import IntegratedProfile, TransferFleetSpec

DGT_SINGLE_YAML = "configs/terminals/dgt_armg.yaml"


def build_dgt_approx_profile(single_yaml: str | Path = DGT_SINGLE_YAML
                             ) -> IntegratedProfile:
    """DGT 근사 통합 프로파일 — 전 항목 assumed (실측은 YR-002 후).

    - 블록·크레인 스펙: dgt_armg.yaml (10열×6단·bay 24 가정·Kalmar ASC 문헌 속도)
    - 크레인 2기 (topology: 블록당 2기 — 역할분리 없이 동일 스펙 근사)
    - AGV: 60대/23블록 ≈ 3대/블록 (공개값 유도, 왕복시간 assumed 180s)
    """
    single = load_profile(single_yaml)
    base = single.crane
    cranes = tuple(replace(base, crane_id=cid) for cid in ("YC-L", "YC-W"))
    return IntegratedProfile(
        terminal_id="DGT-APPROX-2CR",
        profile_date="2026-07-15",
        assumed=True,
        block=single.block,
        cranes=cranes,
        lane_graph=LaneGraph(("L1", "L2"), (("L1", "L2"),)),
        transfer=TransferFleetSpec("AGV1", "AGV", n_units=3, move_time_s=180.0),
        long_wait_sla_s=single.long_wait_sla_s,
        decision_horizon_s=single.decision_horizon_s,
        gate_travel_estimate_s=single.gate_travel_estimate_s,
    )
