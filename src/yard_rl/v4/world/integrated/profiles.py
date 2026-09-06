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
HJNC_SINGLE_YAML = "configs/terminals/hjnc_armg.yaml"


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


#: ★야드트랙터 왕복시간(초) — **v4 전용 재보정** ([[YR-248]] 1단계 · 2026-09-06)
#:
#: ■ 왜 바꿨나
#:   v3(논문)의 180초는 공급 간격을 60초로 만들어 STS 수요(130.9초)보다 **2.18배**
#:   빨랐다. 안벽 버퍼가 마르지 않으니 배가 굶지 않았고, 본선 유휴가 Φ의 **0.84%**
#:   에 그쳤다 — 문헌이 보고하는 안벽 크레인 성능 손실 10~20% 범위 밖이다
#:   (`.claude/docs/references/본선-야드-병목-문헌.md`).
#:
#:   그 결과 **크레인이 본선 일감을 뒤로 미뤄도 손해가 안 생겼다.** 비용을 낮추도록
#:   학습하는 정책이 본선 축을 무시하는 게 합리적인 무대였다 — 꼬리표(`is_vessel`)는
#:   있는데 **배울 결과가 없었다.**
#:
#: ■ 왜 400초인가
#:   공급 간격 400/3 = 133.3초로 STS 수요 130.9초와 거의 같다(여유 0.98배). 평소엔
#:   겨우 맞추다가 **야드 크레인이 트럭에 붙들리면 배가 굶는다.** 그리고 왕복 6~7분은
#:   현실 터미널(5~10분) 안이다 — 후하게 준 값을 현실로 되돌린 것이지 인위적 조작이 아니다.
#:
#: ⚠️ v3 는 180초 그대로다. **두 세대의 무대가 다르므로 성능 비교 불가.**
YT_ROUND_TRIP_S = 400.0


def build_calibrated_profile() -> IntegratedProfile:
    """문헌 보정 v2 — "신항 표준 ARMG 블록" (YR-002 재기준화, D5·D1 사용자 확정).

    협약 트랙 폐기 후 공식 기준 프로파일. HJNC·DGT 공개 스펙 종합(두 yaml 은
    공개정보 수준에서 수치 동일 — YR-022 수렴)이며 특정사 실측 주장이 아니다.
    내용 = dgt_armg.yaml 조립(ARMG 문헌 속도·10열 6단·SLA 앵커·gate 210s)에
    중립 terminal_id 만 부여. 근거: strategy-history/2026-07-19-YR-002-D1-D5.
    기존 진단 프로파일(fixtures.build_integrated_profile)은 동결 유지 — 본 빌더는
    opt-in 이다. 부하 현실화는 scenario_gen.calibrated_load_params 가 담당.
    """
    return replace(build_dgt_approx_profile(), terminal_id="SNP-ARMG-STD")


def build_hjnc_approx_profile(single_yaml: str | Path = HJNC_SINGLE_YAML
                              ) -> IntegratedProfile:
    """HJNC 근사 통합 프로파일 — 전 항목 assumed.

    주의 (YR-022 수렴): 공개정보 수준에서 hjnc_armg 과 dgt_armg 은 수치 동일 —
    본 근사에서 실질 차이는 이송 fleet 종류(YT vs AGV, 라벨) 뿐이며 역학은 같다.
    따라서 동일 seed 실행 결과는 DGT 근사와 일치할 것으로 예상 (그 수렴 자체가
    evidence — YR-023 선례). 실차별화는 수평배열·YT 대수 등 협약(🤝) 후.
    """
    single = load_profile(single_yaml)
    base = single.crane
    cranes = tuple(replace(base, crane_id=cid) for cid in ("YC-L", "YC-W"))
    return IntegratedProfile(
        terminal_id="HJNC-APPROX-2CR",
        profile_date="2026-07-15",
        assumed=True,
        block=single.block,
        cranes=cranes,
        lane_graph=LaneGraph(("L1", "L2"), (("L1", "L2"),)),
        transfer=TransferFleetSpec("YT1", "YT", n_units=3, move_time_s=180.0),
        long_wait_sla_s=single.long_wait_sla_s,
        decision_horizon_s=single.decision_horizon_s,
        gate_travel_estimate_s=single.gate_travel_estimate_s,
    )


def build_h21_profile() -> IntegratedProfile:
    """H-21 **수평 공유형** 합성 구조 프로파일 — 이송차종이 YT 다 (YR-150 정합).

    **왜 별도 빌더인가**: Dashboard 는 H-21 을 "두 YC 가 트럭·본선을 공유하고 **YT** 로
    운송하는 구조"로 정의하는데, `build_calibrated_profile()` 은 DGT 근사에서 온
    **AGV** fleet 을 돌려준다. 자격 파일럿이 그것을 그대로 써서 **코드와 구조 정의가
    어긋나 있었다**(외부 감사 지적 2026-08-06).

    현재 YT·AGV 의 역학 수치는 동일(3대·180초)이라 결과는 바뀌지 않지만, 라벨이 어긋난
    채로 성능시험에 들어가면 "무엇을 실험했는가"가 흐려진다. 그래서 **수치는 그대로 두고
    이송차종만 YT 로 맞춘 중립 이름 프로파일**을 둔다.

    특정 터미널 이름을 쓰지 않는다 — H-21 은 합성 구조이지 HJNC 재현이 아니다.
    """
    base = build_calibrated_profile()
    return replace(base, terminal_id="H21-SHARED-YT",
                   transfer=TransferFleetSpec("YT1", "YT",
                                              n_units=base.transfer.n_units,
                                              move_time_s=YT_ROUND_TRIP_S))
