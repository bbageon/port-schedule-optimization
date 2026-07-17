"""통합 전이 계약 스키마 (itc-v2) — YR-035, YR-050.

최종전략(별도 Exp 정책 폐기, 처음부터 차량·본선·이송장비·레인·다중 YC 를 같은
State/Action/Total Cost 계약으로 다루는 단일 통합정책)의 데이터 계약 단일 소스.

설계 원칙 (YR-035 3렌즈 병합):
- provenance(source·time-of-knowledge·unit·ablation)는 **스키마 버전당 1회** frozen
  레지스트리(`_SPECS`)에 고정한다. 인스턴스(FeatureVector)는 value/known/assumed 3채널만
  운반하고 결측은 inf/nan 이 아니라 known=0·value=0 으로 중화한다.
- 레지스트리가 최종전략 §7 전 도메인 필드를 빠짐없이 열거해 완전성을 보장한다.
- GROUND_TRUTH / NEVER 필드는 레지스트리에 두지 않는다 — 진실값(actual arrival·oracle)은
  계약 FeatureVector 에 물리적으로 부재하며 비용 정산용 별도 truth 구조(YR-036 소유)에만 존재.

필드 추가·단위·TOK 변경은 반드시 SCHEMA_VERSION bump + golden fixture 재생성.
가중치·scale·λ 는 스키마가 아닌 assumed config(configs/costs)이므로 버전 불변(§10.3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# v2 (YR-050): predicted_arrival_gap_s 의 clip_lo=0 제거 — 부호 보존(음수 = ETA 경과·미도착
# 연착). 값 범위 재해석이므로 규약(위 "변경은 반드시 bump")에 따라 명시적 버전 상승.
SCHEMA_VERSION = "itc-v2"   # integrated transition contract v2
FLOAT_DECIMALS = 6          # 교차플랫폼 float 정규화 소수자리 (직렬화 idempotent)


# --------------------------------------------------------------------- 열거형
class FieldSource(str, Enum):
    """필드가 어느 시스템에서 오는가 (실자료 매핑 감사용)."""

    TOS = "TOS"
    VBS = "VBS"
    ETA_PROVIDER = "ETA"
    VESSEL_PLAN = "VESSEL_PLAN"
    EQUIPMENT = "EQUIPMENT"
    LANE = "LANE"
    DERIVED = "DERIVED"
    GROUND_TRUTH = "GROUND_TRUTH"   # 시뮬레이터 진실 — 정책 텐서 진입 불가 (레지스트리 부재)


class TimeOfKnowledge(str, Enum):
    """정책이 언제부터 이 값을 알 수 있는가 (정보누출 게이팅 축)."""

    ALWAYS = "ALWAYS"
    PLANNED = "PLANNED"
    PRE_ADVICE = "PRE_ADVICE"
    GATE_IN = "GATE_IN"
    BLOCK_ARRIVAL = "BLOCK_ARRIVAL"
    NEVER = "NEVER"                 # actual arrival·oracle·미래 실현값 (계약 진입 불가)


class AblationGroup(str, Enum):
    """§16.3 기능 제거 ablation 단위 — 동일 정책에서 그룹을 꺼 기여도 측정."""

    CORE = "CORE"
    ETA = "ETA"
    PRE_REHANDLE = "PRE_REHANDLE"
    VESSEL_RISK = "VESSEL_RISK"
    LANE = "LANE"
    MULTI_YC = "MULTI_YC"
    LONG_WAIT = "LONG_WAIT"


class Unit(str, Enum):
    """물리단위 — validate_units 가 혼용(초/미터/비율/원)을 차단."""

    S = "s"
    M = "m"
    COUNT = "count"
    RATIO_0_1 = "ratio_0_1"
    KRW = "krw"
    BOOL01 = "bool01"
    NORM = "norm"


class CandidateKind(str, Enum):
    """최종 Action 후보 종류 — 최종전략 §8.2 (규칙번호 PriorityRule 과 의미가 다름)."""

    SERVE = "SERVE"               # 외부트럭·본선 실작업 수행
    PRE_REHANDLE = "PRE_REHANDLE"  # 도착 전 재조작 선처리
    REPOSITION = "REPOSITION"      # 크레인 위치조정
    WAIT = "WAIT"                  # 양보/대기


# --------------------------------------------------------------- FieldSpec
@dataclass(frozen=True)
class FieldSpec:
    """한 feature 채널의 스키마 명세 (버전당 고정)."""

    name: str                       # 그룹 내 유일, 텐서 채널 순서 고정
    source: FieldSource
    tok: TimeOfKnowledge
    unit: Unit
    group: str                      # "global"|"yc"|"candidate"|"queue"|"vessel"
    ablation: AblationGroup = AblationGroup.CORE
    nullable: bool = True           # known=0 허용 여부 (False 면 결측 시 검증오류)
    assumed_default: float | None = None   # 결측 imputation 값(있으면 assumed=1 로 채움)
    clip_lo: float | None = None
    clip_hi: float | None = None
    note: str = ""                  # 설계문서 § 근거


def _s(name, source, tok, unit, group, ablation=AblationGroup.CORE, *,
       nullable=False, default=None, lo=None, hi=None, note=""):
    return FieldSpec(name=name, source=source, tok=tok, unit=unit, group=group,
                     ablation=ablation, nullable=nullable, assumed_default=default,
                     clip_lo=lo, clip_hi=hi, note=note)


_SRC = FieldSource
_TOK = TimeOfKnowledge
_AB = AblationGroup
_U = Unit

# 전 도메인 필드 전수 — 최종전략 §7 / §10.2 매핑. 채널 순서 = 텐서 순서(고정).
_SPECS: tuple[FieldSpec, ...] = (
    # ---- global (§7.6·7.8·7.11·10.2) — CTDE 학습 시 특권 전역상태 ----
    _s("time_frac", _SRC.DERIVED, _TOK.ALWAYS, _U.RATIO_0_1, "global", lo=0.0, hi=1.0),
    _s("shift_idx", _SRC.DERIVED, _TOK.ALWAYS, _U.COUNT, "global", lo=0.0),
    _s("vessel_count", _SRC.VESSEL_PLAN, _TOK.PLANNED, _U.COUNT, "global", _AB.VESSEL_RISK, lo=0.0),
    _s("lane_congestion_mean", _SRC.LANE, _TOK.ALWAYS, _U.RATIO_0_1, "global", _AB.LANE, lo=0.0, hi=1.0),
    _s("lane_congestion_max", _SRC.LANE, _TOK.ALWAYS, _U.RATIO_0_1, "global", _AB.LANE, lo=0.0, hi=1.0),
    _s("sts_wait_accum_s", _SRC.EQUIPMENT, _TOK.ALWAYS, _U.S, "global", _AB.VESSEL_RISK, lo=0.0),
    _s("transfer_wait_accum_s", _SRC.EQUIPMENT, _TOK.ALWAYS, _U.S, "global", _AB.VESSEL_RISK, lo=0.0),
    _s("backlog_external", _SRC.TOS, _TOK.ALWAYS, _U.COUNT, "global", lo=0.0),
    _s("backlog_vessel", _SRC.VESSEL_PLAN, _TOK.PLANNED, _U.COUNT, "global", _AB.VESSEL_RISK, lo=0.0),
    _s("crane_count", _SRC.EQUIPMENT, _TOK.ALWAYS, _U.COUNT, "global", lo=0.0),
    _s("load_imbalance", _SRC.DERIVED, _TOK.ALWAYS, _U.NORM, "global", _AB.MULTI_YC, lo=0.0),
    # ---- yc (§7.1·7.7) ----
    _s("crane_bay", _SRC.EQUIPMENT, _TOK.ALWAYS, _U.M, "yc", lo=0.0),
    _s("trolley_row", _SRC.EQUIPMENT, _TOK.ALWAYS, _U.M, "yc", lo=0.0),
    _s("available_in_s", _SRC.DERIVED, _TOK.ALWAYS, _U.S, "yc", lo=0.0),
    _s("is_loaded", _SRC.EQUIPMENT, _TOK.ALWAYS, _U.BOOL01, "yc", lo=0.0, hi=1.0),
    _s("last_move_dir", _SRC.EQUIPMENT, _TOK.ALWAYS, _U.NORM, "yc", lo=-1.0, hi=1.0),
    _s("recent_throughput", _SRC.EQUIPMENT, _TOK.ALWAYS, _U.COUNT, "yc", lo=0.0),
    _s("recent_empty_travel_s", _SRC.EQUIPMENT, _TOK.ALWAYS, _U.S, "yc", lo=0.0),
    _s("assigned_load", _SRC.EQUIPMENT, _TOK.ALWAYS, _U.COUNT, "yc", lo=0.0),
    _s("own_queue_len", _SRC.DERIVED, _TOK.ALWAYS, _U.COUNT, "yc", lo=0.0),
    _s("own_oldest_wait_s", _SRC.DERIVED, _TOK.BLOCK_ARRIVAL, _U.S, "yc", _AB.LONG_WAIT, lo=0.0),
    _s("neighbor_load_gap", _SRC.EQUIPMENT, _TOK.ALWAYS, _U.NORM, "yc", _AB.MULTI_YC, nullable=True),
    _s("neighbor_min_gap_bay", _SRC.EQUIPMENT, _TOK.ALWAYS, _U.NORM, "yc", _AB.MULTI_YC, nullable=True, lo=0.0),
    # ---- candidate (§7.2·7.3·7.4·8.2) ----
    _s("action_kind_idx", _SRC.DERIVED, _TOK.ALWAYS, _U.NORM, "candidate", lo=0.0, hi=1.0),
    _s("is_external", _SRC.DERIVED, _TOK.ALWAYS, _U.BOOL01, "candidate", lo=0.0, hi=1.0),
    _s("is_vessel", _SRC.DERIVED, _TOK.ALWAYS, _U.BOOL01, "candidate", lo=0.0, hi=1.0),
    _s("cum_wait_s", _SRC.DERIVED, _TOK.BLOCK_ARRIVAL, _U.S, "candidate", nullable=True, lo=0.0),
    _s("long_wait_excess_s", _SRC.DERIVED, _TOK.BLOCK_ARRIVAL, _U.S, "candidate", _AB.LONG_WAIT, nullable=True, lo=0.0),
    _s("predicted_arrival_gap_s", _SRC.ETA_PROVIDER, _TOK.PRE_ADVICE, _U.S, "candidate", _AB.ETA, nullable=True,
       note="signed (v2, YR-050): 음수 = ETA 경과·미도착 연착"),  # v1 은 lo=0 절단으로 연착 신호 소실
    _s("eta_confidence", _SRC.ETA_PROVIDER, _TOK.PRE_ADVICE, _U.RATIO_0_1, "candidate", _AB.ETA, nullable=True, lo=0.0, hi=1.0),
    _s("deadline_slack_s", _SRC.DERIVED, _TOK.PLANNED, _U.S, "candidate", nullable=True),  # 음수 허용
    _s("reach_s", _SRC.DERIVED, _TOK.ALWAYS, _U.S, "candidate", lo=0.0),
    _s("expected_service_time_s", _SRC.DERIVED, _TOK.ALWAYS, _U.S, "candidate", lo=0.0),
    _s("expected_handling_count", _SRC.DERIVED, _TOK.ALWAYS, _U.COUNT, "candidate", lo=0.0),
    _s("blocker_count", _SRC.TOS, _TOK.ALWAYS, _U.COUNT, "candidate", lo=0.0),
    _s("expected_rehandle_time_s", _SRC.DERIVED, _TOK.ALWAYS, _U.S, "candidate", _AB.PRE_REHANDLE, lo=0.0),
    _s("end_bay", _SRC.DERIVED, _TOK.ALWAYS, _U.M, "candidate", lo=0.0),
    _s("lane_congestion_local", _SRC.LANE, _TOK.ALWAYS, _U.RATIO_0_1, "candidate", _AB.LANE, lo=0.0, hi=1.0),
    _s("interference_penalty_s", _SRC.DERIVED, _TOK.ALWAYS, _U.S, "candidate", _AB.MULTI_YC, lo=0.0),
    _s("resequence_count", _SRC.DERIVED, _TOK.ALWAYS, _U.COUNT, "candidate", lo=0.0),
    _s("vessel_risk_delta", _SRC.DERIVED, _TOK.PLANNED, _U.NORM, "candidate", _AB.VESSEL_RISK, nullable=True),  # 음수 허용
    # ---- queue (permutation-invariant 요약, 결정당 1개 — YR-031-b H-A 지지) ----
    _s("cand_count", _SRC.DERIVED, _TOK.ALWAYS, _U.COUNT, "queue", lo=0.0),
    _s("service_min_s", _SRC.DERIVED, _TOK.ALWAYS, _U.S, "queue", lo=0.0),
    _s("service_mean_s", _SRC.DERIVED, _TOK.ALWAYS, _U.S, "queue", lo=0.0),
    _s("service_max_s", _SRC.DERIVED, _TOK.ALWAYS, _U.S, "queue", lo=0.0),
    _s("reach_min_s", _SRC.DERIVED, _TOK.ALWAYS, _U.S, "queue", lo=0.0),
    _s("reach_mean_s", _SRC.DERIVED, _TOK.ALWAYS, _U.S, "queue", lo=0.0),
    _s("wait_max_s", _SRC.DERIVED, _TOK.BLOCK_ARRIVAL, _U.S, "queue", _AB.LONG_WAIT, lo=0.0),
    _s("wait_mean_s", _SRC.DERIVED, _TOK.BLOCK_ARRIVAL, _U.S, "queue", _AB.LONG_WAIT, lo=0.0),
    _s("outbound_share", _SRC.DERIVED, _TOK.ALWAYS, _U.RATIO_0_1, "queue", lo=0.0, hi=1.0),
    _s("short_service_share", _SRC.DERIVED, _TOK.ALWAYS, _U.RATIO_0_1, "queue", lo=0.0, hi=1.0),
    _s("vessel_urgency_max", _SRC.DERIVED, _TOK.ALWAYS, _U.RATIO_0_1, "queue", _AB.VESSEL_RISK, lo=0.0, hi=1.0),
    _s("lane_cong_mean", _SRC.DERIVED, _TOK.ALWAYS, _U.RATIO_0_1, "queue", _AB.LANE, lo=0.0, hi=1.0),
    _s("over_sla_count", _SRC.DERIVED, _TOK.BLOCK_ARRIVAL, _U.COUNT, "queue", _AB.LONG_WAIT, lo=0.0),
    # ---- vessel (§7.8·7.9·7.10) ----
    _s("slack_s", _SRC.DERIVED, _TOK.PLANNED, _U.S, "vessel", _AB.VESSEL_RISK, nullable=True),  # 음수 허용
    _s("risk", _SRC.DERIVED, _TOK.PLANNED, _U.RATIO_0_1, "vessel", _AB.VESSEL_RISK, nullable=True, lo=0.0, hi=1.0),
    _s("delay_symptom_score", _SRC.DERIVED, _TOK.ALWAYS, _U.RATIO_0_1, "vessel", _AB.VESSEL_RISK, nullable=True, lo=0.0, hi=1.0),
    _s("remaining_moves", _SRC.VESSEL_PLAN, _TOK.PLANNED, _U.COUNT, "vessel", _AB.VESSEL_RISK, lo=0.0),
    _s("remaining_service_time_s", _SRC.DERIVED, _TOK.ALWAYS, _U.S, "vessel", _AB.VESSEL_RISK, default=1200.0, lo=0.0),
    _s("sts_wait_s", _SRC.EQUIPMENT, _TOK.ALWAYS, _U.S, "vessel", _AB.VESSEL_RISK, lo=0.0),
    _s("transfer_wait_s", _SRC.EQUIPMENT, _TOK.ALWAYS, _U.S, "vessel", _AB.VESSEL_RISK, lo=0.0),
    _s("expected_delay_s", _SRC.DERIVED, _TOK.PLANNED, _U.S, "vessel", _AB.VESSEL_RISK, nullable=True, lo=0.0),
)


@dataclass(frozen=True)
class FeatureSchema:
    """`_SPECS` 를 group 별로 색인한 조회 façade."""

    version: str
    specs: tuple[FieldSpec, ...]
    _by_group: dict[str, tuple[FieldSpec, ...]] = field(default_factory=dict, compare=False)
    _by_key: dict[tuple[str, str], FieldSpec] = field(default_factory=dict, compare=False)

    def __post_init__(self):
        by_group: dict[str, list[FieldSpec]] = {}
        for sp in self.specs:
            by_group.setdefault(sp.group, []).append(sp)
            key = (sp.group, sp.name)
            if key in self._by_key:
                raise ValueError(f"중복 FieldSpec {key}")
            self._by_key[key] = sp
        for g, lst in by_group.items():
            self._by_group[g] = tuple(lst)

    def groups(self) -> tuple[str, ...]:
        return tuple(self._by_group.keys())

    def group_specs(self, group: str) -> tuple[FieldSpec, ...]:
        return self._by_group[group]

    def names(self, group: str) -> tuple[str, ...]:
        return tuple(sp.name for sp in self._by_group[group])

    def spec(self, group: str, name: str) -> FieldSpec:
        return self._by_key[(group, name)]


SCHEMA = FeatureSchema(SCHEMA_VERSION, _SPECS)

def schema_descriptor() -> dict:
    """버전 동결용 정규 기술자 — 필드/단위/TOK/ablation 변경 시 golden 이 깨진다.

    fixture 값과 무관하게 **스키마 자체**만 직렬화하므로, 값 조정으로는 흔들리지 않고
    계약 구조 변경만 감지한다 (test_schema_frozen).
    """
    return {
        "version": SCHEMA_VERSION,
        "cost_terms": list(COST_TERMS),
        "vessel_family": sorted(VESSEL_FAMILY),
        "fields": [
            {"group": sp.group, "name": sp.name, "source": sp.source.value,
             "tok": sp.tok.value, "unit": sp.unit.value, "ablation": sp.ablation.value,
             "nullable": sp.nullable, "assumed_default": sp.assumed_default,
             "clip_lo": sp.clip_lo, "clip_hi": sp.clip_hi}
            for sp in _SPECS
        ],
    }


# --------------------------------------------------------------- Cost 계약
# §10.2 총비용 13항 — 이름·순서 고정. 실수치(scale/weight/λ)는 assumed config 위임(YR-038).
COST_TERMS: tuple[str, ...] = (
    "truck_wait", "long_wait", "crane_travel", "empty_travel", "rehandle",
    "sts_wait", "transfer_wait", "vessel_delay", "depart_delay",
    "lane_cong", "interference", "resequence", "imbalance",
)
# §10.6 동적 본선계수 λ_vessel 이 곱해지는 본선 계열 항.
VESSEL_FAMILY: frozenset[str] = frozenset(
    {"sts_wait", "transfer_wait", "vessel_delay", "depart_delay"}
)
