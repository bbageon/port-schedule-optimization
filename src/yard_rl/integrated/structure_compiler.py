"""구조계약 compiler + 체크포인트 호환성 판정 (YR-083 step 1·5).

YR-082 manifest(구조 사실) + archetype physics_base → `StructureContract` 로 무손실 변환하고,
**현 엔진이 소비하지 못하는 필드를 명시적으로 표기**(조용히 버리지 않음 — spec 수용기준)한다.
그리고 채택 체크포인트(FT: 2크레인 대칭·SHARED 역할·214입력)의 무재학습 호환을 판정한다:

  ZERO_SHOT_COMPATIBLE   : 계약이 현 대칭 단일블록 2크레인(SHARED·YT)의 무손실 재표현 —
                           golden 궤적 보존. (수평형 HJNC형)
  SCHEMA_ADAPTATION_REQ  : 엔진이 (YR-083 런타임 후) 구조를 실행할 수 있으나, 역할분리/AGV 가
                           정책 입력 텐서 의미·후보분포를 바꿔 고정 체크포인트 zero-shot 불가 →
                           재증류·재학습 경로. (DGT 목표상태)
  STRUCTURE_UNSUPPORTED  : 현 엔진이 구조를 충실히 실행 불가 — 역할 mask·AGV/WSTP 과정·방향
                           레인·인계점 미모형. 지금 실행하면 구조가 dead field 로 조용히
                           무시됨(금지). (DGT 현재·BNCT·BCT·북항)

이 모듈은 **판정·계약 산출만** 한다. 실제 mask/resolver 엔진 소비는 YR-083 step 2 (엔진 통합,
YR-080 착지 후 별도).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..contract.structure import (BlockStructure, CraneInteraction, CraneRole,
                                   CraneSide, StructureContract, TransferControl,
                                   TransferFleet, TransferPoint, VehicleType)
from ..domain.enums import JobFlow
from ..io.profile_loader import load_profile
from .terminal_registry import MANIFEST, load_manifest

_VEH = {"YT": VehicleType.YT, "AGV": VehicleType.AGV, "SC": VehicleType.SC}
# 현 엔진이 충실히 소비하는 이송차종 (외부트럭 SERVE + YT 이송). AGV/SC 는 라우팅·WSTP 미모형.
_ENGINE_VEHICLES = frozenset({VehicleType.YT, VehicleType.EXTERNAL_TRUCK})
_LANDSIDE_WORK = (JobFlow.GATE_IN.value, JobFlow.GATE_OUT.value)
_WATERSIDE_WORK = (JobFlow.VESSEL_LOAD.value, JobFlow.VESSEL_DISCHARGE.value)


class CompatVerdict(str, Enum):
    ZERO_SHOT_COMPATIBLE = "ZERO_SHOT_COMPATIBLE"
    SCHEMA_ADAPTATION_REQUIRED = "SCHEMA_ADAPTATION_REQUIRED"
    STRUCTURE_UNSUPPORTED = "STRUCTURE_UNSUPPORTED"


@dataclass(frozen=True)
class CompiledStructure:
    terminal_id: str
    archetype: str
    contract: StructureContract
    engine_verdict: CompatVerdict          # 현 엔진 실행 가능성
    target_verdict: CompatVerdict          # YR-083 런타임 후 체크포인트 경로
    supported: tuple[str, ...]             # 현 엔진이 소비하는 측면
    unsupported: tuple[tuple[str, str], ...]  # (측면, 사유) — 실행 바꾸는데 엔진 미구현 → 실행 차단
    # 확인됐으나 엔진에 자리가 없는 구조필드 — **실행 불변**이라 ZERO_SHOT 을 막지 않지만
    # 조용히 버리지 않고 여기 기록한다(compiler 자기규약, 적대검증 반영).
    engine_ignored: tuple[tuple[str, str], ...]
    reasons: tuple[str, ...]


def _block_from_manifest(entry: dict, base_bay: int | None) -> BlockStructure:
    lay = entry.get("layout", {})
    def _v(k):
        f = lay.get(k, {})
        return f.get("v") if f.get("s") in ("confirmed", "derived") else None
    return BlockStructure(
        block_id=f"{entry['archetype']}-B1",
        bay_count=base_bay,                 # 전 터미널 bay 미공개 → physics_base assumed
        row_count=_v("rows"), tier_max=_v("tiers"),
        orientation=_v("orientation"),
        provenance="assumed" if base_bay is not None else "unresolved")


def _crane_roles(entry: dict, role_split: bool, base_bay: int | None) -> tuple[CraneRole, ...]:
    block_id = f"{entry['archetype']}-B1"
    lo, hi = 1, (base_bay or 24)
    if role_split:
        return (
            CraneRole("YC-L", block_id, CraneSide.LANDSIDE, lo, hi, _LANDSIDE_WORK,
                      ("B1-LSTP",), provenance="confirmed"),
            CraneRole("YC-W", block_id, CraneSide.WATERSIDE, lo, hi, _WATERSIDE_WORK,
                      ("B1-WSTP",), provenance="confirmed"))
    return (CraneRole("YC-L", block_id, CraneSide.SHARED, lo, hi, (), (), "assumed"),
            CraneRole("YC-W", block_id, CraneSide.SHARED, lo, hi, (), (), "assumed"))


def _transfer_points(entry: dict, role_split: bool) -> tuple[TransferPoint, ...]:
    if not role_split:
        return ()
    block_id = f"{entry['archetype']}-B1"
    feats = entry.get("features", {})
    hard = feats.get("hard_vehicle_separation", {}).get("s") == "confirmed"
    return (
        TransferPoint("B1-LSTP", block_id, CraneSide.LANDSIDE, (VehicleType.EXTERNAL_TRUCK,),
                      control=TransferControl.TRAFFIC_LIGHT_AND_READY_BUTTON,
                      provenance="confirmed" if hard else "assumed"),
        TransferPoint("B1-WSTP", block_id, CraneSide.WATERSIDE, (VehicleType.AGV,),
                      control=TransferControl.FMS,
                      provenance="confirmed" if hard else "assumed"))


def compile_terminal(terminal_id: str, manifest_path: str | Path = MANIFEST) -> CompiledStructure:
    m = load_manifest(manifest_path)
    if terminal_id not in m["terminals"]:
        raise KeyError(f"미등록 터미널: {terminal_id!r}")
    entry = m["terminals"][terminal_id]
    arch_name = entry["archetype"]
    arch = m["archetypes"][arch_name]
    role_split = bool(arch.get("role_separated"))     # None→False (미확인은 분리로 안 봄)

    base_bay = None
    if arch.get("physics_base"):
        try:
            base_bay = load_profile(arch["physics_base"]).block.bay_count
        except Exception:
            base_bay = None

    tk = entry.get("transfer", {}).get("kind", {})
    veh = _VEH.get(tk.get("v")) if tk.get("s") == "confirmed" else None
    fleets = ()
    if veh is not None:
        fleets = (TransferFleet(f"{terminal_id}-fleet", veh,
                                n_units=None, move_time_s=None,
                                provenance="confirmed"),)

    contract = StructureContract(
        terminal_id=terminal_id, archetype=arch_name,
        blocks=(_block_from_manifest(entry, base_bay),),
        crane_roles=_crane_roles(entry, role_split, base_bay),
        crane_interaction=CraneInteraction(can_cross=None, min_separation_bay=2.0,
                                           provenance="assumed"),
        transfer_points=_transfer_points(entry, role_split),
        road_segments=(),                  # 방향·용량 레인 전부 미확보 (Level2 대기)
        transfer_fleets=fleets,
        role_separated=arch.get("role_separated"),
        provenance="assumed",
        notes=("공개 Level0~1 · 방향/용량 레인·인계점 용량은 Level2 실측 대기",))

    # 확인됐으나 엔진 모델에 자리가 없는 구조필드 — 조용히 버리지 않고 기록(실행은 불변).
    ignored: list[tuple[str, str]] = []
    lay = entry.get("layout", {})
    ori = lay.get("orientation", {})
    if ori.get("s") in ("confirmed", "derived") and ori.get("v"):
        ignored.append(("layout.orientation",
                        f"확인 {ori['v']} — 엔진 BlockGeometry 에 orientation 필드 없음(실행 불변)"))
    bc = lay.get("block_count", {})
    if bc.get("s") in ("confirmed", "derived") and bc.get("v"):
        ignored.append(("layout.block_count",
                        f"확인/유도 {bc['v']}개 — 엔진은 단일 블록만 표현(다중 블록 미모형)"))

    return _judge(contract, arch, base_bay, tuple(ignored))


def _judge(c: StructureContract, arch: dict, base_bay: int | None,
           engine_ignored: tuple[tuple[str, str], ...]) -> CompiledStructure:
    supported, unsupported, reasons = [], [], []

    # 크레인 역할
    if c.is_role_split:
        unsupported.append(("crane_role_split",
                            "육/해측 역할 mask 미구현 — 현 엔진은 SHARED 대칭 2크레인만"))
    else:
        supported.append("crane_role(SHARED)")

    # 이송 차종
    unknown_veh = c.vehicle_types - _ENGINE_VEHICLES
    if unknown_veh:
        unsupported.append(("transfer_fleet",
                            f"{sorted(v.value for v in unknown_veh)} 라우팅·인계 과정 미모형"))
    elif c.vehicle_types:
        supported.append("transfer_fleet(YT)")

    # 인계점
    if c.transfer_points:
        unsupported.append(("transfer_points", "인계점 용량·제어절차 미모형"))

    # 블록 기하 (bay 없으면 실행 자체 불가)
    if base_bay is None:
        unsupported.append(("block_geometry", "physics_base 없음 — bay/좌표 미확보로 실행 프로파일 불가"))
    else:
        supported.append("block_geometry(assumed base)")

    supported.append("crane_interaction(min_separation=2bay)")

    # ── 판정
    can_express = base_bay is not None      # 구조를 실행 프로파일로 표현 가능한가
    if not unsupported:
        engine = CompatVerdict.ZERO_SHOT_COMPATIBLE
        target = CompatVerdict.ZERO_SHOT_COMPATIBLE
        reasons.append("대칭 단일블록 2크레인(SHARED·YT) 실행 substrate 로 환원 — 입력 텐서 "
                       "차원·의미 불변이라 동결 FT 무재학습 실행(golden 보존). 확인됐으나 엔진에 "
                       "자리 없는 구조필드는 engine_ignored 로 명시(조용한 유실 0)")
    else:
        engine = CompatVerdict.STRUCTURE_UNSUPPORTED
        if can_express and c.is_role_split:
            target = CompatVerdict.SCHEMA_ADAPTATION_REQUIRED
            reasons.append("YR-083 런타임 후에도 역할분리가 입력 텐서·후보분포·해측 목적을 바꿔 "
                           "고정 FT 체크포인트 zero-shot 불가 → 재증류/재학습")
        else:
            target = CompatVerdict.STRUCTURE_UNSUPPORTED
            reasons.append("구조 표현 불가(기하/차종 미확보) — Level2 실측·엔진 과정 확장 선결")
    for asp, why in unsupported:
        reasons.append(f"[미지원] {asp}: {why}")
    for asp, why in engine_ignored:
        reasons.append(f"[엔진무시·기록] {asp}: {why}")

    return CompiledStructure(
        terminal_id=c.terminal_id, archetype=c.archetype, contract=c,
        engine_verdict=engine, target_verdict=target,
        supported=tuple(supported), unsupported=tuple(unsupported),
        engine_ignored=engine_ignored, reasons=tuple(reasons))


def compile_all(manifest_path: str | Path = MANIFEST) -> list[CompiledStructure]:
    m = load_manifest(manifest_path)
    return [compile_terminal(tid, manifest_path) for tid in m["terminals"]]
