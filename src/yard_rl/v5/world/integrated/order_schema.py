"""BNCT 실제 오더 규격 정합 — A단계 (YR-187).

■ 무엇을 하나 / 안 하나
  **이름과 구조만** 실제 규격(BNCT TOS `IF_TMNL_EVENT_INFO`·`VBS_RES_MST` 등)을
  따르게 한다. **값은 지금 합성 그대로다.** 실측 분포 교체는 B단계 소관이며,
  데이터 수령 전에는 하지 않는다.

■ 왜 가산(additive)인가
  기존 키(`job_id`·`block`·`arrival_s`·`size_ft40`)의 소비처가 `"block"` 36곳,
  `"job_id"` 70곳이라 일괄 개명은 위험이 크다. 새 키를 **더하고** 기존 키는
  그대로 둔다. 신규 코드(YR-189~)는 새 이름을 쓰고, 구 코드는 안 건드린다.

  해시 안전성 확인(2026-08-18): `realization_hash` 는 scenario 를,
  `run_digest` 는 실현시각·사건열을 해싱하므로 **schedule dict 키 추가에 불변**이다.

■ 대응 (제공 명세 + 우리 코드 대조로 확정된 것만)
  · `doc_key`          ← docKey / pinNo        (구 `job_id`. 블록명 `Y08:` 접두는 합성 산물)
  · `block_assigned`   ← conLoc                최초 배정
  · `block_worked`     ← moveLoc               실제 작업 (판매 확정 시 갱신)
  · `block_previous`   ← previousConLoc        변경 이력
  · `swap_reason`      ← conSwapReason         ★새 필드 (구 `fallback_reason` 과 **다른 개념**)
  · `reserved_s`       ← inOutReserveTime      예약 — **정책 가시**
  · `gate_in_s`        ← gateInTime            실현 — **정책 비가시**
  · `size_class`       ← CNTR_SIZ·CNTR_TYP·ISO_TYP   (구 `size_ft40` 불리언 → 범주)
  · `travel_s`         ← gateInTime→blockInTime      이름 유지, 정의만 실측 파생으로
  · `exit_travel_s`    ← jobDoneTime→gateOutTime     동
  · `target`(=conNo)·`flow`(=inOut)             이름 유지

■ 현 무대의 한계 (정직 고지)
  `reserved_s == gate_in_s` 다 — 24시간 무대의 `ScheduledAnnouncer` 에 예약 준수
  오차가 없다(실측: `adherence` 0건). 둘을 쪼개는 것은 **구조 준비**이며, 값이
  갈리는 것은 [[YR-190]](σ 스윕) 소관이다.

  `expWaitingMin`·`blcWaitingCarCount`·`blockInSeq` 등은 **합성할 수 없어** 제외한다
  (터미널 산출식을 모른다). YR-187 B단계에서 실데이터로 들어온다.
"""
from __future__ import annotations

SCHEMA_VERSION = "bnct-a1"

#: 구 키 → 새 키. 구 키는 **지우지 않는다**(가산 마이그레이션).
RENAMES = {
    "job_id": "doc_key",
    "block": "block_assigned",
    "arrival_s": "reserved_s",
    "size_ft40": "size_class",
}

#: 실데이터로 가면 사라지는 합성 생성기 인공물 (제공 명세 §5 정정 ②).
SYNTHETIC_ONLY = ("requested_flow", "fallback_reason")


def size_class(size_ft40: bool) -> tuple[str, str, str | None]:
    """불리언 → 범주. 실제는 ISO 4자리(`42G1` 등)까지 오지만 합성 단계에서는
    사이즈·타입만 채우고 ISO 는 None 으로 둔다 — **없는 값을 지어내지 않는다**."""
    return ("40" if size_ft40 else "20", "GP", None)


def bnct_view(e: dict) -> dict:
    """schedule 항목 하나 → BNCT 규격 필드. 값은 그대로, 이름·구조만 바뀐다."""
    return {
        "schema": SCHEMA_VERSION,
        "doc_key": e["job_id"],
        # 블록 1개 → 3개 + 사유. 판매 전이므로 배정=실제, 이력 없음.
        "block_assigned": e["block"],
        "block_worked": e["block"],
        "block_previous": None,
        "swap_reason": None,
        # 시각 1개 → 2개. 현 무대는 준수 오차 0 이라 값이 같다(YR-190 이 갈라낸다).
        "reserved_s": e["arrival_s"],
        "gate_in_s": e["arrival_s"],
        "size_class": size_class(e["size_ft40"]),
    }


def attach(e: dict) -> dict:
    """기존 항목에 새 필드를 **더해서** 반환한다(구 키 보존)."""
    e.update(bnct_view(e))
    return e


def record_swap(e: dict, *, new_block: str, reason: str) -> None:
    """판매가 확정돼 블록이 바뀔 때 이력을 남긴다 — `conSwapReason` 대응.

    구 `fallback_reason`(합성 생성기가 반출 대상을 못 찾은 흔적)과는 **다른 개념**이다.
    """
    e["block_previous"] = e.get("block_worked", e.get("block"))
    e["block_worked"] = new_block
    e["swap_reason"] = reason
