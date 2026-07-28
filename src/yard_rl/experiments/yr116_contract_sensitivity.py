"""YR-116 — 후보·리뷰창 계약 보정 뒤 YR-113 확증 대역 민감도 재검.

새 결론을 찾는 실험이 아니라, 이미 확증한 `0.10 이송 vs 이송 0`의 방향이
다음 두 공통 환경 보정에도 유지되는지 같은 확증 대역에서 다시 계산한다.

1. 교착 탈출 목표가 정확히 1 bay일 때 최종 REPOSITION 후보로 보존
2. actual_gate_in == 0인 반입도 review epoch에 포함

같은 대역을 재사용하므로 이 결과는 독립 확증이 아니라 **계약 민감도**다.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import yr105_conditional_transfer as y5
from . import yr113_transfer_net_effect as y113

OUT = Path("outputs/reports/yr116_candidate_review_contract")
OLD = Path("outputs/reports/yr113_transfer_net_effect/results_confirm.json")


def run() -> dict:
    old = json.loads(OLD.read_text(encoding="utf-8"))
    y5.ACHIEVABLE_DEADLINE = True

    pilot = y113._band("pilot", 8)
    select = y113._band("select", 53, set(pilot.all_hashes))
    exclude = set(pilot.all_hashes) | set(select.all_hashes)

    prev = y113.OUT
    y113.OUT = OUT
    try:
        new = y113.run("confirm", 106, exclude)
    finally:
        y113.OUT = prev

    old_ch = old["channels_notransfer_minus_base"]
    new_ch = new["channels_notransfer_minus_base"]
    summary = {
        "purpose": "YR-116 공통 계약 보정 뒤 YR-113 확증 대역 민감도 재검",
        "evidence_level": "기존 확증 대역 재사용 — 독립 확증 아님",
        "old_git": old["repro"]["code"]["git_head"],
        "new_git": new["repro"]["code"]["git_head"],
        "new_git_dirty": new["repro"]["code"]["git_dirty"],
        "n": len(new["rows"]),
        "guards": new["guards"],
        "old_deadlock_escapes": old["deadlock_escapes_total"],
        "new_deadlock_escapes": new["deadlock_escapes_total"],
        "channels": {
            ch: {"old": old_ch[ch], "new": new_ch[ch]}
            for ch in ("truck", "vessel", "move", "total")
        },
        "primary_direction_preserved": (
            old_ch["truck"]["ci"][0] > 0 and new_ch["truck"]["ci"][0] > 0
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contract_sensitivity.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=1))
    raise SystemExit(0 if result["guards"]["ok"] else 2)
