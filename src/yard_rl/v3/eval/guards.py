"""판정 하드가드 — 위반하면 그 셀은 실격이다.

설계 정본: `.claude/docs/architecture/06-학습과-판정.md` §2

  1. 투입 = **전건** (`admitted == 그 부하의 물량`)
  2. 전 셀 존재 (**부하 3 ×** 팔 수 × 날 수) — 한 부하라도 비면 판정 미성립
  3. `policy_exceptions == 0` · 실행 트리 **`code_dirty == False`**
  4. 학습 중 표본 0 인 회차가 있으면 즉시 중단
  5. **`rollout_calls_during_eval == 0`** · **`traded_edges > 0`** — v3 신규

■ ⚠️ `code_dirty` 함정
  WSL 에서 `GIT_DIR` 없이 돌리면 `None` 을 반환해 가드가 **조용히 무력화**된다.
  `None` 은 통과가 아니라 **실패**로 다룬다.

■ v3 신규 둘
  ① **교사 누출 금지** — 판정 실행에서 반사실 rollout 이 한 번이라도 불리면 안 된다.
     교사 라벨이 학생 입력에 섞이면 미래가 새어 들어온다([05](05-정보경계.md) §3).
  ② **거래 0 감지** — 독립 행위자라 Buyer 가 전량 거절하면 거래가 0 이 되고 학습
     신호가 사라진다. "학습이 안 된 것" 과 "거래 자체가 없던 것" 은 다른 실패다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: 판정 대역 재사용 금지 — 이미 쓴 대역
USED_BANDS = (9_200_000, 9_300_000, 9_600_000, 9_700_000, 9_800_000)
#: 비판정(진단) 시드 대역
DIAGNOSTIC_BAND = 9_900_000


@dataclass
class GuardReport:
    checks: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict:
        return {"guards": dict(self.checks), "guard_failures": list(self.failures),
                "guards_passed": self.passed}


def check_cell(*, admitted: int, expected_admitted: int,
               policy_exceptions: int, code_dirty, rollout_calls: int,
               traded_edges: int) -> GuardReport:
    """한 셀(부하 × 팔 × 날)의 하드가드를 전부 본다."""
    rep = GuardReport()

    rep.checks["admitted"] = admitted
    rep.checks["expected_admitted"] = expected_admitted
    if admitted != expected_admitted:
        rep.failures.append(f"투입 미완 — {admitted} != {expected_admitted}")

    rep.checks["policy_exceptions"] = policy_exceptions
    if policy_exceptions != 0:
        rep.failures.append(f"정책 예외 {policy_exceptions}건")

    rep.checks["code_dirty"] = code_dirty
    if code_dirty is None:
        rep.failures.append("code_dirty=None — GIT_DIR 누락으로 가드가 무력화됐다")
    elif code_dirty:
        rep.failures.append("실행 트리가 dirty 하다")

    rep.checks["rollout_calls_during_eval"] = rollout_calls
    if rollout_calls != 0:
        rep.failures.append(
            f"교사 누출 — 판정에서 반사실 rollout 이 {rollout_calls}회 불렸다")

    rep.checks["traded_edges"] = traded_edges
    if traded_edges <= 0:
        rep.failures.append("거래 0 — Buyer 전량 거절. 학습 신호가 없다")

    return rep


def check_bands(seeds) -> list[str]:
    """평가 대역 재사용을 막는다. 진단 대역(9,900,0xx)은 판정에 못 쓴다."""
    bad = []
    for s in seeds:
        base = (int(s) // 100_000) * 100_000
        if base in USED_BANDS:
            bad.append(f"시드 {s} 는 이미 쓴 대역 {base} 이다")
        if (int(s) // 100_000) * 100_000 == DIAGNOSTIC_BAND:
            bad.append(f"시드 {s} 는 비판정 대역이다 — 판정에 쓸 수 없다")
    return bad


def check_matrix(cells, *, loads=(3_500, 5_000, 7_500, 12_500, 15_000),
                 arms=(), days=16) -> list[str]:
    """전 셀 존재 — **부하 3 × 팔 × 날**. 한 부하라도 비면 판정 미성립."""
    missing = []
    have = {(c["load"], c["arm"], c["day"]) for c in cells}
    for load in loads:
        for arm in arms:
            for d in range(days):
                if (load, arm, d) not in have:
                    missing.append(f"빈 셀 load={load} arm={arm} day={d}")
    return missing
