"""도착 곡선 슬라이드가 쓰는 자료 — 코드와 저장된 시드 은행에서 직접 읽는다.

숫자를 이 파일에 적어 두지 않는다. 곡선은 시뮬레이터 함수에서, 실제로 뽑힌 하루치는
확증 캠페인용으로 생성·검증이 끝난 시드 은행 기록에서 가져온다. 생성기가 바뀌면
그림도 같이 바뀌어야 하기 때문이다.

  · 곡선     `yard_rl.v3.world.integrated.terminal_stream.diurnal_rate`
  · 실제 기록 `outputs/reports/yr318_seed_bank/run-d83c082/summary.json` (20달 × 30일)
  · 생성 계약 같은 폴더의 `manifest.json` 안 `generator_contract`
"""
from __future__ import annotations

import collections
import functools
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
BANK = REPO / "outputs" / "reports" / "yr318_seed_bank" / "run-d83c082"

if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from yard_rl.v3.world.integrated.terminal_stream import diurnal_rate  # noqa: E402


@functools.lru_cache(maxsize=1)
def contract() -> dict:
    """시드 은행이 실제로 쓴 생성 계약 — 봉우리·야간 바닥·부하 추첨."""
    spec = json.loads((BANK / "manifest.json").read_text(encoding="utf-8"))["spec"]
    return spec["generator_contract"]


def loads() -> list[tuple[int, float, str]]:
    """(하루 물량, 추첨 확률, 이름) 다섯. 옅은 쪽부터 짙은 쪽 순서다."""
    return [(int(v), float(p), str(name)) for v, p, name in contract()["load_weights"]]


def night_floor(total: int) -> float:
    """24시간 내내 깔리는 바닥 [대/시간]. '야간'은 봉우리가 0인 시간대에 이것만
    남아서 붙은 이름이고, 낮에도 같은 높이로 깔려 있다."""
    return contract()["night_fraction"] * total / 24.0


def curve(total: int, *, step_min: float = 2.0) -> tuple[list[float], list[float]]:
    """설계 곡선 λ(t) — x 는 시각(시), y 는 대/시간."""
    n = int(24 * 60 / step_min)
    xs = [i * step_min / 60.0 for i in range(n + 1)]
    return xs, [diurnal_rate(x * 3600.0, total=total) * 3600.0 for x in xs]


def hourly_theory(total: int) -> list[float]:
    """시간대별 기대 대수 — λ 를 한 시간씩 사다리꼴로 적분한다."""
    out = []
    for h in range(24):
        s = 0.0
        for m in range(60):
            t0 = (h * 60 + m) * 60.0
            s += 0.5 * (diurnal_rate(t0, total=total)
                        + diurnal_rate(t0 + 60.0, total=total)) * 60.0
        out.append(s)
    return out


@functools.lru_cache(maxsize=1)
def realized() -> tuple[dict[int, tuple[int, list[float]]], int]:
    """실제로 생성된 600일 — {물량: (날 수, 시간대별 평균 대수)} 와 전체 날 수."""
    rows = json.loads((BANK / "summary.json").read_text(encoding="utf-8"))["rows"]
    total = collections.defaultdict(lambda: [0.0] * 24)
    days = collections.Counter()
    for row in rows:
        for day in row["days_detail"]:
            days[day["load"]] += 1
            for h, v in enumerate(day["hourly_requests"]):
                total[day["load"]][h] += v
    return ({L: (days[L], [v / days[L] for v in total[L]]) for L in sorted(days)},
            sum(days.values()))


def crane_ceiling(blocks: int = 21, per_block: int = 2, cycle_s: float = 180.0) -> float:
    """크레인이 전부 쉬지 않고 트럭만 칠 때의 시간당 대수 — **낙관적 상한**이다.
    실제 블록별 작업시간은 190~410초이고 본선 작업도 같은 크레인이 한다.
    근거 주석: `terminal_stream.py` 5·6차 계약 머리말."""
    return blocks * per_block * (3600.0 / cycle_s)


if __name__ == "__main__":
    by_load, n_days = realized()
    print(f"생성 계약: 봉우리 {contract()['peaks']} · 바닥 {contract()['night_fraction']}")
    print(f"실제 {n_days}일")
    for L, p, name in loads():
        cnt, prof = by_load[L]
        print(f"  {L:>6,} {name:<5} 추첨 {p:>5.0%} · 실제 {cnt/n_days:>6.2%}"
              f" · 최고 {max(prof):>7.1f}대/h · 바닥 {night_floor(L):>6.1f}대/h")
    print(f"크레인 상한 {crane_ceiling():.0f}대/h")
