"""합성 시나리오 생성기 (YR-017) — 실자료 확보 전 PoC 구동용.

⚠ 실측 데이터의 대체물이 아니다. 분포·물량은 전부 가정값이며, 결과는
'가정 프로파일 + 합성 시나리오' 조건의 예비 PoC 로만 해석한다 (YR-005·009 로 대체 예정).

- 모든 난수는 random.Random(seed) 로 결정론적: 같은 (params, seed) → 같은 시나리오.
- episode = 8시간 shift (03 §2.2), drain 2시간 clear-out.
- 초기 장치: 스택 단위로 규격을 통일해 same-size 적재규칙을 만족.
- rehandle_risk: GATE_OUT 대상이 스택 하부(매몰)일 확률 — blocker 발생량 제어.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..domain.enums import ContainerSize, JobFlow, LoadStatus
from ..domain.models import Container, Job, TerminalProfile
from ..domain.scenario import Scenario


@dataclass(frozen=True)
class GenParams:
    n_external: int = 100          # 외부트럭 작업 수 / shift
    gate_out_share: float = 0.6    # 반출 비중
    n_vessel: int = 8              # 본선·내부 연계 작업 수
    fill_ratio: float = 0.45       # 초기 장치율 (용량 대비)
    rehandle_risk: float = 0.35    # 반출 대상이 매몰 컨테이너일 확률
    peak: bool = False             # True: 2~5h 구간 도착률 2배
    horizon_s: float = 28800.0     # 8h shift
    drain_window_s: float = 7200.0
    gate_offset_range_s: tuple[float, float] = (300.0, 900.0)  # 게이트→블록 소요
    size_mix_ft40: float = 0.7


def _arrival_times(rng: random.Random, n: int, horizon: float, peak: bool) -> list[float]:
    """시간대별 가중 추출 (피크: 2~5h 구간 2배 밀도). 정렬된 도착시각."""
    times = []
    for _ in range(n):
        if peak:
            # 가중 구간 선택: 피크 3h 는 2배 가중
            zones = [(0.0, 7200.0, 1.0), (7200.0, 18000.0, 2.0), (18000.0, horizon, 1.0)]
            weights = [(hi - lo) * w for lo, hi, w in zones]
            lo, hi, _w = rng.choices(zones, weights=weights)[0]
            times.append(rng.uniform(lo, hi))
        else:
            times.append(rng.uniform(0.0, horizon))
    return sorted(times)


def _build_initial_yard(rng: random.Random, profile: TerminalProfile,
                        p: GenParams) -> dict[str, Container]:
    geom = profile.block
    capacity = geom.bay_count * geom.row_count * geom.tier_max
    target_count = int(capacity * p.fill_ratio)
    containers: dict[str, Container] = {}
    seq = 0
    stacks = [(b, r) for b in range(1, geom.bay_count + 1) for r in range(1, geom.row_count + 1)]
    rng.shuffle(stacks)
    for (bay, row) in stacks:
        if seq >= target_count:
            break
        height = min(rng.randint(1, geom.tier_max), target_count - seq)
        size = ContainerSize.FT40 if rng.random() < p.size_mix_ft40 else ContainerSize.FT20
        for tier in range(1, height + 1):
            seq += 1
            cid = f"C{seq:04d}"
            containers[cid] = Container(
                container_id=cid, size=size,
                load_status=LoadStatus.FULL if rng.random() < 0.8 else LoadStatus.EMPTY,
                block=geom.block_id, bay=bay, row=row, tier=tier)
    return containers


def _pick_targets(rng: random.Random, containers: dict[str, Container],
                  n: int, rehandle_risk: float) -> list[str]:
    """반출 대상 선정 — rehandle_risk 확률로 매몰 컨테이너를 고른다."""
    by_stack: dict[tuple[int, int], list[str]] = {}
    for c in sorted(containers.values(), key=lambda x: x.container_id):
        by_stack.setdefault((c.bay, c.row), []).append(c.container_id)
    for pile in by_stack.values():
        pile.sort(key=lambda cid: containers[cid].tier)
    tops = [pile[-1] for pile in by_stack.values()]
    buried = [cid for pile in by_stack.values() for cid in pile[:-1]]
    rng.shuffle(tops)
    rng.shuffle(buried)
    picked: list[str] = []
    for _ in range(n):
        take_buried = buried and rng.random() < rehandle_risk
        pool = buried if take_buried else (tops or buried)
        if not pool:
            break
        picked.append(pool.pop())
    return picked


def generate(profile: TerminalProfile, seed: int, params: GenParams | None = None) -> Scenario:
    p = params or GenParams()
    rng = random.Random(seed)
    containers = _build_initial_yard(rng, profile, p)

    n_out = int(p.n_external * p.gate_out_share)
    n_in = p.n_external - n_out
    out_targets = _pick_targets(rng, containers, n_out, p.rehandle_risk)
    n_out = len(out_targets)  # 야드가 작으면 축소될 수 있음

    jobs: list[Job] = []
    arrivals = _arrival_times(rng, n_out + n_in, p.horizon_s, p.peak)
    for i, arrival in enumerate(arrivals):
        offset = rng.uniform(*p.gate_offset_range_s)
        gate_in = max(0.0, arrival - offset)
        if i < n_out:
            jobs.append(Job(job_id=f"JO{i:04d}", flow=JobFlow.GATE_OUT, release_time=0.0,
                            actual_gate_in=gate_in, actual_block_arrival=arrival,
                            provided_eta=None, target_container=out_targets[i]))
        else:
            size = ContainerSize.FT40 if rng.random() < p.size_mix_ft40 else ContainerSize.FT20
            jobs.append(Job(job_id=f"JI{i:04d}", flow=JobFlow.GATE_IN, release_time=0.0,
                            actual_gate_in=gate_in, actual_block_arrival=arrival,
                            inbound_size=size, inbound_load=LoadStatus.FULL))

    # 본선·내부 연계: 반출 대상과 겹치지 않는 컨테이너
    used = set(out_targets)
    vessel_pool = [cid for cid in sorted(containers) if cid not in used]
    rng.shuffle(vessel_pool)
    for k in range(min(p.n_vessel, len(vessel_pool))):
        rel = rng.uniform(0.0, p.horizon_s * 0.8)
        jobs.append(Job(job_id=f"JV{k:04d}", flow=JobFlow.VESSEL_LOAD, release_time=rel,
                        actual_gate_in=None, actual_block_arrival=None,
                        target_container=vessel_pool[k],
                        deadline=rel + rng.uniform(3600.0, 7200.0), priority_class=1))

    jobs.sort(key=lambda j: j.job_id)
    sid = (f"syn_n{p.n_external}_v{p.n_vessel}_f{int(p.fill_ratio * 100)}"
           f"_r{int(p.rehandle_risk * 100)}_{'peak' if p.peak else 'norm'}_s{seed}")
    return Scenario(scenario_id=sid, seed=seed, horizon_s=p.horizon_s,
                    drain_window_s=p.drain_window_s, jobs=jobs, containers=containers,
                    meta={"generator": "synthetic-v1", "assumed": True,
                          "params": str(p)})
