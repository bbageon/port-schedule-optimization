"""실현 지문·대역 배정 — 서로 독립이라 믿은 실험이 같은 난수 실현을 쓰는 사고 방지 (YR-108).

■ 무엇이 문제였나 (설계감사 + YR-108 정찰 실측)
시드를 `BASE + offset + i` 산술로 정해 왔다. 그런데 **본선 마감 배율(`vessel_deadline_mult`)과
정합 마감 플래그는 난수를 한 번도 소비하지 않는다** — 같은 시드면 배율이 달라도 트럭·컨테이너
실현이 통째로 같다(실측: 배율 5종 × 정합 2종 = 10조합이 실현 지문 1개).
여기에 같은 부하수준을 쓰는 두 셀의 BASE 가 정확히 200 떨어져 있고 대역 offset 도 200씩
벌어져 **공명**했다. 실측 피해:
  · YR-041-a: high-0.40 과 high-0.75 가 같은 BASE(830100) → 명목 24행 중 고유 실현 16,
    두 셀 Δ 상관 0.831, 유효 표본 ≈15.4 → **신뢰구간 반폭 약 25% 과소**.
  · YR-100[3] 평가대역은 YR-098 확증대역의 **완전한 부분집합**(24/24).
  · YR-090 평가·YR-100[2]·YR-075-c 도 YR-098 선택대역과 동일 실현.

■ 교훈 (정찰이 확립한 원칙)
**실현 정체성은 시드로도 설정으로도 결정되지 않고, 오직 생성물 내용으로만 결정된다.**
(같은 시드라도 `time_contract_v2` 여부에 따라 실현족이 갈리고, 다른 셀이라도 배율만
다르면 실현이 같다.) 그래서 시드 산술이 아니라 **생성해 보고 지문을 비교**한다.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

HASH_VERSION = "rz1"          # 지문 정의가 바뀌면 올린다 (과거 지문과 섞이지 않게)

# 지문에 넣는 것 = **확률적 추출 결과**. 마감·ETD 처럼 설계 파라미터에서 결정론적으로
# 나오는 값은 넣지 않는다 — 넣으면 "배율만 다른 같은 실현"이 다른 지문으로 보여
# 잡으려던 사고를 그대로 통과시킨다.
_JOB_FIELDS = ("job_id", "flow", "release_time", "target_container", "inbound_size",
               "inbound_load", "actual_gate_in", "actual_block_arrival", "provided_eta",
               "estimated_block_arrival", "appointment_gate_time", "exit_travel_s",
               "is_external_truck", "vessel_id")


def _val(x):
    if x is None or isinstance(x, (int, float, str, bool)):
        return x
    return getattr(x, "value", None) or str(x)


def realization_payload(scenario) -> dict:
    """지문 계산용 정준 표현 — 무엇이 들어가는지 눈으로 확인 가능해야 한다."""
    cont = sorted((c.container_id, _val(c.size), c.bay, c.row, c.tier)
                  for c in scenario.containers.values())
    jobs = []
    for j in sorted(scenario.jobs, key=lambda x: x.job_id):
        jobs.append([_val(getattr(j, f, None)) for f in _JOB_FIELDS])
    ves = sorted((v.vessel_id, _val(v.work_type), v.plan.planned_start_s,
                  v.plan.total_moves, v.plan.sts_move_interval_s) for v in scenario.vessels)
    return {"v": HASH_VERSION, "containers": cont, "jobs": jobs, "vessels": ves,
            "horizon_s": scenario.horizon_s, "drain_window_s": scenario.drain_window_s}


def realization_hash(scenario) -> str:
    blob = json.dumps(realization_payload(scenario), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)
    return f"{HASH_VERSION}:{hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]}"


@dataclass(frozen=True)
class BandSpec:
    """사전등록에 그대로 박제할 대역 명세 — 시드 **정수 목록**과 실현 지문을 함께 남긴다."""

    family: str
    seeds: dict[str, list[int]]                       # 셀/블록 이름 → 절대 시드
    hashes: dict[str, list[str]] = field(default_factory=dict)

    @property
    def all_hashes(self) -> set[str]:
        return {h for hs in self.hashes.values() for h in hs}

    def freeze_json(self) -> dict:
        return {"family": self.family, "hash_version": HASH_VERSION,
                "seeds": {k: list(v) for k, v in self.seeds.items()},
                "realization_hashes": {k: list(v) for k, v in self.hashes.items()},
                "digest": hashlib.sha256(
                    json.dumps(self.hashes, sort_keys=True).encode()).hexdigest()[:16]}


def assign_band(*, family: str, cells: dict[str, object], n: int, generate,
                exclude: set[str] | None = None, start_seed: int = 900_000,
                max_tries: int = 20_000) -> BandSpec:
    """실현이 겹치지 않는 시드 n개를 셀마다 **찾아서** 배정한다 (산술 배정 금지).

    generate(cell_key, cell_params, seed) → scenario.
    exclude: 이미 다른 대역이 쓴 실현 지문 집합. 여기와 겹치는 시드는 건너뛴다.

    셀 **사이**의 실현 공유는 허용한다 — 같은 대역 안에서 셀끼리 짝지으면 분산이 줄어
    이롭다. 금지 대상은 **대역 사이** 공유(선택↔확증)와 대역 **내부 중복**이다.
    """
    banned = set(exclude or ())
    seeds: dict[str, list[int]] = {}
    hashes: dict[str, list[str]] = {}
    # 시드 커서는 셀을 넘어 **이어서** 진행한다. 셀마다 start_seed 로 되돌리면 서로 다른
    # 셀이 같은 시드 정수를 쓰게 되고(부하수준이 달라 지문은 갈리므로 검사는 통과한다),
    # 그러면 두 블록이 **같은 난수 열**로 구동돼 의도한 블록 간 독립성이 사라진다.
    seed = start_seed
    for key, params in cells.items():
        got_s: list[int] = []
        got_h: list[str] = []
        seen_here: set[str] = set()
        tries = 0
        while len(got_s) < n:
            tries += 1
            if tries > max_tries:
                raise RuntimeError(f"{family}/{key}: 독립 실현 {n}개를 찾지 못함")
            h = realization_hash(generate(key, params, seed))
            if h not in banned and h not in seen_here:
                got_s.append(seed)
                got_h.append(h)
                seen_here.add(h)
            seed += 1
        seeds[key], hashes[key] = got_s, got_h
        banned |= seen_here          # 다음 셀이 같은 실현을 집지 않도록 (대역 내부 중복 금지)
    return BandSpec(family=family, seeds=seeds, hashes=hashes)


def independence_report(band: BandSpec, *, forbidden: dict[str, set[str]] | None = None
                        ) -> dict:
    """대역 내부 중복 + 다른 대역과의 교집합 보고 (판정 전 강제 검사용)."""
    counts: dict[str, int] = {}
    for hs in band.hashes.values():
        for h in hs:
            counts[h] = counts.get(h, 0) + 1
    dup = {h: c for h, c in counts.items() if c > 1}
    inter = {name: sorted(band.all_hashes & hs)
             for name, hs in (forbidden or {}).items() if band.all_hashes & hs}
    n_total = sum(len(v) for v in band.hashes.values())
    return {"n_rows": n_total, "n_unique": len(counts),
            "internal_duplicates": dup, "overlap_with": inter,
            "ok": not dup and not inter,
            # 명목 표본 대비 **유효 표본**의 상한 (중복은 정보를 새로 주지 않는다)
            "effective_n_upper": len(counts)}
