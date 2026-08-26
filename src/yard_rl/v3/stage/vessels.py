"""본선 — 선급 3종을 **무대에 실제로 박는다** ([[YR-212]]).

설계 정본: `.claude/docs/architecture/02b-본선.md`

■ 무엇이 달라지나

| | 구 무대 | v3 |
|---|---|---|
| 선급 | 없음 — 전 본선 동등 | **3종** (50k·100k·150k GT) |
| 척수 | 12척 고정 | **소2 · 중1 · 대 30% 확률** |
| 기항 물량 | 척당 **120 고정** | `TEU × U(15%, 30%)` → 450~4,200 |
| 접안 시간 | 척당 4.8h | **실측 Port Time** 20.4 / 27.7 / 38.6h |
| STS 스트림 | 척당 1 (12스트림) | 선급별 **2 / 4 / 6** — 하루 **8~14 스트림** |
| 스트림↔본선 | 1:1 | **다:1** — 한 배가 여러 블록에서 받는다 |

■ ★대형선 희소성은 '척수'가 아니라 '출현 확률'이다
  접안이 20~47시간이라 동시 3척 안팎이고, 비율로는 희소성을 못 만든다. 그래서
  **오는 날 / 안 오는 날**로 가른다(대형 30%). 학습에도 이쪽이 낫다 — *"오늘 대형선이
  있다"* 가 하루의 성격을 바꿔야 정책이 배울 변동이 생긴다.

  **본선 구성은 판정 축이 아니라 시드별 변동**이다(02b §3). 축을 둘로 만들면 셀 수가
  곱으로 는다 — 판정 축은 트럭 물량 3수준뿐이다.

■ ★블록당 1 스트림 계약은 그대로 지킨다
  엔진은 블록당 스트림 1개를 하드 제약으로 건다. 스트림이 최대 14개라 21블록 안에
  들어가므로 **구조를 안 고치고** 선급별 STS 대수가 표현된다. 한 배의 STS 들이
  서로 다른 블록에서 공급받는 것은 실제 터미널이 그렇다.

■ ★창 경계 — 접안이 하루를 넘는다
  중·대형은 Port Time 이 24시간을 넘는다. 결함이 아니라 현실이다(실제 터미널은
  연속 운영이고 우리 에피소드는 그 창일 뿐).

      창 시작:  이미 접안해 작업 중인 배가 있다 (warm start)
      창 끝:    안 끝난 배가 있다 — 창 안 몫만 센다 (검열)

  그래서 배마다 접안 시작을 창 **앞뒤로 걸치게** 뽑고, 창과 겹치는 구간에 비례해
  **창 안에서 처리할 물량**만 스트림에 싣는다. 트럭 미완료 검열과 같은 원리다.
"""
from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass

from ..world.integrated.terminal_stream import (DIURNAL_DAY_TOTAL,
                                                DIURNAL_DRAIN_S, OBS_24H,
                                                ObservationContract,
                                                TerminalStreamParams,
                                                _background, _clamp_travel,
                                                _retime_vessels, allocate,
                                                attach_bnct, diurnal_arrivals,
                                                distribution_vector)
from ..world.integrated.scenario_gen import trunc_normal
from ..world.integrated.vessel import (VESSEL_CLASSES, VesselClass, port_time_s,
                                       sample_vessel_moves)
from ..world.integrated.yard_layout import YardLayout, terminal_layout

#: 하루 본선 구성 — (선급 이름, 척수, 출현 확률). 02b §3.
DAY_FLEET = (("SMALL", 2, 1.0), ("MEDIUM", 1, 1.0), ("LARGE", 1, 0.30))

#: 스트림 1개(STS 1대)의 생산성 — 25~30 moves/h 의 중앙값.
STREAM_MOVES_PER_H = 27.5


@dataclass(frozen=True)
class DailyVessel:
    """그날 들어온 배 한 척."""

    vessel_id: str
    cls: VesselClass
    moves: int                 # 기항 총 물량
    port_time_s: float         # 접안 시간 (실측표)
    berth_start_s: float       # 접안 시작 — **음수면 창 시작 전부터 붙어 있었다**

    @property
    def gt(self) -> int:
        return self.cls.gt

    def window_overlap_s(self, observe_s: float) -> float:
        """접안 구간과 관측창이 겹치는 길이."""
        lo = max(0.0, self.berth_start_s)
        hi = min(observe_s, self.berth_start_s + self.port_time_s)
        return max(0.0, hi - lo)

    def work_s(self) -> float:
        """순수 하역 시간 = 물량 ÷ (STS × 스트림 생산성). 접안 시간보다 짧다."""
        return self.cls.work_time_s(self.moves)

    def moves_in_window(self, observe_s: float) -> int:
        """창 안에서 처리할 물량. 작업이 창 안에 들어가게 배치하므로 **전량**이다.

        접안 시간(20~47h)이 창을 넘는 것과 별개다 — 넘는 부분은 **구조적 유휴**
        (정박 대기·조선·접이안·검역·교대)라 정책이 못 바꾸고 시뮬레이션하지 않는다.
        """
        end = self.berth_start_s + self.work_s()
        if self.berth_start_s >= observe_s:
            return 0
        if end <= observe_s:
            return self.moves
        share = (observe_s - self.berth_start_s) / max(1e-9, self.work_s())
        return int(round(self.moves * share))


def sample_day_vessels(seed: int, *, obs: ObservationContract | None = None
                       ) -> list[DailyVessel]:
    """그날 오는 배 명단 — **소2 · 중1 · 대 30%** (02b §3). 시드에서 미리 뽑는다.

    ■ 왜 "정상상태 입항률" 로 안 뽑나 (2026-08-23 판단)
      접안이 20~47시간이라 *"어제 온 배가 오늘도 붙어 있다"* 를 반영하면 창에 걸치는
      배가 6~7척·스트림 20개가 되고 **대형선이 80% 의 날에 보인다.** 물리적으로는
      그쪽이 맞지만 **설계 정본(02b §3)이 정한 것은 "그날 처리하는 물량"** 이다 —
      소2·중1·대30% 로 하루 평균 3,982 moves, 스트림 8 또는 14. 문서를 따른다.

      대신 **작업이 창 안에 들어간다**: 작업 시간은 소 12.3h · 중 15.3h · 대 19.1h 로
      셋 다 24시간 안이다(접안 시간이 창을 넘는 것과 별개다 — 접안의 나머지는
      구조적 유휴라 정책과 무관하고 시뮬레이션하지 않는다).

    ■ 대형선 희소성은 **출현 확률**이다
      "오는 날 / 안 오는 날" 로 갈라야 *"오늘 대형선이 있다"* 가 하루의 성격을 바꾸고
      정책이 배울 변동이 생긴다. 매일 같으면 배울 게 없다.
    """
    obs = obs or OBS_24H
    rng = random.Random(f"v3:fleet:{seed}")
    by_name = {c.name: c for c in VESSEL_CLASSES}
    out: list[DailyVessel] = []
    for name, n_day, prob in DAY_FLEET:
        cls = by_name[name]
        for k in range(n_day):
            if rng.random() >= prob:
                continue                       # 오늘은 안 온다 (희소성)
            moves = sample_vessel_moves(cls, rng.random())
            work_s = cls.work_time_s(moves)
            # 작업이 창 안에 들어가도록 시작을 고른다. 남는 폭이 없으면 0 시부터.
            room = max(0.0, obs.observe_s - work_s)
            out.append(DailyVessel(
                vessel_id=f"{name[0]}{k+1}", cls=cls, moves=moves,
                port_time_s=port_time_s(moves),
                berth_start_s=rng.uniform(0.0, room)))
    return out


def plan_streams(vessels: list[DailyVessel], layout: YardLayout, seed: int, *,
                 obs: ObservationContract | None = None) -> list[dict]:
    """배 → STS 스트림들 → 블록 배정. **한 블록에 스트림 하나**를 지킨다."""
    obs = obs or OBS_24H
    blocks = list(layout.ids)
    random.Random(f"v3:vblk:{seed}").shuffle(blocks)

    rows: list[dict] = []
    for v in vessels:
        in_win = v.moves_in_window(obs.observe_s)
        if in_win < v.cls.sts:
            continue                            # 창 안 몫이 스트림 수보다 적다
        per = max(1, in_win // v.cls.sts)
        start = max(0.0, v.berth_start_s)
        for k in range(v.cls.sts):
            rows.append({"vessel_id": v.vessel_id, "vessel_class": v.cls.name,
                         "gt": v.cls.gt, "sts": v.cls.sts,
                         "stream": k, "moves": per,
                         "cadence_s": 3600.0 / STREAM_MOVES_PER_H,
                         "start_s": start,
                         # ★스트림 번호로 양하/적하를 가른다 — **`type_offset` 과 같은 식**.
                         #   한 배가 STS 를 2~6대 붙이고 일부는 내리고 일부는 싣는다(실제가 그렇다).
                         #   ⚠️ 2026-08-26 정정: 전에는 `(len(rows) + k) % 2` 였는데 두 값이
                         #   함께 1씩 늘어 합이 **항상 짝수**였다 → 전 스트림이 양하.
                         #   하루 무대는 이 칸을 안 읽고 `type_offset` 만 봐서 안 드러났지만,
                         #   30일 무대([[YR-239]])는 이 칸으로 배를 붙이므로 야드가 하루
                         #   **+5,000상자**씩 부풀었다 (실측: 19,656 → 24,828).
                         "work": "DISCHARGE" if k % 2 == 0 else "LOAD",
                         "type_offset": k % 2})
    # ★엔진의 '블록당 1 스트림' 계약(≤21). 정상상태로 뽑으면 동시 접안이 흔들려
    #   드물게 넘친다 — **늦게 온 배부터** 통째로 뺀다(스트림만 잘라내면 그 배의
    #   STS 대수가 선급 정의와 어긋난다). 조용히 자르지 않고 `dropped` 로 남긴다.
    dropped: list[str] = []
    while len(rows) > len(blocks):          # 설계상 최대 14 — 여기 걸리면 구성이 바뀐 것
        last = max({r["vessel_id"] for r in rows},
                   key=lambda vid: max(r["start_s"] for r in rows
                                       if r["vessel_id"] == vid))
        rows = [r for r in rows if r["vessel_id"] != last]
        dropped.append(last)
    for i, r in enumerate(rows):
        r["block"] = blocks[i]
        r["dropped_peers"] = tuple(dropped)
    return rows


def build_diurnal_v3(profile, seed: int, *, load: int,
                     obs: ObservationContract | None = None,
                     layout: YardLayout | None = None,
                     params: TerminalStreamParams | None = None,
                     drain_s: float = DIURNAL_DRAIN_S,
                     background_seed: int | None = None) -> dict:
    """하루 무대 — 구 `build_diurnal` 과 같은 형태를 내되 **본선만 v3 선급**이다.

    트럭 쪽(도착 명단·배분·규격·주행)은 구판과 **한 줄도 다르지 않다** — 본선을
    바꾸면서 트럭까지 흔들면 무엇이 결과를 바꿨는지 못 가린다.
    """
    obs = obs or OBS_24H
    layout = layout or terminal_layout()
    params = params or TerminalStreamParams(load_4h=load)
    bseed = seed if background_seed is None else background_seed

    # ── ① 본선 — v3 선급
    fleet = sample_day_vessels(bseed, obs=obs)
    streams = plan_streams(fleet, layout, bseed, obs=obs)
    by_block = {r["block"]: r for r in streams}

    scns = {}
    for i, b in enumerate(layout.ids):
        r = by_block.get(b)
        p_b = params if r is None else dataclasses.replace(
            params, vessel_moves=int(r["moves"]),
            sts_move_interval_s=float(r["cadence_s"]))
        bg = _background(profile, bseed + 1000 * (i + 1), b, obs, p_b,
                         1 if r else 0,
                         vessel_type_offset=(r["type_offset"] if r else 0),
                         drain_s=drain_s)
        if r:
            bg = _retime_vessels(bg, [r["start_s"]])
        scns[b] = bg

    # ── ② 반출 대상 후보 (배경 본선 사용분 제외) — 구판과 동일
    free: dict[str, list[str]] = {}
    for b, scn in scns.items():
        used = {j.target_container for j in scn.jobs if j.target_container}
        cand = sorted(set(scn.containers) - used)
        random.Random(f"v3:tgt:{bseed}:{b}").shuffle(cand)
        free[b] = cand

    # ── ③ 트럭 명단 — 구판과 동일 (본선만 바꾼다)
    p = distribution_vector(layout, params)
    times = diurnal_arrivals(seed, day_s=obs.observe_s, total=load)
    counts = allocate(p, load)
    slots = [b for b in layout.ids for _ in range(counts[b])]
    random.Random(f"h21d:assign:{seed}").shuffle(slots)
    flow_rng = random.Random(f"h21d:flow:{seed}")
    size_rng = random.Random(f"h21d:size:{seed}")
    exit_rng = random.Random(f"h21d:exit:{seed}")
    resid_rng = random.Random(f"h21d:resid:{seed}")
    fallbacks = {b: 0 for b in layout.ids}
    schedule = []
    for i, (t, bid) in enumerate(zip(times, slots)):
        sr = params.resid_travel_sigma_s
        resid = (max(-2 * sr, min(2 * sr, resid_rng.gauss(0.0, sr))) if sr > 0 else 0.0)
        want_out = flow_rng.random() < params.gate_out_share
        out = want_out and bool(free[bid])
        if want_out and not out:
            fallbacks[bid] += 1
        schedule.append({
            "job_id": f"{bid}:D-{i:05d}", "block": bid,
            "arrival_s": round(t, 3),
            "flow": "GATE_OUT" if out else "GATE_IN",
            "requested_flow": "GATE_OUT" if want_out else "GATE_IN",
            "fallback_reason": ("no_free_target" if (want_out and not out) else None),
            "target": free[bid].pop() if out else None,
            "size_ft40": size_rng.random() < params.size_mix_ft40,
            "travel_s": _clamp_travel(layout.gate_to_block_s(bid), resid),
            "travel_base_s": layout.gate_to_block_s(bid),
            "exit_travel_s": trunc_normal(exit_rng, params.exit_travel_mu_s,
                                          0.12, lo=60.0)})
        attach_bnct(schedule[-1])

    for b in layout.ids:
        scns[b] = dataclasses.replace(scns[b], meta={
            **scns[b].meta, "h21_block": b, "h21_mode": "diurnal_24h_v3",
            "h21_day_total": load, "observation": obs.as_dict()})

    return {"scenarios": scns, "schedule": schedule, "p": p, "counts": counts,
            "vessel_schedule": streams, "fleet": fleet,
            "block_vessel": {r["block"]: r for r in streams},
            "day_total": load, "drain_s": drain_s,
            "sim_end_s": obs.observe_s + drain_s,
            "flow_fallbacks": fallbacks,
            "flow_fallbacks_total": sum(fallbacks.values()),
            "observation": obs.as_dict(), "layout": layout.as_dict(),
            "mode": "diurnal_24h_v3"}


def structural_idle_krw(built: dict, end_s: float) -> float:
    """★**구조적 유휴** — 접안 시간 − 작업 시간. 진단 전용이고 Φ 에 안 들어간다.

    실측 Port Time(20.4~38.6h)은 순수 하역 시간이 아니다 — 정박 대기·조선·접이안·
    검역·준비·교대가 들어 있다. 그 차이가 유휴이고, 소형 34~45% · 대형 46~55% 다.

    ■ 왜 Φ 에 안 넣나 (2026-08-23 판단 · 02b §2 문구와 다르다)
      **정책이 못 바꾸는 상수**다. 재배치를 어떻게 하든 접이안·검역 시간은 그대로다.
      Φ 는 최적화 목표이므로 상수를 넣으면 개선 비율만 희석된다 — 반사실 차이에서는
      어차피 상쇄되어 **학습에는 아무 영향이 없다.**
      그래서 Φ 항4 는 **정책이 만든 유휴**(STS 가 야드 공급을 못 받아 멈춘 시간)만
      세고, 구조적 유휴는 여기서 따로 내 보고에만 쓴다.

    창 끝에서 검열한다 — 접안 구간과 `[0, end_s]` 이 겹친 만큼만.
    """
    from ..reward.krw import vessel_idle_krw
    tot = 0.0
    for v in built.get("fleet", []):
        idle = max(0.0, v.port_time_s - v.work_s())
        share = v.window_overlap_s(end_s) / max(1e-9, v.port_time_s)
        tot += vessel_idle_krw(v.gt, idle * share)
    return tot


def fleet_summary(built: dict) -> dict:
    """진단 — 그날 본선이 어떻게 생겼나."""
    obs_s = built["observation"]["observe_s"]
    f = built["fleet"]
    rows = built["vessel_schedule"]
    dropped = rows[0].get("dropped_peers", ()) if rows else ()
    return {"n_vessels": len(f), "n_streams": len(rows), "dropped": list(dropped),
            "classes": [v.cls.name for v in f],
            "has_large": any(v.cls.name == "LARGE" for v in f),
            "moves_total": sum(v.moves for v in f),
            "moves_in_window": sum(v.moves_in_window(obs_s) for v in f),
            "stream_moves_total": sum(r["moves"] for r in rows)}
