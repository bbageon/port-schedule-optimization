"""30일 연속 무대 — 세계가 밤에 끊기지 않는다 ([[YR-239]] · 사용자 지시 2026-08-26).

■ 왜 30일인가
  지금은 하루가 독립이라 **매일 아침 야드가 인공적인 상태**에서 시작하고, 밤에는
  도착이 0 인 배수 2시간이 붙는다. 그래서:

    · 하루 끝에 **가짜 빈 구간**이 생긴다 (B1 에서 악용은 없었지만 구조는 남는다)
    · Φ 네 항이 **다른 창**을 잰다 (A2 — 항1 은 24h 검열, 항2/3/4 는 26h 누적)
    · **본선이 하루를 넘긴다** — 중형 25~31h · 대형 31~47h 라 늘 "미완" 이다
      (그래서 `vessel_slack` 이 죽어 있었다 — A11)

  30일을 **한 시뮬로** 이으면 셋 다 사라진다.

■ 첫날과 마지막날은 버린다
      1일차   야드를 현실적인 상태로 **데운다**
      2~29일  ★학습·평가 28일
      30일차  꼬리 효과를 **흡수**한다

  첫날은 인공적인 초기 적재에서 출발하고, 마지막날은 뒤가 없어 경계 효과를 받는다.

■ 부하는 날마다 **가중 추첨** (사용자 결정 2026-08-26)
  현실처럼 보통날이 많고 초혼잡은 드물다. 학습 분포와 판정 분포를 굳이 같게 두지
  않는다 — **판정은 부하별로 따로** 낸다.

■ ★비용 벽과 그 해법
  반사실 분기가 터미널을 통째로 복제한다(`copy.deepcopy`). 실측:

      복제 ≈ 0.17초 + job 하나당 55μs   (부하 3,500→0.69s · 12,500→1.14s)

  30일이면 job 이 20만 건으로 쌓여 복제가 **11초**가 되고, 더 나쁜 것은
  `inside_count`·`announced_around` 가 **블록의 모든 job 을 훑는다**는 점이다 —
  특징 계산이 20배 느려진다.

  → **하루가 끝나면 완료된 job 을 치운다.** 완료분을 읽는 곳은 없다:
    `inside_count` 는 `gate_out > t` 만, `announced_around` 는 미래 예정만,
    계수기는 `sim.kpis` 에 따로 누적되고, **Φ 는 `records` 에서 온다.**

  ⚠️ 엔진 사본(`v3/world/integrated/engine.py`)은 **한 줄도 안 고친다**
     (사용자 지시). 정리는 이 모듈이 밖에서 한다.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

DAY_S = 86_400.0

#: 부하 가중 추첨 — (부하, 가중, 이름). 합이 1 이다.
LOAD_WEIGHTS: tuple[tuple[int, float, str], ...] = (
    (3_500, 0.30, "원활"),
    (5_000, 0.30, "보통"),
    (7_500, 0.25, "혼잡시작"),
    (12_500, 0.10, "혼잡"),
    (15_000, 0.05, "초혼잡"),
)
#: ★30일 무대의 **초기 장치율** — 하루 무대(0.65)보다 낮다.
#:
#: 블록 용량은 1,440슬롯(24 bay x 10 row x 6 tier)이다. 0.65 면 936상자로 시작하는데,
#: 양하 스트림 하나가 그 블록에 **하루 386상자**를 쌓는다:
#:
#:     936 → 1,322 (92%)  … 그 블록이 이튿날도 양하면 **슬롯이 없어 멈춘다**
#:
#: 실측(2026-08-26 · 부하 300): 1일차 끝에 양하 블록 다섯이 1,323~1,337상자(92%)로
#: 차고 **막힌 양하 job 93건**, 2일차 393건, 3일차 570건으로 불었다. 빈 블록(528상자)은
#: 막힌 게 **0** 이었다 — 원인이 용량이라는 직접 증거다.
#:
#: 0.45 = 648상자로 시작하면 양하 하루 뒤가 1,034(72%)라 여유가 남는다.
MONTH_FILL_RATIO = 0.45

#: 기본 길이 — 첫날·마지막날은 연결용이라 학습은 28일이다.
N_DAYS = 30
WARMUP_DAYS = 1
COOLDOWN_DAYS = 1


@dataclass(frozen=True)
class DayPlan:
    """하루 계획 한 줄."""

    index: int
    load: int
    label: str
    seed: int
    t0: float                      # 이 날의 시작 시각 (초, 월 기준)
    n_days: int = N_DAYS           # 이 달이 며칠짜리인가 (연결용 날 판정에 쓴다)

    @property
    def t1(self) -> float:
        return self.t0 + DAY_S

    @property
    def is_train(self) -> bool:
        """학습·평가에 쓰는 날인가 — 첫날·마지막날은 **연결용**이다."""
        return WARMUP_DAYS <= self.index < self.n_days - COOLDOWN_DAYS

    def as_dict(self) -> dict:
        return {"index": self.index, "load": self.load, "label": self.label,
                "seed": self.seed, "t0": self.t0, "train": self.is_train,
                "n_days": self.n_days}


def plan_month(seed: int, *, n_days: int = N_DAYS,
               weights=LOAD_WEIGHTS) -> list[DayPlan]:
    """30일 계획 — 날마다 부하를 **가중 추첨**한다.

    난수는 시드에서만 나온다(`random.Random(문자열)`). 같은 시드면 같은 달이다 —
    짝비교가 성립해야 하므로 런타임 무작위를 쓰지 않는다.
    """
    rng = random.Random(f"v3:month:{seed}")
    loads = [w[0] for w in weights]
    ws = [w[1] for w in weights]
    names = {w[0]: w[2] for w in weights}
    out = []
    for i in range(n_days):
        load = rng.choices(loads, weights=ws, k=1)[0]
        out.append(DayPlan(index=i, load=load, label=names[load],
                           seed=seed + 1_000 * (i + 1), t0=i * DAY_S,
                           n_days=n_days))
    return out


def summarize(days: list[DayPlan]) -> dict:
    """계획 요약 — 학습에 쓰는 날만 따로 센다."""
    tr = [d for d in days if d.is_train]
    cnt: dict[str, int] = {}
    for d in tr:
        cnt[d.label] = cnt.get(d.label, 0) + 1
    return {"n_days": len(days), "n_train": len(tr),
            "trucks_total": sum(d.load for d in days),
            "trucks_train": sum(d.load for d in tr),
            "mean_load_train": (sum(d.load for d in tr) / max(1, len(tr))),
            "by_label": cnt}


# ─────────────────────────────────────────────────────────── 완료분 정리
def prune_completed(mbt, t: float) -> dict:
    """★하루가 끝나면 **끝난 것을 치운다** — 30일을 굴리는 유일한 방법.

    치우는 조건은 하나다: **끝났다**. 트럭은 게이트를 나갔고(`gate_out ≤ t`),
    본선 작업은 `DONE` 이다. 그 뒤로는 어떤 특징도 그 job 을 안 본다.

    ■ 네 곳을 **함께** 치운다 — 하나라도 빠지면 불변식이 깨진다
      ① `sim.jobs`              — `announced_around`·`inside_count` 가 여기를 훑는다
      ② `mbt.ledger.records`    — `check_invariants` 가 ①과 1:1 을 요구한다
      ③ `time_ledger.records`   — `bridge._sync` 가 **매 epoch 전체를 훑는다**
      ④ `time_ledger._a_sorted` — 이미 지나간 앞부분 (`_a_idx` 도 함께 되돌린다)

      ③이 진짜 벽이다. 30일이면 트럭이 20만 대라 60초마다 20만 줄을 훑게 되고,
      그것만으로 86억 번이다. ④는 삽입이 O(n) memmove 라 길이가 곧 비용이다.

    ⚠️ 적분 상태(`_in_block`·`_o_heap`·`_n_inside`·`terminal_area_s`)는 **안 건드린다** —
      그 값들은 `records` 와 독립이라 잘라도 계산이 안 바뀐다.

    ⚠️ Φ 의 원료인 **v3 기록(`records`)은 안 지운다** — 호출부가 날짜별로 갈라 둔다.
    """
    from ..world.domain.enums import JobStatus

    n_jobs = n_led = n_tl = n_a = 0
    for bid, sim in mbt.blocks.items():
        tl = getattr(sim, "time_ledger", None)
        gone = []
        for jid, j in sim.jobs.items():
            if j.status != JobStatus.DONE:
                continue
            r = None if tl is None else tl.records.get(jid)
            if r is not None:                      # 외부 트럭 — 게이트를 나가야 끝이다
                if r.gate_out is None or r.gate_out > t:
                    continue
            gone.append(jid)
        for jid in gone:
            sim.jobs.pop(jid, None)
            mbt.ledger.records.pop(jid, None)
            if tl is not None and tl.records.pop(jid, None) is not None:
                n_tl += 1
        n_jobs += len(gone)
        n_led += len(gone)
        if tl is not None and tl._a_idx > 0:       # ④ 지나간 A 시각 앞부분
            n_a += tl._a_idx
            del tl._a_sorted[:tl._a_idx]
            tl._a_idx = 0
    return {"jobs": n_jobs, "ledger": n_led, "time_ledger": n_tl, "a_sorted": n_a}


def job_count(mbt) -> int:
    """지금 터미널이 들고 있는 job 수 — 정리가 듣는지 보는 눈금."""
    return sum(len(s.jobs) for s in mbt.blocks.values())


def ledger_load(mbt) -> dict:
    """장부가 얼마나 무거운가 — 정리가 안 들으면 여기가 부풀어 오른다."""
    tl = [s.time_ledger for s in mbt.blocks.values()
          if getattr(s, "time_ledger", None) is not None]
    return {"jobs": job_count(mbt), "ledger": len(mbt.ledger.records),
            "time_ledger": sum(len(t.records) for t in tl),
            "a_sorted": sum(len(t._a_sorted) for t in tl),
            "vessels": sum(len(s.vessels) for s in mbt.blocks.values()),
            # ★야드 재고 — 30일 동안 **표류하는가**. 들어오는 양과 나가는 양이
            #   안 맞으면 야드가 차거나 비고, 그러면 무대가 현실을 안 닮는다.
            "boxes": sum(len(s.stacks.containers) for s in mbt.blocks.values())}


# ─────────────────────────────────────────────────────── 30일 무대 조립
def build_month(seed: int, *, days=None, profile=None, layout=None,
                lead_mode: str = "DIST", n_days: int = N_DAYS,
                fill_ratio: float = MONTH_FILL_RATIO) -> dict:
    """30일치 **도착 명단**을 이어 붙인다.

    ■ 어떻게 잇나
      하루 생성기(`build_stage`)를 날마다 부르고, 그 날의 도착·통지 시각에
      `day.t0` 를 더한다. 도착 곡선 자체는 **하루 주기**로 정의돼 있어
      (`diurnal_rate` 의 `h = t/3600`) 그대로 30일을 못 만든다 — 그래서
      날마다 만들어 옮겨 붙인다.

    ■ 무엇이 하루치만 있어야 하나
      **초기 적재**는 첫날 것만 쓴다. 둘째 날부터는 전날이 남긴 야드가 출발점이다.
      본선은 §본선 참조 — 날을 넘어가므로 따로 배치한다.

    돌려주는 것:
      `days`      : 날 계획 (부하·라벨·시드)
      `schedule`  : 30일치 도착 명단 (시각이 이미 월 기준으로 옮겨져 있다)
      `day0`      : 첫날 `build_stage` 결과 — 시나리오(초기 적재)를 여기서 가져온다
    """
    from .orders import build_stage
    from ..world.integrated.profiles import build_h21_profile
    from ..world.integrated.yard_layout import terminal_layout

    profile = profile or build_h21_profile()
    layout = layout or terminal_layout()
    days = days or plan_month(seed, n_days=n_days)

    day0, schedule = None, []
    for d in days:
        built = build_stage(load=d.load, seed=d.seed, profile=profile,
                            layout=layout, lead_mode=lead_mode,
                            fill_ratio=fill_ratio)
        if day0 is None:
            day0 = built
        for e in built["schedule"]:
            e = dict(e)
            e["job_id"] = f"D{d.index:02d}-{e['job_id']}"   # 날마다 고유하게
            e["arrival_s"] = float(e["arrival_s"]) + d.t0
            e["day"] = d.index
            schedule.append(e)
    schedule.sort(key=lambda e: e["arrival_s"])
    return {"days": days, "schedule": schedule, "day0": day0,
            "month_end_s": n_days * DAY_S, "lead_mode": lead_mode}


# ─────────────────────────────────────────────────────────────── 30일 본선
def plan_month_vessels(days, layout, *, obs=None, truck_net=None) -> dict:
    """날마다 배를 뽑아 **월 시각으로 옮긴다** ([[YR-212]] 선급 3종 그대로).

    하루 무대와 같은 함수(`sample_day_vessels`·`plan_streams`)를 날마다 부르고,
    접안 시작에 `day.t0` 를 더한다. 배 이름에도 날짜를 박아 30일 내내 유일하게 한다.

    ■ ★양하/적하는 **블록별 수지**로 정한다 (2026-08-26 실측 사고)
      `plan_streams` 의 배정(스트림 번호 홀짝)은 **하루 안에서만** 균형이다. 30일이면
      한 블록이 사흘 내리 양하를 맡을 수 있고, 그러면 블록이 용량(1,440슬롯)에 닿아
      **양하가 멈춘다.** 그래서 여기서 다시 정한다 — *"목표에 못 미치는 블록은 내리고,
      넘은 블록은 싣는다."* 수지는 **계획에서만** 계산하므로 정책과 무관하고 시드가
      같으면 같다.

      ★목표는 0 이 아니라 **트럭이 빼 간 만큼**이다(`truck_net`). 트럭은 반출이 60%라
      야드에서 하루 1,000상자를 빼 간다 — 목표를 0 으로 두면 30일이면 야드가 비고,
      그건 현실이 아니다(실제 터미널은 그 구멍을 본선 양하가 메운다).

    ★이 배들은 **무대를 세울 때 안 붙인다** — 그날 아침에 붙인다
    (`month_engine.inject_vessel`). 이유는 그 모듈 머리말에 있다: 5일차 배가 실어
    갈 상자는 t=0 야드에 없기 때문이다.

    돌려주는 것: `{날짜: [스트림 행, ...]}` — 행은 `plan_streams` 와 같은 꼴에
    `key`(스트림 고유 이름)·`ship`(배 이름)·`day` 를 더한 것이다.
    """
    from ..world.integrated.terminal_stream import OBS_24H
    from .vessels import plan_streams, sample_day_vessels

    obs = obs or OBS_24H
    # ★블록별 수지 장부 — 양하로 쌓인 만큼 다음엔 적하를 준다.
    #   첫날은 블록 순번으로 반씩 갈라 시작한다(전부 양하로 몰리지 않게).
    net = {b: (1 if i % 2 else -1) for i, b in enumerate(layout.ids)}
    #: ★목표 수지 — **트럭이 빼 간 만큼 배가 채워야 한다** (`truck_net_by_block` 머리말).
    #   목표가 0 이면 야드가 하루 1,000상자씩 빠져 12일이면 텅 빈다.
    goal = {b: 0 for b in layout.ids}
    out: dict[int, list[dict]] = {}
    for d in days:
        for b, v in (truck_net or {}).get(d.index, {}).items():
            goal[b] = goal.get(b, 0) + v                  # 반출 초과분만큼 목표를 올린다
        fleet = sample_day_vessels(d.seed, obs=obs)
        rows = []
        for r in sorted(plan_streams(fleet, layout, d.seed, obs=obs),
                        key=lambda x: x["block"]):        # 결정론 — 블록 순서 고정
            r = dict(r)
            b = r["block"]
            dis = net.get(b, 0) < goal.get(b, 0)          # 목표에 못 미치면 내린다
            r["work"] = "DISCHARGE" if dis else "LOAD"
            r["type_offset"] = 0 if dis else 1            # 두 칸이 같은 말을 하게
            net[b] = net.get(b, 0) + (r["moves"] if dis else -r["moves"])
            ship = f"D{d.index:02d}-{r['vessel_id']}"
            r.update(ship=ship, day=d.index,
                     key=f"{ship}-s{r['stream']}",
                     start_s=float(r["start_s"]) + d.t0)
            rows.append(r)
        out[d.index] = rows
    return out


def truck_net_by_block(schedule) -> dict:
    """날·블록별 **트럭 수지** = 반출 − 반입 (양수 = 야드에서 빠져나간다).

    이게 왜 필요한가 — `gate_out_share = 0.6` 이라 트럭은 **가져가는 쪽이 많다.**
    부하 5,000 이면 반입 2,000 · 반출 3,000 으로 **하루 −1,000상자**다.

        하루 무대   야드가 아침마다 리셋되므로 −1,000 은 5% 오차로 묻힌다
        30일 무대   30일이면 −30,000 인데 야드는 12,000상자다 → **12일이면 텅 빈다**

    현실 터미널은 이 구멍을 **본선 양하**가 메운다(수입 화물이 배에서 내려 트럭으로
    나간다). 그래서 배의 양하/적하 비율을 0 이 아니라 **이 수지에 맞춘다**.

    명단에서만 센다 — 정책과 무관하고 시드가 같으면 같다.
    """
    out: dict[int, dict[str, int]] = {}
    for e in schedule:
        d = out.setdefault(int(e.get("day", 0)), {})
        b = e["block"]
        d[b] = d.get(b, 0) + (1 if e["flow"] == "GATE_OUT" else -1)
    return out


def vessel_meta(rows) -> dict:
    """스트림 행 → `{스트림 이름: (배 이름, GT, STS 대수)}`. 유휴를 배 단위로 묶는 표."""
    return {r["key"]: (r["ship"], float(r["gt"]), int(r["sts"])) for r in rows}


def month_vessel_idle(mbt, meta: dict, archive: dict | None = None) -> dict:
    """항 4 의 원료 — `{배: (GT, 유휴 초)}`. **배 단위**로 묶는다.

    하루 무대의 `episode.vessel_idle_of` 와 같은 일을 하되 두 가지가 다르다:

      ① 묶는 열쇠가 **블록이 아니라 스트림 이름**이다. 30일이면 한 블록에 날마다
         다른 배가 오므로 블록으로 묶으면 어제 배의 유휴가 오늘 배에 붙는다.
      ② **떠난 배**(`archive`)를 함께 센다. 끝난 배는 `sim.vessels` 에서 치우는데
         (`_advance` 가 매 적분마다 전량을 훑어 30일이면 420척이 된다), 치우기 전에
         적립값을 여기 옮겨 둔다.

    선박 비용은 배 1척·1시간 단위라 스트림 합을 STS 대수로 나눈다 — 대형선 STS 6대가
    1시간씩 막혔다고 배가 6시간 논 게 아니다.
    """
    acc: dict[str, list[float]] = {}
    gt_of: dict[str, float] = {}
    sts_of: dict[str, int] = {}

    def add(key: str, wait: float) -> None:
        m = meta.get(key)
        if m is None:
            return                              # 표에 없는 배 — 셀 근거가 없다
        ship, gt, sts = m
        acc.setdefault(ship, []).append(float(wait))
        gt_of[ship], sts_of[ship] = gt, sts

    for key, wait in (archive or {}).items():
        add(key, wait)
    for sim in mbt.blocks.values():
        for key, v in getattr(sim, "vessels", {}).items():
            if key in (archive or {}):
                continue                        # 이미 적립했다 — 두 번 세지 않는다
            add(key, getattr(v, "sts_wait_accum_s", 0.0))
    return {k: (gt_of[k], sum(w) / max(1, sts_of[k])) for k, w in acc.items()}


#: 배를 치우기 전에 두는 여유(초). 마지막 이송이 도착할 시간을 준다.
RETIRE_LAG_S = 3600.0


def retire_done_vessels(mbt, archive: dict, t: float | None = None,
                        lag_s: float = RETIRE_LAG_S) -> int:
    """끝난 배를 **적립하고 치운다** — `_advance` 가 매 적분마다 전량을 훑기 때문이다.

    30일이면 420 스트림이 쌓인다. 끝난 배는 더 이상 막히지 않으므로 유휴 적립값만
    옮겨 두면 손실이 없다.

    ■ ★언제 치워도 되는가 — 조건 셋을 **모두** 넘어야 한다 (2026-08-26 사고)
      ① `v.done`               STS 가 마지막 move 를 끝냈다
      ② **그 배 앞으로 남은 job 이 하나도 없다**
      ③ 끝난 지 `lag_s` 가 지났다 (t 를 주면)

      ②가 없으면 무너진다. 양하 job 은 **시각이 아니라 박스의 물리 도착**이 푼다
      (`engine._release_next_discharge` — 이송이 닿을 때마다 한 건씩). 그런데 STS 는
      이송보다 훨씬 빠르므로, `v.done` 시점에 **아직 안 풀린 job 이 수백 건** 남아
      있을 수 있다. 그때 배를 치우면 `_transfer_arrive` 가 그 배를 못 찾아
      **남은 job 이 영원히 PLANNED 로 남는다.**

      실측 (부하 300 · 4일): 남은 job 131 → 445 → 642 → **1,566**,
      Φ 1,568만 → **5,812만원**, 하루 소요 72초 → **325초**. 세계가 서서히 막혔다.

      ③은 마지막 이송이 공중에 떠 있는 사이를 막는다 — 그 사이에 치우면
      `_transfer_arrive` 의 `self.vessels[vid]` 가 KeyError 다.

    ⚠️ **`prune_completed` 를 먼저 부르고 이걸 부른다** — 끝난 job 이 치워져야
       ②가 참이 된다.
    """
    n = 0
    for sim in mbt.blocks.values():
        live = {j.vessel_id for j in sim.jobs.values()
                if getattr(j, "vessel_id", None) is not None}
        gone = []
        for key, v in sim.vessels.items():
            if not v.done or key in live:
                continue
            at = getattr(getattr(v, "truth", None), "actual_completion_s", None)
            if t is not None and at is not None and (t - at) < lag_s:
                continue
            gone.append(key)
        for key in gone:
            archive[key] = float(getattr(sim.vessels[key], "sts_wait_accum_s", 0.0))
            sim.vessels.pop(key)
            n += 1
        if gone:
            sim._refresh_rates()
    return n


def make_retarget(seed: int):
    """★반출 대상을 **투입 시각에** 다시 고르는 고름기 ([[YR-239]]).

    왜 필요한지는 `orders.V3Announcer.review` 안의 주석에 있다 — 한 줄로 줄이면
    *"30일이면 컨테이너 이름이 날마다 겹치고 초기 적재는 유한하다"* 이다.

    ■ 규칙
      ① 명단이 찍은 상자가 **아직 야드에 있고 아무도 안 찍었으면** 그대로 쓴다
         (첫날은 하루 무대와 **한 상자도 안 다르게** 굴러야 한다)
      ② 아니면 그 블록에서 **아무도 안 찍은 상자**를 하나 고른다
      ③ 그것도 없으면 `None` — 호출부가 조용히 넘기지 않고 `NO_TARGET` 으로 남긴다

    난수는 **오더 이름에서만** 나온다. 그래야 반사실 분기 세계가 같은 야드에서
    같은 상자를 고른다 — 안 그러면 사실·대안이 다른 트럭을 굴려 라벨이 오염된다.
    """
    import random as _r

    def pick(mbt, bid: str, e: dict):
        sim = mbt.blocks.get(bid)
        if sim is None:
            return None
        taken = {j.target_container for j in sim.jobs.values()
                 if getattr(j, "target_container", None) is not None}
        want = e.get("target")
        if want is not None and want in sim.stacks.containers and want not in taken:
            return want
        cand = [c for c in sorted(sim.stacks.containers) if c not in taken]
        if not cand:
            return None
        return cand[_r.Random(f"v3:month:{seed}:tgt:{e['job_id']}").randrange(len(cand))]

    return pick
