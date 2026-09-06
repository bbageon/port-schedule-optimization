"""무대를 세우고 **오더·기록**을 만든다.

설계 정본: `.claude/docs/architecture/01-오더-스키마.md` · `02-무대.md` §2-1·§4

■ v2 와 무엇이 다른가
  ① **부하가 축이다** — 3,500·5,000·7,500 을 같은 코드로 세운다
  ② **통지 리드타임이 트럭마다 다르다** — `sample_lead_s` 분포에서 뽑는다
     (v2 는 전원 30분 고정이라 "누구를 더 일찍 알았나" 라는 축이 없었다)

■ docKey 에 블록 접두를 붙이지 않는다
  `build_diurnal` 은 `Y07:D-00042` 처럼 블록을 앞에 붙인다. v3 스키마는 이를
  금지한다(01 §2) — 블록은 **바뀌는 값**이라 정체성에 섞으면 재배치가 곧 개명이
  된다. 그래서 무대를 세울 때 **엔진 쪽 id 를 접두 없는 형태로 바꿔** 둘을 일치
  시킨다. 인덱스가 전역이라 접두를 떼도 유일하다.
"""
from __future__ import annotations

import dataclasses
import random

from ..world.integrated.multiblock import TransferError
from ..schema import ExecutionRecord, Order
from ..world.integrated.multiblock import TransferError as _TE  # noqa: F401
from ..world.integrated.terminal_stream import (OBS_24H, TerminalStreamParams,
                                                _job_from_entry, on_grid,
                                                sample_lead_s)
from ..world.integrated.yard_layout import terminal_layout
from .vessels import build_diurnal_v3

#: 투입 검토 격자(초) — 엔진의 review epoch 격자와 같아야 한다.
EPOCH_S = 60.0


def build_stage(*, load: int, seed: int, profile, layout=None, obs=None,
                lead_mode: str = "DIST", fill_ratio: float | None = None) -> dict:
    """하루치 무대를 세운다. `lead_mode` 가 통지 리드타임 축을 가른다.

    | `lead_mode` | 뜻 |
    |---|---|
    | `"DIST"` | 실측 분포에서 트럭마다 뽑는다 (v3 기본) |
    | `"FIXED"` | 전원 30분 고정 (v2 재현 — [[YR-190]] 의 대조 팔) |

    반환은 `build_diurnal` 의 결과에 `lead_s` 를 붙인 것이다. 리드는 **시드에서
    미리 뽑는다** — 런타임 무작위를 쓰면 같은 시드가 같은 하루를 못 만든다.
    """
    if lead_mode not in ("DIST", "FIXED"):
        raise ValueError(f"lead_mode 는 DIST|FIXED — {lead_mode!r}")
    obs = obs or OBS_24H
    layout = layout or terminal_layout()
    prm = TerminalStreamParams(load_4h=load)
    if fill_ratio is not None:
        # ★초기 장치율을 낮춘다 — 30일 무대 전용 ([[YR-239]]).
        #   하루 무대는 0.65(=블록당 936/1440)로 충분하다. 30일은 다르다:
        #   양하 스트림 하나가 블록에 **하루 386상자**를 쌓으므로 936 → 1,322(92%) 가
        #   되고, 그 블록이 이튿날도 양하면 **슬롯이 없어 양하가 멈춘다**(실측).
        prm = dataclasses.replace(prm, fill_ratio=float(fill_ratio))
    built = build_diurnal_v3(profile, seed, load=load, obs=obs, layout=layout,
                             params=prm, background_seed=seed)

    rng = random.Random(f"v3:lead:{seed}:{load}:{lead_mode}")
    for e in built["schedule"]:
        # ★블록 접두 제거 — 엔진 id 와 docKey 를 같은 문자열로 만든다.
        e["job_id"] = e["job_id"].split(":")[-1]
        lead = 1800.0 if lead_mode == "FIXED" else sample_lead_s(rng.random())
        # 통지가 창 시작보다 앞설 수는 없다 — 0 으로 눌러 담는다(02 §4 음수 리드).
        e["lead_s"] = min(float(lead), float(e["arrival_s"]))
    built["lead_mode"] = lead_mode
    return built


def orders_from_schedule(built: dict) -> tuple[dict[str, Order], dict[str, ExecutionRecord]]:
    """명단 → `Order` 6필드 + 빈 `ExecutionRecord`.

    ★기록은 **비어서** 시작한다. 터미널이 사건을 보내야 채워진다 — 무대를 세우는
    시점에 이미 아는 값(도착 시각 등)을 기록에 미리 적으면 정책이 미래를 읽는다.
    """
    orders: dict[str, Order] = {}
    records: dict[str, ExecutionRecord] = {}
    for e in built["schedule"]:
        dk = e["job_id"]
        notice = e["arrival_s"] - e["lead_s"]
        o = Order(doc_key=dk,
                  in_out=(0 if e["flow"] == "GATE_OUT" else 1),
                  copino_notice_s=round(notice, 3),
                  in_out_reserve_s=round(e["arrival_s"], 3),
                  con_loc=e["block"],
                  con_no=str(e.get("con_no") or f"CN{dk}"))
        orders[dk] = o
        records[dk] = ExecutionRecord(doc_key=dk, copino_notice_s=o.copino_notice_s)
    return orders, records


class V3Announcer:
    """명단대로 투입한다 — **트럭마다 자기 통지 시각**에.

    v2 의 `ScheduledAnnouncer` 는 리드가 하나(30분 고정)라 통지 시각이 도착에서
    일정 간격이었다. 여기서는 트럭마다 다르므로 통지 epoch 도 제각각이다.

    개방 루프인 것은 그대로다 — **정책이 무엇을 하든 같은 트럭이 같은 시각에 온다.**
    그래야 두 정책의 짝비교가 성립한다.
    """

    def __init__(self, schedule: list[dict], *, end_s: float | None = None,
                 period_s: float = EPOCH_S, retarget=None):
        self.period_s = float(period_s)
        self.end_s = end_s
        #: ★반출 대상을 **투입 시각에** 다시 고르는 훅 ([[YR-239]]). None = 명단 그대로.
        #: 30일 무대에서는 필수다 — 아래 `review` 머리말 참조.
        self.retarget = retarget
        self.n_retargeted = 0
        self.by_epoch: dict[float, list[dict]] = {}
        for e in schedule:
            notice = max(0.0, e["arrival_s"] - e["lead_s"])
            slot = round((notice // self.period_s) * self.period_s, 6)
            self.by_epoch.setdefault(slot, []).append(e)
        self.n_admitted = 0
        self.n_skipped = 0
        self.skips: list[dict] = []

    def clone_fresh(self) -> V3Announcer:
        """반사실 분기용 사본 — 같은 명단, **자기 계수기**.

        분기 세계가 원본의 계수기를 같이 올리면 투입 검사(`admitted == 전건`)가
        거짓으로 부푼다. 실제로 측정 중에 한 번 밟았던 함정이다.
        """
        c = V3Announcer.__new__(V3Announcer)
        c.period_s, c.end_s, c.by_epoch = self.period_s, self.end_s, self.by_epoch
        c.retarget, c.n_retargeted = self.retarget, 0
        c.n_admitted, c.n_skipped, c.skips = 0, 0, []
        return c

    def window(self, lo: float, hi: float) -> "V3Announcer":
        """★그 구간의 명단만 담은 사본 — **작업자에게 보낼 짐을 줄인다** ([[YR-239]]).

        30일 무대는 명단이 20만 건이라 `_Ctx` 를 그대로 절이면(pickle) 작업자 하나당
        수십 MB 다. 반사실 분기는 `[t, t+H]` 밖의 도착을 **볼 일이 없으므로**
        그날 것만 잘라 보낸다.

        ⚠️ 자른 구간 밖에서 이 사본을 쓰면 트럭이 안 온다 — **분기 전용**이다.
        """
        c = V3Announcer.__new__(V3Announcer)
        c.period_s, c.end_s = self.period_s, self.end_s
        c.by_epoch = {k: v for k, v in self.by_epoch.items() if lo <= k <= hi}
        c.retarget, c.n_retargeted = self.retarget, 0
        c.n_admitted, c.n_skipped, c.skips = 0, 0, []
        return c

    def review(self, mbt, t: float) -> None:
        if not on_grid(t, self.period_s):
            return
        for e in self.by_epoch.get(round(t, 6), []):
            arr = e["arrival_s"]
            if self.end_s is not None and arr + e["travel_s"] > self.end_s:
                self.n_skipped += 1
                self.skips.append({"t": t, "job_id": e["job_id"], "reason": "TAIL"})
                continue
            if self.retarget is not None and e["flow"] == "GATE_OUT":
                # ★반출 대상을 **투입 시각에** 다시 고른다 ([[YR-239]]).
                #   하루 무대는 대상을 무대 세울 때 정해도 된다 — 그 상자가 t=0 야드에
                #   있고 하루 안에 아무도 안 건드리기 때문이다. 30일은 다르다:
                #     · 컨테이너 이름이 **날마다 겹친다**(`C0123` 이 매일 다시 나온다)
                #     · 초기 적재는 유한한데 반출은 30일 동안 계속된다
                #   ⇒ 5일차 트럭이 찍은 상자를 2일차 트럭이 이미 가져가 버린다.
                #   그러면 `admit_external_job` 이 "반출 대상 부재" 로 거절하고
                #   **트럭이 조용히 사라진다**(부하가 저절로 줄어든다).
                tgt = self.retarget(mbt, e["block"], e)
                if tgt is None:
                    self.n_skipped += 1
                    self.skips.append({"t": t, "job_id": e["job_id"],
                                       "reason": "NO_TARGET"})
                    continue
                if tgt != e.get("target"):
                    e = {**e, "target": tgt}       # ★원본을 안 건드린다 — 분기와 공유한다
                    self.n_retargeted += 1
            job = _job_from_entry(e, arr)
            try:
                mbt.admit_external_job(e["block"], job, gate_in_s=arr,
                                       travel_s=e["travel_s"])
                self.n_admitted += 1
            except TransferError as ex:
                self.n_skipped += 1
                self.skips.append({"t": t, "job_id": e["job_id"], "reason": str(ex)})
