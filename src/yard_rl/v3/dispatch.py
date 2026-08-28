"""크레인 바닥 — 고전 배차 규칙 5종 ([[YR-243]]).

■ 왜 필요한가 — **일반성 축**이다
  v3 판정은 지금까지 배차 하나(`SF_SPT`) 위에서만 돌았다([[YR-242]]). 그래서
  *"어떤 배차에서든 재배치가 이득"* 이라는 부 주장을 세울 수 없다. 바닥을 여럿
  깔고 각 바닥에서 재배치 이득을 재야 그 주장이 선다.

■ ★왜 `v3/world/` 가 아니라 여기인가
  `v3/world/` 는 원본의 **사본**이고 `tests/v3/test_world_clone.py` 가
  *"원본 없는 사본"* 을 잡는다. 새 파일을 거기 두면 시험이 깨진다. 그래서 바닥은
  사본 **밖**에 두고 `BaselinePreference` 만 가져다 쓴다(사본은 안 건드린다).

■ ★공통 고정 — "무엇을 할까" 는 바닥마다 안 바꾼다
  고전 규칙은 *"어느 일감 먼저"* 만 답한다. 우리 크레인은 *"일감을 할까 · 미리
  파낼까 · 옮길까 · 쉴까"* 도 답해야 하는데 여기엔 문헌 대응물이 없다.
  그래서 **실작업(SERVE) 우선**을 다섯 바닥에 공통으로 고정하고, 그 안의 **순서
  규칙만** 바꾼다. 그러면 바뀌는 변수가 하나라 비교가 깨끗하다.

  고정하지 않으면 퇴화한다 — 순수 SPT 는 60초짜리 REPOSITION 이 300초짜리
  SERVE 를 이겨서 크레인이 일은 안 하고 자리이동만 고른다([[YR-039]] 무효 사유:
  결정의 54~81%가 REPOSITION 이었다).

■ 정보 경계 — 다섯 바닥 전부 **공개 정보만** 쓴다
  `release_time`(선택 가능해진 시각)·`provided_eta`(제공 ETA)·`duration_s`·
  크레인 현재 위치·누적 대기만 본다. **실현 도착(`actual_block_arrival`)은 한 줄도
  안 읽는다** — 기존 `FIFOPreference` 가 그걸 읽어 미래를 알던 누출을 반복하지
  않는다([[YR-107]]). 그래서 다섯 다 `USES_FUTURE_INFORMATION = False` 다.

■ 현행 `SF_SPT` 의 실체 (실측 2026-08-28 · [[YR-242]] §2)
  정렬키가 다섯 자리지만 실제로 승부를 가르는 건 **소요시간 61.6% · 이름순 33.6%**
  뿐이고 `본선 우선` 은 **0%** 다. 즉 사실상 순수 SPT 다. 그래서 여기에 "순수 SPT"
  를 따로 두지 않는다 — 같은 점이 둘이 된다.
"""
from __future__ import annotations

import hashlib

from .world.contract.schema import CandidateKind
from .world.integrated.resolver import BaselinePreference

#: WAIT 자리 — `BaselinePreference` 규약과 같다(항상 최하위).
_WAIT_KEY: tuple = (2, 0.0, "")


class RuleBase(BaselinePreference):
    """바닥 공통 골격 — 실작업 우선 → `key()` → job_id.

    하위 클래스는 `key(sim, crane_id, gc, ref)` 하나만 채운다. 작을수록 먼저다.
    끝에 `job_id` 를 두는 것은 **결정론**을 위해서다(같은 입력이면 같은 배정).
    """

    USES_FUTURE_INFORMATION = False
    name = "RULE"

    def key(self, sim, crane_id, gc, ref) -> float:
        raise NotImplementedError

    def rank(self, sim, crane_id, gc) -> tuple:
        ref = gc.job_ref
        if ref is None:
            return _WAIT_KEY
        serve = 0 if gc.kind == CandidateKind.SERVE else 1
        return (serve, float(self.key(sim, crane_id, gc, ref)), ref.job_id)


def _notice_s(sim, ref) -> float:
    """**공개** 통지 시각 — 이 일감이 정책에 보이기 시작한 때.

    실현 도착이 아니다. `release_time` 이 없으면 제공 ETA 로 물러선다.
    """
    j = sim.jobs.get(ref.job_id)
    if j is None:
        return 0.0
    v = getattr(j, "release_time", None)
    if v is None:
        v = getattr(j, "provided_eta", None)
    return float(v) if v is not None else 0.0


class FIFO(RuleBase):
    """선착순 — 먼저 통지된 일감부터. 가장 흔한 대조군."""

    name = "FIFO"

    def key(self, sim, crane_id, gc, ref) -> float:
        return _notice_s(sim, ref)


class LIFO(RuleBase):
    """후착순 — 나중 통지된 일감부터. 스택 규칙(선착순의 반대편)."""

    name = "LIFO"

    def key(self, sim, crane_id, gc, ref) -> float:
        return -_notice_s(sim, ref)


class SPT(RuleBase):
    """최단 작업 우선 — 소요시간이 짧은 것부터.

    현행 `SF_SPT` 와 사실상 같은 바닥이다(위 머리말). **대조 기준점**으로 둔다 —
    새 바닥들이 옛 바닥과 얼마나 다른지 재려면 옛 바닥도 같은 틀에 있어야 한다.
    """

    name = "SPT"

    def key(self, sim, crane_id, gc, ref) -> float:
        return gc.plan.duration_s if gc.plan is not None else float("inf")


class Nearest(RuleBase):
    """최근접 — 크레인이 **덜 움직이는** 일감부터 (갠트리 이동거리 최소).

    `duration_s` 와 다르다. 소요시간에는 이동 + 들고내림 + 재조작이 다 들어가는데
    여기서는 **이동만** 본다 — 가깝지만 재조작이 많아 오래 걸리는 컨테이너가
    소요시간 규칙과 갈리는 지점이다.
    """

    name = "NEAREST"

    def key(self, sim, crane_id, gc, ref) -> float:
        st = sim.fleet.get(crane_id).state
        if gc.plan is None:
            return float("inf")
        lo, hi = gc.plan.corridor
        target = lo if abs(lo - st.position_bay) <= abs(hi - st.position_bay) else hi
        return abs(float(target) - float(st.position_bay))


class LongestWait(RuleBase):
    """최장 대기 우선 — 가장 오래 기다린 트럭부터 (공정성 규칙).

    ⚠️ 본선 일감은 누적 대기가 정의상 0 이라 **트럭보다 늘 뒤**가 된다.
    `SF_SPT` 가 본선을 3번째 자리에 두고도 0% 만 쓰던 것과 달리, 이 바닥은
    본선을 **적극적으로 뒤로 민다**. 그것이 이 규칙의 성질이다.
    """

    name = "LWKR"

    def key(self, sim, crane_id, gc, ref) -> float:
        return -(sim.cum_wait(ref.job_id) if ref.is_external else 0.0)


class RandomOrder(RuleBase):
    """무작위 — 기준 없음. **바닥 품질 축의 하한**이다.

    이게 있어야 *"바닥이 좋아질수록 재배치 이득이 어떻게 되나"* 가 점으로 그려진다.
    없으면 바닥들이 다 비슷한 데 몰려 관계가 안 보인다.

    난수를 쓰지 않고 **해시**를 쓴다 — 같은 시드가 같은 배정을 내야 짝비교가
    성립한다. `hash()` 는 프로세스마다 달라서 못 쓴다(`PYTHONHASHSEED`).
    """

    name = "RANDOM"

    def __init__(self, seed: int = 0):
        self.seed = int(seed)

    def key(self, sim, crane_id, gc, ref) -> float:
        h = hashlib.blake2b(f"{self.seed}:{ref.job_id}".encode(),
                            digest_size=8).digest()
        return int.from_bytes(h, "big") / 2.0 ** 64


#: 이름 → 만드는 법. `episode.DISPATCHERS_READY` 가 이 표를 쓴다.
RULE_BASES: dict[str, type] = {
    "FIFO": FIFO, "LIFO": LIFO, "SPT": SPT,
    "NEAREST": Nearest, "LWKR": LongestWait, "RANDOM": RandomOrder,
}


def make_preference(name: str, *, seed: int = 0):
    """바닥 이름으로 `Preference` 를 만든다. 모르는 이름이면 즉시 거절한다."""
    if name not in RULE_BASES:
        raise ValueError(f"모르는 크레인 바닥: {name!r} (있는 것: "
                         f"{', '.join(sorted(RULE_BASES))})")
    cls = RULE_BASES[name]
    return cls(seed=seed) if cls is RandomOrder else cls()
