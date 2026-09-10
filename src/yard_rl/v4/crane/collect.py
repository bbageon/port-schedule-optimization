"""크레인 결정을 표본해 라벨로 바꾼다 ([[YR-308]]).

■ 왜 표본하나 — 전수는 물리적으로 불가능하다
  크레인 결정은 하루 **수만 건**이고 한 건을 라벨링하려면 세계 둘(≈30초)을 굴려야
  한다. 전수면 회차당 수백 시간이다. 재배정층과 **같은 규모(하루 64건)** 로 뽑는다.
  메커니즘은 그대로 두고 표본 수만 줄이는 것이라 우회가 아니다.

  뽑는 방법은 **결정론 등간격**이다 — 난수를 쓰면 같은 시드가 같은 라벨 집합을
  못 만들어 재현이 깨진다 (`RolloutBudget` 을 그대로 쓴다).

■ ★쓸 수 있는 결정만 예산을 쓴다 (2026-09-10 실측으로 고침)
  처음엔 재배정층처럼 *"모든 결정을 세고 stride 마다 하나"* 로 짰다. 그런데
  크레인 결정은 **셋 중 둘이 라벨이 안 된다** (부하 300 · 하루 3,920 결정×크레인):

      WAIT 33.0%   실행가능 일감 하나뿐 37.0%   ← 합쳐서 70%
      쓸 수 있음 29.5%   필수등급 불일치 0.5%

  그대로 두니 예산 4건이 전부 못 쓰는 결정에 나가 **라벨 0건**이 나왔다.
  그래서 **먼저 쓸 수 있는지 보고, 그 다음에 예산을 쓴다.** 판정은 후보 목록을
  훑는 것뿐이라 싸다(복제·특징 계산은 뽑힌 뒤에만 한다).

■ 어떤 결정이 라벨이 되나 — 셋 다 만족해야 한다
  ① 그 크레인이 **실제로 일감을 잡았다** (WAIT 는 강제할 손잡이가 없다)
  ② **다른 일감**이 최소 하나 더 실행 가능했다 (비교 대상이 있어야 한다)
  ③ 그 대안이 picked 와 **같은 필수(mandatory) 등급**이다
     — resolver 가 필수를 어떤 선호보다 앞에 두므로, 등급이 다르면 강제가 안 먹는다

■ 무엇을 세나 — 버린 것도 센다
  `no_alt` / `factual_mismatch` / `force_failed` 를 따로 센다. 라벨 수만 보고하면
  *"64건 뽑았다"* 가 실제로는 12건이어도 안 보인다.
"""
from __future__ import annotations

import copy

from ..stage.rollout import RolloutBudget
from .features import crane_features
from .rollout import CraneBranchJob, CranePool
from .teacher import CraneLabelCollector


class CraneCollector:
    """에피소드가 도는 동안 결정을 표본하고, 끝나면 라벨을 조립한다."""

    def __init__(self, ctx, *, budget: RolloutBudget, horizon_s: float,
                 workers: int = 1):
        self.ctx = ctx
        self.budget = budget
        self.pool = CranePool(ctx, horizon_s=horizon_s, workers=workers)
        #: 결정 열쇠 → (고른 것의 특징, 대안의 특징). 작업자에게는 안 보낸다.
        self.rows: dict[tuple, tuple[list, list]] = {}
        #: 회계 — **버린 것까지** 남긴다. 라벨 수만 보면 "64건 뽑았다" 가 실제로
        #: 12건이어도 안 보인다.
        self.stats = {"seen": 0,       # 하루의 전 크레인 결정
                      "usable": 0,     # 그중 라벨이 될 수 있던 것
                      "sampled": 0,    # 그중 예산이 실제로 뽑은 것
                      "no_alt": 0, "no_pick": 0,
                      "factual_mismatch": 0, "force_failed": 0, "labeled": 0}

    # ------------------------------------------------------------------ 훅
    def hook(self, mbt, market, orders, records):
        """`_rule_policy(on_crane=...)` 에 꽂을 함수를 만든다.

        `mbt`·`market`·`records` 는 에피소드가 들고 있는 **살아 있는** 객체라
        여기서 잡아 두고, 표본된 결정에서만 그 순간 상태를 복제한다.
        """
        #: 시뮬 → 블록 이름. 엔진은 자기 블록 이름을 안 들고 다녀서 여기서 맨다.
        #: (`mbt.blocks` 는 런 중에 안 바뀐다 — 한 번만 만든다.)
        bmap = {id(s): b for b, s in mbt.blocks.items()}

        def on_crane(sim, dp, gb, assign):
            self.stats["seen"] += 1
            # ★① 먼저 쓸 수 있는 결정인지 본다 (머리말 참조 — 이 순서여야 한다)
            pick = self._labelable(sim, gb, assign)
            if pick is None:
                return
            self.stats["usable"] += 1
            # ★② 계수기는 **쓸 수 있는 결정 전부**에서 돈다. 뽑힌 것만 세면
            #    stride 가 영원히 안 맞아 첫 한 건만 뽑힌다.
            if not self.budget.take():
                return
            self.stats["sampled"] += 1
            crane, picked_gc, alt_gc = pick
            block = bmap[id(sim)]
            key = (round(sim.clock, 6), block, crane)
            self.rows[key] = (crane_features(sim, crane, picked_gc),
                              crane_features(sim, crane, alt_gc))
            self.pool.submit(CraneBranchJob(
                key=key, t=sim.clock, crane=crane,
                #: ★결정 **직전** 스냅샷. `_pending` 이 열린 채로 뜬다 —
                #:  분기 세계는 그 결정을 손으로 한 번 돌려서 연다.
                mbt=copy.deepcopy(mbt), decision=dp, block=block,
                orders=dict(orders), records=copy.deepcopy(records),
                decided=set(getattr(market, "decided", ())),
                picked_job=picked_gc.job_ref.job_id,
                alt_job=alt_gc.job_ref.job_id))
        return on_crane

    def _labelable(self, sim, gb, assign):
        """라벨을 만들 수 있는 (크레인, 고른 후보, 대안) — 없으면 `None`."""
        for crane in sorted(gb):
            gc = assign.get(crane)
            ref = getattr(gc, "job_ref", None)
            if ref is None:                      # ① WAIT — 강제할 손잡이가 없다
                continue
            cands = [g for g in gb[crane].items
                     if g.feasible and g.job_ref is not None
                     and g.job_ref.job_id != ref.job_id
                     and g.mandatory == gc.mandatory]   # ③ 같은 필수 등급
            if not cands:                        # ② 비교 대상이 없다
                continue
            alt = min(cands, key=lambda g: (self._rank(sim, crane, g),
                                            g.job_ref.job_id))
            return crane, gc, alt
        # 여기까지 왔다 = 전 크레인이 WAIT 이거나 대안이 없었다
        if all(getattr(assign.get(c), "job_ref", None) is None for c in gb):
            self.stats["no_pick"] += 1
        else:
            self.stats["no_alt"] += 1
        return None

    def _rank(self, sim, crane, gc) -> tuple:
        """정책이 이 후보를 몇 번째로 보나 — **차점자**를 고르는 잣대."""
        r = self._pref.rank(sim, crane, gc)
        return tuple(r)

    #: 정책 손잡이 — `attach` 로 꽂는다(에피소드가 만든 그 선호여야 한다).
    _pref = None

    def attach(self, pref) -> None:
        self._pref = pref

    # --------------------------------------------------------------- 마무리
    def finish(self):
        """굴린 결과를 라벨로 바꾼다. 진단이 틀린 세계는 **버린다**."""
        out = CraneLabelCollector()
        for r in self.pool.results():
            if not r["factual_ok"]:
                # 사실 가지가 실제와 다른 결정을 냈다 = 분기 재조립이 상태를
                # 복원하지 못한 것. 이 라벨은 다른 결정의 값이라 거짓이다.
                self.stats["factual_mismatch"] += 1
                continue
            if not r["forced_ok"]:
                self.stats["force_failed"] += 1
                continue
            picked_row, alt_row = self.rows[r["key"]]
            out.add(picked_row=picked_row, alt_row=alt_row,
                    phi_factual=r["phi_factual"], phi_alt=r["phi_alt"],
                    crane=r["crane"], job=r["picked_job"])
            self.stats["labeled"] += 1
        out.note_worlds(self.pool.n_worlds)
        return out.result()

    def close(self) -> None:
        self.pool.close()
