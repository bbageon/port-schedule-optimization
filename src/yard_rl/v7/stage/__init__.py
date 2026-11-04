"""무대 결합 — v3 판단층을 **실제 엔진 루프**에 꽂는 자리.

설계 정본: `.claude/docs/architecture/00-라이프사이클.md` · [[YR-215]]

■ 여기서 하는 일은 **번역**뿐이다
  엔진의 사건을 v3 기록으로, v3 의 거래를 엔진의 원자 트랜잭션으로 옮긴다.
  굴러가는 세계 자체는 `v3/world/`(v3 소유 사본)이고 이 패키지는 그 위에 얹힌다.

■ 꽂는 자리는 이미 있다
      MultiBlockTerminal.run(policy_fn, review_fn=..., cost_fn=...)
                                        └─ 전 블록이 같은 t 에 park 했을 때 불린다
  v2 는 여기에 `UnifiedSellOrchestrator` 를 꽂았다. v3 는 `MarketBridge` 를 꽂는다.

■ ★한 epoch 안의 순서가 계약이다
      ① 투입(공고)  ② 사건 수집  ③ 시장  ④ 확정
  ② 가 ③ 앞이어야 정책이 **그 시각의 진짜 상태**를 본다. ④ 가 ③ 뒤여야 한 epoch
  안에서 결정이 서로의 결과를 못 본다(동시 결정).

■ 정보 경계 — 여기가 새기 가장 쉬운 자리다
  엔진 장부에는 **미래 시각이 이미 들어 있다**(투입 시 `gate_in` 을 미리 적는다).
  그래서 사건 수집은 `값 ≤ t` 인 것만 기록으로 옮긴다 — 그러지 않으면 정책이
  아직 안 일어난 게이트인을 읽는다.
"""

from .branchpool import BranchJob, BranchPool, default_workers
from .bridge import MarketBridge, epoch_on_grid
from .episode import ARMS, EpisodeResult, run_episode
from .month import (DAY_S, LOAD_WEIGHTS, N_DAYS, DayPlan, build_month,
                    plan_month, plan_month_vessels, prune_completed, summarize)
from .month_engine import MonthTerminal, inject_vessel
from .month_run import DayReport, MonthResult, run_month
from .orders import V3Announcer, build_stage, orders_from_schedule
from .rollout import (BranchResult, RolloutBudget, SnapshotRollout,
                      identity_check)
from .vessels import (DAY_FLEET, DailyVessel, build_diurnal_v3, fleet_summary,
                      plan_streams, sample_day_vessels, structural_idle_krw)

__all__ = ["MarketBridge", "epoch_on_grid", "build_stage", "orders_from_schedule",
           "V3Announcer", "SnapshotRollout", "identity_check", "RolloutBudget",
           "BranchResult",
           "run_episode", "EpisodeResult", "ARMS",
           "DailyVessel", "sample_day_vessels", "plan_streams",
           "build_diurnal_v3", "fleet_summary", "structural_idle_krw",
           "DAY_FLEET", "BranchPool", "BranchJob", "default_workers"]
