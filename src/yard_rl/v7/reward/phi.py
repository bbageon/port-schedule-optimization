"""Φ — 터미널 하루 총비용, 원화 네 항.

설계 정본: `.claude/docs/architecture/04-비용과-보상.md`

    Φ = C_wait + C_move + C_rehandle + C_vessel        [원]

구 Φ 는 "비용시간(cost-hour)" 단위였고 블록별 합·주행·기사 외부 대기가 흩어져
있었다. 네 항·원화로 통일한다.

■ 이중 계상 금지 (04 §1-4)
  · 재조작 lift 시간은 **항3에만** — 항2의 "추가 이동" 에서 뺀다
  · 트럭 시간은 **게이트 진입→작업 완료** — 완료 후 출차 시간은 진단 열
  · 고시 그대로의 법정 재현값은 진단 열. Φ 에 더하지 않는다
"""
from __future__ import annotations

from dataclasses import dataclass

from ..schema.lifecycle import (TRUCK_TIME_CONTRACT, censored_turn_time_s,
    censored_gate_to_job_done_s, censored_job_done_to_gate_out_s)
from .krw import (rehandle_krw, truck_wait_krw, vessel_idle_krw, yc_move_krw)


@dataclass(frozen=True)
class PhiBreakdown:
    """네 항과 합계. 진단 열은 Φ 에 안 들어간다."""

    wait: float
    move: float
    rehandle: float
    vessel: float

    # ── 진단 열 (보고용 · Φ 에 더하지 않는다)
    n_trucks: int = 0
    n_censored: int = 0
    mean_turn_time_s: float = 0.0
    over_ratio: float = 0.0          # 턴타임이 1시간을 넘은 트럭의 비율
    #: 출차 기준의 과거 진단을 보존한다. 새 작업 완료 분위와 혼합하지 않는다.
    p50_turn_time_s: float = 0.0
    p90_turn_time_s: float = 0.0
    p99_turn_time_s: float = 0.0

    # Primary work metrics; legacy turn_time/n_censored still mean gate exit.
    n_job_censored: int = 0
    n_job_done: int = 0
    n_exit_pending: int = 0
    mean_gate_to_job_done_s: float = 0.0
    p50_gate_to_job_done_s: float = 0.0
    p90_gate_to_job_done_s: float = 0.0
    p99_gate_to_job_done_s: float = 0.0
    job_over_ratio: float = 0.0
    post_job_gate_s: float = 0.0
    mean_job_done_to_gate_out_s: float = 0.0

    @property
    def total(self) -> float:
        return self.wait + self.move + self.rehandle + self.vessel

    def as_dict(self) -> dict:
        return {
            "phi_krw": self.total,
            "c_wait": self.wait, "c_move": self.move,
            "c_rehandle": self.rehandle, "c_vessel": self.vessel,
            "n_trucks": self.n_trucks, "n_censored": self.n_censored,
            "mean_turn_time_s": self.mean_turn_time_s,
            "over_ratio": self.over_ratio,
            "p50_turn_time_s": self.p50_turn_time_s,
            "p90_turn_time_s": self.p90_turn_time_s,
            "p99_turn_time_s": self.p99_turn_time_s,
            "truck_time_contract": TRUCK_TIME_CONTRACT,
            **{k: getattr(self, k) for k in WORK_METRIC_KEYS},
        }


WORK_METRIC_KEYS = ('n_job_censored', 'n_job_done', 'n_exit_pending',
    'mean_gate_to_job_done_s', 'p50_gate_to_job_done_s', 'p90_gate_to_job_done_s',
    'p99_gate_to_job_done_s', 'job_over_ratio', 'post_job_gate_s',
    'mean_job_done_to_gate_out_s')


def terminal_cost_krw(records, *, end_s: float,
                      vessel_idle: dict[str, tuple[float, float]] | None = None,
                      yc_extra_move_s: float = 0.0,
                      rehandles: int = 0) -> PhiBreakdown:
    """Φ 를 네 항으로 계산한다.

    `records`         : {doc_key: ExecutionRecord}
    `end_s`           : 평가 종료 시각 — 미완료 트럭을 여기서 **검열**한다
    `vessel_idle`     : {vessel_id: (GT, 선박 유휴 초)}
    `yc_extra_move_s` : 터미널 전체의 YC **빈 주행** 시간(초)
    `rehandles`       : 터미널 전체의 재조작 횟수

    ★항2·항3·항4 는 **기록이 아니라 터미널 누적 계수기**에서 온다 (2026-08-22).
      기록에서 합산하면 창을 못 자른다 — 트럭 시간(항1)은 기록에 A·C 시각이 남아
      `end_s` 로 검열되지만, "지금까지 재조작 몇 번" 은 시각이 없어 과거 시점 값을
      되짚을 수 없다. 그래서 호출부가 **그 시점의 계수기 값**을 넣어 준다.
      `rec.rehandles`·`rec.yc_extra_move_s` 는 오더별 진단으로 남지만 Φ 에 더하지
      않는다 — 더하면 같은 사실이 두 번 들어간다(04 §1-4 이중 계상 금지).

    **미완료 검열**: 완료차는 `min(C,T)−A`, 미완료차는 `T−A`. 그래야 에피소드 끝에 남은
    트럭이 비용을 피하지 못한다.
    """
    wait = 0.0
    n = censored = 0
    tt_sum = 0.0
    over = 0
    samples: list[float] = []
    work_samples: list[float] = []
    work_sum = post_sum = 0.0
    work_censored = exit_pending = work_over = 0

    for rec in records.values():
        tt = censored_turn_time_s(rec, end_s)
        if tt is None:
            continue                       # 아직 게이트도 안 지났다 = 무대 밖
        work = censored_gate_to_job_done_s(rec, end_s)
        wait += truck_wait_krw(work)
        # ── 아래는 **진단 열**이라 Φ 에 안 들어간다. 창 기준으로 센다.
        if rec.gate_in_s > end_s:
            continue                       # 창 뒤에 올 트럭 — 이 창의 표본이 아니다
        n += 1
        tt_sum += tt
        samples.append(tt)
        work_sum += work
        work_samples.append(work)
        post = censored_job_done_to_gate_out_s(rec, end_s)
        if post is None:
            work_censored += 1
        else:
            post_sum += post
            exit_pending += rec.gate_out_s is None or rec.gate_out_s > end_s
        work_over += work > 3600.0
        if rec.gate_out_s is None or rec.gate_out_s > end_s:
            censored += 1                  # 창 끝에 아직 안 나갔다
        if tt > 3600.0:
            over += 1

    vessel = 0.0
    for gt, idle_s in (vessel_idle or {}).values():
        vessel += vessel_idle_krw(gt, idle_s)

    move = yc_move_krw(float(yc_extra_move_s))
    rehab = rehandle_krw(int(rehandles))

    samples.sort()
    work_samples.sort()

    def q(p: float, values=samples) -> float:
        if not values:
            return 0.0
        return values[min(len(values) - 1, int(p * len(values)))]

    return PhiBreakdown(
        wait=wait, move=move, rehandle=rehab, vessel=vessel,
        n_trucks=n, n_censored=censored,
        mean_turn_time_s=(tt_sum / n if n else 0.0),
        over_ratio=(over / n if n else 0.0),
        p50_turn_time_s=q(0.50), p90_turn_time_s=q(0.90),
        p99_turn_time_s=q(0.99),
        n_job_censored=work_censored, n_job_done=n-work_censored,
        n_exit_pending=exit_pending,
        mean_gate_to_job_done_s=work_sum/n if n else 0.0,
        p50_gate_to_job_done_s=q(.50, work_samples),
        p90_gate_to_job_done_s=q(.90, work_samples),
        p99_gate_to_job_done_s=q(.99, work_samples),
        job_over_ratio=work_over/n if n else 0.0, post_job_gate_s=post_sum,
        mean_job_done_to_gate_out_s=post_sum/(n-work_censored) if n>work_censored else 0.0,
    )
