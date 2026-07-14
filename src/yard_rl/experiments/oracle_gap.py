"""YR-031 — Oracle 상한 측정: 전지적 beam search vs greedy (사전등록 동결).

질문: 현 조건에서 greedy(SPT)를 이길 상금이 애초에 얼마인가?
방법: 결정 lockstep beam search (누적비용 = 최종 mean_wait 항등) + greedy
궤적 상시 시드 → best_found ≤ greedy 구성적 보장. 찾은 개선 = 상금의 하한.
사전등록: .claude/docs/strategy-history/2026-07-14-YR-031-oracle-gap-prereg.md
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Callable

from ..envs.direct_job_env import DirectJobEnv, SLAMode
from ..io.profile_loader import load_profile
from ..policies.direct_baselines import DirectJobRulePolicy, DirectRule
from .coverage_ablation import _gen_params
from .direct_job_runner import (_git_state, _json_dump, _profile_digest,
                                _scenario)

EXPERIMENT_ID = "YR-031-oracle-gap"
ARM = SLAMode.OFF
SCHEMA = "v1_final"
# 사전등록 §3 판정 임계 (분)
CLOSED_BELOW = 0.15
OPEN_ABOVE = 0.30


@dataclass(frozen=True)
class OracleGapConfig:
    test_episodes: int = 100
    test_seed0: int = 160_000      # YR-012 와 동일 test band (사후 분석 — §2)
    beam_width: int = 12
    n_external: int = 100
    drain_window_s: float = 86_400.0
    quick: bool = False

    def __post_init__(self) -> None:
        if min(self.test_episodes, self.beam_width, self.n_external) <= 0:
            raise ValueError("all sizes must be positive")

    @property
    def test_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.test_seed0, self.test_seed0 + self.test_episodes))


def quick_oracle_config() -> OracleGapConfig:
    return OracleGapConfig(test_episodes=3, beam_width=4, n_external=10,
                           quick=True)


@dataclass(frozen=True)
class _CfgShim:
    _cfg: OracleGapConfig

    @property
    def n_external(self) -> int:
        return self._cfg.n_external

    @property
    def drain_window_s(self) -> float:
        return self._cfg.drain_window_s


def _new_env(profile, cfg: OracleGapConfig) -> DirectJobEnv:
    # invariant 검사는 greedy 대조 실행(check ON)에서 1회 수행 — beam 분기
    # (동일 결정론 엔진)의 반복 검사는 비용만 크므로 생략 (사전등록 §2)
    return DirectJobEnv(profile, sla_mode=ARM, expected_n_config=cfg.n_external,
                        state_schema=SCHEMA, check_invariants=False)


def _greedy_day(profile, scenario, cfg: OracleGapConfig) -> tuple[float, list[str]]:
    """greedy 1일 실행 — (mean_wait, 선택 궤적). invariant 검사 ON."""
    env = DirectJobEnv(profile, sla_mode=ARM, expected_n_config=cfg.n_external,
                       state_schema=SCHEMA, check_invariants=True)
    greedy = DirectJobRulePolicy(DirectRule.IMMEDIATE_COST_GREEDY)
    state, info = env.reset(scenario)
    trace: list[str] = []
    while state is not None:
        pick = greedy.act(state, info.candidates)
        trace.append(pick.job_id)
        state, _c, _d, info = env.step(pick)
    return env.cumulative_cost, trace


def _beam_day(profile, scenario, greedy_trace: list[str],
              cfg: OracleGapConfig) -> float:
    """결정 lockstep beam — greedy 노드를 별도 트랙으로 항상 유지 (§2).

    매 step: greedy 노드 + beam 노드의 모든 feasible 자식을 만들고 누적비용
    상위 W 를 beam 으로. greedy 노드 자체는 궤적을 따라 in-place 전진 —
    beam 에서 탈락해도 소멸하지 않으므로 best_found ≤ greedy 가 보장된다.
    """
    # node = (누적비용, env, info, 실행 궤적 tuple). 궤적이 상태를 유일 결정
    # (결정론 엔진) — 궤적 키 dedup 으로 중복 상태 제거 (리뷰 확정 결함:
    # 중복 미제거 시 유효 폭이 W/2~W/3 로 붕괴 → 상금 과소평가·CLOSED 편향)
    def expand(cum: float, env: DirectJobEnv, info, trace: tuple
               ) -> dict[tuple, tuple]:
        out: dict[tuple, tuple] = {}
        for cand in info.candidates:
            child = copy.deepcopy(env)
            _s, cost, _d, child_info = child.step(cand.job_id)
            key = trace + (cand.job_id,)
            out[key] = (cum + cost, child, child_info, key)
        return out

    g_env = _new_env(profile, cfg)          # greedy 전용 트랙 (결정론 재현)
    _gs, g_info = g_env.reset(scenario)
    g_cum, g_trace = 0.0, ()
    # beam 은 빈 상태로 시작 — step 0 의 root 자식은 greedy 트랙 expand 가 전부
    # 공급한다 (root 별도 시드는 완전 중복이라 제거)
    beam: dict[tuple, tuple] = {}
    for job_id in greedy_trace:
        children = expand(g_cum, g_env, g_info, g_trace)  # greedy 이웃 (1-이탈)
        for cum, env, info, trace in beam.values():
            for key, node in expand(cum, env, info, trace).items():
                children.setdefault(key, node)  # 동일 궤적 = 동일 상태·비용
        ranked = sorted(children.values(), key=lambda n: (n[0], n[3]))
        beam = {n[3]: n for n in ranked[:cfg.beam_width]}
        _s, cost, _d, g_info = g_env.step(job_id)         # greedy 트랙 전진
        g_cum += cost
        g_trace += (job_id,)
    return min([g_cum] + [n[0] for n in beam.values()])


def run_oracle_gap(profile_path: str = "configs/terminals/hjnc_armg.yaml",
                   out_dir: str = "outputs/reports/oracle_gap_hjnc",
                   cfg: OracleGapConfig | None = None,
                   progress: Callable[[str], None] = print) -> Path:
    cfg = cfg or OracleGapConfig()
    started = time.time()
    git = _git_state()
    if not cfg.quick and (git["commit"] == "unknown" or git["dirty"]):
        raise RuntimeError("full YR-031 run requires a clean committed tree")
    profile = load_profile(profile_path)
    params = _gen_params(_CfgShim(cfg))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress(f"[YR-031] profile={profile.terminal_id} W={cfg.beam_width} "
             f"days={cfg.test_episodes}")

    rows: list[dict[str, float | int]] = []
    for index, seed in enumerate(cfg.test_seeds, start=1):
        scenario = _scenario(profile, seed, params, cfg.n_external)
        greedy_mean, trace = _greedy_day(profile, scenario, cfg)
        best = _beam_day(profile, scenario, trace, cfg)
        if best > greedy_mean + 1e-9:
            raise AssertionError("beam best 가 greedy 초과 — 시드 보장 위반")
        rows.append({"seed": seed, "greedy_mean": greedy_mean,
                     "best_found_mean": best,
                     "improvement": greedy_mean - best})
        if index % 10 == 0 or index == cfg.test_episodes:
            running = fmean(r["improvement"] for r in rows)
            progress(f"[beam] {index}/{cfg.test_episodes} 일 — 상금(하한) "
                     f"누적평균 {running:.3f}분")

    imps = [float(r["improvement"]) for r in rows]
    prize = fmean(imps)
    verdict = ("CLOSED" if prize < CLOSED_BELOW
               else "OPEN" if prize >= OPEN_ABOVE else "INTERMEDIATE")
    payload = {
        "manifest": {
            "schema_version": 1, "strategy_id": EXPERIMENT_ID,
            "mode": "quick" if cfg.quick else "full",
            "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "profile": {"path": str(profile_path),
                        "terminal_id": profile.terminal_id,
                        "assumed": profile.assumed,
                        "sha256": _profile_digest(profile_path)},
            "git": git, "config": asdict(cfg), "arm": ARM.value,
            "elapsed_s": time.time() - started,
        },
        "per_day": rows,
        "prize_lower_bound_mean_min": prize,
        "prize_max_day_min": max(imps),
        "days_with_any_improvement": sum(i > 1e-9 for i in imps),
        "verdict": verdict,
        "thresholds": {"closed_below": CLOSED_BELOW, "open_above": OPEN_ABOVE},
    }
    _json_dump(out / "oracle_gap_results.json", payload)
    report = _build_report(payload, out)
    progress(f"[YR-031] completed in {payload['manifest']['elapsed_s']:.1f}s "
             f"-> {report} (verdict={verdict})")
    return report


def _build_report(payload: dict, out: Path) -> Path:
    rows = payload["per_day"]
    prize = payload["prize_lower_bound_mean_min"]
    L: list[str] = []
    L.append("# YR-031 — Oracle 상한 측정 (평균대기 게임의 총 상금)")
    L.append("")
    L.append("> ⚠ 가정 프로파일 + 합성 시나리오. 전지적 beam search 가 찾은 개선 = "
             "상금의 **하한** (찾은 만큼은 확실히 달성 가능).")
    L.append("")
    L.append(f"- **총상금 (하한, {len(rows)}일 평균)**: **{prize:+.3f}분**")
    L.append(f"- 개선이 발견된 날: {payload['days_with_any_improvement']}/{len(rows)}"
             f" · 최대 개선일: {payload['prize_max_day_min']:.3f}분")
    L.append(f"- **판정**: **{payload['verdict']}** (닫힘 <"
             f"{payload['thresholds']['closed_below']} / 열림 ≥"
             f"{payload['thresholds']['open_above']}) · 참고: RL(Δ-net) 잔여 격차 +0.083분")
    L.append("")
    L.append("## 일별 (앞 20일)")
    L.append("")
    L.append("| seed | greedy | best_found | 개선 |")
    L.append("|---|---|---|---|")
    for r in rows[:20]:
        L.append(f"| {r['seed']} | {r['greedy_mean']:.3f} "
                 f"| {r['best_found_mean']:.3f} | {r['improvement']:+.3f} |")
    L.append("")
    L.append("*생성: yard_rl.experiments.oracle_gap — 원자료 oracle_gap_results.json*")
    path = out / "oracle_gap_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
