"""YR-018 — reward weight 민감도: w_tail {0,.1,.3,1} × 학습예산 grid.

목적: Exp-1/23 에서 관찰된 "평균대기 개선 ↔ P95 악화" trade-off 가 보상의
tail 가중치(w_tail)로 제어 가능한지 — 곡선을 그려 P95 를 지키는 지점이
존재하는지 확인한다 (03 §1.2 weight 원칙, 후보 {0,0.1,0.3,1.0}).

수렴진단(YR-020-수렴진단-2026-07-14) 요건 반영:
- 학습예산 축(epochs) 을 grid 에 동반 — 결론이 예산에 따라 뒤집히는지 확인.
- 평가 중 fallback(미방문 상태)·thin(선택 (s,a) 방문<5) 비율을 리포트에 박제 —
  "수렴 전 스냅샷 비교" 함정을 리포트가 스스로 드러내게 한다.

Scale 은 FIFO train fit 을 전 가중치가 공유한다 (w 는 합성 단계에만 작용
— 02 §7). bucket 도 동일. 따라서 가중치 간 차이는 순수하게 보상 구조 차이다.
"""
from __future__ import annotations

import json
import statistics as _st
import time
from dataclasses import replace
from pathlib import Path

from ..domain.enums import ControlScope, InformationLevel, PriorityRule
from ..envs.yard_env import YardEnv
from ..io.profile_loader import load_profile
from ..io.scenario_gen import GenParams
from ..policies.baselines import FixedRulePolicy
from ..policies.q_learning import QLearningAgent, QLearningConfig, train
from .report import _mean, _series
from .runner import (TEST_SEED0, TRAIN_SEED0, EpisodeResult, PolicySpec,
                     check_seed_bands, evaluate_paired, fit_buckets_and_scales,
                     make_scenarios, run_episode)
from .statistics import paired_diff

_LEVEL = InformationLevel.BLOCK_ARRIVAL
_SCOPE = ControlScope.SEQUENCE_ONLY


class DiagPolicy:
    """QL 정책 wrapper — 평가 결정별 수렴 진단 (결과에 영향 없음, 관찰만)."""

    def __init__(self, agent: QLearningAgent):
        self.agent = agent
        self.decisions = 0
        self.fallback = 0   # 미방문 상태 또는 시도된 valid action 없음
        self.thin = 0       # 선택 (s,a) 의 방문 횟수 < 5
        self.chosen_n: list[int] = []

    @property
    def name(self) -> str:
        return self.agent.name

    def act(self, state, mask):
        self.decisions += 1
        t = self.agent.table
        tried_valid = (state in t.q and
                       any(mask[a] and t.n[state][a] > 0 for a in range(t.n_actions)))
        a = self.agent.act(state, mask)
        if tried_valid:
            n = t.n[state][a]
            self.chosen_n.append(n)
            if n < 5:
                self.thin += 1
        else:
            self.fallback += 1
        return a

    def stats(self) -> dict:
        d = max(1, self.decisions)
        return {
            "decisions": self.decisions,
            "fallback_pct": 100.0 * self.fallback / d,
            "thin_pct": 100.0 * self.thin / d,
            "chosen_n_median": float(_st.median(self.chosen_n)) if self.chosen_n else 0.0,
        }


def _wname(w: float) -> str:
    return f"QL_WT{w:g}"


def run_wtail_grid(*, profile_path: str, out_dir: str, n_train: int = 30,
                   n_eval: int = 12, epochs_list: tuple[int, ...] = (4, 10),
                   weights: tuple[float, ...] = (0.0, 0.1, 0.3, 1.0)) -> Path:
    t0 = time.time()
    check_seed_bands(n_train, 4, n_eval)
    profile = load_profile(profile_path)
    params = GenParams()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[prep] {profile.terminal_id}: train {n_train} / test {n_eval} + FIFO fit")
    train_scs = make_scenarios(profile, TRAIN_SEED0, n_train, params)
    test_scs = make_scenarios(profile, TEST_SEED0, n_eval, params)
    buckets, reward0 = fit_buckets_and_scales(profile, train_scs, _LEVEL)
    buckets.save(out / "buckets.json")
    reward0.save(out / "reward_scales.json")

    per_budget: dict[int, dict[str, list[EpisodeResult]]] = {}
    diags: dict[int, dict[str, dict]] = {}
    for epochs in epochs_list:
        specs = [PolicySpec("FIFO", FixedRulePolicy(PriorityRule.FIFO))]
        diag_pols: dict[str, DiagPolicy] = {}
        for w in weights:
            name = _wname(w)
            print(f"[train] e{epochs} {name}")
            reward_w = replace(reward0, w_tail=w)
            env = YardEnv(profile, info_level=_LEVEL, control_scope=_SCOPE,
                          bucket_cfg=buckets, reward_cfg=reward_w)
            agent = QLearningAgent(QLearningConfig(), seed=0, policy_name=name)
            train(agent, env, train_scs, epochs=epochs)
            agent.table.save(out / f"qtable_e{epochs}_{name}.json")
            diag = DiagPolicy(agent)
            diag_pols[name] = diag
            specs.append(PolicySpec(name, diag))
        print(f"[eval] e{epochs} paired ({len(specs)} 조건 × {n_eval} seeds)")
        # DiagPolicy 는 paired 평가 결정에서 fallback/thin 을 그대로 집계한다
        per_budget[epochs] = evaluate_paired(specs, profile, test_scs,
                                             buckets=buckets, reward=reward0,
                                             check_invariants=True)
        diags[epochs] = {n: p.stats() for n, p in diag_pols.items()}

    path = _build_report(per_budget, diags, weights=weights, out=out, meta={
        "실험": "YR-018 w_tail 민감도 — QL_EXP1 조건 (BLOCK_ARRIVAL · sequence_only · 단일 YC)",
        "프로파일": f"{profile.terminal_id} (assumed={profile.assumed})",
        "grid": f"w_tail {list(weights)} × epochs {list(epochs_list)}",
        "seeds": f"train {TRAIN_SEED0}..{TRAIN_SEED0 + n_train - 1} / "
                 f"test {TEST_SEED0}..{TEST_SEED0 + n_eval - 1} (paired)",
        "reward": "정규화 Core Cost — w_tail 만 변경, 나머지 w=(1,·,.1,.1,.3)·"
                  "Scale=train FIFO fit 공유 (가중치 간 순수 비교)",
        "수렴진단": "fallback=미방문 상태 결정 비율, thin=방문<5 (s,a) 결정 비율 — "
                 "[YR-020-수렴진단-2026-07-14](../../../.claude/docs/YR-020-수렴진단-2026-07-14.md)",
        "주의": "CURRENT_RULE 미확보 — 휴리스틱 대비 비교만 유효",
    })
    print(f"완료 ({time.time() - t0:.1f}s) → {path}")
    return path


def _build_report(per_budget, diags, *, weights, out: Path, meta: dict) -> Path:
    (out / "wtail_results.json").write_text(
        json.dumps({f"e{ep}": {p: [r.__dict__ for r in rs] for p, rs in res.items()}
                    for ep, res in per_budget.items()},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    L: list[str] = []
    L.append("# YR-018 — w_tail 민감도 grid (합성 시나리오)")
    L.append("")
    L.append("> ⚠ **가정 프로파일(assumed) + 합성 시나리오** 예비 실험 — 실운영 대비 아님.")
    L.append("> 목적: 평균대기 개선을 얼마나 반납하면 P95(tail) 악화를 없앨 수 있는지의 곡선.")
    L.append("")
    L.append("## 실행 조건")
    L.append("")
    for k, v in meta.items():
        L.append(f"- **{k}**: {v}")
    for ep, results in per_budget.items():
        base = results["FIFO"]
        L.append("")
        L.append(f"## 예산 e{ep} — paired vs FIFO + 수렴 진단")
        L.append("")
        L.append("| w_tail | mean_wait Δ% (유의) | p95 Δ% (유의) | p95 Δ 95% CI | "
                 "tail_area Δ% | travel Δ% | fallback% | thin% | 방문중앙값 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for w in weights:
            name = _wname(w)
            rs = results[name]
            dw = paired_diff(_series(base, "mean_wait_min"), _series(rs, "mean_wait_min"))
            dp = paired_diff(_series(base, "p95_wait_min"), _series(rs, "p95_wait_min"))
            dt = paired_diff(_series(base, "tail_area_h"), _series(rs, "tail_area_h"))
            dv = paired_diff(_series(base, "travel_km"), _series(rs, "travel_km"))
            g = diags[ep][name]
            L.append(
                f"| {w:g} | {dw['pct_change']:+.1f}% ({dw['seeds_same_direction']}/{dw['n']}"
                f"{'✔' if dw['significant'] else '—'}) "
                f"| {dp['pct_change']:+.1f}% ({'✔' if dp['significant'] else '—'}) "
                f"| [{dp['ci_lo']:+.2f}, {dp['ci_hi']:+.2f}]분 "
                f"| {dt['pct_change']:+.1f}% | {dv['pct_change']:+.1f}% "
                f"| {g['fallback_pct']:.1f}% | {g['thin_pct']:.1f}% "
                f"| {g['chosen_n_median']:.0f} |")
        L.append("")
        L.append(f"### e{ep} 인접 가중치 사다리 (paired, b vs a)")
        L.append("")
        L.append("| 비교 | 지표 | a | b | Δ (95% CI) | 유의 |")
        L.append("|---|---|---|---|---|---|")
        for a, b in zip(weights, weights[1:]):
            for key in ("mean_wait_min", "p95_wait_min", "tail_area_h"):
                d = paired_diff(_series(results[_wname(a)], key),
                                _series(results[_wname(b)], key))
                L.append(f"| w{a:g}→w{b:g} | {key} | {d['mean_base']:.2f} "
                         f"| {d['mean_alt']:.2f} | {d['mean_diff']:+.2f} "
                         f"[{d['ci_lo']:+.2f}, {d['ci_hi']:+.2f}] "
                         f"| {'✔' if d['significant'] else '—'} |")
        L.append("")
        L.append(f"### e{ep} 절대값 (분)")
        L.append("")
        L.append("| 지표 | FIFO | " + " | ".join(_wname(w) for w in weights) + " |")
        L.append("|" + "---|" * (len(weights) + 2))
        for key in ("mean_wait_min", "p95_wait_min", "tail_area_h", "sla_exceed_count"):
            row = [f"{_mean(results['FIFO'], key):.2f}"] + \
                  [f"{_mean(results[_wname(w)], key):.2f}" for w in weights]
            L.append(f"| {key} | " + " | ".join(row) + " |")
    L.append("")
    L.append("*생성: yard_rl.experiments.wtail_grid — 원자료 wtail_results.json*")
    path = out / "wtail_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
