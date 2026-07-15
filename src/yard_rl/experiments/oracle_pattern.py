"""YR-031-b — Oracle 개선 순서 패턴 분석 (사전등록 동결).

사용자 가설 (2026-07-15) 2건을 oracle beam 궤적으로 판정한다:
- **H-A (feature 예측가능)**: "이탈해야 할 순간"이 관측 가능한(인과적) feature 로
  예측된다 — 참이면 그 feature 가 Δ-net 의 다음 입력.
- **H-B (조합 의존)**: 이탈 선택이 후보쌍 비교만으로 안 되고 후보 집합 맥락에
  의존한다 — 참이면 후보 독립 스코어링을 넘는 구조 필요.

방법: YR-031 beam(알고리즘 동일, 궤적 반환)을 재실행 → oracle 최적 궤적을
리플레이하며 매 결정에서 greedy 선택과 대조 → 이탈 이벤트 추출 → 로지스틱
회귀 (day-그룹 5-fold CV, torch) 로 AUC 판정. 재학습 없음 — 분석 실험.
사전등록: .claude/docs/strategy-history/2026-07-15-YR-031-b-oracle-pattern-prereg.md
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Callable, Sequence

import torch

from ..envs.direct_job_env import DirectJobEnv, SLAMode
from ..io.profile_loader import load_profile
from ..policies.direct_baselines import DirectJobRulePolicy, DirectRule
from .coverage_ablation import _gen_params
from .direct_job_runner import _git_state, _json_dump, _profile_digest, _scenario
from .oracle_gap import OracleGapConfig, _greedy_day, beam_day_with_trace

EXPERIMENT_ID = "YR-031-b-oracle-pattern"
ARM = SLAMode.OFF
SCHEMA = "v1_final"
# 사전등록 판정 임계
HA_PREDICTABLE = 0.75   # AUC ≥ → H-A 지지 / 0.60~0.75 부분적 / < 0.60 기각
HB_SET_GAIN = 0.05      # 집합 맥락 추가로 AUC 이득 ≥ → H-B 지지


@dataclass(frozen=True)
class PatternConfig:
    test_episodes: int = 100
    test_seed0: int = 160_000      # YR-031 과 동일 band (동일 oracle 재현)
    beam_width: int = 12
    n_external: int = 100
    drain_window_s: float = 86_400.0
    cv_folds: int = 5
    logistic_steps: int = 600
    logistic_lr: float = 0.05
    torch_seed: int = 74_031
    quick: bool = False

    def __post_init__(self) -> None:
        if min(self.test_episodes, self.beam_width, self.n_external,
               self.cv_folds, self.logistic_steps) <= 0:
            raise ValueError("all sizes must be positive")

    @property
    def test_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.test_seed0, self.test_seed0 + self.test_episodes))

    def oracle_cfg(self) -> OracleGapConfig:
        return OracleGapConfig(test_episodes=self.test_episodes,
                               test_seed0=self.test_seed0,
                               beam_width=self.beam_width,
                               n_external=self.n_external,
                               drain_window_s=self.drain_window_s,
                               quick=self.quick)


def quick_pattern_config() -> PatternConfig:
    return PatternConfig(test_episodes=4, beam_width=4, n_external=10,
                         cv_folds=2, logistic_steps=120, quick=True)


# ---------------------------------------------------------------- 이벤트 추출

def _set_aggregates(cands) -> list[float]:
    """후보 집합 맥락 8 features (인과적 — 결정 시점 관측만)."""
    services = [c.estimated_service_s for c in cands]
    reaches = [c.reach_s for c in cands]
    mean_s = fmean(services)
    return [
        float(len(cands)),
        min(services), mean_s, max(services),
        min(reaches), fmean(reaches),
        sum(c.transfer_direction == "YARD_TO_TRUCK" for c in cands) / len(cands),
        sum(s < mean_s for s in services) / len(cands),   # 짧은작업 비율 (집합 내)
    ]


def _cand_feats(c) -> list[float]:
    return [1.0 if c.transfer_direction == "YARD_TO_TRUCK" else 0.0,
            float(c.wait_s), float(c.reach_s), float(c.estimated_service_s),
            float(c.blocker_count)]


def collect_day_events(profile, scenario, oracle_trace: Sequence[str],
                       cfg: PatternConfig) -> list[dict]:
    """oracle 궤적을 리플레이 — 매 결정에서 greedy 대조·feature 스냅샷."""
    env = DirectJobEnv(profile, sla_mode=ARM, expected_n_config=cfg.n_external,
                       state_schema=SCHEMA, check_invariants=False)
    greedy = DirectJobRulePolicy(DirectRule.IMMEDIATE_COST_GREEDY)
    state, info = env.reset(scenario)
    rows: list[dict] = []
    for step_idx, job_id in enumerate(oracle_trace):
        cands = {c.job_id: c for c in info.candidates}
        pick = cands[job_id]
        g_pick = greedy.act(state, info.candidates)
        g = pick.global_raw                     # (진행률, bay, 대기, 최장, 30분+)
        f = pick.future_raw                     # (남은수, 남은총량, 짧은비율, 근접거리)
        base = list(g) + list(f) + _set_aggregates(list(cands.values())) \
            + _cand_feats(g_pick)
        diverged = pick.job_id != g_pick.job_id
        row = {
            "step": step_idx, "n_candidates": len(cands),
            "features_context": [float(v) for v in base],   # H-A 입력 (22 dim)
            "diverged": diverged,
        }
        if diverged:
            pair = [b - a for a, b in zip(_cand_feats(g_pick), _cand_feats(pick))]
            row["pair_diff"] = [float(v) for v in pair]      # H-B 입력 P (5 dim)
            row["immediate_sacrifice"] = float(pick.prior_cost - g_pick.prior_cost)
            row["oracle_longer_service"] = pick.estimated_service_s > g_pick.estimated_service_s
            row["oracle_longer_wait"] = pick.wait_s > g_pick.wait_s
            row["oracle_farther"] = pick.reach_s > g_pick.reach_s
            row["direction_switch"] = (pick.transfer_direction
                                       != g_pick.transfer_direction)
        rows.append(row)
        state, _c, _d, info = env.step(job_id)
    return rows


# ------------------------------------------------------- 로지스틱 CV (torch)

def _auc(y: Sequence[int], score: Sequence[float]) -> float:
    pairs = sorted(zip(score, y))
    pos = sum(y)
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum, rank = 0.0, 1
    i = 0
    while i < len(pairs):                       # 동점 평균 순위
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        rank_sum += avg_rank * sum(p[1] for p in pairs[i:j])
        rank += j - i
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def _standardize(train_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    mean = train_x.mean(0, keepdim=True)
    std = train_x.std(0, keepdim=True).clamp_min(1e-6)
    return (x - mean) / std


def logistic_cv_auc(rows_x: list[list[float]], rows_y: list[int],
                    day_of_row: list[int], cfg: PatternConfig,
                    seed_offset: int = 0) -> dict:
    """day-그룹 k-fold CV 로지스틱 AUC (fold 별 표준화·학습 재현 고정)."""
    days = sorted(set(day_of_row))
    folds = [set(days[i::cfg.cv_folds]) for i in range(cfg.cv_folds)]
    scores, labels = [], []
    for fold_idx, held in enumerate(folds):
        tr = [i for i, d in enumerate(day_of_row) if d not in held]
        te = [i for i, d in enumerate(day_of_row) if d in held]
        if not tr or not te or len({rows_y[i] for i in tr}) < 2:
            continue
        torch.manual_seed(cfg.torch_seed + seed_offset * 100 + fold_idx)
        x_tr = torch.tensor([rows_x[i] for i in tr], dtype=torch.float32)
        y_tr = torch.tensor([float(rows_y[i]) for i in tr])
        x_te = torch.tensor([rows_x[i] for i in te], dtype=torch.float32)
        model = torch.nn.Linear(x_tr.shape[1], 1)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.logistic_lr)
        pos_w = torch.tensor(max(1.0, (len(y_tr) - y_tr.sum().item())
                                 / max(1.0, y_tr.sum().item())))
        xn_tr = _standardize(x_tr, x_tr)
        for _ in range(cfg.logistic_steps):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(xn_tr).squeeze(-1), y_tr, pos_weight=pos_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            s = model(_standardize(x_tr, x_te)).squeeze(-1).tolist()
        scores.extend(s if isinstance(s, list) else [s])
        labels.extend(rows_y[i] for i in te)
    return {"auc": _auc(labels, scores), "n": len(labels),
            "positives": int(sum(labels))}


# ---------------------------------------------------------------- 실행기

def run_oracle_pattern(profile_path: str = "configs/terminals/hjnc_armg.yaml",
                       out_dir: str = "outputs/reports/oracle_pattern_hjnc",
                       cfg: PatternConfig | None = None,
                       progress: Callable[[str], None] = print) -> Path:
    cfg = cfg or PatternConfig()
    started = time.time()
    git = _git_state()
    if not cfg.quick and (git["commit"] == "unknown" or git["dirty"]):
        raise RuntimeError("full YR-031-b run requires a clean committed tree")
    profile = load_profile(profile_path)
    params = _gen_params(_ParamShim(cfg))
    ocfg = cfg.oracle_cfg()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress(f"[YR-031-b] profile={profile.terminal_id} W={cfg.beam_width} "
             f"days={cfg.test_episodes}")

    all_rows: list[dict] = []
    day_records: list[dict] = []
    for index, seed in enumerate(cfg.test_seeds, start=1):
        scenario = _scenario(profile, seed, params, cfg.n_external)
        greedy_mean, trace = _greedy_day(profile, scenario, ocfg)
        best, best_trace = beam_day_with_trace(profile, scenario, trace, ocfg)
        rows = collect_day_events(profile, scenario, best_trace, cfg)
        n_div = sum(r["diverged"] for r in rows)
        for r in rows:
            r["seed"] = seed
        all_rows.extend(rows)
        day_records.append({"seed": seed, "greedy_mean": greedy_mean,
                            "best_found_mean": best,
                            "improvement": greedy_mean - best,
                            "decisions": len(rows), "divergences": n_div})
        if index % 10 == 0 or index == cfg.test_episodes:
            progress(f"[pattern] {index}/{cfg.test_episodes}일 — 이탈 "
                     f"{sum(d['divergences'] for d in day_records)}건 누적")
    _json_dump(out / "divergence_events.json", all_rows)
    _json_dump(out / "per_day.json", day_records)

    # ---- H-A: 이탈 시점 예측가능성 (전 결정, day-그룹 CV)
    ha = logistic_cv_auc([r["features_context"] for r in all_rows],
                         [int(r["diverged"]) for r in all_rows],
                         [r["seed"] for r in all_rows], cfg, seed_offset=0)
    ha["verdict"] = ("SUPPORTED" if ha["auc"] >= HA_PREDICTABLE
                     else "PARTIAL" if ha["auc"] >= 0.60 else "REJECTED")

    # ---- H-B: 이탈 선택의 조합 의존 (이탈 이벤트만, P vs S 모델)
    div = [r for r in all_rows if r["diverged"]]
    hb: dict[str, object] = {"n_divergences": len(div)}
    if len(div) >= 30 and len({r["seed"] for r in div}) >= cfg.cv_folds:
        # P: greedy 선택 대비 oracle 선택의 쌍별 feature 차이만으로
        # "이탈 방향" 재구성 가능한가 — 부호 뒤집기 증강으로 이진화
        xs_p, xs_s, ys, ds = [], [], [], []
        for r in div:
            ctx = r["features_context"][:17]          # 전역 5 + 미래 4 + 집합 8
            for sign, label in ((1.0, 1), (-1.0, 0)):
                pair = [sign * v for v in r["pair_diff"]]
                xs_p.append(pair)
                xs_s.append(pair + ctx)
                ys.append(label)
                ds.append(r["seed"])
        model_p = logistic_cv_auc(xs_p, ys, ds, cfg, seed_offset=1)
        model_s = logistic_cv_auc(xs_s, ys, ds, cfg, seed_offset=2)
        gain = model_s["auc"] - model_p["auc"]
        hb.update({"pairwise_auc": model_p["auc"], "set_auc": model_s["auc"],
                   "set_gain": gain,
                   "verdict": "SUPPORTED" if gain >= HB_SET_GAIN else "REJECTED"})
    else:
        hb["verdict"] = "INSUFFICIENT_DATA"

    # ---- 기술 통계: 이탈의 해부
    taxonomy = {}
    if div:
        taxonomy = {
            "oracle_longer_service_pct": 100 * fmean(
                r["oracle_longer_service"] for r in div),
            "oracle_longer_wait_pct": 100 * fmean(
                r["oracle_longer_wait"] for r in div),
            "oracle_farther_pct": 100 * fmean(r["oracle_farther"] for r in div),
            "direction_switch_pct": 100 * fmean(r["direction_switch"] for r in div),
            "median_immediate_sacrifice_min": median(
                r["immediate_sacrifice"] for r in div),
            "divergence_rate_pct": 100 * len(div) / max(1, len(all_rows)),
        }
    tertiles = sorted(day_records, key=lambda d: d["greedy_mean"])
    third = max(1, len(tertiles) // 3)
    congestion = {
        "calm_third": fmean(d["divergences"] for d in tertiles[:third]),
        "mid_third": fmean(d["divergences"] for d in tertiles[third:2 * third]),
        "congested_third": fmean(d["divergences"] for d in tertiles[2 * third:]),
        "prize_share_congested_third": (
            sum(d["improvement"] for d in tertiles[2 * third:])
            / max(1e-9, sum(d["improvement"] for d in tertiles))),
    }

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
            "thresholds": {"ha_predictable_auc": HA_PREDICTABLE,
                           "hb_set_gain": HB_SET_GAIN},
            "elapsed_s": time.time() - started,
        },
        "hypothesis_a": ha, "hypothesis_b": hb,
        "taxonomy": taxonomy, "congestion": congestion,
    }
    _json_dump(out / "oracle_pattern_results.json", payload)
    report = _build_report(payload, out)
    progress(f"[YR-031-b] completed in {payload['manifest']['elapsed_s']:.1f}s "
             f"-> {report} (H-A={ha['verdict']}, H-B={hb['verdict']})")
    return report


@dataclass(frozen=True)
class _ParamShim:
    _cfg: PatternConfig

    @property
    def n_external(self) -> int:
        return self._cfg.n_external

    @property
    def drain_window_s(self) -> float:
        return self._cfg.drain_window_s


def _build_report(payload: dict, out: Path) -> Path:
    ha, hb = payload["hypothesis_a"], payload["hypothesis_b"]
    tax, cong = payload["taxonomy"], payload["congestion"]
    th = payload["manifest"]["thresholds"]
    L: list[str] = []
    L.append("# YR-031-b — Oracle 개선 순서 패턴 분석")
    L.append("")
    L.append("> ⚠ 가정 프로파일 + 합성 시나리오. oracle(전지적 beam)이 greedy 를 이긴")
    L.append("> 결정들의 구조 분석 — 사용자 가설 H-A(feature 예측가능)·H-B(조합 의존) 판정.")
    L.append("")
    L.append(f"- **H-A (이탈 시점이 관측 feature 로 예측되는가)**: AUC "
             f"**{ha['auc']:.3f}** (임계 {th['ha_predictable_auc']}) → **{ha['verdict']}** "
             f"(n={ha['n']}, 이탈 {ha['positives']}건)")
    if "set_gain" in hb:
        L.append(f"- **H-B (이탈 선택이 집합 맥락에 의존하는가)**: 쌍별 AUC "
                 f"{hb['pairwise_auc']:.3f} → +집합맥락 {hb['set_auc']:.3f} "
                 f"(이득 {hb['set_gain']:+.3f}, 임계 {th['hb_set_gain']}) → "
                 f"**{hb['verdict']}**")
    else:
        L.append(f"- **H-B**: {hb['verdict']} (이탈 {hb['n_divergences']}건)")
    L.append("")
    if tax:
        L.append("## 이탈의 해부 (divergence taxonomy)")
        L.append("")
        L.append(f"- 이탈률: 결정의 {tax['divergence_rate_pct']:.1f}%")
        L.append(f"- oracle 이 **더 긴 작업**을 고른 비율: "
                 f"{tax['oracle_longer_service_pct']:.0f}% (anti-SPT)")
        L.append(f"- oracle 이 더 오래 기다린 트럭을 고른 비율: "
                 f"{tax['oracle_longer_wait_pct']:.0f}%")
        L.append(f"- oracle 이 더 먼 작업을 고른 비율: {tax['oracle_farther_pct']:.0f}% "
                 f"(포지셔닝형)")
        L.append(f"- 방향 전환(반입↔반출) 비율: {tax['direction_switch_pct']:.0f}%")
        L.append(f"- 즉시 희생 중앙값: {tax['median_immediate_sacrifice_min']:.4f}분/결정")
        L.append("")
    L.append("## 혼잡도 층화 (일 greedy_mean 3분위)")
    L.append("")
    L.append(f"- 일평균 이탈 수: 한산 {cong['calm_third']:.1f} · 중간 "
             f"{cong['mid_third']:.1f} · 혼잡 {cong['congested_third']:.1f}")
    L.append(f"- 상금의 혼잡 상위 1/3 집중도: "
             f"{100 * cong['prize_share_congested_third']:.0f}%")
    L.append("")
    L.append("*생성: yard_rl.experiments.oracle_pattern — 원자료 "
             "oracle_pattern_results.json·divergence_events.json*")
    path = out / "oracle_pattern_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
