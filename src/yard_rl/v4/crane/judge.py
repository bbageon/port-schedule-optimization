"""크레인 정책 판정 — 규칙 6종 대비 · 판정 대역 · 28일 짝비교 ([[YR-314]]).

    PYTHONPATH=src python -m yard_rl.v4.crane.judge \\
        --ckpt outputs/v4/crane-continue/crane_071.pt --seed 9000700 --workers 10

■ 무엇을 재나
  같은 28일을 **학습 크레인**과 **규칙 크레인 7종**(SF_SPT · FIFO · LIFO · SPT · NEAREST
  · LWKR · RANDOM)으로 각각 굴려, 날마다 짝지어 *"학습이 규칙보다 얼마나 쌌나"* 를
  잰다. 재배정은 **끈다**(`NO_REALLOC`) — 1단계는 크레인 축만이다.

■ 규약 — 재배정 논문([[YR-233]])과 같다
  · 날은 달 계획기(`plan_month`)가 뽑는다 — 실제 달처럼 부하가 섞인다
  · 검정은 부호검정 · 정확 윌콕슨 · 층화 부트스트랩 (`eval/paired.py`)
  · 판정 대역은 **한 번만** — `check_bands` 가 막고, 판정 뒤 `USED_BANDS` 에 박는다
  · 하드가드: 전건 투입 · 정책 예외 0 · 코드 청결 · **반사실 rollout 0**
    (재배정층의 `traded_edges > 0` 가드는 재배정을 끄므로 **적용하지 않는다**)

■ 무엇을 남기나
  `cells.json` (셀마다 Φ·항목별 비용·가드) · `stats.json` (규칙별 검정) · `report.md`
  · 판정에 쓴 망의 sha256 과 코드 커밋.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..dispatch import RL_CRANE
from ..eval.guards import check_bands
from ..eval.paired import paired_summary
from ..stage.month import plan_month

#: 규칙 크레인 — `SF_SPT` 가 주 비교군(가장 강한 규칙), 나머지는 부수 결과
RULE_ARMS = ("SF_SPT", "FIFO", "LIFO", "SPT", "NEAREST", "LWKR", "RANDOM")
N_JUDGE_DAYS = 28
SPLIT_KEYS = ("c_wait", "c_vessel", "c_rehandle", "c_move")
#: ★성과지표 — **턴타임 = 게이트 아웃 − 게이트 인** (사용자 지시 2026-09-22).
#:  비용 총액이 아니라 이것을 앞에 놓는다: 비용은 단위가 크고 항목 구성에 따라
#:  흔들리지만 턴타임은 **현장에서 쓰는 잣대**다. 정의는 `schema/lifecycle.py`.
TURN_KEYS = ("mean_turn_time_s", "p50_turn_time_s", "p90_turn_time_s",
             "n_trucks", "n_censored", "over_ratio")

# ──────────────────────────────────────────────────────────── 작업자
_NET = None
_CKPT = None


def _net_for(ckpt: str):
    """작업자마다 망을 **한 번만** 읽는다."""
    global _NET, _CKPT
    if _NET is None or _CKPT != ckpt:
        import torch
        from .policy import CraneNet
        torch.set_num_threads(1)
        net = CraneNet()
        net.load_state_dict(torch.load(ckpt)["crane"])
        net.eval()
        _NET, _CKPT = net, ckpt
    return _NET


def run_cell(job: dict) -> dict:
    """셀 하나 = (날, 팔). 작업자 프로세스에서 돈다."""
    from ..reward.counterfactual import reset_rollout_calls, rollout_calls
    from ..stage.episode import run_episode

    reset_rollout_calls()
    kw = dict(load=job["load"], arm="NO_REALLOC", seed=job["seed"])
    if job["arm"] == RL_CRANE:
        r = run_episode(dispatcher=RL_CRANE, crane_net=_net_for(job["ckpt"]), **kw)
    else:
        r = run_episode(dispatcher=job["arm"], **kw)
    b = r.breakdown
    return {"day": job["day"], "load": job["load"], "label": job["label"],
            "seed": job["seed"], "arm": job["arm"], "phi": r.phi_krw,
            "split": {k: float(b.get(k, 0.0)) for k in SPLIT_KEYS},
            "turn": {k: float(b.get(k, 0.0)) for k in TURN_KEYS},
            "admitted": r.admitted, "policy_exceptions": r.policy_exceptions,
            "rollout_calls": rollout_calls()}


# ──────────────────────────────────────────────────────────── 가드
def code_state(commit: str | None = None, dirty: bool | None = None) -> dict:
    """코드 커밋과 청결 상태. git 이 없으면 `dirty=None` — 가드는 이것을 **실패**로 본다.

    ★WSL 함정: 이 저장소는 git worktree 라 `.git` 이 **윈도우 경로**를 가리킨다.
      WSL 안에서 git 을 부르면 `fatal: not a git repository` 가 난다(2026-09-15 실측).
      그래서 **띄우는 쪽(윈도우)이 커밋·청결을 재서 넘길 수 있다** — 넘긴 값은
      `source="launcher"` 로 남겨 어디서 왔는지 보이게 한다.
    """
    if commit is not None and dirty is not None:
        return {"commit": commit, "dirty": bool(dirty), "source": "launcher"}
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.STDOUT).strip()
        st = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True,
            stderr=subprocess.STDOUT)
        return {"commit": head, "dirty": bool(st.strip()), "status": st.strip(),
                "source": "git"}
    except Exception as e:                       # noqa: BLE001
        return {"commit": None, "dirty": None, "source": "git",
                "error": str(getattr(e, "output", e))[:300]}


def guard_cell(c: dict) -> list[str]:
    bad = []
    if c["admitted"] != c["load"]:
        bad.append(f"투입 미완 {c['admitted']} != {c['load']}")
    if c["policy_exceptions"] != 0:
        bad.append(f"정책 예외 {c['policy_exceptions']}건")
    if c["rollout_calls"] != 0:
        bad.append(f"교사 누출 — 반사실 rollout {c['rollout_calls']}회")
    return bad


def sha256_of(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ──────────────────────────────────────────────────────────── 본체
def judge(*, ckpt: str, seed: int, out_dir: str | Path, workers: int = 10,
          n_days: int = N_JUDGE_DAYS, arms=RULE_ARMS, log=print,
          code_commit: str | None = None, code_dirty: bool | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    days = plan_month(seed, n_days=n_days)
    bad_band = check_bands([d.seed for d in days] + [seed])
    if bad_band:
        raise RuntimeError("판정 대역 위반:\n  " + "\n  ".join(bad_band))
    code = code_state(code_commit, code_dirty)
    meta = {"ckpt": ckpt, "ckpt_sha256": sha256_of(ckpt), "seed": seed,
            "n_days": n_days, "arms": [RL_CRANE, *arms], "code": code,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "days": [d.as_dict() for d in days]}
    (out / "judge_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    log(f"■ 판정 · 시드 {seed:,} · {n_days}일 · 팔 {len(meta['arms'])} · 작업자 {workers}")
    log(f"  망 {ckpt} sha256 {meta['ckpt_sha256'][:12]}… · 코드 {str(code['commit'])[:10]} "
        f"dirty={code['dirty']}")
    log("  날별 부하: " + " ".join(str(d.load // 1000) for d in days))

    jobs = [{"day": d.index, "load": d.load, "label": d.label, "seed": d.seed,
             "arm": arm, "ckpt": ckpt}
            for d in days for arm in (RL_CRANE, *arms)]
    cells: list[dict] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(run_cell, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            c = f.result()
            cells.append(c)
            if i % 16 == 0 or i == len(jobs):
                log(f"  {i:>3}/{len(jobs)} 셀 · {(time.time()-t0)/60:.1f}분")
    cells.sort(key=lambda c: (c["day"], c["arm"]))
    (out / "cells.json").write_text(json.dumps(cells, ensure_ascii=False, indent=1),
                                    encoding="utf-8")

    # ── 가드
    failures = {f"{c['day']}/{c['arm']}": guard_cell(c) for c in cells}
    failures = {k: v for k, v in failures.items() if v}
    if code["dirty"] is None:
        failures["code"] = ["code_dirty=None — git 상태를 못 읽었다"]
    elif code["dirty"]:
        failures["code"] = ["실행 트리가 dirty 하다: " + code["status"]]
    n_expected = n_days * len(meta["arms"])
    if len(cells) != n_expected:
        failures["matrix"] = [f"셀 {len(cells)} != {n_expected}"]

    # ── 통계
    phi = {arm: {} for arm in meta["arms"]}
    split = {arm: {} for arm in meta["arms"]}
    turn = {arm: {} for arm in meta["arms"]}
    for c in cells:
        phi[c["arm"]][c["day"]] = c["phi"]
        split[c["arm"]][c["day"]] = c["split"]
        turn[c["arm"]][c["day"]] = c.get("turn", {})
    strata = {d.index: d.label for d in days}
    stats = {}
    #: ★턴타임 짝비교 — 비용과 **같은 방식**으로(날마다 짝지어 부호검정·윌콕슨).
    tt = {arm: {d: v.get("mean_turn_time_s", 0.0) for d, v in turn[arm].items()}
          for arm in meta["arms"]}
    p90 = {arm: {d: v.get("p90_turn_time_s", 0.0) for d, v in turn[arm].items()}
           for arm in meta["arms"]}
    for arm in arms:
        s = paired_summary(phi[RL_CRANE], phi[arm], strata)
        s["turn_mean"] = paired_summary(tt[RL_CRANE], tt[arm], strata)
        s["turn_p90"] = paired_summary(p90[RL_CRANE], p90[arm], strata)
        # 부하 구간별 · 항목별 평균 차이
        by_label: dict = {}
        for k in s["days"]:
            by_label.setdefault(strata[k], []).append(
                (phi[RL_CRANE][k] - phi[arm][k]) / max(1e-9, phi[arm][k]))
        s["by_label_mean_ratio"] = {lab: sum(v) / len(v) for lab, v in by_label.items()}
        s["split_mean_diff_krw"] = {
            key: sum(split[RL_CRANE][k][key] - split[arm][k][key] for k in s["days"])
            / max(1, len(s["days"])) for key in SPLIT_KEYS}
        stats[arm] = s
    result = {"meta": meta, "guard_failures": failures, "stats": stats,
              "secs": time.time() - t0}
    (out / "stats.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    (out / "report.md").write_text(render_report(result), encoding="utf-8")
    log(render_report(result))
    return result


def render_report(res: dict) -> str:
    m, st = res["meta"], res["stats"]
    L = [f"# 크레인 판정 — 시드 {m['seed']:,} · {m['n_days']}일 · 망 {m['ckpt_sha256'][:12]}…",
         "", f"코드 `{str(m['code']['commit'])[:10]}` · dirty={m['code']['dirty']} · "
         f"{res['secs']/3600:.2f}시간", ""]
    if res["guard_failures"]:
        L += ["## ⚠️ 가드 실패 — **판정 미성립**", ""]
        for k, v in res["guard_failures"].items():
            L.append(f"- {k}: {'; '.join(v)}")
        L.append("")
    #: ★턴타임을 **앞에** 둔다 — 성과지표가 이것이다 (사용자 지시 2026-09-22)
    L += ["## ① 턴타임 — 게이트 아웃 − 게이트 인 (음수 = 학습이 짧다)", "",
          "| 규칙 | 짧은 날 | 부호검정 p | 평균 턴타임 감소 | 90분위 감소 |",
          "|---|---|---|---|---|"]
    for arm, s in st.items():
        a, b = s.get("turn_mean"), s.get("turn_p90")
        if a is None:
            L.append(f"| {arm} | (이 실행에는 턴타임 기록 없음) | | | |")
            continue
        L.append(f"| {arm} | {a['sign']['win']}/{a['sign']['n']} | {a['sign']['p']:.4f} | "
                 f"{a['mean_diff_krw']/60:+.1f}분 ({a['median_ratio']:+.2%}) | "
                 f"{b['mean_diff_krw']/60:+.1f}분 ({b['median_ratio']:+.2%}) |")
    L += ["", "## ② 비용 — 학습 − 규칙 (음수 = 학습이 쌌다)", "",
          "| 규칙 | 이긴 날 | 부호검정 p | 윌콕슨 p | 감소율 중앙 | 평균 | 95% 구간(하루 평균 차이·백만원) |",
          "|---|---|---|---|---|---|---|"]
    for arm, s in st.items():
        b = s["boot"]
        L.append(f"| {arm} | {s['sign']['win']}/{s['sign']['n']} | {s['sign']['p']:.4f} | "
                 f"{s['wilcoxon']['p']:.4f} | {s['median_ratio']:+.2%} | {s['mean_ratio']:+.2%} | "
                 f"[{b['lo']/1e6:+.1f}, {b['hi']/1e6:+.1f}] |")
    L += ["", "## ③ 부하 구간별 평균 비용 격차 (학습 − 규칙)", "",
          "| 규칙 | " + " | ".join(next(iter(st.values()))["by_label_mean_ratio"]) + " |",
          "|---|" + "---|" * len(next(iter(st.values()))["by_label_mean_ratio"])]
    for arm, s in st.items():
        L.append(f"| {arm} | " + " | ".join(f"{v:+.2%}" for v in s["by_label_mean_ratio"].values()) + " |")
    L += ["", "## ④ 비용 항목별 하루 평균 차이 (학습 − 규칙 · 백만원)", "",
          "| 규칙 | 트럭 대기 | 본선 유휴 | 파내기 | 이동 |", "|---|---|---|---|---|"]
    for arm, s in st.items():
        d = s["split_mean_diff_krw"]
        L.append(f"| {arm} | {d['c_wait']/1e6:+.2f} | {d['c_vessel']/1e6:+.2f} | "
                 f"{d['c_rehandle']/1e6:+.2f} | {d['c_move']/1e6:+.2f} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="yard_rl.v4.crane.judge",
                                 description="크레인 정책 판정 — 규칙 6종 대비 28일 짝비교")
    ap.add_argument("--ckpt", required=True, help="판정할 망 (체크포인트)")
    ap.add_argument("--seed", type=int, required=True, help="판정 대역 시드 — 한 번만 쓴다")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--days", type=int, default=N_JUDGE_DAYS)
    ap.add_argument("--arms", default=",".join(RULE_ARMS))
    ap.add_argument("--out", default="outputs/v4/judge-crane")
    ap.add_argument("--code-commit", default=None,
                    help="띄우는 쪽에서 잰 코드 커밋 (WSL 안에서는 git 이 worktree 를 못 읽는다)")
    ap.add_argument("--code-dirty", default=None, choices=("0", "1"),
                    help="띄우는 쪽에서 잰 청결 상태 — 0 이면 깨끗, 1 이면 dirty")
    a = ap.parse_args(argv)
    judge(ckpt=a.ckpt, seed=a.seed, out_dir=a.out, workers=a.workers,
          n_days=a.days, arms=tuple(x for x in a.arms.split(",") if x),
          code_commit=a.code_commit,
          code_dirty=(None if a.code_dirty is None else a.code_dirty == "1"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
