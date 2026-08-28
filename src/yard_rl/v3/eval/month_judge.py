"""30일 무대 판정 — **같은 달을 팔만 바꿔 다시 굴려** 날 단위로 짝비교한다 ([[YR-239]]).

■ 왜 이 모양인가
  30일은 **세계가 하나**다. 하루 무대처럼 "같은 시드로 안 팔기를 한 번 더" 를
  에피소드 안에서 할 수가 없다 — 야드가 갈라지면 5일째부터는 다른 세계다.

  그래서 **달 전체를 팔마다 한 번씩** 굴린다. 같은 시드·같은 도착 명단·같은 본선
  이므로 날 d 의 Φ 는 팔끼리 짝이 맞는다. 갈라지는 것은 오직 정책의 선택뿐이다.

■ 판정식 — **부호검정** (06 §3 · [[YR-228]])
  회차 격차는 판정력이 없다([[YR-228]] — 전부 |t| < 2). 날 단위로 *"어느 쪽이
  쌌나"* 를 세고 이항검정을 한다. 크기가 아니라 **부호**를 세므로 이상치에 안 흔들린다.

■ ★부하별로 따로 낸다 (사용자 지시 2026-08-26)
  가중 추첨이라 부하마다 날 수가 다르다 — 초혼잡은 28일 중 1~3일뿐이다.
  **표본이 얇으면 그 사실을 결과에 붙여 보고한다.** 얇다고 빼지 않고, 얇은 채로 적는다.
"""
from __future__ import annotations

import json
import math
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

from ..stage.month import N_DAYS, plan_month
from ..stage.month_run import run_month

#: 주판정 상대 — 고전 규칙 5종 + 안 팔기. 진단 팔(RL_SPACE·RL_TIME)은 따로.
#: `NEAREST` 제외 — 목적지가 20/21 블록에서 `Y01` 로 고정되는 구조 결함
#: (`actors/classical.py` 머리말 · 사용자 결정 2026-08-28).
JUDGE_ARMS = ("NO_REALLOC", "FCFS", "SPT", "LEAST_SLACK", "NETGAIN")
ALPHA = 0.05

#: ★"초혼잡"의 문턱 — 이 부하 이상인 날의 **다음 날**을 따로 묶는다.
HEAVY_LOAD = 12_500


def sign_test(diffs) -> dict:
    """양측 부호검정 — `diff < 0` 이 *"RL 이 쌌다"* 다.

    동점(0)은 **버린다**(Wilcoxon 관례). 버린 수도 함께 돌려준다 — 조용히 사라지면
    표본 수가 부풀어 보인다.
    """
    d = [x for x in diffs if abs(x) > 1e-9]
    ties = len(diffs) - len(d)
    n = len(d)
    win = sum(1 for x in d if x < 0)
    if n == 0:
        return {"n": 0, "win": 0, "ties": ties, "p": 1.0, "median": 0.0}
    # 이항 양측 p — n 이 작으므로 정확검정을 그대로 쓴다
    k = min(win, n - win)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    p = min(1.0, 2.0 * tail)
    srt = sorted(d)
    med = (srt[n // 2] if n % 2 else 0.5 * (srt[n // 2 - 1] + srt[n // 2]))
    return {"n": n, "win": win, "ties": ties, "p": p, "median": med}


@dataclass
class ArmMonth:
    """팔 하나가 굴린 한 달."""

    arm: str
    phi_by_day: dict = field(default_factory=dict)     # {날 index: Φ}
    traded: int = 0
    n_space: int = 0
    n_time: int = 0


def _run_arm(kw) -> ArmMonth:
    """작업자 프로세스에서 도는 함수 — 팔 하나로 달 하나를 굴린다.

    `_label` 은 결과를 부를 이름이다 — 같은 `arm="RL"` 을 **다른 정책**으로 두 번
    굴릴 때(학습 전/후) 이름이 겹치지 않게 한다.
    """
    kw = dict(kw)
    label = kw.pop("_label", kw["arm"])
    res = run_month(**kw)
    return ArmMonth(arm=label,
                    phi_by_day={d.index: d.phi_krw for d in res.days},
                    traded=res.traded_edges, n_space=res.n_space,
                    n_time=res.n_time)


def judge_month(*, seed: int, seller_net=None, buyer_net=None,
                arms=JUDGE_ARMS, n_days: int = N_DAYS, days=None,
                trigger_k: dict | None = None, workers: int = 0,
                extra_policies=None, ckpt_dir=None, log=print) -> dict:
    """`RL` 과 각 팔을 **같은 달** 위에서 겨룬다. 부하별로 표를 낸다.

    `workers` — 팔을 몇 프로세스로 나눌까. 0 이면 팔 수만큼(달 하나는 단일 스레드다).
    `trigger_k` — 고전 팔의 트리거 상위 비율 `{팔: k}`. 없으면 기본값.
    """
    days = list(days) if days else plan_month(seed, n_days=n_days)
    todo = ("RL",) + tuple(a for a in arms if a != "RL")
    base = dict(seed=seed, days=days, seller_net=seller_net,
                buyer_net=buyer_net)
    jobs = []
    for a in todo:
        kw = dict(base, arm=a, _label=a)
        if trigger_k and a in trigger_k:
            kw["trigger_top_k"] = float(trigger_k[a])
        jobs.append(kw)
    # ★같은 팔(RL)을 **다른 정책**으로 한 번 더 — 학습 전/후를 ε=0 에서 견준다.
    #   탐색이 둘 다 0 이라 "탐색이 줄어 좋아졌다" 와 "배워서 좋아졌다" 가 갈린다.
    for name, (s_n, b_n) in (extra_policies or {}).items():
        jobs.append(dict(base, arm="RL", _label=name,
                         seller_net=s_n, buyer_net=b_n))
        todo = todo + (name,)

    log(f"■ 판정 — 팔 {len(jobs)}개 × {len(days)}일 (시드 {seed:,})")

    # ★팔 하나가 끝날 때마다 **바로 저장**한다 (2026-08-28 사고).
    #   전에는 `ex.map` 이 전부 끝나야 돌아와서, 5시간 23분을 굴린 판정이 죽었을 때
    #   **한 줄도 안 남았다.** 이제 끝난 팔은 건지고, 다시 돌리면 **건너뛴다**.
    ck = pathlib.Path(ckpt_dir) if ckpt_dir else None
    if ck:
        ck.mkdir(parents=True, exist_ok=True)

    def _load(label):
        if not ck:
            return None
        f = ck / f"arm_{label}.json"
        if not f.exists():
            return None
        d = json.loads(f.read_text(encoding="utf-8"))
        return ArmMonth(arm=d["arm"],
                        phi_by_day={int(k): v for k, v in d["phi_by_day"].items()},
                        traded=d["traded"], n_space=d["n_space"], n_time=d["n_time"])

    def _save(g):
        if ck:
            (ck / f"arm_{g.arm}.json").write_text(
                json.dumps({"arm": g.arm, "phi_by_day": g.phi_by_day,
                            "traded": g.traded, "n_space": g.n_space,
                            "n_time": g.n_time}, ensure_ascii=False),
                encoding="utf-8")

    got, pending_jobs = [], []
    for k in jobs:
        cached = _load(k.get("_label", k["arm"]))
        if cached is not None:
            log(f"  · {cached.arm} — 이미 있음(건너뜀)")
            got.append(cached)
        else:
            pending_jobs.append(k)

    n_w = workers or max(1, len(pending_jobs))
    if pending_jobs:
        if n_w <= 1:
            for k in pending_jobs:
                g = _run_arm(k)
                _save(g)
                log(f"  · {g.arm} 완료")
                got.append(g)
        else:
            with ProcessPoolExecutor(max_workers=n_w) as ex:
                futs = [ex.submit(_run_arm, k) for k in pending_jobs]
                for f in as_completed(futs):
                    g = f.result()
                    _save(g)
                    log(f"  · {g.arm} 완료 ({len(got)+1}/{len(jobs)})")
                    got.append(g)
    by_arm = {g.arm: g for g in got}

    train = [d for d in days if d.is_train]
    rl = by_arm["RL"]
    # ★**RL 이 거래를 안 했으면 판정이 무의미하다** — "무승부" 가 *"비겼다"* 가 아니라
    #   *"아무것도 안 했다"* 라는 뜻이 된다. 실측(2026-08-26): 학습 전 무작위 망은 시드에
    #   따라 5,179 결정에 **거래 3건**만 내기도 한다(같은 코드가 다른 시드에서는 1,255건).
    #   조용히 넘기면 표에 `p=1.0000 무승부` 로 찍혀 **안 한 것과 못 이긴 것이 안 갈린다.**
    #   문턱은 **다른 팔에 견줘** 잡는다 — 절대값은 부하마다 달라 못 쓴다.
    #   고전 팔은 규칙상 반드시 거래하므로 그 중앙값이 "정상 거래량" 의 눈금이 된다.
    peer = sorted(by_arm[a].traded for a in todo[1:] if a != "NO_REALLOC")
    ref = peer[len(peer) // 2] if peer else 0
    silent = rl.traded == 0 or (ref > 0 and rl.traded < 0.05 * ref)
    out: dict = {"seed": seed, "n_train": len(train), "arms": {},
                 "by_load": {}, "rl_traded": rl.traded,
                 "peer_traded": ref, "rl_silent": bool(silent)}
    if silent:
        log(f"⚠️ ★RL 이 거래를 거의 안 했다 — {len(train)}일에 **{rl.traded}건** "
            f"(고전 팔 중앙 {ref:,}건). "
            f"아래 '무승부' 는 *비겼다* 가 아니라 *아무것도 안 했다* 는 뜻이다. "
            f"학습된 체크포인트(`--ckpt`)로 다시 돌려라.")
    for a in todo[1:]:
        other = by_arm[a]
        diffs = [rl.phi_by_day[d.index] - other.phi_by_day[d.index] for d in train]
        out["arms"][a] = sign_test(diffs)

    def _slice(ds, key):
        row = {"n_days": len(ds), "label": key, "thin": len(ds) < 4, "arms": {}}
        for a in todo[1:]:
            other = by_arm[a]
            diffs = [rl.phi_by_day[d.index] - other.phi_by_day[d.index] for d in ds]
            row["arms"][a] = sign_test(diffs)
        row["beaten"] = sum(1 for a, r in row["arms"].items()
                            if r["p"] < ALPHA and r["median"] < 0)
        return row

    loads = sorted({d.load for d in train})
    for load in loads:
        ds = [d for d in train if d.load == load]
        out["by_load"][load] = _slice(ds, ds[0].label)

    # ★"초혼잡 다음 날" 축 (사용자 지적 2026-08-28) — 밀린 일감을 누가 잘 푸나.
    #   하루 무대는 이 축이 **원리적으로 없다**(매일 아침 야드가 리셋된다).
    prev = {d.index: days[d.index - 1].load if d.index > 0 else 0 for d in train}
    after = [d for d in train if prev[d.index] >= HEAVY_LOAD]
    calm = [d for d in train if prev[d.index] < HEAVY_LOAD]
    out["by_wake"] = {}
    if after:
        out["by_wake"]["초혼잡 다음 날"] = _slice(after, "초혼잡 다음 날")
    if calm:
        out["by_wake"]["평상 다음 날"] = _slice(calm, "평상 다음 날")

    out["traded"] = {a: by_arm[a].traded for a in todo}
    _log_table(out, log)
    return out


def _log_table(out: dict, log) -> None:
    tag = "  ⚠️RL 거래 거의 없음" if out.get("rl_silent") else ""
    log(f"■ 전체 (학습 {out['n_train']}일 · RL 거래 {out.get('rl_traded', 0):,}건{tag})")
    for a, r in out["arms"].items():
        mark = "★승" if (r["p"] < ALPHA and r["median"] < 0) else (
            "패" if (r["p"] < ALPHA) else "무승부")
        log(f"   RL vs {a:<12} {r['win']:>2}/{r['n']:<2} "
            f"중앙 {r['median']:>+15,.0f}원 p={r['p']:.4f}  {mark}")
    def _rows(title, table, fmt):
        log(title)
        for k, row in table.items():
            thin = "  ⚠️표본 얇음" if row["thin"] else ""
            log(f" {fmt(k, row)} · {row['n_days']}일 · "
                f"RL 이 넘은 팔 **{row['beaten']}/{len(row['arms'])}**{thin}")
            for a, r in row["arms"].items():
                mark = "★" if (r["p"] < ALPHA and r["median"] < 0) else " "
                log(f"     {mark} {a:<12} {r['win']:>2}/{r['n']:<2} "
                    f"중앙 {r['median']:>+15,.0f}원 p={r['p']:.4f}")

    _rows("■ 부하별 (★사용자 지시 — 얇은 표본도 그대로 적는다)", out["by_load"],
          lambda k, r: f"부하 {k:>6,} ({r['label']})")
    if out.get("by_wake"):
        _rows("■ ★밀림별 — 초혼잡이 남긴 일감을 누가 잘 푸나 (30일 무대에만 있는 축)",
              out["by_wake"], lambda k, r: f"{k:<14}")
