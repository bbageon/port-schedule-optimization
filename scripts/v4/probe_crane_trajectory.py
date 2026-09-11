"""회차마다 정책이 얼마나 흔들리나 — **전 체크포인트**를 한 날에 굴린다 ([[YR-309]]).

■ 왜 필요한가
  고정 평가일은 5회차마다 잰다. 그래서 *"5회차 사이에 서서히 나빠졌나"* 와
  *"매 회차 크게 흔들리는데 우연히 그 지점이 나빴나"* 를 못 가린다.
  회차별로 재면 그 둘이 갈린다 — 후자면 원인은 **회차마다 망을 그날 라벨에 다시
  맞추는 것**(라벨을 회차마다 버리므로 버퍼가 없다)이다.

■ 왜 한 날만 굴리나
  전 체크포인트 × 세 날이면 6시간이다. 한 날(부하 7,500)만 보면 1시간이고,
  **흔들림의 크기**를 묻는 데는 그것으로 충분하다. 규칙 바닥은 그 날에 한 번만
  굴리면 된다 — 망과 무관하므로 매번 다시 굴릴 이유가 없다.

    PYTHONPATH=src python scripts/v4/probe_crane_trajectory.py \
        --ckpt-dir outputs/v4/crane-scale-perday --load 7500
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

import torch

from yard_rl.v4.crane.policy import CraneNet
from yard_rl.v4.crane.train import EVAL_SEED_BASE, EVAL_LOADS
from yard_rl.v4.stage.episode import run_episode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="체크포인트별 궤적 — 흔들림의 크기")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--load", type=int, default=7_500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    d = Path(a.ckpt_dir)
    ckpts = sorted(d.glob("crane_*.pt"))
    if not ckpts:
        raise SystemExit(f"체크포인트가 없다: {d}")
    #: ★고정 평가일과 **같은 시드**를 쓴다 — 다른 날을 쓰면 5회차 지점 값과 못 잇는다
    i = EVAL_LOADS.index(a.load) if a.load in EVAL_LOADS else 0
    seed = EVAL_SEED_BASE + i * 10 + a.load // 1_000

    t0 = time.time()
    rule = run_episode(load=a.load, arm="NO_REALLOC", dispatcher="SF_SPT", seed=seed)
    print(f"규칙 바닥 Φ {rule.phi_krw:,.0f}원 ({time.time()-t0:.0f}초) · "
          f"부하 {a.load:,} · 시드 {seed:,}\n")

    rows = []
    for c in ckpts:
        it = int(c.stem.split("_")[1])
        net = CraneNet()
        net.load_state_dict(torch.load(c)["crane"])
        ep = run_episode(load=a.load, arm="NO_REALLOC", dispatcher="RL_CRANE",
                         seed=seed, crane_net=net)
        g = (ep.phi_krw - rule.phi_krw) / max(1e-9, rule.phi_krw)
        rows.append({"it": it, "phi": ep.phi_krw, "gap_ratio": g})
        print(f"  [{it:>3}] {g:>+8.2%}")

    v = [r["gap_ratio"] for r in rows]
    step = [abs(v[i] - v[i - 1]) for i in range(1, len(v))]
    print(f"\n■ 회차 {len(v)}개 · 평균 {st.fmean(v):+.2%} · 최저 {min(v):+.2%} "
          f"· 최고 {max(v):+.2%} · 표준편차 {st.pstdev(v):.2%}")
    print(f"  ★한 회차 만의 변동 — 중앙 {st.median(step):.2%} · 최대 {max(step):.2%}")
    print("  (한 회차 변동이 전체 표준편차만큼 크면 = **매 회차 크게 흔들린다**)")
    out = Path(a.out or (d / f"trajectory_{a.load}.json"))
    out.write_text(json.dumps({"load": a.load, "seed": seed,
                               "phi_rule": rule.phi_krw, "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
