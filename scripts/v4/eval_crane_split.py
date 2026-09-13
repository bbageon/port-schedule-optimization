"""크레인 망 하나를 고른 날들에 넣고 **본선 / 트럭** 비용을 갈라 보여준다.

    PYTHONPATH=src python scripts/v4/eval_crane_split.py \
        --ckpt outputs/v4/crane-continue/crane_071.pt --loads 3500,7500,12500,15000

같은 날을 학습 크레인과 규칙 크레인으로 각각 굴려(짝비교) 네 항목으로 나눈다.
합계만 보면 *"붐비는 날에 진다"* 가 배 때문인지 트럭 때문인지 모른다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from yard_rl.v4.crane.policy import CraneNet
from yard_rl.v4.crane.train import EVAL_SEED_BASE, evaluate, split_table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--loads", default="3500,7500,12500,15000")
    ap.add_argument("--seed-base", type=int, default=EVAL_SEED_BASE)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    loads = tuple(int(x) for x in a.loads.split(",") if x)
    net = CraneNet()
    net.load_state_dict(torch.load(a.ckpt)["crane"])
    ev = evaluate(net, loads=loads, seed_base=a.seed_base)
    print(f"■ {a.ckpt} · 시드 바닥 {a.seed_base:,} · 중앙 {ev['median_gap_ratio']:+.2%} "
          f"· 이긴 날 {ev['n_win']}/{len(ev['rows'])}")
    print(split_table(ev))
    print()
    print("  규칙 크레인의 비용 구성 (그날 합계 대비):")
    for r in ev["rows"]:
        b = r["split_rule"]; t = r["phi_rule"]
        print(f"  {r['load']:>7,}  트럭 {b['truck']/t:5.1%} · 본선 {b['vessel']/t:5.1%} "
              f"· 파내기 {b['rehandle']/t:5.1%}")
    out = Path(a.out or Path(a.ckpt).with_suffix("").parent / f"split_{Path(a.ckpt).stem}.json")
    out.write_text(json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
