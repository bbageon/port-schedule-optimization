"""Rebuild the common, policy-independent thirty-day fixed cargo manifest."""
import argparse
import json
from pathlib import Path

from yard_rl.v7.stage.fixed_seed import build_fixed_seed, sha
from yard_rl.v7.stage.month import (build_month, initial_stock_by_block, plan_month,
                                    plan_month_vessels, truck_net_by_block)
from yard_rl.v7.stage.seed_bundle import save_seed_bundle
from yard_rl.v7.world.integrated.yard_layout import terminal_layout

SEED = 9400777
OUT = Path("outputs/reports/yr332_month_cranes")
MANIFEST = Path("build/yr332_month_cranes/fixed.json.gz")


def build(seed, days, path):
    built = build_month(seed, days=days)
    vessels = plan_month_vessels(days, terminal_layout(), truck_net=truck_net_by_block(
        built["schedule"]), initial_stock=initial_stock_by_block(built))
    document, audit = build_fixed_seed(built, vessels, seed=seed)
    if not audit["passed"]:
        raise ValueError("Invalid cargo manifest")
    if seed == SEED:
        previous = json.loads(Path("outputs/reports/yr332_month_supply/baseline.json").read_text("utf8"))
        if sha(document) != previous["document_sha256"]:
            raise ValueError("Common month differs from the registered cargo manifest")
    return save_seed_bundle(document, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        from yard_rl.v7.ppo.continuous import make_plan
        seed = 9900306
        days = make_plan(seed, 3, 60)
        path = MANIFEST.with_name("smoke.json.gz")
    else:
        seed, days, path = SEED, plan_month(SEED), MANIFEST
    info = build(seed, days, path)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("smoke-input.json" if args.smoke else "input.json")).write_text(
        json.dumps(dict(seed=seed, days=[d.as_dict() for d in days], **info), indent=2), "utf8")
