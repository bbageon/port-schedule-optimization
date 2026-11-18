"""YR-332-ba: separate confirmed source times from unconfirmed gate records.

Only aggregate profiles/reports are public. Quarantine is written under ignored
outputs/private, never under the original source or reports directories.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics as st

from calibrate_order_information import read, stamp as booking_time
from gate_time_provenance import CONFIRMED, load_evidence, parse_time, partition
from recheck_transport_queue import digest, inventory
from yard_rl.integrated.repro import repro_stamp

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "outputs/reports/yr332_gate_time_provenance"
PRIVATE = BASE / "outputs/private/yr332_gate_time_provenance"
PREREG = "docs/research/v7-discharge/gate-time-provenance-prereg.md"
OLD_PROFILE = BASE / "outputs/reports/yr332_real_order_information/profile.json"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8", newline="\n")


def stats(values):
    xs = sorted(values)
    if not xs:
        return dict(n=0)
    return dict(n=len(xs), mean=st.mean(xs), median=st.median(xs), min=xs[0], max=xs[-1],
                p90=xs[round((len(xs)-1)*.9)], p99=xs[round((len(xs)-1)*.99)], negative=sum(x < 0 for x in xs))


def collect_pairs(visits, orders, proof):
    by_doc = defaultdict(list)
    for row in orders:
        notice, reserved = booking_time(row["log_time"]), booking_time(row["in_out_reserve_time"])
        if row["copino_err_type"] == "NM" and notice and reserved and reserved.year >= 2000:
            by_doc[row["doc_key"]].append((notice, reserved, row["in_out"]))
    groups = {name:defaultdict(list) for name in ("legacy", "confirmed", "unconfirmed")}
    rejected = Counter()
    for row in visits:
        arrival = parse_time(row.get("gate_in_time"))
        if arrival is None:
            rejected["no_gate_entry"] += 1
            continue
        candidates = sorted({item for doc in re.split(r"[|;,]", row["doc_keys"])
            for item in by_doc.get(doc.strip(), []) if item[0] <= arrival and item[2] == row["in_out"]})
        if not candidates:
            rejected["no_matching_pre_entry_valid_reservation"] += 1
            continue
        notice, reserved, flow = candidates[0]
        first = booking_time(row["copino_first_time"])
        if first is None or notice != first:
            rejected["first_notice_not_observed_with_valid_reservation"] += 1
            continue
        date = arrival.date().isoformat()
        if not "2026-10-28" <= date <= "2026-11-09":
            raise ValueError("Source date outside frozen calibration/check period")
        cohort = ("calibration" if date <= "2026-11-05" else "holdout", flow)
        pair = (int((arrival-reserved).total_seconds()), int((arrival-notice).total_seconds()))
        groups["legacy"][cohort].append(pair)
        arm = "confirmed" if proof.classify(row) in CONFIRMED else "unconfirmed"
        groups[arm][cohort].append(pair)
    return groups, dict(rejected)


def flows_for(cohorts):
    flows = {}
    for flow, label in (("IN", "GATE_IN"), ("OUT", "GATE_OUT")):
        pairs = cohorts.get(("calibration", flow), [])
        if not pairs:
            raise ValueError("Empty confirmed calibration flow")
        flows[label] = dict(n=len(pairs), eta_offset_s=st.median(p[0] for p in pairs),
            samples=[[e,lead,n] for (e,lead),n in sorted(Counter(pairs).items())])
    return flows


def aggregate_orders(rows):
    by_day, minutes, flows = defaultdict(lambda: {"IN":[0]*24,"OUT":[0]*24}), defaultdict(Counter), Counter()
    hours = [0]*24
    for row in rows:
        at = parse_time(row["gate_in_time"]); day = at.date().isoformat(); flow = row["in_out"]
        by_day[day][flow][at.hour] += 1; hours[at.hour] += 1; flows[flow] += 1
        minutes[(day, at.hour)][at.minute] += 1
    return dict(n=len(rows), flow_counts=dict(flows), gate_out_share=flows["OUT"]/len(rows), hour_counts=hours,
        per_date=[dict(date=day, orders=sum(map(sum,fs.values())), gate_out_share=sum(fs["OUT"])/sum(map(sum,fs.values())),
                       hour_counts=[fs["IN"][h]+fs["OUT"][h] for h in range(24)], by_flow=fs)
                  for day,fs in sorted(by_day.items())],
        minute_counts={f"{day}:{hour:02d}":[c[m] for m in range(60)] for (day,hour),c in sorted(minutes.items())})


def visit_patterns(visits):
    from audit_visit_links import pattern_of
    patterns, unknown, over = Counter(), Counter(), Counter()
    for visit in visits:
        p = pattern_of(visit["rows"])
        if p is None:
            unknown[str(len(visit["rows"]))] += 1
        elif sum(s[0] for s in p) > 2 or sum(s[1] for s in p) > 2:
            over["|".join(map(str,p))] += 1
        else:
            patterns[p] += 1
    return dict(patterns=[dict(stops=[list(s) for s in p], count=c) for p,c in sorted(patterns.items())],
                known=sum(patterns.values()), unknown=dict(unknown), over_capacity=dict(over))


def demand_impacts(entered, confirmed, out, source_manifest):
    from audit_visit_units import gate_records, physical_visits, templates, MERGE_S, dispersion
    before, after = aggregate_orders(entered), aggregate_orders(confirmed)
    old_visits = physical_visits(gate_records(entered), MERGE_S)
    new_visits = physical_visits(gate_records(confirmed), MERGE_S)
    old_templates = templates(old_visits, source_manifest)
    new_templates = templates(new_visits, source_manifest)
    old_patterns, new_patterns = visit_patterns(old_visits), visit_patterns(new_visits)
    saved_templates = json.loads((BASE/"outputs/reports/yr332_visit_arrivals/visit-arrival-templates.json").read_text(encoding="utf-8"))
    saved_links = json.loads((BASE/"outputs/reports/yr332_visit_links/link-definition.json").read_text(encoding="utf-8"))
    saved_demand = json.loads((BASE/"outputs/reports/yr332_order_representativeness/demand-profiles.json").read_text(encoding="utf-8"))["profiles"]["observed_mix_hours"]
    checks = dict(legacy_minute_templates_reproduced=old_templates["minute_counts"] == saved_templates["minute_counts"],
        legacy_visit_patterns_reproduced=old_patterns["patterns"] == saved_links["spec"]["patterns"],
        legacy_hour_counts_reproduced=before["hour_counts"] == saved_demand["hour_counts"],
        legacy_flow_share_reproduced=before["gate_out_share"] == saved_demand["gate_out_share"])
    if not all(checks.values()):
        raise ValueError("Legacy arrival calibration could not be reproduced")
    new_templates["definition"] += "; only original-request-confirmed gate times retained before grouping"
    new_templates["scope"] = "Candidate recalibration only; not applied to the frozen synthetic arrival generator in ba."
    spec = dict(saved_links["spec"], patterns=new_patterns["patterns"],
        source=dict(definition="Same ak/al grouping/pattern rules, after explicit gate-time verification",
                    scope="Candidate only; all 13 development dates already viewed"))
    save(out/"visit-arrival-templates-confirmed.json", new_templates)
    save(out/"visit-patterns-confirmed.json", spec)
    changes = []
    for old,new in zip(before["per_date"],after["per_date"]):
        assert old["date"] == new["date"]
        changes.append(dict(date=old["date"], before=old["orders"], after=new["orders"],
                            removed=old["orders"]-new["orders"], gate_out_share_delta_pp=100*(new["gate_out_share"]-old["gate_out_share"])))
    result = dict(checks=checks, order_records=dict(before=before, confirmed=after, per_date_changes=changes,
        gate_out_share_delta_pp=100*(after["gate_out_share"]-before["gate_out_share"]),
        max_hour_share_delta_pp=max(abs(100*(a/after["n"]-b/before["n"])) for a,b in zip(after["hour_counts"],before["hour_counts"]))),
        visits=dict(before=len(old_visits), confirmed=len(new_visits),
            final_order_rows_before=sum(len(v["rows"]) for v in old_visits),
            final_order_rows_confirmed=sum(len(v["rows"]) for v in new_visits),
            merged_gate_records_before=sum(len(v["superseded"]) for v in old_visits),
            merged_gate_records_confirmed=sum(len(v["superseded"]) for v in new_visits),
            dispersion_1min_before=dispersion([v["arrival"] for v in old_visits],60),
            dispersion_1min_confirmed=dispersion([v["arrival"] for v in new_visits],60),
            changed_minute_cells=sum(a!=b for h in old_templates["minute_counts"]
                for ad,bd in zip(new_templates["minute_counts"][h],old_templates["minute_counts"][h]) for a,b in zip(ad,bd))),
        patterns=dict(before=old_patterns, confirmed=new_patterns),
        interpretation="Separately measured dependency impacts. Only reservation/notice timing is applied in the ba input integration check.")
    save(out/"arrival-impacts.json", result)
    return result


def build(root, out, private):
    root, out, private = root.resolve(), out.resolve(), private.resolve()
    allowed = (BASE/"outputs/private").resolve()
    if not private.is_relative_to(allowed) or out.is_relative_to(root) or out == root:
        raise ValueError("Private quarantine must stay in ignored outputs/private; original data is read-only")
    if (out/"profile-confirmed.json").exists() or (private/"unconfirmed-visits.jsonl").exists():
        raise FileExistsError("Use a new output directory; previous calibration is immutable")
    manifest, comparison = inventory(root)
    if comparison["unchanged"] != 600 or comparison["files"] != 600 or any(comparison[k] for k in ("changed","new","missing")):
        raise ValueError("Original collection changed; review source contract before calibration")
    proof = load_evidence((root/"기사/_raw").glob("*.gz"))
    visits = read(sorted((root/"기사/방문").glob("*.csv")))
    orders = read(sorted((root/"기사/오더").glob("*.csv")))
    if len({r["visit_id"] for r in visits}) != len(visits):
        raise ValueError("Duplicate source visit IDs")
    parts = partition(visits, proof)
    confirmed = [r for r in visits if proof.classify(r) in CONFIRMED]
    uncertain = [r for r in visits if proof.classify(r).startswith("unconfirmed_")]
    groups, rejected = collect_pairs(visits, orders, proof)
    old_profile = json.loads(OLD_PROFILE.read_text(encoding="utf-8"))
    legacy_match = flows_for(groups["legacy"]) == old_profile["flows"]
    if not legacy_match:
        raise ValueError("Legacy full timing histogram did not reproduce")
    source_path = out/"source-manifest.json"
    save(source_path, manifest)
    profile = dict(old_profile, flows=flows_for(groups["confirmed"]),
        selection=old_profile["selection"]+"; gate time must be explicitly present in a matching original GateIn request",
        source_event_contract=dict(schema="explicit-gate-request-time-v1", log_clock_substitution=False,
            other_message_confirmation=True, error_flag_is_not_selection_rule=True,
            source_manifest_sha256=digest(source_path), legacy_profile_sha256=digest(OLD_PROFILE)))
    save(out/"profile-confirmed.json", profile)
    cohort_report = {}
    for cohort in sorted(groups["legacy"]):
        arms = {}
        for arm,rows_by_cohort in groups.items():
            pairs = rows_by_cohort[cohort]
            arms[arm] = dict(n=len(pairs), error_seconds=stats([p[0] for p in pairs]),
                            notice_lead_seconds=stats([p[1] for p in pairs]))
        cohort_report["_".join(cohort)] = arms
        assert Counter(groups["legacy"][cohort]) == Counter(groups["confirmed"][cohort])+Counter(groups["unconfirmed"][cohort])
    save(out/"calibration-comparison.json", dict(legacy_histogram_reproduced=legacy_match,
        legacy_booking_exclusions=rejected, cohorts=cohort_report,
        heldout_used_in_fit=False, retained_tails=True))
    private.mkdir(parents=True, exist_ok=True)
    quarantine = private/"unconfirmed-visits.jsonl"
    with quarantine.open("w", encoding="utf-8", newline="\n") as handle:
        for row in uncertain:
            handle.write(json.dumps(dict(status=proof.classify(row), source_row=row),ensure_ascii=False,sort_keys=True)+"\n")
    status_counts = {name:len(rows) for name,rows in parts.items()}
    save(out/"provenance-summary.json", dict(status_counts=status_counts, raw_counts=dict(proof.counts),
        explicit_keys=len(proof.explicit), log_clock_keys=len(proof.log_only), source_files=comparison,
        confirmed_error_flags="Not used for selection: an error flag alone cannot prove/disprove entry.",
        private_quarantine=dict(path=quarantine.relative_to(BASE).as_posix(), rows=len(uncertain),sha256=digest(quarantine)),
        no_gate_time_rows_preserved_in_source=len(parts["no_gate_time"])))
    impact = demand_impacts([r for r in visits if parse_time(r["gate_in_time"])], confirmed, out,
                           {m["path"]:m["sha256"] for m in manifest if m["path"].startswith("기사/방문/")})
    config = dict(raw_files=600, original_rows=len(visits), strict_event_contract=profile["source_event_contract"],
        calibration_dates=profile["calibration_dates"], holdout_dates=profile["holdout_dates"],
        synthetic_input_seeds=[9975100,9975101,9975102], loads=[3500,7500,15000],
        applied_axis="reservation_notice_only", performance_evaluation=False)
    code_paths = [Path(__file__), BASE/"scripts/v7/gate_time_provenance.py", BASE/PREREG,
                  BASE/"scripts/v7/calibrate_order_information.py", BASE/"scripts/v7/audit_visit_units.py",
                  BASE/"scripts/v7/audit_visit_links.py"]
    save(out/"build-record.json", dict(configuration=config,
        repro=repro_stamp(experiment="YR-332-ba", seeds=dict(input=[9975100,9975101,9975102]), params=config,prereg=PREREG),
        source_sha256={p.relative_to(BASE).as_posix():digest(p) for p in code_paths},
        old_profile_sha256=digest(OLD_PROFILE), new_profile_sha256=digest(out/"profile-confirmed.json")))
    save(out/"input-bundle.json", dict(schema="gate-time-calibration-bundle-v1",
        information_profile="profile-confirmed.json", applied_axis="reservation_notice_only",
        frozen_demand="outputs/reports/yr332_visit_links/demand-profiles.json",
        frozen_curve_bank="outputs/reports/yr332_global_value_context/bank.json",
        candidate_only=["visit-arrival-templates-confirmed.json","visit-patterns-confirmed.json"],
        note="Old experiments keep their explicit old paths. New timing is selected explicitly; no global mutable default."))
    print(json.dumps(dict(status="BUILT", classification=status_counts,
        profile_counts={k:v["n"] for k,v in profile["flows"].items()}, visit_counts=impact["visits"]),ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=BASE/"오더데이터")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--private-out", type=Path, default=PRIVATE)
    args = parser.parse_args()
    build(args.root,args.out,args.private_out)
