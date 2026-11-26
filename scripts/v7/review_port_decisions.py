"""Review the completed BT comparisons using their frozen simulator and saved paired branches."""
import argparse
from collections import defaultdict
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import traceback


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "outputs/reports"
BT = REPORTS / "yr332_planned_times"
OUT = REPORTS / "yr332_port_decision_review"
TASK = "yard-rl-port-decision-review"
NAMES = {"classic": "고전 최단 작업 우선", "rule": "최단 작업 + 24분 승격",
         "bp": "기존 30분 학습", "eta": "트럭 도착 예정", "etw": "본선 시작 예정",
         "both": "도착 + 시작 예정", "both_need": "두 예정시각 + 상자 필요 예정"}


def read(path):
    return path.read_text(encoding="utf8")


def data(path):
    return json.loads(read(path))


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(content, encoding="utf8")
    tmp.replace(path)


def json_file(path, value):
    write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def candidate(sim, gc):
    ref = gc.job_ref
    job = sim.jobs.get(ref.job_id) if ref else None
    item = dict(kind=gc.kind.value, job=job.job_id if job else None,
                flow=job.flow.value if job else None)
    if job and job.is_external_truck:
        for name, stamp in (("gate_age_min", job.actual_gate_in), ("block_age_min", job.actual_block_arrival)):
            item[name] = (sim.clock - stamp) / 60 if stamp is not None and stamp <= sim.clock else None
    if job and job.vessel_id:
        v = sim.vessels[job.vessel_id]
        item.update(stream=job.vessel_id, buffer=v.buffer_level, started=v.started,
                    quay_blocked=v.sts_blocked_since_s is not None,
                    remaining_work_min=v.remaining_service_time_s() / 60)
    return item


def recover_context(pairs, cfg, checkpoint, manifest, scratch):
    from planned_time_comparison import PairedResolver, month_crane_replay

    expected = {int(d): row["event"] for d, row in pairs["days"].items() if row["disagreement"]}
    captured = {}
    if not expected:
        return captured

    class CapturedAll(Exception):
        pass

    class Recorder(PairedResolver):
        def resolve(self, sim, decision, generated):
            chosen = super().resolve(sim, decision, generated)
            day = int(sim.clock // 86400) + 1
            if day not in expected or day in captured or day not in self.events:
                return chosen
            event = self.events[day]
            if event != expected[day]:
                raise ValueError(f"First assignment or preceding decisions differ on day {day}")
            changes = []
            for crane, idx in event["model"].items():
                alt = event["rule"][crane]
                if idx != alt:
                    items = generated[crane].items
                    changes.append(dict(crane=crane, model=candidate(sim, items[idx]),
                                        rule=candidate(sim, items[alt])))
            captured[day] = changes
            if len(captured) == len(expected):
                raise CapturedAll
            return chosen

    try:
        month_crane_replay.run("trained", False, info_path=cfg["test"], manifest_dir=manifest.parent,
                               checkpoint=checkpoint, folder=scratch, horizon=cfg["horizon"],
                               resolver_factory=lambda *a, **kw: Recorder(*a, days=expected, **kw),
                               inference_device="cpu")
    except CapturedAll:
        pass
    if captured.keys() != expected.keys():
        raise ValueError("Could not reproduce all registered disagreements")
    return captured


def ledger(folder):
    with gzip.open(folder / "trucks.jsonl.gz", "rt", encoding="utf8") as stream:
        trucks = {r["job"]: (r["gate"], r["finish"]) for r in map(json.loads, stream)}
    ships = {r["ship"]: r for r in data(folder / "ships.json")}
    counts = data(folder / "result.json")["windows"]["all"]
    if (len(trucks), len(ships), 0, 0) != (counts["trucks"], counts["ships"],
                                           counts["unfinished_trucks"], counts["unfinished_ships"]):
        raise ValueError("Incomplete or duplicate cohort")
    return trucks, ships


def changes_in_time(base, branch, t):
    """Rule first minus model first, over the whole terminal. Both cohorts have fixed arrivals."""
    if base.keys() != branch.keys():
        raise ValueError("Different cohorts")
    short, total = [], []
    for key, (start, finish) in base.items():
        other_start, other_finish = branch[key]
        if other_start != start or min(finish, other_finish) < start:
            raise ValueError(f"Different starts or invalid finishes for {key}")
        if finish == other_finish:
            continue
        if min(finish, other_finish) < t - 1e-6:
            raise ValueError(f"Outcome changed before the intervention for {key}")
        short.append(min(other_finish, t + 1800) - min(finish, t + 1800))
        total.append(other_finish - finish)
    return math.fsum(short) / 60, math.fsum(total) / 60


def audit_cases(folder, pairs, contexts, manifest):
    with gzip.open(manifest, "rt", encoding="utf8") as src:
        schedule = json.load(src)["vessels_by_day"]
    stream_ids = defaultdict(list)
    index = {}
    for rows in schedule.values():
        for row in rows:
            stream_ids[row["ship"]].append(row["key"])
            index[row["key"]] = row["ship"]
    base_trucks, base_ships = ledger(folder / "model")
    cases = {}
    for day, row in pairs["days"].items():
        if not row["disagreement"]:
            cases[day] = {"type": "선택 일치"}
            continue
        changes = contexts[int(day)]
        other_trucks, other_ships = ledger(folder / f"day-{day}")
        t = row["event"]["time"]
        truck30, truck = changes_in_time(base_trucks, other_trucks, t)
        def ship_times(ships):
            return {k: (v["start"], v["finish"]) for k, v in ships.items()}
        ship30, ship = changes_in_time(ship_times(base_ships), ship_times(other_ships), t)
        observed = (truck / len(base_trucks), ship / len(base_ships) / 60, truck + 67 * ship)
        expected = row["rule_minus_model"]
        for actual, key in zip(observed, ("truck_minutes", "ship_hours", "weighted_minutes")):
            if not math.isclose(actual, expected[key], abs_tol=1e-6, rel_tol=1e-8):
                raise ValueError(f"Paired total differs for {day} / {key}")
        for change in changes:
            for side in ("model", "rule"):
                item = change[side]
                if item.get("stream"):
                    stream = item["stream"]
                    call = index[stream]
                    i = stream_ids[call].index(stream)
                    base, other = base_ships[call], other_ships[call]
                    for record in (base, other):
                        if (len(record["stream_finishes"]) != len(stream_ids[call])
                                or record["finish"] != max(record["stream_finishes"])):
                            raise ValueError(f"Stream ledger mismatch for {call}")
                    a, b = base["stream_finishes"][i], other["stream_finishes"][i]
                    item.update(call=call, last_in_model=a == base["finish"],
                                last_in_rule=b == other["finish"], stream_delta_min=(b - a) / 60,
                                call_delta_min=(other["finish"] - base["finish"]) / 60)
                elif item["job"] in base_trucks:
                    jid = item["job"]
                    item["order_delta_min"] = (other_trucks[jid][1] - base_trucks[jid][1]) / 60
        selected = [changes[0][k] for k in ("model", "rule")]
        direct = (len(changes) == 1 and all(x["kind"] == "SERVE" for x in selected)
                  and sum(bool(x.get("stream")) for x in selected) == 1
                  and sum(x["job"] in base_trucks for x in selected) == 1)
        cases[day] = dict(type="트럭·본선 직접 선택" if direct else "다른 작업 또는 복수 배정 차이",
                          event=row["event"], changes=changes, truck30=truck30, truck=truck,
                          ship30=ship30, ship=ship, weighted=observed[2])
    return cases


def render(month, cases, smoke=False):
    lines = ["# 3일 작동 검산" if smoke else "# 예정시각 실험 후 배차 가설 검토", "",
             "시험 월 한 달을 끝까지 처리한 결과다. 트럭은 게이트 진입부터 작업 완료까지 건당 평균이고, "
             "본선은 기항당 마지막 안벽 작업까지의 평균이다. 학습 정책에는 기존 24분 승격이 적용됐다.", "",
             "| 방식 | 트럭 분/건 | 본선 시간/척 | 고전 최단 대비 가중 합계 차이(%) |",
             "|---|---:|---:|---:|"]
    if smoke:
        lines = [lines[0], "", "등록된 3일 자료에서 비교 시점·선택·장부를 복원한 검산이다."]
    for arm, r in month.items():
        gap = 100 * (r["weighted_minutes"] / month["classic"]["weighted_minutes"] - 1)
        lines.append(f"| {NAMES[arm]} | {r['truck_minutes']:.2f} | {r['ship_hours']:.2f} | {gap:+.2f} |")
    if "both_need" in month:
        delta = month["both_need"]["ship_hours"] - month["both"]["ship_hours"]
        lines.extend(["", f"상자 필요 예정시각을 추가한 정책의 본선 시간 차이는 척당 {delta:+.3f}시간이다. "
                      "예정시각의 효과에는 해당 시각 입력으로 재학습한 효과도 포함된다."])
    lines.extend(["", "## 같은 상황에서 첫 배정만 바꾼 사례", "",
                  "아래 차이는 첫 배정만 규칙을 따른 경우에서 첫 배정도 모델을 따른 경우를 뺀 분이다. "
                  "양수이면 모델 선택이 시간을 아꼈다. 두 갈래 모두 이후에는 같은 모델을 썼다. "
                  "여기의 규칙은 원실험의 24분 승격 조건이다. 30분 값은 터미널 전체의 누적 체류 차이다.", "",
                  "| 날짜 | 선택 유형 | 트럭 30분 내 / 끝까지, 총분 | 본선 30분 내 / 끝까지, 총분 |",
                  "|---|---|---:|---:|"])
    for day, c in cases.items():
        if "changes" in c:
            lines.append(f"| {day} | {c['type']} | {c['truck30']:+.3f} / {c['truck']:+.3f} | "
                         f"{c['ship30']:+.3f} / {c['ship']:+.3f} |")
        else:
            lines.append(f"| {day} | 선택 일치 | 해당 없음 | 해당 없음 |")
    direct = [(d, c) for d, c in cases.items() if c["type"] == "트럭·본선 직접 선택"]
    lines.extend(["", f"지정한 {len(cases)}일 중 같은 크레인이 트럭과 본선 사이에서 직접 고른 사례는 "
                  f"{len(direct)}건이다."])
    favors = {"본선": 0, "트럭": 0, "동일": 0}
    for day, c in direct:
        change = c["changes"][0]
        policy_ship = "stream" in change["model"]
        v, tr = (change["model"], change["rule"]) if policy_ship else (change["rule"], change["model"])
        sign = 1 if policy_ship else -1
        ship_first_saving = sign * c["weighted"]
        winner = "동일" if math.isclose(ship_first_saving, 0, abs_tol=1e-6) else (
            "본선" if ship_first_saving > 0 else "트럭")
        favors[winner] += 1
        picked = "본선" if policy_ship else "트럭"
        blocked = "멈춰 있었다" if v["quay_blocked"] else "멈춤이 기록되지 않았다"
        tail = "마지막이었다" if v["last_in_model"] else "다른 작업 줄이 더 늦게 끝났다"
        lines.extend(["", f"- {day}일차에 모델은 {picked}을 골랐다. 결정 당시 해당 안벽 작업 줄의 보관량은 "
                      f"{v['buffer']}개였고 안벽은 {blocked}. 모델 경로에서 이 작업 줄은 {tail}. "
                      f"본선부터 처리해 앞당긴 시간은 해당 작업 줄 {sign * v['stream_delta_min']:+.3f}분, "
                      f"해당 선박 {sign * v['call_delta_min']:+.3f}분이다. "
                      f"트럭부터 처리해 해당 트럭이 절약한 시간은 {-sign * tr['order_delta_min']:+.3f}분이다."])
    if direct:
        lines.extend(["", f"끝까지 누적한 가중 합계에서 본선 우선이 유리한 사례 {favors['본선']}건, "
                      f"트럭 우선이 유리한 사례 {favors['트럭']}건, 동률 {favors['동일']}건이다.",
                      "지정 사례 안에서 유리한 방향이 바뀌었다." if favors["트럭"] and favors["본선"]
                      else "지정 사례에서 양방향 전환을 확인하기에는 근거가 부족하다."])
    lines.extend(["", "## 주장별 해석", "",
                  "- 위의 직접 선택 사례에 한해 안벽 작업 줄과 배 전체의 완료 차이를 대조할 수 있다. "
                  "직접 사례가 없으면 이 실험에서 우선순위 전환 가설은 판정 보류다. 네 날짜의 첫 불일치로 빈도나 일반적인 정책 우위를 추정할 수 없다.",
                  "- 안벽 보관량과 결정 당시 멈춤은 모형 관측값이다. 특정 선적 위치에 맞는 상자와 대체 가능 순서는 적재계획 자료가 필요하다.",
                  "- 게이트 장기 체류 가중치는 승격을 해제한 별도 30일 실험에서 같은 방문끼리 비교한다.",
                  "- 실제 운반차 인계·복귀, 시간대별 트럭 집중, 예상 대비 실제 크레인 작업시간은 각각 운반차 이력, 예약·도착 이력, 장비 시작·완료 시각을 대조해야 한다."])
    return "\n".join(lines) + "\n"


def review(frozen, smoke, output):
    os.chdir(frozen)
    sys.path[:0] = [str(frozen / "scripts/v7"), str(frozen / "src")]
    from after_bp_planned_times import contract, sha
    from planned_time_comparison import DAYS, read_month
    from joint_turnaround_train import setting
    import torch
    torch.set_num_threads(1)
    reg = data(BT / "registration-cuda-cpu.json")
    if reg["sources"] != contract():
        raise ValueError("Original registered files changed")
    cfg = setting(smoke)
    cache = Path("build/yr332_joint_turnaround/planned-times/both")
    folder = cache / "smoke-pairs" if smoke else BT / "paired"
    pairs = data(folder / "summary.json")
    info = data(cfg["test"])
    manifest = cfg["manifests"] / info["path"]
    checkpoint = cache / ("smoke-models/update-0.pt" if smoke else "models/update-3.pt")
    days = (1,) if smoke else DAYS
    if (set(pairs["days"]) != {str(d) for d in days} or sha(checkpoint) != pairs["checkpoint"]
            or info["sha256"] != pairs["manifest"] or sha(manifest) != pairs["manifest"]):
        raise ValueError("Registered days, checkpoint or cargo differ")
    for day, case in pairs["days"].items():
        p = folder / (f"day-{day}" if case["disagreement"] else "model")
        record = data(p / "pair.json")
        if (record["events"].get(day) != case.get("event") or record["manifest"] != pairs["manifest"]
                or record["checkpoint"] != pairs["checkpoint"]):
            raise ValueError(f"Paired record mismatch for {day}")
    if smoke:
        month = {}
    else:
        published = data(BT / "comparison.json")
        paths = {"classic": REPORTS / "yr332_unguarded_spt/evaluation",
                 "rule": REPORTS / "yr332_joint_turnaround/evaluation/SF_SPT",
                 "bp": REPORTS / "yr332_short_window/evaluation/retrained"}
        paths.update({arm: BT / arm / "evaluation/retrained" for arm in reg["arms"]})
        month = {arm: read_month(path, pairs["manifest"]) for arm, path in paths.items()}
        if len({r["cohort"] for r in month.values()}) != 1:
            raise ValueError("Monthly cohorts differ")
        for arm in published:
            if any(month[arm][k] != published[arm][k] for k in month[arm]):
                raise ValueError(f"Published comparison differs for {arm}")
        if set(published) != set(month) - {"classic"} or month["both"] != data(folder / "model/pair.json")["metrics"]:
            raise ValueError("Missing arm or different control policy")
    with TemporaryDirectory() as scratch:
        contexts = recover_context(pairs, cfg, checkpoint, manifest, Path(scratch))
    cases = audit_cases(folder, pairs, contexts, manifest)
    evidence = dict(smoke=smoke, registered_commit=reg["commit"], checkpoint=pairs["checkpoint"],
                    manifest=pairs["manifest"], pairs_sha256=sha(folder / "summary.json"), month=month, cases=cases)
    json_file(output / "review.json", evidence)
    write(output / "README.md", render(month, cases, smoke))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--smoke-output", type=Path, help="Replay only the archived wiring case into a separate directory")
    parser.add_argument("--scheduled", action="store_true")
    args = parser.parse_args()
    if args.smoke_output:
        out = args.smoke_output.resolve()
        if out == OUT or out.is_relative_to(BT):
            parser.error("Smoke output must be separate from experiment results")
        review(args.frozen.resolve(), True, out)
        return
    phase = data(BT / "queue-status.json")["stage"]
    if phase != "complete":
        json_file(OUT / "status.json", dict(state="waiting", experiment_stage=phase))
        return
    try:
        # Task Scheduler serializes runs. Publish the completion marker last so a failed run can be retried.
        status = OUT / "status.json"
        if not (status.exists() and data(status)["state"] == "complete"
                and (OUT / "README.md").is_file() and (OUT / "review.json").is_file()):
            json_file(status, dict(state="reviewing"))
            review(args.frozen.resolve(), False, OUT)
            json_file(status, dict(state="complete"))
        if args.scheduled:
            subprocess.run(["schtasks", "/delete", "/tn", TASK, "/f"], check=True)
    except Exception:
        json_file(OUT / "status.json", dict(state="error", detail=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
