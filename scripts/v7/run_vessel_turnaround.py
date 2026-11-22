"""Run independent trial lanes; preserve completed atomic outputs on restart."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from vessel_turnaround_experiment import DAY, DRAIN, LOADS, OUT, PREVIOUS, SEEDS, previous, save


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def launch(*args):
    command = [sys.executable, "-u", "scripts/v7/vessel_turnaround_experiment.py", *map(str, args)]
    subprocess.run(command, check=True)


def evaluate(arm, seed, index, load):
    p = OUT/"evaluation"/f"{arm}-{seed}-{index}-{load}.json"
    if p.exists():
        row = json.loads(p.read_text(encoding="utf8"))
        cp = previous(seed) if arm == "previous" else OUT/"models"/str(seed)/"update-5.pt"
        if arm == "rule" or row["checkpoint_sha256"] == sha(cp):
            return
    launch("evaluate", "--arm", arm, "--seed", seed, "--index", index, "--load", load)


def main():
    git = os.environ.get("TURNAROUND_GIT", "git")
    commit = subprocess.check_output([git,"rev-parse","HEAD"], text=True).strip()
    if subprocess.check_output([git,"status","--porcelain","--untracked-files=no"]).strip():
        raise RuntimeError("Tracked changes at trial start")
    manifest = dict(code_commit=commit, prereg_sha256=sha(Path(
        ".claude/docs/dashboard-task-specs/YR-332-bj-vessel-turnaround.md")),
        source_bundle=sha(PREVIOUS/"bundle.json"), initial={str(s):sha(previous(s)) for s in SEEDS},
        model_seeds=list(SEEDS), train_seed_base=10084100, test_seed_base=10084300,
        loads=LOADS, intake_s=DAY, max_drain_s=DRAIN, scheduled_updates=6, days=12,
        max_samples=2048, epochs=4, batch_size=64, continuation="same policy")
    path = OUT/"campaign.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf8"))
        if stored["code_commit"] != commit:
            # Documentation, test and publication commits may follow the trial.
            subprocess.run([git, "diff", "--exit-code", "--quiet", stored["code_commit"], commit, "--",
                "src/yard_rl", "scripts/v7/vessel_turnaround_experiment.py", "scripts/v7/global_policy_inputs.py",
                "scripts/v7/value_stability.py"], check=True)
            manifest["code_commit"] = stored["code_commit"]
        if stored != json.loads(json.dumps(manifest)):
            raise ValueError("Existing campaign provenance differs")
    else:
        save(path, manifest)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        work = {pool.submit(launch, "train", "--seed", seed): ("train",seed) for seed in SEEDS}
        baseline_arms = [] if "--models-only" in sys.argv else [("rule",(0,)), ("previous",SEEDS)]
        for arm, seeds in baseline_arms:
            for idx in range(3):
                for load in LOADS:
                    for seed in seeds:
                        work[pool.submit(evaluate,arm,seed,idx,load)] = (arm,seed)
        failures = []
        while work:
            done, _ = concurrent.futures.wait(work, return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                arm, seed = work.pop(f)
                try:
                    f.result()
                except Exception as error:
                    failures.append(str(error))
                    continue
                if arm == "train":
                    for idx in range(3):
                        for load in LOADS:
                            work[pool.submit(evaluate,"trained",seed,idx,load)] = ("trained",seed)
        if failures:
            raise RuntimeError(failures)


if __name__ == "__main__":
    main()
