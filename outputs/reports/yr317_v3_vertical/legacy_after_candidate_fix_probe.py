"""Recheck the legacy default after opt-in candidate-generator integration."""
from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import legacy_equivalence_probe as probe

OUT = Path(__file__).resolve().parent
LABEL = "workspace-after-candidate-fix"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    os.sched_setaffinity(0, {17})
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    prior_path = OUT / "legacy-equivalence.json"
    prior = read(prior_path)
    baseline = prior["artifacts"]["frozen"]
    frozen_path = OUT / baseline["path"]
    frozen_meta_path = OUT / baseline["source_manifest_path"]
    assert prior["passed"] is True
    assert sha(frozen_path) == baseline["compressed_sha256"]
    assert sha(frozen_meta_path) == baseline["source_manifest_sha256"]
    frozen_raw = gzip.decompress(frozen_path.read_bytes())
    assert hashlib.sha256(frozen_raw).hexdigest() == baseline["uncompressed_sha256"]
    frozen = json.loads(frozen_raw)
    assert probe.digest(frozen) == prior["common_result_sha256"]
    preserved = {p.name: sha(p) for p in (prior_path, frozen_path, frozen_meta_path,
        OUT / "workspace-result.json.gz", OUT / "workspace-meta.json")}
    record = dict(schema="yr317.v3.vertical.legacy-after-candidate-fix.v1",
        started_at=datetime.now(timezone.utc).isoformat(), cpu=17, threads=1,
        seed=probe.SEED, case=prior["case"], new_training_runs=0,
        new_workspace_diagnostic_runs=1, repeated_frozen_runs=0, claim_eligible=False,
        compared=prior["compared"], limits=prior["limits"],
        baseline_receipt=dict(path=prior_path.name, raw_sha256=sha(prior_path)),
        state="running", passed=False)
    output = OUT / "legacy-after-candidate-fix-equivalence.json"
    probe.dump(output, record)
    with (OUT / "legacy-after-candidate-fix.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log):
            probe.child(probe.WORKSPACE, LABEL)
    new_path = OUT / f"{LABEL}-result.json"
    new_raw = new_path.read_bytes()
    new = json.loads(new_raw)
    empty_manifest = new["result"].get("environment_manifest") == {}
    if empty_manifest:
        del new["result"]["environment_manifest"]
    meta_path = OUT / f"{LABEL}-meta.json"
    meta = read(meta_path)
    old_sources = read(frozen_meta_path)["source_before"]
    sources = meta["source_before"]
    changed = sorted(p for p in old_sources.keys() & sources.keys()
                     if old_sources[p]["lf_sha256"] != sources[p]["lf_sha256"])
    added = sorted(sources.keys() - old_sources.keys())
    removed = sorted(old_sources.keys() - sources.keys())
    allowed = {"src/yard_rl/v3/stage/month_run.py", "src/yard_rl/v3/stage/month_engine.py",
               "src/yard_rl/v3/stage/episode.py"}
    unexpected = sorted(set(changed) - allowed)
    differences = probe.differences(frozen, new)
    checks = dict(empty_new_environment_manifest=empty_manifest,
        exact_serialized_results=probe.digest(frozen) == probe.digest(new),
        no_differences=not differences, expected_request_count=meta["requested"] == 21,
        source_stable=meta["source_before"] == meta["source_after"],
        loaded_sources_stable=not meta["loaded_sources_changed_during_run"],
        scoped_source_changes=not unexpected and not removed and all(
            p.startswith("src/yard_rl/v3/layouts/") for p in added),
        no_rollouts=meta["rollout_calls"] == 0,
        cpu_one_thread=meta["cpu_affinity"] == [17] and
            meta["torch_threads"] == meta["interop_threads"] == 1,
        prior_evidence_unchanged=all(sha(OUT / p) == h for p, h in preserved.items()))
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as compressed:
        compressed.write(new_raw)
    packed = buffer.getvalue()
    assert gzip.decompress(packed) == new_raw
    compressed_path = new_path.with_suffix(".json.gz")
    compressed_path.write_bytes(packed)
    record.update(state="completed", completed_at=datetime.now(timezone.utc).isoformat(),
        passed=all(checks.values()), checks=checks, differences=differences[:30],
        common_result_sha256=probe.digest(frozen),
        workspace_normalized_result_sha256=probe.digest(new),
        source_difference=dict(lf_changed=changed, added=added, removed=removed,
                               unexpected_changed=unexpected),
        workspace_run={k: v for k, v in meta.items() if k not in ("source_before", "source_after")},
        artifacts=dict(frozen=baseline, workspace=dict(path=compressed_path.name,
            compressed_bytes=len(packed), compressed_sha256=sha(compressed_path),
            uncompressed_bytes=len(new_raw), uncompressed_sha256=hashlib.sha256(new_raw).hexdigest(),
            normalized_semantic_sha256=probe.digest(new), source_manifest_path=meta_path.name,
            source_manifest_sha256=sha(meta_path))),
        probe_sha256=sha(Path(__file__).resolve()), helper_sha256=sha(Path(probe.__file__)))
    probe.dump(output, record)
    print(json.dumps(dict(passed=record["passed"], checks=checks,
        differences=differences[:5], elapsed_s=meta["elapsed_s"])), flush=True)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
