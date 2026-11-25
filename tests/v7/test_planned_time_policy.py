import gzip
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from yard_rl.v7.crane.actor_critic import GraphState, pack_states
from yard_rl.v7.crane.planned_times import ARMS, PublicSchedule, encode_gap
from yard_rl.v7.world.domain.enums import JobStatus


def test_notice_etw_progress_and_expiry_use_only_visible_fields():
    stream = NS(started=False, done=False, plan=NS(planned_start_s=500., total_moves=8,
                                                    sts_move_interval_s=60.), remaining_moves=-1)
    truck = NS(job_id="a", vessel_id=None, status=JobStatus.PLANNED, provided_eta=220., is_external_truck=True)
    job = NS(job_id="b", vessel_id="v", release_time=740., is_external_truck=False)
    sim = NS(clock=100., jobs={"b": job}, vessels={"v": stream})
    def candidate(key):
        return (None, NS(job_ref=NS(job_id=key)))
    times = PublicSchedule("both_need")
    assert times.candidates(sim, [candidate("b")])[0] == [0., 0., *encode_gap(500, 100), 0., 0.]
    sim.jobs["a"] = truck
    times.notice(sim, truck)
    assert times.candidates(sim, [candidate("b")])[0][:2] == encode_gap(220, 100)
    truck.status = JobStatus.WAITING
    sim.clock, stream.started, stream.remaining_moves = 800., True, 6
    row = times.candidates(sim, [candidate("b")])[0]
    assert row == [0., 0., 0., 0., *encode_gap(920, 800)]
    assert encode_gap(800, 800) != encode_gap(None, 800)


@pytest.mark.parametrize("arm", ARMS)
def test_widened_checkpoint_preserves_logits_and_learns_new_columns(arm, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path("scripts/v7").resolve()))
    import short_window_train as bp
    from value_stability import restore
    original, _, _ = restore(bp.START)
    for name in ("ROOT", "OUT", "START", "PLANNED_ARM"):
        monkeypatch.setattr(bp, name, getattr(bp, name))
    monkeypatch.setattr(bp, "CACHE", tmp_path)
    bp.configure_planned_times(arm, initialize=True)
    wider, _, _ = restore(bp.START)
    generator = torch.Generator().manual_seed(43)
    x = torch.randn(7, 10, generator=generator)
    extra = torch.randn(7, 2*len(ARMS[arm]), generator=generator)
    def state(data):
        return GraphState(data, torch.arange(7), torch.arange(7) % 2, .3, .4)
    a, v = original(pack_states([state(x)]))
    b, w = wider(pack_states([state(torch.cat([x, extra], 1))]))
    torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(v, w, atol=2e-6, rtol=1e-6)
    b.sum().backward()
    assert wider.embedding.weight.grad[:, 10:].abs().sum() > 0


def test_dependency_requires_all_four_matching_updates_and_completed_test(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path("scripts/v7").resolve()))
    import after_bp_planned_times as queue
    monkeypatch.setattr(queue, "BL", tmp_path)
    out, cache = tmp_path/"out", tmp_path/"cache"
    queue.save(tmp_path/"test-input.json", {"sha256": "test", "orders": 7})
    assert not queue.finished(out, cache)
    for i in range(4):
        p = cache/"models"/f"update-{i}.pt"
        p.parent.mkdir(exist_ok=True, parents=True)
        p.write_bytes(bytes([i]))
        queue.save(out/"training"/f"fit-{i}.json", {"update": i, "checkpoint_sha256": queue.sha(p)})
    assert not queue.finished(out, cache)
    dest = out/"evaluation/retrained/result.json"
    data = dict(days=30, manifest_sha256="test", checkpoint_sha256=queue.sha(p),
                windows={"all": dict(trucks=7, ships=1, unfinished_trucks=0, unfinished_ships=0)})
    queue.save(dest, data)
    assert queue.finished(out, cache)
    data["windows"]["all"]["unfinished_ships"] = 1
    queue.save(dest, data)
    with pytest.raises(ValueError):
        queue.finished(out, cache)


def test_stored_old_anchor_stream_matches_json_across_chunk_boundaries(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts/v7").resolve()))
    from short_window_samples import anchors_from
    rows = [{"index": i, "payload": "x" * n} for i,n in enumerate((65536, 1, 2048))]
    p = tmp_path/"decisions.json.gz"
    with gzip.open(p, "wt", encoding="utf8") as f:
        json.dump(dict(horizon=1, anchors=rows), f)
    assert list(anchors_from(p)) == rows


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA runtime unavailable"))])
def test_first_action_can_be_substituted_from_identical_live_prefix(tmp_path, monkeypatch, device):
    monkeypatch.setenv("YARD_INFERENCE_DEVICE", device)
    monkeypatch.syspath_prepend(str(Path("scripts/v7").resolve()))
    import short_window_train as bp
    from planned_time_comparison import PairedResolver
    cfg = bp.setting(True)
    if not (cfg["manifests"] / bp.read(cfg["test"])["path"]).exists():
        pytest.skip("Local three-day replay input has not been generated")
    for name in ("ROOT", "OUT", "START", "PLANNED_ARM"):
        monkeypatch.setattr(bp, name, getattr(bp, name))
    monkeypatch.setattr(bp, "CACHE", tmp_path)
    bp.configure_planned_times("both", initialize=True)
    seen = []

    class PrefixDone(Exception):
        pass

    for branch in (None, 1):
        class Probe(PairedResolver):
            def resolve(self, sim, decision, generated):
                if self.events:
                    # The previous (divergent) assignment was actually committed by the engine.
                    seen.append(self.events[1])
                    raise PrefixDone
                return super().resolve(sim, decision, generated)

        from functools import partial
        with pytest.raises(PrefixDone):
            bp.month_crane_replay.run("trained", False, info_path=cfg["test"], manifest_dir=cfg["manifests"],
                                      checkpoint=bp.START, folder=tmp_path/str(branch), horizon=cfg["horizon"],
                                      resolver_factory=partial(Probe, days=(1,), intervention=branch))
    assert len(seen) == 2 and seen[0] == seen[1]
    assert seen[0]["model"] != seen[0]["rule"]
