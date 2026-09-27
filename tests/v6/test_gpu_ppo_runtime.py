"""★**구간 보상과 수집 흐름**이 v5 와 같은가 ([[YR-327]] 조각 8 · key=runtime).

옮긴 것은 `ppo/runtime.py:59-215` 의 **학습 회계**다 — 60초 경계마다 "비용이 그 60초 동안
얼마나 늘었나" 를 보상으로 바꾸고, 그 사이 결정을 블록별로 모아 두는 절차. 검사 대상은
`src/yard_rl/v6/gpu/ppo_runtime.py` 하나다.

■ 무엇을 지키나 — 기대값을 손으로 적지 않는다. v5 를 **실제로 굴려** 받은 값과 맞댄다.
  (A) **무대 재생 대조** (`test_stage_*`) — v5 `ppo/run.run_debug` 과 같은 세계(시드 9900302 ·
      `sample_actions=False`(최고점) · `stop_s` 절단)를 돌리며 `boundary`·`select`·`_update` 를
      **호출 순서대로 한 줄씩** 가로채 테이프에 적고, 같은 테이프를 배열 루프에 흘려 대조한다:
        · 경계 수·순서·시각                                            — `==`
        · 경계마다 Φ(누적 비용) — v5 `reward/phi.terminal_cost_krw` 대 조각 5 `gpu/phi.py`  — `==` (비트)
        · 구간 보상 `−(Φ(t)−Φ(t−60))/1e6`                             — `==` (수집된 구간은 v5 `Interval.reward` 원본과)
        · 어느 경계가 구간을 냈나(`advanced`)·수집했나(`collected`)·갱신을 걸었나(`do_update`)
          ·절단했나(`stop`)                                            — `==`
        · 누적 칸 다섯 (`intervals`·`learning_intervals`·`total_reward`·`learning_reward`·`len(buffer)`) — `==`
        · 결정마다 pending 에 담겼나·몇 번째 칸인가·역할 계수기        — `==`
        · 수집된 구간의 상태 (B,37)·가치 (B,)·결정 묶음(역할·시각·행동·로그확률·37칸·마스크) — `==`
      무대 셋: ① 명세 무대(부하 60 · 2시간) ② 넉넉한 무대(부하 300 · 6시간 — 판매자·구매자·크레인
      **세 역할 전부**와 본선·재조작·빈주행 비용이 실제로 도는 곳) ③ 학습창 무대(창 [1800, 5400))
      ④ 고정 운영 무대(`training=False` — 수집도 갱신도 0 이어야 한다).
  (B) **불변식 대조** (`test_invariant_*`) — v5 가 **예외를 던지는** 여덟 자리를 그 자리에서 재현하고
      (v5 쪽은 `pytest.raises` 로 실제 예외를 확인), 배열판이 같은 조건에서 **위반 비트**를 켜는지 본다.
      jit 안에서는 예외를 못 던지므로 "조용히 이상한 답" 대신 비트로 크게 알리는 것이 규약이다.
  (C) **학습창 스칼라 논리** (`test_window_stub_matches_v5`) — v5 회귀시험
      `tests/v6/test_ppo_continuous.py:18-45` 와 **같은 대체 입력**(상태·비용을 함수로 갈아끼운 1블록
      세계)을 v5 런타임과 배열 런타임에 **둘 다** 먹여 구간 수·보상·수집 배치 시작시각·갱신 시점을 대조.
  (D) **격자 사진** (`test_counter_tape_*`) — Φ 의 항2(빈주행)·항3(재조작)·항4(본선유휴)는 누적
      계수기라 지나가면 되짚을 수 없다. v5 `stage/month_run._MonthTape` 를 실제 세계에서 찍고
      `read`/`diff` 를 배열판(`tape_read`/`tape_diff`)과 대조하고, v5 `_phi_of_day`(하루치 Φ)까지
      같은 값이 나오는지 본다.

■ 범위 밖 (다른 담당·조각)
  · GAE·미니배치·Adam 갱신 산술        → 조각 8 `buffer`·`update` 담당
  · 망 순전파·행동 고르기              → 조각 7 `v5net.py` (이 시험은 v5 가 고른 행동을 **받아** 적는다)
  · 30일 무대 조립(`run_month`)·시장    → 조각 8 `month_engine`·`market` 담당
  · `workload` 잠재함수 성형            → v5 PPO 정본 경로가 `workload=None` 이라 0 (모듈 머리말)
  · 추첨(sample) 경로                   → `torch.multinomial` 난수열은 재현 불가 — 동등성은 최고점(argmax) 축

실행 (Git Bash · Windows 파이썬 · CPU x64):
    PYTHONPATH="src;tests/v6" PYTHONIOENCODING=utf-8 .venv-jax/Scripts/python.exe \
        -m pytest tests/v6/test_gpu_ppo_runtime.py -q -p no:cacheprovider
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np                                  # ★numpy → jax → torch 순서 (OMP #15 방어)
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)           # ★float64 — 동등성은 x64 에서만
jnp = jax.numpy
torch = pytest.importorskip("torch")

from yard_rl.v6.gpu import ppo_runtime as PR                                          # noqa: E402
from yard_rl.v6.gpu.phi import orders_from_records                                    # noqa: E402
from yard_rl.v6.gpu.state import empty_orders                                         # noqa: E402
# ── v5 정본 (읽기만 한다) ──────────────────────────────────────────────────────
from yard_rl.v6.ppo.model import BlockPolicy, encode                                  # noqa: E402
from yard_rl.v6.ppo.runtime import DebugStop, PPOConfig, PPORuntime                   # noqa: E402
from yard_rl.v6.stage.episode import rehandles_of, yc_empty_travel_s                  # noqa: E402
from yard_rl.v6.stage.month import DAY_S, DayPlan, month_vessel_idle                  # noqa: E402
from yard_rl.v6.stage import month_run                                                # noqa: E402

SEED = 9900302                 # 진단 대역 (`eval/guards.DIAGNOSTIC_BAND`)
N_BLOCKS = 21                  # H21 프로파일은 고정이다 — v5 는 1블록 세계를 못 만든다
CMAX = 64                      # 한 구간·한 블록 결정 칸 (12시간 무대 실측 최대 39)
AMAX = 29                      # 후보 칸 (판매자 1+20+8)
ROLE_CODE = {r: i for i, r in enumerate(PR.ROLES)}


# ════════════════════════════════════════════════ 고정 화물 입력 (conftest 복사)
def _fixed_cargo_patch(mp) -> None:
    """`tests/v6/conftest.py:fixed_container_input` 과 **같은 두 패치**.

    거기 것은 함수 범위 fixture 라 무대를 모듈 범위로 **한 번만** 굴려 여러 시험이 나눠 쓰는 데
    쓸 수 없다 (무대 하나가 2~10초다). 그래서 같은 내용을 `pytest.MonkeyPatch` 로 다시 건다.
    내용이 갈리면 두 파일이 서로 다른 세계를 보게 되므로, 바뀌면 함께 고친다.
    """
    original_build, original_vessels = month_run.build_month, month_run.plan_month_vessels
    context: dict = {}

    def build(*args, **kwargs):
        data = original_build(*args, **kwargs)
        assert len(data["schedule"]) <= 2000, "작은 시험 입력 — 운영 수리가 아니다"
        available = {b: set(s.containers) for b, s in data["day0"]["scenarios"].items()}
        used: dict = defaultdict(set)
        for e in data["schedule"]:
            if e["flow"] != "GATE_OUT":
                continue
            bid, cid = e["block"], e["target"]
            if cid not in available[bid] or cid in used[bid]:
                cid = sorted(available[bid] - used[bid])[0]
            e["target"] = e["con_no"] = cid
            used[bid].add(cid)
        context.update(available=available, used=used)
        return data

    def vessels(*args, **kwargs):
        rows = original_vessels(*args, **kwargs)
        for day, items in rows.items():
            if day != 0:
                rows[day] = []
                continue
            for row in items:
                if row["work"] == "LOAD":
                    bid = row["block"]
                    pool = sorted(context["available"][bid] - context["used"][bid])
                    assert len(pool) >= row["moves"]
                    row["targets"] = pool[:row["moves"]]
                    context["used"][bid].update(row["targets"])
        return rows

    mp.setattr(month_run, "build_month", build)
    mp.setattr(month_run, "plan_month_vessels", vessels)


# ════════════════════════════════════════════════ v5 를 굴려 테이프를 적는다
class _BufferSpy(list):
    """`self.buffer.append(Interval(...))` 를 그 자리에서 낚아챈다 (clear 뒤에도 원본이 남는다)."""

    def __init__(self, sink):
        super().__init__()
        self._sink = sink

    def append(self, item):
        self._sink.append(item)
        super().append(item)


def _record_times(records) -> tuple[np.ndarray, np.ndarray]:
    """기록 사전 → (게이트인 A, 게이트아웃 O) — `phi.orders_from_records` 와 **같은 순서**.

    Φ 는 이 두 열과 `is_external` 만 읽으므로 경계마다 스물다섯 열을 다 만들지 않는다
    (`test_stage_spec_phi_snapshot_matches_orders_from_records` 가 같은 값임을 확인한다).
    """
    n = len(records)
    a, o = np.full(n, np.inf), np.full(n, np.inf)
    for i, rec in enumerate(records.values()):
        if rec.gate_in_s is not None:
            a[i] = float(rec.gate_in_s)
        if rec.gate_out_s is not None:
            o[i] = float(rec.gate_out_s)
    return a, o


class _KindSpy(dict):
    """`rt.crane_actions[name] = … + 1` (`ppo/crane.py:82`) 를 **순서대로** 적는다."""

    def __init__(self, sink):
        super().__init__()
        self._sink = sink

    def __setitem__(self, key, value):
        self._sink.append(key)
        super().__setitem__(key, value)


def _run_v5(*, load: int, duration_s: float, window=None, training: bool = True,
            snap_s: float | None = None, n_days: int = 2) -> dict:
    """v5 학습 루프를 실제로 굴리고 **호출 순서 그대로** 테이프를 적는다.

    `ppo/run.run_debug` 의 본문(시드 검사·2일 세계·`stop_s` 절단)을 그대로 쓰되 `sample_actions=False`
    (최고점) 로 고정한다 — `run_debug` 는 그 손잡이를 안 받고 `training=True` 면 추첨이 되기 때문이다
    (추첨은 `torch.multinomial` 난수열이라 배열판이 같은 표본을 낼 수 없다 · 조각 7 규약).
    """
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    config = PPOConfig(rollout_intervals=60, epochs=2, minibatch_size=64)
    rt = PPORuntime(BlockPolicy(), config=config, seed=SEED, training=training,
                    stop_s=duration_s, sample_actions=False, learning_window_s=window)
    intervals: list = []
    rt.buffer = _BufferSpy(intervals)
    kinds: list = []                  # 크레인이 고른 종류 — 호출 순서
    rt.crane_actions = _KindSpy(kinds)
    events: list = []                 # ('B', …) / ('S', …) — v5 호출 순서
    updates: list = []                # `_update` 가 불린 자리
    phi_in: dict = {}                 # 이 경계의 Φ 원료 (read_cost 안에서 채운다)
    tape5 = month_run._MonthTape({}, {})      # (D) 격자 사진 — bind 뒤에 meta/archive 를 꽂는다

    orig_read = rt.read_cost
    orig_select = rt.select
    orig_boundary = rt.boundary
    orig_update = rt._update

    def read_cost(t):
        total = orig_read(t)
        a, o = _record_times(rt.bridge.records)
        vessels = month_vessel_idle(rt.mbt, rt.meta, rt.archive)
        phi_in.clear()
        phi_in.update(t=float(t), total=float(total), breakdown=dict(rt.cost_breakdown),
                      a=a, o=o, n_records=len(rt.bridge.records),
                      vessel_gt=np.array([float(g) for g, _ in vessels.values()]),
                      vessel_idle=np.array([float(i) for _, i in vessels.values()]),
                      yc=float(yc_empty_travel_s(rt.mbt)), rh=int(rehandles_of(rt.mbt)))
        return total

    def select(role, bid, t, rows, mask=None):
        block = rt.index[bid]
        before = len(rt.pending[block])
        action = orig_select(role, bid, t, rows, mask)
        after = rt.pending[block]
        recorded = len(after) > before
        choice = after[-1] if recorded else None
        x = encode(rows, role)
        m = (torch.ones(len(x), dtype=torch.bool) if mask is None else mask.clone().bool())
        events.append(("S", dict(
            role=role, bid=bid, block=block, t=float(t), n=len(rows), action=int(action),
            x=x.numpy().copy(), mask=m.numpy().copy(), recorded=recorded,
            time_s=rt.time_s, slot=(before if recorded else -1),
            log_prob=(0.0 if choice is None else float(choice.log_prob)),
            choice_x=(None if choice is None else choice.rows.numpy().copy()),
            choice_mask=(None if choice is None else choice.mask.numpy().copy()),
            choice_role=(None if choice is None else choice.role),
            choice_t=(None if choice is None else float(choice.time_s)))))
        return action

    def _update(bootstrap):
        n_before = len(rt.updates)
        pending_batch = list(rt.buffer)
        orig_update(bootstrap)
        updates.append(dict(t=rt.time_s, bootstrap=np.asarray(bootstrap, np.float64).copy(),
                            did=len(rt.updates) > n_before, n=len(pending_batch),
                            starts=[float(r.start_s) for r in pending_batch]))

    def boundary(t, *, terminated=False, final=False):
        snap = dict(t=float(t), terminated=bool(terminated), final=bool(final),
                    started=rt.time_s is not None, time_s=rt.time_s, cost_krw=rt.cost_krw,
                    intervals=rt.intervals, learning_intervals=rt.learning_intervals,
                    n_buffered=len(rt.buffer), total_reward=rt.total_reward,
                    learning_reward=rt.learning_reward,
                    pending_before=[len(p) for p in getattr(rt, "pending", [])])
        n_iv, n_up = len(intervals), len(updates)
        stopped = False
        try:
            orig_boundary(t, terminated=terminated, final=final)
        except DebugStop:
            stopped = True
        collected = len(intervals) > n_iv
        iv = intervals[-1] if collected else None
        snap.update(
            after_time_s=rt.time_s, after_cost=rt.cost_krw, after_intervals=rt.intervals,
            after_learning=rt.learning_intervals, after_buffered=len(rt.buffer),
            after_total_reward=rt.total_reward, after_learning_reward=rt.learning_reward,
            after_truncated=rt.truncated, after_updates=len(rt.updates),
            collected=collected, stopped=stopped,
            update_called=(len(updates) > n_up),
            did_update=(len(updates) > n_up and updates[-1]["did"]),
            bootstrap=(updates[-1]["bootstrap"] if len(updates) > n_up else None),
            states=rt.states.numpy().copy(), values=np.asarray(rt.values, np.float64).copy(),
            initial_cost=rt.initial_cost,
            interval=(None if iv is None else dict(
                start_s=float(iv.start_s), end_s=float(iv.end_s), reward=float(iv.reward),
                terminated=bool(iv.terminated),
                states=iv.states.numpy().copy(),
                values=np.asarray(iv.values, np.float64).copy(),
                n_choices=[len(c) for c in iv.choices],
                choices=[[(c.role, float(c.time_s), int(c.action), float(c.log_prob))
                          for c in blk] for blk in iv.choices])),
            phi=dict(phi_in))
        events.append(("B", snap))
        if snap_s is not None and float(t) % snap_s == 0.0:
            tape5.snap(rt.mbt, float(t))
        if stopped:
            raise DebugStop("re-raised after recording")

    orig_finish = rt.finish

    def finish(t, *, terminated=False):
        """`finish` 는 `boundary` 가 **돌아온 뒤** `truncated` 를 고친다 (runtime.py:212-214).

        그래서 경계 안에서 찍은 `after_truncated` 는 아직 옛값이다 — 마지막 줄을 여기서 바로잡는다.
        (배열판은 `PR.finish` 한 번에 둘을 다 하므로 이 보정이 없으면 시험 쪽이 틀린다.)
        """
        orig_finish(t, terminated=terminated)
        events[-1][1]["after_truncated"] = rt.truncated      # events 는 ('B', 사전) 짝이다

    rt.read_cost, rt.select, rt.boundary, rt._update = read_cost, select, boundary, _update
    rt.finish = finish
    days = [DayPlan(index=i, load=load, label="v5-debug", seed=SEED + 1000 * (i + 1),
                    t0=i * DAY_S, n_days=n_days) for i in range(n_days)]
    orig_bind = rt.bind

    def bind(mbt, bridge, meta, archive):
        orig_bind(mbt, bridge, meta, archive)
        tape5.meta, tape5.archive = meta, archive

    rt.bind = bind
    try:
        month_run.run_month(seed=SEED, days=days, ppo=rt)
    except DebugStop:
        pass
    report = rt.report()
    # 한 구간·한 블록에 실제로 쌓인 결정의 최댓값 — 배열 칸(`cmax`)이 이보다 작으면 넘친다
    need = max([0] + [max(e["pending_before"], default=0) for k, e in events if k == "B"]
               + [max(e["interval"]["n_choices"], default=0) for k, e in events
                  if k == "B" and e["interval"] is not None])
    return dict(events=events, updates=updates, report=report, runtime=rt, config=config,
                intervals=intervals, window=window, training=training, duration_s=duration_s,
                tape5=tape5, records=rt.bridge.records, kinds=kinds, need_cmax=need)


_STAGES: dict[tuple, dict] = {}


def _stage(**kwargs) -> dict:
    """무대 하나를 **한 번만** 굴려 여러 시험이 나눠 쓴다 (v5 실행이 2~10초)."""
    key = tuple(sorted(kwargs.items()))
    if key not in _STAGES:
        with pytest.MonkeyPatch.context() as mp:
            _fixed_cargo_patch(mp)
            _STAGES[key] = _run_v5(**kwargs)
    return _STAGES[key]


@pytest.fixture(scope="module")
def stage_spec():
    """명세 무대 — 시드 9900302 · 부하 60 · 2시간 · 최고점 (조각 8 명세가 지정한 작은 무대)."""
    return _stage(load=60, duration_s=7200.0)


@pytest.fixture(scope="module")
def stage_rich():
    """넉넉한 무대 — 부하 300 · 6시간. 세 역할 전부와 본선·재조작·빈주행 비용이 실제로 돈다."""
    return _stage(load=300, duration_s=21600.0, snap_s=3600.0)


@pytest.fixture(scope="module")
def stage_window():
    """학습창 무대 — 창 [1800, 5400). 창 밖 구간은 모으지 않고, 창 끝에서 한 번 비운다."""
    return _stage(load=300, duration_s=7200.0, window=(1800.0, 5400.0))


@pytest.fixture(scope="module")
def stage_eval():
    """고정 운영 무대 — `training=False`. 수집 0 · 갱신 0 이어야 한다."""
    return _stage(load=300, duration_s=7200.0, training=False)


@pytest.fixture(scope="module")
def stage_full():
    """★끝까지 가는 무대 — 하루 + 배수 2시간. `stop_s` 에 안 닿아 `finish(final=True)` 를 밟는다.

    절단(`DebugStop`)으로 끝나는 무대는 마지막 경계가 `final=False` 라 `finish` 갈래가 한 번도
    안 밟힌다. 여기서는 세계가 스스로 끝나므로 마지막 경계가 `final=True` 이고, 배수 구간 때문에
    **60초가 아닌 구간**(172800→180000 꼴)도 한 번 생긴다 — 구간 길이를 상수로 박지 않았는지 본다.
    """
    return _stage(load=60, duration_s=1e9, n_days=1)


# ════════════════════════════════════════════════ Φ 를 배열로 (조각 5 재사용)
def _phi_one(a, o, ext, end_s, gt, idle, vmask, yc, rh):
    orders = empty_orders(a.shape[0])._replace(gate_in_s=a, gate_out_s=o, is_external=ext)
    return PR.read_cost(orders, end_s, vessel_gt=gt, vessel_idle_s=idle, vessel_mask=vmask,
                        yc_extra_move_s=yc, rehandles=rh)


_PHI_BATCH = jax.jit(jax.vmap(_phi_one, in_axes=(0, 0, None, 0, 0, 0, 0, 0, 0)))


def _array_phi(rows: list[dict]):
    """경계 전부의 Φ 를 **한 번에** (세계를 쌓는 v6 본래 쓰임 — `terminal_cost_krw_batch` 와 같은 축)."""
    n = rows[0]["phi"]["n_records"]
    assert all(r["phi"]["n_records"] == n for r in rows), "기록 칸이 런 중에 바뀌면 배치가 안 된다"
    v = max(1, max(len(r["phi"]["vessel_gt"]) for r in rows))
    pad = lambda x, w, fill=0.0: np.pad(np.asarray(x, np.float64), (0, w - len(x)),
                                        constant_values=fill)
    gt = np.stack([pad(r["phi"]["vessel_gt"], v) for r in rows])
    idle = np.stack([pad(r["phi"]["vessel_idle"], v) for r in rows])
    vmask = np.stack([np.arange(v) < len(r["phi"]["vessel_gt"]) for r in rows])
    return _PHI_BATCH(jnp.asarray(np.stack([r["phi"]["a"] for r in rows])),
                      jnp.asarray(np.stack([r["phi"]["o"] for r in rows])),
                      jnp.ones((n,), jnp.bool_),
                      jnp.asarray(np.array([r["phi"]["t"] for r in rows])),
                      jnp.asarray(gt), jnp.asarray(idle), jnp.asarray(vmask),
                      jnp.asarray(np.array([r["phi"]["yc"] for r in rows])),
                      jnp.asarray(np.array([r["phi"]["rh"] for r in rows], np.int32)))


# ════════════════════════════════════════════════ 배열 루프를 같은 테이프로 굴린다
def _cfg_of(stage) -> PR.RuntimeConfig:
    """★칸 `cmax` 는 그 무대가 실제로 요구하는 만큼 잡는다 (`test_required_decision_slots` 참조).

    60초 격자 무대는 64 로 넉넉하지만, **배수 구간(2시간)** 이 붙는 무대는 한 구간에 수백 건이
    쌓인다 — v5 는 파이썬 리스트라 상한이 없다. 조용히 자르는 대신 칸을 그만큼 잡는다.
    """
    return PR.RuntimeConfig.from_ppo_config(
        stage["config"], n_blocks=N_BLOCKS, cmax=max(CMAX, stage["need_cmax"]), amax=AMAX,
        training=stage["training"], stop_s=stage["duration_s"],
        learning_window_s=stage["window"])


def _pad_rows(x: np.ndarray) -> jnp.ndarray:
    out = np.zeros((AMAX, 37))
    out[:len(x)] = x
    return jnp.asarray(out)


def _pad_mask(m: np.ndarray) -> jnp.ndarray:
    out = np.zeros((AMAX,), bool)
    out[:len(m)] = m
    return jnp.asarray(out)


def _replay(stage) -> dict:
    """v5 테이프를 배열 루프에 흘리고, 경계·결정마다 v5 와 맞댄다. 산출은 대조 결과 묶음."""
    cfg = _cfg_of(stage)
    st = PR.new_state(cfg)
    rows = [e for k, e in stage["events"] if k == "B"]
    phi = _array_phi(rows)
    costs = np.asarray(phi.total)
    diffs = []                     # (t, 무엇, v5, 배열)
    b_index = 0
    for kind, ev in stage["events"]:
        if kind == "S":
            st, out = PR.select_record(
                st, cfg, role=ROLE_CODE[ev["role"]], block=ev["block"], t=ev["t"],
                rows=_pad_rows(ev["x"]), mask=_pad_mask(ev["mask"]),
                action=ev["action"], log_prob=ev["log_prob"], n_actions=ev["n"])
            if bool(out.recorded) != ev["recorded"]:
                diffs.append((ev["t"], "select.recorded", ev["recorded"], bool(out.recorded)))
            if int(out.slot) != ev["slot"]:
                diffs.append((ev["t"], "select.slot", ev["slot"], int(out.slot)))
            elif ev["recorded"]:
                # 담긴 칸이 v5 `Choice` 여섯 칸을 그대로 들고 있나 (37칸·마스크 포함)
                b, s, n = ev["block"], int(out.slot), ev["n"]
                p = st.pending
                for name, want, got in (
                        ("role", ROLE_CODE[ev["choice_role"]], int(p.role[b, s])),
                        ("time_s", ev["choice_t"], float(p.time_s[b, s])),
                        ("action", ev["action"], int(p.action[b, s])),
                        ("log_prob", ev["log_prob"], float(p.log_prob[b, s])),
                        ("n_actions", n, int(p.n_actions[b, s]))):
                    if want != got:
                        diffs.append((ev["t"], f"choice.{name}", want, got))
                if not np.array_equal(np.asarray(p.rows[b, s])[:n], ev["choice_x"]):
                    diffs.append((ev["t"], "choice.rows", ev["choice_x"],
                                  np.asarray(p.rows[b, s])[:n]))
                if not np.array_equal(np.asarray(p.mask[b, s])[:n], ev["choice_mask"]):
                    diffs.append((ev["t"], "choice.mask", ev["choice_mask"],
                                  np.asarray(p.mask[b, s])[:n]))
                if np.asarray(p.rows[b, s])[n:].any() or np.asarray(p.mask[b, s])[n:].any():
                    diffs.append((ev["t"], "choice.pad", 0, "패딩 칸에 값이 남았다"))
            continue
        i, b_index = b_index, b_index + 1
        pend_before = np.asarray(st.pending.n)
        if list(pend_before) != ev["pending_before"] and ev["pending_before"]:
            diffs.append((ev["t"], "pending_before", ev["pending_before"], list(pend_before)))
        # v5 는 `bootstrap` 을 갱신 **전** 망으로 잰다 — 갱신이 걸린 경계는 그 값이 관측된다
        values_pre = ev["bootstrap"] if (ev["bootstrap"] is not None
                                         and not ev["terminated"]) else ev["values"]
        # `final=True` 는 v5 `finish(t, terminated=)` 가 부른 것이다 (runtime.py:212-214)
        call = PR.finish if ev["final"] else PR.boundary
        kw = ({} if ev["final"] else {"final": False})
        st, out = call(st, cfg, ev["t"], costs[i], ev["states"], values_pre,
                       terminated=ev["terminated"], **kw)
        # ── 경계 한 줄 대조
        if float(out.cost) != ev["phi"]["total"]:
            diffs.append((ev["t"], "cost", ev["phi"]["total"], float(out.cost)))
        advanced_v5 = ev["started"] and ev["t"] > ev["time_s"] + 1e-6
        if bool(out.advanced) != advanced_v5:
            diffs.append((ev["t"], "advanced", advanced_v5, bool(out.advanced)))
        if bool(out.interval.valid) != ev["collected"]:
            diffs.append((ev["t"], "collected", ev["collected"], bool(out.interval.valid)))
        if bool(out.do_update) != ev["did_update"]:
            diffs.append((ev["t"], "do_update", ev["did_update"], bool(out.do_update)))
        if bool(out.stop) != ev["stopped"]:
            diffs.append((ev["t"], "stop", ev["stopped"], bool(out.stop)))
        want_reward = (ev["interval"]["reward"] if ev["interval"] is not None
                       else (-(ev["phi"]["total"] - ev["cost_krw"]) / 1e6 if advanced_v5 else 0.0))
        if float(out.reward) != want_reward:
            diffs.append((ev["t"], "reward", want_reward, float(out.reward)))
        if ev["bootstrap"] is not None:
            want = np.zeros_like(ev["bootstrap"]) if ev["terminated"] else ev["bootstrap"]
            if not np.array_equal(np.asarray(out.bootstrap), want):
                diffs.append((ev["t"], "bootstrap", want, np.asarray(out.bootstrap)))
        if ev["interval"] is not None:
            iv, av = ev["interval"], out.interval
            for name, want, got in (("start_s", iv["start_s"], float(av.start_s)),
                                    ("end_s", iv["end_s"], float(av.end_s)),
                                    ("terminated", iv["terminated"], bool(av.terminated))):
                if want != got:
                    diffs.append((ev["t"], f"interval.{name}", want, got))
            if not np.array_equal(np.asarray(av.states), iv["states"]):
                diffs.append((ev["t"], "interval.states", iv["states"], np.asarray(av.states)))
            if not np.array_equal(np.asarray(av.values), iv["values"]):
                diffs.append((ev["t"], "interval.values", iv["values"], np.asarray(av.values)))
            if list(np.asarray(av.choices.n)) != iv["n_choices"]:
                diffs.append((ev["t"], "interval.n_choices", iv["n_choices"],
                              list(np.asarray(av.choices.n))))
            else:
                for b, blk in enumerate(iv["choices"]):
                    for j, (role, ct, action, logp) in enumerate(blk):
                        got = (PR.ROLES[int(av.choices.role[b, j])], float(av.choices.time_s[b, j]),
                               int(av.choices.action[b, j]), float(av.choices.log_prob[b, j]))
                        if got != (role, ct, action, logp):
                            diffs.append((ev["t"], f"choice[{b}][{j}]",
                                          (role, ct, action, logp), got))
        if not bool(out.ignored):
            st = PR.refresh_values(st, ev["values"])
        # ── 누적 칸 대조
        for name, want, got in (
                ("intervals", ev["after_intervals"], int(st.intervals)),
                ("learning_intervals", ev["after_learning"], int(st.learning_intervals)),
                ("n_buffered", ev["after_buffered"], int(st.n_buffered)),
                ("total_reward", ev["after_total_reward"], float(st.total_reward)),
                ("learning_reward", ev["after_learning_reward"], float(st.learning_reward)),
                ("cost_krw", ev["after_cost"], float(st.cost_krw)),
                ("time_s", ev["after_time_s"], float(st.time_s)),
                ("truncated", ev["after_truncated"], bool(st.truncated)),
                ("updates", ev["after_updates"], int(st.updates))):
            if want != got:
                diffs.append((ev["t"], name, want, got))
    return dict(state=st, cfg=cfg, diffs=diffs, costs=costs, phi=phi, n_boundaries=len(rows))


def _replayed(stage) -> dict:
    if "_replay" not in stage:
        stage["_replay"] = _replay(stage)
    return stage["_replay"]


def _first(diffs, limit=6):
    return "\n".join(f"  t={t:g} {what}: v5={want!r} 배열={got!r}" for t, what, want, got
                     in diffs[:limit])


# ════════════════════════════════════════════════ (A) 무대 재생 대조
@pytest.mark.parametrize("name", ["spec", "rich", "window", "eval", "full"])
def test_stage_replay_matches_v5(name, request):
    """v5 경계·결정 테이프를 그대로 흘렸을 때 **한 줄도 안 갈린다**."""
    stage = request.getfixturevalue(f"stage_{name}")
    res = _replayed(stage)
    assert not res["diffs"], (f"{name} 무대에서 {len(res['diffs'])} 곳 갈림:\n"
                              + _first(res["diffs"]))
    assert not PR.flag_names(int(res["state"].flags)), "정상 무대인데 위반 비트가 켜졌다"
    assert int(res["state"].pending.overflow.sum()) == 0, f"결정 칸 {res['cfg'].cmax} 부족"


@pytest.mark.parametrize("name", ["spec", "rich", "window", "eval", "full"])
def test_stage_report_matches_v5(name, request):
    """무대 끝의 누적 보고 — 같은 키를 배열 상태에서 되짚어도 같다."""
    stage = request.getfixturevalue(f"stage_{name}")
    res = _replayed(stage)
    st, v5 = res["state"], stage["report"]
    got = PR.report(st, res["cfg"])
    assert got["intervals"] == v5["intervals"]
    assert got["learning_intervals"] == v5["learning_intervals"]
    assert got["team_reward"] == v5["team_reward"]
    assert got["learning_reward"] == v5["learning_reward"]
    assert got["cost_krw"] == v5["cost_krw"]
    assert got["initial_cost_krw"] == v5["initial_cost_krw"]
    assert got["time_s"] == v5["time_s"]
    assert got["truncated"] == v5["truncated"]
    assert got["blocks"] == v5["blocks"]
    assert got["roles"] == v5["roles"]
    assert got["learning_window_s"] == (None if stage["window"] is None else stage["window"])
    assert len(stage["intervals"]) == int(st.learning_intervals)
    assert sum(1 for u in stage["updates"] if u["did"]) == int(st.updates)


def test_stage_spec_is_the_specified_world(stage_spec):
    """명세 무대가 정말 명세 그대로인가 — 21블록 · 120구간 · 2시간 · 최고점."""
    v5 = stage_spec["report"]
    rows = [e for k, e in stage_spec["events"] if k == "B"]
    assert v5["blocks"] == N_BLOCKS and v5["intervals"] == 120
    assert [r["t"] for r in rows] == [60.0 * i for i in range(121)]
    assert v5["truncated"] and v5["time_s"] == 7200.0
    assert stage_spec["runtime"].sample_actions is False
    assert v5["cost_krw"] > 0 and v5["team_reward"] == pytest.approx(-v5["cost_krw"] / 1e6)
    # 절단(DebugStop)이 **정확히 한 번** 일어났다 — `stop` 대조가 빈말이 아니다
    assert sum(1 for r in rows if r["stopped"]) == 1 and rows[-1]["stopped"]
    assert not any(r["final"] for r in rows)          # 절단 무대는 finish 를 안 밟는다


def test_stage_full_walks_the_finish_branch(stage_full):
    """★끝까지 가는 무대 — 마지막 경계가 `final=True` 이고 60초 아닌 구간도 하나 있다."""
    rows = [e for k, e in stage_full["events"] if k == "B"]
    assert sum(1 for r in rows if r["final"]) == 1 and rows[-1]["final"]
    assert not any(r["stopped"] for r in rows)
    lens = {round(r["t"] - r["time_s"], 6) for r in rows[1:] if r["started"]}
    assert lens - {60.0}, f"구간 길이가 60초뿐이면 배수 구간이 안 잡혔다: {sorted(lens)}"
    assert stage_full["report"]["truncated"] is True   # terminated=False → 시간 제한으로 본다
    res = _replayed(stage_full)
    assert bool(res["state"].truncated) and not res["diffs"]


def test_stage_rich_actually_exercises_every_role_and_cost_term(stage_rich):
    """넉넉한 무대에서 판매자·구매자·크레인과 Φ 네 항이 **실제로** 돌았나.

    안 돌았으면 위 대조는 '0 과 0 이 같다' 는 말밖에 안 된다.
    """
    v5 = stage_rich["report"]
    assert all(v5["roles"].get(r, 0) > 0 for r in ("seller", "buyer", "crane")), v5["roles"]
    brk = v5["cost_breakdown"]
    assert brk["c_wait"] > 0 and brk["c_move"] > 0 and brk["c_rehandle"] > 0 and brk["c_vessel"] > 0
    assert sum(1 for u in stage_rich["updates"] if u["did"]) >= 5
    res = _replayed(stage_rich)
    assert int(res["state"].role_counts.sum()) == sum(v5["roles"].values())
    assert int(res["state"].pending.n.sum()) == 0      # 마지막 경계가 비웠다


def test_stage_phi_is_bit_identical_to_v5(stage_spec, stage_rich):
    """★① 보상의 원천 — 경계마다 Φ 가 **비트까지** 같다 (조각 5 `gpu/phi.py` 재사용)."""
    for stage in (stage_spec, stage_rich):
        rows = [e for k, e in stage["events"] if k == "B"]
        res = _replayed(stage)
        want = np.array([r["phi"]["total"] for r in rows])
        assert np.array_equal(res["costs"], want)
        field = {"c_wait": "wait", "c_move": "move", "c_rehandle": "rehandle",
                 "c_vessel": "vessel", "n_trucks": "n_trucks", "n_censored": "n_censored",
                 "mean_turn_time_s": "mean_turn_time_s", "over_ratio": "over_ratio",
                 "p50_turn_time_s": "p50_turn_time_s", "p90_turn_time_s": "p90_turn_time_s",
                 "p99_turn_time_s": "p99_turn_time_s"}
        for key, name in field.items():
            got = np.asarray(getattr(res["phi"].phi, name))
            assert np.array_equal(got, np.array([r["phi"]["breakdown"][key] for r in rows])), key


def test_stage_spec_phi_snapshot_matches_orders_from_records(stage_spec):
    """경계마다 두 열(A·O)만 찍은 것이 `phi.orders_from_records` 전체와 같은가."""
    o = orders_from_records(stage_spec["records"])
    a2, o2 = _record_times(stage_spec["records"])
    assert np.array_equal(np.asarray(o.gate_in_s), a2)
    assert np.array_equal(np.asarray(o.gate_out_s), o2)
    assert bool(np.asarray(o.is_external).all())


def test_stage_window_collects_only_inside_the_window(stage_window):
    """★③ 학습창 — 창 안 구간만 모으고, 창을 나갈 때 한 번 비운다."""
    start, end = stage_window["window"]
    v5 = stage_window["report"]
    starts = [float(iv.start_s) for iv in stage_window["intervals"]]
    assert starts and all(start <= s < end for s in starts)
    assert v5["learning_intervals"] == len(starts) < v5["intervals"]
    res = _replayed(stage_window)
    assert int(res["state"].learning_intervals) == len(starts)
    # ★담길지 말지는 결정 시각이 아니라 **구간 시작 시각**(time_s)이 정한다 (runtime.py:143)
    sels = [e for k, e in stage_window["events"] if k == "S"]
    outside = [e for e in sels if not (start <= e["time_s"] < end)]
    assert outside and len(outside) < len(sels), "창 안/밖 결정이 둘 다 있어야 이 시험이 뭔가를 본다"
    assert all(e["recorded"] == (start <= e["time_s"] < end) for e in sels)


def test_stage_eval_collects_nothing(stage_eval):
    """★고정 운영 — `training=False` 면 구간을 하나도 모으지 않고 갱신도 없다."""
    v5 = stage_eval["report"]
    assert v5["learning_intervals"] == 0 and v5["updates"] == [] and v5["intervals"] > 0
    assert not stage_eval["intervals"]
    res = _replayed(stage_eval)
    assert int(res["state"].learning_intervals) == 0 and int(res["state"].updates) == 0
    assert int(res["state"].n_buffered) == 0
    assert int(res["state"].role_counts.sum()) > 0        # 결정은 실제로 있었다
    assert int(res["state"].pending.n.sum()) == 0


def test_required_decision_slots(stage_spec, stage_rich, stage_window, stage_eval, stage_full):
    """★결정 칸(`cmax`)이 얼마나 필요한가 — **60초 격자는 넉넉하지만 배수 구간은 아니다**.

    60초 구간에 한 블록이 내리는 결정은 수십 건이라 64칸으로 충분하다. 그런데 마지막 날 뒤
    **배수 구간(2시간)** 은 검토 격자가 없어 한 구간이 7,200초가 되고, 그 안에 수백 건이 쌓인다.
    v5 는 파이썬 리스트라 상한이 없지만 배열판은 칸을 미리 잡아야 한다 — 통합 단계는 (가) 칸을
    그만큼 잡거나 (나) 배수 구간에도 검토 경계를 넣어야 한다 (open_issues 로 넘긴다).
    """
    grid = {"명세": stage_spec, "넉넉": stage_rich, "학습창": stage_window, "고정운영": stage_eval}
    for name, stage in grid.items():
        assert stage["need_cmax"] <= CMAX, f"{name} 무대가 칸 {CMAX} 를 넘었다: {stage['need_cmax']}"
    print(f"\n필요한 결정 칸 — 60초 격자 무대 {[s['need_cmax'] for s in grid.values()]} "
          f"· 배수 구간 포함 무대 {stage_full['need_cmax']}")
    assert stage_full["need_cmax"] > CMAX, "배수 구간이 칸을 넘기지 않으면 이 경고는 필요 없다"


def test_crane_action_counter_matches_v5(stage_rich, stage_full):
    """크레인 행동 계수기 — v5 가 **고른 순서 그대로** 세어도 같은 (4,) 분포가 나온다.

    종류 순서(`SERVE·PRE_REHANDLE·REPOSITION·WAIT`)가 어긋나면 보고가 조용히 거짓이 된다.
    """
    for stage in (stage_rich, stage_full):
        kinds, v5 = stage["kinds"], stage["report"]["crane_actions"]
        assert kinds, "크레인 결정이 없으면 이 시험은 아무것도 안 본다"
        st = PR.new_state(_cfg_of(stage))
        for name in kinds:
            st = PR.count_crane_action(st, PR.CRANE_KINDS.index(name))
        got = {k: int(st.crane_actions[i]) for i, k in enumerate(PR.CRANE_KINDS)
               if int(st.crane_actions[i])}
        assert got == v5
        assert int(st.crane_actions.sum()) == len(kinds)


def test_terminated_interval_zeroes_the_bootstrap():
    """★종료 구간 — v5 는 부트스트랩을 0 으로 바꾸고 구간에 `terminated` 를 적는다 (runtime.py:200)."""
    rt = _stub_v5()
    seen = []
    rt._update = lambda bootstrap: seen.append(np.asarray(bootstrap, np.float64).copy())
    rt.boundary(0.0)
    rt.finish(60.0, terminated=True)
    assert rt.buffer[-1].terminated is True and rt.truncated is False
    assert np.array_equal(seen[-1], np.zeros(1))
    st, _ = PR.boundary(PR.new_state(_CFG1), _CFG1, 0.0, 100.0, _S1, jnp.asarray([7.0]))
    st, out = PR.finish(st, _CFG1, 60.0, 220.0, _S1, jnp.asarray([7.0]), terminated=True)
    assert bool(out.interval.terminated) and bool(out.interval.valid)
    assert np.array_equal(np.asarray(out.bootstrap), np.zeros(1))
    assert not bool(st.truncated)
    assert float(out.reward) == rt.buffer[-1].reward


def test_update_trigger_batches_sixty_intervals(stage_rich):
    """갱신 방아쇠 — 구간 60개마다 (`rollout_intervals`) · 마지막·절단에서 한 번 더."""
    dids = [u for u in stage_rich["updates"] if u["did"]]
    assert [u["n"] for u in dids[:-1]] == [60] * (len(dids) - 1)
    assert 1 <= dids[-1]["n"] <= 60
    res = _replayed(stage_rich)
    assert int(res["state"].updates) == len(dids)


# ════════════════════════════════════════════════ (B) 불변식 — v5 예외 ↔ 배열 비트
_CFG1 = PR.RuntimeConfig(n_blocks=1, cmax=4, amax=2, rollout_intervals=2)
_S1, _V1 = jnp.zeros((1, 37)), jnp.zeros((1,))


def _stub_v5(**kw) -> PPORuntime:
    """1블록 세계 — 상태·비용을 함수로 갈아끼운 v5 런타임 (`test_ppo_continuous.py:18-45` 꼴)."""
    rt = PPORuntime(BlockPolicy(), **kw)
    rt.bids, rt.index = ["b"], {"b": 0}
    rt.states_at = lambda t: encode([[0.0]], "state")
    rt.read_cost = lambda t: 100.0 + 2.0 * t
    return rt


def test_invariant_review_time_must_be_finite_and_nonnegative():
    rt = _stub_v5()
    for bad in (-1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            rt.boundary(bad)
        _, out = PR.boundary(PR.new_state(_CFG1), _CFG1, bad, 100.0, _S1, _V1)
        assert int(out.flags) & PR.F_REVIEW_TIME_BAD, bad


def test_invariant_review_clock_must_not_go_backwards():
    rt = _stub_v5()
    rt.boundary(120.0)
    with pytest.raises(RuntimeError, match="backwards"):
        rt.boundary(60.0)
    st, _ = PR.boundary(PR.new_state(_CFG1), _CFG1, 120.0, 100.0, _S1, _V1)
    _, out = PR.boundary(st, _CFG1, 60.0, 100.0, _S1, _V1)
    assert int(out.flags) & PR.F_CLOCK_BACKWARD


def test_invariant_cumulative_cost_must_not_fall():
    """`delta < -1e-5` — 누적 비용이 줄면 회계 자료가 유실됐다는 뜻이다 (runtime.py:177)."""
    rt = _stub_v5()
    rt.read_cost = lambda t: 1000.0 - t          # 줄어드는 비용
    rt.boundary(0.0)
    with pytest.raises(RuntimeError, match="Cumulative cost fell"):
        rt.boundary(60.0)
    st, _ = PR.boundary(PR.new_state(_CFG1), _CFG1, 0.0, 1000.0, _S1, _V1)
    _, out = PR.boundary(st, _CFG1, 60.0, 940.0, _S1, _V1)
    assert int(out.flags) & PR.F_COST_FELL
    # 문턱 바로 안쪽(−1e-5 보다 작은 감소)은 v5 도 통과시킨다 — 비트도 켜지지 않는다
    _, ok = PR.boundary(st, _CFG1, 60.0, 1000.0 - 5e-6, _S1, _V1)
    assert not int(ok.flags) & PR.F_COST_FELL
    rt2 = _stub_v5()
    rt2.read_cost = lambda t: 1000.0 - (5e-6 if t else 0.0)
    rt2.boundary(0.0)
    rt2.boundary(60.0)                            # v5 도 예외를 안 던진다


def test_invariant_nonfinite_cost():
    rt = _stub_v5()
    rt.read_cost = lambda t: float("inf")
    with pytest.raises(FloatingPointError):
        rt.boundary(0.0)
    _, out = PR.boundary(PR.new_state(_CFG1), _CFG1, 0.0, jnp.inf, _S1, _V1)
    assert int(out.flags) & PR.F_COST_NONFINITE
    o = PR.read_cost(empty_orders(1)._replace(gate_in_s=jnp.asarray([0.0]),
                                              is_external=jnp.asarray([True])),
                     jnp.inf)
    assert int(o.flags) & PR.F_COST_NONFINITE     # Φ 자체가 비유한인 경로도 같은 비트


def test_invariant_learning_window_edge_must_have_its_own_review():
    """창 경계가 구간 **안쪽**에 들어오면 그 구간 보상이 창 안/밖에 잘못 붙는다."""
    rt = _stub_v5(learning_window_s=(60.0, 180.0))
    rt.boundary(0.0)
    with pytest.raises(RuntimeError, match="exactly"):
        rt.boundary(120.0)                        # 60 이 (0, 120) 안쪽이다
    cfg = PR.RuntimeConfig(n_blocks=1, cmax=4, amax=2, learning_window_s=(60.0, 180.0))
    st, _ = PR.boundary(PR.new_state(cfg), cfg, 0.0, 100.0, _S1, _V1)
    _, out = PR.boundary(st, cfg, 120.0, 340.0, _S1, _V1)
    assert int(out.flags) & PR.F_WINDOW_EDGE_MISSED
    _, ok = PR.boundary(st, cfg, 60.0, 220.0, _S1, _V1)      # 경계에 딱 맞으면 괜찮다
    assert not int(ok.flags) & PR.F_WINDOW_EDGE_MISSED


def test_invariant_repeated_review_changes_nothing():
    """같은 시각을 두 번 물으면 v5 는 **그냥 돌아간다** — 비용을 두 번 계상하거나 결정을 지우면 안 된다."""
    rt = _stub_v5()
    rt.boundary(0.0)
    rt.boundary(60.0)
    rt.select("seller", "b", 60.0, [[0.0], [1.0]])
    before = (rt.time_s, rt.cost_krw, rt.intervals, len(rt.buffer), len(rt.pending[0]))
    rt.boundary(60.0)
    assert (rt.time_s, rt.cost_krw, rt.intervals, len(rt.buffer),
            len(rt.pending[0])) == before
    st = PR.new_state(_CFG1)
    st, _ = PR.boundary(st, _CFG1, 0.0, 100.0, _S1, _V1)
    st, _ = PR.boundary(st, _CFG1, 60.0, 220.0, _S1, _V1)
    st, _ = PR.select_record(st, _CFG1, role=0, block=0, t=60.0, rows=jnp.zeros((2, 37)),
                             mask=jnp.asarray([True, True]), action=0, log_prob=-0.5, n_actions=2)
    st2, out = PR.boundary(st, _CFG1, 60.0, 220.0, _S1, _V1)
    assert bool(out.ignored) and not bool(out.interval.valid) and not bool(out.do_update)
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(st2, st)
               if not isinstance(a, PR.PendingTape))
    assert int(st2.pending.n[0]) == int(st.pending.n[0]) == 1


def test_invariant_decision_before_initial_boundary():
    rt = _stub_v5()
    with pytest.raises(RuntimeError, match="initial boundary"):
        rt.select("seller", "b", 0.0, [[0.0], [1.0]])
    assert rt.role_counts == {} and not hasattr(rt, "pending")   # 던지기 전이라 안 셌다
    st2, out = PR.select_record(PR.new_state(_CFG1), _CFG1, role=0, block=0, t=0.0,
                                rows=jnp.zeros((2, 37)), mask=jnp.asarray([True, True]),
                                action=0, log_prob=0.0, n_actions=2)
    assert int(out.flags) & PR.F_DECISION_NO_BOUNDARY
    assert int(st2.role_counts.sum()) == 0 and int(st2.pending.n.sum()) == 0


def test_invariant_decision_before_interval_start():
    rt = _stub_v5()
    rt.boundary(120.0)
    with pytest.raises(RuntimeError, match="precedes"):
        rt.select("seller", "b", 60.0, [[0.0], [1.0]])
    assert rt.role_counts == {} and not any(rt.pending)           # 던지기 전이라 안 셌다
    st, _ = PR.boundary(PR.new_state(_CFG1), _CFG1, 120.0, 100.0, _S1, _V1)
    st2, out = PR.select_record(st, _CFG1, role=0, block=0, t=60.0, rows=jnp.zeros((2, 37)),
                               mask=jnp.asarray([True, True]), action=0, log_prob=0.0, n_actions=2)
    assert int(out.flags) & PR.F_DECISION_EARLY
    assert int(st2.role_counts.sum()) == 0 and int(st2.pending.n.sum()) == 0
    for bad in (-1.0, float("nan")):
        with pytest.raises(ValueError):
            rt.select("seller", "b", bad, [[0.0], [1.0]])
        _, o2 = PR.select_record(st, _CFG1, role=0, block=0, t=bad, rows=jnp.zeros((2, 37)),
                                 mask=jnp.asarray([True, True]), action=0, log_prob=0.0,
                                 n_actions=2)
        assert int(o2.flags) & PR.F_DECISION_TIME_BAD, bad


def test_invariant_empty_action_mask():
    rt = _stub_v5()
    rt.boundary(0.0)
    with pytest.raises(ValueError, match="mask"):
        rt.select("crane", "b", 0.0, [[0.0], [1.0]],
                  mask=torch.zeros(2, dtype=torch.bool))
    assert rt.role_counts == {} and not any(rt.pending)           # 던지기 전이라 안 셌다
    st, _ = PR.boundary(PR.new_state(_CFG1), _CFG1, 0.0, 100.0, _S1, _V1)
    st2, out = PR.select_record(st, _CFG1, role=2, block=0, t=0.0, rows=jnp.zeros((2, 37)),
                               mask=jnp.asarray([False, False]), action=0, log_prob=0.0,
                               n_actions=2)
    assert int(out.flags) & PR.F_MASK_EMPTY
    assert int(st2.role_counts.sum()) == 0 and int(st2.pending.n.sum()) == 0


def test_pending_overflow_is_loud_not_silent():
    """칸이 넘치면 **조용히 버리지 않는다** — v5 는 파이썬 리스트라 상한이 없다 (v6 의 fail-loud)."""
    st, _ = PR.boundary(PR.new_state(_CFG1), _CFG1, 0.0, 100.0, _S1, _V1)
    for i in range(_CFG1.cmax + 2):
        st, out = PR.select_record(st, _CFG1, role=0, block=0, t=0.0, rows=jnp.zeros((2, 37)),
                                   mask=jnp.asarray([True, True]), action=i % 2, log_prob=0.0,
                                   n_actions=2)
    assert int(st.pending.n[0]) == _CFG1.cmax and int(st.pending.overflow[0]) == 2
    assert int(st.flags) & PR.F_PENDING_OVERFLOW
    assert int(st.role_counts[0]) == _CFG1.cmax + 2        # 계수기는 담기든 말든 센다
    # 후보가 칸(`amax`)보다 많으면 v5 는 리스트라 다 담지만 배열판은 잘린다 → 비트로 알린다
    _, out = PR.select_record(st, _CFG1, role=0, block=0, t=0.0, rows=jnp.zeros((2, 37)),
                              mask=jnp.asarray([True, True]), action=0, log_prob=0.0,
                              n_actions=_CFG1.amax + 1)
    assert int(out.flags) & PR.F_ACTION_OVERFLOW


def test_boundary_and_select_are_jittable_and_vmappable():
    """통합 단계는 이 둘을 `jit` 안에서, 세계를 쌓아(`vmap`) 부른다 — 그때도 같은 답인가.

    `cfg` 는 파이썬 값이라 정적으로 닫아 넣는다 (`static_argnums` 자리).
    """
    cfg = PR.RuntimeConfig(n_blocks=1, cmax=4, amax=2, rollout_intervals=2)
    step = lambda st, t, c: PR.boundary(st, cfg, t, c, _S1, _V1)
    pick = lambda st, t, a: PR.select_record(
        st, cfg, role=0, block=0, t=t, rows=jnp.zeros((2, 37)),
        mask=jnp.asarray([True, True]), action=a, log_prob=-0.25, n_actions=2)
    jstep, jpick = jax.jit(step), jax.jit(pick)

    eager = PR.new_state(cfg)
    jitted = PR.new_state(cfg)
    for k, t in enumerate((0.0, 60.0, 120.0)):
        eager, oe = step(eager, t, 100.0 + 2.0 * t)
        jitted, oj = jstep(jitted, t, 100.0 + 2.0 * t)
        assert float(oe.reward) == float(oj.reward) and float(oe.cost) == float(oj.cost)
        eager, _ = pick(eager, t, k % 2)
        jitted, _ = jpick(jitted, t, k % 2)
    for a, b in zip(eager, jitted):
        leaves = zip(a, b) if isinstance(a, PR.PendingTape) else [(a, b)]
        for x, y in leaves:
            assert np.array_equal(np.asarray(x), np.asarray(y))

    # ── 세계 B=3 을 쌓아도 낱개와 같다 (비용만 다르게 준다)
    costs = jnp.asarray([100.0, 250.0, 400.0])
    stacked = jax.tree.map(lambda x: jnp.stack([x] * 3), PR.new_state(cfg))
    vstep = jax.jit(jax.vmap(lambda st, t, c: PR.boundary(st, cfg, t, c, _S1, _V1),
                             in_axes=(0, None, 0)))
    stacked, _ = vstep(stacked, 0.0, costs)
    stacked, out = vstep(stacked, 60.0, costs + 60.0)
    singles = []
    for c in (100.0, 250.0, 400.0):
        s = PR.new_state(cfg)
        s, _ = step(s, 0.0, c)
        s, o = step(s, 60.0, c + 60.0)
        singles.append(float(o.reward))
    assert [float(r) for r in out.reward] == singles
    assert list(np.asarray(stacked.intervals)) == [1, 1, 1]


def test_config_rejects_impossible_settings():
    for kw in (dict(learning_window_s=(-1.0, 60.0)), dict(learning_window_s=(60.0, 60.0)),
               dict(learning_window_s=(120.0, 60.0)), dict(learning_window_s=(0.0, math.inf)),
               dict(stop_s=0.0), dict(stop_s=math.inf), dict(n_blocks=0), dict(cmax=0),
               dict(reward_scale_krw=0.0)):
        with pytest.raises(ValueError):
            PR.RuntimeConfig(**kw)
        if "learning_window_s" in kw:
            with pytest.raises(ValueError, match="window"):
                PPORuntime(BlockPolicy(), learning_window_s=kw["learning_window_s"])


# ════════════════════════════════════════════════ (C) 학습창 스칼라 논리 (대체 입력)
def test_window_stub_matches_v5():
    """v5 회귀시험 `test_ppo_continuous.py:18-45` 와 **같은 입력**을 둘에 먹인다.

    비용을 `100 + 2t`, 상태를 `[[0]]` 로 갈아끼운 1블록 세계 — 엔진 없이 학습창·구간·갱신
    시점만 본다. 기대값을 적지 않고 v5 를 그 자리에서 함께 굴려 받는다.
    """
    window = (60.0, 180.0)
    times = (0.0, 60.0, 120.0, 180.0, 240.0)
    rt = _stub_v5(learning_window_s=window)
    batches, at_times, bootstraps = [], [], []

    def collect(bootstrap):
        if rt.buffer:
            batches.append([float(r.start_s) for r in rt.buffer])
            at_times.append(float(rt.time_s))
            bootstraps.append(np.asarray(bootstrap, np.float64).copy())
            rt.buffer.clear()
    rt._update = collect
    pend = []
    for t in times:
        rt.boundary(t)
        rt.select("seller", "b", t, [[0.0], [1.0]])
        pend.append(len(rt.pending[0]))
    rt.finish(300.0)

    cfg = PR.RuntimeConfig(n_blocks=1, cmax=4, amax=2, learning_window_s=window)
    st = PR.new_state(cfg)
    a_batches, a_pend, a_updates = [], [], []
    open_batch: list[float] = []
    for t in times + (300.0,):
        final = t == 300.0
        cost = 100.0 + 2.0 * t
        fn = PR.finish if final else PR.boundary
        st, out = fn(st, cfg, t, cost, _S1, _V1)
        if bool(out.interval.valid):
            open_batch.append(float(out.interval.start_s))
        if bool(out.do_update):
            a_batches.append(list(open_batch))
            a_updates.append(float(t))
            open_batch = []
        if not final:
            st, _ = PR.select_record(st, cfg, role=0, block=0, t=t, rows=jnp.zeros((2, 37)),
                                     mask=jnp.asarray([True, True]), action=0, log_prob=0.0,
                                     n_actions=2)
            a_pend.append(int(st.pending.n[0]))

    assert a_pend == pend
    assert a_batches == batches
    assert a_updates == at_times      # 갱신 시점 — 학습창을 나가는 t=180 에서 한 번 비운다
    assert int(st.intervals) == rt.intervals
    assert int(st.learning_intervals) == rt.learning_intervals
    assert float(st.total_reward) == rt.total_reward
    assert float(st.learning_reward) == rt.learning_reward
    assert bool(st.truncated) == rt.truncated


def test_collecting_at_matches_v5_scalar():
    """`collecting_at` — 창·학습여부 조합 전부에서 v5 와 같은 참·거짓."""
    for window in (None, (60.0, 180.0)):
        for training in (True, False):
            rt = PPORuntime(BlockPolicy(), training=training, learning_window_s=window)
            cfg = PR.RuntimeConfig(n_blocks=1, cmax=2, amax=2, training=training,
                                   learning_window_s=window)
            for t in (0.0, 59.999, 60.0, 120.0, 179.999, 180.0, 1e6):
                assert bool(PR.collecting_at(cfg, t)) == rt.collecting_at(t), (window, training, t)


# ════════════════════════════════════════════════ (D) 격자 사진 (_MonthTape)
def _vessel_table(tape, got: dict, names: dict, t) -> dict:
    """배열 표 → `{배 이름: (GT, 유휴초)}` — v5 사전과 맞대기 위해 번호를 이름으로 되돌린다."""
    back = {i: n for n, i in names.items()}
    i = int(PR._tape_index(tape, t))
    ids = np.asarray(tape.vessel_id[max(i, 0)])
    return {back[int(vid)]: (float(g), float(idl))
            for vid, g, idl, m in zip(ids, np.asarray(got["vessel_gt"]),
                                      np.asarray(got["vessel_idle_s"]),
                                      np.asarray(got["vessel_mask"])) if m}


def test_counter_tape_read_matches_v5(stage_rich):
    """사진 읽기 — `t` 이하 가장 늦은 사진. v5 `_MonthTape.read` 와 같은 표·같은 수."""
    tape5 = stage_rich["tape5"]
    assert len(tape5.at) >= 4, "사진이 너무 적으면 이 시험은 아무것도 안 본다"
    tape, names = PR.counter_tape_from_v5(tape5)
    keys = sorted(tape5.at)
    probes = [keys[0] - 1.0] + list(keys) + [k + 30.0 for k in keys]
    assert any(v for v, _, _ in (tape5.read(t) for t in keys)), "본선 표가 빈 무대면 표 대조가 없다"
    for t in probes:
        v5_v, v5_yc, v5_rh = tape5.read(t)
        got = PR.tape_read(tape, t)
        empty = t < keys[0]
        assert float(got["yc_extra_move_s"]) == (0.0 if empty else v5_yc), t
        assert int(got["rehandles"]) == (0 if empty else v5_rh), t
        assert _vessel_table(tape, got, names, t) == ({} if empty else v5_v), t


def test_counter_tape_diff_matches_v5(stage_rich):
    """★하루치 Φ 의 원료 — 두 사진의 **차분**이 v5 `_MonthTape.diff` 와 같다."""
    tape5 = stage_rich["tape5"]
    tape, names = PR.counter_tape_from_v5(tape5)
    keys = sorted(tape5.at)
    pairs = [(keys[0], keys[-1]), (keys[0], keys[1]), (keys[-2], keys[-1]),
             (keys[0] - 1.0, keys[-1])]
    for t0, t1 in pairs:
        v5_v, v5_yc, v5_rh = tape5.diff(t0, t1)
        got = PR.tape_diff(tape, t0, t1)
        assert float(got["yc_extra_move_s"]) == v5_yc, (t0, t1)
        assert int(got["rehandles"]) == v5_rh, (t0, t1)
        assert _vessel_table(tape, got, names, t1) == v5_v, (t0, t1)


def test_day_phi_from_tape_matches_v5(stage_rich):
    """★사진 차분 + 하루치 기록으로 낸 Φ 가 v5 `_phi_of_day` 와 **비트까지** 같다."""
    tape5, records = stage_rich["tape5"], stage_rich["records"]
    keys = sorted(tape5.at)
    t0, t1 = keys[0], keys[-1]
    end_s = t1
    v5 = month_run._phi_of_day(records, 0, end_s=end_s, tape=tape5, t0=t0, t1=t1)
    day = month_run._day_records(records, 0)
    assert day, "첫날 기록이 없으면 이 시험은 아무것도 안 본다"
    tape, _ = PR.counter_tape_from_v5(tape5)
    got = PR.read_cost(orders_from_records(day), end_s, **PR.tape_diff(tape, t0, t1))
    d = got.phi.as_dict()
    assert d["phi_krw"] == v5.total
    assert (d["c_wait"], d["c_move"], d["c_rehandle"], d["c_vessel"]) == (
        v5.wait, v5.move, v5.rehandle, v5.vessel)
    assert (d["n_trucks"], d["n_censored"]) == (v5.n_trucks, v5.n_censored)


def test_zz_report(stage_spec, stage_rich, stage_window, stage_eval, stage_full):
    """무대 다섯의 규모를 한 줄로 남긴다 — '무엇이 실제로 대조됐나' 의 증거."""
    lines = []
    for name, stage in (("명세(60·2h)", stage_spec), ("넉넉(300·6h)", stage_rich),
                        ("학습창", stage_window), ("고정운영", stage_eval),
                        ("끝까지(60·1일+배수)", stage_full)):
        res = _replayed(stage)
        v5 = stage["report"]
        lines.append(f"{name}: 경계 {res['n_boundaries']} · 결정 {sum(v5['roles'].values())} "
                     f"({v5['roles']}) · 구간 {v5['intervals']} · 학습구간 "
                     f"{v5['learning_intervals']} · 갱신 {int(res['state'].updates)} · "
                     f"Φ {v5['cost_krw']:,.0f}원 · 갈림 {len(res['diffs'])}")
    print("\n" + "\n".join(lines))
    assert all("갈림 0" in line for line in lines)
