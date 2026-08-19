"""YR-170 — 5차 계약(24시간 이중 피크) 무대에서 **판매 PPO 학습**.

사용자 결정 2026-08-11: *"학습부터 해서 학습이 안 됐다가 나와야 디버깅을 하는 것"*.
그동안 0B 의 논리("손으로 만든 견적이 좋은 판매를 못 고르면 학습해도 안 된다")를
전제로 견적을 진단해 왔으나, **학습기는 견적을 보상으로 쓰지 않는다** —
보상 = `−(Φ_final − Φ(t_k⁻)) + V(s_end)` 로 **실현 터미널 총비용**에서 나온다
(yr151_transfer_ppo.build_batch). 견적의 부정확은 학습을 막을 이유가 아니라
학습이 할 일이 남았다는 뜻이다. 그래서 무대만 5차로 옮겨 **먼저 돌린다**.

바뀌는 것은 **무대뿐**이다 — 학습 루프·보상·PPO 갱신·재현 스탬프는
`yr151_transfer_ppo.train_one(episode_fn=...)` 을 그대로 공유한다.

  4차(고정 WIP)                     → 5차(도착률 24h 이중 피크)
  build_fixed_wip + hotspot         → build_diurnal (명단 사전 확정)
  WipAdmissionController(pool)      → ScheduledAnnouncer (개방 루프)
  ObservationContract() 5h          → OBS_24H (워밍업 2h + 측정 22h, 배수 2h)
  _sim_from(scn)                    → _sim_from(scn, prof) + ensure_time_ledger

성능 주장 없음 — 이 파일은 **학습이 되는지 보는 실행**이다. 판정은 별도 사전등록.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                    _apply, _wait_of)
from ..integrated.candidates import CandidateGenerator
from ..integrated.multiblock import MultiBlockTerminal
from ..integrated.policy_config import ADOPTED_C0_GUARD, LEGACY_DEFAULT
from ..integrated.profiles import build_h21_profile
from ..integrated.sell_review import (ANNOUNCE_LEAD_S, UnifiedSellOrchestrator)
from ..integrated.terminal_stream import (DIURNAL_DAY_TOTAL, OBS_24H,
                                          ScheduledAnnouncer, TerminalStreamParams,
                                          admission_epochs, build_diurnal,
                                          ensure_time_ledger)
from ..integrated.yard_layout import terminal_layout
from .yr088_joint_rl import LEVEL
from .yr149_load_cells import _sim_from
from .yr151_transfer_ppo import (EXEC_TS_DEFAULT as EXEC_TS,
                                 AdoptedExecFleet, PhiRecorder, _Chain,
                                 build_joint_transitions, load_adopted_execution_head,
                                 phi_terminal, train_one)

OUT = Path("outputs/reports/yr170_sell_ppo_diurnal")
SPEC = ".claude/docs/dashboard-task-specs/YR-170-sell-ppo-diurnal.md"
# 학습 시드 — 앞 3개는 기존 실행분과 동일(비교 가능). 뒤 3개는 유휴 코어용 확장.
# 에피소드 1개 = 1 코어(이산사건 시뮬레이션은 순차 진행)이므로 24코어 환경에서
# 시드 6 × 에피소드 4 = 24 동시 실행이 코어를 다 쓴다.
TRAIN_SEEDS = (8_400_000, 8_500_000, 8_600_000,
               8_700_000, 8_800_000, 8_900_000)   # 자격·관찰·파일럿과 교집합 0
SPACE_ONLY = True      # 시간 판매 축 차단 (YR-171 로 계약 교체 전까지)
# ★yr139 앵커 복원 (YR-170 진단 2026-08-12) — 셋 다 "선언돼 있었으나 승계에서
#   떨어진" 항목이다. 새 가설이 아니라 복원이다.
RET_SCALE = 1000.0     # 24h Φ≈2,300~3,000 → 수익 목표를 [−3, 0] 으로
MINIBATCH = 512        # 전배치 1 step → iteration 당 수백 step
GRAD_CLIP = 1.0        # yr139 에 있던 기울기 상한
# ★자기참조 제거 (YR-170 발산 진단 2026-08-12): 목표식이
#   r2go = −(Φ_final − Φ_pre)/SCALE + V(s_end) 라 critic 이 **자기 출력을 목표에
#   더한다**. γ=1 이라 수축이 없어 V↑ → 목표↑ → V↑ 의 되먹임이 발산한다
#   (실측: v_loss 0.09 → 2,700만, critic 출력 −29,087). 정규화 전에는 boot 가
#   목표의 0.06% 라 잠복해 있었다. 24시간 에피소드는 사실상 완결이고 검열 비용은
#   Φ 가 이미 반영하므로 종료 bootstrap 을 쓰지 않는다(순수 MC).
USE_END_BOOTSTRAP = False


def run_episode_diurnal(seed: int, policy, kf, *,
                        obs=None, exec_head: str = "adopted",
                        exec_fleet=None, exec_config=None,
                        day_plan_public: bool = False,
                        time_slots: bool = False, buy_net=None, q_scorer=None,
                        sell_margin: float = 0.0, reuse_handoff: bool = False,
                        margin_space: float | None = None,
                        margin_time: float | None = None,
                        _return_mbt: bool = False,
                        _extra_review=None) -> dict:
    """5차 계약 1 에피소드 — 4차 `run_episode` 와 **반환 형태 동일**(학습 루프 공유).

    무대만 다르다: 도착 명단 사전 확정 + 개방 루프 투입 + 24시간 이중 피크.
    투입이 정책과 무관하므로 두 정책이 같은 트럭·같은 시각을 받는다.
    """
    if exec_head not in ("adopted", "sf"):
        raise ValueError(f"exec_head 는 'adopted'|'sf' 만 허용: {exec_head!r}")
    if exec_head == "adopted":
        if exec_config is None:
            raise ValueError("adopted execution requires explicit exec_config")
        if exec_config != ADOPTED_C0_GUARD:
            raise ValueError("performance harness requires ADOPTED_C0_GUARD")
    else:
        exec_config = LEGACY_DEFAULT

    obs = obs or OBS_24H
    prof, layout = build_h21_profile(), terminal_layout()
    params = TerminalStreamParams(load_4h=DIURNAL_DAY_TOTAL)
    built = build_diurnal(prof, seed, obs=obs, layout=layout, params=params,
                          background_seed=seed)
    mbt = MultiBlockTerminal(
        {b: ensure_time_ledger(_sim_from(s, prof))
         for b, s in built["scenarios"].items()},
        extra_review_epochs=admission_epochs(obs))
    # ★YR-171-A 정보 계약 — **명시적 opt-in**. 기본값 False 는 현 계약(30분 통지)을
    # 바이트 동일하게 보존한다: YR-174 판정 결과가 조용히 바뀌면 안 된다.
    if day_plan_public:
        from ..integrated.day_plan import attach as _attach_day_plan
        _attach_day_plan(mbt, built["schedule"])
    ann = ScheduledAnnouncer(built["schedule"], lead_s=ANNOUNCE_LEAD_S,
                             end_s=built["sim_end_s"])
    orch = UnifiedSellOrchestrator(policy, layout, kf,
                                   time_slots=time_slots, buy_net=buy_net,
                                   q_scorer=q_scorer, sell_margin=sell_margin,
                                   reuse_handoff=reuse_handoff,
                                   margin_space=margin_space,
                                   margin_time=margin_time,
                                   dry_run=(getattr(policy, "mode", None) == "shadow"))
    rec = PhiRecorder()

    gens: dict[int, CandidateGenerator] = {}
    exc = {"n": 0}
    if exec_head == "adopted":
        if exec_fleet is None:
            exec_fleet = AdoptedExecFleet(
                *load_adopted_execution_head(), config=exec_config)
        elif exec_fleet.config != exec_config:
            raise ValueError("exec_fleet and exec_config do not match")

        def exec_policy(sim, dp):
            g = gens.setdefault(id(sim), CandidateGenerator(config=exec_config))
            gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
            _apply(sim, exec_fleet.get(sim).decide(sim, dp, gb))   # 예외 = 즉시 실격
    else:
        pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")

        def exec_policy(sim, dp):
            g = gens.setdefault(id(sim), CandidateGenerator(config=exec_config))
            gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
            try:
                _apply(sim, pol.decide(sim, dp, gb))
            except Exception:
                exc["n"] += 1
                _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})

    # Φ 는 **행동 직전** 기록: 투입 → Φ → 판매 (4차 계약과 같은 순서 계약)
    # _extra_review 는 **관측 전용** 훅(데이터 적재 등) — 판매 확정 전에 상황을
    # 찍어야 하므로 Φ 기록 뒤·판매 앞에 끼운다. 결정에 개입하지 않는다.
    parts = ([ann, rec, _Wrap(_extra_review), orch] if _extra_review
             else [ann, rec, orch])
    mbt.run(exec_policy, review_fn=_Chain(*parts).review)

    # 장부 보존 — 5차 계약판(채움 개념 없음): 등록 = 투입, 누락 0
    n_rows = sum(len(s.time_ledger.records) for s in mbt.blocks.values()
                 if getattr(s, "time_ledger", None) is not None)
    n_skip = sum(1 for e in ann.ledger if e["event"] in ("SKIP", "SKIP_TAIL"))
    if n_rows != ann.n_admitted or n_skip or ann.n_admitted != len(built["schedule"]):
        raise RuntimeError(
            f"장부 불일치 — 등록 {n_rows} / 투입 {ann.n_admitted} / "
            f"계획 {len(built['schedule'])} / 누락 {n_skip} (에피소드 실격)")

    phi_final = phi_terminal(mbt, obs.observe_s)
    from ..integrated.transfer_head import critic_input
    end_inputs = {b: critic_input(mbt, b, obs.observe_s, 0, layout=layout)
                  for b in mbt.blocks}
    res_mbt = {"_mbt": mbt} if _return_mbt else {}   # 진단 전용(비용 분해)
    return {**res_mbt,
            "phi": rec.samples, "phi_final": phi_final, "sell_ledger": orch.ledger,
            "exec_policy_config": exec_config.as_dict(),
            "end_inputs": end_inputs,
            "joint": build_joint_transitions(policy.trail, orch.ledger, rec.samples,
                                             phi_final, obs.observe_s),
            "q_rows": orch.q_rows,
            "n_space": orch.n_space, "n_time": orch.n_time,
            "policy_exceptions": exc["n"], "admitted": ann.n_admitted}


class SpaceOnly:
    """정책이 **반입(GATE_IN)만 지명**하게 감싼다 — 시간 판매 축 차단.

    축 분리 진단(2026-08-13): 시간 판매는 현 계약(고정 +15분)에서 1건당 +14.28분
    **구조적 손해**다(기사 외부 대기가 내부와 같은 단가 → 순이득 = 절약분 − 15분,
    고정 이연은 혼잡 첨두를 고를 수 없다). 그 축을 켠 채 학습시키는 것은 정책에게
    불가능한 것을 배우라고 하는 셈이라, 계약을 고치기 전(YR-171)까지 끈다.
    """

    def __init__(self, inner):
        self.inner = inner
        self.mode = getattr(inner, "mode", "live")

    @property
    def trail(self):
        return self.inner.trail

    def decide(self, mbt, src: str, cands: list, t: float) -> str | None:
        c = [x for x in cands if x[2] == "GATE_IN"]
        return self.inner.decide(mbt, src, c, t) if c else None


class _Wrap:
    """콜러블을 _Chain 이 요구하는 .review 인터페이스로 감싼다."""

    def __init__(self, fn):
        self.review = fn


class KeepAllTrail:
    """기준선 K — 아무것도 팔지 않는다. 학습 루프와 같은 인터페이스(trail 보유)."""

    mode = "live"

    def __init__(self):
        self.trail: list[dict] = []

    def decide(self, mbt, src: str, cands: list, t: float) -> str | None:
        return None


def baseline(seed: int, *, exec_head: str = "adopted") -> dict:
    """전건 KEEP 의 Φ — 학습 결과를 읽을 기준점(성능 판정 아님, 참조값)."""
    from .yr151_transfer_ppo import load_kf
    ep = run_episode_diurnal(
        seed, KeepAllTrail(), load_kf(), exec_head=exec_head,
        exec_config=(ADOPTED_C0_GUARD if exec_head == "adopted" else None))
    return {"seed": seed, "exec_head": exec_head,
            "phi_final": round(ep["phi_final"], 4),
            "n_space": ep["n_space"], "n_time": ep["n_time"],
            "admitted": ep["admitted"]}


def _episode_worker(args) -> dict:
    """자식 프로세스 1개 = 에피소드 1개. 같은 iteration 은 **같은 가중치**를 쓴다
    (on-policy 계약 유지 — 병렬은 순서만 바꾸고 정책 스냅샷은 동일)."""
    import torch
    import torch.multiprocessing as _mp
    from ..integrated.transfer_head import (PpoSellPolicy, TransferActor,
                                            TransferCritic)
    from .yr151_transfer_ppo import load_kf
    torch.set_num_threads(1)                      # 프로세스 다중 실행 시 스레드 경합 방지
    # 텐서를 프로세스 경계로 넘길 때 기본(file_descriptor) 방식은 fd 를 소진해
    # 'Too many open files' 로 죽는다(실측 2026-08-11). file_system 으로 고정.
    _mp.set_sharing_strategy("file_system")
    seed, sd_a, sd_c, pol_seed = args
    actor, critic = TransferActor(), TransferCritic()
    actor.load_state_dict(sd_a)
    critic.load_state_dict(sd_c)
    pol = PpoSellPolicy(actor, critic, mode="live", sample=True,
                        seed=pol_seed, layout=terminal_layout())
    ep = run_episode_diurnal(seed, SpaceOnly(pol) if SPACE_ONLY else pol,
                             load_kf(), exec_config=ADOPTED_C0_GUARD)
    # 텐서는 detach().clone() 으로 공유메모리에서 떼어 보낸다(fd 누수 방지)
    trail = [{k: (v.detach().clone() if hasattr(v, "detach") else v)
              for k, v in tr.items()} for tr in pol.trail]
    end_inputs = {b: x.detach().clone() for b, x in ep["end_inputs"].items()}
    return {"trail": trail, "joint": ep["joint"], "phi_final": ep["phi_final"],
            "end_inputs": end_inputs, "n_space": ep["n_space"],
            "n_time": ep["n_time"]}


def train_parallel(ts: int, *, n_iter: int, eps_per_iter: int,
                   out_root: Path = OUT) -> Path:
    """학습 루프 — iteration 안의 에피소드를 **프로세스 병렬**로 돌린다.

    에피소드 1회가 약 10분이라 순차로는 시드당 27시간이다. 한 iteration 의
    에피소드들은 같은 정책 스냅샷을 쓰므로 병렬 실행이 on-policy 계약을 깨지 않는다.
    갱신·보상·재현 스탬프는 yr151 의 함수를 그대로 쓴다(잣대 불변).
    """
    import torch
    import torch.multiprocessing as _mp
    from concurrent.futures import ProcessPoolExecutor
    _mp.set_sharing_strategy("file_system")
    from ..integrated.repro import code_dirty, repro_stamp
    from ..integrated.transfer_head import TransferActor, TransferCritic
    from .yr151_transfer_ppo import (CLIP, ENT, LR, build_batch,
                                     exec_config_hash, load_adopted_execution_head,
                                     ppo_update)
    # 앵커 복원값은 스탬프에 남긴다(무엇으로 돌렸는지 산출물만 보고 알 수 있게)
    torch.set_num_threads(1)
    torch.manual_seed(ts)
    actor, critic = TransferActor(), TransferCritic()
    opt_a = torch.optim.Adam(actor.parameters(), lr=LR)
    opt_c = torch.optim.Adam(critic.parameters(), lr=LR)
    exec_actor, _ = load_adopted_execution_head()
    exec_hash0 = exec_config_hash(exec_actor, EXEC_TS, ADOPTED_C0_GUARD)
    out = out_root / f"ppo_s{ts}"
    out.mkdir(parents=True, exist_ok=True)
    hist: list[dict] = []
    for it in range(n_iter):
        sd_a = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
        sd_c = {k: v.detach().cpu() for k, v in critic.state_dict().items()}
        jobs = [(ts + it * eps_per_iter + e, sd_a, sd_c, ts + it * 100 + e)
                for e in range(eps_per_iter)]
        with ProcessPoolExecutor(max_workers=eps_per_iter) as pool:
            eps = list(pool.map(_episode_worker, jobs))
        batch_all: list[dict] = []
        for ep in eps:
            with torch.no_grad():          # 기록용 — 사용 여부는 아래 플래그
                v_end = {b: float(critic(x).item())
                         for b, x in ep["end_inputs"].items()}
            batch_all += build_batch(
                ep["trail"], ep["joint"], ep["phi_final"],
                v_end if USE_END_BOOTSTRAP else None, ret_scale=RET_SCALE)
        stats = ppo_update(actor, critic, opt_a, opt_c, batch_all,
                           minibatch=MINIBATCH, seed=ts + it,
                           grad_clip=GRAD_CLIP)
        if exec_config_hash(exec_actor, EXEC_TS, ADOPTED_C0_GUARD) != exec_hash0:
            raise RuntimeError("실행 구성이 변했다 — 계약 위반")
        hist.append({"iter": it, **stats,
                     "phi_final_mean": sum(e["phi_final"] for e in eps) / len(eps),
                     "n_space_mean": sum(e["n_space"] for e in eps) / len(eps),
                     "n_time_mean": sum(e["n_time"] for e in eps) / len(eps),
                     "n_batch": len(batch_all)})
        # ★진행 중 관측 가능하게 매 iteration 저장(구판은 종료 후 1회 — 장시간 런에서
        #   무슨 일이 일어나는지 볼 수 없었다). 최종 스탬프는 종료 시 덮어쓴다.
        (out / "train.json").write_text(json.dumps(
            {"history": hist, "in_progress": True, "n_iter_target": n_iter},
            ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()},
                   out / "net.pt")
    stamp = repro_stamp(
        experiment="YR-170 판매 PPO — 5차 계약 24시간 이중 피크 무대(병렬 학습)",
        seeds={"train": [ts], "exec_head": [EXEC_TS]},
        params={"N_ITER": n_iter, "EPS_PER_ITER": eps_per_iter,
                "LEAD_S": ANNOUNCE_LEAD_S, "stage": "diurnal_24h",
                "exec_head": f"adopted C0+guard (yr143 confirm ppo_s{EXEC_TS})",
                "exec_head_hash": exec_hash0,
                "exec_policy_config": ADOPTED_C0_GUARD.as_dict(),
                "parallel": "ProcessPoolExecutor(에피소드 병렬 — 같은 정책 스냅샷)",
                "RET_SCALE": RET_SCALE, "MINIBATCH": MINIBATCH,
                "GRAD_CLIP": GRAD_CLIP,
                "anchor_restored": "yr139 규약 복원 — 수익 정규화·미니배치·기울기 상한",
                "USE_END_BOOTSTRAP": USE_END_BOOTSTRAP,
                "bootstrap_note": "γ=1 자기참조 발산 제거 — 순수 MC",
                "SPACE_ONLY": SPACE_ONLY,
                "axis_note": "시간 판매 축 차단(1건당 +14.28분 구조적 손해)"},
        prereg=SPEC)
    (out / "train.json").write_text(json.dumps(
        {"history": hist, "in_progress": False, "code_dirty": bool(code_dirty()),
         "stamp": stamp}, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    return out


def train(ts: int, *, n_iter: int, eps_per_iter: int) -> Path:
    return train_one(ts, out_root=OUT, episode_fn=run_episode_diurnal,
                     n_iter=n_iter, eps_per_iter=eps_per_iter,
                     experiment="YR-170 판매 PPO — 5차 계약 24시간 이중 피크 무대",
                     prereg=SPEC)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-idx", type=int, default=0)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--eps-per-iter", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help="에피소드 1회만 돌려 배선 확인 (학습 없음)")
    ap.add_argument("--parallel", action="store_true",
                    help="iteration 안 에피소드를 프로세스 병렬로")
    ap.add_argument("--baseline", action="store_true",
                    help="전건 KEEP 기준선 Φ (참조값 — 판정 아님)")
    a = ap.parse_args()
    if a.baseline:
        r = baseline(TRAIN_SEEDS[a.seed_idx])
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"baseline_s{r['seed']}.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(r, ensure_ascii=False))
    elif a.smoke:
        import torch
        from ..integrated.transfer_head import (PpoSellPolicy, TransferActor,
                                                TransferCritic)
        from .yr151_transfer_ppo import load_kf
        torch.manual_seed(0)
        pol = PpoSellPolicy(TransferActor(), TransferCritic(), mode="live",
                            sample=True, seed=0, layout=terminal_layout())
        ep = run_episode_diurnal(TRAIN_SEEDS[0], pol, load_kf(),
                                 exec_config=ADOPTED_C0_GUARD)
        OUT.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"phi_final": round(ep["phi_final"], 4),
                          "n_phi_samples": len(ep["phi"]),
                          "n_joint": len(ep["joint"]),
                          "n_space": ep["n_space"], "n_time": ep["n_time"],
                          "admitted": ep["admitted"],
                          "n_decisions": len(pol.trail)}, ensure_ascii=False))
    else:
        fn = train_parallel if a.parallel else train
        p = fn(TRAIN_SEEDS[a.seed_idx], n_iter=a.n_iter,
               eps_per_iter=a.eps_per_iter)
        print(json.dumps({"out": str(p)}, ensure_ascii=False))
    print("DONE")
