"""Candidate Double DQN 학습기 + 에피소드 드라이버 (YR-039 Stage B, 매핑 §3).

- SMDP per-crane 전이 스티칭: 크레인 c 의 연속 '실행' 결정 (k, k') 사이 구간비용을
  할인 합산 C = Σ γ^{(t_j−t_k)/ref}·cost_j, 부트스트랩 할인 γ^{(t_k'−t_k)/ref}.
  팀 공유비용을 각 크레인에 동일 귀속 (독립 학습자 v1 — mixer 는 YR-013).
  WAIT/yield 결정은 표본 제외 (환경의 일부로 취급, 매핑 §3).
- variant: dqn(target argmin=target) / ddqn(online argmin→target 평가, 기본) /
  dueling(ddqn + dueling head). Huber loss·grad clip·target sync.
- 탐험: 크레인별 ε 확률로 selectable 후보 1개의 score 를 −∞ 로 강제 (그 크레인
  내 최우선) — mandatory-우선·feasibility·dry_run 은 resolver 골격이 계속 전담.

torch 는 optional [rl].
"""
from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass, field

import torch
from torch import nn

from ..domain.enums import InformationLevel
from .adapter import _max_vessel_risk_state, capture
from .candidates import CandidateGenerator
from .cost_config import RewardCalculator
from .encoding import DecisionEncoding, encode_observation, encoding_dims
from .qnet import CandidateQNet, QNetConfig, QPreference, score_decision
from .resolver import CentralResolver

VARIANTS = ("dqn", "ddqn", "dueling")
_FORCE = -1e9    # 탐험 강제 선호값 (score 스케일 밖)


@dataclass(frozen=True)
class LearnerConfig:
    variant: str = "ddqn"
    gamma: float = 0.95
    ref_s: float = 60.0
    lr: float = 1e-3
    grad_clip: float = 1.0
    hidden: int = 128
    replay_capacity: int = 50_000
    batch_size: int = 64
    min_replay: int = 500
    target_sync_every: int = 500       # gradient step 단위
    updates_per_decision: int = 1
    cost_scale: float = 1.0            # 학습 표적 정규화 (train baseline fit — test 미접촉)
    device: str = "cpu"                # "cpu" | "cuda" | "auto" (가용 시 cuda — 매핑 §3)

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if not 0.0 < self.gamma <= 1.0 or self.ref_s <= 0:
            raise ValueError("gamma∈(0,1], ref_s>0")
        if min(self.lr, self.grad_clip, self.hidden, self.replay_capacity,
               self.batch_size, self.min_replay, self.target_sync_every,
               self.updates_per_decision) <= 0:
            raise ValueError("learner 수치는 전부 양수")
        if self.min_replay > self.replay_capacity:
            raise ValueError("min_replay <= replay_capacity")
        if self.cost_scale <= 0:
            raise ValueError("cost_scale must be positive")


@dataclass(frozen=True)
class Sample:
    enc: DecisionEncoding
    action_pos: int                    # candidate_ids 내 위치 (선택 후보)
    c_disc: float                      # 할인 합산 구간비용
    gamma_dt: float                    # 0.0 = terminal
    next_enc: DecisionEncoding | None


class CandidateDQNLearner:
    def __init__(self, cfg: LearnerConfig, dims: tuple[int, int, int, int],
                 seed: int = 0):
        self.cfg = cfg
        self.dims = dims
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        self.rng = random.Random(seed)
        resolved = ("cuda" if (cfg.device == "auto" and torch.cuda.is_available())
                    else ("cpu" if cfg.device == "auto" else cfg.device))
        self.device = torch.device(resolved)
        qcfg = QNetConfig(hidden=cfg.hidden, dueling=(cfg.variant == "dueling"))
        self.online = CandidateQNet(dims, qcfg).to(self.device)
        self.target = copy.deepcopy(self.online)
        self.target.eval()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.replay: deque[Sample] = deque(maxlen=cfg.replay_capacity)
        self.grad_steps = 0

    # ------------------------------------------------------------- 추론
    def scores_for(self, enc: DecisionEncoding) -> dict[int, float]:
        return score_decision(self.online, enc, device=self.device)

    # ------------------------------------------------------------- 배치
    def _tensors(self, encs: list[DecisionEncoding]):
        dev = self.device
        f32 = torch.float32
        return (torch.tensor([list(e.g) for e in encs], dtype=f32, device=dev),
                torch.tensor([list(e.yc) for e in encs], dtype=f32, device=dev),
                torch.tensor([list(e.queue) for e in encs], dtype=f32, device=dev),
                torch.tensor([[list(c) for c in e.cand] for e in encs],
                             dtype=f32, device=dev),
                torch.tensor([list(e.actionable) for e in encs],
                             dtype=torch.bool, device=dev))

    def _masked_min(self, q: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
        return q.masked_fill(~sel, float("inf")).min(dim=1)

    def learn_step(self) -> float | None:
        if len(self.replay) < self.cfg.min_replay:
            return None
        batch = self.rng.sample(list(self.replay), min(self.cfg.batch_size,
                                                       len(self.replay)))
        g, yc, qs, cand, sel = self._tensors([s.enc for s in batch])
        acts = torch.tensor([s.action_pos for s in batch], device=self.device)
        costs = torch.tensor([s.c_disc for s in batch], dtype=torch.float32,
                             device=self.device)
        gammas = torch.tensor([s.gamma_dt for s in batch], dtype=torch.float32,
                              device=self.device)
        nonterm = [i for i, s in enumerate(batch) if s.next_enc is not None]
        boot = torch.zeros(len(batch), dtype=torch.float32, device=self.device)
        if nonterm:
            ng, nyc, nqs, ncand, nsel = self._tensors(
                [batch[i].next_enc for i in nonterm])
            with torch.no_grad():
                q_t = self.target(ng, nyc, nqs, ncand, nsel)
                if self.cfg.variant == "dqn":
                    nxt = self._masked_min(q_t, nsel).values
                else:                                   # ddqn / dueling
                    q_o = self.online(ng, nyc, nqs, ncand, nsel)
                    a_star = self._masked_min(q_o, nsel).indices
                    nxt = q_t.gather(1, a_star.unsqueeze(1)).squeeze(1)
            boot[torch.tensor(nonterm, device=self.device)] = nxt
        y = costs + gammas * boot
        q_pred = self.online(g, yc, qs, cand, sel).gather(
            1, acts.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q_pred, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.opt.step()
        self.grad_steps += 1
        if self.grad_steps % self.cfg.target_sync_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())

    # ------------------------------------------------------------- 저장
    def save(self, path) -> None:
        """device-independent (매핑 §3·YR-033): 텐서를 CPU 로 이동해 저장."""
        cpu = lambda sd: {k: v.cpu() for k, v in sd.items()}  # noqa: E731
        torch.save({"format": "yard-rl-candidate-dqn-v1", "dims": self.dims,
                    "config": self.cfg.__dict__, "grad_steps": self.grad_steps,
                    "scaler": None,   # 프로토콜 슬롯 — 정규화는 cost_scale(config)
                    "online": cpu(self.online.state_dict()),
                    "target": cpu(self.target.state_dict()),
                    "optimizer": self.opt.state_dict()}, str(path))

    @classmethod
    def load(cls, path) -> "CandidateDQNLearner":
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if payload.get("format") != "yard-rl-candidate-dqn-v1":
            raise ValueError("unsupported candidate DQN format")
        learner = cls(LearnerConfig(**payload["config"]),
                      tuple(payload["dims"]))
        learner.online.load_state_dict(payload["online"])
        learner.target.load_state_dict(payload["target"])
        learner.opt.load_state_dict(payload["optimizer"])
        learner.grad_steps = int(payload["grad_steps"])
        return learner


# ================================================================= 드라이버
@dataclass
class EpisodeResult:
    total_cost: float
    n_decisions: int
    completion_rate: float
    backlog: int
    mean_wait_min: float
    p95_wait_min: float
    vessel_delay_min: float
    invariants_ok: bool
    samples: list[Sample] = field(default_factory=list)


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * q))]


def run_episode(sim, *, level: InformationLevel, preference,
                learner: CandidateDQNLearner | None = None,
                reward_calc: RewardCalculator | None = None,
                epsilon: float = 0.0, explore_rng: random.Random | None = None,
                generator: CandidateGenerator | None = None,
                collect: bool = False, learn: bool = False) -> EpisodeResult:
    """capture→score→resolve→cost 루프 (record_episode 동형, assemble 생략).

    preference 가 QPreference 면 결정마다 learner 점수(+ε 탐험 강제)를 주입.
    collect=True 면 per-crane SMDP 스티칭 표본을 EpisodeResult.samples 로 반환,
    learn=True 면 결정마다 learner.learn_step() 을 updates_per_decision 회 수행.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    gen = generator or CandidateGenerator()
    rc = reward_calc or RewardCalculator.assumed_default()
    resolver = CentralResolver(preference)
    rng = explore_rng or random.Random(0)
    sim.info_level = level
    times: list[float] = []
    costs: list[float] = []
    events: dict[str, list[tuple[int, DecisionEncoding, int | None]]] = {}
    dp = sim.run_until_decision()
    sim.cost.cut()                       # [0,t0) 선행 결정 없음 — 폐기 (record_episode 동일)
    k = 0
    ok = True
    while dp is not None:
        state, obs, gen_by = capture(sim, dp.crane_ids, level, "drive", k)
        encs = {ob.crane_id: encode_observation(state, ob) for ob in obs}
        if isinstance(preference, QPreference):
            scores: dict[tuple[str, int], float] = {}
            for cid, enc in encs.items():
                s = (learner.scores_for(enc) if learner is not None
                     else {c: 0.0 for i, c in enumerate(enc.candidate_ids)
                           if enc.actionable[i]})
                if epsilon > 0.0 and rng.random() < epsilon:
                    pool = [c for i, c in enumerate(enc.candidate_ids)
                            if enc.actionable[i]]
                    if pool:
                        s[rng.choice(pool)] = _FORCE
                scores.update({(cid, c): v for c, v in s.items()})
            preference.set_scores(scores)
        resn = resolver.resolve(sim, dp, gen_by)
        resolver.apply(sim, resn, gen_by)
        for r in resn.resolutions:
            enc = encs[r.crane_id]
            # WAIT 는 계약상 candidate_id=None → 인코딩의 WAIT 행으로 매핑해 표본에 포함 (YR-043).
            pos = (enc.wait_pos if r.chosen_candidate_id is None
                   else enc.candidate_ids.index(r.chosen_candidate_id))
            events.setdefault(r.crane_id, []).append((k, enc, pos))
        t_k = dp.time
        dp = sim.run_until_decision()
        raw = sim.cost.cut()
        cost = rc.cost_for(interval_start_s=t_k, interval_end_s=sim.now, raw=raw,
                           risk_max=_max_vessel_risk_state(state))
        times.append(t_k)
        costs.append(cost.total_normalized)
        if learn and learner is not None:
            for _ in range(learner.cfg.updates_per_decision):
                learner.learn_step()
        k += 1

    samples: list[Sample] = []
    if collect and learner is not None:
        scaled = [c / learner.cfg.cost_scale for c in costs]  # 표적 정규화
        samples = stitch_samples(times, scaled, events, learner.cfg.gamma,
                                 learner.cfg.ref_s)
        for s in samples:
            learner.replay.append(s)

    jobs = list(sim.jobs.values())
    done = sum(1 for j in jobs if j.status.name == "DONE")
    waits = [w / 60.0 for w in sim.kpis.wait_samples_s]
    return EpisodeResult(
        total_cost=sum(costs), n_decisions=k,
        completion_rate=done / max(1, len(jobs)),
        backlog=len(jobs) - done,
        mean_wait_min=(sum(waits) / len(waits)) if waits else 0.0,
        p95_wait_min=_percentile(waits, 0.95),
        vessel_delay_min=sim.kpis.vessel_delay_s / 60.0,
        invariants_ok=ok, samples=samples)


def stitch_samples(times: list[float], costs: list[float],
                   events: dict[str, list[tuple[int, DecisionEncoding, int | None]]],
                   gamma: float, ref_s: float) -> list[Sample]:
    """per-crane SMDP 전이 (매핑 §3). YR-043: WAIT 도 실제 행동이라 표본에 포함
    (pos=WAIT 행). pos=None 은 인코딩에 WAIT 행이 없는 예외뿐."""
    end_k = len(times)
    samples: list[Sample] = []
    for _cid, evs in sorted(events.items()):
        acted = [(k, enc, pos) for (k, enc, pos) in evs if pos is not None]
        for i, (k, enc, pos) in enumerate(acted):
            nxt = acted[i + 1] if i + 1 < len(acted) else None
            k_end = nxt[0] if nxt else end_k
            c = sum(gamma ** ((times[j] - times[k]) / ref_s) * costs[j]
                    for j in range(k, k_end))
            if nxt is None:
                samples.append(Sample(enc, pos, c, 0.0, None))
            else:
                g_dt = gamma ** ((times[nxt[0]] - times[k]) / ref_s)
                samples.append(Sample(enc, pos, c, g_dt, nxt[1]))
    return samples
