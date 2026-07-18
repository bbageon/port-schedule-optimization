"""QMIX 협조학습 (YR-013) — resolver 하 CTDE, 매핑 문서 §2~3.

- agent 망(CandidateQNet, 공유)은 기존과 동일 — 실행은 QPreference+중앙 resolver 그대로.
- 학습만 변경: 결정 단위 joint 전이를 MonotonicMixer 로 팀 총비용에 정렬 (Double joint target).
- 비용 최소화 규약: mixer 는 단조증가(|W|) → 각 크레인 argmin ≡ 팀 argmin (IGM).
- SMDP: 참여 크레인 1~2 가변 — presence mask 로 부재 슬롯의 값·가중 모두 0.
- replay 에는 resolver 가 실제 실행한 joint action 만 저장 (06 §6 — 사후 교체 기록 금지).

torch 는 optional [rl].
"""
from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass

import torch
from torch import nn

from .dqn_learner import VARIANTS
from .encoding import DecisionEncoding
from .qnet import CandidateQNet, QNetConfig, score_decision


@dataclass(frozen=True)
class QmixConfig:
    variant: str = "dueling"          # agent 망 (YR-045 RL 최선 계열)
    n_agents: int = 2
    gamma: float = 0.95
    ref_s: float = 60.0
    lr: float = 1e-3
    grad_clip: float = 1.0
    hidden: int = 128                 # agent 망 은닉
    mix_embed: int = 32               # mixer 은닉
    replay_capacity: int = 50_000
    batch_size: int = 64
    min_replay: int = 500
    target_sync_every: int = 500
    updates_per_decision: int = 1
    cost_scale: float = 1.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if self.n_agents < 1:
            raise ValueError("n_agents >= 1")
        if not 0.0 < self.gamma <= 1.0 or self.ref_s <= 0 or self.cost_scale <= 0:
            raise ValueError("gamma∈(0,1], ref_s>0, cost_scale>0")
        if min(self.lr, self.grad_clip, self.hidden, self.mix_embed,
               self.replay_capacity, self.batch_size, self.min_replay,
               self.target_sync_every, self.updates_per_decision) <= 0:
            raise ValueError("learner 수치는 전부 양수")


@dataclass(frozen=True)
class JointSample:
    """결정 1회의 joint 전이 — encs/action_pos 는 crane_id 정렬 순 (참여자만)."""

    encs: tuple[DecisionEncoding, ...]
    action_pos: tuple[int, ...]
    c_disc: float
    gamma_dt: float                    # 0.0 = terminal
    next_encs: tuple[DecisionEncoding, ...] | None


def stitch_joint_samples(times: list[float], costs: list[float],
                         events: list[tuple[int, tuple]], gamma: float,
                         ref_s: float) -> list[JointSample]:
    """결정 k → k+1 의 joint 전이 (모든 결정이 창 경계 — per-crane 스티칭보다 단순).

    pos=None 크레인(인코딩에 WAIT 행 없는 계약 예외)은 그 결정의 예측 대상에서 제외,
    전원 제외면 표본 생략. bootstrap 상태(next_encs)로는 그대로 사용 가능.
    """
    samples: list[JointSample] = []
    for i, (k, rec) in enumerate(events):
        acted = [(enc, pos) for (_cid, enc, pos) in rec if pos is not None]
        if not acted:
            continue
        if i + 1 < len(events):
            nk, nrec = events[i + 1]
            g_dt = gamma ** ((times[nk] - times[k]) / ref_s)
            nxt = tuple(enc for (_cid, enc, _pos) in nrec)
            samples.append(JointSample(tuple(e for e, _ in acted),
                                       tuple(p for _, p in acted),
                                       costs[k], g_dt, nxt))
        else:
            samples.append(JointSample(tuple(e for e, _ in acted),
                                       tuple(p for _, p in acted),
                                       costs[k], 0.0, None))
    return samples


class MonotonicMixer(nn.Module):
    """Q_tot = |w2(g)|ᵀ·ELU(Σ_i |W1(g)|_i q_i + b1(g)) + V(g) — ∂Q_tot/∂q_i ≥ 0.

    g = 전역 feature + presence 플래그. 부재 슬롯은 q=0 이고 |W1| 열도 presence 로
    0 처리해 값·기울기 모두 차단 (1-크레인 결정의 자연 퇴화).
    """

    def __init__(self, n_agents: int, g_dim: int, embed: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.embed = embed
        self.hyper_w1 = nn.Linear(g_dim, n_agents * embed)
        self.hyper_b1 = nn.Linear(g_dim, embed)
        self.hyper_w2 = nn.Linear(g_dim, embed)
        self.v = nn.Sequential(nn.Linear(g_dim, embed), nn.ELU(),
                               nn.Linear(embed, 1))

    def forward(self, q: torch.Tensor, presence: torch.Tensor,
                g: torch.Tensor) -> torch.Tensor:
        # q/presence: [B, n], g: [B, g_dim] → [B]
        w1 = self.hyper_w1(g).abs().view(-1, self.n_agents, self.embed)
        w1 = w1 * presence.unsqueeze(-1)
        h = nn.functional.elu((q.unsqueeze(-1) * w1).sum(dim=1) + self.hyper_b1(g))
        w2 = self.hyper_w2(g).abs()
        return (h * w2).sum(dim=-1) + self.v(g).squeeze(-1)


def _enc_tensors(encs: list[DecisionEncoding], device):
    f32 = torch.float32
    return (torch.tensor([list(e.g) for e in encs], dtype=f32, device=device),
            torch.tensor([list(e.yc) for e in encs], dtype=f32, device=device),
            torch.tensor([list(e.queue) for e in encs], dtype=f32, device=device),
            torch.tensor([[list(c) for c in e.cand] for e in encs],
                         dtype=f32, device=device),
            torch.tensor([list(e.actionable) for e in encs],
                         dtype=torch.bool, device=device))


class QmixLearner:
    """run_episode duck-type 호환 (scores_for/learn_step/cfg) — 실행 경로 재사용."""

    def __init__(self, cfg: QmixConfig, dims: tuple[int, int, int, int], seed: int = 0):
        self.cfg = cfg
        self.dims = dims
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        self.rng = random.Random(seed)
        self.device = torch.device(
            "cuda" if (cfg.device == "auto" and torch.cuda.is_available())
            else ("cpu" if cfg.device == "auto" else cfg.device))
        qcfg = QNetConfig(hidden=cfg.hidden, dueling=(cfg.variant == "dueling"))
        self.agent = CandidateQNet(dims, qcfg).to(self.device)
        self.target_agent = copy.deepcopy(self.agent)
        self.target_agent.eval()
        g_dim = dims[0] + cfg.n_agents          # 전역 feature + presence 플래그
        self.mixer = MonotonicMixer(cfg.n_agents, g_dim, cfg.mix_embed).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.target_mixer.eval()
        self.opt = torch.optim.Adam(
            list(self.agent.parameters()) + list(self.mixer.parameters()), lr=cfg.lr)
        self.replay: deque[JointSample] = deque(maxlen=cfg.replay_capacity)
        self.grad_steps = 0

    # ------------------------------------------------------------- 추론 (실행 경로)
    def scores_for(self, enc: DecisionEncoding) -> dict[int, float]:
        return score_decision(self.agent, enc, device=self.device)

    # ------------------------------------------------------------- 내부
    def _presence_and_g(self, encs_list: list[tuple[DecisionEncoding, ...]]):
        """샘플별 참여자 → presence [B,n]·mixer 입력 g [B, g_dim]. 슬롯은 참여 순."""
        n = self.cfg.n_agents
        pres = torch.zeros(len(encs_list), n, device=self.device)
        gfeat = torch.tensor([list(encs[0].g) for encs in encs_list],
                             dtype=torch.float32, device=self.device)
        for b, encs in enumerate(encs_list):
            pres[b, :min(n, len(encs))] = 1.0
        return pres, torch.cat([gfeat, pres], dim=1)

    def _q_chosen(self, net, encs_list, pos_list) -> torch.Tensor:
        """실행된 action 의 utility → [B, n] (부재 슬롯 0)."""
        n = self.cfg.n_agents
        flat, index = [], []
        for b, (encs, poss) in enumerate(zip(encs_list, pos_list)):
            for slot, (enc, pos) in enumerate(zip(encs, poss)):
                if slot >= n:
                    break
                flat.append((enc, pos))
                index.append((b, slot))
        q = torch.zeros(len(encs_list), n, device=self.device)
        if flat:
            g, yc, qs, cand, sel = _enc_tensors([e for e, _ in flat], self.device)
            vals = net(g, yc, qs, cand, sel).gather(
                1, torch.tensor([[p] for _, p in flat], device=self.device)).squeeze(1)
            for (b, slot), v in zip(index, vals):
                q[b, slot] = v
        return q

    def _q_double_next(self, encs_list) -> torch.Tensor:
        """Double: online argmin(mask) → target 평가 → [B, n] (부재 슬롯 0)."""
        n = self.cfg.n_agents
        flat, index = [], []
        for b, encs in enumerate(encs_list):
            for slot, enc in enumerate(encs):
                if slot >= n:
                    break
                flat.append(enc)
                index.append((b, slot))
        q = torch.zeros(len(encs_list), n, device=self.device)
        if flat:
            g, yc, qs, cand, sel = _enc_tensors(flat, self.device)
            with torch.no_grad():
                a_star = self.agent(g, yc, qs, cand, sel).masked_fill(
                    ~sel, float("inf")).min(dim=1).indices
                vals = self.target_agent(g, yc, qs, cand, sel).gather(
                    1, a_star.unsqueeze(1)).squeeze(1)
            for (b, slot), v in zip(index, vals):
                q[b, slot] = v
        return q

    # ------------------------------------------------------------- 학습
    def learn_step(self) -> float | None:
        if len(self.replay) < self.cfg.min_replay:
            return None
        batch = self.rng.sample(list(self.replay),
                                min(self.cfg.batch_size, len(self.replay)))
        pres, g_mix = self._presence_and_g([s.encs for s in batch])
        q = self._q_chosen(self.agent, [s.encs for s in batch],
                           [s.action_pos for s in batch])
        q_tot = self.mixer(q, pres, g_mix)
        costs = torch.tensor([s.c_disc for s in batch], dtype=torch.float32,
                             device=self.device)
        gammas = torch.tensor([s.gamma_dt for s in batch], dtype=torch.float32,
                              device=self.device)
        boot = torch.zeros(len(batch), dtype=torch.float32, device=self.device)
        nonterm = [i for i, s in enumerate(batch) if s.next_encs is not None]
        if nonterm:
            nlist = [batch[i].next_encs for i in nonterm]
            npres, ng_mix = self._presence_and_g(nlist)
            with torch.no_grad():
                nq = self._q_double_next(nlist)
                nxt = self.target_mixer(nq, npres, ng_mix)
            boot[torch.tensor(nonterm, device=self.device)] = nxt
        y = costs + gammas * boot
        loss = nn.functional.smooth_l1_loss(q_tot, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.agent.parameters())
                                 + list(self.mixer.parameters()), self.cfg.grad_clip)
        self.opt.step()
        self.grad_steps += 1
        if self.grad_steps % self.cfg.target_sync_every == 0:
            self.target_agent.load_state_dict(self.agent.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        return float(loss.detach())

    def absorb_joint(self, joint_sink: dict) -> int:
        """에피소드의 joint_sink → 표적 정규화(cost_scale) 후 replay 적재."""
        scaled = [c / self.cfg.cost_scale for c in joint_sink.get("costs", [])]
        samples = stitch_joint_samples(joint_sink.get("times", []), scaled,
                                       joint_sink.get("events", []),
                                       self.cfg.gamma, self.cfg.ref_s)
        for s in samples:
            self.replay.append(s)
        return len(samples)

    # ------------------------------------------------------------- 저장
    def save(self, path) -> None:
        cpu = lambda sd: {k: v.cpu() for k, v in sd.items()}  # noqa: E731
        torch.save({"format": "yard-rl-qmix-v1", "dims": self.dims,
                    "config": self.cfg.__dict__, "grad_steps": self.grad_steps,
                    "agent": cpu(self.agent.state_dict()),
                    "target_agent": cpu(self.target_agent.state_dict()),
                    "mixer": cpu(self.mixer.state_dict()),
                    "target_mixer": cpu(self.target_mixer.state_dict()),
                    "optimizer": self.opt.state_dict()}, str(path))

    @classmethod
    def load(cls, path) -> "QmixLearner":
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if payload.get("format") != "yard-rl-qmix-v1":
            raise ValueError("unsupported qmix format")
        learner = cls(QmixConfig(**payload["config"]), tuple(payload["dims"]))
        learner.agent.load_state_dict(payload["agent"])
        learner.target_agent.load_state_dict(payload["target_agent"])
        learner.mixer.load_state_dict(payload["mixer"])
        learner.target_mixer.load_state_dict(payload["target_mixer"])
        learner.opt.load_state_dict(payload["optimizer"])
        learner.grad_steps = int(payload["grad_steps"])
        return learner
