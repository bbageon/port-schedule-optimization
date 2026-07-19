"""YR-073 — 중앙 공동가치망: JR_NEW(계층목적) 순위 증류 (torch).

계약 (prereg 2026-07-19-YR-073 동결):
- 표본 = 결정 단위: 두 크레인 인코딩 + 공동 feasible 조합(위치쌍) + 교사 선택 +
  SF-SPT 선택 + 조합별 tier-A(트럭대기) 창 비용.
- 손실 = listwise CE (softmax(−score) 가 교사 선택 분류 — 순위 주목표, 조건 5)
  + 0.5·MSE(aux, tier-A 표준화 — advantage 크기 보조). scalar 단일 회귀 없음.
- 분기(교사≠SF) 결정 가중 ×4 (조건 6).
- 학생 정책은 교사와 동일한 조합 열거(_admissible_combos·_feasible_joint)·중앙
  resolver 안전장치 위에서 net argmin 만 대체 — rollout 0회.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
from torch import nn

from ..domain.enums import InformationLevel
from .adapter import capture
from .baselines import JointRolloutGreedy, _feasible_joint, _wait_of
from .encoding import StateNorm, encode_observation

LEVEL = InformationLevel.PRE_ADVICE
DISAGREE_W = 4.0
AUX_W = 0.5


@dataclass
class JointDecisionSample:
    """한 결정의 증류 표본 — 슬롯은 프로파일 크레인 id 오름차순 고정 (A, B).

    SMDP 계약상 결정은 1크레인일 수 있다 — 부재 슬롯은 빈 tuple 로 저장하고
    combo 위치는 -1. 입력 행에선 결측 규약(known=0 → 전부 0)으로 0 채움된다.
    """

    ga: tuple                                  # 공유 global
    yca: tuple                                 # 부재 시 ()
    qa: tuple
    canda: tuple                               # [K_a][2Fc], 부재 시 ()
    ycb: tuple
    qb: tuple
    candb: tuple
    combos: tuple                              # ((posA|-1, posB|-1), ...)
    tier_a: tuple                              # 조합별 tier-A 창 비용
    teacher_pos: int                           # combos 인덱스
    sf_pos: int | None
    disagree: bool
    round_tag: str = "r0"


def combo_matrix(s: JointDecisionSample) -> torch.Tensor:
    dim_yc = len(s.yca) or len(s.ycb)
    dim_q = len(s.qa) or len(s.qb)
    dim_c = len(s.canda[0]) if s.canda else len(s.candb[0])
    z_yc, z_q, z_c = [0.0] * dim_yc, [0.0] * dim_q, [0.0] * dim_c
    yca = list(s.yca) if s.yca else z_yc
    qa = list(s.qa) if s.qa else z_q
    ycb = list(s.ycb) if s.ycb else z_yc
    qb = list(s.qb) if s.qb else z_q
    rows = [list(s.ga) + yca + qa + (list(s.canda[i]) if i >= 0 else z_c)
            + ycb + qb + (list(s.candb[j]) if j >= 0 else z_c)
            for i, j in s.combos]
    return torch.tensor(rows, dtype=torch.float32)


class JointPairNet(nn.Module):
    """조합 입력 → (score, aux). 실행 비교는 score 순위 (작을수록 좋음)."""

    def __init__(self, in_dim: int, hidden: int = 256, h2: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, h2), nn.ReLU())
        self.score_head = nn.Linear(h2, 1)
        self.aux_head = nn.Linear(h2, 1)
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.score_head(h).squeeze(-1), self.aux_head(h).squeeze(-1)


@dataclass
class TrainResult:
    net: JointPairNet
    aux_mu: float
    aux_sd: float
    history: list = field(default_factory=list)
    best_tag: str = ""


def _decision_loss(net, s: JointDecisionSample, aux_mu, aux_sd) -> torch.Tensor:
    x = combo_matrix(s)
    score, aux = net(x)
    ce = nn.functional.cross_entropy(
        (-score).unsqueeze(0), torch.tensor([s.teacher_pos]))
    tgt = (torch.tensor(s.tier_a, dtype=torch.float32) - aux_mu) / aux_sd
    loss = ce + AUX_W * nn.functional.mse_loss(aux, tgt)
    return loss * (DISAGREE_W if s.disagree else 1.0)


def train_joint_net(samples: list, *, epochs: int = 30, lr: float = 1e-3,
                    seed: int = 73_000, batch: int = 32, val_fn=None,
                    val_every: int = 5, progress=print) -> TrainResult:
    """지도 순위 증류 — val_fn(net) 이 주어지면 val_every epoch 마다 평가, 최소값 선택."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    ta = [t for s in samples for t in s.tier_a]
    mu = float(sum(ta) / len(ta))
    sd = float((sum((t - mu) ** 2 for t in ta) / max(1, len(ta) - 1)) ** 0.5) or 1.0
    net = JointPairNet(combo_matrix(samples[0]).shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best_val, best_state, hist, best_tag = float("inf"), None, [], ""
    for ep in range(1, epochs + 1):
        order = list(range(len(samples)))
        rng.shuffle(order)
        tot = 0.0
        for k in range(0, len(order), batch):
            opt.zero_grad()
            ls = [_decision_loss(net, samples[i], mu, sd) for i in order[k:k + batch]]
            loss = torch.stack(ls).mean()
            loss.backward()
            opt.step()
            tot += float(loss) * len(ls)
        row = {"epoch": ep, "loss": round(tot / len(order), 4)}
        if val_fn is not None and (ep % val_every == 0 or ep == epochs):
            net.eval()
            with torch.no_grad():
                row["val"] = val_fn(net)
            net.train()
            if row["val"] < best_val:
                best_val, best_tag = row["val"], f"ep{ep}"
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        hist.append(row)
        progress(f"[train] {row}")
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return TrainResult(net, mu, sd, hist, best_tag)


def top1_agreement(net: JointPairNet, samples: list) -> dict:
    """순위 정확도 — 전체/분기 한정 top-1 일치율 (보고 의무, 조건 6)."""
    hit = dis_hit = dis_n = 0
    with torch.no_grad():
        for s in samples:
            score, _ = net(combo_matrix(s))
            ok = int(int(torch.argmin(score)) == s.teacher_pos)
            hit += ok
            if s.disagree:
                dis_n += 1
                dis_hit += ok
    return {"top1_all": round(hit / max(1, len(samples)), 4),
            "top1_disagree": round(dis_hit / max(1, dis_n), 4), "n_disagree": dis_n}


class CentralJointValuePolicy:
    """학생 — 교사와 동일한 조합 열거·공동 실행가능 판정 위에서 net argmin.

    rollout 0회 (조합당 forward 1회). 안전은 기존 계약 그대로: _feasible_joint
    (dry_run 오라클·token)·mask 후보만 열거되므로 net 은 순서만 정한다.
    """

    def __init__(self, net: JointPairNet, norm: StateNorm | None, generator,
                 slots: tuple[str, str], name: str = "STUDENT"):
        self.net = net
        self.norm = norm
        self.gen = generator
        self.slots = slots                     # 프로파일 크레인 id 오름차순 고정
        self.name = name
        self._jr = JointRolloutGreedy(None, generator=generator)   # 조합 열거 전용
        self._k = 0

    def decide(self, sim, dp, gen_by) -> dict:
        state, obs, _ = capture(sim, dp.crane_ids, LEVEL, "student", self._k,
                                generator=self.gen)
        self._k += 1
        encs = {ob.crane_id: encode_observation(state, ob, norm=self.norm)
                for ob in obs}
        ca, cb = self.slots
        ea, eb = encs.get(ca), encs.get(cb)
        ref = ea or eb
        z_yc, z_q = [0.0] * len(ref.yc), [0.0] * len(ref.queue)
        z_c = [0.0] * len(ref.cand[0])
        ctx_a = list(ref.g) + (list(ea.yc) + list(ea.queue) if ea else z_yc + z_q)
        ctx_b = (list(eb.yc) + list(eb.queue)) if eb else z_yc + z_q
        rows, assigns = [], []
        for combo in self._jr._admissible_combos(sim, dp, gen_by):
            assign = dict(zip(dp.crane_ids, combo))
            if not _feasible_joint(sim, assign):
                continue
            blk_a = (list(ea.cand[ea.candidate_ids.index(assign[ca].candidate_id)])
                     if ea and ca in assign else z_c)
            blk_b = (list(eb.cand[eb.candidate_ids.index(assign[cb].candidate_id)])
                     if eb and cb in assign else z_c)
            rows.append(ctx_a + blk_a + ctx_b + blk_b)
            assigns.append(assign)
        if not assigns:
            return {c: _wait_of(gen_by[c]) for c in dp.crane_ids}
        with torch.no_grad():
            score, _ = self.net(torch.tensor(rows, dtype=torch.float32))
        # 완전순서 tie-break (결정론): score → 후보 id 조합
        best = min(range(len(assigns)),
                   key=lambda i: (round(float(score[i]), 9),
                                  tuple((c, assigns[i][c].candidate_id)
                                        for c in sorted(dp.crane_ids))))
        return assigns[best]


def save_student(path, tr: TrainResult, norm_refs: dict) -> None:
    torch.save({"fmt": "yard-rl-joint-distill-v1", "state": tr.net.state_dict(),
                "in_dim": tr.net.in_dim, "aux_mu": tr.aux_mu, "aux_sd": tr.aux_sd,
                "norm_refs": norm_refs, "best_tag": tr.best_tag}, path)


def load_student(path) -> tuple[JointPairNet, StateNorm]:
    d = torch.load(path, weights_only=False)
    assert d["fmt"] == "yard-rl-joint-distill-v1"
    net = JointPairNet(d["in_dim"])
    net.load_state_dict(d["state"])
    net.eval()
    return net, StateNorm(refs=d["norm_refs"], basis="fitted_baseline_p90")
