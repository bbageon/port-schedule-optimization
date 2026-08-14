"""YR-171/173 — **BUY 견적망**: "이 작업을 블록 b 의 슬롯 t 에 넣으면 b 의 부담이
얼마나 느는가"를 한 번에 예측하는 신경망.

■ 무엇을 예측하는가 (경계 확정)
    BUY부담(작업 j, 블록 b, 슬롯 t) = J_b(t 에 수용) − J_b(수용 안 함)   [비용시간 단위]

**수신 블록 내부 부담만** 낸다. 추가주행(게이트→블록 차이)·기사 시간변경(외부 대기)·
불확실성 여유는 resolver 가 자기 저울에서 따로 더하므로(YR-171 총 추가비용 식) 여기에
넣으면 **이중 계상**이 된다. 그래서 입력에도 주행거리·소스 블록 상태를 넣지 않는다.

■ 왜 신경망인가 (YR-173)
가상 실행으로 견적하면 OFFER 1건당 수신 20 × 슬롯 48 = 960회, 하루 약 1,380만 회 →
에피소드당 약 173시간으로 실행 불가다. 견적은 **정답이 있는 문제**(굴려보면 실제 값이
나온다)라 지도학습이 성립하고, 960 후보 일괄 질의는 GPU 가 잘하는 형태다.
★단, 여기서 배우는 것은 **이 시뮬레이터의 근사**다 — 실제 항만 성능 예측이 아니다.

■ 정보 경계 (전부 공개 정보)
통지 진입시각(`notified_gate_in`)·공개 ETA(`public_block_eta`)·규격·SLA 여유·예상
처리시간·이송/이연 이력·본선 계획만 읽는다. 실현 미래값(`actual_gate_in` 이 미래인 것,
`actual_block_arrival`)은 **한 줄도 읽지 않는다**.

■ 입력을 어디서 받는가 (통합 지점 — 결합은 만들지 않았다)
블록 계획 (B,48,slot_dim) 은 `slot_plan.terminal_slot_plan(mbt, now)` 의 [48,F] 표를
블록 순서대로 쌓으면 그대로 맞는다(`slot_dim = slot_plan.N_FEATURES`). 후보 작업은
`job_feature_batch` 가 (N, JOB_DIM) 으로 만든다. 일부러 import 하지 않았다 — 이 망은
계획표의 열 구성에 의존하지 않고 차원만 받는다(계획표가 바뀌어도 망은 그대로).

■ 지위
배선·구조만 제공한다. 특징 스케일 상수는 assumed(사전등록 동결 대상)이고, 이 망의
견적 성능은 YR-171-B(BUY 견적 shadow)에서 실현 부담과 대조해 **따로 증명**해야 한다.
검증 없이 resolver 에 투입하지 않는다.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .block_congestion import SVC_REF_S
from .cost_curve_v2 import truck_target_s
from .pre_gate import MAX_TRANSFERS, public_block_eta
from .scenario_gen import GATE_BLOCK_MEAN_S
from .time_sell import MAX_ENTRY_DEFERRALS, notified_gate_in

# 슬롯 계약 (YR-171): 같은 작업일 00:00~24:00 을 30분 단위 48슬롯으로 본다.
SLOT_S = 1_800.0
SLOTS_PER_DAY = 48
DAY_S = SLOT_S * SLOTS_PER_DAY          # 86,400초 — 무대 계약과 동일

# 특징 정규화 상수 — 전부 **assumed**(사전등록 동결 대상, 결과 보고 튜닝 금지).
# 근거: 30분 = 판매 검토 창·슬롯 크기 / 4×SVC_REF_S = 재조작 3회까지의 처리시간 /
#       2시간 = SLA 여유의 실질 지원범위.
_SVC_REF_MAX_S = 4.0 * SVC_REF_S
_SLACK_REF_S = 2.0 * 3_600.0


# ------------------------------------------------------------------ 작업 특징
JOB_FEATURES = (
    "eta_in_30m",     # 공개 ETA 까지 남은 시간 / 30분 (임박도 — 창 안에서만 변별)
    "eta_in_day",     # 공개 ETA 까지 남은 시간 / 24시간 (하루 창 안의 거리)
    "eta_slot_pos",   # 공개 ETA 가 속한 30분 슬롯 index / 48 (계획 시간축과 정렬)
    "gate_in_30m",    # 통지 진입시각까지 남은 시간 / 30분 (변경 마감까지의 시계)
    "is_out",         # 반출(GATE_OUT)=1 · 반입(GATE_IN)=0
    "big_box",        # 40/45ft = 1 (20ft = 0) — 점유·작업량 규격
    "is_full",        # 적(FULL)=1 · 공(EMPTY)=0
    "vessel_linked",  # 본선 연계(공개 계획)=1 — 지연이 선석으로 번지는 작업
    "svc_est",        # 예상 처리시간 / (4×표준서비스) — 재조작 예상 포함
    "sla_slack",      # SLA 여유(초) / 2시간, [-1,1] clip — 음수 = 이미 빠듯함
    "transfer_used",  # 이송 이력 / 상한 (1 이면 더 못 옮김)
    "defer_used",     # 이연 이력 / 상한 (1 이면 더 못 미룸)
)
JOB_DIM = len(JOB_FEATURES)


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _enum_value(v) -> str:
    """Enum·문자열 어느 쪽이 와도 값 문자열만 뽑는다 (파이썬 판올림에 안 흔들리게)."""
    if v is None:
        return ""
    return str(getattr(v, "value", v))


def _box_spec(sim, job) -> tuple[str, str]:
    """(규격, 공/적) — 반입은 job 의 예고값, 반출은 **현재 야드에 있는 대상 컨테이너**.

    반출에는 `inbound_size`·`inbound_load` 가 없다. 그대로 두면 모든 반출이 "20ft 공"으로
    보여 규격 특징이 반출 여부와 뒤섞인다. 대상 컨테이너의 규격·공적은 지금 야드에 놓인
    사실(공개 관측)이므로 읽어도 정보경계를 넘지 않는다.
    """
    size = _enum_value(getattr(job, "inbound_size", None))
    load = _enum_value(getattr(job, "inbound_load", None))
    if size and load:
        return size, load
    tgt = getattr(job, "target_container", None)
    if tgt is not None:
        c = getattr(getattr(sim, "stacks", None), "containers", {}).get(tgt)
        if c is not None:
            return (size or _enum_value(getattr(c, "size", None)),
                    load or _enum_value(getattr(c, "load_status", None)))
    return size, load


def _expected_service_s(sim, job) -> float:
    """예상 처리시간 — 표준 서비스 × (1 + 대상 위에 쌓인 컨테이너 수).

    재조작 수는 **현재 야드 상태**(지금 관측 가능한 적재 배열)에서 세므로 공개 정보다.
    미래 재조작(그때 무엇이 위에 있을지)은 알 수 없으니 현 상태를 proxy 로 쓴다 —
    반입(GATE_IN)은 대상 컨테이너가 없어 항상 표준 서비스 1건이다.
    """
    tgt = getattr(job, "target_container", None)
    if tgt is None:
        return SVC_REF_S
    try:
        n_block = len(sim.stacks.blockers_above(tgt))
    except (AttributeError, KeyError, ValueError):
        # 이 블록 재고에 없는 대상(미등록·타 블록) = 셀 수 없음 → 재조작 0 으로 두고
        # 표준 서비스로 되돌린다. 모르는 값을 크게 잡아 부담을 부풀리지 않는다.
        n_block = 0
    return SVC_REF_S * (1.0 + n_block)


def job_features(mbt, src: str, job_id: str, now: float) -> list[float]:
    """판매 후보 작업 1건의 특징 (순서 = `JOB_FEATURES` 고정).

    공개 정보만 쓴다 — 통지 진입시각·공개 ETA·규격·SLA 여유·예상 처리시간·이송/이연
    이력·본선 연계 여부. 실현 미래값은 읽지 않는다.

    `src` 는 **작업을 지금 들고 있는 블록**이다(어디로 팔릴지는 여기서 모른다). 소스
    블록의 혼잡 상태는 일부러 넣지 않는다 — 이 망은 "수신 블록 부담"만 예측하고,
    소스 절감·주행·기사 시간변경은 resolver 가 자기 저울에서 계산한다(이중 계상 방지).
    """
    sim = mbt.blocks[src]
    j = sim.jobs[job_id]
    rec = mbt.ledger.records.get(job_id)

    # ---- 공개 시각 2종. 하나가 없으면 다른 하나에서 계약 평균 주행으로 유도한다
    #      (실현값으로 메우지 않는다 — 결측을 누출로 갚으면 안 된다).
    eta = public_block_eta(j)
    gi = notified_gate_in(j)
    if eta is None:
        eta = now if gi is None else gi + GATE_BLOCK_MEAN_S
    if gi is None:
        gi = eta - GATE_BLOCK_MEAN_S

    flow = _enum_value(getattr(j, "flow", ""))
    size, load = _box_spec(sim, j)
    vessel = (getattr(j, "vessel_id", None) is not None
              or int(getattr(j, "priority_class", 0) or 0) > 0)

    # SLA 여유 = 목표 진출시각 − (도착 + 표준서비스 + 출문주행). 남은 대기 예산이며
    # 음수면 이미 빠듯하다. 목표시각 D_T 는 v2 비용계약(cost_curve_v2)의 유도 앵커.
    # 정직 고지: 현 시나리오는 (공개 ETA − 통지 진입)이 계약 상수라 이 값이 사실상
    # 상수로 나온다 — ETA 오차·예약 준수오차가 들어와야 변별력이 생긴다(block_congestion
    # 의 D 특징과 같은 한계). 그래도 계약이 바뀌면 자동으로 살아나도록 식으로 둔다.
    slack_s = truck_target_s(sim, gi) - (eta + SVC_REF_S + GATE_BLOCK_MEAN_S)

    return [
        _clip((eta - now) / SLOT_S),
        _clip((eta - now) / DAY_S),
        (int(max(0.0, eta) // SLOT_S) % SLOTS_PER_DAY) / float(SLOTS_PER_DAY),
        _clip((gi - now) / SLOT_S),
        1.0 if flow == "GATE_OUT" else 0.0,
        1.0 if size in ("FT40", "FT45") else 0.0,
        1.0 if load == "FULL" else 0.0,
        1.0 if vessel else 0.0,
        _clip(_expected_service_s(sim, j) / _SVC_REF_MAX_S),
        _clip(slack_s / _SLACK_REF_S, -1.0, 1.0),
        _clip(float(getattr(rec, "transfer_count", 0)) / max(1, MAX_TRANSFERS)),
        _clip(float(getattr(rec, "entry_deferrals", 0)) / max(1, MAX_ENTRY_DEFERRALS)),
    ]


def job_feature_batch(mbt, offers, now: float) -> torch.Tensor:
    """(src, job_id) 목록 → (N, JOB_DIM) 텐서. 순서는 준 순서 그대로(감사 대응)."""
    rows = [job_features(mbt, src, jid, now) for src, jid in offers]
    if not rows:
        return torch.zeros((0, JOB_DIM), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


# ------------------------------------------------------------------ 견적망
class BuyEstimator(nn.Module):
    """블록 하루 계획 × 후보 작업 → (작업, 블록, 슬롯) 예상 부담.

    구조
      ① 블록 계획 (B,48,slot_dim) → Linear + 슬롯 위치 임베딩 → **시간축 self-attention**
         → 하루 계획 표현 (B,48,hid). 슬롯 하나의 부담은 그 슬롯만의 함수가 아니라
         앞뒤 슬롯의 밀림(볼록 비용의 이월)에 좌우되므로 시간축을 서로 보게 한다.
      ② 작업 (N,job_dim) → MLP → 질의 (N,hid).
      ③ **cross-attention**: 각 (작업, 블록) 쌍에서 질의가 그 블록의 48슬롯을 key/value 로
         조회 → 문맥 (N,B,hid).
      ④ 문맥이 슬롯 표현을 변조(FiLM)한 좁은 결합 채널을 비용 MLP 에 통과 → (N,B,48).

    ★메모리: ③을 (N·B,1,hid) 로 펴면 key/value 사영이 (N·B,48,hid) 를 만든다. key/value 는
    작업 축에 **의존하지 않으므로**, 질의를 블록 배치의 시퀀스로 두고 (B,N,hid)×(B,48,hid)
    로 한 번에 계산하면 같은 값을 얻으면서 그 텐서를 아예 만들지 않는다(수학적 동치).
    ④도 (N,B,48,hid) 대신 좁은 결합 채널 `joint_ch`(기본 hid/4)에서만 슬롯 축을 펼친다.

    ★출력 단위 = 비용시간(트럭 1대·1시간 = 1). 슬롯 유효성(임박 잠금·용량·반출의 공간
    불가)은 **상위 resolver 가 출력에 마스크**로 건다 — 망은 계획 전체를 본다.
    """

    def __init__(self, slot_dim: int, job_dim: int, hid: int = 64, heads: int = 4,
                 *, joint_ch: int | None = None, slots: int = SLOTS_PER_DAY):
        super().__init__()
        if hid % heads != 0:
            raise ValueError(f"hid({hid}) 는 heads({heads}) 의 배수여야 함")
        if slot_dim <= 0 or job_dim <= 0:
            raise ValueError("slot_dim·job_dim 은 1 이상")
        self.slot_dim = slot_dim
        self.job_dim = job_dim
        self.hid = hid
        self.slots = slots
        self.joint_ch = joint_ch if joint_ch is not None else max(4, hid // 4)

        # ① 하루 계획 인코더 (시간축 self-attention)
        self.slot_in = nn.Linear(slot_dim, hid)
        # 위치 임베딩 — 슬롯은 "몇 번째 30분"이라는 절대 위치가 의미를 갖는다(첨두 시간대).
        self.slot_pos = nn.Parameter(torch.randn(1, slots, hid) * 0.02)
        self.time_attn = nn.MultiheadAttention(hid, heads, batch_first=True)
        self.ln_time = nn.LayerNorm(hid)
        self.slot_ff = nn.Sequential(nn.Linear(hid, 2 * hid), nn.ReLU(),
                                     nn.Linear(2 * hid, hid))
        self.ln_slot = nn.LayerNorm(hid)

        # ② 작업 질의
        self.job_mlp = nn.Sequential(nn.Linear(job_dim, hid), nn.ReLU(),
                                     nn.Linear(hid, hid))
        self.ln_query = nn.LayerNorm(hid)

        # ③ 질의 → 계획 조회
        self.cross_attn = nn.MultiheadAttention(hid, heads, batch_first=True)
        self.ln_ctx = nn.LayerNorm(hid)

        # ④ 좁은 결합 채널 + 비용 head
        self.film = nn.Linear(hid, 2 * self.joint_ch)      # 문맥 → (배율, 오프셋)
        self.slot_joint = nn.Linear(hid, self.joint_ch)    # 슬롯 표현 → 결합 채널
        self.cost = nn.Sequential(nn.Linear(self.joint_ch, self.joint_ch), nn.ReLU(),
                                  nn.Linear(self.joint_ch, 1))
        # 작업과 무관한 슬롯 기저 부담 — 잔차로 분리해두면 "그 시간대는 원래 비싸다"를
        # 문맥 항이 다시 배우지 않아도 된다(학습 안정).
        self.slot_base = nn.Linear(hid, 1)

    # ---- ① 하루 계획 표현 (작업과 무관 — 여러 질의에서 재사용 가능)
    def plan_repr(self, plans: torch.Tensor) -> torch.Tensor:
        """plans: (B, T, slot_dim) → (B, T, hid). T ≤ slots."""
        if plans.dim() != 3:
            raise ValueError(f"plans 는 (B,T,slot_dim) 3차원이어야 함: {tuple(plans.shape)}")
        b, t, d = plans.shape
        if d != self.slot_dim:
            raise ValueError(f"slot_dim 불일치: {d} != {self.slot_dim}")
        if t > self.slots:
            raise ValueError(f"슬롯 수 {t} > 계약 {self.slots} (하루 30분 격자)")
        h = self.slot_in(plans) + self.slot_pos[:, :t]
        a, _ = self.time_attn(h, h, h, need_weights=False)
        h = self.ln_time(h + a)
        return self.ln_slot(h + self.slot_ff(h))

    def forward(self, plans: torch.Tensor, jobs: torch.Tensor) -> torch.Tensor:
        """plans: (B_blocks, 48, slot_dim) · jobs: (N_jobs, job_dim)

        반환: (N_jobs, B_blocks, 48) — 작업×블록×슬롯 예상 부담(비용시간, 0 이상).
        """
        if jobs.dim() != 2:
            raise ValueError(f"jobs 는 (N,job_dim) 2차원이어야 함: {tuple(jobs.shape)}")
        if jobs.shape[1] != self.job_dim:
            raise ValueError(f"job_dim 불일치: {jobs.shape[1]} != {self.job_dim}")
        p = self.plan_repr(plans)                       # (B,T,H)
        b, t, h = p.shape
        n = jobs.shape[0]

        q = self.ln_query(self.job_mlp(jobs))           # (N,H)
        # 질의를 **블록 배치의 시퀀스**로 둔다: 블록 b 의 배치 원소 안에서 N개 질의가
        # 각자 그 블록의 T슬롯을 본다 → (N·B,48,hid) key/value 사영을 만들지 않는다.
        qb = q.unsqueeze(0).expand(b, n, h)             # (B,N,H) — expand 는 복사 없음
        ctx, _ = self.cross_attn(qb, p, p, need_weights=False)
        ctx = self.ln_ctx(qb + ctx).transpose(0, 1)     # (N,B,H)

        gamma, beta = self.film(ctx).chunk(2, dim=-1)   # (N,B,C) 각각
        v = self.slot_joint(p)                          # (B,T,C)
        # 여기서 처음이자 마지막으로 슬롯 축을 펼친다: (N,B,T,C) — C = joint_ch(≪ hid)
        z = gamma.unsqueeze(2) * v.unsqueeze(0) + beta.unsqueeze(2)
        raw = self.cost(z).squeeze(-1) + self.slot_base(p).squeeze(-1).unsqueeze(0)
        # 부담은 **음수가 아니다** — 작업을 하나 더 받았는데 그 블록 내부비용이 줄어드는
        # 일은 없다(볼록 비용 위의 한계비용 ≥ 0). 이득은 소스 절감·주행 차이에서 나오며
        # 그건 resolver 몫이다. 그래서 출력에 softplus 를 씌운다.
        # 왜 ReLU 가 아니라 softplus 인가 — **0 부담도 가능해야** 하기 때문이다(아주 한산한
        # 슬롯은 받아도 비용이 사실상 안 는다). softplus 는 엄밀히는 0 을 못 내지만
        # pre-activation 이 충분히 음수면 float32 에서 ≈0(10^-13 이하)으로 붙고, ReLU 와
        # 달리 그 구간에서도 기울기가 살아 있어 "0 근처" 표본이 학습을 죽이지 않는다.
        return F.softplus(raw)

    # ---- 감사
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def spec(self) -> dict:
        return {"slot_dim": self.slot_dim, "job_dim": self.job_dim, "hid": self.hid,
                "heads": self.time_attn.num_heads, "joint_ch": self.joint_ch,
                "slots": self.slots, "n_params": self.n_params(),
                "output": "BUY부담(비용시간, ≥0) — 수신 블록 내부만(주행·기사 시간변경 제외)"}
