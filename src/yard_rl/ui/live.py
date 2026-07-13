"""즉석 실행 백엔드 (YR-015-d) — UI 에서 환경·정책·부하를 골라 바로 시뮬레이션.

streamlit 미의존. 읽기 전용 원칙 유지: UI 가 시뮬레이터를 직접 조작하지 않고
즉석 실행도 record → replay 경로를 그대로 탄다 (04 §2.1 — 검증 보장 불변).
즉석 실행분은 LIVE_OUT 하위에 기록 — 박제 replay(git 추적)와 분리.
"""
from __future__ import annotations

from pathlib import Path

from ..domain.enums import ControlScope, InformationLevel, PriorityRule
from ..envs.observations import BucketConfig
from ..envs.yard_env import YardEnv
from ..experiments.recorder import record_episode
from ..experiments.runner import make_scenarios
from ..io.profile_loader import load_profile
from ..io.scenario_gen import GenParams
from ..policies.baselines import FixedRulePolicy
from ..policies.q_learning import QLearningAgent, QLearningConfig, QTable

# 프로파일 ↔ 학습 산출물 등록부 — buckets/qtable 은 프로파일별로 fit/학습되므로
# 잘못 짝지으면 정책이 무의미해진다 (자동 추측 금지, 신규 프로파일은 여기 등록)
EXP_DIRS = {
    "hjnc_armg": "outputs/reports/exp1_hjnc",
    "dgt_armg": "outputs/reports/exp1_dgt",
    "poc_single_crane": "outputs/reports/exp_matrix",
}
LIVE_OUT = "outputs/replays/live"
_BASELINES = ["FIFO", "LONGEST_WAIT", "NEAREST_JOB", "MIN_REHANDLE"]


def profile_choices(config_dir: str = "configs/terminals") -> list[tuple[str, str]]:
    """(terminal_id, yaml 경로) 목록 — UI 셀렉터용."""
    out = []
    for p in sorted(Path(config_dir).glob("*.yaml")):
        try:
            out.append((load_profile(p).terminal_id, str(p)))
        except Exception:  # 손상 프로파일은 선택지에서 제외
            continue
    return out


def policy_choices(profile_path: str | Path) -> list[str]:
    """baseline 4종 + 해당 프로파일 학습 산출물의 qtable_* 정책."""
    pols = list(_BASELINES)
    exp = EXP_DIRS.get(Path(profile_path).stem)
    if exp and (Path(exp) / "buckets.json").exists():
        pols += sorted(q.stem.removeprefix("qtable_")
                       for q in Path(exp).glob("qtable_*.json"))
    return pols


def run_and_record(profile_path: str | Path, policy_name: str, seed: int, *,
                   n_external: int = 100, n_vessel: int = 8, peak: bool = False,
                   out_dir: str | Path = LIVE_OUT) -> Path:
    """환경·정책·부하 파라미터로 에피소드 1개 실행·기록. 반환: replay.json 경로.

    주의: QL 정책은 기본 부하(트럭 100·본선 8) train 시나리오로 학습된 것 —
    다른 부하는 '부하 일반화' 시험이지 학습 조건 재현이 아니다.
    """
    profile = load_profile(profile_path)
    exp = EXP_DIRS.get(Path(profile_path).stem)
    buckets_path = Path(exp) / "buckets.json" if exp else None
    if policy_name in _BASELINES:
        policy = FixedRulePolicy(PriorityRule[policy_name])
        # rule 은 state 를 안 쓰므로 bucket 은 기록용 — 산출물 없으면 기본값
        buckets = (BucketConfig.load(buckets_path)
                   if buckets_path and buckets_path.exists() else BucketConfig())
    else:
        if not (buckets_path and buckets_path.exists()):
            raise ValueError(f"{profile_path}: 학습 산출물 미등록 — {policy_name} 불가")
        buckets = BucketConfig.load(buckets_path)
        agent = QLearningAgent(QLearningConfig(), seed=0, policy_name=policy_name)
        agent.table = QTable.load(Path(exp) / f"qtable_{policy_name}.json",
                                  agent.cfg.n_actions)
        policy = agent
    params = GenParams(n_external=n_external, n_vessel=n_vessel, peak=peak)
    scenario = make_scenarios(profile, seed, 1, params)[0]
    env = YardEnv(profile, info_level=InformationLevel.BLOCK_ARRIVAL,
                  control_scope=ControlScope.SEQUENCE_ONLY,
                  bucket_cfg=buckets, check_invariants=True)
    tag = f"_t{n_external}v{n_vessel}" + ("p" if peak else "")
    run_id = f"{profile.terminal_id}_{policy_name}_seed{seed}{tag}"
    return record_episode(policy, env, scenario, run_id=run_id,
                          policy_name=policy_name, out_dir=out_dir)
