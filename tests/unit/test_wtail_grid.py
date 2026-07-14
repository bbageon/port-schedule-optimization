"""YR-018 w_tail grid — 스모크 + 진단 wrapper 계약 테스트."""
import json

from yard_rl.envs.action_mask import N_ACTIONS
from yard_rl.experiments.wtail_grid import DiagPolicy, run_wtail_grid
from yard_rl.policies.q_learning import QLearningAgent, QLearningConfig

PROFILE = "configs/terminals/poc_single_crane.yaml"


def test_diag_policy_counts_fallback_and_thin():
    agent = QLearningAgent(QLearningConfig(), seed=0, policy_name="QL_T")
    diag = DiagPolicy(agent)
    mask = [True] + [False] * (N_ACTIONS - 1)
    s = (0,) * 9
    diag.act(s, mask)                      # 미방문 상태 → fallback
    assert (diag.decisions, diag.fallback, diag.thin) == (1, 1, 0)
    agent.table.row(s)
    agent.table.visit(s, 0)                # 방문 1회 → tried_valid, thin(n<5)
    diag.act(s, mask)
    assert (diag.decisions, diag.fallback, diag.thin) == (2, 1, 1)
    st = diag.stats()
    assert st["fallback_pct"] == 50.0 and st["chosen_n_median"] == 1.0


def test_wtail_grid_smoke_report_and_artifacts(tmp_path):
    out = tmp_path / "wtail"
    path = run_wtail_grid(profile_path=PROFILE, out_dir=str(out),
                          n_train=4, n_eval=2, epochs_list=(1,),
                          weights=(0.0, 1.0))
    text = path.read_text(encoding="utf-8")
    assert "QL_WT0" in text and "QL_WT1" in text
    assert "fallback%" in text and "인접 가중치 사다리" in text
    results = json.loads((out / "wtail_results.json").read_text(encoding="utf-8"))
    assert set(results) == {"e1"}
    assert set(results["e1"]) == {"FIFO", "QL_WT0", "QL_WT1"}
    assert all(len(rs) == 2 for rs in results["e1"].values())  # paired 2 seeds
    # 가중치·예산별 qtable 박제 (재현성 evidence)
    assert (out / "qtable_e1_QL_WT0.json").exists()
    assert (out / "qtable_e1_QL_WT1.json").exists()
    # w_tail 이 실제로 달라야 학습이 갈린다 — 두 테이블이 동일하면 주입 실패
    q0 = (out / "qtable_e1_QL_WT0.json").read_text(encoding="utf-8")
    q1 = (out / "qtable_e1_QL_WT1.json").read_text(encoding="utf-8")
    assert q0 != q1
