"""YR-015-a recorder·replay 검증 — 04 §8.1/8.2 축소판.

핵심: UI 기록이 시뮬레이션 결과를 바꾸지 않고(regression), 정보 누출이 없다.
"""
import json

import pytest

from yard_rl.domain.enums import PriorityRule
from yard_rl.experiments.recorder import record_episode
from yard_rl.experiments.runner import make_scenarios, run_episode
from yard_rl.envs.yard_env import YardEnv
from yard_rl.io.profile_loader import load_profile
from yard_rl.io.scenario_gen import GenParams
from yard_rl.policies.baselines import FixedRulePolicy
from yard_rl.ui.replay import decision_at, load_replay, scan_runs

PROFILE = "configs/terminals/poc_single_crane.yaml"
PARAMS = GenParams(n_external=25, n_vessel=2, horizon_s=10800.0, drain_window_s=3600.0)


@pytest.fixture(scope="module")
def replay_path(tmp_path_factory):
    profile = load_profile(PROFILE)
    sc = make_scenarios(profile, 301, 1, PARAMS)[0]
    env = YardEnv(profile, check_invariants=True)
    return record_episode(FixedRulePolicy(PriorityRule.FIFO), env, sc,
                          run_id="TEST_FIFO_seed301",
                          out_dir=tmp_path_factory.mktemp("replays"))


def test_recording_does_not_change_results(replay_path):
    """기록된 final_metrics == 기록 없는 run_episode 결과 (04 §8.2 읽기 전용 보장)."""
    profile = load_profile(PROFILE)
    sc = make_scenarios(profile, 301, 1, PARAMS)[0]
    plain = run_episode(FixedRulePolicy(PriorityRule.FIFO),
                        YardEnv(profile, check_invariants=True), sc)
    rec = json.loads(replay_path.read_text(encoding="utf-8"))
    assert rec["manifest"]["final_metrics"] == pytest.approx(plain.metrics)
    assert rec["manifest"]["n_decisions"] == plain.metrics["n_decisions"]
    assert len(rec["decisions"]) == rec["manifest"]["n_decisions"]


def test_no_future_information_leakage(replay_path):
    """대기열 스냅샷에 미도착 작업이 없고, 미공개는 개수만 (04 §3.2)."""
    rec = json.loads(replay_path.read_text(encoding="utf-8"))
    jobs = rec["jobs"]
    for d in rec["decisions"]:
        assert d["hidden_job_count"] >= 0
        for q in d["queue"]:
            assert q["wait_s"] >= 0.0  # 도착 이전이면 음수 대기 — 누출
            assert jobs[q["job"]]["arrival"] <= d["t"] + 1e-6


def test_job_meta_captured_before_dispatch(replay_path):
    """target_bay 는 초기 상태에서 캡처 — dispatch 후 컨테이너 삭제로 소실되면
    선택 마커·트럭 위치가 전부 틀린다 (리뷰 확정 결함 회귀 가드)."""
    rec = json.loads(replay_path.read_text(encoding="utf-8"))
    targeted = [m for m in rec["jobs"].values() if m["flow"] != "GATE_IN"]
    assert targeted and all(m["target_bay"] is not None for m in targeted)


def test_yard3d_figure_builds(replay_path):
    """3D 뷰 figure 구성 — mesh 정합(vertex/intensity 길이)·필수 trace 존재."""
    pytest.importorskip("plotly")
    from yard_rl.ui.yard3d import build_yard_figure
    rec = json.loads(replay_path.read_text(encoding="utf-8"))
    man = rec["manifest"]
    assert "bay_length_m" in man["block"]  # 치수는 프로파일→manifest 가 원본
    for i in (0, len(rec["decisions"]) // 2, len(rec["decisions"]) - 1):
        fig = build_yard_figure(rec["decisions"][i], rec["jobs"], man["block"],
                                man["sla_s"])
        names = {t.name for t in fig.data}
        assert {"컨테이너", "크레인 (YC)", "크레인 위치"} <= names
        for tr in fig.data:
            if tr.type == "mesh3d" and tr.intensity is not None:
                assert len(tr.intensity) == len(tr.x)


def test_replay_repository_roundtrip(replay_path):
    runs = scan_runs(replay_path.parent.parent)
    assert [r.run_id for r in runs] == ["TEST_FIFO_seed301"]
    rep = load_replay(str(runs[0].path))
    assert decision_at(rep, 10 ** 9)["i"] == rep["manifest"]["n_decisions"] - 1
    assert decision_at(rep, -5)["i"] == 0


def test_animation_figure_builds(replay_path):
    """애니메이션 모드 — 프레임 수·동적 trace 정합 (YR-015-e)."""
    pytest.importorskip("plotly")
    from yard_rl.ui.yard3d import build_animation_figure
    rec = json.loads(replay_path.read_text(encoding="utf-8"))
    fig = build_animation_figure(rec)
    assert len(fig.frames) == rec["manifest"]["n_decisions"]
    n_dyn = len(fig.frames[0].traces)
    assert all(len(f.data) == n_dyn and f.traces == fig.frames[0].traces
               for f in fig.frames)  # 프레임별 trace 수·대상 인덱스 불변
    assert fig.layout.updatemenus and fig.layout.sliders


def test_live_run_and_record(tmp_path):
    """즉석 실행 백엔드 (YR-015-d) — 환경·정책·부하 파라미터로 replay 생성."""
    from yard_rl.ui.live import policy_choices, run_and_record
    pols = policy_choices(PROFILE)
    assert pols[:4] == ["FIFO", "LONGEST_WAIT", "NEAREST_JOB", "MIN_REHANDLE"]
    assert "QL_EXP1" in pols  # exp_matrix 산출물 등록 확인
    p = run_and_record(PROFILE, "FIFO", 777, n_external=20, n_vessel=2,
                       out_dir=tmp_path)
    rec = json.loads(p.read_text(encoding="utf-8"))
    assert rec["manifest"]["policy_id"] == "FIFO"
    assert rec["manifest"]["final_metrics"]["completed_external"] == 20.0
    assert "t20v2" in rec["manifest"]["run_id"]  # 부하 파라미터가 run_id 에 반영


def test_streamlit_app_renders():
    """UI 스모크 (streamlit 설치 + 실제 replay 존재 시에만 — dev 기본은 skip)."""
    st = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    if not scan_runs():
        pytest.skip("outputs/replays 에 replay 없음")
    at = AppTest.from_file("src/yard_rl/ui/app.py", default_timeout=60)
    at.run()
    assert not at.exception, at.exception[0].value
    at.slider[0].set_value(min(50, at.slider[0].max)).run()
    assert not at.exception
    # 자동재생: 위젯 생성 후 key 수정 금지 위반 회귀 가드 (끝 인덱스에서 토글 ON)
    at.slider[0].set_value(at.slider[0].max).run()
    at.toggle[0].set_value(True).run()
    assert not at.exception, at.exception[0].value
