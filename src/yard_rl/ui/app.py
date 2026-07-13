"""Replay UI (04 §4) — 읽기 전용.

실행: pip install -e .[ui]  →  streamlit run src/yard_rl/ui/app.py
레이아웃: 제목 → 정책 결정·KPI (가로) → 🌊 실시간 3D (Three.js, 전폭) /
▦ 평면 뷰 → 대기열 타임라인 → 이벤트 로그. 04 §4.3 색상 규칙 준수.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# streamlit run 은 스크립트로 실행 — 패키지 경로 부트스트랩
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from yard_rl.ui.live import policy_choices, profile_choices, run_and_record
from yard_rl.ui.replay import (decision_at, events_window, load_replay,
                               queue_series, scan_runs)
from yard_rl.ui.viewer3d import viewer_html

st.set_page_config(page_title="yard_rl replay", layout="wide")


# ----------------------------------------------------------------- run 선택
@st.cache_data(ttl=30)  # rerun 마다 전체 replay.json 재파싱 방지 (리뷰 확정 결함)
def _cached_runs():
    return scan_runs()


runs = _cached_runs()

# ---- 새 실행 (실시간 테스트): 환경·정책·부하 파라미터 선택 → 즉석 시뮬 → 재생
st.sidebar.title("Yard RL 시뮬레이션")
with st.sidebar.expander("🎬 새 실행 — 환경·정책 골라 바로 보기",
                         expanded=not runs):
    profs = profile_choices()
    p_i = st.selectbox("터미널 환경", range(len(profs)),
                       format_func=lambda i: profs[i][0], key="live_prof")
    pol = st.selectbox("정책", policy_choices(profs[p_i][1]), key="live_pol")
    lc1, lc2 = st.columns(2)
    seed_v = lc1.number_input("시나리오 seed", 1, 99999, 301,
                              help="같은 seed = 같은 하루 (정책 간 비교용)")
    trucks = lc2.number_input("외부트럭 수", 10, 300, 100, step=10,
                              help="8h shift 도착 대수 (기본 100)")
    vessels = st.slider("본선 작업 수", 0, 20, 8)
    peak = st.checkbox("피크 도착 패턴", value=False)
    st.caption("⚠ QL 정책은 기본 부하(트럭 100·본선 8)로 학습됨 — "
               "다른 부하는 일반화 시험입니다")
    if st.button("▶ 시뮬레이션 실행", type="primary", width='stretch'):
        with st.spinner("시뮬레이션 실행 중…"):
            rp = run_and_record(profs[p_i][1], pol, int(seed_v),
                                n_external=int(trucks), n_vessel=int(vessels),
                                peak=peak)
        _cached_runs.clear()
        load_replay.cache_clear()
        st.session_state["pending_run"] = rp.parent.name
        st.rerun()

if not runs:
    st.info("아직 replay 가 없습니다 — 사이드바 '🎬 새 실행'으로 첫 시뮬레이션을 "
            "실행하세요.")
    st.stop()

# 방금 실행한 run 자동 선택 (위젯 인스턴스화 전에 session_state 세팅)
run_ids = [r.run_id for r in runs]
_pending = st.session_state.pop("pending_run", None)
if _pending in run_ids:
    st.session_state["run_select"] = _pending
    st.session_state[f"idx_{_pending}"] = 0
if st.session_state.get("run_select") not in run_ids:
    st.session_state.pop("run_select", None)  # 삭제된 run 참조 방지

_by_id = {r.run_id: r for r in runs}
sel_id = st.sidebar.selectbox(
    "run", run_ids, key="run_select",
    format_func=lambda rid: f"{_by_id[rid].terminal_id} · {_by_id[rid].policy_id} "
                            f"· seed{_by_id[rid].seed}"
                            + (" · live" if "live" in str(_by_id[rid].path) else ""))
replay = load_replay(str(_by_id[sel_id].path))
man = replay["manifest"]
n_dec = man["n_decisions"]

if man.get("profile_assumed", True):
    st.sidebar.warning("⚠ assumed 프로파일 + 합성 시나리오 — 검증용 replay 이며 "
                       "실운영 재현이 아님 (YR-009 전).")

# ------------------------------------------------- 결정 스텝 조작 (사이드바)
ss = st.session_state
key_i = f"idx_{man['run_id']}"
ss.setdefault(key_i, 0)
# 자동 재생 전진은 슬라이더 '인스턴스화 이전'에 처리 — 위젯 생성 후 같은 run 에서
# widget-key session_state 를 수정하면 StreamlitAPIException (리뷰 확정 결함).
if ss.get("auto_play") and ss[key_i] < n_dec - 1:
    ss[key_i] += 1
st.sidebar.markdown("**결정 스텝 검증** (정책 패널·평면 뷰 연동)")
c1, c2, c3, c4 = st.sidebar.columns(4)
if c1.button("⏮", help="처음"):
    ss[key_i] = 0
if c2.button("◀", help="이전 결정"):
    ss[key_i] = max(0, ss[key_i] - 1)
if c3.button("▶", help="다음 결정"):
    ss[key_i] = min(n_dec - 1, ss[key_i] + 1)
if c4.button("⏭", help="끝"):
    ss[key_i] = n_dec - 1
idx = st.sidebar.slider("의사결정 #", 0, n_dec - 1, key=key_i)
play = st.sidebar.toggle("▶ 스텝 자동 진행", key="auto_play",
                         help="정책 패널·평면 뷰용 — 3D 재생은 뷰어 안 ▶ 버튼")
speed = st.sidebar.select_slider("진행 속도", [1, 2, 5, 10], value=5)

d = decision_at(replay, idx)
t = d["t"]
sla_s = man["sla_s"]
blk = man["block"]
B, R = blk["bay_count"], blk["row_count"]

# ----------------------------------------------------------- 제목 + 정책 결정
st.title(f"{man['terminal_id']} — {man['policy_id']} (seed {man['seed']})")
st.caption(f"info={man['info_level']} · scope={man['control_scope']} · "
           f"결정 {n_dec}건 · 미공개 작업은 개수만 표시 (누출 방지)")

pc1, pc2, pc3 = st.columns([1.25, 1.1, 1.35])
with pc1:
    st.markdown(f"##### 정책 결정 — #{idx + 1}/{n_dec} · t={t / 3600:.2f}h")
    st.markdown(f"- **rule**: `{d['rule']}` → **선택**: `{d['selected']}`\n"
                f"- **허용 (mask)**: {', '.join(d['mask'])}\n"
                f"- **state**: `{d['state_key']}` · 미공개 {d['hidden_job_count']}건")
with pc2:
    if d["q_values"]:
        qv = d["q_values"]
        bar = go.Figure(go.Bar(
            x=list(qv.values()), y=list(qv.keys()), orientation="h",
            marker_color=["#e06c00" if k == d["rule"] else "#90a4ae"
                          for k in qv]))
        bar.update_layout(height=120 + 16 * len(qv),
                          margin=dict(l=10, r=10, t=22, b=8),
                          title=dict(text="Q-values (시도된 action)",
                                     font=dict(size=13)))
        st.plotly_chart(bar, width='stretch')
    else:
        st.caption("Q-value 없음 — baseline rule 또는 미방문 상태(fallback)")
with pc3:
    k = d["kpis"]
    m1, m2, m3 = st.columns(3)
    m1.metric("대기 트럭", k["waiting_now"])
    m2.metric("완료", k["completed"])
    m3.metric("재조작", k["rehandles"])
    m4, m5, m6 = st.columns(3)
    m4.metric("queue-area (h)", f"{k['queue_area_h']:.1f}")
    m5.metric("이동 (km)", f"{k['travel_km']:.2f}")
    m6.metric("본선지연 (분)", f"{k['vessel_delay_min']:.1f}")
    with st.expander("종료 시점 최종 KPI"):
        st.json({key: round(v, 2) for key, v in man["final_metrics"].items()})

# ----------------------------------------------------------------- 야드 뷰
tab_live, tab2d = st.tabs(["🌊 실시간 3D (Three.js)", "▦ 평면 뷰"])
with tab_live:
    components.html(viewer_html(replay, height=680), height=690, scrolling=False)
    st.caption("뷰어 안 **▶ 재생** — 연속 시간 재생 (크레인 활주·트럭 진입/대기/"
               "퇴장). 드래그=회전 · 휠=확대. CDN(three.js) 로드 필요 — 오프라인이면 "
               "평면 뷰 사용. 장치 스택은 결정 시점 단위로 갱신")
with tab2d:
    heights = d["stack_heights"]  # [bay][row]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=[[heights[b][r] for b in range(B)] for r in range(R)],
        x=list(range(1, B + 1)), y=list(range(1, R + 1)),
        colorscale=[[0, "#f5f5f5"], [1, "#4a6fa5"]],
        zmin=0, zmax=blk["tier_max"], showscale=True,
        colorbar=dict(title="tier", thickness=10),
        hovertemplate="bay %{x} · row %{y} · %{z}단<extra></extra>"))
    b0, b1 = d["crane_bay_before"], d["crane_bay_after"]
    fig.add_shape(type="line", x0=b1, x1=b1, y0=0.5, y1=R + 0.5,
                  line=dict(color="#e06c00", width=3))
    if abs(b1 - b0) > 0.01:
        fig.add_annotation(x=b1, y=R + 0.7, ax=b0, ay=R + 0.7,
                           xref="x", axref="x", yref="y", ayref="y",
                           showarrow=True, arrowhead=2, arrowcolor="#e06c00")
    fig.add_trace(go.Scatter(x=[b1], y=[R + 0.7], mode="markers+text",
                             marker=dict(symbol="square", size=14, color="#e06c00"),
                             text=["YC"], textposition="middle right",
                             name="크레인", hovertext=f"bay {b0:g}→{b1:g}"))
    jobs = replay["jobs"]
    qx, qc, qs, qtext = [], [], [], []
    for q in d["queue"]:
        meta = jobs.get(q["job"], {})
        qx.append(meta.get("target_bay") or 0)
        over = q["wait_s"] >= sla_s
        qc.append("#c62828" if over else "#1565c0")
        qs.append("x" if over else "circle")
        qtext.append(f"{q['job']} ({q['flow']}) 대기 {q['wait_s'] / 60:.0f}분"
                     + (" ⚠SLA 초과" if over else ""))
    if qx:
        fig.add_trace(go.Scatter(x=qx, y=[0] * len(qx), mode="markers",
                                 marker=dict(size=12, color=qc, symbol=qs),
                                 name="대기 트럭", hovertext=qtext,
                                 hoverinfo="text"))
    vx, vy, vtext = [], [], []
    for jid in d["vessel_candidates"]:
        meta = jobs.get(jid, {})
        if meta.get("target_bay"):
            vx.append(meta["target_bay"])
            vy.append(meta.get("target_row") or 0)
            vtext.append(f"{jid} (본선, 마감 {meta['deadline'] / 3600:.1f}h)"
                         if meta.get("deadline") else jid)
    if vx:
        fig.add_trace(go.Scatter(x=vx, y=vy, mode="markers",
                                 marker=dict(symbol="diamond", size=13,
                                             color="#7b1fa2",
                                             line=dict(width=1, color="white")),
                                 name="본선 후보", hovertext=vtext, hoverinfo="text"))
    sel_job = d["selected"]
    sel_meta = jobs.get(sel_job.split(":")[-1], {}) if sel_job else {}
    if sel_meta.get("target_bay"):
        fig.add_trace(go.Scatter(x=[sel_meta["target_bay"]],
                                 y=[sel_meta.get("target_row") or 0],
                                 mode="markers",
                                 marker=dict(symbol="circle-open", size=22,
                                             color="#2e7d32", line=dict(width=3)),
                                 name="선택 작업", hovertext=[sel_job or ""],
                                 hoverinfo="text"))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title="bay", range=[-0.8, B + 1], dtick=2),
                      yaxis=dict(title="row (0=차선)", range=[-0.8, R + 1.3], dtick=1),
                      legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fig, width='stretch')

# ------------------------------------------------------------ 대기열 타임라인
ts, ns = queue_series(replay)
tl = go.Figure()
tl.add_trace(go.Scatter(x=[x / 3600 for x in ts], y=ns, mode="lines",
                        line=dict(color="#1565c0"), name="대기 트럭 수"))
tl.add_vline(x=t / 3600, line_color="#e06c00", line_dash="dash")
tl.update_layout(height=150, margin=dict(l=10, r=10, t=5, b=5),
                 xaxis_title="시간 (h)", yaxis_title="대기")
st.plotly_chart(tl, width='stretch')

# ----------------------------------------------------------------- 이벤트 로그
kinds = sorted({kind for _, kind, _ in replay["events"]})
with st.expander(f"이벤트 로그 (현재 ±10분, 총 {len(replay['events'])}건)"):
    pick = st.multiselect("종류 필터", kinds, default=kinds)
    rows = events_window(replay, t, kinds=set(pick))
    st.dataframe([{"t(s)": et, "t(h)": round(et / 3600, 2), "종류": kind,
                   "대상": payload} for et, kind, payload in rows],
                 width='stretch', height=240)

# ---------------------------------------------------------- 스텝 자동 진행
# 인덱스 전진은 위젯 생성 전(상단)에서 수행 — 여기서는 다음 rerun 만 예약
if play and ss[key_i] < n_dec - 1:
    time.sleep(1.0 / speed)
    st.rerun()
