"""단일 시나리오 Replay UI (04 §4, UI-2) — 읽기 전용.

실행: pip install -e .[ui]  →  streamlit run src/yard_rl/ui/app.py
화면: 야드 2D 평면도(스택 높이·크레인·대기 트럭) + 정책 결정 패널 + KPI
+ 대기열 타임라인 + 이벤트 로그. 04 §4.3 색상 규칙(색+기호 병행) 준수.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# streamlit run 은 스크립트로 실행 — 패키지 경로 부트스트랩
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go
import streamlit as st

from yard_rl.ui.replay import (decision_at, events_window, load_replay,
                               queue_series, scan_runs)

st.set_page_config(page_title="yard_rl replay", layout="wide")

# ----------------------------------------------------------------- run 선택
runs = scan_runs()
if not runs:
    st.error("outputs/replays/ 에 replay 가 없습니다 — "
             "`python -m yard_rl.cli record-replay ...` 로 먼저 생성하세요.")
    st.stop()

st.sidebar.title("Replay 선택")
labels = [f"{r.terminal_id} · {r.policy_id} · seed{r.seed}" for r in runs]
sel = st.sidebar.selectbox("run", range(len(runs)), format_func=lambda i: labels[i])
replay = load_replay(str(runs[sel].path))
man = replay["manifest"]
n_dec = man["n_decisions"]

if man.get("profile_assumed", True):
    st.sidebar.warning("⚠ assumed 프로파일 + 합성 시나리오 — 검증용 replay 이며 "
                       "실운영 재현이 아님 (YR-009 전).")

# ----------------------------------------------------------------- 시간 조작
ss = st.session_state
key_i = f"idx_{man['run_id']}"
ss.setdefault(key_i, 0)
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
play = st.sidebar.toggle("▶ 자동 재생")
speed = st.sidebar.select_slider("재생 속도", [1, 2, 5, 10], value=5,
                                 help="초당 의사결정 수")

d = decision_at(replay, idx)
t = d["t"]
sla_s = man["sla_s"]
blk = man["block"]
B, R = blk["bay_count"], blk["row_count"]

st.title(f"{man['terminal_id']} — {man['policy_id']} (seed {man['seed']})")
st.caption(f"t = {t / 3600:.2f} h ({t:,.0f} s) · 결정 {idx + 1}/{n_dec} · "
           f"info={man['info_level']} · scope={man['control_scope']} · "
           f"미공개 작업 {d['hidden_job_count']}건 (개수만 표시)")

left, right = st.columns([2.1, 1])

# ----------------------------------------------------------------- 야드 평면도
with left:
    fig = go.Figure()
    heights = d["stack_heights"]  # [bay][row]
    fig.add_trace(go.Heatmap(
        z=[[heights[b][r] for b in range(B)] for r in range(R)],
        x=list(range(1, B + 1)), y=list(range(1, R + 1)),
        colorscale=[[0, "#f5f5f5"], [1, "#4a6fa5"]],
        zmin=0, zmax=blk["tier_max"], showscale=True,
        colorbar=dict(title="tier", thickness=10),
        hovertemplate="bay %{x} · row %{y} · %{z}단<extra></extra>"))
    # 크레인: 이번 결정의 이동 (before → after)
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
    # 대기 외부트럭 (차선 y=0): GATE_OUT 은 대상 bay, GATE_IN 은 게이트측(0) 표기
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
    # 본선 연계 후보 (보라 ◆, 대상 컨테이너 위치)
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
    # 선택된 작업 표시 (초록 테두리)
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

    # 대기열 타임라인 (현재 위치 표시)
    ts, ns = queue_series(replay)
    tl = go.Figure()
    tl.add_trace(go.Scatter(x=[x / 3600 for x in ts], y=ns, mode="lines",
                            line=dict(color="#1565c0"), name="대기 트럭 수"))
    tl.add_vline(x=t / 3600, line_color="#e06c00", line_dash="dash")
    tl.update_layout(height=160, margin=dict(l=10, r=10, t=5, b=5),
                     xaxis_title="시간 (h)", yaxis_title="대기")
    st.plotly_chart(tl, width='stretch')

# --------------------------------------------------------------- 정책 결정 패널
with right:
    st.subheader("정책 결정")
    st.markdown(f"- **rule**: `{d['rule']}`\n- **선택 작업**: `{d['selected']}`\n"
                f"- **허용 rule (mask)**: {', '.join(d['mask'])}\n"
                f"- **state key**: `{d['state_key']}`")
    if d["q_values"]:
        qv = d["q_values"]
        bar = go.Figure(go.Bar(
            x=list(qv.values()), y=list(qv.keys()), orientation="h",
            marker_color=["#e06c00" if k == d["rule"] else "#90a4ae"
                          for k in qv]))
        bar.update_layout(height=180 + 18 * len(qv), title="Q-values (시도된 action)",
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(bar, width='stretch')
        st.caption("Q-value 는 인과 설명이 아니라 정책이 사용한 입력·점수의 사실 표시 (04 §4.4)")
    else:
        st.caption("Q-value 없음 — baseline rule 또는 미방문 상태(fallback)")

    k = d["kpis"]
    st.subheader("누적 KPI (현재 시점)")
    m1, m2, m3 = st.columns(3)
    m1.metric("대기 트럭", k["waiting_now"])
    m2.metric("queue-area (h)", f"{k['queue_area_h']:.1f}")
    m3.metric("완료", k["completed"])
    m4, m5, m6 = st.columns(3)
    m4.metric("이동 (km)", f"{k['travel_km']:.2f}")
    m5.metric("재조작", k["rehandles"])
    m6.metric("본선지연 (분)", f"{k['vessel_delay_min']:.1f}")
    with st.expander("종료 시점 최종 KPI"):
        fm = man["final_metrics"]
        st.json({key: round(v, 2) for key, v in fm.items()})

# ----------------------------------------------------------------- 이벤트 로그
kinds = sorted({kind for _, kind, _ in replay["events"]})
with st.expander(f"이벤트 로그 (현재 ±10분, 총 {len(replay['events'])}건)"):
    pick = st.multiselect("종류 필터", kinds, default=kinds)
    rows = events_window(replay, t, kinds=set(pick))
    st.dataframe([{"t(s)": et, "t(h)": round(et / 3600, 2), "종류": kind,
                   "대상": payload} for et, kind, payload in rows],
                 width='stretch', height=240)

# ----------------------------------------------------------------- 자동 재생
if play and ss[key_i] < n_dec - 1:
    time.sleep(1.0 / speed)
    ss[key_i] += 1
    st.rerun()
