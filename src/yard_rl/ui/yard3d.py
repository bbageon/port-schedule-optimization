"""야드 2.5D/3D 입체 뷰 (YR-015-c) — plotly mesh3d, streamlit 미의존.

recorder 의 decision snapshot(스택 높이·크레인·대기열)을 실측 비례 좌표계
(bay_length × row_width × tier_height, m)로 렌더한다. 04 §4.3 색상 규칙 준수:
파랑=대기 트럭 · 빨강=SLA 초과 · 보라=본선 후보 · 초록=선택 작업 · 주황=크레인.
"""
from __future__ import annotations

import plotly.graph_objects as go

# 컨테이너 표면 색 (tier 낮음=어두움 → 높음=밝음: 적층이 눈에 읽히는 방향)
_CONTAINER_SCALE = [[0.0, "#5b7285"], [0.5, "#8ba3b8"], [1.0, "#cfdde8"]]
_GAP = 0.35          # 박스 간 시각 간격 (m)
_LANE_W = 4.0        # 인계 차선 폭 (m)
_GATE_SPACING = 8.0  # 게이트측(GATE_IN) 대기 트럭 간격 (m)


class _Mesh:
    """단일 mesh3d trace 로 합치는 박스 누적기 (성능: 트레이스 수 최소화)."""

    def __init__(self):
        self.x, self.y, self.z = [], [], []
        self.i, self.j, self.k = [], [], []
        self.intensity = []

    def add_box(self, x0, x1, y0, y1, z0, z1, val: float = 0.0):
        b = len(self.x)
        for (xx, yy, zz) in [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]:
            self.x.append(xx)
            self.y.append(yy)
            self.z.append(zz)
        for (a, c, d) in [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
                          (0, 5, 1), (0, 4, 5), (3, 2, 6), (3, 6, 7),
                          (1, 6, 2), (1, 5, 6), (0, 3, 7), (0, 7, 4)]:
            self.i.append(b + a)
            self.j.append(b + c)
            self.k.append(b + d)
        self.intensity.extend([val] * 8)

    def trace(self, *, name: str, color: str | None = None, colorscale=None,
              opacity: float = 1.0, showlegend: bool = False) -> go.Mesh3d:
        kw = dict(x=self.x, y=self.y, z=self.z, i=self.i, j=self.j, k=self.k,
                  name=name, opacity=opacity, flatshading=True, hoverinfo="skip",
                  showlegend=showlegend,
                  lighting=dict(ambient=0.72, diffuse=0.5, specular=0.06,
                                roughness=0.9, fresnel=0.05),
                  lightposition=dict(x=-1000, y=-3000, z=8000))
        if colorscale is not None:
            kw.update(intensity=self.intensity, colorscale=colorscale,
                      cmin=0, cmax=1, showscale=False)
        else:
            kw.update(color=color)
        return go.Mesh3d(**kw)


def _crane_traces(bay_x: float, yard_w: float, top_z: float, trolley_row_y: float,
                  bay_w: float):
    """ARMG 갠트리: 다리 2 + 상부 빔 + 트롤리 박스. bay_x = 크레인 중심 x (m)."""
    leg_y0, leg_y1 = -_LANE_W - 1.0, yard_w + 1.0
    beam_z = top_z + 2.2
    w = 1.3  # 구조재 반폭 (가시성 우선 — 실측 비례 아님)
    m = _Mesh()
    for ly in (leg_y0, leg_y1):  # 다리 (수직 기둥 2)
        m.add_box(bay_x - w, bay_x + w, ly - w, ly + w, 0, beam_z)
    m.add_box(bay_x - w, bay_x + w, leg_y0, leg_y1, beam_z, beam_z + 1.4)  # 빔
    m.add_box(bay_x - 1.6, bay_x + 1.6, trolley_row_y - 1.4, trolley_row_y + 1.4,
              beam_z - 1.8, beam_z)  # 트롤리
    tr = m.trace(name="크레인 (YC)", color="#e8710a", showlegend=True)
    tr.hoverinfo = "name"
    # 작업 bay 지면 밴드 — 반투명 슬라이스는 렌더에서 안 보임(리뷰 확정) →
    # 불투명 주황 밴드를 차선~야드 전폭 지면에 깔아 어느 각도에서도 보이게
    band = _Mesh()
    band.add_box(bay_x - bay_w / 2, bay_x + bay_w / 2, leg_y0, leg_y1, -0.015, 0.06)
    return [band.trace(name="크레인 위치", color="#f5a04a"), tr]


def build_yard_figure(d: dict, jobs: dict, block: dict, sla_s: float,
                      *, height: int = 560) -> go.Figure:
    """decision snapshot → 입체 야드 뷰."""
    B, R, T = block["bay_count"], block["row_count"], block["tier_max"]
    # 치수 원본은 프로파일 → recorder manifest.block. 기본값은 *_m 필드가 없는
    # 구버전 replay 호환용 폴백일 뿐이다 (하드코딩 금지 원칙 — 리뷰 확정 결함)
    BL = float(block.get("bay_length_m", 6.5))
    RW = float(block.get("row_width_m", 3.1))
    TH = float(block.get("tier_height_m", 2.6))
    yard_l, yard_w = B * BL, R * RW
    heights = d["stack_heights"]  # [bay][row] top tier

    fig = go.Figure()
    # ---- 지면 + 차선 (해측/육측 구분 배경)
    gate_x = -_GATE_SPACING * 2.6  # 게이트측 진입열 영역
    ground = _Mesh()
    ground.add_box(gate_x, yard_l + BL * 0.5, -_LANE_W - 3.5, yard_w + 3.5,
                   -0.6, -0.02)
    fig.add_trace(ground.trace(name="지면", color="#e7e3dc"))
    lane = _Mesh()
    lane.add_box(gate_x, yard_l + BL * 0.5, -_LANE_W, -0.2, -0.01, 0.0)
    fig.add_trace(lane.trace(name="인계 차선", color="#b9bfc7"))

    # ---- 컨테이너 스택 (tier 로 음영)
    boxes = _Mesh()
    hover_x, hover_y, hover_z, hover_t = [], [], [], []
    for b in range(B):
        for r in range(R):
            h = heights[b][r]
            x0, x1 = b * BL + _GAP, (b + 1) * BL - _GAP
            y0, y1 = r * RW + _GAP / 2, (r + 1) * RW - _GAP / 2
            for k in range(h):
                boxes.add_box(x0, x1, y0, y1, k * TH + 0.02, (k + 1) * TH - 0.08,
                              val=(k + 1) / T)
            if h:
                hover_x.append((x0 + x1) / 2)
                hover_y.append((y0 + y1) / 2)
                hover_z.append(h * TH + 0.9)
                hover_t.append(f"bay {b + 1} · row {r + 1} · {h}단")
    fig.add_trace(boxes.trace(name="컨테이너", colorscale=_CONTAINER_SCALE))
    fig.add_trace(go.Scatter3d(  # 스택 상단 hover 포인트 (비가시 — hover 전용)
        x=hover_x, y=hover_y, z=hover_z, mode="markers",
        marker=dict(size=6, color="rgba(0,0,0,0)"),
        hovertext=hover_t, hoverinfo="text", name="스택", showlegend=False))

    # ---- 대기 트럭 (차선): GATE_OUT=대상 bay 앞, GATE_IN=게이트측 진입열
    lane_y0, lane_y1 = -_LANE_W + 0.5, -0.7
    normal, exceed = _Mesh(), _Mesh()
    t_hx, t_hy, t_hz, t_ht = [], [], [], []
    gate_i = 0
    occupied: dict[float, int] = {}  # 같은 지점 대기 수 — 차선 뒤로 스태거
    for q in d["queue"]:
        meta = jobs.get(q["job"], {})
        if meta.get("target_bay"):
            cx = (meta["target_bay"] - 0.5) * BL
        else:
            # 게이트측 2열 × 열 내 전진 오프셋 (겹침 방지 — 리뷰 확정 결함)
            cx = -_GATE_SPACING * (gate_i % 2 + 1) - 7.5 * (gate_i // 2)
            gate_i += 1
        rank = occupied.get(cx, 0)
        occupied[cx] = rank + 1
        cx -= 7.0 * rank  # 동일 대상 bay 트럭은 차선을 따라 뒤로 정렬
        over = q["wait_s"] >= sla_s
        mesh = exceed if over else normal
        mesh.add_box(cx - 3.2, cx + 3.2, lane_y0, lane_y1, 0.0, 2.9)
        t_hx.append(cx)
        t_hy.append((lane_y0 + lane_y1) / 2)
        t_hz.append(4.0)
        t_ht.append(f"{q['job']} ({q['flow']}) 대기 {q['wait_s'] / 60:.0f}분"
                    + (" ⚠SLA 초과" if over else ""))
    if normal.x:
        fig.add_trace(normal.trace(name="대기 트럭", color="#1565c0", showlegend=True))
    if exceed.x:
        fig.add_trace(exceed.trace(name="SLA 초과 트럭", color="#c62828",
                                   showlegend=True))
    if t_hx:
        fig.add_trace(go.Scatter3d(  # hover 전용 (비가시 — 색 신호는 박스가 담당)
            x=t_hx, y=t_hy, z=t_hz, mode="markers",
            marker=dict(size=6, color="rgba(0,0,0,0)"),
            hovertext=t_ht, hoverinfo="text", showlegend=False))

    # ---- 본선 후보 (보라 ◆ + 스택 위 수직 지시선)
    vx, vy, vz, vt = [], [], [], []
    for jid in d.get("vessel_candidates", []):
        meta = jobs.get(jid, {})
        if not meta.get("target_bay"):
            continue
        b, r = meta["target_bay"], meta.get("target_row") or 1
        top = heights[b - 1][r - 1] * TH
        vx.append((b - 0.5) * BL)
        vy.append((r - 0.5) * RW)
        vz.append(top + 3.2)
        dl = meta.get("deadline")
        vt.append(f"{jid} (본선{f', 마감 {dl / 3600:.1f}h' if dl else ''})")
    if vx:
        fig.add_trace(go.Scatter3d(x=vx, y=vy, z=vz, mode="markers",
                                   marker=dict(symbol="diamond", size=7,
                                               color="#8e24aa"),
                                   hovertext=vt, hoverinfo="text",
                                   name="본선 후보", showlegend=True))

    # ---- 선택된 작업 (초록 링 마커)
    sel = d.get("selected") or ""
    meta = jobs.get(sel.split(":")[-1], {})
    if meta.get("target_bay"):
        b, r = meta["target_bay"], meta.get("target_row") or 1
        top = heights[b - 1][r - 1] * TH
        fig.add_trace(go.Scatter3d(
            x=[(b - 0.5) * BL], y=[(r - 0.5) * RW], z=[top + 5.2],
            mode="markers+text", text=["▼ 선택"], textposition="top center",
            textfont=dict(color="#1b8a3a", size=13),
            marker=dict(symbol="circle-open", size=10, color="#1b8a3a",
                        line=dict(width=4)),
            hovertext=[sel], hoverinfo="text", name="선택 작업", showlegend=True))

    # ---- 크레인 + 이동 경로 (지면 점선)
    top_z = T * TH
    bx0 = (d["crane_bay_before"] - 0.5) * BL
    bx1 = (d["crane_bay_after"] - 0.5) * BL
    for tr in _crane_traces(bx1, yard_w, top_z, yard_w / 2, BL):
        fig.add_trace(tr)
    if abs(bx1 - bx0) > 0.5:
        fig.add_trace(go.Scatter3d(
            x=[bx0, bx1], y=[-_LANE_W - 2.2] * 2, z=[0.3, 0.3],
            mode="lines+markers", line=dict(color="#e8710a", width=6, dash="dot"),
            marker=dict(size=[3, 7], color="#e8710a"),
            hovertext=[f"이동 시작 bay {d['crane_bay_before']:g}",
                       f"현재 bay {d['crane_bay_after']:g}"],
            hoverinfo="text", name="크레인 이동", showlegend=True))

    # ---- 장면
    bay_ticks = [1] + list(range(4, B + 1, 4))
    # 세로 여백 제거: 실측 비례(aspect data)는 긴 블록이 가는 띠로 축소됨 —
    # y·z 를 시각 과장 (라벨로 명시). 축 스케일만 바뀌고 상대 위치는 보존.
    ext_x = (yard_l + BL * 0.5) - (-_GATE_SPACING * 2.6)
    ext_y = yard_w + _LANE_W + 7.0
    ext_z = top_z + 8.5
    # manual aspect 는 박스 크기가 곧 화면 배율 — ZOOM 으로 균일 확대 (왜곡 없음)
    ZOOM, Y_EX, Z_EX = 2.7, 1.9, 1.9
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(orientation="h", y=0.97, x=0.02, bgcolor="rgba(255,255,255,0.6)"),
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=ZOOM, y=ext_y / ext_x * Y_EX * ZOOM,
                             z=ext_z / ext_x * Z_EX * ZOOM),
            xaxis=dict(title="",
                       # bay 라벨은 해당 bay 의 '중심' 좌표에 (경계에 찍으면 ±1 오독)
                       tickvals=[(b - 0.5) * BL for b in bay_ticks],
                       ticktext=[f"bay {b}" for b in bay_ticks],
                       tickfont=dict(size=10), showgrid=False, zeroline=False,
                       showspikes=False, backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="", tickvals=[], showgrid=False, zeroline=False,
                       showspikes=False, backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(title="", tickvals=[], range=[-1.5, top_z + 7],
                       showgrid=False, zeroline=False, showspikes=False,
                       backgroundcolor="rgba(0,0,0,0)"),
            # 등각(orthographic) 2.5D — 원근 왜곡 없이 블록 전체가 프레임에 들어옴
            camera=dict(projection=dict(type="orthographic"),
                        eye=dict(x=0.18, y=-1.75, z=0.95),
                        center=dict(x=0.0, y=0.0, z=-0.02)),
        ))
    return fig
