"""야드 2.5D/3D 입체 뷰 (YR-015-c/e) — plotly mesh3d, streamlit 미의존.

recorder 의 decision snapshot 을 실측 비례 좌표계(m)로 렌더한다. 좌표계:
x=bay 방향(게이트→), y=row 방향(육측 차선 → 해측 안벽), z=단(tier).
04 §4.3 색상 규칙: 파랑=대기 트럭 · 빨강=SLA 초과 · 보라=본선 후보 ·
초록=선택 작업 · 주황=크레인.

두 모드:
- build_yard_figure: 결정 1개 스냅샷 (스텝/슬라이더 검증용 — 스택 정확)
- build_animation_figure: 전 결정 클라이언트 재생 (plotly frames — 서버
  rerun 없이 부드러움). ⚠ 장치(스택) 상태는 시작 시점 고정 — 표시 단순화.
"""
from __future__ import annotations

import plotly.graph_objects as go

# 컨테이너 색 팔레트 — 실제 선사 컨테이너 톤 (muted; 신호색과 충돌 회피).
# slot 좌표 해시로 결정론 배색.
_CONTAINER_PALETTE = ["#7a4a38", "#31597f", "#4e6e51", "#8a8d93", "#a06a30",
                      "#27605c", "#6e5a7e", "#95793f", "#5b7285", "#874f56"]
_GAP = 0.35          # 박스 간 시각 간격 (m)
_LANE_W = 4.0        # 인계 차선 폭 (m)
_GATE_SPACING = 8.0  # 게이트측(GATE_IN) 대기 트럭 간격 (m)
_CRANE = "#e8710a"
_TRUCK_CAB = "#3b4252"


def _shade(hex_color: str, frac: float) -> str:
    """흰색 쪽으로 frac(0~1)만큼 밝힘 — 위 tier 일수록 밝아 적층이 읽히게."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    mix = lambda c: int(c + (255 - c) * frac)  # noqa: E731 — 3회 반복 축약
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


class _Mesh:
    """단일 mesh3d trace 로 합치는 박스 누적기 (성능: 트레이스 수 최소화)."""

    def __init__(self):
        self.x, self.y, self.z = [], [], []
        self.i, self.j, self.k = [], [], []
        self.facecolor = []  # 박스별 개별색 (삼각형 12개 단위)

    def add_box(self, x0, x1, y0, y1, z0, z1, color: str | None = None):
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
        if color is not None:
            self.facecolor.extend([color] * 12)

    def trace(self, *, name: str, color: str | None = None, opacity: float = 1.0,
              showlegend: bool = False) -> go.Mesh3d:
        kw = dict(x=self.x, y=self.y, z=self.z, i=self.i, j=self.j, k=self.k,
                  name=name, opacity=opacity, flatshading=True, hoverinfo="skip",
                  showlegend=showlegend,
                  lighting=dict(ambient=0.72, diffuse=0.5, specular=0.06,
                                roughness=0.9, fresnel=0.05),
                  lightposition=dict(x=-1000, y=-3000, z=8000))
        if self.facecolor:
            kw.update(facecolor=self.facecolor)
        else:
            kw.update(color=color)
        return go.Mesh3d(**kw)


class _Geom:
    """블록 치수·파생 좌표 묶음 (build_* 공용)."""

    def __init__(self, block: dict):
        self.B = block["bay_count"]
        self.R = block["row_count"]
        self.T = block["tier_max"]
        # 치수 원본은 프로파일 → recorder manifest.block. 기본값은 *_m 필드가
        # 없는 구버전 replay 호환 폴백일 뿐 (하드코딩 금지 원칙)
        self.BL = float(block.get("bay_length_m", 6.5))
        self.RW = float(block.get("row_width_m", 3.1))
        self.TH = float(block.get("tier_height_m", 2.6))
        self.yard_l = self.B * self.BL
        self.yard_w = self.R * self.RW
        self.top_z = self.T * self.TH
        self.gate_x = -_GATE_SPACING * 2.6
        self.quay_y0 = self.yard_w + 3.5        # 해측 안벽 시작
        self.water_y0 = self.yard_w + 9.0       # 바다 시작

    def bay_cx(self, bay: float) -> float:
        return (bay - 0.5) * self.BL


# ------------------------------------------------------------- 정적 씨너리
def _scenery_traces(g: _Geom) -> list:
    """항만 환경: 지면·차선(마킹)·게이트 구역·안벽·바다 + 구역 라벨."""
    out = []
    x1 = g.yard_l + g.BL * 0.5
    ground = _Mesh()  # 야드 포장 (콘크리트 톤)
    ground.add_box(g.gate_x, x1, -_LANE_W - 3.5, g.quay_y0, -0.6, -0.02,
                   color="#ddd8cf")
    out.append(ground.trace(name="지면"))
    lane = _Mesh()    # 인계 차선 (아스팔트 톤)
    lane.add_box(g.gate_x, x1, -_LANE_W, -0.2, -0.01, 0.0, color="#9aa1ab")
    out.append(lane.trace(name="인계 차선"))
    marks = _Mesh()   # 차선 중앙 점선 마킹
    my = (-_LANE_W - 0.2) / 2
    mx = g.gate_x + 2.0
    while mx < x1 - 4.0:
        marks.add_box(mx, mx + 2.6, my - 0.12, my + 0.12, 0.0, 0.02,
                      color="#f2f2ee")
        mx += 8.0
    out.append(marks.trace(name="차선 마킹"))
    apron = _Mesh()   # 해측 안벽 에이프런
    apron.add_box(g.gate_x, x1, g.quay_y0, g.water_y0, -0.5, -0.05,
                  color="#c9c4ba")
    out.append(apron.trace(name="안벽"))
    water = _Mesh()   # 바다
    water.add_box(g.gate_x, x1, g.water_y0, g.water_y0 + 16.0, -0.9, -0.45,
                  color="#6fa8c9")
    out.append(water.trace(name="바다"))
    gate = _Mesh()    # 게이트 캐노피 (육측 진입부 표시)
    gx = g.gate_x + 2.0
    gate.add_box(gx, gx + 1.2, -_LANE_W - 0.6, 0.2, 0.0, 5.2, color="#8d99ae")
    gate.add_box(gx - 0.8, gx + 2.0, -_LANE_W - 1.2, 0.8, 5.2, 6.0,
                 color="#6c7a92")
    out.append(gate.trace(name="게이트"))
    out.append(go.Scatter3d(  # 구역 라벨
        x=[g.gate_x + 4.0, g.yard_l / 2], y=[-_LANE_W - 2.4, g.water_y0 + 7.0],
        z=[7.5, 1.5], mode="text", text=["게이트 (육측)", "안벽 · 해측"],
        textfont=dict(size=11, color="#5b6470"), hoverinfo="skip",
        showlegend=False))
    return out


def _container_traces(g: _Geom, heights: list[list[int]]) -> list:
    """컨테이너 스택 (slot 해시 배색) + 상단 hover 포인트."""
    boxes = _Mesh()
    hx, hy, hz, ht = [], [], [], []
    for b in range(g.B):
        for r in range(g.R):
            h = heights[b][r]
            x0, x1 = b * g.BL + _GAP, (b + 1) * g.BL - _GAP
            y0, y1 = r * g.RW + _GAP / 2, (r + 1) * g.RW - _GAP / 2
            for k in range(h):
                base = _CONTAINER_PALETTE[(b * 7919 + r * 104729 + k * 1299709)
                                          % len(_CONTAINER_PALETTE)]
                boxes.add_box(x0, x1, y0, y1, k * g.TH + 0.02,
                              (k + 1) * g.TH - 0.08,
                              color=_shade(base, 0.10 + 0.28 * k / max(1, g.T - 1)))
            if h:
                hx.append((x0 + x1) / 2)
                hy.append((y0 + y1) / 2)
                hz.append(h * g.TH + 0.9)
                ht.append(f"bay {b + 1} · row {r + 1} · {h}단")
    return [boxes.trace(name="컨테이너"),
            go.Scatter3d(x=hx, y=hy, z=hz, mode="markers",
                         marker=dict(size=6, color="rgba(0,0,0,0)"),
                         hovertext=ht, hoverinfo="text", name="스택",
                         showlegend=False)]


# ------------------------------------------------------------- 동적 요소
def _crane_mesh(g: _Geom, bay: float, stack_top_z: float) -> _Mesh:
    """ARMG 갠트리: 다리 2 + 빔 + 트롤리 + 스프레더 (스택 위까지 하강)."""
    bay_x = g.bay_cx(bay)
    leg_y0, leg_y1 = -_LANE_W - 1.0, g.yard_w + 1.0
    beam_z = g.top_z + 2.2
    w = 1.3  # 구조재 반폭 (가시성 우선 — 실측 비례 아님)
    m = _Mesh()
    for ly in (leg_y0, leg_y1):
        m.add_box(bay_x - w, bay_x + w, ly - w, ly + w, 0, beam_z, color=_CRANE)
    m.add_box(bay_x - w, bay_x + w, leg_y0, leg_y1, beam_z, beam_z + 1.4,
              color=_CRANE)
    ty = g.yard_w / 2
    m.add_box(bay_x - 1.6, bay_x + 1.6, ty - 1.4, ty + 1.4, beam_z - 1.6, beam_z,
              color=_CRANE)
    # 스프레더: 트롤리에서 스택 상단으로 내려온 와이어+빔 (작업감 연출)
    sp_z = min(beam_z - 1.6, stack_top_z + 1.2)
    m.add_box(bay_x - 0.25, bay_x + 0.25, ty - 0.25, ty + 0.25, sp_z, beam_z - 1.6,
              color="#b3560a")
    m.add_box(bay_x - 1.4, bay_x + 1.4, ty - 1.1, ty + 1.1, sp_z - 0.5, sp_z,
              color="#b3560a")
    return m


def _truck_boxes(mesh: _Mesh, cx: float, body_color: str):
    """트럭 = 운전석(진회색) + 적재함(신호색). 차선 안에서 게이트 방향을 봄."""
    y0, y1 = -_LANE_W + 0.5, -0.7
    mesh.add_box(cx - 5.2, cx - 3.4, y0 + 0.3, y1 - 0.3, 0.0, 2.5,
                 color=_TRUCK_CAB)                       # 운전석
    mesh.add_box(cx - 3.2, cx + 3.2, y0, y1, 0.35, 2.95, color=body_color)  # 적재함
    mesh.add_box(cx - 3.2, cx + 3.2, y0, y1, 0.0, 0.3, color="#2b2f36")     # 차대


def _dynamic_traces(g: _Geom, d: dict, jobs: dict, sla_s: float,
                    heights: list[list[int]]) -> list:
    """결정 1개의 동적 요소 — 애니메이션 프레임과 스텝 뷰가 공유.

    trace 수·순서 고정 (프레임 정합): [bay 밴드, 크레인, 이동 경로,
    대기 트럭, SLA 초과 트럭, 트럭 hover, 본선 후보, 선택 작업]
    """
    leg_y0, leg_y1 = -_LANE_W - 1.0, g.yard_w + 1.0
    bay = d["crane_bay_after"]
    # 작업 bay 지면 밴드 (반투명 슬라이스는 안 보임 — 리뷰 확정)
    band = _Mesh()
    bx = g.bay_cx(bay)
    band.add_box(bx - g.BL / 2, bx + g.BL / 2, leg_y0, leg_y1, -0.015, 0.06,
                 color="#f5a04a")
    col = heights[min(int(bay) - 1, g.B - 1) if bay >= 1 else 0]
    crane = _crane_mesh(g, bay, max(col) * g.TH if col else 0.0)
    crane_tr = crane.trace(name="크레인 (YC)", showlegend=True)
    crane_tr.hoverinfo = "name"
    # 이동 경로 (지면 점선)
    b0x, b1x = g.bay_cx(d["crane_bay_before"]), bx
    path_x, path_y, path_z = ([b0x, b1x], [leg_y0 - 1.2] * 2, [0.3, 0.3]) \
        if abs(b1x - b0x) > 0.5 else ([], [], [])
    path = go.Scatter3d(x=path_x, y=path_y, z=path_z, mode="lines+markers",
                        line=dict(color=_CRANE, width=6, dash="dot"),
                        marker=dict(size=4, color=_CRANE),
                        hoverinfo="skip", name="크레인 이동", showlegend=True)
    # 대기 트럭 (운전석+적재함) — 대상 bay 앞, 동일 지점은 차선 뒤로 스태거
    normal, exceed = _Mesh(), _Mesh()
    t_hx, t_hy, t_hz, t_ht = [], [], [], []
    gate_i = 0
    occupied: dict[float, int] = {}
    for q in d["queue"]:
        meta = jobs.get(q["job"], {})
        if meta.get("target_bay"):
            cx = g.bay_cx(meta["target_bay"])
        else:
            cx = -_GATE_SPACING * (gate_i % 2 + 1) - 7.5 * (gate_i // 2)
            gate_i += 1
        rank = occupied.get(cx, 0)
        occupied[cx] = rank + 1
        cx -= 9.5 * rank
        over = q["wait_s"] >= sla_s
        _truck_boxes(exceed if over else normal, cx,
                     "#c62828" if over else "#1565c0")
        t_hx.append(cx)
        t_hy.append(-_LANE_W / 2)
        t_hz.append(4.2)
        t_ht.append(f"{q['job']} ({q['flow']}) 대기 {q['wait_s'] / 60:.0f}분"
                    + (" ⚠SLA 초과" if over else ""))
    # 본선 후보 (보라 ◆)
    vx, vy, vz, vt = [], [], [], []
    for jid in d.get("vessel_candidates", []):
        meta = jobs.get(jid, {})
        if not meta.get("target_bay"):
            continue
        b, r = meta["target_bay"], meta.get("target_row") or 1
        vx.append(g.bay_cx(b))
        vy.append((r - 0.5) * g.RW)
        vz.append(heights[b - 1][r - 1] * g.TH + 3.2)
        dl = meta.get("deadline")
        vt.append(f"{jid} (본선{f', 마감 {dl / 3600:.1f}h' if dl else ''})")
    # 선택 작업 (초록)
    sel = d.get("selected") or ""
    smeta = jobs.get(sel.split(":")[-1], {})
    sx, sy, sz, stx = [], [], [], []
    if smeta.get("target_bay"):
        b, r = smeta["target_bay"], smeta.get("target_row") or 1
        sx, sy = [g.bay_cx(b)], [(r - 0.5) * g.RW]
        sz, stx = [heights[b - 1][r - 1] * g.TH + 5.2], [sel]
    return [
        band.trace(name="크레인 위치"),
        crane_tr,
        path,
        normal.trace(name="대기 트럭", showlegend=True),
        exceed.trace(name="SLA 초과 트럭", showlegend=True),
        go.Scatter3d(x=t_hx, y=t_hy, z=t_hz, mode="markers",
                     marker=dict(size=6, color="rgba(0,0,0,0)"),
                     hovertext=t_ht, hoverinfo="text", showlegend=False,
                     name="트럭 hover"),
        go.Scatter3d(x=vx, y=vy, z=vz, mode="markers",
                     marker=dict(symbol="diamond", size=7, color="#8e24aa"),
                     hovertext=vt, hoverinfo="text", name="본선 후보",
                     showlegend=True),
        go.Scatter3d(x=sx, y=sy, z=sz, mode="markers+text", text=["▼ 선택"] * len(sx),
                     textposition="top center",
                     textfont=dict(color="#1b8a3a", size=13),
                     marker=dict(symbol="circle-open", size=10, color="#1b8a3a",
                                 line=dict(width=4)),
                     hovertext=stx, hoverinfo="text", name="선택 작업",
                     showlegend=True),
    ]


# ------------------------------------------------------------- 레이아웃
def _scene_layout(g: _Geom, height: int) -> dict:
    bay_ticks = [1] + list(range(4, g.B + 1, 4))
    ext_x = (g.yard_l + g.BL * 0.5) - g.gate_x
    ext_y = g.water_y0 + 16.0 + _LANE_W + 3.5
    ext_z = g.top_z + 8.5
    # manual aspect 는 박스 크기가 곧 화면 배율 — ZOOM 으로 균일 확대 (왜곡 없음)
    ZOOM, Y_EX, Z_EX = 3.1, 1.35, 1.9
    return dict(
        height=height, margin=dict(l=0, r=0, t=0, b=0), showlegend=True,
        legend=dict(orientation="h", y=0.97, x=0.02,
                    bgcolor="rgba(255,255,255,0.6)"),
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=ZOOM, y=ext_y / ext_x * Y_EX * ZOOM,
                             z=ext_z / ext_x * Z_EX * ZOOM),
            xaxis=dict(title="",
                       # bay 라벨은 bay '중심' 좌표에 (경계에 찍으면 ±1 오독)
                       tickvals=[g.bay_cx(b) for b in bay_ticks],
                       ticktext=[f"bay {b}" for b in bay_ticks],
                       tickfont=dict(size=10), showgrid=False, zeroline=False,
                       showspikes=False, backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="", tickvals=[], showgrid=False, zeroline=False,
                       showspikes=False, backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(title="", tickvals=[], range=[-1.5, g.top_z + 7],
                       showgrid=False, zeroline=False, showspikes=False,
                       backgroundcolor="rgba(0,0,0,0)"),
            # 등각(orthographic) 2.5D 조감 — 원근 왜곡 없음
            camera=dict(projection=dict(type="orthographic"),
                        eye=dict(x=0.18, y=-1.45, z=1.75),
                        center=dict(x=0.0, y=0.0, z=-0.02)),
        ))


# ------------------------------------------------------------- 공개 API
def build_yard_figure(d: dict, jobs: dict, block: dict, sla_s: float,
                      *, height: int = 560) -> go.Figure:
    """결정 1개 스냅샷 뷰 (스텝/슬라이더 — 스택 정확)."""
    g = _Geom(block)
    heights = d["stack_heights"]
    fig = go.Figure()
    for tr in (_scenery_traces(g) + _container_traces(g, heights)
               + _dynamic_traces(g, d, jobs, sla_s, heights)):
        fig.add_trace(tr)
    fig.update_layout(**_scene_layout(g, height))
    return fig


def build_animation_figure(replay: dict, *, height: int = 560,
                           frame_ms: int = 350) -> go.Figure:
    """전 결정 클라이언트 재생 (plotly frames) — 서버 rerun 없이 부드러움.

    ⚠ 장치(스택)는 첫 결정 시점으로 고정 — 프레임 payload 를 동적 요소
    (크레인·트럭·마커)로 한정해 재생을 가볍게 유지. 스택 변화까지 정확히
    보려면 스텝 모드를 쓴다 (04 검증 목적은 스텝 모드가 원본).
    """
    man = replay["manifest"]
    g = _Geom(man["block"])
    jobs = replay["jobs"]
    sla_s = man["sla_s"]
    ds = replay["decisions"]
    heights0 = ds[0]["stack_heights"]
    static = _scenery_traces(g) + _container_traces(g, heights0)
    base = len(static)
    dyn0 = _dynamic_traces(g, ds[0], jobs, sla_s, heights0)
    fig = go.Figure(data=static + dyn0)
    dyn_idx = list(range(base, base + len(dyn0)))
    frames, steps = [], []
    for d in ds:
        frames.append(go.Frame(
            name=str(d["i"]), traces=dyn_idx,
            data=_dynamic_traces(g, d, jobs, sla_s, heights0)))
        steps.append(dict(method="animate", label=f"{d['t'] / 3600:.1f}h",
                          args=[[str(d["i"])],
                                dict(mode="immediate",
                                     frame=dict(duration=0, redraw=True),
                                     transition=dict(duration=0))]))
    fig.frames = frames
    layout = _scene_layout(g, height)
    layout.update(
        updatemenus=[dict(
            type="buttons", direction="right", x=0.02, y=0.06,
            bgcolor="rgba(255,255,255,0.7)",
            buttons=[
                dict(label="▶ 재생", method="animate",
                     args=[None, dict(fromcurrent=True,
                                      frame=dict(duration=frame_ms, redraw=True),
                                      transition=dict(duration=0))]),
                dict(label="⏸ 일시정지", method="animate",
                     args=[[None], dict(mode="immediate",
                                        frame=dict(duration=0, redraw=True),
                                        transition=dict(duration=0))]),
            ])],
        sliders=[dict(x=0.02, y=0.0, len=0.96, pad=dict(t=2),
                      currentvalue=dict(prefix="t = ", font=dict(size=12)),
                      steps=steps)])
    fig.update_layout(**layout)
    return fig
