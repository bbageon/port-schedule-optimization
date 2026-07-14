"""Three.js 실시간 3D 뷰어 (YR-015-f) — 연속 시간 기반 재생.

replay.json 의 이벤트 타임스탬프(BLOCK_ARRIVAL/DISPATCH/JOB_COMPLETED)로
시뮬레이션 시계를 재생한다: 크레인은 bay 사이를 활주, 트럭은 게이트로 진입 →
대기(SLA 초과 시 적색) → 서비스 → 퇴장. plotly frames 의 mesh 갱신 한계
(크레인 정지)를 WebGL 로 대체. 읽기 전용 원칙 불변 — 데이터는 recorder 산출물.

streamlit 미의존 (HTML 문자열 생성만) — app.py 가 components.html 로 임베드.
Three.js 는 CDN(ES module) 로드: 오프라인 환경이면 평면 뷰로 폴백.
"""
from __future__ import annotations

import json

_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0"


def viewer_html(replay: dict, *, height: int = 700) -> str:
    """replay → self-contained HTML (iframe 임베드용)."""
    man = replay["manifest"]
    data = {
        "block": man["block"],
        "sla_s": man["sla_s"],
        "end_s": man["end_time_s"],
        "policy": man["policy_id"],
        "terminal": man["terminal_id"],
        "jobs": replay["jobs"],
        "decisions": [{k: d[k] for k in
                       ("i", "t", "crane_bay_before", "crane_bay_after",
                        "task_end_t", "rule", "selected", "stack_heights")}
                      for d in replay["decisions"]],
        "events": replay["events"],
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return (_TEMPLATE
            .replace("__HEIGHT__", str(height))
            .replace("__CDN__", _CDN)
            .replace("__DATA__", payload))


_TEMPLATE = r"""
<!-- yard_rl three.js viewer -->
<style>
  html,body{margin:0;padding:0;overflow:hidden;font-family:system-ui,sans-serif}
  #wrap{position:relative;width:100%;height:__HEIGHT__px;background:#dfeef7}
  #hud{position:absolute;top:10px;left:12px;background:rgba(255,255,255,.85);
       border-radius:8px;padding:8px 12px;font-size:13px;line-height:1.5;
       box-shadow:0 1px 4px rgba(0,0,0,.15);min-width:230px}
  #hud b{color:#1a1a2e} #hud .rule{color:#b3560a;font-weight:600}
  #bar{position:absolute;left:12px;right:12px;bottom:10px;display:flex;
       gap:8px;align-items:center;background:rgba(255,255,255,.85);
       border-radius:8px;padding:6px 12px;box-shadow:0 1px 4px rgba(0,0,0,.15)}
  #bar button{border:1px solid #cbd5e1;background:#fff;border-radius:6px;
       padding:3px 12px;font-size:14px;cursor:pointer}
  #bar button:hover{background:#f1f5f9}
  #tl{flex:1} #spd{font-size:12px}
  #legend{position:absolute;top:10px;right:12px;background:rgba(255,255,255,.85);
       border-radius:8px;padding:6px 10px;font-size:11.5px;line-height:1.6;
       box-shadow:0 1px 4px rgba(0,0,0,.15)}
  .sw{display:inline-block;width:10px;height:10px;border-radius:2px;
      margin-right:4px;vertical-align:-1px}
</style>
<div id="wrap">
  <div id="hud">로딩 중…</div>
  <div id="legend">
    <span class="sw" style="background:#e8710a"></span>크레인 (YC)<br>
    <span class="sw" style="background:#1565c0"></span>대기 트럭 ·
    <span class="sw" style="background:#c62828"></span>SLA 초과<br>
    <span class="sw" style="background:#1b8a3a"></span>작업 중 bay ·
    <span class="sw" style="background:#8e24aa"></span>본선 대상
  </div>
  <div id="bar">
    <button id="play">▶ 재생</button>
    <input id="tl" type="range" min="0" max="1000" value="0">
    <select id="spd">
      <option value="60">60×</option><option value="120" selected>120×</option>
      <option value="300">300×</option><option value="600">600×</option>
    </select>
  </div>
</div>
<script type="importmap">
{"imports":{"three":"__CDN__/build/three.module.js",
            "three/addons/":"__CDN__/examples/jsm/"}}
</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const D = __DATA__;
const B=D.block.bay_count, R=D.block.row_count, T=D.block.tier_max;
const BL=D.block.bay_length_m||6.5, RW=D.block.row_width_m||3.1,
      TH=D.block.tier_height_m||2.6;
const L=B*BL, W=R*RW, LANE=4.2, GATEX=-30;
// 좌표: three.js y=up. sim(x=bay축, row축, tier) → three(x, z=row축, y=높이)
const bayX = b => (b-0.5)*BL;
const rowZ = r => (r-0.5)*RW;

// ---------- 장면 기본 ----------
const wrap = document.getElementById('wrap');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xdfeef7);
scene.fog = new THREE.Fog(0xdfeef7, 260, 560);
const camera = new THREE.PerspectiveCamera(42, wrap.clientWidth/wrap.clientHeight, 1, 900);
camera.position.set(L*0.35, 95, -78);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(wrap.clientWidth, wrap.clientHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
wrap.prepend(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(L/2, 0, W/2);
controls.maxPolarAngle = Math.PI/2 - 0.05;
controls.update();

scene.add(new THREE.HemisphereLight(0xffffff, 0x8fa3b0, 1.05));
const sun = new THREE.DirectionalLight(0xfff4e0, 1.6);
sun.position.set(-60, 120, -80);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, {left:-140, right:L+60, top:120, bottom:-90});
sun.shadow.camera.far = 400;
sun.shadow.camera.updateProjectionMatrix();  // 경계 변경 반영 (기본 ±5m 절두체 방지)
scene.add(sun);

const box = (w,h,d,color,x,y,z,parent=scene,shadow=true)=>{
  const m = new THREE.Mesh(new THREE.BoxGeometry(w,h,d),
    new THREE.MeshStandardMaterial({color, roughness:.85, metalness:.05}));
  m.position.set(x,y,z); m.castShadow=shadow; m.receiveShadow=true;
  parent.add(m); return m;
};

// ---------- 씨너리: 포장·차선·게이트·안벽·바다 ----------
box(L-GATEX+80, 1, W+LANE+16, 0xd8d3c9, (L+GATEX)/2, -0.55, W/2-2, scene, false);
box(L-GATEX+80, 0.1, LANE, 0x8f979f, (L+GATEX)/2, -0.02, -LANE/2-0.2, scene, false);
for (let x=GATEX+2; x<L+16; x+=8)             // 차선 점선
  box(2.6, 0.05, 0.24, 0xf2f2ee, x, 0.05, -LANE/2-0.2, scene, false);
box(L-GATEX+80, 0.6, 6, 0xbfb9ae, (L+GATEX)/2, -0.32, W+4.5, scene, false); // 안벽
const sea = box(L-GATEX+120, 0.4, 30, 0x4d8ab5, (L+GATEX)/2, -0.55, W+22.5, scene, false);
sea.material.transparent = true; sea.material.opacity = .92;
box(4, 0.4, 6, 0x555f6e, GATEX+6, 5.4, -LANE/2, scene);   // 게이트 캐노피
box(0.6, 5.4, 0.6, 0x555f6e, GATEX+4.2, 2.7, -LANE-0.4, scene);
box(0.6, 5.4, 0.6, 0x555f6e, GATEX+7.8, 2.7, 1.4, scene);
// 볼라드 (안벽 위 계선주)
for (let x=0; x<L; x+=24) box(1.2, 0.9, 1.2, 0x3d434b, x+6, 0.15, W+6.2, scene);
const label = (txt,x,z,size=140)=>{
  const c=document.createElement('canvas'); c.width=512; c.height=128;
  const g=c.getContext('2d'); g.font='48px system-ui'; g.fillStyle='#44505e';
  g.textAlign='center'; g.fillText(txt,256,80);
  const s=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),
    transparent:true, depthWrite:false}));
  s.position.set(x, 7, z); s.scale.set(size/4, size/16, 1); scene.add(s);
};
label('게이트 (육측)', GATEX+6, -LANE-6);
label('안벽 · 해측', L/2, W+13);

// ---------- 컨테이너 (InstancedMesh, slot 해시 배색) ----------
const PALETTE=[0x7a4a38,0x31597f,0x4e6e51,0x8a8d93,0xa06a30,
               0x27605c,0x6e5a7e,0x95793f,0x5b7285,0x874f56].map(c=>new THREE.Color(c));
const contGeo = new THREE.BoxGeometry(BL-0.8, TH-0.14, RW-0.4);
const contMat = new THREE.MeshStandardMaterial({roughness:.8, metalness:.08});
const cont = new THREE.InstancedMesh(contGeo, contMat, B*R*T);
cont.castShadow = cont.receiveShadow = true;
cont.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
scene.add(cont);
const mtx = new THREE.Matrix4();
function setStacks(hs){
  let n=0;
  for (let b=0;b<B;b++) for (let r=0;r<R;r++){
    const h=hs[b][r];
    for (let k=0;k<h;k++){
      mtx.makeTranslation(bayX(b+1), k*TH+TH/2, rowZ(r+1));
      cont.setMatrixAt(n, mtx);
      const base = PALETTE[(b*7919+r*104729+k*1299709)%PALETTE.length].clone();
      cont.setColorAt(n, base.lerp(new THREE.Color(0xffffff), .08+.26*k/Math.max(1,T-1)));
      n++;
    }
  }
  cont.count=n; cont.instanceMatrix.needsUpdate=true;
  if (cont.instanceColor) cont.instanceColor.needsUpdate=true;
}

// ---------- 크레인 (ARMG 갠트리 + 스프레더) ----------
const crane = new THREE.Group(); scene.add(crane);
const beamY = T*TH+2.4, CO=0xe8710a;
box(2.4, beamY, 2.4, CO, 0, beamY/2, -LANE-1, crane);
box(2.4, beamY, 2.4, CO, 0, beamY/2, W+1, crane);
box(2.6, 1.6, W+LANE+4.5, CO, 0, beamY+0.8, (W-LANE)/2, crane);
box(1.0, 0.5, W+LANE+4.5, 0xc4600a, 0, 0.25, (W-LANE)/2, crane); // 하부 레일보
const trolley = box(3.2, 1.5, 3.0, 0xb3560a, 0, beamY-0.75, W/2, crane);
const spreader = new THREE.Group(); crane.add(spreader);
const cable = box(0.18, 1, 0.18, 0x333333, 0, 0, W/2, spreader, false);
const spBeam = box(BL*0.75, 0.55, RW*0.8, 0xb3560a, 0, 0, W/2, spreader);
function setSpreader(y){  // y = 스프레더 하단 높이
  const top = beamY-1.5;
  cable.scale.y = Math.max(0.01, top-y);
  cable.position.y = (top+y)/2;
  spBeam.position.y = y;
}
setSpreader(beamY-2.5);
// 작업 bay 표시 (지면 초록 링)
const ring = new THREE.Mesh(new THREE.RingGeometry(BL*0.45, BL*0.62, 32),
  new THREE.MeshBasicMaterial({color:0x1b8a3a, transparent:true, opacity:.5,
                               side:THREE.DoubleSide}));
ring.rotation.x = -Math.PI/2; ring.position.y = 0.06; scene.add(ring);

// ---------- 트럭 (운전석+차대+적재함+바퀴) ----------
const BLUE=new THREE.Color(0x1565c0), RED=new THREE.Color(0xc62828);
function makeTruck(){
  const g=new THREE.Group();
  box(1.9, 2.3, 2.4, 0x3b4252, -4.2, 1.45, 0, g);                 // 운전석
  const body=box(6.2, 2.5, 2.4, 0x1565c0, 0.4, 1.85, 0, g);       // 적재함
  box(8.6, 0.5, 2.2, 0x2b2f36, -0.4, 0.45, 0, g);                 // 차대
  const wg=new THREE.CylinderGeometry(0.42,0.42,0.3,10);
  const wm=new THREE.MeshStandardMaterial({color:0x1c1e22});
  [[-4.4,1.1],[-4.4,-1.1],[1.2,1.1],[1.2,-1.1],[3.2,1.1],[3.2,-1.1]].forEach(([x,z])=>{
    const w=new THREE.Mesh(wg,wm); w.rotation.x=Math.PI/2;
    w.position.set(x,0.42,z); w.castShadow=true; g.add(w);
  });
  g.userData.body=body; scene.add(g); g.visible=false; return g;
}

// ---------- 이벤트 인덱싱 ----------
const arrive={}, dispatch={}, complete={};
for (const [t,kind,job] of D.events){
  if (kind==='BLOCK_ARRIVAL') arrive[job]=t;
  else if (kind==='DISPATCH' && !(job in dispatch)) dispatch[job]=t;
  else if (kind==='JOB_COMPLETED' && !(job in complete)) complete[job]=t;
}
const trucks=[];   // {g, job, arrive, dispatch, complete, parkX, gateSlot}
let gateSlot=0;
for (const [job,meta] of Object.entries(D.jobs)){
  if (!meta.external || !(job in arrive)) continue;
  const t=makeTruck();
  const parkX = meta.target_bay ? bayX(meta.target_bay) : GATEX+14+9*(gateSlot++%3);
  trucks.push({g:t, job, meta, a:arrive[job], d:dispatch[job]??1e18,
               c:complete[job]??1e18, parkX});
}
// 본선 대상 표식 (보라 비콘)
const beacons=[];
for (const [job,meta] of Object.entries(D.jobs)){
  if (!meta.vessel || !meta.target_bay) continue;
  const bk=new THREE.Mesh(new THREE.OctahedronGeometry(0.9),
    new THREE.MeshStandardMaterial({color:0x8e24aa, emissive:0x3d0a52}));
  bk.visible=false; scene.add(bk);
  beacons.push({m:bk, meta, done:complete[job]??1e18});
}

// ---------- 재생 루프 ----------
const ds=D.decisions, END=D.end_s;
let simT=0, playing=false, lastReal=0, stackIdx=-1;
const hud=document.getElementById('hud'), tl=document.getElementById('tl'),
      playBtn=document.getElementById('play'), spd=document.getElementById('spd');
playBtn.onclick=()=>{playing=!playing; playBtn.textContent=playing?'⏸ 일시정지':'▶ 재생';};
tl.oninput=()=>{simT=tl.value/1000*END; stackIdx=-1;};
const fmt=s=>`${String(Math.floor(s/3600)).padStart(2,'0')}:${String(Math.floor(s%3600/60)).padStart(2,'0')}`;
const lerp=(a,b,f)=>a+(b-a)*Math.min(1,Math.max(0,f));

function decisionAt(t){
  let lo=0, hi=ds.length-1, ans=0;
  while (lo<=hi){const m=(lo+hi)>>1; if (ds[m].t<=t){ans=m;lo=m+1;} else hi=m-1;}
  return ans;
}

function update(){
  // 장치 스택: 현재 결정 스냅샷 (dispatch 시점 일괄 반영 — recorder 계약)
  const di=decisionAt(simT), d=ds[di];
  if (di!==stackIdx){ setStacks(d.stack_heights); stackIdx=di; }
  // 크레인: 결정 시점부터 이동 (속도 비례) 후 작업 — 다음 결정까지 유지
  const fromX=bayX(d.crane_bay_before), toX=bayX(d.crane_bay_after);
  const moveDur=Math.max(20, Math.abs(toX-fromX)/4.0);   // 4 m/s 갠트리
  const cx=lerp(fromX, toX, (simT-d.t)/moveDur);
  crane.position.x=cx;
  ring.position.x=toX;
  // 스프레더: 작업 구간(t~task_end) 동안 하강-상승 사이클
  if (simT>d.t+moveDur && simT<d.task_end_t){
    const f=(simT-d.t-moveDur)/Math.max(1,d.task_end_t-d.t-moveDur);
    const col=d.stack_heights[Math.min(B-1,Math.max(0,Math.round(d.crane_bay_after)-1))];
    const top=Math.max(...col)*TH+1.2;
    setSpreader(lerp(beamY-2.5, top, Math.sin(Math.PI*f)));
  } else setSpreader(beamY-2.5);
  // 트럭 상태기계: 진입(게이트→주차) → 대기 → 서비스 → 퇴장(+x)
  let waiting=0;
  for (const t of trucks){
    const g=t.g, IN=45, OUT=40;                    // 진입/퇴장 소요(sim s)
    if (simT<t.a-IN || simT>t.c+OUT){ g.visible=false; continue; }
    g.visible=true;
    let x;
    if (simT<t.a) x=lerp(GATEX-14, t.parkX, (simT-(t.a-IN))/IN);
    else if (simT<t.c) x=t.parkX;
    else x=lerp(t.parkX, t.parkX+60+ (L-t.parkX), (simT-t.c)/OUT);
    g.position.set(x, 0, -LANE/2-0.2);
    if (simT>=t.a && simT<t.d){
      waiting++;
      g.userData.body.material.color.copy(simT-t.a>D.sla_s?RED:BLUE);
    } else g.userData.body.material.color.copy(BLUE);
  }
  for (const b of beacons){
    b.m.visible = simT<b.done;
    if (b.m.visible){
      b.m.position.set(bayX(b.meta.target_bay), T*TH+2.2+0.5*Math.sin(simT/40),
                       rowZ(b.meta.target_row||1));
      b.m.rotation.y=simT/30;
    }
  }
  let done=0; for (const t of trucks) if (simT>=t.c) done++;
  hud.innerHTML=`<b>${D.terminal} · ${D.policy}</b><br>`
    +`⏱ ${fmt(simT)} / ${fmt(END)}<br>`
    +`결정 #${d.i+1}/${ds.length} — <span class="rule">${d.rule}</span>`
    +` → ${d.selected??''}<br>`
    +`대기 트럭 <b>${waiting}</b> · 완료 <b>${done}</b>/${trucks.length}`;
  tl.value=simT/END*1000;
}

renderer.setAnimationLoop((now)=>{
  if (playing){
    const dt=(now-lastReal)/1000;
    simT=Math.min(END, simT+dt*Number(spd.value));
    if (simT>=END){playing=false; playBtn.textContent='▶ 재생';}
  }
  lastReal=now;
  update();
  renderer.render(scene, camera);
});
addEventListener('resize', ()=>{
  camera.aspect=wrap.clientWidth/wrap.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(wrap.clientWidth, wrap.clientHeight);
});
</script>
"""
