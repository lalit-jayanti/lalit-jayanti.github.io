// Interactive demo: a simulated Unitree G1 tracks motion clips while you push it around.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { toCreasedNormals } from 'three/addons/utils/BufferGeometryUtils.js';
import { Controller } from './controller.js';

const MUJOCO_URL = 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@4.0.0/mujoco.js';
const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.bundle.min.mjs';
const BASE = new URL('./', import.meta.url);
const ASSETS = new URL('assets/', BASE);

const DRAG_GAIN = 900; // N per metre of drag distance
const DRAG_DAMP = 40; // N per (m/s) at the grab point
const DRAG_MAX = 260; // N
const PUSH_FORCE = 180; // N
const PUSH_TIME = 0.2; // s
const VISUAL_GROUP = 2;

const ICONS = {
  play: '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M4 2.5v11l9-5.5z"/></svg>',
  pause: '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z"/></svg>',
  reset: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M2.5 8a5.5 5.5 0 1 0 1.7-4"/><path d="M2.5 2.5v3h3"/></svg>',
  push: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2 8h10M9 4.5 12.5 8 9 11.5"/><path d="M14.5 3v10"/></svg>',
};

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

function loadCss(href) {
  if (document.querySelector(`link[href="${href}"]`)) return;
  const l = document.createElement('link');
  l.rel = 'stylesheet';
  l.href = href;
  document.head.appendChild(l);
}

// Fetch with byte-level progress reporting.
async function fetchBytes(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} → HTTP ${res.status}`);
  if (!res.body) {
    const b = new Uint8Array(await res.arrayBuffer());
    onBytes(b.length);
    return b;
  }
  const reader = res.body.getReader();
  const chunks = [];
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    n += value.length;
    onBytes(value.length);
  }
  const out = new Uint8Array(n);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

function stiffnessFromSlider(v, lo, hi) {
  return Math.exp(Math.log(lo) + (Math.log(hi) - Math.log(lo)) * v);
}

export async function mount(root, info) {
  loadCss(new URL('../common.css', BASE).href);
  const wrap = el('div', 'cg');
  const stage = el('div', 'cg-stage');
  const canvas = el('canvas');
  canvas.tabIndex = 0;
  canvas.setAttribute('aria-label', 'Robot simulation. Drag the robot to push it; drag elsewhere to orbit the camera.');
  stage.appendChild(canvas);

  const loading = el('div', 'cg-loading', '<div class="track"><div class="fill"></div></div><div class="msg">Loading…</div>');
  stage.appendChild(loading);
  const fill = loading.querySelector('.fill');
  const msg = loading.querySelector('.msg');

  const panel = el('div', 'cg-panel');
  panel.innerHTML = `
    <div>
      <div class="cg-label">Motion</div>
      <div class="cg-motions"></div>
    </div>
    <div>
      <div class="cg-label">Stiffness K <span class="cg-value"></span></div>
      <div class="cg-slider"><input type="range" min="0" max="1000" step="1" aria-label="Stiffness K"></div>
      <div class="cg-scale"><span>soft · 40</span><span>600 · stiff</span></div>
    </div>
    <div class="cg-actions">
      <button class="cg-btn" data-act="pause">${ICONS.pause}<span>Pause</span></button>
      <button class="cg-btn" data-act="push">${ICONS.push}<span>Push</span></button>
      <button class="cg-btn" data-act="reset">${ICONS.reset}<span>Reset</span></button>
    </div>`;
  const hint = el('div', 'cg-hint', '<b>Drag the robot</b> to push it<span class="more"> · drag elsewhere to orbit · scroll to zoom</span>');
  const forceBox = el('div', 'cg-force', 'Force <span>0 N</span><div class="bar"><i></i></div>');
  const toast = el('div', 'cg-toast');
  stage.append(hint, forceBox, toast);
  wrap.append(stage, panel);
  root.appendChild(wrap);
  renderInfo(info);

  let disposed = false;
  const cleanups = [];
  const api = {
    dispose() {
      disposed = true;
      for (const f of cleanups.splice(0)) {
        try {
          f();
        } catch (e) {
          /* already released */
        }
      }
    },
  };

  try {
    await start();
  } catch (err) {
    console.error(err);
    loading.innerHTML = `<div class="err">The demo could not start in this browser.<br><small>${String(err.message || err)}</small></div>`;
  }
  return api;

  async function start() {
    // ── downloads ──
    const config = await (await fetch(new URL('config.json', ASSETS))).json();
    const motionIndex = await (await fetch(new URL('motions/index.json', ASSETS))).json();
    const xml = await (await fetch(new URL('robot/scene.xml', ASSETS))).text();
    const meshFiles = [...new Set([...xml.matchAll(/file="([^"]+)"/g)].map((m) => m[1]))];

    const sizes = { mujoco: 11.2e6, ort: 11e6, policy: 17.7e6, meshes: 11.7e6 };
    const total = Object.values(sizes).reduce((a, b) => a + b, 0);
    let got = 0;
    const bump = (n) => {
      got += n;
      fill.style.width = `${Math.min(100, (100 * got) / total).toFixed(1)}%`;
    };
    msg.textContent = 'Loading physics, policy and robot…';

    const mujocoP = import(MUJOCO_URL).then(async (m) => {
      const mod = await m.default();
      bump(sizes.mujoco);
      return mod;
    });
    const ortP = import(ORT_URL).then((m) => {
      bump(sizes.ort);
      return m;
    });
    let policyBytesSeen = 0;
    const policyP = fetchBytes(new URL('policy.onnx', ASSETS), (n) => {
      policyBytesSeen += n;
      bump(n);
    });
    const meshP = Promise.all(
      meshFiles.map((f) =>
        fetchBytes(new URL('robot/meshes/' + f, ASSETS), (n) => bump(n)).then((b) => [f, b])
      )
    );
    const motionsP = Promise.all(
      motionIndex.map((m) => fetch(new URL(`motions/${m.id}.json`, ASSETS)).then((r) => r.json()).then((c) => [m.id, c]))
    );
    const [mujoco, ortMod, policyBytes, meshes, motionList] = await Promise.all([mujocoP, ortP, policyP, meshP, motionsP]);
    if (disposed) return;
    bump(sizes.policy - policyBytesSeen);
    const ort = ortMod.default || ortMod;
    const motions = Object.fromEntries(motionList);

    // ── physics ──
    msg.textContent = 'Starting simulation…';
    const vfs = new mujoco.MjVFS();
    for (const [f, b] of meshes) vfs.addBuffer('meshes/' + f, b);
    vfs.addBuffer('scene.xml', new TextEncoder().encode(xml));
    const model = mujoco.MjModel.mj_loadXML('scene.xml', vfs);
    const data = new mujoco.MjData(model);
    cleanups.push(() => data.delete(), () => model.delete(), () => vfs.delete());

    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(policyBytes, { executionProviders: ['wasm'] });
    cleanups.push(() => session.release());
    const ctrl = new Controller({ mujoco, model, data, config, session, ort });

    // ── scene ──
    const view = buildScene(model, data, canvas, stage);
    cleanups.push(() => view.dispose());

    // ── state ──
    const [Klo, Khi] = config.stiffness_range;
    const slider = panel.querySelector('input[type=range]');
    const kLabel = panel.querySelector('.cg-value');
    let motionId = 'stand';
    let paused = false;
    let fallTime = 0;
    let push = null;
    let drag = null;

    function setStiffness(v) {
      ctrl.stiffness = stiffnessFromSlider(v, Klo, Khi);
      kLabel.innerHTML = `${Math.round(ctrl.stiffness)}<small>N/m</small>`;
      slider.parentElement.style.setProperty('--p', `${(v * 100).toFixed(1)}%`);
      slider.style.setProperty('--p', `${(v * 100).toFixed(1)}%`);
    }
    slider.value = String(Math.round(1000 * ((Math.log(150) - Math.log(Klo)) / (Math.log(Khi) - Math.log(Klo)))));
    setStiffness(slider.value / 1000);
    slider.addEventListener('input', () => setStiffness(slider.value / 1000));

    const motionBox = panel.querySelector('.cg-motions');
    for (const m of motionIndex) {
      const b = el('button', 'cg-btn', m.label);
      b.dataset.motion = m.id;
      b.addEventListener('click', () => selectMotion(m.id));
      motionBox.appendChild(b);
    }
    function selectMotion(id) {
      motionId = id;
      motionBox.querySelectorAll('.cg-btn').forEach((b) => b.classList.toggle('active', b.dataset.motion === id));
      ctrl.setClip(motions[id]);
    }

    function resetAll() {
      ctrl.resetRobot(config.start_qpos);
      selectMotion(motionId);
      drag = null;
      push = null;
      fallTime = 0;
      view.snapCamera(data);
    }

    let toastTimer = 0;
    function showToast(text) {
      toast.textContent = text;
      toast.classList.add('on');
      clearTimeout(toastTimer);
      toastTimer = setTimeout(() => toast.classList.remove('on'), 1800);
    }

    const pauseBtn = panel.querySelector('[data-act=pause]');
    function setPaused(p) {
      paused = p;
      pauseBtn.innerHTML = p ? `${ICONS.play}<span>Play</span>` : `${ICONS.pause}<span>Pause</span>`;
    }
    pauseBtn.addEventListener('click', () => setPaused(!paused));
    panel.querySelector('[data-act=reset]').addEventListener('click', resetAll);
    panel.querySelector('[data-act=push]').addEventListener('click', () => {
      const yaw = Math.random() * 2 * Math.PI;
      push = { t: 0, dir: [Math.cos(yaw), Math.sin(yaw), 0] };
    });

    // ── external forces, applied at every physics step ──
    const torsoId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'torso_link');
    const dt = model.opt.timestep;
    ctrl.applyExternal = (d) => {
      const xf = d.xfrc_applied;
      xf.fill(0);
      if (push) {
        const b = torsoId;
        for (let i = 0; i < 3; i++) xf[6 * b + i] += PUSH_FORCE * push.dir[i];
        push.t += dt;
        if (push.t >= PUSH_TIME) push = null;
      }
      if (drag && drag.target) {
        const b = drag.body;
        const p = anchorWorld(d, b, drag.local);
        const v = drag.prev ? p.map((x, i) => (x - drag.prev[i]) / dt) : [0, 0, 0];
        drag.prev = p;
        let F = [0, 1, 2].map((i) => DRAG_GAIN * (drag.target[i] - p[i]) - DRAG_DAMP * v[i]);
        const mag = Math.hypot(...F);
        if (mag > DRAG_MAX) F = F.map((x) => (x * DRAG_MAX) / mag);
        drag.force = Math.min(mag, DRAG_MAX);
        drag.anchor = p;
        const c = [d.xipos[3 * b], d.xipos[3 * b + 1], d.xipos[3 * b + 2]];
        const r = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        xf[6 * b] += F[0];
        xf[6 * b + 1] += F[1];
        xf[6 * b + 2] += F[2];
        xf[6 * b + 3] += r[1] * F[2] - r[2] * F[1];
        xf[6 * b + 4] += r[2] * F[0] - r[0] * F[2];
        xf[6 * b + 5] += r[0] * F[1] - r[1] * F[0];
      }
    };

    // ── dragging ──
    const raycaster = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    const dragPlane = new THREE.Plane();
    const tmp = new THREE.Vector3();
    function toNdc(e) {
      const r = canvas.getBoundingClientRect();
      ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    }
    function onDown(e) {
      if (e.button !== 0 && e.pointerType === 'mouse') return;
      toNdc(e);
      raycaster.setFromCamera(ndc, view.camera);
      const hit = raycaster.intersectObjects(view.pickables, false)[0];
      if (!hit) return;
      const b = hit.object.userData.body;
      const p = hit.point;
      const R = data.xmat;
      const x = data.xpos;
      const dp = [p.x - x[3 * b], p.y - x[3 * b + 1], p.z - x[3 * b + 2]];
      const local = [0, 1, 2].map((i) => R[9 * b + i] * dp[0] + R[9 * b + 3 + i] * dp[1] + R[9 * b + 6 + i] * dp[2]);
      view.camera.getWorldDirection(tmp);
      dragPlane.setFromNormalAndCoplanarPoint(tmp.clone().negate(), p);
      drag = { body: b, local, target: [p.x, p.y, p.z], anchor: [p.x, p.y, p.z], prev: null, force: 0 };
      view.controls.enabled = false;
      canvas.setPointerCapture(e.pointerId);
      canvas.classList.add('grabbing');
      hint.style.opacity = '0';
      e.preventDefault();
      e.stopImmediatePropagation();
    }
    function onMove(e) {
      if (!drag) {
        if (e.pointerType === 'mouse') {
          toNdc(e);
          raycaster.setFromCamera(ndc, view.camera);
          canvas.classList.toggle('grab', raycaster.intersectObjects(view.pickables, false).length > 0);
        }
        return;
      }
      toNdc(e);
      raycaster.setFromCamera(ndc, view.camera);
      if (raycaster.ray.intersectPlane(dragPlane, tmp)) drag.target = [tmp.x, tmp.y, Math.max(0.02, tmp.z)];
    }
    function onUp(e) {
      if (!drag) return;
      drag = null;
      view.controls.enabled = true;
      canvas.classList.remove('grabbing');
      try {
        canvas.releasePointerCapture(e.pointerId);
      } catch (err) {
        /* not captured */
      }
    }
    canvas.addEventListener('pointerdown', onDown, { capture: true });
    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerup', onUp);
    canvas.addEventListener('pointercancel', onUp);

    const onKey = (e) => {
      if (e.target instanceof HTMLInputElement) return;
      if (e.code === 'Space' && document.activeElement === canvas) {
        setPaused(!paused);
        e.preventDefault();
      }
    };
    window.addEventListener('keydown', onKey);
    cleanups.push(() => window.removeEventListener('keydown', onKey));

    // ── run only while visible ──
    let visible = true;
    const io = new IntersectionObserver(([en]) => (visible = en.isIntersecting), { threshold: 0.05 });
    io.observe(stage);
    cleanups.push(() => io.disconnect());

    resetAll();
    loading.classList.add('done');

    // ── main loop: policy at 50 Hz in real time, rendering every frame ──
    const controlDt = config.control_dt;
    let acc = 0;
    let last = performance.now();
    let busy = false;
    let raf = 0;
    const stats = { frames: 0, render: 0, act: 0, physics: 0, steps: 0 };
    if (new URLSearchParams(location.search).has('debug')) Object.assign(window, { __g1stats: stats, __g1view: view });
    async function tick() {
      busy = true;
      try {
        // Catch up to real time within a per-frame budget, so slow rendering does not slow the simulation.
        const tStart = performance.now();
        while (acc >= controlDt) {
          if (performance.now() - tStart > 35) {
            acc = Math.min(acc, controlDt);
            break;
          }
          const t0 = performance.now();
          await ctrl.act();
          const t1 = performance.now();
          ctrl.physics();
          ctrl.advance();
          stats.act += t1 - t0;
          stats.physics += performance.now() - t1;
          stats.steps += 1;
          acc -= controlDt;
          const p = ctrl.posture();
          if (p.height < 0.35 || p.tilt > 70) fallTime += controlDt;
          else fallTime = 0;
          if (fallTime > 1.0) {
            showToast('The robot fell, resetting');
            resetAll();
            break;
          }
        }
      } catch (err) {
        console.error(err);
        paused = true;
      }
      busy = false;
    }
    function frame(now) {
      if (disposed) return;
      raf = requestAnimationFrame(frame);
      const elapsed = Math.min(0.25, (now - last) / 1000);
      last = now;
      if (!visible || document.hidden) return;
      view.adapt(elapsed);
      if (!paused) acc += elapsed;
      if (!busy && !paused) tick();
      const tr = performance.now();
      view.render(data, drag);
      stats.render += performance.now() - tr;
      stats.frames += 1;
      const F = drag ? drag.force : push ? PUSH_FORCE : 0;
      forceBox.classList.toggle('on', F > 0);
      forceBox.querySelector('span').textContent = `${Math.round(F)} N`;
      forceBox.querySelector('i').style.width = `${Math.min(100, (100 * F) / DRAG_MAX)}%`;
    }
    raf = requestAnimationFrame(frame);
    cleanups.unshift(() => cancelAnimationFrame(raf));
  }
}

function anchorWorld(d, b, local) {
  const R = d.xmat;
  const x = d.xpos;
  return [0, 1, 2].map(
    (i) => x[3 * b + i] + R[9 * b + 3 * i] * local[0] + R[9 * b + 3 * i + 1] * local[1] + R[9 * b + 3 * i + 2] * local[2]
  );
}

function floorTexture() {
  const s = 512;
  const c = document.createElement('canvas');
  c.width = c.height = s;
  const g = c.getContext('2d');
  g.fillStyle = '#f4f4f6';
  g.fillRect(0, 0, s, s);
  g.fillStyle = '#ececf0';
  g.fillRect(0, 0, s / 2, s / 2);
  g.fillRect(s / 2, s / 2, s / 2, s / 2);
  g.strokeStyle = '#dcdce2';
  g.lineWidth = 2;
  g.strokeRect(0, 0, s, s);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = 8;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function buildScene(model, data, canvas, stage) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  const maxRatio = Math.min(2, window.devicePixelRatio || 1);
  let ratio = maxRatio;
  renderer.setPixelRatio(ratio);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#f3f3f5');
  scene.fog = new THREE.Fog('#f3f3f5', 9, 22);

  const camera = new THREE.PerspectiveCamera(40, 16 / 9, 0.05, 60);
  camera.up.set(0, 0, 1);
  camera.position.set(2.6, -2.6, 1.5);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 1.2;
  controls.maxDistance = 9;
  controls.maxPolarAngle = Math.PI * 0.495;
  controls.target.set(0, 0, 0.75);

  scene.add(new THREE.HemisphereLight('#ffffff', '#d8d4e4', 1.3));
  const sun = new THREE.DirectionalLight('#ffffff', 2.2);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  sun.shadow.camera.left = sun.shadow.camera.bottom = -2.5;
  sun.shadow.camera.right = sun.shadow.camera.top = 2.5;
  sun.shadow.camera.near = 0.5;
  sun.shadow.camera.far = 12;
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.02;
  scene.add(sun, sun.target);
  const rim = new THREE.DirectionalLight('#e9ddff', 0.6);
  rim.position.set(-3, 3, 2);
  scene.add(rim);

  const tex = floorTexture();
  tex.repeat.set(40, 40);
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(80, 80),
    new THREE.MeshStandardMaterial({ map: tex, roughness: 0.95, metalness: 0 })
  );
  floor.receiveShadow = true;
  scene.add(floor);

  // Robot meshes from the compiled model (visual group only).
  const geoms = [];
  const pickables = [];
  const vert = model.mesh_vert;
  const face = model.mesh_face;
  const vadr = model.mesh_vertadr;
  const vnum = model.mesh_vertnum;
  const fadr = model.mesh_faceadr;
  const fnum = model.mesh_facenum;
  const matCache = new Map();
  for (let g = 0; g < model.ngeom; g++) {
    if (model.geom_group[g] !== VISUAL_GROUP || model.geom_type[g] !== 7) continue;
    const m = model.geom_dataid[g];
    const v0 = vadr[m];
    const pos = new Float32Array(vert.subarray(3 * v0, 3 * (v0 + vnum[m])));
    const idx = new Uint32Array(face.subarray(3 * fadr[m], 3 * (fadr[m] + fnum[m])));
    let geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    // Smooth shading across curved surfaces, hard edges kept where faces meet at more than 35 degrees.
    geo = toCreasedNormals(geo, (35 * Math.PI) / 180);
    const mid = model.geom_matid[g];
    const rgba = mid >= 0 ? model.mat_rgba.subarray(4 * mid, 4 * mid + 4) : model.geom_rgba.subarray(4 * g, 4 * g + 4);
    const key = Array.from(rgba).join(',');
    if (!matCache.has(key)) {
      const dark = rgba[0] < 0.4;
      matCache.set(
        key,
        new THREE.MeshStandardMaterial({
          color: dark ? new THREE.Color(0.16, 0.16, 0.19) : new THREE.Color(0.86, 0.86, 0.9),
          roughness: dark ? 0.55 : 0.42,
          metalness: dark ? 0.1 : 0.05,
        })
      );
    }
    const mesh = new THREE.Mesh(geo, matCache.get(key));
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.matrixAutoUpdate = false;
    mesh.userData.body = model.geom_bodyid[g];
    mesh.userData.geom = g;
    scene.add(mesh);
    geoms.push(mesh);
    pickables.push(mesh);
  }

  // Drag indicator: a line from the grab point to the cursor and two dots.
  const lineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  const line = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color: '#db2777', depthTest: false }));
  line.renderOrder = 10;
  const dotMat = new THREE.MeshBasicMaterial({ color: '#db2777', depthTest: false });
  const dotA = new THREE.Mesh(new THREE.SphereGeometry(0.018, 16, 12), dotMat);
  const dotB = new THREE.Mesh(new THREE.SphereGeometry(0.026, 16, 12), new THREE.MeshBasicMaterial({ color: '#7c3aed', depthTest: false }));
  dotA.renderOrder = dotB.renderOrder = 11;
  const dragGroup = new THREE.Group();
  dragGroup.add(line, dotA, dotB);
  dragGroup.visible = false;
  scene.add(dragGroup);

  const resize = () => {
    const w = stage.clientWidth;
    const h = stage.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  const ro = new ResizeObserver(resize);
  ro.observe(stage);
  resize();

  const m4 = new THREE.Matrix4();
  const follow = new THREE.Vector3();
  function pelvis(d) {
    return new THREE.Vector3(d.qpos[0], d.qpos[1], 0.75);
  }

  // Adaptive resolution: drop the pixel ratio when frames are slow, restore it when there is headroom.
  let frameEma = 1 / 60;
  let sinceChange = 0;
  const fixedRatio = new URLSearchParams(location.search).has('debug');
  function adapt(dt) {
    if (fixedRatio) return;
    frameEma += 0.1 * (dt - frameEma);
    sinceChange += dt;
    if (sinceChange < 1) return;
    let next = ratio;
    if (frameEma > 1 / 30 && ratio > 0.75) next = Math.max(0.75, ratio * 0.8);
    else if (frameEma < 1 / 55 && ratio < maxRatio) next = Math.min(maxRatio, ratio / 0.8);
    if (next !== ratio) {
      ratio = next;
      renderer.setPixelRatio(ratio);
      resize();
      sinceChange = 0;
    }
  }

  return {
    adapt,
    renderer,
    scene,
    camera,
    controls,
    pickables,
    snapCamera(d) {
      const p = pelvis(d);
      const off = camera.position.clone().sub(controls.target);
      controls.target.copy(p);
      camera.position.copy(p).add(off);
      follow.copy(p);
    },
    render(d, drag) {
      const xpos = d.geom_xpos;
      const xmat = d.geom_xmat;
      for (const mesh of geoms) {
        const g = mesh.userData.geom;
        const R = xmat.subarray(9 * g, 9 * g + 9);
        m4.set(R[0], R[1], R[2], xpos[3 * g], R[3], R[4], R[5], xpos[3 * g + 1], R[6], R[7], R[8], xpos[3 * g + 2], 0, 0, 0, 1);
        mesh.matrix.copy(m4);
        mesh.matrixWorldNeedsUpdate = true;
      }
      // camera and light follow the pelvis
      const p = pelvis(d);
      const delta = p.clone().sub(follow).multiplyScalar(0.08);
      follow.add(delta);
      controls.target.add(delta);
      camera.position.add(delta);
      sun.position.set(follow.x + 2.5, follow.y - 1.5, 5);
      sun.target.position.set(follow.x, follow.y, 0);
      controls.update();

      if (drag && drag.anchor) {
        const a = new THREE.Vector3(...drag.anchor);
        const b = new THREE.Vector3(...drag.target);
        lineGeo.setFromPoints([a, b]);
        dotA.position.copy(a);
        dotB.position.copy(b);
        dragGroup.visible = true;
      } else dragGroup.visible = false;
      renderer.render(scene, camera);
    },
    dispose() {
      ro.disconnect();
      controls.dispose();
      scene.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) o.material.dispose();
      });
      tex.dispose();
      renderer.dispose();
    },
  };
}

function renderInfo(info) {
  info.innerHTML = `
    <div class="cg-info">
      <h2>Compliant General Motion Tracker</h2>
      <p>An implementation of C-GMT.</p>
      <ul class="cg-refs">${REFERENCES.map(refItem).join('')}</ul>
    </div>`;
}

const REFERENCES = [
  {
    title: 'Compliant GMT: Making General Motion Tracking Compliant via Residual Reference Adaptation',
    authors: 'Venki Parsuram, Michelle Wang, Jin-Hyuk Hong, Pulkit Agrawal',
    year: 2026,
    arxiv: '2610.22119',
  },
];

function refItem(r) {
  return `
    <li>
      <div class="cg-ref-title">${r.title}</div>
      <div class="cg-ref-authors">${r.authors} · ${r.year}</div>
      <div class="cg-ref-links">
        <a href="https://arxiv.org/abs/${r.arxiv}" target="_blank" rel="noopener">arXiv</a><a href="https://arxiv.org/pdf/${r.arxiv}" target="_blank" rel="noopener">PDF</a>
      </div>
    </li>`;
}
