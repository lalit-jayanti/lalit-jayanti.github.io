import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { toCreasedNormals, mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { Course } from './course.js';
import { Controller, buildSceneXml, fillHeightField } from './controller.js';
import { Stars } from './stars.js';

const MUJOCO_URL = 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@4.0.0/mujoco.js';
const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.bundle.min.mjs';
const BASE = new URL('./', import.meta.url);
const ASSETS = new URL('assets/', BASE);
const ROBOT = new URL('../g1-tracking/assets/robot/', BASE);
const VISUAL_GROUP = 2;
const KEYS = { KeyW: 'w', KeyA: 'a', KeyS: 's', KeyD: 'd', KeyQ: 'q', KeyE: 'e' };
const SPEED = { fwd: 0.6, fast: 0.9, back: 0.4, side: 0.25, turn: 0.8 };

const ICONS = {
  play: '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M4 2.5v11l9-5.5z"/></svg>',
  pause: '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z"/></svg>',
  reset: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M2.5 8a5.5 5.5 0 1 0 1.7-4"/><path d="M2.5 2.5v3h3"/></svg>',
  cam: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="8" cy="8" r="5.5"/><path d="M8 4.5v3.5l2.5 1.5"/></svg>',
  steps: '<svg viewBox="0 0 16 16" fill="currentColor"><ellipse cx="5" cy="11" rx="2" ry="3"/><ellipse cx="11" cy="5" rx="2" ry="3"/></svg>',
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

export async function mount(root, info) {
  loadCss(new URL('../common.css', BASE).href);
  loadCss(new URL('demo.css', BASE).href);
  const wrap = el('div', 'cg pl');
  const stage = el('div', 'cg-stage');
  const canvas = el('canvas');
  canvas.tabIndex = 0;
  canvas.setAttribute('aria-label', 'Robot simulation. Use W and S to walk, W+Q or W+E to steer, A and D to side-step; drag to orbit the camera.');
  stage.appendChild(canvas);

  const loading = el('div', 'cg-loading', '<div class="track"><div class="fill"></div></div><div class="msg">Loading…</div>');
  stage.appendChild(loading);
  const fill = loading.querySelector('.fill');
  const msg = loading.querySelector('.msg');

  const panel = el('div', 'cg-panel');
  panel.innerHTML = `
    <div>
      <div class="cg-label">Go to</div>
      <div class="cg-motions pl-sections"></div>
    </div>
    <div>
      <div class="cg-label">Command <span class="cg-value pl-cmd"></span></div>
      <div class="pl-bars">
        <div><span>fwd</span><i><b data-k="0"></b></i></div>
        <div><span>side</span><i><b data-k="1"></b></i></div>
        <div><span>turn</span><i><b data-k="2"></b></i></div>
      </div>
    </div>
    <div class="pl-toggles">
      <button class="cg-btn active" data-act="chase">${ICONS.cam}<span>Chase camera</span></button>
      <button class="cg-btn active" data-act="steps">${ICONS.steps}<span>Footsteps</span></button>
    </div>
    <div class="cg-actions pl-actions">
      <button class="cg-btn" data-act="pause">${ICONS.pause}<span>Pause</span></button>
      <button class="cg-btn" data-act="reset">${ICONS.reset}<span>Reset</span></button>
    </div>`;
  const hint = el('div', 'cg-hint', '<b>W S</b> to walk · <b>W+Q</b> / <b>W+E</b> to steer · <b>A D</b> to side-step<span class="more"> · <b>Shift</b> to hurry</span>');
  const where = el('div', 'pl-where', '');
  const score = el('div', 'pl-score', '<svg viewBox="0 0 16 16"><path d="M8 1.2l2 4.3 4.7.6-3.4 3.2.9 4.7L8 11.7 3.8 14l.9-4.7L1.3 6.1 6 5.5z"/></svg><span>0</span>');
  const toast = el('div', 'cg-toast');
  const keypad = el('div', 'cg-keys on', ['q', 'w', 'e', 'a', 's', 'd'].map((k) => `<button tabindex="-1" data-key="${k}">${k.toUpperCase()}</button>`).join(''));
  keypad.setAttribute('aria-hidden', 'true');
  stage.append(hint, where, score, toast, keypad);
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
    const [config, legs, robotXml, courseMeta] = await Promise.all([
      fetch(new URL('config.json', ASSETS)).then((r) => r.json()),
      fetch(new URL('legs.json', ASSETS)).then((r) => r.json()),
      fetch(new URL('scene.xml', ROBOT)).then((r) => r.text()),
      fetch(new URL('course.json', ASSETS)).then((r) => r.json()),
    ]);
    const meshFiles = [...new Set([...robotXml.matchAll(/file="([^"]+)"/g)].map((m) => m[1]))];
    const sizes = { mujoco: 11.2e6, ort: 11e6, policy: 15.2e6, meshes: 11.7e6, course: 0.62e6 };
    const total = Object.values(sizes).reduce((a, b) => a + b, 0);
    let got = 0;
    const bump = (n) => {
      got += n;
      fill.style.width = `${Math.min(100, (100 * got) / total).toFixed(1)}%`;
    };
    msg.textContent = 'Loading physics, policy, robot and course…';
    const mujocoP = import(MUJOCO_URL).then(async (m) => {
      const mod = await m.default();
      bump(sizes.mujoco);
      return mod;
    });
    const ortP = import(ORT_URL).then((m) => {
      bump(sizes.ort);
      return m;
    });
    const policyP = fetchBytes(new URL('policy.onnx', ASSETS), bump);
    const courseP = fetchBytes(new URL('course.bin', ASSETS), bump);
    const meshP = Promise.all(meshFiles.map((f) => fetchBytes(new URL('meshes/' + f, ROBOT), bump).then((b) => [f, b])));
    const [mujoco, ortMod, policyBytes, courseBytes, meshes] = await Promise.all([mujocoP, ortP, policyP, courseP, meshP]);
    if (disposed) return;
    const ort = ortMod.default || ortMod;

    msg.textContent = 'Building the course…';
    await new Promise((r) => setTimeout(r, 0));
    const course = new Course(courseMeta, courseBytes.buffer.slice(courseBytes.byteOffset, courseBytes.byteOffset + courseBytes.byteLength));
    const vfs = new mujoco.MjVFS();
    for (const [f, b] of meshes) vfs.addBuffer('meshes/' + f, b);
    vfs.addBuffer('scene.xml', new TextEncoder().encode(buildSceneXml(robotXml, course, config)));
    const model = mujoco.MjModel.mj_loadXML('scene.xml', vfs);
    fillHeightField(model, course);
    const data = new mujoco.MjData(model);
    cleanups.push(() => data.delete(), () => model.delete(), () => vfs.delete());

    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(policyBytes, { executionProviders: ['wasm'] });
    cleanups.push(() => session.release());
    const ctrl = new Controller({ mujoco, model, data, course, legs, config, session, ort, latency: 5 });

    const view = buildScene(model, course, canvas, stage);
    cleanups.push(() => view.dispose());

    let paused = false;
    let fallTime = 0;
    let section = 'bridge';
    let showSteps = true;

    let toastTimer = 0;
    function showToast(text) {
      toast.textContent = text;
      toast.classList.add('on');
      clearTimeout(toastTimer);
      toastTimer = setTimeout(() => toast.classList.remove('on'), 1800);
    }

    const scoreNum = score.querySelector('span');
    let firstWave = true;
    const stars = new Stars(view.scene, course, {
      onCollect: (n) => {
        scoreNum.textContent = String(n);
        score.classList.remove('bump');
        void score.offsetWidth;
        score.classList.add('bump');
      },
      onWave: (region) => {
        if (firstWave) {
          firstWave = false;
          return;
        }
        showToast(`More stars appeared at the ${region.toLowerCase()}`);
      },
    });
    cleanups.unshift(() => stars.dispose());

    const secBox = panel.querySelector('.pl-sections');
    const SECTIONS = courseMeta.spawns.map((p) => ({ id: p.id, label: p.label, at: [p.x, p.y, p.yaw] }));
    for (const s of SECTIONS) {
      const b = el('button', 'cg-btn', s.label);
      b.dataset.sec = s.id;
      b.addEventListener('click', () => goTo(s.id));
      secBox.appendChild(b);
    }
    function goTo(id) {
      const s = SECTIONS.find((x) => x.id === id);
      section = id;
      secBox.querySelectorAll('.cg-btn').forEach((b) => b.classList.toggle('active', b.dataset.sec === id));
      ctrl.reset(...s.at);
      fallTime = 0;
      view.snapCamera(data, s.at[2]);
      canvas.focus({ preventScroll: true });
    }

    const pauseBtn = panel.querySelector('[data-act=pause]');
    function setPaused(p) {
      paused = p;
      pauseBtn.innerHTML = p ? `${ICONS.play}<span>Play</span>` : `${ICONS.pause}<span>Pause</span>`;
    }
    pauseBtn.addEventListener('click', () => setPaused(!paused));
    panel.querySelector('[data-act=reset]').addEventListener('click', () => goTo(section));
    const chaseBtn = panel.querySelector('[data-act=chase]');
    chaseBtn.addEventListener('click', () => {
      view.chase = !view.chase;
      chaseBtn.classList.toggle('active', view.chase);
    });
    const stepsBtn = panel.querySelector('[data-act=steps]');
    stepsBtn.addEventListener('click', () => {
      showSteps = !showSteps;
      stepsBtn.classList.toggle('active', showSteps);
    });

    const held = new Set();
    let shift = false;
    const showKeys = () => keypad.querySelectorAll('button').forEach((b) => b.classList.toggle('down', held.has(b.dataset.key)));
    function press(k, down) {
      if (down) held.add(k);
      else held.delete(k);
      showKeys();
    }
    const releaseAll = () => {
      held.clear();
      shift = false;
      showKeys();
    };
    const onKey = (e) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      shift = e.shiftKey;
      if (e.code === 'Space' && document.activeElement === canvas) {
        setPaused(!paused);
        e.preventDefault();
        return;
      }
      const k = KEYS[e.code];
      if (!k || e.ctrlKey || e.metaKey || e.altKey || !visible) return;
      if (!e.repeat) press(k, true);
      e.preventDefault();
    };
    const onKeyUp = (e) => {
      shift = e.shiftKey;
      const k = KEYS[e.code];
      if (k) press(k, false);
    };
    keypad.addEventListener('pointerdown', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      b.setPointerCapture(e.pointerId);
      press(b.dataset.key, true);
      e.preventDefault();
    });
    const keypadUp = (e) => {
      const b = e.target.closest('button');
      if (b) press(b.dataset.key, false);
    };
    keypad.addEventListener('pointerup', keypadUp);
    keypad.addEventListener('pointercancel', keypadUp);
    window.addEventListener('keydown', onKey);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', releaseAll);
    cleanups.push(
      () => window.removeEventListener('keydown', onKey),
      () => window.removeEventListener('keyup', onKeyUp),
      () => window.removeEventListener('blur', releaseAll)
    );
    function command() {
      const k = (c) => (held.has(c) ? 1 : 0);
      const vx = (shift ? SPEED.fast : SPEED.fwd) * k('w') - SPEED.back * k('s');
      return [vx, SPEED.side * (k('a') - k('d')), SPEED.turn * (k('q') - k('e'))];
    }

    let visible = true;
    const io = new IntersectionObserver(([en]) => (visible = en.isIntersecting), { threshold: 0.05 });
    io.observe(stage);
    cleanups.push(() => io.disconnect());

    const bars = [...panel.querySelectorAll('.pl-bars b')];
    const cmdLabel = panel.querySelector('.pl-cmd');
    const MAXC = [SPEED.fast, SPEED.side, SPEED.turn];
    function hud(cmd) {
      const eff = ctrl.track.gen.gen.cmdEff;
      bars.forEach((b, i) => {
        const v = Math.max(-1, Math.min(1, eff[i] / MAXC[i]));
        b.style.left = v < 0 ? `${50 + 50 * v}%` : '50%';
        b.style.width = `${50 * Math.abs(v)}%`;
      });
      cmdLabel.innerHTML = `${eff[0].toFixed(2)}<small>m/s</small>`;
      const p = data.qpos;
      const zone = zoneAt(course, p[0], p[1]);
      where.textContent = zone;
      where.classList.toggle('on', !!zone);
    }

    goTo('bridge');
    loading.classList.add('done');
    canvas.focus({ preventScroll: true });

    const controlDt = 0.02;
    let acc = 0;
    let last = performance.now();
    let busy = false;
    let raf = 0;
    const stats = { steps: 0, act: 0, physics: 0, ref: 0, frames: 0, render: 0 };
    if (new URLSearchParams(location.search).has('debug')) Object.assign(window, { __plstats: stats, __plctrl: ctrl, __plview: view, __plstars: stars });
    async function tick() {
      busy = true;
      try {
        const tStart = performance.now();
        while (acc >= controlDt) {
          if (performance.now() - tStart > 35) {
            acc = Math.min(acc, controlDt);
            break;
          }
          ctrl.setCommand(command());
          const t0 = performance.now();
          await ctrl.act();
          const t1 = performance.now();
          ctrl.physics();
          const t2 = performance.now();
          ctrl.advance();
          stats.act += t1 - t0;
          stats.physics += t2 - t1;
          stats.ref += performance.now() - t2;
          stats.steps += 1;
          acc -= controlDt;
          if (ctrl.fallen()) fallTime += controlDt;
          else fallTime = 0;
          if (fallTime > 1.0) {
            showToast('The robot fell, standing it up again');
            const p = ctrl.posture().pos;
            ctrl.reset(p[0], p[1], ctrl.heading());
            fallTime = 0;
            break;
          }
        }
      } catch (err) {
        console.error(err);
        setPaused(true);
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
      stars.update(elapsed, [data.qpos[0], data.qpos[1], data.qpos[2]], view.controls.target);
      view.render(data, ctrl, showSteps, elapsed);
      stats.render += performance.now() - tr;
      stats.frames += 1;
      hud();
    }
    raf = requestAnimationFrame(frame);
    cleanups.unshift(() => cancelAnimationFrame(raf));
  }
}

function zoneAt(course, x, y) {
  for (const z of course.meta.zones) {
    const [x0, x1, y0, y1] = z.r;
    if (x >= x0 && x <= x1 && y >= y0 && y <= y1) return z.label;
  }
  return '';
}

function canvasTexture(size, draw) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  draw(c.getContext('2d'), size);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = 8;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function rng(seed) {
  let r = seed;
  return () => ((r = (r * 16807) % 2147483647) / 2147483647);
}

function marbleTexture(seed, veins, alpha) {
  return canvasTexture(1024, (g, s) => {
    const rnd = rng(seed);
    g.fillStyle = '#ffffff';
    g.fillRect(0, 0, s, s);
    for (let i = 0; i < 40; i++) {
      const x = rnd() * s, y = rnd() * s, r = 80 + rnd() * 220;
      const grd = g.createRadialGradient(x, y, 0, x, y, r);
      grd.addColorStop(0, `rgba(212,212,214,${0.05 + 0.05 * rnd()})`);
      grd.addColorStop(1, 'rgba(212,212,214,0)');
      for (const ox of [-s, 0, s]) for (const oy of [-s, 0, s]) {
        g.save();
        g.translate(ox, oy);
        g.fillStyle = grd;
        g.fillRect(x - r, y - r, 2 * r, 2 * r);
        g.restore();
      }
    }
    g.lineCap = 'round';
    for (let i = 0; i < veins; i++) {
      let x = rnd() * s, y = rnd() * s;
      let ang = rnd() * Math.PI * 2;
      const pts = [[x, y]];
      for (let k = 0; k < 30; k++) {
        ang += (rnd() - 0.5) * 0.7;
        x += Math.cos(ang) * 18;
        y += Math.sin(ang) * 18;
        pts.push([x, y]);
      }
      const w = 0.6 + rnd() * 1.6;
      const a = alpha * (0.4 + rnd() * 0.8);
      for (const ox of [-s, 0, s]) for (const oy of [-s, 0, s]) {
        g.beginPath();
        g.moveTo(pts[0][0] + ox, pts[0][1] + oy);
        for (const [px, py] of pts.slice(1)) g.lineTo(px + ox, py + oy);
        g.strokeStyle = `rgba(146,146,150,${a})`;
        g.lineWidth = w;
        g.stroke();
        g.strokeStyle = `rgba(146,146,150,${a * 0.35})`;
        g.lineWidth = w * 4;
        g.stroke();
      }
    }
  });
}

const TONE = {
  stair: '#fbfbfa', bridge: '#fbfbfa', stone: '#f7f7f6', ramp: '#fbfbfa', plateau: '#fbfbfa',
  curb: '#fbfbfa', block: '#f5f5f4', box: '#f9f9f8',
};
const FLOOR = { ground: '#e6e6e6', verge: '#e6e6e6', sunken: '#e2e2e2' };

function courseMeshes(course, tex) {
  const groups = { marble: { pos: [], nrm: [], col: [], uv: [], scale: 3 }, floor: { pos: [], nrm: [], col: [], uv: [], scale: 6 } };
  let seed = 11;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const quad = (g, p, n, c) => {
    const k = g.scale;
    for (const i of [0, 1, 2, 0, 2, 3]) {
      const q = p[i];
      g.pos.push(...q);
      g.nrm.push(...n);
      g.col.push(c.r, c.g, c.b);
      if (Math.abs(n[2]) > 0.5) g.uv.push(q[0] / k, q[1] / k);
      else if (Math.abs(n[0]) > 0.5) g.uv.push(q[1] / k + 0.37, q[2] / k);
      else g.uv.push(q[0] / k + 0.61, q[2] / k);
    }
  };
  const gaps = [];
  for (const pc of course.meta.pieces) {
    if (pc.k === 'gap') {
      gaps.push(pc);
      continue;
    }
    const tone = TONE[pc.k] || FLOOR[pc.k] || FLOOR.ground;
    const g = TONE[pc.k] ? groups.marble : groups.floor;
    const [x0, x1, y0, y1] = pc.r;
    const [a, b, c] = pc.p;
    const z = (x, y) => a * x + b * y + c;
    const top = new THREE.Color(tone);
    if (pc.k === 'stone' || pc.k === 'block') top.multiplyScalar(0.985 + 0.03 * rnd());
    const side = top.clone().multiplyScalar(0.93);
    const nl = Math.hypot(a, b, 1);
    quad(g, [[x0, y0, z(x0, y0)], [x1, y0, z(x1, y0)], [x1, y1, z(x1, y1)], [x0, y1, z(x0, y1)]], [-a / nl, -b / nl, 1 / nl], top);
    quad(g, [[x0, y0, 0], [x1, y0, 0], [x1, y0, z(x1, y0)], [x0, y0, z(x0, y0)]], [0, -1, 0], side);
    quad(g, [[x1, y0, 0], [x1, y1, 0], [x1, y1, z(x1, y1)], [x1, y0, z(x1, y0)]], [1, 0, 0], side);
    quad(g, [[x1, y1, 0], [x0, y1, 0], [x0, y1, z(x0, y1)], [x1, y1, z(x1, y1)]], [0, 1, 0], side);
    quad(g, [[x0, y1, 0], [x0, y0, 0], [x0, y0, z(x0, y0)], [x0, y1, z(x0, y1)]], [-1, 0, 0], side);
  }
  const out = [];
  const mats = {
    marble: new THREE.MeshStandardMaterial({ vertexColors: true, map: tex.marble, roughness: 0.35, metalness: 0, envMapIntensity: 0.3 }),
    floor: new THREE.MeshStandardMaterial({ vertexColors: true, map: tex.floor, roughness: 0.6, envMapIntensity: 0.6 }),
  };
  for (const [name, g] of Object.entries(groups)) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(g.pos, 3));
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(g.nrm, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(g.col, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(g.uv, 2));
    const mesh = new THREE.Mesh(geo, mats[name]);
    mesh.receiveShadow = true;
    mesh.castShadow = true;
    out.push(mesh);
  }
  const gapMat = new THREE.MeshStandardMaterial({ color: '#c8c8ca', roughness: 0.9, envMapIntensity: 0.2 });
  for (const pc of gaps) {
    const [x0, x1, y0, y1] = pc.r;
    const m = new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0, y1 - y0), gapMat);
    m.position.set((x0 + x1) / 2, (y0 + y1) / 2, pc.p[2] + 0.002);
    m.receiveShadow = true;
    out.push(m);
  }
  return out;
}

function buildScene(model, course, canvas, stage) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  const maxRatio = Math.min(2, window.devicePixelRatio || 1);
  let ratio = maxRatio;
  renderer.setPixelRatio(ratio);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.toneMappingExposure = 0.82;

  const scene = new THREE.Scene();
  const sky = '#ebebec';
  scene.background = new THREE.Color(sky);
  scene.fog = new THREE.Fog(sky, 14, 40);
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envTex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.45;
  scene.environment = envTex;
  pmrem.dispose();

  const camera = new THREE.PerspectiveCamera(42, 16 / 9, 0.05, 120);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 1.5;
  controls.maxDistance = 14;
  controls.maxPolarAngle = Math.PI * 0.47;
  controls.enablePan = false;

  scene.add(new THREE.HemisphereLight('#ffffff', '#d2d2d4', 0.35));
  const sun = new THREE.DirectionalLight('#ffffff', 2.9);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = sun.shadow.camera.bottom = -7;
  sun.shadow.camera.right = sun.shadow.camera.top = 7;
  sun.shadow.camera.near = 0.5;
  sun.shadow.camera.far = 30;
  sun.shadow.bias = -0.0005;
  sun.shadow.normalBias = 0.03;
  scene.add(sun, sun.target);

  const textures = { marble: marbleTexture(3, 18, 0.15), floor: marbleTexture(9, 0, 0) };
  textures.floor.repeat.set(1, 1);
  for (const m of courseMeshes(course, textures)) scene.add(m);
  const [SX, SY] = course.meta.size;
  const G = course.ground;
  const F = 200;
  const outerGeos = [[-F, SX + F, -F, 0], [-F, SX + F, SY, SY + F], [-F, 0, 0, SY], [SX, SX + F, 0, SY]].map(([x0, x1, y0, y1]) => {
    const g = new THREE.PlaneGeometry(x1 - x0, y1 - y0).translate((x0 + x1) / 2, (y0 + y1) / 2, G);
    const uv = g.attributes.uv, p = g.attributes.position;
    for (let k = 0; k < uv.count; k++) uv.setXY(k, p.getX(k) / 6, p.getY(k) / 6);
    return g;
  });
  const floor = new THREE.Mesh(mergeGeometries(outerGeos), new THREE.MeshStandardMaterial({ color: '#e6e6e6', map: textures.floor, roughness: 0.6, envMapIntensity: 0.6 }));
  floor.receiveShadow = true;
  scene.add(floor);

  const geoms = [];
  const vert = model.mesh_vert, face = model.mesh_face;
  const vadr = model.mesh_vertadr, vnum = model.mesh_vertnum, fadr = model.mesh_faceadr, fnum = model.mesh_facenum;
  const matCache = new Map();
  for (let g = 0; g < model.ngeom; g++) {
    if (model.geom_group[g] !== VISUAL_GROUP || model.geom_type[g] !== 7) continue;
    const m = model.geom_dataid[g];
    const v0 = vadr[m];
    let geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vert.subarray(3 * v0, 3 * (v0 + vnum[m]))), 3));
    geo.setIndex(new THREE.BufferAttribute(new Uint32Array(face.subarray(3 * fadr[m], 3 * (fadr[m] + fnum[m]))), 1));
    geo = toCreasedNormals(geo, (35 * Math.PI) / 180);
    const mid = model.geom_matid[g];
    const rgba = mid >= 0 ? model.mat_rgba.subarray(4 * mid, 4 * mid + 4) : model.geom_rgba.subarray(4 * g, 4 * g + 4);
    const dark = rgba[0] < 0.4;
    const key = dark ? 'dark' : 'light';
    if (!matCache.has(key)) {
      matCache.set(key, new THREE.MeshStandardMaterial({
        color: dark ? new THREE.Color(0.16, 0.16, 0.19) : new THREE.Color(0.88, 0.88, 0.91),
        roughness: dark ? 0.55 : 0.4,
        metalness: dark ? 0.1 : 0.05,
      }));
    }
    const mesh = new THREE.Mesh(geo, matCache.get(key));
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.matrixAutoUpdate = false;
    mesh.userData.geom = g;
    scene.add(mesh);
    geoms.push(mesh);
  }

  const stepMat = new THREE.MeshBasicMaterial({ color: '#7c3aed', transparent: true, opacity: 0.55, depthWrite: false });
  const stepGeo = new THREE.CircleGeometry(0.055, 24);
  const ringGeo = new THREE.RingGeometry(0.075, 0.09, 32);
  const steps = [];
  for (let i = 0; i < 4; i++) {
    const g = new THREE.Group();
    const dot = new THREE.Mesh(stepGeo, stepMat.clone());
    const ring = new THREE.Mesh(ringGeo, stepMat.clone());
    g.add(dot, ring);
    g.renderOrder = 3;
    scene.add(g);
    steps.push(g);
  }

  const resize = () => {
    const w = stage.clientWidth, h = stage.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  const ro = new ResizeObserver(resize);
  ro.observe(stage);
  resize();

  const follow = new THREE.Vector3();
  let lastUser = -1e9;
  controls.addEventListener('start', () => (lastUser = performance.now()));
  controls.addEventListener('end', () => (lastUser = performance.now()));
  let userActive = false;
  controls.addEventListener('start', () => (userActive = true));
  controls.addEventListener('end', () => (userActive = false));
  const m4 = new THREE.Matrix4();
  const spherical = new THREE.Spherical();
  const offset = new THREE.Vector3();
  const zUp = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 1, 0));
  const zUpInv = zUp.clone().invert();

  let frameEma = 1 / 60;
  let sinceChange = 0;
  const fixedRatio = new URLSearchParams(location.search).has('debug');
  const view = {
    renderer,
    scene,
    camera,
    controls,
    chase: true,
    adapt(dt) {
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
    },
    snapCamera(d, yaw) {
      const p = new THREE.Vector3(d.qpos[0], d.qpos[1], d.qpos[2] - 0.05);
      follow.copy(p);
      controls.target.copy(p);
      camera.position.set(p.x - 3.6 * Math.cos(yaw), p.y - 3.6 * Math.sin(yaw), p.z + 1.5);
      controls.update();
    },
    render(d, ctrl, showSteps, dt) {
      const xpos = d.geom_xpos, xmat = d.geom_xmat;
      for (const mesh of geoms) {
        const g = mesh.userData.geom;
        const R = xmat.subarray(9 * g, 9 * g + 9);
        m4.set(R[0], R[1], R[2], xpos[3 * g], R[3], R[4], R[5], xpos[3 * g + 1], R[6], R[7], R[8], xpos[3 * g + 2], 0, 0, 0, 1);
        mesh.matrix.copy(m4);
        mesh.matrixWorldNeedsUpdate = true;
      }
      const target = new THREE.Vector3(d.qpos[0], d.qpos[1], d.qpos[2] - 0.05);
      const a = 1 - Math.exp(-dt * 6);
      const delta = new THREE.Vector3((target.x - follow.x) * a, (target.y - follow.y) * a, (target.z - follow.z) * (1 - Math.exp(-dt * 3)));
      follow.add(delta);
      controls.target.add(delta);
      camera.position.add(delta);
      if (this.chase && !userActive && performance.now() - lastUser > 1500) {
        offset.copy(camera.position).sub(controls.target).applyQuaternion(zUp);
        spherical.setFromVector3(offset);
        const heading = ctrl.heading();
        const want = Math.atan2(-Math.cos(heading), Math.sin(heading));
        let dth = want - spherical.theta;
        dth = Math.atan2(Math.sin(dth), Math.cos(dth));
        spherical.theta += dth * (1 - Math.exp(-dt * 1.6));
        offset.setFromSpherical(spherical).applyQuaternion(zUpInv);
        camera.position.copy(controls.target).add(offset);
      }
      sun.position.set(follow.x + 5, follow.y - 6, follow.z + 7);
      sun.target.position.set(follow.x, follow.y, follow.z - 0.7);
      controls.update();

      const mk = ctrl.refMarkers();
      steps.forEach((s, i) => {
        const f = mk.feet[i + 1];
        s.visible = showSteps && !!f;
        if (!s.visible) return;
        s.position.set(f[0], f[1], f[2] + 0.012);
        const o = 0.7 - 0.15 * i;
        s.children[0].material.opacity = o * 0.8;
        s.children[1].material.opacity = o;
      });
      renderer.render(scene, camera);
    },
    dispose() {
      ro.disconnect();
      controls.dispose();
      scene.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose());
      });
      Object.values(textures).forEach((t) => t.dispose());
      envTex.dispose();
      renderer.dispose();
    },
  };
  return view;
}

function renderInfo(info) {
  info.innerHTML = `
    <div class="cg-info">
      <h2>Perceptive Locomotion</h2>
      <p>Walk a simulated Unitree G1 around a circuit of stairs, stepping stones, slopes, curbs and blocks. Based on
      Terrain Consistent Reference-Guided RL.</p>
      <ul class="cg-refs">${REFERENCES.map(refItem).join('')}</ul>
    </div>`;
}

const REFERENCES = [
  {
    title: 'Terrain Consistent Reference-Guided RL for Humanoid Navigation Autonomy',
    authors: 'William D. Compton, Zachary Olkin, Aaron D. Ames',
    year: 2026,
    arxiv: '2605.15517',
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
