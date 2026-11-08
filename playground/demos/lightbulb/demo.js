// Interactive demo: a Franka arm with a five-finger hand screws a light bulb into its socket while you get in its way.
// The physics and the policy run in a worker (sim_worker.js); this module draws the scene and handles input.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

const BASE = new URL('./', import.meta.url);
const ASSETS = new URL('assets/', BASE);
const RENDER_DELAY = 0.05; // s behind the newest pose, so there is always a pose on each side to interpolate
const AUTO_NEXT_S = 4; // seated this long -> next round
const DEBUG = new URLSearchParams(location.search).has('debug');

const ICONS = {
  play: '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M4 2.5v11l9-5.5z"/></svg>',
  pause: '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z"/></svg>',
  reset: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M2.5 8a5.5 5.5 0 1 0 1.7-4"/><path d="M2.5 2.5v3h3"/></svg>',
  knock: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2 8h8M7 4.5 10.5 8 7 11.5"/><circle cx="13" cy="8" r="1.8"/></svg>',
  open: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M4 9V4.5M6.5 8V3M9 8V3.5M11.5 9V5"/><path d="M3 9.5c0 3 2 4.5 5 4.5s4.5-2 4.5-4.5"/></svg>',
  move: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M8 2v12M2 8h12M8 2 6 4M8 2l2 2M8 14l-2-2M8 14l2-2M2 8l2-2M2 8l2 2M14 8l-2-2M14 8l-2 2"/></svg>',
};

const KINDS = [
  { id: 'table', label: 'On the table' },
  { id: 'grasp', label: 'Mid-grasp' },
  { id: 'air', label: 'In hand' },
  { id: 'socket', label: 'In socket' },
];

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

export async function mount(root, info) {
  loadCss(new URL('../common.css', BASE).href);
  loadCss(new URL('demo.css', BASE).href);
  const wrap = el('div', 'cg lb');
  const stage = el('div', 'cg-stage');
  const canvas = el('canvas');
  canvas.tabIndex = 0;
  canvas.setAttribute('aria-label', 'Robot simulation. Drag the bulb or the robot to push them; drag elsewhere to orbit the camera.');
  stage.appendChild(canvas);
  const loading = el('div', 'cg-loading', '<div class="track"><div class="fill"></div></div><div class="msg">Loading…</div>');
  stage.appendChild(loading);
  const fill = loading.querySelector('.fill');
  const msg = loading.querySelector('.msg');

  const panel = el('div', 'cg-panel');
  panel.innerHTML = `
    <div>
      <div class="cg-label">Start</div>
      <div class="cg-motions lb-starts"></div>
    </div>
    <div>
      <div class="cg-label">Get in the way</div>
      <div class="lb-perturb">
        <button class="cg-btn" data-act="knock">${ICONS.knock}<span>Knock the bulb</span></button>
        <button class="cg-btn" data-act="open">${ICONS.open}<span>Open the hand</span></button>
        <button class="cg-btn" data-act="move">${ICONS.move}<span>Move the bulb</span></button>
      </div>
    </div>
    <div class="cg-actions lb-actions">
      <button class="cg-btn" data-act="pause">${ICONS.pause}<span>Pause</span></button>
      <button class="cg-btn" data-act="reset">${ICONS.reset}<span>Restart</span></button>
    </div>`;
  const status = el('div', 'lb-status', '<div class="lb-phase">Starting…</div><div class="bar"><i></i></div>');
  const hint = el('div', 'cg-hint', '<b>Drag the bulb or the hand</b><span class="more"> · drag elsewhere to orbit · scroll to zoom</span>');
  const forceBox = el('div', 'cg-force lb-force', 'Force <span>0 N</span>');
  const toast = el('div', 'cg-toast');
  stage.append(status, hint, forceBox, toast);
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

  const worker = new Worker(new URL('sim_worker.js', BASE), { type: 'module' });
  cleanups.push(() => worker.terminate());

  // ── scene state fed by the worker ──
  let view = null;
  let starts = [];
  let current = { kind: 'table', index: null };
  const frames = []; // [{t, epoch, xpos, xquat, ...}]
  let renderT = 0;
  let lastFrame = null;
  let seatedSince = null;
  let paused = false;
  let drag = null;
  // Rounds that stop making progress (it happens: the last few degrees of the thread) move on by themselves.
  let progressMark = { best: 0, at: null };
  let stuckWarned = false;
  let lit = false;

  let toastTimer = 0;
  function showToast(text) {
    toast.textContent = text;
    toast.classList.add('on');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('on'), 2200);
  }

  function pickStart(kind) {
    const idx = starts.map((k, i) => [k, i]).filter(([k]) => k === kind).map(([, i]) => i);
    return idx[Math.floor(Math.random() * idx.length)];
  }

  function start(kind, index) {
    current = { kind, index: index ?? pickStart(kind) };
    seatedSince = null;
    progressMark = { best: 0, at: null };
    stuckWarned = false;
    panel.querySelectorAll('.lb-starts .cg-btn').forEach((b) => b.classList.toggle('active', b.dataset.kind === kind));
    worker.postMessage({ type: 'start', index: current.index });
  }

  worker.onmessage = async (e) => {
    const m = e.data;
    if (disposed) return;
    if (m.type === 'progress') {
      fill.style.width = `${(100 * m.frac).toFixed(1)}%`;
      if (m.msg) msg.textContent = m.msg;
    } else if (m.type === 'scene') {
      starts = m.starts;
      msg.textContent = 'Building the scene…';
      view = await buildScene(m, canvas, stage);
      cleanups.push(() => view.dispose());
      setupInput();
      loading.classList.add('done');
      worker.postMessage({ type: 'ready' });
    } else if (m.type === 'pose') {
      if (lastFrame && m.epoch !== lastFrame.epoch) frames.length = 0;
      frames.push(m);
      if (DEBUG) window.__lbLast = { t: m.t, epoch: m.epoch, stepMs: m.stepMs, thread: m.thread, wall: performance.now() };
      if (frames.length > 40) frames.shift();
      lastFrame = m;
      if (frames.length === 1) renderT = m.t;
    } else if (m.type === 'started') {
      current = { kind: m.kind, index: m.index };
      panel.querySelectorAll('.lb-starts .cg-btn').forEach((b) => b.classList.toggle('active', b.dataset.kind === m.kind));
      progressMark = { best: 0, at: null };
      stuckWarned = false;
    } else if (m.type === 'toast') showToast(m.text);
    else if (m.type === 'error') {
      console.error(m.message);
      loading.classList.remove('done');
      loading.innerHTML = `<div class="err">The demo could not start in this browser.<br><small>${m.message}</small></div>`;
    }
  };
  worker.onerror = (e) => {
    loading.innerHTML = `<div class="err">The demo could not start in this browser.<br><small>${e.message || 'worker error'}</small></div>`;
  };

  // start buttons
  const startBox = panel.querySelector('.lb-starts');
  for (const k of KINDS) {
    const b = el('button', 'cg-btn', k.label);
    b.dataset.kind = k.id;
    b.addEventListener('click', () => start(k.id));
    startBox.appendChild(b);
  }
  startBox.querySelector('[data-kind=table]').classList.add('active');
  const pauseBtn = panel.querySelector('[data-act=pause]');
  function setPaused(p) {
    paused = p;
    worker.postMessage({ type: 'pause', value: p });
    pauseBtn.innerHTML = p ? `${ICONS.play}<span>Play</span>` : `${ICONS.pause}<span>Pause</span>`;
  }
  pauseBtn.addEventListener('click', () => setPaused(!paused));
  panel.querySelector('[data-act=reset]').addEventListener('click', () => start(current.kind, current.index));
  for (const act of ['knock', 'open', 'move']) {
    panel.querySelector(`[data-act=${act}]`).addEventListener('click', () => {
      seatedSince = null;
      progressMark = { best: 0, at: null };
      stuckWarned = false;
      worker.postMessage({ type: act });
    });
  }

  // pause the simulation while the stage is off screen or the tab is hidden
  let visible = true;
  const io = new IntersectionObserver(([en]) => {
    visible = en.isIntersecting;
    if (!paused) worker.postMessage({ type: 'pause', value: !visible });
  }, { threshold: 0.05 });
  io.observe(stage);
  const onVis = () => {
    if (!paused) worker.postMessage({ type: 'pause', value: document.hidden || !visible });
  };
  document.addEventListener('visibilitychange', onVis);
  cleanups.push(() => io.disconnect(), () => document.removeEventListener('visibilitychange', onVis));

  // ── input: drag a body with a spring ──
  function setupInput() {
    const raycaster = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    const plane = new THREE.Plane();
    const tmp = new THREE.Vector3();
    const toNdc = (ev) => {
      const r = canvas.getBoundingClientRect();
      ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    };
    const pick = (ev) => {
      toNdc(ev);
      raycaster.setFromCamera(ndc, view.camera);
      return raycaster.intersectObjects(view.pickables, false)[0];
    };
    canvas.addEventListener('pointerdown', (ev) => {
      if (ev.button !== 0 && ev.pointerType === 'mouse') return;
      const hit = pick(ev);
      if (!hit) return;
      const group = hit.object.parent;
      const body = group.userData.body;
      const local = group.worldToLocal(hit.point.clone());
      view.camera.getWorldDirection(tmp);
      plane.setFromNormalAndCoplanarPoint(tmp.clone().negate(), hit.point);
      drag = { body };
      worker.postMessage({ type: 'drag', body, local: local.toArray(), target: hit.point.toArray() });
      view.controls.enabled = false;
      canvas.setPointerCapture(ev.pointerId);
      canvas.classList.add('grabbing');
      hint.style.opacity = '0';
      seatedSince = null;
      ev.preventDefault();
      ev.stopImmediatePropagation();
    }, { capture: true });
    canvas.addEventListener('pointermove', (ev) => {
      if (!drag) {
        if (ev.pointerType === 'mouse') canvas.classList.toggle('grab', !!pick(ev));
        return;
      }
      toNdc(ev);
      raycaster.setFromCamera(ndc, view.camera);
      if (raycaster.ray.intersectPlane(plane, tmp)) {
        worker.postMessage({ type: 'dragTo', target: [tmp.x, tmp.y, Math.max(0.005, tmp.z)] });
      }
    });
    const end = (ev) => {
      if (!drag) return;
      drag = null;
      worker.postMessage({ type: 'dragEnd' });
      view.controls.enabled = true;
      canvas.classList.remove('grabbing');
      try {
        canvas.releasePointerCapture(ev.pointerId);
      } catch (err) {
        /* not captured */
      }
    };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
  }

  // ── render loop ──
  let last = performance.now();
  let raf = 0;
  const phaseEl = status.querySelector('.lb-phase');
  const barEl = status.querySelector('.bar i');
  function frame(now) {
    if (disposed) return;
    raf = requestAnimationFrame(frame);
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (!view || !frames.length || !visible) return;
    const newest = frames[frames.length - 1];
    if (!paused && !document.hidden) renderT += dt;
    renderT = Math.min(Math.max(renderT, newest.t - 0.25), newest.t - (paused ? 0 : RENDER_DELAY));
    if (paused) renderT = newest.t;
    let a = frames[0];
    let b = newest;
    for (let i = frames.length - 1; i > 0; i--) {
      if (frames[i - 1].t <= renderT) {
        a = frames[i - 1];
        b = frames[i];
        break;
      }
    }
    const w = b.t > a.t ? Math.min(1, Math.max(0, (renderT - a.t) / (b.t - a.t))) : 1;
    view.setPoses(a, b, w);
    const th = newest.thread;
    // Once seated, the light stays on while the bulb stays screwed home (the thread friction lets it settle back a
    // few degrees after the last stroke).
    lit = th.seated || (lit && th.engaged && th.progress > 0.95);
    view.setLit(lit);
    view.setDrag(newest.drag);
    // status
    let phase;
    if (lit) phase = 'Screwed in. Light on!';
    else if (th.engaged) phase = `Screwing in · ${Math.round(100 * th.progress)}%`;
    else phase = 'Picking up the bulb';
    phaseEl.textContent = phase;
    barEl.style.width = `${(100 * (lit ? 1 : th.engaged ? th.progress : 0)).toFixed(1)}%`;
    status.classList.toggle('lit', lit);
    const F = newest.drag ? newest.drag.force : 0;
    forceBox.classList.toggle('on', !!newest.drag);
    forceBox.querySelector('span').textContent = `${F.toFixed(1)} N`;
    // next round after a while seated
    if (lit && !paused) {
      if (seatedSince === null) seatedSince = newest.t;
      else if (newest.t - seatedSince > AUTO_NEXT_S) {
        showToast('Done. Here comes another bulb');
        start('table');
      }
    } else seatedSince = null;
    // no progress for a while: suggest a nudge, then move on
    if (!lit && !paused && !drag) {
      const p = th.engaged ? th.progress : 0;
      if (progressMark.at === null || p > progressMark.best + 0.02 || newest.t < progressMark.at) progressMark = { best: Math.max(p, progressMark.best), at: newest.t };
      const idle = newest.t - progressMark.at;
      if (idle > 15 && !stuckWarned) {
        stuckWarned = true;
        showToast(th.engaged ? 'Stuck on the last turn. Try a nudge, or Restart' : 'Stuck. Try a nudge, or Restart');
      } else if (idle > 30) {
        showToast('Moving on to another bulb');
        start('table');
      }
    }
    view.render();
  }
  raf = requestAnimationFrame(frame);
  cleanups.unshift(() => cancelAnimationFrame(raf));

  // the first round: a bulb on the table
  worker.postMessage({ type: 'init', kind: 'table' });
  return api;
}

async function buildScene(sc, canvas, stage) {
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
  scene.fog = new THREE.Fog('#f3f3f5', 4, 12);
  const camera = new THREE.PerspectiveCamera(38, 16 / 9, 0.02, 40);
  camera.up.set(0, 0, 1);
  camera.position.set(1.02, -0.52, 0.44);
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0.53, -0.08, 0.11);
  if (DEBUG) window.__lbView = { camera, controls, scene };
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.35;
  controls.maxDistance = 3.5;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.update();

  scene.add(new THREE.HemisphereLight('#ffffff', '#d4d0de', 1.25));
  const sun = new THREE.DirectionalLight('#ffffff', 2.0);
  sun.position.set(1.6, -1.2, 2.4);
  sun.target.position.set(0.45, 0, 0);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  Object.assign(sun.shadow.camera, { left: -0.9, right: 0.9, top: 0.9, bottom: -0.9, near: 0.5, far: 6 });
  sun.shadow.bias = -0.0005;
  sun.shadow.normalBias = 0.01;
  scene.add(sun, sun.target);
  const rim = new THREE.DirectionalLight('#e9ddff', 0.5);
  rim.position.set(-1, 1.5, 1.2);
  scene.add(rim);

  // floor
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(30, 30), new THREE.MeshStandardMaterial({ color: '#e9e9ee', roughness: 1 }));
  floor.position.z = -0.76;
  floor.receiveShadow = true;
  scene.add(floor);

  // one group per body; geoms are its children at their pose in the body frame
  const groups = sc.bodies.map((n, b) => {
    const g = new THREE.Group();
    g.userData.body = b;
    g.userData.name = n;
    g.matrixAutoUpdate = true;
    scene.add(g);
    return g;
  });
  const pickables = [];
  const materials = new Map();
  const mat = (key, make) => {
    if (!materials.has(key)) materials.set(key, make());
    return materials.get(key);
  };
  const robotMat = (rgba) => {
    const dark = rgba[0] < 0.4;
    return mat(dark ? 'robot-dark' : 'robot-light', () => new THREE.MeshStandardMaterial({
      color: dark ? new THREE.Color(0.17, 0.17, 0.2) : new THREE.Color(0.9, 0.9, 0.93),
      roughness: dark ? 0.5 : 0.38,
      metalness: 0.05,
    }));
  };
  const handMat = (rgba) => mat(`hand-${rgba.join(',')}`, () => {
    const c = new THREE.Color(rgba[0], rgba[1], rgba[2]);
    return new THREE.MeshStandardMaterial({ color: c.lerp(new THREE.Color(0.86, 0.87, 0.92), 0.35), roughness: 0.45, metalness: 0.05 });
  });
  const bulbMat = new THREE.MeshPhysicalMaterial({
    color: '#f7f1df', roughness: 0.18, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.1,
    emissive: '#ffcf6b', emissiveIntensity: 0,
  });
  const baseMat = new THREE.MeshStandardMaterial({ color: '#4a4c55', roughness: 0.4, metalness: 0.35 });
  const tableMat = new THREE.MeshStandardMaterial({ color: '#3a3b41', roughness: 0.85 });

  const crease = (geo) => creasedNormals(geo, (35 * Math.PI) / 180);
  for (const gm of sc.geoms) {
    let mesh = null;
    if (gm.vert) {
      let geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(gm.vert, 3));
      geo.setIndex(new THREE.BufferAttribute(gm.face, 1));
      mesh = new THREE.Mesh(crease(geo), robotMat(gm.rgba));
    } else if (gm.bodyName === 'table') {
      const [x, y, z] = gm.size;
      mesh = new THREE.Mesh(new THREE.BoxGeometry(2 * x, 2 * y, 2 * z), tableMat);
    } else continue;
    mesh.position.set(...gm.pos);
    mesh.quaternion.set(gm.quat[1], gm.quat[2], gm.quat[3], gm.quat[0]);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    groups[gm.body].add(mesh);
    if (gm.bodyName !== 'table') pickables.push(mesh);
  }

  // visual meshes for the hand and the lamp parts, in their body frames
  const manifest = await (await fetch(new URL('visual/manifest.json', ASSETS))).json();
  const loader = new STLLoader();
  const byName = new Map(sc.bodies.map((n, b) => [n, b]));
  await Promise.all(manifest.map(async (v) => {
    const buf = await (await fetch(new URL('visual/' + v.file, ASSETS))).arrayBuffer();
    let geo = loader.parse(buf);
    geo.deleteAttribute('normal');
    geo = crease(geo);
    const isBulb = v.body.startsWith('lamp_bulb/');
    const isBase = v.body.startsWith('lamp_base/');
    const m = new THREE.Mesh(geo, isBulb ? bulbMat : isBase ? baseMat : handMat(v.rgba));
    m.castShadow = true;
    m.receiveShadow = true;
    groups[byName.get(v.body)].add(m);
    if (!isBase) pickables.push(m);
  }));

  // the bulb's light, on when it is seated
  const bulbGroup = groups[sc.bulbBody];
  const glow = new THREE.PointLight('#ffcf6b', 0, 1.2, 1.6);
  glow.position.set(0, 0, sc.globeZ);
  bulbGroup.add(glow);
  const haloTex = (() => {
    const c = document.createElement('canvas');
    c.width = c.height = 128;
    const g = c.getContext('2d');
    const grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    grd.addColorStop(0, 'rgba(255,214,120,0.85)');
    grd.addColorStop(0.35, 'rgba(255,200,90,0.35)');
    grd.addColorStop(1, 'rgba(255,190,80,0)');
    g.fillStyle = grd;
    g.fillRect(0, 0, 128, 128);
    return new THREE.CanvasTexture(c);
  })();
  const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: haloTex, transparent: true, depthWrite: false, opacity: 0 }));
  halo.scale.set(0.22, 0.22, 1);
  halo.position.set(0, 0, sc.globeZ);
  bulbGroup.add(halo);
  let litLevel = 0;
  let litTarget = 0;

  // drag indicator
  const lineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  const line = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color: '#db2777', depthTest: false }));
  const dotA = new THREE.Mesh(new THREE.SphereGeometry(0.006, 12, 8), new THREE.MeshBasicMaterial({ color: '#db2777', depthTest: false }));
  const dotB = new THREE.Mesh(new THREE.SphereGeometry(0.009, 12, 8), new THREE.MeshBasicMaterial({ color: '#7c3aed', depthTest: false }));
  line.renderOrder = dotA.renderOrder = dotB.renderOrder = 10;
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

  // adaptive resolution
  let frameEma = 1 / 60;
  let sinceChange = 0;
  let lastT = performance.now();
  const qa = new THREE.Quaternion();
  const qb = new THREE.Quaternion();
  return {
    camera,
    controls,
    pickables,
    setPoses(a, b, w) {
      for (let i = 0; i < groups.length; i++) {
        const g = groups[i];
        g.position.set(
          a.xpos[3 * i] + w * (b.xpos[3 * i] - a.xpos[3 * i]),
          a.xpos[3 * i + 1] + w * (b.xpos[3 * i + 1] - a.xpos[3 * i + 1]),
          a.xpos[3 * i + 2] + w * (b.xpos[3 * i + 2] - a.xpos[3 * i + 2])
        );
        qa.set(a.xquat[4 * i + 1], a.xquat[4 * i + 2], a.xquat[4 * i + 3], a.xquat[4 * i]);
        qb.set(b.xquat[4 * i + 1], b.xquat[4 * i + 2], b.xquat[4 * i + 3], b.xquat[4 * i]);
        g.quaternion.slerpQuaternions(qa, qb, w);
      }
    },
    setLit(on) {
      litTarget = on ? 1 : 0;
    },
    setDrag(d) {
      if (d && d.anchor) {
        lineGeo.setFromPoints([new THREE.Vector3(...d.anchor), new THREE.Vector3(...d.target)]);
        dotA.position.set(...d.anchor);
        dotB.position.set(...d.target);
        dragGroup.visible = true;
      } else dragGroup.visible = false;
    },
    render() {
      const now = performance.now();
      const dt = Math.min(0.1, (now - lastT) / 1000);
      lastT = now;
      litLevel += (litTarget - litLevel) * Math.min(1, dt * 4);
      bulbMat.emissiveIntensity = 1.6 * litLevel;
      glow.intensity = 1.2 * litLevel;
      halo.material.opacity = litLevel;
      controls.update();
      renderer.render(scene, camera);
      frameEma += 0.1 * (dt - frameEma);
      sinceChange += dt;
      if (sinceChange > 1) {
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
    },
    dispose() {
      ro.disconnect();
      controls.dispose();
      scene.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) o.material.dispose();
      });
      haloTex.dispose();
      renderer.dispose();
    },
  };
}

// Smooth normals that keep hard edges: each corner averages the normals of the faces around its vertex that lie
// within `creaseAngle` of its own face, weighted by the corner angle (stable on CAD meshes full of thin slivers).
function creasedNormals(geometry, creaseAngle) {
  const src = geometry.index ? geometry.toNonIndexed() : geometry;
  const pos = src.getAttribute('position').array;
  const nTri = pos.length / 9;
  const fn = new Float32Array(nTri * 3);
  const wt = new Float32Array(nTri * 3);
  const keyOf = (i) => `${Math.round(pos[3 * i] * 1e5)},${Math.round(pos[3 * i + 1] * 1e5)},${Math.round(pos[3 * i + 2] * 1e5)}`;
  const shared = new Map();
  const corner = (a, b, c, k) => {
    const ux = pos[3 * b] - pos[3 * a], uy = pos[3 * b + 1] - pos[3 * a + 1], uz = pos[3 * b + 2] - pos[3 * a + 2];
    const vx = pos[3 * c] - pos[3 * a], vy = pos[3 * c + 1] - pos[3 * a + 1], vz = pos[3 * c + 2] - pos[3 * a + 2];
    const lu = Math.hypot(ux, uy, uz), lv = Math.hypot(vx, vy, vz);
    wt[k] = lu && lv ? Math.acos(Math.max(-1, Math.min(1, (ux * vx + uy * vy + uz * vz) / (lu * lv)))) : 0;
  };
  for (let t = 0; t < nTri; t++) {
    const a = 3 * t, b = a + 1, c = a + 2;
    const ux = pos[3 * b] - pos[3 * a], uy = pos[3 * b + 1] - pos[3 * a + 1], uz = pos[3 * b + 2] - pos[3 * a + 2];
    const vx = pos[3 * c] - pos[3 * a], vy = pos[3 * c + 1] - pos[3 * a + 1], vz = pos[3 * c + 2] - pos[3 * a + 2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.hypot(nx, ny, nz) || 1;
    fn[3 * t] = nx / l; fn[3 * t + 1] = ny / l; fn[3 * t + 2] = nz / l;
    corner(a, b, c, a); corner(b, c, a, b); corner(c, a, b, c);
    for (const i of [a, b, c]) {
      const k = keyOf(i);
      let list = shared.get(k);
      if (!list) shared.set(k, (list = []));
      list.push(i);
    }
  }
  const cosLim = Math.cos(creaseAngle);
  const out = new Float32Array(pos.length);
  for (const list of shared.values()) {
    for (const i of list) {
      const t = Math.floor(i / 3);
      let sx = 0, sy = 0, sz = 0;
      for (const j of list) {
        const u = Math.floor(j / 3);
        if (fn[3 * t] * fn[3 * u] + fn[3 * t + 1] * fn[3 * u + 1] + fn[3 * t + 2] * fn[3 * u + 2] < cosLim) continue;
        sx += wt[j] * fn[3 * u]; sy += wt[j] * fn[3 * u + 1]; sz += wt[j] * fn[3 * u + 2];
      }
      const l = Math.hypot(sx, sy, sz);
      if (l > 0) { out[3 * i] = sx / l; out[3 * i + 1] = sy / l; out[3 * i + 2] = sz / l; }
      else { out[3 * i] = fn[3 * t]; out[3 * i + 1] = fn[3 * t + 1]; out[3 * i + 2] = fn[3 * t + 2]; }
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(out, 3));
  return geo;
}

function renderInfo(info) {
  info.innerHTML = `
    <div class="cg-info">
      <h2>Dexterous light-bulb assembly</h2>
      <div class="cg-section-title">Credits</div>
      <p class="cg-credits">Physics by <a href="https://mujoco.org" target="_blank" rel="noopener">MuJoCo 4</a> (WebAssembly build),
      inference by <a href="https://onnxruntime.ai" target="_blank" rel="noopener">ONNX Runtime Web</a>, rendering by
      <a href="https://threejs.org" target="_blank" rel="noopener">three.js</a>. Lamp model from
      <a href="https://github.com/clvrai/furniture-bench" target="_blank" rel="noopener">FurnitureBench</a>. Franka Panda model from
      <a href="https://github.com/google-deepmind/mujoco_menagerie" target="_blank" rel="noopener">MuJoCo Menagerie</a>;
      hand model by Wuji Technology.</p>
    </div>`;
}
