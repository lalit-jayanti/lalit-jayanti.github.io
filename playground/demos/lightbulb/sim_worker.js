// Simulation worker: MuJoCo + the policy, stepped in real time off the render thread. Streams poses to the page
// and takes perturbations (drag, knock, open hand, move bulb) from it.
import { LightbulbController } from './controller.js';

const MUJOCO_URL = 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.13.0/mujoco.js';
const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.bundle.min.mjs';
const ASSETS = new URL('./assets/', import.meta.url);

const CONTROL_DT = 0.04;
const KNOCK_SPEED = 0.7; // m/s given to the bulb
const KNOCK_TIME = 0.04; // s over which it is applied

let mujoco;
let model;
let data;
let ctl;
let starts = [];
let paused = false;
let ready = false; // the page has built its scene: until then the round waits at its start
let drag = null;
let knock = null;
let simTime = 0;
let epoch = 0; // bumped on every jump in state (new start, moved bulb): the page does not interpolate across it
let stepMs = 0;

const post = (msg, transfer) => self.postMessage(msg, transfer || []);

async function fetchBytes(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
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

const name = (obj, i) => mujoco.mj_id2name(model, obj, i) || '';

async function init() {
  const json = async (p) => (await fetch(new URL(p, ASSETS))).json();
  const [cfg, gains, startList, xml] = await Promise.all([
    json('config.json'),
    json('gains.json'),
    json('starts.json'),
    fetch(new URL('scene/scene.xml', ASSETS)).then((r) => r.text()),
  ]);
  starts = startList;
  const meshFiles = [...new Set([...xml.matchAll(/file="([^"]+)"/g)].map((m) => m[1]))];
  const total = 10.2e6 + 11e6 + 1.1e6 + 5.3e6;
  let got = 0;
  const bump = (n) => {
    got += n;
    post({ type: 'progress', frac: Math.min(1, got / total) });
  };
  const [mj, ortMod, policyBytes, meshes] = await Promise.all([
    import(MUJOCO_URL).then(async (m) => {
      const mod = await m.default();
      bump(10.2e6);
      return mod;
    }),
    import(ORT_URL).then((m) => {
      bump(11e6);
      return m;
    }),
    fetchBytes(new URL('policy.onnx', ASSETS), bump),
    Promise.all(meshFiles.map((f) => fetchBytes(new URL('scene/meshes/' + f, ASSETS), bump).then((b) => [f, b]))),
  ]);
  mujoco = mj;
  const ort = ortMod.default || ortMod;
  post({ type: 'progress', frac: 1, msg: 'Starting simulation…' });

  const vfs = new mujoco.MjVFS();
  for (const [f, b] of meshes) vfs.addBuffer('meshes/' + f, b);
  vfs.addBuffer('scene.xml', new TextEncoder().encode(xml));
  model = mujoco.MjModel.mj_loadXML('scene.xml', vfs);
  data = new mujoco.MjData(model);
  vfs.delete();
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(policyBytes, { executionProviders: ['wasm'] });
  ctl = new LightbulbController({ mujoco, model, data, cfg, gains, session, ort });
  ctl.applyExternal = applyExternal;
  // Stream poses every 5 physics steps (10 ms of sim time).
  ctl.onSubstep = (k) => {
    if ((k + 1) % 5 === 0) {
      simTime += 0.01;
      sendFrame();
    }
  };
  const sc = describeScene();
  post({ type: 'scene', geoms: sc.geoms, bodies: sc.bodies, bulbBody: sc.bulbBody, globeZ: sc.globeZ, starts: starts.map((s) => s.kind) }, sc.transfer);
}

// Everything the page needs to draw: render geoms (visual meshes + primitives) and body names.
function describeScene() {
  const geoms = [];
  const transfer = [];
  const MESH = mujoco.mjtGeom.mjGEOM_MESH.value;
  for (let g = 0; g < model.ngeom; g++) {
    const type = model.geom_type[g];
    const visual = model.geom_contype[g] === 0 && model.geom_conaffinity[g] === 0;
    const body = model.geom_bodyid[g];
    const bodyName = name(mujoco.mjtObj.mjOBJ_BODY.value, body);
    // Visual meshes are drawn; collision geoms only for the lamp parts and the table (their visual meshes stand in
    // where they exist), never for the robot.
    if (type === MESH && !visual) continue;
    if (!visual && !(bodyName === 'table' || name(mujoco.mjtObj.mjOBJ_GEOM.value, g) === 'floor')) continue;
    const mid = model.geom_matid[g];
    const rgba = Array.from(mid >= 0 ? model.mat_rgba.subarray(4 * mid, 4 * mid + 4) : model.geom_rgba.subarray(4 * g, 4 * g + 4));
    const e = {
      g, type, body, bodyName, rgba,
      size: Array.from(model.geom_size.subarray(3 * g, 3 * g + 3)),
      pos: Array.from(model.geom_pos.subarray(3 * g, 3 * g + 3)),
      quat: Array.from(model.geom_quat.subarray(4 * g, 4 * g + 4)),
    };
    if (type === MESH) {
      const m = model.geom_dataid[g];
      const v0 = model.mesh_vertadr[m];
      const f0 = model.mesh_faceadr[m];
      e.vert = new Float32Array(model.mesh_vert.subarray(3 * v0, 3 * (v0 + model.mesh_vertnum[m])));
      e.face = new Uint32Array(model.mesh_face.subarray(3 * f0, 3 * (f0 + model.mesh_facenum[m])));
      transfer.push(e.vert.buffer, e.face.buffer);
    }
    geoms.push(e);
  }
  const bodies = [];
  for (let b = 0; b < model.nbody; b++) bodies.push(name(mujoco.mjtObj.mjOBJ_BODY.value, b));
  return { geoms, bodies, bulbBody: ctl.bulbBody, globeZ: ctl.cfg.globe_center_z, transfer };
}

function anchorWorld(d, b, local) {
  const R = d.xmat;
  const x = d.xpos;
  return [0, 1, 2].map((i) => x[3 * b + i] + R[9 * b + 3 * i] * local[0] + R[9 * b + 3 * i + 1] * local[1] + R[9 * b + 3 * i + 2] * local[2]);
}

function applyExternal(d) {
  const xf = d.xfrc_applied;
  const dt = model.opt.timestep;
  if (knock) {
    const b = ctl.bulbBody;
    const mass = model.body_mass[b];
    for (let i = 0; i < 3; i++) xf[6 * b + i] += (mass * knock.v[i]) / KNOCK_TIME;
    knock.t += dt;
    if (knock.t >= KNOCK_TIME) knock = null;
  }
  if (drag && drag.target) {
    const b = drag.body;
    const p = anchorWorld(d, b, drag.local);
    const v = drag.prev ? p.map((x, i) => (x - drag.prev[i]) / dt) : [0, 0, 0];
    drag.prev = p;
    let F = [0, 1, 2].map((i) => drag.k * (drag.target[i] - p[i]) - drag.c * v[i]);
    const mag = Math.hypot(...F);
    if (mag > drag.fmax) F = F.map((x) => (x * drag.fmax) / mag);
    drag.force = Math.min(mag, drag.fmax);
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
}

function pickKind(kind) {
  const idx = starts.map((s, i) => [s.kind, i]).filter(([k]) => !kind || k === kind).map(([, i]) => i);
  return idx[Math.floor(Math.random() * idx.length)];
}

function startEpisode(i) {
  const index = i ?? pickKind();
  const s = starts[index];
  ctl.setState(s);
  post({ type: 'started', index, kind: s.kind });
  drag = null;
  knock = null;
  simTime = 0;
  epoch += 1;
  sendFrame();
}

// Put the bulb down somewhere else on the table, lying on its side, at rest.
function moveBulb() {
  const d = ctl.vw();
  const bq = ctl.bq;
  const yaw = Math.random() * 2 * Math.PI;
  const x = 0.42 + Math.random() * 0.2;
  const y = -0.02 + Math.random() * 0.2;
  // lying on its side: a quarter turn about x, then a random yaw about z
  const qx = [Math.SQRT1_2, Math.SQRT1_2, 0, 0];
  const qz = [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)];
  const q = [
    qz[0] * qx[0] - qz[3] * qx[3],
    qz[0] * qx[1] - qz[3] * qx[2],
    qz[0] * qx[2] + qz[3] * qx[1],
    qz[0] * qx[3] + qz[3] * qx[0],
  ];
  d.qpos[bq] = x;
  d.qpos[bq + 1] = y;
  d.qpos[bq + 2] = 0.032;
  for (let i = 0; i < 4; i++) d.qpos[bq + 3 + i] = q[i];
  for (let i = 0; i < 6; i++) d.qvel[ctl.bv + i] = 0;
  mujoco.mj_forward(model, data);
  ctl.invalidate();
  ctl.thread.relatch();
  epoch += 1;
  sendFrame();
}

function openHand() {
  for (let i = 0; i < 20; i++) ctl.handTarget[i] = Math.min(ctl.cfg.hand_upper[i], Math.max(ctl.cfg.hand_lower[i], 0));
  ctl.target.set(ctl.handTarget, 7);
}

// Body poses (and the task state) for the page, which interpolates between them.
function sendFrame() {
  const th = ctl.thread;
  post({
    type: 'pose',
    t: simTime,
    epoch,
    xpos: Float32Array.from(data.xpos),
    xquat: Float32Array.from(data.xquat),
    thread: { progress: th.progress, engaged: th.engaged, seated: th.seated, depth: th.depth },
    drag: drag ? { anchor: drag.anchor || null, target: drag.target, force: drag.force || 0 } : null,
    stepMs,
  });
}

// Real-time loop: one control step per 40 ms of wall time; slower machines run slower than real time.
let wall0 = 0;
let simAt = 0;
async function loop() {
  for (;;) {
    if (paused || !ctl || !ready) {
      await new Promise((r) => setTimeout(r, 50));
      wall0 = performance.now();
      simAt = 0;
      continue;
    }
    const t0 = performance.now();
    try {
      await ctl.step();
    } catch (err) {
      post({ type: 'error', message: String(err && err.message ? err.message : err) });
      paused = true;
      continue;
    }
    if (!Number.isFinite(data.qpos[0])) {
      post({ type: 'toast', text: 'The simulation became unstable, restarting' });
      startEpisode();
    }
    stepMs = 0.9 * stepMs + 0.1 * (performance.now() - t0);
    simAt += CONTROL_DT * 1000;
    const ahead = simAt - (performance.now() - wall0);
    if (ahead > 0) await new Promise((r) => setTimeout(r, ahead));
    else if (ahead < -250) {
      // too far behind: drop the backlog instead of fast-forwarding
      wall0 = performance.now();
      simAt = 0;
    }
  }
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      await init();
      startEpisode(pickKind(m.kind));
      wall0 = performance.now();
      simAt = 0;
      loop();
    } else if (!ctl) {
      return;
    } else if (m.type === 'start') startEpisode(m.index);
    else if (m.type === 'pause') paused = m.value;
    else if (m.type === 'ready') ready = true;
    else if (m.type === 'knock') {
      const yaw = Math.random() * 2 * Math.PI;
      knock = { t: 0, v: [KNOCK_SPEED * Math.cos(yaw), KNOCK_SPEED * Math.sin(yaw), 0.35] };
    } else if (m.type === 'open') openHand();
    else if (m.type === 'move') moveBulb();
    else if (m.type === 'drag') {
      const robot = m.body !== ctl.bulbBody;
      drag = {
        body: m.body,
        local: m.local,
        target: m.target,
        prev: null,
        k: robot ? 900 : 60,
        c: robot ? 40 : 1.5,
        fmax: robot ? 80 : 6,
      };
    } else if (m.type === 'dragTo') {
      if (drag) drag.target = m.target;
    } else if (m.type === 'dragEnd') drag = null;
    if (paused && m.type !== 'pause') sendFrame();
  } catch (err) {
    post({ type: 'error', message: String(err && err.message ? err.message : err) });
  }
};
