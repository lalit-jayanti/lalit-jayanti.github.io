import { ReferenceStepper, DT, wrapToPi, setAccelLimit } from './lip.js';
export { setAccelLimit };

const J = 29;
const OBS = 2 * J + 6 + J;
const HIST = 10;
const GOAL_FRAMES = 10;
const GOAL_STRIDE = 5;
const GOAL_FRAME_DIM = 2 * J + 6 + 3;
const AHEAD = 46;
const REANCHOR_XY = 0.35;
const SCAN_X = 17;
const SCAN_Y = 11;
const Z0 = 0.67;

export function buildSceneXml(robotXml, course, cfg) {
  let xml = robotXml.replace(
    /<option[^>]*\/>/,
    `<option timestep="${cfg.sim_dt}" integrator="implicitfast" iterations="10" ls_iterations="20"/>`
  );
  const { nx, ny, cell, ox, oy } = course;
  const rx = 0.5 * (nx - 1) * cell;
  const ry = 0.5 * (ny - 1) * cell;
  let top = 0;
  for (let i = 0; i < course.hs.length; i++) if (course.hs[i] > top) top = course.hs[i];
  course.hfTop = top + 0.5;
  xml = xml.replace('<asset>', `<asset>\n    <hfield name="course" nrow="${ny}" ncol="${nx}" size="${rx} ${ry} ${course.hfTop} 0.1"/>`);
  xml = xml.replace(
    /<geom name="floor"[^>]*\/>/,
    `<geom name="floor" size="0 0 0.05" type="plane"/>\n    <geom name="course" type="hfield" hfield="course" pos="${ox + rx} ${oy + ry} -0.5"/>`
  );
  const act = cfg.joint_names
    .map((j, i) => {
      const kp = cfg.kp[i], kd = cfg.kd[i], e = cfg.effort[i];
      return `    <general name="${j}" joint="${j}" gaintype="fixed" biastype="affine" gainprm="${kp}" biasprm="0 ${-kp} ${-kd}" forcelimited="true" forcerange="${-e} ${e}"/>`;
    })
    .join('\n');
  xml = xml.replace('</mujoco>', `  <actuator>\n${act}\n  </actuator>\n</mujoco>`);
  return xml;
}

export function fillHeightField(model, course) {
  const d = model.hfield_data;
  const { nx, ny } = course;
  for (let r = 0; r < ny; r++) for (let c = 0; c < nx; c++) d[r * nx + c] = (course.hs[c * ny + r] + 0.5) / course.hfTop;
}

function quatToR(q) {
  const [w, x, y, z] = q;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ];
}

class Track {
  constructor(course, legs, latency) {
    this.gen = new ReferenceStepper(course, legs);
    this.latency = latency;
    this.frames = new Map();
    this.frame = 0;
    this.head = -1;
    this.cmd = [0, 0, 0];
    this.genCmd = [0, 0, 0];
  }

  reset(x, y, yaw) {
    this.gen.reset(x, y, yaw, Z0);
    this.restart(0);
  }

  resetFromFeet(feetPos, feetRpy) {
    this.gen.resetFromFeet(feetPos, feetRpy, Z0);
    this.restart(this.frame);
  }

  restart(k0) {
    this.frames.clear();
    this.frame = k0;
    this.head = k0 - 1;
    this.genCmd = this.cmd.slice();
    this.extend(k0 + AHEAD);
  }

  extend(to) {
    while (this.head < to) {
      const f = this.gen.step(this.genCmd, Z0);
      const q = f.qpos;
      this.head += 1;
      this.frames.set(this.head, {
        pos: [q[0], q[1], q[2]],
        quat: [q[3], q[4], q[5], q[6]],
        yaw: 2 * Math.atan2(q[6], q[3]),
        dof: q.slice(7),
        foot: f.footTarget,
        contacts: f.contacts,
        state: this.latency < AHEAD ? this.gen.save() : null,
      });
    }
  }

  setCommand(cmd) {
    this.cmd = cmd.slice();
    if (this.latency >= AHEAD) {
      this.genCmd = this.cmd.slice();
      return;
    }
    if (cmd.every((v, i) => v === this.genCmd[i])) return;
    const k = this.frame + this.latency - 1;
    const f = this.frames.get(k);
    if (!f || !f.state) return;
    this.gen.load(f.state);
    for (let j = k + 1; j <= this.head; j++) this.frames.delete(j);
    this.head = k;
    this.genCmd = this.cmd.slice();
    this.extend(this.frame + AHEAD);
  }

  advance() {
    this.frame += 1;
    for (const k of this.frames.keys()) if (k < this.frame - 2) this.frames.delete(k);
    this.extend(this.frame + AHEAD);
  }

  dofVel(k) {
    const f = this.frames.get(k);
    const a = this.frames.get(k - 1) || f;
    const b = this.frames.get(k + 1) || f;
    const span = Math.max(DT * ((this.frames.has(k - 1) ? 1 : 0) + (this.frames.has(k + 1) ? 1 : 0)), DT);
    return f.dof.map((_, i) => (b.dof[i] - a.dof[i]) / span);
  }
}

export class Controller {
  constructor({ mujoco, model, data, course, legs, config, session, ort, latency = 5 }) {
    this.mj = mujoco;
    this.model = model;
    this.data = data;
    this.course = course;
    this.cfg = config;
    this.session = session;
    this.ort = ort;
    const jid = (n) => mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, n);
    this.qadr = config.joint_names.map((n) => model.jnt_qposadr[jid(n)]);
    this.vadr = config.joint_names.map((n) => model.jnt_dofadr[jid(n)]);
    for (let j = 0; j < J; j++) model.dof_armature[this.vadr[j]] = config.armature[j];
    this.pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'pelvis');
    this.feetIds = ['left_ankle_roll_link', 'right_ankle_roll_link'].map((n) => mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, n));
    this.q0 = Float64Array.from(config.q0);
    this.scale = Float64Array.from(config.action_scale);
    this.track = new Track(course, legs, latency);
    this.hist = new Float32Array(HIST * OBS);
    this.goal = new Float32Array(GOAL_FRAMES * GOAL_FRAME_DIM);
    this.scan = new Float32Array(SCAN_X * SCAN_Y);
    this.action = new Float32Array(J);
    this.prevAction = new Float32Array(J);
    this.fresh = true;
    this.reanchors = 0;
    this.offXY = 0;
    this.applyExternal = null;
  }

  reset(x, y, yaw) {
    const p = this.validStart(x, y, yaw);
    this.track.cmd = [0, 0, 0];
    this.track.reset(p[0], p[1], yaw);
    this.placeOnReference();
  }

  validStart(x, y, yaw) {
    const c = this.course;
    const d = [Math.cos(yaw), Math.sin(yaw)];
    const l = [-Math.sin(yaw), Math.cos(yaw)];
    const ok = (bx, by) => {
      const h0 = c.height(bx, by);
      for (const f of [-0.2, 0, 0.25]) {
        for (const s of [-0.25, -0.12, 0, 0.12, 0.25]) {
          const px = bx + f * d[0] + s * l[0], py = by + f * d[1] + s * l[1];
          if (!c.inside(px, py) || Math.abs(c.height(px, py) - h0) > 0.02) return false;
        }
      }
      return true;
    };
    if (ok(x, y)) return [x, y];
    for (let r = 0.05; r < 3.0; r += 0.05) {
      for (let a = 0; a < 16; a++) {
        const th = (a / 16) * 2 * Math.PI;
        const bx = x + r * Math.cos(th), by = y + r * Math.sin(th);
        if (ok(bx, by)) return [bx, by];
      }
    }
    return [x, y];
  }

  placeOnReference() {
    const d = this.data;
    const t = this.track;
    const f = t.frames.get(t.frame);
    const g = t.frames.get(t.frame + 1);
    d.qpos.fill(0);
    d.qvel.fill(0);
    d.qfrc_applied.fill(0);
    d.xfrc_applied.fill(0);
    for (let i = 0; i < 3; i++) {
      d.qpos[i] = f.pos[i];
      d.qvel[i] = (g.pos[i] - f.pos[i]) / DT;
    }
    for (let i = 0; i < 4; i++) d.qpos[3 + i] = f.quat[i];
    d.qvel[5] = wrapToPi(g.yaw - f.yaw) / DT;
    const dv = t.dofVel(t.frame);
    for (let j = 0; j < J; j++) {
      d.qpos[this.qadr[j]] = f.dof[j];
      d.qvel[this.vadr[j]] = dv[j];
      d.ctrl[j] = f.dof[j];
    }
    this.mj.mj_forward(this.model, d);
    this.fresh = true;
    this.prevAction.fill(0);
  }

  reanchor() {
    const d = this.data;
    const pos = [], rpy = [];
    for (const b of this.feetIds) {
      pos.push([d.xpos[3 * b], d.xpos[3 * b + 1], d.xpos[3 * b + 2]]);
      const R = d.xmat.subarray(9 * b, 9 * b + 9);
      rpy.push([0, 0, Math.atan2(R[3], R[0])]);
    }
    this.track.resetFromFeet(pos, rpy);
    this.reanchors += 1;
  }

  setCommand(cmd) {
    this.track.setCommand(cmd);
  }

  rootState() {
    const d = this.data;
    const q = [d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]];
    const n = Math.hypot(...q);
    return { pos: [d.qpos[0], d.qpos[1], d.qpos[2]], R: quatToR(q.map((v) => v / n)) };
  }

  observation(R) {
    const d = this.data;
    const o = new Float32Array(OBS);
    for (let j = 0; j < J; j++) {
      o[j] = d.qpos[this.qadr[j]] - this.q0[j];
      o[J + j] = d.qvel[this.vadr[j]];
    }
    o[2 * J] = d.qvel[3];
    o[2 * J + 1] = d.qvel[4];
    o[2 * J + 2] = d.qvel[5];
    o[2 * J + 3] = -R[6];
    o[2 * J + 4] = -R[7];
    o[2 * J + 5] = -R[8];
    o.set(this.prevAction, 2 * J + 6);
    return o;
  }

  buildGoal(p, R) {
    const t = this.track;
    const g = this.goal;
    for (let k = 0; k < GOAL_FRAMES; k++) {
      const idx = t.frame + GOAL_STRIDE * k;
      const f = t.frames.get(idx);
      const dv = t.dofVel(idx);
      const b = k * GOAL_FRAME_DIM;
      for (let j = 0; j < J; j++) {
        g[b + j] = f.dof[j];
        g[b + J + j] = dv[j];
      }
      const Rf = quatToR(f.quat);
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 2; c++) g[b + 2 * J + 2 * r + c] = R[r] * Rf[c] + R[3 + r] * Rf[3 + c] + R[6 + r] * Rf[6 + c];
      }
      const dp = [f.pos[0] - p[0], f.pos[1] - p[1], f.pos[2] - p[2]];
      for (let r = 0; r < 3; r++) g[b + 2 * J + 6 + r] = R[r] * dp[0] + R[3 + r] * dp[1] + R[6 + r] * dp[2];
    }
  }

  buildScan(p, R) {
    let c = R[0], s = R[3];
    const n = Math.max(Math.hypot(c, s), 1e-6);
    c /= n;
    s /= n;
    for (let i = 0; i < SCAN_X; i++) {
      const x = -0.8 + 0.1 * i;
      for (let j = 0; j < SCAN_Y; j++) {
        const y = -0.5 + 0.1 * j;
        this.scan[i * SCAN_Y + j] = this.course.scanHeight(p[0] + c * x - s * y, p[1] + s * x + c * y) - p[2];
      }
    }
  }

  async act() {
    const { pos, R } = this.rootState();
    const obs = this.observation(R);
    if (this.fresh) {
      for (let h = 0; h < HIST; h++) this.hist.set(obs, h * OBS);
      this.fresh = false;
    } else {
      this.hist.copyWithin(0, OBS);
      this.hist.set(obs, (HIST - 1) * OBS);
    }
    this.buildGoal(pos, R);
    this.buildScan(pos, R);
    const ort = this.ort;
    const out = await this.session.run({
      obs_hist: new ort.Tensor('float32', this.hist, [1, HIST, OBS]),
      goal: new ort.Tensor('float32', this.goal, [1, this.goal.length]),
      scan: new ort.Tensor('float32', this.scan, [1, this.scan.length]),
    });
    const a = out.action.data;
    let finite = true;
    for (let j = 0; j < J; j++) {
      if (!Number.isFinite(a[j])) finite = false;
      this.action[j] = a[j];
    }
    if (!finite) this.action.fill(0);
    const ref = this.track.frames.get(this.track.frame);
    this.offXY = Math.hypot(pos[0] - ref.pos[0], pos[1] - ref.pos[1]);
    return this.action;
  }

  physics() {
    const d = this.data;
    for (let j = 0; j < J; j++) d.ctrl[j] = this.q0[j] + this.scale[j] * this.action[j];
    for (let s = 0; s < this.cfg.decimation; s++) {
      if (this.applyExternal) this.applyExternal(d);
      this.mj.mj_step(this.model, d);
    }
  }

  advance(autoReanchor = true) {
    this.prevAction.set(this.action);
    const t = this.track;
    t.advance();
    const ref = t.frames.get(t.frame);
    const d = this.data;
    if (autoReanchor && Math.hypot(d.qpos[0] - ref.pos[0], d.qpos[1] - ref.pos[1]) > REANCHOR_XY) {
      this.reanchor();
      return true;
    }
    return false;
  }

  async step() {
    await this.act();
    this.physics();
    return this.advance();
  }

  posture() {
    const { pos, R } = this.rootState();
    return { clearance: pos[2] - this.course.height(pos[0], pos[1]), up: R[8], pos };
  }

  fallen() {
    const p = this.posture();
    return p.up < 0.4 || p.clearance < 0.35;
  }

  refMarkers() {
    const t = this.track;
    const f = t.frames.get(t.frame);
    const feet = [];
    for (let k = t.frame; k <= t.frame + 30 && feet.length < 4; k += 10) {
      const g = t.frames.get(k);
      if (g) feet.push(g.foot);
    }
    return { pelvis: f.pos, feet };
  }

  heading() {
    const { R } = this.rootState();
    return Math.atan2(R[3], R[0]);
  }
}
