// Arm + dexterous hand controller for the light-bulb task: builds the policy inputs from the simulated scene,
// runs the ONNX policy at 25 Hz, maps its output to joint targets (differential IK for the arm, an integrated
// target for the hand) and drives the joints with a PD law at every physics step. The thread between the bulb
// and the socket is a force model applied to the bulb at every physics step.
// Runs unchanged in the browser and in Node (mujoco and onnxruntime are passed in).

const OUT_ABOVE = 0.005;
const OUT_RADIAL_SLACK = 0.004;
const RELATCH_SLACK = 0.0005;

const wrap = (a) => {
  const t = (a + Math.PI) % (2 * Math.PI);
  return (t < 0 ? t + 2 * Math.PI : t) - Math.PI;
};
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

function quatToR(q) {
  const [w, x, y, z] = q;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ];
}

// Solve A x = b for a small dense SPD matrix (Gaussian elimination with partial pivoting); A is n*n row-major.
function solve(A, b, n) {
  const M = A.slice();
  const x = b.slice();
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r * n + c]) > Math.abs(M[p * n + c])) p = r;
    if (p !== c) {
      for (let k = 0; k < n; k++) [M[c * n + k], M[p * n + k]] = [M[p * n + k], M[c * n + k]];
      [x[c], x[p]] = [x[p], x[c]];
    }
    for (let r = c + 1; r < n; r++) {
      const f = M[r * n + c] / M[c * n + c];
      for (let k = c; k < n; k++) M[r * n + k] -= f * M[c * n + k];
      x[r] -= f * x[c];
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    let s = x[r];
    for (let k = r + 1; k < n; k++) s -= M[r * n + k] * x[k];
    x[r] = s / M[r * n + r];
  }
  return x;
}

class Thread {
  constructor(ctl) {
    this.c = ctl;
    const p = ctl.cfg.thread;
    this.p = p;
    const d = ctl.data;
    const b = ctl.baseBody;
    const R = Array.from(d.xmat.subarray(9 * b, 9 * b + 9));
    this.Rs = R;
    this.axis = [R[2], R[5], R[8]];
    const st = p.socket_top;
    this.rim = [0, 1, 2].map((i) => d.xpos[3 * b + i] + R[3 * i] * st[0] + R[3 * i + 1] * st[1] + R[3 * i + 2] * st[2]);
    this.tipLocal = p.stem_tip;
    this.thetaSeat = ctl.cfg.theta_seat;
    this.seatedTip = this.rim.map((r, i) => r - p.seat_depth * this.axis[i]);
    this.engaged = false;
    this.psi0 = this.psi = this.psiA = 0;
    this.ePrev = null;
    this.depth = this.theta = this.radial = this.tilt = 0;
    this.seated = false;
    this.tip = [0, 0, 0];
  }

  measure() {
    const d = this.c.vw();
    const b = this.c.bulbBody;
    const pos = [d.xpos[3 * b], d.xpos[3 * b + 1], d.xpos[3 * b + 2]];
    const R = d.xmat.subarray(9 * b, 9 * b + 9);
    const tl = this.tipLocal;
    const tip = [0, 1, 2].map((i) => pos[i] + R[3 * i] * tl[0] + R[3 * i + 1] * tl[1] + R[3 * i + 2] * tl[2]);
    const a = this.axis;
    const dd = [tip[0] - this.rim[0], tip[1] - this.rim[1], tip[2] - this.rim[2]];
    const along = dd[0] * a[0] + dd[1] * a[1] + dd[2] * a[2];
    const rVec = [dd[0] - along * a[0], dd[1] - along * a[1], dd[2] - along * a[2]];
    const tilt = Math.acos(clamp(R[2] * a[0] + R[5] * a[1] + R[8] * a[2], -1, 1));
    // R_rel = Rs^T R; psi = atan2(R_rel[1,0], R_rel[0,0])
    const S = this.Rs;
    const r10 = S[1] * R[0] + S[4] * R[3] + S[7] * R[6];
    const r00 = S[0] * R[0] + S[3] * R[3] + S[6] * R[6];
    return { pos, tip, depth: -along, rVec, tilt, psiW: Math.atan2(r10, r00) };
  }

  store(tip, depth, rVec, tilt, theta) {
    const p = this.p;
    const cfg = this.c.cfg;
    this.tip = tip;
    this.depth = depth;
    this.radial = Math.hypot(...rVec);
    this.tilt = tilt;
    this.theta = this.engaged ? theta : 0;
    const seated =
      depth >= p.seat_depth - cfg.seated_depth_tol && theta >= this.thetaSeat - (cfg.seated_theta_tol_deg * Math.PI) / 180;
    this.seated = this.engaged && seated;
  }

  relatch() {
    const p = this.p;
    const { tip, depth, rVec, tilt, psiW } = this.measure();
    const radial = Math.hypot(...rVec);
    const inHole = depth >= -OUT_ABOVE && radial <= p.hole_radius + OUT_RADIAL_SLACK;
    this.engaged = inHole && depth >= p.free_depth - RELATCH_SLACK;
    const theta = clamp((2 * Math.PI * (depth - p.free_depth)) / p.lead, 0, this.thetaSeat);
    this.psi = this.psiA = psiW;
    this.psi0 = psiW + theta;
    this.ePrev = null;
    this.store(tip, depth, rVec, tilt, -(this.psi - this.psi0));
  }

  refresh() {
    const { tip, depth, rVec, tilt, psiW } = this.measure();
    const psi = this.psi + wrap(psiW - this.psi);
    this.store(tip, depth, rVec, tilt, -(psi - this.psi0));
  }

  // Force (world), torque (world) and the point (stem tip) to apply this physics step.
  wrench() {
    const p = this.p;
    const c = this.c;
    const { pos, tip, depth, rVec, tilt, psiW } = this.measure();
    const a = this.axis;
    this.psi += wrap(psiW - this.psi);
    const radial = Math.hypot(...rVec);
    const [vel, ang] = c.bulbVelWorld();
    const r = [tip[0] - pos[0], tip[1] - pos[1], tip[2] - pos[2]];
    const vTip = [
      vel[0] + ang[1] * r[2] - ang[2] * r[1],
      vel[1] + ang[2] * r[0] - ang[0] * r[2],
      vel[2] + ang[0] * r[1] - ang[1] * r[0],
    ];
    const ddepth = -(vTip[0] * a[0] + vTip[1] * a[1] + vTip[2] * a[2]);
    const wAx = ang[0] * a[0] + ang[1] * a[1] + ang[2] * a[2];
    const out = depth < -OUT_ABOVE || radial > p.hole_radius + OUT_RADIAL_SLACK;
    if (out) {
      this.engaged = false;
      this.ePrev = null;
    }
    let lamStop = 0;
    if (!this.engaged && !out) {
      const eStop = depth - p.free_depth;
      if (eStop > 0) {
        lamStop = p.k_ax * eStop + p.c_ax * Math.max(ddepth, 0);
        if (radial < p.capture_radius && tilt < (p.capture_tilt_deg * Math.PI) / 180) {
          this.engaged = true;
          this.psi0 = this.psiA = this.psi;
          this.ePrev = null;
        }
      }
    }
    const theta = -(this.psi - this.psi0);
    if (theta < -Math.PI / 2) this.engaged = false;
    let lam = 0;
    let torqueAx = 0;
    if (this.engaged) {
      const dStar = p.free_depth + (p.lead * Math.min(theta, this.thetaSeat)) / (2 * Math.PI);
      const e = depth - dStar;
      const eDot = this.ePrev === null ? 0 : (e - this.ePrev) / c.dt;
      this.ePrev = e;
      lam = p.k_ax * e + p.c_ax * eDot;
      torqueAx = theta < this.thetaSeat
        ? (-lam * p.lead) / (2 * Math.PI)
        : p.k_rot_stop * (theta - this.thetaSeat) - p.c_rot_stop * wAx;
      const limit = p.friction_torque / p.k_stick;
      const slip = this.psi - this.psiA;
      if (Math.abs(slip) > limit) this.psiA = this.psi - Math.sign(slip) * limit;
      const tau = -p.k_stick * (this.psi - this.psiA) - p.c_stick * wAx;
      torqueAx += clamp(tau, -p.friction_torque, p.friction_torque);
    }
    const f = lamStop + lam;
    return { force: [f * a[0], f * a[1], f * a[2]], torque: [torqueAx * a[0], torqueAx * a[1], torqueAx * a[2]], tip };
  }

  // Screw-in progress in [0, 1].
  get progress() {
    return clamp(this.theta / this.thetaSeat, 0, 1);
  }
}

export class LightbulbController {
  constructor({ mujoco, model, data, cfg, gains, session, ort }) {
    this.mj = mujoco;
    this.model = model;
    this.data = data;
    this.cfg = cfg;
    this.session = session;
    this.ort = ort;
    this.dt = model.opt.timestep;
    const body = (n) => {
      const id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, n);
      if (id < 0) throw new Error(`body ${n} not found in the model`);
      return id;
    };
    this.bulbBody = body(cfg.bulb_body);
    this.baseBody = body(cfg.base_body);
    this.eeBody = body(cfg.ee_body);
    this.wristBody = body(cfg.wrist_body);
    this.tipBodies = cfg.fingertip_bodies.map(body);
    this.aq = cfg.arm_qpos;
    this.av = cfg.arm_qvel;
    this.hq = cfg.hand_qpos;
    this.hv = cfg.hand_qvel;
    this.bq = cfg.bulb_qpos;
    this.bv = cfg.bulb_qvel;
    this.qIdx = [...this.aq, ...this.hq];
    this.vIdx = [...this.av, ...this.hv];
    this.kp = [...gains.robot.kp, ...gains.hand.kp];
    this.kd = [...gains.robot.kv, ...gains.hand.kv];
    this.fLo = [...gains.robot.force_lo, ...gains.hand.force_lo];
    this.fHi = [...gains.robot.force_hi, ...gains.hand.force_hi];
    this.target = new Float64Array(27);
    this.handTarget = new Float64Array(20);
    this.lastAction = new Float64Array(26);
    this.obsBuf = new Float32Array(139);
    const nv = model.nv;
    this.jacp = new mujoco.DoubleBuffer(3 * nv);
    this.jacr = new mujoco.DoubleBuffer(3 * nv);
    // Extra wrenches the page applies each physics step (dragging, pushes): fn(data) writing xfrc_applied.
    this.applyExternal = null;
    // Called after every physics step with its index in the control period (the page streams poses from it).
    this.onSubstep = null;
    this._v = null;
    mujoco.mj_forward(model, data);
    this.thread = new Thread(this);
  }

  dispose() {
    this.jacp.delete();
    this.jacr.delete();
  }

  // Typed-array views into the simulation state, fetched once and reused until the next step (each property
  // access on MjData builds a new view, which is slow in a per-joint loop).
  vw() {
    if (!this._v) {
      const d = this.data;
      this._v = {
        qpos: d.qpos, qvel: d.qvel, xpos: d.xpos, xmat: d.xmat, xipos: d.xipos,
        qfrc_applied: d.qfrc_applied, xfrc_applied: d.xfrc_applied,
      };
    }
    return this._v;
  }

  // Call after changing the state outside step()/setState() (e.g. mj_forward from the page).
  invalidate() {
    this._v = null;
  }

  view(buf) {
    return buf.GetView ? buf.GetView() : buf.getView();
  }

  bulbVelWorld() {
    const d = this.vw();
    const b = this.bulbBody;
    const v = [d.qvel[this.bv], d.qvel[this.bv + 1], d.qvel[this.bv + 2]];
    const w = [d.qvel[this.bv + 3], d.qvel[this.bv + 4], d.qvel[this.bv + 5]];
    const R = d.xmat.subarray(9 * b, 9 * b + 9);
    return [v, [0, 1, 2].map((i) => R[3 * i] * w[0] + R[3 * i + 1] * w[1] + R[3 * i + 2] * w[2])];
  }

  // Place everything from a start state.
  setState(s) {
    const d = this.data;
    const m = this.model;
    d.qpos.set(m.qpos0);
    d.qvel.fill(0);
    this.aq.forEach((a, i) => (d.qpos[a] = s.arm_q[i]));
    this.av.forEach((a, i) => (d.qvel[a] = s.arm_dq[i]));
    this.hq.forEach((a, i) => (d.qpos[a] = s.hand_q[i]));
    this.hv.forEach((a, i) => (d.qvel[a] = s.hand_dq[i]));
    const q = s.bulb_quat;
    const n = Math.hypot(...q);
    for (let i = 0; i < 3; i++) d.qpos[this.bq + i] = s.bulb_pos[i];
    for (let i = 0; i < 4; i++) d.qpos[this.bq + 3 + i] = q[i] / n;
    const R = quatToR(q.map((x) => x / n));
    for (let i = 0; i < 3; i++) d.qvel[this.bv + i] = s.bulb_vel[i];
    const w = s.bulb_ang;
    for (let i = 0; i < 3; i++) d.qvel[this.bv + 3 + i] = R[i] * w[0] + R[3 + i] * w[1] + R[6 + i] * w[2];
    d.qacc_warmstart.fill(0);
    d.qfrc_applied.fill(0);
    d.xfrc_applied.fill(0);
    this.mj.mj_forward(m, d);
    this._v = null;
    for (let i = 0; i < 20; i++) this.handTarget[i] = clamp(s.hand_target[i], this.cfg.hand_lower[i], this.cfg.hand_upper[i]);
    for (let i = 0; i < 7; i++) this.target[i] = s.arm_q[i];
    this.target.set(this.handTarget, 7);
    this.lastAction.fill(0);
    this.thread.relatch();
  }

  observation() {
    const d = this.vw();
    const o = this.obsBuf;
    const th = this.thread;
    const seat = th.seatedTip;
    let k = 0;
    for (const a of this.aq) o[k++] = d.qpos[a];
    for (const a of this.av) o[k++] = d.qvel[a];
    for (const a of this.hq) o[k++] = d.qpos[a];
    for (const a of this.hv) o[k++] = d.qvel[a];
    for (let i = 0; i < 20; i++) o[k++] = this.handTarget[i];
    const wb = this.wristBody;
    for (let i = 0; i < 3; i++) o[k++] = d.xpos[3 * wb + i] - seat[i];
    const Rw = d.xmat.subarray(9 * wb, 9 * wb + 6); // first two rows of R
    for (let i = 0; i < 6; i++) o[k++] = Rw[i];
    for (let i = 0; i < 3; i++) o[k++] = th.tip[i] - seat[i];
    const bb = this.bulbBody;
    const Rb = d.xmat.subarray(9 * bb, 9 * bb + 9);
    for (let i = 0; i < 6; i++) o[k++] = Rb[i];
    const [v, w] = this.bulbVelWorld();
    for (let i = 0; i < 3; i++) o[k++] = v[i];
    for (let i = 0; i < 3; i++) o[k++] = w[i];
    const gz = this.cfg.globe_center_z;
    const globe = [0, 1, 2].map((i) => d.xpos[3 * bb + i] + Rb[3 * i + 2] * gz);
    for (const t of this.tipBodies) for (let i = 0; i < 3; i++) o[k++] = d.xpos[3 * t + i] - globe[i];
    for (let i = 0; i < 26; i++) o[k++] = this.lastAction[i];
    return o;
  }

  async policy(obs) {
    const out = await this.session.run({ obs: new this.ort.Tensor('float32', obs, [1, 139]) });
    return out.action.data;
  }

  applyAction(a) {
    const m = this.model;
    const d = this.vw();
    const cfg = this.cfg;
    const nv = m.nv;
    this.mj.mj_jacBody(m, this.data, this.jacp, this.jacr, this.eeBody);
    const jp = this.view(this.jacp);
    const jr = this.view(this.jacr);
    const J = new Float64Array(42); // 6 x 7
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 7; c++) {
        J[r * 7 + c] = jp[r * nv + this.av[c]];
        J[(r + 3) * 7 + c] = jr[r * nv + this.av[c]];
      }
    }
    const delta = [0, 1, 2, 3, 4, 5].map((i) => cfg.arm_scale[i] * Math.tanh(a[i]));
    const A = new Array(36).fill(0);
    for (let i = 0; i < 6; i++) {
      for (let j = 0; j < 6; j++) {
        let s = 0;
        for (let c = 0; c < 7; c++) s += J[i * 7 + c] * J[j * 7 + c];
        A[i * 6 + j] = s + (i === j ? cfg.arm_ik_damping : 0);
      }
    }
    const y = solve(A, delta, 6);
    for (let c = 0; c < 7; c++) {
      let dq = 0;
      for (let i = 0; i < 6; i++) dq += J[i * 7 + c] * y[i];
      this.target[c] = d.qpos[this.aq[c]] + clamp(dq, -cfg.arm_max_dq, cfg.arm_max_dq);
    }
    for (let i = 0; i < 20; i++) {
      this.handTarget[i] = clamp(this.handTarget[i] + cfg.hand_scale * clamp(a[6 + i], -1, 1), cfg.hand_lower[i], cfg.hand_upper[i]);
    }
    this.target.set(this.handTarget, 7);
    for (let i = 0; i < 26; i++) this.lastAction[i] = a[i];
  }

  physics() {
    const m = this.model;
    const b = this.bulbBody;
    for (let s = 0; s < this.cfg.decimation; s++) {
      const d = this.vw();
      const { force, torque, tip } = this.thread.wrench();
      const qf = d.qfrc_applied;
      qf.fill(0);
      for (let j = 0; j < 27; j++) {
        const u = this.kp[j] * (this.target[j] - d.qpos[this.qIdx[j]]) - this.kd[j] * d.qvel[this.vIdx[j]];
        qf[this.vIdx[j]] = clamp(u, this.fLo[j], this.fHi[j]);
      }
      const xf = d.xfrc_applied;
      xf.fill(0);
      const r = [tip[0] - d.xipos[3 * b], tip[1] - d.xipos[3 * b + 1], tip[2] - d.xipos[3 * b + 2]];
      xf[6 * b] = force[0];
      xf[6 * b + 1] = force[1];
      xf[6 * b + 2] = force[2];
      xf[6 * b + 3] = torque[0] + r[1] * force[2] - r[2] * force[1];
      xf[6 * b + 4] = torque[1] + r[2] * force[0] - r[0] * force[2];
      xf[6 * b + 5] = torque[2] + r[0] * force[1] - r[1] * force[0];
      if (this.applyExternal) this.applyExternal(d);
      this.mj.mj_step(m, this.data);
      this._v = null;
      this.thread.refresh();
      if (this.onSubstep) this.onSubstep(s);
    }
  }

  async step() {
    const a = await this.policy(this.observation());
    this.applyAction(a);
    this.physics();
    return a;
  }
}
