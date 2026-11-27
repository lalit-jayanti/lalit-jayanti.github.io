const J = 29;
const SEARCH_EVERY = 10;
const HALFLIFE = 0.12;
const TRAJ_TAU = 0.12;
const MIN_JUMP = 8;
const KEEP_BIAS = 0.15;
const WEIGHTS = { traj_pos: 2.5, traj_dir: 1.5, foot_pos: 0.25, foot_vel: 0.35, root_vel: 0.5 };

export function commandFromKeys(keys) {
  const fwd = (keys.has('w') ? 1 : 0) - (keys.has('s') ? 1 : 0);
  const side = (keys.has('a') ? 1 : 0) - (keys.has('d') ? 1 : 0);
  const turn = (keys.has('q') ? 1 : 0) - (keys.has('e') ? 1 : 0);
  if (fwd > 0 && turn) return [1.45, 0, 0.85 * turn];
  return [fwd > 0 ? 1.1 : fwd < 0 ? -0.8 : 0, 1.45 * side, 1.1 * turn];
}

function quatMul(a, b) {
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
}
const conj = (q) => [q[0], -q[1], -q[2], -q[3]];
const yawQuat = (y) => [Math.cos(y / 2), 0, 0, Math.sin(y / 2)];
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

function quatLog(q) {
  const s = q[0] < 0 ? -1 : 1;
  const w = s * q[0];
  const v = [s * q[1], s * q[2], s * q[3]];
  const n = Math.hypot(v[0], v[1], v[2]);
  if (n < 1e-9) return [0, 0, 0];
  const a = (2 * Math.atan2(n, w)) / n;
  return [v[0] * a, v[1] * a, v[2] * a];
}

function quatExp(r) {
  const th = Math.hypot(r[0], r[1], r[2]);
  if (th < 1e-9) return [1, 0, 0, 0];
  const s = Math.sin(th / 2) / th;
  return [Math.cos(th / 2), r[0] * s, r[1] * s, r[2] * s];
}

export function parseLibrary(meta, buffer) {
  const P = meta.pose_dim;
  const F = meta.feat_dim;
  const nPose = meta.clips.reduce((a, c) => a + c.frames, 0);
  const nFeat = meta.clips.reduce((a, c) => a + c.valid, 0);
  const raw = new Int16Array(buffer);
  const clips = [];
  let po = 0;
  let fo = 0;
  for (const c of meta.clips) {
    const n = c.frames;
    const quat = new Float32Array(4 * n);
    const dof = new Float32Array(J * n);
    const head = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const r = (po + i) * P;
      for (let k = 0; k < 4; k++) quat[4 * i + k] = raw[r + k] / 30000;
      for (let j = 0; j < J; j++) dof[J * i + j] = raw[r + 4 + j] / 10000;
      head[i] = raw[r + 4 + J] / 10000;
    }
    const fps = meta.fps;
    const dofv = new Float32Array(J * n);
    for (let i = 0; i < n; i++) {
      const a = Math.max(0, i - 1);
      const b = Math.min(n - 1, i + 1);
      for (let j = 0; j < J; j++) dofv[J * i + j] = ((dof[J * b + j] - dof[J * a + j]) * fps) / (b - a);
    }
    clips.push({ name: c.name, frames: n, valid: c.valid, quat, dof, dofv, head, featStart: fo });
    po += n;
    fo += c.valid;
  }
  const feat = new Float32Array(nFeat * F);
  const fr = raw.subarray(nPose * P, nPose * P + nFeat * F);
  for (let i = 0; i < feat.length; i++) feat[i] = fr[i] / 1000;
  const w = new Float32Array(F);
  for (const [g, [a, b]] of Object.entries(meta.groups)) for (let i = a; i < b; i++) w[i] = WEIGHTS[g] ?? 1;
  return { meta, clips, feat, F, w, fps: meta.fps };
}

export class WalkMatcher {
  constructor(lib) {
    this.lib = lib;
    this.input = { vx: 0, vy: 0, wz: 0 };
    this.query = new Float32Array(lib.F);
    this.y = (2 * Math.LN2) / HALFLIFE;
    this.dofOff = new Float64Array(J);
    this.dofOffV = new Float64Array(J);
    this.rotOff = [0, 0, 0];
    this.rotOffV = [0, 0, 0];
    this.out = { quat: [1, 0, 0, 0], dof: new Float32Array(J), dofv: new Float32Array(J) };
    this.standClip = Math.max(0, lib.clips.findIndex((c) => c.name === 'stand'));
    this.lastSwitch = null;
    this.reset(0);
  }

  reset(yaw) {
    this.c = this.standClip;
    this.f = 0;
    this.yawOff = yaw - this.lib.clips[this.c].head[0];
    this.dofOff.fill(0);
    this.dofOffV.fill(0);
    this.rotOff = [0, 0, 0];
    this.rotOffV = [0, 0, 0];
    this.t = 1e3;
    this.sinceSearch = 0;
    this.dirty = true;
  }

  setInput(vx, vy, wz) {
    const i = this.input;
    if (i.vx !== vx || i.vy !== vy || i.wz !== wz) {
      i.vx = vx;
      i.vy = vy;
      i.wz = wz;
      this.dirty = true;
    }
  }

  spring(x0, v0, t) {
    const y = this.y;
    const e = Math.exp(-y * t);
    const j = v0 + y * x0;
    return [(x0 + j * t) * e, (v0 - y * j * t) * e];
  }

  sample(k, out = this.out) {
    const c = this.lib.clips[this.c];
    const g = Math.min(c.frames - 1, this.f + k);
    const tt = this.t + k / this.lib.fps;
    for (let j = 0; j < J; j++) {
      const [x, v] = this.spring(this.dofOff[j], this.dofOffV[j], tt);
      out.dof[j] = c.dof[J * g + j] + x;
      out.dofv[j] = c.dofv[J * g + j] + v;
    }
    const r = [0, 0, 0];
    for (let a = 0; a < 3; a++) r[a] = this.spring(this.rotOff[a], this.rotOffV[a], tt)[0];
    const qc = [c.quat[4 * g], c.quat[4 * g + 1], c.quat[4 * g + 2], c.quat[4 * g + 3]];
    out.quat = quatMul(quatExp(r), quatMul(yawQuat(this.yawOff), qc));
    return out;
  }

  heading() {
    return this.lib.clips[this.c].head[this.f] + this.yawOff;
  }

  advance() {
    this.f += 1;
    this.t += 1 / this.lib.fps;
    this.sinceSearch += 1;
    if (this.dirty || this.sinceSearch >= SEARCH_EVERY) this.search();
  }

  buildQuery() {
    const { lib, query: q } = this;
    const { feat_mean: mu, feat_std: sd, groups, traj_frames: tf } = lib.meta;
    const F = lib.F;
    const c = lib.clips[this.c];
    const row = (c.featStart + Math.min(this.f, c.valid - 1)) * F;
    for (let i = 0; i < F; i++) q[i] = lib.feat[row + i];
    const rv = groups.root_vel[0];
    let vx = q[rv] * sd[rv] + mu[rv];
    let vy = q[rv + 1] * sd[rv + 1] + mu[rv + 1];
    const td = groups.traj_dir[0];
    let wz = Math.atan2(q[td + 1] * sd[td + 1] + mu[td + 1], q[td] * sd[td] + mu[td]) / (tf[0] / lib.fps);
    const { vx: gx, vy: gy, wz: gw } = this.input;
    const dt = 1 / lib.fps;
    const a = 1 - Math.exp(-dt / TRAJ_TAU);
    let x = 0;
    let y = 0;
    let h = 0;
    let s = 0;
    const tp = groups.traj_pos[0];
    for (let k = 1; k <= tf[tf.length - 1]; k++) {
      vx += a * (gx - vx);
      vy += a * (gy - vy);
      wz += a * (gw - wz);
      h += wz * dt;
      x += (Math.cos(h) * vx - Math.sin(h) * vy) * dt;
      y += (Math.sin(h) * vx + Math.cos(h) * vy) * dt;
      if (k === tf[s]) {
        const p = tp + 2 * s;
        q[p] = (x - mu[p]) / sd[p];
        q[p + 1] = (y - mu[p + 1]) / sd[p + 1];
        const d = td + 2 * s;
        q[d] = (Math.cos(h) - mu[d]) / sd[d];
        q[d + 1] = (Math.sin(h) - mu[d + 1]) / sd[d + 1];
        s += 1;
      }
    }
    return q;
  }

  cost(row, q) {
    const { feat, F, w } = this.lib;
    let s = 0;
    const o = row * F;
    for (let i = 0; i < F; i++) {
      const d = feat[o + i] - q[i];
      s += w[i] * d * d;
    }
    return s;
  }

  search() {
    this.dirty = false;
    this.sinceSearch = 0;
    const q = this.buildQuery();
    const lib = this.lib;
    const cur = lib.clips[this.c];
    const contOk = this.f <= cur.valid - 1 - SEARCH_EVERY;
    const contCost = contOk ? this.cost(cur.featStart + this.f, q) : Infinity;
    let best = Infinity;
    let bc = -1;
    let bf = -1;
    for (let ci = 0; ci < lib.clips.length; ci++) {
      const c = lib.clips[ci];
      const last = c.valid - 1 - SEARCH_EVERY;
      for (let f = 0; f <= last; f++) {
        if (ci === this.c && f >= this.f && f - this.f < MIN_JUMP) continue;
        const v = this.cost(c.featStart + f, q);
        if (v < best) {
          best = v;
          bc = ci;
          bf = f;
        }
      }
    }
    if (bc < 0 || contCost <= best + KEEP_BIAS) return;
    this.jump(bc, bf);
  }

  jump(ci, fi) {
    const before0 = this.sample(0, { quat: null, dof: new Float32Array(J), dofv: new Float32Array(J) });
    const before1 = this.sample(1, { quat: null, dof: new Float32Array(J), dofv: new Float32Array(J) });
    const heading = this.heading();
    const c = this.lib.clips[ci];
    this.c = ci;
    this.f = fi;
    this.yawOff = wrap(heading - c.head[fi]);
    this.t = 0;
    for (let j = 0; j < J; j++) {
      this.dofOff[j] = before0.dof[j] - c.dof[J * fi + j];
      this.dofOffV[j] = before0.dofv[j] - c.dofv[J * fi + j];
    }
    const raw = (g) => quatMul(yawQuat(this.yawOff), [c.quat[4 * g], c.quat[4 * g + 1], c.quat[4 * g + 2], c.quat[4 * g + 3]]);
    const r0 = quatLog(quatMul(before0.quat, conj(raw(fi))));
    const r1 = quatLog(quatMul(before1.quat, conj(raw(Math.min(c.frames - 1, fi + 1)))));
    this.rotOff = r0;
    this.rotOffV = [0, 1, 2].map((a) => (r1[a] - r0[a]) * this.lib.fps);
    this.lastSwitch = c.name;
  }
}
