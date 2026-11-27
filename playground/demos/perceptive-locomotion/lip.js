export const DT = 0.02;
const PI = Math.PI;

const CFG = {
  T: 0.4,
  z0: 0.67,
  w: 0.25,
  swingHeight: 0.1,
  comAmp: 0.01,
  accel: [1.0, 0.8, 2.0],
  vmax: 1.0,
  standThreshold: 0.1,
  heightRate: 0.25,
  S: 20,
  reachProgress: 0.5,
  reachK: 32,
  reachLateral: 0.2,
  reachM: 9,
  minStepWidth: 0.1,
  maxStep: 0.65,
  sole: [0.066, 0.142, 0.04],
  edgeClearance: 0.03,
  skipRange: [-0.45, 0.45],
  skipK: 31,
  skipPath: 40,
  soleOffset: 0.037,
  centreOffset: 0.038,
  comOffsetZ: -0.07605,
  halfWidth: 0.1185,
};

const JUMP_TAU = 0.12;
const ARM_SWING = 0.3;
const ARM_AMP_TAU = 0.3;
const ARM_TAU = 0.05;
const TOE = [0.141, 0.0, -0.035];
const MAX_HEEL_LIFT = 0.7;
const HEEL_TOL = 0.005;
const TOE_ARM = 0.15;
const HEEL_ITERS = 4;
const SWING_LIFT_PHASE = 0.35;
const SWING_LAND_PHASE = 0.6;
const SWING_HEEL_RATE = 2.0;
const MIN_KNEE = 0.25;
const PRESWING_PHASE = 0.7;
const PRESWING_KNEE = 0.6;
const KNEE_TO_HEEL = 0.5;
const DROP_RISE = 0.01;
const DROP_FALL = 0.004;
const KNEE_DROP_STEP = 0.004;
const KNEE_COST = 0.02;
const ARM_COLS = [15, 22];
const ANKLE = [4, 10];
const KNEE = [3, 9];

export function setAccelLimit(a) {
  CFG.accel = a.slice();
}

export function wrapToPi(a) {
  let w = (a + PI) % (2 * PI);
  if (w < 0) w += 2 * PI;
  if (w === 0 && a > 0) return PI;
  return w - PI;
}

const rot2 = (x, y, a) => [Math.cos(a) * x - Math.sin(a) * y, Math.sin(a) * x + Math.cos(a) * y];

export function rpyMatrix(r, p, y) {
  const cr = Math.cos(r), sr = Math.sin(r), cp = Math.cos(p), sp = Math.sin(p), cy = Math.cos(y), sy = Math.sin(y);
  return [
    cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
    sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
    -sp, cp * sr, cp * cr,
  ];
}
const mv = (R, v) => [R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2], R[6] * v[0] + R[7] * v[1] + R[8] * v[2]];
function mm(A, B) {
  const C = new Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
  return C;
}
function mtm(A, B) {
  const C = new Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[3 * i + j] = A[i] * B[j] + A[3 + i] * B[3 + j] + A[6 + i] * B[6 + j];
  return C;
}

function bezier(t, T) {
  const s = Math.min(Math.max(t / T, 0), 1);
  return [s * s * (3 - 2 * s), (6 * s * (1 - s)) / T];
}
function swingHeight(t, T, h) {
  t = Math.min(Math.max(t, 0), T);
  if (t < 0.5 * T) {
    const [b, db] = bezier(2 * t, T);
    return [h * b, h * 2 * db];
  }
  const [b, db] = bezier(2 * (T - t), T);
  return [h * b, -h * 2 * db];
}
function upperConvexHull(h) {
  const n = h.length;
  const out = h.slice();
  if (n < 2) return out;
  const s = (k) => k / (n - 1);
  for (let i = 0; i < n - 1; i++) {
    let suf = -Infinity;
    const sl = new Array(n);
    for (let k = n - 1; k > i; k--) {
      const v = (h[k] - h[i]) / (s(k) - s(i));
      if (v > suf) suf = v;
      sl[k] = suf;
    }
    for (let k = i + 1; k < n; k++) {
      const c = h[i] + (s(k) - s(i)) * sl[k];
      if (c > out[k]) out[k] = c;
    }
  }
  return out;
}
function hullAt(hull, frac) {
  const n = hull.length;
  const pos = Math.min(Math.max(frac, 0), 1) * (n - 1);
  const i = Math.min(Math.max(Math.floor(pos), 0), n - 2);
  return [hull[i] + (pos - i) * (hull[i + 1] - hull[i]), (hull[i + 1] - hull[i]) * (n - 1)];
}
function rollPitchFromNormal(nrm, yaw) {
  const l = Math.max(Math.hypot(nrm[0], nrm[1], nrm[2]), 1e-9);
  const n0 = nrm[0] / l, n1 = nrm[1] / l, n2 = nrm[2] / l;
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const nx = c * n0 + s * n1;
  const ny = -s * n0 + c * n1;
  return [-Math.asin(Math.min(Math.max(ny, -1), 1)), Math.atan2(nx, n2)];
}
function eulerRatesToBodyOmega(r, p, dr, dp, dy) {
  const sr = Math.sin(r), cr = Math.cos(r), sp = Math.sin(p), cp = Math.cos(p);
  return [dr - dy * sp, dp * cr + dy * cp * sr, -dp * sr + dy * cp * cr];
}

function fracF32(g, gaitTime, T) {
  const f = Math.fround;
  const t = gaitTime >= T ? f(gaitTime - f(T)) : gaitTime;
  const lam = f(Math.sqrt(9.81 / f(g.z0)));
  const half = f(0.5 * lam);
  const g1 = f(half / f(Math.tanh(f(half * f(T)))));
  const lt = f(lam * t);
  return f(f(-0.5 * f(Math.cosh(lt))) + f(f(g1 * f(Math.sinh(lt))) / lam));
}

class LipGait {
  constructor() {
    this.T = CFG.T;
    this.w = CFG.w;
    this.setZ0(CFG.z0);
  }
  setZ0(z) {
    this.z0 = z;
    this.lam = Math.sqrt(9.81 / z);
    this.ch = Math.cosh(this.lam * this.T);
    this.sh = Math.sinh(this.lam * this.T);
  }
  desiredStep(cmd, left) {
    return [cmd[0] * this.T, cmd[1] * this.T + (left ? -1 : 1) * this.w];
  }
  sagittalGain() {
    return [-0.5, (0.5 * this.lam) / Math.tanh(0.5 * this.lam * this.T)];
  }
  postImpact(cmd, left) {
    const g = this.sagittalGain();
    const ux = cmd[0] * this.T;
    const uy1 = cmd[1] * this.T + this.w;
    const uy2 = cmd[1] * this.T - this.w;
    const y1 = [-0.5 * uy2, (this.lam * (uy1 + this.ch * uy2)) / (2 * this.sh)];
    if (!left) return [ux * g[0], ux * g[1], y1[0], y1[1]];
    const y2 = [this.ch * y1[0] + (this.sh / this.lam) * y1[1] - uy1, this.lam * this.sh * y1[0] + this.ch * y1[1]];
    return [ux * g[0], ux * g[1], y2[0], y2[1]];
  }
  flow(p0, v0, t) {
    const c = Math.cosh(this.lam * t), s = Math.sinh(this.lam * t);
    return [p0 * c + (v0 * s) / this.lam, p0 * this.lam * s + v0 * c];
  }
}

class Generator {
  constructor(course) {
    this.c = course;
    this.T = CFG.T;
    this.gait = new LipGait();
    this.z0Ref = CFG.z0;
    this.heightCmd = CFG.z0;
    this.gaitTime = 0;
    this.left = true;
    this.standing = false;
    this.awaitStep = false;
    this.fresh = true;
    this.cmdEff = [0, 0, 0];
    this.stPos = [0, 0, 0];
    this.stRp = [0, 0];
    this.stYaw = 0;
    this.swPos = [0, 0, 0];
    this.swRpy = [0, 0, 0];
    this.trail = 0;
    this.feetPos = [[0, 0, 0], [0, 0, 0]];
    this.feetRpy = [[0, 0, 0], [0, 0, 0]];
    this.landPos = [0, 0, 0];
    this.landRpy = [0, 0, 0];
  }

  resetFromFeet(feetPos, feetRpy, height) {
    this.feetPos = feetPos.map((p) => p.slice());
    this.feetRpy = feetRpy.map((p) => p.slice());
    this.gaitTime = 0;
    this.left = true;
    this.standing = false;
    this.awaitStep = false;
    this.fresh = true;
    this.cmdEff = [0, 0, 0];
    this.heightCmd = height;
    this.z0Ref = height;
    this.gait.setZ0(height);
    this.snapshot(true, this.feetPos, this.feetRpy);
  }

  get stepTime() {
    return this.gaitTime >= this.T ? this.gaitTime - this.T : this.gaitTime;
  }

  step(cmdIn, height) {
    if (height !== undefined && height !== null) this.heightCmd = height;
    const f = Math.fround;
    for (let i = 0; i < 3; i++) {
      const lim = f(f(CFG.accel[i]) * f(DT));
      this.cmdEff[i] = f(this.cmdEff[i] + Math.max(Math.min(f(f(cmdIn[i]) - this.cmdEff[i]), lim), -lim));
    }
    const cmd = this.cmdEff.slice();
    const [handOver, resume] = this.advanceClock(cmd);
    const fresh = this.fresh;
    this.fresh = false;
    const newStep = handOver || resume || fresh;
    this.land(handOver && !fresh);
    this.snapshot(newStep, this.feetPos, this.feetRpy);
    this.updateHeight(newStep);
    return this.compute(cmd);
  }

  evaluate(cmd) {
    return this.compute(cmd.slice());
  }

  advanceClock(cmd) {
    const T = this.T;
    const f = Math.fround;
    const standingCmd = f(Math.hypot(cmd[0], cmd[1], cmd[2])) < f(CFG.standThreshold);
    let t = f(this.gaitTime + f(DT));
    const crossedHalf = this.gaitTime < T && t >= T - 1e-6;
    const crossedFull = t >= 2 * T - 1e-6;
    const handOver = (crossedHalf || crossedFull) && !this.standing;
    t = crossedFull ? f(t - f(2 * T)) : t;
    t = Math.max(t, 0);
    const startStanding = (handOver || this.fresh) && standingCmd;
    const resume = this.standing && !standingCmd;
    this.standing = (this.standing || startStanding) && !resume;
    this.gaitTime = this.standing || resume ? 0 : t;
    this.left = this.gaitTime < T;
    this.awaitStep = this.awaitStep && !(handOver || resume || this.standing);
    return [handOver, resume];
  }

  land(landed) {
    if (!landed) return;
    const i = this.left ? 0 : 1;
    this.feetPos[i] = this.landPos.slice();
    this.feetRpy[i] = this.landRpy.slice();
  }

  snapshot(mask, feetPos, feetRpy) {
    const st = this.left ? 0 : 1;
    const sw = 1 - st;
    const sp = feetPos[st], wp = feetPos[sw];
    const sr = feetRpy[st], wr = feetRpy[sw];
    const g0 = this.c.height(sp[0], sp[1]);
    const g1 = this.c.height(wp[0], wp[1]);
    const stPos = [sp[0], sp[1], g0 + CFG.soleOffset];
    const yaw = sr[2];
    const rel = [wp[0] - stPos[0], wp[1] - stPos[1], wp[2] - stPos[2]];
    const r = rot2(rel[0], rel[1], -yaw);
    if (!mask) return;
    this.trail = g1 - g0;
    this.stPos = stPos;
    this.stRp = [sr[0], sr[1]];
    this.stYaw = yaw;
    this.swPos = [r[0], r[1], rel[2]];
    this.swRpy = [wr[0], wr[1], wrapToPi(wr[2] - yaw)];
  }

  updateHeight(newStep) {
    const lim = CFG.heightRate * DT;
    this.z0Ref += Math.min(Math.max(this.heightCmd - this.z0Ref, -lim), lim);
    if (newStep) this.gait.setZ0(this.z0Ref);
  }

  project(x, y, yaw, out = {}) {
    const ox = CFG.centreOffset * Math.cos(yaw), oy = CFG.centreOffset * Math.sin(yaw);
    this.c.project(x + ox, y + oy, true, out);
    out.x -= ox;
    out.y -= oy;
    return out;
  }

  reachAhead(cmd, tx, ty, landingYaw, P) {
    const speed = Math.hypot(cmd[0], cmd[1]);
    const heading = this.stYaw + 0.5 * cmd[2] * this.T;
    const s = Math.max(speed, 1e-6);
    const [dx, dy] = rot2(cmd[0] / s, cmd[1] / s, heading);
    const sx = this.stPos[0], sy = this.stPos[1];
    const nominal = (tx - sx) * dx + (ty - sy) * dy;
    const progress = (P.x - sx) * dx + (P.y - sy) * dy;
    const search = progress < CFG.reachProgress * nominal && speed >= CFG.standThreshold && nominal > 0;
    if (!search) return;
    const K = CFG.reachK, M = CFG.reachM;
    const lx = -dy, ly = dx;
    const cs = Math.cos(this.stYaw), sn = Math.sin(this.stYaw);
    let best = Infinity, bx = 0, by = 0, bp = -1;
    const q = {};
    for (let k = 0; k < K; k++) {
      const extra = (CFG.maxStep * (k + 1)) / K;
      for (let m = 0; m < M; m++) {
        const lat = -CFG.reachLateral + (2 * CFG.reachLateral * m) / (M - 1);
        this.project(tx + extra * dx + lat * lx, ty + extra * dy + lat * ly, landingYaw, q);
        const rx = q.x - sx, ry = q.y - sy;
        const latSt = -sn * rx + cs * ry;
        const own = (this.left ? -latSt : latSt) >= CFG.minStepWidth;
        const valid = q.dist < 1e-5 && Math.hypot(rx, ry) <= CFG.maxStep && own;
        if (!valid) continue;
        const cost = Math.hypot(q.x - tx, q.y - ty);
        if (cost < best) {
          best = cost;
          bx = q.x;
          by = q.y;
          bp = q.poly;
        }
      }
    }
    if (bp >= 0) {
      P.x = bx;
      P.y = by;
      P.poly = bp;
      P.dist = Math.hypot(bx - tx, by - ty);
    }
  }

  skips(scx, scy, fx, fy, yaw) {
    const ex = fx + CFG.centreOffset * Math.cos(yaw), ey = fy + CFG.centreOffset * Math.sin(yaw);
    const n = CFG.skipPath;
    const ins = new Array(n);
    for (let i = 0; i < n; i++) {
      const s = i / (n - 1);
      ins[i] = this.c.inside(scx + s * (ex - scx), scy + s * (ey - scy));
    }
    let a = 0;
    while (a < n && ins[a]) a++;
    let b = n - 1;
    while (b >= 0 && ins[b]) b--;
    let count = 0;
    for (let i = 0; i < n; i++) if (ins[i] && i >= a && i <= b) count++;
    return count >= 2;
  }

  avoidSkips(cmd, tx, ty, landingYaw, P) {
    const sx = this.stPos[0], sy = this.stPos[1];
    const scx = sx + CFG.centreOffset * Math.cos(this.stYaw), scy = sy + CFG.centreOffset * Math.sin(this.stYaw);
    if (!this.skips(scx, scy, P.x, P.y, landingYaw)) return;
    const speed = Math.hypot(cmd[0], cmd[1]);
    const heading = this.stYaw + 0.5 * cmd[2] * this.T;
    const s = Math.max(speed, 1e-6);
    let [dx, dy] = rot2(cmd[0] / s, cmd[1] / s, heading);
    if (!(speed >= CFG.standThreshold)) {
      const l = Math.max(Math.hypot(tx - sx, ty - sy), 1e-6);
      dx = (tx - sx) / l;
      dy = (ty - sy) / l;
    }
    const K = CFG.skipK;
    let best = Infinity, bx = 0, by = 0, bp = -1;
    const q = {};
    for (let k = 0; k < K; k++) {
      const lam = CFG.skipRange[0] + ((CFG.skipRange[1] - CFG.skipRange[0]) * k) / (K - 1);
      this.project(tx + lam * dx, ty + lam * dy, landingYaw, q);
      const ok = !this.skips(scx, scy, q.x, q.y, landingYaw) && Math.hypot(q.x - sx, q.y - sy) <= CFG.maxStep;
      if (!ok) continue;
      const cost = Math.hypot(q.x - tx, q.y - ty);
      if (cost < best) {
        best = cost;
        bx = q.x;
        by = q.y;
        bp = q.poly;
      }
    }
    if (bp >= 0) {
      P.x = bx;
      P.y = by;
      P.poly = bp;
    }
  }

  footprintHeights(sx, sy, heights, startYaw, landingYaw, soleZ) {
    const [back, front, half] = CFG.sole;
    const S = sx.length;
    const out = heights.slice();
    const dyaw = wrapToPi(landingYaw - startYaw);
    for (let k = 1; k < S - 1; k++) {
      const yaw = startYaw + (k / (S - 1)) * dyaw;
      const c = Math.cos(yaw), s = Math.sin(yaw);
      let hmax = -Infinity, hmin = Infinity;
      for (let a = 0; a < 4; a++) {
        const lon = -back + ((front + back) * a) / 3;
        for (const lat of [-half, 0, half]) {
          const h = this.c.height(sx[k] + c * lon - s * lat, sy[k] + s * lon + c * lat) - soleZ;
          if (h > hmax) hmax = h;
          if (h < hmin) hmin = h;
        }
      }
      if (hmax - hmin > 0.01) hmax += CFG.edgeClearance;
      out[k] = Math.max(hmax, heights[k]);
    }
    return out;
  }

  compute(cmd) {
    const T = this.T;
    const t = this.stepTime;
    const [b, db] = bezier(t, T);
    const velRatio = cmd[0] / CFG.vmax;
    const yaw0 = this.stYaw;
    const left = this.left;
    const g = this.gait;

    const comYaw = cmd[2] * t;
    const x0 = g.postImpact(cmd, left);
    const [px, vx] = g.flow(x0[0], x0[1], t);
    const [py, vy] = g.flow(x0[2], x0[3], t);
    let [cx, cy] = rot2(px, py, comYaw);
    let [cvx, cvy] = rot2(vx, vy, comYaw);
    const ds = g.desiredStep(cmd, left);
    const stepNom = rot2(ds[0], ds[1], cmd[2] * T);
    const swingYawTarget = cmd[2] * T;
    const landingYaw = yaw0 + swingYawTarget;

    const soleZ = this.stPos[2] - CFG.soleOffset;
    const tw = rot2(stepNom[0], stepNom[1], yaw0);
    const tx = this.stPos[0] + tw[0], ty = this.stPos[1] + tw[1];
    const P = this.project(tx, ty, landingYaw);
    this.reachAhead(cmd, tx, ty, landingYaw, P);
    this.avoidSkips(cmd, tx, ty, landingYaw, P);
    const projectionDistance = Math.hypot(P.x - tx, P.y - ty);
    const step = rot2(P.x - this.stPos[0], P.y - this.stPos[1], -yaw0);
    const [rollT, pitchT] = rollPitchFromNormal(this.c.planeNormal(P.poly), landingYaw);
    const sw0 = rot2(this.swPos[0], this.swPos[1], yaw0);
    const swx = this.stPos[0] + sw0[0], swy = this.stPos[1] + sw0[1];
    const S = CFG.S;
    const sxs = new Array(S), sys = new Array(S);
    let heights = new Array(S);
    for (let k = 0; k < S; k++) {
      const s = k / (S - 1);
      sxs[k] = swx * (1 - s) + P.x * s;
      sys[k] = swy * (1 - s) + P.y * s;
      heights[k] = this.c.height(sxs[k], sys[k]) - soleZ;
    }
    heights = this.footprintHeights(sxs, sys, heights, yaw0 + this.swRpy[2], landingYaw, soleZ);
    const hull = upperConvexHull(heights);
    let inTime = new Array(S);
    let amax = 0;
    for (let j = 0; j < S; j++) {
      const [bk] = bezier((T * j) / (S - 1), T);
      inTime[j] = hullAt(hull, bk)[0];
      if (inTime[j] > inTime[amax]) amax = j;
    }
    const maj = upperConvexHull(inTime);
    inTime = inTime.map((v, j) => (j >= amax ? maj[j] : v));
    const [hullZ, slope] = hullAt(inTime, t / T);
    const hullVz = slope / T;
    const stepHeight = heights[S - 1];
    const footTarget = [P.x, P.y, heights[S - 1] + soleZ];

    const g2 = g.sagittalGain();
    const [frac, dfrac] = g.flow(g2[0], g2[1], t);
    const offx = step[0] - stepNom[0], offy = step[1] - stepNom[1];
    cx += (frac + 0.5) * offx;
    cy += (frac + 0.5) * offy;
    cvx += dfrac * offx;
    cvy += dfrac * offy;

    const swing = [this.swPos[0] * (1 - b) + step[0] * b, this.swPos[1] * (1 - b) + step[1] * b];
    const swingV = [db * (step[0] - this.swPos[0]), db * (step[1] - this.swPos[1])];
    const [bump, dBump] = swingHeight(t, T, CFG.swingHeight);
    const swingZ = bump + hullZ;
    const swingVz = dBump + hullVz;
    const tgt = [rollT, pitchT, swingYawTarget];
    const sRpy = [0, 1, 2].map((i) => tgt[i] * b + (1 - b) * this.swRpy[i]);
    const sdRpy = [0, 1, 2].map((i) => db * (tgt[i] - this.swRpy[i]));

    const omega = (2 * PI) / T;
    const arg = omega * t + PI;
    let comZ = this.z0Ref + CFG.comAmp * Math.abs(velRatio) * (Math.sin(arg) - 1);
    let comVz = CFG.comAmp * Math.abs(velRatio) * omega * Math.cos(arg);
    if (fracF32(this.gait, this.gaitTime, T) < 0) {
      comZ += -frac * this.trail;
      comVz += -dfrac * this.trail;
    } else {
      comZ += frac * stepHeight;
      comVz += dfrac * stepHeight;
    }

    let y = [cx, cy, comZ, 0, 0, comYaw, swing[0], swing[1], swingZ, sRpy[0], sRpy[1], sRpy[2]];
    let dy = [cvx, cvy, comVz, 0, 0, cmd[2], swingV[0], swingV[1], swingVz, ...eulerRatesToBodyOmega(sRpy[0], sRpy[1], sdRpy[0], sdRpy[1], sdRpy[2])];
    const hold = this.standing || this.awaitStep;
    if (hold) {
      y = [0.5 * this.swPos[0], 0.5 * this.swPos[1], this.z0Ref + 0.5 * this.swPos[2], 0, 0, 0.5 * this.swRpy[2],
        this.swPos[0], this.swPos[1], this.swPos[2], this.swRpy[0], this.swRpy[1], this.swRpy[2]];
      dy = new Array(12).fill(0);
    }
    this.landPos = [footTarget[0], footTarget[1], footTarget[2] + CFG.soleOffset];
    this.landRpy = [rollT, pitchT, landingYaw];
    return this.frame(y, dy, hold, footTarget, projectionDistance);
  }

  frame(y, dy, hold, footTarget, projectionDistance) {
    const yaw0 = this.stYaw;
    const sp = this.stPos;
    const c = rot2(y[0], y[1], yaw0);
    const cv = rot2(dy[0], dy[1], yaw0);
    const s = rot2(y[6], y[7], yaw0);
    const sv = rot2(dy[6], dy[7], yaw0);
    const com = [sp[0] + c[0], sp[1] + c[1], sp[2] + y[2]];
    const comVel = [cv[0], cv[1], dy[2]];
    const swing = [sp[0] + s[0], sp[1] + s[1], sp[2] + y[8]];
    const swingVel = [sv[0], sv[1], dy[8]];
    const swingRpy = [y[9], y[10], wrapToPi(yaw0 + y[11])];
    const stanceRpy = [this.stRp[0], this.stRp[1], yaw0];
    const left = this.left;
    return {
      comPos: com,
      comVel,
      pelvisYaw: wrapToPi(yaw0 + y[5]),
      pelvisYawRate: dy[5],
      feetPos: left ? [sp.slice(), swing] : [swing, sp.slice()],
      feetRpy: left ? [stanceRpy, swingRpy] : [swingRpy, stanceRpy],
      feetVel: left ? [[0, 0, 0], swingVel] : [swingVel, [0, 0, 0]],
      left,
      holding: hold,
      contacts: [left || hold ? 1 : 0, !left || hold ? 1 : 0],
      footTarget,
      projectionDistance,
      height: this.z0Ref,
    };
  }
}

function skew(a) {
  return [0, -a[2], a[1], a[2], 0, -a[0], -a[1], a[0], 0];
}

class LegChain {
  constructor(j) {
    this.offset = j.offset;
    this.rotation = j.rotation.map((R) => [...R[0], ...R[1], ...R[2]]);
    this.axis = j.axis;
    this.limits = j.limits;
    this.K = this.axis.map(skew);
    this.KK = this.K.map((K) => mm(K, K));
  }
}

function legFk(ch, q) {
  let pos = [0, 0, 0];
  let rot = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const axes = [], origins = [];
  for (let i = 0; i < 6; i++) {
    const o = mv(rot, ch.offset[i]);
    pos = [pos[0] + o[0], pos[1] + o[1], pos[2] + o[2]];
    rot = mm(rot, ch.rotation[i]);
    axes.push(mv(rot, ch.axis[i]));
    origins.push(pos);
    const s = Math.sin(q[i]), c = 1 - Math.cos(q[i]);
    const K = ch.K[i], KK = ch.KK[i];
    const J = [1 + s * K[0] + c * KK[0], s * K[1] + c * KK[1], s * K[2] + c * KK[2],
      s * K[3] + c * KK[3], 1 + s * K[4] + c * KK[4], s * K[5] + c * KK[5],
      s * K[6] + c * KK[6], s * K[7] + c * KK[7], 1 + s * K[8] + c * KK[8]];
    rot = mm(rot, J);
  }
  return { pos, rot, axes, origins };
}

function rotationError(target, cur) {
  const e = new Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) e[3 * i + j] = target[3 * i] * cur[3 * j] + target[3 * i + 1] * cur[3 * j + 1] + target[3 * i + 2] * cur[3 * j + 2];
  const angle = Math.acos(Math.min(Math.max(0.5 * (e[0] + e[4] + e[8] - 1), -1), 1));
  const v = [e[7] - e[5], e[2] - e[6], e[3] - e[1]];
  const k = angle < 1e-6 ? 0.5 : angle / (2 * Math.max(Math.sin(angle), 1e-9));
  return [k * v[0], k * v[1], k * v[2]];
}

function solve6(A, b) {
  const L = new Float64Array(36);
  for (let i = 0; i < 6; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[6 * i + j];
      for (let k = 0; k < j; k++) s -= L[6 * i + k] * L[6 * j + k];
      L[6 * i + j] = i === j ? Math.sqrt(Math.max(s, 1e-300)) : s / L[6 * j + j];
    }
  }
  const y = new Float64Array(6);
  for (let i = 0; i < 6; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[6 * i + k] * y[k];
    y[i] = s / L[6 * i + i];
  }
  const x = new Float64Array(6);
  for (let i = 5; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < 6; k++) s -= L[6 * k + i] * x[k];
    x[i] = s / L[6 * i + i];
  }
  return x;
}

function evalLeg(ch, q, tp, tr, rw) {
  const f = legFk(ch, q);
  const pe = Math.hypot(tp[0] - f.pos[0], tp[1] - f.pos[1], tp[2] - f.pos[2]);
  const r = rotationError(tr, f.rot);
  const re = Math.hypot(r[0], r[1], r[2]);
  return { f, r, pe, re, cost: pe + rw * re };
}

function legIk(chains, tps, trs, q0s, iters = 40, rw = 0.3, tolerance = 5e-4, damping = 0.03, maxStep = 0.15, checkEvery = 5) {
  const n = chains.length;
  const q = q0s.map((x) => x.slice());
  const best = q.map((x) => ({ q: x.slice(), cost: Infinity, pe: Infinity, re: Infinity }));
  const w = [1, 1, 1, rw, rw, rw];
  const d2 = damping * damping;
  const keep = (i, e) => {
    if (e.cost < best[i].cost) best[i] = { q: q[i].slice(), cost: e.cost, pe: e.pe, re: e.re };
  };
  for (let it = 0; it < iters; it++) {
    if (it > 0 && it % checkEvery === 0 && Math.max(...best.map((b) => b.cost)) < tolerance) break;
    for (let i = 0; i < n; i++) {
      const ch = chains[i];
      const e = evalLeg(ch, q[i], tps[i], trs[i], rw);
      keep(i, e);
      const { pos, axes, origins } = e.f;
      const J = new Float64Array(36);
      for (let k = 0; k < 6; k++) {
        const a = axes[k];
        const d = [pos[0] - origins[k][0], pos[1] - origins[k][1], pos[2] - origins[k][2]];
        const cr = [a[1] * d[2] - a[2] * d[1], a[2] * d[0] - a[0] * d[2], a[0] * d[1] - a[1] * d[0]];
        J[0 * 6 + k] = cr[0] * w[0];
        J[1 * 6 + k] = cr[1] * w[1];
        J[2 * 6 + k] = cr[2] * w[2];
        J[3 * 6 + k] = a[0] * w[3];
        J[4 * 6 + k] = a[1] * w[4];
        J[5 * 6 + k] = a[2] * w[5];
      }
      const tp = tps[i];
      const err = [(tp[0] - pos[0]) * w[0], (tp[1] - pos[1]) * w[1], (tp[2] - pos[2]) * w[2], e.r[0] * w[3], e.r[1] * w[4], e.r[2] * w[5]];
      const A = new Float64Array(36);
      for (let r = 0; r < 6; r++) {
        for (let c = 0; c < 6; c++) {
          let s = 0;
          for (let k = 0; k < 6; k++) s += J[r * 6 + k] * J[c * 6 + k];
          A[r * 6 + c] = s + (r === c ? d2 : 0);
        }
      }
      const y = solve6(A, err);
      for (let k = 0; k < 6; k++) {
        let dq = 0;
        for (let r = 0; r < 6; r++) dq += J[r * 6 + k] * y[r];
        dq = Math.min(Math.max(dq, -maxStep), maxStep);
        q[i][k] = Math.min(Math.max(q[i][k] + dq, ch.limits[k][0]), ch.limits[k][1]);
      }
    }
  }
  for (let i = 0; i < n; i++) keep(i, evalLeg(chains[i], q[i], tps[i], trs[i], rw));
  return best;
}

const Q_INIT = [-0.312, 0.0, 0.0, 0.669, -0.363, 0.0];

class FastLegIK {
  constructor(legs) {
    this.chains = [new LegChain(legs.left), new LegChain(legs.right)];
    this.q = [Q_INIT.slice(), Q_INIT.slice()];
  }
  reset() {
    this.q = [Q_INIT.slice(), Q_INIT.slice()];
  }
  solve(tp, tr) {
    let res = legIk(this.chains, tp, tr, this.q, 8, 0.3, 1e-4);
    const q = res.map((r) => r.q), pe = res.map((r) => r.pe), re = res.map((r) => r.re);
    for (const [seed, rw] of [['init', 0.3], ['best', 0.05]]) {
      const miss = [0, 1].filter((i) => pe[i] > 0.01);
      if (miss.length === 0) break;
      const sub = legIk(miss.map((i) => this.chains[i]), miss.map((i) => tp[i]), miss.map((i) => tr[i]),
        miss.map((i) => (seed === 'init' ? Q_INIT.slice() : q[i])), 12, rw);
      miss.forEach((i, k) => {
        const s = sub[k];
        const better = seed === 'init' ? s.pe + 0.3 * s.re < pe[i] + 0.3 * re[i] - 1e-4 : s.pe < pe[i] - 1e-4;
        if (better) {
          q[i] = s.q;
          pe[i] = s.pe;
          re[i] = s.re;
        }
      });
    }
    this.q = q.map((x) => x.slice());
    return { q: [...q[0], ...q[1]], pe, re };
  }
}

class Motion {
  constructor(legs) {
    this.ik = new FastLegIK(legs);
    this.default = legs.default.slice();
  }
  pelvis(frame) {
    return [frame.comPos[0], frame.comPos[1], frame.comPos[2] - CFG.comOffsetZ];
  }
  pose(frame, { reachFix = false, drop = 0, reachFeet = null, minKnee = 0, tolerance = 2e-3, maxDrop = 0.25 } = {}) {
    const pelvis = this.pelvis(frame);
    const R = rpyMatrix(0, 0, frame.pelvisYaw);
    const rotB = frame.feetRpy.map((r) => mtm(R, rpyMatrix(r[0], r[1], r[2])));
    const qPrev = this.ik.q.map((x) => x.slice());
    const stance = frame.left ? 0 : 1;
    let best = null;
    for (let tr = 0; tr < (reachFix ? 5 : 1); tr++) {
      this.ik.q = qPrev.map((x) => x.slice());
      const base = [pelvis[0], pelvis[1], pelvis[2] - drop];
      const posB = frame.feetPos.map((p) => {
        const d = [p[0] - base[0], p[1] - base[1], p[2] - base[2]];
        return [R[0] * d[0] + R[3] * d[1] + R[6] * d[2], R[1] * d[0] + R[4] * d[1] + R[7] * d[2], R[2] * d[0] + R[5] * d[1] + R[8] * d[2]];
      });
      const s = this.ik.solve(posB, rotB);
      let bent = [Math.max(minKnee - s.q[3], 0), Math.max(minKnee - s.q[9], 0)];
      if (reachFeet) bent = bent.map((v, i) => v * reachFeet[i]);
      const bentMax = Math.max(bent[0], bent[1]);
      const cost = Math.max(s.pe[0], s.pe[1]) + KNEE_COST * bentMax;
      if (best === null || cost < best.cost - 1e-4) best = { cost, q: s.q, pe: s.pe, re: s.re, drop, ikq: this.ik.q.map((x) => x.slice()) };
      let short = reachFeet ? Math.max(s.pe[0] * reachFeet[0], s.pe[1] * reachFeet[1]) : s.pe[stance];
      if (bentMax > 0) short = Math.max(short, KNEE_DROP_STEP);
      if (!reachFix || short <= tolerance) break;
      drop = Math.min(drop + (short > tolerance ? short : 0), maxDrop);
    }
    this.ik.q = best.ikq;
    const dof = this.default.slice();
    for (let k = 0; k < 12; k++) dof[k] = best.q[k];
    const y = frame.pelvisYaw;
    return {
      rootPos: [pelvis[0], pelvis[1], pelvis[2] - best.drop],
      rootQuat: [Math.cos(0.5 * y), 0, 0, Math.sin(0.5 * y)],
      dof,
      pe: best.pe,
      re: best.re,
      drop: best.drop,
    };
  }
}

function heelLifted(feetPos, feetRpy, theta) {
  const pos = [], rpy = [];
  for (let i = 0; i < 2; i++) {
    const r = feetRpy[i];
    const lifted = [r[0], r[1] + theta[i], r[2]];
    const a = mv(rpyMatrix(r[0], r[1], r[2]), TOE);
    const b = mv(rpyMatrix(lifted[0], lifted[1], lifted[2]), TOE);
    const p = feetPos[i];
    pos.push([p[0] + a[0] - b[0], p[1] + a[1] - b[1], p[2] + a[2] - b[2]]);
    rpy.push(lifted);
  }
  return [pos, rpy];
}

export class ReferenceStepper {
  constructor(course, legs) {
    this.c = course;
    this.gen = new Generator(course);
    this.motion = new Motion(legs);
    this.arm0 = [this.motion.default[ARM_COLS[0]], this.motion.default[ARM_COLS[1]]];
    this.ankleLo = legs.left.limits[4][0];
    this.decay = Math.exp(-DT / JUMP_TAU);
    this.aAmp = 1 - Math.exp(-DT / ARM_AMP_TAU);
    this.aArm = 1 - Math.exp(-DT / ARM_TAU);
    this.clear();
  }

  clear() {
    this.first = true;
    this.prevRaw = null;
    this.prevYawRaw = 0;
    this.off = [0, 0, 0];
    this.offYaw = 0;
    this.theta = [0, 0];
    this.amp = 0;
    this.arm = this.arm0.slice();
    this.dropS = 0;
    this.motion.ik.reset();
  }

  reset(x, y, yaw, height = CFG.z0) {
    const half = CFG.halfWidth;
    const lat = [-Math.sin(yaw) * half, Math.cos(yaw) * half];
    const feetXY = [[x + lat[0], y + lat[1]], [x - lat[0], y - lat[1]]];
    const fwd = [Math.cos(yaw) * CFG.centreOffset, Math.sin(yaw) * CFG.centreOffset];
    const pos = [], rpy = [];
    for (const f of feetXY) {
      const P = this.c.project(f[0] + fwd[0], f[1] + fwd[1], false);
      const [roll, pitch] = rollPitchFromNormal(this.c.planeNormal(P.poly), yaw);
      pos.push([f[0], f[1], this.c.planeHeight(P.poly, f[0], f[1]) + CFG.soleOffset]);
      rpy.push([roll, pitch, yaw]);
    }
    this.resetFromFeet(pos, rpy, height);
  }

  resetFromFeet(feetPos, feetRpy, height = CFG.z0) {
    this.gen.resetFromFeet(feetPos, feetRpy, height);
    this.clear();
  }

  step(cmd, height = CFG.z0) {
    const gen = this.gen;
    const motion = this.motion;
    if (this.first) this.dropS = 0;
    const frame = this.first ? gen.evaluate(cmd) : gen.step(cmd, height);
    this.first = false;
    const raw = frame.comPos.slice();
    const yawRaw = frame.pelvisYaw;
    if (this.prevRaw !== null) {
      for (let i = 0; i < 3; i++) this.off[i] = this.decay * (this.off[i] - (raw[i] - this.prevRaw[i] - frame.comVel[i] * DT));
      this.offYaw = this.decay * (this.offYaw - wrapToPi(yawRaw - this.prevYawRaw - frame.pelvisYawRate * DT));
    }
    this.prevRaw = raw;
    this.prevYawRaw = yawRaw;
    frame.comPos = [raw[0] + this.off[0], raw[1] + this.off[1], raw[2] + this.off[2]];
    frame.pelvisYaw = wrapToPi(yawRaw + this.offYaw);
    const stance = frame.contacts.map((c) => c > 0.5);
    const phase = gen.stepTime / gen.T;
    const leaving = stance.map((s) => !s && phase < SWING_LIFT_PHASE);
    const landing = stance.map((s) => !s && phase > SWING_LAND_PHASE);
    const reach = [0, 1].map((i) => stance[i] || leaving[i] || landing[i]);
    let theta = [0, 1].map((i) => (stance[i] ? this.theta[i] : Math.max(this.theta[i] - SWING_HEEL_RATE * DT, 0)));
    const basePos = frame.feetPos.map((p) => p.slice());
    const baseRpy = frame.feetRpy.map((p) => p.slice());
    const fixed = this.dropS;
    let pose = null;
    let knee = [0, 0];
    for (let it = 0; it < HEEL_ITERS; it++) {
      [frame.feetPos, frame.feetRpy] = heelLifted(basePos, baseRpy, theta);
      pose = motion.pose(frame, { reachFix: false, drop: fixed });
      knee = [pose.dof[KNEE[0]], pose.dof[KNEE[1]]];
      let any = false;
      const inc = [0, 0];
      for (let i = 0; i < 2; i++) {
        const atLimit = pose.dof[ANKLE[i]] <= this.ankleLo + 0.02;
        const byRot = stance[i] && atLimit && pose.re[i] > 0.03;
        const byPos = (stance[i] || leaving[i]) && pose.pe[i] > HEEL_TOL;
        let target = PRESWING_KNEE * Math.min(Math.max((phase - PRESWING_PHASE) / (1 - PRESWING_PHASE), 0), 1);
        target = leaving[i] ? PRESWING_KNEE : stance[i] ? target : 0;
        const byKnee = (stance[i] || leaving[i]) && knee[i] < target - 0.02 && theta[i] < MAX_HEEL_LIFT;
        if (byRot || byPos || byKnee) any = true;
        inc[i] = Math.max(byRot ? pose.re[i] : 0, byPos ? pose.pe[i] / TOE_ARM : 0);
        inc[i] = Math.max(inc[i], byKnee ? KNEE_TO_HEEL * (target - knee[i]) : 0);
      }
      if (!any) break;
      theta = theta.map((v, i) => Math.min(v + inc[i], MAX_HEEL_LIFT));
    }
    const short = [0, 1].some((i) => (pose.pe[i] > 0.005 && reach[i]) || (knee[i] < MIN_KNEE && reach[i]));
    if (short || fixed > 0) {
      const need = motion.pose(frame, { reachFix: true, reachFeet: reach.map(Number), minKnee: MIN_KNEE }).drop;
      const nw = Math.min(Math.max(need, fixed - DROP_FALL), fixed + DROP_RISE);
      if (Math.abs(nw - fixed) > 1e-4) pose = motion.pose(frame, { reachFix: false, drop: nw });
      this.dropS = nw;
    }
    for (let i = 0; i < 2; i++) {
      const margin = pose.dof[ANKLE[i]] - this.ankleLo;
      const idle = stance[i] && margin > 0.05 && pose.pe[i] < 0.5 * HEEL_TOL;
      this.theta[i] = idle ? Math.max(theta[i] - Math.min(margin - 0.05, 0.02), 0) : theta[i];
    }
    this.amp += this.aAmp * ((ARM_SWING * cmd[0]) / CFG.vmax - this.amp);
    const swing = this.amp * Math.cos((PI * gen.gaitTime) / gen.T);
    this.arm[0] += this.aArm * (this.arm0[0] + swing - this.arm[0]);
    this.arm[1] += this.aArm * (this.arm0[1] - swing - this.arm[1]);
    const dof = pose.dof.slice();
    dof[ARM_COLS[0]] = this.arm[0];
    dof[ARM_COLS[1]] = this.arm[1];
    return {
      qpos: [...pose.rootPos, ...pose.rootQuat, ...dof],
      footTarget: frame.footTarget,
      contacts: frame.contacts,
      ikErr: pose.pe,
      feet: frame.feetPos,
    };
  }

  save() {
    return structuredClone({ gen: stateOf(this.gen), ik: this.motion.ik.q, s: stateOf(this, ['c', 'gen', 'motion']) });
  }
  load(st) {
    const s = structuredClone(st);
    Object.assign(this.gen, s.gen);
    this.gen.gait.setZ0(s.gen._z0);
    this.motion.ik.q = s.ik;
    Object.assign(this, s.s);
  }
}

function stateOf(o, skip = ['c', 'gait']) {
  const out = {};
  for (const [k, v] of Object.entries(o)) {
    if (skip.includes(k) || typeof v === 'function') continue;
    out[k] = v;
  }
  if (o.gait) out._z0 = o.gait.z0;
  return out;
}
