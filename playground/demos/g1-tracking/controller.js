// Whole-body tracking controller for the Unitree G1: builds the policy inputs from the simulated
// robot, runs the ONNX policy at 50 Hz and drives the joints with a PD law at every physics step.
// Runs unchanged in the browser and in Node (mujoco and onnxruntime are passed in).

const J = 29;
const OBS_DIM = 3 * J + 6;

function quatMul(a, b) {
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
}

// Row-major 3x3 rotation matrix of a wxyz quaternion.
function quatToR(q) {
  const [w, x, y, z] = q;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ];
}

export function yawOf(q) {
  const [w, x, y, z] = q;
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

function yawQuat(yaw) {
  return [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)];
}

export class Controller {
  constructor({ mujoco, model, data, config, session, ort }) {
    this.mj = mujoco;
    this.model = model;
    this.data = data;
    this.cfg = config;
    this.session = session;
    this.ort = ort;
    this.q0 = Float64Array.from(config.q0);
    this.kp = Float64Array.from(config.kp);
    this.kd = Float64Array.from(config.kd);
    this.scale = Float64Array.from(config.action_scale);
    this.qIdx = config.qpos_idx;
    this.vIdx = config.qvel_idx;
    const sensor = (name) => {
      const id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR.value, name);
      if (id < 0) throw new Error(`sensor ${name} not found in the model`);
      return model.sensor_adr[id];
    };
    this.gyroAdr = sensor('imu_gyro');
    this.quatAdr = sensor('imu_quat');
    // Joint armature including the PD damping term kd * dt, set after compilation.
    for (let j = 0; j < J; j++) model.dof_armature[this.vIdx[j]] = config.armature[j];
    const [lo, hi] = config.stiffness_range;
    this.logCenter = 0.5 * (Math.log(lo) + Math.log(hi));
    this.logHalf = 0.5 * (Math.log(hi) - Math.log(lo));
    this.hist = new Float32Array(config.obs_history * OBS_DIM);
    this.goal = new Float32Array(config.goal_frames * (2 * J + 6));
    this.prevAction = new Float32Array(J);
    this.action = new Float32Array(J);
    this.target = Float64Array.from(this.q0);
    this.stiffness = 150;
    this.clip = null;
    this.walk = null; // keyboard-walk motion matcher, used instead of a clip when set
    this.frame = 0;
    this.yaw = 0;
    this.fresh = true;
    // Extra wrenches on bodies (6 per body, world frame, at the body CoM), set by the UI each physics step.
    this.applyExternal = null;
  }

  setClip(clip) {
    this.clip = clip;
    this.walk = null;
    this.restart();
  }

  setWalk(matcher) {
    this.walk = matcher;
    this.restart();
  }

  // Start the clip from frame 0 (or the walk from standing), turned to face the robot's current heading.
  restart() {
    this.frame = 0;
    this.prevAction.fill(0);
    this.fresh = true;
    if (this.walk) this.walk.reset(yawOf(this.imuQuat()));
    else this.yaw = yawOf(this.imuQuat()) - yawOf(this.clip.root_quat[0]);
  }

  // Place the robot standing at the origin in its default pose, at rest.
  resetRobot(startQpos) {
    const d = this.data;
    d.qpos.fill(0);
    d.qvel.fill(0);
    d.qfrc_applied.fill(0);
    d.xfrc_applied.fill(0);
    if (startQpos) {
      for (let i = 0; i < startQpos.length; i++) d.qpos[i] = startQpos[i];
      d.qpos[0] = 0;
      d.qpos[1] = 0;
      d.qpos[3] = 1; d.qpos[4] = 0; d.qpos[5] = 0; d.qpos[6] = 0;
    }
    this.mj.mj_forward(this.model, d);
    this.target = Float64Array.from(this.q0);
  }

  // Pelvis IMU orientation (wxyz), as the robot's IMU reports it.
  imuQuat() {
    const s = this.data.sensordata;
    const a = this.quatAdr;
    return [s[a], s[a + 1], s[a + 2], s[a + 3]];
  }

  rootQuat() {
    const p = this.data.qpos;
    return [p[3], p[4], p[5], p[6]];
  }

  observation() {
    const d = this.data;
    const R = quatToR(this.imuQuat());
    const o = new Float32Array(OBS_DIM);
    for (let j = 0; j < J; j++) {
      o[j] = d.qpos[this.qIdx[j]] - this.q0[j];
      o[J + j] = d.qvel[this.vIdx[j]];
    }
    // Pelvis gyro: angular velocity in the body frame.
    o[2 * J] = d.sensordata[this.gyroAdr];
    o[2 * J + 1] = d.sensordata[this.gyroAdr + 1];
    o[2 * J + 2] = d.sensordata[this.gyroAdr + 2];
    // Gravity direction in the body frame: R^T (0, 0, -1).
    o[2 * J + 3] = -R[6];
    o[2 * J + 4] = -R[7];
    o[2 * J + 5] = -R[8];
    o.set(this.prevAction, 2 * J + 6);
    return o;
  }

  buildGoal() {
    const Rr = quatToR(this.imuQuat());
    if (this.walk) {
      for (let k = 0; k < this.cfg.goal_frames; k++) {
        const s = this.walk.sample(this.cfg.goal_stride * k);
        this.fillGoal(k, s.dof, s.dofv, quatToR(s.quat), Rr);
      }
      return this.goal;
    }
    const c = this.clip;
    const n = c.frames;
    for (let k = 0; k < this.cfg.goal_frames; k++) {
      let t = this.frame + this.cfg.goal_stride * k;
      const lap = Math.floor(t / n);
      t -= lap * n;
      const yaw = this.yaw + lap * c.lap_yaw;
      const qRef = quatMul(yawQuat(yaw), c.root_quat[t]);
      this.fillGoal(k, c.dof_pos[t], c.dof_vel[t], quatToR(qRef), Rr);
    }
    return this.goal;
  }

  fillGoal(k, dp, dv, Rf, Rr) {
    const g = this.goal;
    const base = k * (2 * J + 6);
    for (let j = 0; j < J; j++) {
      g[base + j] = dp[j];
      g[base + J + j] = dv[j];
    }
    // rel = Rr^T Rf; first two columns, row-major: [r00, r01, r10, r11, r20, r21]
    for (let r = 0; r < 3; r++) {
      for (let col = 0; col < 2; col++) {
        g[base + 2 * J + 2 * r + col] =
          Rr[0 * 3 + r] * Rf[0 * 3 + col] + Rr[1 * 3 + r] * Rf[1 * 3 + col] + Rr[2 * 3 + r] * Rf[2 * 3 + col];
      }
    }
  }

  async act() {
    const obs = this.observation();
    const H = this.cfg.obs_history;
    if (this.fresh) {
      for (let h = 0; h < H; h++) this.hist.set(obs, h * OBS_DIM);
      this.fresh = false;
    } else {
      this.hist.copyWithin(0, OBS_DIM);
      this.hist.set(obs, (H - 1) * OBS_DIM);
    }
    this.buildGoal();
    const cmd = (Math.log(this.stiffness) - this.logCenter) / this.logHalf;
    const ort = this.ort;
    const out = await this.session.run({
      obs_hist: new ort.Tensor('float32', this.hist, [1, H, OBS_DIM]),
      goal: new ort.Tensor('float32', this.goal, [1, this.goal.length]),
      command: new ort.Tensor('float32', new Float32Array([cmd]), [1, 1]),
    });
    const a = out.action.data;
    let finite = true;
    for (let j = 0; j < J; j++) {
      if (!Number.isFinite(a[j])) finite = false;
      this.action[j] = a[j];
    }
    if (!finite) this.action.fill(0);
    for (let j = 0; j < J; j++) this.target[j] = this.q0[j] + this.scale[j] * this.action[j];
    return this.action;
  }

  // Advance one control period: PD torques recomputed at every physics step.
  physics() {
    const d = this.data;
    const m = this.model;
    for (let s = 0; s < this.cfg.substeps; s++) {
      for (let j = 0; j < J; j++) {
        const q = d.qpos[this.qIdx[j]];
        const dq = d.qvel[this.vIdx[j]];
        d.qfrc_applied[this.vIdx[j]] = this.kp[j] * (this.target[j] - q) - this.kd[j] * dq;
      }
      if (this.applyExternal) this.applyExternal(d);
      this.mj.mj_step(m, d);
    }
  }

  advance() {
    this.prevAction.set(this.action);
    if (this.walk) {
      this.walk.advance();
      return;
    }
    this.frame += 1;
    if (this.frame >= this.clip.frames) {
      this.frame -= this.clip.frames;
      this.yaw += this.clip.lap_yaw;
    }
  }

  async step() {
    await this.act();
    this.physics();
    this.advance();
  }

  // Pelvis height and tilt from vertical (degrees), for fall detection.
  posture() {
    const R = quatToR(this.rootQuat());
    return { height: this.data.qpos[2], tilt: (Math.acos(Math.max(-1, Math.min(1, R[8]))) * 180) / Math.PI };
  }
}
