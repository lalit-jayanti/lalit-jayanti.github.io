import * as THREE from 'three';

const REGIONS = ['Bridge', 'Slopes', 'Curbs', 'Blocks', 'Center'];
const WAVE = 5;
const REFILL_AT = 2;
const SPACING = 1.3;
const LIFT = 0.85;
const REACH_XY = 0.45;
const REACH_Z = 0.75;
const GLOW = '#ffcf6b';

function haloTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grd.addColorStop(0, 'rgba(255,214,120,0.85)');
  grd.addColorStop(0.35, 'rgba(255,200,90,0.35)');
  grd.addColorStop(1, 'rgba(255,190,80,0)');
  g.fillStyle = grd;
  g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function starGeometry() {
  const s = new THREE.Shape();
  for (let i = 0; i < 10; i++) {
    const r = i % 2 === 0 ? 0.16 : 0.068;
    const a = Math.PI / 2 + (i * Math.PI) / 5;
    if (i === 0) s.moveTo(r * Math.cos(a), r * Math.sin(a));
    else s.lineTo(r * Math.cos(a), r * Math.sin(a));
  }
  s.closePath();
  const geo = new THREE.ExtrudeGeometry(s, { depth: 0.04, bevelEnabled: true, bevelThickness: 0.016, bevelSize: 0.014, bevelSegments: 3 });
  geo.center();
  geo.rotateX(Math.PI / 2);
  return geo;
}

function regionOf(course, x, y) {
  for (const z of course.meta.zones) {
    const [x0, x1, y0, y1] = z.r;
    if (x >= x0 && x <= x1 && y >= y0 && y <= y1) return z.label;
  }
  return '';
}

function standHeight(course, x, y) {
  let h = -Infinity;
  for (const dx of [-0.3, 0, 0.3]) for (const dy of [-0.3, 0, 0.3]) h = Math.max(h, course.height(x + dx, y + dy));
  return h;
}

function candidateSpots(course) {
  const spots = [];
  const loop = course.meta.loop;
  for (let k = 0; k < loop.length; k++) {
    const [ax, ay] = loop[k];
    const [bx, by] = loop[(k + 1) % loop.length];
    const L = Math.hypot(bx - ax, by - ay);
    const ux = (bx - ax) / L, uy = (by - ay) / L;
    for (let s = 1.6; s < L - 1.4; s += 0.75) {
      for (const v of [-0.55, 0, 0.55]) spots.push([ax + s * ux - v * uy, ay + s * uy + v * ux]);
    }
  }
  const c = course.meta.zones.find((z) => z.label === 'Center');
  if (c) {
    const [x0, x1, y0, y1] = c.r;
    for (let x = x0 + 0.8; x < x1 - 0.5; x += 0.9) for (let y = y0 + 0.8; y < y1 - 0.5; y += 0.9) spots.push([x, y]);
  }
  const out = {};
  for (const [x, y] of spots) {
    const r = regionOf(course, x, y);
    if (!REGIONS.includes(r)) continue;
    (out[r] ||= []).push({ x, y, z: standHeight(course, x, y) + LIFT });
  }
  return out;
}

export class Stars {
  constructor(scene, course, { onCollect, onWave } = {}) {
    this.scene = scene;
    this.course = course;
    this.onCollect = onCollect;
    this.onWave = onWave;
    this.spots = candidateSpots(course);
    this.group = new THREE.Group();
    scene.add(this.group);
    this.geo = starGeometry();
    this.mat = new THREE.MeshStandardMaterial({ color: '#ffd77a', emissive: '#ffb53a', emissiveIntensity: 1.5, roughness: 0.28, metalness: 0.15, transparent: true });
    this.haloTex = haloTexture();
    this.haloMat = new THREE.SpriteMaterial({ map: this.haloTex, transparent: true, depthWrite: false, fog: false, opacity: 0.95 });
    this.lights = [0, 1].map(() => {
      const l = new THREE.PointLight(GLOW, 0, 2.6, 1.6);
      this.group.add(l);
      return l;
    });
    this.sparkMat = new THREE.PointsMaterial({ map: this.haloTex, color: GLOW, size: 0.16, transparent: true, depthWrite: false, opacity: 1 });
    this.stars = [];
    this.bursts = [];
    this.score = 0;
    this.next = 0;
    this.time = 0;
    this.spawnWave(REGIONS[0]);
  }

  reset() {
    for (const s of this.stars) this.group.remove(s.obj);
    for (const b of this.bursts) this.group.remove(b.obj);
    this.stars = [];
    this.bursts = [];
    this.score = 0;
    this.next = 0;
    this.spawnWave(REGIONS[0]);
  }

  get remaining() {
    return this.stars.filter((s) => s.state !== 'collect').length;
  }

  spawnWave(region) {
    const pool = (this.spots[region] || []).slice();
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    const placed = this.stars.map((s) => s.spot);
    let n = 0;
    for (const p of pool) {
      if (n >= WAVE) break;
      if (placed.some((q) => Math.hypot(q.x - p.x, q.y - p.y) < SPACING)) continue;
      placed.push(p);
      this.add(p, n * 0.12);
      n++;
    }
    this.next = (REGIONS.indexOf(region) + 1) % REGIONS.length;
    if (this.onWave) this.onWave(region, n);
  }

  add(spot, delay) {
    const obj = new THREE.Group();
    const mesh = new THREE.Mesh(this.geo, this.mat.clone());
    mesh.castShadow = true;
    const halo = new THREE.Sprite(this.haloMat.clone());
    halo.scale.set(0.8, 0.8, 1);
    obj.add(halo, mesh);
    obj.position.set(spot.x, spot.y, spot.z);
    obj.scale.setScalar(0.001);
    this.group.add(obj);
    this.stars.push({ spot, obj, mesh, halo, state: 'spawn', t: -delay, phase: Math.random() * 6.28 });
  }

  burst(p) {
    const n = 28;
    const pos = new Float32Array(3 * n);
    const vel = [];
    for (let i = 0; i < n; i++) {
      pos.set([p.x, p.y, p.z], 3 * i);
      const a = Math.random() * 6.28, e = (Math.random() - 0.2) * 1.4;
      const sp = 0.9 + Math.random() * 1.2;
      vel.push([sp * Math.cos(a) * Math.cos(e), sp * Math.sin(a) * Math.cos(e), sp * Math.sin(e) + 0.6]);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const obj = new THREE.Points(geo, this.sparkMat.clone());
    this.group.add(obj);
    this.bursts.push({ obj, vel, t: 0 });
  }

  update(dt, pelvis, focus) {
    this.time += dt;
    const keep = [];
    for (const s of this.stars) {
      s.t += dt;
      const o = s.obj;
      s.mesh.rotation.z += dt * 1.8;
      const bob = 0.05 * Math.sin(2.2 * this.time + s.phase);
      if (s.state === 'spawn') {
        const u = Math.min(Math.max(s.t / 0.45, 0), 1);
        const k = 1.7;
        o.scale.setScalar(Math.max(0.001, 1 + (k + 1) * Math.pow(u - 1, 3) + k * Math.pow(u - 1, 2)));
        o.position.z = s.spot.z + bob;
        if (u >= 1) s.state = 'idle';
      } else if (s.state === 'idle') {
        o.position.z = s.spot.z + bob;
        s.halo.material.opacity = 0.8 + 0.15 * Math.sin(3 * this.time + s.phase);
        if (pelvis && Math.hypot(pelvis[0] - s.spot.x, pelvis[1] - s.spot.y) < REACH_XY && Math.abs(pelvis[2] - s.spot.z) < REACH_Z) {
          s.state = 'collect';
          s.t = 0;
          this.score += 1;
          this.burst(o.position);
          if (this.onCollect) this.onCollect(this.score);
          if (this.remaining <= REFILL_AT) this.refill(pelvis);
        }
      } else if (s.state === 'collect') {
        const u = Math.min(s.t / 0.45, 1);
        s.mesh.rotation.z += dt * 10 * (1 - u);
        o.scale.setScalar(1 + 0.9 * u);
        o.position.z = s.spot.z + 0.45 * u * (2 - u);
        s.mesh.material.opacity = 1 - u;
        s.halo.material.opacity = 1.2 * (1 - u);
        if (u >= 1) {
          this.group.remove(o);
          s.mesh.material.dispose();
          s.halo.material.dispose();
          continue;
        }
      }
      keep.push(s);
    }
    this.stars = keep;

    const bk = [];
    for (const b of this.bursts) {
      b.t += dt;
      const p = b.obj.geometry.attributes.position;
      for (let i = 0; i < p.count; i++) {
        const v = b.vel[i];
        v[2] -= 2.5 * dt;
        p.setXYZ(i, p.getX(i) + v[0] * dt, p.getY(i) + v[1] * dt, p.getZ(i) + v[2] * dt);
      }
      p.needsUpdate = true;
      b.obj.material.opacity = Math.max(0, 1 - b.t / 0.8);
      if (b.t >= 0.8) {
        this.group.remove(b.obj);
        b.obj.geometry.dispose();
        b.obj.material.dispose();
        continue;
      }
      bk.push(b);
    }
    this.bursts = bk;

    const live = this.stars.filter((s) => s.state !== 'collect');
    if (focus) live.sort((a, b) => Math.hypot(a.spot.x - focus.x, a.spot.y - focus.y) - Math.hypot(b.spot.x - focus.x, b.spot.y - focus.y));
    this.lights.forEach((l, i) => {
      const s = live[i];
      l.intensity = s ? 1.6 * Math.min(1, s.obj.scale.x) : 0;
      if (s) l.position.copy(s.obj.position);
    });
  }

  refill(pelvis) {
    const here = pelvis ? regionOf(this.course, pelvis[0], pelvis[1]) : '';
    for (let k = 0; k < REGIONS.length; k++) {
      const r = REGIONS[(this.next + k) % REGIONS.length];
      if (r === here) continue;
      if (this.stars.some((s) => s.state !== 'collect' && regionOf(this.course, s.spot.x, s.spot.y) === r)) continue;
      this.spawnWave(r);
      return;
    }
  }

  dispose() {
    this.scene.remove(this.group);
    this.group.traverse((o) => {
      if (o.material && o.material !== this.mat) o.material.dispose();
    });
    this.geo.dispose();
    this.mat.dispose();
    this.haloMat.dispose();
    this.sparkMat.dispose();
    this.haloTex.dispose();
  }
}
