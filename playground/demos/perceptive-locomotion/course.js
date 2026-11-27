export class Course {
  constructor(meta, bin) {
    this.meta = meta;
    this.cell = meta.cell;
    this.ox = meta.origin[0];
    this.oy = meta.origin[1];
    [this.nx, this.ny] = meta.shape;
    this.ground = meta.ground;
    const blob = (k, Type) => {
      const b = meta.blobs[k];
      const n = b.shape.reduce((a, c) => a * c, 1);
      return new Type(bin, b.offset, n);
    };
    this.verts = blob('verts', Float32Array);
    this.inset = blob('inset', Float32Array);
    this.planes = blob('planes', Float32Array);
    this.cells = blob('cells', Int16Array);
    [this.gx, this.gy, this.gm] = meta.blobs.cells.shape;
    this.maskBits = blob('mask', Uint8Array);
    [this.mx, this.my] = meta.mask_shape;
    this.h = rasterise(meta);
    this.hs = new Float32Array(this.h.length);
    for (let i = 0; i < this.h.length; i++) this.hs[i] = this.h[i] > 1e-3 ? this.h[i] : -0.5;
  }

  height(x, y) {
    return interp(this.h, this, x, y);
  }

  scanHeight(x, y) {
    return Math.max(0, interp(this.hs, this, x, y));
  }

  project(px, py, inset, out = {}) {
    let cx = Math.floor((px - this.meta.fh_origin[0]) / this.meta.fh_cell);
    let cy = Math.floor((py - this.meta.fh_origin[1]) / this.meta.fh_cell);
    cx = Math.min(Math.max(cx, 0), this.gx - 1);
    cy = Math.min(Math.max(cy, 0), this.gy - 1);
    const base = (cx * this.gy + cy) * this.gm;
    let best = Infinity;
    let bp = -1;
    let bx = 0;
    let by = 0;
    for (let m = 0; m < this.gm; m++) {
      const p = this.cells[base + m];
      if (p < 0) continue;
      projectPolygon(px, py, this.verts, p, _pp);
      if (_pp.dist < best) {
        best = _pp.dist;
        bp = p;
        bx = _pp.x;
        by = _pp.y;
      }
    }
    if (inset) {
      projectPolygon(px, py, this.inset, bp, _pp);
      bx = _pp.x;
      by = _pp.y;
    }
    out.x = bx;
    out.y = by;
    out.poly = bp;
    out.dist = best;
    return out;
  }

  planeHeight(p, x, y) {
    const P = this.planes;
    return P[3 * p] * x + P[3 * p + 1] * y + P[3 * p + 2];
  }

  planeNormal(p) {
    const P = this.planes;
    const a = -P[3 * p];
    const b = -P[3 * p + 1];
    const n = Math.hypot(a, b, 1);
    return [a / n, b / n, 1 / n];
  }

  inside(x, y) {
    const i = Math.floor((x - this.meta.mask_origin[0]) / this.meta.mask_res);
    const j = Math.floor((y - this.meta.mask_origin[1]) / this.meta.mask_res);
    if (i < 0 || i >= this.mx || j < 0 || j >= this.my) return false;
    const k = i * this.my + j;
    return ((this.maskBits[k >> 3] >> (k & 7)) & 1) === 1;
  }
}

const _pp = { x: 0, y: 0, dist: 0 };

function projectPolygon(px, py, V, p, out) {
  const o = 8 * p;
  let inside = true;
  let best = Infinity;
  let cx = 0;
  let cy = 0;
  for (let k = 0; k < 4; k++) {
    const x0 = V[o + 2 * k];
    const y0 = V[o + 2 * k + 1];
    const k1 = (k + 1) & 3;
    const ex = V[o + 2 * k1] - x0;
    const ey = V[o + 2 * k1 + 1] - y0;
    const rx = px - x0;
    const ry = py - y0;
    if (ex * ry - ey * rx < -1e-9) inside = false;
    let t = (rx * ex + ry * ey) / Math.max(ex * ex + ey * ey, 1e-12);
    t = Math.min(Math.max(t, 0), 1);
    const qx = x0 + t * ex;
    const qy = y0 + t * ey;
    const d = (px - qx) * (px - qx) + (py - qy) * (py - qy);
    if (d < best) {
      best = d;
      cx = qx;
      cy = qy;
    }
  }
  if (inside) {
    out.x = px;
    out.y = py;
    out.dist = 0;
  } else {
    out.x = cx;
    out.y = cy;
    out.dist = Math.sqrt(Math.max(best, 0));
  }
}

function interp(h, c, x, y) {
  let u = (x - c.ox) / c.cell;
  let v = (y - c.oy) / c.cell;
  if (!Number.isFinite(u)) u = 0;
  if (!Number.isFinite(v)) v = 0;
  u = Math.min(Math.max(u, 0), c.nx - 1);
  v = Math.min(Math.max(v, 0), c.ny - 1);
  const r = Math.min(Math.floor(u), c.nx - 2);
  const col = Math.min(Math.floor(v), c.ny - 2);
  const fu = u - r;
  const fv = v - col;
  const b = r * c.ny + col;
  const h00 = h[b];
  const h10 = h[b + c.ny];
  const h01 = h[b + 1];
  const h11 = h[b + c.ny + 1];
  if (fu + fv <= 1) return h00 + fu * (h10 - h00) + fv * (h01 - h00);
  return h11 + (1 - fu) * (h01 - h11) + (1 - fv) * (h10 - h11);
}

function rasterise(meta) {
  const [nx, ny] = meta.shape;
  const { cell } = meta;
  const [ox, oy] = meta.origin;
  const h = new Float32Array(nx * ny).fill(meta.ground);
  for (const pc of meta.pieces) {
    const [x0, x1, y0, y1] = pc.r;
    const [a, b, c] = pc.p;
    const i0 = Math.max(0, Math.ceil((x0 - ox) / cell - 1e-9));
    const i1 = Math.min(nx - 1, Math.ceil((x1 - ox) / cell - 1e-9) - 1);
    const j0 = Math.max(0, Math.ceil((y0 - oy) / cell - 1e-9));
    const j1 = Math.min(ny - 1, Math.ceil((y1 - oy) / cell - 1e-9) - 1);
    for (let i = i0; i <= i1; i++) {
      const x = ox + cell * i;
      for (let j = j0; j <= j1; j++) h[i * ny + j] = a * x + b * (oy + cell * j) + c;
    }
  }
  return h;
}

export async function loadCourse(baseUrl, fetchFn = fetch) {
  const [meta, bin] = await Promise.all([
    fetchFn(new URL('course.json', baseUrl)).then((r) => r.json()),
    fetchFn(new URL('course.bin', baseUrl)).then((r) => r.arrayBuffer()),
  ]);
  return new Course(meta, bin);
}
