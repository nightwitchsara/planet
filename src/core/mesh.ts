import Delaunator from 'delaunator';
import { Rng } from './rng';

/**
 * Spherical Delaunay triangulation of jittered Fibonacci points. Each point is a
 * cell (its Voronoi region); triangles are used for interpolation and point
 * location. All positions are unit vectors with +z as the polar axis.
 */
export interface SphereMesh {
  n: number;
  x: Float64Array;
  y: Float64Array;
  z: Float64Array;
  lat: Float32Array;
  lon: Float32Array;
  /** Voronoi cell area in steradians (sums to 4π). */
  area: Float32Array;
  /** CSR adjacency, neighbors ordered around each cell. */
  nbrStart: Int32Array;
  nbr: Int32Array;
  /** Arc distance to each neighbor (radians). */
  nbrDist: Float32Array;
  /** Length of the shared Voronoi edge (radians). */
  nbrEdge: Float32Array;
  nTri: number;
  /** Triangle vertex ids, counter-clockwise seen from outside. */
  tri: Int32Array;
  /** Opposite halfedge for each halfedge (3 per triangle). */
  halfedge: Int32Array;
  /** Circumcenters (Voronoi vertices), 3 per triangle. */
  cc: Float64Array;
  /** One triangle touching each cell, used to seed walks. */
  cellTri: Int32Array;
  /** Mean neighbor distance in radians. */
  spacing: number;
  gridW: number;
  gridH: number;
  gridCell: Int32Array;
}

export const nextHalfedge = (e: number): number => (e % 3 === 2 ? e - 2 : e + 1);

function sphTriArea(
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
): number {
  const det = ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx);
  const d = 1 + (ax * bx + ay * by + az * bz) + (bx * cx + by * cy + bz * cz) + (cx * ax + cy * ay + cz * az);
  return Math.abs(2 * Math.atan2(Math.abs(det), d));
}

export function buildSphereMesh(n: number, seed: number, jitter = 0.45): SphereMesh {
  const rng = new Rng(seed);
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  const z = new Float64Array(n);
  const golden = Math.PI * (3 - Math.sqrt(5));
  const spacing0 = Math.sqrt((4 * Math.PI) / n);
  for (let i = 0; i < n; i++) {
    const zz = 1 - (2 * (i + 0.5)) / n;
    const r = Math.sqrt(Math.max(0, 1 - zz * zz));
    const th = golden * i;
    let px = r * Math.cos(th);
    let py = r * Math.sin(th);
    let pz = zz;
    if (jitter > 0 && i !== n - 1) {
      let ox = rng.next() * 2 - 1;
      let oy = rng.next() * 2 - 1;
      let oz = rng.next() * 2 - 1;
      const d = ox * px + oy * py + oz * pz;
      ox -= d * px;
      oy -= d * py;
      oz -= d * pz;
      const ol = Math.hypot(ox, oy, oz) || 1;
      const mag = jitter * spacing0 * 0.5 * Math.sqrt(rng.next());
      px += (ox / ol) * mag;
      py += (oy / ol) * mag;
      pz += (oz / ol) * mag;
      const l = Math.hypot(px, py, pz);
      px /= l;
      py /= l;
      pz /= l;
    }
    x[i] = px;
    y[i] = py;
    z[i] = pz;
  }

  // Stereographic projection from the last point; the planar Delaunay of the rest
  // plus a fan around the projection point is the spherical Delaunay.
  const P = n - 1;
  const Px = x[P];
  const Py = y[P];
  const Pz = z[P];
  let ux: number, uy: number, uz: number;
  if (Math.abs(Pz) < 0.9) {
    // u = normalize(P × ẑ)
    ux = Py;
    uy = -Px;
    uz = 0;
  } else {
    // u = normalize(P × x̂)
    ux = 0;
    uy = Pz;
    uz = -Py;
  }
  const ul = Math.hypot(ux, uy, uz);
  ux /= ul;
  uy /= ul;
  uz /= ul;
  const vx = Py * uz - Pz * uy;
  const vy = Pz * ux - Px * uz;
  const vz = Px * uy - Py * ux;
  const coords = new Float64Array(2 * (n - 1));
  for (let i = 0; i < n - 1; i++) {
    const denom = 1 - (x[i] * Px + y[i] * Py + z[i] * Pz);
    coords[2 * i] = (x[i] * ux + y[i] * uy + z[i] * uz) / denom;
    coords[2 * i + 1] = (x[i] * vx + y[i] * vy + z[i] * vz) / denom;
  }
  const del = new Delaunator(coords);
  const t2 = del.triangles;
  const hull = del.hull;
  const nTri = t2.length / 3 + hull.length;
  const tri = new Int32Array(nTri * 3);
  for (let i = 0; i < t2.length; i++) tri[i] = t2[i];
  let k = t2.length;
  for (let i = 0; i < hull.length; i++) {
    tri[k++] = P;
    tri[k++] = hull[(i + 1) % hull.length];
    tri[k++] = hull[i];
  }
  // Make every triangle counter-clockwise when seen from outside.
  const flipped = new Uint8Array(nTri);
  for (let t = 0; t < nTri; t++) {
    const a = tri[3 * t];
    const b = tri[3 * t + 1];
    const c = tri[3 * t + 2];
    const det =
      x[a] * (y[b] * z[c] - z[b] * y[c]) + y[a] * (z[b] * x[c] - x[b] * z[c]) + z[a] * (x[b] * y[c] - y[b] * x[c]);
    if (det < 0) {
      tri[3 * t + 1] = c;
      tri[3 * t + 2] = b;
      flipped[t] = 1;
    }
  }

  // Halfedge pairing. Delaunator pairs the planar triangles' edges already;
  // a flipped triangle (a, c, b) holds its edges a–b and c–a in swapped slots.
  const nHe = nTri * 3;
  const halfedge = new Int32Array(nHe).fill(-1);
  const moved = (e: number): number => {
    const j = e % 3;
    return flipped[(e - j) / 3] && j !== 1 ? e - j + 2 - j : e;
  };
  const dHe = del.halfedges;
  for (let e = 0; e < t2.length; e++) if (dHe[e] >= 0) halfedge[moved(e)] = moved(dHe[e]);
  // The rest (the planar hull's edges and the fan around the projection
  // point, a few hundred) are paired by their end points.
  const open = new Map<number, number>();
  let unpaired = 0;
  for (let e = 0; e < nHe; e++) {
    if (halfedge[e] >= 0) continue;
    const a = tri[e];
    const b = tri[nextHalfedge(e)];
    const back = open.get(b * n + a);
    if (back === undefined) open.set(a * n + b, e);
    else {
      halfedge[e] = back;
      halfedge[back] = e;
      open.delete(b * n + a);
    }
  }
  unpaired = open.size;
  if (unpaired > 0) console.warn(`sphere mesh: ${unpaired} unpaired halfedges`);

  // Circumcenters.
  const cc = new Float64Array(nTri * 3);
  for (let t = 0; t < nTri; t++) {
    const a = tri[3 * t];
    const b = tri[3 * t + 1];
    const c = tri[3 * t + 2];
    const e1x = x[b] - x[a];
    const e1y = y[b] - y[a];
    const e1z = z[b] - z[a];
    const e2x = x[c] - x[a];
    const e2y = y[c] - y[a];
    const e2z = z[c] - z[a];
    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * e2x - e1x * e2z;
    let nz = e1x * e2y - e1y * e2x;
    const l = Math.hypot(nx, ny, nz) || 1;
    nx /= l;
    ny /= l;
    nz /= l;
    cc[3 * t] = nx;
    cc[3 * t + 1] = ny;
    cc[3 * t + 2] = nz;
  }

  // Adjacency (CSR), walking the halfedges around each point.
  const incoming = new Int32Array(n).fill(-1);
  for (let e = 0; e < nHe; e++) incoming[tri[nextHalfedge(e)]] = e;
  const nbrStart = new Int32Array(n + 1);
  for (let p = 0; p < n; p++) {
    const e0 = incoming[p];
    let e = e0;
    let cnt = 0;
    do {
      cnt++;
      e = halfedge[nextHalfedge(e)];
    } while (e !== e0 && e !== -1 && cnt < 64);
    nbrStart[p + 1] = nbrStart[p] + cnt;
  }
  const nAdj = nbrStart[n];
  const nbr = new Int32Array(nAdj);
  const nbrDist = new Float32Array(nAdj);
  const nbrEdge = new Float32Array(nAdj);
  const area = new Float32Array(n);
  const cellTri = new Int32Array(n);
  let distSum = 0;
  for (let p = 0; p < n; p++) {
    const e0 = incoming[p];
    cellTri[p] = Math.floor(e0 / 3);
    let e = e0;
    let k2 = nbrStart[p];
    let a = 0;
    do {
      const q = tri[e];
      const t1 = Math.floor(e / 3);
      const t2i = Math.floor(halfedge[e] / 3);
      nbr[k2] = q;
      const d = Math.acos(Math.min(1, x[p] * x[q] + y[p] * y[q] + z[p] * z[q]));
      nbrDist[k2] = d;
      distSum += d;
      nbrEdge[k2] = Math.acos(
        Math.min(1, cc[3 * t1] * cc[3 * t2i] + cc[3 * t1 + 1] * cc[3 * t2i + 1] + cc[3 * t1 + 2] * cc[3 * t2i + 2]),
      );
      a += sphTriArea(
        x[p], y[p], z[p],
        cc[3 * t1], cc[3 * t1 + 1], cc[3 * t1 + 2],
        cc[3 * t2i], cc[3 * t2i + 1], cc[3 * t2i + 2],
      );
      k2++;
      e = halfedge[nextHalfedge(e)];
    } while (e !== e0 && e !== -1 && k2 < nbrStart[p + 1]);
    area[p] = a;
  }
  let areaSum = 0;
  for (let p = 0; p < n; p++) areaSum += area[p];
  const areaScale = (4 * Math.PI) / areaSum;
  for (let p = 0; p < n; p++) area[p] *= areaScale;

  const lat = new Float32Array(n);
  const lon = new Float32Array(n);
  for (let p = 0; p < n; p++) {
    lat[p] = Math.asin(Math.max(-1, Math.min(1, z[p])));
    lon[p] = Math.atan2(y[p], x[p]);
  }

  const gridH = Math.max(16, Math.min(256, Math.round(Math.sqrt(n / 8))));
  const gridW = gridH * 2;
  const mesh: SphereMesh = {
    n, x, y, z, lat, lon, area, nbrStart, nbr, nbrDist, nbrEdge, nTri, tri, halfedge, cc, cellTri,
    spacing: distSum / nAdj,
    gridW, gridH, gridCell: new Int32Array(gridW * gridH),
  };
  // Lookup grid: nearest cell to each bin center, found by walking from the previous bin.
  let start = 0;
  for (let j = 0; j < gridH; j++) {
    const la = Math.PI / 2 - ((j + 0.5) * Math.PI) / gridH;
    for (let i = 0; i < gridW; i++) {
      const lo = -Math.PI + ((i + 0.5) * 2 * Math.PI) / gridW;
      const c = Math.cos(la);
      start = walkNearest(mesh, c * Math.cos(lo), c * Math.sin(lo), Math.sin(la), i === 0 && j > 0 ? mesh.gridCell[(j - 1) * gridW] : start);
      mesh.gridCell[j * gridW + i] = start;
    }
  }
  return mesh;
}

function walkNearest(mesh: SphereMesh, px: number, py: number, pz: number, start: number): number {
  const { x, y, z, nbrStart, nbr } = mesh;
  let c = start;
  let best = x[c] * px + y[c] * py + z[c] * pz;
  for (;;) {
    let next = -1;
    for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
      const m = nbr[k];
      const d = x[m] * px + y[m] * py + z[m] * pz;
      if (d > best) {
        best = d;
        next = m;
      }
    }
    if (next < 0) return c;
    c = next;
  }
}

/** The lookup grid's cell for the bin holding the unit vector p: a cell near p. */
function gridCellAt(mesh: SphereMesh, px: number, py: number, pz: number): number {
  const la = Math.asin(Math.max(-1, Math.min(1, pz)));
  const lo = Math.atan2(py, px);
  let j = Math.floor(((Math.PI / 2 - la) / Math.PI) * mesh.gridH);
  let i = Math.floor(((lo + Math.PI) / (2 * Math.PI)) * mesh.gridW);
  j = Math.max(0, Math.min(mesh.gridH - 1, j));
  i = Math.max(0, Math.min(mesh.gridW - 1, i));
  return mesh.gridCell[j * mesh.gridW + i];
}

/**
 * A triangle near the unit vector p, to start locateTriangle from. Cells are
 * numbered along a Fibonacci spiral, so consecutive cells lie far apart: a
 * walk from the last cell's triangle would cross half the globe each time.
 */
export function gridTriangle(mesh: SphereMesh, px: number, py: number, pz: number): number {
  return mesh.cellTri[gridCellAt(mesh, px, py, pz)];
}

/** Cell whose center is closest to the unit vector (px, py, pz). */
export function nearestCell(mesh: SphereMesh, px: number, py: number, pz: number, start = -1): number {
  return walkNearest(mesh, px, py, pz, start < 0 ? gridCellAt(mesh, px, py, pz) : start);
}

export function nearestCellLatLon(mesh: SphereMesh, lat: number, lon: number): number {
  const c = Math.cos(lat);
  return nearestCell(mesh, c * Math.cos(lon), c * Math.sin(lon), Math.sin(lat));
}

/** Triangle containing the unit vector p, found by a visibility walk. */
export function locateTriangle(mesh: SphereMesh, px: number, py: number, pz: number, startTri: number): number {
  const { x, y, z, tri, halfedge } = mesh;
  let t = startTri;
  for (let iter = 0; iter < 100000; iter++) {
    const base = 3 * t;
    let moved = false;
    for (let s = 0; s < 3; s++) {
      const edge = base + ((s + iter) % 3);
      const a = tri[edge];
      const b = tri[nextHalfedge(edge)];
      // det(a, b, p) = p · (a × b); negative means p lies outside this edge.
      const d =
        px * (y[a] * z[b] - z[a] * y[b]) + py * (z[a] * x[b] - x[a] * z[b]) + pz * (x[a] * y[b] - y[a] * x[b]);
      if (d < 0) {
        t = Math.floor(halfedge[edge] / 3);
        moved = true;
        break;
      }
    }
    if (!moved) return t;
  }
  return t;
}

/** Barycentric weights of p within triangle t (written to out[0..2]). */
export function triWeights(mesh: SphereMesh, t: number, px: number, py: number, pz: number, out: Float64Array): void {
  const { x, y, z, tri } = mesh;
  const a = tri[3 * t];
  const b = tri[3 * t + 1];
  const c = tri[3 * t + 2];
  const wa = px * (y[b] * z[c] - z[b] * y[c]) + py * (z[b] * x[c] - x[b] * z[c]) + pz * (x[b] * y[c] - y[b] * x[c]);
  const wb = px * (y[c] * z[a] - z[c] * y[a]) + py * (z[c] * x[a] - x[c] * z[a]) + pz * (x[c] * y[a] - y[c] * x[a]);
  const wc = px * (y[a] * z[b] - z[a] * y[b]) + py * (z[a] * x[b] - x[a] * z[b]) + pz * (x[a] * y[b] - y[a] * x[b]);
  const s = wa + wb + wc;
  if (s > 1e-18) {
    out[0] = Math.max(0, wa / s);
    out[1] = Math.max(0, wb / s);
    out[2] = Math.max(0, wc / s);
  } else {
    out[0] = 1;
    out[1] = 0;
    out[2] = 0;
  }
}

/**
 * Cells within an angular radius of a center cell (breadth-first, by true
 * great-circle distance). Returns parallel arrays of cell ids and distances.
 */
export function cellsWithin(mesh: SphereMesh, center: number, radius: number): { cells: number[]; dist: number[] } {
  const { x, y, z, nbrStart, nbr } = mesh;
  const cx = x[center];
  const cy = y[center];
  const cz = z[center];
  const cosR = Math.cos(radius);
  const cells = [center];
  const dist = [0];
  const seen = new Set<number>([center]);
  for (let i = 0; i < cells.length; i++) {
    const c = cells[i];
    for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
      const m = nbr[k];
      if (seen.has(m)) continue;
      seen.add(m);
      const d = x[m] * cx + y[m] * cy + z[m] * cz;
      if (d >= cosR) {
        cells.push(m);
        dist.push(Math.acos(Math.min(1, d)));
      }
    }
  }
  return { cells, dist };
}

/**
 * Interpolation map from the cells of mesh `from` onto mesh `to`: for each
 * `from` cell, the containing `to` triangle's vertices and weights.
 */
export interface MeshMap {
  idx: Int32Array; // 3 per source cell
  w: Float32Array; // 3 per source cell
  nearest: Int32Array; // 1 per source cell
}

export function buildMeshMap(from: SphereMesh, to: SphereMesh): MeshMap {
  const idx = new Int32Array(from.n * 3);
  const w = new Float32Array(from.n * 3);
  const nearest = new Int32Array(from.n);
  const tmp = new Float64Array(3);
  for (let i = 0; i < from.n; i++) {
    const t = locateTriangle(to, from.x[i], from.y[i], from.z[i], gridTriangle(to, from.x[i], from.y[i], from.z[i]));
    triWeights(to, t, from.x[i], from.y[i], from.z[i], tmp);
    let best = 0;
    for (let k = 0; k < 3; k++) {
      idx[3 * i + k] = to.tri[3 * t + k];
      w[3 * i + k] = tmp[k];
      if (tmp[k] > tmp[best]) best = k;
    }
    nearest[i] = to.tri[3 * t + best];
  }
  return { idx, w, nearest };
}

/** Interpolate a per-cell field of the target mesh back onto the source cells. */
export function sampleField(map: MeshMap, field: ArrayLike<number>, out: Float32Array, stride = 1, offset = 0): Float32Array {
  const { idx, w } = map;
  const n = out.length;
  for (let i = 0; i < n; i++) {
    const j = 3 * i;
    out[i] =
      field[idx[j] * stride + offset] * w[j] +
      field[idx[j + 1] * stride + offset] * w[j + 1] +
      field[idx[j + 2] * stride + offset] * w[j + 2];
  }
  return out;
}
