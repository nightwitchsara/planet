import { MinHeap } from '../../core/heap';
import type { SphereMesh } from '../../core/mesh';
import type { Geology } from '../types';

export const B_NONE = 0;
export const B_CONVERGENT = 1;
export const B_DIVERGENT = 2;
export const B_TRANSFORM = 3;

export const ROLE_NONE = 0;
export const ROLE_OVERRIDING = 1;
export const ROLE_SUBDUCTING = -1;
export const ROLE_COLLISION = 2;

/** Present-day plate-boundary analysis (derived from plates + motions + crust). */
export interface Boundaries {
  type: Uint8Array;
  /** Convergence rate (km/Myr = mm/yr); negative for divergence. */
  rate: Float32Array;
  shear: Float32Array;
  other: Int16Array;
  role: Int8Array;
  /** Distance (km) from the trench, measured within the overriding plate. */
  dArc: Float32Array;
  rArc: Float32Array;
  arcCont: Uint8Array;
  /** Distance (km) from the trench, measured within the subducting plate. */
  dTrench: Float32Array;
  rTrench: Float32Array;
  dColl: Float32Array;
  rColl: Float32Array;
  dDiv: Float32Array;
  rDiv: Float32Array;
}

const FAR = 1e9;

/**
 * Multi-source Dijkstra that only spreads within the plate of each source.
 * Returns distances (km) and the source cell that reached each cell.
 */
export function plateDistance(
  mesh: SphereMesh,
  plate: Int16Array,
  sources: number[],
  maxKm: number,
  radiusKm: number,
): { dist: Float32Array; src: Int32Array } {
  const { n, nbrStart, nbr, nbrDist } = mesh;
  const dist = new Float32Array(n).fill(FAR);
  const src = new Int32Array(n).fill(-1);
  const heap = new MinHeap(Math.max(16, sources.length * 4));
  for (const s of sources) {
    dist[s] = 0;
    src[s] = s;
    heap.push(0, s);
  }
  while (heap.size > 0) {
    const c = heap.pop();
    const d = heap.lastKey;
    if (d > dist[c]) continue;
    const pc = plate[c];
    for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
      const m = nbr[k];
      if (plate[m] !== pc) continue;
      const nd = d + nbrDist[k] * radiusKm;
      if (nd < dist[m] && nd <= maxKm) {
        dist[m] = nd;
        src[m] = src[c];
        heap.push(nd, m);
      }
    }
  }
  return { dist, src };
}

export function analyzeBoundaries(mesh: SphereMesh, geo: Geology, radiusKm: number): Boundaries {
  const { n, x, y, z, nbrStart, nbr } = mesh;
  const { plate, ctype, age } = geo;
  const plates = geo.plates;

  // Neighborhood-averaged crust character keeps subduction polarity stable
  // against single-cell noise along the boundary.
  const cont = new Float32Array(n);
  const ageS = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    let sc = ctype[c];
    let sa = ctype[c] ? 0 : age[c];
    let cnt = 1;
    let ocnt = ctype[c] ? 0 : 1;
    for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
      const m = nbr[k];
      if (plate[m] !== plate[c]) continue;
      sc += ctype[m];
      cnt++;
      if (!ctype[m]) {
        sa += age[m];
        ocnt++;
      }
    }
    cont[c] = sc / cnt;
    ageS[c] = ocnt ? sa / ocnt : 200;
  }

  const type = new Uint8Array(n);
  const rate = new Float32Array(n);
  const shear = new Float32Array(n);
  const other = new Int16Array(n).fill(-1);
  const role = new Int8Array(n);
  const partner = new Int32Array(n).fill(-1);
  const arcSrc: number[] = [];
  const trenchSrc: number[] = [];
  const collSrc: number[] = [];
  const divSrc: number[] = [];

  for (let c = 0; c < n; c++) {
    const pc = plate[c];
    if (pc < 0) continue;
    const A = plates[pc];
    let bestAbs = -1;
    let bestConv = 0;
    let bestShear = 0;
    let bestM = -1;
    for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
      const m = nbr[k];
      const pm = plate[m];
      if (pm === pc || pm < 0) continue;
      const B = plates[pm];
      let mx = x[c] + x[m];
      let my = y[c] + y[m];
      let mz = z[c] + z[m];
      const ml = Math.hypot(mx, my, mz);
      mx /= ml;
      my /= ml;
      mz /= ml;
      const dwx = B.wx - A.wx;
      const dwy = B.wy - A.wy;
      const dwz = B.wz - A.wz;
      // relative velocity of B with respect to A at the midpoint
      const rx = dwy * mz - dwz * my;
      const ry = dwz * mx - dwx * mz;
      const rz = dwx * my - dwy * mx;
      let dx = x[m] - x[c];
      let dy = y[m] - y[c];
      let dz = z[m] - z[c];
      const dl = Math.hypot(dx, dy, dz);
      dx /= dl;
      dy /= dl;
      dz /= dl;
      const along = rx * dx + ry * dy + rz * dz;
      const conv = -along * radiusKm;
      const sh = Math.hypot(rx - along * dx, ry - along * dy, rz - along * dz) * radiusKm;
      if (Math.abs(conv) > bestAbs) {
        bestAbs = Math.abs(conv);
        bestConv = conv;
        bestShear = sh;
        bestM = m;
      }
    }
    if (bestM < 0) continue;
    other[c] = plate[bestM];
    rate[c] = bestConv;
    shear[c] = bestShear;
    partner[c] = bestM;
  }

  // Ragged cell-scale boundaries make the local normal noisy; average the
  // motion over nearby cells of the same boundary before classifying.
  const sRate = new Float32Array(n);
  const sShear = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    if (partner[c] < 0) continue;
    let sr = rate[c];
    let ss = shear[c];
    let cnt = 1;
    for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
      const m = nbr[k];
      for (let k2 = nbrStart[m], e2 = nbrStart[m + 1]; k2 < e2; k2++) {
        const q = nbr[k2];
        if (q === c || partner[q] < 0 || plate[q] !== plate[c] || other[q] !== other[c]) continue;
        sr += rate[q];
        ss += shear[q];
        cnt++;
      }
    }
    sRate[c] = sr / cnt;
    sShear[c] = ss / cnt;
  }

  for (let c = 0; c < n; c++) {
    const bestM = partner[c];
    if (bestM < 0) continue;
    rate[c] = sRate[c];
    shear[c] = sShear[c];
    const conv = sRate[c];
    if (Math.abs(conv) < 0.5 * sShear[c] || Math.abs(conv) < 4) {
      type[c] = B_TRANSFORM;
    } else if (conv > 0) {
      type[c] = B_CONVERGENT;
      const cC = cont[c] >= 0.5;
      const cM = cont[bestM] >= 0.5;
      if (cC && cM) role[c] = ROLE_COLLISION;
      else if (cC) role[c] = ROLE_OVERRIDING;
      else if (cM) role[c] = ROLE_SUBDUCTING;
      else role[c] = ageS[c] > ageS[bestM] ? ROLE_SUBDUCTING : ROLE_OVERRIDING;
      if (role[c] === ROLE_COLLISION) collSrc.push(c);
      else if (role[c] === ROLE_OVERRIDING) arcSrc.push(c);
      else trenchSrc.push(c);
    } else {
      type[c] = B_DIVERGENT;
      divSrc.push(c);
    }
  }

  const arc = plateDistance(mesh, plate, arcSrc, 900, radiusKm);
  const tr = plateDistance(mesh, plate, trenchSrc, 400, radiusKm);
  const co = plateDistance(mesh, plate, collSrc, 1200, radiusKm);
  const dv = plateDistance(mesh, plate, divSrc, 600, radiusKm);
  const rArc = new Float32Array(n);
  const arcCont = new Uint8Array(n);
  const rTrench = new Float32Array(n);
  const rColl = new Float32Array(n);
  const rDiv = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    if (arc.src[c] >= 0) {
      rArc[c] = rate[arc.src[c]];
      arcCont[c] = cont[arc.src[c]] >= 0.5 ? 1 : 0;
    }
    if (tr.src[c] >= 0) rTrench[c] = rate[tr.src[c]];
    if (co.src[c] >= 0) rColl[c] = rate[co.src[c]];
    if (dv.src[c] >= 0) rDiv[c] = -rate[dv.src[c]];
  }
  return {
    type, rate, shear, other, role,
    dArc: arc.dist, rArc, arcCont,
    dTrench: tr.dist, rTrench,
    dColl: co.dist, rColl,
    dDiv: dv.dist, rDiv,
  };
}
