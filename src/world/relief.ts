import { smoothstep } from '../core/math';
import type { SphereMesh } from '../core/mesh';
import { Simplex3 } from '../core/noise';
import { seedFor } from '../core/rng';
import type { Boundaries } from './tectonics/boundaries';
import { areaQuantile } from './tectonics/init';
import { rawElevation, shelfCompress } from './tectonics/isostasy';
import type { Geology, WorldParams } from './types';

export interface Relief {
  /** Elevation relative to sea level (m), before erosion. */
  elev: Float32Array;
  /** Local relief amplitude (m) for sub-cell detail when rendering. */
  rough: Float32Array;
  /** 1 = atoll on a subsided volcano, 2 = fringing/shelf reef. */
  reef: Uint8Array;
  /** Raw (pre-sea-level) elevations, kept for sea-level fitting. */
  raw: Float32Array;
}

const rateFactor = (r: number): number => Math.min(1.3, Math.max(0.3, r / 50));

/** Uplift/subsidence from active plate boundaries and mantle plumes (m). */
export function boundaryRelief(geo: Geology, bnd: Boundaries, c: number): number {
  let e = 0;
  const dT = bnd.dTrench[c];
  if (dT < 350) {
    const f = rateFactor(bnd.rTrench[c]);
    e += -3200 * f * Math.exp(-((dT / 55) ** 2)) + 350 * f * Math.exp(-(((dT - 160) / 60) ** 2));
  }
  const dA = bnd.dArc[c];
  if (dA < 800) {
    const f = rateFactor(bnd.rArc[c]);
    if (bnd.arcCont[c]) e += 1700 * f * Math.exp(-(((dA - 220) / 110) ** 2));
    else e += 2400 * f * Math.exp(-(((dA - 150) / 70) ** 2));
  }
  const dC = bnd.dColl[c];
  if (dC < 1000) e += 1300 * rateFactor(bnd.rColl[c]) * Math.exp(-((dC / 200) ** 2));
  const dD = bnd.dDiv[c];
  if (dD < 500 && geo.ctype[c]) {
    e += -900 * Math.exp(-((dD / 35) ** 2)) + 500 * Math.exp(-(((dD - 110) / 70) ** 2));
  }
  return e;
}

/** Sea level (m) that leaves `landFraction` of the surface dry. */
export function seaLevelForLandFraction(mesh: SphereMesh, raw: Float32Array, landFraction: number): number {
  return areaQuantile(mesh, raw, landFraction);
}

export function computeRelief(mesh: SphereMesh, geo: Geology, bnd: Boundaries, params: WorldParams): Relief {
  const { n, x, y, z, lat } = mesh;
  const R = params.radiusKm;
  const noise = new Simplex3(seedFor(params.seed, 'relief'));
  const raw = new Float32Array(n);
  const rough = new Float32Array(n);
  const hs = geo.hotspots.filter((h) => h.active);

  for (let c = 0; c < n; c++) {
    const px = x[c];
    const py = y[c];
    const pz = z[c];
    let e = rawElevation(geo.ctype[c], geo.thick[c], geo.age[c], geo.volc[c]) + boundaryRelief(geo, bnd, c);
    for (const h of hs) {
      const dot = px * h.x + py * h.y + pz * h.z;
      if (dot < 0.95) continue;
      const d = Math.acos(Math.min(1, dot)) * R;
      e += (geo.ctype[c] ? 400 : 900) * Math.exp(-((d / 550) ** 2));
    }
    const wx = px + 0.03 * noise.noise(px * 9, py * 9, pz * 9);
    const wy = py + 0.03 * noise.noise(px * 9 + 7.1, py * 9, pz * 9);
    const wz = pz + 0.03 * noise.noise(px * 9, py * 9 + 3.7, pz * 9);
    const broad = noise.fbm(wx * 5, wy * 5, wz * 5, 4);
    const medium = noise.fbm(wx * 14 + 2.2, wy * 14, wz * 14, 3);
    const youth = 0.35 + 0.65 * Math.exp(-geo.orogeny[c] / 120);
    if (e > -1500) {
      const mtn = smoothstep(500, 3000, e);
      const ridge = (noise.ridged(wx * 22, wy * 22, wz * 22, 4) - 0.5) * 1800 * mtn * youth;
      e += 160 * broad + 70 * medium + ridge;
      rough[c] = 25 + 0.08 * Math.max(0, e) * youth + 350 * mtn * youth;
    } else {
      const young = 1 + Math.exp(-geo.age[c] / 20);
      e += (140 * medium + 60 * broad) * young;
      rough[c] = 60 * young;
    }
    raw[c] = e;
  }

  const offset = params.seaLevel;
  const elev = new Float32Array(n);
  const reef = new Uint8Array(n);
  for (let c = 0; c < n; c++) {
    const e = raw[c] - offset;
    elev[c] = geo.ctype[c] && e < 0 ? shelfCompress(e) : e;
  }
  // Coral keeps pace with a sinking volcano, so once the island itself has
  // gone under, its reef survives as an atoll ring at sea level: one per
  // extinct volcano, on the shallowest summits of drowned edifices, and well
  // away from any high island (whose coral grows as a fringing reef instead).
  const atollR = Math.max(1.5 * mesh.spacing, ATOLL_SPACING_KM / R);
  const cosR = Math.cos(atollR);
  const candidates: number[] = [];
  for (let c = 0; c < n; c++) {
    const e = elev[c];
    if (Math.abs(lat[c]) >= TROPICS || geo.ctype[c] || geo.volc[c] <= 0.2 || geo.volcAge[c] < 2 || e >= 0 || e <= -1200) continue;
    if (!landWithin(mesh, elev, c, cosR)) candidates.push(c);
  }
  candidates.sort((a, b) => elev[b] - elev[a]);
  const atolls: number[] = [];
  for (const c of candidates) {
    if (atolls.every((a) => x[a] * x[c] + y[a] * y[c] + z[a] * z[c] < cosR)) atolls.push(c);
  }
  for (let c = 0; c < n; c++) if (Math.abs(lat[c]) < TROPICS && elev[c] < 0 && elev[c] > -60) reef[c] = 2;
  for (const c of atolls) {
    elev[c] = -8;
    reef[c] = 1;
  }
  return { elev, rough, reef, raw };
}

/** Reef-building coral needs warm water: within ~26° of the equator. */
const TROPICS = 0.45;
/** Volcanoes along a hotspot chain stand roughly this far apart (km). */
const ATOLL_SPACING_KM = 90;

/** Whether any land lies within the angular radius whose cosine is `cosR` of cell c. */
function landWithin(mesh: SphereMesh, elev: Float32Array, c: number, cosR: number): boolean {
  const { x, y, z, nbrStart, nbr } = mesh;
  const seen = new Set<number>([c]);
  const stack = [c];
  while (stack.length) {
    const q = stack.pop()!;
    for (let k = nbrStart[q], e = nbrStart[q + 1]; k < e; k++) {
      const m = nbr[k];
      if (seen.has(m)) continue;
      seen.add(m);
      if (x[c] * x[m] + y[c] * y[m] + z[c] * z[m] < cosR) continue;
      if (elev[m] >= 0) return true;
      stack.push(m);
    }
  }
  return false;
}
