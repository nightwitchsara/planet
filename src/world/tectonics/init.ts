import { MinHeap } from '../../core/heap';
import { type Vec3, normalize3, smoothstep } from '../../core/math';
import type { SphereMesh } from '../../core/mesh';
import { Simplex3 } from '../../core/noise';
import { Rng, seedFor } from '../../core/rng';
import { Geology, type Hotspot, type Plate, type WorldParams, plateColor, plateName } from '../types';
import { analyzeBoundaries, B_DIVERGENT } from './boundaries';

const HOTSPOT_NAMES = [
  'Ashfall', 'Brimstone', 'Cinderwell', 'Duskfire', 'Emberdeep', 'Furnace', 'Glowmaw', 'Hearth', 'Ignis', 'Juniper',
  'Kiln', 'Lava Spring', 'Moltenreach', 'Novaflow', 'Obsidian', 'Pyre', 'Quench', 'Redcap', 'Smolder', 'Tephra',
  'Umbral', 'Vulcan', 'Wyrmfire', 'Xerxes', 'Yewfire', 'Zenith',
];

export function hotspotName(i: number): string {
  return HOTSPOT_NAMES[i % HOTSPOT_NAMES.length] + (i >= HOTSPOT_NAMES.length ? ` ${Math.floor(i / HOTSPOT_NAMES.length) + 1}` : '');
}

/** Area-weighted threshold: the value above which `fraction` of the sphere lies. */
export function areaQuantile(mesh: SphereMesh, values: ArrayLike<number>, fraction: number): number {
  const n = mesh.n;
  const idx = new Int32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  idx.sort((a, b) => values[b] - values[a]);
  const target = fraction * 4 * Math.PI;
  let acc = 0;
  for (let i = 0; i < n; i++) {
    acc += mesh.area[idx[i]];
    if (acc >= target) return values[idx[i]];
  }
  return -Infinity;
}

/** Great-circle distance (km) from the nearest cell satisfying `isSource`. */
export function distanceFrom(mesh: SphereMesh, isSource: (c: number) => boolean, maxKm: number, radiusKm: number): Float32Array {
  const { n, nbrStart, nbr, nbrDist } = mesh;
  const dist = new Float32Array(n).fill(1e9);
  const heap = new MinHeap(1024);
  for (let c = 0; c < n; c++) {
    if (isSource(c)) {
      dist[c] = 0;
      heap.push(0, c);
    }
  }
  while (heap.size > 0) {
    const c = heap.pop();
    const d = heap.lastKey;
    if (d > dist[c]) continue;
    for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
      const m = nbr[k];
      const nd = d + nbrDist[k] * radiusKm;
      if (nd < dist[m] && nd <= maxKm) {
        dist[m] = nd;
        heap.push(nd, m);
      }
    }
  }
  return dist;
}

function randomTangent(rng: Rng, p: Vec3): Vec3 {
  const v = rng.unitVector();
  const d = v[0] * p[0] + v[1] * p[1] + v[2] * p[2];
  return normalize3([v[0] - d * p[0], v[1] - d * p[1], v[2] - d * p[2]]);
}

/** Weighted, noise-perturbed flood fill: irregular plates with a spread of sizes. */
function growPlates(mesh: SphereMesh, count: number, rng: Rng, noise: Simplex3): { plate: Int16Array; count: number } {
  const { n, x, y, z, nbrStart, nbr, nbrDist } = mesh;
  const plate = new Int16Array(n).fill(-1);
  const tent = new Int16Array(n).fill(-1);
  const cost = new Float64Array(n).fill(Infinity);
  const heap = new MinHeap(n);
  const seeds: number[] = [];
  const minCos = Math.cos(0.55 * Math.sqrt((4 * Math.PI) / count));
  for (let tries = 0; seeds.length < count && tries < count * 500; tries++) {
    const c = rng.int(n);
    let ok = true;
    for (const s of seeds) {
      if (x[s] * x[c] + y[s] * y[c] + z[s] * z[c] > minCos) {
        ok = false;
        break;
      }
    }
    if (ok) seeds.push(c);
  }
  const speed = seeds.map(() => Math.exp(rng.normal() * 0.5));
  seeds.forEach((s, p) => {
    cost[s] = 0;
    tent[s] = p;
    heap.push(0, s);
  });
  while (heap.size > 0) {
    const c = heap.pop();
    const k0 = heap.lastKey;
    if (plate[c] >= 0 || k0 > cost[c]) continue;
    const p = tent[c];
    plate[c] = p;
    const sp = speed[p];
    for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
      const m = nbr[k];
      if (plate[m] >= 0) continue;
      const mx = (x[c] + x[m]) * 0.5;
      const my = (y[c] + y[m]) * 0.5;
      const mz = (z[c] + z[m]) * 0.5;
      const mult = Math.exp(1.0 * noise.fbm(mx * 2.3, my * 2.3, mz * 2.3, 3));
      const nc = k0 + (nbrDist[k] * mult) / sp;
      if (nc < cost[m]) {
        cost[m] = nc;
        tent[m] = p;
        heap.push(nc, m);
      }
    }
  }
  return { plate, count: seeds.length };
}

interface ContinentField {
  ctype: Uint8Array;
  center: Vec3 | null;
}

function placeContinents(mesh: SphereMesh, params: WorldParams, rng: Rng, noise: Simplex3): ContinentField {
  const { n, x, y, z } = mesh;
  const score = new Float32Array(n);
  let center: Vec3 | null = null;
  const centers: { c: Vec3; w: number }[] = [];
  if (params.layout === 'supercontinent') {
    center = rng.unitVector();
  } else if (params.layout === 'continents') {
    const k = 3 + rng.int(3);
    for (let i = 0; i < k; i++) centers.push({ c: rng.unitVector(), w: rng.range(0.8, 1.2) });
  }
  for (let c = 0; c < n; c++) {
    const px = x[c];
    const py = y[c];
    const pz = z[c];
    const wx = px + 0.35 * noise.fbm(px * 1.3 + 11.1, py * 1.3, pz * 1.3, 3);
    const wy = py + 0.35 * noise.fbm(px * 1.3, py * 1.3 + 23.4, pz * 1.3, 3);
    const wz = pz + 0.35 * noise.fbm(px * 1.3, py * 1.3, pz * 1.3 + 37.9, 3);
    const l = Math.hypot(wx, wy, wz);
    const qx = wx / l;
    const qy = wy / l;
    const qz = wz / l;
    const detail = 0.24 * noise.fbm(qx * 2.6, qy * 2.6, qz * 2.6, 4) + 0.08 * noise.fbm(qx * 7, qy * 7, qz * 7, 3);
    let s: number;
    if (center) {
      s = qx * center[0] + qy * center[1] + qz * center[2] + detail;
    } else if (centers.length) {
      s = -Infinity;
      for (const k of centers) s = Math.max(s, k.w * (qx * k.c[0] + qy * k.c[1] + qz * k.c[2]));
      s += detail;
    } else {
      s = noise.fbm(qx * 1.7 + 3.3, qy * 1.7, qz * 1.7, 5);
    }
    score[c] = s;
  }
  const thr = areaQuantile(mesh, score, params.continentalFraction);
  const ctype = new Uint8Array(n);
  for (let c = 0; c < n; c++) ctype[c] = score[c] >= thr ? 1 : 0;
  return { ctype, center };
}

function assignMotions(
  mesh: SphereMesh,
  geo: Geology,
  center: Vec3 | null,
  rng: Rng,
  radiusKm: number,
): void {
  const { n, x, y, z, area } = mesh;
  const P = geo.plates.length;
  const tot = new Float64Array(P);
  const cont = new Float64Array(P);
  const cen = new Float64Array(P * 3);
  const ccen = new Float64Array(P * 3);
  for (let c = 0; c < n; c++) {
    const p = geo.plate[c];
    tot[p] += area[c];
    cen[3 * p] += x[c] * area[c];
    cen[3 * p + 1] += y[c] * area[c];
    cen[3 * p + 2] += z[c] * area[c];
    if (geo.ctype[c]) {
      cont[p] += area[c];
      ccen[3 * p] += x[c] * area[c];
      ccen[3 * p + 1] += y[c] * area[c];
      ccen[3 * p + 2] += z[c] * area[c];
    }
  }
  for (let p = 0; p < P; p++) {
    const contFrac = tot[p] > 0 ? cont[p] / tot[p] : 0;
    const anchor = normalize3(
      cont[p] > 0 ? [ccen[3 * p], ccen[3 * p + 1], ccen[3 * p + 2]] : [cen[3 * p], cen[3 * p + 1], cen[3 * p + 2]],
    );
    let dir: Vec3;
    let speed: number;
    if (center && contFrac > 0.15) {
      // Continental plates drift away from the supercontinent's center (breakup).
      const d = anchor[0] * center[0] + anchor[1] * center[1] + anchor[2] * center[2];
      const away: Vec3 = [anchor[0] * d - center[0], anchor[1] * d - center[1], anchor[2] * d - center[2]];
      dir = Math.hypot(...away) < 0.05 ? randomTangent(rng, anchor) : normalize3(away);
      const turn = rng.range(-1.0, 1.0);
      const pxd: Vec3 = [
        anchor[1] * dir[2] - anchor[2] * dir[1],
        anchor[2] * dir[0] - anchor[0] * dir[2],
        anchor[0] * dir[1] - anchor[1] * dir[0],
      ];
      dir = normalize3([
        dir[0] * Math.cos(turn) + pxd[0] * Math.sin(turn),
        dir[1] * Math.cos(turn) + pxd[1] * Math.sin(turn),
        dir[2] * Math.cos(turn) + pxd[2] * Math.sin(turn),
      ]);
      speed = rng.range(15, 38);
    } else {
      dir = randomTangent(rng, anchor);
      speed = contFrac > 0.15 ? rng.range(12, 35) : rng.range(35, 85);
    }
    const axis = normalize3([
      anchor[1] * dir[2] - anchor[2] * dir[1],
      anchor[2] * dir[0] - anchor[0] * dir[2],
      anchor[0] * dir[1] - anchor[1] * dir[0],
    ]);
    const w = speed / radiusKm;
    geo.plates[p].wx = axis[0] * w;
    geo.plates[p].wy = axis[1] * w;
    geo.plates[p].wz = axis[2] * w;
  }
}

/** Seafloor age from distance to spreading ridges at their spreading rates. */
function assignOceanAges(mesh: SphereMesh, geo: Geology, noise: Simplex3, radiusKm: number): void {
  const { n, x, y, z, nbrStart, nbr, nbrDist } = mesh;
  const bnd = analyzeBoundaries(mesh, geo, radiusKm);
  const t = new Float64Array(n).fill(Infinity);
  const v = new Float32Array(n);
  const heap = new MinHeap(1024);
  for (let c = 0; c < n; c++) {
    if (bnd.type[c] === B_DIVERGENT && !geo.ctype[c]) {
      t[c] = 0;
      v[c] = Math.max(10, -bnd.rate[c] / 2);
      heap.push(0, c);
    }
  }
  while (heap.size > 0) {
    const c = heap.pop();
    const tc = heap.lastKey;
    if (tc > t[c]) continue;
    for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
      const m = nbr[k];
      if (geo.ctype[m] || geo.plate[m] !== geo.plate[c]) continue;
      const nt = tc + (nbrDist[k] * radiusKm) / v[c];
      if (nt < t[m] && nt < 220) {
        t[m] = nt;
        v[m] = v[c];
        heap.push(nt, m);
      }
    }
  }
  for (let c = 0; c < n; c++) {
    if (geo.ctype[c]) continue;
    geo.age[c] = Number.isFinite(t[c]) ? t[c] : 90 + 70 * (0.5 + 0.5 * noise.noise(x[c] * 2, y[c] * 2, z[c] * 2));
  }
}

/** Geological state at the start of simulated history. */
export function initialGeology(mesh: SphereMesh, params: WorldParams): Geology {
  const R = params.radiusKm;
  const rng = new Rng(seedFor(params.seed, 'tectonics'));
  const noise = new Simplex3(seedFor(params.seed, 'tectonic-noise'));
  const geo = new Geology(mesh.n);
  const { x, y, z } = mesh;

  const grown = growPlates(mesh, params.plates, rng, noise);
  geo.plate.set(grown.plate);
  for (let p = 0; p < grown.count; p++) {
    const pl: Plate = { id: p, wx: 0, wy: 0, wz: 0, color: plateColor(p), name: plateName(p), alive: true };
    geo.plates.push(pl);
  }

  const cont = placeContinents(mesh, params, rng, noise);
  geo.ctype.set(cont.ctype);

  // Continental structure: thinned passive margins, slightly thicker cratonic
  // cores, and old eroded mountain belts (sutures of earlier collisions).
  const dCoast = distanceFrom(mesh, (c) => geo.ctype[c] === 0, 4000, R);
  for (let c = 0; c < mesh.n; c++) {
    const px = x[c];
    const py = y[c];
    const pz = z[c];
    if (geo.ctype[c]) {
      const d = dCoast[c];
      let t = 35 + 1.5 * smoothstep(400, 1600, d);
      if (d < 320) t = 21 + (t - 21) * Math.pow(d / 320, 0.7);
      const r = noise.ridged(px * 1.5 + 5.5, py * 1.5 - 2.1, pz * 1.5 + 9.7, 4);
      const belt = smoothstep(0.55, 0.85, r) * smoothstep(250, 700, d);
      t += 22 * belt;
      geo.thick[c] = t;
      geo.orogeny[c] = belt > 0.1 ? 180 + 420 * (0.5 + 0.5 * noise.noise(px * 0.9, py * 0.9, pz * 0.9)) : 1500;
      geo.age[c] = 1500 + 1500 * (0.5 + 0.5 * noise.noise(px * 1.1 + 7, py * 1.1, pz * 1.1));
    } else {
      geo.thick[c] = 7;
      geo.orogeny[c] = 1e4;
    }
  }

  assignMotions(mesh, geo, cont.center, rng, R);
  assignOceanAges(mesh, geo, noise, R);

  for (let i = 0; i < params.hotspots; i++) {
    const v = rng.unitVector();
    const h: Hotspot = {
      id: i,
      x: v[0],
      y: v[1],
      z: v[2],
      strength: rng.range(2.5, 7.5),
      radius: rng.range(90, 170) / R,
      name: hotspotName(i),
      active: true,
    };
    geo.hotspots.push(h);
  }
  return geo;
}
