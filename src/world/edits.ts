import { MinHeap } from '../core/heap';
import { bearingOf, compassName, DEG, latLonToVec, rotationFromOmega, smoothstep, type Vec3 } from '../core/math';
import { cellsWithin, nearestCell, nearestCellLatLon } from '../core/mesh';
import { MONTHS } from './climate';
import { hotspotName } from './tectonics/init';
import { ageForDepth, oceanDepth, rawElevation, shelfExpand, thickForRaw } from './tectonics/isostasy';
import { compactPlates, TectonicSim } from './tectonics/sim';
import type { Hotspot } from './types';
import type { DeriveFrom, World } from './world';

export interface LatLon {
  lat: number; // degrees
  lon: number; // degrees
}

export type IslandCause = 'auto' | 'hotspot' | 'arc' | 'atoll' | 'microcontinent' | 'shelf' | 'sediment';

export interface CauseOption {
  id: Exclude<IslandCause, 'auto'>;
  label: string;
  score: number;
  reason: string;
}

export interface EditReport {
  title: string;
  /** Plain-language account of which cause was changed and why. */
  summary: string;
  details: string[];
  options?: CauseOption[];
  chosen?: IslandCause;
  derive: DeriveFrom;
  failed?: boolean;
}

const km2 = (world: World, sr: number) => sr * world.params.radiusKm ** 2;
const fmtKm2 = (v: number) => (v >= 1e6 ? `${(v / 1e6).toFixed(2)} million km²` : `${Math.round(v).toLocaleString('en-US')} km²`);
const fmtKm = (v: number) => `${Math.round(v).toLocaleString('en-US')} km`;
const cmyr = (kmPerMyr: number) => `${(kmPerMyr / 10).toFixed(1)} cm/yr`;

function cellVec(world: World, c: number): Vec3 {
  return [world.fine.x[c], world.fine.y[c], world.fine.z[c]];
}

/** Brush footprint: cell → weight (1 at the center, fading at the rim). */
export function footprint(world: World, points: LatLon[], radiusKm: number): Map<number, number> {
  const mesh = world.fine;
  const rad = radiusKm / world.params.radiusKm;
  const out = new Map<number, number>();
  for (const p of points) {
    const v = latLonToVec(p.lat * DEG, p.lon * DEG);
    const c0 = nearestCellLatLon(mesh, p.lat * DEG, p.lon * DEG);
    const { cells } = cellsWithin(mesh, c0, rad * 1.05);
    for (const c of cells) {
      const d = Math.acos(Math.min(1, mesh.x[c] * v[0] + mesh.y[c] * v[1] + mesh.z[c] * v[2]));
      const w = 1 - smoothstep(0.55, 1, d / rad);
      if (w > (out.get(c) ?? 0)) out.set(c, w);
    }
  }
  return out;
}

/** Dijkstra from `sources` to the nearest cell matching `pred` (within maxKm). */
function nearestWhere(world: World, sources: number[], pred: (c: number) => boolean, maxKm: number): { cell: number; km: number } | null {
  const mesh = world.fine;
  const R = world.params.radiusKm;
  const dist = new Map<number, number>();
  const heap = new MinHeap(256);
  for (const s of sources) {
    dist.set(s, 0);
    heap.push(0, s);
  }
  while (heap.size > 0) {
    const c = heap.pop();
    const d = heap.lastKey;
    if (d > (dist.get(c) ?? Infinity)) continue;
    if (pred(c)) return { cell: c, km: d };
    for (let k = mesh.nbrStart[c], e = mesh.nbrStart[c + 1]; k < e; k++) {
      const m = mesh.nbr[k];
      const nd = d + mesh.nbrDist[k] * R;
      if (nd <= maxKm && nd < (dist.get(m) ?? Infinity)) {
        dist.set(m, nd);
        heap.push(nd, m);
      }
    }
  }
  return null;
}

/**
 * Crustal thickness (km) that would put cell c at elevation eTarget (m),
 * keeping its noise, boundary and volcanic contributions unchanged.
 */
function thickForElevation(world: World, c: number, eTarget: number): number {
  const g = world.geo;
  const other = world.relief.raw[c] - rawElevation(g.ctype[c], g.thick[c], g.age[c], g.volc[c]);
  const rawCrust = shelfExpand(eTarget) + world.params.seaLevel - other - g.volc[c] * 800;
  return Math.max(10, Math.min(75, thickForRaw(rawCrust)));
}

/** Absorb small disconnected pieces of plates into their surrounding plate. */
function cleanupPlateFragments(world: World, minCells = 25): void {
  const { n, nbrStart, nbr } = world.fine;
  const plate = world.geo.plate;
  const comp = new Int32Array(n).fill(-1);
  const comps: { plate: number; cells: number[] }[] = [];
  for (let c = 0; c < n; c++) {
    if (comp[c] >= 0) continue;
    const id = comps.length;
    const cells = [c];
    comp[c] = id;
    for (let i = 0; i < cells.length; i++) {
      const v = cells[i];
      for (let k = nbrStart[v], e = nbrStart[v + 1]; k < e; k++) {
        const m = nbr[k];
        if (comp[m] < 0 && plate[m] === plate[c]) {
          comp[m] = id;
          cells.push(m);
        }
      }
    }
    comps.push({ plate: plate[c], cells });
  }
  const largest = new Map<number, number>();
  comps.forEach((cp, i) => {
    const cur = largest.get(cp.plate);
    if (cur === undefined || comps[cur].cells.length < cp.cells.length) largest.set(cp.plate, i);
  });
  comps.forEach((cp, i) => {
    if (largest.get(cp.plate) === i || cp.cells.length >= minCells) return;
    const tally = new Map<number, number>();
    for (const c of cp.cells) {
      for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
        const q = plate[nbr[k]];
        if (q !== cp.plate) tally.set(q, (tally.get(q) ?? 0) + 1);
      }
    }
    let best = -1;
    let bc = 0;
    for (const [q, v] of tally) if (v > bc) [best, bc] = [q, v];
    if (best >= 0) for (const c of cp.cells) plate[c] = best;
  });
}

/** Passive-margin shelf and slope seaward of new land, inside the host plate. */
function addMargin(world: World, land: number[], hostPlate: number, widthKm: number, hostAge: number): number {
  const mesh = world.fine;
  const g = world.geo;
  const R = world.params.radiusKm;
  const dist = new Map<number, number>();
  const heap = new MinHeap(256);
  for (const s of land) {
    dist.set(s, 0);
    heap.push(0, s);
  }
  let changed = 0;
  while (heap.size > 0) {
    const c = heap.pop();
    const d = heap.lastKey;
    if (d > (dist.get(c) ?? Infinity)) continue;
    if (d > 0) {
      const target = -120 - 2800 * Math.pow(d / widthKm, 1.6);
      if (world.elev[c] < target && !g.ctype[c]) {
        g.ctype[c] = 1;
        g.thick[c] = thickForElevation(world, c, target);
        g.age[c] = hostAge;
        g.orogeny[c] = 800;
        g.plate[c] = hostPlate;
        changed++;
      }
    }
    for (let k = mesh.nbrStart[c], e = mesh.nbrStart[c + 1]; k < e; k++) {
      const m = mesh.nbr[k];
      const nd = d + mesh.nbrDist[k] * R;
      if (nd <= widthKm && nd < (dist.get(m) ?? Infinity) && world.elev[m] < 0) {
        dist.set(m, nd);
        heap.push(nd, m);
      }
    }
  }
  return changed;
}

// ------------------------------------------------------------------ brushes

/**
 * Paint land. The cause is continental crust: the nearest continent's crust is
 * extended and its plate grows to contain it (with a shelf around the new coast).
 * Far from any continent, a microcontinent is created instead.
 */
export function brushLand(world: World, points: LatLon[], radiusKm: number, strength: number): EditReport {
  const g = world.geo;
  const mesh = world.fine;
  const fp = footprint(world, points, radiusKm);
  const cells = [...fp.keys()];
  const host = nearestWhere(world, cells, (c) => g.ctype[c] === 1 && world.elev[c] > 0, 2500);
  const hostPlate = host ? g.plate[host.cell] : -1;
  const hostAge = host ? g.age[host.cell] : 600;
  let newArea = 0;
  const taken = new Map<number, number>();
  const landCells: number[] = [];
  for (const [c, w] of fp) {
    if (w < 0.05) continue;
    const target = 40 + 280 * strength * w;
    landCells.push(c);
    if (world.elev[c] >= target) continue;
    const want = thickForElevation(world, c, target);
    if (!g.ctype[c]) {
      g.ctype[c] = 1;
      g.age[c] = hostAge;
      g.orogeny[c] = 600;
      g.thick[c] = want;
    } else {
      g.thick[c] = Math.max(g.thick[c], want);
    }
    if (world.elev[c] < 0) newArea += mesh.area[c];
    if (hostPlate >= 0 && g.plate[c] !== hostPlate) {
      taken.set(g.plate[c], (taken.get(g.plate[c]) ?? 0) + mesh.area[c]);
      g.plate[c] = hostPlate;
    }
  }
  const margin = hostPlate >= 0 ? addMargin(world, landCells, hostPlate, 220, hostAge) : 0;
  cleanupPlateFragments(world);
  const details: string[] = [];
  if (!host) {
    const p = g.plates[g.plate[cells[0]]];
    return {
      title: 'New microcontinent',
      summary: `No continent lies within 2,500 km, so the new land is a microcontinent: a sliver of continental crust (${fmtKm2(km2(world, newArea))} newly emergent) riding on the ${p.name} plate.`,
      details: ['Continental crust thickened under the brush until it floats above sea level (isostasy).'],
      derive: 'tectonics',
    };
  }
  const hp = g.plates[hostPlate];
  details.push(`Continental crust of the ${hp.name} plate extended; ${fmtKm2(km2(world, newArea))} rose above sea level.`);
  if (taken.size) {
    const list = [...taken.entries()].map(([p, a]) => `${g.plates[p].name} (−${fmtKm2(km2(world, a))})`).join(', ');
    details.push(`${hp.name} plate grew into neighboring plates: ${list}. Its boundaries moved accordingly.`);
  }
  if (margin) details.push(`A passive-margin shelf and slope were added around the new coast (${margin} cells).`);
  return {
    title: `Extended ${hp.name}`,
    summary: `Land added by extending the nearest continent (${fmtKm(host.km)} away) — its continental crust and the ${hp.name} plate grow to include the painted area.`,
    details,
    derive: 'tectonics',
  };
}

/**
 * Paint sea. Shallow: continental crust is thinned and floods (an
 * epicontinental sea or sunken continent). Deep: the crust is rifted and
 * replaced by new oceanic crust.
 */
export function brushSea(world: World, points: LatLon[], radiusKm: number, strength: number, deep: boolean): EditReport {
  const g = world.geo;
  const mesh = world.fine;
  const fp = footprint(world, points, radiusKm);
  let flooded = 0;
  let rifted = 0;
  let volcanoes = 0;
  for (const [c, w] of fp) {
    if (w < 0.05) continue;
    if (deep) {
      const depth = 1500 + 2600 * strength * w;
      if (g.ctype[c]) {
        g.ctype[c] = 0;
        g.thick[c] = 7;
        g.age[c] = ageForDepth(depth);
        g.volc[c] = 0;
        rifted += mesh.area[c];
      } else if (-world.elev[c] < depth) {
        g.age[c] = Math.max(g.age[c], ageForDepth(depth));
        g.volc[c] = 0;
        g.thick[c] = Math.min(g.thick[c], 7);
      }
    } else {
      const target = -(40 + 180 * strength * w);
      if (world.elev[c] <= target) continue;
      if (g.ctype[c]) {
        g.thick[c] = Math.min(g.thick[c], thickForElevation(world, c, target));
        flooded += mesh.area[c];
      } else if (g.volc[c] > 0) {
        g.volc[c] = Math.max(0, g.volc[c] - (world.elev[c] - target) / 1000);
        volcanoes++;
      } else {
        g.thick[c] = 7;
        g.age[c] = Math.max(g.age[c], ageForDepth(-target));
      }
    }
  }
  if (deep) {
    return {
      title: 'Opened ocean floor',
      summary: `Continental crust was rifted apart and replaced by young oceanic crust (${fmtKm2(km2(world, rifted))}) — an embryonic ocean basin like the Red Sea.`,
      details: ['Seafloor depth follows the age–depth cooling curve of the new crust.', 'For the rift to keep opening, the plates on either side should diverge (edit plate motion).'],
      derive: 'tectonics',
    };
  }
  return {
    title: 'Flooded land',
    summary: `Continental crust thinned (stretched/subsided) so ${fmtKm2(km2(world, flooded))} now lies under a shallow sea — an epicontinental sea or sunken continent like Zealandia or Sundaland.`,
    details: volcanoes ? [`${volcanoes} volcanic cells eroded down to sea level.`] : [],
    derive: 'tectonics',
  };
}

/** Raise or lower terrain by thickening/thinning crust; reports the tectonic setting. */
export function brushRelief(
  world: World,
  points: LatLon[],
  radiusKm: number,
  strength: number,
  sign: 1 | -1,
  youthful: boolean,
): EditReport {
  const g = world.geo;
  const bnd = world.bnd;
  const fp = footprint(world, points, radiusKm);
  let nearColl = 0;
  let nearArc = 0;
  let cont = 0;
  let ocean = 0;
  const partner = new Map<number, number>();
  for (const [c, w] of fp) {
    if (w < 0.05) continue;
    if (bnd.dColl[c] < 500) {
      nearColl++;
      if (bnd.other[c] >= 0) partner.set(bnd.other[c], (partner.get(bnd.other[c]) ?? 0) + 1);
    }
    if (bnd.dArc[c] < 450) nearArc++;
    if (g.ctype[c]) {
      cont++;
      g.thick[c] = Math.max(12, Math.min(75, g.thick[c] + sign * 3.5 * strength * w));
      if (sign > 0) g.orogeny[c] = Math.min(g.orogeny[c], youthful ? 1 : 260);
    } else {
      ocean++;
      if (sign > 0) g.thick[c] = Math.min(30, g.thick[c] + 2.5 * strength * w);
      else {
        g.age[c] = Math.min(250, g.age[c] + 25 * strength * w);
        g.thick[c] = Math.max(6, g.thick[c] - 2 * strength * w);
      }
    }
  }
  const total = Math.max(1, cont + ocean);
  let setting: string;
  if (sign < 0) {
    setting = cont > ocean ? 'Crust thinned by extension or erosion; the surface subsides isostatically.' : 'Seafloor deepened (older, colder, denser lithosphere).';
  } else if (nearColl / total > 0.3) {
    setting = 'Mountains attributed to the active continental collision here — crust thickens as the plates converge.';
  } else if (nearArc / total > 0.3) {
    setting = 'Mountains attributed to arc magmatism and shortening above the nearby subduction zone.';
  } else if (cont >= ocean) {
    setting = youthful
      ? 'No active boundary nearby: recorded as intraplate uplift with young, rugged relief. Consider adding convergence (plate tool) to explain it.'
      : 'No active boundary nearby: recorded as an ancient, eroded mountain belt (an old suture, like the Appalachians or Urals).';
  } else {
    setting = 'Ocean floor thickened into an oceanic plateau (a large igneous province, like Ontong Java).';
  }
  return {
    title: sign > 0 ? (youthful ? 'Raised young mountains' : 'Raised old highlands') : 'Lowered terrain',
    summary: setting,
    details: [`Crustal thickness changed under ${cont + ocean} cells (Airy isostasy converts it to elevation).`],
    derive: 'tectonics',
  };
}

// ------------------------------------------------------------------ islands

interface Site {
  c: number;
  depth: number;
  sst: number;
  tropical: boolean;
  nearestContinentKm: number | null;
  river: { cell: number; km: number; q: number } | null;
}

function siteInfo(world: World, c: number): Site {
  let sst = 0;
  for (let m = 0; m < MONTHS; m++) sst += world.temp[c * MONTHS + m] / MONTHS;
  const nc = nearestWhere(world, [c], (m) => world.geo.ctype[m] === 1 && world.elev[m] > 0, 2500);
  const r = nearestWhere(
    world,
    [c],
    (m) => {
      if (world.ocean[m] || world.hydro.discharge[m] < 600) return false;
      const rc = world.hydro.routing.receiver[m];
      return rc >= 0 && world.ocean[rc] === 1;
    },
    450,
  );
  return {
    c,
    depth: -world.elev[c],
    sst,
    tropical: sst >= 21 && Math.abs(world.fine.lat[c]) < 32 * DEG,
    nearestContinentKm: nc ? nc.km : null,
    river: r ? { cell: r.cell, km: r.km, q: world.hydro.discharge[r.cell] } : null,
  };
}

/** Feasibility (0–1) of each natural process that could raise an island here. */
export function islandOptions(world: World, s: Site): CauseOption[] {
  const g = world.geo;
  const b = world.bnd;
  const c = s.c;
  const oceanic = g.ctype[c] === 0;
  const opts: CauseOption[] = [];
  const subductionNear = b.dArc[c] < 600 && !b.arcCont[c];
  let hs = oceanic ? 0.55 : 0.3;
  if (b.dArc[c] > 800 && b.dTrench[c] > 400) hs += 0.1;
  if (s.depth < 250) hs -= 0.2;
  opts.push({
    id: 'hotspot',
    label: 'Hotspot volcano',
    score: Math.max(0.05, hs),
    reason: oceanic ? 'A mantle plume can punch through oceanic crust anywhere; it implies a trailing chain of older seamounts.' : 'Plumes under continental crust make volcanic highlands rather than islands.',
  });
  opts.push({
    id: 'arc',
    label: 'Island arc',
    score: subductionNear ? 0.35 + 0.6 * Math.exp(-(((b.dArc[c] - 150) / 160) ** 2)) : 0.08,
    reason: subductionNear
      ? `Lies ${fmtKm(b.dArc[c])} behind an active trench (convergence ${cmyr(b.rArc[c])}) — classic arc position.`
      : 'No subduction zone nearby — choosing this will change plate motion to create one.',
  });
  opts.push({
    id: 'atoll',
    label: 'Atoll',
    score: s.tropical && oceanic ? (g.volc[c] > 0.3 ? 0.9 : s.depth > 600 ? 0.7 : 0.5) : 0.04,
    reason: s.tropical
      ? `Warm water (mean ${s.sst.toFixed(0)}°C) lets coral keep pace with a subsiding old volcano.`
      : `Water too cold for reef-building coral (mean ${s.sst.toFixed(0)}°C).`,
  });
  opts.push({
    id: 'microcontinent',
    label: 'Continental fragment',
    score: g.ctype[c] ? 0.8 : s.nearestContinentKm !== null && s.nearestContinentKm < 900 ? 0.45 : 0.15,
    reason: g.ctype[c]
      ? 'Already underlain by submerged continental crust — part of a sunken continent can be exposed (like New Zealand).'
      : s.nearestContinentKm !== null
        ? `A sliver rifted off the continent ${fmtKm(s.nearestContinentKm)} away (like Madagascar or the Seychelles).`
        : 'Far from any continent; an isolated continental fragment is possible but unusual.',
  });
  opts.push({
    id: 'shelf',
    label: 'Exposed shelf',
    score: g.ctype[c] && s.depth < 260 ? 0.95 : 0,
    reason: g.ctype[c] && s.depth < 260 ? 'Shallow continental shelf — a small uplift exposes it (like the British Isles).' : 'Needs a shallow continental shelf.',
  });
  opts.push({
    id: 'sediment',
    label: 'Delta / barrier island',
    score: s.river && s.depth < 120 ? 0.85 : 0,
    reason: s.river
      ? `River mouth ${fmtKm(s.river.km)} away (${Math.round(s.river.q).toLocaleString('en-US')} m³/s) supplies sediment.`
      : 'Needs shallow water near a large river mouth.',
  });
  return opts.sort((a, b2) => b2.score - a.score);
}

/** Build up a volcanic edifice so its summit reaches `peak` m (cone profile). */
function buildVolcano(world: World, center: number, radiusKm: number, peak: number, age: number): number {
  const g = world.geo;
  const mesh = world.fine;
  const R = world.params.radiusKm;
  const { cells, dist } = cellsWithin(mesh, center, radiusKm / R);
  let raised = 0;
  for (let i = 0; i < cells.length; i++) {
    const c = cells[i];
    const w = Math.max(0, 1 - (dist[i] * R) / radiusKm);
    const target = world.elev[c] + (peak - world.elev[c]) * Math.pow(w, 0.8);
    if (target <= world.elev[c]) continue;
    g.volc[c] += (target - world.elev[c]) / 1000;
    g.volcAge[c] = Math.min(g.volcAge[c], age);
    raised++;
  }
  return raised;
}

/** Seamount chain left by a plate drifting over a fixed plume (ages tStart..tEnd Myr). */
function hotspotTrack(world: World, h: Vec3, plate: number, tStart: number, tEnd: number): { km: number; cells: number } {
  const g = world.geo;
  const mesh = world.fine;
  const R = world.params.radiusKm;
  const p = g.plates[plate];
  const speed = Math.hypot(p.wx, p.wy, p.wz);
  if (speed < 1e-6) return { km: 0, cells: 0 };
  const step = Math.max(0.5, (mesh.spacing * 0.8) / speed);
  const m = new Float64Array(9);
  let cells = 0;
  let last: Vec3 | null = null;
  let km = 0;
  let cell = nearestCell(mesh, h[0], h[1], h[2]);
  for (let t = tStart; t <= tEnd; t += step) {
    rotationFromOmega(p.wx, p.wy, p.wz, t, m);
    const x = m[0] * h[0] + m[1] * h[1] + m[2] * h[2];
    const y = m[3] * h[0] + m[4] * h[1] + m[5] * h[2];
    const z = m[6] * h[0] + m[7] * h[1] + m[8] * h[2];
    cell = nearestCell(mesh, x, y, z, cell);
    if (g.plate[cell] !== plate) break;
    if (last) km += Math.acos(Math.min(1, last[0] * x + last[1] * y + last[2] * z)) * R;
    last = [x, y, z];
    // Edifices subside with their aging plate and erode: tall young volcanoes, deep old guyots.
    const height = (g.ctype[cell] ? 1.5 : 6.2) * Math.exp(-t / 32);
    if (height < 0.4) break;
    for (const c of cellsWithin(mesh, cell, (90 / R) * 1).cells) {
      const v = c === cell ? height : height * 0.45;
      if (g.volc[c] < v) {
        g.volc[c] = v;
        g.volcAge[c] = t;
        cells++;
      }
    }
  }
  return { km, cells };
}

function motionDescription(world: World, plate: number, c: number): { speed: number; dir: string; trail: string } {
  const p = world.geo.plates[plate];
  const R = world.params.radiusKm;
  const v = cellVec(world, c);
  const vel: Vec3 = [(p.wy * v[2] - p.wz * v[1]) * R, (p.wz * v[0] - p.wx * v[2]) * R, (p.wx * v[1] - p.wy * v[0]) * R];
  const speed = Math.hypot(...vel);
  const b = bearingOf(v, vel);
  return { speed, dir: compassName(b), trail: compassName((b + 180) % 360) };
}

/**
 * Change one plate's motion so that it converges on `overriding` at boundary
 * cell b (the minimum change to its Euler vector), creating a subduction zone.
 */
function forceConvergence(world: World, b: number, target: number, overriding: number, other: number): number {
  const g = world.geo;
  const R = world.params.radiusKm;
  const P = g.plates[overriding];
  const O = g.plates[other];
  const pb = cellVec(world, b);
  const pt = cellVec(world, target);
  // Tangent direction at b pointing toward the overriding plate's interior.
  const d0 = pt[0] * pb[0] + pt[1] * pb[1] + pt[2] * pb[2];
  let dx = pt[0] - d0 * pb[0];
  let dy = pt[1] - d0 * pb[1];
  let dz = pt[2] - d0 * pb[2];
  const dl = Math.hypot(dx, dy, dz) || 1;
  dx /= dl;
  dy /= dl;
  dz /= dl;
  const wx = O.wx - P.wx;
  const wy = O.wy - P.wy;
  const wz = O.wz - P.wz;
  // relative velocity of O w.r.t. P at b (rad/Myr)
  const rx = wy * pb[2] - wz * pb[1];
  const ry = wz * pb[0] - wx * pb[2];
  const rz = wx * pb[1] - wy * pb[0];
  const along = rx * dx + ry * dy + rz * dz;
  const want = 55 / R;
  const dvx = (want - along) * dx;
  const dvy = (want - along) * dy;
  const dvz = (want - along) * dz;
  // ΔΩ = p × Δv gives exactly Δv at p.
  const before = Math.hypot(O.wx, O.wy, O.wz);
  O.wx += pb[1] * dvz - pb[2] * dvy;
  O.wy += pb[2] * dvx - pb[0] * dvz;
  O.wz += pb[0] * dvy - pb[1] * dvx;
  const after = Math.hypot(O.wx, O.wy, O.wz);
  return (after - before) * R;
}

export function addIsland(world: World, at: LatLon, cause: IslandCause, sizeKm: number): EditReport {
  const g = world.geo;
  const mesh = world.fine;
  const R = world.params.radiusKm;
  const c = nearestCellLatLon(mesh, at.lat * DEG, at.lon * DEG);
  if (world.elev[c] > 0) {
    return { title: 'Island', summary: 'That spot is already land — use the land or relief brushes instead.', details: [], derive: 'tectonics', failed: true };
  }
  const site = siteInfo(world, c);
  const options = islandOptions(world, site);
  const chosen = cause === 'auto' ? options[0].id : cause;
  const plate = g.plate[c];
  const pName = g.plates[plate].name;
  const details: string[] = [];
  let title = '';
  let summary = '';
  switch (chosen) {
    case 'hotspot': {
      const v = cellVec(world, c);
      const h: Hotspot = {
        id: g.hotspots.length,
        x: v[0],
        y: v[1],
        z: v[2],
        strength: 4 + sizeKm / 30,
        radius: Math.max(80, sizeKm) / R,
        name: hotspotName(g.hotspots.length),
        active: true,
      };
      g.hotspots.push(h);
      buildVolcano(world, c, sizeKm, 900 + sizeKm * 15, 0);
      const track = hotspotTrack(world, v, plate, 1.5, 90);
      const mo = motionDescription(world, plate, c);
      title = `Hotspot island (${h.name})`;
      summary = `A mantle plume (${h.name}) now sits beneath the ${pName} plate. Because the plate moves ${cmyr(mo.speed)} toward the ${mo.dir}, the plume has left a ${fmtKm(track.km)} chain of older, subsiding seamounts trailing to the ${mo.trail}.`;
      details.push('Older edifices sink as the lithosphere under them cools; tropical ones may carry atolls.');
      details.push('The plume stays fixed in the mantle — “Advance time” will grow the chain.');
      break;
    }
    case 'arc': {
      const b = world.bnd;
      let note = '';
      if (!(b.dArc[c] < 600 && !b.arcCont[c])) {
        const bc = nearestWhere(world, [c], (m) => b.other[m] >= 0 && g.plate[m] === plate, 1800);
        if (!bc) {
          return {
            title: 'Island arc',
            summary: 'No plate boundary within 1,800 km to turn into a subduction zone. Try a hotspot island or split the plate first.',
            details: [],
            options,
            chosen,
            derive: 'tectonics',
            failed: true,
          };
        }
        const other = b.other[bc.cell];
        const dv = forceConvergence(world, bc.cell, c, plate, other);
        note = `The ${g.plates[other].name} plate's motion was adjusted (speed ${dv >= 0 ? '+' : ''}${cmyr(dv)}) so it now dives beneath the ${pName} plate ${fmtKm(bc.km)} from here, opening a trench.`;
        details.push(note);
      }
      // Arc crust: thickened, volcanic ridge parallel to the trench.
      const { cells, dist } = cellsWithin(mesh, c, (sizeKm * 2.5) / R);
      for (let i = 0; i < cells.length; i++) {
        const m = cells[i];
        if (g.plate[m] !== plate) continue;
        const w = 1 - (dist[i] * R) / (sizeKm * 2.5);
        g.thick[m] = Math.max(g.thick[m], 14 + 14 * w);
        if (g.thick[m] >= 20) g.ctype[m] = 1;
        g.orogeny[m] = 0;
      }
      buildVolcano(world, c, sizeKm, 1200 + sizeKm * 12, 0);
      title = 'Island arc';
      summary = note
        ? `Island arc created. ${note}`
        : `Volcanic island arc above the subduction zone: oceanic crust sinking beneath the ${pName} plate melts at ~100 km depth and feeds a line of volcanoes.`;
      details.push('Arc crust thickened along the volcanic front.');
      break;
    }
    case 'atoll': {
      const age = 28;
      const v = cellVec(world, c);
      const p = g.plates[plate];
      // The plume that built this volcano now lies "upstream" along the plate motion.
      const m = new Float64Array(9);
      rotationFromOmega(p.wx, p.wy, p.wz, -age, m);
      const hv: Vec3 = [m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2], m[6] * v[0] + m[7] * v[1] + m[8] * v[2]];
      const track = hotspotTrack(world, hv, plate, age - 10, age + 30);
      // Summit just below sea level: reef growth caps it as a ring of low islets.
      const want = (oceanDepth(g.age[c]) - 40) / 1000 - Math.max(0, g.thick[c] - 7) * 0.15;
      g.volc[c] = Math.max(g.volc[c], want);
      g.volcAge[c] = age;
      for (const n2 of cellsWithin(mesh, c, sizeKm / R).cells) {
        if (n2 !== c) g.volc[n2] = Math.max(g.volc[n2], want * 0.85);
      }
      const mo = motionDescription(world, plate, c);
      title = 'Atoll';
      summary = `Coral reef capping a ~${age} Myr-old volcano that has subsided below sea level (Darwin's subsidence model). It implies an extinct hotspot track: ${fmtKm(track.km)} of seamounts and guyots along the ${pName} plate's motion (${mo.dir}).`;
      details.push('Atolls need warm, clear water; outside the tropics this would be a submerged guyot instead.');
      break;
    }
    case 'microcontinent': {
      const radius = sizeKm * 3;
      const { cells, dist } = cellsWithin(mesh, c, radius / R);
      let above = 0;
      let total = 0;
      for (let i = 0; i < cells.length; i++) {
        const m = cells[i];
        const w = 1 - (dist[i] * R) / radius;
        const target = -1400 + 1750 * Math.pow(w, 1.4);
        if (world.elev[m] >= target) continue;
        g.ctype[m] = 1;
        g.thick[m] = thickForElevation(world, m, target);
        g.age[m] = 500;
        g.orogeny[m] = 350;
        total += mesh.area[m];
        if (target > 0) above += mesh.area[m];
      }
      const from = site.nearestContinentKm !== null ? ` It most likely rifted away from the continent ${fmtKm(site.nearestContinentKm)} away.` : '';
      title = 'Continental fragment';
      summary = `A microcontinent of ${fmtKm2(km2(world, total))}, only ${Math.round((100 * above) / Math.max(total, 1e-12))}% above water — mostly sunken like Zealandia.${from}`;
      details.push(`It rides on the ${pName} plate.`);
      break;
    }
    case 'shelf': {
      const { cells, dist } = cellsWithin(mesh, c, sizeKm / R);
      for (let i = 0; i < cells.length; i++) {
        const m = cells[i];
        if (!g.ctype[m]) continue;
        const w = 1 - (dist[i] * R) / sizeKm;
        const target = 30 + 160 * w;
        if (world.elev[m] < target) g.thick[m] = Math.max(g.thick[m], thickForElevation(world, m, target));
      }
      title = 'Exposed shelf island';
      summary = 'Part of the continental shelf raised above sea level by local crustal uplift — a continental island like Britain or Sri Lanka.';
      break;
    }
    case 'sediment': {
      const { cells, dist } = cellsWithin(mesh, c, sizeKm / R);
      for (let i = 0; i < cells.length; i++) {
        const m = cells[i];
        const w = 1 - (dist[i] * R) / sizeKm;
        const target = 4 + 12 * w;
        if (world.elev[m] >= target) continue;
        g.ctype[m] = 1;
        g.thick[m] = Math.max(g.thick[m], thickForElevation(world, m, target));
      }
      title = 'Delta / barrier island';
      summary = site.river
        ? `Low island built from river sediment (a river ${fmtKm(site.river.km)} away carries ${Math.round(site.river.q).toLocaleString('en-US')} m³/s).`
        : 'Low sedimentary island built on the shallow shelf.';
      break;
    }
  }
  return { title, summary, details, options, chosen, derive: 'tectonics' };
}

/** A new mantle plume; its past is reconstructed as a seamount track. */
export function addHotspot(world: World, at: LatLon, strength: number): EditReport {
  const g = world.geo;
  const R = world.params.radiusKm;
  const v = latLonToVec(at.lat * DEG, at.lon * DEG);
  const c = nearestCellLatLon(world.fine, at.lat * DEG, at.lon * DEG);
  const h: Hotspot = {
    id: g.hotspots.length,
    x: v[0],
    y: v[1],
    z: v[2],
    strength: 2 + 6 * strength,
    radius: (90 + 80 * strength) / R,
    name: hotspotName(g.hotspots.length),
    active: true,
  };
  g.hotspots.push(h);
  const plate = g.plate[c];
  const oceanic = !g.ctype[c];
  buildVolcano(world, c, 60 + 60 * strength, oceanic ? 600 + 2400 * strength : 800 + 1600 * strength, 0);
  const track = hotspotTrack(world, v, plate, 1.5, 90);
  const mo = motionDescription(world, plate, c);
  return {
    title: `Hotspot ${h.name}`,
    summary: `A fixed mantle plume beneath the ${g.plates[plate].name} plate. Plate motion (${cmyr(mo.speed)} toward the ${mo.dir}) implies a ${fmtKm(track.km)} volcanic track trailing ${mo.trail}.`,
    details: [oceanic ? 'On oceanic crust: a shield volcano and seamount chain.' : 'Under a continent: volcanic highlands and flood basalts.'],
    derive: 'tectonics',
  };
}

/** Set a plate's velocity at a point; the Euler pole is placed 90° away. */
export function setPlateVelocity(world: World, plate: number, at: LatLon, vEastCm: number, vNorthCm: number): EditReport {
  const g = world.geo;
  const R = world.params.radiusKm;
  const p = latLonToVec(at.lat * DEG, at.lon * DEG);
  const e: Vec3 = [-Math.sin(at.lon * DEG), Math.cos(at.lon * DEG), 0];
  const nrt: Vec3 = [p[1] * e[2] - p[2] * e[1], p[2] * e[0] - p[0] * e[2], p[0] * e[1] - p[1] * e[0]];
  const vx = ((vEastCm * e[0] + vNorthCm * nrt[0]) * 10) / R;
  const vy = ((vEastCm * e[1] + vNorthCm * nrt[1]) * 10) / R;
  const vz = ((vEastCm * e[2] + vNorthCm * nrt[2]) * 10) / R;
  const P = g.plates[plate];
  P.wx = p[1] * vz - p[2] * vy;
  P.wy = p[2] * vx - p[0] * vz;
  P.wz = p[0] * vy - p[1] * vx;
  const speed = Math.hypot(vEastCm, vNorthCm);
  const dir = compassName((Math.atan2(vEastCm, vNorthCm) / DEG + 360) % 360);
  return {
    title: `${P.name} plate motion`,
    summary: `The ${P.name} plate now moves ${speed.toFixed(1)} cm/yr toward the ${dir} at the chosen point. Its boundaries were re-evaluated: converging edges grow trenches, arcs and mountains; diverging edges open ridges and rifts.`,
    details: ['Use “Advance time” to let the new motion reshape the map over millions of years.'],
    derive: 'tectonics',
  };
}

/** Run the plate simulation forward on the current map. */
export function advanceTime(world: World, myr: number): EditReport {
  const g = world.geo;
  const before = g.plates.length;
  const sim = new TectonicSim(world.fine, g, {
    radiusKm: world.params.radiusKm,
    seed: (world.undoStack.length * 7919) ^ 0x2545f491,
    drift: false,
    merge: true,
    split: true,
  });
  sim.run(myr);
  compactPlates(world.fine, g);
  cleanupPlateFragments(world);
  const details = [`Plates moved along their current Euler poles for ${myr} Myr; ridges made new seafloor, trenches consumed old seafloor.`];
  if (sim.merges.length) details.push(`${sim.merges.length} plate pair(s) collided and welded together.`);
  if (sim.splits.length) details.push(`${sim.splits.length} large plate(s) rifted apart, opening new ocean basins.`);
  if (g.plates.length !== before) details.push(`Plate count ${before} → ${g.plates.length}.`);
  return {
    title: `Advanced ${myr} Myr`,
    summary: `Simulated ${myr} million years of plate motion from the current state: continents drifted, collided or rifted, and the climate, rivers and biomes were re-derived for the new geography.`,
    details,
    derive: 'tectonics',
  };
}
