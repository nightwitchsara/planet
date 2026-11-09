import { MinHeap } from '../core/heap';
import { smoothstep, sortByValueDesc } from '../core/math';
import type { SphereMesh } from '../core/mesh';
import { LAPSE, MONTHS } from './climate';
import { REFERENCE_CELLS } from './hydrology';
import { DDF_ICE, DDF_SNOW, normCdf, positiveDegreeDays, SIGMA_T } from './waterBalance';

/**
 * Glaciers and ice sheets. Ice gathers where a year's snowfall outlasts the
 * summer melt at the ice surface; because a growing ice body raises its own
 * surface into colder air, plateaus can tip into self-sustaining ice caps and
 * sheets. Ice flows down its surface slope, carrying the accumulation into
 * warmer ground until melt eats it up (the snout) or it calves into the sea.
 * Its thickness follows the perfectly plastic profile real ice caps closely
 * match (surface slope × thickness ≈ τ/ρg), and the bed sinks under the load.
 */

/** Basal yield stress over ice density × gravity (m), for τ ≈ 80 kPa. */
const PLASTIC_K = 80e3 / (917 * 9.81);
/** The bed sinks under an ice load by ρ_ice/ρ_mantle of its thickness. */
export const ICE_ISOSTASY = 0.28;
const ICE_PER_WATER = 1000 / 917;
/** Glacial deepening is capped (m) per cell: fjords and troughs are narrow, so a cell's average lowering stays modest. */
export const MAX_GLACIAL_EROSION = 350;
/** Sea level at the last glacial maximum (m): shelves lay dry and ice grounded on them. */
export const ICE_AGE_SEA_LEVEL = -120;

// Positive degree-days and snowfall share by monthly mean temperature, tabulated.
const T_LO = -80;
const T_STEP = 0.05;
const T_N = Math.round(130 / T_STEP) + 1;
const PDD_TABLE = new Float32Array(T_N);
const SNOW_TABLE = new Float32Array(T_N);
for (let i = 0; i < T_N; i++) {
  const t = T_LO + i * T_STEP;
  PDD_TABLE[i] = positiveDegreeDays(t);
  SNOW_TABLE[i] = normCdf((1 - t) / SIGMA_T);
}
function lookup(table: Float32Array, t: number): number {
  let f = (t - T_LO) / T_STEP;
  if (f < 0) f = 0;
  else if (f > T_N - 1.001) f = T_N - 1.001;
  const i = f | 0;
  const r = f - i;
  return table[i] * (1 - r) + table[i + 1] * r;
}

export interface IceClimate {
  /** Monthly sea-level temperature (°C) and precipitation (mm), c*12+m. */
  tempSL: Float32Array;
  precip: Float32Array;
  /** Per-cell cooling (°C) applied to every month, or null. */
  cool: Float32Array | null;
  precipFactor: number;
}

/** Annual surface mass balance (m water equivalent) of a glacier surface at elevation s: snow that outlasts the melt, or ice lost. */
export function massBalance(cl: IceClimate, c: number, s: number): number {
  const drop = (LAPSE * Math.max(0, s)) / 1000 + (cl.cool ? cl.cool[c] : 0);
  let snow = 0;
  let pdd = 0;
  for (let m = 0; m < MONTHS; m++) {
    const t = cl.tempSL[c * MONTHS + m] - drop;
    snow += cl.precip[c * MONTHS + m] * cl.precipFactor * lookup(SNOW_TABLE, t);
    pdd += lookup(PDD_TABLE, t);
  }
  const snowMelt = Math.min(snow, DDF_SNOW * pdd);
  const iceMelt = DDF_ICE * (pdd - snowMelt / DDF_SNOW);
  return (snow - snowMelt - iceMelt) / 1000;
}

/** Warmest-month temperature (°C) at elevation s. */
function warmest(cl: IceClimate, c: number, s: number): number {
  let t = -Infinity;
  for (let m = 0; m < MONTHS; m++) t = Math.max(t, cl.tempSL[c * MONTHS + m]);
  return t - (LAPSE * Math.max(0, s)) / 1000 - (cl.cool ? cl.cool[c] : 0);
}

export interface IceSheet {
  /** Ice thickness (m), averaged over the cell. */
  thick: Float32Array;
  /**
   * Share of the cell under ice (0–1). Where melt outpaces the ice flowing in,
   * a glacier ends inside the cell: its tongue covers only what the inflow feeds.
   */
  cover: Float32Array;
  /** Surface mass balance at the ice surface (m water equivalent per year). */
  smb: Float32Array;
  /** Ice flux leaving each cell (m³ of ice per year). */
  flux: Float32Array;
  /** Where each cell's ice flows: steepest descent of the ice surface (-1 at a pit). */
  receiver: Int32Array;
  /** Floating ice shelves over the sea. */
  shelf: Uint8Array;
  /** Number of cells in the ice body each cell belongs to (0 where ice-free). */
  body: Int32Array;
  /** Ice flowing into the sea (m³ per year). */
  calving: number;
}

/**
 * Perfectly plastic ice surface over the ice-covered cells: from each margin
 * the surface rises inward so that (surface − bed) × slope = τ/ρg. Solved like
 * a shortest-path problem (each cell takes the lowest surface any neighbor
 * allows), which also lets the ice follow a steep bed with thin ice. Returns
 * the thickness averaged over each cell: a partly covered cell holds a tongue
 * as thick as the profile allows over its length.
 */
function plasticThickness(mesh: SphereMesh, bed: Float32Array, cover: Float32Array, radiusM: number): Float32Array {
  const { n, nbrStart, nbr, nbrDist } = mesh;
  const s = new Float64Array(n).fill(Infinity);
  const heap = new MinHeap(1024);
  for (let c = 0; c < n; c++) {
    const f = cover[c];
    if (f <= 0) continue;
    let dMean = 0;
    for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
      dMean += nbrDist[k];
      if (cover[nbr[k]] > 0) continue;
      // The ice edge lies halfway to the ice-free neighbor (a tongue, its share of that).
      const v = bed[c] + Math.sqrt(2 * PLASTIC_K * nbrDist[k] * radiusM * 0.5 * f);
      if (v < s[c]) s[c] = v;
    }
    if (f < 1) {
      // A tongue ends inside its cell whichever way the cell's neighbors lie.
      const v = bed[c] + Math.sqrt(2 * PLASTIC_K * (dMean / (nbrStart[c + 1] - nbrStart[c])) * radiusM * 0.5 * f);
      if (v < s[c]) s[c] = v;
    }
    if (s[c] < Infinity) heap.push(s[c], c);
  }
  while (heap.size > 0) {
    const y = heap.pop();
    const sy = heap.lastKey;
    if (sy > s[y] + 1e-6) continue;
    for (let k = nbrStart[y], e = nbrStart[y + 1]; k < e; k++) {
      const x = nbr[k];
      if (cover[x] <= 0) continue;
      const b = bed[x];
      const d = nbrDist[k] * radiusM;
      const v = 0.5 * (b + sy + Math.sqrt((b - sy) * (b - sy) + 4 * PLASTIC_K * d));
      if (v < s[x]) {
        s[x] = v;
        heap.push(v, x);
      }
    }
  }
  const thick = new Float32Array(n);
  for (let c = 0; c < n; c++) if (cover[c] > 0 && s[c] < Infinity) thick[c] = cover[c] * Math.max(0, s[c] - bed[c]);
  return thick;
}

/**
 * Steady-state ice cover. Cells above `seaLevel` on the bed count as land, so
 * an ice-age run (sea level −120 m) can ground ice on the dry shelves.
 */
export function simulateIce(
  mesh: SphereMesh,
  bed: Float32Array,
  ocean: Uint8Array,
  cl: IceClimate,
  seaLevel: number,
  radiusKm: number,
  withShelves: boolean,
): IceSheet {
  const { n, nbrStart, nbr, nbrDist, area } = mesh;
  const radiusM = radiusKm * 1000;
  const land = new Uint8Array(n);
  let nLand = 0;
  for (let c = 0; c < n; c++) {
    if (!ocean[c] || bed[c] > seaLevel) {
      land[c] = 1;
      nLand++;
    }
  }
  const cells = new Int32Array(nLand);
  for (let c = 0, i = 0; c < n; c++) if (land[c]) cells[i++] = c;
  const thick = new Float32Array(n);
  const surf = new Float32Array(n);
  const smb = new Float32Array(n);
  const flux = new Float32Array(n);
  const receiver = new Int32Array(n).fill(-1);
  const inflow = new Float64Array(n);
  const cover = new Float32Array(n);
  let calving = 0;
  /** Cells whose surface changed in the latest pass (the rest keep their mass balance and receivers). */
  const changed = new Uint8Array(n);
  let first = true;

  const pass = (): void => {
    for (let i = 0; i < nLand; i++) {
      const c = cells[i];
      const s = Math.fround(bed[c] + (1 - ICE_ISOSTASY) * thick[c]);
      changed[c] = first || s !== surf[c] ? 1 : 0;
      if (!changed[c]) continue;
      surf[c] = s;
      smb[c] = massBalance(cl, c, s);
    }
    for (let i = 0; i < nLand; i++) {
      const c = cells[i];
      if (!first && !changed[c]) {
        let near = false;
        for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e && !near; k++) near = changed[nbr[k]] === 1;
        if (!near) continue;
      }
      let best = -1;
      let bestSlope = 0;
      for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
        const m = nbr[k];
        const sm = land[m] ? surf[m] : seaLevel;
        const slope = (surf[c] - sm) / nbrDist[k];
        if (slope > bestSlope) {
          bestSlope = slope;
          best = m;
        }
      }
      receiver[c] = best;
    }
    first = false;
    // Accumulation flows down the ice surface, from the top down, until melt eats it up.
    sortByValueDesc(cells, surf);
    inflow.fill(0);
    cover.fill(0);
    calving = 0;
    for (let i = 0; i < nLand; i++) {
      const c = cells[i];
      const own = smb[c] * area[c] * radiusM * radiusM * ICE_PER_WATER;
      if (own <= 0 && inflow[c] <= 0) {
        flux[c] = 0;
        continue;
      }
      // Below the snow line the ice reaches only as far as its inflow outlasts the melt.
      cover[c] = own >= 0 ? 1 : Math.min(1, inflow[c] / -own);
      const out = Math.max(0, inflow[c] + own);
      flux[c] = out;
      const r = receiver[c];
      if (r < 0 || out <= 0) continue;
      if (land[r]) inflow[r] += out;
      else calving += out;
    }
  };

  for (let it = 0; it < 5; it++) {
    pass();
    const next = plasticThickness(mesh, bed, cover, radiusM);
    for (const c of cells) thick[c] = it === 0 ? next[c] : 0.5 * (thick[c] + next[c]);
  }
  pass();
  const last = plasticThickness(mesh, bed, cover, radiusM);
  const mask = new Uint8Array(n);
  for (const c of cells) {
    thick[c] = cover[c] > 0 ? 0.5 * (thick[c] + last[c]) : 0;
    if (cover[c] > 0) mask[c] = 1;
  }

  // Ice bodies: connected ice, sized for naming (glacier, ice cap, ice sheet).
  const body = new Int32Array(n);
  const stack: number[] = [];
  const seen = new Uint8Array(n);
  for (const c0 of cells) {
    if (!mask[c0] || seen[c0]) continue;
    const members: number[] = [];
    stack.push(c0);
    seen[c0] = 1;
    while (stack.length) {
      const c = stack.pop()!;
      members.push(c);
      for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
        const m = nbr[k];
        if (mask[m] && !seen[m]) {
          seen[m] = 1;
          stack.push(m);
        }
      }
    }
    for (const c of members) body[c] = members.length;
  }

  const shelf = new Uint8Array(n);
  if (withShelves) growShelves(mesh, bed, land, mask, receiver, flux, cl, shelf);
  return { thick, cover, smb, flux, receiver, shelf, body, calving };
}

/**
 * Floating ice shelves: where ice sheets flow into a sea whose summers stay
 * below freezing, the ice spreads out over the water, mostly in sheltered
 * embayments, up to a few cells from the grounding line.
 */
function growShelves(
  mesh: SphereMesh,
  bed: Float32Array,
  land: Uint8Array,
  ice: Uint8Array,
  receiver: Int32Array,
  flux: Float32Array,
  cl: IceClimate,
  shelf: Uint8Array,
): void {
  const { n, nbrStart, nbr } = mesh;
  const cold = (c: number) => !land[c] && bed[c] > -1500 && warmest(cl, c, 0) < -1;
  let front: number[] = [];
  for (let c = 0; c < n; c++) {
    const r = receiver[c];
    if (ice[c] && flux[c] > 0 && r >= 0 && !land[r] && cold(r) && !shelf[r]) {
      shelf[r] = 1;
      front.push(r);
    }
  }
  // Into embayments about three cells of the reference detail (~120 km), whatever the mesh.
  const r = mesh.n / REFERENCE_CELLS;
  const rings = Math.max(1, Math.round(3 * Math.sqrt(r)));
  for (let ring = 0; ring < rings && front.length; ring++) {
    const next: number[] = [];
    for (const c of front) {
      for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
        const m = nbr[k];
        if (shelf[m] || !cold(m)) continue;
        let held = 0;
        for (let j = nbrStart[m], f = nbrStart[m + 1]; j < f; j++) if (land[nbr[j]] || shelf[nbr[j]]) held++;
        if (held >= 3) {
          shelf[m] = 1;
          next.push(m);
        }
      }
    }
    front = next;
  }
  // Sheltering lets a shelf grow only a cell or two out from its coast, so on
  // a finer mesh it is widened to the reference cell's width.
  for (let pass = 1; pass < Math.round(Math.sqrt(r)); pass++) {
    const add: number[] = [];
    for (let c = 0; c < n; c++) {
      if (!shelf[c]) continue;
      for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) if (!shelf[nbr[k]] && cold(nbr[k])) add.push(nbr[k]);
    }
    for (const m of add) shelf[m] = 1;
  }
}

/** Ice-age cooling (°C): a few degrees in the tropics, much more toward the poles. */
export function iceAgeCooling(mesh: SphereMesh): Float32Array {
  const out = new Float32Array(mesh.n);
  for (let c = 0; c < mesh.n; c++) out[c] = 3 + 9 * Math.sin(mesh.lat[c]) ** 2;
  return out;
}

export interface GlacialLegacy {
  /** Bed lowered by ice-age glaciers (m). */
  erosion: Float32Array;
  /** 0–1: how hard the ice-age ice scoured the ground. */
  scour: Float32Array;
}

/**
 * What the last ice age left behind: glaciers that slid fast over a thawed
 * bed (outlets, valley glaciers) deepened their beds into troughs, fjords and
 * lake basins. Ice frozen to its bed under a very cold, thin cover barely
 * eroded. (The land has since rebounded from the ice load.)
 */
export function glacialErosion(mesh: SphereMesh, bed: Float32Array, past: IceSheet, cl: IceClimate, radiusKm: number): GlacialLegacy {
  const n = mesh.n;
  const erosion = new Float32Array(n);
  const scour = new Float32Array(n);
  const width = mesh.spacing * radiusKm * 1000;
  for (let c = 0; c < n; c++) {
    const H = past.thick[c];
    if (H <= 0) continue;
    const u = past.flux[c] / (Math.max(50, H) * width);
    const surface = bed[c] + (1 - ICE_ISOSTASY) * H;
    const thawed = Math.max(smoothstep(-22, -6, warmest(cl, c, surface)), smoothstep(800, 2500, H));
    // Abrasion and quarrying grow with sliding speed: fast outlets cut deep, slow interiors barely.
    erosion[c] = Math.min(MAX_GLACIAL_EROSION, 0.5 * u ** 1.1) * thawed;
    scour[c] = 1 - Math.exp(-H / 300);
    bed[c] -= erosion[c];
  }
  return { erosion, scour };
}
