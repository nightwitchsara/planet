import { Simplex3 } from '../core/noise';
import { BIOME_INFO, KOPPEN_CODES, KOPPEN_COLORS } from '../world/biomes';
import { LAPSE, MONTHS } from '../world/climate';
import { ROCK_INFO } from '../world/geology';
import { ICE_ISOSTASY } from '../world/glaciers';
import { DEPOSIT_COLORS } from '../world/sediment';
import { SOIL_INFO } from '../world/soils';
import { B_CONVERGENT, B_DIVERGENT, B_TRANSFORM } from '../world/tectonics/boundaries';
import { Channel } from '../world/channels';
import { CHANNEL_WIDTH, type FieldData, KERNEL_RADIUS, type MeshData, MIN_STREAM_FRACTION, NET_BINS, NET_INFO, type RenderContext } from './context';
import * as pal from './palettes';

export type LayerId =
  | 'terrain'
  | 'elevation'
  | 'plates'
  | 'crust'
  | 'rocks'
  | 'deposits'
  | 'soils'
  | 'fertility'
  | 'temperature'
  | 'precipitation'
  | 'biomes'
  | 'koppen'
  | 'drainage'
  | 'channels';

export const LAYERS: { id: LayerId; label: string; monthly: boolean }[] = [
  { id: 'terrain', label: 'Terrain (natural color)', monthly: true },
  { id: 'elevation', label: 'Elevation', monthly: false },
  { id: 'plates', label: 'Tectonic plates', monthly: false },
  { id: 'crust', label: 'Crust type, thickness & seafloor age', monthly: false },
  { id: 'rocks', label: 'Rock types (bedrock)', monthly: false },
  { id: 'deposits', label: 'Surface deposits', monthly: false },
  { id: 'soils', label: 'Soils', monthly: false },
  { id: 'fertility', label: 'Soil fertility', monthly: false },
  { id: 'temperature', label: 'Temperature', monthly: true },
  { id: 'precipitation', label: 'Precipitation', monthly: true },
  { id: 'biomes', label: 'Biomes', monthly: false },
  { id: 'koppen', label: 'Köppen climate', monthly: false },
  { id: 'drainage', label: 'Rivers & drainage basins', monthly: false },
  { id: 'channels', label: 'River channel patterns', monthly: false },
];

export interface RenderOptions {
  layer: LayerId;
  /** 0–11, or -1 for the annual mean. */
  month: number;
  rivers: boolean;
}

/**
 * Pixel pyramid: level L is an equirectangular image of TILE·2^(L+1) × TILE·2^L
 * pixels (north up, 180°W at the left), cut into TILE×TILE tiles. Level 2 is
 * the 2048×1024 overview; each level below it doubles the resolution, down to
 * level 11 at about 38 m per pixel (on an Earth-sized planet).
 */
export const TILE = 256;
export const BASE_LEVEL = 2;
export const MAX_LEVEL = 11;
export const levelWidth = (L: number): number => TILE * 2 ** (L + 1);
export const levelHeight = (L: number): number => TILE * 2 ** L;
/** Pyramid level whose full width is nearest `width` pixels (at most the finest level). */
export const levelForWidth = (width: number): number => Math.min(MAX_LEVEL, Math.round(Math.log2(width / TILE)) - 1);

export interface PixelWindow {
  level: number;
  /** Top-left pixel in the level's global pixel grid (x may lie outside [0, width) and wraps). */
  x0: number;
  y0: number;
  w: number;
  h: number;
}

export interface RenderResult {
  rgba: Uint8ClampedArray;
  /** Rendered elevation (m) per pixel, when requested. */
  elevation: Float32Array | null;
}

const APRON = 2;
const LAND = 0;
const SEA = 1;
const LAKE = 2;
const SALT = 3;

const L_TERRAIN = 0;
const L_ELEV = 1;
const L_PLATES = 2;
const L_CRUST = 3;
const L_TEMP = 4;
const L_PRECIP = 5;
const L_BIOMES = 6;
const L_KOPPEN = 7;
const L_DRAIN = 8;
const L_ROCK = 9;
const L_DEPOSITS = 10;
const L_SOILS = 11;
const L_FERT = 12;
const L_CHANNELS = 13;
const LAYER_CODE: Record<LayerId, number> = {
  terrain: L_TERRAIN, elevation: L_ELEV, plates: L_PLATES, crust: L_CRUST, temperature: L_TEMP,
  precipitation: L_PRECIP, biomes: L_BIOMES, koppen: L_KOPPEN, drainage: L_DRAIN, rocks: L_ROCK, deposits: L_DEPOSITS,
  soils: L_SOILS, fertility: L_FERT, channels: L_CHANNELS,
};
/** How strongly hillshading shows on each layer. */
const SHADE_K = [1, 1, 0.4, 0.5, 0.35, 0.35, 0.6, 0.4, 0.5, 0.55, 0.6, 0.6, 0.5, 0.55];
const CATEGORICAL = new Set([L_PLATES, L_CRUST, L_BIOMES, L_KOPPEN, L_DRAIN, L_ROCK, L_DEPOSITS, L_SOILS]);
const OUTLINED = new Set([L_TEMP, L_PRECIP, L_KOPPEN, L_CRUST, L_PLATES, L_ROCK, L_DEPOSITS, L_SOILS, L_FERT, L_CHANNELS]);
const RIVER_LAYERS = new Set([L_TERRAIN, L_ELEV, L_BIOMES, L_DRAIN, L_ROCK, L_DEPOSITS, L_SOILS, L_FERT, L_CHANNELS]);

// ------------------------------------------------------------------ per-layer cell fields

const MODE_ALL = 0;
const MODE_LAND = 1;
const MODE_SEA = 2;

interface LayerCells {
  key: string;
  a: Float32Array | null;
  modeA: number;
  b: Float32Array | null;
  modeB: number;
}

let layerCache: LayerCells | null = null;

function monthly(src: Float32Array, n: number, month: number, mode: 'mean' | 'sum' | 'max'): Float32Array {
  const out = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    if (month >= 0) {
      out[c] = src[c * MONTHS + month];
      continue;
    }
    let s = mode === 'max' ? -Infinity : 0;
    for (let m = 0; m < MONTHS; m++) {
      const v = src[c * MONTHS + m];
      s = mode === 'max' ? Math.max(s, v) : s + v;
    }
    out[c] = mode === 'mean' ? s / MONTHS : s;
  }
  return out;
}

function layerCells(f: FieldData, opt: RenderOptions): LayerCells {
  const key = `${f.meshId}|${f.version}|${opt.layer}|${opt.month}`;
  if (layerCache?.key === key) return layerCache;
  const n = f.elev.length;
  let lc: LayerCells = { key, a: null, modeA: MODE_ALL, b: null, modeB: MODE_ALL };
  switch (opt.layer) {
    case 'terrain': {
      // Warmest month (annual view) or the month shown: snow lines and sea ice.
      const t = monthly(f.temp, n, opt.month, opt.month >= 0 ? 'mean' : 'max');
      lc = { key, a: t, modeA: MODE_LAND, b: t, modeB: MODE_SEA };
      break;
    }
    case 'temperature': {
      const t = monthly(f.temp, n, opt.month, 'mean');
      lc = { key, a: t, modeA: MODE_LAND, b: t, modeB: MODE_ALL };
      break;
    }
    case 'precipitation': {
      const p = monthly(f.precip, n, opt.month, 'sum');
      if (opt.month >= 0) for (let c = 0; c < n; c++) p[c] *= 12;
      lc = { key, a: null, modeA: MODE_ALL, b: p, modeB: MODE_ALL };
      break;
    }
    case 'crust':
      lc = { key, a: f.thick, modeA: MODE_ALL, b: f.age, modeB: MODE_ALL };
      break;
    case 'fertility':
      lc = { key, a: f.fertility, modeA: MODE_LAND, b: null, modeB: MODE_ALL };
      break;
  }
  layerCache = lc;
  return lc;
}

// ------------------------------------------------------------------ helpers

let noiseCache: { seed: number; noise: Simplex3 } | null = null;
function noiseFor(seed: number): Simplex3 {
  if (noiseCache?.seed !== seed) noiseCache = { seed, noise: new Simplex3(seed) };
  return noiseCache.noise;
}

const smooth01 = (x: number): number => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));

function hash01(a: number, b: number, k: number): number {
  let h = Math.imul(a ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(b + 0x632be5ab, 0xc2b2ae35) ^ Math.imul(k + 1, 0x27d4eb2f);
  h ^= h >>> 15;
  h = Math.imul(h, 0x2c1b3c6d);
  h ^= h >>> 12;
  h = Math.imul(h, 0x297a2d39);
  h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}

/**
 * Triangle containing unit vector p (visibility walk from `start`). A point
 * within rounding of an edge counts as inside: one sitting on a vertex (a
 * river node) would otherwise circle the vertex until the iteration cap.
 */
function locate(md: MeshData, px: number, py: number, pz: number, start: number): number {
  const { x, y, z, tri, halfedge } = md;
  let t = start;
  for (let iter = 0; iter < 100000; iter++) {
    const base = 3 * t;
    let moved = false;
    for (let s = 0; s < 3; s++) {
      const edge = base + ((s + iter) % 3);
      const a = tri[edge];
      const b = tri[edge % 3 === 2 ? edge - 2 : edge + 1];
      const d = px * (y[a] * z[b] - z[a] * y[b]) + py * (z[a] * x[b] - x[a] * z[b]) + pz * (x[a] * y[b] - y[a] * x[b]);
      if (d < -1e-13) {
        t = Math.floor(halfedge[edge] / 3);
        moved = true;
        break;
      }
    }
    if (!moved) return t;
  }
  return t;
}

function seedTriangle(md: MeshData, lat: number, lon: number): number {
  const j = Math.max(0, Math.min(md.gridH - 1, Math.floor(((Math.PI / 2 - lat) / Math.PI) * md.gridH)));
  let l = lon % (2 * Math.PI);
  if (l < -Math.PI) l += 2 * Math.PI;
  else if (l >= Math.PI) l -= 2 * Math.PI;
  const i = Math.max(0, Math.min(md.gridW - 1, Math.floor(((l + Math.PI) / (2 * Math.PI)) * md.gridW)));
  return md.gridTri[j * md.gridW + i];
}

// ------------------------------------------------------------------ scratch buffers

let capG = 0;
/** Pixel elevation, and its smooth cell-scale part (no sub-cell relief). */
let E = new Float32Array(0);
let SM = new Float32Array(0);
let CLS = new Uint8Array(0);
let CR = new Float32Array(0);
let CG = new Float32Array(0);
let CB = new Float32Array(0);
let TP = new Float32Array(0);
let CAT = new Int32Array(0);
let COVER = new Float32Array(0);
/** A wide channel's banks: coverage of its course widened by a rim (shaded where it isn't water). */
let BANK = new Float32Array(0);
let CARVE = new Float32Array(0);
/** One stream's valley profile while it is rasterized, before it joins CARVE. */
let STREAM = new Float32Array(0);
/** Valley floors of rivers that flow most of the year (riparian vegetation). */
let RIPARIAN = new Float32Array(0);
/** Per-pixel surface flags (F_*). */
let FLAGS = new Uint8Array(0);
/** Floodplain: 0–1, the flat valley floor a river wanders over. */
let FLOOD = new Float32Array(0);
/**
 * Ground between a river's channels: coverage 0–1 of a braided or
 * anastomosing river's active belt, a wandering river's bars and the point
 * bars inside bends; how bare it is (1: sand and gravel bars, 0: vegetated
 * islands); and how gravelly its bars are (0 sand, 1 gravel).
 */
let BELT = new Float32Array(0);
let BARE = new Float32Array(0);
let GRAV = new Float32Array(0);
/** Inside meander bends: 1 + distance from the bank in channel widths (0: none), for the scroll bars a migrating bend leaves. */
let SCRL = new Float32Array(0);
/** Oxbow lakes (still water) and the scars of older cut-off loops (wet meadow), 0–1. */
let OXB = new Float32Array(0);
let SCAR = new Float32Array(0);
/** Channel pattern of the river drawn at each pixel (world/channels.ts Channel + 1; 0: none). */
let PAT = new Uint8Array(0);
/** Flow arrows (drainage layer): fill and outline coverage. */
let ARW = new Float32Array(0);
let ARO = new Float32Array(0);
/** Levee land along a sea delta's distributaries (0–1): dry ground out to every mouth. */
let LEVEE = new Float32Array(0);
/** Salt flat (dry lake bed). */
const F_FLAT = 1;
/** Glacier or ice-sheet surface. */
const F_ICE = 2;
/** Floating ice shelf. */
const F_SHELF = 4;
/** Small lake in ice-scoured country (below the cell scale). */
const F_SCOUR = 8;
const AT = new Int32Array(4);

function scratch(G: number): void {
  if (G <= capG) return;
  capG = G;
  E = new Float32Array(G);
  SM = new Float32Array(G);
  CLS = new Uint8Array(G);
  CR = new Float32Array(G);
  CG = new Float32Array(G);
  CB = new Float32Array(G);
  TP = new Float32Array(G);
  CAT = new Int32Array(G);
  COVER = new Float32Array(G);
  BANK = new Float32Array(G);
  CARVE = new Float32Array(G);
  STREAM = new Float32Array(G);
  RIPARIAN = new Float32Array(G);
  FLAGS = new Uint8Array(G);
  FLOOD = new Float32Array(G);
  BELT = new Float32Array(G);
  BARE = new Float32Array(G);
  GRAV = new Float32Array(G);
  SCRL = new Float32Array(G);
  OXB = new Float32Array(G);
  SCAR = new Float32Array(G);
  PAT = new Uint8Array(G);
  ARW = new Float32Array(G);
  ARO = new Float32Array(G);
  LEVEE = new Float32Array(G);
}

/**
 * Neighbor for hillshading: pixels across a shoreline count as level with
 * this one, so coasts don't shade as cliffs.
 */
function sameSide(q: number, land: boolean): boolean {
  const c = CLS[q];
  return land ? c === LAND : c === SEA;
}

// ------------------------------------------------------------------ rivers

/**
 * Smallest discharge drawn when cells are `cellPx` pixels apart: smaller
 * streams appear as you zoom in, keeping the drawn network about as dense on
 * screen at every level.
 */
export function riverThresholdFor(f: FieldData, cellPx: number): number {
  return f.riverThreshold * Math.max(MIN_STREAM_FRACTION, Math.min(1, (40 / cellPx) ** 2));
}

/**
 * Half-width in pixels of a river with mean discharge q: its cartographic
 * stroke ranks it by mean flow (as the threshold does), but never thinner than
 * the real channel, which is built for the bankfull flood `bank` (about
 * 4·√Q_bankfull m wide: a flashy river cuts a wider bed than a steady one).
 */
function riverHalfWidth(q: number, bank: number, thr: number, L: number, mPerPx: number): number {
  const carto = (0.25 + 0.55 * Math.log10(q / thr + 1)) * (1 + 0.08 * (L - BASE_LEVEL));
  const real = (0.5 * CHANNEL_WIDTH * Math.sqrt(bank)) / mPerPx;
  return Math.max(0.35, carto, real);
}

/** Bessel function of the first kind J_n(x) (power series; small x). */
function besselJ(n: number, x: number): number {
  let term = 1;
  for (let k = 1; k <= n; k++) term *= x / 2 / k;
  let s = 0;
  for (let m = 0; m < 40; m++) {
    s += term;
    term *= -(x * x) / 4 / ((m + 1) * (m + 1 + n));
  }
  return s;
}

/**
 * Channel meanders are sine-generated curves (Langbein & Leopold): the heading
 * swings as ω·cos t about the meander belt's axis, so bends stay round however
 * far a loop swings (ω 1.55: sinuosity 2, bends 2.5 widths in radius). As
 * offsets from the axis they are Bessel series (Jacobi–Anger), crossing it
 * every half-wave: along it Σ DX_n·sin 2nt, across it Σ DY_n·sin (2n+1)t, in
 * units of the channel's length per wave over 2π; the axis advances J0 of
 * that. Writes J0, DX_1..3, DY_0..3 and the furthest swing (at t = π/2).
 */
function meanderShape(omega: number, out: Float64Array, o: number): void {
  out[o] = besselJ(0, omega);
  for (let n = 1; n <= 3; n++) out[o + n] = (2 * (-1) ** n * besselJ(2 * n, omega)) / (2 * n);
  let top = 0;
  for (let n = 0; n <= 3; n++) {
    out[o + 4 + n] = (2 * (-1) ** n * besselJ(2 * n + 1, omega)) / (2 * n + 1);
    top += out[o + 4 + n] * (-1) ** n;
  }
  out[o + 8] = top;
}

/** Half-width of the valley a river cuts into the sub-cell relief, in pixels. */
function valleyHalfWidth(q: number, mPerPx: number): number {
  return (1800 + 2400 * Math.log10(q / 40 + 1)) / mPerPx;
}

/** Valleys keep their width on the ground as you zoom in, up to this many pixels. */
const VALLEY_MAX_PX = 400;

/** Noise octaves of sub-cell detail when cells are `hPx` pixels apart. */
const octavesFor = (hPx: number): number => Math.max(1, Math.min(12, Math.floor(Math.log2(Math.max(1, hPx / 2))) + 1));

/**
 * Fractal coastline: the land/sea field χ (land +1, sea −1, kernel-blended)
 * is roughened where it crosses zero, fading out smoothly toward open sea and
 * inland (an abrupt cutoff would leave water or land ending in clean arcs).
 * Rugged (M: young mountains) and fjord coasts are more intricate. Shared by
 * the pixel loop and the river mouths, so rivers end exactly at the drawn coast.
 */
function roughChi(noise: Simplex3, chi: number, M: number, fj: number, px: number, py: number, pz: number, f0: number, K: number): number {
  if (chi <= -0.99 || chi >= 0.99) return chi;
  let s = 0;
  let a = 1;
  let fr = f0 * 1.1;
  for (let k = 0; k < K; k++) {
    s += a * noise.noise(px * fr + 101.3 - 31.1 * k, py * fr + 7.7 * k, pz * fr - 3.3 * k);
    a *= 0.55;
    fr *= 2;
  }
  const fade = smooth01((0.99 - Math.abs(chi)) / 0.39);
  return chi + Math.min(1.8, 1 + 0.4 * M + 0.8 * fj) * s * 1.17 * fade;
}

/**
 * Sub-cell relief at unit vector p (unit-ish spread): fBm, blended toward a
 * ridged multifractal in young mountains (M: sharp crests; finer detail fades
 * into the valleys). Octaves are added, never rescaled, as pixels shrink.
 */
function subcellRelief(noise: Simplex3, px: number, py: number, pz: number, f0: number, octaves: number, M: number): number {
  const gain = 0.52 + 0.13 * M;
  let s = 0;
  let a = 1;
  let fr = f0;
  let wgt = 1;
  // Main crests rise to summits and dip to cols along their length; at one
  // height they would run on as unbroken walls (and as lines of rock through ice).
  const crest = M > 0.02 ? 0.6 + 0.45 * noise.noise(px * f0 * 1.7 - 47.1, py * f0 * 1.7 + 13.9, pz * f0 * 1.7 + 88.3) : 1;
  for (let k = 0; k < octaves; k++) {
    let v = noise.noise(px * fr + 17.3 * k, py * fr - 9.1 * k, pz * fr + 5.7 * k);
    if (M > 0.02) {
      const r = 1 - Math.abs(v);
      const sig = r * r * wgt * (k === 0 ? crest : 1);
      wgt = Math.max(0.35, Math.min(1, sig * 2));
      v += (2.2 * (sig - 0.42) - v) * M;
    }
    s += a * v;
    a *= gain;
    fr *= 2;
  }
  return (s * 0.5 * Math.sqrt(1 - gain * gain)) / 0.3;
}

/** Surface elevation (m) and flags (F_ICE or 0) of the last landSurface call, and the depth of a trough lake over it (m, 0 if none). */
let surfE = 0;
let surfFlags = 0;
let surfLake = 0;

/**
 * The surface where the land/sea field has land: ice where it covers the
 * ground, else rock and soil cut by the river valleys (cv: the valley profile,
 * 0–1) and, where ice-age glaciers reached the sea, by their troughs. Returns
 * false where that ground lies drowned below the sea after all: a fjord, or
 * the sounds between skerries off its mouth. A trough basin behind the sill
 * holds a lake instead (surfLake). Shared by the pixel loop and the river
 * courses, so rivers end where the drawn sea begins, fjords included. Sets
 * surfE, surfFlags and surfLake.
 */
function landSurface(
  noise: Simplex3,
  px: number,
  py: number,
  pz: number,
  h: number,
  R: number,
  A: number,
  Dn: number,
  cv: number,
  S: number,
  iceH: number,
  cov: number,
  iceFull: number,
  fj: number,
  lakeShare: number,
  seaShare: number,
  K: number,
): boolean {
  /** Ragged ice margins (a third of a cell and finer) and fjord walls (~14 km and finer), finer still at deep zoom (K octaves of sub-cell detail). */
  const fIce = (2.3 * 0.75) / h;
  const fFjord = R / 14000;
  // The ground itself: the cell surface carries the ice's average thickness.
  const ground = S - (1 - ICE_ISOSTASY) * iceH;
  // Where ice-age glaciers reached the sea, the valleys are troughs cut far
  // deeper, whose walls wander: wider and narrower reaches, coves and spurs.
  const nearSea = smooth01(seaShare / 0.1);
  let trough = 0;
  /** 0–1: how far into the trough's floor (rather than up its walls) this ground lies. */
  let floorShare = 0;
  /** Down-valley noise (about −2…2, tens of km): basins between rock steps, and how far inland a fjord's sill lies. */
  let steps = 0;
  /** The trough's full depth here (m). */
  let depth = 0;
  /** Fractal noise (~14 km and finer) roughening the trough's walls, sill and lake ends. */
  let jag = 0;
  /** How far seaward (in nearSea units) the fjord's sill lies here. */
  let sill = 0;
  if (fj > 0.01 && cv > 0) {
    let a = 1;
    let fr = fFjord;
    for (let k = 0, nk = Math.max(4, Math.min(9, K - 3)); k < nk; k++) {
      jag += a * noise.noise(px * fr - 33.7 + 5.1 * k, py * fr + 71.9, pz * fr - 11.3 * k);
      a *= 0.55;
      fr *= 2.3;
    }
    steps = 2 * noise.noise(px * fFjord * 0.45 + 12.9, py * fFjord * 0.45 - 40.2, pz * fFjord * 0.45 + 7.7);
    // Distance from the river as a share of the valley's half-width (cv = (1 − d)^1.5),
    // pushed out into hollows and in along spurs of the ground around.
    const d = 1 - Math.cbrt(cv * cv);
    const dj = Math.min(1, Math.max(0, d + (3 * jag + 0.8 * Dn) * d * (1 - d)));
    // The sill lies at varying distances inland, its line ragged and bowed
    // seaward mid-trough like the end moraines that build many sills.
    const fs = R / 2500;
    sill = 0.15 * steps + 0.12 * jag + 0.12 * (1 - dj) ** 2 + 0.06 * noise.noise(px * fs - 21.3, py * fs + 8.8, pz * fs + 50.1);
    /** 0 at the fjord's head, 1 toward its mouth. */
    const down = smooth01((nearSea - 0.6 - sill) / 0.4);
    // Ice carved it U-shaped, a broad floor between steep walls, so a drowned
    // trough keeps its width up to a blunt head instead of tapering to a point.
    // It is wider where more ice flowed (where it cut deepest), and widens
    // from its head toward the sea as the ice of more valleys joined it.
    // Inland, away from the outlet glaciers, low gentle country was cut far
    // less deep than mountains, and down the valley the floor steps over rock
    // bars between deeper basins.
    const width = 0.4 + 0.15 * Math.sqrt(fj) + 0.2 * down;
    const inland = fj * Math.min(900, 200 + A);
    depth = inland + (900 * fj - inland) * nearSea;
    trough = depth * smooth01((width + 0.17 - dj) / 0.3) * (1 - 0.5 * smooth01((steps - 0.3) / 0.7));
    floorShare = smooth01((width - dj) / 0.3);
  }
  let e0 = 0;
  let flags = 0;
  if (cov > 0.004) {
    // Ice takes the low ground first, running down valleys and river
    // courses as tongues and filling troughs wall to wall, and the ridges
    // last. How much ground it takes follows the cells' ice cover, so a
    // margin runs raggedly between ice-covered and ice-free cells instead
    // of ringing each ice cell.
    let mn = 0;
    let a = 1;
    let fr = fIce;
    for (let k = 0, nk = Math.max(3, K - 1); k < nk; k++) {
      mn += a * noise.noise(px * fr + 61.3 - 7.1 * k, py * fr - 21.7 + 3.3 * k, pz * fr + 9.4);
      a *= 0.5;
      fr *= 2;
    }
    const u = Dn + 0.55 * mn - 0.8 * cv - trough / 600;
    // u spreads about logistically (scale 0.42): this cutoff covers a `cov` share of the
    // ground, sharpened so the kernel's tails don't spill ice into ice-free cells.
    const cs = smooth01((cov - 0.08) / 0.84);
    const z = cs >= 0.995 ? Infinity : cs <= 0 ? -Infinity : 0.42 * Math.log(cs / (1 - cs));
    if (u < z) {
      // As thick as the cells' ice where it lies under a smooth surface,
      // thinning toward its edge, where it drapes the ground it ends on.
      const taper = Math.sqrt(Math.min(1, (z - u) / 0.35));
      const Hp = iceFull * taper;
      const top = ground + (1 - ICE_ISOSTASY) * Hp + Math.max(0.04, 1 - taper) * A * Dn;
      // Peaks whose rock stands higher break through as nunataks.
      if (Hp > 2 && top > ground + A * Dn) {
        e0 = top;
        flags = F_ICE;
      }
    }
  }
  if (flags === 0) {
    // V-shaped valley: deepest at the river, sub-cell relief smoothed only near
    // its floor; lake beds are smoothed too, so shallow lakes keep clean shores.
    const Al = A * (1 - lakeShare) ** 2;
    e0 = ground + Al * Dn * (1 - cv * cv) - (0.3 * Al + 4) * cv - trough;
  }
  surfFlags = flags;
  surfLake = 0;
  if (flags !== 0) {
    surfE = e0;
    return true;
  }
  // A glacial trough drowned by the sea is a fjord reaching inland. The
  // depth it takes grows smoothly away from the sea, the valley floor and
  // fjord country, so fjords taper out instead of ending at a hard line. The
  // sea reaches in only as far as the trough's sill, which lies at varying
  // distances inland, and never into a lake's basin (that water is the lake's).
  const fjT = smooth01(fj / 0.25);
  const valley = Math.max(smooth01((cv - 0.1) / 0.3), floorShare);
  /** Where the valley holds one of the hydrology's own lakes (its water is that lake's). */
  const held = smooth01(lakeShare / 0.3);
  const open = smooth01((nearSea - 0.5 - sill) / 0.25) * (1 - held);
  /** Surface of a lake in the trough here (m), if any. */
  let level = -Infinity;
  if (depth > 0) {
    // The floor shoals over the moraine sill at the fjord's mouth, where rocky
    // knobs stand out of the water as skerries.
    const mouth = floorShare * fjT * smooth01((seaShare - 0.3) / 0.15);
    if (mouth > 0) {
      if (e0 < -40) e0 += 0.8 * mouth * (-40 - e0);
      const fs = R / 2500;
      const knob = noise.noise(px * fs + 5.3, py * fs - 77.1, pz * fs + 31.7) + 0.5 * noise.noise(px * fs * 2.7 - 13.9, py * fs * 2.7 + 4.4, pz * fs * 2.7 - 61.2);
      e0 += mouth * 90 * Math.max(0, knob - 0.25);
    }
    // Up the valley from the fjord's head the trough is no longer the sea's:
    // a rock threshold closes it off, and further up rock bars part its
    // overdeepened basins, each bar standing a little above the lakes the
    // basins hold (they fill to their outlets, above sea level).
    const lakeAt = Math.max(3, ground - (0.3 * A + 4) - 0.5 * depth);
    const basin = smooth01((-0.1 - steps - 0.3 * jag) / 0.5);
    const head = smooth01((open - 0.05) / 0.2) * (1 - smooth01((open - 0.6) / 0.15));
    const hold = (1 - held) * Math.max(head, (1 - basin) * (1 - smooth01((open - 0.25) / 0.25)));
    const crest = lakeAt + 4 + 0.15 * A * Math.max(0, Dn);
    if (e0 < crest) e0 += hold * (crest - e0);
    if (basin > 0 && open < 0.6 && held < 1) level = lakeAt;
  }
  surfE = e0;
  const drownAt = -600 * (1 - nearSea) - 400 * (1 - valley) - 400 * (1 - fjT) - 3000 * smooth01((0.62 - open) / 0.12);
  if (seaShare > 0 && e0 < drownAt) return false;
  if (e0 < level) surfLake = level - e0;
  return true;
}

/**
 * Ragged edges (lake shores, ice-shelf fronts): fractal noise from frequency
 * `f` down to a few pixels, one octave more per zoom level as for the coastline.
 */
function raggedEdge(noise: Simplex3, px: number, py: number, pz: number, f: number, K: number): number {
  let s = noise.noise(px * f + 3.3, py * f - 71.1, pz * f + 29.9);
  let a = 0.5;
  let fr = f * 2;
  for (let k = 1; k < K - 2; k++) {
    s += a * noise.noise(px * fr + 3.3 + 17.1 * k, py * fr - 71.1, pz * fr + 29.9 - 9.7 * k);
    a *= 0.5;
    fr *= 2;
  }
  return s;
}

/** Last triangle found by dryAt or cellChiAt (walk start for the next, nearby point). */
let courseTri = 0;

/**
 * Is a river's course at unit vector p on dry land in the coast drawn with K
 * octaves? The pixel loop's own decision (roughChi, then landSurface), taken
 * with the river's valley carved in (cv 1 on a carved river's line, else 0).
 */
function dryAt(md: MeshData, f: FieldData, noise: Simplex3, K: number, cv: number, px: number, py: number, pz: number): boolean {
  const t = (courseTri = locate(md, px, py, pz, courseTri));
  const { x, y, z, gStart, gCell } = md;
  const { elev, grad, amp, ocean, mtn, fjord, ice, iceCover, lake } = f;
  const h = md.spacing;
  const kr2 = (KERNEL_RADIUS * h) ** 2;
  let sw = 0;
  let swL = 0;
  let swO = 0;
  let sChi = 0;
  let sM = 0;
  let sF = 0;
  let sA = 0;
  let seL = 0;
  let sIce = 0;
  let sCov = 0;
  let sLake = 0;
  for (let k = gStart[t], e = gStart[t + 1]; k < e; k++) {
    const q = gCell[k];
    const dx = px - x[q];
    const dy = py - y[q];
    const dz = pz - z[q];
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 >= kr2) continue;
    let w = 1 - d2 / kr2;
    w *= w;
    sw += w;
    sA += w * amp[q];
    sM += w * mtn[q];
    sF += w * fjord[q];
    if (ocean[q]) {
      swO += w;
      sChi -= w;
    } else {
      swL += w;
      sChi += w;
      seL += w * (elev[q] + grad[3 * q] * dx + grad[3 * q + 1] * dy + grad[3 * q + 2] * dz);
      sIce += w * ice[q];
      sCov += w * iceCover[q];
      if (lake[q] >= 0) sLake += w;
    }
  }
  if (sw <= 0) return !ocean[md.tri[3 * t]];
  if (swL === 0) return false;
  const M = sM / sw;
  const f0 = 0.75 / h;
  if (swO > 0 && roughChi(noise, sChi / sw, M, sF / sw, px, py, pz, f0, K) <= 0) return false;
  const Dn = subcellRelief(noise, px, py, pz, f0, K, M);
  return landSurface(noise, px, py, pz, h, f.radiusKm * 1000, sA / sw, Dn, cv, seL / swL, sIce / swL, sCov / swL, sCov > 0 ? sIce / sCov : 0, sF / sw, sLake / sw, swO / sw, K);
}

/** Most points on a segment's curve: 2.5 px apart along its axis, about 2 along a meandering channel (a coarse mesh's segment spans thousands of pixels at the deepest levels); a mouth's run-on may add as many. */
const CURVE_MAX = 4096;
/** Scratch for river curves: control points, the spline, the spline with meanders (pixels). */
const CURVE_P = new Float64Array(12);
const CURVE_PTS = new Float64Array(4 * CURVE_MAX + 4);
const CURVE_MP = new Float64Array(4 * CURVE_MAX + 4);
/** The same course before the channel's own meanders: the meander belt's axis, which its valley follows. */
const CURVE_AX = new Float64Array(4 * CURVE_MAX + 4);
const CURVE_AMP = new Float64Array(96);
const CURVE_BELT = new Float64Array(96);
/** Where each of a segment's meander half-waves starts, as a fraction of its length, and the meander shape at each crossing (see meanderShape). */
const CURVE_HW = new Float64Array(CURVE_MAX + 1);
const CURVE_SHAPE = new Float64Array(9 * (CURVE_MAX + 1));
const CURVE_END = new Float64Array(2);
/** Points in the last curve, including a mouth's run-on past the sea cell. */
let curveN = 0;
/** Each half-wave's lean (skew) in the last curve. */
const CURVE_LEAN = new Float64Array(CURVE_MAX + 1);
/**
 * The last curve's loop apexes, one per meander half-wave (apexN of them;
 * none when it was drawn without meanders): the channel point furthest from
 * the meander belt's axis (-1 if none) and its signed offset (px). Half-wave
 * j is the chain's half-wave apexW0 + j (stable across tiles and levels).
 */
const CURVE_APEX_S = new Int32Array(CURVE_MAX + 1);
const CURVE_APEX_D = new Float64Array(CURVE_MAX + 1);
let apexN = 0;
let apexW0 = 0;
/**
 * Per point of the last curve (CURVE_MP, CURVE_AX): how far along the segment
 * it lies (0 at its first node, 1 at the next, more on a mouth's run-on), the
 * channel's signed offset from the meander belt's axis (px, + to the axis's
 * left), and the meander half-wave it belongs to (-1: none).
 */
const CURVE_V = new Float64Array(4 * CURVE_MAX + 4);
/**
 * The last curve's belt axis, per point like CURVE_AX: the course with only
 * its wanderings at least four belt half-widths long, which a braided or
 * anastomosing river's belt follows (it doesn't wriggle with the finer ones).
 * Smooth, and continuous from one segment to the next.
 */
const CURVE_BAX = new Float64Array(4 * CURVE_MAX + 4);
const BAX_SRC = new Float64Array(4 * CURVE_MAX + 4);
const CURVE_ACR = new Float64Array(4 * CURVE_MAX + 4);
const CURVE_HWJ = new Int32Array(4 * CURVE_MAX + 4);

/** ω of the sine-generated curve with sinuosity σ = 1/J0(ω), tabulated for σ from 1 to 4. */
const OMEGA_OF_SIN = (() => {
  const t = new Float64Array(129);
  let w = 0;
  for (let k = 0; k <= 128; k++) {
    const sig = 1 + (3 * k) / 128;
    // J0 falls monotonically from 1 to 0 over [0, 2.405): step until 1/J0 reaches σ, then refine by bisection.
    while (w < 2.4 && 1 / besselJ(0, w + 0.01) < sig) w += 0.01;
    let lo = w;
    let hi = Math.min(2.4, w + 0.01);
    for (let it = 0; it < 30; it++) {
      const mid = 0.5 * (lo + hi);
      if (1 / besselJ(0, mid) < sig) lo = mid;
      else hi = mid;
    }
    t[k] = 0.5 * (lo + hi);
  }
  return t;
})();

function omegaFor(sig: number): number {
  const u = (Math.max(1, Math.min(4, sig)) - 1) * (128 / 3);
  const k = Math.min(127, Math.floor(u));
  return Math.min(2, OMEGA_OF_SIN[k] + (OMEGA_OF_SIN[k + 1] - OMEGA_OF_SIN[k]) * (u - k));
}

/**
 * The drawn course of the segment leaving node i of chain ch, in pixels from
 * (x0g, y0g) at pixel size dRad (lonShift: ±2π for copies across the date
 * line): a centripetal Catmull–Rom curve through the cell centers with
 * deterministic meanders that vanish at both nodes, so tiles and zoom levels
 * agree and tributaries meet their trunk exactly. A mouth runs on past the sea
 * cell's center, so it always meets the coastline. Fills CURVE_MP (curveN
 * points) and returns how many lie between the two nodes (0: degenerate).
 */
function riverCurve(md: MeshData, f: FieldData, ch: number, i: number, dRad: number, x0g: number, y0g: number, lonShift: number): number {
  const { rivStart, rivCell, rivSea, rivNext, mtn } = f;
  const { x, y, z } = md;
  const P = CURVE_P;
  const pts = CURVE_PTS;
  const mp = CURVE_MP;
  const s0 = rivStart[ch];
  const s1 = rivStart[ch + 1];
  const a = rivCell[i];
  const b = rivCell[i + 1];
  // Control points: previous node, a, b, next node (reflected at chain ends;
  // past a confluence, the trunk's next node).
  const pa = i > s0 ? rivCell[i - 1] : -1;
  const pb = i + 2 < s1 ? rivCell[i + 2] : i + 2 === s1 ? rivNext[ch] : -1;
  // Into a network's apex the course heads for its main channel; out of its
  // exit it leaves along the network's trunk.
  const netIn = i + 1 < s1 - 1 && f.rivNet[i + 1] >= 0 && f.rivNet[i] < 0 ? f.rivNet[i + 1] : -1;
  const netOut = i > s0 && f.rivNet[i - 1] >= 0 && f.rivNet[i] < 0 ? f.rivNet[i - 1] : -1;
  const gIn = netIn >= 0 ? f.netNext[f.netStart[netIn]] : -1;
  const gOut = netOut >= 0 && f.netExit[netOut] >= 0 ? f.netParent[f.netExit[netOut]] : -1;
  P[3] = x[a];
  P[4] = y[a];
  P[5] = z[a];
  P[6] = x[b];
  P[7] = y[b];
  P[8] = z[b];
  for (let k = 0; k < 3; k++) {
    P[k] = gOut >= 0 ? (k === 0 ? f.netX[gOut] : k === 1 ? f.netY[gOut] : f.netZ[gOut]) : pa >= 0 ? (k === 0 ? x[pa] : k === 1 ? y[pa] : z[pa]) : 2 * P[3 + k] - P[6 + k];
    P[9 + k] = gIn >= 0 ? (k === 0 ? f.netX[gIn] : k === 1 ? f.netY[gIn] : f.netZ[gIn]) : pb >= 0 ? (k === 0 ? x[pb] : k === 1 ? y[pb] : z[pb]) : 2 * P[6 + k] - P[3 + k];
  }
  const seaEnd = rivSea[ch] === 1 && i + 1 === s1 - 1;
  const lonRef = Math.atan2(y[a], x[a]) + lonShift;
  const toPx = (vx: number, vy: number, vz: number, out: Float64Array, o: number): void => {
    const l = Math.hypot(vx, vy, vz);
    const lat = Math.asin(Math.max(-1, Math.min(1, vz / l)));
    let lon = Math.atan2(vy, vx);
    lon += 2 * Math.PI * Math.round((lonRef - lon) / (2 * Math.PI));
    out[o] = (lon + Math.PI) / dRad - 0.5 - x0g;
    out[o + 1] = (Math.PI / 2 - lat) / dRad - 0.5 - y0g;
  };
  // Knot spacing (centripetal: sqrt of chord length).
  const k01 = Math.max(1e-6, Math.sqrt(Math.hypot(P[3] - P[0], P[4] - P[1], P[5] - P[2])));
  const k12 = Math.max(1e-6, Math.sqrt(Math.hypot(P[6] - P[3], P[7] - P[4], P[8] - P[5])));
  const k23 = Math.max(1e-6, Math.sqrt(Math.hypot(P[9] - P[6], P[10] - P[7], P[11] - P[8])));
  const t1 = k01;
  const t2 = t1 + k12;
  const t3 = t2 + k23;
  toPx(P[3], P[4], P[5], pts, 0);
  const ax = pts[0];
  const ay = pts[1];
  toPx(P[6], P[7], P[8], pts, 0);
  const cdx = pts[0] - ax;
  const cdy = pts[1] - ay;
  const lenPx = Math.hypot(cdx, cdy);
  curveN = 0;
  apexN = 0;
  if (lenPx < 1e-6) return 0;
  /** The curve's point at u (0 at a, 1 at b), in pixels: Barry–Goldman pyramid for centripetal Catmull–Rom. */
  const spline = (u: number, out: Float64Array, o: number): void => {
    const t = t1 + u * k12;
    let vx = 0;
    let vy = 0;
    let vz = 0;
    for (let k = 0; k < 3; k++) {
      const p0 = P[k];
      const p1 = P[3 + k];
      const p2 = P[6 + k];
      const p3 = P[9 + k];
      const A1 = ((t1 - t) / t1) * p0 + (t / t1) * p1;
      const A2 = ((t2 - t) / k12) * p1 + ((t - t1) / k12) * p2;
      const A3 = ((t3 - t) / k23) * p2 + ((t - t2) / k23) * p3;
      const B1 = ((t2 - t) / t2) * A1 + (t / t2) * A2;
      const B2 = ((t3 - t) / (t3 - t1)) * A2 + ((t - t1) / (t3 - t1)) * A3;
      const C = ((t2 - t) / k12) * B1 + ((t - t1) / k12) * B2;
      if (k === 0) vx = C;
      else if (k === 1) vy = C;
      else vz = C;
    }
    toPx(vx, vy, vz, out, o);
  };
  const nS = Math.max(2, Math.min(CURVE_MAX, Math.ceil(lenPx / 2.5)));
  for (let s = 0; s < nS; s++) spline(s / (nS - 1), pts, 2 * s);
  let nPts = nS;
  if (seaEnd) {
    // On along the curve's own end direction (toward the open sea it was aimed
    // at), the same at every zoom level however few points the curve has.
    spline(1 - 1 / 64, CURVE_END, 0);
    const ex = pts[2 * (nS - 1)] - CURVE_END[0];
    const ey = pts[2 * (nS - 1) + 1] - CURVE_END[1];
    const el = Math.hypot(ex, ey) || 1;
    // The coast may lie well past a sea cell the coastline noise turned to land
    // (the course stops where it first meets the water).
    const extra = (2 * md.spacing) / dRad;
    const nE = Math.min(CURVE_MAX, Math.ceil(extra / 2.5));
    for (let s = 1; s <= nE; s++) {
      pts[2 * (nS - 1 + s)] = pts[2 * (nS - 1)] + (ex / el) * (extra * s) / nE;
      pts[2 * (nS - 1 + s) + 1] = pts[2 * (nS - 1) + 1] + (ey / el) * (extra * s) / nE;
    }
    nPts = nS + nE;
  }
  // How freely the river winds at each node: by its sinuosity (the channel
  // model's, from stream power, banks and bed), and not at all where valley
  // walls hem it in.
  // (An anastomosing river's main channel winds less: its other channels
  // part from it and wind on their own.)
  const sinuAt = (k: number): number => 1 + (f.rivSinu[k] - 1) * (1 - 0.7 * Math.min(1, f.rivBraid[k] + f.rivAnas[k]));
  const freedom = (k: number): number => smooth01((sinuAt(k) - 1.02) / 0.4) * (1 - 0.85 * f.rivConf[k]);
  const sinA = freedom(i);
  const sinB = freedom(i + 1);
  const confA = f.rivConf[i];
  const confB = f.rivConf[i + 1];
  // Meanders: harmonics with an envelope that vanishes (with its slope) at both
  // nodes. The river wanders at the cell's scale, and its meander belt over a
  // few of the channel's own meanders (below), as Ferguson's disturbed meanders
  // do: a channel swinging regularly along a straight line looks like a coil.
  // The belt's harmonics fall off steeply (it bends smoothly; the channel's
  // meanders are its finer detail) and taper only over its shortest wave at
  // each end.
  const waves = f.rivWave[i + 1] - f.rivWave[i];
  const nh = Math.min(24, Math.floor(lenPx / 6));
  const kLo = Math.max(2, Math.ceil(waves / 12));
  const kHi = Math.min(Math.floor(waves / 3), 3 * kLo, Math.floor(lenPx / 4), CURVE_BELT.length);
  const kTop = Math.max(nh, kHi);
  const flat = 1 - 0.6 * 0.5 * (mtn[a] + mtn[b]);
  const nxp = -cdy / lenPx;
  const nyp = cdx / lenPx;
  const amps = CURVE_AMP;
  const belt = CURVE_BELT;
  for (let k = 1; k <= kTop; k++) {
    amps[k - 1] = k <= nh ? lenPx * 0.085 * k ** -1.3 * flat * (2 * hash01(a, b, k) - 1) : 0;
    belt[k - 1] = k >= kLo && k <= kHi ? ((0.1 * lenPx) / kLo) * (kLo / k) ** 2 * flat * 0.5 * (sinA + sinB) * (2 * hash01(a, b, 100 + k) - 1) : 0;
  }
  // The belt axis keeps the wanderings at least four belt half-widths long.
  const beltPx = Math.min(0.3 * (md.spacing / dRad), (0.5 * Math.max(f.rivBelt[i], f.rivBelt[i + 1])) / (f.radiusKm * 1000 * dRad));
  const kBelt = Math.max(1, Math.floor(lenPx / (2 * Math.max(1, beltPx))));
  for (let s = 0; s < nPts; s++) {
    let d = 0;
    let dL = 0;
    if (s < nS && kTop > 0) {
      // sin(kπu) by the Chebyshev recurrence.
      const u = s / (nS - 1);
      const s1 = Math.sin(Math.PI * u);
      const c2 = 2 * Math.cos(Math.PI * u);
      let sPrev = 0;
      let sK = s1;
      let db = 0;
      let dbL = 0;
      for (let k = 1; k <= kTop; k++) {
        d += amps[k - 1] * sK;
        db += belt[k - 1] * sK;
        if (k <= kBelt) {
          dL += amps[k - 1] * sK;
          dbL += belt[k - 1] * sK;
        }
        const sNext = c2 * sK - sPrev;
        sPrev = sK;
        sK = sNext;
      }
      const taperB = smooth01(u * kHi) * smooth01((1 - u) * kHi);
      d = d * s1 + db * taperB;
      dL = dL * s1 + dbL * taperB;
    }
    mp[2 * s] = pts[2 * s] + nxp * d;
    mp[2 * s + 1] = pts[2 * s + 1] + nyp * d;
    BAX_SRC[2 * s] = pts[2 * s] + nxp * dL;
    BAX_SRC[2 * s + 1] = pts[2 * s + 1] + nyp * dL;
  }
  CURVE_BAX.set(BAX_SRC.subarray(0, 2 * nPts));
  // The course so far is the meander belt's axis, which the valley follows.
  const axis = CURVE_AX;
  axis.set(mp.subarray(0, 2 * nPts));
  curveN = nPts;
  // The channel meanders within it once it is a few pixels wide (narrower,
  // the line is drawn wider than the channel and its bends would zigzag or
  // coil inside it): waves about twelve widths long, a whole number of
  // half-waves to a segment, so it still runs through every node (tributaries
  // meet their trunk, chains stay joined).
  const mPerPx = f.radiusKm * 1000 * dRad;
  const showA = smooth01(((CHANNEL_WIDTH * Math.sqrt(f.rivBank[i])) / mPerPx - 1) / 2);
  const showB = smooth01(((CHANNEL_WIDTH * Math.sqrt(f.rivBank[i + 1])) / mPerPx - 1) / 2);
  const stepV = 2.5 / lenPx;
  if (waves <= 0 || waves > CURVE_MAX || nS < 3 || Math.max(showA, showB) <= 0) {
    for (let s = 0; s < nPts; s++) {
      CURVE_V[s] = s < nS ? s / (nS - 1) : 1 + (s - nS + 1) * stepV;
      CURVE_ACR[s] = 0;
      CURVE_HWJ[s] = -1;
    }
    return nS;
  }
  let axLen = 0;
  for (let s = 1; s < nS; s++) axLen += Math.hypot(mp[2 * s] - mp[2 * s - 2], mp[2 * s + 1] - mp[2 * s - 1]);
  const w0 = f.rivWave[i];
  const seedCell = rivCell[s0];
  const noise = noiseFor(f.noiseSeed);
  /** Each chain's own stretch of the slowly varying noise below (counted in half-waves down the chain, so segments and levels agree). */
  const seedOff = 1000 * hash01(seedCell, 0, 916);
  // Real meanders are irregular (Ferguson's disturbed meanders): loops of
  // uneven length, and reaches that wind tightly or gently, varying slowly
  // down the river (correlated noise over the global half-wave count) with a
  // little jitter from loop to loop. Each half-wave's share of the segment;
  // its swing scales with its length.
  const hw = CURVE_HW;
  hw[0] = 0;
  for (let j = 0; j < waves; j++) hw[j + 1] = hw[j] + Math.exp(0.6 * noise.noise((w0 + j) * 0.23 + seedOff, -8.1, 4.4)) * (0.55 + 0.9 * hash01(seedCell, w0 + j, 914));
  for (let j = 1; j <= waves; j++) hw[j] /= hw[waves];
  // Each crossing of the axis has its own ω, from the channel's sinuosity
  // there (sinuosity = 1/J0(ω)), and a loop blends from the shape at its start
  // to the shape at its end, so the channel never kinks where it crosses.
  const shape = CURVE_SHAPE;
  for (let c = 0; c <= waves; c++) {
    const sig = sinuAt(i) + (sinuAt(i + 1) - sinuAt(i)) * hw[c];
    const vary = Math.exp(0.85 * noise.noise((w0 + c) * 0.19 + seedOff, 3.7, 1.9)) * (0.45 + 1.1 * hash01(seedCell, w0 + c, 915));
    meanderShape(omegaFor(1 + (sig - 1) * vary), shape, 9 * c);
  }
  // Loops lean down- or upstream (Kinoshita's skew), mostly down, in runs.
  const leanOf = CURVE_LEAN;
  for (let j = 0; j < waves; j++) leanOf[j] = 0.06 + 0.22 * noise.noise((w0 + j) * 0.37 + seedOff, 12.3, -6.6) + 0.16 * (hash01(seedCell, w0 + j, 913) - 0.5);
  const ph1 = 2 * Math.PI * hash01(seedCell, 0, 911);
  const ph2 = 2 * Math.PI * hash01(seedCell, 0, 912);
  const sign = w0 & 1 ? -1 : 1;
  for (let j = 0; j < waves; j++) {
    CURVE_APEX_S[j] = -1;
    CURVE_APEX_D[j] = 0;
  }
  apexN = waves;
  apexW0 = w0;
  // The axis becomes the source; the channel, longer, is sampled about every 2 px.
  pts.set(mp.subarray(0, 2 * nPts));
  const nOut = Math.max(nS, Math.min(CURVE_MAX, Math.ceil((axLen * (1 + 1.5 * Math.max(showA, showB))) / 2)));
  let j = 0;
  for (let s = 0; s < nOut; s++) {
    const v = s / (nOut - 1);
    while (j < waves - 1 && v > hw[j + 1]) j++;
    const len = hw[j + 1] - hw[j];
    const tau = Math.min(1, (v - hw[j]) / len);
    const t = Math.PI * (j + tau);
    const psi = w0 + j + tau;
    // Swinging a little wider in some reaches than others, and hardly at all
    // where valley walls or mountains hem it in.
    const m = (showA + (showB - showA) * v) * (1 - 0.85 * (confA + (confB - confA) * v)) * (1 - 0.6 * (mtn[a] + (mtn[b] - mtn[a]) * v)) * (0.72 + 0.18 * Math.sin(0.648 * psi + ph1) + 0.1 * Math.sin(1.461 * psi + ph2));
    const lean = leanOf[j];
    const sin1 = Math.sin(t);
    const sin2 = Math.sin(2 * t);
    const sin3 = Math.sin(3 * t);
    const sin4 = Math.sin(4 * t);
    const sin5 = Math.sin(5 * t);
    const sin6 = Math.sin(6 * t);
    const sin7 = Math.sin(7 * t);
    // The loop's offsets from its axis, in half-wave lengths.
    const blend = smooth01(tau);
    let along = 0;
    let lat = 0;
    for (let e = 0; e < 2; e++) {
      const o = 9 * (j + e);
      const wgt = e === 0 ? 1 - blend : blend;
      const dX = shape[o + 1] * sin2 + shape[o + 2] * sin4 + shape[o + 3] * sin6;
      const dY = shape[o + 4] * sin1 + shape[o + 5] * sin3 + shape[o + 6] * sin5 + shape[o + 7] * sin7;
      const k = wgt / (Math.PI * shape[o]);
      along += k * (dX + (lean * dY * dY) / shape[o + 8]);
      lat += k * dY;
    }
    // Where along the axis (in samples), and how far across it (px).
    const ua = Math.max(0, Math.min(nS - 1, (v + len * m * along) * (nS - 1)));
    const across = sign * len * axLen * m * lat;
    // The loop's apex: where it swings furthest from the axis.
    if (Math.abs(across) > Math.abs(CURVE_APEX_D[j])) {
      CURVE_APEX_D[j] = across;
      CURVE_APEX_S[j] = s;
    }
    CURVE_V[s] = v;
    CURVE_ACR[s] = across;
    CURVE_HWJ[s] = j;
    const k0 = Math.min(nS - 2, Math.floor(ua));
    const fr = ua - k0;
    // The axis direction, blended between its samples (a sudden turn would kink the channel's offset).
    const kp = Math.max(0, k0 - 1);
    const kn = Math.min(nS - 1, k0 + 2);
    const tx = (pts[2 * k0 + 2] - pts[2 * kp]) * (1 - fr) + (pts[2 * kn] - pts[2 * k0]) * fr;
    const ty = (pts[2 * k0 + 3] - pts[2 * kp + 1]) * (1 - fr) + (pts[2 * kn + 1] - pts[2 * k0 + 1]) * fr;
    const tl = Math.hypot(tx, ty) || 1;
    mp[2 * s] = pts[2 * k0] + (pts[2 * k0 + 2] - pts[2 * k0]) * fr - (ty / tl) * across;
    mp[2 * s + 1] = pts[2 * k0 + 1] + (pts[2 * k0 + 3] - pts[2 * k0 + 1]) * fr + (tx / tl) * across;
    // The axis, resampled alongside.
    const uv = v * (nS - 1);
    const j0 = Math.min(nS - 2, Math.floor(uv));
    axis[2 * s] = pts[2 * j0] + (pts[2 * j0 + 2] - pts[2 * j0]) * (uv - j0);
    axis[2 * s + 1] = pts[2 * j0 + 1] + (pts[2 * j0 + 3] - pts[2 * j0 + 1]) * (uv - j0);
    CURVE_BAX[2 * s] = BAX_SRC[2 * j0] + (BAX_SRC[2 * j0 + 2] - BAX_SRC[2 * j0]) * (uv - j0);
    CURVE_BAX[2 * s + 1] = BAX_SRC[2 * j0 + 1] + (BAX_SRC[2 * j0 + 3] - BAX_SRC[2 * j0 + 1]) * (uv - j0);
  }
  // Exactly through the end node, then on along the run-on.
  for (let s = nS - 1; s < nPts; s++) {
    const o = s - nS + nOut;
    mp[2 * o] = axis[2 * o] = CURVE_BAX[2 * o] = s < nS ? BAX_SRC[2 * s] : pts[2 * s];
    mp[2 * o + 1] = axis[2 * o + 1] = CURVE_BAX[2 * o + 1] = s < nS ? BAX_SRC[2 * s + 1] : pts[2 * s + 1];
    CURVE_V[o] = 1 + (s - nS + 1) * stepV;
    CURVE_ACR[o] = 0;
    CURVE_HWJ[o] = -1;
  }
  curveN = nOut + nPts - nS;
  return nOut;
}

/** Per river node, for the segment leaving it at one zoom level: see riverReach. */
export interface Reach {
  /** First curve point in the drawn water, sea or drowned land (the line is drawn up to there), or -1. */
  wetCut: Int32Array;
  /** Discharge above the segment that already ran into the water. */
  wetDrop: Float32Array;
  /** At a mouth whose river ran into its own drowned valley: the first point past the sea cell in water in its own right (the valley runs on to there), or -1. */
  runCut: Int32Array;
}
/** A level's reach, settled as tiles need it: resolve(i) settles the segment leaving node i and everything upstream of it. */
interface LazyReach extends Reach {
  resolve(i: number): void;
}
const reachCache = new WeakMap<FieldData, Map<number, LazyReach>>();

/** The river network as drawn at level L, with that level's smallest drawn discharge (for tests). */
export function riverCourse(ctx: RenderContext, L: number): Reach & { thr: number } {
  const { wetCut, wetDrop, runCut, resolve } = riverReach(ctx.mesh, ctx.fields, L);
  for (const i of ctx.fields.rivOrder) resolve(i);
  return { wetCut, wetDrop, runCut, thr: riverThresholdFor(ctx.fields, (ctx.mesh.spacing * levelHeight(L)) / Math.PI) };
}

const inflowCache = new WeakMap<FieldData, { start: Int32Array; seg: Int32Array }>();

/** The segments ending at each segment's first node, as seg[start[i]..start[i + 1]) (cached per field version). */
function inflows(f: FieldData): { start: Int32Array; seg: Int32Array } {
  let c = inflowCache.get(f);
  if (!c) {
    const { rivStart, rivCell } = f;
    const nNodes = rivCell.length;
    // Every cell starts at most one segment.
    const segOf = new Int32Array(f.elev.length).fill(-1);
    for (let ch = 0; ch < rivStart.length - 1; ch++) for (let i = rivStart[ch]; i < rivStart[ch + 1] - 1; i++) segOf[rivCell[i]] = i;
    const down = new Int32Array(nNodes).fill(-1);
    const start = new Int32Array(nNodes + 1);
    for (let ch = 0; ch < rivStart.length - 1; ch++) {
      for (let k = rivStart[ch]; k < rivStart[ch + 1] - 1; k++) {
        down[k] = segOf[rivCell[k + 1]];
        if (down[k] >= 0) start[down[k] + 1]++;
      }
    }
    for (let i = 0; i < nNodes; i++) start[i + 1] += start[i];
    const seg = new Int32Array(start[nNodes]);
    const at = start.slice(0, nNodes);
    for (let k = 0; k < nNodes; k++) if (down[k] >= 0) seg[at[down[k]]++] = k;
    c = { start, seg };
    inflowCache.set(f, c);
  }
  return c;
}

/**
 * Where the river network meets the water drawn at level L (cached per field
 * version and level), by the pixel loop's own land/sea decision. A river ends
 * where its course first runs into the water: a course that crosses an inlet
 * or a strait must not reappear beyond it as a sourceless fragment, and one
 * that runs into a fjord ends at its head. Downstream, a trunk carries on only
 * with what joined it since (a tributary arriving below still finds its way to
 * the water along the trunk's course). Valleys are carved regardless, all the
 * way to the sea cell (a fjord is a valley drowned, and must reach the sea).
 * Segments are settled only as tiles reach them (with what lies upstream), so
 * a deep level's first tile doesn't wait on every coast in the world.
 */
function riverReach(md: MeshData, f: FieldData, L: number): LazyReach {
  let byLevel = reachCache.get(f);
  if (!byLevel) {
    byLevel = new Map();
    reachCache.set(f, byLevel);
  }
  const hit = byLevel.get(L);
  if (hit) return hit;
  const { rivStart, rivCell, rivQ, rivCoast, rivSea, rivRun, elev } = f;
  const nNodes = rivCell.length;
  const wetCut = new Int32Array(nNodes).fill(-1);
  const wetDrop = new Float32Array(nNodes);
  const runCut = new Int32Array(nNodes).fill(-1);
  /** Per segment: 0 not reached, 1 waiting on the segments above it, 2 settled. */
  const state = new Uint8Array(nNodes);
  const stack = new Int32Array(nNodes + 1);
  const chainOf = chainIndex(f);
  const into = inflows(f);
  const dRad = Math.PI / levelHeight(L);
  const mPerPx = f.radiusKm * 1000 * dRad;
  const thrL = riverThresholdFor(f, md.spacing / dRad);
  const carveOn = valleyHalfWidth(f.riverThreshold, mPerPx) >= 1.2;
  const K = octavesFor(md.spacing / dRad);
  const noise = noiseFor(f.noiseSeed);
  let cv = 0;
  let nMain = 0;
  // In the water drawn at this level, with the river's own valley carved in
  // (which ends at the sea cell: only the river runs on to the drawn coast).
  const wet = (s: number): boolean => {
    const lon = (CURVE_MP[2 * s] + 0.5) * dRad - Math.PI;
    const lat = Math.PI / 2 - (CURVE_MP[2 * s + 1] + 0.5) * dRad;
    const cl = Math.cos(lat);
    return !dryAt(md, f, noise, K, s < nMain ? cv : 0, cl * Math.cos(lon), cl * Math.sin(lon), Math.sin(lat));
  };
  // First wet curve point: every point (2.5 px apart) while a cell spans few
  // pixels, so no strait goes unseen; deeper in every `stride`-th (water under
  // ~8 px wide may be missed), then the steps before the hit.
  const stride = Math.max(1, Math.min(3, Math.floor(md.spacing / dRad / 40)));
  const firstWet = (): number => {
    for (let s = 0, last = curveN - 1; s <= last + stride - 1; s += stride) {
      const sc = Math.min(s, last);
      if (!wet(sc)) continue;
      for (let b = Math.max(0, sc - stride + 1); b < sc; b++) if (wet(b)) return b;
      return sc;
    }
    return -1;
  };
  /** The segment leaving node i, once every segment into its first node is settled. */
  const settle = (i: number): void => {
    const a = rivCell[i];
    const b = rivCell[i + 1];
    const ch = chainOf[i];
    // A segment that ran into the water loses all its flow, others what they lost above.
    let above = 0;
    for (let k = into.start[i]; k < into.start[i + 1]; k++) {
      const u = into.seg[k];
      above += wetCut[u] >= 0 ? rivQ[u] : wetDrop[u];
    }
    const lost = Math.min(rivQ[i], above);
    wetDrop[i] = lost;
    // Only drawn segments near the sea, or deep below it, can meet the water;
    // a carved valley at a mouth may run on past its sea cell.
    const drawn = rivQ[i + 1] - lost >= thrL && (rivCoast[i] || Math.min(elev[a], elev[b]) < -500);
    const mouth = carveOn && rivSea[ch] === 1 && rivRun[ch] === 1 && i + 2 === rivStart[ch + 1] && rivQ[i + 1] >= thrL;
    if ((drawn || mouth) && (nMain = riverCurve(md, f, ch, i, dRad, 0, 0, 0)) > 0) {
      courseTri = seedTriangle(md, Math.PI / 2 - (CURVE_MP[1] + 0.5) * dRad, (CURVE_MP[0] + 0.5) * dRad - Math.PI);
      // The river's own valley runs under its line where it is carved at this level.
      cv = carveOn && valleyHalfWidth(Math.max(rivQ[i], rivQ[i + 1]), mPerPx) >= 1.2 ? 1 : 0;
      if (drawn) wetCut[i] = firstWet();
      if (mouth && wetCut[i] >= 0) {
        // The coastline noise may have put land past the sea cell. Where the
        // river ran into water only because its own valley lies drowned (a
        // fjord's trough), the valley runs on until it reaches water in its own
        // right, so the fjord opens to the sea instead of ending short of it.
        let s = wetCut[i];
        const cvMain = cv;
        cv = 0;
        while (s < curveN && !wet(s)) s++;
        cv = cvMain;
        runCut[i] = s < curveN && s >= nMain ? s : -1;
      }
    }
  };
  // Upstream first, by an explicit stack (a trunk's basin can hold thousands of segments).
  const resolve = (i: number): void => {
    if (state[i] === 2) return;
    let top = 0;
    stack[top++] = i;
    while (top > 0) {
      const t = stack[top - 1];
      if (state[t] === 0) {
        state[t] = 1;
        for (let k = into.start[t]; k < into.start[t + 1]; k++) if (state[into.seg[k]] === 0) stack[top++] = into.seg[k];
        continue;
      }
      top--;
      if (state[t] === 1) {
        settle(t);
        state[t] = 2;
      }
    }
  };
  const r = { wetCut, wetDrop, runCut, resolve };
  byLevel.set(L, r);
  return r;
}

const chainCache = new WeakMap<FieldData, Int32Array>();

/** The chain each river node belongs to (cached per field version). */
function chainIndex(f: FieldData): Int32Array {
  let c = chainCache.get(f);
  if (!c) {
    c = new Int32Array(f.rivCell.length);
    for (let ch = 0; ch < f.rivStart.length - 1; ch++) c.fill(ch, f.rivStart[ch], f.rivStart[ch + 1]);
    chainCache.set(f, c);
  }
  return c;
}

/** A trunk's drawn course around a confluence (pixels), and each tributary point's distance to it. */
const TRUNK = new Float64Array(4 * CURVE_MAX);
const TRUNK_DIST = new Float64Array(CURVE_MAX + 2);
let trunkN = 0;
/** Bounding boxes of the trunk's course in runs of TRUNK_RUN segments (x0, y0, x1, y1). */
const TRUNK_RUN = 32;
const TRUNK_BOX = new Float64Array(4 * Math.ceil((2 * CURVE_MAX) / TRUNK_RUN));

/** The trunk's course through its node j: the segments into and out of it (chain tc). */
function trunkCourse(md: MeshData, f: FieldData, tc: number, j: number, dRad: number, x0g: number, y0g: number, lonShift: number): void {
  trunkN = 0;
  const s0 = f.rivStart[tc];
  const s1 = f.rivStart[tc + 1];
  for (let i = j - 1; i <= j; i++) {
    if (i < s0 || i > s1 - 2) continue;
    const nS = riverCurve(md, f, tc, i, dRad, x0g, y0g, lonShift);
    for (let s = trunkN > 0 ? 1 : 0; s < nS && trunkN < 2 * CURVE_MAX; s++) {
      TRUNK[2 * trunkN] = CURVE_MP[2 * s];
      TRUNK[2 * trunkN + 1] = CURVE_MP[2 * s + 1];
      trunkN++;
    }
  }
  for (let r = 0; r * TRUNK_RUN < trunkN - 1; r++) {
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    for (let k = r * TRUNK_RUN, e = Math.min(trunkN - 1, (r + 1) * TRUNK_RUN); k <= e; k++) {
      x0 = Math.min(x0, TRUNK[2 * k]);
      y0 = Math.min(y0, TRUNK[2 * k + 1]);
      x1 = Math.max(x1, TRUNK[2 * k]);
      y1 = Math.max(y1, TRUNK[2 * k + 1]);
    }
    TRUNK_BOX[4 * r] = x0;
    TRUNK_BOX[4 * r + 1] = y0;
    TRUNK_BOX[4 * r + 2] = x1;
    TRUNK_BOX[4 * r + 3] = y1;
  }
}

/** Distance (pixels) from a point to the trunk's course, or `far` if that is nearer (runs of the course further off are skipped). */
function trunkDistance(px: number, py: number, far: number): number {
  let best = far * far;
  for (let r = 0; r * TRUNK_RUN < trunkN - 1; r++) {
    const bx = Math.max(TRUNK_BOX[4 * r] - px, 0, px - TRUNK_BOX[4 * r + 2]);
    const by = Math.max(TRUNK_BOX[4 * r + 1] - py, 0, py - TRUNK_BOX[4 * r + 3]);
    if (bx * bx + by * by >= best) continue;
    for (let k = r * TRUNK_RUN + 1, e = Math.min(trunkN - 1, (r + 1) * TRUNK_RUN); k <= e; k++) {
      const x0 = TRUNK[2 * k - 2];
      const y0 = TRUNK[2 * k - 1];
      const dx = TRUNK[2 * k] - x0;
      const dy = TRUNK[2 * k + 1] - y0;
      const l2 = dx * dx + dy * dy || 1e-9;
      const t = Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / l2));
      const ex = px - (x0 + t * dx);
      const ey = py - (y0 + t * dy);
      best = Math.min(best, ex * ex + ey * ey);
    }
  }
  return Math.sqrt(best);
}

/** Bed gravel share 0–1 from the grain size D50 (mm): sand below 2 mm, all gravel from 16 mm. */
const gravelOf = (d50: number): number => smooth01((Math.log2(Math.max(1e-3, d50)) - 1) / 3);

const beltCache = new WeakMap<FieldData, number>();

/** Widest active belt or floodplain anywhere (m): how far beyond its nodes a river's stamps can reach. */
function widestBelt(f: FieldData): number {
  let m = beltCache.get(f);
  if (m === undefined) {
    m = 0;
    for (let k = 0; k < f.rivBelt.length; k++) m = Math.max(m, f.rivBelt[k], f.rivPlain[k]);
    beltCache.set(f, m);
  }
  return m;
}

/** Scratch style for the segment being drawn. */
const STYLE = newChannelStyle();

/** Flow arrows on the drainage layer: about this far apart along a river (px). */
const ARROW_GAP = 110;

/**
 * Rasterizes the river network into COVER (line coverage), CARVE (0–1 valley
 * profile) and RIPARIAN (valley floors of rivers that flow most of the year),
 * and each river's channel pattern: FLOOD (floodplains), BELT/BARE/GRAV (bars
 * and islands), extra threads, SCRL (scroll bars), OXB/SCAR (cut-off loops),
 * PAT (pattern per pixel) and, on the drainage layer, flow arrows (ARW/ARO).
 * Each river ends where it first meets the water drawn at this level; its
 * valley runs on to the sea cell and merges with the valleys it meets.
 */
function rasterRivers(md: MeshData, f: FieldData, L: number, x0g: number, y0g: number, Wg: number, Hg: number, dRad: number, draw: boolean, carve: boolean, lcode: number): void {
  const G = Wg * Hg;
  COVER.fill(0, 0, G);
  BANK.fill(0, 0, G);
  CARVE.fill(0, 0, G);
  RIPARIAN.fill(0, 0, G);
  FLOOD.fill(0, 0, G);
  BELT.fill(0, 0, G);
  BARE.fill(0, 0, G);
  GRAV.fill(0, 0, G);
  SCRL.fill(0, 0, G);
  OXB.fill(0, 0, G);
  SCAR.fill(0, 0, G);
  PAT.fill(0, 0, G);
  ARW.fill(0, 0, G);
  ARO.fill(0, 0, G);
  if (!draw && !carve) return;
  const { rivStart, rivCell, rivQ, rivBank, rivFlow, rivBox, rivJoin } = f;
  const { rivPattern, rivThreads, rivSinu, rivBelt, rivPlain, rivGrain, rivBars, rivConf, rivDist, rivTau, rivBraid, rivAnas } = f;
  const R = f.radiusKm * 1000;
  const mPerPx = R * dRad;
  const thrL = riverThresholdFor(f, md.spacing / dRad);
  const thr0 = f.riverThreshold;
  const reach = riverReach(md, f, L);
  const chainOf = chainIndex(f);
  const noise = noiseFor(f.noiseSeed);
  /** Bars, islands, scrolls and oxbows show in natural color and on the pattern layer. */
  const styled = draw && (lcode === L_TERRAIN || lcode === L_CHANNELS);
  const arrows = draw && lcode === L_DRAIN;
  // Small streams carry no arrows: the bigger rivers are enough to read the flow.
  const arrowThr = thrL * 6;
  /** Drawn bankfull discharge at node k of a segment whose drawn mean is q (what reached the water upstream is gone). */
  const bankOf = (k: number, q: number): number => (rivQ[k] > 0 ? (rivBank[k] * q) / rivQ[k] : 0);
  const maxV = carve ? Math.min(VALLEY_MAX_PX, valleyHalfWidth(1e5, mPerPx)) : 0;
  const maxB = Math.min(VALLEY_MAX_PX, (0.6 * widestBelt(f)) / mPerPx);
  const margin = (Math.max(maxV, maxB, 14) + 2) * dRad;
  const lonMin = -Math.PI + x0g * dRad - margin;
  const lonMax = -Math.PI + (x0g + Wg) * dRad + margin;
  const latMax = Math.PI / 2 - y0g * dRad + margin;
  const latMin = Math.PI / 2 - (y0g + Hg) * dRad - margin;
  // A multi-thread belt is laid out over runs of segments (its bends narrow
  // it over a couple of belt widths either way): segments that far beyond
  // the tile join in, so every tile sees the same run where it shows.
  const beltMargin = Math.max(0, 3.2 * Math.min(VALLEY_MAX_PX, 0.3 * (md.spacing / dRad), (0.5 * widestBelt(f)) / mPerPx) + 4 - Math.max(maxV, maxB, 14)) * dRad;
  const mp = CURVE_MP;
  const st = STYLE;
  // A belt can't be drawn wider than the bends of a course through cells this
  // far apart allow (nor a floodplain much wider than a cell).
  const spacingPx = md.spacing / dRad;
  const beltCap = Math.min(VALLEY_MAX_PX, 0.3 * spacingPx);
  const plainCap = Math.min(VALLEY_MAX_PX, 0.75 * spacingPx);
  const nChains = rivStart.length - 1;
  beltSetup(noise, styled, Wg, Hg);
  for (let ch = 0; ch < nChains; ch++) {
    const s0 = rivStart[ch];
    const s1 = rivStart[ch + 1];
    if (streamX0 <= streamX1) mergeStream(Wg);
    const seedCell = rivCell[s0];
    const chainSeed = 100 * hash01(seedCell, 0, 941);
    const arrowPhase = ARROW_GAP * hash01(seedCell, 0, 942);
    for (let i = s0; i < s1 - 1; i++) {
      // The river is drawn until it runs into the drawn water (sea, or a fjord's
      // drowned trough); its valley is carved all the way to the sea cell.
      // (Water lost upstream only ever thins the drawn line.)
      if (rivQ[i + 1] < thrL) continue;
      // A delta's, fan's or inland delta's distributaries are drawn instead.
      if (f.rivNet[i] >= 0) continue;
      const b0 = rivBox[4 * i];
      const b1 = rivBox[4 * i + 1];
      const bm = rivBraid[i] + rivAnas[i] + rivBraid[i + 1] + rivAnas[i + 1] > 0.02 ? beltMargin : 0;
      if (b1 < latMin - bm || b0 > latMax + bm) continue;
      let shift = NaN;
      for (const s of [0, -2 * Math.PI, 2 * Math.PI]) {
        if (rivBox[4 * i + 2] + s <= lonMax + bm && rivBox[4 * i + 3] + s >= lonMin - bm) {
          shift = s;
          break;
        }
      }
      if (Number.isNaN(shift)) continue;
      // (Beyond the ordinary margin a segment only feeds its belt's run.)
      const farOnly = bm > 0 && (b1 < latMin || b0 > latMax || rivBox[4 * i + 2] + shift > lonMax || rivBox[4 * i + 3] + shift < lonMin);
      reach.resolve(i);
      const qaW = Math.max(0, rivQ[i] - reach.wetDrop[i]);
      const qbW = Math.max(0, rivQ[i + 1] - reach.wetDrop[i]);
      const shown = draw && qbW >= thrL && reach.wetCut[i] !== 0;
      if (!carve && !shown) continue;
      // A tributary's last stretch: where are the trunk's channel and valley?
      const jn = i === s1 - 2 ? rivJoin[ch] : -1;
      let trunkW = 0;
      let trunkV = 0;
      let trunkB = 0;
      trunkN = 0;
      if (jn >= 0) {
        reach.resolve(jn);
        const qT = rivQ[jn] - reach.wetDrop[jn];
        const trunkShown = draw && reach.wetCut[jn] !== 0 && rivQ[jn + 1] - reach.wetDrop[jn] >= thrL;
        const trunkCarved = carve && rivQ[jn + 1] >= thrL;
        if (trunkShown || trunkCarved) trunkCourse(md, f, chainOf[jn], jn, dRad, x0g, y0g, shift);
        if (trunkShown) {
          trunkW = riverHalfWidth(qT, bankOf(jn, qT), thr0, L, mPerPx);
          // A braided or anastomosing trunk takes a tributary in where it reaches the belt.
          trunkB = Math.min(1, rivBraid[jn] + rivAnas[jn]) * 0.7 * Math.min(Math.min(VALLEY_MAX_PX, 0.3 * md.spacing / dRad), (0.5 * rivBelt[jn]) / mPerPx);
        }
        if (trunkCarved) trunkV = Math.min(VALLEY_MAX_PX, valleyHalfWidth(rivQ[jn], mPerPx));
      }
      const nS = riverCurve(md, f, ch, i, dRad, x0g, y0g, shift);
      if (nS === 0) continue;
      // The line runs up to (and just into) the water where the river meets it.
      // The valley runs to the sea cell, and on past it only to reach the drawn
      // water: carved across land the coastline noise added, the run-on would
      // cut a straight trough to nowhere (a fjord, where ice was).
      let nPts = reach.wetCut[i] > 0 ? Math.min(curveN, reach.wetCut[i] + 1) : curveN;
      let nCarve = reach.runCut[i] >= 0 ? Math.min(curveN, reach.runCut[i] + 1) : nS;
      // A tributary ends where it first touches the trunk's channel (or belt):
      // curving into the trunk's flow, it would otherwise run alongside it
      // (three channels side by side) or cross it before reaching the junction.
      let contact = -1;
      if (trunkN >= 2) {
        const reachW = Math.max(trunkW, trunkB, 1);
        // (Distances matter only within the trunk's channel and valley, and up to the contact.)
        for (let s = 0; s < nS; s++) {
          if ((TRUNK_DIST[s] = trunkDistance(mp[2 * s], mp[2 * s + 1], Math.max(reachW, trunkV) + 1)) <= reachW) {
            contact = s;
            break;
          }
        }
        if (contact >= 0) {
          nPts = Math.min(nPts, contact + 1);
          nCarve = Math.min(nCarve, contact + 1);
        }
      }
      // Flow regime: rivers that run only part of the year are dashed (seasonal)
      // or dotted (ephemeral) once dashes can be seen, and fainter.
      const months = Math.min(rivFlow[i], rivFlow[i + 1]);
      const wet = smooth01((months - 2) / 8);
      const alpha = months >= 11 ? 1 : months >= 4 ? 0.85 : 0.6;
      // The channel's style along this segment, from its two nodes.
      const pa = rivPattern[i];
      const pb = rivPattern[i + 1];
      const bankA = bankOf(i, qaW);
      const bankB = bankOf(i + 1, qbW);
      st.belt0 = Math.min(beltCap, (0.5 * rivBelt[i]) / mPerPx);
      st.belt1 = Math.min(beltCap, (0.5 * rivBelt[i + 1]) / mPerPx);
      st.plain0 = Math.min(plainCap, (0.5 * rivPlain[i] * (1 - rivConf[i])) / mPerPx);
      st.plain1 = Math.min(plainCap, (0.5 * rivPlain[i + 1] * (1 - rivConf[i + 1])) / mPerPx);
      st.threads0 = rivThreads[i];
      st.threads1 = rivThreads[i + 1];
      st.bars0 = rivBars[i];
      st.bars1 = rivBars[i + 1];
      st.gravel0 = gravelOf(rivGrain[i]);
      st.gravel1 = gravelOf(rivGrain[i + 1]);
      st.braid0 = rivBraid[i];
      st.braid1 = rivBraid[i + 1];
      st.anas0 = rivAnas[i];
      st.anas1 = rivAnas[i + 1];
      // (Point bars, scrolls and oxbows belong to a main channel that shows.)
      st.meander0 = MEANDER_OF[pa] * (1 - rivConf[i]) * (1 - smooth01((Math.min(1, rivBraid[i] + rivAnas[i]) - 0.35) / 0.4));
      st.meander1 = MEANDER_OF[pb] * (1 - rivConf[i + 1]) * (1 - smooth01((Math.min(1, rivBraid[i + 1] + rivAnas[i + 1]) - 0.35) / 0.4));
      st.tau0 = rivTau[i];
      st.tau1 = rivTau[i + 1];
      st.chan0 = (CHANNEL_WIDTH * Math.sqrt(bankA)) / mPerPx;
      st.chan1 = (CHANNEL_WIDTH * Math.sqrt(bankB)) / mPerPx;
      st.sinu0 = rivSinu[i];
      st.sinu1 = rivSinu[i + 1];
      st.pat0 = pa + 1;
      st.pat1 = pb + 1;
      st.seed = chainSeed;
      st.alpha = alpha * (months >= 4 ? 1 : 0.7);
      const multi = st.braid0 + st.braid1 + st.anas0 + st.anas1 > 0.02;
      if (carve && !farOnly) {
        const V0 = Math.min(VALLEY_MAX_PX, valleyHalfWidth(Math.max(rivQ[i], rivQ[i + 1]), mPerPx));
        if (V0 >= 1.2) {
          // Along the meander belt's axis: the channel swings within its valley floor.
          const ax = CURVE_AX;
          const step = Math.max(1, Math.round(V0 / 2.5));
          let px0 = ax[0];
          let py0 = ax[1];
          for (let s = step; s < nCarve + step - 1; s += step) {
            const si = Math.min(nCarve - 1, s);
            // Within the trunk's valley a tributary's valley opens out into it.
            const V = trunkN >= 2 && trunkV > V0 ? V0 + (trunkV - V0) * smooth01(1 - TRUNK_DIST[si] / trunkV) : V0;
            stampCarve(ax[2 * si], ax[2 * si + 1], px0, py0, V, wet, Wg, Hg);
            px0 = ax[2 * si];
            py0 = ax[2 * si + 1];
          }
          // Its floodplain (and active belt): a flat valley floor where it has room.
          if (Math.max(st.plain0, st.plain1, st.belt0, st.belt1) >= 1.5) stampFloodplain(st, nCarve, Wg, Hg);
        }
      }
      if (!shown || nPts < 2) continue;
      // A channel wide enough to show its bed is drawn whole even when it runs dry, only paler.
      const dash = months >= 11 || L < 4 || riverHalfWidth(qbW, bankB, thr0, L, mPerPx) >= 1.5 ? 0 : months >= 4 ? 1 : 2;
      const on = dash === 1 ? 7 : 2.2;
      const period = dash === 1 ? 11 : 6;
      // Over its last couple of trunk widths a tributary opens out a little into
      // a mouth, rounding off the junction.
      const flare = Math.max(2, (4 * trunkW) / 2.5);
      const widthAt = (s: number): number => {
        const u = Math.min(1, s / (nS - 1));
        let w = riverHalfWidth(qaW + (qbW - qaW) * u, bankA + (bankB - bankA) * u, thr0, L, mPerPx);
        if (multi) {
          // Once its belt shows, a braided or anastomosing river's main channel
          // carries only its share of the flow; the other threads the rest.
          const vs = CURVE_V[s];
          const shownBelt = (lerpV(st.braid0, st.braid1, vs) + lerpV(st.anas0, st.anas1, vs)) * smooth01((lerpV(st.belt0, st.belt1, vs) - 2) / 4);
          if (shownBelt > 0) w *= 1 - Math.min(1, shownBelt) * (1 - Math.sqrt(Math.min(1, 1.6 / Math.max(1, lerpV(st.threads0, st.threads1, vs)))));
        }
        if (contact < 0) return w;
        const mouth = Math.min(trunkW, 1.6 * w);
        return mouth > w ? w + (mouth - w) * smooth01(1 - (contact - s) / flare) : w;
      };
      // The belt joins its chain's run (drawn where the run ends).
      // Where a belt's own channels show in full they carry the river (its
      // network is connected, the outermost channels included): the main
      // channel fades out into them.
      const mainVis = (s: number): number => {
        if (!multi) return 1;
        const vs = CURVE_V[s];
        const wm = Math.min(1, lerpV(st.braid0, st.braid1, vs) + lerpV(st.anas0, st.anas1, vs));
        const hb = Math.min(beltCap, lerpV(st.belt0, st.belt1, vs));
        const hc = 0.5 * lerpV(st.chan0, st.chan1, vs);
        // (Quickly: a half-faded main channel lies over the islands like a haze.)
        return 1 - smooth01((wm - 0.28) / 0.22) * smooth01((hc + (hb - hc) * wm - 4) / 4);
      };
      if (multi && Math.max(st.belt0, st.belt1) >= 0.5) beltAppend(st, nPts, beltCap, i, nPts < nS || contact >= 0 || i === s1 - 2, contact >= 0);
      if (farOnly) continue;
      if (styled && st.meander0 + st.meander1 > 0.1) {
        stampBends(st, nPts, widthAt, Wg, Hg);
        if (carve) stampOxbows(st, nPts, widthAt, seedCell, Wg, Hg);
      }
      // Arrows: placed by distance down the chain (its nodes' along-chain
      // distance, spread over the drawn curve), the same in every tile.
      const doArrows = arrows && qbW >= arrowThr;
      let arcTot = 0;
      if (doArrows) for (let s = 1; s < nS; s++) arcTot += Math.hypot(mp[2 * s] - mp[2 * s - 2], mp[2 * s + 1] - mp[2 * s - 1]);
      const gA = rivDist[i] / mPerPx;
      const gB = rivDist[i + 1] / mPerPx;
      const along = (a: number): number => (a <= arcTot ? gA + ((gB - gA) * a) / Math.max(arcTot, 1e-9) : gB + (a - arcTot));
      // None just above a confluence (the trunk has its own) or right at a mouth.
      let arcDrawn = 0;
      if (doArrows) for (let s = 1; s < nPts; s++) arcDrawn += Math.hypot(mp[2 * s] - mp[2 * s - 2], mp[2 * s + 1] - mp[2 * s - 1]);
      const endGap = contact >= 0 ? 0.5 * ARROW_GAP : reach.wetCut[i] > 0 ? 12 : 0;
      let lw = widthAt(0);
      let arc = 0;
      for (let s = 1; s < nPts; s++) {
        const w = widthAt(s);
        const seg = Math.hypot(mp[2 * s] - mp[2 * s - 2], mp[2 * s + 1] - mp[2 * s - 1]);
        const pat = CURVE_V[s] < 0.5 ? st.pat0 : st.pat1;
        const vis = mainVis(s);
        if (vis > 0.12 && (dash === 0 || (arc + seg / 2) % period < on)) stampLine(mp[2 * s - 2], mp[2 * s - 1], mp[2 * s], mp[2 * s + 1], lw, w, alpha * vis, pat, Wg, Hg);
        if (doArrows && arcDrawn - arc > endGap) {
          const g0 = along(arc) + arrowPhase;
          const g1 = along(arc + seg) + arrowPhase;
          const m1 = Math.floor(g1 / ARROW_GAP);
          if (m1 > Math.floor(g0 / ARROW_GAP) && g1 > g0) {
            const fr = (m1 * ARROW_GAP - g0) / (g1 - g0);
            const cx = mp[2 * s - 2] + (mp[2 * s] - mp[2 * s - 2]) * fr;
            const cy = mp[2 * s - 1] + (mp[2 * s + 1] - mp[2 * s - 1]) * fr;
            const ka = Math.max(0, s - 3);
            const kb = Math.min(nPts - 1, s + 2);
            const ux = mp[2 * kb] - mp[2 * ka];
            const uy = mp[2 * kb + 1] - mp[2 * ka + 1];
            const ul = Math.hypot(ux, uy);
            if (ul > 1e-6) {
              const len = Math.max(7, Math.min(14, 6 + 1.6 * w));
              stampArrow(cx, cy, ux / ul, uy / ul, len, 0.9 * len, months >= 11 ? 1 : 0.55, Wg, Hg);
            }
          }
        }
        arc += seg;
        lw = w;
      }
    }
    beltFlush();
  }
  if (streamX0 <= streamX1) mergeStream(Wg);
}

function stampLine(x0: number, y0: number, x1: number, y1: number, h0: number, h1: number, alpha: number, pat: number, Wg: number, Hg: number): void {
  // A channel wide enough to show them gets banks: a rim just outside the water.
  const bank = Math.max(h0, h1) >= 1.5 ? 1.3 : 0;
  const r = Math.max(h0, h1) + 1 + bank;
  const minX = Math.max(0, Math.floor(Math.min(x0, x1) - r));
  const maxX = Math.min(Wg - 1, Math.ceil(Math.max(x0, x1) + r));
  const minY = Math.max(0, Math.floor(Math.min(y0, y1) - r));
  const maxY = Math.min(Hg - 1, Math.ceil(Math.max(y0, y1) + r));
  if (minX > maxX || minY > maxY) return;
  const dx = x1 - x0;
  const dy = y1 - y0;
  const l2 = dx * dx + dy * dy || 1e-9;
  for (let py = minY; py <= maxY; py++) {
    for (let px = minX; px <= maxX; px++) {
      const t = Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / l2));
      const d = Math.hypot(px - (x0 + t * dx), py - (y0 + t * dy));
      const a = h0 + (h1 - h0) * t + 0.5 - d;
      if (a + bank <= 0) continue;
      const p = py * Wg + px;
      if (bank > 0) {
        const vb = Math.min(1, a + bank) * alpha;
        if (vb > BANK[p]) BANK[p] = vb;
      }
      if (a <= 0) continue;
      const v = (a > 1 ? 1 : a) * alpha;
      if (v > COVER[p]) {
        COVER[p] = v;
        PAT[p] = pat;
      }
    }
  }
}

// ------------------------------------------------------------------ channel patterns

/** How much of each pattern (world/channels.ts Channel order) meanders over a floodplain (braided and anastomosing weights come smoothed with the chain: rivBraid, rivAnas). */
const MEANDER_OF = [0.5, 0, 0.2, 1, 0.5, 0, 0.35, 0.4];

/**
 * A channel's style at both ends of a stretch of its course, which the
 * pattern stamps blend between by each curve point's place along the stretch
 * (CURVE_V). Filled from the chain's nodes for rivers; a distributary network
 * fills it from its own nodes. Widths are in pixels at the level drawn.
 */
export interface ChannelStyle {
  /** Active belt half-width (braidplain, anastomosing net). */
  belt0: number;
  belt1: number;
  /** Floodplain half-width (0 where valley walls hem the river in). */
  plain0: number;
  plain1: number;
  /** Channels across the belt (1: one thread). */
  threads0: number;
  threads1: number;
  /** 0–1: how bare the ground between channels is (bars), against vegetated islands. */
  bars0: number;
  bars1: number;
  /** 0–1: how gravelly the bars are. */
  gravel0: number;
  gravel1: number;
  /** 0–1 pattern weights: braided, anastomosing, meandering (point bars, scrolls, oxbows). */
  braid0: number;
  braid1: number;
  anas0: number;
  anas1: number;
  meander0: number;
  meander1: number;
  /** Along-channel coordinate in belt widths (global, so tiles and levels agree). */
  tau0: number;
  tau1: number;
  /** Single-channel width at bankfull, in pixels: the threads share it. */
  chan0: number;
  chan1: number;
  /** Sinuosity of the main channel (oxbows form on tightly winding rivers). */
  sinu0: number;
  sinu1: number;
  /** Pattern codes at the ends (Channel + 1), for the channel-pattern layer. */
  pat0: number;
  pat1: number;
  /** Noise offset of this channel (its own braids and islands, 0–100). */
  seed: number;
  /** Drawn opacity: rivers that run only part of the year are fainter. */
  alpha: number;
}

export function newChannelStyle(): ChannelStyle {
  return {
    belt0: 0, belt1: 0, plain0: 0, plain1: 0, threads0: 1, threads1: 1, bars0: 0, bars1: 0, gravel0: 0, gravel1: 0,
    braid0: 0, braid1: 0, anas0: 0, anas1: 0, meander0: 0, meander1: 0, tau0: 0, tau1: 0, chan0: 0, chan1: 0,
    sinu0: 1, sinu1: 1, pat0: 0, pat1: 0, seed: 0, alpha: 1,
  };
}

const lerpV = (a: number, b: number, v: number): number => a + (b - a) * (v <= 0 ? 0 : v >= 1 ? 1 : v);

/** Unit normal (to the left, px) of the curve CURVE_AX at point s of n: into NRM. */
const NRM = new Float64Array(2);
function axisNormal(s: number, n: number): void {
  const ax = CURVE_AX;
  const a = Math.max(0, s - 1);
  const b = Math.min(n - 1, s + 1);
  const tx = ax[2 * b] - ax[2 * a];
  const ty = ax[2 * b + 1] - ax[2 * a + 1];
  const l = Math.hypot(tx, ty) || 1;
  NRM[0] = -ty / l;
  NRM[1] = tx / l;
}

/**
 * The flat valley floor a river wanders over, and its active belt, into FLOOD
 * (0–1), which the pixel loop flattens toward the river's level. Along the
 * axis of the last curve (points 0..n-1), as CURVE_V places them in `st`.
 */
export function stampFloodplain(st: ChannelStyle, n: number, Wg: number, Hg: number): void {
  const ax = CURVE_AX;
  const v = CURVE_V;
  const half = (s: number): number => Math.min(VALLEY_MAX_PX, Math.max(lerpV(st.plain0, st.plain1, v[s]), 1.1 * lerpV(st.belt0, st.belt1, v[s])));
  let s0 = 0;
  while (s0 < n - 1) {
    const h0 = half(s0);
    const s1 = Math.min(n - 1, s0 + Math.max(1, Math.round(h0 / 2.5)));
    const h1 = half(s1);
    if (Math.max(h0, h1) >= 1.5) stampBand(ax[2 * s0], ax[2 * s0 + 1], ax[2 * s1], ax[2 * s1 + 1], h0, h1, Wg, Hg);
    s0 = s1;
  }
}

function stampBand(x0: number, y0: number, x1: number, y1: number, h0: number, h1: number, Wg: number, Hg: number): void {
  const r = Math.max(h0, h1) + 1;
  const minX = Math.max(0, Math.floor(Math.min(x0, x1) - r));
  const maxX = Math.min(Wg - 1, Math.ceil(Math.max(x0, x1) + r));
  const minY = Math.max(0, Math.floor(Math.min(y0, y1) - r));
  const maxY = Math.min(Hg - 1, Math.ceil(Math.max(y0, y1) + r));
  if (minX > maxX || minY > maxY) return;
  const dx = x1 - x0;
  const dy = y1 - y0;
  const l2 = dx * dx + dy * dy || 1e-9;
  for (let py = minY; py <= maxY; py++) {
    for (let px = minX; px <= maxX; px++) {
      const t = Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / l2));
      const h = h0 + (h1 - h0) * t;
      const ex = px - (x0 + t * dx);
      const ey = py - (y0 + t * dy);
      const d2 = ex * ex + ey * ey;
      if (d2 >= h * h) continue;
      const u = Math.sqrt(d2) / h;
      const val = u < 0.55 ? 1 : smooth01((1 - u) / 0.45);
      const p = py * Wg + px;
      if (val > FLOOD[p]) FLOOD[p] = val;
    }
  }
}

/**
 * A braided, wandering or anastomosing river's belt is drawn from the points
 * of its course's belt axis (CURVE_BAX) gathered over a run of consecutive
 * segments of its chain, so the belt bends, narrows and widens smoothly
 * through the nodes instead of each segment's belt ending on its own. Per
 * point: position, channel and belt half-widths (px), multi-thread weight,
 * τ, bars, gravel, braid and anastomosing weights, threads, channel width
 * (px), opacity, and 1 where a segment ends (a vertex always sits there).
 */
const BB_N = 14;
let BB = new Float64Array(BB_N * 4096);
let bbN = 0;
/** Segment last appended (-2: none), the run's noise key, and whether it ends where a tributary meets its trunk. */
let bbSeg = -2;
let bbKey = 0;
let bbTaper = false;
/** What the current rasterRivers call draws belts with. */
let bbNoise: Simplex3 | null = null;
let bbStyled = false;
let bbWg = 0;
let bbHg = 0;
/**
 * Belt vertices: position, unit normal (left), half-width, outer half-width
 * (px), τ, bars, gravel, edge wobble (left, right), braid and anastomosing
 * weights, threads, channel width (px), opacity.
 */
const BV_N = 16;
let BV = new Float64Array(BV_N * 4096);
/** Per belt vertex: the widest half-width its bend allows (px), and its distance along the belt (px). */
let BEND = new Float64Array(4096);
let ARC_V = new Float64Array(4096);

function beltSetup(noise: Simplex3, styled: boolean, Wg: number, Hg: number): void {
  bbNoise = noise;
  bbStyled = styled;
  bbWg = Wg;
  bbHg = Hg;
  bbN = 0;
  bbSeg = -2;
  bbTaper = false;
}

/**
 * Adds the belt of the last curve (points 0..n-1, styled by `st`) as segment
 * `seg` of the run; a segment that doesn't continue the run starts a new one.
 * `end`: the run stops here (the river reaches water or its trunk); `taper`:
 * a tributary's belt narrows into its channel toward this end.
 */
function beltAppend(st: ChannelStyle, n: number, maxHalf: number, seg: number, end: boolean, taper: boolean): void {
  const key = Math.floor(st.seed * 1000);
  if (bbN > 0 && (seg !== bbSeg + 1 || key !== bbKey)) beltFlush();
  bbKey = key;
  const ax = CURVE_BAX;
  const v = CURVE_V;
  const s0 = bbN > 0 ? 1 : 0;
  if (BB.length < BB_N * (bbN + n)) {
    const nb = new Float64Array(BB_N * 2 * (bbN + n));
    nb.set(BB.subarray(0, BB_N * bbN));
    BB = nb;
  }
  for (let s = s0; s < n; s++) {
    const vs = v[s];
    const o = BB_N * bbN++;
    const br = lerpV(st.braid0, st.braid1, vs);
    const an = lerpV(st.anas0, st.anas1, vs);
    const chan = lerpV(st.chan0, st.chan1, vs);
    BB[o] = ax[2 * s];
    BB[o + 1] = ax[2 * s + 1];
    BB[o + 2] = 0.5 * chan;
    BB[o + 3] = Math.min(maxHalf, lerpV(st.belt0, st.belt1, vs));
    BB[o + 4] = Math.min(1, br + an);
    BB[o + 5] = lerpV(st.tau0, st.tau1, vs);
    BB[o + 6] = lerpV(st.bars0, st.bars1, vs);
    BB[o + 7] = lerpV(st.gravel0, st.gravel1, vs);
    BB[o + 8] = br;
    BB[o + 9] = an;
    BB[o + 10] = lerpV(st.threads0, st.threads1, vs);
    BB[o + 11] = chan;
    BB[o + 12] = st.alpha;
    BB[o + 13] = s === n - 1 || s === 0 ? 1 : 0;
  }
  bbSeg = seg;
  if (taper) bbTaper = true;
  if (end) beltFlush();
}

/**
 * Draws the run gathered so far: vertices every quarter belt half-width or so
 * along the axis (always at the nodes, so every tile places them alike), the
 * belt kept inside the radius of the axis's bends, then each quad between
 * consecutive vertices (see stampQuad).
 */
function beltFlush(): void {
  const n = bbN;
  const taperEnd = bbTaper;
  bbN = 0;
  bbSeg = -2;
  bbTaper = false;
  if (n < 2 || !bbNoise) return;
  const noise = bbNoise;
  const B = BB;
  if (BV.length < BV_N * n) {
    BV = new Float64Array(BV_N * n);
    BEND = new Float64Array(n);
    ARC_V = new Float64Array(n);
  }
  // Where a tributary's belt meets its trunk it narrows into its channel over its last few belt widths.
  let arcTotal = 0;
  if (taperEnd) for (let s = 1; s < n; s++) arcTotal += Math.hypot(B[BB_N * s] - B[BB_N * (s - 1)], B[BB_N * s + 1] - B[BB_N * (s - 1) + 1]);
  let arcHere = 0;
  let nv = 0;
  let acc = 0;
  for (let s = 0; s < n; s++) {
    const o = BB_N * s;
    if (s > 0) {
      const ds = Math.hypot(B[o] - B[o - BB_N], B[o + 1] - B[o - BB_N + 1]);
      acc += ds;
      arcHere += ds;
    }
    const hc = B[o + 2];
    const hb = B[o + 3];
    const taper = taperEnd ? smooth01((arcTotal - arcHere) / Math.max(4, 3 * hb)) : 1;
    // The belt widens out of the channel as the river takes on its pattern.
    const h = hc + (hb - hc) * B[o + 4] * taper;
    if (B[o + 13] === 0 && s < n - 1 && acc < Math.max(2, Math.min(6, 0.25 * h))) continue;
    acc = 0;
    // Normal from the neighboring points, across nodes too.
    const a = BB_N * Math.max(0, s - 1);
    const b = BB_N * Math.min(n - 1, s + 1);
    const tx = B[b] - B[a];
    const ty = B[b + 1] - B[a + 1];
    const tl = Math.hypot(tx, ty) || 1;
    const q = BV_N * nv;
    const tau = B[o + 5];
    BV[q] = B[o];
    BV[q + 1] = B[o + 1];
    BV[q + 2] = -ty / tl;
    BV[q + 3] = tx / tl;
    BV[q + 4] = h;
    BV[q + 6] = tau;
    BV[q + 7] = B[o + 6];
    BV[q + 8] = B[o + 7];
    BV[q + 9] = 0.11 * noise.noise(tau * 1.3 + 3.1, 2.3, bbKey * 0.001);
    BV[q + 10] = 0.11 * noise.noise(tau * 1.3 + 3.1, -2.3, bbKey * 0.001);
    BV[q + 11] = B[o + 8] * taper;
    BV[q + 12] = B[o + 9] * taper;
    BV[q + 13] = B[o + 10];
    BV[q + 14] = B[o + 11];
    BV[q + 15] = B[o + 12];
    nv++;
  }
  // Keep the belt inside the radius of the axis's bends (its quads would fold
  // over), narrowing it gradually toward a tight bend rather than notching it.
  // Each quad's bend radius (its length over the turn of its normals) bounds
  // the outer half-width (1.15 h + 1.5) of the vertices at its ends.
  for (let j = 0; j < nv; j++) BEND[j] = Infinity;
  for (let j = 0; j + 1 < nv; j++) {
    const o = BV_N * j;
    const c = BV[o + 2] * BV[o + BV_N + 3] - BV[o + 3] * BV[o + BV_N + 2];
    const ang = Math.abs(Math.asin(Math.max(-1, Math.min(1, c))));
    if (ang < 1e-6) continue;
    const rad = Math.hypot(BV[o + BV_N] - BV[o], BV[o + BV_N + 1] - BV[o + 1]) / ang;
    const lim = Math.max(0, (0.95 * rad - 2) / 1.3);
    BEND[j] = Math.min(BEND[j], lim);
    BEND[j + 1] = Math.min(BEND[j + 1], lim);
  }
  // (Narrowing toward a tight bend at half a pixel per pixel along the belt.)
  ARC_V[0] = 0;
  for (let j = 1; j < nv; j++) ARC_V[j] = ARC_V[j - 1] + Math.hypot(BV[BV_N * j] - BV[BV_N * j - BV_N], BV[BV_N * j + 1] - BV[BV_N * j - BV_N + 1]);
  for (let j = 0; j < nv; j++) {
    let lim = Infinity;
    const reachArc = 2 * BV[BV_N * j + 4];
    for (let i = j; i >= 0 && ARC_V[j] - ARC_V[i] <= reachArc; i--) lim = Math.min(lim, BEND[i] + 0.5 * (ARC_V[j] - ARC_V[i]));
    for (let i = j + 1; i < nv && ARC_V[i] - ARC_V[j] <= reachArc; i++) lim = Math.min(lim, BEND[i] + 0.5 * (ARC_V[i] - ARC_V[j]));
    const o = BV_N * j;
    if (BV[o + 4] > lim) BV[o + 4] = lim;
  }
  // (The quads reach past the belt's edge, so channels along it are drawn whole.)
  for (let j = 0; j < nv; j++) BV[BV_N * j + 5] = 1.3 * BV[BV_N * j + 4] + 2;
  for (let j = 0; j + 1 < nv; j++) {
    const o0 = BV_N * j;
    const o1 = o0 + BV_N;
    if (Math.max(BV[o0 + 4], BV[o1 + 4]) < 0.5) continue;
    stampQuad(noise, o0, o1, bbKey, bbWg, bbHg);
  }
}

/**
 * A single curve's belt (a distributary's channel): the last curve, points
 * 0..n-1, styled by `st` (see beltAppend), drawn at once.
 */
export function stampBeltField(noise: Simplex3, st: ChannelStyle, n: number, maxHalf: number, belt: boolean, taperEnd: boolean, Wg: number, Hg: number): void {
  if (n < 2) return;
  const saved = [bbNoise, bbStyled, bbWg, bbHg] as const;
  beltFlush();
  bbNoise = noise;
  bbStyled = belt;
  bbWg = Wg;
  bbHg = Hg;
  beltAppend(st, n, maxHalf, -10, true, taperEnd);
  [bbNoise, bbStyled, bbWg, bbHg] = saved;
}

/** Smooth value noise in about [-1, 1] over an integer lattice, with a one-cell cache per slot (neighboring pixels share a cell). */
const VN_KEY = new Float64Array([NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]);
const VN_V = new Float64Array(16);
/** (Slots 0–3: the braid warp's two octaves across and along.) */
function vnoise(slot: number, x: number, y: number, seed: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const o = 4 * slot;
  const k3 = 3 * slot;
  if (VN_KEY[k3] !== ix || VN_KEY[k3 + 1] !== iy || VN_KEY[k3 + 2] !== seed) {
    VN_KEY[k3] = ix;
    VN_KEY[k3 + 1] = iy;
    VN_KEY[k3 + 2] = seed;
    VN_V[o] = ihash(ix, iy, seed);
    VN_V[o + 1] = ihash(ix + 1, iy, seed);
    VN_V[o + 2] = ihash(ix, iy + 1, seed);
    VN_V[o + 3] = ihash(ix + 1, iy + 1, seed);
  }
  const fx = x - ix;
  const fy = y - iy;
  const sx = fx * fx * (3 - 2 * fx);
  const sy = fy * fy * (3 - 2 * fy);
  const a = VN_V[o] + (VN_V[o + 1] - VN_V[o]) * sx;
  const b = VN_V[o + 2] + (VN_V[o + 3] - VN_V[o + 2]) * sx;
  return 2 * (a + (b - a) * sy) - 1;
}

/** Integer hash of three integers to [0, 1). */
function ihash(a: number, b: number, c: number): number {
  let h = Math.imul(a ^ 0x5bd1e995, 0x85ebca6b) ^ Math.imul(b + 0x632be5ab, 0xc2b2ae35) ^ Math.imul(c ^ 0x27d4eb2f, 0x165667b1);
  h ^= h >>> 15;
  h = Math.imul(h, 0x2c1b3c6d);
  h ^= h >>> 12;
  h = Math.imul(h, 0x297a2d39);
  h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}

/**
 * The braid network is laid out on a jittered lattice in belt coordinates:
 * columns NET_LA belt widths long down the river (τ), rows across it. Each
 * lattice cell's Voronoi region is a lens of bar or island; runs of cells
 * along a row are merged into one bar or island, and the channels are the
 * boundaries between them. Boundaries of a partition never end in mid-air,
 * so every channel parts from another and rejoins one: no loose threads.
 */
const NET_LA = 0.95;
/** Rows of cells across the belt, their height (belt half-widths), and where the outermost channels run (belt half-widths from the axis). */
const NET_ROWS = 3;
const NET_LB = 0.6;
const NET_EDGE = 0.9;
/** Lattice points of the cells around the pixel (3×3), coarse then fine: position, column, row. */
const NP_X = new Float64Array(18);
const NP_Y = new Float64Array(18);
const NP_I = new Int32Array(18);
const NP_J = new Int32Array(18);
/** The lattice cell and key each level's points were last made for (neighboring pixels mostly share them). */
const NP_CI = new Float64Array([NaN, NaN]);
const NP_CJ = new Float64Array(2);
const NP_KEY = new Float64Array(2);
/** Results of braidNet: water coverage (0–1, before opacity), bare ground between the channels (0–1), and how much of the pixel is belt rather than floodplain beyond it (0–1). */
let netWater = 0;
let netBare = 1;
let netBelt = 1;

/**
 * Does row j's run of cells break between columns i-1 and i? Runs are at
 * most four cells long (a break every fourth column at least), so a cell's
 * run can be found by looking a few cells either way.
 */
function runBreak(i: number, j: number, key: number, pB: number): boolean {
  return ((i + Math.floor(4 * ihash(j, 7, key))) & 3) === 0 || ihash(i, j, key + 5) < pB;
}

function sameRun(i1: number, i2: number, j: number, key: number, pB: number): boolean {
  const a = i1 < i2 ? i1 : i2;
  const b = i1 < i2 ? i2 : i1;
  for (let i = a + 1; i <= b; i++) if (runBreak(i, j, key, pB)) return false;
  return true;
}

/** Fills level lv's NP_* (9 entries from 9·lv) with the 3×3 lattice points around (X, Y) and returns the nearest's index. */
function nearestPoint(lv: number, X: number, Y: number, key: number, jx: number, jy: number): number {
  const ci = Math.floor(X);
  const cj = Math.floor(Y);
  const k0 = 9 * lv;
  if (ci !== NP_CI[lv] || cj !== NP_CJ[lv] || key !== NP_KEY[lv]) {
    NP_CI[lv] = ci;
    NP_CJ[lv] = cj;
    NP_KEY[lv] = key;
    let k = k0;
    for (let dj = -1; dj <= 1; dj++) {
      const jj = cj + dj;
      for (let di = -1; di <= 1; di++, k++) {
        const ii = ci + di;
        NP_X[k] = ii + 0.5 + jx * (ihash(ii, jj, key) - 0.5);
        NP_Y[k] = jj + 0.5 + jy * (ihash(ii, jj, key + 1) - 0.5);
        NP_I[k] = ii;
        NP_J[k] = jj;
      }
    }
  }
  let best = k0;
  let bd = Infinity;
  for (let k = k0; k < k0 + 9; k++) {
    const dx = NP_X[k] - X;
    const dy = NP_Y[k] - Y;
    const d = dx * dx + dy * dy;
    if (d < bd) {
      bd = d;
      best = k;
    }
  }
  return best;
}

/**
 * The channel network of a multi-thread belt at one pixel: τ along the
 * river, nPx across the belt axis (px), h the belt half-width and pa pixels
 * per unit τ there; br/an the braided and anastomosing weights, nT threads,
 * chan the single channel's width (px), bars how bare the ground between
 * channels is. Writes netWater and netBare.
 *
 * Coarse cells make bars and islands about a belt width long, a few across;
 * braided ground is mostly bare, cut again by finer channels where the river
 * has many threads, with a few older islands overgrown (toward the belt's
 * edges most); an anastomosing river's islands are long and wooded, its
 * channels few and winding; a wandering river has one main channel (drawn
 * separately) and only a few side channels. The lattice is warped so the
 * channels wind instead of running straight between junctions.
 */
function braidNet(noise: Simplex3, tau: number, nPx: number, h: number, pa: number, br: number, an: number, nT: number, chan: number, bars: number, key: number): void {
  const wsum = br + an;
  const fa = an / Math.max(wsum, 1e-6);
  const wander = (1 - Math.min(1, br)) * (1 - fa);
  // Run breaks, quantized so a cell's run and its neighbors' agree from pixel to pixel.
  const pB = Math.round(20 * (0.55 - 0.32 * fa - 0.25 * wander)) / 20;
  // Three rows of cells span the belt between its outermost channels (at
  // about 0.9 of its half-width); rows beyond are the floodplain.
  let X = tau / NET_LA;
  let Y = (nPx / h + NET_EDGE) / NET_LB;
  // (Cheap cached value noise: two simplex calls per pixel cost too much here.)
  // (Gentle enough that no stretch of channel is squeezed thinner than drawn.)
  const wx = vnoise(0, X * 0.8, Y * 1.0, key + 31) + 0.3 * vnoise(2, X * 1.8 + 3.3, Y * 2.2, key + 41);
  const wy = vnoise(1, X * 0.8 + 7.7, Y * 1.0 - 3.1, key + 37) + 0.3 * vnoise(3, X * 1.8 - 5.1, Y * 2.2 + 1.9, key + 43);
  X += 0.38 * wx;
  Y += (0.25 + 0.08 * fa) * wy;
  const best = nearestPoint(0, X, Y, key, 0.8, 0.6);
  const bi = NP_I[best];
  const bj = NP_J[best];
  const outB = bj < 0 || bj >= NET_ROWS;
  // Pixel lengths of a lattice unit along and across.
  const sx = NET_LA * pa;
  const sy = NET_LB * h;
  // Channel half-width here: wider and narrower reaches, varying smoothly
  // along each channel (a few big channels, many small).
  // (The warp's own noise, decorrelated: a third noise call per pixel costs too much.)
  const sW = 0.5 + 0.5 * Math.sin(2.7 * wx - 1.9 * wy + 0.6);
  // (Never under about a pixel wide: a channel fading out reads as a broken thread.)
  // (Nor wider than about a fifth of a row: a great river's channels, as wide
  // as its capped belt allows, would otherwise merge into one sheet of water.)
  const half = Math.max(0.6, Math.min(0.2 * sy, Math.max(0.45, (0.95 * chan) / Math.sqrt(Math.max(1, nT))) * (1 - 0.3 * wander)) * (0.4 + 1.25 * sW * sW));
  // Distance to the nearest boundary between runs, rounded where boundaries
  // meet (a smooth minimum: channels part and join in curves, not corners).
  const r = 0.3 * half + 0.6;
  let sum = 0;
  let dEdge = Infinity;
  let edgeOut = false;
  for (let k = 0; k < 9; k++) {
    if (k === best) continue;
    const jk = NP_J[k];
    const outK = jk < 0 || jk >= NET_ROWS;
    // Cells of one run (or both beyond the belt) have no channel between them.
    if (outB || outK ? outB && outK : jk === bj && sameRun(bi, NP_I[k], bj, key, pB)) continue;
    const ex = NP_X[k] - NP_X[best];
    const ey = NP_Y[k] - NP_Y[best];
    const el = Math.sqrt(ex * ex + ey * ey);
    const nx = ex / el;
    const ny = ey / el;
    // Distance to the boundary (in lattice units, then pixels across it).
    const delta = ((NP_X[k] + NP_X[best]) * 0.5 - X) * nx + ((NP_Y[k] + NP_Y[best]) * 0.5 - Y) * ny;
    // (The belt's outermost channels, along its banks, are narrower: counted
    // farther. Never vanishing, else they would show as loose threads.)
    const dpx = delta / Math.sqrt((nx / sx) * (nx / sx) + (ny / sy) * (ny / sy)) + (outB || outK ? half * (0.15 + 0.35 * (0.5 + 0.5 * Math.sin(1.3 * X + 3 * wx))) : 0);
    if (dpx < dEdge) {
      dEdge = dpx;
      edgeOut = outB || outK;
    }
    if (dpx < half + 5 * r) sum += Math.exp(-Math.max(0, dpx) / r);
  }
  const dS = sum > 0 ? -r * Math.log(sum) : Infinity;
  let water = half - dS + 0.5;
  // The bar or island this pixel lies on: its run, and whether it is overgrown.
  let i0 = bi;
  let i1 = bi;
  if (!outB) {
    for (let s = 0; s < 3 && !runBreak(i0, bj, key, pB); s++) i0--;
    for (let s = 0; s < 3 && !runBreak(i1 + 1, bj, key, pB); s++) i1++;
  }
  const edgeness = bj === 1 ? 0 : 1;
  const pVeg = Math.min(0.97, 0.1 + 0.88 * (1 - bars) + 0.25 * edgeness + 0.05 * (i1 - i0));
  // (Past the belt's edge lies the floodplain.)
  const veg = outB || ihash(i0, bj, key + 9) < pVeg;
  // Finer channels across the bare bars of a river with many threads.
  // (Faded by thinning, not by narrowing: faint hairlines would hatch the bars.)
  const fineW = veg ? 0 : smooth01((nT - 5) / 3) * smooth01((br - 0.5) / 0.4) * smooth01((h - 15) / 20);
  const fine = fineW < 0.35 ? 0 : smooth01((fineW - 0.35) / 0.4);
  if (fine > 0) {
    const X2 = X * 2.3 + 31.7;
    const Y2 = Y * 2.3 + 17.3;
    const b2 = nearestPoint(1, X2, Y2, key + 17, 0.85, 0.7);
    const pB2 = Math.round(20 * (0.3 + 0.3 * smooth01((nT - 6) / 5))) / 20;
    const sx2 = sx / 2.3;
    const sy2 = sy / 2.3;
    const i2 = NP_I[b2];
    const j2 = NP_J[b2];
    const s2 = 0.5 + 0.5 * Math.sin(1.9 * wx + 2.3 * wy + 0.25 * X2 - 1.1);
    const half2 = Math.max(0.5, Math.min(0.2 * sy2, Math.max(0.3, (0.95 * chan) / Math.sqrt(Math.max(1, nT)))) * (0.15 + 0.35 * s2 * s2));
    const r2 = 0.3 * half2 + 0.5;
    let sum2 = 0;
    for (let k = 9; k < 18; k++) {
      if (k === b2) continue;
      if (NP_J[k] === j2 && sameRun(i2, NP_I[k], j2, key + 17, pB2)) continue;
      const ex = NP_X[k] - NP_X[b2];
      const ey = NP_Y[k] - NP_Y[b2];
      const el = Math.sqrt(ex * ex + ey * ey);
      const nx = ex / el;
      const ny = ey / el;
      const delta = ((NP_X[k] + NP_X[b2]) * 0.5 - X2) * nx + ((NP_Y[k] + NP_Y[b2]) * 0.5 - Y2) * ny;
      const d2 = delta / Math.sqrt((nx / sx2) * (nx / sx2) + (ny / sy2) * (ny / sy2));
      if (d2 < half2 + 5 * r2) sum2 += Math.exp(-Math.max(0, d2) / r2);
    }
    if (sum2 > 0) water = Math.max(water, Math.min(1, half2 + r2 * Math.log(sum2) + 0.5) * fine);
  }
  netWater = Math.max(0, Math.min(1, water));
  // Islands keep a sandy fringe along their channels, and so does the bank
  // beyond the belt's outermost channels, where the floodplain begins.
  const hn = edgeOut ? 0.6 * half : half;
  const fringe = 1 - smooth01((dEdge - 1.3 * hn - 0.6) / (0.7 + 0.25 * hn));
  netBare = veg && !outB ? fringe : 1;
  netBelt = outB ? fringe : 1;
}

function stampQuad(noise: Simplex3, o0: number, o1: number, key: number, Wg: number, Hg: number): void {
  const a0x = BV[o0];
  const a0y = BV[o0 + 1];
  const dAx = BV[o1] - a0x;
  const dAy = BV[o1 + 1] - a0y;
  const H0 = BV[o0 + 5];
  const H1 = BV[o1 + 5];
  const m0x = BV[o0 + 2] * H0;
  const m0y = BV[o0 + 3] * H0;
  const dMx = BV[o1 + 2] * H1 - m0x;
  const dMy = BV[o1 + 3] * H1 - m0y;
  const minX = Math.max(0, Math.floor(Math.min(a0x + m0x, a0x - m0x, a0x + dAx + m0x + dMx, a0x + dAx - m0x - dMx) - 1));
  const maxX = Math.min(Wg - 1, Math.ceil(Math.max(a0x + m0x, a0x - m0x, a0x + dAx + m0x + dMx, a0x + dAx - m0x - dMx) + 1));
  const minY = Math.max(0, Math.floor(Math.min(a0y + m0y, a0y - m0y, a0y + dAy + m0y + dMy, a0y + dAy - m0y - dMy) - 1));
  const maxY = Math.min(Hg - 1, Math.ceil(Math.max(a0y + m0y, a0y - m0y, a0y + dAy + m0y + dMy, a0y + dAy - m0y - dMy) + 1));
  if (minX > maxX || minY > maxY) return;
  const qa = -(dAx * dMy - dAy * dMx);
  const cAM = dAx * m0y - dAy * m0x;
  // The map's orientation mid-quad: where it flips, the quad has folded over (skip).
  const jRef = dAx * (m0y + 0.5 * dMy) - dAy * (m0x + 0.5 * dMx);
  // Pixels per unit τ along this quad.
  const pa = Math.hypot(dAx, dAy) / Math.max(1e-6, Math.abs(BV[o1 + 6] - BV[o0 + 6]));
  for (let py = minY; py <= maxY; py++) {
    for (let px = minX; px <= maxX; px++) {
      // Invert the quad's map P = A0 + t·dA + u·(M0 + t·dM) for (t, u).
      const Qx = px - a0x;
      const Qy = py - a0y;
      const qb = Qx * dMy - Qy * dMx - cAM;
      const qc = Qx * m0y - Qy * m0x;
      let t: number;
      if (Math.abs(qa) < 1e-9) {
        if (Math.abs(qb) < 1e-12) continue;
        t = -qc / qb;
      } else {
        const disc = qb * qb - 4 * qa * qc;
        if (disc < 0) continue;
        const sq = Math.sqrt(disc);
        const r1 = (-qb + sq) / (2 * qa);
        const r2 = (-qb - sq) / (2 * qa);
        t = r1 >= -0.03 && r1 <= 1.03 ? r1 : r2;
      }
      if (t < -0.03 || t > 1.03) continue;
      const mx = m0x + t * dMx;
      const my = m0y + t * dMy;
      const m2 = mx * mx + my * my;
      if (m2 <= 1e-12) continue;
      const u = ((Qx - t * dAx) * mx + (Qy - t * dAy) * my) / m2;
      if (u < -1 || u > 1) continue;
      if (((dAx + u * dMx) * my - (dAy + u * dMy) * mx) * jRef <= 0) continue;
      const nPx = u * Math.sqrt(m2);
      const p = py * Wg + px;
      const h = BV[o0 + 4] + (BV[o1 + 4] - BV[o0 + 4]) * t;
      const edge = nPx >= 0 ? BV[o0 + 9] + (BV[o1 + 9] - BV[o0 + 9]) * t : BV[o0 + 10] + (BV[o1 + 10] - BV[o0 + 10]) * t;
      const d = Math.abs(nPx);
      // The belt's ground, and how far out its channels reach.
      const cov = Math.max(0, Math.min(1, (1 + edge) * h - d + 0.5));
      const reachW = Math.min(1, (1.3 + edge) * h + 1.5 - d);
      if (reachW <= 0) continue;
      const tau = BV[o0 + 6] + (BV[o1 + 6] - BV[o0 + 6]) * t;
      const bars = BV[o0 + 7] + (BV[o1 + 7] - BV[o0 + 7]) * t;
      const br = BV[o0 + 11] + (BV[o1 + 11] - BV[o0 + 11]) * t;
      const an = BV[o0 + 12] + (BV[o1 + 12] - BV[o0 + 12]) * t;
      // The network shows once the belt is a few pixels wide.
      const show = smooth01((h - 2) / 3) * Math.min(1, (br + an) * 4);
      netWater = 0;
      netBare = 1;
      netBelt = 1;
      if (show > 0) {
        const nT = BV[o0 + 13] + (BV[o1 + 13] - BV[o0 + 13]) * t;
        const chan = BV[o0 + 14] + (BV[o1 + 14] - BV[o0 + 14]) * t;
        braidNet(noise, tau, nPx, h, pa, br, an, nT, chan, bars, key);
        const wv = Math.min(netWater * show, reachW) * (BV[o0 + 15] + (BV[o1 + 15] - BV[o0 + 15]) * t);
        if (wv > COVER[p]) {
          COVER[p] = wv;
          PAT[p] = an > br ? 7 : br > 0.6 ? 6 : 5;
        }
      }
      const bc = cov * (1 - (1 - netBelt) * show);
      if (bc <= BELT[p]) continue;
      BELT[p] = bc;
      // Bare bars or islands: the network's where it shows, else the belt's overall share.
      BARE[p] = netBare * show + smooth01((bars - 0.25) / 0.5) * (1 - show);
      GRAV[p] = BV[o0 + 8] + (BV[o1 + 8] - BV[o0 + 8]) * t;
    }
  }
}

/**
 * Inside each meander bend: the point bar the migrating channel leaves on its
 * inner bank (bare sand or gravel, into BELT/BARE/GRAV) and the scroll bars
 * beyond it, ridges and swales parallel to the channel (SCRL: 1 + distance
 * from the inner bank in channel widths). hwAt(s): the channel's half-width
 * at curve point s (px).
 */
export function stampBends(st: ChannelStyle, n: number, hwAt: (s: number) => number, Wg: number, Hg: number): void {
  const mp = CURVE_MP;
  let s = 1;
  while (s < n) {
    const acr = CURVE_ACR[s];
    const hw = hwAt(s);
    const step = Math.max(1, Math.round(Math.min(2 * hw, Math.abs(acr)) / 2));
    const s1 = Math.min(n - 1, s + step - 1);
    const sPrev = s - 1;
    const vs = CURVE_V[s];
    const me = lerpV(st.meander0, st.meander1, vs);
    if (me > 0.05 && Math.abs(acr) > 1.2 * hw && hw > 1.1) {
      axisNormal(s, n);
      const sg = acr > 0 ? -1 : 1;
      const reach = Math.min(Math.abs(acr), 6 * hw);
      // Deeper loops build bigger bars.
      const pb = me * lerpV(st.bars0, st.bars1, vs) * smooth01((Math.abs(acr) / (2 * hw) - 0.6) / 0.9);
      stampInner(mp[2 * sPrev], mp[2 * sPrev + 1], mp[2 * s1], mp[2 * s1 + 1], sg * NRM[0], sg * NRM[1], hw, reach, pb, lerpV(st.gravel0, st.gravel1, vs), Wg, Hg);
    }
    s = s1 + 1;
  }
}

function stampInner(x0: number, y0: number, x1: number, y1: number, inx: number, iny: number, hw: number, reach: number, pb: number, gv: number, Wg: number, Hg: number): void {
  const r = reach + 1;
  const minX = Math.max(0, Math.floor(Math.min(x0, x1) - r));
  const maxX = Math.min(Wg - 1, Math.ceil(Math.max(x0, x1) + r));
  const minY = Math.max(0, Math.floor(Math.min(y0, y1) - r));
  const maxY = Math.min(Hg - 1, Math.ceil(Math.max(y0, y1) + r));
  if (minX > maxX || minY > maxY) return;
  const dx = x1 - x0;
  const dy = y1 - y0;
  const l2 = dx * dx + dy * dy || 1e-9;
  const iw = 1 / (2 * hw);
  for (let py = minY; py <= maxY; py++) {
    for (let px = minX; px <= maxX; px++) {
      // Only beside the stretch itself (a round end would ring the bend's scrolls).
      const t = ((px - x0) * dx + (py - y0) * dy) / l2;
      if (t < -0.02 || t > 1.02) continue;
      const ex = px - (x0 + t * dx);
      const ey = py - (y0 + t * dy);
      if (ex * inx + ey * iny <= 0) continue;
      const d = Math.sqrt(ex * ex + ey * ey);
      if (d >= reach) continue;
      const u = Math.max(0, (d - hw) * iw);
      const p = py * Wg + px;
      const sv = 1 + u;
      if (SCRL[p] === 0 || sv < SCRL[p]) SCRL[p] = sv;
      if (pb > 0) {
        const bar = pb * smooth01((0.65 - u) / 0.3);
        if (bar > BELT[p]) {
          BELT[p] = bar;
          BARE[p] = 1;
          GRAV[p] = gv;
        }
      }
    }
  }
}

/**
 * Loops a tightly winding river cut off: oxbow lakes (OXB) just beyond its
 * present bends, and the silted-up scars of older ones (SCAR) further out,
 * only on its floodplain. Each meander half-wave decides by its global index
 * (seedCell: the chain's), so they stay put across tiles and levels.
 */
export function stampOxbows(st: ChannelStyle, n: number, hwAt: (s: number) => number, seedCell: number, Wg: number, Hg: number): void {
  if (apexN <= 0) return;
  const ax = CURVE_AX;
  for (let pass = 0; pass < 2; pass++) {
    const buf = pass === 0 ? OXB : SCAR;
    let prevX = 0;
    let prevY = 0;
    let prevW = 0;
    let prevJ = -2;
    for (let s = 0; s < n; s++) {
      const j = CURVE_HWJ[s];
      const D = j >= 0 && j < apexN ? CURVE_APEX_D[j] : 0;
      const acr = CURVE_ACR[s];
      const vs = CURVE_V[s];
      const me = lerpV(st.meander0, st.meander1, vs);
      const sinu = lerpV(st.sinu0, st.sinu1, vs);
      const hw = hwAt(s);
      const pOx = pass === 0 ? 0.32 * smooth01((sinu - 1.3) / 0.5) : 0.5 * smooth01((sinu - 1.2) / 0.5);
      const single = 1 - Math.min(1, lerpV(st.braid0, st.braid1, vs) + lerpV(st.anas0, st.anas1, vs));
      const on = j >= 0 && me > 0.3 && single > 0.8 && Math.abs(D) > 2 * hw && hash01(seedCell, apexW0 + j, pass === 0 ? 951 : 952) < pOx * me;
      if (!on) {
        prevJ = -2;
        continue;
      }
      const share = Math.abs(acr) / Math.abs(D);
      const lo = pass === 0 ? 0.5 : 0.35;
      const w = (pass === 0 ? 0.85 : 1.3) * hw * smooth01((share - lo) / 0.2);
      axisNormal(s, n);
      const out = pass === 0 ? acr * 1.45 + Math.sign(acr) * 3.5 * hw : acr * 1.85 + Math.sign(acr) * 5.5 * hw;
      const x = ax[2 * s] + NRM[0] * out;
      const y = ax[2 * s + 1] + NRM[1] * out;
      if (prevJ === j && (w > 0 || prevW > 0)) stampOx(prevX, prevY, x, y, prevW, w, buf, Wg, Hg);
      prevX = x;
      prevY = y;
      prevW = w;
      prevJ = j;
    }
  }
}

function stampOx(x0: number, y0: number, x1: number, y1: number, h0: number, h1: number, buf: Float32Array, Wg: number, Hg: number): void {
  const r = Math.max(h0, h1) + 1;
  const minX = Math.max(0, Math.floor(Math.min(x0, x1) - r));
  const maxX = Math.min(Wg - 1, Math.ceil(Math.max(x0, x1) + r));
  const minY = Math.max(0, Math.floor(Math.min(y0, y1) - r));
  const maxY = Math.min(Hg - 1, Math.ceil(Math.max(y0, y1) + r));
  if (minX > maxX || minY > maxY) return;
  const dx = x1 - x0;
  const dy = y1 - y0;
  const l2 = dx * dx + dy * dy || 1e-9;
  for (let py = minY; py <= maxY; py++) {
    for (let px = minX; px <= maxX; px++) {
      const t = Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / l2));
      const d = Math.hypot(px - (x0 + t * dx), py - (y0 + t * dy));
      const a = h0 + (h1 - h0) * t + 0.5 - d;
      if (a <= 0) continue;
      const p = py * Wg + px;
      // Only on the floodplain.
      const val = Math.min(1, a) * smooth01(FLOOD[p] / 0.5);
      if (val > buf[p]) buf[p] = val;
    }
  }
}

const ARROW_P = new Float64Array(8);

/**
 * A flow arrow centered at (cx, cy) pointing along the unit vector (ux, uy):
 * an arrowhead `len` px long and `wid` px across with a notched back and a
 * dark outline, into ARW (fill) and ARO (outline) coverage.
 */
export function stampArrow(cx: number, cy: number, ux: number, uy: number, len: number, wid: number, alpha: number, Wg: number, Hg: number): void {
  const nx = -uy;
  const ny = ux;
  const P = ARROW_P;
  // Tip, one back corner, the notch, the other back corner.
  P[0] = cx + ux * len * 0.5;
  P[1] = cy + uy * len * 0.5;
  P[2] = cx - ux * len * 0.5 + nx * wid * 0.5;
  P[3] = cy - uy * len * 0.5 + ny * wid * 0.5;
  P[4] = cx - ux * len * 0.2;
  P[5] = cy - uy * len * 0.2;
  P[6] = cx - ux * len * 0.5 - nx * wid * 0.5;
  P[7] = cy - uy * len * 0.5 - ny * wid * 0.5;
  const ol = 1.1;
  const minX = Math.max(0, Math.floor(Math.min(P[0], P[2], P[6]) - ol - 1));
  const maxX = Math.min(Wg - 1, Math.ceil(Math.max(P[0], P[2], P[6]) + ol + 1));
  const minY = Math.max(0, Math.floor(Math.min(P[1], P[3], P[7]) - ol - 1));
  const maxY = Math.min(Hg - 1, Math.ceil(Math.max(P[1], P[3], P[7]) + ol + 1));
  for (let py = minY; py <= maxY; py++) {
    for (let px = minX; px <= maxX; px++) {
      // Signed distance (+ inside) to the arrowhead: the union of its two halves.
      const d = Math.max(triDist(px, py, P[0], P[1], P[2], P[3], P[4], P[5]), triDist(px, py, P[0], P[1], P[4], P[5], P[6], P[7]));
      if (d < -ol - 0.5) continue;
      const p = py * Wg + px;
      const fill = Math.max(0, Math.min(1, d + 0.5)) * alpha;
      const edge = Math.max(0, Math.min(1, d + ol + 0.5)) * alpha;
      if (fill > ARW[p]) ARW[p] = fill;
      if (edge > ARO[p]) ARO[p] = edge;
    }
  }
}

/** Signed distance (px, + inside) from (x, y) to the triangle a, b, c (the least of its edges'). */
function triDist(x: number, y: number, ax: number, ay: number, bx: number, by: number, cx: number, cy: number): number {
  const o = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) >= 0 ? 1 : -1;
  return Math.min(edgeDist(x, y, ax, ay, bx, by, o), edgeDist(x, y, bx, by, cx, cy, o), edgeDist(x, y, cx, cy, ax, ay, o));
}

function edgeDist(x: number, y: number, x0: number, y0: number, x1: number, y1: number, o: number): number {
  const ex = x1 - x0;
  const ey = y1 - y0;
  const l = Math.hypot(ex, ey) || 1e-9;
  return (o * (ex * (y - y0) - ey * (x - x0))) / l;
}

/** Pixels a stream's valley touched in STREAM (inclusive box; empty when x0 > x1). */
let streamX0 = 0;
let streamY0 = 0;
let streamX1 = -1;
let streamY1 = -1;

/** One stretch of a stream's valley profile into STREAM (the stream's own stretches overlap: max). */
function stampCarve(x0: number, y0: number, x1: number, y1: number, V: number, wet: number, Wg: number, Hg: number): void {
  const minX = Math.max(0, Math.floor(Math.min(x0, x1) - V));
  const maxX = Math.min(Wg - 1, Math.ceil(Math.max(x0, x1) + V));
  const minY = Math.max(0, Math.floor(Math.min(y0, y1) - V));
  const maxY = Math.min(Hg - 1, Math.ceil(Math.max(y0, y1) + V));
  if (minX > maxX || minY > maxY) return;
  if (streamX0 > streamX1) {
    streamX0 = minX;
    streamX1 = maxX;
    streamY0 = minY;
    streamY1 = maxY;
  } else {
    streamX0 = Math.min(streamX0, minX);
    streamX1 = Math.max(streamX1, maxX);
    streamY0 = Math.min(streamY0, minY);
    streamY1 = Math.max(streamY1, maxY);
  }
  const dx = x1 - x0;
  const dy = y1 - y0;
  const l2 = dx * dx + dy * dy || 1e-9;
  const iv = 1 / V;
  const V2 = V * V;
  for (let py = minY; py <= maxY; py++) {
    for (let px = minX; px <= maxX; px++) {
      const t = Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / l2));
      const ex = px - (x0 + t * dx);
      const ey = py - (y0 + t * dy);
      const d2 = ex * ex + ey * ey;
      if (d2 >= V2) continue;
      const d = Math.sqrt(d2) * iv;
      const c = (1 - d) * Math.sqrt(1 - d);
      const p = py * Wg + px;
      if (c > STREAM[p]) STREAM[p] = c;
      const g = wet * (d < 0.55 ? 1 : (1 - d) / 0.45);
      if (g > RIPARIAN[p]) RIPARIAN[p] = g;
    }
  }
}

/**
 * Adds a stream's valley (STREAM) to the others (CARVE) as a smooth union,
 * 1 − (1 − a)(1 − b): where valleys meet they deepen into one instead of
 * leaving a ridge between them (as the max of their profiles would), so
 * confluent valleys and glacial troughs join rather than fanning out.
 */
function mergeStream(Wg: number): void {
  for (let py = streamY0; py <= streamY1; py++) {
    for (let p = py * Wg + streamX0, e = py * Wg + streamX1; p <= e; p++) {
      const c = STREAM[p];
      if (c <= 0) continue;
      CARVE[p] = 1 - (1 - CARVE[p]) * (1 - c);
      STREAM[p] = 0;
    }
  }
  streamX0 = 0;
  streamX1 = -1;
}

// ------------------------------------------------------------------ deltas, fans and inland deltas

/** Networks whose plain or cone reaches the tile being drawn, how many, and the cosine of each one's reach from its apex. */
let tileNets = new Int32Array(16);
let tileNetN = 0;
let tileCos = new Float64Array(16);
/** Node positions of the network being drawn (px), and which nodes lie on its main channel. */
let NODE_PX = new Float64Array(2048);
let ON_MAIN = new Uint8Array(1024);

/** How much distributaries wind, by form (NET_FORMS order): straight bird's-foot passes, sinuous lobes, inland deltas winding most. */
const NET_BENDS = [0.25, 0.8, 0.5, 0.5, 0.4, 0.35, 0.9];

/** Scratch style for distributaries. */
const NET_STYLE = newChannelStyle();

/**
 * Lists the networks reaching the tile (for the pixel loop's plains and
 * cones) and draws their distributaries: from the apex (where the river's
 * own course stops) to every mouth, a fan's braided washes, and on the
 * drainage layer flow arrows. Around a sea delta's channels, levee land
 * (LEVEE) keeps every channel between banks out to its mouth.
 */
function rasterNetworks(md: MeshData, f: FieldData, L: number, x0g: number, y0g: number, Wg: number, Hg: number, dRad: number, draw: boolean, lcode: number): void {
  tileNetN = 0;
  const nNet = f.netStart.length - 1;
  if (nNet <= 0) return;
  const G = Wg * Hg;
  LEVEE.fill(0, 0, G);
  const margin = 24 * dRad;
  const lonMin = -Math.PI + x0g * dRad - margin;
  const lonMax = -Math.PI + (x0g + Wg) * dRad + margin;
  const latMax = Math.PI / 2 - y0g * dRad + margin;
  const latMin = Math.PI / 2 - (y0g + Hg) * dRad - margin;
  const { netBox, netStart, netX, netY, netZ, netParent, netQ, netW, netMouth, netDist, netInfo } = f;
  const R = f.radiusKm * 1000;
  const mPerPx = R * dRad;
  const thrL = riverThresholdFor(f, md.spacing / dRad);
  const thr0 = f.riverThreshold;
  const arrows = draw && lcode === L_DRAIN;
  const st = NET_STYLE;
  for (let n = 0; n < nNet; n++) {
    const b0 = netBox[4 * n];
    const b1 = netBox[4 * n + 1];
    if (b1 < latMin || b0 > latMax) continue;
    let shift = NaN;
    for (const s of [0, -2 * Math.PI, 2 * Math.PI]) {
      if (netBox[4 * n + 2] + s <= lonMax && netBox[4 * n + 3] + s >= lonMin) {
        shift = s;
        break;
      }
    }
    if (Number.isNaN(shift)) continue;
    if (tileNetN >= tileNets.length) {
      const nt = new Int32Array(2 * tileNets.length);
      nt.set(tileNets);
      tileNets = nt;
      const nc = new Float64Array(2 * tileCos.length);
      nc.set(tileCos);
      tileCos = nc;
    }
    let reachR = 0;
    for (let b = 0; b < NET_BINS; b++) reachR = Math.max(reachR, f.netFront[NET_BINS * n + b]);
    // (Gnomonic distance r is tan of the angle; a little beyond for ragged shores.)
    tileCos[tileNetN] = Math.cos(Math.min(1.5, Math.atan(reachR) + 2000 / R));
    tileNets[tileNetN++] = n;
    const s0 = netStart[n];
    const s1 = netStart[n + 1];
    if (s1 - s0 < 2) continue;
    const io = NET_INFO * n;
    const kind = netInfo[io];
    const months = netInfo[io + 6];
    if (NODE_PX.length < 2 * (s1 - s0)) NODE_PX = new Float64Array(4 * (s1 - s0));
    const lonRef = Math.atan2(netY[s0], netX[s0]) + shift;
    for (let g = s0; g < s1; g++) {
      const lat = Math.asin(Math.max(-1, Math.min(1, netZ[g])));
      let lon = Math.atan2(netY[g], netX[g]);
      lon += 2 * Math.PI * Math.round((lonRef - lon) / (2 * Math.PI));
      NODE_PX[2 * (g - s0)] = (lon + Math.PI) / dRad - 0.5 - x0g;
      NODE_PX[2 * (g - s0) + 1] = (Math.PI / 2 - lat) / dRad - 0.5 - y0g;
    }
    // Opacity and dashes by how many months the river runs.
    const alpha = months >= 11 ? 1 : months >= 4 ? 0.85 : 0.6;
    const half = (g: number): number => {
      const carto = (0.25 + 0.55 * Math.log10(netQ[g] / thr0 + 1)) * (1 + 0.08 * (L - BASE_LEVEL));
      return Math.max(0.35, carto, (0.5 * netW[g]) / mPerPx);
    };
    // A fan's channels are braided washes over bare gravel or sand; a delta's
    // and an inland delta's single channels between levees.
    const fan = kind === 1;
    st.seed = 100 * hash01(s0, n, 947);
    st.alpha = alpha * (months >= 4 ? 1 : 0.7);
    st.pat0 = st.pat1 = Channel.Distributary + 1;
    st.braid0 = st.braid1 = fan ? 1 : 0;
    st.anas0 = st.anas1 = 0;
    st.threads0 = st.threads1 = 3;
    st.bars0 = st.bars1 = 0.25 + 0.75 * netInfo[io + 7];
    st.gravel0 = st.gravel1 = 1 - smooth01((netInfo[io + 4] - 0.3) / 0.5);
    st.meander0 = st.meander1 = 0;
    st.sinu0 = st.sinu1 = 1;
    // The main channel (the most water at every split) is drawn whenever the
    // river reaching the apex is, so the river always runs on to the sea or
    // its exit however its other channels thin out.
    if (ON_MAIN.length < s1 - s0) ON_MAIN = new Uint8Array(2 * (s1 - s0));
    ON_MAIN.fill(0, 0, s1 - s0);
    if (netQ[s0] >= thrL) for (let g = s0, guard = 0; g >= 0 && guard < s1 - s0; g = f.netNext[g], guard++) ON_MAIN[g - s0] = 1;
    let lastG = -10;
    let seg = 0;
    for (let g = s0 + 1; g < s1; g++) {
      const pg = netParent[g];
      if (pg < 0 || (netQ[g] < thrL && ON_MAIN[g - s0] === 0)) continue;
      const ax = NODE_PX[2 * (pg - s0)];
      const ay = NODE_PX[2 * (pg - s0) + 1];
      const bx = NODE_PX[2 * (g - s0)];
      const by = NODE_PX[2 * (g - s0) + 1];
      const hA = half(pg);
      const hB = half(g);
      const reach = Math.max(hA, hB) * (fan ? 4 : 3) + 4;
      if (Math.max(ax, bx) < -reach || Math.min(ax, bx) > Wg + reach || Math.max(ay, by) < -reach || Math.min(ay, by) > Hg + reach) continue;
      const nS = netCurve(f, n, g, dRad, x0g, y0g, shift);
      if (nS < 2) continue;
      const mp = CURVE_MP;
      if (fan) {
        // The wash: a braided belt a few channels wide.
        st.belt0 = 2.6 * hA;
        st.belt1 = 2.6 * hB;
        st.chan0 = 2 * hA;
        st.chan1 = 2 * hB;
        const bw = 2.6 * Math.max(1, netW[g]);
        st.tau0 = netDist[pg] / bw;
        st.tau1 = netDist[g] / bw;
        st.plain0 = st.plain1 = 0;
        seg = pg === lastG ? seg + 1 : seg + 2;
        if (draw && Math.max(st.belt0, st.belt1) >= 1.5) beltAppend(st, nS, VALLEY_MAX_PX, seg, netMouth[g] === 1, false);
        lastG = g;
      }
      // Levee land along a sea delta's channels: out past the front only on
      // river-built deltas (a bird's foot's fingers, lobes); waves rework a
      // mouth's levees and tides keep its funnel open.
      if (kind === 0 && (netInfo[io + 1] <= 1 || netMouth[g] === 0)) {
        const lv0 = 2.5 * hA + 1.5;
        const lv1 = 2.5 * hB + 1.5;
        for (let s = 1; s < nS; s++) {
          const u0 = (s - 1) / (nS - 1);
          const u1 = s / (nS - 1);
          stampLevee(mp[2 * s - 2], mp[2 * s - 1], mp[2 * s], mp[2 * s + 1], lv0 + (lv1 - lv0) * u0, lv0 + (lv1 - lv0) * u1, Wg, Hg);
        }
      }
      if (!draw) continue;
      // The channel (a wash shows as threads over its bed: drawn thinner).
      const kc = fan ? 0.55 : 1;
      const dash = months >= 11 || L < 4 || hB >= 1.5 ? 0 : months >= 4 ? 1 : 2;
      const on = dash === 1 ? 7 : 2.2;
      const period = dash === 1 ? 11 : 6;
      let arc = 0;
      let arcTot = 0;
      for (let s = 1; s < nS; s++) arcTot += Math.hypot(mp[2 * s] - mp[2 * s - 2], mp[2 * s + 1] - mp[2 * s - 1]);
      const gA = netDist[pg] / mPerPx;
      const gB = netDist[g] / mPerPx;
      const phase = ARROW_GAP * hash01(s0, 0, 948);
      for (let s = 1; s < nS; s++) {
        const u0 = (s - 1) / (nS - 1);
        const u1 = s / (nS - 1);
        const segL = Math.hypot(mp[2 * s] - mp[2 * s - 2], mp[2 * s + 1] - mp[2 * s - 1]);
        const w0 = (hA + (hB - hA) * u0) * kc;
        const w1 = (hA + (hB - hA) * u1) * kc;
        if (dash === 0 || (arc + segL / 2) % period < on) stampLine(mp[2 * s - 2], mp[2 * s - 1], mp[2 * s], mp[2 * s + 1], w0, w1, alpha, Channel.Distributary + 1, Wg, Hg);
        if (arrows && netQ[g] >= thrL * 6 && arcTot > 0) {
          const g0 = gA + ((gB - gA) * arc) / arcTot + phase;
          const g1 = gA + ((gB - gA) * (arc + segL)) / arcTot + phase;
          const m1 = Math.floor(g1 / ARROW_GAP);
          if (m1 > Math.floor(g0 / ARROW_GAP) && g1 > g0 && netMouth[g] === 0) {
            const fr = (m1 * ARROW_GAP - g0) / (g1 - g0);
            const cx = mp[2 * s - 2] + (mp[2 * s] - mp[2 * s - 2]) * fr;
            const cy = mp[2 * s - 1] + (mp[2 * s + 1] - mp[2 * s - 1]) * fr;
            const ka = Math.max(0, s - 3);
            const kb = Math.min(nS - 1, s + 2);
            const ux = mp[2 * kb] - mp[2 * ka];
            const uy = mp[2 * kb + 1] - mp[2 * ka + 1];
            const ul = Math.hypot(ux, uy);
            if (ul > 1e-6) {
              const len = Math.max(7, Math.min(14, 6 + 1.6 * w1));
              stampArrow(cx, cy, ux / ul, uy / ul, len, 0.9 * len, months >= 11 ? 1 : 0.55, Wg, Hg);
            }
          }
        }
        arc += segL;
      }
    }
    if (fan) beltFlush();
  }
}

/** Levee land along a delta's channel: a band of half-widths h0 → h1 (px) into LEVEE. */
function stampLevee(x0: number, y0: number, x1: number, y1: number, h0: number, h1: number, Wg: number, Hg: number): void {
  const r = Math.max(h0, h1) + 1;
  const minX = Math.max(0, Math.floor(Math.min(x0, x1) - r));
  const maxX = Math.min(Wg - 1, Math.ceil(Math.max(x0, x1) + r));
  const minY = Math.max(0, Math.floor(Math.min(y0, y1) - r));
  const maxY = Math.min(Hg - 1, Math.ceil(Math.max(y0, y1) + r));
  if (minX > maxX || minY > maxY) return;
  const dx = x1 - x0;
  const dy = y1 - y0;
  const l2 = dx * dx + dy * dy || 1e-9;
  for (let py = minY; py <= maxY; py++) {
    for (let px = minX; px <= maxX; px++) {
      const t = Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / l2));
      const h = h0 + (h1 - h0) * t;
      const d = Math.hypot(px - (x0 + t * dx), py - (y0 + t * dy));
      const v = Math.min(1, h - d + 0.5);
      if (v <= 0) continue;
      const p = py * Wg + px;
      if (v > LEVEE[p]) LEVEE[p] = v;
    }
  }
}

/**
 * The drawn course of network n's channel ending at node g (global index),
 * from its parent: a centripetal Catmull–Rom curve through the nodes (past
 * the apex and the exit, the river's own nodes shape it), in pixels from
 * (x0g, y0g). Fills CURVE_MP, CURVE_AX, CURVE_BAX and CURVE_V like
 * riverCurve; returns the number of points.
 */
function netCurve(f: FieldData, n: number, g: number, dRad: number, x0g: number, y0g: number, lonShift: number): number {
  const { netX, netY, netZ, netParent, netNext, netUp, netDown, netExit } = f;
  const P = CURVE_P;
  const pg = netParent[g];
  const pp = netParent[pg];
  const nn = netNext[g];
  P[3] = netX[pg];
  P[4] = netY[pg];
  P[5] = netZ[pg];
  P[6] = netX[g];
  P[7] = netY[g];
  P[8] = netZ[g];
  for (let k = 0; k < 3; k++) {
    const up = netUp[3 * n + k];
    P[k] = pp >= 0 ? (k === 0 ? netX[pp] : k === 1 ? netY[pp] : netZ[pp]) : Number.isNaN(up) ? 2 * P[3 + k] - P[6 + k] : up;
    const dn = netDown[3 * n + k];
    P[9 + k] = nn >= 0 ? (k === 0 ? netX[nn] : k === 1 ? netY[nn] : netZ[nn]) : g === netExit[n] && !Number.isNaN(dn) ? dn : 2 * P[6 + k] - P[3 + k];
  }
  const lonRef = Math.atan2(P[4], P[3]) + lonShift;
  const pts = CURVE_MP;
  const toPx = (vx: number, vy: number, vz: number, o: number): void => {
    const l = Math.hypot(vx, vy, vz);
    const lat = Math.asin(Math.max(-1, Math.min(1, vz / l)));
    let lon = Math.atan2(vy, vx);
    lon += 2 * Math.PI * Math.round((lonRef - lon) / (2 * Math.PI));
    pts[o] = (lon + Math.PI) / dRad - 0.5 - x0g;
    pts[o + 1] = (Math.PI / 2 - lat) / dRad - 0.5 - y0g;
  };
  const k01 = Math.max(1e-9, Math.sqrt(Math.hypot(P[3] - P[0], P[4] - P[1], P[5] - P[2])));
  const k12 = Math.max(1e-9, Math.sqrt(Math.hypot(P[6] - P[3], P[7] - P[4], P[8] - P[5])));
  const k23 = Math.max(1e-9, Math.sqrt(Math.hypot(P[9] - P[6], P[10] - P[7], P[11] - P[8])));
  const t1 = k01;
  const t2 = t1 + k12;
  const t3 = t2 + k23;
  toPx(P[3], P[4], P[5], 0);
  const sx = pts[0];
  const sy = pts[1];
  toPx(P[6], P[7], P[8], 0);
  const lenPx = Math.hypot(pts[0] - sx, pts[1] - sy);
  const nS = Math.max(2, Math.min(CURVE_MAX, Math.ceil(lenPx / 2.5) + 1));
  for (let s = 0; s < nS; s++) {
    const t = t1 + (s / (nS - 1)) * k12;
    let vx = 0;
    let vy = 0;
    let vz = 0;
    for (let k = 0; k < 3; k++) {
      const p0 = P[k];
      const p1 = P[3 + k];
      const p2 = P[6 + k];
      const p3 = P[9 + k];
      const A1 = ((t1 - t) / t1) * p0 + (t / t1) * p1;
      const A2 = ((t2 - t) / k12) * p1 + ((t - t1) / k12) * p2;
      const A3 = ((t3 - t) / k23) * p2 + ((t - t2) / k23) * p3;
      const B1 = ((t2 - t) / t2) * A1 + (t / t2) * A2;
      const B2 = ((t3 - t) / (t3 - t1)) * A2 + ((t - t1) / (t3 - t1)) * A3;
      const C = ((t2 - t) / k12) * B1 + ((t - t1) / k12) * B2;
      if (k === 0) vx = C;
      else if (k === 1) vy = C;
      else vz = C;
    }
    toPx(vx, vy, vz, 2 * s);
  }
  // Gentle bends of its own, once they would show (between nodes a channel
  // would otherwise run as a ruled line): by distance down the channels, so
  // branches leave a split alike and tiles agree; none at the apex or exit,
  // where the river's own course joins.
  const io = NET_INFO * n;
  const mPerPx = f.radiusKm * 1000 * dRad;
  const wA = f.netW[pg];
  const wB = f.netW[g];
  const kS = NET_BENDS[f.netInfo[io + 1]] ?? 0.5;
  if (kS * Math.max(wA, wB) > 0.08 * mPerPx) {
    const dA = f.netDist[pg];
    const dB = f.netDist[g];
    const ph = 2 * Math.PI * hash01(f.netStart[n], 0, 949);
    const toExit = g === netExit[n];
    BAX_SRC.set(pts.subarray(0, 2 * nS));
    for (let s = 0; s < nS; s++) {
      const u = s / (nS - 1);
      const W = wA + (wB - wA) * u;
      const lam = 12 * W;
      const d = dA + (dB - dA) * u;
      const show = smooth01((lam / mPerPx - 6) / 6) * smooth01(d / lam) * (toExit ? smooth01(((1 - u) * (dB - dA)) / lam) : 1);
      const off = (kS * W * show * Math.sin((2 * Math.PI * d) / lam + ph)) / mPerPx;
      if (off === 0) continue;
      const a = Math.max(0, s - 1);
      const b = Math.min(nS - 1, s + 1);
      const tx = BAX_SRC[2 * b] - BAX_SRC[2 * a];
      const ty = BAX_SRC[2 * b + 1] - BAX_SRC[2 * a + 1];
      const tl = Math.hypot(tx, ty) || 1;
      pts[2 * s] -= (ty / tl) * off;
      pts[2 * s + 1] += (tx / tl) * off;
    }
  }
  CURVE_AX.set(pts.subarray(0, 2 * nS));
  CURVE_BAX.set(pts.subarray(0, 2 * nS));
  for (let s = 0; s < nS; s++) {
    CURVE_V[s] = s / (nS - 1);
    CURVE_ACR[s] = 0;
    CURVE_HWJ[s] = -1;
  }
  curveN = nS;
  return nS;
}

/** Results of netAt: the network (index) whose plain or cone holds the point (-1: none), how far out from its apex (0) toward its shoreline (1), the shoreline's distance beyond the point (radians), and how firmly it is the network's rather than the coast's beside it (0–1, fading at its flanks). */
let netHit = -1;
let netRho = 0;
let netGap = 0;
let netFirm = 0;

/** atan2 to about 1e-5 rad, several times faster (called for every pixel near a network). */
function fastAtan2(y: number, x: number): number {
  const ax = Math.abs(x);
  const ay = Math.abs(y);
  const a = Math.min(ax, ay) / (Math.max(ax, ay) + 1e-300);
  const s = a * a;
  let t = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a;
  if (ay > ax) t = 1.5707963267948966 - t;
  if (x < 0) t = Math.PI - t;
  return y < 0 ? -t : t;
}

/** Which of the tile's networks holds unit vector p (see netHit); `slack` (radians) extends shorelines outward. */
function netAt(f: FieldData, px: number, py: number, pz: number, slack: number): void {
  netHit = -1;
  const { netFrame, netTheta, netFront } = f;
  for (let k = 0; k < tileNetN; k++) {
    const n = tileNets[k];
    const o = 9 * n;
    const d = px * netFrame[o] + py * netFrame[o + 1] + pz * netFrame[o + 2];
    if (d < tileCos[k]) continue;
    const u = (px * netFrame[o + 3] + py * netFrame[o + 4] + pz * netFrame[o + 5]) / d;
    const v = (px * netFrame[o + 6] + py * netFrame[o + 7] + pz * netFrame[o + 8]) / d;
    const t0 = netTheta[2 * n];
    const t1 = netTheta[2 * n + 1];
    const th = fastAtan2(v, u);
    if (th < t0 || th > t1) continue;
    const a = ((th - t0) / (t1 - t0)) * (NET_BINS - 1);
    const b = Math.min(NET_BINS - 2, Math.floor(a));
    const Rf = netFront[NET_BINS * n + b] + (netFront[NET_BINS * n + b + 1] - netFront[NET_BINS * n + b]) * (a - b);
    const r = Math.hypot(u, v);
    if (r >= Rf + slack) continue;
    netHit = n;
    netRho = r / Math.max(Rf, 1e-9);
    netGap = Rf - r;
    const af = (th - t0) / (t1 - t0);
    netFirm = smooth01(af / 0.07) * smooth01((1 - af) / 0.07);
    return;
  }
}

// ------------------------------------------------------------------ the renderer

/**
 * Renders a window of the pixel pyramid. Cell values are blended with a smooth
 * compact kernel over slope-limited local planes (C¹, no facets). Land and sea
 * are interpolated separately and the coastline follows the cells' own
 * land/sea split, roughened by fractal noise. Sub-cell relief, coastline detail,
 * meanders and valleys gain octaves as the pixel size shrinks, so every zoom
 * level adds detail while agreeing with the levels above it.
 */
export function renderWindow(ctx: RenderContext, opt: RenderOptions, win: PixelWindow, wantElevation = false): RenderResult {
  const md = ctx.mesh;
  const f = ctx.fields;
  const L = win.level;
  const HL = levelHeight(L);
  const dRad = Math.PI / HL;
  const R = f.radiusKm * 1000;
  const h = md.spacing;
  const hPx = h / dRad;
  const K = octavesFor(hPx);
  const Wg = win.w + 2 * APRON;
  const Hg = win.h + 2 * APRON;
  const x0g = win.x0 - APRON;
  const y0g = win.y0 - APRON;
  scratch(Wg * Hg);
  const noise = noiseFor(f.noiseSeed);
  const lcode = LAYER_CODE[opt.layer];
  const lc = layerCells(f, opt);
  const showRivers = opt.rivers && RIVER_LAYERS.has(lcode) && f.rivCell.length > 0;
  const carveOn = f.rivCell.length > 0 && valleyHalfWidth(f.riverThreshold, R * dRad) >= 1.2;
  rasterRivers(md, f, L, x0g, y0g, Wg, Hg, dRad, showRivers, carveOn, lcode);
  // Deltas, fans and inland deltas: their plains shape the coast whether or not rivers show.
  rasterNetworks(md, f, L, x0g, y0g, Wg, Hg, dRad, showRivers, lcode);

  const { x: mx, y: my, z: mz, gStart, gCell, tri } = md;
  const { elev, grad, amp, mtn, ocean, lake, lakeLevel, lakeKind, reef, color, dryness, ice, iceCover, iceMelt, shelf, scour, fjord, cellNet } = f;
  const kr2 = (KERNEL_RADIUS * h) ** 2;
  const invKr2 = 1 / kr2;
  const f0 = 0.75 / h;
  const fa = lc.a;
  const fb = lc.b;
  const modeA = lc.modeA;
  const modeB = lc.modeB;
  const wantColor = lcode === L_TERRAIN;
  const needCat = CATEGORICAL.has(lcode);
  const ringW = Math.max(1.2 * dRad, 0.008 * h);
  const fMotu = 0.75 / (0.07 * h);
  /** Small ice-scoured lakes: about a sixth of a cell across, finer ones a third of that. */
  const fScour = 5 * f0;
  /** How far a delta's ragged shoreline may reach past its outline (radians). */
  const netSlack = 1500 / R;
  /** Any network's ground reaching this tile (their boxes hold their cells' kernels). */
  const netsHere = tileNetN > 0;

  // Domain warp for categorical boundaries, so they wander instead of tracing
  // the cells' straight Voronoi edges (blocky once cells span a few pixels).
  // The base octave is on whenever a cell spans 4 px or more; finer octaves
  // join as cells grow on screen. Evaluated on a lattice fine enough for the
  // cell size (2, 4 or 8 px: steps that divide tile offsets, so tiles agree).
  const warpOct = needCat && hPx >= 4 ? Math.min(4, Math.max(1, Math.floor(Math.log2(hPx / 32)) + 1)) : 0;
  const WB = hPx >= 32 ? 8 : hPx >= 12 ? 4 : 2;
  const wcw = Math.ceil(Wg / WB) + 1;
  const wch = Math.ceil(Hg / WB) + 1;
  const warp = warpOct > 0 ? new Float32Array(wcw * wch * 3) : null;
  if (warp) {
    const aw = 0.35 * h;
    for (let bj = 0; bj < wch; bj++) {
      const lat = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, Math.PI / 2 - (y0g + bj * WB + 0.5) * dRad));
      for (let bi = 0; bi < wcw; bi++) {
        const lon = -Math.PI + (x0g + bi * WB + 0.5) * dRad;
        const px = Math.cos(lat) * Math.cos(lon);
        const py = Math.cos(lat) * Math.sin(lon);
        const pz = Math.sin(lat);
        for (let c = 0; c < 3; c++) {
          let s = 0;
          let a = 1;
          let fr = f0;
          for (let k = 0; k < warpOct; k++) {
            s += a * noise.noise(px * fr + 41.7 * c + 3.1 * k, py * fr - 13.3 * c, pz * fr + 7.9 * c - 5.3 * k);
            a *= 0.5;
            fr *= 2;
          }
          warp[3 * (bj * wcw + bi) + c] = aw * s;
        }
      }
    }
  }

  const cosL = new Float64Array(Wg);
  const sinL = new Float64Array(Wg);
  for (let i = 0; i < Wg; i++) {
    const lon = -Math.PI + (x0g + i + 0.5) * dRad;
    cosL[i] = Math.cos(lon);
    sinL[i] = Math.sin(lon);
  }
  const lat0 = Math.PI / 2 - (y0g + 0.5) * dRad;
  let tRow = seedTriangle(md, Math.max(-1.57, Math.min(1.57, lat0)), -Math.PI + (x0g + 0.5) * dRad);

  for (let j = 0; j < Hg; j++) {
    const lat = Math.max(-Math.PI / 2 + 1e-7, Math.min(Math.PI / 2 - 1e-7, Math.PI / 2 - (y0g + j + 0.5) * dRad));
    const cl = Math.cos(lat);
    const pz = Math.sin(lat);
    let t = tRow;
    for (let i = 0; i < Wg; i++) {
      const p = j * Wg + i;
      const px = cl * cosL[i];
      const py = cl * sinL[i];
      t = locate(md, px, py, pz, t);
      if (i === 0) tRow = t;

      let sw = 0;
      let swL = 0;
      let swO = 0;
      let seL = 0;
      let seO = 0;
      let nAt = 0;
      let sChi = 0;
      let sA = 0;
      let sM = 0;
      let sLake = 0;
      let sReef = 0;
      let sa = 0;
      let sb = 0;
      let cr = 0;
      let cg = 0;
      let cb = 0;
      let sDry = 0;
      let sIce = 0;
      let sCov = 0;
      let sMelt = 0;
      let sShelf = 0;
      let sScour = 0;
      let sFjord = 0;
      let sNet = 0;
      let netCellW = 0;
      let netOfCells = -1;
      let bestL = -1;
      let bestLw = 0;
      let bestO = -1;
      let bestOw = 0;
      let lakeId = -1;
      let lakeW = 0;
      const g0 = gStart[t];
      const g1 = gStart[t + 1];
      for (let k = g0; k < g1; k++) {
        const q = gCell[k];
        const dx = px - mx[q];
        const dy = py - my[q];
        const dz = pz - mz[q];
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 >= kr2) continue;
        let w = 1 - d2 * invKr2;
        w *= w;
        const v = elev[q] + grad[3 * q] * dx + grad[3 * q + 1] * dy + grad[3 * q + 2] * dz;
        sw += w;
        sA += w * amp[q];
        sM += w * mtn[q];
        sScour += w * scour[q];
        sFjord += w * fjord[q];
        if (ocean[q]) {
          swO += w;
          seO += w * v;
          sChi -= w;
          sShelf += w * shelf[q];
          if (w > bestOw) {
            bestOw = w;
            bestO = q;
          }
          const rf = reef[q];
          if (rf === 1) {
            if (nAt < 4) AT[nAt++] = q;
          } else if (rf === 2) sReef += w;
          if (fa !== null && modeA !== MODE_LAND) sa += w * fa[q];
          if (fb !== null && modeB !== MODE_LAND) sb += w * fb[q];
        } else {
          swL += w;
          seL += w * v;
          sChi += w;
          sIce += w * ice[q];
          sCov += w * iceCover[q];
          sMelt += w * iceMelt[q];
          const cn = netsHere ? cellNet[q] : -1;
          if (cn >= 0) {
            sNet += w;
            if (w > netCellW) {
              netCellW = w;
              netOfCells = cn;
            }
          }
          if (w > bestLw) {
            bestLw = w;
            bestL = q;
          }
          const lk = lake[q];
          if (lk >= 0) {
            sLake += w;
            if (w > lakeW) {
              lakeW = w;
              lakeId = lk;
            }
          }
          if (wantColor) {
            cr += w * color[3 * q];
            cg += w * color[3 * q + 1];
            cb += w * color[3 * q + 2];
            sDry += w * dryness[q];
          }
          if (fa !== null && modeA !== MODE_SEA) sa += w * fa[q];
          if (fb !== null && modeB !== MODE_SEA) sb += w * fb[q];
        }
      }
      if (sw <= 0) {
        // Degenerate gather (should not happen): fall back to the first vertex.
        const q = tri[3 * t];
        sw = 1;
        sA = amp[q];
        if (ocean[q]) {
          swO = 1;
          seO = elev[q];
          sChi = -1;
          bestO = q;
        } else {
          swL = 1;
          seL = elev[q];
          sChi = 1;
          bestL = q;
          cr = color[3 * q];
          cg = color[3 * q + 1];
          cb = color[3 * q + 2];
        }
        if (fa !== null) sa = fa[q];
        if (fb !== null) sb = fb[q];
      }
      const A = sA / sw;
      const M = sM / sw;
      let chi = sChi / sw;
      if (swL > 0 && swO > 0) chi = roughChi(noise, chi, M, sFjord / sw, px, py, pz, f0, K);
      const Dn = subcellRelief(noise, px, py, pz, f0, swL > 0 ? K : Math.min(K, 3), M);

      let e: number;
      let cls = LAND;
      let flags = 0;
      let lakeDepth = 0;
      let S: number;
      const meshLand = swO === 0 ? true : swL === 0 ? false : chi > 0;
      // Deltas, fans and inland deltas reaching this pixel: a sea delta's plain
      // (out to its shoreline, ragged like any coast and smoother where waves
      // straighten it) and the levees along its channels are land.
      let dNet = -1;
      let dRho = 0;
      let dGap = 0;
      let deltaLand = false;
      if (tileNetN > 0) {
        netAt(f, px, py, pz, netSlack);
        if (netHit >= 0) {
          dNet = netHit;
          dRho = netRho;
          dGap = netGap / dRad;
          const io = NET_INFO * netHit;
          if (f.netInfo[io] === 0 && netFirm > 0.5) {
            // (The ragged edge only matters near the shoreline.)
            const amp = netSlack * (1 - 0.6 * f.netInfo[io + 2]);
            deltaLand = netGap > amp ? true : netGap + amp * 0.5 * raggedEdge(noise, px, py, pz, f0 * 3, K) > 0;
          }
        }
        if (LEVEE[p] > 0.5) deltaLand = true;
      }
      const isLand = meshLand || deltaLand;
      // How much this is the network's own plain or cone: a delta's deposits
      // and levees and the land it built out; a fan's or inland delta's
      // ground, fading out at its margin and flanks.
      let plainW = 0;
      if (netOfCells >= 0 && swL > 0) {
        // (Over the network's own cells, blended: the plain follows the ground it built.)
        if (dNet < 0 || netOfCells !== dNet) {
          dNet = netOfCells;
          dRho = 0.5;
          dGap = 1e9;
        }
        plainW = smooth01((sNet / swL - 0.12) / 0.5);
      }
      if (dNet >= 0 && f.netInfo[NET_INFO * dNet] === 0) plainW = !meshLand ? 1 : Math.max(plainW, Math.min(1, LEVEE[p]));
      if (isLand && !meshLand) {
        // Delta plain built out over the sea: just above it, highest toward the apex.
        S = 0.5 + 1.8 * (1 - Math.min(1, dRho)) ** 2;
        e = S + 0.35 * Dn;
      } else if (isLand) {
        S = seL / swL;
        const dry = landSurface(noise, px, py, pz, h, R, A, Dn, CARVE[p], S, sIce / swL, sCov / swL, sCov > 0 ? sIce / sCov : 0, sFjord / sw, sLake / sw, swO / sw, K);
        let e0 = surfE;
        flags = surfFlags;
        if (!dry) {
          cls = SEA;
          e = e0;
        } else {
          // A floodplain or braidplain: a flat valley floor at the river's level
          // (only ever lowered: glacial troughs and lake basins keep their depth).
          const fpl = FLOOD[p];
          if (fpl > 0 && flags === 0 && surfLake === 0) {
            const cv = CARVE[p];
            // Hummocks of the sub-cell relief smoothed away over the whole floodplain...
            e0 -= 0.7 * fpl * A * Dn * (1 - cv * cv);
            // ...and the valley's floor flattened where it is cut deep.
            const floorE = S - (0.3 * A + 4);
            if (e0 > floorE) e0 += (floorE - e0) * 0.9 * fpl * smooth01((cv - 0.25) / 0.5);
          }
          // Coastal land rises from the shoreline.
          if (swO > 0) e0 = 1 + (e0 - 1) * smooth01(chi / 0.3);
          // Hollows of the sub-cell relief and valley floors stay above the sea
          // unless the cells themselves lie below it (a closed depression): a
          // soft floor, so low plains level out gradually wherever they are
          // (a floor only near the coast left a stepped arc where it ended).
          const floor = Math.min(S, 0.5);
          const dFl = e0 - floor;
          e0 = floor + 0.5 * (dFl + Math.sqrt(dFl * dFl + 2.25));
          e = e0;
          if (surfLake > 0) {
            // A lake in a glacial trough's basin behind its sill.
            cls = LAKE;
            lakeDepth = surfLake;
          } else if (
            flags === 0 &&
            lakeId >= 0 &&
            e < lakeLevel[lakeId] &&
            // Where the ground alone doesn't hem the lake in, a ragged edge rather
            // than a circle, gaining finer coves and points as you zoom in.
            sLake / sw + 0.3 * raggedEdge(noise, px, py, pz, f0 * 2.5, K) > 0.22
          ) {
            const kind = lakeKind[lakeId];
            if (kind === 2) {
              // Salt flat: a dead-level crust where the basin's floods evaporate.
              e = lakeLevel[lakeId];
              flags = F_FLAT;
            } else {
              cls = kind === 1 ? SALT : LAKE;
              lakeDepth = lakeLevel[lakeId] - e;
            }
          } else if (flags === 0 && hPx >= 20 && sScour > 0.01 * sw) {
            // Ice-scoured lowlands: countless small lakes in the hollows the ice
            // left (finer ones appear as you zoom in), thinning out smoothly
            // toward unscoured or mountainous ground.
            const sc = sScour / sw;
            let nl = 0.7 * noise.noise(px * fScour + 211.7, py * fScour - 63.1, pz * fScour + 17.9);
            if (hPx >= 56) nl += 0.45 * noise.noise(px * fScour * 2.8 - 91.3, py * fScour * 2.8 + 44.4, pz * fScour * 2.8 + 8.8);
            // Their shores keep gaining detail at the deepest levels.
            if (hPx >= 150) nl += 0.25 * noise.noise(px * fScour * 7.6 + 30.1, py * fScour * 7.6 - 12.7, pz * fScour * 7.6 + 61.3);
            if (hPx >= 400) nl += 0.14 * noise.noise(px * fScour * 20.5 - 4.9, py * fScour * 20.5 + 77.2, pz * fScour * 20.5 - 33.6);
            const thr = 0.95 - 0.6 * sc + 0.8 * smooth01(M / 0.6);
            if (nl > thr) {
              cls = LAKE;
              flags = F_SCOUR;
              lakeDepth = (nl - thr) * 300;
            }
          }
        }
      } else {
        S = seO / swO;
        if (swL > 0) {
          // Extrapolated landward along their slopes, the ocean cells' surface
          // can rise above the sea near the shore: it stays below a shoreface
          // deepening away from the coast (else the shallows there would be
          // clipped flat at the waterline, a plateau with a hard edge).
          // (It lifts away out where the land's kernel ends: no edge there.)
          const face = -2 - 40 * smooth01(-chi / 0.5) + 4000 * smooth01((-chi - 0.6) / 0.35);
          const dFc = S - face;
          S = face + 0.5 * (dFc - Math.sqrt(dFc * dFc + 16));
        }
        let e0 = S + (0.6 * A + 12) * Dn;
        if (swL > 0) e0 = -1 + (e0 + 1) * smooth01(-chi / 0.3);
        const dSe = e0 + 0.5;
        e = -0.5 + 0.5 * (dSe - Math.sqrt(dSe * dSe + 0.36));
        cls = SEA;
        const shelfK = sShelf / swO;
        if (shelfK > 0.05) {
          // Floating ice shelf with a ragged calving front.
          const front = raggedEdge(noise, px, py, pz, f0 * 3, K);
          if (shelfK + 0.3 * front > 0.55) {
            flags = F_SHELF;
            e = -1;
          }
        }
      }

      // Ponds and swamps between the channels of a river-built plain: lakes of
      // a wet inland delta, the marshy outer plain of a river-dominated delta.
      if (dNet >= 0 && plainW > 0.6 && cls === LAND && flags === 0 && hPx >= 12) {
        const io = NET_INFO * dNet;
        const kind = f.netInfo[io];
        const wet = 1 - smooth01((f.netInfo[io + 7] - 0.4) / 0.3);
        if (wet > 0 && LEVEE[p] < 0.2 && COVER[p] < 0.05 && (kind === 2 || (kind === 0 && f.netInfo[io + 1] <= 1))) {
          const fp = R / 4500;
          // (Ponds a few pixels across or more: below that they would be speckle.)
          const thr = (kind === 2 ? 0.42 : 0.78 - 0.38 * smooth01((dRho - 0.4) / 0.45)) + 0.6 * (1 - wet) + 2 * (1 - smooth01((fp * dRad) ** -1 / 3 - 1)) + 0.5 * (1 - plainW);
          let pn = noise.noise(px * fp + 71.3, py * fp - 12.9, pz * fp + 33.1);
          // (Finer octaves only where they could still lift it past the threshold.)
          if (hPx >= 40 && pn > thr - 0.65) pn += 0.45 * noise.noise(px * fp * 2.7 - 5.5, py * fp * 2.7 + 48.2, pz * fp * 2.7 - 9.1);
          if (hPx >= 120 && pn > thr - 0.2) pn += 0.2 * noise.noise(px * fp * 7.3 + 15.1, py * fp * 7.3 - 2.6, pz * fp * 7.3 + 60.4);
          if (pn > thr) {
            cls = LAKE;
            lakeDepth = 1 + 8 * (pn - thr);
          }
        }
      }

      // Atolls: a reef ring around a lagoon, crowned with sandy islets, rising
      // steeply from the open seafloor around it (never over land or shelf ice).
      let atollZone = 0;
      let motu = 0;
      if (nAt > 0 && cls === SEA && flags === 0) {
        let d2min = Infinity;
        for (let k = 0; k < nAt; k++) {
          const q = AT[k];
          const dx = px - mx[q];
          const dy = py - my[q];
          const dz = pz - mz[q];
          d2min = Math.min(d2min, dx * dx + dy * dy + dz * dz);
        }
        const dist = Math.sqrt(d2min);
        const wob = noise.noise(px * f0 * 4 + 7.7, py * f0 * 4, pz * f0 * 4 - 2.2);
        const ringR = 0.16 * h * (1 + 0.25 * wob);
        if (dist < ringR - ringW) {
          e = -10 - 26 * (1 - dist / ringR);
          cls = SEA;
          atollZone = 1;
        } else if (dist <= ringR + ringW) {
          motu = noise.noise(px * fMotu - 3.3, py * fMotu + 1.9, pz * fMotu);
          if (motu > 0.2) {
            e = 2 + 4 * (motu - 0.2);
            cls = LAND;
            atollZone = 3;
          } else {
            e = -1;
            cls = SEA;
            atollZone = 2;
          }
        } else if (cls === SEA) {
          const flank = -1 - 4200 * ((dist - ringR - ringW) / (0.5 * h)) ** 0.75;
          if (flank > e) e = Math.min(-0.5, flank);
        }
      }
      E[p] = e;
      SM[p] = flags & (F_FLAT | F_SHELF) ? e : S;
      CLS[p] = cls;
      FLAGS[p] = flags;

      // Category cell (nearest cell of the pixel's own kind, at a warped position).
      if (needCat) {
        let cat = cls === SEA ? (bestO >= 0 ? bestO : bestL) : bestL >= 0 ? bestL : dNet >= 0 ? f.netCell[dNet] : bestO;
        if (warp) {
          const fx = i / WB;
          const fy = j / WB;
          const bi = Math.floor(fx);
          const bj = Math.floor(fy);
          const tx = fx - bi;
          const ty = fy - bj;
          const o00 = 3 * (bj * wcw + bi);
          const o10 = o00 + 3;
          const o01 = o00 + 3 * wcw;
          const o11 = o01 + 3;
          const wx = px + (warp[o00] * (1 - tx) + warp[o10] * tx) * (1 - ty) + (warp[o01] * (1 - tx) + warp[o11] * tx) * ty;
          const wy = py + (warp[o00 + 1] * (1 - tx) + warp[o10 + 1] * tx) * (1 - ty) + (warp[o01 + 1] * (1 - tx) + warp[o11 + 1] * tx) * ty;
          const wz = pz + (warp[o00 + 2] * (1 - tx) + warp[o10 + 2] * tx) * (1 - ty) + (warp[o01 + 2] * (1 - tx) + warp[o11 + 2] * tx) * ty;
          let best = -2;
          const wantSea = cls === SEA ? 1 : 0;
          for (let k = g0; k < g1; k++) {
            const q = gCell[k];
            if (ocean[q] !== wantSea) continue;
            const d = wx * mx[q] + wy * my[q] + wz * mz[q];
            if (d > best) {
              best = d;
              cat = q;
            }
          }
        }
        CAT[p] = cat;
      }

      // Base color before shading.
      let r = 0;
      let g = 0;
      let b = 0;
      const water = cls !== LAND;
      switch (lcode) {
        case L_TERRAIN: {
          if (cls === LAND) {
            if (atollZone === 3) {
              const veg = smooth01((motu - 0.3) / 0.25);
              r = 226 + (96 - 226) * veg;
              g = 220 + (140 - 220) * veg;
              b = 190 + (74 - 190) * veg;
              TP[p] = 25;
            } else if (flags & F_FLAT) {
              const v = 1 + 0.025 * Dn;
              r = 232 * v;
              g = 228 * v;
              b = 214 * v;
              TP[p] = sa / swL;
            } else if (flags & F_ICE) {
              // Snow-covered ice in the accumulation zone; bare blue ice where it melts.
              const melt = 0.8 * (sMelt / swL);
              const v = 1 + 0.02 * Dn;
              r = (240 + (188 - 240) * melt) * v;
              g = (244 + (212 - 244) * melt) * v;
              b = (248 + (232 - 248) * melt) * v;
              TP[p] = -30;
            } else {
              let tp: number;
              if (swL > 0) {
                const v = (1 + 0.05 * Dn) / swL;
                r = cr * v;
                g = cg * v;
                b = cb * v;
                tp = sa / swL - (LAPSE * (e - S)) / 1000;
              } else {
                // Plain built out past the cells' land: the apex's ground.
                const io = NET_INFO * Math.max(0, dNet);
                const v = 1 + 0.05 * Dn;
                r = f.netInfo[io + 8] * v;
                g = f.netInfo[io + 9] * v;
                b = f.netInfo[io + 10] * v;
                tp = f.netInfo[io + 11];
              }
              if (dNet >= 0) {
                const io = NET_INFO * dNet;
                const kind = f.netInfo[io];
                const dryN = f.netInfo[io + 7];
                if (kind === 1) {
                  // A fan: gravel and sand, bare in dry country.
                  const k = (0.12 + 0.5 * dryN) * (1 - 0.4 * dRho) * plainW;
                  r += (196 - r) * k;
                  g += (182 - g) * k;
                  b += (152 - b) * k;
                } else {
                  // Marsh and wet meadow; mudflats where it is dry.
                  const k = 0.32 * smooth01((tp + 2) / 8) * plainW;
                  r += (90 + 76 * dryN - r) * k;
                  g += (114 + 34 * dryN - g) * k;
                  b += (68 + 58 * dryN - b) * k;
                  if (kind === 0) {
                    const form = f.netInfo[io + 1];
                    const R0 = f.netInfo[io + 5] / dRad;
                    if (form === 4) {
                      // Tidal flats along a tide-built front, threaded by creeks.
                      const fw = (0.06 + 0.1 * f.netInfo[io + 3]) * R0;
                      const kf = 0.65 * (1 - smooth01(dGap / Math.max(1, fw))) * netFirm;
                      if (kf > 0) {
                        r += (158 - r) * kf;
                        g += (146 - g) * kf;
                        b += (122 - b) * kf;
                        const fc = R / 1400;
                        const cn = Math.abs(noise.noise(px * fc - 3.7, py * fc + 21.1, pz * fc - 8.4));
                        const kc = kf * (1 - smooth01((cn - 0.02) / 0.03)) * smooth01(1 / (fc * dRad) / 3 - 1);
                        r += (84 - r) * kc;
                        g += (116 - g) * kc;
                        b += (138 - b) * kc;
                      }
                    } else if (form >= 1 && form <= 3) {
                      // Beach ridges parallel to a wave-built front, older inland.
                      const lam = 380 / (R * dRad);
                      const zone = (0.15 + 0.35 * f.netInfo[io + 2]) * R0;
                      const kr = (form === 1 ? 0.35 : 1) * (1 - smooth01(dGap / Math.max(1, zone))) * smooth01((lam - 3) / 3) * netFirm;
                      if (kr > 0) {
                        const kk = 0.22 * kr * Math.cos((2 * Math.PI * dGap) / lam + 0.8 * Dn);
                        r += (kk > 0 ? 214 - r : 70 - r) * Math.abs(kk);
                        g += (kk > 0 ? 200 - g : 98 - g) * Math.abs(kk);
                        b += (kk > 0 ? 156 - b : 58 - b) * Math.abs(kk);
                      }
                    }
                  }
                }
              }
              // Green ribbons where rivers run through dry country (if summers are warm enough for shrubs).
              const rip = RIPARIAN[p] * smooth01((sDry / swL - 0.2) / 0.5) * smooth01((tp - 6) / 8) * 0.75;
              if (rip > 0) {
                r += (70 - r) * rip;
                g += (104 - g) * rip;
                b += (50 - b) * rip;
              }
              // Scroll bars inside meander bends: ridges and swales parallel to
              // the channel, fading away from it.
              const sc = SCRL[p];
              if (sc > 1) {
                const u = sc - 1;
                const k = 0.075 * (1 - smooth01(u / 4.5)) * Math.cos((2 * Math.PI * u) / 0.65);
                r *= 1 + k;
                g *= 1 + 0.85 * k;
                b *= 1 + 0.6 * k;
              }
              // Loops cut off long ago and silted up: wet meadow.
              const scar = SCAR[p];
              if (scar > 0) {
                const k = 0.3 * scar * smooth01((tp + 2) / 8);
                r += (92 - r) * k;
                g += (122 - g) * k;
                b += (76 - b) * k;
              }
              // A braided or anastomosing river's belt and the point bars in
              // bends: bare sand or gravel, or bars overgrown into islands.
              const bt = BELT[p];
              if (bt > 0) {
                const bare = BARE[p];
                const gv = GRAV[p];
                const tex = 1 + 0.035 * Dn;
                const kb = bt * bare;
                r += ((214 + (170 - 214) * gv) * tex - r) * kb;
                g += ((202 + (163 - 202) * gv) * tex - g) * kb;
                b += ((172 + (150 - 172) * gv) * tex - b) * kb;
                const kv = bt * (1 - bare) * 0.3 * smooth01((tp + 2) / 8);
                r += (62 - r) * kv;
                g += (94 - g) * kv;
                b += (52 - b) * kv;
              }
              TP[p] = tp;
            }
          } else if (cls === SEA) {
            if (flags & F_SHELF) {
              [r, g, b] = [228, 236, 244];
            } else if (atollZone === 1) {
              [r, g, b] = [78 + 30 * (e / -34), 190 - 20 * (e / -34), 190];
            } else if (atollZone === 2) {
              [r, g, b] = [168, 222, 210];
            } else {
              [r, g, b] = pal.ramp(pal.SEA, e);
              // Reef flats: turquoise where tropical shallows are, fading out smoothly.
              const reefK = 0.4 * smooth01((sReef / swO - 0.25) / 0.5) * smooth01((e + 70) / 40);
              if (reefK > 0) {
                r += (70 - r) * reefK;
                g += (176 - g) * reefK;
                b += (184 - b) * reefK;
              }
              const ice = Math.max(0, Math.min(1, (-1.5 - sb / swO) / 4));
              r += (226 - r) * ice;
              g += (236 - g) * ice;
              b += (242 - b) * ice;
            }
          } else if (swL > 0 && sa / swL < -1) {
            // Frozen over (all year in the annual view, or in the month shown).
            [r, g, b] = [212, 226, 236];
          } else if (cls === SALT) {
            const k = Math.min(1, lakeDepth / 20);
            r = 176 + (120 - 176) * k;
            g = 204 + (168 - 204) * k;
            b = 192 + (170 - 192) * k;
          } else {
            const k = Math.min(1, lakeDepth / 60);
            r = 70 + (40 - 70) * k;
            g = 124 + (82 - 124) * k;
            b = 156 + (120 - 156) * k;
          }
          break;
        }
        case L_ELEV:
          [r, g, b] = water ? (cls === SEA ? pal.ramp(pal.BATHY, e) : [70, 130, 190]) : pal.ramp(pal.HYPSO, e);
          break;
        case L_PLATES: {
          const c = CAT[p];
          const pl = f.plate[c];
          const k = f.ctype[c] ? 1 : 0.62;
          r = f.plateColor[3 * pl] * k;
          g = f.plateColor[3 * pl + 1] * k;
          b = f.plateColor[3 * pl + 2] * k;
          break;
        }
        case L_CRUST:
          [r, g, b] = f.ctype[CAT[p]] ? pal.ramp(pal.CONT_THICK, sa / sw) : pal.ramp(pal.OCEAN_AGE, sb / sw);
          break;
        case L_TEMP:
          [r, g, b] = pal.ramp(pal.TEMPERATURE, cls === LAND && swL > 0 ? sa / swL - (LAPSE * (e - S)) / 1000 : sb / sw);
          break;
        case L_PRECIP:
          [r, g, b] = pal.ramp(pal.PRECIP, sb / sw);
          break;
        case L_BIOMES: {
          const c = CAT[p];
          if (water) [r, g, b] = cls === SEA ? pal.ramp(pal.SEA, e) : BIOME_INFO[16].color;
          else if (flags & F_FLAT) [r, g, b] = BIOME_INFO[17].color;
          else [r, g, b] = BIOME_INFO[f.biome[c] === 16 || f.biome[c] === 17 ? 7 : f.biome[c]].color;
          break;
        }
        case L_KOPPEN: {
          const c = CAT[p];
          if (water) [r, g, b] = cls === SEA ? [40, 52, 72] : [120, 160, 200];
          else [r, g, b] = KOPPEN_COLORS[KOPPEN_CODES[f.koppen[c]]] ?? [128, 128, 128];
          break;
        }
        case L_DRAIN: {
          const c = CAT[p];
          if (water) [r, g, b] = cls === SEA ? [28, 36, 52] : [90, 140, 200];
          else [r, g, b] = pal.hashColor(f.drainage[c]);
          break;
        }
        case L_CHANNELS: {
          if (cls === SEA) [r, g, b] = [28, 36, 52];
          else if (water) [r, g, b] = [90, 140, 200];
          else {
            // Pale ground; floodplains a little warmer, belts tinted by what lies between their channels.
            r = 212;
            g = 210;
            b = 204;
            const fpl = FLOOD[p];
            if (fpl > 0) {
              r += (234 - r) * 0.6 * fpl;
              g += (228 - g) * 0.6 * fpl;
              b += (206 - b) * 0.6 * fpl;
            }
            const bt = BELT[p];
            if (bt > 0) {
              const veg = 1 - BARE[p];
              const k = 0.75 * bt;
              r += (240 + (150 - 240) * veg - r) * k;
              g += (190 + (204 - 190) * veg - g) * k;
              b += (130 + (140 - 130) * veg - b) * k;
            }
            // Delta plains, fans and inland deltas tinted as their distributaries are.
            if (dNet >= 0) {
              const dc = pal.CHANNEL_COLORS[Channel.Distributary + 1];
              const k = (f.netInfo[NET_INFO * dNet] === 1 ? 0.2 : 0.26) * plainW;
              r += (dc[0] - r) * k;
              g += (dc[1] - g) * k;
              b += (dc[2] - b) * k;
            }
            if (OXB[p] > 0) {
              const k = OXB[p];
              r += (90 - r) * k;
              g += (150 - g) * k;
              b += (210 - b) * k;
            }
          }
          break;
        }
        case L_ROCK: {
          // The seafloor's rock shows dimmed under the water.
          const rc = ROCK_INFO[f.rock[CAT[p]]].color;
          if (cls === SEA) [r, g, b] = [14 + 0.45 * rc[0], 22 + 0.45 * rc[1], 40 + 0.45 * rc[2]];
          else if (water) [r, g, b] = [90, 140, 200];
          else [r, g, b] = rc;
          break;
        }
        case L_DEPOSITS: {
          // Shelf sediment shows under the sea; deeper water stays dark.
          const k = f.deposit[CAT[p]];
          if (cls === SEA) [r, g, b] = k === 14 && e > -400 ? [40, 70, 98] : [22, 30, 46];
          else if (water) [r, g, b] = [90, 140, 200];
          else [r, g, b] = DEPOSIT_COLORS[k] ?? DEPOSIT_COLORS[0];
          break;
        }
        case L_SOILS: {
          const c = CAT[p];
          if (cls === SEA) [r, g, b] = [22, 30, 46];
          else if (water) [r, g, b] = [90, 140, 200];
          else if (flags & F_ICE) [r, g, b] = SOIL_INFO[12].color;
          else {
            [r, g, b] = SOIL_INFO[f.soil[c]].color;
            if (f.saline[c]) {
              // Salt crust: the desert soil whitened.
              r += (250 - r) * 0.45;
              g += (250 - g) * 0.45;
              b += (246 - b) * 0.45;
            }
          }
          break;
        }
        case L_FERT:
          if (cls === SEA) [r, g, b] = [22, 30, 46];
          else if (water) [r, g, b] = [90, 140, 200];
          else if (flags & F_ICE) [r, g, b] = SOIL_INFO[12].color;
          else [r, g, b] = pal.ramp(pal.FERTILITY, sa / swL);
          break;
      }
      CR[p] = r;
      CG[p] = g;
      CB[p] = b;
    }
  }

  // ---------------------------------------------------------------- shading and composition
  const W = win.w;
  const H = win.h;
  const out = new Uint8ClampedArray(W * H * 4);
  const elevation = wantElevation ? new Float32Array(W * H) : null;
  // Cell-scale relief needs strong exaggeration to read at overview scales;
  // the sub-cell relief is closer to true slopes and gets much less. Neither
  // is ever flattened below its true slope.
  const exagS = Math.max(1, 22 * 2 ** (-0.55 * (L - BASE_LEVEL)));
  const exagD = Math.max(2.5, 10 * 2 ** (-0.75 * (L - BASE_LEVEL)));
  const dy = R * dRad;
  const shadeK = SHADE_K[lcode];
  const lineR = L >= 5 ? 2 : 1;
  const riverCol = lcode === L_TERRAIN ? [58, 100, 140] : [40, 90, 200];
  const lakeCol = lcode === L_TERRAIN ? [70, 124, 156] : [90, 140, 200];
  for (let j = 0; j < H; j++) {
    const jg = j + APRON;
    const lat = Math.PI / 2 - (win.y0 + j + 0.5) * dRad;
    const dx = Math.max(dy * 0.05, Math.cos(lat) * R * dRad);
    for (let i = 0; i < W; i++) {
      const p = jg * Wg + i + APRON;
      const o = 4 * (j * W + i);
      const cls = CLS[p];
      const e = E[p];
      if (elevation) elevation[j * W + i] = e;
      const isLand = cls === LAND;
      const qe = sameSide(p + 1, isLand) ? p + 1 : p;
      const qw = sameSide(p - 1, isLand) ? p - 1 : p;
      const qn = sameSide(p - Wg, isLand) ? p - Wg : p;
      const qs = sameSide(p + Wg, isLand) ? p + Wg : p;
      const sx = (E[qe] - E[qw]) / (2 * dx);
      const sy = (E[qn] - E[qs]) / (2 * dy);
      const smx = (SM[qe] - SM[qw]) / (2 * dx);
      const smy = (SM[qn] - SM[qs]) / (2 * dy);
      const gx = smx * exagS + (sx - smx) * exagD;
      const gy = smy * exagS + (sy - smy) * exagD;
      const len = Math.sqrt(gx * gx + gy * gy + 1);
      let s = Math.max(0.35, Math.min(1.45, (0.6 * gx - 0.6 * gy + 0.53) / len / 0.53));
      let r = CR[p];
      let g = CG[p];
      let b = CB[p];
      let k = shadeK;
      let snow = 0;
      const fl = FLAGS[p];
      if (fl & F_ICE) {
        // Glacier ice: its own color; any drainage on it runs as ice, not open water.
        snow = 1;
        k = Math.min(k, 0.85);
      } else if (lcode === L_TERRAIN) {
        if (isLand) {
          const slope = Math.hypot(sx, sy);
          const rock = fl & F_FLAT ? 0 : Math.max(Math.max(0, Math.min(1, (e - 1800) / 2200)) * 0.55, smooth01((slope - 0.3) / 0.45) * 0.7);
          r += (128 - r) * rock;
          g += (118 - g) * rock;
          b += (108 - b) * rock;
          snow = Math.max(0, Math.min(1, (1.5 - TP[p]) / 4)) * (1 - 0.6 * smooth01((slope - 0.6) / 0.5));
          r += (240 - r) * snow;
          g += (243 - g) * snow;
          b += (247 - b) * snow;
        } else k = cls === SEA && !(fl & F_SHELF) ? 0.5 : 0;
      }
      s = 1 + (s - 1) * k;
      r *= s;
      g *= s;
      b *= s;
      if (OUTLINED.has(lcode) && isLand) {
        if (CLS[p + 1] !== LAND || CLS[p + Wg] !== LAND || CLS[p - 1] !== LAND || CLS[p - Wg] !== LAND) {
          r *= 0.35;
          g *= 0.35;
          b *= 0.35;
        } else if (lcode === L_ROCK) {
          // Geological contacts: thin lines where one rock gives way to another.
          const rk = f.rock[CAT[p]];
          if (f.rock[CAT[p + 1]] !== rk || f.rock[CAT[p + Wg]] !== rk) {
            r *= 0.72;
            g *= 0.72;
            b *= 0.72;
          }
        }
      }
      if (lcode === L_PLATES) {
        const pc = f.plate[CAT[p]];
        let edge = false;
        for (let d = 1; d <= lineR && !edge; d++) {
          edge = f.plate[CAT[p + d]] !== pc || f.plate[CAT[p - d]] !== pc || f.plate[CAT[p + d * Wg]] !== pc || f.plate[CAT[p - d * Wg]] !== pc;
        }
        if (edge) {
          const bt = f.bndType[CAT[p]];
          [r, g, b] = bt === B_CONVERGENT ? [235, 50, 45] : bt === B_DIVERGENT ? [60, 140, 255] : bt === B_TRANSFORM ? [70, 220, 110] : [240, 240, 240];
        }
      }
      if (showRivers && isLand) {
        // Under permanent snow the drainage runs as ice, not open water.
        const a = COVER[p] * (1 - snow);
        // A wide channel's banks: a darker, earthier rim along the water.
        const bk = (BANK[p] - COVER[p]) * (1 - snow);
        if (bk > 0) {
          const k = 1 - 0.32 * bk;
          r = r * k + 14 * bk;
          g = g * k + 8 * bk;
          b *= k;
        }
        // Oxbow lakes on the floodplain (still water, under the rivers).
        const ox = OXB[p] * (1 - snow);
        if (ox > 0 && lcode !== L_CHANNELS) {
          r += (lakeCol[0] - r) * ox;
          g += (lakeCol[1] - g) * ox;
          b += (lakeCol[2] - b) * ox;
        }
        if (a > 0) {
          // The pattern layer colors each river by its channel pattern.
          const col = lcode === L_CHANNELS ? pal.CHANNEL_COLORS[PAT[p]] ?? riverCol : riverCol;
          r += (col[0] - r) * a;
          g += (col[1] - g) * a;
          b += (col[2] - b) * a;
        }
      }
      if (lcode === L_DRAIN && showRivers) {
        // Flow arrows: a dark outline, then a white arrowhead.
        const ao = ARO[p];
        if (ao > 0) {
          const af = ARW[p];
          r += (18 - r) * ao;
          g += (26 - g) * ao;
          b += (52 - b) * ao;
          r += (252 - r) * af;
          g += (252 - g) * af;
          b += (246 - b) * af;
        }
      }
      out[o] = r;
      out[o + 1] = g;
      out[o + 2] = b;
      out[o + 3] = 255;
    }
  }
  return { rgba: out, elevation };
}
