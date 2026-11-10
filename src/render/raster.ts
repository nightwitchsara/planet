import { Simplex3 } from '../core/noise';
import { BIOME_INFO, KOPPEN_CODES, KOPPEN_COLORS } from '../world/biomes';
import { LAPSE, MONTHS } from '../world/climate';
import { ROCK_INFO } from '../world/geology';
import { ICE_ISOSTASY } from '../world/glaciers';
import { DEPOSIT_COLORS } from '../world/sediment';
import { B_CONVERGENT, B_DIVERGENT, B_TRANSFORM } from '../world/tectonics/boundaries';
import { CHANNEL_WIDTH, type FieldData, KERNEL_RADIUS, type MeshData, MIN_STREAM_FRACTION, type RenderContext } from './context';
import * as pal from './palettes';

export type LayerId =
  | 'terrain'
  | 'elevation'
  | 'plates'
  | 'crust'
  | 'rocks'
  | 'deposits'
  | 'temperature'
  | 'precipitation'
  | 'biomes'
  | 'koppen'
  | 'drainage';

export const LAYERS: { id: LayerId; label: string; monthly: boolean }[] = [
  { id: 'terrain', label: 'Terrain (natural color)', monthly: true },
  { id: 'elevation', label: 'Elevation', monthly: false },
  { id: 'plates', label: 'Tectonic plates', monthly: false },
  { id: 'crust', label: 'Crust type, thickness & seafloor age', monthly: false },
  { id: 'rocks', label: 'Rock types (bedrock)', monthly: false },
  { id: 'deposits', label: 'Surface deposits', monthly: false },
  { id: 'temperature', label: 'Temperature', monthly: true },
  { id: 'precipitation', label: 'Precipitation', monthly: true },
  { id: 'biomes', label: 'Biomes', monthly: false },
  { id: 'koppen', label: 'Köppen climate', monthly: false },
  { id: 'drainage', label: 'Rivers & drainage basins', monthly: false },
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
const LAYER_CODE: Record<LayerId, number> = {
  terrain: L_TERRAIN, elevation: L_ELEV, plates: L_PLATES, crust: L_CRUST, temperature: L_TEMP,
  precipitation: L_PRECIP, biomes: L_BIOMES, koppen: L_KOPPEN, drainage: L_DRAIN, rocks: L_ROCK, deposits: L_DEPOSITS,
};
/** How strongly hillshading shows on each layer. */
const SHADE_K = [1, 1, 0.4, 0.5, 0.35, 0.35, 0.6, 0.4, 0.5, 0.55, 0.6];
const CATEGORICAL = new Set([L_PLATES, L_CRUST, L_BIOMES, L_KOPPEN, L_DRAIN, L_ROCK, L_DEPOSITS]);
const OUTLINED = new Set([L_TEMP, L_PRECIP, L_KOPPEN, L_CRUST, L_PLATES, L_ROCK, L_DEPOSITS]);
const RIVER_LAYERS = new Set([L_TERRAIN, L_ELEV, L_BIOMES, L_DRAIN, L_ROCK, L_DEPOSITS]);

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
  P[3] = x[a];
  P[4] = y[a];
  P[5] = z[a];
  P[6] = x[b];
  P[7] = y[b];
  P[8] = z[b];
  for (let k = 0; k < 3; k++) {
    P[k] = pa >= 0 ? (k === 0 ? x[pa] : k === 1 ? y[pa] : z[pa]) : 2 * P[3 + k] - P[6 + k];
    P[9 + k] = pb >= 0 ? (k === 0 ? x[pb] : k === 1 ? y[pb] : z[pb]) : 2 * P[6 + k] - P[3 + k];
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
  // Well past Leopold & Wolman's threshold 0.0125·Q_bf^−0.44 (beyond which a
  // channel braids) a river runs straighter: its slope at each node, from the
  // drop between cell centers, which overstates the channel's own (it winds,
  // and the cells' mean surface stands above its valley floor).
  const surf = (c: number): number => (f.ocean[c] ? 0 : f.lake[c] >= 0 ? f.lakeLevel[f.lake[c]] : f.elev[c]);
  const slope = (k: number): number => {
    const c0 = rivCell[k];
    const c1 = rivCell[k + 1];
    return Math.max(0, surf(c0) - surf(c1)) / Math.max(1, Math.acos(Math.min(1, x[c0] * x[c1] + y[c0] * y[c1] + z[c0] * z[c1])) * f.radiusKm * 1000);
  };
  const sinuous = (k: number): number => {
    const S = k > s0 && k < s1 - 1 ? 0.5 * (slope(k - 1) + slope(k)) : slope(k > s0 ? k - 1 : k);
    return 0.25 + 0.75 * smooth01((3 - S / (4 * 0.0125 * Math.max(1, f.rivBank[k]) ** -0.44)) / 2);
  };
  const sinA = sinuous(i);
  const sinB = sinuous(i + 1);
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
  for (let s = 0; s < nPts; s++) {
    let d = 0;
    if (s < nS && kTop > 0) {
      // sin(kπu) by the Chebyshev recurrence.
      const u = s / (nS - 1);
      const s1 = Math.sin(Math.PI * u);
      const c2 = 2 * Math.cos(Math.PI * u);
      let sPrev = 0;
      let sK = s1;
      let db = 0;
      for (let k = 1; k <= kTop; k++) {
        d += amps[k - 1] * sK;
        db += belt[k - 1] * sK;
        const sNext = c2 * sK - sPrev;
        sPrev = sK;
        sK = sNext;
      }
      d = d * s1 + db * smooth01(u * kHi) * smooth01((1 - u) * kHi);
    }
    mp[2 * s] = pts[2 * s] + nxp * d;
    mp[2 * s + 1] = pts[2 * s + 1] + nyp * d;
  }
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
  if (waves <= 0 || waves > CURVE_MAX || nS < 3 || Math.max(showA, showB) <= 0) return nS;
  let axLen = 0;
  for (let s = 1; s < nS; s++) axLen += Math.hypot(mp[2 * s] - mp[2 * s - 2], mp[2 * s + 1] - mp[2 * s - 1]);
  const w0 = f.rivWave[i];
  const seedCell = rivCell[s0];
  // Loops of uneven length: each half-wave's share of the segment, its swing
  // scaled to its length.
  const hw = CURVE_HW;
  hw[0] = 0;
  for (let j = 0; j < waves; j++) hw[j + 1] = hw[j] + 0.4 + 1.2 * hash01(seedCell, w0 + j, 914);
  for (let j = 1; j <= waves; j++) hw[j] /= hw[waves];
  // And of uneven shape, from gentle bends to tight loops: each crossing of
  // the axis has its own ω, and a loop blends from the shape at its start to
  // the shape at its end, so the channel never kinks where it crosses.
  const shape = CURVE_SHAPE;
  for (let c = 0; c <= waves; c++) meanderShape(0.6 + 1.2 * hash01(seedCell, w0 + c, 915), shape, 9 * c);
  const ph1 = 2 * Math.PI * hash01(seedCell, 0, 911);
  const ph2 = 2 * Math.PI * hash01(seedCell, 0, 912);
  const sign = w0 & 1 ? -1 : 1;
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
    // Swinging wide in some reaches and less in others (least on steep or
    // mountain reaches), each loop leaning a little up- or downstream.
    const m = (showA + (showB - showA) * v) * (sinA + (sinB - sinA) * v) * (1 - 0.6 * (mtn[a] + (mtn[b] - mtn[a]) * v)) * (0.7 + 0.2 * Math.sin(0.648 * psi + ph1) + 0.1 * Math.sin(1.461 * psi + ph2));
    const lean = 0.6 * hash01(seedCell, w0 + j, 913) - 0.3;
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
  }
  // Exactly through the end node, then on along the run-on.
  for (let s = nS - 1; s < nPts; s++) {
    const o = s - nS + nOut;
    mp[2 * o] = axis[2 * o] = pts[2 * s];
    mp[2 * o + 1] = axis[2 * o + 1] = pts[2 * s + 1];
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

/**
 * Rasterizes the river network into COVER (line coverage), CARVE (0–1 valley
 * profile) and RIPARIAN (valley floors of rivers that flow most of the year).
 * Each river ends where it first meets the water drawn at this level; its
 * valley runs on to the sea cell and merges with the valleys it meets.
 */
function rasterRivers(md: MeshData, f: FieldData, L: number, x0g: number, y0g: number, Wg: number, Hg: number, dRad: number, draw: boolean, carve: boolean): void {
  COVER.fill(0, 0, Wg * Hg);
  BANK.fill(0, 0, Wg * Hg);
  CARVE.fill(0, 0, Wg * Hg);
  RIPARIAN.fill(0, 0, Wg * Hg);
  if (!draw && !carve) return;
  const { rivStart, rivQ, rivBank, rivFlow, rivBox, rivJoin } = f;
  const R = f.radiusKm * 1000;
  const mPerPx = R * dRad;
  const thrL = riverThresholdFor(f, md.spacing / dRad);
  const thr0 = f.riverThreshold;
  const reach = riverReach(md, f, L);
  const chainOf = chainIndex(f);
  /** Drawn bankfull discharge at node k of a segment whose drawn mean is q (what reached the water upstream is gone). */
  const bankOf = (k: number, q: number): number => (rivQ[k] > 0 ? (rivBank[k] * q) / rivQ[k] : 0);
  const maxV = carve ? Math.min(VALLEY_MAX_PX, valleyHalfWidth(1e5, mPerPx)) : 0;
  const margin = (Math.max(maxV, 12) + 2) * dRad;
  const lonMin = -Math.PI + x0g * dRad - margin;
  const lonMax = -Math.PI + (x0g + Wg) * dRad + margin;
  const latMax = Math.PI / 2 - y0g * dRad + margin;
  const latMin = Math.PI / 2 - (y0g + Hg) * dRad - margin;
  const mp = CURVE_MP;
  const nChains = rivStart.length - 1;
  for (let ch = 0; ch < nChains; ch++) {
    const s0 = rivStart[ch];
    const s1 = rivStart[ch + 1];
    if (streamX0 <= streamX1) mergeStream(Wg);
    for (let i = s0; i < s1 - 1; i++) {
      // The river is drawn until it runs into the drawn water (sea, or a fjord's
      // drowned trough); its valley is carved all the way to the sea cell.
      // (Water lost upstream only ever thins the drawn line.)
      if (rivQ[i + 1] < thrL) continue;
      const b0 = rivBox[4 * i];
      const b1 = rivBox[4 * i + 1];
      if (b1 < latMin || b0 > latMax) continue;
      let shift = NaN;
      for (const s of [0, -2 * Math.PI, 2 * Math.PI]) {
        if (rivBox[4 * i + 2] + s <= lonMax && rivBox[4 * i + 3] + s >= lonMin) {
          shift = s;
          break;
        }
      }
      if (Number.isNaN(shift)) continue;
      reach.resolve(i);
      const qaW = Math.max(0, rivQ[i] - reach.wetDrop[i]);
      const qbW = Math.max(0, rivQ[i + 1] - reach.wetDrop[i]);
      const shown = draw && qbW >= thrL && reach.wetCut[i] !== 0;
      if (!carve && !shown) continue;
      // A tributary's last stretch: where are the trunk's channel and valley?
      const jn = i === s1 - 2 ? rivJoin[ch] : -1;
      let trunkW = 0;
      let trunkV = 0;
      trunkN = 0;
      if (jn >= 0) {
        reach.resolve(jn);
        const qT = rivQ[jn] - reach.wetDrop[jn];
        const trunkShown = draw && reach.wetCut[jn] !== 0 && rivQ[jn + 1] - reach.wetDrop[jn] >= thrL;
        const trunkCarved = carve && rivQ[jn + 1] >= thrL;
        if (trunkShown || trunkCarved) trunkCourse(md, f, chainOf[jn], jn, dRad, x0g, y0g, shift);
        if (trunkShown) trunkW = riverHalfWidth(qT, bankOf(jn, qT), thr0, L, mPerPx);
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
      // A tributary ends where it first touches the trunk's channel: curving
      // into the trunk's flow, it would otherwise run alongside it (three
      // channels side by side) or cross it before reaching the junction.
      let contact = -1;
      if (trunkN >= 2) {
        const reachW = Math.max(trunkW, 1);
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
      if (carve) {
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
        }
      }
      if (!shown || nPts < 2) continue;
      // A channel wide enough to show its bed is drawn whole even when it runs dry, only paler.
      const dash = months >= 11 || L < 4 || riverHalfWidth(qbW, bankOf(i + 1, qbW), thr0, L, mPerPx) >= 1.5 ? 0 : months >= 4 ? 1 : 2;
      const on = dash === 1 ? 7 : 2.2;
      const period = dash === 1 ? 11 : 6;
      const alpha = months >= 11 ? 1 : months >= 4 ? 0.85 : 0.6;
      const bankA = bankOf(i, qaW);
      const bankB = bankOf(i + 1, qbW);
      // Over its last couple of trunk widths a tributary opens out a little into
      // a mouth, rounding off the junction.
      const flare = Math.max(2, (4 * trunkW) / 2.5);
      const widthAt = (s: number): number => {
        const u = Math.min(1, s / (nS - 1));
        const w = riverHalfWidth(qaW + (qbW - qaW) * u, bankA + (bankB - bankA) * u, thr0, L, mPerPx);
        if (contact < 0) return w;
        const mouth = Math.min(trunkW, 1.6 * w);
        return mouth > w ? w + (mouth - w) * smooth01(1 - (contact - s) / flare) : w;
      };
      let lw = widthAt(0);
      let arc = 0;
      for (let s = 1; s < nPts; s++) {
        const w = widthAt(s);
        const seg = Math.hypot(mp[2 * s] - mp[2 * s - 2], mp[2 * s + 1] - mp[2 * s - 1]);
        if (dash === 0 || (arc + seg / 2) % period < on) stampLine(mp[2 * s - 2], mp[2 * s - 1], mp[2 * s], mp[2 * s + 1], lw, w, alpha, Wg, Hg);
        arc += seg;
        lw = w;
      }
    }
  }
  if (streamX0 <= streamX1) mergeStream(Wg);
}

function stampLine(x0: number, y0: number, x1: number, y1: number, h0: number, h1: number, alpha: number, Wg: number, Hg: number): void {
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
      if (v > COVER[p]) COVER[p] = v;
    }
  }
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
  rasterRivers(md, f, L, x0g, y0g, Wg, Hg, dRad, showRivers, carveOn);

  const { x: mx, y: my, z: mz, gStart, gCell, tri } = md;
  const { elev, grad, amp, mtn, ocean, lake, lakeLevel, lakeKind, reef, color, dryness, ice, iceCover, iceMelt, shelf, scour, fjord } = f;
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

  // Domain warp for categorical boundaries, evaluated on an 8-px lattice.
  const warpOct = needCat && hPx >= 32 ? Math.min(4, Math.floor(Math.log2(hPx / 32)) + 1) : 0;
  const WB = 8;
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
      const isLand = swO === 0 ? true : swL === 0 ? false : chi > 0;
      if (isLand) {
        S = seL / swL;
        const dry = landSurface(noise, px, py, pz, h, R, A, Dn, CARVE[p], S, sIce / swL, sCov / swL, sCov > 0 ? sIce / sCov : 0, sFjord / sw, sLake / sw, swO / sw, K);
        let e0 = surfE;
        flags = surfFlags;
        if (!dry) {
          cls = SEA;
          e = e0;
        } else {
          if (swO > 0) {
            // Coastal land rises from the shoreline; it stays above sea level
            // near the sea (inland depressions may lie below it).
            const r = smooth01(chi / 0.3);
            e0 = 1 + (e0 - 1) * r;
            const floor = 0.5 - 2000 * (1 - smooth01(swO / sw / 0.05));
            if (e0 < floor) e0 = floor;
          }
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
        let e0 = S + (0.6 * A + 12) * Dn;
        if (swL > 0) e0 = -1 + (e0 + 1) * smooth01(-chi / 0.3);
        e = Math.min(e0, -0.5);
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
        let cat = cls === SEA ? (bestO >= 0 ? bestO : bestL) : bestL >= 0 ? bestL : bestO;
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
              const v = (1 + 0.05 * Dn) / swL;
              r = cr * v;
              g = cg * v;
              b = cb * v;
              const tp = sa / swL - (LAPSE * (e - S)) / 1000;
              // Green ribbons where rivers run through dry country (if summers are warm enough for shrubs).
              const rip = RIPARIAN[p] * smooth01((sDry / swL - 0.2) / 0.5) * smooth01((tp - 6) / 8) * 0.75;
              if (rip > 0) {
                r += (70 - r) * rip;
                g += (104 - g) * rip;
                b += (50 - b) * rip;
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
        if (a > 0) {
          r += (riverCol[0] - r) * a;
          g += (riverCol[1] - g) * a;
          b += (riverCol[2] - b) * a;
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
