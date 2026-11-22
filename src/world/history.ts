import { DEG, smoothstep } from '../core/math';
import type { SphereMesh } from '../core/mesh';
import { computeBiomes } from './biomes';
import { type Climate, computeClimate, declination, insolation, LAPSE, MONTHS, type Orbit, orbitFactor } from './climate';
import { ICE_AGE_SEA_LEVEL, ICE_ISOSTASY, iceAgeCooling, simulateIce } from './glaciers';
import { computeHydrology, LakeKind, oceanMask, slopeScale } from './hydrology';
import { computeWaterBalance } from './waterBalance';
import type { World } from './world';

/**
 * From the last glacial maximum to today. The world's ice-age state (its ice
 * sheets, seas 120 m lower) and its present are both known; the history steps
 * between them on the climate mesh, driven by the slow changes of the orbit,
 * the greenhouse and the ocean:
 *
 * - Orbit: precession walked perihelion from northern winter (as today) to
 *   northern summer about 11 ka and back, and obliquity peaked about 9.5 ka,
 *   so northern summers got ~8% more sunshine than today's. The climate model
 *   meets that sunshine itself: hot continental summers pull the monsoon rains
 *   poleward, greening deserts without any region being singled out.
 * - Greenhouse and ice: CO₂ rose from 190 to 280 ppm, and the shrinking ice
 *   sheets reflected less sunshine. Together they carry the ice age's cooling
 *   (3 °C in the tropics to 12 °C at the poles) away; each slice keeps the
 *   share given by the CO₂ forcing and the ice still standing.
 * - Ocean: the subpolar warm drift was shut down in the ice age, came back in
 *   the Bølling–Allerød warming (14.7 ka) and faltered again in the Younger
 *   Dryas (12.9–11.7 ka), when meltwater freshened the subpolar seas (a
 *   prescribed curve: the model has no overturning circulation to stall).
 *
 * The ice sheets step from the ice age's through each slice's climate, each
 * slice's equilibrium reached from the last (so sheets that keep themselves
 * cold outlast the climate that built them), thinning at most ~0.35 m a year
 * as real ice sheets do. Sea level follows the ice volume; the land the ice
 * pressed down rebounds over a few thousand years after it goes, so seas
 * flood into the hollows the ice left and today's coasts there are raised
 * beaches. Each slice's lakes and climate zones come from its own water
 * balance on the climate mesh; the slice on show is built at full detail.
 */

/** Time slices (ka, thousands of years ago), finest through the deglaciation's fastest change. */
export const SLICE_KA: readonly number[] = [21, 20, 19, 18, 17, 16, 15, 14.5, 14, 13.5, 13, 12.5, 12, 11.5, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
/** Times the climate model is run for (today's climate is the world's own); slices between are interpolated. */
const KEY_KA: readonly number[] = [21, 18, 15, 12, 9, 6, 3];
/** Fastest an ice sheet thins (m per kyr): even a collapsing sheet loses only metres a year. */
const MAX_THINNING = 350;
/** Relaxation time (kyr) of the mantle under a removed ice load. */
const REBOUND_KYR = 4;
/** Share of the ice age's drying (the glacial climate's 0.8 rain factor) at full glacial index. */
const GLACIAL_DRYING = 0.2;
/**
 * Vegetation feedback on the monsoons (Charney 1975; Claussen & Gayler 1997):
 * where stronger summer sunshine brings more rain to subtropical drylands,
 * plants spread, darken the ground and give the rain back to the air, pulling
 * the monsoon further in. Without it climate models green the Sahara only a
 * little. The extra rain is this many times the climate model's own increase,
 * over drylands (under DRYLAND_MM a year today) within ~35° of the equator.
 */
const VEGETATION_FEEDBACK = 2;
const DRYLAND_MM = 700;
/** Ice-age and present CO₂ (ppm). */
const CO2_GLACIAL = 190;
const CO2_PRESENT = 280;

export interface Forcing {
  ka: number;
  orbit: Orbit;
  /** Atmospheric CO₂ (ppm). */
  co2: number;
  /** Strength of the subpolar warm drift, 0 (shut down) to 1 (as today). */
  drift: number;
  /** Mid-July top-of-atmosphere sunshine at 65°N (W/m², daily mean). */
  july65N: number;
}

export interface SliceLake {
  areaKm2: number;
  kind: LakeKind;
  /** Held up by an ice sheet: its way out runs onto the ice. */
  dammed: boolean;
  closed: boolean;
  lat: number;
  lon: number;
  cells: number[];
}

export interface Slice {
  ka: number;
  forcing: Forcing;
  /** 0–1: the share of the ice age's greenhouse and ice-albedo cooling still in force. */
  glacial: number;
  /** Sea level (m, relative to today's), anchored to −120 m at the glacial maximum. */
  seaLevel: number;
  /** What the ice volume alone gives, from the ocean area (m). */
  seaLevelRaw: number;
  /** Ice on land beyond today's (km³ on the climate mesh). */
  iceVolume: number;
  /** Climate mesh: ice thickness (m). */
  ice: Float32Array;
  /** Climate mesh: 0–1, how much of the ice age's ice (beyond today's) still stands. */
  iceLeft: Float32Array;
  /** Climate mesh: how much lower the land stands than today (m) because its rebound from earlier ice isn't done. */
  depression: Float32Array;
  /** Climate mesh: share of each cell that is land (ice-covered or not). */
  land: Float32Array;
  /** Climate mesh: Köppen class and biome under this slice's climate. */
  koppen: Uint8Array;
  biome: Uint8Array;
  /** Climate mesh: lake of each cell (-1 if none), and the lakes. */
  lakeOf: Int32Array;
  lakes: SliceLake[];
  /** Share of the planet that is land. */
  landFraction: number;
  /** Ice area beyond today's (km²). */
  iceAreaKm2: number;
  /** Global mean sea-level air temperature against today's (°C). */
  dT: number;
  /** Water running off the land of closed basins (km³ a year), and the area of their lakes (km²). */
  closedRunoff: number;
  closedLakeKm2: number;
}

export interface HistoryEvent {
  ka: number;
  text: string;
}

export interface History {
  slices: Slice[];
  events: HistoryEvent[];
  /** The climate model's runs (climate mesh), today's (the world's own) last. */
  keys: { ka: number; climate: Climate }[];
  /** Ice-age cooling pattern (°C, climate mesh) the glacial index scales. */
  coolPattern: Float32Array;
  /** Latitude (°) of each climate-mesh cell. */
  lat: Float32Array;
  timeMs: number;
}

// ------------------------------------------------------------------ forcing

/** Obliquity (°) back to ~25 ka: the 41-kyr cycle, highest ~9.5 ka (24.2°), today 23.44°. */
export function obliquityAt(ka: number): number {
  return 23.34 + 0.892 * Math.cos((2 * Math.PI * (ka - 9.5)) / 41);
}

/** Eccentricity: ~0.019 through the deglaciation, 0.0167 today. */
export function eccentricityAt(ka: number): number {
  return 0.0167 + 0.0027 * smoothstep(0, 10, ka);
}

/** Day of the year of perihelion: ~3 January today, earlier in the calendar going back (~21.7 kyr cycle). */
export function perihelionDayAt(ka: number): number {
  const d = (3 - (365.25 * ka) / 21.7) % 365.25;
  return d < 0 ? d + 365.25 : d;
}

/** Atmospheric CO₂ (ppm), from the Antarctic ice cores (smoothed). */
export function co2At(ka: number): number {
  const pts: readonly (readonly [number, number])[] = [
    [21, 190], [18, 190], [17, 196], [14.7, 230], [13, 238], [11.7, 244], [10.5, 262], [8, 260], [6, 263], [0, 280],
  ];
  if (ka >= pts[0][0]) return pts[0][1];
  for (let i = 1; i < pts.length; i++) {
    const [a, va] = pts[i - 1];
    const [b, vb] = pts[i];
    if (ka >= b) return vb + ((va - vb) * (ka - b)) / (a - b);
  }
  return CO2_PRESENT;
}

/** Strength of the subpolar warm drift: off in the ice age and the Younger Dryas. */
export function driftAt(ka: number): number {
  const on = 1 - smoothstep(14.6, 14.75, ka);
  const youngerDryas = smoothstep(11.6, 11.75, ka) * (1 - smoothstep(12.85, 12.95, ka));
  return on * (1 - youngerDryas);
}

export function forcingAt(ka: number, tiltDeg: number): Forcing {
  const dTiltDeg = obliquityAt(ka) - obliquityAt(0);
  const orbit: Orbit = { dTiltDeg, ecc: eccentricityAt(ka), perihelionDay: perihelionDayAt(ka) };
  const july = 6.5;
  const july65N = insolation(65 * DEG, declination(july, (tiltDeg + dTiltDeg) * DEG)) * orbitFactor(((july + 0.5) * 365.25) / 12, orbit);
  return { ka, orbit, co2: co2At(ka), drift: driftAt(ka), july65N };
}

/**
 * Share (0–1) of the ice age's ice beyond today's that still stood on Earth,
 * from the global sea-level record (Lambeck et al. 2014): the meltwater
 * pulses about 14.5 and 11.4 ka, the last ice sheets gone about 7 ka. It sets
 * how much ice-albedo cooling the world as a whole feels; each region's ice
 * still answers its own climate.
 */
export function globalIceAt(ka: number): number {
  const pts: readonly (readonly [number, number])[] = [
    [21, 1], [19, 0.97], [16.5, 0.9], [14.6, 0.8], [14.3, 0.68], [13, 0.6], [11.6, 0.5], [11.3, 0.42], [10, 0.33], [9, 0.25], [8, 0.13], [7, 0.05], [6, 0.02], [0, 0],
  ];
  if (ka >= pts[0][0]) return 1;
  for (let i = 1; i < pts.length; i++) {
    const [a, va] = pts[i - 1];
    const [b, vb] = pts[i];
    if (ka >= b) return vb + ((va - vb) * (ka - b)) / (a - b);
  }
  return 0;
}

/** 0–1 share of the ice age's CO₂ forcing still in force (logarithmic in CO₂). */
const co2Index = (co2: number): number => Math.max(0, Math.min(1, Math.log(CO2_PRESENT / co2) / Math.log(CO2_PRESENT / CO2_GLACIAL)));

// ------------------------------------------------------------------ helpers

const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);

/** Area-weighted mean of a fine-mesh field over each climate-mesh cell. */
function aggregate(world: World, f: ArrayLike<number>): Float32Array {
  const n = world.coarse.n;
  const s = new Float64Array(n);
  const a = new Float64Array(n);
  const near = world.f2c.nearest;
  const area = world.fine.area;
  for (let c = 0; c < f.length; c++) {
    const k = near[c];
    s[k] += area[c] * f[c];
    a[k] += area[c];
  }
  const out = new Float32Array(n);
  for (let k = 0; k < n; k++) out[k] = a[k] > 0 ? s[k] / a[k] : f[world.c2fNearest[k]];
  return out;
}

/** A climate-mesh field at a fine cell, from its climate triangle. */
export function sampleCoarse(world: World, f: ArrayLike<number>, c: number): number {
  const { idx, w } = world.f2c;
  return f[idx[3 * c]] * w[3 * c] + f[idx[3 * c + 1]] * w[3 * c + 1] + f[idx[3 * c + 2]] * w[3 * c + 2];
}

function lerpArr<T extends Float32Array | Float64Array>(a: T, b: T, f: number): T {
  const out = new (a.constructor as { new (n: number): T })(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + (b[i] - a[i]) * f;
  return out;
}

/** The climate between two of the model's runs (f: weight of b). */
function interpClimate(a: Climate, b: Climate, f: number): Climate {
  if (f <= 0) return a;
  if (f >= 1) return b;
  return {
    n: a.n,
    tempSL: lerpArr(a.tempSL, b.tempSL, f),
    precip: lerpArr(a.precip, b.precip, f),
    windE: lerpArr(a.windE, b.windE, f),
    windN: lerpArr(a.windN, b.windN, f),
    maritime: lerpArr(a.maritime, b.maritime, f),
    current: lerpArr(a.current, b.current, f),
    drift: lerpArr(a.drift, b.drift, f),
    flowE: lerpArr(a.flowE, b.flowE, f),
    flowN: lerpArr(a.flowN, b.flowN, f),
    upwell: lerpArr(a.upwell, b.upwell, f),
    itcz: lerpArr(a.itcz, b.itcz, f),
    zonalT: lerpArr(a.zonalT, b.zonalT, f),
  };
}

/** Steepest slope to a neighbour on a mesh, scaled to the reference detail as the world's own. */
function meshSlope(mesh: SphereMesh, elev: Float32Array, radiusKm: number): Float32Array {
  const { n, nbrStart, nbr, nbrDist } = mesh;
  const R = radiusKm * 1000;
  const scale = slopeScale(n);
  const out = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    let s = 0;
    for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) s = Math.max(s, Math.abs(elev[nbr[k]] - elev[c]) / (nbrDist[k] * R));
    out[c] = s * scale;
  }
  return out;
}

const fmtKm2 = (a: number): string =>
  a >= 1e6 ? `${(a / 1e6).toFixed(a >= 1e7 ? 0 : 1)} million km²` : `${(Math.round(a / 1000) * 1000).toLocaleString('en-US')} km²`;
const fmtKa = (ka: number): string => (ka >= 1 ? `${Number(ka.toFixed(1))} ka` : `${Math.round(ka * 1000)} years ago`);
const latLonText = (lat: number, lon: number): string =>
  `${Math.abs(lat).toFixed(0)}°${lat >= 0 ? 'N' : 'S'} ${Math.abs(lon).toFixed(0)}°${lon >= 0 ? 'E' : 'W'}`;

function centroid(mesh: SphereMesh, cells: readonly number[]): { lat: number; lon: number } {
  let x = 0;
  let y = 0;
  let z = 0;
  for (const c of cells) {
    x += mesh.x[c] * mesh.area[c];
    y += mesh.y[c] * mesh.area[c];
    z += mesh.z[c] * mesh.area[c];
  }
  const r = Math.hypot(x, y, z) || 1;
  return { lat: Math.asin(z / r) / DEG, lon: Math.atan2(y, x) / DEG };
}

// ------------------------------------------------------------------ the history

/** Glacial-index cooling (°C) and rain factor of a slice. */
export function sliceAnomaly(h: History, s: Slice, cl: Climate): { cool: Float32Array; precipFactor: number } {
  const n = cl.n;
  const cool = new Float32Array(n);
  const off = 1 - s.forcing.drift;
  for (let k = 0; k < n; k++) cool[k] = h.coolPattern[k] * s.glacial + cl.drift[k] * off;
  return { cool, precipFactor: 1 - GLACIAL_DRYING * s.glacial };
}

/** The model's climate at time ka (climate mesh), before the glacial-index cooling: between the two nearest runs. */
export function baseClimate(h: History, ka: number): Climate {
  const keys = h.keys;
  for (let i = 1; i < keys.length; i++) {
    const a = keys[i - 1];
    const b = keys[i];
    if (ka <= a.ka && ka >= b.ka) return interpClimate(a.climate, b.climate, (a.ka - ka) / (a.ka - b.ka));
  }
  return ka > keys[0].ka ? keys[0].climate : keys[keys.length - 1].climate;
}

/**
 * A slice's climate on the climate mesh: the model's (orbit, coastlines, ice
 * sheets) less the glacial-index cooling and drift, with the glacial drying.
 */
export function sliceClimate(h: History, s: Slice): Climate {
  const base = baseClimate(h, s.ka);
  if (s.ka === 0) return base;
  const today = h.keys[h.keys.length - 1].climate;
  const { cool, precipFactor } = sliceAnomaly(h, s, base);
  const tempSL = new Float32Array(base.tempSL.length);
  const precip = new Float32Array(base.precip.length);
  for (let k = 0; k < base.n; k++) {
    let p0 = 0;
    let p1 = 0;
    for (let m = 0; m < MONTHS; m++) {
      p0 += today.precip[k * MONTHS + m];
      p1 += base.precip[k * MONTHS + m];
    }
    const latDeg = Math.abs(h.lat[k]);
    let veg = 1;
    if (p1 > p0 && p0 < DRYLAND_MM && latDeg < 35) {
      const dry = 1 - smoothstep(0.6 * DRYLAND_MM, DRYLAND_MM, p0);
      const tropic = 1 - smoothstep(28, 35, latDeg);
      veg = 1 + Math.min(2, VEGETATION_FEEDBACK * (p1 / Math.max(1, p0) - 1)) * dry * tropic;
    }
    for (let m = 0; m < MONTHS; m++) {
      tempSL[k * MONTHS + m] = base.tempSL[k * MONTHS + m] - cool[k];
      precip[k * MONTHS + m] = base.precip[k * MONTHS + m] * precipFactor * veg;
    }
  }
  return { ...base, tempSL, precip };
}

export function buildHistory(world: World, progress: (stage: string, frac: number) => void = () => {}): History {
  const t0 = performance.now();
  const { coarse: C, fine, params } = world;
  const n = C.n;
  const R = params.radiusKm;
  const kmA = (k: number): number => C.area[k] * R * R;
  const near = world.f2c.nearest;

  // The climate mesh's view of the world.
  const bedC = aggregate(world, world.bed);
  const pastC = aggregate(world, world.iceAge.thick);
  const presC = aggregate(world, world.ice.thick);
  const roughC = aggregate(world, world.relief.rough);
  const base = world.relief.elev;
  const presentOcean = oceanMask(fine, base);
  const landC0 = new Float32Array(n);
  {
    const a = new Float64Array(n);
    const l = new Float64Array(n);
    for (let c = 0; c < fine.n; c++) {
      a[near[c]] += fine.area[c];
      if (!presentOcean[c]) l[near[c]] += fine.area[c];
    }
    for (let k = 0; k < n; k++) landC0[k] = a[k] > 0 ? l[k] / a[k] : presentOcean[world.c2fNearest[k]] ? 0 : 1;
  }
  let oceanKm2 = 0;
  for (let k = 0; k < n; k++) oceanKm2 += kmA(k) * (1 - landC0[k]);
  const volume = (ice: Float32Array): number => {
    let v = 0;
    for (let k = 0; k < n; k++) v += (ice[k] * kmA(k)) / 1000;
    return v;
  };

  const vPres = volume(presC);
  const coolPattern = iceAgeCooling(C);

  /** Climate inputs (mean land height above the sea, land share) for a coastline and ice state. */
  const climateInputs = (sl: number, ice: Float32Array): { elevC: Float32Array; landC: Float32Array } => {
    const e = new Float32Array(fine.n);
    for (let c = 0; c < fine.n; c++) {
      const k = near[c];
      const extra = ice[k] - presC[k];
      e[c] = extra > 30 ? Math.max(1, base[c] - sl + (1 - ICE_ISOSTASY) * extra) : base[c] - sl;
    }
    const ocean = oceanMask(fine, e);
    const sE = new Float64Array(n);
    const sL = new Float64Array(n);
    const sA = new Float64Array(n);
    for (let c = 0; c < fine.n; c++) {
      const k = near[c];
      const a = fine.area[c];
      sA[k] += a;
      sE[k] += a * Math.max(0, e[c]);
      if (!ocean[c]) sL[k] += a;
    }
    const elevC = new Float32Array(n);
    const landC = new Float32Array(n);
    for (let k = 0; k < n; k++) {
      elevC[k] = sA[k] > 0 ? sE[k] / sA[k] : Math.max(0, e[world.c2fNearest[k]]);
      landC[k] = sA[k] > 0 ? sL[k] / sA[k] : ocean[world.c2fNearest[k]] ? 0 : 1;
    }
    return { elevC, landC };
  };

  // The climate model's runs, each with the ice and coastline of the slice before it.
  const keys: { ka: number; climate: Climate }[] = [];
  const keyDone = new Set<number>();
  const runKey = (ka: number, sl: number, ice: Float32Array): void => {
    if (keyDone.has(ka)) return;
    keyDone.add(ka);
    const { elevC, landC } = climateInputs(sl, ice);
    keys.push({ ka, climate: computeClimate(C, elevC, landC, params, forcingAt(ka, params.axialTiltDeg).orbit) });
    keys.sort((p, q) => q.ka - p.ka);
  };
  const h: History = { slices: [], events: [], keys, coolPattern, lat: Float32Array.from(C.lat, (v) => v / DEG), timeMs: 0 };

  // ---- ice sheets, sea level and the land's rebound, slice by slice
  // The climate mesh's own present-day ice, grown from the ice age's under
  // today's climate as the world's own was: its cells are ~130 km across, so
  // its sheets hang on longer than at full detail. Each cell's retreat is
  // measured from the ice age's ice toward this, and applied to the ice the
  // ice age left beyond today's at full detail ("excess"). Where the climate
  // mesh keeps nearly all its ice-age ice even today, the cell follows the
  // world's deglaciation as a whole.
  const eqPres = simulateIce(C, bedC, oceanMask(C, bedC), { tempSL: world.climate.tempSL, precip: world.climate.precip, cool: null, precipFactor: 1 }, 0, R, false, pastC, 0, 0).thick;
  // ...and its own glacial-maximum ice under the ice age's climate, which the steps start from.
  runKey(SLICE_KA[0], ICE_AGE_SEA_LEVEL, pastC);
  const lgmClimate = keys.find((k) => k.ka === SLICE_KA[0])!.climate;
  const lgmCool = Float32Array.from(coolPattern, (v, k) => v + lgmClimate.drift[k]);
  const eqLgm = simulateIce(
    C, bedC, oceanMask(C, Float32Array.from(bedC, (b) => b - ICE_AGE_SEA_LEVEL)),
    { tempSL: lgmClimate.tempSL, precip: lgmClimate.precip, cool: lgmCool, precipFactor: 1 - GLACIAL_DRYING },
    ICE_AGE_SEA_LEVEL, R, false, pastC, 0, 0,
  ).thick;
  const excess = new Float32Array(n);
  for (let k = 0; k < n; k++) excess[k] = Math.max(0, pastC[k] - presC[k]);
  const vExcess = volume(excess);
  // The model's deglaciation, held within 1.5 times the share of ice the
  // global record still had (plus a margin fading toward today) so that it
  // finishes in the late Holocene as Earth's did, whatever sheets the climate
  // mesh cannot melt.
  const envelope = (ka: number): number => clamp01(1.5 * globalIceAt(ka) + 0.1 * smoothstep(0, 6, ka));
  const leftOf = (ice: Float32Array, ka: number): Float32Array => {
    const out = new Float32Array(n);
    const g = globalIceAt(ka);
    const cap = envelope(ka);
    for (let k = 0; k < n; k++) {
      if (excess[k] <= 0) continue;
      const span = eqLgm[k] - eqPres[k];
      const model = clamp01((ice[k] - eqPres[k]) / Math.max(50, span));
      const w = clamp01(span / (0.5 * excess[k]));
      out[k] = ka >= SLICE_KA[0] ? 1 : ka <= 0 ? 0 : Math.min(cap, model * w + g * (1 - w));
    }
    return out;
  };
  const loadOf = (left: Float32Array): Float32Array => Float32Array.from(presC, (p, k) => p + left[k] * excess[k]);
  const seaOf = (left: Float32Array): number => {
    if (vExcess <= 0) return 0;
    let v = 0;
    for (let k = 0; k < n; k++) v += (left[k] * excess[k] * kmA(k)) / 1000;
    return (ICE_AGE_SEA_LEVEL * v) / vExcess;
  };
  const raw: { ka: number; forcing: Forcing; glacial: number; ice: Float32Array; D: Float32Array; v: number; left: Float32Array }[] = [];
  let ice = Float32Array.from(eqLgm);
  let left = leftOf(ice, SLICE_KA[0]);
  let load = loadOf(left);
  // The lithosphere bends rather than sinking cell by cell: an ice load presses
  // down a moat ~200 km beyond its margin too, where meltwater pooled.
  const flex = (f: Float32Array): Float32Array => {
    let a = f;
    for (let pass = 0; pass < 2; pass++) {
      const b = new Float32Array(n);
      for (let k = 0; k < n; k++) {
        let s = 0;
        let m = 0;
        for (let e = C.nbrStart[k]; e < C.nbrStart[k + 1]; e++) {
          s += a[C.nbr[e]];
          m++;
        }
        b[k] = 0.5 * a[k] + (0.5 * s) / Math.max(1, m);
      }
      a = b;
    }
    return a;
  };
  let D = flex(load).map((v) => ICE_ISOSTASY * v);
  let sl = ICE_AGE_SEA_LEVEL;
  keys.push({ ka: 0, climate: world.climate });
  // Through the deglaciation no ice sheet grows past its glacial-maximum size (re-advances stayed inside it).
  const ceiling = new Float32Array(n);
  for (let k = 0; k < n; k++) ceiling[k] = Math.max(eqLgm[k], presC[k]);
  for (let i = 0; i < SLICE_KA.length; i++) {
    const ka = SLICE_KA[i];
    progress('History: ice sheets and seas', 0.05 + (0.6 * i) / SLICE_KA.length);
    const forcing = forcingAt(ka, params.axialTiltDeg);
    let glacial = 1;
    if (i > 0) {
      const dt = SLICE_KA[i - 1] - ka;
      // The younger climate run bracketing this slice, made with the latest ice and coasts.
      const younger = KEY_KA.find((kk) => kk <= ka);
      if (younger !== undefined) runKey(younger, sl, load);
      glacial = ka === 0 ? 0 : 0.5 * co2Index(forcing.co2) + 0.5 * globalIceAt(ka);
      const s = { ka, forcing, glacial } as Slice;
      const cl = baseClimate(h, ka);
      const { cool, precipFactor } = sliceAnomaly(h, s, cl);
      const ocean = oceanMask(C, Float32Array.from(bedC, (b) => b - sl));
      const eq = simulateIce(C, bedC, ocean, { tempSL: cl.tempSL, precip: cl.precip, cool, precipFactor }, sl, R, false, ice, 8, 0);
      const next = new Float32Array(n);
      const cap = MAX_THINNING * dt;
      for (let k = 0; k < n; k++) next[k] = Math.min(ceiling[k], Math.max(eq.thick[k], ice[k] - cap));
      ice = next;
      left = leftOf(ice, ka);
      load = loadOf(left);
      // The mantle relaxes toward the new load.
      const relax = Math.exp(-dt / REBOUND_KYR);
      const D2 = new Float32Array(n);
      const spread = flex(load);
      for (let k = 0; k < n; k++) {
        const eqD = ICE_ISOSTASY * spread[k];
        D2[k] = eqD + (D[k] - eqD) * relax;
      }
      D = D2;
      sl = seaOf(left);
    }
    raw.push({ ka, forcing, glacial, ice: load, D, v: volume(load), left });
  }

  // ---- today's closed basins at full detail: their lakes are followed through the slices
  // from each slice's runoff into the basin and the open-water evaporation at its lake.
  const basins: { km2: number; cellsC: Map<number, number>; lakeNow: number; sink: number; est: number[]; lat: number; lon: number }[] = [];
  {
    const hydroNow = world.hydro;
    const index = new Map<number, number>();
    const lakeSeen: Set<number>[] = [];
    const fineKm2 = (c: number): number => fine.area[c] * R * R;
    for (let c = 0; c < fine.n; c++) {
      if (!hydroNow.endorheic[c] || world.ocean[c]) continue;
      const d = hydroNow.drainage[c];
      let b = index.get(d);
      if (b === undefined) {
        b = basins.length;
        index.set(d, b);
        basins.push({ km2: 0, cellsC: new Map(), lakeNow: 0, sink: near[d >= 0 ? d : c], est: [], lat: fine.lat[d >= 0 ? d : c] / DEG, lon: fine.lon[d >= 0 ? d : c] / DEG });
        lakeSeen.push(new Set());
      }
      const B = basins[b];
      B.km2 += fineKm2(c);
      B.cellsC.set(near[c], (B.cellsC.get(near[c]) ?? 0) + fineKm2(c));
      const id = hydroNow.lake[c];
      if (id >= 0 && !lakeSeen[b].has(id)) {
        lakeSeen[b].add(id);
        B.lakeNow += hydroNow.lakes[id].areaKm2;
        B.sink = near[c];
      }
    }
    for (let b = basins.length - 1; b >= 0; b--) if (basins[b].km2 < 20e3) basins.splice(b, 1);
  }

  // ---- sea level from the ice volume (anchored at −120 m), and the rebound still to come
  const end = raw[raw.length - 1];
  const nonEqEnd = new Float32Array(n);
  for (let k = 0; k < n; k++) nonEqEnd[k] = end.D[k] - ICE_ISOSTASY * end.ice[k];
  for (let i = 0; i < raw.length; i++) {
    const r = raw[i];
    progress('History: lakes and climate zones', 0.65 + (0.3 * i) / raw.length);
    const seaLevel = seaOf(r.left);
    const seaLevelRaw = (-(r.v - vPres) * (1 / ICE_PER_WATER_RATIO) * 1000) / oceanKm2;
    const depression = new Float32Array(n);
    const iceLeft = r.left;
    if (r.ka !== 0) for (let k = 0; k < n; k++) depression[k] = r.D[k] - ICE_ISOSTASY * r.ice[k] - nonEqEnd[k];
    // Land: the fine mesh's ground above this sea level (ice-covered ground is land).
    const land = new Float32Array(n);
    {
      const a = new Float64Array(n);
      const l = new Float64Array(n);
      for (let c = 0; c < fine.n; c++) {
        const k = near[c];
        a[k] += fine.area[c];
        const iced = r.ice[k] - presC[k] > 30 || world.ice.thick[c] > 0;
        if (iced || world.elev[c] - depression[k] - seaLevel > 0) l[k] += fine.area[c];
      }
      for (let k = 0; k < n; k++) land[k] = a[k] > 0 ? l[k] / a[k] : landC0[k];
    }
    let landA = 0;
    let iceA = 0;
    for (let k = 0; k < n; k++) {
      landA += land[k] * kmA(k);
      if (r.ice[k] - presC[k] > 30) iceA += kmA(k);
    }
    // Climate zones and lakes from the slice's own water balance.
    const s: Slice = {
      ka: r.ka,
      forcing: r.forcing,
      glacial: r.glacial,
      seaLevel,
      seaLevelRaw,
      iceVolume: r.v - vPres,
      ice: r.ice,
      iceLeft,
      depression,
      land,
      koppen: new Uint8Array(n),
      biome: new Uint8Array(n),
      lakeOf: new Int32Array(n).fill(-1),
      lakes: [],
      landFraction: landA / (4 * Math.PI * R * R),
      iceAreaKm2: iceA,
      dT: 0,
      closedRunoff: 0,
      closedLakeKm2: 0,
    };
    const cl = r.ka === 0 ? world.climate : sliceClimate(h, s);
    const elevC = new Float32Array(n);
    const oceanIn = new Float32Array(n);
    for (let k = 0; k < n; k++) {
      elevC[k] = bedC[k] + (1 - ICE_ISOSTASY) * r.ice[k] - depression[k] - seaLevel;
      if (land[k] >= 0.5 && elevC[k] < 1) elevC[k] = 1;
      if (land[k] < 0.5 && elevC[k] >= 0) elevC[k] = -1;
      oceanIn[k] = elevC[k];
    }
    const oceanC = oceanMask(C, oceanIn);
    const tempC = new Float32Array(n * MONTHS);
    let dT = 0;
    let wT = 0;
    const today = world.climate;
    for (let k = 0; k < n; k++) {
      const drop = oceanC[k] ? 0 : (LAPSE * Math.max(0, elevC[k])) / 1000;
      for (let m = 0; m < MONTHS; m++) {
        tempC[k * MONTHS + m] = cl.tempSL[k * MONTHS + m] - drop;
        dT += (cl.tempSL[k * MONTHS + m] - today.tempSL[k * MONTHS + m]) * C.area[k];
      }
      wT += C.area[k] * MONTHS;
    }
    s.dT = dT / wT;
    const iceCover = new Float32Array(n);
    for (let k = 0; k < n; k++) iceCover[k] = r.ice[k] > 30 ? 1 : 0;
    const slopeC = meshSlope(C, elevC, R);
    const dTilt = r.forcing.orbit.dTiltDeg;
    const wb = computeWaterBalance(C, tempC, cl.precip, oceanC, slopeC, cl.maritime, params.axialTiltDeg + dTilt, iceCover);
    const hy = computeHydrology(C, elevC, oceanC, wb, cl.precip, tempC, slopeC, roughC, R, undefined, false, undefined, { iceDammedLakes: true });
    const bio = computeBiomes(C, elevC, tempC, cl.precip, oceanC, hy, wb);
    s.koppen = bio.koppen;
    s.biome = bio.biome;
    s.lakeOf = hy.lake;
    // Today's closed basins: inflow (km³ a year) over open-water loss (mm a year) gives the lake they can hold.
    for (const B of basins) {
      let inflow = 0;
      for (const [k, a] of B.cellsC) inflow += (wb.runoffYear[k] * a) / 1e6;
      let pYear = 0;
      for (let m = 0; m < MONTHS; m++) pYear += cl.precip[B.sink * MONTHS + m];
      const loss = Math.max(100, 1.1 * wb.petYear[B.sink] - pYear);
      B.est.push((inflow * 1e6) / loss);
      s.closedRunoff += inflow;
    }
    const cellsOf: number[][] = hy.lakes.map(() => []);
    for (let k = 0; k < n; k++) if (hy.lake[k] >= 0) cellsOf[hy.lake[k]].push(k);
    s.lakes = hy.lakes.map((lk, j) => {
      const cells = cellsOf[j];
      let dammed = false;
      for (const k of cells) {
        for (let e = C.nbrStart[k]; e < C.nbrStart[k + 1] && !dammed; e++) dammed = r.ice[C.nbr[e]] > 30;
        if (dammed) break;
      }
      const { lat, lon } = cells.length ? centroid(C, cells) : { lat: 0, lon: 0 };
      return { areaKm2: lk.areaKm2, kind: lk.kind, dammed, closed: lk.outflow <= 0, lat, lon, cells };
    });

    h.slices.push(s);
  }

  // Each basin's lake through time: today's (full detail) scaled by the water it can hold.
  for (let i = 0; i < h.slices.length; i++) {
    for (const B of basins) h.slices[i].closedLakeKm2 += basinLake(B, i);
  }
  progress('History: events', 0.97);
  h.events = historyEvents(world, h, presC);
  for (const B of basins) {
    let peak = 0;
    let at = -1;
    for (let i = 0; i < h.slices.length - 1; i++) {
      const a = basinLake(B, i);
      if (a > peak) {
        peak = a;
        at = i;
      }
    }
    if (at < 0 || peak < 20e3 || peak < 1.8 * B.lakeNow) continue;
    const today = B.lakeNow > 1000 ? `today ${fmtKm2(B.lakeNow)}` : 'today dry or a salt flat';
    h.events.push({ ka: h.slices[at].ka, text: `Pluvial highstand: the closed lake near ${latLonText(B.lat, B.lon)} spread to about ${fmtKm2(peak)} (${today})` });
  }
  h.events.sort((p, q) => q.ka - p.ka);
  h.timeMs = performance.now() - t0;
  return h;
}

/**
 * A closed basin's lake at slice i (km²): today's (full detail) scaled by the
 * water the basin can hold then against now; a basin dry today gets the
 * excess. At most 40% of the basin drowns (beyond that it would spill).
 */
function basinLake(B: { km2: number; lakeNow: number; est: number[] }, i: number): number {
  const now = B.est[B.est.length - 1];
  const est = B.est[i];
  const a = B.lakeNow > 1000 && now > 0 ? (B.lakeNow * est) / now : Math.max(0, est - now);
  return Math.min(0.4 * B.km2, a);
}

/** Ice per unit of meltwater: 1000/917; its inverse turns ice volume into water. */
const ICE_PER_WATER_RATIO = 1000 / 917;

// ------------------------------------------------------------------ events

function components(mesh: SphereMesh, inSet: (k: number) => boolean): Int32Array {
  const comp = new Int32Array(mesh.n).fill(-1);
  const stack: number[] = [];
  let id = 0;
  for (let k0 = 0; k0 < mesh.n; k0++) {
    if (comp[k0] >= 0 || !inSet(k0)) continue;
    comp[k0] = id;
    stack.push(k0);
    while (stack.length) {
      const k = stack.pop()!;
      for (let e = mesh.nbrStart[k]; e < mesh.nbrStart[k + 1]; e++) {
        const m = mesh.nbr[e];
        if (comp[m] < 0 && inSet(m)) {
          comp[m] = id;
          stack.push(m);
        }
      }
    }
    id++;
  }
  return comp;
}

function historyEvents(world: World, h: History, presC: Float32Array): HistoryEvent[] {
  const C = world.coarse;
  const n = C.n;
  const R = world.params.radiusKm;
  const kmA = (k: number): number => C.area[k] * R * R;
  const S = h.slices;
  const last = S[S.length - 1];
  const events: HistoryEvent[] = [];
  const first = S[0];
  const shelf = Math.max(0, first.landFraction - last.landFraction) * 4 * Math.PI * R * R;
  events.push({
    ka: first.ka,
    text: `Last glacial maximum: ice sheets covered ${fmtKm2(first.iceAreaKm2)} beyond today's ice, the seas stood ${Math.round(-first.seaLevel)} m lower and ${fmtKm2(shelf)} of continental shelf lay dry; the air was ${Math.abs(first.dT).toFixed(1)} °C colder on average`,
  });

  // The seas' fastest rise, and any meltwater pulse (faster than 15 m per thousand years).
  {
    let bi = -1;
    let br = 0;
    for (let i = 1; i < S.length; i++) {
      const r = (S[i].seaLevel - S[i - 1].seaLevel) / (S[i - 1].ka - S[i].ka);
      if (r > br) {
        br = r;
        bi = i;
      }
    }
    if (bi > 0 && br < 15) {
      events.push({ ka: S[bi - 1].ka, text: `The seas rose fastest, ${Math.round(br)} m per thousand years, as the great ice sheets broke up (${fmtKa(S[bi - 1].ka)}–${fmtKa(S[bi].ka)})` });
    }
  }
  for (let i = 1; i < S.length; i++) {
    const dt = S[i - 1].ka - S[i].ka;
    const rate = (S[i].seaLevel - S[i - 1].seaLevel) / dt;
    if (rate < 15) continue;
    let j = i;
    while (j + 1 < S.length && (S[j + 1].seaLevel - S[j].seaLevel) / (S[j].ka - S[j + 1].ka) >= 15) j++;
    const rise = S[j].seaLevel - S[i - 1].seaLevel;
    const years = Math.round((S[i - 1].ka - S[j].ka) * 1000);
    events.push({ ka: S[i - 1].ka, text: `Meltwater pulse: the seas rose ${Math.round(rise)} m in ${years.toLocaleString('en-US')} years as the ice sheets collapsed` });
    i = j;
  }

  // The warm drift's comings and goings.
  events.push({ ka: 14.7, text: 'Bølling–Allerød warming: the subpolar warm drift returned and the lands around the northern seas warmed abruptly' });
  events.push({ ka: 12.9, text: 'Younger Dryas: meltwater freshened the subpolar seas and the warm drift faltered, plunging the lands around them back toward ice-age cold' });
  events.push({ ka: 11.7, text: 'The Younger Dryas ended within decades: the warm drift came back and the Holocene began' });

  // Land bridges: today's landmasses joined by dry shelf, and when the sea parted them.
  const landAt = (s: Slice) => (k: number) => s.land[k] >= 0.5;
  const today = components(C, landAt(last));
  const todayArea = new Map<number, number>();
  for (let k = 0; k < n; k++) if (today[k] >= 0) todayArea.set(today[k], (todayArea.get(today[k]) ?? 0) + kmA(k));
  const plateName = (k: number): string => {
    const p = world.geo.plate[world.c2fNearest[k]];
    return world.geo.plates[p]?.name ?? 'an unnamed plate';
  };
  const landName = (id: number): string => {
    let best = -1;
    let bestA = -1;
    const counts = new Map<number, number>();
    for (let k = 0; k < n; k++) {
      if (today[k] !== id) continue;
      const p = world.geo.plate[world.c2fNearest[k]];
      const a = (counts.get(p) ?? 0) + kmA(k);
      counts.set(p, a);
      if (a > bestA) {
        bestA = a;
        best = k;
      }
    }
    const a = todayArea.get(id) ?? 0;
    if (a >= 3e6) return `the ${plateName(best)} landmass`;
    const cells: number[] = [];
    for (let k = 0; k < n; k++) if (today[k] === id) cells.push(k);
    const { lat, lon } = centroid(C, cells);
    return `${a >= 300e3 ? 'the large island' : 'the island'} at ${latLonText(lat, lon)}`;
  };
  const MIN_LAND = 50e3;
  const joined = new Map<string, { a: number; b: number; youngest: number; at: { lat: number; lon: number } }>();
  for (let i = 0; i < S.length - 1; i++) {
    const comp = components(C, landAt(S[i]));
    const members = new Map<number, Set<number>>();
    for (let k = 0; k < n; k++) {
      if (comp[k] < 0 || today[k] < 0 || (todayArea.get(today[k]) ?? 0) < MIN_LAND) continue;
      let set = members.get(comp[k]);
      if (!set) members.set(comp[k], (set = new Set()));
      set.add(today[k]);
    }
    for (const [cid, set] of members) {
      if (set.size < 2) continue;
      const ids = [...set].sort((p, q) => (todayArea.get(q) ?? 0) - (todayArea.get(p) ?? 0)).slice(0, 4);
      for (let x = 0; x < ids.length; x++) {
        for (let y = x + 1; y < ids.length; y++) {
          const key = `${ids[x]}-${ids[y]}`;
          const prev = joined.get(key);
          if (prev) {
            prev.youngest = Math.min(prev.youngest, S[i].ka);
            continue;
          }
          // The bridge: the exposed shelf cell, joined to both over this
          // slice's land, farthest from neither (hops through the landmass).
          const hops = (id: number): Int32Array => {
            const d = new Int32Array(n).fill(-1);
            let front: number[] = [];
            for (let k = 0; k < n; k++) if (today[k] === id && comp[k] === cid) {
              d[k] = 0;
              front.push(k);
            }
            for (let r = 1; front.length; r++) {
              const next: number[] = [];
              for (const k of front) {
                for (let e = C.nbrStart[k]; e < C.nbrStart[k + 1]; e++) {
                  const m = C.nbr[e];
                  if (d[m] < 0 && comp[m] === cid && today[m] < 0) {
                    d[m] = r;
                    next.push(m);
                  }
                }
              }
              front = next;
            }
            return d;
          };
          const da = hops(ids[x]);
          const db = hops(ids[y]);
          let bestK = -1;
          let bestD = Infinity;
          for (let k = 0; k < n; k++) {
            if (comp[k] !== cid || today[k] >= 0 || da[k] < 0 || db[k] < 0) continue;
            const d = Math.max(da[k], db[k]) + 0.01 * (da[k] + db[k]);
            if (d < bestD) {
              bestD = d;
              bestK = k;
            }
          }
          if (bestK < 0) continue;
          const at = bestK >= 0 ? { lat: C.lat[bestK] / DEG, lon: C.lon[bestK] / DEG } : { lat: 0, lon: 0 };
          joined.set(key, { a: ids[x], b: ids[y], youngest: S[i].ka, at });
        }
      }
    }
  }
  for (const j of joined.values()) {
    const idx = S.findIndex((s) => s.ka === j.youngest);
    const drown = idx >= 0 && idx + 1 < S.length ? (S[idx].ka + S[idx + 1].ka) / 2 : j.youngest;
    events.push({ ka: drown, text: `The land bridge joining ${landName(j.a)} and ${landName(j.b)} near ${latLonText(j.at.lat, j.at.lon)} drowned as the seas rose` });
  }

  // Ice-dammed lakes that drained when their dams retreated.
  for (let i = 0; i + 1 < S.length; i++) {
    const younger = S[i + 1];
    for (const lk of S[i].lakes) {
      if (!lk.dammed || lk.areaKm2 < 20e3) continue;
      let kept = 0;
      let all = 0;
      for (const k of lk.cells) {
        all += kmA(k);
        if (younger.lakeOf[k] >= 0) kept += kmA(k);
      }
      if (all <= 0 || kept / all > 0.4) continue;
      events.push({ ka: (S[i].ka + younger.ka) / 2, text: `An ice-dammed lake of ${fmtKm2(lk.areaKm2)} near ${latLonText(lk.lat, lk.lon)} burst out as its ice dam retreated, a catastrophic outburst flood` });
    }
  }

  // Desert wet phases: stretches of today's desert that were steppe,
  // grassland or savanna. Each region is followed through the slices by its
  // cells; the cause told from when and where it greened.
  const desert = (code: number): boolean => code === 3 || code === 4;
  const green = (code: number): boolean => code < 3 || (code >= 5 && code < 28);
  const regions: { cells: Set<number>; peak: number; peakKa: number; from: number; to: number; lat: number; lon: number }[] = [];
  for (let i = 0; i < S.length - 1; i++) {
    const s = S[i];
    const comp = components(C, (k) => desert(last.koppen[k]) && green(s.koppen[k]) && s.land[k] >= 0.5 && last.land[k] >= 0.5);
    const groups = new Map<number, number[]>();
    for (let k = 0; k < n; k++) if (comp[k] >= 0) (groups.get(comp[k]) ?? groups.set(comp[k], []).get(comp[k])!).push(k);
    for (const cells of groups.values()) {
      let a = 0;
      for (const k of cells) a += kmA(k);
      if (a < 150e3) continue;
      let r = regions.find((g) => cells.some((k) => g.cells.has(k)));
      if (!r) {
        const { lat, lon } = centroid(C, cells);
        r = { cells: new Set(), peak: 0, peakKa: s.ka, from: s.ka, to: s.ka, lat, lon };
        regions.push(r);
      }
      for (const k of cells) r.cells.add(k);
      r.from = Math.max(r.from, s.ka);
      r.to = Math.min(r.to, s.ka);
      if (a > r.peak) {
        r.peak = a;
        r.peakKa = s.ka;
        const c = centroid(C, cells);
        r.lat = c.lat;
        r.lon = c.lon;
      }
    }
  }
  regions.sort((p, q) => q.peak - p.peak);
  for (const r of regions.slice(0, 3)) {
    if (r.peak < 300e3) continue;
    // Precession favoured northern summers ~14–5 ka (southern ones today and at the glacial maximum).
    const north = r.lat > 0 && r.lat < 35 && r.peakKa <= 14 && r.peakKa >= 5;
    const south = r.lat < 0 && r.lat > -35 && (r.peakKa <= 3 || r.peakKa >= 19);
    const cause = north || south
      ? 'stronger summer sunshine pushed the monsoon rains poleward'
      : r.peakKa >= 14 ? 'the cooler glacial climate lost less water to evaporation' : 'the rain belts shifted';
    events.push({
      ka: r.from,
      text: `Desert wet phase (${fmtKa(r.from)}–${fmtKa(r.to)}): ${cause}, and up to ${fmtKm2(r.peak)} of today's desert near ${latLonText(r.lat, r.lon)} was steppe, grassland or savanna (wettest ~${fmtKa(r.peakKa)})`,
    });
  }

  // The great ice sheets' last remnants.
  const lgmIce = components(C, (k) => first.ice[k] - presC[k] > 30);
  const sheetArea = new Map<number, number>();
  for (let k = 0; k < n; k++) if (lgmIce[k] >= 0) sheetArea.set(lgmIce[k], (sheetArea.get(lgmIce[k]) ?? 0) + kmA(k));
  for (const [id, a] of sheetArea) {
    if (a < 1e6) continue;
    const cells: number[] = [];
    for (let k = 0; k < n; k++) if (lgmIce[k] === id) cells.push(k);
    let gone = -1;
    for (let i = 1; i < S.length; i++) {
      let left = 0;
      for (const k of cells) if (S[i].ice[k] - presC[k] > 30) left += kmA(k);
      if (left < 0.05 * a) {
        gone = (S[i - 1].ka + S[i].ka) / 2;
        break;
      }
    }
    if (gone < 0) continue;
    const { lat, lon } = centroid(C, cells);
    events.push({ ka: gone, text: `The ice-age ice sheet of ${fmtKm2(a)} centred near ${latLonText(lat, lon)} melted away` });
  }

  events.sort((p, q) => q.ka - p.ka);
  return events;
}
