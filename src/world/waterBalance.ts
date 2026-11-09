import { DEG, smoothstep } from '../core/math';
import type { SphereMesh } from '../core/mesh';
import { DAYS_PER_MONTH, declination, insolation, MONTHS } from './climate';

/**
 * Monthly land-surface water balance (Thornthwaite–Mather style bucket model):
 * potential evapotranspiration from radiation and temperature (Hargreaves),
 * snow that falls and melts by positive degree-days, a soil store that
 * evaporates and drains, and a groundwater store that sustains baseflow
 * through dry seasons. Runs a few spin-up years so the stores reach a steady
 * annual cycle.
 */

/** Daily-mean radiation (W/m²) → evaporation-equivalent depth (mm/day). */
const W_TO_MM = 86400 / 2.45e6;
/** Snowmelt per positive degree-day (mm water equivalent). */
export const DDF_SNOW = 4;
/** Bare glacier ice melts faster: it is darker than snow. */
export const DDF_ICE = 8;
/** Day-to-day spread of temperature around the monthly mean (°C). */
export const SIGMA_T = 4.5;
/** Snowpack beyond this (mm w.e.) flows away as glacier ice. */
const SNOW_CAP = 2000;
/** Fraction of groundwater released as baseflow each month. */
const BASEFLOW = 1 - Math.exp(-0.35);
const SPIN_UP_YEARS = 2;

export interface WaterBalance {
  /** Monthly values (mm), index c*12+m: potential and actual evapotranspiration, runoff, snow and ice melt. */
  pet: Float32Array;
  aet: Float32Array;
  runoff: Float32Array;
  melt: Float32Array;
  /** Snowpack at the end of each month (mm water equivalent). */
  snow: Float32Array;
  /** Annual totals (mm). */
  petYear: Float32Array;
  aetYear: Float32Array;
  runoffYear: Float32Array;
  meltYear: Float32Array;
  /** Mean soil moisture as a fraction of its capacity. */
  soil: Float32Array;
  /** Mean groundwater storage (mm). */
  ground: Float32Array;
  /** Snow that survives the whole year (glaciers, ice caps). */
  glacier: Uint8Array;
  snowMonths: Uint8Array;
}

function erf(x: number): number {
  // Abramowitz & Stegun 7.1.26
  const s = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}
export const normCdf = (x: number): number => 0.5 * (1 + erf(x / Math.SQRT2));
const normPdf = (x: number): number => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);

/** Expected positive degree-days in a month of mean temperature t (°C). */
export function positiveDegreeDays(t: number): number {
  const z = t / SIGMA_T;
  return DAYS_PER_MONTH * (SIGMA_T * normPdf(z) + t * normCdf(z));
}

/**
 * Hargreaves reference evapotranspiration (mm/month) from daily-mean
 * top-of-atmosphere radiation `ra` (W/m²), temperature and diurnal range.
 * It fades out below freezing, where frozen ground and ice-covered water give
 * up little more than a trickle of sublimation.
 */
export function hargreaves(ra: number, t: number, dtr: number): number {
  if (t <= -8) return 0;
  return 0.0023 * ra * W_TO_MM * (t + 17.8) * Math.sqrt(dtr) * DAYS_PER_MONTH * smoothstep(-8, 2, t);
}

/**
 * Diurnal temperature range (°C), which drives Hargreaves evaporation: wide in
 * dry continental interiors, narrow over humid or maritime land.
 */
export function diurnalRange(ra: number, precipMonth: number, maritime: number): number {
  const arid = Math.max(0, Math.min(1, 1 - precipMonth / 80));
  return (4 + 5 * (1 - maritime) + 9 * arid) * (0.6 + 0.4 * Math.min(1, ra / 350));
}

/** Daily-mean top-of-atmosphere radiation by latitude (0.25° rows) and month. */
function radiationTable(tiltDeg: number): { rows: number; ra: Float32Array } {
  const rows = 721;
  const ra = new Float32Array(rows * MONTHS);
  for (let m = 0; m < MONTHS; m++) {
    const decl = declination(m, tiltDeg * DEG);
    for (let i = 0; i < rows; i++) ra[i * MONTHS + m] = insolation(-Math.PI / 2 + (i * Math.PI) / (rows - 1), decl);
  }
  return { rows, ra };
}

export function computeWaterBalance(
  mesh: SphereMesh,
  temp: Float32Array,
  precip: Float32Array,
  ocean: Uint8Array,
  slope: Float32Array,
  maritime: Float32Array,
  tiltDeg: number,
  /** Share of each cell under glacier ice: there snow turns to ice and flows away, and summer melts bare ice. */
  iceCover: Float32Array | null = null,
): WaterBalance {
  const n = mesh.n;
  const N = n * MONTHS;
  const pet = new Float32Array(N);
  const aet = new Float32Array(N);
  const runoff = new Float32Array(N);
  const melt = new Float32Array(N);
  const snow = new Float32Array(N);
  const petYear = new Float32Array(n);
  const aetYear = new Float32Array(n);
  const runoffYear = new Float32Array(n);
  const meltYear = new Float32Array(n);
  const soil = new Float32Array(n);
  const ground = new Float32Array(n);
  const glacier = new Uint8Array(n);
  const snowMonths = new Uint8Array(n);
  const { rows, ra: raTab } = radiationTable(tiltDeg);
  const pe = new Float64Array(MONTHS);
  const snowFrac = new Float64Array(MONTHS);
  const pdd = new Float64Array(MONTHS);
  const gRunoff = new Float64Array(MONTHS);
  const gMelt = new Float64Array(MONTHS);
  const gSnow = new Float64Array(MONTHS);

  for (let c = 0; c < n; c++) {
    if (ocean[c]) continue;
    const off = c * MONTHS;
    const fr = ((mesh.lat[c] + Math.PI / 2) / Math.PI) * (rows - 1);
    const i0 = Math.min(rows - 2, Math.floor(fr));
    const ft = fr - i0;
    for (let m = 0; m < MONTHS; m++) {
      const t = temp[off + m];
      const ra = raTab[i0 * MONTHS + m] * (1 - ft) + raTab[(i0 + 1) * MONTHS + m] * ft;
      pe[m] = hargreaves(ra, t, diurnalRange(ra, precip[off + m], maritime[c]));
      snowFrac[m] = normCdf((1 - t) / SIGMA_T);
      pdd[m] = positiveDegreeDays(t);
    }
    const iced = iceCover ? iceCover[c] : 0;
    if (iced > 0) {
      // Glacier surface: no soil. Snow beyond the cap turns to ice and flows
      // away; summer melts the snow and then bare ice without limit, since
      // the glacier keeps supplying it from above.
      let S = 0;
      for (let y = 0; y <= SPIN_UP_YEARS; y++) {
        for (let m = 0; m < MONTHS; m++) {
          const p = precip[off + m];
          const snowfall = p * snowFrac[m];
          S += snowfall;
          const snowMelt = Math.min(S, DDF_SNOW * pdd[m]);
          S -= snowMelt;
          const iceMelt = DDF_ICE * (pdd[m] - snowMelt / DDF_SNOW);
          if (S > SNOW_CAP) S = SNOW_CAP;
          if (y < SPIN_UP_YEARS) continue;
          gRunoff[m] = p - snowfall + snowMelt + iceMelt;
          gMelt[m] = snowMelt + iceMelt;
          gSnow[m] = S;
        }
      }
      if (iced >= 1) {
        for (let m = 0; m < MONTHS; m++) {
          const k = off + m;
          pet[k] = pe[m];
          runoff[k] = gRunoff[m];
          melt[k] = gMelt[m];
          snow[k] = gSnow[m];
          snowMonths[c]++;
          petYear[c] += pet[k];
          runoffYear[c] += runoff[k];
          meltYear[c] += melt[k];
        }
        soil[c] = 1;
        glacier[c] = 1;
        continue;
      }
    }
    // Thin, fast-draining soils on steep ground; deep ones on plains.
    const s = slope[c];
    const wmax = 60 + 160 * Math.exp(-s / 0.012);
    const steep = smoothstep(0.003, 0.03, s);
    const quickShare = 0.35 + 0.4 * steep;
    let S = 0;
    let W = 0.5 * wmax;
    let G = 0;
    let soilSum = 0;
    let groundSum = 0;
    let minSnow = Infinity;
    for (let y = 0; y <= SPIN_UP_YEARS; y++) {
      const record = y === SPIN_UP_YEARS;
      for (let m = 0; m < MONTHS; m++) {
        const p = precip[off + m];
        const t = temp[off + m];
        const snowfall = p * snowFrac[m];
        S += snowfall;
        const mlt = Math.min(S, DDF_SNOW * pdd[m]);
        S -= mlt;
        let ice = 0;
        if (S > SNOW_CAP) {
          ice = S - SNOW_CAP;
          S = SNOW_CAP;
        }
        const input = p - snowfall + mlt;
        // Frozen or steep ground sheds part of the water before it soaks in.
        const frozen = Math.max(0, Math.min(1, -t / 5));
        const direct = input * (0.04 + 0.12 * steep + 0.4 * frozen);
        const infil = input - direct;
        let et: number;
        if (infil >= pe[m]) {
          et = pe[m];
          W += infil - pe[m];
        } else {
          const dried = W * Math.exp(-(pe[m] - infil) / wmax);
          et = infil + (W - dried);
          W = dried;
        }
        let surplus = 0;
        if (W > wmax) {
          surplus = W - wmax;
          W = wmax;
        }
        const quick = quickShare * surplus;
        G += surplus - quick;
        const base = G * BASEFLOW;
        G -= base;
        if (!record) continue;
        const k = off + m;
        pet[k] = pe[m];
        aet[k] = et;
        runoff[k] = direct + quick + base + ice;
        melt[k] = mlt + ice;
        snow[k] = S;
        soilSum += W / wmax;
        groundSum += G;
        if (S > 1) snowMonths[c]++;
        minSnow = Math.min(minSnow, S);
      }
    }
    if (iced > 0) {
      // A glacier tongue that ends in this cell covers part of it; that part runs as glacier.
      for (let m = 0; m < MONTHS; m++) {
        const k = off + m;
        aet[k] *= 1 - iced;
        runoff[k] += iced * (gRunoff[m] - runoff[k]);
        melt[k] += iced * (gMelt[m] - melt[k]);
        snow[k] += iced * (gSnow[m] - snow[k]);
      }
    }
    for (let m = 0; m < MONTHS; m++) {
      petYear[c] += pet[off + m];
      aetYear[c] += aet[off + m];
      runoffYear[c] += runoff[off + m];
      meltYear[c] += melt[off + m];
    }
    soil[c] = soilSum / MONTHS;
    ground[c] = groundSum / MONTHS;
    glacier[c] = minSnow > 50 || iced >= 0.5 ? 1 : 0;
  }
  return { pet, aet, runoff, melt, snow, petYear, aetYear, runoffYear, meltYear, soil, ground, glacier, snowMonths };
}
