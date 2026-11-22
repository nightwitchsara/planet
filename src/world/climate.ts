import { MinHeap } from '../core/heap';
import { DEG, smoothstep } from '../core/math';
import { nearestCellLatLon, type SphereMesh } from '../core/mesh';
import type { WorldParams } from './types';

export const MONTHS = 12;
export const DAYS_PER_MONTH = 365.25 / 12;
const S0 = 1361;
/** Outgoing longwave radiation A + B·T (W/m², T in °C) — North et al. */
const A_OLR = 203.3;
const B_OLR = 2.09;
/** Meridional heat diffusion of the zonal energy-balance model (W/m²/K). */
const D_EBM = 0.649;
/**
 * The Hadley cells and tropical ocean currents carry heat poleward far more
 * efficiently than mid-latitude eddies (Lindzen & Farrell 1977), which keeps
 * the tropics nearly uniform in temperature: the diffusion is stronger by this
 * factor near the equator, fading with this width (°).
 */
const HADLEY_TRANSPORT = 1.4;
const HADLEY_WIDTH = 16;
/** Extra planetary albedo of the deep convective clouds over the ITCZ, fading with this width (°). */
const TROPICAL_CLOUD_ALBEDO = 0.02;
const TROPICAL_CLOUD_WIDTH = 15;
/**
 * Coastal upwelling: along the eastern sides of the oceans the equatorward
 * winds on the subtropical highs' flanks drive surface water offshore and cold
 * water wells up within a few hundred km of the coast (Humboldt, Benguela,
 * Canary, California), 6–9 °C below the zonal mean together with the gyre's
 * own cold eastern limb.
 */
const UPWELLING_MAX = 7;
const UPWELLING_KM = 230;
/**
 * Sea breezes and alongshore flow mix marine air inland whatever the
 * large-scale wind: e-folding distance (km) and strength at the coast. On
 * upwelling coasts the marine layer is trapped against the shore under the
 * inversion, so the coast is almost wholly maritime (Lima's months span 7 °C).
 */
const COAST_MIX_KM = 120;
const COAST_MIX_BASE = 0.15;
const COAST_MIX_UPWELLING = 0.6;
/**
 * Air over cold water is stable: a shallow marine layer under the subsidence
 * inversion gives fog and stratus but almost no rain (the Atacama, Namib and
 * Baja coasts). It reaches this far inland (km) below this depth (m).
 */
const MARINE_LAYER_KM = 150;
const MARINE_LAYER_DEPTH = 1200;
/**
 * Where the westerlies' storms come ashore, friction slows the air and the
 * coastal ranges lift it: rain multiplies on windward coasts (the Pacific
 * Northwest, Norway, southern Chile get 1,500–3,000+ mm). Extra storm rain at
 * the coast for wind blowing straight onshore, fading inland over this (km).
 */
const COAST_STORM = 3.0;
const COAST_STORM_KM = 220;
/** Ocean evaporation at 20 °C (mm/day) in a 7 m/s wind. */
const EVAP_20 = 3.4;
/** Peak rain rate (per day) of the extratropical storm systems. */
const STORM_RATE = 0.12;
const OMEGA = (2 * Math.PI) / (365.25 * 86400);
/** Heat capacity of a 30 m ocean mixed layer vs. land + atmosphere (J/m²/K). */
const C_OCEAN = 1.2e8;
const C_LAND = 2.0e7;
/** Local radiative + transport damping of seasonal anomalies (W/m²/K). */
const B_EFF = 4.5;
/** Strongest thermal (monsoon) wind perturbation (m/s). */
const MONSOON_MAX = 6;
/** Moisture mixing by extratropical storm systems (m²/s). */
const EDDY_DIFFUSIVITY = 3e6;
/**
 * Latitudes (°) where the eddy mixing sets in and reaches full strength: the
 * storm tracks start at the subtropics' poleward edge and carry most of the
 * water that reaches the mid-latitudes, peaking near 40°.
 */
const EDDY_FROM = 22;
const EDDY_FULL = 42;
/** Poleward wind (m/s) on the moist western flank of a subtropical high (and equatorward on its eastern flank). */
const HIGH_FLANK_WIND = 3;
/** How much the high's western flank weakens subsidence there, and its eastern flank strengthens it. */
const HIGH_FLANK_SUBSIDENCE = 0.7;
/** How far (°) the storm track shifts toward the equator in winter. */
const STORM_WINTER_SHIFT = 7;
export const LAPSE = 6.5; // K per km
const SECTORS = 24;
const LAT_BANDS = 180;

export interface Climate {
  n: number;
  /** Sea-level air temperature (°C), index c*12+m. */
  tempSL: Float32Array;
  /** Precipitation (mm/month), index c*12+m. */
  precip: Float32Array;
  /** Surface wind (m/s), index c*12+m. */
  windE: Float32Array;
  windN: Float32Array;
  /** 1 over open ocean, decaying downwind over land. */
  maritime: Float32Array;
  /** Sea-surface anomaly from wind-driven currents (°C), ocean cells. */
  current: Float32Array;
  /**
   * Warmth (°C) the subpolar warm drift adds to the annual mean, over sea and
   * the land downwind. In the ice age the drift faltered (sea ice reached
   * ~45° in the North Atlantic), so the glacial climate goes without it.
   */
  drift: Float32Array;
  /**
   * Surface ocean current (m/s, eastward and northward) from the same
   * wind-driven gyres as `current`: ocean cells, and coastal land cells
   * carrying their nearest sea's flow so that it interpolates up to the coast.
   */
  flowE: Float32Array;
  flowN: Float32Array;
  /** 0–1 per ocean cell: how strongly coastal upwelling brings cold deep water up. */
  upwell: Float32Array;
  /** ITCZ latitude (degrees) per month and longitude sector. */
  itcz: Float32Array;
  /** Zonal-mean annual temperature from the energy-balance model (1° bands). */
  zonalT: Float64Array;
}

/**
 * Extra albedo of high-latitude land through the sunlit season. Under a low
 * sun the winter's snow lingers deep into summer, and polar summers are
 * cloudy, so land beyond the polar circles absorbs a smaller share of its
 * long days' sunshine. Earth's polar lands peak at 0–8 °C in their warmest
 * month despite 24-hour daylight, while the continental response alone would
 * give a dark polar interior Siberian summers.
 */
const POLAR_LAND_ALBEDO = 0.2;
const POLAR_LAND_FROM = 62;
const POLAR_LAND_FULL = 85;

/** Albedo the seasonal cycle of sunshine meets (low sun and clouds toward the poles). */
export function seasonalAlbedo(lat: number): number {
  const s = Math.sin(lat);
  return 0.32 + 0.1 * s * s;
}

/** The same over land, which keeps its snow into the summer near the poles. */
export function landSeasonalAlbedo(lat: number): number {
  return seasonalAlbedo(lat) + POLAR_LAND_ALBEDO * smoothstep(POLAR_LAND_FROM, POLAR_LAND_FULL, Math.abs(lat) / DEG);
}

/** Daily-mean top-of-atmosphere insolation (W/m²). */
export function insolation(lat: number, decl: number): number {
  const t = -Math.tan(lat) * Math.tan(decl);
  const h0 = t >= 1 ? 0 : t <= -1 ? Math.PI : Math.acos(t);
  return (S0 / Math.PI) * (h0 * Math.sin(lat) * Math.sin(decl) + Math.cos(lat) * Math.cos(decl) * Math.sin(h0));
}

export function declination(month: number, tilt: number): number {
  const day = ((month + 0.5) * 365.25) / 12;
  return -tilt * Math.cos((2 * Math.PI * (day + 10)) / 365.25);
}

/**
 * The orbit at some time in the past, relative to today's Earth: the change in
 * obliquity (°), the eccentricity and the day of the year of perihelion
 * (today ~3 January). Precession walks perihelion through the calendar every
 * ~22 kyr; with it near the northern summer solstice (~11 ka) northern
 * summers got ~8% more sunshine than today's.
 */
export interface Orbit {
  dTiltDeg: number;
  ecc: number;
  perihelionDay: number;
}
const ECC_NOW = 0.0167;
const PERIHELION_NOW = 3;

/** Sunshine on a day of the year (0–365) relative to today's orbit: the inverse square of the Sun's distance. */
export function orbitFactor(day: number, orbit: Orbit | null | undefined): number {
  if (!orbit) return 1;
  const f = (e: number, p: number): number => (1 + e * Math.cos((2 * Math.PI * (day - p)) / 365.25)) ** 2;
  return f(orbit.ecc, orbit.perihelionDay) / f(ECC_NOW, PERIHELION_NOW);
}

const bandLat = (j: number): number => -Math.PI / 2 + ((j + 0.5) * Math.PI) / LAT_BANDS;

function bandOf(lat: number): number {
  return Math.max(0, Math.min(LAT_BANDS - 1, Math.floor(((lat + Math.PI / 2) / Math.PI) * LAT_BANDS)));
}

/** Annual-mean zonal temperature from a Budyko–Sellers EBM with ice-albedo feedback. */
function zonalEBM(meanQ: Float64Array, offset: number): Float64Array {
  const J = LAT_BANDS;
  const dphi = Math.PI / J;
  const T = new Float64Array(J).fill(10);
  const lo = new Float64Array(J);
  const di = new Float64Array(J);
  const up = new Float64Array(J);
  const rhs = new Float64Array(J);
  const cp = new Float64Array(J);
  const dp = new Float64Array(J);
  const diffusivity = (phi: number): number => D_EBM * (1 + HADLEY_TRANSPORT * Math.exp(-((phi / (HADLEY_WIDTH * DEG)) ** 2)));
  for (let iter = 0; iter < 40; iter++) {
    for (let j = 0; j < J; j++) {
      const phi = bandLat(j);
      const s = Math.sin(phi);
      // Planetary albedo rising poleward (low sun, clouds): 0.25 at the equator
      // (with the ITCZ's convective cloud), 0.52 at the poles.
      const free = 0.32 + 0.2 * ((3 * s * s - 1) / 2) + TROPICAL_CLOUD_ALBEDO * Math.exp(-((phi / (TROPICAL_CLOUD_WIDTH * DEG)) ** 2));
      const ice = 1 / (1 + Math.exp((T[j] + 10) / 2));
      const albedo = free + (0.62 - free) * ice;
      const cj = Math.cos(phi);
      const da = diffusivity(phi + dphi / 2);
      const db = diffusivity(phi - dphi / 2);
      const a = Math.max(0, Math.cos(phi + dphi / 2)) / (cj * dphi * dphi);
      const b = Math.max(0, Math.cos(phi - dphi / 2)) / (cj * dphi * dphi);
      lo[j] = j > 0 ? -db * b : 0;
      up[j] = j < J - 1 ? -da * a : 0;
      di[j] = B_OLR + (j < J - 1 ? da * a : 0) + (j > 0 ? db * b : 0);
      rhs[j] = meanQ[j] * (1 - albedo) - (A_OLR - B_OLR * offset);
    }
    // Thomas algorithm
    cp[0] = up[0] / di[0];
    dp[0] = rhs[0] / di[0];
    for (let j = 1; j < J; j++) {
      const m = di[j] - lo[j] * cp[j - 1];
      cp[j] = up[j] / m;
      dp[j] = (rhs[j] - lo[j] * dp[j - 1]) / m;
    }
    T[J - 1] = dp[J - 1];
    for (let j = J - 2; j >= 0; j--) T[j] = dp[j] - cp[j] * T[j + 1];
  }
  return T;
}

/** 0 in a hemisphere's winter month, 1 at the height of its summer (sea-level air lags the sun by about a month). */
const summerOf = (phiDeg: number, month: number): number => {
  const c = Math.cos((2 * Math.PI * (month + 0.5 - 7)) / 12);
  return 0.5 + 0.5 * (phiDeg >= 0 ? c : -c);
};

const bump = (v: number, a: number, peak: number, b: number): number =>
  v <= a || v >= b ? 0 : v < peak ? Math.sin(((v - a) / (peak - a)) * (Math.PI / 2)) : Math.cos(((v - peak) / (b - peak)) * (Math.PI / 2));

/**
 * Sea-surface temperature anomaly from wind-driven gyres, by position within
 * each ocean basin along a latitude circle: warm western boundary currents and
 * cold eastern upwelling in the subtropics, warm drift reaching the eastern
 * side of subpolar basins.
 */
function gyreGrid(mesh: SphereMesh, land: Float32Array): { isLand: Uint8Array; W: number; H: number } {
  const W = 180;
  const H = 90;
  const isLand = new Uint8Array(W * H);
  for (let j = 0; j < H; j++) {
    const la = Math.PI / 2 - ((j + 0.5) * Math.PI) / H;
    for (let i = 0; i < W; i++) {
      const lo = -Math.PI + ((i + 0.5) * 2 * Math.PI) / W;
      isLand[j * W + i] = land[nearestCellLatLon(mesh, la, lo)] > 0.5 ? 1 : 0;
    }
  }
  // Islands narrower than ~8° don't block a gyre: treat them as open water.
  for (let j = 0; j < H; j++) {
    const row = isLand.subarray(j * W, j * W + W);
    let i0 = -1;
    for (let i = 0; i < W; i++) {
      if (!row[i]) continue;
      if (i0 < 0) i0 = i;
      if (i === W - 1 || !row[i + 1]) {
        if (i - i0 + 1 < 4 && !(i0 === 0 && row[W - 1])) row.fill(0, i0, i + 1);
        i0 = -1;
      }
    }
  }
  return { isLand, W, H };
}

/** Blur a gyre-grid field across rows and its neighbors (bins where `keep` is 0 only), then sample it per cell. */
function blurAndSample(mesh: SphereMesh, grid: Float32Array, isLand: Uint8Array, W: number, H: number, oceanOnly: boolean): Float32Array {
  const tmp = new Float32Array(W * H);
  for (let pass = 0; pass < 3; pass++) {
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const p = j * W + i;
        if (oceanOnly && isLand[p]) continue;
        let s = 0;
        let wsum = 0;
        for (let dj = -2; dj <= 2; dj++) {
          const jj = j + dj;
          if (jj < 0 || jj >= H) continue;
          for (let di = -1; di <= 1; di++) {
            const q = jj * W + ((i + di + W) % W);
            if (oceanOnly && isLand[q]) continue;
            s += grid[q];
            wsum++;
          }
        }
        tmp[p] = s / wsum;
      }
    }
    for (let p = 0; p < W * H; p++) if (!oceanOnly || !isLand[p]) grid[p] = tmp[p];
  }
  const out = new Float32Array(mesh.n);
  for (let c = 0; c < mesh.n; c++) {
    const j = Math.max(0, Math.min(H - 1, Math.floor(((Math.PI / 2 - mesh.lat[c]) / Math.PI) * H)));
    const i = Math.max(0, Math.min(W - 1, Math.floor(((mesh.lon[c] + Math.PI) / (2 * Math.PI)) * W)));
    out[c] = grid[j * W + i];
  }
  return out;
}

/**
 * Sea-surface temperature anomaly from wind-driven gyres, by position within
 * each ocean basin along a latitude circle (see currentAnomaly's caller).
 */
function currentAnomaly(mesh: SphereMesh, land: Float32Array, retrograde: boolean): Float32Array {
  const { isLand, W, H } = gyreGrid(mesh, land);
  const anomGrid = new Float32Array(W * H);
  for (let j = 0; j < H; j++) {
    const la = Math.PI / 2 - ((j + 0.5) * Math.PI) / H;
    const latDeg = Math.abs(la / DEG);
    const wSub = bump(latDeg, 8, 27, 46);
    const wPol = bump(latDeg, 40, 57, 74);
    if (wSub === 0 && wPol === 0) continue;
    const row = isLand.subarray(j * W, j * W + W);
    let start = -1;
    for (let i = 0; i < W; i++) {
      if (row[i]) {
        start = i;
        break;
      }
    }
    if (start < 0) continue; // circumpolar ocean: no gyre boundaries
    const kmPerBin = (2 * Math.PI * 6371 * Math.cos(la)) / W;
    for (let k = 0; k < W; k++) {
      const i0 = (start + k) % W;
      if (row[i0] || !row[(i0 - 1 + W) % W]) continue;
      let len = 0;
      while (len < W && !row[(i0 + len) % W]) len++;
      const widthFactor = Math.min(1, (len * kmPerBin) / 4000);
      for (let t = 0; t < len; t++) {
        let u = (t + 0.5) / len;
        if (retrograde) u = 1 - u;
        const a =
          wSub * (4.5 * Math.exp(-u / 0.12) - 5.5 * Math.exp(-(1 - u) / 0.15)) +
          wPol * (11 * Math.exp(-(1 - u) / 0.35) - 4 * Math.exp(-u / 0.12));
        anomGrid[j * W + ((i0 + t) % W)] = a * widthFactor;
      }
    }
  }
  // Currents are broad features: blur across rows (ocean bins only) so that
  // row-to-row differences in coastline geometry don't streak the field.
  const out = blurAndSample(mesh, anomGrid, isLand, W, H, true);
  for (let c = 0; c < mesh.n; c++) if (land[c] > 0.5) out[c] = 0;
  return out;
}

/**
 * Which side of a subtropical high each place lies on, -1 to 1. The highs sit
 * over the eastern parts of the ocean basins: on their western flank warm,
 * moist air flows poleward onto the eastern sides of the continents (Florida,
 * southern China, eastern Australia, southern Brazil: +1), while on their
 * eastern flank dry air sinks and flows toward the equator along the western
 * coasts (the Sahara and Namib, Atacama, Sonora, western Australia: -1).
 * Along each latitude circle: +1 at the western edge of an ocean basin and the
 * eastern coast of a continent, -1 at the eastern edge of a basin and the
 * western coast of a continent. Retrograde rotation mirrors it.
 */
function subtropicalSide(mesh: SphereMesh, land: Float32Array, retrograde: boolean): Float32Array {
  const { isLand, W, H } = gyreGrid(mesh, land);
  const side = new Float32Array(W * H);
  for (let j = 0; j < H; j++) {
    const la = Math.PI / 2 - ((j + 0.5) * Math.PI) / H;
    if (bump(Math.abs(la / DEG), 6, 27, 48) === 0) continue;
    const row = isLand.subarray(j * W, j * W + W);
    let start = -1;
    for (let i = 0; i < W; i++) {
      if (row[i] !== row[(i - 1 + W) % W]) {
        start = i;
        break;
      }
    }
    if (start < 0) continue; // all ocean or all land: no highs anchored to coasts
    const kmPerBin = (2 * Math.PI * 6371 * Math.cos(la)) / W;
    for (let k = 0; k < W; k++) {
      const i0 = (start + k) % W;
      if (row[i0] === row[(i0 - 1 + W) % W]) continue;
      const kind = row[i0];
      let len = 0;
      while (len < W && row[(i0 + len) % W] === kind) len++;
      for (let t = 0; t < len; t++) {
        const u = (t + 0.5) / len;
        let v: number;
        if (kind) {
          // Land: moist flow reaches ~1500 km in from the east coast, dry air ~1000 km in from the west.
          const fromEast = (len - t - 0.5) * kmPerBin;
          const fromWest = (t + 0.5) * kmPerBin;
          v = Math.exp(-fromEast / 1500) - Math.exp(-fromWest / 1000);
        } else {
          v = (1 - 2 * u) * Math.min(1, (len * kmPerBin) / 3000);
        }
        side[j * W + ((i0 + t) % W)] = retrograde ? -v : v;
      }
    }
  }
  return blurAndSample(mesh, side, isLand, W, H, false);
}

/**
 * How close each ocean cell lies to a coast where upwelling can happen, 0–1:
 * the eastern edge of a wide ocean (the western edge with retrograde
 * rotation), decaying offshore over UPWELLING_KM. Worked on a 1° grid along
 * latitude circles so that only coasts with open ocean to their west count.
 */
function upwellingProximity(mesh: SphereMesh, land: Float32Array, retrograde: boolean): Float32Array {
  const W = 360;
  const H = 180;
  const isLand = new Uint8Array(W * H);
  for (let j = 0; j < H; j++) {
    const la = Math.PI / 2 - ((j + 0.5) * Math.PI) / H;
    if (Math.abs(la) > 50 * DEG) continue;
    for (let i = 0; i < W; i++) {
      const lo = -Math.PI + ((i + 0.5) * 2 * Math.PI) / W;
      isLand[j * W + i] = land[nearestCellLatLon(mesh, la, lo)] > 0.5 ? 1 : 0;
    }
    // Islands narrower than ~4° don't hold a coastal current against them.
    const row = isLand.subarray(j * W, j * W + W);
    let i0 = -1;
    for (let i = 0; i < W; i++) {
      if (!row[i]) continue;
      if (i0 < 0) i0 = i;
      if (i === W - 1 || !row[i + 1]) {
        if (i - i0 + 1 < 4 && !(i0 === 0 && row[W - 1])) row.fill(0, i0, i + 1);
        i0 = -1;
      }
    }
  }
  const grid = new Float32Array(W * H);
  for (let j = 0; j < H; j++) {
    const la = Math.PI / 2 - ((j + 0.5) * Math.PI) / H;
    if (Math.abs(la) > 48 * DEG) continue;
    const row = isLand.subarray(j * W, j * W + W);
    let start = -1;
    for (let i = 0; i < W; i++) {
      if (row[i]) {
        start = i;
        break;
      }
    }
    if (start < 0) continue;
    const kmPerBin = (2 * Math.PI * 6371 * Math.cos(la)) / W;
    for (let k = 0; k < W; k++) {
      const i0 = (start + k) % W;
      if (row[i0] || !row[(i0 - 1 + W) % W]) continue;
      let len = 0;
      while (len < W && !row[(i0 + len) % W]) len++;
      // The highs and their equatorward coastal winds need a wide ocean.
      const basin = smoothstep(1200, 3500, len * kmPerBin);
      if (basin <= 0) continue;
      for (let t = 0; t < len; t++) {
        const d = (retrograde ? t + 0.5 : len - t - 0.5) * kmPerBin;
        grid[j * W + ((i0 + t) % W)] = basin * Math.exp(-d / UPWELLING_KM);
      }
    }
  }
  // One light pass across neighbouring rows smooths the coastline's stair steps.
  const sm = new Float32Array(W * H);
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const p = j * W + i;
      if (isLand[p]) continue;
      let s = 0;
      let w = 0;
      for (let dj = -1; dj <= 1; dj++) {
        const jj = j + dj;
        if (jj < 0 || jj >= H) continue;
        for (let di = -1; di <= 1; di++) {
          const q = jj * W + ((i + di + W) % W);
          if (isLand[q]) continue;
          const wt = dj === 0 && di === 0 ? 2 : 1;
          s += wt * grid[q];
          w += wt;
        }
      }
      sm[p] = s / w;
    }
  }
  const out = new Float32Array(mesh.n);
  for (let c = 0; c < mesh.n; c++) {
    if (land[c] > 0.5) continue;
    const fj = ((Math.PI / 2 - mesh.lat[c]) / Math.PI) * H - 0.5;
    const fi = ((mesh.lon[c] + Math.PI) / (2 * Math.PI)) * W - 0.5;
    const j0 = Math.floor(fj);
    const i0 = Math.floor(fi);
    let s = 0;
    let w = 0;
    for (let dj = 0; dj <= 1; dj++) {
      const jj = Math.max(0, Math.min(H - 1, j0 + dj));
      const wj = dj ? fj - j0 : 1 - (fj - j0);
      for (let di = 0; di <= 1; di++) {
        const q = jj * W + ((i0 + di + W) % W);
        if (isLand[q]) continue;
        const wt = wj * (di ? fi - i0 : 1 - (fi - i0));
        s += wt * sm[q];
        w += wt;
      }
    }
    out[c] = w > 1e-6 ? s / w : 0;
  }
  return out;
}

/** Sverdrup transport scale (m²/s per unit of gyreStrength): gyre interiors drift at ~0.04 m/s. */
const GYRE_PSI = 2e5;
/** Width of the boundary currents the gyres' return flow crowds into (km). */
const BOUNDARY_KM = 150;
/** Equatorward coastal jet along upwelling coasts at full strength (m/s). */
const UPWELLING_JET = 0.25;
/** Fastest surface current (m/s): the Gulf Stream and Kuroshio peak near 2. */
const CURRENT_MAX = 2;

/**
 * How much the land's seasonal swing is damped near the equator (fading out by
 * ~32°). Summer monsoon clouds and rain shade and cool tropical land before
 * its peak, and wet soils spend the extra sunshine evaporating water, so land
 * at 10–20° spans ~6–11 °C over the year (the Deccan, the Sahel) rather than
 * the 12–14 °C a dry continent's heat capacity alone would give.
 */
const TROPICAL_LAND_DAMP = 0.35;
/**
 * Showers of the equatorial trough (per day, at full strength): within ~10°
 * of the equator the sun stands high all year and the trades of both
 * hemispheres meet, so warm, humid land keeps raining between the rain belt's
 * passes (the Amazon, Congo and Indonesia get 60–150 mm even in their driest
 * months).
 */
const EQUATORIAL_SHOWERS = 0.06;
/**
 * Share of its evaporative demand that equatorial forest keeps transpiring
 * through the dry season, from water its deep soils stored in the wet months
 * (the Amazon's evapotranspiration barely drops in the dry season). It keeps
 * the air humid, so the rain belt's passes are bridged by local showers.
 */
const EQUATORIAL_TRANSPIRATION = 0.35;

/**
 * Meridional structure of the wind-driven circulation in the northern
 * hemisphere, in units of the subtropical gyre (positive = clockwise): the
 * tropical cell whose flanks are the westward equatorial current and the
 * eastward countercurrent at 5–10°, the subtropical gyre between the trades and
 * the westerlies, and the subpolar gyre beyond them. The southern hemisphere
 * mirrors it.
 */
function gyreStrength(absLatDeg: number): number {
  const a = absLatDeg;
  if (a < 10) return 0.6 * Math.sin((Math.PI * a) / 10);
  if (a < 45) return Math.sin((Math.PI * (a - 10)) / 35);
  if (a < 65) return -0.6 * Math.sin((Math.PI * (a - 45)) / 20);
  return 0;
}

/** d gyreStrength / d|latitude| (per degree). */
function gyreStrengthSlope(absLatDeg: number): number {
  const a = absLatDeg;
  if (a < 10) return ((0.6 * Math.PI) / 10) * Math.cos((Math.PI * a) / 10);
  if (a < 45) return (Math.PI / 35) * Math.cos((Math.PI * (a - 10)) / 35);
  if (a < 65) return ((-0.6 * Math.PI) / 20) * Math.cos((Math.PI * (a - 45)) / 20);
  return 0;
}

/**
 * Surface ocean currents, eastward and northward (m/s), from the wind-driven
 * gyres on a 1° grid along latitude circles. Each ocean basin between two
 * coasts carries a Stommel gyre: a slow Sverdrup drift across its interior and
 * the return flow crowded into a fast current along its western edge (the
 * eastern edge with retrograde rotation), poleward in the subtropics (the Gulf
 * Stream, Kuroshio, Brazil and East Australian currents) and toward the equator
 * in the subpolar gyres (the Labrador and Oyashio currents). The zonal flow
 * comes from the streamfunction's change between rows, so it follows coasts
 * that run east–west. Rows with no coast at all carry a zonal current driven by
 * the wind (eastward under the westerlies, like the Antarctic Circumpolar
 * Current). Upwelling coasts add an equatorward coastal jet.
 */
function currentVelocity(
  mesh: SphereMesh,
  land: Float32Array,
  retrograde: boolean,
  upwell: Float32Array,
  radiusKm: number,
): { flowE: Float32Array; flowN: Float32Array } {
  const W = 360;
  const H = 180;
  const rot = retrograde ? -1 : 1;
  const isLand = new Uint8Array(W * H);
  for (let j = 0; j < H; j++) {
    const la = Math.PI / 2 - ((j + 0.5) * Math.PI) / H;
    for (let i = 0; i < W; i++) {
      const lo = -Math.PI + ((i + 0.5) * 2 * Math.PI) / W;
      isLand[j * W + i] = land[nearestCellLatLon(mesh, la, lo)] > 0.5 ? 1 : 0;
    }
    // Islands narrower than ~8° don't split a gyre (as for its temperature anomaly).
    const row = isLand.subarray(j * W, j * W + W);
    let i0 = -1;
    for (let i = 0; i < W; i++) {
      if (!row[i]) continue;
      if (i0 < 0) i0 = i;
      if (i === W - 1 || !row[i + 1]) {
        if (i - i0 + 1 < 8 && !(i0 === 0 && row[W - 1])) row.fill(0, i0, i + 1);
        i0 = -1;
      }
    }
  }
  const dyM = (radiusKm * 1000 * Math.PI) / H;
  const perDeg = (radiusKm * 1000 * Math.PI) / 180;
  const psi = new Float64Array(W * H);
  const u = new Float32Array(W * H);
  const v = new Float32Array(W * H);
  /** Analytic eastward flow at fixed basin position (rows beside an open row). */
  const uA = new Float32Array(W * H);
  const open = new Uint8Array(H);
  for (let j = 0; j < H; j++) {
    const la = Math.PI / 2 - ((j + 0.5) * Math.PI) / H;
    const latDeg = la / DEG;
    const a = Math.abs(latDeg);
    const Fh = Math.sign(latDeg) * gyreStrength(a);
    const dF = gyreStrengthSlope(a);
    const row = isLand.subarray(j * W, j * W + W);
    let start = -1;
    for (let i = 0; i < W; i++) {
      if (row[i]) {
        start = i;
        break;
      }
    }
    if (start < 0) {
      // No coast on this circle: the wind drives a zonal current right round it.
      open[j] = 1;
      const { phiP } = circulationLat(latDeg, 3);
      const zonal = 0.04 * rot * cellWind(phiP)[0];
      for (let i = 0; i < W; i++) u[j * W + i] = zonal;
      continue;
    }
    const kmPerBin = (2 * Math.PI * radiusKm * Math.cos(la)) / W;
    for (let k = 0; k < W; k++) {
      const i0 = (start + k) % W;
      if (row[i0] || !row[(i0 - 1 + W) % W]) continue;
      let len = 0;
      while (len < W && !row[(i0 + len) % W]) len++;
      const Lkm = len * kmPerBin;
      const d = Math.min(0.3, BOUNDARY_KM / Math.max(1, Lkm));
      for (let t = 0; t < len; t++) {
        const p = j * W + ((i0 + t) % W);
        const x = (t + 0.5) / len;
        const xr = retrograde ? 1 - x : x;
        const ex = Math.exp(-xr / d);
        const X = (1 - ex) * (1 - xr);
        const dX = (ex / d) * (1 - xr) - (1 - ex);
        psi[p] = rot * GYRE_PSI * Fh * (Lkm / 5000) * X;
        v[p] = (GYRE_PSI * Fh * dX) / 5e6;
        uA[p] = (-rot * GYRE_PSI * (Lkm / 5000) * X * dF) / perDeg;
      }
    }
  }
  // Islands and capes split a basin in some rows and not in their neighbours:
  // blur the streamfunction and the meridional flow across rows (ocean bins of
  // rows with coasts) so that the zonal flow doesn't streak into jets.
  {
    const KW = [1, 2, 3, 4, 3, 2, 1];
    const tp = new Float64Array(W * H);
    const tv = new Float32Array(W * H);
    for (let pass = 0; pass < 3; pass++) {
      for (let j = 0; j < H; j++) {
        if (open[j]) continue;
        for (let i = 0; i < W; i++) {
          const p = j * W + i;
          if (isLand[p]) continue;
          let s0 = 0;
          let s1 = 0;
          let ws = 0;
          for (let d = -3; d <= 3; d++) {
            const jj = j + d;
            if (jj < 0 || jj >= H || open[jj]) continue;
            const q = jj * W + i;
            if (isLand[q]) continue;
            const wt = KW[d + 3];
            s0 += wt * psi[q];
            s1 += wt * v[q];
            ws += wt;
          }
          tp[p] = s0 / ws;
          tv[p] = s1 / ws;
        }
      }
      for (let p = 0; p < W * H; p++) {
        if (isLand[p] || open[Math.floor(p / W)]) continue;
        psi[p] = tp[p];
        v[p] = tv[p];
      }
    }
  }
  // Zonal flow from the streamfunction's change between rows (zero on land), or
  // at fixed basin position where a neighboring row is open ocean.
  for (let j = 0; j < H; j++) {
    if (open[j]) continue;
    for (let i = 0; i < W; i++) {
      const p = j * W + i;
      if (isLand[p]) continue;
      if (j === 0 || j === H - 1 || open[j - 1] || open[j + 1]) {
        u[p] = uA[p];
        continue;
      }
      u[p] = -(psi[p - W] - psi[p + W]) / (2 * dyM);
    }
  }
  // One light pass across neighbouring bins smooths the coastline's stair steps.
  const su = new Float32Array(W * H);
  const sv = new Float32Array(W * H);
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const p = j * W + i;
      if (isLand[p]) continue;
      let s0 = 0;
      let s1 = 0;
      let w = 0;
      for (let dj = -1; dj <= 1; dj++) {
        const jj = j + dj;
        if (jj < 0 || jj >= H) continue;
        for (let di = -1; di <= 1; di++) {
          const q = jj * W + ((i + di + W) % W);
          if (isLand[q]) continue;
          const wt = dj === 0 && di === 0 ? 4 : 1;
          s0 += wt * u[q];
          s1 += wt * v[q];
          w += wt;
        }
      }
      su[p] = s0 / w;
      sv[p] = s1 / w;
    }
  }
  // Onto the mesh from the ocean bins around each cell (land cells by the coast
  // take their sea's flow, so that it interpolates right up to the shore).
  const flowE = new Float32Array(mesh.n);
  const flowN = new Float32Array(mesh.n);
  for (let c = 0; c < mesh.n; c++) {
    const fj = ((Math.PI / 2 - mesh.lat[c]) / Math.PI) * H - 0.5;
    const fi = ((mesh.lon[c] + Math.PI) / (2 * Math.PI)) * W - 0.5;
    const j0 = Math.floor(fj);
    const i0 = Math.floor(fi);
    let se = 0;
    let sn = 0;
    let w = 0;
    for (let dj = 0; dj <= 1; dj++) {
      const jj = Math.max(0, Math.min(H - 1, j0 + dj));
      const wj = dj ? fj - j0 : 1 - (fj - j0);
      for (let di = 0; di <= 1; di++) {
        const q = jj * W + ((i0 + di + W) % W);
        if (isLand[q]) continue;
        const wt = wj * (di ? fi - i0 : 1 - (fi - i0));
        se += wt * su[q];
        sn += wt * sv[q];
        w += wt;
      }
    }
    if (w <= 1e-6) continue;
    let e = se / w;
    let nn = sn / w;
    // The trades along an upwelling coast drive a broad current toward the equator.
    if (land[c] <= 0.5) nn -= Math.sign(mesh.lat[c]) * UPWELLING_JET * upwell[c];
    const s = Math.hypot(e, nn);
    if (s > CURRENT_MAX) {
      e *= CURRENT_MAX / s;
      nn *= CURRENT_MAX / s;
    }
    flowE[c] = e;
    flowN[c] = nn;
  }
  return { flowE, flowN };
}

/**
 * Distance (km) from each land cell to the nearest ocean cell over land, and
 * which ocean cell that is (ocean cells: 0 and themselves).
 */
function coastalReach(mesh: SphereMesh, land: Float32Array, radiusKm: number): { dist: Float32Array; src: Int32Array } {
  const { n, nbrStart, nbr, nbrDist } = mesh;
  const dist = new Float32Array(n).fill(Infinity);
  const src = new Int32Array(n).fill(-1);
  const heap = new MinHeap(n);
  for (let c = 0; c < n; c++) {
    if (land[c] > 0.5) continue;
    dist[c] = 0;
    src[c] = c;
    heap.push(0, c);
  }
  while (heap.size > 0) {
    const c = heap.pop();
    const d = heap.lastKey;
    if (d > dist[c]) continue;
    for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
      const m = nbr[k];
      if (land[m] <= 0.5) continue;
      const dm = d + nbrDist[k] * radiusKm;
      if (dm < dist[m]) {
        dist[m] = dm;
        src[m] = src[c];
        heap.push(dm, m);
      }
    }
  }
  // Measure from the coastline, about half a cell from the first land cell's centre.
  const half = 0.5 * mesh.spacing * radiusKm;
  for (let c = 0; c < n; c++) {
    if (src[c] < 0) {
      // Land with no route to the sea (shouldn't happen on a sphere with oceans).
      dist[c] = 1e9;
      src[c] = c;
    } else if (dist[c] > 0) dist[c] = Math.max(0, dist[c] - half);
  }
  return { dist, src };
}

/**
 * Map latitude onto an idealized circulation coordinate (0 at the ITCZ, 30 at
 * the subtropical high, 60 at the polar front, 90 at the pole). The highs and
 * fronts follow the ITCZ only partially, as on Earth.
 */
function circulationLat(phi: number, itcz: number): { phiP: number; north: boolean } {
  const north = phi >= itcz;
  const sth = (north ? 30 : -30) + 0.4 * itcz;
  const pf = (north ? 60 : -60) + 0.2 * itcz;
  const pole = north ? 90 : -90;
  const a = Math.abs(phi - itcz);
  const s1 = Math.abs(sth - itcz);
  const s2 = Math.abs(pf - sth);
  const s3 = Math.abs(pole - pf);
  let phiP: number;
  if (a <= s1) phiP = (a / s1) * 30;
  else if (a <= s1 + s2) phiP = 30 + ((a - s1) / s2) * 30;
  else phiP = 60 + Math.min(1, (a - s1 - s2) / s3) * 30;
  return { phiP, north };
}

/** Three-cell circulation relative to the ITCZ. Returns [eastward, poleward] m/s. */
function cellWind(phiP: number): [number, number] {
  if (phiP < 30) {
    const s = Math.sin((Math.PI * phiP) / 30);
    return [-6.5 * s, -1.8 * s];
  }
  if (phiP < 60) {
    const s = Math.sin((Math.PI * (phiP - 30)) / 30);
    return [8.5 * s, 1.2 * s];
  }
  const s = Math.sin((Math.PI * (phiP - 60)) / 30);
  return [-4.5 * s, -1.0 * s];
}

interface MeshGeom {
  ex: Float32Array; ey: Float32Array; ez: Float32Array;
  nx: Float32Array; ny: Float32Array; nz: Float32Array;
  /**
   * Unit normal of each Voronoi edge at its midpoint, pointing from the cell
   * toward the neighbor (3 per adjacency). Using the midpoint tangent rather
   * than the cell's own tangent keeps the discrete divergence of smooth flows
   * near zero.
   */
  dir: Float32Array;
  /** Voronoi edge length (m) per adjacency. */
  len: Float32Array;
  /** Cell area (m²). */
  areaM: Float64Array;
}

function meshGeom(mesh: SphereMesh, radiusKm: number): MeshGeom {
  const { n, x, y, z, nbrStart, nbr, nbrEdge, area } = mesh;
  const R = radiusKm * 1000;
  const g: MeshGeom = {
    ex: new Float32Array(n), ey: new Float32Array(n), ez: new Float32Array(n),
    nx: new Float32Array(n), ny: new Float32Array(n), nz: new Float32Array(n),
    dir: new Float32Array(nbr.length * 3),
    len: new Float32Array(nbr.length),
    areaM: new Float64Array(n),
  };
  for (let c = 0; c < n; c++) {
    const r = Math.hypot(x[c], y[c]) || 1e-9;
    const ex = -y[c] / r;
    const ey = x[c] / r;
    g.ex[c] = ex;
    g.ey[c] = ey;
    g.ez[c] = 0;
    g.nx[c] = y[c] * 0 - z[c] * ey;
    g.ny[c] = z[c] * ex - x[c] * 0;
    g.nz[c] = x[c] * ey - y[c] * ex;
    g.areaM[c] = area[c] * R * R;
    for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
      const m = nbr[k];
      let mx = x[m] + x[c];
      let my = y[m] + y[c];
      let mz = z[m] + z[c];
      const ml = Math.hypot(mx, my, mz) || 1;
      mx /= ml;
      my /= ml;
      mz /= ml;
      let dx = x[m] - x[c];
      let dy = y[m] - y[c];
      let dz = z[m] - z[c];
      const d = dx * mx + dy * my + dz * mz;
      dx -= d * mx;
      dy -= d * my;
      dz -= d * mz;
      const l = Math.hypot(dx, dy, dz) || 1;
      g.dir[3 * k] = dx / l;
      g.dir[3 * k + 1] = dy / l;
      g.dir[3 * k + 2] = dz / l;
      g.len[k] = nbrEdge[k] * R;
    }
  }
  return g;
}

/** Green–Gauss gradient of a scalar field (units per metre, 3D tangent vector). */
function gradient(mesh: SphereMesh, g: MeshGeom, f: ArrayLike<number>, out: Float32Array): void {
  const { n, nbrStart, nbr } = mesh;
  for (let c = 0; c < n; c++) {
    let gx = 0;
    let gy = 0;
    let gz = 0;
    for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
      const w = 0.5 * (f[nbr[k]] - f[c]) * g.len[k];
      gx += w * g.dir[3 * k];
      gy += w * g.dir[3 * k + 1];
      gz += w * g.dir[3 * k + 2];
    }
    const a = g.areaM[c];
    out[3 * c] = gx / a;
    out[3 * c + 1] = gy / a;
    out[3 * c + 2] = gz / a;
  }
}

export function computeClimate(mesh: SphereMesh, elev: Float32Array, land: Float32Array, params: WorldParams, orbit: Orbit | null = null): Climate {
  const { n, lat, lon, nbrStart, nbr } = mesh;
  const R = params.radiusKm;
  const tilt = (params.axialTiltDeg + (orbit ? orbit.dTiltDeg : 0)) * DEG;
  const rot = params.retrograde ? -1 : 1;
  const geom = meshGeom(mesh, R);

  // Insolation table: per 1° band and month, plus annual mean and harmonics.
  const Qm = new Float64Array(LAT_BANDS * MONTHS);
  const Q0 = new Float64Array(LAT_BANDS);
  const Qre = [new Float64Array(LAT_BANDS), new Float64Array(LAT_BANDS)];
  const Qim = [new Float64Array(LAT_BANDS), new Float64Array(LAT_BANDS)];
  const decl = Array.from({ length: MONTHS }, (_, m) => declination(m, tilt));
  for (let j = 0; j < LAT_BANDS; j++) {
    const phi = bandLat(j);
    let sum = 0;
    for (let s = 0; s < 96; s++) sum += insolation(phi, declination((s / 96) * 12 - 0.5, tilt)) * orbitFactor((s * 365.25) / 96, orbit);
    Q0[j] = sum / 96;
    for (let m = 0; m < MONTHS; m++) {
      const q = insolation(phi, decl[m]) * orbitFactor(((m + 0.5) * 365.25) / 12, orbit);
      Qm[j * MONTHS + m] = q;
      for (let k = 1; k <= 2; k++) {
        const ph = (k * 2 * Math.PI * (m + 0.5)) / 12;
        Qre[k - 1][j] += (2 / 12) * q * Math.cos(ph);
        Qim[k - 1][j] -= (2 / 12) * q * Math.sin(ph);
      }
    }
  }
  const zonalT = zonalEBM(Q0, params.temperatureOffset);

  const current = currentAnomaly(mesh, land, params.retrograde);
  const upwelling = upwellingProximity(mesh, land, params.retrograde);
  const upwell = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    if (land[c] > 0.5) continue;
    upwell[c] = bump(Math.abs(lat[c] / DEG), 0, 18, 44) * upwelling[c];
    current[c] -= UPWELLING_MAX * upwell[c];
  }
  const { flowE, flowN } = currentVelocity(mesh, land, params.retrograde, upwell, R);
  const side = subtropicalSide(mesh, land, params.retrograde);
  const reach = coastalReach(mesh, land, R);

  // Annual-mean wind for the maritime-influence tracer.
  const annE = new Float32Array(n);
  const annN = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    const { phiP, north } = circulationLat(lat[c] / DEG, 3);
    const [u, v] = cellWind(phiP);
    annE[c] = rot * u;
    annN[c] = north ? v : -v;
  }

  // Maritime influence & advected current anomaly: relaxed toward upwind values,
  // decaying with distance inland (~1200 km e-folding).
  const maritime = new Float32Array(n);
  const anomLand = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    if (land[c] <= 0.5) {
      maritime[c] = 1;
      anomLand[c] = current[c];
    }
  }
  const beta = (mesh.spacing * R) / 1200;
  const upw = new Float32Array(nbr.length);
  const upwSum = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
      const m = nbr[k];
      // How directly the wind at m blows toward c.
      const wx = annE[m] * geom.ex[m] + annN[m] * geom.nx[m];
      const wy = annE[m] * geom.ey[m] + annN[m] * geom.ny[m];
      const wz = annE[m] * geom.ez[m] + annN[m] * geom.nz[m];
      const sp = Math.hypot(wx, wy, wz) || 1;
      const toward = -(wx * geom.dir[3 * k] + wy * geom.dir[3 * k + 1] + wz * geom.dir[3 * k + 2]) / sp;
      upw[k] = Math.max(0.08, toward) ** 2;
      upwSum[c] += upw[k];
    }
  }
  for (let it = 0; it < 60; it++) {
    for (let c = 0; c < n; c++) {
      if (land[c] <= 0.5) continue;
      let sm = 0;
      let sa = 0;
      for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
        const m = nbr[k];
        sm += upw[k] * maritime[m];
        sa += upw[k] * anomLand[m];
      }
      maritime[c] = (1 - beta) * (sm / upwSum[c]);
      anomLand[c] = (1 - beta) * (sa / upwSum[c]);
    }
  }
  // Coastal mixing: a strip along every coast takes on the sea's character
  // and its current's temperature, most strongly on upwelling coasts.
  for (let c = 0; c < n; c++) {
    if (land[c] <= 0.5) continue;
    const s = reach.src[c];
    const cm = (COAST_MIX_BASE + COAST_MIX_UPWELLING * upwelling[s]) * Math.exp(-reach.dist[c] / COAST_MIX_KM);
    maritime[c] = 1 - (1 - maritime[c]) * (1 - cm);
    anomLand[c] += cm * (current[s] - anomLand[c]);
  }
  const drift = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    drift[c] = Math.max(0, land[c] > 0.5 ? anomLand[c] : current[c]) * smoothstep(35, 50, Math.abs(lat[c] / DEG));
  }
  // Stability of the air from the sea surface beneath or upwind of it: cold
  // water under the subtropical highs gives a stable marine layer (fog and
  // stratus, scarcely any rain); warm currents feed convection. Storms
  // override the effect poleward of ~45°.
  const stable = new Float32Array(n);
  const unstable = new Float32Array(n);
  // Unit tangent from each land cell toward its nearest sea, for onshore winds.
  const seaDir = new Float32Array(3 * n);
  for (let c = 0; c < n; c++) {
    if (land[c] <= 0.5) continue;
    const s = reach.src[c];
    let dx = mesh.x[s] - mesh.x[c];
    let dy = mesh.y[s] - mesh.y[c];
    let dz = mesh.z[s] - mesh.z[c];
    const dd = dx * mesh.x[c] + dy * mesh.y[c] + dz * mesh.z[c];
    dx -= dd * mesh.x[c];
    dy -= dd * mesh.y[c];
    dz -= dd * mesh.z[c];
    const l = Math.hypot(dx, dy, dz);
    if (l < 1e-12) continue;
    seaDir[3 * c] = dx / l;
    seaDir[3 * c + 1] = dy / l;
    seaDir[3 * c + 2] = dz / l;
  }
  for (let c = 0; c < n; c++) {
    const s = reach.src[c];
    // (The marine layer is the subtropical highs' doing: poleward of ~45° the storm track lifts it away.)
    const lw = 1 - smoothstep(30, 45, Math.abs(lat[c] / DEG));
    const cold = smoothstep(1.5, 6, -current[s]) * lw;
    const warm = smoothstep(1, 5, current[s]);
    if (land[c] <= 0.5) {
      stable[c] = cold;
      unstable[c] = warm;
    } else {
      const d = reach.dist[c];
      stable[c] = cold * Math.exp(-d / MARINE_LAYER_KM) * Math.exp(-Math.max(0, elev[c] - 300) / MARINE_LAYER_DEPTH);
      unstable[c] = warm * Math.exp(-d / 500);
    }
  }

  // Monthly sea-level temperature: zonal EBM mean + current anomaly + seasonal
  // harmonics damped by the local heat capacity (ocean vs. continental).
  const tempSL = new Float32Array(n * MONTHS);
  for (let c = 0; c < n; c++) {
    const j = bandOf(lat[c]);
    const phi = lat[c];
    const albedo = seasonalAlbedo(phi);
    const albedoLand = landSeasonalAlbedo(phi);
    const m = Math.min(1, maritime[c]);
    // Snowy high-latitude interiors radiate heat away in winter: colder annual means.
    const interior = land[c] > 0.5 ? -3.5 * (1 - m) * Math.max(0, Math.min(1, (Math.abs(phi / DEG) - 20) / 40)) : 0;
    const base = zonalT[j] + (land[c] > 0.5 ? anomLand[c] : current[c]) + interior;
    let h1re = 0;
    let h1im = 0;
    let h2re = 0;
    let h2im = 0;
    for (let k = 1; k <= 2; k++) {
      // H = (1-α) / (B + i k ω C). A place under maritime air part of the
      // time follows the ocean's damped, lagging cycle that share of the time
      // and the land's own the rest: blending the responses, not the heat
      // capacities (which would let a little sea air flatten a whole continent's
      // seasons), keeps continental summers hot and winters cold.
      const resp = (cap: number, alb: number): [number, number] => {
        const br = B_EFF;
        const bi = k * OMEGA * cap;
        const den = br * br + bi * bi;
        return [((1 - alb) * br) / den, (-(1 - alb) * bi) / den];
      };
      const [or, oi] = resp(C_OCEAN, albedo);
      const [lr0, li0] = resp(C_LAND, albedoLand);
      const damp = 1 - TROPICAL_LAND_DAMP * (1 - smoothstep(12, 32, Math.abs(phi / DEG)));
      const lr = lr0 * damp;
      const li = li0 * damp;
      const hr = m * or + (1 - m) * lr;
      const hi = m * oi + (1 - m) * li;
      const qr = Qre[k - 1][j];
      const qi = Qim[k - 1][j];
      if (k === 1) {
        h1re = hr * qr - hi * qi;
        h1im = hr * qi + hi * qr;
      } else {
        h2re = hr * qr - hi * qi;
        h2im = hr * qi + hi * qr;
      }
    }
    for (let mo = 0; mo < MONTHS; mo++) {
      const ph = (2 * Math.PI * (mo + 0.5)) / 12;
      const t =
        base + (h1re * Math.cos(ph) - h1im * Math.sin(ph)) + (h2re * Math.cos(2 * ph) - h2im * Math.sin(2 * ph));
      tempSL[c * MONTHS + mo] = t;
    }
  }

  // ITCZ: latitude of the warmest sea-level air within ±30°, per longitude sector.
  const itcz = new Float32Array(MONTHS * SECTORS);
  {
    const bins = 61;
    const sum = new Float64Array(SECTORS * bins);
    const cnt = new Float64Array(SECTORS * bins);
    for (let mo = 0; mo < MONTHS; mo++) {
      sum.fill(0);
      cnt.fill(0);
      for (let c = 0; c < n; c++) {
        const ld = lat[c] / DEG;
        if (ld < -30 || ld > 30) continue;
        const b = Math.round(ld + 30);
        const sct = Math.min(SECTORS - 1, Math.floor(((lon[c] + Math.PI) / (2 * Math.PI)) * SECTORS));
        sum[sct * bins + b] += tempSL[c * MONTHS + mo] * mesh.area[c];
        cnt[sct * bins + b] += mesh.area[c];
      }
      const raw = new Float32Array(SECTORS);
      for (let sct = 0; sct < SECTORS; sct++) {
        let best = -Infinity;
        let bl = 0;
        for (let b = 0; b < bins; b++) {
          if (cnt[sct * bins + b] <= 0) continue;
          // light smoothing across neighboring latitude bins
          let s = 0;
          let w = 0;
          for (let d = -3; d <= 3; d++) {
            const bb = b + d;
            if (bb < 0 || bb >= bins || cnt[sct * bins + bb] <= 0) continue;
            s += sum[sct * bins + bb] / cnt[sct * bins + bb];
            w++;
          }
          const v = s / w;
          if (v > best) {
            best = v;
            bl = b - 30;
          }
        }
        raw[sct] = Math.max(-20, Math.min(20, bl));
      }
      // Oceans tie the ITCZ toward its zonal mean; continents pull it further.
      let mean = 0;
      for (let sct = 0; sct < SECTORS; sct++) mean += raw[sct] / SECTORS;
      for (let sct = 0; sct < SECTORS; sct++) raw[sct] = 0.65 * raw[sct] + 0.35 * mean;
      for (let pass = 0; pass < 3; pass++) {
        const tmp = raw.slice();
        for (let sct = 0; sct < SECTORS; sct++) {
          raw[sct] = 0.25 * tmp[(sct + SECTORS - 1) % SECTORS] + 0.5 * tmp[sct] + 0.25 * tmp[(sct + 1) % SECTORS];
        }
      }
      itcz.set(raw, mo * SECTORS);
    }
  }

  // Surface gradient of the terrain for orographic lift.
  const hSurf = new Float32Array(n);
  for (let c = 0; c < n; c++) hSurf[c] = Math.max(0, elev[c]);
  const gradH = new Float32Array(n * 3);
  gradient(mesh, geom, hSurf, gradH);

  const windE = new Float32Array(n * MONTHS);
  const windN = new Float32Array(n * MONTHS);
  const precip = new Float32Array(n * MONTHS);

  // Sweep orders for the steady-state moisture solve (fast sweeping).
  const orders: Int32Array[] = [];
  {
    const idx = Int32Array.from({ length: n }, (_, i) => i);
    const byLon = idx.slice().sort((a, b) => lon[a] - lon[b]);
    const byLat = idx.slice().sort((a, b) => lat[a] - lat[b]);
    orders.push(byLon, byLon.slice().reverse(), byLat, byLat.slice().reverse());
  }

  const wx = new Float32Array(n);
  const wy = new Float32Array(n);
  const wz = new Float32Array(n);
  const pres = new Float32Array(n);
  const tmp = new Float32Array(n);
  const gradP = new Float32Array(n * 3);
  const tSurf = new Float32Array(n);
  const div = new Float32Array(n);
  const outC = new Float32Array(n);
  const inC = new Float32Array(nbr.length);
  const evap = new Float32Array(n);
  const rate = new Float32Array(n);
  const q = new Float32Array(n);
  const P = new Float32Array(n);
  const bandSum = new Float64Array(LAT_BANDS);
  const bandW = new Float64Array(LAT_BANDS);
  // Transient eddies (passing storm systems) mix moisture down its gradient.
  // Outside the tropics this, not the mean wind, carries most of the water
  // that reaches high latitudes; the Hadley cell dominates closer in.
  const eddy = new Float32Array(nbr.length);
  const eddySum = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
      const m = nbr[k];
      const phi = (0.5 * (Math.abs(lat[c]) + Math.abs(lat[m]))) / DEG;
      const K = EDDY_DIFFUSIVITY * smoothstep(EDDY_FROM, EDDY_FULL, phi);
      eddy[k] = (K * geom.len[k]) / (mesh.nbrDist[k] * R * 1000 * geom.areaM[c]);
      eddySum[c] += eddy[k];
    }
  }

  for (let mo = 0; mo < MONTHS; mo++) {
    // Thermal pressure anomaly: air warmer than its latitude's mean rises (low pressure).
    bandSum.fill(0);
    bandW.fill(0);
    for (let c = 0; c < n; c++) {
      const j = bandOf(lat[c]);
      bandSum[j] += tempSL[c * MONTHS + mo] * mesh.area[c];
      bandW[j] += mesh.area[c];
    }
    for (let c = 0; c < n; c++) {
      const j = bandOf(lat[c]);
      const bm = bandW[j] > 0 ? bandSum[j] / bandW[j] : tempSL[c * MONTHS + mo];
      pres[c] = -(tempSL[c * MONTHS + mo] - bm);
    }
    for (let pass = 0; pass < 6; pass++) {
      for (let c = 0; c < n; c++) {
        let s = 0;
        let cnt = 0;
        for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
          s += pres[nbr[k]];
          cnt++;
        }
        tmp[c] = 0.5 * pres[c] + (0.5 * s) / cnt;
      }
      pres.set(tmp);
    }
    gradient(mesh, geom, pres, gradP);

    for (let c = 0; c < n; c++) {
      const phiDeg = lat[c] / DEG;
      const fpos = ((lon[c] + Math.PI) / (2 * Math.PI)) * SECTORS - 0.5;
      const s0 = (Math.floor(fpos) + SECTORS) % SECTORS;
      const s1 = (s0 + 1) % SECTORS;
      const fr = fpos - Math.floor(fpos);
      const phiI = itcz[mo * SECTORS + s0] * (1 - fr) + itcz[mo * SECTORS + s1] * fr;
      const { phiP, north } = circulationLat(phiDeg, phiI);
      const [u, v] = cellWind(phiP);
      const e = rot * u;
      const nn = north ? v : -v;
      // Monsoon/thermal component: down-gradient flow plus Coriolis turning.
      const gpx = gradP[3 * c] * R * 1000;
      const gpy = gradP[3 * c + 1] * R * 1000;
      const gpz = gradP[3 * c + 2] * R * 1000;
      const px = mesh.x[c];
      const py = mesh.y[c];
      const pz = mesh.z[c];
      const kx = py * gpz - pz * gpy;
      const ky = pz * gpx - px * gpz;
      const kz = px * gpy - py * gpx;
      const turn = rot * Math.sign(phiDeg) * Math.min(1, Math.abs(phiDeg) / 15);
      const a1 = 0.08;
      const a2 = 0.1 * turn;
      let mx = -a1 * gpx + a2 * kx;
      let my = -a1 * gpy + a2 * ky;
      let mz = -a1 * gpz + a2 * kz;
      const ms = Math.hypot(mx, my, mz);
      if (ms > MONSOON_MAX) {
        const k = MONSOON_MAX / ms;
        mx *= k;
        my *= k;
        mz *= k;
      }
      // The subtropical highs' flanks: poleward on the western flank, equatorward on the eastern; strongest in summer.
      const flank = HIGH_FLANK_WIND * side[c] * bump(Math.abs(phiDeg), 10, 27, 42) * (0.6 + 0.4 * summerOf(phiDeg, mo)) * Math.sign(phiDeg);
      wx[c] = e * geom.ex[c] + (nn + flank) * geom.nx[c] + mx;
      wy[c] = e * geom.ey[c] + (nn + flank) * geom.ny[c] + my;
      wz[c] = e * geom.ez[c] + (nn + flank) * geom.nz[c] + mz;
      windE[c * MONTHS + mo] = wx[c] * geom.ex[c] + wy[c] * geom.ey[c] + wz[c] * geom.ez[c];
      windN[c * MONTHS + mo] = wx[c] * geom.nx[c] + wy[c] * geom.ny[c] + wz[c] * geom.nz[c];
      tSurf[c] = tempSL[c * MONTHS + mo] - (LAPSE * hSurf[c]) / 1000;
    }

    // Conservative finite volumes: one normal velocity per Voronoi edge (the
    // average of both cells' winds), upwind donor-cell fluxes.
    for (let c = 0; c < n; c++) {
      let o = 0;
      let dv = 0;
      for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
        const m = nbr[k];
        const u =
          0.5 *
          ((wx[c] + wx[m]) * geom.dir[3 * k] + (wy[c] + wy[m]) * geom.dir[3 * k + 1] + (wz[c] + wz[m]) * geom.dir[3 * k + 2]);
        const f = u * geom.len[k];
        if (u > 0) o += f;
        inC[k] = u < 0 ? -f / geom.areaM[c] : 0;
        dv += f;
      }
      outC[c] = o / geom.areaM[c];
      div[c] = dv / geom.areaM[c];
    }

    // Steady-state moisture balance, solved by Gauss–Seidel fast sweeping
    // with Picard updates of the nonlinear source/sink terms.
    for (let c = 0; c < n; c++) {
      const qs = 7 * Math.exp(0.065 * tSurf[c]);
      q[c] = land[c] > 0.5 ? 0.3 * qs : 0.7 * qs;
      P[c] = 2;
    }
    for (let it = 0; it < 8; it++) {
      for (let c = 0; c < n; c++) {
        const T = tSurf[c];
        const qs = 7 * Math.exp(0.065 * T);
        const speed = Math.hypot(wx[c], wy[c], wz[c]);
        if (land[c] <= 0.5) {
          const sst = tempSL[c * MONTHS + mo];
          let e = EVAP_20 * Math.exp(0.06 * (sst - 20)) * (0.5 + 0.5 * Math.min(2, speed / 7));
          if (sst < -1.5) e *= 0.1;
          evap[c] = e;
        } else {
          const pet = T > 0 ? (3.3 * T ** 1.25) / DAYS_PER_MONTH : 0;
          // Land gives back the share of its rain the Budyko curve (Fu's form)
          // expects: most of it where evaporative demand far outstrips rain,
          // under half where rain is plentiful. Dry interiors recycle their
          // rain, which carries moisture deep into the continents.
          const phi = pet / Math.max(1e-6, P[c]);
          evap[c] = P[c] * (1 + phi - (1 + phi ** 2.6) ** (1 / 2.6));
          const eqW = 1 - smoothstep(6, 14, Math.abs(lat[c] / DEG));
          if (eqW > 0) evap[c] = Math.max(evap[c], EQUATORIAL_TRANSPIRATION * eqW * pet * (1 - stable[c]) * (1 - smoothstep(-0.2, 0.5, -side[c])));
        }
        const rh = q[c] / qs;
        const phiDeg = Math.abs(lat[c] / DEG);
        // Storm tracks follow the polar front toward the equator in winter.
        const shift = STORM_WINTER_SHIFT * (1 - summerOf(lat[c] / DEG, mo));
        let storm = STORM_RATE * bump(phiDeg, 28 - shift, 52 - shift, 92);
        // Warm surfaces destabilize the column: summer convective showers.
        // A stable marine layer over cold water damps everything but storms;
        // warm currents feed the showers.
        const st = stable[c];
        const un = unstable[c];
        if (land[c] > 0.5 && storm > 0) {
          const onshore = Math.max(0, -(wx[c] * seaDir[3 * c] + wy[c] * seaDir[3 * c + 1] + wz[c] * seaDir[3 * c + 2])) / Math.max(1, speed);
          // (Cyclones running up a coast blow onshore ahead of them even where the
          // mean wind is offshore: nor'easters on the eastern seaboards.)
          storm *= 1 + COAST_STORM * (0.35 + 0.65 * onshore) * Math.exp(-reach.dist[c] / COAST_STORM_KM) * (1 - st);
        }
        // Under a high's eastern flank the air aloft sinks all year: heat lows
        // beneath it stay shallow and their inflow brings no rain (the
        // Sahara, Namib, Atacama, Sonora and western Australia).
        const sub = side[c] * bump(phiDeg, 10, 27, 42);
        // The highs follow the sun: the capped zone spans ~5–40° in winter and
        // ~10–45° in summer, when the monsoons reach the coasts at 10–15°.
        const sm = summerOf(lat[c] / DEG, mo);
        const capped = smoothstep(-0.2, 0.5, -side[c]) * bump(phiDeg, 5 + 5 * sm, 20 + 8 * sm, 40 + 5 * sm);
        const convective =
          0.045 * Math.max(0, Math.min(1, (T - 10) / 18)) * Math.min(1, rh / 0.45) * (1 - st) * (1 + 1.5 * un) * (1 - 0.9 * capped);
        let r =
          0.1 * Math.max(0, (rh - 0.25) / 0.75) ** 1.3 * (1 - 0.92 * st) * (1 + 0.5 * un) * (1 - 0.7 * capped) +
          storm * Math.min(1, rh / 0.5) * (1 - 0.6 * st) +
          convective;
        if (div[c] > 0) r *= Math.exp(-div[c] * 1.6e5 * (1 - HIGH_FLANK_SUBSIDENCE * (sub > 0 ? sub : 0.7 * sub)));
        else r += 4e4 * -div[c] * (1 - 0.95 * st) * (1 - 0.95 * capped);
        // (The equatorial trough's showers come from the land's own heating, which
        // the trades' gentle spreading between the rain belt's passes doesn't stop.)
        if (land[c] > 0.5) {
          r += EQUATORIAL_SHOWERS * (1 - smoothstep(4, 14, phiDeg)) * Math.max(0, Math.min(1, (T - 18) / 8)) * Math.min(1, rh / 0.5) * (1 - st) * (1 - capped);
        }
        const lift = wx[c] * gradH[3 * c] + wy[c] * gradH[3 * c + 1] + wz[c] * gradH[3 * c + 2];
        if (lift > 0) r += 9 * lift * (1 - 0.85 * st) * (1 - 0.7 * capped);
        else r *= Math.exp(lift * 40);
        if (rh > 1) r += 2 * (rh - 1) * (1 - 0.9 * st) * (1 - 0.8 * capped);
        rate[c] = r;
      }
      for (const order of orders) {
        for (let i = 0; i < n; i++) {
          const c = order[i];
          let num = evap[c] / 86400;
          for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
            const qm = q[nbr[k]];
            if (inC[k] > 0) num += inC[k] * qm;
            num += eddy[k] * qm;
          }
          q[c] = num / Math.max(1e-9, outC[c] + eddySum[c] + rate[c] / 86400);
        }
      }
      for (let c = 0; c < n; c++) P[c] = rate[c] * q[c];
    }
    for (let c = 0; c < n; c++) precip[c * MONTHS + mo] = P[c] * DAYS_PER_MONTH;
  }

  return { n, tempSL, precip, windE, windN, maritime, current, drift, flowE, flowN, upwell, itcz, zonalT };
}
