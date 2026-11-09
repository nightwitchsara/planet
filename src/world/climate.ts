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
const OMEGA = (2 * Math.PI) / (365.25 * 86400);
/** Heat capacity of a 30 m ocean mixed layer vs. land + atmosphere (J/m²/K). */
const C_OCEAN = 1.2e8;
const C_LAND = 2.0e7;
/** Local radiative + transport damping of seasonal anomalies (W/m²/K). */
const B_EFF = 4.5;
/** Strongest thermal (monsoon) wind perturbation (m/s). */
const MONSOON_MAX = 6;
/** Moisture mixing by extratropical storm systems (m²/s). */
const EDDY_DIFFUSIVITY = 1.5e6;
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
  /** ITCZ latitude (degrees) per month and longitude sector. */
  itcz: Float32Array;
  /** Zonal-mean annual temperature from the energy-balance model (1° bands). */
  zonalT: Float64Array;
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
  for (let iter = 0; iter < 40; iter++) {
    for (let j = 0; j < J; j++) {
      const phi = bandLat(j);
      const s = Math.sin(phi);
      // Planetary albedo rising poleward (low sun, clouds): 0.22 at the equator, 0.52 at the poles.
      const free = 0.32 + 0.2 * ((3 * s * s - 1) / 2);
      const ice = 1 / (1 + Math.exp((T[j] + 10) / 2));
      const albedo = free + (0.62 - free) * ice;
      const cj = Math.cos(phi);
      const a = Math.max(0, Math.cos(phi + dphi / 2)) / (cj * dphi * dphi);
      const b = Math.max(0, Math.cos(phi - dphi / 2)) / (cj * dphi * dphi);
      lo[j] = j > 0 ? -D_EBM * b : 0;
      up[j] = j < J - 1 ? -D_EBM * a : 0;
      di[j] = B_OLR + D_EBM * ((j < J - 1 ? a : 0) + (j > 0 ? b : 0));
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

const bump = (v: number, a: number, peak: number, b: number): number =>
  v <= a || v >= b ? 0 : v < peak ? Math.sin(((v - a) / (peak - a)) * (Math.PI / 2)) : Math.cos(((v - peak) / (b - peak)) * (Math.PI / 2));

/**
 * Sea-surface temperature anomaly from wind-driven gyres, by position within
 * each ocean basin along a latitude circle: warm western boundary currents and
 * cold eastern upwelling in the subtropics, warm drift reaching the eastern
 * side of subpolar basins.
 */
function currentAnomaly(mesh: SphereMesh, land: Float32Array, retrograde: boolean): Float32Array {
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
          wPol * (5.5 * Math.exp(-(1 - u) / 0.35) - 3 * Math.exp(-u / 0.12));
        anomGrid[j * W + ((i0 + t) % W)] = a * widthFactor;
      }
    }
  }
  // Currents are broad features: blur across rows (ocean bins only) so that
  // row-to-row differences in coastline geometry don't streak the field.
  const tmp = new Float32Array(W * H);
  for (let pass = 0; pass < 3; pass++) {
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const p = j * W + i;
        if (isLand[p]) continue;
        let s = 0;
        let wsum = 0;
        for (let dj = -2; dj <= 2; dj++) {
          const jj = j + dj;
          if (jj < 0 || jj >= H) continue;
          for (let di = -1; di <= 1; di++) {
            const q = jj * W + ((i + di + W) % W);
            if (isLand[q]) continue;
            s += anomGrid[q];
            wsum++;
          }
        }
        tmp[p] = s / wsum;
      }
    }
    for (let p = 0; p < W * H; p++) if (!isLand[p]) anomGrid[p] = tmp[p];
  }
  const out = new Float32Array(mesh.n);
  for (let c = 0; c < mesh.n; c++) {
    if (land[c] > 0.5) continue;
    const j = Math.max(0, Math.min(H - 1, Math.floor(((Math.PI / 2 - mesh.lat[c]) / Math.PI) * H)));
    const i = Math.max(0, Math.min(W - 1, Math.floor(((mesh.lon[c] + Math.PI) / (2 * Math.PI)) * W)));
    out[c] = anomGrid[j * W + i];
  }
  return out;
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

export function computeClimate(mesh: SphereMesh, elev: Float32Array, land: Float32Array, params: WorldParams): Climate {
  const { n, lat, lon, nbrStart, nbr } = mesh;
  const R = params.radiusKm;
  const tilt = params.axialTiltDeg * DEG;
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
    for (let s = 0; s < 96; s++) sum += insolation(phi, declination((s / 96) * 12 - 0.5, tilt));
    Q0[j] = sum / 96;
    for (let m = 0; m < MONTHS; m++) {
      const q = insolation(phi, decl[m]);
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

  // Monthly sea-level temperature: zonal EBM mean + current anomaly + seasonal
  // harmonics damped by the local heat capacity (ocean vs. continental).
  const tempSL = new Float32Array(n * MONTHS);
  for (let c = 0; c < n; c++) {
    const j = bandOf(lat[c]);
    const phi = lat[c];
    const s = Math.sin(phi);
    const albedo = 0.32 + 0.1 * s * s;
    const m = Math.min(1, maritime[c]);
    const C = m * C_OCEAN + (1 - m) * C_LAND;
    // Snowy high-latitude interiors radiate heat away in winter: colder annual means.
    const interior = land[c] > 0.5 ? -3.5 * (1 - m) * Math.max(0, Math.min(1, (Math.abs(phi / DEG) - 20) / 40)) : 0;
    const base = zonalT[j] + (land[c] > 0.5 ? anomLand[c] : current[c]) + interior;
    let h1re = 0;
    let h1im = 0;
    let h2re = 0;
    let h2im = 0;
    for (let k = 1; k <= 2; k++) {
      // H = (1-α) / (B + i k ω C)
      const br = B_EFF;
      const bi = k * OMEGA * C;
      const den = br * br + bi * bi;
      const hr = ((1 - albedo) * br) / den;
      const hi = (-(1 - albedo) * bi) / den;
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
      const K = EDDY_DIFFUSIVITY * smoothstep(25, 50, phi);
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
      wx[c] = e * geom.ex[c] + nn * geom.nx[c] + mx;
      wy[c] = e * geom.ey[c] + nn * geom.ny[c] + my;
      wz[c] = e * geom.ez[c] + nn * geom.nz[c] + mz;
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
          let e = 3.4 * Math.exp(0.06 * (sst - 20)) * (0.5 + 0.5 * Math.min(2, speed / 7));
          if (sst < -1.5) e *= 0.1;
          evap[c] = e;
        } else {
          const pet = T > 0 ? (3.3 * T ** 1.25) / DAYS_PER_MONTH : 0;
          evap[c] = Math.min(pet, 0.6 * P[c]);
        }
        const rh = q[c] / qs;
        const phiDeg = Math.abs(lat[c] / DEG);
        const storm = 0.12 * bump(phiDeg, 28, 52, 92);
        // Warm surfaces destabilize the column: summer convective showers.
        const convective = 0.045 * Math.max(0, Math.min(1, (T - 10) / 18)) * Math.min(1, rh / 0.45);
        let r = 0.1 * Math.max(0, (rh - 0.25) / 0.75) ** 1.3 + storm * Math.min(1, rh / 0.5) + convective;
        if (div[c] > 0) r *= Math.exp(-div[c] * 1.6e5);
        else r += 4e4 * -div[c];
        const lift = wx[c] * gradH[3 * c] + wy[c] * gradH[3 * c + 1] + wz[c] * gradH[3 * c + 2];
        if (lift > 0) r += 9 * lift;
        else r *= Math.exp(lift * 40);
        if (rh > 1) r += 2 * (rh - 1);
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

  return { n, tempSL, precip, windE, windN, maritime, current, itcz, zonalT };
}
