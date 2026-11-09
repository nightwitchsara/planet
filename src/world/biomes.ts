import type { SphereMesh } from '../core/mesh';
import { MONTHS } from './climate';
import { type Hydrology, LakeKind, Wetland } from './hydrology';
import type { WaterBalance } from './waterBalance';

export const KOPPEN_CODES = [
  'Af', 'Am', 'Aw', 'BWh', 'BWk', 'BSh', 'BSk', 'Csa', 'Csb', 'Csc', 'Cwa', 'Cwb', 'Cwc', 'Cfa', 'Cfb', 'Cfc',
  'Dsa', 'Dsb', 'Dsc', 'Dsd', 'Dwa', 'Dwb', 'Dwc', 'Dwd', 'Dfa', 'Dfb', 'Dfc', 'Dfd', 'ET', 'EF',
] as const;

export const KOPPEN_NAMES: Record<string, string> = {
  Af: 'Tropical rainforest', Am: 'Tropical monsoon', Aw: 'Tropical savanna',
  BWh: 'Hot desert', BWk: 'Cold desert', BSh: 'Hot semi-arid', BSk: 'Cold semi-arid',
  Csa: 'Hot-summer Mediterranean', Csb: 'Warm-summer Mediterranean', Csc: 'Cold-summer Mediterranean',
  Cwa: 'Monsoon subtropical', Cwb: 'Subtropical highland', Cwc: 'Cold subtropical highland',
  Cfa: 'Humid subtropical', Cfb: 'Oceanic', Cfc: 'Subpolar oceanic',
  Dsa: 'Dry-summer continental', Dsb: 'Dry-summer continental', Dsc: 'Dry-summer subarctic', Dsd: 'Dry-summer subarctic',
  Dwa: 'Monsoon continental', Dwb: 'Monsoon continental', Dwc: 'Monsoon subarctic', Dwd: 'Monsoon subarctic',
  Dfa: 'Hot-summer continental', Dfb: 'Warm-summer continental', Dfc: 'Subarctic', Dfd: 'Extreme subarctic',
  ET: 'Tundra', EF: 'Ice cap',
};

export const KOPPEN_COLORS: Record<string, [number, number, number]> = {
  Af: [0, 0, 254], Am: [0, 119, 255], Aw: [70, 169, 250],
  BWh: [254, 0, 0], BWk: [254, 150, 149], BSh: [245, 165, 0], BSk: [255, 220, 100],
  Csa: [255, 255, 0], Csb: [198, 199, 0], Csc: [150, 150, 0],
  Cwa: [150, 255, 150], Cwb: [99, 199, 100], Cwc: [50, 150, 51],
  Cfa: [198, 255, 78], Cfb: [102, 255, 51], Cfc: [51, 199, 1],
  Dsa: [255, 0, 254], Dsb: [198, 0, 199], Dsc: [150, 50, 149], Dsd: [150, 100, 150],
  Dwa: [171, 177, 255], Dwb: [90, 119, 219], Dwc: [76, 81, 181], Dwd: [50, 0, 135],
  Dfa: [0, 255, 255], Dfb: [56, 199, 255], Dfc: [0, 126, 125], Dfd: [0, 69, 94],
  ET: [178, 178, 178], EF: [104, 104, 104],
};

export const enum Biome {
  Ocean = 0, IceSheet, Tundra, Alpine, Boreal, TemperateRainforest, TemperateForest, Grassland, Mediterranean,
  HotDesert, ColdDesert, Shrubland, Savanna, TropicalDryForest, Rainforest, Wetland, Lake, SaltFlat,
}

export const BIOME_INFO: { name: string; color: [number, number, number] }[] = [
  { name: 'Ocean', color: [30, 60, 110] },
  { name: 'Ice sheet', color: [238, 244, 250] },
  { name: 'Tundra', color: [150, 150, 120] },
  { name: 'Alpine', color: [140, 130, 120] },
  { name: 'Boreal forest', color: [44, 78, 58] },
  { name: 'Temperate rainforest', color: [28, 84, 50] },
  { name: 'Temperate forest', color: [56, 104, 48] },
  { name: 'Grassland / steppe', color: [160, 158, 92] },
  { name: 'Mediterranean scrub', color: [128, 128, 74] },
  { name: 'Hot desert', color: [222, 196, 142] },
  { name: 'Cold desert', color: [176, 162, 128] },
  { name: 'Xeric shrubland', color: [186, 164, 112] },
  { name: 'Savanna', color: [164, 154, 78] },
  { name: 'Tropical dry forest', color: [96, 122, 52] },
  { name: 'Tropical rainforest', color: [24, 86, 34] },
  { name: 'Wetland', color: [64, 96, 76] },
  { name: 'Lake', color: [70, 120, 170] },
  { name: 'Salt flat', color: [232, 228, 214] },
];

/** Natural color of each kind of wetland (floodplain, inland delta, peat bog, swamp, marsh). */
const WETLAND_COLOR: [number, number, number][] = [
  [64, 96, 76], [74, 112, 64], [78, 116, 86], [104, 106, 82], [44, 84, 58], [96, 116, 78],
];

/** Köppen–Geiger class (index into KOPPEN_CODES) after Peel et al. (2007). */
export function koppenClass(T: ArrayLike<number>, P: ArrayLike<number>, off: number, northern: boolean): number {
  let mat = 0;
  let map = 0;
  let tCold = Infinity;
  let tHot = -Infinity;
  let m10 = 0;
  let pDry = Infinity;
  let pSum = 0;
  let pWin = 0;
  let psDry = Infinity;
  let psWet = 0;
  let pwDry = Infinity;
  let pwWet = 0;
  for (let m = 0; m < MONTHS; m++) {
    const t = T[off + m];
    const p = P[off + m];
    mat += t / MONTHS;
    map += p;
    if (t < tCold) tCold = t;
    if (t > tHot) tHot = t;
    if (t >= 10) m10++;
    if (p < pDry) pDry = p;
    // Summer half: Apr–Sep in the north, Oct–Mar in the south.
    const summer = (m >= 3 && m <= 8) === northern;
    if (summer) {
      pSum += p;
      psDry = Math.min(psDry, p);
      psWet = Math.max(psWet, p);
    } else {
      pWin += p;
      pwDry = Math.min(pwDry, p);
      pwWet = Math.max(pwWet, p);
    }
  }
  const idx = (code: string) => KOPPEN_CODES.indexOf(code as (typeof KOPPEN_CODES)[number]);
  if (tHot < 10) return idx(tHot > 0 ? 'ET' : 'EF');
  const pth = pWin >= 0.7 * map ? 2 * mat : pSum >= 0.7 * map ? 2 * mat + 28 : 2 * mat + 14;
  if (map < 10 * pth) return idx((map < 5 * pth ? 'BW' : 'BS') + (mat >= 18 ? 'h' : 'k'));
  if (tCold >= 18) {
    if (pDry >= 60) return idx('Af');
    if (pDry >= 100 - map / 25) return idx('Am');
    return idx('Aw');
  }
  // Dry season (Kottek et al. 2006): the dry season's minimum must undercut the
  // other season's; ties go to whichever half-year is wetter overall.
  const sDry = psDry < 40 && pwWet > 3 * psDry && (psDry < pwDry || (psDry === pwDry && pWin > pSum));
  const wDry = psWet > 10 * pwDry && (pwDry < psDry || (psDry === pwDry && pSum >= pWin));
  const s = sDry ? 's' : wDry ? 'w' : 'f';
  if (tCold > 0) return idx('C' + s + (tHot >= 22 ? 'a' : m10 >= 4 ? 'b' : 'c'));
  return idx('D' + s + (tHot >= 22 ? 'a' : m10 >= 4 ? 'b' : tCold < -38 ? 'd' : 'c'));
}

export interface BiomeResult {
  koppen: Uint8Array;
  biome: Uint8Array;
  /** Natural ("satellite") land color per cell, 3 bytes per cell. */
  color: Uint8Array;
}

function mix(a: readonly number[], b: readonly number[], t: number): [number, number, number] {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

export function computeBiomes(
  mesh: SphereMesh,
  elev: Float32Array,
  temp: Float32Array,
  precip: Float32Array,
  ocean: Uint8Array,
  hydro: Hydrology,
  water: WaterBalance,
): BiomeResult {
  const { n, lat } = mesh;
  const { lake, lakes, wetland } = hydro;
  const koppen = new Uint8Array(n);
  const biome = new Uint8Array(n);
  const color = new Uint8Array(n * 3);
  for (let c = 0; c < n; c++) {
    if (ocean[c]) {
      biome[c] = Biome.Ocean;
      koppen[c] = 255;
      continue;
    }
    const off = c * MONTHS;
    const k = koppenClass(temp, precip, off, lat[c] >= 0);
    koppen[c] = k;
    const code = KOPPEN_CODES[k];
    let map = 0;
    let mat = 0;
    let tHot = -Infinity;
    for (let m = 0; m < MONTHS; m++) {
      map += precip[off + m];
      mat += temp[off + m] / MONTHS;
      tHot = Math.max(tHot, temp[off + m]);
    }
    // What would grow here; lakes and salt flats keep it as their shore color.
    let b: Biome;
    if (code === 'EF' || water.glacier[c]) b = Biome.IceSheet;
    else if (code === 'ET') b = elev[c] > 1800 && Math.abs(lat[c]) < 1.0 ? Biome.Alpine : Biome.Tundra;
    else if (code === 'BWh') b = Biome.HotDesert;
    else if (code === 'BWk') b = Biome.ColdDesert;
    else if (code === 'BSh') b = Biome.Shrubland;
    else if (code === 'BSk') b = Biome.Grassland;
    else if (code === 'Af' || code === 'Am') b = Biome.Rainforest;
    else if (code === 'Aw') b = map > 1300 ? Biome.TropicalDryForest : Biome.Savanna;
    else if (code[1] === 's') b = code[0] === 'C' ? Biome.Mediterranean : map > 500 ? Biome.Boreal : Biome.Grassland;
    else if (code[0] === 'C') b = map > 1800 && code[2] !== 'a' ? Biome.TemperateRainforest : map > 650 ? Biome.TemperateForest : Biome.Grassland;
    else b = code[2] === 'c' || code[2] === 'd' ? Biome.Boreal : map > 550 ? Biome.TemperateForest : Biome.Grassland;
    // Waterlogged ground from the hydrology: floodplains, inland deltas, bogs, swamps.
    if (b !== Biome.IceSheet && wetland[c] !== Wetland.None) b = Biome.Wetland;
    biome[c] = lake[c] < 0 ? b : lakes[lake[c]].kind === LakeKind.SaltFlat ? Biome.SaltFlat : Biome.Lake;

    // Natural color: biome base tinted by aridity and warmth, snow on cold peaks.
    let col: [number, number, number] = [...(b === Biome.Wetland ? WETLAND_COLOR[wetland[c]] : BIOME_INFO[b].color)];
    const aridity = Math.max(0, Math.min(1, 1 - map / 900));
    if (b === Biome.Grassland || b === Biome.Savanna || b === Biome.Shrubland) {
      col = mix(col, [196, 176, 120], aridity * 0.5);
    }
    if (b === Biome.TemperateForest && mat > 14) col = mix(col, [70, 112, 44], 0.4);
    if (tHot < 3 && b !== Biome.IceSheet) col = mix(col, [236, 240, 245], Math.min(1, (3 - tHot) / 6));
    color[3 * c] = Math.round(col[0]);
    color[3 * c + 1] = Math.round(col[1]);
    color[3 * c + 2] = Math.round(col[2]);
  }
  // Ocean cells borrow their land neighbors' colors so coastal pixels that
  // interpolate toward them don't darken (and new land previews look sane).
  const { nbrStart, nbr } = mesh;
  for (let c = 0; c < n; c++) {
    if (!ocean[c]) continue;
    let r = 0;
    let g = 0;
    let b = 0;
    let k = 0;
    for (let j = nbrStart[c], e = nbrStart[c + 1]; j < e; j++) {
      const m = nbr[j];
      if (ocean[m]) continue;
      r += color[3 * m];
      g += color[3 * m + 1];
      b += color[3 * m + 2];
      k++;
    }
    const fallback = [176, 168, 122];
    color[3 * c] = k ? r / k : fallback[0];
    color[3 * c + 1] = k ? g / k : fallback[1];
    color[3 * c + 2] = k ? b / k : fallback[2];
  }
  return { koppen, biome, color };
}
