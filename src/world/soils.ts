import { smoothstep } from '../core/math';
import type { SphereMesh } from '../core/mesh';
import { Biome } from './biomes';
import { MONTHS } from './climate';
import { Rock } from './geology';
import { type Hydrology, LakeKind, Wetland } from './hydrology';
import { Deposit, type Deposits } from './sediment';
import type { WaterBalance } from './waterBalance';

/**
 * Soils by the twelve USDA soil orders, keyed in the USDA order: permafrost,
 * peat, volcanic ash, then the dry, young, cracking-clay, old tropical,
 * leached, podzolized and grassland or forest soils.
 */
export const enum Soil {
  Gelisol = 0, Histosol, Spodosol, Andisol, Oxisol, Vertisol, Aridisol, Ultisol, Mollisol, Alfisol, Inceptisol, Entisol,
  /** No soil: the sea. (Lake beds and ground under ice keep the soil they would have.) */
  None,
}

/**
 * Per order: plain name, map color, how the soil forms, and its natural
 * fertility before parent material and site (0–1).
 */
export const SOIL_INFO: { order: string; name: string; color: [number, number, number]; long: string; fertility: number }[] = [
  { order: 'Gelisol', name: 'frozen tundra soil', color: [150, 172, 206], long: 'permafrost a metre or two down; frost churns the thawed layer each summer', fertility: 0.08 },
  { order: 'Histosol', name: 'peat soil', color: [92, 64, 44], long: 'metres of half-rotted plants in waterlogged ground', fertility: 0.3 },
  { order: 'Spodosol', name: 'podzol', color: [190, 160, 210], long: 'ash-grey leached sand under conifers, its iron and humus washed into a dark subsoil', fertility: 0.2 },
  { order: 'Andisol', name: 'volcanic ash soil', color: [220, 88, 150], long: 'young volcanic ash weathering to light, porous, mineral-rich soil', fertility: 0.8 },
  { order: 'Oxisol', name: 'tropical red earth', color: [176, 42, 38], long: 'rock rotted for millions of years under tropical rain: iron and aluminium oxides, little else left', fertility: 0.25 },
  { order: 'Vertisol', name: 'black cracking clay', color: [62, 78, 110], long: 'swelling clay that cracks deep in the dry season and seals in the wet', fertility: 0.62 },
  { order: 'Aridisol', name: 'desert soil', color: [236, 214, 150], long: 'too dry to leach: lime, gypsum or salt gathers a little way down', fertility: 0.18 },
  { order: 'Ultisol', name: 'red-yellow forest soil', color: [232, 132, 52], long: 'long-leached, acid clay soil of warm humid forests', fertility: 0.36 },
  { order: 'Mollisol', name: 'prairie black earth', color: [64, 88, 40], long: 'deep black, humus-rich topsoil built by grass roots, rich in lime and nutrients', fertility: 0.9 },
  { order: 'Alfisol', name: 'brown forest soil', color: [138, 186, 92], long: 'moderately leached, clay-enriched soil that keeps its nutrients', fertility: 0.74 },
  { order: 'Inceptisol', name: 'young brown soil', color: [194, 160, 112], long: 'weakly developed soil on young surfaces: mountain slopes, recent glacial drift, river terraces', fertility: 0.52 },
  { order: 'Entisol', name: 'raw young soil', color: [214, 206, 190], long: 'scarcely any soil yet: fresh river silt, dune sand or thin rubble over rock', fertility: 0.5 },
  { order: 'None', name: 'no soil (ice or water)', color: [236, 242, 248], long: '', fertility: 0 },
];
export const SOIL_COUNT = 12;

/** Base supply of each bedrock (0–1): carbonate, basalt and ash weather to nutrient-rich soils, granite and quartz sand to poor ones. */
const ROCK_BASES = [0.7, 0.6, 0.85, 0.85, 0.35, 0.3, 0.4, 0.25, 0.5, 0.3, 0.6, 0.85, 0.15];
/** The same for the loose deposits over it (NaN: the bedrock's own). */
const DEPOSIT_BASES = [NaN, 0.2, NaN, 0.85, 0.55, 0.85, 0.7, 0.05, 0.12, 0.95, 0.65, 0.45, 0.3, 0.3, NaN];

/** Slope (reference-scale gradient) where soils start to thin and creep away, and where only rubble stays. */
const STEEP_FROM = 0.006;
const STEEP_TO = 0.03;
/** Relief within a cell (m) from rolling hills to rugged young mountains: cells average slopes out, ridges keep them. */
const RUGGED_FROM = 150;
const RUGGED_TO = 400;

export interface Soils {
  order: Uint8Array;
  /** 0–1 farming fertility. */
  fertility: Float32Array;
  /** Salt gathers in the soil (dry basins and salt flats). */
  saline: Uint8Array;
}

export interface SoilInputs {
  ocean: Uint8Array;
  elev: Float32Array;
  slope: Float32Array;
  /** Relief within each cell (m): ridge-and-valley country hides steep slopes inside gentle-looking cells. */
  rough: Float32Array;
  /** Monthly surface temperature (°C) and precipitation (mm), c*12+m. */
  temp: Float32Array;
  precip: Float32Array;
  water: WaterBalance;
  hydro: Hydrology;
  biome: Uint8Array;
  rock: Uint8Array;
  rockAge: Float32Array;
  deposits: Deposits;
  /** Ice-age ice thickness (m): ground it covered is only ~15 kyr old. */
  iceAgeThick: Float32Array;
  /** 0–1 volcanic activity and Myr since it last erupted (from the plate history). */
  volc: Float32Array;
  volcAge: Float32Array;
}

/**
 * Each land cell's soil order from its parent material (bedrock, or the
 * deposit over it, with its texture), climate (warmth, wetness, a dry
 * season), vegetation, drainage, slope and surface age; then its farming
 * fertility: the order's natural fertility blended with the parent
 * material's nutrients (young soils inherit nearly all of theirs), lowered
 * by steep ground, waterlogging, salt, thin soil, cold and drought.
 */
export function computeSoils(mesh: SphereMesh, s: SoilInputs): Soils {
  const n = mesh.n;
  const order = new Uint8Array(n).fill(Soil.None);
  const fertility = new Float32Array(n);
  const saline = new Uint8Array(n);
  const { kind, thickness, regolith, sand } = s.deposits;
  const { wetland, lake, lakes, endorheic } = s.hydro;
  const pet = s.water.pet;
  for (let c = 0; c < n; c++) {
    if (s.ocean[c]) continue;
    // Lake beds and the ground under glacier ice get the soil they would have
    // (the shores and ice margins show it), but ice grows nothing.
    const lk = lake[c];
    let tAnn = 0;
    let p = 0;
    let e = 0;
    let dry = 0;
    for (let m = 0; m < MONTHS; m++) {
      const k = c * MONTHS + m;
      tAnn += s.temp[k] / MONTHS;
      p += s.precip[k];
      e += pet[k];
      if (s.precip[k] < 0.5 * pet[k]) dry++;
    }
    /** Moisture index: rain over evaporative demand. */
    const mi = p / Math.max(1, e);
    const dep = kind[c];
    const rock = s.rock[c];
    const slope = s.slope[c];
    const biome = s.biome[c];
    const wet = wetland[c];
    const glaciated = s.iceAgeThick[c] > 0;
    // Steep or rugged ground: soil creeps and washes away about as fast as it forms.
    const cliffs = smoothstep(STEEP_FROM, STEEP_TO, slope);
    const steep = Math.max(cliffs, smoothstep(RUGGED_FROM, RUGGED_TO, s.rough[c]));
    const thin = dep === Deposit.Bedrock && regolith[c] < 0.6;
    const fluvial = dep === Deposit.Alluvium || dep === Deposit.Delta || dep === Deposit.Fan;
    const fresh = fluvial && (wet === Wetland.Floodplain || wet === Wetland.InlandDelta || dep === Deposit.Delta || (dep === Deposit.Fan && mi < 0.5));
    // Swelling (smectite) clay comes from basalt, marl and shale, and from fine lake and backswamp mud.
    const clay =
      (sand[c] < 0.3 && (rock === Rock.Basalt || rock === Rock.Shale || rock === Rock.Limestone || rock === Rock.OceanBasalt) && (dep === Deposit.Bedrock || dep === Deposit.Colluvium)) ||
      (sand[c] < 0.2 && (dep === Deposit.Lake || dep === Deposit.Alluvium || dep === Deposit.Delta));
    const acid = (dep === Deposit.Bedrock || dep === Deposit.Residual || dep === Deposit.Colluvium ? rock === Rock.Granite || rock === Rock.Basement || rock === Rock.Sandstone || rock === Rock.Metamorphic : false) || dep === Deposit.Outwash || dep === Deposit.Dunes || dep === Deposit.Moraine || sand[c] > 0.65;
    const volcanic = rock === Rock.ArcVolcanic || (rock === Rock.Basalt && s.rockAge[c] < 5) || (s.volc[c] > 0.05 && s.volcAge[c] < 3);
    // Salt gathers where water evaporates instead of draining away: salt flats, playas, closed dry hollows.
    const salty =
      dep === Deposit.Playa ||
      (lk >= 0 && lakes[lk].kind === LakeKind.SaltFlat) ||
      (mi < 0.3 && endorheic[c] === 1 && slope < 0.01 && (dep === Deposit.Lake || fluvial || wet === Wetland.Marsh));

    let o: Soil;
    // Gelisols need permafrost within a metre or two of the surface: continuous
    // below about -5 °C; in the warmer, patchy permafrost zone it lies too deep or
    // too scattered to make the soil, which develops as in the boreal forest.
    if (tAnn < -5) o = Soil.Gelisol;
    else if (dep === Deposit.Peat) o = Soil.Histosol;
    else if (volcanic && mi > 0.5 && !thin) o = Soil.Andisol;
    else if (salty) o = Soil.Aridisol;
    // Too dry to leach (the aridic moisture regime: about the UNEP arid limit), or hot and nearly so.
    else if (mi < 0.2) o = fresh || dep === Deposit.Dunes || dep === Deposit.Fan || thin ? Soil.Entisol : Soil.Aridisol;
    else if (dep === Deposit.Dunes) o = tAnn < 12 && mi > 0.8 ? Soil.Spodosol : Soil.Entisol;
    else if (fresh) o = clay && dry >= 5 && tAnn > 12 ? Soil.Vertisol : Soil.Entisol;
    // Mountains and bare rock: weakly developed soil where it is wet enough to weather, rubble where it is not.
    else if (thin || steep > 0.5) o = mi > 0.45 && cliffs < 0.85 ? Soil.Inceptisol : Soil.Entisol;
    else if (clay && dry >= 5 && dry <= 8 && tAnn > 12 && mi > 0.35 && mi < 0.9 && slope < 0.01) o = Soil.Vertisol;
    // River terraces in humid country are too young to be leached like the old land around them.
    else if (fluvial && mi > 0.8) o = Soil.Inceptisol;
    else if (tAnn > 20 && mi > 0.9 && !glaciated && s.rockAge[c] > 15 && (dep === Deposit.Residual || regolith[c] > 3)) o = Soil.Oxisol;
    else if (tAnn > 11 && mi > 1 && !glaciated) o = acid || s.rockAge[c] > 20 || dep === Deposit.Residual ? Soil.Ultisol : Soil.Alfisol;
    else if (tAnn < 11 && mi > 0.8 && (biome === Biome.Boreal || biome === Biome.TemperateRainforest || acid) && dep !== Deposit.Loess && rock !== Rock.Limestone) o = Soil.Spodosol;
    else if (mi < 0.3 && tAnn > 15) o = Soil.Aridisol;
    else if (mi < 0.95 && tAnn < 20 && (biome === Biome.Grassland || biome === Biome.Shrubland || biome === Biome.ColdDesert || biome === Biome.Mediterranean || biome === Biome.Savanna || mi < 0.6)) o = Soil.Mollisol;
    // Young ground in humid country: drift the ice left ~15 kyr ago, tundra and cold
    // forest where soils barely develop between freeze-ups.
    else if (glaciated && tAnn < 10 && (dep === Deposit.Till || dep === Deposit.Moraine || dep === Deposit.Outwash || dep === Deposit.Bedrock)) o = Soil.Inceptisol;
    else if (biome === Biome.Tundra || biome === Biome.Alpine || (tAnn < 2 && mi > 0.6)) o = Soil.Inceptisol;
    else o = Soil.Alfisol;
    order[c] = o;
    saline[c] = salty ? 1 : 0;

    // Fertility. Mature soils owe it mostly to how they formed, young ones to their parent material.
    const parentBases = Number.isNaN(DEPOSIT_BASES[dep]) ? ROCK_BASES[rock] : DEPOSIT_BASES[dep];
    const young = o === Soil.Entisol || o === Soil.Inceptisol || o === Soil.Andisol ? 0.3 : 0.7;
    let f = young * SOIL_INFO[o].fertility + (1 - young) * parentBases;
    // Thin or steep ground: little soil to root in, and it washes off once ploughed.
    f *= 1 - 0.7 * steep;
    if (dep === Deposit.Bedrock) f *= 0.4 + 0.6 * smoothstep(0.3, 1.5, regolith[c] + thickness[c]);
    // Waterlogged ground drowns roots; floodplains only flood now and then.
    if (wet === Wetland.Swamp || wet === Wetland.Marsh || wet === Wetland.Peatland) f *= 0.6;
    else if (wet === Wetland.Floodplain || wet === Wetland.InlandDelta) f *= 0.92;
    if (salty) f *= 0.25;
    // Short, cold growing seasons and drought (farming without irrigation).
    f *= 0.12 + 0.88 * smoothstep(-6, 6, tAnn);
    f *= 0.2 + 0.8 * smoothstep(0.1, 0.5, mi);
    fertility[c] = s.water.glacier[c] ? 0 : Math.max(0, Math.min(1, f));
  }
  return { order, fertility, saline };
}

/** Plain-language name for one cell's soil, naming what a young soil formed in. */
export function soilName(order: number, dep: number, saline: boolean): string {
  if (saline) return 'salt-crusted desert soil';
  if (order === Soil.Entisol) {
    if (dep === Deposit.Alluvium || dep === Deposit.Delta || dep === Deposit.Fan) return 'fresh alluvial soil';
    if (dep === Deposit.Dunes) return 'loose dune sand';
    if (dep === Deposit.Bedrock || dep === Deposit.Colluvium) return 'thin stony soil';
  }
  if (order === Soil.Inceptisol) {
    if (dep === Deposit.Till || dep === Deposit.Moraine || dep === Deposit.Outwash) return 'young soil on glacial drift';
    if (dep === Deposit.Bedrock || dep === Deposit.Colluvium) return 'thin mountain soil';
  }
  return SOIL_INFO[order].name;
}
