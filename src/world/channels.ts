import { nearestCell, type SphereMesh } from '../core/mesh';
import { Rng, seedFor } from '../core/rng';
import { Biome } from './biomes';
import { detailRatio, type Hydrology, Wetland } from './hydrology';
import { Deposit, type DeltaKind, type DeltaRecord, type Deposits } from './sediment';

/**
 * River channel patterns (meandering, braided, anastomosing, ...) and the
 * distributary networks of deltas, inland deltas and alluvial fans.
 *
 * This file is the contract between the world model (src/world) and the
 * renderer (src/render): the world fills `Channels`, and render/context.ts
 * copies what tiles need into FieldData. The names and meanings of the
 * exported types are fixed: add fields rather than renaming or removing them.
 */

export const enum Channel {
  /** No river channel in this cell (or too small to classify). */
  None = 0,
  /** Hemmed in by valley walls or cut into rock: steep, little or no floodplain (torrents, canyons, gorges). */
  Confined = 1,
  /** Single thread with sinuosity below ~1.3. */
  Straight = 2,
  /** Single thread, meandering freely over its floodplain (sinuosity ~1.3–3). */
  Meandering = 3,
  /** Gravel bed: one dominant thread with bars and side channels (between meandering and braided). */
  Wandering = 4,
  /** Many shifting threads across bare sand or gravel bars (high stream power, abundant bedload, weak banks). */
  Braided = 5,
  /** Several stable, sinuous channels around vegetated islands (low gradient, cohesive banks, aggrading). */
  Anastomosing = 6,
  /** Splitting downstream over a delta plain, inland delta or fan (geometry in `Channels.networks`). */
  Distributary = 7,
}

export const CHANNEL_NAMES: readonly string[] = [
  'No channel',
  'Confined (gorge or bedrock)',
  'Straight',
  'Meandering',
  'Wandering gravel-bed',
  'Braided',
  'Anastomosing',
  'Distributary (delta or fan)',
];

/**
 * Plan shape of a delta, inland delta or fan: river-dominated deltas are
 * bird's-foot (muddy, into deep water: long fingers along the distributaries)
 * or lobate (sandy or shallow); wave-dominated ones arcuate or cuspate (a
 * smooth front of beach ridges, pointed where waves win outright);
 * tide-dominated ones estuarine (funnel-shaped mouths between tidal islands).
 */
export type DeltaForm = 'birdfoot' | 'lobate' | 'arcuate' | 'cuspate' | 'estuarine' | 'fan' | 'inland';

/**
 * One delta, inland delta or fan: a tree of distributary channels from its
 * apex. Node 0 is the apex; every other node's parent comes before it, and a
 * channel runs from each node's parent to the node.
 */
export interface Distributaries {
  kind: 'delta' | 'inland' | 'fan';
  /** Sea deltas: index into world.deposits.deltas; otherwise -1. */
  delta: number;
  /** Cell at the apex, where the river first splits. */
  apex: number;
  /** Cells whose surface the delta plain or fan built. */
  cells: Int32Array;
  /** Node positions on the unit sphere (like mesh points). */
  x: Float64Array;
  y: Float64Array;
  z: Float64Array;
  /** Parent node, -1 for the apex. */
  parent: Int32Array;
  /** Bankfull discharge (m³/s) of the channel ending at each node (node 0: the river arriving at the apex). */
  q: Float32Array;
  /** 1 = a channel end: a mouth at the sea or a lake, or where a distributary dies out on a fan or inland delta. */
  mouth: Uint8Array;
  // ---- added by the channel model ----
  /** Plan shape (see DeltaForm). */
  form: DeltaForm;
  /** Bankfull width (m) of the channel ending at each node: ~4√q, widening seaward into funnels on tidal deltas. */
  width: Float32Array;
  /**
   * The delta's shoreline (fans and inland deltas: their outer margin), from
   * one flank round the front to the other, as unit-sphere x, y, z triples.
   * Seen from the apex it sweeps clockwise to counter-clockwise; the flanks
   * end on the coast (or ground) the delta grew from.
   */
  outline: Float64Array;
  /** Distance from the apex to the front along the delta's axis (km). */
  radiusKm: number;
  /** 0–1 wave energy and tidal range at the mouth (0 for fans and inland deltas); tidal creeks thread the plain in proportion to `tide`. */
  wave: number;
  tide: number;
  /** Sand-and-gravel share of the sediment that built it (0–1). */
  sand: number;
  /** Node where the trunk river leaves a fan or inland delta and runs on (-1 for sea deltas, or where every branch dies out). */
  exit: number;
}

export interface Channels {
  /** Per cell: Channel pattern of the river through the cell (None where there is no river). */
  pattern: Uint8Array;
  /** Per cell: active channels across the belt at bankfull (1 = single thread; braided ~3–12; anastomosing ~2–6). */
  threads: Float32Array;
  /** Per cell: sinuosity of the main thread (channel length / valley length, 1–3). */
  sinuosity: Float32Array;
  /** Per cell: width of the active channel belt (m): the channel itself for one thread, else the braidplain or anastomosing net. */
  belt: Float32Array;
  /** Per cell: width of the valley floor the river wanders over (m): meander belt and floodplain; 0 where confined. */
  floodplain: Float32Array;
  /** Per cell: bed grain size D50 (mm). */
  grain: Float32Array;
  /** Per cell: 0–1 bank strength (cohesive mud, roots, permafrost). */
  banks: Float32Array;
  /** Per cell: specific stream power at bankfull (W/m²). */
  power: Float32Array;
  /** Distributary networks of deltas, inland deltas and fans. */
  networks: Distributaries[];
  // ---- added by the channel model ----
  /** Per cell: index into `networks` of the delta, fan or inland delta covering it, else -1. */
  networkOf: Int32Array;
  /** Per cell: the channel's valley slope (m/m) its hydraulics were worked out on (see VALLEY_GRADE). */
  slope: Float32Array;
}

export interface ChannelInputs {
  radiusKm: number;
  seed: string;
  elev: Float32Array;
  ocean: Uint8Array;
  hydro: Hydrology;
  deposits: Deposits;
  /** Bedrock erodibility (geology.ts), and the share of it that weathers to sand (sediment.ts rockTraits). */
  erodibility: Float32Array;
  sandyRock: Float32Array;
  biome: Uint8Array;
  /** Monthly temperature (°C), c*12+m. */
  temp: Float32Array;
  precipYear: Float32Array;
  petYear: Float32Array;
  /** Sub-cell relief amplitude (m). */
  rough: Float32Array;
  /** Share of each cell under present-day ice, and how much the last ice age deepened the ground (m). */
  iceCover: Float32Array;
  iceAgeErosion: Float32Array;
  /** Smallest mean discharge (m³/s) classified: the smallest stream drawn at the deepest zoom. */
  minDischarge: number;
}

// ------------------------------------------------------------------ constants

const RHO_G = 9810;
/** Width used for potential specific stream power (Kleinhans & van den Berg 2011): 4.7 √Q_bankfull. */
const REF_WIDTH = 4.7;
/** Valley slope: the fall over about this far up and down the river (km). */
const VALLEY_HALF_KM = 50;
/**
 * Valley slopes over a fixed distance depend on the mesh: finer meshes
 * resolve more of the relief of hill country (small and middling rivers come
 * out steeper) and grade the lowland courses of big rivers lower. A valley
 * slope is converted to what the default detail would give by the detail
 * ratio to this power, by log10 of the mean discharge (fitted on two seeds at
 * 80k, 320k and 1.28M cells), so a river's pattern does not depend on the mesh.
 */
const SLOPE_DETAIL: readonly (readonly [number, number])[] = [
  [0.5, 0],
  [1.3, -0.12],
  [1.9, -0.1],
  [2.45, 0.19],
  [3.5, 0.17],
];
/** The detail the patterns are calibrated at: the default mesh (1.28M cells). */
const CALIBRATED_CELLS = 1_280_000;

function slopeDetail(q: number): number {
  const l = Math.log10(Math.max(q, 1));
  const t = SLOPE_DETAIL;
  if (l <= t[0][0]) return t[0][1];
  for (let i = 1; i < t.length; i++) {
    if (l <= t[i][0]) return t[i - 1][1] + ((t[i][1] - t[i - 1][1]) * (l - t[i - 1][0])) / (t[i][0] - t[i - 1][0]);
  }
  return t[t.length - 1][1];
}
/**
 * A river's own valley falls more gently than the cell-to-cell course the
 * mesh draws: the valley winds below the cell scale, and the model's lowland
 * rivers have not had the time Earth's have had to grade their lower courses
 * nearly flat. Hydraulics (depth, grain, stream power, backwater) use the
 * mesh's fall times this; calibrated so patterns come out in Earth's
 * proportions (braided ~10–20% of river length, meandering the most common).
 */
const VALLEY_GRADE = 0.4;
/** Manning roughness for bankfull depth. */
const MANNING_N = 0.035;
/** Bankfull Shields stress of a gravel bed over the critical one (gravel rivers sit near threshold: Parker 2007). */
const GRAVEL_SHIELDS = 0.055;
/** Below this (mm) gravel gives way to sand: the gravel–sand transition is abrupt (Ferguson 2003). */
const GRAVEL_MIN = 5;
/** Valley slope (m/m) above which a stream is a step-pool or cascade torrent. */
const TORRENT_SLOPE = 0.012;
/** Sub-cell relief (m) of mountain valleys: too narrow for a floodplain unless glaciers widened them. */
const MOUNTAIN_ROUGH = 160;
/** Specific stream power (W/m²) below which a river barely moves its banks (laterally stable: Nanson & Croke 1992). */
const STABLE_POWER = 3;
/** Anastomosing rivers are low-energy: below ~10 W/m², more for the great sandy anabranching rivers (Nanson & Knighton 1996). */
const ANASTOMOSING_POWER = 12;
/**
 * Bankfull discharge (m³/s) of the great rivers that anabranch: within a band
 * of stream power around the braiding threshold they split around vegetated
 * islands; below it they meander (the Mississippi), above it braid (the
 * Brahmaputra).
 */
const GREAT_RIVER = 5000;

/** Riparian vegetation by biome: roots and trees binding the banks (0–1). */
const VEGETATION: Record<number, number> = {
  [Biome.Ocean]: 0,
  [Biome.IceSheet]: 0,
  [Biome.Tundra]: 0.35,
  [Biome.Alpine]: 0.3,
  [Biome.Boreal]: 0.85,
  [Biome.TemperateRainforest]: 0.95,
  [Biome.TemperateForest]: 0.9,
  [Biome.Grassland]: 0.6,
  [Biome.Mediterranean]: 0.5,
  [Biome.HotDesert]: 0.1,
  [Biome.ColdDesert]: 0.15,
  [Biome.Shrubland]: 0.4,
  [Biome.Savanna]: 0.55,
  [Biome.TropicalDryForest]: 0.75,
  [Biome.Rainforest]: 0.95,
  [Biome.Wetland]: 0.9,
  [Biome.Lake]: 0.3,
  [Biome.SaltFlat]: 0.05,
};

const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);
const LN2 = Math.LN2;

/** Rock hardness 0–1 from erodibility: hard rock yields coarse, durable gravel. */
const hardness = (k: number): number => Math.max(0.05, Math.min(1, 1.25 - 0.45 * k));
/**
 * Bed of a sand-bed river (mm): fine sand from muddy country, medium-coarse
 * sand from sandy, coarser where the river is steeper (it carries the finer
 * sand on through: bed size scales with depth-slope).
 */
const sandBed = (sandShare: number, grade: number): number =>
  Math.max(0.06, Math.min(1.5, (0.07 + 0.45 * clamp01(sandShare) ** 1.3) * (Math.max(grade, 1e-6) / 2e-4) ** 0.4));
/** Bankfull channel width (m) for bank strength 0–1: cohesive, vegetated banks make narrower, deeper channels. */
export const channelWidth = (q: number, banks: number): number => 4 * (1.25 - 0.5 * banks) * Math.sqrt(Math.max(0, q));

// ------------------------------------------------------------------ the model

export function computeChannels(mesh: SphereMesh, inp: ChannelInputs): Channels {
  const { n, x, y, z } = mesh;
  const Rkm = inp.radiusKm;
  const h = inp.hydro;
  const { bankfull, discharge, dischargeMonthly, flowMonths, lake, wetland, meltShare } = h;
  const { receiver, order, count, fill } = h.routing;
  const { kind, sand } = inp.deposits;
  const { ocean, rough } = inp;
  const detail = detailRatio(n) / detailRatio(CALIBRATED_CELLS);

  const isRiver = new Uint8Array(n);
  for (let c = 0; c < n; c++) {
    isRiver[c] = !ocean[c] && lake[c] < 0 && inp.iceCover[c] < 0.5 && receiver[c] >= 0 && discharge[c] >= inp.minDischarge ? 1 : 0;
  }
  /** Distance (km) to the receiver. */
  const dRc = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    const r = receiver[c];
    if (r >= 0) dRc[c] = Math.acos(Math.min(1, x[c] * x[r] + y[c] * y[r] + z[c] * z[r])) * Rkm;
  }
  const mainDonor = new Int32Array(n).fill(-1);
  for (let c = 0; c < n; c++) {
    const r = receiver[c];
    if (!isRiver[c] || r < 0 || !isRiver[r]) continue;
    if (mainDonor[r] < 0 || discharge[c] > discharge[mainDonor[r]]) mainDonor[r] = c;
  }

  // Valley slope: the fall over ~100 km of river centered on the cell, on the
  // depression-filled surface (lakes and the sea are flat).
  const slope = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    if (!isRiver[c]) continue;
    let dDown = 0;
    let cur = c;
    while (dDown < VALLEY_HALF_KM) {
      const r = receiver[cur];
      if (r < 0) break;
      dDown += dRc[cur];
      cur = r;
      if (ocean[cur] || lake[cur] >= 0) break;
    }
    const zDown = ocean[cur] ? 0 : fill[cur];
    let dUp = 0;
    cur = c;
    while (dUp < VALLEY_HALF_KM) {
      const u = mainDonor[cur];
      if (u < 0) break;
      dUp += dRc[u];
      cur = u;
    }
    slope[c] = Math.max(1e-6, ((fill[cur] - zDown) / (1000 * Math.max(0.1, dUp + dDown))) * detail ** slopeDetail(discharge[c]));
  }

  const grade = new Float32Array(n);
  for (let c = 0; c < n; c++) grade[c] = slope[c] * VALLEY_GRADE;
  // Bankfull depth (Manning, reference width) and the climate the river runs through.
  const depth = new Float32Array(n);
  const flashN = new Float32Array(n);
  const tAnn = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    if (!isRiver[c]) continue;
    const Q = Math.max(bankfull[c], 0.1);
    depth[c] = ((MANNING_N * Q) / (4 * Math.sqrt(Q) * Math.sqrt(grade[c]))) ** 0.6;
    let mx = 0;
    let mean = 0;
    let t = 0;
    for (let m = 0; m < 12; m++) {
      const q = dischargeMonthly[c * 12 + m];
      mx = Math.max(mx, q);
      mean += q / 12;
      t += inp.temp[c * 12 + m] / 12;
    }
    tAnn[c] = t;
    // Flashy floods (monsoons, snowmelt pulses, desert storms) braid more readily.
    flashN[c] = clamp01((mean > 0 ? mx / mean - 3 : 0) / 4);
  }

  // Bed grain size: gravel from steep, rocky or glaciated headwaters is carried
  // down and worn finer (Sternberg's law, faster in soft rock) until the river
  // can no longer move it or it is worn to sand; then the bed turns to sand
  // abruptly. Tributaries bring their own gravel.
  const grain = new Float32Array(n);
  const gravelIn = new Float32Array(n);
  for (let i = count - 1; i >= 0; i--) {
    const c = order[i];
    if (!isRiver[c]) continue;
    const H = hardness(inp.erodibility[c]);
    let src = 0;
    if (slope[c] > 0.008 || rough[c] > 220) src = (12 + 80 * H) * (1 - 0.6 * inp.sandyRock[c]);
    const k = kind[c];
    if (k === Deposit.Outwash || k === Deposit.Moraine || inp.iceAgeErosion[c] > 40 || meltShare[c] > 0.5) src = Math.max(src, 45);
    const supply = Math.max(gravelIn[c], src);
    const competent = (depth[c] * grade[c] * 1000) / (1.65 * GRAVEL_SHIELDS);
    let bed = Math.min(supply, competent);
    if (bed < GRAVEL_MIN) bed = 0;
    grain[c] = bed > 0 ? bed : sandBed(sand[c], grade[c]);
    const r = receiver[c];
    if (bed > 0 && r >= 0 && isRiver[r]) {
      // Halved every ~40–200 km by wear and sorting (hard rock lasts longest).
      const d = bed * Math.exp((-LN2 * dRc[c]) / (40 + 160 * H));
      if (d > gravelIn[r]) gravelIn[r] = d;
    }
  }

  // Bank strength: mud in the floodplain, roots, frozen ground, peat.
  const banks = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    if (!isRiver[c]) continue;
    const k = kind[c];
    const alluvial = k === Deposit.Alluvium || k === Deposit.Delta || k === Deposit.Lake || k === Deposit.Peat || k === Deposit.Loess || k === Deposit.Fan || k === Deposit.Playa;
    let mud = 1 - clamp01(sand[c]);
    if (!alluvial) mud *= 0.6;
    if (grain[c] >= 2) mud *= 0.6;
    const arid = inp.petYear[c] > 0 ? inp.precipYear[c] / inp.petYear[c] : 1;
    let veg = VEGETATION[inp.biome[c]] ?? 0.5;
    // Gallery forest along rivers that flow all year through dry country.
    if (arid < 0.6 && flowMonths[c] >= 10) veg = Math.max(veg, 0.45);
    const frozen = clamp01((-2 - tAnn[c]) / 6);
    const peat = k === Deposit.Peat || wetland[c] === Wetland.Peatland ? 1 : 0;
    banks[c] = clamp01(0.08 + 0.45 * mud + 0.35 * veg + 0.2 * frozen + 0.25 * peat);
  }

  // Distance down to the river's mouth (sea or lake) and the backwater length
  // h/S there: the reach the sea level reaches up, where rivers aggrade.
  const toMouth = new Float32Array(n);
  const backwater = new Float32Array(n);
  for (let i = 0; i < count; i++) {
    const c = order[i];
    if (!isRiver[c]) continue;
    const r = receiver[c];
    if (r < 0 || ocean[r] || lake[r] >= 0 || !isRiver[r]) {
      toMouth[c] = 0.5 * dRc[c];
      backwater[c] = ocean[r] || lake[r] >= 0 ? depth[c] / grade[c] / 1000 : 0;
    } else {
      toMouth[c] = toMouth[r] + dRc[c];
      backwater[c] = backwater[r];
    }
  }

  // Nearness downstream to present-day glaciers (1 at the ice, fading over
  // ~60 km of river): their meltwater rivers braid across outwash.
  const glacial = new Float32Array(n);
  for (let i = count - 1; i >= 0; i--) {
    const c = order[i];
    if (inp.iceCover[c] > 0.2) glacial[c] = 1;
    const r = receiver[c];
    if (r >= 0 && !ocean[r]) glacial[r] = Math.max(glacial[r], glacial[c] * Math.exp(-dRc[c] / 60));
  }

  // Pattern.
  const pattern = new Uint8Array(n);
  const ratio = new Float32Array(n);
  const power = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    if (!isRiver[c]) continue;
    const Q = Math.max(bankfull[c], 0.1);
    const S = slope[c];
    const k = kind[c];
    const D = grain[c];
    const potential = (RHO_G * Q * grade[c]) / (REF_WIDTH * Math.sqrt(Q));
    power[c] = (RHO_G * Q * grade[c]) / channelWidth(Q, banks[c]);
    const arid = inp.petYear[c] > 0 ? inp.precipYear[c] / inp.petYear[c] : 1;
    // Desert and steppe rivers that run in flash floods: the floods that shape
    // their beds peak far above any monthly mean, and little binds their sand
    // (Tooth 2000): wide, shallow, braided or straight washes. The monthly
    // water balance keeps a trickle in most dry-land rivers, so small rivers
    // of dry country count as flashy whatever their months; big rivers that
    // cross deserts from wetter mountains (the Nile, the Indus) do not.
    const local = clamp01(1 - Math.log10(Math.max(1, discharge[c] / 10)) / 1.5);
    const dryFlood = Math.max(
      (flowMonths[c] <= 3 ? 1 : flowMonths[c] <= 6 ? 0.5 : 0) * clamp01((0.65 - arid) / 0.45),
      clamp01((0.5 - arid) / 0.35) * local,
    );
    // Braided above van den Berg's (1995) line, ω = 900·D50^0.42 (Kleinhans &
    // van den Berg 2011). Bank strength raises it steeply (Eaton et al. 2010:
    // roots and mud let a river keep one channel at powers that braid a bare
    // one); flashy floods, ephemeral flow and the open ground of fans and
    // sandar lower it.
    let braid = 900 * (D / 1000) ** 0.42 * (0.45 + 1.1 * banks[c]) ** 1.6 * (1 - 0.3 * flashN[c]);
    if (flowMonths[c] <= 4) braid *= 0.8;
    braid *= 1 - 0.65 * dryFlood;
    // Meltwater rivers just below glaciers: heavy bedload, daily floods and
    // bare, unstable banks (sandur).
    braid *= 1 - 0.55 * glacial[c];
    if (k === Deposit.Fan) braid *= 0.6;
    if (k === Deposit.Outwash) braid *= 0.4;
    ratio[c] = potential / braid;
    // Glacial troughs and outwash: wide valley floors, braided valley trains.
    const trough = inp.iceAgeErosion[c] > 30 || k === Deposit.Outwash || glacial[c] > 0.3;
    const rocky = k === Deposit.Bedrock || k === Deposit.Residual;
    const canyon = arid < 0.5 && inp.elev[c] > 700 && rough[c] > 70 && S > 6e-4 && (rocky || k === Deposit.Colluvium);
    const confined =
      (S > TORRENT_SLOPE && !((k === Deposit.Outwash || glacial[c] > 0.3) && S < 0.03)) || (!trough && rough[c] > MOUNTAIN_ROUGH && S > 0.0025) || (rocky && S > 0.004 && !trough) || canyon;
    const w = wetland[c];
    const aggrading =
      (toMouth[c] < backwater[c] && Q >= 50) ||
      w === Wetland.InlandDelta ||
      w === Wetland.Peatland ||
      w === Wetland.Swamp ||
      w === Wetland.Marsh ||
      k === Deposit.Peat ||
      (w === Wetland.Floodplain && S < 6e-5);
    let p: Channel;
    if (confined) p = Channel.Confined;
    // The great sand-bed rivers mostly split around vegetated islands rather
    // than braid or meander (Latrubesse 2008); only the most powerful braid.
    else if (D < 2 && Q > GREAT_RIVER && banks[c] > 0.45 && ratio[c] > 0.4 && ratio[c] < 1.4) p = Channel.Anastomosing;
    else if (ratio[c] > 1) p = Channel.Braided;
    else if (D >= 2 && ratio[c] > 0.5) p = Channel.Wandering;
    else if (D < 2 && banks[c] > 0.55 && aggrading && potential < ANASTOMOSING_POWER && Q >= 30) p = Channel.Anastomosing;
    else if (potential < STABLE_POWER && banks[c] > 0.6) p = Channel.Straight;
    // Sandy washes: braided low-water threads across a flat bed where the
    // floods have the power, else a straight, wide channel without the banks
    // to hold a meander.
    else if (dryFlood > 0.5 && banks[c] < 0.45) p = ratio[c] > 0.25 && D < 2 ? Channel.Braided : Channel.Straight;
    else p = Channel.Meandering;
    pattern[c] = p;
  }
  // A single cell of one pattern between two of another reads as noise at this scale.
  const smooth = pattern.slice();
  for (let c = 0; c < n; c++) {
    if (!isRiver[c]) continue;
    const u = mainDonor[c];
    const r = receiver[c];
    if (u < 0 || r < 0 || !isRiver[r]) continue;
    if (pattern[u] === pattern[r] && pattern[c] !== pattern[r]) smooth[c] = pattern[r];
  }
  pattern.set(smooth);

  // Distributary networks.
  const networkOf = new Int32Array(n).fill(-1);
  const networks = buildNetworks(mesh, inp, { isRiver, mainDonor, dRc, slope: grade, depth, banks, networkOf });
  const channelsIn = new Uint16Array(n);
  networks.forEach((net, i) => {
    for (const c of net.cells) {
      networkOf[c] = i;
      pattern[c] = Channel.Distributary;
    }
    // Channels crossing each cell: a branch runs from one split to the next.
    const m = net.parent.length;
    const kids = new Uint16Array(m);
    for (let j = 1; j < m; j++) kids[net.parent[j]]++;
    const branch = new Int32Array(m);
    let branches = 0;
    const seen = new Set<number>();
    let cell = net.apex;
    for (let j = 1; j < m; j++) {
      const p = net.parent[j];
      branch[j] = p > 0 && kids[p] === 1 ? branch[p] : branches++;
      cell = nearestCell(mesh, net.x[j], net.y[j], net.z[j], cell);
      const key = cell * 65536 + (branch[j] & 65535);
      if (networkOf[cell] === i && !seen.has(key)) {
        seen.add(key);
        channelsIn[cell]++;
      }
    }
  });

  // Geometry of each pattern.
  const threads = new Float32Array(n).fill(1);
  const sinuosity = new Float32Array(n).fill(1);
  const belt = new Float32Array(n);
  const floodplain = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    if (!pattern[c]) continue;
    const Q = Math.max(bankfull[c], 0.1);
    const b = banks[c];
    const W = channelWidth(Q, b);
    const rt = ratio[c];
    const open = clamp01(1 - (rough[c] - 40) / 200);
    const mud = 1 - clamp01(sand[c]);
    let k = 1;
    let P = 1;
    let B = W;
    let F = 0;
    switch (pattern[c] as Channel) {
      case Channel.Confined: {
        const canyon = (inp.petYear[c] > 0 ? inp.precipYear[c] / inp.petYear[c] : 1) < 0.5 && slope[c] < TORRENT_SLOPE;
        // Incised meanders in dry plateaus keep the bends of an older alluvial river.
        P = canyon ? 1.2 + 0.5 * mud : 1.05 + 0.1 * clamp01(1 - slope[c] / 0.05);
        break;
      }
      case Channel.Straight:
        P = 1.05 + 0.2 * b;
        F = W * (3 + 5 * open);
        break;
      case Channel.Meandering: {
        // Sinuosity rises with the silt and clay in the channel's perimeter
        // (Schumm 1963: P = 0.94·M^0.25, M the percent silt-clay, ~2–30% for
        // most rivers; the floodplain's mud share overstates it) and falls
        // toward the braiding threshold.
        const M = 100 * Math.min(0.6, 0.03 + 0.3 * mud ** 1.5 * (0.3 + 0.7 * b));
        const base = 0.94 * M ** 0.25;
        P = Math.max(1.3, Math.min(3, 1 + (base - 1) * (1 - 0.45 * clamp01(rt))));
        // Meander belt about ten widths per unit of extra sinuosity; the floodplain spreads wider in open country.
        F = W * (2 + 10 * (P - 1)) * (1.5 + 3 * open);
        break;
      }
      case Channel.Wandering:
        k = Math.max(1.5, Math.min(2.5, 1.5 + rt));
        P = 1.15 + 0.25 * b;
        B = W * (1 + 1.2 * (k - 1));
        F = Math.max(2 * B, W * (2 + 10 * (P - 1))) * (1.2 + 2 * open);
        break;
      case Channel.Braided: {
        // Braid index grows with stream power over the threshold, and with size.
        const sized = 3 + 3 * Math.log10(Math.max(1, Q / 10));
        k = Math.max(3, Math.min(12, sized, 2.5 + 4 * Math.log(Math.max(1, rt))));
        P = 1.08 + 0.2 * b;
        // The braidplain spans about 2.5 times the threads' combined width
        // (k threads sharing the flow are ~W/√k wide each): the Platte's ~0.6 km
        // at a few hundred m³/s, the Brahmaputra's ~10 km at ~60,000.
        B = 2.5 * W * Math.sqrt(k);
        F = B * (1.3 + 1.5 * open);
        break;
      }
      case Channel.Anastomosing: {
        const agg = clamp01(backwater[c] > 0 ? 1 - toMouth[c] / backwater[c] : 0.5);
        k = Math.max(2, Math.min(6, 2 + 1.2 * Math.log10(Math.max(1, Q / 100)) + 1.5 * agg));
        P = 1.2 + 0.8 * b;
        // Channels and the islands between them: a few kilometres for a large
        // river, ~10–20 km for the greatest (the Negro's and the Amazon's).
        B = W * (1 + k) * (1 + 0.5 * agg);
        F = B * (1.5 + 2 * open);
        break;
      }
      case Channel.Distributary:
        k = Math.max(1, channelsIn[c]);
        P = 1.15;
        B = W * k;
        F = 3 * B;
        break;
    }
    threads[c] = k;
    sinuosity[c] = P;
    belt[c] = B;
    floodplain[c] = F;
  }

  return { pattern, threads, sinuosity, belt, floodplain, grain, banks, power, networks, networkOf, slope: grade };
}

// ------------------------------------------------------------------ distributary networks

interface Shared {
  isRiver: Uint8Array;
  mainDonor: Int32Array;
  dRc: Float32Array;
  slope: Float32Array;
  depth: Float32Array;
  banks: Float32Array;
  networkOf: Int32Array;
}

/** Smallest delta plain (km²) worth a network of distributaries, and the smallest river (bankfull m³/s). */
const MIN_DELTA_KM2 = 40;
const MIN_DELTA_Q = 150;
/** Smallest fan (km², a large fan: smaller ones are below the cells the deposits are mapped on, and a finer mesh would find ever more) and river (bankfull m³/s) with a network. */
const MIN_FAN_KM2 = 1150;
const MIN_FAN_Q = 60;
/**
 * Smallest catchment (km²) above a fan's apex. Fans are counted by the rivers
 * that leave the hills, and a finer mesh resolves more small ones; with this
 * floor (and MIN_FAN_Q) a world has about as many fans at any detail.
 */
const MIN_FAN_CATCHMENT = 2500;
/** Smallest inland delta (km²) and river (bankfull m³/s) with a network: the Okavango's is ~15,000 km². */
const MIN_INLAND_KM2 = 5000;
const MIN_INLAND_Q = 150;
/** Distributaries split about this many channel widths apart (mouth-bar spacing: Edmonds & Slingerland 2007). */
const SPLIT_WIDTHS = 25;

/**
 * A local map around an apex: gnomonic, so straight lines in it are great
 * circles. u runs along the axis (seaward or down-fan), v to its left; km.
 */
class Frame {
  ox: number;
  oy: number;
  oz: number;
  ex: number;
  ey: number;
  ez: number;
  nx: number;
  ny: number;
  nz: number;
  constructor(
    o: [number, number, number],
    toward: [number, number, number],
    readonly R: number,
  ) {
    [this.ox, this.oy, this.oz] = o;
    const d = toward[0] * o[0] + toward[1] * o[1] + toward[2] * o[2];
    let ex = toward[0] - d * o[0];
    let ey = toward[1] - d * o[1];
    let ez = toward[2] - d * o[2];
    let l = Math.hypot(ex, ey, ez);
    if (!(l > 1e-12)) {
      // Any tangent will do.
      ex = -o[1];
      ey = o[0];
      ez = 0;
      l = Math.hypot(ex, ey, ez) || 1;
      if (l < 1e-9) [ex, ey, ez, l] = [1, 0, 0, 1];
    }
    this.ex = ex / l;
    this.ey = ey / l;
    this.ez = ez / l;
    this.nx = o[1] * this.ez - o[2] * this.ey;
    this.ny = o[2] * this.ex - o[0] * this.ez;
    this.nz = o[0] * this.ey - o[1] * this.ex;
  }
  xyz(u: number, v: number, out: Float64Array, i: number): void {
    const px = this.ox + (u * this.ex + v * this.nx) / this.R;
    const py = this.oy + (u * this.ey + v * this.ny) / this.R;
    const pz = this.oz + (u * this.ez + v * this.nz) / this.R;
    const l = Math.hypot(px, py, pz);
    out[i] = px / l;
    out[i + 1] = py / l;
    out[i + 2] = pz / l;
  }
  uv(px: number, py: number, pz: number): [number, number] {
    const d = px * this.ox + py * this.oy + pz * this.oz;
    if (d <= 1e-6) return [0, 0];
    return [((px * this.ex + py * this.ey + pz * this.ez) / d) * this.R, ((px * this.nx + py * this.ny + pz * this.nz) / d) * this.R];
  }
}

/** A tree under construction, in a frame's u, v (km). */
class Tree {
  u: number[] = [];
  v: number[] = [];
  parent: number[] = [];
  q: number[] = [];
  mouth: number[] = [];
  add(u: number, v: number, parent: number, q: number, mouth = 0): number {
    this.u.push(u);
    this.v.push(v);
    this.parent.push(parent);
    this.q.push(q);
    this.mouth.push(mouth);
    return this.u.length - 1;
  }
}

/**
 * Drops each branch whose mouth `stranded` rejects, from the split it left to
 * its mouth, handing its water to the branches that split with it (scaling
 * their whole subtrees, so discharge is conserved at every split). The trunk
 * itself, with no split above the mouth, is kept.
 */
function pruneStranded(tree: Tree, stranded: (j: number) => boolean): void {
  const m = tree.u.length;
  const kids = new Int32Array(m);
  for (let j = 1; j < m; j++) kids[tree.parent[j]]++;
  const drop = new Uint8Array(m);
  let dropped = false;
  for (let j = 0; j < m; j++) {
    if (!tree.mouth[j] || drop[j] || !stranded(j)) continue;
    let s = j;
    while (tree.parent[s] > 0 && kids[tree.parent[s]] === 1) s = tree.parent[s];
    const split = tree.parent[s];
    if (split < 0 || kids[split] < 2) continue;
    // The chain s … j has one child per node: drop it.
    for (let t = j; ; t = tree.parent[t]) {
      drop[t] = 1;
      if (t === s) break;
    }
    kids[split]--;
    const qSplit = tree.q[split];
    const f = qSplit / Math.max(1e-9, qSplit - tree.q[s]);
    const under = new Uint8Array(m);
    under[split] = 1;
    for (let t = split + 1; t < m; t++) {
      const p = tree.parent[t];
      if (p >= 0 && under[p] && !drop[t]) {
        under[t] = 1;
        tree.q[t] *= f;
      }
    }
    dropped = true;
  }
  if (!dropped) return;
  const index = new Int32Array(m).fill(-1);
  let k = 0;
  for (let j = 0; j < m; j++) if (!drop[j]) index[j] = k++;
  const keep = (a: number[]): number[] => a.filter((_, j) => !drop[j]);
  tree.u = keep(tree.u);
  tree.v = keep(tree.v);
  tree.q = keep(tree.q);
  tree.mouth = keep(tree.mouth);
  tree.parent = keep(tree.parent).map((p) => (p < 0 ? -1 : index[p]));
}

/**
 * The plan of a delta or fan as seen from its apex: the distance R(θ) (km)
 * from the apex to its front at bearing θ from the axis, on a grid of bearings.
 */
class Front {
  readonly r: Float64Array;
  constructor(
    readonly half: number,
    readonly steps: number,
  ) {
    this.r = new Float64Array(2 * steps + 1);
  }
  theta(j: number): number {
    return (this.half * (j - this.steps)) / this.steps;
  }
  at(theta: number): number {
    const t = ((theta / this.half) * this.steps + this.steps) | 0;
    const f = (theta / this.half) * this.steps + this.steps - t;
    const a = this.r[Math.max(0, Math.min(this.r.length - 1, t))];
    const b = this.r[Math.max(0, Math.min(this.r.length - 1, t + 1))];
    return a + (b - a) * Math.max(0, Math.min(1, f));
  }
}

function buildNetworks(mesh: SphereMesh, inp: ChannelInputs, sh: Shared): Distributaries[] {
  const { n, x, y, z, area, nbrStart, nbr, nbrDist } = mesh;
  const Rkm = inp.radiusKm;
  const { ocean } = inp;
  const { kind } = inp.deposits;
  const { bankfull, lake, wetland } = inp.hydro;
  const { receiver } = inp.hydro.routing;
  const km2 = (c: number): number => area[c] * Rkm * Rkm;
  const spacingKm = mesh.spacing * Rkm;
  const out: Distributaries[] = [];
  const taken = sh.networkOf;
  const tmp = new Float64Array(3);
  const at = (c: number): [number, number, number] => [x[c], y[c], z[c]];
  const mid = (a: number, b: number, f = 0.5): [number, number, number] => {
    const px = x[a] + (x[b] - x[a]) * f;
    const py = y[a] + (y[b] - y[a]) * f;
    const pz = z[a] + (z[b] - z[a]) * f;
    const l = Math.hypot(px, py, pz);
    return [px / l, py / l, pz / l];
  };

  /** Ray from the apex at bearing θ: distance (km) to the first ground `inside` rejects, up to `max`. */
  const reach = (fr: Frame, theta: number, max: number, start: number, inside: (c: number) => boolean): number => {
    const step = Math.max(0.5, spacingKm / 8);
    const cu = Math.cos(theta);
    const sv = Math.sin(theta);
    let cell = start;
    for (let s = step; s <= max; s += step) {
      fr.xyz(s * cu, s * sv, tmp, 0);
      cell = nearestCell(mesh, tmp[0], tmp[1], tmp[2], cell);
      if (!inside(cell)) return s - 0.5 * step;
    }
    return max;
  };

  // ---- sea deltas
  const deltas = inp.deposits.deltas;
  // Where each land cell's water reaches the sea: its river's last land cell
  // (-1 for water that never does). A delta keeps to its own river's ground:
  // another sizeable river reaching the sea elsewhere bounds it, so its
  // channels never spread over a neighbouring basin and cross that river.
  const outMouth = new Int32Array(n).fill(-1);
  {
    const { order, count } = inp.hydro.routing;
    for (let i = 0; i < count; i++) {
      const c = order[i];
      if (ocean[c]) continue;
      const r = receiver[c];
      outMouth[c] = r < 0 ? -1 : ocean[r] ? c : outMouth[r];
    }
  }
  /** Ground a delta of river L (bankfull q) may not spread over: another sizeable river's course. */
  const barrierFor = (L: number, q: number): ((c: number) => boolean) => {
    const qBar = Math.max(MIN_DELTA_Q, 0.15 * q);
    return (c) => bankfull[c] >= qBar && outMouth[c] !== L && !ocean[c];
  };
  // Delta plain cells belong to the nearest delta outlet across the plain.
  const owner = new Int32Array(n).fill(-1);
  const best = new Float64Array(n).fill(Infinity);
  {
    const heap: [number, number][] = [];
    const push = (d: number, c: number): void => {
      heap.push([d, c]);
      let i = heap.length - 1;
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (heap[p][0] <= heap[i][0]) break;
        [heap[p], heap[i]] = [heap[i], heap[p]];
        i = p;
      }
    };
    const pop = (): [number, number] => {
      const top = heap[0];
      const last = heap.pop()!;
      if (heap.length > 0) {
        heap[0] = last;
        let i = 0;
        for (;;) {
          const l = 2 * i + 1;
          const r = l + 1;
          let m = i;
          if (l < heap.length && heap[l][0] < heap[m][0]) m = l;
          if (r < heap.length && heap[r][0] < heap[m][0]) m = r;
          if (m === i) break;
          [heap[m], heap[i]] = [heap[i], heap[m]];
          i = m;
        }
      }
      return top;
    };
    deltas.forEach((d, i) => {
      if (d.river < 0 || ocean[d.river]) return;
      best[d.river] = 0;
      owner[d.river] = i;
      push(0, d.river);
    });
    const qBarOf = deltas.map((d) => (d.river >= 0 ? Math.max(MIN_DELTA_Q, 0.15 * bankfull[d.river]) : Infinity));
    while (heap.length) {
      const [dist, c] = pop();
      if (dist > best[c]) continue;
      const o = owner[c];
      const Lo = deltas[o].river;
      for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
        const m = nbr[k];
        if (ocean[m] || lake[m] >= 0 || kind[m] !== Deposit.Delta) continue;
        // (Not across another sizeable river: the plain beyond is that river's.)
        if (bankfull[m] >= qBarOf[o] && outMouth[m] !== Lo) continue;
        const dm = dist + nbrDist[k] * Rkm;
        if (dm < best[m] && dm < 700) {
          best[m] = dm;
          owner[m] = owner[c];
          push(dm, m);
        }
      }
    }
  }
  const plainCells: number[][] = deltas.map(() => []);
  for (let c = 0; c < n; c++) if (owner[c] >= 0 && (kind[c] === Deposit.Delta || c === deltas[owner[c]].river)) plainCells[owner[c]].push(c);

  deltas.forEach((d, i) => {
    const L = d.river;
    const M = d.mouth;
    if (L < 0 || M < 0 || ocean[L] || !ocean[M]) return;
    const Q = bankfull[L];
    const cells = plainCells[i];
    let meshKm2 = 0;
    for (const c of cells) if (kind[c] === Deposit.Delta) meshKm2 += km2(c);
    // A delta smaller than a cell: its plain from its volume (a few metres of
    // topset over the shallow water it filled), the rest is its front.
    const subKm2 = Math.min(0.9 * km2(L), (d.front * 1000) / 25 + (d.plain * 1000) / 6);
    const A = Math.max(meshKm2, subKm2);
    if (A < MIN_DELTA_KM2 || Q < MIN_DELTA_Q) return;
    const sandShare = d.supply > 0 ? Math.min(1, d.sand / d.supply) : 0.4;
    const form = deltaForm(d, sandShare, inp.elev, mesh, M);
    const rng = new Rng(seedFor(inp.seed, `delta:${L}`));
    const net = growDelta(d, i, form, A, meshKm2 > 0 ? cells : [L], sandShare, rng);
    if (net) out.push(net);
  });

  function growDelta(d: DeltaRecord, index: number, form: DeltaForm, A: number, cells: number[], sandShare: number, rng: Rng): Distributaries | null {
    const L = d.river;
    const M = d.mouth;
    const Q = bankfull[L];
    const inPlain = new Set(cells);
    // Apex: up the river across the plain, no farther than the backwater length.
    const Lb = Math.max(0.3 * Math.sqrt(A), sh.depth[L] / sh.slope[L] / 1000);
    let apexCell = L;
    let travelled = 0.5 * sh.dRc[L];
    let apex = mid(L, M, 0.5);
    for (;;) {
      const u = sh.mainDonor[apexCell];
      if (u < 0) {
        apex = at(apexCell);
        break;
      }
      const leg = sh.dRc[u];
      if (!inPlain.has(u) || travelled + leg > Lb) {
        const f = Math.max(0, Math.min(1, (Lb - travelled) / leg));
        // Within the plain the apex lies at its landward edge (half way to the next cell up).
        apex = mid(apexCell, u, inPlain.has(u) ? f : Math.min(0.5, f));
        if (apexCell === L && !inPlain.has(u)) {
          // A delta smaller than its cell: the apex sits back from the coast by about its radius.
          const r = Math.sqrt(A / 1.2);
          const back = Math.min(0.95, r / Math.max(1, 0.5 * sh.dRc[L]));
          apex = mid(M, L, 0.5 + 0.5 * back);
        }
        break;
      }
      travelled += leg;
      apexCell = u;
    }
    // Axis: from the apex toward the plain's sea front (or the mouth).
    let tx = 0;
    let ty = 0;
    let tz = 0;
    let wsum = 0;
    for (const c of cells) {
      for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
        const m = nbr[k];
        if (!ocean[m]) continue;
        const p = mid(c, m);
        tx += p[0];
        ty += p[1];
        tz += p[2];
        wsum++;
      }
    }
    // (Mostly toward the river's own mouth: the delta grows around where the
    // river reaches the sea, leaning a little toward the middle of its plain.)
    const toward: [number, number, number] = wsum > 0 ? [0.75 * x[M] + (0.25 * tx) / wsum, 0.75 * y[M] + (0.25 * ty) / wsum, 0.75 * z[M] + (0.25 * tz) / wsum] : at(M);
    const fr = new Frame(apex, toward, Rkm);
    const apexAt = nearestCell(mesh, apex[0], apex[1], apex[2], L);
    // Where the coast lies now, bearing by bearing. A bearing that meets
    // another sizeable river before the sea is closed, like one that meets a
    // headland: the delta keeps to its own river's ground.
    const spread = form === 'birdfoot' ? 0.75 : form === 'estuarine' ? 0.8 : form === 'cuspate' ? 1.1 : 1.0;
    const half = Math.min(1.45, spread + 0.35);
    const front = new Front(half, 24);
    const maxR = 3 * Math.sqrt(A) + 2 * spacingKm;
    const blocked = barrierFor(L, Q);
    /** Bearings whose way to the sea crosses another sizeable river (or that meet no sea at all). */
    const closedJ = new Uint8Array(front.r.length);
    const reachSea = (j: number): number => {
      const theta = front.theta(j);
      const st = Math.max(0.5, spacingKm / 8);
      let cell = apexAt;
      let crossed = false;
      for (let s = st; s <= maxR; s += st) {
        fr.xyz(s * Math.cos(theta), s * Math.sin(theta), tmp, 0);
        cell = nearestCell(mesh, tmp[0], tmp[1], tmp[2], cell);
        if (ocean[cell]) {
          closedJ[j] = crossed ? 1 : 0;
          return s - 0.5 * st;
        }
        if (!crossed && blocked(cell)) crossed = true;
      }
      closedJ[j] = 1;
      return maxR;
    };
    for (let j = 0; j < front.r.length; j++) front.r[j] = reachSea(j);
    // The way the river arrives at the apex, in the frame (bearing 0 = the axis).
    let psiIn = 0;
    {
      const up = sh.mainDonor[apexCell];
      if (up >= 0) {
        const [uu, vu] = fr.uv(x[up], y[up], z[up]);
        const [ua, va] = fr.uv(x[apexCell], y[apexCell], z[apexCell]);
        if (Math.hypot(ua - uu, va - vu) > 1e-6) psiIn = Math.max(-1.1, Math.min(1.1, Math.atan2(va - vu, ua - uu)));
      }
    }
    // Bearings whose rays meet no sea (the coast curving away, a headland)
    // are closed: the delta spreads only over the open ones around its axis,
    // and its flanks keep to the coast at the open range's edges.
    const reaches = (j: number): boolean => front.r[j] < maxR && closedJ[j] === 0;
    let jc = front.steps;
    if (!reaches(jc)) {
      jc = -1;
      for (let o = 1; o <= front.steps && jc < 0; o++) {
        if (reaches(front.steps - o)) jc = front.steps - o;
        else if (reaches(front.steps + o)) jc = front.steps + o;
      }
      if (jc < 0) return null;
    }
    // Open bearings meet the sea about where a straight coast through the
    // nearest front would (r ≈ r₀/cos θ): farther than that, the ray runs
    // along the coast or across a headland to some other shore.
    const r0 = front.r[jc] + 0.5 * spacingKm;
    const th0 = front.theta(jc);
    const isOpen = (j: number): boolean => reaches(j) && front.r[j] < (1.4 * r0) / Math.max(0.2, Math.cos(front.theta(j) - th0));
    let jlo = jc;
    let jhi = jc;
    while (jlo > 0 && isOpen(jlo - 1)) jlo--;
    while (jhi < front.r.length - 1 && isOpen(jhi + 1)) jhi++;
    // (Closed bearings keep to the coast that is there, never past it.)
    for (let j = 0; j < jlo; j++) front.r[j] = Math.min(front.r[j], front.r[jlo]);
    for (let j = jhi + 1; j < front.r.length; j++) front.r[j] = Math.min(front.r[j], front.r[jhi]);
    smoothFront(front.r, 2);
    const thLo = front.theta(jlo);
    const thHi = front.theta(jhi);
    // The built-out front meets the coast smoothly at the edges of the open
    // bearings (no wedge with straight flanks), and is never a perfect curve:
    // a few broad lobes and embayments of its own.
    const edgeW = Math.max(0.06, 0.2 * (thHi - thLo));
    const ease = (x: number): number => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));
    const ph1 = 2 * Math.PI * rng.next();
    const ph2 = 2 * Math.PI * rng.next();
    const ph3 = 2 * Math.PI * rng.next();
    const wob = (t: number): number => 1 + 0.35 * (0.5 * Math.sin(5 * t + ph1) + 0.3 * Math.sin(11 * t + ph2) + 0.2 * Math.sin(23 * t + ph3));
    // Build out the front: a delta smaller than the cells progrades past the
    // drawn coast; one the mesh resolves already has its plain in the cells.
    const base = new Front(half, 24);
    const shape = (t: number): number => {
      const a = Math.abs(t) / spread;
      if (a >= 1 || t < thLo || t > thHi) return 0;
      const taper = ease((t - thLo) / edgeW) * ease((thHi - t) / edgeW) * wob(t);
      switch (form) {
        case 'cuspate':
          return taper * (1 - a) ** 1.3;
        case 'arcuate':
          return taper * Math.sqrt(1 - a * a);
        case 'birdfoot':
          return taper * 0.5 * Math.cos((a * Math.PI) / 2);
        case 'estuarine':
          return taper * Math.cos((a * Math.PI) / 2) ** 0.4;
        default:
          return taper * Math.cos((a * Math.PI) / 2) ** 0.7;
      }
    };
    // Plain area for a progradation P0: ∫ ((C + P0 s)² − C²)/2 dθ = a2 P0² + a1 P0.
    let a1 = 0;
    let a2 = 0;
    const dth = front.half / front.steps;
    for (let j = 0; j < front.r.length; j++) {
      const s = shape(front.theta(j));
      a1 += front.r[j] * s * dth;
      a2 += 0.5 * s * s * dth;
    }
    const needed = Math.max(0.15 * A, A - meshArea(cells));
    const P0 = a2 > 0 ? (-a1 + Math.sqrt(a1 * a1 + 4 * a2 * needed)) / (2 * a2) : 0;
    for (let j = 0; j < base.r.length; j++) base.r[j] = front.r[j] + P0 * shape(front.theta(j));
    const R0 = base.at(0);

    // Number of mouths by type and size.
    let N: number;
    if (form === 'lobate') N = Math.round(3 + 2.2 * Math.log2(1 + A / 150));
    else if (form === 'birdfoot') N = Math.round(2 + 1.2 * Math.log2(1 + A / 300));
    else if (form === 'estuarine') N = Math.round(3 + 1.5 * Math.log2(1 + A / 400));
    else N = d.wave > 0.6 ? 1 : 2 + (rng.next() < 0.4 ? 1 : 0);
    // The bearings it may spread over: its own, cut to those open to the sea.
    let ta = Math.max(-spread * 0.95, thLo + 0.02);
    let tb = Math.min(spread * 0.95, thHi - 0.02);
    if (tb - ta < 0.05) {
      const m = 0.5 * (Math.max(-spread, thLo) + Math.min(spread, thHi));
      ta = m - 0.025;
      tb = m + 0.025;
    }
    // A delta squeezed into part of its fan has fewer mouths.
    N = Math.round(N * Math.min(1, (tb - ta) / (1.9 * spread)) ** 0.5);
    N = Math.max(1, Math.min(N, form === 'lobate' ? 20 : 10, Math.floor(Q / 40)));

    const tree = new Tree();
    tree.add(0, 0, -1, Q);
    const qSplit = Q / N;
    const W = (q: number): number => channelWidth(q, sh.banks[L]) / 1000;
    const step = Math.max(0.2, Math.min(spacingKm / 4, R0 / 8));
    const tipOf = (theta: number, q: number): number => {
      const r = base.at(theta);
      // Bird's-foot passes run out on their own levees, the biggest farthest
      // (the Mississippi's reach some 20–30 km past its marsh, a tenth or so
      // of the delta's size).
      if (form === 'birdfoot') return r + 0.04 * R0 + 0.1 * R0 * Math.sqrt(q / Q);
      if (form === 'lobate') return r * (1 + 0.06 * Math.sqrt(q / Q));
      return r;
    };
    /** Grows a channel from node `from` heading ψ inside bearings [ta, tb] down to the front or its next split. */
    const grow = (from: number, psi: number, ta: number, tb: number, q: number, depth: number, now = false, first = 0): void => {
      let u = tree.u[from];
      let v = tree.v[from];
      let node = from;
      const centre = 0.5 * (ta + tb);
      const margin = Math.min(0.04, 0.2 * (tb - ta));
      const split = q >= 1.6 * qSplit && depth < 7 && N > 1;
      const segment = first > 0 ? first : Math.max(4 * step, Math.min(0.4 * R0, (SPLIT_WIDTHS * W(q)) * (1 + 0.5 * rng.next())));
      let run = 0;
      for (let guard = 0; guard < 400 && !(split && now); guard++) {
        const lo = ta + margin;
        const hi = tb - margin;
        // The trunk eases round from the way its river arrives toward the
        // middle of its bearings (no hairpin at the apex)...
        const easing = from === 0 && run < trunkEase;
        if (easing) {
          psi += Math.max(-0.07, Math.min(0.07, centre - psi)) + 0.05 * (rng.next() - 0.5);
        } else {
          // ...and every channel steers toward the middle of its own bearings,
          // wandering a little as it goes (channels are never ruled lines);
          // just past a split it starts outside them, and turns in.
          const d = centre - psi;
          psi += Math.max(-0.12, Math.min(0.12, d)) * 0.2 + 0.12 * (rng.next() - 0.5);
          const thHere = Math.atan2(v, u);
          if (thHere < lo) psi += Math.min(0.2, 3 * (lo - thHere) + 0.04);
          else if (thHere > hi) psi -= Math.min(0.2, 3 * (thHere - hi) + 0.04);
        }
        let nu = u + step * Math.cos(psi);
        let nv = v + step * Math.sin(psi);
        // Kept to its bearings (within a little), so channels never cross.
        let th = Math.atan2(nv, nu);
        const tol = Math.min(0.1, 0.5 * (tb - ta));
        if (!easing && (th < lo - tol || th > hi + tol)) {
          th = Math.max(lo - tol, Math.min(hi + tol, th));
          const rr = Math.hypot(nu, nv);
          nu = rr * Math.cos(th);
          nv = rr * Math.sin(th);
        }
        const rr = Math.hypot(nu, nv);
        const tip = tipOf(th, q);
        if (rr >= tip) {
          // The front: a mouth, a little out into the sea.
          const f = tip / Math.max(rr, 1e-9);
          tree.add(nu * f + 0.3 * step * Math.cos(th), nv * f + 0.3 * step * Math.sin(th), node, q, 1);
          return;
        }
        node = tree.add(nu, nv, node, q);
        u = nu;
        v = nv;
        run += step;
        if (split && run >= segment && tip - rr > 2 * step) break;
      }
      if (!split) {
        tree.mouth[node] = 1;
        return;
      }
      // Mouth-bar bifurcation: an uneven split, the bigger branch turning less.
      const f = 0.55 + 0.15 * rng.next();
      // (Distributaries part at modest angles, 20–45° between them.)
      const angle = (0.35 + 0.45 * rng.next()) * (form === 'estuarine' ? 0.7 : 1);
      const left = rng.next() < 0.5;
      const cut = left ? tb - (tb - ta) * f : ta + (tb - ta) * f;
      // The main branch takes the larger side of the bearings.
      const [mainLo, mainHi, minorLo, minorHi] = left ? [cut, tb, ta, cut] : [ta, cut, cut, tb];
      const sgn = left ? 1 : -1;
      grow(node, psi + sgn * angle * (1 - f), mainLo, mainHi, q * f, depth + 1);
      grow(node, psi - sgn * angle * f, minorLo, minorHi, q * (1 - f), depth + 1);
    };
    // The river splits at the apex itself.
    // The trunk runs on across the upper plain before it first splits: far
    // for a bird's foot (the Mississippi's passes part near its mouth), less
    // for lobes and wave-built fronts, hardly at all where tides spread its
    // channels from the apex.
    const firstRun = (form === 'birdfoot' ? 0.5 : form === 'estuarine' ? 0.06 : form === 'lobate' ? 0.2 : 0.3) * R0;
    const trunkEase = Math.max(3 * step, 0.3 * R0);
    if (N <= 1) grow(0, psiIn, ta, tb, Q, 99);
    else grow(0, psiIn, ta, tb, Q, 0, false, Math.max(2 * step, firstRun));

    // Every mouth must reach the sea: a little farther out at most.
    const push = Math.max(3 * step, 0.25 * R0);
    for (let j = 0; j < tree.u.length; j++) {
      if (!tree.mouth[j]) continue;
      const th = Math.atan2(tree.v[j], tree.u[j]);
      let r = Math.hypot(tree.u[j], tree.v[j]);
      fr.xyz(tree.u[j], tree.v[j], tmp, 0);
      let cell = nearestCell(mesh, tmp[0], tmp[1], tmp[2], L);
      for (let g = 0; g < 64 && !ocean[cell] && g * step < push; g++) {
        r += step;
        tree.u[j] = r * Math.cos(th);
        tree.v[j] = r * Math.sin(th);
        fr.xyz(tree.u[j], tree.v[j], tmp, 0);
        cell = nearestCell(mesh, tmp[0], tmp[1], tmp[2], cell);
      }
    }
    // A mouth the push could not bring to the sea (a headland ahead) turns
    // aside to the nearest open water along the front.
    for (let j = 0; j < tree.u.length; j++) {
      if (!tree.mouth[j]) continue;
      fr.xyz(tree.u[j], tree.v[j], tmp, 0);
      if (ocean[nearestCell(mesh, tmp[0], tmp[1], tmp[2], L)]) continue;
      const th0 = Math.atan2(tree.v[j], tree.u[j]);
      const r0 = Math.hypot(tree.u[j], tree.v[j]);
      let found = false;
      for (let k = 1; k <= 12 && !found; k++) {
        for (const sg of [1, -1]) {
          const th = th0 + sg * 0.04 * k;
          for (let rr = r0; rr <= r0 + 2 * push && !found; rr += step) {
            fr.xyz(rr * Math.cos(th), rr * Math.sin(th), tmp, 0);
            if (ocean[nearestCell(mesh, tmp[0], tmp[1], tmp[2], L)]) {
              tree.u[j] = rr * Math.cos(th);
              tree.v[j] = rr * Math.sin(th);
              found = true;
            }
          }
          if (found) break;
        }
      }
    }
    // A mouth still short of the sea (a bay's far shore in the way): its
    // branch is dropped and its water joins the branches it split from.
    pruneStranded(tree, (j) => {
      fr.xyz(tree.u[j], tree.v[j], tmp, 0);
      return !ocean[nearestCell(mesh, tmp[0], tmp[1], tmp[2], L)];
    });

    // The shoreline: the built-out front, with lobes at the mouths of a
    // lobate delta, fingers along a bird's foot's passes, and funnels where
    // a tidal delta's channels open to the sea.
    const outline: number[] = [];
    const mouths: { th: number; r: number; w: number }[] = [];
    for (let j = 0; j < tree.u.length; j++) {
      if (!tree.mouth[j]) continue;
      const r = Math.hypot(tree.u[j], tree.v[j]);
      mouths.push({ th: Math.atan2(tree.v[j], tree.u[j]), r, w: W(tree.q[j]) });
    }
    const K = 4 * front.steps + 1;
    for (let j = 0; j < K; j++) {
      const th = front.half * ((2 * j) / (K - 1) - 1);
      let r = base.at(th);
      for (const m of mouths) {
        const dth = th - m.th;
        if (form === 'birdfoot') {
          // A finger of levee land a few channel widths either side of the pass.
          const halfW = (3 * m.w + 0.02 * R0) / Math.max(m.r, 1e-3);
          const g = Math.exp(-((dth / halfW) ** 2));
          r = Math.max(r, r + (m.r - r) * g);
        } else if (form === 'lobate') {
          const g = Math.exp(-((dth / ((6 * m.w + 0.05 * R0) / Math.max(m.r, 1e-3))) ** 2));
          r = Math.max(r, r + (m.r - r) * g);
        } else if (form === 'estuarine') {
          // (Each funnel its own size: no row of matching scallops.)
          const hm = Math.abs(Math.sin(m.th * 91.7 + m.r * 0.37) * 43758.5453) % 1;
          const g = Math.exp(-((dth / (((2.5 + 4 * hm) * m.w + 0.03 * R0) / Math.max(m.r, 1e-3))) ** 2));
          r -= (0.1 + 0.25 * hm) * R0 * d.tide * g;
        } else {
          // Arcs and cusps: the front reaches out to each mouth.
          const g = Math.exp(-((dth / ((5 * m.w + 0.05 * R0) / Math.max(m.r, 1e-3))) ** 2));
          r = Math.max(r, r + (m.r - 0.5 * step - r) * g);
        }
      }
      // Past the delta's flanks the shoreline is the coast it grew from; on
      // bearings closed to the sea it turns back to the coast at the edge of
      // the open ones.
      const thc = Math.max(thLo, Math.min(thHi, th));
      const edge = front.at(th > 0 ? spread : -spread);
      const fl = thc !== th ? front.at(thc) : Math.abs(th) > spread ? Math.min(front.at(th), 1.2 * edge) : r;
      fr.xyz(fl * Math.cos(thc), fl * Math.sin(thc), tmp, 0);
      outline.push(tmp[0], tmp[1], tmp[2]);
    }

    const net = finish(tree, fr, 'delta', index, nearestCell(mesh, apex[0], apex[1], apex[2], L), form, R0, d.wave, d.tide, sandShare, new Float64Array(outline), -1, (j, q) => {
      if (form !== 'estuarine') return channelWidth(q, sh.banks[L]);
      // Tidal channels widen seaward (exponential funnels).
      const rr = Math.hypot(tree.u[j], tree.v[j]) / Math.max(R0, 1e-3);
      return channelWidth(q, sh.banks[L]) * (1 + 4 * d.tide * Math.max(0, rr) ** 2);
    });
    // The plain, and any land the channels cross.
    const own = new Set(cells);
    let cell = L;
    for (let j = 0; j < net.parent.length; j++) {
      cell = nearestCell(mesh, net.x[j], net.y[j], net.z[j], cell);
      if (!ocean[cell] && lake[cell] < 0) own.add(cell);
    }
    net.cells = Int32Array.from([...own].filter((c) => taken[c] < 0));
    for (const c of net.cells) taken[c] = out.length;
    return net;
  }

  function meshArea(cells: number[]): number {
    let a = 0;
    for (const c of cells) if (kind[c] === Deposit.Delta) a += km2(c);
    return a;
  }

  /** Converts a tree in a frame to a network. */
  function finish(
    tree: Tree,
    fr: Frame,
    k: Distributaries['kind'],
    delta: number,
    apex: number,
    form: DeltaForm,
    radiusKm: number,
    wave: number,
    tide: number,
    sandShare: number,
    outline: Float64Array,
    exit: number,
    width: (j: number, q: number) => number,
  ): Distributaries {
    const m = tree.u.length;
    const nx = new Float64Array(m);
    const ny = new Float64Array(m);
    const nz = new Float64Array(m);
    const w = new Float32Array(m);
    for (let j = 0; j < m; j++) {
      fr.xyz(tree.u[j], tree.v[j], tmp, 0);
      nx[j] = tmp[0];
      ny[j] = tmp[1];
      nz[j] = tmp[2];
      w[j] = width(j, tree.q[j]);
    }
    return {
      kind: k,
      delta,
      apex,
      cells: new Int32Array(0),
      x: nx,
      y: ny,
      z: nz,
      parent: Int32Array.from(tree.parent),
      q: Float32Array.from(tree.q),
      mouth: Uint8Array.from(tree.mouth),
      form,
      width: w,
      outline,
      radiusKm,
      wave,
      tide,
      sand: sandShare,
      exit,
    };
  }

  // ---- fans and inland deltas: radiating channels over a patch of ground
  const patchOf = new Int32Array(n).fill(-1);
  const upArea = new Float64Array(n);
  {
    const { order, count } = inp.hydro.routing;
    for (let i = count - 1; i >= 0; i--) {
      const c = order[i];
      upArea[c] += km2(c);
      const r = receiver[c];
      if (r >= 0 && !ocean[r]) upArea[r] += upArea[c];
    }
  }
  const isFan = (c: number): boolean => !ocean[c] && lake[c] < 0 && kind[c] === Deposit.Fan && taken[c] < 0;
  const isInland = (c: number): boolean => !ocean[c] && lake[c] < 0 && wetland[c] === Wetland.InlandDelta && taken[c] < 0;
  let patches = 0;
  const fanCands: { cells: number[]; id: number; entry: number; Q: number; A: number }[] = [];
  for (const [test, k] of [
    [isInland, 'inland'],
    [isFan, 'fan'],
  ] as const) {
    for (let s = 0; s < n; s++) {
      if (patchOf[s] >= 0 || !test(s)) continue;
      const id = patches++;
      const cells = [s];
      patchOf[s] = id;
      for (let i = 0; i < cells.length; i++) {
        const c = cells[i];
        for (let kk = nbrStart[c], e = nbrStart[c + 1]; kk < e; kk++) {
          const m = nbr[kk];
          if (patchOf[m] < 0 && test(m)) {
            patchOf[m] = id;
            cells.push(m);
          }
        }
      }
      // The river feeding it: the biggest one arriving from outside.
      let entry = -1;
      for (const c of cells) {
        if (!sh.isRiver[c]) continue;
        const u = sh.mainDonor[c];
        if (u >= 0 && patchOf[u] === id) continue;
        if (entry < 0 || bankfull[c] > bankfull[entry]) entry = c;
      }
      if (entry < 0) continue;
      const Q = bankfull[entry];
      let patchKm2 = 0;
      for (const c of cells) patchKm2 += km2(c);
      // Fan area grows with its catchment (Bull 1964); inland deltas fill their patch.
      const A = k === 'fan' ? Math.min(patchKm2, 0.5 * upArea[entry] ** 0.9) : patchKm2;
      if (k === 'fan' ? A < MIN_FAN_KM2 || Q < MIN_FAN_Q || upArea[entry] < MIN_FAN_CATCHMENT : A < MIN_INLAND_KM2 || Q < MIN_INLAND_Q) continue;
      if (k === 'fan') {
        fanCands.push({ cells, id, entry, Q, A });
        continue;
      }
      const rng = new Rng(seedFor(inp.seed, `${k}:${entry}`));
      const net = growFan(k, cells, id, entry, Q, A, rng);
      if (net) out.push(net);
    }
  }
  // Fans, biggest river first, each where no bigger one's cone already lies:
  // along a mountain front they sit side by side, as many as the front has
  // room for, however finely the mesh splits its fan ground into patches.
  fanCands.sort((a, b) => b.Q - a.Q || a.entry - b.entry);
  const fanAt: { x: number; y: number; z: number; r: number }[] = [];
  for (const cand of fanCands) {
    const r = Math.sqrt(cand.A / 0.9) / Rkm;
    const e = cand.entry;
    let clear = true;
    for (const o of fanAt) {
      const d = Math.acos(Math.min(1, x[e] * o.x + y[e] * o.y + z[e] * o.z));
      if (d < 0.8 * (r + o.r)) {
        clear = false;
        break;
      }
    }
    if (!clear) continue;
    const rng = new Rng(seedFor(inp.seed, `fan:${e}`));
    const net = growFan('fan', cand.cells, cand.id, e, cand.Q, cand.A, rng);
    if (!net) continue;
    out.push(net);
    fanAt.push({ x: x[e], y: y[e], z: z[e], r });
  }

  function growFan(k: 'fan' | 'inland', cells: number[], id: number, entry: number, Q: number, A: number, rng: Rng): Distributaries | null {
    const u = sh.mainDonor[entry];
    // (At the entry cell itself, where the river's own course hands over.)
    const apex = at(entry);
    // Down the trunk's course: where it leaves the patch.
    let last = entry;
    for (let g = 0; g < 10000; g++) {
      const r = receiver[last];
      if (r < 0 || patchOf[r] !== id) break;
      last = r;
    }
    const after = receiver[last];
    const runsOn = after >= 0 && !ocean[after] && sh.isRiver[after] && lake[after] < 0;
    const toward: [number, number, number] = after >= 0 ? at(after) : at(last);
    const fr = new Frame(apex, toward[0] * apex[0] + toward[1] * apex[1] + toward[2] * apex[2] > 0.999999999 ? at(last) : toward, Rkm);
    const half = k === 'fan' ? 0.9 : 1.1;
    const front = new Front(half + 0.3, 16);
    const inside = (c: number): boolean => patchOf[c] === id || (!ocean[c] && lake[c] < 0 && kind[c] !== Deposit.Bedrock && c !== u);
    const radius = Math.sqrt(A / half);
    const maxR = Math.max(radius * 1.2, 1);
    for (let j = 0; j < front.r.length; j++) {
      const th = front.theta(j);
      const a = Math.abs(th) / half;
      const want = a >= 1 ? 0.15 * radius : radius * Math.cos((a * Math.PI) / 2) ** 0.5;
      front.r[j] = Math.min(want, reach(fr, th, maxR, entry, inside));
    }
    smoothFront(front.r, 1);
    const R0 = Math.max(front.at(0), 0.5);
    // A river that runs on leaves where the patch ends: its trunk goes there,
    // no farther (else it would overshoot and double back to the river).
    let exitR = Infinity;
    if (runsOn) {
      const [ue, ve] = fr.uv(x[after], y[after], z[after]);
      exitR = Math.max(0.5, Math.hypot(ue, ve));
    }
    const N = Math.max(2, Math.min(k === 'fan' ? 6 : 9, Math.round(2 + Math.log2(1 + A / 40)), Math.floor(Q / 4)));
    const tree = new Tree();
    tree.add(0, 0, -1, Q);
    const step = Math.max(0.1, Math.min(spacingKm / 4, R0 / 6));
    // The trunk runs straight through (and on, if the river continues); the rest fan out and die away.
    const shares: number[] = [];
    let rest = 1;
    const trunkShare = runsOn ? 0.55 + 0.2 * rng.next() : 1 / N + 0.1;
    shares.push(trunkShare);
    rest -= trunkShare;
    for (let b = 1; b < N; b++) shares.push(rest / (N - 1));
    const bearings = shares.map((_, b) => (b === 0 ? 0 : (b % 2 ? 1 : -1) * half * 0.85 * Math.ceil(b / 2) / Math.ceil((N - 1) / 2)));
    let exit = -1;
    // Branches leave the trunk one after another down the fan.
    let trunk = 0;
    let qTrunk = Q;
    const order = bearings.map((th, b) => ({ th, b })).filter((o) => o.b > 0);
    const legs = Math.max(1, order.length);
    // The trunk wanders gently down the fan (never a ruled line), pulled back
    // toward its axis.
    let tPsi = 0;
    for (let s = 0; s <= legs; s++) {
      // Trunk leg.
      // (Branches leave near the apex, so the channels radiate over the cone.)
      const rT = Math.min(R0, exitR);
      const goal = s === legs ? Math.min(front.at(0), exitR) : rT * (0.05 + (0.2 * (s + 1)) / (legs + 1));
      for (let g = 0; g < 400; g++) {
        const tu = tree.u[trunk];
        const tv = tree.v[trunk];
        if (Math.hypot(tu, tv) + step >= goal) break;
        tPsi += 0.16 * (rng.next() - 0.5) - 0.2 * tPsi - 0.5 * Math.atan2(tv, Math.max(tu, 1e-6));
        tPsi = Math.max(-0.5, Math.min(0.5, tPsi));
        trunk = tree.add(tu + step * Math.cos(tPsi), tv + step * Math.sin(tPsi), trunk, qTrunk);
      }
      if (s === legs) {
        const tu = tree.u[trunk];
        const tv = tree.v[trunk];
        const rr = Math.max(goal, Math.hypot(tu, tv) + 0.5 * step);
        const th = Math.atan2(tv, Math.max(tu, 1e-6));
        trunk = tree.add(rr * Math.cos(th), rr * Math.sin(th), trunk, qTrunk, runsOn ? 0 : 1);
        if (runsOn) exit = trunk;
        break;
      }
      // A wash off to one side: radiating over the cone and dying away short
      // of its toe as its water sinks into the gravel.
      const o = order[s];
      const q = Q * shares[o.b];
      qTrunk -= q;
      let node = trunk;
      const end = front.at(o.th) * (0.6 + 0.35 * rng.next());
      let uu = tree.u[trunk];
      let vv = tree.v[trunk];
      // Off the trunk at a modest angle, then curving round toward its bearing.
      let psi = (o.th > 0 ? 1 : -1) * (0.35 + 0.3 * rng.next());
      for (let g = 0; g < 200; g++) {
        const th = Math.atan2(vv, uu);
        psi += Math.max(-0.12, Math.min(0.12, o.th + 0.5 * (o.th - th) - psi)) + 0.12 * (rng.next() - 0.5);
        uu += step * Math.cos(psi);
        vv += step * Math.sin(psi);
        if (Math.hypot(uu, vv) >= end || g === 199) {
          node = tree.add(uu, vv, node, q, 1);
          break;
        }
        node = tree.add(uu, vv, node, q);
      }
    }
    const outline: number[] = [];
    const K = 2 * front.steps + 1;
    for (let j = 0; j < K; j++) {
      const th = front.theta(j);
      const r = front.r[j];
      fr.xyz(r * Math.cos(th), r * Math.sin(th), tmp, 0);
      outline.push(tmp[0], tmp[1], tmp[2]);
    }
    const net = finish(tree, fr, k, -1, entry, k, R0, 0, 0, 0.6, new Float64Array(outline), exit, (_j, q) => channelWidth(q, sh.banks[entry]));
    const own = new Set<number>(cells);
    let cell = entry;
    for (let j = 0; j < net.parent.length; j++) {
      cell = nearestCell(mesh, net.x[j], net.y[j], net.z[j], cell);
      if (!ocean[cell] && lake[cell] < 0 && (patchOf[cell] === id || cell === entry)) own.add(cell);
    }
    net.cells = Int32Array.from([...own].filter((c) => taken[c] < 0));
    for (const c of net.cells) taken[c] = out.length;
    return net;
  }

  return out;
}

/** Plan form of a sea delta from Galloway's three forces and its sediment. */
function deltaForm(d: DeltaRecord, sandShare: number, elev: Float32Array, mesh: SphereMesh, M: number): DeltaForm {
  const k: DeltaKind = d.kind;
  if (k === 'tide') return 'estuarine';
  if (k === 'wave') return d.wave > 0.55 && d.tide < 0.3 ? 'cuspate' : 'arcuate';
  // River-dominated: muddy into deep water builds a bird's foot; sandy or shallow, lobes.
  let deep = elev[M];
  for (let k2 = mesh.nbrStart[M], e = mesh.nbrStart[M + 1]; k2 < e; k2++) deep = Math.min(deep, elev[mesh.nbr[k2]]);
  return sandShare < 0.3 && deep < -40 ? 'birdfoot' : 'lobate';
}

/** Moving average over 2w+1 bearings, ignoring nothing: fronts are cell-jagged. */
function smoothFront(r: Float64Array, w: number): void {
  const src = r.slice();
  for (let j = 0; j < r.length; j++) {
    let s = 0;
    let c = 0;
    for (let k = -w; k <= w; k++) {
      const i = j + k;
      if (i < 0 || i >= r.length) continue;
      s += src[i];
      c++;
    }
    r[j] = s / c;
  }
}
