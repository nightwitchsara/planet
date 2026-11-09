import { smoothstep } from '../core/math';
import { cellsWithin, type MeshMap, type SphereMesh } from '../core/mesh';
import { type Climate, MONTHS } from './climate';
import { type GlacialLegacy, ICE_AGE_SEA_LEVEL, type IceSheet } from './glaciers';
import { depressionCells, detailRatio, type Hydrology, LakeKind, REFERENCE_CELLS, route, slopeScale, Wetland } from './hydrology';

// ------------------------------------------------------------------ deposit kinds

/** What covers the bedrock at the surface. */
export const enum Deposit {
  Bedrock = 0,
  Residual = 1,
  Colluvium = 2,
  Alluvium = 3,
  Fan = 4,
  Delta = 5,
  Lake = 6,
  Playa = 7,
  Dunes = 8,
  Loess = 9,
  Till = 10,
  Moraine = 11,
  Outwash = 12,
  Peat = 13,
  Shelf = 14,
}

export const DEPOSIT_NAMES: readonly string[] = [
  'Bedrock under thin soil',
  'Deeply weathered rock (saprolite, laterite)',
  'Slope deposits (colluvium)',
  'River alluvium',
  'Alluvial fan',
  'Delta',
  'Lake beds',
  'Playa and evaporites',
  'Dune sand',
  'Loess',
  'Glacial till',
  'Moraine',
  'Glacial outwash',
  'Peat',
  'Shelf sediment',
];

export const DEPOSIT_COLORS: readonly (readonly [number, number, number])[] = [
  [150, 146, 140],
  [178, 96, 62],
  [170, 152, 122],
  [104, 152, 96],
  [212, 160, 88],
  [64, 158, 146],
  [118, 146, 186],
  [236, 230, 210],
  [240, 204, 112],
  [216, 186, 140],
  [178, 140, 178],
  [124, 84, 146],
  [214, 192, 224],
  [88, 64, 44],
  [86, 124, 160],
];

// Fluvial deposit parts tracked per cell (m).
const P_FAN = 0;
const P_ALLUVIUM = 1;
const P_LAKE = 2;
const P_DELTA = 3;
const P_SHELF = 4;
const P_COLLUVIUM = 5;
const PARTS = 6;

export type DeltaKind = 'river' | 'wave' | 'tide';

/** A river mouth and the sediment its river delivered over the erosion run. */
export interface DeltaRecord {
  /** Sea cell at the mouth, and the last land cell of the river. */
  mouth: number;
  river: number;
  /** Runoff-weighted drainage area of the river (km²). */
  basin: number;
  /** Sediment delivered to the mouth (km³), and its sand-and-gravel part. */
  supply: number;
  sand: number;
  /** Delta plain built above the sea, and shelf and delta front filled in front of it (km³). */
  plain: number;
  front: number;
  /** 0–1 wave energy and tidal range at the mouth. */
  wave: number;
  tide: number;
  /** Which force shapes it (Galloway's triangle): the river's supply, the waves or the tides. */
  kind: DeltaKind;
}

/** Where eroded rock went over the whole run (km³). */
export interface FluvialBudget {
  /** Bedrock worn away by rivers, and by hillslope creep and wash. */
  eroded: number;
  crept: number;
  /** Carried off in solution. */
  dissolved: number;
  /** Laid down: lake fills, floodplains, fans, slope deposits, deltas, shelves (net of later reworking). */
  basins: number;
  floodplains: number;
  fans: number;
  colluvium: number;
  deltas: number;
  shelves: number;
  /** Carried past the shelf edge into the deep sea. */
  lost: number;
}

export interface ErosionResult {
  /** Net river and slope sediment over the bedrock per cell (m): basin fills, floodplains, fans, deltas, shelves. */
  deposit: Float32Array;
  /** Its sand-and-gravel part (m). */
  sand: Float32Array;
  /** The same sediment by how it was laid down (m per cell, see P_*). */
  parts: Float32Array[];
  /** Bedrock worn down per cell (m). */
  lowering: Float32Array;
  /** 0–1 chemical and frost weathering intensity. */
  chem: Float32Array;
  frost: Float32Array;
  deltas: DeltaRecord[];
  budget: FluvialBudget;
}

/** Climate and rock controls on weathering, per cell. */
export interface WeatheringInputs {
  /** Annual mean surface temperature (°C) and precipitation (mm). */
  tempYear: Float32Array;
  precipYear: Float32Array;
  /** Multiplier on the stream-power erodibility from the bedrock (1 = average rock). */
  erodibility?: Float32Array;
  /** 0–1: how much of the rock weathers to sand and gravel, and how much dissolves. */
  sandy?: Float32Array;
  soluble?: Float32Array;
}

/**
 * How each rock weathers, from its name: sandstone, granite and gneiss
 * crumble to sand and gravel; shale, basalt and deep-sea clays to mud;
 * limestone and evaporites largely dissolve.
 */
export function rockTraits(rock: Uint8Array, names: readonly string[]): { sandy: Float32Array; soluble: Float32Array } {
  const table = names.map((name) => {
    const s = name.toLowerCase();
    const soluble = /evaporite|salt|gypsum/.test(s) ? 0.9 : /limestone|dolomite|carbonate|chalk/.test(s) ? 0.6 : 0;
    const sandy = /sandstone|conglomerate|quartzite/.test(s)
      ? 0.9
      : /schist/.test(s)
        ? 0.55
        : /granite|gneiss|basement|shield/.test(s)
          ? 0.7
          : /strata|folded/.test(s)
            ? 0.45
            : /volcanic|andesite|ophiolite/.test(s)
              ? 0.35
              : /basalt/.test(s)
                ? 0.25
                : /shale|mudstone|clay|deep-sea|ooze/.test(s)
                  ? 0.1
                  : soluble > 0
                    ? 0.15
                    : 0.5;
    return { sandy, soluble };
  });
  const sandy = new Float32Array(rock.length);
  const soluble = new Float32Array(rock.length);
  for (let c = 0; c < rock.length; c++) {
    const t = table[rock[c]];
    sandy[c] = t ? t.sandy : 0.5;
    soluble[c] = t ? t.soluble : 0;
  }
  return { sandy, soluble };
}

// ------------------------------------------------------------------ rivers and hillslopes

/** Gradient (m/m) of the alluvial plain a basin builds once sediment has filled it to its outlet. */
const ALLUVIAL_GRADE = 1.5e-4;
/** Rivers flatter than this (m/m) drop part of their load on their floodplain. */
const FLOODPLAIN_SLOPE = 6e-4;
/** Sediment-filled shelves and delta fronts build up to this depth (m); deltas then rise above the sea. */
const SHELF_FILL = -4;
const DELTA_TOP = 2;
/** Only rivers draining at least this much (runoff-weighted km²) carry enough sediment to build a delta plain. */
const DELTA_BASIN = 4e5;
/** Water deeper than this carries river sediment away down the continental slope. */
export const SHELF_EDGE = -400;
/** Unconsolidated sediment erodes this many times faster than average bedrock, once this thick (m). */
const K_SEDIMENT = 3;
const SEDIMENT_REF = 20;
/**
 * Sand and gravel a river can carry per unit of runoff-weighted area and
 * slope (bedload transport capacity ∝ Q·S). Where a steep stream leaves the
 * mountains, its capacity collapses and it drops the excess as a fan.
 */
const BEDLOAD_CAPACITY = 600;
/** Share of the excess bedload a reach drops each pass, and the steepest fan surface it can build (m/m). */
const FAN_SHARE = 0.5;
const FAN_SLOPE = 0.012;
/** A reach fed by a stream at least this steep (m/m) and several times its own slope lies at a mountain front. */
const FAN_APEX = 0.006;
/** Share of the mud a river carries that its plume takes out over the shelf edge. */
const MUD_BYPASS = 0.35;

/**
 * Weathering regime per cell: chemical weathering grows with warmth and
 * water (Arrhenius-like); frost shattering peaks a few degrees below
 * freezing; hot deserts crack rock by heating and salt. Chemical weathering
 * dissolves more and leaves fine clay; mechanical weathering yields sand
 * and gravel.
 */
function weatheringRegime(n: number, w: WeatheringInputs) {
  const chem = new Float32Array(n);
  const frost = new Float32Array(n);
  const dissolved = new Float32Array(n);
  const coarse = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    const t = w.tempYear[c];
    const p = w.precipYear[c];
    const ch = smoothstep(-2, 26, t) * smoothstep(150, 2500, p);
    const fr = Math.exp(-(((t + 5) / 6) ** 2)) * smoothstep(80, 500, p);
    const hot = (1 - smoothstep(150, 700, p)) * smoothstep(8, 26, t);
    const sandy = w.sandy ? w.sandy[c] : 0.5;
    const soluble = w.soluble ? w.soluble[c] : 0;
    chem[c] = ch;
    frost[c] = fr;
    dissolved[c] = Math.max(0.05, Math.min(0.9, 0.1 + 0.3 * ch + 0.6 * soluble - 0.05 * fr));
    coarse[c] = Math.max(0.05, Math.min(0.85, 0.2 + 0.3 * fr + 0.2 * hot - 0.22 * ch + 0.3 * sandy - 0.2 * soluble));
  }
  return { chem, frost, dissolved, coarse };
}

/**
 * Fluvial erosion, transport and deposition with a closed sediment budget.
 * Detachment-limited stream power is solved implicitly in flood order
 * (Braun & Willett 2013): dh/dt = -K·k·A^m·S, with k the rock's erodibility
 * (loose sediment erodes faster than bedrock). Weathering splits the eroded
 * rock into a dissolved load and sand-and-gravel and mud, by climate and rock.
 * The load travels downstream: depressions trap it and silt up from the
 * bottom (most small lakes vanish, big tectonic basins persist); where a
 * steep stream reaches gentle ground its bedload capacity collapses and it
 * builds a fan; gentle reaches build floodplains of overbank mud; rivers fill
 * their shelf and, draining a large basin, build a delta plain out to sea,
 * while a plume carries part of the mud past the shelf edge. Hillslope creep
 * and wash move material downhill as slope deposits and spill off coasts.
 */
export function erode(
  mesh: SphereMesh,
  elev: Float32Array,
  ocean: Uint8Array,
  runoff: Float32Array,
  strength: number,
  radiusKm: number,
  weathering: WeatheringInputs,
  iterations = 6,
): ErosionResult {
  const { n, nbrStart, nbr, nbrDist, nbrEdge, area, x, y, z } = mesh;
  const sed = new Float32Array(n);
  const sand = new Float32Array(n);
  const parts = Array.from({ length: PARTS }, () => new Float32Array(n));
  const lowering = new Float32Array(n);
  const budget: FluvialBudget = { eroded: 0, crept: 0, dissolved: 0, basins: 0, floodplains: 0, fans: 0, colluvium: 0, deltas: 0, shelves: 0, lost: 0 };
  const { chem, frost, dissolved, coarse } = weatheringRegime(n, weathering);
  const deltaMap = new Map<number, DeltaRecord>();
  const R2 = radiusKm * radiusKm;
  const areaKm2 = (c: number) => area[c] * R2;
  const distKm = (a: number, b: number) => Math.acos(Math.min(1, x[a] * x[b] + y[a] * y[b] + z[a] * z[b])) * radiusKm;

  const addDep = (c: number, dh: number, sandShare: number, p: number): void => {
    if (!(dh > 0)) return;
    elev[c] += dh;
    sed[c] += dh;
    sand[c] += dh * sandShare;
    parts[p][c] += dh;
  };
  /** Takes `r` m of loose sediment off cell c (its surface was already lowered); returns the sand part (m). */
  const takeSediment = (c: number, r: number): number => {
    const s = sed[c];
    if (r <= 0 || s <= 0) return 0;
    if (r >= s) {
      const sa = sand[c];
      sed[c] = 0;
      sand[c] = 0;
      for (let p = 0; p < PARTS; p++) parts[p][c] = 0;
      return sa;
    }
    const f = (s - r) / s;
    const sa = sand[c] * (1 - f);
    sand[c] -= sa;
    sed[c] = s - r;
    for (let p = 0; p < PARTS; p++) parts[p][c] *= f;
    return sa;
  };
  /** Fill the lowest of `cells` (sorted by height) like water filling a bowl, with volume v (km²·m), up to `cap`. */
  const fillBasin = (cells: number[], v: number, cap: number, sandShare: number, p: number): number => {
    if (v <= 0 || cells.length === 0 || elev[cells[0]] >= cap) return v;
    let areaSum = 0;
    let level = elev[cells[0]];
    let k = 0;
    for (; k < cells.length; k++) {
      const next = Math.min(cap, k + 1 < cells.length ? elev[cells[k + 1]] : cap);
      areaSum += areaKm2(cells[k]);
      const need = (next - level) * areaSum;
      if (need >= v) {
        level += v / areaSum;
        v = 0;
        break;
      }
      v -= need;
      level = next;
      if (level >= cap) break;
    }
    for (let i = 0; i <= Math.min(k, cells.length - 1); i++) {
      const c = cells[i];
      if (elev[c] < level) addDep(c, level - elev[c], sandShare, p);
    }
    return v;
  };

  const finish = (): ErosionResult => {
    for (let c = 0; c < n; c++) {
      const a = areaKm2(c);
      budget.fans += parts[P_FAN][c] * a;
      budget.floodplains += parts[P_ALLUVIUM][c] * a;
      budget.basins += parts[P_LAKE][c] * a;
      budget.deltas += parts[P_DELTA][c] * a;
      budget.shelves += parts[P_SHELF][c] * a;
      budget.colluvium += parts[P_COLLUVIUM][c] * a;
    }
    // km²·m → km³
    for (const k of Object.keys(budget) as (keyof FluvialBudget)[]) budget[k] /= 1000;
    const deltas = classifyDeltas(mesh, ocean, elev, [...deltaMap.values()], radiusKm);
    return { deposit: sed, sand, parts, lowering, chem, frost, deltas, budget };
  };
  if (strength <= 0) return finish();

  // Stream power on cells: a hillslope cell drains about its own width times
  // its distance from the divide, so the erosion it sees grows with the cell
  // size (√ of it). Scaling K the other way keeps a world the same at every
  // detail level (calibrated at 320k cells).
  const K = 0.1 * strength * detailRatio(n) ** 0.25;
  const erod = weathering.erodibility;
  /** River gradients as the reference detail measures them (floodplains and fans form where they are gentle). */
  const flatScale = slopeScale(n);
  /** A big river's delta plain covers about one cell of the reference detail. */
  const deltaArea = (4 * Math.PI * R2) / REFERENCE_CELLS;
  const A = new Float64Array(n);
  const madeC = new Float64Array(n);
  const madeF = new Float64Array(n);
  const qsC = new Float64Array(n);
  const qsF = new Float64Array(n);
  /** Steepest stream (reference-scale gradient) arriving at each cell: fans start where it is much steeper than the cell. */
  const steepIn = new Float32Array(n);
  const shelf: number[] = [];

  // Hillslope transport, conservative between neighbors: the volume a cell
  // passes downhill to a neighbor ∝ height difference × shared edge / distance.
  // The coefficient matches a relaxation of 0.08 toward the neighbors' mean.
  const wEdge = new Float32Array(nbr.length);
  let wSum = 0;
  for (let k = 0; k < nbr.length; k++) {
    wEdge[k] = nbrEdge[k] / nbrDist[k];
    wSum += wEdge[k];
  }
  const D = (0.08 * (4 * Math.PI)) / wSum;
  const e0 = new Float32Array(n);
  const outV = new Float64Array(n);
  const gain = new Float64Array(n);
  const gainSand = new Float64Array(n);

  /** River sediment (km²·m) reaching the sea at ocean cell `o`, from `river` with weighted basin area `basinArea`. */
  const toCoast = (o: number, qc: number, qf: number, basinArea: number, river: number): void => {
    const supply = qc + qf;
    if (supply <= 0) return;
    let rec = deltaMap.get(o);
    if (!rec) {
      rec = { mouth: o, river, basin: 0, supply: 0, sand: 0, plain: 0, front: 0, wave: 0, tide: 0, kind: 'river' };
      deltaMap.set(o, rec);
    }
    rec.supply += supply;
    rec.sand += qc;
    if (basinArea > rec.basin) {
      rec.basin = basinArea;
      rec.river = river;
    }
    if (elev[o] < SHELF_EDGE) {
      budget.lost += supply;
      return;
    }
    const plume = qf * MUD_BYPASS;
    budget.lost += plume;
    qf -= plume;
    shelf.length = 0;
    shelf.push(o);
    for (let k = nbrStart[o], e = nbrStart[o + 1]; k < e; k++) {
      const m = nbr[k];
      if (ocean[m] && elev[m] >= SHELF_EDGE) shelf.push(m);
    }
    shelf.sort((a, b) => elev[a] - elev[b]);
    let v = qc + qf;
    const sandShare = qc / v;
    const left = fillBasin(shelf, v, SHELF_FILL, sandShare, P_SHELF);
    rec.front += v - left;
    v = left;
    // Once its shelf is full, a big river builds its delta plain out above the sea.
    const need = (DELTA_TOP - elev[o]) * areaKm2(o);
    if (basinArea >= DELTA_BASIN && need > 0 && v >= need) {
      addDep(o, DELTA_TOP - elev[o], sandShare, P_DELTA);
      rec.plain += need;
      v -= need;
      // On a finer mesh the plain builds on out over the shallowest shelf beside the mouth.
      let built = areaKm2(o);
      for (const s of shelf) {
        if (s === o || elev[s] >= DELTA_TOP || built + areaKm2(s) > 1.2 * deltaArea) continue;
        const more = (DELTA_TOP - elev[s]) * areaKm2(s);
        if (v < more) break;
        addDep(s, DELTA_TOP - elev[s], sandShare, P_DELTA);
        rec.plain += more;
        v -= more;
        built += areaKm2(s);
      }
    }
    budget.lost += v;
  };

  /**
   * A basin filled to its outlet keeps aggrading as an alluvial plain that
   * rises gently away from the outlet, so the rivers crossing it keep
   * draining out instead of leaving a dead-flat floor.
   */
  const gradeToOutlet = (cells: number[], spill: number, v: number, sandShare: number): number => {
    let wedge = 0;
    const rise = (c: number) => distKm(c, spill) * 1000 * ALLUVIAL_GRADE;
    for (const c of cells) wedge += Math.max(0, elev[spill] + rise(c) - elev[c]) * areaKm2(c);
    if (wedge <= 0) return v;
    const f = Math.min(1, v / wedge);
    const top = elev[spill];
    for (const c of cells) addDep(c, f * Math.max(0, top + rise(c) - elev[c]), sandShare, P_ALLUVIUM);
    return v - f * wedge;
  };

  const PT: Record<string, number> = { route: 0, incise: 0, sed: 0, slope: 0 };
  (globalThis as any).__PT = PT;
  for (let it = 0; it < iterations; it++) {
    let tq = performance.now();
    const r = route(mesh, elev, ocean);
    PT.route += performance.now() - tq; tq = performance.now();
    const { order, receiver, basin, count } = r;
    for (let c = 0; c < n; c++) A[c] = ocean[c] ? 0 : areaKm2(c) * (0.3 + 0.7 * Math.min(2, runoff[c] / 500));
    for (let i = count - 1; i >= 0; i--) {
      const c = order[i];
      const rc = receiver[c];
      if (rc >= 0 && !ocean[rc]) A[rc] += A[c];
    }

    // Incision, receivers first; loose sediment goes before bedrock.
    madeC.fill(0);
    madeF.fill(0);
    for (let i = 0; i < count; i++) {
      const c = order[i];
      const rc = receiver[c];
      if (rc < 0 || basin[c] >= 0) continue;
      const hr = ocean[rc] ? 0 : elev[rc];
      if (elev[c] <= hr) continue;
      const k = (erod ? erod[c] : 1) * (1 + (K_SEDIMENT - 1) * Math.min(1, sed[c] / SEDIMENT_REF));
      const F = (K * k * Math.sqrt(A[c])) / Math.max(1, distKm(c, rc));
      const h = (elev[c] + F * hr) / (1 + F);
      const lower = elev[c] - h;
      elev[c] = h;
      const fromSed = Math.min(sed[c], lower);
      const sandOut = takeSediment(c, fromSed);
      const rock = lower - fromSed;
      const a = areaKm2(c);
      lowering[c] += rock;
      const rv = rock * a;
      budget.eroded += rv;
      budget.dissolved += rv * dissolved[c];
      const solid = rv * (1 - dissolved[c]);
      madeC[c] = sandOut * a + solid * coarse[c];
      madeF[c] = (fromSed - sandOut) * a + solid * (1 - coarse[c]);
    }

    PT.incise += performance.now() - tq; tq = performance.now();
    // Sediment routing, upstream to downstream.
    const pits = depressionCells(r, elev);
    qsC.fill(0);
    qsF.fill(0);
    steepIn.fill(0);
    for (let i = count - 1; i >= 0; i--) {
      const c = order[i];
      const rc = receiver[c];
      let qc = qsC[c] + madeC[c];
      let qf = qsF[c] + madeF[c];
      const b = basin[c];
      if (b >= 0) {
        // Everything entering a depression reaches its outlet cell; trap what fits.
        if ((rc < 0 || basin[rc] !== b) && qc + qf > 0) {
          const share = qc / (qc + qf);
          let left = fillBasin(pits[b], qc + qf, r.basinLevel[b], share, P_LAKE);
          if (left > 0) left = gradeToOutlet(pits[b], r.basinSpill[b], left, share);
          qc = left * share;
          qf = left - qc;
        }
      } else if (rc >= 0 && !ocean[rc]) {
        const dx = distKm(c, rc) * 1000;
        const s = ((elev[c] - elev[rc]) / dx) * flatScale;
        const a = areaKm2(c);
        const cap = BEDLOAD_CAPACITY * A[c] * Math.max(0, s);
        if (qc > cap) {
          // Bedload beyond what the reach can move: a fan at a mountain front, else braided channel fill.
          const room = Math.max(0, elev[rc] + (FAN_SLOPE / flatScale) * dx - elev[c]);
          const vol = Math.min((qc - cap) * FAN_SHARE, room * a);
          if (vol > 0) {
            const fromC = Math.min(qc, 0.85 * vol);
            const fromF = Math.min(qf, vol - fromC);
            addDep(c, (fromC + fromF) / a, fromC / (fromC + fromF), steepIn[c] > Math.max(FAN_APEX, 3 * s) ? P_FAN : P_ALLUVIUM);
            qc -= fromC;
            qf -= fromF;
          }
        }
        if (s < FLOODPLAIN_SLOPE && qc + qf > 0) {
          // Overbank floods leave mostly silt and clay on the floodplain.
          const d = (qc + qf) * 0.25 * (1 - Math.max(0, s) / FLOODPLAIN_SLOPE);
          const vol = Math.min(d, Math.max(0, elev[rc] + FLOODPLAIN_SLOPE * dx - elev[c]) * a);
          if (vol > 0) {
            const fromF = Math.min(qf, 0.75 * vol);
            const fromC = Math.min(qc, vol - fromF);
            addDep(c, (fromC + fromF) / a, fromC / (fromC + fromF), P_ALLUVIUM);
            qc -= fromC;
            qf -= fromF;
          }
        }
        if (s > steepIn[rc]) steepIn[rc] = s;
      }
      if (rc < 0) {
        budget.lost += qc + qf;
        continue;
      }
      if (ocean[rc]) {
        // The river builds out into the shallowest water beside its mouth.
        let o = rc;
        for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) if (ocean[nbr[k]] && elev[nbr[k]] > elev[o]) o = nbr[k];
        toCoast(o, qc, qf, A[c], c);
      } else {
        qsC[rc] += qc;
        qsF[rc] += qf;
      }
    }

    PT.sed += performance.now() - tq; tq = performance.now();
    // Hillslope creep and wash: downhill to neighbors, gathering at slope feet
    // and spilling off coasts. Land is not pulled below 1 m.
    e0.set(elev);
    outV.fill(0);
    for (let c = 0; c < n; c++) {
      if (ocean[c]) continue;
      for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
        const m = nbr[k];
        const hm = ocean[m] ? Math.min(e0[m], 0) : e0[m];
        if (e0[c] > hm) outV[c] += D * (e0[c] - hm) * wEdge[k];
      }
    }
    gain.fill(0);
    gainSand.fill(0);
    for (let c = 0; c < n; c++) {
      if (ocean[c] || outV[c] <= 0) continue;
      const scale = Math.min(1, (Math.max(0, e0[c] - 1) * area[c]) / outV[c]);
      if (scale <= 0) continue;
      const loss = (outV[c] * scale) / area[c];
      const fromSed = Math.min(sed[c], loss);
      const sandOut = takeSediment(c, fromSed);
      const rock = loss - fromSed;
      if (rock > 0) {
        lowering[c] += rock;
        budget.crept += rock * areaKm2(c);
      }
      elev[c] -= loss;
      const share = (sandOut + rock * coarse[c]) / loss;
      for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
        const m = nbr[k];
        const hm = ocean[m] ? Math.min(e0[m], 0) : e0[m];
        if (e0[c] <= hm) continue;
        const v = D * (e0[c] - hm) * wEdge[k] * scale;
        gain[m] += v;
        gainSand[m] += v * share;
      }
    }
    for (let m = 0; m < n; m++) {
      if (gain[m] <= 0) continue;
      if (!ocean[m]) {
        addDep(m, gain[m] / area[m], gainSand[m] / gain[m], P_COLLUVIUM);
        continue;
      }
      // Off a coast it settles on the shelf (never building it above the fill depth); the rest goes down the slope.
      const fit = Math.min(gain[m], Math.max(0, SHELF_FILL - elev[m]) * area[m]);
      addDep(m, fit / area[m], gainSand[m] / gain[m], P_SHELF);
      budget.lost += (gain[m] - fit) * R2;
    }
    PT.slope += performance.now() - tq;
  }
  const tf = performance.now(); const res = finish(); PT.finish = performance.now() - tf;
  return res;
}

/**
 * Wave energy and tidal range at each river mouth, and the delta type they
 * make with the river's supply (Galloway 1975): open coasts in the storm
 * belts are wave-dominated, wide shelves and funnel-shaped bays amplify the
 * tides, and a big supply into quiet water builds a river-dominated delta.
 */
function classifyDeltas(mesh: SphereMesh, ocean: Uint8Array, elev: Float32Array, recs: DeltaRecord[], radiusKm: number): DeltaRecord[] {
  const keep = recs.filter((r) => r.plain > 0 || r.supply >= 1500).map((r) => ({
    ...r,
    supply: r.supply / 1000,
    sand: r.sand / 1000,
    plain: r.plain / 1000,
    front: r.front / 1000,
  }));
  if (keep.length === 0) return keep;
  const supplies = keep.map((r) => r.supply).sort((a, b) => a - b);
  const typical = supplies[Math.floor(supplies.length / 2)];
  const reach = 400 / radiusKm;
  for (const r of keep) {
    const { cells } = cellsWithin(mesh, r.mouth, reach);
    let sea = 0;
    let shelf = 0;
    for (const c of cells) {
      if (!ocean[c]) continue;
      sea++;
      if (elev[c] > -200) shelf++;
    }
    const open = sea / cells.length;
    const shelfShare = sea > 0 ? shelf / sea : 0;
    const lat = Math.abs((mesh.lat[r.mouth] * 180) / Math.PI);
    const storms = Math.exp(-(((lat - 52) / 13) ** 2));
    const trades = Math.exp(-(((lat - 18) / 9) ** 2));
    r.wave = Math.min(1, open ** 1.5 * (0.25 + 0.85 * storms + 0.35 * trades));
    r.tide = Math.min(1, shelfShare * (0.45 + 0.7 * (1 - open)));
    const river = r.supply / (r.supply + 3 * typical);
    r.kind = river >= r.wave && river >= r.tide ? 'river' : r.wave >= r.tide ? 'wave' : 'tide';
  }
  return keep;
}

// ------------------------------------------------------------------ ice-age glaciers

export interface GlacialDeposits {
  /** Lodgement till, end moraine and outwash laid down by the ice-age ice and its meltwater (m). */
  till: Float32Array;
  moraine: Float32Array;
  outwash: Float32Array;
  /** Rock flour (km²·m) the meltwater spread at the head of each outwash plain, open to the wind. */
  flour: Float32Array;
  /** Bedrock ground away and weathered mantle stripped, and where it went (km³). */
  budget: { eroded: number; stripped: number; till: number; moraine: number; outwash: number; marine: number; exported: number; flour: number };
}

/** How long the erosion run stands for, to turn its lowering into a rate (kyr). */
const RUN_KYR = 8000;

/**
 * Bedrock weathered in place (m), by the soil production function (Heimsath
 * et al. 1997): soil thins where erosion strips it fast and thickens toward
 * a few metres where it is slow; on stable, warm, wet ground the weathering
 * front sinks deep into saprolite.
 */
function regolithDepth(lowering: number, chem: number, frost: number): { soil: number; saprolite: number } {
  const E = (lowering * 1000) / RUN_KYR;
  const P0 = 15 + 250 * chem + 50 * frost;
  const ratio = P0 / Math.max(1, E);
  return { soil: 0.6 * Math.log(1 + ratio), saprolite: 40 * chem * (1 - Math.exp(-ratio / 30)) };
}

/** Share of the debris a cell's ice lodges as till: little under thick, fast ice; much under its thin outer zone. */
const TILL_BASE = 0.04;
const TILL_OUTER = 0.3;
/** At the margin: share dumped as end moraine, and share washed out onto the outwash plain (the rest is rock flour). */
const MORAINE_SHARE = 0.4;
const OUTWASH_SHARE = 0.35;
const TILL_MAX = 60;
const MORAINE_MAX = 120;
const OUTWASH_MAX = 40;
/** How far outwash plains run beyond the ice margin (km). */
const OUTWASH_KM = 350;

/**
 * Where the rock the ice-age glaciers ground away went. Debris rides the ice
 * down its flow lines: thin, slow ice near the margin lodges it as till
 * (thick till plains in the outer zone, little on the scoured interior); at
 * the margin it piles into end moraines and meltwater spreads sand and gravel
 * across outwash plains, letting out rock flour to rivers and the wind. Ice
 * reaching the sea drops its load on the shelf. The bed rises by what is laid
 * down.
 */
export function glacialDeposits(
  mesh: SphereMesh,
  bed: Float32Array,
  past: IceSheet,
  legacy: GlacialLegacy,
  fl: ErosionResult,
  radiusKm: number,
): GlacialDeposits {
  const { n, nbrStart, nbr, area } = mesh;
  const R2 = radiusKm * radiusKm;
  const till = new Float32Array(n);
  const moraine = new Float32Array(n);
  const outwash = new Float32Array(n);
  const flour = new Float32Array(n);
  const budget = { eroded: 0, stripped: 0, till: 0, moraine: 0, outwash: 0, marine: 0, exported: 0, flour: 0 };
  const H = past.thick;
  // Flow order down the ice: every cell after all the cells whose ice flows into it.
  const indeg = new Int32Array(n);
  let nIce = 0;
  for (let c = 0; c < n; c++) {
    if (H[c] <= 0) continue;
    nIce++;
    const rc = past.receiver[c];
    if (rc >= 0 && H[rc] > 0) indeg[rc]++;
  }
  const ice: number[] = [];
  for (let c = 0; c < n; c++) if (H[c] > 0 && indeg[c] === 0) ice.push(c);
  for (let i = 0; i < ice.length; i++) {
    const rc = past.receiver[ice[i]];
    if (rc >= 0 && H[rc] > 0 && --indeg[rc] === 0) ice.push(rc);
  }
  if (ice.length < nIce) for (let c = 0; c < n; c++) if (H[c] > 0 && indeg[c] > 0) ice.push(c);
  const load = new Float64Array(n);
  const steps = Math.max(2, Math.round(OUTWASH_KM / (mesh.spacing * radiusKm)));
  const lay = (arr: Float32Array, c: number, v: number, max: number): number => {
    const a = area[c] * R2;
    const dh = Math.min(v / a, Math.max(0, max - arr[c]));
    if (dh <= 0) return 0;
    arr[c] += dh;
    bed[c] += dh;
    return dh * a;
  };
  for (const c of ice) {
    const a = area[c] * R2;
    // The ice grinds away bedrock and first strips the weathered mantle the warm interglacial ages left.
    const pre = regolithDepth(fl.lowering[c], fl.chem[c], fl.frost[c]);
    const strip = (pre.soil + pre.saprolite) * legacy.scour[c];
    bed[c] -= strip;
    const own = (legacy.erosion[c] + strip) * a;
    budget.eroded += legacy.erosion[c] * a;
    budget.stripped += strip * a;
    let v = load[c] + own;
    if (v <= 0) continue;
    const rc = past.receiver[c];
    if (bed[c] > ICE_AGE_SEA_LEVEL) {
      const t = lay(till, c, v * (TILL_BASE + TILL_OUTER * (1 - legacy.scour[c])), TILL_MAX);
      budget.till += t;
      v -= t;
    }
    if (rc >= 0 && H[rc] > 0) {
      load[rc] += v;
      continue;
    }
    // The ice gives out here: dead ice in a hollow, a calving front, or a land margin.
    if (rc < 0) {
      const t = lay(till, c, v, TILL_MAX + 60);
      budget.till += t;
      budget.exported += v - t;
      continue;
    }
    if (bed[rc] <= ICE_AGE_SEA_LEVEL) {
      // A calving front: a grounding-zone wedge on the shelf; the rest drifts off in icebergs and plumes.
      const dh = Math.min(80, v / (area[rc] * R2));
      bed[rc] += dh;
      budget.marine += dh * area[rc] * R2;
      budget.exported += v - dh * area[rc] * R2;
      continue;
    }
    const m = lay(moraine, c, v * MORAINE_SHARE, MORAINE_MAX);
    budget.moraine += m;
    let wash = v * OUTWASH_SHARE + (v * MORAINE_SHARE - m);
    const rockFlour = v * (1 - MORAINE_SHARE - OUTWASH_SHARE);
    // Meltwater spreads sand and gravel down the valley: steepest descent over ice-free ground.
    let cur = rc;
    flour[cur] += rockFlour;
    budget.flour += rockFlour;
    for (let s = 0; s < steps && wash > 0; s++) {
      if (H[cur] <= 0) {
        const o = lay(outwash, cur, wash * 0.35, OUTWASH_MAX);
        budget.outwash += o;
        wash -= o;
      }
      let next = -1;
      let lo = bed[cur];
      for (let k = nbrStart[cur], e = nbrStart[cur + 1]; k < e; k++) {
        const q = nbr[k];
        if (bed[q] < lo) {
          lo = bed[q];
          next = q;
        }
      }
      if (next < 0 || bed[next] < 0) break;
      cur = next;
    }
    budget.exported += wash;
  }
  for (const k of Object.keys(budget) as (keyof typeof budget)[]) budget[k] /= 1000;
  return { till, moraine, outwash, flour, budget };
}

// ------------------------------------------------------------------ wind

export interface AeolianDeposits {
  /** Wind-blown sand and loess laid down (m). */
  dunes: Float32Array;
  loess: Float32Array;
  /** Sand and fines blown off each cell (m). */
  deflation: Float32Array;
  budget: { deflated: number; fromFlour: number; dunes: number; loess: number; toSea: number };
}

/** Share of the loose sand and of the fines on bare desert ground the wind takes over the run. */
const DEFLATE_SAND = 0.5;
const DEFLATE_FINES = 0.3;
/** Share of the ice-age rock flour on outwash plains that blew away as dust. */
const FLOUR_TO_WIND = 0.5;

/**
 * Wind-blown sand and dust. Bare ground (deserts, and the outwash plains of
 * the ice age) gives up sand from fans, river beds and outwash and fines
 * from dry lake beds, floodplains and glacial rock flour. Both drift
 * downwind in the dry-season winds: sand only a few hundred kilometres,
 * gathering in basins and lows as dune fields and sand seas and stopping
 * where vegetation pins it; dust travels far and settles where grassland
 * traps it, building loess sheets downwind of deserts and outwash plains.
 * Solved on the climate mesh; the bed changes by what is blown on and off.
 */
export function aeolianDeposits(
  fine: SphereMesh,
  coarse: SphereMesh,
  f2c: MeshMap,
  climate: Climate,
  bed: Float32Array,
  ocean: Uint8Array,
  precipYear: Float32Array,
  petYear: Float32Array,
  tempYear: Float32Array,
  fl: ErosionResult,
  gl: GlacialDeposits,
  radiusKm: number,
): AeolianDeposits {
  const n = fine.n;
  const R2 = radiusKm * radiusKm;
  const dunes = new Float32Array(n);
  const loess = new Float32Array(n);
  const deflation = new Float32Array(n);
  const budget = { deflated: 0, fromFlour: 0, dunes: 0, loess: 0, toSea: 0 };
  const bare = new Float32Array(n);
  const trap = new Float32Array(n);
  const nc = coarse.n;
  const srcS = new Float64Array(nc);
  const srcL = new Float64Array(nc);
  const landArea = new Float64Array(nc);
  const sumAI = new Float64Array(nc);
  const sumBare = new Float64Array(nc);
  const sumElev = new Float64Array(nc);
  for (let c = 0; c < n; c++) {
    if (ocean[c]) continue;
    const ai = precipYear[c] / Math.max(1, petYear[c]);
    const t = tempYear[c];
    const coldBare = 0.5 * smoothstep(-2, -14, t) * (1 - smoothstep(0.4, 1.2, ai));
    bare[c] = Math.max(1 - smoothstep(0.06, 0.35, ai), coldBare);
    trap[c] = smoothstep(0.12, 0.5, ai);
    const a = fine.area[c] * R2;
    const k = f2c.nearest[c];
    landArea[k] += a;
    sumAI[k] += Math.min(3, ai) * a;
    sumBare[k] += bare[c] * a;
    sumElev[k] += bed[c] * a;
    // Sand: loose fluvial sand and gravel, and outwash; fines: the muddy part of river and lake deposits.
    const sed = fl.deposit[c];
    const sandF = Math.min(fl.sand[c], 40) * DEFLATE_SAND * bare[c];
    const fineF = Math.min(Math.max(0, sed - fl.sand[c]), 30) * DEFLATE_FINES * bare[c];
    const ow = gl.outwash[c] * DEFLATE_SAND * Math.max(bare[c], 0.3);
    const flourV = gl.flour[c] * FLOUR_TO_WIND;
    if (sandF + fineF > 0 && sed > 0) {
      // Deflation lowers the river deposits it came from.
      const f = (sed - sandF - fineF) / sed;
      for (let p = 0; p < PARTS; p++) fl.parts[p][c] *= f;
      fl.sand[c] -= sandF;
      fl.deposit[c] = sed - sandF - fineF;
    }
    gl.outwash[c] -= ow;
    deflation[c] = sandF + fineF + ow;
    bed[c] -= deflation[c];
    srcS[k] += (sandF + ow) * a;
    srcL[k] += fineF * a + flourV;
    budget.deflated += (sandF + fineF + ow) * a + flourV;
    budget.fromFlour += flourV;
  }

  // Dry-season wind on the climate mesh, and how much of each cell's airborne load goes to each neighbor.
  const { x, y, z, nbrStart, nbr } = coarse;
  const wE = new Float32Array(nc);
  const wN = new Float32Array(nc);
  for (let k = 0; k < nc; k++) {
    let se = 0;
    let sn = 0;
    let sw = 0;
    for (let m = 0; m < MONTHS; m++) {
      const w = 1 / (1 + climate.precip[k * MONTHS + m] / 40);
      se += climate.windE[k * MONTHS + m] * w;
      sn += climate.windN[k * MONTHS + m] * w;
      sw += w;
    }
    wE[k] = se / sw;
    wN[k] = sn / sw;
  }
  const share = new Float32Array(nbr.length);
  for (let k = 0; k < nc; k++) {
    const r = Math.hypot(x[k], y[k]) || 1e-9;
    const ex = -y[k] / r;
    const ey = x[k] / r;
    const nx = -(z[k] * x[k]) / r;
    const ny = -(z[k] * y[k]) / r;
    const nz = r;
    const vx = wE[k] * ex + wN[k] * nx;
    const vy = wE[k] * ey + wN[k] * ny;
    const vz = wN[k] * nz;
    const sp = Math.hypot(vx, vy, vz) || 1e-9;
    let sum = 0;
    for (let i = nbrStart[k], e = nbrStart[k + 1]; i < e; i++) {
      const j = nbr[i];
      let dx = x[j] - x[k];
      let dy = y[j] - y[k];
      let dz = z[j] - z[k];
      const along = dx * x[k] + dy * y[k] + dz * z[k];
      dx -= along * x[k];
      dy -= along * y[k];
      dz -= along * z[k];
      const dl = Math.hypot(dx, dy, dz) || 1e-9;
      const toward = Math.max(0, (vx * dx + vy * dy + vz * dz) / (sp * dl));
      share[i] = toward * toward;
      sum += share[i];
    }
    for (let i = nbrStart[k], e = nbrStart[k + 1]; i < e; i++) share[i] = sum > 0 ? share[i] / sum : 0;
  }
  const pS = new Float32Array(nc);
  const pL = new Float32Array(nc);
  const isSea = new Uint8Array(nc);
  const elevC = new Float32Array(nc);
  for (let k = 0; k < nc; k++) elevC[k] = landArea[k] > 0 ? sumElev[k] / landArea[k] : -1000;
  for (let k = 0; k < nc; k++) {
    const land = landArea[k] / (coarse.area[k] * R2);
    if (land < 0.3) {
      isSea[k] = 1;
      pS[k] = 1;
      pL[k] = 0.3;
      continue;
    }
    let mean = 0;
    let cnt = 0;
    for (let i = nbrStart[k], e = nbrStart[k + 1]; i < e; i++) {
      mean += elevC[nbr[i]];
      cnt++;
    }
    const low = smoothstep(0, 400, mean / cnt - elevC[k]);
    const b = sumBare[k] / landArea[k];
    const ai = sumAI[k] / landArea[k];
    pS[k] = Math.max(0.15, Math.min(1, 0.25 + 0.4 * low + 0.7 * (1 - b)));
    pL[k] = Math.max(0.02, Math.min(0.7, 0.02 + 0.35 * smoothstep(0.12, 0.5, ai)));
  }
  const settle = (src: Float64Array, p: Float32Array): Float64Array => {
    const dep = new Float64Array(nc);
    let inflow = new Float64Array(nc);
    let next = new Float64Array(nc);
    let moving = 0;
    for (let k = 0; k < nc; k++) moving += src[k];
    if (moving <= 0) return dep;
    for (let it = 0; it < 240; it++) {
      next.fill(0);
      let left = 0;
      for (let k = 0; k < nc; k++) {
        const C = (it === 0 ? src[k] : 0) + inflow[k];
        if (C <= 0) continue;
        let d = C * p[k];
        let go = C - d;
        if (nbrStart[k + 1] === nbrStart[k] || go <= 0) {
          d = C;
          go = 0;
        }
        let sent = 0;
        for (let i = nbrStart[k], e = nbrStart[k + 1]; i < e && go > 0; i++) {
          if (share[i] <= 0) continue;
          next[nbr[i]] += go * share[i];
          sent += go * share[i];
        }
        dep[k] += d + (go - sent);
        left += sent;
      }
      [inflow, next] = [next, inflow];
      if (left < 1e-6 * moving) break;
    }
    for (let k = 0; k < nc; k++) dep[k] += inflow[k];
    return dep;
  };
  const depS = settle(srcS, pS);
  const depL = settle(srcL, pL);

  // Back onto the fine cells: sand gathers in lows on bare ground, loess drapes gentle vegetated ground.
  const densS = new Float32Array(nc);
  const densL = new Float32Array(nc);
  let totS = 0;
  let totL = 0;
  for (let k = 0; k < nc; k++) {
    if (isSea[k]) {
      budget.toSea += depS[k] + depL[k];
      continue;
    }
    densS[k] = depS[k] / landArea[k];
    densL[k] = depL[k] / landArea[k];
    totS += depS[k];
    totL += depL[k];
  }
  // Where in its neighborhood each fine cell catches sand (bare lows: dune fields
  // and sand seas) and dust (gentle vegetated ground), relative to the
  // neighborhood's mean, so each region keeps what settled on it.
  const { idx, w } = f2c;
  const fn = fine.nbrStart;
  const fnb = fine.nbr;
  const catchS = new Float32Array(n);
  const catchL = new Float32Array(n);
  const meanS = new Float64Array(nc);
  const meanL = new Float64Array(nc);
  for (let c = 0; c < n; c++) {
    if (ocean[c]) continue;
    let mean = 0;
    let steep = 0;
    let cnt = 0;
    for (let k = fn[c], e = fn[c + 1]; k < e; k++) {
      const m = fnb[k];
      if (ocean[m]) continue;
      mean += bed[m];
      steep = Math.max(steep, Math.abs(bed[m] - bed[c]) / (fine.nbrDist[k] * radiusKm * 1000));
      cnt++;
    }
    const low = cnt ? smoothstep(0, 150, mean / cnt - bed[c]) : 0;
    catchS[c] = 0.02 + bare[c] * bare[c] * (0.15 + low) ** 1.5;
    catchL[c] = (0.3 + trap[c]) * (1 - 0.7 * smoothstep(0.02, 0.12, steep));
    const a = fine.area[c] * R2;
    meanS[f2c.nearest[c]] += catchS[c] * a;
    meanL[f2c.nearest[c]] += catchL[c] * a;
  }
  for (let k = 0; k < nc; k++) {
    meanS[k] = landArea[k] > 0 ? meanS[k] / landArea[k] : 0;
    meanL[k] = landArea[k] > 0 ? meanL[k] / landArea[k] : 0;
  }
  let rawS = 0;
  let rawL = 0;
  for (let c = 0; c < n; c++) {
    if (ocean[c]) continue;
    const j = 3 * c;
    const ds = densS[idx[j]] * w[j] + densS[idx[j + 1]] * w[j + 1] + densS[idx[j + 2]] * w[j + 2];
    const dl = densL[idx[j]] * w[j] + densL[idx[j + 1]] * w[j + 1] + densL[idx[j + 2]] * w[j + 2];
    const ms = meanS[idx[j]] * w[j] + meanS[idx[j + 1]] * w[j + 1] + meanS[idx[j + 2]] * w[j + 2];
    const ml = meanL[idx[j]] * w[j] + meanL[idx[j + 1]] * w[j + 1] + meanL[idx[j + 2]] * w[j + 2];
    dunes[c] = ds > 0 ? (ds * catchS[c]) / Math.max(1e-4, ms) : 0;
    loess[c] = dl > 0 ? (dl * catchL[c]) / Math.max(1e-4, ml) : 0;
    const a = fine.area[c] * R2;
    rawS += dunes[c] * a;
    rawL += loess[c] * a;
  }
  const kS = rawS > 0 ? totS / rawS : 0;
  const kL = rawL > 0 ? totL / rawL : 0;
  if (rawS <= 0) budget.toSea += totS;
  if (rawL <= 0) budget.toSea += totL;
  for (let c = 0; c < n; c++) {
    if (ocean[c]) continue;
    dunes[c] *= kS;
    loess[c] *= kL;
    bed[c] += dunes[c] + loess[c];
    const a = fine.area[c] * R2;
    budget.dunes += dunes[c] * a;
    budget.loess += loess[c] * a;
  }
  for (const k of Object.keys(budget) as (keyof typeof budget)[]) budget[k] /= 1000;
  return { dunes, loess, deflation, budget };
}

// ------------------------------------------------------------------ the surface

/** Everything eroded, carried and laid down over the run (km³). */
export interface SedimentBudget extends FluvialBudget {
  glacial: GlacialDeposits['budget'];
  wind: AeolianDeposits['budget'];
}

export interface Deposits {
  /** The surface unit over the bedrock, and its thickness (m). */
  kind: Uint8Array;
  thickness: Float32Array;
  /** All loose material over the bedrock (m): river, glacial and wind-blown sediment and peat. */
  cover: Float32Array;
  /** Bedrock weathered in place (m): soil over saprolite. */
  regolith: Float32Array;
  /** 0–1 sand-and-gravel share of the mineral cover (its texture). */
  sand: Float32Array;
  deltas: DeltaRecord[];
  budget: SedimentBudget;
}

/** Slope deposits only show as the surface unit where they gather this thick (m), at the foot of hill country. */
const COLLUVIUM_SHOWS = 15;

/**
 * The surface unit at each cell, youngest on top: peat, dunes and loess over
 * playa crusts and glacial deposits, over river sediment; where nothing was
 * laid down, bedrock weathered in place (see regolithDepth), thin where the
 * ice age scoured it fresh.
 */
export function finalizeDeposits(
  mesh: SphereMesh,
  ocean: Uint8Array,
  fl: ErosionResult,
  gl: GlacialDeposits,
  ae: AeolianDeposits,
  hydro: Hydrology,
  tempYear: Float32Array,
  precipYear: Float32Array,
  petYear: Float32Array,
  scour: Float32Array,
): Deposits {
  const n = mesh.n;
  const kind = new Uint8Array(n);
  const thickness = new Float32Array(n);
  const cover = new Float32Array(n);
  const regolith = new Float32Array(n);
  const sand = new Float32Array(n);
  const parts = fl.parts;
  for (let c = 0; c < n; c++) {
    const river = fl.deposit[c];
    const glacial = gl.till[c] + gl.moraine[c] + gl.outwash[c];
    const wind = ae.dunes[c] + ae.loess[c];
    const t = tempYear[c];
    const p = precipYear[c];
    let peat = 0;
    let playa = false;
    if (!ocean[c]) {
      const wet = hydro.wetland[c];
      if (wet === Wetland.Peatland) peat = 1.5 + 3 * smoothstep(-8, 4, t);
      else if (wet === Wetland.Swamp && t > 18 && p > 1800) peat = 4;
      const lk = hydro.lake[c];
      if (lk >= 0 && hydro.lakes[lk].kind !== LakeKind.Fresh) playa = true;
      else if (lk < 0 && parts[P_LAKE][c] > 10 && parts[P_LAKE][c] > 0.5 * river && p / Math.max(1, petYear[c]) < 0.15) playa = true;
    }
    const mineral = river + glacial + wind;
    cover[c] = mineral + peat;
    const sandV = fl.sand[c] + 0.85 * gl.outwash[c] + 0.55 * gl.moraine[c] + 0.35 * gl.till[c] + ae.dunes[c] + 0.15 * ae.loess[c];
    sand[c] = mineral > 0 ? Math.min(1, sandV / mineral) : 0.5;

    // Weathering in place; under a thick cover only the soil formed in the cover.
    const rg = regolithDepth(fl.lowering[c], fl.chem[c], fl.frost[c]);
    const fresh = 1 - 0.9 * scour[c];
    regolith[c] = Math.min(60, rg.soil * fresh + (mineral > 2 ? 0 : rg.saprolite * fresh * fresh));

    let k: Deposit = Deposit.Bedrock;
    let th = 0;
    if (ocean[c]) {
      if (river >= 2) {
        k = Deposit.Shelf;
        th = river;
      }
    } else if (peat > 0) {
      k = Deposit.Peat;
      th = peat;
    } else if (ae.dunes[c] >= 5 && ae.dunes[c] >= ae.loess[c]) {
      k = Deposit.Dunes;
      th = ae.dunes[c];
    } else if (ae.loess[c] >= 2) {
      k = Deposit.Loess;
      th = ae.loess[c];
    } else if (playa) {
      k = Deposit.Playa;
      th = Math.max(5, parts[P_LAKE][c]);
    } else if (gl.moraine[c] >= 5) {
      k = Deposit.Moraine;
      th = gl.moraine[c];
    } else if (gl.outwash[c] >= 3) {
      k = Deposit.Outwash;
      th = gl.outwash[c];
    } else if (gl.till[c] >= 2) {
      k = Deposit.Till;
      th = gl.till[c];
    } else if (river - parts[P_COLLUVIUM][c] >= 2 || parts[P_COLLUVIUM][c] >= COLLUVIUM_SHOWS) {
      let best = P_ALLUVIUM;
      for (const q of [P_FAN, P_DELTA, P_LAKE]) if (parts[q][c] > parts[best][c]) best = q;
      if (parts[P_COLLUVIUM][c] >= COLLUVIUM_SHOWS && parts[P_COLLUVIUM][c] > parts[best][c]) best = P_COLLUVIUM;
      k = best === P_FAN ? Deposit.Fan : best === P_DELTA ? Deposit.Delta : best === P_LAKE ? Deposit.Lake : best === P_COLLUVIUM ? Deposit.Colluvium : Deposit.Alluvium;
      th = river;
    } else if (regolith[c] >= 8) {
      k = Deposit.Residual;
      th = regolith[c];
    }
    kind[c] = k;
    thickness[c] = th;
  }
  const budget: SedimentBudget = { ...fl.budget, glacial: gl.budget, wind: ae.budget };
  return { kind, thickness, cover, regolith, sand, deltas: fl.deltas, budget };
}
