import { MinHeap } from '../core/heap';
import type { SphereMesh } from '../core/mesh';
import { DAYS_PER_MONTH, MONTHS } from './climate';
import { normCdf, type WaterBalance } from './waterBalance';

const SECONDS_PER_MONTH = DAYS_PER_MONTH * 86400;

/**
 * Ocean = water below sea level that belongs to a large connected body.
 * Small enclosed below-sea-level basins are land depressions (lakes or dry pans).
 */
export function oceanMask(mesh: SphereMesh, elev: Float32Array): Uint8Array {
  const { n, nbrStart, nbr, area } = mesh;
  const comp = new Int32Array(n).fill(-1);
  const compArea: number[] = [];
  const stack: number[] = [];
  for (let c = 0; c < n; c++) {
    if (elev[c] >= 0 || comp[c] >= 0) continue;
    const id = compArea.length;
    let a = 0;
    comp[c] = id;
    stack.push(c);
    while (stack.length) {
      const v = stack.pop()!;
      a += area[v];
      for (let k = nbrStart[v], e = nbrStart[v + 1]; k < e; k++) {
        const m = nbr[k];
        if (elev[m] < 0 && comp[m] < 0) {
          comp[m] = id;
          stack.push(m);
        }
      }
    }
    compArea.push(a);
  }
  let big = 0;
  for (const a of compArea) if (a > big) big = a;
  const ocean = new Uint8Array(n);
  for (let c = 0; c < n; c++) {
    const id = comp[c];
    if (id >= 0 && (compArea[id] >= 0.002 * 4 * Math.PI || compArea[id] >= big * 0.5)) ocean[c] = 1;
  }
  return ocean;
}

export interface Routing {
  /** Downstream cell (an ocean cell at the coast), or -1 for ocean cells. */
  receiver: Int32Array;
  /** Land cells in processing order: every receiver precedes its donors. */
  order: Int32Array;
  count: number;
  /** Elevation with depressions filled to their spill level. */
  fill: Float32Array;
  /** Depression id for cells below their spill level, else -1. */
  basin: Int32Array;
  /** Spill (outlet) cell and spill level per depression. */
  basinSpill: number[];
  basinLevel: number[];
}

/** Priority-flood (Barnes et al. 2014) routing from the coast inward. */
export function route(mesh: SphereMesh, elev: Float32Array, ocean: Uint8Array): Routing {
  const { n, nbrStart, nbr } = mesh;
  const fill = Float32Array.from(elev);
  const receiver = new Int32Array(n).fill(-1);
  const closed = new Uint8Array(n);
  const order = new Int32Array(n);
  const basin = new Int32Array(n).fill(-1);
  const basinSpill: number[] = [];
  const basinLevel: number[] = [];
  const heap = new MinHeap(4096);
  for (let c = 0; c < n; c++) {
    if (ocean[c]) {
      closed[c] = 1;
      continue;
    }
    for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
      const m = nbr[k];
      if (ocean[m]) {
        receiver[c] = m;
        closed[c] = 1;
        heap.push(elev[c], c);
        break;
      }
    }
  }
  let count = 0;
  while (heap.size > 0) {
    const c = heap.pop();
    order[count++] = c;
    const fc = fill[c];
    for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
      const m = nbr[k];
      if (closed[m]) continue;
      closed[m] = 1;
      receiver[m] = c;
      if (elev[m] <= fc) {
        fill[m] = fc + 1e-3;
        if (basin[c] >= 0) {
          basin[m] = basin[c];
        } else {
          basin[m] = basinSpill.length;
          basinSpill.push(c);
          basinLevel.push(fc);
        }
      } else {
        fill[m] = elev[m];
      }
      heap.push(fill[m], m);
    }
  }
  return { receiver, order, count, fill, basin, basinSpill, basinLevel };
}

/** Group depression cells by depression, each list sorted from the lowest cell up. */
export function depressionCells(r: Routing, elev: Float32Array): number[][] {
  const out: number[][] = Array.from({ length: r.basinSpill.length }, () => []);
  for (let i = 0; i < r.count; i++) {
    const c = r.order[i];
    if (r.basin[c] >= 0) out[r.basin[c]].push(c);
  }
  for (const cells of out) cells.sort((a, b) => elev[a] - elev[b]);
  return out;
}

/**
 * The priority flood can enter one depression from several rim cells at the
 * same height and split it into several basins. Touching basins with matching
 * spill levels are one body of water: group them.
 */
function basinGroups(mesh: SphereMesh, r: Routing): { groupOf: Int32Array; groups: number } {
  const nb = r.basinSpill.length;
  const parent = Int32Array.from({ length: nb }, (_, i) => i);
  const find = (i: number): number => {
    while (parent[i] !== i) i = parent[i] = parent[parent[i]];
    return i;
  };
  const { nbrStart, nbr } = mesh;
  for (let i = 0; i < r.count; i++) {
    const c = r.order[i];
    const b = r.basin[c];
    if (b < 0) continue;
    for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
      const b2 = r.basin[nbr[k]];
      if (b2 < 0 || b2 === b || Math.abs(r.basinLevel[b] - r.basinLevel[b2]) > 1) continue;
      const ra = find(b);
      const rb = find(b2);
      if (ra !== rb) parent[ra] = rb;
    }
  }
  const groupOf = new Int32Array(nb);
  const id = new Int32Array(nb).fill(-1);
  let groups = 0;
  for (let b = 0; b < nb; b++) {
    const root = find(b);
    if (id[root] < 0) id[root] = groups++;
    groupOf[b] = id[root];
  }
  return { groupOf, groups };
}

/**
 * Cells of each depression group in the order rising water covers them:
 * outward from the lowest point, always over the lowest shore next, so every
 * lake is one connected body even on a dead-flat basin floor.
 */
function floodOrder(mesh: SphereMesh, r: Routing, elev: Float32Array, groupOf: Int32Array, groups: number): number[][] {
  const { nbrStart, nbr } = mesh;
  const lowest = new Int32Array(groups).fill(-1);
  for (let i = 0; i < r.count; i++) {
    const c = r.order[i];
    if (r.basin[c] < 0) continue;
    const g = groupOf[r.basin[c]];
    if (lowest[g] < 0 || elev[c] < elev[lowest[g]]) lowest[g] = c;
  }
  const seen = new Uint8Array(mesh.n);
  const heap = new MinHeap(256);
  const out: number[][] = [];
  for (let g = 0; g < groups; g++) {
    const cells: number[] = [];
    out.push(cells);
    const start = lowest[g];
    if (start < 0) continue;
    let tick = 0;
    heap.clear();
    heap.push(elev[start], start);
    seen[start] = 1;
    while (heap.size > 0) {
      const c = heap.pop();
      cells.push(c);
      for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
        const m = nbr[k];
        if (seen[m] || r.basin[m] < 0 || groupOf[r.basin[m]] !== g) continue;
        seen[m] = 1;
        // Ties (flat floors) flood in arrival order.
        heap.push(elev[m] + 1e-7 * ++tick, m);
      }
    }
  }
  return out;
}

/** Mesh detail at which the cell-scale processes were tuned; finer and coarser meshes are scaled to match it. */
export const REFERENCE_CELLS = 320000;
/**
 * Mesh detail relative to the reference, for those scalings. Below ~80k cells
 * the climate mesh coarsens too and the scalings no longer hold, so drafts
 * keep the 80k corrections.
 */
export const detailRatio = (n: number): number => Math.max(n, 80000) / REFERENCE_CELLS;
/** Converts slopes between neighboring cells of an n-cell mesh to those of the reference detail. */
export const slopeScale = (n: number): number => detailRatio(n) ** -0.225;

// ------------------------------------------------------------------ rivers and lakes

/** Open water evaporates a little faster than the land reference surface. */
const LAKE_EVAPORATION = 1.1;
/**
 * Spread of ground heights within a cell, as a share of its sub-cell relief
 * (the relief the renderer draws, plus a floor for the smoothest plains).
 */
const SUBCELL_SPREAD = 0.58;
const SUBCELL_FLOOR = 15;
/** A shore cell this much under a lake's water (share of its ground) drains into that lake. */
const SHORE_SHARE = 0.3;
/** Lake stage above the outlet sill that passes the mean outflow (m): sets how slowly a lake answers its inflow. */
const LAKE_STAGE = 1;
/**
 * Bankfull (≈1.5-year) flood over the wettest month's mean flow: storms,
 * rain on snow and wet years add to the monthly mean, most on small catchments
 * (the flood wave flattens downstream as tributaries fall out of step) and not
 * at all for water that has been through a lake.
 */
const FLOOD_EXCESS = 1.5;
const FLOOD_AREA = 2000;
/** In dry months a small river loses up to this share of its flow per 80 km to seepage and evaporation. */
const CHANNEL_LOSS = 0.5;
const CHANNEL_LOSS_Q = 25;

export const enum LakeKind {
  Fresh = 0,
  Saline = 1,
  SaltFlat = 2,
}

export interface Lake {
  kind: LakeKind;
  /** Water surface, or the salt flat's floor (m). */
  level: number;
  areaKm2: number;
  /** Mean inflow and outflow (m³/s). */
  inflow: number;
  outflow: number;
  /** The cell the lake drains (or would drain) through. */
  outlet: number;
}

export const enum Wetland {
  None = 0,
  Floodplain = 1,
  InlandDelta = 2,
  Peatland = 3,
  Swamp = 4,
  /** A closed hollow in a climate too wet for a salt flat but too dry for a lasting lake. */
  Marsh = 5,
}

export interface Hydrology {
  routing: Routing;
  ocean: Uint8Array;
  /** Mean annual discharge leaving each cell (m³/s), after channel losses and lakes. */
  discharge: Float32Array;
  /** Monthly discharge (m³/s), index c*12+m. */
  dischargeMonthly: Float32Array;
  /**
   * Bankfull discharge (m³/s): the ≈1.5-year flood a channel is built to carry,
   * from the wettest month's flow and how flashy the river is. Sets channel size.
   */
  bankfull: Float32Array;
  /** Months a year the river really flows. */
  flowMonths: Uint8Array;
  /** Share of the flow that is snow and glacier melt. */
  meltShare: Float32Array;
  /** Lake id for lake and salt-flat cells, else -1. */
  lake: Int32Array;
  lakes: Lake[];
  /** Annual runoff generated in each cell (mm). */
  runoff: Float32Array;
  /** Drainage basin id (the cell where its water leaves: a river mouth or a closed lake). */
  drainage: Int32Array;
  /** Land whose water ends in a closed lake or salt flat instead of the sea. */
  endorheic: Uint8Array;
  wetland: Uint8Array;
}

/**
 * Rivers, lakes and wetlands from the monthly water balance. Monthly runoff is
 * routed down the flow network. In dry months rivers lose water to their beds,
 * so small desert streams wither and die. Each depression fills until it
 * spills (a freshwater lake that evens out the seasons downstream) or until
 * evaporation from its surface balances the inflow (a salt lake), and a basin
 * too dry for even that holds only a salt flat. Rising water covers each cell
 * gradually, as its ground spans heights about its mean (`rough`, its sub-cell
 * relief), so a shallow hollow brims and spills long before it would drown
 * every cell of its floor.
 */
export function computeHydrology(
  mesh: SphereMesh,
  elev: Float32Array,
  ocean: Uint8Array,
  wb: WaterBalance,
  precip: Float32Array,
  temp: Float32Array,
  slope: Float32Array,
  rough: Float32Array,
  radiusKm: number,
  routing: Routing = route(mesh, elev, ocean),
  shoresDone = false,
): Hydrology {
  const { n, area, x, y, z } = mesh;
  const { order, receiver, basin, count } = routing;
  const areaM2 = (c: number) => area[c] * radiusKm * radiusKm * 1e6;
  const Q = new Float64Array(n * MONTHS);
  const meltQ = new Float64Array(n);
  const lossShare = new Float32Array(n);
  const terminal = new Uint8Array(n);
  const lake = new Int32Array(n).fill(-1);
  const lakes: Lake[] = [];
  const { groupOf, groups } = basinGroups(mesh, routing);
  const pits = floodOrder(mesh, routing, elev, groupOf, groups);
  // A group's lake is settled at the last of its outlets, once all its inflow is in.
  const spillOf = new Float64Array(groups).fill(Infinity);
  const outlets: number[][] = Array.from({ length: groups }, () => []);
  const remaining = new Int32Array(groups);
  for (let b = 0; b < routing.basinSpill.length; b++) {
    const g = groupOf[b];
    spillOf[g] = Math.min(spillOf[g], routing.basinLevel[b]);
    remaining[g]++;
  }
  const held = new Float64Array(groups * MONTHS);
  const heldMelt = new Float64Array(groups);
  const marsh = new Uint8Array(n);
  const precipYearOf = (c: number): number => {
    let s = 0;
    for (let m = 0; m < MONTHS; m++) s += precip[c * MONTHS + m];
    return s;
  };

  /** Extra loss (m³/s) in month m if cell c were flooded: open-water evaporation less rain, plus the runoff it no longer yields. */
  const floodCost = (c: number, m: number): number => {
    const k = c * MONTHS + m;
    return ((LAKE_EVAPORATION * wb.pet[k] - precip[k] + wb.runoff[k]) / 1000) * (areaM2(c) / SECONDS_PER_MONTH);
  };
  const annualCost = (c: number): number => {
    let s = 0;
    for (let m = 0; m < MONTHS; m++) s += floodCost(c, m);
    return s / MONTHS;
  };
  /** Share of cell c under water at level z: its ground spans heights about its mean. */
  const wetShare = (c: number, z: number): number => normCdf((z - elev[c]) / (SUBCELL_SPREAD * (rough[c] + SUBCELL_FLOOR)));
  /** Water that has passed through a lake (m³/s, mean): its floods are damped. */
  const settled = new Float64Array(n);

  const settleLake = (g: number, outlet: number): void => {
    const cells = pits[g];
    const spill = spillOf[g];
    if (cells.length === 0 || spill - elev[cells[0]] < 3) return;
    const o = outlet * MONTHS;
    let inflow = 0;
    for (let m = 0; m < MONTHS; m++) inflow += Q[o + m] / MONTHS;
    const costs = cells.map(annualCost);
    /** What the lake loses at level z (m³/s): open-water evaporation less rain, and the runoff of the ground it drowns. */
    const loss = (z: number): number => {
      let s = 0;
      for (let j = 0; j < cells.length; j++) s += wetShare(cells[j], z) * costs[j];
      return s;
    };
    const id = lakes.length;
    const R2 = radiusKm * radiusKm;
    if (loss(spill) <= inflow) {
      // Fills and spills: the lake evens out the seasons for the river below.
      let areaKm2 = 0;
      const share = cells.map((c) => wetShare(c, spill));
      for (let j = 0; j < cells.length; j++) {
        lake[cells[j]] = id;
        areaKm2 += area[cells[j]] * R2 * share[j];
      }
      const raw = new Float64Array(MONTHS);
      let mean = 0;
      for (let m = 0; m < MONTHS; m++) {
        let lost = 0;
        for (let j = 0; j < cells.length; j++) lost += share[j] * floodCost(cells[j], m);
        raw[m] = Q[o + m] - lost;
        mean += raw[m] / MONTHS;
      }
      mean = Math.max(0, mean);
      // A lake answers its inflow slowly: a linear reservoir whose time scale is
      // the water a stage rise of LAKE_STAGE holds over the outflow. A lake on a
      // big river passes its floods on; a big lake on a small one flattens them.
      const tau = mean > 0 ? (areaKm2 * 1e6 * LAKE_STAGE) / mean : Infinity;
      const keep = Math.exp(-SECONDS_PER_MONTH / tau);
      let out = mean;
      for (let pass = 0; pass < 3; pass++) {
        for (let m = 0; m < MONTHS; m++) {
          out = keep * out + (1 - keep) * raw[m];
          if (pass === 2) raw[m] = Math.max(0, out);
        }
      }
      let sum = 0;
      for (let m = 0; m < MONTHS; m++) sum += raw[m] / MONTHS;
      for (let m = 0; m < MONTHS; m++) Q[o + m] = sum > 0 ? (raw[m] * mean) / sum : 0;
      meltQ[outlet] *= inflow > 0 ? mean / inflow : 0;
      settled[outlet] = mean;
      lakes.push({ kind: LakeKind.Fresh, level: spill, areaKm2, inflow, outflow: mean, outlet });
      return;
    }
    // Closed basin: nothing flows on. Salt builds up only where the climate is
    // dry; in wetter country a closed lake stays fresh.
    for (const out of outlets[g]) terminal[out] = 1;
    for (let m = 0; m < MONTHS; m++) Q[o + m] = 0;
    meltQ[outlet] = 0;
    const floorCell = cells[0];
    const moisture = wb.petYear[floorCell] > 0 ? precipYearOf(floorCell) / wb.petYear[floorCell] : 1;
    // Short of its brim, the lake fills the floor cell by cell, lowest first,
    // until its losses match the inflow (water reaches a cell's hollows only
    // once the lake does, so the whole floor doesn't flood at once).
    let k = -1;
    let cost = 0;
    for (let j = 0; j < cells.length; j++) {
      if (cost + costs[j] > inflow) break;
      cost += costs[j];
      k = j;
    }
    if (k >= 0) {
      let areaKm2 = 0;
      let top = -Infinity;
      for (let j = 0; j <= k; j++) {
        lake[cells[j]] = id;
        areaKm2 += area[cells[j]] * R2;
        top = Math.max(top, elev[cells[j]]);
      }
      const kind = moisture < 0.6 ? LakeKind.Saline : LakeKind.Fresh;
      lakes.push({ kind, level: Math.min(spill, top + 2), areaKm2, inflow, outflow: 0, outlet });
    } else if (inflow > 0.15 * annualCost(floorCell)) {
      // Too little water for a lasting lake: floods now and then spread over
      // the lowest ground (sized for floods of about three years' inflow). In
      // dry country they evaporate to a salt flat; in wetter country the
      // hollow is a marsh.
      const floor = elev[floorCell];
      let areaKm2 = 0;
      let wet = 0;
      const flat = moisture < 0.35;
      for (const c of cells) {
        if (elev[c] > floor + 2 || (areaKm2 > 0 && wet + annualCost(c) > 3 * inflow)) break;
        wet += annualCost(c);
        if (flat) lake[c] = id;
        else marsh[c] = 1;
        areaKm2 += area[c] * R2;
      }
      if (flat) lakes.push({ kind: LakeKind.SaltFlat, level: floor + 2, areaKm2, inflow, outflow: 0, outlet });
    }
  };

  const heldSettled = new Float64Array(groups);
  const upArea = new Float64Array(n);
  // A hollow too shallow to hold a lake is just flat ground: rivers cross it
  // along their own courses instead of having their water pooled and let out
  // through whichever of its outlets comes last.
  const shallow = new Uint8Array(groups);
  for (let g = 0; g < groups; g++) shallow[g] = pits[g].length === 0 || spillOf[g] - elev[pits[g][0]] < 3 ? 1 : 0;
  for (let i = count - 1; i >= 0; i--) {
    const c = order[i];
    const rc = receiver[c];
    const o = c * MONTHS;
    const a = areaM2(c);
    for (let m = 0; m < MONTHS; m++) Q[o + m] += ((wb.runoff[o + m] / 1000) * a) / SECONDS_PER_MONTH;
    meltQ[c] += ((wb.meltYear[c] / 1000) * a) / (SECONDS_PER_MONTH * MONTHS);
    upArea[c] += a / 1e6;
    const b = basin[c];
    if (b >= 0 && !shallow[groupOf[b]]) {
      if (rc < 0 || basin[rc] !== b) {
        // A basin outlet: hold its water until every outlet of its group is in.
        const g = groupOf[b];
        outlets[g].push(c);
        for (let m = 0; m < MONTHS; m++) {
          held[g * MONTHS + m] += Q[o + m];
          Q[o + m] = 0;
        }
        heldMelt[g] += meltQ[c];
        meltQ[c] = 0;
        heldSettled[g] += settled[c];
        settled[c] = 0;
        if (--remaining[g] === 0) {
          for (let m = 0; m < MONTHS; m++) Q[o + m] = held[g * MONTHS + m];
          meltQ[c] = heldMelt[g];
          settled[c] = heldSettled[g];
          settleLake(g, c);
          if (terminal[c]) settled[c] = 0;
        }
      }
    } else if (rc >= 0) {
      // Channel losses where evaporation outpaces rain; small rivers suffer most.
      const dxKm = Math.acos(Math.min(1, x[c] * x[rc] + y[c] * y[rc] + z[c] * z[rc])) * radiusKm;
      let before = 0;
      let after = 0;
      for (let m = 0; m < MONTHS; m++) {
        const q = Q[o + m];
        const e = wb.pet[o + m];
        const arid = e > 0 ? Math.max(0, Math.min(1, 1 - precip[o + m] / e)) : 0;
        const f = Math.min(0.9, ((CHANNEL_LOSS * arid * CHANNEL_LOSS_Q) / (q + CHANNEL_LOSS_Q)) * (dxKm / 80));
        before += q;
        Q[o + m] = q * (1 - f);
        after += Q[o + m];
      }
      lossShare[c] = before > 0 ? 1 - after / before : 0;
      meltQ[c] *= 1 - lossShare[c];
      settled[c] *= 1 - lossShare[c];
    }
    if (rc >= 0 && !ocean[rc]) {
      const ro = rc * MONTHS;
      for (let m = 0; m < MONTHS; m++) Q[ro + m] += Q[o + m];
      meltQ[rc] += meltQ[c];
      settled[rc] += settled[c];
      upArea[rc] += upArea[c];
    }
  }

  const discharge = new Float32Array(n);
  const dischargeMonthly = new Float32Array(n * MONTHS);
  const bankfull = new Float32Array(n);
  const flowMonths = new Uint8Array(n);
  const meltShare = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    if (ocean[c]) continue;
    const o = c * MONTHS;
    let mean = 0;
    let peak = 0;
    for (let m = 0; m < MONTHS; m++) {
      dischargeMonthly[o + m] = Q[o + m];
      mean += Q[o + m] / MONTHS;
      peak = Math.max(peak, Q[o + m]);
    }
    discharge[c] = mean;
    // Floods above the wettest month's mean: smaller on big catchments, and
    // none for the share of the flow a lake has already evened out.
    const fresh = mean > 0 ? Math.max(0, Math.min(1, 1 - settled[c] / mean)) : 1;
    bankfull[c] = peak * (1 + FLOOD_EXCESS * fresh * (FLOOD_AREA / (FLOOD_AREA + upArea[c])) ** 0.4);
    let months = 0;
    for (let m = 0; m < MONTHS; m++) if (Q[o + m] >= Math.max(0.05, 0.03 * mean)) months++;
    flowMonths[c] = months;
    meltShare[c] = mean > 0 ? Math.min(1, meltQ[c] / mean) : 0;
  }

  // Land a lake covers in good part (its ground dipping below the water) drains
  // into that lake, though the cell's mean may stand above it and the steepest
  // way on lead elsewhere: route such shores into the lake and run the water
  // again. Shore cells stand above their lake's spill level, so the lake cell
  // comes before them in the routing order.
  if (!shoresDone) {
    const pos = new Int32Array(n);
    for (let i = 0; i < count; i++) pos[order[i]] = i;
    const { nbrStart, nbr } = mesh;
    let moved = 0;
    for (let q = 0; q < n; q++) {
      const L = lake[q];
      if (L < 0 || lakes[L].kind === LakeKind.SaltFlat) continue;
      for (let k = nbrStart[q], e = nbrStart[q + 1]; k < e; k++) {
        const c = nbr[k];
        const rc = receiver[c];
        if (ocean[c] || lake[c] >= 0 || basin[c] >= 0 || rc < 0 || lake[rc] === L || pos[q] >= pos[c]) continue;
        if (wetShare(c, lakes[L].level) < SHORE_SHARE) continue;
        receiver[c] = q;
        moved++;
      }
    }
    if (moved > 0) return computeHydrology(mesh, elev, ocean, wb, precip, temp, slope, rough, radiusKm, routing, true);
  }

  // Drainage basins: each land cell is labelled by where its water leaves.
  const drainage = new Int32Array(n).fill(-1);
  const endorheic = new Uint8Array(n);
  for (let i = 0; i < count; i++) {
    const c = order[i];
    const rc = receiver[c];
    if (rc < 0 || ocean[rc] || terminal[c]) {
      drainage[c] = c;
      endorheic[c] = terminal[c];
    } else {
      drainage[c] = drainage[rc];
      endorheic[c] = endorheic[rc];
    }
  }

  // Wetlands on flat ground: river floodplains, deltas of rivers dying in dry
  // country, peat bogs in cold waterlogged lowlands, and warm swamps fed by
  // groundwater.
  const wetland = new Uint8Array(n);
  for (let c = 0; c < n; c++) {
    if (ocean[c] || lake[c] >= 0 || wb.glacier[c] || (slope[c] > 1.2e-3 && !marsh[c])) continue;
    const o = c * MONTHS;
    let tAnn = 0;
    let p = 0;
    let qMax = 0;
    for (let m = 0; m < MONTHS; m++) {
      tAnn += temp[o + m] / MONTHS;
      p += precip[o + m];
      qMax = Math.max(qMax, Q[o + m]);
    }
    const q = discharge[c];
    if (marsh[c]) wetland[c] = Wetland.Marsh;
    else if (q > 150 && (qMax > 1.8 * q || elev[c] < 150)) wetland[c] = Wetland.Floodplain;
    else if (lossShare[c] > 0.25 && q / (1 - lossShare[c]) > 15) wetland[c] = Wetland.InlandDelta;
    else if (tAnn < 4 && wb.soil[c] > 0.75 && wb.runoffYear[c] > 80) wetland[c] = Wetland.Peatland;
    else if (tAnn > 16 && elev[c] < 250 && wb.ground[c] > 80 && p > wb.petYear[c]) wetland[c] = Wetland.Swamp;
  }
  // A floodplain spans about a reference cell's width (~40 km) on any mesh: on
  // finer ones it spreads from its river's cells onto their flat neighbors.
  const reach = 0.46 * mesh.spacing * Math.sqrt(n / REFERENCE_CELLS);
  if (n > 1.5 * REFERENCE_CELLS) {
    const { nbrStart, nbr, nbrDist } = mesh;
    const plain: number[] = [];
    for (let c = 0; c < n; c++) if (wetland[c] === Wetland.Floodplain) plain.push(c);
    for (const c of plain) {
      for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
        const m = nbr[k];
        if (nbrDist[k] > reach || wetland[m] !== Wetland.None || ocean[m] || lake[m] >= 0 || wb.glacier[m] || slope[m] > 1.2e-3) continue;
        wetland[m] = Wetland.Floodplain;
      }
    }
  }

  return {
    routing,
    ocean,
    discharge,
    dischargeMonthly,
    bankfull,
    flowMonths,
    meltShare,
    lake,
    lakes,
    runoff: wb.runoffYear,
    drainage,
    endorheic,
    wetland,
  };
}
