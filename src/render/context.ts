import { smoothstep } from '../core/math';
import type { SphereMesh } from '../core/mesh';
import { seedFor } from '../core/rng';
import { MAX_GLACIAL_EROSION } from '../world/glaciers';
import type { World } from '../world/world';

/**
 * Everything the pixel renderer needs, as plain typed arrays so it can be
 * posted (or shared) to tile workers. MeshData is fixed for a world; FieldData
 * is rebuilt after every change.
 */

/** Kernel support radius, in mean cell spacings. */
export const KERNEL_RADIUS = 1.25;

export interface MeshData {
  id: string;
  n: number;
  nTri: number;
  /** Mean neighbor distance (radians). */
  spacing: number;
  x: Float64Array;
  y: Float64Array;
  z: Float64Array;
  tri: Int32Array;
  halfedge: Int32Array;
  /** Triangle near the center of each lat/lon bin, to seed point location. */
  gridW: number;
  gridH: number;
  gridTri: Int32Array;
  /** Per triangle (CSR): every cell whose kernel can reach into the triangle. */
  gStart: Int32Array;
  gCell: Int32Array;
}

export interface FieldData {
  meshId: string;
  version: number;
  noiseSeed: number;
  radiusKm: number;
  riverThreshold: number;
  elev: Float32Array;
  /** Slope-limited elevation gradient per cell (3 per cell, m per radian, tangent). */
  grad: Float32Array;
  /** Amplitude of sub-cell relief (m). */
  amp: Float32Array;
  /** 0–1: how much of the sub-cell relief is sharp-crested (young mountains). */
  mtn: Float32Array;
  ocean: Uint8Array;
  lake: Int32Array;
  /** Per lake: water surface (or salt-flat floor) and kind (0 fresh, 1 saline, 2 salt flat). */
  lakeLevel: Float32Array;
  lakeKind: Uint8Array;
  /** 0–1: how far evaporation outstrips rain (green ribbons follow rivers through dry land). */
  dryness: Float32Array;
  /** Ice thickness (m, cell average), share of the cell under ice, and 0–1 how hard it melts (bare blue ice in the ablation zone). */
  ice: Float32Array;
  iceCover: Float32Array;
  iceMelt: Float32Array;
  /** Floating ice shelf. */
  shelf: Uint8Array;
  /** 0–1: ice-age scouring of lowlands (country dotted with small lakes). */
  scour: Float32Array;
  /** 0–1: ice-age glaciers cut this coast into fjords. */
  fjord: Float32Array;
  reef: Uint8Array;
  color: Uint8Array;
  biome: Uint8Array;
  koppen: Uint8Array;
  /** Bedrock class (geology.ts Rock). */
  rock: Uint8Array;
  /** Surface deposit over the bedrock (sediment.ts Deposit). */
  deposit: Uint8Array;
  plate: Int16Array;
  plateColor: Uint8Array;
  ctype: Uint8Array;
  thick: Float32Array;
  age: Float32Array;
  bndType: Uint8Array;
  drainage: Int32Array;
  /** Monthly surface temperature (°C) and precipitation (mm), c*12+m. */
  temp: Float32Array;
  precip: Float32Array;
  /** River network: chains of cells in downstream order (CSR). */
  rivStart: Int32Array;
  rivCell: Int32Array;
  /** Discharge (m³/s) at each chain node. */
  rivQ: Float32Array;
  /** Bankfull discharge (m³/s) at each chain node: the flood the channel is built for, which sets its width. */
  rivBank: Float32Array;
  /**
   * Meander half-waves counted along each chain up to its node: a segment holds
   * a whole number of them (about twelve channel widths per wave), so the
   * channel's meanders cross every node.
   */
  rivWave: Int32Array;
  /** Months a year each chain node flows (fewer = intermittent, drawn dashed). */
  rivFlow: Uint8Array;
  /** 1 = the chain ends in the sea (extend its mouth to the coastline). */
  rivSea: Uint8Array;
  /**
   * Per chain: the cell after its last node, shaping the curve's end: at a
   * confluence the trunk's next node (the tributary curves into the trunk's
   * flow at an acute angle), at a mouth the open sea beyond; else -1.
   */
  rivNext: Int32Array;
  /**
   * Per chain: 1 = the largest of the rivers ending in its sea cell, whose
   * valley alone runs on past it (rivers meeting at a mouth share one run-on).
   */
  rivRun: Uint8Array;
  /** Per chain ending at a confluence: the trunk's node there (the segment leaving it), else -1. */
  rivJoin: Int32Array;
  /** Nodes that start a segment, upstream before downstream (every segment into a node comes before the one leaving it). */
  rivOrder: Int32Array;
  /** 1 = the segment leaving this node passes within reach of the sea (its course may meet the drawn coastline). */
  rivCoast: Uint8Array;
  /** Bounding box of the segment leaving each node: lat0, lat1, lon0, lon1 (radians, lon may exceed ±π). */
  rivBox: Float32Array;
}

export interface RenderContext {
  mesh: MeshData;
  fields: FieldData;
}

const meshCache = new WeakMap<SphereMesh, MeshData>();

export function meshData(mesh: SphereMesh, id: string): MeshData {
  let md = meshCache.get(mesh);
  if (!md || md.id !== id) {
    md = buildMeshData(mesh, id);
    meshCache.set(mesh, md);
  }
  return md;
}

function buildMeshData(mesh: SphereMesh, id: string): MeshData {
  const { n, nTri, tri, x, y, z, nbrStart, nbr, cc } = mesh;
  const r = KERNEL_RADIUS * mesh.spacing;
  const gStart = new Int32Array(nTri + 1);
  let gCell = new Int32Array(nTri * 12);
  const mark = new Int32Array(n).fill(-1);
  let k = 0;
  for (let t = 0; t < nTri; t++) {
    // Candidates: the 1-rings of the three corners, kept if their kernel can
    // reach the triangle's circumcircle.
    const cx = cc[3 * t];
    const cy = cc[3 * t + 1];
    const cz = cc[3 * t + 2];
    const a = tri[3 * t];
    const rc = Math.acos(Math.min(1, cx * x[a] + cy * y[a] + cz * z[a]));
    const lim = Math.cos(Math.min(Math.PI, r + rc));
    gStart[t] = k;
    if (k + 64 > gCell.length) {
      const g2 = new Int32Array(gCell.length * 2);
      g2.set(gCell);
      gCell = g2;
    }
    for (let s = 0; s < 3; s++) {
      const v = tri[3 * t + s];
      for (let j = nbrStart[v] - 1, e = nbrStart[v + 1]; j < e; j++) {
        const q = j < nbrStart[v] ? v : nbr[j];
        if (mark[q] === t) continue;
        mark[q] = t;
        if (cx * x[q] + cy * y[q] + cz * z[q] >= lim) gCell[k++] = q;
      }
    }
  }
  gStart[nTri] = k;
  const gridTri = new Int32Array(mesh.gridCell.length);
  for (let i = 0; i < gridTri.length; i++) gridTri[i] = mesh.cellTri[mesh.gridCell[i]];
  return {
    id, n, nTri, spacing: mesh.spacing,
    x, y, z, tri, halfedge: mesh.halfedge,
    gridW: mesh.gridW, gridH: mesh.gridH, gridTri,
    gStart, gCell: gCell.slice(0, k),
  };
}

/**
 * Least-squares gradient per cell, limited (Barth–Jespersen) so the local
 * plane never overshoots its neighbors in any direction: no false ridges or
 * pits between cells. Neighbors across the coastline count as sea level, so
 * a seafloor plane rises only to the shore and a land plane slopes down to it
 * (left free, a shelf cell beside deep water could tilt hundreds of metres
 * above sea level toward the land and flatten the sea there into a sheet).
 */
function limitedGradient(mesh: SphereMesh, elev: Float32Array, ocean: Uint8Array): Float32Array {
  const { n, x, y, z, nbrStart, nbr } = mesh;
  const out = new Float32Array(3 * n);
  const us = new Float64Array(64);
  const vs = new Float64Array(64);
  for (let c = 0; c < n; c++) {
    const X = x[c];
    const Y = y[c];
    const Z = z[c];
    let ex = -Y;
    let ey = X;
    let el = Math.hypot(ex, ey);
    if (el < 1e-9) {
      ex = 1;
      ey = 0;
      el = 1;
    }
    ex /= el;
    ey /= el;
    const nx = -Z * ey;
    const ny = Z * ex;
    const nz = X * ey - Y * ex;
    const e0 = elev[c];
    const oc = ocean[c];
    let suu = 0;
    let suv = 0;
    let svv = 0;
    let sue = 0;
    let sve = 0;
    let cnt = 0;
    let emax = e0;
    let emin = e0;
    for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
      const m = nbr[k];
      if (cnt >= 64) break;
      const em = ocean[m] === oc ? elev[m] : 0;
      const dx = x[m] - X;
      const dy = y[m] - Y;
      const dz = z[m] - Z;
      const u = dx * ex + dy * ey;
      const v = dx * nx + dy * ny + dz * nz;
      const de = em - e0;
      const w = 1 / (u * u + v * v);
      suu += w * u * u;
      suv += w * u * v;
      svv += w * v * v;
      sue += w * u * de;
      sve += w * v * de;
      us[cnt] = u;
      vs[cnt] = v;
      cnt++;
      if (em > emax) emax = em;
      if (em < emin) emin = em;
    }
    const det = suu * svv - suv * suv;
    if (cnt < 2 || Math.abs(det) < 1e-30) continue;
    let gu = (sue * svv - sve * suv) / det;
    let gv = (sve * suu - sue * suv) / det;
    let phi = 1;
    for (let i = 0; i < cnt; i++) {
      const pred = 0.6 * (gu * us[i] + gv * vs[i]);
      if (pred > 1e-6) phi = Math.min(phi, (emax - e0) / pred);
      else if (pred < -1e-6) phi = Math.min(phi, (emin - e0) / pred);
    }
    gu *= phi;
    gv *= phi;
    out[3 * c] = gu * ex + gv * nx;
    out[3 * c + 1] = gu * ey + gv * ny;
    out[3 * c + 2] = gv * nz;
  }
  return out;
}

/**
 * Atoll cells hold the reef's sea-level elevation. For drawing, they take the
 * depth of the seafloor around them instead; the renderer builds each reef
 * ring on top, so the ocean between atolls stays deep.
 */
function atollsSubmerged(mesh: SphereMesh, elev: Float32Array, reef: Uint8Array): Float32Array {
  const atolls: number[] = [];
  for (let c = 0; c < mesh.n; c++) if (reef[c] === 1) atolls.push(c);
  if (atolls.length === 0) return elev;
  const out = Float32Array.from(elev);
  for (const c of atolls) out[c] = -1500;
  const { nbrStart, nbr } = mesh;
  for (let it = 0; it < 4; it++) {
    for (const c of atolls) {
      let s = 0;
      let k = 0;
      for (let j = nbrStart[c], e = nbrStart[c + 1]; j < e; j++) {
        if (out[nbr[j]] >= 0) continue;
        s += out[nbr[j]];
        k++;
      }
      if (k) out[c] = Math.min(-200, s / k);
    }
  }
  return out;
}

/** Lowest discharge drawn at the deepest zoom, as a fraction of the river threshold. */
export const MIN_STREAM_FRACTION = 1 / 128;
/** Hydraulic geometry: a channel is about this many times √Q_bankfull metres wide. */
export const CHANNEL_WIDTH = 4;
/** Meander wavelength in channel widths. */
const MEANDER_WAVE = 12;

function buildRivers(
  world: World,
): Pick<FieldData, 'rivStart' | 'rivCell' | 'rivQ' | 'rivBank' | 'rivWave' | 'rivFlow' | 'rivSea' | 'rivNext' | 'rivRun' | 'rivJoin' | 'rivOrder' | 'rivCoast' | 'rivBox'> {
  const mesh = world.fine;
  const { n, x, y, z, lat, lon, nbrStart, nbr } = mesh;
  const { discharge, bankfull, lake, routing, flowMonths } = world.hydro;
  const recv = routing.receiver;
  const ocean = world.ocean;
  const thr = Math.max(1, world.params.riverThreshold * MIN_STREAM_FRACTION);
  const isRiver = new Uint8Array(n);
  for (let c = 0; c < n; c++) isRiver[c] = !ocean[c] && lake[c] < 0 && discharge[c] >= thr ? 1 : 0;
  // Each chain is one stream (Horton): from its source down its main stem,
  // through every confluence where it is the larger branch, to the node where
  // it joins a larger stream, the sea or a lake. Trunks then run unbroken
  // through their confluences and tributaries end on them.
  const mainDonor = new Int32Array(n).fill(-1);
  for (let c = 0; c < n; c++) {
    const r = recv[c];
    if (!isRiver[c] || r < 0 || !isRiver[r]) continue;
    if (mainDonor[r] < 0 || discharge[c] > discharge[mainDonor[r]]) mainDonor[r] = c;
  }
  /**
   * Past its mouth a river heads for open water: the sea cell's neighbor with
   * the most sea around it, then the deepest. It is the same for every river
   * ending in that cell, so rivers meeting at a mouth run on together (each
   * going on along its own course could run along the shore, across land the
   * coastline noise added, or fan out into a star of fjord troughs).
   */
  const seaward = (o: number): number => {
    let best = -1;
    let bestScore = -Infinity;
    for (let k = nbrStart[o], e = nbrStart[o + 1]; k < e; k++) {
      const m = nbr[k];
      if (!ocean[m]) continue;
      let open = 0;
      for (let j = nbrStart[m], f = nbrStart[m + 1]; j < f; j++) open += ocean[nbr[j]];
      const score = open - world.elev[m] / 1e5;
      if (score > bestScore) {
        bestScore = score;
        best = m;
      }
    }
    return best;
  };
  const start: number[] = [0];
  const cells: number[] = [];
  const qs: number[] = [];
  const banks: number[] = [];
  const flows: number[] = [];
  const sea: number[] = [];
  const next: number[] = [];
  for (let c0 = 0; c0 < n; c0++) {
    if (!isRiver[c0] || mainDonor[c0] >= 0) continue;
    const chain: number[] = [];
    // A river leaving a lake starts inside it, so it meets the shoreline.
    for (let k = nbrStart[c0], e = nbrStart[c0 + 1]; k < e; k++) {
      const m = nbr[k];
      if (lake[m] >= 0 && recv[m] === c0) {
        chain.push(m);
        break;
      }
    }
    chain.push(c0);
    let cur = c0;
    let toSea = 0;
    let joins = false;
    for (let guard = 0; guard < n; guard++) {
      const r = recv[cur];
      if (r < 0) break;
      chain.push(r);
      if (ocean[r]) toSea = 1;
      if (ocean[r] || lake[r] >= 0 || !isRiver[r]) break;
      if (mainDonor[r] !== cur) {
        joins = true;
        break;
      }
      cur = r;
    }
    if (chain.length < 2) continue;
    const last = chain.length - 1;
    for (let i = 0; i < chain.length; i++) {
      const c = chain[i];
      cells.push(c);
      // Wet ends (lake or sea) carry the discharge of the river node beside
      // them, and so does a confluence: up to the junction the tributary
      // carries only its own water (with the trunk's, every tributary's last
      // stretch would be drawn, however small the stream).
      const own = ocean[c] || lake[c] >= 0 || (joins && i === last);
      const beside = chain[i === 0 ? 1 : i - 1];
      qs.push(own ? discharge[beside] : discharge[c]);
      banks.push(own ? bankfull[beside] : bankfull[c]);
      flows.push(own ? flowMonths[beside] : flowMonths[c]);
    }
    sea.push(toSea);
    next.push(joins ? recv[chain[last]] : toSea ? seaward(chain[last]) : -1);
    start.push(cells.length);
  }
  const nNodes = cells.length;
  const nChains = start.length - 1;
  const run = new Uint8Array(nChains);
  const runOf = new Map<number, number>();
  for (let ch = 0; ch < nChains; ch++) {
    if (!sea[ch]) continue;
    const o = cells[start[ch + 1] - 1];
    const best = runOf.get(o);
    if (best === undefined || qs[start[ch + 1] - 1] > qs[start[best + 1] - 1]) runOf.set(o, ch);
  }
  for (const ch of runOf.values()) run[ch] = 1;
  const box = new Float32Array(nNodes * 4);
  const h = mesh.spacing;
  for (let ch = 0; ch < start.length - 1; ch++) {
    for (let i = start[ch]; i < start[ch + 1] - 1; i++) {
      const a = cells[i];
      const b = cells[i + 1];
      const d = Math.acos(Math.min(1, x[a] * x[b] + y[a] * y[b] + z[a] * z[b]));
      const end = i + 1 === start[ch + 1] - 1 && sea[ch];
      const m = 0.3 * d + (end ? 2 * h : 0);
      let lb = lon[b];
      if (lb - lon[a] > Math.PI) lb -= 2 * Math.PI;
      else if (lon[a] - lb > Math.PI) lb += 2 * Math.PI;
      const mLon = m / Math.max(0.05, Math.cos(Math.max(Math.abs(lat[a]), Math.abs(lat[b])) + m));
      box[4 * i] = Math.min(lat[a], lat[b]) - m;
      box[4 * i + 1] = Math.max(lat[a], lat[b]) + m;
      box[4 * i + 2] = Math.min(lon[a], lb) - mLon;
      box[4 * i + 3] = Math.max(lon[a], lb) + mLon;
    }
  }
  // Cells with the sea within two rings: a segment touching one can meet the
  // coastline (the renderer's kernel reaches 1.25 cells from its course).
  const near = Uint8Array.from(ocean);
  for (let pass = 0; pass < 2; pass++) {
    const prev = Uint8Array.from(near);
    for (let c = 0; c < n; c++) {
      if (prev[c]) continue;
      for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
        if (prev[nbr[k]]) {
          near[c] = 1;
          break;
        }
      }
    }
  }
  const coast = new Uint8Array(nNodes);
  const segOf = new Int32Array(n).fill(-1);
  for (let ch = 0; ch < start.length - 1; ch++) {
    for (let i = start[ch]; i < start[ch + 1] - 1; i++) {
      coast[i] = near[cells[i]] | near[cells[i + 1]];
      segOf[cells[i]] = i;
    }
  }
  // Every segment leaves its own cell and the cells drain in routing order
  // (receivers first), so walking that order backwards visits upstream first.
  const order: number[] = [];
  for (let k = routing.count - 1; k >= 0; k--) {
    const i = segOf[routing.order[k]];
    if (i >= 0) order.push(i);
  }
  const join = new Int32Array(nChains).fill(-1);
  for (let ch = 0; ch < nChains; ch++) {
    if (sea[ch] || next[ch] < 0) continue;
    join[ch] = segOf[cells[start[ch + 1] - 1]];
  }
  // Meander waves are about MEANDER_WAVE channel widths long; each segment gets
  // the nearest whole number of half-waves.
  const wave = new Int32Array(nNodes);
  const Rm = world.params.radiusKm * 1000;
  for (let ch = 0; ch < nChains; ch++) {
    let w = cells[start[ch]] & 1;
    for (let i = start[ch]; i < start[ch + 1]; i++) {
      wave[i] = w;
      if (i + 1 === start[ch + 1]) break;
      const a = cells[i];
      const b = cells[i + 1];
      const len = Math.acos(Math.min(1, x[a] * x[b] + y[a] * y[b] + z[a] * z[b])) * Rm;
      const lambda = MEANDER_WAVE * CHANNEL_WIDTH * Math.sqrt(Math.max(1, 0.5 * (banks[i] + banks[i + 1])));
      w += Math.max(1, Math.round((2 * len) / lambda));
    }
  }
  return {
    rivStart: Int32Array.from(start),
    rivCell: Int32Array.from(cells),
    rivQ: Float32Array.from(qs),
    rivBank: Float32Array.from(banks),
    rivWave: wave,
    rivFlow: Uint8Array.from(flows),
    rivSea: Uint8Array.from(sea),
    rivNext: Int32Array.from(next),
    rivRun: run,
    rivJoin: join,
    rivOrder: Int32Array.from(order),
    rivCoast: coast,
    rivBox: box,
  };
}

/**
 * Per-version fields, taken from whatever the world currently holds (during a
 * brush stroke that is the previewed relief with the last climate and rivers).
 */
export function buildFields(world: World, meshId: string, version: number): FieldData {
  const mesh = world.fine;
  const n = mesh.n;
  const { ocean, geo } = world;
  const elev = atollsSubmerged(mesh, world.elev, world.relief.reef);
  const grad = limitedGradient(mesh, elev, ocean);
  const amp = new Float32Array(n);
  const mtn = new Float32Array(n);
  const rough = world.relief.rough;
  const h = mesh.spacing;
  for (let c = 0; c < n; c++) {
    const g = Math.hypot(grad[3 * c], grad[3 * c + 1], grad[3 * c + 2]) * h;
    if (ocean[c]) {
      amp[c] = 0.8 * rough[c] + 0.05 * g;
      continue;
    }
    const youth = 0.4 + 0.6 * Math.exp(-geo.orogeny[c] / 150);
    mtn[c] = smoothstep(600, 2600, elev[c]) * youth;
    // Young ranges hide kilometres of ridge-and-valley relief inside one cell;
    // even plains keep some low hills and swales.
    amp[c] = Math.hypot(rough[c] + 450 * mtn[c] + 15, 0.2 * g);
  }
  const plateColor = new Uint8Array(geo.plates.length * 3);
  geo.plates.forEach((p, i) => plateColor.set(p.color, 3 * i));
  const hydro = world.hydro;
  const dryness = new Float32Array(n);
  for (let c = 0; c < n; c++) {
    const pet = world.water.petYear[c];
    if (!ocean[c] && pet > 0) dryness[c] = Math.max(0, Math.min(1, 1 - world.precipYear[c] / pet));
  }
  const ice = world.ice;
  const iceMelt = new Float32Array(n);
  const scour = new Float32Array(n);
  const fjord = new Float32Array(n);
  const { nbrStart, nbr } = mesh;
  for (let c = 0; c < n; c++) {
    if (ice.thick[c] > 0) iceMelt[c] = smoothstep(0, 1.5, -ice.smb[c]);
    // Lake-dotted scoured lowlands: where ice-age ice sat, outside today's ice and the high mountains.
    if (!ocean[c]) scour[c] = world.iceAge.scour[c] * (1 - mtn[c]) * (1 - ice.cover[c]);
    // Fjords: coasts whose ice-age glaciers cut troughs below sea level.
    let coastal = false;
    let cut = world.iceAge.erosion[c];
    for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
      if (ocean[nbr[k]] !== ocean[c]) coastal = true;
      cut = Math.max(cut, world.iceAge.erosion[nbr[k]]);
    }
    if (coastal) fjord[c] = Math.min(1, cut / (0.4 * MAX_GLACIAL_EROSION));
  }
  return {
    meshId,
    version,
    noiseSeed: seedFor(world.params.seed, 'detail'),
    radiusKm: world.params.radiusKm,
    riverThreshold: world.params.riverThreshold,
    elev,
    grad,
    amp,
    mtn,
    ocean,
    lake: hydro.lake,
    lakeLevel: Float32Array.from(hydro.lakes, (l) => l.level),
    lakeKind: Uint8Array.from(hydro.lakes, (l) => l.kind),
    dryness,
    ice: ice.thick,
    iceCover: ice.cover,
    iceMelt,
    shelf: ice.shelf,
    scour,
    fjord,
    reef: world.relief.reef,
    color: world.biomes.color,
    biome: world.biomes.biome,
    koppen: world.biomes.koppen,
    rock: world.lithology.rock,
    deposit: world.deposits.kind,
    plate: geo.plate,
    plateColor,
    ctype: geo.ctype,
    thick: geo.thick,
    age: geo.age,
    bndType: world.bnd.type,
    drainage: hydro.drainage,
    temp: world.temp,
    precip: world.precip,
    ...buildRivers(world),
  };
}

type Typed = Float64Array | Float32Array | Int32Array | Int16Array | Uint8Array;

/**
 * Copy every typed array into SharedArrayBuffer storage when the page is
 * cross-origin isolated, so tile workers share one copy instead of each
 * receiving a clone. Otherwise the object is returned as is.
 */
export function shareable<T extends object>(obj: T): T {
  if (typeof SharedArrayBuffer === 'undefined' || !(globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated) return obj;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) {
    if (ArrayBuffer.isView(v) && !(v.buffer instanceof SharedArrayBuffer)) {
      const src = v as Typed;
      const ctor = src.constructor as { new (b: SharedArrayBuffer): Typed };
      const dst = new ctor(new SharedArrayBuffer(src.byteLength));
      dst.set(src as never);
      out[k] = dst;
    } else out[k] = v;
  }
  return out as T;
}
