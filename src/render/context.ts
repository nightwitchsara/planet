import { smoothstep } from '../core/math';
import type { SphereMesh } from '../core/mesh';
import { seedFor } from '../core/rng';
import { Channel, type DeltaForm } from '../world/channels';
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
  /** Soil order (soils.ts Soil), 0–1 farming fertility, and salt in the soil. */
  soil: Uint8Array;
  fertility: Float32Array;
  saline: Uint8Array;
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
  /**
   * Channel pattern at each chain node (world/channels.ts Channel): the
   * commonest over a few nodes, so a reach doesn't flicker between patterns.
   */
  rivPattern: Uint8Array;
  /** Channels across the active belt at each node (1 = a single thread), smoothed along the chain. */
  rivThreads: Float32Array;
  /** Sinuosity of the main channel at each node (1–3.5). */
  rivSinu: Float32Array;
  /** Width (m) of the active belt at each node: the braidplain or anastomosing net (for one thread, its channel). */
  rivBelt: Float32Array;
  /** Width (m) of the floodplain at each node (0 where valley walls hem the river in). */
  rivPlain: Float32Array;
  /** Bed grain size D50 (mm) at each node. */
  rivGrain: Float32Array;
  /** 0–1 at each node: bare sand and gravel bars between the channels (1: braided) or vegetated islands (0: anastomosing). */
  rivBars: Float32Array;
  /** 0–1 at each node: hemmed in by valley walls (a straight course, no meander belt or floodplain). */
  rivConf: Float32Array;
  /**
   * Distance (m) down the chain from its first node, and the same counted in
   * belt widths: the along-channel coordinates that braids, bars and islands
   * are laid out on (global, so tiles and zoom levels agree).
   */
  rivDist: Float32Array;
  rivTau: Float32Array;
  /**
   * 0–1 at each node: how braided (1; wandering 0.45) and how anastomosing
   * the river is, averaged over several nodes so its belt opens and closes
   * gradually where the pattern changes.
   */
  rivBraid: Float32Array;
  rivAnas: Float32Array;
  /**
   * Per chain node: the delta, fan or inland delta (index into the net* arrays)
   * whose distributaries are drawn instead of the segment leaving it, else -1.
   */
  rivNet: Int32Array;
  /**
   * Distributary networks (world.channels.networks): nodes of network i are
   * netStart[i]..netStart[i+1]-1, the first its apex (moved onto the river's
   * node there, and a fan's or inland delta's exit onto the river's node where
   * it runs on, so the drawn river is unbroken). Per node: position on the unit
   * sphere, parent (global node index, -1 at the apex), the child carrying most
   * water (-1 at an end), mean discharge (m³/s) and width (m) of the channel
   * ending there, 1 at a channel end, and distance down the channels from the
   * apex (m).
   */
  netStart: Int32Array;
  netX: Float32Array;
  netY: Float32Array;
  netZ: Float32Array;
  netParent: Int32Array;
  netNext: Int32Array;
  netQ: Float32Array;
  netW: Float32Array;
  netMouth: Uint8Array;
  netDist: Float32Array;
  /**
   * Per network: the river node before its apex and the one after its exit
   * (unit-sphere x, y, z; NaN if none), shaping the curves where the river
   * runs into and out of it, and its exit node (global index, -1 if none).
   */
  netUp: Float32Array;
  netDown: Float32Array;
  netExit: Int32Array;
  /**
   * Per network, its plain or cone as seen from the apex: a gnomonic frame
   * (apex, axis and normal unit vectors: 9 numbers), the bearings it spans
   * (2: from, to; radians from the axis), and the distance from the apex to
   * its shoreline or margin at NET_BINS evenly spaced bearings (radians).
   */
  netFrame: Float64Array;
  netTheta: Float32Array;
  netFront: Float32Array;
  /**
   * Per network (NET_INFO each): kind (0 delta, 1 fan, 2 inland), form
   * (NET_FORMS), wave, tide, sand share, radius (radians), months of flow,
   * dryness, and the apex's ground color (r, g, b) and mean temperature (°C)
   * for plain built out past the cells' land.
   */
  netInfo: Float32Array;
  /** Per network: the cell at its apex (its biome, soil and so on stand for plain built out over the sea). */
  netCell: Int32Array;
  /** Per cell: the network whose plain, cone or wetland covers it (world.channels.networkOf), else -1. */
  cellNet: Int32Array;
  /** Per network: lat0, lat1, lon0, lon1 (radians, lon may exceed ±π). */
  netBox: Float32Array;
}

/** Bearings at which a network's shoreline is sampled (netFront). */
export const NET_BINS = 96;
/** Numbers per network in netInfo. */
export const NET_INFO = 12;
/** Delta forms in netInfo order. */
export const NET_FORMS: readonly DeltaForm[] = ['birdfoot', 'lobate', 'arcuate', 'cuspate', 'estuarine', 'fan', 'inland'];

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
export const MEANDER_WAVE = 12;

type RiverFields =
  | 'rivStart' | 'rivCell' | 'rivQ' | 'rivBank' | 'rivWave' | 'rivFlow' | 'rivSea' | 'rivNext' | 'rivRun' | 'rivJoin' | 'rivOrder' | 'rivCoast' | 'rivBox'
  | 'rivPattern' | 'rivThreads' | 'rivSinu' | 'rivBelt' | 'rivPlain' | 'rivGrain' | 'rivBars' | 'rivConf' | 'rivDist' | 'rivTau' | 'rivBraid' | 'rivAnas';

/**
 * Share of the sub-cell relief that shows through each surface deposit
 * (sediment.ts Deposit order): the flat tops of delta plains, lake beds,
 * playas and bogs, smoothed floodplains, outwash and fans (a delta plain
 * standing 2 m above the sea has no 40 m hills).
 */
const BURIED_RELIEF = [1, 1, 1, 0.5, 0.45, 0.07, 0.12, 0.05, 1, 0.8, 0.85, 1, 0.45, 0.15, 1];

/** How bare the ground between a pattern's channels is (Channel order): bars of a braided river, vegetated islands of an anastomosing one. */
const PATTERN_BARS = [0.3, 0.3, 0.4, 0.6, 0.8, 1, 0.05, 0.4];

/**
 * Channel style at each chain node from world.channels (wet ends take the
 * land node beside them), smoothed over a few nodes so patterns change
 * gradually along a river instead of flickering from node to node.
 */
function channelStyle(
  world: World,
  start: number[],
  cells: number[],
  src: number[],
  banks: number[],
): Pick<FieldData, 'rivPattern' | 'rivThreads' | 'rivSinu' | 'rivBelt' | 'rivPlain' | 'rivGrain' | 'rivBars' | 'rivConf' | 'rivDist' | 'rivTau' | 'rivBraid' | 'rivAnas'> {
  const mesh = world.fine;
  const { x, y, z } = mesh;
  const nNodes = cells.length;
  const ch = world.channels;
  const have = ch !== undefined && ch.pattern.length === mesh.n;
  const rawP = new Uint8Array(nNodes);
  const raw = [0, 1, 2, 3, 4, 5, 6].map(() => new Float32Array(nNodes));
  const [rThr, rSin, rBelt, rPlain, rGrain, rBars, rConf] = raw;
  for (let k = 0; k < nNodes; k++) {
    const s = src[k];
    const w = CHANNEL_WIDTH * Math.sqrt(Math.max(0, banks[k]));
    const p = have ? ch.pattern[s] : Channel.None;
    rawP[k] = p === Channel.None ? Channel.Meandering : p;
    if (!have || p === Channel.None) {
      // Unclassified (small) streams: one gently winding thread.
      rThr[k] = 1;
      rSin[k] = 1.2;
      rBelt[k] = Math.log(1 + w);
      rPlain[k] = Math.log(1 + 6 * w);
      rGrain[k] = Math.log(1);
      rBars[k] = 0.3;
      rConf[k] = 0.3;
      continue;
    }
    const g = ch.grain[s] > 0 ? ch.grain[s] : 1;
    rThr[k] = Math.max(1, Math.min(12, ch.threads[s] || 1));
    rSin[k] = Math.max(1, Math.min(3.5, ch.sinuosity[s] || 1));
    rBelt[k] = Math.log(1 + Math.max(w, ch.belt[s] || 0));
    rPlain[k] = Math.log(1 + Math.max(0, ch.floodplain[s] || 0));
    rGrain[k] = Math.log(g);
    // Mud-bedded single channels build few bare bars.
    const single = p === Channel.Meandering || p === Channel.Straight || p === Channel.Confined;
    rBars[k] = PATTERN_BARS[p] * (single ? smoothstep(0.03, 0.3, g) : 1);
    rConf[k] = p === Channel.Confined ? 1 : 0;
  }
  const out = raw.map(() => new Float32Array(nNodes));
  const pattern = new Uint8Array(nNodes);
  const KW = [1, 2, 3, 2, 1];
  const votes = new Float32Array(8);
  for (let c = 0; c < start.length - 1; c++) {
    const s0 = start[c];
    const s1 = start[c + 1];
    for (let k = s0; k < s1; k++) {
      votes.fill(0);
      let sw = 0;
      for (let d = -2; d <= 2; d++) {
        const j = k + d;
        if (j < s0 || j >= s1) continue;
        const wgt = KW[d + 2];
        sw += wgt;
        votes[rawP[j]] += wgt;
        for (let f = 0; f < raw.length; f++) out[f][k] += wgt * raw[f][j];
      }
      for (let f = 0; f < raw.length; f++) out[f][k] /= sw;
      // The commonest pattern nearby; ties keep the node's own.
      let best = rawP[k];
      for (let p = 0; p < 8; p++) if (votes[p] > votes[best]) best = p;
      pattern[k] = best;
    }
  }
  const [thr, sin, belt, plain, grain, bars, conf] = out;
  // Braided and anastomosing weights over nine nodes (a belt that closed at
  // every node of another pattern would pinch to a point there).
  const braidW = new Float32Array(nNodes);
  const anasW = new Float32Array(nNodes);
  for (let c = 0; c < start.length - 1; c++) {
    const s0 = start[c];
    const s1 = start[c + 1];
    for (let k = s0; k < s1; k++) {
      let sw = 0;
      let sb = 0;
      let sa = 0;
      for (let d = -4; d <= 4; d++) {
        const j = k + d;
        if (j < s0 || j >= s1) continue;
        const wgt = 5 - Math.abs(d);
        sw += wgt;
        const p = rawP[j];
        sb += wgt * (p === Channel.Braided ? 1 : p === Channel.Wandering ? 0.45 : 0);
        sa += wgt * (p === Channel.Anastomosing ? 1 : 0);
      }
      braidW[k] = sb / sw;
      anasW[k] = sa / sw;
    }
  }
  // A belt is drawn no wider than the bends of a course through cells this far
  // apart allow, nor a floodplain much wider than a cell.
  const spacingM = mesh.spacing * world.params.radiusKm * 1000;
  for (let k = 0; k < nNodes; k++) {
    belt[k] = Math.min(0.6 * spacingM, Math.expm1(belt[k]));
    plain[k] = Math.min(1.5 * spacingM, Math.expm1(plain[k]));
    grain[k] = Math.exp(grain[k]);
  }
  // Along-channel coordinates: metres, and belt widths (the braids' wavelength scales with the belt).
  const dist = new Float32Array(nNodes);
  const tau = new Float32Array(nNodes);
  const Rm = world.params.radiusKm * 1000;
  for (let c = 0; c < start.length - 1; c++) {
    let dm = 0;
    let tb = 0;
    for (let k = start[c]; k < start[c + 1]; k++) {
      if (k > start[c]) {
        const a = cells[k - 1];
        const b = cells[k];
        const len = Math.acos(Math.min(1, x[a] * x[b] + y[a] * y[b] + z[a] * z[b])) * Rm;
        dm += len;
        tb += len / Math.max(1, 0.5 * (belt[k - 1] + belt[k]));
      }
      dist[k] = dm;
      tau[k] = tb;
    }
  }
  return { rivPattern: pattern, rivThreads: thr, rivSinu: sin, rivBelt: belt, rivPlain: plain, rivGrain: grain, rivBars: bars, rivConf: conf, rivDist: dist, rivTau: tau, rivBraid: braidW, rivAnas: anasW };
}

function buildRivers(world: World): Pick<FieldData, RiverFields> {
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
  /** The land cell whose channel each node takes (a wet end's or a junction's neighbor up the chain). */
  const srcs: number[] = [];
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
      srcs.push(own ? beside : c);
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
    ...channelStyle(world, start, cells, srcs, banks),
  };
}

type NetFields =
  | 'rivNet' | 'netStart' | 'netX' | 'netY' | 'netZ' | 'netParent' | 'netNext' | 'netQ' | 'netW' | 'netMouth' | 'netDist'
  | 'netFrame' | 'netTheta' | 'netFront' | 'netInfo' | 'netBox' | 'netUp' | 'netDown' | 'netExit' | 'netCell' | 'cellNet';

/**
 * The distributary networks for drawing, tied into the river chains: each
 * network takes over the chain from its apex (to the sea, or to where a fan's
 * or inland delta's trunk leaves it and runs on), and its apex and exit are
 * moved onto the chain's nodes there so the drawn river runs on unbroken.
 */
function buildNetworks(world: World, riv: Pick<FieldData, RiverFields>): Pick<FieldData, NetFields> {
  const mesh = world.fine;
  const { x, y, z } = mesh;
  const nets = world.channels?.networks ?? [];
  const { rivStart, rivCell, rivQ, rivFlow } = riv;
  const nNodes = rivCell.length;
  const Rm = world.params.radiusKm * 1000;
  // The chain node starting the segment out of each cell (a cell starts at most one).
  const nodeOf = new Int32Array(mesh.n).fill(-1);
  const chainEnd = new Int32Array(nNodes);
  for (let c = 0; c < rivStart.length - 1; c++) {
    for (let k = rivStart[c]; k < rivStart[c + 1]; k++) {
      chainEnd[k] = rivStart[c + 1] - 1;
      if (k < rivStart[c + 1] - 1) nodeOf[rivCell[k]] = k;
    }
  }
  const rivNet = new Int32Array(nNodes).fill(-1);
  let total = 0;
  for (const n of nets) total += n.parent.length;
  const netStart = new Int32Array(nets.length + 1);
  const nx = new Float32Array(total);
  const ny = new Float32Array(total);
  const nz = new Float32Array(total);
  const parent = new Int32Array(total);
  const next = new Int32Array(total).fill(-1);
  const q = new Float32Array(total);
  const w = new Float32Array(total);
  const mouth = new Uint8Array(total);
  const dist = new Float32Array(total);
  const frame = new Float64Array(9 * nets.length);
  const theta = new Float32Array(2 * nets.length);
  const front = new Float32Array(NET_BINS * nets.length);
  const info = new Float32Array(NET_INFO * nets.length);
  const box = new Float32Array(4 * nets.length);
  const up = new Float32Array(3 * nets.length).fill(NaN);
  const down = new Float32Array(3 * nets.length).fill(NaN);
  const exitG = new Int32Array(nets.length).fill(-1);
  const cellOf = new Int32Array(nets.length);
  const color = world.biomes.color;
  const deltas = world.deposits.deltas;
  const pet = world.water.petYear;
  let base = 0;
  nets.forEach((net, i) => {
    netStart[i] = base;
    const m = net.parent.length;
    const ax = net.x[0];
    const ay = net.y[0];
    const az = net.z[0];
    // Where the network meets its river's chain.
    let k0 = -1;
    let kEnd = -1;
    let exitTo = -1;
    if (net.kind === 'delta') {
      const L = net.delta >= 0 && net.delta < deltas.length ? deltas[net.delta].river : -1;
      const k = L >= 0 ? nodeOf[L] : -1;
      if (k >= 0) {
        // Up the chain to its node nearest the apex.
        let best = k;
        let bd = Infinity;
        for (let j = k; j >= 0 && chainEnd[j] === chainEnd[k]; j--) {
          const c = rivCell[j];
          const d = 1 - (x[c] * ax + y[c] * ay + z[c] * az);
          if (d < bd) {
            bd = d;
            best = j;
          } else if (d > 4 * bd + 1e-9 && j < best - 2) break;
        }
        k0 = best;
        kEnd = chainEnd[k] - 1;
      }
    } else {
      const k = nodeOf[net.apex];
      if (k >= 0) {
        const own = new Set<number>(net.cells);
        own.add(net.apex);
        let j = k;
        while (j + 1 < chainEnd[k] && own.has(rivCell[j + 1])) j++;
        k0 = k;
        kEnd = j;
        if (net.exit >= 0 && j + 1 <= chainEnd[k]) exitTo = rivCell[j + 1];
      }
    }
    if (k0 >= 0) {
      for (let j = k0; j <= kEnd; j++) rivNet[j] = i;
      if (k0 > 0 && chainEnd[k0 - 1] === chainEnd[k0]) up.set([x[rivCell[k0 - 1]], y[rivCell[k0 - 1]], z[rivCell[k0 - 1]]], 3 * i);
      if (exitTo >= 0 && kEnd + 2 <= chainEnd[k0]) down.set([x[rivCell[kEnd + 2]], y[rivCell[kEnd + 2]], z[rivCell[kEnd + 2]]], 3 * i);
    }
    if (net.exit >= 0 && exitTo >= 0) exitG[i] = base + net.exit;
    // The river's mean flow over the network's bankfull one.
    const qRatio = k0 >= 0 ? rivQ[k0] / Math.max(1e-6, net.q[0]) : 0.3;
    for (let j = 0; j < m; j++) {
      const g = base + j;
      let px = net.x[j];
      let py = net.y[j];
      let pz = net.z[j];
      const snap = j === 0 && k0 >= 0 ? rivCell[k0] : j === net.exit && exitTo >= 0 ? exitTo : -1;
      if (snap >= 0) {
        px = x[snap];
        py = y[snap];
        pz = z[snap];
      }
      nx[g] = px;
      ny[g] = py;
      nz[g] = pz;
      const p = net.parent[j];
      parent[g] = p >= 0 ? base + p : -1;
      q[g] = net.q[j] * qRatio;
      w[g] = net.width[j];
      mouth[g] = net.mouth[j];
      if (p >= 0) {
        const pg = base + p;
        dist[g] = dist[pg] + Math.acos(Math.min(1, nx[g] * nx[pg] + ny[g] * ny[pg] + nz[g] * nz[pg])) * Rm;
        if (next[pg] < 0 || net.q[j] > net.q[next[pg] - base]) next[pg] = g;
      }
    }
    // Gnomonic frame at the apex, its axis toward the outline's middle (bearing 0).
    const ol = net.outline;
    const K = ol.length / 3;
    const mid = 3 * Math.floor(K / 2);
    const dm = ol[mid] * ax + ol[mid + 1] * ay + ol[mid + 2] * az;
    let ex = ol[mid] - dm * ax;
    let ey = ol[mid + 1] - dm * ay;
    let ez = ol[mid + 2] - dm * az;
    const el = Math.hypot(ex, ey, ez) || 1;
    ex /= el;
    ey /= el;
    ez /= el;
    const fx = ay * ez - az * ey;
    const fy = az * ex - ax * ez;
    const fz = ax * ey - ay * ex;
    frame.set([ax, ay, az, ex, ey, ez, fx, fy, fz], 9 * i);
    const th: number[] = [];
    const rr: number[] = [];
    for (let k = 0; k < K; k++) {
      const d = ol[3 * k] * ax + ol[3 * k + 1] * ay + ol[3 * k + 2] * az;
      const u = (ol[3 * k] * ex + ol[3 * k + 1] * ey + ol[3 * k + 2] * ez) / d;
      const v = (ol[3 * k] * fx + ol[3 * k + 1] * fy + ol[3 * k + 2] * fz) / d;
      th.push(Math.atan2(v, u));
      rr.push(Math.hypot(u, v));
    }
    if (th[0] > th[K - 1]) {
      th.reverse();
      rr.reverse();
    }
    const t0 = th[0];
    const t1 = Math.max(th[K - 1], t0 + 1e-6);
    theta[2 * i] = t0;
    theta[2 * i + 1] = t1;
    let s = 0;
    for (let b = 0; b < NET_BINS; b++) {
      const tb = t0 + ((t1 - t0) * b) / (NET_BINS - 1);
      while (s < K - 2 && th[s + 1] < tb) s++;
      const f = th[s + 1] > th[s] ? Math.max(0, Math.min(1, (tb - th[s]) / (th[s + 1] - th[s]))) : 1;
      front[NET_BINS * i + b] = rr[s] + (rr[s + 1] - rr[s]) * f;
    }
    // A bird's foot's fingers are its levees, drawn along its passes: its
    // front here is the plain between them (the outline's narrow spikes trimmed).
    if (net.form === 'birdfoot') {
      const src = front.slice(NET_BINS * i, NET_BINS * (i + 1));
      for (let b = 0; b < NET_BINS; b++) {
        let mn = Infinity;
        for (let d = -3; d <= 3; d++) mn = Math.min(mn, src[Math.max(0, Math.min(NET_BINS - 1, b + d))]);
        front[NET_BINS * i + b] = mn;
      }
    }
    const ac = k0 >= 0 ? rivCell[k0] : net.apex;
    cellOf[i] = ac;
    const dryness = pet[ac] > 0 ? Math.max(0, Math.min(1, 1 - world.precipYear[ac] / pet[ac])) : 0;
    let tMean = 0;
    for (let mo = 0; mo < 12; mo++) tMean += world.temp[ac * 12 + mo] / 12;
    info.set(
      [
        net.kind === 'delta' ? 0 : net.kind === 'fan' ? 1 : 2,
        Math.max(0, NET_FORMS.indexOf(net.form)),
        net.wave,
        net.tide,
        net.sand,
        (net.radiusKm * 1000) / Rm,
        k0 >= 0 ? rivFlow[k0] : 12,
        dryness,
        color[3 * ac],
        color[3 * ac + 1],
        color[3 * ac + 2],
        tMean,
      ],
      NET_INFO * i,
    );
    // Bounding box of its nodes and outline.
    const lonA = Math.atan2(ay, ax);
    let la0 = Infinity;
    let la1 = -Infinity;
    let lo0 = Infinity;
    let lo1 = -Infinity;
    const take = (px: number, py: number, pz: number): void => {
      const la = Math.asin(Math.max(-1, Math.min(1, pz / Math.hypot(px, py, pz))));
      let lo = Math.atan2(py, px);
      lo += 2 * Math.PI * Math.round((lonA - lo) / (2 * Math.PI));
      la0 = Math.min(la0, la);
      la1 = Math.max(la1, la);
      lo0 = Math.min(lo0, lo);
      lo1 = Math.max(lo1, lo);
    };
    for (let j = 0; j < m; j++) take(nx[base + j], ny[base + j], nz[base + j]);
    for (let k = 0; k < K; k++) take(ol[3 * k], ol[3 * k + 1], ol[3 * k + 2]);
    // (Its cells too, and their kernels' reach: the plain is blended from them.)
    for (const c of net.cells) take(x[c], y[c], z[c]);
    const pad = 0.02 * (la1 - la0 + lo1 - lo0) + mesh.spacing * (KERNEL_RADIUS + 0.25);
    const padLon = pad / Math.max(0.05, Math.cos(Math.max(Math.abs(la0), Math.abs(la1)) + pad));
    box.set([la0 - pad, la1 + pad, lo0 - padLon, lo1 + padLon], 4 * i);
    base += m;
  });
  netStart[nets.length] = base;
  return { rivNet, netStart, netX: nx, netY: ny, netZ: nz, netParent: parent, netNext: next, netQ: q, netW: w, netMouth: mouth, netDist: dist, netFrame: frame, netTheta: theta, netFront: front, netInfo: info, netBox: box, netUp: up, netDown: down, netExit: exitG, netCell: cellOf, cellNet: world.channels?.networkOf ?? new Int32Array(mesh.n).fill(-1) };
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
  const depKind = world.deposits.kind;
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
    // even plains keep some low hills and swales, except where sediment has
    // buried them: delta plains, lake beds and playas are nearly dead level.
    amp[c] = Math.hypot((rough[c] + 15) * (BURIED_RELIEF[depKind[c]] ?? 1) + 450 * mtn[c], 0.2 * g);
  }
  const rivers = buildRivers(world);
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
    soil: world.soils.order,
    fertility: world.soils.fertility,
    saline: world.soils.saline,
    plate: geo.plate,
    plateColor,
    ctype: geo.ctype,
    thick: geo.thick,
    age: geo.age,
    bndType: world.bnd.type,
    drainage: hydro.drainage,
    temp: world.temp,
    precip: world.precip,
    ...rivers,
    ...buildNetworks(world, rivers),
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
