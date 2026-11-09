/// <reference lib="webworker" />
import { compassName, DEG, latLonToVec, type Vec3 } from '../core/math';
import { nearestCellLatLon } from '../core/mesh';
import { deflateBrowser, encodePNG } from '../io/png';
import { deserializeWorld, serializeWorld } from '../io/save';
import { buildFields, type FieldData, type MeshData, meshData, shareable } from '../render/context';
import { BASE_LEVEL, levelForWidth, levelHeight, levelWidth, type PixelWindow, type RenderOptions, renderWindow } from '../render/raster';
import { BIOME_INFO, KOPPEN_CODES, KOPPEN_NAMES } from '../world/biomes';
import { MONTHS } from '../world/climate';
import * as edits from '../world/edits';
import { ROCK_INFO } from '../world/geology';
import { LakeKind } from '../world/hydrology';
import { Deposit, DEPOSIT_NAMES } from '../world/sediment';
import {
  B_CONVERGENT, B_DIVERGENT, B_TRANSFORM, ROLE_COLLISION, ROLE_OVERRIDING, ROLE_SUBDUCTING,
} from '../world/tectonics/boundaries';
import type { WorldParams } from '../world/types';
import { type DeriveFrom, World } from '../world/world';
import type { Arrow, BrushTool, ContextMessage, EditOp, InspectInfo, Request, Response, StripResult, WorldSummary } from './protocol';

const ctx = self as unknown as DedicatedWorkerGlobalScope;
let world: World | null = null;
let strokeReport: edits.EditReport | null = null;
let strokeTool: BrushTool | null = null;
/** Eroded elevation and raw relief from before the current brush stroke. */
let strokeBase: { elev: Float32Array; relief: Float32Array } | null = null;

/** Channels to the tile workers, and what they were last sent. */
let ports: MessagePort[] = [];
let worldCount = 0;
let mesh: MeshData | null = null;
let fields: FieldData | null = null;
let version = 0;

function post(msg: Response, transfer: Transferable[] = []): void {
  ctx.postMessage(msg, transfer);
}

function progress(stage: string, frac: number): void {
  post({ type: 'progress', stage, frac });
}

function requireWorld(): World {
  if (!world) throw new Error('No world generated yet');
  return world;
}

function broadcast(msg: ContextMessage): void {
  for (const p of ports) p.postMessage(msg);
}

/** A new world (new mesh): tile workers get the mesh once. */
function adopt(w: World): void {
  world = w;
  worldCount++;
  mesh = meshData(w.fine, `world-${worldCount}`);
  broadcast({ type: 'mesh', mesh: shareable(mesh) });
}

/** The world changed: rebuild the render fields under a new version. */
function publish(w: World): void {
  version++;
  fields = buildFields(w, mesh!.id, version);
  broadcast({ type: 'fields', fields: shareable(fields) });
}

let stripId = 0;
const stripWaiters = new Map<number, (data: ArrayBuffer) => void>();

/**
 * Render windows of the current world on the tile workers in parallel (two in
 * flight per worker), handing each result to `done` as it arrives.
 */
function renderParallel(jobs: PixelWindow[], render: RenderOptions, elevation: boolean, done: (job: PixelWindow, data: ArrayBuffer) => void): Promise<void> {
  if (ports.length === 0) {
    for (const job of jobs) {
      const r = renderWindow({ mesh: mesh!, fields: fields! }, render, job, elevation);
      done(job, (elevation ? r.elevation!.buffer : r.rgba.buffer) as ArrayBuffer);
    }
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    let next = 0;
    let finished = 0;
    const send = (port: MessagePort) => {
      if (next >= jobs.length) return;
      const job = jobs[next++];
      const id = ++stripId;
      stripWaiters.set(id, (data) => {
        stripWaiters.delete(id);
        done(job, data);
        if (++finished === jobs.length) resolve();
        else send(port);
      });
      port.postMessage({ type: 'strip', id, render, window: job, elevation } satisfies ContextMessage);
    };
    for (const p of ports) {
      send(p);
      send(p);
    }
  });
}

function plateVelocityAt(w: World, plate: number, v: Vec3): { vE: number; vN: number } {
  const p = w.geo.plates[plate];
  const R = w.params.radiusKm;
  const vx = (p.wy * v[2] - p.wz * v[1]) * R;
  const vy = (p.wz * v[0] - p.wx * v[2]) * R;
  const vz = (p.wx * v[1] - p.wy * v[0]) * R;
  const r = Math.hypot(v[0], v[1]) || 1e-9;
  const e: Vec3 = [-v[1] / r, v[0] / r, 0];
  const n: Vec3 = [v[1] * e[2] - v[2] * e[1], v[2] * e[0] - v[0] * e[2], v[0] * e[1] - v[1] * e[0]];
  // km/Myr → cm/yr
  return { vE: (vx * e[0] + vy * e[1] + vz * e[2]) / 10, vN: (vx * n[0] + vy * n[1] + vz * n[2]) / 10 };
}

function summary(w: World): WorldSummary {
  const arrows: Arrow[] = [];
  for (let lat = -72; lat <= 72; lat += 12) {
    const step = 12 / Math.max(0.25, Math.cos(lat * DEG));
    for (let lon = -180 + step / 2; lon < 180; lon += step) {
      const c = nearestCellLatLon(w.fine, lat * DEG, lon * DEG);
      const plate = w.geo.plate[c];
      const v = plateVelocityAt(w, plate, latLonToVec(lat * DEG, lon * DEG));
      arrows.push({ lat, lon, plate, ...v });
    }
  }
  const undo = w.undoStack[w.undoStack.length - 1];
  const redo = w.redoStack[w.redoStack.length - 1];
  return {
    params: w.params,
    cells: w.fine.n,
    plates: w.plateStats(),
    hotspots: w.geo.hotspots.map((h) => ({
      name: h.name,
      lat: Math.asin(h.z) / DEG,
      lon: Math.atan2(h.y, h.x) / DEG,
      active: h.active,
    })),
    arrows,
    landFraction: w.landFraction(),
    timeMyr: w.geo.time,
    historyLog: w.historyLog,
    canUndo: !!undo,
    canRedo: !!redo,
    undoLabel: undo?.label,
    redoLabel: redo?.label,
    timings: w.timings,
  };
}

function describeSetting(w: World, c: number): { setting: string; boundary: string | null } {
  const g = w.geo;
  const b = w.bnd;
  const name = (p: number) => g.plates[p]?.name ?? '?';
  let boundary: string | null = null;
  if (b.other[c] >= 0) {
    const kind = b.type[c] === B_CONVERGENT ? 'convergent' : b.type[c] === B_DIVERGENT ? 'divergent' : b.type[c] === B_TRANSFORM ? 'transform' : 'inactive';
    boundary = `${kind} boundary with ${name(b.other[c])} (${(Math.abs(b.rate[c]) / 10).toFixed(1)} cm/yr ${b.rate[c] >= 0 ? 'closing' : 'opening'})`;
    if (b.type[c] === B_CONVERGENT) {
      if (b.role[c] === ROLE_SUBDUCTING) boundary += ' — this side subducts';
      else if (b.role[c] === ROLE_OVERRIDING) boundary += ' — this side overrides';
      else if (b.role[c] === ROLE_COLLISION) boundary += ' — continental collision';
    }
  }
  const cont = g.ctype[c] === 1;
  const e = w.elev[c];
  let setting: string;
  if (w.relief.reef[c] === 1) setting = `Atoll: coral reef on a volcano that has subsided for ~${g.volcAge[c].toFixed(0)} Myr`;
  else if (g.volc[c] > 0.4 && g.volcAge[c] < 3) setting = 'Active hotspot volcano above a mantle plume';
  else if (b.dTrench[c] < 120) setting = `Ocean trench — ${name(g.plate[c])} plate sinking into the mantle`;
  else if (b.dArc[c] < 320) setting = b.arcCont[c] ? 'Continental volcanic arc above a subduction zone (Andean-type)' : 'Volcanic island arc above a subduction zone';
  else if (b.dColl[c] < 450) setting = 'Active collision belt — crust thickened between converging continents';
  else if (b.dDiv[c] < 200 && cont) setting = 'Continental rift valley — the plate is pulling apart';
  else if (g.volc[c] > 0.3) setting = `Extinct volcano / seamount (~${g.volcAge[c].toFixed(0)} Myr old)`;
  else if (!cont && g.age[c] < 8) setting = 'Mid-ocean ridge — new seafloor forming';
  else if (!cont && g.thick[c] > 12) setting = 'Oceanic plateau (thick volcanic crust)';
  else if (!cont) setting = `Abyssal ocean floor, ${g.age[c].toFixed(0)} Myr old`;
  else if (g.thick[c] > 42 && g.orogeny[c] < 40) setting = 'Young mountain belt';
  else if (g.thick[c] > 38) setting = `Old mountain belt (last uplift ~${g.orogeny[c].toFixed(0)} Myr ago, now eroded)`;
  else if (e >= 0 && w.sediment[c] > 1 && nextToSea(w, c) && e < 15) setting = 'River delta — land built out into the sea by river sediment';
  else if (e >= 0 && w.sediment[c] > 20) setting = `Alluvial plain — ~${Math.round(w.sediment[c])} m of river sediment over the bedrock`;
  else if (e < 0 && w.sediment[c] > 20) setting = 'Sediment-filled shelf off a river mouth';
  else if (e < 0) setting = 'Submerged continental crust (shelf or sunken continent)';
  else if (g.orogeny[c] > 1000) setting = 'Stable continental interior (craton)';
  else setting = 'Continental platform / lowland';
  return { setting, boundary };
}

function describeBedrock(w: World, c: number): string | null {
  if (!w.lithology) return null;
  const { long } = ROCK_INFO[w.lithology.rock[c]];
  const a = w.lithology.rockAge[c];
  const age = a >= 1000 ? `~${(a / 1000).toFixed(1)} Gyr` : a >= 1 ? `~${Math.round(a)} Myr` : 'under 1 Myr';
  return `${long[0].toUpperCase()}${long.slice(1)}, ${age}`;
}

const DELTA_FORMS = {
  river: 'river-dominated: distributaries reach out into quiet water',
  wave: 'wave-dominated: a smooth arc of beach ridges',
  tide: 'tide-dominated: funnel-shaped channels between tidal flats',
};

function describeSurface(w: World, c: number): string | null {
  const d = w.deposits;
  if (!d) return null;
  const k = d.kind[c];
  const m = (v: number) => `~${v >= 10 ? Math.round(v).toLocaleString('en-US') : v.toFixed(1)} m`;
  if (w.ocean[c]) return k === Deposit.Shelf ? `Shelf sediment, ${m(d.thickness[c])}` : null;
  if (k === Deposit.Bedrock) return d.regolith[c] < 0.3 ? 'Bare rock and rubble' : `Bedrock under ${m(d.regolith[c])} of soil`;
  if (k === Deposit.Residual) return `Rock rotted in place: ${m(d.regolith[c])} of saprolite under a leached, iron-rich soil`;
  let s = `${DEPOSIT_NAMES[k]}, ${m(d.thickness[c])}`;
  if (k === Deposit.Delta) {
    const { x, y, z } = w.fine;
    let best = null;
    let bd = -2;
    for (const r of d.deltas) {
      const dot = x[c] * x[r.mouth] + y[c] * y[r.mouth] + z[c] * z[r.mouth];
      if (dot > bd) {
        bd = dot;
        best = r;
      }
    }
    if (best) s += `, ${DELTA_FORMS[best.kind]}`;
  }
  const sand = d.sand[c];
  if (k !== Deposit.Dunes && k !== Deposit.Peat && k !== Deposit.Playa) s += sand > 0.7 ? ', sand and gravel' : sand > 0.4 ? ', loamy' : ', silt and clay';
  // Wind, ice and bog deposits lie on whatever came before them.
  const below = k >= Deposit.Playa && k <= Deposit.Peat ? d.cover[c] - d.thickness[c] : 0;
  if (below >= 2) s += `; ${m(below)} of older sediment beneath`;
  return s;
}

function nextToSea(w: World, c: number): boolean {
  const { nbrStart, nbr } = w.fine;
  for (let k = nbrStart[c]; k < nbrStart[c + 1]; k++) if (w.ocean[nbr[k]]) return true;
  return false;
}

const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** Plain-language flow regime of the river through cell c. */
function regimeOf(w: World, c: number): string | null {
  const h = w.hydro;
  const mean = h.discharge[c];
  if (w.ocean[c] || h.lake[c] >= 0 || mean < 1) return null;
  const q = Array.from(h.dischargeMonthly.subarray(c * MONTHS, c * MONTHS + MONTHS));
  let hi = 0;
  let lo = 0;
  for (let m = 1; m < MONTHS; m++) {
    if (q[m] > q[hi]) hi = m;
    if (q[m] < q[lo]) lo = m;
  }
  const melt = h.meltShare[c];
  const source = melt > 0.5 ? 'fed mostly by snow and glacier melt' : melt > 0.2 ? 'fed by rain and snowmelt' : 'rain-fed';
  const months = h.flowMonths[c];
  if (months <= 3) return `Ephemeral: flows only ~${months} month${months === 1 ? '' : 's'} a year, around ${MONTH_NAMES[hi]}`;
  if (months <= 10) {
    // The dry season: the longest run of months without flow (wrapping around the year).
    const dry = q.map((v) => v < Math.max(0.05, 0.03 * mean));
    let best = 0;
    let bestStart = 0;
    for (let s = 0; s < MONTHS; s++) {
      let len = 0;
      while (len < MONTHS && dry[(s + len) % MONTHS]) len++;
      if (len > best) {
        best = len;
        bestStart = s;
      }
    }
    const span = best > 1 ? `${MONTH_NAMES[bestStart]}–${MONTH_NAMES[(bestStart + best - 1) % MONTHS]}` : MONTH_NAMES[bestStart];
    return `Seasonal: dry ${span}, peaks in ${MONTH_NAMES[hi]}; ${source}`;
  }
  const swing = q[lo] > 0 ? q[hi] / q[lo] : Infinity;
  return `Perennial, ${source}; ${swing < 2.5 ? 'steady through the year' : `high water in ${MONTH_NAMES[hi]}, lowest in ${MONTH_NAMES[lo]}`}`;
}

const WETLAND_NAMES = [
  '',
  'River floodplain, flooded in high water',
  'Inland delta: a river spreading out and evaporating',
  'Peat bog on waterlogged cold ground',
  'Swamp fed by groundwater',
  'Marsh in a closed hollow that never fills to a lake',
];

function inspect(w: World, lat: number, lon: number): InspectInfo {
  const c = nearestCellLatLon(w.fine, lat * DEG, lon * DEG);
  const g = w.geo;
  const v = latLonToVec(lat * DEG, lon * DEG);
  const vel = plateVelocityAt(w, g.plate[c], v);
  const speed = Math.hypot(vel.vE, vel.vN);
  const dir = compassName((Math.atan2(vel.vE, vel.vN) / DEG + 360) % 360);
  const { setting, boundary } = describeSetting(w, c);
  const h = w.hydro;
  const lk = h.lake[c] >= 0 ? h.lakes[h.lake[c]] : null;
  const water = w.ocean[c] ? 'ocean' : lk ? (lk.kind === LakeKind.Fresh ? 'lake' : lk.kind === LakeKind.Saline ? 'salt lake' : 'salt flat') : null;
  const lake = lk
    ? {
        description:
          lk.kind === LakeKind.Fresh
            ? lk.outflow > 0
              ? 'Freshwater lake: fills its basin and overflows'
              : 'Freshwater lake with no outlet: evaporation holds it below its rim'
            : lk.kind === LakeKind.Saline
              ? 'Salt lake: no outlet; evaporation balances the inflow'
              : 'Salt flat: a dry lake bed, wet only after floods',
        areaKm2: lk.areaKm2,
        inflow: lk.inflow,
        outflow: lk.outflow,
      }
    : null;
  const k = w.biomes.koppen[c];
  const code = k < KOPPEN_CODES.length ? KOPPEN_CODES[k] : null;
  const regime = regimeOf(w, c);
  return {
    lat,
    lon,
    elevation: w.elev[c],
    water,
    plate: { id: g.plate[c], name: g.plates[g.plate[c]].name, speed, dir },
    crust: { continental: g.ctype[c] === 1, thickness: g.thick[c], age: g.age[c] },
    setting,
    bedrock: describeBedrock(w, c),
    surface: describeSurface(w, c),
    boundary,
    tempMonthly: Array.from(w.temp.subarray(c * MONTHS, c * MONTHS + MONTHS)),
    precipMonthly: Array.from(w.precip.subarray(c * MONTHS, c * MONTHS + MONTHS)),
    koppen: code,
    koppenName: code ? KOPPEN_NAMES[code] : null,
    biome: BIOME_INFO[w.biomes.biome[c]].name,
    discharge: w.ocean[c] || lk ? 0 : h.discharge[c],
    runoff: h.runoff[c],
    flowMonthly: regime ? Array.from(h.dischargeMonthly.subarray(c * MONTHS, c * MONTHS + MONTHS)) : null,
    regime,
    lake,
    pet: w.water.petYear[c],
    aet: w.water.aetYear[c],
    snowMonths: w.water.snowMonths[c],
    glacier: w.water.glacier[c] === 1,
    wetland: h.wetland[c] ? WETLAND_NAMES[h.wetland[c]] : null,
    endorheic: h.endorheic[c] === 1,
    ice: describeIce(w, c),
    iceAge: describeIceAge(w, c),
  };
}

function describeIce(w: World, c: number): string | null {
  const ice = w.ice;
  if (ice.shelf[c]) return 'Floating ice shelf, fed by the ice sheet behind it';
  const H = ice.thick[c];
  if (H <= 0) return null;
  const km2 = Math.round((ice.body[c] * 4 * Math.PI * w.params.radiusKm ** 2) / w.fine.n / 1000) * 1000;
  const kind = ice.body[c] >= 40 ? 'Ice sheet' : ice.body[c] >= 3 ? 'Ice cap' : 'Glacier';
  const smb = ice.smb[c] * 1000;
  const zone = smb >= 0 ? `accumulation zone, keeping ${Math.round(smb)} mm of water a year as new ice` : `melt zone, losing ${Math.round(-smb)} mm a year that flowing ice replaces`;
  const f = ice.cover[c];
  const here = f < 0.95 ? `its tongue ends here, covering ~${Math.round(f * 100)}% of the ground ${Math.round(H / f).toLocaleString('en-US')} m thick` : `${Math.round(H).toLocaleString('en-US')} m thick here`;
  return `${kind} (~${km2.toLocaleString('en-US')} km²), ${here}; ${zone}`;
}

function describeIceAge(w: World, c: number): string | null {
  const H = w.iceAge.thick[c];
  if (H <= 0) return null;
  const cut = w.iceAge.erosion[c];
  const carved = cut >= 5 ? `; its glaciers deepened the ground here by ~${Math.round(cut)} m` : '';
  return `Under ~${Math.round(H / 100) * 100} m of ice in the last ice age${carved}`;
}

/**
 * Brush preview: re-render only the part of the overview the stroke can have
 * changed, from the previewed relief.
 */
async function previewPatch(w: World, points: edits.LatLon[], radiusKm: number, render: RenderOptions): Promise<void> {
  const W = levelWidth(BASE_LEVEL);
  const H = levelHeight(BASE_LEVEL);
  const mLat = (radiusKm + 400) / w.params.radiusKm / DEG;
  let latMin = 90;
  let latMax = -90;
  let dMin = 0;
  let dMax = 0;
  const lon0 = points[0].lon;
  for (const p of points) {
    latMin = Math.min(latMin, p.lat);
    latMax = Math.max(latMax, p.lat);
    const d = p.lon - lon0 - 360 * Math.round((p.lon - lon0) / 360);
    dMin = Math.min(dMin, d);
    dMax = Math.max(dMax, d);
  }
  latMin = Math.max(-90, latMin - mLat);
  latMax = Math.min(90, latMax + mLat);
  const mLon = mLat / Math.max(0.05, Math.cos(Math.max(Math.abs(latMin), Math.abs(latMax)) * DEG));
  let x0 = Math.floor(((lon0 + dMin - mLon + 180) / 360) * W);
  let x1 = Math.ceil(((lon0 + dMax + mLon + 180) / 360) * W);
  if (x1 - x0 >= W) {
    x0 = 0;
    x1 = W;
  }
  const y0 = Math.max(0, Math.floor(((90 - latMax) / 180) * H));
  const y1 = Math.min(H, Math.ceil(((90 - latMin) / 180) * H));
  const f = buildFields(w, mesh!.id, -1);
  const win = { level: BASE_LEVEL, x0, y0, w: x1 - x0, h: y1 - y0 };
  const { rgba } = renderWindow({ mesh: mesh!, fields: f }, render, win);
  const bitmap = await createImageBitmap(new ImageData(rgba as Uint8ClampedArray<ArrayBuffer>, win.w, win.h));
  post({ type: 'patch', x0, y0, w: win.w, h: win.h, bitmap }, [bitmap]);
}

function ready(w: World, reason: string, report?: edits.EditReport, changed = true): void {
  if (changed) publish(w);
  post({ type: 'ready', summary: summary(w), reason, report, version });
}

function applyBrush(w: World, tool: BrushTool, points: edits.LatLon[], radiusKm: number, strength: number): edits.EditReport {
  switch (tool) {
    case 'land':
      return edits.brushLand(w, points, radiusKm, strength);
    case 'sea':
      return edits.brushSea(w, points, radiusKm, strength, false);
    case 'deepsea':
      return edits.brushSea(w, points, radiusKm, strength, true);
    case 'raise':
      return edits.brushRelief(w, points, radiusKm, strength, 1, true);
    case 'raiseOld':
      return edits.brushRelief(w, points, radiusKm, strength, 1, false);
    case 'lower':
      return edits.brushRelief(w, points, radiusKm, strength, -1, false);
  }
}

function applyOp(w: World, op: EditOp): edits.EditReport {
  switch (op.kind) {
    case 'island':
      return edits.addIsland(w, op.at, op.cause, op.sizeKm);
    case 'hotspot':
      return edits.addHotspot(w, op.at, op.strength);
    case 'plateVelocity': {
      const c = nearestCellLatLon(w.fine, op.at.lat * DEG, op.at.lon * DEG);
      return edits.setPlateVelocity(w, w.geo.plate[c], op.at, op.vEast, op.vNorth);
    }
    case 'advance':
      return edits.advanceTime(w, op.myr);
  }
}

function deriveFor(changed: Partial<WorldParams>): DeriveFrom | null {
  if ('seaLevel' in changed) return 'relief';
  if ('axialTiltDeg' in changed || 'temperatureOffset' in changed || 'retrograde' in changed || 'erosion' in changed) return 'climate';
  return null;
}

/** Full-planet image at the pyramid level of `width`, rendered in strips across the tile workers. */
async function exportFile(w: World, what: 'heightmap16' | 'layer', render: RenderOptions, width: number): Promise<void> {
  const level = levelForWidth(width);
  const W = levelWidth(level);
  const H = levelHeight(level);
  const seed = w.params.seed.replace(/[^\w-]+/g, '_');
  const STRIP = 64;
  const heights = what === 'heightmap16';
  const lo = -11000;
  const hi = 9000;
  const data16 = heights ? new Uint16Array(W * H) : null;
  const rgba = heights ? null : new Uint8ClampedArray(W * H * 4);
  const jobs: PixelWindow[] = [];
  for (let y = 0; y < H; y += STRIP) jobs.push({ level, x0: 0, y0: y, w: W, h: Math.min(STRIP, H - y) });
  let done = 0;
  progress(`Rendering ${W} × ${H}`, 0);
  await renderParallel(jobs, heights ? { layer: 'elevation', month: -1, rivers: false } : render, heights, (job, data) => {
    if (data16) {
      const E = new Float32Array(data);
      const o = job.y0 * W;
      for (let i = 0; i < E.length; i++) data16[o + i] = Math.round(Math.max(0, Math.min(1, (E[i] - lo) / (hi - lo))) * 65535);
    } else rgba!.set(new Uint8ClampedArray(data), job.y0 * W * 4);
    progress(`Rendering ${W} × ${H}`, (0.85 * ++done) / jobs.length);
  });
  progress('Encoding PNG', 0.9);
  if (data16) {
    const png = await encodePNG(W, H, data16, { channels: 1, bitDepth: 16 }, deflateBrowser);
    post({ type: 'file', name: `${seed}-heightmap-16bit_${lo}m_to_${hi}m.png`, mime: 'image/png', buffer: png.buffer as ArrayBuffer }, [png.buffer as ArrayBuffer]);
  } else {
    const png = await encodePNG(W, H, rgba!, { channels: 4, bitDepth: 8 }, deflateBrowser);
    const m = render.month >= 0 ? `-m${String(render.month + 1).padStart(2, '0')}` : '';
    post({ type: 'file', name: `${seed}-${render.layer}${m}-${W}.png`, mime: 'image/png', buffer: png.buffer as ArrayBuffer }, [png.buffer as ArrayBuffer]);
  }
  progress('Done', 1);
}

async function handle(msg: Request): Promise<void> {
  switch (msg.type) {
    case 'init':
      ports = msg.ports;
      for (const p of ports) p.onmessage = (e: MessageEvent<StripResult>) => stripWaiters.get(e.data.id)?.(e.data.data);
      if (world && mesh && fields) {
        broadcast({ type: 'mesh', mesh: shareable(mesh) });
        broadcast({ type: 'fields', fields: shareable(fields) });
      }
      break;
    case 'generate': {
      adopt(World.generate(msg.params, progress));
      ready(world!, 'generated');
      break;
    }
    case 'inspect': {
      const w = world;
      post({ type: 'inspect', id: msg.id, info: w ? inspect(w, msg.lat, msg.lon) : null });
      break;
    }
    case 'stroke': {
      const w = requireWorld();
      if (msg.phase === 'start') {
        w.beginEdit();
        strokeTool = msg.tool;
        strokeReport = null;
        strokeBase = { elev: w.elev, relief: w.relief.elev };
      }
      if (msg.points.length) {
        const r = applyBrush(w, msg.tool, msg.points, msg.radiusKm, msg.strength);
        strokeReport ??= r;
      }
      if (msg.phase === 'end') {
        strokeBase = null;
        progress('Re-deriving climate, rivers and biomes', 0);
        w.derive('tectonics', progress);
        const rec = w.commitEdit(strokeReport?.title ?? strokeTool ?? 'Brush');
        ready(w, rec ? 'edit' : 'noop', strokeReport ?? undefined);
        strokeTool = null;
      } else if (msg.points.length) {
        w.derivePreview(strokeBase ?? undefined);
        await previewPatch(w, msg.points, msg.radiusKm, msg.render);
      }
      break;
    }
    case 'edit': {
      const w = requireWorld();
      w.beginEdit();
      const report = applyOp(w, msg.op);
      if (report.failed) {
        w.cancelEdit();
        ready(w, 'failed', report, false);
        break;
      }
      progress('Re-deriving the world', 0);
      w.derive(report.derive, progress);
      w.commitEdit(report.title);
      ready(w, 'edit', report);
      break;
    }
    case 'undo':
    case 'redo': {
      const w = requireWorld();
      const rec = msg.type === 'undo' ? w.undo() : w.redo();
      if (rec) {
        progress(`${msg.type === 'undo' ? 'Undoing' : 'Redoing'} ${rec.label}`, 0);
        w.derive('tectonics', progress);
      }
      ready(w, msg.type, undefined, !!rec);
      break;
    }
    case 'setParams': {
      const w = requireWorld();
      const from = deriveFor(msg.params);
      w.beginEdit();
      w.params = { ...w.params, ...msg.params };
      if (from) w.derive(from, progress);
      w.commitEdit('Planet settings');
      ready(w, 'params');
      break;
    }
    case 'fitSeaLevel': {
      const w = requireWorld();
      w.beginEdit();
      w.params = { ...w.params, seaLevel: Math.round(w.seaLevelFor(msg.landFraction)) };
      w.derive('relief', progress);
      w.commitEdit('Sea level');
      ready(w, 'params');
      break;
    }
    case 'save': {
      const w = requireWorld();
      const buf = serializeWorld(w.params, w.geo, w.historyLog);
      post({ type: 'file', name: `${w.params.seed.replace(/[^\w-]+/g, '_')}.planet`, mime: 'application/octet-stream', buffer: buf }, [buf]);
      break;
    }
    case 'load': {
      const { params, geo, historyLog } = deserializeWorld(msg.buffer);
      const w = World.fromGeology(params, geo, progress);
      w.historyLog = historyLog;
      adopt(w);
      ready(w, 'loaded');
      break;
    }
    case 'export':
      await exportFile(requireWorld(), msg.what, msg.render, msg.width);
      break;
  }
}

ctx.onmessage = (ev: MessageEvent<Request>) => {
  handle(ev.data).catch((err: unknown) => {
    console.error(err);
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  });
};
