/**
 * Smoke test: generation, every edit operation, undo/redo, save/load
 * round-trip and the 16-bit heightmap export.
 *   npx tsx scripts/selftest.ts [outDir]
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';
import { encodePNG } from '../src/io/png';
import { deserializeWorld, serializeWorld } from '../src/io/save';
import { buildFields, meshData, NET_INFO } from '../src/render/context';
import { type LayerId, LAYERS, levelHeight, levelWidth, MAX_LEVEL, renderWindow, riverCourse, TILE } from '../src/render/raster';
import { DEG, latLonToVec } from '../src/core/math';
import { KOPPEN_CODES } from '../src/world/biomes';
import { Channel, CHANNEL_NAMES } from '../src/world/channels';
import { nearestCell } from '../src/core/mesh';
import * as edits from '../src/world/edits';
import { Rock, ROCK_INFO } from '../src/world/geology';
import { ICE_ISOSTASY } from '../src/world/glaciers';
import { Deposit } from '../src/world/sediment';
import { Soil, SOIL_COUNT } from '../src/world/soils';
import { DEFAULT_PARAMS } from '../src/world/types';
import { World } from '../src/world/world';

const outDir = process.argv[2] ?? 'out';
mkdirSync(outDir, { recursive: true });
let failures = 0;
const check = (ok: boolean, what: string) => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}`);
  if (!ok) failures++;
};

const world = World.generate({ ...DEFAULT_PARAMS, seed: 'selftest', cells: 20000 });
/** Worlds several checks read (none of them edits these): generated once each. */
const shared = new Map<string, World>();
function sharedWorld(seed: string, cells: number): World {
  const key = `${seed}@${cells}`;
  let w = shared.get(key);
  if (!w) shared.set(key, (w = World.generate({ ...DEFAULT_PARAMS, seed, cells })));
  return w;
}
check(world.geo.plates.length > 3, `generated ${world.geo.plates.length} plates, ${(world.landFraction() * 100).toFixed(1)}% land`);

// Hydrology: rain on ice-free land must leave as evaporation or runoff (the
// stores are in a steady annual cycle; glaciers instead pass snow downhill as
// ice), and rivers must carry water to the sea.
{
  const { fine, ocean, water, hydro, precipYear, ice } = world;
  let p = 0;
  let out = 0;
  for (let c = 0; c < fine.n; c++) {
    if (ocean[c] || ice.thick[c] > 0) continue;
    p += precipYear[c] * fine.area[c];
    out += (water.aetYear[c] + water.runoffYear[c]) * fine.area[c];
  }
  check(Math.abs(out - p) / p < 0.03, `land water balance closes (evaporation + runoff = ${((100 * out) / p).toFixed(1)}% of rain)`);
  let toSea = 0;
  for (let c = 0; c < fine.n; c++) {
    const rc = hydro.routing.receiver[c];
    if (!ocean[c] && rc >= 0 && ocean[rc]) toSea += hydro.discharge[c];
  }
  const kinds = [0, 1, 2].map((k) => hydro.lakes.filter((l) => l.kind === k).length);
  check(toSea > 0, `rivers deliver ${Math.round(toSea).toLocaleString('en-US')} m³/s to the sea; lakes: ${kinds[0]} fresh, ${kinds[1]} salt, ${kinds[2]} salt flats`);

  // Rivers keep their water across flat hollows (it is not pooled and let out
  // through another outlet), routing stays a tree after shores are turned into
  // their lakes, and channels are built for floods at least as big as the
  // wettest month's flow.
  const { order, receiver, count } = hydro.routing;
  const pos = new Int32Array(fine.n).fill(-1);
  for (let i = 0; i < count; i++) pos[order[i]] = i;
  let vanish = 0;
  let loops = 0;
  let smallBank = 0;
  for (let c = 0; c < fine.n; c++) {
    if (ocean[c]) continue;
    const rc = receiver[c];
    if (rc >= 0 && !ocean[rc] && pos[rc] >= pos[c]) loops++;
    let peak = 0;
    for (let m = 0; m < 12; m++) peak = Math.max(peak, hydro.dischargeMonthly[c * 12 + m]);
    if (hydro.bankfull[c] < peak - 1e-3) smallBank++;
    if (hydro.lake[c] >= 0 || hydro.discharge[c] < 50 || rc < 0 || ocean[rc] || hydro.lake[rc] >= 0 || hydro.endorheic[rc]) continue;
    if (hydro.discharge[rc] < 0.5 * hydro.discharge[c]) vanish++;
  }
  check(vanish === 0 && loops === 0 && smallBank === 0, `rivers run on unbroken (${vanish} lose half their water in a step), routing is a tree (${loops} loops), bankfull ≥ peak month everywhere (${smallBank} not)`);
}

// Ice: a colder planet must grow more of it, and ice must sit on its bed.
{
  const iceArea = (w: World) => {
    let a = 0;
    for (let c = 0; c < w.fine.n; c++) if (w.ice.thick[c] > 0) a += w.fine.area[c];
    return (a / (4 * Math.PI)) * 100;
  };
  const cold = World.generate({ ...DEFAULT_PARAMS, seed: 'selftest', cells: 20000, temperatureOffset: -8 });
  let ok = true;
  for (let c = 0; c < cold.fine.n; c++) {
    const want = cold.bed[c] + (1 - ICE_ISOSTASY) * cold.ice.thick[c];
    if (cold.ice.thick[c] < 0 || Math.abs(cold.elev[c] - want) > 1e-3) ok = false;
  }
  check(ok && iceArea(cold) > iceArea(world), `ice cover ${iceArea(world).toFixed(1)}% of the globe, ${iceArea(cold).toFixed(1)}% 8 °C colder`);

  // Ice is conserved cell by cell: what flows on is what flows in plus what the
  // ice-covered ground gains or loses, so a glacier ends in a partly covered
  // cell where its inflow runs out instead of blanketing the ground beyond.
  const { ice, fine } = cold;
  const R = cold.params.radiusKm * 1000;
  const inflow = new Float64Array(fine.n);
  for (let c = 0; c < fine.n; c++) {
    const r = ice.receiver[c];
    if (ice.cover[c] > 0 && r >= 0 && ice.cover[r] > 0) inflow[r] += ice.flux[c];
  }
  let bad = 0;
  let snouts = 0;
  for (let c = 0; c < fine.n; c++) {
    if (ice.cover[c] <= 0) continue;
    const own = ice.smb[c] * fine.area[c] * R * R * (1000 / 917);
    const cover = own >= 0 ? 1 : Math.min(1, inflow[c] / -own);
    const scale = Math.abs(own) + inflow[c] + 1;
    if (Math.abs(ice.flux[c] - Math.max(0, inflow[c] + own)) > 1e-3 * scale || Math.abs(ice.cover[c] - cover) > 1e-3 || !(ice.thick[c] > 0)) bad++;
    if (ice.cover[c] < 1) snouts++;
  }
  check(bad === 0, `ice mass balance closes on every glacier cell (${snouts} glacier tongues end inside their cell)`);
}

// Atolls crown drowned volcanoes in open water, clear of land and of each other
// (checked on a second world too: this one's tropical seas have none at this detail).
{
  let atolls = 0;
  let crowded = 0;
  for (const w of [world, World.generate({ ...DEFAULT_PARAMS, seed: 'aurelia', cells: 20000 })]) {
    const { fine, relief, ocean } = w;
    for (let c = 0; c < fine.n; c++) {
      if (relief.reef[c] !== 1) continue;
      atolls++;
      for (let k = fine.nbrStart[c]; k < fine.nbrStart[c + 1]; k++) if (!ocean[fine.nbr[k]] || relief.reef[fine.nbr[k]] === 1) crowded++;
    }
  }
  check(crowded === 0, `${atolls} atolls over two worlds, none beside land or another atoll`);
}

// Bedrock follows the plate history: every rock class turns up over a few
// worlds, ocean floor carries only oceanic rocks, and continents never do.
{
  const oceanic = new Set<number>([Rock.OceanBasalt, Rock.DeepSea, Rock.ArcVolcanic, Rock.Basalt, Rock.Limestone]);
  const seen = new Set<number>();
  let wrong = 0;
  let badK = 0;
  for (const w of [world, ...['aurelia', 'vale-fen-46'].map((seed) => sharedWorld(seed, 40000))]) {
    const { rock, erodibility } = w.lithology;
    for (let c = 0; c < w.fine.n; c++) {
      seen.add(rock[c]);
      if (w.geo.ctype[c] ? rock[c] === Rock.OceanBasalt || rock[c] === Rock.DeepSea : !oceanic.has(rock[c])) wrong++;
      if (!(erodibility[c] >= 0.25 && erodibility[c] <= 3.5)) badK++;
    }
  }
  check(seen.size === ROCK_INFO.length && wrong === 0 && badK === 0, `bedrock: ${seen.size} of ${ROCK_INFO.length} rock classes over three worlds, ${wrong} on the wrong crust, ${badK} erodibilities out of range`);
}

/** Cells of world w (default: the edited one) within km of a point. */
function cellsNear(at: { lat: number; lon: number }, km: number, w = world): number[] {
  const [vx, vy, vz] = latLonToVec(at.lat * DEG, at.lon * DEG);
  const cosR = Math.cos(km / w.params.radiusKm);
  const { x, y, z, n } = w.fine;
  const out: number[] = [];
  for (let c = 0; c < n; c++) if (x[c] * vx + y[c] * vy + z[c] * vz > cosR) out.push(c);
  return out;
}

// Mountains painted onto a quiet craton fold its cover and expose metamorphic cores.
{
  const w = World.generate({ ...DEFAULT_PARAMS, seed: 'selftest', cells: 40000 });
  const { geo, bnd, fine } = w;
  let at: edits.LatLon | null = null;
  for (let c = 0; c < fine.n && !at; c++) {
    if (geo.ctype[c] && geo.orogeny[c] > 1000 && geo.thick[c] > 34 && bnd.dColl[c] > 1500 && bnd.dArc[c] > 1500 && bnd.dDiv[c] > 1500 && w.elev[c] > 0) {
      at = { lat: fine.lat[c] / DEG, lon: fine.lon[c] / DEG };
    }
  }
  const orogenic = [Rock.Folded, Rock.Metamorphic, Rock.Granite, Rock.Ophiolite, Rock.Basalt];
  let raised: number[] = [];
  if (at) {
    for (let i = 0; i < 4; i++) edits.brushRelief(w, [at], 300, 1, 1, true);
    w.derive('tectonics');
    raised = cellsNear(at, 250, w).filter((c) => w.geo.ctype[c] && w.geo.orogeny[c] <= 1 && w.geo.thick[c] > 38);
  }
  const bad = raised.filter((c) => !orogenic.includes(w.lithology.rock[c] as Rock)).length;
  check(raised.length > 0 && bad === 0, `raised mountains carry mountain-belt rocks on ${raised.length - bad} of ${raised.length} cells`);
}

// Cell-scale processes are calibrated to one reference detail: the same world
// on a mesh four times finer erodes about as much rock (uncalibrated, a quarter less).
{
  const worlds = [80000, 320000].map((cells) => sharedWorld('selftest', cells));
  const [a, b] = worlds.map((w) => w.sedimentBudget.eroded);
  check(Math.abs(b / a - 1) < 0.12, `erosion matches across mesh details (${(a / 1e6).toFixed(2)}M km³ at 80k cells, ${(b / 1e6).toFixed(2)}M at 320k)`);

  // Sediment is conserved: what rivers, slopes, ice and wind take away is laid
  // down on land or shelf, dissolved, or carried to the deep sea.
  const w = worlds[0];
  const s = w.deposits.budget;
  const g = s.glacial;
  const wind = s.wind;
  const gap = (into: number, out: number) => Math.abs(out - into) / Math.max(1e-9, into);
  const fluvial = gap(s.eroded + s.crept, s.dissolved + s.basins + s.floodplains + s.fans + s.colluvium + s.deltas + s.shelves + s.lost);
  const glacial = gap(g.eroded + g.stripped, g.till + g.moraine + g.outwash + g.marine + g.exported + g.flour);
  const aeolian = gap(wind.deflated, wind.dunes + wind.loess + wind.toSea);
  const km3 = (v: number) => `${(v / 1e6).toFixed(2)}M`;
  check(
    Math.max(fluvial, glacial, aeolian) < 0.005,
    `sediment budget closes: rivers and slopes ${km3(s.eroded + s.crept)} km³ (${Math.round((100 * s.lost) / (s.eroded + s.crept))}% to the deep sea), ice ${km3(g.eroded + g.stripped)}, wind ${km3(wind.deflated)} (gaps ${(100 * fluvial).toFixed(2)}%, ${(100 * glacial).toFixed(2)}%, ${(100 * aeolian).toFixed(2)}%)`,
  );
  const kinds = new Set<number>();
  for (let c = 0; c < w.fine.n; c++) if (!w.ocean[c]) kinds.add(w.deposits.kind[c]);
  const deltaKinds = new Set(w.deposits.deltas.map((d) => d.kind));
  check(kinds.size >= 10 && deltaKinds.size === 3, `${kinds.size} kinds of surface deposit on land, ${w.deposits.deltas.length} river mouths with ${[...deltaKinds].join(', ')}-dominated deltas`);

  // Earth-like drylands and inland drainage over a few worlds: about a fifth of
  // the land drains to closed basins (Earth ~18-20%), Köppen B ~30% (here up to
  // 40% for supercontinent worlds), P/PET < 0.2 on ~20% (Earth 19.6%).
  const sample = [w, ...['aurelia', 'vale-fen-46'].map((seed) => World.generate({ ...DEFAULT_PARAMS, seed, cells: 80000 }))];
  const shares = sample.map((v) => {
    let land = 0;
    let endo = 0;
    let dryB = 0;
    let arid = 0;
    for (let c = 0; c < v.fine.n; c++) {
      if (v.ocean[c]) continue;
      const a = v.fine.area[c];
      land += a;
      if (v.hydro.endorheic[c]) endo += a;
      if (KOPPEN_CODES[v.biomes.koppen[c]]?.startsWith('B')) dryB += a;
      if (v.precipYear[c] < 0.2 * v.water.petYear[c]) arid += a;
    }
    return { endo: (100 * endo) / land, dryB: (100 * dryB) / land, arid: (100 * arid) / land };
  });
  const mean = (k: 'endo' | 'dryB' | 'arid') => shares.reduce((s, v) => s + v[k], 0) / shares.length;
  check(
    mean('endo') > 12 && mean('endo') < 28 && shares.every((v) => v.endo < 35),
    `inland drainage on ${shares.map((v) => v.endo.toFixed(0)).join('/')}% of land over three worlds (mean ${mean('endo').toFixed(1)}%)`,
  );
  check(
    mean('dryB') > 25 && mean('dryB') < 40 && mean('arid') > 14 && mean('arid') < 26,
    `drylands: Köppen B ${mean('dryB').toFixed(1)}% and P/PET < 0.2 on ${mean('arid').toFixed(1)}% of land (means over three worlds)`,
  );
}

// Soils: every order turns up over a few worlds, and fertility ranks by its
// physical causes: loess, river alluvium and volcanic ash soils farm better
// than desert, permafrost and podzol soils.
{
  const worlds = [world, sharedWorld('aurelia', 40000), sharedWorld('vale-fen-46', 40000), sharedWorld('selftest', 80000), sharedWorld('selftest', 320000)];
  const seen = new Set<number>();
  const sum = new Map<string, [number, number]>();
  const add = (k: string, a: number, f: number) => {
    const s = sum.get(k) ?? [0, 0];
    sum.set(k, [s[0] + a * f, s[1] + a]);
  };
  let bad = 0;
  for (const w of worlds) {
    const { order, fertility } = w.soils;
    for (let c = 0; c < w.fine.n; c++) {
      if (w.ocean[c]) continue;
      const o = order[c];
      const a = w.fine.area[c];
      const f = fertility[c];
      seen.add(o);
      if (!(f >= 0 && f <= 1) || o >= SOIL_COUNT) bad++;
      if (w.deposits.kind[c] === Deposit.Loess) add('loess', a, f);
      if (w.deposits.kind[c] === Deposit.Alluvium) add('alluvium', a, f);
      if (o === Soil.Andisol) add('volcanic', a, f);
      if (o === Soil.Aridisol) add('desert', a, f);
      if (o === Soil.Gelisol) add('permafrost', a, f);
      if (o === Soil.Spodosol) add('podzol', a, f);
    }
  }
  const mean = (k: string) => (sum.get(k)?.[1] ? sum.get(k)![0] / sum.get(k)![1] : NaN);
  const rich = ['loess', 'alluvium', 'volcanic'].map(mean);
  const poor = ['desert', 'permafrost', 'podzol'].map(mean);
  const f2 = (v: number[]) => v.map((x) => x.toFixed(2)).join('/');
  check(
    seen.size === SOIL_COUNT && bad === 0 && Math.min(...rich) > Math.max(...poor),
    `soils: ${seen.size} of ${SOIL_COUNT} orders over five worlds; fertility on loess/alluvium/volcanic ash ${f2(rich)} vs desert/permafrost/podzol ${f2(poor)}`,
  );
}

function op(label: string, fn: () => edits.EditReport): void {
  world.beginEdit();
  const r = fn();
  if (r.failed) {
    world.cancelEdit();
    console.log(`  (${label} declined: ${r.summary})`);
    return;
  }
  world.derive(r.derive);
  world.commitEdit(r.title);
  console.log(`  ${label}: ${r.title} — ${r.summary.slice(0, 110)}…`);
  if (r.options) console.log('     causes: ' + r.options.map((o) => `${o.id} ${Math.round(o.score * 100)}`).join(', '));
}

// Find an ocean point far from land and a coastal land point.
let oceanAt: edits.LatLon = { lat: 0, lon: 0 };
let landAt: edits.LatLon = { lat: 0, lon: 0 };
for (let c = 0; c < world.fine.n; c++) {
  if (world.ocean[c] && world.elev[c] < -4000 && Math.abs(world.fine.lat[c]) < 0.35) oceanAt = { lat: (world.fine.lat[c] * 180) / Math.PI, lon: (world.fine.lon[c] * 180) / Math.PI };
  if (!world.ocean[c] && world.elev[c] > 200) landAt = { lat: (world.fine.lat[c] * 180) / Math.PI, lon: (world.fine.lon[c] * 180) / Math.PI };
}
const before = world.elev.slice();
op('island auto', () => edits.addIsland(world, oceanAt, 'auto', 70));
{
  // A new oceanic island is volcanic rock, or the coral limestone capping a drowned volcano.
  const isle = cellsNear(oceanAt, 80).filter((c) => !world.geo.ctype[c] && (world.geo.volc[c] > 0.15 || world.relief.reef[c] === 1));
  const bad = isle.filter((c) => ![Rock.Basalt, Rock.ArcVolcanic, Rock.Limestone].includes(world.lithology.rock[c] as Rock)).length;
  check(isle.length > 0 && bad === 0, `the new island is ${isle.length} cells of volcanic rock or reef limestone (${bad} not)`);
}
op('island arc (forced)', () => edits.addIsland(world, { lat: oceanAt.lat + 5, lon: oceanAt.lon + 5 }, 'arc', 60));
op('microcontinent', () => edits.addIsland(world, { lat: oceanAt.lat - 6, lon: oceanAt.lon - 8 }, 'microcontinent', 60));
op('hotspot', () => edits.addHotspot(world, { lat: oceanAt.lat + 10, lon: oceanAt.lon - 15 }, 0.6));
op('land brush', () => edits.brushLand(world, [landAt, { lat: landAt.lat + 4, lon: landAt.lon + 4 }], 300, 0.7));
op('sea brush', () => edits.brushSea(world, [landAt], 250, 0.6, false));
op('rift brush', () => edits.brushSea(world, [{ lat: landAt.lat + 2, lon: landAt.lon }], 200, 0.6, true));
op('mountains', () => edits.brushRelief(world, [landAt], 250, 0.8, 1, true));
op('plate velocity', () => edits.setPlateVelocity(world, world.geo.plate[0], landAt, 3, -2));
op('advance time', () => edits.advanceTime(world, 5));

const edited = world.elev.slice();
const steps = world.undoStack.length;
while (world.undo()) {
  /* rewind */
}
world.derive('tectonics');
let maxDiff = 0;
for (let i = 0; i < before.length; i++) maxDiff = Math.max(maxDiff, Math.abs(before[i] - world.elev[i]));
check(maxDiff < 1e-3, `undo ×${steps} restores the original elevation (max diff ${maxDiff.toFixed(4)} m)`);
while (world.redo()) {
  /* replay */
}
world.derive('tectonics');
maxDiff = 0;
for (let i = 0; i < edited.length; i++) maxDiff = Math.max(maxDiff, Math.abs(edited[i] - world.elev[i]));
check(maxDiff < 1e-3, `redo replays every edit (max diff ${maxDiff.toFixed(4)} m)`);

const buf = serializeWorld(world.params, world.geo, world.historyLog);
const loaded = deserializeWorld(buf);
const w2 = World.fromGeology(loaded.params, loaded.geo);
maxDiff = 0;
for (let i = 0; i < edited.length; i++) maxDiff = Math.max(maxDiff, Math.abs(w2.elev[i] - world.elev[i]));
check(maxDiff < 1e-3, `save/load round-trip (${(buf.byteLength / 1024).toFixed(0)} KB, max diff ${maxDiff.toFixed(4)} m)`);

// Rendering: the overview, a deep tile, and agreement between neighboring tiles.
const ctx = { mesh: meshData(world.fine, 'selftest'), fields: buildFields(world, 'selftest', 1) };
const level = 1;
const { elevation } = renderWindow(ctx, { layer: 'elevation', month: -1, rivers: false }, { level, x0: 0, y0: 0, w: levelWidth(level), h: levelHeight(level) }, true);
const E = elevation!;
const data = new Uint16Array(E.length);
for (let i = 0; i < E.length; i++) data[i] = Math.round(Math.max(0, Math.min(1, (E[i] + 11000) / 20000)) * 65535);
const png = await encodePNG(levelWidth(level), levelHeight(level), data, { channels: 1, bitDepth: 16 }, (d) => deflateSync(d));
writeFileSync(`${outDir}/heightmap16.png`, png);
let landPx = 0;
for (const e of E) if (e > 0) landPx++;
check(png.length > 1000 && landPx > 0 && landPx < E.length, `16-bit heightmap PNG written (${(png.length / 1024).toFixed(0)} KB, ${((100 * landPx) / E.length).toFixed(0)}% land pixels)`);

for (const l of LAYERS) {
  const { rgba } = renderWindow(ctx, { layer: l.id as LayerId, month: 6, rivers: true }, { level: 7, x0: 40 * TILE, y0: 50 * TILE, w: TILE, h: TILE });
  let ok = true;
  for (let i = 3; i < rgba.length; i += 4) if (rgba[i] !== 255) ok = false;
  if (!ok) check(false, `${l.id} tile is fully opaque`);
}
check(true, `every layer renders a level-7 tile`);

// A tile rendered alone must match the same pixels inside a larger window: at
// a middle level, and at the deepest across the largest inland river (its
// channel at true width, banks, meanders and valley).
const seamAt = (level: number, tx: number, ty: number): number => {
  const opts = { layer: 'terrain' as const, month: -1, rivers: true };
  const big = renderWindow(ctx, opts, { level, x0: tx * TILE - 64, y0: ty * TILE - 64, w: TILE + 128, h: TILE + 128 }).rgba;
  const one = renderWindow(ctx, opts, { level, x0: tx * TILE, y0: ty * TILE, w: TILE, h: TILE }).rgba;
  let seam = 0;
  for (let y = 0; y < TILE; y++) {
    for (let x = 0; x < TILE; x++) {
      for (let c = 0; c < 3; c++) seam = Math.max(seam, Math.abs(one[4 * (y * TILE + x) + c] - big[4 * ((y + 64) * (TILE + 128) + x + 64) + c]));
    }
  }
  return seam;
};
let trunk = -1;
for (let ch = 0; ch < ctx.fields.rivStart.length - 1; ch++) {
  for (let i = ctx.fields.rivStart[ch]; i < ctx.fields.rivStart[ch + 1] - 1; i++) {
    if (!ctx.fields.rivCoast[i] && (trunk < 0 || ctx.fields.rivQ[i + 1] > ctx.fields.rivQ[trunk + 1])) trunk = i;
  }
}
const tc = ctx.fields.rivCell[trunk];
const deepX = Math.floor((((world.fine.lon[tc] + Math.PI) / (2 * Math.PI)) * levelWidth(MAX_LEVEL)) / TILE);
const deepY = Math.floor((((Math.PI / 2 - world.fine.lat[tc]) / Math.PI) * levelHeight(MAX_LEVEL)) / TILE);
const seams = [seamAt(6, 300, 40), seamAt(MAX_LEVEL, deepX, deepY)];
check(Math.max(...seams) <= 1, `tiles are seamless (max channel difference ${seams[0]} at level 6, ${seams[1]} at level ${MAX_LEVEL} on the largest river)`);

// Rivers end where they run into the drawn sea (open, or a drowned fjord) or a
// lake, or die away in dry country: none stops short on land, and a course
// that crosses an inlet does not reappear beyond it as a fragment without a
// source.
{
  const f = ctx.fields;
  const { rivStart, rivCell, rivQ } = f;
  const segOf = new Map<number, number>();
  for (let ch = 0; ch < rivStart.length - 1; ch++) for (let i = rivStart[ch]; i < rivStart[ch + 1] - 1; i++) segOf.set(rivCell[i], i);
  let mouths = 0;
  let stranded = 0;
  for (const L of [3, 5, 7, 9, MAX_LEVEL]) {
    const { wetCut, wetDrop, thr } = riverCourse(ctx, L);
    const drawn = (i: number) => rivQ[i + 1] - wetDrop[i] >= thr && wetCut[i] !== 0;
    for (let ch = 0; ch < rivStart.length - 1; ch++) {
      for (let i = rivStart[ch]; i < rivStart[ch + 1] - 1; i++) {
        if (!drawn(i)) continue;
        if (wetCut[i] > 0) {
          mouths++;
          continue;
        }
        const b = rivCell[i + 1];
        if (f.lake[b] >= 0) continue;
        const j = segOf.get(b);
        // Past its sea cell without meeting the water, or running on undrawn
        // (its node not in the water) though not drying up.
        if (f.ocean[b] || (j !== undefined && wetCut[j] !== 0 && !drawn(j) && rivQ[j + 1] >= rivQ[j])) stranded++;
      }
    }
  }
  check(stranded === 0, `rivers end in the drawn water (${mouths} mouths over 5 zoom levels), at a lake or where they dry up; ${stranded} stop short`);

  // A tributary carries only its own water down to the junction: with the
  // trunk's, the last stretch of every stream, however small, would be drawn
  // (a fishbone of stubs along each river).
  let joins = 0;
  let wrong = 0;
  for (let ch = 0; ch < rivStart.length - 1; ch++) {
    const last = rivStart[ch + 1] - 1;
    const b = rivCell[last];
    if (f.rivNext[ch] < 0 || f.ocean[b] || f.lake[b] >= 0) continue;
    joins++;
    if (rivQ[last] !== rivQ[last - 1]) wrong++;
  }
  check(joins > 0 && wrong === 0, `${joins} tributaries reach their trunks with only their own flow`);
}

// Lakes. A closed basin's rivers end in its lake, which evaporates about what
// they bring (none runs on past the lake to the basin's rim and stops dead
// there), and a lake draws as one body that its rivers' mouths and its outlet
// all reach.
{
  const worlds = [sharedWorld('selftest', 320000), sharedWorld('aurelia', 40000), sharedWorld('vale-fen-46', 40000)];
  let dead = 0;
  let closedN = 0;
  let lo = Infinity;
  let hi = 0;
  for (const w of worlds) {
    const { fine, hydro, ocean } = w;
    const rec = hydro.routing.receiver;
    for (let c = 0; c < fine.n; c++) {
      const rc = rec[c];
      // (A closed hollow too wet for a salt flat and too dry for a lake ends its rivers in a marsh.)
      if (ocean[c] || hydro.lake[c] >= 0 || hydro.discharge[c] < 50 || rc < 0 || ocean[rc] || hydro.lake[rc] >= 0 || !hydro.endorheic[rc] || hydro.wetland[rc]) continue;
      if (hydro.discharge[rc] < 0.5 * hydro.discharge[c]) dead++;
    }
    // Open-water evaporation less rain, and the runoff of the ground it drowns (m³/s).
    const toM3s = (w.params.radiusKm ** 2 * 1e6) / (365.25 * 86400 * 1000);
    const lost = new Float64Array(hydro.lakes.length);
    for (let c = 0; c < fine.n; c++) {
      const L = hydro.lake[c];
      if (L >= 0) lost[L] += (1.1 * w.water.petYear[c] - w.precipYear[c] + w.water.runoffYear[c]) * fine.area[c] * toM3s * hydro.lakeWet[c];
    }
    hydro.lakes.forEach((l, L) => {
      if (l.kind === 2 || l.outflow > 0 || l.inflow < 200) return;
      closedN++;
      lo = Math.min(lo, lost[L] / l.inflow);
      hi = Math.max(hi, lost[L] / l.inflow);
    });
  }
  check(
    dead === 0 && closedN > 0 && lo > 0.6 && hi < 1.05,
    `rivers end in their closed basins' lakes (${dead} stop dead on the way); ${closedN} closed lakes evaporate ${(100 * lo).toFixed(0)}–${(100 * hi).toFixed(0)}% of their inflow`,
  );

  // Drawn: the largest lakes, each wide enough to see; their water (the bed
  // under the surface) reached from the lowest point, pixel to pixel.
  const w = worlds[0];
  const { fine } = w;
  const lctx = { mesh: meshData(fine, 'lakes'), fields: buildFields(w, 'lakes', 1) };
  const f = lctx.fields;
  const thr = w.params.riverThreshold / 128;
  const big = w.hydro.lakes
    .map((l, i) => ({ l, i }))
    .filter(({ l }) => l.kind !== 2 && l.areaKm2 > 1500)
    .sort((a, b) => b.l.areaKm2 - a.l.areaKm2)
    .slice(0, 8);
  let ends = 0;
  let reached = 0;
  for (const { l, i } of big) {
    let low = -1;
    const mouths: number[] = [];
    for (let c = 0; c < fine.n; c++) {
      if (f.lake[c] === i && (low < 0 || w.elev[c] < w.elev[low])) low = c;
      const r = w.hydro.routing.receiver[c];
      if (f.lake[c] < 0 && !f.ocean[c] && r >= 0 && f.lake[r] === i && w.hydro.discharge[c] >= thr) mouths.push(r);
    }
    if (l.outflow > 0) mouths.push(l.outlet);
    const across = 2 * Math.sqrt(l.areaKm2 / Math.PI);
    const L = Math.max(3, Math.min(9, Math.round(Math.log2((2 * Math.PI * w.params.radiusKm * 160) / (256 * across)))));
    const W = 512;
    const toX = (c: number): number => ((fine.lon[c] + Math.PI) / (2 * Math.PI)) * levelWidth(L);
    const toY = (c: number): number => ((Math.PI / 2 - fine.lat[c]) / Math.PI) * levelHeight(L);
    const x0 = Math.round(toX(low) - W / 2);
    const y0 = Math.max(0, Math.round(toY(low) - W / 2));
    const { elevation } = renderWindow(lctx, { layer: 'terrain', month: -1, rivers: true }, { level: L, x0, y0, w: W, h: W }, true);
    const level = f.lakeLevel[i];
    const seen = new Uint8Array(W * W);
    const pix = (c: number): number => {
      const x = Math.round(toX(c) - x0);
      const y = Math.round(toY(c) - y0);
      return x >= 0 && x < W && y >= 0 && y < W ? y * W + x : -1;
    };
    const start = pix(low);
    if (start < 0 || elevation![start] >= level - 0.5) continue;
    const stack = [start];
    seen[start] = 1;
    while (stack.length > 0) {
      const k = stack.pop()!;
      const x = k % W;
      for (const q of [x > 0 ? k - 1 : -1, x < W - 1 ? k + 1 : -1, k - W, k + W]) {
        if (q < 0 || q >= W * W || seen[q] || elevation![q] >= level - 0.5) continue;
        seen[q] = 1;
        stack.push(q);
      }
    }
    for (const c of mouths) {
      const k = pix(c);
      if (k < 0) continue;
      ends++;
      if (seen[k]) reached++;
    }
  }
  check(ends > 0 && reached === ends, `the ${big.length} largest lakes draw as one body: ${reached} of ${ends} river mouths and outlets reach it`);
}

// ---- River channels (channels.ts) ----
// Patterns in Earth's proportions, the same at any mesh detail; distributary
// networks that conserve water at every split and reach the sea.
{
  const worlds = [80000, 320000].map((cells) => sharedWorld('selftest', cells));
  const patternShares = (w: World) => {
    const { fine, hydro, channels: ch } = w;
    const R = w.params.radiusKm;
    const rec = hydro.routing.receiver;
    const { x, y, z } = fine;
    const len = (c: number): number => {
      const r = rec[c];
      return r >= 0 ? Math.acos(Math.min(1, x[c] * x[r] + y[c] * y[r] + z[c] * z[r])) * R : 0;
    };
    const by = (keep: (c: number) => boolean): number[] => {
      const L = new Array<number>(8).fill(0);
      let total = 0;
      for (let c = 0; c < fine.n; c++) {
        if (!ch.pattern[c] || !keep(c)) continue;
        const l = len(c);
        L[ch.pattern[c]] += l;
        total += l;
      }
      return L.map((v) => (100 * v) / Math.max(total, 1e-9));
    };
    const drawn = (c: number): boolean => hydro.discharge[c] >= w.params.riverThreshold / 128;
    return {
      drawn: by(drawn),
      big: by((c) => hydro.discharge[c] >= 50),
      low: by((c) => drawn(c) && w.relief.rough[c] < 60),
      mountain: by((c) => drawn(c) && w.relief.rough[c] > 150),
      wet: by((c) => drawn(c) && hydro.wetland[c] !== 0),
      dry: by((c) => drawn(c) && hydro.wetland[c] === 0),
    };
  };
  const [coarse, fine] = worlds.map(patternShares);
  const pct = (v: number): string => v.toFixed(1);
  const lowland = fine.low.indexOf(Math.max(...fine.low.slice(1)));
  check(
    fine.drawn[Channel.Braided] >= 4 && fine.drawn[Channel.Braided] <= 25 && lowland === Channel.Meandering && fine.mountain[Channel.Confined] >= 60,
    `river patterns by length: ${pct(fine.drawn[Channel.Braided])}% braided, ${pct(fine.drawn[Channel.Meandering])}% meandering, ` +
      `${CHANNEL_NAMES[lowland].toLowerCase()} the commonest in lowlands, ${pct(fine.mountain[Channel.Confined])}% of mountain rivers confined`,
  );
  const compared = [Channel.Confined, Channel.Meandering, Channel.Braided, Channel.Anastomosing];
  const drift = Math.max(...compared.map((p) => Math.abs(coarse.big[p] - fine.big[p])));
  check(drift <= 10, `patterns of rivers over 50 m³/s agree at 80k and 320k cells (largest difference ${pct(drift)} points)`);
  check(
    fine.wet[Channel.Anastomosing] >= 3 * fine.dry[Channel.Anastomosing],
    `anastomosing rivers gather in wetlands (${pct(fine.wet[Channel.Anastomosing])}% of wetland river length, ${pct(fine.dry[Channel.Anastomosing])}% elsewhere)`,
  );

  const w = worlds[1];
  const ch = w.channels;
  let order = 0;
  let water = 0;
  let mouths = 0;
  let stranded = 0;
  let mismatched = 0;
  let stray = 0;
  for (const net of ch.networks) {
    const m = net.parent.length;
    const sum = new Float64Array(m);
    const kids = new Uint16Array(m);
    for (let j = 1; j < m; j++) {
      const p = net.parent[j];
      if (!(p >= 0 && p < j) || !(net.width[j] > 0)) order++;
      else {
        sum[p] += net.q[j];
        kids[p]++;
      }
    }
    for (let j = 0; j < m; j++) if (kids[j] > 1 && Math.abs(sum[j] - net.q[j]) > 1e-3 * net.q[j]) water++;
    if (net.kind === 'delta') {
      const kind = w.deposits.deltas[net.delta]?.kind;
      const tidal = net.form === 'estuarine';
      const waves = net.form === 'arcuate' || net.form === 'cuspate';
      if (tidal !== (kind === 'tide') || waves !== (kind === 'wave')) mismatched++;
      for (let j = 0; j < m; j++) {
        if (!net.mouth[j]) continue;
        mouths++;
        if (!w.ocean[nearestCell(w.fine, net.x[j], net.y[j], net.z[j], net.apex)]) stranded++;
      }
    }
    if (net.outline.length % 3 !== 0 || net.outline.length === 0) order++;
    for (const c of net.cells) if (ch.pattern[c] !== Channel.Distributary || ch.networks[ch.networkOf[c]] !== net) stray++;
  }
  const count = (k: string): number => ch.networks.filter((d) => d.kind === k).length;
  check(
    order === 0 && water === 0 && stranded === 0 && mismatched === 0 && stray === 0 && count('delta') > 0 && count('fan') > 0,
    `${count('delta')} delta, ${count('fan')} fan and ${count('inland')} inland-delta networks: parents before children, ` +
      `water conserved at every split, all ${mouths} delta mouths at sea, forms matching waves and tides`,
  );
  check(w.timings.channels < 1500, `channel patterns and networks in ${w.timings.channels.toFixed(0)} ms at 320k cells`);
}

// Advancing time moves continents without wasting them away: their crust is
// carried, not resampled, and only collisions and arcs change its area.
{
  const w = World.generate({ ...DEFAULT_PARAMS, seed: 'selftest-drift', cells: 20000 });
  const continent = () => {
    let a = 0;
    for (let c = 0; c < w.fine.n; c++) if (w.geo.ctype[c]) a += w.fine.area[c];
    return a;
  };
  const a0 = continent();
  for (let i = 0; i < 3; i++) edits.advanceTime(w, 34);
  const a1 = continent();
  check(Math.abs(a1 / a0 - 1) < 0.08, `continental crust after 102 Myr of plate motion: ${((100 * a1) / a0).toFixed(1)}% of its area at the start`);
}

// ---- River channel drawing (render) ----
// Braided and anastomosing reaches show several channels across their belts,
// patterns tile seamlessly, the pattern layer colors rivers by pattern, and
// flow arrows appear on the drainage layer only.
{
  const w = sharedWorld('selftest', 320000);
  const rc = { mesh: meshData(w.fine, 'selftest-320k'), fields: buildFields(w, 'selftest-320k', 1) };
  const f = rc.fields;
  const L = 10;
  const pxOf = (c: number) => ({
    x: ((w.fine.lon[c] + Math.PI) / (2 * Math.PI)) * levelWidth(L),
    y: ((Math.PI / 2 - w.fine.lat[c]) / Math.PI) * levelHeight(L),
  });
  /** Inland nodes of a pattern whose neighbors share it, widest belts first. */
  const nodesOf = (p: number, count: number): number[] => {
    const out: number[] = [];
    for (let ch = 0; ch < f.rivStart.length - 1; ch++) {
      for (let k = f.rivStart[ch] + 2; k < f.rivStart[ch + 1] - 2; k++) {
        if (f.rivPattern[k] !== p || f.rivPattern[k - 2] !== p || f.rivPattern[k + 2] !== p || f.rivCoast[k] || f.rivCoast[k - 1]) continue;
        if (w.hydro.lake[f.rivCell[k]] >= 0 || f.rivQ[k] < 50) continue;
        out.push(k);
      }
    }
    return out.sort((a, b) => f.rivBelt[b] - f.rivBelt[a]).slice(0, count);
  };
  const isWater = (r: number, g: number, b: number) => b > r + 40 && b > g + 10;
  /** Separate channels met along the cross-section through node k (perpendicular to the river there). */
  const channelsAcross = (k: number): number => {
    const a = pxOf(f.rivCell[k - 1]);
    const c = pxOf(f.rivCell[k]);
    const b = pxOf(f.rivCell[k + 1]);
    const tl = Math.hypot(b.x - a.x, b.y - a.y) || 1;
    const nx = -(b.y - a.y) / tl;
    const ny = (b.x - a.x) / tl;
    const half = Math.min(300, (0.6 * f.rivBelt[k]) / (w.params.radiusKm * 1000 * (Math.PI / levelHeight(L))));
    const S = 640;
    const x0 = Math.round(c.x) - S / 2;
    const y0 = Math.max(0, Math.round(c.y) - S / 2);
    const { rgba } = renderWindow(rc, { layer: 'terrain', month: -1, rivers: true }, { level: L, x0, y0, w: S, h: S });
    let runs = 0;
    let inWater = false;
    for (let t = -half; t <= half; t += 0.5) {
      const px = Math.round(c.x + nx * t) - x0;
      const py = Math.round(c.y + ny * t) - y0;
      if (px < 0 || py < 0 || px >= S || py >= S) continue;
      const o = 4 * (py * S + px);
      const wet = isWater(rgba[o], rgba[o + 1], rgba[o + 2]);
      if (wet && !inWater) runs++;
      inWater = wet;
    }
    return runs;
  };
  const braided = nodesOf(Channel.Braided, 5);
  const anas = nodesOf(Channel.Anastomosing, 5);
  const runsB = braided.map(channelsAcross);
  const runsA = anas.map(channelsAcross);
  const multi = (runs: number[]) => runs.filter((r) => r >= 2).length;
  check(
    braided.length > 0 && anas.length > 0 && multi(runsB) >= Math.ceil(0.6 * runsB.length) && multi(runsA) >= Math.ceil(0.6 * runsA.length),
    `braided and anastomosing reaches show several channels across their belts at level ${L} (braided: ${runsB.join(', ')}; anastomosing: ${runsA.join(', ')})`,
  );
  // Seamless where a braided belt crosses tile edges, and where arrows do.
  const seam = (layer: LayerId, level: number, c: number): number => {
    const tx = Math.floor((((w.fine.lon[c] + Math.PI) / (2 * Math.PI)) * levelWidth(level)) / TILE);
    const ty = Math.floor((((Math.PI / 2 - w.fine.lat[c]) / Math.PI) * levelHeight(level)) / TILE);
    const opts = { layer, month: -1, rivers: true };
    const big = renderWindow(rc, opts, { level, x0: tx * TILE - TILE, y0: Math.max(0, ty * TILE - TILE), w: 3 * TILE, h: 3 * TILE }).rgba;
    let worst = 0;
    for (const [ox, oy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
      const one = renderWindow(rc, opts, { level, x0: (tx - 1 + ox) * TILE + TILE / 2, y0: Math.max(0, ty * TILE - TILE) + oy * TILE + TILE / 2, w: TILE, h: TILE }).rgba;
      for (let y = 0; y < TILE; y++) {
        for (let x = 0; x < TILE; x++) {
          const a = 4 * (y * TILE + x);
          const b = 4 * ((oy * TILE + TILE / 2 + y) * 3 * TILE + ox * TILE + TILE / 2 + x);
          for (let k = 0; k < 3; k++) worst = Math.max(worst, Math.abs(one[a + k] - big[b + k]));
        }
      }
    }
    return worst;
  };
  let trunkNode = 0;
  for (let k = 0; k < f.rivQ.length; k++) if (!f.rivCoast[k] && f.rivQ[k] > f.rivQ[trunkNode]) trunkNode = k;
  const seams = [braided.length ? seam('terrain', 11, f.rivCell[braided[0]]) : 0, anas.length ? seam('terrain', 10, f.rivCell[anas[0]]) : 0, seam('drainage', 6, f.rivCell[trunkNode])];
  check(Math.max(...seams) <= 1, `channel patterns and flow arrows tile seamlessly (max difference ${seams.join(', ')})`);
  // Flow arrows: white arrowheads on the drainage layer, none without rivers or on other layers.
  const c = f.rivCell[trunkNode];
  const L6 = 6;
  const win = {
    level: L6,
    x0: Math.round(((w.fine.lon[c] + Math.PI) / (2 * Math.PI)) * levelWidth(L6)) - 256,
    y0: Math.max(0, Math.round(((Math.PI / 2 - w.fine.lat[c]) / Math.PI) * levelHeight(L6)) - 256),
    w: 512,
    h: 512,
  };
  const whites = (layer: LayerId, rivers: boolean): number => {
    const { rgba } = renderWindow(rc, { layer, month: -1, rivers }, win);
    let n = 0;
    for (let i = 0; i < rgba.length; i += 4) if (rgba[i] >= 240 && rgba[i + 1] >= 240 && rgba[i + 2] >= 230) n++;
    return n;
  };
  const arrowsOn = whites('drainage', true);
  const arrowsOff = whites('drainage', false);
  const terrainOn = whites('terrain', true);
  const terrainOff = whites('terrain', false);
  check(arrowsOn - arrowsOff > 8 && terrainOn <= terrainOff, `flow arrows on the drainage layer (${arrowsOn - arrowsOff} arrow pixels around the largest river), none in natural color`);
  // The pattern layer colors a braided reach orange.
  if (braided.length) {
    const p = pxOf(f.rivCell[braided[0]]);
    const { rgba } = renderWindow(rc, { layer: 'channels', month: -1, rivers: true }, { level: 8, x0: Math.round(p.x / 4) - 128, y0: Math.max(0, Math.round(p.y / 4) - 128), w: 256, h: 256 });
    let orange = 0;
    for (let i = 0; i < rgba.length; i += 4) if (rgba[i] > 180 && rgba[i + 1] > 70 && rgba[i + 1] < 150 && rgba[i + 2] < 90) orange++;
    check(orange > 10, `the channel-pattern layer colors a braided reach as braided (${orange} px)`);
  }
  // Braids form one network: their channels part from and rejoin one another
  // (and the main channel), so no thread lies loose in the belt. Water the
  // rivers add (lakes are there either way) that touches nothing else.
  const loose = (k: number): number => {
    const c0 = pxOf(f.rivCell[k]);
    const S = 384;
    const win = { level: L, x0: Math.round(c0.x) - S / 2, y0: Math.max(0, Math.round(c0.y) - S / 2), w: S, h: S };
    const on = renderWindow(rc, { layer: 'terrain', month: -1, rivers: true }, win).rgba;
    const off = renderWindow(rc, { layer: 'terrain', month: -1, rivers: false }, win).rgba;
    const wet = new Uint8Array(S * S);
    for (let i = 0; i < S * S; i++) wet[i] = isWater(on[4 * i], on[4 * i + 1], on[4 * i + 2]) && !isWater(off[4 * i], off[4 * i + 1], off[4 * i + 2]) ? 1 : 0;
    const seen = new Uint8Array(S * S);
    let total = 0;
    let stray = 0;
    const stack: number[] = [];
    for (let s = 0; s < S * S; s++) {
      if (!wet[s] || seen[s]) continue;
      let n = 0;
      let edge = false;
      stack.push(s);
      seen[s] = 1;
      while (stack.length) {
        const p = stack.pop()!;
        n++;
        const x = p % S;
        const y = (p - x) / S;
        if (x === 0 || y === 0 || x === S - 1 || y === S - 1) edge = true;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx;
            const yy = y + dy;
            if (xx < 0 || yy < 0 || xx >= S || yy >= S) continue;
            const o = yy * S + xx;
            if (wet[o] && !seen[o]) {
              seen[o] = 1;
              stack.push(o);
            }
          }
        }
      }
      total += n;
      if (!edge && n >= 6) stray += n;
    }
    return total > 0 ? (100 * stray) / total : 0;
  };
  const looseB = braided.slice(0, 3).map(loose);
  const looseA = anas.slice(0, 3).map(loose);
  check(
    [...looseB, ...looseA].every((v) => v < 3),
    `braided and anastomosing channels join up: loose river water ${[...looseB, ...looseA].map((v) => v.toFixed(1)).join(', ')}% of the belts' water at level ${L}`,
  );
  // A delta's main channel runs over land the delta built, from its apex to
  // its mouth; the river's own course stops at the apex.
  {
    const nNet = f.netStart.length - 1;
    const deltas: number[] = [];
    for (let n = 0; n < nNet; n++) if (f.netInfo[NET_INFO * n] === 0) deltas.push(n);
    deltas.sort((a, b) => f.netInfo[NET_INFO * b + 5] - f.netInfo[NET_INFO * a + 5]);
    const shares: number[] = [];
    let replaced = 0;
    for (let k = 0; k < f.rivNet.length; k++) if (f.rivNet[k] >= 0) replaced++;
    for (const n of deltas.slice(0, 3)) {
      const R0 = f.netInfo[NET_INFO * n + 5];
      const Ld = Math.max(4, Math.min(10, Math.round(Math.log2((150 * Math.PI) / (256 * R0)))));
      const s0 = f.netStart[n];
      const path: number[] = [];
      for (let g = s0, guard = 0; g >= 0 && guard < 10000; g = f.netNext[g], guard++) path.push(g);
      const ll = (g: number) => ({ lat: Math.asin(f.netZ[g]), lon: Math.atan2(f.netY[g], f.netX[g]) });
      const a = ll(s0);
      const cx = Math.round(((a.lon + Math.PI) / (2 * Math.PI)) * levelWidth(Ld));
      const cy = Math.round(((Math.PI / 2 - a.lat) / Math.PI) * levelHeight(Ld));
      const S = 640;
      const win = { level: Ld, x0: cx - S / 2, y0: Math.max(0, cy - S / 2), w: S, h: S };
      // (On the pattern layer the sea is a flat dark blue; channels show only over land.)
      const { rgba } = renderWindow(rc, { layer: 'channels', month: -1, rivers: true }, win);
      let land = 0;
      let all = 0;
      for (let i = 1; i < path.length - 1; i++) {
        const g0 = path[i - 1];
        const g1 = path[i];
        let mx = f.netX[g0] + f.netX[g1];
        let my = f.netY[g0] + f.netY[g1];
        let mz = f.netZ[g0] + f.netZ[g1];
        const l = Math.hypot(mx, my, mz);
        mx /= l;
        my /= l;
        mz /= l;
        let lon = Math.atan2(my, mx);
        lon += 2 * Math.PI * Math.round((a.lon - lon) / (2 * Math.PI));
        const px = Math.round(((lon + Math.PI) / (2 * Math.PI)) * levelWidth(Ld)) - win.x0;
        const py = Math.round(((Math.PI / 2 - Math.asin(mz)) / Math.PI) * levelHeight(Ld)) - win.y0;
        if (px < 0 || py < 0 || px >= S || py >= S) continue;
        all++;
        const o = 4 * (py * S + px);
        if (!(rgba[o] < 50 && rgba[o + 1] < 60 && rgba[o + 2] < 85)) land++;
      }
      shares.push(all > 0 ? (100 * land) / all : 100);
    }
    check(
      deltas.length > 0 && replaced > 0 && shares.every((s) => s >= 90),
      `the largest deltas' main channels run over delta land from apex to mouth (${shares.map((s) => s.toFixed(0)).join(', ')}% of their length); rivers hand over to ${nNet} networks at their apexes`,
    );
  }

  // Coasts shoal toward the shore: no trough dug along them, no step out at
  // sea where a shoreface rule lets go.
  {
    const mesh = w.fine;
    const coasts: number[] = [];
    for (let c = 0; c < mesh.n; c += 53) {
      if (!w.ocean[c] || w.elev[c] < -60) continue;
      let land = false;
      for (let k = mesh.nbrStart[c]; k < mesh.nbrStart[c + 1]; k++) if (!w.ocean[mesh.nbr[k]]) land = true;
      if (land) coasts.push(c);
    }
    const Lc = 8;
    const S = 384;
    let windows = 0;
    let deeper = 0;
    const gaps: number[] = [];
    for (let s = 0; s < 8 && coasts.length > 0; s++) {
      const c = coasts[Math.floor(((s + 0.5) / 8) * coasts.length)];
      const cx = Math.round(((mesh.lon[c] + Math.PI) / (2 * Math.PI)) * levelWidth(Lc));
      const cy = Math.round(((Math.PI / 2 - mesh.lat[c]) / Math.PI) * levelHeight(Lc));
      const E = renderWindow(rc, { layer: 'elevation', month: -1, rivers: false }, { level: Lc, x0: cx - S / 2, y0: Math.max(0, cy - S / 2), w: S, h: S }, true).elevation!;
      // Distance from land (px), out to 24.
      const dist = new Int16Array(S * S).fill(-1);
      let queue: number[] = [];
      for (let p = 0; p < S * S; p++) if (E[p] >= 0) {
        dist[p] = 0;
        queue.push(p);
      }
      for (let d = 1; d <= 24 && queue.length; d++) {
        const next: number[] = [];
        for (const p of queue) {
          const px = p % S;
          for (const q of [px > 0 ? p - 1 : -1, px < S - 1 ? p + 1 : -1, p - S, p + S]) {
            if (q < 0 || q >= S * S || dist[q] >= 0) continue;
            dist[q] = d;
            next.push(q);
          }
        }
        queue = next;
      }
      let n1 = 0;
      let s1 = 0;
      let n2 = 0;
      let s2 = 0;
      for (let p = 0; p < S * S; p++) {
        if (dist[p] >= 2 && dist[p] <= 8) {
          n1++;
          s1 += E[p];
        } else if (dist[p] >= 14 && dist[p] <= 22) {
          n2++;
          s2 += E[p];
        }
      }
      if (n1 < 200 || n2 < 200) continue;
      windows++;
      const gap = s1 / n1 - s2 / n2;
      gaps.push(gap);
      if (gap < -3) deeper++;
    }
    check(windows >= 4 && deeper === 0, `coasts shoal toward the shore over ${windows} coastlines (inshore minus offshore depth ${gaps.map((g) => g.toFixed(0)).join(', ')} m; ${deeper} deeper inshore)`);
  }

  // Rivers run at or below the ground beside them, never along a ridge
  // across a hollow of the relief.
  {
    const Lr = 9;
    const S = 768;
    const mPx = w.params.radiusKm * 1000 * (Math.PI / levelHeight(Lr));
    const picks: number[] = [];
    for (let ch = 0; ch < f.rivStart.length - 1; ch++) {
      for (let k = f.rivStart[ch] + 3; k < f.rivStart[ch + 1] - 3; k += 9) {
        const c = f.rivCell[k];
        if (f.rivQ[k] >= 80 && !w.ocean[c] && w.hydro.lake[c] < 0 && f.amp[c] > 120 && f.rivNet[k] < 0) picks.push(k);
      }
    }
    picks.sort((a, b) => f.amp[f.rivCell[b]] - f.amp[f.rivCell[a]]);
    let tested = 0;
    let raised = 0;
    for (const k of picks.filter((_, i) => i % Math.max(1, Math.floor(picks.length / 10)) === 0).slice(0, 10)) {
      const c0 = f.rivCell[k];
      const cx = Math.round(((w.fine.lon[c0] + Math.PI) / (2 * Math.PI)) * levelWidth(Lr));
      const cy = Math.round(((Math.PI / 2 - w.fine.lat[c0]) / Math.PI) * levelHeight(Lr));
      const x0 = cx - S / 2;
      const y0 = Math.max(0, cy - S / 2);
      const E = renderWindow(rc, { layer: 'elevation', month: -1, rivers: false }, { level: Lr, x0, y0, w: S, h: S }, true).elevation!;
      for (let ch = 0; ch < f.rivStart.length - 1; ch++) {
        for (let j = f.rivStart[ch] + 1; j < f.rivStart[ch + 1] - 1; j++) {
          const c = f.rivCell[j];
          if (f.rivQ[j] < 40 || w.ocean[c] || w.hydro.lake[c] >= 0 || f.rivNet[j] >= 0) continue;
          let lon = w.fine.lon[c];
          lon += 2 * Math.PI * Math.round((w.fine.lon[c0] - lon) / (2 * Math.PI));
          const px = Math.round(((lon + Math.PI) / (2 * Math.PI)) * levelWidth(Lr)) - x0;
          const py = Math.round(((Math.PI / 2 - w.fine.lat[c]) / Math.PI) * levelHeight(Lr)) - y0;
          // Out past the valley's flank: its half-width and half again.
          const r = Math.round((1.6 * (1800 + 2400 * Math.log10(f.rivQ[j] / 40 + 1))) / mPx);
          if (px < r || py < r || px >= S - r || py >= S - r) continue;
          const ring: number[] = [];
          for (let a = 0; a < 16; a++) ring.push(E[Math.round(py + r * Math.sin((a * Math.PI) / 8)) * S + Math.round(px + r * Math.cos((a * Math.PI) / 8))]);
          ring.sort((u, v) => u - v);
          tested++;
          if (E[py * S + px] > ring[8] + 5) raised++;
        }
      }
    }
    check(tested >= 20 && raised <= 0.1 * tested, `rivers run below the ground beside them (${raised} of ${tested} nodes in rugged country above the median of their surroundings)`);
  }

  // Ponds on delta and wetland plains are lakes, not specks: none under a few
  // pixels from afar.
  {
    const nets: number[] = [];
    // (Warm, wet plains: in cold country the ice-scoured lowlands have countless small lakes of their own.)
    for (let n = 0; n < f.netStart.length - 1; n++) if (f.netInfo[NET_INFO * n] !== 1 && f.netInfo[NET_INFO * n + 7] < 0.5 && f.netInfo[NET_INFO * n + 11] > 8) nets.push(n);
    nets.sort((a, b) => f.netInfo[NET_INFO * b + 5] - f.netInfo[NET_INFO * a + 5]);
    let specks = 0;
    let ponds = 0;
    let windows = 0;
    for (const n of nets.slice(0, 4)) {
      for (const Lp of [7, 8]) {
        const g = f.netStart[n];
        const lon = Math.atan2(f.netY[g], f.netX[g]);
        const lat = Math.asin(f.netZ[g]);
        const S = 320;
        const cx = Math.round(((lon + Math.PI) / (2 * Math.PI)) * levelWidth(Lp));
        const cy = Math.round(((Math.PI / 2 - lat) / Math.PI) * levelHeight(Lp));
        const { rgba } = renderWindow(rc, { layer: 'terrain', month: -1, rivers: false }, { level: Lp, x0: cx - S / 2, y0: Math.max(0, cy - S / 2), w: S, h: S });
        windows++;
        const seen = new Uint8Array(S * S);
        for (let p = 0; p < S * S; p++) {
          if (seen[p] || !isWater(rgba[4 * p], rgba[4 * p + 1], rgba[4 * p + 2])) continue;
          let size = 0;
          const stack = [p];
          const comp: number[] = [];
          seen[p] = 1;
          while (stack.length && size < 12) {
            const q = stack.pop()!;
            size++;
            comp.push(q);
            const qx = q % S;
            for (const r of [qx > 0 ? q - 1 : -1, qx < S - 1 ? q + 1 : -1, q - S, q + S]) {
              if (r < 0 || r >= S * S || seen[r] || !isWater(rgba[4 * r], rgba[4 * r + 1], rgba[4 * r + 2])) continue;
              seen[r] = 1;
              stack.push(r);
            }
          }
          if (size > 3) {
            ponds++;
            continue;
          }
          // (A speck stands alone: bits of a ragged shore lie beside their water.)
          const px0 = p % S;
          const py0 = Math.floor(p / S);
          let alone = true;
          for (let dy = -3; dy <= 3 && alone; dy++) {
            for (let dx = -3; dx <= 3; dx++) {
              const qx = px0 + dx;
              const qy = py0 + dy;
              if (qx < 0 || qy < 0 || qx >= S || qy >= S) continue;
              const q = qy * S + qx;
              if (!comp.includes(q) && isWater(rgba[4 * q], rgba[4 * q + 1], rgba[4 * q + 2])) {
                alone = false;
                break;
              }
            }
          }
          if (alone) specks++;
        }
      }
    }
    check(windows > 0 && specks <= 2 * windows, `delta and wetland ponds read as lakes from afar (${specks} specks of 3 px or less, ${ponds} larger waters over ${windows} views)`);
  }

  // Deltas keep to their own river's ground: no distributary crosses another
  // sizeable river reaching the sea elsewhere, and the river runs on into its
  // delta's main channel with no hairpin at the apex.
  {
    const mesh = w.fine;
    const { receiver, order, count } = w.hydro.routing;
    const outM = new Int32Array(mesh.n).fill(-1);
    for (let i = 0; i < count; i++) {
      const c = order[i];
      if (w.ocean[c]) continue;
      const r = receiver[c];
      outM[c] = r < 0 ? -1 : w.ocean[r] ? c : outM[r];
    }
    let edges = 0;
    let crossing = 0;
    for (const net of w.channels.networks) {
      if (net.kind !== 'delta' || net.delta < 0) continue;
      const L = w.deposits.deltas[net.delta].river;
      const qBar = Math.max(150, 0.15 * w.hydro.bankfull[L]);
      let cell = L;
      for (let j = 1; j < net.parent.length; j++) {
        const p = net.parent[j];
        edges++;
        for (let t = 0.25; t < 1; t += 0.25) {
          const vx = net.x[p] + (net.x[j] - net.x[p]) * t;
          const vy = net.y[p] + (net.y[j] - net.y[p]) * t;
          const vz = net.z[p] + (net.z[j] - net.z[p]) * t;
          cell = nearestCell(mesh, vx, vy, vz, cell);
          if (!w.ocean[cell] && w.hydro.bankfull[cell] >= qBar && outM[cell] !== L) {
            crossing++;
            break;
          }
        }
      }
    }
    let turns = 0;
    let sharp = 0;
    for (let n = 0; n < f.netStart.length - 1; n++) {
      if (f.netInfo[NET_INFO * n] !== 0 || Number.isNaN(f.netUp[3 * n])) continue;
      const g0 = f.netStart[n];
      const R0 = f.netInfo[NET_INFO * n + 5];
      let gm = g0;
      for (let guard = 0; guard < 400; guard++) {
        const nx = f.netNext[gm];
        if (nx < 0) break;
        gm = nx;
        if (Math.acos(Math.min(1, f.netX[gm] * f.netX[g0] + f.netY[gm] * f.netY[g0] + f.netZ[gm] * f.netZ[g0])) > 0.12 * R0) break;
      }
      if (gm === g0) continue;
      const ix = f.netX[g0] - f.netUp[3 * n];
      const iy = f.netY[g0] - f.netUp[3 * n + 1];
      const iz = f.netZ[g0] - f.netUp[3 * n + 2];
      const ox = f.netX[gm] - f.netX[g0];
      const oy = f.netY[gm] - f.netY[g0];
      const oz = f.netZ[gm] - f.netZ[g0];
      const cosA = (ix * ox + iy * oy + iz * oz) / (Math.hypot(ix, iy, iz) * Math.hypot(ox, oy, oz) || 1);
      turns++;
      if (cosA < Math.cos((100 * Math.PI) / 180)) sharp++;
    }
    check(
      edges > 0 && crossing <= 0.01 * edges && turns > 0 && sharp <= 0.03 * turns,
      `deltas keep to their own rivers' ground (${crossing} of ${edges} distributaries cross another river) and rivers run on into their main channels (${sharp} of ${turns} turn back more than 100° at the apex)`,
    );
  }
}

// ---- Fjords, trough lakes and strandflats (render) ----
// Glacial troughs drowned by the sea wind inland and branch up side valleys,
// tile seamlessly, hold ribbon lakes their rivers run through (never drawn over
// the water), and strandflat islets stand only off glaciated hard-rock coasts.
{
  const w = sharedWorld('selftest', 320000);
  const rc = { mesh: meshData(w.fine, 'selftest-320k-fjord'), fields: buildFields(w, 'selftest-320k-fjord', 1) };
  const f = rc.fields;
  const { fine } = w;
  const Rk = w.params.radiusKm;
  const arc = (a: number, b: number): number => Math.acos(Math.min(1, fine.x[a] * fine.x[b] + fine.y[a] * fine.y[b] + fine.z[a] * fine.z[b])) * Rk;
  const nCh = f.rivStart.length - 1;
  let systems = 0;
  let arms = 0;
  let longest = 0;
  const winding: number[] = [];
  /** A mid-stem node of the longest fjord (for the seam test). */
  let fjordNode = -1;
  for (let ch = 0; ch < nCh; ch++) {
    const last = f.rivStart[ch + 1] - 1;
    if (f.rivFjord[last] < 0.5 || f.rivFjord[last - 1] < 0.5) continue;
    let k = last;
    let len = 0;
    while (k > f.rivStart[ch] && f.rivFjord[k - 1] >= 0.5) {
      len += arc(f.rivCell[k], f.rivCell[k - 1]);
      k--;
    }
    if (!f.rivSea[ch]) {
      if (len > 0) arms++;
      continue;
    }
    systems++;
    if (last - k >= 3) winding.push(len / Math.max(1, arc(f.rivCell[k], f.rivCell[last])));
    if (len > longest) {
      longest = len;
      fjordNode = (k + last) >> 1;
    }
  }
  winding.sort((a, b) => a - b);
  const medWind = winding.length ? winding[winding.length >> 1] : 1;
  check(
    systems >= 20 && arms >= 5 && longest >= 60 && medWind >= 1.04,
    `fjords: ${systems} drowned glacial troughs and ${arms} side arms off them, the longest ${Math.round(longest)} km, winding ${medWind.toFixed(2)}× their straight length (median)`,
  );
  // Seamless across tile edges on the longest fjord.
  const seam = (layer: LayerId, level: number, c: number): number => {
    const tx = Math.floor((((fine.lon[c] + Math.PI) / (2 * Math.PI)) * levelWidth(level)) / TILE);
    const ty = Math.floor((((Math.PI / 2 - fine.lat[c]) / Math.PI) * levelHeight(level)) / TILE);
    const opts = { layer, month: -1, rivers: true };
    const big = renderWindow(rc, opts, { level, x0: tx * TILE - TILE, y0: Math.max(0, ty * TILE - TILE), w: 3 * TILE, h: 3 * TILE }).rgba;
    let worst = 0;
    for (const [ox, oy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
      const one = renderWindow(rc, opts, { level, x0: (tx - 1 + ox) * TILE + TILE / 2, y0: Math.max(0, ty * TILE - TILE) + oy * TILE + TILE / 2, w: TILE, h: TILE }).rgba;
      for (let y = 0; y < TILE; y++) {
        for (let x = 0; x < TILE; x++) {
          const a = 4 * (y * TILE + x);
          const b = 4 * ((oy * TILE + TILE / 2 + y) * 3 * TILE + ox * TILE + TILE / 2 + x);
          for (let q = 0; q < 3; q++) worst = Math.max(worst, Math.abs(one[a + q] - big[b + q]));
        }
      }
    }
    return worst;
  };
  const seams = fjordNode >= 0 ? [seam('terrain', 8, f.rivCell[fjordNode]), seam('terrain', 10, f.rivCell[fjordNode])] : [99];
  check(Math.max(...seams) <= 1, `fjords tile seamlessly (max channel difference ${seams.join(', ')} at levels 8 and 10)`);
  // Ribbon lakes in troughs inland of the fjords: water a river runs through, never drawn over it.
  const isLake = (r: number, g: number, b: number): boolean => {
    const k = (156 - b) / 36;
    return k >= -0.03 && k <= 1.03 && Math.abs(r - (70 - 30 * k)) <= 2 && Math.abs(g - (124 - 42 * k)) <= 2;
  };
  const inland: number[] = [];
  for (let k = 0; k < f.rivCell.length; k++) {
    const c = f.rivCell[k];
    if (f.rivGlac[k] >= 300 && f.rivFjord[k] === 0 && f.rivQ[k] >= 40 && w.ice.thick[c] <= 0 && w.hydro.lake[c] < 0) inland.push(k);
  }
  inland.sort((a, b) => f.rivGlac[b] - f.rivGlac[a]);
  let lakePx = 0;
  let riverPx = 0;
  let overLake = 0;
  let windows = 0;
  const Lt = 9;
  const S = 384;
  for (let i = 0; i < inland.length && windows < 6; i += Math.max(1, Math.floor(inland.length / 6))) {
    const c = f.rivCell[inland[i]];
    const cx = Math.round(((fine.lon[c] + Math.PI) / (2 * Math.PI)) * levelWidth(Lt));
    const cy = Math.round(((Math.PI / 2 - fine.lat[c]) / Math.PI) * levelHeight(Lt));
    const win = { level: Lt, x0: cx - S / 2, y0: Math.max(0, cy - S / 2), w: S, h: S };
    const on = renderWindow(rc, { layer: 'terrain', month: -1, rivers: true }, win).rgba;
    const off = renderWindow(rc, { layer: 'terrain', month: -1, rivers: false }, win).rgba;
    windows++;
    for (let p = 0; p < S * S; p++) {
      const o = 4 * p;
      const differs = on[o] !== off[o] || on[o + 1] !== off[o + 1] || on[o + 2] !== off[o + 2];
      if (differs) riverPx++;
      if (isLake(off[o], off[o + 1], off[o + 2])) {
        lakePx++;
        if (differs) overLake++;
      }
    }
  }
  check(
    windows > 0 && lakePx > 200 && riverPx > 200 && overLake === 0,
    `trough lakes: ${lakePx} px of lake water around ${windows} glacial troughs at level ${Lt}, their rivers drawn on ${riverPx} px and on ${overLake} px of the water`,
  );
  // Strandflats: only on glaciated hard-rock coasts, never on deltas or soft rock; islets off them.
  let strandLand = 0;
  let wrongCoast = 0;
  for (let c = 0; c < fine.n; c++) {
    if (w.ocean[c] || f.strand[c] <= 0) continue;
    strandLand++;
    const d = w.deposits.kind[c];
    const r = w.lithology.rock[c];
    if (d === Deposit.Delta || d === Deposit.Alluvium || r === Rock.Sandstone || r === Rock.Shale || (w.iceAge.erosion[c] <= 0 && w.iceAge.scour[c] < 0.4)) wrongCoast++;
  }
  /** Small islands (land components under 400 px) in a window centered off the coast at ocean cell c. */
  const islets = (c: number, L: number): number => {
    const W = 384;
    const cx = Math.round(((fine.lon[c] + Math.PI) / (2 * Math.PI)) * levelWidth(L));
    const cy = Math.round(((Math.PI / 2 - fine.lat[c]) / Math.PI) * levelHeight(L));
    const { elevation } = renderWindow(rc, { layer: 'elevation', month: -1, rivers: false }, { level: L, x0: cx - W / 2, y0: Math.max(0, cy - W / 2), w: W, h: W }, true);
    const E = elevation!;
    const seen = new Uint8Array(W * W);
    let n = 0;
    const stack: number[] = [];
    for (let p = 0; p < W * W; p++) {
      if (seen[p] || E[p] < 0.5) continue;
      let size = 0;
      let edge = false;
      stack.push(p);
      seen[p] = 1;
      while (stack.length) {
        const q = stack.pop()!;
        size++;
        const x = q % W;
        const y = (q - x) / W;
        if (x === 0 || y === 0 || x === W - 1 || y === W - 1) edge = true;
        for (const r of [q - 1, q + 1, q - W, q + W]) {
          if (r < 0 || r >= W * W || seen[r] || E[r] < 0.5 || (r === q - 1 && x === 0) || (r === q + 1 && x === W - 1)) continue;
          seen[r] = 1;
          stack.push(r);
        }
      }
      if (size < 400 && !edge) n++;
    }
    return n;
  };
  let flatSea = -1;
  let plainSea = -1;
  for (let c = 0; c < fine.n; c++) {
    if (!w.ocean[c] || w.elev[c] < -150) continue;
    let landNb = -1;
    for (let k = fine.nbrStart[c]; k < fine.nbrStart[c + 1]; k++) if (!w.ocean[fine.nbr[k]]) landNb = fine.nbr[k];
    if (landNb < 0) continue;
    if (f.strand[c] > 0.8 && (flatSea < 0 || f.strand[c] > f.strand[flatSea])) flatSea = c;
    if (f.strand[c] === 0 && f.fjord[landNb] === 0 && w.deposits.kind[landNb] === Deposit.Delta && plainSea < 0) plainSea = c;
  }
  const isl = flatSea >= 0 ? islets(flatSea, 9) : 0;
  const islPlain = plainSea >= 0 ? islets(plainSea, 9) : 0;
  check(
    strandLand > 20 && wrongCoast === 0 && isl >= 20 && isl > 4 * islPlain,
    `strandflats on ${strandLand} glaciated hard-rock coast cells (${wrongCoast} on deltas, soft rock or unglaciated ground): ${isl} islets off one at level 9, ${islPlain} off a delta coast`,
  );
}

if (failures) {
  console.error(`${failures} check(s) failed`);
  process.exit(1);
}
console.log('all checks passed');
