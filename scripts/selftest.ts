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
}

if (failures) {
  console.error(`${failures} check(s) failed`);
  process.exit(1);
}
console.log('all checks passed');
