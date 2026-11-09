/**
 * Smoke test: generation, every edit operation, undo/redo, save/load
 * round-trip and the 16-bit heightmap export.
 *   npx tsx scripts/selftest.ts [outDir]
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';
import { encodePNG } from '../src/io/png';
import { deserializeWorld, serializeWorld } from '../src/io/save';
import { buildFields, meshData } from '../src/render/context';
import { type LayerId, LAYERS, levelHeight, levelWidth, MAX_LEVEL, renderWindow, riverCourse, TILE } from '../src/render/raster';
import { DEG, latLonToVec } from '../src/core/math';
import * as edits from '../src/world/edits';
import { Rock, ROCK_INFO } from '../src/world/geology';
import { ICE_ISOSTASY } from '../src/world/glaciers';
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
  for (const w of [world, ...['aurelia', 'vale-fen-46'].map((seed) => World.generate({ ...DEFAULT_PARAMS, seed, cells: 40000 }))]) {
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
  const worlds = [80000, 320000].map((cells) => World.generate({ ...DEFAULT_PARAMS, seed: 'selftest', cells }));
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

if (failures) {
  console.error(`${failures} check(s) failed`);
  process.exit(1);
}
console.log('all checks passed');
