/**
 * Headless world generation + layer rendering for quick inspection.
 *   npx tsx scripts/render.ts [seed] [outDir] [layers...]
 * env: CELLS, W (overview width, default 2048), MONTHS=0,6,
 *      ZOOM="lat,lon,level[,w,h];…" renders close-ups centered on each point.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';
import { encodePNG } from '../src/io/png';
import { buildFields, meshData } from '../src/render/context';
import { type LayerId, levelForWidth, levelHeight, levelWidth, renderWindow } from '../src/render/raster';
import { MONTHS } from '../src/world/climate';
import { DEFAULT_PARAMS } from '../src/world/types';
import { World } from '../src/world/world';

const seed = process.argv[2] ?? DEFAULT_PARAMS.seed;
const outDir = process.argv[3] ?? 'out';
const layers = (process.argv.slice(4).length ? process.argv.slice(4) : ['terrain', 'elevation', 'temperature', 'precipitation', 'biomes', 'koppen', 'plates']) as LayerId[];
mkdirSync(outDir, { recursive: true });

const t0 = performance.now();
const world = World.generate({ ...DEFAULT_PARAMS, seed, cells: Number(process.env.CELLS ?? DEFAULT_PARAMS.cells) });
console.log(`generate: ${(performance.now() - t0).toFixed(0)} ms`, Object.fromEntries(Object.entries(world.timings).map(([k, v]) => [k, Math.round(v)])));
console.log(`plates ${world.geo.plates.length}, land ${(world.landFraction() * 100).toFixed(1)}%`);
for (const l of world.historyLog) console.log('  ' + l);

// Climate sanity numbers.
{
  const { fine, temp, precipYear, ocean } = world;
  let tSum = 0;
  let pSum = 0;
  let aSum = 0;
  let pLand = 0;
  let aLand = 0;
  const bands = new Map<number, { t: number; p: number; a: number }>();
  for (let c = 0; c < fine.n; c++) {
    let t = 0;
    for (let m = 0; m < MONTHS; m++) t += temp[c * MONTHS + m] / MONTHS;
    const a = fine.area[c];
    tSum += t * a;
    pSum += precipYear[c] * a;
    aSum += a;
    if (!ocean[c]) {
      pLand += precipYear[c] * a;
      aLand += a;
    }
    const b = Math.round((fine.lat[c] * 180) / Math.PI / 15) * 15;
    const e = bands.get(b) ?? { t: 0, p: 0, a: 0 };
    e.t += t * a;
    e.p += precipYear[c] * a;
    e.a += a;
    bands.set(b, e);
  }
  console.log(`global T ${(tSum / aSum).toFixed(1)}°C, P ${(pSum / aSum).toFixed(0)} mm/yr (land ${(pLand / aLand).toFixed(0)})`);
  console.log([...bands.entries()].sort((a, b) => b[0] - a[0]).map(([b, e]) => `${b}°: ${(e.t / e.a).toFixed(1)}°C ${(e.p / e.a).toFixed(0)}mm`).join(' | '));

  // Hydrology, with Earth for comparison.
  const h = world.hydro;
  const wb = world.water;
  const R = world.params.radiusKm;
  const share = (pred: (c: number) => boolean) => {
    let a = 0;
    for (let c = 0; c < fine.n; c++) if (!ocean[c] && pred(c)) a += fine.area[c];
    return `${((100 * a) / ((aLand / 1) || 1)).toFixed(1)}%`;
  };
  let aet = 0;
  let ro = 0;
  for (let c = 0; c < fine.n; c++) {
    if (ocean[c]) continue;
    aet += wb.aetYear[c] * fine.area[c];
    ro += wb.runoffYear[c] * fine.area[c];
  }
  console.log(`water balance on land: P ${(pLand / aLand).toFixed(0)} = ET ${(aet / aLand).toFixed(0)} + runoff ${(ro / aLand).toFixed(0)} mm (Earth ≈ 800 = 480 + 320)`);
  const kinds = [0, 0, 0];
  const big: string[] = [];
  for (const l of [...h.lakes].sort((a, b) => b.areaKm2 - a.areaKm2)) {
    kinds[l.kind]++;
    if (big.length < 6) big.push(`${['fresh', 'salt', 'flat'][l.kind]} ${Math.round(l.areaKm2 / 1000)}k km²`);
  }
  console.log(`lakes: ${kinds[0]} fresh, ${kinds[1]} salt, ${kinds[2]} salt flats; lake area ${share((c) => h.lake[c] >= 0 && h.lakes[h.lake[c]].kind !== 2)} of land (Earth ≈ 2–3%); largest: ${big.join(', ')}`);
  console.log(`endorheic ${share((c) => h.endorheic[c] === 1)} of land (Earth ≈ 18%), wetlands ${share((c) => h.wetland[c] > 0)} (Earth ≈ 6–9%), glaciers ${share((c) => wb.glacier[c] === 1)}`);
  let rivers = 0;
  let seasonal = 0;
  let ephemeral = 0;
  let streams = 0;
  let streamsDry = 0;
  for (let c = 0; c < fine.n; c++) {
    if (ocean[c] || h.lake[c] >= 0) continue;
    if (h.discharge[c] >= 5) {
      streams++;
      if (h.flowMonths[c] <= 10) streamsDry++;
    }
    if (h.discharge[c] < world.params.riverThreshold) continue;
    rivers++;
    if (h.flowMonths[c] <= 3) ephemeral++;
    else if (h.flowMonths[c] <= 10) seasonal++;
  }
  console.log(`streams ≥ 5 m³/s: ${streams} cells, ${((100 * streamsDry) / streams).toFixed(0)}% run dry part of the year`);
  const mouths: number[] = [];
  let toSea = 0;
  let delta = 0;
  for (let c = 0; c < fine.n; c++) {
    const rc = h.routing.receiver[c];
    if (!ocean[c] && rc >= 0 && ocean[rc]) {
      mouths.push(h.discharge[c]);
      toSea += h.discharge[c];
    }
    if (!ocean[c] && world.sediment[c] > 1 && world.elev[c] < 15 && world.relief.elev[c] < 0) delta++;
  }
  mouths.sort((a, b) => b - a);
  console.log(`rivers: ${rivers} cells, ${((100 * seasonal) / rivers).toFixed(0)}% seasonal, ${((100 * ephemeral) / rivers).toFixed(0)}% ephemeral; to the sea ${(toSea / 1000).toFixed(0)}k m³/s (Earth ≈ 1250k); largest ${mouths.slice(0, 6).map((q) => `${(q / 1000).toFixed(0)}k`).join(', ')}`);
  console.log(`delta land ${delta} cells (${Math.round(delta * (4 * Math.PI * R * R) / fine.n / 1000)}k km²)`);
  // Ice, now and in the last ice age (Earth: ~10% of land under ice today, ~25% at the glacial maximum).
  {
    const ice = world.ice;
    let iceArea = 0;
    let vol = 0;
    let maxH = 0;
    let pastArea = 0;
    let eroded = 0;
    let erodedArea = 0;
    let shelfCells = 0;
    let bodies = 0;
    const seen = new Set<number>();
    for (let c = 0; c < fine.n; c++) {
      const a = fine.area[c] * R * R;
      if (ice.thick[c] > 0) {
        iceArea += a;
        vol += a * ice.thick[c] / 1000;
        maxH = Math.max(maxH, ice.thick[c]);
        if (!seen.has(ice.body[c])) seen.add(ice.body[c]);
      }
      if (ice.shelf[c]) shelfCells++;
      if (world.iceAge.thick[c] > 0) pastArea += a;
      if (world.iceAge.erosion[c] > 0) {
        eroded += world.iceAge.erosion[c] * a;
        erodedArea += a;
      }
    }
    for (let c = 0; c < fine.n; c++) if (ice.thick[c] > 0 && ice.body[c] >= 3) bodies++;
    const landKm2 = aLand * R * R;
    console.log(`ice today ${((100 * iceArea) / landKm2).toFixed(1)}% of land, ${Math.round(vol / 1e3)}k km³ (sea-level equivalent ${(vol * 0.917 / 361e6 * 1000).toFixed(1)} m), thickest ${Math.round(maxH)} m, ${shelfCells} ice-shelf cells, calving ${(ice.calving / 1e9).toFixed(0)} km³/yr`);
    console.log(`ice age: ice on ${((100 * pastArea) / landKm2).toFixed(1)}% of today's land area; glaciers deepened ${Math.round(erodedArea / 1000)}k km² by ${erodedArea ? Math.round(eroded / erodedArea) : 0} m on average`);
    void bodies;
  }
  const sb = world.sedimentBudget;
  const pct = (v: number) => `${((100 * v) / (sb.eroded || 1)).toFixed(0)}%`;
  console.log(`sediment ${(sb.eroded / 1e6).toFixed(2)}M km³: basins ${pct(sb.basins)}, floodplains ${pct(sb.floodplains)}, shelves ${pct(sb.shelves)}, deltas ${pct(sb.deltas)}, to the deep sea ${pct(sb.lost)}`);
  for (const l of [...h.lakes].sort((a, b) => b.areaKm2 - a.areaKm2).slice(0, 4)) {
    const o = l.outlet;
    console.log(`  ${['fresh', 'salt', 'flat'][l.kind]} lake ${Math.round(l.areaKm2 / 1000)}k km² at ${((fine.lat[o] * 180) / Math.PI).toFixed(0)}°, ${((fine.lon[o] * 180) / Math.PI).toFixed(0)}°: inflow ${Math.round(l.inflow)} m³/s, PET ${Math.round(wb.petYear[o])} mm, P ${Math.round(world.precipYear[o])} mm`);
  }
}

const t1 = performance.now();
const ctx = { mesh: meshData(world.fine, 'headless'), fields: buildFields(world, 'headless', 1) };
console.log(`render context: ${(performance.now() - t1).toFixed(0)} ms, ${ctx.fields.rivStart.length - 1} river chains`);

async function save(name: string, rgba: Uint8ClampedArray, w: number, h: number): Promise<void> {
  const rgb = new Uint8Array(w * h * 3);
  for (let i = 0; i < w * h; i++) {
    rgb[3 * i] = rgba[4 * i];
    rgb[3 * i + 1] = rgba[4 * i + 1];
    rgb[3 * i + 2] = rgba[4 * i + 2];
  }
  writeFileSync(`${outDir}/${name}.png`, await encodePNG(w, h, rgb, { channels: 3, bitDepth: 8 }, (d) => deflateSync(d)));
}

const W = Number(process.env.W ?? 2048);
const level = levelForWidth(W);
const months = process.env.MONTHS ? process.env.MONTHS.split(',').map(Number) : [-1];
for (const layer of layers) {
  for (const month of months) {
    const t2 = performance.now();
    const { rgba } = renderWindow(ctx, { layer, month, rivers: true }, { level, x0: 0, y0: 0, w: levelWidth(level), h: levelHeight(level) });
    const name = month >= 0 ? `${layer}-m${month + 1}` : layer;
    console.log(`${name}: ${(performance.now() - t2).toFixed(0)} ms`);
    await save(name, rgba, levelWidth(level), levelHeight(level));
  }
}

for (const spec of (process.env.ZOOM ?? '').split(';').filter(Boolean)) {
  const [lat, lon, L, w = 1024, h = 640] = spec.split(',').map(Number);
  const cx = Math.round(((lon + 180) / 360) * levelWidth(L));
  const cy = Math.round(((90 - lat) / 180) * levelHeight(L));
  const win = { level: L, x0: cx - Math.round(w / 2), y0: Math.max(0, cy - Math.round(h / 2)), w, h };
  for (const layer of layers) {
    const t3 = performance.now();
    const { rgba } = renderWindow(ctx, { layer, month: -1, rivers: true }, win);
    const ms = performance.now() - t3;
    console.log(`zoom ${lat},${lon} L${L} ${layer}: ${ms.toFixed(0)} ms (${((ms * 65536) / (w * h)).toFixed(1)} ms per 256² tile)`);
    await save(`zoom-${lat}_${lon}-L${L}-${layer}`, rgba, w, h);
  }
}
