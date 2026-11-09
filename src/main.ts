import { BASE_LEVEL, LAYERS, type LayerId, levelHeight, levelWidth, type RenderOptions, TILE } from './render/raster';
import { inspectorHtml, legendHtml, reportHtml } from './ui/panels';
import { baseTiles, type TileKey, TileManager } from './ui/tiles';
import { GlobeView, type LatLonDeg, MapView, type View } from './ui/views';
import type { EditReport, IslandCause } from './world/edits';
import { DEFAULT_PARAMS, type Layout, type WorldParams } from './world/types';
import type { BrushTool, Request, Response, TileWorkerRequest, WorldSummary } from './worker/protocol';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const D = Math.PI / 180;

const worker = new Worker(new URL('./worker/worker.ts', import.meta.url), { type: 'module' });
const send = (msg: Request, transfer: Transferable[] = []) => worker.postMessage(msg, transfer);

// Tile workers render the map in parallel; the simulation worker feeds them
// the world over direct channels after every change.
const tileWorkers: Worker[] = [];
const tilePorts: MessagePort[] = [];
for (let i = 0; i < Math.max(1, Math.min(6, (navigator.hardwareConcurrency || 4) - 2)); i++) {
  const tw = new Worker(new URL('./worker/tiles.ts', import.meta.url), { type: 'module' });
  const ch = new MessageChannel();
  tw.postMessage({ type: 'port', port: ch.port2 } satisfies TileWorkerRequest, [ch.port2]);
  tilePorts.push(ch.port1);
  tileWorkers.push(tw);
}
send({ type: 'init', ports: tilePorts }, tilePorts);
const tiles = new TileManager(tileWorkers);

type Tool = 'inspect' | BrushTool | 'island' | 'hotspot' | 'plate';
const BRUSHES: Tool[] = ['land', 'sea', 'deepsea', 'raise', 'raiseOld', 'lower'];
const TOOL_HINTS: Record<Tool, string> = {
  inspect: 'Drag to pan (map) or rotate (globe); scroll to zoom. Hover for details.',
  land: 'Paint land: the nearest continent’s crust extends and its plate grows to include it. Right-drag pans.',
  sea: 'Flood land: continental crust thins and subsides into a shallow sea.',
  deepsea: 'Rift: continental crust is replaced by young ocean floor.',
  raise: 'Thicken crust into young, rugged mountains (attributed to nearby boundaries if any).',
  raiseOld: 'Thicken crust into old, eroded highlands (an ancient orogen).',
  lower: 'Thin the crust; land subsides, seafloor deepens.',
  island: 'Click in the sea to raise an island. Its cause is chosen below or picked as most feasible.',
  hotspot: 'Click to place a mantle plume; its past track is reconstructed from plate motion.',
  plate: 'Drag from a point on a plate in the direction it should move; arrow length sets speed.',
};
const WORLD_RADIUS = () => state.summary?.params.radiusKm ?? DEFAULT_PARAMS.radiusKm;

const state = {
  summary: null as WorldSummary | null,
  /** World version shown (bumped by the simulation worker on every change). */
  version: -1,
  layer: 'terrain' as LayerId,
  month: -1,
  tool: 'inspect' as Tool,
  hover: null as LatLonDeg | null,
  mouse: null as { x: number; y: number } | null,
  generating: false,
};

// ------------------------------------------------------------------ views

const mapView = new MapView($<HTMLCanvasElement>('mapCanvas'));
let globeView: GlobeView | null = null;
try {
  globeView = new GlobeView($<HTMLCanvasElement>('globeCanvas'));
} catch (err) {
  console.warn('Globe disabled:', err);
  document.querySelector<HTMLButtonElement>('[data-view="globe"]')!.disabled = true;
}
let view: View = mapView;
globeView?.show(false);

// The overview image, assembled from level-2 tiles (and brush previews).
const baseCanvas = document.createElement('canvas');
baseCanvas.width = levelWidth(BASE_LEVEL);
baseCanvas.height = levelHeight(BASE_LEVEL);
const baseCtx = baseCanvas.getContext('2d', { willReadFrequently: true })!;
baseCtx.fillStyle = '#0b0f17';
baseCtx.fillRect(0, 0, baseCanvas.width, baseCanvas.height);
mapView.base = baseCanvas;
mapView.tiles = tiles;
if (globeView) globeView.tiles = tiles;

const overlay = $<HTMLCanvasElement>('overlay');
const octx = overlay.getContext('2d')!;
let overlaySize = { w: 1, h: 1, dpr: 1 };

function resize(): void {
  const vp = $('viewport');
  const w = vp.clientWidth;
  const h = vp.clientHeight;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  mapView.resize(w, h, dpr);
  globeView?.resize(w, h, dpr);
  overlay.width = Math.round(w * dpr);
  overlay.height = Math.round(h * dpr);
  overlaySize = { w, h, dpr };
  redraw();
}
new ResizeObserver(resize).observe($('viewport'));

let drawPending = false;
function redraw(): void {
  if (drawPending) return;
  drawPending = true;
  const run = () => {
    if (!drawPending) return;
    drawPending = false;
    requestTiles();
    view.draw();
    drawOverlay();
  };
  // rAF can be dropped while the page is hidden; the timer guarantees a draw.
  requestAnimationFrame(run);
  setTimeout(run, 80);
}

function setView(kind: 'map' | 'globe'): void {
  if (kind === 'globe' && !globeView) return;
  view = kind === 'globe' ? globeView! : mapView;
  mapView.show(kind === 'map');
  globeView?.show(kind === 'globe');
  document.querySelectorAll<HTMLButtonElement>('#viewToggle button').forEach((b) => b.classList.toggle('on', b.dataset.view === kind));
  redraw();
}

function setDetail(on: boolean): void {
  mapView.showDetail = on;
  if (globeView) globeView.showDetail = on;
}
document.querySelectorAll<HTMLButtonElement>('#viewToggle button').forEach((b) => (b.onclick = () => setView(b.dataset.view as 'map' | 'globe')));

// ------------------------------------------------------------------ overlay

function arrowTo(x0: number, y0: number, x1: number, y1: number, color: string, width = 1.6): void {
  const a = Math.atan2(y1 - y0, x1 - x0);
  const len = Math.hypot(x1 - x0, y1 - y0);
  const head = Math.min(7, len * 0.45);
  octx.strokeStyle = color;
  octx.fillStyle = color;
  octx.lineWidth = width;
  octx.beginPath();
  octx.moveTo(x0, y0);
  octx.lineTo(x1, y1);
  octx.stroke();
  octx.beginPath();
  octx.moveTo(x1, y1);
  octx.lineTo(x1 - head * Math.cos(a - 0.45), y1 - head * Math.sin(a - 0.45));
  octx.lineTo(x1 - head * Math.cos(a + 0.45), y1 - head * Math.sin(a + 0.45));
  octx.closePath();
  octx.fill();
}

/** Screen direction of a ground vector (east/north components) at a point. */
function screenDir(lat: number, lon: number, vE: number, vN: number): { x: number; y: number; ok: boolean } {
  const s = 0.4 / Math.max(1e-6, Math.hypot(vE, vN));
  const p0 = view.project(lat, lon);
  const p1 = view.project(lat + vN * s, lon + (vE * s) / Math.max(0.05, Math.cos(lat * D)));
  const dx = p1.x - p0.x;
  const dy = p1.y - p0.y;
  const l = Math.hypot(dx, dy);
  return { x: dx / (l || 1), y: dy / (l || 1), ok: p0.visible && l > 1e-6 };
}

let plateDrag: { start: LatLonDeg; sx: number; sy: number; x: number; y: number } | null = null;

function drawOverlay(): void {
  const { w, h, dpr } = overlaySize;
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  octx.clearRect(0, 0, w, h);
  const s = state.summary;
  if (!s) return;
  const showArrows = $<HTMLInputElement>('arrows').checked || state.tool === 'plate';
  if (showArrows) {
    for (const a of s.arrows) {
      const p = view.project(a.lat, a.lon);
      if (!p.visible || p.x < -20 || p.y < -20 || p.x > w + 20 || p.y > h + 20) continue;
      const speed = Math.hypot(a.vE, a.vN);
      if (speed < 0.05) continue;
      const d = screenDir(a.lat, a.lon, a.vE, a.vN);
      if (!d.ok) continue;
      const len = 4 + speed * 4.5;
      octx.globalAlpha = 0.9;
      arrowTo(p.x, p.y, p.x + d.x * len, p.y + d.y * len, '#ffffffd0');
      octx.globalAlpha = 1;
    }
  }
  if ($<HTMLInputElement>('hotspots').checked || state.tool === 'hotspot') {
    for (const hs of s.hotspots) {
      const p = view.project(hs.lat, hs.lon);
      if (!p.visible) continue;
      octx.beginPath();
      octx.arc(p.x, p.y, 5, 0, Math.PI * 2);
      octx.fillStyle = hs.active ? '#ff6a2a' : '#888';
      octx.strokeStyle = '#000a';
      octx.lineWidth = 1.5;
      octx.fill();
      octx.stroke();
      octx.fillStyle = '#ffd7b8';
      octx.font = '11px system-ui';
      octx.fillText(hs.name, p.x + 7, p.y - 6);
    }
  }
  if (state.mouse && state.hover && BRUSHES.includes(state.tool)) {
    const r = Number($<HTMLInputElement>('brushRadius').value) * view.pxPerKm(WORLD_RADIUS());
    octx.beginPath();
    octx.arc(state.mouse.x, state.mouse.y, Math.max(2, r), 0, Math.PI * 2);
    octx.strokeStyle = '#ffffffcc';
    octx.lineWidth = 1.2;
    octx.setLineDash([5, 4]);
    octx.stroke();
    octx.setLineDash([]);
  }
  if (plateDrag) {
    const len = Math.hypot(plateDrag.x - plateDrag.sx, plateDrag.y - plateDrag.sy);
    arrowTo(plateDrag.sx, plateDrag.sy, plateDrag.x, plateDrag.y, '#ffb35c', 3);
    octx.fillStyle = '#ffb35c';
    octx.font = '12px system-ui';
    octx.fillText(`${dragSpeed(len).toFixed(1)} cm/yr`, plateDrag.x + 8, plateDrag.y + 4);
  }
}

// ------------------------------------------------------------------ rendering

function renderOpts(): RenderOptions {
  return { layer: state.layer, month: state.month, rivers: $<HTMLInputElement>('rivers').checked };
}

let baseRendering = false;

/** The world or the layer changed: re-render the overview, then the detail in view. */
function refreshTiles(): void {
  if (state.version < 0) return;
  tiles.reset(state.version, renderOpts());
  globeView?.resetDetail();
  baseRendering = true;
  setBusy('Rendering…', 0);
  redraw();
}

/** Overview tiles on screen first, then the detail the view wants, then the rest of the overview. */
function requestTiles(): void {
  if (state.version < 0) return;
  const front: (TileKey & { d: number })[] = [];
  const back: TileKey[] = [];
  const vp = $('viewport');
  for (const t of baseTiles()) {
    const p = view.baseTileCenter(t);
    const onScreen = p.visible && p.x > -200 && p.y > -200 && p.x < vp.clientWidth + 200 && p.y < vp.clientHeight + 200;
    if (onScreen) front.push({ ...t, d: Math.hypot(p.x - vp.clientWidth / 2, p.y - vp.clientHeight / 2) });
    else back.push(t);
  }
  front.sort((a, b) => a.d - b.d);
  tiles.want([...front, ...view.wantedTiles(), ...back]);
}

tiles.onTile = (t, bmp) => {
  if (t.level === BASE_LEVEL) {
    baseCtx.drawImage(bmp, t.x * TILE, t.y * TILE);
    globeView?.putBase(t.x * TILE, t.y * TILE, bmp);
    if (baseRendering) {
      if (tiles.baseProgress >= 1) {
        baseRendering = false;
        setBusy(null);
      } else setBusy('Rendering…', tiles.baseProgress);
    }
  } else globeView?.putDetail(t, bmp);
  redraw();
};

/** A brush preview: paste it into the overview (wrapping around the date line). */
function applyPatch(x0: number, y0: number, w: number, h: number, bitmap: ImageBitmap): void {
  const W = baseCanvas.width;
  for (const shift of [0, -W, W]) {
    const x = x0 + shift;
    if (x + w <= 0 || x >= W) continue;
    baseCtx.drawImage(bitmap, x, y0);
    if (globeView) {
      const sx = Math.max(0, x);
      const ex = Math.min(W, x + w);
      globeView.putBase(sx, y0, baseCtx.getImageData(sx, y0, ex - sx, h));
    }
  }
  bitmap.close();
}

// ------------------------------------------------------------------ busy / log

const busy = $('busy');
function setBusy(label: string | null, frac = 0): void {
  if (label === null) {
    busy.hidden = true;
    return;
  }
  busy.hidden = false;
  busy.querySelector<HTMLElement>('.label')!.textContent = label;
  busy.querySelector<HTMLElement>('.fill')!.style.width = `${Math.round(frac * 100)}%`;
}

function logReport(r: EditReport): void {
  const log = $('log');
  if (log.querySelector('.muted') && !log.querySelector('.report')) log.innerHTML = '';
  log.insertAdjacentHTML('afterbegin', reportHtml(r));
  while (log.children.length > 30) log.lastElementChild!.remove();
}

function logMessage(title: string, text: string, failed = false): void {
  logReport({ title, summary: text, details: [], derive: 'tectonics', failed });
}

// ------------------------------------------------------------------ parameters

const GEN_INPUTS: [keyof WorldParams, string][] = [
  ['plates', 'plates'],
  ['continentalFraction', 'continentalFraction'],
  ['historyMyr', 'historyMyr'],
  ['hotspots', 'hotspotCount'],
];
const PLANET_INPUTS: (keyof WorldParams)[] = ['seaLevel', 'axialTiltDeg', 'temperatureOffset', 'erosion', 'riverThreshold'];

function fmtParam(key: string, v: number): string {
  if (key === 'continentalFraction') return `${Math.round(v * 100)}%`;
  if (key === 'axialTiltDeg') return `${v}°`;
  if (key === 'temperatureOffset') return `${v > 0 ? '+' : ''}${v}`;
  if (key === 'brushStrength') return v.toFixed(2);
  return String(v);
}

function bindOutput(input: HTMLInputElement, key: string): void {
  const out = document.querySelector<HTMLOutputElement>(`output[for="${input.id}"]`);
  const upd = () => out && (out.textContent = fmtParam(key, Number(input.value)));
  input.addEventListener('input', upd);
  upd();
}

function setInputs(p: WorldParams): void {
  $<HTMLInputElement>('seed').value = p.seed;
  $<HTMLSelectElement>('cells').value = String(p.cells);
  $<HTMLSelectElement>('layout').value = p.layout;
  for (const [k, id] of GEN_INPUTS) $<HTMLInputElement>(id).value = String(p[k]);
  for (const k of PLANET_INPUTS) $<HTMLInputElement>(k).value = String(p[k]);
  $<HTMLSelectElement>('retrograde').value = String(p.retrograde);
  document.querySelectorAll<HTMLInputElement>('input[type=range]').forEach((i) => i.dispatchEvent(new Event('input')));
}

function readParams(): WorldParams {
  const p: WorldParams = { ...(state.summary?.params ?? DEFAULT_PARAMS) };
  p.seed = $<HTMLInputElement>('seed').value.trim() || 'world';
  p.cells = Number($<HTMLSelectElement>('cells').value);
  p.layout = $<HTMLSelectElement>('layout').value as Layout;
  for (const [k, id] of GEN_INPUTS) (p as unknown as Record<string, number>)[k] = Number($<HTMLInputElement>(id).value);
  for (const k of PLANET_INPUTS) (p as unknown as Record<string, number>)[k] = Number($<HTMLInputElement>(k).value);
  p.retrograde = $<HTMLSelectElement>('retrograde').value === 'true';
  return p;
}

for (const [k, id] of GEN_INPUTS) bindOutput($<HTMLInputElement>(id), k);
for (const k of PLANET_INPUTS) {
  const input = $<HTMLInputElement>(k);
  bindOutput(input, k);
  input.addEventListener('change', () => {
    if (!state.summary) return;
    send({ type: 'setParams', params: { [k]: Number(input.value) } });
  });
}
$<HTMLSelectElement>('retrograde').addEventListener('change', () => {
  if (state.summary) send({ type: 'setParams', params: { retrograde: $<HTMLSelectElement>('retrograde').value === 'true' } });
});
bindOutput($<HTMLInputElement>('brushRadius'), 'brushRadius');
bindOutput($<HTMLInputElement>('brushStrength'), 'brushStrength');
bindOutput($<HTMLInputElement>('islandSize'), 'islandSize');
bindOutput($<HTMLInputElement>('hotspotStrength'), 'brushStrength');

$('fitSea').onclick = () => {
  const pct = Number($<HTMLInputElement>('landPct').value);
  if (state.summary && pct > 0 && pct < 100) send({ type: 'fitSeaLevel', landFraction: pct / 100 });
};

const SEED_WORDS = ['ash', 'brine', 'cairn', 'dusk', 'ember', 'fen', 'gale', 'heath', 'isle', 'jade', 'kelp', 'loam', 'mire', 'nadir', 'opal', 'peak', 'quill', 'reef', 'shale', 'tor', 'umber', 'vale', 'weald', 'yew'];
function randomSeed(): string {
  const pick = () => SEED_WORDS[Math.floor(Math.random() * SEED_WORDS.length)];
  return `${pick()}-${pick()}-${Math.floor(Math.random() * 1000)}`;
}
$('dice').onclick = () => {
  $<HTMLInputElement>('seed').value = randomSeed();
};

let toastTimer = 0;
function toast(text: string): void {
  const t = $('toast');
  t.textContent = text;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => (t.hidden = true), 5000);
}

/** Settings that change the generated world (the planet settings apply live). */
const WORLD_KEYS: (keyof WorldParams)[] = ['seed', 'cells', 'layout', 'plates', 'continentalFraction', 'historyMyr', 'hotspots'];

function generate(newSeed = false): void {
  const cur = state.summary;
  if (cur?.canUndo && !confirm('Generate a new world? Unsaved edits to the current world will be lost.')) return;
  if (newSeed) $<HTMLInputElement>('seed').value = randomSeed();
  const params = readParams();
  if (cur && !cur.canUndo && WORLD_KEYS.every((k) => params[k] === cur.params[k])) {
    toast('The same seed and settings always make the same world. Change the seed (🎲) or press New random world.');
    return;
  }
  state.generating = true;
  setBusy('Generating…', 0);
  send({ type: 'generate', params });
}
$('generate').onclick = () => generate();
$('newWorld').onclick = () => generate(true);

// ------------------------------------------------------------------ layers & toggles

const layerSel = $<HTMLSelectElement>('layer');
for (const l of LAYERS) layerSel.add(new Option(l.label, l.id));
layerSel.onchange = () => {
  state.layer = layerSel.value as LayerId;
  updateLayerUi();
  refreshTiles();
};
$<HTMLSelectElement>('month').onchange = () => {
  state.month = Number($<HTMLSelectElement>('month').value);
  refreshTiles();
};
$('rivers').onchange = refreshTiles;
$('arrows').onchange = redraw;
$('hotspots').onchange = redraw;

function updateLayerUi(): void {
  const monthly = LAYERS.find((l) => l.id === state.layer)?.monthly ?? false;
  $('monthField').style.opacity = monthly ? '1' : '0.4';
  $<HTMLSelectElement>('month').disabled = !monthly;
  $('legend').innerHTML = legendHtml(state.layer, state.summary);
}

// ------------------------------------------------------------------ tools

function setTool(t: Tool): void {
  state.tool = t;
  document.querySelectorAll<HTMLButtonElement>('#tools button').forEach((b) => b.classList.toggle('on', b.dataset.tool === t));
  $('brushOpts').hidden = !BRUSHES.includes(t);
  $('islandOpts').hidden = t !== 'island';
  $('hotspotOpts').hidden = t !== 'hotspot';
  $('toolHint').textContent = TOOL_HINTS[t];
  overlay.style.cursor = t === 'inspect' ? 'grab' : 'crosshair';
  redraw();
}
document.querySelectorAll<HTMLButtonElement>('#tools button').forEach((b) => (b.onclick = () => setTool(b.dataset.tool as Tool)));

$('advance').onclick = () => {
  const myr = Math.max(1, Math.min(100, Number($<HTMLInputElement>('advanceMyr').value) || 10));
  setBusy(`Advancing ${myr} Myr…`, 0);
  send({ type: 'edit', op: { kind: 'advance', myr } });
};

$('undo').onclick = () => send({ type: 'undo' });
$('redo').onclick = () => send({ type: 'redo' });

// ------------------------------------------------------------------ files

$('save').onclick = () => send({ type: 'save' });
$('load').onclick = () => $<HTMLInputElement>('loadFile').click();
$<HTMLInputElement>('loadFile').onchange = async () => {
  const f = $<HTMLInputElement>('loadFile').files?.[0];
  if (!f) return;
  const buf = await f.arrayBuffer();
  setBusy('Loading world…', 0);
  send({ type: 'load', buffer: buf }, [buf]);
  $<HTMLInputElement>('loadFile').value = '';
};
$('export').onclick = () => {
  $('exportMenu').hidden = !$('exportMenu').hidden;
};
function exportReq(what: 'heightmap16' | 'layer'): void {
  $('exportMenu').hidden = true;
  setBusy('Exporting…', 0);
  send({ type: 'export', what, render: renderOpts(), width: Number($<HTMLSelectElement>('exportWidth').value) });
}
$('exportHeight').onclick = () => exportReq('heightmap16');
$('exportLayer').onclick = () => exportReq('layer');

function download(name: string, mime: string, buffer: ArrayBuffer): void {
  const url = URL.createObjectURL(new Blob([buffer], { type: mime }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

// ------------------------------------------------------------------ pointer interaction

function local(e: PointerEvent | WheelEvent): { x: number; y: number } {
  const r = overlay.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

function gcKm(a: LatLonDeg, b: LatLonDeg): number {
  const s =
    Math.sin(a.lat * D) * Math.sin(b.lat * D) + Math.cos(a.lat * D) * Math.cos(b.lat * D) * Math.cos((b.lon - a.lon) * D);
  return Math.acos(Math.max(-1, Math.min(1, s))) * WORLD_RADIUS();
}

const dragSpeed = (px: number) => Math.max(0.3, Math.min(15, px / 12));

/** Points every `stepKm` along the great circle from a (exclusive) to b (inclusive). */
function interpolatePath(a: LatLonDeg, b: LatLonDeg, stepKm: number): LatLonDeg[] {
  const dist = gcKm(a, b);
  const n = Math.max(1, Math.ceil(dist / stepKm));
  const va = [Math.cos(a.lat * D) * Math.cos(a.lon * D), Math.cos(a.lat * D) * Math.sin(a.lon * D), Math.sin(a.lat * D)];
  const vb = [Math.cos(b.lat * D) * Math.cos(b.lon * D), Math.cos(b.lat * D) * Math.sin(b.lon * D), Math.sin(b.lat * D)];
  const out: LatLonDeg[] = [];
  for (let i = 1; i <= n; i++) {
    const t = i / n;
    const x = va[0] * (1 - t) + vb[0] * t;
    const y = va[1] * (1 - t) + vb[1] * t;
    const z = va[2] * (1 - t) + vb[2] * t;
    const l = Math.hypot(x, y, z) || 1;
    out.push({ lat: Math.asin(z / l) / D, lon: Math.atan2(y, x) / D });
  }
  return out;
}

function extendStroke(to: LatLonDeg): void {
  if (!stroke) return;
  const step = brushParams().radiusKm * 0.3;
  if (gcKm(stroke.last, to) <= step) return;
  stroke.pending.push(...interpolatePath(stroke.last, to, step));
  stroke.last = to;
}

let pan: { x: number; y: number } | null = null;
let stroke: { tool: BrushTool; pending: LatLonDeg[]; last: LatLonDeg; waiting: boolean } | null = null;
let click: { x: number; y: number; ll: LatLonDeg } | null = null;
let inspectTimer = 0;
let inspectId = 0;

function brushParams() {
  return {
    radiusKm: Number($<HTMLInputElement>('brushRadius').value),
    strength: Number($<HTMLInputElement>('brushStrength').value),
  };
}

function flushStroke(phase: 'move' | 'end'): void {
  if (!stroke) return;
  if (phase === 'move' && (stroke.waiting || stroke.pending.length === 0)) return;
  const pts = stroke.pending;
  stroke.pending = [];
  stroke.waiting = phase === 'move';
  send({ type: 'stroke', phase, tool: stroke.tool, points: pts, ...brushParams(), render: renderOpts() });
  if (phase === 'end') {
    stroke = null;
    setBusy('Re-deriving the world…', 0);
  }
}

overlay.addEventListener('contextmenu', (e) => e.preventDefault());

overlay.addEventListener('pointerdown', (e) => {
  overlay.setPointerCapture(e.pointerId);
  const p = local(e);
  const ll = view.screenToLatLon(p.x, p.y);
  if (e.button !== 0 || state.tool === 'inspect' || !ll || !state.summary || state.generating) {
    pan = p;
    overlay.style.cursor = 'grabbing';
    return;
  }
  if (BRUSHES.includes(state.tool)) {
    stroke = { tool: state.tool as BrushTool, pending: [], last: ll, waiting: true };
    // Previews update the overview only; detail tiles return once the edit is derived.
    setDetail(false);
    send({ type: 'stroke', phase: 'start', tool: stroke.tool, points: [ll], ...brushParams(), render: renderOpts() });
  } else if (state.tool === 'plate') {
    plateDrag = { start: ll, sx: p.x, sy: p.y, x: p.x, y: p.y };
  } else {
    click = { x: p.x, y: p.y, ll };
  }
});

overlay.addEventListener('pointermove', (e) => {
  const p = local(e);
  state.mouse = p;
  state.hover = view.screenToLatLon(p.x, p.y);
  if (pan) {
    view.panBy(p.x - pan.x, p.y - pan.y);
    pan = p;
    redraw();
    return;
  }
  if (stroke && state.hover) {
    extendStroke(state.hover);
    flushStroke('move');
  }
  if (plateDrag) {
    plateDrag.x = p.x;
    plateDrag.y = p.y;
  }
  if (!stroke && !plateDrag && state.hover && !inspectTimer) {
    inspectTimer = window.setTimeout(() => {
      inspectTimer = 0;
      if (state.hover) send({ type: 'inspect', id: ++inspectId, lat: state.hover.lat, lon: state.hover.lon });
    }, 70);
  }
  redraw();
});

overlay.addEventListener('pointerup', (e) => {
  const p = local(e);
  if (pan) {
    pan = null;
    overlay.style.cursor = state.tool === 'inspect' ? 'grab' : 'crosshair';
    return;
  }
  if (stroke) {
    const end = view.screenToLatLon(p.x, p.y);
    if (end) extendStroke(end);
    flushStroke('end');
    return;
  }
  if (plateDrag) {
    const d = plateDrag;
    plateDrag = null;
    const end = view.screenToLatLon(p.x, p.y);
    const lenPx = Math.hypot(p.x - d.sx, p.y - d.sy);
    redraw();
    if (!end || lenPx < 8) return;
    // Tangent direction at the start point toward the end point.
    const a = [Math.cos(d.start.lat * D) * Math.cos(d.start.lon * D), Math.cos(d.start.lat * D) * Math.sin(d.start.lon * D), Math.sin(d.start.lat * D)];
    const b = [Math.cos(end.lat * D) * Math.cos(end.lon * D), Math.cos(end.lat * D) * Math.sin(end.lon * D), Math.sin(end.lat * D)];
    const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const t = [b[0] - dot * a[0], b[1] - dot * a[1], b[2] - dot * a[2]];
    const east = [-Math.sin(d.start.lon * D), Math.cos(d.start.lon * D), 0];
    const north = [a[1] * east[2] - a[2] * east[1], a[2] * east[0] - a[0] * east[2], a[0] * east[1] - a[1] * east[0]];
    const tE = t[0] * east[0] + t[1] * east[1] + t[2] * east[2];
    const tN = t[0] * north[0] + t[1] * north[1] + t[2] * north[2];
    const tl = Math.hypot(tE, tN) || 1;
    const speed = dragSpeed(lenPx);
    setBusy('Changing plate motion…', 0);
    send({ type: 'edit', op: { kind: 'plateVelocity', at: d.start, vEast: (tE / tl) * speed, vNorth: (tN / tl) * speed } });
    return;
  }
  if (click && Math.hypot(p.x - click.x, p.y - click.y) < 6) {
    if (state.tool === 'island') {
      setBusy('Explaining the new island…', 0);
      send({
        type: 'edit',
        op: {
          kind: 'island',
          at: click.ll,
          cause: $<HTMLSelectElement>('islandCause').value as IslandCause,
          sizeKm: Number($<HTMLInputElement>('islandSize').value),
        },
      });
    } else if (state.tool === 'hotspot') {
      setBusy('Placing hotspot…', 0);
      send({ type: 'edit', op: { kind: 'hotspot', at: click.ll, strength: Number($<HTMLInputElement>('hotspotStrength').value) } });
    }
  }
  click = null;
});

overlay.addEventListener('pointerleave', () => {
  state.mouse = null;
  redraw();
});

overlay.addEventListener(
  'wheel',
  (e) => {
    e.preventDefault();
    const p = local(e);
    view.zoomAt(p.x, p.y, Math.exp(-e.deltaY * 0.0015));
    redraw();
  },
  { passive: false },
);

window.addEventListener('keydown', (e) => {
  const tag = (e.target as HTMLElement).tagName;
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
    e.preventDefault();
    send({ type: e.shiftKey ? 'redo' : 'undo' });
  } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
    e.preventDefault();
    send({ type: 'redo' });
  } else if (e.key === 'g') {
    setView(view.kind === 'map' ? 'globe' : 'map');
  } else if (e.key === '[' || e.key === ']') {
    const r = $<HTMLInputElement>('brushRadius');
    r.value = String(Number(r.value) * (e.key === ']' ? 1.2 : 1 / 1.2));
    r.dispatchEvent(new Event('input'));
    redraw();
  } else if (/^[0-9]$/.test(e.key)) {
    const tools: Tool[] = ['plate', 'inspect', 'land', 'sea', 'deepsea', 'raise', 'raiseOld', 'lower', 'island', 'hotspot'];
    setTool(tools[Number(e.key)]);
  }
});

// ------------------------------------------------------------------ worker messages

function applySummary(s: WorldSummary): void {
  state.summary = s;
  $<HTMLButtonElement>('undo').disabled = !s.canUndo;
  $<HTMLButtonElement>('redo').disabled = !s.canRedo;
  $('undo').title = s.canUndo ? `Undo ${s.undoLabel} (Ctrl+Z)` : 'Undo (Ctrl+Z)';
  $('redo').title = s.canRedo ? `Redo ${s.redoLabel} (Ctrl+Y)` : 'Redo (Ctrl+Y)';
  for (const k of PLANET_INPUTS) {
    $<HTMLInputElement>(k).value = String(s.params[k]);
    $<HTMLInputElement>(k).dispatchEvent(new Event('input'));
  }
  $<HTMLSelectElement>('retrograde').value = String(s.params.retrograde);
  const t = s.timings.total ? ` · generated in ${(s.timings.total / 1000).toFixed(1)} s` : '';
  $('worldStats').textContent = `${s.cells.toLocaleString('en-US')} cells · ${s.plates.length} plates · ${(s.landFraction * 100).toFixed(1)}% land${t}`;
  const hist = s.historyLog.length
    ? s.historyLog
        .slice()
        .sort((a, b) => parseFloat(b) - parseFloat(a))
        .map((l) => `<p>${l.replace(/[<>&]/g, '')}</p>`)
        .join('')
    : '<p class="muted">No rifting or collisions recorded.</p>';
  $('history').innerHTML = `<p class="muted">${s.params.historyMyr} Myr simulated from a ${s.params.layout === 'supercontinent' ? 'supercontinent' : s.params.layout === 'continents' ? 'several-continent' : 'scattered'} start.</p>${hist}`;
  updateLayerUi();
}

worker.onmessage = (ev: MessageEvent<Response>) => {
  const msg = ev.data;
  switch (msg.type) {
    case 'progress':
      setBusy(msg.stage, msg.frac);
      break;
    case 'ready':
      state.generating = false;
      applySummary(msg.summary);
      if (msg.reason === 'generated' || msg.reason === 'loaded') {
        setInputs(msg.summary.params);
        $('log').innerHTML = `<p class="muted">${msg.reason === 'loaded' ? 'World loaded.' : 'New world generated.'} Edits and their attributed causes appear here.</p>`;
      }
      if (msg.report) logReport(msg.report);
      else if (msg.reason === 'undo' || msg.reason === 'redo') logMessage(msg.reason === 'undo' ? 'Undo' : 'Redo', 'Geology restored; every derived layer was recomputed.');
      setDetail(true);
      if (msg.version !== state.version) {
        state.version = msg.version;
        refreshTiles();
      } else {
        setBusy(null);
        redraw();
      }
      break;
    case 'patch':
      applyPatch(msg.x0, msg.y0, msg.w, msg.h, msg.bitmap);
      redraw();
      if (stroke) {
        stroke.waiting = false;
        flushStroke('move');
      }
      break;
    case 'inspect':
      if (msg.id === inspectId) $('inspector').innerHTML = inspectorHtml(msg.info);
      break;
    case 'file':
      download(msg.name, msg.mime, msg.buffer);
      setBusy(null);
      break;
    case 'error':
      state.generating = false;
      setBusy(null);
      logMessage('Error', msg.message, true);
      console.error(msg.message);
      break;
  }
};

// ------------------------------------------------------------------ start

// Debug handle for the browser console.
(window as unknown as Record<string, unknown>).planetForge = { state, mapView, globeView, tiles, send, get view() { return view; } };

setInputs(DEFAULT_PARAMS);
setTool('inspect');
updateLayerUi();
$('inspector').innerHTML = inspectorHtml(null);
resize();
generate();
