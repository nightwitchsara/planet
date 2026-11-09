import { BASE_LEVEL, levelHeight, levelWidth, type RenderOptions, TILE } from '../render/raster';
import type { TileRequest, TileResponse } from '../worker/protocol';

export interface TileKey {
  level: number;
  x: number;
  y: number;
}

export interface TileSource {
  get(level: number, x: number, y: number): ImageBitmap | undefined;
}

const keyOf = (t: TileKey): string => `${t.level}/${t.x}/${t.y}`;

/** Number of tiles across and down a pyramid level. */
export const tilesAcross = (level: number): number => levelWidth(level) / TILE;
export const tilesDown = (level: number): number => levelHeight(level) / TILE;

/** Every tile of the overview level. */
export function baseTiles(): TileKey[] {
  const out: TileKey[] = [];
  for (let y = 0; y < tilesDown(BASE_LEVEL); y++) for (let x = 0; x < tilesAcross(BASE_LEVEL); x++) out.push({ level: BASE_LEVEL, x, y });
  return out;
}

/**
 * Requests pyramid tiles from a pool of tile workers, keeps each worker at most
 * a couple of tiles deep so the wish list can be re-prioritized as the view
 * moves, and caches finished detail tiles (least recently used are evicted).
 * Overview tiles are handed straight to `onTile` and not cached.
 */
export class TileManager implements TileSource {
  private load: number[];
  private version = -1;
  private render: RenderOptions = { layer: 'terrain', month: -1, rivers: true };
  private renderKey = '';
  private cache = new Map<string, { bmp: ImageBitmap; used: number }>();
  private inflight = new Set<string>();
  private baseDone = new Set<string>();
  private queue: TileKey[] = [];
  private clock = 0;
  private nextId = 1;
  /** Request id → generation it belongs to, so late answers can be dropped. */
  private gen = new Map<number, string>();
  onTile: (t: TileKey, bmp: ImageBitmap) => void = () => {};
  static readonly MAX_CACHE = 400;
  static readonly PER_WORKER = 2;

  constructor(private workers: Worker[]) {
    this.load = workers.map(() => 0);
    workers.forEach((w, i) => {
      w.onmessage = (ev: MessageEvent<TileResponse>) => this.receive(i, ev.data);
    });
  }

  /** A new world version or new render options: everything shown so far is stale. */
  reset(version: number, render: RenderOptions): void {
    this.version = version;
    this.render = { ...render };
    this.renderKey = `${version}|${render.layer}|${render.month}|${render.rivers}`;
    for (const t of this.cache.values()) t.bmp.close();
    this.cache.clear();
    this.inflight.clear();
    this.baseDone.clear();
    this.queue = [];
  }

  get baseProgress(): number {
    return this.baseDone.size / (tilesAcross(BASE_LEVEL) * tilesDown(BASE_LEVEL));
  }

  get(level: number, x: number, y: number): ImageBitmap | undefined {
    const e = this.cache.get(`${level}/${x}/${y}`);
    if (e) e.used = ++this.clock;
    return e?.bmp;
  }

  /** Replace the wish list (most wanted first). */
  want(list: TileKey[]): void {
    this.queue = list.filter((t) => {
      const k = keyOf(t);
      return !this.inflight.has(k) && !this.baseDone.has(k) && !this.cache.has(k);
    });
    this.pump();
  }

  private pump(): void {
    while (this.queue.length) {
      let wi = -1;
      for (let i = 0; i < this.workers.length; i++) {
        if (this.load[i] < TileManager.PER_WORKER && (wi < 0 || this.load[i] < this.load[wi])) wi = i;
      }
      if (wi < 0) return;
      const t = this.queue.shift()!;
      const id = this.nextId++;
      this.inflight.add(keyOf(t));
      this.gen.set(id, this.renderKey);
      this.load[wi]++;
      const req: TileRequest = { type: 'tile', id, version: this.version, render: this.render, level: t.level, x: t.x, y: t.y };
      this.workers[wi].postMessage(req);
    }
  }

  private receive(i: number, msg: TileResponse): void {
    this.load[i]--;
    const g = this.gen.get(msg.id);
    this.gen.delete(msg.id);
    if (g !== this.renderKey) {
      msg.bitmap?.close();
      this.pump();
      return;
    }
    const k = keyOf(msg);
    this.inflight.delete(k);
    if (!msg.bitmap) {
      this.pump();
      return;
    }
    if (msg.level === BASE_LEVEL) {
      this.baseDone.add(k);
      this.onTile(msg, msg.bitmap);
      msg.bitmap.close();
    } else {
      this.cache.set(k, { bmp: msg.bitmap, used: ++this.clock });
      this.evict();
      this.onTile(msg, msg.bitmap);
    }
    this.pump();
  }

  private evict(): void {
    if (this.cache.size <= TileManager.MAX_CACHE) return;
    const entries = [...this.cache.entries()].sort((a, b) => a[1].used - b[1].used);
    for (const [k, e] of entries.slice(0, this.cache.size - TileManager.MAX_CACHE * 0.8)) {
      e.bmp.close();
      this.cache.delete(k);
    }
  }
}
