/// <reference lib="webworker" />
/**
 * Tile worker: renders pyramid tiles on request. The simulation worker sends it
 * the mesh and, after every change, the world's fields over a MessageChannel
 * (shared memory when the page is cross-origin isolated).
 */
import type { FieldData, MeshData } from '../render/context';
import { renderWindow, TILE } from '../render/raster';
import type { ContextMessage, StripResult, TileRequest, TileResponse, TileWorkerRequest } from './protocol';

const ctx = self as unknown as DedicatedWorkerGlobalScope;
let mesh: MeshData | null = null;
let fields: FieldData | null = null;
let waiting: TileRequest[] = [];

function respond(msg: TileResponse): void {
  ctx.postMessage(msg, msg.bitmap ? [msg.bitmap] : []);
}

async function render(req: TileRequest): Promise<void> {
  if (!mesh || !fields || fields.meshId !== mesh.id || fields.version < req.version) {
    waiting.push(req);
    return;
  }
  const base = { type: 'tile' as const, id: req.id, version: req.version, level: req.level, x: req.x, y: req.y };
  if (fields.version > req.version) {
    respond({ ...base, ms: 0 });
    return;
  }
  const t0 = performance.now();
  const { rgba } = renderWindow({ mesh, fields }, req.render, { level: req.level, x0: req.x * TILE, y0: req.y * TILE, w: TILE, h: TILE });
  const bitmap = await createImageBitmap(new ImageData(rgba as Uint8ClampedArray<ArrayBuffer>, TILE, TILE));
  respond({ ...base, bitmap, ms: performance.now() - t0 });
}

function onContext(port: MessagePort, msg: ContextMessage): void {
  if (msg.type === 'strip') {
    // Export strips arrive after the fields they belong to (same channel, in order).
    const r = renderWindow({ mesh: mesh!, fields: fields! }, msg.render, msg.window, msg.elevation);
    const data = (msg.elevation ? r.elevation!.buffer : r.rgba.buffer) as ArrayBuffer;
    port.postMessage({ type: 'strip', id: msg.id, data } satisfies StripResult, [data]);
    return;
  }
  if (msg.type === 'mesh') mesh = msg.mesh;
  else fields = msg.fields;
  const queued = waiting;
  waiting = [];
  for (const r of queued) void render(r);
}

ctx.onmessage = (ev: MessageEvent<TileWorkerRequest>) => {
  const msg = ev.data;
  if (msg.type === 'port') {
    const port = msg.port;
    port.onmessage = (e: MessageEvent<ContextMessage>) => onContext(port, e.data);
    return;
  }
  render(msg).catch((err: unknown) => {
    console.error(err);
    respond({ type: 'tile', id: msg.id, version: msg.version, level: msg.level, x: msg.x, y: msg.y, ms: 0 });
  });
};
