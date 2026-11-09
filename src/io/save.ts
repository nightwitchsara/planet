import { Geology, type Hotspot, type Plate, type WorldParams } from '../world/types';

const MAGIC = 'PFW1';

interface Header {
  version: 1;
  n: number;
  params: WorldParams;
  plates: Plate[];
  hotspots: Hotspot[];
  time: number;
  historyLog: string[];
}

/**
 * Binary world file: magic, header length, JSON header, then the geology
 * arrays. Everything else (climate, rivers, biomes) is re-derived on load.
 */
export function serializeWorld(params: WorldParams, geo: Geology, historyLog: string[]): ArrayBuffer {
  const header: Header = { version: 1, n: geo.n, params, plates: geo.plates, hotspots: geo.hotspots, time: geo.time, historyLog };
  const json = new TextEncoder().encode(JSON.stringify(header));
  const arrays = geo.arrays();
  let size = 8 + json.length;
  size = (size + 3) & ~3;
  const offsets: number[] = [];
  for (const a of arrays) {
    offsets.push(size);
    size += a.byteLength;
    size = (size + 3) & ~3;
  }
  const buf = new ArrayBuffer(size);
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < 4; i++) bytes[i] = MAGIC.charCodeAt(i);
  new DataView(buf).setUint32(4, json.length, true);
  bytes.set(json, 8);
  arrays.forEach((a, i) => bytes.set(new Uint8Array(a.buffer, a.byteOffset, a.byteLength), offsets[i]));
  return buf;
}

export function deserializeWorld(buf: ArrayBuffer): { params: WorldParams; geo: Geology; historyLog: string[] } {
  const bytes = new Uint8Array(buf);
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
  if (magic !== MAGIC) throw new Error('Not a Planet Forge world file');
  const len = new DataView(buf).getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + len))) as Header;
  const geo = new Geology(header.n);
  let off = (8 + len + 3) & ~3;
  for (const a of geo.arrays()) {
    new Uint8Array(a.buffer, a.byteOffset, a.byteLength).set(bytes.subarray(off, off + a.byteLength));
    off = (off + a.byteLength + 3) & ~3;
  }
  geo.plates = header.plates;
  geo.hotspots = header.hotspots;
  geo.time = header.time;
  return { params: header.params, geo, historyLog: header.historyLog ?? [] };
}
