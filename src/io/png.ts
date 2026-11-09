const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf: Uint8Array, start: number, end: number): number {
  let c = 0xffffffff;
  for (let i = start; i < end; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

/** zlib-wrapped deflate (RFC 1950), sync or async. */
export type Deflate = (data: Uint8Array) => Uint8Array | Promise<Uint8Array>;

export async function deflateBrowser(data: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([data as BlobPart]).stream().pipeThrough(new CompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export interface PngFormat {
  /** 1 = gray, 3 = RGB, 4 = RGBA */
  channels: 1 | 3 | 4;
  bitDepth: 8 | 16;
}

/**
 * Encode a PNG. `data` holds width*height*channels samples (Uint8 for 8-bit,
 * Uint16 for 16-bit). Rows use the Sub filter, which compresses heightmaps well.
 */
export async function encodePNG(
  width: number,
  height: number,
  data: ArrayLike<number>,
  fmt: PngFormat,
  deflate: Deflate,
): Promise<Uint8Array> {
  const bpp = fmt.channels * (fmt.bitDepth / 8);
  const stride = width * bpp;
  const rawRows = new Uint8Array(height * (stride + 1));
  const row = new Uint8Array(stride);
  for (let yy = 0; yy < height; yy++) {
    const base = yy * width * fmt.channels;
    if (fmt.bitDepth === 16) {
      for (let i = 0; i < width * fmt.channels; i++) {
        const v = data[base + i];
        row[2 * i] = (v >>> 8) & 0xff;
        row[2 * i + 1] = v & 0xff;
      }
    } else {
      for (let i = 0; i < width * fmt.channels; i++) row[i] = data[base + i];
    }
    const o = yy * (stride + 1);
    rawRows[o] = 1; // Sub filter
    for (let i = 0; i < stride; i++) rawRows[o + 1 + i] = (row[i] - (i >= bpp ? row[i - bpp] : 0)) & 0xff;
  }
  const idat = await deflate(rawRows);
  const colorType = fmt.channels === 1 ? 0 : fmt.channels === 3 ? 2 : 6;
  const chunks: [string, Uint8Array][] = [];
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, width);
  dv.setUint32(4, height);
  ihdr[8] = fmt.bitDepth;
  ihdr[9] = colorType;
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;
  chunks.push(['IHDR', ihdr], ['IDAT', idat], ['IEND', new Uint8Array(0)]);
  let total = 8;
  for (const [, d] of chunks) total += 12 + d.length;
  const out = new Uint8Array(total);
  out.set([137, 80, 78, 71, 13, 10, 26, 10], 0);
  let p = 8;
  const odv = new DataView(out.buffer);
  for (const [type, d] of chunks) {
    odv.setUint32(p, d.length);
    for (let i = 0; i < 4; i++) out[p + 4 + i] = type.charCodeAt(i);
    out.set(d, p + 8);
    odv.setUint32(p + 8 + d.length, crc32(out, p + 4, p + 8 + d.length));
    p += 12 + d.length;
  }
  return out;
}
