import { BASE_LEVEL, levelWidth, MAX_LEVEL, TILE } from '../render/raster';
import { type TileKey, type TileSource, tilesAcross, tilesDown } from './tiles';

export interface ScreenPoint {
  x: number;
  y: number;
  visible: boolean;
}

export interface LatLonDeg {
  lat: number;
  lon: number;
}

/** Common interface of the flat map and the globe. Coordinates are CSS pixels. */
export interface View {
  readonly kind: 'map' | 'globe';
  /** Draw detail tiles over the overview (off while a brush preview is showing). */
  showDetail: boolean;
  resize(width: number, height: number, dpr: number): void;
  draw(): void;
  screenToLatLon(x: number, y: number): LatLonDeg | null;
  project(lat: number, lon: number): ScreenPoint;
  panBy(dx: number, dy: number): void;
  zoomAt(x: number, y: number, factor: number): void;
  /** Pixels per km of ground distance near a point (for brush cursors). */
  pxPerKm(radiusKm: number): number;
  show(visible: boolean): void;
  /** Detail tiles worth rendering for the current view, most central first. */
  wantedTiles(): TileKey[];
  /** Screen position of an overview tile's center (to render visible ones first). */
  baseTileCenter(t: TileKey): ScreenPoint;
}

const D = Math.PI / 180;
const TAU = 2 * Math.PI;
const wrapTile = (x: number, n: number): number => ((x % n) + n) % n;
/** Pyramid pixels per radian at a level. */
const levelPxPerRad = (L: number): number => levelWidth(L) / TAU;

/**
 * Pyramid level nearest in scale to `pxPerRad` device pixels per radian (within
 * a factor √2, so thin rivers survive minification). BASE_LEVEL means the
 * overview is enough.
 */
function levelFor(pxPerRad: number): number {
  let L = BASE_LEVEL;
  while (L < MAX_LEVEL && levelPxPerRad(L) < pxPerRad * Math.SQRT1_2) L++;
  return L;
}

export class MapView implements View {
  readonly kind = 'map';
  showDetail = true;
  /** The composed overview image (BASE_LEVEL, 2048 × 1024). */
  base: HTMLCanvasElement | null = null;
  tiles: TileSource | null = null;
  private ctx: CanvasRenderingContext2D;
  private w = 1;
  private h = 1;
  private dpr = 1;
  lon0 = 0;
  lat0 = 0;
  /** Pixels per radian. */
  k = 100;
  /** Until the user zooms or pans, keep the whole map fitted to the viewport. */
  private userMoved = false;

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
  }

  show(visible: boolean): void {
    this.canvas.style.display = visible ? 'block' : 'none';
  }

  resize(width: number, height: number, dpr: number): void {
    this.w = width;
    this.h = height;
    this.dpr = dpr;
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    if (!this.userMoved) this.fit();
    this.clamp();
  }

  fit(): void {
    this.k = Math.min(this.w / TAU, this.h / Math.PI);
    this.lon0 = 0;
    this.lat0 = 0;
  }

  private clamp(): void {
    const minK = Math.min(this.w / TAU, this.h / Math.PI) * 0.75;
    // Deep enough for the finest pyramid level at one pixel per screen pixel.
    this.k = Math.max(minK, Math.min(this.k, levelPxPerRad(MAX_LEVEL) / this.dpr));
    const halfH = this.h / 2 / this.k;
    const limit = Math.max(0, Math.PI / 2 - halfH);
    this.lat0 = Math.max(-limit, Math.min(limit, this.lat0));
    this.lon0 = ((((this.lon0 + Math.PI) % TAU) + TAU) % TAU) - Math.PI;
  }

  /** Device-pixel geometry of the pyramid at level L: origin of tile (0, 0) and tile size. */
  private tileFrame(L: number): { ox: number; oy: number; size: number } {
    const kd = this.k * this.dpr;
    return {
      ox: (this.w / 2) * this.dpr - (this.lon0 + Math.PI) * kd,
      oy: (this.h / 2) * this.dpr - (Math.PI / 2 - this.lat0) * kd,
      size: (TAU / tilesAcross(L)) * kd,
    };
  }

  private visibleRange(L: number): { tx0: number; tx1: number; ty0: number; ty1: number } {
    const { ox, oy, size } = this.tileFrame(L);
    return {
      tx0: Math.floor(-ox / size),
      tx1: Math.floor((this.w * this.dpr - ox) / size),
      ty0: Math.max(0, Math.floor(-oy / size)),
      ty1: Math.min(tilesDown(L) - 1, Math.floor((this.h * this.dpr - oy) / size)),
    };
  }

  wantedTiles(): TileKey[] {
    const L = levelFor(this.k * this.dpr);
    if (L <= BASE_LEVEL || !this.showDetail) return [];
    const { ox, oy, size } = this.tileFrame(L);
    const { tx0, tx1, ty0, ty1 } = this.visibleRange(L);
    const cx = (this.w * this.dpr) / 2;
    const cy = (this.h * this.dpr) / 2;
    const out: (TileKey & { d: number })[] = [];
    const n = tilesAcross(L);
    for (let ty = ty0; ty <= ty1; ty++) {
      for (let tx = tx0; tx <= tx1 && tx - tx0 < n; tx++) {
        const d = Math.hypot(ox + (tx + 0.5) * size - cx, oy + (ty + 0.5) * size - cy);
        out.push({ level: L, x: wrapTile(tx, n), y: ty, d });
      }
    }
    return out.sort((a, b) => a.d - b.d);
  }

  baseTileCenter(t: TileKey): ScreenPoint {
    const lon = ((t.x + 0.5) / tilesAcross(BASE_LEVEL)) * TAU - Math.PI;
    const lat = Math.PI / 2 - ((t.y + 0.5) / tilesDown(BASE_LEVEL)) * Math.PI;
    return this.project(lat / D, lon / D);
  }

  draw(): void {
    const { ctx, dpr } = this;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#0b0f17';
    ctx.fillRect(0, 0, this.w, this.h);
    if (!this.base) return;
    const cx = this.w / 2;
    const cy = this.h / 2;
    const dw = TAU * this.k;
    const dh = Math.PI * this.k;
    let x0 = cx + (-Math.PI - this.lon0) * this.k;
    const y0 = cy - (Math.PI / 2 - this.lat0) * this.k;
    while (x0 > 0) x0 -= dw;
    ctx.imageSmoothingEnabled = this.k * TAU < this.base.width * 1.5;
    ctx.imageSmoothingQuality = 'high';
    for (let x = x0; x < this.w; x += dw) ctx.drawImage(this.base, x, y0, dw, dh);

    const L = levelFor(this.k * dpr);
    if (L <= BASE_LEVEL || !this.showDetail || !this.tiles) return;
    // Detail tiles in device pixels, edges rounded so neighbors meet exactly.
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.imageSmoothingEnabled = true;
    const { ox, oy, size } = this.tileFrame(L);
    const { tx0, tx1, ty0, ty1 } = this.visibleRange(L);
    const n = tilesAcross(L);
    for (let ty = ty0; ty <= ty1; ty++) {
      const y0d = Math.round(oy + ty * size);
      const y1d = Math.round(oy + (ty + 1) * size);
      for (let tx = tx0; tx <= tx1; tx++) {
        const x0d = Math.round(ox + tx * size);
        const x1d = Math.round(ox + (tx + 1) * size);
        const xw = wrapTile(tx, n);
        // Fall back to the nearest coarser detail tile while this one renders.
        for (let l = L; l > BASE_LEVEL; l--) {
          const s = L - l;
          const bmp = this.tiles.get(l, xw >> s, ty >> s);
          if (!bmp) continue;
          const sub = TILE >> s;
          ctx.drawImage(bmp, (xw & ((1 << s) - 1)) * sub, (ty & ((1 << s) - 1)) * sub, sub, sub, x0d, y0d, x1d - x0d, y1d - y0d);
          break;
        }
      }
    }
  }

  screenToLatLon(x: number, y: number): LatLonDeg | null {
    const lat = this.lat0 - (y - this.h / 2) / this.k;
    if (lat > Math.PI / 2 || lat < -Math.PI / 2) return null;
    let lon = this.lon0 + (x - this.w / 2) / this.k;
    lon = ((((lon + Math.PI) % TAU) + TAU) % TAU) - Math.PI;
    return { lat: lat / D, lon: lon / D };
  }

  project(lat: number, lon: number): ScreenPoint {
    let dl = lon * D - this.lon0;
    dl = ((((dl + Math.PI) % TAU) + TAU) % TAU) - Math.PI;
    return { x: this.w / 2 + dl * this.k, y: this.h / 2 - (lat * D - this.lat0) * this.k, visible: true };
  }

  panBy(dx: number, dy: number): void {
    this.userMoved = true;
    this.lon0 -= dx / this.k;
    this.lat0 += dy / this.k;
    this.clamp();
  }

  zoomAt(x: number, y: number, factor: number): void {
    this.userMoved = true;
    const before = this.screenToLatLon(x, y);
    this.k *= factor;
    this.clamp();
    if (before) {
      const after = this.project(before.lat, before.lon);
      this.panBy(after.x - x, after.y - y);
    }
  }

  pxPerKm(radiusKm: number): number {
    return this.k / radiusKm;
  }
}

const VS = `#version 300 es
in vec2 pos;
void main() { gl_Position = vec4(pos, 0.0, 1.0); }`;

const FS = `#version 300 es
precision highp float;
uniform sampler2D tex;
uniform sampler2D detail;
uniform mat3 rot;
uniform vec2 center;
uniform float radius;
uniform float texW;
// Detail mosaic: origin and size in tiles, tiles across/down the level, texture scale, level px per radian.
uniform vec4 dInfo;
uniform vec2 dGrid;
uniform float dScale;
uniform float dPxPerRad;
uniform float dOn;
out vec4 outColor;
const float PI = 3.14159265358979;
void main() {
  vec2 d = (gl_FragCoord.xy - center) / radius;
  float r2 = dot(d, d);
  if (r2 > 1.0) {
    float r = sqrt(r2);
    float glow = exp(-(r - 1.0) * 26.0);
    vec3 bg = vec3(0.043, 0.059, 0.09);
    outColor = vec4(mix(bg, vec3(0.42, 0.62, 0.95), glow * 0.55), 1.0);
    return;
  }
  float z = sqrt(1.0 - r2);
  vec3 pw = rot * vec3(d, z);
  float lat = asin(clamp(pw.z, -1.0, 1.0));
  float lon = atan(pw.y, pw.x);
  vec2 uv = vec2((lon + PI) / (2.0 * PI), (PI * 0.5 - lat) / PI);
  float lod = log2(max(1.0, texW / (2.0 * PI * radius * max(z, 0.08))));
  vec3 col = textureLod(tex, uv, lod).rgb;
  if (dOn > 0.5) {
    float tu = uv.x * dGrid.x - dInfo.x;
    tu -= dGrid.x * floor(tu / dGrid.x);
    float tv = uv.y * dGrid.y - dInfo.y;
    float texelsPerPx = dPxPerRad / (radius * max(z, 0.05));
    if (tu < dInfo.z && tv >= 0.0 && tv < dInfo.w && texelsPerPx < 2.5) {
      vec4 dc = textureLod(detail, vec2(tu, tv) * dScale, 0.0);
      col = mix(col, dc.rgb, dc.a * (1.0 - smoothstep(1.4, 2.5, texelsPerPx)));
    }
  }
  float limb = 0.78 + 0.22 * z;
  float rim = smoothstep(0.82, 1.0, sqrt(r2)) * 0.18;
  outColor = vec4(col * limb + vec3(0.35, 0.5, 0.8) * rim, 1.0);
}`;

interface Mosaic {
  level: number;
  x0: number;
  y0: number;
  nx: number;
  ny: number;
}

export class GlobeView implements View {
  readonly kind = 'globe';
  showDetail = true;
  tiles: TileSource | null = null;
  private gl: WebGL2RenderingContext;
  private prog: WebGLProgram;
  private tex: WebGLTexture;
  private detailTex: WebGLTexture;
  private detailFbo: WebGLFramebuffer;
  /** Detail texture size and how many tiles fit across it. */
  private dSize: number;
  private slots: number;
  private mosaic: Mosaic | null = null;
  private mipDirty = false;
  private w = 1;
  private h = 1;
  private dpr = 1;
  lat0 = 20 * D;
  lon0 = 0;
  radius = 300;
  private fitted = false;
  private u: Record<string, WebGLUniformLocation | null> = {};

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { antialias: false });
    if (!gl) throw new Error('WebGL2 is not available');
    this.gl = gl;
    const sh = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? 'shader error');
      return s;
    };
    const prog = gl.createProgram()!;
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog) ?? 'link error');
    this.prog = prog;
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'pos');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    for (const name of ['tex', 'detail', 'rot', 'center', 'radius', 'texW', 'dInfo', 'dGrid', 'dScale', 'dPxPerRad', 'dOn']) {
      this.u[name] = gl.getUniformLocation(prog, name);
    }
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

    // Overview texture, filled tile by tile.
    this.tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    const bw = levelWidth(BASE_LEVEL);
    const bh = bw / 2;
    const dark = new Uint8Array(bw * bh * 4);
    new Uint32Array(dark.buffer).fill(0xff170f0b);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, bw, bh, 0, gl.RGBA, gl.UNSIGNED_BYTE, dark);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.generateMipmap(gl.TEXTURE_2D);

    // Detail mosaic texture: a block of tiles around the view.
    this.dSize = Math.min(4096, gl.getParameter(gl.MAX_TEXTURE_SIZE) as number);
    this.slots = Math.floor(this.dSize / TILE);
    this.detailTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.detailTex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, this.dSize, this.dSize);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.detailFbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.detailFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.detailTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.clearDetail();
  }

  show(visible: boolean): void {
    this.canvas.style.display = visible ? 'block' : 'none';
  }

  resize(width: number, height: number, dpr: number): void {
    this.w = width;
    this.h = height;
    this.dpr = dpr;
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    if (!this.fitted) {
      this.radius = Math.min(width, height) * 0.44;
      this.fitted = true;
    }
  }

  /** Update part of the overview texture (an overview tile or a brush preview). */
  putBase(x: number, y: number, src: ImageBitmap | ImageData): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, x, y, gl.RGBA, gl.UNSIGNED_BYTE, src);
    this.mipDirty = true;
  }

  /** A finished detail tile: upload it if it belongs to the current mosaic. */
  putDetail(t: TileKey, bmp: ImageBitmap): void {
    const m = this.mosaic;
    if (!m || t.level !== m.level) return;
    const rx = wrapTile(t.x - m.x0, tilesAcross(m.level));
    const ry = t.y - m.y0;
    if (rx >= m.nx || ry < 0 || ry >= m.ny) return;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.detailTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, rx * TILE, ry * TILE, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
  }

  /** Everything uploaded so far is stale (new world version or layer); keep the layout. */
  resetDetail(): void {
    this.clearDetail();
  }

  private clearDetail(): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.detailFbo);
    gl.viewport(0, 0, this.dSize, this.dSize);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private basisCache = { lat: NaN, lon: NaN, b: [] as number[] };

  /** Columns: east, north, out (view → world). */
  private basis(): number[] {
    const bc = this.basisCache;
    if (bc.lat === this.lat0 && bc.lon === this.lon0) return bc.b;
    const cl = Math.cos(this.lat0);
    const p = [cl * Math.cos(this.lon0), cl * Math.sin(this.lon0), Math.sin(this.lat0)];
    const e = [-Math.sin(this.lon0), Math.cos(this.lon0), 0];
    const n = [p[1] * e[2] - p[2] * e[1], p[2] * e[0] - p[0] * e[2], p[0] * e[1] - p[1] * e[0]];
    this.basisCache = { lat: this.lat0, lon: this.lon0, b: [...e, ...n, ...p] };
    return this.basisCache.b;
  }

  /** Latitude range and longitude span (relative to the view center) visible on screen. */
  private visibleBounds(): { latMin: number; latMax: number; dlMin: number; dlMax: number; allLon: boolean } {
    let latMin = Infinity;
    let latMax = -Infinity;
    let dlMin = Infinity;
    let dlMax = -Infinity;
    const add = (ll: LatLonDeg | null) => {
      if (!ll) return;
      const lat = ll.lat * D;
      let dl = ll.lon * D - this.lon0;
      dl -= TAU * Math.round(dl / TAU);
      latMin = Math.min(latMin, lat);
      latMax = Math.max(latMax, lat);
      dlMin = Math.min(dlMin, dl);
      dlMax = Math.max(dlMax, dl);
    };
    const N = 24;
    for (let j = 0; j <= N; j++) for (let i = 0; i <= N; i++) add(this.screenToLatLon((i / N) * this.w, (j / N) * this.h));
    for (let a = 0; a < 96; a++) {
      const x = this.w / 2 + Math.cos((a / 96) * TAU) * this.radius * 0.999;
      const y = this.h / 2 + Math.sin((a / 96) * TAU) * this.radius * 0.999;
      if (x >= 0 && y >= 0 && x <= this.w && y <= this.h) add(this.screenToLatLon(x, y));
    }
    let allLon = dlMax - dlMin > Math.PI * 1.5;
    for (const pole of [90, -90]) {
      const p = this.project(pole, 0);
      if (p.visible && p.x >= 0 && p.y >= 0 && p.x <= this.w && p.y <= this.h) {
        allLon = true;
        if (pole > 0) latMax = Math.PI / 2;
        else latMin = -Math.PI / 2;
      }
    }
    return { latMin, latMax, dlMin, dlMax, allLon };
  }

  wantedTiles(): TileKey[] {
    let L = levelFor(this.radius * this.dpr);
    if (L <= BASE_LEVEL || !this.showDetail) {
      this.mosaic = null;
      return [];
    }
    const b = this.visibleBounds();
    if (!Number.isFinite(b.latMin)) return [];
    let range: Mosaic | null = null;
    for (; L > BASE_LEVEL; L--) {
      const nx = tilesAcross(L);
      const ny = tilesDown(L);
      const ty0 = Math.max(0, Math.floor(((Math.PI / 2 - b.latMax) / Math.PI) * ny));
      const ty1 = Math.min(ny - 1, Math.floor(((Math.PI / 2 - b.latMin) / Math.PI) * ny));
      const tx0 = b.allLon ? 0 : Math.floor(((this.lon0 + b.dlMin + Math.PI) / TAU) * nx);
      const tx1 = b.allLon ? nx - 1 : Math.floor(((this.lon0 + b.dlMax + Math.PI) / TAU) * nx);
      if (tx1 - tx0 + 1 <= this.slots && ty1 - ty0 + 1 <= this.slots) {
        range = { level: L, x0: tx0, y0: ty0, nx: tx1 - tx0 + 1, ny: ty1 - ty0 + 1 };
        break;
      }
    }
    if (!range) {
      this.mosaic = null;
      return [];
    }
    this.useMosaic(range);
    // Tiles in range that actually show on screen, nearest the center first.
    const out: (TileKey & { d: number })[] = [];
    const nAcross = tilesAcross(range.level);
    const nDown = tilesDown(range.level);
    for (let ty = range.y0; ty < range.y0 + range.ny; ty++) {
      for (let tx = range.x0; tx < range.x0 + range.nx; tx++) {
        let best = Infinity;
        for (const [fx, fy] of [[0.5, 0.5], [0, 0], [1, 0], [0, 1], [1, 1]]) {
          const lon = ((tx + fx) / nAcross) * TAU - Math.PI;
          const lat = Math.PI / 2 - ((ty + fy) / nDown) * Math.PI;
          const p = this.project(lat / D, lon / D);
          if (!p.visible || p.x < -TILE || p.y < -TILE || p.x > this.w + TILE || p.y > this.h + TILE) continue;
          best = Math.min(best, Math.hypot(p.x - this.w / 2, p.y - this.h / 2));
        }
        if (best < Infinity) out.push({ level: range.level, x: wrapTile(tx, nAcross), y: ty, d: best });
      }
    }
    return out.sort((a, c) => a.d - c.d);
  }

  /** Keep the current mosaic while it still covers the view; otherwise lay out a new one. */
  private useMosaic(r: Mosaic): void {
    const m = this.mosaic;
    const nAcross = tilesAcross(r.level);
    if (m && m.level === r.level) {
      const rx = wrapTile(r.x0 - m.x0, nAcross);
      if (rx + r.nx <= m.nx && r.y0 >= m.y0 && r.y0 + r.ny <= m.y0 + m.ny) return;
    }
    const nx = Math.min(this.slots, nAcross, r.nx + 2);
    const ny = Math.min(this.slots, tilesDown(r.level), r.ny + 2);
    const next: Mosaic = {
      level: r.level,
      x0: r.x0 - Math.floor((nx - r.nx) / 2),
      y0: Math.max(0, Math.min(tilesDown(r.level) - ny, r.y0 - Math.floor((ny - r.ny) / 2))),
      nx,
      ny,
    };
    this.mosaic = next;
    this.clearDetail();
    if (!this.tiles) return;
    for (let ty = next.y0; ty < next.y0 + ny; ty++) {
      for (let tx = next.x0; tx < next.x0 + nx; tx++) {
        const t = { level: next.level, x: wrapTile(tx, nAcross), y: ty };
        const bmp = this.tiles.get(t.level, t.x, t.y);
        if (bmp) this.putDetail(t, bmp);
      }
    }
  }

  baseTileCenter(t: TileKey): ScreenPoint {
    const lon = ((t.x + 0.5) / tilesAcross(BASE_LEVEL)) * TAU - Math.PI;
    const lat = Math.PI / 2 - ((t.y + 0.5) / tilesDown(BASE_LEVEL)) * Math.PI;
    return this.project(lat / D, lon / D);
  }

  draw(): void {
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.043, 0.059, 0.09, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (this.mipDirty) {
      gl.bindTexture(gl.TEXTURE_2D, this.tex);
      gl.generateMipmap(gl.TEXTURE_2D);
      this.mipDirty = false;
    }
    gl.useProgram(this.prog);
    const u = this.u;
    gl.uniformMatrix3fv(u.rot, false, new Float32Array(this.basis()));
    gl.uniform2f(u.center, (this.w / 2) * this.dpr, (this.h / 2) * this.dpr);
    gl.uniform1f(u.radius, this.radius * this.dpr);
    gl.uniform1f(u.texW, levelWidth(BASE_LEVEL));
    gl.uniform1i(u.tex, 0);
    gl.uniform1i(u.detail, 1);
    const m = this.showDetail ? this.mosaic : null;
    gl.uniform1f(u.dOn, m ? 1 : 0);
    if (m) {
      gl.uniform4f(u.dInfo, m.x0, m.y0, m.nx, m.ny);
      gl.uniform2f(u.dGrid, tilesAcross(m.level), tilesDown(m.level));
      gl.uniform1f(u.dScale, TILE / this.dSize);
      gl.uniform1f(u.dPxPerRad, levelPxPerRad(m.level));
    }
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.detailTex);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  screenToLatLon(x: number, y: number): LatLonDeg | null {
    const dx = (x - this.w / 2) / this.radius;
    const dy = (this.h / 2 - y) / this.radius;
    const r2 = dx * dx + dy * dy;
    if (r2 > 1) return null;
    const z = Math.sqrt(1 - r2);
    const b = this.basis();
    const px = b[0] * dx + b[3] * dy + b[6] * z;
    const py = b[1] * dx + b[4] * dy + b[7] * z;
    const pz = b[2] * dx + b[5] * dy + b[8] * z;
    return { lat: Math.asin(Math.max(-1, Math.min(1, pz))) / D, lon: Math.atan2(py, px) / D };
  }

  project(lat: number, lon: number): ScreenPoint {
    const cl = Math.cos(lat * D);
    const q = [cl * Math.cos(lon * D), cl * Math.sin(lon * D), Math.sin(lat * D)];
    const b = this.basis();
    const x = q[0] * b[0] + q[1] * b[1] + q[2] * b[2];
    const y = q[0] * b[3] + q[1] * b[4] + q[2] * b[5];
    const z = q[0] * b[6] + q[1] * b[7] + q[2] * b[8];
    return { x: this.w / 2 + x * this.radius, y: this.h / 2 - y * this.radius, visible: z > 0.02 };
  }

  panBy(dx: number, dy: number): void {
    this.lon0 -= dx / this.radius / Math.max(0.3, Math.cos(this.lat0));
    this.lat0 = Math.max(-89 * D, Math.min(89 * D, this.lat0 + dy / this.radius));
  }

  zoomAt(_x: number, _y: number, factor: number): void {
    // Up to one finest-level pixel per screen pixel at the center.
    this.radius = Math.max(80, Math.min(this.radius * factor, levelPxPerRad(MAX_LEVEL) / this.dpr));
  }

  pxPerKm(radiusKm: number): number {
    return this.radius / radiusKm;
  }
}
