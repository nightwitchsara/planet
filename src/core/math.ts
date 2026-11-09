export type Vec3 = [number, number, number];

export const DEG = Math.PI / 180;

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function smoothstep(e0: number, e1: number, v: number): number {
  const t = clamp((v - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
}

export function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function cross3(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function len3(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}

export function normalize3(a: Vec3): Vec3 {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}

export function scale3(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}

export function add3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function sub3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

/** Unit vector for latitude/longitude in radians. The polar axis is +z. */
export function latLonToVec(lat: number, lon: number): Vec3 {
  const c = Math.cos(lat);
  return [c * Math.cos(lon), c * Math.sin(lon), Math.sin(lat)];
}

export function vecToLatLon(x: number, y: number, z: number): [number, number] {
  const l = Math.hypot(x, y, z) || 1;
  return [Math.asin(clamp(z / l, -1, 1)), Math.atan2(y, x)];
}

/** Great-circle angle between two unit vectors (radians). */
export function angleBetween(ax: number, ay: number, az: number, bx: number, by: number, bz: number): number {
  const cx = ay * bz - az * by;
  const cy = az * bx - ax * bz;
  const cz = ax * by - ay * bx;
  return Math.atan2(Math.hypot(cx, cy, cz), ax * bx + ay * by + az * bz);
}

/**
 * Row-major 3x3 rotation matrix rotating by `angle` (radians, right-handed)
 * about the unit `axis`.
 */
export function rotationMatrix(ax: number, ay: number, az: number, angle: number, out: Float64Array = new Float64Array(9)): Float64Array {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const t = 1 - c;
  out[0] = c + ax * ax * t;
  out[1] = ax * ay * t - az * s;
  out[2] = ax * az * t + ay * s;
  out[3] = ay * ax * t + az * s;
  out[4] = c + ay * ay * t;
  out[5] = ay * az * t - ax * s;
  out[6] = az * ax * t - ay * s;
  out[7] = az * ay * t + ax * s;
  out[8] = c + az * az * t;
  return out;
}

/** Rotation that turns a point by an angular-velocity vector `w` (rad/unit time) over time `t`. */
export function rotationFromOmega(wx: number, wy: number, wz: number, t: number, out: Float64Array = new Float64Array(9)): Float64Array {
  const w = Math.hypot(wx, wy, wz);
  if (w < 1e-12) {
    out.fill(0);
    out[0] = out[4] = out[8] = 1;
    return out;
  }
  return rotationMatrix(wx / w, wy / w, wz / w, w * t, out);
}

export function applyMat(m: Float64Array, x: number, y: number, z: number): Vec3 {
  return [m[0] * x + m[1] * y + m[2] * z, m[3] * x + m[4] * y + m[5] * z, m[6] * x + m[7] * y + m[8] * z];
}

/** Local east and north unit vectors at a unit position (polar axis +z). */
export function eastNorth(x: number, y: number, z: number): { e: Vec3; n: Vec3 } {
  const r = Math.hypot(x, y);
  if (r < 1e-9) return { e: [0, 1, 0], n: [-1, 0, 0] };
  const e: Vec3 = [-y / r, x / r, 0];
  // n = p Ã— e  (points toward +z along the meridian)
  const n: Vec3 = [y * e[2] - z * e[1], z * e[0] - x * e[2], x * e[1] - y * e[0]];
  return { e, n };
}

/** Compass bearing (degrees clockwise from north) of a tangent vector at position p. */
export function bearingOf(p: Vec3, v: Vec3): number {
  const { e, n } = eastNorth(p[0], p[1], p[2]);
  const b = Math.atan2(dot3(v, e), dot3(v, n)) / DEG;
  return (b + 360) % 360;
}

export function compassName(bearing: number): string {
  const names = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
  return names[Math.round(bearing / 22.5) % 16];
}

const RADIX_F = new Float32Array(1);
const RADIX_U = new Uint32Array(RADIX_F.buffer);

/**
 * Sorts the indices in `idx` in place by `value[idx[i]]`, highest first. Stable
 * (equal values keep their order), so it gives exactly the order of
 * `idx.sort((a, b) => value[b] - value[a])` on a plain array, in linear time:
 * a least-significant-digit radix sort over the float bits.
 */
export function sortByValueDesc(idx: Int32Array, value: Float32Array): void {
  const m = idx.length;
  let key = new Uint32Array(m);
  let tmpKey = new Uint32Array(m);
  let tmpIdx: Int32Array = new Int32Array(m);
  let src: Int32Array = idx;
  for (let i = 0; i < m; i++) {
    // −0 sorts with +0, as the comparator sees them equal.
    RADIX_F[0] = value[idx[i]] === 0 ? 0 : value[idx[i]];
    const u = RADIX_U[0];
    // Float order as unsigned order (negative floats reversed), then flipped for highest first.
    key[i] = ~(u & 0x80000000 ? ~u : u | 0x80000000) >>> 0;
  }
  const count = new Uint32Array(256);
  for (let shift = 0; shift < 32; shift += 8) {
    count.fill(0);
    for (let i = 0; i < m; i++) count[(key[i] >>> shift) & 255]++;
    let sum = 0;
    for (let b = 0; b < 256; b++) {
      const c = count[b];
      count[b] = sum;
      sum += c;
    }
    for (let i = 0; i < m; i++) {
      const j = count[(key[i] >>> shift) & 255]++;
      tmpKey[j] = key[i];
      tmpIdx[j] = src[i];
    }
    [key, tmpKey] = [tmpKey, key];
    [src, tmpIdx] = [tmpIdx, src];
  }
  // Four passes: the sorted data is back in the array it started from.
  if (src !== idx) idx.set(src);
}
