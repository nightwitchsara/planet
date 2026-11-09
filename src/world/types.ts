export type Layout = 'supercontinent' | 'continents' | 'scattered';

export interface WorldParams {
  seed: string;
  /** Cells on the fine (output) mesh. */
  cells: number;
  plates: number;
  /** Share of the sphere covered by continental crust at the start of history. */
  continentalFraction: number;
  /**
   * Eustatic sea level (m) relative to the isostatic datum where 33 km of
   * continental crust sits at the shoreline. Land fraction emerges from it.
   */
  seaLevel: number;
  layout: Layout;
  /** Million years of plate motion simulated before the present. */
  historyMyr: number;
  hotspots: number;
  radiusKm: number;
  axialTiltDeg: number;
  /** Global temperature shift (°C) relative to an Earth-like baseline. */
  temperatureOffset: number;
  retrograde: boolean;
  /** Fluvial erosion multiplier (0 disables). */
  erosion: number;
  /** Minimum mean discharge (m³/s) for a drawn river. */
  riverThreshold: number;
}

export const DEFAULT_PARAMS: WorldParams = {
  seed: 'aurelia',
  cells: 1280000,
  plates: 14,
  continentalFraction: 0.4,
  seaLevel: 0,
  layout: 'supercontinent',
  historyMyr: 160,
  hotspots: 16,
  radiusKm: 6371,
  axialTiltDeg: 23.4,
  temperatureOffset: 0,
  retrograde: false,
  erosion: 1,
  riverThreshold: 400,
};

export interface Plate {
  id: number;
  /** Angular velocity vector (rad/Myr); surface velocity is ω × p. */
  wx: number;
  wy: number;
  wz: number;
  color: [number, number, number];
  name: string;
  alive: boolean;
}

export interface Hotspot {
  id: number;
  /** Fixed in the mantle reference frame. */
  x: number;
  y: number;
  z: number;
  /** Volcanic growth rate at the plume center (km of edifice per Myr). */
  strength: number;
  /** Plume radius in radians. */
  radius: number;
  name: string;
  active: boolean;
}

export const CRUST_OCEANIC = 0;
export const CRUST_CONTINENTAL = 1;

/**
 * Present-day geological state: the editable "cause" layer from which every
 * other layer is derived.
 */
export class Geology {
  n: number;
  plate: Int16Array;
  ctype: Uint8Array;
  /** Crustal thickness (km). */
  thick: Float32Array;
  /** Crust formation age (Myr). For oceanic crust this sets depth. */
  age: Float32Array;
  /** Myr since the last mountain-building (thickening) event. */
  orogeny: Float32Array;
  /** Volcanic edifice height above the surrounding crust (km). */
  volc: Float32Array;
  /** Myr since the edifice was last active. */
  volcAge: Float32Array;
  plates: Plate[] = [];
  hotspots: Hotspot[] = [];
  /** Simulated Myr elapsed since the start of history. */
  time = 0;

  constructor(n: number) {
    this.n = n;
    this.plate = new Int16Array(n);
    this.ctype = new Uint8Array(n);
    this.thick = new Float32Array(n);
    this.age = new Float32Array(n);
    this.orogeny = new Float32Array(n);
    this.volc = new Float32Array(n);
    this.volcAge = new Float32Array(n).fill(1e4);
  }

  arrays(): (Int16Array | Uint8Array | Float32Array)[] {
    return [this.plate, this.ctype, this.thick, this.age, this.orogeny, this.volc, this.volcAge];
  }

  static readonly ARRAY_NAMES = ['plate', 'ctype', 'thick', 'age', 'orogeny', 'volc', 'volcAge'] as const;

  clone(): Geology {
    const g = new Geology(this.n);
    g.plate.set(this.plate);
    g.ctype.set(this.ctype);
    g.thick.set(this.thick);
    g.age.set(this.age);
    g.orogeny.set(this.orogeny);
    g.volc.set(this.volc);
    g.volcAge.set(this.volcAge);
    g.plates = this.plates.map((p) => ({ ...p, color: [...p.color] as [number, number, number] }));
    g.hotspots = this.hotspots.map((h) => ({ ...h }));
    g.time = this.time;
    return g;
  }
}

/** Plate speed (km/Myr = mm/yr) at a unit position. */
export function plateVelocity(p: Plate, x: number, y: number, z: number, radiusKm: number): [number, number, number] {
  return [
    (p.wy * z - p.wz * y) * radiusKm,
    (p.wz * x - p.wx * z) * radiusKm,
    (p.wx * y - p.wy * x) * radiusKm,
  ];
}

const PLATE_NAMES = [
  'Aster', 'Boreal', 'Caldera', 'Dorian', 'Ember', 'Fjall', 'Garnet', 'Halcyon', 'Ivory', 'Jade', 'Kestrel',
  'Lumen', 'Marrow', 'Nadir', 'Onyx', 'Pallas', 'Quartz', 'Rime', 'Sable', 'Tamsin', 'Umber', 'Vesper', 'Wren',
  'Xanthe', 'Yarrow', 'Zephyr', 'Argent', 'Basalt', 'Cinder', 'Dune', 'Estuary', 'Flint',
];

export function plateName(i: number): string {
  const base = PLATE_NAMES[i % PLATE_NAMES.length];
  return i < PLATE_NAMES.length ? base : `${base} ${Math.floor(i / PLATE_NAMES.length) + 1}`;
}

export function plateColor(i: number): [number, number, number] {
  // Golden-angle hue spacing, muted for map legibility.
  const h = (i * 137.508) % 360;
  const s = 0.45 + 0.2 * ((i * 7) % 3) / 2;
  const l = 0.55 + 0.1 * ((i * 5) % 2);
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = h / 60;
  const xx = c * (1 - Math.abs((hp % 2) - 1));
  let r = 0, g = 0, b = 0;
  if (hp < 1) [r, g, b] = [c, xx, 0];
  else if (hp < 2) [r, g, b] = [xx, c, 0];
  else if (hp < 3) [r, g, b] = [0, c, xx];
  else if (hp < 4) [r, g, b] = [0, xx, c];
  else if (hp < 5) [r, g, b] = [xx, 0, c];
  else [r, g, b] = [c, 0, xx];
  const m = l - c / 2;
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}
