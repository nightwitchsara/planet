export type RGB = [number, number, number];
type Stops = [number, RGB][];

export function ramp(stops: Stops, v: number): RGB {
  if (v <= stops[0][0]) return stops[0][1];
  for (let i = 1; i < stops.length; i++) {
    if (v <= stops[i][0]) {
      const [v0, a] = stops[i - 1];
      const [v1, b] = stops[i];
      const t = (v - v0) / (v1 - v0);
      return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
    }
  }
  return stops[stops.length - 1][1];
}

export const HYPSO: Stops = [
  [0, [64, 112, 74]],
  [200, [102, 142, 82]],
  [600, [168, 170, 104]],
  [1200, [196, 168, 110]],
  [2000, [168, 128, 92]],
  [3200, [140, 110, 100]],
  [4500, [190, 186, 182]],
  [6000, [250, 250, 250]],
];

export const BATHY: Stops = [
  [-8000, [8, 22, 60]],
  [-5000, [18, 44, 98]],
  [-3000, [28, 66, 126]],
  [-1000, [44, 100, 160]],
  [-200, [66, 140, 186]],
  [0, [96, 176, 206]],
];

/** Natural-color water: deep navy to shelf teal. */
export const SEA: Stops = [
  [-7000, [10, 24, 52]],
  [-4500, [16, 38, 76]],
  [-2500, [22, 52, 96]],
  [-600, [30, 72, 118]],
  [-150, [40, 104, 140]],
  [-20, [62, 140, 158]],
  [0, [84, 160, 166]],
];

export const TEMPERATURE: Stops = [
  [-40, [60, 30, 110]],
  [-25, [70, 80, 180]],
  [-10, [90, 150, 220]],
  [0, [190, 225, 240]],
  [10, [180, 220, 140]],
  [20, [245, 210, 90]],
  [28, [235, 120, 50]],
  [36, [180, 30, 40]],
];

/** Precipitation (mm/yr, log-ish stops). */
export const PRECIP: Stops = [
  [0, [120, 70, 30]],
  [150, [200, 150, 80]],
  [350, [235, 215, 140]],
  [700, [170, 210, 120]],
  [1200, [80, 170, 120]],
  [2000, [40, 130, 160]],
  [3200, [40, 70, 160]],
  [5000, [70, 30, 120]],
];

export const OCEAN_AGE: Stops = [
  [0, [240, 60, 50]],
  [20, [250, 160, 60]],
  [50, [240, 230, 110]],
  [90, [120, 200, 140]],
  [140, [70, 140, 200]],
  [200, [60, 70, 160]],
];

export const CONT_THICK: Stops = [
  [15, [120, 110, 100]],
  [30, [200, 180, 140]],
  [38, [220, 170, 100]],
  [50, [200, 110, 60]],
  [65, [150, 50, 50]],
];

export function hashColor(i: number): RGB {
  let h = Math.imul(i ^ 0x9e3779b9, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return [80 + (h & 127), 80 + ((h >>> 8) & 127), 80 + ((h >>> 16) & 127)];
}
