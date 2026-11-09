/**
 * Elevation from crustal structure.
 *
 * Continents float by Airy isostasy: each km of extra crust raises the surface
 * by ~150 m in air (ρc 2.8, ρm 3.3) or ~220 m when water fills the space.
 * Oceanic crust sits at a depth set by how long the lithosphere has cooled
 * (GDH1 age–depth relation, Stein & Stein 1992).
 */

/** Continental crust thickness (km) whose surface sits at raw sea level. */
export const H_SEA = 33;
export const CONT_ABOVE = 150; // m per km of crust above H_SEA
export const CONT_BELOW = 220; // m per km below H_SEA (water-loaded)
export const OCEAN_PLATEAU = 150; // m per km of oceanic crust beyond 7 km

/** Seafloor depth (m, positive down) for lithosphere of a given age (Myr). */
export function oceanDepth(age: number): number {
  const t = Math.max(0, age);
  return t < 20 ? 2600 + 365 * Math.sqrt(t) : 5651 - 2473 * Math.exp(-0.0278 * t);
}

/** Inverse of oceanDepth (Myr) for depths within its range. */
export function ageForDepth(depth: number): number {
  if (depth <= 2600) return 0;
  if (depth < 4232) return ((depth - 2600) / 365) ** 2;
  const r = (5651 - Math.min(depth, 5640)) / 2473;
  return -Math.log(r) / 0.0278;
}

/** Elevation (m) before the sea-level offset and shelf sediment infill. */
export function rawElevation(ctype: number, thick: number, age: number, volc: number): number {
  if (ctype === 1) {
    const e = thick >= H_SEA ? (thick - H_SEA) * CONT_ABOVE : (thick - H_SEA) * CONT_BELOW;
    return e + volc * 800;
  }
  return -oceanDepth(age) + Math.max(0, thick - 7) * OCEAN_PLATEAU + volc * 1000;
}

/** Continental thickness (km) whose raw elevation equals eRaw (m). */
export function thickForRaw(eRaw: number): number {
  return eRaw >= 0 ? H_SEA + eRaw / CONT_ABOVE : H_SEA + eRaw / CONT_BELOW;
}

/**
 * Submerged continental crust collects sediment up to wave base, so the raw
 * isostatic depth is compressed into a shallow shelf, then a steeper slope.
 */
export const SHELF_RAW = 1000; // raw depth (m) mapped to the shelf break
export const SHELF_BREAK = 200; // shelf-break depth (m)

export function shelfCompress(e: number): number {
  if (e >= 0) return e;
  if (e > -SHELF_RAW) return -SHELF_BREAK * Math.pow(-e / SHELF_RAW, 1.5);
  return -SHELF_BREAK + (e + SHELF_RAW) * 1.2;
}

export function shelfExpand(e: number): number {
  if (e >= 0) return e;
  if (e > -SHELF_BREAK) return -SHELF_RAW * Math.pow(-e / SHELF_BREAK, 1 / 1.5);
  return (e + SHELF_BREAK) / 1.2 - SHELF_RAW;
}
