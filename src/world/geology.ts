import { clamp, smoothstep } from '../core/math';
import type { MeshMap, SphereMesh } from '../core/mesh';
import { Simplex3 } from '../core/noise';
import { seedFor } from '../core/rng';
import type { Relief } from './relief';
import type { Boundaries } from './tectonics/boundaries';
import type { Geology, WorldParams } from './types';

/**
 * Bedrock classes, read from each place's plate history: how old its crust
 * is, when it was last squeezed into mountains, its volcanoes, and the plate
 * boundaries around it today.
 */
export const enum Rock {
  OceanBasalt = 0, DeepSea, ArcVolcanic, Basalt, Granite, Basement, Metamorphic, Ophiolite, Folded, Sandstone, Shale, Limestone,
  Evaporite,
}

/**
 * Per rock: map name, inspector description, color, and how readily rivers cut
 * it (a multiplier on the stream-power erodibility, 1 = the average rock the
 * erosion law was calibrated for). Crystalline rock holds up ridges; shale and
 * salt wear down into vales and basins.
 */
export const ROCK_INFO: { name: string; long: string; color: [number, number, number]; erodibility: number }[] = [
  { name: 'Oceanic basalt', long: 'oceanic basalt: pillow lava from the spreading ridge', color: [62, 84, 112], erodibility: 0.5 },
  { name: 'Deep-sea sediment', long: 'deep-sea clay and ooze over oceanic basalt', color: [98, 116, 140], erodibility: 2.5 },
  { name: 'Arc volcanics', long: 'arc volcanics: andesite lava and ash', color: [222, 108, 104], erodibility: 0.9 },
  { name: 'Basalt lava', long: 'basalt lava flows', color: [126, 64, 82], erodibility: 0.45 },
  { name: 'Granite', long: 'granite: a magma chamber cooled at depth, now exhumed', color: [240, 132, 160], erodibility: 0.35 },
  { name: 'Shield basement', long: 'granite–gneiss basement', color: [196, 152, 184], erodibility: 0.3 },
  { name: 'Schist & gneiss', long: "schist and gneiss from a mountain belt's core", color: [146, 118, 198], erodibility: 0.5 },
  { name: 'Ophiolite', long: 'ophiolite: ocean floor thrust onto land along a suture', color: [34, 130, 98], erodibility: 0.8 },
  { name: 'Folded strata', long: 'folded and thrust sedimentary rocks', color: [148, 168, 100], erodibility: 1.3 },
  { name: 'Sandstone', long: 'sandstone and conglomerate', color: [230, 198, 118], erodibility: 0.9 },
  { name: 'Shale', long: 'shale and mudstone', color: [126, 144, 164], erodibility: 2.5 },
  { name: 'Limestone', long: 'limestone and dolomite', color: [162, 208, 222], erodibility: 0.8 },
  { name: 'Evaporite', long: 'evaporites: rock salt and gypsum', color: [244, 240, 220], erodibility: 3 },
];
export const ROCK_NAMES = ROCK_INFO.map((r) => r.name);
export const ROCK_COLORS = ROCK_INFO.map((r) => r.color);

export interface Lithology {
  rock: Uint8Array;
  /** Multiplier on the stream-power erodibility K (1 = the rock the erosion law was calibrated for). */
  erodibility: Float32Array;
  /** Myr since the rock formed, or was last metamorphosed. */
  rockAge: Float32Array;
}

/** Distances (km) from the boundary analysis beyond this mean "no such boundary". */
const FAR = 1e8;
/** Smoothed elevation per stratigraphic belt (m): basins expose young strata in their centers, older ones toward their rims. */
const BAND_M = 140;

/**
 * Bedrock everywhere. Active boundaries set the rock near them (volcanic arcs
 * with batholiths behind, collision belts with metamorphic cores, ophiolite
 * sutures, fold-and-thrust belts and molasse forelands, rifts with red beds,
 * salt and basalt). Away from them, the crust's history decides: crust born
 * in island arcs is a mosaic of volcanics, granite and schist; eroded old
 * mountain belts expose their metamorphic roots; ancient cratons are bare
 * shield or buried under platform sediments, whose belts of sandstone, shale
 * and limestone ring the basins (carbonate in the warm latitudes, salt in the
 * dry belts). Deep ocean floor is basalt under pelagic ooze.
 */
export function computeLithology(
  mesh: SphereMesh,
  coarse: SphereMesh,
  f2c: MeshMap,
  geo: Geology,
  bnd: Boundaries,
  relief: Relief,
  params: WorldParams,
): Lithology {
  const { n, x, y, z, lat, nbrStart, nbr } = mesh;
  const R = params.radiusKm;
  const noise = new Simplex3(seedFor(params.seed, 'lithology'));
  const { ctype, thick, age, orogeny, volc, volcAge } = geo;
  const { dArc, arcCont, dColl, dDiv } = bnd;
  const raw = relief.raw;
  const level = regionalLevel(mesh, coarse, f2c, raw, R);
  const spacingKm = mesh.spacing * R;
  const sutureKm = Math.max(15, 0.55 * spacingKm);
  const riftAxisKm = Math.max(30, 0.6 * spacingKm);
  // Folded beds alternate hard and soft every few cells (never finer than the mesh resolves).
  const foldKm = Math.max(12, 0.8 * spacingKm);
  // Flood-basalt provinces around plumes under continents: ragged, up to ~400 km beyond the plume head.
  const hot = geo.hotspots
    .filter((h) => h.active)
    .map((h) => ({ x: h.x, y: h.y, z: h.z, reach: h.radius * R + 220, cos: Math.cos(h.radius + 420 / R) }));

  const rock = new Uint8Array(n);
  const erodibility = new Float32Array(n);
  const rockAge = new Float32Array(n);

  for (let c = 0; c < n; c++) {
    const px = x[c];
    const py = y[c];
    const pz = z[c];
    const t = thick[c];
    const a = age[c];
    const o = orogeny[c];
    const dA = dArc[c];
    let r: Rock;
    let ra = a;
    let kmul = 1;

    if (!ctype[c]) {
      if (relief.reef[c] === 1) {
        r = Rock.Limestone;
        ra = volcAge[c];
      } else if (volc[c] > 0.15) {
        r = dA < 450 ? Rock.ArcVolcanic : Rock.Basalt;
        ra = volcAge[c];
      } else if (dA > 70 && dA < 320 && t > 10) {
        r = Rock.ArcVolcanic;
        ra = Math.min(a, Math.max(0.5, o));
      } else if (t > 12) {
        r = Rock.Basalt;
        ra = a;
      } else {
        r = a < 25 ? Rock.OceanBasalt : Rock.DeepSea;
        ra = a;
      }
    } else {
      const e = raw[c];
      const S = level[c];
      const latDeg = (Math.abs(lat[c]) * 180) / Math.PI;
      // Province-scale (~400 km), detail (~100 km) and continental-scale noise.
      const nm = noise.fbm(px * 16, py * 16, pz * 16, 2);
      const nf = noise.noise(px * 60 + 3.3, py * 60, pz * 60);
      const nl = noise.fbm(px * 2.5 + 11, py * 2.5, pz * 2.5, 2);
      kmul = Math.exp(0.15 * nf);
      const dC = dColl[c];
      const dD = dDiv[c];
      const warm = smoothstep(36, 22, latDeg + 7 * nl);
      const arid = smoothstep(10, 18, latDeg) * smoothstep(38, 28, latDeg + 5 * nl);
      // Folded strata: deposited well before the squeeze that folded them.
      const foldedAge = Math.min(a, Math.max(1, o) + 60 + 150 * (0.5 + 0.5 * nm));

      // Platform cover: belts of sandstone, shale and limestone ringing the basins.
      const platform = (): Rock => {
        if (e < 0) {
          ra = 2 + 40 * (0.5 + 0.5 * nm);
          return warm > 0.5 ? Rock.Limestone : nf > 0.3 ? Rock.Sandstone : Rock.Shale;
        }
        const k = Math.floor((S + 1000) / BAND_M + 1.3 * nm);
        const phase = ((k % 3) + 3) % 3;
        ra = clamp(25 + 0.5 * (S + 300) + 60 * nm, 8, 520);
        if (phase === 1) return arid > 0.5 && nf > 0.45 && e < 400 ? Rock.Evaporite : Rock.Shale;
        return warm > (phase === 0 ? 0.6 : 0.25) ? Rock.Limestone : Rock.Sandstone;
      };

      // Away from active boundaries: the crust's own history.
      const background = (): Rock => {
        if (a < 300 && o <= a + 1) {
          // Crust born in island arcs, swept together into the continent.
          if (e < 120 && S < 150) return platform();
          if (o < 30) {
            ra = Math.max(0.5, o);
            return Rock.ArcVolcanic;
          }
          ra = nm > 0.25 ? o + 3 : nm < -0.3 ? o : Math.min(a, o + 10);
          return nm > 0.25 ? Rock.Granite : nm < -0.3 ? Rock.Metamorphic : Rock.ArcVolcanic;
        }
        if (o < 1000 && t > 50 && nf > 0.35 && isCrest(c)) {
          // The seam of an old collision.
          ra = Math.max(1, o);
          return Rock.Ophiolite;
        }
        if (o < 200) {
          // A young mountain belt (or one being raised now): folded cover, and the core's metamorphic roots once it is thick.
          if (t > 37) ra = t > 48 ? Math.max(1, o) : foldedAge;
          return t > 48 ? (nm > 0.4 ? Rock.Granite : Rock.Metamorphic) : t > 37 ? Rock.Folded : platform();
        }
        if (o < 1000) {
          // An old belt worn down to its roots, partly buried by younger strata.
          if (0.6 * nl + (S - 300) / 700 + 0.5 * smoothstep(37, 48, t) <= 0.32) return platform();
          ra = nm > 0.35 ? o + 10 : o;
          return nm > 0.35 ? Rock.Granite : Rock.Metamorphic;
        }
        // Ancient craton: bare shield where it stands high, platform cover where it sagged.
        if (nl + (S - 450) / 900 + 0.04 * (t - 36) <= 0.26) return platform();
        ra = nm > 0.5 ? 0.6 * a : a;
        return nm > 0.5 ? Rock.Granite : Rock.Basement;
      };

      let hotspot = false;
      for (const h of hot) {
        const dot = px * h.x + py * h.y + pz * h.z;
        if (dot > h.cos && Math.acos(Math.min(1, dot)) * R < h.reach + 200 * nm + 60 * nf) hotspot = true;
      }

      if (volc[c] > 0.1) {
        r = dA < 450 ? Rock.ArcVolcanic : Rock.Basalt;
        ra = volcAge[c];
      } else if (hotspot) {
        // A plume under the continent floods it with basalt.
        r = Rock.Basalt;
        ra = 0.5 + 4 * (0.5 + 0.5 * nf);
      } else if (dC < 820) {
        if (dC <= sutureKm && nf > 0.1) {
          r = Rock.Ophiolite;
          ra = Math.max(1, o);
        } else if (dC < 220 && t > 46) {
          r = nm > 0.45 ? Rock.Granite : Rock.Metamorphic;
          ra = Math.max(1, o);
        } else if ((dC < 480 && t > 39) || (o < 5 && t > 37)) {
          r = Rock.Folded;
          ra = foldedAge;
        } else if (e < 1500 && t < 42) {
          // Molasse: debris shed off the rising belt into its foreland basin.
          r = dC > 650 ? Rock.Shale : Rock.Sandstone;
          ra = Math.max(1, o) + 2;
        } else r = background();
      } else if (dA < 750 && arcCont[c]) {
        if (dA < 70 || (dA < 120 && o < 5 && t > 37)) {
          r = Rock.Folded;
          ra = 5 + 20 * (0.5 + 0.5 * nm);
        } else if (dA < 120) {
          r = nm > 0 ? Rock.Sandstone : Rock.Shale;
          ra = 2 + 30 * (0.5 + 0.5 * nf);
        } else if (dA < 290) {
          r = nm > 0.35 ? Rock.Granite : Rock.ArcVolcanic;
          ra = Math.max(0.5, o) + (r === Rock.Granite ? 5 : 0);
        } else if (dA < 430 && t > 38) {
          r = Rock.Granite;
          ra = Math.max(1, o) + 8;
        } else if ((dA < 650 && t > 42) || (o < 5 && t > 37)) {
          r = Rock.Folded;
          ra = foldedAge;
        } else if (e < 1200 && t < 42 && t > 34) {
          r = Rock.Sandstone;
          ra = Math.max(1, o) + 2;
        } else r = background();
      } else if (dA > 70 && dA < 320) {
        r = Rock.ArcVolcanic;
        ra = Math.max(0.5, o);
      } else if (dD < 220 && t < 36) {
        if (dD < riftAxisKm && nf > -0.1) {
          r = Rock.Basalt;
          ra = 0.5 + 5 * (0.5 + 0.5 * nm);
        } else {
          r = arid > 0.5 && nm > 0.25 ? Rock.Evaporite : nm < -0.25 ? Rock.Shale : Rock.Sandstone;
          ra = 2 + 30 * (0.5 + 0.5 * nf);
        }
      } else r = background();

      if (r === Rock.Folded) {
        // Alternating hard and soft beds run along the belt: ridges and vales.
        const q = dC < FAR ? dC : dA < FAR ? dA : 25 * t;
        kmul = Math.sin(q / foldKm + 2.5 * nm) > 0 ? 0.5 : 1.6;
      }
    }

    rock[c] = r;
    rockAge[c] = ra;
    erodibility[c] = clamp(ROCK_INFO[r].erodibility * kmul, 0.25, 3.5);
  }
  return { rock, erodibility, rockAge };

  /** Thickest crust among its neighbors: the axis of a mountain belt. */
  function isCrest(c: number): boolean {
    for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) if (thick[nbr[k]] > thick[c] + 0.3) return false;
    return true;
  }
}

/**
 * Elevation averaged over a few hundred km (m): low where the crust has sagged
 * into sedimentary basins (and toward subsiding margins), high over arches
 * and uplands. Smoothed on the coarse mesh, then interpolated back.
 */
function regionalLevel(mesh: SphereMesh, coarse: SphereMesh, f2c: MeshMap, raw: Float32Array, R: number): Float32Array {
  const m = coarse.n;
  const sum = new Float64Array(m);
  const wsum = new Float64Array(m);
  for (let c = 0; c < mesh.n; c++) {
    const k = f2c.nearest[c];
    sum[k] += mesh.area[c] * Math.max(-400, raw[c]);
    wsum[k] += mesh.area[c];
  }
  let v = new Float32Array(m);
  let def = new Uint8Array(m);
  for (let k = 0; k < m; k++) {
    if (wsum[k] > 0) {
      v[k] = sum[k] / wsum[k];
      def[k] = 1;
    }
  }
  let nv = new Float32Array(m);
  let nd = new Uint8Array(m);
  const iters = Math.max(2, Math.round((450 / (coarse.spacing * R)) ** 2));
  for (let it = 0; it < iters; it++) {
    for (let k = 0; k < m; k++) {
      let s = def[k] ? v[k] : 0;
      let w = def[k];
      for (let j = coarse.nbrStart[k], e = coarse.nbrStart[k + 1]; j < e; j++) {
        const q = coarse.nbr[j];
        if (def[q]) {
          s += v[q];
          w++;
        }
      }
      nv[k] = w > 0 ? s / w : 0;
      nd[k] = w > 0 ? 1 : 0;
    }
    [v, nv] = [nv, v];
    [def, nd] = [nd, def];
  }
  const out = new Float32Array(mesh.n);
  const { idx, w } = f2c;
  for (let c = 0; c < mesh.n; c++) out[c] = v[idx[3 * c]] * w[3 * c] + v[idx[3 * c + 1]] * w[3 * c + 1] + v[idx[3 * c + 2]] * w[3 * c + 2];
  return out;
}
