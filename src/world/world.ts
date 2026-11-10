import { buildMeshMap, buildSphereMesh, nearestCell, type MeshMap, type SphereMesh } from '../core/mesh';
import { seedFor } from '../core/rng';
import { type BiomeResult, computeBiomes } from './biomes';
import { type Climate, computeClimate, LAPSE, MONTHS } from './climate';
import { computeLithology, type Lithology, ROCK_NAMES } from './geology';
import { glacialErosion, ICE_AGE_SEA_LEVEL, ICE_ISOSTASY, iceAgeCooling, type IceSheet, simulateIce } from './glaciers';
import { computeHydrology, type Hydrology, oceanMask, slopeScale } from './hydrology';
import { aeolianDeposits, type Deposits, erode, type ErosionResult, finalizeDeposits, glacialDeposits, rockTraits } from './sediment';
import { computeRelief, type Relief, seaLevelForLandFraction } from './relief';
import { analyzeBoundaries, type Boundaries } from './tectonics/boundaries';
import { initialGeology } from './tectonics/init';
import { compactPlates, TectonicSim } from './tectonics/sim';
import { transferGeology } from './tectonics/transfer';
import { Geology, type Hotspot, type Plate, type WorldParams } from './types';
import { computeWaterBalance, type WaterBalance } from './waterBalance';

export type Progress = (stage: string, frac: number) => void;

/** Which part of the pipeline must be recomputed. */
export type DeriveFrom = 'tectonics' | 'relief' | 'climate';

type GeoArray = Int16Array | Uint8Array | Float32Array;

interface ArrayDiff {
  index: number;
  idx: Int32Array | null; // null = full copy
  before: GeoArray;
  after: GeoArray;
}

export interface EditRecord {
  label: string;
  arrays: ArrayDiff[];
  platesBefore: Plate[];
  platesAfter: Plate[];
  hotspotsBefore: Hotspot[];
  hotspotsAfter: Hotspot[];
  timeBefore: number;
  timeAfter: number;
  paramsBefore: WorldParams;
  paramsAfter: WorldParams;
}

export function coarseCellsFor(fineCells: number): number {
  return Math.round(Math.max(8000, Math.min(30000, fineCells / 2.5)));
}

const clonePlates = (ps: Plate[]): Plate[] => ps.map((p) => ({ ...p, color: [...p.color] as [number, number, number] }));
const cloneHotspots = (hs: Hotspot[]): Hotspot[] => hs.map((h) => ({ ...h }));

export class World {
  params: WorldParams;
  readonly fine: SphereMesh;
  readonly coarse: SphereMesh;
  /** Fine cell → containing coarse triangle (climate interpolation). */
  readonly f2c: MeshMap;
  /** Coarse cell → nearest fine cell (fallback when aggregating). */
  readonly c2fNearest: Int32Array;
  geo: Geology;
  historyLog: string[] = [];

  bnd!: Boundaries;
  relief!: Relief;
  /** Bedrock: rock type, how readily it erodes, and its age. */
  lithology!: Lithology;
  climate!: Climate;
  /** Final elevation (m) after erosion. */
  elev!: Float32Array;
  ocean!: Uint8Array;
  /** Monthly surface temperature (°C) and precipitation (mm), index c*12+m. */
  temp!: Float32Array;
  precip!: Float32Array;
  precipYear!: Float32Array;
  petYear!: Float32Array;
  /** Downwind maritime influence (1 at sea, decaying inland). */
  maritime!: Float32Array;
  /** Monthly land-surface water balance: evaporation, snow, soil, runoff. */
  water!: WaterBalance;
  /** Sediment laid down by rivers (m): silted-up basins, floodplains, shelves and deltas. */
  sediment!: Float32Array;
  /** Where the rock rivers and hillslopes wore away ended up (km³). */
  sedimentBudget!: ErosionResult['budget'];
  /** What covers the bedrock: river, glacial and wind-blown deposits, peat, weathered rock; deltas; the full sediment budget. */
  deposits!: Deposits;
  /** Bedrock (or sediment) surface under any ice (m); `elev` is the ice surface where there is ice. */
  bed!: Float32Array;
  /** Present-day glaciers, ice caps, ice sheets and ice shelves. */
  ice!: IceSheet;
  /** The last ice age: its ice thickness (m), how much it deepened the bed (m) and how hard it scoured (0–1). */
  iceAge!: { thick: Float32Array; erosion: Float32Array; scour: Float32Array };
  hydro!: Hydrology;
  biomes!: BiomeResult;
  slope!: Float32Array;
  timings: Record<string, number> = {};

  undoStack: EditRecord[] = [];
  redoStack: EditRecord[] = [];
  private snapshot: { geo: Geology; params: WorldParams } | null = null;

  private constructor(params: WorldParams, fine: SphereMesh, coarse: SphereMesh, geo: Geology) {
    this.params = { ...params };
    this.fine = fine;
    this.coarse = coarse;
    this.f2c = buildMeshMap(fine, coarse);
    this.c2fNearest = new Int32Array(coarse.n);
    for (let c = 0; c < coarse.n; c++) this.c2fNearest[c] = nearestCell(fine, coarse.x[c], coarse.y[c], coarse.z[c]);
    this.geo = geo;
  }

  static buildMeshes(params: WorldParams): { fine: SphereMesh; coarse: SphereMesh } {
    const fine = buildSphereMesh(params.cells, seedFor(params.seed, `mesh-${params.cells}`));
    const coarse = buildSphereMesh(coarseCellsFor(params.cells), seedFor(params.seed, 'coarse-mesh'));
    return { fine, coarse };
  }

  static generate(params: WorldParams, progress: Progress = () => {}): World {
    const t0 = performance.now();
    progress('Building sphere meshes', 0);
    const { fine, coarse } = World.buildMeshes(params);
    const tMesh = performance.now();
    progress('Initial continents and plates', 0.05);
    const g0 = initialGeology(coarse, params);
    progress('Simulating plate tectonics', 0.1);
    const sim = new TectonicSim(coarse, g0, {
      radiusKm: params.radiusKm,
      seed: seedFor(params.seed, 'sim'),
      drift: true,
      merge: true,
      split: true,
    });
    sim.run(params.historyMyr, (f) => progress('Simulating plate tectonics', 0.1 + 0.4 * f));
    const nameOf = (i: number) => g0.plates[i]?.name ?? `#${i}`;
    const log = [
      ...sim.splits.map((s) => `${(params.historyMyr - s.time).toFixed(0)} Ma: ${nameOf(s.from)} rifted, forming ${nameOf(s.into)}`),
      ...sim.merges.map((m) => `${(params.historyMyr - m.time).toFixed(0)} Ma: ${nameOf(m.from)} collided and welded onto ${nameOf(m.into)}`),
    ];
    compactPlates(coarse, g0);
    const tSim = performance.now();
    progress('Resampling geology', 0.5);
    const geo = transferGeology(coarse, g0, fine, seedFor(params.seed, 'transfer'));
    const world = new World(params, fine, coarse, geo);
    world.historyLog = log;
    world.timings.mesh = tMesh - t0;
    world.timings.history = tSim - tMesh;
    world.derive('tectonics', (s, f) => progress(s, 0.55 + 0.45 * f));
    world.timings.total = performance.now() - t0;
    return world;
  }

  /** Restore a world from saved geology (meshes are rebuilt deterministically). */
  static fromGeology(params: WorldParams, geo: Geology, progress: Progress = () => {}): World {
    const { fine, coarse } = World.buildMeshes(params);
    if (geo.n !== fine.n) throw new Error(`Saved world has ${geo.n} cells; mesh has ${fine.n}`);
    const world = new World(params, fine, coarse, geo);
    world.derive('tectonics', progress);
    return world;
  }

  /** Recompute every layer that depends on the geology (or climate parameters). */
  derive(from: DeriveFrom, progress: Progress = () => {}): void {
    const { fine, params } = this;
    const R = params.radiusKm;
    const T = (k: string, t: number) => (this.timings[k] = performance.now() - t);
    let t = performance.now();
    if (from === 'tectonics' || !this.bnd) {
      progress('Analyzing plate boundaries', 0);
      this.bnd = analyzeBoundaries(fine, this.geo, R);
      T('boundaries', t);
    }
    if (from !== 'climate' || !this.relief) {
      t = performance.now();
      progress('Isostasy and relief', 0.08);
      this.relief = computeRelief(fine, this.geo, this.bnd, params);
      T('relief', t);
      t = performance.now();
      this.lithology = computeLithology(fine, this.coarse, this.f2c, this.geo, this.bnd, this.relief, params);
      T('lithology', t);
    }
    const base = this.relief.elev;
    const ocean0 = oceanMask(fine, base);

    t = performance.now();
    progress('Climate: radiation, winds, rain', 0.15);
    const { elevC, landC } = this.aggregateToCoarse(base, ocean0);
    this.climate = computeClimate(this.coarse, elevC, landC, params);
    T('climate', t);

    // Interpolate climate onto the fine mesh.
    const n = fine.n;
    const tempSL = new Float32Array(n * MONTHS);
    const precip = new Float32Array(n * MONTHS);
    const maritime = new Float32Array(n);
    const { idx, w } = this.f2c;
    const cT = this.climate.tempSL;
    const cP = this.climate.precip;
    const cM = this.climate.maritime;
    for (let c = 0; c < n; c++) {
      const ia = idx[3 * c];
      const ib = idx[3 * c + 1];
      const id = idx[3 * c + 2];
      const a = ia * MONTHS;
      const b = ib * MONTHS;
      const d = id * MONTHS;
      const wa = w[3 * c];
      const wb = w[3 * c + 1];
      const wd = w[3 * c + 2];
      for (let m = 0; m < MONTHS; m++) {
        tempSL[c * MONTHS + m] = cT[a + m] * wa + cT[b + m] * wb + cT[d + m] * wd;
        precip[c * MONTHS + m] = cP[a + m] * wa + cP[b + m] * wb + cP[d + m] * wd;
      }
      maritime[c] = Math.min(1, cM[ia] * wa + cM[ib] * wb + cM[id] * wd);
    }
    this.precip = precip;
    this.maritime = maritime;
    this.precipYear = new Float32Array(n);
    for (let c = 0; c < n; c++) {
      let s = 0;
      for (let m = 0; m < MONTHS; m++) s += precip[c * MONTHS + m];
      this.precipYear[c] = s;
    }

    // Erosion is driven by the runoff of the uneroded land; its sediment fills
    // basins, builds floodplains and deltas.
    t = performance.now();
    progress('Water balance and erosion', 0.55);
    this.temp = this.surfaceTemperature(tempSL, base, ocean0);
    const wb0 = computeWaterBalance(fine, this.temp, precip, ocean0, this.computeSlope(base), maritime, params.axialTiltDeg);
    const bed = Float32Array.from(base);
    const tempYear = new Float32Array(n);
    for (let c = 0; c < n; c++) {
      let s = 0;
      for (let m = 0; m < MONTHS; m++) s += this.temp[c * MONTHS + m];
      tempYear[c] = s / MONTHS;
    }
    const lith = this.lithology;
    const eroded = erode(fine, bed, ocean0, wb0.runoffYear, params.erosion, R, {
      tempYear,
      precipYear: this.precipYear,
      erodibility: lith.erodibility,
      ...rockTraits(lith.rock, ROCK_NAMES),
    });
    this.sediment = eroded.deposit;
    this.sedimentBudget = eroded.budget;
    T('erosion', t);

    // The last ice age (a colder, drier planet with seas 120 m lower) carved
    // troughs and fjords and scoured the lowlands; today's ice sits on that bed.
    t = performance.now();
    progress('Ice ages and glaciers', 0.72);
    const iceAgeClimate = { tempSL, precip, cool: iceAgeCooling(fine), precipFactor: 0.8 };
    const past = simulateIce(fine, bed, oceanMask(fine, bed), iceAgeClimate, ICE_AGE_SEA_LEVEL, R, false);
    const legacy = glacialErosion(fine, bed, past, iceAgeClimate, R);
    this.iceAge = { thick: past.thick, ...legacy };
    // What the ice ground away became till, moraines and outwash; the wind
    // then reworked bare desert and outwash ground into dunes and loess.
    const tDep = performance.now();
    const glacial = glacialDeposits(fine, bed, past, legacy, eroded, R);
    const wind = aeolianDeposits(fine, this.coarse, this.f2c, this.climate, bed, ocean0, this.precipYear, wb0.petYear, tempYear, eroded, glacial, R);
    const tDeposits = performance.now() - tDep;
    this.ice = simulateIce(fine, bed, oceanMask(fine, bed), { tempSL, precip, cool: null, precipFactor: 1 }, 0, R, true);
    this.bed = bed;
    const elev = new Float32Array(n);
    for (let c = 0; c < n; c++) elev[c] = bed[c] + (1 - ICE_ISOSTASY) * this.ice.thick[c];
    this.elev = elev;
    this.ocean = oceanMask(fine, elev);
    T('glaciers', t);

    t = performance.now();
    progress('Rivers and lakes', 0.8);
    this.temp = this.surfaceTemperature(tempSL, elev, this.ocean);
    this.slope = this.computeSlope(elev);
    this.water = computeWaterBalance(fine, this.temp, precip, this.ocean, this.slope, maritime, params.axialTiltDeg, this.ice.cover);
    this.petYear = this.water.petYear;
    this.hydro = computeHydrology(fine, elev, this.ocean, this.water, precip, this.temp, this.slope, this.relief.rough, R);
    T('hydrology', t);
    t = performance.now();
    this.deposits = finalizeDeposits(fine, this.ocean, eroded, glacial, wind, this.hydro, tempYear, this.precipYear, this.petYear, legacy.scour);
    this.timings.deposits = tDeposits + performance.now() - t;

    t = performance.now();
    progress('Biomes', 0.92);
    this.biomes = computeBiomes(fine, elev, this.temp, precip, this.ocean, this.hydro, this.water);
    T('biomes', t);
    progress('Done', 1);
  }

  /**
   * Fast partial update while a brush stroke is in progress: boundaries and
   * relief only; climate, rivers and biomes keep their previous values. With
   * `base` (the eroded elevation and raw relief from before the stroke), only
   * the relief's change is applied to the eroded surface, so untouched areas
   * keep their valleys.
   */
  derivePreview(base?: { elev: Float32Array; relief: Float32Array }): void {
    this.bnd = analyzeBoundaries(this.fine, this.geo, this.params.radiusKm);
    this.relief = computeRelief(this.fine, this.geo, this.bnd, this.params);
    if (base) {
      const e = new Float32Array(this.fine.n);
      for (let c = 0; c < e.length; c++) e[c] = base.elev[c] + (this.relief.elev[c] - base.relief[c]);
      this.elev = e;
    } else this.elev = this.relief.elev;
    this.ocean = oceanMask(this.fine, this.elev);
  }

  private aggregateToCoarse(elev: Float32Array, ocean: Uint8Array): { elevC: Float32Array; landC: Float32Array } {
    const { fine, coarse } = this;
    const sumA = new Float64Array(coarse.n);
    const sumE = new Float64Array(coarse.n);
    const sumL = new Float64Array(coarse.n);
    const near = this.f2c.nearest;
    for (let c = 0; c < fine.n; c++) {
      const k = near[c];
      const a = fine.area[c];
      sumA[k] += a;
      sumE[k] += a * Math.max(0, elev[c]);
      if (!ocean[c]) sumL[k] += a;
    }
    const elevC = new Float32Array(coarse.n);
    const landC = new Float32Array(coarse.n);
    for (let k = 0; k < coarse.n; k++) {
      if (sumA[k] > 0) {
        elevC[k] = sumE[k] / sumA[k];
        landC[k] = sumL[k] / sumA[k];
      } else {
        const f = this.c2fNearest[k];
        elevC[k] = Math.max(0, elev[f]);
        landC[k] = ocean[f] ? 0 : 1;
      }
    }
    return { elevC, landC };
  }

  private surfaceTemperature(tempSL: Float32Array, elev: Float32Array, ocean: Uint8Array): Float32Array {
    const n = this.fine.n;
    const out = new Float32Array(n * MONTHS);
    for (let c = 0; c < n; c++) {
      const drop = ocean[c] ? 0 : (LAPSE * Math.max(0, elev[c])) / 1000;
      for (let m = 0; m < MONTHS; m++) out[c * MONTHS + m] = tempSL[c * MONTHS + m] - drop;
    }
    return out;
  }

  /**
   * Steepest slope to a neighbor, as measured between cells of the reference
   * detail: finer cells resolve more of the terrain's fractal relief, so
   * their raw slopes run steeper (∝ spacing^−0.45) for the same landscape.
   */
  private computeSlope(elev: Float32Array): Float32Array {
    const { n, nbrStart, nbr, nbrDist } = this.fine;
    const R = this.params.radiusKm * 1000;
    const scale = slopeScale(n);
    const out = new Float32Array(n);
    for (let c = 0; c < n; c++) {
      let s = 0;
      for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
        s = Math.max(s, Math.abs(elev[nbr[k]] - elev[c]) / (nbrDist[k] * R));
      }
      out[c] = s * scale;
    }
    return out;
  }

  // ---------------------------------------------------------------- statistics

  landFraction(): number {
    let a = 0;
    for (let c = 0; c < this.fine.n; c++) if (!this.ocean[c]) a += this.fine.area[c];
    return a / (4 * Math.PI);
  }

  /** Sea level (m) that would leave the requested share of the surface dry. */
  seaLevelFor(landFraction: number): number {
    return seaLevelForLandFraction(this.fine, this.relief.raw, landFraction);
  }

  plateStats(): { id: number; name: string; color: [number, number, number]; areaKm2: number; continental: number; speed: number }[] {
    const { fine, geo, params } = this;
    const R = params.radiusKm;
    const area = new Float64Array(geo.plates.length);
    const cont = new Float64Array(geo.plates.length);
    for (let c = 0; c < fine.n; c++) {
      area[geo.plate[c]] += fine.area[c];
      if (geo.ctype[c]) cont[geo.plate[c]] += fine.area[c];
    }
    return geo.plates.map((p, i) => ({
      id: i,
      name: p.name,
      color: p.color,
      areaKm2: area[i] * R * R,
      continental: area[i] > 0 ? cont[i] / area[i] : 0,
      speed: Math.hypot(p.wx, p.wy, p.wz) * R,
    }));
  }

  // ---------------------------------------------------------------- undo/redo

  beginEdit(): void {
    if (!this.snapshot) this.snapshot = { geo: this.geo.clone(), params: { ...this.params } };
  }

  get editing(): boolean {
    return this.snapshot !== null;
  }

  /** Record the change since beginEdit() as one undoable step. */
  commitEdit(label: string): EditRecord | null {
    const snap = this.snapshot;
    this.snapshot = null;
    if (!snap) return null;
    const before = snap.geo.arrays();
    const after = this.geo.arrays();
    const arrays: ArrayDiff[] = [];
    const n = this.geo.n;
    for (let a = 0; a < before.length; a++) {
      const b0 = before[a];
      const a0 = after[a];
      const changed: number[] = [];
      for (let i = 0; i < n; i++) if (b0[i] !== a0[i]) changed.push(i);
      if (changed.length === 0) continue;
      if (changed.length > n / 3) {
        arrays.push({ index: a, idx: null, before: b0.slice(), after: a0.slice() });
      } else {
        const idx = Int32Array.from(changed);
        const ctor = b0.constructor as { new (len: number): GeoArray };
        const bv = new ctor(idx.length);
        const av = new ctor(idx.length);
        for (let i = 0; i < idx.length; i++) {
          bv[i] = b0[idx[i]];
          av[i] = a0[idx[i]];
        }
        arrays.push({ index: a, idx, before: bv, after: av });
      }
    }
    const rec: EditRecord = {
      label,
      arrays,
      platesBefore: clonePlates(snap.geo.plates),
      platesAfter: clonePlates(this.geo.plates),
      hotspotsBefore: cloneHotspots(snap.geo.hotspots),
      hotspotsAfter: cloneHotspots(this.geo.hotspots),
      timeBefore: snap.geo.time,
      timeAfter: this.geo.time,
      paramsBefore: snap.params,
      paramsAfter: { ...this.params },
    };
    this.undoStack.push(rec);
    if (this.undoStack.length > 60) this.undoStack.shift();
    this.redoStack = [];
    return rec;
  }

  /** Abandon an edit in progress, restoring the snapshot. */
  cancelEdit(): void {
    if (!this.snapshot) return;
    this.geo = this.snapshot.geo;
    this.params = this.snapshot.params;
    this.snapshot = null;
  }

  private applyRecord(rec: EditRecord, dir: 'before' | 'after'): void {
    const arrs = this.geo.arrays();
    for (const d of rec.arrays) {
      const target = arrs[d.index];
      const src = d[dir];
      if (d.idx === null) target.set(src as never);
      else for (let i = 0; i < d.idx.length; i++) target[d.idx[i]] = src[i];
    }
    this.geo.plates = clonePlates(dir === 'before' ? rec.platesBefore : rec.platesAfter);
    this.geo.hotspots = cloneHotspots(dir === 'before' ? rec.hotspotsBefore : rec.hotspotsAfter);
    this.geo.time = dir === 'before' ? rec.timeBefore : rec.timeAfter;
    this.params = { ...(dir === 'before' ? rec.paramsBefore : rec.paramsAfter) };
  }

  undo(): EditRecord | null {
    const rec = this.undoStack.pop();
    if (!rec) return null;
    this.applyRecord(rec, 'before');
    this.redoStack.push(rec);
    return rec;
  }

  redo(): EditRecord | null {
    const rec = this.redoStack.pop();
    if (!rec) return null;
    this.applyRecord(rec, 'after');
    this.undoStack.push(rec);
    return rec;
  }
}
