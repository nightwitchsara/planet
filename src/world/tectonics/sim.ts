import { MinHeap } from '../../core/heap';
import { normalize3, rotationFromOmega } from '../../core/math';
import { nearestCell, type SphereMesh } from '../../core/mesh';
import { Simplex3 } from '../../core/noise';
import { Rng } from '../../core/rng';
import { type Geology, plateColor, plateName } from '../types';
import { rawElevation } from './isostasy';

export interface SimConfig {
  radiusKm: number;
  seed: number;
  /** Slowly random-walk plate motions (real plates change course over tens of Myr). */
  drift: boolean;
  /** Weld plates together after sustained continental collision. */
  merge: boolean;
  /** Periodically rift the largest plate in two (new ocean basins mid-history). */
  split: boolean;
}

/** Myr between attempts to rift the largest plate. */
const SPLIT_INTERVAL = 30;
/** Plates larger than this share of the sphere are candidates for rifting. */
const SPLIT_MIN_SHARE = 0.1;

/** Crustal growth at the arc axis (km/Myr) above fast (≥ 50 mm/yr) subduction. */
const ARC_RATE = 0.35;
/**
 * Island-arc crust stops thickening around this thickness (km): beyond it the
 * dense lower arc root founders into the mantle, so island arcs stay chains of
 * volcanic islands instead of growing into continuous mountain walls.
 */
const ARC_MAX_ISLAND = 35;
/** Old continental crust above a trench also shortens and thickens (an Andean-type margin, km). */
const ARC_MAX_ANDEAN = 70;
/** Continental crust older than this (Myr) is a true continental margin, not juvenile arc crust. */
const OLD_CRUST = 300;
/** Distance (km) of the volcanic front behind the trench, and the arc's half-width. */
const ARC_AXIS = 140;
const ARC_WIDTH = 90;
const ARC_REACH = ARC_AXIS + 2.5 * ARC_WIDTH;
/** E-folding time of thickened crust (erosion plus isostatic rebound). */
const OROGEN_TAU = 300;
/** Subsidence/erosion of inactive volcanic edifices. */
const VOLC_TAU = 45;
const VOLC_SUBAERIAL_TAU = 6;
const MAX_THICK = 80;
/** Share of the under-thrust crust that thickens the collision zone. */
const COLLISION_CONSERVE = 1;
/** New crust in a continental rift is this share of its walls' thickness; the walls thin to supply it. */
const STRETCH = 0.6;
/** Rifted continental crust thinner than this (km, mean of the rift walls) breaks up into ocean floor. */
const BREAKUP = 24;
/** Diffusivity (km²/Myr) of the lower-crust flow that spreads over-thickened crust. */
const FLOW_K = 1000;
/** Crust flows laterally only once it is this thick (km): hot, plateau-forming orogens like Tibet. */
const FLOW_ONSET = 50;
/** Accumulated collision area (sr) that welds two plates. */
const MERGE_AREA = 0.002 * 4 * Math.PI;

/** Continental crust bookkeeping (area in sr, volume in km·sr) for diagnostics. */
export interface CrustBudget {
  /** Continental area consumed by collisions (its thickness goes into the orogen). */
  collided: number;
  /** Continental area created by rifting (volume taken from the rift walls). */
  rifted: number;
  /** Oceanic area turned continental by arc magmatism. */
  arcs: number;
  /** Volume removed above the thickness cap after spreading. */
  capped: number;
}

/**
 * Kinematic plate tectonics on a fixed spherical mesh, as particles in cells.
 * The crust is a set of parcels that ride their plates at their true
 * positions; a cell only shows the nearest parcel. Inside a plate nothing is
 * ever resampled, so coastlines and mountain belts keep their shape however
 * far they travel. Where parcels of converging plates meet, the less buoyant
 * one is consumed (continents never subduct, younger oceanic lithosphere
 * overrides older) by subduction or collision; cells left without crust open
 * as ridges or rifts. Hotspots are fixed in the mantle frame, so plates carry
 * their volcanoes away as chains.
 */
export class TectonicSim {
  // Parcels.
  private cap = 0;
  private P = 0;
  private dead = 0;
  private alive!: Uint8Array;
  private qx!: Float64Array;
  private qy!: Float64Array;
  private qz!: Float64Array;
  private pPlate!: Int16Array;
  private pCtype!: Uint8Array;
  private pThick!: Float32Array;
  private pAge!: Float32Array;
  private pOrog!: Float32Array;
  private pVolc!: Float32Array;
  private pVolcAge!: Float32Array;
  private pArea!: Float32Array;
  private pCell!: Int32Array;
  // Parcels per cell (CSR), the plate of a cell's parcels (-1 none, -2 several) and the parcel each cell shows.
  private head: Int32Array;
  private cursor: Int32Array;
  private list!: Int32Array;
  private binnedP = 0;
  private cellPlate: Int16Array;
  private show: Int32Array;
  private work: Float32Array;
  private mark: Int32Array;
  private markId = 0;
  private gap: Uint8Array;
  private hotI: Float32Array;
  private hotS: Float32Array;
  private hotCells: number[] = [];
  private hotCache: number[] = [];
  private stamp: Int32Array;
  private stampId = 0;
  private queue: Int32Array;
  private mats: Float64Array[] = [];
  private arcDist: Float32Array;
  private arcRate: Float32Array;
  private heap = new MinHeap(1024);
  private cellDelta: Float32Array;
  private collide = new Map<number, number>();
  private rng: Rng;
  private splitNoise: Simplex3;
  private sinceDrift = 0;
  private slowDt = 0;
  /** Plates merged during the run, for reporting. */
  readonly merges: { into: number; from: number; time: number }[] = [];
  readonly splits: { from: number; into: number; time: number }[] = [];
  readonly budget: CrustBudget = { collided: 0, rifted: 0, arcs: 0, capped: 0 };

  constructor(
    private mesh: SphereMesh,
    private geo: Geology,
    private cfg: SimConfig,
  ) {
    const n = mesh.n;
    this.head = new Int32Array(n + 1);
    this.cursor = new Int32Array(n);
    this.cellPlate = new Int16Array(n);
    this.show = new Int32Array(n);
    this.work = new Float32Array(n);
    this.mark = new Int32Array(n);
    this.gap = new Uint8Array(n);
    this.hotI = new Float32Array(n);
    this.hotS = new Float32Array(n);
    this.stamp = new Int32Array(n);
    this.queue = new Int32Array(n);
    this.arcDist = new Float32Array(n).fill(Infinity);
    this.arcRate = new Float32Array(n);
    this.cellDelta = new Float32Array(n);
    this.rng = new Rng(cfg.seed);
    this.splitNoise = new Simplex3(cfg.seed ^ 0x5bd1e995);
    this.reserve(Math.ceil(n * 1.25));
    for (let c = 0; c < n; c++) this.show[c] = this.addParcel(c, geo.plate[c], geo.ctype[c], geo.thick[c], geo.age[c], geo.orogeny[c], geo.volc[c], geo.volcAge[c]);
  }

  stableDt(): number {
    let wmax = 1e-6;
    for (const p of this.geo.plates) if (p.alive) wmax = Math.max(wmax, Math.hypot(p.wx, p.wy, p.wz));
    return Math.min(2, (0.75 * this.mesh.spacing) / wmax);
  }

  run(duration: number, onProgress?: (f: number) => void): void {
    let t = 0;
    let last = 0;
    while (t < duration - 1e-9) {
      const dt = Math.min(this.stableDt(), duration - t);
      this.step(dt);
      t += dt;
      if (onProgress && t - last > duration / 40) {
        onProgress(t / duration);
        last = t;
      }
    }
    this.slowProcesses(true);
  }

  /** Make room for `cap` parcels. */
  private reserve(cap: number): void {
    if (cap <= this.cap) return;
    cap = Math.max(cap, Math.ceil(this.cap * 1.5));
    const g8 = (a: Uint8Array | undefined) => {
      const b = new Uint8Array(cap);
      if (a) b.set(a);
      return b;
    };
    const g16 = (a: Int16Array | undefined) => {
      const b = new Int16Array(cap);
      if (a) b.set(a);
      return b;
    };
    const g32 = (a: Int32Array | undefined) => {
      const b = new Int32Array(cap);
      if (a) b.set(a);
      return b;
    };
    const gf = (a: Float32Array | undefined) => {
      const b = new Float32Array(cap);
      if (a) b.set(a);
      return b;
    };
    const gd = (a: Float64Array | undefined) => {
      const b = new Float64Array(cap);
      if (a) b.set(a);
      return b;
    };
    this.alive = g8(this.alive);
    this.qx = gd(this.qx);
    this.qy = gd(this.qy);
    this.qz = gd(this.qz);
    this.pPlate = g16(this.pPlate);
    this.pCtype = g8(this.pCtype);
    this.pThick = gf(this.pThick);
    this.pAge = gf(this.pAge);
    this.pOrog = gf(this.pOrog);
    this.pVolc = gf(this.pVolc);
    this.pVolcAge = gf(this.pVolcAge);
    this.pArea = gf(this.pArea);
    this.pCell = g32(this.pCell);
    this.list = g32(this.list);
    this.cap = cap;
  }

  /** New parcel at the center of cell c. */
  private addParcel(c: number, plate: number, ctype: number, thick: number, age: number, orog: number, volc: number, volcAge: number): number {
    this.reserve(this.P + 1);
    const i = this.P++;
    this.alive[i] = 1;
    this.qx[i] = this.mesh.x[c];
    this.qy[i] = this.mesh.y[c];
    this.qz[i] = this.mesh.z[c];
    this.pPlate[i] = plate;
    this.pCtype[i] = ctype;
    this.pThick[i] = thick;
    this.pAge[i] = age;
    this.pOrog[i] = orog;
    this.pVolc[i] = volc;
    this.pVolcAge[i] = volcAge;
    this.pArea[i] = this.mesh.area[c];
    this.pCell[i] = c;
    return i;
  }

  /** Index the live parcels by the cell they are in; note cells holding parcels of several plates. */
  private bin(): void {
    const n = this.mesh.n;
    const { head, cursor, alive, pCell, pPlate, cellPlate } = this;
    head.fill(0);
    for (let i = 0; i < this.P; i++) if (alive[i]) head[pCell[i] + 1]++;
    for (let c = 0; c < n; c++) head[c + 1] += head[c];
    cursor.set(head.subarray(0, n));
    const list = this.list;
    cellPlate.fill(-1);
    for (let i = 0; i < this.P; i++) {
      if (!alive[i]) continue;
      const c = pCell[i];
      list[cursor[c]++] = i;
      if (cellPlate[c] === -1) cellPlate[c] = pPlate[i];
      else if (cellPlate[c] !== pPlate[i]) cellPlate[c] = -2;
    }
    this.binnedP = this.P;
  }

  /** Visit cells of `plate` within `radius` (rad) of `center`, breadth-first. */
  private withinPlate(center: number, radius: number, plate: number, fn: (c: number, d: number) => void): void {
    const { x, y, z, nbrStart, nbr } = this.mesh;
    const id = ++this.stampId;
    const q = this.queue;
    const cosR = Math.cos(radius);
    const cx = x[center];
    const cy = y[center];
    const cz = z[center];
    let head = 0;
    let tail = 0;
    q[tail++] = center;
    this.stamp[center] = id;
    while (head < tail) {
      const c = q[head++];
      const dot = x[c] * cx + y[c] * cy + z[c] * cz;
      fn(c, Math.acos(Math.min(1, dot)));
      for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
        const m = nbr[k];
        if (this.stamp[m] === id) continue;
        this.stamp[m] = id;
        if (plate >= 0 && this.geo.plate[m] !== plate) continue;
        if (x[m] * cx + y[m] * cy + z[m] * cz < cosR) continue;
        q[tail++] = m;
      }
    }
  }

  private computeHotspots(): void {
    for (const c of this.hotCells) {
      this.hotI[c] = 0;
      this.hotS[c] = 0;
    }
    this.hotCells = [];
    const { x, y, z } = this.mesh;
    this.geo.hotspots.forEach((h, i) => {
      if (!h.active) return;
      const c0 = nearestCell(this.mesh, h.x, h.y, h.z, this.hotCache[i] ?? -1);
      this.hotCache[i] = c0;
      this.withinPlate(c0, h.radius * 1.2, -1, (c) => {
        const d = Math.acos(Math.min(1, x[c] * h.x + y[c] * h.y + z[c] * h.z));
        const w = Math.max(0, 1 - (d / h.radius) ** 2);
        if (w <= 0) return;
        if (this.hotI[c] === 0) this.hotCells.push(c);
        if (w > this.hotI[c]) {
          this.hotI[c] = w;
          this.hotS[c] = h.strength;
        }
      });
    });
  }

  private addCollision(a: number, b: number, area: number): void {
    const key = a < b ? a * 4096 + b : b * 4096 + a;
    this.collide.set(key, (this.collide.get(key) ?? 0) + area);
  }

  step(dt: number): void {
    const mesh = this.mesh;
    const geo = this.geo;
    const plates = geo.plates;
    while (this.mats.length < plates.length) this.mats.push(new Float64Array(9));
    for (let p = 0; p < plates.length; p++) rotationFromOmega(plates[p].wx, plates[p].wy, plates[p].wz, dt, this.mats[p]);
    this.computeHotspots();
    if (this.dead > 0.2 * this.P) this.compact();

    // Every parcel rides its plate; follow it to the nearest cell center.
    const { alive, qx, qy, qz, pPlate, pCell } = this;
    const { x, y, z, nbrStart, nbr } = mesh;
    for (let i = 0; i < this.P; i++) {
      if (!alive[i]) continue;
      const m = this.mats[pPlate[i]];
      const a = qx[i];
      const b = qy[i];
      const c0 = qz[i];
      let X = m[0] * a + m[1] * b + m[2] * c0;
      let Y = m[3] * a + m[4] * b + m[5] * c0;
      let Z = m[6] * a + m[7] * b + m[8] * c0;
      const l = 1 / Math.hypot(X, Y, Z);
      X *= l;
      Y *= l;
      Z *= l;
      qx[i] = X;
      qy[i] = Y;
      qz[i] = Z;
      let c = pCell[i];
      let best = x[c] * X + y[c] * Y + z[c] * Z;
      for (;;) {
        let next = -1;
        for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
          const nb = nbr[k];
          const d = x[nb] * X + y[nb] * Y + z[nb] * Z;
          if (d > best) {
            best = d;
            next = nb;
          }
        }
        if (next < 0) break;
        c = next;
      }
      pCell[i] = c;
    }
    this.bin();
    this.converge();
    if (this.showCells() > 0) this.fillGaps();
    this.writeView();

    this.postProcess(dt);
    this.writeView();
    if (this.cfg.merge) this.mergePlates();
    // Rifting is scheduled on the planet's clock, so it continues across successive runs.
    if (this.cfg.split && Math.floor((geo.time + dt) / SPLIT_INTERVAL) > Math.floor(geo.time / SPLIT_INTERVAL)) this.splitLargest();
    if (this.cfg.drift) {
      this.sinceDrift += dt;
      if (this.sinceDrift >= 4) {
        this.sinceDrift = 0;
        for (const p of plates) {
          if (!p.alive) continue;
          const s = Math.hypot(p.wx, p.wy, p.wz);
          const r = this.rng.unitVector();
          p.wx += r[0] * 0.08 * s;
          p.wy += r[1] * 0.08 * s;
          p.wz += r[2] * 0.08 * s;
          const s2 = Math.hypot(p.wx, p.wy, p.wz) || 1;
          const k = (s * (1 + 0.03 * this.rng.normal())) / s2;
          p.wx *= k;
          p.wy *= k;
          p.wz *= k;
        }
      }
    }
    geo.time += dt;
  }

  /** Buoyancy ranking of a parcel; `home` favors the plate already holding its cell. */
  private score(i: number): number {
    const { pCtype, pThick, pAge } = this;
    let s = pCtype[i] ? 1000 + pThick[i] : 500 - pAge[i] + Math.max(0, pThick[i] - 7) * 3;
    if (this.geo.plate[this.pCell[i]] === this.pPlate[i]) s += pCtype[i] ? 8 : 15;
    return s;
  }

  /**
   * Where parcels of two plates close in on each other, the less buoyant one
   * goes down: ocean floor subducts (its seamounts are scraped off onto the
   * overriding plate) and continental crust is thrust under the other side,
   * thickening it. Parcels sliding past each other (transform boundaries) are
   * left alone.
   */
  private converge(): void {
    const { nbrStart, nbr, spacing } = this.mesh;
    const { head, list, alive, qx, qy, qz, pPlate, pCtype, pThick, pVolc, pArea, pOrog, cellPlate } = this;
    const plates = this.geo.plates;
    const R = this.cfg.radiusKm;
    const cosNear = Math.cos(0.5 * spacing);
    const n = this.mesh.n;
    for (let c = 0; c < n; c++) {
      // Only where parcels of different plates are near each other.
      const pc = cellPlate[c];
      if (pc === -1) continue;
      let mixed = pc === -2;
      for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e && !mixed; k++) {
        const q = cellPlate[nbr[k]];
        if (q !== -1 && q !== pc) mixed = true;
      }
      if (!mixed) continue;
      for (let ia = head[c], ea = head[c + 1]; ia < ea; ia++) {
        const a = list[ia];
        if (!alive[a]) continue;
        const pa = pPlate[a];
        // Partners: later parcels in this cell, and parcels in higher-numbered neighbor cells.
        let k = nbrStart[c] - 1;
        let ib = ia + 1;
        let eb = ea;
        for (;;) {
          if (ib >= eb) {
            k++;
            if (k >= nbrStart[c + 1]) break;
            const m = nbr[k];
            if (m < c) continue;
            ib = head[m];
            eb = head[m + 1];
            continue;
          }
          const b = list[ib++];
          if (!alive[b] || pPlate[b] === pa) continue;
          // Parcels in one cell overlap; across cells only when close.
          if (k >= nbrStart[c] && qx[a] * qx[b] + qy[a] * qy[b] + qz[a] * qz[b] < cosNear) continue;
          // Are the two plates closing in along the line between these parcels?
          const A = plates[pa];
          const B = plates[pPlate[b]];
          let mx = qx[a] + qx[b];
          let my = qy[a] + qy[b];
          let mz = qz[a] + qz[b];
          const ml = Math.hypot(mx, my, mz);
          mx /= ml;
          my /= ml;
          mz /= ml;
          const dwx = B.wx - A.wx;
          const dwy = B.wy - A.wy;
          const dwz = B.wz - A.wz;
          const rx = dwy * mz - dwz * my;
          const ry = dwz * mx - dwx * mz;
          const rz = dwx * my - dwy * mx;
          const rl = Math.hypot(rx, ry, rz);
          if (rl * R < 2) continue;
          let ux = qx[b] - qx[a];
          let uy = qy[b] - qy[a];
          let uz = qz[b] - qz[a];
          const ul = Math.hypot(ux, uy, uz);
          if (ul > 1e-12) {
            ux /= ul;
            uy /= ul;
            uz /= ul;
            if (-(rx * ux + ry * uy + rz * uz) < 0.7 * rl) continue;
          }
          const aWins = this.score(a) >= this.score(b);
          const w = aWins ? a : b;
          const l = aWins ? b : a;
          const f = pArea[l] / pArea[w];
          if (pCtype[l] && pCtype[w]) {
            pThick[w] += pThick[l] * COLLISION_CONSERVE * f;
            pOrog[w] = 0;
            this.addCollision(pPlate[w], pPlate[l], pArea[l]);
            this.budget.collided += pArea[l];
            if (pThick[w] > MAX_THICK) this.spill(w);
          } else {
            // Seamounts and plateaus are scraped off onto the overriding plate.
            pThick[w] += (Math.max(0, pThick[l] - 10) * 0.5 + pVolc[l] * 0.3) * f;
            if (pThick[w] > MAX_THICK) this.spill(w);
          }
          alive[l] = 0;
          this.dead++;
          if (l === a) break;
        }
      }
    }
  }

  /** Crust stacked past the cap is pushed into the surrounding continental crust. */
  private spill(w: number): void {
    const { nbrStart, nbr } = this.mesh;
    const { head, list, alive, pCtype, pThick, pArea, pCell } = this;
    let excess = (pThick[w] - MAX_THICK) * pArea[w];
    const c = pCell[w];
    let room = 0;
    for (let k = nbrStart[c] - 1, e = nbrStart[c + 1]; k < e; k++) {
      const m = k < nbrStart[c] ? c : nbr[k];
      for (let i = head[m], ie = head[m + 1]; i < ie; i++) {
        const j = list[i];
        if (j !== w && alive[j] && pCtype[j]) room += Math.max(0, MAX_THICK - pThick[j]) * pArea[j];
      }
    }
    if (room > 0) {
      const f = Math.min(1, excess / room);
      for (let k = nbrStart[c] - 1, e = nbrStart[c + 1]; k < e; k++) {
        const m = k < nbrStart[c] ? c : nbr[k];
        for (let i = head[m], ie = head[m + 1]; i < ie; i++) {
          const j = list[i];
          if (j !== w && alive[j] && pCtype[j]) pThick[j] += f * Math.max(0, MAX_THICK - pThick[j]);
        }
      }
      excess -= f * room;
    }
    pThick[w] = MAX_THICK;
    this.budget.capped += Math.max(0, excess);
  }

  /**
   * Each cell shows the nearest parcel in it or next to it. A cell with no
   * crust within ~a cell spacing has been left behind by diverging plates.
   */
  private showCells(): number {
    const { n, x, y, z, nbrStart, nbr, spacing } = this.mesh;
    const { head, list, alive, qx, qy, qz, show, gap } = this;
    const cosGap = Math.cos(0.9 * spacing);
    let gaps = 0;
    for (let c = 0; c < n; c++) {
      const cx = x[c];
      const cy = y[c];
      const cz = z[c];
      let best = -1;
      let bd = -2;
      for (let i = head[c], e = head[c + 1]; i < e; i++) {
        const j = list[i];
        if (!alive[j]) continue;
        const d = qx[j] * cx + qy[j] * cy + qz[j] * cz;
        if (d > bd) {
          bd = d;
          best = j;
        }
      }
      if (best < 0) {
        for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
          const m = nbr[k];
          for (let i = head[m], ie = head[m + 1]; i < ie; i++) {
            const j = list[i];
            if (!alive[j]) continue;
            const d = qx[j] * cx + qy[j] * cy + qz[j] * cz;
            if (d > bd) {
              bd = d;
              best = j;
            }
          }
        }
        if (bd < cosGap) best = -1;
      }
      show[c] = best;
      gap[c] = best < 0 ? 1 : 0;
      if (best < 0) gaps++;
    }
    return gaps;
  }

  /** Copy the shown parcels into the geology arrays. */
  private writeView(): void {
    const { plate, ctype, thick, age, orogeny, volc, volcAge } = this.geo;
    const { show, pPlate, pCtype, pThick, pAge, pOrog, pVolc, pVolcAge } = this;
    for (let c = 0; c < this.mesh.n; c++) {
      const i = show[c];
      if (i < 0) continue;
      plate[c] = pPlate[i];
      ctype[c] = pCtype[i];
      thick[c] = pThick[i];
      age[c] = pAge[i];
      orogeny[c] = pOrog[i];
      volc[c] = pVolc[i];
      volcAge[c] = pVolcAge[i];
    }
  }

  /**
   * New crust where plates pull apart. Continental crust is stretched into the
   * rift (its walls thin to supply the new crust, conserving volume) until it
   * is too thin to hold together; then ocean floor forms.
   */
  private fillGaps(): void {
    const { n, nbrStart, nbr, area } = this.mesh;
    const { gap, show, hotI } = this;
    let pending: number[] = [];
    for (let c = 0; c < n; c++) if (gap[c]) pending.push(c);
    this.reserve(this.P + pending.length);
    const { pPlate, pCtype, pThick, pAge, pArea } = this;
    const pl: number[] = [];
    const pc: number[] = [];
    const walls: number[] = [];
    for (let pass = 0; pending.length > 0 && pass < 24; pass++) {
      const next: number[] = [];
      const dec: number[] = [];
      for (const c of pending) {
        let tot = 0;
        let contN = 0;
        let contT = 0;
        let contA = 0;
        pl.length = 0;
        pc.length = 0;
        for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
          const m = nbr[k];
          if (gap[m]) continue;
          tot++;
          const j = show[m];
          const p = pPlate[j];
          const idx = pl.indexOf(p);
          if (idx < 0) {
            pl.push(p);
            pc.push(1);
          } else pc[idx]++;
          if (pCtype[j]) {
            contN++;
            contT += pThick[j];
            contA += pAge[j];
          }
        }
        if (tot === 0) {
          next.push(c);
          continue;
        }
        let bi = 0;
        for (let i = 1; i < pl.length; i++) if (pc[i] > pc[bi]) bi = i;
        const meanT = contN ? contT / contN : 0;
        if (contN * 2 >= tot && meanT >= BREAKUP) dec.push(c, pl[bi], 1, meanT * STRETCH, contA / contN);
        else dec.push(c, pl[bi], 0, 7 + 18 * hotI[c], 0);
      }
      for (let i = 0; i < dec.length; i += 5) {
        const c = dec[i];
        const cont = dec[i + 2];
        if (cont) {
          // The rift walls (crust that was already there) supply the new crust's volume.
          walls.length = 0;
          let sum = 0;
          for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
            const m = nbr[k];
            if (gap[m]) continue;
            const j = show[m];
            if (!pCtype[j] || walls.includes(j)) continue;
            walls.push(j);
            sum += pThick[j] * pArea[j];
          }
          if (sum > 0) {
            const f = Math.max(0, 1 - (dec[i + 3] * area[c]) / sum);
            for (const j of walls) pThick[j] *= f;
          }
          this.budget.rifted += area[c];
        }
        show[c] = this.addParcel(c, dec[i + 1], cont, dec[i + 3], dec[i + 4], cont ? 500 : 1e4, 0, 1e4);
      }
      for (let i = 0; i < dec.length; i += 5) gap[dec[i]] = 0;
      pending = next;
    }
    for (const c of pending) {
      show[c] = this.addParcel(c, Math.max(0, this.geo.plate[c]), 0, 7, 0, 1e4, 0, 1e4);
      gap[c] = 0;
    }
  }

  private postProcess(dt: number): void {
    const mesh = this.mesh;
    const geo = this.geo;
    const { n, nbrStart, nbr } = mesh;
    const { alive, pCell, pCtype, pThick, pAge, pOrog, pVolc, pVolcAge } = this;

    // Hotspot volcanism: shield volcanoes on ocean floor, uplift + basalts on continents.
    for (let i = 0; i < this.P; i++) {
      if (!alive[i]) continue;
      const c = pCell[i];
      if (this.hotI[c] <= 0) continue;
      const g = this.hotS[c] * this.hotI[c] * dt;
      if (!pCtype[i]) {
        pVolc[i] = Math.min(9, pVolc[i] + g);
      } else {
        pVolc[i] = Math.min(2.5, pVolc[i] + 0.2 * g);
        pThick[i] = Math.min(MAX_THICK, pThick[i] + 0.08 * g);
      }
      pVolcAge[i] = 0;
    }

    for (let i = 0; i < this.P; i++) {
      if (!alive[i]) continue;
      pAge[i] += dt;
      pOrog[i] += dt;
      pVolcAge[i] += dt;
      if (pCtype[i] && pThick[i] > 35) pThick[i] -= (pThick[i] - 35) * (dt / OROGEN_TAU);
      if (pVolc[i] > 0) {
        let k = dt / VOLC_TAU;
        if (pVolcAge[i] > 1) {
          const top = rawElevation(pCtype[i], pThick[i], pAge[i], pVolc[i]);
          if (top > 0) k += (dt / VOLC_SUBAERIAL_TAU) * Math.min(1, top / (pVolc[i] * 1000));
        }
        pVolc[i] = pVolc[i] * Math.max(0, 1 - k);
        if (pVolc[i] < 0.01) pVolc[i] = 0;
      }
    }

    this.slowDt += dt;
    this.slowProcesses(false);

    // Remove single-cell plate fragments left by discretization.
    const { plate } = geo;
    for (let c = 0; c < n; c++) {
      const p = plate[c];
      let same = 0;
      let alt = -1;
      for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
        const q = plate[nbr[k]];
        if (q === p) same++;
        else alt = q;
      }
      if (same === 0 && alt >= 0) {
        plate[c] = alt;
        this.applyToCells([c], (i) => {
          if (this.pPlate[i] === p) this.pPlate[i] = alt;
        });
      }
    }
  }

  /** Arc growth and lower-crust flow are slow; run them about once a Myr on the accumulated time. */
  private slowProcesses(flush: boolean): void {
    if (this.slowDt <= 0 || (!flush && this.slowDt < 1)) return;
    const dt = this.slowDt;
    this.slowDt = 0;
    this.growArcs(dt);
    this.lowerCrustFlow(dt);
    if (flush) this.writeView();
  }

  /** Apply a per-cell change to every parcel in the cell (including crust created this step). */
  private applyToCells(cells: number[], fn: (i: number, c: number) => void): void {
    const { head, list, alive, show, binnedP } = this;
    for (const c of cells) {
      for (let i = head[c], e = head[c + 1]; i < e; i++) {
        const j = list[i];
        if (alive[j]) fn(j, c);
      }
      if (show[c] >= binnedP) fn(show[c], c);
    }
  }

  /**
   * Arc magmatism builds crust ~140 km behind every active trench on the
   * overriding plate, at a rate set by the convergence speed. Distances come
   * from the trench line itself, so the growth rate does not depend on how
   * finely the mesh samples the trench.
   */
  private growArcs(dt: number): void {
    const { n, x, y, z, nbrStart, nbr, nbrDist, area } = this.mesh;
    const { plate, ctype, age } = this.geo;
    const plates = this.geo.plates;
    const R = this.cfg.radiusKm;
    const dist = this.arcDist;
    const rate = this.arcRate;
    const heap = this.heap;
    const touched: number[] = [];
    heap.clear();
    for (let c = 0; c < n; c++) {
      const A = plates[plate[c]];
      let best = 0;
      let bestShear = 0;
      let bm = -1;
      for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
        const m = nbr[k];
        if (plate[m] === plate[c]) continue;
        const B = plates[plate[m]];
        let mx = x[c] + x[m];
        let my = y[c] + y[m];
        let mz = z[c] + z[m];
        const ml = Math.hypot(mx, my, mz);
        mx /= ml;
        my /= ml;
        mz /= ml;
        const dwx = B.wx - A.wx;
        const dwy = B.wy - A.wy;
        const dwz = B.wz - A.wz;
        const rx = dwy * mz - dwz * my;
        const ry = dwz * mx - dwx * mz;
        const rz = dwx * my - dwy * mx;
        let dx = x[m] - x[c];
        let dy = y[m] - y[c];
        let dz = z[m] - z[c];
        const dl = Math.hypot(dx, dy, dz);
        dx /= dl;
        dy /= dl;
        dz /= dl;
        const along = rx * dx + ry * dy + rz * dz;
        const conv = -along * R;
        if (conv > best) {
          best = conv;
          bestShear = Math.hypot(rx - along * dx, ry - along * dy, rz - along * dz) * R;
          bm = m;
        }
      }
      if (bm < 0 || best < 5 || best < 0.5 * bestShear) continue;
      // Continents never subduct; between oceans the older, denser plate sinks.
      const overrides = ctype[c] ? !ctype[bm] : !ctype[bm] && age[c] < age[bm];
      if (!overrides) continue;
      dist[c] = 0;
      rate[c] = best;
      touched.push(c);
      heap.push(0, c);
    }
    while (heap.size > 0) {
      const c = heap.pop();
      const d = heap.lastKey;
      if (d > dist[c]) continue;
      for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
        const m = nbr[k];
        if (plate[m] !== plate[c]) continue;
        const nd = d + nbrDist[k] * R;
        if (nd < dist[m] && nd <= ARC_REACH) {
          if (dist[m] === Infinity) touched.push(m);
          dist[m] = nd;
          rate[m] = rate[c];
          heap.push(nd, m);
        }
      }
    }
    const { pCtype, pThick, pOrog, pAge } = this;
    this.applyToCells(touched, (i, c) => {
      const wgt = Math.exp(-(((dist[c] - ARC_AXIS) / ARC_WIDTH) ** 2));
      const cap = pCtype[i] && pAge[i] > OLD_CRUST ? ARC_MAX_ANDEAN : ARC_MAX_ISLAND;
      const room = Math.min(1, Math.max(0, (cap - pThick[i]) / 10));
      pThick[i] += ARC_RATE * Math.min(1.3, rate[c] / 50) * wgt * room * dt;
      if (wgt > 0.5) pOrog[i] = 0;
      if (!pCtype[i] && pThick[i] >= 20) {
        pCtype[i] = 1;
        this.budget.arcs += area[c];
      }
    });
    for (const c of touched) dist[c] = Infinity;
  }

  /** Over-thickened crust spreads under its own weight (lower-crust flow), conserving volume. */
  private lowerCrustFlow(dt: number): void {
    const { n, nbrStart, nbr, nbrDist, nbrEdge, area } = this.mesh;
    const { ctype, thick } = this.geo;
    const R2 = this.cfg.radiusKm ** 2;
    const cells: number[] = [];
    let cmax = 0;
    for (let c = 0; c < n; c++) {
      if (!ctype[c] || thick[c] < FLOW_ONSET + 4) continue;
      cells.push(c);
      let g = 0;
      for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) g += nbrEdge[k] / nbrDist[k];
      cmax = Math.max(cmax, g / area[c]);
    }
    if (!cells.length) return;
    // Explicit diffusion on the shown thickness, sub-stepped to stay stable on fine meshes.
    const sub = Math.max(1, Math.ceil((1.5 * FLOW_K * dt * cmax) / R2 / 0.4));
    const h = dt / sub;
    const { cellDelta: delta, work, mark } = this;
    const id = ++this.markId;
    const changed: number[] = [];
    for (const c of cells) {
      for (let k = nbrStart[c] - 1, e = nbrStart[c + 1]; k < e; k++) {
        const m = k < nbrStart[c] ? c : nbr[k];
        if (mark[m] === id) continue;
        mark[m] = id;
        work[m] = thick[m];
        changed.push(m);
      }
    }
    for (let it = 0; it < sub; it++) {
      for (const c of cells) {
        for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
          const m = nbr[k];
          if (!ctype[m] || work[m] >= work[c]) continue;
          const avg = 0.5 * (work[c] + work[m]);
          const K = FLOW_K * Math.min(1.5, Math.max(0, (avg - FLOW_ONSET) / 20));
          if (K <= 0) continue;
          const f = (K * h * (nbrEdge[k] / nbrDist[k]) * (work[c] - work[m])) / R2;
          delta[c] -= f / area[c];
          delta[m] += f / area[m];
        }
      }
      for (const m of changed) {
        work[m] += delta[m];
        delta[m] = 0;
      }
    }
    // Hand the change in each cell's thickness to the parcels in it.
    const { pThick } = this;
    this.applyToCells(changed, (i, c) => {
      if (work[c] !== thick[c]) pThick[i] = Math.max(0, pThick[i] + work[c] - thick[c]);
    });
  }

  /** Rift the largest plate along a wavy great circle; the halves diverge. */
  private splitLargest(): void {
    const { n, x, y, z, area } = this.mesh;
    const geo = this.geo;
    const P = geo.plates.length;
    const ar = new Float64Array(P);
    const cen = new Float64Array(P * 3);
    for (let c = 0; c < n; c++) {
      const p = geo.plate[c];
      ar[p] += area[c];
      cen[3 * p] += x[c] * area[c];
      cen[3 * p + 1] += y[c] * area[c];
      cen[3 * p + 2] += z[c] * area[c];
    }
    let best = -1;
    for (let p = 0; p < P; p++) {
      if (geo.plates[p].alive && ar[p] > SPLIT_MIN_SHARE * 4 * Math.PI && (best < 0 || ar[p] > ar[best])) best = p;
    }
    if (best < 0) return;
    const c0 = normalize3([cen[3 * best], cen[3 * best + 1], cen[3 * best + 2]]);
    const r = this.rng.unitVector();
    const dd = r[0] * c0[0] + r[1] * c0[1] + r[2] * c0[2];
    const d = normalize3([r[0] - dd * c0[0], r[1] - dd * c0[1], r[2] - dd * c0[2]]);
    // Normal of the splitting great circle through the centroid.
    const nrm = normalize3([c0[1] * d[2] - c0[2] * d[1], c0[2] * d[0] - c0[0] * d[2], c0[0] * d[1] - c0[1] * d[0]]);
    const id = P;
    const noise = this.splitNoise;
    const off = this.rng.range(0, 100);
    const side = (px: number, py: number, pz: number) =>
      px * nrm[0] + py * nrm[1] + pz * nrm[2] + 0.15 * noise.fbm(px * 3 + off, py * 3, pz * 3, 3) > 0;
    const { alive, qx, qy, qz, pPlate } = this;
    let moved = 0;
    let total = 0;
    for (let i = 0; i < this.P; i++) {
      if (!alive[i] || pPlate[i] !== best) continue;
      total++;
      if (side(qx[i], qy[i], qz[i])) moved++;
    }
    if (moved < total * 0.2 || moved > total * 0.8) return;
    for (let i = 0; i < this.P; i++) if (alive[i] && pPlate[i] === best && side(qx[i], qy[i], qz[i])) pPlate[i] = id;
    this.writeView();
    // Half-spreading velocity added to each side so the new boundary opens.
    const s = this.rng.range(15, 35) / this.cfg.radiusKm / 2;
    const dx = (c0[1] * nrm[2] - c0[2] * nrm[1]) * s;
    const dy = (c0[2] * nrm[0] - c0[0] * nrm[2]) * s;
    const dz = (c0[0] * nrm[1] - c0[1] * nrm[0]) * s;
    const A = geo.plates[best];
    geo.plates.push({
      id,
      wx: A.wx + dx,
      wy: A.wy + dy,
      wz: A.wz + dz,
      color: plateColor(id),
      name: plateName(id),
      alive: true,
    });
    A.wx -= dx;
    A.wy -= dy;
    A.wz -= dz;
    this.splits.push({ from: best, into: id, time: geo.time });
  }

  private mergePlates(): void {
    const geo = this.geo;
    for (const [key, val] of this.collide) {
      if (val < MERGE_AREA) continue;
      this.collide.delete(key);
      const a = Math.floor(key / 4096);
      const b = key % 4096;
      const A = geo.plates[a];
      const B = geo.plates[b];
      if (!A?.alive || !B?.alive) continue;
      let areaA = 0;
      let areaB = 0;
      for (let c = 0; c < this.mesh.n; c++) {
        if (geo.plate[c] === a) areaA += this.mesh.area[c];
        else if (geo.plate[c] === b) areaB += this.mesh.area[c];
      }
      const [big, small, aBig, aSmall] = areaA >= areaB ? [a, b, areaA, areaB] : [b, a, areaB, areaA];
      const P = geo.plates[big];
      const S = geo.plates[small];
      const t = aBig + aSmall;
      P.wx = (P.wx * aBig + S.wx * aSmall) / t;
      P.wy = (P.wy * aBig + S.wy * aSmall) / t;
      P.wz = (P.wz * aBig + S.wz * aSmall) / t;
      for (let i = 0; i < this.P; i++) if (this.pPlate[i] === small) this.pPlate[i] = big;
      for (let c = 0; c < this.mesh.n; c++) if (geo.plate[c] === small) geo.plate[c] = big;
      S.alive = false;
      this.merges.push({ into: big, from: small, time: geo.time });
    }
  }

  /** Drop consumed parcels from the arrays. */
  private compact(): void {
    const { alive } = this;
    let j = 0;
    for (let i = 0; i < this.P; i++) {
      if (!alive[i]) continue;
      if (i !== j) {
        alive[j] = 1;
        this.qx[j] = this.qx[i];
        this.qy[j] = this.qy[i];
        this.qz[j] = this.qz[i];
        this.pPlate[j] = this.pPlate[i];
        this.pCtype[j] = this.pCtype[i];
        this.pThick[j] = this.pThick[i];
        this.pAge[j] = this.pAge[i];
        this.pOrog[j] = this.pOrog[i];
        this.pVolc[j] = this.pVolc[i];
        this.pVolcAge[j] = this.pVolcAge[i];
        this.pArea[j] = this.pArea[i];
        this.pCell[j] = this.pCell[i];
      }
      j++;
    }
    for (let i = j; i < this.P; i++) alive[i] = 0;
    this.P = j;
    this.dead = 0;
  }
}

/** Drop dead or vanished plates and absorb tiny fragments into neighbors. */
export function compactPlates(mesh: SphereMesh, geo: Geology, minCells = 12): void {
  const { n, nbrStart, nbr } = mesh;
  const P = geo.plates.length;
  const counts = new Int32Array(P);
  for (let c = 0; c < n; c++) counts[geo.plate[c]]++;
  // Absorb tiny plates into the neighbor they share most boundary with.
  for (let p = 0; p < P; p++) {
    if (counts[p] === 0 || counts[p] >= minCells) continue;
    const tally = new Map<number, number>();
    for (let c = 0; c < n; c++) {
      if (geo.plate[c] !== p) continue;
      for (let k = nbrStart[c], e = nbrStart[c + 1]; k < e; k++) {
        const q = geo.plate[nbr[k]];
        if (q !== p) tally.set(q, (tally.get(q) ?? 0) + 1);
      }
    }
    let best = -1;
    let bc = -1;
    for (const [q, v] of tally) if (v > bc) [best, bc] = [q, v];
    if (best < 0) continue;
    for (let c = 0; c < n; c++) if (geo.plate[c] === p) geo.plate[c] = best;
    counts[best] += counts[p];
    counts[p] = 0;
  }
  const remap = new Int16Array(P).fill(-1);
  const kept = [];
  for (let p = 0; p < P; p++) {
    if (counts[p] > 0) {
      remap[p] = kept.length;
      kept.push({ ...geo.plates[p], id: kept.length, alive: true });
    }
  }
  for (let c = 0; c < n; c++) geo.plate[c] = remap[geo.plate[c]];
  geo.plates = kept;
}
