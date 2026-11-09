import { gridTriangle, locateTriangle, triWeights, type SphereMesh } from '../../core/mesh';
import { Simplex3 } from '../../core/noise';
import { Geology } from '../types';

/**
 * Resample geology from the coarse history mesh onto the fine output mesh.
 * Sampling positions are domain-warped by about one coarse cell so that plate
 * and coastline boundaries don't inherit the coarse triangulation.
 */
export function transferGeology(coarse: SphereMesh, src: Geology, fine: SphereMesh, seed: number): Geology {
  const g = new Geology(fine.n);
  g.plates = src.plates.map((p) => ({ ...p, color: [...p.color] as [number, number, number] }));
  g.hotspots = src.hotspots.map((h) => ({ ...h }));
  g.time = src.time;
  const noise = new Simplex3(seed);
  const amp = 0.8 * coarse.spacing;
  const F = 1 / (4 * coarse.spacing);
  const w = new Float64Array(3);
  const v = [0, 0, 0];
  let t = 0;
  for (let i = 0; i < fine.n; i++) {
    const px = fine.x[i];
    const py = fine.y[i];
    const pz = fine.z[i];
    let qx = px + amp * noise.fbm(px * F, py * F, pz * F, 2);
    let qy = py + amp * noise.fbm(px * F + 31.7, py * F, pz * F, 2);
    let qz = pz + amp * noise.fbm(px * F, py * F + 47.3, pz * F, 2);
    const l = Math.hypot(qx, qy, qz);
    qx /= l;
    qy /= l;
    qz /= l;
    t = locateTriangle(coarse, qx, qy, qz, gridTriangle(coarse, qx, qy, qz));
    triWeights(coarse, t, qx, qy, qz, w);
    v[0] = coarse.tri[3 * t];
    v[1] = coarse.tri[3 * t + 1];
    v[2] = coarse.tri[3 * t + 2];
    let best = 0;
    if (w[1] > w[best]) best = 1;
    if (w[2] > w[best]) best = 2;
    const vb = v[best];
    const ct = src.ctype[vb];
    g.plate[i] = src.plate[vb];
    g.ctype[i] = ct;
    let th = 0;
    let vo = 0;
    let va = 0;
    let vw = 0;
    let sw = 0;
    let ag = 0;
    let og = 0;
    for (let k = 0; k < 3; k++) {
      const c = v[k];
      th += w[k] * src.thick[c];
      vo += w[k] * src.volc[c];
      vw += w[k] * src.volc[c];
      va += w[k] * src.volc[c] * src.volcAge[c];
      if (src.ctype[c] === ct) {
        sw += w[k];
        ag += w[k] * src.age[c];
        og += w[k] * src.orogeny[c];
      }
    }
    g.thick[i] = th;
    g.volc[i] = vo;
    g.volcAge[i] = vw > 1e-6 ? va / vw : 1e4;
    g.age[i] = sw > 0 ? ag / sw : src.age[vb];
    g.orogeny[i] = sw > 0 ? og / sw : src.orogeny[vb];
  }
  return g;
}
