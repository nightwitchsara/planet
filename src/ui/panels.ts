import * as pal from '../render/palettes';
import type { LayerId } from '../render/raster';
import { BIOME_INFO, KOPPEN_CODES, KOPPEN_COLORS, KOPPEN_NAMES } from '../world/biomes';
import type { EditReport } from '../world/edits';
import { ROCK_INFO } from '../world/geology';
import { DEPOSIT_COLORS, DEPOSIT_NAMES } from '../world/sediment';
import type { InspectInfo, WorldSummary } from '../worker/protocol';

const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

export function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]!);
}

const rgb = (c: readonly number[]) => `rgb(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])})`;

function gradientBar(stops: [number, pal.RGB][], fmt: (v: number) => string, ticks: number[]): string {
  const lo = stops[0][0];
  const hi = stops[stops.length - 1][0];
  const css = stops.map(([v, c]) => `${rgb(c)} ${(((v - lo) / (hi - lo)) * 100).toFixed(1)}%`).join(',');
  const labels = ticks
    .map((t) => `<span style="left:${(((t - lo) / (hi - lo)) * 100).toFixed(1)}%">${escapeHtml(fmt(t))}</span>`)
    .join('');
  return `<div class="gbar" style="background:linear-gradient(90deg,${css})"></div><div class="gticks">${labels}</div>`;
}

function swatches(items: { color: readonly number[]; label: string }[]): string {
  return `<div class="swatches">${items
    .map((i) => `<div class="sw"><i style="background:${rgb(i.color)}"></i>${escapeHtml(i.label)}</div>`)
    .join('')}</div>`;
}

export function legendHtml(layer: LayerId, summary: WorldSummary | null): string {
  switch (layer) {
    case 'terrain':
      return `<p class="muted">Natural color from simulated biomes, with relief shading, rivers, lakes, snow, sea ice and reefs. Dashed rivers run only part of the year; dotted ones after rare rains.</p>${swatches(
        [1, 2, 4, 6, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17].map((b) => ({ color: BIOME_INFO[b].color, label: BIOME_INFO[b].name })),
      )}${swatches([
        { color: [70, 124, 156], label: 'Freshwater lake' },
        { color: [176, 204, 192], label: 'Salt lake' },
        { color: [212, 226, 236], label: 'Frozen lake' },
        { color: [240, 244, 248], label: 'Glacier / ice sheet' },
        { color: [188, 212, 232], label: 'Bare melting ice' },
        { color: [228, 236, 244], label: 'Ice shelf (afloat)' },
      ])}`;
    case 'elevation':
      return (
        gradientBar(pal.BATHY, (v) => `${v / 1000}`, [-8000, -4000, 0]) +
        gradientBar(pal.HYPSO, (v) => `${v / 1000}`, [0, 2000, 4000, 6000]) +
        '<p class="muted">Kilometres below / above sea level.</p>'
      );
    case 'plates':
      return `${swatches([
        { color: [235, 50, 45], label: 'Convergent (trench / collision)' },
        { color: [60, 140, 255], label: 'Divergent (ridge / rift)' },
        { color: [70, 220, 110], label: 'Transform' },
      ])}<p class="muted">Light = continental crust, dark = oceanic. Arrows show plate velocity.</p>${
        summary
          ? swatches(
              summary.plates.map((p) => ({
                color: p.color,
                label: `${p.name} · ${(p.speed / 10).toFixed(1)} cm/yr · ${Math.round(p.continental * 100)}% cont.`,
              })),
            )
          : ''
      }`;
    case 'crust':
      return (
        '<p class="muted">Continental crust thickness (km)</p>' +
        gradientBar(pal.CONT_THICK, (v) => `${v}`, [15, 30, 45, 65]) +
        '<p class="muted">Seafloor age (Myr) — young at ridges, old near trenches</p>' +
        gradientBar(pal.OCEAN_AGE, (v) => `${v}`, [0, 50, 100, 150, 200])
      );
    case 'temperature':
      return gradientBar(pal.TEMPERATURE, (v) => `${v}°`, [-40, -20, 0, 20, 36]) + '<p class="muted">Surface air temperature (°C).</p>';
    case 'precipitation':
      return gradientBar(pal.PRECIP, (v) => (v >= 1000 ? `${v / 1000}k` : `${v}`), [0, 700, 2000, 3200, 5000]) + '<p class="muted">mm per year (monthly view: annualized rate).</p>';
    case 'biomes':
      return swatches(BIOME_INFO.slice(1).map((b) => ({ color: b.color, label: b.name })));
    case 'koppen':
      return swatches(KOPPEN_CODES.map((k) => ({ color: KOPPEN_COLORS[k], label: `${k} — ${KOPPEN_NAMES[k]}` })));
    case 'rocks':
      return (
        '<p class="muted">Bedrock under any soil or loose sediment, from the plate history. Hard rock (granite, gneiss, basalt) holds up ridges; shale and salt wear down into valleys. Seafloor rock is shown dimmed.</p>' +
        swatches(ROCK_INFO.map((r) => ({ color: r.color, label: r.name })))
      );
    case 'deposits':
      return (
        '<p class="muted">The loose material over the bedrock: what rivers, ice and wind laid down, peat in bogs, or rock weathered in place. Fans and dune fields gather in dry basins, loess downwind of deserts and ice-age outwash, till and moraines where the ice sheets stood.</p>' +
        swatches(DEPOSIT_NAMES.map((name, i) => ({ color: DEPOSIT_COLORS[i], label: name })))
      );
    case 'drainage':
      return '<p class="muted">Each color is one drainage basin: all land whose water leaves through the same river mouth or ends in the same inland lake or salt flat. Blue lines are rivers above the discharge threshold; dashed ones run only part of the year.</p>';
  }
}

function climograph(t: number[], p: number[]): string {
  const W = 260;
  const H = 96;
  const pMax = Math.max(100, ...p);
  const tMin = Math.min(-10, ...t);
  const tMax = Math.max(30, ...t);
  const bw = W / 12;
  const bars = p
    .map((v, i) => {
      const h = (v / pMax) * (H - 16);
      return `<rect x="${(i * bw + 2).toFixed(1)}" y="${(H - 12 - h).toFixed(1)}" width="${(bw - 4).toFixed(1)}" height="${h.toFixed(1)}" fill="#3d7fd6" opacity="0.8"/>`;
    })
    .join('');
  const ty = (v: number) => (H - 12 - ((v - tMin) / (tMax - tMin)) * (H - 16)).toFixed(1);
  const line = t.map((v, i) => `${(i * bw + bw / 2).toFixed(1)},${ty(v)}`).join(' ');
  const zero = tMin < 0 ? `<line x1="0" x2="${W}" y1="${ty(0)}" y2="${ty(0)}" stroke="#ffffff22" stroke-dasharray="3 3"/>` : '';
  const labels = MONTH_NAMES.map((m, i) => `<text x="${(i * bw + bw / 2).toFixed(1)}" y="${H - 1}" text-anchor="middle">${m[0]}</text>`).join('');
  return `<svg class="climo" viewBox="0 0 ${W} ${H}" width="100%">${bars}${zero}<polyline points="${line}" fill="none" stroke="#ff7a59" stroke-width="2"/>${labels}</svg>
  <div class="climo-key"><span style="color:#ff7a59">— temperature (${tMin.toFixed(0)}…${tMax.toFixed(0)}°C)</span> <span style="color:#3d7fd6">▮ precipitation (max ${pMax.toFixed(0)} mm)</span></div>`;
}

function hydrograph(q: number[]): string {
  const W = 260;
  const H = 70;
  const qMax = Math.max(...q, 1e-6);
  const bw = W / 12;
  const bars = q
    .map((v, i) => {
      const h = (v / qMax) * (H - 16);
      return `<rect x="${(i * bw + 2).toFixed(1)}" y="${(H - 12 - h).toFixed(1)}" width="${(bw - 4).toFixed(1)}" height="${Math.max(0.5, h).toFixed(1)}" fill="#3fb7c9" opacity="0.85"/>`;
    })
    .join('');
  const labels = MONTH_NAMES.map((m, i) => `<text x="${(i * bw + bw / 2).toFixed(1)}" y="${H - 1}" text-anchor="middle">${m[0]}</text>`).join('');
  const fmt = (v: number) => (v >= 100 ? Math.round(v).toLocaleString('en-US') : v.toFixed(1));
  return `<svg class="climo" viewBox="0 0 ${W} ${H}" width="100%">${bars}${labels}</svg>
  <div class="climo-key"><span style="color:#3fb7c9">▮ river flow by month (peak ${fmt(qMax)} m³/s)</span></div>`;
}

export function inspectorHtml(info: InspectInfo | null): string {
  if (!info) return '<p class="muted">Hover the map to inspect a location.</p>';
  const tAnn = info.tempMonthly.reduce((a, b) => a + b, 0) / 12;
  const pAnn = info.precipMonthly.reduce((a, b) => a + b, 0);
  const ns = info.lat >= 0 ? 'N' : 'S';
  const ew = info.lon >= 0 ? 'E' : 'W';
  const elev = info.water === 'ocean' ? `${Math.round(-info.elevation).toLocaleString('en-US')} m deep` : `${Math.round(info.elevation).toLocaleString('en-US')} m`;
  const rows: [string, string][] = [
    ['Location', `${Math.abs(info.lat).toFixed(1)}°${ns}, ${Math.abs(info.lon).toFixed(1)}°${ew}`],
    [info.water === 'ocean' ? 'Depth' : 'Elevation', elev + (info.water && info.water !== 'ocean' ? ` (${info.water})` : '')],
    ['Setting', info.setting],
    ['Plate', `${info.plate.name} · ${info.plate.speed.toFixed(1)} cm/yr → ${info.plate.dir}`],
    ['Crust', `${info.crust.continental ? 'Continental' : 'Oceanic'}, ${info.crust.thickness.toFixed(0)} km${info.crust.continental ? '' : `, ${info.crust.age.toFixed(0)} Myr old`}`],
  ];
  if (info.surface) rows.push(['Surface', info.surface]);
  if (info.bedrock) rows.push(['Bedrock', info.bedrock]);
  if (info.boundary) rows.push(['Boundary', info.boundary]);
  rows.push(['Climate', `${tAnn.toFixed(1)}°C mean, ${Math.round(pAnn).toLocaleString('en-US')} mm/yr`]);
  if (info.koppen) rows.push(['Köppen', `${info.koppen} — ${info.koppenName}`]);
  if (info.water !== 'ocean') rows.push(['Biome', info.biome]);
  const num = (v: number) => (Math.round(v) || 0).toLocaleString('en-US');
  if (info.water !== 'ocean') {
    if (!info.ice) rows.push(['Water balance', `evaporation ${num(info.aet)} of ${num(info.pet)} mm possible; runoff ${num(info.runoff)} mm`]);
    if (info.glacier) rows.push(['Snow', 'Permanent snow and ice']);
    else if (info.snowMonths > 0) rows.push(['Snow', `Snow cover ${info.snowMonths} month${info.snowMonths === 1 ? '' : 's'} a year`]);
  }
  if (info.lake) {
    const flow = info.lake.outflow > 0 ? `inflow ${num(info.lake.inflow)} m³/s, outflow ${num(info.lake.outflow)} m³/s` : `inflow ${num(info.lake.inflow)} m³/s, no outlet`;
    rows.push(['Lake', `${info.lake.description} · ${num(info.lake.areaKm2)} km² · ${flow}`]);
  }
  if (info.discharge > 1) rows.push(['River', `${num(info.discharge)} m³/s mean${info.regime ? ` · ${info.regime}` : ''}`]);
  if (info.wetland) rows.push(['Wetland', info.wetland]);
  if (info.endorheic && info.water !== 'ocean') rows.push(['Drainage', 'Endorheic: drains to an inland basin, not the sea']);
  if (info.ice) rows.push(['Ice', info.ice]);
  if (info.iceAge) rows.push(['Ice age', info.iceAge]);
  const flow = info.flowMonthly ? hydrograph(info.flowMonthly) : '';
  return `<dl class="kv">${rows.map(([k, v]) => `<dt>${k}</dt><dd>${escapeHtml(v)}</dd>`).join('')}</dl>${climograph(info.tempMonthly, info.precipMonthly)}${flow}`;
}

export function reportHtml(r: EditReport): string {
  const opts = r.options
    ? `<div class="causes">${r.options
        .map(
          (o) =>
            `<div class="cause${o.id === r.chosen ? ' chosen' : ''}" title="${escapeHtml(o.reason)}"><span class="cname">${escapeHtml(o.label)}</span><span class="cbar"><b style="width:${Math.round(o.score * 100)}%"></b></span><span class="cscore">${Math.round(o.score * 100)}</span></div>`,
        )
        .join('')}</div><p class="muted small">Feasibility of each natural cause (hover for reasoning). Highlighted = applied.</p>`
    : '';
  const details = r.details.length ? `<ul>${r.details.map((d) => `<li>${escapeHtml(d)}</li>`).join('')}</ul>` : '';
  return `<div class="report${r.failed ? ' failed' : ''}"><h4>${escapeHtml(r.title)}</h4><p>${escapeHtml(r.summary)}</p>${details}${opts}</div>`;
}
