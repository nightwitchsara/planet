# Planet Forge

A semi-random, process-driven world generator and editor for grand-strategy maps.
The planet is simulated rather than painted, and edits change the geological causes
so every effect is re-derived from them.

```
npm install
npm run dev        # http://localhost:5317
```

## How a world is built

Everything downstream of the **geology layer** is derived; nothing is painted directly.

| Stage | Model |
| --- | --- |
| Mesh | Jittered Fibonacci sphere, spherical Delaunay/Voronoi (20k–1.28M cells; 1.28M by default, about 20 s to generate) plus a coarse mesh for history and climate. Cell-scale processes (stream-power erosion, slope-dependent soils and wetlands, floodplain and delta widths, ice-shelf reach) are calibrated to 320k cells and scaled to the mesh, so a world looks the same at every detail level, only finer |
| Plate tectonics | Kinematic simulation of 0–300 Myr on rigid plates with Euler poles. The crust is a set of parcels carried at their true positions (each cell shows the nearest), so coastlines and mountain belts keep their shape however far they travel. Gaps become new crust: stretched continent first (the rift walls thin to supply it, conserving volume), then ocean floor at ridges. Where converging plates overlap, buoyancy decides: continents never subduct, younger and lighter oceanic lithosphere overrides older. Continental collisions stack crust (volume conserved, spreading as a plateau when very thick) and weld the plates; the largest plate rifts in two every 30 Myr, also while advancing time. Arc magmatism grows crust behind trenches at a rate set by convergence: island arcs mature into continental crust but stay island chains (~35 km), Andean-type margins thicken to ~70 km. |
| Hotspots | Plumes fixed in the mantle frame. Plates carry their volcanoes away, so chains, subsidence and guyots emerge. |
| Relief | Airy isostasy from crustal thickness; seafloor depth from age (GDH1 cooling curve). On top of that: trenches, outer rises, volcanic arcs, collision belts, continental rifts, hotspot swells, sediment-filled shelves, Darwin-style atolls (one per extinct, drowned tropical volcano, clear of high islands, which get fringing reefs), and fractal detail scaled by orogen age. |
| Bedrock geology | Thirteen rock types read from the plate history. Near active boundaries: arc volcanics with granite batholiths behind them, forearc and back-arc fold belts and molasse forelands; collision belts with schist-and-gneiss cores, ophiolite slivers along the suture and folded flanks; rifts with red beds, salt in the dry belts and basalt along the axis; flood basalts around plumes under continents. Elsewhere the crust's own record decides: crust born in island arcs is a mosaic of volcanics, granite and schist; old belts are worn down to their metamorphic roots (with ophiolites along their crests); ancient cratons are bare shield where they stand high and buried under platform strata where they sagged, in belts of sandstone, shale and limestone ringing each basin (carbonates in warm latitudes, evaporites in the dry belts); drowned shelves carry limestone or mud; ocean floor is basalt under deep-sea ooze, atolls are reef limestone. Each rock has an erodibility (granite and gneiss ~0.3, basalt ~0.45, sandstone ~0.9, shale ~2.5, salt ~3; folded strata alternate hard and soft beds along the belt), so erosion leaves ridges, escarpments and vales; the land average is ~1, the rock the erosion law is calibrated for. Edits change the geology the rocks are read from, so painted mountains fold their cover and new islands are lava or reef. |
| Climate (12 months) | 1D Budyko–Sellers energy balance with ice–albedo feedback; seasonal response damped by local heat capacity, with land/sea character carried downwind; wind-driven gyre currents; a three-cell circulation following a seasonal ITCZ; thermal monsoon flow; steady-state moisture transport by the winds plus mixing by passing storm systems (which carries water to high latitudes), with convection, convergence, storm tracks, orographic lift and rain shadows. |
| Water balance (12 months) | Hargreaves evaporative demand from radiation, temperature and an estimated daily temperature range; snow that accumulates and melts by positive degree-days, with permanent snow becoming glaciers; a soil store that dries exponentially; groundwater that sustains baseflow through dry seasons. Spun up to a steady annual cycle. |
| Erosion and sediment | Implicit stream-power incision (Braun–Willett) plus hillslope diffusion. The eroded rock travels downstream: depressions trap it and silt up from the bottom (most small lakes vanish, big tectonic basins keep their lakes), gentle reaches build floodplains, and rivers fill their shelf and, if they drain a large basin, build a delta plain out to sea. |
| Glaciers and ice sheets | Ice gathers where a year's snow outlasts the summer melt (positive degree-days) at the ice surface, so a growing ice cap raises itself into colder air. Ice flows down its surface, from accumulation zones into warmer ground, until melt balances what arrives or it calves into the sea; where it gives out, a glacier's tongue covers only the share of a cell its inflow can feed. Its thickness follows the perfectly plastic profile real ice caps match, and the bed sinks under the load. Cold coasts grow floating ice shelves in their embayments. Glacier ice feeds rivers with summer meltwater. |
| Ice ages | The same model runs for the last glacial maximum: a colder (−3 °C in the tropics to −12 °C at the poles), drier planet with seas 120 m lower. Its fast-sliding outlet and valley glaciers deepened their beds, so formerly glaciated land has troughs, fjord coasts, and lake basins large and small. |
| Rivers and lakes | Priority-flood routing of the monthly runoff. In dry months rivers lose water into their beds, so small desert streams wither while big rivers cross deserts. Each depression fills until it overflows (a freshwater lake that evens out the seasons downstream), until evaporation balances its inflow (a salt lake), or, if even that is out of reach, holds only a salt flat. Rising water covers each cell gradually (its ground spans heights about its mean), so a shallow hollow brims and spills before it would drown its whole floor, shores the lake covers in good part drain into it, and a hollow too shallow for a lake is crossed by its rivers. A lake damps floods as a reservoir: one on a big river passes them on, a big lake on a small river flattens them. Every river gets a flow regime (perennial, seasonal or ephemeral, rain- or melt-fed) and a bankfull flood (the ≈1.5-year flood its channel is built for, from the wettest month and how flashy the catchment is) that sets the channel's width; land draining to closed basins is marked endorheic. |
| Wetlands | River floodplains, inland deltas where rivers die out in dry country, peat bogs on cold waterlogged ground, and groundwater-fed swamps. |
| Biomes | Köppen–Geiger (Peel 2007 / Kottek 2006) and a biome map used for natural-color rendering; glaciers, wetlands and salt flats come from the hydrology |
| Rendering | A tiled pixel pyramid from the 2048 × 1024 overview down to ~38 m per pixel (on an Earth-sized world), rendered in parallel by tile workers. Cell values are blended with a smooth kernel over slope-limited local planes (no triangle facets). Below cell size, detail is synthesized but stays consistent with the cells: coastlines follow the cells' land/sea split, roughened fractally; ridged relief in young mountains, gentle swells on plains; rivers wander between their cells and cut V-shaped valleys that merge where they meet (so confluent glacial troughs join into one fjord, open to the sea), are dashed where they run only part of the year, and carry green ribbons through dry country; zoomed in, a channel is drawn at its true width (about 4·√Q m for its bankfull flood Q) between darker banks, and once it is a pixel or two wide it meanders within its valley as a sine-generated curve, loops about twelve widths long but each of its own length and tightness, its meander belt wandering too, and swinging less on reaches too steep to meander (Leopold & Wolman's braiding threshold) or in young mountains; each stream runs unbroken down its main stem, a tributary is drawn on its own flow and curves into its trunk at an acute angle, ending in a slight mouth where it first touches the trunk's channel while its valley opens out into the trunk's, and every river ends where it first runs into the water drawn at that zoom, sea or a fjord's drowned trough, by the same land/sea decision the pixels use (a course across an inlet never reappears beyond it); lakes fill to their simulated level, freeze over in cold months, and dry basins show salt flats; ice sheets have smooth surfaces with rock peaks breaking through and blue bare ice where they melt, and their margins take the low ground first, so outlet tongues run down valleys and ridges stand clear; ice shelves float off cold coasts; ice-age glacial troughs drown into fjords with wandering walls, narrow at the head and widening toward a shallow, skerry-strewn mouth sill; up the valley beyond the fjord's head threshold the trough's overdeepened basins hold ribbon lakes between rock bars, standing above sea level; scoured lowlands are dotted with countless small lakes; atolls become reef rings on drowned volcanoes in open water; snow and bare rock follow the local height and slope. Each zoom level only adds finer octaves, so levels and neighboring tiles agree. |

## Editing: changes propagate to their causes

| Tool | What it changes upstream |
| --- | --- |
| Land | Extends the **nearest continent's crust**. Its plate grows to contain the new land (taking cells from neighbors) and a passive-margin shelf is added. Far from any continent it becomes a microcontinent. |
| Shallow sea / Rift ocean | Thins continental crust into an epicontinental sea or sunken continent, or replaces it with young oceanic crust |
| Mountains / Old highlands / Lower | Thickens or thins crust (isostasy does the rest) and reports whether an active collision or arc explains it, or records an ancient orogen |
| Island | Scores every plausible natural cause and applies the best one, or the one you pick: hotspot volcano (with its trailing seamount chain), island arc (**changes plate motion to create subduction** if none is nearby), atoll on a subsided volcano (with its extinct hotspot track), continental fragment (Zealandia-like), exposed shelf, delta/barrier island |
| Hotspot | New mantle plume; its past track is reconstructed from plate motion |
| Plate motion | Drag to set a plate's velocity (Euler pole 90° away); boundaries re-classify immediately |
| Advance time | Runs the tectonic simulation forward from the edited present (continental crust is conserved apart from collisions and new arcs; the largest plate keeps rifting on schedule) |

Every edit is a sparse diff of the geology layer, so undo/redo is exact. Each edit is logged with its causal explanation.

## Output

* **Save / Load**: `.planet` files hold the geology layer only (~0.5–5 MB). All other layers are re-derived on load.
* **Export**: a 16-bit grayscale heightmap (−11 000 m…+9 000 m mapped to 0…65535) and any layer as PNG,
  up to 8192 × 4096, with the same sub-cell detail as the map. Images are equirectangular, north up,
  180°W at the left edge.

## Development

```
npm run typecheck
npm run selftest             # every edit op, undo/redo, save/load, export
npm run render -- <seed> out # headless layer PNGs (env: CELLS, W, MONTHS=0,6, ZOOM="lat,lon,level;…" close-ups)
```

Source layout: `src/core` (mesh, noise, RNG), `src/world` (tectonics, relief, climate,
hydrology, biomes, edits, `World` pipeline), `src/render` (render context, pixel renderer,
palettes), `src/worker` (the simulation worker and the tile workers), `src/ui` (map/globe
views, tile manager, panels).

The dev server sends cross-origin-isolation headers so the simulation worker can share the
world with the tile workers through `SharedArrayBuffer`; without them each tile worker gets
its own copy, which still works but uses more memory.

## Known limitations / next passes

* Detail below the cell size is synthesized, not simulated, and edits still act on whole cells
  (about 20 km apart at the default detail): a brush stroke is no finer than that, however far you zoom.
* At the default detail a brush stroke previews in ~1.5 s and settles in ~10 s, and Advance time
  takes ~25 s per 10 Myr: the plate step must stay under a cell's width, so finer meshes take more steps.
* On the globe, views that include a pole use coarser detail tiles.
* Ocean currents use a basin-geometry heuristic rather than a circulation solve.
* Ice sheets are found in steady state for today's and the ice-age climate, not grown through
  glacial cycles; ice doesn't change the climate around it beyond its own height (no local
  albedo feedback), and mountain glaciers smaller than a cell show only as snow on the peaks.
  Within a cell, which ground the ice covers is synthesized (low ground first), not simulated.
* Lakes and rivers don't feed back on the climate (no lake-effect rain). The subtropics are drier
  than Earth's, so deserts and endorheic basins run larger.
* No soils or rock types yet.
* Plates keep their motions while advancing time (except collisions welding them and scheduled
  rifts), so long runs aggregate continents faster than Earth's ever-changing plate motions would.
  Eroded mountain rock leaves the crust instead of building margins.
* Climate-level edits (for example "make this wetter") are not yet traced back to causes.
