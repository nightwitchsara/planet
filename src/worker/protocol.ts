import type { FieldData, MeshData } from '../render/context';
import type { PixelWindow, RenderOptions } from '../render/raster';
import type { EditReport, IslandCause, LatLon } from '../world/edits';
import type { WorldParams } from '../world/types';

export type BrushTool = 'land' | 'sea' | 'deepsea' | 'raise' | 'raiseOld' | 'lower';

export type EditOp =
  | { kind: 'island'; at: LatLon; cause: IslandCause; sizeKm: number }
  | { kind: 'hotspot'; at: LatLon; strength: number }
  | { kind: 'plateVelocity'; at: LatLon; vEast: number; vNorth: number }
  | { kind: 'advance'; myr: number };

/** Main thread → simulation worker. */
export type Request =
  | { type: 'init'; ports: MessagePort[] }
  | { type: 'generate'; params: WorldParams }
  | { type: 'inspect'; id: number; lat: number; lon: number }
  | { type: 'stroke'; phase: 'start' | 'move' | 'end'; tool: BrushTool; points: LatLon[]; radiusKm: number; strength: number; render: RenderOptions }
  | { type: 'edit'; op: EditOp }
  | { type: 'undo' }
  | { type: 'redo' }
  | { type: 'setParams'; params: Partial<WorldParams> }
  | { type: 'fitSeaLevel'; landFraction: number }
  | { type: 'save' }
  | { type: 'load'; buffer: ArrayBuffer }
  | { type: 'export'; what: 'heightmap16' | 'layer'; render: RenderOptions; width: number };

/** Simulation worker → tile workers (over a MessageChannel). */
export type ContextMessage =
  | { type: 'mesh'; mesh: MeshData }
  | { type: 'fields'; fields: FieldData }
  /** Render a strip of an export; `elevation` asks for heights instead of colors. */
  | { type: 'strip'; id: number; render: RenderOptions; window: PixelWindow; elevation: boolean };

/** Tile worker → simulation worker: a finished export strip. */
export interface StripResult {
  type: 'strip';
  id: number;
  /** RGBA bytes, or Float32 heights when elevation was asked for. */
  data: ArrayBuffer;
}

/** Main thread ↔ tile workers. */
export interface TileRequest {
  type: 'tile';
  id: number;
  /** World version the tile must show. */
  version: number;
  render: RenderOptions;
  level: number;
  x: number;
  y: number;
}

export type TileWorkerRequest = { type: 'port'; port: MessagePort } | TileRequest;

export interface TileResponse {
  type: 'tile';
  id: number;
  version: number;
  level: number;
  x: number;
  y: number;
  /** Missing when the request was overtaken by a newer world version. */
  bitmap?: ImageBitmap;
  ms: number;
}

export interface PlateSummary {
  id: number;
  name: string;
  color: [number, number, number];
  areaKm2: number;
  continental: number;
  speed: number;
}

export interface Arrow {
  lat: number;
  lon: number;
  vE: number;
  vN: number;
  plate: number;
}

export interface WorldSummary {
  params: WorldParams;
  cells: number;
  plates: PlateSummary[];
  hotspots: { name: string; lat: number; lon: number; active: boolean }[];
  arrows: Arrow[];
  landFraction: number;
  timeMyr: number;
  historyLog: string[];
  canUndo: boolean;
  canRedo: boolean;
  undoLabel?: string;
  redoLabel?: string;
  timings: Record<string, number>;
}

export interface InspectInfo {
  lat: number;
  lon: number;
  elevation: number;
  water: 'ocean' | 'lake' | 'salt lake' | 'salt flat' | null;
  plate: { id: number; name: string; speed: number; dir: string };
  crust: { continental: boolean; thickness: number; age: number };
  setting: string;
  /** Bedrock (under any loose sediment) and its age. */
  bedrock: string | null;
  /** Loose material over the bedrock: what it is, how thick, what lies under it. */
  surface: string | null;
  boundary: string | null;
  tempMonthly: number[];
  precipMonthly: number[];
  koppen: string | null;
  koppenName: string | null;
  biome: string;
  discharge: number;
  runoff: number;
  /** Monthly river flow (m³/s) where a river runs. */
  flowMonthly: number[] | null;
  /** How the river's flow varies through the year, and what feeds it. */
  regime: string | null;
  lake: { description: string; areaKm2: number; inflow: number; outflow: number } | null;
  /** Annual potential and actual evapotranspiration (mm). */
  pet: number;
  aet: number;
  snowMonths: number;
  glacier: boolean;
  wetland: string | null;
  /** Drains to a closed basin rather than the sea. */
  endorheic: boolean;
  /** Glacier, ice cap, ice sheet or ice shelf here today. */
  ice: string | null;
  /** What the last ice age did here. */
  iceAge: string | null;
}

/** Simulation worker → main thread. */
export type Response =
  | { type: 'progress'; stage: string; frac: number }
  | { type: 'ready'; summary: WorldSummary; reason: string; report?: EditReport; version: number }
  /** Brush preview: part of the overview (pyramid level 2); x0 may lie outside the image and wraps. */
  | { type: 'patch'; x0: number; y0: number; w: number; h: number; bitmap: ImageBitmap }
  | { type: 'inspect'; id: number; info: InspectInfo | null }
  | { type: 'file'; name: string; mime: string; buffer: ArrayBuffer }
  | { type: 'error'; message: string };
