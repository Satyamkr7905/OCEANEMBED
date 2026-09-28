import type { IndexVar, MapLayerId, OceanProfile, RasterGrid } from "@/types/ocean";
import { DEPTHS } from "@/types/ocean";

const DEFAULT_BASE = process.env.NEXT_PUBLIC_API_URL ?? "/api/v1";

export class ApiOfflineError extends Error {
  constructor(message = "OceanEmbed gateway unreachable") {
    super(message);
    this.name = "ApiOfflineError";
  }
}

function unwrapProfile(payload: Record<string, unknown>): OceanProfile | null {
  const props =
    (payload.features as Array<{ properties?: Record<string, unknown> }> | undefined)?.[0]?.properties ?? payload;
  if (!props || (!Array.isArray(props.depths) && !Array.isArray(props.potential_temperature))) {
    return null;
  }
  const depths = (props.depths as number[] | undefined) ?? [...DEPTHS];
  const theta = (props.potential_temperature as number[]) ?? [];
  const sp = (props.practical_salinity as number[]) ?? [];
  const sigT = (props.uncertainty_theta as number[]) ?? theta.map(() => 0.2);
  const sigS = (props.uncertainty_sp as number[]) ?? sp.map(() => 0.05);
  const inversion = theta.some((t, i) => i > 0 && t > theta[i - 1] + 0.05);
  return {
    lat: Number(props.lat),
    lon: Number(props.lon),
    date: String(props.date),
    depths,
    potentialTemperature: theta.map(Number),
    practicalSalinity: sp.map(Number),
    conservativeTemperature: ((props.conservative_temperature as number[]) ?? theta).map(Number),
    absoluteSalinity: ((props.absolute_salinity as number[]) ?? sp).map(Number),
    uncertaintyTheta: sigT.map(Number),
    uncertaintySp: sigS.map(Number),
    argoTemperature: Array.isArray(props.argo_temperature)
      ? (props.argo_temperature as Array<number | null>).map((v) => (v == null ? null : Number(v)))
      : depths.map(() => null),
    argoSalinity: Array.isArray(props.argo_salinity)
      ? (props.argo_salinity as Array<number | null>).map((v) => (v == null ? null : Number(v)))
      : depths.map(() => null),
    tchp: Number(props.TCHP ?? 0),
    mld: Number(props.MLD ?? 0),
    z20: Number(props.Z20 ?? 0),
    blt: props.BLT == null ? null : Number(props.BLT),
    cip: Number(props.CIP ?? 0),
    inversion,
    inferenceMs: Number((props.metadata as { inference_time_ms?: number } | undefined)?.inference_time_ms ?? 0),
    source: "api",
    backend: String((props.metadata as { backend?: string } | undefined)?.backend ?? "api"),
  };
}

export async function fetchHealth(base = DEFAULT_BASE): Promise<{ ok: boolean; detail?: unknown }> {
  try {
    const res = await fetch(`${base}/health`, { cache: "no-store" });
    if (!res.ok) return { ok: false };
    return { ok: true, detail: await res.json() };
  } catch {
    return { ok: false };
  }
}

export async function fetchProfile(
  lat: number,
  lon: number,
  date: string,
  base = DEFAULT_BASE,
): Promise<OceanProfile> {
  const url = `${base}/ocean/profile?lat=${lat}&lon=${lon}&date=${date}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new ApiOfflineError(`profile ${res.status}`);
  const json = (await res.json()) as Record<string, unknown>;
  const parsed = unwrapProfile(json);
  if (!parsed) throw new ApiOfflineError("malformed profile payload");
  return parsed;
}

const INDEX_LAYERS = new Set<MapLayerId>(["tchp", "mld", "z20", "blt", "cip"]);

function layerVariable(layer: MapLayerId): "temperature" | "salinity" | "uncertainty_theta" {
  if (layer === "sp") return "salinity";
  if (layer === "sigma_theta") return "uncertainty_theta";
  return "temperature";
}

export async function fetchRaster(
  layer: MapLayerId,
  date: string,
  depth: number,
  base = DEFAULT_BASE,
): Promise<RasterGrid> {
  const url = INDEX_LAYERS.has(layer)
    ? `${base}/ocean/indices?date=${date}&index=${layer}`
    : `${base}/ocean/layer?date=${date}&depth=${depth}&variable=${layerVariable(layer)}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new ApiOfflineError(`raster ${res.status}`);
  const json = (await res.json()) as {
    features?: Array<{ properties?: Record<string, unknown> }>;
  };
  const props = json.features?.[0]?.properties;
  const values = props?.grid as Array<Array<number | null>> | undefined;
  const lat = props?.lat as number[] | undefined;
  const lon = props?.lon as number[] | undefined;
  if (!values || !lat || !lon) throw new ApiOfflineError("malformed raster payload");
  return {
    lat: lat.map(Number),
    lon: lon.map(Number),
    values: values.map((row) => row.map((v) => (v == null || Number.isNaN(Number(v)) ? null : Number(v)))),
    units: String(props?.units ?? ""),
    label: String(props?.variable ?? props?.index ?? layer),
  };
}

export async function pingGateway(base = DEFAULT_BASE): Promise<boolean> {
  const h = await fetchHealth(base);
  return h.ok;
}

export interface AdvisoryRequest {
  lat: number;
  lon: number;
  date: string;
  sst: number;
  tchp: number;
  mld: number;
  z20: number;
  inversion_flag: boolean;
  uncertainty: number;
}

export interface AdvisoryResponse {
  status: string;
  advisory: string;
}

export async function fetchAdvisory(
  payload: AdvisoryRequest,
  base = DEFAULT_BASE,
): Promise<AdvisoryResponse> {
  const res = await fetch(`${base}/ocean/advisory`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new ApiOfflineError(`advisory ${res.status}`);
  }
  const json = (await res.json()) as AdvisoryResponse;
  if (!json.advisory) {
    throw new ApiOfflineError("malformed advisory payload");
  }
  return { status: json.status || "success", advisory: json.advisory };
}

export type { IndexVar };
