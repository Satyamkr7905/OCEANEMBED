import { Request, Response, NextFunction } from "express";
import { cacheSet } from "../middleware/cache.js";
import { fetchAdvisory, fetchHealth, fetchIndex, fetchLayer, fetchProfile } from "../services/inferenceClient.js";
import type { GridDto, ProfileDto } from "../types/ocean.js";
import type { AdvisoryBody, IndicesQuery, LayerQuery, ProfileQuery } from "../middleware/schemas.js";

function bboxPolygon(bbox: [number, number, number, number]): number[][][] {
  const [west, south, east, north] = bbox;
  return [[
    [west, south],
    [east, south],
    [east, north],
    [west, north],
    [west, south],
  ]];
}

export function profileToGeoJson(dto: ProfileDto) {
  return {
    type: "FeatureCollection" as const,
    features: [
      {
        type: "Feature" as const,
        geometry: { type: "Point" as const, coordinates: [dto.lon, dto.lat] as [number, number] },
        properties: { ...dto },
      },
    ],
  };
}

export function gridToGeoJson(dto: GridDto, extra: Record<string, unknown> = {}) {
  return {
    type: "FeatureCollection" as const,
    features: [
      {
        type: "Feature" as const,
        geometry: { type: "Polygon" as const, coordinates: bboxPolygon(dto.bbox) },
        properties: {
          date: dto.date,
          depth: dto.depth,
          variable: dto.variable,
          nrows: dto.nrows,
          ncols: dto.ncols,
          bbox: dto.bbox,
          units: dto.units ?? extra.units,
          inference_time_ms: dto.inference_time_ms,
          model_version: dto.model_version,
          grid: dto.values,
          lat: dto.lat,
          lon: dto.lon,
          ...extra,
        },
      },
    ],
  };
}

export async function getProfile(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const q = req.query as unknown as ProfileQuery;
    const dto = await fetchProfile(q.lat, q.lon, q.date);
    const body = profileToGeoJson(dto);
    cacheSet(req, res, body);
    res.setHeader("X-Cache", "MISS");
    res.json(body);
  } catch (err) {
    next(err);
  }
}

export async function getLayer(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const q = req.query as unknown as LayerQuery;
    const dto = await fetchLayer(q.date, q.depth, q.variable);
    const body = gridToGeoJson(dto);
    cacheSet(req, res, body);
    res.setHeader("X-Cache", "MISS");
    res.json(body);
  } catch (err) {
    next(err);
  }
}

export async function getIndices(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const q = req.query as unknown as IndicesQuery;
    const dto = await fetchIndex(q.date, q.index);
    const units: Record<string, string> = {
      tchp: "kJ cm-2",
      mld: "m",
      z20: "m",
      blt: "m",
      cip: "0-1",
    };
    const body = gridToGeoJson(dto, { index: q.index, units: units[q.index] });
    cacheSet(req, res, body);
    res.setHeader("X-Cache", "MISS");
    res.json(body);
  } catch (err) {
    next(err);
  }
}

export async function postAdvisory(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const body = req.body as AdvisoryBody;
    const result = await fetchAdvisory(body);
    res.json({
      status: result.status ?? "success",
      advisory: result.advisory,
    });
  } catch (err) {
    next(err);
  }
}

export async function getHealth(_req: Request, res: Response): Promise<void> {
  let inference: unknown;
  try {
    inference = await fetchHealth();
  } catch (err) {
    inference = {
      status: "down",
      detail: err instanceof Error ? err.message : "unreachable",
    };
  }
  res.json({
    status: "ok",
    gateway: "oceanembed-express",
    inference,
  });
}
