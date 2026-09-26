export const NIO = {
  latMin: 5,
  latMax: 30,
  lonMin: 45,
  lonMax: 105,
  depths: [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] as const,
} as const;

export type DepthM = (typeof NIO.depths)[number];
export type LayerVariable = "temperature" | "salinity" | "uncertainty_theta" | "uncertainty_sp";
export type IndexName = "tchp" | "mld" | "z20" | "blt" | "cip";

export interface ProfileDto {
  lat: number;
  lon: number;
  date: string;
  depths: number[];
  potential_temperature: Array<number | null>;
  practical_salinity: Array<number | null>;
  conservative_temperature: Array<number | null>;
  absolute_salinity: Array<number | null>;
  uncertainty_theta: Array<number | null>;
  uncertainty_sp: Array<number | null>;
  TCHP: number | null;
  MLD: number | null;
  Z20: number | null;
  BLT: number | null;
  CIP: number | null;
  metadata: {
    model_version: string;
    inference_time_ms: number;
    teos10_ms: number;
    calibration_applied: boolean;
    backend: string;
    synthetic_inputs: boolean;
    grid_j: number;
    grid_i: number;
  };
}

export interface GridDto {
  date: string;
  depth: number | null;
  variable: string;
  lat: number[];
  lon: number[];
  values: Array<Array<number | null>>;
  nrows: number;
  ncols: number;
  bbox: [number, number, number, number];
  inference_time_ms: number;
  model_version?: string;
  units?: string;
}

export interface GeoJsonPointCollection {
  type: "FeatureCollection";
  features: Array<{
    type: "Feature";
    geometry: { type: "Point"; coordinates: [number, number] };
    properties: Record<string, unknown>;
  }>;
}

export interface GeoJsonGridCollection {
  type: "FeatureCollection";
  features: Array<{
    type: "Feature";
    geometry: {
      type: "Polygon";
      coordinates: number[][][];
    };
    properties: Record<string, unknown>;
  }>;
}
