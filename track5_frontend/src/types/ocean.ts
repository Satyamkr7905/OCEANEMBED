export const DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] as const;
export type DepthM = (typeof DEPTHS)[number];

export const NIO = {
  latMin: 5,
  latMax: 30,
  lonMin: 45,
  lonMax: 105,
  depths: DEPTHS,
} as const;

export type SurfaceVar = "sst" | "sss" | "sla" | "winds";
export type SubsurfaceVar = "theta" | "sp" | "sigma_theta";
export type IndexVar = "tchp" | "mld" | "z20" | "blt" | "cip";
export type MapLayerId = SurfaceVar | SubsurfaceVar | IndexVar;

export interface ProfilePoint {
  depth: number;
  theta: number;
  sp: number;
  sigmaTheta: number;
  sigmaSp: number;
  argoTheta?: number;
  argoSp?: number;
}

export interface OceanProfile {
  lat: number;
  lon: number;
  date: string;
  depths: number[];
  potentialTemperature: number[];
  practicalSalinity: number[];
  conservativeTemperature: number[];
  absoluteSalinity: number[];
  uncertaintyTheta: number[];
  uncertaintySp: number[];
  argoTemperature: Array<number | null>;
  argoSalinity: Array<number | null>;
  tchp: number;
  mld: number;
  z20: number;
  blt: number | null;
  cip: number;
  inversion: boolean;
  inferenceMs: number;
  source: "api";
  backend?: string;
}

export interface RasterGrid {
  lat: number[];
  lon: number[];
  values: Array<Array<number | null>>;
  units: string;
  label: string;
}

export interface ArgoFloat {
  id: string;
  lat: number;
  lon: number;
  date: string;
}

export interface RamaMooring {
  id: string;
  name: string;
  lat: number;
  lon: number;
}

export interface CycloneTrack {
  id: string;
  name: string;
  basin: "Bay of Bengal" | "Arabian Sea";
  dates: string[];
  coordinates: [number, number][];
  notes: string;
}

export interface ConsoleState {
  date: string;
  depth: DepthM;
  layer: MapLayerId;
  selected: { lat: number; lon: number } | null;
}
