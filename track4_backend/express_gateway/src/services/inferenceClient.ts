import axios, { AxiosInstance } from "axios";
import type { GridDto, IndexName, LayerVariable, ProfileDto } from "../types/ocean.js";

function client(): AxiosInstance {
  const baseURL = process.env.INFERENCE_URL ?? "http://127.0.0.1:8000";
  return axios.create({ baseURL, timeout: 60_000 });
}

export async function fetchProfile(lat: number, lon: number, date: string): Promise<ProfileDto> {
  const { data } = await client().post<ProfileDto>("/predict/profile", { lat, lon, date });
  return data;
}

export async function fetchLayer(
  date: string,
  depth: number,
  variable: LayerVariable,
): Promise<GridDto> {
  const { data } = await client().post<GridDto>("/predict/grid", {
    date,
    depth,
    variable,
    format: "json",
  });
  return data;
}

export async function fetchIndex(date: string, index: IndexName): Promise<GridDto> {
  const { data } = await client().post<GridDto>("/predict/indices", { date, index });
  return data;
}

export async function fetchHealth(): Promise<unknown> {
  const { data } = await client().get("/health");
  return data;
}

export interface AdvisoryPayload {
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

export async function fetchAdvisory(payload: AdvisoryPayload): Promise<AdvisoryResponse> {
  const { data } = await client().post<AdvisoryResponse>("/advisory/generate", payload, {
    timeout: 90_000,
  });
  return data;
}
