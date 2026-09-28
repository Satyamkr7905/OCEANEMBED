import axios, { AxiosInstance } from "axios";
import type { GridDto, IndexName, LayerVariable, ProfileDto } from "../types/ocean.js";

function client(): AxiosInstance {
  const baseURL = process.env.INFERENCE_URL ?? "http://127.0.0.1:8000";
  return axios.create({ baseURL, timeout: 60_000 });
}

/**
 * Execute an Axios request with retry logic designed for cloud cold starts (Render free tier).
 * Retries on 502, 503, 504 HTTP status codes or connection network errors.
 */
async function requestWithRetry<T>(requestFn: () => Promise<{ data: T }>, retries = 5, delayMs = 3000): Promise<T> {
  let lastError: unknown;
  for (let i = 0; i <= retries; i++) {
    try {
      const response = await requestFn();
      return response.data;
    } catch (err: any) {
      lastError = err;
      const status = err.response?.status;
      const isColdStartError =
        !err.response || status === 502 || status === 503 || status === 504 || status === 524;

      if (isColdStartError && i < retries) {
        console.log(`[Express Gateway] FastAPI engine cold-starting (attempt ${i + 1}/${retries + 1}). Retrying in ${delayMs}ms...`);
        await new Promise((resolve) => setTimeout(resolve, delayMs));
        continue;
      }
      throw err;
    }
  }
  throw lastError;
}

export async function fetchProfile(lat: number, lon: number, date: string): Promise<ProfileDto> {
  return requestWithRetry(() => client().post<ProfileDto>("/predict/profile", { lat, lon, date }), 6, 3500);
}

export async function fetchLayer(
  date: string,
  depth: number,
  variable: LayerVariable,
): Promise<GridDto> {
  return requestWithRetry(
    () =>
      client().post<GridDto>("/predict/grid", {
        date,
        depth,
        variable,
        format: "json",
      }),
    6,
    3500,
  );
}

export async function fetchIndex(date: string, index: IndexName): Promise<GridDto> {
  return requestWithRetry(() => client().post<GridDto>("/predict/indices", { date, index }), 6, 3500);
}

export async function fetchHealth(): Promise<unknown> {
  return requestWithRetry(() => client().get("/health"), 4, 2500);
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
  return requestWithRetry(
    () => client().post<AdvisoryResponse>("/advisory/generate", payload, { timeout: 90_000 }),
    5,
    3000,
  );
}

