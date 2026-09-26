import { z } from "zod";
import { NIO } from "../types/ocean.js";

const isoDate = z
  .string()
  .regex(/^\d{4}-\d{2}-\d{2}$/, "date must be YYYY-MM-DD")
  .refine((value) => !Number.isNaN(Date.parse(value)), "invalid calendar date");

export const profileQuery = z.object({
  lat: z.coerce.number().min(NIO.latMin).max(NIO.latMax),
  lon: z.coerce.number().min(NIO.lonMin).max(NIO.lonMax),
  date: isoDate,
});

export const layerQuery = z.object({
  date: isoDate,
  depth: z.coerce
    .number()
    .refine((d) => (NIO.depths as readonly number[]).includes(d), "depth must be a standard OceanEmbed level"),
  variable: z.enum(["temperature", "salinity", "uncertainty_theta", "uncertainty_sp"]),
});

export const indicesQuery = z.object({
  date: isoDate,
  index: z.enum(["tchp", "mld", "z20", "blt", "cip"]),
});

export const advisoryBody = z.object({
  lat: z.coerce.number().min(NIO.latMin).max(NIO.latMax),
  lon: z.coerce.number().min(NIO.lonMin).max(NIO.lonMax),
  date: isoDate,
  sst: z.coerce.number(),
  tchp: z.coerce.number(),
  mld: z.coerce.number(),
  z20: z.coerce.number(),
  inversion_flag: z.coerce.boolean(),
  uncertainty: z.coerce.number().min(0),
});

export type ProfileQuery = z.infer<typeof profileQuery>;
export type LayerQuery = z.infer<typeof layerQuery>;
export type IndicesQuery = z.infer<typeof indicesQuery>;
export type AdvisoryBody = z.infer<typeof advisoryBody>;
