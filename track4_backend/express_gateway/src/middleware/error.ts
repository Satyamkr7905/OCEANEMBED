import { NextFunction, Request, Response } from "express";
import { AxiosError } from "axios";
import { ZodError } from "zod";
import { formatZod } from "./validate.js";

export function errorHandler(err: unknown, _req: Request, res: Response, _next: NextFunction): void {
  if (err instanceof ZodError) {
    res.status(400).json({ error: formatZod(err) });
    return;
  }
  if (err instanceof AxiosError) {
    const status = err.response?.status ?? 502;
    res.status(status).json({
      error: "inference_upstream_error",
      detail: err.response?.data ?? err.message,
    });
    return;
  }
  const message = err instanceof Error ? err.message : "internal_error";
  res.status(500).json({ error: message });
}
