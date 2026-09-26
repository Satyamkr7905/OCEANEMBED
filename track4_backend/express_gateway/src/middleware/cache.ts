import { createHash } from "node:crypto";
import { NextFunction, Request, Response } from "express";
import { getRedis } from "../services/redis.js";

const TTL = Number(process.env.CACHE_TTL_SECONDS ?? 86_400);

function keyFor(req: Request): string {
  const raw = `${req.method}:${req.originalUrl}`;
  return `oceanembed:${createHash("sha1").update(raw).digest("hex")}`;
}

export function cacheGet(req: Request, res: Response, next: NextFunction): void {
  const redis = getRedis();
  if (!redis) {
    next();
    return;
  }
  const key = keyFor(req);
  redis
    .get(key)
    .then((hit) => {
      if (!hit) {
        next();
        return;
      }
      res.setHeader("X-Cache", "HIT");
      res.type("application/json").send(hit);
    })
    .catch(() => next());
}

export function cacheSet(req: Request, _res: Response, body: unknown): void {
  const redis = getRedis();
  if (!redis) return;
  const key = keyFor(req);
  void redis.set(key, JSON.stringify(body), "EX", TTL).catch(() => undefined);
}
