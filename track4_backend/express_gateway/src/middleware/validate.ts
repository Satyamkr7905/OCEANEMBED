import { NextFunction, Request, Response } from "express";
import { ZodError, ZodTypeAny } from "zod";

export function validate(schema: ZodTypeAny, where: "query" | "body" = "query") {
  return (req: Request, _res: Response, next: NextFunction): void => {
    const parsed = schema.safeParse(where === "query" ? req.query : req.body);
    if (!parsed.success) {
      next(parsed.error);
      return;
    }
    if (where === "query") {
      req.query = parsed.data as Request["query"];
    } else {
      req.body = parsed.data;
    }
    next();
  };
}

export function formatZod(error: ZodError): { message: string; issues: unknown } {
  return {
    message: "Request validation failed",
    issues: error.flatten(),
  };
}
