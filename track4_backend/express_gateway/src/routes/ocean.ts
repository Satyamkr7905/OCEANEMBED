import { Router } from "express";
import { cacheGet } from "../middleware/cache.js";
import { advisoryBody, indicesQuery, layerQuery, profileQuery } from "../middleware/schemas.js";
import { validate } from "../middleware/validate.js";
import { getHealth, getIndices, getLayer, getProfile, postAdvisory } from "../controllers/oceanController.js";

export const oceanRouter = Router();

oceanRouter.get("/health", getHealth);
oceanRouter.get("/ocean/profile", validate(profileQuery), cacheGet, getProfile);
oceanRouter.get("/ocean/layer", validate(layerQuery), cacheGet, getLayer);
oceanRouter.get("/ocean/indices", validate(indicesQuery), cacheGet, getIndices);
oceanRouter.post("/ocean/advisory", validate(advisoryBody, "body"), postAdvisory);
