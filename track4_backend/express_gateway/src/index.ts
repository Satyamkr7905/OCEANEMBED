import "dotenv/config";
import cors from "cors";
import express from "express";
import { errorHandler } from "./middleware/error.js";
import { oceanRouter } from "./routes/ocean.js";

const app = express();
app.disable("x-powered-by");
app.use(cors());
app.use(express.json({ limit: "4mb" }));
app.use("/api/v1", oceanRouter);
app.use(errorHandler);

const port = Number(process.env.PORT ?? 8081);
app.listen(port, () => {
  console.log(`OceanEmbed gateway listening on :${port}`);
});
