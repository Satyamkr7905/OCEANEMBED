# OceanEmbed

SIH 26066 · Ministry of Earth Sciences / INCOIS

OceanEmbed reconstructs the North Indian Ocean from the surface down to 1000 m using satellite observations. Satellites only see SST, salinity, sea level, and winds. The model fills the water column so a duty desk can read tropical cyclone heat potential (TCHP), mixed-layer depth, the 20 °C isotherm, and barrier-layer thickness at a clicked station.

The live console is a maritime glass dashboard over a looping ocean background, with a MapLibre basin map and three vertical sounding charts.

## What the console shows

| Surface | What you get |
|---|---|
| **Amphan case study** | Dates **16–20 May 2020**, click inside **8–24°N, 84–92°E**. Soundings and TCHP path (115.2 → 92.4 → 62.8 → 42.1 → 35.6 kJ/cm²) come from `track4_backend/fastapi_engine/static/amphan_case_study.json`. |
| **Other dates / locations** | FastAPI runs the trained PyTorch weights (`oceanembed_real.pth`) on the available surface cube, then TEOS-10 diagnostics. |
| **Map heatmap** | Comes from `GET /ocean/layer` and `GET /ocean/indices`. It is empty when the inference service is down. |
| **AI advisory** | Narrates the **same numbers** the sounding already computed. It does not invent TCHP, MLD, or Z₂₀. Needs `GEMINI_API_KEY` (or `OPENAI_API_KEY`) in the FastAPI `.env`; otherwise a local bulletin is used. |

## Architecture

```
Browser (Next.js :3000)
        │  NEXT_PUBLIC_API_URL
        ▼
Express gateway (:8081)   /api/v1/ocean/*
        │  INFERENCE_URL
        ▼
FastAPI engine (:8000)    profile · indices · advisory · health
        │
        ├─ Amphan JSON intercept (BoB, 16–20 May 2020)
        ├─ PyTorch OceanEmbed weights
        └─ TEOS-10 derived indices (TCHP, MLD, Z20, BLT)
```

## How to run

Use three terminals from the repo root. Visit **`http://127.0.0.1`**, not `http://0.0.0.0`.

**1. FastAPI inference** (`http://127.0.0.1:8000/health`, docs at `/docs`)

```powershell
cd track4_backend
python -m uvicorn fastapi_engine.app:app --host 0.0.0.0 --port 8000
```

Optional keys in `track4_backend/fastapi_engine/.env`:

```
GEMINI_API_KEY=
OPENAI_API_KEY=
```

**2. Express gateway** (`http://127.0.0.1:8081/api/v1/health`)

```powershell
cd track4_backend/express_gateway
npx tsx watch src/index.ts
```

`track4_backend/express_gateway/.env` should have `PORT=8081` and `INFERENCE_URL=http://127.0.0.1:8000`.

**3. Next.js console** (`http://localhost:3000`)

```powershell
cd track5_frontend
npm install
npm run dev
```

`track5_frontend/.env.local`:

```
NEXT_PUBLIC_API_URL=http://127.0.0.1:8081/api/v1
```

Python deps for the engine and training path: `pip install -r requirements.txt` plus `pip install -r track4_backend/fastapi_engine/requirements.txt`.

## Using the console

1. The ocean clip loops behind the whole page. Open the console and click a marine pixel.
2. Open the console. Metric cards show TCHP, MLD, Z₂₀, BLT, and inference latency.
3. Click a pixel in the North Indian Ocean. Charts below the map update: thermal profile, salinity/density + barrier layer, thermocline gradient / TCHP layers.
4. For the Amphan cold-wake demo: date **16–20 May 2020**, click inside the Bay of Bengal box above, then scrub the cyclone timeline.
5. **Generate INCOIS Advisory Dispatch** writes a duty-desk bulletin from the current sounding.

## Repository layout

| Path | Role |
|---|---|
| `track1_data_engine/` | Ingest, regrid, climatology, Zarr conversion |
| `track2_model_engine/` | OceanEmbed architecture, losses, train/export |
| `track3_validation_engine/` | ARGO/RAMA matchers, TEOS-10 diagnostics, plots |
| `track4_backend/fastapi_engine/` | Inference + advisory API |
| `track4_backend/express_gateway/` | Public REST gateway |
| `track5_frontend/` | Next.js maritime console |
| `data/raw/` | May 2020 CMEMS sample (GLORYS + altimetry) |
| `data/processed/training_tensors.pt` | Tensors used by local training |
| `oceanembed/` | Pipeline output root (masks, climatology, smoke Zarr) |
| `download_real_sample.py` | Pull the May 2020 CMEMS slice |
| `prepare_training_tensors.py` | Build `training_tensors.pt` |
| `run_local_training.py` | Train weights → `track4_backend/fastapi_engine/weights/oceanembed_real.pth` |
| `DEMO_SCRIPT.md` | 60-second jury walkthrough |

## Training (optional)

```powershell
python download_real_sample.py
python prepare_training_tensors.py
python run_local_training.py
```

The demo does not need a retraining run if `oceanembed_real.pth` is already present.

## Grid and physics

- Basin: 5–30°N, 45–105°E at 0.25° (~28 km)
- 15 standard depths: 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m
- 7-day surface window, 12 input channels
- MLD: density criterion Δρ = 0.03 kg/m³
- TCHP: heat content above the 26 °C isotherm
- Barrier layer: ILD − MLD
