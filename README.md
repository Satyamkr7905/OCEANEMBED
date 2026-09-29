# OceanEmbed

<div align="center">

### Physics-Informed AI for 3D Subsurface Ocean Reconstruction

[![Smart India Hackathon 2025](https://img.shields.io/badge/Smart%20India%20Hackathon-2025-blue?style=for-the-badge)](https://sih.gov.in/)
[![Problem SIH26066](https://img.shields.io/badge/Problem-SIH26066-orange?style=for-the-badge)](https://sih.gov.in/)
![Team Bitminds](https://img.shields.io/badge/Team-Bitminds-purple?style=for-the-badge)
[![License MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[Live Demo](https://oceanembed-phi.vercel.app/) • [Watch Demo Video](https://youtu.be/GUdMyuLT980) • [Full Demo Recording](https://drive.google.com/file/d/1NlPlJ7SpMDJHK6ciq_2xGxzthqxUZaIK/view?usp=drivesdk) • [Report Bug](https://github.com/Satyamkr7905/OCEANEMBED/issues) • [Request Feature](https://github.com/Satyamkr7905/OCEANEMBED/issues)

</div>

## 📖 Overview

OceanEmbed is an end-to-end, physics-informed deep learning system that reconstructs the hidden 3D temperature and salinity structure of the North Indian Ocean using only surface satellite observations.

The system generates daily, basin-scale 3D ocean fields at 0.25° × 0.25° spatial resolution across 15 standard depths (0 m to 1000 m) over the domain 5°N–30°N, 45°E–105°E, covering both the Arabian Sea and the Bay of Bengal.

### Why This Matters
India loses thousands of crores annually to cyclones. Forecasters can see storms from space, but they cannot see the warm water reservoir below the surface that fuels rapid intensification. ARGO floats provide some subsurface data, but their coverage is dangerously sparse — especially during cyclones when data is needed most.

OceanEmbed solves this by learning the nonlinear relationship between surface satellite observations and the hidden ocean interior, delivering actionable ocean intelligence to INCOIS duty officers.

---

## 🎯 Problem Statement

**SIH26066 — MoES / INCOIS**  
*Development of a Satellite Embedding-Based Deep Learning Framework to reconstruct depth-wise subsurface temperature from daily surface satellite observations at 0.25° spatial resolution for the North Indian Ocean.*

| Attribute | Detail |
|---|---|
| **Theme** | Disaster Management |
| **Category** | Software |
| **Organisation** | Ministry of Earth Sciences (MoES) / INCOIS |
| **Domain** | North Indian Ocean (5°N–30°N, 45°E–105°E) |

---

## 🎥 Demo Videos

| Video | Description | Link |
|---|---|---|
| **YouTube Pitch** | Official SIH submission video with project walkthrough | [Watch on YouTube](https://youtu.be/GUdMyuLT980) |
| **Full Demo Recording** | Complete dashboard walkthrough with all features | [Watch on Google Drive](https://drive.google.com/file/d/1NlPlJ7SpMDJHK6ciq_2xGxzthqxUZaIK/view?usp=drivesdk) |
| **Live Dashboard** | Interactive operational console | [oceanembed-phi.vercel.app](https://oceanembed-phi.vercel.app/) |

---

## 🚀 Key Features

### Physics-Informed AI
- **Ekman Pumping**: Wind-driven vertical velocity computed from wind stress curl
- **Regularized Coriolis**: Prevents equatorial singularity with bounded formulation
- **Climatological Anomaly Decomposition**: Residual learning over 20-year daily climatology
- **TEOS-10 Thermodynamics**: International standard for seawater density and heat content

### Deep Learning Architecture
- **Dual-Stream Encoder**: Separate branches for local eddies (receptive ~7 cells) and basin-scale waves (receptive ~60 cells)
- **Latent Ocean Embedding**: 512-dimensional compressed representation
- **Depth-Attention Decoder**: Learnable depth queries with multi-head cross-attention
- **Sub-Basin Heads**: Specialized projection weights for Arabian Sea and Bay of Bengal
- **Joint T-S Prediction**: Simultaneous temperature and salinity output for accurate density
- **Calibrated Uncertainty**: Monte Carlo Dropout + ensemble + temperature scaling

### Operational Dashboard
- **Interactive 2D/3D Geospatial Viewer**: Click-to-profile at any coordinate
- **Vertical Diagnostics**: Three synchronized charts (thermal profile, halocline, thermocline gradient)
- **ARGO Comparison View**: Side-by-side validation against real float soundings
- **Cyclone Case Study Playback**: Track TCHP evolution during historical events
- **Uncertainty Layers**: Confidence intervals per depth
- **INCOIS Advisory Copilot**: One-click AI-generated maritime bulletins

### Derived Oceanographic Indices
- **TCHP**: Tropical Cyclone Heat Potential (kJ/cm²)
- **MLD**: Mixed Layer Depth (m)
- **Z₂₀**: 20°C Isotherm Depth (m)
- **BLT**: Barrier Layer Thickness (m)
- **CIP**: Cyclone Intensification Potential

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                     SATELLITE INPUT (0.25° × 0.25°)                  │
│             [SST] [SSS] [SLA] [Currents U,V] [Winds U,V]            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 PHYSICAL INDUCTIVE BIAS PRE-PROCESSOR               │
│         • Coriolis (f̃ = sign(f)·max(|f|, f₀))                       │
│         • Ekman Pumping (wₑ = ∇×τ / ρ₀f̃)                           │
│         • Climatology Anomaly Decomposition                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                DUAL-STREAM SPATIO-TEMPORAL ENCODER                   │
│   ┌─────────────────┐                     ┌─────────────────┐       │
│   │    STREAM A     │                     │    STREAM B     │       │
│   │  Local Eddies   │                     │   Basin Waves   │       │
│   │  Receptive ~7   │                     │  Receptive ~60  │       │
│   └────────┬────────┘                     └────────┬────────┘       │
│            └──────────────────┬────────────────────┘                │
│                               ▼                                     │
│                LATENT OCEAN EMBEDDING (512-dim)                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                DEPTH-WISE CROSS-ATTENTION DECODER                   │
│         • 15 Learnable Depth Queries                                │
│         • Sub-Basin Heads (Arabian Sea / Bay of Bengal)              │
│         • Joint Temperature + Salinity + Uncertainty Heads          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    OUTPUT: 3D THERMOHALINE FIELDS                   │
│         • Temperature (15 depths) + σ_θ                             │
│         • Salinity (15 depths) + σ_S                                │
│         • Derived Indices via TEOS-10                               │
└──────────────────────────────┬──────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Frontend
| Technology | Purpose |
|---|---|
| **Next.js 14** | React framework with App Router |
| **TypeScript** | Type-safe development |
| **TailwindCSS** | Utility-first styling |
| **Deck.gl** | WebGL-powered geospatial visualization |
| **Plotly.js** | Scientific charting |
| **Leaflet** | Interactive maps |
| **Zustand** | State management |
| **React Query** | Server state management |

### Backend
| Technology | Purpose |
|---|---|
| **FastAPI** | ML inference and scientific computation |
| **Express.js** | API gateway and orchestration |
| **PyTorch** | Deep learning framework |
| **PyTorch-Lightning** | Training pipeline |
| **gsw** | TEOS-10 Gibbs SeaWater library |
| **PCHIP** | Shape-preserving vertical interpolation |
| **Google Gemini 2.5** | LLM advisory copilot |

### Data & Storage
| Technology | Purpose |
|---|---|
| **Zarr** | Chunked array storage |
| **xarray** | Multidimensional labeled arrays |
| **Dask** | Parallel computing |
| **xesmf** | Conservative regridding |
| **Redis** | Caching layer |
| **PostgreSQL** | Metadata and user data |

### DevOps
| Technology | Purpose |
|---|---|
| **Docker** | Containerization |
| **Render** | Backend deployment |
| **Vercel** | Frontend deployment |
| **GitHub Actions** | CI/CD |

---

## 📂 Project Structure

```text
OceanEmbed/
├── frontend/                  # Next.js 14 frontend
│   ├── app/
│   │   ├── page.tsx           # Landing page
│   │   ├── dashboard/         # Main dashboard
│   │   │   ├── page.tsx
│   │   │   ├── map/           # Interactive map
│   │   │   ├── profile/       # Vertical profile viewer
│   │   │   ├── cyclone/       # Cyclone case studies
│   │   │   └── validation/    # ARGO validation view
│   │   └── api/proxy/         # API proxy routes
│   ├── components/
│   │   ├── MapViewer.tsx      # Deck.gl map component
│   │   ├── ProfileViewer.tsx  # Plotly vertical profile
│   │   ├── UncertaintyBand.tsx# Uncertainty visualization
│   │   ├── CyclonePlayback.tsx# Cyclone timeline
│   │   └── ArgoComparison.tsx # ARGO overlay
│   └── lib/
│       ├── api.ts             # API client
│       └── hooks/             # Custom hooks
│
├── backend-fastapi/           # FastAPI ML service
│   ├── main.py                # Application entry
│   ├── api/
│   │   ├── predict.py         # Prediction endpoints
│   │   ├── indices.py         # Derived index endpoints
│   │   └── validation.py      # Validation endpoints
│   ├── models/
│   │   ├── oceanembed.py      # Neural network
│   │   ├── encoder.py         # Dual-stream encoder
│   │   └── decoder.py         # Depth attention decoder
│   ├── physics/
│   │   ├── ekman.py           # Ekman pumping
│   │   ├── coriolis.py        # Regularized Coriolis
│   │   └── teos10.py          # TEOS-10 conversions
│   ├── utils/
│   │   ├── pchip.py           # PCHIP interpolation
│   │   └── climatology.py     # Anomaly decomposition
│   └── requirements.txt
│
├── backend-express/           # Express.js gateway
│   ├── server.js              # Application entry
│   ├── routes/
│   │   ├── auth.js            # JWT authentication
│   │   ├── forecast.js        # Forecast proxy
│   │   └── argo.js            # ARGO proxy
│   ├── middleware/
│   │   ├── cache.js           # Redis caching
│   │   ├── rateLimit.js       # Rate limiting
│   │   └── validation.js      # Zod validation
│   └── package.json
│
├── ml/                        # ML training pipeline
│   ├── data/
│   │   ├── ingestion.py       # Satellite download
│   │   ├── regridding.py      # Conservative regridding
│   │   └── zarr_store.py      # Zarr storage
│   ├── training/
│   │   ├── train.py           # Training loop
│   │   ├── losses.py          # Physics-informed loss
│   │   └── configs/           # YAML configs
│   ├── evaluation/
│   │   ├── argo_validation.py # ARGO co-location
│   │   ├── rama_validation.py # RAMA validation
│   │   └── calibration.py     # Uncertainty calibration
│   └── notebooks/
│       ├── 01_data_exploration.ipynb
│       ├── 02_model_development.ipynb
│       └── 03_ablation_studies.ipynb
│
├── docker-compose.yml         # Local development
├── Dockerfile.fastapi         # FastAPI container
├── Dockerfile.express         # Express container
├── README.md                  # This file
└── LICENSE
```

---

## 🚦 Getting Started

### Prerequisites
- Node.js 20+
- Python 3.10+
- Docker & Docker Compose (optional)
- NVIDIA T4 GPU (recommended for inference)

### Local Development

#### 1. Clone the Repository
```bash
git clone https://github.com/Satyamkr7905/OCEANEMBED.git
cd OCEANEMBED
```

#### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev # Runs on http://localhost:3000
```

Environment variables (`.env.local`):
```env
NEXT_PUBLIC_API_URL=http://localhost:8081
NEXT_PUBLIC_WS_URL=ws://localhost:8081
```

#### 3. FastAPI Backend Setup
```bash
cd backend-fastapi
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000 # Runs on http://localhost:8000
```

#### 4. Express Gateway Setup
```bash
cd backend-express
npm install
npm run dev # Runs on http://localhost:8081
```

Environment variables (`.env`):
```env
FASTAPI_URL=http://localhost:8000
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost:5432/oceanembed
JWT_SECRET=your-secret-key
```

#### 5. Docker Compose (All Services)
```bash
docker-compose up --build
```
This starts:
- Frontend on port 3000
- Express gateway on port 8081
- FastAPI backend on port 8000
- Redis on port 6379
- PostgreSQL on port 5432

---

## 🔌 API Reference

### FastAPI Endpoints
| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Health check |
| GET | `/docs` | Interactive Swagger documentation |
| POST | `/predict/point` | Single coordinate prediction |
| POST | `/predict/basin` | Full basin prediction |
| POST | `/predict/batch` | Batch prediction |
| GET | `/indices/tchp` | Tropical Cyclone Heat Potential |
| GET | `/indices/mld` | Mixed Layer Depth |
| GET | `/indices/z20` | 20°C Isotherm Depth |
| GET | `/indices/blt` | Barrier Layer Thickness |
| GET | `/validate/argo` | ARGO co-location results |
| GET | `/validate/rama` | RAMA co-location results |
| GET | `/model/info` | Model metadata |

**Example Request:**
```bash
curl -X POST "https://oceanembed-fastapi.onrender.com/predict/point" \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 15.5,
    "lon": 86.3,
    "date": "2020-05-18"
  }'
```

**Example Response:**
```json
{
  "depths": [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000],
  "potential_temperature": [30.85, 30.82, 30.79, 30.65, 30.12, 28.45, 25.30, 21.80, 18.50, 15.60, 12.10, 8.40, 5.20, 4.10, 3.20],
  "practical_salinity": [32.15, 32.18, 32.22, 32.40, 32.85, 33.40, 34.10, 34.65, 34.85, 34.92, 34.95, 34.98, 35.01, 35.03, 35.05],
  "uncertainty_theta": [0.12, 0.14, 0.15, 0.18, 0.22, 0.28, 0.35, 0.42, 0.38, 0.32, 0.25, 0.18, 0.12, 0.09, 0.07],
  "uncertainty_salinity": [0.08, 0.09, 0.10, 0.12, 0.15, 0.18, 0.22, 0.25, 0.22, 0.19, 0.15, 0.11, 0.08, 0.06, 0.05],
  "indices": {
    "tchp": 62.8,
    "mld": 52.3,
    "z20": 94.0,
    "blt": 3.4
  },
  "metadata": {
    "model_version": "oceanembed-v1.0.0",
    "inference_time_ms": 142,
    "calibration_applied": true
  }
}
```

### Express Gateway Endpoints
| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/auth/login` | User authentication |
| POST | `/api/auth/register` | User registration |
| GET | `/api/forecast` | Proxy to FastAPI forecast |
| GET | `/api/argo` | Proxy to FastAPI ARGO |
| GET | `/api/cyclone` | Proxy to FastAPI cyclone |
| GET | `/api/health` | Health check |
| WS | `/ws/updates` | Real-time WebSocket |

---

## 📊 Data Sources

| Variable | Source | Product | Resolution |
|---|---|---|---|
| **SST** | NOAA | OISST v2.1 | 0.25° daily |
| **SSS** | NASA/ESA | SMAP/SMOS | 0.25° daily |
| **SSH/SLA** | CMEMS | SEALEVEL | 0.25° daily |
| **Currents** | CMEMS | SEALEVEL (geostrophic) | 0.25° daily |
| **Winds** | EUMETSAT/ECMWF | ASCAT/ERA5 | 0.25° daily |
| **Training Target** | CMEMS | GLORYS12V1 | 1/12°, 50 levels |
| **Validation** | Coriolis/INCOIS | ARGO | Profile-based |
| **Validation** | NOAA/PMEL | RAMA | Moored buoys |
| **Physics** | IOC/SCOR/IAPSO | TEOS-10 | Standard |

### Data Attribution
- Copernicus Marine Service — [marine.copernicus.eu](https://marine.copernicus.eu/)
- NOAA — [ncei.noaa.gov](https://www.ncei.noaa.gov/products/optimum-interpolation-sst)
- NASA — [podaac.jpl.nasa.gov](https://podaac.jpl.nasa.gov/SMAP)
- ESA — [esa.int](https://www.esa.int/Applications/Observing_the_Earth/FutureEO/SMOS)
- EUMETSAT — [eumetsat.int](https://user.eumetsat.int/)
- TEOS-10 — [teos-10.org](https://www.teos-10.org/)
- ARGO — [argo.ucsd.edu](https://argo.ucsd.edu/)

---

## 📈 Model Performance

| Metric | Value | Validation Source |
|---|---|---|
| **Thermocline RMSE (50–200 m)** | 0.88°C | ARGO Δt=0 + GLORYS holdout |
| **Climatology Skill Score** | > 0.78 | GLORYS 2020–21 holdout |
| **MSE Reduction** | 87.28% | Training convergence |
| **ARGO Validation RMSE** | 0.107°C | ARGO Δt=0, N=1,420 |
| **Uncertainty Coverage** | 71.4% @ ±1σ | ARGO Δt=0 validation set |
| **Neural Inference** | 142 ms | NVIDIA T4, batch=1 |
| **Full Pipeline** | 38.4 s | End-to-end benchmark |

### Cyclone Amphan Case Study (May 2020)
| Phase | TCHP (kJ/cm²) | MLD (m) | Z₂₀ (m) | SST (°C) |
|---|---|---|---|---|
| **Pre-Storm (May 16)** | 115.2 | 48.0 | 112.0 | 31.2 |
| **Eye Transit (May 18)** | 62.8 | 52.3 | 94.0 | 29.1 |
| **Post-Storm (May 20)** | 35.6 | 75.0 | 78.0 | 27.8 |
| **Net Change** | −79.6 | +27.0 | −34.0 | −3.4 |

---

## 🧪 Testing

### Run Backend Tests
```bash
cd backend-fastapi
pytest tests/ -v
```

### Run Frontend Tests
```bash
cd frontend
npm run test
```

### Run ML Pipeline Tests
```bash
cd ml
pytest tests/ -v
```

---

## 🤝 Contributing

We welcome contributions from the oceanographic and machine learning communities.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use TypeScript strict mode for frontend
- Write unit tests for new features
- Update documentation for API changes

---

## 👥 Team Bitminds

| Role | Responsibility |
|---|---|
| **ML/DL Lead** | Model architecture, training, ablation studies |
| **Backend Developer** | FastAPI, Express.js, database, caching |
| **Frontend Developer** | Next.js, visualization, UX |
| **Data Engineer** | Ingestion, regridding, Zarr storage |
| **DevOps Engineer** | Docker, CI/CD, monitoring |
| **Scientific Advisor** | Physics validation, methodology review |

- **Team ID:** 141034
- **Problem Statement:** SIH26066 (MoES / INCOIS)
- **Theme:** Disaster Management
- **Category:** Software

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Ministry of Earth Sciences (MoES) and INCOIS for the problem statement
- Smart India Hackathon 2025 organizers
- Copernicus Marine Service for open ocean reanalysis data
- NOAA, NASA, ESA, EUMETSAT for satellite products
- ARGO Program and RAMA Array for in-situ validation data
- Open-source community — PyTorch, FastAPI, Next.js, and the scientific Python ecosystem

---

## 📞 Contact

**Email:** satyamkumar9250@gmail.com

| Channel | Link |
|---|---|
| **Live Dashboard** | [oceanembed-phi.vercel.app](https://oceanembed-phi.vercel.app/) |
| **YouTube Pitch** | [youtu.be/GUdMyuLT980](https://youtu.be/GUdMyuLT980) |
| **Full Demo Video** | [Google Drive](https://drive.google.com/file/d/1NlPlJ7SpMDJHK6ciq_2xGxzthqxUZaIK/view?usp=drivesdk) |
| **GitHub Repository** | [github.com/Satyamkr7905/OCEANEMBED](https://github.com/Satyamkr7905/OCEANEMBED) |
| **FastAPI Backend** | [oceanembed-fastapi.onrender.com](https://oceanembed-fastapi.onrender.com/) |
| **Express Gateway** | [oceanembed-gateway.onrender.com](https://oceanembed-gateway.onrender.com/) |
| **API Documentation** | [oceanembed-fastapi.onrender.com/docs](https://oceanembed-fastapi.onrender.com/docs) |
| **Issues** | [GitHub Issues](https://github.com/Satyamkr7905/OCEANEMBED/issues) |

<div align="center">

### OceanEmbed
*From Satellite Skin to Ocean Depth*  
**Ready for INCOIS. Ready for India. Ready for the World.**

Made with 🌊 by Team Bitminds for SIH 2026

</div>
