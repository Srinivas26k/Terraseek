# TerraSeek — Technical Requirements Document (TRD)

**Version:** 1.0  
**Author:** Srinivas Nampalli
**Date:** April 2026  
**Status:** Draft  
**Stack:** Python (backend/ML) + React (frontend)

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
│   React + Vite Dashboard          External API Consumers        │
│   (Cloudflare Pages — free)       (curl, Python, JS)            │
└───────────────────────┬─────────────────────────────────────────┘
                        │ HTTPS
┌───────────────────────▼─────────────────────────────────────────┐
│                    EDGE LAYER (FREE)                             │
│              Cloudflare Workers + KV Cache                       │
│   Route: GET /forecast?lat=&lon=&hours=                          │
│   Cache hit → return KV value (TTL: 1hr)                        │
│   Cache miss → call Python inference service                     │
└───────────────────────┬─────────────────────────────────────────┘
                        │ HTTP (internal)
┌───────────────────────▼─────────────────────────────────────────┐
│               INFERENCE SERVICE (Python)                         │
│   FastAPI app — hosted on Hugging Face Spaces (free)            │
│   ┌────────────────────────────────────────────────┐            │
│   │  1. Open-Meteo API → raw weather features      │            │
│   │  2. Prophet → seasonality decomposition        │            │
│   │  3. TimesFM → 48h zero-shot forecast           │            │
│   │  4. Gemma 4 (AI Studio) → explanation text     │            │
│   └────────────────────────────────────────────────┘            │
└───────────────────────┬─────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────┐
│                    DATA SOURCES (all free)                       │
│   Open-Meteo API     NREL PSM v3 (eval only)    HF Model Hub    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Free-Tier Infrastructure Plan

| Service | What it does | Free limit | Cost |
|---------|-------------|-----------|------|
| **Hugging Face Spaces** | Host FastAPI inference app (Python) | 2 CPU cores, 16GB RAM | $0 |
| **Google AI Studio** | Gemma 4 (gemma-4-e4b-it) inference | 1,500 req/day, 1M tokens/day | $0 |
| **Cloudflare Workers** | Edge API routing + response caching | 100,000 req/day | $0 |
| **Cloudflare KV** | Cache forecast responses | 100,000 reads/day, 1,000 writes/day | $0 |
| **Cloudflare Pages** | Host React frontend | Unlimited sites | $0 |
| **Open-Meteo API** | Weather forecast data | Unlimited (non-commercial) | $0 |
| **NREL PSM v3** | Solar irradiance ground truth (eval) | Free with API key | $0 |
| **GitHub** | Source code, CI/CD via Actions | Free public repos | $0 |
| **Hugging Face Hub** | TimesFM model weights | Free public models | $0 |

**Total monthly cost: $0**

---

## 3. Repository Structure

```
terraseek/
├── backend/                    # Python — FastAPI + ML pipeline
│   ├── main.py                 # FastAPI app entry point
│   ├── pipeline/
│   │   ├── ingest.py           # Open-Meteo data fetching
│   │   ├── decompose.py        # Prophet seasonality decomposition
│   │   ├── forecast.py         # TimesFM inference wrapper
│   │   └── explain.py          # Gemma 4 explanation generation
│   ├── eval/
│   │   ├── nrel_loader.py      # NREL PSM v3 data loader
│   │   └── metrics.py          # MAE, RMSE calculation
│   ├── requirements.txt
│   └── Dockerfile              # For local dev
├── edge/                       # Cloudflare Workers (JavaScript)
│   ├── worker.js               # Main Worker script
│   └── wrangler.toml           # CF Workers config
├── frontend/                   # React + Vite
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── Map.jsx         # Leaflet map (click to query)
│   │   │   ├── ForecastChart.jsx  # Recharts line chart
│   │   │   └── ExplanationPanel.jsx
│   │   └── api/
│   │       └── terraseek.js    # API client
│   ├── package.json
│   └── vite.config.js
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_prophet_decomposition.ipynb
│   ├── 03_timesfm_evaluation.ipynb
│   └── 04_gemma_explanations.ipynb
├── README.md
├── CONTRIBUTING.md
└── .github/
    └── workflows/
        └── ci.yml              # GitHub Actions: lint + test
```

---

## 4. Backend — Python

### 4.1 Dependencies

```
# requirements.txt
fastapi==0.115.0
uvicorn==0.29.0
httpx==0.27.0          # async HTTP for Open-Meteo
timesfm==1.2.0         # Google TimesFM
prophet==1.1.5         # Meta Prophet
pandas==2.2.0
numpy==1.26.0
google-generativeai==0.8.0   # Gemma 4 via AI Studio
pydantic==2.6.0
```

### 4.2 Data Ingestion — `pipeline/ingest.py`

**Source:** Open-Meteo Forecast API (free, no key required)

```python
# Target variables from Open-Meteo
OPEN_METEO_PARAMS = {
    "hourly": [
        "shortwave_radiation",      # GHI proxy (W/m²) → solar forecast input
        "direct_radiation",         # DNI (W/m²)
        "windspeed_10m",            # Wind speed at 10m (m/s)
        "winddirection_10m",        # Wind direction (degrees)
        "cloudcover",               # Cloud cover % → key feature
        "temperature_2m",           # Air temp (°C)
    ],
    "forecast_days": 3,             # 72h of data, we use 48h
    "timezone": "auto"
}

# Endpoint
BASE_URL = "https://api.open-meteo.com/v1/forecast"

# Returns: pd.DataFrame with DatetimeIndex, columns = above params
```

**Rate limit:** None for non-commercial use. Add a 0.5s delay between requests when doing batch evaluation.

### 4.3 Seasonality Decomposition — `pipeline/decompose.py`

Prophet is used to strip daily and weekly seasonality patterns from the raw weather signal before TimesFM runs. This improves zero-shot forecast quality significantly.

```python
from prophet import Prophet

def decompose_series(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Input:  df with columns ['ds' (datetime), 'y' (target)]
    Output: df with added column 'y_deseasonalized'
    """
    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,   # Not enough history for yearly
        changepoint_prior_scale=0.05
    )
    m.fit(df[['ds', 'y']])
    forecast = m.predict(df[['ds']])
    df['y_deseasonalized'] = df['y'] - forecast['seasonal']
    return df
```

### 4.4 Forecasting — `pipeline/forecast.py`

**Model:** `google/timesfm-1.0-200m` from Hugging Face Hub  
TimesFM is a zero-shot foundation model — no fine-tuning required.

```python
import timesfm

def run_timesfm_forecast(
    context_series: list[float],   # Last 168 hours (7 days) of deseasonalized values
    horizon: int = 48              # Forecast 48 hours ahead
) -> list[float]:
    """
    Returns: list of 48 floats — forecasted deseasonalized values
    Post-process: add back seasonality from Prophet before returning to user
    """
    tfm = timesfm.TimesFm(
        context_len=168,
        horizon_len=horizon,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend="cpu",         # HF Spaces CPU tier — no GPU needed for inference
    )
    tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

    forecast_input = [context_series]
    freq_input = [0]           # 0 = high-frequency (hourly)

    point_forecast, _ = tfm.forecast(forecast_input, freq=freq_input)
    return point_forecast[0].tolist()
```

**Memory:** TimesFM 200M uses ~800MB RAM on CPU — fits within HF Spaces 16GB free tier. Load once at startup, reuse per request.

### 4.5 Explanation Generation — `pipeline/explain.py`

**Model:** Gemma 4 (`gemma-4-e4b-it`) via Google AI Studio  
**Free limit:** 1,500 requests/day, 1M tokens/day

```python
import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])  # Free key from aistudio.google.com

SYSTEM_PROMPT = """You are an expert energy analyst. Given a renewable energy forecast 
summary, write a 2-3 sentence plain-English explanation of why output is expected to 
change. Be specific about the weather drivers. Do not use jargon. Keep it under 60 words."""

def generate_explanation(
    location: str,
    solar_delta_pct: float,    # % change in solar vs 24h prior
    wind_delta_pct: float,     # % change in wind vs 24h prior
    key_weather_changes: dict  # e.g. {"cloudcover": "+40%", "windspeed": "+12%"}
) -> str:
    model = genai.GenerativeModel("gemma-4-e4b-it")
    
    prompt = f"""
Location: {location}
Solar output change: {solar_delta_pct:+.1f}% vs yesterday
Wind output change: {wind_delta_pct:+.1f}% vs yesterday
Weather changes: {key_weather_changes}

Write a 2-3 sentence explanation for grid operators.
"""
    response = model.generate_content(prompt)
    return response.text.strip()
```

### 4.6 FastAPI App — `main.py`

```python
from fastapi import FastAPI, Query
from pipeline.ingest import fetch_weather
from pipeline.decompose import decompose_series
from pipeline.forecast import run_timesfm_forecast
from pipeline.explain import generate_explanation

app = FastAPI(title="TerraSeek API", version="1.0")

@app.get("/forecast")
async def forecast(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    hours: int = Query(default=48, ge=1, le=48)
):
    """
    Returns 48-hour solar and wind forecast for any coordinate.
    Response time target: < 2s (cache miss), < 100ms (cache hit via CF KV)
    """
    # 1. Fetch weather context (last 7 days + 2 days forecast)
    weather_df = await fetch_weather(lat, lon)

    # 2. Decompose solar and wind separately
    solar_df = decompose_series(weather_df[['ds','shortwave_radiation']].rename(
        columns={'shortwave_radiation':'y'}), 'y')
    wind_df  = decompose_series(weather_df[['ds','windspeed_10m']].rename(
        columns={'windspeed_10m':'y'}), 'y')

    # 3. TimesFM forecast (context = last 168h, output = 48h)
    solar_forecast = run_timesfm_forecast(solar_df['y'].tail(168).tolist())
    wind_forecast  = run_timesfm_forecast(wind_df['y'].tail(168).tolist())

    # 4. Gemma 4 explanation
    explanation = generate_explanation(
        location=f"{lat:.2f}, {lon:.2f}",
        solar_delta_pct=compute_delta(solar_forecast),
        wind_delta_pct=compute_delta(wind_forecast),
        key_weather_changes=extract_key_changes(weather_df)
    )

    # 5. Build response
    timestamps = generate_hourly_timestamps(hours)
    return {
        "location": {"lat": lat, "lon": lon},
        "forecast_hours": hours,
        "solar_ghi_wm2": solar_forecast[:hours],
        "wind_speed_ms": wind_forecast[:hours],
        "timestamps": timestamps,
        "explanation": explanation,
        "model_versions": {
            "forecasting": "google/timesfm-1.0-200m",
            "explanation": "gemma-4-e4b-it"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}
```

---

## 5. Edge Layer — Cloudflare Workers

**File:** `edge/worker.js`

The Worker acts as a thin caching proxy in front of the HF Spaces backend. All requests go through the Worker first — if a cache hit exists in KV, the HF Spaces service is never called.

```javascript
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    
    // Only handle /forecast routes
    if (!url.pathname.startsWith('/forecast')) {
      return new Response('Not Found', { status: 404 });
    }

    // Build a cache key from lat (1dp), lon (1dp), hours
    const lat  = parseFloat(url.searchParams.get('lat')).toFixed(1);
    const lon  = parseFloat(url.searchParams.get('lon')).toFixed(1);
    const hrs  = url.searchParams.get('hours') || '48';
    const cacheKey = `forecast:${lat}:${lon}:${hrs}`;

    // Try KV cache first
    const cached = await env.TERRASEEK_KV.get(cacheKey);
    if (cached) {
      return new Response(cached, {
        headers: { 'Content-Type': 'application/json', 'X-Cache': 'HIT' }
      });
    }

    // Cache miss — call HF Spaces backend
    const backendUrl = `${env.BACKEND_URL}/forecast?lat=${lat}&lon=${lon}&hours=${hrs}`;
    const response = await fetch(backendUrl);
    const data = await response.text();

    // Store in KV with 1-hour TTL
    await env.TERRASEEK_KV.put(cacheKey, data, { expirationTtl: 3600 });

    return new Response(data, {
      headers: { 'Content-Type': 'application/json', 'X-Cache': 'MISS' }
    });
  }
};
```

**`wrangler.toml`:**
```toml
name = "terraseek-api"
main = "worker.js"
compatibility_date = "2024-09-01"

[[kv_namespaces]]
binding = "TERRASEEK_KV"
id = "YOUR_KV_NAMESPACE_ID"

[vars]
BACKEND_URL = "https://YOUR_HF_SPACE.hf.space"
```

---

## 6. Frontend — React + Vite

### 6.1 Dependencies

```json
{
  "dependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "recharts": "^2.12.0",
    "leaflet": "^1.9.4",
    "react-leaflet": "^4.2.1",
    "axios": "^1.7.0"
  },
  "devDependencies": {
    "vite": "^5.2.0",
    "@vitejs/plugin-react": "^4.2.0",
    "tailwindcss": "^3.4.0"
  }
}
```

### 6.2 Key Components

**`Map.jsx`** — Leaflet map. On click, extract lat/lon and call `terraseek.js` API client. Show loading spinner while forecast loads.

**`ForecastChart.jsx`** — Recharts `ComposedChart` with:
- Yellow `Area` for solar GHI (W/m²) — right Y-axis
- Blue `Line` for wind speed (m/s) — left Y-axis
- X-axis: hourly timestamps, formatted as "Mon 3pm"
- Responsive container, no legend clutter

**`ExplanationPanel.jsx`** — Simple card below the chart. Shows Gemma 4 explanation text. Includes a "Why?" badge and the model version label.

### 6.3 API Client — `api/terraseek.js`

```javascript
const BASE_URL = import.meta.env.VITE_API_URL; // Cloudflare Worker URL

export async function getForecast(lat, lon, hours = 48) {
  const res = await fetch(
    `${BASE_URL}/forecast?lat=${lat.toFixed(4)}&lon=${lon.toFixed(4)}&hours=${hours}`
  );
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}
```

### 6.4 Deployment

```bash
# Build
npm run build

# Deploy to Cloudflare Pages (free)
npx wrangler pages deploy dist --project-name terraseek
```

---

## 7. Evaluation — `eval/`

Forecast accuracy is validated against **NREL PSM v3** solar irradiance data (free with API key registration).

### 7.1 Test Set

- 10 US locations covering diverse climates (desert, coastal, midwest, northeast)
- Hold-out period: Q1 2024 (January–March, winter variability stress test)
- Metrics: MAE (W/m²), RMSE (W/m²), MAPE (%)

### 7.2 Baseline Comparisons

| Model | Description |
|-------|-------------|
| `naive_24h` | Yesterday's value shifted 24h forward |
| `prophet_only` | Prophet forecast without TimesFM |
| `timesfm_only` | TimesFM without Prophet decomposition |
| **`terraseek_v1`** | **Prophet decompose → TimesFM (this project)** |

The comparison table becomes a key result for the blog post and arXiv preprint.

---

## 8. Local Development Setup

### 8.1 Prerequisites

- Python 3.11+
- Node.js 20+
- Git
- Free accounts: Google AI Studio, Cloudflare, Hugging Face

### 8.2 Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set env var (get free key from aistudio.google.com)
export GEMINI_API_KEY=your_key_here

uvicorn main:app --reload --port 8000
# API now at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 8.3 Frontend

```bash
cd frontend
npm install

# Create .env.local
echo "VITE_API_URL=http://localhost:8000" > .env.local

npm run dev
# Dashboard at http://localhost:5173
```

### 8.4 Run Evaluation Notebooks

```bash
pip install jupyter
jupyter notebook notebooks/
# Start with 01_data_exploration.ipynb
```

---

## 9. CI/CD — GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r backend/requirements.txt
      - run: pip install pytest httpx
      - run: pytest backend/tests/ -v

  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: "20" }
      - run: cd frontend && npm ci
      - run: cd frontend && npm run build
```

---

## 10. Environment Variables

| Variable | Where set | Description |
|----------|-----------|-------------|
| `GEMINI_API_KEY` | HF Spaces secret / `.env.local` | Google AI Studio free API key |
| `BACKEND_URL` | Cloudflare Workers var | HF Spaces app URL |
| `VITE_API_URL` | Cloudflare Pages env / `.env.local` | Cloudflare Worker URL |
| `NREL_API_KEY` | Local only (eval) | Free from developer.nrel.gov |

---

## 11. API Response Schema

```json
{
  "location": {
    "lat": 37.77,
    "lon": -122.42
  },
  "forecast_hours": 48,
  "timestamps": [
    "2026-04-17T00:00:00Z",
    "2026-04-17T01:00:00Z"
  ],
  "solar_ghi_wm2": [0.0, 0.0, 12.4, 85.3, "..."],
  "wind_speed_ms": [4.2, 4.8, 5.1, 5.6, "..."],
  "explanation": "Solar output is expected to drop 18% tomorrow afternoon due to incoming marine layer cloud cover. Wind speeds increase moderately from the northwest, partially offsetting the reduced solar generation.",
  "model_versions": {
    "forecasting": "google/timesfm-1.0-200m",
    "explanation": "gemma-4-e4b-it"
  }
}
```

---

## 12. Known Constraints and Mitigations

| Constraint | Mitigation |
|-----------|------------|
| HF Spaces cold start (~30s) | Keep-alive ping from Cloudflare Worker every 5 min via a scheduled cron trigger |
| Gemma 4 rate limit (1,500 req/day) | Cache explanation per (lat, lon, date) in KV — only regenerate when forecast changes significantly (> 5%) |
| TimesFM requires 168h context | If user queries a region with < 7 days Open-Meteo history, fall back to climatological mean as padding |
| Open-Meteo resolution (0.1° grid) | Snap user lat/lon to nearest 0.1° grid point before querying |
| No GPU on HF Spaces free tier | TimesFM 200M runs on CPU in ~800ms — acceptable for our latency target |

---

*End of TRD*