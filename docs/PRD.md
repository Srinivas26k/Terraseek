# TerraSeek — Product Requirements Document (PRD)

**Version:** 1.0  
**Author:** Srinivas Nampalli
**Date:** April 2026  
**Status:** Draft

---

## 1. Overview

### 1.1 Problem Statement

Electrical grid operators, researchers, climate NGOs, and energy startups need accurate, explainable forecasts of renewable energy output (solar, wind) to plan grid dispatch, reduce carbon emissions, and optimize energy trading. Existing tools are either paywalled, proprietary, or lack natural-language explanations.

### 1.2 What is TerraSeek?

TerraSeek is a **free, open-source AI platform** that:

- Forecasts solar and wind power output for any region, 48 hours ahead
- Explains forecast changes in plain English
- Exposes a public REST API and a live React dashboard
- Runs entirely on **free-tier infrastructure** — no cloud bill, no API keys required for basic use

### 1.3 Why Now?

- Google released **TimesFM** (2024) — the first zero-shot time-series foundation model that requires no fine-tuning on new domains
- Google released **Gemma 4** (2025) — a multimodal open-weights model (4B parameters) optimized for reasoning at low cost, with a free inference tier via Google AI Studio
- **Cloudflare Workers AI** offers free-tier edge inference globally
- Climate tech is the #1 funded VC category in 2025–2026
- No public project has combined these three tools on real energy grid data

---

## 2. Goals

### 2.1 Product Goals

| Goal | Success Metric |
|------|----------------|
| Accurate renewable forecasts | MAE < 10% vs NREL ground truth on 24h horizon |
| Plain-English explanations | User rating ≥ 4/5 in usability test |
| Public API live | API returns forecast in < 2s p95 latency |
| Open source traction | 200+ GitHub stars within 60 days of launch |
| Zero infra cost | $0/month on free tiers throughout development |

### 2.2 Non-Goals (v1)

- Real-time (sub-minute) grid monitoring
- Battery storage optimization
- Electricity price forecasting
- Native mobile app

---

## 3. Users

### 3.1 Primary Users

**Independent researchers and students**
- Need: Access to working AI energy forecasting without institutional subscriptions
- Use: Pull API data for papers, coursework, side projects

**Climate NGOs and energy journalists**
- Need: Understand why renewable output changed on a given day
- Use: Natural-language explanations from the dashboard

### 3.2 Secondary Users

**Energy startups (early-stage)**
- Need: Prototype forecasting before building custom models
- Use: REST API as a plug-in data source

**University admissions reviewers / open source contributors**
- Need: Evidence of real engineering and research quality
- Use: Read the GitHub repo, documentation, and arXiv preprint

---

## 4. Features

### 4.1 Core Features (v1 — must ship)

#### F1 — Renewable Energy Forecast
- Input: latitude/longitude + date range (up to 48h ahead)
- Output: Hourly solar (GHI, W/m²) and wind (m/s at 10m) forecasts
- Model pipeline: Open-Meteo weather data → Meta Prophet seasonality decomposition → Google TimesFM zero-shot forecast
- Evaluation: MAE and RMSE vs NREL PSM v3 ground truth

#### F2 — Natural-Language Explanation
- Input: Forecast delta (why did output drop at 3pm tomorrow?)
- Output: 2–3 sentence plain-English explanation
- Model: Gemma 4 (4B, via Google AI Studio free tier)
- Examples: "Solar output drops at 3 PM tomorrow due to a forecasted cloud band moving in from the southwest. Wind picks up by 18% to partially compensate."

#### F3 — Public REST API
- Endpoint: `GET /forecast?lat=&lon=&hours=48`
- Response: JSON with hourly solar + wind values + explanation string
- Hosted: Cloudflare Workers (free tier, 100k req/day)
- Caching: Cloudflare KV — cache per (lat, lon, date) for 1 hour

#### F4 — React Dashboard
- Live map: Click any region → see 48h forecast chart
- Explanation panel: Shows Gemma 4 text explanation for the selected region
- Charts: Recharts line chart — solar (yellow) + wind (blue) + actual vs forecast overlay
- Mobile-responsive
- Stack: React + Vite, deployed on Cloudflare Pages (free)

#### F5 — GitHub Repository + Documentation
- README with architecture diagram, setup guide, and demo GIF
- `CONTRIBUTING.md` for open source contributors
- Jupyter notebooks showing data pipeline and model evaluation
- MIT License

### 4.2 Stretch Features (v2 — post-launch)

- Carbon intensity overlay (gCO₂/kWh) using Electricity Maps free API
- Forecast accuracy leaderboard comparing TimesFM vs ARIMA vs LSTM
- Email/webhook alerts when forecast changes by > 20%
- Streamlit notebook companion for researchers

---

## 5. User Stories

| ID | As a... | I want to... | So that... |
|----|---------|--------------|------------|
| US1 | Researcher | Query forecast data for any GPS coordinate | I can use it in my energy paper |
| US2 | NGO analyst | Read a plain-English explanation of a forecast drop | I can explain it to non-technical stakeholders |
| US3 | Developer | Call a REST API and get JSON back in < 2 seconds | I can prototype a product on top of TerraSeek |
| US4 | Student | Run the project locally with zero paid services | I can learn from it without spending money |
| US5 | Open source contributor | Understand the codebase in under 30 minutes | I can submit a PR |

---

## 6. Out of Scope (v1)

- User accounts or authentication
- Paid tiers or monetization
- Real-time streaming data (WebSocket)
- Non-renewable energy sources (nuclear, hydro, gas)
- Historical data beyond 2 years

---

## 7. Timeline

| Week | Milestone |
|------|-----------|
| 1–2 | Data pipeline: ingest Open-Meteo + NREL, store in flat JSON/Parquet |
| 3–4 | Forecasting: Prophet decomposition + TimesFM integration, eval notebooks |
| 5 | Gemma 4 explanation layer via Google AI Studio API |
| 6 | Cloudflare Workers API + KV caching |
| 7 | React dashboard (Vite + Recharts + Leaflet map) |
| 8 | Polish, README, arXiv preprint draft, Product Hunt launch |

---

## 8. Success Criteria

TerraSeek v1 is considered successful when:

1. The API returns 48h forecasts for any coordinate in < 2 seconds
2. Gemma 4 explanations are judged coherent and useful by 3 independent testers
3. Forecast MAE is below 10% on a held-out NREL test set
4. The GitHub repository is public with complete setup instructions
5. At least one blog post (dev.to or Hashnode) is published linking to the repo

---

*End of PRD*