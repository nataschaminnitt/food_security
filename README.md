# 🌾 Ethiopia Food Prices – Exploration & Forecasting

This repository contains an end-to-end pipeline and dashboard for exploring and forecasting staple food prices across regions in Ethiopia.

The project:

- Cleans and harmonises monthly price data from multiple sources.
- Builds an analysis-ready “Tier A” panel of staple food prices.
- Benchmarks several time-series models (Naive, ARIMA/SARIMA, XGBoost, hybrids).
- Serves an interactive Streamlit dashboard for exploration & 3-month forecasts.

---

## 1. Project Overview

**Goal:**  
Provide a lightweight, transparent forecasting tool for staple food prices in Ethiopia, focused on a short-term **3-month planning horizon** for humanitarian / food security use cases.

**Key design choices:**

- Unit of analysis: `(admin_1, product)` (region–product pairs).
- Temporal unit: monthly data.
- Target: `value_imputed` (retail prices per standard unit, cleaned & imputed).
- Operational forecast model: **Naive (last observed value) with horizon = 3 months**.

A full exploration + modelling notebook exists (`model_comparison_and_export.py` logic), but the dashboard only uses the final chosen model + comparison table.

---
