# Semiconductor Process Control (SPC) & Chamber Drift Analytics Platform

```markdown


[![Next.js](https://img.shields.io/badge/Next.js-15.5-black?logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.4-EB5B00)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.2-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![Google Cloud Run](https://img.shields.io/badge/Google_Cloud_Run-Deployed-4285F4?logo=google-cloud)](https://semiconductor-drift-analytics-6511273324.asia-south1.run.app/docs)
[![Gemini 2.5 Flash](https://img.shields.io/badge/AI-Gemini_2.5_Flash-8E75B2?logo=google-gemini)](https://deepmind.google/technologies/gemini/)

An end-to-end Fault Detection and Classification (FDC) and Statistical Process Control (SPC) analytics platform designed for semiconductor wafer fabrication. The system combines a **semi-supervised Teacher-Student XGBoost pipeline** with real-time **Statistical Process Control telemetry** and **LLM-driven root-cause diagnostics** to detect chamber drift, identify inline-to-probe sensor correlations, and simulate real-time wafer excursion risk.

---

## Live Deployments

* **Interactive Next.js Frontend Dashboard:** `<YOUR_VERCEL_URL_HERE>`
* **FastAPI Production Backend (Google Cloud Run):** `https://semiconductor-drift-analytics-6511273324.asia-south1.run.app`
* **Interactive OpenAPI / Swagger Documentation:** `https://semiconductor-drift-analytics-6511273324.asia-south1.run.app/docs`

---

## System Architecture

The platform is engineered as a decoupled, cloud-native microservices architecture:

```text
+-------------------------------------------------------------------------+
|                   DATA & MACHINE LEARNING PIPELINE                      |
|                                                                         |
|  [Raw Wafer Sensor Telemetry]                                           |
|        │                                                                |
|        ▼                                                                |
|  (pipeline_step1_clean.py) ──> Imputation, Zero-Variance Pruning,       |
|        │                       Golden Baseline Metadata Extraction      |
|        ├──> clean_labeled_wafers.csv                                    |
|        └──> clean_unlabeled_stream.csv                                  |
|                    │                                                    |
|                    ▼                                                    |
|  (pipeline_step2_model_and_drift.py & pipeline_step2b_semi_supervised)  |
|        ├──> Teacher XGBoost Model (Trained on Golden Labeled Wafers)    |
|        ├──> Pseudo-Labeling High-Confidence Unlabeled Production Stream |
|        └──> Student XGBoost Model (Distilled Production Fault Detector) |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|                BACKEND SERVICE (Google Cloud Run • Docker)              |
|                                                                         |
|  FastAPI + Uvicorn (app_backend.py)                                     |
|   ├── In-Memory Model & Telemetry Engine (XGBoost + Pandas/NumPy)       |
|   ├── Real-Time SPC & Z-Score Drift Evaluator                           |
|   ├── What-If Perturbation & Excursion Simulator                        |
|   └── AI Root-Cause Diagnostic Engine (Google Gemini 2.5 Flash SDK)     |
+-------------------------------------------------------------------------+
                                     │  REST API (JSON over HTTPS)
                                     ▼
+-------------------------------------------------------------------------+
|                     FRONTEND CLIENT (Vercel Edge)                       |
|                                                                         |
|  Next.js 15 (App Router + Turbopack + TypeScript)                       |
|   ├── Multi-Sensor Statistical Process Control (SPC) Charting           |
|   ├── Inline-to-Probe Yield Correlation Matrix                          |
|   ├── Interactive "What-If" Chamber Drift Simulator                     |
|   └── Automated Process Engineering Diagnostic Reports                  |
+-------------------------------------------------------------------------+

```

---

## Core Engineering Capabilities

### 1. Semi-Supervised Teacher-Student XGBoost Architecture

In high-volume semiconductor manufacturing, inline metrology data is abundant, but physical electrical probe test labels are sparse and expensive to acquire.

* **Data Sanitization (`pipeline_step1_clean.py`):** Filters out constant/dead chamber sensors, handles high-dimensional missing sensor traces, and establishes golden baseline statistical boundaries ($\mu$, $\sigma$, UCL/LCL) across normal production runs (`golden_baseline_metadata.json`).
* **Teacher Model (`xgb_teacher_model.json`):** Trained on verified, labeled wafer runs (`clean_labeled_wafers.csv`) to capture initial non-linear fault boundaries across chamber sensors.
* **Self-Training & Student Model (`xgb_student_model.json`):** Evaluates unlabelled production wafer streams (`clean_unlabeled_stream.csv`), generates high-confidence pseudo-labels, and trains a generalized Student XGBoost classifier capable of scoring continuous production drift (`scored_production_drift_stream.csv`).

### 2. Statistical Process Control (SPC) & Inline-Probe Correlation

* **Dynamic Control Limits:** Computes and visualizes real-time $\pm 3\sigma$ Upper and Lower Control Limits (UCL/LCL) against golden baseline chamber parameters.
* **Batch & Wafer Filtering:** Supports granular inspection across individual production batch files and historical wafer windows.
* **Sensor Importance & Probe Correlation:** Ranks chamber sensors by their contribution to yield excursions, allowing process engineers to isolate physical hardware degradation (e.g., RF power drift, gas flow controller instability, pressure variance).

### 3. Real-Time "What-If" Chamber Drift Simulator

* Interactive perturbation engine allowing engineers to adjust top-ranked sensor values in real time.
* Executes sub-millisecond inference against the compiled XGBoost model binary in memory to quantify how specific chamber parameter shifts impact overall wafer fault probability.

### 4. GenAI Process Engineering Diagnostics

* Integrates `google-genai` (**Gemini 2.5 Flash**) to synthesize quantitative Z-score deviations, sensor drift vectors, and XGBoost fault probabilities into structured, actionable root-cause analysis and maintenance recommendations for process engineers.
* Includes deterministic fallback diagnostics when operated in air-gapped or offline environments without an external API key.

---

## Repository Structure

```text
semiconductor-drift-analytics/
├── frontend/                              # Next.js 15 TypeScript application
│   ├── app/
│   │   └── page.tsx                       # Main SPC, Correlation & Simulator Dashboard
│   ├── next.config.mjs                    # Next.js production build configuration
│   └── package.json                       # Frontend dependencies
├── app_backend.py                         # FastAPI production server & inference engine
├── pipeline_step1_clean.py                # Telemetry preprocessing & baseline extraction
├── pipeline_step2_model_and_drift.py      # Initial XGBoost training & drift scoring
├── pipeline_step2b_semi_supervised.py     # Teacher-Student pseudo-labeling pipeline
├── xgb_teacher_model.json                 # Serialized Teacher XGBoost model binary
├── xgb_student_model.json                 # Serialized Student XGBoost model binary
├── golden_baseline_metadata.json          # Reference sensor means, std devs, and thresholds
├── wafer_sensor_fault_detection.csv       # Raw wafer sensor telemetry dataset
├── clean_labeled_wafers.csv               # Sanitized labeled wafer dataset
├── clean_unlabeled_stream.csv             # Sanitized unlabeled wafer production stream
├── scored_production_drift_stream.csv     # Production stream scored with drift probabilities
├── requirements.txt                       # Locked Python production dependencies
└── Dockerfile                             # Container specification for Google Cloud Run

```

---

## API Reference

All backend endpoints are prefixed with `/api` and return structured JSON payloads:

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/overview` | Returns high-level fab metrics, total production wafers, excursion counts, and model performance KPIs. |
| `GET` | `/api/inline-probe-correlation` | Returns ranked sensor feature importances and inline-to-probe fault correlation metrics. |
| `GET` | `/api/spc-chart/{sensor_name}` | Returns time-series wafer readings, golden baseline mean, and $\pm 3\sigma$ control limits (`limit` and `batch_file` query params supported). |
| `POST` | `/api/simulate-drift` | Accepts custom sensor overrides and executes real-time XGBoost inference to compute updated fault probabilities. |
| `POST` | `/api/ai-diagnostics` | Invokes Gemini 2.5 Flash with current wafer Z-scores and drift telemetry to generate root-cause engineering insights. |

---

## Local Development Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/pp9555/semiconductor-drift-analytics.git](https://github.com/pp9555/semiconductor-drift-analytics.git)
cd semiconductor-drift-analytics

```

### 2. Run the FastAPI Backend

Ensure you have Python 3.11+ installed.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Optional: Set your Gemini API key for live AI diagnostics
export GEMINI_API_KEY="your_api_key_here"

# Start the Uvicorn server on port 8000
uvicorn app_backend:app --host 0.0.0.0 --port 8000 --reload

```

### 3. Run the Next.js Frontend

Open a second terminal window:

```bash
cd frontend
npm install

# Configure the local backend endpoint
echo "NEXT_PUBLIC_API_URL=http://localhost:8000/api" > .env.local

npm run dev

```

Open `http://localhost:3000` in your browser to interact with the platform.

---

## Cloud Production Deployment

### Backend: Google Cloud Run (Docker)

The backend is containerized using `python:3.11-slim` and deployed on Google Cloud Run (`asia-south1`) with **2 GiB RAM** and **2 vCPUs** to support low-latency in-memory Pandas operations and XGBoost inference:

```bash
docker build -t semiconductor-drift-backend .
docker run -p 8080:8080 -e GEMINI_API_KEY="your_api_key_here" semiconductor-drift-backend

```

### Frontend: Vercel

The `frontend/` directory is deployed on Vercel using the Next.js 15 build pipeline with the following production environment configuration:

* `NEXT_PUBLIC_API_URL`: `https://semiconductor-drift-analytics-6511273324.asia-south1.run.app/api`

---

## Author

**Pranshu Pal**

*B.Tech in Metallurgical and Materials Engineering, National Institute of Technology (NIT) Srinagar*

* GitHub: [@pp9555](https://www.google.com/search?q=https://github.com/pp9555)
