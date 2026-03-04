# HiDrift Testing Documentation

## 1. Purpose
This document defines how to test HiDrift reliably across:
1. Unit tests
2. Integration tests
3. Regression tests
4. API smoke tests
5. Evaluation and visualization validation

## 2. Test Environment Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,api]"
```

Optional Gemini env for API runtime:

```env
GEMINI_API_KEY=your_real_key
HIDRIFT_LLM_PROVIDER=gemini
HIDRIFT_LLM_MODEL=gemini-2.5-flash
```

Notes:
1. Most tests are designed to run without live Gemini calls.
2. Eval baselines use fallback model to avoid quota-related failures.

## 3. Full Test Suite

### 3.1 Fast pass/fail
```powershell
pytest -q
```

### 3.2 Detailed test output
```powershell
pytest -v --durations=10
```

## 4. Suite Breakdown

### 4.1 Unit tests (`tests/unit`)
Coverage includes:
1. Drift score calculations
2. Memory router monotonic behavior
3. Consolidation decay and semantic dedup
4. Semantic graph upsert, conflict resolution, and hybrid retrieval shape

Run only unit tests:

```powershell
pytest tests/unit -q
```

### 4.2 Integration tests (`tests/integration`)
Coverage includes:
1. End-to-end turn ingest/retrieve flow
2. Drift-triggered consolidation execution
3. Hybrid semantic fact creation and supersession behavior

Run only integration tests:

```powershell
pytest tests/integration -q
```

### 4.3 Regression tests (`tests/regression`)
Coverage includes:
1. Fixed-seed eval output shape checks
2. No-drift guard behavior
3. Hybrid proxy threshold checks

Run only regression tests:

```powershell
pytest tests/regression -q
```

## 5. API Smoke Testing

Start server:

```powershell
uvicorn hidrift.api:create_app --factory --reload
```

Open:
1. `http://127.0.0.1:8000/docs`

Smoke sequence:
1. `POST /v1/memory/ingest` with minimal payload
2. `POST /v1/memory/retrieve` and verify:
   - `hard_constraints`
   - `supporting_context`
3. `GET /v1/drift/current`
4. `POST /v1/consolidation/run`
5. `GET /v1/semantic/facts`
6. `GET /v1/semantic/graph/subgraph`
7. `GET /v1/semantic/conflicts`

## 6. Evaluation Validation

Run evaluation:

```powershell
python scripts/run_eval.py
```

Expected:
1. New `artifacts/eval_*.json` file created
2. JSON contains `systems` and `aggregate_mean` entries

Run calibration:

```powershell
python scripts/train_calibrator.py
```

Expected:
1. `artifacts/calibration.json` created

## 7. Visualization Validation

```powershell
python scripts/export_figures.py
```

Expected outputs:
1. `paper/tables/aggregate_metrics.md`
2. `paper/tables/aggregate_metrics.json`
3. `paper/tables/hybrid_semantic_metrics.md`
4. `paper/figures/aggregate_higher_is_better.png`
5. `paper/figures/aggregate_lower_is_better.png`
6. `paper/figures/hybrid_constraint_hit_rate.png`
7. `paper/figures/conflict_resolution_accuracy.png`
8. `paper/figures/drift_trigger_timeline.png`
9. `paper/figures/consolidation_event_count.png`

## 8. Suggested CI Command Order

```powershell
pytest -q
python scripts/run_eval.py
python scripts/export_figures.py
```

## 9. Failure Diagnosis Guide

### 9.1 `ModuleNotFoundError`
Fix:
```powershell
pip install -e ".[dev,api]"
```

### 9.2 Gemini key errors
Symptom:
1. `GEMINI_API_KEY is not set`

Fix:
1. Set env variable or `.env`
2. Run `python scripts/check_gemini.py`

### 9.3 Gemini quota errors
Symptom:
1. `429 RESOURCE_EXHAUSTED`

Fix:
1. Wait/reset quota or use paid billing project
2. Continue local tests/eval (fallback mode)

### 9.4 Missing chart outputs
Cause:
1. `matplotlib` not installed

Fix:
```powershell
pip install matplotlib
python scripts/export_figures.py
```

## 10. Acceptance Checklist
1. `pytest -q` passes
2. Eval artifact generated successfully
3. Visualization files generated successfully
4. API endpoints accessible from Swagger
5. Semantic graph endpoints return valid JSON

