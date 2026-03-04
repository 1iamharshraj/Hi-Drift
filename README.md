# HiDrift

HiDrift is a research-oriented implementation of a drift-aware hierarchical memory system for long-horizon LLM agents.  
It combines:
1. Drift detection
2. Working + episodic + semantic memory hierarchy
3. Hybrid semantic memory retrieval (graph + vector)
4. Consolidation pipelines
5. Evaluation and visualization workflows

## Documentation Map
1. Main reference guide: `DOCUMENTATION.md`
2. Architecture and data-flow diagrams: `ARCHITECTURE.md`
3. Testing guide: `TESTING.md`
4. API contract source: `src/hidrift/api.py`
5. Config defaults: `configs/`
6. Scripts for eval/plots/calibration: `scripts/`
7. Publication gate checks: `scripts/check_publication_readiness.py`, `scripts/check_benchmark_registry.py`
8. ICCV-grade plan: `ICCV_PLAN.md`

## Table Of Contents
1. Project Goals
2. Repository Layout
3. Environment Setup
4. Gemini Configuration
5. Running The API
6. API Endpoint Guide
7. Testing Guide
8. Evaluation Guide
9. Visualization Guide
10. Known Operational Notes
11. Troubleshooting
12. Development Workflow

## 1) Project Goals
HiDrift addresses memory degradation over long agent horizons by:
1. Detecting context/behavior/performance drift
2. Consolidating episodic experiences into structured semantic facts
3. Maintaining semantic facts in a graph-augmented memory layer
4. Retrieving both hard constraints and supporting context
5. Benchmarking performance over controlled drift scenarios

## 2) Repository Layout
High-value paths:
1. `src/hidrift/agent/` runtime orchestration
2. `src/hidrift/drift/` drift feature and scoring logic
3. `src/hidrift/memory/` working/episodic/semantic memory
4. `src/hidrift/semantic_graph/` graph adapter/store/query/reasoning
5. `src/hidrift/consolidation/` clustering, summarization, consolidation worker
6. `src/hidrift/eval/` simulator, baselines, metrics, experiment runner
7. `src/hidrift/api.py` FastAPI app and all routes
8. `configs/` default model/memory/drift/eval settings
9. `tests/` unit/integration/regression coverage
10. `paper/` generated tables and figures for reporting

## 3) Environment Setup

### Windows PowerShell (recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,api]"
```

### Verify install
```powershell
python -c "import hidrift; print('hidrift import ok')"
```

## 4) Gemini Configuration
Default runtime provider is Gemini (`gemini-2.5-flash`) for API usage.

### Option A: temporary shell env
```powershell
$env:GEMINI_API_KEY="your_real_key"
$env:HIDRIFT_LLM_PROVIDER="gemini"
$env:HIDRIFT_LLM_MODEL="gemini-2.5-flash"
```

### Option B: `.env` file in repo root
```env
GEMINI_API_KEY=your_real_key
HIDRIFT_LLM_PROVIDER=gemini
HIDRIFT_LLM_MODEL=gemini-2.5-flash
```

The code auto-loads `.env` through `src/hidrift/env.py` + `src/hidrift/llm/factory.py`.

### Validate key + model connectivity
```powershell
python scripts/check_gemini.py
```

If you hit quota/rate limits, tests/eval still run because eval baselines are configured to use fallback LLM mode.

## 5) Running The API
```powershell
uvicorn hidrift.api:create_app --factory --reload
```

Local URLs:
1. Swagger UI: `http://127.0.0.1:8000/docs`
2. OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

## 6) API Endpoint Guide

### Memory + Drift Core
1. `POST /v1/memory/ingest`
2. `POST /v1/memory/retrieve`
3. `GET /v1/drift/current`
4. `POST /v1/consolidation/run`
5. `GET /v1/eval/run/{run_id}`

### Semantic Graph Endpoints
1. `GET /v1/semantic/facts?query=...&k=...`
2. `GET /v1/semantic/graph/subgraph?entity_id=...&hops=...`
3. `POST /v1/semantic/facts/upsert`
4. `GET /v1/semantic/conflicts?entity_id=...`

### Ingest example (agent output auto-generated if omitted)
```json
{
  "session_id": "s-001",
  "user_id": "u-001",
  "user_input": "Plan my day in concise format",
  "task_label": "calendar"
}
```

### Retrieve example
```json
{
  "query": "what are user style preferences",
  "k_working": 5,
  "k_episodic": 5,
  "k_semantic": 5
}
```

Expected retrieve output includes:
1. `working`
2. `episodic`
3. `semantic`
4. `hard_constraints`
5. `supporting_context`

## 7) Testing Guide

### Fast feedback
```powershell
pytest -q
```

### Detailed + timing view
```powershell
pytest -v --durations=10
```

### Test suite layout
1. `tests/unit/` logic-level behavior
2. `tests/integration/` runtime/API/memory interaction behavior
3. `tests/regression/` fixed-seed eval behavior checks

## 8) Evaluation Guide

### Run single eval
```powershell
python scripts/run_eval.py
```

### Run publishable matrix (main comparison + ablations)
```powershell
python scripts/run_eval_matrix.py --config configs/eval/matrix_publishable.json
```

### Run publication gates
```powershell
python scripts/check_benchmark_registry.py
python scripts/check_publication_readiness.py
```

### One-command publishability pipeline
```powershell
make paper_ready
```

### ICCV-grade pipeline
```powershell
python scripts/prepare_official_benchmarks.py
make eval_iccv
python scripts/export_figures.py
make iccv_check
```

Output:
1. `artifacts/eval_<uuid>.json`
2. `artifacts/eval_matrix_<uuid>.json` (for matrix runs)
2. Includes:
   - baseline + ablation systems (`RAG-only`, `MemGPT-style`, `GenerativeAgents-style`, `FlatMem-TopK`, `VectorOnly-noGraph`, `GraphOnly-noVector`, `HiDrift-noConflict`, `HiDrift-noDriftSignal`, `HiDrift-full`)
   - multi-scenario reports (`personal_assistant_drift`, `tool_api_drift`, `contradiction_drift`, `semi_real_trace`, `locomo_like_trace`, `longmem_like_trace`)
   - per-turn trace logs
   - significance summary vs `HiDrift-full` with Holm-Bonferroni correction
3. Publication checks generate:
   - `paper/tables/benchmark_registry_check.md`
   - `paper/tables/publication_readiness.md`

### Drift threshold calibration artifact
```powershell
python scripts/train_calibrator.py
```

Output:
1. `artifacts/calibration.json`

## 9) Visualization Guide
Generate tables + PNG charts from latest eval artifact:

```powershell
python scripts/export_figures.py
```

Generated artifacts:
1. `paper/tables/aggregate_metrics.md`
2. `paper/tables/aggregate_metrics.json`
3. `paper/tables/significance_report.md`
4. `paper/tables/scenario_metrics.md`
5. `paper/figures/task_success_with_errorbars.png`
6. `paper/figures/adaptation_latency_distribution.png`
7. `paper/figures/scenario_success_heatmap.png`
8. `paper/figures/drift_score_trace.png`
9. `paper/figures/memory_growth_trace.png`
10. `paper/figures/constraint_violation_trend.png`

## 10) Known Operational Notes
1. API runtime defaults to Gemini and requires `GEMINI_API_KEY`
2. Evaluation baselines intentionally use fallback model to keep tests/eval reproducible under Gemini quota limits
3. Semantic graph persistence is file-based (`artifacts/semantic_graph.json`)
4. Current graph backend is `networkx` (single-process baseline)

## 11) Troubleshooting

### `GEMINI_API_KEY is not set`
1. Ensure `.env` exists in repository root
2. Ensure key is exactly `GEMINI_API_KEY=...` (no spaces around `=`)
3. Re-run:
```powershell
python scripts/check_gemini.py
```

### 429 quota/rate-limit from Gemini
1. Wait for quota reset or use billed project
2. Keep running tests/eval in fallback mode

### `uv` not found
Use standard venv + pip flow shown above.

### Import errors after dependency changes
```powershell
pip install -e ".[dev,api]"
```

## 12) Development Workflow
Suggested loop:
1. Edit feature/config/tests
2. Run `pytest -q`
3. Run `python scripts/run_eval_matrix.py --config configs/eval/matrix_publishable.json`
4. Run `python scripts/export_figures.py`
5. Review `paper/tables/significance_report.md` for p-values/effect sizes
6. Review `paper/tables/hypothesis_results.md` and `paper/tables/cost_latency_table.md`
7. Run `python scripts/check_benchmark_registry.py`
8. Run `python scripts/check_publication_readiness.py`
9. Review `paper/figures/scenario_success_heatmap.png` and trace plots for discriminative behavior
10. Update docs when behavior or contracts change
