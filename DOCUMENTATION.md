# HiDrift Full Documentation

## 1. Overview

HiDrift is a drift-aware hierarchical memory framework for long-horizon agents.
Detailed diagrams are available in `ARCHITECTURE.md`.

Core goals:
- Detect drift in user/task/performance distributions over time
- Store interactions across multi-level memory (working, episodic, semantic)
- Consolidate episodic memory into semantic memory when drift or schedule triggers
- Evaluate long-horizon behavior against baselines with reproducible artifacts

## 2. Current Implementation Status

Implemented:
- Core memory hierarchy and retrieval
- Drift scoring with hysteresis/cooldown triggering
- Consolidation worker (cluster, summarize, dedup, decay)
- Gemini integration (`gemini-2.5-flash`) with API-key based generation
- FastAPI endpoints for ingest/retrieve/drift/consolidation
- Synthetic evaluation harness and baseline runner
- Tests and visualization export pipeline

Not fully implemented yet (research roadmap items):
- Real vector DB backend integration (currently in-memory structures)
- Full MLflow/Hydra orchestration
- Advanced statistical significance pipeline beyond mean aggregation

## 3. Repository Structure

Top-level:
- `src/hidrift/` core package
- `configs/` model/memory/drift/eval configs
- `scripts/` utility and pipeline scripts
- `tests/` unit/integration/regression tests
- `paper/` figures/tables and paper outline
- `artifacts/` generated run artifacts

Important modules:
- `src/hidrift/agent/runtime.py`
- `src/hidrift/memory/`
- `src/hidrift/drift/`
- `src/hidrift/consolidation/`
- `src/hidrift/eval/`
- `src/hidrift/api.py`
- `src/hidrift/llm/`

## 4. Environment Setup

### 4.1 Python venv setup (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,api]"
```

### 4.2 Configure Gemini

Option A: environment variables in terminal

```powershell
$env:GEMINI_API_KEY="your_real_key"
$env:HIDRIFT_LLM_PROVIDER="gemini"
$env:HIDRIFT_LLM_MODEL="gemini-2.5-flash"
```

Option B: `.env` in repository root

```env
GEMINI_API_KEY=your_real_key
HIDRIFT_LLM_PROVIDER=gemini
HIDRIFT_LLM_MODEL=gemini-2.5-flash
```

Notes:
- `.env` is auto-loaded by `src/hidrift/llm/factory.py`
- Model typo form `gemini-2,5-flash` is normalized to `gemini-2.5-flash`

### 4.3 Validate Gemini connectivity

```powershell
python scripts/check_gemini.py
```

Expected output: model response containing `GEMINI_OK`.

## 5. Architecture

Data flow:
1. Interaction enters runtime (`AgentRuntime.handle_turn`)
2. Drift service computes `D_t` signal
3. Event is routed into working + episodic memory
4. Retrieval fuses working/episodic/semantic context
5. Consolidation is triggered when drift threshold policy is met
6. Consolidation writes semantic memory items and decays episodic importance

### 5.1 Memory layers

Working memory:
- Ring buffer of recent interactions

Episodic memory:
- Per-turn records with goal/actions/outcomes/reward/embedding/importance

Semantic memory:
- Consolidated memory statements with confidence/stability/evidence linkage

### 5.2 Drift model

Score components:
- Behavioral shift (embedding deviation)
- Task shift (distribution divergence)
- Performance drop (reward trend degradation)

Weighted score:
- `alpha=0.45`, `beta=0.30`, `gamma=0.25`

Trigger policy:
- Threshold + hysteresis (`m=3`) + cooldown

### 5.3 Consolidation model

Pipeline:
1. Gather episodes
2. Apply decay
3. Cluster episodes by goal (current deterministic baseline)
4. Summarize each cluster (Gemini-backed with fallback)
5. Deduplicate semantic memory
6. Store semantic items with evidence IDs

## 6. API Usage

Run API:

```powershell
uvicorn hidrift.api:create_app --factory --reload
```

Open Swagger UI:
- `http://127.0.0.1:8000/docs`

Endpoints:
- `POST /v1/memory/ingest`
- `POST /v1/memory/retrieve`
- `GET /v1/drift/current`
- `POST /v1/consolidation/run`
- `GET /v1/eval/run/{run_id}`
- `GET /v1/semantic/facts?query=...`
- `GET /v1/semantic/graph/subgraph?entity_id=...&hops=2`
- `POST /v1/semantic/facts/upsert`
- `GET /v1/semantic/conflicts?entity_id=...`

### 6.1 Example ingest request

You can omit `agent_output` to let Gemini generate it.

```json
{
  "session_id": "s1",
  "user_id": "u1",
  "user_input": "Plan my day in concise format",
  "task_label": "calendar"
}
```

### 6.2 Example retrieve request

```json
{
  "query": "user style preference",
  "k_working": 5,
  "k_episodic": 5,
  "k_semantic": 5
}
```

## 7. Testing

Quick:

```powershell
pytest -q
```

Detailed with timing:

```powershell
pytest -v --durations=10
```

Test layout:
- `tests/unit/` isolated logic tests
- `tests/integration/` runtime flow tests
- `tests/regression/` eval stability tests

## 8. Evaluation Pipeline

Run full evaluation:

```powershell
python scripts/run_eval.py
```

Generates:
- `artifacts/eval_<run_id>.json`

Calibrate drift threshold artifact:

```powershell
python scripts/train_calibrator.py
```

Generates:
- `artifacts/calibration.json`

## 9. Visualization and Reporting

Export tables and charts:

```powershell
python scripts/export_figures.py
```

Outputs:
- `paper/tables/aggregate_metrics.md`
- `paper/tables/aggregate_metrics.json`
- `paper/tables/hybrid_semantic_metrics.md`
- `paper/figures/aggregate_higher_is_better.png`
- `paper/figures/aggregate_lower_is_better.png`
- `paper/figures/hybrid_constraint_hit_rate.png`
- `paper/figures/conflict_resolution_accuracy.png`

Recommended demo sequence:
1. `pytest -q`
2. `python scripts/check_gemini.py`
3. `python scripts/run_eval.py`
4. `python scripts/export_figures.py`
5. Open `paper/tables/aggregate_metrics.md` and chart PNGs
6. Start API and demonstrate `/docs`

## 10. Gemini-Specific Troubleshooting

### Error: `GEMINI_API_KEY is not set`

Cause:
- Key not exported in current shell and `.env` missing/incorrect

Fix:
1. Verify `.env` exists at repo root
2. Ensure line is exactly:
   - `GEMINI_API_KEY=...`
3. Restart terminal or re-run command in same shell
4. Validate with:
   - `python scripts/check_gemini.py`

### Error: package import issue

Fix:
- Ensure venv is active
- Reinstall editable package:
  - `pip install -e ".[dev,api]"`

### API starts but Gemini fails at request time

Cause:
- API key invalid or network restriction

Fix:
1. Recheck key
2. Run `python scripts/check_gemini.py`
3. Retry endpoint call from Swagger

## 11. Important Files Map

Configuration:
- `configs/model/default.yaml`
- `configs/memory/default.yaml`
- `configs/drift/default.yaml`
- `configs/eval/default.yaml`
- `configs/eval/seeds.yaml`

Execution scripts:
- `scripts/check_gemini.py`
- `scripts/run_eval.py`
- `scripts/export_figures.py`
- `scripts/train_calibrator.py`

Core runtime:
- `src/hidrift/agent/runtime.py`
- `src/hidrift/api.py`

LLM:
- `src/hidrift/llm/factory.py`
- `src/hidrift/llm/gemini.py`
- `src/hidrift/env.py`

## 12. Known Limitations

- Retrieval and storage are in-memory baseline implementations
- Evaluation currently synthetic; real-world traces are future work
- Metrics export is aggregate-first; richer experiment analytics can be added

## 13. Next Recommended Upgrades

1. Add persistent memory backend (LanceDB/Chroma)
2. Add MLflow logging for each run and seed
3. Add per-turn plots (drift score over time, consolidation trigger points)
4. Add real benchmark datasets beyond synthetic assistant simulation
