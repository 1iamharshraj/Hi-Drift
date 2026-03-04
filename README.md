# HiDrift

HiDrift is a research implementation of drift-aware hierarchical memory for long-horizon agents.

Full project guide: `DOCUMENTATION.md`
Architecture diagrams: `ARCHITECTURE.md`

## Quickstart

```bash
uv venv
uv pip install -e .[dev]
set GEMINI_API_KEY=your_key_here
python scripts/check_gemini.py
pytest
python scripts/run_eval.py
```

PowerShell:

```powershell
$env:GEMINI_API_KEY="your_key_here"
```

Or use `.env` in repo root:

```env
GEMINI_API_KEY=your_key_here
HIDRIFT_LLM_PROVIDER=gemini
HIDRIFT_LLM_MODEL=gemini-2.5-flash
```

## API

```bash
uv pip install -e .[api]
uvicorn hidrift.api:create_app --factory --reload
```

Endpoints:
- `POST /v1/memory/ingest`
- `POST /v1/memory/retrieve`
- `GET /v1/drift/current`
- `POST /v1/consolidation/run`
- `GET /v1/eval/run/{run_id}`
- `GET /v1/semantic/facts`
- `GET /v1/semantic/graph/subgraph`
- `POST /v1/semantic/facts/upsert`
- `GET /v1/semantic/conflicts`

`POST /v1/memory/ingest` can omit `agent_output`; then HiDrift generates it via Gemini.

## Gemini model

Default backend model is `gemini-2.5-flash`.

You can override:

```powershell
$env:HIDRIFT_LLM_PROVIDER="gemini"
$env:HIDRIFT_LLM_MODEL="gemini-2.5-flash"
```

The code also normalizes the typo form `gemini-2,5-flash` to `gemini-2.5-flash`.

## Visualize Results

Run evaluation and export visual artifacts:

```powershell
python scripts/run_eval.py
python scripts/export_figures.py
```

Generated files:
- `paper/tables/aggregate_metrics.md` (readable leaderboard table)
- `paper/tables/aggregate_metrics.json`
- `paper/tables/hybrid_semantic_metrics.md`
- `paper/figures/aggregate_higher_is_better.png`
- `paper/figures/aggregate_lower_is_better.png`
- `paper/figures/hybrid_constraint_hit_rate.png`
- `paper/figures/conflict_resolution_accuracy.png`

## Show Testing Clearly

Quick test run:

```powershell
pytest -q
```

Detailed test run with timing:

```powershell
pytest -v --durations=10
```
