# HiDrift Technical Documentation

## 1. Purpose
This document is the detailed technical reference for HiDrift implementation, covering runtime architecture, memory internals, API contracts, configuration semantics, evaluation methodology, and operational workflows.

For diagram-first explanation, see `ARCHITECTURE.md`.  
For quick operational onboarding, see `README.md`.
For dedicated test execution and validation flows, see `TESTING.md`.

## 2. System Summary
HiDrift is a long-horizon agent memory stack with:
1. Drift-aware control loop
2. Hierarchical memory (working, episodic, semantic)
3. Hybrid semantic memory retrieval (vector + graph)
4. Consolidation from episodes to semantic facts
5. Reproducible testing and reporting pipeline
6. Benchmark adapter layer for internal + external trace suites

## 3. Runtime Components

### 3.1 Agent Runtime
Main entrypoint:
1. `src/hidrift/agent/runtime.py`

Responsibilities:
1. Turn handling (`handle_turn`)
2. Optional LLM response generation (Gemini/fallback)
3. Drift signal processing
4. Memory ingestion and retrieval orchestration
5. Consolidation trigger execution

### 3.2 Drift Service
Files:
1. `src/hidrift/drift/features.py`
2. `src/hidrift/drift/scoring.py`
3. `src/hidrift/drift/service.py`

Signal components:
1. Behavioral shift (embedding distance from running mean)
2. Task shift (KL divergence over label distribution)
3. Performance drop (moving-window reward decline)

Trigger policy:
1. Threshold
2. Hysteresis count
3. Cooldown turns

### 3.3 Memory Service
Files:
1. `src/hidrift/memory/working.py`
2. `src/hidrift/memory/episodic.py`
3. `src/hidrift/memory/semantic.py`
4. `src/hidrift/memory/service.py`

Hierarchy:
1. Working memory: short recent-turn deque
2. Episodic memory: full episode records with embeddings/importance
3. Semantic memory:
   - legacy vector semantic items
   - structured semantic facts
   - graph store with persistence

### 3.4 Consolidation
Files:
1. `src/hidrift/consolidation/cluster.py`
2. `src/hidrift/consolidation/summarize.py`
3. `src/hidrift/consolidation/worker.py`

Flow:
1. Gather episodic records
2. Apply decay and pruning
3. Group by goal cluster
4. Generate summary statement(s)
5. Convert to structured semantic facts
6. Write facts to graph+vector semantic memory
7. Resolve fact conflicts

## 4. Hybrid Semantic Graph Layer

### 4.1 Modules
1. `src/hidrift/semantic_graph/adapter.py`
2. `src/hidrift/semantic_graph/networkx_store.py`
3. `src/hidrift/semantic_graph/query.py`
4. `src/hidrift/semantic_graph/reasoning.py`

### 4.2 Graph Backend
Default: `networkx.MultiDiGraph` in-memory.

Persistence:
1. Writes graph + fact snapshots to `artifacts/semantic_graph.json`
2. Loads snapshot on semantic memory initialization if file exists

### 4.3 Fact Conflict Resolution
Conflict resolution policy in current implementation:
1. Group by `(subject, relation)`
2. Keep winner by:
   - highest `confidence`
   - then highest `version`
   - then latest `created_at`
3. Mark only winner as `is_active=True`

## 5. Data Schemas
Defined in `src/hidrift/schemas.py`.

Primary runtime schemas:
1. `InteractionEvent`
2. `EpisodeRecord`
3. `DriftSignal`
4. `SemanticMemoryItem` (vector layer)
5. `SemanticFact` (structured semantic graph layer)
6. `GraphNode`, `GraphEdge`
7. `SemanticRelation`

`SemanticFact` key fields:
1. Identity: `fact_id`, `version`
2. Triple-like core: `subject`, `relation`, `object`
3. Quality: `confidence`, `stability`
4. Lifecycle: `is_active`, `valid_from`, `valid_to`
5. Traceability: `evidence_episode_ids`, `drift_event_ids`

## 6. Configuration Reference

### 6.1 Model config
File: `configs/model/default.yaml`
1. `llm.provider`: default `gemini`
2. `llm.model_name`: default `gemini-2.5-flash`
3. embedding metadata placeholders

### 6.2 Memory config
File: `configs/memory/default.yaml`
1. `working_maxlen`
2. `dedup_threshold`
3. `semantic_backend`: `hybrid`
4. `graph_backend`: `networkx`
5. `graph_persistence_path`
6. `graph_query_hops`
7. `fusion_weights.graph`
8. `fusion_weights.vector`
9. `fusion_weights.confidence`
10. `fusion_weights.stability`
11. importance and decay controls

### 6.3 Drift config
File: `configs/drift/default.yaml`
1. weighted coefficients
2. trigger threshold/hysteresis/cooldown
3. calibration quantile

### 6.4 Eval config
Files:
1. `configs/eval/default.yaml`
2. `configs/eval/seeds.yaml`
3. `configs/eval/benchmark_manifest.json`
4. `configs/eval/matrix_publishable.json`

## 7. API Contract
Implemented in `src/hidrift/api.py`.

### 7.1 Core endpoints
1. `POST /v1/memory/ingest`
2. `POST /v1/memory/retrieve`
3. `GET /v1/drift/current`
4. `POST /v1/consolidation/run`
5. `GET /v1/eval/run/{run_id}`

### 7.2 Semantic endpoints
1. `GET /v1/semantic/facts`
2. `GET /v1/semantic/graph/subgraph`
3. `POST /v1/semantic/facts/upsert`
4. `GET /v1/semantic/conflicts`

### 7.3 Retrieve response shape
`/v1/memory/retrieve` returns:
1. `working`
2. `episodic`
3. `semantic`
4. `hard_constraints`
5. `supporting_context`

## 8. LLM Provider Behavior

### 8.1 Gemini runtime
Production API runtime (`create_app`) starts with:
1. `llm_provider="gemini"`
2. `llm_model="gemini-2.5-flash"`
3. `require_llm=True`

### 8.2 Fallback runtime
Evaluation baseline builder (`src/hidrift/eval/baselines.py`) uses:
1. `llm_provider="fallback"`
2. `require_llm=False`

Reason:
1. Keep tests/eval deterministic under API quota/rate constraints
2. Prevent CI/local regression failures from external API limits

## 9. Testing Strategy

### 9.1 Unit tests
Coverage includes:
1. drift scoring
2. memory router monotonicity
3. consolidation decay/dedup
4. semantic graph upsert/conflict/hybrid retrieve behavior

### 9.2 Integration tests
Coverage includes:
1. end-to-end turn ingestion + retrieval
2. consolidation invocation under forced drift
3. semantic fact creation and superseded fact deactivation

### 9.3 Regression tests
Coverage includes:
1. fixed-seed eval output shape
2. no-drift guard checks
3. hybrid proxy metric threshold behavior

## 10. Evaluation Pipeline

### 10.1 Run evaluation
Command:
```powershell
python scripts/run_eval.py
```

Key options:
1. `--benchmark-profile` (`internal_v1`, `external_v1`, `mixed_v1`, `publishable_v1`)
2. `--benchmark-manifest` (JSON file for external traces)
3. `--systems` and `--seeds` for targeted sweeps

### 10.2 Run publishable matrix
Command:
```powershell
python scripts/run_eval_matrix.py --config configs/eval/matrix_publishable.json
```

Generates:
1. Multiple `artifacts/eval_<uuid>.json` reports (one per run block)
2. `artifacts/eval_matrix_<uuid>.json` index of matrix run IDs and metadata

### 10.3 Eval report contents
Each report stores:
1. `artifacts/eval_<uuid>.json`
2. `systems` (baseline/ablation matrix)
3. `scenario_reports` (per-scenario aggregates)
4. `traces` (per-turn observability logs)
5. `significance_vs_hidrift_full` (paired permutation p-values + effect sizes + Holm-adjusted p-values)
6. `benchmark_protocol` (seeds, profile, scenario list, reference system)

### 10.4 Calibrate drift threshold
Command:
```powershell
python scripts/train_calibrator.py
```

Generates:
1. `artifacts/calibration.json`

### 10.5 Benchmark registry validation
Command:
```powershell
python scripts/check_benchmark_registry.py
```

Outputs:
1. `paper/tables/benchmark_registry_check.json`
2. `paper/tables/benchmark_registry_check.md`

### 10.6 Publication readiness gates
Command:
```powershell
python scripts/check_publication_readiness.py
```

Gate outputs:
1. `paper/tables/publication_readiness.json`
2. `paper/tables/publication_readiness.md`

Current gate families:
1. minimum seeds
2. required external scenario coverage
3. baseline coverage
4. practical gain thresholds
5. significance thresholds (Holm-adjusted)
6. hypothesis-report completeness

## 11. Visualization Pipeline
Command:
```powershell
python scripts/export_figures.py
```

Outputs:
1. Aggregate tables:
   - `paper/tables/aggregate_metrics.md`
   - `paper/tables/aggregate_metrics.json`
2. Statistical tables:
   - `paper/tables/significance_report.md`
   - `paper/tables/scenario_metrics.md`
   - `paper/tables/hypothesis_results.md`
   - `paper/tables/cost_latency_table.md`
   - `paper/tables/qualitative_failure_cases.md`
3. Charts:
   - `paper/figures/task_success_with_errorbars.png`
   - `paper/figures/adaptation_latency_distribution.png`
   - `paper/figures/scenario_success_heatmap.png`
   - `paper/figures/drift_score_trace.png`
   - `paper/figures/memory_growth_trace.png`
   - `paper/figures/constraint_violation_trend.png`

## 12. Setup And Operations

### 12.1 Environment setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,api]"
```

### 12.2 `.env` example
```env
GEMINI_API_KEY=your_real_key
HIDRIFT_LLM_PROVIDER=gemini
HIDRIFT_LLM_MODEL=gemini-2.5-flash
```

### 12.3 Connectivity check
```powershell
python scripts/check_gemini.py
```

## 13. Troubleshooting

### 13.1 Gemini key missing
Symptom:
1. `RuntimeError: GEMINI_API_KEY is not set`

Fix:
1. Set env var or create `.env` in repo root
2. Re-run `python scripts/check_gemini.py`

### 13.2 Gemini 429 quota/rate limit
Symptom:
1. `RESOURCE_EXHAUSTED`

Fix:
1. Wait for quota reset or use paid/billed project
2. Continue local tests/eval via fallback mode

### 13.3 Missing dependencies
Fix:
```powershell
pip install -e ".[dev,api]"
```

## 14. Limitations
1. Semantic graph backend is local `networkx`, not distributed
2. Benchmark realism improved but still includes synthetic-heavy components
3. Statistical tests use paired permutation + effect size + Holm-Bonferroni correction, but benchmark diversity is still limited to included traces
4. Vector memory is lightweight in-process rather than dedicated vector DB

## 15. Future Extension Points
1. Replace networkx with Neo4j/Memgraph adapter while retaining interface
2. Add LanceDB/FAISS integration for semantic vector persistence
3. Add MLflow experiment tracking and Hydra config composition
4. Add richer timeline-level plotting from raw turn traces
5. Extend conflict policies with contradiction/supersession provenance scoring
