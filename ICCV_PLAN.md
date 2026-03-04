# ICCV-Grade Plan (Implemented Scaffold)

## Goal
Deliver ICCV-grade empirical rigor on top of HiDrift by enforcing:
1. Official benchmark adapters
2. Fixed protocol matrix
3. Strict readiness gates
4. Artifact traceability

## What Is Implemented
1. `iccv_v1` benchmark profile wired into eval runner.
2. Official benchmark adapter loader:
   - `src/hidrift/eval/official_benchmarks.py`
3. ICCV manifests/config:
   - `configs/eval/official_benchmark_manifest.json`
   - `configs/eval/matrix_iccv.json`
   - `configs/eval/benchmark_registry_iccv.json`
   - `configs/eval/iccv_policy.json`
4. ICCV readiness checker:
   - `scripts/check_iccv_readiness.py`
5. Make targets:
   - `make eval_iccv`
   - `make iccv_check`

## Required Next Data Step
Populate official benchmark files:
1. `data/benchmarks/official/locomo/locomo_official.jsonl`
2. `data/benchmarks/official/longmem/longmem_official.jsonl`

Raw source files used by converter:
1. `data/benchmarks/official/locomo/locomo10.json`
2. `data/benchmarks/official/longmem/longmemeval_oracle.json`
3. `data/benchmarks/official/longmem/longmemeval_s_cleaned.json`
4. `data/benchmarks/official/longmem/longmemeval_m_cleaned.json`

Generate JSONL via:
1. `python scripts/prepare_official_benchmarks.py`

Each line format:
```json
{
  "user_input": "text",
  "expected_style": "concise|detailed|bullet",
  "task_label": "task",
  "oracle_fact": "fact-string",
  "drift": false
}
```

## ICCV Pipeline Commands
1. `make eval_iccv`
2. `python scripts/export_figures.py`
3. `make iccv_check`

## PASS Condition
`paper/tables/iccv_readiness_summary.json` reports `"status": "PASS"`.
