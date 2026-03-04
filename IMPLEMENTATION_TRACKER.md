# Publishable Upgrade Tracker

## Goal
Upgrade HiDrift from prototype evaluation to publishable-grade evidence pipeline.

## Milestones
- [x] M1: Define tracked execution plan in repo
- [x] M2: Multi-benchmark suite (synthetic + semi-real traces)
- [x] M3: Strong baseline matrix (external-style + ablations)
- [x] M4: Statistical reporting (CI, paired significance, effect size)
- [x] M5: Per-turn trace logging and discriminative visualizations
- [x] M6: Tests updated for new report schema and robustness
- [x] M7: Docs fully aligned with publishable workflow
- [x] M8: Automated publication readiness gates
- [x] M9: Publication gates passing on current matrix report

## Task Board

### Benchmark realism
- [x] Add scenario families: preference, goal, tool/API, contradiction, mixed drift
- [x] Add semi-real trace loader from local dataset
- [x] Add scenario-level reporting slices

### Baseline rigor
- [x] Add vector-only semantic baseline
- [x] Add graph-only semantic baseline
- [x] Add no-conflict-resolution ablation
- [x] Add no-drift-trigger ablation
- [x] Add literature-style wrappers (MemGPT-style, GenerativeAgents-style, FlatMem-TopK)

### Statistics
- [x] Bootstrap 95% CI for core metrics
- [x] Paired permutation significance test
- [x] Effect size estimates
- [x] Save significance report artifact
- [x] Add publication pass/fail gate checks

### Visualization
- [x] Scenario-wise comparison plots
- [x] Adaptation latency distribution plot
- [x] Memory growth curves from turn traces
- [x] Constraint violation trend plot

### Documentation
- [x] Update README quick path for publishable runs
- [x] Update DOCUMENTATION with methods + stats protocol
- [x] Update TESTING with benchmark/stat validation
- [x] Update ARCHITECTURE with benchmark/report flow
- [x] Add explicit publication gate outputs and commands
- [x] Record latest PASS status artifacts for publication checks
