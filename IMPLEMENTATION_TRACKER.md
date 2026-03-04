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
- [ ] M7: Docs fully aligned with publishable workflow

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

### Statistics
- [x] Bootstrap 95% CI for core metrics
- [x] Paired permutation significance test
- [x] Effect size estimates
- [x] Save significance report artifact

### Visualization
- [x] Scenario-wise comparison plots
- [x] Adaptation latency distribution plot
- [x] Memory growth curves from turn traces
- [x] Constraint violation trend plot

### Documentation
- [ ] Update README quick path for publishable runs
- [ ] Update DOCUMENTATION with methods + stats protocol
- [ ] Update TESTING with benchmark/stat validation
- [ ] Update ARCHITECTURE with benchmark/report flow
