# HiDrift Presentation Brief

## 1. One-Line Pitch
HiDrift is a drift-aware hierarchical memory framework for long-horizon LLM agents that combines working, episodic, and hybrid semantic memory (graph + vector) with consolidation to improve stability over evolving tasks.

## 2. Problem Statement
Long-horizon agents fail over time because:
1. Memory bloat accumulates irrelevant traces.
2. Retrieval quality degrades as context drifts.
3. Preference/task/environment changes are not detected early.
4. Flat vector memory cannot enforce hard constraints well.

## 3. Core Contribution
HiDrift introduces a full memory loop:
1. `interaction -> drift scoring -> hierarchical routing`
2. `episodic storage -> consolidation -> semantic fact graph`
3. `hybrid retrieval -> hard constraints + supporting context`

Key novelty components:
1. Drift-triggered consolidation with hysteresis.
2. Structured semantic facts with conflict handling and versioning.
3. Hybrid retrieval that fuses graph relevance and semantic similarity.

## 4. Architecture Walkthrough
Use [ARCHITECTURE.md](D:/project/HiDrift-implemntation/ARCHITECTURE.md) as slide source.

Presentation flow:
1. High-level system diagram (runtime, drift, memory, consolidation).
2. Sequence diagram (ingest turn lifecycle).
3. Drift state machine.
4. Semantic graph entity model.
5. Evaluation and reporting flow.

## 5. Technical Design
### Memory Layers
1. Working memory: recent turn buffer.
2. Episodic memory: structured experience records with importance.
3. Semantic memory: vector items + graph facts with active/inactive lifecycle.

### Drift Model
1. Behavioral shift from embedding dynamics.
2. Task shift from online distribution change.
3. Performance drop from reward trend.
4. Triggered by threshold + hysteresis + cooldown.

### Consolidation
1. Episodic decay and pruning.
2. Cluster by goal.
3. Generate summaries and structured facts.
4. Upsert facts into graph with evidence links.
5. Resolve conflicts and retain active facts.

## 6. Experimental Setup
### Systems Compared
1. `RAG-only`
2. `HierMemory-noDrift`
3. `MemGPT-style`
4. `GenerativeAgents-style`
5. `FlatMem-TopK`
6. `VectorOnly-noGraph`
7. `HiDrift-noDriftSignal`
8. `HiDrift-noConflict`
9. `HiDrift-full`

### Scenario Families
1. Personal assistant drift.
2. Tool/API schema drift.
3. Contradiction/reversal drift.
4. Semi-real local traces.
5. Official converted benchmark traces.

### Metrics
1. Task success rate.
2. Retrieval precision@k and recall@k.
3. Constraint violation rate.
4. Hallucination rate.
5. Adaptation latency.
6. Memory bloat.
7. Stability score.

## 7. Result Storyline (Slide Narrative)
1. HiDrift increases long-horizon success over non-adaptive baselines.
2. Drift-triggered consolidation reduces adaptation latency versus fixed-schedule memory.
3. Hybrid semantic design controls memory growth better than flat memory.
4. Ablations show drift trigger and conflict logic are meaningful components.

Use these generated artifacts:
1. [aggregate_metrics.md](D:/project/HiDrift-implemntation/paper/tables/aggregate_metrics.md)
2. [significance_report.md](D:/project/HiDrift-implemntation/paper/tables/significance_report.md)
3. [task_success_with_errorbars.png](D:/project/HiDrift-implemntation/paper/figures/task_success_with_errorbars.png)
4. [adaptation_latency_distribution.png](D:/project/HiDrift-implemntation/paper/figures/adaptation_latency_distribution.png)
5. [scenario_success_heatmap.png](D:/project/HiDrift-implemntation/paper/figures/scenario_success_heatmap.png)

## 8. Reproducibility Slide
Show exact command chain:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e ".[dev,api]"
python scripts/prepare_official_benchmarks.py
make eval_official
python scripts/export_figures.py
pytest -q
```

## 9. Limitations (State Clearly)
1. Converted official traces are still an adapter representation.
2. Graph backend is local `networkx` (not distributed production graph).
3. LLM response quality depends on provider/quota if live model is used.

## 10. Future Work Slide
1. Native loaders for full official benchmark schemas.
2. Larger-scale long-context evaluation and stress tests.
3. Multi-agent and multi-user extension.
4. Learned consolidation policies beyond fixed heuristics.

## 11. Suggested Presentation Structure (12-15 Slides)
1. Title + motivation
2. Failure modes in long-horizon agents
3. HiDrift core idea
4. Architecture
5. Drift detection method
6. Hierarchical memory design
7. Consolidation + semantic graph
8. Experimental protocol
9. Main quantitative results
10. Ablation results
11. Qualitative trace examples
12. Limitations and risk
13. Conclusion and contributions
14. Backup: API/tests/reproducibility commands
