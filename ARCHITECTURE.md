# HiDrift Architecture And Flow Diagrams

## 1. High-Level System

```mermaid
graph TD
    User[User] --> API[FastAPI Layer]
    API --> Runtime[AgentRuntime]
    Runtime --> Drift[DriftService]
    Runtime --> Memory[MemoryService]
    Memory --> WM[WorkingMemory]
    Memory --> EM[EpisodicMemory]
    Memory --> SM[SemanticMemory]
    SM --> Vector[Vector Semantic Index]
    SM --> Graph[Semantic Graph Store]
    Runtime --> Consolidator[ConsolidationWorker]
    Consolidator --> EM
    Consolidator --> SM
    Runtime --> LLM[Gemini Or Fallback LLM]
    Runtime --> API
```

## 2. Request Lifecycle (Ingest)

```mermaid
sequenceDiagram
    participant U as User
    participant A as API
    participant R as AgentRuntime
    participant D as DriftService
    participant M as MemoryService
    participant C as ConsolidationWorker

    U->>A: POST /v1/memory/ingest
    A->>R: handle_turn(payload)
    alt agent_output omitted
        R->>R: llm.generate()
    end
    R->>D: process(event)
    D-->>R: DriftSignal
    R->>M: ingest_interaction(event, drift)
    R->>M: retrieve(query)
    alt drift.triggered == true
        R->>C: run_once()
        C->>M: write semantic items/facts
    end
    R-->>A: event + drift + retrieval summary
    A-->>U: response JSON
```

## 3. Drift Detection Internals

```mermaid
flowchart LR
    Event[InteractionEvent] --> Embed[Embedding]
    Embed --> Behav[Behavioral Shift]
    Event --> Task[Task Distribution Shift]
    Event --> Perf[Performance Drop]
    Behav --> Score[Weighted Drift Score]
    Task --> Score
    Perf --> Score
    Score --> Hys[Hysteresis Check]
    Hys --> Trig{Trigger?}
    Trig -->|Yes| Consolidate[Consolidation Trigger]
    Trig -->|No| Continue[Continue Turn Loop]
```

## 4. Hybrid Retrieval Pipeline

```mermaid
flowchart TB
    Q[User Query] --> QE[Query Embedding]
    QE --> VS[Vector Similarity Over Semantic Facts]
    QE --> GS[Graph Relevance Scoring]
    GS --> HC[Hard Constraints]
    VS --> SC[Supporting Context]
    HC --> Fuse[Fusion Ranker]
    SC --> Fuse
    Fuse --> Bundle[Retrieval Bundle]
    Bundle --> Response[Agent Uses Constraints + Context]
```

## 5. Semantic Fact Consolidation Path

```mermaid
flowchart LR
    Episodic[Episodic Records] --> Decay[Decay + Prune]
    Decay --> Cluster[Cluster By Goal]
    Cluster --> Summarize[LLM Or Deterministic Summary]
    Summarize --> Facts[Structured Semantic Facts]
    Facts --> GraphWrite[Write Graph Nodes And Edges]
    Facts --> VectorWrite[Write Vector Semantic Facts]
    GraphWrite --> Conflict[Conflict Resolution]
    VectorWrite --> Conflict
    Conflict --> Active[Active Fact Set]
```

## 6. Drift Trigger State Machine

```mermaid
stateDiagram-v2
    [*] --> Stable
    Stable --> Watch: score > threshold
    Watch --> Stable: score <= threshold
    Watch --> Triggered: above threshold for m turns
    Triggered --> Cooldown
    Cooldown --> Stable: cooldown turns elapsed
```

## 7. Semantic Graph Entity Model

```mermaid
erDiagram
    SEMANTIC_FACT {
        string fact_id
        string subject
        string relation
        string object
        float confidence
        bool is_active
        int version
    }
    EPISODE {
        string episode_id
    }
    DRIFT_EVENT {
        string drift_id
    }
    ENTITY {
        string entity_id
        string label
    }

    ENTITY ||--o{ SEMANTIC_FACT : HAS_FACT
    SEMANTIC_FACT ||--o{ EPISODE : OBSERVED_IN
    SEMANTIC_FACT ||--o{ DRIFT_EVENT : TRIGGERED_BY_DRIFT
    SEMANTIC_FACT ||--o{ SEMANTIC_FACT : SUPPORTS
    SEMANTIC_FACT ||--o{ SEMANTIC_FACT : CONTRADICTS
    SEMANTIC_FACT ||--o{ SEMANTIC_FACT : SUPERSEDES
```

## 8. API Surface Map

```mermaid
graph LR
    Client[Client] --> Ingest["POST /v1/memory/ingest"]
    Client --> Retrieve["POST /v1/memory/retrieve"]
    Client --> Drift["GET /v1/drift/current"]
    Client --> Consolidate["POST /v1/consolidation/run"]
    Client --> Eval["GET /v1/eval/run/{run_id}"]
    Client --> Facts["GET /v1/semantic/facts"]
    Client --> Subgraph["GET /v1/semantic/graph/subgraph"]
    Client --> Upsert["POST /v1/semantic/facts/upsert"]
    Client --> Conflicts["GET /v1/semantic/conflicts"]
```

## 9. Evaluation And Reporting Flow

```mermaid
flowchart TB
    MatrixScript[run_eval_matrix.py] --> EvalScript[run_eval.py]
    EvalScript --> Suite[Benchmark Suite]
    Suite --> B1[personal_assistant_drift]
    Suite --> B2[tool_api_drift]
    Suite --> B3[contradiction_drift]
    Suite --> B4[semi_real_trace]
    Suite --> B5[locomo_like_trace]
    Suite --> B6[longmem_like_trace]
    B1 --> EvalArtifact["artifacts/eval_RUN_ID.json"]
    B2 --> EvalArtifact
    B3 --> EvalArtifact
    B4 --> EvalArtifact
    B5 --> EvalArtifact
    B6 --> EvalArtifact
    MatrixScript --> MatrixArtifact["artifacts/eval_matrix_MATRIX_ID.json"]
    EvalArtifact --> Sig[Significance Analysis]
    EvalArtifact --> ExportScript[export_figures.py]
    EvalArtifact --> BenchCheck[check_benchmark_registry.py]
    EvalArtifact --> PubCheck[check_publication_readiness.py]
    Sig --> T1[paper/tables/significance_report.md]
    BenchCheck --> T0[paper/tables/benchmark_registry_check.md]
    PubCheck --> TP[paper/tables/publication_readiness.md]
    ExportScript --> T2[paper/tables/aggregate_metrics.md]
    ExportScript --> T3[paper/tables/scenario_metrics.md]
    ExportScript --> T4[paper/tables/hypothesis_results.md]
    ExportScript --> T5[paper/tables/cost_latency_table.md]
    ExportScript --> T6[paper/tables/qualitative_failure_cases.md]
    ExportScript --> F1[paper/figures/task_success_with_errorbars.png]
    ExportScript --> F2[paper/figures/adaptation_latency_distribution.png]
    ExportScript --> F3[paper/figures/scenario_success_heatmap.png]
    ExportScript --> F4[paper/figures/drift_score_trace.png]
    ExportScript --> F5[paper/figures/memory_growth_trace.png]
    ExportScript --> F6[paper/figures/constraint_violation_trend.png]
```

## 10. Developer Navigation Map

```mermaid
graph TD
    Root[Project Root] --> Src[src/hidrift]
    Root --> Cfg[configs]
    Root --> Scripts[scripts]
    Root --> Tests[tests]
    Root --> Paper[paper]

    Src --> Agent[agent/runtime.py]
    Src --> Drift[drift/*]
    Src --> Memory[memory/*]
    Src --> GraphMod[semantic_graph/*]
    Src --> Consolidation[consolidation/*]
    Src --> Eval[eval/*]
    Src --> API[src/hidrift/api.py]

    Tests --> Unit[tests/unit]
    Tests --> Integration[tests/integration]
    Tests --> Regression[tests/regression]
```
