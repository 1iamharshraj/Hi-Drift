# HiDrift: Formal Definitions

This document provides the mathematical formalization for all core algorithms
implemented in the HiDrift system. Each definition maps directly to source code
with file and line references.

---

## 1. Drift Detection

### 1.1 Composite Drift Score

At each turn t, the system computes a scalar drift score from three signals:

```
D(t) = alpha * B(t) + beta * T(t) + gamma * P(t)
```

where alpha = 0.45, beta = 0.30, gamma = 0.25.

**Source**: `src/hidrift/drift/scoring.py` DriftScorer.score()

### 1.2 Behavioral Shift (Embedding Distance)

Let e_t be the embedding of the user-agent interaction at turn t, and
mu_t be the online running mean of all embeddings up to turn t.

```
B(t) = || e_t - mu_{t-1} ||_2

mu_t = (1 - 1/n_t) * mu_{t-1} + (1/n_t) * e_t
```

where n_t is the count of embeddings seen through turn t.

**Source**: `src/hidrift/drift/features.py` OnlineDriftState.update_embedding()

### 1.3 Task Distribution Shift (KL Divergence)

Let p_t be the normalized histogram of task labels through turn t.

```
T(t) = D_KL( p_t || p_{t-1} )
     = sum_k  p_t(k) * log( p_t(k) / p_{t-1}(k) )
```

with epsilon smoothing (eps = 1e-8) to avoid division by zero.

**Source**: `src/hidrift/drift/features.py` OnlineDriftState.update_task_distribution()
**Source**: `src/hidrift/utils.py` kl_divergence()

### 1.4 Performance Drop (Sliding Window)

Let r_t be the reward at turn t, and W be a sliding window of size w = 20.

```
P(t) = max( mean(r_{t-w : t-w/2}) - mean(r_{t-w/2 : t}), 0 )
```

The drop is clamped to non-negative: a performance *increase* does not
register as drift.

**Source**: `src/hidrift/drift/features.py` OnlineDriftState.update_performance()

---

## 2. Trigger Policy (State Machine)

The drift signal triggers consolidation according to a hysteresis policy
with cooldown:

```
consecutive_above(t) =
    0                                if cooldown(t) > 0
    consecutive_above(t-1) + 1       if D(t) > tau
    0                                otherwise

triggered(t) = 1  iff  consecutive_above(t) >= h

cooldown(t) =
    c                                if triggered(t)
    max(cooldown(t-1) - 1, 0)        otherwise
```

Parameters: threshold tau = 0.35, hysteresis h = 3, cooldown c = 2.

**Source**: `src/hidrift/drift/service.py` DriftService.process()

---

## 3. Memory Hierarchy

### 3.1 Importance Score

For an episodic record e, the importance score determines retention priority:

```
I(e) = w_r * recency(e) + w_u * usage(e) + w_p * reward(e) + w_s * stability(e)
```

where w_r = 0.25, w_u = 0.25, w_p = 0.30, w_s = 0.20.

**Source**: `src/hidrift/memory/router.py` MemoryRouter.importance_score()

### 3.2 Keep Probability

The probability of retaining an episode under drift pressure:

```
P_keep(e) = sigmoid( I(e) - delta_drift )
```

where sigmoid(x) = 1 / (1 + exp(-x)).

**Source**: `src/hidrift/memory/router.py` MemoryRouter.keep_probability()

### 3.3 Episodic Decay

Episodic importance decays exponentially with age:

```
I'(e) = I(e) * exp( -k * age_days(e) )
```

where k = 0.08 (default decay rate).

**Source**: `src/hidrift/memory/episodic.py` EpisodicMemory.apply_decay()

### 3.4 Episodic Retrieval Score

When retrieving episodes for a query q:

```
score(e, q) = 0.7 * cosine_sim(emb(q), emb(e)) + 0.3 * I(e)
```

Top-k episodes are returned, ranked by this score.

**Source**: `src/hidrift/memory/episodic.py` EpisodicMemory.top_k()

---

## 4. Hybrid Semantic Retrieval

### 4.1 Fusion Scoring

For each active semantic fact f and query embedding q:

```
score(f, q) = w_g * R_graph(f) + w_v * sim(q, f) + w_c * conf(f) + w_s * stab(f)
```

where:
- w_g = 0.45 (graph relevance weight)
- w_v = 0.30 (vector similarity weight)
- w_c = 0.15 (confidence weight)
- w_s = 0.10 (stability weight)
- R_graph(f) = 1.0 if f.relation in {PREFERS, RULE_FOR, SUPERSEDES}, else 0.6
- sim(q, f) = cosine_similarity(q, f.embedding)
- conf(f) = f.confidence
- stab(f) = f.stability

The output is partitioned into:
- **hard_constraints**: facts with relation in {PREFERS, RULE_FOR, SUPERSEDES}
- **supporting_context**: top-k facts by fusion score

**Source**: `src/hidrift/memory/semantic.py` SemanticMemory.hybrid_retrieve()

### 4.2 Deduplication

New semantic memory items are rejected if their cosine similarity
to any existing item exceeds the dedup threshold:

```
reject(m_new) = exists m in M : cosine_sim(m_new, m) >= theta_dedup
```

where theta_dedup = 0.88.

**Source**: `src/hidrift/memory/semantic.py` SemanticMemory.add()

---

## 5. Conflict Resolution

### 5.1 Winner Selection

For each group of facts sharing the same (subject, relation) pair,
the active fact is selected by lexicographic maximization:

```
winner = argmax_{f in group} ( confidence(f), version(f), created_at(f) )

is_active(f) = 1  iff  f = winner
```

All non-winner facts in the group are set to is_active = False.
They remain in the store for provenance but are excluded from retrieval.

**Source**: `src/hidrift/semantic_graph/reasoning.py` resolve_conflicts()

### 5.2 Fact Upsert (Merge Policy)

When inserting a fact f_new with the same (subject, relation, object) triple
as an existing fact f_old:

```
f_merged.confidence   = max(f_old.confidence, f_new.confidence)
f_merged.stability    = max(f_old.stability, f_new.stability)
f_merged.version      = max(f_old.version, f_new.version) + 1
f_merged.evidence     = f_old.evidence  UNION  f_new.evidence
f_merged.statement    = longer(f_old.statement, f_new.statement)
f_merged.is_active    = True
```

**Source**: `src/hidrift/memory/semantic.py` SemanticMemory.upsert_fact()

---

## 6. Consolidation Pipeline

When drift triggers consolidation, the following steps execute:

1. **Decay**: Apply exponential decay to all episodic importance scores.
2. **Prune**: Remove episodes below min_importance threshold; cap at 40.
3. **Cluster**: Group remaining episodes by goal label.
4. **Summarize**: For each cluster, generate a summary statement (LLM or deterministic).
5. **Extract**: Derive structured SemanticFact triples from each cluster.
6. **Upsert**: Write facts into the semantic graph with evidence links.
7. **Resolve**: Run conflict resolution to deactivate superseded facts (if enabled).

**Source**: `src/hidrift/consolidation/worker.py` ConsolidationWorker.run_once()

---

## 7. Evaluation Metrics

### 7.1 Task Success Rate

```
success(t) = 1  iff  reward(t) >= 0.75  AND  precision(t) >= 0.65
TSR = (1/N) * sum_{t=1}^{N} success(t)
```

### 7.2 Adaptation Latency

```
latency(d) = t_recover - t_drift
AL = mean over all drift events d where recovery occurred
```

If no recovery occurs, latency = N (max penalty).

### 7.3 Constraint Violation Rate

```
violated(t) = 1  iff  exists f in hard_constraints :
    task_label(t) in f.subject  AND
    expected_style(t) not in f.object  AND
    expected_style(t) not in f.statement

CVR = (1/N) * sum_{t=1}^{N} violated(t)
```

### 7.4 Hallucination Rate

```
hallucinated(t) = 1  iff  exists f in supporting_context :
    task_label(t) in f.statement  AND
    exists s in KNOWN_STYLES \ {expected_style(t)} : s in f.statement

HR = (1/N) * sum_{t=1}^{N} hallucinated(t)
```

**Source**: `src/hidrift/eval/runner.py`, `src/hidrift/eval/metrics.py`
