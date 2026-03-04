| System | Avg Turn Latency (ms) | Consolidation Events / 100 turns | Memory Bloat |
| --- | --- | --- | --- |
| RAG-only | 4.567 | 0.000 | 63.000 |
| HierMemory-noDrift | 5.405 | 0.417 | 25.500 |
| VectorOnly-noGraph | 6.632 | 60.875 | 37.517 |
| GraphOnly-noVector | 5.875 | 60.875 | 37.517 |
| MemGPT-style | 1.284 | 4.167 | 25.867 |
| GenerativeAgents-style | 1.299 | 5.000 | 25.867 |
| FlatMem-TopK | 1.647 | 0.000 | 63.000 |
| HiDrift-noConflict | 15.679 | 60.875 | 37.517 |
| HiDrift-noDriftSignal | 4.134 | 2.500 | 25.750 |
| HiDrift-full | 8.537 | 60.875 | 37.517 |
