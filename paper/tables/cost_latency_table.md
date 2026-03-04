| System | Avg Turn Latency (ms) | Consolidation Events / 100 turns | Memory Bloat |
| --- | --- | --- | --- |
| RAG-only | 0.964 | 0.000 | 63.000 |
| HierMemory-noDrift | 1.592 | 3.333 | 69.333 |
| VectorOnly-noGraph | 4.187 | 60.875 | 77.817 |
| GraphOnly-noVector | 3.668 | 60.875 | 77.817 |
| MemGPT-style | 1.106 | 4.167 | 70.667 |
| GenerativeAgents-style | 1.061 | 5.000 | 71.833 |
| FlatMem-TopK | 1.008 | 0.000 | 63.000 |
| HiDrift-noConflict | 3.831 | 60.875 | 77.817 |
| HiDrift-noDriftSignal | 1.163 | 2.500 | 67.833 |
| HiDrift-full | 3.804 | 60.875 | 77.817 |
