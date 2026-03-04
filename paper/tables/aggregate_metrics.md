| System | Success(mean) | Success CI95 | Precision(mean) | Hallucination(mean) | Adapt Latency(mean) |
| --- | --- | --- | --- | --- | --- |
| RAG-only | 0.7043 | [0.6809, 0.7417] | 0.9543 | 0.0000 | 1.0382 |
| HierMemory-noDrift | 0.7347 | [0.7319, 0.7417] | 0.9847 | 0.2319 | 0.9393 |
| VectorOnly-noGraph | 0.7043 | [0.6809, 0.7417] | 0.9543 | 0.3758 | 1.0382 |
| GraphOnly-noVector | 0.7418 | [0.7417, 0.7427] | 0.9918 | 0.3758 | 0.5471 |
| MemGPT-style | 0.7043 | [0.6809, 0.7417] | 0.9543 | 0.2361 | 1.0382 |
| GenerativeAgents-style | 0.7326 | [0.7250, 0.7417] | 0.9826 | 0.2250 | 1.0737 |
| FlatMem-TopK | 0.7043 | [0.6809, 0.7417] | 0.9543 | 0.0000 | 1.0382 |
| HiDrift-noConflict | 0.7418 | [0.7417, 0.7427] | 0.9918 | 0.3758 | 0.5471 |
| HiDrift-noDriftSignal | 0.7218 | [0.7100, 0.7417] | 0.9718 | 0.1972 | 1.5286 |
| HiDrift-full | 0.7418 | [0.7417, 0.7427] | 0.9918 | 0.3758 | 0.5471 |
