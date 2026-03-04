| System | Success(mean) | Success CI95 | Precision(mean) | Hallucination(mean) | Adapt Latency(mean) |
| --- | --- | --- | --- | --- | --- |
| RAG-only | 0.5981 | [0.5630, 0.6542] | 0.9315 | 0.0000 | 1.4740 |
| HierMemory-noDrift | 0.6438 | [0.6396, 0.6542] | 0.9771 | 0.3479 | 1.3257 |
| VectorOnly-noGraph | 0.5981 | [0.5630, 0.6542] | 0.9315 | 0.4388 | 1.4740 |
| GraphOnly-noVector | 0.6544 | [0.6542, 0.6558] | 0.9877 | 0.4388 | 0.7373 |
| HiDrift-noConflict | 0.6544 | [0.6542, 0.6558] | 0.9877 | 0.4388 | 0.7373 |
| HiDrift-noDriftSignal | 0.6244 | [0.6067, 0.6542] | 0.9577 | 0.2958 | 2.2095 |
| HiDrift-full | 0.6544 | [0.6542, 0.6558] | 0.9877 | 0.4388 | 0.7373 |
