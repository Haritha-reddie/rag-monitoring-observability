# RAG Monitoring and Observability

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/) [![Langfuse](https://img.shields.io/badge/Langfuse-Tracing-purple.svg)](https://langfuse.com/) [![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)

A production-grade monitoring and observability layer for RAG systems. Adds Langfuse tracing, p50/p95/p99 latency tracking, cost per request, quality score monitoring, a live dashboard, and a CI regression gate to any RAG pipeline.

This is Project 3 in the AI Engineer Portfolio Series and is a direct continuation of Project 1 (Production RAG Application). It does not work as a standalone project. The monitoring layer wraps the RAG pipeline from Project 1 and adds full observability on top of it.

Project 1 repo: https://github.com/Haritha-reddie/production-rag

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Why Monitoring Matters](#why-monitoring-matters)
- [Components](#components)
  - [Langfuse Tracing](#langfuse-tracing)
  - [Metrics Collector](#metrics-collector)
  - [Query Cache](#query-cache)
  - [Monitoring Dashboard](#monitoring-dashboard)
  - [CI Regression Gate](#ci-regression-gate)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the Dashboard](#running-the-dashboard)
- [CI Gate](#ci-gate)
- [Sample Dashboard Output](#sample-dashboard-output)

---

## What This Project Does

Every time a user asks a question in the RAG system, this monitoring layer:

1. Sends a trace to Langfuse showing the full span tree: retrieval time, rerank time, generation time, token counts, and estimated cost
2. Records latency and cost metrics locally for percentile calculations
3. Serves a live dashboard at localhost:8002 showing p50/p95/p99 latency, total cost, error rate, and recent query history
4. Runs a CI regression gate that fails the GitHub Actions build if p95 latency exceeds 5 seconds or error rate exceeds 5 percent

---

## Why Monitoring Matters

Most AI engineers build the pipeline and stop there. Production systems need answers to these questions continuously:

- Is the system getting slower over time?
- Which queries are the most expensive?
- Is the hallucination rate increasing after a prompt change?
- Did the last code change break anything?

Without monitoring, these questions can only be answered after users complain. With monitoring, regressions are caught automatically before they reach users.

This is why the video this project is based on says monitoring is 70 percent of production AI work that nobody puts in their portfolio.

---

## Components

### Langfuse Tracing

Every RAG query is traced end to end with a span tree showing each step.

```python
from src.monitoring.tracer import RAGTracer

tracer = RAGTracer()

with tracer.trace(query="What is the return policy?") as trace:
    # log the retrieval step
    trace.log_retrieval(
        candidates=20,
        reranked=3,
        latency=0.8,
        top_scores=[0.916, 0.089, 0.005]
    )

    # log the generation step with token counts and cost
    trace.log_generation(
        model="llama-3.3-70b-versatile",
        prompt=prompt_text,
        response=answer,
        input_tokens=312,
        output_tokens=148,
        latency=2.1
    )

    # log quality scores from Ragas
    trace.log_quality(faithfulness=0.92, relevancy=0.88)

    trace.set_output(answer)
```

Each trace appears in your Langfuse dashboard at cloud.langfuse.com with:
- Full span tree showing time spent at each step
- Token counts and estimated cost in USD
- Faithfulness and relevancy scores
- Model name and parameters

**Cost estimation**

```python
MODEL_COST = {
    "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
}

cost = (input_tokens / 1000) * rates["input"] + (output_tokens / 1000) * rates["output"]
```

---

### Metrics Collector

Records every query locally and computes percentile latency, cost aggregates, and quality score averages.

```python
from src.monitoring.metrics import get_collector, QueryMetric
import time

collector = get_collector()

collector.record(QueryMetric(
    timestamp=time.time(),
    query="What is the return policy?",
    total_latency=2.84,
    retrieval_latency=0.72,
    generation_latency=2.12,
    model="llama-3.3-70b-versatile",
    input_tokens=312,
    output_tokens=148,
    cost_usd=0.000301,
))
```

Getting a summary over the last 24 hours:

```python
summary = collector.summary(window_hours=24)

print(summary.p50_latency)    # 2.461
print(summary.p95_latency)    # 6.174
print(summary.p99_latency)    # 6.989
print(summary.total_cost_usd) # 0.0022
print(summary.error_rate)     # 0.0
```

The difference between average and p95 latency is important. An average of 2.8 seconds looks fine. A p95 of 6.2 seconds means 5 percent of users are waiting over 6 seconds. That is what production SRE teams measure.

---

### Query Cache

Caches high-frequency queries to reduce latency and cost for repeated questions.

```python
from src.monitoring.cache import get_cache

cache = get_cache()

# check cache before running the RAG pipeline
cached = cache.get("What is the return policy?")
if cached:
    return cached  # returns in ~5ms instead of ~3 seconds

# run the full pipeline
response = rag_chain.run(query)

# store result with 1 hour TTL
cache.set("What is the return policy?", {"answer": response.answer})
```

Cache statistics:

```python
stats = cache.stats()
# {
#   "total_entries": 12,
#   "active_entries": 10,
#   "expired_entries": 2,
#   "total_hits": 47,
#   "max_entries": 500
# }
```

The cache uses SHA-256 hashing of the normalized query as the key, TTL-based expiry at 1 hour, and LRU eviction when the cache exceeds 500 entries.

---

### Monitoring Dashboard

A live dashboard served at localhost:8002 showing all metrics in real time.

```bash
python -m src.monitoring.dashboard
```

The dashboard shows:

- Total queries and error rate
- Average latency in seconds
- Total cost and average cost per query
- Average faithfulness and relevancy scores
- Latency percentile bars for p50, p95, p99, and average
- CI regression gate status per metric
- Recent queries table with timestamp, latency, tokens, cost, and status

The dashboard auto-refreshes every 10 seconds and supports time window filtering: 1 hour, 6 hours, 24 hours, and all time.

---

### CI Regression Gate

Checks current metrics against thresholds and exits with code 1 if any check fails.

```python
from src.monitoring.metrics import get_collector

collector = get_collector()
checks = collector.check_regression(
    p95_threshold=5.0,
    faithfulness_threshold=0.80,
    error_rate_threshold=0.05,
)
```

Sample output from a passing run:

```
Check                     Value    Threshold   Status
p95_latency               4.23s    5.000s      PASS
faithfulness              0.91     0.800       PASS
error_rate                0.000    0.050       PASS

All checks passed. CI gate APPROVED.
```

Sample output from a failing run:

```
Check                     Value    Threshold   Status
p95_latency               6.17s    5.000s      FAIL
faithfulness              0.91     0.800       PASS
error_rate                0.000    0.050       PASS

CI gate FAILED:
  p95_latency: 6.174 exceeded threshold 5.0
```

---

## Project Structure

```
rag-monitoring/
├── src/
│   └── monitoring/
│       ├── __init__.py
│       ├── tracer.py          Langfuse tracing wrapper with cost estimation
│       ├── metrics.py         p50/p95/p99 latency collector and aggregator
│       ├── cache.py           TTL query cache with LRU eviction
│       └── dashboard.py       FastAPI dashboard server and UI
└── ci/
    └── run_monitoring_gate.py CI gate script, exits 1 on regression
```

---

## Setup

**Step 1: Get a free Langfuse account**

Go to cloud.langfuse.com, sign up, create a project named production-rag, and copy your API keys from Settings.

**Step 2: Add keys to your .env**

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
P95_LATENCY_THRESHOLD=5.0
FAITHFULNESS_THRESHOLD=0.80
ERROR_RATE_THRESHOLD=0.05
```

**Step 3: Install dependencies**

```bash
pip install langfuse==2.36.2 fastapi==0.115.0 uvicorn==0.30.6 numpy
```

---

## Running the Dashboard

```bash
set -a && source .env && set +a
python -m src.monitoring.dashboard
```

Open http://localhost:8002

---

## CI Gate

```bash
python ci/run_monitoring_gate.py
```

The gate reads metrics from the last 1 hour. If no queries exist in that window it skips the check and exits 0.

In GitHub Actions, add this step after your Ragas evaluation:

```yaml
- name: Run monitoring regression gate
  env:
    P95_LATENCY_THRESHOLD: "5.0"
    FAITHFULNESS_THRESHOLD: "0.80"
    ERROR_RATE_THRESHOLD: "0.05"
  run: python ci/run_monitoring_gate.py
```

---

## Sample Dashboard Output

After running 10 queries against the RAG system:

```
Total queries:    10
Error rate:       0.0%
Avg latency:      2.827s
Total cost:       $0.0022
Avg cost/query:   $0.000216

Latency percentiles:
  p50:  2.461s
  p95:  6.174s
  p99:  6.989s
  avg:  2.827s

CI Regression Gate:
  p95_latency:   6.174s  threshold 5.0s   FAIL
  faithfulness:  N/A     threshold 0.80   PASS
  error_rate:    0.000   threshold 0.05   PASS
```

The p95 latency fails because query transformation generates 6 query variants, adding time. Raising the threshold to 8 seconds in .env resolves this for systems using query transformation.

---

## Part of the AI Engineer Portfolio Series

| # | Project | Status |
|---|---|---|
| 1 | Production RAG Application | Complete |
| 2 | Local SLM App with Ollama | Complete |
| 3 | Monitoring and Observability | Complete |
| 4 | Fine-Tuning with LoRA and DPO | Coming soon |
| 5 | Real-Time Multimodal App | Coming soon |

---

## Author

Haritha Gurram,AI Engineer based in Dallas, TX.

harithagurram5@gmail.com | github.com/Haritha-reddie

