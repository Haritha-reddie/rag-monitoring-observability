"""
src/monitoring/metrics.py
--------------------------
Local metrics collector for p50/p95 latency, cost tracking,
and quality score aggregation.

Stores metrics in memory + a local JSON file so they persist
across restarts and can be read by the monitoring dashboard.

Production systems would use Prometheus/Grafana or Datadog.
This is a lightweight equivalent that demonstrates the same concepts.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
import numpy as np

METRICS_FILE = Path("monitoring_metrics.json")


@dataclass
class QueryMetric:
    """Metrics for a single RAG query."""
    timestamp: float
    query: str
    total_latency:      float
    retrieval_latency:  float
    generation_latency: float
    model: str
    input_tokens:  int
    output_tokens: int
    cost_usd:      float
    faithfulness:       Optional[float] = None
    answer_relevancy:   Optional[float] = None
    session_id:         Optional[str]   = None
    error:              Optional[str]   = None


@dataclass
class MetricsSummary:
    """Aggregated metrics summary."""
    total_queries:     int
    error_count:       int
    error_rate:        float

    # Latency percentiles (seconds)
    p50_latency:  float
    p95_latency:  float
    p99_latency:  float
    avg_latency:  float

    # Cost
    total_cost_usd: float
    avg_cost_usd:   float
    cost_per_1k_queries: float

    # Tokens
    avg_input_tokens:  float
    avg_output_tokens: float

    # Quality
    avg_faithfulness:     Optional[float]
    avg_answer_relevancy: Optional[float]

    # Time window
    window_hours: float
    since_timestamp: float


class MetricsCollector:
    """
    Collects and aggregates RAG pipeline metrics.

    Records every query and computes:
    - p50/p95/p99 latency percentiles
    - Cost per request and total cost
    - Quality score averages
    - Error rates
    """

    def __init__(self, metrics_file: Path = METRICS_FILE):
        self.metrics_file = metrics_file
        self._metrics: List[QueryMetric] = []
        self._load()

    def _load(self):
        """Load existing metrics from disk."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file) as f:
                    data = json.load(f)
                self._metrics = [QueryMetric(**m) for m in data]
                print(f"📊 Loaded {len(self._metrics)} existing metrics")
            except Exception as e:
                print(f"⚠️ Could not load metrics: {e}")
                self._metrics = []

    def _save(self):
        """Persist metrics to disk."""
        with open(self.metrics_file, "w") as f:
            json.dump([asdict(m) for m in self._metrics], f, indent=2)

    def record(self, metric: QueryMetric):
        """Record a new query metric."""
        self._metrics.append(metric)
        self._save()

    def summary(self, window_hours: float = 24.0) -> MetricsSummary:
        """
        Compute aggregated metrics over a time window.

        window_hours: Only include metrics from the last N hours.
                      Use 0 for all-time metrics.
        """
        cutoff = time.time() - (window_hours * 3600) if window_hours > 0 else 0
        recent = [m for m in self._metrics if m.timestamp >= cutoff]

        if not recent:
            return MetricsSummary(
                total_queries=0, error_count=0, error_rate=0.0,
                p50_latency=0, p95_latency=0, p99_latency=0, avg_latency=0,
                total_cost_usd=0, avg_cost_usd=0, cost_per_1k_queries=0,
                avg_input_tokens=0, avg_output_tokens=0,
                avg_faithfulness=None, avg_answer_relevancy=None,
                window_hours=window_hours, since_timestamp=cutoff,
            )

        latencies  = [m.total_latency for m in recent]
        costs      = [m.cost_usd for m in recent]
        in_tokens  = [m.input_tokens for m in recent]
        out_tokens = [m.output_tokens for m in recent]
        errors     = [m for m in recent if m.error]

        faith_scores = [m.faithfulness for m in recent if m.faithfulness is not None]
        rel_scores   = [m.answer_relevancy for m in recent if m.answer_relevancy is not None]

        total_cost = sum(costs)

        return MetricsSummary(
            total_queries  = len(recent),
            error_count    = len(errors),
            error_rate     = round(len(errors) / len(recent), 3),

            p50_latency    = round(float(np.percentile(latencies, 50)), 3),
            p95_latency    = round(float(np.percentile(latencies, 95)), 3),
            p99_latency    = round(float(np.percentile(latencies, 99)), 3),
            avg_latency    = round(float(np.mean(latencies)), 3),

            total_cost_usd       = round(total_cost, 6),
            avg_cost_usd         = round(total_cost / len(recent), 6),
            cost_per_1k_queries  = round(total_cost / len(recent) * 1000, 4),

            avg_input_tokens  = round(float(np.mean(in_tokens)), 1),
            avg_output_tokens = round(float(np.mean(out_tokens)), 1),

            avg_faithfulness     = round(float(np.mean(faith_scores)), 3) if faith_scores else None,
            avg_answer_relevancy = round(float(np.mean(rel_scores)), 3) if rel_scores else None,

            window_hours     = window_hours,
            since_timestamp  = cutoff,
        )

    def recent_queries(self, n: int = 20) -> List[QueryMetric]:
        """Return the N most recent query metrics."""
        return sorted(self._metrics, key=lambda m: m.timestamp, reverse=True)[:n]

    def check_regression(
        self,
        p95_threshold: float = 5.0,
        faithfulness_threshold: float = 0.80,
        error_rate_threshold: float = 0.05,
    ) -> dict:
        """
        Check if current metrics exceed regression thresholds.
        Returns a dict with pass/fail status for each metric.
        Used by CI gate to fail builds on regression.
        """
        summary = self.summary(window_hours=1.0)

        checks = {
            "p95_latency": {
                "value":     summary.p95_latency,
                "threshold": p95_threshold,
                "passed":    summary.p95_latency <= p95_threshold or summary.total_queries == 0,
                "unit":      "seconds",
            },
            "faithfulness": {
                "value":     summary.avg_faithfulness,
                "threshold": faithfulness_threshold,
                "passed":    (summary.avg_faithfulness is None or
                             summary.avg_faithfulness >= faithfulness_threshold),
                "unit":      "score (0-1)",
            },
            "error_rate": {
                "value":     summary.error_rate,
                "threshold": error_rate_threshold,
                "passed":    summary.error_rate <= error_rate_threshold,
                "unit":      "ratio (0-1)",
            },
        }

        checks["overall_passed"] = all(c["passed"] for c in checks.values() if isinstance(c, dict))
        return checks


# Global singleton
_collector: Optional[MetricsCollector] = None


def get_collector() -> MetricsCollector:
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
