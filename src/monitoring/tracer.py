"""
src/monitoring/tracer.py
-------------------------
Langfuse tracing wrapper for the Production RAG pipeline.

Every query gets a trace with:
- Full span tree (retrieval → rerank → generation)
- Latency per step
- Token counts
- Estimated cost
- Quality scores (faithfulness, relevancy)
- Model name and parameters
"""

import os
import time
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context


# Initialise Langfuse client (reads keys from env automatically)
langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)


# ── Cost table ($ per 1K tokens) ───────────────────────────────
# Using Groq pricing as proxy for free-tier estimation
MODEL_COST = {
    "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
    "gpt-4o":                  {"input": 0.005,   "output": 0.015},
    "gpt-4o-mini":             {"input": 0.00015, "output": 0.0006},
    "claude-3-5-sonnet":       {"input": 0.003,   "output": 0.015},
}

DEFAULT_COST = {"input": 0.001, "output": 0.002}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a generation call."""
    rates = MODEL_COST.get(model, DEFAULT_COST)
    cost = (input_tokens / 1000) * rates["input"] + \
           (output_tokens / 1000) * rates["output"]
    return round(cost, 6)


class RAGTracer:
    """
    Wraps the RAG pipeline with Langfuse tracing.

    Usage:
        tracer = RAGTracer()
        with tracer.trace(query="What is the return policy?") as trace:
            # retrieval
            trace.log_retrieval(candidates=20, reranked=3, latency=0.8)
            # generation
            trace.log_generation(model="llama-3.3-70b", tokens=450, latency=2.1)
            # quality
            trace.log_quality(faithfulness=0.92, relevancy=0.88)
    """

    def __init__(self):
        self.langfuse = langfuse

    def trace(self, query: str, session_id: Optional[str] = None):
        """Start a new trace for a RAG query."""
        return RAGTrace(
            langfuse=self.langfuse,
            query=query,
            session_id=session_id,
        )

    def flush(self):
        """Flush all pending traces to Langfuse."""
        self.langfuse.flush()


class RAGTrace:
    """Context manager for a single RAG query trace."""

    def __init__(self, langfuse: Langfuse, query: str, session_id: Optional[str]):
        self.langfuse   = langfuse
        self.query      = query
        self.session_id = session_id
        self.trace_obj  = None
        self.start_time = None
        self.metadata   = {}

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.trace_obj  = self.langfuse.trace(
            name="rag-query",
            input=self.query,
            session_id=self.session_id,
            metadata={"pipeline": "production-rag"},
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        total_latency = round(time.perf_counter() - self.start_time, 3)
        self.metadata["total_latency_seconds"] = total_latency

        if exc_type:
            self.trace_obj.update(
                output="ERROR",
                metadata={**self.metadata, "error": str(exc_val)},
                level="ERROR",
            )
        else:
            self.trace_obj.update(
                metadata=self.metadata,
            )

        self.langfuse.flush()
        return False  # Don't suppress exceptions

    def log_retrieval(
        self,
        candidates: int,
        reranked: int,
        latency: float,
        top_scores: list = None,
    ):
        """Log the retrieval + reranking step."""
        span = self.trace_obj.span(
            name="hybrid-retrieval",
            input={"query": self.query, "top_k": candidates},
            output={"reranked_count": reranked, "top_scores": top_scores or []},
            metadata={
                "candidates_retrieved": candidates,
                "reranked_to": reranked,
                "latency_seconds": latency,
            },
        )
        self.metadata["retrieval_latency_seconds"] = latency
        self.metadata["candidates_retrieved"] = candidates
        self.metadata["reranked_to"] = reranked

    def log_generation(
        self,
        model: str,
        prompt: str,
        response: str,
        input_tokens: int,
        output_tokens: int,
        latency: float,
        temperature: float = 0.0,
    ):
        """Log the LLM generation step with cost estimation."""
        cost = estimate_cost(model, input_tokens, output_tokens)

        generation = self.trace_obj.generation(
            name="llm-generation",
            model=model,
            input=prompt,
            output=response,
            usage={
                "input":        input_tokens,
                "output":       output_tokens,
                "total":        input_tokens + output_tokens,
                "unit":         "TOKENS",
                "input_cost":   round((input_tokens / 1000) * MODEL_COST.get(model, DEFAULT_COST)["input"], 6),
                "output_cost":  round((output_tokens / 1000) * MODEL_COST.get(model, DEFAULT_COST)["output"], 6),
                "total_cost":   cost,
            },
            metadata={
                "latency_seconds": latency,
                "temperature":     temperature,
            },
        )

        self.metadata["generation_latency_seconds"] = latency
        self.metadata["model"] = model
        self.metadata["input_tokens"] = input_tokens
        self.metadata["output_tokens"] = output_tokens
        self.metadata["estimated_cost_usd"] = cost

    def log_quality(
        self,
        faithfulness: float,
        relevancy: float,
        answer: str = "",
    ):
        """Log Ragas quality scores as Langfuse scores."""
        if self.trace_obj:
            self.langfuse.score(
                trace_id=self.trace_obj.id,
                name="faithfulness",
                value=faithfulness,
                comment="Ragas faithfulness — does answer only use context?",
            )
            self.langfuse.score(
                trace_id=self.trace_obj.id,
                name="answer_relevancy",
                value=relevancy,
                comment="Ragas answer relevancy — does answer address the question?",
            )

        self.metadata["faithfulness"] = faithfulness
        self.metadata["answer_relevancy"] = relevancy

    def set_output(self, answer: str):
        """Set the final answer as the trace output."""
        if self.trace_obj:
            self.trace_obj.update(output=answer)
