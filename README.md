# Production RAG — Ask My Docs

I built this because I was frustrated with how most RAG tutorials end. They show you how to embed some text, do a similarity search, and call an LLM. That works for a demo. It falls apart in production.

This project is my attempt to build a RAG system the way I'd actually want to deploy one — with hybrid retrieval, proper reranking, citation enforcement, and automated quality checks that run on every single push.

---

## The problem I was trying to solve

Most document Q&A systems have two failure modes that nobody talks about:

**The vocabulary problem.** Your user asks "can I get a refund?" but your document says "return policy." Vector search might catch this. BM25 won't. But then your user asks "what does GDPR article 17 require?" and now vector search fails because the meaning is too diluted across the embedding space, while BM25 finds the exact phrase instantly. You need both.

**The hallucination problem.** LLMs are trained to sound confident. If you ask a question and the retrieved context doesn't contain the answer, the model will often make something up rather than say it doesn't know. I've seen this wreck trust in AI systems faster than anything else. The fix isn't hoping the model behaves — it's building the system so the model literally cannot answer without a source.

This project tackles both.

---

## How it works

### Step 1 — Ingestion

Drop any `.txt`, `.pdf`, or `.md` file into `data/docs/`. The system chunks it into 512-token pieces with 50-token overlap (so context isn't lost at chunk boundaries), then builds two indexes:

- A **ChromaDB vector index** using HuggingFace embeddings for semantic search
- A **BM25 keyword index** for exact-match search

Both indexes are saved to disk and reloaded on startup.

### Step 2 — Hybrid retrieval

When a question comes in, both indexes are queried simultaneously. BM25 returns its top 20 candidates. Vector search returns its top 20. Then Reciprocal Rank Fusion merges them — a chunk that appears in both lists scores higher than one that appears in only one. The formula is simple: `score = 1/(rank + 60)` summed across methods.

The result is a merged, deduplicated list of the most relevant chunks, combining the precision of keyword search with the flexibility of semantic search.

### Step 3 — Reranking

The top 20 candidates from hybrid retrieval go to Cohere's reranker. This is the part most people skip, and it makes a huge difference.

The difference between a bi-encoder (embeddings) and a cross-encoder (reranker) is this: embeddings encode the query and the document separately, then compare the vectors. A cross-encoder reads the query and the document together and scores their relationship directly. It's much more accurate, but too slow to run on thousands of chunks. Running it on 20 candidates is fast and gives you near-perfect precision.

The top 3 chunks after reranking go to the LLM.

### Step 4 — Generation with citation enforcement

The system prompt tells the LLM three things:
1. Only use the provided context chunks to answer
2. Tag every factual claim with `[Source: chunk_id]`
3. If the context doesn't contain the answer, say so — don't guess

That third rule is the most important. When the reranker finds no relevant chunks (top score below threshold), the pipeline returns "I don't have enough information in the provided documents" instead of generating a potentially fabricated answer.

### Step 5 — Automated evaluation

After every push to main, GitHub Actions runs Ragas evaluation against a test set of Q&A pairs. Four metrics are checked:

| Metric | What it actually measures | Threshold |
|---|---|---|
| Faithfulness | Did the LLM only use the retrieved context? | ≥ 0.85 |
| Answer relevancy | Did the answer address what was asked? | ≥ 0.80 |
| Context precision | Were the retrieved chunks actually relevant? | ≥ 0.75 |
| Context recall | Did retrieval find everything it needed? | ≥ 0.75 |

If any metric drops below its threshold, the build fails. This catches regressions from prompt changes, chunking changes, or retrieval parameter changes before they reach users.

---

## Project layout

```
production-rag/
├── src/
│   ├── ingestion/
│   │   ├── loader.py          load docs, chunk them, stamp each with a chunk_id
│   │   └── indexer.py         build ChromaDB + BM25 indexes, persist to disk
│   ├── retrieval/
│   │   ├── hybrid.py          BM25 + vector search, RRF fusion
│   │   └── reranker.py        Cohere cross-encoder, top 20 → top 3
│   ├── generation/
│   │   ├── prompt.py          system prompt with citation rules
│   │   └── rag_chain.py       orchestrates the full pipeline
│   └── evaluation/
│       └── eval_pipeline.py   Ragas runner
├── ui/
│   ├── server.py              FastAPI — upload, query, status endpoints
│   └── index.html             drag-drop upload + chat interface
├── ci/
│   └── run_evals.py           CI gate script, exits 1 if metrics fail
├── tests/
│   ├── test_retrieval.py      unit tests for RRF and BM25
│   ├── test_generation.py     unit tests for prompt citation enforcement
│   └── eval_dataset.json      ground truth Q&A pairs
├── data/docs/
│   └── acme_policy.txt        sample document
└── .github/workflows/
    └── rag_eval.yml           GitHub Actions pipeline
```

---

## Getting it running

You need three things: a Groq API key (free), a Cohere API key (free trial), and Python 3.11.

```bash
git clone https://github.com/Haritha-reddie/production-rag.git
cd production-rag

conda activate ragenv
pip install -r requirements.txt

cp .env.example .env
# add your GROQ_API_KEY and COHERE_API_KEY

set -a && source .env && set +a
python main.py ingest
python main.py query "What is the return policy?"
```

To use the web UI:
```bash
python -m uvicorn ui.server:app --host 0.0.0.0 --port 8000 --reload
```

Then open `http://localhost:8000`, drag in any document, and start asking questions.

---

## Things worth testing

The sample document (`acme_policy.txt`) is an ACME Corp policy guide. These questions should all get good cited answers:

```
What is the return policy?
How do I contact customer support?
What payment methods are accepted?
How long does shipping take?
```

These should trigger the hallucination fallback — "I don't have enough information":

```
What is the CEO's name?
What are the store hours on Sundays?
What is the price of the premium plan?
```

That second set is important. If the system answers those with made-up information, something is wrong with the citation enforcement or the confidence threshold.

---

## Running the evaluation

```bash
python main.py eval
```

This runs the full Ragas suite against `tests/eval_dataset.json` and prints a table showing each metric and whether it passed. The same script runs in CI on every push.

---

## Stack

- **LLM** — Groq (Llama 3.3 70B) — free tier, fast inference
- **Embeddings** — HuggingFace all-MiniLM-L6-v2 — runs locally, no API needed
- **Vector store** — ChromaDB — persisted locally
- **Keyword search** — rank-bm25 — BM25Okapi, pickled to disk
- **Reranking** — Cohere rerank-english-v3.0 — free trial key
- **Evaluation** — Ragas 0.2.x
- **Backend** — FastAPI
- **CI** — GitHub Actions

---

## What I'd add next

A few things I deliberately left out to keep the scope focused:

- **Redis cache** — right now repeated queries hit the LLM every time. A simple Redis cache on the query hash would cut latency to milliseconds for frequent questions
- **RAPTOR indexing** — building a hierarchical summary tree over the chunks so the system can answer high-level questions that span multiple sections
- **Streaming responses** — the UI currently waits for the full answer before displaying anything, which feels slow for longer responses

---

## Author

Haritha Gurram — Data Scientist and AI Engineer based in Dallas, TX.

harithagurram5@gmail.com | [github.com/Haritha-reddie](https://github.com/Haritha-reddie)
