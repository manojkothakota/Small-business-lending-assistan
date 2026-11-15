# Harbor Street Capital: small business lending assistant (portfolio demo)
Fictional lender, synthetic data, no affiliation with any bank. Estimates only, not credit decisions.

## What it does
- **Eligibility check** with deterministic rules, estimated payment, matched discounts and the application steps.
- **Policy assistant** (RAG): answers only from documents we wrote, with citations, and refuses off-topic questions.
- **Growth insights** from synthetic applications: near-miss applicants, weak spots, product and partnership ideas.

## RAG pipeline
ingest (our documents) -> chunk by paragraph -> embed (TF-IDF + SVD vectors) -> store (SQLite + in-memory matrix)
-> retrieve (cosine top 3 + relevance gate) -> generate (Grok or offline extractive, context only) -> check citations
-> cache repeat questions and log queries. Live numbers are on the page and at `/api/pipeline`.

## Run
```
pip install -r requirements.txt
cp .env.example .env     # add XAI_API_KEY to use Grok; works offline without it
uvicorn app.main:app --reload
pytest && python -m eval.run
```

## Limits and next steps
Local vectors instead of a hosted embedding model; SQLite instead of Postgres with pgvector; in-memory cache;
synthetic applications; no auth. Next: hybrid search with reranking, larger eval set, human review queue.
