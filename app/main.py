import logging
import re
import os
import time
import uuid
from collections import deque

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .generate import REFUSAL, clean_citations, get_generator
from fastapi.responses import FileResponse
from .insights import build_insights
from .lending import assess
from .rag import BM25Index, load_chunks
from . import store
from .vectors import VectorIndex

DATA_DIR = os.getenv("DATA_DIR", "data")
log = logging.getLogger("mini-rag")
logging.basicConfig(level=logging.INFO, format="%(message)s")

app = FastAPI(title="Mini-RAG", description="Grounded Q&A with citations. Sample documents only.")
index = BM25Index(load_chunks(DATA_DIR))
vindex = VectorIndex(index.chunks)
db = store.connect()
store.save_chunks(db, index.chunks, vindex.E)
cache: dict = {}
cstats = {"hits": 0, "misses": 0}
generator = get_generator()
audit: deque = deque(maxlen=50)


def check_key(x_api_key: str | None = Header(default=None)):
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(401, "Invalid API key")


class AskIn(BaseModel):
    question: str = Field(min_length=2, max_length=300)
    k: int = Field(default=3, ge=1, le=5)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(index.chunks), "generator": generator.name}


@app.post("/ask", dependencies=[Depends(check_key)])
def ask(body: AskIn):
    rid, start = str(uuid.uuid4()), time.time()
    key = (body.question.strip().lower(), body.k)
    if key in cache:
        cstats["hits"] += 1
        out = {**cache[key], "request_id": rid, "cached": True, "latency_ms": round((time.time() - start) * 1000, 2)}
        store.log_query(db, body.question, out["grounded"], True, out["latency_ms"])
        return out
    cstats["misses"] += 1
    res = index.search(body.question, k=body.k)
    if res["grounded"]:
        res["hits"] = vindex.search(body.question, k=body.k) or res["hits"]
        answer = clean_citations(generator.answer(body.question, res["hits"]), len(res["hits"]))
    else:
        answer = REFUSAL
    ms = round((time.time() - start) * 1000, 2)
    entry = {"request_id": rid, "question": body.question, "grounded": res["grounded"],
             "coverage": res["coverage"], "generator": generator.name, "latency_ms": ms}
    audit.appendleft(entry)
    log.info(entry)
    cites = [{"n": n, "source": h["source"], "heading": h["heading"], "text": h["text"], "score": h["score"]}
             for n, h in enumerate(res["hits"], 1)] if res["grounded"] else []
    out = {"answer": answer, "grounded": res["grounded"], "coverage": res["coverage"],
           "citations": cites, "request_id": rid, "latency_ms": ms, "cached": False}
    cache[key] = out
    store.log_query(db, body.question, res["grounded"], False, ms)
    return out


@app.get("/audit", dependencies=[Depends(check_key)])
def get_audit():
    return list(audit)


class Applicant(BaseModel):
    months_in_business: int = Field(ge=0, le=1200)
    annual_revenue: float = Field(ge=0, le=1e9)
    monthly_cash_flow: float = Field(ge=0, le=1e8)
    existing_monthly_debt: float = Field(ge=0, le=1e8)
    credit_score: int = Field(ge=300, le=850)
    amount: float = Field(ge=1000, le=1e7)
    term_months: int = Field(ge=6, le=60)


@app.post("/api/eligibility")
def eligibility(a: Applicant):
    out = assess(a.model_dump())
    for c in out["checks"]:
        hit = vindex.search(c["policy_query"], k=1)
        c["policy"] = hit[0]["text"] if hit else None
    for o in out["offers"]:
        hit = vindex.search(o["query"], k=1)
        o["text"] = hit[0]["text"] if hit else None
    proc = vindex.search("how to apply steps", k=1)
    out["process_steps"] = re.split(r"\s+(?=Step \d)", proc[0]["text"]) if proc else []
    return out


@app.get("/api/vector")
def vector(q: str = ""):
    return {"dims": vindex.vt.shape[0], "map": vindex.map2d(q or None), "hits": vindex.search(q, k=3) if q else []}


@app.get("/api/insights")
def insights():
    return build_insights()


@app.get("/api/pipeline")
def pipeline():
    n = len(index.chunks)
    return {"stages": [
        {"name": "Ingest", "detail": f"{len({c['source'] for c in index.chunks})} documents written by us (policy, offers, process) in the data folder."},
        {"name": "Chunk", "detail": f"Split into {n} paragraph chunks, each tagged with its source file and heading."},
        {"name": "Embed", "detail": f"{len(vindex.vocab)} distinct terms turned into {vindex.vt.shape[0]}-number vectors (TF-IDF, then SVD)."},
        {"name": "Store", "detail": f"Text and vectors saved in SQLite ({store.count(db, 'chunks')} rows) and held in memory for fast search."},
        {"name": "Retrieve", "detail": "The question becomes a vector. Cosine similarity picks the top 3 chunks. Off-topic questions are refused."},
        {"name": "Generate", "detail": f"The {generator.name} generator writes the answer from those chunks only, with citations."},
        {"name": "Cache and log", "detail": f"{cstats['hits']} repeat questions answered from cache, {cstats['misses']} computed, {store.count(db, 'query_log')} logged for analytics."}]}


@app.get("/", response_class=FileResponse)
def home():
    return FileResponse("static/index.html")
