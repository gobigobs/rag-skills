"""rag-skills demo API — real eval results + a live TF-IDF retrieval demo.

Endpoints:
    GET  /            live demo page (type a question, see retrieved chunks)
    GET  /health      liveness check
    GET  /results     Gopi's real eval results (v2 / v3 / v4-corpus-v2)
    GET  /results/{version}   one eval run: v2 | v3 | v4_corpus_v2
    POST /ask         retrieval demo over the SAMPLE corpus (TF-IDF + BM25
                      hybrid via RRF). No model download, no LLM call — it
                      returns the top chunks and says "I don't know" when
                      the top chunk shares too few content words with the
                      question (RRF scores cannot gate: see data/results.json).

Run:
    .venv/bin/uvicorn api.main:app --port 8000
    # or: python3 -m api.main  (uses only the stdlib fallback server? no —
    # this file needs FastAPI; see requirements.txt)
"""

import json
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))

from eval import Bm25Retriever, HybridRetriever, TfidfRetriever, load_chunks  # noqa: E402

DATA_PATH = os.path.join(HERE, "data", "results.json")
CORPUS_DIR = os.path.join(HERE, "sample-corpus")
ABSTAIN_THRESHOLD = 0.02  # RRF score below this -> "I don't know"

with open(DATA_PATH) as f:
    RESULTS = json.load(f)

app = FastAPI(
    title="rag-skills demo API",
    description="Real RAG eval results from Gopi's rag-doc-qa project, plus a "
                "live TF-IDF/BM25 retrieval demo over a tiny sample corpus.",
    version="1.0.0",
)

# Build the demo index once at startup (chunking A: 500/100).
CHUNKS = load_chunks(CORPUS_DIR, 500, 100)
RETRIEVER = HybridRetriever()
RETRIEVER.fit(CHUNKS)

# Abstention gate: RRF scores CANNOT gate abstention — Gopi's evals measured
# a hit-control gap of ~0.000 for hybrid, i.e. RRF scores are uncalibrated.
# A TF-IDF cosine gate fails too on a tiny corpus, because stopwords
# ("what", "is", "the") dominate the cosine. So the gate is deliberately
# simple and transparent: the fraction of the question's content words
# (stopwords removed) that appear in the top chunk. A question about
# something not in the corpus scores ~0 and the API says "I don't know".
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "which", "who",
    "whom", "how", "why", "when", "where", "do", "does", "did", "can",
    "could", "should", "would", "will", "shall", "may", "might", "i", "me",
    "my", "we", "you", "he", "she", "it", "its", "they", "them", "this",
    "that", "these", "those", "there", "here", "in", "on", "at", "to", "of",
    "for", "with", "by", "from", "and", "or", "but", "not", "no", "if",
    "then", "than", "so", "as", "am", "be", "been", "being", "have", "has",
    "had", "into", "over", "under", "about",
}
ABSTAIN_THRESHOLD = 0.34  # min. content-word overlap with the top chunk


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500,
                          description="Question about the sample corpus")
    top_k: int = Field(3, ge=1, le=5, description="Chunks to return")


class ChunkHit(BaseModel):
    source: str
    chunk_id: int
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    abstained: bool
    top_score: float
    gate_score: float
    chunks: list[ChunkHit]


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(CHUNKS),
            "corpus": "sample-corpus (SAMPLE — 4 docs)"}


@app.get("/results")
def all_results():
    """Gopi's real eval results: v2 (21q), v3 (90q), v4 corpus-v2 (57q)."""
    return RESULTS


@app.get("/results/{version}")
def one_result(version: str):
    if version not in RESULTS or version == "meta":
        raise HTTPException(404, f"unknown version {version!r}; "
                                 "use v2, v3 or v4_corpus_v2")
    return RESULTS[version]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    ranked = RETRIEVER.search(req.question)[: req.top_k]
    top_score = ranked[0][0] if ranked else 0.0
    # Gate on content-word overlap with the top chunk, NOT the RRF score:
    # RRF scores are rank-only and carry no abstention signal (measured
    # hit-control gap ~0.000 in the real evals).
    from eval import tokenize as _tok
    content = [t for t in _tok(req.question) if t not in STOPWORDS]
    top_text = _tok(CHUNKS[ranked[0][1]]["text"]) if ranked else []
    gate_score = (sum(1 for t in set(content) if t in top_text) / len(set(content))
                  if content else 0.0)
    abstained = gate_score < ABSTAIN_THRESHOLD
    hits = [ChunkHit(source=CHUNKS[i]["source"], chunk_id=CHUNKS[i]["chunk_id"],
                     score=round(s, 4), text=CHUNKS[i]["text"])
            for s, i in ranked]
    if abstained:
        answer = ("I don't know — nothing in the sample corpus answers this "
                  "question confidently. (Only "
                  f"{gate_score:.0%} of the question's content words appear "
                  "in the best passage — below the "
                  f"{ABSTAIN_THRESHOLD:.0%} gate.)")
    else:
        answer = ("Most relevant passage is from "
                  f"{hits[0].source} (score {hits[0].score}). "
                  "This demo retrieves passages only — it does not generate "
                  "an answer, so read the passage itself.")
    return AskResponse(question=req.question, answer=answer,
                       abstained=abstained, top_score=round(top_score, 4),
                       gate_score=round(gate_score, 4), chunks=hits)


@app.get("/", response_class=HTMLResponse)
def demo_page():
    with open(os.path.join(HERE, "api", "demo.html")) as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
