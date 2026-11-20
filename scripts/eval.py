#!/usr/bin/env python3
"""rag-skills sample eval — compare 5 retrievers on the SAMPLE corpus.

Runs TF-IDF, BM25, SBERT, hybrid (RRF) and cross-encoder rerank against the
tiny sample corpus in ``sample-corpus/`` and prints a comparison table with
the same metrics as Gopi's real evals (accuracy@1, MRR, margin, control
scores, latency).

Zero-install design: TF-IDF and BM25 are implemented in pure Python (stdlib
only), so the demo always runs. SBERT and cross-encoder rerank need
``sentence-transformers`` — if it is not installed those two methods are
skipped with a clear note instead of failing. No model is ever downloaded by
this script. (Gopi's real evals used ``all-MiniLM-L6-v2`` and
``cross-encoder/ms-marco-MiniLM-L-6-v2``; their measured numbers live in
``data/results.json``.)

Usage:
    python3 scripts/eval.py [--chunking A|B] [--top-k N] [--json out.json]

Chunkings mirror the real evals: A = 500 chars / 100 overlap,
B = 250 chars / 50 overlap.
"""

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN_RE = re.compile(r"[^a-z0-9\s]")

CHUNKINGS = {"A": (500, 100), "B": (250, 50)}
RRF_K = 60  # Cormack, Clarke & Buttcher, SIGIR 2009


# ----------------------------------------------------------------- utilities
def tokenize(text):
    return TOKEN_RE.sub(" ", text.lower()).split()


def load_documents(corpus_dir):
    docs = {}
    for name in sorted(os.listdir(corpus_dir)):
        if name.endswith(".md"):
            with open(os.path.join(corpus_dir, name)) as f:
                docs[name] = f.read()
    return docs


def chunk_text(text, size, overlap):
    """Greedy char chunks, split on word boundaries, never mid-word."""
    chunks, start = [], 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_chunks(corpus_dir, chunk_size, overlap):
    chunks = []
    for source, text in load_documents(corpus_dir).items():
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append({"source": source, "chunk_id": i, "text": piece})
    return chunks


# --------------------------------------------------------------- retrievers
class TfidfRetriever:
    """Classic TF-IDF with cosine similarity (pure Python)."""

    name = "tfidf"

    def fit(self, chunks):
        self.chunks = chunks
        self.docs = [tokenize(c["text"]) for c in chunks]
        n = len(self.docs)
        df = Counter()
        for toks in self.docs:
            df.update(set(toks))
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self.vecs, self.norms = [], []
        for toks in self.docs:
            tf = Counter(toks)
            vec = {t: (cnt / len(toks)) * self.idf[t] for t, cnt in tf.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.vecs.append(vec)
            self.norms.append(norm)

    def search(self, query):
        qtoks = tokenize(query)
        tf = Counter(qtoks)
        qvec = {t: (c / len(qtoks)) * self.idf.get(t, 0.0)
                for t, c in tf.items() if t in self.idf}
        qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        out = []
        for i, vec in enumerate(self.vecs):
            dot = sum(qvec[t] * vec.get(t, 0.0) for t in qvec)
            out.append((dot / (qnorm * self.norms[i]), i))
        return sorted(out, key=lambda x: -x[0])


class Bm25Retriever:
    """BM25 Okapi (Robertson et al.), pure Python. k1=1.5, b=0.75."""

    name = "bm25"

    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    def fit(self, chunks):
        self.chunks = chunks
        self.docs = [tokenize(c["text"]) for c in chunks]
        n = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(n, 1)
        df = Counter()
        for toks in self.docs:
            df.update(set(toks))
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5))
                    for t, c in df.items()}

    def search(self, query):
        qtoks = tokenize(query)
        out = []
        for i, toks in enumerate(self.docs):
            tf = Counter(toks)
            dl = len(toks)
            score = 0.0
            for t in qtoks:
                if t not in self.idf:
                    continue
                f = tf.get(t, 0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += self.idf[t] * f * (self.k1 + 1) / denom
            out.append((score, i))
        return sorted(out, key=lambda x: -x[0])


class SbertRetriever:
    """Dense retrieval with sentence-transformers (optional dependency).

    Skipped cleanly when the library is not installed — the demo never
    downloads a model on its own. Gopi's real SBERT numbers are in
    data/results.json (all-MiniLM-L6-v2).
    """

    name = "sbert"
    available = False

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            SbertRetriever.available = True
        except Exception as exc:  # noqa: BLE001 - optional dep, any failure skips
            self.skip_reason = f"sentence-transformers unavailable ({exc})"

    def fit(self, chunks):
        import numpy as np
        self.chunks = chunks
        self.emb = self.model.encode([c["text"] for c in chunks],
                                     normalize_embeddings=True,
                                     show_progress_bar=False)
        self.np = np

    def search(self, query):
        q = self.model.encode([query], normalize_embeddings=True,
                              show_progress_bar=False)[0]
        scores = self.emb @ q
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [(float(scores[i]), i) for i in order]


class HybridRetriever:
    """Reciprocal Rank Fusion over TF-IDF + BM25 (+ SBERT when available).

    RRF(d) = sum over rankings of 1 / (k + rank(d)), k = 60 (Cormack et al.,
    SIGIR 2009). Uses ranks only, so scores on different scales never mix.
    """

    name = "hybrid"

    def __init__(self, sbert=None):
        self.parts = [TfidfRetriever(), Bm25Retriever()]
        if sbert is not None:
            self.parts.append(sbert)

    def fit(self, chunks):
        self.chunks = chunks
        for p in self.parts:
            p.fit(chunks)

    def search(self, query):
        fused = defaultdict(float)
        for p in self.parts:
            for rank, (_, idx) in enumerate(p.search(query), start=1):
                fused[idx] += 1.0 / (RRF_K + rank)
        return [(score, idx) for idx, score in
                sorted(fused.items(), key=lambda kv: -kv[1])]


class RerankRetriever:
    """Cross-encoder rerank over the hybrid top-20 (optional dependency).

    Skipped cleanly when sentence-transformers is not installed. Gopi's real
    rerank numbers are in data/results.json (ms-marco-MiniLM-L-6-v2).
    """

    name = "rerank"
    available = False

    def __init__(self, base):
        self.base = base
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            RerankRetriever.available = True
        except Exception as exc:  # noqa: BLE001 - optional dep, any failure skips
            self.skip_reason = f"cross-encoder unavailable ({exc})"

    def fit(self, chunks):
        self.chunks = chunks
        self.base.fit(chunks)

    def search(self, query, topn=20):
        ranked = self.base.search(query)[:topn]
        pairs = [(query, self.chunks[i]["text"]) for _, i in ranked]
        scores = self.model.predict(pairs)
        order = sorted(range(len(ranked)), key=lambda j: -scores[j])
        return [(float(scores[j]), ranked[j][1]) for j in order]


# -------------------------------------------------------------------- eval
def evaluate(retriever, chunks, questions):
    rows, hit_scores, ctrl_scores, latencies = [], [], [], []
    for q in questions:
        t0 = time.perf_counter()
        ranked = retriever.search(q["question"])
        latencies.append((time.perf_counter() - t0) * 1000)
        # rank (1-based) of first chunk from the expected doc; None if absent
        rank = None
        for pos, (_, idx) in enumerate(ranked, start=1):
            if chunks[idx]["source"] == q["expected_source"]:
                rank = pos
                break
        top_score = ranked[0][0] if ranked else 0.0
        margin = (ranked[0][0] - ranked[1][0]) if len(ranked) > 1 else 0.0
        if q["type"] == "control":
            ctrl_scores.append(top_score)
            verdict = "CONTROL"
        else:
            hit_scores.append(top_score)
            verdict = "HIT" if rank == 1 else "MISS"
        rows.append({"id": q["id"], "type": q["type"],
                     "expected": q["expected_source"],
                     "top_source": chunks[ranked[0][1]]["source"] if ranked else None,
                     "rank": rank, "top_score": round(top_score, 4),
                     "margin": round(margin, 4), "verdict": verdict})
    scored = [r for r in rows if r["type"] != "control"]
    hits = sum(1 for r in scored if r["verdict"] == "HIT")
    mrr = statistics.mean([1.0 / r["rank"] for r in scored if r["rank"]]) \
        if scored else 0.0
    return {
        "accuracy": round(hits / len(scored), 3) if scored else 0.0,
        "hits": hits, "scored": len(scored), "mrr": round(mrr, 3),
        "avg_margin": round(statistics.mean(
            [r["margin"] for r in scored]) if scored else 0, 4),
        "mean_hit_score": round(statistics.mean(hit_scores), 4) if hit_scores else 0.0,
        "mean_control_score": round(statistics.mean(ctrl_scores), 4) if ctrl_scores else 0.0,
        "hit_control_gap": round(
            (statistics.mean(hit_scores) - statistics.mean(ctrl_scores))
            if hit_scores and ctrl_scores else 0, 4),
        "mean_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "rows": rows,
    }


def print_table(results, chunking_label, n_chunks):
    print(f"\n== sample eval | chunking {chunking_label} ({n_chunks} chunks) ==")
    hdr = f"{'retriever':<10} {'acc@1':>6} {'MRR':>6} {'margin':>7} {'ctrl':>7} {'gap':>7} {'ms/q':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        if r is None:
            continue
        print(f"{name:<10} {r['accuracy'] * 100:5.0f}% "
              f"{r['mrr']:6.3f} {r['avg_margin']:7.3f} "
              f"{r['mean_control_score']:7.3f} {r['hit_control_gap']:7.3f} "
              f"{r['mean_latency_ms']:7.2f}")
    print("\nctrl = mean top-1 score on unanswerable controls (lower = better "
          "calibration); gap = hit score minus control score.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunking", choices=["A", "B"], default="A")
    ap.add_argument("--top-k", type=int, default=3,
                    help="chunks shown per question in verbose mode")
    ap.add_argument("--json", default=None, help="write full results to file")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    size, overlap = CHUNKINGS[args.chunking]
    corpus_dir = os.path.join(HERE, "sample-corpus")
    chunks = load_chunks(corpus_dir, size, overlap)
    with open(os.path.join(corpus_dir, "sample_questions.json")) as f:
        questions = json.load(f)["questions"]

    print(f"sample corpus: {len(load_documents(corpus_dir))} docs, "
          f"{len(chunks)} chunks ({size}/{overlap}) | "
          f"{len([q for q in questions if q['type'] != 'control'])} scored + "
          f"{len([q for q in questions if q['type'] == 'control'])} controls")

    sbert = SbertRetriever()
    sbert_ok = getattr(sbert, "available", False)
    reranker = RerankRetriever(HybridRetriever(sbert if sbert_ok else None)) \
        if sbert_ok else None

    retrievers = [TfidfRetriever(), Bm25Retriever()]
    if sbert_ok:
        retrievers.append(sbert)
    retrievers.append(HybridRetriever(sbert if sbert_ok else None))
    if reranker is not None and getattr(reranker, "available", False):
        retrievers.append(reranker)

    results, skipped = {}, []
    if not sbert_ok:
        skipped.append("sbert: sentence-transformers not installed "
                       "(pip install sentence-transformers to enable; "
                       "no download happens otherwise)")
        skipped.append("rerank: needs the cross-encoder model — same reason")
    for r in retrievers:
        r.fit(chunks)
        results[r.name] = evaluate(r, chunks, questions)

    print_table(results, args.chunking, len(chunks))
    if skipped:
        print("\nskipped (optional, needs a model download):")
        for s in skipped:
            print(f"  - {s}")
        print("Real SBERT / rerank numbers from Gopi's evals: data/results.json")

    if args.verbose:
        for name, r in results.items():
            print(f"\n-- {name} --")
            for row in r["rows"]:
                print(f"  Q{row['id']:>2} {row['type']:<11} {row['verdict']:<7} "
                      f"rank={row['rank']} top={row['top_source']}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"chunking": args.chunking, "n_chunks": len(chunks),
                       "retrievers": {k: v for k, v in results.items()}},
                      f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
