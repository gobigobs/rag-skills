---
name: rag-build
description: Build a retrieval-augmented generation (RAG) pipeline — chunking, embeddings, vector search, hybrid retrieval with reciprocal rank fusion, cross-encoder reranking, and an abstention gate. Use when building, fixing, or reviewing any RAG system, choosing a retriever, or deciding chunk size.
---

# rag-build — build a RAG pipeline that earns its keep

Build in this order: **chunk → embed → store → retrieve → rerank → gate**.
Never start with the LLM. Retrieval quality decides everything downstream;
a good generator cannot rescue bad chunks.

## 1. Chunking — one idea per chunk

- Target **300–800 characters** per chunk. Gopi's evals tested 500/100
  (chunking A) against 250/50 (chunking B) — smaller chunks helped TF-IDF
  (+5pp on the 21-question eval) and hybrid (+10pp, to a perfect 21/21),
  while SBERT and BM25 were unchanged. Direction matches the literature;
  magnitude on 3 docs may not transfer.
- Overlap **50–150 chars** so ideas spanning a boundary survive.
- Split on word boundaries, never mid-word. Recursive splitting: paragraphs
  → sentences → words → hard cut as a last resort.
- Keep a `source` field on every chunk. All evals here score by source doc,
  so a chunk without provenance is unmeasurable.

## 2. Embeddings — meaning as vectors

- Start with a small sentence-embedding model
  (`all-MiniLM-L6-v2`, 384 dims, ~80 MB). It is fast on CPU (~20–45 ms/query
  in the real evals) and good enough to beat every lexical method.
- Normalise embeddings and use cosine similarity. Store them in a vector
  index (FAISS for local work; pgvector/Qdrant/Milvus when you need a service).
- Re-embed when the chunking changes. Stale embeddings from old chunks are a
  silent correctness bug.

## 3. Retrieval — start lexical, add semantic, fuse with care

Run all three baselines before you believe any of them:

| Retriever | What it is | When it wins |
|---|---|---|
| TF-IDF | Term frequency × inverse doc frequency, cosine | Baseline only. Dies on paraphrases with zero keyword overlap (71–82% in the real evals). |
| BM25 | TF-IDF with term saturation + length normalisation (k1=1.5, b=0.75) | Best latency/accuracy trade for keyword-heavy traffic: 74–85% at ~0.1 ms/query. |
| SBERT (dense) | Embedding cosine similarity | The balanced winner: 89–95% across evals. Gets every hard paraphrase lexical methods miss. |

**Hybrid (RRF):** fuse BM25 + SBERT ranked lists with reciprocal rank
fusion — `RRF(d) = Σ 1 / (k + rank(d))`, k = 60 (Cormack, Clarke &
Büttcher, SIGIR 2009). Ranks only, never raw scores, because the two
signals live on incomparable scales.

> ⚠️ Fusion is not free magic. On the 21-question eval, hybrid on small
> chunks hit 100% — then dropped to 82% (A) / 90% (B) on the 90-question
> stress test, *worse than SBERT alone* on both chunkings. RRF dragged
> SBERT down: 10 of hybrid's misses were cases where BM25 noise pulled the
> wrong document to the top. Measure fusion on your own questions; if it
> loses, ship SBERT alone. A learned weighted combination (Bruch, Gai &
> Ingber 2022) beats fixed RRF once you have labelled queries.

## 4. Reranking — a careful second pass, not a first pass

- Retrieve top-20 with the cheap retriever, then score each
  (question, chunk) pair with a cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`). It reads the pair together,
  so it judges semantics pairwise instead of counting terms.
- Cost is real: ~0.6–2.5 s/query on CPU in the real evals, 25–55× hybrid.
- The reversal that matters: on the 21-question eval reranking *hurt*
  (90% vs 95% SBERT — it demoted correct answers on paraphrase-hard
  questions, over-trusting lexical overlap). On the 90-question eval it
  was the **best retriever on chunking A at 94%**, because ambiguous and
  multi-hop questions are exactly where pairwise judgement earns its keep.
  It fixed 10 of hybrid's misses almost entirely in the ambiguous category.
- Rule: let the cross-encoder rerank only when your eval proves it helps
  on *your* question mix — especially vague queries. Never assume.

## 5. Abstention — the "I don't know" gate

- **RRF scores cannot gate.** Measured hit−control gap ≈ 0.000–0.001:
  RRF scores are rank-only mush with no calibration. Do not threshold them.
- **Cross-encoder logits are the best gate measured**: controls at −9.6 vs
  hits, gap +7.3 (90-question eval) and +11.7 (corpus-v2). Run the
  cross-encoder as a *verifier* even when it does not rerank: if its top
  score is far below typical hit scores, abstain.
- BM25's gap (+3.4 to +8.8) is usable in a pinch; TF-IDF's (+0.07–0.09)
  and SBERT's (+0.24–0.32) are weak.
- Ship the gate from day one. A confident wrong answer is worse than none.

## Worked example — the decision this skill produces

Corpus-v2 eval (12 docs, 3,778 words, 52 scored questions, chunking A):

| Retriever | acc@1 | MRR | Latency |
|---|---|---|---|
| TF-IDF | 81% (42/52) | 0.884 | 0.6 ms |
| BM25 | 85% (44/52) | 0.907 | 0.2 ms |
| SBERT | 90% (47/52) | 0.935 | 45 ms |
| Hybrid (RRF) | 88% (46/52) | 0.934 | 46 ms |
| Rerank (CE) | **92% (48/52)** | **0.957** | 2,472 ms |

Production pick from these numbers: **SBERT for ranking + cross-encoder
as verifier/abstention gate**. Rerank-as-ranker wins 2pp here but costs
55× the latency; rerank-as-verifier buys the abstention safety without
paying for reordering on every query. Hybrid loses to SBERT alone, so it
stays out until learned weights are tuned on a held-out split.

## Checklist

- [ ] Chunks are 300–800 chars, word-boundary splits, overlap 50–150, every chunk has a `source`
- [ ] TF-IDF, BM25 and SBERT baselines all measured before any fusion
- [ ] Hybrid (RRF, k=60) measured on your own questions — shipped only if it beats SBERT alone
- [ ] Cross-encoder evaluated as reranker *and* as verifier; latency budget written down
- [ ] Abstention gate exists and is calibrated on unanswerable controls (never on RRF scores)
- [ ] Embeddings re-built whenever chunking changes
- [ ] Latency measured per query (mean + p50), not guessed

## References (in-repo)

- `data/results.json` — the real numbers behind every claim above
- `scripts/eval.py` — runnable harness (TF-IDF/BM25/hybrid, no downloads)
- `api/main.py` — live demo; its abstention gate is the worked example of §5
- `skills/rag-eval/SKILL.md` — how to measure all of this
