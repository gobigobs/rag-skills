---
name: rag-eval
description: Evaluate a RAG retriever honestly — build a discriminating question set, score accuracy@1 / MRR / margin / abstention calibration / latency per method, read per-category results, and spot when a headline number is an artifact. Use when measuring, comparing, or reporting retrieval quality.
---

# rag-eval — measure retrieval before you believe it

A retrieval eval answers one question: **which retriever deserves to serve
this system?** Everything below is how Gopi's rag-doc-qa evals actually did
it — including the times the headline finding flipped.

## 1. Build the question set (this is the whole game)

- **Categories, not a pile.** Use 5 types, ~16 each at scale:
  - *direct* — keywords overlap the answer ("What does RAG stand for?")
  - *paraphrase* — same fact, near-zero keyword overlap
  - *multi-hop* — needs facts from several chunks (incl. cross-doc)
  - *adversarial-negation* — a negation/reasoning trap in the wording
  - *ambiguous* — vague ("How do I start?", "Where does it run?")
- **Controls (unanswerable questions).** 10–20% of the set. Questions whose
  answer is not in the corpus ("What is the population of Chennai?").
  Without them you cannot calibrate abstention — and abstention is half
  the product.
- **Flag hard questions.** Mark the ones you expect to break retrievers;
  they are where methods actually separate.
- **Write questions before you look at rankings.** Questions written by
  someone who knows the docs bias toward/against specific retrievers.
  Say so in the limitations.
- **Corpus size caps question diversity.** 90 questions were squeezed out
  of ~2,500 words in v3 and the strain showed: TF-IDF hit 100% on
  paraphrase (not enough lexical confusability left) and every retriever
  hit 100% on adversarial (the trap was answer-time, not retrieval-time).
  When categories saturate, grow the corpus — not the question count.

Real sets: v2 = 21 scored + 3 controls (3 docs); v3 = 80 scored + 10
controls (same 3 docs); v4 corpus-v2 = 52 scored + 5 controls (12 docs,
3,778 words). Full summaries in `data/results.json`.

## 2. Metrics — five numbers, no fewer

| Metric | Definition | What it catches |
|---|---|---|
| **accuracy@1** | top chunk's source doc == expected doc | The headline. Report as hits/scored, never a bare %. |
| **MRR** | mean reciprocal rank of first correct-source chunk | Rank-2/3 near-misses accuracy@1 hides. |
| **avg margin** | mean (top1 − top2 score) on hits, native scale | Confidence. Never compare margins across retrievers — scales differ. |
| **control top-score + hit−control gap** | mean top-1 score on controls; hits minus controls | Abstention calibration. The gap is the only number that says whether a threshold can exist. |
| **latency** | mean + p50 ms/query (indexing excluded) | Whether the accuracy is affordable. |

Run every retriever × every chunking (here: 5 × 2). Report per-category
accuracy too — the aggregate hides the story (see §4).

## 3. How to run it

```bash
python3 scripts/eval.py --chunking A          # sample corpus, no downloads
python3 scripts/eval.py --chunking B --verbose
python3 scripts/eval.py --json /tmp/sample_eval.json
```

TF-IDF and BM25 run in pure Python; SBERT and cross-encoder rerank are
optional and skip cleanly without `sentence-transformers`. The real
5-method numbers are served by the API (`GET /results`) and stored in
`data/results.json`.

## 4. Worked example — reading the v3 results (90 questions, chunking A)

| Retriever | acc@1 | MRR | Hit−control gap | ms/query |
|---|---|---|---|---|
| TF-IDF | 82% (66/80) | 0.898 | +0.07 | 0.4 |
| BM25 | 74% (59/80) | 0.855 | +3.35 | 0.1 |
| SBERT | 89% (71/80) | 0.930 | +0.24 | 31 |
| Hybrid (RRF) | 82% (66/80) | 0.897 | +0.00 | 25 |
| Rerank (CE) | **94% (75/80)** | **0.964** | **+7.32** | 616 |

Per-category accuracy (chunking A) is where the insight lives:

| Retriever | direct | paraphrase | multi-hop | adversarial | ambiguous |
|---|---|---|---|---|---|
| TF-IDF | 94% | 100% | 69% | 100% | 50% |
| BM25 | 94% | 75% | 62% | 100% | 38% |
| SBERT | 100% | 88% | 88% | 100% | 69% |
| Hybrid | 100% | 88% | 75% | 100% | 50% |
| Rerank | 100% | 100% | 94% | 100% | **75%** |

Read it like this:

1. **Ambiguous is the hardest category for every retriever (38–75%)**
   and the only one where rankings truly separate. Vague queries have
   almost no lexical signal, so term-matching guesses — and RRF fusion
   *amplifies* the guess into the wrong document.
2. **"Hybrid 100%" did not generalise.** v2's perfect hybrid score was a
   21-question artifact. On 80 questions hybrid fell to 82%, worse than
   SBERT alone. A headline built on 21 questions is a rumour.
3. **"Cross-encoder hurts" reversed.** v2 said reranking demoted correct
   answers; v3 says it is the best retriever on chunking A. The finding
   flipped because the question mix changed — paraphrase-hard in v2,
   ambiguous/multi-hop in v3. Findings are conditional on the eval set.
4. **"RRF scores are uncalibrated mush" held — and it is load-bearing.**
   Gap +0.000: RRF scores carry zero abstention signal. Never threshold them.
5. **One question beat everything.** Q74 ("What do I type in the terminal
   to begin?") defeated every retriever on both chunkings. Keep a list of
   undefeated questions — they are your next corpus gap, not noise.

## 5. How to report results honestly

- Always give **hits/scored** next to the % (66/80, not "82%").
- State corpus size, question count, and category mix. A % without them
  is meaningless.
- Name the controls and the gap. If you cannot abstain, say so.
- List limitations before conclusions: tiny corpus, hand-written
  questions, single-doc multi-hop, retrieval-only (no answer-quality
  eval yet — that needs an LLM judge, and a small local judge is noisy;
  the local trial scores are deliberately not published here).
- When a finding flips between eval rounds, report the flip. It is the
  most valuable line in the report.

## Checklist

- [ ] Question set covers all 5 categories + controls (10–20%)
- [ ] Every retriever × every chunking run; per-category table produced
- [ ] accuracy@1 reported as hits/scored; MRR, margin, control gap, latency all present
- [ ] Headline checked against a larger question set before claiming it
- [ ] Undefeated questions listed, not hidden
- [ ] Limitations section written before conclusions
- [ ] Answer-level eval flagged as separate work (retrieval ≠ answer quality)

## References (in-repo)

- `data/results.json` — v2 / v3 / v4-corpus-v2 summaries (the source of every number above)
- `scripts/eval.py` — runnable harness with the same metrics
- `skills/rag-build/SKILL.md` — what to build once the eval points somewhere
