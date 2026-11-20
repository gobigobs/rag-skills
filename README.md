# rag-skills — Measure first, trust later

**An agent skill pack for building and evaluating RAG, proven on a real retrieval shootout: 90 questions, 5 retrievers, 2 chunkings. Cross-encoder reranking won at 94% accuracy@1; hybrid fusion lost to plain SBERT; RRF scores carry zero abstention signal. Try it live — no model download, no API key.**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![No model download](https://img.shields.io/badge/demo-no%20model%20download-orange)
![Real evals](https://img.shields.io/badge/evals-90%20questions%20%C2%B7%205%20retrievers-gold)

> **Proof, not mockups:** screenshots below are from the real running demo, captured with headless Chromium. The top shot shows the sample question asked and answered; the full shot shows the real v3 results table underneath.
>
> ![Demo — question asked, passages retrieved](docs/screenshots/demo-top.png)
> ![Demo — full page with real results table](docs/screenshots/demo-full.png)
>
> Re-capture with `node ~/workspace/.scratch/shot/rag-skills-shot.mjs` (uses the API already running on port 8001).

## Quick start — 30 seconds, 3 steps

| Step | Do this |
|---|---|
| 1. Copy the skills | `cp -r skills/rag-build skills/rag-eval .claude/skills/` (or `~/.claude/skills/` for user-wide) |
| 2. Run the eval | `python3 scripts/eval.py --chunking A` — pure stdlib, no downloads |
| 3. Try the API | `uvicorn api.main:app --port 8000`, open `http://127.0.0.1:8000/` and ask a question |

Then ask your agent naturally: *"build a RAG pipeline for our docs"* loads `rag-build`; *"which retriever should serve this system?"* loads `rag-eval`.

## What's inside

| Piece | What it is | Why it matters |
|---|---|---|
| `skills/rag-build/SKILL.md` | Build skill: chunking → embedding → retrieval → rerank → abstention gate | The decisions the evals earned, not tutorial defaults |
| `skills/rag-eval/SKILL.md` | Eval skill: question sets, metrics, honest reading of results | Stops a 21-question headline becoming a production decision |
| `scripts/eval.py` | Runnable eval harness, pure Python stdlib | TF-IDF + BM25 run anywhere; SBERT / cross-encoder optional, never auto-downloaded |
| `api/main.py` + `demo.html` | FastAPI demo: `GET /health`, `GET /results`, `POST /ask` | The live page above — retrieval with a real "I don't know" gate |
| `data/results.json` | Real eval summaries: v2 (21q), v3 (90q), v4 corpus-v2 (57q) | Extracted programmatically from the source evals, not retyped |
| `sample-corpus/` | 4 tiny SAMPLE docs + 10 SAMPLE questions (toy data, labelled) | Something to ask the demo immediately |

Agent Skills is an open format (Anthropic, [agentskills.io](https://agentskills.io)): a folder with a `SKILL.md` (YAML frontmatter: `name` + `description`) plus supporting files. The agent loads only the description until the task matches, then reads the body. Other agents (Codex, Copilot, Cursor, Gemini CLI) read the same folders.

## The real results

### v3 — 90-question stress test (80 scored + 10 controls, 3 docs, chunking A)

| Retriever | acc@1 | MRR | Hit−control gap | ms/query |
|---|---|---|---|---|
| TF-IDF | 82% (66/80) | 0.898 | +0.07 | 0.4 |
| BM25 | 74% (59/80) | 0.855 | +3.35 | 0.1 |
| SBERT | 89% (71/80) | 0.930 | +0.24 | 31 |
| Hybrid (RRF) | 82% (66/80) | 0.897 | +0.00 | 25 |
| **Cross-encoder rerank** | **94% (75/80)** | **0.964** | **+7.32** | 616 |

Ambiguous questions are the hardest category for every retriever (38–75%) and the only one where rankings truly separate — rerank 75%, SBERT 69%, TF-IDF/hybrid 50%, BM25 38%.

### v4 — corpus-v2 (12 docs, 3,778 words, 52 scored + 5 controls)

| Retriever | Chunking A (500/100) | Chunking B (250/50) |
|---|---|---|
| TF-IDF | 81% (42/52), MRR 0.884 | 77% (40/52), MRR 0.859 |
| BM25 | 85% (44/52), MRR 0.907 | 81% (42/52), MRR 0.886 |
| SBERT | 90% (47/52), MRR 0.935 | **92% (48/52)**, MRR 0.962 |
| Hybrid (RRF) | 88% (46/52), MRR 0.934 | 88% (46/52), MRR 0.926 |
| **Rerank (CE)** | **92% (48/52)**, MRR 0.957 | 90% (47/52), MRR 0.943 |

Cross-encoder hit−control gap on corpus-v2: **+11.68** (A) / **+11.17** (B) — the strongest abstention signal measured anywhere in the project.

### v2 — the 21-question pilot (3 docs)

| Retriever | Chunking A | Chunking B |
|---|---|---|
| TF-IDF | 71% (15/21) | 76% (16/21) |
| BM25 | 81% (17/21) | 81% (17/21) |
| SBERT | 95% (20/21) | 95% (20/21) |
| Hybrid (RRF) | 90% (19/21) | **100% (21/21)** |
| Rerank (CE) | 90% (19/21) | 90% (19/21) |

### Findings that flipped — read these before citing anything

1. **"Hybrid 100%" did not generalise.** v2's perfect hybrid score was a 21-question artifact; on 80 questions it fell to 82%, worse than SBERT alone. A headline built on 21 questions is a rumour.
2. **"Cross-encoder hurts" reversed.** v2 said reranking demoted correct answers; v3 says it is the best retriever on chunking A. Findings are conditional on the question mix.
3. **"RRF scores are uncalibrated mush" held — and it is load-bearing.** Gap +0.000: RRF scores carry zero abstention signal. Never threshold them.
4. **Production pick: SBERT for ranking + cross-encoder as verifier / abstention gate.** Rerank-as-ranker wins a few points but costs 25–55× the latency; rerank-as-verifier buys the safety without paying for reordering on every query.

## Run the eval yourself

```bash
python3 scripts/eval.py --chunking A
```

Real output, sample corpus (4 docs, 10 chunks, 8 scored + 2 controls):

```
sample corpus: 4 docs, 10 chunks (500/100) | 8 scored + 2 controls

== sample eval | chunking A (10 chunks) ==
retriever   acc@1    MRR  margin    ctrl     gap    ms/q
--------------------------------------------------------
tfidf         88%  0.938   0.104   0.191   0.090    0.06
bm25          88%  0.917   2.589   2.503   4.845    0.52
hybrid        88%  0.938   0.001   0.033   0.000    0.92
```

```bash
python3 scripts/eval.py --chunking B --verbose     # small chunks, per-question rows
python3 scripts/eval.py --json /tmp/sample_eval.json
```

## Live demo + API

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn api.main:app --port 8000
# open http://127.0.0.1:8000/ — ask the sample corpus, see the real results table
```

| Endpoint | What it does |
|---|---|
| `GET /` | Live demo page: ask a question, see retrieved passages |
| `GET /health` | Liveness check |
| `GET /results` | All real eval results (v2 / v3 / v4 corpus-v2) |
| `GET /results/{v2\|v3\|v4_corpus_v2}` | One eval run |
| `POST /ask` | Retrieval demo over the sample corpus (TF-IDF + BM25, RRF-fused). Returns top chunks and says **"I don't know"** when nothing matches confidently |

```bash
$ curl -s -X POST localhost:8000/ask -d '{"question":"What does RAG stand for?","top_k":1}' -H 'Content-Type: application/json'
{"question":"What does RAG stand for?","answer":"Most relevant passage is from
sample-rag-basics.md (score 0.0328)...","abstained":false,"gate_score":0.5,...}

$ curl -s -X POST localhost:8000/ask -d '{"question":"What is the population of Chennai?","top_k":1}' -H 'Content-Type: application/json'
{"question":"What is the population of Chennai?","answer":"I don't know — nothing in
the sample corpus answers this question confidently. (Only 0% of the question's
content words appear in the best passage...)","abstained":true,"gate_score":0.0,...}
```

The abstention gate is itself the lesson: RRF scores cannot gate (measured hit−control gap ≈ 0.000), so the demo gates on content-word overlap with the top chunk instead. See `rag-build` §5.

## Limitations (read before citing)

- Small corpora (3 docs, then 12); rankings may shuffle on a real knowledge base.
- Questions hand-written by someone who knows the docs — possible bias.
- "Multi-hop" questions are single-doc; true cross-document multi-hop needs a bigger corpus and multi-source scoring (not implemented).
- Retrieval only — no answer-quality (LLM-judge) eval. The small-model local answer-level trial was noisy and is deliberately **not** published here; its scores appear nowhere in this repo.

## The honest money angle

No income is promised — this is a portfolio and lead-generation asset, not a product with revenue. The realistic paths: **jobs** (the repo is evidence for Senior AI / RAG roles — measured retrieval work, not a tutorial copy); **freelance RAG audits** (teams shipping chatbots over their docs pay for "which retriever deserves to serve this, and when should it say I don't know?" — this repo is the demo, the skills are the delivery playbook); **workshops / content** (a finding that flipped, a perfect score that didn't generalise — teachable material). If none convert, the repo still costs nothing to keep and keeps working as proof of work.

## Knowledge used

- **Agent Skills format** — Anthropic's open standard ([agentskills.io](https://agentskills.io), Dec 2025): folder + `SKILL.md`, YAML frontmatter (`name` ≤ 64 chars, lowercase/hyphens; `description` ≤ 1,024 chars), progressive disclosure, body under ~5,000 tokens.
- **Reciprocal Rank Fusion** — Cormack, Clarke & Büttcher, SIGIR 2009: `RRF(d) = Σ 1/(k + rank(d))`, k = 60; rank-only fusion, the default in Elasticsearch / OpenSearch / Qdrant.
- **BM25** — Robertson et al. (Okapi, k1 = 1.5, b = 0.75); **TF-IDF** cosine baseline; **dense retrieval** with sentence embeddings (`all-MiniLM-L6-v2`); **cross-encoder reranking** (MS MARCO `MiniLM-L-6-v2`) as a pairwise judge.
- **Eval practice** — accuracy@1, MRR, margin, abstention calibration via unanswerable controls (hit−control gap), per-category breakdowns; RAGAS/TruLens-style answer-level eval noted as future work.
- **Learned fusion** — Bruch, Gai & Ingber (2022): a tuned convex combination of normalised scores beats fixed RRF once labelled queries exist — the reason this repo does not recommend RRF blindly.
- **The real evals** — `rag-doc-qa`: `eval_v2/v3/v4.py`, `eval_results_v2/v3/v4.json`, `REPORT.md`. Every number here was extracted programmatically from those JSON files, not retyped.

Built by [Gopi](https://www.linkedin.com/in/rgobinath) (Senior AI Product Engineer, 11+ years).

## License

MIT — see [LICENSE](LICENSE).
