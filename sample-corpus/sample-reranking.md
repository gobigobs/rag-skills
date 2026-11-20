# Reranking and knowing when to abstain (SAMPLE)

Reranking is a second pass over retrieval results. A fast first-stage
retriever fetches the top 20 or so candidate chunks, and a cross-encoder model
then reads each question-chunk pair together and scores how well the chunk
actually answers the question. The cross-encoder is slower but much more
careful, so it is used only on a shortlist, never on the whole corpus.

Reranking also gives the best abstention signal. When every candidate scores
far below the scores of real answers, the system should say "I don't know"
instead of guessing. A trustworthy "don't answer" gate matters more than one
extra point of accuracy: a wrong answer presented confidently is worse than
no answer at all.

A practical pattern is therefore two-stage: a cheap retriever ranks, and the
cross-encoder verifies. Use its score as the abstention gate, and only let it
reorder results when your own eval proves it helps on your questions.
