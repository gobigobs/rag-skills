# Embeddings and vector search (SAMPLE)

An embedding turns a piece of text into a list of numbers, a vector, that
captures what the text means. Texts with similar meanings end up with similar
vectors, so meaning can be compared with arithmetic: cosine similarity measures
the angle between two vectors, and a small angle means a close match.

A vector database stores these embeddings and finds the nearest neighbours of
a query vector in milliseconds, even across millions of chunks. FAISS is a
popular open-source library for this kind of fast similarity search, and
managed vector databases offer the same idea as a service.

Embeddings are what let retrieval understand paraphrases. A question asked in
completely different words can still match the right chunk, because the vectors
are close even when no keyword overlaps.
