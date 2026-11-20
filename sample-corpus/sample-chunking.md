# Chunking documents (SAMPLE)

Chunking is the step where long documents are split into small pieces before
embedding. A good chunk is one idea, not one page: keep chunks between 300 and
800 characters, with a small overlap of 50 to 150 characters so that ideas
spanning a chunk boundary are not lost.

Overlap matters because a sentence cut in half at a boundary loses its
meaning in both neighbouring chunks. With overlap, the tail of one chunk
repeats at the head of the next, so no idea is ever split across a hard edge.

Always split on word boundaries, never mid-word. Splitting mid-word creates
broken tokens that no retriever can match. Recursive splitting works well in
practice: try paragraph breaks first, then sentences, then words, and only use
a hard character cut as a last resort.
