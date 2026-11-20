# RAG basics (SAMPLE)

RAG stands for Retrieval-Augmented Generation. It is a technique that gives a
language model access to external documents at answer time, instead of relying
only on what the model memorized during training.

The pipeline has five steps. First, ingest: documents are split into small
chunks. Second, embed: each chunk is converted into a vector embedding, a list
of numbers that captures its meaning. Third, store: the vectors go into a
vector database for fast similarity search. Fourth, retrieve: when a user asks
a question, the question is embedded the same way and the most similar chunks
are fetched. Fifth, generate: the retrieved chunks are passed to the language
model as context, and the model writes an answer grounded in that context.

RAG matters because answers stay up to date without retraining the model, the
model can cite its sources, and private documents like company wikis and
manuals become searchable by chat.
