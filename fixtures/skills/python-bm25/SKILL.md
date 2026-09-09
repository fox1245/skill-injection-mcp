---
id: python-bm25
name: Python BM25 Sparse Retriever
description: Implement BM25 and SQLite FTS5 sparse full-text search in Python for keyword retrieval.
tags: [bm25, fts5, sparse, python, retrieval]
---

# Python BM25 Sparse Retriever

Use this skill when you need **sparse** lexical retrieval with BM25 or FTS5.

## Capabilities
- Build SQLite FTS5 indexes with porter tokenization
- Rank documents with BM25
- Unique diagnostic token: FTS5PorterTokenXyz
- Python-only sparse path — does **not** provide dense embeddings or vector search

## When not to use
- Dense semantic similarity
- Hybrid RRF fusion end-to-end (needs a dense partner skill)
