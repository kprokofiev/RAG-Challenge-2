# RAG Retrieval Relevance TODO

- [ ] Add INN/brand to section questions (e.g., "ibuprofen") to anchor retrieval.
- [ ] Set `doc_kind_preference` per section (clinical → ctgov_*, regulatory → epar/label/grls, patents → patent_pdf).
- [ ] Add hard filter: drop candidates if chunk/doc title/source lacks drug name/aliases.
- [ ] Implement query expansion (synonyms/brands) and merge multi-query results.
