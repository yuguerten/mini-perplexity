# Non-CLI Pipeline (Current)

This document describes the current API-driven research pipeline (`/research`) and excludes the CLI flow.

## 1. Request Entry
- Endpoint: `POST /research`
- Input model: `ResearchRequest`
- Options: `summarize`, `store_for_rag`, `export_pdf`
- Background task starts `run_research(query_id, query, options)`.

## 2. Paper Retrieval and Ranking (`ResearchService.search_academic_papers`)

### 2.1 Query Expansion
- Generates up to 3 close subqueries from the user query (with fallback variants).

### 2.2 Multi-source Retrieval
- For each subquery, fetches candidates from:
  - Semantic Scholar
  - arXiv
  - PubMed
- Retrieval depth is controlled by `SEARCH_PER_SUBQUERY_LIMIT`.

### 2.3 Candidate Cleanup
- Deduplicate by stable paper key.
- Prefer duplicate entries with richer metadata (PDF/abstract).
- Enrich arXiv citations via Semantic Scholar where possible.

### 2.4 Filters
- Recency filter keeps papers within `RECENT_PAPER_YEARS`.
- Topical lexical filter enforces minimum query-term/anchor-term overlap.

### 2.5 Two-stage Ranking
- Stage 1 (embedding rank):
  - Multi-query scoring (original query + subqueries)
  - Field-aware similarity (`title` and `abstract` weighted)
  - Combines semantic score with citation/pdf/lexical/recency factors
- Stage 2 (reranker):
  - Optional reranking over top candidates using `ollama.rerank`
  - Controlled by `TWO_STAGE_RANKING`, `RERANK_MODEL`, and candidate limits
- Final selected count is controlled by `FINAL_PAPER_LIMIT`.

## 3. Content Acquisition Per Selected Paper
For each selected paper:
- Download PDF if available.
- Extract text with Docling (page-limited and CPU-safe options available).
- Optionally generate concise per-paper summary (`summarize=true`).
- Persist progressive state to `data/queries/{query_id}.json`.

## 4. Multi-pass Detailed Report Generation (Internal)
When `summarize=true`, the pipeline now prefers a multi-pass report flow:

### 4.1 Evidence Card Extraction (Internal Structured Layer)
- Build one evidence card per paper from metadata, abstract, summary, and full-text excerpt.
- Card contains:
  - problem
  - method
  - setting/data
  - key findings
  - strengths
  - limitations
  - open questions
  - relevance to query
  - confidence
- This structure is internal and not user-facing.

### 4.2 Outline Planning
- Build report outline with fixed section titles and map each section to relevant paper indices.
- Falls back to deterministic mapping if planning fails.

### 4.3 Section-by-section Writing
- Generate each section independently from only its mapped evidence cards.
- Enforce citation bounds to allowed paper indices for that section.

### 4.4 Verification and Repair
- Validate sections for citation validity/coverage.
- Regenerate flagged sections once with targeted revision instructions.
- Clean invalid citations.

### 4.5 Final Synthesis
- Generate a final **Bottom Line** paragraph (current field status + confidence labels).
- Assemble full markdown report text with required sections.

### 4.6 Fallback
- If multi-pass generation fails or returns empty content, fallback to legacy single-pass combined summary generation.

## 5. PDF Export (`ExportService.generate_pdf`)
If `export_pdf=true`:
- Build PDF using PyLaTeX.
- Include:
  - Title/query/date
  - Summary section (the generated detailed report markdown)
  - Optional extracted artifacts (figures/tables/equations) if enabled
  - References
- Compile with `pdflatex`.
- Save to `data/exports/{query_id}.pdf`.

## 6. Optional RAG Storage
If `store_for_rag=true`:
- After pipeline completion, paper full text is chunked by section and embedded.
- Chunks are stored in Chroma for `/rag/query` retrieval.

## 7. State Persistence
- Pipeline state is saved in `data/queries/{query_id}.json` with:
  - status
  - selected papers
  - summary text
  - pdf path
  - errors/timestamps
