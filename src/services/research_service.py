import asyncio
import json
import os
import re
import uuid
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Tuple
import aiohttp

from ..api.models import (
    PaperMetadata,
    PaperContent,
    ResearchOptions,
    ResearchStatus,
)

class ResearchService:
    def __init__(self, storage_dir: str = "data"):
        self.storage_dir = Path(storage_dir)
        self.papers_dir = self.storage_dir / "papers"
        self.queries_dir = self.storage_dir / "queries"
        self.summaries_dir = self.storage_dir / "summaries"
        self.debug_citations = os.getenv("DEBUG_CITATIONS", "false").lower() in {"1", "true", "yes", "on"}
        self.debug_query_expansion = os.getenv("DEBUG_QUERY_EXPANSION", "false").lower() in {"1", "true", "yes", "on"}
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "embeddinggemma")
        self.keyword_model = os.getenv("KEYWORD_MODEL", "gpt-oss:20b")
        self.embedding_max_chars = int(os.getenv("EMBEDDING_MAX_CHARS", "2000"))
        self.keyword_min_hits = int(os.getenv("KEYWORD_MIN_HITS", "1"))
        self.recent_paper_years = int(os.getenv("RECENT_PAPER_YEARS", "2"))
        self.search_per_subquery_limit = int(os.getenv("SEARCH_PER_SUBQUERY_LIMIT", "30"))
        self.final_paper_limit = int(os.getenv("FINAL_PAPER_LIMIT", "8"))
        self.two_stage_ranking = os.getenv("TWO_STAGE_RANKING", "true").lower() in {"1", "true", "yes", "on"}
        self.stage1_candidate_factor = int(os.getenv("STAGE1_CANDIDATE_FACTOR", "6"))
        self.stage2_candidate_limit = int(os.getenv("STAGE2_CANDIDATE_LIMIT", "30"))
        self.stage1_title_weight = float(os.getenv("STAGE1_TITLE_WEIGHT", "0.65"))
        self.stage1_abstract_weight = float(os.getenv("STAGE1_ABSTRACT_WEIGHT", "0.35"))
        self.stage1_best_query_weight = float(os.getenv("STAGE1_BEST_QUERY_WEIGHT", "0.75"))
        self.stage1_lexical_weight = float(os.getenv("STAGE1_LEXICAL_WEIGHT", "0.6"))
        self.stage1_recency_weight = float(os.getenv("STAGE1_RECENCY_WEIGHT", "0.6"))
        self.rerank_model = os.getenv("RERANK_MODEL", "bge-reranker-v2-m3")
        self.rerank_doc_max_chars = int(os.getenv("RERANK_DOC_MAX_CHARS", "3000"))
        self.report_min_words = int(os.getenv("REPORT_MIN_WORDS", "900"))
        self.report_max_words = int(os.getenv("REPORT_MAX_WORDS", "1400"))
        self.report_per_paper_chars = int(os.getenv("REPORT_PER_PAPER_CHARS", "2200"))
        self.report_multi_stage = os.getenv("REPORT_MULTI_STAGE", "true").lower() in {"1", "true", "yes", "on"}
        self.report_evidence_max_chars = int(os.getenv("REPORT_EVIDENCE_MAX_CHARS", "4500"))
        self.report_section_target_words = int(os.getenv("REPORT_SECTION_TARGET_WORDS", "150"))
        self.report_verification_enabled = os.getenv("REPORT_VERIFICATION_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        self.report_structured_model = os.getenv("REPORT_STRUCTURED_MODEL", "gpt-oss:20b")
        self.report_writer_model = os.getenv("REPORT_WRITER_MODEL", "gpt-oss:20b")
        self.report_verifier_model = os.getenv("REPORT_VERIFIER_MODEL", "gpt-oss:20b")
        self.docling_force_cpu = os.getenv("DOCLING_FORCE_CPU", "true").lower() in {"1", "true", "yes", "on"}
        self.docling_page_log = os.getenv("DOCLING_PAGE_LOG", "true").lower() in {"1", "true", "yes", "on"}
        self.docling_max_pages = int(os.getenv("DOCLING_MAX_PAGES", "4"))
        self.search_per_subquery_limit = max(10, self.search_per_subquery_limit)
        self.final_paper_limit = max(1, self.final_paper_limit)
        self.stage1_candidate_factor = max(1, self.stage1_candidate_factor)
        self.stage2_candidate_limit = max(1, self.stage2_candidate_limit)
        self.rerank_doc_max_chars = max(200, self.rerank_doc_max_chars)
        self.report_min_words = max(300, self.report_min_words)
        self.report_max_words = max(self.report_min_words, self.report_max_words)
        self.report_per_paper_chars = max(600, self.report_per_paper_chars)
        self.report_evidence_max_chars = max(1000, self.report_evidence_max_chars)
        self.report_section_target_words = max(80, self.report_section_target_words)
        total_field_weight = self.stage1_title_weight + self.stage1_abstract_weight
        if total_field_weight <= 0:
            self.stage1_title_weight = 0.65
            self.stage1_abstract_weight = 0.35
        else:
            self.stage1_title_weight /= total_field_weight
            self.stage1_abstract_weight /= total_field_weight
        self.stage1_best_query_weight = min(max(self.stage1_best_query_weight, 0.0), 1.0)

    def _normalize_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value).encode("utf-8", errors="replace").decode("utf-8")

    def _postprocess_summary_text(self, text: str, paper_count: int = 0) -> str:
        cleaned = self._normalize_text(text or "")
        cleaned = cleaned.replace("—", "-").replace("–", "-")

        lines = cleaned.splitlines()
        normalized_lines: List[str] = []
        for raw in lines:
            line = raw.strip()
            if not line:
                normalized_lines.append("")
                continue
            if re.fullmatch(r"\*\*.+\*\*", line):
                normalized_lines.append(line)
                continue
            if line.startswith(("#", "- ", "* ", r"\[", "$$", "|")):
                normalized_lines.append(line)
                continue
            if re.match(r"^\d+\.\s+", line):
                normalized_lines.append(line)
                continue
            if (
                len(line) <= 72
                and len(line.split()) <= 9
                and not re.search(r"[.!?]$", line)
                and line == line.title()
            ):
                normalized_lines.append(f"**{line}**")
                continue
            normalized_lines.append(line)

        cleaned = "\n".join(normalized_lines).strip()

        if paper_count > 0 and not re.search(r"\[\d+\]", cleaned):
            refs = ", ".join(f"[{i}]" for i in range(1, min(paper_count, 6) + 1))
            cleaned = f"{cleaned}\n\n**Citations:** {refs}"

        return cleaned

        for d in [self.papers_dir, self.queries_dir, self.summaries_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _debug_selected_citations(self, query: str, papers: List[PaperMetadata], selected_limit: int):
        if not self.debug_citations:
            return
        selected = papers[:selected_limit]
        total_citations = sum(int(p.citations or 0) for p in selected)
        print(
            f"[CITATION_DEBUG] query={query!r} selected={len(selected)} total_citations={total_citations}"
        )
        for rank, paper in enumerate(selected, 1):
            print(
                f"[CITATION_DEBUG] rank={rank} paper_id={paper.paper_id} "
                f"source={paper.source} citations={paper.citations or 0} title={paper.title!r}"
            )

    def _debug_query(self, message: str):
        if self.debug_query_expansion:
            print(f"[QUERY_DEBUG] {message}")

    def _report_section_specs(self) -> List[Tuple[str, str]]:
        return [
            ("Topic Overview", "Define the topic and why it matters for the query."),
            ("Core Mechanisms", "Explain core technical mechanisms and intuition."),
            ("Evolution Of The Field", "Summarize how approaches evolved over time."),
            ("Current State Of The Art", "Describe what is strongest right now based on evidence."),
            ("Comparative Analysis Of Approaches", "Compare major approaches with concrete tradeoffs."),
            ("Practical Guidance", "Give actionable guidance for practitioners."),
            ("Limitations And Open Problems", "Detail unresolved limitations and blockers."),
            ("Future Directions", "Project credible next research directions."),
        ]

    def _strip_json_fence(self, text: str) -> str:
        content = (text or "").strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()

    def _safe_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            parsed = json.loads(self._strip_json_fence(text))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
        return None

    def _normalize_index_list(self, values: Any, valid_indices: Set[int]) -> List[int]:
        if not isinstance(values, list):
            return []
        normalized: List[int] = []
        for raw in values:
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value in valid_indices and value not in normalized:
                normalized.append(value)
        return normalized

    def _citation_indices_in_text(self, text: str) -> Set[int]:
        return {int(x) for x in re.findall(r"\[(\d+)\]", text or "")}

    def _clean_invalid_citations(self, text: str, valid_indices: Set[int]) -> str:
        if not text:
            return ""

        def repl(match: re.Match[str]) -> str:
            try:
                idx = int(match.group(1))
            except (TypeError, ValueError):
                return ""
            return f"[{idx}]" if idx in valid_indices else ""

        cleaned = re.sub(r"\[(\d+)\]", repl, text)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip()

    async def _generate_subqueries(self, query: str) -> List[str]:
        """
        Generate exactly 3 close subqueries for deeper search.
        Returns deterministic fallbacks if LLM generation fails.
        """
        base_query = query.strip()
        if not base_query:
            return []

        subqueries: List[str] = []
        try:
            import ollama
            system_prompt = (
                "You generate search queries for academic paper APIs (Semantic Scholar, arXiv, PubMed). "
                "Given a user query, produce exactly 3 compact keyword-style subqueries for the same technical topic. "
                "Keep core domain terms from the original query. "
                "Do not introduce unrelated domains or broad meta wording. "
                "Avoid generic terms such as evolution, improvements, changed, developments, history. "
                'Return valid JSON only: {"subqueries":["q1","q2","q3"]}'
            )
            user_prompt = f"User query: {base_query}"
            response = ollama.chat(
                model="gpt-oss:20b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.get("message", {}).get("content", "").strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            parsed = json.loads(content.strip())
            for q in parsed.get("subqueries", []):
                q = str(q).strip()
                if q and q.lower() not in {x.lower() for x in subqueries}:
                    subqueries.append(q)
                if len(subqueries) >= 3:
                    break
        except Exception as e:
            self._debug_query(f"LLM subquery generation failed: {type(e).__name__}: {e!r}")

        fallback = [
            base_query,
            f"{base_query} 2025",
            f"{base_query} 2026",
        ]
        for q in fallback:
            if len(subqueries) >= 3:
                break
            if q.lower() not in {x.lower() for x in subqueries}:
                subqueries.append(q)

        if base_query.lower() not in {x.lower() for x in subqueries}:
            subqueries.insert(0, base_query)

        subqueries = subqueries[:3]
        self._debug_query(f"Generated subqueries={subqueries}")
        return subqueries

    def _paper_key(self, paper: PaperMetadata) -> str:
        if paper.paper_id:
            return paper.paper_id.strip().lower()
        title = re.sub(r"\s+", " ", (paper.title or "").strip().lower())
        return f"{paper.source}:{title}"

    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]{3,}", (text or "").lower()))

    def _paper_text(self, paper: PaperMetadata) -> str:
        title = (paper.title or "").strip()
        abstract = (paper.abstract or "").strip()
        if abstract:
            return f"{title}\n{abstract}"
        return title

    def _paper_title_text(self, paper: PaperMetadata) -> str:
        return (paper.title or "").strip()

    def _paper_abstract_text(self, paper: PaperMetadata) -> str:
        return (paper.abstract or "").strip()

    def _paper_rerank_text(self, paper: PaperMetadata) -> str:
        title = self._paper_title_text(paper)
        abstract = self._paper_abstract_text(paper)
        if abstract:
            return f"Title: {title}\nAbstract: {abstract}"
        return f"Title: {title}"

    def _query_variants_for_ranking(self, query: str, subqueries: List[str]) -> List[str]:
        variants: List[str] = []
        for item in [query] + (subqueries or []):
            q = (item or "").strip()
            if not q:
                continue
            if q.lower() in {v.lower() for v in variants}:
                continue
            variants.append(q)
        return variants

    def _extract_year(self, published_date: Optional[str]) -> Optional[int]:
        if not published_date:
            return None
        text = str(published_date).strip()
        match = re.search(r"(19|20)\d{2}", text)
        if not match:
            return None
        try:
            return int(match.group(0))
        except ValueError:
            return None

    def _normalize_query_token(self, token: str) -> str:
        t = re.sub(r"[^a-z0-9]", "", (token or "").lower())
        if not t:
            return ""
        yolo_match = re.fullmatch(r"yolov?(\d+)", t)
        if yolo_match:
            return f"yolo{yolo_match.group(1)}"
        return t

    def _extract_query_terms(self, query: str) -> Set[str]:
        stopwords = {
            "the", "and", "for", "with", "from", "that", "this", "those", "these",
            "into", "over", "under", "between", "within", "across", "about", "paper",
            "papers", "research", "study", "analysis", "approach", "method", "methods",
            "model", "models", "architecture", "system", "systems", "what", "changed",
            "change", "now", "recent", "latest", "new", "using", "based", "after",
            "since", "need", "all", "you", "your", "our", "their", "its", "his",
            "her", "is", "are", "was", "were", "be", "been", "being", "to", "of",
            "on", "in", "at", "by", "an", "a", "as", "it", "or", "if", "how", "why",
            "when", "where", "which",
        }
        tokens = re.findall(r"[a-z0-9][a-z0-9\-]{1,}", (query or "").lower())
        terms: Set[str] = set()
        for raw in tokens:
            norm = self._normalize_query_token(raw)
            if not norm:
                continue
            if norm in stopwords:
                continue
            if len(norm) < 3 and not any(ch.isdigit() for ch in norm):
                continue
            terms.add(norm)
        ql = (query or "").lower()
        if "attention is all you need" in ql:
            terms.update({"transformer", "selfattention", "multihead", "scaleddotproduct"})
        return terms

    def _extract_anchor_terms(self, query_terms: Set[str], query: str) -> Set[str]:
        anchors = {t for t in query_terms if any(ch.isdigit() for ch in t)}
        if anchors:
            return anchors

        weak_terms = {
            "attention", "mechanism", "mechanisms", "method", "methods", "model",
            "models", "paper", "papers", "research", "system", "systems",
        }
        high_signal = {
            "distillation", "transformer", "selfattention", "diffusion", "segmentation",
            "detection", "retrieval", "summarization", "llm", "bert", "gpt", "yolo",
            "flashattention", "linearattention", "multihead", "scaleddotproduct",
        }

        anchors = {t for t in query_terms if t in high_signal}
        if "attention is all you need" in (query or "").lower():
            anchors.update({"transformer", "selfattention", "multihead"})

        if anchors:
            return anchors

        return {t for t in query_terms if len(t) >= 7 and t not in weak_terms}

    def _paper_tokens(self, paper: PaperMetadata) -> Set[str]:
        text = self._paper_text(paper)
        raw_tokens = re.findall(r"[a-z0-9][a-z0-9\-]{1,}", (text or "").lower())
        normalized: Set[str] = set()
        for tok in raw_tokens:
            n = self._normalize_query_token(tok)
            if n:
                normalized.add(n)
        return normalized

    def _paper_term_hits(
        self, paper: PaperMetadata, query_terms: Set[str], anchor_terms: Set[str]
    ) -> Tuple[int, int]:
        if not query_terms:
            return 0, 0
        paper_tokens = self._paper_tokens(paper)
        hits = len(query_terms & paper_tokens)
        anchor_hits = len(anchor_terms & paper_tokens) if anchor_terms else 0
        return hits, anchor_hits

    def _cosine_sim(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama."""
        try:
            import ollama
            embeddings: List[List[float]] = []
            for text in texts:
                response = ollama.embeddings(
                    model=self.embedding_model,
                    prompt=text[: self.embedding_max_chars],
                )
                embeddings.append(response.get("embedding", []))
            return embeddings
        except Exception as e:
            print(f"Embedding error: {e}")
            return [[0.0] * 768 for _ in texts]

    async def _extract_keywords(self, query: str) -> List[str]:
        """Extract 5-8 key terms/phrases to guide relevance filtering."""
        base_query = query.strip()
        if not base_query:
            return []
        try:
            import ollama
            system_prompt = (
                "Extract 5-8 key terms or short phrases from the user query. "
                "Focus on the core topic and technical terms. "
                'Return valid JSON only: {"keywords":["k1","k2"]}'
            )
            user_prompt = f"Query: {base_query}"
            response = ollama.chat(
                model=self.keyword_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.get("message", {}).get("content", "").strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            parsed = json.loads(content.strip())
            keywords = []
            for k in parsed.get("keywords", []):
                k = str(k).strip()
                if k and k.lower() not in {x.lower() for x in keywords}:
                    keywords.append(k)
            return keywords[:8]
        except Exception as e:
            self._debug_query(f"Keyword extraction failed: {type(e).__name__}: {e!r}")

        stopwords = {
            "the", "and", "for", "with", "from", "that", "this", "those", "these",
            "into", "over", "under", "between", "within", "across", "about", "paper",
            "research", "study", "analysis", "approach", "method", "methods", "model",
            "models", "architecture", "system", "systems",
        }
        tokens = [t for t in self._tokenize(base_query) if t not in stopwords]
        return tokens[:8]

    def _keyword_hits(self, keywords: List[str], text: str) -> int:
        if not keywords:
            return 0
        text_l = text.lower()
        return sum(1 for k in keywords if k.lower() in text_l)

    def _score_paper(
        self,
        paper: PaperMetadata,
        query_embeddings: List[List[float]],
        title_embedding: List[float],
        abstract_embedding: List[float],
        has_abstract: bool,
        term_hits: int,
        anchor_hits: int,
        year: int,
        cutoff_year: int,
    ) -> Tuple[float, float]:
        per_query_semantic: List[float] = []
        for query_emb in query_embeddings:
            title_sim = self._cosine_sim(query_emb, title_embedding)
            if has_abstract:
                abstract_sim = self._cosine_sim(query_emb, abstract_embedding)
                semantic = (self.stage1_title_weight * title_sim) + (self.stage1_abstract_weight * abstract_sim)
            else:
                semantic = title_sim
            per_query_semantic.append(semantic)

        best_semantic = max(per_query_semantic) if per_query_semantic else 0.0
        mean_semantic = (sum(per_query_semantic) / len(per_query_semantic)) if per_query_semantic else 0.0
        combined_semantic = (
            (self.stage1_best_query_weight * best_semantic)
            + ((1.0 - self.stage1_best_query_weight) * mean_semantic)
        )

        citation_score = math.log1p(float(paper.citations or 0)) * 0.45
        pdf_bonus = 0.3 if paper.pdf_url else 0.0
        lexical_bonus = ((term_hits * 1.8) + (anchor_hits * 3.0)) * self.stage1_lexical_weight
        recency_bonus = (max(0, year - cutoff_year) * 0.05) * self.stage1_recency_weight
        final_score = (combined_semantic * 10.0) + citation_score + pdf_bonus + lexical_bonus + recency_bonus
        return final_score, best_semantic

    def _parse_rerank_scores(self, response: Any) -> Dict[int, float]:
        if not isinstance(response, dict):
            return {}
        rows = response.get("results")
        if not isinstance(rows, list):
            return {}
        parsed: Dict[int, float] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            index = row.get("index")
            if not isinstance(index, int):
                continue
            score_raw = row.get("relevance_score", row.get("score"))
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                continue
            parsed[index] = score
        return parsed

    def _rerank_candidates(
        self,
        query: str,
        candidates: List[PaperMetadata],
        stage1_scores: Dict[str, float],
    ) -> List[PaperMetadata]:
        if len(candidates) <= 1:
            return candidates

        try:
            import ollama
        except Exception as e:
            self._debug_query(f"rerank unavailable (ollama import failed): {type(e).__name__}: {e!r}")
            return []

        rerank_fn = getattr(ollama, "rerank", None)
        if not callable(rerank_fn):
            self._debug_query("rerank unavailable (ollama.rerank not present); using stage1 only")
            return []

        docs = [self._paper_rerank_text(p)[: self.rerank_doc_max_chars] for p in candidates]
        try:
            response = rerank_fn(
                model=self.rerank_model,
                query=query,
                documents=docs,
                top_n=len(docs),
            )
        except Exception as e:
            self._debug_query(f"rerank call failed: {type(e).__name__}: {e!r}")
            return []

        rerank_scores = self._parse_rerank_scores(response)
        if not rerank_scores:
            self._debug_query("rerank returned no usable scores; using stage1 only")
            return []

        indexed = list(enumerate(candidates))
        ordered = sorted(
            indexed,
            key=lambda item: (
                1 if item[0] in rerank_scores else 0,
                rerank_scores.get(item[0], float("-inf")),
                stage1_scores.get(item[1].paper_id, float("-inf")),
            ),
            reverse=True,
        )
        reranked = [paper for _, paper in ordered]
        return reranked
    
    async def search_academic_papers(
        self,
        query: str,
    ) -> List[PaperMetadata]:
        """Search by LLM-generated subqueries, then dedupe and rerank to the original query."""
        subqueries = await self._generate_subqueries(query)
        if not subqueries:
            subqueries = [query]

        all_papers: List[PaperMetadata] = []
        per_query_limit = self.search_per_subquery_limit
        final_limit = self.final_paper_limit

        for subquery in subqueries:
            tasks = [
                self._search_semantic_scholar(subquery, per_query_limit),
                self._search_arxiv(subquery, per_query_limit),
                self._search_pubmed(subquery, per_query_limit),
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            subquery_count = 0
            for result in results:
                if isinstance(result, list):
                    subquery_count += len(result)
                    all_papers.extend(result)
            self._debug_query(f"subquery={subquery!r} raw_hits={subquery_count}")

        deduped: Dict[str, PaperMetadata] = {}
        for paper in all_papers:
            key = self._paper_key(paper)
            if key not in deduped:
                deduped[key] = paper
            else:
                existing = deduped[key]
                existing_score = (1 if existing.pdf_url else 0) + (1 if existing.abstract else 0)
                current_score = (1 if paper.pdf_url else 0) + (1 if paper.abstract else 0)
                if current_score > existing_score:
                    deduped[key] = paper

        papers = list(deduped.values())
        if not papers:
            return []

        await self._enrich_arxiv_citations(papers)

        window_years = max(self.recent_paper_years, 1)
        cutoff_year = datetime.now(timezone.utc).year - window_years
        recent_papers = []
        for paper in papers:
            year = self._extract_year(paper.published_date)
            if year is None:
                continue
            if year >= cutoff_year:
                recent_papers.append(paper)
        self._debug_query(
            f"recent_filter window_years={window_years} cutoff_year={cutoff_year} "
            f"kept={len(recent_papers)} removed={len(papers) - len(recent_papers)}"
        )
        papers = recent_papers
        if not papers:
            return []

        query_terms = self._extract_query_terms(query)
        anchor_terms = self._extract_anchor_terms(query_terms, query)
        term_stats: Dict[str, Tuple[int, int]] = {}

        if query_terms:
            min_hits = 2 if len(query_terms) <= 4 else 3
            if not anchor_terms:
                min_hits = max(min_hits, 3)
            topical_papers: List[PaperMetadata] = []
            for paper in papers:
                hits, anchor_hits = self._paper_term_hits(paper, query_terms, anchor_terms)
                term_stats[paper.paper_id] = (hits, anchor_hits)
                if hits < min_hits:
                    continue
                if anchor_terms and anchor_hits == 0:
                    continue
                topical_papers.append(paper)

            self._debug_query(
                f"topical_filter terms={sorted(query_terms)} anchors={sorted(anchor_terms)} "
                f"min_hits={min_hits} kept={len(topical_papers)} removed={len(papers) - len(topical_papers)}"
            )
            if topical_papers:
                papers = topical_papers
            else:
                self._debug_query("topical_filter yielded no papers; returning empty result")
                return []
        else:
            self._debug_query("topical_filter skipped (no usable query terms)")

        query_variants = self._query_variants_for_ranking(query, subqueries)
        query_embeddings = self._get_embeddings(query_variants)
        title_texts = [self._paper_title_text(p) for p in papers]
        abstract_texts = [self._paper_abstract_text(p) for p in papers]
        title_embeddings = self._get_embeddings(title_texts)
        abstract_inputs = [a if a else title_texts[i] for i, a in enumerate(abstract_texts)]
        abstract_embeddings = self._get_embeddings(abstract_inputs)

        scored = []
        stage1_scores: Dict[str, float] = {}
        for paper, title_emb, abstract_emb, abstract_text in zip(
            papers, title_embeddings, abstract_embeddings, abstract_texts
        ):
            year = self._extract_year(paper.published_date) or 0
            hits, anchor_hits = term_stats.get(paper.paper_id, self._paper_term_hits(paper, query_terms, anchor_terms))
            score, best_sem = self._score_paper(
                paper,
                query_embeddings,
                title_emb,
                abstract_emb,
                bool(abstract_text),
                hits,
                anchor_hits,
                year,
                cutoff_year,
            )
            self._debug_query(
                f"stage1_score={score:.4f} best_sem={best_sem:.4f} year={year} hits={hits} anchors={anchor_hits} source={paper.source} "
                f"paper_id={paper.paper_id} title={paper.title!r}"
            )
            scored.append((paper, score, year))
            stage1_scores[paper.paper_id] = score

        ordered = sorted(scored, key=lambda x: (x[1], x[2]), reverse=True)
        ranked = [p for p, _, _ in ordered]

        if self.two_stage_ranking and ranked:
            candidate_limit = min(
                len(ranked),
                max(
                    final_limit,
                    min(self.stage2_candidate_limit, final_limit * self.stage1_candidate_factor),
                ),
            )
            candidates = ranked[:candidate_limit]
            reranked_candidates = self._rerank_candidates(query, candidates, stage1_scores)
            if reranked_candidates:
                reranked_ids = {p.paper_id for p in reranked_candidates}
                tail = [p for p in ranked if p.paper_id not in reranked_ids]
                ranked = reranked_candidates + tail
                self._debug_query(
                    f"stage2_rerank model={self.rerank_model!r} candidates={len(candidates)} applied=true"
                )
            else:
                self._debug_query(
                    f"stage2_rerank model={self.rerank_model!r} candidates={len(candidates)} applied=false"
                )

        self._debug_query(
            f"raw_total={len(all_papers)} deduped={len(ranked)} selected={min(final_limit, len(ranked))}"
        )
        self._debug_selected_citations(query, ranked, final_limit)
        return ranked[:final_limit]
    
    async def _search_semantic_scholar(
        self, 
        query: str, 
        limit: int = 5
    ) -> List[PaperMetadata]:
        """Search Semantic Scholar API"""
        papers = []
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,abstract,url,year,citationCount,openAccessPdf"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for item in data.get("data", []):
                            pdf_url = None
                            if item.get("openAccessPdf"):
                                pdf_url = item["openAccessPdf"].get("url")
                            
                            papers.append(PaperMetadata(
                                paper_id=f"ss_{item.get('paperId', str(uuid.uuid4()))}",
                                title=item.get("title", "Untitled"),
                                authors=[a.get("name", "") for a in item.get("authors", []) if a.get("name")],
                                abstract=item.get("abstract"),
                                url=item.get("url", ""),
                                pdf_url=pdf_url,
                                source="semantic_scholar",
                                published_date=str(item.get("year")) if item.get("year") else None,
                                citations=item.get("citationCount"),
                            ))
        except Exception as e:
            print(f"Semantic Scholar search error: {e}")
        
        return papers
    
    async def _search_arxiv(
        self, 
        query: str, 
        limit: int = 5
    ) -> List[PaperMetadata]:
        """Search arXiv API"""
        papers = []
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "max_results": limit,
            "sortBy": "relevance",
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(text)
                        ns = {"atom": "http://www.w3.org/2005/Atom"}
                        
                        for entry in root.findall("atom:entry", ns):
                            paper_id = entry.find("atom:id", ns).text.split("/")[-1]
                            title = entry.find("atom:title", ns).text.replace("\n", " ")
                            summary = entry.find("atom:summary", ns).text.replace("\n", " ")
                            doi = None
                            doi_elem = entry.find("{http://arxiv.org/schemas/atom}doi")
                            if doi_elem is not None and doi_elem.text:
                                doi = doi_elem.text.strip()
                            
                            authors = []
                            for author in entry.findall("atom:author", ns):
                                name = author.find("atom:name", ns)
                                if name is not None:
                                    authors.append(name.text)
                            
                            pdf_url = None
                            for link in entry.findall("atom:link", ns):
                                if link.get("title") == "pdf":
                                    pdf_url = link.get("href")
                            
                            papers.append(PaperMetadata(
                                paper_id=f"arxiv_{paper_id}",
                                title=title,
                                authors=authors,
                                abstract=summary,
                                url=entry.find("atom:id", ns).text,
                                pdf_url=pdf_url,
                                source="arxiv",
                                published_date=entry.find("atom:published", ns).text[:10] if entry.find("atom:published", ns) is not None else None,
                                citations=None,
                                doi=doi,
                            ))
        except Exception as e:
            print(f"arXiv search error: {e}")
        
        return papers
    
    async def _search_pubmed(
        self, 
        query: str, 
        limit: int = 5
    ) -> List[PaperMetadata]:
        """Search PubMed API"""
        papers = []
        
        try:
            async with aiohttp.ClientSession() as session:
                search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                params = {
                    "db": "pubmed",
                    "term": query,
                    "retmax": limit,
                    "retmode": "json",
                    "sort": "pub date",
                }
                
                async with session.get(search_url, params=params, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        id_list = data.get("esearchresult", {}).get("idlist", [])
                        
                        if id_list:
                            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                            fetch_params = {
                                "db": "pubmed",
                                "id": ",".join(id_list),
                                "retmode": "json"
                            }
                            
                            async with session.get(fetch_url, params=fetch_params, timeout=30) as fetch_resp:
                                if fetch_resp.status == 200:
                                    summary_data = await fetch_resp.json()
                                    for pubmed_id, info in summary_data.get("result", {}).items():
                                        if pubmed_id == "uids":
                                            continue

                                        authors = []
                                        for author in info.get("authors", []):
                                            if isinstance(author, dict):
                                                name = author.get("name")
                                                if name:
                                                    authors.append(name)
                                            elif isinstance(author, str):
                                                authors.append(author)
                                        
                                        papers.append(PaperMetadata(
                                            paper_id=f"pubmed_{pubmed_id}",
                                            title=info.get("title", "Untitled"),
                                            authors=authors,
                                            abstract=None,
                                            url=f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/",
                                            pdf_url=None,
                                            source="pubmed",
                                            published_date=info.get("pubdate"),
                                            citations=None,
                                        ))
        except Exception as e:
            print(f"PubMed search error: {e}")
        
        return papers

    def _normalize_arxiv_id(self, arxiv_id: str) -> str:
        return re.sub(r"v\d+$", "", (arxiv_id or "").strip())

    async def _fetch_semantic_scholar_citation(
        self, session: aiohttp.ClientSession, paper: PaperMetadata
    ) -> Optional[int]:
        if paper.citations is not None:
            return paper.citations

        paper_id = None
        if paper.doi:
            paper_id = f"DOI:{paper.doi}"
        elif paper.paper_id and paper.paper_id.startswith("arxiv_"):
            raw_id = paper.paper_id.split("arxiv_", 1)[1]
            paper_id = f"ARXIV:{self._normalize_arxiv_id(raw_id)}"
        else:
            return None

        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
        params = {"fields": "citationCount"}
        try:
            async with session.get(url, params=params, timeout=30) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
        except Exception:
            return None

        count = data.get("citationCount")
        if isinstance(count, int):
            return count
        return None

    async def _enrich_arxiv_citations(self, papers: List[PaperMetadata]) -> None:
        arxiv_papers = [p for p in papers if p.source == "arxiv"]
        if not arxiv_papers:
            return

        sem = asyncio.Semaphore(5)

        async def enrich_one(paper: PaperMetadata):
            async with sem:
                citation = await self._fetch_semantic_scholar_citation(session, paper)
                if citation is not None:
                    paper.citations = citation

        async with aiohttp.ClientSession() as session:
            await asyncio.gather(*(enrich_one(p) for p in arxiv_papers))
    
    async def download_paper(self, paper: PaperMetadata) -> Optional[Path]:
        """Download paper PDF to storage"""
        if not paper.pdf_url:
            return None
        
        pdf_path = self.papers_dir / f"{paper.paper_id}.pdf"
        
        if pdf_path.exists():
            return pdf_path
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(paper.pdf_url, timeout=60) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        pdf_path.write_bytes(content)
                        return pdf_path
        except Exception as e:
            print(f"Download error for {paper.paper_id}: {e}")
        
        return None
    
    async def extract_content(self, paper: PaperMetadata, pdf_path: Optional[Path]) -> str:
        """Extract text content from PDF using docling"""
        if not pdf_path or not pdf_path.exists():
            return paper.abstract or ""
        
        try:
            if self.docling_force_cpu:
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
                os.environ.setdefault("ORT_DISABLE_GPU", "1")
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()

            if self.docling_page_log:
                try:
                    from pypdf import PdfReader, PdfWriter
                except Exception:
                    self.docling_page_log = False
                    result = converter.convert(str(pdf_path))
                    return self._normalize_text(result.document.export_to_markdown())

                reader = PdfReader(str(pdf_path))
                total_pages = len(reader.pages)
                if total_pages == 0:
                    return paper.abstract or ""
                pages_to_process = total_pages
                if self.docling_max_pages > 0:
                    pages_to_process = min(total_pages, self.docling_max_pages)

                markdown_parts = []
                for i in range(pages_to_process):
                    print(f"[DOC] {paper.paper_id} page {i + 1}/{pages_to_process} (source_total={total_pages})")
                    writer = PdfWriter()
                    writer.add_page(reader.pages[i])
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                            tmp_path = tmp.name
                            writer.write(tmp)
                        result = converter.convert(tmp_path)
                        markdown_parts.append(self._normalize_text(result.document.export_to_markdown()))
                    finally:
                        if tmp_path:
                            try:
                                os.unlink(tmp_path)
                            except Exception:
                                pass
                return "\n\n".join(markdown_parts)

            result = converter.convert(str(pdf_path))
            markdown_text = result.document.export_to_markdown()
            return self._normalize_text(markdown_text)
        except Exception as e:
            print(f"Docling extraction error for {paper.paper_id}: {e}")
            return paper.abstract or ""
    
    async def summarize_paper(
        self, 
        paper: PaperMetadata, 
        content: str,
        query: str
    ) -> str:
        """Generate summary using Ollama"""
        try:
            print(f"[SUM] start paper_id={paper.paper_id}")
            import ollama
            
            system_prompt = """You are an academic research assistant.
Write a concise paper summary:
1. Focus on points directly relevant to the research query.
2. Keep it to 2 short paragraphs, max 120 words total.
3. No equations, no tables, no markdown lists.
4. Use plain punctuation and use '-' instead of em dash."""
            
            user_prompt = f"""Research query: {query}

Paper title: {paper.title}
Authors: {", ".join(paper.authors) if paper.authors else "Unknown"}

Abstract: {paper.abstract or "Not available"}

Full content (excerpt):
{content[:8000]}

Generate a concise summary focused on the research query."""
            
            response = ollama.chat(
                model="gpt-oss:20b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            summary_text = self._postprocess_summary_text(
                response.get("message", {}).get("content", "").strip()
            )
            print(f"[SUM] end paper_id={paper.paper_id}")
            return summary_text
        except Exception as e:
            print(f"Summarization error for {paper.paper_id}: {e}")
            return paper.abstract or ""

    def _fallback_evidence_card(self, query: str, paper_index: int, paper: PaperContent) -> Dict[str, Any]:
        metadata = paper.metadata
        year = self._extract_year(metadata.published_date)
        abstract = self._normalize_text(metadata.abstract or "")
        summary = self._normalize_text(paper.summary or "")
        excerpt = self._normalize_text(paper.full_text or "")[: self.report_evidence_max_chars]
        combined = "\n".join([x for x in [summary, abstract, excerpt] if x]).strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", combined) if s.strip()]
        findings = sentences[:3] if sentences else ["Insufficient evidence in selected papers."]
        return {
            "paper_index": paper_index,
            "title": metadata.title,
            "year": year if year is not None else "Unknown",
            "source": metadata.source,
            "citations": metadata.citations if metadata.citations is not None else "Unknown",
            "problem": findings[0],
            "method": findings[1] if len(findings) > 1 else "Insufficient evidence in selected papers.",
            "setting_data": "Insufficient evidence in selected papers.",
            "key_findings": findings,
            "strengths": ["Insufficient evidence in selected papers."],
            "limitations": ["Insufficient evidence in selected papers."],
            "open_questions": ["Insufficient evidence in selected papers."],
            "relevance_to_query": f"Potentially relevant to query: {query}",
            "confidence": "low",
        }

    async def _extract_evidence_card(self, query: str, paper_index: int, paper: PaperContent) -> Dict[str, Any]:
        metadata = paper.metadata
        fallback = self._fallback_evidence_card(query, paper_index, paper)
        try:
            import ollama
            system_prompt = """You extract structured evidence from one academic paper for a state-of-the-art report.
Return valid JSON only with keys:
paper_index, problem, method, setting_data, key_findings, strengths, limitations, open_questions, relevance_to_query, confidence.
Rules:
1. confidence must be one of: high, medium, low.
2. key_findings, strengths, limitations, open_questions must be arrays of short strings.
3. Use only evidence present in the input.
4. If missing, write 'Insufficient evidence in selected papers.'"""
            content_excerpt = self._normalize_text(paper.full_text or "")[: self.report_evidence_max_chars]
            user_prompt = f"""Research query: {query}
Paper index: {paper_index}
Title: {metadata.title}
Authors: {", ".join(metadata.authors) if metadata.authors else "Unknown"}
Year: {metadata.published_date or "Unknown"}
Source: {metadata.source}
Citations: {metadata.citations if metadata.citations is not None else "Unknown"}
Abstract:
{metadata.abstract or "Not available"}
Summary:
{paper.summary or "Not available"}
Full text excerpt:
{content_excerpt or "Not available"}"""
            response = ollama.chat(
                model=self.report_structured_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            parsed = self._safe_parse_json(response.get("message", {}).get("content", ""))
            if not parsed:
                return fallback

            def to_list(name: str, limit: int = 6) -> List[str]:
                raw = parsed.get(name, [])
                if not isinstance(raw, list):
                    return fallback.get(name, [])[:limit]
                out: List[str] = []
                for item in raw:
                    text = self._normalize_text(item).strip()
                    if text and text not in out:
                        out.append(text)
                    if len(out) >= limit:
                        break
                return out or fallback.get(name, [])[:limit]

            confidence = self._normalize_text(parsed.get("confidence", "")).strip().lower()
            if confidence not in {"high", "medium", "low"}:
                confidence = "low"

            return {
                "paper_index": paper_index,
                "title": metadata.title,
                "year": self._extract_year(metadata.published_date) or "Unknown",
                "source": metadata.source,
                "citations": metadata.citations if metadata.citations is not None else "Unknown",
                "problem": self._normalize_text(parsed.get("problem", fallback["problem"])).strip() or fallback["problem"],
                "method": self._normalize_text(parsed.get("method", fallback["method"])).strip() or fallback["method"],
                "setting_data": self._normalize_text(parsed.get("setting_data", fallback["setting_data"])).strip() or fallback["setting_data"],
                "key_findings": to_list("key_findings"),
                "strengths": to_list("strengths"),
                "limitations": to_list("limitations"),
                "open_questions": to_list("open_questions"),
                "relevance_to_query": self._normalize_text(
                    parsed.get("relevance_to_query", fallback["relevance_to_query"])
                ).strip() or fallback["relevance_to_query"],
                "confidence": confidence,
            }
        except Exception as e:
            self._debug_query(f"evidence card extraction failed idx={paper_index}: {type(e).__name__}: {e!r}")
            return fallback

    async def _build_evidence_cards(self, query: str, papers: List[PaperContent]) -> List[Dict[str, Any]]:
        cards: List[Dict[str, Any]] = []
        for i, paper in enumerate(papers, 1):
            card = await self._extract_evidence_card(query, i, paper)
            cards.append(card)
        return cards

    def _render_card_for_prompt(self, card: Dict[str, Any]) -> str:
        def render_list(values: Any) -> str:
            if not isinstance(values, list):
                return "- Insufficient evidence in selected papers."
            rows = [f"- {self._normalize_text(v).strip()}" for v in values if self._normalize_text(v).strip()]
            return "\n".join(rows) if rows else "- Insufficient evidence in selected papers."

        return "\n".join(
            [
                f"Paper [{card.get('paper_index')}]",
                f"Title: {card.get('title', 'Unknown')}",
                f"Year: {card.get('year', 'Unknown')}",
                f"Source: {card.get('source', 'Unknown')}",
                f"Citations: {card.get('citations', 'Unknown')}",
                f"Problem: {card.get('problem', 'Insufficient evidence in selected papers.')}",
                f"Method: {card.get('method', 'Insufficient evidence in selected papers.')}",
                f"Setting/Data: {card.get('setting_data', 'Insufficient evidence in selected papers.')}",
                "Key findings:",
                render_list(card.get("key_findings")),
                "Strengths:",
                render_list(card.get("strengths")),
                "Limitations:",
                render_list(card.get("limitations")),
                "Open questions:",
                render_list(card.get("open_questions")),
                f"Relevance to query: {card.get('relevance_to_query', 'Insufficient evidence in selected papers.')}",
                f"Confidence: {card.get('confidence', 'low')}",
            ]
        )

    async def _build_report_outline(self, query: str, cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        specs = self._report_section_specs()
        valid_indices = {int(c.get("paper_index")) for c in cards if isinstance(c.get("paper_index"), int)}
        if not valid_indices:
            valid_indices = set(range(1, len(cards) + 1))

        default_outline = [
            {
                "title": title,
                "objective": objective,
                "paper_indices": sorted(valid_indices),
            }
            for title, objective in specs
        ]

        try:
            import ollama
            cards_digest = "\n\n".join(self._render_card_for_prompt(c) for c in cards)
            section_titles = ", ".join([title for title, _ in specs])
            system_prompt = """You plan a report outline and map section coverage to evidence cards.
Return valid JSON only:
{"sections":[{"title":"...", "objective":"...", "paper_indices":[1,2]}]}
Rules:
1. Only use section titles exactly as provided by the user prompt.
2. paper_indices must reference existing card indices only.
3. Every section must include at least one paper index."""
            user_prompt = f"""Research query: {query}
Allowed section titles: {section_titles}
Evidence cards:
{cards_digest}
Return outline JSON only."""
            response = ollama.chat(
                model=self.report_structured_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            parsed = self._safe_parse_json(response.get("message", {}).get("content", ""))
            if not parsed:
                return default_outline
            rows = parsed.get("sections", [])
            if not isinstance(rows, list):
                return default_outline

            normalized: List[Dict[str, Any]] = []
            by_title = {
                self._normalize_text(item.get("title", "")).strip().lower(): item
                for item in rows
                if isinstance(item, dict)
            }
            for title, objective in specs:
                src = by_title.get(title.lower(), {})
                mapped = self._normalize_index_list(src.get("paper_indices"), valid_indices)
                if not mapped:
                    mapped = sorted(valid_indices)
                normalized.append(
                    {
                        "title": title,
                        "objective": self._normalize_text(src.get("objective", objective)).strip() or objective,
                        "paper_indices": mapped,
                    }
                )
            return normalized
        except Exception as e:
            self._debug_query(f"outline planning failed: {type(e).__name__}: {e!r}")
            return default_outline

    async def _write_report_section(
        self,
        query: str,
        title: str,
        objective: str,
        paper_indices: List[int],
        cards_by_index: Dict[int, Dict[str, Any]],
        revision_reason: str = "",
    ) -> str:
        valid_indices = [i for i in paper_indices if i in cards_by_index]
        cards_text = "\n\n---\n\n".join(self._render_card_for_prompt(cards_by_index[i]) for i in valid_indices)
        if not cards_text:
            return "Insufficient evidence in selected papers."
        try:
            import ollama
            system_prompt = f"""You write one section of a technical state-of-the-art report.
Output plain markdown paragraphs only; do not output a heading.
Target length: about {self.report_section_target_words} words.
Rules:
1. Use only provided evidence cards.
2. Cite factual claims with bracket citations like [1], [2].
3. Only allowed citation indices are: {valid_indices}.
4. If evidence is weak, explicitly state 'Insufficient evidence in selected papers.'"""
            user_prompt = f"""Research query: {query}
Section title: {title}
Section objective: {objective}
{f"Revision request: {revision_reason}" if revision_reason else ""}
Evidence cards:
{cards_text}
Write the section body only."""
            response = ollama.chat(
                model=self.report_writer_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = self._normalize_text(response.get("message", {}).get("content", "")).strip()
            text = self._clean_invalid_citations(text, set(valid_indices))
            if not text:
                text = "Insufficient evidence in selected papers."
            if valid_indices and not self._citation_indices_in_text(text):
                text = f"{text}\n\nInsufficient evidence in selected papers."
            return text
        except Exception as e:
            self._debug_query(f"section generation failed title={title!r}: {type(e).__name__}: {e!r}")
            return "Insufficient evidence in selected papers."

    async def _verify_and_repair_sections(
        self,
        query: str,
        sections: List[Dict[str, Any]],
        cards_by_index: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not self.report_verification_enabled:
            return sections

        for section in sections:
            allowed = set(section.get("paper_indices", []))
            section["text"] = self._clean_invalid_citations(section.get("text", ""), allowed)

        try:
            import ollama
            checks_input = []
            for section in sections:
                checks_input.append(
                    {
                        "title": section.get("title"),
                        "allowed_indices": section.get("paper_indices", []),
                        "text": section.get("text", ""),
                    }
                )
            system_prompt = """You verify section quality for a technical report.
Return valid JSON only:
{"section_checks":[{"title":"...", "status":"ok|revise", "reason":"..."}]}
Mark revise when citations are missing/invalid or claims appear unsupported by provided section text."""
            user_prompt = (
                f"Research query: {query}\n"
                f"Sections JSON:\n{json.dumps(checks_input, ensure_ascii=False)}\n"
                "Return verification JSON only."
            )
            response = ollama.chat(
                model=self.report_verifier_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            parsed = self._safe_parse_json(response.get("message", {}).get("content", ""))
            checks = parsed.get("section_checks", []) if parsed else []
            revise_by_title: Dict[str, str] = {}
            if isinstance(checks, list):
                for row in checks:
                    if not isinstance(row, dict):
                        continue
                    title = self._normalize_text(row.get("title", "")).strip()
                    if not title:
                        continue
                    status = self._normalize_text(row.get("status", "")).strip().lower()
                    reason = self._normalize_text(row.get("reason", "")).strip() or "Missing support or citation issues."
                    if status == "revise":
                        revise_by_title[title] = reason

            fixed: List[Dict[str, Any]] = []
            for section in sections:
                title = section.get("title", "")
                allowed = section.get("paper_indices", [])
                text = section.get("text", "")
                needs_revision = False
                reason = revise_by_title.get(title, "")
                if allowed and not self._citation_indices_in_text(text):
                    needs_revision = True
                    if not reason:
                        reason = "Add concrete supported claims with valid citations."
                if reason:
                    needs_revision = True

                if needs_revision:
                    rewritten = await self._write_report_section(
                        query=query,
                        title=title,
                        objective=section.get("objective", ""),
                        paper_indices=allowed,
                        cards_by_index=cards_by_index,
                        revision_reason=reason,
                    )
                    section["text"] = self._clean_invalid_citations(rewritten, set(allowed))
                fixed.append(section)
            return fixed
        except Exception as e:
            self._debug_query(f"report verification failed: {type(e).__name__}: {e!r}")
            return sections

    async def _generate_bottom_line(
        self,
        query: str,
        cards: List[Dict[str, Any]],
    ) -> str:
        valid_indices = {
            int(c.get("paper_index"))
            for c in cards
            if isinstance(c.get("paper_index"), int)
        }
        cards_text = "\n\n---\n\n".join(self._render_card_for_prompt(c) for c in cards)
        try:
            import ollama
            system_prompt = """Write a short bottom-line synthesis for a state-of-the-art report.
Rules:
1. 80-140 words.
2. State where the field stands right now.
3. Include confidence labels (high/medium/low) for major claims.
4. Use citations like [1], [2] only from allowed indices."""
            user_prompt = f"""Research query: {query}
Allowed citation indices: {sorted(valid_indices)}
Evidence cards:
{cards_text}
Write only the bottom-line paragraph."""
            response = ollama.chat(
                model=self.report_writer_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = self._normalize_text(response.get("message", {}).get("content", "")).strip()
            text = self._clean_invalid_citations(text, valid_indices)
            return text or "Insufficient evidence in selected papers."
        except Exception as e:
            self._debug_query(f"bottom-line generation failed: {type(e).__name__}: {e!r}")
            return "Insufficient evidence in selected papers."

    async def _generate_state_of_art_report(self, query: str, papers: List[PaperContent]) -> str:
        cards = await self._build_evidence_cards(query, papers)
        if not cards:
            return ""

        cards_by_index = {
            int(card["paper_index"]): card
            for card in cards
            if isinstance(card.get("paper_index"), int)
        }
        outline = await self._build_report_outline(query, cards)
        sections: List[Dict[str, Any]] = []
        for row in outline:
            title = row.get("title", "").strip()
            objective = row.get("objective", "").strip()
            paper_indices = row.get("paper_indices", [])
            text = await self._write_report_section(
                query=query,
                title=title,
                objective=objective,
                paper_indices=paper_indices,
                cards_by_index=cards_by_index,
            )
            sections.append(
                {
                    "title": title,
                    "objective": objective,
                    "paper_indices": paper_indices,
                    "text": text,
                }
            )

        sections = await self._verify_and_repair_sections(query, sections, cards_by_index)
        bottom_line = await self._generate_bottom_line(query, cards)

        parts: List[str] = []
        for section in sections:
            title = section.get("title", "").strip()
            if not title:
                continue
            body = section.get("text", "").strip() or "Insufficient evidence in selected papers."
            parts.append(f"**{title}**\n{body}")
        parts.append(f"**Bottom Line**\n{bottom_line}")
        report = "\n\n".join(parts).strip()
        return self._postprocess_summary_text(report, paper_count=len(papers))
    
    async def run_research_pipeline(
        self,
        query: str,
        options: ResearchOptions,
        query_id: str
    ) -> Dict[str, Any]:
        """Run the complete research pipeline"""
        result = {
            "query_id": query_id,
            "query": query,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": ResearchStatus.PROCESSING,
            "papers": [],
            "summary": None,
            "pdf_path": None,
            "error": None,
        }
        
        try:
            papers = await self.search_academic_papers(query)
            if not papers:
                result["status"] = ResearchStatus.COMPLETED
                result["summary"] = (
                    f"No relevant papers were found for '{query}' in the last "
                    f"{self.recent_paper_years} year(s). "
                    "Try broadening the query terms, using a nearby model name, or increasing the time window."
                )
                await self._save_query_state(query_id, result)
                return result
            
            paper_contents = []
            
            for paper in papers:
                pdf_path = await self.download_paper(paper)
                
                content = await self.extract_content(paper, pdf_path)
                
                summary = None
                if options.summarize:
                    summary = await self.summarize_paper(paper, content, query)
                
                paper_content = PaperContent(
                    metadata=paper,
                    full_text=self._normalize_text(content) if content else None,
                    summary=summary,
                )
                paper_contents.append(paper_content)
                
                result["papers"] = [
                    p.model_dump()
                    for p in paper_contents
                ]
                
                await self._save_query_state(query_id, result)
            
            if options.summarize and paper_contents:
                if self.report_multi_stage:
                    result["summary"] = await self._generate_state_of_art_report(
                        query, paper_contents
                    )
                if not result.get("summary"):
                    result["summary"] = await self._generate_combined_summary(
                        query, paper_contents
                    )
            
            if options.export_pdf:
                from .export_service import ExportService
                export_service = ExportService(self.storage_dir)
                try:
                    print(f"[EXPORT] invoke query_id={query_id}")
                    pdf_path = export_service.generate_pdf(
                        query_id=query_id,
                        query=query,
                        papers=paper_contents,
                        summary=result["summary"]
                    )
                    result["pdf_path"] = pdf_path
                except Exception as e:
                    print(f"[EXPORT] error query_id={query_id}: {e}")
                    result["pdf_path"] = None
            
            result["status"] = ResearchStatus.COMPLETED
            
        except Exception as e:
            result["status"] = ResearchStatus.FAILED
            result["error"] = str(e)
        
        await self._save_query_state(query_id, result)
        return result
    
    async def _generate_combined_summary(
        self, 
        query: str, 
        papers: List[PaperContent]
    ) -> str:
        """Generate a combined summary from all papers"""
        try:
            print("[COMBINE] start")
            import ollama
            
            paper_summaries = []
            for i, p in enumerate(papers, 1):
                metadata = p.metadata
                abstract_text = (metadata.abstract or "").strip() or "Not available"
                excerpt_source = (p.full_text or p.summary or metadata.abstract or "").strip()
                excerpt = excerpt_source[: self.report_per_paper_chars] if excerpt_source else "Not available"
                year = self._extract_year(metadata.published_date)
                paper_summaries.append(
                    "\n".join(
                        [
                            f"Paper [{i}]",
                            f"Title: {metadata.title}",
                            f"Year: {year if year is not None else 'Unknown'}",
                            f"Source: {metadata.source}",
                            f"Citations: {metadata.citations if metadata.citations is not None else 'Unknown'}",
                            f"Abstract: {abstract_text}",
                            "Evidence excerpt:",
                            excerpt,
                        ]
                    )
                )
            
            combined_text = "\n\n---\n\n".join(paper_summaries)
            
            system_prompt = f"""You are an academic research assistant writing a detailed state-of-the-art report.
Audience: technically literate readers who want both foundations and current frontier status.
Write between {self.report_min_words} and {self.report_max_words} words.
Required output structure (exact section titles, each on its own line in bold markdown):
**Topic Overview**
**Core Mechanisms**
**Evolution Of The Field**
**Current State Of The Art**
**Comparative Analysis Of Approaches**
**Practical Guidance**
**Limitations And Open Problems**
**Future Directions**
Output rules:
1. Explain the concept itself before discussing recent papers.
2. Base all factual claims only on provided paper evidence.
3. Add bracket citations like [1], [2] tied to provided paper indices.
4. Use only citation indices that exist in the provided list.
5. For each section, include concrete evidence-backed claims and tradeoffs.
6. If evidence is insufficient for a claim, explicitly say "Insufficient evidence in selected papers."
7. No equations, no tables, no emoji, and use '-' instead of em dash.
8. End with a short paragraph titled **Bottom Line** that states where the field stands right now."""
            
            user_prompt = f"""Research query: {query}

Paper evidence:

{combined_text}

Generate a detailed state-of-the-art report addressing the research query."""
            
            response = ollama.chat(
                model="gpt-oss:20b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            combined = self._postprocess_summary_text(
                response.get("message", {}).get("content", "").strip(),
                paper_count=len(papers),
            )
            print("[COMBINE] end")
            return combined
        except Exception as e:
            print(f"Combined summary error: {e}")
            return ""
    
    async def _save_query_state(self, query_id: str, result: Dict[str, Any]):
        """Save query state to disk"""
        query_file = self.queries_dir / f"{query_id}.json"
        previous_state = None
        if query_file.exists():
            try:
                content = query_file.read_text(encoding="utf-8")
                previous_state = json.loads(content)
            except Exception:
                previous_state = None

        state = {
            "query_id": query_id,
            "query": result["query"],
            "status": result["status"].value if isinstance(result["status"], ResearchStatus) else result["status"],
            "papers": result.get("papers", []),
            "summary": result.get("summary"),
            "pdf_path": result.get("pdf_path"),
            "error": result.get("error"),
            "created_at": result.get("created_at") or (previous_state or {}).get("created_at") or datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat() if result["status"] == ResearchStatus.COMPLETED else None,
        }
        
        query_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    
    async def get_query_state(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Get query state from disk"""
        query_file = self.queries_dir / f"{query_id}.json"
        
        if not query_file.exists():
            return None
        
        content = query_file.read_text(encoding="utf-8")
        return json.loads(content)
    
    async def list_queries(self) -> List[Dict[str, Any]]:
        """List all queries"""
        queries = []
        
        for query_file in self.queries_dir.glob("*.json"):
            try:
                content = query_file.read_text(encoding="utf-8")
                data = json.loads(content)
                queries.append({
                    "query_id": data.get("query_id"),
                    "query": data.get("query"),
                    "status": data.get("status"),
                    "paper_count": len(data.get("papers", [])),
                    "created_at": data.get("created_at"),
                    "completed_at": data.get("completed_at"),
                })
            except Exception as e:
                print(f"Error reading {query_file}: {e}")
        
        queries.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return queries
