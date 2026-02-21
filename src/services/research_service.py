import asyncio
import json
import os
import re
import uuid
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
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
        self.docling_force_cpu = os.getenv("DOCLING_FORCE_CPU", "true").lower() in {"1", "true", "yes", "on"}
        self.docling_page_log = os.getenv("DOCLING_PAGE_LOG", "true").lower() in {"1", "true", "yes", "on"}
        self.docling_max_pages = int(os.getenv("DOCLING_MAX_PAGES", "4"))

    def _normalize_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value).encode("utf-8", errors="replace").decode("utf-8")

        for d in [self.papers_dir, self.queries_dir, self.summaries_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _debug_selected_citations(self, query: str, papers: List[PaperMetadata], max_papers: int):
        if not self.debug_citations:
            return
        selected = papers[:max_papers]
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
                "You are an academic search query optimizer. "
                "Given a user query, generate exactly 3 close subqueries for paper retrieval. "
                "Keep the same intent, no topic drift. "
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
                if q and q.lower() != base_query.lower() and q.lower() not in {x.lower() for x in subqueries}:
                    subqueries.append(q)
                if len(subqueries) >= 3:
                    break
        except Exception as e:
            self._debug_query(f"LLM subquery generation failed: {type(e).__name__}: {e!r}")

        fallback = [
            f"{base_query} state of the art",
            f"{base_query} recent advances 2024 2025",
            f"{base_query} survey",
        ]
        for q in fallback:
            if len(subqueries) >= 3:
                break
            if q.lower() not in {x.lower() for x in subqueries}:
                subqueries.append(q)

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
        query_embedding: List[float],
        paper_embedding: List[float],
        keywords: List[str],
    ) -> float:
        sim = self._cosine_sim(query_embedding, paper_embedding)
        hits = self._keyword_hits(keywords, self._paper_text(paper))
        keyword_boost = min(hits, 3) * 0.8
        citation_score = math.log1p(float(paper.citations or 0)) * 0.2
        pdf_bonus = 0.3 if paper.pdf_url else 0.0
        return (sim * 10.0) + keyword_boost + citation_score + pdf_bonus
    
    async def search_academic_papers(
        self, 
        query: str, 
        max_papers: int = 5
    ) -> List[PaperMetadata]:
        """Search by LLM-generated subqueries, then dedupe and rerank to the original query."""
        subqueries = await self._generate_subqueries(query)
        if not subqueries:
            subqueries = [query]

        all_papers: List[PaperMetadata] = []
        per_query_limit = max(5, max_papers)

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

        keywords = await self._extract_keywords(query)
        if keywords:
            self._debug_query(f"keywords={keywords}")

        query_embedding = self._get_embeddings([query])[0]
        paper_texts = [self._paper_text(p) for p in papers]
        paper_embeddings = self._get_embeddings(paper_texts)

        scored = []
        for paper, emb in zip(papers, paper_embeddings):
            score = self._score_paper(paper, query_embedding, emb, keywords)
            hits = self._keyword_hits(keywords, self._paper_text(paper)) if keywords else 0
            scored.append((paper, score, hits))

        if keywords:
            primary = [s for s in scored if s[2] >= self.keyword_min_hits]
            secondary = [s for s in scored if s[2] < self.keyword_min_hits]
            if not primary:
                primary = scored
                secondary = []
            primary.sort(key=lambda x: x[1], reverse=True)
            secondary.sort(key=lambda x: x[1], reverse=True)
            ordered = primary + secondary
        else:
            ordered = sorted(scored, key=lambda x: x[1], reverse=True)

        ranked = [p for p, _, _ in ordered]

        self._debug_query(
            f"raw_total={len(all_papers)} deduped={len(ranked)} selected={min(max_papers, len(ranked))}"
        )
        self._debug_selected_citations(query, ranked, max_papers)
        return ranked[:max_papers]
    
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
                    "retmode": "json"
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
Generate a concise summary of the paper that:
1. Focuses on content relevant to the research query
2. Highlights key findings and methodology
3. Uses factual and objective language
4. Is 2-3 paragraphs maximum"""
            
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
            summary_text = response.get("message", {}).get("content", "").strip()
            print(f"[SUM] end paper_id={paper.paper_id}")
            return summary_text
        except Exception as e:
            print(f"Summarization error for {paper.paper_id}: {e}")
            return paper.abstract or ""
    
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
            papers = await self.search_academic_papers(query, options.max_papers)
            
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
                summary_text = p.summary or p.metadata.abstract or "No summary available"
                paper_summaries.append(f"Paper {i}: {p.metadata.title}\n{summary_text}")
            
            combined_text = "\n\n---\n\n".join(paper_summaries)
            
            system_prompt = """You are an academic research assistant.
Create a consolidated literature review that:
1. Synthesizes findings across multiple papers
2. Addresses the research query directly
3. Uses academic tone and structure
4. Includes proper citations [1], [2], etc.
5. Avoids markdown tables, emoji, checkmarks, and decorative symbols
6. Uses plain section headings and concise paragraphs"""
            
            user_prompt = f"""Research query: {query}

Papers summary:

{combined_text}

Generate a consolidated literature review addressing the research query."""
            
            response = ollama.chat(
                model="gpt-oss:20b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            combined = response.get("message", {}).get("content", "").strip()
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
