import asyncio
import json
import os
import uuid
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
        self.exports_dir = self.storage_dir / "exports"
        self.debug_pdf_download = os.getenv("DEBUG_PDF_DOWNLOAD", "false").lower() in {"1", "true", "yes", "on"}
        
        for d in [self.papers_dir, self.queries_dir, self.summaries_dir, self.exports_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _debug(self, message: str):
        if self.debug_pdf_download:
            print(f"[PDF_DEBUG] {message}")
    
    async def search_academic_papers(
        self, 
        query: str, 
        max_papers: int = 5
    ) -> List[PaperMetadata]:
        """Search for papers across academic sources"""
        papers = []
        
        tasks = [
            self._search_semantic_scholar(query, max_papers),
            self._search_arxiv(query, max_papers),
            self._search_pubmed(query, max_papers),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                papers.extend(result)
        
        papers.sort(key=lambda p: p.citations or 0, reverse=True)
        return papers[:max_papers]
    
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
    
    async def download_paper(self, paper: PaperMetadata) -> Optional[Path]:
        """Download paper PDF to storage"""
        if not paper.pdf_url:
            self._debug(
                f"Skip download (no pdf_url) paper_id={paper.paper_id} source={paper.source} "
                f"paper_url={paper.url}"
            )
            return None
        
        pdf_path = self.papers_dir / f"{paper.paper_id}.pdf"
        
        if pdf_path.exists():
            self._debug(
                f"Reusing existing PDF paper_id={paper.paper_id} path={pdf_path} pdf_url={paper.pdf_url}"
            )
            return pdf_path
        
        try:
            self._debug(
                f"Downloading paper_id={paper.paper_id} source={paper.source} "
                f"paper_url={paper.url} pdf_url={paper.pdf_url}"
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(paper.pdf_url, timeout=60) as resp:
                    if resp.status == 200:
                        content_type = resp.headers.get("Content-Type", "")
                        content = await resp.read()
                        if "pdf" not in content_type.lower():
                            self._debug(
                                f"Unexpected content-type paper_id={paper.paper_id} "
                                f"status=200 content_type={content_type} bytes={len(content)}"
                            )
                        pdf_path.write_bytes(content)
                        self._debug(
                            f"Download success paper_id={paper.paper_id} status=200 "
                            f"bytes={len(content)} saved={pdf_path}"
                        )
                        return pdf_path
                    self._debug(
                        f"Download failed paper_id={paper.paper_id} status={resp.status} "
                        f"pdf_url={paper.pdf_url}"
                    )
        except Exception as e:
            self._debug(
                f"Download exception paper_id={paper.paper_id} pdf_url={paper.pdf_url} error={e}"
            )
            print(f"Download error for {paper.paper_id}: {e}")
        
        return None
    
    async def extract_content(self, paper: PaperMetadata, pdf_path: Optional[Path]) -> str:
        """Extract text content from PDF using docling"""
        if not pdf_path or not pdf_path.exists():
            return paper.abstract or ""
        
        try:
            from docling.document_converter import DocumentConverter
            
            converter = DocumentConverter()
            result = converter.convert(str(pdf_path))
            
            return result.document.export_to_markdown()
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
                model="lfm2.5-thinking:latest",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            
            return response.get("message", {}).get("content", "").strip()
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
                self._debug(
                    f"Pipeline paper paper_id={paper.paper_id} source={paper.source} "
                    f"paper_url={paper.url} pdf_url={paper.pdf_url or 'None'}"
                )
                pdf_path = await self.download_paper(paper)
                
                content = await self.extract_content(paper, pdf_path)
                
                summary = None
                if options.summarize:
                    summary = await self.summarize_paper(paper, content, query)
                
                paper_content = PaperContent(
                    metadata=paper,
                    full_text=content[:5000] if content else None,
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
                self._cleanup_previous_exports_for_query(query=query, current_query_id=query_id)
                from .export_service import ExportService
                export_service = ExportService(self.storage_dir)
                pdf_path = export_service.generate_pdf(
                    query_id=query_id,
                    query=query,
                    papers=paper_contents,
                    summary=result["summary"]
                )
                result["pdf_path"] = pdf_path
            
            result["status"] = ResearchStatus.COMPLETED
            
        except Exception as e:
            result["status"] = ResearchStatus.FAILED
            result["error"] = str(e)
        
        await self._save_query_state(query_id, result)
        return result

    def _cleanup_previous_exports_for_query(self, query: str, current_query_id: str):
        """Keep only the newest export for the same query text."""
        for query_file in self.queries_dir.glob("*.json"):
            try:
                data = json.loads(query_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            if data.get("query") != query:
                continue
            if data.get("query_id") == current_query_id:
                continue

            old_pdf = data.get("pdf_path")
            if old_pdf:
                old_pdf_path = Path(old_pdf)
                if old_pdf_path.exists():
                    try:
                        old_pdf_path.unlink()
                    except Exception as e:
                        print(f"Failed to remove old export {old_pdf_path}: {e}")
    
    async def _generate_combined_summary(
        self, 
        query: str, 
        papers: List[PaperContent]
    ) -> str:
        """Generate a combined summary from all papers"""
        try:
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
4. Includes proper citations [1], [2], etc."""
            
            user_prompt = f"""Research query: {query}

Papers summary:

{combined_text}

Generate a consolidated literature review addressing the research query."""
            
            response = ollama.chat(
                model="lfm2.5-thinking:latest",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            
            return response.get("message", {}).get("content", "").strip()
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
