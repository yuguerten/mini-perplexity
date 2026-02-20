import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from .models import (
    ResearchRequest,
    ResearchResponse,
    ResearchOptions,
    ResearchStatus,
    RAGQueryRequest,
    RAGQueryResponse,
    ResearchHistoryItem,
)
from ..services.research_service import ResearchService
from ..services.rag_service import RAGService
from ..services.export_service import ExportService


research_service: ResearchService = None
rag_service: RAGService = None
export_service: ExportService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global research_service, rag_service, export_service
    
    storage_dir = os.getenv("STORAGE_DIR", "data")
    
    research_service = ResearchService(storage_dir)
    rag_service = RAGService(storage_dir)
    export_service = ExportService(storage_dir)
    
    yield
    
    research_service = None
    rag_service = None
    export_service = None


app = FastAPI(
    title="Mini-Perplexity Academic Research API",
    description="AI-powered academic research assistant for students, researchers, and professors",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "name": "Mini-Perplexity Academic Research API",
        "version": "2.0.0",
        "description": "AI-powered academic research assistant"
    }


@app.post("/research", response_model=ResearchResponse)
async def create_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks
):
    query_id = str(uuid4())
    options = request.options or ResearchOptions()
    
    created_at = datetime.now(timezone.utc)
    
    initial_response = ResearchResponse(
        query_id=query_id,
        status=ResearchStatus.PROCESSING,
        query=request.query,
        created_at=created_at,
    )
    
    background_tasks.add_task(
        run_research,
        query_id,
        request.query,
        options
    )
    
    return initial_response


async def run_research(query_id: str, query: str, options: ResearchOptions):
    await research_service.run_research_pipeline(query, options, query_id)
    
    if options.store_for_rag:
        state = await research_service.get_query_state(query_id)
        if state and state.get("papers"):
            for paper_data in state["papers"]:
                metadata = paper_data.get("metadata", {})
                full_text = paper_data.get("full_text", "")
                
                if full_text:
                    rag_service.add_paper(
                        paper_id=metadata.get("paper_id"),
                        content=full_text,
                        metadata={
                            "title": metadata.get("title"),
                            "authors": ", ".join(metadata.get("authors", [])),
                            "source": metadata.get("source"),
                            "url": metadata.get("url"),
                        }
                    )


@app.get("/research/{query_id}", response_model=ResearchResponse)
async def get_research(query_id: str):
    state = await research_service.get_query_state(query_id)
    
    if not state:
        raise HTTPException(status_code=404, detail="Research not found")
    
    completed_at = None
    if state.get("completed_at"):
        completed_at = datetime.fromisoformat(state["completed_at"])
    
    return ResearchResponse(
        query_id=state["query_id"],
        status=ResearchStatus(state["status"]),
        query=state["query"],
        papers=state.get("papers"),
        summary=state.get("summary"),
        pdf_path=state.get("pdf_path"),
        error=state.get("error"),
        created_at=datetime.fromisoformat(state["created_at"]),
        completed_at=completed_at,
    )


@app.post("/rag/query", response_model=RAGQueryResponse)
async def query_rag(request: RAGQueryRequest):
    sources = rag_service.query(request.query, request.top_k)
    
    if not sources:
        return RAGQueryResponse(
            answer="No relevant papers found in the knowledge base. Please run a research query first.",
            sources=[]
        )
    
    answer = rag_service.generate_answer(request.query, sources)
    
    return RAGQueryResponse(
        answer=answer,
        sources=[
            {
                "title": s.get("title"),
                "section": s.get("section"),
                "paper_id": s.get("paper_id"),
            }
            for s in sources
        ]
    )


@app.get("/history", response_model=list[ResearchHistoryItem])
async def get_history():
    queries = await research_service.list_queries()
    
    return [
        ResearchHistoryItem(
            query_id=q["query_id"],
            query=q["query"],
            status=ResearchStatus(q["status"]),
            paper_count=q.get("paper_count", 0),
            created_at=datetime.fromisoformat(q["created_at"]),
            completed_at=datetime.fromisoformat(q["completed_at"]) if q.get("completed_at") else None,
        )
        for q in queries
    ]


@app.get("/export/{query_id}")
async def export_pdf(query_id: str):
    pdf_path = export_service.get_pdf_path(query_id)
    
    if not pdf_path:
        state = await research_service.get_query_state(query_id)
        if not state:
            raise HTTPException(status_code=404, detail="Research not found")
        if not state.get("pdf_path"):
            raise HTTPException(status_code=404, detail="PDF not generated")
        pdf_path = state["pdf_path"]
    
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"research_{query_id}.pdf"
    )


@app.get("/rag/stats")
async def rag_stats():
    return rag_service.get_collection_stats()


@app.delete("/rag/paper/{paper_id}")
async def delete_paper(paper_id: str):
    rag_service.delete_paper(paper_id)
    return {"message": f"Paper {paper_id} deleted"}


@app.delete("/rag/clear")
async def clear_rag():
    rag_service.clear_all()
    return {"message": "All papers cleared from RAG"}
