from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ResearchStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchOptions(BaseModel):
    max_papers: int = Field(default=5, ge=1, le=20)
    summarize: bool = True
    store_for_rag: bool = True
    export_pdf: bool = True


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    options: Optional[ResearchOptions] = None


class PaperMetadata(BaseModel):
    paper_id: str
    title: str
    authors: List[str]
    abstract: Optional[str] = None
    url: str
    pdf_url: Optional[str] = None
    source: str  # semantic_scholar, arxiv, pubmed, core
    published_date: Optional[str] = None
    citations: Optional[int] = None
    doi: Optional[str] = None


class PaperContent(BaseModel):
    metadata: PaperMetadata
    full_text: Optional[str] = None
    summary: Optional[str] = None


class ResearchResponse(BaseModel):
    query_id: str
    status: ResearchStatus
    query: str
    papers: Optional[List[PaperContent]] = None
    summary: Optional[str] = None
    pdf_path: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class RAGQueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    query_id: Optional[str] = None


class ResearchHistoryItem(BaseModel):
    query_id: str
    query: str
    status: ResearchStatus
    paper_count: int
    created_at: datetime
    completed_at: Optional[datetime] = None
