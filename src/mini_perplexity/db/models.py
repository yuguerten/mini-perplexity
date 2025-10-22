"""
SQLAlchemy models for mini-perplexity database.

Includes support for pgvector embeddings and full research session tracking.
"""
import uuid
from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    DateTime,
    ForeignKey,
    Enum,
    ARRAY,
    JSON,
    UniqueConstraint,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
import enum


# ============================================================================
# Enums
# ============================================================================

class QueryType(str, enum.Enum):
    """Type of query in a research session."""
    INITIAL = "initial"
    CLARIFICATION = "clarification"
    FOLLOW_UP = "follow_up"


class SessionStatus(str, enum.Enum):
    """Status of a research session."""
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class SummaryType(str, enum.Enum):
    """Type of summary generated."""
    INDIVIDUAL = "individual"  # Summary of a single piece of content
    COMBINED = "combined"      # Summary combining multiple sources
    FINAL = "final"            # Final answer to the research question


# ============================================================================
# Base Model
# ============================================================================

class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all models with async support."""
    pass


# ============================================================================
# Models
# ============================================================================

class ResearchSession(Base):
    """
    Represents a complete research session.
    
    A session starts with an initial question and may contain multiple
    queries, clarifications, and follow-ups.
    """
    __tablename__ = "research_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )
    original_question: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus),
        default=SessionStatus.ACTIVE,
        nullable=False,
        index=True
    )
    metadata_: Mapped[Optional[dict]] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
        comment="Additional metadata like user info, tags, etc."
    )

    # Relationships
    queries: Mapped[List["Query"]] = relationship(
        "Query",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Query.created_at"
    )

    def __repr__(self) -> str:
        return f"<ResearchSession(id={self.id}, question='{self.original_question[:50]}...', status={self.status})>"


class Query(Base):
    """
    Represents a single query within a research session.
    
    Includes the query text, type, categories used, and embeddings
    for semantic search.
    """
    __tablename__ = "queries"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("research_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    query_text: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )
    query_type: Mapped[QueryType] = mapped_column(
        Enum(QueryType),
        default=QueryType.INITIAL,
        nullable=False
    )
    categories_used: Mapped[Optional[List[str]]] = mapped_column(
        ARRAY(String),
        nullable=True,
        comment="SearXNG categories used for this query"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(384),  # Dimension for all-MiniLM-L6-v2
        nullable=True,
        comment="Query embedding for semantic search"
    )
    metadata_: Mapped[Optional[dict]] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
        comment="Query analysis results, optimization info, etc."
    )

    # Relationships
    session: Mapped["ResearchSession"] = relationship(
        "ResearchSession",
        back_populates="queries"
    )
    search_results: Mapped[List["SearchResult"]] = relationship(
        "SearchResult",
        back_populates="query",
        cascade="all, delete-orphan"
    )
    summaries: Mapped[List["Summary"]] = relationship(
        "Summary",
        back_populates="query",
        cascade="all, delete-orphan"
    )

    # Index for vector similarity search
    __table_args__ = (
        Index('idx_query_embedding', 'embedding', postgresql_using='ivfflat'),
    )

    def __repr__(self) -> str:
        return f"<Query(id={self.id}, text='{self.query_text[:50]}...', type={self.query_type})>"


class SearchResult(Base):
    """
    Represents a single URL returned from a search.
    
    Links queries to the URLs that were found and potentially crawled.
    """
    __tablename__ = "search_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False
    )
    query_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("queries.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True
    )
    title: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    rank: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Position in search results (1-based)"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )
    metadata_: Mapped[Optional[dict]] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
        comment="Search engine, snippet, etc."
    )

    # Relationships
    query: Mapped["Query"] = relationship(
        "Query",
        back_populates="search_results"
    )
    crawled_content: Mapped[Optional["CrawledContent"]] = relationship(
        "CrawledContent",
        back_populates="search_result",
        uselist=False
    )

    def __repr__(self) -> str:
        return f"<SearchResult(id={self.id}, url='{self.url}', rank={self.rank})>"


class CrawledContent(Base):
    """
    Represents content crawled from a URL.
    
    Stores both raw and markdown content with embeddings for RAG.
    Uses content_hash for deduplication.
    """
    __tablename__ = "crawled_content"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False
    )
    search_result_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("search_results.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        unique=True,
        index=True,
        comment="URL is unique to prevent duplicate crawls"
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Raw text content"
    )
    markdown: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Markdown formatted content"
    )
    crawled_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(384),
        nullable=True,
        comment="Content embedding for RAG"
    )
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="SHA256 hash of content for deduplication"
    )
    metadata_: Mapped[Optional[dict]] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
        comment="Crawl metadata: status code, headers, crawl duration, etc."
    )

    # Relationships
    search_result: Mapped["SearchResult"] = relationship(
        "SearchResult",
        back_populates="crawled_content"
    )

    # Indexes
    __table_args__ = (
        Index('idx_content_embedding', 'embedding', postgresql_using='ivfflat'),
        Index('idx_content_hash', 'content_hash'),
    )

    def __repr__(self) -> str:
        return f"<CrawledContent(id={self.id}, url='{self.url}', hash='{self.content_hash[:8]}...')>"


class Summary(Base):
    """
    Represents a generated summary.
    
    Can be an individual source summary, a combined summary,
    or a final answer. Includes embeddings for similarity search.
    """
    __tablename__ = "summaries"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False
    )
    query_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("queries.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    summary_text: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )
    summary_type: Mapped[SummaryType] = mapped_column(
        Enum(SummaryType),
        nullable=False,
        index=True
    )
    model_used: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="LLM model used for generation (e.g., 'llama3.2:3b')"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(384),
        nullable=True,
        comment="Summary embedding for similarity search"
    )
    sources: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Array of source URLs and their relevance scores"
    )
    metadata_: Mapped[Optional[dict]] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
        comment="Generation metadata: tokens used, temperature, etc."
    )

    # Relationships
    query: Mapped["Query"] = relationship(
        "Query",
        back_populates="summaries"
    )

    # Index for vector similarity search
    __table_args__ = (
        Index('idx_summary_embedding', 'embedding', postgresql_using='ivfflat'),
    )

    def __repr__(self) -> str:
        return f"<Summary(id={self.id}, type={self.summary_type}, model='{self.model_used}')>"
