"""
Repository for managing research sessions.

Provides data access layer for ResearchSession model with methods for
creating, reading, updating, and deleting research sessions.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID

from sqlalchemy import select, func, and_
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import ResearchSession, Query, SearchResult, CrawledContent, Summary, SessionStatus


class SessionRepository:
    """Repository for research session operations."""
    
    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.
        
        Args:
            session: SQLAlchemy async session
        """
        self.session = session
    
    # ========================================================================
    # CREATE Operations
    # ========================================================================
    
    async def create_session(
        self,
        original_question: str,
        status: SessionStatus = SessionStatus.ACTIVE,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ResearchSession:
        """
        Create a new research session.
        
        Args:
            original_question: The initial research question
            status: Session status (default: ACTIVE)
            metadata: Additional metadata (tags, user info, etc.)
        
        Returns:
            Created ResearchSession instance
        
        Example:
            session = await repo.create_session(
                "What is RAG in AI?",
                metadata={"tags": ["AI", "NLP"], "user": "john"}
            )
        """
        research_session = ResearchSession(
            original_question=original_question,
            status=status,
            metadata_=metadata or {}
        )
        
        self.session.add(research_session)
        await self.session.flush()  # Get the ID without committing
        
        return research_session
    
    # ========================================================================
    # READ Operations
    # ========================================================================
    
    async def get_session(self, session_id: UUID) -> Optional[ResearchSession]:
        """
        Get a specific research session by ID.
        
        Args:
            session_id: UUID of the session
        
        Returns:
            ResearchSession or None if not found
        """
        result = await self.session.execute(
            select(ResearchSession).where(ResearchSession.id == session_id)
        )
        return result.scalar_one_or_none()
    
    async def get_active_sessions(self, limit: Optional[int] = None) -> List[ResearchSession]:
        """
        Get all active research sessions.
        
        Args:
            limit: Maximum number of sessions to return
        
        Returns:
            List of active ResearchSession instances
        """
        query = select(ResearchSession).where(
            ResearchSession.status == SessionStatus.ACTIVE
        ).order_by(ResearchSession.created_at.desc())
        
        if limit:
            query = query.limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_all_sessions(
        self,
        status: Optional[SessionStatus] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[ResearchSession]:
        """
        Get all research sessions with optional filtering.
        
        Args:
            status: Filter by session status
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip (for pagination)
        
        Returns:
            List of ResearchSession instances
        """
        query = select(ResearchSession).order_by(ResearchSession.created_at.desc())
        
        if status:
            query = query.where(ResearchSession.status == status)
        
        if offset:
            query = query.offset(offset)
        
        if limit:
            query = query.limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_session_with_queries(self, session_id: UUID) -> Optional[ResearchSession]:
        """
        Get a session with all its queries eagerly loaded.
        
        Args:
            session_id: UUID of the session
        
        Returns:
            ResearchSession with queries loaded, or None if not found
        """
        result = await self.session.execute(
            select(ResearchSession)
            .options(selectinload(ResearchSession.queries))
            .where(ResearchSession.id == session_id)
        )
        return result.scalar_one_or_none()
    
    async def get_session_summary(self, session_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get complete session data including all related entities.
        
        This loads the session with all queries, search results, content, and summaries.
        Useful for getting full research context.
        
        Args:
            session_id: UUID of the session
        
        Returns:
            Dictionary with complete session data or None if not found
        """
        result = await self.session.execute(
            select(ResearchSession)
            .options(
                selectinload(ResearchSession.queries)
                .selectinload(Query.search_results)
                .selectinload(SearchResult.crawled_content)
            )
            .options(
                selectinload(ResearchSession.queries)
                .selectinload(Query.summaries)
            )
            .where(ResearchSession.id == session_id)
        )
        
        research_session = result.scalar_one_or_none()
        
        if not research_session:
            return None
        
        # Build comprehensive summary
        return {
            "session": {
                "id": str(research_session.id),
                "question": research_session.original_question,
                "status": research_session.status.value,
                "created_at": research_session.created_at.isoformat(),
                "updated_at": research_session.updated_at.isoformat(),
                "metadata": research_session.metadata_,
            },
            "queries": [
                {
                    "id": str(query.id),
                    "text": query.query_text,
                    "type": query.query_type.value,
                    "categories": query.categories_used,
                    "created_at": query.created_at.isoformat(),
                    "num_results": len(query.search_results),
                    "num_summaries": len(query.summaries),
                }
                for query in research_session.queries
            ],
            "stats": {
                "total_queries": len(research_session.queries),
                "total_results": sum(len(q.search_results) for q in research_session.queries),
                "total_summaries": sum(len(q.summaries) for q in research_session.queries),
            }
        }
    
    async def search_sessions(
        self,
        search_term: str,
        limit: int = 10
    ) -> List[ResearchSession]:
        """
        Search sessions by question text.
        
        Args:
            search_term: Text to search for in questions
            limit: Maximum number of results
        
        Returns:
            List of matching ResearchSession instances
        """
        result = await self.session.execute(
            select(ResearchSession)
            .where(ResearchSession.original_question.ilike(f"%{search_term}%"))
            .order_by(ResearchSession.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    # ========================================================================
    # UPDATE Operations
    # ========================================================================
    
    async def update_session_status(
        self,
        session_id: UUID,
        status: SessionStatus
    ) -> Optional[ResearchSession]:
        """
        Update the status of a research session.
        
        Args:
            session_id: UUID of the session
            status: New status (ACTIVE, COMPLETED, ARCHIVED)
        
        Returns:
            Updated ResearchSession or None if not found
        """
        research_session = await self.get_session(session_id)
        
        if not research_session:
            return None
        
        research_session.status = status
        research_session.updated_at = datetime.utcnow()
        
        await self.session.flush()
        
        return research_session
    
    async def update_session_metadata(
        self,
        session_id: UUID,
        metadata: Dict[str, Any],
        merge: bool = True
    ) -> Optional[ResearchSession]:
        """
        Update session metadata.
        
        Args:
            session_id: UUID of the session
            metadata: New metadata dictionary
            merge: If True, merge with existing metadata; if False, replace
        
        Returns:
            Updated ResearchSession or None if not found
        """
        research_session = await self.get_session(session_id)
        
        if not research_session:
            return None
        
        if merge and research_session.metadata_:
            # Merge new metadata with existing
            research_session.metadata_.update(metadata)
        else:
            # Replace metadata
            research_session.metadata_ = metadata
        
        research_session.updated_at = datetime.utcnow()
        
        await self.session.flush()
        
        return research_session
    
    # ========================================================================
    # DELETE Operations
    # ========================================================================
    
    async def delete_session(self, session_id: UUID) -> bool:
        """
        Delete a research session and all related data (cascade).
        
        This will automatically delete all queries, search results,
        crawled content, and summaries associated with this session.
        
        Args:
            session_id: UUID of the session to delete
        
        Returns:
            True if deleted, False if not found
        """
        research_session = await self.get_session(session_id)
        
        if not research_session:
            return False
        
        await self.session.delete(research_session)
        await self.session.flush()
        
        return True
    
    async def archive_old_sessions(self, days: int = 30) -> int:
        """
        Archive sessions older than specified days.
        
        Args:
            days: Number of days after which to archive sessions
        
        Returns:
            Number of sessions archived
        """
        from datetime import timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        result = await self.session.execute(
            select(ResearchSession).where(
                and_(
                    ResearchSession.status == SessionStatus.COMPLETED,
                    ResearchSession.updated_at < cutoff_date
                )
            )
        )
        
        sessions = result.scalars().all()
        count = 0
        
        for session in sessions:
            session.status = SessionStatus.ARCHIVED
            session.updated_at = datetime.utcnow()
            count += 1
        
        await self.session.flush()
        
        return count
    
    # ========================================================================
    # STATISTICS & ANALYTICS
    # ========================================================================
    
    async def get_session_count(self, status: Optional[SessionStatus] = None) -> int:
        """
        Get total count of sessions, optionally filtered by status.
        
        Args:
            status: Filter by session status
        
        Returns:
            Number of sessions
        """
        query = select(func.count(ResearchSession.id))
        
        if status:
            query = query.where(ResearchSession.status == status)
        
        result = await self.session.execute(query)
        return result.scalar() or 0
    
    async def get_recent_sessions(self, limit: int = 10) -> List[ResearchSession]:
        """
        Get most recently created sessions.
        
        Args:
            limit: Maximum number of sessions to return
        
        Returns:
            List of recent ResearchSession instances
        """
        result = await self.session.execute(
            select(ResearchSession)
            .order_by(ResearchSession.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def exists(self, session_id: UUID) -> bool:
        """
        Check if a session exists.
        
        Args:
            session_id: UUID of the session
        
        Returns:
            True if exists, False otherwise
        """
        result = await self.session.execute(
            select(func.count(ResearchSession.id))
            .where(ResearchSession.id == session_id)
        )
        count = result.scalar()
        return count > 0
