"""
Repository for managing generated summaries.

Provides data access layer for Summary model with methods for creating,
retrieving summaries, and performing semantic similarity searches.
"""
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from uuid import UUID

from sqlalchemy import select, func, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Summary, SummaryType


class SummaryRepository:
    """Repository for summary operations."""
    
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
    
    async def create_summary(
        self,
        query_id: UUID,
        summary_text: str,
        summary_type: SummaryType,
        model_used: str,
        sources: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Summary:
        """
        Create a new summary.
        
        Args:
            query_id: UUID of the query this summary answers
            summary_text: The generated summary text
            summary_type: Type (INDIVIDUAL, COMBINED, FINAL)
            model_used: LLM model identifier (e.g., 'llama3.2:3b')
            sources: Dictionary with source URLs and relevance scores
            metadata: Additional metadata (tokens, temperature, etc.)
        
        Returns:
            Created Summary instance
        
        Example:
            summary = await repo.create_summary(
                query_id=query_id,
                summary_text="RAG is a technique that combines...",
                summary_type=SummaryType.FINAL,
                model_used="llama3.2:3b",
                sources={
                    "urls": [
                        {"url": "https://example.com/1", "relevance": 0.95},
                        {"url": "https://example.com/2", "relevance": 0.87}
                    ]
                },
                metadata={"tokens": 500, "temperature": 0.7}
            )
        """
        summary = Summary(
            query_id=query_id,
            summary_text=summary_text,
            summary_type=summary_type,
            model_used=model_used,
            sources=sources or {},
            metadata_=metadata or {}
        )
        
        self.session.add(summary)
        await self.session.flush()
        
        return summary
    
    # ========================================================================
    # READ Operations
    # ========================================================================
    
    async def get_summary(self, summary_id: UUID) -> Optional[Summary]:
        """
        Get a specific summary by ID.
        
        Args:
            summary_id: UUID of the summary
        
        Returns:
            Summary or None if not found
        """
        result = await self.session.execute(
            select(Summary).where(Summary.id == summary_id)
        )
        return result.scalar_one_or_none()
    
    async def get_summaries_by_query(
        self,
        query_id: UUID,
        summary_type: Optional[SummaryType] = None
    ) -> List[Summary]:
        """
        Get all summaries for a specific query.
        
        Args:
            query_id: UUID of the query
            summary_type: Optional filter by summary type
        
        Returns:
            List of Summary instances ordered by creation time
        """
        query = select(Summary).where(
            Summary.query_id == query_id
        ).order_by(Summary.created_at.asc())
        
        if summary_type:
            query = query.where(Summary.summary_type == summary_type)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_final_summary(self, query_id: UUID) -> Optional[Summary]:
        """
        Get the final summary for a query.
        
        Args:
            query_id: UUID of the query
        
        Returns:
            Final Summary or None if not found
        """
        result = await self.session.execute(
            select(Summary).where(
                and_(
                    Summary.query_id == query_id,
                    Summary.summary_type == SummaryType.FINAL
                )
            ).order_by(Summary.created_at.desc()).limit(1)
        )
        return result.scalar_one_or_none()
    
    async def get_final_summary_by_session(self, session_id: UUID) -> Optional[Summary]:
        """
        Get the final summary for a research session.
        
        Args:
            session_id: UUID of the research session
        
        Returns:
            Final Summary or None if not found
        """
        from ..models import Query
        
        result = await self.session.execute(
            select(Summary)
            .join(Query, Summary.query_id == Query.id)
            .where(
                and_(
                    Query.session_id == session_id,
                    Summary.summary_type == SummaryType.FINAL
                )
            )
            .order_by(Summary.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def get_summaries_by_model(
        self,
        model_used: str,
        limit: Optional[int] = None
    ) -> List[Summary]:
        """
        Get summaries generated by a specific model.
        
        Args:
            model_used: Model identifier
            limit: Optional maximum number of results
        
        Returns:
            List of Summary instances
        """
        query = select(Summary).where(
            Summary.model_used == model_used
        ).order_by(Summary.created_at.desc())
        
        if limit:
            query = query.limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_summaries_by_type(
        self,
        summary_type: SummaryType,
        limit: Optional[int] = None
    ) -> List[Summary]:
        """
        Get summaries of a specific type.
        
        Args:
            summary_type: Type (INDIVIDUAL, COMBINED, FINAL)
            limit: Optional maximum number of results
        
        Returns:
            List of Summary instances
        """
        query = select(Summary).where(
            Summary.summary_type == summary_type
        ).order_by(Summary.created_at.desc())
        
        if limit:
            query = query.limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def search_summaries(
        self,
        search_term: str,
        limit: int = 10
    ) -> List[Summary]:
        """
        Search summaries by text content.
        
        Args:
            search_term: Text to search for in summary text
            limit: Maximum number of results
        
        Returns:
            List of matching Summary instances
        """
        result = await self.session.execute(
            select(Summary)
            .where(Summary.summary_text.ilike(f"%{search_term}%"))
            .order_by(Summary.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    # ========================================================================
    # UPDATE Operations
    # ========================================================================
    
    async def add_embedding(
        self,
        summary_id: UUID,
        embedding: List[float]
    ) -> Optional[Summary]:
        """
        Add or update the embedding vector for a summary.
        
        Args:
            summary_id: UUID of the summary
            embedding: Vector embedding (384 dimensions)
        
        Returns:
            Updated Summary or None if not found
        """
        summary = await self.get_summary(summary_id)
        
        if not summary:
            return None
        
        summary.embedding = embedding
        await self.session.flush()
        
        return summary
    
    async def update_summary(
        self,
        summary_id: UUID,
        new_text: str
    ) -> Optional[Summary]:
        """
        Update the text of a summary.
        
        Useful for refining summaries as research progresses.
        
        Args:
            summary_id: UUID of the summary
            new_text: New summary text
        
        Returns:
            Updated Summary or None if not found
        """
        summary = await self.get_summary(summary_id)
        
        if not summary:
            return None
        
        summary.summary_text = new_text
        await self.session.flush()
        
        return summary
    
    async def update_summary_sources(
        self,
        summary_id: UUID,
        sources: Dict[str, Any],
        merge: bool = True
    ) -> Optional[Summary]:
        """
        Update the sources for a summary.
        
        Args:
            summary_id: UUID of the summary
            sources: New sources dictionary
            merge: If True, merge with existing; if False, replace
        
        Returns:
            Updated Summary or None if not found
        """
        summary = await self.get_summary(summary_id)
        
        if not summary:
            return None
        
        if merge and summary.sources:
            summary.sources.update(sources)
        else:
            summary.sources = sources
        
        await self.session.flush()
        
        return summary
    
    async def update_summary_metadata(
        self,
        summary_id: UUID,
        metadata: Dict[str, Any],
        merge: bool = True
    ) -> Optional[Summary]:
        """
        Update summary metadata.
        
        Args:
            summary_id: UUID of the summary
            metadata: New metadata dictionary
            merge: If True, merge with existing; if False, replace
        
        Returns:
            Updated Summary or None if not found
        """
        summary = await self.get_summary(summary_id)
        
        if not summary:
            return None
        
        if merge and summary.metadata_:
            summary.metadata_.update(metadata)
        else:
            summary.metadata_ = metadata
        
        await self.session.flush()
        
        return summary
    
    # ========================================================================
    # DELETE Operations
    # ========================================================================
    
    async def delete_summary(self, summary_id: UUID) -> bool:
        """
        Delete a summary.
        
        Args:
            summary_id: UUID of the summary to delete
        
        Returns:
            True if deleted, False if not found
        """
        summary = await self.get_summary(summary_id)
        
        if not summary:
            return False
        
        await self.session.delete(summary)
        await self.session.flush()
        
        return True
    
    async def delete_summaries_by_query(self, query_id: UUID) -> int:
        """
        Delete all summaries for a query.
        
        Args:
            query_id: UUID of the query
        
        Returns:
            Number of summaries deleted
        """
        summaries = await self.get_summaries_by_query(query_id)
        count = len(summaries)
        
        for summary in summaries:
            await self.session.delete(summary)
        
        await self.session.flush()
        
        return count
    
    # ========================================================================
    # VECTOR SIMILARITY SEARCH
    # ========================================================================
    
    async def find_similar_summaries(
        self,
        embedding: List[float],
        limit: int = 5,
        similarity_threshold: Optional[float] = None,
        summary_type: Optional[SummaryType] = None,
        exclude_query_id: Optional[UUID] = None
    ) -> List[Tuple[Summary, float]]:
        """
        Find semantically similar summaries using vector similarity search.
        
        Useful for finding related past research to avoid duplicate work.
        
        Args:
            embedding: Query embedding vector
            limit: Maximum number of results
            similarity_threshold: Optional minimum similarity score
            summary_type: Optional filter by summary type
            exclude_query_id: Optional query ID to exclude
        
        Returns:
            List of tuples (Summary, similarity_score) ordered by similarity
        
        Example:
            similar = await repo.find_similar_summaries(
                question_embedding,
                limit=5,
                similarity_threshold=0.8,
                summary_type=SummaryType.FINAL
            )
            for summary, score in similar:
                print(f"Similar research: {score:.2f}")
        """
        # Build query with cosine similarity
        query = select(
            Summary,
            (1 - (Summary.embedding.cosine_distance(embedding) / 2)).label('similarity')
        ).where(
            Summary.embedding.isnot(None)
        ).order_by(
            Summary.embedding.cosine_distance(embedding)
        ).limit(limit)
        
        # Optional filters
        if summary_type:
            query = query.where(Summary.summary_type == summary_type)
        
        if exclude_query_id:
            query = query.where(Summary.query_id != exclude_query_id)
        
        result = await self.session.execute(query)
        rows = result.all()
        
        # Filter by similarity threshold if provided
        if similarity_threshold:
            rows = [(s, sim) for s, sim in rows if sim >= similarity_threshold]
        
        return rows
    
    async def find_similar_summaries_by_summary_id(
        self,
        summary_id: UUID,
        limit: int = 5,
        similarity_threshold: Optional[float] = None
    ) -> List[Tuple[Summary, float]]:
        """
        Find summaries similar to a specific summary.
        
        Args:
            summary_id: UUID of the summary to find similar ones for
            limit: Maximum number of results
            similarity_threshold: Optional minimum similarity
        
        Returns:
            List of tuples (Summary, similarity_score)
        """
        source_summary = await self.get_summary(summary_id)
        
        if not source_summary or not source_summary.embedding:
            return []
        
        # Find similar summaries, excluding the source
        return await self.find_similar_summaries(
            source_summary.embedding,
            limit=limit,
            similarity_threshold=similarity_threshold,
            exclude_query_id=source_summary.query_id
        )
    
    # ========================================================================
    # STATISTICS & ANALYTICS
    # ========================================================================
    
    async def get_summary_count(
        self,
        summary_type: Optional[SummaryType] = None,
        model_used: Optional[str] = None
    ) -> int:
        """
        Get count of summaries, optionally filtered.
        
        Args:
            summary_type: Filter by summary type
            model_used: Filter by model
        
        Returns:
            Number of summaries
        """
        query = select(func.count(Summary.id))
        
        conditions = []
        if summary_type:
            conditions.append(Summary.summary_type == summary_type)
        if model_used:
            conditions.append(Summary.model_used == model_used)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        result = await self.session.execute(query)
        return result.scalar() or 0
    
    async def get_summaries_with_embeddings_count(self) -> int:
        """
        Get count of summaries that have embeddings.
        
        Returns:
            Number of summaries with embeddings
        """
        result = await self.session.execute(
            select(func.count(Summary.id)).where(
                Summary.embedding.isnot(None)
            )
        )
        return result.scalar() or 0
    
    async def get_summaries_without_embeddings(
        self,
        limit: Optional[int] = None
    ) -> List[Summary]:
        """
        Get summaries that don't have embeddings yet.
        
        Useful for batch embedding generation.
        
        Args:
            limit: Maximum number of summaries to return
        
        Returns:
            List of Summary instances without embeddings
        """
        query = select(Summary).where(
            Summary.embedding.is_(None)
        ).order_by(Summary.created_at.desc())
        
        if limit:
            query = query.limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_model_usage_stats(self) -> List[Tuple[str, int]]:
        """
        Get statistics on model usage.
        
        Returns:
            List of tuples (model_name, usage_count) ordered by usage
        """
        result = await self.session.execute(
            select(
                Summary.model_used,
                func.count(Summary.id).label('count')
            )
            .group_by(Summary.model_used)
            .order_by(desc('count'))
        )
        
        return [(row[0], row[1]) for row in result.all()]
    
    async def get_summary_type_stats(self) -> Dict[str, int]:
        """
        Get count of summaries by type.
        
        Returns:
            Dictionary mapping summary type to count
        """
        result = await self.session.execute(
            select(
                Summary.summary_type,
                func.count(Summary.id).label('count')
            )
            .group_by(Summary.summary_type)
        )
        
        return {row[0].value: row[1] for row in result.all()}
    
    async def get_average_summary_length(
        self,
        summary_type: Optional[SummaryType] = None
    ) -> float:
        """
        Get average length of summaries.
        
        Args:
            summary_type: Optional filter by summary type
        
        Returns:
            Average character length
        """
        query = select(func.avg(func.length(Summary.summary_text)))
        
        if summary_type:
            query = query.where(Summary.summary_type == summary_type)
        
        result = await self.session.execute(query)
        return result.scalar() or 0.0
    
    async def get_recent_summaries(
        self,
        limit: int = 10,
        summary_type: Optional[SummaryType] = None
    ) -> List[Summary]:
        """
        Get most recently created summaries.
        
        Args:
            limit: Maximum number of summaries to return
            summary_type: Optional filter by summary type
        
        Returns:
            List of recent Summary instances
        """
        query = select(Summary).order_by(Summary.created_at.desc()).limit(limit)
        
        if summary_type:
            query = query.where(Summary.summary_type == summary_type)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about summaries.
        
        Returns:
            Dictionary with various statistics
        """
        total_count = await self.get_summary_count()
        embedded_count = await self.get_summaries_with_embeddings_count()
        type_stats = await self.get_summary_type_stats()
        model_stats = await self.get_model_usage_stats()
        avg_length = await self.get_average_summary_length()
        
        return {
            "total_summaries": total_count,
            "with_embeddings": embedded_count,
            "without_embeddings": total_count - embedded_count,
            "embedding_coverage": (embedded_count / total_count * 100) if total_count > 0 else 0,
            "by_type": type_stats,
            "by_model": dict(model_stats),
            "average_length": int(avg_length),
        }
    
    async def exists(self, summary_id: UUID) -> bool:
        """
        Check if a summary exists.
        
        Args:
            summary_id: UUID of the summary
        
        Returns:
            True if exists, False otherwise
        """
        result = await self.session.execute(
            select(func.count(Summary.id)).where(Summary.id == summary_id)
        )
        count = result.scalar()
        return count > 0
