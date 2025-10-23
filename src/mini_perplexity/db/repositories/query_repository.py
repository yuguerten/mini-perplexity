"""
Repository for managing queries within research sessions.

Provides data access layer for Query model with methods for creating,
reading, updating queries, and performing semantic similarity searches.
"""
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from uuid import UUID

from sqlalchemy import select, func, and_, desc
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Query, QueryType, SearchResult, Summary


class QueryRepository:
    """Repository for query operations."""
    
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
    
    async def create_query(
        self,
        session_id: UUID,
        query_text: str,
        query_type: QueryType = QueryType.INITIAL,
        categories_used: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Query:
        """
        Create a new query within a research session.
        
        Args:
            session_id: UUID of the parent research session
            query_text: The actual query text
            query_type: Type of query (INITIAL, CLARIFICATION, FOLLOW_UP)
            categories_used: SearXNG categories used for this query
            metadata: Additional metadata (analysis results, optimization info)
        
        Returns:
            Created Query instance
        
        Example:
            query = await repo.create_query(
                session_id=session_id,
                query_text="What is RAG in AI?",
                query_type=QueryType.INITIAL,
                categories_used=["general", "science"],
                metadata={"intent": "technical", "confidence": 0.95}
            )
        """
        query = Query(
            session_id=session_id,
            query_text=query_text,
            query_type=query_type,
            categories_used=categories_used,
            metadata_=metadata or {}
        )
        
        self.session.add(query)
        await self.session.flush()
        
        return query
    
    # ========================================================================
    # READ Operations
    # ========================================================================
    
    async def get_query(self, query_id: UUID) -> Optional[Query]:
        """
        Get a specific query by ID.
        
        Args:
            query_id: UUID of the query
        
        Returns:
            Query or None if not found
        """
        result = await self.session.execute(
            select(Query).where(Query.id == query_id)
        )
        return result.scalar_one_or_none()
    
    async def get_queries_by_session(
        self,
        session_id: UUID,
        query_type: Optional[QueryType] = None
    ) -> List[Query]:
        """
        Get all queries for a specific research session.
        
        Args:
            session_id: UUID of the research session
            query_type: Optional filter by query type
        
        Returns:
            List of Query instances ordered by creation time
        """
        query = select(Query).where(
            Query.session_id == session_id
        ).order_by(Query.created_at.asc())
        
        if query_type:
            query = query.where(Query.query_type == query_type)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_query_with_results(self, query_id: UUID) -> Optional[Query]:
        """
        Get a query with all its search results eagerly loaded.
        
        Args:
            query_id: UUID of the query
        
        Returns:
            Query with search_results loaded, or None if not found
        """
        result = await self.session.execute(
            select(Query)
            .options(selectinload(Query.search_results))
            .where(Query.id == query_id)
        )
        return result.scalar_one_or_none()
    
    async def get_query_with_summaries(self, query_id: UUID) -> Optional[Query]:
        """
        Get a query with all its summaries eagerly loaded.
        
        Args:
            query_id: UUID of the query
        
        Returns:
            Query with summaries loaded, or None if not found
        """
        result = await self.session.execute(
            select(Query)
            .options(selectinload(Query.summaries))
            .where(Query.id == query_id)
        )
        return result.scalar_one_or_none()
    
    async def get_query_full(self, query_id: UUID) -> Optional[Query]:
        """
        Get a query with all related data loaded (results, summaries).
        
        Args:
            query_id: UUID of the query
        
        Returns:
            Query with all relationships loaded, or None if not found
        """
        result = await self.session.execute(
            select(Query)
            .options(
                selectinload(Query.search_results),
                selectinload(Query.summaries)
            )
            .where(Query.id == query_id)
        )
        return result.scalar_one_or_none()
    
    async def get_initial_query(self, session_id: UUID) -> Optional[Query]:
        """
        Get the initial query for a session.
        
        Args:
            session_id: UUID of the research session
        
        Returns:
            Initial Query or None if not found
        """
        result = await self.session.execute(
            select(Query).where(
                and_(
                    Query.session_id == session_id,
                    Query.query_type == QueryType.INITIAL
                )
            ).order_by(Query.created_at.asc()).limit(1)
        )
        return result.scalar_one_or_none()
    
    async def search_queries(
        self,
        search_term: str,
        limit: int = 10
    ) -> List[Query]:
        """
        Search queries by text.
        
        Args:
            search_term: Text to search for in query text
            limit: Maximum number of results
        
        Returns:
            List of matching Query instances
        """
        result = await self.session.execute(
            select(Query)
            .where(Query.query_text.ilike(f"%{search_term}%"))
            .order_by(Query.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    # ========================================================================
    # UPDATE Operations
    # ========================================================================
    
    async def add_embedding(
        self,
        query_id: UUID,
        embedding: List[float]
    ) -> Optional[Query]:
        """
        Add or update the embedding vector for a query.
        
        Args:
            query_id: UUID of the query
            embedding: Vector embedding (384 dimensions for all-MiniLM-L6-v2)
        
        Returns:
            Updated Query or None if not found
        
        Example:
            query = await repo.add_embedding(
                query_id,
                [0.123, 0.456, ...]  # 384-dimensional vector
            )
        """
        query = await self.get_query(query_id)
        
        if not query:
            return None
        
        query.embedding = embedding
        await self.session.flush()
        
        return query
    
    async def update_query_categories(
        self,
        query_id: UUID,
        categories: List[str]
    ) -> Optional[Query]:
        """
        Update the SearXNG categories used for a query.
        
        Args:
            query_id: UUID of the query
            categories: List of category names
        
        Returns:
            Updated Query or None if not found
        """
        query = await self.get_query(query_id)
        
        if not query:
            return None
        
        query.categories_used = categories
        await self.session.flush()
        
        return query
    
    async def update_query_metadata(
        self,
        query_id: UUID,
        metadata: Dict[str, Any],
        merge: bool = True
    ) -> Optional[Query]:
        """
        Update query metadata.
        
        Args:
            query_id: UUID of the query
            metadata: New metadata dictionary
            merge: If True, merge with existing metadata; if False, replace
        
        Returns:
            Updated Query or None if not found
        """
        query = await self.get_query(query_id)
        
        if not query:
            return None
        
        if merge and query.metadata_:
            query.metadata_.update(metadata)
        else:
            query.metadata_ = metadata
        
        await self.session.flush()
        
        return query
    
    # ========================================================================
    # DELETE Operations
    # ========================================================================
    
    async def delete_query(self, query_id: UUID) -> bool:
        """
        Delete a query and all related data (cascade).
        
        This will automatically delete all search results and summaries
        associated with this query.
        
        Args:
            query_id: UUID of the query to delete
        
        Returns:
            True if deleted, False if not found
        """
        query = await self.get_query(query_id)
        
        if not query:
            return False
        
        await self.session.delete(query)
        await self.session.flush()
        
        return True
    
    # ========================================================================
    # VECTOR SIMILARITY SEARCH
    # ========================================================================
    
    async def find_similar_queries(
        self,
        embedding: List[float],
        limit: int = 5,
        similarity_threshold: Optional[float] = None,
        exclude_session_id: Optional[UUID] = None
    ) -> List[Tuple[Query, float]]:
        """
        Find semantically similar queries using vector similarity search.
        
        Uses cosine distance to find queries with similar embeddings.
        Lower distance = more similar (0 = identical).
        
        Args:
            embedding: Query embedding vector to search with
            limit: Maximum number of results to return
            similarity_threshold: Optional minimum similarity (1 - distance)
            exclude_session_id: Optional session ID to exclude from results
        
        Returns:
            List of tuples (Query, similarity_score) ordered by similarity
        
        Example:
            similar = await repo.find_similar_queries(
                query_embedding,
                limit=5,
                similarity_threshold=0.8
            )
            for query, score in similar:
                print(f"{query.query_text}: {score:.2f}")
        """
        # Build base query with cosine distance
        # Note: embedding <=> other_embedding returns distance (0 = identical, 2 = opposite)
        # Similarity = 1 - (distance / 2)
        query = select(
            Query,
            (1 - (Query.embedding.cosine_distance(embedding) / 2)).label('similarity')
        ).where(
            Query.embedding.isnot(None)  # Only queries with embeddings
        ).order_by(
            Query.embedding.cosine_distance(embedding)
        ).limit(limit)
        
        # Optionally exclude queries from a specific session
        if exclude_session_id:
            query = query.where(Query.session_id != exclude_session_id)
        
        # Execute query
        result = await self.session.execute(query)
        rows = result.all()
        
        # Filter by similarity threshold if provided
        if similarity_threshold:
            rows = [(q, sim) for q, sim in rows if sim >= similarity_threshold]
        
        return rows
    
    async def find_similar_queries_by_query_id(
        self,
        query_id: UUID,
        limit: int = 5,
        similarity_threshold: Optional[float] = None
    ) -> List[Tuple[Query, float]]:
        """
        Find queries similar to a specific query.
        
        Args:
            query_id: UUID of the query to find similar ones for
            limit: Maximum number of results
            similarity_threshold: Optional minimum similarity
        
        Returns:
            List of tuples (Query, similarity_score)
        """
        # Get the source query
        source_query = await self.get_query(query_id)
        
        if not source_query or not source_query.embedding:
            return []
        
        # Find similar queries, excluding the source query's session
        return await self.find_similar_queries(
            source_query.embedding,
            limit=limit,
            similarity_threshold=similarity_threshold,
            exclude_session_id=source_query.session_id
        )
    
    # ========================================================================
    # STATISTICS & ANALYTICS
    # ========================================================================
    
    async def get_query_count(
        self,
        session_id: Optional[UUID] = None,
        query_type: Optional[QueryType] = None
    ) -> int:
        """
        Get count of queries, optionally filtered.
        
        Args:
            session_id: Filter by session
            query_type: Filter by query type
        
        Returns:
            Number of queries
        """
        query = select(func.count(Query.id))
        
        conditions = []
        if session_id:
            conditions.append(Query.session_id == session_id)
        if query_type:
            conditions.append(Query.query_type == query_type)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        result = await self.session.execute(query)
        return result.scalar() or 0
    
    async def get_queries_with_embeddings_count(self) -> int:
        """
        Get count of queries that have embeddings.
        
        Returns:
            Number of queries with embeddings
        """
        result = await self.session.execute(
            select(func.count(Query.id)).where(Query.embedding.isnot(None))
        )
        return result.scalar() or 0
    
    async def get_queries_without_embeddings(
        self,
        limit: Optional[int] = None
    ) -> List[Query]:
        """
        Get queries that don't have embeddings yet.
        
        Useful for batch embedding generation.
        
        Args:
            limit: Maximum number of queries to return
        
        Returns:
            List of Query instances without embeddings
        """
        query = select(Query).where(
            Query.embedding.is_(None)
        ).order_by(Query.created_at.desc())
        
        if limit:
            query = query.limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_popular_categories(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get most frequently used SearXNG categories.
        
        Args:
            limit: Maximum number of categories to return
        
        Returns:
            List of tuples (category_name, usage_count)
        """
        # This requires unnesting the ARRAY and grouping
        # PostgreSQL specific query
        from sqlalchemy import text
        
        query_text = """
        SELECT category, COUNT(*) as count
        FROM queries, unnest(categories_used) as category
        WHERE categories_used IS NOT NULL
        GROUP BY category
        ORDER BY count DESC
        LIMIT :limit
        """
        
        result = await self.session.execute(
            text(query_text),
            {"limit": limit}
        )
        
        return [(row[0], row[1]) for row in result.all()]
    
    async def get_recent_queries(
        self,
        limit: int = 10,
        query_type: Optional[QueryType] = None
    ) -> List[Query]:
        """
        Get most recently created queries.
        
        Args:
            limit: Maximum number of queries to return
            query_type: Optional filter by query type
        
        Returns:
            List of recent Query instances
        """
        query = select(Query).order_by(Query.created_at.desc()).limit(limit)
        
        if query_type:
            query = query.where(Query.query_type == query_type)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def exists(self, query_id: UUID) -> bool:
        """
        Check if a query exists.
        
        Args:
            query_id: UUID of the query
        
        Returns:
            True if exists, False otherwise
        """
        result = await self.session.execute(
            select(func.count(Query.id)).where(Query.id == query_id)
        )
        count = result.scalar()
        return count > 0
