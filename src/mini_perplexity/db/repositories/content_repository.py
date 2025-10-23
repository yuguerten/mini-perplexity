"""
Repository for managing crawled web content.

Provides data access layer for CrawledContent model with methods for storing,
retrieving, deduplicating content, and performing semantic similarity searches.
"""
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from uuid import UUID
import hashlib

from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import CrawledContent


class ContentRepository:
    """Repository for crawled content operations."""
    
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
    
    async def add_content(
        self,
        url: str,
        content: str,
        search_result_id: UUID,
        markdown: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CrawledContent:
        """
        Add crawled content to the database.
        
        Automatically generates content hash for deduplication.
        
        Args:
            url: The URL that was crawled
            content: Raw text content
            search_result_id: UUID of the search result this came from
            markdown: Optional markdown-formatted content
            metadata: Additional metadata (status code, headers, duration, etc.)
        
        Returns:
            Created CrawledContent instance
        
        Example:
            content = await repo.add_content(
                url="https://example.com/article",
                content="Article text...",
                search_result_id=result_id,
                markdown="# Article\\n\\nText...",
                metadata={"status_code": 200, "duration_ms": 1250}
            )
        """
        # Generate content hash for deduplication
        content_hash = self._generate_content_hash(content)
        
        crawled_content = CrawledContent(
            url=url,
            content=content,
            markdown=markdown,
            search_result_id=search_result_id,
            content_hash=content_hash,
            metadata_=metadata or {}
        )
        
        self.session.add(crawled_content)
        await self.session.flush()
        
        return crawled_content
    
    async def batch_add_content(
        self,
        contents: List[Dict[str, Any]]
    ) -> List[CrawledContent]:
        """
        Add multiple crawled contents efficiently in a batch.
        
        Args:
            contents: List of dicts with keys: url, content, search_result_id,
                     markdown (optional), metadata (optional)
        
        Returns:
            List of created CrawledContent instances
        
        Example:
            contents = [
                {
                    "url": "https://example.com/1",
                    "content": "Content 1",
                    "search_result_id": result_id_1,
                    "markdown": "# Article 1"
                },
                {
                    "url": "https://example.com/2",
                    "content": "Content 2",
                    "search_result_id": result_id_2
                }
            ]
            added = await repo.batch_add_content(contents)
        """
        crawled_contents = []
        
        for content_data in contents:
            content_hash = self._generate_content_hash(content_data["content"])
            
            crawled_content = CrawledContent(
                url=content_data["url"],
                content=content_data["content"],
                markdown=content_data.get("markdown"),
                search_result_id=content_data["search_result_id"],
                content_hash=content_hash,
                metadata_=content_data.get("metadata", {})
            )
            
            self.session.add(crawled_content)
            crawled_contents.append(crawled_content)
        
        await self.session.flush()
        
        return crawled_contents
    
    # ========================================================================
    # READ Operations
    # ========================================================================
    
    async def get_content(self, content_id: UUID) -> Optional[CrawledContent]:
        """
        Get crawled content by ID.
        
        Args:
            content_id: UUID of the content
        
        Returns:
            CrawledContent or None if not found
        """
        result = await self.session.execute(
            select(CrawledContent).where(CrawledContent.id == content_id)
        )
        return result.scalar_one_or_none()
    
    async def get_content_by_url(self, url: str) -> Optional[CrawledContent]:
        """
        Get crawled content by URL.
        
        Useful for checking if a URL has already been crawled.
        
        Args:
            url: The URL to look up
        
        Returns:
            CrawledContent or None if not found
        """
        result = await self.session.execute(
            select(CrawledContent).where(CrawledContent.url == url)
        )
        return result.scalar_one_or_none()
    
    async def get_content_by_search_result(
        self,
        search_result_id: UUID
    ) -> Optional[CrawledContent]:
        """
        Get content associated with a specific search result.
        
        Args:
            search_result_id: UUID of the search result
        
        Returns:
            CrawledContent or None if not found
        """
        result = await self.session.execute(
            select(CrawledContent).where(
                CrawledContent.search_result_id == search_result_id
            )
        )
        return result.scalar_one_or_none()
    
    async def get_multiple_by_urls(self, urls: List[str]) -> List[CrawledContent]:
        """
        Get multiple contents by URLs.
        
        Args:
            urls: List of URLs to retrieve
        
        Returns:
            List of CrawledContent instances found
        """
        result = await self.session.execute(
            select(CrawledContent).where(CrawledContent.url.in_(urls))
        )
        return list(result.scalars().all())
    
    async def url_exists(self, url: str) -> bool:
        """
        Check if a URL has already been crawled.
        
        Args:
            url: The URL to check
        
        Returns:
            True if crawled, False otherwise
        """
        result = await self.session.execute(
            select(func.count(CrawledContent.id)).where(CrawledContent.url == url)
        )
        count = result.scalar()
        return count > 0
    
    async def get_uncrawled_urls(
        self,
        search_result_ids: List[UUID]
    ) -> List[UUID]:
        """
        Find which search results haven't been crawled yet.
        
        Args:
            search_result_ids: List of search result UUIDs to check
        
        Returns:
            List of search result IDs that don't have crawled content
        """
        from ..models import SearchResult
        
        # Get search result IDs that already have content
        crawled_result = await self.session.execute(
            select(CrawledContent.search_result_id).where(
                CrawledContent.search_result_id.in_(search_result_ids)
            )
        )
        crawled_ids = set(row[0] for row in crawled_result.all())
        
        # Return the ones that haven't been crawled
        uncrawled_ids = [
            result_id for result_id in search_result_ids 
            if result_id not in crawled_ids
        ]
        
        return uncrawled_ids
    
    # ========================================================================
    # UPDATE Operations
    # ========================================================================
    
    async def add_embedding(
        self,
        content_id: UUID,
        embedding: List[float]
    ) -> Optional[CrawledContent]:
        """
        Add or update the embedding vector for content.
        
        Args:
            content_id: UUID of the content
            embedding: Vector embedding (384 dimensions for all-MiniLM-L6-v2)
        
        Returns:
            Updated CrawledContent or None if not found
        """
        content = await self.get_content(content_id)
        
        if not content:
            return None
        
        content.embedding = embedding
        await self.session.flush()
        
        return content
    
    async def update_content_metadata(
        self,
        content_id: UUID,
        metadata: Dict[str, Any],
        merge: bool = True
    ) -> Optional[CrawledContent]:
        """
        Update content metadata.
        
        Args:
            content_id: UUID of the content
            metadata: New metadata dictionary
            merge: If True, merge with existing; if False, replace
        
        Returns:
            Updated CrawledContent or None if not found
        """
        content = await self.get_content(content_id)
        
        if not content:
            return None
        
        if merge and content.metadata_:
            content.metadata_.update(metadata)
        else:
            content.metadata_ = metadata
        
        await self.session.flush()
        
        return content
    
    # ========================================================================
    # DELETE Operations
    # ========================================================================
    
    async def delete_content(self, content_id: UUID) -> bool:
        """
        Delete crawled content.
        
        Args:
            content_id: UUID of the content to delete
        
        Returns:
            True if deleted, False if not found
        """
        content = await self.get_content(content_id)
        
        if not content:
            return False
        
        await self.session.delete(content)
        await self.session.flush()
        
        return True
    
    async def delete_by_url(self, url: str) -> bool:
        """
        Delete content by URL.
        
        Args:
            url: The URL to delete
        
        Returns:
            True if deleted, False if not found
        """
        content = await self.get_content_by_url(url)
        
        if not content:
            return False
        
        await self.session.delete(content)
        await self.session.flush()
        
        return True
    
    # ========================================================================
    # DEDUPLICATION
    # ========================================================================
    
    async def find_duplicate_content(
        self,
        content_hash: str
    ) -> List[CrawledContent]:
        """
        Find content with the same hash (duplicate content).
        
        Args:
            content_hash: SHA256 hash of content
        
        Returns:
            List of CrawledContent instances with matching hash
        """
        result = await self.session.execute(
            select(CrawledContent).where(CrawledContent.content_hash == content_hash)
        )
        return list(result.scalars().all())
    
    async def find_duplicate_content_by_text(
        self,
        content: str
    ) -> List[CrawledContent]:
        """
        Find duplicate content by generating hash from text.
        
        Args:
            content: The content text to check
        
        Returns:
            List of CrawledContent instances with same content
        """
        content_hash = self._generate_content_hash(content)
        return await self.find_duplicate_content(content_hash)
    
    async def get_all_duplicates(self) -> List[Tuple[str, int, List[CrawledContent]]]:
        """
        Find all duplicate content in the database.
        
        Returns:
            List of tuples (content_hash, count, list_of_contents)
            Only includes hashes that appear more than once
        """
        # Find content hashes that appear more than once
        duplicate_hashes_result = await self.session.execute(
            select(
                CrawledContent.content_hash,
                func.count(CrawledContent.id).label('count')
            )
            .group_by(CrawledContent.content_hash)
            .having(func.count(CrawledContent.id) > 1)
        )
        
        duplicates = []
        
        for hash_value, count in duplicate_hashes_result.all():
            # Get all content with this hash
            contents = await self.find_duplicate_content(hash_value)
            duplicates.append((hash_value, count, contents))
        
        return duplicates
    
    # ========================================================================
    # VECTOR SIMILARITY SEARCH
    # ========================================================================
    
    async def find_similar_content(
        self,
        embedding: List[float],
        limit: int = 10,
        similarity_threshold: Optional[float] = None,
        exclude_urls: Optional[List[str]] = None
    ) -> List[Tuple[CrawledContent, float]]:
        """
        Find semantically similar content using vector similarity search.
        
        Critical for RAG - finds most relevant content for answering questions.
        
        Args:
            embedding: Query embedding vector
            limit: Maximum number of results
            similarity_threshold: Optional minimum similarity score
            exclude_urls: Optional URLs to exclude from results
        
        Returns:
            List of tuples (CrawledContent, similarity_score) ordered by similarity
        
        Example:
            # Find content relevant to a question
            similar = await repo.find_similar_content(
                question_embedding,
                limit=5,
                similarity_threshold=0.7
            )
            for content, score in similar:
                print(f"{content.url}: {score:.2f}")
        """
        # Build query with cosine similarity
        query = select(
            CrawledContent,
            (1 - (CrawledContent.embedding.cosine_distance(embedding) / 2)).label('similarity')
        ).where(
            CrawledContent.embedding.isnot(None)
        ).order_by(
            CrawledContent.embedding.cosine_distance(embedding)
        ).limit(limit)
        
        # Optionally exclude specific URLs
        if exclude_urls:
            query = query.where(CrawledContent.url.notin_(exclude_urls))
        
        result = await self.session.execute(query)
        rows = result.all()
        
        # Filter by similarity threshold if provided
        if similarity_threshold:
            rows = [(c, sim) for c, sim in rows if sim >= similarity_threshold]
        
        return rows
    
    async def find_similar_content_by_content_id(
        self,
        content_id: UUID,
        limit: int = 10,
        similarity_threshold: Optional[float] = None
    ) -> List[Tuple[CrawledContent, float]]:
        """
        Find content similar to a specific piece of content.
        
        Args:
            content_id: UUID of the content to find similar ones for
            limit: Maximum number of results
            similarity_threshold: Optional minimum similarity
        
        Returns:
            List of tuples (CrawledContent, similarity_score)
        """
        source_content = await self.get_content(content_id)
        
        if not source_content or not source_content.embedding:
            return []
        
        # Find similar content, excluding the source
        return await self.find_similar_content(
            source_content.embedding,
            limit=limit,
            similarity_threshold=similarity_threshold,
            exclude_urls=[source_content.url]
        )
    
    # ========================================================================
    # STATISTICS & ANALYTICS
    # ========================================================================
    
    async def get_content_count(self) -> int:
        """
        Get total count of crawled content.
        
        Returns:
            Number of content items
        """
        result = await self.session.execute(
            select(func.count(CrawledContent.id))
        )
        return result.scalar() or 0
    
    async def get_content_with_embeddings_count(self) -> int:
        """
        Get count of content that has embeddings.
        
        Returns:
            Number of content items with embeddings
        """
        result = await self.session.execute(
            select(func.count(CrawledContent.id)).where(
                CrawledContent.embedding.isnot(None)
            )
        )
        return result.scalar() or 0
    
    async def get_content_without_embeddings(
        self,
        limit: Optional[int] = None
    ) -> List[CrawledContent]:
        """
        Get content that doesn't have embeddings yet.
        
        Useful for batch embedding generation.
        
        Args:
            limit: Maximum number of items to return
        
        Returns:
            List of CrawledContent instances without embeddings
        """
        query = select(CrawledContent).where(
            CrawledContent.embedding.is_(None)
        ).order_by(CrawledContent.crawled_at.desc())
        
        if limit:
            query = query.limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_recent_content(
        self,
        limit: int = 10
    ) -> List[CrawledContent]:
        """
        Get most recently crawled content.
        
        Args:
            limit: Maximum number of items to return
        
        Returns:
            List of recent CrawledContent instances
        """
        result = await self.session.execute(
            select(CrawledContent)
            .order_by(CrawledContent.crawled_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_content_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about crawled content.
        
        Returns:
            Dictionary with various statistics
        """
        total_count = await self.get_content_count()
        embedded_count = await self.get_content_with_embeddings_count()
        
        # Get average content length
        avg_length_result = await self.session.execute(
            select(func.avg(func.length(CrawledContent.content)))
        )
        avg_length = avg_length_result.scalar() or 0
        
        # Get duplicate count
        duplicate_count_result = await self.session.execute(
            select(func.count(func.distinct(CrawledContent.content_hash)))
        )
        unique_hashes = duplicate_count_result.scalar() or 0
        
        return {
            "total_content": total_count,
            "with_embeddings": embedded_count,
            "without_embeddings": total_count - embedded_count,
            "embedding_coverage": (embedded_count / total_count * 100) if total_count > 0 else 0,
            "average_content_length": int(avg_length),
            "unique_content": unique_hashes,
            "duplicate_content": total_count - unique_hashes,
        }
    
    async def exists(self, content_id: UUID) -> bool:
        """
        Check if content exists.
        
        Args:
            content_id: UUID of the content
        
        Returns:
            True if exists, False otherwise
        """
        result = await self.session.execute(
            select(func.count(CrawledContent.id)).where(
                CrawledContent.id == content_id
            )
        )
        count = result.scalar()
        return count > 0
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    @staticmethod
    def _generate_content_hash(content: str) -> str:
        """
        Generate SHA256 hash of content for deduplication.
        
        Args:
            content: The content text
        
        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
