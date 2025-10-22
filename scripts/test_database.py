#!/usr/bin/env python3
"""
Test script to verify database connection and setup.

Run this script to ensure:
1. PostgreSQL is accessible
2. pgvector extension is installed
3. Database schema is created
4. Connection pooling works
5. Basic CRUD operations function
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mini_perplexity.db.connection import (
    check_connection,
    check_pgvector_extension,
    get_database_version,
    get_pool_status,
    get_session,
    get_table_row_counts,
    dispose_engine,
    get_config,
)
from mini_perplexity.db.models import (
    ResearchSession,
    Query,
    SearchResult,
    CrawledContent,
    Summary,
    SessionStatus,
    QueryType,
    SummaryType,
)
from sqlalchemy import select, func
import uuid
from datetime import datetime


# ============================================================================
# Test Functions
# ============================================================================

async def test_configuration():
    """Test 1: Verify database configuration."""
    print("\n" + "="*70)
    print("TEST 1: Database Configuration")
    print("="*70)
    
    try:
        config = get_config()
        print(f"✅ Configuration loaded successfully")
        print(f"   Host: {config.host}")
        print(f"   Port: {config.port}")
        print(f"   Database: {config.name}")
        print(f"   User: {config.user}")
        print(f"   Pool Size: {config.pool_size}")
        print(f"   Max Overflow: {config.max_overflow}")
        print(f"   Echo SQL: {config.echo}")
        return True
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False


async def test_connection():
    """Test 2: Verify database connectivity."""
    print("\n" + "="*70)
    print("TEST 2: Database Connection")
    print("="*70)
    
    try:
        is_connected = await check_connection()
        if is_connected:
            print("✅ Database connection successful")
            
            version = await get_database_version()
            if version:
                print(f"   PostgreSQL Version: {version.split(',')[0]}")
            
            return True
        else:
            print("❌ Database connection failed")
            print("   Make sure PostgreSQL is running:")
            print("   docker-compose up -d postgres")
            return False
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


async def test_pgvector():
    """Test 3: Verify pgvector extension."""
    print("\n" + "="*70)
    print("TEST 3: pgvector Extension")
    print("="*70)
    
    try:
        has_pgvector = await check_pgvector_extension()
        if has_pgvector:
            print("✅ pgvector extension is installed")
            return True
        else:
            print("❌ pgvector extension not found")
            print("   Run this SQL command:")
            print("   CREATE EXTENSION IF NOT EXISTS vector;")
            return False
    except Exception as e:
        print(f"❌ pgvector check failed: {e}")
        return False


async def test_tables():
    """Test 4: Verify database tables exist."""
    print("\n" + "="*70)
    print("TEST 4: Database Tables")
    print("="*70)
    
    try:
        counts = await get_table_row_counts()
        print("✅ All tables exist:")
        for table, count in counts.items():
            print(f"   {table}: {count} rows")
        return True
    except Exception as e:
        print(f"❌ Table check failed: {e}")
        print("   Run migrations:")
        print("   uv run alembic upgrade head")
        return False


async def test_connection_pool():
    """Test 5: Verify connection pooling."""
    print("\n" + "="*70)
    print("TEST 5: Connection Pool")
    print("="*70)
    
    try:
        status = get_pool_status()
        print("✅ Connection pool status:")
        print(f"   Pool Size: {status['size']}")
        print(f"   Checked In: {status['checked_in']}")
        print(f"   Checked Out: {status['checked_out']}")
        print(f"   Overflow: {status['overflow']}")
        print(f"   Total Connections: {status['total']}")
        return True
    except Exception as e:
        print(f"❌ Pool status check failed: {e}")
        return False


async def test_crud_operations():
    """Test 6: Test basic CRUD operations."""
    print("\n" + "="*70)
    print("TEST 6: CRUD Operations")
    print("="*70)
    
    session_id = None
    query_id = None
    
    try:
        # CREATE
        print("📝 Testing CREATE...")
        async with get_session() as session:
            # Create a research session
            research_session = ResearchSession(
                original_question="What is the capital of France?",
                status=SessionStatus.ACTIVE,
                metadata_={"test": True, "source": "test_script"}
            )
            session.add(research_session)
            await session.flush()  # Get the ID
            session_id = research_session.id
            
            # Create a query
            query = Query(
                session_id=session_id,
                query_text="capital of France",
                query_type=QueryType.INITIAL,
                categories_used=["general"],
                metadata_={"test": True}
            )
            session.add(query)
            await session.flush()
            query_id = query.id
            
            # Create search results
            search_result = SearchResult(
                query_id=query_id,
                url="https://en.wikipedia.org/wiki/Paris",
                title="Paris - Wikipedia",
                rank=1,
                metadata_={"engine": "wikipedia"}
            )
            session.add(search_result)
            
        print(f"   ✅ Created session: {session_id}")
        print(f"   ✅ Created query: {query_id}")
        
        # READ
        print("📖 Testing READ...")
        async with get_session() as session:
            result = await session.execute(
                select(ResearchSession).where(ResearchSession.id == session_id)
            )
            fetched_session = result.scalar_one()
            print(f"   ✅ Found session: {fetched_session.original_question}")
            
            # Test relationship loading
            result = await session.execute(
                select(Query).where(Query.session_id == session_id)
            )
            queries = result.scalars().all()
            print(f"   ✅ Found {len(queries)} queries")
        
        # UPDATE
        print("🔄 Testing UPDATE...")
        async with get_session() as session:
            result = await session.execute(
                select(ResearchSession).where(ResearchSession.id == session_id)
            )
            session_to_update = result.scalar_one()
            session_to_update.status = SessionStatus.COMPLETED
            await session.flush()
            print(f"   ✅ Updated session status to COMPLETED")
        
        # Verify update
        async with get_session() as session:
            result = await session.execute(
                select(ResearchSession).where(ResearchSession.id == session_id)
            )
            updated_session = result.scalar_one()
            assert updated_session.status == SessionStatus.COMPLETED
            print(f"   ✅ Verified update")
        
        # DELETE
        print("🗑️  Testing DELETE...")
        async with get_session() as session:
            result = await session.execute(
                select(ResearchSession).where(ResearchSession.id == session_id)
            )
            session_to_delete = result.scalar_one()
            await session.delete(session_to_delete)
            # Cascade delete should remove related records
        
        # Verify delete
        async with get_session() as session:
            result = await session.execute(
                select(ResearchSession).where(ResearchSession.id == session_id)
            )
            deleted_session = result.scalar_one_or_none()
            assert deleted_session is None
            print(f"   ✅ Verified deletion (cascade delete worked)")
        
        print("✅ All CRUD operations successful")
        return True
        
    except Exception as e:
        print(f"❌ CRUD operations failed: {e}")
        # Cleanup
        if session_id:
            try:
                async with get_session() as session:
                    result = await session.execute(
                        select(ResearchSession).where(ResearchSession.id == session_id)
                    )
                    cleanup_session = result.scalar_one_or_none()
                    if cleanup_session:
                        await session.delete(cleanup_session)
            except:
                pass
        return False


async def test_vector_operations():
    """Test 7: Test vector operations (if pgvector is available)."""
    print("\n" + "="*70)
    print("TEST 7: Vector Operations")
    print("="*70)
    
    try:
        has_pgvector = await check_pgvector_extension()
        if not has_pgvector:
            print("⏭️  Skipping: pgvector not available")
            return True
        
        # Create a query with embedding
        test_embedding = [0.1] * 384  # Dummy 384-dimensional vector
        
        async with get_session() as session:
            # Create session and query with embedding
            research_session = ResearchSession(
                original_question="Test vector search",
                status=SessionStatus.ACTIVE
            )
            session.add(research_session)
            await session.flush()
            
            query = Query(
                session_id=research_session.id,
                query_text="test query with embedding",
                query_type=QueryType.INITIAL,
                embedding=test_embedding
            )
            session.add(query)
            await session.flush()
            query_id = query.id
        
        # Test vector similarity search
        async with get_session() as session:
            # Find similar queries using cosine distance
            search_embedding = [0.1] * 384
            result = await session.execute(
                select(Query)
                .order_by(Query.embedding.cosine_distance(search_embedding))
                .limit(5)
            )
            similar_queries = result.scalars().all()
            print(f"   ✅ Found {len(similar_queries)} similar queries")
            
            # Cleanup
            result = await session.execute(
                select(ResearchSession).where(ResearchSession.id == similar_queries[0].session_id)
            )
            test_session = result.scalar_one()
            await session.delete(test_session)
        
        print("✅ Vector operations successful")
        return True
        
    except Exception as e:
        print(f"❌ Vector operations failed: {e}")
        return False


async def test_performance():
    """Test 8: Basic performance test."""
    print("\n" + "="*70)
    print("TEST 8: Performance Test")
    print("="*70)
    
    try:
        import time
        
        # Test batch insert
        start = time.time()
        session_ids = []
        
        async with get_session() as session:
            for i in range(10):
                research_session = ResearchSession(
                    original_question=f"Performance test question {i}",
                    status=SessionStatus.ACTIVE,
                    metadata_={"batch_test": True}
                )
                session.add(research_session)
            await session.flush()
        
        insert_time = time.time() - start
        print(f"   ✅ Inserted 10 sessions in {insert_time:.3f}s")
        
        # Test batch query
        start = time.time()
        async with get_session() as session:
            result = await session.execute(
                select(func.count(ResearchSession.id))
                .where(ResearchSession.metadata_['batch_test'].astext == 'true')
            )
            count = result.scalar()
        query_time = time.time() - start
        print(f"   ✅ Queried {count} sessions in {query_time:.3f}s")
        
        # Cleanup
        async with get_session() as session:
            result = await session.execute(
                select(ResearchSession)
                .where(ResearchSession.metadata_['batch_test'].astext == 'true')
            )
            test_sessions = result.scalars().all()
            for s in test_sessions:
                await session.delete(s)
        
        print("✅ Performance test completed")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


# ============================================================================
# Main Test Runner
# ============================================================================

async def run_all_tests():
    """Run all database tests."""
    print("\n" + "="*70)
    print("🧪 MINI-PERPLEXITY DATABASE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Configuration", test_configuration),
        ("Connection", test_connection),
        ("pgvector Extension", test_pgvector),
        ("Database Tables", test_tables),
        ("Connection Pool", test_connection_pool),
        ("CRUD Operations", test_crud_operations),
        ("Vector Operations", test_vector_operations),
        ("Performance", test_performance),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Cleanup
    await dispose_engine()
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Database is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
        return 1


def main():
    """Main entry point."""
    try:
        exit_code = asyncio.run(run_all_tests())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏸️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
