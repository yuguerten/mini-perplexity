"""
Database connection management for mini-perplexity.

Provides async database engine, session management, and connection pooling.
"""
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from dotenv import load_dotenv
import yaml
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy import event, text

from .models import Base


# Load environment variables
load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

class DatabaseConfig:
    """Database configuration from environment variables and config files."""
    
    def __init__(self):
        # Try to load from config file first
        config_path = Path("config/database.yml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                db_config = config.get('database', {})
        else:
            db_config = {}
        
        # Environment variables override config file
        self.host = os.getenv('DATABASE_HOST', db_config.get('host', 'localhost'))
        self.port = int(os.getenv('DATABASE_PORT', db_config.get('port', 5432)))
        self.name = os.getenv('DATABASE_NAME', db_config.get('name', 'mini_perplexity'))
        self.user = os.getenv('DATABASE_USER', db_config.get('user', 'perplexity_user'))
        self.password = os.getenv('DATABASE_PASSWORD', db_config.get('password', ''))
        
        # Pool configuration
        self.pool_size = int(os.getenv('DATABASE_POOL_SIZE', db_config.get('pool_size', 10)))
        self.max_overflow = int(os.getenv('DATABASE_MAX_OVERFLOW', db_config.get('max_overflow', 20)))
        self.pool_timeout = int(os.getenv('DATABASE_POOL_TIMEOUT', db_config.get('pool_timeout', 30)))
        self.pool_recycle = int(os.getenv('DATABASE_POOL_RECYCLE', db_config.get('pool_recycle', 3600)))
        
        # Debug mode
        self.echo = os.getenv('DATABASE_ECHO', str(db_config.get('echo', 'false'))).lower() == 'true'
    
    @property
    def url(self) -> str:
        """Construct async database URL."""
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )
    
    def __repr__(self) -> str:
        return f"<DatabaseConfig(host={self.host}, port={self.port}, name={self.name})>"


# ============================================================================
# Global Engine and Session Factory
# ============================================================================

_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None
_config: Optional[DatabaseConfig] = None


def get_config() -> DatabaseConfig:
    """Get or create database configuration."""
    global _config
    if _config is None:
        _config = DatabaseConfig()
    return _config


def get_engine(echo: Optional[bool] = None) -> AsyncEngine:
    """
    Get or create async database engine with connection pooling.
    
    Args:
        echo: Override SQL echo setting. If None, uses config value.
    
    Returns:
        AsyncEngine instance with connection pooling.
    """
    global _engine
    
    if _engine is None:
        config = get_config()
        
        # Determine echo setting
        if echo is None:
            echo = config.echo
        
        # Create engine with connection pooling
        _engine = create_async_engine(
            config.url,
            echo=echo,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_timeout=config.pool_timeout,
            pool_recycle=config.pool_recycle,
            pool_pre_ping=True,  # Verify connections before using
        )
        
        # Register event listeners
        _register_engine_events(_engine)
    
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Get or create async session factory.
    
    Returns:
        Async session factory for creating database sessions.
    """
    global _session_factory
    
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,  # Don't expire objects after commit
            autoflush=False,  # Manual flush control
            autocommit=False,
        )
    
    return _session_factory


# ============================================================================
# Session Context Manager
# ============================================================================

@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.
    
    Usage:
        async with get_session() as session:
            result = await session.execute(...)
            await session.commit()
    
    Yields:
        AsyncSession instance that auto-closes on exit.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_session_no_autocommit() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions without auto-commit.
    
    Use this when you need manual transaction control.
    
    Usage:
        async with get_session_no_autocommit() as session:
            result = await session.execute(...)
            await session.commit()  # Manual commit
    
    Yields:
        AsyncSession instance that auto-closes on exit.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ============================================================================
# Database Initialization
# ============================================================================

async def init_db() -> None:
    """
    Initialize database by creating all tables.
    
    WARNING: This should only be used in development.
    In production, use Alembic migrations instead.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database tables created successfully")


async def drop_db() -> None:
    """
    Drop all database tables.
    
    WARNING: This will delete all data! Use with caution.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    print("⚠️  All database tables dropped")


# ============================================================================
# Health Check
# ============================================================================

async def check_connection() -> bool:
    """
    Check if database connection is working.
    
    Returns:
        True if connection successful, False otherwise.
    """
    try:
        engine = get_engine(echo=False)
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            result.fetchone()  # Don't await, fetchone is sync on the result object
        return True
    except Exception as e:
        print(f"Connection check failed: {e}")
        return False


async def get_database_version() -> Optional[str]:
    """
    Get PostgreSQL database version.
    
    Returns:
        Version string or None if unable to fetch.
    """
    try:
        engine = get_engine(echo=False)
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT version()"))
            row = result.fetchone()  # Don't await
            return row[0] if row else None
    except Exception:
        return None


async def check_pgvector_extension() -> bool:
    """
    Check if pgvector extension is installed.
    
    Returns:
        True if pgvector is available, False otherwise.
    """
    try:
        engine = get_engine(echo=False)
        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'")
            )
            row = result.fetchone()  # Don't await
            return row[0] > 0 if row else False
    except Exception:
        return False

# ============================================================================
# Connection Pool Management
# ============================================================================

async def dispose_engine() -> None:
    """
    Dispose of the database engine and close all connections.
    
    Useful for cleanup in tests or application shutdown.
    """
    global _engine, _session_factory
    
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        print("✅ Database engine disposed")


def get_pool_status() -> dict:
    """
    Get current connection pool status.
    
    Returns:
        Dictionary with pool statistics.
    """
    engine = get_engine()
    pool = engine.pool
    
    return {
        "size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "total": pool.size() + pool.overflow(),
    }


# ============================================================================
# Event Listeners
# ============================================================================

def _register_engine_events(engine: AsyncEngine) -> None:
    """Register event listeners for connection management."""
    
    @event.listens_for(engine.sync_engine, "connect")
    def receive_connect(dbapi_conn, connection_record):
        """Set up connection on connect."""
        # Set statement timeout to prevent long-running queries
        # For asyncpg, we need to use the connection directly
        pass  # asyncpg handles this differently, skip for now
    
    @event.listens_for(engine.sync_engine, "checkout")
    def receive_checkout(dbapi_conn, connection_record, connection_proxy):
        """Verify connection on checkout from pool."""
        # This is handled by pool_pre_ping, but we can add custom logic here
        pass


# ============================================================================
# Utility Functions
# ============================================================================

async def execute_raw_sql(sql: str) -> None:
    """
    Execute raw SQL statement.
    
    Args:
        sql: SQL statement to execute.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(sql))


async def get_table_row_counts() -> dict:
    """
    Get row counts for all tables.
    
    Returns:
        Dictionary mapping table names to row counts.
    """
    tables = [
        "research_sessions",
        "queries", 
        "search_results",
        "crawled_content",
        "summaries"
    ]
    
    counts = {}
    try:
        engine = get_engine(echo=False)
        async with engine.connect() as conn:
            for table in tables:
                result = await conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                row = result.fetchone()  # Don't await
                counts[table] = row[0] if row else 0
    except Exception as e:
        print(f"Error getting row counts: {e}")
    
    return counts


# ============================================================================
# Context Manager for Transaction Management
# ============================================================================

@asynccontextmanager
async def transaction() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for explicit transaction management.
    
    Usage:
        async with transaction() as session:
            # Multiple operations in same transaction
            session.add(obj1)
            session.add(obj2)
            # Auto-commits on success, auto-rolls back on error
    
    Yields:
        AsyncSession with transaction management.
    """
    async with get_session() as session:
        async with session.begin():
            yield session
