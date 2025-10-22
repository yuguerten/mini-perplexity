#!/usr/bin/env python3
"""
Quick database connection test.

Simple script to verify PostgreSQL is running and accessible.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def quick_test():
    """Quick connection test."""
    from mini_perplexity.db.connection import (
        check_connection,
        check_pgvector_extension,
        get_database_version,
        get_config,
        dispose_engine,
    )
    
    print("🔍 Quick Database Connection Test\n")
    
    # Configuration
    config = get_config()
    print(f"Configuration:")
    print(f"  Host: {config.host}:{config.port}")
    print(f"  Database: {config.name}")
    print(f"  User: {config.user}\n")
    
    # Connection test
    print("Testing connection...", end=" ")
    is_connected = await check_connection()
    
    if not is_connected:
        print("❌ FAILED")
        print("\n💡 Make sure PostgreSQL is running:")
        print("   docker-compose up -d postgres")
        await dispose_engine()
        return False
    
    print("✅ SUCCESS")
    
    # Version check
    version = await get_database_version()
    if version:
        print(f"  PostgreSQL: {version.split(',')[0]}")
    
    # pgvector check
    print("\nTesting pgvector extension...", end=" ")
    has_pgvector = await check_pgvector_extension()
    
    if not has_pgvector:
        print("❌ NOT FOUND")
        print("\n💡 Install pgvector extension:")
        print("   docker exec -it mini-perplexity-db psql -U perplexity_user -d mini_perplexity")
        print("   Then run: CREATE EXTENSION IF NOT EXISTS vector;")
        await dispose_engine()
        return False
    
    print("✅ INSTALLED")
    
    # Success
    print("\n✅ Database is ready!")
    print("\n📝 Next steps:")
    print("   1. Run migrations: uv run alembic upgrade head")
    print("   2. Run full tests: python scripts/test_database.py")
    
    await dispose_engine()
    return True


def main():
    """Main entry point."""
    try:
        result = asyncio.run(quick_test())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n⏸️  Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. PostgreSQL container is running: docker-compose up -d postgres")
        print("  2. .env file has correct database credentials")
        sys.exit(1)


if __name__ == "__main__":
    main()
