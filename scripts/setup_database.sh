#!/bin/bash

echo "🚀 Setting up Mini-Perplexity Database..."

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Start PostgreSQL container
echo "📦 Starting PostgreSQL with pgvector..."
docker-compose up -d postgres

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
sleep 5

# Check if database is ready
until docker exec mini-perplexity-db pg_isready -U mouhcine -d mini_perplexity > /dev/null 2>&1; do
    echo "   Still waiting..."
    sleep 2
done

echo "✅ PostgreSQL is ready!"

# Run migrations
echo "🔄 Running database migrations..."
uv run alembic upgrade head

echo "✨ Database setup complete!"
echo ""
echo "Connection details:"
echo "  Host: localhost"
echo "  Port: 5432"
echo "  Database: mini_perplexity"
echo "  User: mouhcine"
echo ""
echo "To connect: psql -h localhost -U mouhcine -d mini_perplexity"
