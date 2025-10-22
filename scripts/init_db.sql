-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create enum types
CREATE TYPE query_type AS ENUM ('initial', 'clarification', 'follow_up');
CREATE TYPE session_status AS ENUM ('active', 'completed', 'archived');
CREATE TYPE summary_type AS ENUM ('individual', 'combined', 'final');

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE mini_perplexity TO mouhcine;