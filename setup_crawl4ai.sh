#!/bin/bash

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing required packages..."
uv pip install -U -qq crawl4ai
uv pip install -qq nest_asyncio

echo "Running crawl4ai setup..."
crawl4ai-setup

echo "Checking crawl4ai installation..."
crawl4ai-doctor

echo -e "\nSetup complete!"