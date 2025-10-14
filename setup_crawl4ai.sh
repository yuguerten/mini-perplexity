#!/bin/bash
# Script to setup crawl4ai environment

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install packages using uv
echo "Installing required packages..."
uv pip install -U -qq crawl4ai
uv pip install -qq nest_asyncio

# Run crawl4ai setup
echo "Running crawl4ai setup..."
crawl4ai-setup

# Check installation
echo "Checking crawl4ai installation..."
crawl4ai-doctor

# Print usage instructions
echo -e "\nSetup complete! You can now use the crawl_content.py script."
echo "Example usage:"
echo "python crawl_content.py -l -o results.txt"
echo ""
echo "To see all options:"
echo "python crawl_content.py --help"

# Make crawl_content.py executable
chmod +x crawl_content.py