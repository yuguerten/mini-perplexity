# Mini-Perplexity

A CLI research assistant powered by local LLMs (via Ollama) and web search (via SearXNG).

![Architecture](assets/arch.png)

## Features

- **Interactive Research**: Asks clarifying questions to understand your research needs
- **Web Search**: Uses SearXNG to find relevant sources across the web
- **Content Extraction**: Crawls and summarizes web pages using crawl4ai
- **Local LLM**: Uses Ollama with a predefined model by the user
- **Markdown Reports**: Generates research reports with citations and references

## Quick Start

1. **Start SearXNG (search engine):**
   ```bash
   bash scripts/searx-docker.sh start
   ```

2. **Setup crawl4ai (web crawler):**
   ```bash
   bash scripts/setup_crawl4ai.sh
   ```

3. **Run the assistant:**
   ```bash
   uv run src/cli.py
   ```

## Requirements

- Python 3.12+
- Docker & Docker Compose
- Ollama with deepseek-r1:1.5b model installed
- uv package manager

## Usage

After running `uv run src/cli.py`, the assistant will:
1. Ask what you want to research
2. Ask up to 3 clarifying questions
3. Search the web using SearXNG
4. Crawl and summarize relevant pages
5. Generate a markdown report with citations

Results are saved to:
- `summaries_{uuid}.md` - Research report with consolidated summary
- `crawled_content_{uuid}.txt` - Raw crawled content
- `queries/{uuid}.json` - Query metadata

## Commands

### SearXNG Management

```bash
bash scripts/searx-docker.sh start    # Start SearXNG and Redis
bash scripts/searx-docker.sh stop     # Stop containers
bash scripts/searx-docker.sh restart  # Restart containers
bash scripts/searx-docker.sh logs     # View logs
bash scripts/searx-docker.sh test     # Test connection
```

## Project Structure

```
mini-perplexity/
├── src/
│   ├── cli.py                # Main entry point
│   ├── ollama_client.py      # LLM integration (Ollama)
│   └── searx_client.py       # Search & crawling logic
├── scripts/
│   ├── searx-docker.sh       # SearXNG management script
│   └── setup_crawl4ai.sh     # crawl4ai setup script
├── assets/
│   └── arch.png              # Architecture diagram
├── searxng/settings.yml      # SearXNG configuration
├── queries/                  # Query history (created at runtime)
├── pyproject.toml            # Python dependencies
├── uv.lock                   # Locked dependencies
└── docker-compose.yml        # SearXNG + Redis services
```

## Dependencies

See `pyproject.toml` for full list. Key dependencies:
- `fastapi` - Web framework
- `ollama` - Local LLM client
- `requests` - HTTP requests
- `beautifulsoup4` - HTML parsing
- `crawl4ai` - Web crawling
- `rich` - Terminal UI
- `tqdm` - Progress bars

