from __future__ import annotations

import asyncio
import requests
import re
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from bs4 import BeautifulSoup
from tqdm import tqdm

# Import required for crawl4ai
try:
    import nest_asyncio
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    
    # Apply nest_asyncio to allow nested event loops
    nest_asyncio.apply()
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False

# Import for LLM integration
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


def search(query: str, searx_url: str = None, num_results: int = 10) -> List[str]:
    """
    Search using SearXNG and return a list of URLs.
    
    Args:
        query: The search query (original_question)
        searx_url: The SearXNG instance URL (defaults to Docker container or env var)
        num_results: Maximum number of results to return
        
    Returns:
        List of URLs from search results
    """
    # Use Docker container URL by default, fallback to environment variable or public instance
    if searx_url is None:
        searx_url = os.getenv('SEARXNG_URL', 'http://localhost:8080')
    
    try:
        # Clean query by removing control characters
        query = ''.join(ch for ch in query if ord(ch) >= 32)
        
        # SearXNG search endpoint
        search_url = f"{searx_url}/search"
        
        # Parameters for the search
        params = {
            'q': query,
            'categories': 'general,science',  # Focus on general and science categories for academic papers
            'safesearch': '0',  # No safe search filtering
            'pageno': '1'  # First page
        }
        
        # Headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        print(f"Searching SearXNG for: '{query}'")
        print(f"Using SearXNG instance: {searx_url}")
        
        # Make the request for HTML results
        response = requests.get(search_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML response using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract search results - look for the result elements
        # The exact CSS selector might vary depending on SearXNG version/theme
        result_elements = soup.select('.result')
        
        if not result_elements:
            # Try alternative selectors if nothing found
            result_elements = soup.select('.result-default')
        
        if not result_elements:
            # One more attempt with more generic selectors
            result_elements = soup.select('article')
        
        print(f"Found {len(result_elements)} results")
        
        # Extract URLs and titles
        urls = []
        for i, result in enumerate(result_elements[:num_results], 1):
            # Try to find the URL in different potential elements
            url_element = result.select_one('.url') or result.select_one('a.url_wrapper') or result.select_one('a')
            title_element = result.select_one('h3') or result.select_one('.result_header') or result.select_one('a')
            
            url = None
            if url_element and url_element.has_attr('href'):
                url = url_element['href']
            elif url_element and url_element.text and re.match(r'https?://', url_element.text.strip()):
                url = url_element.text.strip()
                
            title = title_element.text.strip() if title_element else f"Result {i}"
            
            if url:
                urls.append(url)
                print(f"- {title}")
                print(f"  URL: {url}")
        
        print(f"\nExtracted {len(urls)} URLs:")
        for i, url in enumerate(urls, 1):
            print(f"{i}. {url}")
            
        return urls
        
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to SearXNG: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error during search: {e}")
        return []


def search_from_query_file(query_file_path: str, searx_url: str = None, num_results: int = 10) -> List[str]:
    """
    Load a query JSON file and search using the original_question.
    
    Args:
        query_file_path: Path to the query JSON file
        searx_url: The SearXNG instance URL (defaults to Docker container or env var)
        num_results: Maximum number of results to return
        
    Returns:
        List of URLs from search results
    """
    if searx_url is None:
        searx_url = os.getenv('SEARXNG_URL', 'http://localhost:8080')
        
    try:
        with open(query_file_path, 'r', encoding='utf-8') as f:
            query_data = json.load(f)
        
        original_question = query_data.get('original_question', '')
        if not original_question:
            print("No original_question found in query file")
            return []
        
        # Clean up the query - remove any control characters
        original_question = ''.join(ch for ch in original_question if ord(ch) >= 32)
        
        print(f"Loading query from: {query_file_path}")
        print(f"Original question: {original_question}")
        
        return search(original_question, searx_url, num_results)
        
    except FileNotFoundError:
        print(f"Query file not found: {query_file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing query file: {e}")
        return []
    except Exception as e:
        print(f"Error loading query file: {e}")
        return []


def get_latest_query_file() -> Optional[str]:
    """
    Find the latest query JSON file in the queries directory.
    
    Returns:
        Path to the latest query file or None if none found
    """
    try:
        queries_dir = os.path.join(os.getcwd(), "queries")
        if not os.path.exists(queries_dir):
            return None
        
        query_files = [f for f in os.listdir(queries_dir) if f.endswith('.json')]
        if not query_files:
            return None
        
        latest_file = max(query_files, key=lambda f: os.path.getmtime(os.path.join(queries_dir, f)))
        return os.path.join(queries_dir, latest_file)
    except Exception:
        return None


async def crawl_url(url: str, verbose: bool = False) -> Optional[str]:
    """
    Fetch content from a URL using crawl4ai.
    
    Args:
        url: URL to crawl
        verbose: Whether to enable verbose output
    
    Returns:
        Markdown content as a string, or None if failed
    """
    if not CRAWL4AI_AVAILABLE:
        print("crawl4ai package not available. Install it with: uv pip install -U crawl4ai nest_asyncio")
        return None
    
    try:
        if verbose:
            print(f"Crawling: {url}")
        
        async with AsyncWebCrawler(verbose=verbose) as crawler:
            config = CrawlerRunConfig(
                cache_mode=CacheMode.ENABLED,
                excluded_tags=['nav', 'footer', 'aside'],
                remove_overlay_elements=True,
                markdown_generator=DefaultMarkdownGenerator(
                    content_filter=PruningContentFilter(
                        threshold=0.48, 
                        threshold_type="fixed", 
                        min_word_threshold=0
                    ),
                    options={
                        "ignore_links": True
                    }
                ),
            )
            
            result = await crawler.arun(
                url=url,
                config=config,
            )
            
            return result.markdown.raw_markdown
    except Exception as e:
        if verbose:
            print(f"Error crawling {url}: {e}")
        return None


async def crawl_multiple_urls(urls: List[str], verbose: bool = False) -> Dict[str, str]:
    """
    Crawl multiple URLs in parallel and return their content.
    
    Args:
        urls: List of URLs to crawl
        verbose: Whether to enable verbose output
        
    Returns:
        Dictionary mapping URLs to their content
    """
    if not CRAWL4AI_AVAILABLE:
        print("crawl4ai package not available. Install it with: uv pip install -U crawl4ai nest_asyncio")
        return {}
    
    if not urls:
        return {}
    
    print(f"Crawling {len(urls)} URLs...")
    
    # Process each URL in parallel
    tasks = [crawl_url(url, verbose) for url in urls]
    results = await asyncio.gather(*tasks)
    
    # Create a dictionary of URL -> content
    return {url: content for url, content in zip(urls, results) if content is not None}


def crawl_urls(urls: List[str], output_file: Optional[str] = None, verbose: bool = False) -> Dict[str, str]:
    """
    Crawl a list of URLs and save their content to a file if specified.
    
    Args:
        urls: List of URLs to crawl
        output_file: File path to save the results, or None to skip saving
        verbose: Whether to enable verbose output
    
    Returns:
        Dictionary mapping URLs to their content
    """
    if not CRAWL4AI_AVAILABLE:
        print("crawl4ai package not available. Install it with: uv pip install -U crawl4ai nest_asyncio")
        return {}
    
    # Run the crawler
    crawled_content = asyncio.run(crawl_multiple_urls(urls, verbose))
    
    # Report results
    success_count = len(crawled_content)
    print(f"\nCrawling complete: {success_count}/{len(urls)} URLs successfully crawled")
    
    # Write results to file if specified
    if output_file and crawled_content:
        with open(output_file, "w", encoding="utf-8") as f:
            for url, content in crawled_content.items():
                f.write(f"{url}\n")
                f.write("="*50 + "\n")
                f.write(content)
                f.write("\n" + "="*50 + "\n\n")
        
        print(f"Results saved to: {output_file}")
    
    return crawled_content


def search_and_crawl(query: str, searx_url: str = None, num_results: int = 10, 
                     output_file: Optional[str] = None, verbose: bool = False) -> Tuple[List[str], Dict[str, str]]:
    """
    Search using SearXNG and then crawl the resulting URLs.
    
    Args:
        query: The search query
        searx_url: The SearXNG instance URL
        num_results: Maximum number of results to return
        output_file: File to save crawled content to, or None to skip saving
        verbose: Whether to enable verbose output
        
    Returns:
        Tuple of (list of URLs, dictionary mapping URLs to their content)
    """
    # Search for URLs
    urls = search(query, searx_url, num_results)
    
    if not urls:
        return [], {}
    
    # Crawl the URLs
    crawled_content = crawl_urls(urls, output_file, verbose)
    
    return urls, crawled_content


def summarize_content(url: str, content: str, model: str = "gpt-oss:20b") -> str:
    """
    Use an LLM to summarize the content from a URL.
    
    Args:
        url: The URL that was crawled
        content: The content to summarize
        model: The model to use for summarization
        
    Returns:
        Summarized content as a string
    """
    if not OLLAMA_AVAILABLE:
        return "Error: ollama package not available. Install with: uv pip install ollama"
    
    if not content or len(content.strip()) < 50:
        return "No sufficient content to summarize."
    
    # Truncate content if it's too long to avoid token limits
    max_content_length = 7500
    if len(content) > max_content_length:
        content = content[:max_content_length] + "...(content truncated)"
    
    try:
        system_prompt = """
        You are a helpful assistant that creates concise, accurate summaries of web content.
        Your summaries should:
        - Be 2-3 short paragraphs
        - Capture the main points and key information
        - Be factual and objective
        - Focus on substantive content, ignoring navigation, ads, etc.
        - Provide the most useful information from the source
        """
        
        user_prompt = f"""
        Please summarize the following content from URL: {url}
        
        Content:
        {content}
        
        Provide a concise summary of the key information in 2-3 paragraphs.
        """
        
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        
        summary = response.get("message", {}).get("content", "").strip()
        return summary if summary else "Failed to generate summary."
    
    except Exception as e:
        return f"Error generating summary: {str(e)}"


def summarize_crawled_content(crawled_content: Dict[str, str], 
                             model: str = "gpt-oss:20b", 
                             output_file: Optional[str] = None) -> Dict[str, str]:
    """
    Summarize all crawled content and optionally save to a file.
    
    Args:
        crawled_content: Dictionary mapping URLs to their content
        model: The model to use for summarization
        output_file: File to save summaries to, or None to skip saving
        
    Returns:
        Dictionary mapping URLs to their summaries
    """
    if not OLLAMA_AVAILABLE:
        print("Error: ollama package not available. Install with: uv pip install ollama")
        return {}
    
    if not crawled_content:
        print("No content to summarize.")
        return {}
    
    print(f"\nSummarizing content from {len(crawled_content)} URLs...")
    summaries = {}
    
    # Use tqdm to show progress
    for url in tqdm(crawled_content.keys(), desc="Summarizing", unit="url"):
        content = crawled_content[url]
        summary = summarize_content(url, content, model)
        summaries[url] = summary
    
    # Write summaries to file if specified
    if output_file and summaries:
        with open(output_file, "w", encoding="utf-8") as f:
            for url, summary in summaries.items():
                f.write(f"{url}\n")
                f.write("="*50 + "\n")
                f.write(summary)
                f.write("\n" + "="*50 + "\n\n")
        
        print(f"Summaries saved to: {output_file}")
    
    return summaries


def search_crawl_and_summarize(query: str, searx_url: str = None, num_results: int = 10,
                              model: str = "gpt-oss:20b", 
                              crawl_output_file: Optional[str] = None,
                              summary_output_file: Optional[str] = None,
                              verbose: bool = False) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """
    Complete workflow: search, crawl, and summarize content.
    
    Args:
        query: The search query
        searx_url: The SearXNG instance URL
        num_results: Maximum number of results to return
        model: The model to use for summarization
        crawl_output_file: File to save crawled content to, or None to skip saving
        summary_output_file: File to save summaries to, or None to skip saving
        verbose: Whether to enable verbose output
        
    Returns:
        Tuple of (list of URLs, dictionary of contents, dictionary of summaries)
    """
    # Search and crawl
    urls, crawled_content = search_and_crawl(
        query, searx_url, num_results, crawl_output_file, verbose
    )
    
    # Summarize
    summaries = summarize_crawled_content(crawled_content, model, summary_output_file)
    
    return urls, crawled_content, summaries
