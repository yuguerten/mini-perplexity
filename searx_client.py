from __future__ import annotations

import asyncio
import concurrent.futures
import requests
import re
import yaml
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


def load_searx_settings(settings_path: str = None) -> Dict[str, Any]:
    """
    Load SearXNG settings from settings.yml file.
    
    Args:
        settings_path: Path to the settings.yml file (defaults to searxng/settings.yml)
        
    Returns:
        Dictionary containing the SearXNG settings
    """
    if settings_path is None:
        settings_path = os.path.join(os.getcwd(), 'searxng', 'settings.yml')
    
    try:
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = yaml.safe_load(f)
        return settings
    except FileNotFoundError:
        print(f"Settings file not found: {settings_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing settings file: {e}")
        return {}


def detect_categories_for_query(query: str) -> str:
    """
    Determine appropriate search categories based on query content, using a more targeted approach.
    
    Args:
        query: The search query text
        
    Returns:
        Comma-separated string of category names
    """
    query_lower = query.lower()
    
    query_types = {
        'news': ['news', 'latest', 'recent', 'update', 'today', 'current events', 'week', 'month', 'year', 'yesterday'],
        'academic': ['research paper', 'journal', 'academic', 'scholarly', 'publication', 'thesis', 'dissertation', 'doi'],
        'technical': ['programming', 'code', 'software', 'algorithm', 'framework', 'library', 'developer', 'github'],
        'visual': ['image', 'picture', 'photo', 'visual', 'screenshot', 'diagram', 'graph'],
        'video': ['video', 'youtube', 'watch', 'stream', 'tutorial', 'lecture', 'conference'],
        'health': ['medical', 'disease', 'symptom', 'treatment', 'patient', 'clinical', 'hospital', 'doctor'],
        'science': ['physics', 'chemistry', 'biology', 'experiment', 'scientific', 'theory', 'mathematical'],
        'files': ['pdf', 'document', 'download', '.pdf', '.doc', '.txt']
    }

    selected_categories = {'general'}
    primary_type = None
    strongest_match = 0
    
    for query_type, type_keywords in query_types.items():
        match_strength = sum(1 for kw in type_keywords if kw in query_lower)
        if match_strength > strongest_match:
            strongest_match = match_strength
            primary_type = query_type
    
    category_mapping = {
        'news': ['news', 'general', 'web'],
        'academic': ['academic', 'science', 'papers'], 
        'technical': ['it', 'general', 'web'],
        'visual': ['images', 'general'],
        'video': ['videos', 'general'],
        'health': ['health', 'medical', 'science'],
        'science': ['science', 'academic', 'papers'],
        'files': ['files', 'web', 'general']
    }
    
    # Handle news about specific topics differently
    if primary_type == 'news':
        if any(topic in query_lower for topic in ['eu', 'europe', 'european', 'schengen', 'country', 'countries']):
            # For news about countries/regions, we don't need academic or science categories
            return 'news,general,web'
        
        if any(topic in query_lower for topic in ['tech', 'technology', 'software', 'digital']):
            return 'news,it,web,general'
            
        if any(topic in query_lower for topic in ['health', 'medical', 'disease', 'covid', 'virus']):
            return 'news,health,medical,web,general'
    
    # Add relevant categories based on the primary query type
    if primary_type and primary_type in category_mapping:
        for category in category_mapping[primary_type]:
            selected_categories.add(category)
    
    # Special cases that override the general logic
    if 'research paper' in query_lower or 'academic paper' in query_lower or 'journal article' in query_lower:
        # Pure academic search should focus on academic sources
        return 'academic,papers,science,web'
    
    # News-specific optimization
    if 'news' in query_lower or 'recent' in query_lower or 'latest' in query_lower:
        # News queries should prioritize news sources
        return 'news,web,general'
    
    return ','.join(selected_categories)

def search(query: str, searx_url: str = None, num_results: int = 10, settings_path: str = None) -> List[str]:
    """
    Search using SearXNG and return a list of URLs.
    
    Args:
        query: The search query (original_question)
        searx_url: The SearXNG instance URL (defaults to Docker container or env var)
        num_results: Maximum number of results to return
        settings_path: Path to the settings.yml file
        
    Returns:
        List of URLs from search results
    """
    settings = load_searx_settings(settings_path)
    
    
    if searx_url is None:
        if settings and 'server' in settings and 'base_url' in settings['server'] and settings['server']['base_url']:
            # If base_url is set in settings, use it
            server_base = settings['server']['base_url']
            port = settings['server'].get('port', 8080)
            bind_address = settings['server'].get('bind_address', '127.0.0.1')
            searx_url = f"{server_base}" if server_base.startswith('http') else f"http://{server_base}"
        else:
            # Otherwise use port from settings or default
            port = settings.get('server', {}).get('port', 8080)
            # For client connections, always use localhost instead of bind_address
            # since 0.0.0.0 in the settings is meant for the server to listen on all interfaces
            # Use environment variable if set, otherwise construct URL with localhost
            searx_url = os.getenv('SEARXNG_URL', f"http://localhost:{port}")
    
    try:
        # Clean query by removing control characters
        query = ''.join(ch for ch in query if ord(ch) >= 32)
        
        # SearXNG search endpoint
        search_url = f"{searx_url}/search"
        
        # Detect appropriate categories based on query content
        query_specific_categories = detect_categories_for_query(query)
        default_categories = query_specific_categories
        
        # Use detected categories as they're now more optimized
        # But limit the number of categories to avoid timeouts and irrelevant results
        categories_list = query_specific_categories.split(',')
        
        # If we have too many categories, prioritize the most relevant ones
        max_categories = 4  # Limit to a reasonable number of categories
        if len(categories_list) > max_categories:
            # Always keep 'general' if it's in the list
            if 'general' in categories_list:
                categories_list.remove('general')
                trimmed_categories = categories_list[:max_categories-1]
                trimmed_categories.append('general')
            else:
                trimmed_categories = categories_list[:max_categories]
                
            default_categories = ','.join(trimmed_categories)
        
        # Get safe search setting if available
        safe_search = settings.get('search', {}).get('safe_search', '0')
        
        # Parameters for the search
        params = {
            'q': query,
            'categories': default_categories,
            'safesearch': str(safe_search),
            'pageno': '1'
        }
        
        # Headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        print(f"Searching SearXNG for: '{query}'")
        print(f"Using SearXNG instance: {searx_url}")
        print(f"Using categories: {default_categories}")
        
        # Show debug info when verbose mode is enabled
        if os.environ.get('SEARXNG_DEBUG') == '1':
            print(f"Query type analysis: Original categories were {query_specific_categories}")
            print(f"Category selection was optimized to {default_categories}")
        
        # Make the request for HTML results - with debug info
        print(f"Sending request to: {search_url} with params: {params}")
        try:
            # Increase timeout for SSH tunneling scenarios
            response = requests.get(search_url, params=params, headers=headers, timeout=30)
            print(f"Response received: HTTP {response.status_code}")
            response.raise_for_status()
        except requests.exceptions.Timeout as e:
            print(f"Timeout error: {e}")
            print("Debugging: The request timed out. This can happen with SSH tunneling.")
            print("Try using fewer search categories or increasing the timeout further.")
            return []
        except Exception as e:
            print(f"Request error: {e}")
            return []
        
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


def search_from_query_file(query_file_path: str, searx_url: str = None, num_results: int = 10, settings_path: str = None) -> List[str]:
    """
    Load a query JSON file and search using the original_question.
    
    Args:
        query_file_path: Path to the query JSON file
        searx_url: The SearXNG instance URL (defaults to Docker container or env var)
        num_results: Maximum number of results to return
        settings_path: Path to the settings.yml file
        
    Returns:
        List of URLs from search results
    """
    # Let search function handle the URL and settings
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
        
        return search(original_question, searx_url, num_results, settings_path)
        
    except FileNotFoundError:
        print(f"Query file not found: {query_file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing query file: {e}")
        return []
    except Exception as e:
        print(f"Error loading query file: {e}")
        return []

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


async def crawl_multiple_urls_batched(urls: List[str], batch_size: int = 5, verbose: bool = False) -> Dict[str, str]:
    """
    Crawl multiple URLs in parallel batches and return their content.
    
    Args:
        urls: List of URLs to crawl
        batch_size: Number of URLs to process concurrently
        verbose: Whether to enable verbose output
        
    Returns:
        Dictionary mapping URLs to their content
    """
    if not CRAWL4AI_AVAILABLE:
        print("crawl4ai package not available. Install it with: uv pip install -U crawl4ai nest_asyncio")
        return {}
    
    if not urls:
        return {}
    
    results = {}
    total_batches = (len(urls) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(urls))
        batch_urls = urls[start_idx:end_idx]
        
        print(f"Processing batch {batch_num+1}/{total_batches} ({len(batch_urls)} URLs)...")
        
        # Process each URL in the current batch in parallel
        tasks = [crawl_url(url, verbose) for url in batch_urls]
        batch_results = await asyncio.gather(*tasks)
        
        # Add results to the dictionary
        for url, content in zip(batch_urls, batch_results):
            if content is not None:
                results[url] = content
        
    return results

def crawl_urls(urls: List[str], output_file: Optional[str] = None, 
              verbose: bool = False, batch_size: int = 5) -> Dict[str, str]:
    """
    Crawl a list of URLs using batched processing and save their content.
    
    Args:
        urls: List of URLs to crawl
        output_file: File path to save the results, or None to skip saving
        verbose: Whether to enable verbose output
        batch_size: Number of URLs to process concurrently
    
    Returns:
        Dictionary mapping URLs to their content
    """
    if not CRAWL4AI_AVAILABLE:
        print("crawl4ai package not available. Install it with: uv pip install -U crawl4ai nest_asyncio")
        return {}
    
    # Run the crawler with batching
    crawled_content = asyncio.run(crawl_multiple_urls_batched(urls, batch_size, verbose))
    
    # Report results
    success_count = len(crawled_content)
    print(f"\nCrawling complete: {success_count}/{len(urls)} URLs successfully crawled")
    
    # Write results to file in JSON format if specified
    if output_file and crawled_content:
        json_data = []
        for url, content in crawled_content.items():
            # Split content into lines and filter out empty lines
            content_lines = [line.strip() for line in content.split('\n') if line.strip()]
            # Create a structured JSON object for each URL
            json_entry = {
                "url": url,
                "content": content_lines
            }
            json_data.append(json_entry)
        
        # Write JSON data to file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to: {output_file} in structured JSON format")
    
    return crawled_content


def search_and_crawl(query: str, searx_url: str = None, num_results: int = 10, 
                     output_file: Optional[str] = None, verbose: bool = False,
                     settings_path: str = None, batch_size: int = 5) -> Tuple[List[str], Dict[str, str]]:
    """
    Search using SearXNG and then crawl the resulting URLs.
    
    Args:
        query: The search query
        searx_url: The SearXNG instance URL
        num_results: Maximum number of results to return
        output_file: File to save crawled content to, or None to skip saving
        verbose: Whether to enable verbose output
        settings_path: Path to the settings.yml file
        batch_size: Number of URLs to process concurrently
        
    Returns:
        Tuple of (list of URLs, dictionary mapping URLs to their content)
    """
    # Search for URLs using settings from YAML
    urls = search(query, searx_url, num_results, settings_path)
    
    if not urls:
        return [], {}
    
    # Crawl the URLs using batched processing
    crawled_content = crawl_urls(urls, output_file, verbose, batch_size)
    
    return urls, crawled_content


def summarize_content(url: str, content: str, original_question: str = "", model: str = "gpt-oss:20b", reference_id: int = 0) -> str:
    """
    Use an LLM to summarize the content from a URL in relation to the user's question.
    
    Args:
        url: The URL that was crawled
        content: The content to summarize
        original_question: The user's original research question
        model: The model to use for summarization
        reference_id: The reference ID to use in citations
        
    Returns:
        Summarized content as a string with citation reference
    """
    if not OLLAMA_AVAILABLE:
        return "Error: ollama package not available. Install with: uv pip install ollama"
    
    # Handle content as either a string or list
    if isinstance(content, list):
        # Convert list to string, joining with newlines
        content = "\n".join(content)
    
    if not content or len(content.strip()) < 50:
        return "No sufficient content to summarize."
    
    # Truncate content if it's too long to avoid token limits
    max_content_length = 7500
    if len(content) > max_content_length:
        content = content[:max_content_length] + "...(content truncated)"
    
    try:
        question_context = f"Research question: {original_question}" if original_question else "No specific research question provided."
        
        system_prompt = """
        You are a research assistant that creates focused, relevant summaries of web content.
        Your summaries should:
        - Be directly relevant to the user's research question
        - Extract only information that helps answer the research question
        - Include citations using [n] format where n is the reference number
        - Be factual and objective
        - Focus on substantive content, ignoring navigation, ads, etc.
        - Be concise (2-3 paragraphs maximum)
        
        If the content isn't relevant to the research question, briefly explain why.
        """
        
        user_prompt = f"""
        {question_context}
        
        Please summarize the following content from URL: {url}
        Reference number to use for citation: [{reference_id}]
        
        Content:
        {content}
        
        Provide a concise summary that focuses ONLY on information relevant to the research question.
        Include the citation reference [{ reference_id }] when referencing information from this source.
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


def summarize_content_batch(url_content_pairs: List[Tuple[str, str]], 
                            model: str = "gpt-oss:20b",
                            original_question: str = "",
                            clarification_qa: List[Tuple[str, str]] = None,
                            output_file: Optional[str] = None) -> Dict[str, str]:
    """
    Summarize a batch of URL content pairs using MapReduce pattern.
    
    Args:
        url_content_pairs: List of (url, content) tuples to summarize
        model: The model to use for summarization
        original_question: The user's original research question
        clarification_qa: List of (question, answer) pairs from clarifications
        output_file: File to save summaries to, or None to skip saving
        
    Returns:
        Dictionary mapping URLs to their summaries
    """
    if not OLLAMA_AVAILABLE:
        return {url: "Error: ollama package not available" for url, _ in url_content_pairs}
    
    # MAP: Process each URL content pair in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(url_content_pairs))) as executor:
        future_to_url = {
            executor.submit(
                summarize_content, 
                url, 
                content, 
                original_question, 
                model, 
                i+1
            ): url 
            for i, (url, content) in enumerate(url_content_pairs)
        }
        
        # COLLECT: Gather results
        summaries = {}
        for future in tqdm(concurrent.futures.as_completed(future_to_url), 
                           total=len(future_to_url),
                           desc="Summarizing content"):
            url = future_to_url[future]
            try:
                summary = future.result()
                summaries[url] = summary
            except Exception as e:
                summaries[url] = f"Error summarizing content: {str(e)}"
        
        if output_file and summaries:
            with open(output_file, "w", encoding="utf-8") as f:
                # Write the title and question
                f.write(f"# Research Summary\n\n")
                
                if original_question:
                    f.write(f"**Research Question:** {original_question}\n\n")
                
                # Add clarifications if available
                if clarification_qa:
                    f.write("## Clarifications\n\n")
                    for question, answer in clarification_qa:
                        f.write(f"**Q: {question}**\n\n")
                        f.write(f"A: {answer}\n\n")
                    f.write("\n")
                
                # Write the consolidated summary section
                f.write("## Summary\n\n")
                
                # Generate an integrated summary if we have the capability
                try:
                    if OLLAMA_AVAILABLE and original_question:
                        # Prepare a condensed version of summaries to fit into context
                        condensed_summaries = []
                        for i, (url, summary) in enumerate(summaries.items()):
                            # Keep summaries brief for the consolidated report
                            short_summary = summary[:500] + "..." if len(summary) > 500 else summary
                            condensed_summaries.append(f"Source [{i+1}]: {short_summary}")
                        
                        all_summaries_text = "\n\n".join(condensed_summaries)
                        
                        # Add clarification context
                        clarification_context = ""
                        if clarification_qa:
                            clarification_context = "User clarifications:\n"
                            for q, a in clarification_qa:
                                clarification_context += f"Q: {q}\nA: {a}\n\n"
                        
                        system_prompt = """
                        You are a research assistant creating a consolidated report from multiple sources.
                        Write a coherent, well-structured summary that:
                        1. Directly addresses the research question and user clarifications
                        2. Integrates information from multiple sources using [n] citations
                        3. Is factual and objective
                        4. Presents the most relevant information first
                        5. Is concise but comprehensive (3-4 paragraphs)
                        6. Uses clear language and formatting appropriate for a research report
                        """
                        
                        user_prompt = f"""
                        Research Question: {original_question}
                        
                        {clarification_context if clarification_context else ""}
                        Below are summaries from different sources with their reference numbers:
                        
                        {all_summaries_text}
                        
                        Write a consolidated research report that integrates these sources to address the research question
                        and incorporates the user's clarifications. Use the citation format [n] to reference the sources.
                        """
                        
                        response = ollama.chat(
                            model=model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                        )
                        
                        consolidated_report = response.get("message", {}).get("content", "").strip()
                        
                        # Write the consolidated report
                        f.write(f"{consolidated_report}\n\n")
                    else:
                        # If no LLM available or no question, just write all summaries
                        for i, (url, summary) in enumerate(summaries.items()):
                            f.write(f"{summary}\n\n")
                except Exception as e:
                    print(f"Error generating consolidated report: {e}")
                    # Fallback: write individual summaries
                    for i, (url, summary) in enumerate(summaries.items()):
                        f.write(f"{summary}\n\n")
                
                # Write the references section
                f.write("## References\n\n")
                urls = [url for url, _ in url_content_pairs]
                for i, url in enumerate(urls):
                    f.write(f"[{i+1}] {url}\n\n")
        
        print(f"Research report saved to: {output_file}")
    
    return summaries


def search_with_multiple_queries(queries: List[str], searx_url: str = None, 
                           num_results_per_query: int = 5, verbose: bool = False,
                           settings_path: str = None) -> List[str]:
    """
    Search using multiple queries and return a combined list of unique URLs.
    
    Args:
        queries: List of search queries
        searx_url: The SearXNG instance URL
        num_results_per_query: Maximum number of results to return per query
        verbose: Whether to enable verbose output
        settings_path: Path to the settings.yml file
        
    Returns:
        List of unique URLs from all search results
    """
    all_urls = []
    unique_urls = set()
    
    for i, query in enumerate(queries, 1):
        if verbose:
            print(f"Search query {i}/{len(queries)}: '{query}'")
            
        # For each query, the search function will automatically determine appropriate categories
        urls = search(query, searx_url, num_results_per_query, settings_path)
        
        # Add only new URLs to the list (avoid duplicates)
        for url in urls:
            if url not in unique_urls:
                unique_urls.add(url)
                all_urls.append(url)
    
    if verbose:
        print(f"Found {len(all_urls)} unique URLs across {len(queries)} queries")
        
    return all_urls


def search_crawl_and_summarize(query: str, searx_url: str = None, num_results: int = 10,
                              model: str = "gpt-oss:20b", 
                              crawl_output_file: Optional[str] = None,
                              summary_output_file: Optional[str] = None,
                              verbose: bool = False,
                              additional_queries: List[str] = None,
                              clarification_qa: List[Tuple[str, str]] = None,
                              settings_path: str = None,
                              batch_size: int = 5) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """
    Complete workflow: search, crawl, and summarize content.
    
    Args:
        query: The primary search query (original question)
        searx_url: The SearXNG instance URL
        num_results: Maximum number of results to return per query
        model: The model to use for summarization
        crawl_output_file: File to save crawled content to, or None to skip saving
        summary_output_file: File to save summaries to, or None to skip saving
        verbose: Whether to enable verbose output
        additional_queries: Additional search queries derived from clarifications
        clarification_qa: List of (question, answer) pairs from clarifications
        
    Returns:
        Tuple of (list of URLs, dictionary of contents, dictionary of summaries)
    """
    # Combine all queries
    all_queries = [query]
    if additional_queries:
        all_queries.extend(additional_queries)
    
    # Search with all queries and get unique URLs
    if len(all_queries) > 1:
        # Adjust results per query to avoid overwhelming number of results
        results_per_query = max(3, num_results // len(all_queries))
        urls = search_with_multiple_queries(all_queries, searx_url, results_per_query, verbose, settings_path)
    else:
        # Just use the original query
        urls = search(query, searx_url, num_results, settings_path)
    
    if not urls:
        return [], {}, {}
    
    # Crawl the URLs with batched processing
    crawled_content = crawl_urls(urls, crawl_output_file, verbose, batch_size)
    
    # Process summarization in batches using MapReduce pattern
    url_content_pairs = [(url, content) for url, content in crawled_content.items()]
    
    # Use only batch summarization for all processing
    summaries = summarize_content_batch(
        url_content_pairs, 
        model=model,
        original_question=query,
        clarification_qa=clarification_qa,
        output_file=summary_output_file
    )
    
    return urls, crawled_content, summaries


def read_crawled_content(file_path: str) -> List[Dict]:
    """
    Read crawled content from a JSON file.
    
    Args:
        file_path: Path to the JSON file with crawled content
        
    Returns:
        List of dictionaries with URL and content fields
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file_path}")
        # Try to read as old format
        try:
            old_format_content = {}
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            sections = content.split('=' * 50)
            current_url = None
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                
                lines = section.split('\n')
                if current_url is None:
                    current_url = lines[0].strip()
                    continue
                
                content_text = '\n'.join(lines)
                if content_text:
                    old_format_content[current_url] = content_text
                    current_url = None
            
            return [{"url": url, "content": content.split('\n')} for url, content in old_format_content.items()]
        except Exception as e:
            print(f"Error reading file as old format: {e}")
            return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
