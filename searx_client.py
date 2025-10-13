from __future__ import annotations

import requests
import re
from typing import List, Dict, Any, Optional
import json
import os
from bs4 import BeautifulSoup


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
