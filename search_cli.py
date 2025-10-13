#!/usr/bin/env python3
"""
CLI script to test SearXNG search functionality
"""
import argparse
import json
import sys
from pathlib import Path

from searx_client import search, search_from_query_file


def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(description="Search using SearXNG")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-q", "--query", help="Direct query string to search")
    group.add_argument("-f", "--file", help="JSON query file to load")
    group.add_argument("-l", "--latest", action="store_true", help="Use latest JSON query file")
    
    parser.add_argument("-u", "--url", default="http://localhost:8080", 
                        help="SearXNG URL (default: http://localhost:8080)")
    parser.add_argument("-n", "--num-results", type=int, default=10,
                        help="Maximum number of results to return (default: 10)")
    parser.add_argument("-o", "--output", help="Output JSON file (optional)")
    
    args = parser.parse_args()
    
    # Get query from arguments
    if args.query:
        print(f"Searching SearXNG for: {args.query}")
        urls = search(args.query, args.url, args.num_results)
    elif args.file:
        print(f"Using query file: {args.file}")
        urls = search_from_query_file(args.file, args.url, args.num_results)
    elif args.latest:
        queries_dir = Path("queries")
        if not queries_dir.exists():
            print("Error: No queries directory found", file=sys.stderr)
            return 1
        
        query_files = list(queries_dir.glob("*.json"))
        if not query_files:
            print("Error: No query files found", file=sys.stderr)
            return 1
        
        latest_file = max(query_files, key=lambda f: f.stat().st_mtime)
        print(f"Using latest query file: {latest_file}")
        urls = search_from_query_file(str(latest_file), args.url, args.num_results)
    
    # Output results
    if urls:
        print(f"Found {len(urls)} results")
        
        # Save to output file if specified
        if args.output:
            output_data = {
                "urls": urls,
                "count": len(urls)
            }
            
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)
                print(f"Results saved to {args.output}")
    else:
        print("No results found")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())