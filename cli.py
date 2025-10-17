from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

from ollama_client import generate_clarifying_questions, generate_next_question
from searx_client import search, search_from_query_file, crawl_urls, search_and_crawl, summarize_crawled_content, search_crawl_and_summarize, read_crawled_content

MODEL = "gpt-oss:20b"

def _input_with_default(prompt: str, default: str) -> str:
    raw = input(prompt).strip()
    return raw if raw else default


def _input_int(prompt: str, default: int, min_v: int | None = None, max_v: int | None = None) -> int:
    raw = input(prompt).strip()
    if not raw:
        return default
    try:
        val = int(raw)
        if min_v is not None and val < min_v:
            print(f"Value too small, using {default}")
            return default
        if max_v is not None and val > max_v:
            print(f"Value too large, using {default}")
            return default
        return val
    except ValueError:
        print(f"Invalid number, using {default}")
        return default


def main() -> None:
    print("Using model: ", end="")
    print(MODEL)

    print("What would you like to research? ", end="")
    original_question = input().strip()
    if not original_question:
        print("Please provide a research question. Exiting.")
        return

    breadth = _input_int("Enter research breadth (recommended 2-10, default 4): ", default=4, min_v=1, max_v=20)
    depth = _input_int("Enter research depth (recommended 1-5, default 2): ", default=2, min_v=1, max_v=10)
    report_kind = _input_with_default(
        "Do you want to generate a long report or a specific answer? (report/answer, default report): ",
        "report",
    ).lower()
    if report_kind not in {"report", "answer"}:
        report_kind = "report"

    print("Creating research plan..")

    # Interactive questioning: LLM generates questions one by one
    questions_asked = []
    answers = {}
    
    print("\nTo better understand your research needs, please answer these follow-up questions:\n")
    
    for question_num in range(1, 4):  # Max 3 questions
        # Generate next question based on context
        next_question = generate_next_question(original_question, list(answers.items()), question_num)
        
        if not next_question:
            break
            
        print(next_question)
        try:
            answer = input("Your answer: ").strip()
            if answer:  # Only store non-empty answers
                questions_asked.append(next_question)
                answers[next_question] = answer
            else:
                print("Skipping question...")
                break
        except (EOFError, KeyboardInterrupt):
            print("\nSkipping remaining questions...")
            break

    # Generate fields based on final context  
    if questions_asked:
        # Use the original function to get suggested fields
        clar = generate_clarifying_questions(original_question)
        fields = clar.get("fields", ["timeframe", "scope", "sources", "focus_area"])
    else:
        fields = ["timeframe", "scope", "sources", "focus_area"]

    print("\nStarting research...\n")
    print(f"Created {breadth} queries\n")

    # Persist query state as JSON for later steps
    now = datetime.now(timezone.utc)
    qid = str(uuid4())
    payload = {
        "id": qid,
        "original_question": original_question,
        "clarifying_questions": questions_asked,
        "suggested_fields": fields,
        "answers": answers,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "status": "ready" if questions_asked and all(answers.get(q, "").strip() for q in questions_asked) else "awaiting_answers",
        "plan": {
            "model": MODEL,
            "breadth": breadth,
            "depth": depth,
            "output": report_kind,
        },
    }

    out_dir = Path("queries")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{qid}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved session to {out_path}")
    
    # Perform search using SearXNG for the original question and clarifications
    print("\n=== Searching for relevant sources ===")
    try:
        from ollama_client import generate_search_query_from_answer, generate_combined_search_query
        
        searx_url = os.getenv('SEARXNG_URL', 'http://localhost:8080')
        max_results = 50
        
        # Generate search queries from clarifying answers
        additional_queries = []
        
        # Convert answers dictionary to list of pairs
        qa_pairs = list(answers.items())
        
        if qa_pairs:
            print("Generating optimized search queries based on your answers...")
            
            # Generate individual queries from each question-answer pair
            for question, answer in qa_pairs:
                if answer.strip():  # Only process non-empty answers
                    query = generate_search_query_from_answer(original_question, question, answer)
                    if query and query != original_question:
                        additional_queries.append(query)
                        print(f"Generated query: {query}")
            
            # Also generate a combined query from all Q&A pairs
            if len(qa_pairs) > 1:
                combined_query = generate_combined_search_query(original_question, qa_pairs)
                if combined_query and combined_query != original_question:
                    additional_queries.append(combined_query)
                    print(f"Generated combined query: {combined_query}")
        
        # Perform the search and crawl the results
        crawl_output_file = f"crawled_content_{qid}.txt"
        summary_output_file = f"summaries_{qid}.md"
        
        print(f"\nSearching using {1 + len(additional_queries)} queries...")
        
        urls, crawled_content, summaries = search_crawl_and_summarize(
            original_question, 
            searx_url, 
            max_results,
            model=MODEL,
            crawl_output_file=crawl_output_file,
            summary_output_file=summary_output_file,
            verbose=True,
            additional_queries=additional_queries,
            clarification_qa=qa_pairs
        )
        
        if urls:
            print(f"Found {len(urls)} relevant sources:")
            for i, url in enumerate(urls, 1):
                content_status = "✓ Content extracted" if url in crawled_content else "✗ No content extracted"
                print(f"{i}. {url} - {content_status}")
                
            # Update query data with the search queries that were used
            payload["search_queries"] = {
                "original": original_question,
                "additional": additional_queries,
                "total": 1 + len(additional_queries)
            }
                
            # Process was already handled by search_crawl_and_summarize
            # Just update the output message
            if crawled_content:
                print(f"\nCrawling complete: {len(crawled_content)}/{len(urls)} URLs successfully crawled")
                print(f"Research report saved to: {summary_output_file}")
                
                # Update the query file with crawled content and summary information
                payload["crawled_content"] = {
                    "crawl_file": crawl_output_file,
                    "summary_file": summary_output_file,
                    "count": len(crawled_content),
                    "urls": list(crawled_content.keys())
                }
                
                # Save updated query file
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                
                print(f"\nContent extracted and saved to {crawl_output_file}")
                print(f"Summaries saved to {summary_output_file}")
            else:
                print("\nNo content was extracted from the sources.")
        else:
            print("No relevant sources found. Make sure SearXNG is running (docker compose up -d searxng redis).")
    except Exception as e:
        print(f"Error searching for sources: {e}")
        print("Make sure SearXNG is running (docker compose up -d searxng redis).")


if __name__ == "__main__":
    main()
