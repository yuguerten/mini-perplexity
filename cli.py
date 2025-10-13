from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

from ollama_client import generate_clarifying_questions, generate_next_question
from searx_client import search
from searx_client import search_from_query_file


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
    model_default = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    print(model_default)

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
            "model": model_default,
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
    
    # Perform search using SearXNG for the original question
    print("\n=== Searching for relevant sources ===")
    try:
        searx_url = os.getenv('SEARXNG_URL', 'http://localhost:8080')
        max_results = 10  # Default to 10 results
        
        # Suppress verbose output from search function by temporarily redirecting stdout
        import sys
        from io import StringIO
        
        original_stdout = sys.stdout
        sys.stdout = StringIO()
        
        # Perform the search
        urls = search(original_question, searx_url, max_results)
        
        # Restore stdout
        sys.stdout = original_stdout
        
        if urls:
            print(f"Found {len(urls)} relevant sources:")
            for i, url in enumerate(urls, 1):
                print(f"{i}. {url}")
        else:
            print("No relevant sources found. Make sure SearXNG is running (docker compose up -d searxng redis).")
    except Exception as e:
        print(f"Error searching for sources: {e}")
        print("Make sure SearXNG is running (docker compose up -d searxng redis).")


if __name__ == "__main__":
    main()
