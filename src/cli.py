from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

# Rich for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import print as rprint
from rich.theme import Theme
from rich.prompt import Prompt
from rich.markdown import Markdown
from tqdm.auto import tqdm

from ollama_client import generate_clarifying_questions, generate_next_question
from searx_client import search_crawl_and_summarize
# Create a custom theme
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "highlight": "bold magenta",
    "muted": "dim",
})

# Initialize rich console
console = Console(theme=custom_theme)

MODEL = "gpt-oss:20b"

def _input_with_default(prompt: str, default: str) -> str:
    """Get user input with a default value"""
    raw = Prompt.ask(prompt, default=default)
    return raw.strip()


def _input_int(prompt: str, default: int, min_v: int | None = None, max_v: int | None = None) -> int:
    """Get integer input with validation"""
    raw = Prompt.ask(prompt, default=str(default))
    if not raw:
        return default
    try:
        val = int(raw)
        if min_v is not None and val < min_v:
            console.print(f"[warning]Value too small, using {default}[/warning]")
            return default
        if max_v is not None and val > max_v:
            console.print(f"[warning]Value too large, using {default}[/warning]")
            return default
        return val
    except ValueError:
        console.print(f"[warning]Invalid number, using {default}[/warning]")
        return default


def main() -> None:
    """Main function for the mini-perplexity research assistant"""
    # Display welcome message
    console.print(Panel.fit(
        "[bold cyan]Mini-Perplexity Research Assistant[/bold cyan]\n"
        "[dim]Your AI-powered research companion[/dim]",
        border_style="cyan"
    ))
    
    console.print(f"\nUsing model: [highlight]{MODEL}[/highlight]")

    # Get research question
    original_question = Prompt.ask("\n💡 What would you like to research today?")
    if not original_question:
        console.print("[error]Please provide a research question. Exiting.[/error]")
        return
    
    breadth = 4
    depth = 2
    report_kind = "report"

    # Show spinner while creating research plan
    with console.status("[info]Creating your personalized research plan...[/info]", spinner="dots"):
        time.sleep(1)  # Give visual feedback

    # Interactive questioning: LLM generates questions one by one
    questions_asked = []
    answers = {}
    
    console.print("\n[info]To better understand your research needs, I'll ask a few follow-up questions:[/info]\n")
    
    for question_num in range(1, 4):  # Max 3 questions
        # Show spinner while generating the question
        with console.status("[dim]Thinking of a relevant question...[/dim]", spinner="dots"):
            next_question = generate_next_question(original_question, list(answers.items()), question_num)
        
        if not next_question:
            break
            
        console.print(f"[bold]Q{question_num}:[/bold] {next_question}")
        try:
            answer = Prompt.ask("   Your answer", default="Skip")
            if answer and answer.lower() != "skip":  # Only store non-empty answers
                questions_asked.append(next_question)
                answers[next_question] = answer
            else:
                console.print("[muted]Skipping remaining questions...[/muted]")
                break
        except (EOFError, KeyboardInterrupt):
            console.print("\n[muted]Skipping remaining questions...[/muted]")
            break

    # Generate fields based on final context  
    with console.status("[info]Analyzing your research needs...[/info]", spinner="dots"):
        if questions_asked:
            # Use the original function to get suggested fields
            clar = generate_clarifying_questions(original_question)
            fields = clar.get("fields", ["timeframe", "scope", "sources", "focus_area"])
        else:
            fields = ["timeframe", "scope", "sources", "focus_area"]

    console.print("\n[success]🔍 Starting your research...[/success]")
    
    # Create the unique ID and query state JSON
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

    # Save query state
    out_dir = Path("queries")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{qid}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    
    console.print(f"[dim]Session saved to: {out_path}[/dim]")
    
    # Prepare for search
    try:
        from ollama_client import generate_search_query_from_answer, generate_combined_search_query
        
        searx_url = os.getenv('SEARXNG_URL', 'http://localhost:8080')
        max_results = 10
        
        # Generate search queries from clarifying answers
        additional_queries = []
        
        # Convert answers dictionary to list of pairs
        qa_pairs = list(answers.items())
        
        if qa_pairs:
            with console.status("[info]Optimizing search queries based on your answers...[/info]", spinner="dots"):
                # Generate individual queries from each question-answer pair
                for question, answer in qa_pairs:
                    if answer.strip():  # Only process non-empty answers
                        query = generate_search_query_from_answer(original_question, question, answer)
                        if query and query != original_question:
                            additional_queries.append(query)
                
                # Also generate a combined query from all Q&A pairs
                if len(qa_pairs) > 1:
                    combined_query = generate_combined_search_query(original_question, qa_pairs)
                    if combined_query and combined_query != original_question:
                        additional_queries.append(combined_query)
            
            # Only show the queries if there are some and not too many
            if additional_queries and len(additional_queries) < 5:
                console.print("\n[dim]Search queries generated:[/dim]")
                for i, query in enumerate(additional_queries, 1):
                    console.print(f"[dim]  {i}. {query}[/dim]")
        
        # Prepare output file paths
        crawl_output_file = f"crawled_content_{qid}.txt"
        summary_output_file = f"summaries_{qid}.md"
        
        # Check if this appears to be a news-specific search
        is_news_search = any(word in original_question.lower() for word in ['news', 'recent', 'latest', 'update'])
        if is_news_search:
            console.print("\n[info]Optimizing search for news content...[/info]")
        
        # Show search progress with tqdm
        console.print(f"\n[bold]🔎 Searching for relevant information...[/bold]")
        
        # Use a progress context for the entire search and process
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            search_task = progress.add_task("[cyan]Searching...", total=100)
            progress.update(search_task, advance=20)
            
            urls, crawled_content, summaries = search_crawl_and_summarize(
                original_question, 
                searx_url, 
                max_results,
                model=MODEL,
                crawl_output_file=crawl_output_file,
                summary_output_file=summary_output_file,
                verbose=False,  # We'll handle the progress display here
                additional_queries=additional_queries,
                clarification_qa=qa_pairs,
                batch_size=10  # Process 10 URLs concurrently for better performance
            )
            
            # Update progress
            progress.update(search_task, completed=100)
        
        if urls:
            # Show a summary of results
            console.print(f"\n[success]✓ Found {len(urls)} relevant sources[/success]")
            
            # Show URL summary in a more compact format
            if len(urls) > 10:
                console.print(f"[dim]Showing the first 10 of {len(urls)} sources:[/dim]")
                display_urls = urls[:10]
            else:
                display_urls = urls
                
            for i, url in enumerate(display_urls, 1):
                status_icon = "✓" if url in crawled_content else "○"
                status_style = "green" if url in crawled_content else "yellow"
                console.print(f"[{status_style}]{status_icon}[/{status_style}] [{i}] {url}")
            
            if len(urls) > 10:
                console.print(f"[dim]...and {len(urls) - 10} more[/dim]")
                
            # Update query data with the search queries that were used
            payload["search_queries"] = {
                "original": original_question,
                "additional": additional_queries,
                "total": 1 + len(additional_queries)
            }
                
            # Show results summary
            if crawled_content:
                console.print(f"\n[success]🔍 Research complete![/success]")
                
                # Show success rate
                success_rate = len(crawled_content) / len(urls) * 100
                console.print(f"[info]Successfully extracted content from {len(crawled_content)} of {len(urls)} sources ({success_rate:.1f}%)[/info]")
                
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
                
                # Show final results
                console.print(Panel.fit(
                    f"[bold green]Research Complete![/bold green]\n\n"
                    f"Your research question: [bold]{original_question}[/bold]\n\n"
                    f"📝 [bold]Research report:[/bold] {summary_output_file}\n"
                    f"📊 [dim]Raw content saved to:[/dim] {crawl_output_file}",
                    title="Results",
                    border_style="green"
                ))
                
                # Prompt to open the results
                console.print("\n[info]Tip:[/info] To view your research summary, run:")
                console.print(f"[dim]    cat {summary_output_file}[/dim]")
                
            else:
                console.print("\n[warning]⚠️ No content could be extracted from the found sources.[/warning]")
                console.print("[dim]This can happen with sites that block content extraction or have complex layouts.[/dim]")
        else:
            console.print("[warning]⚠️ No relevant sources found.[/warning]")
            console.print("[info]Make sure SearXNG is running with:[/info]")
            console.print("[dim]    docker compose up -d searxng redis[/dim]")
    except Exception as e:
        console.print(f"[error]❌ Error during research: {e}[/error]")
        console.print("[info]Make sure SearXNG is running with:[/info]")
        console.print("[dim]    docker compose up -d searxng redis[/dim]")


if __name__ == "__main__":
    main()
