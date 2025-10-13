# mini-perplexity

An experimental, minimal clone of Perplexity/DeepSearch flow. The first section implemented here focuses on: user asks a question, the system proposes clarifying questions, and we store the interaction as a JSON "query" object.

The search/refinement phase will be added later.

## Section 1: Interactive question ➜ JSON query

Flow:
- User sends an initial question.
- The system creates 2–10 short clarifying questions and suggested structured fields (e.g., time_range, domain, location). If an Ollama model is available, it's used to generate better suggestions; otherwise a safe fallback is used.
- The user posts answers to some/all clarifying questions.
- We track the state in-memory and mark the query as ready once all clarifying questions are answered.

### JSON shape

Query object returned by the API:

{
	"id": "<uuid>",
	"original_question": "string",
	"clarifying_questions": ["string", ...],
	"suggested_fields": ["time_range", "domain", "location"],
	"answers": { "<question_or_field>": "string" },
	"created_at": "ISO-8601",
	"updated_at": "ISO-8601",
	"status": "awaiting_answers" | "ready"
}

### API

- POST /query/start
	- Body: { "question": "Find recent news in France" }
	- Returns: Query object with generated clarifying questions

- POST /query/{id}/answer
	- Body: { "answers": { "0": "last 7 days", "domain": "economy", "location": "France" } }
	- You can answer by index (as strings) or by field name
	- Returns: Updated Query object; once all questions are answered, status becomes "ready"

- GET /query/{id}
	- Returns: Current state of the query

### Run locally

With uv (recommended):

```bash
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 and use your REST client or the OpenAPI docs at http://localhost:8000/docs

### Docker

```bash
docker build -t mini-perplexity .
docker run --rm -p 8000:8000 mini-perplexity
```

### Optional: Ollama integration

If you have Ollama running locally and an appropriate model pulled (e.g., llama3.1:8b), the service will use it to generate higher-quality clarifying questions. Otherwise, a simple heuristic fallback provides reasonable defaults.

## Interactive CLI

You can also run an interactive CLI that mirrors the experience you described:

```bash
uv run python cli.py
```

Example session:

Using model: llama3.1:8b
What would you like to research? What are the recent news in Morocco last three weeks
Enter research breadth (recommended 2-10, default 4): 5
Enter research depth (recommended 1-5, default 2): 4
Do you want to generate a long report or a specific answer? (report/answer, default report): report
Creating research plan..
Creating research plan...

To better understand your research needs, please answer these follow-up questions:

Can you specify what type of news you're looking for (e.g. politics, economy, sports)?
Your answer: education, health and politics

Are there any specific regions or cities in Morocco you'd like to focus on?
Your answer: Rabat and Casablanca

Would you like the news to be categorized by date or topic?
Your answer: by date

Starting research...

Created 5 queries

Saved session to queries/<uuid>.json

