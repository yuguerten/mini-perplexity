from __future__ import annotations

from typing import Dict, List, Optional
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ollama_client import generate_clarifying_questions


app = FastAPI(title="mini-perplexity", version="0.1.0")


class StartRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The user's initial question")


class QueryState(BaseModel):
    id: str
    original_question: str
    clarifying_questions: List[str] = []
    suggested_fields: List[str] = []
    answers: Dict[str, str] = {}
    created_at: datetime
    updated_at: datetime
    status: str = "awaiting_answers"  # awaiting_answers | ready


# In-memory store for demo purposes
QUERIES: Dict[str, QueryState] = {}


@app.get("/")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "mini-perplexity"}


@app.post("/query/start", response_model=QueryState)
def start_query(req: StartRequest) -> QueryState:
    now = datetime.now(timezone.utc)
    qid = str(uuid4())
    clar = generate_clarifying_questions(req.question)

    state = QueryState(
        id=qid,
        original_question=req.question,
        clarifying_questions=clar.get("questions", []),
        suggested_fields=clar.get("fields", []),
        answers={},
        created_at=now,
        updated_at=now,
    )
    QUERIES[qid] = state
    return state


class AnswerRequest(BaseModel):
    answers: Dict[str, str] = Field(
        default_factory=dict,
        description="Key-value answers to the clarifying questions. Keys can be indices (as strings) or field names.",
    )


@app.post("/query/{qid}/answer", response_model=QueryState)
def submit_answers(qid: str, req: AnswerRequest) -> QueryState:
    state = QUERIES.get(qid)
    if not state:
        raise HTTPException(status_code=404, detail="Query not found")

    # Merge answers; permit both field names and indices as keys
    normalized: Dict[str, str] = dict(state.answers)
    for key, val in (req.answers or {}).items():
        if key.isdigit():
            idx = int(key)
            if 0 <= idx < len(state.clarifying_questions):
                normalized[state.clarifying_questions[idx]] = val
        else:
            normalized[key] = val

    state.answers = normalized
    state.updated_at = datetime.now(timezone.utc)
    # Mark ready if all clarifying questions are answered
    if state.clarifying_questions and all(
        q in state.answers and str(state.answers[q]).strip() for q in state.clarifying_questions
    ):
        state.status = "ready"

    QUERIES[qid] = state
    return state


@app.get("/query/{qid}", response_model=QueryState)
def get_query(qid: str) -> QueryState:
    state = QUERIES.get(qid)
    if not state:
        raise HTTPException(status_code=404, detail="Query not found")
    return state

