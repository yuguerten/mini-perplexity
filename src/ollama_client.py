from __future__ import annotations

from typing import Dict, List

MODEL = "lfm2.5-thinking:latest"


def _fallback_clarifying_questions(question: str) -> Dict[str, List[str]]:
  """Fallback research plan when LLM is unavailable"""
  q_lower = question.lower()
  
  fields: List[str] = ["timeframe", "scope", "sources", "focus_area"]
  
  qs: List[str] = []
  
  if "news" in q_lower or "recent" in q_lower:
    qs.extend([
      "What specific time period should we focus on?",
      "Which types of news sources are most relevant?",
      "Are there particular topics or domains to prioritize?"
    ])
  elif "research" in q_lower or "analysis" in q_lower:
    qs.extend([
      "What is the primary research objective?",
      "What type of sources would be most authoritative?",
      "What level of detail is required?"
    ])
  else:
    qs.extend([
      "What timeframe should we consider?",
      "What sources or domains are most relevant?", 
      "Are there specific aspects to focus on?"
    ])
  
  if any(loc in q_lower for loc in ["country", "city", "region", "morocco", "france", "usa"]):
    qs.append("Should we focus on specific geographic regions or locations?")
  
  return {"fields": fields, "questions": qs[:5]}


def generate_next_question(original_question: str, previous_qa: list[tuple[str, str]], question_num: int) -> str:
  """
  Generate the next clarifying question based on the original question and previous Q&As.
  Returns empty string if no more questions needed or on error.
  """
  if question_num > 3:
    return ""
    
  try:
    import ollama

    context = f"Original question: {original_question}\n"
    if previous_qa:
      context += "Previous clarifications:\n"
      for q, a in previous_qa:
        context += f"Q: {q}\nA: {a}\n"
    
    system = (
      "You are a research assistant. Generate ONE specific clarifying question "
      "to better understand the user's research needs. The question should be "
      "actionable and help refine the search scope. Respond with ONLY the question text, "
      "no additional formatting or explanations."
    )
    
    user = (
      f"{context}\n"
      f"Generate clarifying question #{question_num} (max 3 total) to better understand "
      "what the user is looking for. Make it specific and actionable."
    )
    
    resp = ollama.chat(
      model=MODEL,
      messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
      ],
    )
    
    question = resp.get("message", {}).get("content", "").strip()
    # Clean up any formatting
    question = question.replace('"', '').replace("Q:", "").replace("Question:", "").strip()
    
    return question if question and len(question) > 10 else ""
    
  except Exception as e:
    print(f"LLM generation failed: {e}")
    
    # Fallback questions based on question number
    fallbacks = [
      "What specific time period should we focus on?",
      "What types of sources or domains are most relevant?", 
      "Are there particular aspects or regions to emphasize?"
    ]
    return fallbacks[question_num - 1] if question_num <= len(fallbacks) else ""


def generate_clarifying_questions(question: str) -> Dict[str, List[str]]:
  """
  Use LLM to analyze the user question and generate a research plan with sub-questions.
  If LLM unavailable, return a deterministic fallback.
  """
  try:
    import json
    import ollama

    system = (
      "You are a research assistant that creates detailed research plans. "
      "Given a user question, generate 3-5 specific sub-questions that would help "
      "gather comprehensive information to answer the original question. "
      "Also suggest relevant structured fields for organizing the research. "
      "Reply ONLY in valid JSON format with simple string arrays: "
      '{\"fields\":[\"field1\",\"field2\"],\"questions\":[\"question1\",\"question2\"]}'
    )
    
    user = (
      f"Original question: {question}\n\n"
      "Create a research plan by generating 3-5 specific sub-questions that would help "
      "answer this question comprehensively. Make the sub-questions actionable "
      "and focused on gathering concrete information. "
      "Return as clean JSON with simple string arrays only."
    )
    
    resp = ollama.chat(
      model=MODEL,
      messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
      ],
    )
    content = resp.get("message", {}).get("content", "{}")
    
    # Debug: print raw response for troubleshooting
    # print(f"DEBUG - LLM response: {content}")
    
    # Clean up response - remove any markdown formatting
    content = content.strip()
    if content.startswith("```json"):
      content = content[7:]
    if content.endswith("```"):
      content = content[:-3]
    content = content.strip()
    
    # Try to extract JSON from the response
    try:
      data = json.loads(content)
    except json.JSONDecodeError:
      # If not pure JSON, try to find JSON block in the response
      import re
      json_match = re.search(r'\{[^{}]*"fields"[^{}]*"questions"[^{}]*\}', content)
      if json_match:
        data = json.loads(json_match.group())
      else:
        raise ValueError("No valid JSON found")
    
    if isinstance(data, dict) and "questions" in data:
      # Ensure shape and limit
      fields = [str(x).strip() for x in data.get("fields", []) if str(x).strip()][:8]
      questions = [str(x).strip() for x in data.get("questions", []) if str(x).strip()][:6]
      if questions:  # Must have at least some questions
        return {"fields": fields, "questions": questions}
        
  except Exception as e:
    # Any error falls back to heuristics
    print(f"LLM generation failed, using fallback: {e}")

  return _fallback_clarifying_questions(question)


def generate_search_query_from_answer(original_question: str, question: str, answer: str) -> str:
  """
  Generate a search query based on the original question and a clarifying question-answer pair.
  
  Args:
      original_question: The user's original research question
      question: The clarifying question that was asked
      answer: The user's answer to the clarifying question
      
  Returns:
      A search query that incorporates the clarification
  """
  try:
    import ollama
    
    system = (
      "You are a search query optimizer. Your task is to create an optimized search query "
      "that combines the original research question with additional context from a "
      "clarifying question and its answer. The search query should be concise, focused, "
      "and use keywords that would lead to the most relevant search results."
    )
    
    user = (
      f"Original research question: {original_question}\n\n"
      f"Clarifying question: {question}\n"
      f"User's answer: {answer}\n\n"
      "Based on this information, generate a concise and specific search query that "
      "would return the most relevant results. Focus on using keywords that would "
      "be effective for a search engine. Return only the optimized search query text, "
      "without quotes or explanations."
    )
    
    resp = ollama.chat(
      model=MODEL,
      messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
      ],
    )
    
    query = resp.get("message", {}).get("content", "").strip()
    
    # Clean up any formatting or quotes
    query = query.strip('"\'').strip()
    
    # If the generated query is too short or empty, fall back to original question
    if not query or len(query) < 10:
      return original_question
    
    return query
    
  except Exception as e:
    print(f"Search query generation failed: {e}")
    # Fall back to combining the original question with the answer
    return f"{original_question} {answer}"


def generate_combined_search_query(original_question: str, qa_pairs: list[tuple[str, str]]) -> str:
  """
  Generate a combined search query from the original question and all Q&A pairs.
  
  Args:
      original_question: The user's original research question
      qa_pairs: List of (question, answer) tuples from clarification
      
  Returns:
      A comprehensive search query
  """
  try:
    import ollama
    
    if not qa_pairs:
      return original_question
    
    context = f"Original research question: {original_question}\n\nClarifications:\n"
    for q, a in qa_pairs:
      context += f"Q: {q}\nA: {a}\n\n"
    
    system = (
      "You are a search query optimizer that creates comprehensive search queries. "
      "Given an original research question and clarifications from the user, "
      "create ONE optimized search query that integrates all the information. "
      "The query should be concise but comprehensive, focusing on the most important keywords "
      "that would yield relevant search results."
    )
    
    user = (
      f"{context}\n"
      "Based on all this information, create ONE optimized search query that would return "
      "the most relevant results. Focus on using keywords that would be effective for a search engine. "
      "Return only the optimized search query text, without quotes or explanations."
    )
    
    resp = ollama.chat(
      model=MODEL,
      messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
      ],
    )
    
    query = resp.get("message", {}).get("content", "").strip()
    
    # Clean up any formatting or quotes
    query = query.strip('"\'').strip()
    
    # If the generated query is too short or empty, fall back to original question
    if not query or len(query) < 10:
      return original_question
    
    return query
    
  except Exception as e:
    print(f"Combined search query generation failed: {e}")
    # Fall back to original question
    return original_question
