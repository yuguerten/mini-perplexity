from __future__ import annotations

from typing import Dict, List

MODEL = "gpt-oss:20b"


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

