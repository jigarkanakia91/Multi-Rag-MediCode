ENCODING_SYSTEM_PROMPT = """You are a certified medical coding assistant.
Given a clinical text span, candidate codes, similar past cases, and known
error patterns, select the correct code and return STRICT JSON only.

Rules:
- Only choose codes from the candidate list unless none are suitable (then set code to "UNCODED").
- If retrieved error patterns show a mistake matching this context, avoid repeating it and explain in warnings.
- Never invent a code that is not in the candidate list.
- Confidence must reflect genuine certainty (0.0-1.0), not always high.

Output schema (JSON object only, no markdown):
{
  "code": "...",
  "code_system": "...",
  "description": "...",
  "confidence": 0.0,
  "warnings": ["..."]
}
"""


def build_user_prompt(entity_text, candidates, similar_cases, error_patterns) -> str:
    cand_block = "\n".join(
        f"- {c.code} ({c.code_system}): {c.description} [score={c.score:.2f}]"
        for c in candidates
    ) or "None."

    error_block = "\n".join(
        f"- Past mistake: used {e.original_code}, correct was {e.corrected_code} "
        f"because {e.reason}"
        for e in error_patterns
    ) or "None found."

    similar_block = "\n".join(
        f"- {c.get('chunk_text', '')[:200]}" for c in similar_cases
    ) or "None found."

    return f"""
Clinical text span:
\"\"\"{entity_text}\"\"\"

Candidate codes:
{cand_block}

Similar past cases:
{similar_block}

Known error patterns relevant here:
{error_block}

Return the JSON object now.
"""
